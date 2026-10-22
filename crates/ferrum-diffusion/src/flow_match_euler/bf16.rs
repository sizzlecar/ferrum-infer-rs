//! Managed BF16 deterministic Euler, with an explicit scalar-promotion profile.
//!
//! Source interpretation, not a Torch execution or bitwise qualification:
//! - [Diffusers 0.40.0](https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/schedulers/scheduling_flow_match_euler_discrete.py):
//!   `set_timesteps` puts FP32 sigmas on its `device` argument, replacing the
//!   constructor's CPU sigmas. `step` upcasts sample, but not model_output,
//!   computes sample + dt * model_output, then casts to model_output.dtype.
//! - [Pinned Omni](https://github.com/vllm-project/vllm-omni/blob/c0551d22674d0db81305c800b66ad83035fdc980/vllm_omni/diffusion/models/qwen_image_21/pipeline_qwen_image_21.py):
//!   `prepare_timesteps` supplies the pipeline device; its imported
//!   `retrieve_timesteps` forwards that argument unchanged.
//! - Torch 2.13.0 [TypeProperties](https://github.com/pytorch/pytorch/blob/v2.13.0/aten/src/ATen/native/TypeProperties.cpp)
//!   selects the dimensioned BF16 operand over a zero-dimensional FP32 operand.
//!   [BinaryMulKernel](https://github.com/pytorch/pytorch/blob/v2.13.0/aten/src/ATen/native/cuda/BinaryMulKernel.cu)
//!   and [Loops](https://github.com/pytorch/pytorch/blob/v2.13.0/aten/src/ATen/native/cuda/Loops.cuh)
//!   use BF16 arguments/return with FP32 opmath. The non-CPU scalar goes through
//!   [LoadWithCast](https://github.com/pytorch/pytorch/blob/v2.13.0/aten/src/ATen/native/cuda/MemoryAccess.cuh),
//!   rounding dt to BF16 before multiplication. The CPU-scalar special case
//!   preserves FP32 dt and is a different, unsupported profile here.
//!
//! This Host reference specifies round-to-nearest-even with gradual underflow.
//! It rejects nonfinite inputs/results and does not claim target-device FTZ,
//! compiler behavior, stochastic sampling, CFG, or per-token timestep parity.

use ferrum_runtime::{
    ElementType, HostMemoryAccount, HostMemoryClass, HostMemoryError, ManagedHostBytes,
    ManagedHostTensor, ResolvedTensorLayout, ResolvedTensorSpec, VNextError,
};
use half::bf16;
use thiserror::Error;

use super::{EulerStep, FlowMatchEuler, FlowMatchEulerSchedule};

/// No implicit default: the adapter must select the source arithmetic path.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Bf16EulerProfile {
    /// d=B(RN32(next_sigma-sigma)); p=B(RN32(d*v)); y=B(RN32(x+p)).
    /// B is BF16 round-to-nearest-even. No multiply-add fusion is permitted.
    CudaScalarBf16StagedRneV1,
}

impl Bf16EulerProfile {
    pub const fn id(self) -> &'static str {
        match self {
            Self::CudaScalarBf16StagedRneV1 => "flow_match_euler.bf16.cuda_scalar_staged.rne.v1",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Bf16EulerOperand {
    Sample,
    Velocity,
}

#[derive(Debug, Error)]
pub enum Bf16EulerError {
    #[error("BF16 Euler requires a nonempty contiguous BF16 tensor within the Host address range")]
    InvalidTensorSpec,
    #[error("the Euler schedule is already complete")]
    Complete,
    #[error("{operand:?} tensor does not match the configured shape, dtype and layout")]
    TensorMismatch { operand: Bf16EulerOperand },
    #[error("{operand:?} byte extent does not match the configured tensor")]
    ByteExtentMismatch { operand: Bf16EulerOperand },
    #[error("{operand:?} backing belongs to another Host account")]
    ForeignOwner { operand: Bf16EulerOperand },
    #[error("{operand:?} has unsupported Host classification {actual:?}")]
    WrongClass {
        operand: Bf16EulerOperand,
        actual: HostMemoryClass,
    },
    #[error("{operand:?} element {index} is nonfinite")]
    NonFiniteInput {
        operand: Bf16EulerOperand,
        index: usize,
    },
    #[error("BF16 Euler result element {index} is nonfinite")]
    NonFiniteResult { index: usize },
    #[error(transparent)]
    Host(#[from] HostMemoryError),
    #[error(transparent)]
    Tensor(#[from] VNextError),
}

/// Request-local solver over immutable managed inputs and one shared Host root.
///
/// An update keeps the old sample and velocity alive while reserving exactly
/// one new BF16 sample in InputPreprocess. Thus all three physical backings
/// contribute to the shared total/class limits. No latent-sized Vec, F32
/// materialization, quota reclassification, or device allocation is performed.
/// The returned sample retains its claim after this solver/account is dropped.
pub struct ManagedBf16Euler {
    solver: FlowMatchEuler,
    host: HostMemoryAccount,
    tensor: ResolvedTensorSpec,
    byte_len: usize,
    profile: Bf16EulerProfile,
}

impl ManagedBf16Euler {
    pub fn new(
        schedule: FlowMatchEulerSchedule,
        host: HostMemoryAccount,
        tensor: ResolvedTensorSpec,
        profile: Bf16EulerProfile,
    ) -> Result<Self, Bf16EulerError> {
        if tensor.element_type() != ElementType::Bf16
            || tensor.layout() != &ResolvedTensorLayout::Contiguous
            || tensor.dimensions().is_empty()
            || tensor.dimensions().contains(&0)
        {
            return Err(Bf16EulerError::InvalidTensorSpec);
        }
        let byte_len = tensor
            .minimum_storage_bytes()
            .ok()
            .and_then(|bytes| usize::try_from(bytes).ok())
            .filter(|&bytes| bytes > 0 && bytes <= isize::MAX as usize)
            .ok_or(Bf16EulerError::InvalidTensorSpec)?;
        Ok(Self {
            solver: schedule.start(),
            host,
            tensor,
            byte_len,
            profile,
        })
    }

    pub fn step_index(&self) -> usize {
        self.solver.step_index()
    }
    pub fn current_step(&self) -> Option<EulerStep> {
        self.solver.current_step()
    }
    pub fn is_complete(&self) -> bool {
        self.solver.is_complete()
    }
    pub fn tensor(&self) -> &ResolvedTensorSpec {
        &self.tensor
    }
    pub const fn profile(&self) -> Bf16EulerProfile {
        self.profile
    }

    /// Consume a completed velocity readback (Output class) and immutable
    /// sample (InputPreprocess class), both owned by this solver's Host root.
    /// The output retains the sample's value ID for rebinding as the next input.
    ///
    /// Any error leaves both inputs and the step index unchanged. A partially
    /// calculated output is private and drops its full claim on error; no caller
    /// can observe it. Progress advances only after the complete typed output
    /// has been constructed. CPU work is synchronous and retains no borrows.
    pub fn advance(
        &mut self,
        sample: &ManagedHostTensor,
        velocity_spec: &ResolvedTensorSpec,
        velocity: &ManagedHostBytes,
    ) -> Result<(EulerStep, ManagedHostTensor), Bf16EulerError> {
        let step = self.current_step().ok_or(Bf16EulerError::Complete)?;
        self.validate_input(
            Bf16EulerOperand::Sample,
            sample.tensor(),
            sample.bytes(),
            HostMemoryClass::InputPreprocess,
        )?;
        self.validate_input(
            Bf16EulerOperand::Velocity,
            velocity_spec,
            velocity,
            HostMemoryClass::Output,
        )?;
        let mut output = self
            .host
            .allocate_zeroed(HostMemoryClass::InputPreprocess, self.byte_len)?;
        let delta = match self.profile {
            Bf16EulerProfile::CudaScalarBf16StagedRneV1 => {
                bf16::from_f32(step.delta_sigma()).to_f32()
            }
        };
        for (index, ((x, v), destination)) in sample
            .bytes()
            .bytes()
            .chunks_exact(2)
            .zip(velocity.bytes().chunks_exact(2))
            .zip(output.as_mut().chunks_exact_mut(2))
            .enumerate()
        {
            let x = decode(x);
            let v = decode(v);
            if !x.is_finite() {
                return Err(Bf16EulerError::NonFiniteInput {
                    operand: Bf16EulerOperand::Sample,
                    index,
                });
            }
            if !v.is_finite() {
                return Err(Bf16EulerError::NonFiniteInput {
                    operand: Bf16EulerOperand::Velocity,
                    index,
                });
            }
            let increment = bf16::from_f32(delta * v).to_f32();
            let updated = bf16::from_f32(x + increment);
            if !updated.is_finite() {
                return Err(Bf16EulerError::NonFiniteResult { index });
            }
            destination.copy_from_slice(&updated.to_bits().to_le_bytes());
        }
        let output = ManagedHostTensor::new(
            sample.value_id().clone(),
            self.tensor.clone(),
            output.freeze(),
        )?;
        self.solver.step_index += 1;
        Ok((step, output))
    }

    fn validate_input(
        &self,
        operand: Bf16EulerOperand,
        spec: &ResolvedTensorSpec,
        bytes: &ManagedHostBytes,
        class: HostMemoryClass,
    ) -> Result<(), Bf16EulerError> {
        if spec != &self.tensor {
            return Err(Bf16EulerError::TensorMismatch { operand });
        }
        if bytes.len() != self.byte_len {
            return Err(Bf16EulerError::ByteExtentMismatch { operand });
        }
        if !self.host.owns(bytes) {
            return Err(Bf16EulerError::ForeignOwner { operand });
        }
        if bytes.class() != class {
            return Err(Bf16EulerError::WrongClass {
                operand,
                actual: bytes.class(),
            });
        }
        Ok(())
    }
}

fn decode(bytes: &[u8]) -> f32 {
    bf16::from_bits(u16::from_le_bytes([bytes[0], bytes[1]])).to_f32()
}

#[cfg(test)]
mod tests;
