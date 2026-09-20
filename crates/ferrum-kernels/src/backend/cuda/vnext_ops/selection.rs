//! Precision of semantic logits and their private repetition-penalty view.

use super::*;
use cudarc::driver::{CudaContext, CudaStream};
use ferrum_interfaces::vnext::{
    last_token_masked_argmax_f32_contract, StandardOperationContract,
    LAST_TOKEN_MASKED_ARGMAX_F32_CAPABILITY_ID, LAST_TOKEN_MASKED_ARGMAX_F32_OPERATION_ID,
};

#[derive(Clone, Copy)]
pub(super) enum ArgmaxPrecision {
    F16,
    F32,
}

pub(super) const ARGMAX_PARTITIONS: u32 = 32;
// Two launches are worthwhile for large vocabularies; retain the scalar path
// for small rows, where launch overhead can dominate an active penalty.
const PARALLEL_MIN_VOCAB: i32 = 65536;

pub(super) fn argmax_dispatches(vocabulary_size: i32) -> u64 {
    if vocabulary_size >= PARALLEL_MIN_VOCAB {
        2
    } else {
        1
    }
}

#[derive(Clone)]
pub(super) struct ArgmaxFunctions {
    scalar: CudaFunction,
    partitioned: CudaFunction,
    finalize: CudaFunction,
}

#[derive(Clone, Copy)]
pub(super) struct ArgmaxArguments {
    pub logits: u64,
    pub scratch: u64,
    pub valid_mask: u64,
    pub repetition_offsets: u64,
    pub repetition_token_ids: u64,
    pub repetition_penalty: u64,
    pub output: u64,
    pub vocabulary_size: i32,
    pub repetition_capacity: i32,
}

impl ArgmaxFunctions {
    pub(super) fn load(
        context: &Arc<CudaContext>,
        precision: ArgmaxPrecision,
    ) -> Result<Self, CudaDeviceRuntimeError> {
        let module = context
            .load_module(Ptx::from_src(crate::ptx::ARGMAX_ROWS))
            .map_err(|error| CudaDeviceRuntimeError::driver("masked argmax module load", error))?;
        let load = |name: &str| {
            module.load_function(name).map_err(|error| {
                CudaDeviceRuntimeError::driver("masked argmax function load", error)
            })
        };
        Ok(Self {
            scalar: load(precision.kernel())?,
            partitioned: load(&format!("{}_partitioned", precision.kernel()))?,
            finalize: load("last_token_masked_argmax_finalize")?,
        })
    }

    pub(super) fn launch(
        &self,
        stream: &CudaStream,
        args: ArgmaxArguments,
        parallel: bool,
    ) -> Result<(), CudaDeviceRuntimeError> {
        let function = if parallel {
            &self.partitioned
        } else {
            &self.scalar
        };
        let mut builder = stream.launch_builder(function);
        builder
            .arg(&args.logits)
            .arg(&args.scratch)
            .arg(&args.vocabulary_size)
            .arg(&args.valid_mask)
            .arg(&args.vocabulary_size)
            .arg(&args.repetition_offsets)
            .arg(&args.repetition_token_ids)
            .arg(&args.repetition_penalty)
            .arg(&args.repetition_capacity)
            .arg(&args.output);
        // The caller retains the invocation regions and their scratch lease.
        // Only the large-vocabulary route uses the first 32 eight-byte partials.
        unsafe {
            builder.launch(LaunchConfig {
                grid_dim: (if parallel { ARGMAX_PARTITIONS } else { 1 }, 1, 1),
                block_dim: (THREADS_PER_BLOCK, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .map_err(|error| CudaDeviceRuntimeError::driver("vNext masked argmax launch", error))?;
        if parallel {
            let mut finalize = stream.launch_builder(&self.finalize);
            finalize
                .arg(&args.scratch)
                .arg(&args.repetition_offsets)
                .arg(&args.repetition_penalty)
                .arg(&args.repetition_capacity)
                .arg(&args.output);
            // Same CUDA stream orders every partial write before finalization.
            // Active penalties retain the scalar result; finalization is a no-op.
            unsafe {
                finalize.launch(LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (32, 1, 1),
                    shared_mem_bytes: 0,
                })
            }
            .map_err(|error| {
                CudaDeviceRuntimeError::driver("vNext masked argmax finalize", error)
            })?;
        }
        Ok(())
    }
}

impl ArgmaxPrecision {
    pub(super) fn contract(self) -> Result<StandardOperationContract, VNextError> {
        match self {
            Self::F16 => last_token_masked_argmax_contract(),
            Self::F32 => last_token_masked_argmax_f32_contract(),
        }
    }

    pub(super) fn operation(self) -> &'static str {
        match self {
            Self::F16 => LAST_TOKEN_MASKED_ARGMAX_OPERATION_ID,
            Self::F32 => LAST_TOKEN_MASKED_ARGMAX_F32_OPERATION_ID,
        }
    }

    pub(super) fn capability(self) -> &'static str {
        match self {
            Self::F16 => LAST_TOKEN_MASKED_ARGMAX_F16_CAPABILITY_ID,
            Self::F32 => LAST_TOKEN_MASKED_ARGMAX_F32_CAPABILITY_ID,
        }
    }

    pub(super) fn provider(self) -> &'static str {
        match self {
            Self::F16 => LAST_TOKEN_MASKED_ARGMAX_PROVIDER_ID,
            Self::F32 => "provider.cuda.last_token_masked_argmax.f32",
        }
    }

    pub(super) fn estimator(self) -> &'static str {
        match self {
            Self::F16 => LAST_TOKEN_MASKED_ARGMAX_ESTIMATOR_ID,
            Self::F32 => "resource-estimator.cuda.last_token_masked_argmax.f32",
        }
    }

    pub(super) fn kernel(self) -> &'static str {
        match self {
            Self::F16 => MASKED_ARGMAX_PRESERVING_LOGITS_FUNCTION_NAME,
            Self::F32 => "last_token_masked_argmax_preserving_logits_f32",
        }
    }

    pub(super) fn element(self) -> ElementType {
        match self {
            Self::F16 => ElementType::F16,
            Self::F32 => ElementType::F32,
        }
    }
}

#[cfg(test)]
mod tests;
