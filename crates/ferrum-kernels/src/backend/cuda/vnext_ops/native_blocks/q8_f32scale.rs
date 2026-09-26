//! Explicit Q8 activation policy. Workspace belongs to the invocation; packing
//! and projection are replayable launches and never allocate device memory.

use super::{weights, CudaNativeBlockKernels};
use crate::backend::cuda::vnext_runtime::CudaDeviceRuntimeError;
use crate::gguf_blocks::GgufBlockFormat;
use cudarc::driver::{
    sys::CUdevice_attribute, CudaContext, CudaFunction, CudaStream, LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::Ptx;
use ferrum_interfaces::vnext::ElementType;
use std::sync::Arc;

mod launch_plan;
pub(in crate::backend::cuda::vnext_ops) mod selected;

pub(in crate::backend::cuda::vnext_ops) use crate::gguf_blocks::q8_projection_plan::{
    MatrixPlan, PackLayout, Q8SumPolicy, QuantizedKernel,
};

impl Q8SumPolicy {
    pub fn operation_id(self) -> &'static str {
        match self {
            Self::Quantized => ferrum_interfaces::vnext::DENSE_SWIGLU_Q8_F32SCALE_OPERATION_ID,
            Self::Input => {
                ferrum_interfaces::vnext::DENSE_SWIGLU_Q8_F32SCALE_INPUT_SUM_OPERATION_ID
            }
        }
    }
}

#[derive(Clone)]
struct Projection {
    scalar: CudaFunction,
    tiled: CudaFunction,
    mma: CudaFunction,
}

#[derive(Clone)]
pub(in crate::backend::cuda::vnext_ops) struct Q8F32ScaleKernels {
    policy: Q8SumPolicy,
    pack: CudaFunction,
    q4: Projection,
    q5: Projection,
    q6: Projection,
}

/// One checked numeric description for execution, scratch and future topology.
/// Physical matrix identities remain with their existing retained bindings.
pub(in crate::backend::cuda::vnext_ops) fn matrix_plan(
    parts: &[weights::MatrixPart],
    whole_rows: u64,
    columns: u32,
    output_stride: u32,
    policy: Q8SumPolicy,
) -> Result<MatrixPlan, String> {
    MatrixPlan::new(
        whole_rows,
        columns,
        output_stride,
        policy,
        parts
            .iter()
            .map(|part| crate::gguf_blocks::q8_projection_plan::MatrixPart {
                format: match part.format {
                    weights::MatrixFormat::DenseF16 => None,
                    weights::MatrixFormat::Block(format) => Some(format),
                },
                outputs: part.rows,
                columns: part.columns,
                output_offset: part.output_offset,
                transformed: part.transform.is_some() || part.signs_region.is_some(),
            }),
    )
}

pub(in crate::backend::cuda::vnext_ops) fn matrix_plan_from_parts(
    parts: &[weights::MatrixPart],
    whole_rows: u64,
    policy: Q8SumPolicy,
) -> Result<MatrixPlan, String> {
    let columns = parts.first().ok_or("Q8 matrix plan has no parts")?.columns;
    let stride = parts.iter().try_fold(0_u32, |end, part| {
        part.output_offset
            .checked_add(part.rows)
            .map(|value| end.max(value))
            .ok_or("Q8 matrix output extent overflows")
    })?;
    matrix_plan(parts, whole_rows, columns, stride, policy)
}

impl Q8F32ScaleKernels {
    pub fn supported(context: &CudaContext) -> Result<bool, CudaDeviceRuntimeError> {
        context
            .attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
            .map(|major| major >= 8)
            .map_err(|e| CudaDeviceRuntimeError::driver("Q8 MMA compute capability", e))
    }

    pub fn load(context: &Arc<CudaContext>) -> Result<Self, CudaDeviceRuntimeError> {
        Self::load_with_policy(context, Q8SumPolicy::Quantized)
    }

    pub fn policy(&self) -> Q8SumPolicy {
        self.policy
    }

    pub fn load_with_policy(
        context: &Arc<CudaContext>,
        policy: Q8SumPolicy,
    ) -> Result<Self, CudaDeviceRuntimeError> {
        if !Self::supported(context)? {
            return Err(CudaDeviceRuntimeError::contract(
                "Q8 MMA requires SM80 or newer",
            ));
        }
        let module = context
            .load_module(Ptx::from_src(crate::ptx::VNEXT_GGUF.to_owned()))
            .map_err(|e| CudaDeviceRuntimeError::driver("Q8 projection module load", e))?;
        let load = |name: &str| {
            module
                .load_function(name)
                .map_err(|e| CudaDeviceRuntimeError::driver("Q8 projection function load", e))
        };
        let projection = |format| -> Result<Projection, CudaDeviceRuntimeError> {
            let [scalar, tiled, mma] = launch_plan::entries(format, policy).ok_or_else(|| {
                CudaDeviceRuntimeError::contract("unsupported Q8 projection format")
            })?;
            Ok(Projection {
                scalar: load(scalar)?,
                tiled: load(tiled)?,
                mma: load(mma)?,
            })
        };
        Ok(Self {
            policy,
            pack: load(launch_plan::pack_entry(policy))?,
            q4: projection(GgufBlockFormat::Q4K)?,
            q5: projection(GgufBlockFormat::Q5K)?,
            q6: projection(GgufBlockFormat::Q6K)?,
        })
    }

    /// The caller proves complete input/output spans and retains all pointers.
    /// The same packed activation is shared across gate and up matrix parts.
    #[allow(clippy::too_many_arguments)]
    pub fn launch(
        &self,
        strict: &CudaNativeBlockKernels,
        stream: &CudaStream,
        parts: &[weights::MatrixPart],
        weights: &[u64],
        input: u64,
        output: u64,
        rows: u32,
        columns: u32,
        output_stride: u32,
        workspace: u64,
    ) -> Result<(), CudaDeviceRuntimeError> {
        if parts.is_empty() || weights.len() != parts.len() || rows == 0 || rows > u16::MAX as u32 {
            return Err(CudaDeviceRuntimeError::contract(
                "invalid Q8 projection inventory or rows",
            ));
        }
        let plan = matrix_plan(parts, u64::from(rows), columns, output_stride, self.policy)
            .map_err(CudaDeviceRuntimeError::contract)?;
        let leaf = plan
            .leaf(u64::from(rows))
            .map_err(CudaDeviceRuntimeError::contract)?;
        let packed = if let Some(layout) = leaf.pack {
            if workspace == 0 {
                return Err(CudaDeviceRuntimeError::contract(
                    "Q8 projection requires planned workspace",
                ));
            }
            let words = workspace.checked_add(layout.words_offset).ok_or_else(|| {
                CudaDeviceRuntimeError::contract("Q8 workspace pointer overflows")
            })?;
            let config = launch_plan::pack_config(rows, columns)
                .map_err(CudaDeviceRuntimeError::contract)?;
            let mut launch = stream.launch_builder(&self.pack);
            launch
                .arg(&input)
                .arg(&workspace)
                .arg(&words)
                .arg(&rows)
                .arg(&columns);
            let sums = workspace + layout.scales_bytes;
            if self.policy == Q8SumPolicy::Input {
                launch.arg(&sums);
            }
            // SAFETY: one warp per K32 group; bounds and planned spans checked above.
            unsafe { launch.launch(config) }
                .map_err(|e| CudaDeviceRuntimeError::driver("Q8 activation pack", e))?;
            Some((words, sums))
        } else {
            None
        };
        for (part, &weight) in parts.iter().zip(weights) {
            let projection = match part.format {
                weights::MatrixFormat::Block(GgufBlockFormat::Q4K) => Some(&self.q4),
                weights::MatrixFormat::Block(GgufBlockFormat::Q5K) => Some(&self.q5),
                weights::MatrixFormat::Block(GgufBlockFormat::Q6K) => Some(&self.q6),
                _ => None,
            };
            if let Some(projection) = projection {
                let (words, sums) = packed.ok_or_else(|| {
                    CudaDeviceRuntimeError::contract("Q8 projection lacks packed input")
                })?;
                // The 8/32-row paired measurements include activation packing.
                // MMA amortizes staging from eight rows; narrower launches use
                // the lane mapping under the same numerical policy.
                let kernel = leaf.quantized_kernel.ok_or_else(|| {
                    CudaDeviceRuntimeError::contract("Q8 matrix plan lacks its quantized kernel")
                })?;
                let function = match kernel {
                    QuantizedKernel::Scalar => &projection.scalar,
                    QuantizedKernel::Tiled => &projection.tiled,
                    QuantizedKernel::Mma => &projection.mma,
                };
                let config = launch_plan::project_config(kernel, rows, part.rows);
                let mut launch = stream.launch_builder(function);
                launch
                    .arg(&workspace)
                    .arg(&words)
                    .arg(&weight)
                    .arg(&output)
                    .arg(&rows)
                    .arg(&columns)
                    .arg(&part.rows)
                    .arg(&output_stride)
                    .arg(&part.output_offset);
                if self.policy == Q8SumPolicy::Input
                    && part.format != weights::MatrixFormat::Block(GgufBlockFormat::Q6K)
                {
                    launch.arg(&sums);
                }
                // SAFETY: packed pointers and each matrix/output interval were
                // validated; tail rows/columns participate and suppress stores.
                unsafe { launch.launch(config) }
                    .map_err(|e| CudaDeviceRuntimeError::driver("Q8 native projection", e))?;
            } else {
                strict.transformed_linear(
                    stream,
                    input,
                    weight,
                    output,
                    part,
                    rows,
                    output_stride,
                    ElementType::F16,
                    0,
                    0,
                )?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pack_layout_covers_scales_and_quants_without_aliasing() {
        for (rows, columns) in [(1, 256), (3, 512), (32, 12288)] {
            let layout = PackLayout::new(rows, columns).unwrap();
            assert_eq!(layout.scales_bytes, rows * (columns / 32) * 4);
            assert_eq!(layout.total_bytes - layout.scales_bytes, rows * columns);
            assert_eq!(layout.scales_bytes % 16, 0);
        }
        for (rows, columns) in [
            (0, 256),
            (1, 0),
            (1, 255),
            (u64::MAX, 256),
            (1, u64::MAX - 255),
        ] {
            assert!(PackLayout::new(rows, columns).is_err());
        }
    }

    #[test]
    fn input_sum_pack_layout_keeps_three_disjoint_planned_spans() {
        for (rows, columns) in [(1, 256), (4, 4096), (8, 12288)] {
            let old = PackLayout::new(rows, columns).unwrap();
            let new = PackLayout::with_policy(rows, columns, Q8SumPolicy::Input).unwrap();
            assert_eq!(old.sums_bytes, 0);
            assert_eq!(new.scales_bytes, old.scales_bytes);
            assert_eq!(new.sums_bytes, rows * (columns / 32) * 4);
            assert_eq!(new.words_offset, new.scales_bytes + new.sums_bytes);
            assert_eq!(new.total_bytes - new.words_offset, rows * columns);
            assert_eq!(new.total_bytes - old.total_bytes, new.sums_bytes);
            assert_eq!(new.words_offset % 16, 0);
        }
        assert!(PackLayout::with_policy(1, 255, Q8SumPolicy::Input).is_err());
        assert!(PackLayout::with_policy(u64::MAX / 256, 256, Q8SumPolicy::Input).is_err());
    }
}

#[cfg(test)]
mod input_sum_tests;
