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

#[derive(Clone)]
struct Projection {
    scalar: CudaFunction,
    tiled: CudaFunction,
    mma: CudaFunction,
}

#[derive(Clone)]
pub(in crate::backend::cuda::vnext_ops) struct Q8F32ScaleKernels {
    pack: CudaFunction,
    q4: Projection,
    q5: Projection,
    q6: Projection,
}

/// Scales precede packed signed bytes. Complete K256 blocks make both spans
/// naturally 16-byte aligned, including when several rows share this storage.
#[derive(Clone, Copy, Debug)]
pub(in crate::backend::cuda::vnext_ops) struct PackLayout {
    pub scales_bytes: u64,
    pub total_bytes: u64,
}

impl PackLayout {
    pub fn new(rows: u64, columns: u64) -> Result<Self, String> {
        if rows == 0 || columns == 0 || !columns.is_multiple_of(256) {
            return Err("Q8 activation packing requires rows and complete K256 blocks".into());
        }
        let values = rows
            .checked_mul(columns)
            .ok_or("Q8 activation extent overflows")?;
        let scales_bytes = values / 8;
        let total_bytes = values
            .checked_add(scales_bytes)
            .ok_or("Q8 workspace overflows")?;
        Ok(Self {
            scales_bytes,
            total_bytes,
        })
    }
}

pub(in crate::backend::cuda::vnext_ops) fn quantizes(part: &weights::MatrixPart) -> bool {
    matches!(
        part.format,
        weights::MatrixFormat::Block(
            GgufBlockFormat::Q4K | GgufBlockFormat::Q5K | GgufBlockFormat::Q6K
        )
    )
}

impl Q8F32ScaleKernels {
    pub fn supported(context: &CudaContext) -> Result<bool, CudaDeviceRuntimeError> {
        context
            .attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
            .map(|major| major >= 8)
            .map_err(|e| CudaDeviceRuntimeError::driver("Q8 MMA compute capability", e))
    }

    pub fn load(context: &Arc<CudaContext>) -> Result<Self, CudaDeviceRuntimeError> {
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
        let projection = |format: &str| -> Result<Projection, CudaDeviceRuntimeError> {
            Ok(Projection {
                scalar: load(&format!(
                    "vnext_gguf_{format}_q8_f32scale_dp4a_lane_f16_prototype"
                ))?,
                tiled: load(&format!(
                    "vnext_gguf_{format}_q8_f32scale_dp4a_lane_tiled_f16_prototype"
                ))?,
                mma: load(&format!(
                    "vnext_gguf_{format}_q8_f32scale_mma_f16_prototype"
                ))?,
            })
        };
        Ok(Self {
            pack: load("vnext_gguf_q8_f32scale_pack_f16_prototype")?,
            q4: projection("q4k")?,
            q5: projection("q5k")?,
            q6: projection("q6k")?,
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
        for part in parts {
            if part.transform.is_some()
                || part.signs_region.is_some()
                || part.columns != columns
                || part.rows == 0
                || part
                    .output_offset
                    .checked_add(part.rows)
                    .is_none_or(|end| end > output_stride)
            {
                return Err(CudaDeviceRuntimeError::contract(
                    "Q8 projection requires untransformed, matching matrix parts",
                ));
            }
        }
        let packed = if parts.iter().any(quantizes) {
            let layout = PackLayout::new(u64::from(rows), u64::from(columns))
                .map_err(CudaDeviceRuntimeError::contract)?;
            if workspace == 0 {
                return Err(CudaDeviceRuntimeError::contract(
                    "Q8 projection requires planned workspace",
                ));
            }
            let words = workspace.checked_add(layout.scales_bytes).ok_or_else(|| {
                CudaDeviceRuntimeError::contract("Q8 workspace pointer overflows")
            })?;
            let groups = u64::from(rows) * u64::from(columns / 32);
            let blocks = u32::try_from(groups.div_ceil(4))
                .map_err(|_| CudaDeviceRuntimeError::contract("Q8 pack grid overflows"))?;
            let mut launch = stream.launch_builder(&self.pack);
            launch
                .arg(&input)
                .arg(&workspace)
                .arg(&words)
                .arg(&rows)
                .arg(&columns);
            // SAFETY: one warp per K32 group; bounds and planned spans checked above.
            unsafe {
                launch.launch(LaunchConfig {
                    grid_dim: (blocks, 1, 1),
                    block_dim: (128, 1, 1),
                    shared_mem_bytes: 0,
                })
            }
            .map_err(|e| CudaDeviceRuntimeError::driver("Q8 activation pack", e))?;
            Some(words)
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
                let words = packed.ok_or_else(|| {
                    CudaDeviceRuntimeError::contract("Q8 projection lacks packed input")
                })?;
                // The 8/32-row paired measurements include activation packing.
                // MMA amortizes staging from eight rows; narrower launches use
                // the lane mapping under the same numerical policy.
                let (function, row_tile, column_tile) = if rows >= 8 {
                    (&projection.mma, 32, 16)
                } else if rows > 1 {
                    (&projection.tiled, 8, 4)
                } else {
                    (&projection.scalar, 1, 4)
                };
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
                // SAFETY: packed pointers and each matrix/output interval were
                // validated; tail rows/columns participate and suppress stores.
                unsafe {
                    launch.launch(LaunchConfig {
                        grid_dim: (part.rows.div_ceil(column_tile), rows.div_ceil(row_tile), 1),
                        block_dim: (128, 1, 1),
                        shared_mem_bytes: 0,
                    })
                }
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
}
