//! Exact native linear selector used by launch and passive selected evidence.
use super::*;
use ferrum_interfaces::vnext::ElementType;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum NativeLinearKernel {
    GemmQ4kF16,
    GemmQ5kF16,
    GemmQ6kF16,
    LinearQ4kF16,
    LinearQ4kTiledF16,
    LinearQ5kF16,
    LinearQ5kTiledF16,
    LinearQ6kF16,
    LinearQ6kTiledF16,
    LinearQ6kF32,
    LinearQ6kTiledF32,
    LinearQ6kF32F16,
    LinearQ6kTiledF32F16,
    LinearF16,
    LinearF32,
    LinearTiledF16,
    LinearTiledF32,
    LinearF32F16,
    LinearTiledF32F16,
}

impl NativeLinearKernel {
    pub(super) fn entry(self) -> &'static str {
        match self {
            Self::GemmQ4kF16 => "vnext_gguf_gemm_q4k_f16",
            Self::GemmQ5kF16 => "vnext_gguf_gemm_q5k_f16",
            Self::GemmQ6kF16 => "vnext_gguf_gemm_q6k_f16",
            Self::LinearQ4kF16 => "vnext_gguf_linear_q4k_f16",
            Self::LinearQ4kTiledF16 => "vnext_gguf_linear_q4k_tiled_f16",
            Self::LinearQ5kF16 => "vnext_gguf_linear_q5k_f16",
            Self::LinearQ5kTiledF16 => "vnext_gguf_linear_q5k_tiled_f16",
            Self::LinearQ6kF16 => "vnext_gguf_linear_q6k_f16",
            Self::LinearQ6kTiledF16 => "vnext_gguf_linear_q6k_tiled_f16",
            Self::LinearQ6kF32 => "vnext_gguf_linear_q6k_f32",
            Self::LinearQ6kTiledF32 => "vnext_gguf_linear_q6k_tiled_f32",
            Self::LinearQ6kF32F16 => "vnext_gguf_linear_q6k_f32_f16",
            Self::LinearQ6kTiledF32F16 => "vnext_gguf_linear_q6k_tiled_f32_f16",
            Self::LinearF16 => "vnext_gguf_linear_f16",
            Self::LinearF32 => "vnext_gguf_linear_f32",
            Self::LinearTiledF16 => "vnext_gguf_linear_tiled_f16",
            Self::LinearTiledF32 => "vnext_gguf_linear_tiled_f32",
            Self::LinearF32F16 => "vnext_gguf_linear_f32_f16",
            Self::LinearTiledF32F16 => "vnext_gguf_linear_tiled_f32_f16",
        }
    }
    pub(super) fn function(self, kernels: &CudaNativeBlockKernels) -> &CudaFunction {
        match self {
            Self::GemmQ4kF16 => &kernels.gemm_q4k_f16,
            Self::GemmQ5kF16 => &kernels.gemm_q5k_f16,
            Self::GemmQ6kF16 => &kernels.gemm_q6k_f16,
            Self::LinearQ4kF16 => &kernels.linear_q4k_f16,
            Self::LinearQ4kTiledF16 => &kernels.linear_q4k_tiled_f16,
            Self::LinearQ5kF16 => &kernels.linear_q5k_f16,
            Self::LinearQ5kTiledF16 => &kernels.linear_q5k_tiled_f16,
            Self::LinearQ6kF16 => &kernels.linear_q6k_f16,
            Self::LinearQ6kTiledF16 => &kernels.linear_q6k_tiled_f16,
            Self::LinearQ6kF32 => &kernels.linear_q6k_f32,
            Self::LinearQ6kTiledF32 => &kernels.linear_q6k_tiled_f32,
            Self::LinearQ6kF32F16 => &kernels.linear_q6k_f32_f16,
            Self::LinearQ6kTiledF32F16 => &kernels.linear_q6k_tiled_f32_f16,
            Self::LinearF16 => &kernels.linear_f16,
            Self::LinearF32 => &kernels.linear_f32,
            Self::LinearTiledF16 => &kernels.linear_tiled_f16,
            Self::LinearTiledF32 => &kernels.linear_tiled_f32,
            Self::LinearF32F16 => &kernels.linear_f32_f16,
            Self::LinearTiledF32F16 => &kernels.linear_tiled_f32_f16,
        }
    }
}

pub(super) struct NativeLinearLaunch {
    pub(super) kernel: NativeLinearKernel,
    pub(super) config: LaunchConfig,
    pub(super) tile: [u32; 2],
}

pub(super) fn select(
    part: &weights::MatrixPart,
    rows: u32,
    output_stride: u32,
    input_type: ElementType,
    output_type: ElementType,
) -> Result<NativeLinearLaunch, CudaDeviceRuntimeError> {
    if rows == 0
        || rows > u16::MAX as u32
        || part
            .output_offset
            .checked_add(part.rows)
            .is_none_or(|end| end > output_stride)
    {
        return Err(CudaDeviceRuntimeError::contract(
            "CUDA native linear launch extent is invalid",
        ));
    }
    let row_tile = if rows > 1 { LINEAR_ROW_TILE } else { 1 };
    let q4k = part.format == weights::MatrixFormat::Block(crate::gguf_blocks::GgufBlockFormat::Q4K);
    // Restrict this ABI-only specialization to small rows. Medium/large
    // prefill, transformed F32 inputs and all other formats keep their route.
    let q5k_small = rows < 32
        && part.format == weights::MatrixFormat::Block(crate::gguf_blocks::GgufBlockFormat::Q5K);
    let q6k = part.format == weights::MatrixFormat::Block(crate::gguf_blocks::GgufBlockFormat::Q6K);
    // Select using this physical matrix part, not the combined logical
    // output width. The medium-row range is qualified only for substantial
    // Q5K/Q6K matrices; Q4K and small matrices keep their existing crossover.
    // The F16 interface still reconstructs and accumulates in F32.
    let shared_gemm = if use_shared_gemm(part, rows, input_type, output_type) {
        match part.format {
            weights::MatrixFormat::Block(crate::gguf_blocks::GgufBlockFormat::Q4K) => {
                Some(NativeLinearKernel::GemmQ4kF16)
            }
            weights::MatrixFormat::Block(crate::gguf_blocks::GgufBlockFormat::Q5K) => {
                Some(NativeLinearKernel::GemmQ5kF16)
            }
            weights::MatrixFormat::Block(crate::gguf_blocks::GgufBlockFormat::Q6K) => {
                Some(NativeLinearKernel::GemmQ6kF16)
            }
            _ => None,
        }
    } else {
        None
    };
    let kernel = match (input_type, output_type, row_tile > 1) {
        _ if shared_gemm.is_some() => shared_gemm.unwrap(),
        (ElementType::F16, ElementType::F16, false) if q4k => NativeLinearKernel::LinearQ4kF16,
        (ElementType::F16, ElementType::F16, true) if q4k => NativeLinearKernel::LinearQ4kTiledF16,
        (ElementType::F16, ElementType::F16, false) if q5k_small => {
            NativeLinearKernel::LinearQ5kF16
        }
        (ElementType::F16, ElementType::F16, true) if q5k_small => {
            NativeLinearKernel::LinearQ5kTiledF16
        }
        (ElementType::F16, ElementType::F16, false) if q6k && rows < 32 => {
            NativeLinearKernel::LinearQ6kF16
        }
        (ElementType::F16, ElementType::F16, true) if q6k && rows < 32 => {
            NativeLinearKernel::LinearQ6kTiledF16
        }
        (ElementType::F32, ElementType::F32, false) if q6k => NativeLinearKernel::LinearQ6kF32,
        (ElementType::F32, ElementType::F32, true) if q6k => NativeLinearKernel::LinearQ6kTiledF32,
        (ElementType::F32, ElementType::F16, false) if q6k => NativeLinearKernel::LinearQ6kF32F16,
        (ElementType::F32, ElementType::F16, true) if q6k => {
            NativeLinearKernel::LinearQ6kTiledF32F16
        }
        (ElementType::F16, ElementType::F16, false) => NativeLinearKernel::LinearF16,
        (ElementType::F32, ElementType::F32, false) => NativeLinearKernel::LinearF32,
        (ElementType::F16, ElementType::F16, true) => NativeLinearKernel::LinearTiledF16,
        (ElementType::F32, ElementType::F32, true) => NativeLinearKernel::LinearTiledF32,
        (ElementType::F32, ElementType::F16, false) => NativeLinearKernel::LinearF32F16,
        (ElementType::F32, ElementType::F16, true) => NativeLinearKernel::LinearTiledF32F16,
        _ => {
            return Err(CudaDeviceRuntimeError::contract(
                "CUDA native linear activation dtype is unsupported",
            ))
        }
    };
    Ok(NativeLinearLaunch {
        kernel,
        config: LaunchConfig {
            grid_dim: if shared_gemm.is_some() {
                (part.rows.div_ceil(64), rows.div_ceil(64), 1)
            } else {
                (part.rows.div_ceil(4), rows.div_ceil(row_tile), 1)
            },
            block_dim: if shared_gemm.is_some() {
                (16, 16, 1)
            } else {
                (128, 1, 1)
            },
            shared_mem_bytes: 0,
        },
        tile: if shared_gemm.is_some() {
            [64, 64]
        } else {
            [4, row_tile]
        },
    })
}
