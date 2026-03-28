//! GPU weight storage for the decode runner.
//!
//! Stores model weights on a specific CUDA stream for graph-safe access.

use std::sync::Arc;

use candle_core::{DType, Storage, Tensor};
use cudarc::driver::{CudaSlice, CudaStream};

/// A weight tensor on GPU, owned by a specific stream.
pub struct GpuWeight {
    pub slice: CudaSlice<half::f16>,
    pub len: usize,
}

impl GpuWeight {
    /// Extract GPU weight from a candle Tensor and copy to the given stream.
    pub fn from_tensor(
        tensor: &Tensor,
        target_stream: &Arc<CudaStream>,
    ) -> candle_core::Result<Self> {
        if tensor.dtype() != DType::F16 {
            candle_core::bail!("GpuWeight: expected F16, got {:?}", tensor.dtype());
        }
        let tensor = tensor.contiguous()?;
        let len = tensor.elem_count();
        let (storage, layout) = tensor.storage_and_layout();
        let cuda_storage = match &*storage {
            Storage::Cuda(cs) => cs,
            _ => candle_core::bail!("GpuWeight: tensor must be on CUDA"),
        };
        let src = cuda_storage.as_cuda_slice::<half::f16>()?;
        // Apply layout offset — tensor may be a view into a larger storage
        let offset = layout.start_offset();
        if offset != 0 {
            tracing::warn!(
                "GpuWeight: tensor has non-zero start_offset={}, len={}, storage_len={}",
                offset,
                len,
                src.len()
            );
        }
        let src_view = src.slice(offset..offset + len);
        let owned = target_stream
            .clone_dtod(&src_view)
            .map_err(|e| candle_core::Error::Msg(format!("weight clone_dtod: {e}")))?;
        drop(storage);
        Ok(Self { slice: owned, len })
    }
}

/// INT4 quantized weight on GPU (GPTQ format).
pub struct GpuQuantWeight {
    /// Packed INT4 weights: [K/8, N] int32 (8 values per word)
    pub qweight: CudaSlice<i32>,
    /// Per-group FP16 scales: [K/group_size, N]
    pub scales: CudaSlice<half::f16>,
    /// Per-group packed zero-points: [K/group_size, N/8] int32 (None for symmetric)
    pub qzeros: Option<CudaSlice<i32>>,
    /// Input dimension (K)
    pub k: usize,
    /// Output dimension (N)
    pub n: usize,
    /// Quantization group size (typically 128)
    pub group_size: usize,
    /// Whether symmetric quantization (zero_point fixed at 8)
    pub symmetric: bool,
}

/// A linear layer weight — FP16, INT4 (dequant+cuBLAS), or Marlin (fused INT4xFP16).
pub enum LinearWeight {
    Fp16(GpuWeight),
    Int4(GpuQuantWeight),
    Marlin(crate::marlin::MarlinWeight),
}

impl LinearWeight {
    /// Get the FP16 weight slice (panics for INT4 — use linear_dispatch instead).
    pub fn as_fp16(&self) -> &CudaSlice<half::f16> {
        match self {
            LinearWeight::Fp16(w) => &w.slice,
            LinearWeight::Int4(_) => panic!("Cannot get fp16 slice from INT4 weight"),
        }
    }

    pub fn is_quantized(&self) -> bool {
        matches!(self, LinearWeight::Int4(_))
    }
}

/// Per-layer weights for a transformer decoder layer.
pub struct LayerWeights {
    pub input_ln_w: GpuWeight,
    pub qkv_w: LinearWeight,
    pub q_norm_w: Option<GpuWeight>,
    pub k_norm_w: Option<GpuWeight>,
    pub o_w: LinearWeight,
    pub post_ln_w: GpuWeight,
    pub gate_up_w: LinearWeight,
    pub down_w: LinearWeight,
}

/// All weights for a transformer model on GPU.
/// Architecture-agnostic — uses the same struct for Qwen3, Llama, Qwen2.
pub struct TransformerGpuWeights {
    pub embed_table: GpuWeight,
    pub layers: Vec<LayerWeights>,
    pub final_norm_w: GpuWeight,
    pub lm_head_w: LinearWeight,
    pub rope_cos: GpuWeight,
    pub rope_sin: GpuWeight,
}

/// Backward compat alias.
pub type Qwen3Weights = TransformerGpuWeights;
