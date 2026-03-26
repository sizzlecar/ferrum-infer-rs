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

/// Per-layer weights for a transformer decoder layer.
pub struct LayerWeights {
    pub input_ln_w: GpuWeight,
    pub qkv_w: GpuWeight,
    pub q_norm_w: Option<GpuWeight>,
    pub k_norm_w: Option<GpuWeight>,
    pub o_w: GpuWeight,
    pub post_ln_w: GpuWeight,
    pub gate_up_w: GpuWeight,
    pub down_w: GpuWeight,
}

/// All weights for a transformer model on GPU.
/// Architecture-agnostic — uses the same struct for Qwen3, Llama, Qwen2.
pub struct TransformerGpuWeights {
    pub embed_table: GpuWeight,
    pub layers: Vec<LayerWeights>,
    pub final_norm_w: GpuWeight,
    pub lm_head_w: GpuWeight,
    pub rope_cos: GpuWeight,
    pub rope_sin: GpuWeight,
}

/// Backward compat alias.
pub type Qwen3Weights = TransformerGpuWeights;
