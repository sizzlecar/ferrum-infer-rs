//! GPU weight pointer storage for the decode runner.
//!
//! Extracts raw GPU pointers from candle Tensors at runner initialization.
//! The CudaSlice is cloned from the candle Tensor's storage (cheap — just
//! increments an Arc-like refcount). The underlying GPU memory is shared.

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
    ///
    /// This creates an independent copy on `target_stream` so that all weight
    /// accesses are on the same stream as the decode runner. This is required
    /// for CUDA Graph capture (cross-stream accesses break graph structure).
    pub fn from_tensor(
        tensor: &Tensor,
        target_stream: &Arc<CudaStream>,
    ) -> candle_core::Result<Self> {
        if tensor.dtype() != DType::F16 {
            candle_core::bail!("GpuWeight: expected F16, got {:?}", tensor.dtype());
        }
        let tensor = tensor.contiguous()?;
        let len = tensor.elem_count();
        let (storage, _layout) = tensor.storage_and_layout();
        let cuda_storage = match &*storage {
            Storage::Cuda(cs) => cs,
            _ => candle_core::bail!("GpuWeight: tensor must be on CUDA"),
        };
        let src = cuda_storage.as_cuda_slice::<half::f16>()?;
        // Copy weight data to the target stream (D2D copy).
        // This ensures the CudaSlice is owned by target_stream,
        // avoiding cross-stream event tracking during graph capture.
        let owned = target_stream
            .clone_dtod(src)
            .map_err(|e| candle_core::Error::Msg(format!("weight clone_dtod: {e}")))?;
        drop(storage);
        Ok(Self { slice: owned, len })
    }
}

/// Per-layer weights for a transformer decoder layer.
pub struct LayerWeights {
    pub input_ln_w: GpuWeight,
    pub qkv_w: GpuWeight,
    pub q_norm_w: GpuWeight,
    pub k_norm_w: GpuWeight,
    pub o_w: GpuWeight,
    pub post_ln_w: GpuWeight,
    pub gate_up_w: GpuWeight,
    pub down_w: GpuWeight,
}

/// All weights for a Qwen3-style model, extracted from candle Tensors.
pub struct Qwen3Weights {
    pub embed_table: GpuWeight,
    pub layers: Vec<LayerWeights>,
    pub final_norm_w: GpuWeight,
    pub lm_head_w: GpuWeight,
    pub rope_cos: GpuWeight,
    pub rope_sin: GpuWeight,
}
