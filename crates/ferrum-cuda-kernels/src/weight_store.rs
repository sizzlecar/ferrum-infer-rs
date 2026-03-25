//! GPU weight pointer storage for the decode runner.
//!
//! Extracts raw GPU pointers from candle Tensors at runner initialization.
//! The CudaSlice is cloned from the candle Tensor's storage (cheap — just
//! increments an Arc-like refcount). The underlying GPU memory is shared.

use candle_core::{DType, Storage, Tensor};
use cudarc::driver::CudaSlice;

/// A reference to a weight tensor on GPU.
pub struct GpuWeight {
    pub slice: CudaSlice<half::f16>,
    pub len: usize,
}

impl GpuWeight {
    /// Extract GPU weight from a candle Tensor. The tensor must be F16 and on CUDA.
    pub fn from_tensor(tensor: &Tensor) -> candle_core::Result<Self> {
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
        let owned = cuda_storage.as_cuda_slice::<half::f16>()?.clone();
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
