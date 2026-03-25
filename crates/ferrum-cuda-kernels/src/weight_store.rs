//! GPU weight pointer storage for the decode runner.
//!
//! Extracts raw GPU pointers from candle Tensors at runner initialization.
//! The original Tensors must be kept alive to prevent deallocation.

use candle_core::{DType, Storage, Tensor};
use cudarc::driver::CudaSlice;

/// A reference to a weight tensor on GPU, holding the CudaSlice directly.
///
/// The CudaSlice is cloned from the candle Tensor's storage (cheap — just
/// increments an Arc-like refcount). The underlying GPU memory is shared.
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
        let (storage, layout) = tensor.storage_and_layout();
        let cuda_storage = match &*storage {
            Storage::Cuda(cs) => cs,
            _ => candle_core::bail!("GpuWeight: tensor must be on CUDA"),
        };
        let slice = cuda_storage.as_cuda_slice::<half::f16>()?;
        let slice = slice.slice(layout.start_offset()..);
        // Clone the CudaView into an owned CudaSlice by using the slice range
        // Actually, cudarc's slice() returns a CudaView which borrows. We need
        // to keep the original storage alive. Let's just clone the full slice
        // and note the offset.
        let full_slice = cuda_storage.as_cuda_slice::<half::f16>()?;
        let len = tensor.elem_count();
        drop(storage);
        // We need an owned CudaSlice. Re-extract from the contiguous tensor.
        let (storage, _layout) = tensor.storage_and_layout();
        let cuda_storage = match &*storage {
            Storage::Cuda(cs) => cs,
            _ => unreachable!(),
        };
        let owned = cuda_storage.as_cuda_slice::<half::f16>()?.clone();
        drop(storage);
        Ok(Self { slice: owned, len })
    }
}

/// Per-layer weights for a transformer decoder layer.
pub struct LayerWeights {
    /// Input layer norm weight: [hidden_size]
    pub input_ln_w: GpuWeight,
    /// Fused QKV projection weight: [q_dim + 2*kv_dim, hidden_size]
    pub qkv_w: GpuWeight,
    /// Q-norm weight: [head_dim]
    pub q_norm_w: GpuWeight,
    /// K-norm weight: [head_dim]
    pub k_norm_w: GpuWeight,
    /// O-projection weight: [hidden_size, q_dim]
    pub o_w: GpuWeight,
    /// Post-attention layer norm weight: [hidden_size]
    pub post_ln_w: GpuWeight,
    /// Fused gate+up projection weight: [2*intermediate_size, hidden_size]
    pub gate_up_w: GpuWeight,
    /// Down projection weight: [hidden_size, intermediate_size]
    pub down_w: GpuWeight,
}

/// All weights for a Qwen3-style model, extracted from candle Tensors.
pub struct Qwen3Weights {
    /// Embedding table: [vocab_size, hidden_size]
    pub embed_table: GpuWeight,
    /// Per-layer decoder weights
    pub layers: Vec<LayerWeights>,
    /// Final RMS norm weight: [hidden_size]
    pub final_norm_w: GpuWeight,
    /// LM head projection weight: [vocab_size, hidden_size]
    pub lm_head_w: GpuWeight,
    /// RoPE cos table: [max_seq_len, head_dim/2]
    pub rope_cos: GpuWeight,
    /// RoPE sin table: [max_seq_len, head_dim/2]
    pub rope_sin: GpuWeight,
}
