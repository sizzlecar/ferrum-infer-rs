//! CUDA `KernelOps` implementation.
//!
//! Delegates all operations to candle tensor ops, which candle dispatches to
//! CUDA kernels automatically when tensors reside on a CUDA device. This
//! includes native CUDA implementations for rms-norm, rotary-emb,
//! softmax-last-dim, and all standard tensor ops.
//!
//! Future: wire FlashAttention-2 FFI for `paged_attention`.

use ferrum_interfaces::kernel_ops::{
    ActivationOps, AttentionOps, AttentionParams, KernelOps, LinearOps, NormOps, PositionOps,
    SamplingOps, SamplingParams,
};
use ferrum_interfaces::TensorRef;
use ferrum_runtime::backends::CandleTensor;
use ferrum_types::{FerrumError, Result};
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Helpers (same pattern as CandleKernelOps / MetalKernelOps)
// ---------------------------------------------------------------------------

fn ct(tensor: &TensorRef) -> Result<&candle_core::Tensor> {
    let concrete: &CandleTensor = unsafe { &*(Arc::as_ptr(tensor) as *const CandleTensor) };
    Ok(concrete.inner())
}

fn wrap(tensor: candle_core::Tensor) -> Result<TensorRef> {
    Ok(Arc::new(CandleTensor::new(tensor)?) as TensorRef)
}

fn err(msg: impl std::fmt::Display) -> FerrumError {
    FerrumError::backend(msg.to_string())
}

// ---------------------------------------------------------------------------
// NormOps — candle's rms_norm dispatches to CUDA natively
// ---------------------------------------------------------------------------

pub struct CudaNormOps;

impl NormOps for CudaNormOps {
    fn rms_norm(&self, input: &TensorRef, weight: &TensorRef, eps: f32) -> Result<TensorRef> {
        let input = ct(input)?;
        let weight = ct(weight)?;
        let result = candle_nn::ops::rms_norm(input, weight, eps).map_err(err)?;
        wrap(result)
    }

    fn rms_norm_residual(
        &self,
        input: &TensorRef,
        residual: &TensorRef,
        weight: &TensorRef,
        eps: f32,
    ) -> Result<(TensorRef, TensorRef)> {
        let input = ct(input)?;
        let residual = ct(residual)?;
        let weight = ct(weight)?;
        let updated = (input + residual).map_err(err)?;
        let normed = candle_nn::ops::rms_norm(&updated, weight, eps).map_err(err)?;
        Ok((wrap(normed)?, wrap(updated)?))
    }
}

// ---------------------------------------------------------------------------
// PositionOps — candle's rotary_emb dispatches to CUDA natively
// ---------------------------------------------------------------------------

pub struct CudaPositionOps;

impl PositionOps for CudaPositionOps {
    fn rotary_embedding(
        &self,
        x: &TensorRef,
        cos_cache: &TensorRef,
        sin_cache: &TensorRef,
        position_ids: &[usize],
    ) -> Result<TensorRef> {
        use candle_core::{IndexOp, D};

        let x = ct(x)?;
        let cos_cache = ct(cos_cache)?;
        let sin_cache = ct(sin_cache)?;

        let head_dim = *x.dims().last().ok_or_else(|| err("empty tensor"))?;
        let half_dim = head_dim / 2;
        let target_dtype = x.dtype();

        let pos = position_ids
            .first()
            .copied()
            .ok_or_else(|| err("empty position_ids"))?;
        let cos = cos_cache.i(pos).map_err(err)?;
        let sin = sin_cache.i(pos).map_err(err)?;

        let cos = if cos.dtype() != target_dtype {
            cos.to_dtype(target_dtype).map_err(err)?
        } else {
            cos
        };
        let sin = if sin.dtype() != target_dtype {
            sin.to_dtype(target_dtype).map_err(err)?
        } else {
            sin
        };

        let x1 = x.narrow(D::Minus1, 0, half_dim).map_err(err)?;
        let x2 = x.narrow(D::Minus1, half_dim, half_dim).map_err(err)?;

        let r1 = x1
            .broadcast_mul(&cos)
            .map_err(err)?
            .broadcast_sub(&x2.broadcast_mul(&sin).map_err(err)?)
            .map_err(err)?;
        let r2 = x1
            .broadcast_mul(&sin)
            .map_err(err)?
            .broadcast_add(&x2.broadcast_mul(&cos).map_err(err)?)
            .map_err(err)?;

        let result = candle_core::Tensor::cat(&[r1, r2], D::Minus1).map_err(err)?;
        wrap(result)
    }
}

// ---------------------------------------------------------------------------
// AttentionOps — candle matmul/softmax on CUDA
// ---------------------------------------------------------------------------

pub struct CudaAttentionOps;

impl AttentionOps for CudaAttentionOps {
    fn attention(
        &self,
        q: &TensorRef,
        k: &TensorRef,
        v: &TensorRef,
        params: &AttentionParams,
    ) -> Result<TensorRef> {
        use candle_core::D;

        let q = ct(q)?;
        let k = ct(k)?;
        let v = ct(v)?;

        let q = q.transpose(1, 2).map_err(err)?;
        let k = k.transpose(1, 2).map_err(err)?;
        let v = v.transpose(1, 2).map_err(err)?;

        let n_rep = params.num_heads / params.num_kv_heads;
        let (k, v) = if n_rep > 1 {
            (repeat_kv(&k, n_rep)?, repeat_kv(&v, n_rep)?)
        } else {
            (k, v)
        };

        let q = q.contiguous().map_err(err)?;
        let k = k.contiguous().map_err(err)?;

        let k_t = k.transpose(D::Minus2, D::Minus1).map_err(err)?;
        let k_t = k_t.contiguous().map_err(err)?;
        let scores = q.matmul(&k_t).map_err(err)?;
        let scores = scores
            .affine(params.softmax_scale as f64, 0.0)
            .map_err(err)?;

        let scores = if params.causal {
            let (_, _, q_len, kv_len) = scores.dims4().map_err(err)?;
            let past_len = kv_len.saturating_sub(q_len);
            let mask_data: Vec<f32> = (0..q_len)
                .flat_map(|i| {
                    let max_k = past_len + i;
                    (0..kv_len).map(move |j| if j <= max_k { 0.0 } else { f32::NEG_INFINITY })
                })
                .collect();
            let mask =
                candle_core::Tensor::from_vec(mask_data, (1, 1, q_len, kv_len), scores.device())
                    .map_err(err)?;
            let mask = if mask.dtype() != scores.dtype() {
                mask.to_dtype(scores.dtype()).map_err(err)?
            } else {
                mask
            };
            scores.broadcast_add(&mask).map_err(err)?
        } else {
            scores
        };

        let attn_weights = candle_nn::ops::softmax(&scores, D::Minus1).map_err(err)?;
        let output = attn_weights.matmul(&v).map_err(err)?;
        let output = output.transpose(1, 2).map_err(err)?;
        wrap(output)
    }

    // TODO: wire FlashAttention-2 FFI for paged_attention
}

fn repeat_kv(x: &candle_core::Tensor, n_rep: usize) -> Result<candle_core::Tensor> {
    let (batch, num_kv_heads, seq_len, head_dim) = x.dims4().map_err(err)?;
    let unsqueezed = x.unsqueeze(2).map_err(err)?;
    let repeated: Vec<candle_core::Tensor> = (0..n_rep).map(|_| unsqueezed.clone()).collect();
    let cat = candle_core::Tensor::cat(&repeated, 2).map_err(err)?;
    cat.reshape((batch, num_kv_heads * n_rep, seq_len, head_dim))
        .map_err(err)
}

// ---------------------------------------------------------------------------
// ActivationOps — candle silu/gelu on CUDA
// ---------------------------------------------------------------------------

pub struct CudaActivationOps;

impl ActivationOps for CudaActivationOps {
    fn silu_mul(&self, gate: &TensorRef, up: &TensorRef) -> Result<TensorRef> {
        let gate = ct(gate)?;
        let up = ct(up)?;
        let activated = candle_nn::ops::silu(gate).map_err(err)?;
        let result = activated.mul(up).map_err(err)?;
        wrap(result)
    }

    fn gelu(&self, input: &TensorRef) -> Result<TensorRef> {
        let input = ct(input)?;
        let result = input.gelu().map_err(err)?;
        wrap(result)
    }
}

// ---------------------------------------------------------------------------
// LinearOps — candle matmul on CUDA (cuBLAS)
// ---------------------------------------------------------------------------

pub struct CudaLinearOps;

impl LinearOps for CudaLinearOps {
    fn linear(&self, input: &TensorRef, weight: &TensorRef) -> Result<TensorRef> {
        let input = ct(input)?;
        let weight = ct(weight)?;
        let w_t = weight.transpose(0, 1).map_err(err)?;
        let result = input.matmul(&w_t).map_err(err)?;
        wrap(result)
    }
}

// ---------------------------------------------------------------------------
// SamplingOps — CPU-side sampling (logits transferred back)
// ---------------------------------------------------------------------------

pub struct CudaSamplingOps;

impl SamplingOps for CudaSamplingOps {
    fn sample_token(&self, logits: &TensorRef, _params: &SamplingParams) -> Result<u32> {
        self.argmax(logits)
    }

    fn argmax(&self, logits: &TensorRef) -> Result<u32> {
        logits.argmax_last_dim_u32()
    }
}

// ---------------------------------------------------------------------------
// Umbrella: CudaKernelOps
// ---------------------------------------------------------------------------

/// CUDA `KernelOps` implementation.
///
/// All operations delegate to candle tensor ops which dispatch to CUDA
/// kernels (cuBLAS, custom CUDA kernels in candle) automatically.
pub struct CudaKernelOps {
    norm: CudaNormOps,
    position: CudaPositionOps,
    attention: CudaAttentionOps,
    activation: CudaActivationOps,
    linear: CudaLinearOps,
    sampling: CudaSamplingOps,
}

impl CudaKernelOps {
    pub fn new() -> Self {
        Self {
            norm: CudaNormOps,
            position: CudaPositionOps,
            attention: CudaAttentionOps,
            activation: CudaActivationOps,
            linear: CudaLinearOps,
            sampling: CudaSamplingOps,
        }
    }
}

impl Default for CudaKernelOps {
    fn default() -> Self {
        Self::new()
    }
}

impl KernelOps for CudaKernelOps {
    fn norm_ops(&self) -> Option<&dyn NormOps> {
        Some(&self.norm)
    }
    fn position_ops(&self) -> Option<&dyn PositionOps> {
        Some(&self.position)
    }
    fn attention_ops(&self) -> Option<&dyn AttentionOps> {
        Some(&self.attention)
    }
    fn activation_ops(&self) -> Option<&dyn ActivationOps> {
        Some(&self.activation)
    }
    fn linear_ops(&self) -> Option<&dyn LinearOps> {
        Some(&self.linear)
    }
    fn sampling_ops(&self) -> Option<&dyn SamplingOps> {
        Some(&self.sampling)
    }
    fn backend_name(&self) -> &str {
        "cuda"
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cuda_kernel_ops_all_present() {
        let ops = CudaKernelOps::new();
        assert!(ops.norm_ops().is_some());
        assert!(ops.position_ops().is_some());
        assert!(ops.attention_ops().is_some());
        assert!(ops.activation_ops().is_some());
        assert!(ops.linear_ops().is_some());
        assert!(ops.sampling_ops().is_some());
        assert_eq!(ops.backend_name(), "cuda");
    }
}
