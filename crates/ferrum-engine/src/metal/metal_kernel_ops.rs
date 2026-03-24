//! Metal `KernelOps` implementation.
//!
//! Wraps existing Metal compute pipelines (RmsNorm, Sampling, Q4_0 quantized
//! matmul) behind the `KernelOps` sub-trait interface defined in
//! `ferrum_interfaces::kernel_ops`. Operations without a dedicated Metal
//! kernel delegate to candle tensor ops, which candle dispatches to the Metal
//! GPU automatically when tensors reside on a Metal device.

use crate::metal::compute_pipeline::{Q4_0MatvecPipeline, RmsNormPipeline};
use crate::metal::sampling_ops::MetalSamplingOps as RawMetalSamplingOps;
use crate::metal::MetalContext;
use ferrum_interfaces::kernel_ops::{
    ActivationOps, AttentionOps, AttentionParams, KernelOps, LinearOps, NormOps, PositionOps,
    QuantScheme, SamplingOps, SamplingParams,
};
use ferrum_interfaces::TensorRef;
use ferrum_runtime::backends::CandleTensor;
use ferrum_types::{FerrumError, Result};
use half::f16;
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Helpers (same pattern as CandleKernelOps)
// ---------------------------------------------------------------------------

/// Extract the inner `candle_core::Tensor` from a `TensorRef`.
fn ct(tensor: &TensorRef) -> Result<&candle_core::Tensor> {
    let concrete: &CandleTensor = unsafe { &*(Arc::as_ptr(tensor) as *const CandleTensor) };
    Ok(concrete.inner())
}

/// Wrap a `candle_core::Tensor` into a `TensorRef`.
fn wrap(tensor: candle_core::Tensor) -> Result<TensorRef> {
    Ok(Arc::new(CandleTensor::new(tensor)?) as TensorRef)
}

fn err(msg: impl std::fmt::Display) -> FerrumError {
    FerrumError::backend(msg.to_string())
}

/// Extract a flat `Vec<f32>` from a candle tensor, handling FP16/BF16 → F32.
fn to_f32_vec(t: &candle_core::Tensor) -> Result<Vec<f32>> {
    use candle_core::DType;
    let flat = t.flatten_all().map_err(err)?;
    match flat.dtype() {
        DType::F32 => flat.to_vec1().map_err(err),
        DType::F16 => {
            let v16: Vec<f16> = flat.to_vec1().map_err(err)?;
            Ok(v16.iter().map(|v| f32::from(*v)).collect())
        }
        _ => {
            let converted = flat.to_dtype(DType::F32).map_err(err)?;
            converted.to_vec1().map_err(err)
        }
    }
}

/// Rebuild a candle tensor from f32 data, converting back to `target_dtype`.
fn from_f32_vec(
    data: Vec<f32>,
    shape: &[usize],
    device: &candle_core::Device,
    target_dtype: candle_core::DType,
) -> Result<candle_core::Tensor> {
    let t = candle_core::Tensor::from_vec(data, shape, device).map_err(err)?;
    if target_dtype != candle_core::DType::F32 {
        t.to_dtype(target_dtype).map_err(err)
    } else {
        Ok(t)
    }
}

// ---------------------------------------------------------------------------
// NormOps — wraps RmsNormPipeline (real Metal kernel)
// ---------------------------------------------------------------------------

pub struct MetalNormOps {
    pipeline: RmsNormPipeline,
}

impl MetalNormOps {
    pub fn new(context: Arc<MetalContext>) -> Result<Self> {
        let pipeline = RmsNormPipeline::new(context)?;
        Ok(Self { pipeline })
    }
}

impl NormOps for MetalNormOps {
    fn rms_norm(&self, input: &TensorRef, weight: &TensorRef, eps: f32) -> Result<TensorRef> {
        let input_t = ct(input)?;
        let weight_t = ct(weight)?;
        let dims = input_t.dims();
        let hidden_size = *dims.last().ok_or_else(|| err("empty tensor"))?;
        let batch_size: usize = dims[..dims.len() - 1].iter().product::<usize>().max(1);

        let input_f32 = to_f32_vec(input_t)?;
        let weight_f32 = to_f32_vec(weight_t)?;

        let result =
            self.pipeline
                .forward_batched(&input_f32, &weight_f32, batch_size, hidden_size, eps)?;

        let out = from_f32_vec(result, dims, input_t.device(), input_t.dtype())?;
        wrap(out)
    }

    fn rms_norm_residual(
        &self,
        input: &TensorRef,
        residual: &TensorRef,
        weight: &TensorRef,
        eps: f32,
    ) -> Result<(TensorRef, TensorRef)> {
        let input_t = ct(input)?;
        let residual_t = ct(residual)?;
        let weight_t = ct(weight)?;
        let dims = input_t.dims();
        let hidden_size = *dims.last().ok_or_else(|| err("empty tensor"))?;
        let batch_size: usize = dims[..dims.len() - 1].iter().product::<usize>().max(1);

        let input_f32 = to_f32_vec(input_t)?;
        let residual_f32 = to_f32_vec(residual_t)?;
        let weight_f32 = to_f32_vec(weight_t)?;

        let normed = self.pipeline.forward_with_residual(
            &input_f32,
            &residual_f32,
            &weight_f32,
            batch_size,
            hidden_size,
            eps,
        )?;

        // Compute updated_residual = input + residual
        let updated_residual = (input_t + residual_t).map_err(err)?;

        let normed_t = from_f32_vec(normed, dims, input_t.device(), input_t.dtype())?;
        Ok((wrap(normed_t)?, wrap(updated_residual)?))
    }
}

// ---------------------------------------------------------------------------
// PositionOps — candle-on-Metal (candle auto-dispatches to Metal GPU)
// ---------------------------------------------------------------------------

pub struct MetalPositionOps;

impl PositionOps for MetalPositionOps {
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
// AttentionOps — candle-on-Metal
// ---------------------------------------------------------------------------

pub struct MetalAttentionOps;

impl AttentionOps for MetalAttentionOps {
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

        // [batch, seq, heads, head_dim] → [batch, heads, seq, head_dim]
        let q = q.transpose(1, 2).map_err(err)?;
        let k = k.transpose(1, 2).map_err(err)?;
        let v = v.transpose(1, 2).map_err(err)?;

        // GQA: repeat KV heads
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

        // [batch, heads, seq, dim] → [batch, seq, heads, dim]
        let output = output.transpose(1, 2).map_err(err)?;
        wrap(output)
    }
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
// ActivationOps — candle-on-Metal
// ---------------------------------------------------------------------------

pub struct MetalActivationOps;

impl ActivationOps for MetalActivationOps {
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
// LinearOps — linear via candle-on-Metal, quantized via Q4_0 Metal kernel
// ---------------------------------------------------------------------------

pub struct MetalLinearOps {
    q4_pipeline: Option<Q4_0MatvecPipeline>,
}

impl MetalLinearOps {
    pub fn new(context: Arc<MetalContext>) -> Self {
        let q4_pipeline = Q4_0MatvecPipeline::new(context).ok();
        Self { q4_pipeline }
    }
}

impl LinearOps for MetalLinearOps {
    fn linear(&self, input: &TensorRef, weight: &TensorRef) -> Result<TensorRef> {
        let input = ct(input)?;
        let weight = ct(weight)?;
        let w_t = weight.transpose(0, 1).map_err(err)?;
        let result = input.matmul(&w_t).map_err(err)?;
        wrap(result)
    }

    fn quantized_linear(
        &self,
        input: &TensorRef,
        packed_weight: &TensorRef,
        scheme: &QuantScheme,
    ) -> Result<TensorRef> {
        let QuantScheme::Q4_0 { .. } = scheme else {
            return Err(FerrumError::unsupported(
                "Metal quantized_linear only supports Q4_0",
            ));
        };
        let pipeline = self
            .q4_pipeline
            .as_ref()
            .ok_or_else(|| FerrumError::backend("Q4_0 Metal pipeline not available"))?;

        let input_t = ct(input)?;
        let packed_t = ct(packed_weight)?;

        // packed_weight: [nrows, blocks_per_row * 5] u32 layout
        let packed_dims = packed_t.dims();
        let nrows = packed_dims[0];
        let input_f32 = to_f32_vec(input_t)?;
        let ncols = input_f32.len();

        // Extract packed weights as u32
        let packed_flat = packed_t.flatten_all().map_err(err)?;
        let packed_f32: Vec<f32> = packed_flat.to_vec1().map_err(err)?;
        let packed_u32: Vec<u32> = packed_f32.iter().map(|v| *v as u32).collect();

        let result = pipeline.execute_matvec(&packed_u32, &input_f32, nrows, ncols)?;

        let out = from_f32_vec(result, &[1, nrows], input_t.device(), input_t.dtype())?;
        wrap(out)
    }
}

// ---------------------------------------------------------------------------
// SamplingOps — wraps existing MetalSamplingOps (real Metal kernel)
// ---------------------------------------------------------------------------

pub struct MetalSamplingOpsAdapter {
    inner: RawMetalSamplingOps,
}

impl MetalSamplingOpsAdapter {
    pub fn new(context: Arc<MetalContext>) -> Result<Self> {
        let inner = RawMetalSamplingOps::new(context)?;
        Ok(Self { inner })
    }
}

impl SamplingOps for MetalSamplingOpsAdapter {
    fn sample_token(&self, logits: &TensorRef, params: &SamplingParams) -> Result<u32> {
        let logits_t = ct(logits)?;
        self.inner.sample_token(
            logits_t,
            params.top_k,
            params.top_p,
            params.temperature,
            params.repetition_penalty,
            &params.repetition_token_ids,
            &params.repetition_token_freqs,
            params.rng_seed,
        )
    }

    fn argmax(&self, logits: &TensorRef) -> Result<u32> {
        logits.argmax_last_dim_u32()
    }
}

// ---------------------------------------------------------------------------
// Umbrella: MetalKernelOps
// ---------------------------------------------------------------------------

/// Metal `KernelOps` implementation.
///
/// Uses dedicated Metal compute kernels for RMS-norm, sampling, and Q4_0
/// quantized matmul. All other operations (attention, RoPE, activations,
/// dense linear) delegate to candle tensor ops which candle dispatches to
/// the Metal GPU automatically.
pub struct MetalKernelOps {
    norm: MetalNormOps,
    position: MetalPositionOps,
    attention: MetalAttentionOps,
    activation: MetalActivationOps,
    linear: MetalLinearOps,
    sampling: MetalSamplingOpsAdapter,
}

impl MetalKernelOps {
    pub fn new(context: Arc<MetalContext>) -> Result<Self> {
        Ok(Self {
            norm: MetalNormOps::new(context.clone())?,
            position: MetalPositionOps,
            attention: MetalAttentionOps,
            activation: MetalActivationOps,
            linear: MetalLinearOps::new(context.clone()),
            sampling: MetalSamplingOpsAdapter::new(context)?,
        })
    }
}

impl KernelOps for MetalKernelOps {
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
        "metal"
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // Structural test — does not require Metal hardware.
    #[test]
    fn test_metal_position_ops_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<MetalPositionOps>();
        assert_send_sync::<MetalAttentionOps>();
        assert_send_sync::<MetalActivationOps>();
    }
}
