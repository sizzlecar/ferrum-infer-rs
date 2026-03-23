//! Candle reference implementation of the `KernelOps` sub-traits.
//!
//! All operations are implemented using pure candle tensor ops (CPU/Metal/CUDA
//! via candle's own dispatch). This serves as the baseline that any
//! hardware-specific backend must match.

use candle_core::IndexOp;
use ferrum_interfaces::kernel_ops::{
    ActivationOps, AttentionOps, AttentionParams, KernelOps, LinearOps, NormOps, PositionOps,
    SamplingOps, SamplingParams,
};
use ferrum_interfaces::TensorRef;
use ferrum_types::{FerrumError, Result};
use std::sync::Arc;

use super::candle::CandleTensor;
#[cfg(test)]
use super::candle::CandleTensorOps;

// ---------------------------------------------------------------------------
// Helpers
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
// NormOps
// ---------------------------------------------------------------------------

pub struct CandleNormOps;

impl NormOps for CandleNormOps {
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

        // updated_residual = input + residual
        let updated = (input + residual).map_err(err)?;
        // normed = rms_norm(updated_residual)
        let normed = candle_nn::ops::rms_norm(&updated, weight, eps).map_err(err)?;

        Ok((wrap(normed)?, wrap(updated)?))
    }
}

// ---------------------------------------------------------------------------
// PositionOps
// ---------------------------------------------------------------------------

pub struct CandlePositionOps;

impl PositionOps for CandlePositionOps {
    fn rotary_embedding(
        &self,
        x: &TensorRef,
        cos_cache: &TensorRef,
        sin_cache: &TensorRef,
        position_ids: &[usize],
    ) -> Result<TensorRef> {
        use candle_core::D;

        let x = ct(x)?;
        let cos_cache = ct(cos_cache)?;
        let sin_cache = ct(sin_cache)?;

        let head_dim = *x.dims().last().ok_or_else(|| err("empty tensor"))?;
        let half_dim = head_dim / 2;
        let target_dtype = x.dtype();

        // Index into cos/sin caches for the requested position.
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

        // Split into two halves along last dim.
        let x1 = x.narrow(D::Minus1, 0, half_dim).map_err(err)?;
        let x2 = x.narrow(D::Minus1, half_dim, half_dim).map_err(err)?;

        // Apply rotation: [x1*cos - x2*sin, x1*sin + x2*cos]
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
// AttentionOps
// ---------------------------------------------------------------------------

pub struct CandleAttentionOps;

impl AttentionOps for CandleAttentionOps {
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

        // Input layout: [batch, seq, heads, head_dim]
        // Transpose to [batch, heads, seq, head_dim] for batched matmul.
        let q = q.transpose(1, 2).map_err(err)?;
        let k = k.transpose(1, 2).map_err(err)?;
        let v = v.transpose(1, 2).map_err(err)?;

        // Handle GQA: repeat KV heads to match Q heads.
        let n_rep = params.num_heads / params.num_kv_heads;
        let (k, v) = if n_rep > 1 {
            (repeat_kv(&k, n_rep)?, repeat_kv(&v, n_rep)?)
        } else {
            (k, v)
        };

        // Ensure contiguous for Metal/CUDA matmul.
        let q = q.contiguous().map_err(err)?;
        let k = k.contiguous().map_err(err)?;

        // scores = Q @ K^T / scale
        let k_t = k.transpose(D::Minus2, D::Minus1).map_err(err)?;
        let k_t = k_t.contiguous().map_err(err)?;
        let scores = q.matmul(&k_t).map_err(err)?;
        let scores = scores
            .affine(params.softmax_scale as f64, 0.0)
            .map_err(err)?;

        // Causal mask.
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

        // softmax
        let attn_weights = candle_nn::ops::softmax(&scores, D::Minus1).map_err(err)?;

        // output = weights @ V
        let output = attn_weights.matmul(&v).map_err(err)?;

        // Transpose back: [batch, heads, seq, dim] -> [batch, seq, heads, dim]
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
// ActivationOps
// ---------------------------------------------------------------------------

pub struct CandleActivationOps;

impl ActivationOps for CandleActivationOps {
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
// LinearOps
// ---------------------------------------------------------------------------

pub struct CandleLinearOps;

impl LinearOps for CandleLinearOps {
    fn linear(&self, input: &TensorRef, weight: &TensorRef) -> Result<TensorRef> {
        let input = ct(input)?;
        let weight = ct(weight)?;
        // weight is [out, in], so output = input @ weight^T
        let w_t = weight.transpose(0, 1).map_err(err)?;
        let result = input.matmul(&w_t).map_err(err)?;
        wrap(result)
    }
}

// ---------------------------------------------------------------------------
// SamplingOps
// ---------------------------------------------------------------------------

pub struct CandleSamplingOps;

impl SamplingOps for CandleSamplingOps {
    fn sample_token(&self, logits: &TensorRef, _params: &SamplingParams) -> Result<u32> {
        // Reference impl: greedy (full sampling pipeline is in ferrum-sampler).
        self.argmax(logits)
    }

    fn argmax(&self, logits: &TensorRef) -> Result<u32> {
        logits.argmax_last_dim_u32()
    }
}

// ---------------------------------------------------------------------------
// Umbrella: CandleKernelOps
// ---------------------------------------------------------------------------

/// Reference `KernelOps` implementation backed by Candle tensor ops.
pub struct CandleKernelOps {
    norm: CandleNormOps,
    position: CandlePositionOps,
    attention: CandleAttentionOps,
    activation: CandleActivationOps,
    linear: CandleLinearOps,
    sampling: CandleSamplingOps,
}

impl CandleKernelOps {
    pub fn new() -> Self {
        Self {
            norm: CandleNormOps,
            position: CandlePositionOps,
            attention: CandleAttentionOps,
            activation: CandleActivationOps,
            linear: CandleLinearOps,
            sampling: CandleSamplingOps,
        }
    }
}

impl Default for CandleKernelOps {
    fn default() -> Self {
        Self::new()
    }
}

impl KernelOps for CandleKernelOps {
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
        "candle"
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backends::candle::CandleTensorFactory;
    use ferrum_interfaces::{TensorFactory, TensorOps};
    use ferrum_types::{DataType, Device};

    fn factory() -> CandleTensorFactory {
        CandleTensorFactory::new(Device::CPU)
    }

    // -- NormOps --

    #[test]
    fn test_rms_norm_matches_tensor_ops() {
        let f = factory();
        let input = f
            .from_slice(&[1.0, 2.0, 3.0, 4.0], &[1, 4], DataType::FP32, Device::CPU)
            .unwrap();
        let weight = f
            .from_slice(&[1.0, 1.0, 1.0, 1.0], &[4], DataType::FP32, Device::CPU)
            .unwrap();

        let kernel_result = CandleNormOps.rms_norm(&input, &weight, 1e-5).unwrap();
        let tensor_result = CandleTensorOps.rms_norm(&input, &weight, 1e-5).unwrap();

        let k = kernel_result.to_vec_f32().unwrap();
        let t = tensor_result.to_vec_f32().unwrap();
        assert_eq!(k.len(), t.len());
        for (a, b) in k.iter().zip(t.iter()) {
            assert!((a - b).abs() < 1e-5, "mismatch: {} vs {}", a, b);
        }
    }

    #[test]
    fn test_rms_norm_residual() {
        let f = factory();
        let input = f
            .from_slice(&[1.0, 2.0], &[1, 2], DataType::FP32, Device::CPU)
            .unwrap();
        let residual = f
            .from_slice(&[0.5, 0.5], &[1, 2], DataType::FP32, Device::CPU)
            .unwrap();
        let weight = f
            .from_slice(&[1.0, 1.0], &[2], DataType::FP32, Device::CPU)
            .unwrap();

        let (normed, updated) = CandleNormOps
            .rms_norm_residual(&input, &residual, &weight, 1e-5)
            .unwrap();

        // updated should be input + residual = [1.5, 2.5]
        let u = updated.to_vec_f32().unwrap();
        assert!((u[0] - 1.5).abs() < 1e-5);
        assert!((u[1] - 2.5).abs() < 1e-5);

        // normed should be rms_norm(updated)
        let expected = CandleNormOps
            .rms_norm(&updated, &weight, 1e-5)
            .unwrap()
            .to_vec_f32()
            .unwrap();
        let got = normed.to_vec_f32().unwrap();
        for (a, b) in got.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-5);
        }
    }

    // -- ActivationOps --

    #[test]
    fn test_silu_mul() {
        let f = factory();
        let gate = f
            .from_slice(&[1.0, -1.0, 2.0, 0.0], &[4], DataType::FP32, Device::CPU)
            .unwrap();
        let up = f
            .from_slice(&[2.0, 2.0, 2.0, 2.0], &[4], DataType::FP32, Device::CPU)
            .unwrap();

        let result = CandleActivationOps.silu_mul(&gate, &up).unwrap();
        let vals = result.to_vec_f32().unwrap();

        // silu(x) = x * sigmoid(x)
        // silu(1.0) ≈ 0.7311, * 2 ≈ 1.4621
        assert!(vals[0] > 1.0 && vals[0] < 2.0);
        // silu(0.0) = 0
        assert!(vals[3].abs() < 1e-5);
    }

    #[test]
    fn test_gelu() {
        let f = factory();
        let input = f
            .from_slice(&[0.0, 1.0, -1.0], &[3], DataType::FP32, Device::CPU)
            .unwrap();

        let result = CandleActivationOps.gelu(&input).unwrap();
        let vals = result.to_vec_f32().unwrap();
        // gelu(0) = 0
        assert!(vals[0].abs() < 1e-5);
        // gelu(1) ≈ 0.8412
        assert!(vals[1] > 0.8 && vals[1] < 0.9);
    }

    // -- LinearOps --

    #[test]
    fn test_linear_identity() {
        let f = factory();
        let input = f
            .from_slice(&[1.0, 2.0, 3.0], &[1, 3], DataType::FP32, Device::CPU)
            .unwrap();
        // Identity weight [3, 3]
        let weight = f
            .from_slice(
                &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                &[3, 3],
                DataType::FP32,
                Device::CPU,
            )
            .unwrap();

        let result = CandleLinearOps.linear(&input, &weight).unwrap();
        let vals = result.to_vec_f32().unwrap();
        assert!((vals[0] - 1.0).abs() < 1e-5);
        assert!((vals[1] - 2.0).abs() < 1e-5);
        assert!((vals[2] - 3.0).abs() < 1e-5);
    }

    // -- SamplingOps --

    #[test]
    fn test_argmax() {
        let f = factory();
        let logits = f
            .from_slice(
                &[0.1, 0.5, 0.3, 0.9, 0.2],
                &[5],
                DataType::FP32,
                Device::CPU,
            )
            .unwrap();

        let token = CandleSamplingOps.argmax(&logits).unwrap();
        assert_eq!(token, 3); // 0.9 is at index 3
    }

    // -- KernelOps umbrella --

    #[test]
    fn test_candle_kernel_ops_all_present() {
        let ops = CandleKernelOps::new();
        assert!(ops.norm_ops().is_some());
        assert!(ops.position_ops().is_some());
        assert!(ops.attention_ops().is_some());
        assert!(ops.activation_ops().is_some());
        assert!(ops.linear_ops().is_some());
        assert!(ops.sampling_ops().is_some());
        assert_eq!(ops.backend_name(), "candle");
    }

    // -- KernelOpsDispatch fallback --

    #[test]
    fn test_dispatch_fallback_rms_norm() {
        let f = factory();
        let input = f
            .from_slice(&[1.0, 2.0, 3.0, 4.0], &[1, 4], DataType::FP32, Device::CPU)
            .unwrap();
        let weight = f
            .from_slice(&[1.0, 1.0, 1.0, 1.0], &[4], DataType::FP32, Device::CPU)
            .unwrap();

        let tensor_ops = CandleTensorOps;

        // With kernel_ops = None, should fall back to TensorOps
        let dispatch = ferrum_interfaces::kernel_ops::KernelOpsDispatch::new(None, &tensor_ops);
        let result = dispatch.rms_norm(&input, &weight, 1e-5).unwrap();
        let vals = result.to_vec_f32().unwrap();
        assert_eq!(vals.len(), 4);
    }

    #[test]
    fn test_dispatch_with_kernel_ops_rms_norm() {
        let f = factory();
        let input = f
            .from_slice(&[1.0, 2.0, 3.0, 4.0], &[1, 4], DataType::FP32, Device::CPU)
            .unwrap();
        let weight = f
            .from_slice(&[1.0, 1.0, 1.0, 1.0], &[4], DataType::FP32, Device::CPU)
            .unwrap();

        let kernel_ops = CandleKernelOps::new();
        let tensor_ops = CandleTensorOps;

        // With kernel_ops present, should use KernelOps path
        let dispatch =
            ferrum_interfaces::kernel_ops::KernelOpsDispatch::new(Some(&kernel_ops), &tensor_ops);
        let result = dispatch.rms_norm(&input, &weight, 1e-5).unwrap();
        let vals = result.to_vec_f32().unwrap();
        assert_eq!(vals.len(), 4);
    }

    #[test]
    fn test_dispatch_silu_mul_fallback() {
        let f = factory();
        let gate = f
            .from_slice(&[1.0, 2.0], &[2], DataType::FP32, Device::CPU)
            .unwrap();
        let up = f
            .from_slice(&[3.0, 4.0], &[2], DataType::FP32, Device::CPU)
            .unwrap();

        let tensor_ops = CandleTensorOps;

        // No kernel ops → falls back to silu(gate) * up via TensorOps
        let dispatch = ferrum_interfaces::kernel_ops::KernelOpsDispatch::new(None, &tensor_ops);
        let result = dispatch.silu_mul(&gate, &up).unwrap();
        let vals = result.to_vec_f32().unwrap();
        assert_eq!(vals.len(), 2);
        // silu(1.0)*3 ≈ 2.19
        assert!(vals[0] > 2.0 && vals[0] < 2.5);
    }
}
