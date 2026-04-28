//! `GgufLinear<B>`: a GGUF-sourced linear projection that integrates with
//! ferrum's `Linear<B>` trait.
//!
//! Phase 1B uses an **eager-dequant-at-load** strategy: when constructed from
//! a candle `QTensor`, the quantized payload is decoded to fp32 once on CPU,
//! then handed to `DenseLinear<B>` so the runtime path goes through the
//! standard `B::gemm` kernel. This is the simplest correct path that works
//! uniformly across CPU / Metal / CUDA without per-backend bridging code.
//!
//! Trade-off: we lose GGUF's memory advantage (Q4_K_M @ 4.5 bits/weight
//! becomes fp32 @ 32 bits/weight in RAM) and we don't get fused
//! dequant-matmul perf. Phase 1D will replace this with a real
//! quantization-aware Linear that holds the QTensor and dispatches to
//! Metal / CUDA Q4_K_M kernels.
//!
//! Why a dedicated `GgufLinear<B>` type instead of just returning
//! `DenseLinear<B>`? So Phase 1D can swap the internals (eager dequant →
//! lazy QMatMul) without churning the public API of any `WeightLoader`
//! that already returns `Box<dyn Linear<B>>`.

use candle_core::quantized::QTensor;
use candle_core::{Device, Result as CandleResult};
use ferrum_kernels::backend::Backend;

use crate::dense::DenseLinear;
use crate::traits::Linear;

/// Linear projection backed by a GGUF-sourced quantized tensor.
///
/// Internally a `DenseLinear<B>` (Phase 1B), so the runtime path is the same
/// as a plain dense weight. The distinct type lets later phases evolve the
/// representation without changing call sites.
pub struct GgufLinear<B: Backend> {
    inner: DenseLinear<B>,
}

impl<B: Backend> GgufLinear<B> {
    /// Build from a candle `QTensor` previously read out of a GGUF file.
    ///
    /// Expects a 2-D weight whose shape is `[out_features, in_features]`
    /// (the GGUF convention for linear projections — rows are output
    /// neurons). Errors if the rank is wrong or the dequant step fails.
    pub fn from_qtensor(qt: &QTensor) -> CandleResult<Self> {
        let dims = qt.shape().dims();
        if dims.len() != 2 {
            return Err(candle_core::Error::Msg(format!(
                "GgufLinear: expected 2-D weight tensor, got rank {} (shape {:?})",
                dims.len(),
                dims
            )));
        }
        let out_features = dims[0];
        let in_features = dims[1];
        let weights = dequantize_to_vec(qt)?;
        Ok(Self {
            inner: DenseLinear::<B>::from_rows(&weights, out_features, in_features),
        })
    }

    /// Build with a bias vector. `bias_qt` must be a 1-D `[out_features]`
    /// tensor — typical for Qwen2.5 / Bert / any model with attention bias.
    pub fn from_qtensor_with_bias(qt: &QTensor, bias_qt: &QTensor) -> CandleResult<Self> {
        let weight_dims = qt.shape().dims();
        if weight_dims.len() != 2 {
            return Err(candle_core::Error::Msg(format!(
                "GgufLinear: expected 2-D weight, got rank {}",
                weight_dims.len()
            )));
        }
        let out_features = weight_dims[0];
        let in_features = weight_dims[1];

        let bias_dims = bias_qt.shape().dims();
        if bias_dims.len() != 1 || bias_dims[0] != out_features {
            return Err(candle_core::Error::Msg(format!(
                "GgufLinear: bias shape {:?} doesn't match weight out_features {}",
                bias_dims, out_features
            )));
        }

        let weights = dequantize_to_vec(qt)?;
        let bias = dequantize_to_vec(bias_qt)?;
        Ok(Self {
            inner: DenseLinear::<B>::from_rows_with_bias(
                &weights,
                &bias,
                out_features,
                in_features,
            ),
        })
    }

    /// Build directly from already-dequantized fp32 weights. Useful when the
    /// caller has already paid the dequant cost (e.g. cached weights, or
    /// constructing from synthetic data in tests).
    pub fn from_dense_rows(
        weight_row_major: &[f32],
        out_features: usize,
        in_features: usize,
    ) -> Self {
        Self {
            inner: DenseLinear::<B>::from_rows(weight_row_major, out_features, in_features),
        }
    }
}

impl<B: Backend> Linear<B> for GgufLinear<B> {
    fn in_features(&self) -> usize {
        self.inner.in_features()
    }

    fn out_features(&self) -> usize {
        self.inner.out_features()
    }

    fn forward(&self, ctx: &mut B::Context, input: &B::Buffer, out: &mut B::Buffer, m: usize) {
        self.inner.forward(ctx, input, out, m);
    }
}

/// Convenience: build a boxed `Linear<B>` from a `QTensor`. Useful for
/// `WeightLoader` impls that want a uniform `Box<dyn Linear<B>>` output.
pub fn linear_from_qtensor<B: Backend>(qt: &QTensor) -> CandleResult<Box<dyn Linear<B>>> {
    Ok(Box::new(GgufLinear::<B>::from_qtensor(qt)?))
}

/// Dequantize on CPU, flatten to a contiguous `Vec<f32>` in row-major order.
/// Pulled out so weight + bias paths share the same conversion.
fn dequantize_to_vec(qt: &QTensor) -> CandleResult<Vec<f32>> {
    let dense = qt.dequantize(&Device::Cpu)?;
    let flat = dense.flatten_all()?;
    flat.to_vec1::<f32>()
}
