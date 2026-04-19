//! Dense linear projection — the baseline, uses `B::gemm` directly.
//!
//! Supports an optional learnable bias (Bert / Clip / many encoder models).
//! When `bias` is set, `forward` lowers to `gemm + add_bias` (one extra
//! dispatch on GPU backends, still part of the current command buffer).

use ferrum_kernels::backend::Backend;

use crate::traits::Linear;

/// Dense linear projection.
///
/// Holds a single weight matrix laid out row-major as `[out_features, in_features]`.
/// `forward` delegates to `B::gemm` plus (optional) `B::add_bias`.
pub struct DenseLinear<B: Backend> {
    weight: B::Buffer,
    bias: Option<B::Buffer>,
    in_features: usize,
    out_features: usize,
}

impl<B: Backend> DenseLinear<B> {
    /// Build a weight-only dense projection (no bias).
    pub fn from_rows(weight_row_major: &[f32], out_features: usize, in_features: usize) -> Self {
        debug_assert_eq!(
            weight_row_major.len(),
            out_features * in_features,
            "DenseLinear weight length mismatch"
        );
        let weight = B::from_slice(weight_row_major);
        Self {
            weight,
            bias: None,
            in_features,
            out_features,
        }
    }

    /// Build a dense projection with a bias vector of length `out_features`.
    pub fn from_rows_with_bias(
        weight_row_major: &[f32],
        bias: &[f32],
        out_features: usize,
        in_features: usize,
    ) -> Self {
        debug_assert_eq!(bias.len(), out_features, "DenseLinear bias length mismatch");
        Self {
            weight: B::from_slice(weight_row_major),
            bias: Some(B::from_slice(bias)),
            in_features,
            out_features,
        }
    }

    /// Construct by moving already-allocated `Backend` buffers.
    pub fn from_buffer(weight: B::Buffer, out_features: usize, in_features: usize) -> Self {
        Self {
            weight,
            bias: None,
            in_features,
            out_features,
        }
    }

    pub fn with_bias(mut self, bias: B::Buffer) -> Self {
        self.bias = Some(bias);
        self
    }

    pub fn weight(&self) -> &B::Buffer {
        &self.weight
    }

    pub fn bias(&self) -> Option<&B::Buffer> {
        self.bias.as_ref()
    }
}

impl<B: Backend> Linear<B> for DenseLinear<B> {
    fn in_features(&self) -> usize {
        self.in_features
    }

    fn out_features(&self) -> usize {
        self.out_features
    }

    fn forward(&self, ctx: &mut B::Context, input: &B::Buffer, out: &mut B::Buffer, m: usize) {
        B::gemm(
            ctx,
            input,
            &self.weight,
            out,
            m,
            self.out_features,
            self.in_features,
        );
        if let Some(bias) = &self.bias {
            B::add_bias(ctx, out, bias, m, self.out_features);
        }
    }
}
