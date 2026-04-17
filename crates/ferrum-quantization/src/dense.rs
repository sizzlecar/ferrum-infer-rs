//! Dense linear projection — the baseline, uses `B::gemm` directly.

use ferrum_kernels::backend::Backend;

use crate::traits::Linear;

/// Dense linear projection.
///
/// Holds a single weight matrix laid out row-major as `[out_features, in_features]`.
/// `forward` delegates to `B::gemm`.
pub struct DenseLinear<B: Backend> {
    weight: B::Buffer,
    in_features: usize,
    out_features: usize,
}

impl<B: Backend> DenseLinear<B> {
    pub fn from_rows(weight_row_major: &[f32], out_features: usize, in_features: usize) -> Self {
        debug_assert_eq!(
            weight_row_major.len(),
            out_features * in_features,
            "DenseLinear weight length mismatch"
        );
        let weight = B::from_slice(weight_row_major);
        Self {
            weight,
            in_features,
            out_features,
        }
    }

    /// Construct by moving an already-allocated `Backend` buffer.
    pub fn from_buffer(weight: B::Buffer, out_features: usize, in_features: usize) -> Self {
        Self {
            weight,
            in_features,
            out_features,
        }
    }

    pub fn weight(&self) -> &B::Buffer {
        &self.weight
    }
}

impl<B: Backend> Linear<B> for DenseLinear<B> {
    fn in_features(&self) -> usize {
        self.in_features
    }

    fn out_features(&self) -> usize {
        self.out_features
    }

    fn forward(
        &self,
        ctx: &mut B::Context,
        input: &B::Buffer,
        out: &mut B::Buffer,
        m: usize,
    ) {
        B::gemm(
            ctx,
            input,
            &self.weight,
            out,
            m,
            self.out_features,
            self.in_features,
        );
    }
}
