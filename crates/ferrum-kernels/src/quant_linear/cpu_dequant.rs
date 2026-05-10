//! `Linear<CpuBackend>` impl for GPTQ weights, dequantized at load time.
//!
//! Phase 3e/2: replaces the old `BackendQuantMarlin::gemm_gptq` impl on
//! CpuBackend. The kernel call (`Self::gemm` on dequantized weights)
//! lives inside `CpuGptqLinear::forward` instead of the trait method
//! body.

use crate::backend::cpu::CpuBackend;
use crate::Linear;

/// CPU GPTQ Linear: holds dequantized fp32 weights `[out_features, in_features]`
/// row-major, optional bias `[out_features]`, dispatches via `CpuBackend::gemm`.
///
/// The dequantization happens once in `BackendQuantMarlin::load_gptq` —
/// inference is just a regular f32 GEMM.
pub struct CpuGptqLinear {
    pub weight_f32: Vec<f32>,
    pub bias: Option<Vec<f32>>,
    pub in_features: usize,
    pub out_features: usize,
}

impl Linear<CpuBackend> for CpuGptqLinear {
    fn in_features(&self) -> usize {
        self.in_features
    }

    fn out_features(&self) -> usize {
        self.out_features
    }

    fn forward(
        &self,
        ctx: &mut <CpuBackend as crate::backend::Backend>::Context,
        input: &<CpuBackend as crate::backend::Backend>::Buffer,
        out: &mut <CpuBackend as crate::backend::Backend>::Buffer,
        m: usize,
    ) {
        // out[m, n] = a[m, k] @ w[n, k]^T — same contract as `B::gemm`.
        <CpuBackend as crate::backend::Backend>::gemm(
            ctx,
            input,
            &self.weight_f32,
            out,
            m,
            self.out_features,
            self.in_features,
        );
        if let Some(bias) = &self.bias {
            <CpuBackend as crate::backend::Backend>::add_bias(ctx, out, bias, m, self.out_features);
        }
    }
}
