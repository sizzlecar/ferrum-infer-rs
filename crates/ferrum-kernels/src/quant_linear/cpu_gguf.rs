//! `Linear<CpuBackend>` impl for GGUF k-quant weights.
//!
//! Phase 3e/3: replaces the old `BackendQuantGguf::gemm_quant` impl on
//! CpuBackend. The kernel call (Q4_K dequant + `Self::gemm`) lives
//! inside `CpuGgufLinear::forward` instead of the trait method body.

use crate::backend::cpu::{CpuBackend, CpuQuantStore};
use crate::Linear;

/// CPU GGUF Linear: holds a `CpuQuantStore` (currently Q4_K-dequantised
/// weights) plus shape, dispatches via `CpuBackend::gemm`.
pub struct CpuGgufLinear {
    pub store: CpuQuantStore,
    pub in_features: usize,
    pub out_features: usize,
}

impl Linear<CpuBackend> for CpuGgufLinear {
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
        match &self.store {
            CpuQuantStore::Q4K {
                weights,
                n_rows,
                n_cols,
            } => {
                <CpuBackend as crate::backend::Backend>::gemm(
                    ctx, input, weights, out, m, *n_rows, *n_cols,
                );
            }
        }
    }
}
