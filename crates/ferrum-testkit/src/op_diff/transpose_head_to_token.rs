//! `transpose_head_to_token` op-diff harness — see `crate::op_diff`.
//!
//! Reorders `[tokens, heads, dim]` head-major data to token-major layout.
//! Pure data movement (exact), so accelerator NMSE should be ~0.

use super::{random_vec, OpUnderTest, Output};

pub struct TransposeHeadToTokenOp {
    pub tokens: usize,
    pub heads: usize,
    pub dim: usize,
}

impl TransposeHeadToTokenOp {
    fn elems(&self) -> usize {
        self.tokens * self.heads * self.dim
    }
}

impl OpUnderTest for TransposeHeadToTokenOp {
    fn name(&self) -> &str {
        "transpose_head_to_token"
    }

    fn run_cpu(&self, seed: u64) -> Output {
        use ferrum_kernels::backend::cpu::CpuBackend;
        use ferrum_kernels::backend::Backend;

        let src = random_vec(self.elems(), -2.0, 2.0, seed);
        let mut ctx = CpuBackend::new_context();
        let src_buf = CpuBackend::from_slice(&src);
        let mut dst = CpuBackend::alloc(self.elems());
        CpuBackend::transpose_head_to_token(
            &mut ctx,
            &src_buf,
            &mut dst,
            self.tokens,
            self.heads,
            self.dim,
        );
        CpuBackend::sync(&mut ctx);
        CpuBackend::to_vec(&dst, self.elems())
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    fn run_metal(&self, seed: u64) -> Output {
        use ferrum_kernels::backend::metal::MetalBackend;
        use ferrum_kernels::backend::Backend;

        let src = random_vec(self.elems(), -2.0, 2.0, seed);
        let mut ctx = MetalBackend::new_context();
        let src_buf = MetalBackend::from_slice(&src);
        let mut dst = MetalBackend::alloc(self.elems());
        MetalBackend::transpose_head_to_token(
            &mut ctx,
            &src_buf,
            &mut dst,
            self.tokens,
            self.heads,
            self.dim,
        );
        MetalBackend::sync(&mut ctx);
        MetalBackend::to_vec(&dst, self.elems())
    }

    #[cfg(feature = "cuda")]
    fn run_cuda(&self, seed: u64) -> Output {
        use ferrum_kernels::backend::cuda::CudaBackend;
        use ferrum_kernels::backend::Backend;

        let src = random_vec(self.elems(), -2.0, 2.0, seed);
        let mut ctx = CudaBackend::new_context();
        let src_buf = CudaBackend::from_slice(&src);
        let mut dst = CudaBackend::alloc(self.elems());
        CudaBackend::transpose_head_to_token(
            &mut ctx,
            &src_buf,
            &mut dst,
            self.tokens,
            self.heads,
            self.dim,
        );
        CudaBackend::sync(&mut ctx);
        CudaBackend::to_vec(&dst, self.elems())
    }
}
