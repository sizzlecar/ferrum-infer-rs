//! `fused_add_rms_norm` op-diff harness — see `crate::op_diff`.
//!
//! Fused `residual += x; out = rms_norm(residual, w)`. The output compared
//! across backends is `[residual_after, out]` concatenated, so a divergence
//! in either the in-place residual update or the norm is caught.

use super::{random_vec, OpUnderTest, Output};

pub struct FusedAddRmsNormOp {
    pub tokens: usize,
    pub dim: usize,
    pub eps: f32,
}

impl FusedAddRmsNormOp {
    fn elems(&self) -> usize {
        self.tokens * self.dim
    }

    fn build_input(&self, seed: u64) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let residual = random_vec(self.elems(), -2.0, 2.0, seed);
        let x = random_vec(self.elems(), -2.0, 2.0, seed.wrapping_add(1));
        let w = random_vec(self.dim, 0.5, 1.5, seed.wrapping_add(2));
        (residual, x, w)
    }
}

impl OpUnderTest for FusedAddRmsNormOp {
    fn name(&self) -> &str {
        "fused_add_rms_norm"
    }

    fn run_cpu(&self, seed: u64) -> Output {
        use ferrum_kernels::backend::cpu::CpuBackend;
        use ferrum_kernels::backend::Backend;

        let (residual, x, w) = self.build_input(seed);
        let mut ctx = CpuBackend::new_context();
        let mut residual_buf = CpuBackend::from_slice(&residual);
        let x_buf = CpuBackend::from_slice(&x);
        let w_buf = CpuBackend::from_slice(&w);
        let mut out = CpuBackend::alloc(self.elems());
        CpuBackend::fused_add_rms_norm(
            &mut ctx,
            &mut residual_buf,
            &x_buf,
            &w_buf,
            self.eps,
            &mut out,
            self.tokens,
            self.dim,
        );
        CpuBackend::sync(&mut ctx);
        let mut combined = CpuBackend::to_vec(&residual_buf, self.elems());
        combined.extend(CpuBackend::to_vec(&out, self.elems()));
        combined
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    fn run_metal(&self, seed: u64) -> Output {
        use ferrum_kernels::backend::metal::MetalBackend;
        use ferrum_kernels::backend::Backend;

        let (residual, x, w) = self.build_input(seed);
        let mut ctx = MetalBackend::new_context();
        let mut residual_buf = MetalBackend::from_slice(&residual);
        let x_buf = MetalBackend::from_slice(&x);
        let w_buf = MetalBackend::from_slice(&w);
        let mut out = MetalBackend::alloc(self.elems());
        MetalBackend::fused_add_rms_norm(
            &mut ctx,
            &mut residual_buf,
            &x_buf,
            &w_buf,
            self.eps,
            &mut out,
            self.tokens,
            self.dim,
        );
        MetalBackend::sync(&mut ctx);
        let mut combined = MetalBackend::to_vec(&residual_buf, self.elems());
        combined.extend(MetalBackend::to_vec(&out, self.elems()));
        combined
    }

    #[cfg(feature = "cuda")]
    fn run_cuda(&self, seed: u64) -> Output {
        use ferrum_kernels::backend::cuda::CudaBackend;
        use ferrum_kernels::backend::Backend;

        let (residual, x, w) = self.build_input(seed);
        let mut ctx = CudaBackend::new_context();
        let mut residual_buf = CudaBackend::from_slice(&residual);
        let x_buf = CudaBackend::from_slice(&x);
        let w_buf = CudaBackend::from_slice(&w);
        let mut out = CudaBackend::alloc(self.elems());
        CudaBackend::fused_add_rms_norm(
            &mut ctx,
            &mut residual_buf,
            &x_buf,
            &w_buf,
            self.eps,
            &mut out,
            self.tokens,
            self.dim,
        );
        CudaBackend::sync(&mut ctx);
        let mut combined = CudaBackend::to_vec(&residual_buf, self.elems());
        combined.extend(CudaBackend::to_vec(&out, self.elems()));
        combined
    }
}
