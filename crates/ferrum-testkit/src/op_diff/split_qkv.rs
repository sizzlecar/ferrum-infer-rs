//! `split_qkv` op-diff harness — see `crate::op_diff`.
//!
//! Splits a fused `[tokens, q_dim + 2*kv_dim]` projection into separate q/k/v
//! buffers. Pure slicing (exact); compared output is `[q, k, v]` concatenated.

use super::{random_vec, OpUnderTest, Output};

pub struct SplitQkvOp {
    pub tokens: usize,
    pub q_dim: usize,
    pub kv_dim: usize,
}

impl SplitQkvOp {
    fn fused(&self) -> usize {
        self.tokens * (self.q_dim + 2 * self.kv_dim)
    }
}

impl OpUnderTest for SplitQkvOp {
    fn name(&self) -> &str {
        "split_qkv"
    }

    fn run_cpu(&self, seed: u64) -> Output {
        use ferrum_kernels::backend::cpu::CpuBackend;
        use ferrum_kernels::backend::Backend;

        let qkv = random_vec(self.fused(), -2.0, 2.0, seed);
        let mut ctx = CpuBackend::new_context();
        let qkv_buf = CpuBackend::from_slice(&qkv);
        let mut q = CpuBackend::alloc(self.tokens * self.q_dim);
        let mut k = CpuBackend::alloc(self.tokens * self.kv_dim);
        let mut v = CpuBackend::alloc(self.tokens * self.kv_dim);
        CpuBackend::split_qkv(
            &mut ctx,
            &qkv_buf,
            &mut q,
            &mut k,
            &mut v,
            self.tokens,
            self.q_dim,
            self.kv_dim,
        );
        CpuBackend::sync(&mut ctx);
        let mut out = CpuBackend::to_vec(&q, self.tokens * self.q_dim);
        out.extend(CpuBackend::to_vec(&k, self.tokens * self.kv_dim));
        out.extend(CpuBackend::to_vec(&v, self.tokens * self.kv_dim));
        out
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    fn run_metal(&self, seed: u64) -> Output {
        use ferrum_kernels::backend::metal::MetalBackend;
        use ferrum_kernels::backend::Backend;

        let qkv = random_vec(self.fused(), -2.0, 2.0, seed);
        let mut ctx = MetalBackend::new_context();
        let qkv_buf = MetalBackend::from_slice(&qkv);
        let mut q = MetalBackend::alloc(self.tokens * self.q_dim);
        let mut k = MetalBackend::alloc(self.tokens * self.kv_dim);
        let mut v = MetalBackend::alloc(self.tokens * self.kv_dim);
        MetalBackend::split_qkv(
            &mut ctx,
            &qkv_buf,
            &mut q,
            &mut k,
            &mut v,
            self.tokens,
            self.q_dim,
            self.kv_dim,
        );
        MetalBackend::sync(&mut ctx);
        let mut out = MetalBackend::to_vec(&q, self.tokens * self.q_dim);
        out.extend(MetalBackend::to_vec(&k, self.tokens * self.kv_dim));
        out.extend(MetalBackend::to_vec(&v, self.tokens * self.kv_dim));
        out
    }

    #[cfg(feature = "cuda")]
    fn run_cuda(&self, seed: u64) -> Output {
        use ferrum_kernels::backend::cuda::CudaBackend;
        use ferrum_kernels::backend::Backend;

        let qkv = random_vec(self.fused(), -2.0, 2.0, seed);
        let mut ctx = CudaBackend::new_context();
        let qkv_buf = CudaBackend::from_slice(&qkv);
        let mut q = CudaBackend::alloc(self.tokens * self.q_dim);
        let mut k = CudaBackend::alloc(self.tokens * self.kv_dim);
        let mut v = CudaBackend::alloc(self.tokens * self.kv_dim);
        CudaBackend::split_qkv(
            &mut ctx,
            &qkv_buf,
            &mut q,
            &mut k,
            &mut v,
            self.tokens,
            self.q_dim,
            self.kv_dim,
        );
        CudaBackend::sync(&mut ctx);
        let mut out = CudaBackend::to_vec(&q, self.tokens * self.q_dim);
        out.extend(CudaBackend::to_vec(&k, self.tokens * self.kv_dim));
        out.extend(CudaBackend::to_vec(&v, self.tokens * self.kv_dim));
        out
    }
}
