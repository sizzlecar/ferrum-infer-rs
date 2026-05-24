//! `qk_norm_rope` op-diff harness — covers the fused
//! rms_norm + rotary-position-embedding + head-major transpose used in
//! all transformer attention layers (`split_qkv_norm_rope_into_paged_cache_f16`
//! in nsys traces — top-10 kernel on M3).
//!
//! Layout:
//!   input  `[tokens, heads, head_dim]` — token-major
//!   norm_w `[head_dim]`
//!   cos    `[max_pos, head_dim/2]`
//!   sin    `[max_pos, head_dim/2]`
//!   output `[heads, tokens, head_dim]` — head-major after the fused
//!          transpose
//!
//! `mode = 1` exercises the actual RoPE pairs path; `mode = 0` is the
//! transpose-only fallback (which the harness can also test for fast
//! sanity).

use super::{random_vec, OpUnderTest, Output};

pub struct QkNormRopeOp {
    pub tokens: usize,
    pub heads: usize,
    pub head_dim: usize,
    pub pos_offset: usize,
    pub eps: f32,
    pub mode: i32,
}

impl QkNormRopeOp {
    fn max_pos(&self) -> usize {
        self.pos_offset + self.tokens + 16 // small safety margin
    }
    fn output_len(&self) -> usize {
        self.tokens * self.heads * self.head_dim
    }

    /// Inputs derived from seed:
    ///   x:     [-2, 2)
    ///   norm:  [0.5, 1.5)
    ///   cos/sin: precomputed RoPE rotation tables (theta_i = 10000^{-2i/d})
    fn build_input(&self, seed: u64) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
        let half = self.head_dim / 2;
        let x = random_vec(self.tokens * self.heads * self.head_dim, -2.0, 2.0, seed);
        let norm = random_vec(self.head_dim, 0.5, 1.5, seed.wrapping_add(1));

        let mut cos = Vec::with_capacity(self.max_pos() * half);
        let mut sin = Vec::with_capacity(self.max_pos() * half);
        for pos in 0..self.max_pos() {
            for i in 0..half {
                let theta = 10000f32.powf(-(i as f32) * 2.0 / self.head_dim as f32);
                let angle = pos as f32 * theta;
                cos.push(angle.cos());
                sin.push(angle.sin());
            }
        }
        (x, norm, cos, sin)
    }
}

impl OpUnderTest for QkNormRopeOp {
    fn name(&self) -> &str {
        "qk_norm_rope"
    }

    fn run_cpu(&self, seed: u64) -> Output {
        use ferrum_kernels::backend::cpu::CpuBackend;
        use ferrum_kernels::backend::Backend;
        let (x, w, cos, sin) = self.build_input(seed);
        let mut ctx = CpuBackend::new_context();
        let x_buf = CpuBackend::from_slice(&x);
        let w_buf = CpuBackend::from_slice(&w);
        let cos_buf = CpuBackend::from_slice(&cos);
        let sin_buf = CpuBackend::from_slice(&sin);
        let mut out = CpuBackend::alloc(self.output_len());
        CpuBackend::qk_norm_rope(
            &mut ctx,
            &x_buf,
            &w_buf,
            &cos_buf,
            &sin_buf,
            &mut out,
            self.tokens,
            self.heads,
            self.head_dim,
            self.pos_offset,
            self.eps,
            self.mode,
        );
        CpuBackend::sync(&mut ctx);
        CpuBackend::to_vec(&out, self.output_len())
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    fn run_metal(&self, seed: u64) -> Output {
        use ferrum_kernels::backend::metal::MetalBackend;
        use ferrum_kernels::backend::Backend;
        let (x, w, cos, sin) = self.build_input(seed);
        let mut ctx = MetalBackend::new_context();
        let x_buf = MetalBackend::from_slice(&x);
        let w_buf = MetalBackend::from_slice(&w);
        let cos_buf = MetalBackend::from_slice(&cos);
        let sin_buf = MetalBackend::from_slice(&sin);
        let mut out = MetalBackend::alloc(self.output_len());
        MetalBackend::qk_norm_rope(
            &mut ctx,
            &x_buf,
            &w_buf,
            &cos_buf,
            &sin_buf,
            &mut out,
            self.tokens,
            self.heads,
            self.head_dim,
            self.pos_offset,
            self.eps,
            self.mode,
        );
        MetalBackend::sync(&mut ctx);
        MetalBackend::to_vec(&out, self.output_len())
    }

    #[cfg(feature = "cuda")]
    fn run_cuda(&self, seed: u64) -> Output {
        use ferrum_kernels::backend::cuda::CudaBackend;
        use ferrum_kernels::backend::Backend;
        let (x, w, cos, sin) = self.build_input(seed);
        let mut ctx = CudaBackend::new_context();
        let x_buf = CudaBackend::from_slice(&x);
        let w_buf = CudaBackend::from_slice(&w);
        let cos_buf = CudaBackend::from_slice(&cos);
        let sin_buf = CudaBackend::from_slice(&sin);
        let mut out = CudaBackend::alloc(self.output_len());
        CudaBackend::qk_norm_rope(
            &mut ctx,
            &x_buf,
            &w_buf,
            &cos_buf,
            &sin_buf,
            &mut out,
            self.tokens,
            self.heads,
            self.head_dim,
            self.pos_offset,
            self.eps,
            self.mode,
        );
        CudaBackend::sync(&mut ctx);
        CudaBackend::to_vec(&out, self.output_len())
    }
}
