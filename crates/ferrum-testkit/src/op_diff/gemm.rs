//! `gemm` op-diff harness — covers the basic fp16 matmul that backs
//! `qkv_proj`, `o_proj`, `gate_up_proj`, `down_proj`, and the lm_head
//! projection. Per nsys profile on Vast 4090 / M3, `Marlin<256,...>`
//! Marlin matmul accounts for ~55% of GPU time at c=16; this op-diff
//! validates the non-quantized fallback path against CPU.

use super::{random_vec, OpUnderTest, Output};
use ferrum_kernels::backend::Backend;

/// `C[m, n] = A[m, k] · B[n, k]^T` (row-major, B already transposed
/// to head-major). Matches the Backend::gemm signature used by Linear.
pub struct GemmOp {
    pub m: usize,
    pub n: usize,
    pub k: usize,
}

impl GemmOp {
    fn input_a_len(&self) -> usize { self.m * self.k }
    fn input_b_len(&self) -> usize { self.n * self.k }
    fn output_len(&self) -> usize { self.m * self.n }

    fn build_input(&self, seed: u64) -> (Vec<f32>, Vec<f32>) {
        let a = random_vec(self.input_a_len(), -1.0, 1.0, seed);
        let b = random_vec(self.input_b_len(), -1.0, 1.0, seed.wrapping_add(1));
        (a, b)
    }
}

impl OpUnderTest for GemmOp {
    fn name(&self) -> &str { "gemm" }

    fn run_cpu(&self, seed: u64) -> Output {
        use ferrum_kernels::backend::cpu::CpuBackend;
        let (a, b) = self.build_input(seed);
        let mut ctx = CpuBackend::new_context();
        let a_buf = CpuBackend::from_slice(&a);
        let b_buf = CpuBackend::from_slice(&b);
        let mut out = CpuBackend::alloc(self.output_len());
        CpuBackend::gemm(&mut ctx, &a_buf, &b_buf, &mut out, self.m, self.n, self.k);
        CpuBackend::sync(&mut ctx);
        CpuBackend::to_vec(&out, self.output_len())
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    fn run_metal(&self, seed: u64) -> Output {
        use ferrum_kernels::backend::metal::MetalBackend;
        let (a, b) = self.build_input(seed);
        let mut ctx = MetalBackend::new_context();
        let a_buf = MetalBackend::from_slice(&a);
        let b_buf = MetalBackend::from_slice(&b);
        let mut out = MetalBackend::alloc(self.output_len());
        MetalBackend::gemm(&mut ctx, &a_buf, &b_buf, &mut out, self.m, self.n, self.k);
        MetalBackend::sync(&mut ctx);
        MetalBackend::to_vec(&out, self.output_len())
    }

    #[cfg(feature = "cuda")]
    fn run_cuda(&self, seed: u64) -> Output {
        use ferrum_kernels::backend::cuda::CudaBackend;
        let (a, b) = self.build_input(seed);
        let mut ctx = CudaBackend::new_context();
        let a_buf = CudaBackend::from_slice(&a);
        let b_buf = CudaBackend::from_slice(&b);
        let mut out = CudaBackend::alloc(self.output_len());
        CudaBackend::gemm(&mut ctx, &a_buf, &b_buf, &mut out, self.m, self.n, self.k);
        CudaBackend::sync(&mut ctx);
        CudaBackend::to_vec(&out, self.output_len())
    }
}
