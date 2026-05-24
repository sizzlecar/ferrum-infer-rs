//! `rms_norm` op-diff harness — see `crate::op_diff` for the framework.

use super::{random_vec, OpUnderTest, Output};

/// One concrete rms_norm invocation. Inputs:
///   - `x`: tokens × dim activation
///   - `w`: dim weight (per-channel scale)
///   - `eps`: the usual RMSNorm epsilon
///
/// Output: tokens × dim, same dtype as input on every backend's compute
/// dtype (typically fp16 on Metal/CUDA, fp32 on CPU).
pub struct RmsNormOp {
    pub tokens: usize,
    pub dim: usize,
    pub eps: f32,
}

impl RmsNormOp {
    fn output_len(&self) -> usize {
        self.tokens * self.dim
    }

    /// Inputs are derived from seed so per-backend runs see identical x/w.
    fn build_input(&self, seed: u64) -> (Vec<f32>, Vec<f32>) {
        let x = random_vec(self.tokens * self.dim, -2.0, 2.0, seed);
        let w = random_vec(self.dim, 0.5, 1.5, seed.wrapping_add(1));
        (x, w)
    }
}

impl OpUnderTest for RmsNormOp {
    fn name(&self) -> &str {
        "rms_norm"
    }

    fn run_cpu(&self, seed: u64) -> Output {
        use ferrum_kernels::backend::cpu::CpuBackend;
        use ferrum_kernels::backend::Backend;

        let (x, w) = self.build_input(seed);
        let mut ctx = CpuBackend::new_context();
        let x_buf = CpuBackend::from_slice(&x);
        let w_buf = CpuBackend::from_slice(&w);
        let mut out = CpuBackend::alloc(self.output_len());
        CpuBackend::rms_norm(
            &mut ctx,
            &x_buf,
            &w_buf,
            self.eps,
            &mut out,
            self.tokens,
            self.dim,
        );
        CpuBackend::sync(&mut ctx);
        CpuBackend::to_vec(&out, self.output_len())
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    fn run_metal(&self, seed: u64) -> Output {
        use ferrum_kernels::backend::metal::MetalBackend;
        use ferrum_kernels::backend::Backend;

        let (x, w) = self.build_input(seed);
        let mut ctx = MetalBackend::new_context();
        let x_buf = MetalBackend::from_slice(&x);
        let w_buf = MetalBackend::from_slice(&w);
        let mut out = MetalBackend::alloc(self.output_len());
        MetalBackend::rms_norm(
            &mut ctx,
            &x_buf,
            &w_buf,
            self.eps,
            &mut out,
            self.tokens,
            self.dim,
        );
        MetalBackend::sync(&mut ctx);
        MetalBackend::to_vec(&out, self.output_len())
    }

    #[cfg(feature = "cuda")]
    fn run_cuda(&self, seed: u64) -> Output {
        use ferrum_kernels::backend::cuda::CudaBackend;
        use ferrum_kernels::backend::Backend;

        let (x, w) = self.build_input(seed);
        let mut ctx = CudaBackend::new_context();
        let x_buf = CudaBackend::from_slice(&x);
        let w_buf = CudaBackend::from_slice(&w);
        let mut out = CudaBackend::alloc(self.output_len());
        CudaBackend::rms_norm(
            &mut ctx,
            &x_buf,
            &w_buf,
            self.eps,
            &mut out,
            self.tokens,
            self.dim,
        );
        CudaBackend::sync(&mut ctx);
        CudaBackend::to_vec(&out, self.output_len())
    }
}
