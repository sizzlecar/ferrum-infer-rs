//! `residual_add` (`add_inplace`) op-diff harness — see `crate::op_diff`.
//!
//! `residual += x`, elementwise. Output is the updated residual buffer.

use super::{random_vec, OpUnderTest, Output};

pub struct ResidualAddOp {
    pub len: usize,
}

impl ResidualAddOp {
    fn build_input(&self, seed: u64) -> (Vec<f32>, Vec<f32>) {
        let residual = random_vec(self.len, -2.0, 2.0, seed);
        let x = random_vec(self.len, -2.0, 2.0, seed.wrapping_add(1));
        (residual, x)
    }
}

impl OpUnderTest for ResidualAddOp {
    fn name(&self) -> &str {
        "residual_add"
    }

    fn run_cpu(&self, seed: u64) -> Output {
        use ferrum_kernels::backend::cpu::CpuBackend;
        use ferrum_kernels::backend::Backend;

        let (residual, x) = self.build_input(seed);
        let mut ctx = CpuBackend::new_context();
        let mut residual_buf = CpuBackend::from_slice(&residual);
        let x_buf = CpuBackend::from_slice(&x);
        CpuBackend::add_inplace(&mut ctx, &mut residual_buf, &x_buf, self.len);
        CpuBackend::sync(&mut ctx);
        CpuBackend::to_vec(&residual_buf, self.len)
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    fn run_metal(&self, seed: u64) -> Output {
        use ferrum_kernels::backend::metal::MetalBackend;
        use ferrum_kernels::backend::Backend;

        let (residual, x) = self.build_input(seed);
        let mut ctx = MetalBackend::new_context();
        let mut residual_buf = MetalBackend::from_slice(&residual);
        let x_buf = MetalBackend::from_slice(&x);
        MetalBackend::add_inplace(&mut ctx, &mut residual_buf, &x_buf, self.len);
        MetalBackend::sync(&mut ctx);
        MetalBackend::to_vec(&residual_buf, self.len)
    }

    #[cfg(feature = "cuda")]
    fn run_cuda(&self, seed: u64) -> Output {
        use ferrum_kernels::backend::cuda::CudaBackend;
        use ferrum_kernels::backend::Backend;

        let (residual, x) = self.build_input(seed);
        let mut ctx = CudaBackend::new_context();
        let mut residual_buf = CudaBackend::from_slice(&residual);
        let x_buf = CudaBackend::from_slice(&x);
        CudaBackend::add_inplace(&mut ctx, &mut residual_buf, &x_buf, self.len);
        CudaBackend::sync(&mut ctx);
        CudaBackend::to_vec(&residual_buf, self.len)
    }
}
