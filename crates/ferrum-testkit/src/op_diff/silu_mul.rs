//! `fused_silu_mul_split` op-diff harness.
//!
//! Input layout (matches the kernel API):
//!   - `gate_up`: tokens × (2 * intermediate)
//!   - For each token row: `[gate ‖ up]` concatenated
//! Output:
//!   - `out`: tokens × intermediate, where `out[i,j] = silu(gate[i,j]) * up[i,j]`

use super::{random_vec, OpUnderTest, Output};

pub struct SiluMulOp {
    pub tokens: usize,
    /// One side; the gate_up buffer is `tokens × (2*intermediate)`.
    pub intermediate: usize,
}

impl SiluMulOp {
    fn input_len(&self) -> usize {
        self.tokens * 2 * self.intermediate
    }
    fn output_len(&self) -> usize {
        self.tokens * self.intermediate
    }

    fn build_input(&self, seed: u64) -> Vec<f32> {
        random_vec(self.input_len(), -3.0, 3.0, seed)
    }
}

impl OpUnderTest for SiluMulOp {
    fn name(&self) -> &str {
        "fused_silu_mul"
    }

    fn run_cpu(&self, seed: u64) -> Output {
        use ferrum_kernels::backend::cpu::CpuBackend;
        use ferrum_kernels::backend::Backend;

        let gate_up = self.build_input(seed);
        let mut ctx = CpuBackend::new_context();
        let gu_buf = CpuBackend::from_slice(&gate_up);
        let mut out = CpuBackend::alloc(self.output_len());
        CpuBackend::fused_silu_mul_split(
            &mut ctx,
            &gu_buf,
            &mut out,
            self.tokens,
            self.intermediate,
        );
        CpuBackend::sync(&mut ctx);
        CpuBackend::to_vec(&out, self.output_len())
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    fn run_metal(&self, seed: u64) -> Output {
        use ferrum_kernels::backend::metal::MetalBackend;
        use ferrum_kernels::backend::Backend;

        let gate_up = self.build_input(seed);
        let mut ctx = MetalBackend::new_context();
        let gu_buf = MetalBackend::from_slice(&gate_up);
        let mut out = MetalBackend::alloc(self.output_len());
        MetalBackend::fused_silu_mul_split(
            &mut ctx,
            &gu_buf,
            &mut out,
            self.tokens,
            self.intermediate,
        );
        MetalBackend::sync(&mut ctx);
        MetalBackend::to_vec(&out, self.output_len())
    }

    #[cfg(feature = "cuda")]
    fn run_cuda(&self, seed: u64) -> Output {
        use ferrum_kernels::backend::cuda::CudaBackend;
        use ferrum_kernels::backend::Backend;

        let gate_up = self.build_input(seed);
        let mut ctx = CudaBackend::new_context();
        let gu_buf = CudaBackend::from_slice(&gate_up);
        let mut out = CudaBackend::alloc(self.output_len());
        CudaBackend::fused_silu_mul_split(
            &mut ctx,
            &gu_buf,
            &mut out,
            self.tokens,
            self.intermediate,
        );
        CudaBackend::sync(&mut ctx);
        CudaBackend::to_vec(&out, self.output_len())
    }
}
