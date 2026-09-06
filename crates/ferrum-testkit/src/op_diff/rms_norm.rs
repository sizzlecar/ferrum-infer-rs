//! `rms_norm` op-diff harness — see `crate::op_diff` for the framework.

use super::{random_vec, OpUnderTest, Output};

/// One concrete rms_norm invocation. Inputs:
///   - `x`: tokens × dim activation
///   - `w`: dim weight (per-channel scale)
///   - `eps`: the usual RMSNorm epsilon
///
/// Output: tokens × dim. This fixture uses F32 buffers on CPU and Metal;
/// CUDA's existing default buffers are F16. Host results are returned as f32.
pub struct RmsNormOp {
    pub tokens: usize,
    pub dim: usize,
    pub eps: f32,
}

impl RmsNormOp {
    /// Validate the shared fixture before allocation or backend submission.
    pub fn validate(&self) -> Result<(), String> {
        self.expected_output_len().map(|_| ())
    }

    /// Checked host result size, including the kernels' signed 32-bit row and
    /// column indices and the final column-loop increment. These are kernel/host
    /// limits, not a promise that the allocation fits available device memory.
    pub fn expected_output_len(&self) -> Result<usize, String> {
        if self.tokens == 0 || self.dim == 0 {
            return Err("RMSNorm tokens and dim must be nonzero".into());
        }
        if !self.eps.is_finite() || self.eps <= 0.0 {
            return Err("RMSNorm epsilon must be finite and positive".into());
        }
        let elements = self
            .tokens
            .checked_mul(self.dim)
            .ok_or("RMSNorm tokens * dim overflows usize")?;
        let bytes = elements
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or("RMSNorm f32 buffer byte size overflows usize")?;
        if bytes > isize::MAX as usize {
            return Err("RMSNorm f32 buffer exceeds the host allocation address range".into());
        }
        // Metal advances columns by 32. CUDA's step is its block width, capped
        // at 1024; rounding to a complete warp also covers the current unrounded
        // launch. Even the thread processing column dim - 1 must safely perform
        // its final increment before evaluating the loop condition again.
        let cuda_step = self.dim.min(1024).div_ceil(32) * 32;
        let max_loop_step = cuda_step.max(32);
        let final_increment = (self.dim - 1).checked_add(max_loop_step);
        if elements > i32::MAX as usize
            || final_increment.is_none_or(|index| index > i32::MAX as usize)
        {
            return Err("RMSNorm shape exceeds the kernels' signed 32-bit indexing range".into());
        }
        Ok(elements)
    }

    fn output_len(&self) -> usize {
        self.expected_output_len().expect("invalid RMSNorm fixture")
    }

    /// Inputs are derived from seed so per-backend runs see identical x/w.
    fn build_input(&self, seed: u64) -> (Vec<f32>, Vec<f32>) {
        let x = random_vec(self.output_len(), -2.0, 2.0, seed);
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
        MetalBackend::sync_checked(&mut ctx)
            .unwrap_or_else(|error| panic!("RMSNorm Metal completion failed: {error}"));
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

#[cfg(test)]
mod tests {
    use super::*;

    fn operation(tokens: usize, dim: usize, eps: f32) -> RmsNormOp {
        RmsNormOp { tokens, dim, eps }
    }

    #[test]
    fn validates_result_shape_before_allocating_inputs() {
        let op = operation(3, 33, 1e-6);
        op.validate().unwrap();
        assert_eq!(op.expected_output_len().unwrap(), 99);
        let output = op.run_cpu(7);
        assert_eq!(output.len(), op.expected_output_len().unwrap());
        assert!(output.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn rejects_empty_shapes_and_nonpositive_or_nonfinite_epsilon() {
        for (tokens, dim) in [(0, 32), (1, 0), (0, 0)] {
            assert!(operation(tokens, dim, 1e-6).validate().is_err());
        }
        for eps in [0.0, -0.0, -1e-6, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert!(operation(1, 32, eps).validate().is_err());
        }
    }

    #[test]
    fn rejects_shape_byte_and_kernel_index_overflow_without_allocating() {
        for (tokens, dim) in [
            (usize::MAX, 2),
            (1, usize::MAX / std::mem::size_of::<f32>() + 1),
            (1, i32::MAX as usize),
            (2, i32::MAX as usize / 2 + 1),
        ] {
            assert!(operation(tokens, dim, 1e-6).expected_output_len().is_err());
        }
    }

    #[test]
    #[cfg(target_pointer_width = "64")]
    fn validates_final_cuda_loop_increment_without_allocating() {
        let last_safe_dim = i32::MAX as usize - (1024 - 1);
        assert_eq!(
            operation(1, last_safe_dim, 1e-6).expected_output_len(),
            Ok(last_safe_dim)
        );
        // These shapes fit signed element indexing and host address space, but
        // a CUDA thread's last 1024-wide increment exceeds the signed range.
        for dim in [last_safe_dim + 1, i32::MAX as usize - 31] {
            assert!(operation(1, dim, 1e-6).expected_output_len().is_err());
        }
    }

    #[test]
    fn direct_executor_use_rejects_invalid_fixture_before_allocation() {
        let op = operation(usize::MAX, 2, 1e-6);
        assert!(std::panic::catch_unwind(|| op.run_cpu(7)).is_err());
    }
}
