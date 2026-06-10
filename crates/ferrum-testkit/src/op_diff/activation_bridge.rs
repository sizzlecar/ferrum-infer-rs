//! `activation_bridge` op-diff harness — see `crate::op_diff`.
//!
//! Exercises the buffer-plumbing path each backend uses to move activations
//! host<->device: `from_slice` upload, `copy_slice` device-to-device, and
//! `to_vec` readback. Compared at the fp16-storage tolerance since Metal/CUDA
//! store activations as f16.

use super::{random_vec, OpUnderTest, Output};

pub struct ActivationBridgeOp {
    pub len: usize,
}

macro_rules! run_backend {
    ($B:ty, $self:expr, $seed:expr) => {{
        use ferrum_kernels::backend::Backend;
        let data = random_vec($self.len, -2.0, 2.0, $seed);
        let mut ctx = <$B>::new_context();
        let src = <$B>::from_slice(&data);
        let mut dst = <$B>::alloc($self.len);
        <$B>::copy_slice(&mut ctx, &src, 0, &mut dst, 0, $self.len);
        <$B>::sync(&mut ctx);
        <$B>::to_vec(&dst, $self.len)
    }};
}

impl OpUnderTest for ActivationBridgeOp {
    fn name(&self) -> &str {
        "activation_bridge"
    }

    fn run_cpu(&self, seed: u64) -> Output {
        run_backend!(ferrum_kernels::backend::cpu::CpuBackend, self, seed)
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    fn run_metal(&self, seed: u64) -> Output {
        run_backend!(ferrum_kernels::backend::metal::MetalBackend, self, seed)
    }

    #[cfg(feature = "cuda")]
    fn run_cuda(&self, seed: u64) -> Output {
        run_backend!(ferrum_kernels::backend::cuda::CudaBackend, self, seed)
    }
}
