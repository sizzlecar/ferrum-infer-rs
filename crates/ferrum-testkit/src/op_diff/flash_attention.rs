//! `flash_attention` op-diff harness — see `crate::op_diff`.
//!
//! Dense causal attention. CpuBackend implements the reference; Metal/CUDA
//! run their flash kernels against the same Q/K/V.

use super::{random_vec, OpUnderTest, Output};
use ferrum_kernels::backend::AttnConfig;

pub struct FlashAttentionOp {
    pub batch: usize,
    pub q_len: usize,
    pub kv_len: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
}

impl FlashAttentionOp {
    fn q_elems(&self) -> usize {
        self.batch * self.q_len * self.num_heads * self.head_dim
    }
    fn kv_elems(&self) -> usize {
        self.batch * self.kv_len * self.num_kv_heads * self.head_dim
    }
    fn cfg(&self) -> AttnConfig {
        AttnConfig {
            num_heads: self.num_heads,
            num_kv_heads: self.num_kv_heads,
            head_dim: self.head_dim,
            causal: true,
            scale: 1.0 / (self.head_dim as f32).sqrt(),
            kv_seq_stride: 0,
            sliding_window: 0,
        }
    }
}

macro_rules! run_backend {
    ($B:ty, $self:expr, $seed:expr) => {{
        use ferrum_kernels::backend::Backend;
        let q = random_vec($self.q_elems(), -1.0, 1.0, $seed);
        let k = random_vec($self.kv_elems(), -1.0, 1.0, $seed.wrapping_add(1));
        let v = random_vec($self.kv_elems(), -1.0, 1.0, $seed.wrapping_add(2));
        let mut ctx = <$B>::new_context();
        let qb = <$B>::from_slice(&q);
        let kb = <$B>::from_slice(&k);
        let vb = <$B>::from_slice(&v);
        let mut out = <$B>::alloc($self.q_elems());
        <$B>::flash_attention(
            &mut ctx,
            &qb,
            &kb,
            &vb,
            &mut out,
            $self.batch,
            $self.q_len,
            $self.kv_len,
            0,
            &$self.cfg(),
        );
        <$B>::sync(&mut ctx);
        <$B>::to_vec(&out, $self.q_elems())
    }};
}

impl OpUnderTest for FlashAttentionOp {
    fn name(&self) -> &str {
        "flash_attention"
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
