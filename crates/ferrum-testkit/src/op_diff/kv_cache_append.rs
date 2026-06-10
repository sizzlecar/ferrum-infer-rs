//! `kv_cache_append_head_major` op-diff harness — see `crate::op_diff`.
//!
//! Appends `new_tokens` head-major K/V vectors into a pre-filled cache at
//! position `cache_len`. Pure data movement (exact); the compared output is
//! `[cache_k, cache_v]` after the append.

use super::{random_vec, OpUnderTest, Output};

pub struct KvCacheAppendOp {
    pub nkv: usize,
    pub hd: usize,
    pub capacity: usize,
    pub cache_len: usize,
    pub new_tokens: usize,
}

impl KvCacheAppendOp {
    fn cache_elems(&self) -> usize {
        self.nkv * self.capacity * self.hd
    }
    fn new_elems(&self) -> usize {
        self.nkv * self.new_tokens * self.hd
    }
}

macro_rules! run_backend {
    ($B:ty, $self:expr, $seed:expr) => {{
        use ferrum_kernels::backend::Backend;
        let ck = random_vec($self.cache_elems(), -1.0, 1.0, $seed);
        let cv = random_vec($self.cache_elems(), -1.0, 1.0, $seed.wrapping_add(1));
        let nk = random_vec($self.new_elems(), -2.0, 2.0, $seed.wrapping_add(2));
        let nv = random_vec($self.new_elems(), -2.0, 2.0, $seed.wrapping_add(3));
        let mut ctx = <$B>::new_context();
        let mut cache_k = <$B>::from_slice(&ck);
        let mut cache_v = <$B>::from_slice(&cv);
        let new_k = <$B>::from_slice(&nk);
        let new_v = <$B>::from_slice(&nv);
        <$B>::kv_cache_append_head_major(
            &mut ctx,
            &mut cache_k,
            &mut cache_v,
            $self.cache_len,
            $self.capacity,
            &new_k,
            &new_v,
            $self.new_tokens,
            $self.nkv,
            $self.hd,
        );
        <$B>::sync(&mut ctx);
        let mut out = <$B>::to_vec(&cache_k, $self.cache_elems());
        out.extend(<$B>::to_vec(&cache_v, $self.cache_elems()));
        out
    }};
}

impl OpUnderTest for KvCacheAppendOp {
    fn name(&self) -> &str {
        "kv_cache_append"
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
