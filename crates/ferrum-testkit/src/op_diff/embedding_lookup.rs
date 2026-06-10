//! `embedding_lookup` op-diff harness — see `crate::op_diff`.
//!
//! Gather `n` rows from a `[vocab, dim]` table by token id. Exact (no
//! arithmetic), so accelerator NMSE should be ~0.

use super::{random_vec, OpUnderTest, Output};

pub struct EmbeddingLookupOp {
    pub vocab: usize,
    pub dim: usize,
    pub tokens: usize,
}

impl EmbeddingLookupOp {
    fn build_input(&self, seed: u64) -> (Vec<f32>, Vec<u32>) {
        let table = random_vec(self.vocab * self.dim, -1.0, 1.0, seed);
        // Deterministic ids in [0, vocab) without rand: hash of seed+index.
        let ids: Vec<u32> = (0..self.tokens)
            .map(|i| {
                let h = (seed.wrapping_add(i as u64).wrapping_mul(2654435761)) as u32;
                h % self.vocab as u32
            })
            .collect();
        (table, ids)
    }
}

impl OpUnderTest for EmbeddingLookupOp {
    fn name(&self) -> &str {
        "embedding_lookup"
    }

    fn run_cpu(&self, seed: u64) -> Output {
        use ferrum_kernels::backend::cpu::CpuBackend;
        use ferrum_kernels::backend::Backend;

        let (table, ids) = self.build_input(seed);
        let mut ctx = CpuBackend::new_context();
        let table_buf = CpuBackend::from_slice(&table);
        let mut out = CpuBackend::alloc(self.tokens * self.dim);
        CpuBackend::embedding_lookup(&mut ctx, &table_buf, &ids, &mut out, self.dim);
        CpuBackend::sync(&mut ctx);
        CpuBackend::to_vec(&out, self.tokens * self.dim)
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    fn run_metal(&self, seed: u64) -> Output {
        use ferrum_kernels::backend::metal::MetalBackend;
        use ferrum_kernels::backend::Backend;

        let (table, ids) = self.build_input(seed);
        let mut ctx = MetalBackend::new_context();
        let table_buf = MetalBackend::from_slice(&table);
        let mut out = MetalBackend::alloc(self.tokens * self.dim);
        MetalBackend::embedding_lookup(&mut ctx, &table_buf, &ids, &mut out, self.dim);
        MetalBackend::sync(&mut ctx);
        MetalBackend::to_vec(&out, self.tokens * self.dim)
    }

    #[cfg(feature = "cuda")]
    fn run_cuda(&self, seed: u64) -> Output {
        use ferrum_kernels::backend::cuda::CudaBackend;
        use ferrum_kernels::backend::Backend;

        let (table, ids) = self.build_input(seed);
        let mut ctx = CudaBackend::new_context();
        let table_buf = CudaBackend::from_slice(&table);
        let mut out = CudaBackend::alloc(self.tokens * self.dim);
        CudaBackend::embedding_lookup(&mut ctx, &table_buf, &ids, &mut out, self.dim);
        CudaBackend::sync(&mut ctx);
        CudaBackend::to_vec(&out, self.tokens * self.dim)
    }
}
