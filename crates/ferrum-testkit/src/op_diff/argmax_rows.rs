//! `argmax_rows_f16` op-diff harness — see `crate::op_diff`.
//!
//! Per-row argmax over an `[m, n]` logits buffer. Metal stores logits as f16,
//! so to keep the argmax unambiguous across the f32 reference and the f16
//! kernel, each row gets a well-separated spike at a deterministic column —
//! both backends must select it regardless of f16 rounding. The compared
//! output is the m winning indices (as f32).

use super::{random_vec, OpUnderTest, Output};

pub struct ArgmaxRowsOp {
    pub m: usize,
    pub n: usize,
}

impl ArgmaxRowsOp {
    fn build_input(&self, seed: u64) -> (Vec<f32>, Vec<u32>) {
        let mut logits = random_vec(self.m * self.n, -1.0, 1.0, seed);
        let mut expected = Vec::with_capacity(self.m);
        for row in 0..self.m {
            let col = ((seed as usize).wrapping_add(row.wrapping_mul(7))) % self.n;
            logits[row * self.n + col] = 100.0; // unambiguous spike
            expected.push(col as u32);
        }
        (logits, expected)
    }
}

impl OpUnderTest for ArgmaxRowsOp {
    fn name(&self) -> &str {
        "argmax_rows_f16"
    }

    fn run_cpu(&self, seed: u64) -> Output {
        use ferrum_kernels::backend::cpu::CpuBackend;
        use ferrum_kernels::backend::Backend;

        let (logits, _) = self.build_input(seed);
        let mut ctx = CpuBackend::new_context();
        let buf = CpuBackend::from_slice(&logits);
        let idx = CpuBackend::argmax_rows_f16(&mut ctx, &buf, self.m, self.n)
            .expect("cpu argmax_rows_f16");
        idx.into_iter().map(|i| i as f32).collect()
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    fn run_metal(&self, seed: u64) -> Output {
        use ferrum_kernels::backend::metal::MetalBackend;
        use ferrum_kernels::backend::Backend;

        let (logits, _) = self.build_input(seed);
        let mut ctx = MetalBackend::new_context();
        let buf = MetalBackend::from_slice(&logits);
        let idx = MetalBackend::argmax_rows_f16(&mut ctx, &buf, self.m, self.n)
            .expect("metal argmax_rows_f16");
        idx.into_iter().map(|i| i as f32).collect()
    }

    #[cfg(feature = "cuda")]
    fn run_cuda(&self, seed: u64) -> Output {
        use ferrum_kernels::backend::cuda::CudaBackend;
        use ferrum_kernels::backend::Backend;

        let (logits, _) = self.build_input(seed);
        let mut ctx = CudaBackend::new_context();
        let buf = CudaBackend::from_slice(&logits);
        let idx = CudaBackend::argmax_rows_f16(&mut ctx, &buf, self.m, self.n)
            .expect("cuda argmax_rows_f16");
        idx.into_iter().map(|i| i as f32).collect()
    }
}
