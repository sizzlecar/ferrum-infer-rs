//! Non-quantized `Backend::gemm` comparison against CPU F32.
//! Metal uses F32 buffers (GEMV for m=1, tiled GEMM otherwise); CUDA
//! uses F16 input/output buffers and cuBLAS F32 accumulation. This fixture
//! does not exercise quantized Marlin or the production plan runtime.

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
    /// Validate buffer sizes and the signed indices used by Metal/cuBLAS.
    /// These are arithmetic limits, not a guarantee of available device memory.
    pub fn expected_output_len(&self) -> Result<usize, String> {
        if self.m == 0 || self.n == 0 || self.k == 0 {
            return Err("GEMM m, n and k must be nonzero".into());
        }
        let mut output = 0;
        for (name, rows, cols) in [
            ("A", self.m, self.k),
            ("B", self.n, self.k),
            ("output", self.m, self.n),
        ] {
            let elements = rows
                .checked_mul(cols)
                .ok_or_else(|| format!("GEMM {name} element count overflows usize"))?;
            let bytes = elements
                .checked_mul(std::mem::size_of::<f32>())
                .ok_or_else(|| format!("GEMM {name} f32 byte size overflows usize"))?;
            if bytes > isize::MAX as usize || elements > i32::MAX as usize {
                return Err(format!(
                    "GEMM {name} exceeds host or signed kernel indexing range"
                ));
            }
            output = elements;
        }
        // The Metal kernel adds tile offsets before checking edge coordinates,
        // and increments K by 32. Include that final increment, also for GEMV.
        for (name, dimension, tile) in [("m", self.m, 64), ("n", self.n, 32), ("k", self.k, 32)] {
            let padded = dimension.div_ceil(tile).checked_mul(tile);
            if padded.is_none_or(|value| value > i32::MAX as usize) {
                return Err(format!(
                    "GEMM {name} tile exceeds signed kernel indexing range"
                ));
            }
        }
        Ok(output)
    }

    fn output_len(&self) -> usize {
        self.expected_output_len().expect("invalid GEMM fixture")
    }

    fn build_input(&self, seed: u64) -> (Vec<f32>, Vec<f32>) {
        self.output_len(); // Reject invalid direct fixture use before allocation.
        let a = random_vec(self.m * self.k, -1.0, 1.0, seed);
        let b = random_vec(self.n * self.k, -1.0, 1.0, seed.wrapping_add(1));
        (a, b)
    }
}

impl OpUnderTest for GemmOp {
    fn name(&self) -> &str {
        "gemm"
    }

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
        MetalBackend::sync_checked(&mut ctx)
            .unwrap_or_else(|error| panic!("GEMM Metal completion failed: {error}"));
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::op_diff::{required::compare_outputs, NMSE_FP32_TOL};

    #[test]
    fn validates_gemm_shapes_and_compares_non_square_layout() {
        for (m, n, k) in [(64, 32, 32), (64, 33, 35), (1, 3, 5)] {
            let op = GemmOp { m, n, k };
            assert_eq!(op.expected_output_len(), Ok(m * n));
            let (a, b) = op.build_input(17);
            let actual = op.run_cpu(17);
            let expected: Vec<f32> = a
                .chunks_exact(k)
                .flat_map(|row| {
                    b.chunks_exact(k)
                        .map(move |column| row.iter().zip(column).map(|(x, y)| x * y).sum())
                })
                .collect();
            compare_outputs(&expected, &actual, NMSE_FP32_TOL).unwrap();
            // A missing dispatch/zero-filled output must fail this same oracle.
            assert!(compare_outputs(&expected, &vec![0.0; m * n], NMSE_FP32_TOL).is_err());
        }
    }

    #[test]
    fn rejects_gemm_empty_product_byte_and_tile_overflow_before_allocation() {
        for (m, n, k) in [
            (0, 3, 4),
            (2, 0, 4),
            (2, 3, 0),
            (usize::MAX, 2, 2),
            (2, usize::MAX, 2),
            (2, 2, usize::MAX),
            (1, 1, usize::MAX / std::mem::size_of::<f32>() + 1),
            (i32::MAX as usize, 1, 1),
            (1, i32::MAX as usize, 1),
            (1, 1, i32::MAX as usize),
            (2, 2, i32::MAX as usize / 2 + 1),
        ] {
            assert!(
                GemmOp { m, n, k }.expected_output_len().is_err(),
                "{m}/{n}/{k}"
            );
        }
        assert!(std::panic::catch_unwind(|| GemmOp {
            m: usize::MAX,
            n: 2,
            k: 2
        }
        .run_cpu(0))
        .is_err());
    }
}
