//! Legacy Metal command-context lifecycle, using F32 GEMM -> blit -> SiLU.
//! This proves dependent submission/readback behavior, not quantized kernels,
//! production-plan dispatch or model scheduling performance.

use super::{gemm::GemmOp, random_vec, silu_mul::SiluMulOp, OpUnderTest, Output};
pub use ferrum_bench_core::release_regression::numerics::{SubmissionPhase, SUBMISSION_PHASES};
use ferrum_kernels::backend::{cpu::CpuBackend, Backend};

pub struct MetalContextOp {
    pub tokens: usize,
    pub intermediate: usize,
    pub k: usize,
}

/// Replay every lifecycle segment independently; a high-energy successful
/// segment must not hide a failed lower-energy one in the aggregate NMSE.
pub fn compare_submission_segments(
    op: &MetalContextOp,
    reference: &[f32],
    actual: &[f32],
    tolerance: f64,
) -> Result<Vec<super::required::NumericalMetrics>, String> {
    op.expected_output_len()?;
    ferrum_bench_core::release_regression::numerics::compare_submission_segments(
        op.segment_len()?,
        reference,
        actual,
        tolerance,
    )
}

struct Buffers<B: Backend> {
    a: B::Buffer,
    b: B::Buffer,
    product: B::Buffer,
    copied: B::Buffer,
    output: B::Buffer,
}

impl MetalContextOp {
    pub fn segment_len(&self) -> Result<usize, String> {
        let n = self
            .intermediate
            .checked_mul(2)
            .ok_or("context GEMM width overflow")?;
        GemmOp {
            m: self.tokens,
            n,
            k: self.k,
        }
        .expected_output_len()?;
        SiluMulOp {
            tokens: self.tokens,
            intermediate: self.intermediate,
        }
        .expected_output_len()
    }

    pub fn expected_output_len(&self) -> Result<usize, String> {
        let elements = self
            .segment_len()?
            .checked_mul(SUBMISSION_PHASES.len())
            .ok_or("context output element count overflow")?;
        let bytes = elements
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or("context output byte size overflow")?;
        if bytes > isize::MAX as usize {
            return Err("context output exceeds host indexing range".into());
        }
        Ok(elements)
    }

    fn buffers<B: Backend>(&self, seed: u64) -> Buffers<B> {
        self.expected_output_len().expect("invalid context fixture");
        let n = 2 * self.intermediate;
        Buffers {
            a: B::from_slice(&random_vec(self.tokens * self.k, -0.5, 0.5, seed)),
            b: B::from_slice(&random_vec(n * self.k, -0.5, 0.5, seed.wrapping_add(1))),
            // Distinct finite sentinels make a missing compute/blit visible.
            product: B::from_slice(&vec![11.0; self.tokens * n]),
            copied: B::from_slice(&vec![-7.0; self.tokens * n]),
            output: B::from_slice(&vec![13.0; self.tokens * self.intermediate]),
        }
    }

    fn enqueue<B: Backend>(&self, ctx: &mut B::Context, buffers: &mut Buffers<B>) {
        let n = 2 * self.intermediate;
        B::gemm(
            ctx,
            &buffers.a,
            &buffers.b,
            &mut buffers.product,
            self.tokens,
            n,
            self.k,
        );
        B::copy_slice(
            ctx,
            &buffers.product,
            0,
            &mut buffers.copied,
            0,
            self.tokens * n,
        );
        B::fused_silu_mul_split(
            ctx,
            &buffers.copied,
            &mut buffers.output,
            self.tokens,
            self.intermediate,
        );
    }
}

impl OpUnderTest for MetalContextOp {
    fn name(&self) -> &str {
        "metal_context"
    }

    fn run_cpu(&self, seed: u64) -> Output {
        let len = self.segment_len().expect("valid context shape");
        let mut output = Vec::with_capacity(self.expected_output_len().unwrap());
        for (index, _) in SUBMISSION_PHASES.iter().enumerate() {
            let mut buffers = self.buffers::<CpuBackend>(seed.wrapping_add(index as u64 * 2));
            let mut ctx = CpuBackend::new_context();
            self.enqueue::<CpuBackend>(&mut ctx, &mut buffers);
            CpuBackend::sync(&mut ctx);
            output.extend(CpuBackend::to_vec(&buffers.output, len));
        }
        output
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    fn run_metal(&self, seed: u64) -> Output {
        use ferrum_kernels::backend::metal::MetalBackend;
        let len = self.segment_len().expect("valid context shape");
        let mut output = Vec::with_capacity(self.expected_output_len().unwrap());
        let mut first = self.buffers::<MetalBackend>(seed);
        let mut ctx = MetalBackend::new_context();
        self.enqueue::<MetalBackend>(&mut ctx, &mut first);
        MetalBackend::sync_checked(&mut ctx).expect("initial chain Metal completion");
        output.extend(MetalBackend::to_vec(&first.output, len));

        let mut reused = self.buffers::<MetalBackend>(seed.wrapping_add(2));
        let mut independent = self.buffers::<MetalBackend>(seed.wrapping_add(4));
        let mut peer = MetalBackend::new_context();
        self.enqueue::<MetalBackend>(&mut ctx, &mut reused);
        self.enqueue::<MetalBackend>(&mut peer, &mut independent);
        // Submit the independently recorded context first: the reused context
        // must retain its own command buffer, encoder and bound resources.
        MetalBackend::sync_checked(&mut peer).expect("independent chain Metal completion");
        MetalBackend::sync_checked(&mut ctx).expect("reused chain Metal completion");
        output.extend(MetalBackend::to_vec(&reused.output, len));
        output.extend(MetalBackend::to_vec(&independent.output, len));

        let mut dropped = self.buffers::<MetalBackend>(seed.wrapping_add(6));
        {
            let mut pending = MetalBackend::new_context();
            self.enqueue::<MetalBackend>(&mut pending, &mut dropped);
            // Production Drop flushes and waits. Its API returns no driver
            // status; this segment asserts the actual post-Drop data only.
        }
        output.extend(MetalBackend::to_vec(&dropped.output, len));
        output
    }

    #[cfg(feature = "cuda")]
    fn run_cuda(&self, _seed: u64) -> Output {
        panic!("Metal context lifecycle has no CUDA adapter")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::op_diff::{required::compare_outputs, NMSE_FP32_TOL};

    #[test]
    fn cpu_chain_matches_independent_matmul_and_silu_for_each_lifecycle_segment() {
        let op = MetalContextOp {
            tokens: 3,
            intermediate: 33,
            k: 35,
        };
        let actual = op.run_cpu(7);
        let len = op.segment_len().unwrap();
        assert_eq!(actual.len(), op.expected_output_len().unwrap());
        for (index, segment) in actual.chunks_exact(len).enumerate() {
            let seed = 7 + index as u64 * 2;
            let a = random_vec(op.tokens * op.k, -0.5, 0.5, seed);
            let b = random_vec(2 * op.intermediate * op.k, -0.5, 0.5, seed + 1);
            let expected: Vec<f32> = a
                .chunks_exact(op.k)
                .flat_map(|row| {
                    let product: Vec<f32> = b
                        .chunks_exact(op.k)
                        .map(|column| row.iter().zip(column).map(|(a, b)| a * b).sum())
                        .collect();
                    let (gate, up) = product.split_at(op.intermediate);
                    gate.iter()
                        .zip(up)
                        .map(|(g, u)| g / (1.0 + (-g).exp()) * u)
                        .collect::<Vec<_>>()
                })
                .collect();
            compare_outputs(&expected, segment, NMSE_FP32_TOL).unwrap();
            assert!(compare_outputs(&expected, &vec![13.0; len], NMSE_FP32_TOL).is_err());
            if index > 0 {
                assert!(
                    compare_outputs(&actual[..len], segment, NMSE_FP32_TOL).is_err(),
                    "distinct contexts must not share the first result"
                );
            }
        }
    }

    #[test]
    fn invalid_context_shapes_fail_before_allocation() {
        for (tokens, intermediate, k) in [
            (0, 33, 35),
            (3, 0, 35),
            (3, 33, 0),
            (usize::MAX, 33, 35),
            (3, usize::MAX, 35),
            (3, 33, usize::MAX),
        ] {
            assert!(MetalContextOp {
                tokens,
                intermediate,
                k
            }
            .expected_output_len()
            .is_err());
        }
    }
}
