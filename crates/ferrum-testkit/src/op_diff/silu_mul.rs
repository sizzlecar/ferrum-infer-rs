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
    /// Checked row-split input/output sizes, before allocation or submission.
    /// Both backends use signed 32-bit offsets into the full gate/up buffer.
    pub fn expected_output_len(&self) -> Result<usize, String> {
        if self.tokens == 0 || self.intermediate == 0 {
            return Err("SiLU Mul tokens and intermediate must be nonzero".into());
        }
        let output = self
            .tokens
            .checked_mul(self.intermediate)
            .ok_or("SiLU Mul output element count overflows usize")?;
        let input = output
            .checked_mul(2)
            .ok_or("SiLU Mul gate/up element count overflows usize")?;
        let bytes = input
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or("SiLU Mul f32 input byte size overflows usize")?;
        if bytes > isize::MAX as usize || input > i32::MAX as usize {
            return Err("SiLU Mul input exceeds host or signed kernel indexing range".into());
        }
        // input <= i32::MAX also keeps the 256-wide padded output launch
        // representable, including Metal's uint thread ID -> int conversion.
        Ok(output)
    }

    fn output_len(&self) -> usize {
        self.expected_output_len()
            .expect("invalid SiLU Mul fixture")
    }

    fn build_input(&self, seed: u64) -> Vec<f32> {
        random_vec(self.output_len() * 2, -3.0, 3.0, seed)
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
        MetalBackend::sync_checked(&mut ctx)
            .unwrap_or_else(|error| panic!("SiLU Mul Metal completion failed: {error}"));
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::op_diff::{required::compare_outputs, NMSE_FP32_TOL};

    #[test]
    fn validates_silu_mul_shapes_and_checks_each_row_split() {
        for (tokens, intermediate) in [(4, 256), (3, 257)] {
            let op = SiluMulOp {
                tokens,
                intermediate,
            };
            assert_eq!(op.expected_output_len(), Ok(tokens * intermediate));
            let input = op.build_input(7);
            let actual = op.run_cpu(7);
            let expected: Vec<f32> = input
                .chunks_exact(2 * intermediate)
                .flat_map(|row| {
                    let (gate, up) = row.split_at(intermediate);
                    gate.iter().zip(up).map(|(g, u)| g / (1.0 + (-g).exp()) * u)
                })
                .collect();
            compare_outputs(&expected, &actual, NMSE_FP32_TOL).unwrap();
            // Accidentally returning only gate*up is finite and the right
            // length, but omits SiLU and must fail the numerical comparison.
            let without_activation: Vec<f32> = input
                .chunks_exact(2 * intermediate)
                .flat_map(|row| {
                    row[..intermediate]
                        .iter()
                        .zip(&row[intermediate..])
                        .map(|(g, u)| g * u)
                })
                .collect();
            assert!(compare_outputs(&expected, &without_activation, NMSE_FP32_TOL).is_err());
        }
    }

    #[test]
    fn rejects_silu_mul_empty_product_byte_and_input_index_overflow() {
        for (tokens, intermediate) in [
            (0, 256),
            (1, 0),
            (usize::MAX, 2),
            (1, usize::MAX / 2 + 1),
            (1, usize::MAX / 8 + 1),
            (1, i32::MAX as usize / 2 + 1),
            (2, i32::MAX as usize / 4 + 1),
        ] {
            assert!(
                SiluMulOp {
                    tokens,
                    intermediate
                }
                .expected_output_len()
                .is_err(),
                "{tokens}/{intermediate}"
            );
        }
        assert!(std::panic::catch_unwind(|| SiluMulOp {
            tokens: usize::MAX,
            intermediate: 2
        }
        .run_cpu(0))
        .is_err());
    }
}
