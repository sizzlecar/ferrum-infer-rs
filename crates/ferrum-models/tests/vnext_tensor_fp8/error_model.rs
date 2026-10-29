//! Independent error envelopes, selected from the captured native plan.
//!
//! The Ada envelope is an empirical FP22 model, NOT an IEEE F32 claim or a
//! vendor guarantee of undocumented internals. SageAttention2 §3.4 reports
//! 13 fraction bits and truncation on Ada/Hopper, including zero-input MMAs:
//! https://arxiv.org/html/2411.10958v7#S3.SS4
//! Our unchanged SM89 kernel uses CUTLASS cb4247394dd82148787aed73e5dc7cef33cbf862
//! GemmShape<128,64,128>, warp K128 and instruction K32, OpMultiplyAdd.
//! arch::detail::UseStagedAccumulation selects a temporary accumulator;
//! gemm/threadblock/mma_multistage.h promotes it at warp_mma_k==0 and at the
//! end. Thus the first segment can be K32, later segments at most K128, with
//! a tail segment: at most ceil(K/128)+1 F32 promotions. No split-K is used.
//!
//! Finite E4M3 products are exact in 14 significant bits. Conservatively allow
//! one FP22 truncation per local product addition plus all four K32 instruction
//! boundaries, including padded zero MMAs. Truncation has u=2^-13 (not RNE's
//! half-ulp). Compose gamma_local with gamma_promotions(F32), then account for
//! the two F32 scale multiplies and BF16 RNE separately. Bounds use sum |a*w|,
//! never only the final sum, so cancellation remains covered. This model is
//! intentionally independent of observed maximum errors and is not a claim of
//! full-model quality or cross-device bitwise parity.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorModel {
    IeeeF32,
    Sm89Fp22Block128F32Promotion,
}

impl ErrorModel {
    pub fn for_cuda(architecture: u32, kernel_id: u32) -> Result<Self, String> {
        use ferrum_kernels::native_ops::cutlass_fp8::{
            CUTLASS_FP8_KERNEL_128, CUTLASS_FP8_KERNEL_SM89,
        };
        match (architecture, kernel_id) {
            (890, CUTLASS_FP8_KERNEL_SM89) => Ok(Self::Sm89Fp22Block128F32Promotion),
            (1200, CUTLASS_FP8_KERNEL_128) => Ok(Self::IeeeF32),
            _ => Err(format!(
                "unmodeled FP8 native plan architecture={architecture}, kernel={kernel_id}"
            )),
        }
    }

    pub fn accumulator_bound(self, absolute_sum: f64, k: usize) -> f64 {
        assert!(k > 0 && absolute_sum.is_finite() && absolute_sum >= 0.0);
        match self {
            // Preserve the original IEEE F32 oracle exactly.
            Self::IeeeF32 => gamma(2 * k + 8, 2_f64.powi(-24)) * absolute_sum,
            Self::Sm89Fp22Block128F32Promotion => {
                let local = gamma(k.min(128) + 4, 2_f64.powi(-13));
                let outer = gamma(k.div_ceil(128) + 1, 2_f64.powi(-24));
                (local + outer * (1.0 + local)) * absolute_sum
            }
        }
    }

    pub fn output_bound(self, expected: f64, absolute_sum: f64, k: usize) -> f64 {
        let mut error = self.accumulator_bound(absolute_sum, k);
        if self == Self::Sm89Fp22Block128F32Promotion {
            error += gamma(2, 2_f64.powi(-24)) * (expected.abs() + error);
        }
        error + (expected.abs() + error) / 256.0 + 2_f64.powi(-130)
    }
}

fn gamma(operations: usize, u: f64) -> f64 {
    let nu = operations as f64 * u;
    assert!(nu < 1.0, "error model is outside its finite gamma domain");
    nu / (1.0 - nu)
}
