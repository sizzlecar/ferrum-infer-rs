//! Independent Candle F64 reference for the Reference BF16 LayerNorm profile.
//!
//! Prepare before executing the candidate. Bounds describe balanced pairwise
//! FP32 two-pass statistics, separate sqrt/divide/multiply, and one BF16 RNE.
//! They do not certify CUDA Welford, approximate rsqrt, or FTZ/DAZ. The runtime
//! only preflights structure; rejecting this oracle's numerical domain is not
//! a claim that runtime submission scans intermediate tensor contents.
//!
//! For a binary reduction tree of height h, rounding satisfies
//! |fl(z)-z| <= u|z|+eta. The absolute eta contribution counts all n-1
//! additions, not only tree depth. Variance also includes centering, squaring,
//! summation and division errors. All bounds are evaluated outward in F64.
//! Constant/exact fixtures remain separate strict checks: a conservative bound
//! near a constant row must not replace an independently known exact result.

use candle_core::{Device, Tensor};
use half::bf16;

type Result<T> = std::result::Result<T, String>;

const U32: f64 = 1.0 / 16_777_216.0;
const U64: f64 = 1.0 / 9_007_199_254_740_992.0;
const ETA32: f64 = f64::from_bits((1023_u64 - 150) << 52);
const ETA_BF16: f64 = f64::from_bits((1023_u64 - 134) << 52);
const U_BF16: f64 = 1.0 / 256.0;

#[derive(Debug)]
struct Expected {
    reference: f64,
    allowed: f64,
}

#[derive(Debug)]
pub(super) struct PreparedLayerNormOracle {
    width: usize,
    expected: Vec<Expected>,
}

impl PreparedLayerNormOracle {
    pub(super) fn check(&self, actual_bits: &[u16]) -> Result<()> {
        if actual_bits.len() != self.expected.len() {
            return Err("LayerNorm candidate length differs from the prepared shape".into());
        }
        for (index, (&bits, expected)) in actual_bits.iter().zip(&self.expected).enumerate() {
            let actual = number(bits);
            if !actual.is_finite() {
                return Err(format!(
                    "nonfinite candidate at row {}, column {}",
                    index / self.width,
                    index % self.width
                ));
            }
            let distance = distance_up(actual, expected.reference)?;
            if distance > expected.allowed {
                return Err(format!(
                    "LayerNorm row {}, column {}: actual={actual}, F64={}, distance_upper={distance}, allowed={}",
                    index / self.width, index % self.width, expected.reference, expected.allowed,
                ));
            }
        }
        Ok(())
    }
}

/// `epsilon` is the operation's already converted effective FP32 parameter,
/// not an ideal rational or an independently reparsed decimal.
pub(super) fn prepare(
    input_bits: &[u16],
    width: usize,
    epsilon: f32,
) -> Result<PreparedLayerNormOracle> {
    // Dependency feature unification can select vector math even when this
    // crate did not request it directly. Its sqrt rounding needs separate
    // evidence; this oracle proves Candle's scalar CPU F64 profile only.
    if candle_core::utils::has_accelerate() || candle_core::utils::has_mkl() {
        return Err("LayerNorm F64 oracle does not certify Accelerate/MKL arithmetic".into());
    }
    if width == 0 || width > (1 << 24) || input_bits.is_empty() || input_bits.len() % width != 0 {
        return Err("LayerNorm oracle needs nonempty exact rows and 1 <= width <= 2^24".into());
    }
    if !epsilon.is_finite() || epsilon < 1e-12_f32 || epsilon > 1.0 {
        return Err("effective FP32 epsilon is outside the declared [1e-12_f32, 1] domain".into());
    }
    input_bits
        .len()
        .checked_mul(std::mem::size_of::<Expected>())
        .filter(|&bytes| bytes <= isize::MAX as usize)
        .ok_or("oracle result extent exceeds address space")?;
    if input_bits.iter().any(|&bits| !number(bits).is_finite()) {
        return Err("LayerNorm oracle requires finite BF16 input".into());
    }
    let depth = if width == 1 {
        0
    } else {
        usize::BITS as usize - (width - 1).leading_zeros() as usize
    };
    let fp32 = Arithmetic {
        u: U32,
        eta: ETA32,
        depth,
    };
    // Candle 0.9.2 CPU VecOps<f64>::vec_reduce_sum uses sequential addition.
    // Explicit broadcast_div below avoids mean_keepdim's reciprocal-multiply.
    // BF16 inputs and width <= 2^24 keep every nonzero F64 intermediate far
    // above F64 underflow: sums retain the 2^-133 input lattice; nonzero means
    // are >= 2^-157, differences >= 2^-210, and divided squares >= 2^-444.
    let fp64 = Arithmetic {
        u: U64,
        eta: 0.0,
        depth: width - 1,
    };
    let mut expected = Vec::with_capacity(input_bits.len());
    for (row, input) in input_bits.chunks_exact(width).enumerate() {
        let prepared = prepare_row(input, f64::from(epsilon), fp32, fp64)
            .map_err(|error| format!("oracle row {row}: {error}"))?;
        expected.extend(prepared);
    }
    Ok(PreparedLayerNormOracle { width, expected })
}

fn number(bits: u16) -> f64 {
    f64::from(bf16::from_bits(bits).to_f32())
}

// Nonnegative bound operations, including F64 underflow of an upper bound.
// Exact zero stays zero; a positive operation rounded to zero rounds up to
// the least F64 subnormal instead of silently losing a positive bound.
fn up(value: f64, exact_zero: bool) -> Result<f64> {
    if !value.is_finite() || value < 0.0 {
        return Err("bound arithmetic is nonfinite or negative".into());
    }
    let result = if exact_zero { 0.0 } else { value.next_up() };
    if !result.is_finite() {
        return Err("bound arithmetic overflows F64".into());
    }
    Ok(result)
}

fn add(a: f64, b: f64) -> Result<f64> {
    up(a + b, a == 0.0 && b == 0.0)
}
fn mul(a: f64, b: f64) -> Result<f64> {
    up(a * b, a == 0.0 || b == 0.0)
}
fn div(a: f64, b: f64) -> Result<f64> {
    if !b.is_finite() || b <= 0.0 {
        return Err("bound divisor is not positive finite".into());
    }
    up(a / b, a == 0.0)
}
fn distance_up(a: f64, b: f64) -> Result<f64> {
    up((a - b).abs(), a == b)
}

#[derive(Clone, Copy)]
struct Arithmetic {
    u: f64,
    eta: f64,
    depth: usize,
}

impl Arithmetic {
    fn inverse_margin(self, operations: usize) -> Result<f64> {
        let scaled = mul(operations as f64, self.u)?;
        let denominator = (1.0 - scaled).next_down();
        div(1.0, denominator)
    }
    fn gamma(self, operations: usize) -> Result<f64> {
        if operations == 0 {
            return Ok(0.0);
        }
        mul(
            mul(operations as f64, self.u)?,
            self.inverse_margin(operations)?,
        )
    }
    fn sum_error(self, absolute_sum: f64, n: usize) -> Result<f64> {
        add(
            mul(self.gamma(self.depth)?, absolute_sum)?,
            mul(
                mul((n - 1) as f64, self.eta)?,
                self.inverse_margin(self.depth)?,
            )?,
        )
    }
    fn mean_error(self, absolute_sum: f64, n: usize) -> Result<f64> {
        let error = self.sum_error(absolute_sum, n)?;
        add(
            div(
                add(mul(add(1.0, self.u)?, error)?, mul(self.u, absolute_sum)?)?,
                n as f64,
            )?,
            self.eta,
        )
    }
    fn center_error(self, magnitude: f64, mean_error: f64) -> Result<f64> {
        add(
            add(mean_error, mul(self.u, add(magnitude, mean_error)?)?)?,
            self.eta,
        )
    }
}

#[derive(Clone, Copy)]
struct Interval {
    lo: f64,
    hi: f64,
}

impl Interval {
    fn around(value: f64, error: f64) -> Result<Self> {
        if !value.is_finite() || !error.is_finite() || error < 0.0 {
            return Err("invalid F64 statistic enclosure".into());
        }
        if error == 0.0 {
            return Ok(Self {
                lo: value,
                hi: value,
            });
        }
        let interval = Self {
            lo: (value - error).next_down(),
            hi: (value + error).next_up(),
        };
        if !interval.lo.is_finite() || !interval.hi.is_finite() {
            return Err("statistic enclosure overflows".into());
        }
        Ok(interval)
    }
    fn distance(self, value: f64) -> Result<f64> {
        Ok(distance_up(value, self.lo)?.max(distance_up(value, self.hi)?))
    }
    fn inverse_sqrt(self) -> Result<Self> {
        if self.lo <= 0.0 || !self.hi.is_finite() || self.hi < self.lo {
            return Err("inverse square root interval is not positive finite".into());
        }
        Ok(Self {
            lo: (1.0 / self.hi.sqrt().next_up()).next_down(),
            hi: div(1.0, self.lo.sqrt().next_down())?,
        })
    }
}

struct StatisticsError {
    mean: f64,
    variance: f64,
}

fn no_fp32_overflow(label: &str, upper: f64) -> Result<()> {
    if !upper.is_finite() || upper > f64::from(f32::MAX) {
        return Err(format!(
            "{label}: FP32 intermediate overflow is not excluded"
        ));
    }
    Ok(())
}

fn statistics_error(
    input: &[u16],
    absolute_sum: f64,
    mean: Interval,
    arithmetic: Arithmetic,
    certify_fp32: bool,
) -> Result<StatisticsError> {
    let n = input.len();
    let mean_error = arithmetic.mean_error(absolute_sum, n)?;
    let mut perturbation_sum = 0.0;
    let mut square_sum = 0.0;
    if certify_fp32 {
        no_fp32_overflow(
            "mean reduction",
            add(absolute_sum, arithmetic.sum_error(absolute_sum, n)?)?,
        )?;
        no_fp32_overflow(
            "mean division",
            add(div(absolute_sum, n as f64)?, mean_error)?,
        )?;
    }
    for &bits in input {
        let d = mean.distance(number(bits))?;
        let error = arithmetic.center_error(d, mean_error)?;
        let centered_upper = add(d, error)?;
        let square_upper = mul(centered_upper, centered_upper)?;
        square_sum = add(square_sum, square_upper)?;
        perturbation_sum = add(
            perturbation_sum,
            add(mul(mul(2.0, d)?, error)?, mul(error, error)?)?,
        )?;
        if certify_fp32 {
            no_fp32_overflow("centering", centered_upper)?;
            no_fp32_overflow(
                "centered square",
                add(mul(add(1.0, arithmetic.u)?, square_upper)?, arithmetic.eta)?,
            )?;
        }
    }
    // n squares + n-1 tree additions + division. The eta term is deliberately
    // conservative: division by n can only shrink earlier absolute errors.
    let operations = arithmetic.depth + 2;
    let variance_error = add(
        add(
            div(perturbation_sum, n as f64)?,
            mul(arithmetic.gamma(operations)?, div(square_sum, n as f64)?)?,
        )?,
        mul(
            mul(2.0 * n as f64, arithmetic.eta)?,
            arithmetic.inverse_margin(operations)?,
        )?,
    )?;
    if certify_fp32 {
        let sum_upper = add(
            mul(
                add(1.0, arithmetic.gamma(arithmetic.depth + 1)?)?,
                square_sum,
            )?,
            mul(
                mul((2 * n - 1) as f64, arithmetic.eta)?,
                arithmetic.inverse_margin(arithmetic.depth + 1)?,
            )?,
        )?;
        no_fp32_overflow("variance reduction", sum_upper)?;
        no_fp32_overflow(
            "variance division",
            add(
                mul(div(sum_upper, n as f64)?, add(1.0, arithmetic.u)?)?,
                arithmetic.eta,
            )?,
        )?;
    }
    Ok(StatisticsError {
        mean: mean_error,
        variance: variance_error,
    })
}

struct ReciprocalError {
    upper: f64,
    error: f64,
}

fn reciprocal_error(
    variance: Interval,
    epsilon: f64,
    variance_error: f64,
    arithmetic: Arithmetic,
    certify_fp32: bool,
) -> Result<ReciprocalError> {
    let true_q = Interval {
        lo: (variance.lo + epsilon).next_down(),
        hi: add(variance.hi, epsilon)?,
    };
    let q_error = add(
        add(
            variance_error,
            mul(
                arithmetic.u,
                add(add(variance.hi, variance_error)?, epsilon)?,
            )?,
        )?,
        arithmetic.eta,
    )?;
    // Computed squares, their sum, and division are nonnegative. Monotonic
    // IEEE addition therefore gives qhat >= effective epsilon, even when
    // variance underflows. Never subtract an error into a nonpositive root.
    let computed_q = Interval {
        lo: epsilon.max((true_q.lo - q_error).next_down()),
        hi: add(true_q.hi, q_error)?,
    };
    if certify_fp32 {
        no_fp32_overflow("variance plus epsilon", computed_q.hi)?;
    }
    let true_r = true_q.inverse_sqrt()?;
    let base = computed_q.inverse_sqrt()?;
    let one_minus = (1.0 - arithmetic.u).next_down();
    let one_plus = add(1.0, arithmetic.u)?;
    // sqrt and reciprocal are two distinct rounded normal operations.
    let alpha_lower = (one_minus / one_plus).next_down();
    let alpha_upper = div(one_plus, one_minus)?;
    let computed_r = Interval {
        lo: (alpha_lower * base.lo).next_down(),
        hi: mul(alpha_upper, base.hi)?,
    };
    let error = distance_up(computed_r.lo, true_r.hi)?.max(distance_up(computed_r.hi, true_r.lo)?);
    Ok(ReciprocalError {
        upper: computed_r.hi,
        error,
    })
}

fn output_error(
    d: f64,
    statistics: &StatisticsError,
    reciprocal: &ReciprocalError,
    arithmetic: Arithmetic,
) -> Result<(f64, f64)> {
    let centered_error = arithmetic.center_error(d, statistics.mean)?;
    let product_upper = mul(add(d, centered_error)?, reciprocal.upper)?;
    let rounding = add(mul(arithmetic.u, product_upper)?, arithmetic.eta)?;
    let error = add(
        add(
            mul(d, reciprocal.error)?,
            mul(centered_error, reciprocal.upper)?,
        )?,
        rounding,
    )?;
    Ok((error, add(product_upper, rounding)?))
}

struct F64Reference {
    mean: f64,
    variance: f64,
    output: Vec<f64>,
}

fn candle_reference(input: &[u16], epsilon: f64) -> Result<F64Reference> {
    let run = || -> candle_core::Result<F64Reference> {
        let x = Tensor::from_vec(
            input.iter().map(|&bits| number(bits)).collect::<Vec<_>>(),
            input.len(),
            &Device::Cpu,
        )?;
        let divisor = Tensor::new(input.len() as f64, &Device::Cpu)?;
        let mean = x.sum_all()?.broadcast_div(&divisor)?;
        let centered = x.broadcast_sub(&mean)?;
        let variance = centered.sqr()?.sum_all()?.broadcast_div(&divisor)?;
        let epsilon = Tensor::new(epsilon, &Device::Cpu)?;
        let one = Tensor::new(1.0_f64, &Device::Cpu)?;
        let reciprocal = one.broadcast_div(&variance.broadcast_add(&epsilon)?.sqrt()?)?;
        Ok(F64Reference {
            mean: mean.to_scalar()?,
            variance: variance.to_scalar()?,
            output: centered.broadcast_mul(&reciprocal)?.to_vec1()?,
        })
    };
    let reference = run().map_err(|error| format!("Candle F64 reference: {error}"))?;
    if !reference.mean.is_finite()
        || !reference.variance.is_finite()
        || reference.variance < 0.0
        || reference.output.iter().any(|value| !value.is_finite())
    {
        return Err("Candle F64 produced nonfinite/negative statistics or output".into());
    }
    Ok(reference)
}

fn prepare_row(
    input: &[u16],
    epsilon: f64,
    fp32: Arithmetic,
    fp64: Arithmetic,
) -> Result<Vec<Expected>> {
    let mut absolute_sum = 0.0;
    for &bits in input {
        absolute_sum = add(absolute_sum, number(bits).abs())?;
    }
    let reference = candle_reference(input, epsilon)?;
    // F64 results locate mathematical statistics only after adding their
    // independently derived F64 errors. No candidate value enters this path.
    let mean = Interval::around(reference.mean, fp64.mean_error(absolute_sum, input.len())?)?;
    let errors64 = statistics_error(input, absolute_sum, mean, fp64, false)?;
    let mut variance = Interval::around(reference.variance, errors64.variance)?;
    variance.lo = variance.lo.max(0.0);
    let errors32 = statistics_error(input, absolute_sum, mean, fp32, true)?;
    let reciprocal32 = reciprocal_error(variance, epsilon, errors32.variance, fp32, true)?;
    let reciprocal64 = reciprocal_error(variance, epsilon, errors64.variance, fp64, false)?;
    input
        .iter()
        .zip(reference.output)
        .map(|(&bits, y64)| {
            let d = mean.distance(number(bits))?;
            let (e32, magnitude32) = output_error(d, &errors32, &reciprocal32, fp32)?;
            if magnitude32 > f64::from(bf16::MAX.to_f32()) {
                return Err(
                    "normalization output: finite BF16 conversion is not established".into(),
                );
            }
            let (e64, _) = output_error(d, &errors64, &reciprocal64, fp64)?;
            let base = add(e32, e64)?;
            let allowed = add(add(base, mul(U_BF16, add(y64.abs(), base)?)?)?, ETA_BF16)?;
            Ok(Expected {
                reference: y64,
                allowed,
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bits(values: &[f32]) -> Vec<u16> {
        values
            .iter()
            .map(|&value| bf16::from_f32(value).to_bits())
            .collect()
    }

    #[test]
    fn exact_centering_and_offset_rows_pass_and_perturbation_fails() {
        let input = bits(&[1., 1., 2., 2., 128., 128., 129., 129.]);
        let oracle = prepare(&input, 4, 0.75).unwrap();
        let expected = bits(&[-0.5, -0.5, 0.5, 0.5, -0.5, -0.5, 0.5, 0.5]);
        oracle.check(&expected).unwrap();
        let mut wrong = expected;
        wrong[6] = bf16::from_f32(0.75).to_bits();
        assert!(oracle.check(&wrong).is_err());
    }

    #[test]
    fn odd_width_at_target_epsilon_uses_population_variance() {
        let oracle = prepare(&bits(&[-1., 0., 1.]), 3, 1e-6).unwrap();
        // +/-1/sqrt(2/3 + effective eps32) round to +/-1.2265625.
        oracle.check(&[0xbf9d, 0, 0x3f9d]).unwrap();
        assert!(oracle.check(&[0xbf80, 0, 0x3f80]).is_err());
    }

    #[test]
    fn near_constant_wide_row_has_hand_derived_bf16_outputs() {
        let mut input = vec![0x3f80; 4096];
        input[4095] = 0x3f81;
        // mean=1+2^-19, variance=4095*2^-38; q=2^-20, reciprocal=1024.
        let epsilon = (258_049.0_f64 / 274_877_906_944.0) as f32;
        let oracle = prepare(&input, 4096, epsilon).unwrap();
        let mut expected = vec![0xbb00; 4096];
        expected[4095] = 0x4100;
        oracle.check(&expected).unwrap();
        expected[4095] = 0x4180; // 16, far outside the input-dependent bound.
        assert!(oracle.check(&expected).is_err());
    }

    #[test]
    fn gradual_underflow_is_bounded_instead_of_silently_flushed() {
        let oracle = prepare(&[0x0001, 0x8001], 2, 1.0 / 1_048_576.0).unwrap();
        // BF16 +/-2^-133, FP32 squares underflow, sqrt(eps)=2^-10.
        oracle.check(&[0x0200, 0x8200]).unwrap();
        assert!(oracle.check(&[0, 0]).is_err());
    }

    #[test]
    fn domain_rejects_bad_shapes_epsilon_and_nonfinite_inputs() {
        for (input, width) in [
            (vec![], 1),
            (vec![0], 0),
            (vec![0; 3], 2),
            (vec![0], (1 << 24) + 1),
        ] {
            assert!(prepare(&input, width, 1e-6).is_err());
        }
        for epsilon in [0.0, -1.0, f32::NAN, f32::INFINITY, f32::from_bits(1), 2.0] {
            assert!(prepare(&[0], 1, epsilon).is_err());
        }
        for value in [0x7fc0, 0x7f80, 0xff80] {
            assert!(prepare(&[value], 1, 1e-6).is_err());
        }
    }

    #[test]
    fn domain_rejects_intermediate_sum_and_square_overflow_even_if_ideal_output_is_finite() {
        // Ideal constant-row normalization is zero, but its FP32 sum overflows.
        assert!(prepare(&[bf16::MAX.to_bits(); 2], 2, 1e-6).is_err());
        let large = f32::from_bits((127 + 64) << 23);
        assert!(prepare(&bits(&[-large, large]), 2, 1e-6).is_err());
    }

    #[test]
    fn candidate_nonfinite_and_wrong_extent_never_pass() {
        let oracle = prepare(&bits(&[-0.5, 0.5]), 2, 0.75).unwrap();
        for output in [vec![0x7fc0, 0x3f00], vec![0x7f80, 0x3f00], vec![0xbf00]] {
            assert!(oracle.check(&output).is_err());
        }
    }
}
