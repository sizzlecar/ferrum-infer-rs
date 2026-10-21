//! Independent Candle F64 reference plus a deliberately restricted, preflighted
//! numerical domain. No Reference FP32 accumulation is used to produce targets
//! or to establish the domain.

use anyhow::{bail, Context, Result};
use candle_core::{Device, Tensor};
use half::bf16;
use serde::Serialize;
use serde_json::{json, Value};

const OUTPUT_CHANNEL_TILE: usize = 64;
const U32: f64 = 1.0 / 16_777_216.0;
const U64: f64 = 1.0 / 9_007_199_254_740_992.0;
const UB: f64 = 1.0 / 256.0;

#[derive(Debug, Clone, Copy)]
pub struct ProjectionShape {
    pub rows: usize,
    pub inputs: usize,
    pub outputs: usize,
}

impl ProjectionShape {
    pub fn new(rows: usize, inputs: usize, outputs: usize) -> Result<Self> {
        if rows == 0 || inputs == 0 || outputs == 0 {
            bail!("projection axes must be positive");
        }
        let shape = Self {
            rows,
            inputs,
            outputs,
        };
        shape.input_bytes()?;
        shape.weight_bytes()?;
        shape.output_bytes()?;
        gamma(inputs, U32)?;
        Ok(shape)
    }

    pub fn input_bytes(self) -> Result<usize> {
        bytes(self.rows, self.inputs)
    }
    pub fn weight_bytes(self) -> Result<usize> {
        bytes(self.outputs, self.inputs)
    }
    pub fn output_bytes(self) -> Result<usize> {
        bytes(self.rows, self.outputs)
    }
}

fn bytes(a: usize, b: usize) -> Result<usize> {
    a.checked_mul(b)
        .and_then(|n| n.checked_mul(2))
        .filter(|&n| n <= isize::MAX as usize)
        .context("projection extent exceeds address space")
}

fn upwards(value: f64) -> Result<f64> {
    if !value.is_finite() || value < 0.0 {
        bail!("nonnegative finite bound arithmetic is required");
    }
    // Exact zero does not need inflation. All positive intermediates in this
    // restricted domain are far above F64 subnormal arithmetic.
    let value = if value == 0.0 { value } else { value.next_up() };
    if !value.is_finite() {
        bail!("bound overflows F64");
    }
    Ok(value)
}

fn gamma(k: usize, unit_roundoff: f64) -> Result<f64> {
    let operations = k.checked_mul(2).context("2K overflows")?;
    if k == 0 || operations >= 16_777_216 {
        bail!("numerical domain requires 0 < 2K * 2^-24 < 1");
    }
    // operations is now an exactly representable F64 integer. Round the
    // numerator upward and denominator downward before dividing upward.
    let numerator = upwards(operations as f64 * unit_roundoff)?;
    let denominator = (1.0 - numerator).next_down();
    if denominator <= 0.0 {
        bail!("gamma denominator is not positive");
    }
    upwards(numerator / denominator)
}

fn bits(bytes: &[u8], index: usize) -> u16 {
    u16::from_le_bytes([bytes[index * 2], bytes[index * 2 + 1]])
}

fn number(bytes: &[u8], index: usize) -> f64 {
    f64::from(bf16::from_bits(bits(bytes, index)).to_f32())
}

fn check_length(bytes: &[u8], expected: usize, label: &str) -> Result<()> {
    if bytes.len() != expected {
        bail!("{label} length {} differs from {expected}", bytes.len());
    }
    Ok(())
}

fn check_operands(bytes: &[u8], label: &str) -> Result<()> {
    for (index, pair) in bytes.chunks_exact(2).enumerate() {
        let value = u16::from_le_bytes([pair[0], pair[1]]) & 0x7fff;
        if value >= 0x7f80 || (value != 0 && value < 0x0080) {
            bail!("{label}[{index}] is outside the finite BF16 normal-or-zero domain");
        }
    }
    Ok(())
}

fn product(value: f64) -> Result<()> {
    if value == 0.0 {
        return Ok(());
    }
    let fp32 = value as f32;
    if !fp32.is_finite() || f64::from(fp32) != value {
        bail!("product is not exactly representable in finite FP32");
    }
    let raw = fp32.to_bits() & 0x7fff_ffff;
    let exponent = ((raw >> 23) & 0xff) as i32;
    let fraction = raw & 0x7f_ffff;
    let (significand, least_exponent) = if exponent == 0 {
        (fraction, -149)
    } else {
        ((1 << 23) | fraction, exponent - 127 - 23)
    };
    let quantum_exponent = least_exponent + significand.trailing_zeros() as i32;
    if quantum_exponent < -126 {
        bail!("product binary quantum is below 2^-126; underflow-free partial sums are not established");
    }
    Ok(())
}

fn sum_upper(
    shape: ProjectionShape,
    input: &[u8],
    weight: &[u8],
    row: usize,
    output: usize,
) -> Result<f64> {
    let mut absolute = 0.0;
    for k in 0..shape.inputs {
        // A product of two finite BF16 numbers fits exactly in F64 (including
        // its exponent range). The stronger FP32/lattice condition is checked.
        let value =
            number(input, row * shape.inputs + k) * number(weight, output * shape.inputs + k);
        product(value)
            .with_context(|| format!("product at row {row}, output {output}, input {k}"))?;
        absolute = upwards(absolute + value.abs())?;
    }
    Ok(absolute)
}

pub fn validate_domain(shape: ProjectionShape, input: &[u8], weight: &[u8]) -> Result<()> {
    check_length(input, shape.input_bytes()?, "input")?;
    check_length(weight, shape.weight_bytes()?, "weight")?;
    check_operands(input, "input")?;
    check_operands(weight, "weight")?;
    let expansion = upwards(1.0 + gamma(shape.inputs, U32)?)?;
    for row in 0..shape.rows {
        for output in 0..shape.outputs {
            let magnitude = upwards(sum_upper(shape, input, weight, row, output)? * expansion)?;
            if magnitude > f64::from(bf16::MAX.to_f32()) {
                bail!("row {row}, output {output} cannot guarantee overflow-free FP32 partial sums and BF16 conversion");
            }
        }
    }
    // Every product is a multiple of a binary quantum >= 2^-126. FP32
    // addition/rounding preserves this lattice, so a partial sum is either
    // exactly zero or normal. The absolute-sum check bounds every partial sum
    // independently of reduction order; it also excludes final BF16 overflow.
    Ok(())
}

fn bound(shape: ProjectionShape, reference: f64, absolute_upper: f64) -> Result<f64> {
    if !reference.is_finite() {
        bail!("Candle produced a nonfinite F64 reference");
    }
    let e32 = upwards(gamma(shape.inputs, U32)? * absolute_upper)?;
    let e64 = upwards(gamma(shape.inputs, U64)? * absolute_upper)?;
    let errors = upwards(e32 + e64)?;
    let magnitude = upwards(reference.abs() + errors)?;
    let rounding = upwards(UB * magnitude)?;
    upwards(upwards(errors + rounding)? + 2.0_f64.powi(-134))
}

pub fn declaration(shape: ProjectionShape) -> Value {
    json!({
        "kind": "OperationDefined; independent Candle 0.9.2 CPU F64 matmul",
        "mathematics": "Y = X * transpose(W); BF16 operands exactly expanded to F64; no bias",
        "ferrum_arithmetic": "BF16 operands -> FP32 products/accumulation -> one BF16 RNE conversion",
        "domain": "stricter than general operation: finite BF16 normal-or-zero operands; each product exact in FP32 with least binary bit >= 2^-126; upward S_upper*(1+gamma(2K,2^-24)) <= BF16 MAX; exact-zero cancellation allowed",
        "domain_proof": "product lattice excludes nonzero subnormal partial sums; expanded absolute sum bounds all partial sums without reproducing the Reference fold",
        "required_k": "0 < 2K*2^-24 < 1",
        "bound": "E32 + E64 + 2^-8*(abs(y64)+E64+E32) + 2^-134",
        "E32": "gamma(2K,2^-24)*S_upper",
        "E64": "gamma(2K,2^-53)*S_upper",
        "rounding": "S_upper and every nonnegative bound operation rounded upward; gamma denominator rounded downward",
        "reference_output_channel_tile": OUTPUT_CHANNEL_TILE,
        "k": shape.inputs,
        "input_generation": "analytic rows: zero, basis[0], negative basis[K-1], basis[K/2]; dense rows: deterministic nonzero signed dyadic fractions, all K coordinates participate",
        "upstream_parity": false,
        "candle_feature_note": "CPU F64 backend follows the compiled Candle feature set; this is not native BF16 matmul or a Torch oracle",
        "relative_error": "diagnostic on nonzero references; absolute input-dependent bound decides acceptance"
    })
}

pub fn analytic_inputs(shape: ProjectionShape) -> Result<Vec<u8>> {
    if shape.rows != 4 {
        bail!("analytic probe set requires four rows");
    }
    let mut input = vec![0; shape.input_bytes()?];
    for (row, column, value) in [
        (1, 0, 1.0),
        (2, shape.inputs - 1, -1.0),
        (3, shape.inputs / 2, 1.0),
    ] {
        let index = (row * shape.inputs + column) * 2;
        input[index..index + 2].copy_from_slice(&bf16::from_f32(value).to_bits().to_le_bytes());
    }
    Ok(input)
}

pub fn dense_inputs(shape: ProjectionShape) -> Result<Vec<u8>> {
    let mut input = vec![0; shape.input_bytes()?];
    for row in 0..shape.rows {
        for k in 0..shape.inputs {
            // Reduced operands avoid index arithmetic overflow. Dyadic values
            // are exactly representable and do not require an RNG or libm.
            let residue = (17 * (k % 31) + 11 * (row % 31)) % 31;
            let magnitude = (residue + 1) as f32 / 32.0;
            let value = if (k % 2 + row % 2) % 2 == 0 {
                magnitude
            } else {
                -magnitude
            };
            let index = (row * shape.inputs + k) * 2;
            input[index..index + 2].copy_from_slice(&bf16::from_f32(value).to_bits().to_le_bytes());
        }
    }
    Ok(input)
}

#[derive(Debug, Serialize)]
struct CoordinateError {
    row: usize,
    output: usize,
    expected: f64,
    actual: Option<f64>,
    absolute_error: Option<f64>,
    allowed_absolute_error: f64,
}

#[derive(Debug, Serialize)]
pub struct Comparison {
    reference: &'static str,
    samples: usize,
    finite_samples: usize,
    failures: usize,
    nonfinite_outputs: usize,
    max_absolute_error: Option<f64>,
    max_relative_error_nonzero_reference: Option<f64>,
    rmse: Option<f64>,
    worst_coordinate: Option<CoordinateError>,
    first_failure: Option<CoordinateError>,
    #[serde(skip)]
    square_sum: f64,
}

impl Comparison {
    fn new(reference: &'static str) -> Self {
        Self {
            reference,
            samples: 0,
            finite_samples: 0,
            failures: 0,
            nonfinite_outputs: 0,
            max_absolute_error: None,
            max_relative_error_nonzero_reference: None,
            rmse: None,
            worst_coordinate: None,
            first_failure: None,
            square_sum: 0.0,
        }
    }

    pub fn passed(&self) -> bool {
        self.samples > 0 && self.failures == 0
    }

    fn observe(
        &mut self,
        row: usize,
        output: usize,
        expected: f64,
        actual: f64,
        allowed: f64,
        exact_bits_match: Option<bool>,
    ) {
        self.samples += 1;
        let absolute = actual.is_finite().then(|| (actual - expected).abs());
        let failed = match absolute {
            Some(error) => {
                self.finite_samples += 1;
                self.square_sum += error * error;
                self.rmse = Some((self.square_sum / self.finite_samples as f64).sqrt());
                if self
                    .max_absolute_error
                    .is_none_or(|previous| error > previous)
                {
                    self.max_absolute_error = Some(error);
                    self.worst_coordinate = Some(CoordinateError {
                        row,
                        output,
                        expected,
                        actual: Some(actual),
                        absolute_error: Some(error),
                        allowed_absolute_error: allowed,
                    });
                }
                if expected != 0.0 {
                    let relative = error / expected.abs();
                    if relative.is_finite()
                        && self
                            .max_relative_error_nonzero_reference
                            .is_none_or(|previous| relative > previous)
                    {
                        self.max_relative_error_nonzero_reference = Some(relative);
                    }
                }
                // Round the comparison distance upward as well: a downward
                // rounded subtraction must not create a false acceptance.
                let distance_upper = if error == 0.0 { 0.0 } else { error.next_up() };
                exact_bits_match.is_some_and(|matches| !matches) || distance_upper > allowed
            }
            None => {
                self.nonfinite_outputs += 1;
                true
            }
        };
        if failed {
            self.failures += 1;
            if self.first_failure.is_none() {
                self.first_failure = Some(CoordinateError {
                    row,
                    output,
                    expected,
                    actual: actual.is_finite().then_some(actual),
                    absolute_error: absolute,
                    allowed_absolute_error: allowed,
                });
            }
        }
    }
}

pub fn compare_analytic(
    shape: ProjectionShape,
    weight: &[u8],
    output: &[u8],
) -> Result<Comparison> {
    if shape.rows != 4 {
        bail!("analytic comparison requires four rows");
    }
    check_length(weight, shape.weight_bytes()?, "weight")?;
    check_length(output, shape.output_bytes()?, "output")?;
    check_operands(weight, "weight")?;
    let mut report = Comparison::new("analytic BF16 bits: zero/basis/negative-basis");
    for row in 0..shape.rows {
        for out in 0..shape.outputs {
            let mut expected = match row {
                0 => 0,
                1 => bits(weight, out * shape.inputs),
                2 => bits(weight, out * shape.inputs + shape.inputs - 1) ^ 0x8000,
                3 => bits(weight, out * shape.inputs + shape.inputs / 2),
                _ => unreachable!(),
            };
            // The declared fold starts at +0; additions of zero products under
            // RNE leave +0 even if an individual weight has a signed zero.
            if expected & 0x7fff == 0 {
                expected = 0;
            }
            let actual = bits(output, row * shape.outputs + out);
            report.observe(
                row,
                out,
                f64::from(bf16::from_bits(expected).to_f32()),
                f64::from(bf16::from_bits(actual).to_f32()),
                0.0,
                Some(actual == expected),
            );
        }
    }
    Ok(report)
}

pub fn compare_dense(
    shape: ProjectionShape,
    input: &[u8],
    weight: &[u8],
    output: &[u8],
) -> Result<Comparison> {
    check_length(output, shape.output_bytes()?, "output")?;
    validate_domain(shape, input, weight)?;
    let input_f64 = (0..shape.rows * shape.inputs)
        .map(|index| number(input, index))
        .collect::<Vec<_>>();
    let x = Tensor::from_vec(input_f64, (shape.rows, shape.inputs), &Device::Cpu)?;
    let mut report =
        Comparison::new("Candle CPU F64 GEMM, BF16 expanded exactly, input-dependent bound");
    for start in (0..shape.outputs).step_by(OUTPUT_CHANNEL_TILE) {
        let count = OUTPUT_CHANNEL_TILE.min(shape.outputs - start);
        let weight_f64 = (start * shape.inputs..(start + count) * shape.inputs)
            .map(|index| number(weight, index))
            .collect::<Vec<_>>();
        let w = Tensor::from_vec(weight_f64, (count, shape.inputs), &Device::Cpu)?;
        let expected = x.matmul(&w.t()?)?.flatten_all()?.to_vec1::<f64>()?;
        for row in 0..shape.rows {
            for channel in 0..count {
                let out = start + channel;
                let reference = expected[row * count + channel];
                let allowed = bound(shape, reference, sum_upper(shape, input, weight, row, out)?)?;
                report.observe(
                    row,
                    out,
                    reference,
                    number(output, row * shape.outputs + out),
                    allowed,
                    None,
                );
            }
        }
    }
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn encode(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| bf16::from_f32(*value).to_bits().to_le_bytes())
            .collect()
    }

    #[test]
    fn dense_candle_oracle_accepts_hand_computed_values_and_rejects_perturbation() {
        let shape = ProjectionShape::new(2, 3, 2).unwrap();
        let input = encode(&[1., 2., 3., -1., 0., 2.]);
        let weight = encode(&[2., -1., 1., 1., 3., -2.]);
        let expected = encode(&[3., 1., 0., -5.]);
        assert!(compare_dense(shape, &input, &weight, &expected)
            .unwrap()
            .passed());
        let perturbed = encode(&[3., 1., 0., -4.]);
        let report = compare_dense(shape, &input, &weight, &perturbed).unwrap();
        assert!(!report.passed());
        assert_eq!(report.failures, 1);
        assert_eq!(report.first_failure.unwrap().row, 1);
    }

    #[test]
    fn domain_rejects_nan_subnormal_tiny_product_overflow_and_invalid_k() {
        let shape = ProjectionShape::new(1, 1, 1).unwrap();
        let one = encode(&[1.]);
        for invalid in [0x7fc0_u16, 0x7f80, 0x0001] {
            assert!(validate_domain(shape, &invalid.to_le_bytes(), &one).is_err());
        }
        assert!(validate_domain(shape, &encode(&[f32::MIN_POSITIVE]), &encode(&[0.5])).is_err());
        assert!(validate_domain(shape, &encode(&[bf16::MAX.to_f32()]), &encode(&[2.])).is_err());
        assert!(ProjectionShape::new(1, 1 << 23, 1).is_err());
        assert!(ProjectionShape::new(usize::MAX, 2, 1).is_err());
    }

    #[test]
    fn analytic_probes_have_independent_weight_column_expectations() {
        let shape = ProjectionShape::new(4, 3, 2).unwrap();
        let weight = encode(&[2., -1., 1., 1., 3., -2.]);
        let expected = encode(&[0., 0., 2., 1., -1., 2., -1., 3.]);
        assert!(compare_analytic(shape, &weight, &expected)
            .unwrap()
            .passed());
        assert!(validate_domain(shape, &analytic_inputs(shape).unwrap(), &weight).is_ok());
    }

    #[test]
    fn upward_bound_covers_cancellation_and_nonfinite_output_is_a_failure() {
        let shape = ProjectionShape::new(1, 2, 1).unwrap();
        let x = encode(&[1., 1.]);
        let w = encode(&[1., -1.]);
        assert!(bound(shape, 0.0, 2.0).unwrap() > 0.0);
        assert!(compare_dense(shape, &x, &w, &encode(&[0.]))
            .unwrap()
            .passed());
        let report = compare_dense(shape, &x, &w, &0x7fc0_u16.to_le_bytes()).unwrap();
        assert_eq!(report.nonfinite_outputs, 1);
        assert!(!report.passed());
    }
}
