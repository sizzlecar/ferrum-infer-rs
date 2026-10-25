//! Independent scalar finite-domain oracle. This module imports no native
//! implementation or float8 conversion routine. It enumerates the specified
//! E4M3FN representation and chooses nearest-even, then uses F64 dot products.
use half::bf16;

pub fn decode(bits: u8) -> f64 {
    let sign = if bits & 128 == 0 { 1.0 } else { -1.0 };
    let exponent = (bits >> 3) & 15;
    let mantissa = bits & 7;
    if exponent == 15 && mantissa == 7 {
        return f64::NAN;
    }
    sign * if exponent == 0 {
        (mantissa as f64) * 2_f64.powi(-9)
    } else {
        (1.0 + mantissa as f64 / 8.0) * 2_f64.powi(i32::from(exponent) - 7)
    }
}

pub fn encode_finite(value: f32) -> u8 {
    assert!(value.is_finite());
    let value64 = f64::from(value.abs());
    let mut best = 0;
    let mut distance = f64::INFINITY;
    for bits in 0..=126_u8 {
        let candidate = (decode(bits) - value64).abs();
        if candidate < distance || (candidate == distance && bits & 1 == 0) {
            best = bits;
            distance = candidate;
        }
    }
    best | if value.is_sign_negative() { 128 } else { 0 }
}

pub struct Expected {
    pub quantized_dot: Vec<f64>,
    pub unquantized_dot: Vec<f64>,
    pub absolute_sum: Vec<f64>,
    pub activation_bytes: Vec<u8>,
    pub row_scales: Vec<f32>,
}

pub fn expected(input: &[bf16], weights: &[u8], scale: f32, k: usize, n: usize) -> Expected {
    assert!(k > 0 && n > 0);
    assert_eq!(input.len() % k, 0);
    assert_eq!(weights.len(), n * k);
    assert!(scale.is_finite() && scale > 0.0);
    assert!(input.iter().all(|value| value.is_finite()));
    assert!(weights.iter().all(|&bits| decode(bits).is_finite()));
    let mut result = Expected {
        quantized_dot: Vec::new(),
        unquantized_dot: Vec::new(),
        absolute_sum: Vec::new(),
        activation_bytes: Vec::new(),
        row_scales: Vec::new(),
    };
    for row in input.chunks_exact(k) {
        let maximum = row.iter().map(|x| x.to_f32().abs()).fold(0.0_f32, f32::max);
        let row_scale = (maximum / 448.0).max(1.0 / (448.0 * 512.0));
        result.row_scales.push(row_scale);
        let quantized: Vec<_> = row
            .iter()
            .map(|x| encode_finite((x.to_f32() / row_scale).clamp(-448.0, 448.0)))
            .collect();
        result.activation_bytes.extend(&quantized);
        for weight in weights.chunks_exact(k) {
            let mut q = 0.0;
            let mut original = 0.0;
            let mut sum = 0.0;
            for ((&x, &a), &w) in row.iter().zip(&quantized).zip(weight) {
                let w = decode(w) * f64::from(scale);
                let term = decode(a) * f64::from(row_scale) * w;
                q += term;
                sum += term.abs();
                original += f64::from(x.to_f32()) * w;
            }
            result.quantized_dot.push(q);
            result.unquantized_dot.push(original);
            result.absolute_sum.push(sum);
        }
    }
    result
}

impl Expected {
    /// Validate actual scratch quantization before comparing GEMM. Neither
    /// candidate byte array participates in constructing the expected output.
    pub fn check_quantization(&self, activations: &[u8], scales: &[u8]) -> Result<(), String> {
        if activations.len() != self.activation_bytes.len()
            || scales.len() != self.row_scales.len() * 4
        {
            return Err("FP8 quantization capture extent differs".into());
        }
        for (index, (&actual, &expected)) in
            activations.iter().zip(&self.activation_bytes).enumerate()
        {
            if actual != expected {
                return Err(format!(
                    "activation {index}: GPU=0x{actual:02x}, independent=0x{expected:02x}"
                ));
            }
        }
        for (row, (actual, expected)) in scales.chunks_exact(4).zip(&self.row_scales).enumerate() {
            let bits = u32::from_le_bytes(actual.try_into().unwrap());
            if bits != expected.to_bits() {
                return Err(format!(
                    "row scale {row}: GPU=0x{bits:08x}, independent=0x{:08x}",
                    expected.to_bits()
                ));
            }
        }
        Ok(())
    }
    #[cfg(any(
        feature = "cuda-cutlass-fp8",
        all(feature = "metal", target_os = "macos")
    ))]
    pub fn check(&self, actual: &[u8], k: usize) -> (f64, f64) {
        assert_eq!(actual.len(), self.quantized_dot.len() * 2);
        let mut maximum_error = 0_f64;
        let mut quantization_change = 0_f64;
        // Conservative FP32 accumulate + epilogue rounding and final BF16 RNE.
        // This excludes quantization approximation; that is reported separately.
        let u = 2_f64.powi(-24);
        let gamma = ((2 * k + 8) as f64 * u) / (1.0 - (2 * k + 8) as f64 * u);
        for (i, bytes) in actual.chunks_exact(2).enumerate() {
            let observed =
                f64::from(bf16::from_bits(u16::from_le_bytes([bytes[0], bytes[1]])).to_f32());
            let expected = self.quantized_dot[i];
            let accumulation_bound = gamma * self.absolute_sum[i];
            let tolerance = accumulation_bound
                + (expected.abs() + accumulation_bound) / 256.0
                + 2_f64.powi(-130);
            let error = (observed - expected).abs();
            assert!(observed.is_finite() && error <= tolerance, "index {i}: actual={observed}, quantized F64={expected}, error={error}, tolerance={tolerance}");
            maximum_error = maximum_error.max(error);
            quantization_change =
                quantization_change.max((expected - self.unquantized_dot[i]).abs());
        }
        (maximum_error, quantization_change)
    }
}

#[test]
fn encoding_known_points_ties_saturation_and_signed_zero() {
    for (bits, value) in [
        (0, 0.0),
        (1, 1.0 / 512.0),
        (8, 1.0 / 64.0),
        (56, 1.0),
        (126, 448.0),
        (184, -1.0),
    ] {
        assert_eq!(decode(bits), value);
        assert_eq!(encode_finite(value as f32), bits);
    }
    assert_eq!(encode_finite(-0.0), 128);
    assert_eq!(encode_finite(1.0625), 56);
    assert_eq!(encode_finite(1.1875), 58);
    assert_eq!(encode_finite(1000.0), 126);
    assert!(decode(127).is_nan());
}

#[test]
fn independent_rows_zero_floor_and_non_symmetric_projection() {
    let inputs: Vec<_> = [0.0, 0.0, 1.0, -0.5, 0.001, 0.002]
        .into_iter()
        .map(bf16::from_f32)
        .collect();
    let oracle = expected(&inputs, &[56, 64, 184, 56], 0.5, 2, 2);
    assert_eq!(oracle.row_scales[0], 1.0 / (448.0 * 512.0));
    assert_eq!(&oracle.activation_bytes[..2], &[0, 0]);
    assert_ne!(oracle.row_scales[1], oracle.row_scales[2]);
    assert_eq!(&oracle.quantized_dot[..2], &[0.0, 0.0]);
    assert_ne!(oracle.quantized_dot[2], oracle.quantized_dot[3]);
}

#[test]
fn quantization_comparison_rejects_independent_activation_and_scale_mutations() {
    let inputs: Vec<_> = [0.0, -0.0, 1.0, -0.5]
        .into_iter()
        .map(bf16::from_f32)
        .collect();
    let expected = expected(&inputs, &[56, 64], 0.5, 2, 1);
    let scales: Vec<u8> = expected
        .row_scales
        .iter()
        .flat_map(|s| s.to_le_bytes())
        .collect();
    assert!(expected
        .check_quantization(&expected.activation_bytes, &scales)
        .is_ok());
    let mut changed_a = expected.activation_bytes.clone();
    changed_a[0] ^= 128; // signed zero is part of the actual quantization ABI.
    assert!(expected.check_quantization(&changed_a, &scales).is_err());
    let mut changed_s = scales.clone();
    changed_s[4] ^= 1;
    assert!(expected
        .check_quantization(&expected.activation_bytes, &changed_s)
        .is_err());
    assert!(expected
        .check_quantization(&expected.activation_bytes[..3], &scales)
        .is_err());
}
