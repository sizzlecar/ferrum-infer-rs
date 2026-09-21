//! Test-only reference for the experimental Q4_K × Q8 activation policy.
//! This is not llama Q8_1: activation scales remain F32 and min correction
//! consumes the quantized integer sum, never the original activation sum.

use half::f16;

#[derive(Debug)]
pub(crate) struct PackedRows {
    pub quants: Vec<i8>,
    pub scales: Vec<f32>,
}

pub(crate) fn pack_rows(input: &[f16], rows: usize, inputs: usize) -> PackedRows {
    assert!(rows > 0 && inputs > 0 && inputs.is_multiple_of(32));
    assert_eq!(input.len(), rows.checked_mul(inputs).unwrap());
    let mut quants = Vec::with_capacity(input.len());
    let mut scales = Vec::with_capacity(input.len() / 32);
    for group in input.chunks_exact(32) {
        if group.iter().any(|x| !x.is_finite()) {
            scales.push(f32::NAN);
            quants.extend([0; 32]);
            continue;
        }
        let maximum = group.iter().map(|x| x.to_f32().abs()).fold(0.0, f32::max);
        let scale = maximum / 127.0_f32;
        scales.push(scale);
        quants.extend(group.iter().map(|x| {
            if maximum == 0.0 {
                0
            } else {
                (x.to_f32() / scale).round().clamp(-127.0, 127.0) as i8
            }
        }));
    }
    PackedRows { quants, scales }
}

#[derive(Clone)]
pub(crate) struct Q4Block {
    pub d: f16,
    pub dmin: f16,
    pub scales: [u8; 8],
    pub minima: [u8; 8],
    pub quants: [u8; 256],
}

impl Q4Block {
    pub fn encode(&self) -> [u8; 144] {
        let mut bytes = [0; 144];
        bytes[..2].copy_from_slice(&self.d.to_le_bytes());
        bytes[2..4].copy_from_slice(&self.dmin.to_le_bytes());
        for group in 0..8 {
            let (s, m) = (self.scales[group], self.minima[group]);
            assert!(s <= 63 && m <= 63);
            if group < 4 {
                bytes[4 + group] |= s;
                bytes[8 + group] |= m;
            } else {
                bytes[8 + group] = (s & 15) | ((m & 15) << 4);
                bytes[group] |= (s >> 4) << 6;
                bytes[4 + group] |= (m >> 4) << 6;
            }
        }
        for (i, &q) in self.quants.iter().enumerate() {
            assert!(q <= 15);
            bytes[16 + (i / 64) * 32 + i % 32] |= q << (4 * ((i % 64) / 32));
        }
        bytes
    }

    pub fn ideal_weight(&self, index: usize) -> f64 {
        let group = index / 32;
        f64::from(self.d.to_f32()) * f64::from(self.scales[group]) * f64::from(self.quants[index])
            - f64::from(self.dmin.to_f32()) * f64::from(self.minima[group])
    }

    pub fn strict_weight(&self, index: usize) -> f32 {
        let group = index / 32;
        // Deliberately reproduce the retained coefficient reconstruction;
        // its subtraction rounding is separate from activation quantization.
        (self.d.to_f32() * f32::from(self.scales[group])) * f32::from(self.quants[index])
            - self.dmin.to_f32() * f32::from(self.minima[group])
    }
}

pub(crate) fn fixture_block(column: usize, block: usize) -> Q4Block {
    Q4Block {
        d: f16::from_f32((1 + (column * 7 + block * 3) % 29) as f32 / 4096.0),
        dmin: f16::from_f32((1 + (column * 3 + block * 5) % 17) as f32 / 8192.0),
        scales: std::array::from_fn(|g| ((g * 13 + column * 7 + block * 3) % 64) as u8),
        minima: std::array::from_fn(|g| ((g * 19 + column * 3 + block * 11) % 64) as u8),
        quants: std::array::from_fn(|i| ((i * 7 + i / 32 + column + block * 3) % 16) as u8),
    }
}

#[derive(Debug)]
pub(crate) struct DotReference {
    pub policy: f64,
    pub strict_original: f64,
    pub strict_quantized: f64,
    pub strict_abs_terms: f64,
    /// Before coefficient/min cancellation, to bound the factored GPU path.
    pub expanded_abs_terms: f64,
    pub activation_error_bound: f64,
}

pub(crate) fn dot_reference(input: &[f16], blocks: &[Q4Block]) -> DotReference {
    assert_eq!(input.len(), blocks.len() * 256);
    let packed = pack_rows(input, 1, input.len());
    let mut result = DotReference {
        policy: 0.0,
        strict_original: 0.0,
        strict_quantized: 0.0,
        strict_abs_terms: 0.0,
        expanded_abs_terms: 0.0,
        activation_error_bound: 0.0,
    };
    for (block_index, block) in blocks.iter().enumerate() {
        for group in 0..8 {
            let start = block_index * 256 + group * 32;
            let delta = f64::from(packed.scales[start / 32]);
            let a = f64::from(block.d.to_f32()) * f64::from(block.scales[group]);
            let b = f64::from(block.dmin.to_f32()) * f64::from(block.minima[group]);
            let mut dot = 0_i64;
            let mut sum = 0_i64;
            for lane in 0..32 {
                let qw = i64::from(block.quants[group * 32 + lane]);
                let qx = i64::from(packed.quants[start + lane]);
                dot += qw * qx;
                sum += qx;
                let x = f64::from(input[start + lane].to_f32());
                let xhat = delta * qx as f64;
                let w = f64::from(block.strict_weight(group * 32 + lane));
                result.strict_original += w * x;
                result.strict_abs_terms += (w * x).abs();
                result.strict_quantized += w * xhat;
                result.expanded_abs_terms +=
                    (delta * a * qw as f64 * qx as f64).abs() + (delta * b * qx as f64).abs();
                result.activation_error_bound += w.abs() * (xhat - x).abs();
            }
            assert!(i32::try_from(dot).is_ok() && i32::try_from(sum).is_ok());
            result.policy += delta * (a * dot as f64 - b * sum as f64);
        }
    }
    result
}

#[test]
fn q4k_q8_pack_has_signed_ties_zero_and_f32_scale() {
    let mut input = vec![f16::NEG_ZERO; 3 * 32];
    for (i, x) in [127.0, -127.0, 0.5, -0.5, 1.5, -1.5, 126.5, -126.5]
        .into_iter()
        .enumerate()
    {
        input[32 + i] = f16::from_f32(x);
    }
    input[64] = f16::ONE;
    input[65] = f16::from_f32(1.0 / 256.0);
    let packed = pack_rows(&input, 3, 32);
    assert_eq!(packed.scales[0].to_bits(), 0.0_f32.to_bits());
    assert_eq!(&packed.quants[..32], &[0; 32]);
    assert_eq!(packed.scales[1], 1.0);
    assert_eq!(
        &packed.quants[32..40],
        &[127, -127, 1, -1, 2, -2, 127, -127]
    );
    assert_eq!(packed.scales[2].to_bits(), (1.0_f32 / 127.0).to_bits());
    assert_ne!(packed.scales[2], f16::from_f32(packed.scales[2]).to_f32());
    assert_eq!(&packed.quants[64..66], &[127, 0]);
}

#[test]
fn q4k_q8_pack_handles_f16_extremes_and_marks_nonfinite_groups() {
    let mut input = vec![f16::ZERO; 4 * 32];
    input[0] = f16::from_bits(1);
    input[1] = -f16::from_bits(1);
    input[32] = f16::MAX;
    input[33] = f16::MIN;
    input[34] = f16::from_bits(1);
    input[64] = f16::NAN;
    input[65] = f16::ONE;
    input[96] = f16::INFINITY;
    input[97] = f16::NEG_INFINITY;
    let p = pack_rows(&input, 1, input.len());
    assert!(p.scales[0].is_normal() && p.scales[0] > 0.0);
    assert_eq!(&p.quants[..2], &[127, -127]);
    assert!(p.scales[1].is_finite());
    assert_eq!(&p.quants[32..35], &[127, -127, 0]);
    assert!(p.scales[2..].iter().all(|d| d.is_nan()));
    assert!(p.quants[64..].iter().all(|&q| q == 0));
}

#[test]
#[should_panic]
fn q4k_q8_pack_rejects_partial_groups() {
    pack_rows(&[f16::ONE; 31], 1, 31);
}

#[test]
fn q4k_q8_min_correction_uses_quantized_integer_sum() {
    let block = Q4Block {
        d: f16::ONE,
        dmin: f16::ONE,
        scales: [1; 8],
        minima: [1; 8],
        quants: [0; 256],
    };
    let mut x = vec![f16::ZERO; 256];
    x[0] = f16::ONE;
    x[1] = f16::from_f32(1.0 / 256.0);
    let r = dot_reference(&x, &[block]);
    let expected = -f64::from(1.0_f32 / 127.0) * 127.0;
    assert_eq!(r.policy, expected);
    assert_eq!(r.strict_original, -257.0 / 256.0);
    assert_ne!(r.policy, r.strict_original);
    assert_eq!(r.policy, r.strict_quantized);
    assert!((r.policy - r.strict_original).abs() <= r.activation_error_bound);
}

#[test]
fn q4k_q8_reference_separates_reconstruction_and_quantization_errors() {
    let mut blocks = vec![fixture_block(3, 0), fixture_block(3, 1)];
    blocks[0].d = f16::from_bits(1);
    blocks[0].dmin = f16::from_f32(63.0);
    let x: Vec<_> = (0..512)
        .map(|i| {
            f16::from_f32(if i % 2 == 0 {
                (i % 29 + 1) as f32 / 32.0
            } else {
                -(i % 31 + 1) as f32 / 32.0
            })
        })
        .collect();
    let packed = pack_rows(&x, 1, x.len());
    let r = dot_reference(&x, &blocks);
    let independent = x
        .iter()
        .enumerate()
        .map(|(i, _)| {
            blocks[i / 256].ideal_weight(i % 256)
                * f64::from(packed.scales[i / 32])
                * f64::from(packed.quants[i])
        })
        .sum::<f64>();
    let tolerance = 512.0 * f64::EPSILON * r.expanded_abs_terms;
    assert!((r.policy - independent).abs() <= tolerance);
    let reconstruction = r.policy - r.strict_quantized;
    let activation = r.strict_quantized - r.strict_original;
    assert_ne!(reconstruction, 0.0);
    assert_ne!(activation, 0.0);
    assert!((activation.abs() - r.activation_error_bound) <= tolerance);
    assert!((r.policy - r.strict_original - reconstruction - activation).abs() <= tolerance);
    for block in &blocks {
        let mut decoded = [0.0_f32; 256];
        super::GgufBlockFormat::Q4K
            .decode(&block.encode(), &mut decoded)
            .unwrap();
        for (i, value) in decoded.iter().enumerate() {
            assert_eq!(value.to_bits(), block.strict_weight(i).to_bits());
        }
    }
}

#[test]
fn q4k_q8_cancellation_bound_retains_both_expanded_terms() {
    let block = Q4Block {
        d: f16::ONE,
        dmin: f16::ONE,
        scales: [1; 8],
        minima: [15; 8],
        quants: [15; 256],
    };
    let r = dot_reference(&[f16::ONE; 256], &[block]);
    assert_eq!(r.policy, 0.0);
    assert_eq!(r.strict_original, 0.0);
    assert!(r.expanded_abs_terms > 7000.0);
}
