//! Independent serial F64 reference and explicit error accounting.
//! No CUDA reduction order or benchmark timing is reproduced here.
use super::Shape;
use half::f16;
use serde::Serialize;

#[derive(Clone, Copy)]
pub(super) struct Dot {
    pub(super) value: f64,
    absolute_products: f64,
    accumulation_bound: f64,
}

pub(super) fn reference(shape: Shape, x: &[u16], w: &[u16]) -> Vec<Dot> {
    let x = x
        .iter()
        .map(|&v| f16::from_bits(v).to_f64())
        .collect::<Vec<_>>();
    let mut result = vec![
        Dot {
            value: 0.0,
            absolute_products: 0.0,
            accumulation_bound: 0.0
        };
        shape.rows * shape.outputs
    ];
    // F16 products are exactly representable in F64. This increasing-K loop is
    // independent of both the CUDA warp tree and the opaque cuBLAS algorithm.
    for output in 0..shape.outputs {
        for (k, &bits) in w[output * shape.inputs..][..shape.inputs]
            .iter()
            .enumerate()
        {
            let weight = f16::from_bits(bits).to_f64();
            for row in 0..shape.rows {
                let product = x[row * shape.inputs + k] * weight;
                let dot = &mut result[row * shape.outputs + output];
                dot.value += product;
                dot.absolute_products += product.abs();
            }
        }
    }
    let ku = shape.inputs as f64 * f64::from(f32::EPSILON) * 0.5;
    assert!(ku < 1.0);
    for dot in &mut result {
        dot.accumulation_bound = ku / (1.0 - ku) * dot.absolute_products;
    }
    result
}

fn near_halfway(dot: Dot) -> bool {
    let rounded = f16::from_f64(dot.value);
    if !rounded.is_finite() {
        return false;
    }
    let bits = rounded.to_bits();
    let below = f16::from_bits(if bits == 0 {
        0x8001
    } else if bits & 0x8000 != 0 {
        bits + 1
    } else {
        bits - 1
    });
    let above = f16::from_bits(if bits == 0x8000 {
        1
    } else if bits & 0x8000 != 0 {
        bits - 1
    } else {
        bits + 1
    });
    [below, above].into_iter().any(|neighbor| {
        neighbor.is_finite()
            && (dot.value - (rounded.to_f64() + neighbor.to_f64()) * 0.5).abs()
                <= dot.accumulation_bound
    })
}

#[derive(Default, Serialize)]
pub(super) struct ErrorStats {
    elements: usize,
    rounded_f64_bit_mismatches: usize,
    relative_1e_3_vs_rounded_f64_violations: usize,
    first_relative_violation: Option<usize>,
    near_halfway: usize,
    max_abs_f64: f64,
    rmse_f64: f64,
    rel_l2_f64: Option<f64>,
}

pub(super) fn validate(actual: &[u16], reference: &[Dot]) -> ErrorStats {
    assert_eq!(actual.len(), reference.len());
    let mut stats = ErrorStats {
        elements: actual.len(),
        ..Default::default()
    };
    let mut squared_error = 0.0;
    let mut squared_reference = 0.0;
    for (index, (&bits, &dot)) in actual.iter().zip(reference).enumerate() {
        let value = f16::from_bits(bits).to_f64();
        let rounded = f16::from_f64(dot.value);
        if dot.value.is_nan() {
            assert!(value.is_nan(), "NaN propagation at {index}");
            continue;
        }
        if !rounded.is_finite() {
            assert_eq!(value, rounded.to_f64(), "infinity/overflow at {index}");
            continue;
        }
        assert!(value.is_finite(), "nonfinite output at {index}");
        let error = (value - dot.value).abs();
        // Standard F32 summation bound, followed by one round-to-nearest F16
        // store. This is not substituted for the operation's relative oracle.
        let bound = dot.accumulation_bound
            + 0.00048828125 * (dot.value.abs() + dot.accumulation_bound)
            + f16::from_bits(1).to_f64() * 0.5;
        assert!(
            error <= bound,
            "F64 forward bound at {index}: actual={value}, exact={}, bound={bound}",
            dot.value
        );
        stats.rounded_f64_bit_mismatches += usize::from(bits != rounded.to_bits());
        if (value - rounded.to_f64()).abs() > 0.001 * rounded.to_f64().abs() {
            stats.relative_1e_3_vs_rounded_f64_violations += 1;
            stats.first_relative_violation.get_or_insert(index);
        }
        stats.near_halfway += usize::from(near_halfway(dot));
        stats.max_abs_f64 = stats.max_abs_f64.max(error);
        squared_error += error * error;
        squared_reference += dot.value * dot.value;
    }
    stats.rmse_f64 = (squared_error / actual.len() as f64).sqrt();
    stats.rel_l2_f64 =
        (squared_reference != 0.0).then(|| (squared_error / squared_reference).sqrt());
    stats
}
