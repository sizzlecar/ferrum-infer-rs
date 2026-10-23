//! Independent full-key F64 softmax, no provider geometry/kernel/helper calls.
use super::*;

fn bits(value: f32) -> u16 {
    bf16::from_f32(value).to_bits()
}
fn value(bits: u16) -> f64 {
    f64::from(bf16::from_bits(bits).to_f32())
}

pub(super) fn nonuniform(shape: Shape) -> Vec<u16> {
    let mut input = vec![0; 3 * shape.elements()];
    for b in 0..shape.batch as usize {
        for c in 0..shape.channels as usize {
            for s in 0..shape.spatial as usize {
                // Dyadic BF16 inputs, all channels contribute; positions/batches
                // have different Q/K/V to expose wrong strides or split heads.
                input[shape.at(b, 0, c, s)] =
                    bits((((c * 3 + s * 5 + b * 7) % 19) as i32 - 9) as f32 / 16.0);
                input[shape.at(b, 1, c, s)] =
                    bits((((c * 7 + s * 3 + b * 11) % 23) as i32 - 11) as f32 / 16.0);
                input[shape.at(b, 2, c, s)] = bits(
                    (((c * 11 + s * 7 + b * 13) % 29) as i32 - 14) as f32 / 8.0 + b as f32 / 4.0,
                );
            }
        }
    }
    input
}

pub(super) fn reference(shape: Shape, input: &[u16]) -> Vec<f64> {
    let (batch, channels, spatial) = (
        shape.batch as usize,
        shape.channels as usize,
        shape.spatial as usize,
    );
    let mut result = vec![0.0; shape.elements()];
    for b in 0..batch {
        for query in 0..spatial {
            let scores: Vec<f64> = (0..spatial)
                .map(|key| {
                    (0..channels)
                        .map(|c| {
                            value(input[shape.at(b, 0, c, query)])
                                * value(input[shape.at(b, 1, c, key)])
                        })
                        .sum::<f64>()
                        / 8.0
                })
                .collect();
            if scores.iter().any(|score| !score.is_finite()) {
                for c in 0..channels {
                    result[(b * channels + c) * spatial + query] = f64::NAN;
                }
                continue;
            }
            let maximum = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let exponentials: Vec<f64> =
                scores.iter().map(|score| (score - maximum).exp()).collect();
            let denominator: f64 = exponentials.iter().sum();
            for c in 0..channels {
                result[(b * channels + c) * spatial + query] = exponentials
                    .iter()
                    .enumerate()
                    .map(|(key, exp)| exp / denominator * value(input[shape.at(b, 2, c, key)]))
                    .sum();
            }
        }
    }
    result
}

type Case = (&'static str, Shape, Vec<u16>, Vec<f64>);
pub(super) fn nonfinite_cases() -> Vec<Case> {
    let shape = Shape {
        batch: 2,
        channels: 3,
        spatial: 257,
    };
    let mut scores = nonuniform(shape);
    scores[shape.at(0, 0, 0, 0)] = 0x7fc1;
    scores[shape.at(0, 0, 1, 256)] = 0x7f80;
    scores[shape.at(1, 0, 2, 127)] = 0xff80;
    let score_expected = reference(shape, &scores);
    assert_eq!(score_expected.iter().filter(|x| x.is_nan()).count(), 9);

    let mut values = nonuniform(shape);
    values[shape.at(0, 2, 0, 256)] = 0x7fc1;
    values[shape.at(0, 2, 1, 128)] = 0x7f80;
    values[shape.at(0, 2, 2, 0)] = 0xff80;
    let value_expected = reference(shape, &values);
    // Batch one stays finite: nonfinite V in batch zero cannot contaminate it.
    assert!(value_expected[shape.elements() / 2..]
        .iter()
        .all(|x| x.is_finite()));

    // Analytic F32-probability underflow: exp(-256) rounds to zero. IEEE
    // multiplication must still evaluate 0*(+/-Inf)=NaN per affected channel.
    // An F64-only softmax would not underflow here, so this case is separate.
    let underflow_shape = Shape {
        batch: 1,
        channels: 3,
        spatial: 2,
    };
    let mut underflow = vec![0; underflow_shape.elements() * 3];
    for query in 0..2 {
        underflow[underflow_shape.at(0, 0, 0, query)] = bits(1.0);
    }
    underflow[underflow_shape.at(0, 1, 0, 1)] = bits(-2048.0);
    for c in 0..3 {
        underflow[underflow_shape.at(0, 2, c, 0)] = bits(2.0);
    }
    underflow[underflow_shape.at(0, 2, 0, 1)] = 0x7f80;
    underflow[underflow_shape.at(0, 2, 1, 1)] = 0xff80;
    underflow[underflow_shape.at(0, 2, 2, 1)] = bits(7.0);
    vec![
        ("score_ieee", shape, scores, score_expected),
        ("value_ieee", shape, values, value_expected),
        (
            "zero_times_infinity",
            underflow_shape,
            underflow,
            vec![f64::NAN, f64::NAN, f64::NAN, f64::NAN, 2.0, 2.0],
        ),
    ]
}
