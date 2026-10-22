//! Independent F64 mathematical reference for the bounded GPU probes.
//!
//! This computes a complete stable softmax, not the provider's online update,
//! packing, mask helper or libdevice implementation. The fixed acceptance
//! criterion is an engineering threshold for these samples, not a proof of a
//! uniform vendor error bound or a claim of IEEE-profile bitwise equivalence.
use super::*;

pub(super) const SEQUENCE: usize = 7;
pub(super) const HEADS: usize = 2;
pub(super) const HEAD_DIM: usize = 128;
pub(super) const COLUMNS: usize = HEADS * HEAD_DIM;
pub(super) const ABSOLUTE_TOLERANCE: f64 = 1.0 / 256.0;
pub(super) const RELATIVE_TOLERANCE: f64 = 1.0 / 128.0;

pub(super) struct Probe {
    pub config: AttentionConfig,
    pub inputs: [Vec<u16>; 3],
    pub valid_keys: Vec<[bool; SEQUENCE]>,
}

impl Probe {
    pub fn segmented() -> Self {
        let valid_keys = vec![
            [true, false, true, true, false, true, true],
            [false, true, false, true, true, false, true],
            [false, false, false, false, false, false, true],
        ];
        // The first bidirectional range has one empty sample; the second
        // has three nonempty samples with different packed KV lengths.
        let cfg = config(
            3 * SEQUENCE as u64,
            COLUMNS as u64,
            SEQUENCE as u64,
            HEAD_DIM as u64,
            [
                vec![2, 3, 2],
                vec![0, 1, 1],
                vec![5, 6, 2],
                vec![1, 1, 2, 1, 2, 1, 1, 1, 2, 1, 1, 6, 1],
                vec![1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            ],
        );
        Self::new(cfg, valid_keys)
    }

    pub fn empty() -> Self {
        Self::new(
            config(
                2 * SEQUENCE as u64,
                COLUMNS as u64,
                SEQUENCE as u64,
                HEAD_DIM as u64,
                [
                    vec![2, 3, 2],
                    vec![0, 1, 1],
                    vec![1, 1],
                    vec![7, 7],
                    vec![0, 0],
                ],
            ),
            vec![[false; SEQUENCE]; 2],
        )
    }

    fn new(mut config: AttentionConfig, valid_keys: Vec<[bool; SEQUENCE]>) -> Self {
        config.attributes.insert(
            id("scale"),
            SemanticValue::Rational(CanonicalRational::new(1, 8).unwrap()),
        );
        let elements = valid_keys.len() * SEQUENCE * COLUMNS;
        let mut inputs = std::array::from_fn(|_| vec![0; elements]);
        for batch in 0..valid_keys.len() {
            for row in 0..SEQUENCE {
                let has_keys = (0..query_end(row)).any(|key| valid_keys[batch][key]);
                for head in 0..HEADS {
                    for channel in 0..HEAD_DIM {
                        let index = ((batch * SEQUENCE + row) * HEADS + head) * HEAD_DIM + channel;
                        let q =
                            (((row + head * 2 + batch) % 3) as i32 - 1) * 2 + (channel % 2) as i32;
                        let k = (((row * 2 + head + batch) % 5) as i32 - 2) * 2
                            + ((channel / 2) % 2) as i32;
                        let v = ((row * 3 + head * 5 + channel * 2 + batch * 7) % 17) as i32 - 8;
                        inputs[0][index] = if has_keys {
                            bits(q as f32 / 16.0)
                        } else {
                            0x7fc1
                        };
                        inputs[1][index] = if valid_keys[batch][row] {
                            bits(k as f32 / 16.0)
                        } else if channel % 2 == 0 {
                            0x7f80 // masked +Inf
                        } else {
                            0x7fc1 // masked NaN
                        };
                        inputs[2][index] = if valid_keys[batch][row] {
                            bits(v as f32 / 8.0)
                        } else if channel % 2 == 0 {
                            0xff80 // masked -Inf
                        } else {
                            0x7fc1
                        };
                    }
                }
            }
        }
        Self {
            config,
            inputs,
            valid_keys,
        }
    }

    pub fn sanitized_inputs(&self) -> [Vec<u16>; 3] {
        let mut inputs = self.inputs.clone();
        for (batch, valid) in self.valid_keys.iter().enumerate() {
            for (row, included) in valid.iter().copied().enumerate() {
                if !included {
                    let start = (batch * SEQUENCE + row) * COLUMNS;
                    // Replace masked K/V with conspicuous finite values. Keep
                    // Q unchanged: invalid keys are still real output queries.
                    inputs[1][start..start + COLUMNS].fill(bits(4.0));
                    inputs[2][start..start + COLUMNS].fill(bits(-4.0));
                }
            }
        }
        inputs
    }
}

fn bits(value: f32) -> u16 {
    bf16::from_f32(value).to_bits()
}
fn value(bits: u16) -> f64 {
    f64::from(bf16::from_bits(bits).to_f32())
}
fn query_end(query: usize) -> usize {
    // Independent fixture rule: two causal prefix positions, followed by
    // bidirectional blocks [2,5) and [5,7). Never call the provider's mask.
    match query {
        0..=1 => query + 1,
        2..=4 => 5,
        _ => 7,
    }
}

pub(super) struct Expected {
    pub values: Vec<f64>,
    pub empty_rows: Vec<bool>,
}
#[derive(Debug)]
pub(super) struct Statistics {
    pub maximum_absolute_error: f64,
    pub mean_absolute_error: f64,
    pub maximum_error_to_limit: f64,
    pub compared_elements: usize,
}

impl Expected {
    pub fn prepare(probe: &Probe) -> Self {
        let mut values = vec![0.0; probe.inputs[0].len()];
        let mut empty_rows = vec![false; probe.valid_keys.len() * SEQUENCE];
        for batch in 0..probe.valid_keys.len() {
            for query in 0..SEQUENCE {
                let keys: Vec<usize> = (0..query_end(query))
                    .filter(|&key| probe.valid_keys[batch][key])
                    .collect();
                empty_rows[batch * SEQUENCE + query] = keys.is_empty();
                if keys.is_empty() {
                    continue; // Do not inspect even Q for an empty query.
                }
                for head in 0..HEADS {
                    let at = |row, channel| {
                        ((batch * SEQUENCE + row) * HEADS + head) * HEAD_DIM + channel
                    };
                    let scores: Vec<f64> = keys
                        .iter()
                        .map(|&key| {
                            let dot: f64 = (0..HEAD_DIM)
                                .map(|channel| {
                                    let q = value(probe.inputs[0][at(query, channel)]);
                                    let k = value(probe.inputs[1][at(key, channel)]);
                                    assert!(q.is_finite() && k.is_finite());
                                    q * k
                                })
                                .sum();
                            let score = dot / 8.0;
                            assert!(score.abs() <= 2.0, "probe leaves its declared score domain");
                            score
                        })
                        .collect();
                    let maximum = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                    let weights: Vec<f64> = scores
                        .iter()
                        .map(|score| libm::exp(score - maximum))
                        .collect();
                    let denominator: f64 = weights.iter().sum();
                    for channel in 0..HEAD_DIM {
                        let numerator: f64 = keys
                            .iter()
                            .zip(&weights)
                            .map(|(&key, weight)| {
                                let v = value(probe.inputs[2][at(key, channel)]);
                                assert!(v.is_finite() && v.abs() <= 1.0);
                                weight * v
                            })
                            .sum();
                        values[at(query, channel)] = numerator / denominator;
                    }
                }
            }
        }
        Self { values, empty_rows }
    }

    pub fn check(&self, actual: &[u16]) -> Result<Statistics, String> {
        if actual.len() != self.values.len() {
            return Err("fused attention output length differs from the prepared oracle".into());
        }
        let mut stats = Statistics {
            maximum_absolute_error: 0.0,
            mean_absolute_error: 0.0,
            maximum_error_to_limit: 0.0,
            compared_elements: actual.len(),
        };
        let mut failures = 0;
        let mut first_failure = None;
        for (index, (&actual_bits, &expected)) in actual.iter().zip(&self.values).enumerate() {
            let actual = value(actual_bits);
            let difference = (actual - expected).abs();
            let limit = ABSOLUTE_TOLERANCE + RELATIVE_TOLERANCE * expected.abs();
            let empty = self.empty_rows[index / COLUMNS];
            if !actual.is_finite() || (empty && actual_bits != 0) || difference > limit {
                failures += 1;
                first_failure.get_or_insert((index, actual_bits, expected, limit));
            }
            if actual.is_finite() {
                stats.maximum_absolute_error = stats.maximum_absolute_error.max(difference);
                stats.mean_absolute_error += difference;
                stats.maximum_error_to_limit = stats.maximum_error_to_limit.max(difference / limit);
            }
        }
        stats.mean_absolute_error /= actual.len() as f64;
        if failures != 0 {
            return Err(format!(
                "{failures} fused attention mismatches; first={first_failure:?}; stats={stats:?}"
            ));
        }
        Ok(stats)
    }
}

#[test]
fn independent_reference_rejects_changed_output_nonfinite_and_negative_empty_zero() {
    let probe = Probe::segmented();
    let expected = Expected::prepare(&probe);
    let mut actual: Vec<u16> = expected.values.iter().map(|&x| bits(x as f32)).collect();
    expected.check(&actual).unwrap();
    let index = expected.values.iter().position(|x| x.abs() > 0.1).unwrap();
    let original = actual[index];
    actual[index] = bits(value(original) as f32 + 0.5);
    assert!(expected.check(&actual).is_err());
    actual[index] = 0x7fc1;
    assert!(expected.check(&actual).is_err());
    actual[index] = original;
    let empty = expected.empty_rows.iter().position(|&x| x).unwrap() * COLUMNS;
    actual[empty] = 0x8000;
    assert!(expected.check(&actual).is_err());
}

#[test]
fn independent_reference_uses_nonuniform_scores_and_keeps_masked_key_queries() {
    let probe = Probe::segmented();
    let expected = Expected::prepare(&probe);
    assert_eq!(
        expected.empty_rows,
        [
            false, false, false, false, false, false, false, true, false, false, false, false,
            false, false, true, true, true, true, true, false, false
        ]
    );
    // Batch zero's key 1 is masked, but its query has visible key 0.
    assert!(!probe.valid_keys[0][1]);
    assert!(expected.values[COLUMNS..2 * COLUMNS]
        .iter()
        .any(|x| x.abs() > 0.1));
    // Queries 0 and 1 in sample 0 see only key 0, independently fixing a
    // complete pair of output rows without using any exponential calculation.
    for channel in 0..COLUMNS {
        assert_eq!(expected.values[channel], value(probe.inputs[2][channel]));
        assert_eq!(
            expected.values[COLUMNS + channel],
            value(probe.inputs[2][channel])
        );
    }
    // Replacing Q/K by zero gives a different mathematical result; these are
    // not solely uniform-score means or an all-zero correctness fixture.
    let mut uniform = Probe::segmented();
    uniform.inputs[0].fill(0);
    uniform.inputs[1].fill(0);
    let means = Expected::prepare(&uniform);
    assert!(expected
        .values
        .iter()
        .zip(means.values)
        .any(|(x, y)| (x - y).abs() > 0.01));
    // This full-empty case deliberately contains nonfinite Q/K/V everywhere.
    let empty = Expected::prepare(&Probe::empty());
    assert!(empty.values.iter().all(|&x| x == 0.0));
    empty.check(&vec![0; empty.values.len()]).unwrap();
}
