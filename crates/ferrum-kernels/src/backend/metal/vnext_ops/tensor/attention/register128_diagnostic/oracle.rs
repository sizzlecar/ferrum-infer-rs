//! Independent scalar FP64 reference for the periodic, exactly BF16 fixture.
//! No GPU fragment layout, tile reduction or running-max recurrence is copied.
use super::family::Config;

pub(super) fn value(index: u64, phase: u64) -> f64 {
    (((index + phase) % 7) as i64 - 3) as f64 / 2.0
}

pub(super) struct Reference {
    scores: [[f64; 7]; 7],
    query_phase: u64,
    key_phase: u64,
}
impl Reference {
    pub(super) fn new(query_phase: u64, key_phase: u64) -> Self {
        let mut scores = [[0.0; 7]; 7];
        for (q, row) in scores.iter_mut().enumerate() {
            for (k, score) in row.iter_mut().enumerate() {
                *score = (0..128)
                    .map(|d| value(q as u64 + d, 0) * value(k as u64 + d, 0))
                    .sum::<f64>()
                    / 8.0;
                // Products are quarter-integers, sum <=288, score is an exact
                // multiple of 1/32. Every FP32 dot partial is representable.
                assert_eq!((*score as f32) as f64, *score);
                assert_eq!((*score * 32.0).fract(), 0.0);
            }
            let max = row.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            assert!(row
                .iter()
                .all(|s| (*s - max).exp() > f32::MIN_POSITIVE as f64));
        }
        Self {
            scores,
            query_phase,
            key_phase,
        }
    }

    pub(super) fn expected(&self, c: Config, row: u64) -> (f64, f64) {
        let batch = row / (c.sequence * c.heads);
        let head = row % c.heads;
        let qp = ((row * 128 + self.query_phase) % 7) as usize;
        let start = ((batch * c.sequence * c.heads + head) * 128 + self.key_phase) % 7;
        let step = (c.heads * 128) % 7;
        let mut counts = [0_u64; 7];
        // Count by position residues, also correct if step==0 (H multiple7).
        for position in 0..7 {
            let count = c.sequence / 7 + u64::from(position < c.sequence % 7);
            counts[((start + position * step) % 7) as usize] += count;
        }
        assert_eq!(counts.iter().sum::<u64>(), c.sequence);
        let maximum = (0..7)
            .filter(|&k| counts[k] != 0)
            .map(|k| self.scores[qp][k])
            .fold(f64::NEG_INFINITY, f64::max);
        let sum = (0..7)
            .map(|k| counts[k] as f64 * (self.scores[qp][k] - maximum).exp())
            .sum();
        (maximum, sum)
    }

    pub(super) fn relative_bound(sequence: u64) -> f64 {
        // Normal exp <=4 ULP: e=2^-20 conservatively bounds relative error,
        // including exp(0); do not assume alpha bit-exact1. u=2^-23 also
        // covers FP32 round-toward-zero. Each positive contribution traverses
        // <=M exp factors and <=2M+16 additions/multiplications (M=ceil(S/16)).
        // The first16 keys contain every reachable period7 phase, so the exact
        // maximum never changes afterward. All exponentials stay normal.
        // Product bound: t=M*e+(2M+16)*u<1 => |error| <= t/(1-t).
        let tiles = sequence.div_ceil(16) as f64;
        let t = tiles * 2_f64.powi(-20) + (2.0 * tiles + 16.0) * 2_f64.powi(-23);
        assert!(t < 1.0, "diagnostic error envelope exhausted");
        (t / (1.0 - t)).next_up()
    }

    pub(super) fn verify(&self, c: Config, bytes: &[u8]) -> f64 {
        assert_eq!(bytes.len() as u64, c.batch * c.sequence * c.heads * 8);
        let relative = Self::relative_bound(c.sequence);
        let mut max_error = 0.0_f64;
        for (row, pair) in bytes.chunks_exact(8).enumerate() {
            let actual_max = f32::from_le_bytes(pair[..4].try_into().unwrap());
            let actual_sum = f32::from_le_bytes(pair[4..].try_into().unwrap());
            let (maximum, sum) = self.expected(c, row as u64);
            assert_eq!(
                actual_max.to_bits(),
                (maximum as f32).to_bits(),
                "row {row} max"
            );
            // Separate FP64 reference allowance for seven libm exp/weighted
            // sum terms; this is not an interval-certified libm implementation.
            let reference_rounding = 64.0 * f64::EPSILON * sum;
            let bound = (sum * relative + reference_rounding).next_up();
            let error = (actual_sum as f64 - sum).abs();
            assert!(
                actual_sum.is_finite() && error <= bound,
                "row {row}: sum={actual_sum}, FP64={sum}, error={error}, bound={bound}"
            );
            max_error = max_error.max(error / sum);
        }
        max_error
    }
}

#[test]
fn phase_counts_match_explicit_scalar_rows_across_batch_head_and_tail() {
    for heads in [2, 7] {
        let c = Config {
            batch: 2,
            sequence: 33,
            heads,
            visibility: super::family::Visibility::AllValidBidirectional,
            mode: super::family::Mode::QkSoftmaxState,
            value_fixture: super::family::ValueFixture::PositivePeriodic,
        };
        let r = Reference::new(1, 3);
        for row in 0..c.batch * c.sequence * c.heads {
            let batch = row / (c.sequence * c.heads);
            let head = row % c.heads;
            let scores: Vec<_> = (0..c.sequence)
                .map(|k| {
                    (0..128)
                        .map(|d| {
                            value(row * 128 + d, 1)
                                * value(((batch * c.sequence + k) * c.heads + head) * 128 + d, 3)
                        })
                        .sum::<f64>()
                        / 8.0
                })
                .collect();
            let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let sum = scores.iter().map(|s| (*s - max).exp()).sum::<f64>();
            let expected = r.expected(c, row);
            assert_eq!(expected.0, max);
            assert!((expected.1 - sum).abs() <= 64.0 * f64::EPSILON * sum);
        }
    }
}

/// Positive, exactly BF16 values vary with key, column, head and batch.
pub(super) fn positive_value(batch: u64, key: u64, head: u64, column: u64) -> f64 {
    (1 + (key + 2 * column + 3 * head + 5 * batch) % 7) as f64 / 4.0
}

pub(super) struct ValueReference {
    config: Config,
    query_phase: u64,
    // (batch, head, query phase, column phase): independent of sequence length
    // and output width. No per-output traversal of all keys or new large buffer.
    expected: Vec<f64>,
}
impl ValueReference {
    pub(super) fn new(c: Config, query_phase: u64, key_phase: u64) -> Self {
        let reference = Reference::new(query_phase, key_phase);
        let count = usize::try_from(
            c.batch
                .checked_mul(c.heads)
                .unwrap()
                .checked_mul(49)
                .unwrap(),
        )
        .unwrap();
        let mut expected = vec![0.0; count];
        for batch in 0..c.batch {
            for head in 0..c.heads {
                let start = ((batch * c.sequence * c.heads + head) * 128 + key_phase) % 7;
                let step = (c.heads * 128) % 7;
                let counts: [u64; 7] = std::array::from_fn(|r| {
                    c.sequence / 7 + u64::from((r as u64) < c.sequence % 7)
                });
                for qp in 0..7 {
                    let scores: [f64; 7] = std::array::from_fn(|r| {
                        reference.scores[qp][((start + r as u64 * step) % 7) as usize]
                    });
                    let maximum = (0..7)
                        .filter(|&r| counts[r] != 0)
                        .map(|r| scores[r])
                        .fold(f64::NEG_INFINITY, f64::max);
                    let weights: [f64; 7] =
                        std::array::from_fn(|r| counts[r] as f64 * (scores[r] - maximum).exp());
                    let denominator = weights.iter().sum::<f64>();
                    for column in 0..7 {
                        let numerator = (0..7)
                            .map(|r| weights[r] * positive_value(batch, r as u64, head, column))
                            .sum::<f64>();
                        expected
                            [((batch * c.heads + head) * 49 + qp as u64 * 7 + column) as usize] =
                            numerator / denominator;
                    }
                }
            }
        }
        Self {
            config: c,
            query_phase,
            expected,
        }
    }

    pub(super) fn expected(&self, row: u64, column: u64) -> f64 {
        let c = self.config;
        let batch = row / (c.sequence * c.heads);
        let head = row % c.heads;
        let qp = (row * 128 + self.query_phase) % 7;
        self.expected[((batch * c.heads + head) * 49 + qp * 7 + column % 7) as usize]
    }

    pub(super) fn relative_bound(sequence: u64) -> f64 {
        // This is a separate positive-finite F32 diagnostic contract, not the
        // production BF16 oracle. QK scores are exact dyadics. All exp arguments
        // are normal and the first tile contains every reachable key phase.
        // Retain the conservative <=M exp errors, including rescale exp(0).
        // u=2^-23 covers RN/RTZ; e=2^-20 conservatively covers normal exp <=4ULP.
        // Numerator: <=2S product/add roundings, <=2M online rescale roundings,
        // and 32 spare roundings. Denominator uses the existing 2M+16 envelope.
        // Positive V avoids cancellation. No overflow/subnormal occurs here.
        // MSL precise reciprocal and multiplication each add one rounding.
        // We do not claim this is an observation of the matrix instruction ISA.
        let m = sequence.div_ceil(16) as f64;
        let u = 2_f64.powi(-23);
        let e = 2_f64.powi(-20);
        let td = (m * e + (2.0 * m + 16.0) * u).next_up();
        let tn = (m * e + (2.0 * sequence as f64 + 2.0 * m + 32.0) * u).next_up();
        assert!(tn < 1.0 && td < 1.0, "diagnostic error envelope exhausted");
        let dn = (tn / (1.0 - tn).next_down()).next_up();
        let dd = (td / (1.0 - td).next_down()).next_up();
        let numerator = ((1.0 + dn).next_up() * (1.0 + u).powi(2).next_up()).next_up();
        (numerator / (1.0 - dd).next_down() - 1.0).next_up()
    }

    pub(super) fn verify(&self, bytes: &[u8]) -> f64 {
        let c = self.config;
        let columns = c.mode.columns();
        assert_eq!(
            bytes.len() as u64,
            c.batch * c.sequence * c.heads * columns * 4
        );
        let relative = Self::relative_bound(c.sequence);
        let mut max_error = 0.0_f64;
        for (i, word) in bytes.chunks_exact(4).enumerate() {
            let actual = f32::from_le_bytes(word.try_into().unwrap());
            let expected = self.expected(i as u64 / columns, i as u64 % columns);
            // FP64 seven-term weighted reference allowance; host libm is not
            // interval-certified. This is separate from the GPU F32 envelope.
            let bound = (expected * (relative + 128.0 * f64::EPSILON)).next_up();
            let error = (actual as f64 - expected).abs();
            assert!(
                actual.is_finite() && error <= bound,
                "output {i}: F32={actual}, FP64={expected}, error={error}, bound={bound}"
            );
            max_error = max_error.max(error / expected);
        }
        max_error
    }
}

#[test]
fn weighted_phase_cache_matches_direct_fp64_rows_for_both_output_widths() {
    use super::family::{Mode, Visibility};
    for heads in [2, 7] {
        let c = Config {
            batch: 2,
            sequence: 33,
            heads,
            visibility: Visibility::AllValidBidirectional,
            mode: Mode::ValueWidth64,
            value_fixture: super::family::ValueFixture::PositivePeriodic,
        };
        for (query_phase, key_phase) in [(1, 3), (5, 0)] {
            let narrow = ValueReference::new(c, query_phase, key_phase);
            let wide = ValueReference::new(
                Config {
                    mode: Mode::ValueWidth128,
                    ..c
                },
                query_phase,
                key_phase,
            );
            for row in 0..c.batch * c.sequence * c.heads {
                let batch = row / (c.sequence * c.heads);
                let head = row % c.heads;
                let scores: Vec<_> = (0..c.sequence)
                    .map(|k| {
                        (0..128)
                            .map(|d| {
                                value(row * 128 + d, query_phase)
                                    * value(
                                        ((batch * c.sequence + k) * c.heads + head) * 128 + d,
                                        key_phase,
                                    )
                            })
                            .sum::<f64>()
                            / 8.0
                    })
                    .collect();
                let maximum = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let weights: Vec<_> = scores
                    .iter()
                    .map(|score| (*score - maximum).exp())
                    .collect();
                let denominator = weights.iter().sum::<f64>();
                for column in 0..128 {
                    let numerator = (0..c.sequence)
                        .map(|k| weights[k as usize] * positive_value(batch, k, head, column))
                        .sum::<f64>();
                    let expected = numerator / denominator;
                    let actual = wide.expected(row, column);
                    assert!((actual - expected).abs() <= 128.0 * f64::EPSILON * expected);
                    if column < 64 {
                        assert_eq!(narrow.expected(row, column), actual);
                    }
                }
            }
        }
    }
    for sequence in [33, 4096] {
        let bound = ValueReference::relative_bound(sequence);
        assert!(bound.is_finite() && bound > Reference::relative_bound(sequence));
    }
}

pub(super) fn constant_value(head: u64, column: u64) -> f64 {
    (((head * 128 + column) % 17) as i64 - 8) as f64 / 16.0
}

pub(super) fn verify_decode_payloads(bytes: &[u8], count: u64) {
    assert_eq!(bytes.len() as u64, count * 4);
    for (i, word) in bytes.chunks_exact(4).enumerate() {
        let input_bits = i as u16;
        let actual = u32::from_le_bytes(word.try_into().unwrap());
        let expected = u32::from(input_bits) << 16;
        if input_bits & 0x7fff > 0x7f80 {
            // Conversion may canonicalize any signaling or quiet NaN payload.
            assert!(
                f32::from_bits(actual).is_nan(),
                "BF16 NaN {input_bits:#06x}"
            );
        } else {
            assert_eq!(actual, expected, "BF16 decode {input_bits:#06x}; includes signed zeros, both subnormal ranges and infinities");
        }
    }
}

impl ValueReference {
    pub(super) fn bf16_relative_bound(c: Config) -> f64 {
        use super::family::Mode;
        let bf16_u = 2_f64.powi(-8); // MSL float->bfloat round-to-nearest/ties-even.
                                     // Both production-source modes use the same finite-fixture arithmetic
                                     // and output rounding. This does not certify native decode over IEEE
                                     // exceptional/subnormal payloads outside this diagnostic's inputs.
        if matches!(c.mode, Mode::ProductionBf16 | Mode::ProductionNativeBf16) {
            return ((1.0 + Self::relative_bound(c.sequence)).next_up() * (1.0 + bf16_u) - 1.0)
                .next_up();
        }
        assert!(matches!(
            c.mode,
            Mode::SteelReference | Mode::CompactDenseBf16
        ));
        // Independent Steel and derived compact-core profiles: both preserve
        // log2e/fast-exp2/division, not the natural-exp<=4ULP profile.
        // The 49 exact natural-score table has |log2 score|<25 and spread<36.
        // u=2^-23: log2e constant+scale+score multiply perturb each score by
        // <4u*25; subtraction adds rounding, giving argument error <251u.
        // MSL fast::exp2 error <=3+floor(|2x|) ULP (<=75 here). Combining its
        // normal-range error and argument perturbation gives eP=2^-14.
        // After the first tile all reachable phases occur; alpha=exp2(0),
        // conservatively bounded by eA=2^-20, not assumed exactly one.
        // Positive numerator/denominator paths have no cancellation. The final
        // original Steel division adds one rounding, then native BF16 RNE.
        let m = c.sequence.div_ceil(16) as f64;
        let u = 2_f64.powi(-23);
        let ep = 2_f64.powi(-14);
        let ea = 2_f64.powi(-20);
        let td = (ep + m * ea + (2.0 * m + 16.0) * u).next_up();
        let tn = (ep + m * ea + (2.0 * c.sequence as f64 + 2.0 * m + 32.0) * u).next_up();
        assert!(
            tn < 1.0 && td < 1.0,
            "Steel diagnostic error envelope exhausted"
        );
        let dn = (tn / (1.0 - tn).next_down()).next_up();
        let dd = (td / (1.0 - td).next_down()).next_up();
        let upper = ((1.0 + dn).next_up() * (1.0 + u)).next_up();
        let upper = (upper / (1.0 - dd).next_down()).next_up();
        (upper * (1.0 + bf16_u) - 1.0).next_up()
    }

    pub(super) fn verify_bf16(&self, bytes: &[u8]) -> f64 {
        use super::family::ValueFixture;
        let c = self.config;
        assert!(c.mode.is_bf16());
        assert_eq!(bytes.len() as u64, c.batch * c.sequence * c.heads * 128 * 2);
        let relative = Self::bf16_relative_bound(c);
        let mut max_error = 0.0_f64;
        for (i, word) in bytes.chunks_exact(2).enumerate() {
            let bits = u16::from_le_bytes(word.try_into().unwrap());
            let row = i as u64 / 128;
            let column = i as u64 % 128;
            if c.value_fixture == ValueFixture::ChannelConstant {
                let expected = constant_value(row % c.heads, column) as f32;
                assert_eq!(expected.to_bits() & 0xffff, 0);
                assert_eq!(
                    bits,
                    (expected.to_bits() >> 16) as u16,
                    "constant-channel BF16 output {i}"
                );
            } else {
                let actual = f32::from_bits((bits as u32) << 16);
                let expected = self.expected(row, column);
                let error = (actual as f64 - expected).abs();
                let bound = (expected * (relative + 128.0 * f64::EPSILON)).next_up();
                assert!(actual.is_finite() && error <= bound,
                    "BF16 output {i}: actual={actual}, FP64={expected}, error={error}, bound={bound}, mode={:?}", c.mode);
                max_error = max_error.max(error / expected);
            }
        }
        max_error
    }
}

#[test]
fn bf16_reference_bound_declares_steel_log2_domain_and_constant_channels() {
    let reference = Reference::new(0, 0);
    for scores in reference.scores {
        assert!(scores
            .iter()
            .all(|v| (v * std::f64::consts::LOG2_E).abs() < 25.0));
        let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        assert!(scores
            .iter()
            .all(|v| (max - v) * std::f64::consts::LOG2_E < 36.0));
    }
    for head in 0..3 {
        for column in 0..128 {
            let value = constant_value(head, column) as f32;
            assert_eq!(value.to_bits() & 0xffff, 0);
        }
    }
    for mode in [
        super::family::Mode::ProductionBf16,
        super::family::Mode::ProductionNativeBf16,
        super::family::Mode::SteelReference,
        super::family::Mode::CompactDenseBf16,
    ] {
        let config = Config {
            batch: 2,
            sequence: 33,
            heads: 2,
            visibility: super::family::Visibility::AllValidBidirectional,
            mode,
            value_fixture: super::family::ValueFixture::PositivePeriodic,
        };
        assert!(ValueReference::bf16_relative_bound(config).is_finite());
    }
}
