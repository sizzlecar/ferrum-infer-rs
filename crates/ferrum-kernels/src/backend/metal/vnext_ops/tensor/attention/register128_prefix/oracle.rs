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
