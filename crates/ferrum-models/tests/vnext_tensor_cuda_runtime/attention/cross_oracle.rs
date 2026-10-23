//! Small independently prepared F64 mathematical softmax, not vendor arithmetic.
use super::*;
pub(super) const BATCH: usize = 2;
pub(super) const QUERIES: usize = 3;
pub(super) const KEYS: usize = 5;
pub(super) const HEADS: usize = 2;
pub(super) const DIM: usize = 128;
pub(super) const COLUMNS: usize = HEADS * DIM;
// Engineering thresholds fixed before candidate execution. No full-domain
// vendor error guarantee or Torch/Reference bitwise claim is made.
pub(super) const ABSOLUTE: f64 = 1.0 / 256.0;
pub(super) const RELATIVE: f64 = 1.0 / 128.0;

pub(super) struct Probe {
    pub config: CrossConfig,
    pub input: [Vec<u16>; 3],
    pub valid: [[bool; KEYS]; BATCH],
}
impl Probe {
    pub fn new(empty_second: bool) -> Self {
        let valid = [
            [true, false, true, true, true],
            if empty_second {
                [false; KEYS]
            } else {
                [false, true, true, false, true]
            },
        ];
        let (counts, lengths, flags) = if empty_second {
            (vec![3, 1], vec![1, 1, 3, 5], vec![1, 0, 1, 0])
        } else {
            (
                vec![3, 4],
                vec![1, 1, 3, 1, 2, 1, 1],
                vec![1, 0, 1, 0, 1, 0, 1],
            )
        };
        let attributes = [
            ("batch", SemanticValue::Unsigned(BATCH as u64)),
            ("head_dim", SemanticValue::Unsigned(DIM as u64)),
            (
                "scale",
                SemanticValue::Rational(CanonicalRational::new(1, 4).unwrap()),
            ),
            ("key_run_counts", SemanticValue::Integers(counts)),
            ("key_run_lengths", SemanticValue::Integers(lengths)),
            ("key_run_valid", SemanticValue::Integers(flags)),
        ]
        .into_iter()
        .map(|(k, v)| (id(k), v))
        .collect();
        let config = CrossConfig {
            query_rows: (BATCH * QUERIES) as u64,
            key_value_rows: (BATCH * KEYS) as u64,
            columns: COLUMNS as u64,
            attributes,
        };
        let mut q = vec![0; BATCH * QUERIES * COLUMNS];
        let mut k = vec![0; BATCH * KEYS * COLUMNS];
        let mut v = k.clone();
        for (b, valid_keys) in valid.iter().enumerate() {
            for r in 0..QUERIES {
                for h in 0..HEADS {
                    let start = (b * QUERIES + r) * COLUMNS + h * DIM;
                    if !valid_keys.iter().any(|&x| x) {
                        q[start..start + DIM].fill(0x7fc1);
                    } else {
                        q[start] = bf16::from_f32((r as f32 - 1.) * (b + 1) as f32 + h as f32 / 2.)
                            .to_bits();
                    }
                }
            }
            for (r, &included) in valid_keys.iter().enumerate() {
                for h in 0..HEADS {
                    let start = (b * KEYS + r) * COLUMNS + h * DIM;
                    if included {
                        k[start] =
                            bf16::from_f32((r as f32 - 2.) / 4. + b as f32 / 8. + h as f32 / 16.)
                                .to_bits();
                        for c in 0..DIM {
                            v[start + c] = bf16::from_f32(
                                ((b * 17 + r * 5 + h * 3 + c % 11) % 29) as f32 / 8. - 14. / 8.,
                            )
                            .to_bits();
                        }
                    } else {
                        k[start..start + DIM].fill(0x7fc1);
                        v[start..start + DIM].fill(0xff80);
                    }
                }
            }
        }
        Self {
            config,
            input: [q, k, v],
            valid,
        }
    }
    pub fn sanitized(&self) -> [Vec<u16>; 3] {
        let mut input = self.input.clone();
        for (b, valid) in self.valid.iter().enumerate() {
            if !valid.iter().any(|&x| x) {
                input[0][b * QUERIES * COLUMNS..(b + 1) * QUERIES * COLUMNS].fill(0x3f80);
            }
            for (k, &included) in valid.iter().enumerate() {
                if !included {
                    let range = (b * KEYS + k) * COLUMNS..(b * KEYS + k + 1) * COLUMNS;
                    input[1][range.clone()].fill(0x3f80);
                    input[2][range].fill(0xc000);
                }
            }
        }
        input
    }
}

pub(super) struct Expected {
    pub values: Vec<f64>,
    pub empty: Vec<bool>,
}
impl Expected {
    pub fn prepare(probe: &Probe) -> Self {
        let f = |port: usize, index: usize| bf16::from_bits(probe.input[port][index]).to_f64();
        let mut values = vec![0.; BATCH * QUERIES * COLUMNS];
        let mut empty = vec![false; BATCH * QUERIES];
        for b in 0..BATCH {
            for q in 0..QUERIES {
                if !probe.valid[b].iter().any(|&x| x) {
                    empty[b * QUERIES + q] = true;
                    continue;
                }
                for h in 0..HEADS {
                    let qs = (b * QUERIES + q) * COLUMNS + h * DIM;
                    let mut scores = Vec::new();
                    for k in 0..KEYS {
                        if probe.valid[b][k] {
                            let ks = (b * KEYS + k) * COLUMNS + h * DIM;
                            let dot = (0..DIM).map(|c| f(0, qs + c) * f(1, ks + c)).sum::<f64>();
                            scores.push((ks, dot / 4.));
                        }
                    }
                    let maximum = scores
                        .iter()
                        .map(|(_, s)| *s)
                        .fold(f64::NEG_INFINITY, f64::max);
                    let weights: Vec<_> = scores
                        .iter()
                        .map(|(ks, s)| (*ks, libm::exp(s - maximum)))
                        .collect();
                    let denominator = weights.iter().map(|(_, p)| p).sum::<f64>();
                    for c in 0..DIM {
                        values[qs + c] =
                            weights.iter().map(|(ks, p)| p * f(2, ks + c)).sum::<f64>()
                                / denominator;
                        assert!(values[qs + c].is_finite());
                    }
                }
            }
        }
        Self { values, empty }
    }
    pub fn check(&self, actual: &[u16]) -> Result<f64, String> {
        if actual.len() != self.values.len() {
            return Err("cross output length".into());
        }
        let mut maximum = 0.0_f64;
        for (i, (&bits, &expected)) in actual.iter().zip(&self.values).enumerate() {
            if self.empty[i / COLUMNS] {
                if bits != 0 {
                    return Err(format!("empty query must be +0 at {i}, bits={bits:04x}"));
                }
                continue;
            }
            let value = bf16::from_bits(bits).to_f64();
            let error = (value - expected).abs();
            if !value.is_finite() || error > ABSOLUTE + RELATIVE * expected.abs() {
                return Err(format!(
                    "cross[{i}] actual={value} reference={expected} abs_error={error}"
                ));
            }
            maximum = maximum.max(error);
        }
        Ok(maximum)
    }
}
