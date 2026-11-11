use rayon::prelude::*;
use serde::Serialize;

pub const CLASSES: usize = 8;
pub const ABS_TOLERANCE: f64 = 2e-5;
pub const REL_TOLERANCE: f64 = 2e-5;

#[derive(Clone, Copy, Debug, Serialize)]
pub struct Shape {
    pub batches: usize,
    pub sequence: usize,
    pub heads: usize,
    pub dimension: usize,
}

impl Shape {
    pub fn elements(self) -> usize {
        self.batches * self.sequence * self.heads * self.dimension
    }
    pub fn offset(self, batch: usize, row: usize, head: usize) -> usize {
        ((batch * self.sequence + row) * self.heads + head) * self.dimension
    }
}

pub struct Fixture {
    pub shape: Shape,
    pub scale: f32,
    pub query: Vec<f32>,
    pub key: Vec<f32>,
    pub value: Vec<f32>,
}

// Full f32 mantissas, deterministic and finite. Every K/V row and head differs;
// only Q rows intentionally repeat, permitting exact mathematical oracle reuse.
fn sample(index: usize, salt: u32) -> f32 {
    let mut x = (index as u32).wrapping_add(salt);
    x ^= x >> 16;
    x = x.wrapping_mul(0x7feb_352d);
    x ^= x >> 15;
    x = x.wrapping_mul(0x846c_a68b);
    x ^= x >> 16;
    ((x >> 8) as f32 - 8_388_608.0) / 8_388_608.0
}

impl Fixture {
    pub fn new(shape: Shape) -> Self {
        assert!(shape.sequence >= CLASSES && shape.dimension > 0);
        let mut query = vec![0.0; shape.elements()];
        for b in 0..shape.batches {
            for row in 0..shape.sequence {
                for h in 0..shape.heads {
                    let offset = shape.offset(b, row, h);
                    let identity =
                        ((b * shape.heads + h) * CLASSES + row % CLASSES) * shape.dimension;
                    for column in 0..shape.dimension {
                        query[offset + column] = 2.0 * sample(identity + column, 0x361b_7a19);
                    }
                }
            }
        }
        let key = (0..shape.elements())
            .map(|i| 2.0 * sample(i, 0x94d0_49bb))
            .collect();
        let value = (0..shape.elements())
            .map(|i| sample(i, 0x859a_3151))
            .collect();
        Self {
            shape,
            scale: 1.0 / (shape.dimension as f32).sqrt(),
            query,
            key,
            value,
        }
    }

    // Independent F64 dot, stable softmax and weighted sum. Consume actual
    // uploaded f32 values and scale; no CUTLASS or strict-F32 kernel internals.
    fn reference_row(&self, b: usize, row: usize, h: usize) -> (Vec<f64>, f64, f64) {
        let s = self.shape;
        let q = s.offset(b, row, h);
        let mut probabilities = Vec::with_capacity(s.sequence);
        for key in 0..s.sequence {
            let offset = s.offset(b, key, h);
            let mut dot = 0.0_f64;
            for d in 0..s.dimension {
                dot += f64::from(self.query[q + d]) * f64::from(self.key[offset + d]);
            }
            probabilities.push(dot * f64::from(self.scale));
        }
        let maximum = probabilities
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        let minimum = probabilities.iter().copied().fold(f64::INFINITY, f64::min);
        let mut denominator = 0.0;
        for p in &mut probabilities {
            *p = (*p - maximum).exp();
            denominator += *p;
        }
        let mut output = vec![0.0_f64; s.dimension];
        let mut uniform = vec![0.0_f64; s.dimension];
        for (key, probability) in probabilities.iter().enumerate() {
            let offset = s.offset(b, key, h);
            let weight = probability / denominator;
            for d in 0..s.dimension {
                let value = f64::from(self.value[offset + d]);
                output[d] += weight * value;
                uniform[d] += value / s.sequence as f64;
            }
        }
        let uniform_difference = output
            .iter()
            .zip(uniform)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        (output, maximum - minimum, uniform_difference)
    }

    pub fn reference(&self) -> Reference {
        let s = self.shape;
        let rows: Vec<_> = (0..s.batches * s.heads * CLASSES)
            .into_par_iter()
            .map(|identity| {
                let class = identity % CLASSES;
                let head = (identity / CLASSES) % s.heads;
                let batch = identity / (CLASSES * s.heads);
                self.reference_row(batch, class, head)
            })
            .collect();
        Reference {
            minimum_score_range: rows.iter().map(|r| r.1).fold(f64::INFINITY, f64::min),
            minimum_uniform_difference: rows.iter().map(|r| r.2).fold(f64::INFINITY, f64::min),
            classes: rows.into_iter().flat_map(|r| r.0).collect(),
            shape: s,
        }
    }
}

pub struct Reference {
    shape: Shape,
    classes: Vec<f64>,
    pub minimum_score_range: f64,
    pub minimum_uniform_difference: f64,
}

#[derive(Debug, Serialize)]
pub struct Accuracy {
    pub elements: usize,
    pub nonfinite: usize,
    pub outside_tolerance: usize,
    pub maximum_absolute_error: f64,
    pub maximum_normalized_error: f64,
    pub root_mean_square_error: f64,
    pub maximum_reference_magnitude: f64,
    pub minimum_score_range: f64,
    pub minimum_uniform_difference: f64,
    pub absolute_tolerance: f64,
    pub relative_tolerance: f64,
    pub passed: bool,
}

impl Reference {
    fn get(&self, b: usize, row: usize, h: usize, d: usize) -> f64 {
        self.classes
            [(((b * self.shape.heads + h) * CLASSES + row % CLASSES) * self.shape.dimension) + d]
    }
    pub fn compare(&self, output: &[f32]) -> Accuracy {
        let s = self.shape;
        assert_eq!(output.len(), s.elements());
        let mut a = Accuracy {
            elements: output.len(),
            nonfinite: 0,
            outside_tolerance: 0,
            maximum_absolute_error: 0.0,
            maximum_normalized_error: 0.0,
            root_mean_square_error: 0.0,
            maximum_reference_magnitude: 0.0,
            minimum_score_range: self.minimum_score_range,
            minimum_uniform_difference: self.minimum_uniform_difference,
            absolute_tolerance: ABS_TOLERANCE,
            relative_tolerance: REL_TOLERANCE,
            passed: false,
        };
        for b in 0..s.batches {
            for row in 0..s.sequence {
                for h in 0..s.heads {
                    for d in 0..s.dimension {
                        let reference = self.get(b, row, h, d);
                        a.maximum_reference_magnitude =
                            a.maximum_reference_magnitude.max(reference.abs());
                        let value = output[s.offset(b, row, h) + d];
                        if !value.is_finite() {
                            a.nonfinite += 1;
                            continue;
                        }
                        let error = (f64::from(value) - reference).abs();
                        let normalized = error / (ABS_TOLERANCE + REL_TOLERANCE * reference.abs());
                        a.maximum_absolute_error = a.maximum_absolute_error.max(error);
                        a.maximum_normalized_error = a.maximum_normalized_error.max(normalized);
                        a.root_mean_square_error += error * error;
                        a.outside_tolerance += usize::from(normalized > 1.0);
                    }
                }
            }
        }
        a.root_mean_square_error = (a.root_mean_square_error / output.len() as f64).sqrt();
        a.passed = a.nonfinite == 0
            && a.outside_tolerance == 0
            && a.minimum_score_range > 0.1
            && a.minimum_uniform_difference > 10.0 * ABS_TOLERANCE
            && a.maximum_reference_magnitude > 10.0 * ABS_TOLERANCE;
        a
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn reused_query_reference_matches_every_row_and_rejects_zero_and_nonfinite() {
        let f = Fixture::new(Shape {
            batches: 2,
            sequence: 17,
            heads: 3,
            dimension: 9,
        });
        let r = f.reference();
        let s = f.shape;
        let mut output = vec![0.0; s.elements()];
        for b in 0..s.batches {
            for row in 0..s.sequence {
                for h in 0..s.heads {
                    let (direct, _, _) = f.reference_row(b, row, h);
                    for (d, expected) in direct.into_iter().enumerate() {
                        assert_eq!(r.get(b, row, h, d).to_bits(), expected.to_bits());
                        output[s.offset(b, row, h) + d] = expected as f32;
                    }
                }
            }
        }
        assert!(r.compare(&output).passed);
        assert!(!r.compare(&vec![0.0; output.len()]).passed);
        output[13] = f32::NAN;
        assert_eq!(r.compare(&output).nonfinite, 1);
        assert!(!r.compare(&output).passed);
    }
}
