use super::{bridge::Bridge, Result};
use half::bf16;
use serde::Serialize;
use serde_json::{json, Value};

pub const SCALE: f32 = 0.125;

#[derive(Debug, Clone, Copy, Serialize)]
pub struct Shape {
    pub batch: usize,
    pub queries: usize,
    pub keys: usize,
    pub heads: usize,
    pub dim: usize,
}

impl Shape {
    pub fn validate(self) -> Result<()> {
        if [self.batch, self.queries, self.keys, self.heads, self.dim]
            .into_iter()
            .any(|x| x == 0 || x > i32::MAX as usize)
        {
            return Err("all dimensions must be positive MLX i32 dimensions".into());
        }
        // These BF16 head dimensions have the upstream full-attention path.
        // Avoid accidentally benchmarking the quadratic unfused fallback.
        if ![64, 80, 96, 128, 256].contains(&self.dim) {
            return Err(
                "head-dim must be one of 64,80,96,128,256 for this full-attention diagnostic"
                    .into(),
            );
        }
        self.host_bytes()?;
        Ok(())
    }

    fn elements(self, sequence: usize) -> Result<usize> {
        [self.batch, sequence, self.heads, self.dim]
            .into_iter()
            .try_fold(1usize, |n, d| n.checked_mul(d))
            .filter(|n| *n <= isize::MAX as usize / 2)
            .ok_or_else(|| "BF16 tensor byte extent overflows addressable memory".into())
    }
    pub fn query_elements(self) -> Result<usize> {
        self.elements(self.queries)
    }
    pub fn key_elements(self) -> Result<usize> {
        self.elements(self.keys)
    }
    pub fn host_bytes(self) -> Result<usize> {
        self.query_elements()?
            .checked_add(self.key_elements()?)
            .and_then(|n| n.checked_mul(4))
            .filter(|n| *n <= isize::MAX as usize)
            .ok_or_else(|| "Rust Q/K/V/output extent overflows".into())
    }
}

pub struct Inputs {
    pub q: Vec<u16>,
    pub k: Vec<u16>,
    pub v: Vec<u16>,
}

pub fn zeroed_words(count: usize) -> Result<Vec<u16>> {
    let mut values = Vec::new();
    values.try_reserve_exact(count)?;
    values.resize(count, 0);
    Ok(values)
}

fn words(count: usize, mut value: impl FnMut(usize) -> f32) -> Result<Vec<u16>> {
    let mut data = zeroed_words(count)?;
    for (i, word) in data.iter_mut().enumerate() {
        *word = bf16::from_f32(value(i)).to_bits();
    }
    Ok(data)
}

fn constant_value(shape: Shape, i: usize) -> u16 {
    bf16::from_f32((((i % (shape.heads * shape.dim)) % 17) as i32 - 8) as f32 / 16.0).to_bits()
}

pub fn benchmark_inputs(shape: Shape) -> Result<Inputs> {
    let qk = |i: usize| ((i % 7) as i32 - 3) as f32 / 2.0;
    Ok(Inputs {
        q: words(shape.query_elements()?, qk)?,
        k: words(shape.key_elements()?, qk)?,
        v: words(shape.key_elements()?, |i| {
            bf16::from_bits(constant_value(shape, i)).to_f32()
        })?,
    })
}

pub fn check_constant_output(shape: Shape, output: &[u16]) -> Result<()> {
    if output.len() != shape.query_elements()? {
        return Err("output extent mismatch".into());
    }
    for (i, &bits) in output.iter().enumerate() {
        let expected = constant_value(shape, i);
        if bits != expected {
            return Err(format!(
                "constant-V exact oracle failed at {i}: BF16 {bits:04x} != {expected:04x}"
            )
            .into());
        }
    }
    Ok(())
}

fn probe_shapes() -> [Shape; 2] {
    [
        Shape {
            batch: 2,
            queries: 3,
            keys: 5,
            heads: 2,
            dim: 128,
        },
        Shape {
            batch: 2,
            queries: 33,
            keys: 17,
            heads: 3,
            dim: 128,
        },
    ]
}

pub fn probe_host_bytes() -> Result<usize> {
    let s = probe_shapes()[1];
    s.query_elements()?
        .checked_mul(8)
        .and_then(|n| n.checked_add(s.host_bytes().ok()?))
        .ok_or_else(|| "layout probe host extent overflows".into())
}

fn probe_inputs(s: Shape) -> Result<Inputs> {
    Ok(Inputs {
        q: words(s.query_elements()?, |i| ((i % 11) as i32 - 5) as f32 / 16.0)?,
        k: words(s.key_elements()?, |i| {
            (((i * 7 + 3) % 13) as i32 - 6) as f32 / 16.0
        })?,
        // Every key, batch and head has distinct values, unlike timed constant V.
        v: words(s.key_elements()?, |i| ((i % 19) as i32 - 9) as f32 / 16.0)?,
    })
}

fn reference(s: Shape, input: &Inputs) -> Vec<f64> {
    let value = |data: &[u16], i| bf16::from_bits(data[i]).to_f64();
    let mut result = vec![0.0; s.query_elements().unwrap()];
    for b in 0..s.batch {
        for q in 0..s.queries {
            for h in 0..s.heads {
                let q_base = ((b * s.queries + q) * s.heads + h) * s.dim;
                let key_base = |k| ((b * s.keys + k) * s.heads + h) * s.dim;
                let scores: Vec<_> = (0..s.keys)
                    .map(|k| {
                        (0..s.dim)
                            .map(|d| value(&input.q, q_base + d) * value(&input.k, key_base(k) + d))
                            .sum::<f64>()
                            * f64::from(SCALE)
                    })
                    .collect();
                let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let weights: Vec<_> = scores.iter().map(|x| (x - max).exp()).collect();
                let denominator: f64 = weights.iter().sum();
                for d in 0..s.dim {
                    result[q_base + d] = (0..s.keys)
                        .map(|k| weights[k] * value(&input.v, key_base(k) + d))
                        .sum::<f64>()
                        / denominator;
                }
            }
        }
    }
    result
}

fn check_reference(expected: &[f64], output: &[u16]) -> Result<f64> {
    if expected.len() != output.len() {
        return Err("probe extent differs".into());
    }
    let mut max_error = 0.0_f64;
    for (i, (&reference, &bits)) in expected.iter().zip(output).enumerate() {
        let actual = bf16::from_bits(bits).to_f64();
        // Predeclared moderate-input layout diagnostic: final BF16 rounding
        // plus 1e-4 absolute arithmetic allowance. Not a full-domain MLX bound,
        // nor a replacement/relaxation of Ferrum's independent attention suite.
        let bound = reference.abs() / 256.0 + 1e-4;
        let error = (actual - reference).abs();
        if !actual.is_finite() || error > bound {
            return Err(format!(
                "FP64 layout probe failed at {i}: {actual} vs {reference}, bound {bound}"
            )
            .into());
        }
        max_error = max_error.max(error);
    }
    Ok(max_error)
}

pub fn layout_probe(bridge: &Bridge, maximum_host_bytes: usize) -> Result<Value> {
    if probe_host_bytes()? > maximum_host_bytes {
        return Err("layout probe exceeds Rust Host bound".into());
    }
    let mut reports = Vec::new();
    for shape in probe_shapes() {
        let input = probe_inputs(shape)?;
        // Construct expectation before any candidate execution.
        let expected = reference(shape, &input);
        let mut output = zeroed_words(shape.query_elements()?)?;
        let mut session = bridge.create(shape, &input)?;
        session.evaluate()?;
        session.copy_output(&mut output)?;
        let max_error = check_reference(&expected, &output)?;
        session.close()?;
        reports.push(json!({"shape":shape,"checked_elements":output.len(),"maximum_absolute_error":max_error,"closed":true}));
    }
    Ok(
        json!({"event":"layout_probe","cases":reports,"bound":"abs_error <= abs(FP64_reference)/256 + 1e-4; moderate finite input fixture only","oracle":"independent Rust FP64 dot/stable softmax/PV with nonconstant V, full output"}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn independent_layout_oracle_detects_head_and_batch_permutation() {
        let s = probe_shapes()[0];
        let expected = reference(s, &probe_inputs(s).unwrap());
        let correct: Vec<_> = expected
            .iter()
            .map(|&x| bf16::from_f64(x).to_bits())
            .collect();
        check_reference(&expected, &correct).unwrap();
        for offset in [s.dim, s.queries * s.heads * s.dim] {
            let mut wrong = correct.clone();
            wrong.rotate_left(offset);
            assert!(check_reference(&expected, &wrong).is_err());
        }
    }

    #[test]
    fn benchmark_has_distinct_nonzero_scores_and_exact_constant_values() {
        let s = Shape {
            batch: 1,
            queries: 2,
            keys: 3,
            heads: 32,
            dim: 128,
        };
        let input = benchmark_inputs(s).unwrap();
        let scores: Vec<f64> = (0..s.keys)
            .map(|k| {
                (0..s.dim)
                    .map(|d| {
                        bf16::from_bits(input.q[d]).to_f64()
                            * bf16::from_bits(input.k[k * s.heads * s.dim + d]).to_f64()
                    })
                    .sum()
            })
            .collect();
        assert!(scores.iter().any(|x| *x != 0.0));
        assert!(scores.windows(2).any(|x| x[0] != x[1]));
        let mut output: Vec<_> = (0..s.query_elements().unwrap())
            .map(|i| constant_value(s, i))
            .collect();
        check_constant_output(s, &output).unwrap();
        output[0] ^= 1;
        assert!(check_constant_output(s, &output).is_err());
    }
}
