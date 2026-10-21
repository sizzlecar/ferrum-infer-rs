//! Predeclared one-row basis oracle. No candidate values produce expectations.
//! Pure interval and integer-rounding helpers are shared with primitive qualification.
#[path = "../../../ferrum-kernels/src/backend/reference/transcendental/oracle/allowed.rs"]
mod allowed;
#[path = "../../../ferrum-kernels/src/backend/reference/transcendental/oracle/interval.rs"]
mod interval;
#[path = "../../../ferrum-kernels/src/backend/reference/transcendental/oracle/rounding.rs"]
mod rounding;

use anyhow::{bail, Context, Result};
use ferrum_execution::TensorExecutionOutput;
use ferrum_models::qwen_image21::QwenImage21Block0MlpFamily as Family;
use serde::Serialize;
use serde_json::{json, Value};
use std::collections::BTreeSet;

fn bits(bytes: &[u8], index: usize) -> u16 {
    u16::from_le_bytes([bytes[2 * index], bytes[2 * index + 1]])
}
fn value(bits: u16) -> f64 {
    f64::from(f32::from_bits(u32::from(bits) << 16))
}
fn up(value: f64) -> Result<f64> {
    let value = if value == 0.0 { value } else { value.next_up() };
    if !value.is_finite() || value < 0.0 {
        bail!("oracle bound is not nonnegative finite");
    }
    Ok(value)
}

fn activation(gate: u16, up_bits: u16) -> Result<Vec<u16>> {
    // An exact real zero does not determine a backend reduction's zero sign.
    // Enumerate both signs before propagating the actual FP32/BF16 stages.
    let gates = [gate, gate ^ 0x8000];
    let ups = [up_bits, up_bits ^ 0x8000];
    let gate_count = if gate & 0x7fff == 0 { 2 } else { 1 };
    let up_count = if up_bits & 0x7fff == 0 { 2 } else { 1 };
    let mut results = BTreeSet::new();
    for &g in &gates[..gate_count] {
        for &u in &ups[..up_count] {
            results.extend(activation_one(g, u)?);
        }
    }
    Ok(results.into_iter().collect())
}

fn activation_one(gate: u16, up_bits: u16) -> Result<Vec<u16>> {
    let x = -value(gate);
    let exponential = if x.abs() <= 1.0 / 33_554_432.0 {
        allowed::AllowedF32::exact(0x3f80_0000)
    } else if x >= 128.0 {
        allowed::AllowedF32::exact(0x7f80_0000)
    } else if x <= -256.0 {
        allowed::AllowedF32::exact(0)
    } else {
        allowed::AllowedF32::from_interval(interval::exp(x).map_err(anyhow::Error::msg)?)
            .map_err(anyhow::Error::msg)?
    };
    let mut result = BTreeSet::new();
    exponential
        .try_for_each(|exp| {
            let denominator = rounding::one_plus_positive(exp.bits())?;
            let quotient = rounding::divide_bf16_by_positive_f32(gate, denominator)?;
            let rounded = rounding::bf16_from_f32(quotient);
            result.insert(rounding::multiply_bf16(rounded, up_bits));
            Ok(())
        })
        .map_err(anyhow::Error::msg)?;
    if result.is_empty() || result.iter().any(|&bits| !value(bits).is_finite()) {
        bail!("activation oracle has no finite allowed output set");
    }
    Ok(result.into_iter().collect())
}

fn product_lattice(product: f64) -> Result<()> {
    if product == 0.0 {
        return Ok(());
    }
    let x = product as f32;
    if !x.is_finite() || f64::from(x) != product {
        bail!("output sample product is not exact finite FP32");
    }
    let raw = x.to_bits() & 0x7fff_ffff;
    let exponent = ((raw >> 23) & 0xff) as i32;
    let fraction = raw & 0x7f_ffff;
    let (significand, least) = if exponent == 0 {
        (fraction, -149)
    } else {
        ((1 << 23) | fraction, exponent - 150)
    };
    if least + (significand.trailing_zeros() as i32) < -126 {
        bail!("output sample product lattice does not exclude FP32 underflow");
    }
    Ok(())
}

#[derive(Serialize)]
struct Sample {
    channel: usize,
    mathematical_lower: f64,
    mathematical_upper: f64,
    arithmetic_and_bf16_bound: f64,
}

pub struct Prepared {
    input: Vec<u8>,
    gate: Vec<u16>,
    up: Vec<u16>,
    activation: Vec<Vec<u16>>,
    samples: Vec<Sample>,
    hidden: usize,
}

impl Prepared {
    pub fn new(
        hidden: usize,
        width: usize,
        gate_weight: &[u8],
        up_weight: &[u8],
        out_weight: &[u8],
    ) -> Result<Self> {
        let bytes = hidden
            .checked_mul(width)
            .and_then(|n| n.checked_mul(2))
            .context("weight shape overflows")?;
        if hidden == 0
            || width == 0
            || [gate_weight, up_weight, out_weight]
                .iter()
                .any(|weight| weight.len() != bytes)
        {
            bail!("oracle requires complete exact BF16 MLP matrices");
        }
        for weight in [gate_weight, up_weight] {
            for pair in weight.chunks_exact(2) {
                let bits = u16::from_le_bytes([pair[0], pair[1]]) & 0x7fff;
                if bits >= 0x7f80 || (bits > 0 && bits < 0x0080) {
                    bail!("basis oracle requires finite normal-or-zero gate/up weights");
                }
            }
        }
        let mut input = vec![0; hidden.checked_mul(2).context("input extent overflows")?];
        input[..2].copy_from_slice(&0x3f80_u16.to_le_bytes());
        let column = |weight: &[u8]| {
            (0..width)
                .map(|row| {
                    let bit = bits(weight, row * hidden);
                    if bit & 0x7fff == 0 {
                        0
                    } else {
                        bit
                    }
                })
                .collect::<Vec<_>>()
        };
        let gate = column(gate_weight);
        let up_values = column(up_weight);
        let activations = gate
            .iter()
            .zip(&up_values)
            .map(|(&g, &u)| activation(g, u))
            .collect::<Result<Vec<_>>>()?;
        let operations = width
            .checked_mul(2)
            .context("bound operation count overflow")?;
        if operations >= 16_777_216 {
            bail!("dense oracle requires 2K*2^-24 < 1");
        }
        let numerator = up(operations as f64 / 16_777_216.0)?;
        let gamma = up(numerator / (1.0 - numerator).next_down())?;
        // Coordinates depend only on declared shape, before any candidate.
        let channels = [
            0,
            1.min(hidden - 1),
            hidden / 4,
            hidden / 2,
            hidden * 3 / 4,
            hidden.saturating_sub(2),
            hidden - 1,
        ]
        .into_iter()
        .collect::<BTreeSet<_>>();
        let mut samples = Vec::new();
        for channel in channels {
            let (mut lower, mut upper, mut absolute) = (0.0_f64, 0.0_f64, 0.0_f64);
            for (k, choices) in activations.iter().enumerate() {
                let w = value(bits(out_weight, channel * width + k));
                if !w.is_finite() {
                    bail!("sampled output row contains nonfinite weights");
                }
                let products = choices.iter().map(|&a| value(a) * w).collect::<Vec<_>>();
                for &product in &products {
                    product_lattice(product)?;
                }
                let lo = products.iter().copied().fold(f64::INFINITY, f64::min);
                let hi = products.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                lower = (lower + lo).next_down();
                upper = (upper + hi).next_up();
                absolute = up(absolute + lo.abs().max(hi.abs()))?;
            }
            if up(up(1.0 + gamma)? * absolute)? > value(0x7f7f) {
                bail!("sampled output cannot prove overflow-free accumulation/BF16 conversion");
            }
            let arithmetic = up(gamma * absolute)?;
            let rounding = up(up(lower.abs().max(upper.abs()) + arithmetic)? / 256.0)?;
            let bound = up(up(arithmetic + rounding)? + 2.0_f64.powi(-134))?;
            samples.push(Sample {
                channel,
                mathematical_lower: lower,
                mathematical_upper: upper,
                arithmetic_and_bf16_bound: bound,
            });
        }
        Ok(Self {
            input,
            gate,
            up: up_values,
            activation: activations,
            samples,
            hidden,
        })
    }

    pub fn input(&self) -> &[u8] {
        &self.input
    }
    pub fn declaration(&self) -> Value {
        json!({"input":"R=1; z=basis[0], already-modulated MLP entrance", "gate_up":"all channels exact BF16 column[0] from each real OI matrix; exact zeros admit either sign, propagated into activation choices", "activation":"all channels independently prepared real exp interval -> intersection of finite 4-FP32-step neighborhoods (classification-first, exact tiny exp rule) -> exact RN32 add/div and two BF16 RNE boundaries", "activation_nonunique_sets":self.activation.iter().filter(|values| values.len()>1).count(), "output_samples":self.samples, "output_bound":"outward F64 sum interval includes every allowed activation choice; E=gamma(2K,2^-24)*S_upper, plus 2^-8*(max(abs(interval))+E)+2^-134", "domain":"finite BF16 gate/up normal-or-zero; sampled output products exact FP32 with binary lattice >=2^-126, expanded absolute sum <= BF16 MAX", "qualification":"primitive envelope is a predeclared test criterion, not assumed libdevice correctness; these composed samples do not prove the primitive profile for all inputs or complete MLP output correctness", "candidate_independence":"all inputs, choices, coordinates and bounds prepared before runtime creation; no candidate intermediate is used to construct expected values"})
    }

    pub fn compare(&self, outputs: &[TensorExecutionOutput]) -> Result<Value> {
        let get = |id: &str, length: usize| -> Result<&[u8]> {
            let matches = outputs
                .iter()
                .filter(|output| output.value_id().as_str() == id)
                .collect::<Vec<_>>();
            if matches.len() != 1 || matches[0].bytes().len() != length {
                bail!("MLP output {id} missing or wrong size");
            }
            Ok(matches[0].bytes().bytes())
        };
        if outputs.len() != 4 {
            bail!("MLP diagnostic expects four output ports");
        }
        let gate = get(Family::GATE_ID, self.gate.len() * 2)?;
        let up_bytes = get(Family::UP_ID, self.up.len() * 2)?;
        let act = get(Family::ACTIVATION_ID, self.activation.len() * 2)?;
        let output = get(Family::OUTPUT_ID, self.hidden * 2)?;
        let equal = |actual: u16, expected: u16| {
            actual == expected || (actual & 0x7fff == 0 && expected & 0x7fff == 0)
        };
        let gate_errors = self
            .gate
            .iter()
            .enumerate()
            .filter(|&(i, expected)| !equal(bits(gate, i), *expected))
            .count();
        let up_errors = self
            .up
            .iter()
            .enumerate()
            .filter(|&(i, expected)| !equal(bits(up_bytes, i), *expected))
            .count();
        let activation_errors = self
            .activation
            .iter()
            .enumerate()
            .filter(|&(i, choices)| !choices.contains(&bits(act, i)))
            .count();
        let mut failures = 0;
        let samples = self.samples.iter().map(|sample| {
            let actual = value(bits(output, sample.channel));
            let low = (sample.mathematical_lower - sample.arithmetic_and_bf16_bound).next_down();
            let high = (sample.mathematical_upper + sample.arithmetic_and_bf16_bound).next_up();
            let accepted = actual.is_finite() && actual >= low && actual <= high;
            if !accepted { failures += 1; }
            json!({"channel":sample.channel, "actual_bits":bits(output,sample.channel), "actual":actual.is_finite().then_some(actual), "allowed_lower":low, "allowed_upper":high, "passed":accepted})
        }).collect::<Vec<_>>();
        Ok(
            json!({"passed":gate_errors+up_errors+activation_errors+failures==0, "gate_bit_failures":gate_errors, "up_bit_failures":up_errors, "activation_set_failures":activation_errors, "sampled_output_failures":failures, "output_samples":samples, "scope":"one fixed basis row; all gate/up/activation channels, predeclared subset of final channels; not full-domain MLP, full block or Torch parity"}),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn activation_uses_independent_nested_bf16_rounding() {
        assert_eq!(activation(0x3f80, 0x3f81).unwrap(), vec![0x3f3c]);
        assert!(!activation(0x3f80, 0x3f81).unwrap().contains(&0x3f3d));
        assert_eq!(activation(0xc2b4, 0x3f80).unwrap(), vec![0x8000]);
        assert_eq!(activation(0, 0x3f80).unwrap(), vec![0, 0x8000]);
        assert_eq!(activation(0x3f80, 0x8000).unwrap(), vec![0, 0x8000]);
    }
    #[test]
    fn sampled_bounds_are_prepared_from_weights_and_reject_nonfinite_sources() {
        let bytes = |values: &[f32]| {
            values
                .iter()
                .flat_map(|&x| half::bf16::from_f32(x).to_bits().to_le_bytes())
                .collect::<Vec<_>>()
        };
        let gate = bytes(&[1., 0., 0., 1.]);
        let up = bytes(&[1., 0., 0., 1.]);
        let out = bytes(&[1., -1., 2., 3.]);
        let prepared = Prepared::new(2, 2, &gate, &up, &out).unwrap();
        assert_eq!(prepared.input(), &[0x80, 0x3f, 0, 0]);
        assert_eq!(prepared.gate, vec![0x3f80, 0]);
        assert_eq!(prepared.samples.len(), 2);
        assert!(Prepared::new(2, 2, &bytes(&[f32::NAN, 0., 0., 1.]), &up, &out).is_err());
    }
}
