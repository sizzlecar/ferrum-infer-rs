//! Sampled final projection: exact BF16 product lattice and independent F64
//! outward summation, including all permitted attention BF16 alternatives.
use super::{bits, value};
use anyhow::{bail, Context, Result};
use serde::Serialize;
use serde_json::{json, Value};
use std::collections::BTreeSet;
fn up(x: f64) -> Result<f64> {
    let result = if x == 0.0 { x } else { x.next_up() };
    if !result.is_finite() || result < 0.0 {
        bail!("unresolved: dense bound is not nonnegative finite");
    }
    Ok(result)
}
fn product_lattice(product: f64) -> Result<()> {
    if product == 0.0 {
        return Ok(());
    }
    let rounded = product as f32;
    if !rounded.is_finite() || f64::from(rounded) != product {
        bail!("unresolved: sampled product is not exact finite FP32");
    }
    let raw = rounded.to_bits() & 0x7fff_ffff;
    let exponent = ((raw >> 23) & 255) as i32;
    let fraction = raw & 0x7f_ffff;
    let (significand, least) = if exponent == 0 {
        (fraction, -149)
    } else {
        ((1 << 23) | fraction, exponent - 150)
    };
    if least + (significand.trailing_zeros() as i32) < -126 {
        bail!("unresolved: sampled product lattice permits FP32 underflow");
    }
    Ok(())
}
#[derive(Debug, Serialize)]
pub(super) struct Sample {
    pub(super) row: usize,
    pub(super) channel: usize,
    pub(super) mathematical_lower: f64,
    pub(super) mathematical_upper: f64,
    pub(super) arithmetic_and_bf16_bound: f64,
}
pub(super) fn prepare(
    inputs: &[Vec<u16>],
    hidden: usize,
    rows: usize,
    weights: &[u8],
) -> Result<Vec<Sample>> {
    if hidden == 0
        || inputs.len()
            != rows
                .checked_mul(hidden)
                .context("sample dimensions overflow")?
        || weights.len()
            != hidden
                .checked_mul(hidden)
                .and_then(|x| x.checked_mul(2))
                .context("sample weight bytes overflow")?
    {
        bail!("unresolved: sampled dense dimensions mismatch");
    }
    let operations = hidden
        .checked_mul(2)
        .context("dense operation count overflow")?;
    if operations >= 16_777_216 {
        bail!("unresolved: dense gamma requires 2K*2^-24<1");
    }
    let nu = up(operations as f64 / 16_777_216.0)?;
    let gamma = up(nu / (1.0 - nu).next_down())?;
    let coordinates = [
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
    for row in 0..rows {
        for &channel in &coordinates {
            let (mut low, mut high, mut absolute) = (0.0_f64, 0.0_f64, 0.0_f64);
            for (k, choices) in inputs[row * hidden..(row + 1) * hidden].iter().enumerate() {
                if choices.is_empty() {
                    bail!("unresolved: empty dense input set");
                }
                let w = value(bits(weights, channel * hidden + k));
                if !w.is_finite() {
                    bail!("unresolved: sampled output weight is nonfinite");
                }
                let (mut min, mut max) = (f64::INFINITY, f64::NEG_INFINITY);
                for &a in choices {
                    let product = value(a) * w;
                    product_lattice(product)?;
                    min = min.min(product);
                    max = max.max(product);
                }
                low = (low + min).next_down();
                high = (high + max).next_up();
                absolute = up(absolute + min.abs().max(max.abs()))?;
            }
            if up(up(1.0 + gamma)? * absolute)? > value(0x7f7f) {
                bail!("unresolved: sampled dense cannot exclude accumulation/BF16 overflow");
            }
            let arithmetic = up(gamma * absolute)?;
            let rounding = up(up(low.abs().max(high.abs()) + arithmetic)? / 256.0)?;
            let bound = up(up(arithmetic + rounding)? + 2.0_f64.powi(-134))?;
            samples.push(Sample {
                row,
                channel,
                mathematical_lower: low,
                mathematical_upper: high,
                arithmetic_and_bf16_bound: bound,
            });
        }
    }
    Ok(samples)
}
pub(super) fn compare(samples: &[Sample], raw: &[u8], hidden: usize) -> Vec<Value> {
    samples.iter().filter_map(|s|{let raw_bits=bits(raw,s.row*hidden+s.channel);let actual=value(raw_bits);let lo=(s.mathematical_lower-s.arithmetic_and_bf16_bound).next_down();let hi=(s.mathematical_upper+s.arithmetic_and_bf16_bound).next_up();if actual.is_finite()&&actual>=lo&&actual<=hi{None}else{Some(json!({"row":s.row,"channel":s.channel,"actual_bits":raw_bits,"allowed_lower":lo,"allowed_upper":hi}))}}).collect()
}
