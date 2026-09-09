//! Sequential recurrent semantics with bounded, caller-owned row workspace.
//! Parallelism is across independent value heads; token/state order is fixed.

use std::collections::BTreeMap;

use ferrum_interfaces::vnext::{
    AttributeId, GatedDeltaDecayParameterization, GatedDeltaValueHeadMapping, SemanticValue,
};
use rayon::prelude::*;

use super::super::vnext_runtime::CpuRuntimeError;
use super::provider::unsigned;
use super::scalar::CpuFloat;

#[derive(Clone, Copy, Debug)]
pub(super) struct GatedDeltaShape {
    pub(super) hidden: usize,
    pub(super) key_heads: usize,
    pub(super) value_heads: usize,
    pub(super) key_dim: usize,
    pub(super) value_dim: usize,
    pub(super) convolution: usize,
    pub(super) epsilon: f32,
    pub(super) decay: GatedDeltaDecayParameterization,
    pub(super) mapping: GatedDeltaValueHeadMapping,
}

impl GatedDeltaShape {
    pub(super) fn from_attributes(
        attributes: &BTreeMap<AttributeId, SemanticValue>,
    ) -> Result<Self, CpuRuntimeError> {
        let size = |name| {
            usize::try_from(unsigned(attributes, name)?)
                .map_err(|_| CpuRuntimeError::new("CPU GDN dimension exceeds addressable memory"))
        };
        let text = |name| match attributes
            .iter()
            .find(|(id, _)| id.as_str() == name)
            .map(|(_, value)| value)
        {
            Some(SemanticValue::Text(value)) => Ok(value.as_str()),
            _ => Err(CpuRuntimeError::new(format!(
                "CPU GDN lacks typed text attribute {name}"
            ))),
        };
        let epsilon = match attributes
            .iter()
            .find(|(id, _)| id.as_str() == "epsilon")
            .map(|(_, value)| value)
        {
            Some(SemanticValue::Rational(value)) => {
                (value.numerator() as f64 / value.denominator() as f64) as f32
            }
            _ => return Err(CpuRuntimeError::new("CPU GDN lacks rational epsilon")),
        };
        let shape = Self {
            hidden: size("hidden_size")?,
            key_heads: size("key_heads")?,
            value_heads: size("value_heads")?,
            key_dim: size("key_head_dim")?,
            value_dim: size("value_head_dim")?,
            convolution: size("conv_kernel")?,
            epsilon,
            decay: GatedDeltaDecayParameterization::parse(text("decay_parameterization")?)
                .ok_or_else(|| {
                    CpuRuntimeError::new("CPU GDN decay parameterization is unsupported")
                })?,
            mapping: GatedDeltaValueHeadMapping::parse(text("value_head_mapping")?)
                .ok_or_else(|| CpuRuntimeError::new("CPU GDN value-head mapping is unsupported"))?,
        };
        shape.validate()?;
        for (name, expected) in [
            ("qkv_features", shape.qkv()),
            ("value_features", shape.values()),
            ("qkvz_features", shape.qkv() + shape.values()),
            ("ba_features", shape.value_heads * 2),
            ("qkvzba_features", shape.mixed()),
            ("conv_state_width", shape.convolution - 1),
        ] {
            if size(name)? != expected {
                return Err(CpuRuntimeError::new(format!(
                    "CPU GDN attribute {name} disagrees with its shape"
                )));
            }
        }
        Ok(shape)
    }

    fn validate(self) -> Result<(), CpuRuntimeError> {
        let qk = self.key_heads.checked_mul(self.key_dim);
        let values = self.value_heads.checked_mul(self.value_dim);
        let mixed = qk
            .and_then(|n| n.checked_mul(2))
            .and_then(|n| values.and_then(|v| v.checked_mul(2).and_then(|v| n.checked_add(v))))
            .and_then(|n| {
                self.value_heads
                    .checked_mul(2)
                    .and_then(|heads| n.checked_add(heads))
            });
        if self.hidden == 0
            || self.key_heads == 0
            || self.value_heads == 0
            || self.key_dim == 0
            || self.value_dim == 0
            || self.convolution < 2
            || !self.value_heads.is_multiple_of(self.key_heads)
            || !self.epsilon.is_finite()
            || self.epsilon <= 0.0
            || mixed.and_then(|n| n.checked_mul(4)).is_none()
            || values
                .and_then(|n| n.checked_mul(self.key_dim))
                .and_then(|n| n.checked_mul(4))
                .is_none()
            || mixed
                .and_then(|n| n.checked_mul(self.convolution))
                .and_then(|n| n.checked_mul(2))
                .is_none()
            || self.hidden.checked_mul(4).is_none()
        {
            return Err(CpuRuntimeError::new(
                "CPU GDN shape is empty, inconsistent or overflows",
            ));
        }
        Ok(())
    }

    pub(super) fn qk(self) -> usize {
        self.key_heads * self.key_dim
    }
    pub(super) fn values(self) -> usize {
        self.value_heads * self.value_dim
    }
    pub(super) fn qkv(self) -> usize {
        self.qk() * 2 + self.values()
    }
    pub(super) fn mixed(self) -> usize {
        self.qkv() + self.values() + self.value_heads * 2
    }
}

pub(super) struct GatedDeltaWeights<'a> {
    pub(super) convolution: &'a [u8],
    pub(super) decay: &'a [u8],
    pub(super) dt_bias: &'a [u8],
    pub(super) norm: &'a [u8],
}

pub(super) struct GatedDeltaState<'a> {
    pub(super) convolution: &'a mut [u8],
    pub(super) recurrent: &'a mut [u8],
}

pub(super) struct GatedDeltaScratch<'a> {
    pub(super) qkv: &'a mut [u8],
    pub(super) core: &'a mut [u8],
    pub(super) output: &'a mut [u8],
}

fn sigmoid(value: f32) -> f32 {
    if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let exp = value.exp();
        exp / (1.0 + exp)
    }
}

pub(super) fn step(
    shape: GatedDeltaShape,
    mixed: &[u8],
    weights: GatedDeltaWeights<'_>,
    state: GatedDeltaState<'_>,
    scratch: GatedDeltaScratch<'_>,
) -> Result<(), CpuRuntimeError> {
    shape.validate()?;
    let f16 = CpuFloat::F16;
    let f32 = CpuFloat::F32;
    let history = shape.convolution - 1;
    for (actual, expected) in [
        (mixed.len(), shape.mixed() * 2),
        (
            weights.convolution.len(),
            shape.qkv() * shape.convolution * 2,
        ),
        (weights.decay.len(), shape.value_heads * 4),
        (weights.dt_bias.len(), shape.value_heads * 4),
        (weights.norm.len(), shape.value_dim * 4),
        (state.convolution.len(), shape.qkv() * history * 2),
        (state.recurrent.len(), shape.values() * shape.key_dim * 4),
        (scratch.qkv.len(), shape.qkv() * 4),
        (scratch.core.len(), shape.values() * 4),
        (scratch.output.len(), shape.values() * 2),
    ] {
        if actual != expected {
            return Err(CpuRuntimeError::new(
                "CPU GDN buffer differs from its validated physical shape",
            ));
        }
    }

    for channel in 0..shape.qkv() {
        let current = f16.read(mixed, channel);
        let mut convolved = 0.0_f32;
        for index in 0..history {
            convolved += f16.read(state.convolution, channel * history + index)
                * f16.read(weights.convolution, channel * shape.convolution + index);
        }
        convolved += current * f16.read(weights.convolution, channel * shape.convolution + history);
        let stored = &mut state.convolution[channel * history * 2..(channel + 1) * history * 2];
        stored.copy_within(2..history * 2, 0);
        f16.write(stored, history - 1, current);
        f32.write(scratch.qkv, channel, convolved * sigmoid(convolved));
    }
    for kind in 0..2 {
        for head in 0..shape.key_heads {
            let start = kind * shape.qk() + head * shape.key_dim;
            let mut squared = 0.0_f32;
            for column in 0..shape.key_dim {
                let value = f32.read(scratch.qkv, start + column);
                squared += value * value;
            }
            let scale = (squared + 1.0e-6).sqrt().recip();
            for column in 0..shape.key_dim {
                let value = f32.read(scratch.qkv, start + column);
                f32.write(scratch.qkv, start + column, value * scale);
            }
        }
    }
    let qkv: &[u8] = scratch.qkv;
    state
        .recurrent
        .par_chunks_exact_mut(shape.value_dim * shape.key_dim * 4)
        .zip(scratch.core.par_chunks_exact_mut(shape.value_dim * 4))
        .enumerate()
        .for_each(|(head, (state, core))| {
            let key_head = match shape.mapping {
                GatedDeltaValueHeadMapping::GroupedByKeyHead => {
                    head / (shape.value_heads / shape.key_heads)
                }
                GatedDeltaValueHeadMapping::InterleavedByKeyHead => head % shape.key_heads,
            };
            let gates = shape.qkv() + shape.values();
            let beta = sigmoid(f16.read(mixed, gates + head));
            let a =
                f16.read(mixed, gates + shape.value_heads + head) + f32.read(weights.dt_bias, head);
            let softplus = if a > 20.0 {
                a
            } else if a < -20.0 {
                a.exp()
            } else {
                a.exp().ln_1p()
            };
            let rate = match shape.decay {
                GatedDeltaDecayParameterization::LogRate => -f32.read(weights.decay, head).exp(),
                GatedDeltaDecayParameterization::NegativeRate => f32.read(weights.decay, head),
            };
            let decay = (rate * softplus).exp();
            let query_scale = (shape.key_dim as f32).sqrt().recip();
            for column in 0..shape.value_dim {
                let mut predicted = 0.0_f32;
                for key_column in 0..shape.key_dim {
                    let index = column * shape.key_dim + key_column;
                    let decayed = f32.read(state, index) * decay;
                    f32.write(state, index, decayed);
                    predicted +=
                        decayed * f32.read(qkv, shape.qk() + key_head * shape.key_dim + key_column);
                }
                let value = f32.read(qkv, shape.qk() * 2 + head * shape.value_dim + column);
                let delta = (value - predicted) * beta;
                let mut result = 0.0_f32;
                for key_column in 0..shape.key_dim {
                    let index = column * shape.key_dim + key_column;
                    let updated = f32.read(state, index)
                        + delta * f32.read(qkv, shape.qk() + key_head * shape.key_dim + key_column);
                    f32.write(state, index, updated);
                    result += updated * f32.read(qkv, key_head * shape.key_dim + key_column);
                }
                f32.write(core, column, result * query_scale);
            }
        });
    scratch
        .output
        .par_chunks_exact_mut(shape.value_dim * 2)
        .enumerate()
        .for_each(|(head, output)| {
            let start = head * shape.value_dim;
            let mut squared = 0.0_f32;
            for column in 0..shape.value_dim {
                let value = f32.read(scratch.core, start + column);
                squared += value * value;
            }
            let scale = (squared / shape.value_dim as f32 + shape.epsilon)
                .sqrt()
                .recip();
            for column in 0..shape.value_dim {
                let z = f16.read(mixed, shape.qkv() + start + column);
                f16.write(
                    output,
                    column,
                    f32.read(scratch.core, start + column)
                        * scale
                        * f32.read(weights.norm, column)
                        * (z * sigmoid(z)),
                );
            }
        });
    Ok(())
}

#[cfg(test)]
mod tests;
