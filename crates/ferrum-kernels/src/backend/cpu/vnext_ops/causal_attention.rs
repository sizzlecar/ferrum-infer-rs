//! Causal attention over an admitted paged KV prefix. F16 query/KV and
//! F32 softmax accumulation follow the operation's declared boundaries.

use ferrum_interfaces::vnext::{
    AttributeId, DynamicStorageAllocator, DynamicStorageProfile, DynamicStorageView, SemanticValue,
    VNextError,
};
use rayon::prelude::*;
use std::collections::BTreeMap;

use super::super::vnext_runtime::CpuRuntimeError;
use super::provider::unsigned;
use super::scalar::CpuFloat;

pub(super) const KV_PAGE_BYTES: u64 = 64 * 1024;

pub(super) fn kv_storage_profile() -> Result<DynamicStorageProfile, VNextError> {
    DynamicStorageProfile::new(
        DynamicStorageAllocator::FixedBlockArena {
            block_bytes: KV_PAGE_BYTES,
        },
        DynamicStorageView::PagedRegions {
            block_bytes: KV_PAGE_BYTES,
        },
    )
}

pub(super) trait CpuKvStorage: Sync {
    fn len_bytes(&self) -> usize;
    fn read_half(&self, index: usize) -> f32;
    fn write_half(&mut self, index: usize, value: f32);
}

#[cfg(test)]
impl CpuKvStorage for [u8] {
    fn len_bytes(&self) -> usize {
        self.len()
    }
    fn read_half(&self, index: usize) -> f32 {
        CpuFloat::F16.read(self, index)
    }
    fn write_half(&mut self, index: usize, value: f32) {
        CpuFloat::F16.write(self, index, value);
    }
}

/// Borrow physical pages in logical order. No KV flattening or secondary cache.
pub(super) struct CpuKvPages<'a, 'b> {
    pages: &'a mut [&'b mut [u8]],
    elements_per_page: usize,
    bytes: usize,
}

impl<'a, 'b> CpuKvPages<'a, 'b> {
    pub(super) fn new(pages: &'a mut [&'b mut [u8]]) -> Result<Self, CpuRuntimeError> {
        let page_bytes = pages.first().map_or(0, |page| page.len());
        let bytes = page_bytes
            .checked_mul(pages.len())
            .ok_or_else(|| CpuRuntimeError::new("CPU KV page capacity overflows"))?;
        if page_bytes == 0
            || !page_bytes.is_multiple_of(2)
            || pages.iter().any(|page| page.len() != page_bytes)
        {
            return Err(CpuRuntimeError::new(
                "CPU KV pages have inconsistent F16 geometry",
            ));
        }
        Ok(Self {
            pages,
            elements_per_page: page_bytes / 2,
            bytes,
        })
    }
}

impl CpuKvStorage for CpuKvPages<'_, '_> {
    fn len_bytes(&self) -> usize {
        self.bytes
    }
    fn read_half(&self, index: usize) -> f32 {
        CpuFloat::F16.read(
            self.pages[index / self.elements_per_page],
            index % self.elements_per_page,
        )
    }
    fn write_half(&mut self, index: usize, value: f32) {
        CpuFloat::F16.write(
            self.pages[index / self.elements_per_page],
            index % self.elements_per_page,
            value,
        );
    }
}

#[derive(Clone, Copy, Debug)]
pub(super) struct CausalShape {
    pub(super) hidden: usize,
    pub(super) query_heads: usize,
    pub(super) kv_heads: usize,
    pub(super) head_dim: usize,
    pub(super) rope_dim: usize,
    pub(super) maximum_context: usize,
    pub(super) epsilon: f32,
    pub(super) theta: f32,
    pub(super) interleaved: bool,
    pub(super) gated: bool,
}

impl CausalShape {
    pub(super) fn from_attributes(
        attributes: &BTreeMap<AttributeId, SemanticValue>,
    ) -> Result<Self, CpuRuntimeError> {
        let size = |name| {
            usize::try_from(unsigned(attributes, name)?).map_err(|_| {
                CpuRuntimeError::new("CPU attention dimension exceeds addressable memory")
            })
        };
        let value = |name| {
            attributes
                .iter()
                .find(|(id, _)| id.as_str() == name)
                .map(|(_, value)| value)
        };
        let rational = |name| match value(name) {
            Some(SemanticValue::Rational(value)) => {
                Ok((value.numerator() as f64 / value.denominator() as f64) as f32)
            }
            _ => Err(CpuRuntimeError::new(format!(
                "CPU attention lacks rational attribute {name}"
            ))),
        };
        let boolean = |name| match value(name) {
            Some(SemanticValue::Bool(value)) => Ok(*value),
            _ => Err(CpuRuntimeError::new(format!(
                "CPU attention lacks boolean attribute {name}"
            ))),
        };
        let shape = Self {
            hidden: size("hidden_size")?,
            query_heads: size("query_heads")?,
            kv_heads: size("key_value_heads")?,
            head_dim: size("head_dim")?,
            rope_dim: size("rope_dim")?,
            maximum_context: size("maximum_context_tokens")?,
            epsilon: rational("epsilon")?,
            theta: rational("rope_theta")?,
            interleaved: boolean("rope_interleaved")?,
            gated: boolean("output_gate")?,
        };
        shape.validate()?;
        if !boolean("causal")?
            || size("query_features")? != shape.queries()
            || size("kv_features")? != shape.kv()
            || size("query_projection_features")? != shape.query_projection()
        {
            return Err(CpuRuntimeError::new(
                "CPU attention attributes disagree with the causal shape",
            ));
        }
        Ok(shape)
    }

    fn validate(self) -> Result<(), CpuRuntimeError> {
        if self.hidden == 0
            || self.query_heads == 0
            || self.kv_heads == 0
            || self.head_dim == 0
            || !self.query_heads.is_multiple_of(self.kv_heads)
            || self.rope_dim == 0
            || self.rope_dim > self.head_dim
            || !self.rope_dim.is_multiple_of(2)
            || self.maximum_context == 0
            || !self.epsilon.is_finite()
            || self.epsilon <= 0.0
            || !self.theta.is_finite()
            || self.theta <= 0.0
            || self
                .query_heads
                .checked_mul(self.head_dim)
                .and_then(|n| n.checked_mul(4))
                .is_none()
            || self
                .kv_heads
                .checked_mul(self.head_dim)
                .and_then(|n| n.checked_mul(4))
                .and_then(|n| n.checked_mul(self.maximum_context))
                .is_none()
            || self.hidden.checked_mul(4).is_none()
        {
            return Err(CpuRuntimeError::new(
                "CPU causal attention shape is empty, inconsistent or overflows",
            ));
        }
        Ok(())
    }
    pub(super) fn queries(self) -> usize {
        self.query_heads * self.head_dim
    }
    pub(super) fn kv(self) -> usize {
        self.kv_heads * self.head_dim
    }
    pub(super) fn query_projection(self) -> usize {
        self.queries() * if self.gated { 2 } else { 1 }
    }
    pub(super) fn state_bytes_per_token(self) -> usize {
        self.kv() * 4
    }
}

pub(super) struct CausalInputs<'a> {
    pub(super) query: &'a [u8],
    pub(super) key: &'a [u8],
    pub(super) value: &'a [u8],
    pub(super) query_norm: &'a [u8],
    pub(super) key_norm: &'a [u8],
}

pub(super) struct CausalScratch<'a> {
    pub(super) query: &'a mut [u8],
    pub(super) accumulated: &'a mut [u8],
    pub(super) context: &'a mut [u8],
}

fn normalized_rotary(
    shape: CausalShape,
    position: usize,
    raw: &[u8],
    norm: &[u8],
    mut output: impl FnMut(usize, f32),
) {
    let half = CpuFloat::F16;
    let mut squared = 0.0_f32;
    for column in 0..shape.head_dim {
        let value = half.read(raw, column);
        squared += value * value;
    }
    let inverse = (squared / shape.head_dim as f32 + shape.epsilon)
        .sqrt()
        .recip();
    let normalized = |column| half.read(raw, column) * inverse * half.read(norm, column);
    for pair in 0..shape.rope_dim / 2 {
        let (low, high) = if shape.interleaved {
            (pair * 2, pair * 2 + 1)
        } else {
            (pair, pair + shape.rope_dim / 2)
        };
        let angle = position as f32
            * shape
                .theta
                .powf(-((pair * 2) as f32) / shape.rope_dim as f32);
        let (sine, cosine) = angle.sin_cos();
        let (left, right) = (normalized(low), normalized(high));
        output(low, left * cosine - right * sine);
        output(high, left * sine + right * cosine);
    }
    for column in shape.rope_dim..shape.head_dim {
        output(column, normalized(column));
    }
}

pub(super) fn step<K: CpuKvStorage + ?Sized>(
    shape: CausalShape,
    position: usize,
    input: CausalInputs<'_>,
    kv: &mut K,
    scratch: CausalScratch<'_>,
) -> Result<(), CpuRuntimeError> {
    shape.validate()?;
    if position >= shape.maximum_context
        || kv.len_bytes() < (position + 1) * shape.state_bytes_per_token()
        || input.query.len() != shape.query_projection() * 2
        || input.key.len() != shape.kv() * 2
        || input.value.len() != shape.kv() * 2
        || input.query_norm.len() != shape.head_dim * 2
        || input.key_norm.len() != shape.head_dim * 2
        || scratch.query.len() != shape.queries() * 2
        || scratch.accumulated.len() != shape.queries() * 4
        || scratch.context.len() != shape.queries() * 2
    {
        return Err(CpuRuntimeError::new(
            "CPU attention buffers or position exceed the admitted causal prefix",
        ));
    }
    let query_stride = shape.head_dim * if shape.gated { 2 } else { 1 };
    for head in 0..shape.query_heads {
        normalized_rotary(
            shape,
            position,
            &input.query[head * query_stride * 2..(head * query_stride + shape.head_dim) * 2],
            input.query_norm,
            |column, value| {
                CpuFloat::F16.write(scratch.query, head * shape.head_dim + column, value)
            },
        );
    }
    let token_start = position * shape.kv() * 2;
    for head in 0..shape.kv_heads {
        normalized_rotary(
            shape,
            position,
            &input.key[head * shape.head_dim * 2..(head + 1) * shape.head_dim * 2],
            input.key_norm,
            |column, value| kv.write_half(token_start + head * shape.head_dim + column, value),
        );
    }
    for column in 0..shape.kv() {
        kv.write_half(
            token_start + shape.kv() + column,
            CpuFloat::F16.read(input.value, column),
        );
    }
    let kv: &K = kv;
    let query: &[u8] = scratch.query;
    scratch
        .accumulated
        .par_chunks_exact_mut(shape.head_dim * 4)
        .zip(scratch.context.par_chunks_exact_mut(shape.head_dim * 2))
        .enumerate()
        .for_each(|(head, (accumulated, context))| {
            let half = CpuFloat::F16;
            let float = CpuFloat::F32;
            let kv_head = head / (shape.query_heads / shape.kv_heads);
            let mut maximum = f32::NEG_INFINITY;
            let mut denominator = 0.0_f32;
            accumulated.fill(0);
            for token in 0..=position {
                let keys = token * shape.kv() * 2 + kv_head * shape.head_dim;
                let mut score = 0.0_f32;
                for column in 0..shape.head_dim {
                    score += half.read(query, head * shape.head_dim + column)
                        * kv.read_half(keys + column);
                }
                score /= (shape.head_dim as f32).sqrt();
                let next_maximum = maximum.max(score);
                let previous_scale = (maximum - next_maximum).exp();
                let probability = (score - next_maximum).exp();
                denominator = denominator * previous_scale + probability;
                for column in 0..shape.head_dim {
                    let value = float.read(accumulated, column) * previous_scale
                        + probability * kv.read_half(keys + shape.kv() + column);
                    float.write(accumulated, column, value);
                }
                maximum = next_maximum;
            }
            for column in 0..shape.head_dim {
                let mut value = float.read(accumulated, column) / denominator;
                if shape.gated {
                    let gate =
                        half.read(input.query, head * query_stride + shape.head_dim + column);
                    value *= if gate >= 0.0 {
                        1.0 / (1.0 + (-gate).exp())
                    } else {
                        let exp = gate.exp();
                        exp / (1.0 + exp)
                    };
                }
                half.write(context, column, value);
            }
        });
    Ok(())
}

#[cfg(test)]
mod tests;
