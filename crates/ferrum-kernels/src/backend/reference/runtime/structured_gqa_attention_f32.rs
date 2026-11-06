//! Grouped-query ascending-key online softmax; compact K/V and no S² scores.
//! Finite-domain numerical qualification is separate from structural preflight.
use super::{
    AllocationLocks, DeviceCommandLogicalWork, ElementType, ReferenceBufferRegion,
    ReferenceDeviceRuntimeError,
};
use ferrum_runtime::{StructuredGqaAttentionF32Geometry, STRUCTURED_ATTENTION_BF16_MAX_HEAD_DIM};

// The CUDA adjacent-pair tree combines a complete power-of-two prefix with
// the remaining subtree. Carry unmatched subtrees without adding zero, which
// preserves signed zeros and every existing power-of-two reduction boundary.
fn dot(n: usize, start: usize, read: &impl Fn(usize) -> (f32, f32)) -> f32 {
    if n == 1 {
        let (a, b) = read(start);
        a * b
    } else {
        let left = n.next_power_of_two() / 2;
        dot(left, start, read) + dot(n - left, start + left, read)
    }
}

pub(crate) struct ReferenceStructuredGqaAttentionF32Launch {
    pub(crate) query: ReferenceBufferRegion,
    pub(crate) key: ReferenceBufferRegion,
    pub(crate) value: ReferenceBufferRegion,
    pub(crate) output: ReferenceBufferRegion,
    pub(crate) geometry: StructuredGqaAttentionF32Geometry,
}
impl ReferenceStructuredGqaAttentionF32Launch {
    pub(crate) fn validate(&self) -> Result<(), ReferenceDeviceRuntimeError> {
        let bytes = self
            .geometry
            .element_count()
            .checked_mul(4)
            .and_then(|n| usize::try_from(n).ok())
            .ok_or_else(|| invalid("byte extent exceeds usize"))?;
        let kv_bytes = self
            .geometry
            .rows()
            .checked_mul(self.geometry.key_value_columns())
            .and_then(|n| n.checked_mul(4))
            .and_then(|n| usize::try_from(n).ok())
            .ok_or_else(|| invalid("KV byte extent exceeds usize"))?;
        for (region, expected) in [
            (&self.query, bytes),
            (&self.key, kv_bytes),
            (&self.value, kv_bytes),
            (&self.output, bytes),
        ] {
            region.validate_range()?;
            if region.element_type != ElementType::F32
                || region.length_bytes != expected
                || (region.offset_bytes - region.allocation.logical_offset_bytes) % 4 != 0
            {
                return Err(invalid("requires exact F32 ranges and scalar alignment"));
            }
        }
        for read in [&self.query, &self.key, &self.value] {
            if self.output.overlaps(read) {
                return Err(invalid("output overlaps a read operand"));
            }
        }
        self.logical_work()?;
        Ok(())
    }
    pub(crate) fn logical_work(
        &self,
    ) -> Result<DeviceCommandLogicalWork, ReferenceDeviceRuntimeError> {
        DeviceCommandLogicalWork::tensor(
            self.geometry.element_count(),
            self.geometry.logical_arithmetic_operations(),
        )
        .map_err(|e| invalid(e.to_string()))
    }
    pub(crate) fn validate_runtime(&self, runtime: u64) -> Result<(), ReferenceDeviceRuntimeError> {
        for region in [&self.query, &self.key, &self.value, &self.output] {
            region.validate_runtime(runtime)?;
        }
        self.validate()
    }
}
pub(super) fn execute(launch: &ReferenceStructuredGqaAttentionF32Launch) {
    let mut storage =
        AllocationLocks::new([&launch.query, &launch.key, &launch.value, &launch.output]);
    let geometry = &launch.geometry;
    let sequence = geometry.sequence_length() as usize;
    let columns = geometry.query_columns() as usize;
    let kv_columns = geometry.key_value_columns() as usize;
    let width = geometry.head_dim() as usize;
    let scale = geometry.effective_scale();
    let mask = geometry.mask();
    for batch in 0..geometry.batch() as usize {
        for query in 0..sequence {
            for head in 0..geometry.query_heads() as usize {
                let kv_head = head / geometry.group_size() as usize;
                let query_start = (batch * sequence + query) * columns + head * width;
                // Bounded stack state (1 KiB), independent of S. No heap scratch or
                // intermediate F32 probability/accumulator rounding.
                let mut accumulator = [0.0_f32; STRUCTURED_ATTENTION_BF16_MAX_HEAD_DIM as usize];
                let mut maximum = 0.0_f32;
                let mut denominator = 0.0_f32;
                let mut seen = false;
                let mut poisoned = false;
                let end = mask.visible_end(query as u32).expect("checked query range") as usize;
                for key in 0..end {
                    if mask.key_is_valid(batch, key as u32) != Some(true) {
                        continue;
                    }
                    let key_start = (batch * sequence + key) * kv_columns + kv_head * width;
                    let sum = dot(width, 0, &|column| {
                        (
                            storage.read_f32(0, query_start + column),
                            storage.read_f32(1, key_start + column),
                        )
                    });
                    let score = sum * scale;
                    if !score.is_finite() {
                        poisoned = true;
                        break;
                    }
                    if !seen {
                        maximum = score;
                        denominator = 1.0;
                        seen = true;
                        for (column, accumulator) in accumulator[..width].iter_mut().enumerate() {
                            *accumulator = storage.read_f32(2, key_start + column);
                        }
                        continue;
                    }
                    let next_maximum = maximum.max(score);
                    let alpha = libm::expf(maximum - next_maximum);
                    let probability = libm::expf(score - next_maximum);
                    denominator = denominator * alpha + probability;
                    for (column, accumulator) in accumulator[..width].iter_mut().enumerate() {
                        let value = storage.read_f32(2, key_start + column);
                        let old = *accumulator * alpha;
                        let new = probability * value;
                        *accumulator = old + new;
                    }
                    maximum = next_maximum;
                }
                let inverse = if seen { 1.0_f32 / denominator } else { 0.0 };
                for (column, accumulator) in accumulator[..width].iter().enumerate() {
                    let output = if poisoned {
                        f32::NAN
                    } else if seen {
                        *accumulator * inverse
                    } else {
                        0.0
                    };
                    storage.write_f32(3, query_start + column, output);
                }
            }
        }
    }
}
fn invalid(reason: impl Into<String>) -> ReferenceDeviceRuntimeError {
    ReferenceDeviceRuntimeError::contract(format!(
        "reference structured GQA attention: {}",
        reason.into()
    ))
}
#[cfg(test)]
mod tests;
