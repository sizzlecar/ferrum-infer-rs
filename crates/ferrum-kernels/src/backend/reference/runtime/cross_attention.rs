//! Ascending-key online softmax with a fixed 256-scalar accumulator, never S² scores.
//! Finite-domain numerical qualification is separate from structural preflight.
use super::structured_attention::dot;
use super::{
    bf16, AllocationLocks, DeviceCommandLogicalWork, ElementType, ReferenceBufferRegion,
    ReferenceDeviceRuntimeError,
};
use ferrum_runtime::CrossAttentionBf16Geometry;
const MAX_HEAD_DIM: usize = 256;

pub(crate) struct ReferenceCrossAttentionLaunch {
    pub(crate) query: ReferenceBufferRegion,
    pub(crate) key: ReferenceBufferRegion,
    pub(crate) value: ReferenceBufferRegion,
    pub(crate) output: ReferenceBufferRegion,
    pub(crate) geometry: CrossAttentionBf16Geometry,
}
impl ReferenceCrossAttentionLaunch {
    pub(crate) fn validate(&self) -> Result<(), ReferenceDeviceRuntimeError> {
        for (region, bytes) in [
            (&self.query, self.geometry.query_bytes()),
            (&self.key, self.geometry.key_value_bytes()),
            (&self.value, self.geometry.key_value_bytes()),
            (&self.output, self.geometry.query_bytes()),
        ] {
            let bytes = usize::try_from(bytes).map_err(|_| invalid("byte extent exceeds usize"))?;
            region.validate_range()?;
            if region.element_type != ElementType::Bf16
                || region.length_bytes != bytes
                || (region.offset_bytes - region.allocation.logical_offset_bytes) % 2 != 0
            {
                return Err(invalid("requires exact BF16 ranges and scalar alignment"));
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
pub(super) fn execute(launch: &ReferenceCrossAttentionLaunch) {
    let mut storage =
        AllocationLocks::new([&launch.query, &launch.key, &launch.value, &launch.output]);
    let geometry = &launch.geometry;
    let query_length = geometry.query_length() as usize;
    let key_length = geometry.key_value_length() as usize;
    let columns = geometry.columns() as usize;
    let width = geometry.head_dim() as usize;
    let scale = geometry.effective_scale();
    let mask = geometry;
    for batch in 0..geometry.batch() as usize {
        for query in 0..query_length {
            for head in 0..geometry.heads() as usize {
                let query_start = (batch * query_length + query) * columns + head * width;
                // Bounded stack state (1 KiB), independent of S. No heap scratch or
                // intermediate BF16 probability/accumulator rounding.
                let mut accumulator = [0.0_f32; MAX_HEAD_DIM];
                let mut maximum = 0.0_f32;
                let mut denominator = 0.0_f32;
                let mut seen = false;
                let mut poisoned = false;
                for key in 0..key_length {
                    if mask.key_is_valid(batch, key as u32) != Some(true) {
                        continue;
                    }
                    let key_start = (batch * key_length + key) * columns + head * width;
                    let sum = dot(width, 0, &|column| {
                        (
                            bf16::from_bits(storage.read_bits(0, query_start + column)).to_f32(),
                            bf16::from_bits(storage.read_bits(1, key_start + column)).to_f32(),
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
                            *accumulator =
                                bf16::from_bits(storage.read_bits(2, key_start + column)).to_f32();
                        }
                        continue;
                    }
                    let next_maximum = maximum.max(score);
                    let alpha = libm::expf(maximum - next_maximum);
                    let probability = libm::expf(score - next_maximum);
                    denominator = denominator * alpha + probability;
                    for (column, accumulator) in accumulator[..width].iter_mut().enumerate() {
                        let value =
                            bf16::from_bits(storage.read_bits(2, key_start + column)).to_f32();
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
                    storage.write_bits(3, query_start + column, bf16::from_f32(output).to_bits());
                }
            }
        }
    }
}
fn invalid(reason: impl Into<String>) -> ReferenceDeviceRuntimeError {
    ReferenceDeviceRuntimeError::contract(format!("reference cross attention: {}", reason.into()))
}
#[cfg(test)]
mod tests;
