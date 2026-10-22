//! Whole-row FP32 RMS and zero-centered affine, with only the final BF16 boundary.
//! Numerical nonfinite/overflow propagation is IEEE; preflight checks structure.

use super::{
    bf16, AllocationLocks, DeviceCommandLogicalWork, ElementType, ReferenceBufferRegion,
    ReferenceDeviceRuntimeError,
};
use ferrum_runtime::{CanonicalRational, ZERO_CENTERED_RMS_NORM_BF16_MAX_WIDTH};

#[derive(Clone, Copy)]
pub(crate) struct ReferenceZeroCenteredRmsNormShape {
    pub(crate) rows: usize,
    pub(crate) columns: usize,
    pub(crate) epsilon: CanonicalRational,
}

impl ReferenceZeroCenteredRmsNormShape {
    pub(crate) fn byte_extents(self) -> Result<(usize, usize), ReferenceDeviceRuntimeError> {
        if self.rows == 0
            || self.columns == 0
            || self.columns as u128 > u128::from(ZERO_CENTERED_RMS_NORM_BF16_MAX_WIDTH)
        {
            return Err(invalid(
                "requires positive rows and whole-row width in 1..=2^24",
            ));
        }
        let numerator = i128::from(self.epsilon.numerator());
        let denominator = i128::from(self.epsilon.denominator());
        if numerator * 1_000_000_000_000 < denominator || numerator > denominator {
            return Err(invalid("epsilon must be rational in [1e-12, 1]"));
        }
        let matrix = self
            .rows
            .checked_mul(self.columns)
            .and_then(|n| n.checked_mul(2))
            .ok_or_else(|| invalid("matrix byte extent overflows usize"))?;
        let weight = self
            .columns
            .checked_mul(2)
            .ok_or_else(|| invalid("weight byte extent overflows usize"))?;
        Ok((matrix, weight))
    }

    pub(crate) fn logical_work(
        self,
    ) -> Result<DeviceCommandLogicalWork, ReferenceDeviceRuntimeError> {
        self.byte_extents()?;
        let rows = u64::try_from(self.rows).map_err(|_| invalid("rows exceed u64"))?;
        let columns = u64::try_from(self.columns).map_err(|_| invalid("columns exceed u64"))?;
        let elements = rows
            .checked_mul(columns)
            .ok_or_else(|| invalid("element count overflows"))?;
        let operations = columns
            .checked_mul(5)
            .and_then(|n| n.checked_add(3))
            .and_then(|n| rows.checked_mul(n))
            .ok_or_else(|| invalid("logical arithmetic count overflows"))?;
        // Per row: C squares, C-1 adds, divide/epsilon/sqrt/reciprocal,
        // C normalization products, C weight offsets, C learned products.
        // Conversions excluded; logical operations, not hardware FLOPs.
        DeviceCommandLogicalWork::tensor(elements, operations).map_err(|e| invalid(e.to_string()))
    }
}

pub(crate) struct ReferenceZeroCenteredRmsNormLaunch {
    pub(crate) input: ReferenceBufferRegion,
    pub(crate) weight: ReferenceBufferRegion,
    pub(crate) output: ReferenceBufferRegion,
    pub(crate) shape: ReferenceZeroCenteredRmsNormShape,
}

impl ReferenceZeroCenteredRmsNormLaunch {
    pub(crate) fn validate(&self) -> Result<(), ReferenceDeviceRuntimeError> {
        let (matrix, weight) = self.shape.byte_extents()?;
        for (region, bytes) in [
            (&self.input, matrix),
            (&self.weight, weight),
            (&self.output, matrix),
        ] {
            region.validate_range()?;
            if region.element_type != ElementType::Bf16
                || region.length_bytes != bytes
                || (region.offset_bytes - region.allocation.logical_offset_bytes) % 2 != 0
            {
                return Err(invalid(
                    "region dtype, exact byte extent or alignment differs",
                ));
            }
        }
        if self.output.overlaps(&self.input) || self.output.overlaps(&self.weight) {
            return Err(invalid("output overlaps a read input"));
        }
        self.shape.logical_work()?;
        Ok(())
    }

    pub(crate) fn validate_runtime(&self, runtime: u64) -> Result<(), ReferenceDeviceRuntimeError> {
        for region in [&self.input, &self.weight, &self.output] {
            region.validate_runtime(runtime)?;
        }
        self.validate()
    }
}

// At most 24 active frames for C<=2^24. Leaves of size 1/2 are direct;
// every odd internal node has floor(C/2) elements on its left.
fn pairwise_squares(n: usize, start: usize, read: &impl Fn(usize) -> f32) -> f32 {
    match n {
        1 => {
            let x = read(start);
            x * x
        }
        2 => {
            let x = read(start);
            let y = read(start + 1);
            x * x + y * y
        }
        _ => {
            let left = n / 2;
            pairwise_squares(left, start, read) + pairwise_squares(n - left, start + left, read)
        }
    }
}

pub(super) fn execute(launch: &ReferenceZeroCenteredRmsNormLaunch) {
    let mut storage = AllocationLocks::new([&launch.input, &launch.weight, &launch.output]);
    let shape = launch.shape;
    let epsilon = (shape.epsilon.numerator() as f64 / shape.epsilon.denominator() as f64) as f32;
    for row in 0..shape.rows {
        let start = row * shape.columns;
        let sum = pairwise_squares(shape.columns, start, &|index| {
            bf16::from_bits(storage.read_bits(0, index)).to_f32()
        });
        let mean = sum / shape.columns as f32;
        let inverse = 1.0_f32 / (mean + epsilon).sqrt();
        for column in 0..shape.columns {
            let index = start + column;
            let x = bf16::from_bits(storage.read_bits(0, index)).to_f32();
            let weight = bf16::from_bits(storage.read_bits(1, column)).to_f32();
            let normalized = x * inverse;
            let scale = weight + 1.0_f32;
            storage.write_bits(2, index, bf16::from_f32(normalized * scale).to_bits());
        }
    }
}

fn invalid(reason: impl Into<String>) -> ReferenceDeviceRuntimeError {
    ReferenceDeviceRuntimeError::contract(format!(
        "reference BF16 zero-centered RMSNorm: {}",
        reason.into()
    ))
}

#[cfg(test)]
mod tests;
