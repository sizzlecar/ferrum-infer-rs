//! Head-local FP32 pairwise RMS reduction with two explicit BF16 boundaries.
//! Numerical nonfinite/overflow propagation is IEEE; preflight checks structure.

use super::{
    bf16, AllocationLocks, DeviceCommandLogicalWork, ElementType, ReferenceBufferRegion,
    ReferenceDeviceRuntimeError,
};
use ferrum_runtime::{CanonicalRational, HEAD_RMS_NORM_BF16_MAX_HEAD_DIM};

#[derive(Clone, Copy)]
pub(crate) struct ReferenceHeadRmsNormShape {
    pub(crate) rows: usize,
    pub(crate) columns: usize,
    pub(crate) head_dim: usize,
    pub(crate) epsilon: CanonicalRational,
}

impl ReferenceHeadRmsNormShape {
    pub(crate) fn byte_extents(self) -> Result<(usize, usize), ReferenceDeviceRuntimeError> {
        if self.rows == 0
            || self.columns == 0
            || self.head_dim == 0
            || self.head_dim as u128 > u128::from(HEAD_RMS_NORM_BF16_MAX_HEAD_DIM)
            || self.columns % self.head_dim != 0
        {
            return Err(invalid(
                "requires positive rows, columns divisible by head_dim, and head_dim in 1..=2^24",
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
        let gamma = self
            .head_dim
            .checked_mul(2)
            .ok_or_else(|| invalid("gamma byte extent overflows usize"))?;
        Ok((matrix, gamma))
    }

    pub(crate) fn logical_work(
        self,
    ) -> Result<DeviceCommandLogicalWork, ReferenceDeviceRuntimeError> {
        self.byte_extents()?;
        let rows = u64::try_from(self.rows).map_err(|_| invalid("rows exceed u64"))?;
        let columns = u64::try_from(self.columns).map_err(|_| invalid("columns exceed u64"))?;
        let head_dim = u64::try_from(self.head_dim).map_err(|_| invalid("head_dim exceeds u64"))?;
        let elements = rows
            .checked_mul(columns)
            .ok_or_else(|| invalid("element count overflows"))?;
        let operations = elements
            .checked_div(head_dim)
            .and_then(|heads| heads.checked_mul(head_dim.checked_mul(4)?.checked_add(3)?))
            .ok_or_else(|| invalid("logical arithmetic count overflows"))?;
        // Per head: D squares, D-1 additions, divide/eps-add/sqrt/reciprocal,
        // D normalization products and D gamma products. Conversions excluded.
        // This is logical arithmetic, not an estimate of hardware FLOPs.
        DeviceCommandLogicalWork::tensor(elements, operations).map_err(|e| invalid(e.to_string()))
    }
}

pub(crate) struct ReferenceHeadRmsNormLaunch {
    pub(crate) input: ReferenceBufferRegion,
    pub(crate) gamma: ReferenceBufferRegion,
    pub(crate) output: ReferenceBufferRegion,
    pub(crate) shape: ReferenceHeadRmsNormShape,
}

impl ReferenceHeadRmsNormLaunch {
    pub(crate) fn validate(&self) -> Result<(), ReferenceDeviceRuntimeError> {
        let (matrix, gamma) = self.shape.byte_extents()?;
        for (region, bytes) in [
            (&self.input, matrix),
            (&self.gamma, gamma),
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
        if self.output.overlaps(&self.input) || self.output.overlaps(&self.gamma) {
            return Err(invalid("output overlaps a read input"));
        }
        self.shape.logical_work()?;
        Ok(())
    }

    pub(crate) fn validate_runtime(&self, runtime: u64) -> Result<(), ReferenceDeviceRuntimeError> {
        for region in [&self.input, &self.gamma, &self.output] {
            region.validate_runtime(runtime)?;
        }
        self.validate()
    }
}

// At most 24 active frames for D<=2^24. Leaves of size 1/2 are direct;
// every odd internal node has floor(D/2) elements on its left.
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

pub(super) fn execute(launch: &ReferenceHeadRmsNormLaunch) {
    let mut storage = AllocationLocks::new([&launch.input, &launch.gamma, &launch.output]);
    let shape = launch.shape;
    let epsilon = (shape.epsilon.numerator() as f64 / shape.epsilon.denominator() as f64) as f32;
    let head_count = shape.rows * (shape.columns / shape.head_dim);
    for head in 0..head_count {
        let start = head * shape.head_dim;
        let sum = pairwise_squares(shape.head_dim, start, &|index| {
            bf16::from_bits(storage.read_bits(0, index)).to_f32()
        });
        let mean = sum / shape.head_dim as f32;
        let inverse = 1.0_f32 / (mean + epsilon).sqrt();
        for column in 0..shape.head_dim {
            let index = start + column;
            let x = bf16::from_bits(storage.read_bits(0, index)).to_f32();
            let gamma = bf16::from_bits(storage.read_bits(1, column)).to_f32();
            let normalized = bf16::from_f32(x * inverse).to_f32();
            storage.write_bits(2, index, bf16::from_f32(normalized * gamma).to_bits());
        }
    }
}

fn invalid(reason: impl Into<String>) -> ReferenceDeviceRuntimeError {
    ReferenceDeviceRuntimeError::contract(format!("reference BF16 head RMSNorm: {}", reason.into()))
}

#[cfg(test)]
mod tests;
