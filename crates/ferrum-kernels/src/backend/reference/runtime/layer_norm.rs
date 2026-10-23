//! BF16 storage, two-pass FP32 population LayerNorm without affine parameters.
//!
//! Submission preflight is structural. Input may be produced by an earlier
//! command in the same batch, so it cannot be scanned for numerical validity
//! before submission. Execution follows IEEE arithmetic, including nonfinite
//! propagation; a numerical oracle must check its own finite/overflow domain.

use ferrum_runtime::{CanonicalRational, LAYER_NORM_BF16_MAX_HIDDEN_SIZE};

use super::{
    bf16, AllocationLocks, DeviceCommandLogicalWork, ElementType, ReferenceBufferRegion,
    ReferenceDeviceRuntimeError,
};

pub(crate) struct ReferenceLayerNormLaunch {
    pub(crate) input: ReferenceBufferRegion,
    pub(crate) output: ReferenceBufferRegion,
    pub(crate) rows: usize,
    pub(crate) hidden_size: usize,
    pub(crate) epsilon: CanonicalRational,
}

impl ReferenceLayerNormLaunch {
    pub(crate) fn validate(&self) -> Result<(), ReferenceDeviceRuntimeError> {
        let bytes = validate_parameters(self.rows, self.hidden_size, self.epsilon)?;
        for region in [&self.input, &self.output] {
            region.validate_range()?;
            if region.element_type != ElementType::Bf16
                || region.length_bytes != bytes
                || (region.offset_bytes - region.allocation.logical_offset_bytes) % 2 != 0
            {
                return Err(ReferenceDeviceRuntimeError::contract(
                    "reference BF16 layer-norm shape, range, alignment or scalar type differs",
                ));
            }
        }
        if self.output.overlaps(&self.input) {
            return Err(ReferenceDeviceRuntimeError::contract(
                "reference BF16 layer-norm output overlaps its input",
            ));
        }
        Ok(())
    }

    pub(super) fn logical_work(
        &self,
    ) -> Result<DeviceCommandLogicalWork, ReferenceDeviceRuntimeError> {
        let count = u64::try_from(self.hidden_size)
            .ok()
            .zip(u64::try_from(self.rows).ok())
            .and_then(|(width, rows)| {
                Some((
                    rows.checked_mul(width)?,
                    rows.checked_mul(width.checked_mul(6)?.checked_add(3)?)?,
                ))
            })
            .ok_or_else(|| {
                ReferenceDeviceRuntimeError::contract(
                    "reference BF16 layer-norm logical work overflows",
                )
            })?;
        // Scalar logical arithmetic: add/sub/mul/div/sqrt each count once;
        // BF16 conversion does not. This is not a hardware FLOP estimate.
        DeviceCommandLogicalWork::tensor(count.0, count.1)
            .map_err(|error| ReferenceDeviceRuntimeError::contract(error.to_string()))
    }
}

fn validate_parameters(
    rows: usize,
    hidden_size: usize,
    epsilon: CanonicalRational,
) -> Result<usize, ReferenceDeviceRuntimeError> {
    if rows == 0
        || hidden_size == 0
        || hidden_size as u128 > u128::from(LAYER_NORM_BF16_MAX_HIDDEN_SIZE)
    {
        return Err(ReferenceDeviceRuntimeError::contract(
            "reference BF16 layer-norm requires positive rows and width in 1..=2^24",
        ));
    }
    let numerator = u128::try_from(epsilon.numerator()).unwrap_or(0);
    let denominator = u128::from(epsilon.denominator());
    if numerator == 0 || numerator > denominator || numerator * 1_000_000_000_000 < denominator {
        return Err(ReferenceDeviceRuntimeError::contract(
            "reference BF16 layer-norm epsilon must be rational in [1e-12, 1]",
        ));
    }
    rows.checked_mul(hidden_size)
        .and_then(|elements| elements.checked_mul(2))
        .ok_or_else(|| {
            ReferenceDeviceRuntimeError::contract("reference BF16 layer-norm byte extent overflows")
        })
}

/// There are at most 24 active recursion frames for n <= 2^24: lengths 1
/// and 2 are direct leaves. Odd lengths put floor(n/2) elements on the left.
/// Each internal addition and leaf transform rounds independently to FP32.
fn pairwise_sum(n: usize, start: usize, leaf: &impl Fn(usize) -> f32) -> f32 {
    match n {
        1 => leaf(start),
        2 => leaf(start) + leaf(start + 1),
        _ => {
            let left_len = n / 2;
            let left = pairwise_sum(left_len, start, leaf);
            let right = pairwise_sum(n - left_len, start + left_len, leaf);
            left + right
        }
    }
}

pub(super) fn execute(launch: &ReferenceLayerNormLaunch) {
    let mut storage = AllocationLocks::new([&launch.input, &launch.output]);
    let epsilon = (launch.epsilon.numerator() as f64 / launch.epsilon.denominator() as f64) as f32;
    // Width <= 2^24 makes this conversion exact.
    let width = launch.hidden_size as f32;
    for row in 0..launch.rows {
        let start = row * launch.hidden_size;
        let mean = pairwise_sum(launch.hidden_size, start, &|index| {
            bf16::from_bits(storage.read_bits(0, index)).to_f32()
        }) / width;
        let variance = pairwise_sum(launch.hidden_size, start, &|index| {
            let centered = bf16::from_bits(storage.read_bits(0, index)).to_f32() - mean;
            centered * centered
        }) / width;
        let reciprocal = 1.0_f32 / (variance + epsilon).sqrt();
        for column in 0..launch.hidden_size {
            let index = start + column;
            let centered = bf16::from_bits(storage.read_bits(0, index)).to_f32() - mean;
            let normalized = centered * reciprocal;
            storage.write_bits(1, index, bf16::from_f32(normalized).to_bits());
        }
    }
}

#[cfg(test)]
mod tests;
