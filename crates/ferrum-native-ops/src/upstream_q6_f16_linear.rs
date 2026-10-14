//! Independent Q6_K F16-boundary D4/MMQ ABI. No device calls or route qualification here.
use crate::upstream_linear::{UpstreamLinearDevice, UpstreamLinearError};
use std::mem::{align_of, offset_of, size_of};

pub const UPSTREAM_Q6_F16_LINEAR_OPERATOR: &str =
    crate::upstream_q6_f32_linear::UPSTREAM_Q6_F32_LINEAR_OPERATOR;
pub const UPSTREAM_Q6_F16_LINEAR_ABI: u32 = 2;
pub const UPSTREAM_Q6_F16_LINEAR_EXPORTS: &[&str] = &[
    "ferrum_upstream_q6_f16_plan_v1",
    "ferrum_upstream_q6_f16_pack_v1",
    "ferrum_upstream_q6_f16_dot_v1",
    "ferrum_upstream_q6_f16_check_weights_v1",
    "ferrum_upstream_q6_f16_cast_v1",
];

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[repr(C)]
pub struct UpstreamQ6F16RequestV1 {
    pub abi: u32,
    pub size: u32,
    pub format: u32,
    pub layout: u32,
    pub rows: u32,
    pub inputs: u32,
    pub outputs: u32,
    pub cc: u32,
    pub sm_count: u32,
    pub reserved: u32,
    pub shared_limit: u64,
}
impl UpstreamQ6F16RequestV1 {
    pub fn new(
        rows: u32,
        inputs: u32,
        outputs: u32,
        device: UpstreamLinearDevice,
    ) -> Result<Self, UpstreamLinearError> {
        let value = Self {
            abi: UPSTREAM_Q6_F16_LINEAR_ABI,
            size: size_of::<Self>() as u32,
            format: 14,
            layout: 0,
            rows,
            inputs,
            outputs,
            cc: device.architecture,
            sm_count: device.multiprocessors,
            reserved: 0,
            shared_limit: device.maximum_dynamic_shared_bytes,
        };
        value.validate()?;
        Ok(value)
    }

    pub fn validate(&self) -> Result<(), UpstreamLinearError> {
        if self.abi != UPSTREAM_Q6_F16_LINEAR_ABI
            || self.size as usize != size_of::<Self>()
            || self.format != 14
            || self.layout != 0
            || !(1..=32).contains(&self.rows)
            || self.inputs == 0
            || self.inputs % 256 != 0
            || self.outputs == 0
            || self.cc < 800
            || self.sm_count == 0
            || self.shared_limit == 0
            || self.reserved != 0
        {
            return Err(UpstreamLinearError::InvalidArgument);
        }
        let (m, k, n) = (
            u64::from(self.rows),
            u64::from(self.inputs),
            u64::from(self.outputs),
        );
        let pk = (k + 511) & !511;
        if m * k > i32::MAX as u64
            || m * n > i32::MAX as u64
            || n * (k / 256) > i32::MAX as u64
            || k + 512 > i32::MAX as u64
            || m * pk * 9 / 8 > i32::MAX as u64
        {
            return Err(UpstreamLinearError::Extent);
        }
        Ok(())
    }
}

/// Independently typed F16 boundary plan. `output_bytes` describes the F32
/// intermediate consumed by the final F16 cast, not the strided destination.
/// Independently typed raw plan. Every native launch reconstructs all fields;
/// this supplementary validation neither owns memory nor certifies its leases.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[repr(C)]
pub struct UpstreamQ6F16PlanV1 {
    pub request: UpstreamQ6F16RequestV1,
    pub abi: u32,
    pub size: u32,
    pub algorithm: u32,
    pub pack_abi: u32,
    pub padded_inputs: u32,
    pub padded_outputs: u32,
    pub guard_blocks: u32,
    pub j: u32,
    pub i: u32,
    pub nthreads: u32,
    pub shared_bytes: u32,
    pub blocks: u32,
    pub tiles_y: u32,
    pub fixup: u32,
    pub ncols: u32,
    pub channels: u32,
    pub nwarps: u32,
    pub rows_per_block: u32,
    pub small_k: u32,
    pub reserved: u32,
    pub weight_bytes: u64,
    pub converted_bytes: u64,
    pub packed_bytes: u64,
    pub output_bytes: u64,
    pub fixup_bytes: u64,
}
impl UpstreamQ6F16PlanV1 {
    pub fn validate_identity(
        &self,
        request: &UpstreamQ6F16RequestV1,
    ) -> Result<(), UpstreamLinearError> {
        request.validate()?;
        let (m, k, n) = (
            u64::from(request.rows),
            u64::from(request.inputs),
            u64::from(request.outputs),
        );
        let pk = (k + 511) & !511;
        if self.request != *request
            || self.abi != UPSTREAM_Q6_F16_LINEAR_ABI
            || self.size as usize != size_of::<Self>()
            || self.algorithm != 1
            || self.pack_abi != 1
            || u64::from(self.padded_inputs) != pk
            || self.padded_outputs != request.outputs
            || !matches!(self.j, 8 | 16 | 24 | 32)
            || self.i != 128
            || self.nthreads != 256
            || self.guard_blocks > 512
            || self.blocks == 0
            || u64::from(self.tiles_y) != n.div_ceil(128)
            || self.fixup > 1
            || u64::from(self.shared_bytes) > request.shared_limit
            || [
                self.ncols,
                self.channels,
                self.nwarps,
                self.rows_per_block,
                self.small_k,
                self.reserved,
            ] != [0; 6]
        {
            return Err(UpstreamLinearError::AbiMismatch);
        }
        let j = u64::from(self.j);
        let read_bytes = (j * 36).div_ceil(256) * 256 * 4;
        let read_end = ((k / 128 - 1) * m + (m.div_ceil(j) - 1) * j) * 144 + read_bytes;
        let expected_packed = m * (pk / 128) * 144 + u64::from(self.guard_blocks) * 144;
        let expected_fixup = if self.fixup != 0 {
            u64::from(self.blocks) * j * 128 * 4
        } else {
            0
        };
        if self.weight_bytes != n * (k / 256) * 210
            || self.converted_bytes != m * pk * 4
            || self.output_bytes != m * n * 4
            || self.packed_bytes != expected_packed
            || self.packed_bytes < read_end
            || self.fixup_bytes != expected_fixup
        {
            return Err(UpstreamLinearError::AbiMismatch);
        }
        Ok(())
    }
}

const _: () = {
    assert!(size_of::<UpstreamQ6F16RequestV1>() == 48);
    assert!(size_of::<UpstreamQ6F16PlanV1>() == 168);
    assert!(align_of::<UpstreamQ6F16PlanV1>() == 8);
    assert!(offset_of!(UpstreamQ6F16RequestV1, shared_limit) == 40);
    assert!(offset_of!(UpstreamQ6F16PlanV1, abi) == 48);
    assert!(offset_of!(UpstreamQ6F16PlanV1, weight_bytes) == 128);
};

#[cfg(test)]
mod tests;
