//! Separate large-row qualification. The small-row request stays closed.
use super::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UpstreamExtraLinearPrefillRequestV2(UpstreamExtraLinearRequestV1);
impl UpstreamExtraLinearPrefillRequestV2 {
    pub fn new(
        format: UpstreamExtraLinearFormat,
        rows: u32,
        inputs: u32,
        outputs: u32,
        device: UpstreamLinearDevice,
    ) -> Result<Self, UpstreamLinearError> {
        let value = Self(UpstreamExtraLinearRequestV1(UpstreamLinearRequestV1 {
            abi: UPSTREAM_LINEAR_ABI,
            size: size_of::<UpstreamLinearRequestV1>() as u32,
            format: format as u32,
            layout: UpstreamLinearLayout::Columns as u32,
            rows,
            inputs,
            outputs,
            cc: device.architecture,
            sm_count: device.multiprocessors,
            reserved: 0,
            shared_limit: device.maximum_dynamic_shared_bytes,
        }));
        value.validate()?;
        Ok(value)
    }
    pub fn as_raw(&self) -> &UpstreamLinearRequestV1 {
        self.0.as_raw()
    }
    pub fn validate(&self) -> Result<(), UpstreamLinearError> {
        self.0.validate_domain(33, 2048, true)?;
        let r = self.as_raw();
        let (m, k, n) = (u64::from(r.rows), u64::from(r.inputs), u64::from(r.outputs));
        let qk = if r.format == 20 { 32 } else { 256 };
        let kp = k.div_ceil(512) * 512;
        // Mirror native int-index and fastdiv domains before calling CUDA.
        if k + 512 > i32::MAX as u64
            || m * k > i32::MAX as u64
            || m * n > i32::MAX as u64
            || n * (k / qk) > i32::MAX as u64
            || m * kp * 9 / 8 > i32::MAX as u64
            || m.div_ceil(32) * n.div_ceil(128) * (k / qk) >= (1 << 30)
        {
            return Err(UpstreamLinearError::Extent);
        }
        Ok(())
    }
    /// Five aligned workspace regions. The retained weight flag is separate.
    pub fn marker_scratch_upper_bound(&self) -> Result<u64, UpstreamLinearError> {
        self.validate()?;
        let r = self.as_raw();
        let kp = u64::from(r.inputs).div_ceil(512) * 512;
        Ok(
            u64::from(r.rows) * (41 * kp / 8 + 4 * u64::from(r.outputs) + 4)
                + 16384 * u64::from(r.sm_count)
                + 512 * 144
                + 4 * 15,
        )
    }
    pub fn validate_plan_identity(
        &self,
        p: &UpstreamLinearPlanV1,
    ) -> Result<(), UpstreamLinearError> {
        self.validate()?;
        self.0
            .validate_plan_fields(p, UpstreamLinearAlgorithm::Mmq)?;
        let r = self.as_raw();
        let (m, k, n) = (u64::from(r.rows), u64::from(r.inputs), u64::from(r.outputs));
        let kp = k.div_ceil(512) * 512;
        let tiles_y = n.div_ceil(128);
        let tiles = m.div_ceil(32) * tiles_y;
        let blocks = if p.fixup == 1 {
            u64::from(r.sm_count)
        } else {
            tiles
        };
        if p.j != 32
            || p.i != 128
            || p.nthreads != 256
            || p.fixup > 1
            || p.shared_bytes == 0
            || u64::from(p.shared_bytes) > r.shared_limit
            || u64::from(p.padded_inputs) != kp
            || p.padded_outputs != r.outputs
            || p.guard_blocks > 512
            || p.guard_blocks > r.rows
            || p.guard_blocks % 8 != 0
            || u64::from(p.tiles_y) != tiles_y
            || u64::from(p.blocks) != blocks
            || (p.fixup == 1 && tiles % blocks == 0)
            || p.converted_bytes != 4 * m * kp
            || p.packed_bytes != m * kp * 9 / 8 + u64::from(p.guard_blocks) * 144
            || p.output_bytes != 4 * m * n
            || p.fixup_bytes != if p.fixup == 1 { blocks * 16384 } else { 0 }
            || p.ncols != 0
            || p.channels != 0
            || p.nwarps != 0
            || p.rows_per_block != 0
            || p.small_k != 0
        {
            return Err(UpstreamLinearError::AbiMismatch);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests;
