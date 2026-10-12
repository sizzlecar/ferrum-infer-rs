//! Versioned, CUDA-independent ABI for the optional upstream linear artifact.
//! V1 geometry is shared by V1 arithmetic and MarkerV2; a plan alone does not
//! certify either numerical domain, own memory, or authorize a provider route.
use std::mem::{align_of, offset_of, size_of};

pub const UPSTREAM_LINEAR_ABI: u32 = 1;
pub const UPSTREAM_LINEAR_OPERATOR: &str = "ferrum.cuda.upstream_linear";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum UpstreamLinearAlgorithm {
    Mmq = 1,
    Mmvq = 2,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum UpstreamLinearFormat {
    /// Independent MMQ MarkerV2 pilot; the original three-format exports
    /// continue to reject this wire format.
    #[allow(non_camel_case_types)]
    Q8_0 = 8,
    Q4K = 12,
    Q5K = 13,
    Iq4Xs = 23,
}
impl UpstreamLinearFormat {
    pub const fn block_elements(self) -> u32 {
        match self {
            Self::Q8_0 => 32,
            Self::Q4K | Self::Q5K | Self::Iq4Xs => 256,
        }
    }

    pub const fn block_bytes(self) -> u32 {
        match self {
            Self::Q8_0 => 34,
            Self::Q4K => 144,
            Self::Q5K => 176,
            Self::Iq4Xs => 136,
        }
    }
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum UpstreamLinearLayout {
    Columns = 0,
    Channels = 1,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UpstreamLinearArithmetic {
    V1,
    MarkerV2,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UpstreamLinearDevice {
    /// CUDA major*100 + minor*10, e.g. 890 or 1200.
    pub architecture: u32,
    pub multiprocessors: u32,
    pub maximum_dynamic_shared_bytes: u64,
}

/// Raw FFI request; ordinary callers construct it using `new`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[repr(C)]
pub struct UpstreamLinearRequestV1 {
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
impl UpstreamLinearRequestV1 {
    pub fn new(
        format: UpstreamLinearFormat,
        layout: UpstreamLinearLayout,
        rows: u32,
        inputs: u32,
        outputs: u32,
        device: UpstreamLinearDevice,
    ) -> Result<Self, UpstreamLinearError> {
        let value = Self {
            abi: UPSTREAM_LINEAR_ABI,
            size: size_of::<Self>() as u32,
            format: format as u32,
            layout: layout as u32,
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
        if self.abi != UPSTREAM_LINEAR_ABI
            || self.size as usize != size_of::<Self>()
            || self.reserved != 0
            || !matches!(self.format, 8 | 12 | 13 | 23)
            || self.layout > 1
            || self.rows == 0
            || self.rows > 2048
            || self.inputs == 0
            || self.inputs % 256 != 0
            || self.outputs == 0
            || self.cc < 800
            || self.sm_count == 0
            || self.shared_limit == 0
        {
            return Err(UpstreamLinearError::InvalidArgument);
        }
        if self.format == UpstreamLinearFormat::Q8_0 as u32
            && (!(2..=32).contains(&self.rows)
                || self.layout != UpstreamLinearLayout::Columns as u32)
        {
            return Err(UpstreamLinearError::UnsupportedGeometry);
        }
        Ok(())
    }

    /// Conservative five-region MarkerV2 scratch extent for the explicit
    /// MMQ Columns prefill planner. This does not qualify a device or profile.
    ///
    /// Converted F32 + packed Q8 + F32 output + row flags are linear in M.
    /// Pinned J_max is at most 512 guard blocks. With fixed J32/I128, a
    /// nonempty stream-K fixup uses SM blocks; full-tile dispatch needs none.
    /// Four internal alignments contribute at most 4*15 bytes. The retained
    /// weight-validation flag belongs to Plan storage, not this scratch.
    pub fn mmq_prefill_marker_scratch_upper_bound(&self) -> Result<u64, UpstreamLinearError> {
        self.validate()?;
        if self.layout != UpstreamLinearLayout::Columns as u32 || self.rows <= 32 {
            return Err(UpstreamLinearError::UnsupportedGeometry);
        }
        let m = u64::from(self.rows);
        let k = u64::from(self.inputs);
        let n = u64::from(self.outputs);
        let padded = k.checked_add(511).ok_or(UpstreamLinearError::Extent)? & !511;
        let mul = |a: u64, b: u64| a.checked_mul(b).ok_or(UpstreamLinearError::Extent);
        let add = |a: u64, b: u64| a.checked_add(b).ok_or(UpstreamLinearError::Extent);
        if add(k, 512)? > i32::MAX as u64
            || mul(m, k)? > i32::MAX as u64
            || mul(m, n)? > i32::MAX as u64
            || mul(n, k / 256)? > i32::MAX as u64
            || mul(mul(m, padded)?, 9)? / 8 > i32::MAX as u64
            || mul(mul(m.div_ceil(32), n.div_ceil(128))?, k / 256)? >= 1 << 30
        {
            return Err(UpstreamLinearError::Extent);
        }
        let per_row = add(add(mul(padded, 41)? / 8, mul(n, 4)?)?, 4)?;
        let fixed = add(512 * 144 + 60, mul(u64::from(self.sm_count), 16384)?)?;
        add(mul(m, per_row)?, fixed)
    }
}

/// Byte-for-byte C ABI. Native code re-derives every field before each launch.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[repr(C)]
pub struct UpstreamLinearPlanV1 {
    pub request: UpstreamLinearRequestV1,
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
impl UpstreamLinearPlanV1 {
    /// Supplemental ABI sanity; authoritative geometry is computed by native
    /// code and privately retained by the kernel adapter, not accepted as wire.
    pub fn validate_identity(
        &self,
        request: &UpstreamLinearRequestV1,
        algorithm: UpstreamLinearAlgorithm,
    ) -> Result<(), UpstreamLinearError> {
        request.validate()?;
        if request.format == UpstreamLinearFormat::Q8_0 as u32
            && algorithm != UpstreamLinearAlgorithm::Mmq
        {
            return Err(UpstreamLinearError::UnsupportedGeometry);
        }
        if self.request != *request
            || self.abi != UPSTREAM_LINEAR_ABI
            || self.size as usize != size_of::<Self>()
            || self.reserved != 0
            || self.algorithm != algorithm as u32
            || self.padded_inputs < request.inputs
            || self.padded_inputs % 512 != 0
            || self.padded_outputs < request.outputs
        {
            return Err(UpstreamLinearError::AbiMismatch);
        }
        let expected_pack = match algorithm {
            UpstreamLinearAlgorithm::Mmq => {
                if matches!(request.format, 8 | 23) {
                    1
                } else {
                    2
                }
            }
            UpstreamLinearAlgorithm::Mmvq => 3,
        };
        if self.pack_abi != expected_pack {
            return Err(UpstreamLinearError::AbiMismatch);
        }
        if request.format == UpstreamLinearFormat::Q8_0 as u32 {
            let mul = |a: u64, b: u64| a.checked_mul(b).ok_or(UpstreamLinearError::Extent);
            let rows = u64::from(request.rows);
            let outputs = u64::from(request.outputs);
            let padded_inputs = u64::from(self.padded_inputs);
            let weight_bytes = mul(mul(outputs, u64::from(request.inputs / 32))?, 34)?;
            let converted_bytes = mul(mul(rows, padded_inputs)?, 4)?;
            // D4 has four independent FP32 scales and 128 signed Q8 values.
            let packed_bytes = mul(mul(rows, padded_inputs / 128)?, 144)?
                .checked_add(mul(u64::from(self.guard_blocks), 144)?)
                .ok_or(UpstreamLinearError::Extent)?;
            let output_bytes = mul(mul(rows, outputs)?, 4)?;
            let fixup_bytes = if self.fixup == 0 {
                0
            } else {
                mul(
                    mul(
                        mul(u64::from(self.blocks), u64::from(self.j))?,
                        u64::from(self.i),
                    )?,
                    4,
                )?
            };
            if self.padded_outputs != request.outputs
                || self.weight_bytes != weight_bytes
                || self.converted_bytes != converted_bytes
                || self.packed_bytes != packed_bytes
                || self.output_bytes != output_bytes
                || self.fixup_bytes != fixup_bytes
                || self.ncols != 0
                || self.channels != 0
                || self.nwarps != 0
                || self.rows_per_block != 0
                || self.small_k != 0
            {
                return Err(UpstreamLinearError::AbiMismatch);
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum UpstreamLinearError {
    #[error("upstream linear artifact feature is unavailable")]
    Unavailable,
    #[error("invalid upstream linear argument or ABI version")]
    InvalidArgument,
    #[error("unsupported upstream linear geometry or device")]
    UnsupportedGeometry,
    #[error("upstream linear byte extent overflow")]
    Extent,
    #[error("upstream linear dispatch is unavailable")]
    Dispatch,
    #[error("upstream linear device span is undersized, overlapping or misaligned")]
    Span,
    #[error("upstream native invariant failed; no fallback was attempted")]
    NativeInvariant,
    #[error("unexpected upstream native exception")]
    NativeException,
    #[error("upstream linear native plan ABI mismatch")]
    AbiMismatch,
    #[error("CUDA runtime returned status {0}")]
    CudaRuntime(i32),
    #[error("unknown upstream native status {0}")]
    UnknownStatus(i32),
}
impl UpstreamLinearError {
    pub fn from_status(status: i32) -> Result<(), Self> {
        Err(match status {
            0 => return Ok(()),
            -1 => Self::InvalidArgument,
            -2 => Self::UnsupportedGeometry,
            -3 => Self::Extent,
            -4 => Self::Dispatch,
            -5 => Self::Span,
            -6 => Self::NativeInvariant,
            -7 => Self::NativeException,
            value if value > 0 => Self::CudaRuntime(value),
            value => Self::UnknownStatus(value),
        })
    }
}

const _: () = {
    assert!(size_of::<UpstreamLinearRequestV1>() == 48);
    assert!(size_of::<UpstreamLinearPlanV1>() == 168);
    assert!(align_of::<UpstreamLinearPlanV1>() == 8);
    assert!(offset_of!(UpstreamLinearRequestV1, shared_limit) == 40);
    assert!(offset_of!(UpstreamLinearPlanV1, abi) == 48);
    assert!(offset_of!(UpstreamLinearPlanV1, weight_bytes) == 128);
};

#[cfg(test)]
mod tests {
    use super::*;
    fn request() -> UpstreamLinearRequestV1 {
        UpstreamLinearRequestV1::new(
            UpstreamLinearFormat::Iq4Xs,
            UpstreamLinearLayout::Columns,
            8,
            5120,
            1024,
            UpstreamLinearDevice {
                architecture: 800,
                multiprocessors: 1,
                maximum_dynamic_shared_bytes: 65536,
            },
        )
        .unwrap()
    }
    fn q8_plan(rows: u32) -> UpstreamLinearPlanV1 {
        let request = UpstreamLinearRequestV1::new(
            UpstreamLinearFormat::Q8_0,
            UpstreamLinearLayout::Columns,
            rows,
            5120,
            48,
            UpstreamLinearDevice {
                architecture: 800,
                multiprocessors: 1,
                maximum_dynamic_shared_bytes: 65536,
            },
        )
        .unwrap();
        UpstreamLinearPlanV1 {
            request,
            abi: UPSTREAM_LINEAR_ABI,
            size: size_of::<UpstreamLinearPlanV1>() as u32,
            algorithm: UpstreamLinearAlgorithm::Mmq as u32,
            pack_abi: 1,
            padded_inputs: 5120,
            padded_outputs: 48,
            guard_blocks: 1,
            weight_bytes: 261120,
            converted_bytes: u64::from(rows) * 5120 * 4,
            packed_bytes: u64::from(rows) * 40 * 144 + 144,
            output_bytes: u64::from(rows) * 48 * 4,
            ..Default::default()
        }
    }

    #[test]
    fn q8_request_rejects_unsupported_rows_layout_and_mmq_quantum() {
        let base = q8_plan(8).request;
        assert_eq!(base.format, 8);
        assert_eq!(UpstreamLinearFormat::Q8_0.block_elements(), 32);
        assert_eq!(UpstreamLinearFormat::Q8_0.block_bytes(), 34);
        for bad in [
            UpstreamLinearRequestV1 { rows: 1, ..base },
            UpstreamLinearRequestV1 { rows: 33, ..base },
            UpstreamLinearRequestV1 {
                layout: UpstreamLinearLayout::Channels as u32,
                ..base
            },
        ] {
            assert_eq!(
                bad.validate(),
                Err(UpstreamLinearError::UnsupportedGeometry)
            );
        }
        // A complete Q8_0 weight block is insufficient for MMQ's K256 tile.
        assert_eq!(
            UpstreamLinearRequestV1 {
                inputs: 5152,
                ..base
            }
            .validate(),
            Err(UpstreamLinearError::InvalidArgument)
        );
    }

    #[test]
    fn q8_plan_rejects_wrong_pack_weight_block_and_scratch_extents() {
        for rows in [8, 32] {
            let plan = q8_plan(rows);
            plan.validate_identity(&plan.request, UpstreamLinearAlgorithm::Mmq)
                .unwrap();
            for bad in [
                UpstreamLinearPlanV1 {
                    pack_abi: 2,
                    ..plan
                },
                UpstreamLinearPlanV1 {
                    weight_bytes: 48 * 20 * 144,
                    ..plan
                },
                UpstreamLinearPlanV1 {
                    padded_outputs: 128,
                    ..plan
                },
                UpstreamLinearPlanV1 {
                    packed_bytes: plan.packed_bytes - 144,
                    ..plan
                },
                UpstreamLinearPlanV1 {
                    converted_bytes: plan.converted_bytes - 4,
                    ..plan
                },
                UpstreamLinearPlanV1 {
                    fixup_bytes: 4,
                    ..plan
                },
            ] {
                assert_eq!(
                    bad.validate_identity(&plan.request, UpstreamLinearAlgorithm::Mmq),
                    Err(UpstreamLinearError::AbiMismatch)
                );
            }
            let fixup = UpstreamLinearPlanV1 {
                blocks: 2,
                j: 32,
                i: 128,
                fixup: 1,
                fixup_bytes: 32768,
                ..plan
            };
            fixup
                .validate_identity(&fixup.request, UpstreamLinearAlgorithm::Mmq)
                .unwrap();
            assert_eq!(
                UpstreamLinearPlanV1 {
                    fixup_bytes: 32764,
                    ..fixup
                }
                .validate_identity(&fixup.request, UpstreamLinearAlgorithm::Mmq),
                Err(UpstreamLinearError::AbiMismatch)
            );
        }
    }

    #[test]
    fn q8_plan_rejects_mmvq_and_overflowing_weight_extent() {
        let plan = q8_plan(8);
        assert_eq!(
            plan.validate_identity(&plan.request, UpstreamLinearAlgorithm::Mmvq),
            Err(UpstreamLinearError::UnsupportedGeometry)
        );
        let request = UpstreamLinearRequestV1 {
            inputs: 0xffff_fe00,
            outputs: u32::MAX,
            ..plan.request
        };
        request.validate().unwrap();
        assert_eq!(
            UpstreamLinearPlanV1 {
                request,
                padded_inputs: request.inputs,
                padded_outputs: request.outputs,
                ..plan
            }
            .validate_identity(&request, UpstreamLinearAlgorithm::Mmq),
            Err(UpstreamLinearError::Extent)
        );
    }
    #[test]
    fn abi_rejects_unknown_reserved_and_partial_quant_blocks() {
        let base = request();
        for bad in [
            UpstreamLinearRequestV1 {
                reserved: 1,
                ..base
            },
            UpstreamLinearRequestV1 { abi: 2, ..base },
            UpstreamLinearRequestV1 {
                inputs: base.inputs + 1,
                ..base
            },
            UpstreamLinearRequestV1 { layout: 2, ..base },
        ] {
            assert_eq!(bad.validate(), Err(UpstreamLinearError::InvalidArgument));
        }
    }
    #[test]
    fn cuda_errors_and_native_invariants_remain_distinct() {
        assert_eq!(
            UpstreamLinearError::from_status(719),
            Err(UpstreamLinearError::CudaRuntime(719))
        );
        assert_eq!(
            UpstreamLinearError::from_status(-6),
            Err(UpstreamLinearError::NativeInvariant)
        );
        assert!(UpstreamLinearError::from_status(0).is_ok());
    }
    #[test]
    fn plan_rejects_pack_kind_or_request_substitution() {
        let r = request();
        let p = UpstreamLinearPlanV1 {
            request: r,
            abi: 1,
            size: 168,
            algorithm: 1,
            pack_abi: 1,
            padded_inputs: r.inputs,
            padded_outputs: r.outputs,
            ..Default::default()
        };
        assert!(p
            .validate_identity(&r, UpstreamLinearAlgorithm::Mmq)
            .is_ok());
        assert!(UpstreamLinearPlanV1 { pack_abi: 3, ..p }
            .validate_identity(&r, UpstreamLinearAlgorithm::Mmq)
            .is_err());
        assert!(p
            .validate_identity(
                &UpstreamLinearRequestV1 { rows: 4, ..r },
                UpstreamLinearAlgorithm::Mmq
            )
            .is_err());
    }

    #[test]
    fn prefill_request_admission_is_bounded_and_scratch_checks_native_index_limits() {
        let base = request();
        for rows in [33, 54, 155, 747, 2048] {
            let r = UpstreamLinearRequestV1 { rows, ..base };
            r.validate().unwrap();
            let bound = r.mmq_prefill_marker_scratch_upper_bound().unwrap();
            assert!(bound >= u64::from(rows) * u64::from(r.inputs) * 4);
        }
        assert!(UpstreamLinearRequestV1 { rows: 2049, ..base }
            .validate()
            .is_err());
        assert!(base.mmq_prefill_marker_scratch_upper_bound().is_err());
        assert!(UpstreamLinearRequestV1 {
            rows: 33,
            layout: UpstreamLinearLayout::Channels as u32,
            ..base
        }
        .mmq_prefill_marker_scratch_upper_bound()
        .is_err());
        for bad in [
            UpstreamLinearRequestV1 {
                rows: 2048,
                inputs: 1 << 24,
                ..base
            },
            UpstreamLinearRequestV1 {
                rows: 2048,
                outputs: 1 << 24,
                ..base
            },
            UpstreamLinearRequestV1 {
                rows: 2048,
                inputs: 1 << 20,
                outputs: 1 << 20,
                ..base
            },
        ] {
            assert_eq!(
                bad.mmq_prefill_marker_scratch_upper_bound(),
                Err(UpstreamLinearError::Extent)
            );
        }
    }
}
