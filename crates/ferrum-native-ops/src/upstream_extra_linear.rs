//! Independent extra-format ABI over the shared repr(C) geometry.
//! This does not widen the original operator's accepted format or row domain.
use crate::upstream_linear::*;
use std::mem::size_of;

pub const UPSTREAM_EXTRA_LINEAR_OPERATOR: &str = "ferrum.cuda.upstream_extra_linear";
mod prefill;
pub use prefill::UpstreamExtraLinearPrefillRequestV2;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum UpstreamExtraLinearFormat {
    Q3K = 11,
    Iq4Nl = 20,
    Iq3S = 21,
}
impl UpstreamExtraLinearFormat {
    pub const fn block_elements(self) -> u32 {
        match self {
            Self::Iq4Nl => 32,
            Self::Q3K | Self::Iq3S => 256,
        }
    }
    pub const fn block_bytes(self) -> u32 {
        match self {
            Self::Iq4Nl => 18,
            Self::Q3K | Self::Iq3S => 110,
        }
    }
}

/// A typed request for the independently linked operator. The shared raw ABI
/// is read-only; construction and validation never call the old format checker.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(transparent)]
pub struct UpstreamExtraLinearRequestV1(UpstreamLinearRequestV1);
impl UpstreamExtraLinearRequestV1 {
    pub fn new(
        format: UpstreamExtraLinearFormat,
        layout: UpstreamLinearLayout,
        rows: u32,
        inputs: u32,
        outputs: u32,
        device: UpstreamLinearDevice,
    ) -> Result<Self, UpstreamLinearError> {
        let value = Self(UpstreamLinearRequestV1 {
            abi: UPSTREAM_LINEAR_ABI,
            size: size_of::<UpstreamLinearRequestV1>() as u32,
            format: format as u32,
            layout: layout as u32,
            rows,
            inputs,
            outputs,
            cc: device.architecture,
            sm_count: device.multiprocessors,
            reserved: 0,
            shared_limit: device.maximum_dynamic_shared_bytes,
        });
        value.validate()?;
        Ok(value)
    }
    pub fn as_raw(&self) -> &UpstreamLinearRequestV1 {
        &self.0
    }
    pub fn validate(&self) -> Result<(), UpstreamLinearError> {
        self.validate_domain(1, 32, false)
    }
    fn validate_domain(
        &self,
        first: u32,
        last: u32,
        columns_only: bool,
    ) -> Result<(), UpstreamLinearError> {
        let r = &self.0;
        if r.abi != UPSTREAM_LINEAR_ABI
            || r.size as usize != size_of::<UpstreamLinearRequestV1>()
            || r.reserved != 0
            || !matches!(r.format, 11 | 20 | 21)
            || r.layout > 1
            || r.rows < first
            || r.rows > last
            || (columns_only && r.layout != UpstreamLinearLayout::Columns as u32)
            || r.inputs == 0
            || r.inputs % 256 != 0
            || r.outputs == 0
            || r.cc < 800
            || r.sm_count == 0
            || r.shared_limit == 0
        {
            return Err(UpstreamLinearError::InvalidArgument);
        }
        Ok(())
    }
    pub fn validate_plan_identity(
        &self,
        plan: &UpstreamLinearPlanV1,
        algorithm: UpstreamLinearAlgorithm,
    ) -> Result<(), UpstreamLinearError> {
        self.validate()?;
        self.validate_plan_fields(plan, algorithm)
    }
    fn validate_plan_fields(
        &self,
        plan: &UpstreamLinearPlanV1,
        algorithm: UpstreamLinearAlgorithm,
    ) -> Result<(), UpstreamLinearError> {
        let r = &self.0;
        let format = match r.format {
            11 => UpstreamExtraLinearFormat::Q3K,
            20 => UpstreamExtraLinearFormat::Iq4Nl,
            21 => UpstreamExtraLinearFormat::Iq3S,
            _ => return Err(UpstreamLinearError::InvalidArgument),
        };
        let weight_bytes = u64::from(plan.padded_outputs)
            .checked_mul(u64::from(r.inputs / format.block_elements()))
            .and_then(|v| v.checked_mul(u64::from(format.block_bytes())))
            .ok_or(UpstreamLinearError::Extent)?;
        if plan.request != *r
            || plan.abi != UPSTREAM_LINEAR_ABI
            || plan.size as usize != size_of::<UpstreamLinearPlanV1>()
            || plan.reserved != 0
            || plan.algorithm != algorithm as u32
            || plan.padded_inputs < r.inputs
            || plan.padded_inputs % 512 != 0
            || plan.padded_outputs < r.outputs
            || plan.pack_abi
                != match algorithm {
                    UpstreamLinearAlgorithm::Mmq => 1,
                    UpstreamLinearAlgorithm::Mmvq => 3,
                }
            || plan.weight_bytes != weight_bytes
        {
            return Err(UpstreamLinearError::AbiMismatch);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn device() -> UpstreamLinearDevice {
        UpstreamLinearDevice {
            architecture: 800,
            multiprocessors: 1,
            maximum_dynamic_shared_bytes: 65536,
        }
    }
    fn request(format: UpstreamExtraLinearFormat, rows: u32) -> UpstreamExtraLinearRequestV1 {
        UpstreamExtraLinearRequestV1::new(
            format,
            UpstreamLinearLayout::Columns,
            rows,
            768,
            17,
            device(),
        )
        .unwrap()
    }
    #[test]
    fn extra_requests_do_not_widen_original_format_or_prefill_domain() {
        for format in [
            UpstreamExtraLinearFormat::Q3K,
            UpstreamExtraLinearFormat::Iq3S,
            UpstreamExtraLinearFormat::Iq4Nl,
        ] {
            for rows in [1, 4, 8, 16, 32] {
                let r = request(format, rows);
                assert!(r.validate().is_ok());
                assert!(r.as_raw().validate().is_err());
                let mut wrong = r;
                wrong.0.format = 12;
                assert!(wrong.validate().is_err());
            }
            for rows in [0, 33, 2048] {
                assert!(UpstreamExtraLinearRequestV1::new(
                    format,
                    UpstreamLinearLayout::Columns,
                    rows,
                    768,
                    17,
                    device()
                )
                .is_err());
            }
        }
    }
    #[test]
    fn extra_plan_checks_d4_and_actual_weight_block_length() {
        for format in [
            UpstreamExtraLinearFormat::Q3K,
            UpstreamExtraLinearFormat::Iq3S,
            UpstreamExtraLinearFormat::Iq4Nl,
        ] {
            let r = request(format, 8);
            for algorithm in [UpstreamLinearAlgorithm::Mmq, UpstreamLinearAlgorithm::Mmvq] {
                let p = UpstreamLinearPlanV1 {
                    request: *r.as_raw(),
                    abi: 1,
                    size: size_of::<UpstreamLinearPlanV1>() as u32,
                    algorithm: algorithm as u32,
                    pack_abi: if algorithm == UpstreamLinearAlgorithm::Mmq {
                        1
                    } else {
                        3
                    },
                    padded_inputs: 1024,
                    padded_outputs: 18,
                    weight_bytes: 18
                        * u64::from(768 / format.block_elements())
                        * u64::from(format.block_bytes()),
                    ..Default::default()
                };
                r.validate_plan_identity(&p, algorithm).unwrap();
                let mut wrong = p;
                wrong.pack_abi = 2;
                assert_eq!(
                    r.validate_plan_identity(&wrong, algorithm),
                    Err(UpstreamLinearError::AbiMismatch)
                );
                wrong = p;
                wrong.weight_bytes += 1;
                assert!(r.validate_plan_identity(&wrong, algorithm).is_err());
                if format == UpstreamExtraLinearFormat::Iq4Nl {
                    wrong = p;
                    wrong.weight_bytes = 18 * (768 / 256) * 18;
                    assert!(r.validate_plan_identity(&wrong, algorithm).is_err());
                }
            }
        }
    }
}
