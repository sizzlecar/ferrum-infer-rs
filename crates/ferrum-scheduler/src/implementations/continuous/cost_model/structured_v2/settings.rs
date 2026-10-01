//! Explicit empirical drift policy. No probability or future timing guarantee.
use super::*;
use std::num::NonZeroU64;

/// Optional extra buffer learned only from the independent residual population.
/// A residual span measures observed variation, not a rate of temporal drift.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum StructuredLearnedDriftV2 {
    #[default]
    #[serde(deserialize_with = "deserialize_disabled")]
    Disabled,
    ObservedResidualSpanV1 {
        maximum_span_margin_ns: NonZeroU64,
    },
}
fn deserialize_disabled<'de, D>(deserializer: D) -> std::result::Result<(), D::Error>
where
    D: serde::Deserializer<'de>,
{
    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct Empty {}
    Empty::deserialize(deserializer).map(|_| ())
}
impl StructuredLearnedDriftV2 {
    pub fn is_disabled(&self) -> bool {
        matches!(self, Self::Disabled)
    }
    /// Stable policy identity shared by live writers and strict replay.
    pub fn signature(&self) -> [u8; 32] {
        use sha2::{Digest, Sha256};
        let mut digest = Sha256::new();
        digest.update(b"ferrum.structured-learned-drift.v1\0");
        match self {
            Self::Disabled => digest.update([0]),
            Self::ObservedResidualSpanV1 {
                maximum_span_margin_ns,
            } => {
                digest.update([1]);
                digest.update(maximum_span_margin_ns.get().to_le_bytes());
            }
        }
        digest.finalize().into()
    }
    pub(super) fn freeze(&self, minimum: i128, maximum: i128) -> Result<u64> {
        match self {
            Self::Disabled => Ok(0),
            Self::ObservedResidualSpanV1 {
                maximum_span_margin_ns,
            } => {
                let span = maximum
                    .checked_sub(minimum)
                    .and_then(|v| u64::try_from(v).ok())
                    .ok_or(StructuredUnknown::Numerical)?;
                if span > maximum_span_margin_ns.get() {
                    // A cap is a fail-closed resource/safety bound, never clipping.
                    return Err(StructuredUnknown::Capacity);
                }
                Ok(span)
            }
        }
    }
}

/// V2-specific settings keep the old V1 numerical and wire contract unchanged.
/// Missing learned_drift preserves the original V2 declaration and hash bytes.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct StructuredSettingsV2 {
    pub min_phase_samples: usize,
    pub min_fit_redundancy: usize,
    pub max_phase_samples: usize,
    pub max_axes: usize,
    pub max_rank: usize,
    pub max_wave_ns: u64,
    pub max_sample_age_ns: u64,
    pub static_margin_ns: u64,
    #[serde(default, skip_serializing_if = "StructuredLearnedDriftV2::is_disabled")]
    pub learned_drift: StructuredLearnedDriftV2,
}
impl Default for StructuredSettingsV2 {
    fn default() -> Self {
        super::super::structured::StructuredSettingsV1::default().into()
    }
}
impl From<super::super::structured::StructuredSettingsV1> for StructuredSettingsV2 {
    fn from(value: super::super::structured::StructuredSettingsV1) -> Self {
        Self {
            min_phase_samples: value.min_phase_samples,
            min_fit_redundancy: value.min_fit_redundancy,
            max_phase_samples: value.max_phase_samples,
            max_axes: value.max_axes,
            max_rank: value.max_rank,
            max_wave_ns: value.max_wave_ns,
            max_sample_age_ns: value.max_sample_age_ns,
            static_margin_ns: value.static_margin_ns,
            learned_drift: StructuredLearnedDriftV2::Disabled,
        }
    }
}
impl StructuredSettingsV2 {
    pub fn validate(&self) -> Result<()> {
        super::super::structured::StructuredSettingsV1 {
            min_phase_samples: self.min_phase_samples,
            min_fit_redundancy: self.min_fit_redundancy,
            max_phase_samples: self.max_phase_samples,
            max_axes: self.max_axes,
            max_rank: self.max_rank,
            max_wave_ns: self.max_wave_ns,
            max_sample_age_ns: self.max_sample_age_ns,
            static_margin_ns: self.static_margin_ns,
        }
        .validate()?;
        if let StructuredLearnedDriftV2::ObservedResidualSpanV1 {
            maximum_span_margin_ns,
        } = self.learned_drift
        {
            if maximum_span_margin_ns.get() > self.max_wave_ns {
                return Err(StructuredUnknown::InvalidSettings);
            }
        }
        Ok(())
    }
}
