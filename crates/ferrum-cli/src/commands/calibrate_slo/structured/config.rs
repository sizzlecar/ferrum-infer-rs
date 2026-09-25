use super::*;
use ferrum_engine::continuous_engine::{
    StructuredCalibrationOptions, StructuredCalibrationScopeV1,
};
use ferrum_scheduler::implementations::continuous::{
    cost_model::structured::StructuredSettingsV1, cost_profile::CostProfileLoadLimits,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::num::{NonZeroU64, NonZeroUsize};

/// Source, scope and populations are declared before the first live owner.
/// Discover the domain independently; this command never retunes it from fit
/// or qualification outcomes. Members count prepared eligible waves, not tokens.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct CaptureConfig {
    pub profile: PathBuf,
    pub source: PathBuf,
    pub rows: NonZeroUsize,
    pub domain_signature: [u8; 32],
    pub settings: Settings,
    pub fit_members: NonZeroUsize,
    pub residual_members: NonZeroUsize,
    pub qualification_members: NonZeroUsize,
    pub maximum_offered_waves: NonZeroUsize,
    pub maximum_file_bytes: NonZeroU64,
    /// Operator declaration over the complete original capture interval.
    pub declared_source_clock_error_ns: u64,
}

/// Explicit wire settings; defaults come from the numerical core. The fully
/// expanded values are included in the manifest digest and source header.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub(crate) struct Settings {
    pub min_phase_samples: usize,
    pub min_fit_redundancy: usize,
    pub max_phase_samples: usize,
    pub max_axes: usize,
    pub max_rank: usize,
    pub max_wave_ns: u64,
    pub max_sample_age_ns: u64,
    pub static_margin_ns: u64,
}
impl Default for Settings {
    fn default() -> Self {
        let value = StructuredSettingsV1::default();
        Self {
            min_phase_samples: value.min_phase_samples,
            min_fit_redundancy: value.min_fit_redundancy,
            max_phase_samples: value.max_phase_samples,
            max_axes: value.max_axes,
            max_rank: value.max_rank,
            max_wave_ns: value.max_wave_ns,
            max_sample_age_ns: value.max_sample_age_ns,
            static_margin_ns: value.static_margin_ns,
        }
    }
}
impl Settings {
    pub(in crate::commands::calibrate_slo) fn core(&self) -> StructuredSettingsV1 {
        StructuredSettingsV1 {
            min_phase_samples: self.min_phase_samples,
            min_fit_redundancy: self.min_fit_redundancy,
            max_phase_samples: self.max_phase_samples,
            max_axes: self.max_axes,
            max_rank: self.max_rank,
            max_wave_ns: self.max_wave_ns,
            max_sample_age_ns: self.max_sample_age_ns,
            static_margin_ns: self.static_margin_ns,
        }
    }
}
impl CaptureConfig {
    pub(crate) fn validate(&self, manifest: &manifest::Manifest) -> Result<()> {
        self.settings.core().validate().map_err(|reason| {
            FerrumError::config(format!("structured calibration settings: {reason:?}"))
        })?;
        let counts = self.members();
        if self.domain_signature == [0; 32]
            || self.rows.get() > 128
            || self.rows.get() > manifest.protocol.maximum_requests.get()
            || self.maximum_offered_waves.get() > 65_536
            || self.maximum_offered_waves.get() as u64
                > manifest.protocol.maximum_wave_attempts.get()
            || self.maximum_file_bytes.get()
                > ferrum_types::SloCostProfileImportConfig::MAX_FILE_BYTES as u64
            || counts.iter().any(|&n| {
                n < self.settings.min_phase_samples || n > self.settings.max_phase_samples
            })
            || counts[2] < self.rows.get() + 1
            || counts.iter().sum::<usize>() > self.maximum_offered_waves.get()
        {
            return Err(FerrumError::config(
                "structured calibration needs a nonzero independently frozen domain and bounded complete phase populations",
            ));
        }
        if manifest.reference.is_some()
            || manifest.validation_model.residual().is_empty()
            || manifest.validation_model.residual().len() > 256
        {
            return Err(FerrumError::config(
                "structured calibration needs independent bounded residual cohorts; collect discovery/reference in a separate session",
            ));
        }
        Ok(())
    }

    pub(super) fn members(&self) -> [usize; 3] {
        [
            self.fit_members.get(),
            self.residual_members.get(),
            self.qualification_members.get(),
        ]
    }

    pub(super) fn options(
        &self,
        manifest: &manifest::Manifest,
    ) -> Result<StructuredCalibrationOptions> {
        self.validate(manifest)?;
        let protocol = serde_json::to_vec(manifest)
            .map_err(|error| FerrumError::config(format!("encode structured protocol: {error}")))?;
        Ok(StructuredCalibrationOptions {
            observations_path: self.source.clone(),
            protocol_sha256: Sha256::digest(protocol).into(),
            scope: StructuredCalibrationScopeV1 {
                rows: self.rows,
                domain_signature: self.domain_signature,
            },
            settings: self.settings.core(),
            fit_members: self.fit_members,
            residual_members: self.residual_members,
            qualification_members: self.qualification_members,
            maximum_offered_waves: self.maximum_offered_waves,
            maximum_file_bytes: self.maximum_file_bytes,
        })
    }
}

pub(in crate::commands::calibrate_slo) fn export_limits(
    value: &ferrum_types::SloCostProfileImportConfig,
) -> CostProfileLoadLimits {
    CostProfileLoadLimits {
        max_file_bytes: value.max_file_bytes,
        max_samples: value.max_samples,
        max_total_shape_rows: value.max_total_shape_rows,
        max_source_field_bytes: value.max_source_field_bytes,
        max_profile_age_ns: value.max_profile_age_ns,
        max_clock_error_ns: value.max_clock_error_ns,
    }
}
