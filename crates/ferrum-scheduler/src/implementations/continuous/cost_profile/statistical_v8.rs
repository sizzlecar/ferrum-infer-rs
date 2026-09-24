//! Explicit work-support model. V2 family and original numeric evidence stay
//! unchanged; profile7 remains a different model and cannot be relabeled.
use super::super::cost_model::statistical::model::{
    CalibrationPartitionV1, WholeWaveModelRevision, WholeWaveObservationV1, WholeWaveSettingsV1,
};
use super::statistical_v6::{
    loader::{import_selected_profile_revision, load_selected_profile_path},
    ImportedWholeWaveModelV1,
};
use super::statistical_v7::CostProfileFileV7;
use super::*;
pub const COST_PROFILE_SCHEMA_VERSION_V8: u32 = 8;
/// Same bounded record representation, different explicit model semantics.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct CostProfileFileV8(CostProfileFileV7);
impl CostProfileFileV8 {
    pub fn from_capture_observations(
        fingerprint: &ExecutionFingerprint,
        settings: &WholeWaveSettingsV1,
        partition: CalibrationPartitionV1,
        source: ProfileSource,
        generated_monotonic_ns: u64,
        generated_unix_ns: u64,
        source_clock_max_error_ns: u64,
        frozen_fit_parameters_sha256: [u8; 32],
        fit: &[WholeWaveObservationV1],
        residual: &[WholeWaveObservationV1],
    ) -> Result<Self, CostProfileError> {
        CostProfileFileV7::from_capture_for_revision(
            fingerprint,
            settings,
            partition,
            source,
            generated_monotonic_ns,
            generated_unix_ns,
            source_clock_max_error_ns,
            frozen_fit_parameters_sha256,
            fit,
            residual,
            WholeWaveModelRevision::IndependentAttentionWorkSupportV1,
            COST_PROFILE_SCHEMA_VERSION_V8,
        )
        .map(Self)
    }
    pub fn to_bounded_bytes(&self, maximum: usize) -> Result<Vec<u8>, CostProfileError> {
        self.0.to_bounded_bytes(maximum)
    }
}
pub fn load_whole_wave_profile_v8(
    path: &Path,
    fingerprint: &ExecutionFingerprint,
    settings: &WholeWaveSettingsV1,
    limits: &CostProfileLoadLimits,
    clock: ProfileLoadClock,
) -> Result<ImportedWholeWaveModelV1, CostProfileError> {
    load_selected_profile_path(
        path,
        fingerprint,
        settings,
        limits,
        clock,
        load_whole_wave_profile_v8_bytes,
    )
}
pub fn load_whole_wave_profile_v8_bytes(
    bytes: &[u8],
    fingerprint: &ExecutionFingerprint,
    settings: &WholeWaveSettingsV1,
    limits: &CostProfileLoadLimits,
    clock: ProfileLoadClock,
) -> Result<ImportedWholeWaveModelV1, CostProfileError> {
    limits.validate()?;
    settings
        .validate()
        .map_err(|_| CostProfileError::Metadata("invalid whole-wave settings"))?;
    if bytes.len() > limits.max_file_bytes.get() {
        return Err(CostProfileError::Limit("file byte limit exceeded"));
    }
    #[derive(Deserialize)]
    struct Version {
        schema_version: u32,
    }
    let version: Version = serde_json::from_slice(bytes)?;
    if version.schema_version != COST_PROFILE_SCHEMA_VERSION_V8 {
        return Err(CostProfileError::UnsupportedVersion(version.schema_version));
    }
    let file: CostProfileFileV8 = serde_json::from_slice(bytes)?;
    import_selected_profile_revision(
        file.0,
        bytes,
        fingerprint,
        settings,
        limits,
        clock,
        WholeWaveModelRevision::IndependentAttentionWorkSupportV1,
    )
}
#[cfg(test)]
mod tests;
