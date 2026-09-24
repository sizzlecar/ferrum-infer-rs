//! Explicit profile7 for independent-attention empirical families. The fitting,
//! residual, source/clock, capacity and exact-binding rules are shared with V1;
//! old profile6 records cannot provide or synthesize the required V2 sidecar.
use super::super::cost_model::statistical::{
    model::{
        CalibrationPartitionV1, FittedWholeWaveModelV1, WholeWaveObservationV1,
        WholeWaveSettingsV1, INDEPENDENT_ATTENTION_MODEL_REVISION,
    },
    SelectedStatisticalFamily,
};
use super::statistical_v6::loader::{
    import_selected_profile, load_selected_profile_path, SelectedProfileRecord,
};
use super::statistical_v6::{
    ImportedWholeWaveModelV1, WholeWaveProfileFile, WholeWaveProfilePhaseV6,
    WholeWaveProfileSampleV6, WholeWaveProfileShapeV6,
};
use super::*;
use ferrum_interfaces::execution_cost::{
    IndependentAttentionWaveEvidenceWireV2, StatisticalWaveEvidenceWireV1,
};

pub const COST_PROFILE_SCHEMA_VERSION_V7: u32 = 7;
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WholeWaveProfileSampleV7 {
    pub phase: WholeWaveProfilePhaseV6,
    pub accepted_ordinal: u64,
    pub call_id: u64,
    pub measured_unix_ns: u64,
    pub shape: WholeWaveProfileShapeV6,
    /// Original ordered evidence is retained, never reconstructed from V2.
    pub selected: StatisticalWaveEvidenceWireV1,
    pub independent_attention: IndependentAttentionWaveEvidenceWireV2,
    pub boundary: v3::ProfileCostBoundaryV3,
    pub outcome: ProfileObservationOutcome,
    pub wall_ns: u64,
}
pub type CostProfileFileV7 = WholeWaveProfileFile<WholeWaveProfileSampleV7>;
impl SelectedProfileRecord for WholeWaveProfileSampleV7 {
    fn into_parts(
        self,
    ) -> (
        WholeWaveProfileSampleV6,
        Option<IndependentAttentionWaveEvidenceWireV2>,
    ) {
        (
            WholeWaveProfileSampleV6 {
                phase: self.phase,
                accepted_ordinal: self.accepted_ordinal,
                call_id: self.call_id,
                measured_unix_ns: self.measured_unix_ns,
                shape: self.shape,
                selected: self.selected,
                boundary: self.boundary,
                outcome: self.outcome,
                wall_ns: self.wall_ns,
            },
            Some(self.independent_attention),
        )
    }
}
impl CostProfileFileV7 {
    /// Same real capture identity and irreversible fit freeze as profile6, with
    /// the new family revision explicitly bound into the fitted parameter hash.
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
        if source.observation_artifact_sha256 == [0; 32] || frozen_fit_parameters_sha256 == [0; 32]
        {
            return Err(CostProfileError::Metadata(
                "unsealed source or missing frozen fit",
            ));
        }
        let frozen = FittedWholeWaveModelV1::fit_independent_attention_v2(
            fingerprint.clone(),
            settings.clone(),
            partition,
            fit,
            generated_monotonic_ns,
        )
        .map_err(|_| CostProfileError::Metadata("invalid independent-attention fit"))?;
        if frozen.parameter_signature() != frozen_fit_parameters_sha256 {
            return Err(CostProfileError::Metadata(
                "fit changed after its pre-residual freeze",
            ));
        }
        frozen
            .calibrate(residual, generated_monotonic_ns)
            .map_err(|_| CostProfileError::Metadata("invalid independent-attention residual"))?;
        let mut samples = Vec::with_capacity(fit.len() + residual.len());
        for (phase, rows) in [
            (WholeWaveProfilePhaseV6::Fit, fit),
            (WholeWaveProfilePhaseV6::Residual, residual),
        ] {
            for row in rows {
                let age = generated_monotonic_ns
                    .checked_sub(row.observed_at_ns)
                    .ok_or(CostProfileError::Clock("future monotonic observation"))?;
                let measured_unix_ns = generated_unix_ns
                    .checked_sub(age)
                    .filter(|x| *x > 0)
                    .ok_or(CostProfileError::Clock("wall timestamp underflow"))?;
                let selected =
                    row.selected
                        .independent_attention_v2()
                        .ok_or(CostProfileError::Metadata(
                            "missing independent-attention producer",
                        ))?;
                selected.validate_exact(&row.exact).map_err(|_| {
                    CostProfileError::Metadata("unbound independent-attention producer")
                })?;
                samples.push(WholeWaveProfileSampleV7 {
                    phase,
                    accepted_ordinal: row.accepted_ordinal,
                    call_id: row.call_id,
                    measured_unix_ns,
                    shape: (&row.exact).try_into()?,
                    selected: row.selected.to_wire_v1(),
                    independent_attention: selected.to_wire_v2(),
                    boundary: v3::ProfileCostBoundaryV3::PreparationToHostSettledV1,
                    outcome: ProfileObservationOutcome::Completed {},
                    wall_ns: row.wall_ns,
                });
            }
        }
        Ok(Self {
            capture_identity_sha256: partition.source_sha256,
            fit_parameters_sha256: frozen_fit_parameters_sha256,
            schema_version: COST_PROFILE_SCHEMA_VERSION_V7,
            model_revision: INDEPENDENT_ATTENTION_MODEL_REVISION.into(),
            fingerprint: fingerprint.into(),
            settings: settings.into(),
            generated_unix_ns,
            source_clock_max_error_ns: Some(source_clock_max_error_ns),
            source,
            protocol_sha256: partition.protocol_sha256,
            fit_through_ordinal: partition.fit_through_ordinal,
            residual_through_ordinal: partition.residual_through_ordinal,
            samples,
        })
    }
}
pub fn load_whole_wave_profile_v7(
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
        load_whole_wave_profile_v7_bytes,
    )
}
pub fn load_whole_wave_profile_v7_bytes(
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
    if version.schema_version != COST_PROFILE_SCHEMA_VERSION_V7 {
        return Err(CostProfileError::UnsupportedVersion(version.schema_version));
    }
    let file: CostProfileFileV7 = serde_json::from_slice(bytes)?;
    import_selected_profile(
        file,
        bytes,
        fingerprint,
        settings,
        limits,
        clock,
        SelectedStatisticalFamily::IndependentAttentionV2,
    )
}
#[cfg(test)]
mod tests;
