//! Explicit prompt-range statistics over original host-settled observations.
//! The observation payload is unchanged; only this schema/model pair permits
//! prompt-total pooling. V4 is parsed independently and never relabeled.
use super::*;
use v2::{ParsedProfile, ParsedSample, ProfileModelSettingsV2};
pub use v4::{
    ProfileCostBoundaryV4 as ProfileCostBoundaryV5, ProfileSampleV4 as ProfileSampleV5,
    ProfileWaveShapeV4 as ProfileWaveShapeV5,
};

pub const COST_PROFILE_SCHEMA_VERSION_V5: u32 = 5;
#[cfg(test)]
mod tests;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CostProfileFileV5 {
    pub schema_version: u32,
    pub fingerprint: ProfileFingerprint,
    pub settings: ProfileModelSettingsV2,
    pub generated_unix_ns: u64,
    pub source_clock_max_error_ns: Option<u64>,
    pub source: ProfileSource,
    #[serde(deserialize_with = "bounded_samples")]
    pub samples: Vec<ProfileSampleV5>,
}
fn bounded_samples<'de, D: Deserializer<'de>>(d: D) -> Result<Vec<ProfileSampleV5>, D::Error> {
    bounded_vec::<D, ProfileSampleV5, HARD_SAMPLES>(d)
}
pub(super) fn parse(bytes: &[u8]) -> Result<ParsedProfile, CostProfileError> {
    let wire: CostProfileFileV5 = serde_json::from_slice(bytes)?;
    if !matches!(
        wire.settings.feature_model,
        CostFeatureModel::EmpiricalPromptRangeV3 { .. }
    ) {
        return Err(CostProfileError::Metadata(
            "profile v5 requires the explicit empirical prompt-range model",
        ));
    }
    let mut settings: CostModelSettings = wire.settings.exact.into();
    settings.feature_model = wire.settings.feature_model;
    let mut ordinals = HashSet::new();
    let mut samples = Vec::with_capacity(wire.samples.len());
    for sample in wire.samples {
        if sample.accepted_ordinal == 0 || !ordinals.insert(sample.accepted_ordinal) {
            return Err(CostProfileError::Metadata(
                "invalid or duplicate accepted prompt-range ordinal",
            ));
        }
        let mut shape: WaveExecutionShape = sample.shape.exact.exact.into();
        shape.numeric_features = sample.shape.exact.numeric_features;
        shape.host_content_features = sample.shape.host_content_features;
        shape.row_multiset_features = Some(sample.shape.row_multiset_features);
        samples.push(ParsedSample {
            source_record: sample.source_record,
            measured_unix_ns: sample.measured_unix_ns,
            shape,
            boundary: CostBoundary::PreparationToHostSettledV1,
            outcome: sample.outcome,
            timing: sample.timing,
        });
    }
    Ok(ParsedProfile {
        schema_version: wire.schema_version,
        fingerprint: wire.fingerprint,
        settings,
        generated_unix_ns: wire.generated_unix_ns,
        source_clock_max_error_ns: wire.source_clock_max_error_ns,
        source: wire.source,
        samples,
    })
}
