//! Explicit row-multiset empirical observations. No legacy file is relabeled
//! and no new row class is inferred from an irreversible V1 whole-wave hash.
use super::*;
use ferrum_interfaces::execution_cost::{HostContentCostFeaturesV1, HostRowMultisetCostFeaturesV2};
use v2::{ParsedProfile, ParsedSample, ProfileModelSettingsV2, ProfileWaveShapeV2};
pub use v3::ProfileCostBoundaryV3 as ProfileCostBoundaryV4;

pub const COST_PROFILE_SCHEMA_VERSION_V4: u32 = 4;
#[cfg(test)]
mod tests;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProfileWaveShapeV4 {
    pub exact: ProfileWaveShapeV2,
    /// Retained actual legacy identity, never used to invent V2 row classes.
    pub host_content_features: Option<HostContentCostFeaturesV1>,
    pub row_multiset_features: HostRowMultisetCostFeaturesV2,
}
impl TryFrom<&WaveExecutionShape> for ProfileWaveShapeV4 {
    type Error = CostProfileError;
    fn try_from(shape: &WaveExecutionShape) -> Result<Self, Self::Error> {
        let rows = shape.decode_kv_tokens.len() + shape.prefill_chunks.len();
        let features = shape
            .row_multiset_features
            .as_ref()
            .ok_or(CostProfileError::Metadata(
                "missing actual row-multiset features",
            ))?;
        features
            .validate(rows)
            .map_err(|_| CostProfileError::Metadata("invalid row-multiset features"))?;
        shape
            .numeric_features
            .as_ref()
            .ok_or(CostProfileError::Metadata("missing row numeric evidence"))?
            .validate(rows)
            .map_err(|_| CostProfileError::Metadata("invalid row numeric evidence"))?;
        Ok(Self {
            exact: shape.into(),
            host_content_features: shape.host_content_features,
            row_multiset_features: features.clone(),
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProfileSampleV4 {
    pub source_record: u64,
    pub accepted_ordinal: u64,
    pub measured_unix_ns: u64,
    pub shape: ProfileWaveShapeV4,
    pub boundary: ProfileCostBoundaryV4,
    pub outcome: ProfileObservationOutcome,
    pub timing: ProfileWaveTiming,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CostProfileFileV4 {
    pub schema_version: u32,
    pub fingerprint: ProfileFingerprint,
    pub settings: ProfileModelSettingsV2,
    pub generated_unix_ns: u64,
    pub source_clock_max_error_ns: Option<u64>,
    pub source: ProfileSource,
    #[serde(deserialize_with = "bounded_samples")]
    pub samples: Vec<ProfileSampleV4>,
}
fn bounded_samples<'de, D: Deserializer<'de>>(d: D) -> Result<Vec<ProfileSampleV4>, D::Error> {
    bounded_vec::<D, ProfileSampleV4, HARD_SAMPLES>(d)
}
pub(super) fn parse(bytes: &[u8]) -> Result<ParsedProfile, CostProfileError> {
    let wire: CostProfileFileV4 = serde_json::from_slice(bytes)?;
    if !matches!(
        wire.settings.feature_model,
        CostFeatureModel::EmpiricalRowMultisetV2 { .. }
    ) {
        return Err(CostProfileError::Metadata(
            "profile v4 requires the explicit row-multiset model",
        ));
    }
    let mut settings: CostModelSettings = wire.settings.exact.into();
    settings.feature_model = wire.settings.feature_model;
    let mut ordinals = HashSet::new();
    let mut samples = Vec::with_capacity(wire.samples.len());
    for sample in wire.samples {
        if sample.accepted_ordinal == 0 || !ordinals.insert(sample.accepted_ordinal) {
            return Err(CostProfileError::Metadata(
                "invalid or duplicate accepted row-multiset ordinal",
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
