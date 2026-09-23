//! Host-content observations have a distinct wire and timing boundary. Legacy
//! profile samples never acquire a content capability or a later wall endpoint.
use super::*;
use ferrum_interfaces::execution_cost::HostContentCostFeaturesV1;
use v2::{ParsedProfile, ParsedSample, ProfileModelSettingsV2, ProfileWaveShapeV2};

pub const COST_PROFILE_SCHEMA_VERSION_V3: u32 = 3;
#[cfg(test)]
mod tests;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProfileCostBoundaryV3 {
    PreparationToHostSettledV1,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProfileWaveShapeV3 {
    pub exact: ProfileWaveShapeV2,
    pub host_content_features: HostContentCostFeaturesV1,
}
impl TryFrom<&WaveExecutionShape> for ProfileWaveShapeV3 {
    type Error = CostProfileError;
    fn try_from(shape: &WaveExecutionShape) -> Result<Self, Self::Error> {
        let features = shape
            .host_content_features
            .as_ref()
            .filter(|features| features.schema_version == 1)
            .ok_or(CostProfileError::Metadata(
                "missing host-content domain evidence",
            ))?;
        let numeric = shape
            .numeric_features
            .as_ref()
            .ok_or(CostProfileError::Metadata(
                "missing host-content numeric evidence",
            ))?;
        numeric
            .validate(shape.decode_kv_tokens.len() + shape.prefill_chunks.len())
            .map_err(|_| CostProfileError::Metadata("invalid host-content numeric evidence"))?;
        Ok(Self {
            exact: shape.into(),
            host_content_features: features.clone(),
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProfileSampleV3 {
    /// The source record is a host-stage record, not a legacy cost observation.
    pub source_record: u64,
    pub accepted_ordinal: u64,
    pub measured_unix_ns: u64,
    pub shape: ProfileWaveShapeV3,
    pub boundary: ProfileCostBoundaryV3,
    pub outcome: ProfileObservationOutcome,
    pub timing: ProfileWaveTiming,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CostProfileFileV3 {
    pub schema_version: u32,
    pub fingerprint: ProfileFingerprint,
    pub settings: ProfileModelSettingsV2,
    pub generated_unix_ns: u64,
    pub source_clock_max_error_ns: Option<u64>,
    pub source: ProfileSource,
    #[serde(deserialize_with = "bounded_samples")]
    pub samples: Vec<ProfileSampleV3>,
}
fn bounded_samples<'de, D: Deserializer<'de>>(d: D) -> Result<Vec<ProfileSampleV3>, D::Error> {
    bounded_vec::<D, ProfileSampleV3, HARD_SAMPLES>(d)
}

pub(super) fn parse(bytes: &[u8]) -> Result<ParsedProfile, CostProfileError> {
    let wire: CostProfileFileV3 = serde_json::from_slice(bytes)?;
    if !matches!(
        wire.settings.feature_model,
        CostFeatureModel::EmpiricalHostContentV1 { .. }
    ) {
        return Err(CostProfileError::Metadata(
            "profile v3 requires the host-content model",
        ));
    }
    let mut settings: CostModelSettings = wire.settings.exact.into();
    settings.feature_model = wire.settings.feature_model;
    let mut samples = Vec::with_capacity(wire.samples.len());
    let mut ordinals = HashSet::new();
    for sample in wire.samples {
        if sample.accepted_ordinal == 0 || !ordinals.insert(sample.accepted_ordinal) {
            return Err(CostProfileError::Metadata(
                "invalid or duplicate accepted host-stage ordinal",
            ));
        }
        if sample.shape.host_content_features.schema_version != 1 {
            return Err(CostProfileError::Metadata(
                "unsupported host-content domain version",
            ));
        }
        let mut shape: WaveExecutionShape = sample.shape.exact.exact.into();
        shape.numeric_features = sample.shape.exact.numeric_features;
        shape.host_content_features = Some(sample.shape.host_content_features);
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
