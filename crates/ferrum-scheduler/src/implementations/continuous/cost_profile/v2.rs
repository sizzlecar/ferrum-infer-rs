//! Strict v2 wire envelope. V1 types/keys remain unchanged and never fabricate
//! numeric evidence. Both versions train through the same age-preserving loader.
use super::*;
use ferrum_interfaces::execution_cost::CanonicalWaveCostFeatures;

pub const COST_PROFILE_SCHEMA_VERSION_V2: u32 = 2;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProfileModelSettingsV2 {
    pub exact: ProfileModelSettings,
    pub feature_model: CostFeatureModel,
}

impl From<&CostModelSettings> for ProfileModelSettingsV2 {
    fn from(settings: &CostModelSettings) -> Self {
        Self {
            exact: settings.into(),
            feature_model: settings.feature_model.clone(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProfileWaveShapeV2 {
    pub exact: ProfileWaveShape,
    /// None is explicitly exact-only. It cannot train BoundedNumericV1.
    pub numeric_features: Option<CanonicalWaveCostFeatures>,
}

impl From<&WaveExecutionShape> for ProfileWaveShapeV2 {
    fn from(shape: &WaveExecutionShape) -> Self {
        Self {
            exact: ProfileWaveShape {
                kind: shape.kind.into(),
                path: shape.path.into(),
                provider_signature: shape.provider_signature,
                output_policy_signature: shape.output_policy_signature,
                graph_state: shape.graph_state.into(),
                order: shape.order.into(),
                decode_kv_tokens: shape.decode_kv_tokens.clone(),
                prefill_chunks: shape
                    .prefill_chunks
                    .iter()
                    .map(|chunk| ProfilePrefillShape {
                        offset: chunk.offset,
                        count: chunk.count,
                        total_prompt_tokens: chunk.total_prompt_tokens,
                    })
                    .collect(),
                recurrent_state_bytes: shape.recurrent_state_bytes,
                restore_bytes: shape.restore_bytes,
                maintenance_bytes: shape.maintenance_bytes,
                maintenance_units: shape.maintenance_units,
            },
            numeric_features: shape.numeric_features.clone(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProfileSampleV2 {
    pub source_record: u64,
    pub measured_unix_ns: u64,
    pub shape: ProfileWaveShapeV2,
    pub boundary: ProfileCostBoundary,
    pub outcome: ProfileObservationOutcome,
    pub timing: ProfileWaveTiming,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CostProfileFileV2 {
    pub schema_version: u32,
    pub fingerprint: ProfileFingerprint,
    pub settings: ProfileModelSettingsV2,
    pub generated_unix_ns: u64,
    pub source_clock_max_error_ns: Option<u64>,
    pub source: ProfileSource,
    #[serde(deserialize_with = "bounded_samples_v2")]
    pub samples: Vec<ProfileSampleV2>,
}

fn bounded_samples_v2<'de, D: Deserializer<'de>>(d: D) -> Result<Vec<ProfileSampleV2>, D::Error> {
    bounded_vec::<D, ProfileSampleV2, HARD_SAMPLES>(d)
}

pub(super) struct ParsedProfile {
    pub schema_version: u32,
    pub fingerprint: ProfileFingerprint,
    pub settings: CostModelSettings,
    pub generated_unix_ns: u64,
    pub source_clock_max_error_ns: Option<u64>,
    pub source: ProfileSource,
    pub samples: Vec<ParsedSample>,
}

pub(super) struct ParsedSample {
    pub source_record: u64,
    pub measured_unix_ns: u64,
    pub shape: WaveExecutionShape,
    pub boundary: CostBoundary,
    pub outcome: ProfileObservationOutcome,
    pub timing: ProfileWaveTiming,
}

pub(super) fn parse(bytes: &[u8]) -> Result<ParsedProfile, CostProfileError> {
    // This bounded-file first pass only selects the schema; serde skips the
    // other values without constructing arbitrary JSON maps/arrays. The chosen
    // full parser below strictly rejects unknown and duplicate wire fields.
    #[derive(Deserialize)]
    struct Version {
        schema_version: u32,
    }
    let version: Version = serde_json::from_slice(bytes)?;
    match version.schema_version {
        COST_PROFILE_SCHEMA_VERSION => {
            let profile: CostProfileFile = serde_json::from_slice(bytes)?;
            Ok(ParsedProfile {
                schema_version: profile.schema_version,
                fingerprint: profile.fingerprint,
                settings: profile.settings.into(),
                generated_unix_ns: profile.generated_unix_ns,
                source_clock_max_error_ns: profile.source_clock_max_error_ns,
                source: profile.source,
                samples: profile
                    .samples
                    .into_iter()
                    .map(|sample| ParsedSample {
                        source_record: sample.source_record,
                        measured_unix_ns: sample.measured_unix_ns,
                        shape: sample.shape.into(),
                        boundary: sample.boundary.into(),
                        outcome: sample.outcome,
                        timing: sample.timing,
                    })
                    .collect(),
            })
        }
        COST_PROFILE_SCHEMA_VERSION_V2 => {
            let wire: CostProfileFileV2 = serde_json::from_slice(bytes)?;
            if matches!(
                wire.settings.feature_model,
                CostFeatureModel::EmpiricalHostContentV1 { .. }
                    | CostFeatureModel::EmpiricalRowMultisetV2 { .. }
            ) {
                return Err(CostProfileError::Metadata(
                    "host-content model requires profile v3",
                ));
            }
            let mut samples = Vec::with_capacity(wire.samples.len());
            for sample in wire.samples {
                let mut shape: WaveExecutionShape = sample.shape.exact.into();
                shape.numeric_features = sample.shape.numeric_features;
                samples.push(ParsedSample {
                    source_record: sample.source_record,
                    measured_unix_ns: sample.measured_unix_ns,
                    shape,
                    boundary: sample.boundary.into(),
                    outcome: sample.outcome,
                    timing: sample.timing,
                });
            }
            let mut settings: CostModelSettings = wire.settings.exact.into();
            settings.feature_model = wire.settings.feature_model;
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
        v3::COST_PROFILE_SCHEMA_VERSION_V3 => v3::parse(bytes),
        v4::COST_PROFILE_SCHEMA_VERSION_V4 => v4::parse(bytes),
        other => Err(CostProfileError::UnsupportedVersion(other)),
    }
}

#[cfg(test)]
mod tests;
