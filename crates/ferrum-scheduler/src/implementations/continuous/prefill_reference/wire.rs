use super::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReferenceEstimator {
    UpperMedianWallV1,
}

/// Historical routes allowed in the frozen reference. This does not predict
/// graph residency or grant a future execution permit. Every trial must still
/// match its complete declared shape and original committed source record.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReferenceGraphRoutes {
    #[default]
    DisabledOnly,
    ExactObserved,
}
impl ReferenceGraphRoutes {
    pub fn is_disabled_only(&self) -> bool {
        *self == Self::DisabledOnly
    }
    pub(super) fn accepts(self, graph: ProfileGraphState) -> bool {
        match (self, graph) {
            (Self::DisabledOnly, ProfileGraphState::Disabled)
            | (
                Self::ExactObserved,
                ProfileGraphState::Disabled
                | ProfileGraphState::Cold
                | ProfileGraphState::Warm
                | ProfileGraphState::ConfiguredEager,
            ) => true,
            (
                Self::DisabledOnly,
                ProfileGraphState::Cold
                | ProfileGraphState::Warm
                | ProfileGraphState::ConfiguredEager,
            ) => false,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReferenceProtocolV1 {
    #[serde(
        default,
        skip_serializing_if = "ReferenceGraphRoutes::is_disabled_only"
    )]
    pub graph_routes: ReferenceGraphRoutes,
    pub granule_tokens: NonZeroU32,
    /// Predeclared repetitions, not a statistical quality guarantee.
    pub repetitions: NonZeroUsize,
    pub estimator: ReferenceEstimator,
    pub input_preprocessing_sha256: [u8; 32],
    /// Warm/cold setup, capacity, codec, sampler and measurement conditions.
    pub measurement_conditions_sha256: [u8; 32],
    pub prefill_host: HostCostFeaturesV1,
    pub decode_host: HostCostFeaturesV1,
    /// Exact singleton shape selected before sampling; mixed wall cannot be
    /// divided by row/token count to manufacture this unit.
    pub decode_shape: ProfileWaveShapeV2,
}
impl ReferenceProtocolV1 {
    pub fn sha256(&self) -> Result<[u8; 32], ReferenceError> {
        let bytes = serde_json::to_vec(self)?;
        let mut hash = Sha256::new();
        hash.update(b"ferrum.prefill-reference-protocol.v1\0");
        hash.update(bytes);
        Ok(hash.finalize().into())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReferenceRecordId {
    pub source_sha256: [u8; 32],
    pub ordinal: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReferenceStateOrigin {
    Fresh,
    CommittedContinuation,
    PreparedDecode,
    Restored,
    Recomputed,
}

/// Actual host commit evidence accompanying one profile record. A calibration
/// collector supplies this from the executor/engine receipt, never CLI intent.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReferenceCommitReceipt {
    pub owner_incarnation: NonZeroU64,
    pub work_generation: NonZeroU64,
    pub origin: ReferenceStateOrigin,
    pub previous_record: Option<ReferenceRecordId>,
    pub prefix_before: u32,
    pub prefix_after: u32,
    pub generated_before: u32,
    pub generated_after: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReferenceObservedSample {
    pub record: ReferenceRecordId,
    pub observation: ProfileSampleV2,
    pub commit: ReferenceCommitReceipt,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReferencePrefillTrial {
    pub trial_index: usize,
    pub input_tokens_sha256: [u8; 32],
    #[serde(deserialize_with = "segments")]
    pub samples: Vec<ReferenceObservedSample>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReferenceCurveInput {
    pub total_prompt_tokens: NonZeroU32,
    /// Actual reference decomposition. These shapes are declared before trials
    /// and must match every sample, including numeric host/readback categories.
    #[serde(deserialize_with = "segments")]
    pub partition: Vec<ProfileWaveShapeV2>,
    #[serde(deserialize_with = "repetitions")]
    pub trials: Vec<ReferencePrefillTrial>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReferenceCalibrationV1 {
    pub schema_version: u32,
    pub reference_revision: NonZeroU64,
    pub fingerprint: ProfileFingerprint,
    pub generated_unix_ns: NonZeroU64,
    pub protocol: ReferenceProtocolV1,
    #[serde(deserialize_with = "repetitions")]
    pub decode_samples: Vec<ReferenceObservedSample>,
    #[serde(deserialize_with = "curves")]
    pub curves: Vec<ReferenceCurveInput>,
}

fn segments<'de, D: Deserializer<'de>, T: Deserialize<'de>>(d: D) -> Result<Vec<T>, D::Error> {
    bounded_vec::<D, T, PREFILL_REFERENCE_MAX_POINTS_PER_CURVE>(d)
}
pub(super) fn repetitions<'de, D: Deserializer<'de>, T: Deserialize<'de>>(
    d: D,
) -> Result<Vec<T>, D::Error> {
    bounded_vec::<D, T, PREFILL_REFERENCE_MAX_REPETITIONS>(d)
}
pub(super) fn curves<'de, D: Deserializer<'de>>(
    d: D,
) -> Result<Vec<ReferenceCurveInput>, D::Error> {
    bounded_vec::<D, ReferenceCurveInput, PREFILL_REFERENCE_MAX_CURVES>(d)
}
