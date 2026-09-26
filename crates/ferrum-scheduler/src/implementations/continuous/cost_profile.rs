//! Bounded, versioned offline calibration import. Loading is a slow path.
//!
//! Files contain observations, never serialized predictor internals. Every
//! eligible observation passes through `CostModelTrainer`. Wall-clock evidence
//! establishes conservative age at import; one explicit biased monotonic clock
//! then ages imported and live observations alike, including at process start.

use super::cost_model::*;
use serde::{de::SeqAccess, Deserialize, Deserializer, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, HashSet},
    fmt,
    fs::File,
    io::Read,
    marker::PhantomData,
    num::{NonZeroU32, NonZeroU64, NonZeroUsize},
    path::{Path, PathBuf},
    sync::Arc,
};

pub const COST_PROFILE_SCHEMA_VERSION: u32 = 1;
pub mod statistical_v6;
pub mod statistical_v7;
pub mod statistical_v8;
pub mod structured_v9;
pub mod v2;
pub mod v3;
pub mod v4;
pub mod v5;
const HARD_FILE_BYTES: usize = ferrum_types::SloCostProfileImportConfig::MAX_FILE_BYTES;
const HARD_SAMPLES: usize = 131_072;
const HARD_ROWS_PER_VECTOR: usize = 1024;
const HARD_TOTAL_ROWS: usize = 1_048_576;

macro_rules! profile_enum {
    ($wire:ident => $native:ident { $($variant:ident),+ $(,)? }) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
        #[serde(rename_all = "snake_case")]
        pub enum $wire { $($variant),+ }
        impl From<$wire> for $native {
            fn from(value: $wire) -> Self { match value { $($wire::$variant => Self::$variant),+ } }
        }
        impl From<$native> for $wire {
            fn from(value: $native) -> Self { match value { $($native::$variant => Self::$variant),+ } }
        }
    };
}

profile_enum!(ProfileWaveKind => WaveKind { Decode, Prefill, Mixed, Restore, Maintenance });
profile_enum!(ProfileExecutionPath => WaveExecutionPath {
    PlanRuntime, NativeUnified, LegacySplit, UnsupportedFallback, CapacityFallback
});
profile_enum!(ProfileGraphState => WaveGraphState { Disabled, Cold, Warm, ConfiguredEager });
profile_enum!(ProfileBatchOrder => BatchOrderSemantics { Ordered, IndependentRows });
/// V1/V2 wire boundaries remain closed; host-settled evidence requires V3.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProfileCostBoundary {
    PreparationToCommit,
    DeviceOnly,
}

impl From<ProfileCostBoundary> for CostBoundary {
    fn from(value: ProfileCostBoundary) -> Self {
        match value {
            ProfileCostBoundary::PreparationToCommit => Self::PreparationToCommit,
            ProfileCostBoundary::DeviceOnly => Self::DeviceOnly,
        }
    }
}
impl TryFrom<CostBoundary> for ProfileCostBoundary {
    type Error = CostProfileError;
    fn try_from(value: CostBoundary) -> Result<Self, Self::Error> {
        match value {
            CostBoundary::PreparationToCommit => Ok(Self::PreparationToCommit),
            CostBoundary::DeviceOnly => Ok(Self::DeviceOnly),
            CostBoundary::PreparationToHostSettledV1 => Err(CostProfileError::Metadata(
                "host-settled boundary requires profile v3",
            )),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProfileFingerprint {
    pub model_weights: [u8; 32],
    pub numerical_policy: [u8; 32],
    pub device_runtime: [u8; 32],
    pub execution_config: [u8; 32],
}

impl From<&ExecutionFingerprint> for ProfileFingerprint {
    fn from(value: &ExecutionFingerprint) -> Self {
        Self {
            model_weights: value.model_weights,
            numerical_policy: value.numerical_policy,
            device_runtime: value.device_runtime,
            execution_config: value.execution_config,
        }
    }
}

impl From<ProfileFingerprint> for ExecutionFingerprint {
    fn from(value: ProfileFingerprint) -> Self {
        Self {
            model_weights: value.model_weights,
            numerical_policy: value.numerical_policy,
            device_runtime: value.device_runtime,
            execution_config: value.execution_config,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProfileShapeLimits {
    pub max_rows: NonZeroUsize,
    pub max_context_tokens: NonZeroU32,
    pub max_prefill_tokens_per_wave: NonZeroU64,
    pub max_state_bytes: NonZeroU64,
    pub max_maintenance_units: NonZeroU32,
}

/// Every setting is explicit; importing a file cannot silently weaken the
/// caller's minimum coverage, age, memory, timing or numerical requirements.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProfileModelSettings {
    pub max_buckets: NonZeroUsize,
    pub max_samples_per_bucket: NonZeroUsize,
    pub min_samples: NonZeroUsize,
    pub max_retained_samples: NonZeroUsize,
    pub max_retained_shape_rows: NonZeroUsize,
    pub residual_quantile: f64,
    pub drift_margin_ns: u64,
    pub max_wave_ns: NonZeroU64,
    pub max_sample_age_ns: NonZeroU64,
    pub context_bucket_tokens: NonZeroU32,
    pub prefill_offset_bucket_tokens: NonZeroU32,
    pub shape_limits: ProfileShapeLimits,
}

impl From<&CostModelSettings> for ProfileModelSettings {
    fn from(value: &CostModelSettings) -> Self {
        Self {
            max_buckets: value.max_buckets,
            max_samples_per_bucket: value.max_samples_per_bucket,
            min_samples: value.min_samples,
            max_retained_samples: value.max_retained_samples,
            max_retained_shape_rows: value.max_retained_shape_rows,
            residual_quantile: value.residual_quantile,
            drift_margin_ns: value.drift_margin_ns,
            max_wave_ns: value.max_wave_ns,
            max_sample_age_ns: value.max_sample_age_ns,
            context_bucket_tokens: value.context_bucket_tokens,
            prefill_offset_bucket_tokens: value.prefill_offset_bucket_tokens,
            shape_limits: ProfileShapeLimits {
                max_rows: value.shape_limits.max_rows,
                max_context_tokens: value.shape_limits.max_context_tokens,
                max_prefill_tokens_per_wave: value.shape_limits.max_prefill_tokens_per_wave,
                max_state_bytes: value.shape_limits.max_state_bytes,
                max_maintenance_units: value.shape_limits.max_maintenance_units,
            },
        }
    }
}

impl From<ProfileModelSettings> for CostModelSettings {
    fn from(value: ProfileModelSettings) -> Self {
        Self {
            feature_model: CostFeatureModel::ExactV1 {},
            max_buckets: value.max_buckets,
            max_samples_per_bucket: value.max_samples_per_bucket,
            min_samples: value.min_samples,
            max_retained_samples: value.max_retained_samples,
            max_retained_shape_rows: value.max_retained_shape_rows,
            residual_quantile: value.residual_quantile,
            drift_margin_ns: value.drift_margin_ns,
            max_wave_ns: value.max_wave_ns,
            max_sample_age_ns: value.max_sample_age_ns,
            context_bucket_tokens: value.context_bucket_tokens,
            prefill_offset_bucket_tokens: value.prefill_offset_bucket_tokens,
            shape_limits: CostShapeLimits {
                max_rows: value.shape_limits.max_rows,
                max_context_tokens: value.shape_limits.max_context_tokens,
                max_prefill_tokens_per_wave: value.shape_limits.max_prefill_tokens_per_wave,
                max_state_bytes: value.shape_limits.max_state_bytes,
                max_maintenance_units: value.shape_limits.max_maintenance_units,
            },
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProfilePrefillShape {
    pub offset: u32,
    pub count: NonZeroU32,
    pub total_prompt_tokens: NonZeroU32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProfileWaveShape {
    pub kind: ProfileWaveKind,
    pub path: ProfileExecutionPath,
    pub provider_signature: [u8; 32],
    pub output_policy_signature: [u8; 32],
    pub graph_state: ProfileGraphState,
    pub order: ProfileBatchOrder,
    #[serde(deserialize_with = "bounded_rows")]
    pub decode_kv_tokens: Vec<u32>,
    #[serde(deserialize_with = "bounded_rows")]
    pub prefill_chunks: Vec<ProfilePrefillShape>,
    pub recurrent_state_bytes: u64,
    pub restore_bytes: u64,
    pub maintenance_bytes: u64,
    pub maintenance_units: u32,
}

impl From<ProfileWaveShape> for WaveExecutionShape {
    fn from(value: ProfileWaveShape) -> Self {
        Self {
            row_multiset_features: None,
            host_content_features: None,
            numeric_features: None,
            kind: value.kind.into(),
            path: value.path.into(),
            provider_signature: value.provider_signature,
            output_policy_signature: value.output_policy_signature,
            graph_state: value.graph_state.into(),
            order: value.order.into(),
            decode_kv_tokens: value.decode_kv_tokens,
            prefill_chunks: value
                .prefill_chunks
                .into_iter()
                .map(|chunk| PrefillShape {
                    offset: chunk.offset,
                    count: chunk.count,
                    total_prompt_tokens: chunk.total_prompt_tokens,
                })
                .collect(),
            recurrent_state_bytes: value.recurrent_state_bytes,
            restore_bytes: value.restore_bytes,
            maintenance_bytes: value.maintenance_bytes,
            maintenance_units: value.maintenance_units,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum ProfileObservationOutcome {
    // Empty struct variants deliberately enforce deny_unknown_fields. Serde's
    // internally tagged unit variants otherwise discard trailing metadata.
    Completed {},
    PartiallyCompleted {
        timing_covers_actual_shape_only: bool,
    },
    NotSubmitted {},
    Deferred {},
    FailedAfterSubmit {},
}

impl From<ProfileObservationOutcome> for WaveObservationOutcome {
    fn from(value: ProfileObservationOutcome) -> Self {
        match value {
            ProfileObservationOutcome::Completed {} => Self::Completed,
            ProfileObservationOutcome::PartiallyCompleted {
                timing_covers_actual_shape_only,
            } => Self::PartiallyCompleted {
                timing_covers_actual_shape_only,
            },
            ProfileObservationOutcome::NotSubmitted {} => Self::NotSubmitted,
            ProfileObservationOutcome::Deferred {} => Self::Deferred,
            ProfileObservationOutcome::FailedAfterSubmit {} => Self::FailedAfterSubmit,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProfileMeasuredSpan {
    pub start_ns: u64,
    pub end_ns: u64,
}

impl From<ProfileMeasuredSpan> for MeasuredSpan {
    fn from(value: ProfileMeasuredSpan) -> Self {
        Self {
            start_ns: value.start_ns,
            end_ns: value.end_ns,
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProfileStageTimings {
    pub prepare: Option<ProfileMeasuredSpan>,
    pub device_wait: Option<ProfileMeasuredSpan>,
    pub commit: Option<ProfileMeasuredSpan>,
    pub restore: Option<ProfileMeasuredSpan>,
    pub maintenance: Option<ProfileMeasuredSpan>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProfileWaveTiming {
    pub wall_total_ns: u64,
    pub device_elapsed_ns: Option<u64>,
    pub stages: ProfileStageTimings,
}

impl From<ProfileWaveTiming> for WaveTiming {
    fn from(value: ProfileWaveTiming) -> Self {
        Self {
            wall_total_ns: value.wall_total_ns,
            device_elapsed_ns: value.device_elapsed_ns,
            stages: WaveStageTimings {
                prepare: value.stages.prepare.map(Into::into),
                device_wait: value.stages.device_wait.map(Into::into),
                commit: value.stages.commit.map(Into::into),
                restore: value.stages.restore.map(Into::into),
                maintenance: value.stages.maintenance.map(Into::into),
            },
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProfileSource {
    pub generator: String,
    pub generator_revision: String,
    pub measurement_protocol: String,
    /// Digest of the source observation artifact, not of this profile file.
    pub observation_artifact_sha256: [u8; 32],
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProfileSample {
    /// Unique record ordinal in the source artifact; file order need not be chronological.
    pub source_record: u64,
    pub measured_unix_ns: u64,
    pub shape: ProfileWaveShape,
    pub boundary: ProfileCostBoundary,
    pub outcome: ProfileObservationOutcome,
    pub timing: ProfileWaveTiming,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CostProfileFile {
    pub schema_version: u32,
    pub fingerprint: ProfileFingerprint,
    pub settings: ProfileModelSettings,
    pub generated_unix_ns: u64,
    /// None means the source wall clock's accuracy is unknown: import rejects it.
    pub source_clock_max_error_ns: Option<u64>,
    pub source: ProfileSource,
    #[serde(deserialize_with = "bounded_samples")]
    pub samples: Vec<ProfileSample>,
}

fn bounded_samples<'de, D: Deserializer<'de>>(d: D) -> Result<Vec<ProfileSample>, D::Error> {
    bounded_vec::<D, ProfileSample, HARD_SAMPLES>(d)
}

fn bounded_rows<'de, D: Deserializer<'de>, T: Deserialize<'de>>(d: D) -> Result<Vec<T>, D::Error> {
    bounded_vec::<D, T, HARD_ROWS_PER_VECTOR>(d)
}

pub(super) fn bounded_vec<'de, D: Deserializer<'de>, T: Deserialize<'de>, const LIMIT: usize>(
    d: D,
) -> Result<Vec<T>, D::Error> {
    struct Visitor<T, const LIMIT: usize>(PhantomData<T>);
    impl<'de, T: Deserialize<'de>, const LIMIT: usize> serde::de::Visitor<'de> for Visitor<T, LIMIT> {
        type Value = Vec<T>;
        fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "an array of at most {LIMIT} entries")
        }
        fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Self::Value, A::Error> {
            let mut values = Vec::new();
            while let Some(value) = seq.next_element()? {
                if values.len() == LIMIT {
                    return Err(serde::de::Error::custom("profile array exceeds hard limit"));
                }
                values.push(value);
            }
            Ok(values)
        }
    }
    d.deserialize_seq(Visitor::<T, LIMIT>(PhantomData))
}

#[derive(Debug, Clone)]
pub struct CostProfileLoadLimits {
    pub max_file_bytes: NonZeroUsize,
    pub max_samples: NonZeroUsize,
    pub max_total_shape_rows: NonZeroUsize,
    pub max_source_field_bytes: NonZeroUsize,
    pub max_profile_age_ns: NonZeroU64,
    pub max_clock_error_ns: u64,
}

impl Default for CostProfileLoadLimits {
    fn default() -> Self {
        Self {
            max_file_bytes: NonZeroUsize::new(16 * 1024 * 1024).unwrap(),
            max_samples: NonZeroUsize::new(16_384).unwrap(),
            max_total_shape_rows: NonZeroUsize::new(262_144).unwrap(),
            max_source_field_bytes: NonZeroUsize::new(4096).unwrap(),
            max_profile_age_ns: NonZeroU64::new(86_400_000_000_000).unwrap(),
            max_clock_error_ns: 1_000_000_000,
        }
    }
}

impl CostProfileLoadLimits {
    fn validate(&self) -> Result<(), CostProfileError> {
        if self.max_file_bytes.get() > HARD_FILE_BYTES
            || self.max_samples.get() > HARD_SAMPLES
            || self.max_total_shape_rows.get() > HARD_TOTAL_ROWS
            || self.max_source_field_bytes.get() > 16_384
        {
            return Err(CostProfileError::Limit("loader hard limit exceeded"));
        }
        Ok(())
    }
}

/// Caller supplies a paired wall/monotonic reading on the eventual live model
/// clock. None explicitly represents an unavailable or untrusted wall clock.
#[derive(Debug, Clone, Copy)]
pub struct ProfileLoadClock {
    pub wall_unix_ns: Option<u64>,
    pub wall_max_error_ns: Option<u64>,
    pub monotonic_now_ns: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct ProfileObservationClock {
    pub source_monotonic_anchor_ns: u64,
    pub model_anchor_ns: u64,
}

impl ProfileObservationClock {
    /// Use this conversion for EVERY subsequent live observation/publication.
    pub fn model_now_ns(&self, local_now_ns: u64) -> Result<u64, CostProfileError> {
        let elapsed = local_now_ns
            .checked_sub(self.source_monotonic_anchor_ns)
            .ok_or(CostProfileError::Clock(
                "local monotonic clock moved backwards",
            ))?;
        self.model_anchor_ns
            .checked_add(elapsed)
            .ok_or(CostProfileError::Clock("model clock overflow"))
    }
}

#[derive(Debug, Clone)]
pub struct ImportedCostSnapshot {
    snapshot: Arc<CostModelSnapshot>,
    pub clock: ProfileObservationClock,
}

impl ImportedCostSnapshot {
    pub fn planning_boundary(&self) -> CostBoundary {
        self.snapshot.planning_boundary()
    }
    pub fn model_version(&self) -> u64 {
        self.snapshot.model_version()
    }
    pub fn bucket_count(&self) -> usize {
        self.snapshot.bucket_count()
    }
    /// Fast path: immutable data, integer clock conversion, and model lookup.
    pub fn predict(
        &self,
        fingerprint: &ExecutionFingerprint,
        shape: &WaveExecutionShape,
        boundary: CostBoundary,
        local_now_ns: u64,
    ) -> CostPrediction {
        match self.clock.model_now_ns(local_now_ns) {
            Ok(now) => self.snapshot.predict(fingerprint, shape, boundary, now),
            Err(_) => CostPrediction::Unknown(CostUnknownReason::ClockMovedBackwards),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ProfileSkippedObservation {
    NotSubmitted,
    Deferred,
    FailedAfterSubmit,
    UnisolatedPartial,
    MissingDeviceTiming,
}

impl From<ObservationSkipReason> for ProfileSkippedObservation {
    fn from(value: ObservationSkipReason) -> Self {
        match value {
            ObservationSkipReason::NotSubmitted => Self::NotSubmitted,
            ObservationSkipReason::Deferred => Self::Deferred,
            ObservationSkipReason::FailedAfterSubmit => Self::FailedAfterSubmit,
            ObservationSkipReason::UnisolatedPartial => Self::UnisolatedPartial,
            ObservationSkipReason::MissingDeviceTiming => Self::MissingDeviceTiming,
        }
    }
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct ProfileImportCounts {
    pub offered_samples: usize,
    pub recorded_samples: usize,
    pub stale_samples: usize,
    pub skipped_samples: BTreeMap<ProfileSkippedObservation, usize>,
}

#[derive(Debug, Clone, Serialize)]
pub struct CostProfileProvenance {
    pub schema_version: u32,
    pub file_sha256: String,
    pub file_bytes: usize,
    pub loaded_from: Option<PathBuf>,
    pub generated_unix_ns: u64,
    pub loaded_unix_ns: u64,
    pub conservative_clock_error_ns: u64,
    pub oldest_imported_age_ns: Option<u64>,
    pub newest_imported_age_ns: Option<u64>,
    pub source: ProfileSource,
    pub counts: ProfileImportCounts,
}

#[derive(Debug)]
pub struct LoadedCostProfile {
    trainer: CostModelTrainer,
    pub snapshot: ImportedCostSnapshot,
    pub provenance: CostProfileProvenance,
}

impl LoadedCostProfile {
    /// Continue learning without mixing process-relative and imported epochs.
    pub fn observe_live(
        &mut self,
        mut observation: WaveCostObservation,
        local_now_ns: u64,
    ) -> Result<ObservationDisposition, CostProfileError> {
        observation.observed_at_ns = self.snapshot.clock.model_now_ns(local_now_ns)?;
        Ok(self.trainer.observe(observation)?)
    }

    pub fn publish(&mut self, local_now_ns: u64) -> Result<ImportedCostSnapshot, CostProfileError> {
        let clock = self.snapshot.clock;
        self.snapshot = ImportedCostSnapshot {
            snapshot: self.trainer.publish(clock.model_now_ns(local_now_ns)?)?,
            clock,
        };
        Ok(self.snapshot.clone())
    }
}

#[derive(Debug, thiserror::Error)]
pub enum CostProfileError {
    #[error("cost profile IO: {0}")]
    Io(#[from] std::io::Error),
    #[error("cost profile JSON: {0}")]
    Json(#[from] serde_json::Error),
    #[error("unsupported cost profile version {0}")]
    UnsupportedVersion(u32),
    #[error("cost profile limit: {0}")]
    Limit(&'static str),
    #[error("cost profile clock: {0}")]
    Clock(&'static str),
    #[error("cost profile metadata: {0}")]
    Metadata(&'static str),
    #[error("cost profile execution fingerprint mismatch")]
    FingerprintMismatch,
    #[error("cost profile settings differ from the explicit runtime settings")]
    SettingsMismatch,
    #[error("cost profile sample {source_record}: {reason}")]
    Sample {
        source_record: u64,
        reason: CostModelError,
    },
    #[error(transparent)]
    Model(#[from] CostModelError),
}

pub fn load_cost_profile(
    path: &Path,
    fingerprint: &ExecutionFingerprint,
    settings: &CostModelSettings,
    limits: &CostProfileLoadLimits,
    clock: ProfileLoadClock,
) -> Result<LoadedCostProfile, CostProfileError> {
    limits.validate()?;
    let mut file = File::open(path)?;
    let metadata = file.metadata()?;
    if !metadata.is_file() {
        return Err(CostProfileError::Metadata("profile must be a regular file"));
    }
    let length = usize::try_from(metadata.len())
        .ok()
        .filter(|length| *length <= limits.max_file_bytes.get())
        .ok_or(CostProfileError::Limit("file byte limit exceeded"))?;
    // Allocate only the bounded regular-file length. `read_to_end` plus a
    // limit+1 sentinel can otherwise double Vec capacity at the configured
    // boundary. A separate byte probe rejects growth without that allocation.
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(length)
        .map_err(|_| CostProfileError::Limit("profile byte allocation failed"))?;
    bytes.resize(length, 0);
    file.read_exact(&mut bytes)?;
    if file.read(&mut [0_u8; 1])? != 0 {
        return Err(CostProfileError::Metadata("profile grew while reading"));
    }
    let mut loaded = load_cost_profile_bytes(&bytes, fingerprint, settings, limits, clock)?;
    loaded.provenance.loaded_from = Some(path.canonicalize()?);
    Ok(loaded)
}

pub fn load_cost_profile_bytes(
    bytes: &[u8],
    fingerprint: &ExecutionFingerprint,
    settings: &CostModelSettings,
    limits: &CostProfileLoadLimits,
    clock: ProfileLoadClock,
) -> Result<LoadedCostProfile, CostProfileError> {
    limits.validate()?;
    settings.validate()?;
    if bytes.len() > limits.max_file_bytes.get() {
        return Err(CostProfileError::Limit("file byte limit exceeded"));
    }
    let profile = v2::parse(bytes)?;
    if profile.fingerprint != ProfileFingerprint::from(fingerprint) {
        return Err(CostProfileError::FingerprintMismatch);
    }
    let profile_settings = profile.settings;
    profile_settings.validate()?;
    if &profile_settings != settings {
        return Err(CostProfileError::SettingsMismatch);
    }
    if profile.samples.is_empty() {
        return Err(CostProfileError::Metadata("profile has no observations"));
    }
    if profile.samples.len() > limits.max_samples.get() {
        return Err(CostProfileError::Limit("sample limit exceeded"));
    }
    for field in [
        &profile.source.generator,
        &profile.source.generator_revision,
        &profile.source.measurement_protocol,
    ] {
        if field.is_empty()
            || field.len() > limits.max_source_field_bytes.get()
            || field.chars().any(char::is_control)
        {
            return Err(CostProfileError::Metadata(
                "source field must be nonempty, bounded and printable",
            ));
        }
    }
    let wall = clock
        .wall_unix_ns
        .filter(|now| *now > 0)
        .ok_or(CostProfileError::Clock("trusted wall clock is unavailable"))?;
    let source_error = profile
        .source_clock_max_error_ns
        .ok_or(CostProfileError::Clock(
            "source wall clock accuracy is unknown",
        ))?;
    let local_error = clock.wall_max_error_ns.ok_or(CostProfileError::Clock(
        "local wall clock accuracy is unknown",
    ))?;
    if source_error > limits.max_clock_error_ns || local_error > limits.max_clock_error_ns {
        return Err(CostProfileError::Clock(
            "wall clock uncertainty exceeds policy",
        ));
    }
    let uncertainty = source_error
        .checked_add(local_error)
        .ok_or(CostProfileError::Clock("clock uncertainty overflow"))?;
    let age = |at: u64| -> Result<u64, CostProfileError> {
        if at == 0 {
            return Err(CostProfileError::Clock("missing wall timestamp"));
        }
        wall.checked_sub(at)
            .and_then(|age| age.checked_add(uncertainty))
            .ok_or(CostProfileError::Clock(
                "future timestamp or wall age overflow",
            ))
    };
    if age(profile.generated_unix_ns)? > limits.max_profile_age_ns.get() {
        return Err(CostProfileError::Clock("profile generation is too old"));
    }
    let model_clock = ProfileObservationClock {
        source_monotonic_anchor_ns: clock.monotonic_now_ns,
        model_anchor_ns: settings.max_sample_age_ns.get(),
    };
    let mut records = HashSet::new();
    let mut total_rows = 0usize;
    let mut timed = Vec::with_capacity(profile.samples.len());
    for sample in profile.samples {
        if !records.insert(sample.source_record) {
            return Err(CostProfileError::Metadata("duplicate source record"));
        }
        if sample.measured_unix_ns > profile.generated_unix_ns {
            return Err(CostProfileError::Clock(
                "sample timestamp follows profile generation",
            ));
        }
        let rows = sample
            .shape
            .decode_kv_tokens
            .len()
            .checked_add(sample.shape.prefill_chunks.len())
            .ok_or(CostProfileError::Limit("shape row count overflow"))?;
        if rows > settings.shape_limits.max_rows.get() {
            return Err(CostProfileError::Limit("per-wave shape row limit exceeded"));
        }
        if let Some(features) = &sample.shape.numeric_features {
            features
                .validate(rows)
                .map_err(|_| CostProfileError::Metadata("invalid numeric shape features"))?;
        }
        if let Some(features) = &sample.shape.row_multiset_features {
            features
                .validate(rows)
                .map_err(|_| CostProfileError::Metadata("invalid row-multiset features"))?;
        }
        if sample
            .shape
            .host_content_features
            .as_ref()
            .is_some_and(|features| features.validate().is_err())
        {
            return Err(CostProfileError::Metadata("invalid host-content features"));
        }
        total_rows = total_rows
            .checked_add(
                (rows
                    + sample
                        .shape
                        .numeric_features
                        .as_ref()
                        .map_or(0, |features| features.rows.len())
                    + sample
                        .shape
                        .row_multiset_features
                        .as_ref()
                        .map_or(0, |features| features.rows.len()))
                .max(1),
            )
            .ok_or(CostProfileError::Limit("total shape row count overflow"))?;
        if total_rows > limits.max_total_shape_rows.get() {
            return Err(CostProfileError::Limit("total shape row limit exceeded"));
        }
        timed.push((age(sample.measured_unix_ns)?, sample));
    }
    // Oldest first is required by the trainer's monotonic receipt contract.
    timed.sort_by(|(a, _), (b, _)| b.cmp(a));
    let mut counts = ProfileImportCounts {
        offered_samples: timed.len(),
        ..Default::default()
    };
    let mut oldest = None;
    let mut newest = None;
    let mut trainer = CostModelTrainer::new(fingerprint.clone(), profile_settings)?;
    for (age, sample) in timed {
        if age > settings.max_sample_age_ns.get() {
            counts.stale_samples += 1;
            continue;
        }
        let disposition = trainer
            .observe(WaveCostObservation {
                fingerprint: fingerprint.clone(),
                actual_shape: sample.shape,
                boundary: sample.boundary,
                outcome: sample.outcome.into(),
                timing: sample.timing.into(),
                observed_at_ns: model_clock.model_anchor_ns - age,
            })
            .map_err(|reason| CostProfileError::Sample {
                source_record: sample.source_record,
                reason,
            })?;
        match disposition {
            ObservationDisposition::Recorded => {
                counts.recorded_samples += 1;
                oldest = Some(oldest.map_or(age, |previous: u64| previous.max(age)));
                newest = Some(newest.map_or(age, |previous: u64| previous.min(age)));
            }
            ObservationDisposition::Skipped(reason) => {
                *counts.skipped_samples.entry(reason.into()).or_default() += 1;
            }
        }
    }
    let snapshot = ImportedCostSnapshot {
        snapshot: trainer.publish(model_clock.model_anchor_ns)?,
        clock: model_clock,
    };
    Ok(LoadedCostProfile {
        trainer,
        snapshot,
        provenance: CostProfileProvenance {
            schema_version: profile.schema_version,
            file_sha256: format!("sha256:{:x}", Sha256::digest(bytes)),
            file_bytes: bytes.len(),
            loaded_from: None,
            generated_unix_ns: profile.generated_unix_ns,
            loaded_unix_ns: wall,
            conservative_clock_error_ns: uncertainty,
            oldest_imported_age_ns: oldest,
            newest_imported_age_ns: newest,
            source: profile.source,
            counts,
        },
    })
}

#[cfg(test)]
mod tests;

mod structured_v10;
pub use structured_v10::{
    export_structured_profile_v10, export_structured_profile_v11, export_structured_profile_v12,
    load_structured_profile_v10, load_structured_profile_v11, load_structured_profile_v12,
    structured_prefix_source_header_v5, structured_shared_source_header_v4,
    ImportedStructuredCatalogV11, ImportedStructuredCatalogV12, ImportedStructuredModelV2,
    StructuredImportProvenanceV10, StructuredPhaseProvenanceV10, StructuredProfileExportReceiptV10,
    StructuredProfileExportReceiptV11, StructuredProfileExportReceiptV12,
    StructuredProfilePhaseV10, COST_PROFILE_SCHEMA_VERSION_V10,
};
