//! Empirical costs for the executor's actual, non-returning wave.
//!
//! The trainer is a bounded slow path. Readers use immutable snapshots without
//! locks, JSON, device access, or calibration work. Unknown is never a zero-cost
//! prediction. These empirical margins are not statistical confidence bounds.
//!
//! All `*_at_ns` and `now_ns` values belong to one monotonic observation clock
//! relative to the trainer's local epoch. This module deliberately does not
//! serialize snapshots or timestamps. A profile loader must verify freshness
//! and explicitly re-anchor imported observations before inserting them; an old
//! process's `Instant` offset must never be used as a current-process timestamp.

use std::{
    collections::{BTreeMap, VecDeque},
    num::{NonZeroU32, NonZeroU64, NonZeroUsize},
    sync::Arc,
};

use ferrum_interfaces::execution_cost::{
    CanonicalWaveCostFeatures, HostContentCostFeaturesV1, HostRowMultisetCostFeaturesV2,
};
pub use ferrum_types::SloCostFeatureModel as CostFeatureModel;
#[cfg(test)]
#[path = "cost_model/host_content_tests.rs"]
mod host_content_tests;
mod multiset;
mod numeric;
#[cfg(test)]
mod numeric_tests;

#[cfg(test)]
#[path = "cost_model/expiry_tests.rs"]
mod expiry_tests;
#[cfg(test)]
mod tests;

/// Stable execution-relevant identities, computed once by the execution layer.
///
/// Do not include request IDs, deadlines, temporary paths, or incidental CLI
/// arguments. `numerical_policy` includes weight/KV precision and numerical
/// policy; `device_runtime` includes backend/device/topology/runtime versions;
/// `execution_config` includes execution-relevant serving configuration only.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExecutionFingerprint {
    pub model_weights: [u8; 32],
    pub numerical_policy: [u8; 32],
    pub device_runtime: [u8; 32],
    pub execution_config: [u8; 32],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum WaveKind {
    Decode,
    Prefill,
    Mixed,
    Restore,
    Maintenance,
}

/// Actual executor path; a planned native wave that fell back is not Native.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum WaveExecutionPath {
    PlanRuntime,
    NativeUnified,
    LegacySplit,
    UnsupportedFallback,
    CapacityFallback,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum WaveGraphState {
    Disabled,
    Cold,
    Warm,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum BatchOrderSemantics {
    /// Physical row order is part of the key, including any ordered dependency.
    Ordered,
    /// The executor guarantees row-permutation invariance for this wave.
    IndependentRows,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct PrefillShape {
    pub offset: u32,
    pub count: NonZeroU32,
    pub total_prompt_tokens: NonZeroU32,
}

/// Shape of the work actually performed, not the originally requested batch.
///
/// Within each row vector, preserve executor order unless `IndependentRows` is
/// explicitly guaranteed. Providers with interleaved or other ordered work
/// must encode that schedule in `provider_signature` or use separate waves.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct WaveExecutionShape {
    /// Explicit empirical row-class evidence. Physical order remains intact;
    /// only the opt-in V2 model canonicalizes a private statistical copy.
    pub row_multiset_features: Option<HostRowMultisetCostFeaturesV2>,
    /// Separate, capability-backed empirical content identity. Legacy model
    /// keys ignore this extension; it never authorizes an executable shape.
    pub host_content_features: Option<HostContentCostFeaturesV1>,
    /// Versioned work evidence. ExactV1 deliberately excludes this extension
    /// from its bucket identity; old profiles do not acquire invented features.
    pub numeric_features: Option<CanonicalWaveCostFeatures>,
    pub kind: WaveKind,
    pub path: WaveExecutionPath,
    pub provider_signature: [u8; 32],
    /// Sampling/logits/structured-output policy and recurrent execution variant.
    pub output_policy_signature: [u8; 32],
    pub graph_state: WaveGraphState,
    pub order: BatchOrderSemantics,
    pub decode_kv_tokens: Vec<u32>,
    pub prefill_chunks: Vec<PrefillShape>,
    pub recurrent_state_bytes: u64,
    pub restore_bytes: u64,
    pub maintenance_bytes: u64,
    pub maintenance_units: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum CostBoundary {
    /// Measured wall interval from preparation through irreversible token commit
    /// (or restore/maintenance completion for a wave of that kind).
    PreparationToCommit,
    /// Independently measured device elapsed interval; never the sum of kernels.
    DeviceOnly,
    /// Complete actual wave through all host row settlement, including an
    /// eligible terminal owner. This is never an upgraded token-commit sample.
    PreparationToHostSettledV1,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MeasuredSpan {
    /// Offsets within the measured wall interval, not clock timestamps.
    pub start_ns: u64,
    pub end_ns: u64,
}

/// Optional, mutually non-overlapping wall-stage diagnostics. Concurrent work
/// must not be presented as distinct non-overlapping stages.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct WaveStageTimings {
    pub prepare: Option<MeasuredSpan>,
    pub device_wait: Option<MeasuredSpan>,
    pub commit: Option<MeasuredSpan>,
    pub restore: Option<MeasuredSpan>,
    pub maintenance: Option<MeasuredSpan>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WaveTiming {
    /// Primary sample. Never reconstructed by summing overlapping operations.
    pub wall_total_ns: u64,
    pub device_elapsed_ns: Option<u64>,
    pub stages: WaveStageTimings,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WaveObservationOutcome {
    Completed,
    /// Only isolated timing of the actual completed shape may train the model.
    PartiallyCompleted {
        timing_covers_actual_shape_only: bool,
    },
    NotSubmitted,
    Deferred,
    FailedAfterSubmit,
}

#[derive(Debug, Clone)]
pub struct WaveCostObservation {
    pub fingerprint: ExecutionFingerprint,
    pub actual_shape: WaveExecutionShape,
    pub boundary: CostBoundary,
    pub outcome: WaveObservationOutcome,
    pub timing: WaveTiming,
    /// Receipt time on the trainer's monotonic observation clock. Delivering a
    /// completed measurement later must use its receipt time, not a past offset.
    pub observed_at_ns: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CostShapeLimits {
    pub max_rows: NonZeroUsize,
    pub max_context_tokens: NonZeroU32,
    pub max_prefill_tokens_per_wave: NonZeroU64,
    pub max_state_bytes: NonZeroU64,
    pub max_maintenance_units: NonZeroU32,
}

/// Bounded experimental calibration settings, not validated production values.
#[derive(Debug, Clone, PartialEq)]
pub struct CostModelSettings {
    pub feature_model: CostFeatureModel,
    pub max_buckets: NonZeroUsize,
    pub max_samples_per_bucket: NonZeroUsize,
    pub min_samples: NonZeroUsize,
    pub max_retained_samples: NonZeroUsize,
    pub max_retained_shape_rows: NonZeroUsize,
    pub residual_quantile: f64,
    pub drift_margin_ns: u64,
    pub max_wave_ns: NonZeroU64,
    pub max_sample_age_ns: NonZeroU64,
    /// One means exact lookup. Wider buckets still require observed coverage.
    pub context_bucket_tokens: NonZeroU32,
    pub prefill_offset_bucket_tokens: NonZeroU32,
    pub shape_limits: CostShapeLimits,
}

impl Default for CostModelSettings {
    fn default() -> Self {
        Self {
            feature_model: CostFeatureModel::default(),
            max_buckets: NonZeroUsize::new(1024).unwrap(),
            max_samples_per_bucket: NonZeroUsize::new(128).unwrap(),
            min_samples: NonZeroUsize::new(8).unwrap(),
            max_retained_samples: NonZeroUsize::new(16_384).unwrap(),
            max_retained_shape_rows: NonZeroUsize::new(262_144).unwrap(),
            residual_quantile: 0.99,
            drift_margin_ns: 100_000,
            max_wave_ns: NonZeroU64::new(60_000_000_000).unwrap(),
            max_sample_age_ns: NonZeroU64::new(300_000_000_000).unwrap(),
            context_bucket_tokens: NonZeroU32::new(1).unwrap(),
            prefill_offset_bucket_tokens: NonZeroU32::new(1).unwrap(),
            shape_limits: CostShapeLimits {
                max_rows: NonZeroUsize::new(128).unwrap(),
                max_context_tokens: NonZeroU32::new(262_144).unwrap(),
                max_prefill_tokens_per_wave: NonZeroU64::new(65_536).unwrap(),
                max_state_bytes: NonZeroU64::new(1 << 40).unwrap(),
                max_maintenance_units: NonZeroU32::new(1_048_576).unwrap(),
            },
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum CostModelError {
    #[error("invalid cost-model settings: {0}")]
    InvalidSettings(&'static str),
    #[error("execution fingerprint does not match calibration")]
    FingerprintMismatch,
    #[error("invalid wave shape: {0}")]
    InvalidShape(&'static str),
    #[error("invalid wave timing: {0}")]
    InvalidTiming(&'static str),
    #[error("observation clock moved backwards")]
    ClockMovedBackwards,
    #[error("cost-model capacity exhausted: {0}")]
    CapacityExceeded(&'static str),
    #[error("cost-model arithmetic overflow")]
    ArithmeticOverflow,
    #[error("cost-model published version exhausted")]
    VersionExhausted,
    #[error("drift-margin decrease requires a new calibration epoch")]
    MarginDecreaseRequiresRecalibration,
}

impl CostModelSettings {
    pub fn validate(&self) -> Result<(), CostModelError> {
        if !self.residual_quantile.is_finite()
            || self.residual_quantile <= 0.0
            || self.residual_quantile > 1.0
        {
            return Err(CostModelError::InvalidSettings(
                "residual quantile must be finite in (0, 1]",
            ));
        }
        if self.max_buckets.get() > 65_536
            || self.max_samples_per_bucket.get() > 4096
            || self.max_retained_samples.get() > 131_072
            || self.max_retained_shape_rows.get() > 1_048_576
            || self.shape_limits.max_rows.get() > 1024
        {
            return Err(CostModelError::InvalidSettings(
                "hard retention limit exceeded",
            ));
        }
        if self.min_samples > self.max_samples_per_bucket
            || self.min_samples > self.max_retained_samples
            || self.shape_limits.max_rows > self.max_retained_shape_rows
        {
            return Err(CostModelError::InvalidSettings(
                "minimum coverage exceeds capacity",
            ));
        }
        self.max_wave_ns
            .get()
            .checked_add(self.drift_margin_ns)
            .ok_or(CostModelError::ArithmeticOverflow)?;
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ObservationSkipReason {
    NotSubmitted,
    Deferred,
    FailedAfterSubmit,
    UnisolatedPartial,
    MissingDeviceTiming,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ObservationDisposition {
    Recorded,
    Skipped(ObservationSkipReason),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CostUnknownReason {
    FingerprintMismatch,
    InvalidShape,
    UnobservedBucket,
    InsufficientSamples,
    OutsideObservedCoverage,
    StaleSamples,
    ClockMovedBackwards,
    NumericFeaturesMissing,
    NumericRowOrderUnsupported,
    OutsideJointNumericSupport,
    HostContentFeaturesMissing,
    BoundaryUnsupported,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PredictionErrorSummary {
    /// Samples that were compared with a valid previously published prediction.
    pub compared_samples: u64,
    pub underestimates: u64,
    pub max_underestimate_ns: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WaveCostPrediction {
    pub typical_ns: u64,
    pub planning_ns: u64,
    /// Remaining inclusive sample-freshness interval from the prediction's
    /// query time. A duration survives imported clock re-anchoring unchanged.
    pub valid_for_ns: u64,
    pub sample_count: usize,
    pub model_version: u64,
    pub boundary: CostBoundary,
    /// Nonnegative empirical residual quantile added to the median cost.
    pub residual_margin_ns: u64,
    /// NumericV1 uses the maximum complete-wave observation in its support
    /// group before drift/floors. None denotes the legacy quantile model.
    pub empirical_envelope_ns: Option<u64>,
    pub drift_margin_ns: u64,
    /// Maximum of this bucket's earlier published planning costs and the
    /// epoch floor retained from expired buckets of the same kind/boundary.
    /// Neither is freshness or evidence of sample coverage.
    pub prior_planning_floor_ns: u64,
    pub oldest_sample_at_ns: u64,
    pub newest_sample_at_ns: u64,
    pub errors: PredictionErrorSummary,
    pub coverage: Arc<ObservedShapeCoverage>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CostPrediction {
    Known(WaveCostPrediction),
    Unknown(CostUnknownReason),
}

/// Rectangular interpolation envelope inside a single explicitly enabled
/// bucket. Width-one defaults remain exact. No extrapolation is permitted.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ObservedShapeCoverage {
    pub decode_context_ranges: Vec<(u32, u32)>,
    pub prefill_offset_ranges: Vec<(u32, u32)>,
    /// A single real joint point must cover all work axes. Coordinate extrema
    /// from different observations are never combined into a fictitious point.
    numeric: Option<numeric::NumericSupport>,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct BucketKey {
    boundary: CostBoundary,
    shape: WaveExecutionShape,
    // A final prompt fragment can emit a first token and run a different head.
    // Explicit offset bucketing must never merge that work with an intermediate.
    prefill_finishes_prompt: Vec<bool>,
}

#[derive(Debug, Clone)]
struct CostSample {
    shape: WaveExecutionShape,
    cost_ns: u64,
    observed_at_ns: u64,
}

#[derive(Debug, Default)]
struct TrainingBucket {
    samples: VecDeque<CostSample>,
    errors: PredictionErrorSummary,
}

#[derive(Debug, Clone)]
struct CalibratedBucket {
    prediction: Option<WaveCostPrediction>,
    retained_sample_count: usize,
}

/// Immutable and bounded. Publishing another version does not change readers
/// holding this snapshot. Predictions also check sample age at the read time.
#[derive(Debug)]
pub struct CostModelSnapshot {
    fingerprint: ExecutionFingerprint,
    settings: CostModelSettings,
    model_version: u64,
    published_at_ns: u64,
    buckets: BTreeMap<BucketKey, CalibratedBucket>,
}

impl CostModelSnapshot {
    pub fn planning_boundary(&self) -> CostBoundary {
        if matches!(
            self.settings.feature_model,
            CostFeatureModel::EmpiricalHostContentV1 { .. }
                | CostFeatureModel::EmpiricalRowMultisetV2 { .. }
        ) {
            CostBoundary::PreparationToHostSettledV1
        } else {
            CostBoundary::PreparationToCommit
        }
    }
    pub fn model_version(&self) -> u64 {
        self.model_version
    }
    pub fn fingerprint(&self) -> &ExecutionFingerprint {
        &self.fingerprint
    }
    pub fn bucket_count(&self) -> usize {
        self.buckets.len()
    }

    pub fn predict(
        &self,
        fingerprint: &ExecutionFingerprint,
        shape: &WaveExecutionShape,
        boundary: CostBoundary,
        now_ns: u64,
    ) -> CostPrediction {
        if fingerprint != &self.fingerprint {
            return CostPrediction::Unknown(CostUnknownReason::FingerprintMismatch);
        }
        if now_ns < self.published_at_ns {
            return CostPrediction::Unknown(CostUnknownReason::ClockMovedBackwards);
        }
        let mut canonical = match canonical_shape(shape, &self.settings.shape_limits) {
            Ok(shape) => shape,
            Err(_) => return CostPrediction::Unknown(CostUnknownReason::InvalidShape),
        };
        if let Err(reason) = numeric::validate_mode(&canonical, &self.settings.feature_model) {
            return CostPrediction::Unknown(reason);
        }
        if multiset::enabled(&self.settings.feature_model) {
            multiset::statistical_order(&mut canonical);
        }
        if !boundary_supported(&self.settings.feature_model, boundary) {
            return CostPrediction::Unknown(CostUnknownReason::BoundaryUnsupported);
        }
        let key = bucket_key(&canonical, boundary, &self.settings);
        let Some(bucket) = self.buckets.get(&key) else {
            return CostPrediction::Unknown(CostUnknownReason::UnobservedBucket);
        };
        let Some(prediction) = &bucket.prediction else {
            return CostPrediction::Unknown(
                if bucket.retained_sample_count >= self.settings.min_samples.get() {
                    CostUnknownReason::StaleSamples
                } else {
                    CostUnknownReason::InsufficientSamples
                },
            );
        };
        // The published quantile/envelope used every sample, so all must remain
        // fresh. Do not pretend that an old snapshot recalculated its quantile.
        let Some(valid_for_ns) = now_ns
            .checked_sub(prediction.oldest_sample_at_ns)
            .and_then(|age| self.settings.max_sample_age_ns.get().checked_sub(age))
        else {
            return CostPrediction::Unknown(CostUnknownReason::StaleSamples);
        };
        if !prediction.coverage.contains(&canonical) {
            return CostPrediction::Unknown(CostUnknownReason::OutsideObservedCoverage);
        }
        if prediction
            .coverage
            .numeric
            .as_ref()
            .is_some_and(|support| !support.contains(&canonical))
        {
            return CostPrediction::Unknown(CostUnknownReason::OutsideJointNumericSupport);
        }
        let mut current_prediction = prediction.clone();
        current_prediction.valid_for_ns = valid_for_ns;
        CostPrediction::Known(current_prediction)
    }
}

/// Bounded slow-path recorder and publisher. Planning costs and drift margin
/// never decrease in a calibration epoch. A decrease requires a new trainer
/// and fresh `min_samples` coverage, preserving the last published version via
/// `after_version`; old samples are not automatically carried into that epoch.
#[derive(Debug)]
pub struct CostModelTrainer {
    fingerprint: ExecutionFingerprint,
    settings: CostModelSettings,
    buckets: BTreeMap<BucketKey, TrainingBucket>,
    planning_floors: BTreeMap<BucketKey, u64>,
    // Expired identities cannot leave an unbounded tombstone map. Folding their
    // floors into fixed kind/boundary slots preserves conservative recreation
    // without mixing long prefill with decode or wall with device-only costs.
    retired_floors_ns: [u64; 15],
    retained_samples: usize,
    retained_rows: usize,
    last_clock_ns: u64,
    last_version: u64,
    dirty: bool,
    published: Option<Arc<CostModelSnapshot>>,
}

impl CostModelTrainer {
    pub fn new(
        fingerprint: ExecutionFingerprint,
        settings: CostModelSettings,
    ) -> Result<Self, CostModelError> {
        Self::after_version(fingerprint, settings, 0)
    }

    /// Start an explicitly new calibration epoch without reusing sample clocks.
    pub fn after_version(
        fingerprint: ExecutionFingerprint,
        settings: CostModelSettings,
        last_version: u64,
    ) -> Result<Self, CostModelError> {
        settings.validate()?;
        Ok(Self {
            fingerprint,
            settings,
            buckets: BTreeMap::new(),
            planning_floors: BTreeMap::new(),
            retired_floors_ns: [0; 15],
            retained_samples: 0,
            retained_rows: 0,
            last_clock_ns: 0,
            last_version,
            dirty: true,
            published: None,
        })
    }

    pub fn retained_sample_count(&self) -> usize {
        self.retained_samples
    }
    pub fn retained_shape_rows(&self) -> usize {
        self.retained_rows
    }

    /// Conservative kind/boundary floor after expired bucket identities were
    /// reclaimed. Reducing it requires explicit `after_version` recalibration.
    /// It supplies no sample coverage and cannot make an unknown bucket Known.
    pub fn retired_planning_floor_ns(&self, kind: WaveKind, boundary: CostBoundary) -> u64 {
        self.retired_floors_ns[retired_floor_slot(kind, boundary)]
    }

    /// A conservative reaction to measured drift. Reductions require explicit
    /// recalibration; equal values do not manufacture a new published version.
    pub fn raise_drift_margin_ns(&mut self, margin_ns: u64) -> Result<(), CostModelError> {
        if margin_ns < self.settings.drift_margin_ns {
            return Err(CostModelError::MarginDecreaseRequiresRecalibration);
        }
        self.settings
            .max_wave_ns
            .get()
            .checked_add(margin_ns)
            .ok_or(CostModelError::ArithmeticOverflow)?;
        if margin_ns > self.settings.drift_margin_ns {
            self.settings.drift_margin_ns = margin_ns;
            self.dirty = true;
        }
        Ok(())
    }

    pub fn observe(
        &mut self,
        observation: WaveCostObservation,
    ) -> Result<ObservationDisposition, CostModelError> {
        let skip = match observation.outcome {
            WaveObservationOutcome::NotSubmitted => Some(ObservationSkipReason::NotSubmitted),
            WaveObservationOutcome::Deferred => Some(ObservationSkipReason::Deferred),
            WaveObservationOutcome::FailedAfterSubmit => {
                Some(ObservationSkipReason::FailedAfterSubmit)
            }
            WaveObservationOutcome::PartiallyCompleted {
                timing_covers_actual_shape_only: false,
            } => Some(ObservationSkipReason::UnisolatedPartial),
            _ => None,
        };
        if let Some(reason) = skip {
            return Ok(ObservationDisposition::Skipped(reason));
        }
        if observation.fingerprint != self.fingerprint {
            return Err(CostModelError::FingerprintMismatch);
        }
        if observation.observed_at_ns < self.last_clock_ns {
            return Err(CostModelError::ClockMovedBackwards);
        }
        let mut canonical =
            canonical_shape(&observation.actual_shape, &self.settings.shape_limits)?;
        numeric::validate_mode(&canonical, &self.settings.feature_model)
            .map_err(|_| CostModelError::InvalidShape("numeric feature evidence unavailable"))?;
        if multiset::enabled(&self.settings.feature_model) {
            multiset::statistical_order(&mut canonical);
        }
        if !boundary_supported(&self.settings.feature_model, observation.boundary) {
            return Err(CostModelError::InvalidTiming(
                "boundary does not match feature model",
            ));
        }
        if observation.boundary == CostBoundary::PreparationToHostSettledV1
            && observation.outcome != WaveObservationOutcome::Completed
        {
            return Ok(ObservationDisposition::Skipped(
                ObservationSkipReason::UnisolatedPartial,
            ));
        }
        validate_timing(&observation.timing, self.settings.max_wave_ns.get())?;
        let cost_ns = match observation.boundary {
            CostBoundary::PreparationToCommit | CostBoundary::PreparationToHostSettledV1 => {
                observation.timing.wall_total_ns
            }
            CostBoundary::DeviceOnly => match observation.timing.device_elapsed_ns {
                Some(cost) => cost,
                None => {
                    return Ok(ObservationDisposition::Skipped(
                        ObservationSkipReason::MissingDeviceTiming,
                    ))
                }
            },
        };
        let key = bucket_key(&canonical, observation.boundary, &self.settings);
        // First compute prospective expiry and capacity without mutation. Even
        // a valid but unretainable new observation must not alter snapshots,
        // floors, clocks or retained samples by opportunistic cleanup.
        let expiry = self.expiry_plan(&key, observation.observed_at_ns)?;
        let new_buckets = self
            .buckets
            .len()
            .checked_sub(expiry.buckets)
            .and_then(|count| count.checked_add(usize::from(expiry.target_samples == 0)))
            .ok_or(CostModelError::ArithmeticOverflow)?;
        if new_buckets > self.settings.max_buckets.get() {
            return Err(CostModelError::CapacityExceeded("buckets"));
        }
        let replaced = expiry.target_samples >= self.settings.max_samples_per_bucket.get();
        let new_samples = self
            .retained_samples
            .checked_sub(expiry.samples)
            .and_then(|count| count.checked_add(usize::from(!replaced)))
            .ok_or(CostModelError::ArithmeticOverflow)?;
        let rows = shape_rows(&canonical);
        let new_rows = self
            .retained_rows
            .checked_sub(expiry.rows)
            .and_then(|count| {
                count.checked_sub(if replaced {
                    expiry.target_oldest_rows
                } else {
                    0
                })
            })
            .and_then(|count| count.checked_add(rows))
            .ok_or(CostModelError::ArithmeticOverflow)?;
        if new_samples > self.settings.max_retained_samples.get() {
            return Err(CostModelError::CapacityExceeded("retained samples"));
        }
        if new_rows > self.settings.max_retained_shape_rows.get() {
            return Err(CostModelError::CapacityExceeded("retained shape rows"));
        }
        let previous = self.published.as_ref().map(|snapshot| {
            snapshot.predict(
                &self.fingerprint,
                &canonical,
                observation.boundary,
                observation.observed_at_ns,
            )
        });
        // All fallible checks precede mutation, including capacity accounting.
        self.reclaim_expired(observation.observed_at_ns);
        let bucket = self.buckets.entry(key).or_default();
        if let Some(CostPrediction::Known(prediction)) = previous {
            bucket.errors.compared_samples = bucket.errors.compared_samples.saturating_add(1);
            if cost_ns > prediction.planning_ns {
                bucket.errors.underestimates = bucket.errors.underestimates.saturating_add(1);
                bucket.errors.max_underestimate_ns = bucket
                    .errors
                    .max_underestimate_ns
                    .max(cost_ns - prediction.planning_ns);
            }
        }
        if bucket.samples.len() >= self.settings.max_samples_per_bucket.get() {
            bucket.samples.pop_front();
        }
        bucket.samples.push_back(CostSample {
            shape: canonical,
            cost_ns,
            observed_at_ns: observation.observed_at_ns,
        });
        self.retained_samples = new_samples;
        self.retained_rows = new_rows;
        self.last_clock_ns = observation.observed_at_ns;
        self.dirty = true;
        Ok(ObservationDisposition::Recorded)
    }

    fn expiry_plan(&self, target: &BucketKey, now_ns: u64) -> Result<ExpiryPlan, CostModelError> {
        let mut plan = ExpiryPlan::default();
        let ttl = self.settings.max_sample_age_ns.get();
        for (key, bucket) in &self.buckets {
            let mut expired = 0usize;
            for sample in bucket
                .samples
                .iter()
                .take_while(|sample| now_ns - sample.observed_at_ns > ttl)
            {
                expired += 1;
                plan.rows = plan
                    .rows
                    .checked_add(shape_rows(&sample.shape))
                    .ok_or(CostModelError::ArithmeticOverflow)?;
            }
            plan.samples = plan
                .samples
                .checked_add(expired)
                .ok_or(CostModelError::ArithmeticOverflow)?;
            if expired == bucket.samples.len() {
                plan.buckets += 1;
            }
            if key == target {
                plan.target_samples = bucket.samples.len() - expired;
                plan.target_oldest_rows = bucket
                    .samples
                    .get(expired)
                    .map_or(0, |sample| shape_rows(&sample.shape));
            }
        }
        Ok(plan)
    }

    /// Runs only after the incoming observation and resulting capacities have
    /// been validated. Receipt order makes expired samples a prefix per bucket.
    /// Existing snapshots keep their immutable samples/TTL and are not touched.
    fn reclaim_expired(&mut self, now_ns: u64) {
        let ttl = self.settings.max_sample_age_ns.get();
        let floors = &mut self.planning_floors;
        let retired = &mut self.retired_floors_ns;
        self.buckets.retain(|key, bucket| {
            while bucket
                .samples
                .front()
                .is_some_and(|sample| now_ns - sample.observed_at_ns > ttl)
            {
                bucket.samples.pop_front();
            }
            if bucket.samples.is_empty() {
                if let Some(floor) = floors.remove(key) {
                    let slot = retired_floor_slot(key.shape.kind, key.boundary);
                    retired[slot] = retired[slot].max(floor);
                }
                false
            } else {
                true
            }
        });
    }

    /// No new samples or margin change means the same Arc/version is returned.
    /// Age is checked by `predict`; merely advancing time is not new evidence.
    pub fn publish(&mut self, now_ns: u64) -> Result<Arc<CostModelSnapshot>, CostModelError> {
        if now_ns < self.last_clock_ns {
            return Err(CostModelError::ClockMovedBackwards);
        }
        if !self.dirty {
            self.last_clock_ns = now_ns;
            return Ok(Arc::clone(
                self.published
                    .as_ref()
                    .expect("clean trainer has a snapshot"),
            ));
        }
        let version = self
            .last_version
            .checked_add(1)
            .ok_or(CostModelError::VersionExhausted)?;
        let mut calibrated = BTreeMap::new();
        let mut floors = self.planning_floors.clone();
        for (key, bucket) in &self.buckets {
            let fresh: Vec<_> = bucket
                .samples
                .iter()
                .filter(|sample| {
                    now_ns - sample.observed_at_ns <= self.settings.max_sample_age_ns.get()
                })
                .collect();
            let prediction = if fresh.len() >= self.settings.min_samples.get() {
                let mut costs: Vec<_> = fresh.iter().map(|sample| sample.cost_ns).collect();
                costs.sort_unstable();
                let typical = quantile(&costs, 0.5);
                let residual_margin =
                    quantile(&costs, self.settings.residual_quantile).saturating_sub(typical);
                // Numeric support uses a complete-wave empirical envelope,
                // not interpolation of kernel timings or independent axes.
                let empirical_cost = match self.settings.feature_model {
                    CostFeatureModel::ExactV1 {}
                    | CostFeatureModel::EmpiricalHostContentV1 { .. }
                    | CostFeatureModel::EmpiricalRowMultisetV2 { .. } => {
                        typical.checked_add(residual_margin)
                    }
                    CostFeatureModel::BoundedNumericV1 { .. } => costs.last().copied(),
                }
                .ok_or(CostModelError::ArithmeticOverflow)?;
                let prior_floor = floors
                    .get(key)
                    .copied()
                    .unwrap_or(0)
                    .max(self.retired_planning_floor_ns(key.shape.kind, key.boundary));
                let planning_ns = empirical_cost
                    .checked_add(self.settings.drift_margin_ns)
                    .ok_or(CostModelError::ArithmeticOverflow)?
                    .max(prior_floor);
                floors.insert(key.clone(), planning_ns);
                Some(WaveCostPrediction {
                    typical_ns: typical,
                    planning_ns,
                    valid_for_ns: self.settings.max_sample_age_ns.get()
                        - (now_ns - fresh.first().unwrap().observed_at_ns),
                    sample_count: fresh.len(),
                    model_version: version,
                    boundary: key.boundary,
                    residual_margin_ns: residual_margin,
                    empirical_envelope_ns: matches!(
                        self.settings.feature_model,
                        CostFeatureModel::BoundedNumericV1 { .. }
                    )
                    .then_some(empirical_cost),
                    drift_margin_ns: self.settings.drift_margin_ns,
                    prior_planning_floor_ns: prior_floor,
                    oldest_sample_at_ns: fresh.first().unwrap().observed_at_ns,
                    newest_sample_at_ns: fresh.last().unwrap().observed_at_ns,
                    errors: bucket.errors,
                    coverage: Arc::new(ObservedShapeCoverage::from_samples(
                        &fresh,
                        &self.settings.feature_model,
                    )),
                })
            } else {
                None
            };
            calibrated.insert(
                key.clone(),
                CalibratedBucket {
                    prediction,
                    retained_sample_count: bucket.samples.len(),
                },
            );
        }
        let snapshot = Arc::new(CostModelSnapshot {
            fingerprint: self.fingerprint.clone(),
            settings: self.settings.clone(),
            model_version: version,
            published_at_ns: now_ns,
            buckets: calibrated,
        });
        self.planning_floors = floors;
        self.last_version = version;
        self.last_clock_ns = now_ns;
        self.dirty = false;
        self.published = Some(Arc::clone(&snapshot));
        Ok(snapshot)
    }
}

#[derive(Default)]
struct ExpiryPlan {
    buckets: usize,
    samples: usize,
    rows: usize,
    target_samples: usize,
    target_oldest_rows: usize,
}

/// Fixed storage for every currently supported cost population; no hash or
/// provider identity can increase this retention after its bucket expires.
fn retired_floor_slot(kind: WaveKind, boundary: CostBoundary) -> usize {
    let kind = match kind {
        WaveKind::Decode => 0,
        WaveKind::Prefill => 1,
        WaveKind::Mixed => 2,
        WaveKind::Restore => 3,
        WaveKind::Maintenance => 4,
    };
    let boundary = match boundary {
        CostBoundary::PreparationToCommit => 0,
        CostBoundary::DeviceOnly => 1,
        CostBoundary::PreparationToHostSettledV1 => 2,
    };
    3 * kind + boundary
}

fn boundary_supported(mode: &CostFeatureModel, boundary: CostBoundary) -> bool {
    matches!(
        mode,
        CostFeatureModel::EmpiricalHostContentV1 { .. }
            | CostFeatureModel::EmpiricalRowMultisetV2 { .. }
    ) == (boundary == CostBoundary::PreparationToHostSettledV1)
}

fn shape_rows(shape: &WaveExecutionShape) -> usize {
    // Restore/maintenance samples still consume a retained-shape slot.
    (shape.decode_kv_tokens.len()
        + shape.prefill_chunks.len()
        + shape
            .numeric_features
            .as_ref()
            .map_or(0, |features| features.rows.len())
        + shape
            .row_multiset_features
            .as_ref()
            .map_or(0, |features| features.rows.len()))
    .max(1)
}

fn canonical_shape(
    shape: &WaveExecutionShape,
    limits: &CostShapeLimits,
) -> Result<WaveExecutionShape, CostModelError> {
    if shape
        .decode_kv_tokens
        .len()
        .checked_add(shape.prefill_chunks.len())
        .ok_or(CostModelError::ArithmeticOverflow)?
        > limits.max_rows.get()
    {
        return Err(CostModelError::InvalidShape("row limit exceeded"));
    }
    let has_decode = !shape.decode_kv_tokens.is_empty();
    let has_prefill = !shape.prefill_chunks.is_empty();
    if let Some(features) = &shape.numeric_features {
        features
            .validate(shape.decode_kv_tokens.len() + shape.prefill_chunks.len())
            .map_err(|_| CostModelError::InvalidShape("invalid numeric feature evidence"))?;
    }
    if let Some(features) = &shape.row_multiset_features {
        features
            .validate(shape.decode_kv_tokens.len() + shape.prefill_chunks.len())
            .map_err(|_| CostModelError::InvalidShape("invalid row-multiset evidence"))?;
    }
    let kind_valid = match shape.kind {
        WaveKind::Decode => has_decode && !has_prefill,
        WaveKind::Prefill => !has_decode && has_prefill,
        WaveKind::Mixed => has_decode && has_prefill,
        WaveKind::Restore => !has_decode && !has_prefill && shape.restore_bytes > 0,
        WaveKind::Maintenance => {
            !has_decode
                && !has_prefill
                && (shape.maintenance_bytes > 0 || shape.maintenance_units > 0)
        }
    };
    if !kind_valid {
        return Err(CostModelError::InvalidShape(
            "work does not match wave kind",
        ));
    }
    if shape
        .decode_kv_tokens
        .iter()
        .any(|&tokens| tokens == 0 || tokens > limits.max_context_tokens.get())
    {
        return Err(CostModelError::InvalidShape(
            "decode context outside limits",
        ));
    }
    let mut total_prefill = 0_u64;
    for chunk in &shape.prefill_chunks {
        let end = chunk
            .offset
            .checked_add(chunk.count.get())
            .ok_or(CostModelError::ArithmeticOverflow)?;
        if end > chunk.total_prompt_tokens.get()
            || chunk.total_prompt_tokens.get() > limits.max_context_tokens.get()
        {
            return Err(CostModelError::InvalidShape(
                "prefill interval outside prompt",
            ));
        }
        total_prefill = total_prefill
            .checked_add(u64::from(chunk.count.get()))
            .ok_or(CostModelError::ArithmeticOverflow)?;
    }
    if total_prefill > limits.max_prefill_tokens_per_wave.get() {
        return Err(CostModelError::InvalidShape("prefill token limit exceeded"));
    }
    let state_bytes = shape
        .recurrent_state_bytes
        .checked_add(shape.restore_bytes)
        .and_then(|bytes| bytes.checked_add(shape.maintenance_bytes))
        .ok_or(CostModelError::ArithmeticOverflow)?;
    if state_bytes > limits.max_state_bytes.get()
        || shape.maintenance_units > limits.max_maintenance_units.get()
    {
        return Err(CostModelError::InvalidShape(
            "state or maintenance limit exceeded",
        ));
    }
    let mut canonical = shape.clone();
    if shape.order == BatchOrderSemantics::IndependentRows {
        canonical.decode_kv_tokens.sort_unstable();
        // Keep exact chunk count and prompt length ahead of bucketed offset so
        // row alignment remains stable within a bucket.
        canonical
            .prefill_chunks
            .sort_unstable_by_key(|chunk| (chunk.count, chunk.total_prompt_tokens, chunk.offset));
    }
    Ok(canonical)
}

fn bucket_key(
    shape: &WaveExecutionShape,
    boundary: CostBoundary,
    settings: &CostModelSettings,
) -> BucketKey {
    let prefill_finishes_prompt = shape
        .prefill_chunks
        .iter()
        .map(|chunk| chunk.offset + chunk.count.get() == chunk.total_prompt_tokens.get())
        .collect();
    let mut bucket_shape = shape.clone();
    numeric::project_bucket(&mut bucket_shape, &settings.feature_model);
    for context in &mut bucket_shape.decode_kv_tokens {
        *context /= settings.context_bucket_tokens.get();
    }
    for chunk in &mut bucket_shape.prefill_chunks {
        chunk.offset /= settings.prefill_offset_bucket_tokens.get();
    }
    BucketKey {
        boundary,
        shape: bucket_shape,
        prefill_finishes_prompt,
    }
}

pub(super) fn validate_timing(timing: &WaveTiming, max_wave_ns: u64) -> Result<(), CostModelError> {
    if timing.wall_total_ns == 0 || timing.wall_total_ns > max_wave_ns {
        return Err(CostModelError::InvalidTiming(
            "wall time must be positive and within limits",
        ));
    }
    if timing
        .device_elapsed_ns
        .is_some_and(|cost| cost == 0 || cost > timing.wall_total_ns)
    {
        return Err(CostModelError::InvalidTiming(
            "device elapsed interval outside wall interval",
        ));
    }
    let mut spans: Vec<_> = [
        timing.stages.prepare,
        timing.stages.device_wait,
        timing.stages.commit,
        timing.stages.restore,
        timing.stages.maintenance,
    ]
    .into_iter()
    .flatten()
    .collect();
    spans.sort_unstable_by_key(|span| (span.start_ns, span.end_ns));
    let mut previous_end = 0;
    for span in spans {
        if span.start_ns > span.end_ns
            || span.end_ns > timing.wall_total_ns
            || span.start_ns < previous_end
        {
            return Err(CostModelError::InvalidTiming(
                "diagnostic stages overlap or exceed wall interval",
            ));
        }
        previous_end = span.end_ns;
    }
    Ok(())
}

fn quantile(sorted: &[u64], q: f64) -> u64 {
    let rank = ((sorted.len() as f64 * q).ceil() as usize).max(1);
    sorted[rank - 1]
}

impl ObservedShapeCoverage {
    fn from_samples(samples: &[&CostSample], mode: &CostFeatureModel) -> Self {
        let first = &samples[0].shape;
        let mut coverage = Self {
            numeric: numeric::NumericSupport::from_samples(samples, mode),
            decode_context_ranges: first.decode_kv_tokens.iter().map(|&n| (n, n)).collect(),
            prefill_offset_ranges: first
                .prefill_chunks
                .iter()
                .map(|chunk| (chunk.offset, chunk.offset))
                .collect(),
        };
        for sample in &samples[1..] {
            for ((min, max), &context) in coverage
                .decode_context_ranges
                .iter_mut()
                .zip(&sample.shape.decode_kv_tokens)
            {
                *min = (*min).min(context);
                *max = (*max).max(context);
            }
            for ((min, max), chunk) in coverage
                .prefill_offset_ranges
                .iter_mut()
                .zip(&sample.shape.prefill_chunks)
            {
                *min = (*min).min(chunk.offset);
                *max = (*max).max(chunk.offset);
            }
        }
        coverage
    }

    fn contains(&self, shape: &WaveExecutionShape) -> bool {
        self.decode_context_ranges.len() == shape.decode_kv_tokens.len()
            && self.prefill_offset_ranges.len() == shape.prefill_chunks.len()
            && self
                .decode_context_ranges
                .iter()
                .zip(&shape.decode_kv_tokens)
                .all(|(&(min, max), &value)| min <= value && value <= max)
            && self
                .prefill_offset_ranges
                .iter()
                .zip(&shape.prefill_chunks)
                .all(|(&(min, max), chunk)| min <= chunk.offset && chunk.offset <= max)
    }
}
