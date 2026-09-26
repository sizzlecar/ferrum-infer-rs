//! Opt-in capture of original completed observations on the training worker.
//! A published profile is a bounded observed subset, never a complete-wave or
//! statistical guarantee. No predictor state is serialized as a measurement.
use super::audit::HostContentEvaluation;
use super::audit::{PreUpdatePrediction, TrainingAuditSnapshot, TrainingDisposition};
use super::host_stages::ExportEvidence;
use super::*;
use ferrum_scheduler::implementations::continuous::cost_profile::v2 as profile_v2;
use ferrum_scheduler::implementations::continuous::cost_profile::v3 as profile_v3;
use ferrum_scheduler::implementations::continuous::cost_profile::v4 as profile_v4;
use ferrum_scheduler::implementations::continuous::cost_profile::v5 as profile_v5;
use ferrum_scheduler::implementations::continuous::{cost_model as model, cost_profile as profile};
use ferrum_types::SloCostProfileExportConfig;
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::{
    io::Write,
    path::{Path, PathBuf},
};

mod cut;
mod files;
mod host_content;
mod retention;
pub(in crate::continuous_engine::inner) mod selected;
mod session;
pub(in crate::continuous_engine::inner) mod structured;
pub(in crate::continuous_engine::inner) mod structured_v2;
pub(in crate::continuous_engine) use cut::{CostProfileCutPaths, CostProfileCutReceipt};
use files::{ProducerIdentity, PublishedFile, StagedFile};
pub(super) use session::ExportSession;

#[derive(Debug, thiserror::Error)]
pub(in crate::continuous_engine::inner) enum ExportError {
    #[error("cost export IO: {0}")]
    Io(#[from] std::io::Error),
    #[error("cost export serialization: {0}")]
    Json(#[from] serde_json::Error),
    #[error("cost export source: {0}")]
    Source(&'static str),
    #[error("cost export clock: {0}")]
    Clock(&'static str),
    #[error("cost export configuration: {0}")]
    Config(String),
    #[error(
        "cost export preserved raw observations at {path}; profile was not published: {reason}"
    )]
    SourceOnly { path: PathBuf, reason: String },
}

#[derive(Debug, Clone, Copy, Serialize)]
pub(super) struct ExportClockReading {
    pub wall_unix_ns: u64,
    pub monotonic_ns: u64,
}

fn wall_now() -> Result<u64, ExportError> {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .ok()
        .and_then(|duration| u64::try_from(duration.as_nanos()).ok())
        .filter(|at| *at > 0)
        .ok_or(ExportError::Clock("wall time unavailable"))
}

impl ExportClockReading {
    /// Reading wall first makes anchored sample timestamps conservative: any
    /// acquisition delay makes the sample older, not artificially fresher.
    pub fn opening(clock: &dyn CostObservationClock) -> Result<Self, ExportError> {
        let wall_unix_ns = wall_now()?;
        let monotonic_ns = clock
            .now_ns()
            .ok_or(ExportError::Clock("monotonic time unavailable"))?;
        Ok(Self {
            wall_unix_ns,
            monotonic_ns,
        })
    }
    /// Closing reads monotonic first, giving an upper wall-offset endpoint.
    pub fn closing(clock: &dyn CostObservationClock) -> Result<Self, ExportError> {
        let monotonic_ns = clock
            .now_ns()
            .ok_or(ExportError::Clock("monotonic time unavailable"))?;
        Ok(Self {
            wall_unix_ns: wall_now()?,
            monotonic_ns,
        })
    }
}

#[derive(Clone)]
pub(super) struct ExportPlan {
    options: SloCostProfileExportConfig,
    fingerprint: model::ExecutionFingerprint,
    settings: model::CostModelSettings,
    opening: ExportClockReading,
}

impl ExportPlan {
    pub fn new(
        options: &SloCostProfileExportConfig,
        fingerprint: &model::ExecutionFingerprint,
        settings: &model::CostModelSettings,
        opening: ExportClockReading,
    ) -> Result<Self, ExportError> {
        options.validate().map_err(ExportError::Config)?;
        settings
            .validate()
            .map_err(|error| ExportError::Config(error.to_string()))?;
        // Include the transient owned profile-vector conversion at finish,
        // not only the steady retained observations. Row vectors move intact.
        let retained_bytes = options
            .max_samples
            .get()
            .checked_mul(
                std::mem::size_of::<Retained>()
                    + std::mem::size_of::<profile_v2::ProfileSampleV2>()
                    + std::mem::size_of::<HostStageEvidenceV1>()
                    + 2 * (std::mem::size_of::<
                        ferrum_interfaces::execution_cost::UnsettledStructuredWaveEvidenceV1,
                    >() + 2 * std::mem::size_of::<usize>())
                    + 2 * std::mem::size_of::<RetainedHostSample>()
                    + 2 * std::mem::size_of::<usize>(),
            )
            .and_then(|bytes| {
                options
                    .max_total_shape_rows
                    .get()
                    .checked_mul(
                        std::mem::size_of::<profile::ProfilePrefillShape>()
                            .max(std::mem::size_of::<CostRowNumericFeatures>())
                            .max(std::mem::size_of::<HostRowStaticCostFeaturesV2>())
                            .max(std::mem::size_of::<HostRowStageV1>()),
                    )
                    .and_then(|rows| bytes.checked_add(rows))
            });
        if retained_bytes.is_none_or(|bytes| bytes > 128 * 1024 * 1024) {
            return Err(ExportError::Source(
                "retained export allocation exceeds 128 MiB hard bound",
            ));
        }
        if opening.wall_unix_ns == 0 {
            return Err(ExportError::Clock("opening wall time missing"));
        }
        Ok(Self {
            options: options.clone(),
            fingerprint: fingerprint.clone(),
            settings: settings.clone(),
            opening,
        })
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub(in crate::continuous_engine) struct ExportCounts {
    pub received_host_content_evaluations: u64,
    pub raw_retained_host_content_evaluations: u64,
    /// Recorded samples in the distinct profile-v3 host-settled population.
    pub retained_host_content_samples: u64,
    pub received_entries: u64,
    pub raw_retained_entries: u64,
    pub received_host_stages: u64,
    pub raw_retained_host_stages: u64,
    pub host_stages_dropped_sample_limit: u64,
    pub host_stages_dropped_shape_row_limit: u64,
    pub host_stages_dropped_file_byte_limit: u64,
    pub received_observations: u64,
    pub trainer_not_recorded: u64,
    pub non_completed: u64,
    pub retained_samples: u64,
    /// Raw evidence includes completed observations rejected by the trainer.
    /// `retained_samples` retains its v1 meaning: Recorded profile samples.
    pub raw_retained_observations: u64,
    pub received_by_wave: [u64; audit::WAVE_COUNT],
    pub raw_retained_by_wave: [u64; audit::WAVE_COUNT],
    pub profile_retained_by_wave: [u64; audit::WAVE_COUNT],
    pub dropped_sample_limit: u64,
    pub dropped_shape_row_limit: u64,
    pub dropped_file_byte_limit: u64,
    pub counter_exhausted: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(in crate::continuous_engine) enum ExportStatus {
    Disabled,
    Pending,
    Active,
    Published,
    Failed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(in crate::continuous_engine) enum ExportFailureReason {
    Io,
    Json,
    Source,
    Clock,
    Config,
    SourceOnly,
}

#[derive(Debug, Clone, Serialize)]
pub(in crate::continuous_engine) struct ExportAuditSnapshot {
    pub status: ExportStatus,
    pub counts: ExportCounts,
    pub failure: Option<ExportFailureReason>,
    /// Training continued after capture became unavailable; these observations
    /// were not cloned/retained and are not in counts.received_observations.
    pub unavailable_observations: u64,
    pub unavailable_host_stages: u64,
    pub counter_exhausted: bool,
}

#[derive(Debug, Clone, Serialize)]
struct SinkEvidence {
    #[serde(flatten)]
    counters: CostSampleStats,
    wave_order: [&'static str; audit::WAVE_COUNT],
    rejection_order: [&'static str; CostCallRejection::COUNT],
    /// Stable names accompany counts; absence of losses does not attest that
    /// every executor path emitted an observation.
    rejected_by_reason: [(&'static str, u64); CostCallRejection::COUNT],
}

impl SinkEvidence {
    fn from_stats(stats: &CostSampleStats) -> Self {
        use CostCallRejection::*;
        let reasons = [
            ("clock", Clock),
            ("id_exhausted", IdExhausted),
            ("invalid_participants", InvalidParticipants),
            ("recorder_capacity", RecorderCapacity),
            ("identity_unknown", IdentityUnknown),
            ("identity_schema", IdentitySchema),
            ("output_policy_unknown", OutputPolicyUnknown),
            ("calibration_preparation", CalibrationPreparation),
            ("unavailable", Unavailable),
            ("no_physical_wave", NoPhysicalWave),
            ("composite", Composite),
            ("executor_incomplete", ExecutorIncomplete),
            ("executor_failed", ExecutorFailed),
            ("actual_evidence_unknown", ActualEvidenceUnknown),
            ("host_missing", HostMissing),
            ("host_duplicate", HostDuplicate),
            ("host_unexpected", HostUnexpected),
            ("frontier_mismatch", FrontierMismatch),
            ("work_mismatch", WorkMismatch),
            ("host_cancelled", HostCancelled),
            ("host_failed", HostFailed),
            ("invalid_wall", InvalidWall),
            ("abandoned", Abandoned),
        ];
        Self {
            counters: stats.clone(),
            wave_order: audit::WAVE_NAMES,
            rejection_order: reasons.map(|(name, _)| name),
            rejected_by_reason: reasons
                .map(|(name, reason)| (name, stats.rejected[reason.index()])),
        }
    }
}

const RAW_SCHEMA_VERSION: u32 = 4;
const HOST_CONTENT_RAW_SCHEMA_VERSION: u32 = 5;
const ROW_MULTISET_RAW_SCHEMA_VERSION: u32 = 6;
const COVERAGE: &str = "bounded FIFO evidence entries; original completed cost observations and auxiliary host_stages_v1 have separate denominators; accepted_ordinal identifies a FIFO entry, source_record counts cost observations only; only Recorded observations enter profile v2; host stages never train or refresh age and host settlement is not client visibility; source, queue, trainer and export denominators are independent; pre-update predictions query actual shapes retrospectively, not pre-execution candidates or lookahead benefit; uninstrumented waves remain unknown; live counters are not an atomic cut and file hashes do not attest complete execution coverage";

#[derive(Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum RawRecord<'a> {
    Header {
        schema_version: u32,
        fingerprint: &'a profile::ProfileFingerprint,
        settings: &'a profile_v2::ProfileModelSettingsV2,
        producer: &'a ProducerIdentity,
        opening: ExportClockReading,
        declared_clock_max_error_ns: u64,
        coverage: &'static str,
    },
    Observation {
        accepted_ordinal: u64,
        observed_at_monotonic_ns: u64,
        fingerprint: &'a profile::ProfileFingerprint,
        training: TrainingDisposition,
        pre_update_prediction: PreUpdatePrediction,
        sample: &'a profile_v2::ProfileSampleV2,
    },
    HostStagesV1 {
        accepted_ordinal: u64,
        source_record: Option<u64>,
        legacy_rejection: Option<CostCallRejection>,
        evidence: ExportEvidence<'a>,
    },
    HostContentTrainingV1 {
        accepted_ordinal: u64,
        host_source_record: Option<u64>,
        evaluation: HostContentEvaluation,
        sample: Option<&'a profile_v3::ProfileSampleV3>,
    },
    HostRowMultisetTrainingV2 {
        accepted_ordinal: u64,
        host_source_record: Option<u64>,
        evaluation: HostContentEvaluation,
        sample: Option<&'a profile_v4::ProfileSampleV4>,
    },
    Summary {
        closing: ExportClockReading,
        counts: &'a ExportCounts,
        sink: &'a SinkEvidence,
        training_rejected: u64,
        publish_rejected: u64,
        training: &'a TrainingAuditSnapshot,
        coverage: &'static str,
    },
}

struct Retained {
    accepted_ordinal: u64,
    observation: Option<RetainedObservation>,
    stages: Option<Arc<HostStageEvidenceV1>>,
    legacy_rejection: Option<CostCallRejection>,
    host_content: Option<RetainedHostContent>,
}
struct RetainedHostContent {
    host_source_record: Option<u64>,
    evaluation: HostContentEvaluation,
    row_multiset: bool,
    sample: Option<RetainedHostSample>,
}
#[derive(Serialize)]
#[serde(untagged)]
enum RetainedHostSample {
    V3(profile_v3::ProfileSampleV3),
    V4(profile_v4::ProfileSampleV4),
}
impl RetainedHostSample {
    fn measured_unix_ns(&self) -> u64 {
        match self {
            Self::V3(sample) => sample.measured_unix_ns,
            Self::V4(sample) => sample.measured_unix_ns,
        }
    }
}
struct RetainedObservation {
    observed_at_monotonic_ns: u64,
    fingerprint: profile::ProfileFingerprint,
    training: TrainingDisposition,
    pre_update_prediction: PreUpdatePrediction,
    sample: profile_v2::ProfileSampleV2,
}

/// Owned exclusively by the existing CPU training worker. Bounds cover the
/// retained sample structs, allocated row capacities and both output files.
pub(super) struct ProfileExporter {
    plan: ExportPlan,
    producer: ProducerIdentity,
    profile_path: PathBuf,
    raw_path: PathBuf,
    samples: Vec<Retained>,
    rows: usize,
    body_bytes: u64,
    max_body_bytes: u64,
    pub counts: ExportCounts,
    last_accepted_ordinal: u64,
}

impl ProfileExporter {
    pub fn start(plan: ExportPlan) -> Result<Self, ExportError> {
        Self::with_producer(plan, ProducerIdentity::current()?)
    }

    fn with_producer(plan: ExportPlan, producer: ProducerIdentity) -> Result<Self, ExportError> {
        let profile_path = files::destination(&plan.options.path)?;
        let raw_path = files::destination(&plan.options.observations_path)?;
        if profile_path == raw_path {
            return Err(ExportError::Source("profile and source destinations alias"));
        }
        let max = u64::MAX;
        let worst_counts = ExportCounts {
            received_host_content_evaluations: max,
            raw_retained_host_content_evaluations: max,
            retained_host_content_samples: max,
            received_entries: max,
            raw_retained_entries: max,
            received_host_stages: max,
            raw_retained_host_stages: max,
            host_stages_dropped_sample_limit: max,
            host_stages_dropped_shape_row_limit: max,
            host_stages_dropped_file_byte_limit: max,
            received_observations: max,
            trainer_not_recorded: max,
            non_completed: max,
            retained_samples: max,
            raw_retained_observations: max,
            received_by_wave: [max; audit::WAVE_COUNT],
            raw_retained_by_wave: [max; audit::WAVE_COUNT],
            profile_retained_by_wave: [max; audit::WAVE_COUNT],
            dropped_sample_limit: max,
            dropped_shape_row_limit: max,
            dropped_file_byte_limit: max,
            // JSON false is longer than true; reserve the largest footer.
            counter_exhausted: false,
        };
        let worst_sink = SinkEvidence::from_stats(&BoundedCostSampleSink::maximum_stats());
        let worst_training = TrainingAuditSnapshot::maximum_serialized_counts();
        let fingerprint = profile::ProfileFingerprint::from(&plan.fingerprint);
        let settings = profile_v2::ProfileModelSettingsV2::from(&plan.settings);
        let header = RawRecord::Header {
            schema_version: Self::raw_version(&plan),
            fingerprint: &fingerprint,
            settings: &settings,
            producer: &producer,
            opening: plan.opening,
            declared_clock_max_error_ns: plan
                .options
                .declared_clock_max_error_ns
                .expect("validated clock declaration"),
            coverage: host_content::coverage(&plan),
        };
        let raw_overhead = json_size(&header)?
            .checked_add(2)
            .and_then(|size| {
                size.checked_add(
                    json_size(&RawRecord::Summary {
                        closing: ExportClockReading {
                            wall_unix_ns: max,
                            monotonic_ns: max,
                        },
                        counts: &worst_counts,
                        sink: &worst_sink,
                        training_rejected: max,
                        publish_rejected: max,
                        training: &worst_training,
                        coverage: host_content::coverage(&plan),
                    })
                    .ok()?,
                )
            })
            .ok_or(ExportError::Source("source overhead size overflow"))?;
        let profile_overhead = host_content::profile_size(&plan, &producer)?;
        let max_body_bytes = (plan.options.max_file_bytes.get() as u64)
            .checked_sub(raw_overhead.max(profile_overhead))
            .ok_or(ExportError::Source(
                "file limit cannot hold bounded provenance",
            ))?;
        let mut samples = Vec::new();
        samples
            .try_reserve_exact(plan.options.max_samples.get())
            .map_err(|_| ExportError::Source("sample storage allocation failed"))?;
        Ok(Self {
            plan,
            producer,
            profile_path,
            raw_path,
            samples,
            rows: 0,
            body_bytes: 0,
            max_body_bytes,
            counts: ExportCounts::default(),
            last_accepted_ordinal: 0,
        })
    }

    fn profile_file(
        plan: &ExportPlan,
        producer: &ProducerIdentity,
        source_hash: [u8; 32],
        generated: u64,
        samples: Vec<profile_v2::ProfileSampleV2>,
    ) -> profile_v2::CostProfileFileV2 {
        profile_v2::CostProfileFileV2 {
            schema_version: profile_v2::COST_PROFILE_SCHEMA_VERSION_V2,
            fingerprint: profile::ProfileFingerprint::from(&plan.fingerprint),
            settings: profile_v2::ProfileModelSettingsV2::from(&plan.settings),
            generated_unix_ns: generated,
            source_clock_max_error_ns: plan.options.declared_clock_max_error_ns,
            source: profile::ProfileSource {
                generator: "ferrum-engine successful observation export".into(),
                generator_revision: format!("binary-sha256:{}", producer.executable_sha256),
                measurement_protocol: COVERAGE.into(),
                observation_artifact_sha256: source_hash,
            },
            samples,
        }
    }

    fn host_content_mode(plan: &ExportPlan) -> bool {
        matches!(
            plan.settings.feature_model,
            model::CostFeatureModel::EmpiricalHostContentV1 { .. }
                | model::CostFeatureModel::EmpiricalRowMultisetV2 { .. }
                | model::CostFeatureModel::EmpiricalPromptRangeV3 { .. }
        )
    }
    fn prompt_range_mode(plan: &ExportPlan) -> bool {
        matches!(
            plan.settings.feature_model,
            model::CostFeatureModel::EmpiricalPromptRangeV3 { .. }
        )
    }
    fn row_multiset_mode(plan: &ExportPlan) -> bool {
        matches!(
            plan.settings.feature_model,
            model::CostFeatureModel::EmpiricalRowMultisetV2 { .. }
                | model::CostFeatureModel::EmpiricalPromptRangeV3 { .. }
        )
    }
    fn raw_version(plan: &ExportPlan) -> u32 {
        if Self::row_multiset_mode(plan) {
            ROW_MULTISET_RAW_SCHEMA_VERSION
        } else if Self::host_content_mode(plan) {
            HOST_CONTENT_RAW_SCHEMA_VERSION
        } else {
            RAW_SCHEMA_VERSION
        }
    }

    #[cfg(test)]
    pub fn record(
        &mut self,
        observation: &model::WaveCostObservation,
        training: TrainingDisposition,
        pre_update_prediction: PreUpdatePrediction,
    ) -> Result<(), ExportError> {
        // Unit fixtures supply a sequential accepted stream; production passes
        // the queue's real ordinal through record_numbered below.
        let ordinal = self
            .last_accepted_ordinal
            .checked_add(1)
            .ok_or(ExportError::Source("observation ordinal exhausted"))?;
        self.record_numbered(ordinal, observation, training, pre_update_prediction)
    }

    #[cfg(test)]
    pub fn record_numbered(
        &mut self,
        accepted_ordinal: u64,
        observation: &model::WaveCostObservation,
        training: TrainingDisposition,
        pre_update_prediction: PreUpdatePrediction,
    ) -> Result<(), ExportError> {
        self.record_entry(
            accepted_ordinal,
            Some((observation, training, pre_update_prediction)),
            None,
            None,
        )
    }

    fn validate_closing(&self, closing: ExportClockReading) -> Result<(), ExportError> {
        let elapsed = closing
            .monotonic_ns
            .checked_sub(self.plan.opening.monotonic_ns)
            .ok_or(ExportError::Clock(
                "closing monotonic clock moved backwards",
            ))?;
        let wall_elapsed = closing
            .wall_unix_ns
            .checked_sub(self.plan.opening.wall_unix_ns)
            .ok_or(ExportError::Clock("closing wall clock moved backwards"))?;
        let error = self
            .plan
            .options
            .declared_clock_max_error_ns
            .expect("validated clock declaration");
        if u128::from(wall_elapsed) + 2 * u128::from(error) < u128::from(elapsed)
            || self
                .samples
                .iter()
                .filter_map(|entry| entry.observation.as_ref())
                .any(|sample| sample.sample.measured_unix_ns > closing.wall_unix_ns)
            || self
                .samples
                .iter()
                .filter_map(|entry| entry.host_content.as_ref())
                .filter_map(|row| row.sample.as_ref())
                .any(|sample| sample.measured_unix_ns() > closing.wall_unix_ns)
        {
            return Err(ExportError::Clock(
                "wall/monotonic endpoints contradict declared clock accuracy",
            ));
        }
        Ok(())
    }

    pub fn finish(
        self,
        closing: ExportClockReading,
        stats: CostSampleStats,
        training: TrainingAuditSnapshot,
    ) -> Result<ExportReceipt, ExportError> {
        self.validate_closing(closing)?;
        let error = self
            .plan
            .options
            .declared_clock_max_error_ns
            .expect("validated clock declaration");
        let mut raw = StagedFile::create(
            &self.raw_path,
            self.plan.options.max_file_bytes.get() as u64,
        )?;
        let fingerprint = profile::ProfileFingerprint::from(&self.plan.fingerprint);
        let settings = profile_v2::ProfileModelSettingsV2::from(&self.plan.settings);
        raw.json_line(&RawRecord::Header {
            schema_version: Self::raw_version(&self.plan),
            fingerprint: &fingerprint,
            settings: &settings,
            producer: &self.producer,
            opening: self.plan.opening,
            declared_clock_max_error_ns: error,
            coverage: host_content::coverage(&self.plan),
        })?;
        for entry in &self.samples {
            if let Some(record) = entry.observation_record() {
                raw.json_line(&record)?;
            }
            if let Some(record) = entry.stages_record() {
                raw.json_line(&record)?;
            }
            if let Some(record) = entry.host_content_record() {
                raw.json_line(&record)?;
            }
        }
        raw.json_line(&RawRecord::Summary {
            closing,
            counts: &self.counts,
            sink: &SinkEvidence::from_stats(&stats),
            training_rejected: training.training_rejected(),
            publish_rejected: training.publish_rejected(),
            training: &training,
            coverage: host_content::coverage(&self.plan),
        })?;
        // Publish source bytes first. If the profile cannot subsequently be
        // published, this valid raw artifact remains an explicit diagnostic;
        // no existing destination is overwritten or removed during recovery.
        let source = raw.publish()?;
        let source_hash = source.digest;
        if self.profile_sample_count() == 0 {
            return Err(ExportError::SourceOnly {
                path: source.path,
                reason: "no successful observations retained".into(),
            });
        }
        let result = self.write_profile(
            &self.profile_path,
            source_hash,
            closing.wall_unix_ns,
            host_content::coverage(&self.plan),
        );
        let profile = result.map_err(|error: ExportError| ExportError::SourceOnly {
            path: source.path.clone(),
            reason: error.to_string(),
        })?;
        Ok(ExportReceipt {
            profile,
            source,
            coverage: host_content::coverage(&self.plan),
            counts: self.counts,
        })
    }
}

#[derive(Debug, Clone, Serialize)]
pub(super) struct ExportReceipt {
    pub profile: PublishedFile,
    pub source: PublishedFile,
    pub counts: ExportCounts,
    pub coverage: &'static str,
}

fn json_size(value: &impl Serialize) -> Result<u64, ExportError> {
    #[derive(Default)]
    struct Counter(u64);
    impl Write for Counter {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            self.0 = self
                .0
                .checked_add(bytes.len() as u64)
                .ok_or_else(|| std::io::Error::other("JSON byte count overflow"))?;
            Ok(bytes.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }
    let mut count = Counter::default();
    serde_json::to_writer(&mut count, value)?;
    Ok(count.0)
}

fn sample_from_observation(
    observation: &model::WaveCostObservation,
    source_record: u64,
    measured_unix_ns: u64,
) -> Result<profile_v2::ProfileSampleV2, ExportError> {
    let shape = &observation.actual_shape;
    let timing = &observation.timing;
    let span = |span: Option<model::MeasuredSpan>| {
        span.map(|span| profile::ProfileMeasuredSpan {
            start_ns: span.start_ns,
            end_ns: span.end_ns,
        })
    };
    Ok(profile_v2::ProfileSampleV2 {
        source_record,
        measured_unix_ns,
        shape: profile_v2::ProfileWaveShapeV2::from(shape),
        boundary: observation.boundary.try_into().map_err(|_| {
            ExportError::Source("new host boundary cannot enter legacy observation record")
        })?,
        outcome: profile::ProfileObservationOutcome::Completed {},
        timing: profile::ProfileWaveTiming {
            wall_total_ns: timing.wall_total_ns,
            device_elapsed_ns: timing.device_elapsed_ns,
            stages: profile::ProfileStageTimings {
                prepare: span(timing.stages.prepare),
                device_wait: span(timing.stages.device_wait),
                commit: span(timing.stages.commit),
                restore: span(timing.stages.restore),
                maintenance: span(timing.stages.maintenance),
            },
        },
    })
}

#[cfg(test)]
mod tests;
