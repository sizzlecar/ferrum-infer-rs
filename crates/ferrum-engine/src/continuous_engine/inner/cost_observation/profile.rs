//! Startup-only profile import and one clock-preserving calibration adapter.
//! Neither prediction nor a sample producer performs file IO or training.
use super::audit::TrainingErrorReason;
use super::*;
use ferrum_scheduler::implementations::continuous::{
    cost_model as model, cost_profile as file,
    slo_planner::{
        PlanningCost, PlanningCostEvidence, PlanningCostEvidenceRequirement, PlanningCostModel,
    },
};
use ferrum_types::{
    FerrumError, SloCostModelConfig, SloCostObservationConfig, SloCostProfileImportConfig,
    SloCostProfileReceipt,
};
use std::path::Path;
mod selected;
mod structured;

pub(super) fn model_settings(config: &SloCostModelConfig) -> model::CostModelSettings {
    model::CostModelSettings {
        feature_model: config.feature_model.clone(),
        max_buckets: config.max_buckets,
        max_samples_per_bucket: config.max_samples_per_bucket,
        min_samples: config.min_samples,
        max_retained_samples: config.max_retained_samples,
        max_retained_shape_rows: config.max_retained_shape_rows,
        residual_quantile: config.residual_quantile,
        drift_margin_ns: config.drift_margin_ns,
        max_wave_ns: config.max_wave_ns,
        max_sample_age_ns: config.max_sample_age_ns,
        context_bucket_tokens: config.context_bucket_tokens,
        prefill_offset_bucket_tokens: config.prefill_offset_bucket_tokens,
        shape_limits: model::CostShapeLimits {
            max_rows: config.shape_limits.max_rows,
            max_context_tokens: config.shape_limits.max_context_tokens,
            max_prefill_tokens_per_wave: config.shape_limits.max_prefill_tokens_per_wave,
            max_state_bytes: config.shape_limits.max_state_bytes,
            max_maintenance_units: config.shape_limits.max_maintenance_units,
        },
    }
}

fn load_limits(config: &SloCostProfileImportConfig) -> file::CostProfileLoadLimits {
    file::CostProfileLoadLimits {
        max_file_bytes: config.max_file_bytes,
        max_samples: config.max_samples,
        max_total_shape_rows: config.max_total_shape_rows,
        max_source_field_bytes: config.max_source_field_bytes,
        max_profile_age_ns: config.max_profile_age_ns,
        max_clock_error_ns: config.max_clock_error_ns,
    }
}

/// Read monotonic first: file IO and scheduling delays after this pair continue
/// aging the imported evidence. A wall reading never supplies its own accuracy.
pub(super) fn read_load_clock(
    clock: &dyn CostObservationClock,
    policy: &SloCostProfileImportConfig,
) -> Result<file::ProfileLoadClock, FerrumError> {
    let monotonic_now_ns = clock
        .now_ns()
        .ok_or_else(|| FerrumError::config("cost profile monotonic clock unavailable"))?;
    let wall_unix_ns = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .ok()
        .and_then(|duration| u64::try_from(duration.as_nanos()).ok());
    Ok(file::ProfileLoadClock {
        monotonic_now_ns,
        wall_unix_ns,
        wall_max_error_ns: policy.declared_local_clock_max_error_ns,
    })
}

enum Calibration {
    Live(model::CostModelTrainer),
    Imported(file::LoadedCostProfile),
}

pub(super) struct CostTrainer {
    calibration: Calibration,
    fingerprint: model::ExecutionFingerprint,
}

pub(super) struct TrainingSeed {
    pub trainer: Option<CostTrainer>,
    pub snapshot: Option<Arc<EngineCostSnapshot>>,
    pub receipt: Option<SloCostProfileReceipt>,
}

pub(super) fn load_seed(
    identity: &ExecutorCostIdentityAvailability,
    config: &SloCostObservationConfig,
    path: Option<&Path>,
    clock: Option<file::ProfileLoadClock>,
) -> Result<TrainingSeed, FerrumError> {
    config.validate().map_err(FerrumError::config)?;
    let fingerprint = match identity {
        ExecutorCostIdentityAvailability::Known(identity)
            if identity.schema_version == EXECUTOR_COST_IDENTITY_SCHEMA =>
        {
            model::ExecutionFingerprint {
                model_weights: identity.model_weights,
                numerical_policy: identity.numerical_policy,
                device_runtime: identity.device_runtime,
                execution_config: identity.execution_config,
            }
        }
        _ if path.is_some() => {
            return Err(FerrumError::config(
                "configured cost profile requires a known current executor fingerprint",
            ));
        }
        _ => {
            return Ok(TrainingSeed {
                trainer: None,
                snapshot: None,
                receipt: None,
            });
        }
    };
    let settings = model_settings(&config.model);
    settings.validate().map_err(profile_error)?;
    if config.predictor.is_selected() {
        return selected::load_seed(fingerprint, config, path, clock);
    }
    if config.predictor == ferrum_types::SloCostPredictor::StructuredWholeWaveV1 {
        return structured::load_seed(fingerprint, config, path, clock);
    }
    let Some(path) = path else {
        return Ok(TrainingSeed {
            trainer: Some(CostTrainer {
                calibration: Calibration::Live(
                    model::CostModelTrainer::new(fingerprint.clone(), settings)
                        .map_err(profile_error)?,
                ),
                fingerprint,
            }),
            snapshot: None,
            receipt: None,
        });
    };
    let clock = clock.ok_or_else(|| FerrumError::config("cost profile load clock missing"))?;
    let declared_error = config
        .profile_import
        .declared_local_clock_max_error_ns
        .ok_or_else(|| {
            FerrumError::config("cost profile requires declared local wall-clock accuracy")
        })?;
    if clock.wall_max_error_ns != Some(declared_error) {
        return Err(FerrumError::config(
            "cost profile clock differs from declared policy",
        ));
    }
    let profile = file::load_cost_profile(
        path,
        &fingerprint,
        &settings,
        &load_limits(&config.profile_import),
        clock,
    )
    .map_err(profile_error)?;
    let receipt = import_receipt(&profile, declared_error)?;
    let snapshot = Arc::new(EngineCostSnapshot {
        inner: Snapshot::Imported(profile.snapshot.clone()),
        fingerprint: fingerprint.clone(),
    });
    Ok(TrainingSeed {
        trainer: Some(CostTrainer {
            calibration: Calibration::Imported(profile),
            fingerprint,
        }),
        snapshot: Some(snapshot),
        receipt: Some(receipt),
    })
}

fn profile_error(error: impl std::fmt::Display) -> FerrumError {
    FerrumError::config(format!("cost calibration: {error}"))
}

fn import_receipt(
    profile: &file::LoadedCostProfile,
    declared_local_clock_max_error_ns: u64,
) -> Result<SloCostProfileReceipt, FerrumError> {
    let p = &profile.provenance;
    let skipped_samples = p
        .counts
        .skipped_samples
        .iter()
        .map(|(reason, count)| {
            let name = match reason {
                file::ProfileSkippedObservation::NotSubmitted => "not_submitted",
                file::ProfileSkippedObservation::Deferred => "deferred",
                file::ProfileSkippedObservation::FailedAfterSubmit => "failed_after_submit",
                file::ProfileSkippedObservation::UnisolatedPartial => "unisolated_partial",
                file::ProfileSkippedObservation::MissingDeviceTiming => "missing_device_timing",
            };
            (name.to_owned(), *count)
        })
        .collect();
    Ok(SloCostProfileReceipt {
        selected_whole_wave: None,
        structured_whole_wave: None,
        schema_version: p.schema_version,
        path: p
            .loaded_from
            .clone()
            .ok_or_else(|| FerrumError::internal("file import has no source path"))?,
        file_sha256: p.file_sha256.clone(),
        file_bytes: p.file_bytes,
        generated_unix_ns: p.generated_unix_ns,
        loaded_unix_ns: p.loaded_unix_ns,
        conservative_clock_error_ns: p.conservative_clock_error_ns,
        declared_local_clock_max_error_ns,
        oldest_imported_age_ns: p.oldest_imported_age_ns,
        newest_imported_age_ns: p.newest_imported_age_ns,
        offered_samples: p.counts.offered_samples,
        recorded_samples: p.counts.recorded_samples,
        stale_samples: p.counts.stale_samples,
        skipped_samples,
        model_version: profile.snapshot.model_version(),
        bucket_count: profile.snapshot.bucket_count(),
        source_generator: p.source.generator.clone(),
        source_generator_revision: p.source.generator_revision.clone(),
        source_measurement_protocol: p.source.measurement_protocol.clone(),
        source_observation_artifact_sha256: p.source.observation_artifact_sha256,
    })
}

impl CostTrainer {
    pub fn fingerprint(&self) -> &model::ExecutionFingerprint {
        &self.fingerprint
    }

    pub fn observe(
        &mut self,
        observation: model::WaveCostObservation,
    ) -> Result<model::ObservationDisposition, TrainingErrorReason> {
        let receipt = observation.observed_at_ns;
        match &mut self.calibration {
            Calibration::Live(trainer) => trainer.observe(observation).map_err(Into::into),
            Calibration::Imported(profile) => profile
                .observe_live(observation, receipt)
                .map_err(Into::into),
        }
    }

    pub fn publish(
        &mut self,
        receipt_ns: u64,
    ) -> Result<Arc<EngineCostSnapshot>, TrainingErrorReason> {
        let inner = match &mut self.calibration {
            Calibration::Live(trainer) => Snapshot::Live(
                trainer
                    .publish(receipt_ns)
                    .map_err(TrainingErrorReason::from)?,
            ),
            Calibration::Imported(profile) => Snapshot::Imported(
                profile
                    .publish(receipt_ns)
                    .map_err(TrainingErrorReason::from)?,
            ),
        };
        Ok(Arc::new(EngineCostSnapshot {
            inner,
            fingerprint: self.fingerprint.clone(),
        }))
    }
}

enum Snapshot {
    Selected(selected::SelectedSnapshot),
    Structured(file::structured_v9::ImportedStructuredModelV1),
    Live(Arc<model::CostModelSnapshot>),
    Imported(file::ImportedCostSnapshot),
}

/// An immutable snapshot with an explicit adapter from the engine's local
/// observation clock. Imported epochs never escape to the scheduler caller.
pub(in crate::continuous_engine) struct EngineCostSnapshot {
    inner: Snapshot,
    fingerprint: model::ExecutionFingerprint,
}

impl EngineCostSnapshot {
    pub(super) fn feedback_enabled(&self) -> bool {
        matches!(&self.inner, Snapshot::Selected(s) if s.feedback.is_some())
    }
    pub(super) fn selected_import(
        &self,
    ) -> Option<&Arc<file::statistical_v6::ImportedWholeWaveModelV1>> {
        match &self.inner {
            Snapshot::Selected(s) => Some(&s.model),
            _ => None,
        }
    }
    pub(super) fn with_feedback(
        &self,
        feedback: Arc<super::selected_feedback::View>,
    ) -> Option<Arc<Self>> {
        let Snapshot::Selected(s) = &self.inner else {
            return None;
        };
        Some(Arc::new(Self {
            inner: Snapshot::Selected(selected::SelectedSnapshot {
                model: s.model.clone(),
                feedback: Some(feedback),
            }),
            fingerprint: self.fingerprint.clone(),
        }))
    }
    pub(super) fn selected_family_signature<'a>(
        &self,
        evidence: &'a ferrum_interfaces::execution_cost::StatisticalWaveEvidenceV1,
    ) -> Result<&'a [u8; 32], model::statistical::model::ModelUnknown> {
        match &self.inner {
            Snapshot::Selected(snapshot) => snapshot.family_signature(evidence),
            _ => Err(model::statistical::model::ModelUnknown::Evidence(
                ferrum_interfaces::execution_cost::StatisticalEvidenceUnknown::MissingProducer,
            )),
        }
    }
    pub(super) fn feedback_margin(&self, family: &[u8; 32]) -> u64 {
        match &self.inner {
            Snapshot::Selected(s) => s.feedback.as_ref().map_or(0, |v| v.margin(family)),
            _ => 0,
        }
    }
    pub(super) fn current(&self) -> bool {
        match &self.inner {
            Snapshot::Selected(s) => s.feedback.as_ref().is_none_or(|v| v.current()),
            _ => true,
        }
    }
    /// Actual calibration/heldout lookup. It consumes complete canonical and
    /// selected evidence together; legacy snapshots cannot claim this protocol.
    pub fn predict_selected_wave(
        &self,
        exact: &ferrum_interfaces::execution_cost::CanonicalWaveCostShape,
        evidence: &ferrum_interfaces::execution_cost::StatisticalWaveEvidenceV1,
        local_now_ns: u64,
    ) -> Result<
        model::statistical::model::WholeWavePredictionV1,
        model::statistical::model::ModelUnknown,
    > {
        match &self.inner {
            Snapshot::Selected(snapshot) => {
                let value =
                    snapshot
                        .model
                        .predict(&self.fingerprint, exact, evidence, local_now_ns)?;
                snapshot.apply(value, snapshot.family_signature(evidence)?)
            }
            _ => Err(model::statistical::model::ModelUnknown::Evidence(
                ferrum_interfaces::execution_cost::StatisticalEvidenceUnknown::MissingProducer,
            )),
        }
    }

    /// Retrospective calibration diagnostics only. The regular planner path
    /// remains unchanged; the family is returned by the same model lookup.
    pub fn predict_selected_wave_identified(
        &self,
        exact: &ferrum_interfaces::execution_cost::CanonicalWaveCostShape,
        evidence: &ferrum_interfaces::execution_cost::StatisticalWaveEvidenceV1,
        local_now_ns: u64,
    ) -> model::statistical::model::IdentifiedPredictionV1 {
        match &self.inner {
            Snapshot::Selected(snapshot) => {
                let mut result = snapshot.model.predict_identified(
                    &self.fingerprint,
                    exact,
                    evidence,
                    local_now_ns,
                );
                result.prediction = result
                    .prediction
                    .and_then(|value| snapshot.apply(value, snapshot.family_signature(evidence)?));
                result
            }
            _ => model::statistical::model::IdentifiedPredictionV1 {
                query_identity: None,
                prediction: Err(model::statistical::model::ModelUnknown::Evidence(
                    ferrum_interfaces::execution_cost::StatisticalEvidenceUnknown::MissingProducer,
                )),
            },
        }
    }

    pub fn planning_boundary(&self) -> model::CostBoundary {
        match &self.inner {
            Snapshot::Selected(_) | Snapshot::Structured(_) => {
                model::CostBoundary::PreparationToHostSettledV1
            }
            Snapshot::Live(snapshot) => snapshot.planning_boundary(),
            Snapshot::Imported(snapshot) => snapshot.planning_boundary(),
        }
    }
    pub fn model_version(&self) -> u64 {
        match &self.inner {
            Snapshot::Selected(snapshot) => snapshot.feedback.as_ref().map_or(1, |v| v.epoch),
            Snapshot::Structured(_) => 1,
            Snapshot::Live(snapshot) => snapshot.model_version(),
            Snapshot::Imported(snapshot) => snapshot.model_version(),
        }
    }
    pub fn bucket_count(&self) -> usize {
        match &self.inner {
            Snapshot::Selected(snapshot) => snapshot.model.segment_count(),
            Snapshot::Structured(_) => 1,
            Snapshot::Live(snapshot) => snapshot.bucket_count(),
            Snapshot::Imported(snapshot) => snapshot.bucket_count(),
        }
    }
    pub fn fingerprint(&self) -> &model::ExecutionFingerprint {
        &self.fingerprint
    }

    pub fn predict(
        &self,
        fingerprint: &model::ExecutionFingerprint,
        shape: &model::WaveExecutionShape,
        boundary: model::CostBoundary,
        local_now_ns: u64,
    ) -> model::CostPrediction {
        if fingerprint != self.fingerprint() {
            return model::CostPrediction::Unknown(model::CostUnknownReason::FingerprintMismatch);
        }
        match &self.inner {
            Snapshot::Selected(_) | Snapshot::Structured(_) => {
                model::CostPrediction::Unknown(model::CostUnknownReason::NumericFeaturesMissing)
            }
            Snapshot::Live(snapshot) => {
                snapshot.predict(fingerprint, shape, boundary, local_now_ns)
            }
            Snapshot::Imported(snapshot) => {
                snapshot.predict(fingerprint, shape, boundary, local_now_ns)
            }
        }
    }
}

impl PlanningCostModel for EngineCostSnapshot {
    fn evidence_requirement(&self) -> PlanningCostEvidenceRequirement {
        match &self.inner {
            Snapshot::Selected(_) => PlanningCostEvidenceRequirement::Selected,
            Snapshot::Structured(_) => PlanningCostEvidenceRequirement::Structured,
            _ => PlanningCostEvidenceRequirement::None,
        }
    }
    fn requires_statistical_evidence(&self) -> bool {
        self.evidence_requirement() != PlanningCostEvidenceRequirement::None
    }
    fn predict_with_evidence(
        &self,
        fingerprint: &model::ExecutionFingerprint,
        shape: &model::WaveExecutionShape,
        evidence: Option<&PlanningCostEvidence>,
        now_ns: u64,
    ) -> Option<PlanningCost> {
        match &self.inner {
            Snapshot::Selected(snapshot) => selected::predict(
                snapshot,
                fingerprint,
                shape,
                evidence,
                now_ns,
                self.model_version(),
            ),
            Snapshot::Structured(snapshot) => structured::predict(
                snapshot,
                fingerprint,
                shape,
                evidence,
                now_ns,
                self.model_version(),
            ),
            _ => <Self as PlanningCostModel>::predict(self, fingerprint, shape, now_ns),
        }
    }

    fn supports_empirical_host_content(&self) -> bool {
        self.planning_boundary() == model::CostBoundary::PreparationToHostSettledV1
    }
    fn model_version(&self) -> u64 {
        self.model_version()
    }
    fn predict(
        &self,
        fingerprint: &model::ExecutionFingerprint,
        shape: &model::WaveExecutionShape,
        now_ns: u64,
    ) -> Option<PlanningCost> {
        match self.predict(fingerprint, shape, self.planning_boundary(), now_ns) {
            model::CostPrediction::Known(value) => Some(PlanningCost {
                typical_ns: value.typical_ns,
                planning_ns: value.planning_ns,
                model_version: value.model_version,
                valid_for_ns: value.valid_for_ns,
            }),
            model::CostPrediction::Unknown(reason) => {
                tracing::trace!(
                    ?reason,
                    kind = ?shape.kind,
                    decode_kv_tokens = ?shape.decode_kv_tokens,
                    prefill_chunks = ?shape.prefill_chunks,
                    "SLO candidate has no empirical cost prediction"
                );
                None
            }
        }
    }
}

#[cfg(test)]
mod tests;
