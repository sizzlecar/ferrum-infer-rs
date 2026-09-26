//! Bounded training state shared with the CPU worker, without an engine owner.
use super::audit::{
    observed_cost, ObservationFunnelSnapshot, PreUpdatePrediction, TrainingAuditSnapshot,
    TrainingDisposition,
};
use super::checkpoint::FrozenCostCheckpoint;
use super::profile::{CostTrainer, EngineCostSnapshot, TrainingSeed};
use super::profile_export::{ExportPlan, ExportSession};
use super::*;
use parking_lot::{Mutex, RwLock};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
mod host_content;
mod selected;
pub(super) mod structured;
pub(super) mod structured_v2;
pub(super) use host_content::statistical::whole_wave_observation;

pub(super) struct CostTrainingState {
    pub sink: Arc<BoundedCostSampleSink>,
    trainer: Mutex<Option<CostTrainer>>,
    feedback: Mutex<Option<super::selected_feedback::Monitor>>,
    snapshot: RwLock<Option<Arc<EngineCostSnapshot>>>,
    pub receipt: Option<ferrum_types::SloCostProfileReceipt>,
    audit: Mutex<TrainingAuditSnapshot>,
    max_samples_per_update: usize,
    export: Mutex<ExportSession>,
    finalize_export: AtomicBool,
    clock: Arc<dyn CostObservationClock>,
    processed_ordinal: AtomicU64,
    host_content_enabled: bool,
    row_multiset_enabled: bool,
}

impl CostTrainingState {
    pub fn new(
        config: &ferrum_types::SloCostObservationConfig,
        mut seed: TrainingSeed,
        export: Option<ExportPlan>,
        clock: Arc<dyn CostObservationClock>,
    ) -> Result<Self, ferrum_types::FerrumError> {
        let feedback = if !config.selected_feedback.is_disabled() {
            let snapshot = seed.snapshot.as_ref().ok_or_else(|| {
                ferrum_types::FerrumError::config(
                    "selected feedback requires an actually imported profile6",
                )
            })?;
            let model = snapshot.selected_import().ok_or_else(|| {
                ferrum_types::FerrumError::config(
                    "selected feedback cannot reinterpret a legacy model",
                )
            })?;
            super::selected_feedback::Monitor::open(&config.selected_feedback, model)?
        } else if !config.structured_feedback.is_disabled() {
            seed.snapshot
                .as_ref()
                .ok_or_else(|| {
                    ferrum_types::FerrumError::config(
                        "structured feedback requires an actually imported qualified V2 catalog",
                    )
                })?
                .open_structured_feedback(&config.structured_feedback)?
        } else {
            None
        };
        if let Some(monitor) = &feedback {
            let view = monitor.current_view();
            seed.snapshot = seed
                .snapshot
                .as_ref()
                .and_then(|snapshot| snapshot.with_feedback(view));
            if let Some(receipt) = &mut seed.receipt {
                receipt.model_version = seed
                    .snapshot
                    .as_ref()
                    .expect("feedback snapshot")
                    .model_version();
            }
        }
        Ok(Self {
            feedback: Mutex::new(feedback),
            sink: Arc::new(
                BoundedCostSampleSink::new(CostSampleSinkLimits {
                    max_samples: config.max_queued_samples.get(),
                    max_shape_rows: config.max_queued_shape_rows.get(),
                })
                .map_err(|reason| {
                    ferrum_types::FerrumError::config(format!("cost sample limits: {reason:?}"))
                })?,
            ),
            trainer: Mutex::new(seed.trainer),
            snapshot: RwLock::new(seed.snapshot),
            receipt: seed.receipt,
            audit: Mutex::new(TrainingAuditSnapshot {
                selected_serving: config.predictor.is_selected().then(Default::default),
                ..Default::default()
            }),
            max_samples_per_update: config.max_samples_per_update.get(),
            export: Mutex::new(ExportSession::new(export)),
            finalize_export: AtomicBool::new(false),
            clock,
            processed_ordinal: AtomicU64::new(0),
            host_content_enabled: matches!(
                config.model.feature_model,
                ferrum_types::SloCostFeatureModel::EmpiricalHostContentV1 { .. }
                    | ferrum_types::SloCostFeatureModel::EmpiricalRowMultisetV2 { .. }
                    | ferrum_types::SloCostFeatureModel::EmpiricalPromptRangeV3 { .. }
            ),
            row_multiset_enabled: matches!(
                config.model.feature_model,
                ferrum_types::SloCostFeatureModel::EmpiricalRowMultisetV2 { .. }
                    | ferrum_types::SloCostFeatureModel::EmpiricalPromptRangeV3 { .. }
            ),
        })
    }

    /// Only the worker (or deterministic unit harness) calls this. The model
    /// uses receipt-time publication: a concurrent producer may already have
    /// queued older samples than the worker's wall clock. Publishing wall-now
    /// would reject those valid receipts as a clock reversal. Neither queueing
    /// nor consuming changes sample age; prediction still checks its real now.
    pub fn consume_batch(&self) -> bool {
        let started = std::time::Instant::now();
        let mut trainer = self.trainer.lock();
        let mut export = self.export.lock();
        let mut audit = self.audit.lock();
        let mut feedback = self.feedback.lock();
        // One immutable pre-batch snapshot: neither earlier observations in
        // this drain nor the new publication can improve their own coverage.
        // This Arc is never attached to queued or retained raw observations.
        let previous = self.snapshot();
        let mut watermark = None;
        let mut drained = 0;
        for _ in 0..self.max_samples_per_update {
            let Some((ordinal, entry)) = self.sink.pop_numbered() else {
                break;
            };
            drained += 1;
            let mut exhausted = audit.counter_exhausted;
            if let Some(selected_audit) = audit.selected_serving.as_mut() {
                let evaluation =
                    selected::evaluate(&entry, ordinal, previous.as_deref(), self.clock.as_ref());
                selected_audit
                    .presubmit
                    .record(&entry, &evaluation, &mut exhausted);
                if let Some(monitor) = feedback.as_mut() {
                    monitor.observe(ordinal, &entry, &evaluation, previous.as_deref());
                }
                selected_audit.record(evaluation, &mut exhausted);
            }
            if let Some(monitor) = feedback
                .as_mut()
                .filter(|m| m.kind() == super::selected_feedback::FeedbackKind::StructuredV2)
            {
                let evaluation = super::structured_feedback::evaluate(
                    &entry,
                    previous.as_deref(),
                    self.clock.as_ref(),
                );
                monitor.observe_classified(ordinal, evaluation);
            }
            audit.counter_exhausted = exhausted;
            let host_result = if self.host_content_enabled {
                let (stages, legacy) = match &entry {
                    CostEvidenceEntry::Training { stages, .. } => (stages.as_deref(), None),
                    CostEvidenceEntry::StagesOnly {
                        stages,
                        legacy_rejection,
                    } => (Some(stages.as_ref()), Some(*legacy_rejection)),
                };
                let result = host_content::observe(
                    stages,
                    legacy,
                    self.row_multiset_enabled,
                    trainer.as_mut(),
                    previous.as_deref(),
                );
                let kind = stages
                    .and_then(|stages| stages.actual_shape.as_ref())
                    .map(|shape| shape.kind);
                let mut exhausted = audit.counter_exhausted;
                audit
                    .host_content
                    .record(kind, result.evaluation, &mut exhausted);
                audit.counter_exhausted = exhausted;
                if result.evaluation.training.recorded() {
                    watermark = result.sample.as_ref().map(|sample| sample.observed_at_ns);
                }
                Some(result)
            } else {
                None
            };
            let (sample, stages) = match entry {
                CostEvidenceEntry::Training { sample, stages } => (sample, stages),
                CostEvidenceEntry::StagesOnly {
                    stages,
                    legacy_rejection,
                } => {
                    export.record_stages_with_host(
                        ordinal,
                        stages,
                        legacy_rejection,
                        host_result
                            .as_ref()
                            .map(|result| (result.evaluation, result.sample.as_ref())),
                    );
                    // Auxiliary-only records never enter legacy training. The
                    // explicit host-content model was evaluated above using
                    // its own boundary, outcome counters, and receipt clock.
                    self.processed_ordinal.store(ordinal, Ordering::Release);
                    if feedback.as_ref().is_some_and(|m| m.pending_publication()) {
                        break;
                    }
                    continue;
                }
            };
            let observed_at = sample.observed_at_ns;
            let kind = sample.actual_shape.kind;
            let prediction = PreUpdatePrediction::query(previous.as_deref(), &sample);
            let cost = observed_cost(&sample);
            // Only the CPU worker copies an enabled capture sample. Preserve
            // its local receipt before imported training changes clock epoch.
            let original = export.needs_observation().then(|| sample.clone());
            let disposition = if self.host_content_enabled {
                TrainingDisposition::Unavailable
            } else {
                TrainingDisposition::from_result(
                    trainer.as_mut().map(|trainer| trainer.observe(sample)),
                )
            };
            let prediction = prediction.with_comparison(cost, disposition);
            audit.observe(kind, disposition);
            audit.predict(kind, prediction);
            if let Some(original) = original {
                export.record_with_host(
                    ordinal,
                    &original,
                    disposition,
                    prediction,
                    stages,
                    host_result
                        .as_ref()
                        .map(|result| (result.evaluation, result.sample.as_ref())),
                );
            } else {
                export.note_unavailable_observation();
                if stages.is_some() {
                    export.note_unavailable_host_stages();
                }
            }
            if disposition.recorded() {
                metrics::counter!("ferrum.engine.cost_samples_recorded_total").increment(1);
                watermark = Some(observed_at);
            } else {
                metrics::counter!("ferrum.engine.cost_samples_rejected_total").increment(1);
            }
            // Rejected samples still complete their accepted queue position.
            self.processed_ordinal.store(ordinal, Ordering::Release);
            if feedback.as_ref().is_some_and(|m| m.pending_publication()) {
                break;
            }
        }
        if let Some(monitor) = feedback.as_mut() {
            let stats = self.sink.stats();
            let drops = stats
                .entries_dropped_capacity
                .checked_add(stats.entries_dropped_contention);
            if let Some(view) = monitor.publish(drops) {
                // Persisted first; this lock only swaps an Arc, never performs IO.
                let next = self
                    .snapshot
                    .read()
                    .as_ref()
                    .and_then(|snapshot| snapshot.with_feedback(view.clone()));
                *self.snapshot.write() = next;
                view.activate();
            }
        }
        drop(previous);
        if let Some(watermark) = watermark {
            // A watermark exists only after this trainer recorded a sample.
            match trainer
                .as_mut()
                .expect("recorded trainer")
                .publish(watermark)
            {
                Ok(snapshot) => {
                    *self.snapshot.write() = Some(snapshot);
                    audit.publish(Ok(()));
                }
                Err(reason) => {
                    audit.publish(Err(reason));
                    metrics::counter!("ferrum.engine.cost_model_publish_rejected_total")
                        .increment(1);
                }
            }
        }
        let processed = self.processed_ordinal.load(Ordering::Acquire);
        let checkpoint_completed =
            if let Some((cutoff, paths)) = self.sink.begin_checkpoint(processed) {
                let profile_cut = paths.map(|paths| {
                    export
                        .export_cut(
                            paths,
                            cutoff,
                            self.clock.as_ref(),
                            self.sink.stats(),
                            &audit,
                        )
                        .map_err(|error| error.to_string())
                });
                self.sink.complete_checkpoint(FrozenCostCheckpoint {
                    accepted_ordinal: cutoff,
                    snapshot: self.snapshot(),
                    training: audit.clone(),
                    export: export.audit_snapshot(),
                    profile_cut,
                });
                true
            } else {
                false
            };
        drop(trainer);
        let needs_drain = self.sink.needs_drain(processed);
        if !needs_drain && self.finalize_export.load(Ordering::Acquire) {
            export.finish(self.clock.as_ref(), self.sink.stats(), audit.clone());
            if let Some(monitor) = feedback.as_mut() {
                monitor.finish();
            }
        }
        drop(export);
        drop(audit);
        drop(feedback);
        self.publish_metrics();
        metrics::histogram!("ferrum.engine.cost_training_update_seconds")
            .record(started.elapsed().as_secs_f64());
        // A completed barrier may have unblocked post-cut work. Check retained
        // work too, including publication racing the last empty pop on shutdown.
        checkpoint_completed || needs_drain || drained == self.max_samples_per_update
    }

    fn publish_metrics(&self) {
        let stats = self.sink.stats();
        metrics::gauge!("ferrum.engine.cost_observation_published").set(stats.published as f64);
        metrics::gauge!("ferrum.engine.cost_observation_drained").set(stats.drained as f64);
        metrics::gauge!("ferrum.engine.cost_observation_rejected").set(
            stats
                .rejected
                .iter()
                .chain(stats.preparation_rejected.iter())
                .chain(stats.initialization_rejected.iter())
                .copied()
                .fold(0u64, u64::saturating_add) as f64,
        );
        metrics::gauge!("ferrum.engine.cost_observation_lost").set(
            stats
                .entries_dropped_capacity
                .saturating_add(stats.entries_dropped_contention) as f64,
        );
        metrics::gauge!("ferrum.engine.cost_evidence_entries_offered")
            .set(stats.entries_offered as f64);
        metrics::gauge!("ferrum.engine.cost_evidence_entries_published")
            .set(stats.entries_published as f64);
        metrics::gauge!("ferrum.engine.cost_evidence_entries_drained")
            .set(stats.entries_drained as f64);
        metrics::gauge!("ferrum.engine.cost_host_stages_published")
            .set(stats.host_stages_published as f64);
        // A lossless queue says nothing about rejected/uninstrumented calls,
        // training support or retained export coverage.
        metrics::gauge!("ferrum.engine.cost_observation_queue_lossless")
            .set(if stats.has_lost_samples() { 0.0 } else { 1.0 });
        metrics::gauge!("ferrum.engine.cost_training_samples").set(self.trained_samples() as f64);
        if let Some(selected) = &self.audit.lock().selected_serving {
            metrics::gauge!("ferrum.engine.selected_serving_entries_drained")
                .set(selected.drained_entries as f64);
            metrics::gauge!("ferrum.engine.selected_serving_constructable_actual")
                .set(selected.constructable_actual as f64);
            metrics::gauge!("ferrum.engine.selected_serving_known_compared")
                .set(selected.known_compared as f64);
            metrics::gauge!("ferrum.engine.selected_serving_no_published_model")
                .set(selected.no_published_model as f64);
            metrics::gauge!("ferrum.engine.selected_serving_prediction_unknown").set(
                selected
                    .prediction_unknown
                    .iter()
                    .map(|v| v.count)
                    .fold(0u64, u64::saturating_add) as f64,
            );
            metrics::gauge!("ferrum.engine.selected_serving_actual_unavailable").set(
                selected
                    .actual_unavailable
                    .iter()
                    .map(|v| v.count)
                    .fold(0u64, u64::saturating_add) as f64,
            );
            metrics::gauge!("ferrum.engine.selected_serving_not_completed").set(
                selected
                    .not_completed
                    .iter()
                    .map(|v| v.count)
                    .fold(0u64, u64::saturating_add) as f64,
            );
            metrics::gauge!("ferrum.engine.selected_serving_terminal_compared")
                .set(selected.terminal_compared as f64);
            metrics::gauge!("ferrum.engine.selected_serving_underestimates")
                .set(selected.underestimates as f64);
            metrics::gauge!("ferrum.engine.selected_serving_underestimate_max_ns")
                .set(selected.max_underestimate_ns as f64);
            metrics::gauge!("ferrum.engine.selected_presubmit_compared")
                .set(selected.presubmit.compared as f64);
            metrics::gauge!("ferrum.engine.selected_presubmit_underestimates")
                .set(selected.presubmit.underestimates as f64);
            metrics::gauge!("ferrum.engine.selected_presubmit_underestimate_max_ns")
                .set(selected.presubmit.maximum_underestimate_ns as f64);
        }
        if let Some(monitor) = self.feedback.lock().as_ref() {
            let feedback = monitor.audit();
            let names = match monitor.kind() {
                super::selected_feedback::FeedbackKind::Selected => [
                    "ferrum.engine.selected_feedback_epoch",
                    "ferrum.engine.selected_feedback_revoked",
                    "ferrum.engine.selected_feedback_corrections",
                    "ferrum.engine.selected_feedback_maximum_margin_ns",
                    "ferrum.engine.selected_feedback_persistence_failed",
                ],
                super::selected_feedback::FeedbackKind::StructuredV2 => [
                    "ferrum.engine.structured_feedback_epoch",
                    "ferrum.engine.structured_feedback_revoked",
                    "ferrum.engine.structured_feedback_corrections",
                    "ferrum.engine.structured_feedback_maximum_margin_ns",
                    "ferrum.engine.structured_feedback_persistence_failed",
                ],
            };
            metrics::gauge!(names[0]).set(feedback.epoch as f64);
            metrics::gauge!(names[1]).set(if feedback.revoked.is_some() { 1.0 } else { 0.0 });
            metrics::gauge!(names[2]).set(feedback.corrections as f64);
            metrics::gauge!(names[3]).set(feedback.maximum_margin_ns as f64);
            metrics::gauge!(names[4]).set(if feedback.persistence_failed {
                1.0
            } else {
                0.0
            });
        }
        if let Some(snapshot) = self.snapshot() {
            metrics::gauge!("ferrum.engine.cost_model_version")
                .set(snapshot.model_version() as f64);
            metrics::gauge!("ferrum.engine.cost_model_buckets").set(snapshot.bucket_count() as f64);
        }
    }

    pub fn snapshot(&self) -> Option<Arc<EngineCostSnapshot>> {
        self.snapshot
            .read()
            .clone()
            .filter(|snapshot| snapshot.current())
    }

    pub fn try_snapshot(&self) -> Option<Option<Arc<EngineCostSnapshot>>> {
        self.snapshot
            .try_read()
            .map(|snapshot| snapshot.clone().filter(|snapshot| snapshot.current()))
    }

    pub fn trained_samples(&self) -> u64 {
        let audit = self.audit.lock();
        if self.host_content_enabled {
            audit.host_content.outcomes.recorded
        } else {
            audit.outcomes.recorded
        }
    }

    pub fn audit_snapshot(&self) -> ObservationFunnelSnapshot {
        // Same order as the worker, and release each lock before the next.
        let export = self.export.lock().audit_snapshot();
        let training = self.audit.lock().clone();
        let (selected_feedback, structured_feedback) = {
            let feedback = self.feedback.lock();
            match feedback.as_ref() {
                Some(monitor)
                    if monitor.kind() == super::selected_feedback::FeedbackKind::Selected =>
                {
                    (Some(monitor.audit()), None)
                }
                Some(monitor) => (None, Some(monitor.audit())),
                None => (None, None),
            }
        };
        ObservationFunnelSnapshot {
            prospective_capture: None,
            selected_feedback,
            structured_feedback,
            scope: "instrumented calls and offered cost observations; entries_* also count auxiliary host-stage-only records; auxiliary stages train only the explicit empirical-host-content model at their original receipt clock and never become legacy samples; host_content and legacy training populations remain separate; live counters are not an atomic cut; pre-update prediction is retrospective actual-shape diagnostics, not pre-execution candidate coverage; uninstrumented physical waves remain unknown",
            sink: self.sink.stats(),
            training,
            export,
        }
    }

    pub fn request_export_finalization(&self) {
        self.sink.close_checkpoints();
        self.finalize_export.store(true, Ordering::Release);
    }

    pub fn export_result(&self) -> Result<(), ferrum_types::FerrumError> {
        self.export.lock().check_finished()?;
        if let Some(monitor) = self.feedback.lock().as_ref() {
            monitor.check_finished()?;
        }
        Ok(())
    }

    #[cfg(test)]
    pub fn with_training_paused<T>(&self, action: impl FnOnce() -> T) -> T {
        let _guard = self.trainer.lock();
        action()
    }

    #[cfg(test)]
    pub async fn with_training_paused_async<F: std::future::Future>(&self, action: F) -> F::Output {
        // Protocol fixtures control producer/consumer interleaving. Holding
        // this test-only guard never changes the nonblocking production sink.
        let _guard = self.trainer.lock();
        action.await
    }
}

/// Owned by the worker closure itself. It runs on normal exit and unwind, even
/// while EngineCostRuntime still owns the trainer Arc and a caller is waiting.
pub(super) struct TrainingWorkerOwner(pub Arc<CostTrainingState>);
impl TrainingWorkerOwner {
    pub fn consume_batch(&self) -> bool {
        self.0.consume_batch()
    }
}
impl Drop for TrainingWorkerOwner {
    fn drop(&mut self) {
        if std::thread::panicking() || !self.0.finalize_export.load(Ordering::Acquire) {
            if let Some(monitor) = self.0.feedback.lock().as_mut() {
                monitor.worker_stopped_unclean();
            }
        }
        self.0.sink.worker_stopped();
    }
}
