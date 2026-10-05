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
pub(super) mod host_content;
mod live_publication;
mod selected;
pub(super) mod structured;
pub(super) mod structured_v2;
pub(super) use host_content::statistical::whole_wave_observation;

// Test-only context from the original worker turn; no replay, re-query, or
// replacement clock reading participates in feedback classification.
#[cfg(test)]
std::thread_local! {
    static PREVIOUS_WORKER_TURN: std::cell::Cell<(u64, u128, u128, u128)> =
        const { std::cell::Cell::new((0, 0, 0, 0)) };
}

#[cfg(test)]
fn feedback_clock_context(
    observation: &super::selected_feedback::FeedbackObservation,
) -> (&'static str, Option<(u64, u64)>) {
    use super::selected_feedback::FeedbackObservation::*;
    match observation {
        ExpiredModel {
            observed_at_ns,
            consumed_at_ns,
            ..
        } => ("ExpiredModel", Some((*observed_at_ns, *consumed_at_ns))),
        OutsideCatalog {
            observed_at_ns,
            consumed_at_ns,
        } => ("OutsideCatalog", Some((*observed_at_ns, *consumed_at_ns))),
        OutsideSupport {
            observed_at_ns,
            consumed_at_ns,
        } => ("OutsideSupport", Some((*observed_at_ns, *consumed_at_ns))),
        OutsideRoute {
            observed_at_ns,
            consumed_at_ns,
        } => ("OutsideRoute", Some((*observed_at_ns, *consumed_at_ns))),
        ProvenNoSubmission {
            observed_at_ns,
            consumed_at_ns,
        } => (
            "ProvenNoSubmission",
            Some((*observed_at_ns, *consumed_at_ns)),
        ),
        OutsidePreparation {
            observed_at_ns,
            consumed_at_ns,
        } => (
            "OutsidePreparation",
            Some((*observed_at_ns, *consumed_at_ns)),
        ),
        Compared(value) => (
            "Compared",
            Some((value.observed_at_ns, value.consumed_at_ns)),
        ),
        NotSubmitted => ("NotSubmitted", None),
        FailedOrPartial => ("FailedOrPartial", None),
        Uncomparable => ("Uncomparable", None),
        InvalidIdentity => ("InvalidIdentity", None),
    }
}

pub(super) struct CostTrainingState {
    pub sink: Arc<BoundedCostSampleSink>,
    pub(super) prefix: Option<super::prefix::PrefixCostTraining>,
    trainer: Mutex<Option<CostTrainer>>,
    feedback: Mutex<Option<super::selected_feedback::Monitor>>,
    automatic_feedback: Option<ferrum_types::SloStructuredFeedbackPolicy>,
    pub(super) live: Option<Arc<super::live_calibration::LiveCalibration>>,
    snapshot: RwLock<Option<Arc<EngineCostSnapshot>>>,
    structured_epoch: Option<Arc<super::structured_epoch::Epoch>>,
    source_fingerprint:
        Option<ferrum_scheduler::implementations::continuous::cost_model::ExecutionFingerprint>,
    published_receipt: Mutex<Option<ferrum_types::SloCostProfileReceipt>>,
    pub receipt: Option<ferrum_types::SloCostProfileReceipt>,
    audit: Mutex<TrainingAuditSnapshot>,
    max_samples_per_update: usize,
    export: Mutex<ExportSession>,
    finalize_export: AtomicBool,
    clock: Arc<dyn CostObservationClock>,
    processed_ordinal: AtomicU64,
    host_content_enabled: bool,
    row_multiset_enabled: bool,
    reuse: Option<Arc<super::automatic_reuse::AutomaticReuse>>,
    reuse_clean_shutdown: AtomicBool,
}

impl CostTrainingState {
    pub fn new(
        config: &ferrum_types::SloCostObservationConfig,
        seed: TrainingSeed,
        export: Option<ExportPlan>,
        clock: Arc<dyn CostObservationClock>,
        live: Option<Arc<super::live_calibration::LiveCalibration>>,
        source_fingerprint: Option<
            ferrum_scheduler::implementations::continuous::cost_model::ExecutionFingerprint,
        >,
    ) -> Result<Self, ferrum_types::FerrumError> {
        Self::new_with_reuse(
            config,
            seed,
            export,
            clock,
            live,
            source_fingerprint,
            None,
            None,
        )
    }
    #[allow(clippy::too_many_arguments)]
    pub(super) fn new_with_reuse(
        config: &ferrum_types::SloCostObservationConfig,
        mut seed: TrainingSeed,
        export: Option<ExportPlan>,
        clock: Arc<dyn CostObservationClock>,
        live: Option<Arc<super::live_calibration::LiveCalibration>>,
        source_fingerprint: Option<
            ferrum_scheduler::implementations::continuous::cost_model::ExecutionFingerprint,
        >,
        restored_monitor: Option<super::selected_feedback::Monitor>,
        reuse: Option<Arc<super::automatic_reuse::AutomaticReuse>>,
    ) -> Result<Self, ferrum_types::FerrumError> {
        let automatic_feedback = match &config.live_structured_calibration {
            ferrum_types::SloLiveStructuredCalibration::AutomaticV1 { settings }
                if config.structured_feedback.is_disabled() =>
            {
                Some(
                    ferrum_types::SloStructuredFeedbackPolicy::RetrospectiveOwnerMarginV1 {
                        policy: settings.feedback.clone(),
                        storage: ferrum_types::SloSelectedFeedbackStorageV1::MemoryOnly,
                    },
                )
            }
            _ => None,
        };
        let restored = restored_monitor.is_some();
        let mut feedback = if let Some(monitor) = restored_monitor {
            Some(monitor)
        } else if !config.selected_feedback.is_disabled() {
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
        } else if let Some(policy) = &automatic_feedback {
            seed.snapshot
                .as_ref()
                .map(|snapshot| snapshot.open_structured_feedback(policy))
                .transpose()?
                .flatten()
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
        let structured_epoch = (config.predictor
            == ferrum_types::SloCostPredictor::StructuredWholeWaveV2)
            .then(|| {
                let initial = feedback
                    .as_ref()
                    .map(|m| m.current_view().epoch)
                    .or_else(|| seed.snapshot.as_ref().map(|s| s.model_version()))
                    .unwrap_or(0);
                super::structured_epoch::Epoch::new(initial)
            });
        if let Some(epoch) = &structured_epoch {
            if restored {
                epoch.activate(0);
            }
            if let Some(monitor) = &mut feedback {
                monitor.attach_structured_epoch(epoch.clone());
            }
            if let Some(snapshot) = &seed.snapshot {
                let version = feedback
                    .as_ref()
                    .map_or(snapshot.model_version(), |m| m.current_view().epoch);
                seed.snapshot = snapshot.with_structured_epoch(epoch.view(version));
            }
        }
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
        let initial_generation = seed.snapshot.as_ref().map_or(0, |s| s.model_version());
        let feedback_population =
            automatic_feedback
                .as_ref()
                .and_then(|_| match &config.live_structured_calibration {
                    ferrum_types::SloLiveStructuredCalibration::AutomaticV1 { settings } => {
                        Some(settings.route_population)
                    }
                    _ => None,
                });
        let mut state = Self {
            prefix: None,
            reuse,
            reuse_clean_shutdown: AtomicBool::new(false),
            feedback: Mutex::new(feedback),
            automatic_feedback,
            structured_epoch,
            source_fingerprint,
            published_receipt: Mutex::new(seed.receipt.clone()),
            live,
            sink: Arc::new(
                BoundedCostSampleSink::new_with_byte_limits(
                    CostSampleSinkLimits {
                        max_samples: config.max_queued_samples.get(),
                        max_shape_rows: config.max_queued_shape_rows.get(),
                    },
                    CostRecorderByteLimits {
                        maximum_retained_bytes: config.maximum_queued_bytes.get(),
                        maximum_working_bytes: config.maximum_working_bytes.get(),
                    },
                )
                .map_err(|reason| {
                    ferrum_types::FerrumError::config(format!("cost sample limits: {reason:?}"))
                })?
                .with_feedback_population(feedback_population),
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
        };
        state.prefix = state.source_fingerprint.as_ref().and_then(|fingerprint| {
            super::prefix::PrefixCostTraining::new(
                state.sink.clone(),
                fingerprint.clone(),
                state.clock.clone(),
                &config.model,
            )
        });
        if state.structured_epoch.is_some() {
            state.sink.set_source_generation(initial_generation);
        }
        if restored {
            // The replayed Monitor keeps all original margins/counters. Only
            // this new process's epoch gate opens after the complete adapter
            // and new observation sink exist.
            if let Some(monitor) = state.feedback.lock().as_ref() {
                monitor.current_view().activate();
            }
        }
        Ok(state)
    }

    /// Only the worker (or deterministic unit harness) calls this. The model
    /// uses receipt-time publication: a concurrent producer may already have
    /// queued older samples than the worker's wall clock. Publishing wall-now
    /// would reject those valid receipts as a clock reversal. Neither queueing
    /// nor consuming changes sample age; prediction still checks its real now.
    pub fn consume_batch(&self) -> bool {
        let started = std::time::Instant::now();
        if let Some(live) = &self.live {
            live.poll(self.clock.now_ns());
        }
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
        let mut expiry_due = false;
        // Source7 appends one original record between FIFO feedback turns.
        // Avoid accumulating a whole block's conversion/IO ahead of the next
        // observation. Other trainers retain their configured batch limit.
        let batch_limit = if self
            .live
            .as_ref()
            .is_some_and(|live| live.incremental_owner_records())
        {
            1
        } else {
            self.max_samples_per_update
        };
        for _ in 0..batch_limit {
            let Some(input) = self.sink.pop_training_input() else {
                break;
            };
            drained += 1;
            // Maintenance has an original FIFO ordinal but no inference ticket
            // or feedback population. Even invalid maintenance cannot become
            // an Uncomparable inference observation or revoke its catalog.
            let (ordinal, entry, live_ticket, source_generation, original_no_submission) =
                match input {
                    CostTrainingInput::Inference(value) => value,
                    CostTrainingInput::PrefixDiscarded { ordinal } => {
                        self.processed_ordinal.store(ordinal, Ordering::Release);
                        continue;
                    }
                    CostTrainingInput::Prefix {
                        ordinal,
                        sample,
                        memory,
                    } => {
                        let _memory = memory;
                        if let Some(prefix) = &self.prefix {
                            prefix.consume(sample);
                        }
                        self.processed_ordinal.store(ordinal, Ordering::Release);
                        continue;
                    }
                };
            let Some(resolved) = entry else {
                // Resolving failed facts cannot erase their original FIFO cut
                // or turn their original ticket into a replacement sample.
                let had_live_ticket = live_ticket.is_some();
                let no_submission = match (&self.live, live_ticket) {
                    (Some(live), Some(ticket)) if ticket.no_submission().is_some() => {
                        Some(live.consume_no_submission(ticket, ordinal))
                    }
                    (_, ticket) => {
                        drop(ticket);
                        None
                    }
                };
                if source_generation != 0 && source_generation == self.sink.source_generation() {
                    if let Some(monitor) = feedback.as_mut() {
                        let independent_observed = (!had_live_ticket
                            && self.automatic_feedback.is_some())
                        .then(|| {
                            original_no_submission
                                .as_ref()
                                .zip(self.source_fingerprint.as_ref())
                                .and_then(|(proof, fp)| proof.feedback_observed_at(ordinal, fp))
                        })
                        .flatten();
                        let no_submission = no_submission.or_else(|| {
                            independent_observed.map(|observed_at_ns| {
                                super::live_calibration::LiveConsumption::NoSubmission {
                                    observed_at_ns,
                                }
                            })
                        });
                        let observation = match no_submission {
                            Some(super::live_calibration::LiveConsumption::NoSubmission {
                                observed_at_ns,
                            }) if self.automatic_feedback.is_some() => {
                                match self.clock.now_ns().filter(|now| *now >= observed_at_ns) {
                                    Some(consumed_at_ns) => super::selected_feedback::FeedbackObservation::ProvenNoSubmission { observed_at_ns, consumed_at_ns },
                                    None => super::selected_feedback::FeedbackObservation::InvalidIdentity,
                                }
                            }
                            _ => super::selected_feedback::FeedbackObservation::Uncomparable,
                        };
                        let was_revoked = monitor.revocation().is_some();
                        #[cfg(test)]
                        let original_clock = feedback_clock_context(&observation);
                        monitor.observe_classified(ordinal, observation);
                        if !was_revoked && monitor.revocation().is_some() {
                            #[cfg(test)]
                            eprintln!("original feedback revocation: stage=raw_resolution ordinal={ordinal} generation={source_generation} epoch={:?} reason={:?} original={original_clock:?} batch_elapsed_ns={} previous_turn=(processed,total,ingest,advance)_ns={:?}", previous.as_ref().map(|s|s.model_version()), monitor.revocation(), started.elapsed().as_nanos(), PREVIOUS_WORKER_TURN.with(|v|v.get()));
                            tracing::warn!(target: "ferrum::cost_feedback_diagnostics",
                                event = "structured_feedback_revoked_v1",
                                ordinal, source_generation,
                                model_epoch = previous.as_ref().map(|s| s.model_version()),
                                reason = ?monitor.revocation(),
                                stage = "raw_resolution", had_live_ticket,
                                had_no_submission_proof = original_no_submission.is_some(),
                                "Cost feedback revoked the catalog at this original FIFO entry");
                        }
                    }
                }
                self.processed_ordinal.store(ordinal, Ordering::Release);
                if feedback.as_ref().is_some_and(|m| m.pending_publication()) {
                    break;
                }
                continue;
            };
            // Keep the shared payload charged through into_entry and export.
            let _observation_memory = resolved.memory();
            let entry = resolved.entry();
            let live_consumption = if let (Some(live), Some(ticket)) = (&self.live, live_ticket) {
                Some(live.consume_resolved(ticket, ordinal, &resolved))
            } else {
                None
            };
            let outside_route = matches!(
                live_consumption,
                Some(super::live_calibration::LiveConsumption::OutsideDeclaredRoute { .. })
            );
            let mut exhausted = audit.counter_exhausted;
            if let Some(selected_audit) = audit.selected_serving.as_mut() {
                let evaluation = selected::evaluate_resolved(
                    &resolved,
                    ordinal,
                    previous.as_deref(),
                    self.clock.as_ref(),
                );
                selected_audit
                    .presubmit
                    .record(&entry, &evaluation, &mut exhausted);
                if let Some(monitor) = feedback.as_mut() {
                    monitor.observe(ordinal, &entry, &evaluation, previous.as_deref());
                }
                selected_audit.record(evaluation, &mut exhausted);
            }
            if let Some(monitor) = feedback.as_mut().filter(|m| {
                m.kind() == super::selected_feedback::FeedbackKind::StructuredV2
                    && source_generation != 0
                    && source_generation == self.sink.source_generation()
            }) {
                let was_revoked = monitor.revocation().is_some();
                let mut failure = None;
                let evaluation = match live_consumption {
                    Some(super::live_calibration::LiveConsumption::OutsideDeclaredRoute {
                        observed_at_ns,
                    }) if self.automatic_feedback.is_some() => {
                        match self.clock.now_ns().filter(|now| *now >= observed_at_ns) {
                            Some(consumed_at_ns) => {
                                super::selected_feedback::FeedbackObservation::OutsideRoute {
                                    observed_at_ns,
                                    consumed_at_ns,
                                }
                            }
                            None => {
                                failure = Some(super::structured_feedback::Failure::Clock);
                                super::selected_feedback::FeedbackObservation::InvalidIdentity
                            }
                        }
                    }
                    Some(super::live_calibration::LiveConsumption::OutsideDeclaredRoute {
                        ..
                    }) => {
                        failure = Some(super::structured_feedback::Failure::RouteProof);
                        super::selected_feedback::FeedbackObservation::Uncomparable
                    }
                    None if self.automatic_feedback.is_some() => {
                        super::structured_feedback::evaluate_preparation(
                            &resolved,
                            previous.as_deref(),
                            self.clock.as_ref(),
                            &mut failure,
                        )
                        .or_else(|| {
                            super::structured_feedback::evaluate_outside(
                                &entry,
                                previous.as_deref(),
                                self.clock.as_ref(),
                                &mut failure,
                            )
                        })
                        .unwrap_or_else(|| {
                            super::structured_feedback::evaluate_resolved(
                                &resolved,
                                previous.as_deref(),
                                self.clock.as_ref(),
                                &mut failure,
                            )
                        })
                    }
                    _ => super::structured_feedback::evaluate_resolved(
                        &resolved,
                        previous.as_deref(),
                        self.clock.as_ref(),
                        &mut failure,
                    ),
                };
                // Automatic discovery admits new owners through fresh phases.
                // Missing owners cannot qualify via feedback or invalidate an
                // unrelated qualified owner. Explicit legacy policies keep
                // their declared uncomparable-observation semantics.
                let evaluation = match evaluation {
                    super::selected_feedback::FeedbackObservation::OutsideCatalog { .. }
                    | super::selected_feedback::FeedbackObservation::OutsideSupport { .. }
                        if self.automatic_feedback.is_none() =>
                    {
                        super::selected_feedback::FeedbackObservation::Uncomparable
                    }
                    other => other,
                };
                // Finish this original FIFO position, then perform the
                // immutable expiry transition outside the drain locks before
                // consuming another position. The monitor still verifies the
                // original ordinal/domain/epoch/lag and may revoke on failure.
                expiry_due |= matches!(
                    &evaluation,
                    super::selected_feedback::FeedbackObservation::ExpiredModel { .. }
                );
                #[cfg(test)]
                let original_clock = feedback_clock_context(&evaluation);
                monitor.observe_classified(ordinal, evaluation);
                if !was_revoked && monitor.revocation().is_some() {
                    let stages = match &entry {
                        CostEvidenceEntry::Training { stages, .. } => stages.as_deref(),
                        CostEvidenceEntry::StagesOnly { stages, .. } => Some(stages.as_ref()),
                    };
                    let shape = stages.and_then(|s| s.actual_shape.as_ref());
                    #[cfg(test)]
                    eprintln!("original feedback revocation: stage=resolved ordinal={ordinal} generation={source_generation} epoch={:?} reason={:?} original={original_clock:?} failure={failure:?} call_id={:?} wave={:?} rows={:?} batch_elapsed_ns={} previous_turn=(processed,total,ingest,advance)_ns={:?} raw_pending={} raw_wait_max_ns={}", previous.as_ref().map(|s|s.model_version()), monitor.revocation(), stages.map(|s|s.call_id), shape.map(|s|s.kind), stages.map(|s|s.rows.len()), started.elapsed().as_nanos(), PREVIOUS_WORKER_TURN.with(|v|v.get()), self.sink.stats().raw_pending, self.sink.stats().raw_queue_wait_ns_max);
                    tracing::warn!(target: "ferrum::cost_feedback_diagnostics",
                        event = "structured_feedback_revoked_v1",
                        ordinal, source_generation,
                        model_epoch = previous.as_ref().map(|s| s.model_version()),
                        reason = ?monitor.revocation(), stage = "resolved",
                        failure = ?failure,
                        producer_failure = ?resolved.producer_diagnostic(),
                        call_id = stages.map(|s| s.call_id),
                        wave = ?shape.map(|s| s.kind),
                        rows = stages.map(|s| s.rows.len()),
                        "Cost feedback revoked the catalog at this original FIFO entry");
                }
            }
            audit.counter_exhausted = exhausted;
            let host_result = if self.host_content_enabled && !outside_route {
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
            let (sample, stages) = match resolved.into_entry() {
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
                    if expiry_due || feedback.as_ref().is_some_and(|m| m.pending_publication()) {
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
            let disposition = if self.host_content_enabled || outside_route {
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
            if expiry_due || feedback.as_ref().is_some_and(|m| m.pending_publication()) {
                break;
            }
        }
        if let Some(monitor) = feedback.as_mut() {
            let stats = self.sink.stats();
            let drops = stats
                .entries_dropped_capacity
                .checked_add(stats.entries_dropped_contention)
                .and_then(|n| n.checked_add(stats.entries_abandoned));
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
        let checkpoint = self.sink.begin_checkpoint(processed).map(|work| {
            let profile_cut = work.export_paths.map(|paths| {
                export
                    .export_cut(
                        paths,
                        work.cutoff,
                        self.clock.as_ref(),
                        self.sink.stats(),
                        &audit,
                    )
                    .map_err(|error| error.to_string())
            });
            (
                work.catalog_activation,
                FrozenCostCheckpoint {
                    accepted_ordinal: work.cutoff,
                    snapshot: self.snapshot(),
                    training: audit.clone(),
                    export: export.audit_snapshot(),
                    profile_cut,
                    catalog_activation: None,
                },
            )
        });
        drop(trainer);
        let needs_drain = self.sink.needs_drain(processed);
        if !needs_drain && self.finalize_export.load(Ordering::Acquire) {
            export.finish(self.clock.as_ref(), self.sink.stats(), audit.clone());
        }
        drop(export);
        drop(audit);
        drop(feedback);
        if expiry_due {
            // Do not call publication with trainer/feedback locks held. The
            // transition retains original child provenance and absolute age;
            // revoked feedback cannot be resurrected by this subset operation.
            self.prune_expired_live_catalog_audited();
        }
        // Use the real clock for the next feedback observation. No timestamp
        // is captured early to hide time spent converting or writing records.
        // This also runs while shutdown drains accepted original FIFO entries.
        #[cfg(test)]
        let ingest_started = std::time::Instant::now();
        if let Some(live) = &self.live {
            live.ingest_owner_record(processed);
        }
        #[cfg(test)]
        let ingest_elapsed_ns = ingest_started.elapsed().as_nanos();
        let source_pending = self
            .live
            .as_ref()
            .is_some_and(|live| live.has_pending_owner_records());
        // Keep the original control barrier occupied while installing. This
        // runs on the sole worker after releasing drain/feedback locks; no
        // post-cut observation can be consumed into the new model first.
        let checkpoint_completed = if let Some((activation, mut frozen)) = checkpoint {
            if expiry_due {
                frozen.snapshot = self.snapshot();
                frozen.training = self.audit.lock().clone();
            }
            if let Some(publication) = activation {
                frozen.catalog_activation = Some(
                    self.install_startup_catalog(publication)
                        .map_err(|error| error.to_string()),
                );
                frozen.snapshot = self.snapshot();
            }
            self.sink.complete_checkpoint(frozen);
            true
        } else {
            false
        };
        // Service-window fitting, durable writes and replay never hold the
        // inference-facing drain/feedback locks. This remains the sole worker.
        #[cfg(test)]
        let advance_started = std::time::Instant::now();
        if !self.finalize_export.load(Ordering::Acquire) {
            self.advance_live_calibration(processed);
        } else if !needs_drain && !source_pending && self.finish_live_calibration(processed) {
            // A complete final calibration block can replace the catalog
            // during shutdown. Persist that replacement before closing the
            // feedback store; it rejects all writes after finish.
            if let Some(monitor) = self.feedback.lock().as_mut() {
                if let (true, Some(reuse), Some(snapshot)) = (
                    self.reuse_clean_shutdown.load(Ordering::Acquire),
                    &self.reuse,
                    self.snapshot(),
                ) {
                    match reuse.finish(&snapshot, monitor, processed) {
                        Ok(()) => tracing::info!("Automatic cost restart cache prepared after final original FIFO drain; awaiting clean engine shutdown"),
                        Err(reason) => tracing::info!(?reason, "Automatic cost restart cache remains uncommitted"),
                    }
                }
                monitor.finish();
            }
        }
        #[cfg(test)]
        PREVIOUS_WORKER_TURN.with(|value| {
            value.set((
                processed,
                started.elapsed().as_nanos(),
                ingest_elapsed_ns,
                advance_started.elapsed().as_nanos(),
            ))
        });
        self.publish_metrics();
        metrics::histogram!("ferrum.engine.cost_training_update_seconds")
            .record(started.elapsed().as_secs_f64());
        // A completed barrier may have unblocked post-cut work. Check retained
        // work too, including publication racing the last empty pop on shutdown.
        checkpoint_completed || needs_drain || source_pending || drained == batch_limit
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
                .saturating_add(stats.entries_dropped_contention)
                .saturating_add(stats.entries_abandoned) as f64,
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
    pub(super) fn authorize_restart_shutdown(&self) {
        self.reuse_clean_shutdown.store(true, Ordering::Release);
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
            automatic_reuse: self.reuse.as_ref().map(|reuse| reuse.audit()),
            issued_structured_prediction: None,
            prospective_capture: None,
            live_calibration: self.live.as_ref().map(|live| live.audit()),
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
        if let Some(live) = &self.live {
            live.stop_capture();
        }
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
    pub fn consume_progress(&self) -> super::worker::WorkerProgress {
        if self.0.consume_batch() {
            super::worker::WorkerProgress::Ready
        } else if self
            .0
            .live
            .as_ref()
            .is_some_and(|live| live.automatic_computation_pending())
        {
            super::worker::WorkerProgress::Waiting
        } else {
            super::worker::WorkerProgress::Idle
        }
    }
    pub fn consume_batch(&self) -> bool {
        self.0.consume_batch()
    }
}
impl Drop for TrainingWorkerOwner {
    fn drop(&mut self) {
        self.0.close_structured_epoch();
        if std::thread::panicking() || !self.0.finalize_export.load(Ordering::Acquire) {
            if let Some(monitor) = self.0.feedback.lock().as_mut() {
                monitor.worker_stopped_unclean();
            }
        }
        self.0.sink.worker_stopped();
    }
}
