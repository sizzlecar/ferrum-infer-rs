//! Independent, versioned host-settled evidence. It does not change the old
//! preparation-to-commit sample or make terminal calls eligible for training.
use super::*;
use crate::continuous_engine::SequenceState;
use ferrum_interfaces::model_executor::ExecutorCompletionWork;
use ferrum_scheduler::implementations::continuous::cost_model::{
    ExecutionFingerprint, WaveExecutionShape,
};
use ferrum_types::FinishReason;
use serde::Serialize;
mod structured;
mod wire;
pub use structured::{QualifiedStructuredWaveEvidenceV1, StructuredSettlementUnknown};
pub(super) use wire::ExportEvidence;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum HostStageCompleteness {
    CompleteSingleWave,
    MissingEvidence,
    IdentityMismatch,
    InvalidClock,
    Failed,
    AdditionalOrUnknownWork,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum HostStageQueueDisposition {
    Published,
    DroppedCapacity,
    DroppedContended,
    DroppedWorkerStopped,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct HostStageQueueReceipt {
    /// The same FIFO entry as the optional cost sample, never source_record.
    pub accepted_ordinal: Option<u64>,
    pub disposition: HostStageQueueDisposition,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum HostStageWork {
    Decode {
        kv_tokens: u32,
    },
    Prefill {
        offset: u32,
        count: u32,
        total_prompt_tokens: u32,
    },
    Restore,
    Maintenance,
}

#[derive(Debug, Clone, Serialize)]
pub struct HostTerminalStageV1 {
    pub finish_reason: FinishReason,
    pub generated_tokens: u64,
    pub through_output_ordinal: u64,
    pub output_failed: bool,
    pub physical_failed: bool,
    pub scheduler_failed: bool,
    pub terminal_handoff_succeeded: bool,
    pub pending_restore_removed: bool,
    pub admission_cancellation_work: ExecutorCompletionWork,
    pub cache_completion_work: ExecutorCompletionWork,
    /// Legacy KV/recurrent releases have no complete failure receipt here.
    pub other_physical_resources: bool,
    pub request_slot_closed: bool,
    pub owner_matched: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct HostRowStageV1 {
    pub request_id: RequestId,
    pub owner_incarnation: u64,
    pub work_generation: u64,
    pub input_index: u32,
    pub actual_work: HostStageWork,
    /// Actual serial host order, independent of the device's physical row order.
    pub host_processing_ordinal: Option<u32>,
    pub host_started_at_ns: Option<u64>,
    pub token_committed_at_ns: Option<u64>,
    /// Bounded actor handoff, never socket or client-visible delivery.
    pub output_published_at_ns: Option<u64>,
    pub completion_started_at_ns: Option<u64>,
    pub settled_at_ns: Option<u64>,
    pub terminal: Option<HostTerminalStageV1>,
    pub completeness: HostStageCompleteness,
}

#[derive(Debug, Clone, Serialize)]
pub struct HostStageEvidenceV1 {
    pub schema_version: u32,
    pub call_id: u64,
    /// Optional same-call issued bound. It is separate from the worker's later
    /// actual-shape lookup, and does not alter any fit/residual source fields.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presubmit_prediction: Option<super::presubmit::PresubmitPredictionReceiptV1>,
    /// Prospective per-wave provenance only; never a calibration membership.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prospective_capture: Option<super::prospective_capture::ProspectiveCaptureReceiptV1>,
    #[serde(serialize_with = "serialize_fingerprint")]
    pub fingerprint: Option<ExecutionFingerprint>,
    #[serde(serialize_with = "wire::serialize_shape")]
    pub actual_shape: Option<WaveExecutionShape>,
    /// Same receipt/call as actual_shape. Independent versioned evidence; not
    /// eligible for legacy profile 1--5 training or inference.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub statistical_evidence: Option<ferrum_interfaces::execution_cost::StatisticalWaveEvidenceV1>,
    /// Capture-only qualification. Never imported by the current model/profile.
    #[serde(skip)]
    pub structured_evidence:
        Option<Result<QualifiedStructuredWaveEvidenceV1, StructuredSettlementUnknown>>,
    #[serde(skip)]
    pub(super) route_evidence: Option<Arc<super::route_population::LiveRouteEvidence>>,
    pub prepare_started_at_ns: Option<u64>,
    pub executor_returned_at_ns: Option<u64>,
    pub rows: Vec<HostRowStageV1>,
    pub finalized_at_ns: Option<u64>,
    pub full_wall_ns: Option<u64>,
    pub completeness: HostStageCompleteness,
    #[serde(skip)]
    pub(super) observation_memory: Option<Arc<super::memory::ObservationBytePermit>>,
}

/// Borrow of the freshly built original stages. Only this producer can create it;
/// downstream diagnostic copies still require validate_host_stages.
pub(super) struct OriginalHostStages<'a>(&'a HostStageEvidenceV1);
impl<'a> OriginalHostStages<'a> {
    pub(super) fn get(self) -> &'a HostStageEvidenceV1 {
        self.0
    }
}

/// Original, fully settled private calibration. Only the host-stage producer
/// can mint this receipt; diagnostic copies and numerical qualifiers cannot.
/// The Arc is shared transiently while the FIFO entry is resolved.
pub(super) struct CompletePrivateCalibrationSettlement {
    stages: Arc<HostStageEvidenceV1>,
    kind: PrivateCalibrationSettlementKind,
}
#[derive(Clone, Copy)]
enum PrivateCalibrationSettlementKind {
    PrefixPreparation,
    StartupReadiness,
}
impl CompletePrivateCalibrationSettlement {
    pub(super) fn prefix_observed_at(&self, stages: &Arc<HostStageEvidenceV1>) -> Option<u64> {
        matches!(
            self.kind,
            PrivateCalibrationSettlementKind::PrefixPreparation
        )
        .then(|| self.observed_at(stages))
        .flatten()
    }
    pub(super) fn readiness_observed_at(&self, stages: &Arc<HostStageEvidenceV1>) -> Option<u64> {
        matches!(
            self.kind,
            PrivateCalibrationSettlementKind::StartupReadiness
        )
        .then(|| self.observed_at(stages))
        .flatten()
    }
    fn observed_at(&self, stages: &Arc<HostStageEvidenceV1>) -> Option<u64> {
        Arc::ptr_eq(&self.stages, stages)
            .then_some(stages.finalized_at_ns)
            .flatten()
    }
}

fn serialize_fingerprint<S: serde::Serializer>(
    value: &Option<ExecutionFingerprint>,
    serializer: S,
) -> Result<S::Ok, S::Error> {
    value
        .as_ref()
        .map(ferrum_scheduler::implementations::continuous::cost_profile::ProfileFingerprint::from)
        .serialize(serializer)
}

impl HostStageEvidenceV1 {
    /// Read-only diagnostic arithmetic over the original private lifecycle's
    /// timestamps. This creates no qualified receipt and changes no wire or
    /// training eligibility. Actor handoff is not network-visible delivery.
    pub fn diagnostic_wall_partition(
        &self,
    ) -> Result<
        ferrum_interfaces::execution_cost::HostWallPartitionV1,
        ferrum_interfaces::execution_cost::HostWallPartitionUnknownV1,
    > {
        use ferrum_interfaces::execution_cost::{
            diagnose_host_wall_partition_v1, HostRowTimesV1, HostWallTimesV1,
        };
        diagnose_host_wall_partition_v1(
            HostWallTimesV1 {
                schema_version: self.schema_version,
                complete_single_wave: self.completeness
                    == HostStageCompleteness::CompleteSingleWave,
                prepare_started_at_ns: self.prepare_started_at_ns,
                executor_returned_at_ns: self.executor_returned_at_ns,
                finalized_at_ns: self.finalized_at_ns,
                full_wall_ns: self.full_wall_ns,
            },
            self.rows.iter().map(|row| HostRowTimesV1 {
                complete_single_wave: row.completeness == HostStageCompleteness::CompleteSingleWave,
                host_processing_ordinal: row.host_processing_ordinal,
                host_started_at_ns: row.host_started_at_ns,
                token_committed_at_ns: row.token_committed_at_ns,
                output_published_at_ns: row.output_published_at_ns,
                completion_started_at_ns: row.completion_started_at_ns,
                settled_at_ns: row.settled_at_ns,
            }),
        )
    }

    /// Explicit calibration/raw-only borrowed view. Legacy source/profile
    /// serialization intentionally omits the capture-only field.
    pub fn structured_diagnostic_view(&self) -> impl Serialize + '_ {
        #[derive(Serialize)]
        struct View<'a> {
            #[serde(flatten)]
            stages: &'a HostStageEvidenceV1,
            #[serde(skip_serializing_if = "Option::is_none")]
            structured_evidence:
                &'a Option<Result<QualifiedStructuredWaveEvidenceV1, StructuredSettlementUnknown>>,
        }
        View {
            stages: self,
            structured_evidence: &self.structured_evidence,
        }
    }
    /// The original source protocol's borrowed fields. Diagnostic prediction
    /// receipts remain on this object and never become source membership.
    pub(super) fn source_view(&self) -> impl Serialize + '_ {
        wire::SourceEvidence::new(self)
    }

    /// Qualified source evidence, without unrelated diagnostic sidecars. This
    /// preserves the existing source schema and the original settlement proof.
    pub(super) fn structured_source_view(&self) -> impl Serialize + '_ {
        #[derive(Serialize)]
        struct View<'a, S: Serialize> {
            #[serde(flatten)]
            stages: S,
            #[serde(skip_serializing_if = "Option::is_none")]
            structured_evidence:
                &'a Option<Result<QualifiedStructuredWaveEvidenceV1, StructuredSettlementUnknown>>,
        }
        View {
            stages: self.source_view(),
            structured_evidence: &self.structured_evidence,
        }
    }
    pub(super) fn structured_retained_overhead_bytes(&self) -> usize {
        let copies = usize::from(
            self.statistical_evidence
                .as_ref()
                .is_some_and(|value| value.structured_retained_rows() != 0),
        ) + usize::from(self.structured_evidence.as_ref().is_some_and(Result::is_ok));
        copies
            * (std::mem::size_of::<
                ferrum_interfaces::execution_cost::UnsettledStructuredWaveEvidenceV1,
            >() + 2 * std::mem::size_of::<usize>())
    }

    /// Actual retained capacities, shared with the existing bounded FIFO.
    pub(in crate::continuous_engine) fn retained_rows(&self) -> Option<usize> {
        let mut rows = self.rows.capacity();
        // These Arcs may share storage, but charge each retained reference
        // conservatively so FIFO/export bounds cannot miss a backing Vec.
        rows = rows.checked_add(
            self.statistical_evidence
                .as_ref()
                .map_or(0, |value| value.structured_retained_rows()),
        )?;
        rows = rows.checked_add(
            self.structured_evidence
                .as_ref()
                .and_then(|value| value.as_ref().ok())
                .map_or(0, |value| value.retained_rows()),
        )?;
        if let Some(shape) = &self.actual_shape {
            rows = rows.checked_add(shape.decode_kv_tokens.capacity())?;
            rows = rows.checked_add(shape.prefill_chunks.capacity())?;
            rows = rows.checked_add(
                shape
                    .numeric_features
                    .as_ref()
                    .map_or(0, |features| features.rows.capacity()),
            )?;
            rows = rows.checked_add(
                shape
                    .row_multiset_features
                    .as_ref()
                    .map_or(0, |features| features.rows.capacity()),
            )?;
        }
        rows = rows.checked_add(
            self.route_evidence
                .as_ref()
                .map_or(0, |r| r.retained_rows()),
        )?;
        Some(rows)
    }
}

#[derive(Default)]
pub(super) struct HostRowProgress {
    ordinal: Option<u32>,
    started: Option<u64>,
    committed: Option<u64>,
    published: Option<u64>,
    completion_started: Option<u64>,
    settled: Option<u64>,
    terminal: Option<HostTerminalStageV1>,
    work: Option<HostCommittedWork>,
    status: Option<HostStageCompleteness>,
}

impl HostRowProgress {
    pub(super) fn is_pristine(&self) -> bool {
        self.ordinal.is_none()
            && self.started.is_none()
            && self.committed.is_none()
            && self.published.is_none()
            && self.completion_started.is_none()
            && self.settled.is_none()
            && self.terminal.is_none()
            && self.work.is_none()
            && self.status.is_none()
    }

    pub(super) fn committed_work(&self) -> Option<HostCommittedWork> {
        self.work
    }
}

/// Call-local correlation only. It owns no output/KV/scheduler authority.
pub(in crate::continuous_engine) struct PendingHostRow {
    fence: Arc<()>,
    clock: Arc<dyn CostObservationClock>,
    request_id: RequestId,
    owner: u64,
    generation: u64,
    input_index: u32,
    progress: HostRowProgress,
}

/// Non-Clone, single-use receipt. Its constructor consumes the removed owner.
pub(in crate::continuous_engine) struct HostSettledReceipt {
    pending: PendingHostRow,
}

impl PendingHostRow {
    pub(in crate::continuous_engine) fn terminal_handed_off(&mut self) {
        self.progress.published = self.clock.now_ns();
    }
    pub(in crate::continuous_engine) fn matches_owner(&self, sequence: &SequenceState) -> bool {
        sequence.request_id == self.request_id
            && sequence.cost_frontier.is_some_and(|frontier| {
                frontier.owner_incarnation.get() == self.owner
                    && self.generation.checked_add(1) == Some(frontier.work_generation.get())
            })
    }

    pub(in crate::continuous_engine) fn settle(
        mut self,
        sequence: SequenceState,
        mut terminal: HostTerminalStageV1,
    ) -> HostSettledReceipt {
        terminal.owner_matched &= self.matches_owner(&sequence);
        terminal.owner_matched &= self.progress.work.is_some_and(|work| {
            let generated_after = match work {
                HostCommittedWork::Decode {
                    generated_tokens_after,
                    ..
                }
                | HostCommittedWork::Prefill {
                    generated_tokens_after,
                    ..
                } => generated_tokens_after,
            };
            generated_after == terminal.generated_tokens
        });
        // The final port/owner drop is part of this boundary, not a later task.
        drop(sequence);
        self.progress.settled = self.clock.now_ns();
        let terminal_status = if !terminal.owner_matched {
            HostStageCompleteness::IdentityMismatch
        } else if terminal.output_failed
            || terminal.physical_failed
            || terminal.scheduler_failed
            || !terminal.terminal_handoff_succeeded
            || !terminal.request_slot_closed
        {
            HostStageCompleteness::Failed
        } else if terminal.pending_restore_removed
            || terminal.other_physical_resources
            || terminal.admission_cancellation_work != ExecutorCompletionWork::NoAdditionalWork
            || terminal.cache_completion_work != ExecutorCompletionWork::NoAdditionalWork
        {
            HostStageCompleteness::AdditionalOrUnknownWork
        } else {
            HostStageCompleteness::CompleteSingleWave
        };
        if self
            .progress
            .status
            .is_none_or(|status| status == HostStageCompleteness::CompleteSingleWave)
        {
            self.progress.status = Some(terminal_status);
        }
        self.progress.terminal = Some(terminal);
        HostSettledReceipt { pending: self }
    }
}

impl EngineCostCall {
    pub(super) fn record_host_stage_queue(&self, result: &Result<u64, CostSampleDrop>) {
        if let Some(capture) = &self.prospective_capture {
            capture.queue_result(result);
        }
        if let Some(capture) = &self.calibration_capture {
            if capture.host_stages().is_some() {
                capture.complete_host_stage_queue(match result {
                    Ok(ordinal) => HostStageQueueReceipt {
                        accepted_ordinal: Some(*ordinal),
                        disposition: HostStageQueueDisposition::Published,
                    },
                    Err(CostSampleDrop::Capacity) => HostStageQueueReceipt {
                        accepted_ordinal: None,
                        disposition: HostStageQueueDisposition::DroppedCapacity,
                    },
                    Err(CostSampleDrop::Contended) => HostStageQueueReceipt {
                        accepted_ordinal: None,
                        disposition: HostStageQueueDisposition::DroppedContended,
                    },
                    Err(CostSampleDrop::WorkerStopped) => HostStageQueueReceipt {
                        accepted_ordinal: None,
                        disposition: HostStageQueueDisposition::DroppedWorkerStopped,
                    },
                });
            }
        }
    }
    pub(super) fn note_host_failure(&mut self, request_id: &RequestId) {
        if let Some(index) = self
            .participants
            .iter()
            .position(|row| &row.request_id == request_id)
        {
            self.host_stages[index].status = Some(HostStageCompleteness::Failed);
        }
    }
    pub(in crate::continuous_engine) fn begin_host_row(&mut self, request_id: &RequestId) {
        let Some(index) = self
            .participants
            .iter()
            .position(|row| &row.request_id == request_id)
        else {
            return;
        };
        let ordinal = self.host_processing_ordinal;
        self.host_processing_ordinal = self
            .host_processing_ordinal
            .checked_add(1)
            .unwrap_or(u32::MAX);
        let progress = &mut self.host_stages[index];
        if progress.ordinal.is_some() {
            progress.status = Some(HostStageCompleteness::IdentityMismatch);
            return;
        }
        progress.ordinal = Some(ordinal);
        progress.started = self.clock.now_ns();
    }

    pub(in crate::continuous_engine) fn note_host_token_commit(
        &mut self,
        evidence: &HostCommitEvidence,
    ) {
        let Some(index) = self.host_stage_index(evidence) else {
            return;
        };
        let progress = &mut self.host_stages[index];
        if progress.work.is_some() {
            progress.status = Some(HostStageCompleteness::IdentityMismatch);
            return;
        }
        if let HostCommitOutcome::Committed(work) = evidence.outcome {
            progress.work = Some(work);
            progress.committed = self.clock.now_ns();
        } else {
            progress.status = Some(HostStageCompleteness::Failed);
        }
    }

    fn host_stage_index(&self, evidence: &HostCommitEvidence) -> Option<usize> {
        self.participants.iter().position(|row| {
            row.request_id == evidence.request_id
                && row.owner_incarnation == evidence.owner_incarnation
                && row.work_generation == evidence.work_generation
                && row.input_index == evidence.input_index
        })
    }

    pub(super) fn host_publication(
        &mut self,
        evidence: &HostCommitEvidence,
        terminal: bool,
        actor_handoff: bool,
    ) -> Option<PendingHostRow> {
        let index = self.host_stage_index(evidence)?;
        if terminal {
            let mut progress = std::mem::take(&mut self.host_stages[index]);
            progress.completion_started = self.clock.now_ns();
            // Preserve a pending footprint if the future is cancelled before
            // the owned completion can return its single-use receipt.
            self.host_stages[index] = HostRowProgress {
                ordinal: progress.ordinal,
                started: progress.started,
                committed: progress.committed,
                completion_started: progress.completion_started,
                work: progress.work,
                status: progress.status,
                ..HostRowProgress::default()
            };
            Some(PendingHostRow {
                fence: Arc::clone(&self.host_fence),
                clock: Arc::clone(&self.clock),
                request_id: evidence.request_id.clone(),
                owner: evidence.owner_incarnation,
                generation: evidence.work_generation,
                input_index: evidence.input_index,
                progress,
            })
        } else {
            let progress = &mut self.host_stages[index];
            let now = self.clock.now_ns();
            progress.published = actor_handoff.then_some(now).flatten();
            progress.settled = now;
            let produced_token = match evidence.outcome {
                HostCommitOutcome::Committed(HostCommittedWork::Decode { .. }) => true,
                HostCommitOutcome::Committed(HostCommittedWork::Prefill {
                    generated_tokens_before,
                    generated_tokens_after,
                    ..
                }) => generated_tokens_after > generated_tokens_before,
                _ => false,
            };
            if produced_token && !actor_handoff {
                progress
                    .status
                    .get_or_insert(HostStageCompleteness::AdditionalOrUnknownWork);
            }
            progress
                .status
                .get_or_insert(HostStageCompleteness::CompleteSingleWave);
            None
        }
    }

    pub(in crate::continuous_engine) fn record_settled(&mut self, receipt: HostSettledReceipt) {
        let pending = receipt.pending;
        if !Arc::ptr_eq(&pending.fence, &self.host_fence) {
            self.reject(CostCallRejection::FrontierMismatch);
            return;
        }
        let Some(index) = self.participants.iter().position(|row| {
            row.request_id == pending.request_id
                && row.owner_incarnation == pending.owner
                && row.work_generation == pending.generation
                && row.input_index == pending.input_index
        }) else {
            self.reject(CostCallRejection::FrontierMismatch);
            return;
        };
        if self.host_stages[index].settled.is_some() {
            self.reject(CostCallRejection::HostDuplicate);
            return;
        }
        self.host_stages[index] = pending.progress;
    }

    pub(super) fn make_host_stages(&self) -> Option<Arc<HostStageEvidenceV1>> {
        self.make_host_stages_with_preparation()
            .map(|(stages, _)| stages)
    }

    pub(super) fn make_host_stages_with_preparation(
        &self,
    ) -> Option<(
        Arc<HostStageEvidenceV1>,
        Option<CompletePrivateCalibrationSettlement>,
    )> {
        let mut route_settlement_gate = None;
        let result = self.build_host_stages_with_preparation(&mut route_settlement_gate);
        if self.rejection == Some(CostCallRejection::CalibrationPreparation)
            && result.as_ref().is_none_or(|(_, proof)| proof.is_none())
        {
            self.diagnose_unclassified_preparation(
                result.as_ref().map(|(stages, _)| stages.as_ref()),
                route_settlement_gate,
            );
        }
        result
    }

    // Worker-side failure-only diagnostics. Fixed scalar fields also cover
    // early stage-construction failure; no extra clock read, shape serialization
    // or numerical qualification is performed, and successful waves are silent.
    fn diagnose_unclassified_preparation(
        &self,
        stages: Option<&HostStageEvidenceV1>,
        route_settlement_gate: Option<&'static str>,
    ) {
        if !tracing::enabled!(
            target: "ferrum_engine::continuous_engine::inner::cost_observation::runtime",
            tracing::Level::WARN
        ) {
            return;
        }
        let wave = self.recorder.observations().first();
        let actual = wave.and_then(|wave| wave.shape.as_ref());
        // Present only when the original producer opted into DEBUG before
        // submission. Reading these scalar copies cannot recover authority.
        let route_diagnostic = self.recorder.route_diagnostic();
        let first_incomplete_row = stages.and_then(|stages| {
            stages.rows.iter().find(|row| {
                row.completeness != HostStageCompleteness::CompleteSingleWave
                    || row.terminal.is_some()
                    || row.completion_started_at_ns.is_some()
            })
        });
        tracing::warn!(
            target: "ferrum_engine::continuous_engine::inner::cost_observation::runtime",
            event = "calibration_preparation_feedback_unclassified_v1",
            call_id = self.call_id.get(),
            source_generation = self.source_generation,
            rejection = ?self.rejection,
            stage_rejection = ?self.stage_rejection,
            calibration_capture = self.calibration_capture.is_some(),
            original_route_capture = self.calibration_capture.as_ref()
                .is_some_and(|capture| capture.requests_original_route()),
            route_prepared = ?route_diagnostic.and_then(|value| value.prepared),
            route_submitted = ?route_diagnostic.and_then(|value| value.submitted),
            route_first_rejection = ?route_diagnostic.and_then(|value| value.first_rejection),
            route_settlement_gate,
            live_ticket = self.live_ticket.is_some(),
            dispatch_waves = self.dispatch.waves,
            retained_waves = self.recorder.observations().len(),
            dispatch_outcome = ?self.dispatch.outcome,
            dispatch_unknown = ?self.dispatch.unknown,
            call_boundary = ?self.boundary,
            wave_outcome = ?wave.and_then(|wave| wave.outcome),
            wave_boundary = ?wave.map(|wave| wave.boundary),
            path = ?actual.map(|actual| actual.path),
            row_order = ?actual.map(|actual| actual.row_order),
            additional_work = ?actual.map(|actual| (
                actual.restore_bytes, actual.maintenance_bytes, actual.maintenance_units,
            )),
            stage_evidence_available = stages.is_some(),
            stage_completeness = ?stages.map(|stages| stages.completeness),
            fingerprint_available = stages.is_some_and(|stages| stages.fingerprint.is_some()),
            actual_shape_available = stages.is_some_and(|stages| stages.actual_shape.is_some()),
            full_wall_ns = ?stages.and_then(|stages| stages.full_wall_ns),
            prepare_started_at_ns = ?self.prepare_started_at_ns,
            executor_returned_at_ns = ?self.dispatch.returned_at_ns,
            finalized_at_ns = ?stages.and_then(|stages| stages.finalized_at_ns),
            participants = self.participants.len(),
            stage_rows = stages.map_or(0, |stages| stages.rows.len()),
            host_rows_started = self.host_stages.iter().filter(|row| row.ordinal.is_some()).count(),
            first_incomplete_row = ?first_incomplete_row.map(|row| (
                row.input_index, row.completeness,
                row.terminal.is_some(), row.completion_started_at_ns.is_some(),
            )),
            "private calibration preparation lacks complete original feedback exclusion proof"
        );
    }

    fn build_host_stages_with_preparation(
        &self,
        route_settlement_gate: &mut Option<&'static str>,
    ) -> Option<(
        Arc<HostStageEvidenceV1>,
        Option<CompletePrivateCalibrationSettlement>,
    )> {
        if self.host_stages.iter().all(|row| row.ordinal.is_none()) {
            return None;
        }
        let wave = self.recorder.observations().first()?;
        let shape = wave.shape.as_ref();
        let route_evidence = self.make_route_evidence_recording_failure(route_settlement_gate);
        let actual_rows = match shape {
            Some(shape) => shape.rows.as_slice(),
            None if route_evidence.as_ref().is_some_and(|r| r.is_outside()) => {
                self.route_for_settlement()?.rows()
            }
            None => return None,
        };
        let mut rows = Vec::new();
        rows.try_reserve_exact(actual_rows.len()).ok()?;
        let mut completeness = if self.dispatch.waves == 1
            && self.recorder.observations().len() == 1
            && self.dispatch.outcome == Some(ObservedCallOutcome::Completed)
            && wave.outcome == Some(ActualWaveOutcome::Completed)
            && self.boundary == WaveObservationBoundary::IsolatedPreparationToCommit
            && wave.boundary == WaveObservationBoundary::IsolatedPreparationToCommit
            && (self.dispatch.unknown.is_none()
                || route_evidence.as_ref().is_some_and(|r| r.is_outside()))
        {
            HostStageCompleteness::CompleteSingleWave
        } else {
            HostStageCompleteness::AdditionalOrUnknownWork
        };
        let same_rows = actual_rows.len() == self.participants.len()
            && actual_rows.iter().enumerate().all(|(index, actual)| {
                actual_rows[..index].iter().all(|prior| {
                    prior.request_id != actual.request_id && prior.input_index != actual.input_index
                }) && self.participants.iter().any(|row| {
                    row.request_id == actual.request_id
                        && row.owner_incarnation == actual.owner_incarnation
                        && row.work_generation == actual.work_generation
                        && row.input_index == actual.input_index
                })
            });
        if !same_rows {
            completeness = HostStageCompleteness::IdentityMismatch;
        }
        if let Some(rejection) = self.stage_rejection {
            completeness = match rejection {
                CostCallRejection::HostFailed
                | CostCallRejection::HostCancelled
                | CostCallRejection::ExecutorFailed
                | CostCallRejection::Abandoned => HostStageCompleteness::Failed,
                CostCallRejection::Clock | CostCallRejection::InvalidWall => {
                    HostStageCompleteness::InvalidClock
                }
                _ => HostStageCompleteness::IdentityMismatch,
            };
        }
        let finalized = self.observation_time();
        let mut latest = self.prepare_started_at_ns;
        for actual in actual_rows {
            let progress = self
                .participants
                .iter()
                .position(|row| {
                    row.request_id == actual.request_id
                        && row.owner_incarnation == actual.owner_incarnation
                        && row.work_generation == actual.work_generation
                        && row.input_index == actual.input_index
                })
                .map(|index| &self.host_stages[index]);
            let mut status = progress
                .and_then(|row| row.status)
                .unwrap_or(HostStageCompleteness::MissingEvidence);
            if let Some(progress) = progress {
                if status != HostStageCompleteness::Failed
                    && progress
                        .work
                        .is_none_or(|work| super::sample::validate_work(actual.work, work).is_err())
                {
                    status = HostStageCompleteness::IdentityMismatch;
                }
                let times = [
                    self.prepare_started_at_ns,
                    self.dispatch.returned_at_ns,
                    progress.started,
                    progress.committed,
                    progress.settled,
                    finalized,
                ];
                if matches!(
                    status,
                    HostStageCompleteness::CompleteSingleWave
                        | HostStageCompleteness::AdditionalOrUnknownWork
                ) && (times.iter().any(Option::is_none)
                    || times.windows(2).any(|pair| pair[0] > pair[1])
                    || progress.published.is_some_and(|at| {
                        progress.committed.is_none_or(|commit| at < commit)
                            || progress.settled.is_none_or(|settled| at > settled)
                    })
                    || (status == HostStageCompleteness::CompleteSingleWave
                        && match actual.work {
                            ActualRowWork::Decode { .. } => progress.published.is_none(),
                            ActualRowWork::Prefill {
                                offset,
                                count,
                                total_prompt_tokens,
                            } => {
                                if offset.checked_add(count) == Some(total_prompt_tokens) {
                                    progress.published.is_none()
                                } else {
                                    progress.published.is_some()
                                }
                            }
                            _ => true,
                        })
                    || progress.completion_started.is_some_and(|at| {
                        progress.committed.is_none_or(|commit| at < commit)
                            || progress.settled.is_none_or(|settled| at > settled)
                    }))
                {
                    status = HostStageCompleteness::InvalidClock;
                }
                if let Some(settled) = progress.settled {
                    latest = Some(latest.map_or(settled, |at| at.max(settled)));
                }
            }
            if status != HostStageCompleteness::CompleteSingleWave
                && completeness == HostStageCompleteness::CompleteSingleWave
            {
                completeness = status;
            }
            rows.push(HostRowStageV1 {
                request_id: actual.request_id.clone(),
                owner_incarnation: actual.owner_incarnation,
                work_generation: actual.work_generation,
                input_index: actual.input_index,
                actual_work: match actual.work {
                    ActualRowWork::Decode { kv_tokens } => HostStageWork::Decode { kv_tokens },
                    ActualRowWork::Prefill {
                        offset,
                        count,
                        total_prompt_tokens,
                    } => HostStageWork::Prefill {
                        offset,
                        count,
                        total_prompt_tokens,
                    },
                    ActualRowWork::Restore => HostStageWork::Restore,
                    ActualRowWork::Maintenance => HostStageWork::Maintenance,
                },
                host_processing_ordinal: progress.and_then(|row| row.ordinal),
                host_started_at_ns: progress.and_then(|row| row.started),
                token_committed_at_ns: progress.and_then(|row| row.committed),
                output_published_at_ns: progress.and_then(|row| row.published),
                completion_started_at_ns: progress.and_then(|row| row.completion_started),
                settled_at_ns: progress.and_then(|row| row.settled),
                terminal: progress.and_then(|row| row.terminal.clone()),
                completeness: status,
            });
        }
        // Verify observed serial host order without reordering the physical rows.
        let ordinal_coverage = rows.iter().all(|row| {
            row.host_processing_ordinal
                .is_some_and(|ordinal| (ordinal as usize) < rows.len())
        }) && rows.iter().enumerate().all(|(index, row)| {
            rows[..index]
                .iter()
                .all(|prior| prior.host_processing_ordinal != row.host_processing_ordinal)
        });
        if !ordinal_coverage || !same_rows {
            completeness = HostStageCompleteness::IdentityMismatch;
        }
        for row in &rows {
            if let Some(prior) = rows.iter().find(|prior| {
                prior.host_processing_ordinal.and_then(|n| n.checked_add(1))
                    == row.host_processing_ordinal
            }) {
                if completeness == HostStageCompleteness::CompleteSingleWave
                    && prior
                        .settled_at_ns
                        .zip(row.host_started_at_ns)
                        .is_none_or(|(end, start)| end > start)
                {
                    completeness = HostStageCompleteness::InvalidClock;
                }
            }
        }
        let full_wall_ns = (completeness == HostStageCompleteness::CompleteSingleWave)
            .then(|| latest?.checked_sub(self.prepare_started_at_ns?))
            .flatten()
            .filter(|ns| *ns > 0);
        let fingerprint = match &self.identity {
            ExecutorCostIdentityAvailability::Known(identity) => Some(ExecutionFingerprint {
                model_weights: identity.model_weights,
                numerical_policy: identity.numerical_policy,
                device_runtime: identity.device_runtime,
                execution_config: identity.execution_config,
            }),
            _ => None,
        };
        let mut stages = HostStageEvidenceV1 {
            observation_memory: self.observation_memory.clone(),
            schema_version: 1,
            call_id: self.call_id.get(),
            presubmit_prediction: shape
                .and_then(|shape| self.presubmit_prediction.as_ref().map(|p| p.receipt(shape))),
            prospective_capture: None,
            fingerprint,
            actual_shape: shape.and_then(|shape| super::sample::scheduler_shape(shape).ok()),
            statistical_evidence: shape.and_then(|shape| {
                shape
                    .statistical_evidence
                    .as_ref()
                    .filter(|evidence| evidence.validate_actual(shape).is_ok())
                    .cloned()
            }),
            structured_evidence: None,
            route_evidence,
            prepare_started_at_ns: self.prepare_started_at_ns,
            executor_returned_at_ns: self.dispatch.returned_at_ns,
            rows,
            finalized_at_ns: finalized,
            full_wall_ns,
            completeness,
        };
        if let Some(shape) = shape.filter(|_| self.structured_capture) {
            stages.structured_evidence = Some(structured::qualify(self, shape, &stages));
        }
        stages.prospective_capture = shape.and_then(|shape| {
            self.prospective_capture
                .as_ref()
                .map(|capture| capture.receipt(shape, OriginalHostStages(&stages)))
        });
        let stages = Arc::new(stages);
        // Private calibration exclusion comes from this original physical
        // recorder and complete host settlement, never a public reason alone.
        // Prefix intervention retains its stricter no-terminal rule below.
        let complete_private = self.stage_rejection.is_none()
            && self.live_ticket.is_none()
            && self
                .calibration_capture
                .as_ref()
                .is_some_and(|capture| capture.requests_original_route())
            && stages.completeness == HostStageCompleteness::CompleteSingleWave
            && stages.fingerprint.is_some()
            && stages.actual_shape.is_some()
            && stages.full_wall_ns.is_some()
            && !stages.rows.is_empty()
            && stages
                .rows
                .iter()
                .all(|row| row.completeness == HostStageCompleteness::CompleteSingleWave)
            && self.dispatch.unknown.is_none()
            && shape.is_some_and(|actual| {
                actual.path == ActualWavePath::PlanRuntime
                    && actual.row_order == ActualWaveRowOrder::Ordered
                    && actual.restore_bytes == 0
                    && actual.maintenance_bytes == 0
                    && actual.maintenance_units == 0
            });
        let kind = if complete_private
            && self.rejection == Some(CostCallRejection::CalibrationPreparation)
            && stages
                .rows
                .iter()
                .all(|row| row.terminal.is_none() && row.completion_started_at_ns.is_none())
        {
            Some(PrivateCalibrationSettlementKind::PrefixPreparation)
        } else if complete_private
            && matches!(self.rejection, None | Some(CostCallRejection::Composite))
            && self
                .calibration_capture
                .as_ref()
                .is_some_and(|capture| capture.requests_startup_readiness())
        {
            // Unlike a prefix intervention, ordinary readiness may complete a
            // request. CompleteSingleWave above is minted only after the
            // original PendingHostRow consumes its owner, output handoff and
            // no-additional-work completion receipt. Terminal presence alone
            // never establishes this proof; failed/partial/unknown work fails
            // the shared gate. The resolver accepts only a Completed sample or
            // the legacy Composite marker used by actual terminal publication.
            Some(PrivateCalibrationSettlementKind::StartupReadiness)
        } else {
            None
        };
        let preparation = kind.map(|kind| CompletePrivateCalibrationSettlement {
            stages: Arc::clone(&stages),
            kind,
        });
        Some((stages, preparation))
    }
}
