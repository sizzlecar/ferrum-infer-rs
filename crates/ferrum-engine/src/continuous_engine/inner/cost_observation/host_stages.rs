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
mod wire;
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
    #[serde(serialize_with = "serialize_fingerprint")]
    pub fingerprint: Option<ExecutionFingerprint>,
    #[serde(serialize_with = "wire::serialize_shape")]
    pub actual_shape: Option<WaveExecutionShape>,
    /// Same receipt/call as actual_shape. Independent versioned evidence; not
    /// eligible for legacy profile 1--5 training or inference.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub statistical_evidence: Option<ferrum_interfaces::execution_cost::StatisticalWaveEvidenceV1>,
    pub prepare_started_at_ns: Option<u64>,
    pub executor_returned_at_ns: Option<u64>,
    pub rows: Vec<HostRowStageV1>,
    pub finalized_at_ns: Option<u64>,
    pub full_wall_ns: Option<u64>,
    pub completeness: HostStageCompleteness,
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
    /// Actual retained capacities, shared with the existing bounded FIFO.
    pub(in crate::continuous_engine) fn retained_rows(&self) -> Option<usize> {
        let mut rows = self.rows.capacity();
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
        if self.host_stages.iter().all(|row| row.ordinal.is_none()) {
            return None;
        }
        let wave = self.recorder.observations().first()?;
        let shape = wave.shape.as_ref()?;
        let mut rows = Vec::new();
        rows.try_reserve_exact(shape.rows.len()).ok()?;
        let mut completeness = if self.dispatch.waves == 1
            && self.recorder.observations().len() == 1
            && self.dispatch.outcome == Some(ObservedCallOutcome::Completed)
            && wave.outcome == Some(ActualWaveOutcome::Completed)
            && self.boundary == WaveObservationBoundary::IsolatedPreparationToCommit
            && wave.boundary == WaveObservationBoundary::IsolatedPreparationToCommit
            && self.dispatch.unknown.is_none()
        {
            HostStageCompleteness::CompleteSingleWave
        } else {
            HostStageCompleteness::AdditionalOrUnknownWork
        };
        let same_rows = shape.rows.len() == self.participants.len()
            && shape.rows.iter().enumerate().all(|(index, actual)| {
                shape.rows[..index].iter().all(|prior| {
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
        let finalized = self.clock.now_ns();
        let mut latest = self.prepare_started_at_ns;
        for actual in &shape.rows {
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
        Some(Arc::new(HostStageEvidenceV1 {
            schema_version: 1,
            call_id: self.call_id.get(),
            fingerprint,
            actual_shape: super::sample::scheduler_shape(shape).ok(),
            statistical_evidence: shape
                .statistical_evidence
                .as_ref()
                .filter(|evidence| evidence.validate_actual(shape).is_ok())
                .cloned(),
            prepare_started_at_ns: self.prepare_started_at_ns,
            executor_returned_at_ns: self.dispatch.returned_at_ns,
            rows,
            finalized_at_ns: finalized,
            full_wall_ns,
            completeness,
        }))
    }
}
