//! Fair completion uses real owners/resources without asserting cost coverage.
//! The complete queue remains sealed; only executable rows acquire a work view.
use super::*;
use crate::continuous_engine::output_flow_runtime::OutputPlanningCreditView;
use ferrum_interfaces::execution_cost::{
    ActualRowWork, ActualWaveKind, CompletionOnlyReason, ExpectedExecutionWave, ExpectedWaveInput,
    ExpectedWaveWork, ExpectedWorkSelection,
};

struct CompletionSelection {
    proof: ControllerSafetyProof,
    resources: ResourcePlanningView,
    selected: Vec<PlanningWorkSelection>,
    expected: ExpectedExecutionWave,
}

/// A captured runnable wave that cannot be published needs a future snapshot,
/// including early returns caused by contention between the two captures.
struct PublicationRetry<'a> {
    engine: &'a EngineInner,
    armed: bool,
}
impl Drop for PublicationRetry<'_> {
    fn drop(&mut self) {
        if self.armed {
            self.engine
                .arm_controller_retry(retry::ControllerRetryReason::ChangedEvidence);
        }
    }
}

impl EngineInner {
    pub(super) fn completion_allowed(&self) -> bool {
        self.config.scheduler.slo.mode == ferrum_types::SloMode::Enforce
            && self.config.scheduler.slo.admission.time_policy
                == ferrum_types::SloTimeAdmissionPolicy::CompleteRequests
    }

    /// Remember exhaustion/expiry until a fresh bounded iteration. Otherwise a
    /// full unsuccessful search could consume every subsequent fallback budget.
    pub(super) fn request_completion_turn(&self, reason: CompletionOnlyReason) {
        if self.completion_allowed() {
            self.slo_controller.lock().completion_next = Some(reason);
            self.arm_controller_retry(retry::ControllerRetryReason::ChangedEvidence);
        }
    }

    /// Continue an already chosen completion attempt after a typed deferral
    /// proved it submitted nothing. This is a scheduling intent, not cached
    /// work, cost, or feasibility evidence: the next publication captures the
    /// whole current queue again and must pass its ordinary physical guards.
    /// Readiness waiters and capacity maintenance own their existing wakes;
    /// adding a retry timer here would delay maintenance or poll a blocked owner.
    pub(super) fn continue_deferred_completion(&self, work: &owner::ControllerWork) {
        if self.completion_allowed()
            && matches!(
                work.timing,
                owner::ControllerTimingCommitment::CompleteRequests
            )
        {
            if let ferrum_interfaces::execution_cost::WaveCommitment::CompleteRequests(reason) =
                work.expected.commitment()
            {
                self.slo_controller.lock().completion_next = Some(*reason);
            }
        }
    }

    pub(super) fn prepare_completion_controller(
        &self,
        hint: &ferrum_interfaces::BatchHint,
        budget: &Arc<ControllerBudget>,
        reason: CompletionOnlyReason,
    ) -> Result<SloIterationPlan> {
        if !self.completion_allowed() {
            return Ok(SloIterationPlan::Idle);
        }
        self.prepare_completion_selection(
            hint,
            budget,
            reason,
            None,
            NonZeroUsize::new(4096).unwrap(),
        )
    }

    pub(super) fn prepare_exact_calibration_selection(
        &self,
        hint: &ferrum_interfaces::BatchHint,
        budget: &Arc<ControllerBudget>,
        rows: &[super::super::calibration::CalibrationWork],
        maximum_requests: NonZeroUsize,
    ) -> Result<SloIterationPlan> {
        if !self.manual_calibration_driver
            || self.config.scheduler.slo.mode != ferrum_types::SloMode::Observe
        {
            return Err(FerrumError::invalid_request(
                "exact calibration requires its exclusive manual driver",
            ));
        }
        self.prepare_completion_selection(
            hint,
            budget,
            CompletionOnlyReason::CostUnavailable,
            Some(rows),
            maximum_requests,
        )
    }

    fn prepare_completion_selection(
        &self,
        hint: &ferrum_interfaces::BatchHint,
        budget: &Arc<ControllerBudget>,
        reason: CompletionOnlyReason,
        exact: Option<&[super::super::calibration::CalibrationWork]>,
        maximum_requests: NonZeroUsize,
    ) -> Result<SloIterationPlan> {
        self.slo_controller.lock().completion_next = Some(reason);
        let selected = match {
            let _stage = budget.stage(ControllerStage::Capture);
            self.capture_completion_selection(
                hint,
                Arc::clone(budget),
                reason,
                exact,
                maximum_requests,
            )
        } {
            Ok(selected) => selected,
            Err(error) => {
                if let Some(reason) = error.retry {
                    self.arm_controller_retry(reason);
                }
                budget.record_unavailable(error.reason);
                self.record_controller(ControllerObservation {
                    obligations: error.obligations,
                    disposition: if exact.is_some() {
                        "calibration_blocked"
                    } else {
                        "completion_wait"
                    },
                    reason: error.reason,
                });
                return Ok(SloIterationPlan::Idle);
            }
        };
        let mut retry = PublicationRetry {
            engine: self,
            armed: true,
        };
        let _stage = budget.stage(ControllerStage::Publication);
        let requests: Vec<_> = selected
            .proof
            .fences
            .iter()
            .map(|fence| ExecutorResourcePlanningRequest {
                request_id: &fence.key.request_id,
                cache_id: fence.resource_cache_id(),
            })
            .collect();
        if !matches!(
            self.model_executor
                .revalidate_execution_resource_planning_view(
                    &requests,
                    &selected.resources,
                    &mut || budget.poll()
                ),
            ResourcePlanningAvailability::Known(true)
        ) {
            self.arm_controller_retry(retry::ControllerRetryReason::ChangedEvidence);
            return Ok(SloIterationPlan::Idle);
        }
        drop(requests);
        let Some(mut availability) = self.dynamic_admission_availability.try_lock() else {
            return Ok(SloIterationPlan::Idle);
        };
        let Some(epochs) = self
            .model_executor
            .write_execution_capacity_snapshot(&mut availability)?
        else {
            return Ok(SloIterationPlan::Idle);
        };
        if !budget.poll() || !self.completion_frontiers_match(&selected.proof) {
            return Ok(SloIterationPlan::Idle);
        }
        if exact.is_none()
            && !self.completion_work_policy_matches(&selected.proof, &selected.selected, hint)
        {
            return Ok(SloIterationPlan::Idle);
        }
        let published = self.scheduler.try_select_planned_wave(
            &selected.proof.queue,
            &selected.selected,
            hint,
            AdmissionWakeSnapshot::new(
                AdmissionWakeEpochs::new(
                    epochs.coordinator_id,
                    epochs.release_epoch,
                    epochs.capacity_epoch,
                    0,
                ),
                &availability,
            ),
        );
        drop(availability);
        let (batch, receipt) = match published {
            Ok(PlanningSelectionOutcome::Published { batch, receipt }) => (batch, receipt),
            Ok(_) | Err(PlanningStateUnavailable::Busy) => return Ok(SloIterationPlan::Idle),
            Err(error) => {
                return Err(FerrumError::scheduler(format!(
                    "publish completion wave: {error:?}"
                )))
            }
        };
        if !budget.poll() {
            self.keep_unsubmitted_publication(receipt)?;
            return Ok(SloIterationPlan::Idle);
        }
        let reserved = match self.reserve_batch_output(&batch) {
            Ok(reserved) => reserved,
            Err(error) => {
                self.finish_batch_output(&batch, true);
                self.keep_unsubmitted_publication(receipt)?;
                return Err(error);
            }
        };
        if !reserved || !budget.poll() {
            self.finish_batch_output(&batch, true);
            self.keep_unsubmitted_publication(receipt)?;
            return Ok(SloIterationPlan::Idle);
        }
        {
            let mut state = self.slo_controller.lock();
            state.completion_next = None;
            // Rotate all attempted participants, including a later physical
            // deferral. A permanently blocked peer cannot monopolize the head.
            for row in selected.expected.work().participants() {
                let row = row.selection();
                let key = (row.request_id.clone(), row.owner_incarnation.get());
                state.completion_order.retain(|old| old != &key);
                state.completion_order.push_back(key);
            }
        }
        self.record_controller(ControllerObservation {
            obligations: selected.proof.queue.requests().len(),
            disposition: if exact.is_some() {
                "calibration"
            } else {
                "complete_requests"
            },
            reason: match reason {
                CompletionOnlyReason::CostUnavailable => "cost_unavailable",
                CompletionOnlyReason::WitnessExpired => "witness_expired",
                CompletionOnlyReason::SearchInconclusive => "search_inconclusive",
                CompletionOnlyReason::ExistingSloMiss => "existing_slo_miss",
            },
        });
        let result = self
            .install_controller_wave(
                owner::ControllerWork {
                    batch,
                    proof: selected.proof,
                    expected: selected.expected,
                    timing: owner::ControllerTimingCommitment::CompleteRequests,
                },
                receipt,
            )
            .map(SloIterationPlan::Selected);
        retry.armed = result.is_err();
        result
    }

    fn completion_frontiers_match(&self, proof: &ControllerSafetyProof) -> bool {
        let Some(sequences) = self.sequences.try_read() else {
            return false;
        };
        proof.fences.iter().all(|fence| {
            proof.budget.poll()
                && sequences
                    .get(&fence.key.request_id)
                    .is_some_and(|sequence| {
                        fence.matches_sequence(sequence)
                            && sequence.credited_output.as_ref().is_some_and(|output| {
                                output.grant.is_none()
                                    && output.port.planning_snapshot() == fence.output
                            })
                    })
        })
    }

    fn capture_completion_selection(
        &self,
        hint: &ferrum_interfaces::BatchHint,
        budget: Arc<ControllerBudget>,
        reason: CompletionOnlyReason,
        exact: Option<&[super::super::calibration::CalibrationWork]>,
        maximum_requests: NonZeroUsize,
    ) -> ControllerResult<CompletionSelection> {
        let obligations = self.scheduler.active_count() + self.scheduler.waiting_count();
        let unavailable = |reason| Unavailable {
            reason,
            obligations,
            retry: None,
        };
        if self.spec_config.is_some()
            || self.model_executor.execution_resource_authority()
                != ExecutionResourceAuthority::PlanRuntime
        {
            return Err(unavailable("unsupported_execution_authority"));
        }
        if !budget.poll() {
            return Err(unavailable("compute_budget_exhausted")
                .retry(retry::ControllerRetryReason::ComputeBudget));
        }
        let mut availability = self
            .dynamic_admission_availability
            .try_lock()
            .ok_or_else(|| {
                unavailable("capacity_snapshot_busy")
                    .retry(retry::ControllerRetryReason::SnapshotBusy)
            })?;
        let epochs = self
            .model_executor
            .write_execution_capacity_snapshot(&mut availability)
            .map_err(|_| unavailable("capacity_snapshot_unavailable"))?
            .ok_or_else(|| unavailable("capacity_epochs_unavailable"))?;
        let queue = self
            .scheduler
            .planning_state(
                maximum_requests,
                AdmissionWakeSnapshot::new(
                    AdmissionWakeEpochs::new(
                        epochs.coordinator_id,
                        epochs.release_epoch,
                        epochs.capacity_epoch,
                        0,
                    ),
                    &availability,
                ),
            )
            .map_err(|reason| {
                let error = unavailable("scheduler_snapshot_unavailable");
                if reason == PlanningStateUnavailable::Busy {
                    error.retry(retry::ControllerRetryReason::SnapshotBusy)
                } else {
                    error
                }
            })?;
        drop(availability);
        let sequences = self.sequences.try_read().ok_or_else(|| {
            unavailable("sequence_snapshot_busy").retry(retry::ControllerRetryReason::SnapshotBusy)
        })?;
        if sequences.len() != queue.requests().len()
            || queue
                .requests()
                .iter()
                .any(|row| !sequences.contains_key(&row.key.request_id))
        {
            return Err(unavailable("obligation_owner_mismatch"));
        }
        let mut live = Vec::with_capacity(queue.requests().len());
        for row in queue.requests() {
            if !budget.poll() {
                return Err(unavailable("compute_budget_exhausted")
                    .retry(retry::ControllerRetryReason::ComputeBudget));
            }
            let frontier = sequences[&row.key.request_id]
                .cost_frontier
                .ok_or_else(|| unavailable("engine_frontier_unknown"))?;
            live.push((row.key.request_id.clone(), frontier.owner_incarnation.get()));
        }
        let recovery_scope = self
            .slo_controller
            .try_lock()
            .ok_or_else(|| {
                unavailable("controller_busy").retry(retry::ControllerRetryReason::SnapshotBusy)
            })?
            .last_recovery_scope
            .clone();
        let recovery_peers =
            Self::recovery_peers_locked(&queue, &sequences, &budget, recovery_scope.as_deref())?;
        // Do not let a fair-fallback rotation bypass a due executable owner.
        // Exact calibration is a separate diagnostic and never changes its cohort.
        let required = if exact.is_none() {
            recovery::required_peer(&recovery_peers)
        } else {
            None
        };
        let order = if let Some(required) = required {
            VecDeque::from([(required.id.clone(), required.incarnation)])
        } else if let Some(exact) = exact {
            let mut seen = std::collections::HashSet::new();
            let mut order = VecDeque::with_capacity(exact.len());
            for row in exact {
                if !budget.poll() {
                    return Err(unavailable("compute_budget_exhausted")
                        .retry(retry::ControllerRetryReason::ComputeBudget));
                }
                let observed = &row.frontier;
                let sequence = sequences
                    .get(&observed.request_id)
                    .ok_or_else(|| unavailable("calibration_owner_changed"))?;
                let current = sequence
                    .cost_frontier
                    .ok_or_else(|| unavailable("engine_frontier_unknown"))?;
                if !seen.insert(&observed.request_id)
                    || current.owner_incarnation != observed.owner
                    || current.work_generation != observed.generation
                    || sequence.generated_tokens.len() != observed.generated
                    || (!sequence.prefill_complete).then(|| {
                        (
                            sequence.prefill_tokens_processed,
                            sequence.prefill_context_len(),
                        )
                    }) != observed.prefill
                    || sequence
                        .model_kv
                        .as_ref()
                        .map_or(0, |kv| kv.handle().num_tokens())
                        != observed.kv_tokens
                {
                    return Err(unavailable("calibration_frontier_changed"));
                }
                order.push_back((observed.request_id.clone(), observed.owner.get()));
            }
            order
        } else {
            let mut state = self.slo_controller.try_lock().ok_or_else(|| {
                unavailable("controller_busy").retry(retry::ControllerRetryReason::SnapshotBusy)
            })?;
            let live_set: std::collections::HashSet<_> = live.iter().cloned().collect();
            state.completion_order.retain(|key| live_set.contains(key));
            let mut included: std::collections::HashSet<_> =
                state.completion_order.iter().cloned().collect();
            for key in live {
                if included.insert(key.clone()) {
                    state.completion_order.push_back(key);
                }
            }
            state.completion_order.clone()
        };
        let policy_envelope = if exact.is_none() {
            Some(
                self.completion_work_envelope(hint, &queue, &sequences, &budget)
                    .ok_or_else(|| unavailable("work_policy_context_unavailable"))?,
            )
        } else {
            None
        };
        let mut policy_usage =
            ferrum_scheduler::implementations::continuous::work_policy::WaveWorkUsage::default();
        let caps = self.model_executor.capabilities();
        let allow_mixed = policy_envelope.map_or(caps.supports_dynamic_batching, |envelope| {
            envelope.allow_mixed
        });
        let limit = hint.max_batch_size.min(caps.max_batch_size).min(256);
        let mut remaining = hint.max_tokens;
        let mut fences = Vec::new();
        let mut selected = Vec::new();
        let mut expected_rows = Vec::new();
        let mut decode_count = 0;
        let mut prefill_count = 0;
        let by_id: std::collections::HashMap<_, _> = queue
            .requests()
            .iter()
            .map(|row| (&row.key.request_id, row))
            .collect();
        for (id, incarnation) in order {
            if !budget.poll() {
                return Err(unavailable("compute_budget_exhausted")
                    .retry(retry::ControllerRetryReason::ComputeBudget));
            }
            if selected.len() >= limit || remaining == 0 {
                break;
            }
            let row = by_id[&id];
            if !row.readiness.ready() {
                continue;
            }
            let sequence = &sequences[&id];
            let frontier = sequence.cost_frontier.unwrap();
            let decode = row.queue == PlanningQueueKind::Decode;
            if (!decode && row.queue != PlanningQueueKind::Prefill)
                || decode != sequence.prefill_complete
                || row.committed_output_tokens != sequence.generated_tokens.len()
                || (!decode && row.prefill_offset != sequence.prefill_tokens_processed)
            {
                return Err(unavailable("frontier_mismatch"));
            }
            if sequence.generated_tokens.len() >= sequence.sampling_params.max_tokens {
                continue;
            }
            let Some(output) = sequence.credited_output.as_ref() else {
                continue;
            };
            let view = output.port.planning_snapshot();
            if output.failure.is_some()
                || output.port.consumer_closed()
                || output.grant.is_some()
                || matches!(view.readiness, OutputPlanningCreditView::OutputBlocked(_))
                || view.future_capacity.is_none()
            {
                continue;
            }
            let context = sequence
                .model_kv
                .as_ref()
                .map_or(0, |kv| kv.handle().num_tokens());
            let total = sequence.prefill_context_len();
            let (action, input, work, tokens) = if decode {
                if decode_count >= queue.decode_wave_limit() || (prefill_count > 0 && !allow_mixed)
                {
                    continue;
                }
                let cache_id = sequence
                    .model_cache_id()
                    .ok_or_else(|| unavailable("cache_identity_missing"))?;
                let kv_tokens =
                    u32::try_from(context).map_err(|_| unavailable("context_overflow"))?;
                if context >= caps.max_sequence_length {
                    continue;
                }
                decode_count += 1;
                (
                    PlanningWorkAction::Decode,
                    ExpectedWaveInput::Decode {
                        cache_id: cache_id.to_owned(),
                    },
                    ActualRowWork::Decode { kv_tokens },
                    1,
                )
            } else {
                if prefill_count >= queue.prefill_wave_limit() || (decode_count > 0 && !allow_mixed)
                {
                    continue;
                }
                if row.prefill_context_tokens != Some(total) {
                    return Err(unavailable("prefill_boundary_mismatch"));
                }
                let count_limit = total
                    .saturating_sub(row.prefill_offset)
                    .min(remaining)
                    .min(row.prefill_chunk_ceiling.unwrap_or(usize::MAX))
                    .min(
                        self.config
                            .scheduler
                            .prefill_step_chunk
                            .unwrap_or(usize::MAX),
                    );
                let count_limit = if let Some(envelope) = policy_envelope {
                    let limit = count_limit.min(
                        usize::try_from(envelope.prefill_tokens_available(policy_usage))
                            .unwrap_or(usize::MAX),
                    );
                    // A policy cap cannot invent an unsupported partial length.
                    if limit < total.saturating_sub(row.prefill_offset) {
                        self.model_executor
                            .guarded_prefill_granularity()
                            .map_or(limit, |unit| limit - limit % unit.get())
                    } else {
                        limit
                    }
                } else {
                    count_limit
                };
                let count = if let Some(exact) = exact {
                    match exact
                        .iter()
                        .find(|candidate| candidate.frontier.request_id == id)
                        .map(|candidate| candidate.work)
                    {
                        Some(ActualRowWork::Prefill { count, .. })
                            if count as usize <= count_limit =>
                        {
                            count as usize
                        }
                        _ => return Err(unavailable("calibration_exact_span_unavailable")),
                    }
                } else if required.is_some() {
                    // A due owner needs physical progress, not the largest
                    // available chunk. Unknown executor granularity retains
                    // the ordinary legal bound instead of guessing one token.
                    self.model_executor
                        .guarded_prefill_granularity()
                        .filter(|quantum| quantum.get() <= count_limit)
                        .map_or(count_limit, NonZeroUsize::get)
                } else {
                    count_limit
                };
                let Some(count) = NonZeroUsize::new(count) else {
                    continue;
                };
                let chunk = ferrum_interfaces::model_executor::PrefillChunk::new(
                    row.prefill_offset,
                    count.get(),
                    total,
                )
                .map_err(|_| unavailable("prefill_boundary_mismatch"))?;
                let convert = |v| u32::try_from(v).map_err(|_| unavailable("context_overflow"));
                prefill_count += 1;
                (
                    PlanningWorkAction::Prefill {
                        offset: row.prefill_offset,
                        count,
                    },
                    ExpectedWaveInput::Prefill { chunk },
                    ActualRowWork::Prefill {
                        offset: convert(row.prefill_offset)?,
                        count: convert(count.get())?,
                        total_prompt_tokens: convert(total)?,
                    },
                    count.get(),
                )
            };
            if let Some(envelope) = policy_envelope {
                let prefill = (!decode).then(|| NonZeroU64::new(tokens as u64).unwrap());
                if !envelope.include(&mut policy_usage, prefill) {
                    return Err(unavailable("product_work_policy_exceeded"));
                }
            }
            if exact.is_some_and(|rows| {
                rows.iter()
                    .find(|row| row.frontier.request_id == id)
                    .is_none_or(|row| row.work != work)
            }) {
                return Err(unavailable("calibration_work_changed"));
            }
            let calibration_full_logits = decode
                && exact.is_some_and(|rows| {
                    rows.iter().any(|row| {
                        row.frontier.request_id == id
                            && row.decode_route
                                == super::super::calibration::CalibrationDecodeRoute::FullLogits
                    })
                });
            let policy = if decode && !calibration_full_logits {
                sequence.model_decode_logits_policy()
            } else {
                ferrum_interfaces::model_executor::LogitsReturnPolicy::FullLogits
            };
            selected.push(PlanningWorkSelection {
                key: row.key.clone(),
                action,
            });
            expected_rows.push(ExpectedWorkSelection {
                participant_index: fences.len(),
                request_id: id.clone(),
                owner_incarnation: frontier.owner_incarnation,
                work_generation: frontier.work_generation,
                input,
                work,
                decode_policy: decode.then(|| policy.clone()),
            });
            fences.push(EngineFence {
                key: row.key.clone(),
                incarnation,
                generation: frontier.work_generation.get(),
                generated: sequence.generated_tokens.len(),
                context,
                output: view,
                cache_id: sequence.model_cache_id().map(str::to_owned),
                prefill_complete: decode,
                prefill_tokens_processed: sequence.prefill_tokens_processed,
                prefill_total: total,
                logits_policy: policy,
                future_greedy_policy: None,
                host_features: None,
            });
            remaining -= tokens;
        }
        if expected_rows.is_empty() {
            return Err(unavailable("output_or_resource_blocked"));
        }
        if exact.is_some_and(|rows| rows.len() != expected_rows.len()) {
            return Err(unavailable("calibration_exact_cohort_unavailable"));
        }
        let requests: Vec<_> = fences
            .iter()
            .map(|fence| ExecutorResourcePlanningRequest {
                request_id: &fence.key.request_id,
                cache_id: fence.resource_cache_id(),
            })
            .collect();
        let resources = match self.model_executor.execution_resource_planning_view(
            &requests,
            ResourcePlanningLimits {
                maximum_participants: 256,
                maximum_projected_waves: 1,
                ..Default::default()
            },
            &mut || budget.poll(),
        ) {
            ResourcePlanningAvailability::Known(view) => view,
            ResourcePlanningAvailability::Unknown(reason) => {
                if exact.is_some() {
                    self.slo_controller.lock().last_resource_unavailable = Some(reason);
                }
                let error = unavailable(unknown_label(resources::resource_reason(reason)));
                return Err(match reason {
                    ResourcePlanningUnknown::ReadUnavailable(_)
                    | ResourcePlanningUnknown::BusyOrUnavailable => {
                        error.retry(retry::ControllerRetryReason::SnapshotBusy)
                    }
                    ResourcePlanningUnknown::BudgetExhausted => {
                        error.retry(retry::ControllerRetryReason::ComputeBudget)
                    }
                    _ => error,
                });
            }
        };
        drop(requests);
        drop(sequences);
        // The native wave owns canonical authority order, independent of fair
        // selection order or the prefill/decode input vectors.
        expected_rows
            .sort_by_key(|row| resources.participants()[row.participant_index].authority());
        let kind = match (prefill_count != 0, decode_count != 0) {
            (true, true) => ActualWaveKind::Mixed,
            (true, false) => ActualWaveKind::Prefill,
            (false, true) => ActualWaveKind::Decode,
            _ => unreachable!(),
        };
        let work = ExpectedWaveWork::new(&resources, kind, expected_rows)
            .map_err(|_| unavailable("invalid_exact_work"))?;
        if !budget.poll() {
            return Err(unavailable("compute_budget_exhausted")
                .retry(retry::ControllerRetryReason::ComputeBudget));
        }
        Ok(CompletionSelection {
            proof: ControllerSafetyProof {
                recovery_peers,
                protection: None,
                budget,
                queue,
                fences,
            },
            resources,
            selected,
            expected: ExpectedExecutionWave::complete_requests(work, reason),
        })
    }
}
