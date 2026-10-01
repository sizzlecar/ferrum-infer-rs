//! Untimed physical progress supplies real cold maintenance observations. It
//! never installs a follower hold or invents a cost before qualification.
use super::*;
use ferrum_interfaces::execution_cost::{GuardedNotSubmittedReason, HostSubmissionRejection};
use ferrum_interfaces::model_executor::{
    PlanRuntimePrefixRestoreInput, PlanRuntimePrefixRestoreOutcome,
};
use ferrum_interfaces::vnext::{CheckpointTransferSubmissionGuard, PreparedCheckpointTransfer};

pub(in super::super) struct PrefixSample {
    source: SampleOwner,
    target: SampleOwner,
    source_tokens: Vec<TokenId>,
    target_tokens: Vec<TokenId>,
    boundary: usize,
    expires_at: Instant,
}

pub(in crate::continuous_engine) struct PreparedPrefixSampling {
    sample: Arc<PrefixSample>,
    restore: bool,
    budget: Arc<ControllerBudget>,
}

struct SampleOwner {
    id: RequestId,
    identity: Arc<()>,
    frontier: super::super::super::cost_observation::CostFrontier,
    processed: usize,
    maximum_sequence_tokens: usize,
}

impl SampleOwner {
    fn capture(id: &RequestId, sequence: &SequenceState) -> Option<Self> {
        Some(Self {
            id: id.clone(),
            identity: Arc::clone(&sequence.stream_projection_identity),
            frontier: sequence.cost_frontier?,
            processed: sequence.prefill_tokens_processed,
            maximum_sequence_tokens: sequence.model_maximum_sequence_tokens(),
        })
    }
    fn matches(&self, sequence: &SequenceState) -> bool {
        Arc::ptr_eq(&self.identity, &sequence.stream_projection_identity)
            && sequence.cost_frontier == Some(self.frontier)
            && sequence.prefill_tokens_processed == self.processed
            && !sequence.prefill_complete
            && sequence.generated_tokens.is_empty()
            && sequence.model_maximum_sequence_tokens() == self.maximum_sequence_tokens
    }
}

struct SampleGuard {
    engine: std::sync::Weak<EngineInner>,
    sample: Arc<PrefixSample>,
    restore: bool,
}

impl CheckpointTransferSubmissionGuard for SampleGuard {
    fn check(
        &self,
        actual: &PreparedCheckpointTransfer<'_>,
    ) -> std::result::Result<(), GuardedNotSubmittedReason> {
        let result = (|| {
            if (actual.cost_domain().kind()
                == ferrum_interfaces::vnext::NativeCheckpointTransferKind::Restore)
                != self.restore
            {
                return Err(GuardedNotSubmittedReason::ActualRouteMismatch);
            }
            use HostSubmissionRejection::*;
            let check = || {
                let engine = self.engine.upgrade().ok_or(Cancelled)?;
                if engine.shutdown_started.load(Ordering::Acquire) {
                    return Err(Cancelled);
                }
                if slo_clock_now() >= self.sample.expires_at {
                    return Err(WitnessExpired);
                }
                if !engine.prefix_sampling_cost_missing(actual.cost_domain(), actual.host_work()) {
                    return Err(CostModelChanged);
                }
                let sequences = engine.sequences.try_read().ok_or(Busy)?;
                // New finite promises or cancellation cannot slip between the
                // cold decision and the native backend's actual commit.
                if sequences.values().any(|sequence| {
                    sequence
                        .time_admission
                        .as_ref()
                        .is_some_and(|state| state.has_current_time_witness(slo_clock_now()))
                }) {
                    return Err(CostModelChanged);
                }
                for owner in [&self.sample.source, &self.sample.target] {
                    let sequence = sequences.get(&owner.id).ok_or(FrontierChanged)?;
                    if !owner.matches(sequence) {
                        return Err(FrontierChanged);
                    }
                    let output = sequence.credited_output.as_ref().ok_or(OutputRevoked)?;
                    if output.failure.is_some()
                        || output.port.consumer_closed()
                        || output.grant.is_some()
                    {
                        return Err(OutputRevoked);
                    }
                }
                if slo_clock_now() >= self.sample.expires_at {
                    return Err(WitnessExpired);
                }
                if !engine.prefix_sampling_cost_missing(actual.cost_domain(), actual.host_work()) {
                    return Err(CostModelChanged);
                }
                Ok(())
            };
            check().map_err(GuardedNotSubmittedReason::HostRejected)
        })();
        if let Some(engine) = self.engine.upgrade() {
            if let Some(recorder) = &engine.prefix_resource_recorder {
                use crate::continuous_engine::profile::prefix::{Event, Owner};
                let owner = if self.restore {
                    &self.sample.target
                } else {
                    &self.sample.source
                };
                recorder.record(Event::NativeGuard {
                    owner: Owner::from_engine(
                        &owner.id,
                        owner.frontier.owner_incarnation.get(),
                        owner.frontier.work_generation.get(),
                    ),
                    identity: actual.identity().into(),
                    source_capture: actual.source_capture_identity().map(Into::into),
                    // This is unprotected sampling, not a predictive witness.
                    inference_epoch: None,
                    maintenance_epoch: None,
                    rejection: result.as_ref().err().copied(),
                });
            }
        }
        result
    }
}

impl EngineInner {
    pub(in super::super) fn prefix_inference_cost_absent(&self) -> bool {
        matches!(
            self.cost_runtime
                .as_ref()
                .and_then(|runtime| runtime.try_snapshot()),
            Some(None)
        )
    }

    fn prefix_sampling_cost_missing(
        &self,
        domain: &ferrum_interfaces::vnext::NativeCheckpointTransferCostDomain,
        host_work: Option<&ferrum_interfaces::vnext::NativeCheckpointTransferHostWork>,
    ) -> bool {
        let Some(runtime) = &self.cost_runtime else {
            return false;
        };
        match runtime.try_snapshot() {
            Some(None) => true,
            None => false,
            Some(Some(_)) => {
                let model = match runtime.try_prefix_cost_snapshot() {
                    Some(Some(model)) => model,
                    Some(None) => return true,
                    None => return false,
                };
                let Some(now) = runtime.clock.now_ns() else {
                    return false;
                };
                let Ok(shape) =
                    super::super::super::cost_observation::prefix_cost_shape(domain, host_work)
                else {
                    return false;
                };
                model.cost_missing(&shape, now)
            }
        }
    }

    pub(in super::super) fn prepare_prefix_sampling(
        &self,
        budget: &Arc<ControllerBudget>,
        captured: Option<&ControllerSnapshot>,
    ) -> Option<PreparedPrefixSampling> {
        if !budget.poll()
            || self.config.scheduler.slo.mode != ferrum_types::SloMode::Enforce
            || !self.completion_allowed()
            || self.requires_strict_acceptance()
            || self.config.scheduler.slo.admission.time_policy
                != ferrum_types::SloTimeAdmissionPolicy::CompleteRequests
            || !self.model_executor.supports_guarded_prefix_maintenance()
            || self
                .cost_runtime
                .as_ref()
                .and_then(|runtime| runtime.prefix_cost_sink())
                .is_none()
            || self.slo_controller.lock().prefix.is_some()
        {
            return None;
        }
        let max_wait = self.config.scheduler.prefix_rendezvous_max_wait_ms?;
        let now = slo_clock_now();
        let sequences = self.sequences.try_read()?;
        for sequence in sequences.values() {
            if !budget.poll()
                || sequence
                    .time_admission
                    .as_ref()
                    .is_some_and(|state| state.has_current_time_witness(now))
            {
                return None;
            }
        }
        let pending = self.slo_controller.lock().prefix_sample.take();
        if let Some(sample) = pending {
            if now < sample.expires_at
                && [&sample.source, &sample.target].iter().all(|owner| {
                    sequences
                        .get(&owner.id)
                        .is_some_and(|sequence| owner.matches(sequence))
                })
            {
                return Some(PreparedPrefixSampling {
                    sample,
                    restore: true,
                    budget: Arc::clone(budget),
                });
            }
        }
        let mut controller = self.slo_controller.lock();
        // Keep only one last sampled boundary per still-live source owner.
        controller.prefix_samples.retain(|(id, incarnation, _, _)| {
            sequences
                .get(id)
                .and_then(|s| s.cost_frontier)
                .is_some_and(|f| f.owner_incarnation.get() == *incarnation)
        });
        let candidates = self
            .scheduler
            .try_prefix_rendezvous_candidates(sequences.len(), &mut || budget.poll())?;
        for (source_id, source) in sequences.iter() {
            if !budget.poll() {
                return None;
            }
            let boundary = source.prefill_tokens_processed;
            if source.prefill_complete
                || boundary == 0
                || boundary >= source.prefill_context_len()
                || !source.generated_tokens.is_empty()
                || source.preemption_count != 0
            {
                continue;
            }
            let Some(source_candidate) = candidates.iter().find(|candidate| {
                !candidate.waiting
                    && candidate.key.request_id() == source_id
                    && candidate.processed_tokens == boundary
            }) else {
                continue;
            };
            let Some(frontier) = source.cost_frontier else {
                continue;
            };
            if controller
                .prefix_samples
                .iter()
                .any(|(id, incarnation, generation, last)| {
                    id == source_id
                        && *incarnation == frontier.owner_incarnation.get()
                        && *generation == frontier.work_generation.get()
                        && *last == boundary
                })
            {
                continue;
            }
            let source_tokens = source.prefill_context_tokens();
            for (target_id, target) in sequences.iter() {
                if !budget.poll() {
                    return None;
                }
                if source_id == target_id
                    || target.prefill_complete
                    || target.prefill_tokens_processed != 0
                    || !target.generated_tokens.is_empty()
                    || target.preemption_count != 0
                {
                    continue;
                }
                if !candidates.iter().any(|candidate| {
                    !candidate.waiting
                        && candidate.key.request_id() == target_id
                        && candidate.processed_tokens == 0
                        && candidate.priority <= source_candidate.priority
                }) {
                    continue;
                }
                let target_tokens = target.prefill_context_tokens();
                if boundary >= target_tokens.len()
                    || source_tokens[..boundary] != target_tokens[..boundary]
                {
                    continue;
                }
                // Ask the installed checkpoint contract for the same exact
                // already-completed boundary; native authority checks it again.
                let plan =
                    self.model_executor
                        .plan_prefix_capture_boundary(PrefixCaptureBoundary {
                            processed_tokens: 0,
                            source_prompt_tokens: source_tokens.len(),
                            common_prefix_tokens: boundary,
                            follower_prompt_tokens: &[target_tokens.len()],
                        });
                if plan.is_none_or(|plan| plan.boundary != boundary) {
                    continue;
                }
                if !self.prefix_cold_capture_cost_missing(captured, source_id, frontier, boundary) {
                    continue;
                }
                let source = SampleOwner::capture(source_id, source)?;
                let target = SampleOwner::capture(target_id, target)?;
                let expires_at = now.checked_add(Duration::from_millis(max_wait.get()))?;
                controller
                    .prefix_samples
                    .retain(|(id, _, _, _)| id != source_id);
                controller.prefix_samples.push((
                    source_id.clone(),
                    frontier.owner_incarnation.get(),
                    frontier.work_generation.get(),
                    boundary,
                ));
                return Some(PreparedPrefixSampling {
                    sample: Arc::new(PrefixSample {
                        source,
                        target,
                        source_tokens,
                        target_tokens,
                        boundary,
                        expires_at,
                    }),
                    restore: false,
                    budget: Arc::clone(budget),
                });
            }
        }
        None
    }

    fn prefix_cold_capture_cost_missing(
        &self,
        captured: Option<&ControllerSnapshot>,
        source: &RequestId,
        frontier: super::super::super::cost_observation::CostFrontier,
        boundary: usize,
    ) -> bool {
        let Some(runtime) = &self.cost_runtime else {
            return false;
        };
        match runtime.try_snapshot() {
            Some(None) => return true,
            None => return false,
            Some(Some(_)) => {}
        }
        let model = match runtime.try_prefix_cost_snapshot() {
            Some(None) => return true,
            Some(Some(model)) => model,
            None => return false,
        };
        let Some(captured) = captured else {
            return false;
        };
        if !captured.poll_planning() || !self.controller_frontiers_match(captured) {
            return false;
        }
        let Some(source) = captured.snapshot.requests.iter().position(|row| {
            &row.key.request_id == source && row.key.incarnation == frontier.owner_incarnation.get()
        }) else {
            return false;
        };
        let Some(proof) = captured.resources.participants()[source].completed_checkpoint_boundary()
        else {
            return false;
        };
        if proof.completed_tokens() != boundary as u64 {
            return false;
        }
        let mut poll = || captured.poll_planning();
        let result = self.model_executor.project_execution_checkpoint(
            &captured.route,
            &captured.route.initial_state(),
            ferrum_interfaces::vnext::FutureCheckpointCostQuery::Capture {
                source,
                span_start: proof.span_start(),
                boundary: proof.completed_tokens(),
                prompt_tokens: proof.prompt_tokens(),
            },
            &mut captured.budget.observed_resource_budget(&mut poll),
        );
        let ferrum_interfaces::vnext::ExecutionCostRouteAvailability::Known(projected) = result
        else {
            return false;
        };
        let Ok(shape) = super::super::super::cost_observation::prefix_cost_shape(
            &projected.cost_domain,
            Some(&projected.host_work),
        ) else {
            return false;
        };
        let Some(now) = runtime.clock.now_ns() else {
            return false;
        };
        model.cost_missing(&shape, now) && captured.poll_planning()
    }

    pub(in crate::continuous_engine::inner) async fn execute_prefix_sampling(
        self: &Arc<Self>,
        prepared: PreparedPrefixSampling,
    ) -> Result<EngineIterationOutcome> {
        let PreparedPrefixSampling {
            sample,
            restore,
            budget,
        } = prepared;
        let error_request_id = if restore {
            sample.target.id.clone()
        } else {
            sample.source.id.clone()
        };
        let guard = Arc::new(SampleGuard {
            engine: Arc::downgrade(self),
            sample: Arc::clone(&sample),
            restore,
        });
        let result = if !restore {
            self.model_executor
                .try_capture_plan_runtime_prefix_guarded(
                    PrefixCaptureRequest {
                        source_request_id: &sample.source.id,
                        source_tokens: &sample.source_tokens,
                        maximum_sequence_tokens: sample.source.maximum_sequence_tokens,
                        boundary: sample.boundary,
                        expires_at: sample.expires_at,
                    },
                    guard,
                )
                .await
                .map(|captured| {
                    if captured {
                        // The model index now owns a real published checkpoint.
                        // A next-turn restore still performs its normal selection,
                        // physical admission and native target validation.
                        self.slo_controller.lock().prefix_sample = Some(Arc::clone(&sample));
                    }
                })
        } else {
            let request_id = sample.target.id.clone();
            if let Some(prepared) =
                self.scheduler
                    .prepare_prefix_restore(&request_id, 0, sample.target_tokens.len())?
            {
                match self
                    .model_executor
                    .try_restore_plan_runtime_prefix_guarded(
                        PlanRuntimePrefixRestoreInput {
                            request_id: &request_id,
                            input_tokens: &sample.target_tokens,
                            maximum_sequence_tokens: sample.target.maximum_sequence_tokens,
                            checkpoint: None,
                            retry: None,
                        },
                        guard,
                    )
                    .await
                {
                    Ok(PlanRuntimePrefixRestoreOutcome::Restored(output)) => self
                        .commit_prefix_restore_output(
                            &request_id,
                            prepared,
                            &sample.target_tokens,
                            output,
                        ),
                    Ok(
                        PlanRuntimePrefixRestoreOutcome::Unavailable
                        | PlanRuntimePrefixRestoreOutcome::Deferred(_),
                    ) => Ok(()),
                    Err(error) => Err(error),
                }
            } else {
                Ok(())
            }
        };
        self.finish_controller_audit(&budget, "prefix_sampling");
        if let Err(error) = result {
            // Native/model errors may have cancelled the exact target. Never
            // reuse that owner after a failed publication acknowledgement.
            self.complete_request_with_error(&error_request_id, error)
                .await?;
        }
        Ok(EngineIterationOutcome::Progressed)
    }
}
