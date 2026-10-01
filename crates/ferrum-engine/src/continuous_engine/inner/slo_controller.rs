//! One bounded, read-only planning transaction followed by at most one real
//! wave. A witness is finite evidence; scheduler/resource/output owners still
//! supply all execution permissions. Observe never publishes a selection.
use super::*;
use ferrum_interfaces::model_executor::ExecutorResourcePlanningRequest;
use ferrum_interfaces::vnext::{
    ResourcePlanningAvailability, ResourcePlanningLimits, ResourcePlanningUnknown,
    ResourcePlanningView,
};
use ferrum_scheduler::implementations::continuous::{planning_state::*, slo_planner::*};
use std::num::{NonZeroU32, NonZeroU64, NonZeroUsize};

pub(in crate::continuous_engine) mod admission;
use admission::SequenceTimeAdmission;
mod budget;
use budget::{ControllerAudit, ControllerBudget, ControllerOptionalPhase, ControllerStage};
mod commit;
pub(super) use commit::ControllerCommitFence;
pub(super) mod calibration;
mod completion;
mod dispatch;
mod maintenance;
mod owner;
mod prefix;
mod recovery;
mod resources;
mod retry;
pub(super) mod sampling;
pub(super) mod shape;
mod snapshot;
mod submit;
#[cfg(test)]
pub(in crate::continuous_engine) mod tests;
mod work_envelope;

pub(in crate::continuous_engine) enum SloIterationPlan {
    Legacy,
    Idle,
    Selected(submit::PreparedControllerWave),
    PrefixMaintenance(prefix::PreparedPrefixMaintenance),
    PrefixSampling(prefix::PreparedPrefixSampling),
    Progressed,
}

/// Only one iteration publishes a wave at a time. A typed no-submission
/// receipt survives scheduler lock contention and is retried before planning.
#[derive(Default)]
pub(in crate::continuous_engine) struct SloControllerState {
    pending_release: Option<PlanningPublicationReceipt>,
    pending_execution: Option<Arc<owner::ControllerFlight>>,
    pending_maintenance: Option<maintenance::ControllerMaintenance>,
    /// A maintenance retry can need one empty fairness turn when no peer can
    /// publish. It is not a timer or permission to reopen any physical claim.
    pending_maintenance_fairness: bool,
    observations: u64,
    last_observation: Option<ControllerObservation>,
    /// Exact resource failure for the current manual selection attempt. The
    /// calibration entry clears this with last_observation before capture.
    last_resource_unavailable: Option<ResourcePlanningUnknown>,
    last_audit: Option<ControllerAudit>,
    timing: Option<ferrum_types::ControllerTimingMetrics>,
    /// Bounded numeric recovery evidence; retains no request/resource owner.
    last_recovery_scope: Option<Arc<PlanningObligationSet>>,
    /// The next completion-only attempt rotates across the complete queue.
    /// Failed/blocked work remains an obligation and cannot monopolize peers.
    completion_order: VecDeque<(RequestId, u64)>,
    completion_next: Option<ferrum_interfaces::execution_cost::CompletionOnlyReason>,
    retry: Option<retry::ControllerRetryWake>,
    time_activation: admission::TimeActivationState,
    prefix: Option<prefix::SloPrefixCohort>,
    ready_prefix: Option<prefix::ReadyPrefixCohort>,
    // One numeric attempt marker per live owner; no cache or owner lease.
    prefix_cache_captures: Vec<(RequestId, u64, u32)>,
    prefix_cache_preparing: Option<prefix::producer::CacheCaptureContinuation>,
    prefix_samples: Vec<(RequestId, u64, u64, usize)>,
    prefix_sample: Option<Arc<prefix::PrefixSample>>,
}

#[derive(Debug, Clone, Copy)]
struct ControllerObservation {
    obligations: usize,
    disposition: &'static str,
    reason: &'static str,
}

impl SloControllerState {
    pub(in crate::continuous_engine::inner) fn has_pending_calibration_work(&self) -> bool {
        self.pending_release.is_some()
            || self.pending_execution.is_some()
            || self.pending_maintenance.is_some()
    }
}

#[derive(Debug, Clone, Copy)]
struct Unavailable {
    reason: &'static str,
    obligations: usize,
    retry: Option<retry::ControllerRetryReason>,
}
impl Unavailable {
    fn retry(mut self, reason: retry::ControllerRetryReason) -> Self {
        self.retry = Some(reason);
        self
    }
}
type ControllerResult<T> = std::result::Result<T, Unavailable>;

struct EngineFence {
    key: PlanningRequestKey,
    incarnation: u64,
    generation: u64,
    generated: usize,
    context: usize,
    output: crate::continuous_engine::output_flow_runtime::OutputPlanningSnapshot,
    /// Actual host cache identity, including a retained partial-prefill cache.
    /// This is not permission to query that Ready prefill slot as a Decode.
    cache_id: Option<String>,
    prefill_complete: bool,
    prefill_tokens_processed: usize,
    prefill_total: usize,
    logits_policy: ferrum_interfaces::model_executor::LogitsReturnPolicy,
    /// Captured immutable normal mask for a future clean UTF-8 branch. Present
    /// only for the installed empirical plain-text capability; not a prediction
    /// of token content or permission to replace the first wave's exact policy.
    future_greedy_policy: Option<ferrum_interfaces::model_executor::LogitsReturnPolicy>,
    /// Actual unique-history count and installed penalty; no future IDs.
    future_repetition: Option<(u64, f32, u64)>,
    host_features: Option<ferrum_interfaces::execution_cost::HostCostFeaturesV1>,
}

impl EngineFence {
    fn resource_cache_id(&self) -> Option<&str> {
        self.prefill_complete
            .then_some(self.cache_id.as_deref())
            .flatten()
    }

    fn matches_sequence(&self, sequence: &SequenceState) -> bool {
        sequence.cost_frontier.is_some_and(|frontier| {
            frontier.owner_incarnation.get() == self.incarnation
                && frontier.work_generation.get() == self.generation
        }) && sequence.generated_tokens.len() == self.generated
            && sequence.prefill_complete == self.prefill_complete
            && sequence.prefill_tokens_processed == self.prefill_tokens_processed
            && sequence.prefill_context_len() == self.prefill_total
            && sequence.model_cache_id() == self.cache_id.as_deref()
            && sequence
                .model_kv
                .as_ref()
                .map_or(0, |kv| kv.handle().num_tokens())
                == self.context
    }
}

struct ControllerSnapshot {
    prefix_maintenance: Option<Arc<super::cost_observation::PrefixCostSnapshot>>,
    prefix_checkpoint: Option<Arc<dyn ferrum_interfaces::model_executor::PrefixCaptureLease>>,
    budget: Arc<ControllerBudget>,
    optional_phase: Option<ControllerOptionalPhase>,
    queue: PlanningQueueSnapshot,
    snapshot: SchedulerSnapshot,
    origin: PlanningTimeOrigin,
    anchor: PlanningCostClockAnchor,
    resources: ResourcePlanningView,
    route: ferrum_interfaces::vnext::ExecutionCostRouteView,
    fences: Vec<EngineFence>,
    /// Other strict waiters remain in the full queue seal and owner fence,
    /// but carry no accepted time obligation in this candidate's forecast.
    waiting_fences: Vec<EngineFence>,
    before_acceptance_candidate: Option<RequestId>,
    model: Arc<dyn PlanningCostModel + Send + Sync>,
    protection: Arc<PlanningObligationSet>,
    recovery_peers: Vec<recovery::RecoveryPeer>,
}

impl ControllerSnapshot {
    fn poll_planning(&self) -> bool {
        self.optional_phase
            .as_ref()
            .map_or_else(|| self.budget.poll(), ControllerOptionalPhase::poll)
    }

    fn planning_window(&self) -> std::result::Result<PlanningPhaseBudget, PlanningTimeError> {
        self.optional_phase.as_ref().map_or_else(
            || self.budget.planning_window(&self.origin).map(Into::into),
            |phase| phase.planning_window(&self.origin),
        )
    }
}

struct ControllerSafetyProof {
    prefix_maintenance: Option<Arc<super::cost_observation::PrefixCostSnapshot>>,
    _prefix_checkpoint: Option<Arc<dyn ferrum_interfaces::model_executor::PrefixCaptureLease>>,
    recovery_peers: Vec<recovery::RecoveryPeer>,
    protection: Option<Arc<PlanningObligationSet>>,
    budget: Arc<ControllerBudget>,
    queue: PlanningQueueSnapshot,
    fences: Vec<EngineFence>,
    waiting_fences: Vec<EngineFence>,
}

impl ControllerSnapshot {
    fn into_safety(self) -> ControllerSafetyProof {
        ControllerSafetyProof {
            prefix_maintenance: self.prefix_maintenance,
            _prefix_checkpoint: self.prefix_checkpoint,
            protection: self.protection.needs_recovery().then_some(self.protection),
            recovery_peers: self.recovery_peers,
            budget: self.budget,
            queue: self.queue,
            fences: self.fences,
            waiting_fences: self.waiting_fences,
        }
    }
}

impl EngineInner {
    /// Invoked by the actual iteration before any old next_batch/cohort work.
    /// No mutex guard or read view is retained over device execution.
    pub(super) fn prepare_slo_controller(
        &self,
        hint: &ferrum_interfaces::BatchHint,
    ) -> Result<SloIterationPlan> {
        if self.config.scheduler.slo.execution_policy().controller
            == ferrum_types::SloControllerPolicy::Legacy
        {
            return Ok(SloIterationPlan::Legacy);
        }
        // Publication rollback and durable task completion must still notify
        // their waiters. Those notifications cannot bypass retry backoff.
        if self.controller_retry_pending() {
            return Ok(SloIterationPlan::Idle);
        }
        let budget = ControllerBudget::new(
            slo_clock_now(),
            self.config.scheduler.slo.planner.planning_budget(),
        )?;
        if let Some(writer) = &self.required_query_observation {
            let _ = budget.observation.set(writer.begin());
        }
        let diagnostic_span = tracing::trace_span!(target: "ferrum::slo_transaction",
            "slo_transaction", transaction=budget.observation.get().map(|o| o.id()));
        let _diagnostic = diagnostic_span.enter();
        budget.diagnostic_checkpoint("preparation_begin");
        let result = self.prepare_slo_controller_budgeted(hint, &budget);
        self.finish_slo_controller_preparation(&budget, result)
    }

    /// Close the one planning transaction after every preparation outcome,
    /// including publication failures that already returned Idle.
    fn finish_slo_controller_preparation(
        &self,
        budget: &Arc<ControllerBudget>,
        result: Result<SloIterationPlan>,
    ) -> Result<SloIterationPlan> {
        let within_budget = budget.finish_planning();
        budget.diagnostic_checkpoint("preparation_end");
        if !within_budget {
            self.arm_controller_retry(retry::ControllerRetryReason::ComputeBudget);
            if result.is_ok() {
                // Publication can already have returned Idle after consuming
                // the shared budget. Give completion its next fresh turn in
                // that case too, rather than paying for another full search.
                self.request_completion_turn(
                    ferrum_interfaces::execution_cost::CompletionOnlyReason::SearchInconclusive,
                );
            }
        }
        if let Ok(SloIterationPlan::Selected(prepared)) = result {
            if !within_budget {
                // The durable owner performs the normal exact withdrawal. A
                // late publication cannot reach dispatch after budget expiry.
                drop(prepared);
                return Ok(SloIterationPlan::Idle);
            }
            return Ok(SloIterationPlan::Selected(prepared));
        }
        if matches!(
            &result,
            Ok(SloIterationPlan::PrefixMaintenance(_) | SloIterationPlan::PrefixSampling(_))
        ) {
            if within_budget {
                return result;
            }
            drop(result);
            self.finish_controller_audit(budget, "prefix_preparation_expired");
            return Ok(SloIterationPlan::Idle);
        }
        self.finish_controller_audit(
            budget,
            match &result {
                Err(_) => "failed",
                Ok(SloIterationPlan::Legacy) => "observed",
                _ => "idle",
            },
        );
        result
    }

    fn prepare_slo_controller_budgeted(
        &self,
        hint: &ferrum_interfaces::BatchHint,
        budget: &Arc<ControllerBudget>,
    ) -> Result<SloIterationPlan> {
        let mode = self.config.scheduler.slo.mode;
        let released = {
            let _stage = budget.stage(ControllerStage::Release);
            self.retry_slo_publication_release()?
        };
        let within_budget = budget.poll();
        if !within_budget || !released {
            if !released {
                self.arm_controller_retry(retry::ControllerRetryReason::SnapshotBusy);
            }
            budget.record_unavailable(if within_budget {
                "publication_release_busy"
            } else {
                "compute_budget_exhausted"
            });
            self.record_controller(ControllerObservation {
                obligations: self.scheduler.active_count() + self.scheduler.waiting_count(),
                disposition: "unknown",
                reason: if within_budget {
                    "publication_release_busy"
                } else {
                    "compute_budget_exhausted"
                },
            });
            return Ok(if mode == ferrum_types::SloMode::Enforce {
                if !within_budget {
                    self.request_completion_turn(
                        ferrum_interfaces::execution_cost::CompletionOnlyReason::SearchInconclusive,
                    );
                }
                SloIterationPlan::Idle
            } else {
                SloIterationPlan::Legacy
            });
        }
        if self.config.scheduler.slo.execution_policy().controller
            == ferrum_types::SloControllerPolicy::DeadlineOnly
        {
            return self.prepare_completion_controller(
                hint,
                budget,
                ferrum_interfaces::execution_cost::CompletionOnlyReason::DeadlinePolicy,
            );
        }
        // Lock contention is not missing cost. A published inference model
        // always reaches cost-aware planning before new cold work is selected.
        if self.prefix_inference_cost_absent() {
            if let Some(sampling) = self.prepare_prefix_sampling(budget, None) {
                return Ok(SloIterationPlan::PrefixSampling(sampling));
            }
        }
        let completion = self.slo_controller.lock().completion_next;
        if let Some(reason) = completion.filter(|_| {
            let state = self.slo_controller.lock();
            self.completion_allowed()
                && state.prefix.is_none()
                && state.ready_prefix.is_none()
                && state.prefix_sample.is_none()
                && !self.config.runtime.prefix_state_cache_enabled
        }) {
            return self.prepare_completion_controller(hint, budget, reason);
        }
        // CompleteRequests permits safe physical progress without a time
        // witness. Prepare that evidence before optional cost work can spend
        // the transaction. A draft holds no publication or output permission.
        // If it is unavailable, retain the ordinary complete-obligation search.
        budget.diagnostic_checkpoint("completion_draft_begin");
        let mut draft = self
            .completion_allowed()
            .then(|| self.capture_completion_draft(hint, budget).ok())
            .flatten();
        budget.diagnostic_checkpoint(if draft.is_some() {
            "completion_draft_ready"
        } else {
            "completion_draft_unavailable"
        });
        let optional_phase = draft.as_ref().map(|draft| {
            draft.optional_phase.clone().unwrap_or_else(|| {
                budget.completion_optional_phase(
                    draft.preparation_wall,
                    self.config
                        .scheduler
                        .slo
                        .planner
                        .publication_reserve_percent,
                )
            })
        });
        if optional_phase.as_ref().is_some_and(|phase| !phase.poll()) {
            budget.record_search(&PlanningDecision::Unknown {
                reason: PlanningUnknownReason::ComputeBudgetExhausted,
                search: Default::default(),
            });
            if self.drop_slo_prefix_trajectory() {
                return Ok(SloIterationPlan::Progressed);
            }
            return self.prepare_completion_with_draft(
                hint,
                budget,
                draft,
                ferrum_interfaces::execution_cost::CompletionOnlyReason::SearchInconclusive,
            );
        }
        let mut captured = match {
            let _stage = budget.stage(ControllerStage::Capture);
            self.capture_slo_controller_snapshot_with_completion(
                hint,
                Arc::clone(budget),
                optional_phase,
                &mut None,
                draft.as_mut(),
            )
        } {
            Ok(value) => value,
            Err(error) => {
                budget.record_unavailable(error.reason);
                self.record_controller(ControllerObservation {
                    obligations: error.obligations,
                    disposition: "unknown",
                    reason: error.reason,
                });
                if self.drop_slo_prefix_trajectory() {
                    return Ok(SloIterationPlan::Progressed);
                }
                return if self.completion_allowed() {
                    self.prepare_completion_with_draft(
                            hint,
                            budget,
                            draft,
                            if error.reason == "compute_budget_exhausted" {
                                ferrum_interfaces::execution_cost::CompletionOnlyReason::SearchInconclusive
                            } else {
                                ferrum_interfaces::execution_cost::CompletionOnlyReason::CostUnavailable
                            },
                        )
                } else {
                    Ok(if mode == ferrum_types::SloMode::Enforce {
                        SloIterationPlan::Idle
                    } else {
                        SloIterationPlan::Legacy
                    })
                };
            }
        };
        match self.plan_slo_prefix(&captured)? {
            prefix::PrefixPlan::CachePreparation {
                selected,
                maintenance,
            } => {
                captured.prefix_maintenance = Some(maintenance);
                drop(draft);
                return self
                    .prepare_slo_controller_wave_with_admission(captured, selected, hint, None);
            }
            prefix::PrefixPlan::CacheCapture(capture) => {
                drop(draft);
                let model_version = captured.snapshot.cost_model_version;
                return Ok(SloIterationPlan::PrefixMaintenance(
                    prefix::PreparedPrefixMaintenance::CacheCapture(
                        capture.prepare(captured.into_safety(), model_version),
                    ),
                ));
            }
            prefix::PrefixPlan::ReadyMaintenance(ready) => {
                drop(draft);
                let model_version = captured.snapshot.cost_model_version;
                return Ok(SloIterationPlan::PrefixMaintenance(
                    prefix::PreparedPrefixMaintenance::Ready(
                        ready.prepare(captured.into_safety(), model_version),
                    ),
                ));
            }
            prefix::PrefixPlan::None => {
                // No complete comparison was available. A new cold Capture
                // still requires a current physical projection whose exact
                // domain is missing cost; Restore is checked at native submit.
                if captured.poll_planning() {
                    if let Some(sampling) = self.prepare_prefix_sampling(budget, Some(&captured)) {
                        if captured.poll_planning() {
                            return Ok(SloIterationPlan::PrefixSampling(sampling));
                        }
                    }
                }
            }
            prefix::PrefixPlan::CostUnavailable => {
                if captured.poll_planning() {
                    if let Some(sampling) = self.prepare_prefix_sampling(budget, Some(&captured)) {
                        if captured.poll_planning() {
                            return Ok(SloIterationPlan::PrefixSampling(sampling));
                        }
                    }
                }
            }
            prefix::PrefixPlan::IncompleteCostLookup => {
                if captured.poll_planning() {
                    if let Some(sampling) = self.prepare_prefix_sampling(budget, Some(&captured)) {
                        if captured.poll_planning() {
                            return Ok(SloIterationPlan::PrefixSampling(sampling));
                        }
                    }
                }
                return self.prepare_completion_with_draft(
                    hint,
                    budget,
                    draft,
                    ferrum_interfaces::execution_cost::CompletionOnlyReason::SearchInconclusive,
                );
            }
            prefix::PrefixPlan::Changed => return Ok(SloIterationPlan::Progressed),
            prefix::PrefixPlan::Inconclusive => {
                return self.prepare_completion_with_draft(
                    hint,
                    budget,
                    draft,
                    ferrum_interfaces::execution_cost::CompletionOnlyReason::SearchInconclusive,
                );
            }
            prefix::PrefixPlan::Wave {
                selected,
                maintenance,
                checkpoint,
            } => {
                captured.prefix_maintenance = Some(maintenance);
                captured.prefix_checkpoint = Some(checkpoint);
                drop(draft);
                return self
                    .prepare_slo_controller_wave_with_admission(captured, selected, hint, None);
            }
            prefix::PrefixPlan::Maintenance {
                cohort,
                evidence,
                maintenance,
                valid_until,
                predicted_wall_ns,
            } => {
                drop(draft);
                let model_version = captured.snapshot.cost_model_version;
                return Ok(SloIterationPlan::PrefixMaintenance(
                    prefix::PreparedPrefixMaintenance::Rendezvous(
                        prefix::PreparedRendezvousMaintenance {
                            cohort,
                            evidence,
                            maintenance,
                            valid_until,
                            model_version,
                            predicted_wall_ns,
                            proof: captured.into_safety(),
                        },
                    ),
                ));
            }
        }
        self.slo_controller.lock().last_recovery_scope = captured
            .protection
            .needs_recovery()
            .then(|| Arc::clone(&captured.protection));
        if let Some(scope) = self.slo_controller.lock().last_recovery_scope.as_ref() {
            tracing::trace!(scope = ?scope.rows(), required_first_service = ?scope.required_first_service(),
                promises_closed = scope.new_time_promises_closed(), "immutable forward recovery obligations");
        }
        let (decision, admission, deferred, rejected_before_acceptance) = {
            let _stage = budget.stage(ControllerStage::SearchReplay);
            match self.propose_slo_time_admission(&captured) {
                Some(proposal) => (
                    proposal.decision,
                    proposal.pending,
                    proposal.deferred,
                    proposal.rejected_before_acceptance,
                ),
                None => (
                    Some(self.propose_slo_controller(&captured)),
                    None,
                    None,
                    false,
                ),
            }
        };
        if rejected_before_acceptance {
            self.record_controller(ControllerObservation {
                obligations: captured.snapshot.requests.len(),
                disposition: "rejected_before_acceptance",
                reason: "strict_time_admission",
            });
            return self.prepare_completion_with_draft(
                hint,
                budget,
                draft,
                ferrum_interfaces::execution_cost::CompletionOnlyReason::SearchInconclusive,
            );
        }
        if let Some(reason) = deferred {
            self.record_controller(ControllerObservation {
                obligations: captured.snapshot.requests.len(),
                disposition: "deferred_time_promise",
                reason: match reason {
                    ferrum_scheduler::implementations::continuous::slo_planner::time_admission::TimeAdmissionDeferReason::ActiveLimit => "time_active_limit",
                    ferrum_scheduler::implementations::continuous::slo_planner::time_admission::TimeAdmissionDeferReason::ExistingObligationAtRisk => "existing_obligation_at_risk",
                },
            });
            // No second search or fabricated Unknown. Already accepted active
            // work keeps the same safe completion path and physical guard.
            return if mode == ferrum_types::SloMode::Observe {
                Ok(SloIterationPlan::Legacy)
            } else {
                self.prepare_completion_with_draft(
                    hint,
                    budget,
                    draft,
                    ferrum_interfaces::execution_cost::CompletionOnlyReason::SearchInconclusive,
                )
            };
        }
        let Some(decision) = decision else {
            return Err(FerrumError::scheduler(
                "time admission omitted its disposition",
            ));
        };
        budget.record_search(&decision);
        self.record_controller(ControllerObservation {
            obligations: captured.snapshot.requests.len(),
            disposition: match decision {
                PlanningDecision::FeasibleWithinHorizon { .. } => "feasible_within_horizon",
                PlanningDecision::ProtectedWithinHorizon { .. } => "protected_within_horizon",
                PlanningDecision::ProvenImpossibleUnderModel { .. } => "impossible_under_model",
                PlanningDecision::Unknown { .. } => "unknown",
            },
            reason: match &decision {
                PlanningDecision::Unknown { reason, .. } => unknown_label(*reason),
                PlanningDecision::ProtectedWithinHorizon { .. } => "partial_forward_obligations",
                _ => "complete_obligation_set",
            },
        });
        if mode == ferrum_types::SloMode::Observe {
            return Ok(SloIterationPlan::Legacy);
        }
        let first_wave = match decision {
            PlanningDecision::FeasibleWithinHorizon { first_wave, .. }
            | PlanningDecision::ProtectedWithinHorizon { first_wave, .. } => first_wave,
            _ => {
                if self.completion_allowed() {
                    return self.prepare_completion_with_draft(
                        hint,
                        budget,
                        draft,
                        ferrum_interfaces::execution_cost::CompletionOnlyReason::SearchInconclusive,
                    );
                }
                return Ok(SloIterationPlan::Idle);
            }
        };
        drop(draft);
        let _stage = budget.stage(ControllerStage::Publication);
        self.prepare_slo_controller_wave_with_admission(captured, first_wave, hint, admission)
    }

    fn record_controller(&self, observation: ControllerObservation) {
        {
            let mut state = self.slo_controller.lock();
            state.observations = state.observations.saturating_add(1);
            state.last_observation = Some(observation);
        }
        counter!("ferrum.engine.slo_controller_decisions_total",
            "decision" => observation.disposition, "reason" => observation.reason)
        .increment(1);
        tracing::trace!(
            obligations = observation.obligations,
            decision = observation.disposition,
            reason = observation.reason,
            "bounded SLO controller observation"
        );
    }
}

fn unknown_label(reason: PlanningUnknownReason) -> &'static str {
    match reason {
        PlanningUnknownReason::InvalidConfiguration => "invalid_configuration",
        PlanningUnknownReason::InvalidSnapshot => "invalid_snapshot",
        PlanningUnknownReason::ArithmeticOverflow => "arithmetic_overflow",
        PlanningUnknownReason::ClockMovedBackwards => "clock_moved_backwards",
        PlanningUnknownReason::ModelVersionMismatch => "model_version_mismatch",
        PlanningUnknownReason::UnknownReadiness => "unknown_readiness",
        PlanningUnknownReason::InvalidShapeEvidence => "invalid_shape_evidence",
        PlanningUnknownReason::ShapeCapacity => "shape_capacity",
        PlanningUnknownReason::CostUnavailable => "cost_unavailable",
        PlanningUnknownReason::ShapeUnavailable => "shape_unavailable",
        PlanningUnknownReason::UnknownResourceEvidence => "resource_unknown",
        PlanningUnknownReason::ComputeBudgetExhausted => "compute_budget_exhausted",
        PlanningUnknownReason::MissingReferenceWork => "missing_reference_work",
        PlanningUnknownReason::OutputOrResourceBlocked => "output_or_resource_blocked",
        PlanningUnknownReason::UnmodeledMaintenance => "unmodeled_maintenance",
        PlanningUnknownReason::HorizonInsufficient => "horizon_insufficient",
        PlanningUnknownReason::SearchIncomplete => "search_incomplete",
        PlanningUnknownReason::NoWork => "no_work",
        PlanningUnknownReason::RecoveryConflict => "recovery_fairness_conflict",
    }
}
