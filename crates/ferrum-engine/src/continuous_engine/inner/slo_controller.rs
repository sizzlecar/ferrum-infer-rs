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
mod budget;
use budget::{ControllerAudit, ControllerBudget, ControllerStage};
mod commit;
pub(super) use commit::ControllerCommitFence;
pub(super) mod calibration;
mod completion;
mod dispatch;
mod maintenance;
mod owner;
mod recovery;
mod resources;
mod retry;
mod shape;
mod snapshot;
mod submit;
#[cfg(test)]
pub(in crate::continuous_engine) mod tests;
mod work_envelope;

pub(in crate::continuous_engine) enum SloIterationPlan {
    Legacy,
    Idle,
    Selected(submit::PreparedControllerWave),
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
    budget: Arc<ControllerBudget>,
    queue: PlanningQueueSnapshot,
    snapshot: SchedulerSnapshot,
    origin: PlanningTimeOrigin,
    anchor: PlanningCostClockAnchor,
    resources: ResourcePlanningView,
    route: ferrum_interfaces::vnext::ExecutionCostRouteView,
    fences: Vec<EngineFence>,
    model: Arc<dyn PlanningCostModel + Send + Sync>,
    protection: Arc<PlanningObligationSet>,
    recovery_peers: Vec<recovery::RecoveryPeer>,
}

struct ControllerSafetyProof {
    recovery_peers: Vec<recovery::RecoveryPeer>,
    protection: Option<Arc<PlanningObligationSet>>,
    budget: Arc<ControllerBudget>,
    queue: PlanningQueueSnapshot,
    fences: Vec<EngineFence>,
}

impl ControllerSnapshot {
    fn into_safety(self) -> ControllerSafetyProof {
        ControllerSafetyProof {
            protection: self.protection.needs_recovery().then_some(self.protection),
            recovery_peers: self.recovery_peers,
            budget: self.budget,
            queue: self.queue,
            fences: self.fences,
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
        let mode = self.config.scheduler.slo.mode;
        if mode == ferrum_types::SloMode::Off {
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
        let completion = self.slo_controller.lock().completion_next;
        if let Some(reason) = completion.filter(|_| self.completion_allowed()) {
            return self.prepare_completion_controller(hint, budget, reason);
        }
        let captured =
            match {
                let _stage = budget.stage(ControllerStage::Capture);
                self.capture_slo_controller_snapshot(hint, Arc::clone(budget))
            } {
                Ok(value) => value,
                Err(error) => {
                    budget.record_unavailable(error.reason);
                    self.record_controller(ControllerObservation {
                        obligations: error.obligations,
                        disposition: "unknown",
                        reason: error.reason,
                    });
                    return if self.completion_allowed() {
                        self.prepare_completion_controller(hint, budget,
                        ferrum_interfaces::execution_cost::CompletionOnlyReason::CostUnavailable)
                    } else {
                        Ok(if mode == ferrum_types::SloMode::Enforce {
                            SloIterationPlan::Idle
                        } else {
                            SloIterationPlan::Legacy
                        })
                    };
                }
            };
        self.slo_controller.lock().last_recovery_scope = captured
            .protection
            .needs_recovery()
            .then(|| Arc::clone(&captured.protection));
        if let Some(scope) = self.slo_controller.lock().last_recovery_scope.as_ref() {
            tracing::trace!(scope = ?scope.rows(), required_first_service = ?scope.required_first_service(),
                promises_closed = scope.new_time_promises_closed(), "immutable forward recovery obligations");
        }
        let (decision, admission, deferred) = {
            let _stage = budget.stage(ControllerStage::SearchReplay);
            match self.propose_slo_time_admission(&captured) {
                Some(proposal) => (proposal.decision, proposal.pending, proposal.deferred),
                None => (Some(self.propose_slo_controller(&captured)), None, None),
            }
        };
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
                self.prepare_completion_controller(
                    hint,
                    budget,
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
                    return self.prepare_completion_controller(
                        hint,
                        budget,
                        ferrum_interfaces::execution_cost::CompletionOnlyReason::SearchInconclusive,
                    );
                }
                return Ok(SloIterationPlan::Idle);
            }
        };
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
