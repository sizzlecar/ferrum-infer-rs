//! Default completion-first requests may acquire a finite common time witness
//! after acceptance. This is not strict before-acceptance admission: real
//! physical preparation remains owned by the existing admission protocol.
use super::*;
use ferrum_scheduler::implementations::continuous::slo_planner::time_admission::*;

mod defer;
mod state;
pub(super) use defer::TimeActivationState;
pub(super) use state::PendingTimeWitness;
pub(in crate::continuous_engine) use state::SequenceTimeAdmission;
use state::{StartedTimeWitness, TimeAdmissionAssessment, TimeAdmissionAssessmentKind};

pub(super) struct TimeAdmissionProposal {
    pub decision: Option<PlanningDecision>,
    pub deferred: Option<TimeAdmissionDeferReason>,
    pub pending: Option<PendingTimeWitness>,
}

impl TimeAdmissionProposal {
    fn unknown(reason: PlanningUnknownReason) -> Self {
        Self {
            decision: Some(PlanningDecision::Unknown {
                reason,
                search: Default::default(),
            }),
            deferred: None,
            pending: None,
        }
    }
}

impl EngineInner {
    /// At most one admission search replaces the ordinary search in this
    /// transaction. A cost/reference gap does not schedule a second search or
    /// convert an already accepted request into a rejection.
    pub(super) fn propose_slo_time_admission(
        &self,
        captured: &ControllerSnapshot,
    ) -> Option<TimeAdmissionProposal> {
        if self.config.scheduler.slo.admission.time_policy
            != ferrum_types::SloTimeAdmissionPolicy::CompleteRequests
        {
            return None;
        }
        if captured.protection.needs_recovery() {
            // Keep one common search. A protected continuation cannot create
            // another pending promise, even when its new target is healthy.
            if let Ok(Some(target)) = self.time_admission_target(captured) {
                let _ = self.record_slo_time_assessment(
                    captured,
                    &target,
                    TimeAdmissionDecision::Defer {
                        reason: TimeAdmissionDeferReason::ExistingObligationAtRisk,
                        wait: TimeAdmissionWait {
                            review_at_ns: None,
                            strict_expiry_at_ns: None,
                            snapshot_generation: captured.snapshot.generation,
                            cost_model_version: captured.snapshot.cost_model_version,
                        },
                    },
                );
            }
            return None;
        }
        let target = match self.time_admission_target(captured) {
            Ok(Some(target)) => target,
            Ok(None) => return None,
            Err(reason) => return Some(TimeAdmissionProposal::unknown(reason)),
        };
        let mut active = Vec::with_capacity(captured.snapshot.requests.len());
        for (row, queued) in captured
            .snapshot
            .requests
            .iter()
            .zip(captured.queue.requests())
        {
            if !captured.poll_planning() {
                return Some(TimeAdmissionProposal::unknown(
                    PlanningUnknownReason::ComputeBudgetExhausted,
                ));
            }
            if row.key != target
                && !row.timing.completed()
                && matches!(
                    queued.queue,
                    PlanningQueueKind::Prefill | PlanningQueueKind::Decode
                )
            {
                active.push(row.key.clone());
            }
        }
        let planner = BoundedSloPlanner {
            settings: BoundedPlannerSettings {
                search: self.config.scheduler.slo.planner.clone(),
                ..Default::default()
            },
        };
        let cost = AnchoredPlanningCostModel::new(captured.model.as_ref(), captured.anchor);
        let shapes = shape::ExecutorShape {
            engine: self,
            captured,
        };
        let evaluator = TimeAdmissionExecutionEvaluator {
            policy: &self.config.scheduler.slo.admission,
            planner: &planner,
            model: &cost,
            execution: &shapes,
        };
        let Ok(window) = captured.planning_window() else {
            return Some(TimeAdmissionProposal::unknown(
                PlanningUnknownReason::ClockMovedBackwards,
            ));
        };
        let decision = match captured
            .origin
            .assess_admission_execution_with_budget_window(
                &evaluator,
                TimeAdmissionQuery {
                    snapshot: &captured.snapshot,
                    target: &target,
                    active: &active,
                    boundary: TimeAdmissionBoundary::Accepted,
                },
                window,
                slo_clock_now,
            ) {
            Ok(decision) => decision,
            Err(_) => {
                return Some(TimeAdmissionProposal::unknown(
                    PlanningUnknownReason::ClockMovedBackwards,
                ))
            }
        };
        Some(self.record_slo_time_assessment(captured, &target, decision))
    }

    fn time_admission_target(
        &self,
        captured: &ControllerSnapshot,
    ) -> std::result::Result<Option<RequestWorkKey>, PlanningUnknownReason> {
        let sequences = self
            .sequences
            .try_read()
            .ok_or(PlanningUnknownReason::UnknownReadiness)?;
        let mut selected: Option<(&RequestWorkKey, Option<Instant>, Instant)> = None;
        for request in &captured.snapshot.requests {
            if !captured.poll_planning() {
                return Err(PlanningUnknownReason::ComputeBudgetExhausted);
            }
            if request.timing.committed_tokens != 0
                || !matches!(request.phase, RequestPhaseView::Prefill(_))
            {
                continue;
            }
            let sequence = sequences
                .get(&request.key.request_id)
                .ok_or(PlanningUnknownReason::InvalidSnapshot)?;
            let Some(state) = sequence.time_admission.as_ref() else {
                continue;
            };
            if state.started.is_some() || !state.matches(sequence) {
                continue;
            }
            if state
                .deferred
                .as_ref()
                .is_some_and(|wait| wait.pending(captured, slo_clock_now()))
            {
                continue;
            }
            let order = (state.last_assessment.map(|last| last.at), state.ingress);
            if selected.is_none_or(|(_, last, ingress)| order < (last, ingress)) {
                selected = Some((&request.key, order.0, order.1));
            }
        }
        Ok(selected.map(|(key, _, _)| key.clone()))
    }

    fn record_slo_time_assessment(
        &self,
        captured: &ControllerSnapshot,
        target: &RequestWorkKey,
        decision: TimeAdmissionDecision,
    ) -> TimeAdmissionProposal {
        let review_at = match &decision {
            TimeAdmissionDecision::Defer { wait, .. } => match wait.review_at_ns {
                Some(at) => match captured.origin.instant_at_ns(at) {
                    Ok(at) => Some(at),
                    Err(_) => {
                        return TimeAdmissionProposal::unknown(
                            PlanningUnknownReason::ArithmeticOverflow,
                        )
                    }
                },
                None => None,
            },
            _ => None,
        };
        let kind = match &decision {
            TimeAdmissionDecision::Admit { .. } => TimeAdmissionAssessmentKind::Feasible,
            TimeAdmissionDecision::Unknown { reason, .. } => {
                TimeAdmissionAssessmentKind::Unknown(*reason)
            }
            TimeAdmissionDecision::Defer { reason, .. } => {
                TimeAdmissionAssessmentKind::Deferred(*reason)
            }
            TimeAdmissionDecision::BestEffort { .. } => {
                TimeAdmissionAssessmentKind::AlreadyImpossible
            }
            TimeAdmissionDecision::Reject { .. } => {
                TimeAdmissionAssessmentKind::InvalidAcceptanceBoundary
            }
        };
        let Some(mut sequences) = self.sequences.try_write() else {
            return TimeAdmissionProposal::unknown(PlanningUnknownReason::UnknownReadiness);
        };
        let Some(sequence) = sequences.get_mut(&target.request_id) else {
            return TimeAdmissionProposal::unknown(PlanningUnknownReason::InvalidSnapshot);
        };
        let Some(fence) = captured
            .fences
            .iter()
            .find(|fence| fence.key.request_id == target.request_id)
        else {
            return TimeAdmissionProposal::unknown(PlanningUnknownReason::InvalidSnapshot);
        };
        if !fence.matches_sequence(sequence)
            || !sequence
                .time_admission
                .as_ref()
                .is_some_and(|state| state.started.is_none() && state.matches(sequence))
        {
            return TimeAdmissionProposal::unknown(PlanningUnknownReason::InvalidSnapshot);
        }
        let state = sequence.time_admission.as_mut().unwrap();
        let assessment = TimeAdmissionAssessment {
            at: slo_clock_now(),
            snapshot_generation: captured.snapshot.generation,
            model_version: captured.snapshot.cost_model_version,
            kind,
        };
        state.last_assessment = Some(assessment);
        state.deferred = match &decision {
            TimeAdmissionDecision::Defer { .. } => Some(defer::TimeAdmissionReview {
                review_at,
                queue_iteration: captured.queue.iteration(),
                wake_epochs: captured.queue.wake_epochs(),
                obligations: captured.snapshot.requests.len(),
                model_version: captured.model.model_version(),
                capacity_availability: captured.queue.capacity_availability().to_vec(),
                queue_owners: captured
                    .queue
                    .requests()
                    .iter()
                    .map(|row| (row.key.clone(), row.queue))
                    .collect(),
            }),
            _ => None,
        };
        counter!("ferrum.engine.slo_time_admission_assessments_total", "decision" => assessment.kind.label()).increment(1);
        tracing::trace!(request_id = %target.request_id,
            snapshot_generation = assessment.snapshot_generation,
            model_version = assessment.model_version, decision = ?assessment.kind,
            "assessed accepted request against complete finite obligation set");
        match decision {
            TimeAdmissionDecision::Admit {
                first_wave,
                witness,
                search,
            } => {
                let Ok(validated_through) =
                    captured.origin.instant_at_ns(witness.validated_through_ns)
                else {
                    return TimeAdmissionProposal::unknown(
                        PlanningUnknownReason::ArithmeticOverflow,
                    );
                };
                let pending = PendingTimeWitness {
                    request_id: target.request_id.clone(),
                    owner: Arc::clone(&sequence.stream_projection_identity),
                    work_generation: fence.generation,
                    ingress: state.ingress,
                    original_input_tokens: state.original_input_tokens,
                    maximum_output_tokens: state.maximum_output_tokens,
                    evidence: StartedTimeWitness {
                        snapshot_generation: captured.snapshot.generation,
                        model_version: captured.snapshot.cost_model_version,
                        validated_through,
                        obligations_beyond_horizon: witness
                            .requests_with_obligations_beyond_horizon,
                    },
                };
                TimeAdmissionProposal {
                    decision: Some(PlanningDecision::FeasibleWithinHorizon {
                        first_wave,
                        witness,
                        search,
                    }),
                    deferred: None,
                    pending: Some(pending),
                }
            }
            TimeAdmissionDecision::BestEffort { reason } => TimeAdmissionProposal {
                decision: Some(PlanningDecision::ProvenImpossibleUnderModel {
                    reason,
                    model_version: captured.snapshot.cost_model_version,
                    snapshot_generation: captured.snapshot.generation,
                }),
                deferred: None,
                pending: None,
            },
            TimeAdmissionDecision::Unknown { reason, .. } => {
                TimeAdmissionProposal::unknown(match reason {
                    TimeAdmissionUnknown::Planning(reason) => reason,
                    TimeAdmissionUnknown::LengthStatisticsUnavailable
                    | TimeAdmissionUnknown::OutsideSequenceCoverage => {
                        PlanningUnknownReason::CostUnavailable
                    }
                })
            }
            TimeAdmissionDecision::Defer { reason, .. } => TimeAdmissionProposal {
                decision: None,
                deferred: Some(reason),
                pending: None,
            },
            TimeAdmissionDecision::Reject { .. } => {
                TimeAdmissionProposal::unknown(PlanningUnknownReason::InvalidSnapshot)
            }
        }
    }
}

#[cfg(test)]
mod tests;
