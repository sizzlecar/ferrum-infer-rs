//! Frozen, bounded joint verdicts consumed before executor admission. Prediction
//! runs outside queue locks; exact queue incarnations are revalidated inside.

use super::joint_selection::{AdmissionCapMode, JointAdmissionEvaluation};
use super::*;
use crate::cost_prediction::{ComparisonKind, DecisionConsumerOutcome};
use crate::joint_policy::{JointFeasibility, PolicyChoice, Unsupported};
use ferrum_types::{KnownInfeasibleAdmissionPolicy, SloStrategy};

mod phase_commit;

#[derive(Clone)]
struct Member {
    key: PrefixRequestKey,
    phase: RequestPhase,
    prefilled: usize,
    decoded: usize,
    generation: LogicalWorkGeneration,
    priority: Priority,
    prompt_tokens: usize,
    phase_cap: Option<PrefillPhaseCap>,
    frontier: pressure::LogicalWorkFrontier,
    execution_ceiling: Option<usize>,
    restored_tokens: usize,
}

impl Member {
    fn capture(
        scheduler: &ContinuousBatchScheduler,
        request: &ContinuousBatchRequest,
    ) -> std::result::Result<Self, Unsupported> {
        Ok(Self {
            key: PrefixRequestKey::from_request(request).ok_or(Unsupported::InvalidState)?,
            phase: request.phase,
            prefilled: request.prefill_chunk_offset,
            decoded: request.decode_tokens,
            generation: request.logical_work_frontier.progress_generation(),
            priority: request.inner.request.priority,
            prompt_tokens: scheduler
                .prompt_token_estimate(request)
                .ok_or(Unsupported::InvalidState)?,
            phase_cap: request.joint_prefill_phase_cap,
            frontier: request.logical_work_frontier.clone(),
            execution_ceiling: request.prefill_execution_chunk_ceiling,
            restored_tokens: request.prefix_restore.restored_tokens(),
        })
    }

    fn matches(
        &self,
        scheduler: &ContinuousBatchScheduler,
        request: &ContinuousBatchRequest,
    ) -> bool {
        self.key.matches(request)
            && (self.phase == request.phase
                || (self.phase == RequestPhase::Waiting
                    && request.phase == RequestPhase::Prefilling))
            && self.prefilled == request.prefill_chunk_offset
            && self.decoded == request.decode_tokens
            && self.generation == request.logical_work_frontier.progress_generation()
            && self.priority == request.inner.request.priority
            && scheduler.prompt_token_estimate(request) == Some(self.prompt_tokens)
            && self.phase_cap == request.joint_prefill_phase_cap
            && self.execution_ceiling == request.prefill_execution_chunk_ceiling
            && self.restored_tokens == request.prefix_restore.restored_tokens()
    }

    fn matches_exact(
        &self,
        scheduler: &ContinuousBatchScheduler,
        request: &ContinuousBatchRequest,
    ) -> bool {
        self.matches(scheduler, request)
            && self.phase == request.phase
            && self.frontier == request.logical_work_frontier
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Verdict {
    StaticFeasible,
    AllKnownInfeasible,
    RequiresAction,
    Unknown(Unsupported),
}

pub(crate) struct FrozenAdmissionGate {
    members: Vec<Member>,
    verdict: Verdict,
    created_at: Instant,
    apply_before: Instant,
}

impl ContinuousBatchScheduler {
    /// Call before every asynchronous admission round, including fill-first
    /// retries. A supported action may install a positive prefill cap, followed
    /// by a fresh comparison. This never reserves resources or promotes work.
    pub fn prepare_joint_admission_gate(
        &self,
        hint: &BatchHint,
        round: &mut TimePlanningRound,
        now: Instant,
    ) {
        let started = Instant::now();
        let prior = round.prediction_elapsed;
        self.freeze_joint_admission_gate(hint, round, now, started, prior);
        round.prediction_elapsed = prior.saturating_add(started.elapsed());
    }

    fn freeze_joint_admission_gate(
        &self,
        hint: &BatchHint,
        round: &mut TimePlanningRound,
        now: Instant,
        started: Instant,
        prior: Duration,
    ) {
        round.admission_gate = None;
        round
            .capacity_initial_new_sequences
            .get_or_insert_with(|| self.waiting_count());
        if round.config.strategy != SloStrategy::Joint || self.waiting_count() == 0 {
            return;
        }
        let Some(remaining) = Duration::from_micros(round.config.planning.max_elapsed_us)
            .checked_sub(round.prediction_elapsed)
        else {
            return;
        };
        let Some(mut apply_before) = now.checked_add(remaining) else {
            return;
        };
        let Ok(mut members) = self.capture_joint_admission_members(round) else {
            return;
        };
        let mut mode = AdmissionCapMode::Propose;
        let mut comparison_now = now;
        // At most one commitment and one fresh comparison, with the original
        // query counters and wall-time allowance. No recursive replenishment.
        loop {
            round.prediction_elapsed = prior.saturating_add(started.elapsed());
            let Some(member_bytes) = members.len().checked_mul(std::mem::size_of::<Member>())
            else {
                return;
            };
            round.statistics.joint_comparisons += 1;
            let result =
                self.evaluate_joint_admission(hint, round, comparison_now, mode, member_bytes);
            if let Ok(evaluation) = &result {
                apply_before = apply_before.min(evaluation.decision.valid_until);
            }
            let verdict = self.record_joint_admission_result(round, &result);
            if verdict == Verdict::RequiresAction && matches!(mode, AdmissionCapMode::Propose) {
                if let Ok(JointAdmissionEvaluation {
                    prefill_cap: Some(proposal),
                    ..
                }) = result
                {
                    if self.commit_joint_admission_phase(&members, &proposal, apply_before) {
                        round.statistics.joint_admission_prefill_phase_commits = round
                            .statistics
                            .joint_admission_prefill_phase_commits
                            .saturating_add(1);
                        // A cap is an enduring policy commitment, not a grant
                        // of admission. Even a failed fresh comparison leaves
                        // it installed and uses the declared Unknown policy.
                        drop(proposal);
                        drop(members);
                        let Ok(fresh) = self.capture_joint_admission_members(round) else {
                            return;
                        };
                        members = fresh;
                        let Some(fresh_now) = now.checked_add(started.elapsed()) else {
                            return;
                        };
                        comparison_now = fresh_now;
                        mode = AdmissionCapMode::ExistingOnly;
                        continue;
                    }
                }
            }
            round.admission_gate = Some(FrozenAdmissionGate {
                members,
                verdict,
                created_at: comparison_now,
                apply_before,
            });
            let waiting = self.waiting_queue.read();
            self.revalidate_joint_admission_gate(&waiting, round);
            return;
        }
    }

    fn capture_joint_admission_members(
        &self,
        round: &mut TimePlanningRound,
    ) -> std::result::Result<Vec<Member>, Unsupported> {
        let waiting = self.waiting_queue.read();
        round.capacity_initial_new_sequences = Some(
            round
                .capacity_initial_new_sequences
                .unwrap_or_default()
                .max(waiting.len()),
        );
        let prefill = self.prefill_queue.read();
        let decode = self.decode_queue.read();
        let count = waiting
            .len()
            .saturating_add(prefill.len())
            .saturating_add(decode.requests.len());
        if count == 0 || count > round.config.planning.max_participants {
            return Err(Unsupported::ParticipantBudget);
        }
        let Some(mut bytes) = round
            .request_clocks()
            .and_then(|clocks| {
                clocks.len().checked_mul(std::mem::size_of::<(
                    RequestId,
                    crate::cost_prediction::RequestClockFacts,
                )>())
            })
            .and_then(|bytes| bytes.checked_add(round.request_fact_bytes))
            .and_then(|bytes| {
                bytes.checked_add(
                    round
                        .persistent_capacity()
                        .map_or(0, |proof| proof.retained_bytes()),
                )
            })
        else {
            return Err(Unsupported::ParticipantBudget);
        };
        let payload = count
            .checked_mul(std::mem::size_of::<Member>())
            .ok_or(Unsupported::ParticipantBudget)?;
        if bytes
            .checked_add(payload)
            .is_none_or(|bytes| bytes > round.config.planning.max_request_fact_bytes)
        {
            return Err(Unsupported::ParticipantBudget);
        }
        let mut members = Vec::new();
        members
            .try_reserve_exact(count)
            .map_err(|_| Unsupported::ParticipantBudget)?;
        for request in prefill
            .iter()
            .chain(decode.requests.values())
            .chain(waiting.iter())
        {
            bytes = match bytes.checked_add(std::mem::size_of::<Member>()) {
                Some(bytes) if bytes <= round.config.planning.max_request_fact_bytes => bytes,
                _ => return Err(Unsupported::ParticipantBudget),
            };
            let member = Member::capture(self, request)?;
            members.push(member);
        }
        Ok(members)
    }

    fn record_joint_admission_result(
        &self,
        round: &mut TimePlanningRound,
        result: &std::result::Result<JointAdmissionEvaluation, Unsupported>,
    ) -> Verdict {
        let verdict = match result {
            Ok(evaluation)
                if evaluation.decision.static_risk.classify(&round.config)
                    == JointFeasibility::Feasible =>
            {
                Verdict::StaticFeasible
            }
            Ok(evaluation) if evaluation.decision.choice == PolicyChoice::AllKnownInfeasible => {
                Verdict::AllKnownInfeasible
            }
            Ok(evaluation)
                if matches!(evaluation.decision.choice, PolicyChoice::Alternative(_)) =>
            {
                Verdict::RequiresAction
            }
            Ok(_) => Verdict::Unknown(Unsupported::FeasibilityUnresolved),
            Err(error) => Verdict::Unknown(*error),
        };
        if let Ok(evaluation) = result {
            let outcome = match verdict {
                Verdict::StaticFeasible => Some(DecisionConsumerOutcome::StaticFeasible),
                Verdict::AllKnownInfeasible => Some(DecisionConsumerOutcome::AllKnownInfeasible),
                Verdict::RequiresAction => Some(DecisionConsumerOutcome::RequiresAction),
                Verdict::Unknown(Unsupported::FeasibilityUnresolved) => {
                    Some(DecisionConsumerOutcome::FeasibilityUnresolved)
                }
                _ => None,
            };
            if let Some(outcome) = outcome {
                round.statistics.joint_decision.consume(
                    ComparisonKind::Admission,
                    outcome,
                    &evaluation.decision.observation,
                );
            }
        }
        match verdict {
            Verdict::Unknown(reason) => round.statistics.record_joint_unsupported(reason),
            Verdict::RequiresAction => {
                round.statistics.joint_unresolved += 1;
                round.statistics.joint_admission_action_required = round
                    .statistics
                    .joint_admission_action_required
                    .saturating_add(1);
            }
            _ => {}
        }
        verdict
    }

    /// Caller already holds the waiting lock. Permit the exact zero-progress
    /// promotions from this scan, but invalidate on arrival, cancellation,
    /// priority/progress change, native deferral or same-ID replacement.
    pub(super) fn revalidate_joint_admission_gate(
        &self,
        waiting: &DynamicAdmissionQueue<ContinuousBatchRequest>,
        round: &mut TimePlanningRound,
    ) {
        let Some(gate) = round.admission_gate.as_ref() else {
            return;
        };
        let prefill = self.prefill_queue.read();
        let decode = self.decode_queue.read();
        let mut valid = !waiting.has_capacity_deferrals()
            && gate.members.len() == waiting.len() + prefill.len() + decode.requests.len()
            && prefill
                .iter()
                .chain(decode.requests.values())
                .chain(waiting.iter())
                .all(|request| {
                    gate.members
                        .iter()
                        .any(|member| member.matches(self, request))
                });
        let mut previous = None;
        for request in waiting.iter() {
            let position = gate.members.iter().position(|member| {
                member.phase == RequestPhase::Waiting && member.matches(self, request)
            });
            if position.is_none()
                || previous
                    .zip(position)
                    .is_some_and(|(old, next)| old >= next)
            {
                valid = false;
            }
            previous = position;
        }
        if !valid {
            round.admission_gate = None;
        }
    }

    pub(super) fn joint_admission_time_decision(
        &self,
        request: &ContinuousBatchRequest,
        round: &mut TimePlanningRound,
        now: Instant,
    ) -> time_policy::TimeDecision {
        use time_policy::TimeDecision;
        let tokens = self.remaining_prefill_tokens(request).max(1);
        // Logical admission survives physical eviction/recompute. Never strand
        // an accepted request because a later statistical model is unavailable.
        if request.logical_work_frontier.work_kind() != pressure::LogicalWorkKind::Waiting
            || request.logical_work_frontier.progress_generation() != LogicalWorkGeneration::ZERO
            || request.capacity_deferred_from_decode
        {
            return TimeDecision::Allow(tokens);
        }
        let verdict = round
            .admission_gate
            .as_ref()
            .filter(|gate| {
                now >= gate.created_at
                    && now < gate.apply_before
                    && gate.members.iter().any(|member| {
                        member.phase == RequestPhase::Waiting && member.matches(self, request)
                    })
            })
            .map_or(Verdict::Unknown(Unsupported::InvalidState), |gate| {
                gate.verdict
            });
        match verdict {
            Verdict::StaticFeasible => TimeDecision::Allow(tokens),
            Verdict::AllKnownInfeasible => {
                round.statistics.joint_known_infeasible += 1;
                if let KnownInfeasibleAdmissionPolicy::Wait { max_wait_ms } =
                    round.config.joint.known_infeasible_admission
                {
                    if let Some(deadline) = request
                        .inner
                        .ingress()
                        .instant()
                        .checked_add(Duration::from_millis(max_wait_ms))
                        .filter(|deadline| *deadline > now)
                    {
                        round.retry_at =
                            Some(round.retry_at.map_or(deadline, |old| old.min(deadline)));
                        round.time_blocked = true;
                        round.statistics.held_candidates += 1;
                        return TimeDecision::Hold;
                    }
                }
                round.statistics.rejected_candidates += 1;
                TimeDecision::Reject(FerrumError::request_validation(
                    "joint SLO model finds every supported admission action infeasible",
                ))
            }
            Verdict::Unknown(Unsupported::StoppingCoverage) => round.unknown_policy(
                round.config.unknown_stopping,
                crate::cost_prediction::CostUnknownReason::NoCoverage,
                request,
                tokens,
                now,
            ),
            Verdict::RequiresAction | Verdict::Unknown(_) => round.unknown_policy(
                round.config.unknown_cost,
                crate::cost_prediction::CostUnknownReason::NoCoverage,
                request,
                tokens,
                now,
            ),
        }
    }
}

#[cfg(test)]
mod tests;
