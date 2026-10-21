//! Frozen, bounded joint verdicts consumed before executor admission. Prediction
//! runs outside queue locks; exact queue incarnations are revalidated inside.

use super::*;
use crate::joint_policy::{JointFeasibility, PolicyChoice, Unsupported};
use ferrum_types::{KnownInfeasibleAdmissionPolicy, SloStrategy};

#[derive(Clone)]
struct Member {
    key: PrefixRequestKey,
    phase: RequestPhase,
    prefilled: usize,
    decoded: usize,
    generation: LogicalWorkGeneration,
    priority: Priority,
}

impl Member {
    fn capture(request: &ContinuousBatchRequest) -> std::result::Result<Self, Unsupported> {
        Ok(Self {
            key: PrefixRequestKey::from_request(request).ok_or(Unsupported::InvalidState)?,
            phase: request.phase,
            prefilled: request.prefill_chunk_offset,
            decoded: request.decode_tokens,
            generation: request.logical_work_frontier.progress_generation(),
            priority: request.inner.request.priority,
        })
    }

    fn matches(&self, request: &ContinuousBatchRequest) -> bool {
        self.key.matches(request)
            && (self.phase == request.phase
                || (self.phase == RequestPhase::Waiting
                    && request.phase == RequestPhase::Prefilling))
            && self.prefilled == request.prefill_chunk_offset
            && self.decoded == request.decode_tokens
            && self.generation == request.logical_work_frontier.progress_generation()
            && self.priority == request.inner.request.priority
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
    /// retries. This function never reserves resources or mutates queue state.
    pub fn prepare_joint_admission_gate(
        &self,
        hint: &BatchHint,
        round: &mut TimePlanningRound,
        now: Instant,
    ) {
        let started = Instant::now();
        let prior = round.prediction_elapsed;
        self.freeze_joint_admission_gate(hint, round, now);
        round.prediction_elapsed = prior.saturating_add(started.elapsed());
    }

    fn freeze_joint_admission_gate(
        &self,
        hint: &BatchHint,
        round: &mut TimePlanningRound,
        now: Instant,
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
        let members = {
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
                return;
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
            else {
                return;
            };
            let mut members = Vec::with_capacity(count);
            for request in prefill
                .iter()
                .chain(decode.requests.values())
                .chain(waiting.iter())
            {
                bytes = match bytes.checked_add(std::mem::size_of::<Member>()) {
                    Some(bytes) if bytes <= round.config.planning.max_request_fact_bytes => bytes,
                    _ => return,
                };
                let Ok(member) = Member::capture(request) else {
                    return;
                };
                members.push(member);
            }
            members
        };
        round.statistics.joint_comparisons += 1;
        let result = self.evaluate_joint_admission(hint, round, now);
        if let Ok(decision) = &result {
            apply_before = apply_before.min(decision.valid_until);
        }
        let verdict = match result {
            Ok(decision)
                if decision.static_risk.classify(&round.config) == JointFeasibility::Feasible =>
            {
                Verdict::StaticFeasible
            }
            Ok(decision) if decision.choice == PolicyChoice::AllKnownInfeasible => {
                Verdict::AllKnownInfeasible
            }
            Ok(decision) if matches!(decision.choice, PolicyChoice::Alternative(_)) => {
                Verdict::RequiresAction
            }
            Ok(_) => Verdict::Unknown(Unsupported::Cost(
                crate::cost_prediction::CostUnknownReason::NoCoverage,
            )),
            Err(error) => Verdict::Unknown(error),
        };
        match verdict {
            Verdict::Unknown(Unsupported::UnknownKvCapacity | Unsupported::WaitingResources) => {
                round.statistics.joint_unknown_resources += 1;
            }
            Verdict::RequiresAction | Verdict::Unknown(_) => round.statistics.joint_unresolved += 1,
            _ => {}
        }
        round.admission_gate = Some(FrozenAdmissionGate {
            members,
            verdict,
            created_at: now,
            apply_before,
        });
        let waiting = self.waiting_queue.read();
        self.revalidate_joint_admission_gate(&waiting, round);
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
                .all(|request| gate.members.iter().any(|member| member.matches(request)));
        let mut previous = None;
        for request in waiting.iter() {
            let position = gate.members.iter().position(|member| {
                member.phase == RequestPhase::Waiting && member.matches(request)
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
                        member.phase == RequestPhase::Waiting && member.matches(request)
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
