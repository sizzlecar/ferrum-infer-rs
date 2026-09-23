//! Pure time-admission assessment. The caller supplies every live obligation
//! plus the proposed request; no queue, KV, output credit or execution authority
//! is changed here. A finite forecast is explicitly distinct from continuing
//! service without a time promise. Ingress storage limits belong to the queue
//! owner and cannot be enforced by a read-only projection.

use super::*;
use ferrum_types::{SloAdmissionConfig, SloOutputLengthPolicy, SloTimeAdmissionPolicy};

/// Once accepted, even a strict time policy cannot retroactively reject work.
/// The actual transport/lifecycle owner supplies this boundary, not metadata.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TimeAdmissionBoundary {
    BeforeAcceptance,
    Accepted,
}

pub struct TimeAdmissionQuery<'a> {
    pub snapshot: &'a SchedulerSnapshot,
    pub target: &'a RequestWorkKey,
    /// Exact keys of the currently active set, all present in the snapshot.
    /// Other waiting peers retain their timing obligations in the same view.
    pub active: &'a [RequestWorkKey],
    pub boundary: TimeAdmissionBoundary,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TimeAdmissionUnknown {
    Planning(PlanningUnknownReason),
    OutsideSequenceCoverage,
    LengthStatisticsUnavailable,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TimeAdmissionContinuation {
    /// Use the independently checked, bounded and fair execution path, with
    /// fresh physical permissions. This value itself grants no permission.
    BestEffort,
    /// Do not guess missing identity/resource evidence. Register real change
    /// notifications before waiting and preserve cancellation responsiveness.
    WaitForEvidence,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TimeAdmissionDeferReason {
    ActiveLimit,
    ExistingObligationAtRisk,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TimeAdmissionRejectReason {
    /// Only a fact or independently certified lower bound for this target;
    /// beam exhaustion and another request's violation do not prove this.
    TargetTimeImpossible(PlanningImpossibleReason),
    /// An explicit strict acceptance policy expired, not an impossibility proof.
    StrictWaitExpired { expiry_at_ns: u64 },
}

/// Register request/capacity/output/model changes using the real owners, and
/// this optional time wake. Generation/version identify what was evaluated;
/// they are not synthetic capacity epochs or a substitute for race-free waits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TimeAdmissionWait {
    pub review_at_ns: Option<u64>,
    pub strict_expiry_at_ns: Option<u64>,
    pub snapshot_generation: u64,
    pub cost_model_version: u64,
}

#[derive(Debug, Clone, PartialEq)]
pub enum TimeAdmissionDecision {
    /// Includes the new decoder together with every peer, but only through
    /// witness.validated_through_ns. Revalidate before any physical admission.
    Admit {
        first_wave: SelectedWave,
        witness: PlanningWitnessSummary,
        search: PlanningSearchStats,
    },
    Defer {
        reason: TimeAdmissionDeferReason,
        wait: TimeAdmissionWait,
    },
    Reject {
        reason: TimeAdmissionRejectReason,
    },
    Unknown {
        reason: TimeAdmissionUnknown,
        continuation: TimeAdmissionContinuation,
        wait: TimeAdmissionWait,
    },
    /// The SLO is already unattainable; retain that fact and continue work.
    /// This is neither a successful time admission nor a zero-cost forecast.
    BestEffort {
        reason: PlanningImpossibleReason,
    },
}

pub struct TimeAdmissionEvaluator<'a> {
    pub policy: &'a SloAdmissionConfig,
    pub planner: &'a BoundedSloPlanner,
    pub model: &'a dyn PlanningCostModel,
    pub shapes: &'a dyn PlanningShapeResolver,
    pub resources: &'a dyn PlanningResourceResolver,
}

/// Records *every* read, including reads made inside the planner. Comparing
/// only the wrapper's first and last reads misses a reversal after the search.
struct TrackingClock<'a> {
    inner: &'a mut dyn PlanningClock,
    last_ns: u64,
    moved_backwards: bool,
}

impl PlanningClock for TrackingClock<'_> {
    fn now_ns(&mut self) -> u64 {
        let now = self.inner.now_ns();
        self.moved_backwards |= now < self.last_ns;
        self.last_ns = now;
        now
    }

    fn planning_budget_window(&self) -> Option<PlanningBudgetWindow> {
        self.inner.planning_budget_window()
    }
}

impl TimeAdmissionEvaluator<'_> {
    pub fn assess(
        &self,
        query: TimeAdmissionQuery<'_>,
        clock: &mut dyn PlanningClock,
    ) -> TimeAdmissionDecision {
        let mut clock = TrackingClock {
            inner: clock,
            last_ns: query.snapshot.observed_at_ns,
            moved_backwards: false,
        };
        let initial_now = clock.now_ns();
        let strict = self.policy.time_policy == SloTimeAdmissionPolicy::RequireSlo
            && query.boundary == TimeAdmissionBoundary::BeforeAcceptance;
        let mut expiry = None;
        let decision = self.assess_inner(&query, &mut clock, initial_now, strict, &mut expiry);
        // No inner early return can bypass this read, clock validation, wait
        // refresh or the final inclusive witness/strict-expiry intersection.
        let final_now = clock.now_ns();
        self.finalize(
            query.snapshot,
            decision,
            strict,
            expiry,
            final_now,
            clock.moved_backwards,
        )
    }

    fn assess_inner(
        &self,
        query: &TimeAdmissionQuery<'_>,
        clock: &mut dyn PlanningClock,
        now: u64,
        strict: bool,
        expiry: &mut Option<u64>,
    ) -> TimeAdmissionDecision {
        let snapshot = query.snapshot;
        let wait = empty_wait(snapshot);
        if self.policy.validate().is_err() {
            return invalid(PlanningUnknownReason::InvalidConfiguration, wait);
        }
        if now < snapshot.observed_at_ns {
            return invalid(PlanningUnknownReason::ClockMovedBackwards, wait);
        }
        // This bounded identity/timing check does not require a calibrated
        // forecast, reference-work curve or peer readiness. Otherwise those
        // very evidence gaps would erase a strict request's expiry wake.
        *expiry = match self.pending_expiry(query) {
            Ok(expiry) => Some(expiry),
            Err(reason) => return invalid(reason, wait),
        };
        let fallback = continuation(strict);
        if let Err(reason) = super::validation::validate(&self.planner.settings, snapshot, true) {
            return TimeAdmissionDecision::Unknown {
                reason: TimeAdmissionUnknown::Planning(reason),
                continuation: if is_forecast_gap(reason) {
                    fallback
                } else {
                    TimeAdmissionContinuation::WaitForEvidence
                },
                wait,
            };
        }
        if query.active.len() >= self.policy.max_active_requests.get() {
            return TimeAdmissionDecision::Defer {
                reason: TimeAdmissionDeferReason::ActiveLimit,
                wait,
            };
        }
        if self.policy.output_length_policy == SloOutputLengthPolicy::StatisticalCapacity {
            // No independently calibrated length distribution exists in the
            // current profile schema. Never substitute a benchmark reference.
            return TimeAdmissionDecision::Unknown {
                reason: TimeAdmissionUnknown::LengthStatisticsUnavailable,
                continuation: fallback,
                wait,
            };
        }
        for request in &snapshot.requests {
            if request.timing.completed() {
                continue;
            }
            let remaining =
                request.timing.maximum_output_tokens.get() - request.timing.committed_tokens;
            let last_context = match &request.phase {
                RequestPhaseView::Decode => request.context_tokens.checked_add(remaining),
                RequestPhaseView::Prefill(progress) => progress
                    .total_prompt_tokens
                    .get()
                    .checked_add(remaining - 1),
            };
            if last_context.is_none_or(|context| {
                u64::from(context) > self.policy.max_sequence_tokens.get() as u64
            }) {
                return TimeAdmissionDecision::Unknown {
                    reason: TimeAdmissionUnknown::OutsideSequenceCoverage,
                    continuation: fallback,
                    wait,
                };
            }
        }
        match self.planner.propose_admission_with_resources(
            snapshot,
            query.target,
            self.model,
            self.shapes,
            self.resources,
            clock,
        ) {
            PlanningDecision::FeasibleWithinHorizon {
                first_wave,
                witness,
                search,
            } => TimeAdmissionDecision::Admit {
                first_wave,
                witness,
                search,
            },
            // A scoped recovery proof never grants new time promises.
            PlanningDecision::ProtectedWithinHorizon { .. } => TimeAdmissionDecision::Defer {
                reason: TimeAdmissionDeferReason::ExistingObligationAtRisk,
                wait,
            },
            PlanningDecision::Unknown { reason, .. } => TimeAdmissionDecision::Unknown {
                reason: TimeAdmissionUnknown::Planning(reason),
                continuation: if is_forecast_gap(reason) {
                    fallback
                } else {
                    TimeAdmissionContinuation::WaitForEvidence
                },
                wait,
            },
            PlanningDecision::ProvenImpossibleUnderModel { reason, .. } if !strict => {
                TimeAdmissionDecision::BestEffort { reason }
            }
            PlanningDecision::ProvenImpossibleUnderModel { reason, .. } => {
                let key = match &reason {
                    PlanningImpossibleReason::HistoricalViolation { key }
                    | PlanningImpossibleReason::DeadlineAlreadyMissed { key, .. }
                    | PlanningImpossibleReason::CertifiedOptimisticLowerBound { key, .. } => key,
                };
                if key == query.target {
                    TimeAdmissionDecision::Reject {
                        reason: TimeAdmissionRejectReason::TargetTimeImpossible(reason),
                    }
                } else {
                    TimeAdmissionDecision::Defer {
                        reason: TimeAdmissionDeferReason::ExistingObligationAtRisk,
                        wait,
                    }
                }
            }
        }
    }

    fn pending_expiry(&self, query: &TimeAdmissionQuery<'_>) -> Result<u64, PlanningUnknownReason> {
        let snapshot = query.snapshot;
        // Same request bound as the planner. Check before the duplicate and
        // membership scans so arbitrary caller slices cannot add unbounded work.
        if snapshot.requests.len() > 256 || query.active.len() > snapshot.requests.len() {
            return Err(PlanningUnknownReason::InvalidSnapshot);
        }
        if snapshot
            .requests
            .iter()
            .enumerate()
            .any(|(index, request)| {
                snapshot.requests[..index]
                    .iter()
                    .any(|previous| previous.key.request_id == request.key.request_id)
            })
        {
            return Err(PlanningUnknownReason::InvalidSnapshot);
        }
        let target = snapshot
            .requests
            .iter()
            .find(|request| &request.key == query.target)
            .ok_or(PlanningUnknownReason::InvalidSnapshot)?;
        let RequestPhaseView::Prefill(progress) = &target.phase else {
            return Err(PlanningUnknownReason::InvalidSnapshot);
        };
        if target.timing.committed_tokens != 0
            || target.timing.first_commit_at_ns.is_some()
            || target.timing.last_commit_at_ns.is_some()
            || target.timing.ingress_at_ns > snapshot.observed_at_ns
            || target.context_tokens > snapshot.capacity.maximum_context_tokens.get()
            || progress.offset >= progress.total_prompt_tokens.get()
            || progress.logical_high_water < progress.offset
            || progress.logical_high_water > progress.total_prompt_tokens.get()
            || progress.executable_until < progress.offset
            || progress.executable_until > progress.total_prompt_tokens.get()
            || progress.total_prompt_tokens.get() > snapshot.capacity.maximum_context_tokens.get()
            || progress.admitted_at_ns < target.timing.ingress_at_ns
            || progress.admitted_at_ns > snapshot.observed_at_ns
            || query.active.iter().enumerate().any(|(index, key)| {
                key == query.target
                    || query.active[..index].contains(key)
                    || !snapshot
                        .requests
                        .iter()
                        .any(|request| &request.key == key && !request.timing.completed())
            })
        {
            return Err(PlanningUnknownReason::InvalidSnapshot);
        }
        target
            .timing
            .next_deadline_ns()
            .ok_or(PlanningUnknownReason::ArithmeticOverflow)?;
        self.policy
            .max_wait_ms
            .get()
            .checked_mul(1_000_000)
            .and_then(|duration| target.timing.ingress_at_ns.checked_add(duration))
            .ok_or(PlanningUnknownReason::ArithmeticOverflow)
    }

    fn finalize(
        &self,
        snapshot: &SchedulerSnapshot,
        mut decision: TimeAdmissionDecision,
        strict: bool,
        expiry: Option<u64>,
        now: u64,
        clock_moved_backwards: bool,
    ) -> TimeAdmissionDecision {
        if clock_moved_backwards {
            // No deadline/rejection can be inferred from a contradictory clock.
            return invalid(
                PlanningUnknownReason::ClockMovedBackwards,
                empty_wait(snapshot),
            );
        }
        let wait = TimeAdmissionWait {
            review_at_ns: expiry.filter(|expiry| *expiry > now),
            strict_expiry_at_ns: expiry.filter(|_| strict),
            ..empty_wait(snapshot)
        };
        match &mut decision {
            TimeAdmissionDecision::Defer { wait: current, .. }
            | TimeAdmissionDecision::Unknown { wait: current, .. } => *current = wait,
            _ => {}
        }
        // The pending target's identity and original ingress were validated
        // independently. Peer/forecast defects cannot erase its explicit wait
        // expiry; expiration proves only this acceptance policy timed out.
        // Invalid target evidence never sets expiry, and clock contradictions
        // have already returned above without manufacturing a rejection.
        if let Some(expiry_at_ns) = expiry.filter(|expiry| strict && now >= *expiry) {
            return TimeAdmissionDecision::Reject {
                reason: TimeAdmissionRejectReason::StrictWaitExpired { expiry_at_ns },
            };
        }
        if let TimeAdmissionDecision::Unknown {
            reason: TimeAdmissionUnknown::Planning(reason),
            ..
        } = &decision
        {
            if matches!(
                reason,
                PlanningUnknownReason::InvalidConfiguration
                    | PlanningUnknownReason::InvalidSnapshot
                    | PlanningUnknownReason::InvalidShapeEvidence
                    | PlanningUnknownReason::ArithmeticOverflow
                    | PlanningUnknownReason::ClockMovedBackwards
            ) {
                // Forecast invalidity is neither target impossibility nor
                // permission to proceed best-effort. Keep a trusted target's
                // independent future review/expiry while awaiting correction.
                return invalid(*reason, wait);
            }
        }
        if let TimeAdmissionDecision::Admit { first_wave, .. } = &mut decision {
            let Some(allowance) = self
                .planner
                .settings
                .search
                .max_planning_us
                .get()
                .checked_mul(1000)
            else {
                return invalid(PlanningUnknownReason::ArithmeticOverflow, wait);
            };
            if now - snapshot.observed_at_ns >= allowance {
                return TimeAdmissionDecision::Unknown {
                    reason: TimeAdmissionUnknown::Planning(
                        PlanningUnknownReason::ComputeBudgetExhausted,
                    ),
                    continuation: continuation(strict),
                    wait,
                };
            }
            if now < first_wave.planning_observed_at_ns {
                return invalid(
                    PlanningUnknownReason::ClockMovedBackwards,
                    empty_wait(snapshot),
                );
            }
            if let Some(expiry) = expiry.filter(|_| strict) {
                // A delay is inclusive, but strict acceptance at expiry is not.
                // Preserve the planner's original anchor so every later owner
                // can check this same intersection without refreshing its TTL.
                let Some(strict_delay) = expiry
                    .checked_sub(1)
                    .and_then(|last| last.checked_sub(first_wave.planning_observed_at_ns))
                else {
                    return invalid(PlanningUnknownReason::ArithmeticOverflow, wait);
                };
                first_wave.witness_valid_for_ns = first_wave.witness_valid_for_ns.min(strict_delay);
            }
            let valid_until = first_wave
                .planning_observed_at_ns
                .checked_add(first_wave.witness_valid_for_ns);
            if valid_until.is_none_or(|until| now > until) {
                return TimeAdmissionDecision::Unknown {
                    reason: TimeAdmissionUnknown::Planning(
                        PlanningUnknownReason::HorizonInsufficient,
                    ),
                    continuation: continuation(strict),
                    wait,
                };
            }
        }
        decision
    }
}

fn empty_wait(snapshot: &SchedulerSnapshot) -> TimeAdmissionWait {
    TimeAdmissionWait {
        review_at_ns: None,
        strict_expiry_at_ns: None,
        snapshot_generation: snapshot.generation,
        cost_model_version: snapshot.cost_model_version,
    }
}

fn invalid(reason: PlanningUnknownReason, wait: TimeAdmissionWait) -> TimeAdmissionDecision {
    TimeAdmissionDecision::Unknown {
        reason: TimeAdmissionUnknown::Planning(reason),
        continuation: TimeAdmissionContinuation::WaitForEvidence,
        wait,
    }
}

fn continuation(strict: bool) -> TimeAdmissionContinuation {
    if strict {
        TimeAdmissionContinuation::WaitForEvidence
    } else {
        TimeAdmissionContinuation::BestEffort
    }
}

fn is_forecast_gap(reason: PlanningUnknownReason) -> bool {
    matches!(
        reason,
        PlanningUnknownReason::CostUnavailable
            | PlanningUnknownReason::ShapeUnavailable
            | PlanningUnknownReason::ComputeBudgetExhausted
            | PlanningUnknownReason::HorizonInsufficient
            | PlanningUnknownReason::SearchIncomplete
            | PlanningUnknownReason::MissingReferenceWork
            | PlanningUnknownReason::UnmodeledMaintenance
    )
}
