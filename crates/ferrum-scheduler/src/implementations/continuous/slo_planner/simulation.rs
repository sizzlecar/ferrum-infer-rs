use super::{
    execution::{self, PlanningExecutionContext, PlanningExecutionState},
    obligations::PlanningObligationSet,
    types::*,
};
use std::{ops::Deref, sync::Arc};

#[derive(Clone)]
pub(super) struct PlanningState<'epoch> {
    logical: SimulatedSequence,
    execution: Arc<dyn PlanningExecutionState<'epoch> + 'epoch>,
    depth: usize,
    pub first_wave_canonical:
        Option<Arc<ferrum_interfaces::execution_cost::CanonicalWaveCostShape>>,
}

impl Deref for PlanningState<'_> {
    type Target = SimulatedSequence;
    fn deref(&self) -> &Self::Target {
        &self.logical
    }
}

pub(super) struct VerifiedTransition<'epoch> {
    pub wave: WaveCandidate,
    pub state: PlanningState<'epoch>,
}

pub(super) struct TransitionFailure {
    pub cause: SimulationFailure,
    /// Distinguishes an attempted callback from accepted execution evidence.
    pub projected: bool,
}

#[derive(Clone)]
pub(super) struct SimulatedSequence {
    pub requests: Vec<RequestSchedulingView>,
    pub now_ns: u64,
    pub started_at_ns: u64,
    pub output_tokens: u64,
    pub net_prefill_work_ns: u64,
    pub remaining_kv_tokens: u64,
    pub remaining_output_bytes: u64,
    pub first_wave_cost_ns: u64,
    pub first_fairness_rank: u64,
    pub minimum_start_slack_ns: u64,
    pub minimum_cost_freshness_slack_ns: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum SimulationFailure {
    Unknown(PlanningUnknownReason),
    /// A particular sequence failed. This never proves global impossibility.
    SequenceViolation,
}

impl From<PlanningUnknownReason> for SimulationFailure {
    fn from(reason: PlanningUnknownReason) -> Self {
        Self::Unknown(reason)
    }
}

pub(super) fn begin<'epoch>(
    snapshot: &'epoch SchedulerSnapshot,
    context: &'epoch dyn PlanningExecutionContext,
    poll: &mut dyn FnMut() -> Result<(), PlanningUnknownReason>,
    started_at_ns: u64,
) -> Result<PlanningState<'epoch>, SimulationFailure> {
    let execution = execution::checked(poll, |poll| context.begin(snapshot, poll))?;
    Ok(PlanningState {
        execution,
        depth: 0,
        first_wave_canonical: None,
        logical: SimulatedSequence {
            requests: snapshot.requests.clone(),
            now_ns: started_at_ns,
            started_at_ns,
            output_tokens: 0,
            net_prefill_work_ns: 0,
            remaining_kv_tokens: snapshot.capacity.available_kv_tokens,
            remaining_output_bytes: snapshot.capacity.available_output_bytes,
            first_wave_cost_ns: 0,
            first_fairness_rank: u64::MAX,
            minimum_start_slack_ns: u64::MAX,
            minimum_cost_freshness_slack_ns: u64::MAX,
        },
    })
}

/// Expand only this edge. The parent's logical and execution states are never
/// changed, including when a late cost/output/obligation check fails.
pub(super) fn advance<'epoch>(
    snapshot: &SchedulerSnapshot,
    parent: &PlanningState<'epoch>,
    work: &[CandidateWork],
    model: &dyn PlanningCostModel,
    complete_resources: bool,
    poll: &mut dyn FnMut() -> Result<(), PlanningUnknownReason>,
    milestones_enabled: bool,
    protection: Option<&PlanningObligationSet>,
) -> Result<VerifiedTransition<'epoch>, TransitionFailure> {
    let projected = execution::project(
        snapshot,
        &parent.requests,
        work,
        parent.execution.as_ref(),
        parent.depth == 0,
        model.requires_statistical_evidence(),
        poll,
    )
    .map_err(|reason| TransitionFailure {
        cause: reason.into(),
        projected: false,
    })?
    .ok_or(TransitionFailure {
        cause: SimulationFailure::SequenceViolation,
        projected: false,
    })?;
    let mut logical = parent.logical.clone();
    apply(
        snapshot,
        &mut logical,
        &projected.wave,
        model,
        complete_resources,
        poll,
        milestones_enabled,
        protection,
        parent.depth == 0,
    )
    .map_err(|cause| TransitionFailure {
        cause,
        projected: true,
    })?;
    Ok(VerifiedTransition {
        wave: projected.wave,
        state: PlanningState {
            logical,
            execution: projected.successor,
            depth: parent.depth + 1,
            first_wave_canonical: if parent.depth == 0 {
                projected.first_canonical
            } else {
                parent.first_wave_canonical.clone()
            },
        },
    })
}

/// Independent replay initializes a fresh backend state and resolves every
/// selected edge again. Search state is not a publication certificate.
pub(super) fn replay<'epoch>(
    snapshot: &'epoch SchedulerSnapshot,
    waves: &[WaveCandidate],
    model: &dyn PlanningCostModel,
    context: &'epoch dyn PlanningExecutionContext,
    complete_resources: bool,
    poll: &mut dyn FnMut() -> Result<(), PlanningUnknownReason>,
    started_at_ns: u64,
    milestones_enabled: bool,
    protection: Option<&PlanningObligationSet>,
) -> Result<PlanningState<'epoch>, SimulationFailure> {
    let mut state = begin(snapshot, context, poll, started_at_ns)?;
    for wave in waves {
        let transition = advance(
            snapshot,
            &state,
            &wave.work,
            model,
            complete_resources,
            poll,
            milestones_enabled,
            protection,
        )
        .map_err(|failure| failure.cause)?;
        if transition.wave != *wave {
            return Err(SimulationFailure::SequenceViolation);
        }
        state = transition.state;
    }
    Ok(state)
}

#[cfg(test)]
pub(super) fn simulate(
    snapshot: &SchedulerSnapshot,
    waves: &[WaveCandidate],
    model: &dyn PlanningCostModel,
    resolver: &dyn PlanningShapeResolver,
    resources: Option<&dyn PlanningResourceResolver>,
    poll_budget: &mut dyn FnMut() -> Result<(), PlanningUnknownReason>,
    started_at_ns: u64,
    milestones_enabled: bool,
    protection: Option<&PlanningObligationSet>,
) -> Result<SimulatedSequence, SimulationFailure> {
    let context = execution::ReplayContext {
        resolver,
        resources,
    };
    replay(
        snapshot,
        waves,
        model,
        &context,
        resources.is_some(),
        poll_budget,
        started_at_ns,
        milestones_enabled,
        protection,
    )
    .map(|state| state.logical)
}

fn apply(
    snapshot: &SchedulerSnapshot,
    state: &mut SimulatedSequence,
    wave: &WaveCandidate,
    model: &dyn PlanningCostModel,
    complete_resources: bool,
    poll_budget: &mut dyn FnMut() -> Result<(), PlanningUnknownReason>,
    milestones_enabled: bool,
    protection: Option<&PlanningObligationSet>,
    first_wave: bool,
) -> Result<(), SimulationFailure> {
    if wave.based_on_generation != snapshot.generation
        || wave.cost_model_version != snapshot.cost_model_version
    {
        return Err(PlanningUnknownReason::InvalidSnapshot.into());
    }
    if let Some(index) = protection.and_then(|scope| scope.required_service(&state.requests)) {
        if !wave
            .work
            .iter()
            .any(|work| work.key == state.requests[index].key)
        {
            return Err(SimulationFailure::SequenceViolation);
        }
    }
    let shape = &wave.execution_shape;
    if !complete_resources
        && snapshot.capabilities.workspace_bytes_upper_bound
            > snapshot.capacity.available_workspace_bytes
    {
        return Err(PlanningUnknownReason::OutputOrResourceBlocked.into());
    }
    let mut advances = Vec::with_capacity(wave.work.len());
    let mut required_kv = 0_u64;
    let mut required_output = 0_u64;
    for work in &wave.work {
        let index = state
            .requests
            .iter()
            .position(|request| request.key == work.key)
            .ok_or(SimulationFailure::SequenceViolation)?;
        let request = &state.requests[index];
        let (new_context, emits_token) = match (&work.action, &request.phase) {
            (WaveAction::Decode, RequestPhaseView::Decode) => (
                request
                    .context_tokens
                    .checked_add(1)
                    .ok_or(PlanningUnknownReason::ArithmeticOverflow)?,
                true,
            ),
            (WaveAction::Prefill { offset, count }, RequestPhaseView::Prefill(progress)) => {
                let end = offset
                    .checked_add(count.get())
                    .ok_or(PlanningUnknownReason::ArithmeticOverflow)?;
                (
                    request.context_tokens.max(end),
                    // Final PlanRuntime prefill samples one token even when
                    // rebuilding an existing generation after preemption.
                    end == progress.total_prompt_tokens.get(),
                )
            }
            _ => return Err(SimulationFailure::SequenceViolation),
        };
        if new_context > snapshot.capacity.maximum_context_tokens.get() {
            return Err(PlanningUnknownReason::OutputOrResourceBlocked.into());
        }
        required_kv = required_kv
            .checked_add(u64::from(new_context - request.context_tokens))
            .ok_or(PlanningUnknownReason::ArithmeticOverflow)?;
        let (output_credit, additional_output_bytes) = if emits_token {
            request.output_credit.after_token()?
        } else {
            (request.output_credit, 0)
        };
        required_output = required_output
            .checked_add(additional_output_bytes)
            .ok_or(PlanningUnknownReason::ArithmeticOverflow)?;
        advances.push((index, new_context, emits_token, output_credit));
    }
    if (!complete_resources && required_kv > state.remaining_kv_tokens)
        || required_output > state.remaining_output_bytes
    {
        return Err(PlanningUnknownReason::OutputOrResourceBlocked.into());
    }
    if first_wave && shape.exact().is_none() {
        return Err(PlanningUnknownReason::InvalidShapeEvidence.into());
    }
    let cost = domain_cost(
        snapshot,
        model,
        shape,
        wave.cost_evidence.as_ref(),
        state.now_ns,
        poll_budget,
    )?;
    state.minimum_cost_freshness_slack_ns =
        state.minimum_cost_freshness_slack_ns.min(cost.valid_for_ns);
    let end_ns = state
        .now_ns
        .checked_add(cost.planning_ns)
        .ok_or(PlanningUnknownReason::ArithmeticOverflow)?;
    for (index, request) in state.requests.iter().enumerate() {
        if request.timing.completed() || protection.is_some_and(|scope| !scope.protects(index)) {
            continue;
        }
        let deadline = request
            .timing
            .next_deadline_ns()
            .ok_or(PlanningUnknownReason::ArithmeticOverflow)?;
        let emits = advances
            .iter()
            .any(|&(selected, _, emits, _)| selected == index && emits);
        if end_ns > deadline || (end_ns == deadline && !emits) {
            return Err(SimulationFailure::SequenceViolation);
        }
        state.minimum_start_slack_ns = state
            .minimum_start_slack_ns
            .min(deadline - end_ns - u64::from(!emits));
        if milestones_enabled {
            check_milestones(snapshot, index, request, end_ns, false, protection)?;
        }
    }
    if !complete_resources {
        state.remaining_kv_tokens -= required_kv;
    }
    state.remaining_output_bytes -= required_output;
    if let Some(scope) = protection {
        for (index, request) in state.requests.iter_mut().enumerate() {
            if scope.recovery_owner(index)
                && request.readiness == RequestReadiness::Ready
                && !request.timing.completed()
                && !wave.work.iter().any(|work| work.key == request.key)
            {
                request.recovery_service.bypass();
            }
        }
    }
    for (work, (index, new_context, emits, output_credit)) in wave.work.iter().zip(advances) {
        let request = &mut state.requests[index];
        request.recovery_service.progressed();
        if first_wave {
            state.first_fairness_rank = state.first_fairness_rank.min(request.fairness_rank);
        }
        if let (WaveAction::Prefill { offset, count }, RequestPhaseView::Prefill(progress)) =
            (&work.action, &mut request.phase)
        {
            let end = offset
                .checked_add(count.get())
                .ok_or(PlanningUnknownReason::ArithmeticOverflow)?;
            let previous_work = progress
                .reference
                .work_at(progress.logical_high_water)
                .ok_or(PlanningUnknownReason::MissingReferenceWork)?;
            let new_high_water = progress.logical_high_water.max(end);
            let new_work = progress
                .reference
                .work_at(new_high_water)
                .ok_or(PlanningUnknownReason::MissingReferenceWork)?;
            state.net_prefill_work_ns = state
                .net_prefill_work_ns
                .checked_add(new_work - previous_work)
                .ok_or(PlanningUnknownReason::ArithmeticOverflow)?;
            progress.logical_high_water = new_high_water;
            progress.offset = end;
            if end == progress.total_prompt_tokens.get() {
                request.phase = RequestPhaseView::Decode;
            }
        }
        request.context_tokens = new_context;
        if emits {
            if request.timing.committed_tokens == 0 {
                request.timing.first_commit_at_ns = Some(end_ns);
            }
            request.timing.last_commit_at_ns = Some(end_ns);
            request.timing.committed_tokens = request
                .timing
                .committed_tokens
                .checked_add(1)
                .ok_or(PlanningUnknownReason::ArithmeticOverflow)?;
            request.output_credit = output_credit;
            state.output_tokens = state
                .output_tokens
                .checked_add(1)
                .ok_or(PlanningUnknownReason::ArithmeticOverflow)?;
        }
    }
    if milestones_enabled {
        for (index, request) in state.requests.iter().enumerate() {
            check_milestones(snapshot, index, request, end_ns, true, protection)?;
            if let RequestPhaseView::Prefill(progress) = &snapshot.requests[index].phase {
                let completed = logical_work(&snapshot.requests[index], request)?;
                if snapshot.requests[index].timing.committed_tokens == 0 {
                    for (_, milestone) in
                        progress
                            .milestones
                            .iter()
                            .enumerate()
                            .filter(|(m, milestone)| {
                                protection.is_none_or(|scope| scope.requires_milestone(index, *m))
                                    && milestone.at_ns >= end_ns
                                    && milestone.required_reference_work_ns <= completed
                            })
                    {
                        state.minimum_start_slack_ns =
                            state.minimum_start_slack_ns.min(milestone.at_ns - end_ns);
                    }
                }
            }
        }
    }
    state.now_ns = end_ns;
    if first_wave {
        state.first_wave_cost_ns = cost.planning_ns;
    }
    Ok(())
}

/// Every alternative is a complete physical wave with the same logical work.
/// The maximum is an empirical planning envelope, not a hard timing guarantee.
/// All branches must remain covered through its end; final replay additionally
/// subtracts the real decision overhead from this inclusive residual TTL.
pub(super) fn domain_cost(
    snapshot: &SchedulerSnapshot,
    model: &dyn PlanningCostModel,
    domain: &PlanningShapeDomain<super::super::cost_model::WaveExecutionShape>,
    evidence: Option<&PlanningShapeDomain<PlanningCostEvidence>>,
    now_ns: u64,
    poll: &mut dyn FnMut() -> Result<(), PlanningUnknownReason>,
) -> Result<PlanningCost, PlanningUnknownReason> {
    let alternatives = matches!(domain, PlanningShapeDomain::HostContentAlternatives(_));
    if alternatives && !model.supports_empirical_host_content() {
        return Err(PlanningUnknownReason::CostUnavailable);
    }
    if domain.shapes().is_empty() || domain.shapes().len() > 256 {
        return Err(PlanningUnknownReason::ShapeCapacity);
    }
    let mut result = PlanningCost {
        typical_ns: 0,
        planning_ns: 0,
        model_version: snapshot.cost_model_version,
        valid_for_ns: u64::MAX,
    };
    let evidence = evidence.filter(|evidence| {
        evidence.shapes().len() == domain.shapes().len()
            && matches!(
                (domain, *evidence),
                (PlanningShapeDomain::Exact(_), PlanningShapeDomain::Exact(_))
                    | (
                        PlanningShapeDomain::HostContentAlternatives(_),
                        PlanningShapeDomain::HostContentAlternatives(_)
                    )
            )
    });
    for (index, shape) in domain.shapes().iter().enumerate() {
        poll()?;
        let cost = model
            .predict_with_evidence(
                &snapshot.fingerprint,
                shape,
                evidence.map(|domain| &domain.shapes()[index]),
                now_ns,
            )
            .ok_or(PlanningUnknownReason::CostUnavailable)?;
        poll()?;
        if cost.model_version != snapshot.cost_model_version {
            return Err(PlanningUnknownReason::ModelVersionMismatch);
        }
        if cost.typical_ns == 0 || cost.planning_ns < cost.typical_ns {
            return Err(PlanningUnknownReason::CostUnavailable);
        }
        result.typical_ns = result.typical_ns.max(cost.typical_ns);
        result.planning_ns = result.planning_ns.max(cost.planning_ns);
        result.valid_for_ns = result.valid_for_ns.min(cost.valid_for_ns);
    }
    if model.supports_empirical_host_content() {
        result.valid_for_ns = result
            .valid_for_ns
            .checked_sub(result.planning_ns)
            .ok_or(PlanningUnknownReason::CostUnavailable)?;
    }
    Ok(result)
}

fn logical_work(
    initial: &RequestSchedulingView,
    current: &RequestSchedulingView,
) -> Result<u64, PlanningUnknownReason> {
    let RequestPhaseView::Prefill(initial_progress) = &initial.phase else {
        return Ok(0);
    };
    let high_water = match &current.phase {
        RequestPhaseView::Prefill(progress) => progress.logical_high_water,
        RequestPhaseView::Decode => initial_progress.total_prompt_tokens.get(),
    };
    initial_progress
        .reference
        .work_at(high_water)
        .ok_or(PlanningUnknownReason::MissingReferenceWork)
}

fn check_milestones(
    snapshot: &SchedulerSnapshot,
    index: usize,
    current: &RequestSchedulingView,
    end_ns: u64,
    inclusive: bool,
    protection: Option<&PlanningObligationSet>,
) -> Result<(), SimulationFailure> {
    let initial = &snapshot.requests[index];
    let RequestPhaseView::Prefill(progress) = &initial.phase else {
        return Ok(());
    };
    if initial.timing.committed_tokens > 0 {
        return Ok(());
    }
    let completed = logical_work(initial, current)?;
    if progress
        .milestones
        .iter()
        .enumerate()
        .any(|(m, milestone)| {
            protection.is_none_or(|scope| scope.requires_milestone(index, m))
                && (milestone.at_ns < end_ns || (inclusive && milestone.at_ns == end_ns))
                && completed < milestone.required_reference_work_ns
        })
    {
        return Err(SimulationFailure::SequenceViolation);
    }
    Ok(())
}

pub(super) fn ready_witness(
    snapshot: &SchedulerSnapshot,
    state: &SimulatedSequence,
    milestones_enabled: bool,
    protection: Option<&PlanningObligationSet>,
) -> Result<bool, PlanningUnknownReason> {
    for (index, (initial, current)) in snapshot.requests.iter().zip(&state.requests).enumerate() {
        if protection.is_some_and(|scope| !scope.protects(index)) {
            // A ready late owner still needs actual physical service in this
            // common witness. Blocked owners remain resident, not future credit.
            if initial.readiness == RequestReadiness::Ready
                && !initial.timing.completed()
                && current.timing.committed_tokens == initial.timing.committed_tokens
                && current.phase == initial.phase
            {
                return Ok(false);
            }
            continue;
        }
        // Every initial decoder requires actual service in the SAME sequence.
        if matches!(initial.phase, RequestPhaseView::Decode)
            && !initial.timing.completed()
            && current.timing.committed_tokens <= initial.timing.committed_tokens
        {
            return Ok(false);
        }
        if !current.timing.completed()
            && current
                .timing
                .next_deadline_ns()
                .ok_or(PlanningUnknownReason::ArithmeticOverflow)?
                <= snapshot.scope.horizon_end_ns
        {
            return Ok(false);
        }
        if milestones_enabled {
            if let RequestPhaseView::Prefill(progress) = &initial.phase {
                if initial.timing.committed_tokens == 0 {
                    let completed = logical_work(initial, current)?;
                    if progress
                        .milestones
                        .iter()
                        .enumerate()
                        .any(|(m, milestone)| {
                            protection.is_none_or(|scope| scope.requires_milestone(index, m))
                                && milestone.at_ns <= snapshot.scope.horizon_end_ns
                                && completed < milestone.required_reference_work_ns
                        })
                    {
                        return Ok(false);
                    }
                }
            }
        }
    }
    Ok(true)
}

pub(super) fn score_at(
    snapshot: &SchedulerSnapshot,
    state: &SimulatedSequence,
    settings: &BoundedPlannerSettings,
    transaction_started_at_ns: u64,
    ranking_now_ns: u64,
    poll: &mut dyn FnMut() -> Result<(), PlanningUnknownReason>,
) -> Result<(f64, u64), PlanningUnknownReason> {
    // Logical simulation remains anchored where it was constructed/replayed.
    // Ranking projects only its duration onto one common actual CPU instant.
    if ranking_now_ns < state.started_at_ns {
        return Err(PlanningUnknownReason::ClockMovedBackwards);
    }
    let execution_ns = state
        .now_ns
        .checked_sub(state.started_at_ns)
        .ok_or(PlanningUnknownReason::ClockMovedBackwards)?;
    let cpu_ns = ranking_now_ns
        .checked_sub(transaction_started_at_ns)
        .ok_or(PlanningUnknownReason::ClockMovedBackwards)?;
    let occupied_ns = cpu_ns
        .checked_add(execution_ns)
        .filter(|&value| value > 0)
        .ok_or(PlanningUnknownReason::ArithmeticOverflow)?;
    let completion_ns = ranking_now_ns
        .checked_add(execution_ns)
        .ok_or(PlanningUnknownReason::ArithmeticOverflow)?;
    let mut debt = 0_u64;
    for (initial, current) in snapshot.requests.iter().zip(&state.requests) {
        poll()?;
        let RequestPhaseView::Prefill(progress) = &initial.phase else {
            continue;
        };
        if initial.timing.committed_tokens > 0 {
            continue;
        }
        let completed = logical_work(initial, current)?;
        let first_deadline = initial
            .timing
            .ingress_at_ns
            .checked_add(initial.timing.budgets.ttft_ns.get())
            .ok_or(PlanningUnknownReason::ArithmeticOverflow)?;
        let required = progress
            .ideal_reference_work_at(completion_ns, first_deadline)
            .ok_or(PlanningUnknownReason::ArithmeticOverflow)?;
        debt = debt
            .checked_add(required.saturating_sub(completed))
            .ok_or(PlanningUnknownReason::ArithmeticOverflow)?;
    }
    let tau = snapshot.scope.reference_decode_token_ns.get() as f64;
    let value = (state.output_tokens as f64
        + settings.search.prefill_credit_beta * (state.net_prefill_work_ns as f64 / tau)
        - settings.search.prefill_debt_gamma * (debt as f64 / tau))
        / occupied_ns as f64;
    if !value.is_finite() {
        return Err(PlanningUnknownReason::ArithmeticOverflow);
    }
    Ok((value, debt))
}
