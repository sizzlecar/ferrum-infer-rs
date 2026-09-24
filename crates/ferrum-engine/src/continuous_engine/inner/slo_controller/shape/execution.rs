//! One immutable execution-domain successor per logical edge. The scheduler
//! owns logical progress; this adapter never advances a second copy of it.
use super::*;
use ferrum_scheduler::implementations::continuous::slo_planner::{
    PlanningExecutionContext, PlanningExecutionInput, PlanningExecutionState, ProjectedExecution,
};

type ProjectionResult<T> = std::result::Result<T, PlanningUnknownReason>;

struct ExecutorState<'epoch> {
    source: ExecutorShape<'epoch>,
    domain: RouteDomain,
    depth: usize,
}

impl PlanningExecutionContext for ExecutorShape<'_> {
    fn begin<'epoch>(
        &'epoch self,
        snapshot: &'epoch SchedulerSnapshot,
        poll: &mut dyn FnMut() -> ProjectionResult<()>,
    ) -> ProjectionResult<Arc<dyn PlanningExecutionState<'epoch> + 'epoch>> {
        poll()?;
        // This adapter belongs to one controller capture. Numerically equal
        // snapshots from another transaction do not exchange private states.
        if !std::ptr::eq(snapshot, &self.captured.snapshot) {
            return Err(PlanningUnknownReason::InvalidSnapshot);
        }
        // Reuse the captured owner/frontier validation. This temporary view is
        // dropped here; no logical frontier is retained in the execution state.
        self.initial_frontiers(poll)?;
        poll()?;
        Ok(Arc::new(ExecutorState {
            source: ExecutorShape {
                engine: self.engine,
                captured: self.captured,
            },
            domain: RouteDomain::initial(self.captured.route.initial_state()),
            depth: 0,
        }))
    }
}

impl<'epoch> PlanningExecutionState<'epoch> for ExecutorState<'epoch> {
    fn project(
        &self,
        input: &PlanningExecutionInput<'_>,
        poll: &mut dyn FnMut() -> ProjectionResult<()>,
    ) -> ProjectionResult<Option<ProjectedExecution<'epoch>>> {
        poll()?;
        if self.depth
            >= self
                .source
                .engine
                .config
                .scheduler
                .slo
                .planner
                .lookahead_waves
                .get()
        {
            return Err(PlanningUnknownReason::ShapeCapacity);
        }
        let snapshot = &self.source.captured.snapshot;
        let frontiers = input_frontiers(snapshot, input, self.depth, poll)?;
        let mut ordered_work = input.work.to_vec();
        // The existing authority-based ordering is reused, without invoking
        // the old stateless resolver or replaying a previous wave.
        self.source.order_work(snapshot, &mut ordered_work, poll)?;
        let mut rows = Vec::with_capacity(ordered_work.len());
        for work in &ordered_work {
            poll()?;
            let index = input
                .requests
                .iter()
                .position(|r| r.key == work.key)
                .ok_or(PlanningUnknownReason::InvalidShapeEvidence)?;
            let row = input
                .rows
                .iter()
                .find(|row| row.request.key == work.key)
                .ok_or(PlanningUnknownReason::InvalidShapeEvidence)?;
            rows.push(PreparedRow {
                index,
                work: row.work,
            });
        }
        let projected = self
            .source
            .project_domain(&self.domain, &frontiers, &rows, poll)?;
        poll()?;
        let Some((canonical_domain, statistical_evidence, domain)) = projected else {
            return Ok(None);
        };
        if (self.depth == 0 && canonical_domain.exact().is_none())
            || canonical_domain
                .shapes()
                .iter()
                .any(|shape| shape.kind != input.kind)
        {
            return Err(PlanningUnknownReason::InvalidShapeEvidence);
        }
        Ok(Some(ProjectedExecution {
            statistical_evidence,
            ordered_work,
            canonical_domain,
            successor: Arc::new(ExecutorState {
                source: ExecutorShape {
                    engine: self.source.engine,
                    captured: self.source.captured,
                },
                domain,
                depth: self
                    .depth
                    .checked_add(1)
                    .ok_or(PlanningUnknownReason::ArithmeticOverflow)?,
            }),
        }))
    }
}

/// Validate the whole supplied parent and its selected rows before invoking a
/// provider. The returned values are a read view of input.requests, not another
/// transition function: no after_work, advance, or prior-waves replay occurs.
pub(super) fn input_frontiers(
    snapshot: &SchedulerSnapshot,
    input: &PlanningExecutionInput<'_>,
    depth: usize,
    poll: &mut dyn FnMut() -> ProjectionResult<()>,
) -> ProjectionResult<Vec<ProjectedRequest>> {
    poll()?;
    if input.requests.len() != snapshot.requests.len()
        || input.requests.is_empty()
        || input.requests.len() > 256
        || input.work.is_empty()
        || input.work.len() != input.rows.len()
        || input.work.len() > snapshot.capabilities.max_wave_rows.get()
        || depth > 16
    {
        return Err(PlanningUnknownReason::InvalidShapeEvidence);
    }
    let mut frontiers = Vec::with_capacity(input.requests.len());
    for (current, initial) in input.requests.iter().zip(&snapshot.requests) {
        poll()?;
        if current.key != initial.key
            || current.timing.ingress_at_ns != initial.timing.ingress_at_ns
            || current.timing.budgets != initial.timing.budgets
            || current.timing.maximum_output_tokens != initial.timing.maximum_output_tokens
            || current.output_policy_signature != initial.output_policy_signature
            || current.recurrent_state_bytes != initial.recurrent_state_bytes
            || current.context_tokens > snapshot.capacity.maximum_context_tokens.get()
            || current.context_tokens < initial.context_tokens
            || current.timing.committed_tokens < initial.timing.committed_tokens
            || current.timing.committed_tokens > current.timing.maximum_output_tokens.get()
            || usize::try_from(current.timing.committed_tokens - initial.timing.committed_tokens)
                .map_or(true, |delta| delta > depth)
            || (initial.timing.first_commit_at_ns.is_some()
                && current.timing.first_commit_at_ns != initial.timing.first_commit_at_ns)
            || (depth == 0 && !same_request(current, initial))
        {
            return Err(PlanningUnknownReason::InvalidShapeEvidence);
        }
        match (&initial.phase, &current.phase) {
            (RequestPhaseView::Decode, RequestPhaseView::Decode) => {}
            (RequestPhaseView::Prefill(before), RequestPhaseView::Prefill(now)) => {
                if before.total_prompt_tokens != now.total_prompt_tokens
                    || before.admitted_at_ns != now.admitted_at_ns
                    || before.reference_work_at_admission_ns != now.reference_work_at_admission_ns
                    || before.executable_until != now.executable_until
                    || !Arc::ptr_eq(&before.reference, &now.reference)
                    || !Arc::ptr_eq(&before.milestones, &now.milestones)
                    || now.offset < before.offset
                    || now.offset >= now.total_prompt_tokens.get()
                    || now.logical_high_water < before.logical_high_water
                    || now.logical_high_water < now.offset
                    || now.logical_high_water > now.total_prompt_tokens.get()
                    || now.offset > now.executable_until
                {
                    return Err(PlanningUnknownReason::InvalidShapeEvidence);
                }
            }
            (RequestPhaseView::Prefill(before), RequestPhaseView::Decode)
                if current.context_tokens >= before.total_prompt_tokens.get()
                    && current.timing.committed_tokens > initial.timing.committed_tokens => {}
            _ => return Err(PlanningUnknownReason::InvalidShapeEvidence),
        }
        frontiers.push(ProjectedRequest {
            key: current.key.clone(),
            phase: current.phase.clone(),
            context_tokens: current.context_tokens,
            generated: current.timing.committed_tokens,
            maximum_output_tokens: current.timing.maximum_output_tokens,
            output_policy_signature: current.output_policy_signature,
            host_content_changed: current.timing.committed_tokens > initial.timing.committed_tokens,
        });
    }
    let mut recurrent = 0u64;
    let mut has_prefill = false;
    let mut has_decode = false;
    for (position, (work, row)) in input.work.iter().zip(input.rows).enumerate() {
        poll()?;
        if input.work[..position]
            .iter()
            .any(|prior| prior.key == work.key)
        {
            return Err(PlanningUnknownReason::InvalidShapeEvidence);
        }
        let current = input
            .requests
            .iter()
            .find(|r| r.key == work.key)
            .ok_or(PlanningUnknownReason::InvalidShapeEvidence)?;
        if !same_request(current, row.request) {
            return Err(PlanningUnknownReason::InvalidShapeEvidence);
        }
        let expected = match (&work.action, &current.phase) {
            (WaveAction::Decode, RequestPhaseView::Decode) if current.context_tokens > 0 => {
                has_decode = true;
                ActualRowWork::Decode {
                    kv_tokens: current.context_tokens,
                }
            }
            (WaveAction::Prefill { offset, count }, RequestPhaseView::Prefill(progress)) => {
                let end = offset
                    .checked_add(count.get())
                    .ok_or(PlanningUnknownReason::ArithmeticOverflow)?;
                if *offset != progress.offset || end > progress.executable_until {
                    return Err(PlanningUnknownReason::InvalidShapeEvidence);
                }
                has_prefill = true;
                ActualRowWork::Prefill {
                    offset: *offset,
                    count: count.get(),
                    total_prompt_tokens: progress.total_prompt_tokens.get(),
                }
            }
            _ => return Err(PlanningUnknownReason::InvalidShapeEvidence),
        };
        if expected != row.work || current.timing.completed() {
            return Err(PlanningUnknownReason::InvalidShapeEvidence);
        }
        recurrent = recurrent
            .checked_add(current.recurrent_state_bytes)
            .ok_or(PlanningUnknownReason::ArithmeticOverflow)?;
    }
    let kind = match (has_prefill, has_decode) {
        (true, true) => ActualWaveKind::Mixed,
        (true, false) => ActualWaveKind::Prefill,
        (false, true) => ActualWaveKind::Decode,
        _ => return Err(PlanningUnknownReason::InvalidShapeEvidence),
    };
    if kind != input.kind || recurrent != input.recurrent_state_bytes {
        return Err(PlanningUnknownReason::InvalidShapeEvidence);
    }
    poll()?;
    Ok(frontiers)
}

// Arc-bound immutable reference evidence is compared by identity, avoiding an
// unpolled scan of reference curves while still validating every scalar field.
fn same_request(left: &RequestSchedulingView, right: &RequestSchedulingView) -> bool {
    left.key == right.key
        && left.timing == right.timing
        && frontier::same_phase(&left.phase, &right.phase)
        && left.readiness == right.readiness
        && left.context_tokens == right.context_tokens
        && left.recurrent_state_bytes == right.recurrent_state_bytes
        && left.output_credit == right.output_credit
        && left.output_policy_signature == right.output_policy_signature
        && left.fairness_rank == right.fairness_rank
        && left.recovery_service == right.recovery_service
        && left.ranking_service_cost_ns == right.ranking_service_cost_ns
        && left.optimistic_next_service == right.optimistic_next_service
}
