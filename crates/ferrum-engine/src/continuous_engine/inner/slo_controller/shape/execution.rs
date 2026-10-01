//! One immutable execution-domain successor per logical edge. The scheduler
//! owns logical progress; this adapter never advances a second copy of it.
use super::*;
use ferrum_scheduler::implementations::continuous::slo_planner::{
    PlanningExecutionContext, PlanningExecutionInput, PlanningExecutionState, PlanningGraphDomain,
    PlanningProjectionDiagnostic, PrefixRendezvousOffer, ProjectedExecution,
};
mod capture;
mod prefix;
pub(in crate::continuous_engine::inner::slo_controller) use capture::CacheCaptureExecutorShape;
mod ready;
pub(in crate::continuous_engine::inner::slo_controller) use prefix::PrefixExecutorShape;
pub(in crate::continuous_engine::inner::slo_controller) use ready::ReadyPrefixExecutorShape;

type ProjectionResult<T> = std::result::Result<T, PlanningUnknownReason>;

struct ExecutorState<'epoch> {
    source: ExecutorShape<'epoch>,
    domain: RouteDomain,
    depth: usize,
    prefix: Option<PrefixRendezvousOffer>,
    ready_prefix: Option<ReadyPrefixRestoreOffer>,
    cache_capture: Option<capture::CacheCaptureBinding>,
    prefix_lease: Option<&'epoch dyn ferrum_interfaces::model_executor::PrefixCaptureLease>,
    prefix_restored: bool,
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
            prefix: None,
            ready_prefix: None,
            cache_capture: None,
            prefix_lease: None,
            prefix_restored: false,
        }))
    }
}

impl<'epoch> PlanningExecutionState<'epoch> for ExecutorState<'epoch> {
    fn bind_prefix_cache_capture(
        &self,
        input: &PlanningPrefixCacheCaptureBindingInput<'_>,
        poll: &mut dyn FnMut() -> ProjectionResult<()>,
    ) -> ProjectionResult<Option<Arc<dyn PlanningExecutionState<'epoch> + 'epoch>>> {
        self.bind_cache_capture(input, poll)
    }
    fn project_prefix_cache_capture(
        &self,
        input: &PlanningPrefixCacheCaptureInput<'_>,
        poll: &mut dyn FnMut() -> ProjectionResult<()>,
    ) -> ProjectionResult<Option<ProjectedPrefixTransition<'epoch>>> {
        self.project_cache_capture(input, poll)
    }
    fn bind_ready_prefix(
        &self,
        input: &PlanningReadyPrefixInput<'_>,
        poll: &mut dyn FnMut() -> ProjectionResult<()>,
    ) -> ProjectionResult<Option<Arc<dyn PlanningExecutionState<'epoch> + 'epoch>>> {
        self.bind_ready(input, poll)
    }
    fn project_ready_prefix_restore(
        &self,
        input: &PlanningReadyPrefixRestoreInput<'_>,
        poll: &mut dyn FnMut() -> ProjectionResult<()>,
    ) -> ProjectionResult<Option<ProjectedPrefixTransition<'epoch>>> {
        self.project_ready(input, poll)
    }
    fn bind_prefix_continuation(
        &self,
        input: &ferrum_scheduler::implementations::continuous::slo_planner::PlanningPrefixContinuationInput<'_>,
        poll: &mut dyn FnMut() -> ProjectionResult<()>,
    ) -> ProjectionResult<Option<Arc<dyn PlanningExecutionState<'epoch> + 'epoch>>> {
        self.bind_prefix(input, poll)
    }
    fn project_prefix_transition(
        &self,
        input: &ferrum_scheduler::implementations::continuous::slo_planner::PlanningPrefixTransitionInput<'_>,
        poll: &mut dyn FnMut() -> ProjectionResult<()>,
    ) -> ProjectionResult<
        Option<
            ferrum_scheduler::implementations::continuous::slo_planner::ProjectedPrefixTransition<
                'epoch,
            >,
        >,
    > {
        self.project_prefix(input, poll)
    }
    fn cost_workload_domain(
        &self,
    ) -> Option<&ferrum_interfaces::execution_cost::CostWorkloadDomainV1> {
        self.source.engine.cost_runtime.as_ref()?.workload_domain()
    }

    fn graph_domain(&self) -> ProjectionResult<PlanningGraphDomain> {
        use ferrum_interfaces::vnext::DeviceCostGraphConfiguration as Configuration;
        let route = &self.source.captured.route;
        let stream = route.graph_stream_state();
        let catalog = route.graph_catalog();
        // The quiescent lane capture already checks catalog/stream identity.
        // Retain that check here: a catalog is not a substitute for its stream.
        if catalog.is_some_and(|catalog| Some(catalog.stream_state()) != stream) {
            return Err(PlanningUnknownReason::UnknownResourceEvidence);
        }
        match stream {
            // Unsupported graph backends (including Metal) legitimately expose
            // neither stream graph state nor catalog. Exact Disabled stays exact;
            // this fallback cannot authorize any Warm/ConfiguredEager label.
            None => Ok(PlanningGraphDomain::SnapshotExact),
            Some(state) if state.is_unconfigured_empty() => Ok(PlanningGraphDomain::SnapshotExact),
            Some(state) if state.is_ready() && catalog.is_some() => match state.configuration() {
                Configuration::OnDemand => Ok(PlanningGraphDomain::ConfiguredPerWave),
                Configuration::StartupReady => Ok(PlanningGraphDomain::ResidentReplayOnly),
                _ => Err(PlanningUnknownReason::UnknownResourceEvidence),
            },
            _ => Err(PlanningUnknownReason::UnknownResourceEvidence),
        }
        // This only chooses the validation domain. Every projection still has
        // to match its actual core route and uploaded programs in this capture.
    }

    fn project(
        &self,
        input: &PlanningExecutionInput<'_>,
        poll: &mut dyn FnMut() -> ProjectionResult<()>,
    ) -> ProjectionResult<Option<ProjectedExecution<'epoch>>> {
        let _diagnostic = self
            .source
            .captured
            .budget
            .diagnostic_scope("candidate_projection");
        let diagnostic_span = tracing::trace_span!(target: "ferrum::slo_transaction",
            "candidate_projection", generation=self.source.captured.snapshot.generation,
            depth=self.depth, rows=input.rows.len(), kind=?input.kind);
        let _entered = diagnostic_span.enter();
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
        let invalid = |diagnostic: PlanningProjectionDiagnostic| {
            diagnostic.trace(snapshot, Some(self.depth), input.kind, input.rows.len());
            PlanningUnknownReason::InvalidShapeEvidence
        };
        let frontiers = input_frontiers_with_prefix(
            snapshot,
            input,
            self.depth,
            self.prefix.as_ref(),
            self.ready_prefix.as_ref(),
            self.cache_capture.as_ref(),
            self.prefix_restored,
            poll,
        )?;
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
                .ok_or_else(|| invalid(PlanningProjectionDiagnostic::WorkOwnerMissing))?;
            let row = input
                .rows
                .iter()
                .find(|row| row.request.key == work.key)
                .ok_or_else(|| invalid(PlanningProjectionDiagnostic::WorkRowMissing))?;
            rows.push(PreparedRow {
                index,
                work: row.work,
            });
        }
        let projected = self
            .source
            .project_domain(&self.domain, &frontiers, &rows, poll)?;
        poll()?;
        let Some((canonical_domain, statistical_evidence, host_content_forecasts, domain)) =
            projected
        else {
            return Ok(None);
        };
        if self.depth == 0 && canonical_domain.exact().is_none() {
            return Err(invalid(PlanningProjectionDiagnostic::RootNotExact));
        }
        if canonical_domain
            .shapes()
            .iter()
            .any(|shape| shape.kind != input.kind)
        {
            return Err(invalid(PlanningProjectionDiagnostic::ProjectionKind));
        }
        Ok(Some(ProjectedExecution {
            host_content_forecasts,
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
                prefix: self.prefix.clone(),
                ready_prefix: self.ready_prefix.clone(),
                cache_capture: self.cache_capture.clone(),
                prefix_lease: self.prefix_lease,
                prefix_restored: self.prefix_restored,
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
    input_frontiers_with_prefix(snapshot, input, depth, None, None, None, false, poll)
}

fn input_frontiers_with_prefix(
    snapshot: &SchedulerSnapshot,
    input: &PlanningExecutionInput<'_>,
    depth: usize,
    prefix: Option<&PrefixRendezvousOffer>,
    ready: Option<&ReadyPrefixRestoreOffer>,
    capture: Option<&capture::CacheCaptureBinding>,
    prefix_restored: bool,
    poll: &mut dyn FnMut() -> ProjectionResult<()>,
) -> ProjectionResult<Vec<ProjectedRequest>> {
    poll()?;
    let invalid = |diagnostic: PlanningProjectionDiagnostic| {
        diagnostic.trace(snapshot, Some(depth), input.kind, input.rows.len());
        PlanningUnknownReason::InvalidShapeEvidence
    };
    if input.requests.len() != snapshot.requests.len()
        || input.requests.is_empty()
        || input.requests.len() > 256
        || input.work.is_empty()
        || input.work.len() != input.rows.len()
        || input.work.len() > snapshot.capabilities.max_wave_rows.get()
        || depth > 16
    {
        return Err(invalid(PlanningProjectionDiagnostic::FrontierEnvelope));
    }
    let mut frontiers = Vec::with_capacity(input.requests.len());
    for (current, initial) in input.requests.iter().zip(&snapshot.requests) {
        poll()?;
        let adjusted = prefix
            .and_then(|offer| prefix::comparison_initial(offer, initial, current, prefix_restored))
            .or_else(|| {
                ready.and_then(|offer| ready::restored_initial(offer, initial, prefix_restored))
            })
            .or_else(|| {
                capture.and_then(|binding| capture::comparison_initial(binding, initial, current))
            });
        let initial = adjusted.as_ref().unwrap_or(initial);
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
            return Err(invalid(PlanningProjectionDiagnostic::FrontierIdentity));
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
                    return Err(invalid(PlanningProjectionDiagnostic::FrontierPrefill));
                }
            }
            (RequestPhaseView::Prefill(before), RequestPhaseView::Decode)
                if current.context_tokens >= before.total_prompt_tokens.get()
                    && current.timing.committed_tokens > initial.timing.committed_tokens => {}
            _ => return Err(invalid(PlanningProjectionDiagnostic::FrontierPhase)),
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
            return Err(invalid(PlanningProjectionDiagnostic::WorkDuplicateOwner));
        }
        let current = input
            .requests
            .iter()
            .find(|r| r.key == work.key)
            .ok_or_else(|| invalid(PlanningProjectionDiagnostic::WorkOwnerMissing))?;
        if !same_request(current, row.request) {
            return Err(invalid(PlanningProjectionDiagnostic::WorkIdentity));
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
                    return Err(invalid(PlanningProjectionDiagnostic::WorkRange));
                }
                has_prefill = true;
                ActualRowWork::Prefill {
                    offset: *offset,
                    count: count.get(),
                    total_prompt_tokens: progress.total_prompt_tokens.get(),
                }
            }
            _ => return Err(invalid(PlanningProjectionDiagnostic::WorkPhase)),
        };
        if expected != row.work || current.timing.completed() {
            return Err(invalid(PlanningProjectionDiagnostic::WorkRowOrCompleted));
        }
        recurrent = recurrent
            .checked_add(current.recurrent_state_bytes)
            .ok_or(PlanningUnknownReason::ArithmeticOverflow)?;
    }
    let kind = match (has_prefill, has_decode) {
        (true, true) => ActualWaveKind::Mixed,
        (true, false) => ActualWaveKind::Prefill,
        (false, true) => ActualWaveKind::Decode,
        _ => return Err(invalid(PlanningProjectionDiagnostic::WorkPhase)),
    };
    if kind != input.kind || recurrent != input.recurrent_state_bytes {
        return Err(invalid(PlanningProjectionDiagnostic::WorkKindOrRecurrent));
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
