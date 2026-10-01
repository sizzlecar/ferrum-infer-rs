//! Checkpoint edges remain inside the original complete-queue execution state.
use super::*;
use ferrum_interfaces::vnext::FutureCheckpointCostQuery;
use ferrum_scheduler::implementations::continuous::slo_planner::{
    PlanningPrefixContinuationInput, PlanningPrefixTransitionInput, PrefixContinuationPhase,
    PrefixMaintenanceStage, PrefixRestoredFrontier, ProjectedPrefixTransition, RequestReadiness,
};

pub(in crate::continuous_engine::inner::slo_controller) struct PrefixExecutorShape<'a> {
    pub source: ExecutorShape<'a>,
    pub offer: &'a PrefixRendezvousOffer,
    pub lease: Option<&'a dyn ferrum_interfaces::model_executor::PrefixCaptureLease>,
}

impl PlanningExecutionContext for PrefixExecutorShape<'_> {
    fn begin<'epoch>(
        &'epoch self,
        snapshot: &'epoch SchedulerSnapshot,
        poll: &mut dyn FnMut() -> ProjectionResult<()>,
    ) -> ProjectionResult<Arc<dyn PlanningExecutionState<'epoch> + 'epoch>> {
        poll()?;
        if !std::ptr::eq(snapshot, &self.source.captured.snapshot)
            || self.offer.based_on_generation != snapshot.generation
            || self.offer.producer == self.offer.target
        {
            return Err(PlanningUnknownReason::InvalidSnapshot);
        }
        self.source.initial_frontiers(poll)?;
        for key in [&self.offer.producer, &self.offer.target] {
            if !snapshot.requests.iter().any(|row| &row.key == key) {
                return Err(PlanningUnknownReason::InvalidSnapshot);
            }
        }
        poll()?;
        Ok(Arc::new(ExecutorState {
            source: ExecutorShape {
                engine: self.source.engine,
                captured: self.source.captured,
            },
            domain: RouteDomain::initial(self.source.captured.route.initial_state()),
            depth: 0,
            prefix: Some(self.offer.clone()),
            ready_prefix: None,
            cache_capture: None,
            prefix_lease: self.lease,
            prefix_restored: false,
        }))
    }
}

/// Only the offer's producer limit and initially fresh target readiness differ
/// in the waiting branch. Every original ingress/reference/milestone stays put.
pub(super) fn comparison_initial(
    offer: &PrefixRendezvousOffer,
    initial: &RequestSchedulingView,
    current: &RequestSchedulingView,
    restored: bool,
) -> Option<RequestSchedulingView> {
    let mut adjusted = initial.clone();
    let mut changed = false;
    if initial.key == offer.producer {
        if let (RequestPhaseView::Prefill(before), RequestPhaseView::Prefill(now)) =
            (&mut adjusted.phase, &current.phase)
        {
            let boundary = offer.boundary_tokens.get();
            if now.executable_until == boundary
                && boundary <= before.executable_until
                && boundary > before.offset
            {
                before.executable_until = boundary;
                changed = true;
            }
        }
    }
    if initial.key == offer.target
        && initial.readiness == RequestReadiness::Ready
        && current.readiness == RequestReadiness::StateBlocked
        && initial.context_tokens == 0
        && initial.timing.committed_tokens == 0
        && matches!(&initial.phase, RequestPhaseView::Prefill(p) if p.offset == 0)
    {
        adjusted.readiness = RequestReadiness::StateBlocked;
        changed = true;
    }
    if restored && initial.key == offer.target {
        let boundary = offer.boundary_tokens.get();
        if let RequestPhaseView::Prefill(progress) = &mut adjusted.phase {
            if progress.offset == 0
                && initial.context_tokens == 0
                && initial.timing.committed_tokens == 0
            {
                progress.offset = boundary;
                progress.logical_high_water = progress.logical_high_water.max(boundary);
                adjusted.context_tokens = boundary;
                adjusted.readiness = RequestReadiness::Ready;
                changed = true;
            }
        }
    }
    changed.then_some(adjusted)
}

impl<'epoch> ExecutorState<'epoch> {
    pub(super) fn bind_prefix(
        &self,
        input: &PlanningPrefixContinuationInput<'_>,
        poll: &mut dyn FnMut() -> ProjectionResult<()>,
    ) -> ProjectionResult<Option<Arc<dyn PlanningExecutionState<'epoch> + 'epoch>>> {
        poll()?;
        let captured = self.source.captured;
        if self.prefix.as_ref() != Some(input.offer)
            || !std::ptr::eq(input.snapshot, &captured.snapshot)
            || self.depth != 0
            || self.domain.states.len() != 1
            || !self.domain.checkpoints.is_empty()
        {
            return Err(PlanningUnknownReason::InvalidSnapshot);
        }
        let source = input
            .snapshot
            .requests
            .iter()
            .position(|r| r.key == input.offer.producer)
            .ok_or(PlanningUnknownReason::InvalidSnapshot)?;
        let target = input
            .snapshot
            .requests
            .iter()
            .position(|r| r.key == input.offer.target)
            .ok_or(PlanningUnknownReason::InvalidSnapshot)?;
        let boundary = input.offer.boundary_tokens.get();
        let mut domain = self.domain.clone();
        let restored = matches!(input.phase, PrefixContinuationPhase::Restored);
        match input.phase {
            PrefixContinuationPhase::HeldAwaitingProducer => {
                if !matches!(&input.snapshot.requests[source].phase,
                    RequestPhaseView::Prefill(progress) if progress.offset < boundary)
                {
                    return Err(PlanningUnknownReason::InvalidSnapshot);
                }
            }
            PrefixContinuationPhase::AtCaptureBoundary { capture_span_start } => {
                let proof = captured
                    .resources
                    .participants()
                    .get(source)
                    .and_then(|p| p.completed_checkpoint_boundary())
                    .ok_or(PlanningUnknownReason::UnknownResourceEvidence)?;
                if proof.span_start() != u64::from(capture_span_start)
                    || proof.completed_tokens() != u64::from(boundary)
                    || !matches!(&input.snapshot.requests[source].phase,
                        RequestPhaseView::Prefill(progress)
                            if progress.offset == boundary
                                && u64::from(progress.total_prompt_tokens.get()) == proof.prompt_tokens())
                {
                    return Err(PlanningUnknownReason::UnknownResourceEvidence);
                }
            }
            PrefixContinuationPhase::CheckpointReady { capture_span_start } => {
                let lease = self
                    .prefix_lease
                    .ok_or(PlanningUnknownReason::UnknownResourceEvidence)?;
                let mut failure = None;
                let result = {
                    let mut budget = || match poll() {
                        Ok(()) if failure.is_none() => true,
                        Ok(()) => false,
                        Err(reason) => {
                            failure.get_or_insert(reason);
                            false
                        }
                    };
                    self.source
                        .engine
                        .model_executor
                        .bind_execution_retained_checkpoint(
                            &captured.route,
                            &domain.states[0],
                            lease,
                            source,
                            &mut captured.budget.observed_resource_budget(&mut budget),
                        )
                };
                if let Some(reason) = failure {
                    return Err(reason);
                }
                let bound = match result {
                    ExecutionCostRouteAvailability::Known(value) => value,
                    ExecutionCostRouteAvailability::Unknown(
                        ExecutionCostRouteUnknown::BudgetExhausted,
                    ) => return Err(PlanningUnknownReason::ComputeBudgetExhausted),
                    ExecutionCostRouteAvailability::Unknown(_) => return Ok(None),
                };
                if capture_span_start >= boundary
                    || bound.checkpoint.byte_plan().boundary() != u64::from(boundary)
                {
                    return Err(PlanningUnknownReason::InvalidShapeEvidence);
                }
                domain.states = vec![bound.state];
                domain.checkpoints = vec![bound.checkpoint];
            }
            PrefixContinuationPhase::Restored => {
                let lease = self
                    .prefix_lease
                    .ok_or(PlanningUnknownReason::UnknownResourceEvidence)?;
                let mut failure = None;
                let result = {
                    let mut budget = || match poll() {
                        Ok(()) if failure.is_none() => true,
                        Ok(()) => false,
                        Err(reason) => {
                            failure.get_or_insert(reason);
                            false
                        }
                    };
                    self.source
                        .engine
                        .model_executor
                        .execution_checkpoint_restore_completed(
                            &captured.route,
                            lease,
                            target,
                            &mut captured.budget.observed_resource_budget(&mut budget),
                        )
                };
                if let Some(reason) = failure {
                    return Err(reason);
                }
                match result {
                    ExecutionCostRouteAvailability::Known(true) => {}
                    ExecutionCostRouteAvailability::Unknown(
                        ExecutionCostRouteUnknown::BudgetExhausted,
                    ) => return Err(PlanningUnknownReason::ComputeBudgetExhausted),
                    _ => return Ok(None),
                }
            }
        }
        poll()?;
        Ok(Some(Arc::new(ExecutorState {
            source: ExecutorShape {
                engine: self.source.engine,
                captured,
            },
            domain,
            depth: 0,
            prefix: self.prefix.clone(),
            ready_prefix: self.ready_prefix.clone(),
            cache_capture: None,
            prefix_lease: self.prefix_lease,
            prefix_restored: restored,
        })))
    }
    pub(super) fn project_prefix(
        &self,
        input: &PlanningPrefixTransitionInput<'_>,
        poll: &mut dyn FnMut() -> ProjectionResult<()>,
    ) -> ProjectionResult<Option<ProjectedPrefixTransition<'epoch>>> {
        poll()?;
        let captured = self.source.captured;
        if self.prefix.as_ref() != Some(input.offer)
            || !std::ptr::eq(input.snapshot, &captured.snapshot)
            || input.requests.len() != captured.snapshot.requests.len()
        {
            return Err(PlanningUnknownReason::InvalidSnapshot);
        }
        // Retain every owner and original time contract. The scheduler owns
        // logical simulation; resource projection owns physical successors.
        for (current, original) in input.requests.iter().zip(&captured.snapshot.requests) {
            poll()?;
            if current.key != original.key
                || current.timing.ingress_at_ns != original.timing.ingress_at_ns
                || current.timing.budgets != original.timing.budgets
                || current.timing.maximum_output_tokens != original.timing.maximum_output_tokens
                || current.output_policy_signature != original.output_policy_signature
                || current.recurrent_state_bytes != original.recurrent_state_bytes
            {
                return Err(PlanningUnknownReason::InvalidShapeEvidence);
            }
        }
        let source = input
            .requests
            .iter()
            .position(|r| r.key == input.offer.producer)
            .ok_or(PlanningUnknownReason::InvalidShapeEvidence)?;
        let target = input
            .requests
            .iter()
            .position(|r| r.key == input.offer.target)
            .ok_or(PlanningUnknownReason::InvalidShapeEvidence)?;
        let RequestPhaseView::Prefill(follower) = &input.requests[target].phase else {
            return Err(PlanningUnknownReason::InvalidShapeEvidence);
        };
        let boundary = input.offer.boundary_tokens.get();
        if input.capture_span_start >= boundary
            || follower.offset != 0
            || input.requests[target].context_tokens != 0
            || input.requests[target].timing.committed_tokens != 0
            || input.requests[target].readiness != RequestReadiness::StateBlocked
        {
            return Err(PlanningUnknownReason::InvalidShapeEvidence);
        }
        let count = self.domain.states.len();
        if count == 0
            || count
                > self
                    .source
                    .engine
                    .config
                    .scheduler
                    .slo
                    .planner
                    .max_route_states
                    .get()
        {
            return Err(PlanningUnknownReason::ShapeCapacity);
        }
        if input.stage == PrefixMaintenanceStage::Capture && !self.domain.checkpoints.is_empty() {
            return Err(PlanningUnknownReason::InvalidShapeEvidence);
        }
        let mut states = Vec::with_capacity(count);
        let mut checkpoints = Vec::with_capacity(count);
        let mut shapes = Vec::with_capacity(count);
        for state in &self.domain.states {
            poll()?;
            let query = match input.stage {
                PrefixMaintenanceStage::Capture => {
                    let RequestPhaseView::Prefill(producer) = &input.requests[source].phase else {
                        return Err(PlanningUnknownReason::InvalidShapeEvidence);
                    };
                    if producer.offset != boundary {
                        return Err(PlanningUnknownReason::InvalidShapeEvidence);
                    }
                    FutureCheckpointCostQuery::Capture {
                        source,
                        span_start: u64::from(input.capture_span_start),
                        boundary: u64::from(boundary),
                        prompt_tokens: u64::from(producer.total_prompt_tokens.get()),
                    }
                }
                PrefixMaintenanceStage::Restore => {
                    let mut matching = self
                        .domain
                        .checkpoints
                        .iter()
                        .filter(|c| state.retains_checkpoint(c));
                    let checkpoint = matching
                        .next()
                        .ok_or(PlanningUnknownReason::UnknownResourceEvidence)?;
                    if matching.next().is_some() {
                        return Err(PlanningUnknownReason::InvalidShapeEvidence);
                    }
                    FutureCheckpointCostQuery::Restore {
                        checkpoint,
                        target,
                        prompt_tokens: u64::from(follower.total_prompt_tokens.get()),
                    }
                }
            };
            let mut failure = None;
            let projected = {
                let mut budget = || match poll() {
                    Ok(()) if failure.is_none() => true,
                    Ok(()) => false,
                    Err(reason) => {
                        failure.get_or_insert(reason);
                        false
                    }
                };
                self.source
                    .engine
                    .model_executor
                    .project_execution_checkpoint(
                        &captured.route,
                        state,
                        query,
                        &mut captured.budget.observed_resource_budget(&mut budget),
                    )
            };
            if let Some(reason) = failure {
                return Err(reason);
            }
            poll()?;
            let projected = match projected {
                ExecutionCostRouteAvailability::Known(value) => value,
                ExecutionCostRouteAvailability::Unknown(
                    ExecutionCostRouteUnknown::BudgetExhausted,
                ) => return Err(PlanningUnknownReason::ComputeBudgetExhausted),
                ExecutionCostRouteAvailability::Unknown(_) => return Ok(None),
            };
            shapes.push(
                crate::continuous_engine::inner::cost_observation::prefix_cost_shape(
                    &projected.cost_domain,
                    Some(&projected.host_work),
                )?,
            );
            states.push(projected.state);
            checkpoints.push(projected.checkpoint);
        }
        poll()?;
        let cost_domain = if count == 1 {
            PlanningShapeDomain::Exact(
                shapes
                    .pop()
                    .ok_or(PlanningUnknownReason::InvalidShapeEvidence)?,
            )
        } else {
            PlanningShapeDomain::HostContentAlternatives(shapes)
        };
        Ok(Some(ProjectedPrefixTransition {
            cost_domain,
            restored_frontier: (input.stage == PrefixMaintenanceStage::Restore).then(|| {
                PrefixRestoredFrontier {
                    target: input.offer.target.clone(),
                    previous_offset: 0,
                    restored_tokens: boundary,
                }
            }),
            successor: Arc::new(ExecutorState {
                source: ExecutorShape {
                    engine: self.source.engine,
                    captured,
                },
                domain: RouteDomain {
                    states,
                    checkpoints,
                    empirical: self.domain.empirical,
                },
                depth: self.depth,
                prefix: self.prefix.clone(),
                ready_prefix: self.ready_prefix.clone(),
                cache_capture: None,
                prefix_lease: self.prefix_lease,
                prefix_restored: self.prefix_restored
                    || input.stage == PrefixMaintenanceStage::Restore,
            }),
        }))
    }
}
