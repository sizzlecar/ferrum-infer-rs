//! Source-only cache preparation shares the actual execution-domain successor.
use super::*;
use ferrum_interfaces::vnext::FutureCheckpointCostQuery;

pub(in crate::continuous_engine::inner::slo_controller) struct CacheCaptureExecutorShape<'a> {
    pub source: ExecutorShape<'a>,
    pub offer: &'a PrefixCacheCaptureOffer,
    pub phase: PrefixCacheCapturePhase,
}
#[derive(Clone)]
pub(super) struct CacheCaptureBinding {
    offer: PrefixCacheCaptureOffer,
    phase: PrefixCacheCapturePhase,
    bound: bool,
}

pub(super) fn comparison_initial(
    binding: &CacheCaptureBinding,
    original: &RequestSchedulingView,
    current: &RequestSchedulingView,
) -> Option<RequestSchedulingView> {
    if !binding.bound
        || binding.phase != PrefixCacheCapturePhase::Preparing
        || original.key != binding.offer.source
    {
        return None;
    }
    let (RequestPhaseView::Prefill(before), RequestPhaseView::Prefill(now)) =
        (&original.phase, &current.phase)
    else {
        return None;
    };
    let boundary = binding.offer.boundary_tokens.get();
    if now.executable_until != boundary
        || boundary > before.executable_until
        || boundary <= before.offset
    {
        return None;
    }
    let mut adjusted = original.clone();
    let RequestPhaseView::Prefill(progress) = &mut adjusted.phase else {
        unreachable!()
    };
    progress.executable_until = boundary;
    Some(adjusted)
}

impl PlanningExecutionContext for CacheCaptureExecutorShape<'_> {
    fn begin<'epoch>(
        &'epoch self,
        snapshot: &'epoch SchedulerSnapshot,
        poll: &mut dyn FnMut() -> ProjectionResult<()>,
    ) -> ProjectionResult<Arc<dyn PlanningExecutionState<'epoch> + 'epoch>> {
        poll()?;
        if !std::ptr::eq(snapshot, &self.source.captured.snapshot)
            || self.offer.based_on_generation != snapshot.generation
        {
            return Err(PlanningUnknownReason::InvalidSnapshot);
        }
        self.source.initial_frontiers(poll)?;
        poll()?;
        Ok(Arc::new(ExecutorState {
            source: ExecutorShape {
                engine: self.source.engine,
                captured: self.source.captured,
            },
            domain: RouteDomain::initial(self.source.captured.route.initial_state()),
            depth: 0,
            prefix: None,
            ready_prefix: None,
            cache_capture: Some(CacheCaptureBinding {
                offer: self.offer.clone(),
                phase: self.phase,
                bound: false,
            }),
            prefix_lease: None,
            prefix_restored: false,
        }))
    }
}

impl<'epoch> ExecutorState<'epoch> {
    pub(super) fn bind_cache_capture(
        &self,
        input: &PlanningPrefixCacheCaptureBindingInput<'_>,
        poll: &mut dyn FnMut() -> ProjectionResult<()>,
    ) -> ProjectionResult<Option<Arc<dyn PlanningExecutionState<'epoch> + 'epoch>>> {
        poll()?;
        let captured = self.source.captured;
        let binding = self
            .cache_capture
            .as_ref()
            .ok_or(PlanningUnknownReason::InvalidSnapshot)?;
        if input.offer != &binding.offer
            || input.phase != binding.phase
            || binding.bound
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
            .position(|r| r.key == binding.offer.source)
            .ok_or(PlanningUnknownReason::InvalidSnapshot)?;
        let RequestPhaseView::Prefill(progress) = &input.snapshot.requests[source].phase else {
            return Err(PlanningUnknownReason::InvalidSnapshot);
        };
        let boundary = binding.offer.boundary_tokens.get();
        match input.phase {
            PrefixCacheCapturePhase::AtBoundary => {
                let actual = captured
                    .resources
                    .participants()
                    .get(source)
                    .and_then(|p| p.completed_checkpoint_boundary())
                    .ok_or(PlanningUnknownReason::UnknownResourceEvidence)?;
                if progress.offset != boundary
                    || actual.span_start() != u64::from(binding.offer.capture_span_start)
                    || actual.completed_tokens() != u64::from(boundary)
                    || actual.prompt_tokens() != u64::from(progress.total_prompt_tokens.get())
                {
                    return Err(PlanningUnknownReason::UnknownResourceEvidence);
                }
            }
            PrefixCacheCapturePhase::Preparing => {
                if progress.offset != binding.offer.capture_span_start
                    || progress.offset >= boundary
                    || boundary >= progress.total_prompt_tokens.get()
                {
                    return Err(PlanningUnknownReason::InvalidShapeEvidence);
                }
                let chunk = ferrum_interfaces::model_executor::PrefillChunk::new(
                    progress.offset as usize,
                    (boundary - progress.offset) as usize,
                    progress.total_prompt_tokens.get() as usize,
                )
                .map_err(|_| PlanningUnknownReason::InvalidShapeEvidence)?;
                if !self
                    .source
                    .engine
                    .model_executor
                    .plan_prompt_tail_capture_boundary(chunk)
                    .is_some_and(|declared| {
                        declared.boundary == boundary as usize
                            && declared.span.permits(u64::from(boundary - progress.offset))
                    })
                {
                    return Err(PlanningUnknownReason::UnknownResourceEvidence);
                }
            }
        }
        poll()?;
        Ok(Some(Arc::new(ExecutorState {
            source: ExecutorShape {
                engine: self.source.engine,
                captured,
            },
            domain: self.domain.clone(),
            depth: 0,
            prefix: None,
            ready_prefix: None,
            cache_capture: Some(CacheCaptureBinding {
                bound: true,
                ..binding.clone()
            }),
            prefix_lease: None,
            prefix_restored: false,
        })))
    }

    pub(super) fn project_cache_capture(
        &self,
        input: &PlanningPrefixCacheCaptureInput<'_>,
        poll: &mut dyn FnMut() -> ProjectionResult<()>,
    ) -> ProjectionResult<Option<ProjectedPrefixTransition<'epoch>>> {
        poll()?;
        let captured = self.source.captured;
        let binding = self
            .cache_capture
            .as_ref()
            .ok_or(PlanningUnknownReason::InvalidSnapshot)?;
        if !binding.bound
            || input.offer != &binding.offer
            || !std::ptr::eq(input.snapshot, &captured.snapshot)
            || input.requests.len() != captured.snapshot.requests.len()
            || !self.domain.checkpoints.is_empty()
            || self.domain.states.is_empty()
            || self.domain.states.len()
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
            return Err(PlanningUnknownReason::InvalidShapeEvidence);
        }
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
            .position(|r| r.key == binding.offer.source)
            .ok_or(PlanningUnknownReason::InvalidShapeEvidence)?;
        let RequestPhaseView::Prefill(progress) = &input.requests[source].phase else {
            return Err(PlanningUnknownReason::InvalidShapeEvidence);
        };
        let boundary = binding.offer.boundary_tokens.get();
        if progress.offset != boundary
            || input.capture_span_start >= boundary
            || (binding.phase == PrefixCacheCapturePhase::AtBoundary
                && input.capture_span_start != binding.offer.capture_span_start)
            || (binding.phase == PrefixCacheCapturePhase::Preparing
                && input.capture_span_start < binding.offer.capture_span_start)
        {
            return Err(PlanningUnknownReason::InvalidShapeEvidence);
        }
        let mut states = Vec::with_capacity(self.domain.states.len());
        let mut checkpoints = Vec::with_capacity(self.domain.states.len());
        let mut shapes = Vec::with_capacity(self.domain.states.len());
        for state in &self.domain.states {
            poll()?;
            let query = FutureCheckpointCostQuery::Capture {
                source,
                span_start: u64::from(input.capture_span_start),
                boundary: u64::from(boundary),
                prompt_tokens: u64::from(progress.total_prompt_tokens.get()),
            };
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
            let projected = match result {
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
        let cost_domain = if shapes.len() == 1 {
            PlanningShapeDomain::Exact(shapes.pop().unwrap())
        } else {
            PlanningShapeDomain::HostContentAlternatives(shapes)
        };
        Ok(Some(ProjectedPrefixTransition {
            cost_domain,
            restored_frontier: None,
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
                prefix: None,
                ready_prefix: None,
                cache_capture: None,
                prefix_lease: None,
                prefix_restored: false,
            }),
        }))
    }
}
