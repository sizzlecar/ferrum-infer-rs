//! Source-less cache restore uses the captured queue's same numeric successors.
use super::*;
use ferrum_interfaces::vnext::FutureCheckpointCostQuery;

pub(in crate::continuous_engine::inner::slo_controller) struct ReadyPrefixExecutorShape<'a> {
    pub source: ExecutorShape<'a>,
    pub offer: &'a ReadyPrefixRestoreOffer,
    pub lease: &'a dyn ferrum_interfaces::model_executor::PrefixCaptureLease,
}

impl PlanningExecutionContext for ReadyPrefixExecutorShape<'_> {
    fn begin<'epoch>(
        &'epoch self,
        snapshot: &'epoch SchedulerSnapshot,
        poll: &mut dyn FnMut() -> ProjectionResult<()>,
    ) -> ProjectionResult<Arc<dyn PlanningExecutionState<'epoch> + 'epoch>> {
        poll()?;
        if !std::ptr::eq(snapshot, &self.source.captured.snapshot)
            || self.offer.based_on_generation != snapshot.generation
            || !snapshot
                .requests
                .iter()
                .any(|row| row.key == self.offer.target)
            || self.lease.boundary() != self.offer.boundary_tokens.get() as usize
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
            ready_prefix: Some(self.offer.clone()),
            cache_capture: None,
            prefix_lease: Some(self.lease),
            prefix_restored: false,
        }))
    }
}

pub(super) fn restored_initial(
    offer: &ReadyPrefixRestoreOffer,
    initial: &RequestSchedulingView,
    restored: bool,
) -> Option<RequestSchedulingView> {
    if !restored
        || initial.key != offer.target
        || initial.context_tokens != 0
        || initial.timing.committed_tokens != 0
    {
        return None;
    }
    let mut adjusted = initial.clone();
    let RequestPhaseView::Prefill(progress) = &mut adjusted.phase else {
        return None;
    };
    if progress.offset != 0 {
        return None;
    }
    progress.offset = offer.boundary_tokens.get();
    progress.logical_high_water = progress.logical_high_water.max(progress.offset);
    adjusted.context_tokens = progress.offset;
    Some(adjusted)
}

impl<'epoch> ExecutorState<'epoch> {
    pub(super) fn bind_ready(
        &self,
        input: &PlanningReadyPrefixInput<'_>,
        poll: &mut dyn FnMut() -> ProjectionResult<()>,
    ) -> ProjectionResult<Option<Arc<dyn PlanningExecutionState<'epoch> + 'epoch>>> {
        poll()?;
        let captured = self.source.captured;
        if self.ready_prefix.as_ref() != Some(input.offer)
            || !std::ptr::eq(input.snapshot, &captured.snapshot)
            || self.depth != 0
            || self.domain.states.len() != 1
            || !self.domain.checkpoints.is_empty()
        {
            return Err(PlanningUnknownReason::InvalidSnapshot);
        }
        let target = input
            .snapshot
            .requests
            .iter()
            .position(|r| r.key == input.offer.target)
            .ok_or(PlanningUnknownReason::InvalidSnapshot)?;
        let lease = self
            .prefix_lease
            .ok_or(PlanningUnknownReason::UnknownResourceEvidence)?;
        let mut failure = None;
        let mut domain = self.domain.clone();
        let restored = input.phase == ReadyPrefixPhase::Restored;
        {
            let mut budget = || match poll() {
                Ok(()) if failure.is_none() => true,
                Ok(()) => false,
                Err(reason) => {
                    failure.get_or_insert(reason);
                    false
                }
            };
            if restored {
                match self
                    .source
                    .engine
                    .model_executor
                    .execution_checkpoint_restore_completed(
                        &captured.route,
                        lease,
                        target,
                        &mut captured.budget.observed_resource_budget(&mut budget),
                    ) {
                    ExecutionCostRouteAvailability::Known(true) => {}
                    ExecutionCostRouteAvailability::Unknown(
                        ExecutionCostRouteUnknown::BudgetExhausted,
                    ) => {
                        return Err(PlanningUnknownReason::ComputeBudgetExhausted);
                    }
                    _ => return Ok(None),
                }
            } else {
                let bound = match self
                    .source
                    .engine
                    .model_executor
                    .bind_execution_ready_checkpoint(
                        &captured.route,
                        &domain.states[0],
                        lease,
                        &mut captured.budget.observed_resource_budget(&mut budget),
                    ) {
                    ExecutionCostRouteAvailability::Known(value) => value,
                    ExecutionCostRouteAvailability::Unknown(
                        ExecutionCostRouteUnknown::BudgetExhausted,
                    ) => {
                        return Err(PlanningUnknownReason::ComputeBudgetExhausted);
                    }
                    _ => return Ok(None),
                };
                if bound.checkpoint.byte_plan().boundary()
                    != u64::from(input.offer.boundary_tokens.get())
                {
                    return Err(PlanningUnknownReason::InvalidShapeEvidence);
                }
                domain.states = vec![bound.state];
                domain.checkpoints = vec![bound.checkpoint];
            }
        }
        if let Some(reason) = failure {
            return Err(reason);
        }
        poll()?;
        Ok(Some(Arc::new(ExecutorState {
            source: ExecutorShape {
                engine: self.source.engine,
                captured,
            },
            domain,
            depth: 0,
            prefix: None,
            ready_prefix: self.ready_prefix.clone(),
            cache_capture: None,
            prefix_lease: self.prefix_lease,
            prefix_restored: restored,
        })))
    }

    pub(super) fn project_ready(
        &self,
        input: &PlanningReadyPrefixRestoreInput<'_>,
        poll: &mut dyn FnMut() -> ProjectionResult<()>,
    ) -> ProjectionResult<Option<ProjectedPrefixTransition<'epoch>>> {
        poll()?;
        let captured = self.source.captured;
        if self.ready_prefix.as_ref() != Some(input.offer)
            || self.prefix_restored
            || !std::ptr::eq(input.snapshot, &captured.snapshot)
            || input.requests.len() != captured.snapshot.requests.len()
        {
            return Err(PlanningUnknownReason::InvalidSnapshot);
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
        let target = input
            .requests
            .iter()
            .position(|r| r.key == input.offer.target)
            .ok_or(PlanningUnknownReason::InvalidShapeEvidence)?;
        let follower = &input.requests[target];
        let RequestPhaseView::Prefill(progress) = &follower.phase else {
            return Err(PlanningUnknownReason::InvalidShapeEvidence);
        };
        if progress.offset != 0
            || follower.context_tokens != 0
            || follower.timing.committed_tokens != 0
            || follower.readiness != RequestReadiness::Ready
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
        let mut states = Vec::with_capacity(count);
        let mut checkpoints = Vec::with_capacity(count);
        let mut shapes = Vec::with_capacity(count);
        for state in &self.domain.states {
            poll()?;
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
                        FutureCheckpointCostQuery::Restore {
                            checkpoint,
                            target,
                            prompt_tokens: u64::from(progress.total_prompt_tokens.get()),
                        },
                        &mut captured.budget.observed_resource_budget(&mut budget),
                    )
            };
            if let Some(reason) = failure {
                return Err(reason);
            }
            let projected = match result {
                ExecutionCostRouteAvailability::Known(value) => value,
                ExecutionCostRouteAvailability::Unknown(
                    ExecutionCostRouteUnknown::BudgetExhausted,
                ) => {
                    return Err(PlanningUnknownReason::ComputeBudgetExhausted);
                }
                _ => return Ok(None),
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
            PlanningShapeDomain::Exact(shapes.pop().unwrap())
        } else {
            PlanningShapeDomain::HostContentAlternatives(shapes)
        };
        Ok(Some(ProjectedPrefixTransition {
            cost_domain,
            restored_frontier: Some(PrefixRestoredFrontier {
                target: input.offer.target.clone(),
                previous_offset: 0,
                restored_tokens: input.offer.boundary_tokens.get(),
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
                prefix: None,
                ready_prefix: self.ready_prefix.clone(),
                cache_capture: None,
                prefix_lease: self.prefix_lease,
                prefix_restored: true,
            }),
        }))
    }
}
