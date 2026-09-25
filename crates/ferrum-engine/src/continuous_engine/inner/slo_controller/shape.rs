use super::*;
use ferrum_interfaces::{
    execution_cost::{
        host_history_cost_signature, ActualRowWork, ActualWaveKind, CanonicalWaveCostShape,
        HostContentForecastV2, HostPendingConstraintV2,
    },
    vnext::{
        ExecutionCostRouteAvailability, ExecutionCostRouteState, ExecutionCostRouteUnknown,
        FutureCostOutput, FutureHostPendingQueryV2, FutureHostPendingRowV2, FutureWaveCostQuery,
        FutureWaveCostRow,
    },
};

mod domain;
mod execution;
mod frontier;
use domain::{FutureHostMode, RouteDomain};
use frontier::{PreparedRow, ProjectedFrontiers, ProjectedRequest};
#[cfg(test)]
mod tests;

pub(super) struct ExecutorShape<'a> {
    pub(super) engine: &'a EngineInner,
    pub(super) captured: &'a ControllerSnapshot,
}

impl EngineInner {
    pub(super) fn controller_first_wave_shape(
        &self,
        captured: &ControllerSnapshot,
        selected: &SelectedWave,
        valid_until: Instant,
    ) -> Option<CanonicalWaveCostShape> {
        if !captured.budget.poll() || slo_clock_now() > valid_until {
            return None;
        }
        let canonical = selected.replayed_first_wave(&captured.snapshot)?;
        if selected.candidate.execution_shape.exact()? != &canonical_cost_shape(canonical).ok()? {
            return None;
        }
        // Final replay already proved physical order and all canonical fields.
        // This copy crosses the owned ExpectedExecutionCostWave boundary only;
        // it does not re-run the provider, acquire authority or reset time.
        let canonical = canonical.clone();
        (captured.budget.poll() && slo_clock_now() <= valid_until).then_some(canonical)
    }

    pub(super) fn propose_slo_controller(&self, captured: &ControllerSnapshot) -> PlanningDecision {
        let planner = BoundedSloPlanner {
            settings: BoundedPlannerSettings {
                search: self.config.scheduler.slo.planner.clone(),
                ..Default::default()
            },
        };
        let cost = AnchoredPlanningCostModel::new(captured.model.as_ref(), captured.anchor);
        // Use the same Tokio-backed monotonic origin as request commit/waits.
        // The checked wrapper accounts for snapshot construction and replay.
        let Ok(window) = captured.budget.planning_window(&captured.origin) else {
            return PlanningDecision::Unknown {
                reason: PlanningUnknownReason::ClockMovedBackwards,
                search: Default::default(),
            };
        };
        captured
            .origin
            .propose_scoped_with_execution_budget_window(
                &planner,
                &captured.snapshot,
                &cost,
                &ExecutorShape {
                    engine: self,
                    captured,
                },
                captured
                    .protection
                    .needs_recovery()
                    .then(|| Arc::clone(&captured.protection)),
                window,
                slo_clock_now,
            )
            .unwrap_or_else(|_| PlanningDecision::Unknown {
                reason: PlanningUnknownReason::ClockMovedBackwards,
                search: Default::default(),
            })
    }
}

impl ExecutorShape<'_> {
    fn initial_frontiers(
        &self,
        poll: &mut dyn FnMut() -> std::result::Result<(), PlanningUnknownReason>,
    ) -> std::result::Result<ProjectedFrontiers, PlanningUnknownReason> {
        let snapshot = &self.captured.snapshot;
        if snapshot.requests.len() != self.captured.fences.len() {
            return Err(PlanningUnknownReason::InvalidSnapshot);
        }
        let projected = ProjectedFrontiers::new(
            &snapshot.requests,
            snapshot.capacity.maximum_context_tokens.get(),
            snapshot.capabilities.max_wave_rows.get(),
            poll,
        )?;
        for (request, fence) in snapshot.requests.iter().zip(&self.captured.fences) {
            poll()?;
            if request.key.request_id != fence.key.request_id
                || request.key.incarnation != fence.incarnation
                || request.key.work_generation != fence.key.generation
                || usize::try_from(request.context_tokens).ok() != Some(fence.context)
                || usize::try_from(request.timing.committed_tokens).ok() != Some(fence.generated)
                || matches!(request.phase, RequestPhaseView::Decode) != fence.prefill_complete
                || fence.host_features.is_some_and(|host| {
                    host.state.generated_tokens_before != u64::from(request.timing.committed_tokens)
                        || host.state.maximum_output_tokens
                            != u64::from(request.timing.maximum_output_tokens.get())
                })
            {
                return Err(PlanningUnknownReason::InvalidSnapshot);
            }
            if let RequestPhaseView::Prefill(progress) = &request.phase {
                if usize::try_from(progress.offset).ok() != Some(fence.prefill_tokens_processed)
                    || usize::try_from(progress.total_prompt_tokens.get()).ok()
                        != Some(fence.prefill_total)
                {
                    return Err(PlanningUnknownReason::InvalidSnapshot);
                }
            }
        }
        Ok(projected)
    }

    fn project(
        &self,
        state: &ExecutionCostRouteState,
        frontiers: &[ProjectedRequest],
        prepared: &[PreparedRow],
        mode: FutureHostMode,
        poll: &mut dyn FnMut() -> std::result::Result<(), PlanningUnknownReason>,
    ) -> std::result::Result<
        Option<(
            ferrum_interfaces::vnext::ExecutionCostRouteProjection,
            Option<HostContentForecastV2>,
        )>,
        PlanningUnknownReason,
    > {
        let mut rows = Vec::with_capacity(prepared.len());
        let full_logits = ferrum_interfaces::model_executor::LogitsReturnPolicy::FullLogits;
        for selected in prepared {
            poll()?;
            let index = selected.index;
            let request = &frontiers[index];
            let fence = &self.captured.fences[index];
            let Some(host) = domain::row_host(fence.host_features, request, mode)? else {
                return Ok(None);
            };
            let output = match selected.work {
                ActualRowWork::Decode { .. } => FutureCostOutput::Decode {
                    policy: if request.host_content_changed {
                        match mode {
                            FutureHostMode::Exact => return Ok(None),
                            FutureHostMode::Greedy => match fence.future_greedy_policy.as_ref() {
                                Some(policy) => policy,
                                None => return Ok(None),
                            },
                            FutureHostMode::FullLogits => &full_logits,
                        }
                    } else {
                        &fence.logits_policy
                    },
                },
                ActualRowWork::Prefill {
                    offset,
                    count,
                    total_prompt_tokens: total,
                } => {
                    let end = offset
                        .checked_add(count)
                        .ok_or(PlanningUnknownReason::ArithmeticOverflow)?;
                    let chunk = ferrum_interfaces::model_executor::PrefillChunk::new(
                        offset as usize,
                        count as usize,
                        total as usize,
                    )
                    .map_err(|_| PlanningUnknownReason::InvalidSnapshot)?;
                    // A prefix capture boundary could narrow the selected span
                    // or add a checkpoint wave. Neither belongs to this route.
                    if request.generated == 0
                        && self
                            .engine
                            .model_executor
                            .plan_prompt_tail_capture_boundary(chunk)
                            .is_some()
                    {
                        return Ok(None);
                    }
                    FutureCostOutput::Prefill {
                        final_logits: end == total,
                    }
                }
                _ => return Ok(None),
            };
            rows.push(FutureWaveCostRow {
                participant_index: index,
                work: selected.work,
                host_policy_signature: host_history_cost_signature(
                    request.output_policy_signature,
                    u64::from(request.generated),
                ),
                host_features: host,
                output,
            });
        }
        let has_prefill = rows
            .iter()
            .any(|row| matches!(row.work, ActualRowWork::Prefill { .. }));
        let has_decode = rows
            .iter()
            .any(|row| matches!(row.work, ActualRowWork::Decode { .. }));
        let kind = match (has_prefill, has_decode) {
            (true, true) => ActualWaveKind::Mixed,
            (true, false) => ActualWaveKind::Prefill,
            (false, true) => ActualWaveKind::Decode,
            _ => return Err(PlanningUnknownReason::InvalidSnapshot),
        };
        let mut failure = None;
        let needs_forecast = self.captured.model.evidence_requirement()
            == ferrum_scheduler::implementations::continuous::slo_planner::PlanningCostEvidenceRequirement::StructuredV2;
        let mut eligible = Vec::new();
        let mut fixed_full = kind == ActualWaveKind::Prefill;
        if needs_forecast && mode != FutureHostMode::Exact {
            eligible
                .try_reserve_exact(prepared.len())
                .map_err(|_| PlanningUnknownReason::ShapeCapacity)?;
            for (position, selected) in prepared.iter().enumerate() {
                poll()?;
                let request = &frontiers[selected.index];
                match rows[position].output {
                    FutureCostOutput::Decode { policy } if !request.host_content_changed => {
                        fixed_full |= policy.requires_full_logits();
                    }
                    FutureCostOutput::Decode { .. } if mode == FutureHostMode::FullLogits => {
                        let Some(clean_policy) = self.captured.fences[selected.index]
                            .future_greedy_policy
                            .as_ref()
                        else {
                            return Ok(None);
                        };
                        eligible.push(FutureHostPendingRowV2 {
                            physical_position: u32::try_from(position)
                                .map_err(|_| PlanningUnknownReason::ShapeCapacity)?,
                            clean_policy,
                        });
                    }
                    FutureCostOutput::Prefill { final_logits } => fixed_full |= final_logits,
                    _ => {}
                }
            }
        }
        let projection = {
            let mut budget = || match poll() {
                Ok(()) if failure.is_none() => true,
                Ok(()) => false,
                Err(reason) => {
                    failure.get_or_insert(reason);
                    false
                }
            };
            let query = FutureWaveCostQuery { kind, rows: &rows };
            if needs_forecast && mode != FutureHostMode::Exact {
                let host = FutureHostPendingQueryV2 {
                    eligible_rows: &eligible,
                    constraint: if mode == FutureHostMode::Greedy || fixed_full {
                        HostPendingConstraintV2::AnySubset
                    } else {
                        HostPendingConstraintV2::NonEmptySubset
                    },
                };
                match self
                    .engine
                    .model_executor
                    .project_execution_cost_wave_with_host_content(
                        &self.captured.route,
                        state,
                        &query,
                        &host,
                        &mut budget,
                    ) {
                    ExecutionCostRouteAvailability::Known(value) => {
                        ExecutionCostRouteAvailability::Known((
                            value.projection,
                            Some(value.host_content),
                        ))
                    }
                    ExecutionCostRouteAvailability::Unknown(reason) => {
                        ExecutionCostRouteAvailability::Unknown(reason)
                    }
                }
            } else {
                match self.engine.model_executor.project_execution_cost_wave(
                    &self.captured.route,
                    state,
                    &query,
                    &mut budget,
                ) {
                    ExecutionCostRouteAvailability::Known(value) => {
                        ExecutionCostRouteAvailability::Known((
                            value,
                            needs_forecast.then_some(HostContentForecastV2::Exact),
                        ))
                    }
                    ExecutionCostRouteAvailability::Unknown(reason) => {
                        ExecutionCostRouteAvailability::Unknown(reason)
                    }
                }
            }
        };
        let after = poll();
        if let Some(reason) = failure {
            return Err(reason);
        }
        after?;
        match projection {
            ExecutionCostRouteAvailability::Known(value) => Ok(Some(value)),
            ExecutionCostRouteAvailability::Unknown(ExecutionCostRouteUnknown::BudgetExhausted) => {
                Err(PlanningUnknownReason::ComputeBudgetExhausted)
            }
            ExecutionCostRouteAvailability::Unknown(reason) => {
                tracing::trace!(
                    ?reason,
                    ?kind,
                    rows = rows.len(),
                    "SLO candidate has no projected execution route"
                );
                Ok(None)
            }
        }
    }
}

impl PlanningShapeResolver for ExecutorShape<'_> {
    fn order_work(
        &self,
        _: &SchedulerSnapshot,
        work: &mut [CandidateWork],
        poll: &mut dyn FnMut() -> std::result::Result<(), PlanningUnknownReason>,
    ) -> std::result::Result<(), PlanningUnknownReason> {
        let participants = self.captured.resources.participants();
        let mut ordered = Vec::with_capacity(work.len());
        for row in work.iter() {
            poll()?;
            let index = self
                .captured
                .snapshot
                .requests
                .iter()
                .position(|request| request.key == row.key)
                .ok_or(PlanningUnknownReason::InvalidSnapshot)?;
            let authority = participants
                .get(index)
                .ok_or(PlanningUnknownReason::UnknownResourceEvidence)?
                .authority();
            ordered.push((authority, row.clone()));
        }
        ordered.sort_by_key(|(authority, _)| *authority);
        for (slot, (_, row)) in work.iter_mut().zip(ordered) {
            *slot = row;
        }
        poll()
    }
    fn resolve(
        &self,
        query: &PlanningShapeQuery<'_>,
        poll: &mut dyn FnMut() -> std::result::Result<(), PlanningUnknownReason>,
    ) -> std::result::Result<Option<CanonicalWaveCostShape>, PlanningUnknownReason> {
        Ok(match self.resolve_domain(query, poll)? {
            Some(PlanningShapeDomain::Exact(shape)) => Some(shape),
            _ => None,
        })
    }
    fn resolve_domain(
        &self,
        query: &PlanningShapeQuery<'_>,
        poll: &mut dyn FnMut() -> std::result::Result<(), PlanningUnknownReason>,
    ) -> std::result::Result<
        Option<PlanningShapeDomain<CanonicalWaveCostShape>>,
        PlanningUnknownReason,
    > {
        if query.snapshot.generation != self.captured.snapshot.generation
            || query.prior_waves.len()
                > self
                    .engine
                    .config
                    .scheduler
                    .slo
                    .planner
                    .lookahead_waves
                    .get()
        {
            return Err(PlanningUnknownReason::InvalidSnapshot);
        }
        let mut state = RouteDomain::initial(self.captured.route.initial_state());
        let mut frontiers = self.initial_frontiers(poll)?;
        for prior in query.prior_waves {
            poll()?;
            if prior.based_on_generation != self.captured.snapshot.generation
                || prior.cost_model_version != self.captured.snapshot.cost_model_version
            {
                return Err(PlanningUnknownReason::InvalidShapeEvidence);
            }
            let prepared = frontiers.prepare(&prior.work, poll)?;
            let Some((shapes, _, _, next)) =
                self.project_domain(&state, frontiers.requests(), &prepared.rows, poll)?
            else {
                return Ok(None);
            };
            if domain::cost_shapes(&shapes, poll)? != prior.execution_shape {
                return Err(PlanningUnknownReason::InvalidShapeEvidence);
            }
            frontiers.advance(prepared)?;
            state = next;
        }
        let mut work = Vec::with_capacity(query.rows.len());
        for row in query.rows {
            let action = match row.work {
                ActualRowWork::Decode { .. } => WaveAction::Decode,
                ActualRowWork::Prefill { offset, count, .. } => WaveAction::Prefill {
                    offset,
                    count: NonZeroU32::new(count).ok_or(PlanningUnknownReason::InvalidSnapshot)?,
                },
                _ => return Ok(None),
            };
            work.push(CandidateWork {
                key: row.request.key.clone(),
                action,
            });
        }
        let prepared = frontiers.prepare(&work, poll)?;
        for (expected, supplied) in prepared.rows.iter().zip(query.rows) {
            poll()?;
            if expected.work != supplied.work
                || !frontiers
                    .request(expected.index)
                    .matches_query(supplied.request)
            {
                return Err(PlanningUnknownReason::InvalidShapeEvidence);
            }
        }
        let projected = self.project_domain(&state, frontiers.requests(), &prepared.rows, poll)?;
        match projected {
            Some((shapes, _, _, _))
                if shapes.shapes().iter().all(|shape| shape.kind == query.kind) =>
            {
                Ok(Some(shapes))
            }
            Some(_) => Err(PlanningUnknownReason::InvalidShapeEvidence),
            None => Ok(None),
        }
    }
}
