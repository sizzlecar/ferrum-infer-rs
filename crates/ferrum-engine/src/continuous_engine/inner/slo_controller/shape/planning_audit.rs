//! Bounded paths over the original physical projector. No simulated service
//! times, obligation relaxation, publication or model fitting occur here.
use super::*;
use crate::continuous_engine::inner::calibration::*;
use ferrum_scheduler::implementations::continuous::{
    cost_model::structured_v2::{StructuredQueryV2, StructuredUnknownV2},
    slo_planner::{PlanningCostEvidenceRequirement, PlanningStructure},
};

impl EngineInner {
    pub(in crate::continuous_engine::inner) fn audit_required_owners_v2(
        &self,
        frontiers: &[CalibrationFrontier],
        plan: &RequiredFutureAuditPlanV2,
    ) -> Result<RequiredFutureAuditReportV2> {
        plan.validate(frontiers.len())?;
        let mut report = RequiredFutureAuditReportV2::new(plan);
        report.frontier_table = frontiers
            .iter()
            .map(|f| RequiredFutureAuditFrontierV2 {
                request_id: f.request_id.clone(),
                owner_incarnation: f.owner.get(),
                work_generation: f.generation.get(),
                generated_tokens: f.generated,
                kv_tokens: f.kv_tokens,
                prefill_progress: f.prefill,
                input: f.request_evidence,
            })
            .collect();
        let budget = ControllerBudget::new(
            slo_clock_now(),
            std::time::Duration::from_millis(plan.limits.budget_ms.get()),
        )?;
        // Check the complete live population even when no seed was imported.
        // A stale/foreign frontier must not be hidden behind CostUnavailable.
        {
            let sequences = self.sequences.try_read().ok_or_else(|| {
                FerrumError::resource_exhausted("future-owner audit sequence snapshot is busy")
            })?;
            if sequences.len() != frontiers.len() {
                return Err(FerrumError::invalid_request(
                    "future-owner audit omits live owners",
                ));
            }
            for (index, frontier) in frontiers.iter().enumerate() {
                if !budget.poll() {
                    report.truncated = true;
                    report.unavailable = Some(planning_failure(
                        PlanningUnknownReason::ComputeBudgetExhausted,
                    ));
                    return Ok(report);
                }
                let sequence = sequences.get(&frontier.request_id).ok_or_else(|| {
                    FerrumError::invalid_request("future-owner audit owner is no longer live")
                })?;
                if frontiers[..index]
                    .iter()
                    .any(|old| old.request_id == frontier.request_id)
                    || sequence.cost_frontier.is_none_or(|f| {
                        f.owner_incarnation != frontier.owner
                            || f.work_generation != frontier.generation
                    })
                    || sequence.generated_tokens.len() != frontier.generated
                    || (!sequence.prefill_complete).then_some((
                        sequence.prefill_tokens_processed,
                        sequence.prefill_context_len(),
                    )) != frontier.prefill
                    || sequence
                        .model_kv
                        .as_ref()
                        .map_or(0, |kv| kv.handle().num_tokens())
                        != frontier.kv_tokens
                {
                    return Err(FerrumError::invalid_request(
                        "future-owner audit frontier changed",
                    ));
                }
            }
        }
        let Some(runtime) = &self.cost_runtime else {
            report.unavailable = Some(snapshot_failure("cost_unavailable", None));
            return Ok(report);
        };
        // Preserve the real imported model object, not a synthetic all-Unknown
        // implementation. Capture below must have read this same immutable Arc.
        let Some(Some(model)) = runtime.try_snapshot() else {
            report.unavailable = Some(snapshot_failure("cost_unavailable", None));
            return Ok(report);
        };
        if model.evidence_requirement() != PlanningCostEvidenceRequirement::StructuredV2 {
            report.unavailable = Some(snapshot_failure("structured_v2_seed_required", None));
            return Ok(report);
        }
        let mut hint = ferrum_interfaces::BatchHint::simple(self.config.batching.max_batch_size);
        hint.max_tokens = self.config.batching.max_num_batched_tokens;
        let mut route_unknown = None;
        let captured = match self.capture_slo_controller_snapshot_with_diagnostic(
            &hint,
            Arc::clone(&budget),
            &mut route_unknown,
        ) {
            Ok(captured) => captured,
            Err(error) => {
                let reason = snapshot_failure(error.reason, route_unknown);
                report.truncated = is_limit(&reason);
                report.unavailable = Some(reason);
                return Ok(report);
            }
        };
        let model_identity: Arc<dyn PlanningCostModel + Send + Sync> = model.clone();
        if !Arc::ptr_eq(&model_identity, &captured.model) {
            report.unavailable = Some(snapshot_failure("cost_snapshot_changed", None));
            return Ok(report);
        }
        let mut poll = || {
            if budget.poll() {
                Ok(())
            } else {
                Err(PlanningUnknownReason::ComputeBudgetExhausted)
            }
        };
        let slots = match bind_frontiers(frontiers, &captured, &mut poll) {
            Ok(slots) => slots,
            Err(reason) => {
                let reason = planning_failure(reason);
                report.truncated = is_limit(&reason);
                report.unavailable = Some(reason);
                return Ok(report);
            }
        };
        report.snapshot_generation = Some(captured.snapshot.generation);
        report.observed_at_ns = Some(captured.snapshot.observed_at_ns);
        report.model_version = Some(model.model_version());
        let fp = &captured.snapshot.fingerprint;
        report.execution_fingerprint = Some([
            fp.model_weights,
            fp.numerical_policy,
            fp.device_runtime,
            fp.execution_config,
        ]);
        let shape = ExecutorShape {
            engine: self,
            captured: &captured,
        };
        Ok(
            shape.audit_paths(plan, &slots, report, &mut poll, &mut |query| {
                let cost = match runtime.clock.now_ns() {
                    Some(now) => match model.audit_structured_query_v2(&query, now) {
                        Ok(value) => RequiredFutureAuditCostV2::KnownAtRead {
                            local_now_ns: now,
                            planning_ns: value.planning_ns,
                            valid_for_ns: value.valid_for_ns,
                        },
                        Err(reason) => RequiredFutureAuditCostV2::Unknown {
                            local_now_ns: Some(now),
                            reason: format!("{reason:?}"),
                        },
                    },
                    None => RequiredFutureAuditCostV2::Unknown {
                        local_now_ns: None,
                        reason: "Clock".into(),
                    },
                };
                cost
            }),
        )
    }
}

impl ExecutorShape<'_> {
    pub(in crate::continuous_engine::inner::slo_controller) fn audit_paths(
        &self,
        plan: &RequiredFutureAuditPlanV2,
        slots: &[usize],
        mut report: RequiredFutureAuditReportV2,
        mut poll: &mut dyn FnMut() -> std::result::Result<(), PlanningUnknownReason>,
        lookup: &mut dyn FnMut(&StructuredQueryV2) -> RequiredFutureAuditCostV2,
    ) -> RequiredFutureAuditReportV2 {
        for (path_index, path) in plan.paths.iter().enumerate() {
            let mut structure = PlanningStructure::new(&self.captured.snapshot);
            let mut projected = match self.initial_frontiers(&mut poll) {
                Ok(value) => value,
                Err(reason) => {
                    stop(&mut report, path_index, planning_failure(reason));
                    continue;
                }
            };
            let mut route = RouteDomain::initial(self.captured.route.initial_state());
            for (wave_index, declared) in path.waves.iter().enumerate() {
                if wave_index
                    >= self
                        .engine
                        .config
                        .scheduler
                        .slo
                        .planner
                        .lookahead_waves
                        .get()
                {
                    stop(&mut report, path_index, limit("production_lookahead"));
                    break;
                }
                let prepared = (|| {
                    poll()?;
                    let mut work = Vec::with_capacity(declared.len());
                    for row in declared {
                        poll()?;
                        let current = projected.request(slots[row.frontier_index]);
                        let action = match (&row.action, &current.phase) {
                            (RequiredFutureAuditActionV2::Decode, _) => WaveAction::Decode,
                            (
                                RequiredFutureAuditActionV2::Prefill { count },
                                RequestPhaseView::Prefill(p),
                            ) => WaveAction::Prefill {
                                offset: p.offset,
                                count: *count,
                            },
                            _ => return Err(PlanningUnknownReason::InvalidShapeEvidence),
                        };
                        work.push(CandidateWork {
                            key: current.key.clone(),
                            action,
                        });
                    }
                    self.order_work(&self.captured.snapshot, &mut work, &mut poll)?;
                    // Both representations must name the same untimed logical
                    // frontier. The shared scheduler rules additionally retain
                    // readiness, work-policy limits and no-drain output credit.
                    for (request, logical) in projected.requests().iter().zip(structure.requests())
                    {
                        poll()?;
                        if !request.matches_query(logical) {
                            return Err(PlanningUnknownReason::InvalidShapeEvidence);
                        }
                    }
                    let next_structure = structure.advance(&work, &mut poll)?;
                    Ok((projected.prepare(&work, &mut poll)?, next_structure))
                })();
                let (prepared, next_structure) = match prepared {
                    Ok(value) => value,
                    Err(reason) => {
                        stop(&mut report, path_index, planning_failure(reason));
                        break;
                    }
                };
                let mut unavailable = None;
                let domain = self.project_domain_with_diagnostic(
                    &route,
                    projected.requests(),
                    &prepared.rows,
                    &mut poll,
                    &mut unavailable,
                );
                let (canonical, selected, forecasts, successor) = match domain {
                    Ok(Some(value)) => value,
                    Ok(None) => {
                        stop(
                            &mut report,
                            path_index,
                            unavailable.map_or(
                                RequiredFutureAuditFailureV2::UnsupportedHostOrProjection,
                                route_failure,
                            ),
                        );
                        break;
                    }
                    Err(reason) => {
                        stop(&mut report, path_index, planning_failure(reason));
                        break;
                    }
                };
                let shapes = canonical.shapes();
                if wave_index == 0 && canonical.exact().is_none() {
                    stop(
                        &mut report,
                        path_index,
                        planning_failure(PlanningUnknownReason::InvalidShapeEvidence),
                    );
                    break;
                }
                if report
                    .queries
                    .len()
                    .checked_add(shapes.len())
                    .is_none_or(|n| n > plan.limits.maximum_queries.get())
                {
                    stop(&mut report, path_index, limit("queries"));
                    break;
                }
                let mut next_queries = Vec::with_capacity(shapes.len());
                let mut next_coordinates = 0usize;
                let mut failed = None;
                for (alternative_index, exact) in shapes.iter().enumerate() {
                    if let Err(reason) = poll() {
                        failed = Some(planning_failure(reason));
                        break;
                    }
                    let query = (|| {
                        let selected = selected
                            .as_ref()
                            .filter(|s| s.shapes().len() == shapes.len())
                            .and_then(|s| s.shapes().get(alternative_index))
                            .ok_or(StructuredUnknownV2::MissingEvidence)?;
                        let forecast = forecasts
                            .as_ref()
                            .filter(|f| f.shapes().len() == shapes.len())
                            .and_then(|f| f.shapes().get(alternative_index))
                            .ok_or(StructuredUnknownV2::MissingEvidence)?;
                        let recipe = selected
                            .structured_capture()
                            .ok_or(StructuredUnknownV2::MissingEvidence)?
                            .map_err(|_| StructuredUnknownV2::MissingEvidence)?;
                        StructuredQueryV2::from_future(exact, selected, recipe, forecast)
                    })();
                    let mut entry = RequiredFutureAuditQueryV2 {
                        path_index,
                        wave_index,
                        alternative_index,
                        exact_first_wave: wave_index == 0,
                        demand: None,
                        input_unknown: None,
                        cost: RequiredFutureAuditCostV2::NotQueried {
                            reason: "input_unavailable",
                        },
                    };
                    match query.and_then(|query| Ok((query.required_coverage()?, query))) {
                        Ok((demand, query)) => {
                            let coordinates = demand
                                .regression_axes
                                .len()
                                .checked_add(demand.joint_support_coordinates.len());
                            let total = coordinates.and_then(|n| next_coordinates.checked_add(n));
                            if total
                                .and_then(|n| report.retained_coordinates.checked_add(n))
                                .is_none_or(|n| n > plan.limits.maximum_coordinates.get())
                            {
                                failed = Some(limit("coordinates"));
                                break;
                            }
                            next_coordinates = total.expect("checked above");
                            // Read the original local clock for each lookup. A
                            // future logical wave receives no invented timestamp.
                            entry.cost = lookup(&query);
                            entry.demand = Some(demand);
                        }
                        Err(reason) => entry.input_unknown = Some(format!("{reason:?}")),
                    }
                    next_queries.push(entry);
                }
                if let Some(reason) = failed {
                    stop(&mut report, path_index, reason);
                    break;
                }
                if let Err(reason) = poll().and_then(|()| projected.advance(prepared)) {
                    stop(&mut report, path_index, planning_failure(reason));
                    break;
                }
                // Only a complete physical domain advances the logical path.
                // Cost Unknown is diagnostic and cannot remove an alternative.
                route = successor;
                structure = next_structure;
                report.retained_coordinates += next_coordinates;
                report.queries.extend(next_queries);
                report.paths[path_index].projected_waves += 1;
            }
        }
        report.completed_all_declared_paths = !report.truncated
            && report.unavailable.is_none()
            && report
                .paths
                .iter()
                .all(|p| p.stopped.is_none() && p.projected_waves == p.declared_waves)
            && report.queries.iter().all(|q| q.demand.is_some());
        report
    }
}

fn bind_frontiers(
    supplied: &[CalibrationFrontier],
    captured: &ControllerSnapshot,
    poll: &mut dyn FnMut() -> std::result::Result<(), PlanningUnknownReason>,
) -> std::result::Result<Vec<usize>, PlanningUnknownReason> {
    if supplied.len() != captured.fences.len() || supplied.len() != captured.snapshot.requests.len()
    {
        return Err(PlanningUnknownReason::InvalidSnapshot);
    }
    let mut slots = Vec::with_capacity(supplied.len());
    for frontier in supplied {
        poll()?;
        let index = captured
            .fences
            .iter()
            .position(|f| f.key.request_id == frontier.request_id)
            .ok_or(PlanningUnknownReason::InvalidSnapshot)?;
        let f = &captured.fences[index];
        if slots.contains(&index)
            || f.incarnation != frontier.owner.get()
            || f.generation != frontier.generation.get()
            || f.generated != frontier.generated
            || f.context != frontier.kv_tokens
            || frontier.prefill
                != (!f.prefill_complete).then_some((f.prefill_tokens_processed, f.prefill_total))
        {
            return Err(PlanningUnknownReason::InvalidSnapshot);
        }
        slots.push(index);
    }
    Ok(slots)
}
fn snapshot_failure(
    reason: &'static str,
    route: Option<ExecutionCostRouteUnknown>,
) -> RequiredFutureAuditFailureV2 {
    if matches!(
        reason,
        "compute_budget_exhausted"
            | "reference_point_budget"
            | "reference_milestone_budget"
            | "reference_candidate_budget"
    ) {
        return limit(reason);
    }
    if let Some(route) = route.filter(route_is_limit) {
        return route_failure(route);
    }
    RequiredFutureAuditFailureV2::Snapshot {
        reason,
        route_reason: route.map(|r| format!("{r:?}")),
    }
}
fn planning_failure(reason: PlanningUnknownReason) -> RequiredFutureAuditFailureV2 {
    match reason {
        PlanningUnknownReason::ComputeBudgetExhausted => return limit("compute_budget_exhausted"),
        PlanningUnknownReason::ShapeCapacity => return limit("shape_capacity"),
        _ => {}
    }
    RequiredFutureAuditFailureV2::Planning {
        reason: format!("{reason:?}"),
    }
}
fn route_is_limit(reason: &ExecutionCostRouteUnknown) -> bool {
    use ferrum_interfaces::vnext::ResourcePlanningUnknown;
    matches!(
        reason,
        ExecutionCostRouteUnknown::BudgetExhausted
            | ExecutionCostRouteUnknown::Resource(
                ResourcePlanningUnknown::BudgetExhausted | ResourcePlanningUnknown::LimitExceeded
            )
    )
}
fn route_failure(reason: ExecutionCostRouteUnknown) -> RequiredFutureAuditFailureV2 {
    use ferrum_interfaces::vnext::ResourcePlanningUnknown;
    match reason {
        ExecutionCostRouteUnknown::BudgetExhausted => limit("route_budget_exhausted"),
        ExecutionCostRouteUnknown::Resource(ResourcePlanningUnknown::BudgetExhausted) => {
            limit("resource_budget_exhausted")
        }
        ExecutionCostRouteUnknown::Resource(ResourcePlanningUnknown::LimitExceeded) => {
            limit("resource_projection_limit")
        }
        _ => RequiredFutureAuditFailureV2::Route {
            reason: format!("{reason:?}"),
        },
    }
}
fn is_limit(reason: &RequiredFutureAuditFailureV2) -> bool {
    matches!(reason, RequiredFutureAuditFailureV2::Limit { .. })
}
fn limit(resource: &'static str) -> RequiredFutureAuditFailureV2 {
    RequiredFutureAuditFailureV2::Limit { resource }
}
fn stop(
    report: &mut RequiredFutureAuditReportV2,
    path: usize,
    failure: RequiredFutureAuditFailureV2,
) {
    report.truncated |= is_limit(&failure);
    report.paths[path].stopped = Some(failure);
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferrum_interfaces::vnext::ResourcePlanningUnknown;

    #[test]
    fn required_future_audit_classifies_typed_bounds_without_hiding_capacity_failure() {
        for reason in [
            ExecutionCostRouteUnknown::BudgetExhausted,
            ExecutionCostRouteUnknown::Resource(ResourcePlanningUnknown::BudgetExhausted),
            ExecutionCostRouteUnknown::Resource(ResourcePlanningUnknown::LimitExceeded),
        ] {
            assert!(is_limit(&route_failure(reason)));
            assert!(is_limit(&snapshot_failure(
                "route_unavailable",
                Some(reason)
            )));
        }
        for reason in [
            ExecutionCostRouteUnknown::Capacity,
            ExecutionCostRouteUnknown::Resource(ResourcePlanningUnknown::PhysicalCapacity),
            ExecutionCostRouteUnknown::Resource(ResourcePlanningUnknown::LogicalCapacity),
            ExecutionCostRouteUnknown::ExecutionPolicy,
        ] {
            assert!(!is_limit(&route_failure(reason)));
            assert!(!is_limit(&snapshot_failure(
                "route_unavailable",
                Some(reason)
            )));
        }
        for reason in [
            "reference_point_budget",
            "reference_milestone_budget",
            "reference_candidate_budget",
            "compute_budget_exhausted",
        ] {
            assert!(is_limit(&snapshot_failure(reason, None)));
        }
        assert!(!is_limit(&planning_failure(
            PlanningUnknownReason::OutputOrResourceBlocked
        )));
    }
}
