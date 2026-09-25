//! Finite whole-wave alternatives. Unknown text is an empirical latent input,
//! never a guessed exact UTF-8 state. Every reachable physical/residency state
//! survives until the configured bound; exhaustion cannot select a subset.
use super::*;
use ferrum_interfaces::{
    execution_cost::{HostCostFeaturesV1, StatisticalWaveEvidenceV1},
    model_executor::LogitsReturnPolicy,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum FutureHostMode {
    Exact,
    Greedy,
    FullLogits,
}

pub(super) struct RouteDomain {
    states: Vec<ExecutionCostRouteState>,
    empirical: bool,
}
impl RouteDomain {
    pub(super) fn initial(state: ExecutionCostRouteState) -> Self {
        Self {
            states: vec![state],
            empirical: false,
        }
    }
}

/// Outer None is unsupported content; inner None is the unchanged exact-only
/// host identity used by legacy models. Projected values are hypotheses inside
/// an explicit domain, and can never reach the actual first-wave guard.
pub(super) fn row_host(
    original: Option<HostCostFeaturesV1>,
    request: &frontier::ProjectedRequest,
    mode: FutureHostMode,
) -> std::result::Result<Option<Option<HostCostFeaturesV1>>, PlanningUnknownReason> {
    if !request.host_content_changed {
        return Ok(Some(original));
    }
    let Some(mut host) = original.filter(|host| host.supports_empirical_plain_text_content())
    else {
        return Ok(None);
    };
    if mode == FutureHostMode::Exact
        || request.generated == 0
        || u64::from(request.generated) <= host.state.generated_tokens_before
        || u64::from(request.maximum_output_tokens.get()) != host.state.maximum_output_tokens
    {
        return Ok(None);
    }
    host.state.generated_tokens_before = u64::from(request.generated);
    host.state.sampling_history_tokens = u64::from(request.generated);
    // The completion contract remains Satisfied for this installed capability.
    // The two alternatives cover possible routing; this does not assert that
    // a particular generated prefix is, or is not, valid UTF-8.
    host.state.pending_decoded_utf8 = mode == FutureHostMode::FullLogits;
    Ok(Some(Some(host)))
}

impl ExecutorShape<'_> {
    fn future_modes(
        &self,
        frontiers: &[ProjectedRequest],
        prepared: &[PreparedRow],
        poll: &mut dyn FnMut() -> std::result::Result<(), PlanningUnknownReason>,
    ) -> std::result::Result<Option<Vec<FutureHostMode>>, PlanningUnknownReason> {
        let mut changed = false;
        let mut forces_full_logits = false;
        let mut can_change_product = false;
        for selected in prepared {
            poll()?;
            let request = &frontiers[selected.index];
            let fence = &self.captured.fences[selected.index];
            if request.host_content_changed {
                changed = true;
                if !fence
                    .host_features
                    .is_some_and(|host| host.supports_empirical_plain_text_content())
                {
                    return Ok(None);
                }
            }
            match selected.work {
                ActualRowWork::Decode { .. } if request.host_content_changed => {
                    if fence.future_greedy_policy.is_none() {
                        return Ok(None);
                    }
                    can_change_product = true;
                }
                ActualRowWork::Decode { .. } => {
                    forces_full_logits |=
                        matches!(fence.logits_policy, LogitsReturnPolicy::FullLogits);
                }
                ActualRowWork::Prefill {
                    offset,
                    count,
                    total_prompt_tokens,
                } => {
                    forces_full_logits |= offset.checked_add(count) == Some(total_prompt_tokens);
                }
                _ => return Ok(None),
            }
        }
        if !changed {
            return Ok(Some(vec![FutureHostMode::Exact]));
        }
        if !self.captured.model.supports_empirical_host_content() {
            return Ok(None);
        }
        // Product output mode is a whole-wave decision. No 2^owner expansion:
        // a fixed FullLogits peer/final prefill already determines that product.
        Ok(Some(if forces_full_logits || !can_change_product {
            vec![FutureHostMode::FullLogits]
        } else {
            vec![FutureHostMode::Greedy, FutureHostMode::FullLogits]
        }))
    }

    pub(super) fn project_domain(
        &self,
        state: &RouteDomain,
        frontiers: &[ProjectedRequest],
        prepared: &[PreparedRow],
        poll: &mut dyn FnMut() -> std::result::Result<(), PlanningUnknownReason>,
    ) -> std::result::Result<
        Option<(
            PlanningShapeDomain<CanonicalWaveCostShape>,
            Option<PlanningShapeDomain<StatisticalWaveEvidenceV1>>,
            RouteDomain,
        )>,
        PlanningUnknownReason,
    > {
        let Some(modes) = self.future_modes(frontiers, prepared, poll)? else {
            return Ok(None);
        };
        let settings = &self.engine.config.scheduler.slo.planner;
        let count = state
            .states
            .len()
            .checked_mul(modes.len())
            .ok_or(PlanningUnknownReason::ArithmeticOverflow)?;
        if count == 0
            || count > settings.max_route_states.get()
            || count > settings.max_shape_alternatives.get()
        {
            return Err(PlanningUnknownReason::ShapeCapacity);
        }
        let empirical = state.empirical || modes != [FutureHostMode::Exact];
        if empirical && !self.captured.model.supports_empirical_host_content() {
            return Ok(None);
        }
        let collect_statistics = self.captured.model.requires_statistical_evidence()
            || self.captured.route.structured_capture_enabled();
        let mut selected = Vec::new();
        let mut statistics_complete = collect_statistics;
        if collect_statistics {
            selected
                .try_reserve_exact(count)
                .map_err(|_| PlanningUnknownReason::ShapeCapacity)?;
        }
        let mut shapes = Vec::new();
        let mut states = Vec::new();
        shapes
            .try_reserve_exact(count)
            .map_err(|_| PlanningUnknownReason::ShapeCapacity)?;
        states
            .try_reserve_exact(count)
            .map_err(|_| PlanningUnknownReason::ShapeCapacity)?;
        // Keep all states rather than merging by hashes or ignoring allocator
        // and mask differences. H3 needs at most 8 states regardless of width.
        for previous in &state.states {
            for &mode in &modes {
                poll()?;
                let Some(projected) = self.project(previous, frontiers, prepared, mode, poll)?
                else {
                    return Ok(None);
                };
                if empirical && projected.shape.host_content_features.is_none() {
                    return Ok(None);
                }
                if collect_statistics {
                    match projected.statistical_evidence {
                        Some(evidence) => selected.push(evidence),
                        None => statistics_complete = false,
                    }
                }
                shapes.push(projected.shape);
                states.push(projected.state);
            }
        }
        let domain = if empirical {
            PlanningShapeDomain::HostContentAlternatives(shapes)
        } else {
            PlanningShapeDomain::Exact(
                shapes
                    .pop()
                    .ok_or(PlanningUnknownReason::InvalidShapeEvidence)?,
            )
        };
        let statistics = if statistics_complete {
            Some(if empirical {
                PlanningShapeDomain::HostContentAlternatives(selected)
            } else {
                PlanningShapeDomain::Exact(
                    selected
                        .pop()
                        .ok_or(PlanningUnknownReason::InvalidShapeEvidence)?,
                )
            })
        } else {
            None
        };
        Ok(Some((
            domain,
            statistics,
            RouteDomain { states, empirical },
        )))
    }
}

pub(super) fn cost_shapes(
    domain: &PlanningShapeDomain<CanonicalWaveCostShape>,
    poll: &mut dyn FnMut() -> std::result::Result<(), PlanningUnknownReason>,
) -> std::result::Result<
    PlanningShapeDomain<
        ferrum_scheduler::implementations::continuous::cost_model::WaveExecutionShape,
    >,
    PlanningUnknownReason,
> {
    match domain {
        PlanningShapeDomain::Exact(shape) => {
            poll()?;
            Ok(PlanningShapeDomain::Exact(canonical_cost_shape(shape)?))
        }
        PlanningShapeDomain::HostContentAlternatives(shapes) => {
            let mut output = Vec::new();
            output
                .try_reserve_exact(shapes.len())
                .map_err(|_| PlanningUnknownReason::ShapeCapacity)?;
            for shape in shapes {
                poll()?;
                output.push(canonical_cost_shape(shape)?);
            }
            Ok(PlanningShapeDomain::HostContentAlternatives(output))
        }
    }
}
