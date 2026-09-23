use super::types::*;
use std::collections::HashSet;

pub(super) fn validate(
    settings: &BoundedPlannerSettings,
    snapshot: &SchedulerSnapshot,
    exact_resources: bool,
) -> Result<(), PlanningUnknownReason> {
    let config = &settings.search;
    if config.validate().is_err()
        || config.candidate_limit.get() > 256
        || config.beam_width.get() > 64
        || config.lookahead_waves.get() > 16
        || config.candidate_limit.get() * config.beam_width.get() * config.lookahead_waves.get()
            > 65_536
    {
        return Err(PlanningUnknownReason::InvalidConfiguration);
    }
    let caps = &snapshot.capabilities;
    if snapshot.requests.len() > 256
        || caps.max_wave_rows.get() > 256
        || caps.decode_batch_sizes.len() > 64
        || caps.prefill_batch_sizes.len() > 64
        || caps.prefill_chunk_sizes.len() > 64
        || caps
            .decode_batch_sizes
            .iter()
            .chain(&caps.prefill_batch_sizes)
            .any(|count| count.get() > caps.max_wave_rows.get())
        || snapshot.scope.horizon_end_ns <= snapshot.observed_at_ns
    {
        return Err(PlanningUnknownReason::InvalidSnapshot);
    }
    if !exact_resources && !snapshot.capacity.evidence_known {
        return Err(PlanningUnknownReason::UnknownResourceEvidence);
    }
    if caps
        .decode_batch_sizes
        .iter()
        .copied()
        .collect::<HashSet<_>>()
        .len()
        != caps.decode_batch_sizes.len()
        || caps
            .prefill_batch_sizes
            .iter()
            .copied()
            .collect::<HashSet<_>>()
            .len()
            != caps.prefill_batch_sizes.len()
        || caps
            .prefill_chunk_sizes
            .iter()
            .copied()
            .collect::<HashSet<_>>()
            .len()
            != caps.prefill_chunk_sizes.len()
    {
        return Err(PlanningUnknownReason::InvalidSnapshot);
    }
    if snapshot.has_unmodeled_maintenance {
        return Err(PlanningUnknownReason::UnmodeledMaintenance);
    }
    let mut ids = HashSet::new();
    let mut reference_points = 0_usize;
    let mut milestones = 0_usize;
    for request in &snapshot.requests {
        if !ids.insert(request.key.request_id.clone())
            || request.timing.ingress_at_ns > snapshot.observed_at_ns
            || request.context_tokens > snapshot.capacity.maximum_context_tokens.get()
            || request.timing.committed_tokens > request.timing.maximum_output_tokens.get()
            || request.timing.next_deadline_ns().is_none()
        {
            return Err(PlanningUnknownReason::InvalidSnapshot);
        }
        if request.readiness == RequestReadiness::Unknown {
            return Err(PlanningUnknownReason::UnknownReadiness);
        }
        let timing = &request.timing;
        match (
            timing.committed_tokens,
            timing.first_commit_at_ns,
            timing.last_commit_at_ns,
        ) {
            (0, None, None) => {}
            (n, Some(first), Some(last))
                if n > 0
                    && timing.ingress_at_ns <= first
                    && first <= last
                    && last <= snapshot.observed_at_ns
                    && (n != 1 || first == last) => {}
            _ => return Err(PlanningUnknownReason::InvalidSnapshot),
        }
        if matches!(request.phase, RequestPhaseView::Decode)
            && (timing.committed_tokens == 0 || request.context_tokens == 0)
        {
            return Err(PlanningUnknownReason::InvalidSnapshot);
        }
        if let RequestPhaseView::Prefill(progress) = &request.phase {
            reference_points = reference_points
                .checked_add(progress.reference.evidence_point_count())
                .ok_or(PlanningUnknownReason::ArithmeticOverflow)?;
            milestones = milestones
                .checked_add(progress.milestones.len())
                .ok_or(PlanningUnknownReason::ArithmeticOverflow)?;
            if reference_points > 8192
                || milestones > 4096
                || progress.offset >= progress.total_prompt_tokens.get()
                || progress.logical_high_water > progress.total_prompt_tokens.get()
                || progress.logical_high_water < progress.offset
                || progress.executable_until < progress.offset
                || progress.executable_until > progress.total_prompt_tokens.get()
                || progress.total_prompt_tokens.get()
                    > snapshot.capacity.maximum_context_tokens.get()
            {
                return Err(PlanningUnknownReason::InvalidSnapshot);
            }
            let points = &progress.reference.points;
            if !progress.reference.evaluation_is_valid()
                || progress.reference.version != snapshot.scope.reference_work_version
                || points.first()
                    != Some(&ReferenceWorkPoint {
                        prompt_tokens: 0,
                        cumulative_work_ns: 0,
                    })
                || points.last().is_none_or(|point| {
                    point.prompt_tokens != progress.total_prompt_tokens.get()
                        || point.cumulative_work_ns == 0
                })
                || points.windows(2).any(|pair| {
                    pair[0].prompt_tokens >= pair[1].prompt_tokens
                        || pair[0].cumulative_work_ns >= pair[1].cumulative_work_ns
                })
                || progress.reference.work_at(progress.offset).is_none()
                || progress
                    .reference
                    .work_at(progress.logical_high_water)
                    .is_none()
            {
                return Err(PlanningUnknownReason::MissingReferenceWork);
            }
            let first_deadline = timing
                .ingress_at_ns
                .checked_add(timing.budgets.ttft_ns.get())
                .ok_or(PlanningUnknownReason::ArithmeticOverflow)?;
            let total_work = points.last().unwrap().cumulative_work_ns;
            if progress.admitted_at_ns < timing.ingress_at_ns
                || progress.admitted_at_ns > snapshot.observed_at_ns
                || progress.admitted_at_ns >= first_deadline
                || progress.reference_work_at_admission_ns
                    > progress
                        .reference
                        .work_at(progress.logical_high_water)
                        .unwrap()
            {
                return Err(PlanningUnknownReason::InvalidSnapshot);
            }
            if progress.milestones.windows(2).any(|pair| {
                pair[0].at_ns >= pair[1].at_ns
                    || pair[0].required_reference_work_ns > pair[1].required_reference_work_ns
            }) || progress.milestones.iter().any(|milestone| {
                milestone.at_ns < timing.ingress_at_ns
                    || milestone.at_ns > first_deadline
                    || milestone.required_reference_work_ns > total_work
            }) {
                return Err(PlanningUnknownReason::InvalidSnapshot);
            }
        }
    }
    Ok(())
}

pub(super) fn prove_impossible(
    snapshot: &SchedulerSnapshot,
    now_ns: u64,
) -> Result<Option<PlanningImpossibleReason>, PlanningUnknownReason> {
    for request in &snapshot.requests {
        if request.timing.slo_failed {
            return Ok(Some(PlanningImpossibleReason::HistoricalViolation {
                key: request.key.clone(),
            }));
        }
        // The sticky runtime flag is necessary for unrecorded historical ITLs,
        // but visible endpoints independently prove these past violations.
        if let (Some(first), Some(last)) = (
            request.timing.first_commit_at_ns,
            request.timing.last_commit_at_ns,
        ) {
            let first_deadline = request
                .timing
                .ingress_at_ns
                .checked_add(request.timing.budgets.ttft_ns.get())
                .ok_or(PlanningUnknownReason::ArithmeticOverflow)?;
            let prefix_violated = request.timing.committed_tokens >= 2
                && u128::from(last - first)
                    > u128::from(request.timing.committed_tokens - 1)
                        * u128::from(request.timing.budgets.tpot_ns.get());
            if first > first_deadline || prefix_violated {
                return Ok(Some(PlanningImpossibleReason::HistoricalViolation {
                    key: request.key.clone(),
                }));
            }
        }
        if request.timing.completed() {
            continue;
        }
        let deadline_ns = request
            .timing
            .next_deadline_ns()
            .ok_or(PlanningUnknownReason::ArithmeticOverflow)?;
        if deadline_ns < now_ns {
            return Ok(Some(PlanningImpossibleReason::DeadlineAlreadyMissed {
                key: request.key.clone(),
                deadline_ns,
            }));
        }
        if let Some(bound) = request
            .optimistic_next_service
            .filter(|bound| bound.model_version == snapshot.cost_model_version)
        {
            let earliest_completion_ns = now_ns
                .checked_add(bound.duration_ns)
                .ok_or(PlanningUnknownReason::ArithmeticOverflow)?;
            if earliest_completion_ns > deadline_ns {
                return Ok(Some(
                    PlanningImpossibleReason::CertifiedOptimisticLowerBound {
                        key: request.key.clone(),
                        deadline_ns,
                        earliest_completion_ns,
                    },
                ));
            }
        }
    }
    Ok(None)
}
