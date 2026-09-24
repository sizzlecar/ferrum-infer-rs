use super::{obligations::PlanningObligationSet, shape, types::*};
use std::{cmp::Reverse, num::NonZeroU32};

pub(super) struct LogicalCandidates {
    required: Option<RequestWorkKey>,
    pub work: Vec<Vec<CandidateWork>>,
    pub truncated: bool,
    pub attempts: usize,
}

pub(super) fn logical_candidates(
    snapshot: &SchedulerSnapshot,
    requests: &[RequestSchedulingView],
    now_ns: u64,
    limit: usize,
    raw_limit: usize,
    observed_attempts: &mut usize,
    protection: Option<&PlanningObligationSet>,
    resolver: &dyn PlanningShapeResolver,
    poll_budget: &mut dyn FnMut() -> Result<(), PlanningUnknownReason>,
) -> Result<LogicalCandidates, PlanningUnknownReason> {
    poll_budget()?;
    let mut decoders: Vec<_> = requests
        .iter()
        .enumerate()
        .filter(|(_, request)| {
            runnable(request) && matches!(request.phase, RequestPhaseView::Decode)
        })
        .map(|(index, _)| index)
        .collect();
    let mut prefills: Vec<_> = requests
        .iter()
        .enumerate()
        .filter(|(_, request)| {
            runnable(request) && matches!(request.phase, RequestPhaseView::Prefill(_))
        })
        .map(|(index, _)| index)
        .collect();
    // An irreversibly expired deadline is not an earlier rescuable deadline.
    // Keep the frozen forward-protection scope ahead of completion-only work
    // in the urgency family. The fairness family still includes every owner,
    // and due service below overrides this ordering. Sticky historical misses
    // with a live next obligation remain Protected and keep ordinary urgency.
    let priority = |index: usize| {
        (
            protection.is_some_and(|scope| !scope.protects(index)),
            urgency(&requests[index], now_ns),
        )
    };
    decoders.sort_by_key(|&index| priority(index));
    prefills.sort_by_key(|&index| priority(index));
    let required = protection.and_then(|scope| scope.required_service(requests));
    // Mandatory due service is enumerated before optional urgency families;
    // truncating K must not hide every legal wave containing the due owner.
    if let Some(required) = required {
        for indices in [&mut decoders, &mut prefills] {
            if let Some(position) = indices.iter().position(|index| *index == required) {
                indices.rotate_left(position);
            }
        }
    }
    let caps = &snapshot.capabilities;
    let mut result = LogicalCandidates {
        required: required.map(|index| requests[index].key.clone()),
        work: Vec::new(),
        truncated: false,
        attempts: 0,
    };
    // K bounds admitted candidates; 8*K independently bounds ALL raw attempts,
    // including impossible and duplicate shapes. The absolute ceiling is 2048.
    let attempt_limit = limit.min(256).saturating_mul(8).min(raw_limit);
    macro_rules! attempt {
        () => {
            poll_budget()?;
            if result.attempts >= attempt_limit {
                result.truncated = true;
                return Ok(result);
            }
            result.attempts += 1;
            *observed_attempts += 1;
        };
    }
    // Try the complete ready decode cohort first, if it is a declared legal
    // width. Only this one width precedes the interleaved first-token family.
    // Physical resolution stays lazy: a missing route does not consume K.
    if let Some(size) = caps
        .decode_batch_sizes
        .iter()
        .filter(|size| size.get() <= decoders.len() && size.get() <= caps.max_wave_rows.get())
        .max_by_key(|size| size.get())
    {
        attempt!();
        push_decode(
            snapshot,
            requests,
            &decoders,
            size.get(),
            &mut result,
            resolver,
            poll_budget,
        )?;
    }
    // Interleave action families so a small K does not enumerate every decode
    // size before considering any first-token work.
    let rounds = caps
        .decode_batch_sizes
        .len()
        .max(caps.prefill_chunk_sizes.len());
    for round in 0..rounds {
        if let Some(size) = caps.decode_batch_sizes.get(round) {
            attempt!();
            push_decode(
                snapshot,
                requests,
                &decoders,
                size.get(),
                &mut result,
                resolver,
                poll_budget,
            )?;
        }
        if let Some(&chunk) = caps.prefill_chunk_sizes.get(round) {
            for &size in &caps.prefill_batch_sizes {
                attempt!();
                let prefill_work =
                    prefill_work(requests, &prefills, size.get(), chunk, caps, poll_budget)?;
                if let Some(work) = &prefill_work {
                    push(
                        snapshot,
                        requests,
                        work.clone(),
                        &mut result,
                        resolver,
                        poll_budget,
                    )?;
                }
                if caps.native_mixed {
                    for &decode_size in &caps.decode_batch_sizes {
                        attempt!();
                        if let (Some(mut work), Some(prefill)) = (
                            decode_work(requests, &decoders, decode_size.get()),
                            &prefill_work,
                        ) {
                            work.extend(prefill.iter().cloned());
                            push(snapshot, requests, work, &mut result, resolver, poll_budget)?;
                        }
                        if result.truncated {
                            return Ok(result);
                        }
                    }
                }
                if result.truncated {
                    return Ok(result);
                }
            }
        }
        if result.truncated {
            return Ok(result);
        }
    }
    // Fairness supplement: a stable rotation, not nondeterministic map order.
    for indices in [&mut decoders, &mut prefills] {
        if let Some((position, _)) = indices
            .iter()
            .enumerate()
            .min_by_key(|(_, index)| requests[**index].fairness_rank)
        {
            indices.rotate_left(position);
        }
    }
    for &size in &caps.decode_batch_sizes {
        attempt!();
        push_decode(
            snapshot,
            requests,
            &decoders,
            size.get(),
            &mut result,
            resolver,
            poll_budget,
        )?;
        if result.truncated {
            return Ok(result);
        }
    }
    for &chunk in &caps.prefill_chunk_sizes {
        for &size in &caps.prefill_batch_sizes {
            attempt!();
            if let Some(work) =
                prefill_work(requests, &prefills, size.get(), chunk, caps, poll_budget)?
            {
                push(snapshot, requests, work, &mut result, resolver, poll_budget)?;
            }
            if result.truncated {
                return Ok(result);
            }
        }
    }
    Ok(result)
}

fn runnable(request: &RequestSchedulingView) -> bool {
    request.readiness == RequestReadiness::Ready && !request.timing.completed()
}

fn urgency(request: &RequestSchedulingView, now_ns: u64) -> (i128, Reverse<u64>, u64, [u8; 16]) {
    let debt = match &request.phase {
        RequestPhaseView::Prefill(progress) => progress
            .milestones
            .iter()
            .filter(|milestone| milestone.at_ns <= now_ns)
            .map(|milestone| milestone.required_reference_work_ns)
            .max()
            .unwrap_or(0)
            .saturating_sub(
                progress
                    .reference
                    .work_at(progress.logical_high_water)
                    .unwrap_or(0),
            ),
        RequestPhaseView::Decode => 0,
    };
    (
        i128::from(request.timing.next_deadline_ns().unwrap_or(0))
            - i128::from(request.ranking_service_cost_ns.unwrap_or(0)),
        Reverse(debt),
        request.fairness_rank,
        *request.key.request_id.0.as_bytes(),
    )
}

fn decode_work(
    requests: &[RequestSchedulingView],
    indices: &[usize],
    size: usize,
) -> Option<Vec<CandidateWork>> {
    if size > indices.len() {
        return None;
    }
    Some(
        indices[..size]
            .iter()
            .map(|&index| CandidateWork {
                key: requests[index].key.clone(),
                action: WaveAction::Decode,
            })
            .collect(),
    )
}

fn push_decode(
    snapshot: &SchedulerSnapshot,
    requests: &[RequestSchedulingView],
    indices: &[usize],
    size: usize,
    output: &mut LogicalCandidates,
    resolver: &dyn PlanningShapeResolver,
    poll_budget: &mut dyn FnMut() -> Result<(), PlanningUnknownReason>,
) -> Result<(), PlanningUnknownReason> {
    if let Some(work) = decode_work(requests, indices, size) {
        push(snapshot, requests, work, output, resolver, poll_budget)?;
    }
    Ok(())
}

fn prefill_work(
    requests: &[RequestSchedulingView],
    indices: &[usize],
    size: usize,
    chunk: NonZeroU32,
    caps: &BackendPlanningCapabilities,
    poll_budget: &mut dyn FnMut() -> Result<(), PlanningUnknownReason>,
) -> Result<Option<Vec<CandidateWork>>, PlanningUnknownReason> {
    let mut work = Vec::new();
    for &index in indices {
        poll_budget()?;
        let request = &requests[index];
        let RequestPhaseView::Prefill(progress) = &request.phase else {
            continue;
        };
        let remaining = progress.total_prompt_tokens.get() - progress.offset;
        let count = if remaining < chunk.get() && caps.allow_final_short_chunk {
            remaining
        } else {
            chunk.get()
        };
        let Some(end) = progress.offset.checked_add(count) else {
            continue;
        };
        if end > progress.executable_until
            || end > progress.total_prompt_tokens.get()
            || progress.offset % caps.prefill_alignment.get() != 0
            || (count % caps.prefill_alignment.get() != 0
                && !(caps.allow_final_short_chunk && end == progress.total_prompt_tokens.get()))
            || progress.reference.work_at(end).is_none()
        {
            continue;
        }
        work.push(CandidateWork {
            key: request.key.clone(),
            action: WaveAction::Prefill {
                offset: progress.offset,
                count: NonZeroU32::new(count).ok_or(PlanningUnknownReason::InvalidSnapshot)?,
            },
        });
        if work.len() == size {
            return Ok(Some(work));
        }
    }
    Ok(None)
}

fn push(
    snapshot: &SchedulerSnapshot,
    _requests: &[RequestSchedulingView],
    mut work: Vec<CandidateWork>,
    output: &mut LogicalCandidates,
    resolver: &dyn PlanningShapeResolver,
    poll_budget: &mut dyn FnMut() -> Result<(), PlanningUnknownReason>,
) -> Result<(), PlanningUnknownReason> {
    poll_budget()?;
    if output
        .required
        .as_ref()
        .is_some_and(|key| !work.iter().any(|row| &row.key == key))
        || work.is_empty()
        || work.len() > snapshot.capabilities.max_wave_rows.get()
    {
        return Ok(());
    }
    shape::order_work(snapshot, &mut work, resolver, poll_budget)?;
    if output.work.contains(&work) {
        return Ok(());
    }
    output.work.push(work);
    Ok(())
}

#[cfg(test)]
pub(super) struct CandidateSet {
    pub waves: Vec<WaveCandidate>,
    pub truncated: bool,
    pub attempts: usize,
    pub shape_unknown: usize,
}

#[cfg(test)]
pub(super) fn enumerate(
    snapshot: &SchedulerSnapshot,
    requests: &[RequestSchedulingView],
    now_ns: u64,
    limit: usize,
    protection: Option<&PlanningObligationSet>,
    resolver: &dyn PlanningShapeResolver,
    poll_budget: &mut dyn FnMut() -> Result<(), PlanningUnknownReason>,
) -> Result<CandidateSet, PlanningUnknownReason> {
    let logical = logical_candidates(
        snapshot,
        requests,
        now_ns,
        limit,
        usize::MAX,
        &mut 0,
        protection,
        resolver,
        poll_budget,
    )?;
    let mut result = CandidateSet {
        waves: Vec::new(),
        truncated: logical.truncated,
        attempts: logical.attempts,
        shape_unknown: 0,
    };
    for work in logical.work {
        let execution_shape = match shape::resolve(snapshot, requests, &work, resolver, poll_budget)
        {
            Err(PlanningUnknownReason::ShapeUnavailable) => {
                result.shape_unknown += 1;
                continue;
            }
            other => other?,
        };
        let Some(execution_shape) = execution_shape else {
            continue;
        };
        if result.waves.len() == limit {
            result.truncated = true;
            break;
        }
        result.waves.push(WaveCandidate {
            work,
            execution_shape,
            based_on_generation: snapshot.generation,
            cost_model_version: snapshot.cost_model_version,
        });
    }
    Ok(result)
}
