#[cfg(test)]
use super::shape;
use super::{obligations::PlanningObligationSet, types::*};
use std::{cmp::Reverse, num::NonZeroU32};

mod chunk_order;
mod frontier;
pub(super) use frontier::FrontierCursor;

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

fn prefill_work(
    requests: &[RequestSchedulingView],
    indices: &[usize],
    size: usize,
    chunk: NonZeroU32,
    caps: &BackendPlanningCapabilities,
    poll_budget: &mut dyn FnMut() -> Result<(), PlanningUnknownReason>,
) -> Result<Option<Vec<CandidateWork>>, PlanningUnknownReason> {
    let envelope = work_envelope(caps, requests, poll_budget)?;
    let mut work = Vec::new();
    for &index in indices {
        poll_budget()?;
        let request = &requests[index];
        let RequestPhaseView::Prefill(progress) = &request.phase else {
            continue;
        };
        let remaining = progress.total_prompt_tokens.get() - progress.offset;
        let chunk = u64::from(chunk.get())
            .min(envelope.maximum_prefill_chunk.unwrap_or(u64::MAX))
            .min(envelope.maximum_prefill_tokens.unwrap_or(u64::MAX));
        let Ok(chunk) = u32::try_from(chunk) else {
            continue;
        };
        if chunk == 0 {
            continue;
        }
        let count = if remaining < chunk && caps.allow_final_short_chunk {
            remaining
        } else {
            chunk
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
    let mut cursor =
        FrontierCursor::new(snapshot, requests, now_ns, limit, protection, poll_budget)?;
    let mut result = CandidateSet {
        waves: Vec::new(),
        truncated: false,
        attempts: 0,
        shape_unknown: 0,
    };
    while let Some(mut work) = cursor.next(
        snapshot,
        requests,
        &mut result.attempts,
        usize::MAX,
        poll_budget,
    )? {
        shape::order_work(snapshot, &mut work, resolver, poll_budget)?;
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
            cost_evidence: None,
            work,
            execution_shape,
            based_on_generation: snapshot.generation,
            cost_model_version: snapshot.cost_model_version,
        });
    }
    result.truncated |= cursor.truncated;
    Ok(result)
}

/// Only policy arithmetic. The execution provider still owns route/resources.
pub(super) fn work_envelope(
    caps: &BackendPlanningCapabilities,
    requests: &[RequestSchedulingView],
    poll: &mut dyn FnMut() -> Result<(), PlanningUnknownReason>,
) -> Result<super::super::work_policy::WaveWorkEnvelope, PlanningUnknownReason> {
    let mut decoders = 0;
    for request in requests {
        poll()?;
        if runnable(request) && matches!(request.phase, RequestPhaseView::Decode) {
            decoders += 1;
        }
    }
    Ok(caps.work_policy.for_ready_decoders(decoders))
}

pub(super) fn within_work_envelope(
    caps: &BackendPlanningCapabilities,
    requests: &[RequestSchedulingView],
    work: &[CandidateWork],
    poll: &mut dyn FnMut() -> Result<(), PlanningUnknownReason>,
) -> Result<bool, PlanningUnknownReason> {
    let envelope = work_envelope(caps, requests, poll)?;
    let mut used = super::super::work_policy::WaveWorkUsage::default();
    for row in work {
        poll()?;
        let prefill = match row.action {
            WaveAction::Decode => None,
            WaveAction::Prefill { count, .. } => std::num::NonZeroU64::new(u64::from(count.get())),
        };
        if !envelope.include(&mut used, prefill) {
            return Ok(false);
        }
    }
    Ok(true)
}
