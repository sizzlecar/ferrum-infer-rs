//! A necessary per-owner H bound, used only to order existing goal proposals.
//! It supplies no cost, route, resource or execution evidence.
use super::*;
use std::num::NonZeroUsize;

pub(super) fn opening(
    snapshot: &SchedulerSnapshot,
    requests: &[RequestSchedulingView],
    prefills: &[usize],
    sizes: &[usize],
    chunks: &[NonZeroU32],
    goal: Option<&RequestWorkKey>,
    remaining: Option<NonZeroUsize>,
    protection: Option<&PlanningObligationSet>,
    poll: &mut dyn FnMut() -> Result<(), PlanningUnknownReason>,
) -> Result<Option<(usize, NonZeroU32)>, PlanningUnknownReason> {
    let (Some(goal), Some(remaining), Some(&size), Some(&chunk)) =
        (goal, remaining, sizes.first(), chunks.first())
    else {
        return Ok(None);
    };
    // The existing goal/recovery ordering owns membership. If the requested
    // goal cannot execute this frontier, peers cannot impersonate its progress.
    if !prefills.first().is_some_and(|&i| requests[i].key == *goal) {
        return Ok(None);
    }
    let caps = &snapshot.capabilities;
    // A future decoder can finish and relax the work policy. Use the widest
    // declared chunk bound, never today's stricter ready-decoder envelope.
    let Some(maximum) = caps.prefill_chunk_sizes.iter().map(|n| n.get()).max() else {
        return Ok(None);
    };
    let maximum = u64::from(maximum).min(caps.max_prefill_tokens_per_wave.get());
    let tail = remaining.get() - 1;
    let Some(first) = prefill_work(requests, prefills, size, chunk, caps, poll)? else {
        return Ok(None);
    };
    if !first.iter().any(|row| row.key == *goal)
        || !exceeds_owner_bound(
            snapshot, requests, goal, &first, maximum, tail, protection, poll,
        )?
    {
        return Ok(None);
    }
    // The original opening cannot close within H even with unlimited peer
    // packing. Try another existing endpoint/cohort before descending its tail.
    // Retain the old opening and every ordinary cursor stage for Unknown routes.
    let mut widest_first = sizes.to_vec();
    widest_first.sort_unstable_by(|a, b| b.cmp(a));
    poll()?;
    for &chunk in chunks {
        for &size in &widest_first {
            poll()?;
            let Some(work) = prefill_work(requests, prefills, size, chunk, caps, poll)? else {
                continue;
            };
            if !work.iter().any(|row| row.key == *goal)
                || !within_work_envelope(caps, requests, &work, poll)?
            {
                continue;
            }
            let tokens: u64 = work
                .iter()
                .map(|row| match row.action {
                    WaveAction::Prefill { count, .. } => u64::from(count.get()),
                    WaveAction::Decode => 0,
                })
                .sum();
            if tokens > caps.max_prefill_tokens_per_wave.get() {
                continue;
            }
            // Do not charge an unrelated optional owner to the target's first
            // token simply because a larger declared batch exists.
            let mut compulsory = true;
            for row in &work {
                poll()?;
                let Some(index) = requests.iter().position(|request| request.key == row.key) else {
                    return Err(PlanningUnknownReason::InvalidSnapshot);
                };
                compulsory &= required_end(snapshot, requests, index, goal, protection).is_some();
            }
            if compulsory
                && !exceeds_owner_bound(
                    snapshot, requests, goal, &work, maximum, tail, protection, poll,
                )?
            {
                return Ok(Some((size, chunk)));
            }
        }
    }
    Ok(None)
}

fn required_end(
    snapshot: &SchedulerSnapshot,
    requests: &[RequestSchedulingView],
    index: usize,
    goal: &RequestWorkKey,
    protection: Option<&PlanningObligationSet>,
) -> Option<u32> {
    let request = &requests[index];
    // A held follower can advance through a later typed restore. Its prompt
    // is not a lower bound on future physical prefill work.
    if !runnable(request) {
        return None;
    }
    let RequestPhaseView::Prefill(progress) = &request.phase else {
        return None;
    };
    let due = protection.is_none_or(|scope| scope.protects(index))
        && request
            .timing
            .next_deadline_ns()
            .is_some_and(|deadline| deadline <= snapshot.scope.horizon_end_ns);
    if due {
        Some(progress.total_prompt_tokens.get())
    } else if request.key == *goal {
        // Producer / preparation routes cap this at their declared boundary.
        Some(progress.executable_until)
    } else {
        None
    }
}

fn exceeds_owner_bound(
    snapshot: &SchedulerSnapshot,
    requests: &[RequestSchedulingView],
    goal: &RequestWorkKey,
    work: &[CandidateWork],
    maximum_chunk: u64,
    remaining_waves: usize,
    protection: Option<&PlanningObligationSet>,
    poll: &mut dyn FnMut() -> Result<(), PlanningUnknownReason>,
) -> Result<bool, PlanningUnknownReason> {
    // Each owner can occur at most once per model wave. Ignore resource/cost,
    // later decode and milestones here: they can only add work. In particular
    // do NOT sum owners' waves, since all peers may share future waves.
    for (index, request) in requests.iter().enumerate() {
        poll()?;
        let Some(end) = required_end(snapshot, requests, index, goal, protection) else {
            continue;
        };
        let RequestPhaseView::Prefill(progress) = &request.phase else {
            unreachable!()
        };
        let mut offset = progress.offset;
        if let Some(row) = work.iter().find(|row| row.key == request.key) {
            if let WaveAction::Prefill {
                offset: start,
                count,
            } = row.action
            {
                offset = start
                    .checked_add(count.get())
                    .ok_or(PlanningUnknownReason::ArithmeticOverflow)?;
            }
        }
        let tokens = u64::from(end.saturating_sub(offset));
        let waves = tokens.div_ceil(maximum_chunk);
        if u128::from(waves) > remaining_waves as u128 {
            return Ok(true);
        }
    }
    Ok(false)
}
