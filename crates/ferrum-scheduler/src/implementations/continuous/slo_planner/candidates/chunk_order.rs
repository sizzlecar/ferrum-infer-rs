//! Logical progress hints only. Neither W nor a proposal order predicts time.
use super::*;
use std::num::NonZeroUsize;

pub(super) struct Goal<'a> {
    request: &'a RequestSchedulingView,
    progress: &'a PrefillProgressView,
    high_water_work: u64,
    pub per_wave_work: u64,
}

impl<'a> Goal<'a> {
    pub(super) fn new(
        request: &'a RequestSchedulingView,
        horizon_ns: u64,
        waves: NonZeroUsize,
        poll: &mut dyn FnMut() -> Result<(), PlanningUnknownReason>,
    ) -> Result<Self, PlanningUnknownReason> {
        let RequestPhaseView::Prefill(progress) = &request.phase else {
            return Err(PlanningUnknownReason::InvalidSnapshot);
        };
        let high_water_work = progress
            .reference
            .work_at(progress.logical_high_water)
            .ok_or(PlanningUnknownReason::MissingReferenceWork)?;
        let final_work = progress
            .reference
            .work_at(progress.total_prompt_tokens.get())
            .ok_or(PlanningUnknownReason::MissingReferenceWork)?;
        let mut required = high_water_work;
        if request
            .timing
            .next_deadline_ns()
            .is_some_and(|due| due <= horizon_ns)
        {
            required = final_work;
        }
        for milestone in progress.milestones.iter() {
            poll()?;
            if milestone.at_ns <= horizon_ns {
                required = required.max(milestone.required_reference_work_ns);
            }
        }
        // With no near obligation, remaining useful work still provides a
        // scale hint for throughput. It never adds a hard completion promise.
        if required <= high_water_work {
            required = final_work;
        }
        let remaining = required.saturating_sub(high_water_work);
        let waves =
            u64::try_from(waves.get()).map_err(|_| PlanningUnknownReason::ArithmeticOverflow)?;
        let per_wave_work = remaining / waves + u64::from(remaining % waves != 0);
        Ok(Self {
            request,
            progress,
            high_water_work,
            per_wave_work,
        })
    }

    pub(super) fn credit(&self, work: &CandidateWork) -> Result<u64, PlanningUnknownReason> {
        if work.key != self.request.key {
            // An endpoint legal only for another owner cannot impersonate the
            // most urgent owner's progress. It remains a later valid proposal.
            return Ok(0);
        }
        let WaveAction::Prefill { offset, count } = work.action else {
            return Err(PlanningUnknownReason::InvalidSnapshot);
        };
        let end = offset
            .checked_add(count.get())
            .ok_or(PlanningUnknownReason::ArithmeticOverflow)?;
        let work = self
            .progress
            .reference
            .work_at(end.max(self.progress.logical_high_water))
            .ok_or(PlanningUnknownReason::MissingReferenceWork)?;
        work.checked_sub(self.high_water_work)
            .ok_or(PlanningUnknownReason::InvalidSnapshot)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn count(n: u32) -> NonZeroU32 {
        NonZeroU32::new(n).unwrap()
    }

    #[test]
    fn progress_scale_order_keeps_every_nonuniform_legal_choice() {
        let original = vec![count(3), count(9), count(27), count(81)];
        let mut chunks = original.clone();
        interleave(&mut chunks, &[1, 2, 100, 101], 50, &mut || Ok(())).unwrap();
        assert_eq!(chunks, vec![count(27), count(81), count(9), count(3)]);
        chunks.sort();
        assert_eq!(chunks, original);
    }

    #[test]
    fn progress_scale_selection_does_not_hide_a_budget_failure() {
        let mut chunks = vec![count(1), count(2)];
        let original = chunks.clone();
        assert_eq!(
            interleave(&mut chunks, &[1, 2], 1, &mut || Err(
                PlanningUnknownReason::ComputeBudgetExhausted
            )),
            Err(PlanningUnknownReason::ComputeBudgetExhausted)
        );
        assert_eq!(chunks, original);
    }
}

/// Visit a goal-scale action, then alternate larger/smaller scales. All legal
/// entries remain; neither cost monotonicity nor a token-count constant is used.
pub(super) fn interleave(
    chunks: &mut Vec<NonZeroU32>,
    credits: &[u64],
    target: u64,
    poll: &mut dyn FnMut() -> Result<(), PlanningUnknownReason>,
) -> Result<(), PlanningUnknownReason> {
    if chunks.len() != credits.len() {
        return Err(PlanningUnknownReason::InvalidSnapshot);
    }
    if chunks.len() < 2 {
        return poll();
    }
    let mut entries: Vec<_> = chunks
        .iter()
        .copied()
        .zip(credits.iter().copied())
        .collect();
    poll()?;
    entries.sort_by_key(|(chunk, credit)| (*credit, chunk.get()));
    let maximum_credit = entries.last().expect("nonempty").1;
    let target = target.min(maximum_credit);
    let anchor = entries
        .iter()
        .position(|(_, credit)| *credit >= target)
        .expect("bounded target");
    let mut ordered = Vec::with_capacity(entries.len());
    ordered.push(entries[anchor].0);
    for distance in 1..entries.len() {
        poll()?;
        if let Some((chunk, _)) = entries.get(anchor + distance) {
            ordered.push(*chunk);
        }
        if let Some(index) = anchor.checked_sub(distance) {
            ordered.push(entries[index].0);
        }
    }
    *chunks = ordered;
    poll()
}
