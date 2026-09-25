//! Untimed logical projection for bounded demand diagnostics. Uses the same
//! action and output-credit rules as production planning, without costs,
//! deadlines, predicted commits, resource grants or consumer releases.
use super::{output::work_output_advance, shape, types::*};

#[derive(Clone)]
pub struct PlanningStructure<'a> {
    snapshot: &'a SchedulerSnapshot,
    requests: Vec<RequestSchedulingView>,
    remaining_output_bytes: u64,
}

impl<'a> PlanningStructure<'a> {
    pub fn new(snapshot: &'a SchedulerSnapshot) -> Self {
        Self {
            snapshot,
            requests: snapshot.requests.clone(),
            remaining_output_bytes: snapshot.capacity.available_output_bytes,
        }
    }

    pub fn requests(&self) -> &[RequestSchedulingView] {
        &self.requests
    }

    /// Returns a separate local successor. Even a late failure leaves the
    /// parent intact. The caller must also project all physical alternatives
    /// successfully before retaining the successor.
    pub fn advance(
        &self,
        work: &[CandidateWork],
        poll: &mut dyn FnMut() -> Result<(), PlanningUnknownReason>,
    ) -> Result<Self, PlanningUnknownReason> {
        poll()?;
        shape::validate_work(self.snapshot, &self.requests, work, poll)?
            .ok_or(PlanningUnknownReason::InvalidShapeEvidence)?;
        let mut next = self.clone();
        for row in work {
            poll()?;
            let request = next
                .requests
                .iter_mut()
                .find(|request| request.key == row.key)
                .ok_or(PlanningUnknownReason::InvalidShapeEvidence)?;
            let advance = work_output_advance(
                request,
                &row.action,
                self.snapshot.capacity.maximum_context_tokens.get(),
            )?;
            next.remaining_output_bytes = next
                .remaining_output_bytes
                .checked_sub(advance.additional_output_bytes)
                .ok_or(PlanningUnknownReason::OutputOrResourceBlocked)?;
            if let (WaveAction::Prefill { offset, count }, RequestPhaseView::Prefill(progress)) =
                (&row.action, &mut request.phase)
            {
                let end = offset
                    .checked_add(count.get())
                    .ok_or(PlanningUnknownReason::ArithmeticOverflow)?;
                progress.offset = end;
                progress.logical_high_water = progress.logical_high_water.max(end);
                if end == progress.total_prompt_tokens.get() {
                    request.phase = RequestPhaseView::Decode;
                }
            }
            request.context_tokens = advance.context_tokens;
            request.output_credit = advance.output_credit;
            if advance.emits_token {
                request.timing.committed_tokens = request
                    .timing
                    .committed_tokens
                    .checked_add(1)
                    .ok_or(PlanningUnknownReason::ArithmeticOverflow)?;
            }
            // Keep original first/last commit timestamps. This object has no
            // time model and cannot be submitted as a scheduling witness.
        }
        poll()?;
        Ok(next)
    }
}
