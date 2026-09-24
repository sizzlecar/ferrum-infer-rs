//! Candidate-local work frontiers. Advancing numbers is not evidence about
//! future token content, output credit, resource permission, or commit times.
use super::*;

#[derive(Clone)]
pub(super) struct ProjectedRequest {
    pub key: RequestWorkKey,
    pub phase: RequestPhaseView,
    pub context_tokens: u32,
    pub generated: u32,
    pub maximum_output_tokens: NonZeroU32,
    pub output_policy_signature: [u8; 32],
    pub host_content_changed: bool,
}

pub(super) struct PreparedRow {
    pub index: usize,
    pub work: ActualRowWork,
}

pub(super) struct PreparedWork {
    revision: u32,
    pub rows: Vec<PreparedRow>,
    next: Vec<ProjectedRequest>,
}

pub(super) struct ProjectedFrontiers {
    requests: Vec<ProjectedRequest>,
    maximum_context: u32,
    maximum_rows: usize,
    revision: u32,
}

impl ProjectedFrontiers {
    pub fn new(
        requests: &[RequestSchedulingView],
        maximum_context: u32,
        maximum_rows: usize,
        poll: &mut dyn FnMut() -> std::result::Result<(), PlanningUnknownReason>,
    ) -> std::result::Result<Self, PlanningUnknownReason> {
        if requests.is_empty() || requests.len() > 256 || maximum_rows == 0 || maximum_rows > 256 {
            return Err(PlanningUnknownReason::InvalidSnapshot);
        }
        let mut projected: Vec<ProjectedRequest> = Vec::with_capacity(requests.len());
        for request in requests {
            poll()?;
            if request.context_tokens > maximum_context
                || request.timing.committed_tokens > request.timing.maximum_output_tokens.get()
                || projected
                    .iter()
                    .any(|old| old.key.request_id == request.key.request_id)
            {
                return Err(PlanningUnknownReason::InvalidSnapshot);
            }
            if let RequestPhaseView::Prefill(progress) = &request.phase {
                let total = progress.total_prompt_tokens.get();
                if progress.offset >= total
                    || progress.offset > progress.logical_high_water
                    || progress.logical_high_water > total
                    || progress.executable_until < progress.offset
                    || progress.executable_until > total
                    || total > maximum_context
                {
                    return Err(PlanningUnknownReason::InvalidSnapshot);
                }
            }
            projected.push(ProjectedRequest {
                key: request.key.clone(),
                phase: request.phase.clone(),
                context_tokens: request.context_tokens,
                generated: request.timing.committed_tokens,
                maximum_output_tokens: request.timing.maximum_output_tokens,
                output_policy_signature: request.output_policy_signature,
                host_content_changed: false,
            });
        }
        Ok(Self {
            requests: projected,
            maximum_context,
            maximum_rows,
            revision: 0,
        })
    }

    pub fn requests(&self) -> &[ProjectedRequest] {
        &self.requests
    }

    pub fn request(&self, index: usize) -> &ProjectedRequest {
        &self.requests[index]
    }

    /// Check the whole wave before changing any row, including unselected
    /// peers. A rejected later row cannot advance an earlier row's frontier.
    pub fn prepare(
        &self,
        work: &[CandidateWork],
        poll: &mut dyn FnMut() -> std::result::Result<(), PlanningUnknownReason>,
    ) -> std::result::Result<PreparedWork, PlanningUnknownReason> {
        if work.is_empty() || work.len() > self.maximum_rows || self.revision >= 16 {
            return Err(PlanningUnknownReason::InvalidShapeEvidence);
        }
        let mut rows: Vec<PreparedRow> = Vec::with_capacity(work.len());
        let mut successors = Vec::with_capacity(work.len());
        for selected in work {
            poll()?;
            let mut found = None;
            for (index, request) in self.requests.iter().enumerate() {
                poll()?;
                if request.key == selected.key {
                    found = Some(index);
                    break;
                }
            }
            let index = found.ok_or(PlanningUnknownReason::InvalidShapeEvidence)?;
            if rows.iter().any(|row| row.index == index) {
                return Err(PlanningUnknownReason::InvalidShapeEvidence);
            }
            let (work, next) =
                self.requests[index].after_work(selected.action.clone(), self.maximum_context)?;
            rows.push(PreparedRow { index, work });
            successors.push(next);
        }
        Ok(PreparedWork {
            revision: self.revision,
            rows,
            next: successors,
        })
    }

    /// Call only after the complete physical-route projection passed parity.
    /// This changes a local simulation, never the real scheduler or sequence.
    pub fn advance(
        &mut self,
        prepared: PreparedWork,
    ) -> std::result::Result<(), PlanningUnknownReason> {
        if prepared.revision != self.revision {
            return Err(PlanningUnknownReason::InvalidShapeEvidence);
        }
        let revision = self
            .revision
            .checked_add(1)
            .ok_or(PlanningUnknownReason::ArithmeticOverflow)?;
        for (row, next) in prepared.rows.into_iter().zip(prepared.next) {
            self.requests[row.index] = next;
        }
        self.revision = revision;
        Ok(())
    }
}

impl ProjectedRequest {
    fn after_work(
        &self,
        action: WaveAction,
        maximum_context: u32,
    ) -> std::result::Result<(ActualRowWork, Self), PlanningUnknownReason> {
        if self.generated >= self.maximum_output_tokens.get() {
            return Err(PlanningUnknownReason::InvalidShapeEvidence);
        }
        let mut next = self.clone();
        let (work, emits_token) = match (action, &self.phase) {
            (WaveAction::Decode, RequestPhaseView::Decode) if self.context_tokens > 0 => {
                next.context_tokens = self
                    .context_tokens
                    .checked_add(1)
                    .ok_or(PlanningUnknownReason::ArithmeticOverflow)?;
                (
                    ActualRowWork::Decode {
                        kv_tokens: self.context_tokens,
                    },
                    true,
                )
            }
            (WaveAction::Prefill { offset, count }, RequestPhaseView::Prefill(progress)) => {
                let total = progress.total_prompt_tokens.get();
                let end = offset
                    .checked_add(count.get())
                    .ok_or(PlanningUnknownReason::ArithmeticOverflow)?;
                if offset != progress.offset || end > total || end > progress.executable_until {
                    return Err(PlanningUnknownReason::InvalidShapeEvidence);
                }
                next.context_tokens = self.context_tokens.max(end);
                if end == total {
                    // The sampled token is output, not yet a computed KV token.
                    // This also applies to final prefill during recomputation.
                    next.phase = RequestPhaseView::Decode;
                } else if let RequestPhaseView::Prefill(next_progress) = &mut next.phase {
                    next_progress.offset = end;
                    next_progress.logical_high_water = progress.logical_high_water.max(end);
                }
                (
                    ActualRowWork::Prefill {
                        offset,
                        count: count.get(),
                        total_prompt_tokens: total,
                    },
                    end == total,
                )
            }
            _ => return Err(PlanningUnknownReason::InvalidShapeEvidence),
        };
        if next.context_tokens > maximum_context {
            return Err(PlanningUnknownReason::OutputOrResourceBlocked);
        }
        if emits_token {
            next.generated = self
                .generated
                .checked_add(1)
                .ok_or(PlanningUnknownReason::ArithmeticOverflow)?;
            if next.generated > self.maximum_output_tokens.get() {
                return Err(PlanningUnknownReason::InvalidShapeEvidence);
            }
            // Never reuse captured UTF-8/matcher/sampling branch evidence for
            // an unknown token, even if all scalar work is now well defined.
            next.host_content_changed = true;
        }
        Ok((work, next))
    }

    pub fn matches_query(&self, request: &RequestSchedulingView) -> bool {
        self.key == request.key
            && self.context_tokens == request.context_tokens
            && self.generated == request.timing.committed_tokens
            && self.maximum_output_tokens == request.timing.maximum_output_tokens
            && self.output_policy_signature == request.output_policy_signature
            && same_phase(&self.phase, &request.phase)
    }
}

pub(super) fn same_phase(left: &RequestPhaseView, right: &RequestPhaseView) -> bool {
    match (left, right) {
        (RequestPhaseView::Decode, RequestPhaseView::Decode) => true,
        (RequestPhaseView::Prefill(left), RequestPhaseView::Prefill(right)) => {
            left.offset == right.offset
                && left.total_prompt_tokens == right.total_prompt_tokens
                && left.logical_high_water == right.logical_high_water
                && left.executable_until == right.executable_until
                && left.admitted_at_ns == right.admitted_at_ns
                && left.reference_work_at_admission_ns == right.reference_work_at_admission_ns
                && Arc::ptr_eq(&left.reference, &right.reference)
                && Arc::ptr_eq(&left.milestones, &right.milestones)
        }
        _ => false,
    }
}
