//! Publish sampling after the last prefill token without inventing KV work.
use super::*;

/// Exact scheduler owner and prefill boundary captured before sampling.
/// This is logical progress evidence, not permission to execute model work.
#[derive(Debug)]
pub struct PrefillOutputPublication {
    owner: Arc<()>,
    request_id: RequestId,
    ticket: WaitingAdmissionTicket,
    completed_frontier: LogicalWorkFrontier,
    context_tokens: usize,
    output_tokens: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrefillOutputPublicationOutcome {
    Published,
    AlreadyPublished,
}

impl ContinuousBatchScheduler {
    /// Capture before sampling, then finish the existing prefill commit before
    /// publishing. The caller must serialize its sequence owner through both
    /// operations; this receipt additionally rejects scheduler/RequestId reuse.
    pub fn prepare_prefill_output_publication(
        &self,
        request_id: &RequestId,
        context_tokens: usize,
        previous_output_tokens: usize,
    ) -> Result<PrefillOutputPublication> {
        let prefill = self.prefill_queue.read();
        let decode = self.decode_queue.read();
        let request = prefill
            .iter()
            .find(|request| request.inner.request.id == *request_id)
            .or_else(|| decode.requests.get(request_id))
            .ok_or_else(|| FerrumError::scheduler("prefill output owner is absent"))?;
        let ticket = request.waiting_admission_ticket.ok_or_else(|| {
            FerrumError::scheduler("prefill output owner has no admission identity")
        })?;
        let mut completed_frontier = request.logical_work_frontier.clone();
        let (computed, resident, _, outputs, _) = completed_frontier.planning_counters();
        if outputs != previous_output_tokens || context_tokens == 0 {
            return Err(FerrumError::scheduler(
                "prefill output history does not match owner",
            ));
        }
        match request.phase {
            RequestPhase::Prefilling if request.prefill_chunk_offset <= context_tokens => {
                completed_frontier.commit_prefill(
                    context_tokens,
                    context_tokens - request.prefill_chunk_offset,
                );
                completed_frontier.begin_decode();
            }
            // Legacy full-prefill can report its final chunk before sampling.
            RequestPhase::Decoding
                if completed_frontier.prefill_output_pending()
                    && request.prefill_chunk_offset == context_tokens
                    && request.prefill_tokens == context_tokens
                    && computed == context_tokens
                    && resident == context_tokens => {}
            _ => {
                return Err(FerrumError::scheduler(
                    "prefill output has a stale context or phase",
                ))
            }
        }
        let output_tokens = previous_output_tokens
            .checked_add(1)
            .ok_or_else(|| FerrumError::scheduler("prefill output count exhausted"))?;
        if output_tokens > request.inner.request.sampling_params.max_tokens {
            return Err(FerrumError::scheduler(
                "prefill output would exceed the request limit",
            ));
        }
        Ok(PrefillOutputPublication {
            owner: Arc::clone(&self.planning_owner),
            request_id: request_id.clone(),
            ticket,
            completed_frontier,
            context_tokens,
            output_tokens,
        })
    }

    /// Publish exactly one sampled output at the completed prefill boundary.
    /// KV counters and legacy decode statistics are deliberately unchanged.
    pub fn publish_prefill_output_commit(
        &self,
        publication: &PrefillOutputPublication,
        actual_total_outputs: usize,
    ) -> Result<PrefillOutputPublicationOutcome> {
        if !Arc::ptr_eq(&self.planning_owner, &publication.owner)
            || actual_total_outputs != publication.output_tokens
        {
            return Err(FerrumError::scheduler(
                "prefill output publication mismatched owner or count",
            ));
        }
        let mut decode = self.decode_queue.write();
        let request = decode
            .requests
            .get_mut(&publication.request_id)
            .ok_or_else(|| FerrumError::scheduler("prefill output owner is no longer decoding"))?;
        if request.waiting_admission_ticket != Some(publication.ticket)
            || request.phase != RequestPhase::Decoding
            || request.prefill_chunk_offset != publication.context_tokens
            || request.prefill_tokens != publication.context_tokens
        {
            return Err(FerrumError::scheduler(
                "prefill output publication is stale",
            ));
        }
        let mut published_frontier = publication.completed_frontier.clone();
        published_frontier.commit_prefill_output(actual_total_outputs);
        if request.logical_work_frontier == published_frontier {
            return Ok(PrefillOutputPublicationOutcome::AlreadyPublished);
        }
        if request.logical_work_frontier != publication.completed_frontier {
            return Err(FerrumError::scheduler(
                "prefill output frontier changed before publication",
            ));
        }
        request.logical_work_frontier = published_frontier;
        let generation = request.logical_work_frontier.progress_generation();
        drop(decode);
        self.record_pressure_frontier_progress(&publication.request_id, generation);
        Ok(PrefillOutputPublicationOutcome::Published)
    }
}

#[cfg(test)]
mod tests;
