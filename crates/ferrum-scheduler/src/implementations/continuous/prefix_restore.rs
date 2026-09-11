//! Conditional publication of an independently restored prefill prefix.

use std::sync::{Arc, Weak};

use super::{
    ContinuousBatchRequest, ContinuousBatchScheduler, LogicalWorkGeneration, RequestPhase,
    WaitingAdmissionTicket,
};
use ferrum_interfaces::scheduler::PreparedPrefixRestore;
use ferrum_types::{FerrumError, RequestId, RequestState, Result};

#[derive(Debug, Clone, Default)]
pub(super) struct PrefixRestoreState {
    // Allocated lazily on the first preparation. Reset on every real admission,
    // including zero-progress re-admission retaining the same waiting ticket.
    incarnation: Option<Arc<()>>,
    pending: Option<Weak<()>>,
    restored_tokens: usize,
}

impl PrefixRestoreState {
    pub(super) fn begin_admission(&mut self) {
        *self = Self::default();
    }

    pub(super) fn is_pending(&self) -> bool {
        self.pending
            .as_ref()
            .is_some_and(|pending| pending.strong_count() > 0)
    }

    pub(super) const fn restored_tokens(&self) -> usize {
        self.restored_tokens
    }
}

// Neither Clone nor publicly constructible. Holding the reservation keeps this
// exact request out of compute batches; drop releases it without a queue lock.
struct PrefixRestoreProof {
    request_id: RequestId,
    expected_offset: usize,
    prompt_tokens: usize,
    admission_ticket: WaitingAdmissionTicket,
    generation: LogicalWorkGeneration,
    incarnation: Arc<()>,
    reservation: Arc<()>,
}

fn eligible(
    request: &ContinuousBatchRequest,
    expected_offset: usize,
    prompt_tokens: usize,
) -> bool {
    request.phase == RequestPhase::Prefilling
        && request.inner.state == RequestState::Running
        && request.prefill_chunk_offset == expected_offset
        && request
            .logical_work_frontier
            .can_restore_at(expected_offset)
        && (request.prefill_tokens == 0 || request.prefill_tokens == prompt_tokens)
}

impl ContinuousBatchScheduler {
    pub(super) fn prepare_prefix_restore_inner(
        &self,
        request_id: &RequestId,
        expected_offset: usize,
        prompt_tokens: usize,
    ) -> Result<Option<PreparedPrefixRestore>> {
        if expected_offset >= prompt_tokens {
            return Err(FerrumError::scheduler(
                "Prefix restore requires a non-empty suffix",
            ));
        }
        let mut prefill = self.prefill_queue.write();
        let Some(request) = prefill
            .iter_mut()
            .find(|request| request.inner.request.id == *request_id)
        else {
            return Ok(None);
        };
        if self.request_index.read().get(request_id) != Some(&RequestPhase::Prefilling)
            || !eligible(request, expected_offset, prompt_tokens)
            || request.prefix_restore.is_pending()
        {
            return Ok(None);
        }
        let Some(admission_ticket) = request.waiting_admission_ticket else {
            return Err(FerrumError::scheduler(
                "Admitted request lost its waiting admission identity",
            ));
        };
        let incarnation = request
            .prefix_restore
            .incarnation
            .get_or_insert_with(|| Arc::new(()))
            .clone();
        let reservation = Arc::new(());
        request.prefix_restore.pending = Some(Arc::downgrade(&reservation));
        Ok(Some(PreparedPrefixRestore::new(
            request_id.clone(),
            expected_offset,
            prompt_tokens,
            PrefixRestoreProof {
                request_id: request_id.clone(),
                expected_offset,
                prompt_tokens,
                admission_ticket,
                generation: request.logical_work_frontier.progress_generation(),
                incarnation,
                reservation,
            },
        )))
    }

    pub(super) fn commit_prefix_restored_inner(
        &self,
        prepared: PreparedPrefixRestore,
        restored_boundary: usize,
    ) -> Result<()> {
        let proof = prepared.into_proof::<PrefixRestoreProof>()?;
        if restored_boundary <= proof.expected_offset || restored_boundary >= proof.prompt_tokens {
            return Err(FerrumError::scheduler(
                "Restored boundary must advance the prefix and retain a non-empty suffix",
            ));
        }
        let mut prefill = self.prefill_queue.write();
        let request = prefill
            .iter_mut()
            .find(|request| request.inner.request.id == proof.request_id)
            .ok_or_else(|| {
                FerrumError::scheduler("Restored request is no longer admitted for prefill")
            })?;
        if self.request_index.read().get(&proof.request_id) != Some(&RequestPhase::Prefilling)
            || !eligible(request, proof.expected_offset, proof.prompt_tokens)
            || request.waiting_admission_ticket != Some(proof.admission_ticket)
            || request.logical_work_frontier.progress_generation() != proof.generation
            || !request
                .prefix_restore
                .incarnation
                .as_ref()
                .is_some_and(|incarnation| Arc::ptr_eq(incarnation, &proof.incarnation))
            || !request
                .prefix_restore
                .pending
                .as_ref()
                .is_some_and(|reservation| reservation.ptr_eq(&Arc::downgrade(&proof.reservation)))
        {
            return Err(FerrumError::scheduler(
                "Prefix restore preparation is stale or its frontier is executing",
            ));
        }
        let restored_tokens = request
            .prefix_restore
            .restored_tokens
            .checked_add(restored_boundary - proof.expected_offset)
            .ok_or_else(|| FerrumError::scheduler("Restored token accounting overflow"))?;
        // All fallible checks precede the single locked metadata update. Do not
        // route this through mark_prefill_chunk_processed: it records executor
        // work and capacity-fit feedback. Publication must not do either.
        request
            .logical_work_frontier
            .commit_restored_prefix(restored_boundary);
        request.prefill_tokens = proof.prompt_tokens;
        request.prefill_chunk_offset = restored_boundary;
        request.inner.tokens_processed = restored_boundary;
        request.chunked_prefill = true;
        request.prefix_restore.restored_tokens = restored_tokens;
        request.prefix_restore.pending = None;
        Ok(())
    }
}

#[cfg(test)]
mod tests;
