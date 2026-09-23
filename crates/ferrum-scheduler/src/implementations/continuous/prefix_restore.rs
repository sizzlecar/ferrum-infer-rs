//! Conditional publication of an independently restored prefill prefix.

use std::sync::{Arc, Weak};

use super::{
    AdmissionDeferral, AdmissionWakeEpochs, AdmissionWakeSnapshot, ContinuousBatchRequest,
    ContinuousBatchScheduler, ExecutionCapacityQueuePhase, ExecutionCapacityReleaseSnapshot,
    ExecutionMaintenanceRetryTicket, LogicalWorkGeneration, RequestPhase, WaitingAdmissionMode,
    WaitingAdmissionTicket, EXECUTION_READINESS_FAILED, EXECUTION_READINESS_PENDING,
};
use ferrum_interfaces::model_executor::ExecutorExecutionCapacityDeferral;
use ferrum_interfaces::scheduler::PreparedPrefixRestore;
use ferrum_interfaces::vnext::DeferredAction;
use ferrum_types::{FerrumError, RequestId, RequestState, Result};
use std::sync::atomic::Ordering;

#[derive(Debug, Clone, Default)]
pub(super) struct PrefixRestoreState {
    // Allocated lazily on the first preparation. Reset on every real admission,
    // including zero-progress re-admission retaining the same waiting ticket.
    incarnation: Option<Arc<()>>,
    pending: Option<Weak<()>>,
    restored_tokens: usize,
    capacity_hold: bool,
    abandoned: bool,
}

impl PrefixRestoreState {
    pub(super) fn planning_state(&self) -> (bool, usize, bool, bool) {
        (
            self.is_pending(),
            self.restored_tokens,
            self.capacity_hold,
            self.abandoned,
        )
    }
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
        release_orphaned_capacity_hold(request);
        if request.prefix_restore.abandoned
            || self.request_index.read().get(request_id) != Some(&RequestPhase::Prefilling)
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
        request.prefix_restore.capacity_hold = false;
        Ok(())
    }
}

/// A retained optional checkpoint may wait only while an existing capacity
/// gate has a concrete progress source. It never creates a pressure episode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrefixRestoreCapacityStatus {
    Retry,
    Pending,
    Fallback,
    Stale,
}

fn matches_proof(request: &ContinuousBatchRequest, proof: &PrefixRestoreProof) -> bool {
    eligible(request, proof.expected_offset, proof.prompt_tokens)
        && request.waiting_admission_ticket == Some(proof.admission_ticket)
        && request.logical_work_frontier.progress_generation() == proof.generation
        && request
            .prefix_restore
            .incarnation
            .as_ref()
            .is_some_and(|value| Arc::ptr_eq(value, &proof.incarnation))
        && request
            .prefix_restore
            .pending
            .as_ref()
            .is_some_and(|value| value.ptr_eq(&Arc::downgrade(&proof.reservation)))
}

fn clear_capacity_hold(request: &mut ContinuousBatchRequest) {
    if request.prefix_restore.capacity_hold {
        request.execution_capacity_deferral = None;
        request.execution_maintenance_retry = None;
        request.prefix_restore.capacity_hold = false;
    }
}

pub(super) fn release_orphaned_capacity_hold(request: &mut ContinuousBatchRequest) {
    if request.prefix_restore.capacity_hold && !request.prefix_restore.is_pending() {
        clear_capacity_hold(request);
        request.prefix_restore.abandoned = true;
    }
}

impl ContinuousBatchScheduler {
    fn prefix_restore_has_runnable_releaser<'a>(
        &self,
        request_id: &RequestId,
        deferral: &AdmissionDeferral,
        release: &ExecutionCapacityReleaseSnapshot,
        peers: impl Iterator<Item = &'a ContinuousBatchRequest>,
    ) -> bool {
        let iteration = self.current_iteration.load(Ordering::Relaxed);
        peers.into_iter().any(|peer| {
            peer.inner.request.id != *request_id
                && peer.inner.state == RequestState::Running
                && !peer.prefix_restore.is_pending()
                && peer.execution_capacity_deferral.is_none()
                && !peer
                    .execution_readiness_block
                    .as_ref()
                    .is_some_and(|block| {
                        matches!(
                            block.status(),
                            EXECUTION_READINESS_PENDING | EXECUTION_READINESS_FAILED
                        )
                    })
                && !peer
                    .execution_maintenance_retry
                    .is_some_and(|ticket| iteration < ticket.not_before_iteration)
                && (peer.phase == RequestPhase::Decoding
                    || (peer.phase == RequestPhase::Prefilling
                        && peer.prefill_chunk_offset < peer.prefill_tokens))
                && release.can_advance(&peer.inner.request.id, deferral.wait_condition())
        })
    }

    /// Install the ordinary capacity gate only on this still-prepared frontier.
    /// Optional cache reuse never invokes pressure SCC victim selection. A
    /// peer must both own a relevant release authority and be runnable now.
    pub fn defer_prefix_restore_for_capacity(
        &self,
        prepared: &PreparedPrefixRestore,
        capacity: &ExecutorExecutionCapacityDeferral,
        release: &ExecutionCapacityReleaseSnapshot,
    ) -> Result<bool> {
        let proof = prepared.proof_ref::<PrefixRestoreProof>()?;
        let retry =
            capacity.validated_maintenance_retry_scope(std::slice::from_ref(&proof.request_id))?;
        let observed = capacity.observed();
        let deferral = AdmissionDeferral::new(
            DeferredAction::WaitForRelease,
            AdmissionWakeEpochs::new(
                observed.coordinator_id,
                observed.release_epoch,
                observed.capacity_epoch,
                0,
            ),
            capacity.wait_condition().clone(),
        );
        let mut prefill = self.prefill_queue.write();
        let decode = self.decode_queue.read();
        let runnable = self.prefix_restore_has_runnable_releaser(
            &proof.request_id,
            &deferral,
            release,
            prefill.iter().chain(decode.requests.values()),
        );
        let Some(request) = prefill
            .iter_mut()
            .find(|request| request.inner.request.id == proof.request_id)
        else {
            return Ok(false);
        };
        if !matches_proof(request, proof)
            || self.request_index.read().get(&proof.request_id) != Some(&RequestPhase::Prefilling)
        {
            return Ok(false);
        }
        clear_capacity_hold(request);
        if let Some(retry) = retry {
            let epoch = retry.progress().latest_capacity_epoch();
            if !retry.progress().mutations().is_empty()
                && epoch != 0
                && request
                    .last_execution_maintenance_capacity_epoch
                    .is_none_or(|last| epoch > last)
            {
                request.execution_maintenance_retry = Some(ExecutionMaintenanceRetryTicket {
                    not_before_iteration: self
                        .current_iteration
                        .load(Ordering::Relaxed)
                        .saturating_add(1),
                    latest_capacity_epoch: epoch,
                });
                request.last_execution_maintenance_capacity_epoch = Some(epoch);
                request.prefix_restore.capacity_hold = true;
                return Ok(true);
            }
        }
        if !runnable {
            request.prefix_restore.abandoned = true;
            return Ok(false);
        }
        request.execution_capacity_deferral = Some(deferral);
        request.prefix_restore.capacity_hold = true;
        Ok(true)
    }

    /// Recheck the exact existing source epochs before touching native state.
    /// If all potential releasers have become blocked, relinquish this optional
    /// pin and resume cold instead of joining a dependency cycle.
    pub fn resume_prefix_restore_after_capacity(
        &self,
        prepared: &PreparedPrefixRestore,
        wake: AdmissionWakeSnapshot<'_>,
        release: &ExecutionCapacityReleaseSnapshot,
    ) -> Result<PrefixRestoreCapacityStatus> {
        let proof = prepared.proof_ref::<PrefixRestoreProof>()?;
        let mut prefill = self.prefill_queue.write();
        let decode = self.decode_queue.read();
        let Some(index) = prefill
            .iter()
            .position(|request| request.inner.request.id == proof.request_id)
        else {
            return Ok(PrefixRestoreCapacityStatus::Stale);
        };
        if !matches_proof(&prefill[index], proof)
            || self.request_index.read().get(&proof.request_id) != Some(&RequestPhase::Prefilling)
        {
            return Ok(PrefixRestoreCapacityStatus::Stale);
        }
        let request = &mut prefill[index];
        if !request.prefix_restore.capacity_hold {
            return Ok(PrefixRestoreCapacityStatus::Fallback);
        }
        let had_maintenance = request.execution_maintenance_retry.is_some();
        if Self::execution_maintenance_retry_is_blocked(
            request,
            self.current_iteration.load(Ordering::Relaxed),
        )? {
            return Ok(PrefixRestoreCapacityStatus::Pending);
        }
        let mut ignore = |_| {};
        let mut mode = WaitingAdmissionMode::Prepared {
            wake,
            observer: &mut ignore,
        };
        if had_maintenance
            || !Self::execution_capacity_is_blocked(
                request,
                &mut mode,
                ExecutionCapacityQueuePhase::Prefill,
            )?
        {
            clear_capacity_hold(request);
            return Ok(PrefixRestoreCapacityStatus::Retry);
        }
        let deferral = request
            .execution_capacity_deferral
            .clone()
            .expect("blocked gate retained its predicate");
        let runnable = self.prefix_restore_has_runnable_releaser(
            &proof.request_id,
            &deferral,
            release,
            prefill.iter().chain(decode.requests.values()),
        );
        if runnable {
            Ok(PrefixRestoreCapacityStatus::Pending)
        } else {
            clear_capacity_hold(&mut prefill[index]);
            prefill[index].prefix_restore.abandoned = true;
            Ok(PrefixRestoreCapacityStatus::Fallback)
        }
    }

    /// Release only a gate owned by this exact reservation. A stale id cannot
    /// clear a replacement's state, and this incarnation cannot reattach after
    /// choosing cold progress.
    pub fn abandon_prefix_restore_capacity(&self, prepared: &PreparedPrefixRestore) -> Result<()> {
        let proof = prepared.proof_ref::<PrefixRestoreProof>()?;
        let mut prefill = self.prefill_queue.write();
        if let Some(request) = prefill
            .iter_mut()
            .find(|request| request.inner.request.id == proof.request_id)
        {
            if matches_proof(request, proof) {
                clear_capacity_hold(request);
                request.prefix_restore.abandoned = true;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests;
