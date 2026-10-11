//! Bounded whole-segment hints publish only after exact completion and normal
//! step retirement. Tickets own no Step, Sequence, frame lease or device flight.
use super::*;
use crate::vnext::operation::segment_compile::CompiledSegmentBindingRecipe;
use crate::vnext::{
    BatchInvocationId, BatchStepId, DeviceReusableExecutionEntryIdentity,
    DeviceReusableExecutionProgramId, StepParticipantFrameAssignment,
    StepParticipantRetirementDisposition, StepRetirementReceipt,
};

struct CachedSegmentBindingRecipe {
    recipe: Arc<CompiledSegmentBindingRecipe>,
    entry: DeviceReusableExecutionEntryIdentity,
    epoch: u64,
}

#[derive(Default)]
pub(super) struct SegmentBindingRecipeCache {
    // Exactly one current entry per lane, never a history of shapes/requests.
    current: Mutex<Option<CachedSegmentBindingRecipe>>,
}

#[must_use = "publication requires the exact successful completion and normal retirement"]
pub struct SegmentBindingPublication<R: DeviceRuntime> {
    lane: Weak<ExecutionLane<R>>,
    recipe: Arc<CompiledSegmentBindingRecipe>,
    entry: DeviceReusableExecutionEntryIdentity,
    epoch: u64,
    slot_id: CompletionSlotId,
    submission_fingerprint: String,
    batch_step_id: BatchStepId,
    batch_invocation_id: BatchInvocationId,
    lane_id: ExecutionLaneId,
    frames: Vec<StepParticipantFrameAssignment>,
}

#[must_use = "discard this candidate on any readback, output or retirement error"]
pub struct ReadySegmentBindingPublication<R: DeviceRuntime> {
    pending: SegmentBindingPublication<R>,
}

impl<R: DeviceRuntime> SegmentBindingPublication<R> {
    pub fn bind_completion(
        self,
        completion: &OperationCompletionReceipt,
    ) -> Option<ReadySegmentBindingPublication<R>> {
        let submission = completion.submission();
        let batch = submission.batch_identity();
        (matches!(
            completion.disposition(),
            OperationCompletionDisposition::Succeeded
        ) && submission.slot_id() == self.slot_id
            && submission.fingerprint() == self.submission_fingerprint
            && batch.batch_step_id() == self.batch_step_id
            && batch.batch_invocation_id() == self.batch_invocation_id
            && batch.lane_id() == self.lane_id)
            .then_some(ReadySegmentBindingPublication { pending: self })
    }
}

impl<R: DeviceRuntime> ReadySegmentBindingPublication<R> {
    pub fn publish(self, retirement: &StepRetirementReceipt) -> Result<bool, VNextError> {
        let pending = self.pending;
        if retirement.batch_step_id() != pending.batch_step_id
            || retirement.participants().len() != pending.frames.len()
            || retirement
                .participants()
                .iter()
                .zip(&pending.frames)
                .any(|(actual, expected)| {
                    actual.assignment() != *expected
                        || actual.disposition() != StepParticipantRetirementDisposition::Committed
                })
        {
            return Err(invalid_completion(
                "segment publication differs from exact committed step retirement",
            ));
        }
        let Some(lane) = pending.lane.upgrade() else {
            return Ok(false);
        };
        if lane.id != pending.lane_id {
            return Err(invalid_completion(
                "segment publication belongs to another lane",
            ));
        }
        lane.publish_segment_binding_recipe(pending.recipe, pending.entry, pending.epoch)
    }
}

impl<R: DeviceRuntime> CompletionHandle<R> {
    /// Cloned handles share this cell: only one can take a pending candidate.
    /// Ordinary handles allocate neither a cell nor a publication ticket.
    pub fn take_segment_binding_publication(&self) -> Option<SegmentBindingPublication<R>> {
        let cell = self.segment_binding_publication.as_ref()?;
        let mut pending = cell.lock().ok()?;
        pending.take()
    }
}

impl<R: DeviceRuntime> CompletionReservation<R> {
    pub(crate) fn set_segment_binding_candidate(
        &mut self,
        recipe: Arc<CompiledSegmentBindingRecipe>,
        entry: DeviceReusableExecutionEntryIdentity,
        epoch: u64,
    ) -> Result<(), VNextError> {
        if self.segment_binding_candidate.is_some() || self.submission_may_have_happened {
            return Err(invalid_completion(
                "segment candidate may be staged once before submission",
            ));
        }
        let Some(CompletionResourceLease::Wave(wave)) = self.resources.as_ref() else {
            return Err(invalid_completion(
                "segment candidate requires a complete wave reservation",
            ));
        };
        let lane = self
            .lane
            .as_ref()
            .ok_or_else(|| invalid_completion("segment reservation has no lane"))?;
        let receipt = self
            .receipt
            .as_ref()
            .ok_or_else(|| invalid_completion("segment reservation has no receipt"))?;
        if epoch != lane.reusable_execution_epoch() || wave.execution_lane_id() != lane.id {
            return Err(invalid_completion(
                "segment candidate lane epoch changed before staging",
            ));
        }
        let first = wave
            .nodes()
            .first()
            .ok_or_else(|| invalid_completion("segment candidate wave has no nodes"))?;
        self.segment_binding_candidate = Some(SegmentBindingPublication {
            lane: Arc::downgrade(lane),
            recipe,
            entry,
            epoch,
            slot_id: receipt.slot_id(),
            submission_fingerprint: receipt.fingerprint().to_owned(),
            batch_step_id: wave.batch_step_id(),
            batch_invocation_id: wave.batch_invocation_id(),
            lane_id: lane.id,
            frames: first.participant_frames().to_vec(),
        });
        Ok(())
    }
}

impl<R: DeviceRuntime> ExecutionLane<R> {
    pub fn reusable_execution_entry_identity(
        &self,
        program: &DeviceReusableExecutionProgramId,
    ) -> Result<Option<DeviceReusableExecutionEntryIdentity>, VNextError> {
        self.with_quiescent_stream("inspect reusable execution entry", |runtime, stream| {
            runtime.reusable_execution_entry_identity(stream, program)
        })
    }

    pub(crate) fn try_segment_binding_entry(
        &self,
        program: &DeviceReusableExecutionProgramId,
    ) -> Result<Option<(DeviceReusableExecutionEntryIdentity, u64)>, VNextError> {
        if self.fail_closed.load(Ordering::Acquire) {
            return Err(invalid_completion(
                "fail-closed lane cannot inspect segment entry",
            ));
        }
        let state = match self.state.try_lock() {
            Ok(state) => state,
            Err(std::sync::TryLockError::WouldBlock) => return Ok(None),
            Err(std::sync::TryLockError::Poisoned(_)) => {
                return Err(invalid_completion("execution lane state mutex is poisoned"))
            }
        };
        if state.fail_closed || self.fail_closed.load(Ordering::Acquire) {
            return Err(invalid_completion(
                "fail-closed lane cannot inspect segment entry",
            ));
        }
        if state.in_flight != 0 {
            return Ok(None);
        }
        self.checked_segment_binding_entry(&state, program)
            .map(|entry| entry.map(|entry| (entry, self.reusable_execution_epoch())))
    }

    pub(crate) fn lookup_segment_binding_recipe(
        &self,
        program: &DeviceReusableExecutionProgramId,
    ) -> Result<Option<Arc<CompiledSegmentBindingRecipe>>, VNextError> {
        if self.fail_closed.load(Ordering::Acquire) {
            return Err(invalid_completion(
                "fail-closed lane cannot reuse segment recipe",
            ));
        }
        let Some(cache) = self.segment_binding_recipe.get() else {
            return Ok(None);
        };
        let state = match self.state.try_lock() {
            Ok(state) => state,
            Err(std::sync::TryLockError::WouldBlock) => return Ok(None),
            Err(std::sync::TryLockError::Poisoned(_)) => {
                return Err(invalid_completion("execution lane state mutex is poisoned"))
            }
        };
        if state.fail_closed || self.fail_closed.load(Ordering::Acquire) {
            return Err(invalid_completion(
                "fail-closed lane cannot inspect segment entry",
            ));
        }
        if state.in_flight != 0 {
            return Ok(None);
        }
        let entry = self.checked_segment_binding_entry(&state, program)?;
        let mut current = cache
            .current
            .lock()
            .map_err(|_| invalid_completion("segment recipe cache mutex is poisoned"))?;
        let matches = current.as_ref().is_some_and(|cached| {
            cached.recipe.program_id == *program
                && cached.epoch == self.reusable_execution_epoch()
                && entry
                    .as_ref()
                    .is_some_and(|entry| cached.entry.same_entry(entry))
        });
        let (selected, retired) = if matches {
            (
                current.as_ref().map(|cached| Arc::clone(&cached.recipe)),
                None,
            )
        } else {
            (None, current.take())
        };
        drop(current);
        drop(state);
        drop(retired);
        Ok(selected)
    }

    fn publish_segment_binding_recipe(
        &self,
        recipe: Arc<CompiledSegmentBindingRecipe>,
        expected_entry: DeviceReusableExecutionEntryIdentity,
        epoch: u64,
    ) -> Result<bool, VNextError> {
        if self.fail_closed.load(Ordering::Acquire) || epoch != self.reusable_execution_epoch() {
            return Ok(false);
        }
        let state = match self.state.try_lock() {
            Ok(state) => state,
            Err(std::sync::TryLockError::WouldBlock) => return Ok(false),
            Err(std::sync::TryLockError::Poisoned(_)) => {
                return Err(invalid_completion("execution lane state mutex is poisoned"))
            }
        };
        if state.in_flight != 0 || epoch != self.reusable_execution_epoch() {
            return Ok(false);
        }
        let current_entry = self.checked_segment_binding_entry(&state, &recipe.program_id)?;
        if !current_entry.is_some_and(|entry| entry.same_entry(&expected_entry)) {
            return Ok(false);
        }
        let cache = self
            .segment_binding_recipe
            .get_or_init(SegmentBindingRecipeCache::default);
        let mut current = cache
            .current
            .lock()
            .map_err(|_| invalid_completion("segment recipe cache mutex is poisoned"))?;
        let retired = current.replace(CachedSegmentBindingRecipe {
            recipe,
            entry: expected_entry,
            epoch,
        });
        drop(current);
        drop(state);
        drop(retired);
        Ok(true)
    }

    fn checked_segment_binding_entry(
        &self,
        state: &ExecutionLaneState<R::Stream>,
        program: &DeviceReusableExecutionProgramId,
    ) -> Result<Option<DeviceReusableExecutionEntryIdentity>, VNextError> {
        if state.fail_closed || self.fail_closed.load(Ordering::Acquire) || state.in_flight != 0 {
            return Err(invalid_completion(
                "segment entry requires a quiescent usable lane",
            ));
        }
        let inspected = catch_unwind(AssertUnwindSafe(|| {
            if !self.current_descriptor_matches_snapshot()
                || self.runtime.stream_state(&state.stream) != StreamState::Ready
            {
                return Err(invalid_completion(
                    "segment entry inspection requires a stable lane",
                ));
            }
            let entry = self
                .runtime
                .reusable_execution_entry_identity(&state.stream, program)
                .map_err(|error| {
                    invalid_completion(format!("segment entry inspection failed: {error}"))
                })?;
            if !self.current_descriptor_matches_snapshot()
                || self.runtime.stream_state(&state.stream) != StreamState::Ready
            {
                return Err(invalid_completion(
                    "segment entry inspection changed lane state",
                ));
            }
            Ok(entry)
        }));
        let result = inspected.unwrap_or_else(|_| {
            Err(invalid_completion(
                "runtime panicked while inspecting segment entry",
            ))
        });
        if result.is_err() {
            self.fail_closed.store(true, Ordering::Release);
        }
        result
    }

    #[cfg(test)]
    pub(crate) fn segment_binding_locks_available_for_test(&self) -> bool {
        let lane_available = self.state.try_lock().is_ok();
        let cache_available = self
            .segment_binding_recipe
            .get()
            .is_none_or(|cache| cache.current.try_lock().is_ok());
        lane_available && cache_available
    }

    pub(super) fn clear_segment_binding_recipe(&self) -> Result<(), VNextError> {
        let Some(cache) = self.segment_binding_recipe.get() else {
            return Ok(());
        };
        let retired = {
            let mut current = cache
                .current
                .lock()
                .map_err(|_| invalid_completion("segment recipe cache mutex is poisoned"))?;
            current.take()
        };
        drop(retired);
        Ok(())
    }
}

impl<R: DeviceRuntime> ExecutionLaneEnqueue<'_, R> {
    pub(crate) fn validate_segment_binding_recipe(
        &self,
        recipe: &Arc<CompiledSegmentBindingRecipe>,
    ) -> Result<(), VNextError> {
        let cache =
            self.lane.segment_binding_recipe.get().ok_or_else(|| {
                invalid_completion("segment recipe was not published on this lane")
            })?;
        let (entry, epoch) = {
            let current = cache
                .current
                .lock()
                .map_err(|_| invalid_completion("segment recipe cache mutex is poisoned"))?;
            let cached = current
                .as_ref()
                .filter(|cached| Arc::ptr_eq(&cached.recipe, recipe))
                .ok_or_else(|| {
                    invalid_completion("segment recipe is no longer the current lane entry")
                })?;
            (cached.entry.clone(), cached.epoch)
        };
        if epoch != self.lane.reusable_execution_epoch() {
            return Err(invalid_completion("segment epoch changed before enqueue"));
        }
        let actual = self
            .lane
            .checked_segment_binding_entry(&self.state, &recipe.program_id)?;
        if !actual.is_some_and(|actual| actual.same_entry(&entry)) {
            return Err(invalid_completion(
                "segment executable entry changed before enqueue",
            ));
        }
        Ok(())
    }
}
