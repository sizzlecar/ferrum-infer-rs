use super::{invalid_completion, StateTransferIdentity, StateTransferResult};
use crate::vnext::{CompletionSlotId, DeviceRuntime, VNextError};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum StateTransferResultStatus {
    Pending,
    Ready,
    Taken,
}

enum ResultState<R: DeviceRuntime> {
    Pending,
    Ready(StateTransferResult<R>),
    Taken,
}

/// One slot allocated before submission and strongly owned by the same reaper.
/// The parent supplies its existing CompletionSlotId; this is neither a second
/// registry nor a new allocator. A handle may detach without dropping this slot.
pub(crate) struct StateTransferResultSlot<R: DeviceRuntime> {
    slot_id: CompletionSlotId,
    identity: Arc<StateTransferIdentity>,
    state: Mutex<ResultState<R>>,
    consumer_abandoned: AtomicBool,
}

/// Rejected publication returns its owner after unlocking. In particular, a
/// later failure cannot overwrite a previously published successful restore.
pub(crate) struct StateTransferPublishFailure<R: DeviceRuntime> {
    pub(in crate::vnext::completion) error: VNextError,
    pub(in crate::vnext::completion) result: StateTransferResult<R>,
}

impl<R: DeviceRuntime> StateTransferResultSlot<R> {
    pub(in crate::vnext::completion) fn new(
        slot_id: CompletionSlotId,
        identity: Arc<StateTransferIdentity>,
    ) -> Arc<Self> {
        Arc::new(Self {
            slot_id,
            identity,
            state: Mutex::new(ResultState::Pending),
            consumer_abandoned: AtomicBool::new(false),
        })
    }

    /// A public access handle can disappear while another observer holds the
    /// record lock in a fence wait. Mark abandonment without waiting on that
    /// lock; the existing reaper remains the sole native resource owner.
    pub(in crate::vnext::completion) fn abandon_consumer(&self) {
        self.consumer_abandoned.store(true, Ordering::Release);
    }

    pub(in crate::vnext::completion) fn consumer_abandoned(&self) -> bool {
        self.consumer_abandoned.load(Ordering::Acquire)
    }

    pub(crate) fn slot_id(&self) -> CompletionSlotId {
        self.slot_id
    }

    pub(crate) fn status(&self) -> Result<StateTransferResultStatus, VNextError> {
        let state = self
            .state
            .lock()
            .map_err(|_| invalid_completion("native transfer result slot is poisoned"))?;
        Ok(match &*state {
            ResultState::Pending => StateTransferResultStatus::Pending,
            ResultState::Ready(_) => StateTransferResultStatus::Ready,
            ResultState::Taken => StateTransferResultStatus::Taken,
        })
    }

    pub(in crate::vnext::completion) fn publish(
        &self,
        result: StateTransferResult<R>,
    ) -> Result<(), StateTransferPublishFailure<R>> {
        let reason = if result.slot_id() != self.slot_id
            || result.identity().as_ref() != self.identity.as_ref()
        {
            "native transfer result belongs to another reservation"
        } else {
            match self.state.lock() {
                Ok(mut state) if matches!(*state, ResultState::Pending) => {
                    *state = ResultState::Ready(result);
                    return Ok(());
                }
                Ok(state) => {
                    drop(state);
                    "native transfer result was already published or taken"
                }
                Err(poisoned) => {
                    drop(poisoned);
                    "native transfer result slot is poisoned"
                }
            }
        };
        Err(StateTransferPublishFailure {
            error: invalid_completion(reason),
            result,
        })
    }

    pub(crate) fn take(&self) -> Result<Option<StateTransferResult<R>>, VNextError> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| invalid_completion("native transfer result slot is poisoned"))?;
        match &*state {
            ResultState::Pending => Ok(None),
            ResultState::Taken => {
                drop(state);
                Err(invalid_completion(
                    "native transfer result was already taken",
                ))
            }
            ResultState::Ready(_) => {
                let previous = std::mem::replace(&mut *state, ResultState::Taken);
                drop(state);
                let ResultState::Ready(result) = previous else {
                    unreachable!("ready result was validated under its slot lock")
                };
                Ok(Some(result))
            }
        }
    }
}
