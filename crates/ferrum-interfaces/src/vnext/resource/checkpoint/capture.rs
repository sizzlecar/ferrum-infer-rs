use super::{
    invalid_resource, Arc, CheckpointAuthorityId, CheckpointBackingOwner, DeviceRuntime, VNextError,
};
use crate::vnext::SuccessfulCheckpointCaptureSeal;
use serde::Serialize;
use std::num::NonZeroU64;

/// Process-local destination attempt identity. The owner authority prevents
/// equal serials on different checkpoint allocations from matching.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
pub(crate) struct CheckpointCaptureAttemptId {
    checkpoint: CheckpointAuthorityId,
    serial: NonZeroU64,
}

impl CheckpointCaptureAttemptId {
    pub(crate) fn checkpoint_authority(self) -> CheckpointAuthorityId {
        self.checkpoint
    }
    pub(crate) fn serial(self) -> u64 {
        self.serial.get()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CapturePhase {
    Fresh,
    Reserved(CheckpointCaptureAttemptId),
    PossiblySubmitted(CheckpointCaptureAttemptId),
    Captured(CheckpointCaptureAttemptId),
    Poisoned,
}

pub(super) struct CheckpointCaptureState {
    next_serial: Option<NonZeroU64>,
    phase: CapturePhase,
}

impl Default for CheckpointCaptureState {
    fn default() -> Self {
        Self {
            next_serial: Some(NonZeroU64::MIN),
            phase: CapturePhase::Fresh,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PermitPhase {
    Reserved,
    PossiblySubmitted,
}

/// The sole writable-destination authority for one capture attempt. A native
/// completion owner must take this permit before calling the backend submit.
/// Buffer views/retention Arcs alone do not authorize another write.
#[must_use = "capture destination ownership must reach a proven terminal or stay retained"]
pub(crate) struct CheckpointCapturePermit<R: DeviceRuntime> {
    backing: Option<Arc<CheckpointBackingOwner<R>>>,
    attempt: CheckpointCaptureAttemptId,
    phase: PermitPhase,
}

/// Only completion's unforgeable success seal can create this projection.
/// The underlying owner permanently rejects all later capture reservations.
#[must_use = "captured storage retains its physical and logical budget"]
pub(crate) struct CapturedCheckpointBacking<R: DeviceRuntime> {
    backing: Arc<CheckpointBackingOwner<R>>,
    attempt: CheckpointCaptureAttemptId,
}

impl<R: DeviceRuntime> std::fmt::Debug for CheckpointCapturePermit<R> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("CheckpointCapturePermit")
            .field("attempt", &self.attempt)
            .field("phase", &self.phase)
            .finish_non_exhaustive()
    }
}

impl<R: DeviceRuntime> std::fmt::Debug for CapturedCheckpointBacking<R> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("CapturedCheckpointBacking")
            .field("attempt", &self.attempt)
            .finish_non_exhaustive()
    }
}

impl<R: DeviceRuntime> CapturedCheckpointBacking<R> {
    pub(crate) fn backing(&self) -> &Arc<CheckpointBackingOwner<R>> {
        &self.backing
    }
    pub(crate) fn attempt_id(&self) -> CheckpointCaptureAttemptId {
        self.attempt
    }
}

/// A failed transition returns its still-owned permit. Unknown work must stay
/// in the same reaper's recovery path; an error must not silently drop it.
pub(crate) struct CheckpointCaptureFinishFailure<R: DeviceRuntime> {
    error: VNextError,
    permit: CheckpointCapturePermit<R>,
}

impl<R: DeviceRuntime> std::fmt::Debug for CheckpointCaptureFinishFailure<R> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("CheckpointCaptureFinishFailure")
            .field("error", &self.error)
            .field("attempt", &self.permit.attempt)
            .finish()
    }
}

impl<R: DeviceRuntime> CheckpointCaptureFinishFailure<R> {
    pub(crate) fn error(&self) -> &VNextError {
        &self.error
    }
    pub(crate) fn into_parts(self) -> (VNextError, CheckpointCapturePermit<R>) {
        (self.error, self.permit)
    }
}

impl<R: DeviceRuntime> CheckpointBackingOwner<R> {
    pub(crate) fn try_reserve_capture(
        self: &Arc<Self>,
    ) -> Result<CheckpointCapturePermit<R>, VNextError> {
        let _lifecycle = self
            .plan
            .read_lifecycle("reserve checkpoint capture destination")?;
        let mut state = self
            .capture
            .lock()
            .map_err(|_| invalid_resource("checkpoint capture mutex is poisoned"))?;
        if state.phase != CapturePhase::Fresh {
            return Err(invalid_resource(
                "checkpoint storage is not a fresh capture destination",
            ));
        }
        let serial = state.next_serial.ok_or_else(|| {
            invalid_resource("checkpoint capture attempt identities are exhausted")
        })?;
        let attempt = CheckpointCaptureAttemptId {
            checkpoint: self.authority(),
            serial,
        };
        state.next_serial = serial.get().checked_add(1).and_then(NonZeroU64::new);
        state.phase = CapturePhase::Reserved(attempt);
        Ok(CheckpointCapturePermit {
            backing: Some(Arc::clone(self)),
            attempt,
            phase: PermitPhase::Reserved,
        })
    }
}

impl<R: DeviceRuntime> CheckpointCapturePermit<R> {
    pub(crate) fn backing(&self) -> &Arc<CheckpointBackingOwner<R>> {
        self.backing
            .as_ref()
            .expect("live capture permit retains its backing")
    }

    pub(crate) fn attempt_id(&self) -> CheckpointCaptureAttemptId {
        self.attempt
    }

    /// Call before entering submit, since a panic can follow partial backend
    /// submission. There is deliberately no reversible local flag setter.
    pub(crate) fn mark_possibly_submitted(&mut self) -> Result<(), VNextError> {
        let mut state = self
            .backing()
            .capture
            .lock()
            .map_err(|_| invalid_resource("checkpoint capture mutex is poisoned"))?;
        if self.phase != PermitPhase::Reserved
            || state.phase != CapturePhase::Reserved(self.attempt)
        {
            return Err(invalid_resource(
                "capture submission does not own its exact reserved attempt",
            ));
        }
        state.phase = CapturePhase::PossiblySubmitted(self.attempt);
        drop(state);
        self.phase = PermitPhase::PossiblySubmitted;
        Ok(())
    }

    /// Only the backend's DefinitelyNotSubmitted branch may call this after
    /// submit was entered. Pre-submit cancellation simply drops the permit.
    pub(crate) fn definitely_not_submitted(self) -> Result<(), CheckpointCaptureFinishFailure<R>> {
        self.finish_without_capture(CapturePhase::Fresh, false)
    }

    /// Completion calls this for a failed fence or a proven safe drain. It
    /// establishes no valid snapshot and never reopens these bytes for capture.
    pub(crate) fn finish_failed_but_quiescent(
        self,
    ) -> Result<(), CheckpointCaptureFinishFailure<R>> {
        self.finish_without_capture(CapturePhase::Poisoned, true)
    }

    fn finish_without_capture(
        mut self,
        next: CapturePhase,
        recover_poison: bool,
    ) -> Result<(), CheckpointCaptureFinishFailure<R>> {
        let transitioned = (|| {
            let mut state = match self.backing().capture.lock() {
                Ok(state) => state,
                Err(poisoned) if recover_poison => poisoned.into_inner(),
                Err(_) => return Err(invalid_resource("checkpoint capture mutex is poisoned")),
            };
            if self.phase != PermitPhase::PossiblySubmitted
                || state.phase != CapturePhase::PossiblySubmitted(self.attempt)
            {
                return Err(invalid_resource(
                    "capture terminal does not own its exact submitted attempt",
                ));
            }
            state.phase = next;
            Ok(())
        })();
        if let Err(error) = transitioned {
            return Err(CheckpointCaptureFinishFailure {
                error,
                permit: self,
            });
        }
        // The slot transition and lock release precede physical/logical drop.
        drop(self.backing.take());
        Ok(())
    }

    pub(crate) fn finish_succeeded(
        mut self,
        seal: &SuccessfulCheckpointCaptureSeal,
    ) -> Result<CapturedCheckpointBacking<R>, CheckpointCaptureFinishFailure<R>> {
        let transitioned = (|| {
            if seal.attempt_id() != self.attempt {
                return Err(invalid_resource(
                    "capture success belongs to another destination attempt",
                ));
            }
            let mut state = self
                .backing()
                .capture
                .lock()
                .map_err(|_| invalid_resource("checkpoint capture mutex is poisoned"))?;
            if self.phase != PermitPhase::PossiblySubmitted
                || state.phase != CapturePhase::PossiblySubmitted(self.attempt)
            {
                return Err(invalid_resource(
                    "capture success lacks its exact submitted destination permit",
                ));
            }
            state.phase = CapturePhase::Captured(self.attempt);
            Ok(())
        })();
        if let Err(error) = transitioned {
            return Err(CheckpointCaptureFinishFailure {
                error,
                permit: self,
            });
        }
        Ok(CapturedCheckpointBacking {
            backing: self
                .backing
                .take()
                .expect("successful capture retains its backing"),
            attempt: self.attempt,
        })
    }
}

impl<R: DeviceRuntime> Drop for CheckpointCapturePermit<R> {
    fn drop(&mut self) {
        let Some(backing) = self.backing.take() else {
            return;
        };
        let (mut state, was_poisoned) = match backing.capture.lock() {
            Ok(state) => (state, false),
            Err(poisoned) => (poisoned.into_inner(), true),
        };
        if !was_poisoned
            && self.phase == PermitPhase::Reserved
            && state.phase == CapturePhase::Reserved(self.attempt)
        {
            state.phase = CapturePhase::Fresh;
        } else {
            state.phase = CapturePhase::Poisoned;
        }
        drop(state);
        if self.phase == PermitPhase::PossiblySubmitted {
            // Last-resort protocol-misuse containment only: without a terminal
            // or drain proof this owner cannot safely release device buffers.
            // Normal unknown work stays in the native reaper/deferred cleanup,
            // which calls an explicit quiescent finish before releasing it.
            std::mem::forget(backing);
        }
    }
}

#[cfg(test)]
impl<R: DeviceRuntime> CheckpointBackingOwner<R> {
    pub(in crate::vnext::resource) fn test_only_exhaust_capture_serial(&self) {
        self.capture.lock().unwrap().next_serial = Some(NonZeroU64::MAX);
    }

    pub(in crate::vnext::resource) fn test_only_poison_capture_mutex(&self) {
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _state = self.capture.lock().unwrap();
            panic!("injected capture state panic");
        }));
    }
}
