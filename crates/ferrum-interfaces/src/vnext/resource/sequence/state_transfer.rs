use super::{
    invalid_resource, ActiveSequenceSessionState, Arc, DeviceRuntime, NonZeroU64,
    SequenceBackingGeneration, SequenceBackingSnapshot, SequenceSession, SequenceSessionEpoch,
    SequenceSessionFingerprint, SequenceSessionPhase, SequenceSessionSlot,
    SequenceSessionSlotState, VNextError,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SequenceStateTransferKind {
    CaptureRead,
    RestoreWrite,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct StateTransferReservation {
    serial: NonZeroU64,
    kind: SequenceStateTransferKind,
    generation: SequenceBackingGeneration,
}

/// This slot arbitrates state access, not token validity or device completion.
/// The capture boundary proof and imported-state publication are separate
/// authorities; neither can be manufactured by reserving an idle sequence.
#[derive(Debug, Clone)]
pub(crate) struct SequenceStateTransferSlot {
    next_serial: Option<NonZeroU64>,
    active: Option<StateTransferReservation>,
}

impl Default for SequenceStateTransferSlot {
    fn default() -> Self {
        Self {
            next_serial: Some(NonZeroU64::MIN),
            active: None,
        }
    }
}

impl SequenceStateTransferSlot {
    pub(crate) fn is_reserved(&self) -> bool {
        self.active.is_some()
    }

    fn reserve(
        &mut self,
        kind: SequenceStateTransferKind,
        generation: SequenceBackingGeneration,
    ) -> Result<StateTransferReservation, VNextError> {
        if self.is_reserved() {
            return Err(invalid_resource(
                "sequence already owns a state transfer reservation",
            ));
        }
        let serial = self.next_serial.ok_or_else(|| {
            invalid_resource("sequence state transfer reservation identities are exhausted")
        })?;
        let reservation = StateTransferReservation {
            serial,
            kind,
            generation,
        };
        self.next_serial = serial.get().checked_add(1).and_then(NonZeroU64::new);
        self.active = Some(reservation);
        Ok(reservation)
    }
}

pub(crate) enum SequenceStateTransferPreparation<R: DeviceRuntime> {
    Prepared(PreparedSequenceStateTransfer<R>),
    Busy,
    StaleBacking,
}

/// Exclusive reservation before any device submission. It retains both the
/// exact backing generation and the session. Dropping it rolls back only this
/// reservation, including after cancellation; it never reopens a cancelled
/// session or replaces a newer reservation.
///
/// This type deliberately provides no dispatch or successful-restore method.
/// Device submission must first move these owners into the native completion
/// lifecycle; a prepared reservation alone is not copy/restore authority.
#[must_use = "the preparation owns exclusive sequence state access"]
pub(crate) struct PreparedSequenceStateTransfer<R: DeviceRuntime> {
    reservation: PreparedStateTransferHold,
    // Backing drops before its session parent after the reservation rolls back.
    backing: Arc<SequenceBackingSnapshot<R>>,
    session: Arc<SequenceSession<R>>,
}

impl<R: DeviceRuntime> PreparedSequenceStateTransfer<R> {
    pub(crate) fn backing(&self) -> &Arc<SequenceBackingSnapshot<R>> {
        &self.backing
    }

    pub(crate) fn session(&self) -> &Arc<SequenceSession<R>> {
        &self.session
    }

    pub(crate) fn kind(&self) -> SequenceStateTransferKind {
        self.reservation.reservation.kind
    }

    pub(super) fn ensure_active_reservation(
        &self,
        active: &ActiveSequenceSessionState,
    ) -> Result<(), VNextError> {
        if active.epoch != self.reservation.epoch
            || active.fingerprint != self.reservation.fingerprint
            || active.state_transfer.active != Some(self.reservation.reservation)
            || self.reservation.reservation.generation != self.backing.generation()
        {
            return Err(invalid_resource(
                "state transfer reservation does not own this exact session and backing",
            ));
        }
        Ok(())
    }
}

struct PreparedStateTransferHold {
    slot: Arc<SequenceSessionSlot>,
    epoch: SequenceSessionEpoch,
    fingerprint: SequenceSessionFingerprint,
    reservation: StateTransferReservation,
}

impl Drop for PreparedStateTransferHold {
    fn drop(&mut self) {
        let mut state = match self.slot.state.lock() {
            Ok(state) => state,
            Err(poisoned) => {
                *poisoned.into_inner() = SequenceSessionSlotState::FailClosed;
                return;
            }
        };
        match &mut *state {
            SequenceSessionSlotState::Active(active)
                if active.epoch == self.epoch && active.fingerprint == self.fingerprint =>
            {
                if active.state_transfer.active == Some(self.reservation) {
                    active.state_transfer.active = None;
                } else {
                    // An ownership mismatch cannot release another attempt.
                    active.phase = SequenceSessionPhase::Poisoned;
                }
            }
            // Never mutate a different session generation from a stale owner.
            _ => {}
        }
    }
}

fn ensure_transfer_candidate(active: &ActiveSequenceSessionState) -> Result<bool, VNextError> {
    if active.phase != SequenceSessionPhase::Open {
        return Err(invalid_resource(
            "state transfer requires an open sequence session",
        ));
    }
    Ok(active.active_frame.is_none()
        && !active.has_participant_flights()
        && !active.state_transfer.is_reserved())
}

impl<R: DeviceRuntime> SequenceSession<R> {
    /// Reserve state access without allocating, encoding, submitting, or
    /// waiting for device work. The caller separately proves a complete
    /// capture boundary or a fresh restore target before using the reservation.
    pub(crate) fn try_prepare_state_transfer(
        self: &Arc<Self>,
        kind: SequenceStateTransferKind,
        expected_generation: SequenceBackingGeneration,
    ) -> Result<SequenceStateTransferPreparation<R>, VNextError> {
        let _lifecycle = self
            .resources
            .request
            .plan
            .resources
            .read_lifecycle("prepare sequence state transfer")?;
        if self.resources.is_poisoned() {
            return Err(invalid_resource(
                "poisoned sequence cannot prepare state transfer",
            ));
        }
        let fingerprint = self.fingerprint.clone();
        let mut state = self
            .slot
            .state
            .lock()
            .map_err(|_| invalid_resource("sequence session state mutex is poisoned"))?;
        let active = match &mut *state {
            SequenceSessionSlotState::Active(active)
                if active.epoch == self.epoch && active.fingerprint == self.fingerprint =>
            {
                active
            }
            _ => {
                return Err(invalid_resource(
                    "state transfer sequence session is stale or inactive",
                ))
            }
        };
        if !ensure_transfer_candidate(active)? {
            return Ok(SequenceStateTransferPreparation::Busy);
        }
        // Frame acquisition and extension publication use this same lock order.
        let backing = self.resources.lock_backing_state()?;
        if backing.current.generation() != expected_generation {
            return Ok(SequenceStateTransferPreparation::StaleBacking);
        }
        let reservation = active.state_transfer.reserve(kind, expected_generation)?;
        Ok(SequenceStateTransferPreparation::Prepared(
            PreparedSequenceStateTransfer {
                reservation: PreparedStateTransferHold {
                    slot: Arc::clone(&self.slot),
                    epoch: self.epoch,
                    fingerprint,
                    reservation,
                },
                backing: Arc::clone(&backing.current),
                session: Arc::clone(self),
            },
        ))
    }
}

#[cfg(test)]
mod tests;
