//! Retirement of a failed reservation/allocation, before static submission.
//! This owns the original typestate and its recovery ledger, not a new plan.
use super::*;
use crate::{
    defer_device_cleanup, DeferredDeviceCleanupDisposition, DeferredDeviceCleanupTask,
    FailureEnvelope,
};
use std::panic::{catch_unwind, AssertUnwindSafe};

enum PreparationState<D: ResourceTransactionDriver> {
    Reserve(ResourcePrepareTransitionError<D, TransactionNew>),
    Commit(ResourcePrepareTransitionError<D, TransactionReserved>),
    Poisoned(ResourcePoisonedTransaction<D>),
    Reserved(Option<ResourceTransaction<D, TransactionReserved>>),
    Rollback(ResourceRollbackTransitionError<D>),
}

/// Original failed pre-submission preparation and its checked retirement.
///
/// Ordinary errors preserve the exact remaining ledger for explicit retry.
/// A backend hook panic poisons this owner: its unknown side effects cannot be
/// treated as successful compensation or replayed automatically. Dropping the
/// owner transfers it to the existing static-provision cleanup fallback.
#[must_use = "failed preparation must retire or retain its original resources and capacity"]
pub struct ResourcePreparationRetirement<D: ResourceTransactionDriver + 'static> {
    state: Option<PreparationState<D>>,
    failure: FailureEnvelope,
    poison: Option<FailureEnvelope>,
    retirement_started: bool,
}

impl<D: ResourceTransactionDriver + 'static> fmt::Debug for ResourcePreparationRetirement<D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ResourcePreparationRetirement")
            .field("failure", &self.failure)
            .field("retired", &self.is_retired())
            .field("poison", &self.poison)
            .finish_non_exhaustive()
    }
}

impl<D: ResourceTransactionDriver + 'static> ResourcePreparationRetirement<D> {
    pub fn from_reserve(error: ResourcePrepareTransitionError<D, TransactionNew>) -> Self {
        Self {
            failure: error.failure().failure().clone(),
            state: Some(PreparationState::Reserve(error)),
            poison: None,
            retirement_started: false,
        }
    }

    pub fn from_commit(error: ResourceCommitTransitionError<D>) -> Self {
        let (failure, state) = match error {
            ResourceCommitTransitionError::Recoverable(error) => (
                error.failure().failure().clone(),
                PreparationState::Commit(error),
            ),
            ResourceCommitTransitionError::Poisoned(error) => (
                error.failure().failure().clone(),
                PreparationState::Poisoned(error),
            ),
        };
        Self {
            state: Some(state),
            failure,
            poison: None,
            retirement_started: false,
        }
    }

    /// Immutable first preparation cause, independent of later cleanup errors.
    pub fn failure(&self) -> &FailureEnvelope {
        &self.failure
    }
    pub fn is_retired(&self) -> bool {
        self.state.is_none() && self.poison.is_none()
    }

    pub fn retire_in_place(&mut self) -> Result<(), FailureEnvelope> {
        self.retirement_started = true;
        if let Some(poison) = &self.poison {
            return Err(poison.clone());
        }
        match catch_unwind(AssertUnwindSafe(|| self.advance())) {
            Ok(result) => result,
            Err(payload) => {
                let message = if let Some(message) = payload.downcast_ref::<String>() {
                    message.clone()
                } else if let Some(message) = payload.downcast_ref::<&str>() {
                    (*message).to_owned()
                } else {
                    "non-string preparation retirement panic".to_owned()
                };
                let mut message = message
                    .chars()
                    .filter(|c| !c.is_control() || *c == ' ')
                    .take(1024)
                    .collect::<String>();
                if message.trim().is_empty() {
                    message.push_str("preparation retirement panicked");
                }
                let failure = core_resource_failure("preparation_retirement_panic", message, false);
                self.poison = Some(failure.clone());
                Err(failure)
            }
        }
    }

    fn advance(&mut self) -> Result<(), FailureEnvelope> {
        loop {
            match self.state.as_mut() {
                None => return Ok(()),
                Some(PreparationState::Reserve(error)) => {
                    error
                        .recover_in_place()
                        .map_err(|error| error.failure().clone())?;
                    let transaction = error
                        .transaction
                        .as_mut()
                        .expect("reserve owner retains transaction");
                    // Reverse compensation returns to New; there are no live
                    // reservations to roll back and no allocation to release.
                    // Do not fabricate a RolledBack ledger for this state.
                    if transaction
                        .states
                        .iter()
                        .any(|state| *state != ResourceTransactionState::New)
                        || transaction
                            .pending_allocations
                            .iter()
                            .any(|slot| slot.borrow().is_some())
                        || transaction.lease.as_ref().is_none_or(|lease| {
                            lease.slots.iter().any(|slot| slot.buffer.is_some())
                        })
                        || transaction.pending_failure.is_some()
                    {
                        return Err(core_resource_failure(
                            "unreserved_retirement_state",
                            "reserve recovery retained an unexpected resource state",
                            false,
                        ));
                    }
                    if let Some(mut claim) = transaction.capacity_claim.take() {
                        claim.release();
                    }
                    transaction.finalized = true;
                    self.state = None;
                }
                Some(PreparationState::Poisoned(error)) => {
                    error
                        .reconcile_in_place()
                        .map_err(|error| error.failure().clone())?;
                    let recovered = ResourcePrepareTransitionError {
                        transaction: error.transaction.take(),
                        failure: error.failure.clone(),
                    };
                    self.state = Some(PreparationState::Commit(recovered));
                }
                Some(PreparationState::Commit(error)) => {
                    error
                        .recover_in_place()
                        .map_err(|error| error.failure().clone())?;
                    let transaction = error
                        .transaction
                        .take()
                        .expect("commit recovery retains transaction");
                    self.state = Some(PreparationState::Reserved(Some(transaction)));
                }
                Some(PreparationState::Reserved(owner)) => {
                    let transaction = owner
                        .as_mut()
                        .expect("rollback retains its original transaction");
                    match transaction.rollback_in_place() {
                        Ok(receipt) => {
                            let terminal = owner
                                .take()
                                .expect("rollback retains transaction")
                                .advance::<TransactionRolledBack>(Some(receipt));
                            self.state = None;
                            drop(terminal);
                        }
                        Err(failure) => {
                            let transaction = owner.take();
                            let error = failure.failure().clone();
                            self.state = Some(PreparationState::Rollback(
                                ResourceRollbackTransitionError {
                                    transaction,
                                    failure,
                                },
                            ));
                            return Err(error);
                        }
                    }
                }
                Some(PreparationState::Rollback(error)) => {
                    let receipt = error
                        .complete_in_place()
                        .map_err(|error| error.failure().clone())?;
                    let terminal = error
                        .transaction
                        .take()
                        .expect("rollback recovery retains transaction")
                        .advance::<TransactionRolledBack>(Some(receipt));
                    self.state = None;
                    drop(terminal);
                }
            }
        }
    }
}

impl<D: ResourceTransactionDriver + 'static> Drop for ResourcePreparationRetirement<D> {
    fn drop(&mut self) {
        if !self.retirement_started {
            let _ = self.retire_in_place();
        }
        let Some(state) = self.state.take() else {
            return;
        };
        let original = Self {
            state: Some(state),
            failure: self.failure.clone(),
            poison: self.poison.take(),
            retirement_started: true,
        };
        defer_device_cleanup(
            crate::static_initialization::static_initialization_cleanup_domain(),
            DeferredPreparationRetirement { original },
        );
    }
}

struct DeferredPreparationRetirement<D: ResourceTransactionDriver + 'static> {
    original: ResourcePreparationRetirement<D>,
}
impl<D: ResourceTransactionDriver + 'static> DeferredDeviceCleanupTask
    for DeferredPreparationRetirement<D>
{
    fn try_cleanup(&mut self) -> DeferredDeviceCleanupDisposition {
        match self.original.retire_in_place() {
            Ok(()) => DeferredDeviceCleanupDisposition::Completed,
            Err(_) if self.original.poison.is_some() => {
                DeferredDeviceCleanupDisposition::Quarantined
            }
            Err(_) => DeferredDeviceCleanupDisposition::Retryable,
        }
    }
}

#[cfg(test)]
mod tests;
