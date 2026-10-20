//! Admission and terminal observations for the token execution protocol.
//!
//! These identities do not own model memory. A held close observes the original
//! executor's recovery work; dropping the observer cannot cancel that work.

use std::{fmt, future::Future, pin::Pin};

use ferrum_interfaces::adapter::LoraAdapterSelection;
pub use ferrum_types::AllocatorCapacityDomain;
use ferrum_types::Result;

use super::{
    invalid, ExecutionFrontier, ExecutionRecoveryKey, ExecutorSequence, ExecutorSequenceIssuer,
};
use crate::model_executor::{ExecutorPrefillAdmission, ExecutorPrefillMaintenanceDeferral};

/// Immutable semantic binding, retained once at admission by the executor.
#[derive(Debug, Clone, Copy)]
pub struct ExecutionAdmission<'a> {
    pub context: ExecutorPrefillAdmission<'a>,
    pub adapter: Option<LoraAdapterSelection<'a>>,
}

impl ExecutionAdmission<'_> {
    pub fn validate(&self) -> Result<()> {
        self.context.validate()?;
        if self
            .adapter
            .is_some_and(|adapter| adapter.name.trim().is_empty() || adapter.path.trim().is_empty())
        {
            return Err(invalid(
                "adapter admission requires a non-empty name and path",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub enum TokenAdmissionOutcome {
    /// Issued only after actual admission and publication of the retained state.
    Admitted(ExecutionFrontier),
    Deferred(ferrum_runtime::AdmissionDeferred),
    /// A real allocator fit rejection without native coordinator epochs.
    /// Retry only after an actual physical release notification. This snapshot
    /// is diagnostic evidence, never a reservation or permission to execute.
    CapacityDeferred(TokenAdmissionCapacityDeferral),
    MaintenanceDeferred(ExecutorPrefillMaintenanceDeferral),
    PermanentRejected(ferrum_runtime::AdmissionRejected),
}

/// An actual allocator's fit rejection, used both at admission and when an
/// existing sequence needs more KV blocks. Serialization records diagnostics;
/// construction still requires the checked shortfall below. The domain binds
/// the observation to its allocator, without granting native permissions.
#[derive(Debug, Clone, serde::Serialize)]
pub struct TokenAdmissionCapacityDeferral {
    domain: AllocatorCapacityDomain,
    required_blocks: usize,
    snapshot: crate::model_executor::KvSlotCapacitySnapshot,
}

/// Read-only, pressure-path observation of one physical allocator. Candidate
/// indices refer to the caller's exact external incarnations; they never name
/// the blocked owner. A contributor may release only a shared reference by
/// itself. `reclaimable_blocks` counts blocks freed by the complete candidate
/// set, once, after all its real owners reach Ended. This grants neither a
/// reservation nor permission to release an incarnation.
#[derive(Debug, Clone)]
pub struct TokenAllocatorReclaim {
    pub snapshot: crate::model_executor::KvSlotCapacitySnapshot,
    pub reclaimable_blocks: usize,
    pub contributing_candidates: Vec<usize>,
}

impl TokenAdmissionCapacityDeferral {
    /// Called by the actual model owner after its allocator rejected growth.
    /// Impossible total-capacity requests must fail, rather than wait forever.
    pub fn new(
        domain: AllocatorCapacityDomain,
        required_blocks: usize,
        snapshot: crate::model_executor::KvSlotCapacitySnapshot,
    ) -> Result<Self> {
        if snapshot.block_size == 0
            || snapshot.free_blocks > snapshot.total_blocks
            || required_blocks <= snapshot.free_blocks
            || required_blocks > snapshot.total_blocks
        {
            return Err(invalid(
                "capacity deferral requires a satisfiable physical block shortfall",
            ));
        }
        Ok(Self {
            domain,
            required_blocks,
            snapshot,
        })
    }

    pub fn domain(&self) -> &AllocatorCapacityDomain {
        &self.domain
    }

    pub fn required_blocks(&self) -> usize {
        self.required_blocks
    }

    pub fn snapshot(&self) -> &crate::model_executor::KvSlotCapacitySnapshot {
        &self.snapshot
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionEndReason {
    /// Product usage; these counts are not inferred from allocated KV or frames.
    Completed {
        input_tokens: usize,
        output_tokens: usize,
    },
    Cancelled,
}

/// The implementation attests that the original incarnation reached a checked
/// terminal state. This is not a native release authority or a success receipt
/// for any previously interrupted model work.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExecutionEnded {
    sequence: ExecutorSequence,
}

impl ExecutionEnded {
    pub fn sequence(&self) -> &ExecutorSequence {
        &self.sequence
    }
}

impl ExecutorSequence {
    /// Recover a terminal fact explicitly published by the actual executor.
    /// A missing registry entry, closed client, or dropped future cannot create
    /// this fact. It owns no device resources and grants no retry permission.
    pub fn observed_end(&self) -> Option<ExecutionEnded> {
        self.identity
            .ended
            .load(std::sync::atomic::Ordering::Acquire)
            .then(|| ExecutionEnded {
                sequence: self.clone(),
            })
    }
}

/// A checked terminal observation, independent of whether the operation
/// succeeded. The original executor must establish `ended` before constructing
/// this result; an operation error never creates a terminal fact.
#[derive(Debug)]
#[must_use = "terminal observation must preserve both the ended identity and operation error"]
pub struct ExecutionTerminalOutcome {
    pub ended: ExecutionEnded,
    pub error: Option<ferrum_types::FerrumError>,
}

impl ExecutionTerminalOutcome {
    pub fn ended(ended: ExecutionEnded) -> Self {
        Self { ended, error: None }
    }

    pub fn with_error(mut self, error: ferrum_types::FerrumError) -> Self {
        self.error.get_or_insert(error);
        self
    }
}

type CloseFuture = Pin<Box<dyn Future<Output = Result<ExecutionTerminalOutcome>> + Send + 'static>>;

/// Cold-path observer obtained before returning Held, so completion cannot be
/// lost when the original registry entry is removed. The executor/reaper keeps
/// the physical owner independently of this observer's lifetime.
#[must_use = "held execution must be tracked until closure or reported as unreclaimed"]
pub struct ExecutionHeld {
    key: ExecutionRecoveryKey,
    future: CloseFuture,
}

impl fmt::Debug for ExecutionHeld {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ExecutionHeld")
            .field("key", &self.key)
            .finish_non_exhaustive()
    }
}

impl ExecutionHeld {
    pub fn key(&self) -> &ExecutionRecoveryKey {
        &self.key
    }

    /// `Err` means this observation did not prove closure. An `Ok` terminal
    /// result may still carry an operation error; that error must not discard
    /// its checked ended identity or be reported as successful model work.
    pub async fn wait(self) -> Result<ExecutionTerminalOutcome> {
        let terminal = self.future.await?;
        if terminal.ended.sequence() != self.key.sequence() {
            return Err(invalid("close observer returned a different incarnation"));
        }
        Ok(terminal)
    }
}

#[derive(Debug)]
pub enum ExecutionCloseDisposition {
    Ended(ExecutionEnded),
    Held(ExecutionHeld),
}

/// Resource closure and operation failure are independent facts. For example,
/// a terminal receipt can coexist with a failed required event publication.
#[derive(Debug)]
#[must_use = "close outcome must preserve both resource disposition and any operation error"]
pub struct ExecutionCloseOutcome {
    pub disposition: ExecutionCloseDisposition,
    pub error: Option<ferrum_types::FerrumError>,
}

impl ExecutionCloseOutcome {
    pub fn ended(ended: ExecutionEnded) -> Self {
        Self {
            disposition: ExecutionCloseDisposition::Ended(ended),
            error: None,
        }
    }

    pub fn held(held: ExecutionHeld) -> Self {
        Self {
            disposition: ExecutionCloseDisposition::Held(held),
            error: None,
        }
    }

    pub fn with_error(mut self, error: ferrum_types::FerrumError) -> Self {
        self.error.get_or_insert(error);
        self
    }
}

impl ExecutorSequenceIssuer {
    /// Attest closure only after the actual owner proved quiescence and retired
    /// the sequence. Closing flags or missing registry entries are insufficient.
    pub fn ended(&self, sequence: &ExecutorSequence) -> Result<ExecutionEnded> {
        self.require_sequence(sequence)?;
        sequence
            .identity
            .ended
            .store(true, std::sync::atomic::Ordering::Release);
        Ok(ExecutionEnded {
            sequence: sequence.clone(),
        })
    }

    pub fn held<F>(&self, key: ExecutionRecoveryKey, future: F) -> Result<ExecutionHeld>
    where
        F: Future<Output = Result<ExecutionTerminalOutcome>> + Send + 'static,
    {
        self.require_sequence(key.sequence())?;
        Ok(ExecutionHeld {
            key,
            future: Box::pin(future),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::{channel::oneshot, executor::block_on};

    #[test]
    fn published_end_survives_owner_removal_without_closing_another_incarnation() {
        let issuer = ExecutorSequenceIssuer::new();
        let original = issuer.issue_sequence().unwrap();
        let caller = original.clone();
        let replacement = issuer.issue_sequence().unwrap();
        assert!(caller.observed_end().is_none());
        assert!(ExecutorSequenceIssuer::new().ended(&original).is_err());
        assert!(caller.observed_end().is_none());
        drop(issuer.ended(&original).unwrap());
        drop(original);
        drop(issuer);
        assert_eq!(caller.observed_end().unwrap().sequence(), &caller);
        assert!(replacement.observed_end().is_none());
    }

    #[test]
    fn held_observer_keeps_completion_published_before_waiting() {
        let issuer = ExecutorSequenceIssuer::new();
        let sequence = issuer.issue_sequence().unwrap();
        let (sender, receiver) = oneshot::channel();
        let held = issuer
            .held(issuer.recovery_key(&sequence).unwrap(), async move {
                receiver
                    .await
                    .map_err(|_| invalid("terminal publisher disappeared"))
            })
            .unwrap();
        sender
            .send(ExecutionTerminalOutcome::ended(
                issuer.ended(&sequence).unwrap(),
            ))
            .unwrap();
        let terminal = block_on(held.wait()).unwrap();
        assert_eq!(terminal.ended.sequence(), &sequence);
        assert!(terminal.error.is_none());
    }

    #[test]
    fn held_terminal_error_preserves_the_same_ended_identity() {
        let issuer = ExecutorSequenceIssuer::new();
        let sequence = issuer.issue_sequence().unwrap();
        let (sender, receiver) = oneshot::channel();
        let held = issuer
            .held(issuer.recovery_key(&sequence).unwrap(), async move {
                receiver
                    .await
                    .map_err(|_| invalid("terminal publisher disappeared"))
            })
            .unwrap();
        sender
            .send(
                ExecutionTerminalOutcome::ended(issuer.ended(&sequence).unwrap()).with_error(
                    ferrum_types::FerrumError::backend("terminal operation failed"),
                ),
            )
            .unwrap();
        let terminal = block_on(held.wait()).expect("operation failure must not erase closure");
        assert_eq!(terminal.ended.sequence(), &sequence);
        assert!(matches!(terminal.error,
            Some(ferrum_types::FerrumError::Backend { message })
                if message == "terminal operation failed"));
    }

    #[test]
    fn missing_or_wrong_terminal_publication_does_not_prove_closure() {
        let issuer = ExecutorSequenceIssuer::new();
        let sequence = issuer.issue_sequence().unwrap();
        let other = issuer.issue_sequence().unwrap();
        let wrong = issuer.ended(&other).unwrap();
        let held = issuer
            .held(issuer.recovery_key(&sequence).unwrap(), async {
                Ok(ExecutionTerminalOutcome::ended(wrong).with_error(
                    ferrum_types::FerrumError::backend("foreign terminal failure"),
                ))
            })
            .unwrap();
        assert!(block_on(held.wait()).is_err());

        let (sender, receiver) = oneshot::channel();
        let held = issuer
            .held(issuer.recovery_key(&sequence).unwrap(), async move {
                receiver
                    .await
                    .map_err(|_| invalid("terminal publisher disappeared"))
            })
            .unwrap();
        drop(sender);
        assert!(block_on(held.wait()).is_err());
    }

    #[test]
    fn cloned_issuer_preserves_domain_but_never_reuses_incarnation() {
        let issuer = ExecutorSequenceIssuer::new();
        let publisher = issuer.clone();
        let first = issuer.issue_sequence().unwrap();
        let second = publisher.issue_sequence().unwrap();
        assert_ne!(first, second);
        assert_eq!(publisher.ended(&first).unwrap().sequence(), &first);
        assert!(ExecutorSequenceIssuer::new().ended(&first).is_err());
    }

    #[test]
    fn physical_capacity_wait_requires_a_shortfall_that_release_can_resolve() {
        let domain = AllocatorCapacityDomain::new();
        let snapshot = crate::model_executor::KvSlotCapacitySnapshot {
            block_size: 16,
            total_blocks: 8,
            free_blocks: 2,
        };
        let wait =
            TokenAdmissionCapacityDeferral::new(domain.clone(), 3, snapshot.clone()).unwrap();
        assert_eq!(wait.domain(), &domain);
        assert_eq!(wait.required_blocks(), 3);
        assert_eq!(wait.snapshot(), &snapshot);
        // Neither an already-fitting request nor one larger than the entire
        // physical pool can be made runnable by waiting for another release.
        assert!(TokenAdmissionCapacityDeferral::new(domain.clone(), 2, snapshot.clone()).is_err());
        assert!(TokenAdmissionCapacityDeferral::new(domain, 9, snapshot).is_err());
    }
}
