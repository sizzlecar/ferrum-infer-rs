//! Admission and terminal observations for the token execution protocol.
//!
//! These identities do not own model memory. A held close observes the original
//! executor's recovery work; dropping the observer cannot cancel that work.

use std::{fmt, future::Future, pin::Pin};

use ferrum_interfaces::adapter::LoraAdapterSelection;
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
    MaintenanceDeferred(ExecutorPrefillMaintenanceDeferral),
    PermanentRejected(ferrum_runtime::AdmissionRejected),
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

type CloseFuture = Pin<Box<dyn Future<Output = Result<ExecutionEnded>> + Send + 'static>>;

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

    /// An error means closure has not been proved. It never grants permission
    /// to release native resources or retry the original work.
    pub async fn wait(self) -> Result<ExecutionEnded> {
        let ended = self.future.await?;
        if ended.sequence() != self.key.sequence() {
            return Err(invalid("close observer returned a different incarnation"));
        }
        Ok(ended)
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
        Ok(ExecutionEnded {
            sequence: sequence.clone(),
        })
    }

    pub fn held<F>(&self, key: ExecutionRecoveryKey, future: F) -> Result<ExecutionHeld>
    where
        F: Future<Output = Result<ExecutionEnded>> + Send + 'static,
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
        sender.send(issuer.ended(&sequence).unwrap()).unwrap();
        assert_eq!(block_on(held.wait()).unwrap().sequence(), &sequence);
    }

    #[test]
    fn missing_or_wrong_terminal_publication_does_not_prove_closure() {
        let issuer = ExecutorSequenceIssuer::new();
        let sequence = issuer.issue_sequence().unwrap();
        let other = issuer.issue_sequence().unwrap();
        let wrong = issuer.ended(&other).unwrap();
        let held = issuer
            .held(issuer.recovery_key(&sequence).unwrap(), async { Ok(wrong) })
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
}
