//! Publication of restored state before constructing a prefill batch.

use super::PrefixCaptureLease;
use crate::token_execution::{ExecutionFrontier, ExecutorSequence};
use ferrum_types::{FerrumError, RequestId, Result, TokenId};
use std::{fmt, sync::Arc};

/// Observation only: neither an index hit nor a completed copy authorizes use
/// of restored state. `Restored` is emitted after publication acknowledgement.
#[derive(Debug, serde::Serialize)]
pub struct PrefixRestoreObservation<'a> {
    pub request_id: &'a RequestId,
    pub source: PrefixRestoreSource,
    pub decision: PrefixRestoreDecision<'a>,
}

#[derive(Debug, Clone, Copy, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PrefixRestoreSource {
    Index,
    Rendezvous,
}

/// Contains lengths and typed resource evidence, never prompt/token contents.
#[derive(Debug, serde::Serialize)]
#[serde(tag = "outcome", rename_all = "snake_case")]
pub enum PrefixRestoreDecision<'a> {
    NoReusableEntry,
    SequenceCapacityNotReady {
        candidate_prefix_tokens: usize,
        capacity: &'a super::ExecutorExecutionCapacityDeferral,
    },
    RequestStateNotReady {
        candidate_prefix_tokens: usize,
        request_state: &'a super::ExecutorRequestStateDeferral,
    },
    NativeRestoreSkipped {
        candidate_prefix_tokens: usize,
    },
    Restored {
        candidate_prefix_tokens: usize,
    },
}

/// The exact input already admitted by the plan runtime. Restoration does not
/// replace admission or authorize a different request incarnation.
#[derive(Debug, Clone, Copy)]
pub struct PlanRuntimePrefixRestoreInput<'a> {
    pub request_id: &'a RequestId,
    pub input_tokens: &'a [TokenId],
    pub maximum_sequence_tokens: usize,
    /// An authenticated ready source retained by an earlier rendezvous. This
    /// bypasses index lookup, never target admission or native compatibility.
    pub checkpoint: Option<&'a dyn PrefixCaptureLease>,
    /// Exact retained source from a prior capacity deferral; never reselect the index.
    pub retry: Option<&'a PlanRuntimePrefixRestoreDeferral>,
}

/// A capacity deferral submits no restore and retains one immutable source. It
/// grants neither target admission nor permission to wait without a progress
/// source. Dropping it releases the optional checkpoint pin.
#[derive(Debug)]
pub struct PlanRuntimePrefixRestoreDeferral {
    capacity: super::ExecutorExecutionCapacityDeferral,
    checkpoint: Arc<dyn PrefixCaptureLease>,
    source: PrefixRestoreSource,
}

impl PlanRuntimePrefixRestoreDeferral {
    pub fn new(
        capacity: super::ExecutorExecutionCapacityDeferral,
        checkpoint: Arc<dyn PrefixCaptureLease>,
        source: PrefixRestoreSource,
    ) -> Self {
        Self {
            capacity,
            checkpoint,
            source,
        }
    }

    pub fn capacity(&self) -> &super::ExecutorExecutionCapacityDeferral {
        &self.capacity
    }

    pub fn checkpoint(&self) -> &Arc<dyn PrefixCaptureLease> {
        &self.checkpoint
    }

    pub fn source(&self) -> PrefixRestoreSource {
        self.source
    }
}

#[derive(Debug)]
#[must_use = "restore publication or a retained capacity deferral must be handled"]
pub enum PlanRuntimePrefixRestoreOutcome {
    Unavailable,
    Restored(PlanRuntimePrefixRestoreOutput),
    Deferred(PlanRuntimePrefixRestoreDeferral),
}

/// Independently restored state whose execution gate remains closed until the
/// caller validates its admitted incarnation and acknowledges publication.
///
/// The callback owns the native publication guard. Dropping this value must
/// cancel that exact target; a request ID lookup cannot replace its ownership.
/// Source liveness, physical extent and target identity are validated by the
/// callback. These outer fields are observations, not native authorization.
/// At least one remaining prompt token must execute before sampling.
#[must_use = "acknowledge the matching admitted frontier, or drop to cancel the restored target"]
pub struct PlanRuntimePrefixRestoreOutput {
    request_id: RequestId,
    restored_tokens: usize,
    prompt_tokens: usize,
    sequence: ExecutorSequence,
    acknowledge: Box<dyn FnOnce() -> Result<PrefixRestoreAcknowledgement> + Send>,
}

/// The actual publication survives a subsequent observation failure. Consumers
/// retain this frontier before handling `error`; they must not cold-replay it.
#[derive(Debug)]
#[must_use = "retain the actual frontier and handle any post-publication error"]
pub struct PrefixRestoreAcknowledgement {
    pub frontier: ExecutionFrontier,
    pub error: Option<FerrumError>,
}

impl PrefixRestoreAcknowledgement {
    pub fn new(frontier: ExecutionFrontier) -> Self {
        Self {
            frontier,
            error: None,
        }
    }
}

impl PlanRuntimePrefixRestoreOutput {
    /// Retain the original publication callback and exact token incarnation.
    /// Rejected construction drops the callback and its native rollback guard.
    pub fn new_tokens(
        request_id: RequestId,
        restored_tokens: usize,
        prompt_tokens: usize,
        sequence: ExecutorSequence,
        acknowledge: impl FnOnce() -> Result<PrefixRestoreAcknowledgement> + Send + 'static,
    ) -> Result<Self> {
        let output = Self {
            request_id,
            restored_tokens,
            prompt_tokens,
            sequence,
            acknowledge: Box::new(acknowledge),
        };
        output.validate_for(output.request_id(), prompt_tokens, output.token_sequence())?;
        Ok(output)
    }

    pub fn token_sequence(&self) -> &ExecutorSequence {
        &self.sequence
    }

    pub fn request_id(&self) -> &RequestId {
        &self.request_id
    }

    pub fn restored_tokens(&self) -> usize {
        self.restored_tokens
    }

    /// Compare against the caller's actual admitted incarnation before opening
    /// the native gate. Matching request text alone cannot establish this link.
    pub fn validate_for(
        &self,
        request_id: &RequestId,
        prompt_tokens: usize,
        sequence: &ExecutorSequence,
    ) -> Result<()> {
        if self.request_id() != request_id
            || self.prompt_tokens != prompt_tokens
            || self.token_sequence() != sequence
        {
            return Err(FerrumError::backend(
                "prefix restore publication does not match the admitted request incarnation",
            ));
        }
        if self.restored_tokens == 0 || self.restored_tokens >= prompt_tokens {
            return Err(FerrumError::backend(
                "prefix restore must leave a nonempty prompt suffix for execution",
            ));
        }
        if self.sequence.observed_end().is_some() {
            return Err(FerrumError::backend(
                "prefix restore publication belongs to an ended incarnation",
            ));
        }
        Ok(())
    }

    /// Open the original native publication gate and return its actual frontier.
    /// The callback must revalidate cancellation and the physical restored target.
    /// Pre-publication rejection is `Err`; a post-publication failure accompanies
    /// the actual frontier. Either failure requires closing, never cold replay.
    pub fn acknowledge_tokens(self) -> Result<PrefixRestoreAcknowledgement> {
        self.validate_for(&self.request_id, self.prompt_tokens, &self.sequence)?;
        let mut acknowledged = (self.acknowledge)()?;
        if acknowledged.frontier.sequence() != &self.sequence
            || acknowledged.frontier.consumed_tokens() != self.restored_tokens
        {
            let prior = acknowledged.error.take();
            acknowledged.error = Some(FerrumError::backend(match prior {
                Some(error) => format!(
                    "prefix acknowledgement returned a different incarnation or frontier; observation: {error}"
                ),
                None => "prefix acknowledgement returned a different incarnation or frontier".into(),
            }));
        }
        Ok(acknowledged)
    }
}

impl fmt::Debug for PlanRuntimePrefixRestoreOutput {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PlanRuntimePrefixRestoreOutput")
            .field("request_id", &self.request_id)
            .field("restored_tokens", &self.restored_tokens)
            .field("prompt_tokens", &self.prompt_tokens)
            .field("sequence", &self.sequence)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::token_execution::ExecutorSequenceIssuer;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct RetainedCallback(Arc<AtomicUsize>);

    impl Drop for RetainedCallback {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }

    fn publication(
        request: RequestId,
        sequence: ExecutorSequence,
        restored: usize,
        result: ExecutionFrontier,
        called: &Arc<AtomicUsize>,
        dropped: &Arc<AtomicUsize>,
    ) -> Result<PlanRuntimePrefixRestoreOutput> {
        let retained = RetainedCallback(Arc::clone(dropped));
        let called = Arc::clone(called);
        PlanRuntimePrefixRestoreOutput::new_tokens(request, restored, 8, sequence, move || {
            called.fetch_add(1, Ordering::SeqCst);
            drop(retained);
            Ok(PrefixRestoreAcknowledgement::new(result))
        })
    }

    #[test]
    fn publication_requires_the_admitted_request_and_incarnation() {
        let issuer = ExecutorSequenceIssuer::new();
        let sequence = issuer.issue_sequence().unwrap();
        let replacement = issuer.issue_sequence().unwrap();
        let request = RequestId::new();
        let called = Arc::new(AtomicUsize::new(0));
        let dropped = Arc::new(AtomicUsize::new(0));
        let output = publication(
            request.clone(),
            sequence.clone(),
            3,
            issuer.frontier(&sequence, 3).unwrap(),
            &called,
            &dropped,
        )
        .unwrap();
        assert!(output.validate_for(&request, 8, &replacement).is_err());
        assert!(output
            .validate_for(&RequestId::new(), 8, &sequence)
            .is_err());
        assert!(output.validate_for(&request, 9, &sequence).is_err());
        assert_eq!(called.load(Ordering::SeqCst), 0);
        output.validate_for(&request, 8, &sequence).unwrap();
        let acknowledged = output.acknowledge_tokens().unwrap();
        assert!(acknowledged.error.is_none());
        let frontier = acknowledged.frontier;
        assert_eq!(frontier.sequence(), &sequence);
        assert_eq!(frontier.consumed_tokens(), 3);
        assert_eq!(called.load(Ordering::SeqCst), 1);
        assert_eq!(dropped.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn publication_rejects_invalid_extent_without_opening_callback() {
        let issuer = ExecutorSequenceIssuer::new();
        for restored in [0, 8, 9] {
            let sequence = issuer.issue_sequence().unwrap();
            let called = Arc::new(AtomicUsize::new(0));
            let dropped = Arc::new(AtomicUsize::new(0));
            assert!(publication(
                RequestId::new(),
                sequence.clone(),
                restored,
                issuer.frontier(&sequence, restored).unwrap(),
                &called,
                &dropped,
            )
            .is_err());
            assert_eq!(called.load(Ordering::SeqCst), 0);
            assert_eq!(dropped.load(Ordering::SeqCst), 1);
        }
    }

    #[test]
    fn publication_rejects_conflicting_actual_frontier() {
        let issuer = ExecutorSequenceIssuer::new();
        let sequence = issuer.issue_sequence().unwrap();
        let replacement = issuer.issue_sequence().unwrap();
        for result in [
            issuer.frontier(&replacement, 3).unwrap(),
            issuer.frontier(&sequence, 4).unwrap(),
        ] {
            let called = Arc::new(AtomicUsize::new(0));
            let dropped = Arc::new(AtomicUsize::new(0));
            let actual = result.clone();
            let output = publication(
                RequestId::new(),
                sequence.clone(),
                3,
                result,
                &called,
                &dropped,
            )
            .unwrap();
            let acknowledged = output.acknowledge_tokens().unwrap();
            assert!(acknowledged.error.is_some());
            assert_eq!(
                acknowledged.frontier, actual,
                "post-publication rejection retains the exact returned frontier"
            );
            assert_eq!(called.load(Ordering::SeqCst), 1);
            assert_eq!(dropped.load(Ordering::SeqCst), 1);
        }
    }

    #[test]
    fn discarded_or_ended_publication_drops_callback_without_acknowledging() {
        let issuer = ExecutorSequenceIssuer::new();
        for ended in [false, true] {
            let sequence = issuer.issue_sequence().unwrap();
            let called = Arc::new(AtomicUsize::new(0));
            let dropped = Arc::new(AtomicUsize::new(0));
            let output = publication(
                RequestId::new(),
                sequence.clone(),
                3,
                issuer.frontier(&sequence, 3).unwrap(),
                &called,
                &dropped,
            )
            .unwrap();
            if ended {
                issuer.ended(&sequence).unwrap();
                assert!(output.acknowledge_tokens().is_err());
            } else {
                drop(output);
            }
            assert_eq!(called.load(Ordering::SeqCst), 0);
            assert_eq!(dropped.load(Ordering::SeqCst), 1);
        }
    }
}
