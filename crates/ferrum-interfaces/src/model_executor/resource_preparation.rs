//! Resource preparation allocates real backing without executing model work.
//! A receipt is never a future cost or initialized model-state proof.
use crate::execution_cost::ActualWaveKind;
use ferrum_types::{FerrumError, Result};
use std::num::NonZeroUsize;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExecutorResourcePreparationRequest {
    participants: NonZeroUsize,
    sequence_tokens: NonZeroUsize,
    tokens_per_sequence: NonZeroUsize,
    kind: ActualWaveKind,
}

impl ExecutorResourcePreparationRequest {
    pub fn new(
        participants: NonZeroUsize,
        sequence_tokens: NonZeroUsize,
        tokens_per_sequence: NonZeroUsize,
        kind: ActualWaveKind,
    ) -> Result<Self> {
        if tokens_per_sequence > sequence_tokens
            || participants
                .get()
                .checked_mul(tokens_per_sequence.get())
                .is_none()
            || !matches!(kind, ActualWaveKind::Prefill | ActualWaveKind::Decode)
            || (kind == ActualWaveKind::Decode && tokens_per_sequence.get() != 1)
        {
            return Err(FerrumError::invalid_request(
                "invalid resource preparation shape",
            ));
        }
        Ok(Self {
            participants,
            sequence_tokens,
            tokens_per_sequence,
            kind,
        })
    }
    pub fn participants(self) -> usize {
        self.participants.get()
    }
    pub fn sequence_tokens(self) -> usize {
        self.sequence_tokens.get()
    }
    pub fn tokens_per_sequence(self) -> usize {
        self.tokens_per_sequence.get()
    }
    pub fn kind(self) -> ActualWaveKind {
        self.kind
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutorResourcePreparationOutcome {
    /// Every disposable owner was retired after the real allocator accepted
    /// the requested Sequence, Step and Invocation backing. Recapture is still
    /// required; resident bytes may be reclaimed and ranges may not fit.
    Prepared,
    /// The original capacity/maintenance limits could not prepare the shape.
    /// No cost authority is granted; fresh capture retains the actual gap.
    Unavailable,
    /// Default for executors without a resource-only preparation protocol.
    Unsupported,
}

/// Completed disposable owner groups, including partial preparation. Neither
/// this receipt nor its count grants resource or cost authority.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExecutorResourcePreparationReceipt {
    pub outcome: ExecutorResourcePreparationOutcome,
    pub prepared_participants: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    fn n(value: usize) -> NonZeroUsize {
        NonZeroUsize::new(value).unwrap()
    }
    #[test]
    fn resource_preparation_keeps_sequence_frontier_separate_from_wave_work() {
        for (rows, frontier, chunk) in [(1, 4095, 2048), (3, 4095, 682), (8, 37, 4)] {
            let request = ExecutorResourcePreparationRequest::new(
                n(rows),
                n(frontier),
                n(chunk),
                ActualWaveKind::Prefill,
            )
            .unwrap();
            assert_eq!(request.sequence_tokens(), frontier);
            assert_eq!(request.tokens_per_sequence(), chunk);
            assert_eq!(request.participants(), rows);
            assert!(ExecutorResourcePreparationRequest::new(
                n(rows),
                n(frontier),
                n(1),
                ActualWaveKind::Decode
            )
            .is_ok());
        }
        assert!(
            ExecutorResourcePreparationRequest::new(n(1), n(4), n(5), ActualWaveKind::Prefill)
                .is_err()
        );
        assert!(
            ExecutorResourcePreparationRequest::new(n(1), n(4), n(2), ActualWaveKind::Decode)
                .is_err()
        );
        assert!(ExecutorResourcePreparationRequest::new(
            n(usize::MAX),
            n(2),
            n(2),
            ActualWaveKind::Prefill
        )
        .is_err());
        assert!(
            ExecutorResourcePreparationRequest::new(n(1), n(1), n(1), ActualWaveKind::Restore)
                .is_err()
        );
    }
}
