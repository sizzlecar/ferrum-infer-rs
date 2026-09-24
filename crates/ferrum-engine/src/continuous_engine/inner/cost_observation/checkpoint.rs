//! A bounded barrier in accepted-observation order, independent of model TTL.
use super::audit::TrainingAuditSnapshot;
use super::profile::EngineCostSnapshot;
use super::profile_export::{CostProfileCutPaths, CostProfileCutReceipt, ExportAuditSnapshot};
use std::sync::Arc;
use tokio::sync::oneshot;

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub(in crate::continuous_engine) enum CheckpointRequestError {
    #[error("a cost checkpoint already occupies the control slot")]
    Busy,
    #[error("cost training is closing")]
    Closing,
    #[error("cost observation ordinal is exhausted")]
    CounterExhausted,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub(in crate::continuous_engine) enum CheckpointError {
    #[error("cost training stopped before completing the checkpoint")]
    WorkerStopped,
}

/// This cut covers successfully queued observations, including observations the
/// trainer rejected. It does not attest to uninstrumented calls or queue losses.
/// Post-cut observations are excluded; imported seed evidence remains included.
pub(in crate::continuous_engine) struct FrozenCostCheckpoint {
    pub accepted_ordinal: u64,
    pub snapshot: Option<Arc<EngineCostSnapshot>>,
    pub training: TrainingAuditSnapshot,
    pub export: ExportAuditSnapshot,
    /// Independent immutable training evidence, not serialization of snapshot.
    pub profile_cut: Option<Result<CostProfileCutReceipt, String>>,
}

/// Dropping the waiter never withdraws the barrier or discards queued samples.
pub(in crate::continuous_engine) struct CostCheckpointWaiter {
    receiver: oneshot::Receiver<Result<FrozenCostCheckpoint, CheckpointError>>,
}

impl CostCheckpointWaiter {
    pub async fn wait(self) -> Result<FrozenCostCheckpoint, CheckpointError> {
        self.receiver
            .await
            .unwrap_or(Err(CheckpointError::WorkerStopped))
    }
}

pub(super) struct PendingCheckpoint {
    pub cutoff: u64,
    pub freezing: bool,
    pub export_paths: Option<CostProfileCutPaths>,
    reply: oneshot::Sender<Result<FrozenCostCheckpoint, CheckpointError>>,
}

impl PendingCheckpoint {
    pub fn new(
        cutoff: u64,
        export_paths: Option<CostProfileCutPaths>,
    ) -> (Self, CostCheckpointWaiter) {
        let (reply, receiver) = oneshot::channel();
        (
            Self {
                cutoff,
                freezing: false,
                export_paths,
                reply,
            },
            CostCheckpointWaiter { receiver },
        )
    }
    pub fn complete(self, value: Result<FrozenCostCheckpoint, CheckpointError>) {
        // Cancellation releases the returned Arc; it cannot stop training.
        let _ = self.reply.send(value);
    }
}

#[cfg(test)]
mod tests;
