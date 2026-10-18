use super::*;
use crate::completion::StateTransferSweepEntry;
use crate::{CompletionSweepEntry, CompletionSweepObservation, CompletionSweepReceipt};

impl<R: DeviceRuntime> CompletionReaper<R> {
    /// Runs a bounded sweep and blocking recovery for native checkpoint results
    /// abandoned by their public access handles. Call from the existing device
    /// completion worker, not a request/admission thread. Live consumers are
    /// never drained or discarded by this operation.
    ///
    /// Failed drains remain in the same reaper's quarantine for a later retry.
    /// Successful cleanup discards results; it cannot install a restore frontier
    /// or turn drain evidence into a successful capture.
    pub fn recover_abandoned_checkpoints(
        &self,
        maximum_slots: usize,
    ) -> Result<CompletionSweepReceipt, VNextError> {
        let mut receipt = CompletionSweepReceipt {
            entries: Vec::new(),
            state_transfers: Vec::new(),
            retained_after: 0,
            quarantined_after: 0,
        };
        // A generic poll consumes ordinary operation terminal receipts. Those
        // still belong to their waiting readers, even when checkpoint cleanup
        // shares the same worker and reaper.
        for slot_id in self.bounded_slot_snapshot(maximum_slots)? {
            match self.recover_abandoned_state_transfer_slot(slot_id) {
                Ok(Some(observation)) => receipt.state_transfers.push(StateTransferSweepEntry {
                    slot_id,
                    observation,
                }),
                Ok(None) => {}
                Err(error) => receipt.entries.push(CompletionSweepEntry {
                    slot_id,
                    observation: CompletionSweepObservation::Failed(error),
                }),
            }
        }
        receipt.retained_after = self.retained_count();
        receipt.quarantined_after = self.quarantined_count();
        Ok(receipt)
    }
}
