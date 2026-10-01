//! Bound allocator work without treating growth of a new pool as a stalled retry.

use std::collections::BTreeSet;

use super::{prefix_cache::PrefixPressureMaintenance, DynamicPoolGrowthBatchReceipt};

/// Per-call evidence ledger shared by Sequence, Step and Invocation admission.
#[derive(Default)]
pub(super) struct BackingMaintenanceProgress {
    receipts: Vec<DynamicPoolGrowthBatchReceipt>,
    reclamation_limit: Option<u32>,
    reclaimed_attempts: u32,
    last_reclamation: Option<(u64, u64)>,
}

impl BackingMaintenanceProgress {
    pub(super) fn allows_attempt<R: ferrum_interfaces::vnext::DeviceRuntime>(
        &mut self,
        prefix: &PrefixPressureMaintenance,
        attempts: u32,
        resources: &ferrum_interfaces::vnext::PlanRuntimeResources<R>,
    ) -> ferrum_types::Result<bool> {
        // Initialize only on the cold maintenance path, before any eviction.
        // Slot count is exact across pools and is never replenished this call.
        if self.reclamation_limit.is_none() {
            let slots = resources
                .lane_stable_slot_count()
                .map_err(|error| ferrum_types::FerrumError::backend(error.to_string()))?;
            self.reclamation_limit = Some(u32::try_from(slots).unwrap_or(u32::MAX));
        }
        Ok(prefix.allows_backing_attempt(self.attempts_without_progress(attempts)))
    }

    fn attempts_without_progress(&self, attempts: u32) -> u32 {
        attempts_without_new_pool_growth(
            attempts,
            self.receipts.iter().flat_map(|receipt| {
                receipt
                    .growths()
                    .iter()
                    .map(|growth| (growth.pool_id().as_str(), growth.chunk_bytes()))
            }),
        )
        .saturating_sub(self.reclaimed_attempts)
    }

    pub(super) fn push(&mut self, receipt: DynamicPoolGrowthBatchReceipt) {
        self.receipts.push(receipt);
    }

    pub(super) fn receipts(&self) -> &[DynamicPoolGrowthBatchReceipt] {
        &self.receipts
    }

    pub(super) fn record_reclamation(
        &mut self,
        receipt: &ferrum_interfaces::vnext::DynamicLaneSlotReclamation,
    ) {
        let epochs = receipt.epochs();
        self.record_reclamation_epoch(epochs.coordinator_id().get(), epochs.capacity_epoch());
    }

    fn record_reclamation_epoch(&mut self, coordinator: u64, epoch: u64) {
        let Some(limit) = self.reclamation_limit else {
            return;
        };
        // A cloned or stale receipt cannot grant another continuation. Receipts
        // arrive from the owning plan, synchronously, in this maintenance call.
        if self
            .last_reclamation
            .is_some_and(|(previous_coordinator, previous_epoch)| {
                coordinator != previous_coordinator || epoch <= previous_epoch
            })
        {
            return;
        }
        self.last_reclamation = Some((coordinator, epoch));
        self.reclaimed_attempts = self.reclaimed_attempts.saturating_add(1).min(limit);
    }
}

fn attempts_without_new_pool_growth<'a>(
    attempts: u32,
    growths: impl IntoIterator<Item = (&'a str, u64)>,
) -> u32 {
    let pools = growths
        .into_iter()
        .filter_map(|(pool, bytes)| (bytes != 0).then_some(pool))
        .collect::<BTreeSet<_>>();
    let progress = u32::try_from(pools.len()).unwrap_or(u32::MAX);
    attempts.saturating_sub(progress)
}

#[cfg(test)]
mod tests {
    use super::attempts_without_new_pool_growth;

    #[test]
    fn independent_pool_growth_can_reach_the_next_physical_blocker() {
        // Token-mask and state pools became ready on separate probes. Their
        // successful growth must not consume the budget for the scratch pool
        // that is first exposed by the following admission attempt.
        let growths = [("token-mask", 4096), ("state", 16384)];
        assert_eq!(attempts_without_new_pool_growth(2, growths), 0);
        // Once growth stops, subsequent attempts consume the normal budget.
        assert_eq!(attempts_without_new_pool_growth(4, growths), 2);
    }

    #[test]
    fn repeated_chunks_do_not_make_a_stalled_pool_retry_forever() {
        let growths = [("scratch", 1024), ("scratch", 2048), ("scratch", 4096)];
        assert_eq!(attempts_without_new_pool_growth(3, growths), 2);
    }

    #[test]
    fn epoch_changes_and_empty_growth_grant_no_continuation() {
        assert_eq!(attempts_without_new_pool_growth(2, []), 2);
        assert_eq!(attempts_without_new_pool_growth(2, [("scratch", 0)]), 2);
    }
    #[test]
    fn reclamation_progress_is_bounded_and_duplicate_receipts_grant_nothing() {
        let mut progress = super::BackingMaintenanceProgress {
            reclamation_limit: Some(3),
            ..Default::default()
        };
        progress.record_reclamation_epoch(7, 10);
        progress.record_reclamation_epoch(7, 10); // cloned receipt
        progress.record_reclamation_epoch(7, 9); // stale receipt
        progress.record_reclamation_epoch(8, 20); // another coordinator
        assert_eq!(progress.reclaimed_attempts, 1);
        progress.record_reclamation_epoch(7, 11);
        progress.record_reclamation_epoch(7, 12);
        // A concurrently created later slot cannot replenish the frozen bound.
        progress.record_reclamation_epoch(7, 13);
        assert_eq!(progress.reclaimed_attempts, 3);
        assert_eq!(progress.attempts_without_progress(4), 1);
        assert_eq!(progress.attempts_without_progress(5), 2);
    }

    #[test]
    fn reclamation_progress_without_a_captured_slot_budget_grants_nothing() {
        for limit in [None, Some(0)] {
            let mut progress = super::BackingMaintenanceProgress {
                reclamation_limit: limit,
                ..Default::default()
            };
            progress.record_reclamation_epoch(7, 1);
            assert_eq!(progress.reclaimed_attempts, 0);
        }
    }
}
