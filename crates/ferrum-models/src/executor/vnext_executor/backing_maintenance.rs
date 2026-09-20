//! Bound allocator work without treating growth of a new pool as a stalled retry.

use std::collections::BTreeSet;

use super::{prefix_cache::PrefixPressureMaintenance, DynamicPoolGrowthBatchReceipt};

pub(super) fn allows_backing_attempt(
    prefix: &PrefixPressureMaintenance,
    attempts: u32,
    receipts: &[DynamicPoolGrowthBatchReceipt],
) -> bool {
    // These receipts belong to this admission call and come from successful
    // ordinary maintenance. A later pool can become the next blocker only
    // after an earlier one grows. Give each such pool one continuation, while
    // repeated growth of the same pool and epoch-only retries remain bounded.
    // The bound is the original budget plus this immutable plan's pool count;
    // callers must retain the same-plan, current-call receipt provenance.
    // This grants another probe, never allocation or execution authority.
    let attempts = attempts_without_new_pool_growth(
        attempts,
        receipts.iter().flat_map(|receipt| {
            receipt
                .growths()
                .iter()
                .map(|growth| (growth.pool_id().as_str(), growth.chunk_bytes()))
        }),
    );
    prefix.allows_backing_attempt(attempts)
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
}
