//! Backend-neutral planning for reusable command segments.
//!
//! CUDA Graph handles remain backend-owned. This module owns the deterministic
//! cold-path policy that can be verified without a CUDA toolchain: segment
//! discovery, bounded admission, resident protection, and post-capture
//! eviction planning.

use std::ops::Range;

use ferrum_interfaces::vnext::DeviceCommandPhase;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ReusableExecutionSegment {
    range: Range<usize>,
}

impl ReusableExecutionSegment {
    pub(crate) fn range(&self) -> Range<usize> {
        self.range.clone()
    }
}

pub(crate) fn discover_reusable_segments(
    phases: &[DeviceCommandPhase],
    mut command_is_reusable: impl FnMut(usize) -> bool,
) -> Vec<ReusableExecutionSegment> {
    let mut segments = Vec::new();
    let mut index = 0;
    while index < phases.len() {
        if phases[index] != DeviceCommandPhase::Compute || !command_is_reusable(index) {
            index += 1;
            continue;
        }
        let start = index;
        index += 1;
        while index < phases.len()
            && phases[index] == DeviceCommandPhase::Compute
            && command_is_reusable(index)
        {
            index += 1;
        }
        segments.push(ReusableExecutionSegment {
            range: start..index,
        });
    }
    segments
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct BoundedReusableExecutionPlan<K> {
    admitted_misses: Vec<K>,
    capacity_deferred_misses: Vec<K>,
    eviction_order: Vec<K>,
    resident_count: usize,
    maximum_entries: usize,
}

impl<K: Copy + Eq> BoundedReusableExecutionPlan<K> {
    pub(crate) fn admitted_misses(&self) -> &[K] {
        &self.admitted_misses
    }

    pub(crate) fn capacity_deferred_misses(&self) -> &[K] {
        &self.capacity_deferred_misses
    }

    pub(crate) fn required_evictions(&self, successful_captures: usize) -> &[K] {
        let count = self
            .resident_count
            .saturating_add(successful_captures)
            .saturating_sub(self.maximum_entries);
        &self.eviction_order[..count.min(self.eviction_order.len())]
    }
}

pub(crate) fn plan_bounded_reusable_execution<K: Copy + Eq>(
    candidate_keys: &[K],
    resident_last_used: &[(K, u64)],
    rejected_keys: &[K],
    maximum_entries: usize,
) -> BoundedReusableExecutionPlan<K> {
    assert!(maximum_entries > 0, "reusable execution capacity is empty");
    assert!(
        resident_last_used.len() <= maximum_entries,
        "resident reusable execution cache exceeds its capacity"
    );

    let mut protected_residents = Vec::new();
    let mut unique_misses = Vec::new();
    for key in candidate_keys.iter().copied() {
        if resident_last_used
            .iter()
            .any(|(resident, _)| *resident == key)
        {
            push_unique(&mut protected_residents, key);
        } else if !rejected_keys.contains(&key) {
            push_unique(&mut unique_misses, key);
        }
    }

    let admission_limit = maximum_entries.saturating_sub(protected_residents.len());
    let split = admission_limit.min(unique_misses.len());
    let admitted_misses = unique_misses[..split].to_vec();
    let capacity_deferred_misses = unique_misses[split..].to_vec();

    let mut evictable = resident_last_used
        .iter()
        .copied()
        .filter(|(key, _)| !protected_residents.contains(key))
        .collect::<Vec<_>>();
    evictable.sort_by_key(|(_, last_used)| *last_used);
    let eviction_order = evictable.into_iter().map(|(key, _)| key).collect();

    BoundedReusableExecutionPlan {
        admitted_misses,
        capacity_deferred_misses,
        eviction_order,
        resident_count: resident_last_used.len(),
        maximum_entries,
    }
}

fn push_unique<K: Copy + Eq>(values: &mut Vec<K>, value: K) {
    if !values.contains(&value) {
        values.push(value);
    }
}

#[cfg(test)]
mod tests {
    use ferrum_interfaces::vnext::DeviceCommandPhase;

    use super::{discover_reusable_segments, plan_bounded_reusable_execution};

    #[test]
    fn segment_discovery_splits_every_eager_barrier_and_unkeyed_compute() {
        let phases = [
            DeviceCommandPhase::Initialization,
            DeviceCommandPhase::Compute,
            DeviceCommandPhase::Compute,
            DeviceCommandPhase::DynamicBinding,
            DeviceCommandPhase::Compute,
            DeviceCommandPhase::Compute,
            DeviceCommandPhase::Compute,
            DeviceCommandPhase::ResultBinding,
            DeviceCommandPhase::Compute,
        ];
        let reusable = [false, true, true, false, true, false, true, false, true];

        let ranges = discover_reusable_segments(&phases, |index| reusable[index])
            .into_iter()
            .map(|segment| segment.range())
            .collect::<Vec<_>>();

        assert_eq!(ranges, vec![1..3, 4..5, 6..7, 8..9]);
    }

    #[test]
    fn bounded_plan_protects_current_hits_and_evicts_stale_lru_only_after_success() {
        let plan = plan_bounded_reusable_execution(
            &[2_u8, 4, 5, 4],
            &[(1_u8, 10), (2, 20), (3, 5)],
            &[],
            4,
        );

        assert_eq!(plan.admitted_misses(), &[4, 5]);
        assert!(plan.capacity_deferred_misses().is_empty());
        assert!(plan.required_evictions(0).is_empty());
        assert!(plan.required_evictions(1).is_empty());
        assert_eq!(plan.required_evictions(2), &[3]);
    }

    #[test]
    fn bounded_plan_is_deterministic_for_rejections_and_capacity_overflow() {
        let plan = plan_bounded_reusable_execution(&[7_u8, 8, 9, 10, 8], &[(7_u8, 1)], &[9_u8], 2);

        assert_eq!(plan.admitted_misses(), &[8]);
        assert_eq!(plan.capacity_deferred_misses(), &[10]);
        assert!(plan.required_evictions(0).is_empty());
        assert!(plan.required_evictions(1).is_empty());
    }
}
