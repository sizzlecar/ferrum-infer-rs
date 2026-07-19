//! Backend-neutral planning for reusable command segments.
//!
//! CUDA Graph handles remain backend-owned. This module owns the deterministic
//! cold-path policy that can be verified without a CUDA toolchain: segment
//! discovery, bounded admission, resident protection, and post-capture
//! eviction planning.

use std::ops::Range;

use ferrum_interfaces::vnext::{
    DeviceCommandPhase, DeviceReusableExecutionPlan, DeviceReusableExecutionPreparation,
};

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

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
enum ReusableExecutionPreparationLifecycle {
    #[default]
    Unconfigured,
    Preparing(DeviceReusableExecutionPlan),
    Ready(DeviceReusableExecutionPlan),
}

/// Backend-neutral lifecycle accounting for cold reusable-execution
/// preparation. Backend handles remain outside this type.
#[derive(Debug, Default)]
pub(crate) struct ReusableExecutionPreparationTracker {
    lifecycle: ReusableExecutionPreparationLifecycle,
    captured_executables: u64,
    uploaded_executables: u64,
    capacity_deferred_executables: u64,
}

impl ReusableExecutionPreparationTracker {
    pub(crate) fn configure(
        &mut self,
        plan: DeviceReusableExecutionPlan,
        resident_executables: usize,
        rejected_executables: usize,
    ) -> Result<DeviceReusableExecutionPreparation, String> {
        if self.lifecycle != ReusableExecutionPreparationLifecycle::Unconfigured {
            return Err("reusable execution preparation was already configured".to_owned());
        }
        if resident_executables != 0 || rejected_executables != 0 {
            return Err(
                "reusable execution preparation requires an empty executable cache".to_owned(),
            );
        }
        self.lifecycle = ReusableExecutionPreparationLifecycle::Preparing(plan);
        Ok(DeviceReusableExecutionPreparation::preparing(plan))
    }

    pub(crate) const fn capture_is_open(&self) -> bool {
        matches!(
            self.lifecycle,
            ReusableExecutionPreparationLifecycle::Preparing(_)
        )
    }

    pub(crate) const fn maximum_executables(&self) -> Option<usize> {
        match self.lifecycle {
            ReusableExecutionPreparationLifecycle::Preparing(plan)
            | ReusableExecutionPreparationLifecycle::Ready(plan) => {
                Some(plan.maximum_executables())
            }
            ReusableExecutionPreparationLifecycle::Unconfigured => None,
        }
    }

    pub(crate) fn record_batch(
        &mut self,
        captured_executables: usize,
        uploaded_executables: usize,
        capacity_deferred_executables: usize,
    ) -> Result<(), String> {
        if !self.capture_is_open() {
            return Err(
                "reusable execution work was recorded outside its preparation window".to_owned(),
            );
        }
        self.captured_executables = self
            .captured_executables
            .saturating_add(u64::try_from(captured_executables).unwrap_or(u64::MAX));
        self.uploaded_executables = self
            .uploaded_executables
            .saturating_add(u64::try_from(uploaded_executables).unwrap_or(u64::MAX));
        self.capacity_deferred_executables = self
            .capacity_deferred_executables
            .saturating_add(u64::try_from(capacity_deferred_executables).unwrap_or(u64::MAX));
        Ok(())
    }

    pub(crate) fn snapshot(
        &self,
        resident_executables: usize,
        rejected_executables: usize,
    ) -> Result<DeviceReusableExecutionPreparation, String> {
        match self.lifecycle {
            ReusableExecutionPreparationLifecycle::Unconfigured => {
                Err("reusable execution preparation is not configured".to_owned())
            }
            ReusableExecutionPreparationLifecycle::Preparing(plan) => {
                DeviceReusableExecutionPreparation::preparing_with_progress(
                    plan,
                    resident_executables,
                    rejected_executables,
                    self.captured_executables,
                    self.uploaded_executables,
                    self.capacity_deferred_executables,
                )
                .map_err(|error| error.to_string())
            }
            ReusableExecutionPreparationLifecycle::Ready(plan) => {
                DeviceReusableExecutionPreparation::ready(
                    plan,
                    resident_executables,
                    rejected_executables,
                    self.captured_executables,
                    self.uploaded_executables,
                    self.capacity_deferred_executables,
                )
                .map_err(|error| error.to_string())
            }
        }
    }

    pub(crate) fn seal(
        &mut self,
        resident_executables: usize,
        rejected_executables: usize,
    ) -> Result<DeviceReusableExecutionPreparation, String> {
        let ReusableExecutionPreparationLifecycle::Preparing(plan) = self.lifecycle else {
            return Err("reusable execution preparation is not open".to_owned());
        };
        let report = DeviceReusableExecutionPreparation::ready(
            plan,
            resident_executables,
            rejected_executables,
            self.captured_executables,
            self.uploaded_executables,
            self.capacity_deferred_executables,
        )
        .map_err(|error| error.to_string())?;
        self.lifecycle = ReusableExecutionPreparationLifecycle::Ready(plan);
        Ok(report)
    }
}

#[cfg(test)]
mod tests {
    use ferrum_interfaces::vnext::DeviceCommandPhase;

    use super::{
        discover_reusable_segments, plan_bounded_reusable_execution,
        ReusableExecutionPreparationTracker,
    };
    use ferrum_interfaces::vnext::{
        DeviceReusableExecutionPlan, DeviceReusableExecutionPreparationState,
    };

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

    #[test]
    fn preparation_lifecycle_is_explicit_bounded_and_permanently_sealed() {
        let plan = DeviceReusableExecutionPlan::new(4).unwrap();
        let mut tracker = ReusableExecutionPreparationTracker::default();

        let preparing = tracker.configure(plan, 0, 0).unwrap();
        assert_eq!(
            preparing.state(),
            DeviceReusableExecutionPreparationState::Preparing
        );
        assert!(tracker.capture_is_open());
        assert_eq!(tracker.maximum_executables(), Some(4));

        tracker.record_batch(3, 3, 1).unwrap();
        let progress = tracker.snapshot(3, 1).unwrap();
        assert_eq!(
            progress.state(),
            DeviceReusableExecutionPreparationState::Preparing
        );
        assert_eq!(progress.captured_executables(), 3);
        let ready = tracker.seal(3, 1).unwrap();
        assert_eq!(
            ready.state(),
            DeviceReusableExecutionPreparationState::Ready
        );
        assert_eq!(ready.maximum_executables(), 4);
        assert_eq!(ready.resident_executables(), 3);
        assert_eq!(ready.rejected_executables(), 1);
        assert_eq!(ready.captured_executables(), 3);
        assert_eq!(ready.uploaded_executables(), 3);
        assert_eq!(ready.capacity_deferred_executables(), 1);
        assert!(!tracker.capture_is_open());
        assert_eq!(
            tracker.snapshot(3, 1).unwrap().state(),
            DeviceReusableExecutionPreparationState::Ready
        );
        assert!(tracker.record_batch(1, 1, 0).is_err());
        assert!(tracker.seal(3, 1).is_err());
        assert!(tracker.configure(plan, 0, 0).is_err());
    }

    #[test]
    fn preparation_rejects_dirty_cache_and_inconsistent_receipts() {
        let plan = DeviceReusableExecutionPlan::new(2).unwrap();
        let mut dirty = ReusableExecutionPreparationTracker::default();
        assert!(dirty.configure(plan, 1, 0).is_err());

        let mut tracker = ReusableExecutionPreparationTracker::default();
        tracker.configure(plan, 0, 0).unwrap();
        tracker.record_batch(1, 0, 0).unwrap();
        assert!(tracker.seal(1, 0).is_err());
    }
}
