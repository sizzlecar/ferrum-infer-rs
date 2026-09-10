//! Bounded warmup evidence for a single owning execution stream.
//!
//! Seeing a key is not proof that its work completed. Pending keys become
//! eligible only when the caller proves all earlier stream work quiescent.
//! A failed or indeterminate stream must be discarded with this tracker.
use std::collections::{HashMap, HashSet};
use std::hash::Hash;

pub(crate) struct DemandWarmup<K> {
    completed: HashMap<K, u64>,
    pending: HashMap<K, u64>,
    clock: u64,
}

impl<K> Default for DemandWarmup<K> {
    fn default() -> Self {
        Self {
            completed: HashMap::new(),
            pending: HashMap::new(),
            clock: 0,
        }
    }
}

impl<K: Copy + Eq + Hash> DemandWarmup<K> {
    /// Returns missing keys that must still use ordinary execution. Each map
    /// retains at most the resolved executable capacity; forgetting evidence
    /// requires a new warmup, never an optimistic capture.
    pub(crate) fn required(
        &mut self,
        missing: &[K],
        capacity: usize,
        earlier_work_quiescent: bool,
    ) -> HashSet<K> {
        if capacity == 0 {
            self.completed.clear();
            self.pending.clear();
            return missing.iter().copied().collect();
        }
        if earlier_work_quiescent {
            for (key, age) in self.pending.drain() {
                self.completed.insert(key, age);
            }
        }
        Self::trim(&mut self.completed, capacity);
        let mut required = HashSet::new();
        for key in missing {
            self.clock = self.clock.wrapping_add(1).max(1);
            if let Some(age) = self.completed.get_mut(key) {
                *age = self.clock;
            } else {
                required.insert(*key);
                self.pending.insert(*key, self.clock);
            }
        }
        Self::trim(&mut self.pending, capacity);
        required
    }

    fn trim(map: &mut HashMap<K, u64>, capacity: usize) {
        while map.len() > capacity {
            let key = *map.iter().min_by_key(|(_, age)| *age).unwrap().0;
            map.remove(&key);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn first_real_submission_warms_and_only_quiescence_authorizes_capture() {
        let mut warmup = DemandWarmup::default();
        assert_eq!(warmup.required(&[1, 2], 4, true), HashSet::from([1, 2]));
        assert_eq!(warmup.required(&[1, 2], 4, false), HashSet::from([1, 2]));
        assert!(warmup.required(&[1, 2], 4, true).is_empty());
        assert_eq!(warmup.required(&[1, 3], 4, true), HashSet::from([3]));
    }

    #[test]
    fn pending_and_completed_evidence_stay_bounded_and_eviction_requires_warmup() {
        let mut warmup = DemandWarmup::default();
        warmup.required(&[1, 2, 3], 2, true);
        assert_eq!(warmup.pending.len(), 2);
        assert!(warmup.required(&[2, 3], 2, true).is_empty());
        assert_eq!(warmup.completed.len(), 2);
        assert_eq!(warmup.required(&[1], 2, true), HashSet::from([1]));
        assert!(warmup.required(&[1], 2, true).is_empty());
        assert_eq!(warmup.completed.len(), 2);
        assert_eq!(warmup.required(&[2], 2, true), HashSet::from([2]));
    }

    #[test]
    fn a_new_stream_does_not_inherit_warmup_or_capture_when_capacity_is_zero() {
        let mut first = DemandWarmup::default();
        first.required(&[7], 1, true);
        assert!(first.required(&[7], 1, true).is_empty());
        let mut replacement = DemandWarmup::default();
        assert_eq!(replacement.required(&[7], 1, true), HashSet::from([7]));
        assert_eq!(first.required(&[7], 0, true), HashSet::from([7]));
        assert!(first.pending.is_empty());
        assert!(first.completed.is_empty());
    }
}
