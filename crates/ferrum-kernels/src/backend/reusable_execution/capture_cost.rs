//! Bounded admission credit for optional live graph construction.
//!
//! The first completed eager execution is safety warmup, not evidence of a
//! recurring saving: it can contain lazy initialization. Later completed eager
//! submissions earn credit against the largest observed graph preparation
//! cost. This predicts amortization; it does not promise future reuse or bound
//! an unseen graph's first instantiation latency.

use std::collections::HashMap;
use std::hash::Hash;

#[derive(Default)]
struct Evidence {
    last_used: u64,
    eager_ns: u64,
    eager_uses: u64,
}

struct Pending {
    last_used: u64,
    first_ns: u64,
    total_ns: u64,
    count: u64,
}

pub(crate) struct CaptureCost<K> {
    completed: HashMap<K, Evidence>,
    pending: HashMap<K, Pending>,
    clock: u64,
    preparation_ns: Option<u64>,
    replay_ns: Option<u64>,
}

impl<K> Default for CaptureCost<K> {
    fn default() -> Self {
        Self {
            completed: HashMap::new(),
            pending: HashMap::new(),
            clock: 0,
            preparation_ns: None,
            replay_ns: None,
        }
    }
}

impl<K: Copy + Eq + Hash> CaptureCost<K> {
    pub(crate) fn observe_preparation(&mut self, elapsed_ns: u64) {
        self.preparation_ns = Some(self.preparation_ns.unwrap_or(0).max(elapsed_ns).max(1));
    }

    pub(crate) fn observe_replay(&mut self, elapsed_ns: u64) {
        self.clock = self.clock.saturating_add(1);
        self.replay_ns = Some(self.replay_ns.unwrap_or(0).max(elapsed_ns));
    }

    /// A successful enqueue is pending until the owning stream is quiescent.
    /// An indeterminate/failed stream and its entire tracker must be discarded.
    pub(crate) fn observe_eager(&mut self, key: K, elapsed_ns: u64, capacity: usize) {
        if capacity == 0 {
            self.clear_evidence();
            return;
        }
        self.clock = self.clock.saturating_add(1);
        let pending = self.pending.entry(key).or_insert(Pending {
            last_used: self.clock,
            first_ns: elapsed_ns,
            total_ns: 0,
            count: 0,
        });
        pending.last_used = self.clock;
        pending.total_ns = pending.total_ns.saturating_add(elapsed_ns);
        pending.count = pending.count.saturating_add(1);
        Self::trim(&mut self.pending, capacity, |entry| entry.last_used);
    }

    pub(crate) fn promote_completed(&mut self, quiescent: bool, capacity: usize) {
        if capacity == 0 {
            self.clear_evidence();
            return;
        }
        if !quiescent {
            return;
        }
        // Recurrence is local to the bounded observation horizon. A rare key
        // cannot accumulate old credit forever while other shapes run.
        let horizon = u64::try_from(capacity).unwrap_or(u64::MAX);
        self.completed
            .retain(|_, entry| self.clock.saturating_sub(entry.last_used) <= horizon);
        for (key, pending) in self.pending.drain() {
            if self.clock.saturating_sub(pending.last_used) > horizon {
                continue;
            }
            let already_warm = self.completed.contains_key(&key);
            let entry = self.completed.entry(key).or_default();
            entry.last_used = pending.last_used;
            let (total, count) = if already_warm {
                (pending.total_ns, pending.count)
            } else {
                (
                    pending.total_ns.saturating_sub(pending.first_ns),
                    pending.count.saturating_sub(1),
                )
            };
            entry.eager_ns = entry.eager_ns.saturating_add(total);
            entry.eager_uses = entry.eager_uses.saturating_add(count);
        }
        Self::trim(&mut self.completed, capacity, |entry| entry.last_used);
    }

    pub(crate) fn admits(&self, key: K) -> bool {
        self.preparation_ns
            .zip(self.replay_ns)
            .and_then(|(cost, replay)| {
                self.completed.get(&key).map(|entry| {
                    entry
                        .eager_ns
                        .saturating_sub(replay.saturating_mul(entry.eager_uses))
                        >= cost
                })
            })
            .unwrap_or(false)
    }

    pub(crate) fn forget(&mut self, key: K) {
        self.completed.remove(&key);
        self.pending.remove(&key);
    }

    pub(crate) fn clear_evidence(&mut self) {
        self.completed.clear();
        self.pending.clear();
    }

    fn trim<T>(map: &mut HashMap<K, T>, capacity: usize, age: impl Fn(&T) -> u64) {
        while map.len() > capacity {
            let key = *map.iter().min_by_key(|(_, entry)| age(entry)).unwrap().0;
            map.remove(&key);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::CaptureCost;

    #[test]
    fn cold_or_uncompleted_work_cannot_pay_for_capture() {
        let mut cost = CaptureCost::default();
        cost.observe_preparation(100);
        cost.observe_replay(10);
        cost.observe_eager(7, 10_000, 4);
        cost.promote_completed(false, 4);
        assert!(!cost.admits(7));
        cost.promote_completed(true, 4);
        assert!(!cost.admits(7), "cold warmup is not recurring savings");
        cost.observe_eager(7, 110, 4);
        assert!(!cost.admits(7));
        cost.promote_completed(true, 4);
        assert!(cost.admits(7));
    }

    #[test]
    fn measured_cost_changes_the_required_recurrence_without_rejection() {
        let mut cost = CaptureCost::default();
        cost.observe_preparation(250);
        cost.observe_replay(10);
        cost.observe_eager(3, 1000, 4);
        cost.promote_completed(true, 4);
        for _ in 0..2 {
            cost.observe_eager(3, 110, 4);
            cost.promote_completed(true, 4);
            assert!(!cost.admits(3));
        }
        cost.observe_eager(3, 60, 4);
        cost.promote_completed(true, 4);
        assert!(cost.admits(3));
        cost.observe_preparation(500);
        assert!(!cost.admits(3));
        cost.observe_eager(3, 260, 4);
        cost.promote_completed(true, 4);
        assert!(cost.admits(3), "temporary deferral must remain learnable");
    }

    #[test]
    fn missing_costs_and_nonpositive_savings_never_mean_free_capture() {
        let mut cost = CaptureCost::default();
        cost.observe_eager(1, 100, 4);
        cost.promote_completed(true, 4);
        assert!(!cost.admits(1));
        cost.observe_preparation(1);
        cost.observe_replay(100);
        cost.observe_eager(1, 90, 4);
        cost.promote_completed(true, 4);
        assert!(!cost.admits(1));
    }

    #[test]
    fn stale_or_evicted_credit_does_not_survive_its_bounded_owner() {
        let mut cost = CaptureCost::default();
        cost.observe_preparation(100);
        cost.observe_replay(0);
        cost.observe_eager(1, 100, 2);
        cost.observe_eager(1, 100, 2);
        cost.promote_completed(true, 2);
        assert!(cost.admits(1));
        for _ in 0..3 {
            cost.observe_replay(0);
        }
        cost.promote_completed(true, 2);
        assert!(!cost.admits(1), "resident traffic also ages stale credit");
        for key in [2, 3, 4] {
            cost.observe_eager(key, 1000, 2);
        }
        assert!(cost.pending.len() <= 2);
        cost.promote_completed(true, 2);
        assert!(cost.completed.len() <= 2);
        assert!(!cost.admits(1));
        cost.forget(4);
        assert!(!cost.completed.contains_key(&4));
        cost.promote_completed(true, 0);
        assert!(cost.completed.is_empty() && cost.pending.is_empty());
    }
}
