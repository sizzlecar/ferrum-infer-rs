//! Bounded native plan ownership, independent of any device API.
//!
//! A cache entry is reusable but may be reclaimed only when the cache is its
//! sole strong owner. Compiled estimates/plans and submitted commands hold
//! leases, so eviction never invalidates an implementation they selected.

use std::collections::{BTreeMap, VecDeque};
use std::sync::Arc;

pub(crate) struct RetainedPlanCache<K, V> {
    maximum: usize,
    plans: BTreeMap<K, Arc<V>>,
    recency: VecDeque<K>,
    evictions: u64,
    failed_preparations: u64,
    capacity_rejections: u64,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct RetainedPlanCacheSnapshot {
    pub entries: usize,
    pub maximum: usize,
    pub externally_retained_entries: usize,
    pub evictions: u64,
    pub failed_preparations: u64,
    pub capacity_rejections: u64,
}

#[derive(Debug, PartialEq, Eq)]
pub(crate) enum PlanCacheError<E> {
    AllPlansRetained,
    Prepare(E),
}

impl<K: Ord + Clone, V> RetainedPlanCache<K, V> {
    pub(crate) fn new(maximum: usize) -> Self {
        Self {
            maximum,
            plans: BTreeMap::new(),
            recency: VecDeque::new(),
            evictions: 0,
            failed_preparations: 0,
            capacity_rejections: 0,
        }
    }

    pub(crate) fn snapshot(&self) -> RetainedPlanCacheSnapshot {
        RetainedPlanCacheSnapshot {
            entries: self.plans.len(),
            maximum: self.maximum,
            externally_retained_entries: self
                .plans
                .values()
                .filter(|plan| Arc::strong_count(plan) > 1)
                .count(),
            evictions: self.evictions,
            failed_preparations: self.failed_preparations,
            capacity_rejections: self.capacity_rejections,
        }
    }

    pub(crate) fn get(&mut self, key: &K) -> Option<Arc<V>> {
        let plan = self.plans.get(key)?.clone();
        self.recency.retain(|entry| entry != key);
        self.recency.push_back(key.clone());
        Some(plan)
    }

    /// Call under the cache's preparation lock. A failed factory inserts no
    /// entry; a full cache fails before calling it if every plan is still leased.
    pub(crate) fn get_or_try_insert_with<E>(
        &mut self,
        key: K,
        prepare: impl FnOnce() -> Result<Arc<V>, E>,
    ) -> Result<Arc<V>, PlanCacheError<E>> {
        if let Some(plan) = self.get(&key) {
            return Ok(plan);
        }
        if self.plans.len() >= self.maximum {
            let reclaimable = self.recency.iter().position(|key| {
                self.plans
                    .get(key)
                    .is_some_and(|plan| Arc::strong_count(plan) == 1)
            });
            let Some(index) = reclaimable else {
                self.capacity_rejections = self.capacity_rejections.saturating_add(1);
                return Err(PlanCacheError::AllPlansRetained);
            };
            let old = self.recency.remove(index).expect("located cache entry");
            // Drop the old native owner before a replacement can allocate.
            self.plans.remove(&old);
            self.evictions = self.evictions.saturating_add(1);
        }
        let plan = prepare().map_err(|error| {
            self.failed_preparations = self.failed_preparations.saturating_add(1);
            PlanCacheError::Prepare(error)
        })?;
        self.plans.insert(key.clone(), Arc::clone(&plan));
        self.recency.push_back(key);
        Ok(plan)
    }
}

#[cfg(test)]
mod tests;
