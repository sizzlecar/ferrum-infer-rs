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
                return Err(PlanCacheError::AllPlansRetained);
            };
            let old = self.recency.remove(index).expect("located cache entry");
            // Drop the old native owner before a replacement can allocate.
            self.plans.remove(&old);
        }
        let plan = prepare().map_err(PlanCacheError::Prepare)?;
        self.plans.insert(key.clone(), Arc::clone(&plan));
        self.recency.push_back(key);
        Ok(plan)
    }
}

#[cfg(test)]
mod tests;
