//! Memoization belongs to one immutable snapshot, one query time and one
//! comparison. At most two existing slots are inspected per key; collisions
//! may still evict entries, and exact work equality is always required.

use super::*;
use ferrum_execution::execution_attempt::AttemptPrefixRestore;
use rustc_hash::FxHasher;
use std::hash::{Hash, Hasher};

#[cfg(test)]
mod tests;

/// Private to one ConditionalSnapshot::predict call, for the same cache. The
/// borrowed query cannot change while the token exists; insertion accepts no
/// second query that could disagree with this already computed slot. No lock
/// or owned key allocation is retained across the provider call.
pub(super) struct InsertionKey<'a> {
    query: CostQuery<'a>,
    slot: usize,
    at: Instant,
    slots_len: usize,
}

struct Entry {
    work: Box<[CostWork]>,
    exact: bool,
    execution: CostExecution,
    scenarios: Box<[CostScenario]>,
    dependence: CostScenarioDependence,
    support: CostSupport,
    samples: usize,
    observed_at: Instant,
    expires_at: Instant,
}

impl Entry {
    fn matches(&self, query: &CostQuery<'_>) -> bool {
        self.exact == query.work_is_exact
            && self.execution == query.execution
            && self.work.as_ref() == query.work
    }

    fn payload_bytes(&self) -> usize {
        std::mem::size_of_val(self.work.as_ref()) + std::mem::size_of_val(self.scenarios.as_ref())
    }

    fn prediction(&self, rank: usize) -> Option<CostPrediction> {
        let service_time = self.scenarios.get(rank)?.service_time;
        Some(CostPrediction::Known {
            central_service_time: service_time,
            upper_service_time: service_time,
            risk: CostRisk::ConditionalScenario {
                dependence: self.dependence,
            },
            support: self.support,
            samples: self.samples,
            observed_at: self.observed_at,
            expires_at: self.expires_at,
        })
    }
}

pub(super) struct ScenarioCache {
    at: Instant,
    slots: Box<[Option<Entry>]>,
    max_bytes: usize,
    retained_bytes: usize,
    hits: usize,
}

impl ScenarioCache {
    pub(super) fn new(at: Instant, max_bytes: usize, max_queries: usize) -> Self {
        // Include all allocated slot storage, even empty entries. Payloads use
        // boxed slices, so their retained capacity equals the accounted length.
        let minimum = std::mem::size_of::<Option<Entry>>()
            + std::mem::size_of::<CostWork>()
            + std::mem::size_of::<CostScenario>();
        let count = max_queries.min(max_bytes / minimum);
        let slots = std::iter::repeat_with(|| None)
            .take(count)
            .collect::<Box<[_]>>();
        Self {
            at,
            retained_bytes: std::mem::size_of_val(slots.as_ref()),
            slots,
            max_bytes,
            hits: 0,
        }
    }

    pub(super) fn retained_bytes(&self) -> usize {
        self.retained_bytes
    }

    pub(super) fn hits(&self) -> usize {
        self.hits
    }

    #[cfg(test)]
    pub(super) fn get(
        &mut self,
        query: &CostQuery<'_>,
        at: Instant,
        rank: usize,
    ) -> Option<CostPrediction> {
        self.get_with_key(query, at, rank).0
    }

    pub(super) fn get_with_key<'a>(
        &mut self,
        query: &CostQuery<'a>,
        at: Instant,
        rank: usize,
    ) -> (Option<CostPrediction>, Option<InsertionKey<'a>>) {
        if self.at != at || self.slots.is_empty() {
            return (None, None);
        }
        let slot = index(query, self.slots.len());
        let prediction = self.probe_slots(slot).find_map(|slot| {
            self.slots[slot]
                .as_ref()
                .filter(|entry| entry.matches(query))
                .and_then(|entry| entry.prediction(rank))
        });
        if let Some(prediction) = prediction {
            self.hits = self.hits.saturating_add(1);
            return (Some(prediction), None);
        }
        (
            None,
            Some(InsertionKey {
                query: *query,
                slot,
                at,
                slots_len: self.slots.len(),
            }),
        )
    }

    /// Only complete validated distributions with this comparison's weights and
    /// dependence enter the cache. Insufficient memory merely disables this
    /// insertion; uncached calls retain the normal prediction/time budgets.
    pub(super) fn insert(&mut self, query: &CostQuery<'_>, prediction: CostScenarioPrediction) {
        if self.slots.is_empty() || !matches!(&prediction, CostScenarioPrediction::Known { .. }) {
            return;
        }
        // Seed insertion has no preceding lookup. Normal misses use the token.
        let key = InsertionKey {
            query: *query,
            slot: index(query, self.slots.len()),
            at: self.at,
            slots_len: self.slots.len(),
        };
        self.insert_with_key(Some(key), prediction);
    }

    pub(super) fn insert_with_key(
        &mut self,
        key: Option<InsertionKey<'_>>,
        prediction: CostScenarioPrediction,
    ) {
        let Some(key) = key else {
            return;
        };
        if self.at != key.at || self.slots.len() != key.slots_len || key.slot >= self.slots.len() {
            return;
        }
        let query = key.query;
        let CostScenarioPrediction::Known {
            scenarios,
            dependence,
            support,
            samples,
            observed_at,
            expires_at,
            ..
        } = prediction
        else {
            return;
        };
        // The provider runs without the cache lock. Search CURRENT entries for
        // this exact key before considering empties, so an interleaved insert
        // cannot create two copies with different distributions. A third
        // colliding key deterministically replaces its primary slot.
        let slot = self
            .probe_slots(key.slot)
            .find(|slot| {
                self.slots[*slot]
                    .as_ref()
                    .is_some_and(|entry| entry.matches(&query))
            })
            .or_else(|| {
                self.probe_slots(key.slot)
                    .find(|slot| self.slots[*slot].is_none())
            })
            .unwrap_or(key.slot);
        let previous = self.slots[slot].as_ref().map_or(0, Entry::payload_bytes);
        let Some(bytes) = query
            .work
            .len()
            .checked_mul(std::mem::size_of::<CostWork>())
            .and_then(|work| {
                scenarios
                    .len()
                    .checked_mul(std::mem::size_of::<CostScenario>())
                    .and_then(|scenarios| work.checked_add(scenarios))
            })
            .and_then(|new| self.retained_bytes.checked_sub(previous)?.checked_add(new))
        else {
            return;
        };
        if bytes > self.max_bytes {
            return;
        }
        // Release a colliding payload before allocating its replacement key.
        self.slots[slot] = None;
        self.slots[slot] = Some(Entry {
            work: query.work.into(),
            exact: query.work_is_exact,
            execution: query.execution,
            scenarios: scenarios.into_boxed_slice(),
            dependence,
            support,
            samples,
            observed_at,
            expires_at,
        });
        self.retained_bytes = bytes;
    }

    fn probe_slots(&self, primary: usize) -> impl Iterator<Item = usize> {
        debug_assert!(primary < self.slots.len());
        let secondary = if primary == self.slots.len() - 1 {
            0
        } else {
            primary + 1
        };
        // Fixed stack storage only. The one-slot case must neither repeat a
        // lookup nor count a hit twice; zero slots never reach this helper.
        [primary, secondary]
            .into_iter()
            .take(self.slots.len().min(2))
    }
}

fn index(query: &CostQuery<'_>, slots: usize) -> usize {
    let mut hash = FxHasher::default();
    query.work_is_exact.hash(&mut hash);
    (query.execution as u8).hash(&mut hash);
    query.work.len().hash(&mut hash);
    for work in query.work {
        (work.kind as u8).hash(&mut hash);
        work.requested.start().hash(&mut hash);
        work.requested.end().hash(&mut hash);
        work.context_tokens.hash(&mut hash);
        match work.prefix_restore {
            AttemptPrefixRestore::NoneObserved => 0usize.hash(&mut hash),
            AttemptPrefixRestore::Restored { tokens } => {
                1usize.hash(&mut hash);
                tokens.hash(&mut hash);
            }
        }
        work.internally_reused_tokens.hash(&mut hash);
    }
    hash.finish() as usize % slots
}
