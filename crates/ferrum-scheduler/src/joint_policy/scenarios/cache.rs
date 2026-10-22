//! Memoization belongs to one immutable snapshot, one query time and one
//! comparison. One untrusted ordinal hint can avoid hashing; otherwise the
//! original two-slot lookup applies. Exact work equality is always required.

use super::*;
use ferrum_execution::execution_attempt::AttemptPrefixRestore;
use rustc_hash::FxHasher;
use std::hash::{Hash, Hasher};
use std::num::NonZeroUsize;

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
    hint_ordinal: Option<usize>,
}

struct Entry {
    work: Box<[CostWork]>,
    exact: bool,
    execution: CostExecution,
    // Ranks are consecutive and weights/coupling were checked before insert.
    // Keep every rank value; the comparison owns the shared weight vector.
    service_times: Box<[Duration]>,
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
        std::mem::size_of_val(self.work.as_ref())
            + std::mem::size_of_val(self.service_times.as_ref())
    }

    fn prediction(&self, rank: usize) -> Option<CostPrediction> {
        let service_time = *self.service_times.get(rank)?;
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
    /// Slot + 1, never an entry reference or a cached prediction. A changed
    /// rank topology or replacement is harmless because every hit is checked.
    hints: Box<[Option<NonZeroUsize>]>,
    next_hint: usize,
    max_bytes: usize,
    retained_bytes: usize,
    hits: usize,
    #[cfg(test)]
    hint_hits: usize,
    #[cfg(test)]
    hash_lookups: usize,
}

impl ScenarioCache {
    pub(super) fn new(at: Instant, max_bytes: usize, max_queries: usize) -> Self {
        // Include all allocated slot storage, even empty entries. Payloads use
        // boxed slices, so their retained capacity equals the accounted length.
        // Preserve the previous conservative slot geometry. Packing payloads
        // more densely does not also change key hashing or collision buckets.
        let minimum = std::mem::size_of::<Option<Entry>>()
            + std::mem::size_of::<CostWork>()
            + std::mem::size_of::<CostScenario>();
        let count = max_queries.min(max_bytes / minimum);
        let slots = std::iter::repeat_with(|| None)
            .take(count)
            .collect::<Box<[_]>>();
        let slot_bytes = std::mem::size_of_val(slots.as_ref());
        // Keep the existing slot/hash geometry. Hints consume part of the same
        // retained-byte budget, so payload insertion may fail earlier. No new
        // budget is granted and a missing hint only uses the original lookup.
        let hint_count = count.min(
            max_bytes.saturating_sub(slot_bytes) / std::mem::size_of::<Option<NonZeroUsize>>(),
        );
        let hints = vec![None; hint_count].into_boxed_slice();
        Self {
            at,
            retained_bytes: slot_bytes + std::mem::size_of_val(hints.as_ref()),
            slots,
            hints,
            next_hint: 0,
            max_bytes,
            hits: 0,
            #[cfg(test)]
            hint_hits: 0,
            #[cfg(test)]
            hash_lookups: 0,
        }
    }

    /// Retain only untrusted lookup locations from the preceding traversal.
    /// Endpoints, candidate failures and ordering may differ in this rank.
    pub(super) fn begin_rank(&mut self) {
        self.next_hint = 0;
    }

    fn remember_hint(&mut self, ordinal: Option<usize>, slot: usize) {
        if let Some(hint) = ordinal.and_then(|ordinal| self.hints.get_mut(ordinal)) {
            *hint = slot.checked_add(1).and_then(NonZeroUsize::new);
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
        let ordinal = (self.next_hint < self.hints.len()).then_some(self.next_hint);
        self.next_hint = self.next_hint.saturating_add(1);
        let hinted = ordinal
            .and_then(|ordinal| self.hints[ordinal])
            .and_then(|slot| self.slots.get(slot.get() - 1))
            .and_then(Option::as_ref)
            .filter(|entry| entry.matches(query))
            .and_then(|entry| entry.prediction(rank));
        if let Some(prediction) = hinted {
            self.hits = self.hits.saturating_add(1);
            #[cfg(test)]
            {
                self.hint_hits += 1;
            }
            return (Some(prediction), None);
        }
        #[cfg(test)]
        {
            self.hash_lookups += 1;
        }
        let slot = index(query, self.slots.len());
        let prediction = self.probe_slots(slot).find_map(|slot| {
            self.slots[slot]
                .as_ref()
                .filter(|entry| entry.matches(query))
                .and_then(|entry| entry.prediction(rank))
                .map(|prediction| (slot, prediction))
        });
        if let Some((slot, prediction)) = prediction {
            self.remember_hint(ordinal, slot);
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
                hint_ordinal: ordinal,
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
            hint_ordinal: None,
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
                    .checked_mul(std::mem::size_of::<Duration>())
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
            service_times: scenarios
                .into_iter()
                .map(|scenario| scenario.service_time)
                .collect::<Box<[_]>>(),
            dependence,
            support,
            samples,
            observed_at,
            expires_at,
        });
        self.retained_bytes = bytes;
        // The provider ran outside the lock. Associate the actual current slot
        // with the original lookup ordinal, not a later cursor or primary slot.
        self.remember_hint(key.hint_ordinal, slot);
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
