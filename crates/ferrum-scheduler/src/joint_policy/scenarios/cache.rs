//! Memoization belongs to one immutable snapshot, one query time and one
//! comparison. Hash collisions only evict entries; exact work equality is required.

use super::*;
use ferrum_execution::execution_attempt::AttemptPrefixRestore;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

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

    pub(super) fn get(
        &mut self,
        query: &CostQuery<'_>,
        at: Instant,
        rank: usize,
    ) -> Option<CostPrediction> {
        if self.at != at || self.slots.is_empty() {
            return None;
        }
        let slot = self.slots[index(query, self.slots.len())].as_ref()?;
        if slot.exact != query.work_is_exact
            || slot.execution != query.execution
            || slot.work.as_ref() != query.work
        {
            return None;
        }
        let prediction = slot.prediction(rank)?;
        self.hits = self.hits.saturating_add(1);
        Some(prediction)
    }

    /// Only complete validated distributions with this comparison's weights and
    /// dependence enter the cache. Insufficient memory merely disables this
    /// insertion; uncached calls retain the normal prediction/time budgets.
    pub(super) fn insert(&mut self, query: &CostQuery<'_>, prediction: CostScenarioPrediction) {
        if self.slots.is_empty() {
            return;
        }
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
        let slot = index(query, self.slots.len());
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
}

fn index(query: &CostQuery<'_>, slots: usize) -> usize {
    let mut hash = DefaultHasher::new();
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
