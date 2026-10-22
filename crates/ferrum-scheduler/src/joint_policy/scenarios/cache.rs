//! Memoization belongs to one immutable snapshot, one query time and one
//! comparison. One untrusted ordinal hint can avoid hashing; otherwise the
//! original two-slot lookup applies. Only the private concrete replay path can
//! use an owner-bound immutable entry handle instead of repeating work equality.

use super::*;
use ferrum_execution::execution_attempt::AttemptPrefixRestore;
use rustc_hash::FxHasher;
use std::hash::{Hash, Hasher};
use std::num::{NonZeroU64, NonZeroUsize};
use std::sync::atomic::{AtomicU64, Ordering};

#[cfg(test)]
pub(super) mod accounting;
#[cfg(test)]
mod tests;

mod service_times;
use service_times::TimeEncoding;

static NEXT_HANDLE_OWNER: AtomicU64 = AtomicU64::new(1);

// Exhaustion is permanent: never reuse an owner identity, including across
// dropped caches. Relaxed suffices because identity does not publish data.
fn allocate_handle_owner(counter: &AtomicU64) -> Option<NonZeroU64> {
    counter
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |next| {
            next.checked_add(1)
        })
        .ok()
        .and_then(NonZeroU64::new)
}

/// Exact identity of an immutable entry in this comparison's cache. No fields
/// or factory escape this module; the sealed replay path supplies a handle only
/// for work previously copied from that same entry after full wave validation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::joint_policy) struct CacheHandle {
    owner: NonZeroU64,
    slot: usize,
    generation: NonZeroU64,
}

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
    service_times: Box<[u64]>,
    time_encoding: TimeEncoding,
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

    fn service_time(&self, rank: usize) -> Option<Duration> {
        self.time_encoding.get(&self.service_times, rank)
    }

    fn prediction(&self, rank: usize) -> Option<CostPrediction> {
        let service_time = self.service_time(rank)?;
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
    // Separate storage preserves the original Entry layout and hash geometry.
    // Ordinary adapters do not allocate it. Fixed struct metadata, as before,
    // is distinct from the retained allocation budget (not a process RSS bound).
    generations: Box<[u64]>,
    handle_owner: Option<NonZeroU64>,
    handles_started: bool,
    last_generation: u64,
    trace_bytes: usize,
    max_bytes: usize,
    retained_bytes: usize,
    hits: usize,
    #[cfg(test)]
    hint_hits: usize,
    #[cfg(test)]
    hash_lookups: usize,
    #[cfg(test)]
    handle_hits: usize,
    #[cfg(test)]
    accounting: Option<accounting::Counters>,
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
            generations: Box::default(),
            handle_owner: None,
            handles_started: false,
            last_generation: 0,
            trace_bytes: 0,
            max_bytes,
            hits: 0,
            #[cfg(test)]
            hint_hits: 0,
            #[cfg(test)]
            hash_lookups: 0,
            #[cfg(test)]
            handle_hits: 0,
            #[cfg(test)]
            accounting: accounting::Counters::new(),
        }
    }

    /// Enable only before seeding the private concrete replay evaluation. The
    /// existing slot count and ordinal hints stay unchanged. Failure disables
    /// this optimization, without evicting entries or granting another budget.
    pub(super) fn enable_handles(&mut self) -> bool {
        if self.handles_started {
            return self.handle_owner.is_some();
        }
        self.handles_started = true;
        if self.slots.is_empty() || self.slots.iter().any(Option::is_some) || self.next_hint != 0 {
            return false;
        }
        let Some(bytes) = self
            .slots
            .len()
            .checked_mul(std::mem::size_of::<u64>())
            .and_then(|extra| self.retained_bytes.checked_add(extra))
            .filter(|bytes| *bytes <= self.max_bytes)
        else {
            return false;
        };
        let mut generations = Vec::new();
        if generations.try_reserve_exact(self.slots.len()).is_err()
            || !generations
                .capacity()
                .checked_mul(std::mem::size_of::<u64>())
                .and_then(|extra| self.retained_bytes.checked_add(extra))
                .is_some_and(|bytes| bytes <= self.max_bytes)
        {
            return false;
        }
        let Some(owner) = allocate_handle_owner(&NEXT_HANDLE_OWNER) else {
            return false;
        };
        generations.resize(self.slots.len(), 0);
        self.generations = generations.into_boxed_slice();
        self.handle_owner = Some(owner);
        self.retained_bytes = bytes;
        true
    }

    /// Reserve the trace owner's total *allocated* bytes, sharing the original
    /// cache budget. The caller frees capacity before lowering this charge;
    /// rejected growth changes nothing and never evicts a cache entry.
    pub(super) fn set_trace_bytes(&mut self, total: usize) -> bool {
        if total > self.trace_bytes && self.handle_owner.is_none() {
            return false;
        }
        let Some(bytes) = self
            .retained_bytes
            .checked_sub(self.trace_bytes)
            .and_then(|bytes| bytes.checked_add(total))
            .filter(|bytes| *bytes <= self.max_bytes)
        else {
            return false;
        };
        self.retained_bytes = bytes;
        self.trace_bytes = total;
        true
    }

    fn handle_entry(&self, handle: CacheHandle, at: Instant, rank: usize) -> Option<&Entry> {
        if self.at != at
            || self.handle_owner != Some(handle.owner)
            || self.generations.get(handle.slot).copied() != Some(handle.generation.get())
        {
            return None;
        }
        self.slots.get(handle.slot)?.as_ref().filter(|entry| {
            entry.exact
                && entry.execution == CostExecution::SingleInvocation
                && entry.samples > 0
                && entry.observed_at <= at
                && at < entry.expires_at
                && entry.service_time(rank).is_some_and(|time| !time.is_zero())
        })
    }

    fn handle_at(&self, slot: usize, rank: usize) -> Option<CacheHandle> {
        let handle = CacheHandle {
            owner: self.handle_owner?,
            slot,
            generation: NonZeroU64::new(*self.generations.get(slot)?)?,
        };
        self.handle_entry(handle, self.at, rank).map(|_| handle)
    }

    /// Borrow only while the caller holds the cache lock. Every group is checked
    /// before wave copying, and checked again at its original prediction point.
    pub(super) fn handle_work(
        &self,
        handle: CacheHandle,
        at: Instant,
        rank: usize,
    ) -> Option<&[CostWork]> {
        self.handle_entry(handle, at, rank)
            .map(|entry| entry.work.as_ref())
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
        let (prediction, key, _) = self.get_with_handle(query, at, rank, None);
        (prediction, key)
    }

    /// `handle` is trusted only by the sealed replay path: its query work was
    /// copied from this handle's canonical entry, and the complete wave passed
    /// the original builder checks. Ordinary lookup always supplies None and
    /// retains full key equality. A stale handle falls back using the current
    /// query and the SAME ordinal; no second lookup charge/cursor advance.
    pub(super) fn get_with_handle<'a>(
        &mut self,
        query: &CostQuery<'a>,
        at: Instant,
        rank: usize,
        handle: Option<CacheHandle>,
    ) -> (
        Option<CostPrediction>,
        Option<InsertionKey<'a>>,
        Option<CacheHandle>,
    ) {
        if self.at != at || self.slots.is_empty() {
            return (None, None, None);
        }
        let ordinal = (self.next_hint < self.hints.len()).then_some(self.next_hint);
        self.next_hint = self.next_hint.saturating_add(1);
        let trusted = handle
            .filter(|_| query.work_is_exact && query.execution == CostExecution::SingleInvocation)
            .and_then(|handle| {
                self.handle_entry(handle, at, rank)
                    .filter(|entry| entry.work.len() == query.work.len())
                    .and_then(|entry| entry.prediction(rank))
                    .map(|prediction| (handle, prediction))
            });
        if let Some((handle, prediction)) = trusted {
            self.remember_hint(ordinal, handle.slot);
            self.hits = self.hits.saturating_add(1);
            #[cfg(test)]
            {
                self.handle_hits += 1;
            }
            return (Some(prediction), None, Some(handle));
        }
        let hinted = ordinal
            .and_then(|ordinal| self.hints[ordinal])
            .and_then(|slot| {
                let slot = slot.get() - 1;
                self.slots
                    .get(slot)
                    .and_then(Option::as_ref)
                    .filter(|entry| entry.matches(query))
                    .and_then(|entry| entry.prediction(rank))
                    .map(|prediction| (slot, prediction))
            });
        if let Some((slot, prediction)) = hinted {
            self.hits = self.hits.saturating_add(1);
            #[cfg(test)]
            {
                self.hint_hits += 1;
            }
            return (Some(prediction), None, self.handle_at(slot, rank));
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
            return (Some(prediction), None, self.handle_at(slot, rank));
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
            None,
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
        let _ = self.insert_with_handle(key, prediction);
    }

    pub(super) fn insert_with_handle(
        &mut self,
        key: Option<InsertionKey<'_>>,
        prediction: CostScenarioPrediction,
    ) -> Option<CacheHandle> {
        let key = key?;
        if self.at != key.at || self.slots.len() != key.slots_len || key.slot >= self.slots.len() {
            return None;
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
            return None;
        };
        // The provider runs without the cache lock. Search CURRENT entries for
        // this exact key before considering empties, so an interleaved insert
        // cannot create two copies with different distributions. A third
        // colliding key deterministically replaces its primary slot.
        let matching_slot = self.probe_slots(key.slot).find(|slot| {
            self.slots[*slot]
                .as_ref()
                .is_some_and(|entry| entry.matches(&query))
        });
        let slot = matching_slot
            .or_else(|| {
                self.probe_slots(key.slot)
                    .find(|slot| self.slots[*slot].is_none())
            })
            .unwrap_or(key.slot);
        #[cfg(test)]
        if let Some(accounting) = &mut self.accounting {
            accounting.counts.known_insert_attempts =
                accounting.counts.known_insert_attempts.saturating_add(1);
        }
        let previous = self.slots[slot].as_ref().map_or(0, Entry::payload_bytes);
        let time_encoding = TimeEncoding::select(&scenarios);
        let time_bytes = time_encoding.payload_bytes(scenarios.len());
        let Some(bytes) = query
            .work
            .len()
            .checked_mul(std::mem::size_of::<CostWork>())
            .and_then(|work| work.checked_add(time_bytes?))
            .and_then(|new| self.retained_bytes.checked_sub(previous)?.checked_add(new))
        else {
            #[cfg(test)]
            if let Some(accounting) = &mut self.accounting {
                accounting.counts.arithmetic_rejections =
                    accounting.counts.arithmetic_rejections.saturating_add(1);
            }
            return None;
        };
        if bytes > self.max_bytes {
            #[cfg(test)]
            {
                let before = self.accounting_bytes();
                if let Some(accounting) = &mut self.accounting {
                    let rejected = accounting::RejectedInsertion {
                        before,
                        victim_payload_bytes: previous,
                        proposed_retained_bytes: bytes,
                    };
                    accounting.counts.byte_limit_rejections =
                        accounting.counts.byte_limit_rejections.saturating_add(1);
                    accounting
                        .counts
                        .first_byte_rejection
                        .get_or_insert(rejected);
                    accounting.counts.last_byte_rejection = Some(rejected);
                }
            }
            return None;
        }
        #[cfg(test)]
        if let Some(accounting) = &mut self.accounting {
            let (old_work, old_times) = self.slots[slot].as_ref().map_or((0, 0), |entry| {
                (
                    std::mem::size_of_val(entry.work.as_ref()),
                    std::mem::size_of_val(entry.service_times.as_ref()),
                )
            });
            let count = if self.slots[slot].is_none() {
                accounting.counts.occupied_slots += 1;
                &mut accounting.counts.empty_insertions
            } else if matching_slot.is_some() {
                &mut accounting.counts.exact_key_replacements
            } else {
                &mut accounting.counts.collision_replacements
            };
            *count = count.saturating_add(1);
            accounting.work_payload =
                accounting.work_payload - old_work + std::mem::size_of_val(query.work);
            accounting.service_time_payload = accounting.service_time_payload - old_times
                + time_bytes.expect("validated encoded payload bytes");
        }
        // Release a colliding payload before allocating its replacement key.
        self.slots[slot] = None;
        self.slots[slot] = Some(Entry {
            work: query.work.into(),
            exact: query.work_is_exact,
            execution: query.execution,
            service_times: time_encoding.pack(scenarios),
            time_encoding,
            dependence,
            support,
            samples,
            observed_at,
            expires_at,
        });
        self.retained_bytes = bytes;
        if self.handle_owner.is_some() {
            if let Some(generation) = self.last_generation.checked_add(1) {
                self.last_generation = generation;
                self.generations[slot] = generation;
            } else {
                // Never wrap or revive an old handle. Ordinary lookup remains
                // usable; allocated metadata stays charged until cache drop.
                self.handle_owner = None;
            }
        }
        // The provider ran outside the lock. Associate the actual current slot
        // with the original lookup ordinal, not a later cursor or primary slot.
        self.remember_hint(key.hint_ordinal, slot);
        self.handle_at(slot, 0)
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
