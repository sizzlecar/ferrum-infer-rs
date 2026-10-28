//! Materializer-owned converted groups; no global/source-path cache.
use super::*;
use std::collections::VecDeque;
use std::ops::Range;
use std::sync::{Arc, Mutex, MutexGuard, Weak};

// Bounding payload bytes alone would permit excessive metadata for tiny scale
// subsets. Complete model groups already have the same 4096-group limit.
const MAXIMUM_ENTRIES: usize = 4096;

pub(super) struct HostCache {
    maximum_bytes: u64,
    // Includes the complete ordered group configuration and implementation.
    implementation: String,
    state: Mutex<State>,
}

#[derive(Default)]
pub(super) struct State {
    entries: VecDeque<Entry>,
    retained_bytes: u64,
    retentions: Vec<Weak<Retention>>,
}

#[derive(Clone)]
pub(super) struct Key {
    implementation: String,
    sources: Vec<WeightComponentSpec>,
    outputs: Vec<WeightComponentSpec>,
    // Retaining actual immutable owners prevents address reuse/ABA. The trait
    // contract forbids mutation while these owners exist. Metadata alone never
    // proves that a new archive contains the same weights.
    regions: Vec<RetainedHostMemoryRegion>,
}

#[derive(Clone)]
struct Entry {
    key: Key,
    backing: ManagedHostBytes,
    ranges: Vec<Range<usize>>,
}

/// Request-owned aliases retain their original Host claims, like upload aliases.
/// Future publications for declared keys join this lease so repeated prefix
/// initialization cannot reconvert them after another plan evicts the LRU.
pub(super) struct Retention {
    keys: Vec<Key>,
    entries: Mutex<Vec<Entry>>,
}

/// Read-only admission projection. It retains source identity, never converted
/// payload owners, and cannot change live cache order or evict a live entry.
pub(super) struct Forecast {
    maximum_bytes: u64,
    retained_bytes: u64,
    entries: VecDeque<(Key, u64)>,
}

impl Key {
    fn matches(&self, other: &Self) -> bool {
        self.implementation == other.implementation
            && self.sources == other.sources
            && self.outputs == other.outputs
            && self.regions.len() == other.regions.len()
            && self
                .regions
                .iter()
                .zip(&other.regions)
                .all(|(left, right)| std::ptr::eq(left.bytes(), right.bytes()))
    }
}

impl HostCache {
    pub(super) fn new(maximum_bytes: NonZeroU64, implementation: String) -> Self {
        Self {
            maximum_bytes: maximum_bytes.get(),
            implementation,
            state: Mutex::new(State::default()),
        }
    }

    pub(super) fn key(
        &self,
        sources: &[&WeightComponentSpec],
        outputs: &[&WeightComponentSpec],
        payloads: &[WeightComponentPayload<'_>],
    ) -> Result<Key, VNextError> {
        Ok(Key {
            implementation: self.implementation.clone(),
            sources: sources.iter().map(|spec| (*spec).clone()).collect(),
            outputs: outputs.iter().map(|spec| (*spec).clone()).collect(),
            regions: payloads
                .iter()
                .map(|payload| {
                    payload.retained_host_memory().cloned().ok_or_else(|| {
                        invalid("conversion cache requires retained immutable source regions")
                    })
                })
                .collect::<Result<_, _>>()?,
        })
    }

    pub(super) fn lock(&self) -> Result<MutexGuard<'_, State>, VNextError> {
        self.state
            .lock()
            .map_err(|_| invalid("Host conversion cache is poisoned"))
    }

    pub(super) fn fits(&self, bytes: usize) -> bool {
        u64::try_from(bytes).is_ok_and(|bytes| bytes <= self.maximum_bytes)
    }

    pub(super) fn forecast(&self) -> Result<Forecast, VNextError> {
        let state = self.lock()?;
        Ok(Forecast {
            maximum_bytes: self.maximum_bytes,
            retained_bytes: state.retained_bytes,
            entries: state
                .entries
                .iter()
                .map(|entry| (entry.key.clone(), entry.backing.len() as u64))
                .collect(),
        })
    }

    pub(super) fn retain(
        &self,
        keys: Vec<Option<Key>>,
    ) -> Result<(Vec<bool>, Arc<Retention>), VNextError> {
        let mut state = self.lock()?;
        state.prune_retentions();
        let mut hits = Vec::with_capacity(keys.len());
        let mut entries = Vec::new();
        for key in &keys {
            let entry = match key {
                Some(key) => state.entry(key)?,
                None => None,
            };
            hits.push(entry.is_some());
            if let Some(entry) = entry {
                entries.push(entry);
            }
        }
        let retention = Arc::new(Retention {
            keys: keys.into_iter().flatten().collect(),
            entries: Mutex::new(entries),
        });
        state.retentions.push(Arc::downgrade(&retention));
        Ok((hits, retention))
    }

    pub(super) fn make_room(&self, state: &mut State, bytes: usize) {
        let bytes = bytes as u64;
        while state.retained_bytes > self.maximum_bytes - bytes
            || state.entries.len() >= MAXIMUM_ENTRIES
        {
            let entry = state
                .entries
                .pop_front()
                .expect("nonempty bounded cache needs eviction");
            state.retained_bytes -= entry.backing.len() as u64;
            // Upload aliases still retain this backing's original Host claim.
            drop(entry);
        }
    }
}

impl Forecast {
    /// Project the same hit promotion / miss eviction as the real converter.
    /// The caller has already checked that this complete group fits the cap.
    pub(super) fn hit_or_insert(&mut self, key: Key, bytes: usize) -> bool {
        if let Some(index) = self.entries.iter().position(|entry| entry.0.matches(&key)) {
            let entry = self.entries.remove(index).expect("matched forecast entry");
            self.entries.push_back(entry);
            return true;
        }
        let bytes = bytes as u64;
        while self.retained_bytes > self.maximum_bytes - bytes
            || self.entries.len() >= MAXIMUM_ENTRIES
        {
            let (_, bytes) = self.entries.pop_front().expect("bounded forecast entry");
            self.retained_bytes -= bytes;
        }
        self.retained_bytes += bytes;
        self.entries.push_back((key, bytes));
        false
    }
}

impl State {
    fn prune_retentions(&mut self) {
        self.retentions
            .retain(|retention| retention.strong_count() != 0);
    }

    fn entry(&self, key: &Key) -> Result<Option<Entry>, VNextError> {
        if let Some(entry) = self.entries.iter().find(|entry| entry.key.matches(key)) {
            return Ok(Some(entry.clone()));
        }
        for retention in self.retentions.iter().filter_map(Weak::upgrade) {
            let entries = retention
                .entries
                .lock()
                .map_err(|_| invalid("Host conversion retention is poisoned"))?;
            if let Some(entry) = entries.iter().find(|entry| entry.key.matches(key)) {
                return Ok(Some(entry.clone()));
            }
        }
        Ok(None)
    }

    pub(super) fn get(
        &mut self,
        key: &Key,
    ) -> Result<Option<(ManagedHostBytes, Vec<Range<usize>>)>, VNextError> {
        self.prune_retentions();
        if let Some(index) = self.entries.iter().position(|entry| entry.key.matches(key)) {
            let entry = self
                .entries
                .remove(index)
                .expect("matched cache entry exists");
            let result = (entry.backing.clone(), entry.ranges.clone());
            self.entries.push_back(entry);
            return Ok(Some(result));
        }
        Ok(self.entry(key)?.map(|entry| (entry.backing, entry.ranges)))
    }

    pub(super) fn publish(
        &mut self,
        key: Key,
        backing: ManagedHostBytes,
        ranges: Vec<Range<usize>>,
    ) -> Result<(), VNextError> {
        let entry = Entry {
            key,
            backing,
            ranges,
        };
        self.prune_retentions();
        for retention in self.retentions.iter().filter_map(Weak::upgrade) {
            if retention.keys.iter().any(|key| key.matches(&entry.key)) {
                let mut entries = retention
                    .entries
                    .lock()
                    .map_err(|_| invalid("Host conversion retention is poisoned"))?;
                if !entries
                    .iter()
                    .any(|retained| retained.key.matches(&entry.key))
                {
                    entries.push(entry.clone());
                }
            }
        }
        self.retained_bytes += entry.backing.len() as u64;
        self.entries.push_back(entry);
        Ok(())
    }
}
