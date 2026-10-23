//! Materializer-owned converted groups; no global/source-path cache.
use super::*;
use std::collections::VecDeque;
use std::ops::Range;
use std::sync::{Mutex, MutexGuard};

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
}

pub(super) struct Key {
    implementation: String,
    sources: Vec<WeightComponentSpec>,
    outputs: Vec<WeightComponentSpec>,
    // Retaining actual immutable owners prevents address reuse/ABA. The trait
    // contract forbids mutation while these owners exist. Metadata alone never
    // proves that a new archive contains the same weights.
    regions: Vec<RetainedHostMemoryRegion>,
}

struct Entry {
    key: Key,
    backing: ManagedHostBytes,
    ranges: Vec<Range<usize>>,
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

impl State {
    pub(super) fn get(&mut self, key: &Key) -> Option<(ManagedHostBytes, Vec<Range<usize>>)> {
        let index = self
            .entries
            .iter()
            .position(|entry| entry.key.matches(key))?;
        let entry = self
            .entries
            .remove(index)
            .expect("matched cache entry exists");
        let result = (entry.backing.clone(), entry.ranges.clone());
        self.entries.push_back(entry);
        Some(result)
    }

    pub(super) fn publish(
        &mut self,
        key: Key,
        backing: ManagedHostBytes,
        ranges: Vec<Range<usize>>,
    ) {
        self.retained_bytes += backing.len() as u64;
        self.entries.push_back(Entry {
            key,
            backing,
            ranges,
        });
    }
}
