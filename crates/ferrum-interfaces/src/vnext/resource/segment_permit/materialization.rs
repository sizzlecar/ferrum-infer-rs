//! Diagnostic-only scalar keys. They neither retain nor authorize backing.
use super::super::{BackingSegment, BackingSegmentLease, ResidentChunkBacking};
use super::*;
use crate::vnext::ExecutionLaneId;

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct OwnerKey {
    lease: *const BackingSegmentLease,
    chunk: *const (),
    lane: Option<ExecutionLaneId>,
}

#[derive(Clone, Copy)]
struct Entry {
    owner: OwnerKey,
    group: usize,
    offset: u64,
    length: u64,
}

pub(super) struct MaterializationDiagnostics {
    entries: Option<Vec<Entry>>,
    sampled_bindings: u64,
    sampled_ns: u64,
}

impl MaterializationDiagnostics {
    #[inline(always)]
    pub(super) fn new<S: SubmissionWaveDispatchTimingSink>(bindings: usize) -> Self {
        let entries = if S::ENABLED {
            let mut entries = Vec::new();
            // Diagnostics cannot introduce a new correctness failure. A failed
            // reserve leaves this observation unavailable, rather than zero.
            entries.try_reserve_exact(bindings).ok().map(|()| entries)
        } else {
            None
        };
        Self {
            entries,
            sampled_bindings: 0,
            sampled_ns: 0,
        }
    }

    #[inline(always)]
    pub(super) fn begin_sample<S: SubmissionWaveDispatchTimingSink>(
        &self,
        index: usize,
    ) -> Option<Instant> {
        self.begin_sample_with::<S>(index, Instant::now)
    }

    #[inline(always)]
    fn begin_sample_with<S: SubmissionWaveDispatchTimingSink>(
        &self,
        index: usize,
        clock: impl FnOnce() -> Instant,
    ) -> Option<Instant> {
        (S::ENABLED
            && self.entries.is_some()
            && index
                .is_multiple_of(SegmentBackingMaterializationObservation::SAMPLING_STRIDE as usize))
        .then(clock)
    }

    #[inline(always)]
    pub(super) fn finish_sample(&mut self, started: Option<Instant>) {
        if let Some(started) = started {
            self.sampled_bindings = self.sampled_bindings.saturating_add(1);
            self.sampled_ns = self.sampled_ns.saturating_add(ns(started.elapsed()));
        }
    }

    #[inline(always)]
    pub(super) fn record<S: SubmissionWaveDispatchTimingSink, B>(
        &mut self,
        group: usize,
        authority: &LogicalBackingSliceAuthority,
        chunk: &Arc<ResidentChunkBacking<B>>,
        segment: &BackingSegment,
    ) {
        if S::ENABLED {
            if let Some(entries) = &mut self.entries {
                entries.push(Entry {
                    owner: OwnerKey {
                        lease: Arc::as_ptr(&authority.segment_lease),
                        chunk: Arc::as_ptr(chunk).cast(),
                        lane: authority.reusable_lane,
                    },
                    group,
                    offset: segment.offset_bytes(),
                    length: segment.length_bytes(),
                });
            }
        }
    }

    pub(super) fn finish<S: SubmissionWaveDispatchTimingSink>(
        mut self,
        groups: usize,
        complete: bool,
    ) -> Option<SegmentBackingMaterializationObservation> {
        if !S::ENABLED {
            return None;
        }
        let mut entries = self.entries.take()?;
        let started = Instant::now();
        entries.sort_unstable_by_key(|entry| (entry.owner, entry.offset, entry.length));
        let owners = distinct(&entries, |entry| entry.owner);
        let ranges = distinct(&entries, |entry| (entry.owner, entry.offset, entry.length));
        entries.sort_unstable_by_key(|entry| (entry.owner, entry.group));
        let pairs = distinct(&entries, |entry| (entry.owner, entry.group));
        Some(SegmentBackingMaterializationObservation {
            complete,
            unique_authority_groups: count(groups),
            segment_bindings: count(entries.len()),
            unique_lease_chunk_scope_owners: count(owners),
            owner_tuple_group_pairs: count(pairs),
            unique_owner_segment_ranges: count(ranges),
            sampling_stride: SegmentBackingMaterializationObservation::SAMPLING_STRIDE,
            sampled_bindings: self.sampled_bindings,
            sampled_materialization_ns: (self.sampled_bindings > 0).then_some(self.sampled_ns),
            diagnostic_counting_ns: ns(started.elapsed()),
        })
    }
}

fn distinct<K: PartialEq>(entries: &[Entry], key: impl Fn(&Entry) -> K) -> usize {
    entries.first().map_or(0, |_| 1)
        + entries
            .windows(2)
            .filter(|pair| key(&pair[0]) != key(&pair[1]))
            .count()
}

fn count(value: usize) -> u64 {
    u64::try_from(value).unwrap_or(u64::MAX)
}
fn ns(value: Duration) -> u64 {
    u64::try_from(value.as_nanos()).unwrap_or(u64::MAX)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vnext::operation::DisabledSubmissionWaveDispatchTimingSink;

    #[test]
    fn disabled_materialization_has_no_storage_clock_or_observation() {
        type Off = DisabledSubmissionWaveDispatchTimingSink;
        let diagnostic = MaterializationDiagnostics::new::<Off>(usize::MAX);
        assert!(diagnostic.entries.is_none());
        assert!(diagnostic
            .begin_sample_with::<Off>(0, || panic!("Off read a clock"))
            .is_none());
        assert!(diagnostic.finish::<Off>(0, false).is_none());
    }

    #[test]
    fn materialization_sampling_reads_clock_only_at_declared_stride() {
        struct On;
        impl crate::vnext::DeviceSubmissionTimingSink for On {
            const ENABLED: bool = true;
            fn record_device_submission(
                &self,
                _: crate::vnext::DeviceSubmissionStage,
                _: Duration,
            ) {
            }
        }
        impl SubmissionWaveDispatchTimingSink for On {
            fn record(&self, _: SubmissionWaveDispatchStage, _: Duration) {}
        }
        let diagnostic = MaterializationDiagnostics::new::<On>(65);
        let clocks = std::cell::Cell::new(0);
        for index in 0..=64 {
            let started = diagnostic.begin_sample_with::<On>(index, || {
                clocks.set(clocks.get() + 1);
                Instant::now()
            });
            assert_eq!(started.is_some(), index == 0 || index == 64);
        }
        assert_eq!(clocks.get(), 2);
    }
}
