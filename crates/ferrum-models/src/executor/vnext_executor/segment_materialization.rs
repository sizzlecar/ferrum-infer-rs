//! Raw sampled materialization and same-permit owner cardinalities, not a
//! performance estimate. Receipts contain no live ownership or addresses.
use super::*;

#[derive(Default)]
pub(super) struct MaterializationMetrics {
    observations: AtomicU64,
    complete_observations: AtomicU64,
    groups: AtomicU64,
    bindings: AtomicU64,
    owners: AtomicU64,
    pairs: AtomicU64,
    ranges: AtomicU64,
    samples: AtomicU64,
    sample_ns: AtomicU64,
    counting_ns: AtomicU64,
}

impl MaterializationMetrics {
    pub(super) fn record(&self, observation: SegmentBackingMaterializationObservation) {
        for (counter, value) in [
            (&self.observations, 1),
            (&self.complete_observations, u64::from(observation.complete)),
            (&self.groups, observation.unique_authority_groups),
            (&self.bindings, observation.segment_bindings),
            (&self.owners, observation.unique_lease_chunk_scope_owners),
            (&self.pairs, observation.owner_tuple_group_pairs),
            (&self.ranges, observation.unique_owner_segment_ranges),
            (&self.samples, observation.sampled_bindings),
            (
                &self.sample_ns,
                observation.sampled_materialization_ns.unwrap_or(0),
            ),
            (&self.counting_ns, observation.diagnostic_counting_ns),
        ] {
            let _ = counter.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
                Some(current.saturating_add(value))
            });
        }
    }

    pub(super) fn snapshot(&self) -> serde_json::Value {
        let observations = self.observations.load(Ordering::Relaxed);
        let samples = self.samples.load(Ordering::Relaxed);
        serde_json::json!({
            "scope": "same_permit_after_exact_group_dedup_including_partial_checked_errors",
            "limitations": "diagnostic_only_nonatomic_saturating_totals_include_partial_permits_do_not_divide_totals_by_segment_hits_complete_means_permit_checks_only_raw_retention_segment_clone_binding_push_samples_include_clock_overhead_exclude_validation_and_chunk_lookup_not_total_materialization_or_removable_cpu_time_counting_after_unlock_is_inside_outer_host_interval_and_excludes_reserve_entry_collection_sample_clocks_callbacks",
            "observations": observations,
            "complete_observations": self.complete_observations.load(Ordering::Relaxed),
            "unique_authority_groups": self.groups.load(Ordering::Relaxed),
            "segment_bindings": self.bindings.load(Ordering::Relaxed),
            "unique_lease_chunk_scope_owners": self.owners.load(Ordering::Relaxed),
            "owner_tuple_group_pairs": self.pairs.load(Ordering::Relaxed),
            "unique_owner_segment_ranges": self.ranges.load(Ordering::Relaxed),
            "sampling_stride": SegmentBackingMaterializationObservation::SAMPLING_STRIDE,
            "sampled_bindings": samples,
            "sampled_materialization_ns": (samples > 0).then(|| self.sample_ns.load(Ordering::Relaxed)),
            "diagnostic_counting_ns": (observations > 0).then(|| self.counting_ns.load(Ordering::Relaxed)),
        })
    }

    pub(super) fn reset(&self) {
        for counter in [
            &self.observations,
            &self.complete_observations,
            &self.groups,
            &self.bindings,
            &self.owners,
            &self.pairs,
            &self.ranges,
            &self.samples,
            &self.sample_ns,
            &self.counting_ns,
        ] {
            counter.store(0, Ordering::Relaxed);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn materialization_health_keeps_missing_samples_null_and_resets() {
        let metrics = VNextWaveTimingMetrics::default();
        let phase = VNextWaveTimingMetrics::default();
        let sink = VNextWaveTimingSink {
            aggregate: &metrics,
            phase: &phase,
        };
        assert!(
            metrics.segment_backing_materialization.snapshot()["sampled_materialization_ns"]
                .is_null()
        );
        assert!(
            metrics.segment_backing_materialization.snapshot()["diagnostic_counting_ns"].is_null()
        );
        let mut observation = SegmentBackingMaterializationObservation {
            complete: false,
            unique_authority_groups: 3,
            segment_bindings: 0,
            unique_lease_chunk_scope_owners: 0,
            owner_tuple_group_pairs: 0,
            unique_owner_segment_ranges: 0,
            sampling_stride: SegmentBackingMaterializationObservation::SAMPLING_STRIDE,
            sampled_bindings: 0,
            sampled_materialization_ns: None,
            diagnostic_counting_ns: 5,
        };
        sink.record_segment_backing_materialization(observation);
        assert!(
            metrics.segment_backing_materialization.snapshot()["sampled_materialization_ns"]
                .is_null()
        );
        observation.complete = true;
        observation.segment_bindings = 3;
        observation.unique_lease_chunk_scope_owners = 1;
        observation.owner_tuple_group_pairs = 2;
        observation.unique_owner_segment_ranges = 2;
        observation.sampled_bindings = 1;
        observation.sampled_materialization_ns = Some(10);
        sink.record_segment_backing_materialization(observation);
        let value = metrics.segment_backing_permit_metadata_breakdown()["materialization"].clone();
        assert_eq!(value["observations"], 2);
        assert_eq!(value["complete_observations"], 1);
        assert_eq!(value["segment_bindings"], 3);
        assert_eq!(value["unique_lease_chunk_scope_owners"], 1);
        assert_eq!(value["sampled_materialization_ns"], 10);
        assert_eq!(phase.segment_backing_materialization.snapshot(), value);
        metrics.reset();
        let value = metrics.segment_backing_materialization.snapshot();
        assert_eq!(value["observations"], 0);
        assert_eq!(value["sampled_bindings"], 0);
        assert!(value["sampled_materialization_ns"].is_null());
    }
}
