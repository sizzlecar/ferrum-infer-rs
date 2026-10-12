//! Raw sampled materialization and same-permit owner cardinalities, not a
//! performance estimate. Receipts contain no live ownership or addresses.
use super::*;

/// Private adapter keeps the continuity state machine independently testable;
/// the interface snapshot alone implements real lease/chunk weak identity.
pub(super) trait OwnerSnapshotEvidence {
    fn owner_count(&self) -> usize;
    fn compare_owners(&self, previous: &Self) -> [SegmentBackingOwnerContinuity; 5];
}

impl OwnerSnapshotEvidence for SegmentBackingOwnerSnapshot {
    fn owner_count(&self) -> usize {
        self.len()
    }
    fn compare_owners(&self, previous: &Self) -> [SegmentBackingOwnerContinuity; 5] {
        self.compare(previous)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum OwnerContinuityStatus {
    Baseline,
    Compared,
    Miss,
    Error,
    RetryBaseline,
    SnapshotUnavailable,
    ProgramUnavailable,
    EmptySnapshot,
    ContextDiscontinuity,
    NonAdjacentWave,
    OutOfOrderWave,
    DuplicateWave,
    IncompleteDeclarations,
    RepeatedCallback,
    BackendIdentityMismatch,
    PendingWaveOutcome,
    SupersededAttempt,
}

#[derive(Serialize)]
pub(super) struct OwnerContinuityRecord {
    status: OwnerContinuityStatus,
    comparison: Option<[LifetimeContinuityRecord; 5]>,
    comparison_ns: Option<u64>,
    backend_observations_available: Option<bool>,
}

impl OwnerContinuityRecord {
    pub(super) fn pending() -> Self {
        Self {
            status: OwnerContinuityStatus::PendingWaveOutcome,
            comparison: None,
            comparison_ns: None,
            backend_observations_available: None,
        }
    }
    pub(super) fn superseded() -> Self {
        Self {
            status: OwnerContinuityStatus::SupersededAttempt,
            comparison: None,
            comparison_ns: None,
            backend_observations_available: None,
        }
    }
}

#[derive(Serialize)]
struct LifetimeContinuityRecord {
    lifetime: AllocationLifetime,
    comparison_available: bool,
    counts: Option<SegmentBackingOwnerContinuity>,
}

pub(super) struct OwnerAttemptContext<'a> {
    pub wave_ordinal: u64,
    pub attempt_ordinal: usize,
    pub program_id: Option<&'a DeviceReusableExecutionProgramId>,
    pub lane_epoch: u64,
    pub catalog_epoch: Option<u64>,
    pub submitted: bool,
    pub hit: bool,
    pub catalog_miss: bool,
    pub incomplete_declarations: bool,
    pub repeated_callback: bool,
    pub backend_identity_mismatch: bool,
    pub backend_identity_available: bool,
}

struct PreviousOwners<S> {
    wave_ordinal: u64,
    program_id: DeviceReusableExecutionProgramId,
    lane_epoch: u64,
    catalog_epoch: Option<u64>,
    snapshot: S,
    backend_identity_available: bool,
}

#[derive(Serialize)]
struct LifetimeContinuityTotals {
    lifetime: AllocationLifetime,
    compared_pairs: u64,
    present_pairs: u64,
    previous_owners: u64,
    current_owners: u64,
    shared_owners: u64,
    added_owners: u64,
    removed_owners: u64,
    shared_owners_with_changed_ranges: u64,
    shared_owners_with_changed_logical_windows: u64,
    owner_set_equal_pairs: u64,
    range_changed_pairs: u64,
    logical_window_changed_pairs: u64,
}

impl LifetimeContinuityTotals {
    fn new(lifetime: AllocationLifetime) -> Self {
        Self {
            lifetime,
            compared_pairs: 0,
            present_pairs: 0,
            previous_owners: 0,
            current_owners: 0,
            shared_owners: 0,
            added_owners: 0,
            removed_owners: 0,
            owner_set_equal_pairs: 0,
            shared_owners_with_changed_ranges: 0,
            shared_owners_with_changed_logical_windows: 0,
            range_changed_pairs: 0,
            logical_window_changed_pairs: 0,
        }
    }
    fn record(&mut self, value: SegmentBackingOwnerContinuity) {
        let present = value.previous_owners != 0 || value.current_owners != 0;
        self.compared_pairs = self.compared_pairs.saturating_add(u64::from(present));
        self.present_pairs = self.present_pairs.saturating_add(u64::from(present));
        for (counter, increment) in [
            (&mut self.previous_owners, value.previous_owners),
            (&mut self.current_owners, value.current_owners),
            (&mut self.shared_owners, value.shared_owners),
            (&mut self.added_owners, value.added_owners),
            (&mut self.removed_owners, value.removed_owners),
            (
                &mut self.shared_owners_with_changed_ranges,
                value.shared_owners_with_changed_ranges,
            ),
            (
                &mut self.shared_owners_with_changed_logical_windows,
                value.shared_owners_with_changed_logical_windows,
            ),
            (
                &mut self.owner_set_equal_pairs,
                u64::from(present && value.owner_set_equal),
            ),
            (
                &mut self.range_changed_pairs,
                u64::from(present && !value.range_set_equal),
            ),
            (
                &mut self.logical_window_changed_pairs,
                u64::from(present && !value.logical_window_set_equal),
            ),
        ] {
            *counter = counter.saturating_add(increment);
        }
    }
}

/// Only the previous qualified attempt is retained. Snapshot ownership is weak;
/// comparison never upgrades it and never changes admission/dispatch.
pub(super) struct OwnerContinuityTracker<S = SegmentBackingOwnerSnapshot> {
    previous: Option<PreviousOwners<S>>,
    high_water_wave: Option<u64>,
    attempts: u64,
    submitted_hits: u64,
    available_snapshots: u64,
    unavailable_snapshots: u64,
    compared_pairs: u64,
    comparison_ns: u64,
    comparisons_without_backend_observation: u64,
    entire_owner_set_equal_pairs: u64,
    program_discontinuities: u64,
    context_discontinuities: u64,
    boundaries: BTreeMap<&'static str, u64>,
    lifetimes: [LifetimeContinuityTotals; 5],
}

impl<S: OwnerSnapshotEvidence> Default for OwnerContinuityTracker<S> {
    fn default() -> Self {
        Self {
            previous: None,
            high_water_wave: None,
            attempts: 0,
            submitted_hits: 0,
            available_snapshots: 0,
            unavailable_snapshots: 0,
            compared_pairs: 0,
            comparison_ns: 0,
            comparisons_without_backend_observation: 0,
            entire_owner_set_equal_pairs: 0,
            program_discontinuities: 0,
            context_discontinuities: 0,
            boundaries: BTreeMap::new(),
            lifetimes: [
                AllocationLifetime::Plan,
                AllocationLifetime::Request,
                AllocationLifetime::Sequence,
                AllocationLifetime::Step,
                AllocationLifetime::Invocation,
            ]
            .map(LifetimeContinuityTotals::new),
        }
    }
}

impl<S: OwnerSnapshotEvidence> OwnerContinuityTracker<S> {
    pub(super) fn break_chain(&mut self, reason: &'static str) {
        self.previous = None;
        let count = self.boundaries.entry(reason).or_default();
        *count = count.saturating_add(1);
    }

    pub(super) fn observe(
        &mut self,
        context: OwnerAttemptContext<'_>,
        snapshot: Option<S>,
    ) -> OwnerContinuityRecord {
        self.attempts = self.attempts.saturating_add(1);
        if snapshot.is_some() {
            self.available_snapshots = self.available_snapshots.saturating_add(1);
        } else {
            self.unavailable_snapshots = self.unavailable_snapshots.saturating_add(1);
        }
        if context.submitted && context.hit {
            self.submitted_hits = self.submitted_hits.saturating_add(1);
        }
        let rejected = if !context.submitted {
            Some((OwnerContinuityStatus::Error, "error"))
        } else if !context.hit || context.catalog_miss {
            Some((OwnerContinuityStatus::Miss, "miss"))
        } else if context.incomplete_declarations {
            Some((
                OwnerContinuityStatus::IncompleteDeclarations,
                "incomplete_declarations",
            ))
        } else if context.repeated_callback {
            Some((OwnerContinuityStatus::RepeatedCallback, "repeated_callback"))
        } else if context.backend_identity_mismatch {
            Some((
                OwnerContinuityStatus::BackendIdentityMismatch,
                "backend_identity_mismatch",
            ))
        } else if context.program_id.is_none() {
            Some((
                OwnerContinuityStatus::ProgramUnavailable,
                "program_unavailable",
            ))
        } else if snapshot.is_none() {
            Some((
                OwnerContinuityStatus::SnapshotUnavailable,
                "snapshot_unavailable",
            ))
        } else if snapshot.as_ref().is_some_and(|s| s.owner_count() == 0) {
            Some((OwnerContinuityStatus::EmptySnapshot, "empty_snapshot"))
        } else {
            None
        };
        if self
            .high_water_wave
            .is_some_and(|wave| context.wave_ordinal < wave)
        {
            self.break_chain("out_of_order_wave");
            return OwnerContinuityRecord {
                status: OwnerContinuityStatus::OutOfOrderWave,
                comparison: None,
                comparison_ns: None,
                backend_observations_available: None,
            };
        }
        if self.high_water_wave == Some(context.wave_ordinal) && context.attempt_ordinal == 0 {
            self.break_chain("duplicate_wave");
            return OwnerContinuityRecord {
                status: OwnerContinuityStatus::DuplicateWave,
                comparison: None,
                comparison_ns: None,
                backend_observations_available: None,
            };
        }
        self.high_water_wave = Some(context.wave_ordinal);
        if let Some((status, reason)) = rejected {
            self.break_chain(reason);
            return OwnerContinuityRecord {
                status,
                comparison: None,
                comparison_ns: None,
                backend_observations_available: None,
            };
        }
        let snapshot = snapshot.expect("qualified snapshot exists");
        let program_id = context.program_id.expect("qualified program exists");
        let mut status = OwnerContinuityStatus::Baseline;
        let mut comparison = None;
        let mut comparison_ns = None;
        let mut backend_observations_available = None;
        if context.attempt_ordinal != 0 {
            self.break_chain("retry");
            status = OwnerContinuityStatus::RetryBaseline;
        } else if let Some(previous) = self.previous.as_ref() {
            if previous.program_id != *program_id
                || previous.lane_epoch != context.lane_epoch
                || previous.catalog_epoch != context.catalog_epoch
            {
                self.program_discontinuities = self
                    .program_discontinuities
                    .saturating_add(u64::from(previous.program_id != *program_id));
                self.context_discontinuities = self.context_discontinuities.saturating_add(1);
                self.break_chain("context_discontinuity");
                status = OwnerContinuityStatus::ContextDiscontinuity;
            } else if previous.wave_ordinal.checked_add(1) != Some(context.wave_ordinal) {
                self.break_chain("non_adjacent_wave");
                status = OwnerContinuityStatus::NonAdjacentWave;
            } else {
                let started = Instant::now();
                let compared = snapshot.compare_owners(&previous.snapshot);
                let elapsed = started.elapsed().as_nanos().min(u128::from(u64::MAX)) as u64;
                let backend_available =
                    previous.backend_identity_available && context.backend_identity_available;
                backend_observations_available = Some(backend_available);
                self.comparisons_without_backend_observation = self
                    .comparisons_without_backend_observation
                    .saturating_add(u64::from(!backend_available));
                self.comparison_ns = self.comparison_ns.saturating_add(elapsed);
                comparison_ns = Some(elapsed);
                for (totals, value) in self.lifetimes.iter_mut().zip(compared) {
                    totals.record(value);
                }
                self.compared_pairs = self.compared_pairs.saturating_add(1);
                self.entire_owner_set_equal_pairs =
                    self.entire_owner_set_equal_pairs.saturating_add(u64::from(
                        compared.iter().all(|value| value.owner_set_equal),
                    ));
                comparison = Some(compared.map(|counts| {
                    let present = counts.previous_owners != 0 || counts.current_owners != 0;
                    LifetimeContinuityRecord {
                        lifetime: counts.lifetime,
                        comparison_available: present,
                        counts: present.then_some(counts),
                    }
                }));
                status = OwnerContinuityStatus::Compared;
            }
        }
        self.previous = Some(PreviousOwners {
            wave_ordinal: context.wave_ordinal,
            program_id: program_id.clone(),
            lane_epoch: context.lane_epoch,
            catalog_epoch: context.catalog_epoch,
            snapshot,
            backend_identity_available: context.backend_identity_available,
        });
        OwnerContinuityRecord {
            status,
            comparison,
            comparison_ns,
            backend_observations_available,
        }
    }

    pub(super) fn snapshot(&self) -> serde_json::Value {
        serde_json::json!({
            "scope": "adjacent_recorded_complete_segment_hit_submitted_attempts_same_full_declared_core_program_lane_catalog_context",
            "limitations": "weak_exact_identity_counts_only_not_authority_or_cache_hit_or_terminal_gpu_success_all_lifetimes_empty_pairs_excluded_from_owner_set_equal_pairs_per_lifetime_owner_totals_cover_compared_pairs_only_missing_miss_error_retry_duplicate_overflow_unfinalized_or_nonadjacent_waves_break_continuity_no_addresses_no_tps_gain",
            "attempts": self.attempts, "submitted_hits": self.submitted_hits,
            "available_snapshots": self.available_snapshots, "unavailable_snapshots": self.unavailable_snapshots,
            "compared_pairs": self.compared_pairs, "entire_owner_set_equal_pairs": self.entire_owner_set_equal_pairs,
            "comparison_wall_ns": (self.compared_pairs != 0).then_some(self.comparison_ns),
            "comparisons_without_backend_observation": self.comparisons_without_backend_observation,
            "identity_scope": "full_declared_Core_program_context_not_actual_GPU_graph_cache_hit_missing_backend_observation_is_explicit",
            "logical_window_scope": "distinct_owner_qualified_ResourceView_records_including_authority_logical_bytes_resource_group_committed_view_and_complete_typed_window_set_not_individual_window_count",
            "snapshot_capture_wall_ns": serde_json::Value::Null,
            "observation_cost_scope": "comparison_wall_ns_includes_clock_overhead_only_opaque_compare_excludes_totals_callback_snapshot_capture_and_serialization_comparison_at_wave_finalization_after_existing_host_encode_submit_interval_snapshot_capture_is_inside_existing_prepare_timing_but_independent_counter_unavailable",
            "program_discontinuities": self.program_discontinuities, "context_discontinuities": self.context_discontinuities,
            "boundaries": self.boundaries, "per_lifetime": self.lifetimes,
            "previous_snapshot_present": self.previous.is_some(),
        })
    }

    pub(super) fn reset(&mut self) {
        *self = Self::default();
    }
}

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
