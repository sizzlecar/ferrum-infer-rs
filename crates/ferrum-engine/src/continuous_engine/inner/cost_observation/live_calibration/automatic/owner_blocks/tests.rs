use super::*;
use ferrum_types::SloAutomaticCalibrationInputReadinessV1;
use std::num::{NonZeroU64, NonZeroUsize};

#[test]
fn rolling_work_credit_belongs_to_original_blocks_not_source_retries() {
    let settings = SloAutomaticCalibrationSettingsV1::default();
    let schedule = schedule(&settings, &StructuredSettingsV2::default()).unwrap();
    let mut budget = budget::Budget::new(&settings, 1024 * 1024, &schedule, 0).unwrap();
    let allowance = settings.maximum_encoded_source_bytes.get();
    assert!(budget.reserve(1));
    assert!(
        !budget.reserve(2),
        "a second source cannot mint its own allowance"
    );
    budget
        .charge(
            1,
            budget::Work {
                canonical_source_bytes: allowance / 2,
                ..Default::default()
            },
        )
        .unwrap();
    budget.release(1).unwrap();
    assert!(
        !budget.reserve(2),
        "actual spent work is not refunded on source failure"
    );
    budget.complete_original_block(1).unwrap();
    assert!(budget.reserve(2));
    budget.complete_original_block(1).unwrap();
    assert!(
        !budget.reserve(3),
        "ACK/repeated close does not duplicate original block credit"
    );
    assert_eq!(budget.audit().spent.canonical_source_bytes, allowance / 2);
    assert!(budget.release(1).is_err());
    budget.complete_original_block(3).unwrap(); // Block2 failed; it earns no credit.
    assert_eq!(budget.audit().original_blocks_credited, 2);
    assert!(budget.reserve(3));
    assert!(budget.complete_original_block(2).is_err());
}

#[test]
fn rolling_original_ticket_roster_keeps_younger_source_alive_after_old_deadline() {
    let old = tickets::SourceEnrollment {
        generation: 1,
        capture_identity: [1; 32],
        protocol: [2; 32],
        block: 3,
        offered_offset: 16,
        deadline_ns: 10,
    };
    let young = tickets::SourceEnrollment {
        generation: 2,
        capture_identity: [3; 32],
        protocol: [4; 32],
        block: 1,
        offered_offset: 0,
        deadline_ns: 100,
    };
    let window = tickets::Window::with_enrollments(2, 3, 1, 100, vec![old.clone(), young.clone()]);
    let mut ticket = window.reserve(20).unwrap();
    assert_eq!(ticket.ordinal(), 1);
    ticket.bind_call(1);
    ticket.accepted(101);
    assert!(ticket.matches(1, 101, Some(20)));
    assert_eq!(window.enrollment(1), Some(&old));
    assert_eq!(window.enrollment(2), Some(&young));
    assert!(window.enrollment(3).is_none());
    ticket.complete();
    assert!(
        window.complete(21),
        "only each local collector enforces its own expiry"
    );
    assert!(
        !window.complete(101),
        "transport itself remains bounded by the declared last deadline"
    );
}

#[test]
fn rolling_source_and_catalog_partitions_fit_existing_global_retention_sum() {
    for retained in [1, 2, 4, 8] {
        let settings = SloAutomaticCalibrationSettingsV1 {
            maximum_retained_generations: NonZeroUsize::new(retained).unwrap(),
            ..Default::default()
        };
        let generation = settings.maximum_retained_numeric_bytes.get();
        let share = (generation * retained / (retained + 3)).min(generation / 2);
        let native = schedule(&settings, &StructuredSettingsV2::default()).unwrap();
        let budget = budget::Budget::new(&settings, share, &native, 0).unwrap();
        let audit = budget.audit();
        assert_eq!(
            audit.collector_bytes_per_source
                + audit.catalog_bytes_per_source
                + audit.source_metadata_bytes,
            share
        );
        assert!((audit.maximum_source_slots + 2) * share <= generation * retained);
    }
}

#[test]
fn rolling_available_and_reserved_work_share_one_burst_capacity() {
    let settings = SloAutomaticCalibrationSettingsV1::default();
    let native = schedule(&settings, &StructuredSettingsV2::default()).unwrap();
    let mut budget = budget::Budget::new(&settings, 1024 * 1024, &native, 0).unwrap();
    assert!(budget.reserve(1));
    for block in 1..=(settings.maximum_retained_generations.get() as u64 + 3) {
        budget.complete_original_block(block).unwrap();
        let audit = budget.audit();
        assert!(
            audit.available.canonical_source_bytes
                + audit.reserved_remaining.canonical_source_bytes
                <= audit.shared_burst_capacity.canonical_source_bytes
        );
        assert!(
            audit.available.readiness_scalar_visits
                + audit.reserved_remaining.readiness_scalar_visits
                <= audit.shared_burst_capacity.readiness_scalar_visits
        );
        assert!(
            audit.available.readiness_replay_upper_bound
                + audit.reserved_remaining.readiness_replay_upper_bound
                <= audit.shared_burst_capacity.readiness_replay_upper_bound
        );
        assert_eq!(
            audit
                .available
                .add(audit.reserved_remaining)
                .unwrap()
                .add(audit.spent)
                .unwrap()
                .add(audit.forfeited_upper_bound)
                .unwrap()
                .add(audit.discarded_at_burst_limit)
                .unwrap(),
            audit.minted
        );
    }
    budget.forfeit(1).unwrap();
    let audit = budget.audit();
    assert_eq!(audit.spent, budget::Work::default());
    assert!(audit.forfeited_upper_bound.readiness_replay_upper_bound > 0);
    assert_eq!(
        audit
            .available
            .add(audit.reserved_remaining)
            .unwrap()
            .add(audit.spent)
            .unwrap()
            .add(audit.forfeited_upper_bound)
            .unwrap()
            .add(audit.discarded_at_burst_limit)
            .unwrap(),
        audit.minted
    );
    assert_eq!(audit.source_reservations, 0);
}

#[test]
fn rolling_invalid_final_work_releases_custody_without_refunding_unobserved_work() {
    let settings = SloAutomaticCalibrationSettingsV1::default();
    let schedule = schedule(&settings, &StructuredSettingsV2::default()).unwrap();
    let mut ledger = budget::Budget::new(&settings, 1024 * 1024, &schedule, 0).unwrap();
    assert!(ledger.reserve(1));
    let observed = budget::Work {
        canonical_source_bytes: 100,
        ..Default::default()
    };
    ledger.charge(1, observed).unwrap();
    let invalid = budget::Work {
        canonical_source_bytes: settings.maximum_encoded_source_bytes.get() + 1,
        ..Default::default()
    };
    assert!(ledger.charge(1, invalid).is_err()); // Active marks source failure.
    assert!(ledger.close(1, invalid).is_err()); // Retire must still consume custody.
    let audit = ledger.audit();
    assert_eq!(
        audit.source_reservations, 0,
        "no removed source can retain an ownerless reservation"
    );
    assert_eq!(
        audit.spent, observed,
        "out-of-bound data cannot fabricate an observed counter"
    );
    assert_eq!(
        audit.forfeited_upper_bound.canonical_source_bytes,
        settings.maximum_encoded_source_bytes.get() - observed.canonical_source_bytes
    );
    assert_eq!(
        audit
            .available
            .add(audit.reserved_remaining)
            .unwrap()
            .add(audit.spent)
            .unwrap()
            .add(audit.forfeited_upper_bound)
            .unwrap()
            .add(audit.discarded_at_burst_limit)
            .unwrap(),
        audit.minted
    );
    assert!(
        !ledger.reserve(2),
        "cleanup cannot mint an allowance for a retry"
    );
    ledger.complete_original_block(1).unwrap();
    assert!(
        ledger.reserve(2),
        "new genuine input restores admission after failed custody"
    );
    ledger.close(2, budget::Work::default()).unwrap();
    assert_eq!(ledger.audit().source_reservations, 0);
}

#[test]
fn rolling_failed_lease_cleanup_preserves_an_independently_reserved_successor() {
    let settings = SloAutomaticCalibrationSettingsV1::default();
    let native = schedule(&settings, &StructuredSettingsV2::default()).unwrap();
    let mut ledger = budget::Budget::new(&settings, 1024 * 1024, &native, 0).unwrap();
    assert!(ledger.reserve(1));
    ledger.complete_original_block(1).unwrap();
    assert!(ledger.reserve(2));
    let invalid = budget::Work {
        imports: settings.maximum_owners.get() as u64 + 1,
        ..Default::default()
    };
    assert!(ledger.close(1, invalid).is_err());
    assert_eq!(ledger.audit().source_reservations, 1);
    ledger
        .charge(
            2,
            budget::Work {
                canonical_source_bytes: 10,
                ..Default::default()
            },
        )
        .unwrap();
    ledger
        .close(
            2,
            budget::Work {
                canonical_source_bytes: 10,
                ..Default::default()
            },
        )
        .unwrap();
    let audit = ledger.audit();
    assert_eq!(audit.source_reservations, 0);
    assert_eq!(audit.spent.canonical_source_bytes, 10);
    assert_eq!(
        audit
            .available
            .add(audit.reserved_remaining)
            .unwrap()
            .add(audit.spent)
            .unwrap()
            .add(audit.forfeited_upper_bound)
            .unwrap()
            .add(audit.discarded_at_burst_limit)
            .unwrap(),
        audit.minted
    );
}

#[test]
fn automatic_owner_input_readiness_schedule_binds_typed_limits_to_native_members() {
    let settings = SloAutomaticCalibrationSettingsV1::default();
    settings.validate().unwrap();
    let mut numerical = StructuredSettingsV2::default();
    let configured = schedule(&settings, &numerical).unwrap();
    assert_eq!(configured.prediction_validity,
        Some(ferrum_scheduler::implementations::continuous::cost_model::structured_v2::OwnerPredictionValidityPolicyV1::OriginalSampleAgeV1));
    let policy = configured.input_readiness.as_ref().unwrap();
    assert_eq!(
        serde_json::to_value(policy).unwrap()["revision"],
        "work_axes_and_branches_v3"
    );
    assert_eq!(policy.maximum_phase_blocks, [16; 3]);
    assert_eq!(policy.maximum_geometry_visits, 32_000_000);
    assert_eq!(configured.maximum_phase_members, [4096; 3]);
    numerical.max_phase_samples = *configured.maximum_phase_members.iter().max().unwrap();
    configured.validate(&numerical).unwrap();
    numerical.max_phase_samples -= 1;
    assert!(configured.validate(&numerical).is_err());
}

#[test]
fn automatic_owner_input_readiness_explicit_v2_keeps_original_scheduler_revision() {
    let settings = SloAutomaticCalibrationSettingsV1 {
        input_readiness: SloAutomaticCalibrationInputReadinessV1::WorkAxesAndBranchesV2 {
            maximum_phase_blocks: [NonZeroUsize::new(16).unwrap(); 3],
            maximum_geometry_visits: NonZeroU64::new(32_000_000).unwrap(),
        },
        ..Default::default()
    };
    settings.validate().unwrap();
    let numerical = StructuredSettingsV2::default();
    let configured = schedule(&settings, &numerical).unwrap();
    let original = OwnerInputReadinessV1::new_cached_residual_v2([16; 3], 32_000_000).unwrap();
    assert_eq!(configured.input_readiness.as_ref(), Some(&original));
    assert_eq!(
        serde_json::to_value(configured.input_readiness.as_ref().unwrap()).unwrap()["revision"],
        "work_axes_and_branches_v2"
    );
    let defaults = schedule(&SloAutomaticCalibrationSettingsV1::default(), &numerical).unwrap();
    assert_ne!(defaults.input_readiness, configured.input_readiness);
    assert_eq!(
        defaults.maximum_phase_members,
        configured.maximum_phase_members
    );
    assert_eq!(defaults.phase_min_offered, configured.phase_min_offered);
    assert_eq!(defaults.min_members, configured.min_members);
}

#[test]
fn automatic_owner_input_readiness_schedule_keeps_legacy_counts_and_declares_opening_frontier() {
    let settings = SloAutomaticCalibrationSettingsV1 {
        input_readiness: SloAutomaticCalibrationInputReadinessV1::CountOnlyV1 {},
        prediction_validity:
            ferrum_types::SloAutomaticCalibrationPredictionValidityV1::CollectionWindowV1,
        ..Default::default()
    };
    let numerical = StructuredSettingsV2::default();
    let mut configured = schedule(&settings, &numerical).unwrap();
    let original =
        OwnerBlockScheduleV1::new(256, [256; 3], [numerical.min_phase_samples; 3]).unwrap();
    assert_eq!(
        configured.opening_frontier.take(),
        Some(OwnerOpeningFrontierPolicyV1::FirstOfferFifoV1)
    );
    assert_eq!(
        serde_json::to_vec(&configured).unwrap(),
        serde_json::to_vec(&original).unwrap()
    );
    assert!(configured.input_readiness.is_none());
    assert_eq!(configured.maximum_phase_members, [263; 3]);
}

#[test]
fn automatic_owner_input_readiness_schedule_preserves_per_phase_capacity_and_rejects_overflow() {
    let mut settings = SloAutomaticCalibrationSettingsV1 {
        discovery_offered_waves: NonZeroUsize::new(8).unwrap(),
        phase_offered_waves: [NonZeroUsize::new(8).unwrap(); 3],
        input_readiness: SloAutomaticCalibrationInputReadinessV1::WorkAxesAndBranchesV1 {
            maximum_phase_blocks: [1, 2, 3].map(|n| NonZeroUsize::new(n).unwrap()),
            maximum_geometry_visits: NonZeroU64::new(1000).unwrap(),
        },
        ..Default::default()
    };
    settings.validate().unwrap();
    let numerical = StructuredSettingsV2::default();
    let configured = schedule(&settings, &numerical).unwrap();
    assert_eq!(
        serde_json::to_value(configured.input_readiness.as_ref().unwrap()).unwrap()["revision"],
        "work_axes_and_branches_v1"
    );
    assert_eq!(configured.maximum_phase_members, [8, 16, 24]);
    assert_eq!(configured.phase_min_offered, [8; 3]);
    assert_eq!(configured.min_members, [numerical.min_phase_samples; 3]);
    settings.discovery_offered_waves = NonZeroUsize::new(usize::MAX).unwrap();
    assert!(schedule(&settings, &numerical).is_err());
}
