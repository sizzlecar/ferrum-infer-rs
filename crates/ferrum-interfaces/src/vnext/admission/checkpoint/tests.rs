use super::*;

fn domain(value: u32) -> CapacityDomainId {
    CapacityDomainId::new(value).unwrap()
}

fn coordinator(maximum_active_sequences: u32) -> LogicalAdmissionCoordinator {
    coordinator_with_capacity(maximum_active_sequences, Some(24))
}

fn coordinator_with_capacity(
    maximum_active_sequences: u32,
    maximum_retained_bytes: Option<u64>,
) -> LogicalAdmissionCoordinator {
    LogicalAdmissionCoordinator::with_checkpoint_capacity(
        vec![
            (
                domain(1),
                CapacityDomainSpec::new(CapacityUnits::new(10), CapacityUnits::new(20)).unwrap(),
            ),
            (
                domain(2),
                CapacityDomainSpec::new(CapacityUnits::new(4), CapacityUnits::new(4)).unwrap(),
            ),
        ],
        maximum_active_sequences,
        maximum_retained_bytes.map(|bytes| CheckpointCapacityPolicy::new(bytes).unwrap()),
    )
    .unwrap()
}

fn demand(entries: &[(u32, u64)]) -> AdmissionDemand {
    let claim = if entries.is_empty() {
        CapacityVector::empty()
    } else {
        CapacityVector::new(
            entries
                .iter()
                .map(|&(id, units)| {
                    CapacityEntry::new(domain(id), CapacityUnits::new(units)).unwrap()
                })
                .collect(),
        )
        .unwrap()
    };
    AdmissionDemand::from_plan(
        claim.clone(),
        claim,
        AdmissionFitPolicy::ImmediateOnly,
        AdmissionPressureAction::WaitForRelease,
    )
    .unwrap()
}

fn claim(
    coordinator: &LogicalAdmissionCoordinator,
    entries: &[(u32, u64)],
) -> LogicalCheckpointLease {
    match coordinator
        .try_claim_checkpoint(
            &demand(entries),
            entries.iter().map(|(_, bytes)| bytes).sum(),
        )
        .unwrap()
    {
        CheckpointCapacityClaimDecision::Claimed(lease) => lease,
        other => panic!("expected checkpoint capacity, got {other:?}"),
    }
}

fn request(coordinator: &LogicalAdmissionCoordinator) -> LogicalRequestLease {
    match coordinator.try_admit_request(&demand(&[])).unwrap() {
        RequestAdmissionDecision::Admitted(lease) => lease,
        _ => panic!("expected request admission"),
    }
}

fn sequence(
    coordinator: &LogicalAdmissionCoordinator,
    request: &LogicalRequestLease,
) -> LogicalAdmissionLease {
    match coordinator
        .try_admit_sequence_for_request(request, &demand(&[]))
        .unwrap()
    {
        AdmissionDecision::Admitted(lease) => lease,
        _ => panic!("expected sequence admission"),
    }
}

#[test]
fn checkpoint_owns_capacity_without_an_execution_parent_or_slot() {
    let coordinator = coordinator(1);
    let source_request = request(&coordinator);
    let source_sequence = sequence(&coordinator, &source_request);
    let before = coordinator.snapshot().unwrap();
    let checkpoint = claim(&coordinator, &[(1, 6), (2, 2)]);
    let retained = coordinator.snapshot().unwrap();
    assert_eq!(retained.active_requests(), before.active_requests());
    assert_eq!(retained.active_sequences(), before.active_sequences());
    assert_eq!(retained.active_child_claims(), 0);
    assert_eq!(retained.active_checkpoint_claims(), 1);
    assert_eq!(retained.domains()[0].used().get(), 6);
    assert_eq!(retained.domains()[1].used().get(), 2);
    assert_eq!(checkpoint.retained_bytes(), 8);
    assert_eq!(coordinator.checkpoint_retained_bytes().unwrap(), 8);

    drop(source_sequence);
    drop(source_request);
    let independent = coordinator.snapshot().unwrap();
    assert_eq!(independent.active_requests(), 0);
    assert_eq!(independent.active_sequences(), 0);
    assert_eq!(independent.domains(), retained.domains());
    let next_request = request(&coordinator);
    let next_sequence = sequence(&coordinator, &next_request);
    drop(next_sequence);
    drop(next_request);
    drop(checkpoint);
    assert_eq!(coordinator.checkpoint_retained_bytes().unwrap(), 0);
    assert!(coordinator
        .snapshot()
        .unwrap()
        .domains()
        .iter()
        .all(|entry| entry.used() == CapacityUnits::ZERO));
}

#[test]
fn checkpoint_pressure_uses_existing_domains_and_failed_claims_are_atomic() {
    let coordinator = coordinator(1);
    let retained = claim(&coordinator, &[(1, 6)]);
    let before = coordinator.snapshot().unwrap();
    let blocked = coordinator
        .try_claim_checkpoint(&demand(&[(1, 5), (2, 1)]), 6)
        .unwrap();
    let CheckpointCapacityClaimDecision::Deferred(deferred) = blocked else {
        panic!("live domain use must defer the checkpoint");
    };
    assert_eq!(deferred.action(), DeferredAction::WaitForRelease);
    assert_eq!(deferred.blockers().len(), 1);
    assert_eq!(coordinator.snapshot().unwrap(), before);
    assert!(matches!(
        coordinator
            .try_claim_checkpoint(&demand(&[(1, 21)]), 21)
            .unwrap(),
        CheckpointCapacityClaimDecision::PermanentRejected(_)
    ));
    assert!(matches!(
        coordinator.try_claim_checkpoint(&demand(&[(1, 1), (3, 1)]), 2),
        Err(VNextError::DynamicAdmissionContract {
            kind: DynamicAdmissionFaultKind::UnknownDomain,
            ..
        })
    ));
    assert!(coordinator.try_claim_checkpoint(&demand(&[]), 0).is_err());
    assert_eq!(coordinator.snapshot().unwrap(), before);
    drop(retained);
    let growth = coordinator
        .try_claim_checkpoint(&demand(&[(1, 11)]), 11)
        .unwrap();
    let CheckpointCapacityClaimDecision::Deferred(growth) = growth else {
        panic!("unprovisioned domain capacity must defer");
    };
    assert_eq!(growth.action(), DeferredAction::AwaitBackingGrowth);
    assert_eq!(
        coordinator.snapshot().unwrap().active_checkpoint_claims(),
        0
    );
}

#[tokio::test]
async fn checkpoint_last_owner_drop_wakes_only_affected_capacity_sources() {
    let coordinator = coordinator(1);
    let checkpoint = Arc::new(claim(&coordinator, &[(1, 8)]));
    let pin = Arc::clone(&checkpoint);
    let observed = coordinator.wait_snapshot_for_domains([domain(1)]).unwrap();
    let waiter = coordinator
        .register_waiter(observed.wait_condition().clone())
        .unwrap();
    let unaffected = coordinator.wait_snapshot_for_domains([domain(2)]).unwrap();
    let unaffected_waiter = coordinator
        .register_waiter(unaffected.wait_condition().clone())
        .unwrap();
    let mut availability = Vec::new();
    coordinator
        .write_availability_epochs(&mut availability)
        .unwrap();
    let slots = availability
        .iter()
        .find(|entry| entry.source() == CapacityAvailabilitySource::ActiveSequenceSlots)
        .copied()
        .unwrap();
    let before = coordinator.epochs().unwrap();
    drop(checkpoint);
    assert!(!waiter.recheck().unwrap().should_retry());
    assert_eq!(coordinator.checkpoint_retained_bytes().unwrap(), 8);
    assert_eq!(
        coordinator.snapshot().unwrap().active_checkpoint_claims(),
        1
    );
    drop(pin);
    assert_eq!(coordinator.checkpoint_retained_bytes().unwrap(), 0);
    assert!(waiter.recheck().unwrap().should_retry());
    assert_eq!(
        waiter.wait_for_change().await.unwrap(),
        coordinator.epochs().unwrap()
    );
    assert!(!unaffected_waiter.recheck().unwrap().should_retry());
    let after = coordinator
        .write_availability_epochs(&mut availability)
        .unwrap();
    assert_eq!(after.release_epoch(), before.release_epoch() + 1);
    assert_eq!(after.capacity_epoch(), before.capacity_epoch());
    assert_eq!(
        *availability
            .iter()
            .find(|entry| entry.source() == CapacityAvailabilitySource::ActiveSequenceSlots)
            .unwrap(),
        slots
    );
}

#[test]
fn checkpoint_release_is_once_and_authorities_are_owner_scoped_and_never_reused() {
    let owner = coordinator(1);
    let foreign = coordinator(1);
    let mut first = claim(&owner, &[(1, 2)]);
    let other = claim(&foreign, &[(1, 2)]);
    assert_eq!(first.authority().serial(), other.authority().serial());
    assert_ne!(first.authority(), other.authority());
    assert!(owner.clone().owns_checkpoint_claim(&first));
    assert!(!foreign.owns_checkpoint_claim(&first));
    assert!(!owner.owns_checkpoint_claim(&other));
    let old_id = first.authority();
    let before = owner.epochs().unwrap();
    assert!(first.release_inner());
    assert!(!owner.owns_checkpoint_claim(&first));
    let released = owner.snapshot().unwrap();
    assert_eq!(released.release_epoch(), before.release_epoch() + 1);
    assert!(first.release_inner());
    drop(first);
    assert_eq!(owner.snapshot().unwrap(), released);
    let second = claim(&owner, &[(1, 2)]);
    assert_ne!(second.authority(), old_id);
    drop(other);
    assert_eq!(owner.snapshot().unwrap().active_checkpoint_claims(), 1);
}

#[test]
fn checkpoint_close_is_one_way_and_keeps_existing_lease_releasable() {
    let coordinator = coordinator(1);
    let existing = claim(&coordinator, &[(1, 3)]);
    coordinator.close_checkpoint_admission().unwrap();
    coordinator.close_checkpoint_admission().unwrap();
    let before = coordinator.snapshot().unwrap();
    assert!(coordinator
        .try_claim_checkpoint(&demand(&[(1, 1)]), 1)
        .is_err());
    assert_eq!(coordinator.snapshot().unwrap(), before);
    // Closing only checkpoint admission must not change ordinary admission.
    let request = request(&coordinator);
    let sequence = sequence(&coordinator, &request);
    drop(existing);
    assert_eq!(
        coordinator.snapshot().unwrap().active_checkpoint_claims(),
        0
    );
    drop(sequence);
    drop(request);
    assert!(!coordinator.snapshot().unwrap().poisoned());
}

#[test]
fn checkpoint_close_and_claim_are_serialized_without_stranding_a_lease() {
    let coordinator = coordinator(1);
    let start = std::sync::Barrier::new(2);
    let outcome = std::thread::scope(|scope| {
        let attempt = scope.spawn(|| {
            start.wait();
            coordinator.try_claim_checkpoint(&demand(&[(1, 2)]), 2)
        });
        start.wait();
        coordinator.close_checkpoint_admission().unwrap();
        attempt.join().unwrap()
    });
    match outcome {
        Ok(CheckpointCapacityClaimDecision::Claimed(lease)) => {
            // A claim that won the lock precedes close and remains releasable.
            assert_eq!(
                coordinator.snapshot().unwrap().active_checkpoint_claims(),
                1
            );
            drop(lease);
        }
        Err(VNextError::DynamicAdmissionContract {
            kind: DynamicAdmissionFaultKind::InvalidContract,
            ..
        }) => {}
        other => panic!("unexpected concurrent checkpoint admission: {other:?}"),
    }
    assert!(coordinator
        .try_claim_checkpoint(&demand(&[(1, 1)]), 1)
        .is_err());
    let after = coordinator.snapshot().unwrap();
    assert_eq!(after.active_checkpoint_claims(), 0);
    assert_eq!(after.domains()[0].used(), CapacityUnits::ZERO);
    assert!(!after.poisoned());
}

#[test]
fn checkpoint_concurrent_claims_cannot_overcommit() {
    let coordinator = coordinator(2);
    let outcomes = std::thread::scope(|scope| {
        let first = scope.spawn(|| {
            coordinator
                .try_claim_checkpoint(&demand(&[(1, 6)]), 6)
                .unwrap()
        });
        let second = scope.spawn(|| {
            coordinator
                .try_claim_checkpoint(&demand(&[(1, 6)]), 6)
                .unwrap()
        });
        [first.join().unwrap(), second.join().unwrap()]
    });
    assert_eq!(
        outcomes
            .iter()
            .filter(|outcome| matches!(outcome, CheckpointCapacityClaimDecision::Claimed(_)))
            .count(),
        1
    );
    assert_eq!(
        outcomes
            .iter()
            .filter(|outcome| matches!(outcome, CheckpointCapacityClaimDecision::Deferred(_)))
            .count(),
        1
    );
    assert_eq!(coordinator.snapshot().unwrap().domains()[0].used().get(), 6);
    drop(outcomes);
    assert_eq!(
        coordinator.snapshot().unwrap().domains()[0].used(),
        CapacityUnits::ZERO
    );
}

#[test]
fn checkpoint_outstanding_release_is_reserved_by_all_admission_paths() {
    let coordinator = coordinator(4);
    let checkpoint = claim(&coordinator, &[(1, 1)]);
    let request = request(&coordinator);
    let sequence = sequence(&coordinator, &request);
    let child = match coordinator
        .try_claim_for_sequence(&sequence, &demand(&[(1, 1)]))
        .unwrap()
    {
        CapacityClaimDecision::Claimed(lease) => lease,
        _ => panic!("expected child claim"),
    };
    coordinator.inner.state.lock().unwrap().release_epoch = u64::MAX - 4;
    let before = coordinator.snapshot().unwrap();
    let exhausted = |error| {
        assert!(matches!(
            error,
            VNextError::DynamicAdmissionContract {
                kind: DynamicAdmissionFaultKind::EpochExhausted,
                ..
            }
        ))
    };
    exhausted(coordinator.try_admit_request(&demand(&[])).err().unwrap());
    exhausted(
        coordinator
            .try_admit_initial_sequence(&demand(&[]), &demand(&[]))
            .err()
            .unwrap(),
    );
    exhausted(
        coordinator
            .try_admit_sequence_for_request(&request, &demand(&[]))
            .err()
            .unwrap(),
    );
    exhausted(
        coordinator
            .try_claim_for_sequence(&sequence, &demand(&[(1, 1)]))
            .err()
            .unwrap(),
    );
    exhausted(
        coordinator
            .try_claim_checkpoint(&demand(&[(1, 1)]), 1)
            .err()
            .unwrap(),
    );
    assert_eq!(coordinator.snapshot().unwrap(), before);
    drop(child);
    drop(sequence);
    drop(request);
    drop(checkpoint);
    let after = coordinator.snapshot().unwrap();
    assert_eq!(after.release_epoch(), u64::MAX);
    assert_eq!(after.active_checkpoint_claims(), 0);
    assert!(!after.poisoned());
}

#[test]
fn checkpoint_authority_exhaustion_and_unpublishable_release_have_no_side_effects() {
    let coordinator = coordinator(1);
    coordinator
        .inner
        .state
        .lock()
        .unwrap()
        .checkpoint_claims
        .next_serial = u64::MAX;
    let before = coordinator.snapshot().unwrap();
    assert!(matches!(
        coordinator.try_claim_checkpoint(&demand(&[(1, 1)]), 1),
        Err(VNextError::DynamicAdmissionContract {
            kind: DynamicAdmissionFaultKind::AuthorityExhausted,
            ..
        })
    ));
    assert_eq!(coordinator.snapshot().unwrap(), before);
    {
        let mut state = coordinator.inner.state.lock().unwrap();
        state.checkpoint_claims.next_serial = 1;
        state
            .domains
            .get_mut(&domain(1))
            .unwrap()
            .availability_epoch = u64::MAX;
    }
    assert!(matches!(
        coordinator.try_claim_checkpoint(&demand(&[(1, 1)]), 1),
        Err(VNextError::DynamicAdmissionContract {
            kind: DynamicAdmissionFaultKind::EpochExhausted,
            ..
        })
    ));
    assert_eq!(coordinator.snapshot().unwrap(), before);
}

#[test]
fn checkpoint_corrupt_identity_fails_closed_without_releasing_other_capacity() {
    let coordinator = coordinator(1);
    let mut checkpoint = claim(&coordinator, &[(1, 3)]);
    checkpoint.authority.serial += 1;
    let before = coordinator.snapshot().unwrap();
    assert!(!checkpoint.release_inner());
    let failed = coordinator.snapshot().unwrap();
    assert!(failed.poisoned());
    assert_eq!(failed.domains(), before.domains());
    assert_eq!(failed.active_checkpoint_claims(), 1);
    assert_eq!(failed.release_epoch(), before.release_epoch());
    assert!(coordinator
        .try_claim_checkpoint(&demand(&[(1, 1)]), 1)
        .is_err());
}

#[test]
fn checkpoint_unwind_releases_without_poisoning_coordinator() {
    let coordinator = coordinator(1);
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _checkpoint = claim(&coordinator, &[(1, 3)]);
        panic!("checkpoint owner construction aborted");
    }));
    assert!(result.is_err());
    let after = coordinator.snapshot().unwrap();
    assert_eq!(after.active_checkpoint_claims(), 0);
    assert_eq!(after.domains()[0].used(), CapacityUnits::ZERO);
    assert_eq!(coordinator.checkpoint_retained_bytes().unwrap(), 0);
    assert!(!after.poisoned());
}

#[test]
fn checkpoint_retention_is_disabled_by_default_without_affecting_normal_slots() {
    let coordinator = LogicalAdmissionCoordinator::new(
        vec![(
            domain(1),
            CapacityDomainSpec::new(CapacityUnits::new(8), CapacityUnits::new(8)).unwrap(),
        )],
        3,
    )
    .unwrap();
    let before = coordinator.snapshot().unwrap();
    assert!(matches!(
        coordinator
            .try_claim_checkpoint(&demand(&[(1, 4)]), 4)
            .unwrap(),
        CheckpointCapacityClaimDecision::Skipped(CheckpointRetentionSkipReason::Disabled)
    ));
    assert_eq!(coordinator.snapshot().unwrap(), before);
    assert_eq!(coordinator.checkpoint_retained_bytes().unwrap(), 0);
    let request = request(&coordinator);
    let sequences = (0..3)
        .map(|_| sequence(&coordinator, &request))
        .collect::<Vec<_>>();
    assert_eq!(coordinator.snapshot().unwrap().active_sequences(), 3);
    drop(sequences);
    drop(request);
}

#[test]
fn checkpoint_fee_must_equal_every_actual_domain_claim() {
    let coordinator = coordinator(1);
    let before = coordinator.snapshot().unwrap();
    for fee in [0, 4, 6] {
        assert!(coordinator
            .try_claim_checkpoint(&demand(&[(1, 3), (2, 2)]), fee)
            .is_err());
        assert_eq!(coordinator.snapshot().unwrap(), before);
        assert_eq!(coordinator.checkpoint_retained_bytes().unwrap(), 0);
    }
    let lease = claim(&coordinator, &[(1, 3), (2, 2)]);
    assert_eq!(lease.retained_bytes(), 5);
    assert_eq!(coordinator.checkpoint_retained_bytes().unwrap(), 5);
    drop(lease);
    assert_eq!(coordinator.checkpoint_retained_bytes().unwrap(), 0);
}

#[test]
fn checkpoint_aggregate_fee_serializes_competing_distinct_domains_and_recovers() {
    let coordinator = coordinator_with_capacity(3, Some(4));
    let barrier = std::sync::Barrier::new(2);
    let outcomes = std::thread::scope(|scope| {
        let first = scope.spawn(|| {
            barrier.wait();
            coordinator
                .try_claim_checkpoint(&demand(&[(1, 3)]), 3)
                .unwrap()
        });
        let second = scope.spawn(|| {
            barrier.wait();
            coordinator
                .try_claim_checkpoint(&demand(&[(2, 3)]), 3)
                .unwrap()
        });
        [first.join().unwrap(), second.join().unwrap()]
    });
    let mut retained = None;
    let mut skipped = None;
    for outcome in outcomes {
        match outcome {
            CheckpointCapacityClaimDecision::Claimed(lease) => {
                assert!(retained.replace(lease).is_none());
            }
            CheckpointCapacityClaimDecision::Skipped(reason) => {
                assert!(skipped.replace(reason).is_none());
            }
            other => {
                panic!("individually available domains must compete only for retention: {other:?}")
            }
        }
    }
    assert!(retained.is_some());
    assert_eq!(
        skipped,
        Some(CheckpointRetentionSkipReason::Capacity {
            requested_bytes: 3,
            retained_bytes: 3,
            maximum_bytes: 4,
        })
    );
    assert_eq!(coordinator.checkpoint_retained_bytes().unwrap(), 3);
    let snapshot = coordinator.snapshot().unwrap();
    assert_eq!(
        snapshot
            .domains()
            .iter()
            .map(|entry| entry.used().get())
            .sum::<u64>(),
        3
    );
    assert_eq!(snapshot.active_requests(), 0);
    assert_eq!(snapshot.active_sequences(), 0);
    drop(retained);
    let full = claim(&coordinator, &[(1, 2), (2, 2)]);
    assert_eq!(coordinator.checkpoint_retained_bytes().unwrap(), 4);
    drop(full);
    assert_eq!(coordinator.checkpoint_retained_bytes().unwrap(), 0);
}

#[test]
fn checkpoint_fee_sum_overflow_and_full_u64_cap_do_not_wrap() {
    let coordinator = LogicalAdmissionCoordinator::with_checkpoint_capacity(
        [1, 2]
            .map(|id| {
                (
                    domain(id),
                    CapacityDomainSpec::new(
                        CapacityUnits::new(u64::MAX),
                        CapacityUnits::new(u64::MAX),
                    )
                    .unwrap(),
                )
            })
            .to_vec(),
        1,
        Some(CheckpointCapacityPolicy::new(u64::MAX).unwrap()),
    )
    .unwrap();
    let before = coordinator.snapshot().unwrap();
    assert!(matches!(
        coordinator.try_claim_checkpoint(&demand(&[(1, u64::MAX), (2, 1)]), u64::MAX),
        Err(VNextError::DynamicAdmissionContract {
            kind: DynamicAdmissionFaultKind::ArithmeticOverflow,
            ..
        })
    ));
    assert_eq!(coordinator.snapshot().unwrap(), before);
    let retained = claim(&coordinator, &[(1, u64::MAX)]);
    assert!(matches!(
        coordinator
            .try_claim_checkpoint(&demand(&[(2, 1)]), 1)
            .unwrap(),
        CheckpointCapacityClaimDecision::Skipped(CheckpointRetentionSkipReason::Capacity {
            requested_bytes: 1,
            retained_bytes: u64::MAX,
            maximum_bytes: u64::MAX,
        })
    ));
    assert_eq!(coordinator.checkpoint_retained_bytes().unwrap(), u64::MAX);
    drop(retained);
    assert_eq!(coordinator.checkpoint_retained_bytes().unwrap(), 0);
}

#[test]
fn checkpoint_corrupt_fee_fails_closed_without_crediting_domains_or_retention() {
    let coordinator = coordinator(1);
    let mut lease = claim(&coordinator, &[(1, 3)]);
    lease.retained_bytes = 2;
    let before = coordinator.snapshot().unwrap();
    assert!(!lease.release_inner());
    let after = coordinator.snapshot().unwrap();
    assert!(after.poisoned());
    assert_eq!(after.domains(), before.domains());
    assert_eq!(after.active_checkpoint_claims(), 1);
    assert_eq!(coordinator.checkpoint_retained_bytes().unwrap(), 3);
}
