use super::*;
use ferrum_runtime::{
    CapacityAvailabilityEpoch, CapacityAvailabilitySource, CapacityWaitCondition,
};
use ferrum_types::AllocatorCapacityDomain;

fn candidate(
    request_id: &RequestId,
    blocked_on: Option<PressureWaitCondition>,
    advances_wait_source: bool,
) -> PressureCandidate {
    PressureCandidate {
        request_id: request_id.clone(),
        work_kind: LogicalWorkKind::Decode,
        priority: Priority::Normal,
        progress: LogicalWorkGeneration(1),
        recompute_cost: 1,
        advances_wait_source,
        blocked_on,
    }
}

#[test]
fn allocator_pressure_domains_never_alias_native_or_other_pools() {
    let domain = AllocatorCapacityDomain::new();
    let wait = PressureWaitCondition::allocator(domain.clone(), 8);
    let later = PressureWaitCondition::allocator(domain.clone(), 9);
    let other_pool = PressureWaitCondition::allocator(AllocatorCapacityDomain::new(), 8);
    let native = CapacityWaitCondition::from_observation(
        8,
        vec![
            CapacityAvailabilityEpoch::new(CapacityAvailabilitySource::ActiveSequenceSlots, 8)
                .unwrap(),
        ],
    )
    .unwrap();
    let native_wait = PressureWaitCondition::from(native.clone());
    assert!(waits_overlap(&wait, &later));
    assert!(!waits_overlap(&wait, &other_pool));
    assert!(!waits_overlap(&wait, &native_wait));
    assert!(same_source_topology(&wait, &later));
    assert!(!same_source_topology(&wait, &other_pool));
    assert_eq!(wait.as_allocator(), Some((&domain, 8)));
    assert!(wait.as_native().is_none());
    assert_eq!(native_wait.as_native(), Some(&native));
    assert!(native_wait.as_allocator().is_none());
    // The existing native diagnostic shape and real epoch values are intact.
    assert_eq!(
        serde_json::to_value(&native_wait).unwrap(),
        serde_json::to_value(&native).unwrap()
    );

    let mut coordinator = PressureCoordinator::default();
    let original = coordinator
        .find_or_open_episode_for(&wait, &[], &[])
        .unwrap();
    assert_eq!(
        coordinator
            .find_or_open_episode_for(&later, &[], &[])
            .unwrap(),
        original
    );
    let other = coordinator
        .find_or_open_episode_for(&other_pool, &[], &[])
        .unwrap();
    let native_episode = coordinator
        .find_or_open_episode_for(&native_wait, &[], &[])
        .unwrap();
    assert_ne!(original, other);
    assert_ne!(original, native_episode);
    assert_ne!(other, native_episode);
    assert_eq!(coordinator.stats().active_episodes, 3);
    assert_eq!(coordinator.source_index.len(), 3);
}

#[test]
fn allocator_pressure_cannot_bridge_a_request_into_another_pool() {
    let mut coordinator = PressureCoordinator::default();
    let request_id = RequestId::new();
    let peer = RequestId::new();
    let original = PressureWaitCondition::allocator(AllocatorCapacityDomain::new(), 5);
    let foreign = PressureWaitCondition::allocator(AllocatorCapacityDomain::new(), 5);
    let first = coordinator
        .plan_failure(
            std::slice::from_ref(&request_id),
            &original,
            &[
                candidate(&request_id, None, true),
                candidate(&peer, None, true),
            ],
        )
        .unwrap();
    let PressureDecision::Deferred { episode_id, .. } = first else {
        panic!("a runnable peer keeps the original request deferred");
    };
    assert!(matches!(
        coordinator.plan_failure(std::slice::from_ref(&request_id), &foreign, &[]),
        Err(PressureCoordinatorError::OverlappingPendingEpisodes)
    ));
    assert_eq!(
        coordinator.request_index.get(&request_id),
        Some(&episode_id)
    );
    assert_eq!(coordinator.stats().active_episodes, 1);
    assert!(coordinator.indexed_episode_ids(&foreign).is_empty());
}

#[test]
fn allocator_pressure_waits_for_the_same_release_transaction_and_reprobes_without_claiming_fit() {
    let mut coordinator = PressureCoordinator::default();
    let owner = RequestId::new();
    let victim = RequestId::new();
    let another_releaser = RequestId::new();
    let domain = AllocatorCapacityDomain::new();
    let wait = PressureWaitCondition::allocator(domain.clone(), 8);
    let advanced = PressureWaitCondition::allocator(domain, 9);
    assert!(matches!(
        coordinator
            .plan_failure(
                std::slice::from_ref(&victim),
                &wait,
                &[
                    candidate(&owner, None, true),
                    candidate(&victim, None, true)
                ],
            )
            .unwrap(),
        PressureDecision::Deferred { .. }
    ));
    let transaction = match coordinator
        .plan_failure(
            std::slice::from_ref(&owner),
            &wait,
            &[
                candidate(&owner, None, false),
                candidate(&victim, Some(wait.clone()), true),
            ],
        )
        .unwrap()
    {
        PressureDecision::YieldPlanned(transaction) => transaction,
        other => panic!("both blocked participants must plan a release, got {other:?}"),
    };
    assert_eq!(transaction.progress_owner_id(), &owner);
    assert_eq!(transaction.victim_request_id(), &victim);
    assert_eq!(transaction.wait_condition(), &wait);
    let armed = coordinator.arm_release_fence(&transaction).unwrap();
    assert!(matches!(
        coordinator.hold_status(&victim),
        PressureHoldStatus::Held { .. }
    ));
    // A global epoch change is not this exact victim's terminal evidence.
    assert!(matches!(
        coordinator
            .plan_failure(std::slice::from_ref(&owner), &advanced, &[],)
            .unwrap(),
        PressureDecision::Deferred { .. }
    ));
    assert_eq!(coordinator.stats().pending_release_fences, 1);
    let journal_len = coordinator.journal().len();
    let foreign = PressureWaitCondition::allocator(AllocatorCapacityDomain::new(), 9);
    assert!(matches!(
        coordinator.complete_release_fence(&transaction, Some(&foreign)),
        Err(PressureCoordinatorError::OverlappingPendingEpisodes)
    ));
    assert_eq!(coordinator.journal().len(), journal_len);
    assert_eq!(coordinator.stats().pending_release_fences, 1);

    let (released, disposition) = coordinator
        .complete_release_fence(&transaction, None)
        .unwrap();
    let PressureReleaseFenceDisposition::Resumable(resumable) = disposition else {
        panic!("the exact terminal fence must permit the progress owner to resume");
    };
    assert!(transaction.planned_ordinal() < armed && armed < released && released < resumable);
    assert!(matches!(
        coordinator
            .plan_failure(
                std::slice::from_ref(&owner),
                &wait,
                &[
                    candidate(&owner, None, true),
                    candidate(&victim, None, false)
                ],
            )
            .unwrap(),
        PressureDecision::InvariantViolation(PressureInvariantViolation {
            class: PressureInvariantViolationClass::YieldDidNotAdvanceSource,
            ..
        })
    ));

    // An authoritative re-probe can still report shortfall at the new epoch.
    // It remains deferred behind another actual releaser; no capacity grant or
    // progress is manufactured by the changed notification alone.
    assert!(matches!(
        coordinator
            .plan_failure(
                std::slice::from_ref(&owner),
                &advanced,
                &[
                    candidate(&owner, None, true),
                    candidate(&victim, None, false),
                    candidate(&another_releaser, None, true)
                ],
            )
            .unwrap(),
        PressureDecision::Deferred { .. }
    ));
    let episode = &coordinator.episodes[&transaction.episode_id()];
    assert_eq!(episode.observed, observed_map(&advanced));
    assert_eq!(episode.progress_owner.as_ref(), Some(&owner));
    assert_eq!(
        episode.owner_progress_baseline,
        transaction.progress_baseline()
    );
    assert!(matches!(
        coordinator.hold_status(&victim),
        PressureHoldStatus::Held { .. }
    ));
}

#[test]
fn failed_release_observation_preserves_native_and_allocator_fences_until_real_completion() {
    let native = CapacityWaitCondition::from_observation(
        8,
        vec![
            CapacityAvailabilityEpoch::new(CapacityAvailabilitySource::ActiveSequenceSlots, 8)
                .unwrap(),
        ],
    )
    .unwrap();
    for wait in [
        PressureWaitCondition::from(native),
        PressureWaitCondition::allocator(AllocatorCapacityDomain::new(), 8),
    ] {
        let mut coordinator = PressureCoordinator::default();
        let owner = RequestId::new();
        let victim = RequestId::new();
        assert!(matches!(
            coordinator
                .plan_failure(
                    std::slice::from_ref(&victim),
                    &wait,
                    &[
                        candidate(&owner, None, true),
                        candidate(&victim, None, true)
                    ],
                )
                .unwrap(),
            PressureDecision::Deferred { .. }
        ));
        let PressureDecision::YieldPlanned(transaction) = coordinator
            .plan_failure(
                std::slice::from_ref(&owner),
                &wait,
                &[
                    candidate(&owner, None, false),
                    candidate(&victim, Some(wait.clone()), true),
                ],
            )
            .unwrap()
        else {
            panic!("both exact participants require a release fence");
        };
        coordinator.arm_release_fence(&transaction).unwrap();
        let affected = coordinator
            .release_transaction_requests(&transaction)
            .unwrap();
        assert_eq!(affected.len(), 2);
        assert!(affected.contains(&owner) && affected.contains(&victim));
        assert!(coordinator
            .mark_release_observation_failed(&transaction)
            .unwrap());
        assert!(!coordinator
            .mark_release_observation_failed(&transaction)
            .unwrap());
        assert_eq!(coordinator.stats().pending_release_fences, 1);
        assert!(matches!(
            coordinator.hold_status(&victim),
            PressureHoldStatus::Held { .. }
        ));
        assert!(matches!(
            coordinator
                .plan_failure(&[RequestId::new()], &wait, &[],)
                .unwrap(),
            PressureDecision::InvariantViolation(PressureInvariantViolation {
                class: PressureInvariantViolationClass::ReleaseObservationFailed,
                ..
            })
        ));
        // Logical failure does not consume a physical fence. Later original
        // terminal evidence can still complete exactly that transaction once.
        coordinator.record_terminal(&owner).unwrap();
        coordinator.record_terminal(&victim).unwrap();
        assert_eq!(coordinator.stats().pending_release_fences, 1);
        let replacement_wait = PressureWaitCondition::allocator(AllocatorCapacityDomain::new(), 0);
        let PressureDecision::Deferred {
            episode_id: replacement,
            ..
        } = coordinator
            .plan_failure(
                std::slice::from_ref(&owner),
                &replacement_wait,
                &[
                    candidate(&owner, None, false),
                    candidate(&RequestId::new(), None, true),
                ],
            )
            .unwrap()
        else {
            panic!("same-ID replacement must use its new allocator domain");
        };
        assert_ne!(replacement, transaction.episode_id());
        assert!(matches!(
            coordinator
                .complete_release_fence(&transaction, None)
                .unwrap()
                .1,
            PressureReleaseFenceDisposition::Closed { .. }
        ));
        assert_eq!(coordinator.stats().pending_release_fences, 0);
        assert_eq!(coordinator.request_index.get(&owner), Some(&replacement));
        assert!(coordinator
            .complete_release_fence(&transaction, None)
            .is_err());
    }
}
