use super::*;

fn current(phase: PrefixContinuationPhase) -> (SchedulerSnapshot, PrefixRendezvousOffer) {
    let (mut snapshot, offer) = setup();
    snapshot.requests[1].readiness = RequestReadiness::StateBlocked;
    if !matches!(phase, PrefixContinuationPhase::HeldAwaitingProducer) {
        let source = &mut snapshot.requests[0];
        let RequestPhaseView::Prefill(progress) = &mut source.phase else {
            unreachable!()
        };
        progress.offset = 12;
        progress.logical_high_water = 12;
        source.context_tokens = 12;
    }
    if phase == PrefixContinuationPhase::Restored {
        let target = &mut snapshot.requests[1];
        let RequestPhaseView::Prefill(progress) = &mut target.phase else {
            unreachable!()
        };
        progress.offset = 12;
        progress.logical_high_water = 12;
        target.context_tokens = 12;
        target.readiness = RequestReadiness::Ready;
    }
    (snapshot, offer)
}

fn run(
    snapshot: &SchedulerSnapshot,
    offer: &PrefixRendezvousOffer,
    phase: PrefixContinuationPhase,
    fixture: &Fixture,
    maintenance: &Maintenance,
) -> PrefixContinuationDecision {
    planner(12).continue_prefix_rendezvous(
        snapshot,
        offer,
        phase,
        &Model(infer),
        maintenance,
        fixture,
        &mut Clock(100),
    )
}
fn ready(decision: PrefixContinuationDecision) -> PrefixContinuationEvidence {
    match decision {
        PrefixContinuationDecision::Ready { continuation, .. } => continuation,
        other => panic!("expected independently replayed remaining queue: {other:?}"),
    }
}

#[test]
fn prefix_continuation_all_phases_rebind_fresh_lineages_and_preserve_first_action_protocol() {
    for phase in [
        PrefixContinuationPhase::HeldAwaitingProducer,
        PrefixContinuationPhase::AtCaptureBoundary {
            capture_span_start: 8,
        },
        PrefixContinuationPhase::CheckpointReady {
            capture_span_start: 8,
        },
        PrefixContinuationPhase::Restored,
    ] {
        let (snapshot, offer) = current(phase);
        let original = snapshot.clone();
        let fixture = Fixture::default();
        let result = ready(run(
            &snapshot,
            &offer,
            phase,
            &fixture,
            &Maintenance::default(),
        ));
        assert_eq!(result.offer(), &offer);
        assert_eq!(result.phase(), phase);
        assert!(result.valid_until_ns() < offer.expires_at_ns);
        assert!(result.first_action_cost_ns() > 0);
        assert!(
            result
                .remaining()
                .steps()
                .iter()
                .any(|step| matches!(step, PrefixPathStep::Wave(wave)
                if wave.work.iter().any(|row| row.key == snapshot.requests[2].key))),
            "old decoder remains in the complete remaining witness"
        );
        let bindings = fixture.continuation_roots.borrow();
        assert_eq!(bindings.len(), 2, "search and replay each bind once");
        assert_ne!(bindings[0].0, bindings[1].0);
        assert!(bindings.iter().all(|(_, bound)| *bound == phase));
        match (phase, result.action()) {
            (
                PrefixContinuationPhase::HeldAwaitingProducer | PrefixContinuationPhase::Restored,
                PrefixContinuationAction::Wave(wave),
            ) => {
                assert!(wave.replayed_first_wave(&snapshot).is_some());
                assert!(Arc::ptr_eq(
                    wave.protection.as_ref().unwrap(),
                    result.protection()
                ));
                assert_eq!(wave.predicted_wall_ns, result.first_action_cost_ns());
                assert_eq!(wave.snapshot_generation, snapshot.generation);
                assert_eq!(
                    wave.witness_valid_for_ns,
                    result.valid_until_ns() - result.validated_at_ns()
                );
            }
            (
                PrefixContinuationPhase::AtCaptureBoundary { .. },
                PrefixContinuationAction::Maintenance(evidence),
            ) => assert_eq!(evidence.stage, PrefixMaintenanceStage::Capture),
            (
                PrefixContinuationPhase::CheckpointReady { .. },
                PrefixContinuationAction::Maintenance(evidence),
            ) => assert_eq!(evidence.stage, PrefixMaintenanceStage::Restore),
            other => panic!("incorrect first action: {other:?}"),
        }
        assert!(fixture
            .seen_owner_counts
            .borrow()
            .iter()
            .all(|n| *n == snapshot.requests.len()));
        assert_eq!(snapshot, original, "no real or logical snapshot mutation");
    }
}

#[test]
fn prefix_continuation_checkpoint_ready_does_not_require_producer_still_at_boundary() {
    let phase = PrefixContinuationPhase::CheckpointReady {
        capture_span_start: 8,
    };
    let (mut snapshot, offer) = current(phase);
    let mut source = decode(1);
    source.key = offer.producer.clone();
    source.context_tokens = 16;
    snapshot.requests[0] = source;
    let fixture = Fixture::default();
    let result = ready(run(
        &snapshot,
        &offer,
        phase,
        &fixture,
        &Maintenance::default(),
    ));
    assert!(
        matches!(result.action(), PrefixContinuationAction::Maintenance(e)
        if e.stage == PrefixMaintenanceStage::Restore)
    );
    assert!(
        fixture.captured_roots.borrow().is_empty(),
        "retained checkpoint is not captured again"
    );
    assert_eq!(fixture.restored_roots.borrow().len(), 2);
}

#[test]
fn prefix_continuation_requires_current_phase_evidence_again_during_replay() {
    let phase = PrefixContinuationPhase::CheckpointReady {
        capture_span_start: 8,
    };
    let (snapshot, offer) = current(phase);
    for fixture in [
        Fixture {
            refuse_continuation: true,
            ..Default::default()
        },
        Fixture {
            revoke_continuation_replay: true,
            ..Default::default()
        },
    ] {
        assert!(matches!(
            run(&snapshot, &offer, phase, &fixture, &Maintenance::default()),
            PrefixContinuationDecision::Unknown {
                reason: PlanningUnknownReason::UnknownResourceEvidence,
                ..
            }
        ));
    }
    let fixture = Fixture {
        revoke_continuation_replay: true,
        ..Default::default()
    };
    let decision = run(&snapshot, &offer, phase, &fixture, &Maintenance::default());
    assert_eq!(fixture.continuation_roots.borrow().len(), 1);
    assert!(
        matches!(decision, PrefixContinuationDecision::Unknown { search, .. }
        if search.phase == PlanningSearchPhase::Finalization)
    );
}

#[test]
fn prefix_continuation_binding_cannot_swallow_original_planning_budget() {
    let phase = PrefixContinuationPhase::Restored;
    let (snapshot, offer) = current(phase);
    let fixture = Fixture {
        swallow_binding_budget: true,
        now: Cell::new(100),
        ..Default::default()
    };
    struct Current<'a>(&'a Cell<u64>);
    impl PlanningClock for Current<'_> {
        fn now_ns(&mut self) -> u64 {
            self.0.get()
        }
    }
    let decision = planner(12).continue_prefix_rendezvous(
        &snapshot,
        &offer,
        phase,
        &Model(infer),
        &Maintenance::default(),
        &fixture,
        &mut Current(&fixture.now),
    );
    assert!(matches!(
        decision,
        PrefixContinuationDecision::Unknown {
            reason: PlanningUnknownReason::ComputeBudgetExhausted,
            ..
        }
    ));
}

#[test]
fn prefix_continuation_preserves_original_expiry_and_rejects_stale_keys_or_phase() {
    let phase = PrefixContinuationPhase::HeldAwaitingProducer;
    let (snapshot, mut offer) = current(phase);
    let fixture = Fixture::default();
    let result = ready(run(
        &snapshot,
        &offer,
        phase,
        &fixture,
        &Maintenance::default(),
    ));
    assert_eq!(result.offer().expires_at_ns, offer.expires_at_ns);
    let later = planner(12).continue_prefix_rendezvous(
        &snapshot,
        result.offer(),
        phase,
        &Model(infer),
        &Maintenance::default(),
        &fixture,
        &mut Clock(offer.expires_at_ns),
    );
    assert!(matches!(later, PrefixContinuationDecision::Unknown { .. }));
    offer.target.incarnation += 1;
    assert!(matches!(
        run(&snapshot, &offer, phase, &fixture, &Maintenance::default()),
        PrefixContinuationDecision::Unknown {
            reason: PlanningUnknownReason::InvalidSnapshot,
            ..
        }
    ));
    let (mut snapshot, offer) = current(phase);
    snapshot.requests[1].readiness = RequestReadiness::Ready;
    assert!(matches!(
        run(&snapshot, &offer, phase, &fixture, &Maintenance::default()),
        PrefixContinuationDecision::Unknown {
            reason: PlanningUnknownReason::InvalidSnapshot,
            ..
        }
    ));
    let (snapshot, offer) = current(PrefixContinuationPhase::Restored);
    assert!(matches!(
        run(&snapshot, &offer, phase, &fixture, &Maintenance::default()),
        PrefixContinuationDecision::Unknown {
            reason: PlanningUnknownReason::InvalidSnapshot,
            ..
        }
    ));
}

#[test]
fn prefix_continuation_cannot_drop_existing_itl_obligation_for_long_maintenance() {
    let phase = PrefixContinuationPhase::AtCaptureBoundary {
        capture_span_start: 8,
    };
    let (mut snapshot, offer) = current(phase);
    snapshot.scope.horizon_end_ns = 120;
    let old = &mut snapshot.requests[2];
    old.timing.maximum_output_tokens = n32(100);
    old.timing.first_commit_at_ns = Some(99);
    old.timing.last_commit_at_ns = Some(99);
    old.timing.budgets.itl_ns = n64(12);
    old.timing.budgets.tpot_ns = n64(12);
    let decision = run(
        &snapshot,
        &offer,
        phase,
        &Fixture::default(),
        &Maintenance {
            duration: 20,
            ..Default::default()
        },
    );
    assert!(matches!(
        decision,
        PrefixContinuationDecision::Unknown { .. }
    ));
}

#[test]
fn prefix_continuation_missing_restore_cost_or_expired_calibration_is_unknown() {
    let phase = PrefixContinuationPhase::CheckpointReady {
        capture_span_start: 8,
    };
    let (snapshot, offer) = current(phase);
    for maintenance in [
        Maintenance {
            missing: Some(PrefixMaintenanceStage::Restore),
            ..Default::default()
        },
        Maintenance {
            ttl: 0,
            ..Default::default()
        },
    ] {
        let result = run(&snapshot, &offer, phase, &Fixture::default(), &maintenance);
        assert!(
            matches!(result, PrefixContinuationDecision::Unknown { search, .. }
            if search.cost_unknown_candidates > 0)
        );
    }
}

#[test]
fn prefix_continuation_default_execution_provider_does_not_claim_current_checkpoint_proof() {
    let phase = PrefixContinuationPhase::CheckpointReady {
        capture_span_start: 8,
    };
    let (snapshot, offer) = current(phase);
    let context = execution::ReplayContext {
        resolver: &TestResolver,
        resources: None,
    };
    let decision = planner(12).continue_prefix_rendezvous(
        &snapshot,
        &offer,
        phase,
        &Model(infer),
        &Maintenance::default(),
        &context,
        &mut Clock(100),
    );
    assert!(matches!(
        decision,
        PrefixContinuationDecision::Unknown {
            reason: PlanningUnknownReason::UnknownResourceEvidence,
            ..
        }
    ));
}

#[test]
fn prefix_continuation_final_clock_cannot_publish_expired_replay_evidence() {
    let phase = PrefixContinuationPhase::AtCaptureBoundary {
        capture_span_start: 8,
    };
    let (snapshot, offer) = current(phase);
    let fixture = Fixture {
        replay_restore_clock: Some(150),
        now: Cell::new(100),
        ..Default::default()
    };
    struct Current<'a>(&'a Cell<u64>);
    impl PlanningClock for Current<'_> {
        fn now_ns(&mut self) -> u64 {
            self.0.get()
        }
    }
    let decision = planner(12).continue_prefix_rendezvous(
        &snapshot,
        &offer,
        phase,
        &Model(infer),
        &Maintenance {
            ttl: 10,
            ..Default::default()
        },
        &fixture,
        &mut Current(&fixture.now),
    );
    assert_eq!(fixture.continuation_roots.borrow().len(), 2);
    assert!(matches!(decision, PrefixContinuationDecision::Unknown {
        reason: PlanningUnknownReason::SearchIncomplete, search,
    } if search.phase == PlanningSearchPhase::Finalization));
}
