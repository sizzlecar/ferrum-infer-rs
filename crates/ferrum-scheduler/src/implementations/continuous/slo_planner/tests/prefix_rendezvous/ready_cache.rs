use super::*;

pub(super) fn bind<'a>(
    state: &State<'a>,
    input: &PlanningReadyPrefixInput<'_>,
    poll: &mut dyn FnMut() -> Result<(), PlanningUnknownReason>,
) -> Result<Option<Arc<dyn PlanningExecutionState<'a> + 'a>>, PlanningUnknownReason> {
    if state.fixture.swallow_binding_budget {
        let old = state.fixture.now.replace(10_000_000);
        let _ = poll();
        state.fixture.now.set(old);
    } else {
        poll()?;
    }
    if state.fixture.refuse_continuation
        || (state.fixture.revoke_continuation_replay
            && !state.fixture.ready_bindings.borrow().is_empty())
    {
        return Err(PlanningUnknownReason::UnknownResourceEvidence);
    }
    assert_eq!(input.snapshot, state.snapshot);
    state
        .fixture
        .ready_bindings
        .borrow_mut()
        .push((state.root, input.phase));
    let mut next = state.clone();
    // The abstract fixture represents a private retained owner. Native owner,
    // retirement, allocation and ack validation live in the interfaces tests.
    next.checkpoint = Some((input.offer.identity, 0));
    Ok(Some(Arc::new(next)))
}

pub(super) fn restore<'a>(
    state: &State<'a>,
    input: &PlanningReadyPrefixRestoreInput<'_>,
    poll: &mut dyn FnMut() -> Result<(), PlanningUnknownReason>,
) -> Result<Option<ProjectedPrefixTransition<'a>>, PlanningUnknownReason> {
    poll()?;
    state.verify(input.requests);
    if state.fixture.refuse_restore {
        return Err(PlanningUnknownReason::UnknownResourceEvidence);
    }
    assert_eq!(state.checkpoint, Some((input.offer.identity, 0)));
    let target = input
        .requests
        .iter()
        .position(|r| r.key == input.offer.target)
        .unwrap();
    assert_eq!(state.contexts[target], 0);
    let mut next = state.clone();
    next.contexts[target] = input.offer.boundary_tokens.get();
    state.fixture.restored_roots.borrow_mut().push(state.root);
    let shape = maintenance_shape(PrefixMaintenanceStage::Restore);
    let cost_domain = if state.fixture.extra_alternative {
        let mut second = shape.clone();
        second.maintenance_bytes += 1;
        PlanningShapeDomain::HostContentAlternatives(vec![shape, second])
    } else {
        PlanningShapeDomain::Exact(shape)
    };
    Ok(Some(ProjectedPrefixTransition {
        cost_domain,
        restored_frontier: Some(PrefixRestoredFrontier {
            target: input.offer.target.clone(),
            previous_offset: u32::from(state.fixture.bad_frontier),
            restored_tokens: input.offer.boundary_tokens.get(),
        }),
        successor: Arc::new(next),
    }))
}

fn setup_ready() -> (SchedulerSnapshot, ReadyPrefixRestoreOffer) {
    let (mut snapshot, old) = setup();
    snapshot.requests.remove(0);
    let offer = ReadyPrefixRestoreOffer {
        identity: old.identity,
        based_on_generation: snapshot.generation,
        target: old.target,
        boundary_tokens: old.boundary_tokens,
        expires_at_ns: old.expires_at_ns,
    };
    (snapshot, offer)
}
fn plan(
    snapshot: &SchedulerSnapshot,
    offer: &ReadyPrefixRestoreOffer,
    phase: ReadyPrefixPhase,
    fixture: &Fixture,
    maintenance: &Maintenance,
) -> ReadyPrefixDecision {
    planner(6).plan_ready_prefix_restore(
        snapshot,
        offer,
        phase,
        &Model(infer),
        maintenance,
        fixture,
        &mut Clock(100),
    )
}
fn evidence(decision: ReadyPrefixDecision) -> ReadyPrefixEvidence {
    match decision {
        ReadyPrefixDecision::Ready { evidence, .. } => evidence,
        other => panic!("expected ready evidence: {other:?}"),
    }
}

#[test]
fn ready_cache_has_no_producer_and_replays_both_full_queue_trajectories() {
    let (snapshot, offer) = setup_ready();
    let original = snapshot.clone();
    let fixture = Fixture::default();
    let e = evidence(plan(
        &snapshot,
        &offer,
        ReadyPrefixPhase::Ready,
        &fixture,
        &Maintenance::default(),
    ));
    assert!(matches!(e.action(), ReadyPrefixAction::Restore(_)));
    assert!(e.remaining().first_commit_at_ns() < e.direct().unwrap().first_commit_at_ns());
    assert_eq!(e.offer(), &offer);
    assert!(
        fixture.captured_roots.borrow().is_empty(),
        "no future capture or producer action"
    );
    let bound = fixture.ready_bindings.borrow();
    assert_eq!(bound.len(), 2, "search and replay each bind a fresh root");
    assert_ne!(bound[0].0, bound[1].0);
    for path in [e.remaining(), e.direct().unwrap()] {
        assert!(
            path.steps()
                .iter()
                .any(|step| matches!(step, PrefixPathStep::Wave(w)
            if w.work.iter().any(|work| work.key == snapshot.requests[1].key))),
            "decoder obligation preserved"
        );
    }
    assert!(fixture
        .seen_owner_counts
        .borrow()
        .iter()
        .all(|&count| count == snapshot.requests.len()));
    assert_eq!(snapshot, original);
}

#[test]
fn ready_cache_missing_alternative_ttl_or_changed_replay_owner_is_unknown() {
    let (snapshot, offer) = setup_ready();
    for (fixture, maintenance) in [
        (
            Fixture::default(),
            Maintenance {
                missing: Some(PrefixMaintenanceStage::Restore),
                ..Default::default()
            },
        ),
        (
            Fixture::default(),
            Maintenance {
                ttl: 0,
                ..Default::default()
            },
        ),
        (
            Fixture {
                extra_alternative: true,
                ..Default::default()
            },
            Maintenance {
                uncovered_alternative: true,
                ..Default::default()
            },
        ),
        (
            Fixture {
                revoke_continuation_replay: true,
                ..Default::default()
            },
            Maintenance::default(),
        ),
        (
            Fixture {
                bad_frontier: true,
                ..Default::default()
            },
            Maintenance::default(),
        ),
    ] {
        assert!(matches!(
            plan(
                &snapshot,
                &offer,
                ReadyPrefixPhase::Ready,
                &fixture,
                &maintenance
            ),
            ReadyPrefixDecision::Unknown { .. }
        ));
    }
}

#[test]
fn ready_cache_cannot_hide_decoder_deadline_or_force_more_expensive_restore() {
    let (mut snapshot, offer) = setup_ready();
    let decision = plan(
        &snapshot,
        &offer,
        ReadyPrefixPhase::Ready,
        &Fixture::default(),
        &Maintenance {
            duration: 60,
            ..Default::default()
        },
    );
    assert!(
        matches!(decision, ReadyPrefixDecision::PreferDirect { .. }),
        "{decision:?}"
    );
    let decoder = &mut snapshot.requests[1];
    decoder.timing.maximum_output_tokens = n32(100);
    decoder.timing.first_commit_at_ns = Some(99);
    decoder.timing.last_commit_at_ns = Some(99);
    decoder.timing.budgets.itl_ns = n64(12);
    decoder.timing.budgets.tpot_ns = n64(12);
    snapshot.scope.horizon_end_ns = 120;
    let decision = plan(
        &snapshot,
        &offer,
        ReadyPrefixPhase::Ready,
        &Fixture::default(),
        &Maintenance {
            duration: 20,
            ..Default::default()
        },
    );
    assert!(
        !matches!(decision, ReadyPrefixDecision::Ready { .. }),
        "{decision:?}"
    );
}

#[test]
fn ready_cache_ack_continuation_uses_fresh_first_wave_permission_and_original_expiry() {
    let (mut snapshot, mut offer) = setup_ready();
    let target = &mut snapshot.requests[0];
    let RequestPhaseView::Prefill(progress) = &mut target.phase else {
        unreachable!()
    };
    progress.offset = offer.boundary_tokens.get();
    progress.logical_high_water = progress.offset;
    target.context_tokens = progress.offset;
    let fixture = Fixture::default();
    let e = evidence(plan(
        &snapshot,
        &offer,
        ReadyPrefixPhase::Restored,
        &fixture,
        &Maintenance::default(),
    ));
    let ReadyPrefixAction::Wave(wave) = e.action() else {
        panic!("ack must lead to model wave");
    };
    assert!(wave.final_replay_first_wave.is_some());
    assert!(wave.protection.is_some());
    assert_eq!(wave.snapshot_generation, snapshot.generation);
    assert_eq!(fixture.ready_bindings.borrow().len(), 2);
    assert!(
        fixture.restored_roots.borrow().is_empty(),
        "ack continuation must not recopy"
    );
    offer.expires_at_ns = 100;
    assert!(matches!(
        plan(
            &snapshot,
            &offer,
            ReadyPrefixPhase::Restored,
            &Fixture::default(),
            &Maintenance::default()
        ),
        ReadyPrefixDecision::Unknown {
            reason: PlanningUnknownReason::HorizonInsufficient,
            ..
        }
    ));
}

#[test]
fn ready_cache_cannot_swallow_original_controller_budget_or_stale_target_key() {
    let (snapshot, mut offer) = setup_ready();
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
    let decision = planner(6).plan_ready_prefix_restore(
        &snapshot,
        &offer,
        ReadyPrefixPhase::Ready,
        &Model(infer),
        &Maintenance::default(),
        &fixture,
        &mut Current(&fixture.now),
    );
    assert!(matches!(
        decision,
        ReadyPrefixDecision::Unknown {
            reason: PlanningUnknownReason::ComputeBudgetExhausted,
            ..
        }
    ));
    offer.target.incarnation += 1;
    assert!(matches!(
        plan(
            &snapshot,
            &offer,
            ReadyPrefixPhase::Ready,
            &Fixture::default(),
            &Maintenance::default()
        ),
        ReadyPrefixDecision::Unknown {
            reason: PlanningUnknownReason::InvalidSnapshot,
            ..
        }
    ));
}
