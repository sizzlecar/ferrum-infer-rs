//! Source-only capture does not create a target or discard queue obligations.
use super::*;

pub(super) fn bind<'a>(
    state: &State<'a>,
    input: &PlanningPrefixCacheCaptureBindingInput<'_>,
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
            && !state.fixture.cache_capture_bindings.borrow().is_empty())
    {
        return Err(PlanningUnknownReason::UnknownResourceEvidence);
    }
    assert_eq!(input.snapshot, state.snapshot);
    assert_eq!(input.offer.based_on_generation, state.snapshot.generation);
    assert!(state.checkpoint.is_none());
    assert_eq!(
        input.phase == PrefixCacheCapturePhase::Preparing,
        state.fixture.cache_capture_preparing
    );
    state
        .fixture
        .cache_capture_bindings
        .borrow_mut()
        .push(state.root);
    let mut next = state.clone();
    if input.phase == PrefixCacheCapturePhase::Preparing {
        next.cache_preparation = Some((
            input.offer.source.clone(),
            input.offer.boundary_tokens.get(),
        ));
    }
    Ok(Some(Arc::new(next)))
}

pub(super) fn capture<'a>(
    state: &State<'a>,
    input: &PlanningPrefixCacheCaptureInput<'_>,
    poll: &mut dyn FnMut() -> Result<(), PlanningUnknownReason>,
) -> Result<Option<ProjectedPrefixTransition<'a>>, PlanningUnknownReason> {
    state.verify(input.requests);
    if !state.fixture.cache_capture_preparing {
        assert_eq!(input.requests, input.snapshot.requests);
    }
    assert!(state
        .fixture
        .cache_capture_bindings
        .borrow()
        .contains(&state.root));
    let source = input
        .requests
        .iter()
        .position(|r| r.key == input.offer.source)
        .unwrap();
    assert_eq!(state.contexts[source], input.offer.boundary_tokens.get());
    if state.fixture.swallow_budget {
        let old = state.fixture.now.replace(10_000_000);
        let _ = poll();
        state.fixture.now.set(old);
    } else {
        poll()?;
    }
    let mut next = state.clone();
    next.checkpoint = Some((input.offer.identity, input.capture_span_start));
    state.fixture.captured_roots.borrow_mut().push(state.root);
    let shape = maintenance_shape(PrefixMaintenanceStage::Capture);
    let cost_domain = if state.fixture.extra_alternative {
        let mut second = shape.clone();
        second.maintenance_bytes += 1;
        PlanningShapeDomain::HostContentAlternatives(vec![shape, second])
    } else {
        PlanningShapeDomain::Exact(shape)
    };
    Ok(Some(ProjectedPrefixTransition {
        cost_domain,
        restored_frontier: state.fixture.bad_frontier.then(|| PrefixRestoredFrontier {
            target: input.offer.source.clone(),
            previous_offset: 0,
            restored_tokens: input.offer.boundary_tokens.get(),
        }),
        successor: Arc::new(next),
    }))
}

fn setup_capture() -> (SchedulerSnapshot, PrefixCacheCaptureOffer) {
    let (mut snapshot, old) = setup();
    let source = &mut snapshot.requests[0];
    let RequestPhaseView::Prefill(progress) = &mut source.phase else {
        unreachable!()
    };
    progress.offset = old.boundary_tokens.get();
    progress.logical_high_water = progress.offset;
    source.context_tokens = progress.offset;
    source.timing.budgets.ttft_ns = n64(150);
    let offer = PrefixCacheCaptureOffer {
        identity: old.identity,
        based_on_generation: snapshot.generation,
        source: old.producer,
        capture_span_start: 8,
        boundary_tokens: old.boundary_tokens,
        expires_at_ns: snapshot.scope.horizon_end_ns,
    };
    (snapshot, offer)
}
fn fixture() -> Fixture {
    Fixture {
        cache_capture_mode: true,
        ..Default::default()
    }
}
fn plan(
    snapshot: &SchedulerSnapshot,
    offer: &PrefixCacheCaptureOffer,
    fixture: &Fixture,
    maintenance: &Maintenance,
) -> PrefixCacheCaptureDecision {
    planner(6).plan_prefix_cache_capture(
        snapshot,
        offer,
        &Model(infer),
        maintenance,
        fixture,
        &mut Clock(100),
    )
}
fn evidence(decision: PrefixCacheCaptureDecision) -> PrefixCacheCaptureEvidence {
    match decision {
        PrefixCacheCaptureDecision::Known { evidence, .. } => evidence,
        other => panic!("expected complete capture trajectory: {other:?}"),
    }
}

#[test]
fn cache_capture_replays_retained_resource_successor_and_every_original_queue_owner() {
    let (snapshot, offer) = setup_capture();
    let before = snapshot.clone();
    let fixture = fixture();
    let proof = evidence(plan(&snapshot, &offer, &fixture, &Maintenance::default()));
    assert_eq!(proof.offer(), &offer);
    assert_eq!(proof.inference_model_version(), snapshot.cost_model_version);
    assert_eq!(proof.maintenance_model_version(), 23);
    assert!(
        matches!(proof.steps().first(), Some(PrefixPathStep::Maintenance(m))
        if m.stage == PrefixMaintenanceStage::Capture && m.restored_frontier.is_none())
    );
    assert_eq!(proof.capture().capture_span_start, 8);
    assert!(proof.completion_at_ns() > proof.validated_at_ns() + proof.first_action_cost_ns());
    assert!(proof.valid_until_ns() < offer.expires_at_ns);
    assert_eq!(proof.protection().rows().len(), snapshot.requests.len());
    for key in [&offer.source, &snapshot.requests[2].key] {
        assert!(proof
            .steps()
            .iter()
            .any(|step| matches!(step, PrefixPathStep::Wave(w)
            if w.work.iter().any(|work| &work.key == key))));
    }
    assert!(fixture.restored_roots.borrow().is_empty());
    let roots = fixture.cache_capture_bindings.borrow();
    assert_eq!(roots.len(), 2);
    assert_ne!(
        roots[0], roots[1],
        "replay must bind a different resource root"
    );
    assert_eq!(&*fixture.captured_roots.borrow(), &*roots);
    assert!(fixture
        .seen_owner_counts
        .borrow()
        .iter()
        .all(|&n| n == snapshot.requests.len()));
    assert_eq!(
        snapshot, before,
        "capture must not hold or advance any owner"
    );
}

#[test]
fn cache_capture_does_not_spend_a_model_wave_for_the_maintenance_edge() {
    let (mut snapshot, offer) = setup_capture();
    snapshot.requests.truncate(1);
    let proof = evidence(planner(1).plan_prefix_cache_capture(
        &snapshot,
        &offer,
        &Model(infer),
        &Maintenance::default(),
        &fixture(),
        &mut Clock(100),
    ));
    assert_eq!(proof.steps().len(), 2);
    assert!(matches!(proof.steps()[1], PrefixPathStep::Wave(_)));
}

#[test]
fn cache_capture_unknown_domain_stale_cost_or_changed_replay_state_cannot_publish() {
    let (snapshot, offer) = setup_capture();
    for (fixture, maintenance) in [
        (
            fixture(),
            Maintenance {
                missing: Some(PrefixMaintenanceStage::Capture),
                ..Default::default()
            },
        ),
        (
            fixture(),
            Maintenance {
                ttl: 0,
                ..Default::default()
            },
        ),
        (
            Fixture {
                extra_alternative: true,
                ..fixture()
            },
            Maintenance {
                uncovered_alternative: true,
                ..Default::default()
            },
        ),
        (
            Fixture {
                revoke_continuation_replay: true,
                ..fixture()
            },
            Maintenance::default(),
        ),
        (
            Fixture {
                bad_frontier: true,
                ..fixture()
            },
            Maintenance::default(),
        ),
        (
            Fixture {
                revoke_after_capture: true,
                ..fixture()
            },
            Maintenance::default(),
        ),
    ] {
        assert!(matches!(
            plan(&snapshot, &offer, &fixture, &maintenance),
            PrefixCacheCaptureDecision::Unknown { .. }
        ));
    }
}

#[test]
fn cache_capture_cannot_cross_peer_deadline_or_accept_stale_source_authority() {
    let (mut snapshot, mut offer) = setup_capture();
    let peer = &mut snapshot.requests[2];
    peer.timing.first_commit_at_ns = Some(99);
    peer.timing.last_commit_at_ns = Some(99);
    peer.timing.budgets.itl_ns = n64(2);
    peer.timing.budgets.tpot_ns = n64(2);
    assert!(matches!(
        plan(
            &snapshot,
            &offer,
            &fixture(),
            &Maintenance {
                duration: 3,
                ..Default::default()
            }
        ),
        PrefixCacheCaptureDecision::Unknown { .. }
    ));
    let (snapshot, _) = setup_capture();
    offer.source.incarnation += 1;
    assert!(matches!(
        plan(&snapshot, &offer, &fixture(), &Maintenance::default()),
        PrefixCacheCaptureDecision::Unknown {
            reason: PlanningUnknownReason::InvalidSnapshot,
            ..
        }
    ));
    offer.source.incarnation -= 1;
    offer.capture_span_start = offer.boundary_tokens.get();
    assert!(matches!(
        plan(&snapshot, &offer, &fixture(), &Maintenance::default()),
        PrefixCacheCaptureDecision::Unknown {
            reason: PlanningUnknownReason::InvalidSnapshot,
            ..
        }
    ));
}

#[test]
fn cache_capture_original_clock_window_and_swallowed_provider_budget_remain_binding() {
    let (snapshot, offer) = setup_capture();
    struct Current<'a>(&'a Cell<u64>);
    impl PlanningClock for Current<'_> {
        fn now_ns(&mut self) -> u64 {
            self.0.get()
        }
    }
    for at_binding in [false, true] {
        let fixture = Fixture {
            swallow_binding_budget: at_binding,
            swallow_budget: !at_binding,
            now: Cell::new(100),
            ..fixture()
        };
        let decision = planner(6).plan_prefix_cache_capture(
            &snapshot,
            &offer,
            &Model(infer),
            &Maintenance::default(),
            &fixture,
            &mut Current(&fixture.now),
        );
        assert!(matches!(
            decision,
            PrefixCacheCaptureDecision::Unknown {
                reason: PlanningUnknownReason::ComputeBudgetExhausted,
                ..
            }
        ));
    }
    let start = std::time::Instant::now();
    let origin =
        PlanningTimeOrigin::from_origin(start, start + std::time::Duration::from_nanos(100))
            .unwrap();
    let fixture = fixture();
    let decision = origin
        .plan_prefix_cache_capture_with_execution_budget_window(
            &planner(6),
            &snapshot,
            &offer,
            &Model(infer),
            &Maintenance::default(),
            &fixture,
            PlanningPhaseBudget {
                window: PlanningBudgetWindow {
                    started_at_ns: 0,
                    deadline_ns: 10_000,
                },
                planner_deadline_ns: Some(snapshot.observed_at_ns),
            },
            || origin.observed_at(),
        )
        .unwrap();
    assert!(matches!(
        decision,
        PrefixCacheCaptureDecision::Unknown {
            reason: PlanningUnknownReason::ComputeBudgetExhausted,
            ..
        }
    ));
    assert_eq!(fixture.roots.get(), 0);
}

fn setup_preparing() -> (SchedulerSnapshot, PrefixCacheCaptureOffer, Fixture) {
    let (mut snapshot, mut offer) = setup_capture();
    snapshot.requests.truncate(1);
    let source = &mut snapshot.requests[0];
    let RequestPhaseView::Prefill(progress) = &mut source.phase else {
        unreachable!()
    };
    progress.offset = 0;
    progress.logical_high_water = 0;
    source.context_tokens = 0;
    // Put the original first-token obligation inside this finite horizon.
    // Otherwise reaching the capture boundary alone is already a complete
    // witness, and H legitimately excludes the later suffix.
    source.timing.budgets.ttft_ns =
        n64(snapshot.scope.horizon_end_ns - source.timing.ingress_at_ns);
    offer.capture_span_start = 0;
    (
        snapshot,
        offer,
        Fixture {
            cache_capture_preparing: true,
            ..fixture()
        },
    )
}

#[test]
fn cache_capture_preparing_replays_legal_waves_then_dynamic_capture_span_and_tail() {
    let (snapshot, offer, fixture) = setup_preparing();
    let before = snapshot.clone();
    assert_eq!(
        snapshot.requests[0].timing.next_deadline_ns(),
        Some(snapshot.scope.horizon_end_ns),
    );
    let proof = evidence(planner(4).plan_prefix_cache_capture_in_phase(
        &snapshot,
        &offer,
        PrefixCacheCapturePhase::Preparing,
        &Model(infer),
        &Maintenance::default(),
        &fixture,
        &mut Clock(100),
    ));
    assert_eq!(proof.phase(), PrefixCacheCapturePhase::Preparing);
    let PrefixCacheCaptureAction::Wave(selected) = proof.action() else {
        panic!("preparation must return the original selected-wave protocol");
    };
    assert!(selected.final_replay_first_wave.is_some());
    assert_eq!(
        selected.protection.as_ref().unwrap().rows().len(),
        snapshot.requests.len()
    );
    assert_eq!(
        selected.candidate,
        match &proof.steps()[0] {
            PrefixPathStep::Wave(wave) => wave.clone(),
            _ => unreachable!(),
        }
    );
    assert_eq!(
        selected.candidate.work[0].action,
        WaveAction::Prefill {
            offset: 0,
            count: n32(4)
        }
    );
    assert_eq!(proof.capture().capture_span_start, 8);
    assert_ne!(proof.capture().capture_span_start, offer.capture_span_start);
    assert_eq!(
        proof
            .steps()
            .iter()
            .filter(|step| matches!(step, PrefixPathStep::Wave(_)))
            .count(),
        4
    );
    assert!(matches!(proof.steps()[3], PrefixPathStep::Maintenance(_)));
    assert!(matches!(&proof.steps()[4], PrefixPathStep::Wave(wave)
        if wave.work.iter().any(|work| work.action == WaveAction::Prefill { offset: 12, count: n32(4) })));
    let roots = fixture.cache_capture_bindings.borrow();
    assert_eq!(roots.len(), 2);
    assert_ne!(roots[0], roots[1]);
    assert_eq!(&*fixture.captured_roots.borrow(), &*roots);
    assert_eq!(snapshot, before);

    // Moving only the genuine obligation beyond this horizon permits the
    // three boundary waves plus Capture; there is no fabricated tail duty.
    let (mut later, offer, fixture) = setup_preparing();
    later.requests[0].timing.budgets.ttft_ns =
        n64(later.scope.horizon_end_ns + 1 - later.requests[0].timing.ingress_at_ns);
    let shorter = evidence(planner(3).plan_prefix_cache_capture_in_phase(
        &later,
        &offer,
        PrefixCacheCapturePhase::Preparing,
        &Model(infer),
        &Maintenance::default(),
        &fixture,
        &mut Clock(100),
    ));
    assert_eq!(
        shorter
            .steps()
            .iter()
            .filter(|step| matches!(step, PrefixPathStep::Wave(_)))
            .count(),
        3
    );
    assert!(matches!(
        shorter.steps().last(),
        Some(PrefixPathStep::Maintenance(_))
    ));
}

#[test]
fn cache_capture_preparing_keeps_original_caps_horizon_and_replay_authority() {
    let (mut snapshot, mut offer, _) = setup_preparing();
    let run = |snapshot: &SchedulerSnapshot,
               offer: &PrefixCacheCaptureOffer,
               fixture: &Fixture,
               horizon| {
        planner(horizon).plan_prefix_cache_capture_in_phase(
            snapshot,
            offer,
            PrefixCacheCapturePhase::Preparing,
            &Model(infer),
            &Maintenance::default(),
            fixture,
            &mut Clock(100),
        )
    };
    let fixture = || Fixture {
        cache_capture_preparing: true,
        ..fixture()
    };
    assert!(matches!(
        run(&snapshot, &offer, &fixture(), 3),
        PrefixCacheCaptureDecision::Unknown { .. }
    ));
    snapshot.capabilities.prefill_chunk_sizes = vec![n32(16)];
    assert!(
        matches!(
            run(&snapshot, &offer, &fixture(), 4),
            PrefixCacheCaptureDecision::Unknown { .. }
        ),
        "declared boundary cannot invent a chunk absent from existing capabilities"
    );
    snapshot.capabilities.prefill_chunk_sizes = vec![n32(4), n32(16)];
    let revoked = Fixture {
        revoke_continuation_replay: true,
        ..fixture()
    };
    assert!(matches!(
        run(&snapshot, &offer, &revoked, 4),
        PrefixCacheCaptureDecision::Unknown { .. }
    ));
    offer.capture_span_start = 1;
    assert!(matches!(
        run(&snapshot, &offer, &fixture(), 4),
        PrefixCacheCaptureDecision::Unknown {
            reason: PlanningUnknownReason::InvalidSnapshot,
            ..
        }
    ));
}

#[test]
fn cache_capture_preparing_keeps_peer_obligations_and_checks_maintenance_before_wave_permission() {
    let (mut snapshot, mut offer, _) = setup_preparing();
    let mut peer = decode(9);
    peer.timing.first_commit_at_ns = Some(99);
    peer.timing.last_commit_at_ns = Some(99);
    peer.timing.budgets.itl_ns = n64(2);
    peer.timing.budgets.tpot_ns = n64(2);
    snapshot.requests.push(peer);
    let fixture = Fixture {
        cache_capture_preparing: true,
        ..fixture()
    };
    let run = |snapshot: &SchedulerSnapshot,
               offer: &PrefixCacheCaptureOffer,
               maintenance: &Maintenance| {
        planner(6).plan_prefix_cache_capture_in_phase(
            snapshot,
            offer,
            PrefixCacheCapturePhase::Preparing,
            &Model(infer),
            maintenance,
            &fixture,
            &mut Clock(100),
        )
    };
    assert!(matches!(
        run(&snapshot, &offer, &Maintenance::default()),
        PrefixCacheCaptureDecision::Unknown { .. }
    ));
    snapshot.requests.truncate(1);
    assert!(
        matches!(
            run(
                &snapshot,
                &offer,
                &Maintenance {
                    missing: Some(PrefixMaintenanceStage::Capture),
                    ..Default::default()
                }
            ),
            PrefixCacheCaptureDecision::Unknown { .. }
        ),
        "an inference first wave cannot carry an uncosted optional capture promise"
    );
    offer.expires_at_ns = snapshot.observed_at_ns;
    assert!(matches!(
        run(&snapshot, &offer, &Maintenance::default()),
        PrefixCacheCaptureDecision::Unknown { .. }
    ));
}
