//! Abstract resource lineages and virtual clocks. These exercise the planner
//! protocol; backend checkpoint address/layout proofs have separate tests.
use super::*;
use std::cell::{Cell, RefCell};

mod cache_capture;
mod clock;
mod continuation;
mod goal_order;
mod ready_cache;
mod scoped;

#[derive(Default)]
struct Fixture {
    roots: Cell<u64>,
    captured_roots: RefCell<Vec<u64>>,
    restored_roots: RefCell<Vec<u64>>,
    seen_owner_counts: RefCell<Vec<usize>>,
    revoke_after_capture: bool,
    bad_frontier: bool,
    refuse_restore: bool,
    extra_alternative: bool,
    swallow_budget: bool,
    replay_restore_clock: Option<u64>,
    now: Cell<u64>,
    continuation_roots: RefCell<Vec<(u64, PrefixContinuationPhase)>>,
    refuse_continuation: bool,
    revoke_continuation_replay: bool,
    swallow_binding_budget: bool,
    ready_bindings: RefCell<Vec<(u64, ReadyPrefixPhase)>>,
    cache_capture_mode: bool,
    cache_capture_bindings: RefCell<Vec<u64>>,
    cache_capture_preparing: bool,
    rejected_prefill_counts: Vec<u32>,
    shape_rejections: Cell<usize>,
}

#[derive(Clone)]
struct State<'a> {
    fixture: &'a Fixture,
    snapshot: &'a SchedulerSnapshot,
    root: u64,
    contexts: Vec<u32>,
    checkpoint: Option<([u8; 32], u32)>,
    cache_preparation: Option<(RequestWorkKey, u32)>,
}

impl PlanningExecutionContext for Fixture {
    fn begin<'a>(
        &'a self,
        snapshot: &'a SchedulerSnapshot,
        poll: &mut dyn FnMut() -> Result<(), PlanningUnknownReason>,
    ) -> Result<Arc<dyn PlanningExecutionState<'a> + 'a>, PlanningUnknownReason> {
        poll()?;
        if self.revoke_after_capture && !self.captured_roots.borrow().is_empty() {
            return Err(PlanningUnknownReason::UnknownResourceEvidence);
        }
        self.roots.set(self.roots.get() + 1);
        Ok(Arc::new(State {
            fixture: self,
            snapshot,
            root: self.roots.get(),
            contexts: snapshot.requests.iter().map(|r| r.context_tokens).collect(),
            checkpoint: None,
            cache_preparation: None,
        }))
    }
}

impl State<'_> {
    fn verify(&self, requests: &[RequestSchedulingView]) {
        assert_eq!(requests.len(), self.snapshot.requests.len());
        self.fixture
            .seen_owner_counts
            .borrow_mut()
            .push(requests.len());
        for ((initial, current), context) in self
            .snapshot
            .requests
            .iter()
            .zip(requests)
            .zip(&self.contexts)
        {
            assert_eq!(initial.key, current.key);
            assert_eq!(
                current.context_tokens, *context,
                "logical frontier must match this physical lineage"
            );
            assert_eq!(initial.timing.ingress_at_ns, current.timing.ingress_at_ns);
        }
    }
}

impl<'a> PlanningExecutionState<'a> for State<'a> {
    fn bind_prefix_cache_capture(
        &self,
        input: &PlanningPrefixCacheCaptureBindingInput<'_>,
        poll: &mut dyn FnMut() -> Result<(), PlanningUnknownReason>,
    ) -> Result<Option<Arc<dyn PlanningExecutionState<'a> + 'a>>, PlanningUnknownReason> {
        cache_capture::bind(self, input, poll)
    }

    fn project_prefix_cache_capture(
        &self,
        input: &PlanningPrefixCacheCaptureInput<'_>,
        poll: &mut dyn FnMut() -> Result<(), PlanningUnknownReason>,
    ) -> Result<Option<ProjectedPrefixTransition<'a>>, PlanningUnknownReason> {
        cache_capture::capture(self, input, poll)
    }

    fn bind_ready_prefix(
        &self,
        input: &PlanningReadyPrefixInput<'_>,
        poll: &mut dyn FnMut() -> Result<(), PlanningUnknownReason>,
    ) -> Result<Option<Arc<dyn PlanningExecutionState<'a> + 'a>>, PlanningUnknownReason> {
        ready_cache::bind(self, input, poll)
    }

    fn project_ready_prefix_restore(
        &self,
        input: &PlanningReadyPrefixRestoreInput<'_>,
        poll: &mut dyn FnMut() -> Result<(), PlanningUnknownReason>,
    ) -> Result<Option<ProjectedPrefixTransition<'a>>, PlanningUnknownReason> {
        ready_cache::restore(self, input, poll)
    }

    fn bind_prefix_continuation(
        &self,
        input: &PlanningPrefixContinuationInput<'_>,
        poll: &mut dyn FnMut() -> Result<(), PlanningUnknownReason>,
    ) -> Result<Option<Arc<dyn PlanningExecutionState<'a> + 'a>>, PlanningUnknownReason> {
        if self.fixture.swallow_binding_budget {
            let old = self.fixture.now.replace(10_000_000);
            let _ = poll();
            self.fixture.now.set(old);
        } else {
            poll()?;
        }
        if self.fixture.refuse_continuation
            || (self.fixture.revoke_continuation_replay
                && !self.fixture.continuation_roots.borrow().is_empty())
        {
            return Err(PlanningUnknownReason::UnknownResourceEvidence);
        }
        assert_eq!(input.snapshot, self.snapshot);
        assert_eq!(input.offer.based_on_generation, self.snapshot.generation);
        self.fixture
            .continuation_roots
            .borrow_mut()
            .push((self.root, input.phase));
        let mut next = self.clone();
        if let PrefixContinuationPhase::CheckpointReady { capture_span_start } = input.phase {
            // Fixture analogue of fresh binding of the actual retained token.
            // Production must prove this using the runtime-owned checkpoint.
            next.checkpoint = Some((input.offer.identity, capture_span_start));
        }
        Ok(Some(Arc::new(next)))
    }

    fn project(
        &self,
        input: &PlanningExecutionInput<'_>,
        poll: &mut dyn FnMut() -> Result<(), PlanningUnknownReason>,
    ) -> Result<Option<ProjectedExecution<'a>>, PlanningUnknownReason> {
        self.verify(input.requests);
        if !self.fixture.rejected_prefill_counts.is_empty()
            && input.work.len() == self.fixture.rejected_prefill_counts.len()
            && input.work.iter().zip(&self.fixture.rejected_prefill_counts).all(|(row, count)| {
                matches!(row.action, WaveAction::Prefill { count: actual, .. } if actual.get() == *count)
            })
        {
            self.fixture.shape_rejections.set(self.fixture.shape_rejections.get() + 1);
            return Err(PlanningUnknownReason::ShapeUnavailable);
        }
        if self.fixture.cache_capture_mode {
            assert!(
                self.checkpoint.is_some()
                    || self
                        .cache_preparation
                        .as_ref()
                        .is_some_and(|(source, boundary)| {
                            let index = input
                                .requests
                                .iter()
                                .position(|r| &r.key == source)
                                .unwrap();
                            self.contexts[index] < *boundary
                        }),
                "ordinary tail must retain this branch's capture successor"
            );
        }
        let canonical = TestResolver
            .resolve(
                &PlanningShapeQuery {
                    snapshot: self.snapshot,
                    prior_waves: &[],
                    kind: input.kind,
                    rows: input.rows,
                    recurrent_state_bytes: input.recurrent_state_bytes,
                },
                poll,
            )?
            .unwrap();
        let mut next = self.clone();
        for work in input.work {
            let index = input
                .requests
                .iter()
                .position(|r| r.key == work.key)
                .unwrap();
            let advance =
                crate::implementations::continuous::slo_planner::output::work_output_advance(
                    &input.requests[index],
                    &work.action,
                    self.snapshot.capacity.maximum_context_tokens.get(),
                )?;
            next.contexts[index] = advance.context_tokens;
        }
        Ok(Some(ProjectedExecution {
            host_content_forecasts: None,
            statistical_evidence: None,
            ordered_work: input.work.to_vec(),
            canonical_domain: PlanningShapeDomain::Exact(canonical),
            successor: Arc::new(next),
        }))
    }

    fn project_prefix_transition(
        &self,
        input: &PlanningPrefixTransitionInput<'_>,
        poll: &mut dyn FnMut() -> Result<(), PlanningUnknownReason>,
    ) -> Result<Option<ProjectedPrefixTransition<'a>>, PlanningUnknownReason> {
        self.verify(input.requests);
        assert_eq!(input.snapshot.generation, input.offer.based_on_generation);
        let producer = input
            .requests
            .iter()
            .position(|r| r.key == input.offer.producer)
            .unwrap();
        let target = input
            .requests
            .iter()
            .position(|r| r.key == input.offer.target)
            .unwrap();
        if input.stage == PrefixMaintenanceStage::Capture {
            assert_eq!(self.contexts[producer], input.offer.boundary_tokens.get());
        }
        assert!(input.capture_span_start < input.offer.boundary_tokens.get());
        if self.fixture.swallow_budget {
            let old = self.fixture.now.replace(10_000_000);
            let _ = poll();
            self.fixture.now.set(old);
        } else {
            poll()?;
        }
        let mut next = self.clone();
        let restored_frontier = match input.stage {
            PrefixMaintenanceStage::Capture => {
                assert!(self.checkpoint.is_none());
                next.checkpoint = Some((input.offer.identity, input.capture_span_start));
                self.fixture.captured_roots.borrow_mut().push(self.root);
                None
            }
            PrefixMaintenanceStage::Restore => {
                if self.fixture.refuse_restore {
                    return Err(PlanningUnknownReason::UnknownResourceEvidence);
                }
                assert_eq!(
                    self.checkpoint,
                    Some((input.offer.identity, input.capture_span_start)),
                    "restore needs this branch's own capture proof"
                );
                assert_eq!(self.contexts[target], 0);
                next.contexts[target] = input.offer.boundary_tokens.get();
                self.fixture.restored_roots.borrow_mut().push(self.root);
                if self.fixture.captured_roots.borrow().first() != Some(&self.root) {
                    if let Some(now) = self.fixture.replay_restore_clock {
                        self.fixture.now.set(now);
                    }
                }
                Some(PrefixRestoredFrontier {
                    target: input.offer.target.clone(),
                    previous_offset: u32::from(self.fixture.bad_frontier),
                    restored_tokens: input.offer.boundary_tokens.get(),
                })
            }
        };
        let shape = maintenance_shape(input.stage);
        let cost_domain = if self.fixture.extra_alternative {
            let mut second = shape.clone();
            second.maintenance_bytes += 1;
            PlanningShapeDomain::HostContentAlternatives(vec![shape, second])
        } else {
            PlanningShapeDomain::Exact(shape)
        };
        Ok(Some(ProjectedPrefixTransition {
            cost_domain,
            restored_frontier,
            successor: Arc::new(next),
        }))
    }
}

fn maintenance_shape(stage: PrefixMaintenanceStage) -> WaveExecutionShape {
    WaveExecutionShape {
        row_multiset_features: None,
        host_content_features: None,
        numeric_features: None,
        kind: if stage == PrefixMaintenanceStage::Capture {
            WaveKind::Maintenance
        } else {
            WaveKind::Restore
        },
        path: WaveExecutionPath::PlanRuntime,
        provider_signature: [8; 32],
        output_policy_signature: [9; 32],
        graph_state: WaveGraphState::Disabled,
        order: BatchOrderSemantics::Ordered,
        decode_kv_tokens: vec![],
        prefill_chunks: vec![],
        recurrent_state_bytes: 32,
        restore_bytes: if stage == PrefixMaintenanceStage::Restore {
            96
        } else {
            0
        },
        maintenance_bytes: 96,
        maintenance_units: 1,
    }
}

struct Maintenance {
    duration: u64,
    missing: Option<PrefixMaintenanceStage>,
    uncovered_alternative: bool,
    ttl: u64,
}
impl Default for Maintenance {
    fn default() -> Self {
        Self {
            duration: 1,
            missing: None,
            uncovered_alternative: false,
            ttl: u64::MAX,
        }
    }
}
impl PlanningPrefixCostModel for Maintenance {
    fn predict_cache_capture(
        &self,
        fingerprint: &ExecutionFingerprint,
        offer: &PrefixCacheCaptureOffer,
        shape: &WaveExecutionShape,
        _: u64,
    ) -> Option<PlanningCost> {
        assert_eq!(fingerprint.model_weights, [1; 32]);
        assert_eq!(offer.identity, [42; 32]);
        if self.missing == Some(PrefixMaintenanceStage::Capture)
            || (self.uncovered_alternative && shape.maintenance_bytes > 96)
        {
            return None;
        }
        Some(PlanningCost {
            typical_ns: self.duration,
            planning_ns: self.duration,
            model_version: 23,
            valid_for_ns: self.ttl,
        })
    }

    fn predict_ready_restore(
        &self,
        fingerprint: &ExecutionFingerprint,
        offer: &ReadyPrefixRestoreOffer,
        shape: &WaveExecutionShape,
        _: u64,
    ) -> Option<PlanningCost> {
        assert_eq!(fingerprint.model_weights, [1; 32]);
        assert_eq!(offer.identity, [42; 32]);
        if self.missing == Some(PrefixMaintenanceStage::Restore)
            || (self.uncovered_alternative && shape.maintenance_bytes > 96)
        {
            return None;
        }
        Some(PlanningCost {
            typical_ns: self.duration,
            planning_ns: self.duration,
            model_version: 23,
            valid_for_ns: self.ttl,
        })
    }

    fn model_version(&self) -> u64 {
        23
    }
    fn predict(
        &self,
        fingerprint: &ExecutionFingerprint,
        offer: &PrefixRendezvousOffer,
        stage: PrefixMaintenanceStage,
        shape: &WaveExecutionShape,
        _: u64,
    ) -> Option<PlanningCost> {
        assert_eq!(fingerprint.model_weights, [1; 32]);
        assert_eq!(offer.identity, [42; 32]);
        if self.missing == Some(stage)
            || (self.uncovered_alternative && shape.maintenance_bytes > 96)
        {
            return None;
        }
        Some(PlanningCost {
            typical_ns: self.duration,
            planning_ns: self.duration,
            model_version: 23,
            valid_for_ns: self.ttl,
        })
    }
}

fn setup() -> (SchedulerSnapshot, PrefixRendezvousOffer) {
    let mut producer = prefill(1);
    let mut target = prefill(2);
    for request in [&mut producer, &mut target] {
        request.timing.budgets.ttft_ns = n64(500);
        let RequestPhaseView::Prefill(p) = &mut request.phase else {
            unreachable!()
        };
        p.total_prompt_tokens = n32(16);
        p.executable_until = 16;
        p.reference = Arc::new(PrefillReferenceWork {
            evaluation: Default::default(),
            version: 1,
            points: (0..=4)
                .map(|n| ReferenceWorkPoint {
                    prompt_tokens: n * 4,
                    cumulative_work_ns: u64::from(n) * 40,
                })
                .collect(),
        });
    }
    let RequestPhaseView::Prefill(p) = &mut producer.phase else {
        unreachable!()
    };
    p.offset = 8;
    p.logical_high_water = 8;
    producer.context_tokens = 8;
    let mut snapshot = snapshot(vec![producer, target, decode(3)]);
    snapshot.capabilities.prefill_chunk_sizes = vec![n32(4), n32(16)];
    snapshot.capabilities.native_mixed = false;
    let offer = PrefixRendezvousOffer {
        identity: [42; 32],
        based_on_generation: snapshot.generation,
        producer: snapshot.requests[0].key.clone(),
        target: snapshot.requests[1].key.clone(),
        boundary_tokens: n32(12),
        expires_at_ns: 490,
    };
    (snapshot, offer)
}

fn infer(shape: &WaveExecutionShape) -> Option<u64> {
    Some(if shape.prefill_chunks.is_empty() {
        2
    } else {
        shape
            .prefill_chunks
            .iter()
            .map(|row| u64::from(row.count.get()) * 4)
            .sum()
    })
}
fn compare(
    snapshot: &SchedulerSnapshot,
    offer: &PrefixRendezvousOffer,
    fixture: &Fixture,
    maintenance: &Maintenance,
) -> PrefixRendezvousDecision {
    planner(12).compare_prefix_rendezvous(
        snapshot,
        offer,
        &Model(infer),
        maintenance,
        fixture,
        &mut Clock(100),
    )
}
fn comparison(decision: PrefixRendezvousDecision) -> PrefixRendezvousComparison {
    match decision {
        PrefixRendezvousDecision::Compared { comparison, .. } => comparison,
        other => panic!("expected both complete trajectories: {other:?}"),
    }
}

#[test]
fn prefix_comparison_replays_both_complete_queues_and_real_capture_restore_lineages() {
    let (snapshot, offer) = setup();
    let original = snapshot.clone();
    let fixture = Fixture::default();
    let result = comparison(compare(
        &snapshot,
        &offer,
        &fixture,
        &Maintenance::default(),
    ));
    assert!(result.should_hold());
    assert_eq!(result.direct().first_commit_at_ns(), 166);
    assert_eq!(result.waiting().first_commit_at_ns(), 136);
    assert!(
        result.waiting().capture_ready_at_ns().unwrap()
            < result.waiting().restore_ready_at_ns().unwrap()
    );
    for path in [result.direct(), result.waiting()] {
        assert!(
            path.steps()
                .iter()
                .any(|step| matches!(step, PrefixPathStep::Wave(w)
            if w.work.iter().any(|row| row.key == snapshot.requests[2].key))),
            "old decoder remains an obligation"
        );
    }
    let captured = fixture.captured_roots.borrow();
    let restored = fixture.restored_roots.borrow();
    assert!(
        captured.iter().any(|root| Some(root) != captured.first()),
        "fresh replay must capture on a new root"
    );
    assert!(restored.iter().all(|root| captured.contains(root)));
    assert!(fixture
        .seen_owner_counts
        .borrow()
        .iter()
        .all(|count| *count == snapshot.requests.len()));
    assert_eq!(snapshot, original);
}

#[test]
fn prefix_three_model_waves_allow_two_typed_maintenance_phases_without_expanding_model_horizon() {
    let (snapshot, offer) = setup();
    let fixture = Fixture::default();
    let decision = planner(3).compare_prefix_rendezvous(
        &snapshot,
        &offer,
        &Model(infer),
        &Maintenance::default(),
        &fixture,
        &mut Clock(100),
    );
    let result = comparison(decision);
    assert!(result.should_hold());
    assert_eq!(
        result
            .waiting()
            .steps()
            .iter()
            .filter(|step| matches!(step, PrefixPathStep::Wave(_)))
            .count(),
        3
    );
    assert_eq!(
        result
            .waiting()
            .steps()
            .iter()
            .filter(|step| matches!(step, PrefixPathStep::Maintenance(_)))
            .count(),
        2
    );
}

#[test]
fn prefix_direct_goal_does_not_charge_unnecessary_producer_work_to_its_first_token() {
    let (mut snapshot, offer) = setup();
    let RequestPhaseView::Prefill(progress) = &mut snapshot.requests[0].phase else {
        unreachable!()
    };
    progress.offset = 0;
    progress.logical_high_water = 0;
    snapshot.requests[0].context_tokens = 0;
    let result = comparison(compare(
        &snapshot,
        &offer,
        &Fixture::default(),
        &Maintenance::default(),
    ));
    assert_eq!(result.direct().first_commit_at_ns(), 166);
    assert!(result.waiting().first_commit_at_ns() > result.direct().first_commit_at_ns());
    assert!(!result.should_hold());
}

#[test]
fn prefix_missing_transfer_cost_physical_restore_or_one_alternative_never_authorizes_hold() {
    let (snapshot, offer) = setup();
    for stage in [
        PrefixMaintenanceStage::Capture,
        PrefixMaintenanceStage::Restore,
    ] {
        let decision = compare(
            &snapshot,
            &offer,
            &Fixture::default(),
            &Maintenance {
                missing: Some(stage),
                ..Default::default()
            },
        );
        assert!(
            matches!(decision, PrefixRendezvousDecision::Unknown { search, .. } if search.cost_unknown_candidates > 0)
        );
    }
    let decision = compare(
        &snapshot,
        &offer,
        &Fixture {
            refuse_restore: true,
            ..Default::default()
        },
        &Maintenance::default(),
    );
    assert!(
        matches!(decision, PrefixRendezvousDecision::Unknown { search, .. } if search.resource_unknown_candidates > 0)
    );
    let decision = compare(
        &snapshot,
        &offer,
        &Fixture {
            extra_alternative: true,
            ..Default::default()
        },
        &Maintenance {
            uncovered_alternative: true,
            ..Default::default()
        },
    );
    assert!(
        matches!(decision, PrefixRendezvousDecision::Unknown { search, .. } if search.cost_unknown_candidates > 0)
    );
}

#[test]
fn prefix_restore_receipt_cannot_replace_target_frontier_or_fresh_replay() {
    let (snapshot, offer) = setup();
    let decision = compare(
        &snapshot,
        &offer,
        &Fixture {
            bad_frontier: true,
            ..Default::default()
        },
        &Maintenance::default(),
    );
    assert!(matches!(
        decision,
        PrefixRendezvousDecision::Unknown {
            reason: PlanningUnknownReason::InvalidShapeEvidence,
            ..
        }
    ));
    let fixture = Fixture {
        revoke_after_capture: true,
        ..Default::default()
    };
    let decision = compare(&snapshot, &offer, &fixture, &Maintenance::default());
    assert!(!fixture.captured_roots.borrow().is_empty());
    assert!(
        matches!(decision, PrefixRendezvousDecision::Unknown { reason: PlanningUnknownReason::UnknownResourceEvidence, search }
        if search.phase == PlanningSearchPhase::Finalization)
    );
}

#[test]
fn prefix_long_capture_cannot_hide_an_existing_decoders_itl_obligation() {
    let (mut snapshot, offer) = setup();
    snapshot.scope.horizon_end_ns = 120;
    let old = &mut snapshot.requests[2];
    old.timing.maximum_output_tokens = n32(100);
    old.timing.first_commit_at_ns = Some(99);
    old.timing.last_commit_at_ns = Some(99);
    old.timing.budgets.itl_ns = n64(12);
    old.timing.budgets.tpot_ns = n64(12);
    let fast = Model(|shape: &WaveExecutionShape| {
        Some(if shape.prefill_chunks.is_empty() {
            2
        } else {
            4
        })
    });
    let decision = planner(16).compare_prefix_rendezvous(
        &snapshot,
        &offer,
        &fast,
        &Maintenance {
            duration: 20,
            ..Default::default()
        },
        &Fixture::default(),
        &mut Clock(100),
    );
    assert!(
        matches!(decision, PrefixRendezvousDecision::Unknown { .. }),
        "a capture longer than its peer ITL cannot be erased: {decision:?}"
    );
}

#[test]
fn prefix_maintenance_cannot_swallow_the_original_budget_or_outlive_cost_ttl() {
    let (snapshot, offer) = setup();
    let fixture = Fixture {
        swallow_budget: true,
        now: Cell::new(100),
        ..Default::default()
    };
    struct Current<'a>(&'a Cell<u64>);
    impl PlanningClock for Current<'_> {
        fn now_ns(&mut self) -> u64 {
            self.0.get()
        }
    }
    let decision = planner(12).compare_prefix_rendezvous(
        &snapshot,
        &offer,
        &Model(infer),
        &Maintenance::default(),
        &fixture,
        &mut Current(&fixture.now),
    );
    assert!(matches!(
        decision,
        PrefixRendezvousDecision::Unknown {
            reason: PlanningUnknownReason::ComputeBudgetExhausted,
            ..
        }
    ));
    let decision = compare(
        &snapshot,
        &offer,
        &Fixture::default(),
        &Maintenance {
            ttl: 0,
            ..Default::default()
        },
    );
    assert!(
        matches!(decision, PrefixRendezvousDecision::Unknown { search, .. } if search.cost_unknown_candidates > 0)
    );
}

#[test]
fn prefix_final_common_clock_rechecks_both_replayed_cost_lifetimes() {
    let (snapshot, offer) = setup();
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
    let decision = planner(12).compare_prefix_rendezvous(
        &snapshot,
        &offer,
        &Model(infer),
        &Maintenance {
            ttl: 10,
            ..Default::default()
        },
        &fixture,
        &mut Current(&fixture.now),
    );
    assert!(fixture
        .restored_roots
        .borrow()
        .iter()
        .any(|root| Some(root) != fixture.restored_roots.borrow().first()));
    assert!(
        matches!(decision, PrefixRendezvousDecision::Unknown { reason: PlanningUnknownReason::SearchIncomplete, search }
        if search.phase == PlanningSearchPhase::Finalization),
        "late second replay cannot publish stale first/maintenance evidence: {decision:?}"
    );
}

#[test]
fn prefix_maintenance_preserves_prior_credit_and_dates_restored_milestones_once() {
    use crate::implementations::continuous::slo_planner::prefix_rendezvous::PathOffer;

    let (mut snapshot, offer) = setup();
    snapshot.requests.truncate(2);
    snapshot.requests[0].context_tokens = 12;
    let RequestPhaseView::Prefill(source) = &mut snapshot.requests[0].phase else {
        unreachable!()
    };
    source.offset = 12;
    source.logical_high_water = 12;
    source.milestones = Arc::from([PrefillMilestone {
        at_ns: 105,
        required_reference_work_ns: 80,
    }]);
    let RequestPhaseView::Prefill(target) = &mut snapshot.requests[1].phase else {
        unreachable!()
    };
    target.milestones = Arc::from([PrefillMilestone {
        at_ns: 110,
        required_reference_work_ns: 80,
    }]);
    let original = snapshot.clone();
    let protection = PlanningObligationSet::capture(&snapshot, 100).unwrap();
    let fixture = Fixture::default();
    let maintenance = Maintenance {
        duration: 2,
        ..Default::default()
    };
    let mut initial = simulation::begin(&snapshot, &fixture, &mut || Ok(()), 100).unwrap();
    simulation::restrict_prefix_wait(&mut initial, &offer).unwrap();
    let (_, captured) = simulation::advance_prefix(
        &snapshot,
        &initial,
        PathOffer::Rendezvous(&offer),
        PrefixMaintenanceStage::Capture,
        Some(8),
        &maintenance,
        23,
        &mut || Ok(()),
        true,
        &protection,
    )
    .unwrap();
    assert!(captured.minimum_start_slack_ns > 6);
    let (_, restored) = simulation::advance_prefix(
        &snapshot,
        &captured,
        PathOffer::Rendezvous(&offer),
        PrefixMaintenanceStage::Restore,
        Some(8),
        &maintenance,
        23,
        &mut || Ok(()),
        true,
        &protection,
    )
    .unwrap();
    assert_eq!(restored.now_ns, 104);
    assert_eq!(restored.minimum_start_slack_ns, 6);
    let suffix = simulation::advance(
        &snapshot,
        &restored,
        &[CandidateWork {
            key: offer.target.clone(),
            action: WaveAction::Prefill {
                offset: 12,
                count: n32(4),
            },
        }],
        &Model(infer),
        true,
        &mut || Ok(()),
        true,
        Some(&protection),
    )
    .unwrap_or_else(|_| panic!("the suffix keeps the earlier restore credit"));
    assert_eq!(suffix.state.now_ns, 120);
    assert_eq!(suffix.state.minimum_start_slack_ns, 6);

    // Moving the actual restore past the target's still-unmet milestone is
    // rejected; source's historical credit cannot satisfy the target's debt.
    let mut late = simulation::begin(&snapshot, &fixture, &mut || Ok(()), 107).unwrap();
    simulation::restrict_prefix_wait(&mut late, &offer).unwrap();
    let (_, late_capture) = simulation::advance_prefix(
        &snapshot,
        &late,
        PathOffer::Rendezvous(&offer),
        PrefixMaintenanceStage::Capture,
        Some(8),
        &maintenance,
        23,
        &mut || Ok(()),
        true,
        &protection,
    )
    .unwrap();
    assert!(matches!(
        simulation::advance_prefix(
            &snapshot,
            &late_capture,
            PathOffer::Rendezvous(&offer),
            PrefixMaintenanceStage::Restore,
            Some(8),
            &maintenance,
            23,
            &mut || Ok(()),
            true,
            &protection,
        ),
        Err(simulation::SimulationFailure::SequenceViolation)
    ));
    assert_eq!(snapshot, original);
}

#[test]
fn prefix_stale_offer_and_original_hold_deadline_are_rejected() {
    let (snapshot, mut offer) = setup();
    offer.target.incarnation += 1;
    assert!(matches!(
        compare(
            &snapshot,
            &offer,
            &Fixture::default(),
            &Maintenance::default()
        ),
        PrefixRendezvousDecision::Unknown {
            reason: PlanningUnknownReason::InvalidSnapshot,
            ..
        }
    ));
    let (snapshot, mut offer) = setup();
    offer.expires_at_ns = 118; // source reaches boundary at 118; capture still needs time.
    assert!(matches!(
        compare(
            &snapshot,
            &offer,
            &Fixture::default(),
            &Maintenance::default()
        ),
        PrefixRendezvousDecision::Unknown { .. }
    ));
}
