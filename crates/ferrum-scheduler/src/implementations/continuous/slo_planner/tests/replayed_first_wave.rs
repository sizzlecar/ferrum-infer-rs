//! Delivery provenance and fail-closed binding. These synthetic callbacks test
//! the planner dataflow; the engine Metal fixture checks real native parity.
use super::*;
use std::cell::{Cell, RefCell};

#[derive(Default)]
struct Context {
    begins: Cell<usize>,
    projected: RefCell<Vec<(usize, usize)>>,
    change_final_route: bool,
    expire_final: bool,
    now: Cell<u64>,
}
struct State<'a> {
    source: &'a Context,
    snapshot: &'a SchedulerSnapshot,
    epoch: usize,
}
impl PlanningExecutionContext for Context {
    fn begin<'a>(
        &'a self,
        snapshot: &'a SchedulerSnapshot,
        poll: &mut dyn FnMut() -> Result<(), PlanningUnknownReason>,
    ) -> Result<Arc<dyn PlanningExecutionState<'a> + 'a>, PlanningUnknownReason> {
        poll()?;
        let epoch = self.begins.get() + 1;
        self.begins.set(epoch);
        Ok(Arc::new(State {
            source: self,
            snapshot,
            epoch,
        }))
    }
}
impl<'a> PlanningExecutionState<'a> for State<'a> {
    fn project(
        &self,
        input: &PlanningExecutionInput<'_>,
        poll: &mut dyn FnMut() -> Result<(), PlanningUnknownReason>,
    ) -> Result<Option<ProjectedExecution<'a>>, PlanningUnknownReason> {
        let mut canonical = TestResolver
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
        if self.epoch > 1 && self.source.change_final_route {
            canonical.provider_signature[0] ^= 1;
        }
        self.source
            .projected
            .borrow_mut()
            .push((self.epoch, canonical.rows.as_ptr() as usize));
        if self.epoch > 1 && self.source.expire_final {
            self.source.now.set(10_000_000);
        }
        Ok(Some(ProjectedExecution {
            host_content_forecasts: None,
            statistical_evidence: None,
            ordered_work: input.work.to_vec(),
            canonical_domain: PlanningShapeDomain::Exact(canonical),
            successor: Arc::new(State {
                source: self.source,
                snapshot: self.snapshot,
                epoch: self.epoch,
            }),
        }))
    }
}

fn select(source: &Context, snapshot: &SchedulerSnapshot) -> PlanningDecision {
    let mut planner = planner(1);
    planner.settings.search.candidate_limit = nz(1);
    planner.propose_with_execution(
        snapshot,
        &Model(|_: &WaveExecutionShape| Some(4)),
        source,
        &mut Clock(100),
    )
}

#[test]
fn delivered_first_canonical_is_the_independent_replay_value_without_another_projection() {
    let snapshot = snapshot(vec![decode(1)]);
    let source = Context::default();
    let (selected, _, _) = feasible(select(&source, &snapshot));
    let callbacks = source.projected.borrow().clone();
    assert_eq!(source.begins.get(), 2, "search then independent replay");
    assert_eq!(callbacks.len(), 2);
    let replayed = selected.replayed_first_wave(&snapshot).unwrap();
    // The Vec allocation is moved from the callback into an Arc. Its address
    // only proves local ownership transfer; it grants no execution authority.
    assert_eq!(replayed.rows.as_ptr() as usize, callbacks[1].1);
    assert_ne!(replayed.rows.as_ptr() as usize, callbacks[0].1);
    assert_eq!(
        canonical_cost_shape(replayed).unwrap(),
        *selected.candidate.execution_shape.exact().unwrap()
    );
    assert!(selected.replayed_first_wave(&snapshot).is_some());
    assert_eq!(*source.projected.borrow(), callbacks);
}

#[test]
fn replay_delivery_rejects_changed_work_epoch_model_shape_and_missing_proof() {
    let original = snapshot(vec![decode(1)]);
    let source = Context::default();
    let (selected, _, _) = feasible(select(&source, &original));
    let mut bad = selected.clone();
    bad.candidate.work[0].key.incarnation += 1;
    assert!(bad.replayed_first_wave(&original).is_none());
    let mut bad = selected.clone();
    let PlanningShapeDomain::Exact(shape) = &mut bad.candidate.execution_shape else {
        unreachable!()
    };
    shape.provider_signature[0] ^= 1;
    assert!(bad.replayed_first_wave(&original).is_none());
    for field in 0..3 {
        let mut changed = original.clone();
        match field {
            0 => changed.generation += 1,
            1 => changed.observed_at_ns += 1,
            _ => changed.cost_model_version += 1,
        }
        assert!(selected.replayed_first_wave(&changed).is_none());
    }
    let mut missing = selected;
    missing.final_replay_first_wave = None;
    assert!(missing.replayed_first_wave(&original).is_none());
}

#[test]
fn changed_route_during_final_replay_cannot_publish_the_search_canonical() {
    let snapshot = snapshot(vec![decode(1)]);
    let source = Context {
        change_final_route: true,
        ..Default::default()
    };
    assert!(matches!(
        select(&source, &snapshot),
        PlanningDecision::Unknown { .. }
    ));
    assert_eq!(source.begins.get(), 2);
}

#[test]
fn final_projection_expiry_cannot_deliver_a_saved_first_proof() {
    struct LiveClock<'a>(&'a Cell<u64>);
    impl PlanningClock for LiveClock<'_> {
        fn now_ns(&mut self) -> u64 {
            self.0.get()
        }
    }
    let snapshot = snapshot(vec![decode(1)]);
    let source = Context {
        expire_final: true,
        now: Cell::new(100),
        ..Default::default()
    };
    let mut planner = planner(1);
    planner.settings.search.candidate_limit = nz(1);
    let decision = planner.propose_with_execution(
        &snapshot,
        &Model(|_: &WaveExecutionShape| Some(4)),
        &source,
        &mut LiveClock(&source.now),
    );
    assert!(matches!(
        decision,
        PlanningDecision::Unknown {
            reason: PlanningUnknownReason::ComputeBudgetExhausted,
            ..
        }
    ));
    assert_eq!(source.begins.get(), 2);
}
