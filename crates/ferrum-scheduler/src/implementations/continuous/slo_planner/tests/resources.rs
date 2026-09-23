use super::*;
use std::cell::{Cell, RefCell};

/// A small shared-pool contract fixture: every selected row retains one unit
/// while the whole wave temporarily needs one more. Allocator layout behavior
/// belongs to executor/resource tests; these tests exercise planner wiring.
struct SharedPool {
    free: u64,
    revoked: Cell<bool>,
    observed_contexts: RefCell<Vec<Vec<u32>>>,
}

impl SharedPool {
    fn new(free: u64) -> Self {
        Self {
            free,
            revoked: Cell::new(false),
            observed_contexts: RefCell::default(),
        }
    }
}

struct Projection<'a> {
    pool: &'a SharedPool,
    remaining: u64,
}

impl PlanningResourceResolver for SharedPool {
    fn begin(
        &self,
        _: &SchedulerSnapshot,
        _: &mut dyn FnMut() -> Result<(), PlanningUnknownReason>,
    ) -> Result<Box<dyn PlanningResourceProjection + '_>, PlanningUnknownReason> {
        if self.revoked.get() {
            return Err(PlanningUnknownReason::UnknownResourceEvidence);
        }
        Ok(Box::new(Projection {
            pool: self,
            remaining: self.free,
        }))
    }
}

impl PlanningResourceProjection for Projection<'_> {
    fn apply(
        &mut self,
        query: &PlanningResourceQuery<'_>,
        _: &mut dyn FnMut() -> Result<(), PlanningUnknownReason>,
    ) -> Result<(), PlanningUnknownReason> {
        assert_eq!(query.requests.len(), query.snapshot.requests.len());
        let contexts: Vec<_> = query
            .wave
            .work
            .iter()
            .map(|work| {
                query
                    .requests
                    .iter()
                    .find(|request| request.key == work.key)
                    .expect("each participant must carry its current frontier")
                    .context_tokens
            })
            .collect();
        self.pool.observed_contexts.borrow_mut().push(contexts);
        let growth = query.wave.work.len() as u64;
        if growth + 1 > self.remaining {
            return Err(PlanningUnknownReason::OutputOrResourceBlocked);
        }
        self.remaining -= growth;
        Ok(())
    }
}

fn propose(snapshot: &SchedulerSnapshot, pool: &SharedPool) -> PlanningDecision {
    planner(4).propose_with_resources(
        snapshot,
        &Model(|_: &WaveExecutionShape| Some(4)),
        &TestResolver,
        pool,
        &mut Clock(100),
    )
}

#[test]
fn exact_resources_replace_unproved_scalar_capacity_without_inventing_values() {
    let mut snapshot = snapshot(vec![decode(1), decode(2)]);
    snapshot.capacity.evidence_known = false;
    snapshot.capacity.available_kv_tokens = 0;
    snapshot.capacity.available_workspace_bytes = 0;
    snapshot.capabilities.workspace_bytes_upper_bound = u64::MAX;
    let pool = SharedPool::new(3);
    feasible(propose(&snapshot, &pool));
    assert_eq!(
        pool.free, 3,
        "a projection cannot consume physical capacity"
    );
    assert!(matches!(
        planner(4).propose(
            &snapshot,
            &Model(|_: &WaveExecutionShape| Some(4)),
            &TestResolver,
            &mut Clock(100),
        ),
        PlanningDecision::Unknown {
            reason: PlanningUnknownReason::UnknownResourceEvidence,
            ..
        }
    ));
}

#[test]
fn persistent_and_temporary_demand_compete_for_the_same_pool() {
    let mut snapshot = snapshot(vec![decode(1), decode(2)]);
    snapshot.capacity.available_kv_tokens = u64::MAX;
    snapshot.capacity.available_workspace_bytes = u64::MAX;
    assert!(matches!(
        propose(&snapshot, &SharedPool::new(2)),
        PlanningDecision::Unknown {
            reason: PlanningUnknownReason::OutputOrResourceBlocked,
            ..
        }
    ));
    feasible(propose(&snapshot, &SharedPool::new(3)));
}

#[test]
fn resource_projection_retains_growth_and_receives_future_frontiers() {
    let mut request = decode(1);
    request.timing.maximum_output_tokens = n32(4);
    request.timing.budgets.itl_ns = n64(15);
    request.timing.budgets.tpot_ns = n64(15);
    let initial_context = request.context_tokens;
    let mut snapshot = snapshot(vec![request]);
    snapshot.scope.horizon_end_ns = 120;
    assert!(matches!(
        propose(&snapshot, &SharedPool::new(2)),
        PlanningDecision::Unknown {
            reason: PlanningUnknownReason::OutputOrResourceBlocked,
            ..
        }
    ));
    let pool = SharedPool::new(4);
    let (_, witness, _) = feasible(propose(&snapshot, &pool));
    assert!(witness.waves > 1);
    assert!(pool
        .observed_contexts
        .borrow()
        .iter()
        .any(|contexts| contexts.as_slice() == [initial_context + 1]));
}

#[test]
fn unknown_real_resources_cannot_fall_back_to_scalar_headroom() {
    let snapshot = snapshot(vec![decode(1)]);
    let pool = SharedPool::new(u64::MAX);
    pool.revoked.set(true);
    assert!(matches!(
        propose(&snapshot, &pool),
        PlanningDecision::Unknown {
            reason: PlanningUnknownReason::UnknownResourceEvidence,
            ..
        }
    ));
}

#[test]
fn successful_resource_projection_is_rechecked_before_returning_a_witness() {
    let snapshot = snapshot(vec![decode(1)]);
    let pool = SharedPool::new(4);
    let model = Model(|_: &WaveExecutionShape| {
        // Revoke the captured allocation evidence after an initially valid
        // resource projection. Its earlier success must not authorize replay.
        pool.revoked.set(true);
        Some(4)
    });
    assert!(matches!(
        planner(4)
            .propose_with_resources(&snapshot, &model, &TestResolver, &pool, &mut Clock(100),),
        PlanningDecision::Unknown {
            reason: PlanningUnknownReason::UnknownResourceEvidence,
            ..
        }
    ));
    assert!(!pool.observed_contexts.borrow().is_empty());
}

struct IgnoredPoll;
impl PlanningResourceProjection for IgnoredPoll {
    fn apply(
        &mut self,
        _: &PlanningResourceQuery<'_>,
        poll: &mut dyn FnMut() -> Result<(), PlanningUnknownReason>,
    ) -> Result<(), PlanningUnknownReason> {
        let _ = poll();
        Ok(())
    }
}

#[test]
fn a_resource_callback_cannot_swallow_a_budget_failure() {
    let snapshot = snapshot(vec![decode(1)]);
    let (selected, _, _) = feasible(planner(1).propose(
        &snapshot,
        &Model(|_: &WaveExecutionShape| Some(4)),
        &TestResolver,
        &mut Clock(100),
    ));
    let mut polls = VecDeque::from([
        Ok(()),
        Err(PlanningUnknownReason::ComputeBudgetExhausted),
        Ok(()),
    ]);
    let result = super::super::resources::apply(
        &mut IgnoredPoll,
        &PlanningResourceQuery {
            snapshot: &snapshot,
            requests: &snapshot.requests,
            wave: &selected.candidate,
        },
        &mut || polls.pop_front().unwrap(),
    );
    assert_eq!(result, Err(PlanningUnknownReason::ComputeBudgetExhausted));
}

#[test]
fn exact_resource_checks_preserve_context_and_output_limits() {
    let mut snapshot = snapshot(vec![decode(1)]);
    let pool = SharedPool::new(4);
    snapshot.capacity.maximum_context_tokens = n32(snapshot.requests[0].context_tokens);
    assert!(matches!(
        propose(&snapshot, &pool),
        PlanningDecision::Unknown {
            reason: PlanningUnknownReason::OutputOrResourceBlocked,
            ..
        }
    ));
    snapshot.capacity.maximum_context_tokens = n32(100);
    snapshot.capacity.available_output_bytes = 0;
    assert!(matches!(
        propose(&snapshot, &pool),
        PlanningDecision::Unknown {
            reason: PlanningUnknownReason::OutputOrResourceBlocked,
            ..
        }
    ));
}

struct SharedClock<'a>(&'a Cell<u64>);
impl PlanningClock for SharedClock<'_> {
    fn now_ns(&mut self) -> u64 {
        self.0.get()
    }
}

struct ExpensiveResources<'a> {
    clock: &'a Cell<u64>,
    over_budget_at: u64,
    slow_begin: bool,
}

impl PlanningResourceResolver for ExpensiveResources<'_> {
    fn begin(
        &self,
        _: &SchedulerSnapshot,
        _: &mut dyn FnMut() -> Result<(), PlanningUnknownReason>,
    ) -> Result<Box<dyn PlanningResourceProjection + '_>, PlanningUnknownReason> {
        if self.slow_begin {
            self.clock.set(self.over_budget_at);
        }
        Ok(Box::new(ExpensiveProjection(self)))
    }
}

struct ExpensiveProjection<'a>(&'a ExpensiveResources<'a>);
impl PlanningResourceProjection for ExpensiveProjection<'_> {
    fn apply(
        &mut self,
        _: &PlanningResourceQuery<'_>,
        _: &mut dyn FnMut() -> Result<(), PlanningUnknownReason>,
    ) -> Result<(), PlanningUnknownReason> {
        self.0.clock.set(self.0.over_budget_at);
        Ok(())
    }
}

#[test]
fn non_polling_resource_callbacks_cannot_hide_planning_time() {
    let snapshot = snapshot(vec![decode(1)]);
    let planner = planner(2);
    for slow_begin in [true, false] {
        let clock = Cell::new(100);
        let resources = ExpensiveResources {
            clock: &clock,
            over_budget_at: 100 + planner.settings.search.max_planning_us.get() * 1000,
            slow_begin,
        };
        assert!(matches!(
            planner.propose_with_resources(
                &snapshot,
                &Model(|_: &WaveExecutionShape| Some(4)),
                &TestResolver,
                &resources,
                &mut SharedClock(&clock),
            ),
            PlanningDecision::Unknown {
                reason: PlanningUnknownReason::ComputeBudgetExhausted,
                ..
            }
        ));
    }
}
