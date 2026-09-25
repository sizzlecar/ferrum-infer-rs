//! Rolling-controller time is distinct from GPU/whole-wave execution cost.
//! These tests use the product default 2 ms reservation and actual model clocks.
use super::*;
use std::cell::RefCell;

const MS: u64 = 1_000_000;
const START: u64 = 100 * MS;

fn rolling_snapshot(itl_ns: u64, outputs: u32) -> SchedulerSnapshot {
    let mut request = decode(1);
    request.timing.first_commit_at_ns = Some(START);
    request.timing.last_commit_at_ns = Some(START);
    request.timing.maximum_output_tokens = n32(outputs);
    request.timing.budgets = PlannerLatencyBudgets {
        ttft_ns: n64(100 * MS),
        tpot_ns: n64(100 * MS),
        itl_ns: n64(itl_ns),
    };
    let mut s = snapshot(vec![request]);
    s.observed_at_ns = START;
    s.scope.horizon_end_ns = START + 100 * MS;
    s
}

struct ExpiringModel {
    valid_until_ns: u64,
    queries: RefCell<Vec<u64>>,
}
impl PlanningCostModel for ExpiringModel {
    fn model_version(&self) -> u64 {
        7
    }
    fn supports_empirical_host_content(&self) -> bool {
        true
    }
    fn predict(
        &self,
        _: &ExecutionFingerprint,
        _: &WaveExecutionShape,
        now_ns: u64,
    ) -> Option<PlanningCost> {
        self.queries.borrow_mut().push(now_ns);
        Some(PlanningCost {
            typical_ns: 4 * MS,
            planning_ns: 4 * MS,
            model_version: 7,
            valid_for_ns: self.valid_until_ns.checked_sub(now_ns)?,
        })
    }
}

fn product_reserve() -> u64 {
    let settings = BoundedPlannerSettings::default();
    settings
        .future_controller_time
        .reserved_ns(&settings.search)
        .unwrap()
}

#[test]
fn default_shared_budget_is_reserved_once_without_multiplying_attempts() {
    let settings = BoundedPlannerSettings::default();
    assert_eq!(product_reserve(), 2 * MS);
    let mut search = settings.search.clone();
    search.max_replan_attempts = nz(search.max_replan_attempts.get() + 1);
    assert_eq!(
        settings.future_controller_time.reserved_ns(&search),
        Ok(product_reserve())
    );
    search.max_planning_us = n64(3_000);
    assert_eq!(
        settings.future_controller_time.reserved_ns(&search),
        Ok(3 * MS)
    );
    search.max_planning_us = n64(u64::MAX);
    assert_eq!(
        settings.future_controller_time.reserved_ns(&search),
        Err(PlanningUnknownReason::ArithmeticOverflow)
    );
}

#[test]
fn default_controller_budget_changes_the_mult_wave_itl_boundary() {
    let model = Model(|_: &WaveExecutionShape| Some(4 * MS));
    let strict = rolling_snapshot(5 * MS, 3);
    // An explicitly ideal test controller fits 4 ms + 4 ms. The actual product
    // policy has a 2 ms gap: second ITL is 6 ms, so there is no common witness.
    assert!(matches!(
        planner(8).propose(&strict, &model, &TestResolver, &mut Clock(START)),
        PlanningDecision::FeasibleWithinHorizon { .. }
    ));
    assert!(matches!(
        BoundedSloPlanner::default().propose(&strict, &model, &TestResolver, &mut Clock(START)),
        PlanningDecision::Unknown { .. }
    ));
    let boundary = rolling_snapshot(6 * MS, 3);
    let (first, witness, _) = feasible(BoundedSloPlanner::default().propose(
        &boundary,
        &model,
        &TestResolver,
        &mut Clock(START),
    ));
    assert_eq!(first.predicted_wall_ns, 4 * MS);
    assert_eq!(witness.completion_at_ns, START + 10 * MS);
}

#[test]
fn subsequent_query_and_independent_replay_share_control_time_and_ttl() {
    let s = rolling_snapshot(100 * MS, 3);
    let context = execution::ReplayContext {
        resolver: &TestResolver,
        resources: None,
    };
    let model = ExpiringModel {
        valid_until_ns: START + 11 * MS,
        queries: RefCell::new(Vec::new()),
    };
    let first_state = simulation::begin_with_controller_time(
        &s,
        &context,
        &mut || Ok(()),
        START,
        product_reserve(),
    )
    .unwrap();
    let work = vec![CandidateWork {
        key: s.requests[0].key.clone(),
        action: WaveAction::Decode,
    }];
    let first = simulation::advance(
        &s,
        &first_state,
        &work,
        &model,
        false,
        &mut || Ok(()),
        true,
        None,
    )
    .unwrap_or_else(|e| panic!("first: {:?}", e.cause));
    let second = simulation::advance(
        &s,
        &first.state,
        &work,
        &model,
        false,
        &mut || Ok(()),
        true,
        None,
    )
    .unwrap_or_else(|e| panic!("second: {:?}", e.cause));
    assert_eq!(*model.queries.borrow(), vec![START, START + 6 * MS]);
    assert_eq!(second.state.now_ns, START + 10 * MS);
    assert_eq!(second.state.minimum_cost_freshness_slack_ns, MS);
    assert_eq!(second.state.first_wave_cost_ns, 4 * MS);
    let waves = vec![first.wave, second.wave];
    let replay = simulation::replay(
        &s,
        &waves,
        &model,
        &context,
        false,
        &mut || Ok(()),
        START,
        product_reserve(),
        true,
        None,
    )
    .unwrap();
    assert_eq!(replay.now_ns, second.state.now_ns);
    assert_eq!(replay.minimum_cost_freshness_slack_ns, MS);
    assert_eq!(
        *model.queries.borrow(),
        vec![START, START + 6 * MS, START, START + 6 * MS]
    );
    let settings = BoundedPlannerSettings::default();
    let (score, _) =
        simulation::score_at(&s, &replay, &settings, START, START + MS, &mut || Ok(())).unwrap();
    // Two outputs / (8 ms physical + 2 ms future controller + 1 ms current CPU).
    assert!((score - 2.0 / (11 * MS) as f64).abs() < 1e-20);

    let expires_during_second = ExpiringModel {
        valid_until_ns: START + 9 * MS,
        queries: RefCell::new(Vec::new()),
    };
    // Execution-only replay completes at 8 ms and is still covered.
    assert!(simulation::replay(
        &s,
        &waves,
        &expires_during_second,
        &context,
        false,
        &mut || Ok(()),
        START,
        0,
        true,
        None,
    )
    .is_ok());
    assert!(matches!(
        simulation::replay(
            &s,
            &waves,
            &expires_during_second,
            &context,
            false,
            &mut || Ok(()),
            START,
            product_reserve(),
            true,
            None,
        ),
        Err(simulation::SimulationFailure::Unknown(
            PlanningUnknownReason::CostUnavailable
        ))
    ));
}

#[test]
fn single_wave_keeps_execution_label_and_current_cpu_is_not_charged_twice() {
    struct CurrentClock<'a>(&'a std::cell::Cell<u64>);
    impl PlanningClock for CurrentClock<'_> {
        fn now_ns(&mut self) -> u64 {
            self.0.get()
        }
    }
    let s = rolling_snapshot(5 * MS, 2);
    let now = std::cell::Cell::new(START);
    let model = Model(|_: &WaveExecutionShape| {
        now.set(START + MS / 2);
        Some(4 * MS)
    });
    let (first, witness, _) = feasible(BoundedSloPlanner::default().propose(
        &s,
        &model,
        &TestResolver,
        &mut CurrentClock(&now),
    ));
    assert_eq!(first.predicted_wall_ns, 4 * MS);
    assert_eq!(witness.completion_at_ns, START + MS / 2 + 4 * MS);
}
