//! Slot membership changes at wave boundaries. The concrete paired path keeps
//! original endpoints, including a prefill whose first output completes cap1.

use super::*;
use std::cell::Cell;

fn preserved_endpoints(
    enabled: bool,
    requests: &[SimulatedRequest],
    concrete: &CohortTrajectory,
    at: Instant,
    mut inspect: impl FnMut(ScenarioRank, &Comparison),
) -> (SemanticReceipt, Option<Unsupported>, usize) {
    let snapshot = Snapshot {
        at,
        boundary: Boundary::Known,
        limits: AtomicUsize::new(0),
        calls: Mutex::new(Vec::new()),
    };
    let adapter = Counted {
        adapter: concrete,
        planned: AtomicUsize::new(0),
        replayed: AtomicUsize::new(0),
        planned_groups: Mutex::new(Vec::new()),
    };
    let mut config: SchedulerSloConfig = "ttft:1000,tpot:100,itl:100".parse().unwrap();
    config.planning.max_elapsed_us = 1_000_000;
    config.planning.max_scenarios = 2;
    config.planning.max_cost_cache_bytes = 2 * 1024 * 1024;
    let mut aggregate = ScenarioRiskAccumulator::new(
        requests,
        0,
        &config,
        &vec![
            Some(ProbabilityInterval {
                lower: 1.0,
                upper: 1.0
            });
            requests.len()
        ],
    )
    .unwrap();
    let mut comparisons = Vec::new();
    let mut diagnostics = ScenarioDiagnostics::default();
    let input = EvaluationInput {
        snapshot: &snapshot,
        now: at,
        config: &config,
        execution: concrete.execution,
        requests,
        resources: ResourceSupport {
            waiting_requests: concrete.admission.waiting_requests(),
            kv_reserved_through_scenario_end: true,
        },
        adapter: &adapter,
        static_action: &FirstAction::Static,
        alternatives: &[],
        horizon: EvaluationHorizon::ScenarioCompletion,
    };
    let visit = |rank: ScenarioRank, comparison: &Comparison| {
        inspect(rank, comparison);
        aggregate.observe(rank, comparison).unwrap();
        comparisons.push(format!("{} {comparison:?}", rank.index));
    };
    let result = if enabled {
        SloPolicy::evaluate_replayable_scenarios_with_diagnostics(
            input,
            |_, _| Ok(()),
            visit,
            &mut diagnostics,
        )
    } else {
        SloPolicy::evaluate_scenarios_with_diagnostics(
            input,
            |_, _| Ok(()),
            visit,
            &mut diagnostics,
        )
    };
    let error = result.as_ref().err().copied();
    let decision = result
        .as_ref()
        .ok()
        .map(|summary| aggregate.finish(*summary, &[]));
    let result = result.map(|mut summary| {
        assert!(summary.cache_retained_bytes <= config.planning.max_cost_cache_bytes);
        summary.cache_retained_bytes = 0;
        summary
    });
    assert!(diagnostics.cache_peak_bytes <= config.planning.max_cost_cache_bytes);
    diagnostics.cache_peak_bytes = 0;
    diagnostics.elapsed_us = 0;
    for failure in [
        &mut diagnostics.first_failure,
        &mut diagnostics.terminal_failure,
    ]
    .into_iter()
    .flatten()
    {
        failure.elapsed_us = 0;
    }
    let planned_groups = adapter.planned_groups.lock().unwrap().clone();
    (
        SemanticReceipt {
            traversal: format!("{result:?}"),
            comparisons,
            decision: format!("{decision:?}"),
            provider_work: snapshot.calls.into_inner().unwrap(),
            planned_groups,
            provider_limit_calls: snapshot.limits.load(Ordering::Relaxed),
            diagnostics,
        },
        error,
        adapter.replayed.load(Ordering::Relaxed),
    )
}

#[test]
fn active_mask_cap1_prefill_releases_only_at_the_next_split_wave() {
    let mut requests = vec![request(1, 0), request(1, 1), request(1, 0)];
    for (request, end) in requests.iter_mut().zip([1, 3, 1]) {
        request.output_cap = end;
        request.scenario_end_tokens = end;
    }
    requests[1].central_clock.first_output = Some(Duration::ZERO);
    requests[1].central_clock.last_output = Some(Duration::ZERO);
    requests[1].risk_clock = requests[1].central_clock;
    let mut adapter = trajectory(Vec::new(), vec![shape(None); 3]);
    adapter.admission = AdmissionSlotPlan::new(
        &[
            InitialAdmission::Resident,
            InitialAdmission::Resident,
            InitialAdmission::Waiting { order: 0 },
        ],
        2,
        3,
    )
    .unwrap();
    let at = Instant::now();
    let inspect = |rank: ScenarioRank, comparison: &Comparison| {
        let result = comparison.static_result.complete().unwrap();
        let step = Duration::from_micros(1000 + rank.index as u64);
        assert!(result.requests.iter().all(SimulatedRequest::finished));
        // The first prefill group completes member0, but member1's already
        // planned decode group still executes before the waiter may start.
        assert_eq!(result.requests[0].central_clock.completed, Some(step));
        assert_eq!(
            result.requests[2].central_clock.first_output,
            Some(step * 3)
        );
        assert_eq!(result.requests[2].central_clock.completed, Some(step * 3));
        assert_eq!(result.requests[1].central_clock.completed, Some(step * 4));
        assert_eq!(result.queries, 4);
        assert_eq!(result.central_elapsed, step * 4);
    };
    let old = preserved_endpoints(false, &requests, &adapter, at, inspect);
    let reused = preserved_endpoints(true, &requests, &adapter, at, inspect);
    assert_eq!(old.1, None);
    assert_eq!(reused.1, None);
    assert_eq!(old.0, reused.0);
    assert_eq!(reused.0.diagnostics.rank_completed, 2);
    assert_eq!(old.2, 0);
    assert!(reused.2 > 0, "exercise the private concrete trace path");
    let groups = reused
        .0
        .planned_groups
        .iter()
        .map(|group| {
            group
                .iter()
                .map(|(index, work)| (*index, work.kind))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    // Seed plans both groups. The following four are rank0's two waves.
    assert_eq!(
        &groups[2..6],
        &[
            vec![(0, AttemptWorkKind::Prefill)],
            vec![(1, AttemptWorkKind::Decode)],
            vec![(2, AttemptWorkKind::Prefill)],
            vec![(1, AttemptWorkKind::Decode)],
        ]
    );
}

#[test]
fn active_mask_reuse_still_rejects_initial_waiter_progress_before_any_query() {
    let (original, adapter) = fixture(true, CostExecution::SequentialPrefillThenDecode);
    for (prefilled, output) in [(1, 0), (original[1].prompt_tokens, 1)] {
        let mut requests = original.clone();
        requests[1].prefill_tokens = prefilled;
        requests[1].output_tokens = output;
        if output > 0 {
            requests[1].central_clock.first_output = Some(Duration::ZERO);
            requests[1].central_clock.last_output = Some(Duration::ZERO);
            requests[1].risk_clock = requests[1].central_clock;
        }
        let at = Instant::now();
        for enabled in [false, true] {
            let (receipt, error, replayed) =
                preserved_endpoints(enabled, &requests, &adapter, at, |_, _| {
                    panic!("invalid initial admission cannot produce a comparison")
                });
            assert_eq!(error, Some(Unsupported::InvalidState));
            assert!(receipt.provider_work.is_empty());
            assert!(receipt.planned_groups.is_empty());
            assert_eq!(receipt.diagnostics.rank_attempts, 0);
            assert_eq!(replayed, 0);
        }
    }
}

/// Deliberately generic: changing an exposed slot plan is not permitted by the
/// private ReplayableTrajectory contract. Public adapters retain per-wave calls.
struct DynamicSlots {
    narrow: AdmissionSlotPlan,
    wide: AdmissionSlotPlan,
    expanded: Cell<bool>,
    topology: CohortTrajectory,
    masks: Mutex<Vec<Vec<bool>>>,
}

impl TrajectoryAdapter<FirstAction> for DynamicSlots {
    fn admission_slots(&self) -> Option<&AdmissionSlotPlan> {
        Some(if self.expanded.get() {
            &self.wide
        } else {
            &self.narrow
        })
    }

    fn plan(
        &self,
        _: Option<&FirstAction>,
        requests: &[SimulatedRequest],
        plan: &mut PlanBuilder,
    ) -> Result<(), Unsupported> {
        self.masks
            .lock()
            .unwrap()
            .push(plan.active_mask().unwrap().to_vec());
        self.topology.plan(None, requests, plan)?;
        self.expanded.set(true);
        Ok(())
    }
}

#[test]
fn generic_adapter_observes_a_changed_slot_plan_without_a_completion() {
    let mut requests = vec![request(1, 1), request(1, 0)];
    requests[0].central_clock.first_output = Some(Duration::ZERO);
    requests[0].central_clock.last_output = Some(Duration::ZERO);
    requests[0].risk_clock = requests[0].central_clock;
    requests[1].output_cap = 1;
    requests[1].scenario_end_tokens = 1;
    let initial = [
        InitialAdmission::Resident,
        InitialAdmission::Waiting { order: 0 },
    ];
    let adapter = DynamicSlots {
        narrow: AdmissionSlotPlan::new(&initial, 1, 2).unwrap(),
        wide: AdmissionSlotPlan::new(&initial, 2, 2).unwrap(),
        expanded: Cell::new(false),
        topology: trajectory(Vec::new(), vec![shape(None); 2]),
        masks: Mutex::new(Vec::new()),
    };
    let snapshot = Snapshot {
        at: Instant::now(),
        boundary: Boundary::Known,
        limits: AtomicUsize::new(0),
        calls: Mutex::new(Vec::new()),
    };
    let mut config: SchedulerSloConfig = "ttft:1000,tpot:100,itl:100".parse().unwrap();
    config.planning.max_elapsed_us = 1_000_000;
    let comparison = SloPolicy::evaluate(EvaluationInput {
        snapshot: &snapshot,
        now: snapshot.at,
        config: &config,
        execution: CostExecution::SequentialPrefillThenDecode,
        requests: &requests,
        resources: ResourceSupport {
            waiting_requests: 1,
            kv_reserved_through_scenario_end: true,
        },
        adapter: &adapter,
        static_action: &FirstAction::Static,
        alternatives: &[],
        horizon: EvaluationHorizon::ScenarioCompletion,
    })
    .unwrap();
    let result = comparison.static_result.complete().unwrap();
    let masks = adapter.masks.lock().unwrap();
    assert_eq!(&masks[..2], &[vec![true, false], vec![true, true]]);
    assert_eq!(
        result.requests[1].central_clock.completed,
        Some(Duration::from_millis(2))
    );
    assert_eq!(
        result.requests[0].central_clock.completed,
        Some(Duration::from_millis(4))
    );
    assert_eq!(result.queries, 4);
    assert!(result.requests.iter().all(SimulatedRequest::finished));
}
