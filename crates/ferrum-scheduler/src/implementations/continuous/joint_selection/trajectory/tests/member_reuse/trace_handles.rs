//! Pair the same concrete geometry with trace replay disabled/enabled. Complete
//! ordered work, per-rank clocks/risk/decisions and failures remain the oracle;
//! test counters establish that the enabled case really consumes a trace.

use super::*;

mod active_mask_reuse;

struct Counted<'a> {
    adapter: &'a CohortTrajectory,
    planned: AtomicUsize,
    replayed: AtomicUsize,
    planned_groups: Mutex<Vec<Vec<(usize, CostWork)>>>,
}

impl Counted<'_> {
    fn record(&self, plan: &PlanBuilder) {
        self.planned_groups
            .lock()
            .unwrap()
            .extend(plan.groups_for_test().map(|group| {
                group
                    .iter()
                    .map(|member| (member.request, member.work))
                    .collect()
            }));
    }
}

impl TrajectoryAdapter<FirstAction> for Counted<'_> {
    fn admission_slots(&self) -> Option<&AdmissionSlotPlan> {
        self.adapter.admission_slots()
    }

    fn continuation_phase(
        &self,
        action: &FirstAction,
        requests: &[SimulatedRequest],
        phase: &mut PhaseBuilder<'_>,
    ) -> Result<(), Unsupported> {
        self.adapter.continuation_phase(action, requests, phase)
    }

    fn plan_phase(
        &self,
        action: &FirstAction,
        phase: &ContinuationPhase,
        requests: &[SimulatedRequest],
        plan: &mut PlanBuilder,
    ) -> Result<(), Unsupported> {
        self.planned.fetch_add(1, Ordering::Relaxed);
        self.adapter.plan_phase(action, phase, requests, plan)?;
        self.record(plan);
        Ok(())
    }

    fn plan(
        &self,
        action: Option<&FirstAction>,
        requests: &[SimulatedRequest],
        plan: &mut PlanBuilder,
    ) -> Result<(), Unsupported> {
        self.planned.fetch_add(1, Ordering::Relaxed);
        self.adapter.plan(action, requests, plan)?;
        self.record(plan);
        Ok(())
    }
}

// Test-only counting wrapper delegates the exact same concrete sealed contract.
impl ReplayableTrajectory<FirstAction> for Counted<'_> {
    fn record_replay(&self, plan: &PlanBuilder) {
        self.replayed.fetch_add(1, Ordering::Relaxed);
        self.record(plan);
    }
}

#[derive(Clone, Copy)]
enum Endpoints {
    Fixed,
    Changed,
    UnresolvedRank,
}

#[derive(Debug, PartialEq)]
struct SemanticReceipt {
    traversal: String,
    comparisons: Vec<String>,
    decision: String,
    provider_work: Vec<Vec<CostWork>>,
    // Every successfully planned group, including those later served from the
    // cost cache. This is geometry, not a claim every planned group executed.
    planned_groups: Vec<Vec<(usize, CostWork)>>,
    provider_limit_calls: usize,
    diagnostics: ScenarioDiagnostics,
}

fn compare(
    enabled: bool,
    requests: &[SimulatedRequest],
    adapter: &CohortTrajectory,
    boundary: Boundary,
    endpoints: Endpoints,
    bytes: usize,
    horizon: EvaluationHorizon,
    at: Instant,
) -> (SemanticReceipt, usize, usize) {
    let snapshot = Snapshot {
        at,
        boundary,
        limits: AtomicUsize::new(0),
        calls: Mutex::new(Vec::new()),
    };
    let adapter = Counted {
        adapter,
        planned: AtomicUsize::new(0),
        replayed: AtomicUsize::new(0),
        planned_groups: Mutex::new(Vec::new()),
    };
    let mut config: SchedulerSloConfig = "ttft:1000,tpot:100,itl:100".parse().unwrap();
    config.planning.max_elapsed_us = 1_000_000;
    config.planning.max_scenarios = 4;
    config.planning.max_cost_cache_bytes = bytes;
    if matches!(boundary, Boundary::QueryBudget) {
        config.planning.max_queries = 3;
    }
    let limit = NonZeroUsize::new(1).unwrap();
    let waiting = adapter.adapter.admission.waiting_requests();
    let alternatives = if waiting > 0 {
        vec![
            FirstAction::DeferredPrefillCapCommit(limit),
            FirstAction::DeferredReservedPrefillCapCommit(limit),
        ]
    } else {
        vec![
            FirstAction::PrefillLimit(limit),
            FirstAction::DecodeOnly,
            FirstAction::PrefillPhaseLimit(limit),
            FirstAction::ReservedPrefillPhaseLimit(limit),
        ]
    };
    let mut aggregate = ScenarioRiskAccumulator::new(
        requests,
        alternatives.len(),
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
    let mut aggregate_error = None;
    let mut comparisons = Vec::new();
    let mut diagnostics = ScenarioDiagnostics::default();
    let input = EvaluationInput {
        snapshot: &snapshot,
        now: at,
        config: &config,
        execution: adapter.adapter.execution,
        requests,
        resources: ResourceSupport {
            waiting_requests: waiting,
            kv_reserved_through_scenario_end: true,
        },
        adapter: &adapter,
        static_action: &FirstAction::Static,
        alternatives: &alternatives,
        horizon,
    };
    let assign = |rank: ScenarioRank, requests: &mut [SimulatedRequest]| {
        if matches!(endpoints, Endpoints::UnresolvedRank) && rank.index == 2 {
            return Err(Unsupported::StoppingCoverage);
        }
        for request in requests {
            request.scenario_end_tokens = if matches!(endpoints, Endpoints::Changed) {
                3 + rank.index / 2
            } else {
                4
            };
        }
        Ok(())
    };
    let visit = |rank: ScenarioRank, comparison: &Comparison| {
        comparisons.push(format!("{} {comparison:?}", rank.index));
        if aggregate_error.is_none() {
            aggregate_error = aggregate.observe(rank, comparison).err();
        }
    };
    let result = if enabled {
        SloPolicy::evaluate_replayable_scenarios_with_diagnostics(
            input,
            assign,
            visit,
            &mut diagnostics,
        )
    } else {
        SloPolicy::evaluate_scenarios_with_diagnostics(input, assign, visit, &mut diagnostics)
    };
    let decision = if aggregate_error.is_none() {
        result.as_ref().ok().map(|summary| {
            format!(
                "{:?}",
                aggregate.finish(*summary, &vec![None; alternatives.len()])
            )
        })
    } else {
        Some(format!("{aggregate_error:?}"))
    };
    let result = result.map(|mut summary| {
        // Retained memo storage intentionally differs; it is bounded and tested
        // separately, not semantic risk or a performance equality assertion.
        assert!(summary.cache_retained_bytes <= bytes);
        summary.cache_retained_bytes = 0;
        summary
    });
    assert!(diagnostics.cache_peak_bytes <= bytes);
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
        adapter.planned.load(Ordering::Relaxed),
        adapter.replayed.load(Ordering::Relaxed),
    )
}

#[test]
fn trace_handles_preserve_concrete_rank_phase_slot_work_and_risk() {
    for execution in [
        CostExecution::SingleInvocation,
        CostExecution::SequentialPrefillThenDecode,
    ] {
        for waiting in [false, true] {
            for endpoints in [
                Endpoints::Fixed,
                Endpoints::Changed,
                Endpoints::UnresolvedRank,
            ] {
                let (requests, adapter) = fixture(waiting, execution);
                let at = Instant::now();
                let old = compare(
                    false,
                    &requests,
                    &adapter,
                    Boundary::Known,
                    endpoints,
                    2 * 1024 * 1024,
                    EvaluationHorizon::ScenarioCompletion,
                    at,
                );
                let replay = compare(
                    true,
                    &requests,
                    &adapter,
                    Boundary::Known,
                    endpoints,
                    2 * 1024 * 1024,
                    EvaluationHorizon::ScenarioCompletion,
                    at,
                );
                assert_eq!(old.0, replay.0);
                assert_eq!(old.2, 0);
                assert!(
                    replay.2 > 0,
                    "the paired route must actually replay geometry"
                );
                assert_eq!(old.1, replay.1 + replay.2);
            }
        }
    }
}

#[test]
fn trace_handles_preserve_support_and_hard_budget_failures() {
    for boundary in [Boundary::Outside, Boundary::QueryBudget] {
        for execution in [
            CostExecution::SingleInvocation,
            CostExecution::SequentialPrefillThenDecode,
        ] {
            let (requests, adapter) = fixture(false, execution);
            let at = Instant::now();
            let old = compare(
                false,
                &requests,
                &adapter,
                boundary,
                Endpoints::Fixed,
                2 * 1024 * 1024,
                EvaluationHorizon::ScenarioCompletion,
                at,
            );
            let replay = compare(
                true,
                &requests,
                &adapter,
                boundary,
                Endpoints::Fixed,
                2 * 1024 * 1024,
                EvaluationHorizon::ScenarioCompletion,
                at,
            );
            assert_eq!(old.0, replay.0);
            assert!(replay.0.diagnostics.failure_count > 0);
            if matches!(boundary, Boundary::QueryBudget) {
                assert!(replay.0.diagnostics.query_budget_exhausted);
                assert!(replay.0.comparisons.is_empty());
            }
        }
    }
}

#[test]
fn trace_handles_fallback_for_no_storage_and_noncompletion_horizon() {
    let (requests, adapter) = fixture(false, CostExecution::SequentialPrefillThenDecode);
    for (bytes, horizon) in [
        (0, EvaluationHorizon::ScenarioCompletion),
        (1, EvaluationHorizon::ScenarioCompletion),
        (2 * 1024 * 1024, EvaluationHorizon::StaticRounds(2)),
    ] {
        let at = Instant::now();
        let old = compare(
            false,
            &requests,
            &adapter,
            Boundary::Known,
            Endpoints::Fixed,
            bytes,
            horizon,
            at,
        );
        let replay = compare(
            true,
            &requests,
            &adapter,
            Boundary::Known,
            Endpoints::Fixed,
            bytes,
            horizon,
            at,
        );
        assert_eq!(old.0, replay.0);
        assert_eq!(replay.2, 0);
        assert_eq!(old.1, replay.1);
    }
}
