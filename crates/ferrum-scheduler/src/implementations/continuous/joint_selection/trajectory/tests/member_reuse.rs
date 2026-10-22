//! Storage-only equivalence against the same planner with fresh member storage
//! on every adapter call. No topology, prediction, or decision is replayed.

use super::*;
use ferrum_types::{CostQueryLimits, SchedulerSloConfig};
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Mutex,
};
use std::time::Instant;

struct Allocating<'a>(&'a CohortTrajectory);

impl TrajectoryAdapter<FirstAction> for Allocating<'_> {
    fn admission_slots(&self) -> Option<&AdmissionSlotPlan> {
        self.0.admission_slots()
    }

    fn continuation_phase(
        &self,
        action: &FirstAction,
        requests: &[SimulatedRequest],
        phase: &mut PhaseBuilder<'_>,
    ) -> Result<(), Unsupported> {
        self.0.continuation_phase(action, requests, phase)
    }

    fn plan_phase(
        &self,
        action: &FirstAction,
        phase: &ContinuationPhase,
        requests: &[SimulatedRequest],
        plan: &mut PlanBuilder,
    ) -> Result<(), Unsupported> {
        let (FirstAction::PrefillPhaseLimit(limit)
        | FirstAction::ReservedPrefillPhaseLimit(limit)
        | FirstAction::PrefillCapCommit(limit)
        | FirstAction::ReservedPrefillCapCommit(limit)
        | FirstAction::DeferredPrefillCapCommit(limit)
        | FirstAction::DeferredReservedPrefillCapCommit(limit)) = action
        else {
            return Err(Unsupported::InvalidPlan);
        };
        let members = self.0.natural_members_from_active(
            requests,
            Some((*limit, phase.prefill_targets(), action.packing_mode())),
            plan.active_mask().ok_or(Unsupported::InvalidState)?,
        )?;
        self.0.push(&members, plan)
    }

    fn plan(
        &self,
        action: Option<&FirstAction>,
        requests: &[SimulatedRequest],
        plan: &mut PlanBuilder,
    ) -> Result<(), Unsupported> {
        self.0.plan_into(action, requests, &mut Vec::new(), plan)
    }
}

#[derive(Clone, Copy)]
enum Boundary {
    Known,
    Outside,
    ChangedLimits,
    PanickedLimits,
    QueryBudget,
}

struct Snapshot {
    at: Instant,
    boundary: Boundary,
    limits: AtomicUsize,
    calls: Mutex<Vec<Vec<CostWork>>>,
}

impl CostSnapshot for Snapshot {
    fn revision(&self) -> u64 {
        1
    }

    fn query_limits(&self) -> CostQueryLimits {
        let call = self.limits.fetch_add(1, Ordering::Relaxed);
        if call >= 2 && matches!(self.boundary, Boundary::PanickedLimits) {
            panic!("later wave query limits");
        }
        let mut limits = CostQueryLimits::default();
        if call >= 2 && matches!(self.boundary, Boundary::ChangedLimits) {
            limits.max_participants = 1;
        }
        limits
    }

    fn predict(&self, query: &CostQuery<'_>, _: Instant) -> CostPrediction {
        self.calls.lock().unwrap().push(query.work.to_vec());
        CostPrediction::Known {
            central_service_time: Duration::from_millis(1),
            upper_service_time: Duration::from_millis(1),
            risk: CostRisk::EngineeringMargin,
            support: CostSupport::ValidatedFeatureDomain,
            samples: 2,
            observed_at: self.at,
            expires_at: self.at + Duration::from_secs(60),
        }
    }

    fn cost_scenarios(
        &self,
        query: &CostQuery<'_>,
        _: Instant,
        count: usize,
    ) -> CostScenarioPrediction {
        let mut calls = self.calls.lock().unwrap();
        calls.push(query.work.to_vec());
        if matches!(self.boundary, Boundary::Outside)
            && query.work.iter().any(|work| {
                work.kind == AttemptWorkKind::Prefill && work.requested.as_range().start >= 4
            })
        {
            return CostScenarioPrediction::Unknown(CostUnknownReason::UnsupportedWithDetail(
                CostUnsupportedDetail::OutsideObservedSupport,
            ));
        }
        CostScenarioPrediction::Known {
            central_service_time: Duration::from_millis(1),
            scenarios: (0..count)
                .map(|rank| CostScenario {
                    rank,
                    weight: 1.0 / count as f64,
                    service_time: Duration::from_micros(1000 + rank as u64),
                })
                .collect(),
            dependence: CostScenarioDependence::SharedResidualRank,
            support: CostSupport::ValidatedFeatureDomain,
            samples: count,
            observed_at: self.at,
            expires_at: self.at + Duration::from_secs(60),
        }
    }
}

fn fixture(waiting: bool, execution: CostExecution) -> (Vec<SimulatedRequest>, CohortTrajectory) {
    let mut requests = vec![request(1, 1), request(9, 0), request(7, 0)];
    requests[0].central_clock.first_output = Some(Duration::ZERO);
    requests[0].central_clock.last_output = Some(Duration::ZERO);
    requests[0].risk_clock = requests[0].central_clock;
    let mut adapter = trajectory(Vec::new(), vec![shape(None); 3]);
    adapter.execution = execution;
    adapter.admission = AdmissionSlotPlan::new(
        &if waiting {
            vec![
                InitialAdmission::Resident,
                InitialAdmission::Waiting { order: 0 },
                InitialAdmission::Waiting { order: 1 },
            ]
        } else {
            vec![InitialAdmission::Resident; 3]
        },
        if waiting { 1 } else { 3 },
        3,
    )
    .unwrap();
    if waiting {
        adapter.cap_commit_targets = (1..3)
            .map(|index| PrefillTarget {
                request: index,
                prompt_tokens: requests[index].prompt_tokens,
            })
            .collect();
    } else {
        adapter.first_wave = adapter.natural_members(&requests).unwrap();
    }
    (requests, adapter)
}

fn run(
    requests: &[SimulatedRequest],
    adapter: &impl TrajectoryAdapter<FirstAction>,
    execution: CostExecution,
    waiting: bool,
    boundary: Boundary,
    at: Instant,
) -> (
    String,
    Vec<String>,
    Vec<Vec<CostWork>>,
    usize,
    ScenarioDiagnostics,
) {
    let snapshot = Snapshot {
        at,
        boundary,
        limits: AtomicUsize::new(0),
        calls: Mutex::new(Vec::new()),
    };
    let mut config: SchedulerSloConfig = "ttft:1000,tpot:100,itl:100".parse().unwrap();
    // Correctness equivalence, not a wall-time performance gate.
    config.planning.max_elapsed_us = 1_000_000;
    config.planning.max_scenarios = 2;
    if matches!(boundary, Boundary::QueryBudget) {
        config.planning.max_queries = 3;
    }
    let limit = NonZeroUsize::new(1).unwrap();
    let alternatives = if waiting {
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
    let mut comparisons = Vec::new();
    let mut diagnostics = ScenarioDiagnostics::default();
    let result = SloPolicy::evaluate_scenarios_with_diagnostics(
        EvaluationInput {
            snapshot: &snapshot,
            now: at,
            config: &config,
            execution,
            requests,
            resources: ResourceSupport {
                waiting_requests: if waiting { 2 } else { 0 },
                kv_reserved_through_scenario_end: true,
            },
            adapter,
            static_action: &FirstAction::Static,
            alternatives: &alternatives,
            horizon: EvaluationHorizon::ScenarioCompletion,
        },
        |rank, requests| {
            for request in requests {
                request.scenario_end_tokens = 3 + rank.index;
            }
            Ok(())
        },
        |rank, comparison| comparisons.push(format!("{} {comparison:?}", rank.index)),
        &mut diagnostics,
    );
    // Actual elapsed remains a production guard, but is not an equality oracle.
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
    (
        format!("{result:?}"),
        comparisons,
        snapshot.calls.into_inner().unwrap(),
        snapshot.limits.load(Ordering::Relaxed),
        diagnostics,
    )
}

#[test]
fn reused_members_preserve_complete_rank_phase_and_slot_traces() {
    for execution in [
        CostExecution::SingleInvocation,
        CostExecution::SequentialPrefillThenDecode,
    ] {
        for waiting in [false, true] {
            let (requests, adapter) = fixture(waiting, execution);
            let at = Instant::now();
            let allocating = run(
                &requests,
                &Allocating(&adapter),
                execution,
                waiting,
                Boundary::Known,
                at,
            );
            let reused = run(&requests, &adapter, execution, waiting, Boundary::Known, at);
            assert_eq!(reused, allocating);
            assert_eq!(reused.1.len(), 2, "both ranks reach complete comparisons");
            assert_eq!(reused.4.rank_completed, 2);
            assert_eq!(reused.4.failure_count, 0);
        }
    }
}

#[test]
fn reused_members_preserve_unknown_and_query_boundaries() {
    for boundary in [Boundary::Outside, Boundary::QueryBudget] {
        for execution in [
            CostExecution::SingleInvocation,
            CostExecution::SequentialPrefillThenDecode,
        ] {
            let (requests, adapter) = fixture(false, execution);
            let at = Instant::now();
            let allocating = run(
                &requests,
                &Allocating(&adapter),
                execution,
                false,
                boundary,
                at,
            );
            let reused = run(&requests, &adapter, execution, false, boundary, at);
            assert_eq!(reused, allocating);
            assert!(
                reused.4.failure_count > 0,
                "the boundary must actually be exercised"
            );
        }
    }
}

fn direct_limits(
    requests: &[SimulatedRequest],
    adapter: &impl TrajectoryAdapter<FirstAction>,
    execution: CostExecution,
    boundary: Boundary,
    at: Instant,
) -> (Unsupported, Vec<Vec<CostWork>>, usize) {
    let snapshot = Snapshot {
        at,
        boundary,
        limits: AtomicUsize::new(0),
        calls: Mutex::new(Vec::new()),
    };
    let mut config: SchedulerSloConfig = "ttft:1000,tpot:100,itl:100".parse().unwrap();
    config.planning.max_elapsed_us = 1_000_000;
    let result = SloPolicy::evaluate(EvaluationInput {
        snapshot: &snapshot,
        now: at,
        config: &config,
        execution,
        requests,
        resources: ResourceSupport {
            waiting_requests: 0,
            kv_reserved_through_scenario_end: true,
        },
        adapter,
        static_action: &FirstAction::Static,
        alternatives: &[],
        horizon: EvaluationHorizon::ScenarioCompletion,
    });
    (
        result.unwrap_err(),
        snapshot.calls.into_inner().unwrap(),
        snapshot.limits.load(Ordering::Relaxed),
    )
}

#[test]
fn reused_members_preserve_per_wave_limit_refresh_and_panic_containment() {
    for boundary in [Boundary::ChangedLimits, Boundary::PanickedLimits] {
        for execution in [
            CostExecution::SingleInvocation,
            CostExecution::SequentialPrefillThenDecode,
        ] {
            let (requests, adapter) = fixture(false, execution);
            let at = Instant::now();
            let allocating =
                direct_limits(&requests, &Allocating(&adapter), execution, boundary, at);
            let reused = direct_limits(&requests, &adapter, execution, boundary, at);
            assert_eq!(reused, allocating);
            assert!(
                reused.2 >= 3,
                "limits are refreshed beyond the initial wave"
            );
            assert_eq!(
                reused.0,
                if matches!(boundary, Boundary::ChangedLimits) {
                    Unsupported::ParticipantBudget
                } else {
                    Unsupported::Cost(CostUnknownReason::UnsupportedDomain)
                }
            );
        }
    }
}
