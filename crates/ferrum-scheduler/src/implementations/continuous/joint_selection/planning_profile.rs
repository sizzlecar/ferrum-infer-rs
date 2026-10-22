//! Ignored CPU diagnostic of the real dispatch trajectory and evaluator.
//! The low-cost Known provider isolates planner work; it is neither a learned
//! model nor a replay of the R10 workload. No speed or completion ratio is a gate.

use super::*;
use ferrum_execution::{execution_attempt::AttemptWorkKind, model_executor::PrefillShapePolicy};
use ferrum_types::{RequestId, SchedulerSloConfig, SloStrategy};
use serde::Serialize;
use std::sync::atomic::{AtomicUsize, Ordering};

#[derive(Clone, Copy, Serialize)]
struct ProfilePlan {
    iterations: usize,
    residents: usize,
    initial_prefillers: usize,
    prompt_tokens: usize,
    output_cap: usize,
    max_tokens: usize,
    prefill_step: usize,
}

const PROFILE: ProfilePlan = ProfilePlan {
    // Usually provides a seconds-long sampling opportunity. This is a work
    // bound, not a guaranteed duration; there is no sleep or adaptive refill.
    iterations: 1024,
    residents: 16,
    initial_prefillers: 8,
    prompt_tokens: 256,
    output_cap: 64,
    max_tokens: 512,
    prefill_step: 128,
};

struct CheapKnownCosts {
    observed_at: Instant,
    calls: AtomicUsize,
}

impl CostSnapshot for CheapKnownCosts {
    fn revision(&self) -> u64 {
        1
    }

    fn query_limits(&self) -> ferrum_types::CostQueryLimits {
        Default::default()
    }

    fn predict(&self, _: &CostQuery<'_>, _: Instant) -> CostPrediction {
        panic!("scenario evaluation must not call the scalar predictor")
    }

    fn cost_scenarios(
        &self,
        _: &CostQuery<'_>,
        _: Instant,
        count: usize,
    ) -> CostScenarioPrediction {
        self.calls.fetch_add(1, Ordering::Relaxed);
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
            samples: 32,
            observed_at: self.observed_at,
            expires_at: self.observed_at + Duration::from_secs(60),
        }
    }
}

fn fixture() -> (
    SchedulerSloConfig,
    Vec<SimulatedRequest>,
    CohortTrajectory,
    Vec<FirstAction>,
) {
    let mut config: SchedulerSloConfig = "ttft:15684.62,tpot:110.56,itl:196.55".parse().unwrap();
    config.strategy = SloStrategy::Joint;
    config.planning.max_elapsed_us = 10_000;
    config.planning.max_queries = 4096;
    config.planning.max_scenarios = 32;
    config.validate().unwrap();
    let requests = (0..PROFILE.residents)
        .map(|index| {
            let decoder = index >= PROFILE.initial_prefillers;
            let clock = if decoder {
                RequestClock {
                    first_output: Some(Duration::ZERO),
                    last_output: Some(Duration::ZERO),
                    ..Default::default()
                }
            } else {
                RequestClock::default()
            };
            SimulatedRequest {
                id: RequestId::new(),
                prompt_tokens: PROFILE.prompt_tokens,
                prefill_tokens: if decoder { PROFILE.prompt_tokens } else { 0 },
                output_tokens: usize::from(decoder),
                output_cap: PROFILE.output_cap,
                scenario_end_tokens: PROFILE.output_cap,
                // One declared maximum planning delay, common to all actions.
                age: Duration::from_micros(config.planning.max_elapsed_us),
                central_clock: clock,
                risk_clock: clock,
            }
        })
        .collect::<Vec<_>>();
    let mut adapter = CohortTrajectory {
        first_wave: Vec::new(),
        cap_commit_targets: Vec::new(),
        shapes: vec![
            RequestShape {
                restored_tokens: 0,
                execution_ceiling: None,
                phase_cap: None,
                decode_only_allowed: true,
            };
            requests.len()
        ],
        admission: AdmissionSlotPlan::new(
            &vec![InitialAdmission::Resident; requests.len()],
            PROFILE.residents,
            config.planning.max_participants,
        )
        .unwrap(),
        execution: CostExecution::SequentialPrefillThenDecode,
        shape_policy: PrefillShapePolicy::Identity,
        max_batch: PROFILE.residents,
        max_tokens: PROFILE.max_tokens,
        prefill_step_chunk: Some(PROFILE.prefill_step),
        active_decode_prefill_chunk: Some(256),
        active_decode_prefill_budget: None,
        decode_pressure_threshold: PROFILE.residents,
        default_active_chunk: 128,
        chunking: true,
    };
    adapter.first_wave = adapter.natural_members(&requests).unwrap();
    let largest = adapter
        .first_wave
        .iter()
        .filter(|member| member.work.kind == AttemptWorkKind::Prefill)
        .map(|member| member.work.requested.as_range().len())
        .max()
        .unwrap();
    let half = NonZeroUsize::new(largest.div_ceil(2)).unwrap();
    let quarter = NonZeroUsize::new(largest.div_ceil(4)).unwrap();
    assert!(quarter < half);
    assert!(adapter.allows_decode_only(&adapter.first_wave));
    assert!(adapter.allows_prefill_phase(&requests, half));
    assert!(adapter.allows_prefill_phase(&requests, quarter));
    // These are the current legal dispatch alternatives for this fixed actual
    // first wave. Admission precommit and unimplemented actions are excluded.
    let alternatives = vec![
        FirstAction::PrefillLimit(half),
        FirstAction::PrefillLimit(quarter),
        FirstAction::DecodeOnly,
        FirstAction::PrefillPhaseLimit(half),
        FirstAction::PrefillPhaseLimit(quarter),
    ];
    (config, requests, adapter, alternatives)
}

#[derive(Serialize)]
struct TraversalReceipt {
    ranks: usize,
    resolved_mass: f64,
    unresolved_mass: f64,
    unresolved_ranks: usize,
    partial_static_ranks: usize,
    queries: usize,
    cache_hits: usize,
    cache_retained_bytes: usize,
    valid_for: Duration,
}

#[derive(Serialize)]
struct ComparisonReceipt {
    iteration: usize,
    // Includes accumulator creation, complete scenario evaluation and finish;
    // excludes fixture creation, serialization and printing.
    comparison_elapsed_us: u64,
    original_provider_calls: usize,
    diagnostics: ScenarioDiagnostics,
    traversal: Option<TraversalReceipt>,
    evaluator_error: Option<Unsupported>,
    aggregation_error: Option<Unsupported>,
    choice: Option<PolicyChoice>,
    // Mirrors only the final outer elapsed check, not Engine admission/apply.
    post_finish_time_rejected: bool,
}

fn compare_once(
    iteration: usize,
    config: &SchedulerSloConfig,
    requests: &[SimulatedRequest],
    adapter: &CohortTrajectory,
    alternatives: &[FirstAction],
) -> ComparisonReceipt {
    let now = Instant::now();
    let snapshot = CheapKnownCosts {
        observed_at: now,
        calls: AtomicUsize::new(0),
    };
    let mut aggregate = ScenarioRiskAccumulator::new(
        requests,
        alternatives.len(),
        config,
        &vec![
            Some(ProbabilityInterval {
                lower: 1.0,
                upper: 1.0
            });
            requests.len()
        ],
    )
    .unwrap();
    let mut diagnostics = ScenarioDiagnostics::default();
    let mut aggregation_error = None;
    let summary = SloPolicy::evaluate_scenarios_with_diagnostics(
        EvaluationInput {
            snapshot: &snapshot,
            now,
            config,
            execution: CostExecution::SequentialPrefillThenDecode,
            requests,
            // Explicit fixture assumptions, not measured physical headroom or
            // stopping/completion qualification. No actual resource is owned.
            resources: ResourceSupport {
                waiting_requests: 0,
                kv_reserved_through_scenario_end: true,
            },
            adapter,
            static_action: &FirstAction::Static,
            alternatives,
            horizon: EvaluationHorizon::ScenarioCompletion,
        },
        |_, _| Ok(()), // fixed, explicit cap64 endpoints in every rank
        |rank, comparison| {
            if aggregation_error.is_none() {
                aggregation_error = aggregate.observe(rank, comparison).err();
            }
        },
        &mut diagnostics,
    );
    let evaluator_error = summary.as_ref().err().copied();
    // An incomplete traversal is never fed to finish or relabeled a decision.
    let decision = if aggregation_error.is_none() {
        summary
            .as_ref()
            .ok()
            .map(|summary| aggregate.finish(*summary, &vec![None; alternatives.len()]))
    } else {
        None
    };
    let choice = match decision {
        Some(Ok(decision)) => Some(decision.choice),
        Some(Err(reason)) => {
            aggregation_error = Some(reason);
            None
        }
        None => None,
    };
    let elapsed = now.elapsed();
    let original_provider_calls = snapshot.calls.load(Ordering::Relaxed);
    assert_eq!(original_provider_calls as u64, diagnostics.provider_calls);
    ComparisonReceipt {
        iteration,
        comparison_elapsed_us: u64::try_from(elapsed.as_micros()).unwrap(),
        original_provider_calls,
        diagnostics,
        traversal: summary.ok().map(|summary| TraversalReceipt {
            ranks: summary.ranks,
            resolved_mass: summary.resolved_mass,
            unresolved_mass: summary.unresolved_mass,
            unresolved_ranks: summary.unresolved_ranks,
            partial_static_ranks: summary.partial_static_ranks,
            queries: summary.queries,
            cache_hits: summary.cache_hits,
            cache_retained_bytes: summary.cache_retained_bytes,
            valid_for: summary.valid_until.saturating_duration_since(now),
        }),
        evaluator_error,
        aggregation_error,
        choice,
        post_finish_time_rejected: choice.is_some()
            && elapsed > Duration::from_micros(config.planning.max_elapsed_us),
    }
}

#[test]
#[ignore = "CPU diagnostic: no completion-rate or speed assertion; sample only by explicit request"]
fn profile_dispatch_split_cohort_with_production_budget() {
    let (config, requests, adapter, alternatives) = fixture();
    let mut receipts = Vec::with_capacity(PROFILE.iterations);
    eprintln!(
        "JOINT_PLANNING_PROFILE_START {}",
        serde_json::json!({
            "pid": std::process::id(),
            "plan": PROFILE,
            "planning_budget": config.planning,
            "alternatives": alternatives.iter().map(|a| format!("{a:?}")).collect::<Vec<_>>(),
            "scope": "dispatch core with Known fixture provider; not R10 replay or Engine apply"
        })
    );
    let loop_started = Instant::now();
    for iteration in 0..PROFILE.iterations {
        receipts.push(compare_once(
            iteration,
            &config,
            &requests,
            &adapter,
            &alternatives,
        ));
    }
    let loop_elapsed = loop_started.elapsed();
    // Formatting/I/O happen after all timed comparisons and their sampling
    // opportunity. This bounded Vec is diagnostic storage, not planner RSS.
    println!(
        "JOINT_PLANNING_PROFILE_END {}",
        serde_json::json!({
            "pid": std::process::id(),
            "iterations": receipts.len(),
            "loop_elapsed": loop_elapsed,
            "returned_traversals": receipts.iter().filter(|r| r.traversal.is_some()).count(),
            "evaluator_failures": receipts.iter().filter(|r| r.evaluator_error.is_some()).count(),
            "aggregation_failures": receipts.iter().filter(|r| r.aggregation_error.is_some()).count(),
            "decisions_computed": receipts.iter().filter(|r| r.choice.is_some()).count(),
            "post_finish_time_rejected": receipts.iter().filter(|r| r.post_finish_time_rejected).count()
        })
    );
    for receipt in receipts {
        println!(
            "JOINT_PLANNING_PROFILE_COMPARISON {}",
            serde_json::to_string(&receipt).unwrap()
        );
    }
}
