//! Packing/support counterexample, not a reconstruction of an R11 decision.
//! The provider below declares a typed test domain; it is not a fitted Learner.

use super::*;
use ferrum_types::{CostQueryLimits, SchedulerSloConfig};
use std::sync::Mutex;
use std::time::Instant;

fn fixture() -> (Vec<SimulatedRequest>, CohortTrajectory) {
    // Different already-computed offsets identify owners in actual CostWork.
    // Every resident still has exactly two ordinary 128-token chunks to run.
    let requests = (0..7)
        .map(|index| SimulatedRequest {
            id: RequestId::new(),
            prompt_tokens: index * 128 + 256,
            prefill_tokens: index * 128,
            output_tokens: 0,
            output_cap: 64,
            scenario_end_tokens: 64,
            age: Duration::ZERO,
            central_clock: RequestClock::default(),
            risk_clock: RequestClock::default(),
        })
        .collect::<Vec<_>>();
    let mut adapter = CohortTrajectory {
        first_wave: Vec::new(),
        cap_commit_targets: Vec::new(),
        shapes: vec![shape(None); requests.len()],
        admission: AdmissionSlotPlan::new(
            &vec![InitialAdmission::Resident; requests.len()],
            16,
            16,
        )
        .unwrap(),
        execution: CostExecution::SequentialPrefillThenDecode,
        shape_policy: PrefillShapePolicy::Identity,
        max_batch: 16,
        max_tokens: 512,
        prefill_step_chunk: Some(128),
        active_decode_prefill_chunk: Some(128),
        active_decode_prefill_budget: None,
        decode_pressure_threshold: 16,
        default_active_chunk: 128,
        chunking: true,
    };
    adapter.cap_commit_targets = adapter
        .natural_members(&requests)
        .unwrap()
        .iter()
        .filter(|member| member.work.kind == AttemptWorkKind::Prefill)
        .map(|member| PrefillTarget {
            request: member.request,
            prompt_tokens: requests[member.request].prompt_tokens,
        })
        .collect();
    (requests, adapter)
}

fn geometry(wave: &[GroupMember]) -> Vec<(usize, usize, usize, usize)> {
    wave.iter()
        .map(|member| {
            assert_eq!(member.work.kind, AttemptWorkKind::Prefill);
            assert_eq!(
                member.work.prefix_restore,
                AttemptPrefixRestore::NoneObserved
            );
            let range = member.work.requested.as_range();
            (
                member.request,
                range.start,
                range.len(),
                member.work.context_tokens,
            )
        })
        .collect()
}

fn preview(
    adapter: &CohortTrajectory,
    requests: &[SimulatedRequest],
    cap: usize,
) -> Vec<GroupMember> {
    let cap = NonZeroUsize::new(cap).unwrap();
    assert!(adapter.allows_cap_commit(requests, cap));
    adapter
        .natural_members_with_phase(requests, Some((cap, &adapter.cap_commit_targets)))
        .unwrap()
}

#[test]
fn cap_commit_repacking_expands_prefill_width_without_extending_targets() {
    let (requests, adapter) = fixture();
    let baseline = adapter.natural_members(&requests).unwrap();
    let half = preview(&adapter, &requests, 64);
    let quarter = preview(&adapter, &requests, 32);
    assert_eq!(
        geometry(&baseline),
        vec![
            (0, 0, 128, 256),
            (1, 128, 128, 384),
            (2, 256, 128, 512),
            (3, 384, 128, 640)
        ]
    );
    assert_eq!(
        geometry(&half),
        vec![
            (0, 0, 64, 256),
            (1, 128, 64, 384),
            (2, 256, 64, 512),
            (3, 384, 64, 640),
            (4, 512, 128, 768),
            (5, 640, 128, 896)
        ]
    );
    assert_eq!(
        geometry(&quarter),
        vec![
            (0, 0, 32, 256),
            (1, 128, 32, 384),
            (2, 256, 32, 512),
            (3, 384, 32, 640),
            (4, 512, 128, 768),
            (5, 640, 128, 896),
            (6, 768, 128, 1024)
        ]
    );
    assert_eq!(
        adapter
            .cap_commit_targets
            .iter()
            .map(|target| target.request)
            .collect::<Vec<_>>(),
        vec![0, 1, 2, 3]
    );
    let active = adapter.admission.active_mask(&requests).unwrap();
    assert!(active.iter().all(|active| *active));
    for wave in [&baseline, &half, &quarter] {
        assert!(wave.len() <= adapter.max_batch);
        assert!(wave.iter().all(|member| active[member.request]));
        assert_eq!(
            wave.iter()
                .map(|member| member.work.requested.as_range().len())
                .sum::<usize>(),
            adapter.max_tokens
        );
    }
    assert!(adapter.shapes.iter().all(|shape| shape.phase_cap.is_none()));
}

struct WidthDomain {
    at: Instant,
    prefill_width: usize,
    queries: Mutex<Vec<Vec<CostWork>>>,
}

impl WidthDomain {
    fn new(at: Instant, prefill_width: usize) -> Self {
        Self {
            at,
            prefill_width,
            queries: Mutex::new(Vec::new()),
        }
    }
}

impl CostSnapshot for WidthDomain {
    fn revision(&self) -> u64 {
        1
    }
    fn query_limits(&self) -> CostQueryLimits {
        CostQueryLimits::default()
    }
    fn predict(&self, query: &CostQuery<'_>, _: Instant) -> CostPrediction {
        assert!(query.work_is_exact);
        assert_eq!(query.execution, CostExecution::SingleInvocation);
        assert!(!query.work.is_empty() && query.work.len() <= 16);
        let kind = query.work[0].kind;
        assert!(query.work.iter().all(|work| work.kind == kind));
        self.queries.lock().unwrap().push(query.work.to_vec());
        // Prefill and Decode are separate domains under the actual Split plan.
        // The wider Decode domain must not make static fail for another reason.
        if kind == AttemptWorkKind::Prefill && query.work.len() > self.prefill_width {
            return CostPrediction::Unknown(CostUnknownReason::UnsupportedWithDetail(
                CostUnsupportedDetail::OutsideObservedSupport,
            ));
        }
        CostPrediction::Known {
            central_service_time: Duration::from_millis(1),
            upper_service_time: Duration::from_millis(1),
            risk: CostRisk::EngineeringMargin,
            support: CostSupport::ValidatedFeatureDomain,
            samples: 32,
            observed_at: self.at,
            expires_at: self.at + Duration::from_secs(60),
        }
    }
}

fn evaluate(
    requests: &[SimulatedRequest],
    adapter: &CohortTrajectory,
    snapshot: &WidthDomain,
    initial: FirstAction,
    alternatives: &[FirstAction],
) -> Comparison {
    let mut config: SchedulerSloConfig = "ttft:1000,tpot:50,itl:15".parse().unwrap();
    // Same CPU-only timeout tolerance as the neighboring future-waiter fixture.
    // It is not the service's 10ms profile or a measured speed requirement.
    config.planning.max_elapsed_us = 1_000_000;
    assert_eq!(config.planning.max_queries, 4096);
    SloPolicy::evaluate(EvaluationInput {
        snapshot,
        now: snapshot.at,
        config: &config,
        execution: adapter.execution,
        requests,
        resources: ResourceSupport {
            waiting_requests: 0,
            kv_reserved_through_scenario_end: true,
        },
        adapter,
        static_action: &initial,
        alternatives,
        horizon: EvaluationHorizon::ScenarioCompletion,
    })
    .unwrap()
}

fn assert_endpoints(result: &TrajectoryResult, initial: &[SimulatedRequest]) {
    for (actual, original) in result.requests.iter().zip(initial) {
        assert_eq!(actual.id, original.id);
        assert_eq!(actual.prefill_tokens, original.prompt_tokens);
        assert_eq!(actual.output_tokens, original.scenario_end_tokens);
        assert_eq!(actual.central_clock.interval_count, 63);
        assert!(actual.central_clock.completed.is_some());
    }
    assert_eq!(result.requests.len(), initial.len());
}

#[test]
fn cap_commit_repacking_crosses_only_prefill_support_and_keeps_real_phase_endpoints() {
    let (requests, adapter) = fixture();
    let half = NonZeroUsize::new(64).unwrap();
    let quarter = NonZeroUsize::new(32).unwrap();
    // The support fence comes from the half-packed wave, not a machine/model ID.
    let known_width = preview(&adapter, &requests, half.get()).len();
    let snapshot = WidthDomain::new(Instant::now(), known_width);
    let comparison = evaluate(
        &requests,
        &adapter,
        &snapshot,
        FirstAction::Static,
        &[
            FirstAction::PrefillCapCommit(half),
            FirstAction::PrefillCapCommit(quarter),
        ],
    );
    assert_endpoints(comparison.static_result.complete().unwrap(), &requests);
    assert_endpoints(comparison.alternatives[0].as_ref().unwrap(), &requests);
    assert!(matches!(
        comparison.alternatives[1],
        Err(Unsupported::Cost(CostUnknownReason::UnsupportedWithDetail(
            CostUnsupportedDetail::OutsideObservedSupport
        )))
    ));
    let rejected = snapshot.queries.lock().unwrap().last().unwrap().clone();
    assert_eq!(
        rejected,
        preview(&adapter, &requests, quarter.get())
            .iter()
            .map(|member| member.work)
            .collect::<Vec<_>>()
    );

    // A distinct, wider typed provider is a positive control, not mutation or
    // widening of the prior snapshot. Drive the same real phase to all64 ends.
    let broad = WidthDomain::new(Instant::now(), requests.len());
    let complete = evaluate(
        &requests,
        &adapter,
        &broad,
        FirstAction::PrefillCapCommit(quarter),
        &[],
    );
    assert_endpoints(complete.static_result.complete().unwrap(), &requests);
    let predicted = broad.queries.lock().unwrap().clone();
    assert_eq!(
        predicted[1][0].requested.as_range(),
        32..64,
        "the core invokes plan_phase with the durable cap after the first wave"
    );

    // Counterfactual installed policy uses the ordinary natural planner, with
    // no preview targets. Only the originally captured four caps are changed.
    let (_, mut installed) = fixture();
    for target in &adapter.cap_commit_targets {
        installed.shapes[target.request].phase_cap = Some(quarter);
    }
    installed.cap_commit_targets.clear();
    assert!(installed.shapes[4..]
        .iter()
        .all(|shape| shape.phase_cap.is_none()));
    let installed_costs = WidthDomain::new(Instant::now(), requests.len());
    let installed_result = evaluate(
        &requests,
        &installed,
        &installed_costs,
        FirstAction::Static,
        &[],
    );
    assert_endpoints(
        installed_result.static_result.complete().unwrap(),
        &requests,
    );
    assert_eq!(predicted, *installed_costs.queries.lock().unwrap(),
        "preview and installed natural scheduling must price identical ordered work through completion");
}
