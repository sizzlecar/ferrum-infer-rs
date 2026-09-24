use super::{
    super::{cost_model::*, LogicalWorkGeneration},
    *,
};
use ferrum_types::{RequestId, SloPlannerConfig};
use std::{
    collections::VecDeque,
    num::{NonZeroU32, NonZeroU64, NonZeroUsize},
    sync::Arc,
};

mod admission;
mod boundaries;
mod budget_phases;
mod host_domain;
mod joint_execution;
mod lazy_search;
mod ordering;
mod output;
mod recovery;
mod resolver;
mod resources;
use resolver::TestResolver;

fn n32(value: u32) -> NonZeroU32 {
    NonZeroU32::new(value).unwrap()
}
fn n64(value: u64) -> NonZeroU64 {
    NonZeroU64::new(value).unwrap()
}
fn nz(value: usize) -> NonZeroUsize {
    NonZeroUsize::new(value).unwrap()
}

struct Clock(u64);
impl PlanningClock for Clock {
    fn now_ns(&mut self) -> u64 {
        self.0
    }
}

struct ScriptClock {
    times: VecDeque<u64>,
    last: u64,
}
impl ScriptClock {
    fn new(times: &[u64]) -> Self {
        Self {
            times: times.iter().copied().collect(),
            last: 0,
        }
    }
}
impl PlanningClock for ScriptClock {
    fn now_ns(&mut self) -> u64 {
        if let Some(now) = self.times.pop_front() {
            self.last = now;
        }
        self.last
    }
}

struct Model<F>(F);
impl<F: Fn(&WaveExecutionShape) -> Option<u64>> PlanningCostModel for Model<F> {
    fn model_version(&self) -> u64 {
        7
    }
    fn predict(
        &self,
        _: &ExecutionFingerprint,
        shape: &WaveExecutionShape,
        _: u64,
    ) -> Option<PlanningCost> {
        (self.0)(shape).map(|cost| PlanningCost {
            typical_ns: cost,
            planning_ns: cost,
            model_version: 7,
            valid_for_ns: u64::MAX,
        })
    }
}

fn decode(id: u128) -> RequestSchedulingView {
    RequestSchedulingView {
        key: RequestWorkKey {
            request_id: RequestId(uuid::Uuid::from_u128(id)),
            incarnation: id as u64,
            work_generation: LogicalWorkGeneration::default(),
        },
        timing: RequestTimingView {
            ingress_at_ns: 0,
            first_commit_at_ns: Some(90),
            last_commit_at_ns: Some(90),
            committed_tokens: 1,
            maximum_output_tokens: n32(2),
            budgets: PlannerLatencyBudgets {
                ttft_ns: n64(100),
                tpot_ns: n64(100),
                itl_ns: n64(100),
            },
            slo_failed: false,
        },
        phase: RequestPhaseView::Decode,
        readiness: RequestReadiness::Ready,
        context_tokens: id as u32 * 10,
        recurrent_state_bytes: 0,
        output_credit: OutputCreditView {
            available_token_commands: 16,
            byte_backing: OutputByteBacking::Incremental {
                available_bytes: 1024,
                bytes_per_token_upper_bound: Some(n64(32)),
            },
        },
        output_policy_signature: [3; 32],
        recovery_service: RecoveryServiceDebt::new(std::num::NonZeroUsize::new(256).unwrap()),
        fairness_rank: id as u64,
        ranking_service_cost_ns: None,
        optimistic_next_service: None,
    }
}

fn prefill(id: u128) -> RequestSchedulingView {
    let mut request = decode(id);
    request.context_tokens = 0;
    request.timing.first_commit_at_ns = None;
    request.timing.last_commit_at_ns = None;
    request.timing.committed_tokens = 0;
    request.timing.maximum_output_tokens = n32(1);
    request.timing.budgets.ttft_ns = n64(200);
    request.phase = RequestPhaseView::Prefill(PrefillProgressView {
        admitted_at_ns: 0,
        reference_work_at_admission_ns: 0,
        offset: 0,
        total_prompt_tokens: n32(8),
        logical_high_water: 0,
        executable_until: 8,
        reference: Arc::new(PrefillReferenceWork {
            evaluation: Default::default(),
            version: 1,
            points: vec![
                ReferenceWorkPoint {
                    prompt_tokens: 0,
                    cumulative_work_ns: 0,
                },
                ReferenceWorkPoint {
                    prompt_tokens: 4,
                    cumulative_work_ns: 40,
                },
                ReferenceWorkPoint {
                    prompt_tokens: 8,
                    cumulative_work_ns: 100,
                },
            ],
        }),
        milestones: Arc::from([]),
    });
    request
}

fn snapshot(requests: Vec<RequestSchedulingView>) -> SchedulerSnapshot {
    SchedulerSnapshot {
        observed_at_ns: 100,
        generation: 9,
        cost_model_version: 7,
        fingerprint: ExecutionFingerprint {
            model_weights: [1; 32],
            numerical_policy: [2; 32],
            device_runtime: [3; 32],
            execution_config: [4; 32],
        },
        requests,
        capabilities: BackendPlanningCapabilities {
            path: WaveExecutionPath::NativeUnified,
            graph_state: WaveGraphState::Warm,
            order: BatchOrderSemantics::Ordered,
            decode_batch_sizes: vec![nz(1), nz(2)],
            prefill_batch_sizes: vec![nz(1)],
            prefill_chunk_sizes: vec![n32(4), n32(8)],
            prefill_alignment: n32(4),
            allow_final_short_chunk: true,
            native_mixed: true,
            max_wave_rows: nz(8),
            max_prefill_tokens_per_wave: n64(32),
            workspace_bytes_upper_bound: 64,
        },
        capacity: CapacityReadView {
            evidence_known: true,
            available_kv_tokens: 1024,
            maximum_context_tokens: n32(4096),
            available_workspace_bytes: 1024,
            available_output_bytes: 4096,
        },
        scope: PlanningScope {
            horizon_end_ns: 180,
            reference_decode_token_ns: n64(10),
            reference_work_version: 1,
        },
        has_unmodeled_maintenance: false,
    }
}

fn planner(depth: usize) -> BoundedSloPlanner {
    BoundedSloPlanner {
        settings: BoundedPlannerSettings {
            search: SloPlannerConfig {
                lookahead_waves: nz(depth),
                beam_width: nz(16),
                ..Default::default()
            },
        },
    }
}

fn feasible(
    decision: PlanningDecision,
) -> (SelectedWave, PlanningWitnessSummary, PlanningSearchStats) {
    match decision {
        PlanningDecision::FeasibleWithinHorizon {
            first_wave,
            witness,
            search,
        } => (first_wave, witness, search),
        other => panic!("expected a common finite witness, got {other:?}"),
    }
}

#[test]
fn deadlines_use_current_token_count_and_both_itl_and_tpot() {
    let mut request = decode(1);
    request.timing.first_commit_at_ns = Some(960);
    request.timing.last_commit_at_ns = Some(992);
    request.timing.committed_tokens = 3;
    request.timing.budgets.tpot_ns = n64(18);
    request.timing.budgets.itl_ns = n64(20);
    assert_eq!(request.timing.next_deadline_ns(), Some(1012));
    request.timing.budgets.itl_ns = n64(100);
    assert_eq!(request.timing.next_deadline_ns(), Some(1014));
    request.timing.committed_tokens = u32::MAX;
    request.timing.budgets.tpot_ns = n64(u64::MAX);
    assert_eq!(request.timing.next_deadline_ns(), None);
}

#[test]
fn proposal_preserves_identity_generation_and_does_not_mutate_snapshot() {
    let snapshot = snapshot(vec![decode(1), decode(2)]);
    let before = snapshot.clone();
    let (first, witness, _) = feasible(planner(2).propose(
        &snapshot,
        &Model(|_: &WaveExecutionShape| Some(5)),
        &TestResolver,
        &mut Clock(100),
    ));
    assert_eq!(snapshot, before);
    assert_eq!(first.snapshot_generation, 9);
    assert_eq!(first.cost_model_version, 7);
    assert_eq!(first.candidate.based_on_generation, 9);
    assert_eq!(first.candidate.work.len(), 2);
    assert_eq!(witness.predicted_output_tokens, 2);
    assert_eq!(witness.requests_with_obligations_beyond_horizon, 0);
}

#[test]
fn independently_rescuable_requests_do_not_imply_a_common_sequence() {
    let mut a = decode(1);
    let mut b = decode(2);
    a.timing.budgets.tpot_ns = n64(22); // deadline 112
    b.timing.budgets.tpot_ns = n64(25); // deadline 115
    let mut snapshot = snapshot(vec![a, b]);
    snapshot.capabilities.decode_batch_sizes = vec![nz(1)];
    let model = Model(|shape: &WaveExecutionShape| {
        Some(if shape.decode_kv_tokens[0] == 10 {
            10
        } else {
            8
        })
    });
    assert!(matches!(
        planner(2).propose(&snapshot, &model, &TestResolver, &mut Clock(100)),
        PlanningDecision::Unknown { .. }
    ));
    let (_, witness, _) = feasible(planner(2).propose(
        &snapshot,
        &Model(|_: &WaveExecutionShape| Some(4)),
        &TestResolver,
        &mut Clock(100),
    ));
    assert_eq!(witness.waves, 2);
    assert_eq!(witness.predicted_output_tokens, 2);
}

#[test]
fn horizon_must_cover_actual_service_of_every_initial_decoder() {
    let mut snapshot = snapshot(vec![decode(1), decode(2)]);
    snapshot.capabilities.decode_batch_sizes = vec![nz(1)];
    snapshot.scope.horizon_end_ns = 105;
    assert!(matches!(
        planner(1).propose(
            &snapshot,
            &Model(|_: &WaveExecutionShape| Some(1)),
            &TestResolver,
            &mut Clock(100)
        ),
        PlanningDecision::Unknown { .. }
    ));
    let (_, witness, _) = feasible(planner(2).propose(
        &snapshot,
        &Model(|_: &WaveExecutionShape| Some(1)),
        &TestResolver,
        &mut Clock(100),
    ));
    assert_eq!(witness.predicted_output_tokens, 2);
}

#[test]
fn first_token_creates_decode_obligations_inside_the_horizon() {
    let mut request = prefill(1);
    request.timing.maximum_output_tokens = n32(3);
    request.timing.budgets.itl_ns = n64(10);
    request.timing.budgets.tpot_ns = n64(10);
    let mut snapshot = snapshot(vec![request]);
    snapshot.scope.horizon_end_ns = 120;
    snapshot.capabilities.prefill_chunk_sizes = vec![n32(8)];
    let model = Model(|_: &WaveExecutionShape| Some(5));
    assert!(matches!(
        planner(1).propose(&snapshot, &model, &TestResolver, &mut Clock(100)),
        PlanningDecision::Unknown { .. }
    ));
    let (_, witness, _) =
        feasible(planner(3).propose(&snapshot, &model, &TestResolver, &mut Clock(100)));
    assert_eq!(witness.predicted_output_tokens, 3);
    assert_eq!(witness.waves, 3);
}

#[test]
fn split_backend_never_invents_native_mixed_efficiency() {
    let mut snapshot = snapshot(vec![decode(1), prefill(2)]);
    snapshot.capabilities.native_mixed = false;
    snapshot.capabilities.path = WaveExecutionPath::LegacySplit;
    snapshot.scope.horizon_end_ns = 200;
    let model = Model(|shape: &WaveExecutionShape| {
        assert_ne!(shape.kind, WaveKind::Mixed);
        Some(5)
    });
    let (_, witness, _) =
        feasible(planner(2).propose(&snapshot, &model, &TestResolver, &mut Clock(100)));
    assert_eq!(witness.waves, 2);
}

#[test]
fn legal_alignment_and_ceiling_are_respected_before_prediction() {
    let mut request = prefill(1);
    if let RequestPhaseView::Prefill(progress) = &mut request.phase {
        progress.executable_until = 4;
    }
    let snapshot = snapshot(vec![request]);
    let candidates = candidates::enumerate(
        &snapshot,
        &snapshot.requests,
        100,
        16,
        None,
        &TestResolver,
        &mut || Ok(()),
    )
    .unwrap();
    assert_eq!(candidates.waves.len(), 1);
    assert!(
        matches!(candidates.waves[0].work[0].action, WaveAction::Prefill { offset: 0, count } if count.get() == 4)
    );
    let mut invalid = candidates.waves[0].clone();
    invalid.work[0].action = WaveAction::Prefill {
        offset: 4,
        count: n32(4),
    };
    assert!(shape::resolve(
        &snapshot,
        &snapshot.requests,
        &invalid.work,
        &TestResolver,
        &mut || Ok(())
    )
    .unwrap()
    .is_none());
}

#[test]
fn output_credits_never_recover_without_observed_releases() {
    let mut request = decode(1);
    request.timing.maximum_output_tokens = n32(4);
    request.timing.budgets.itl_ns = n64(15);
    request.timing.budgets.tpot_ns = n64(15);
    request.output_credit.available_token_commands = 1;
    let mut snapshot = snapshot(vec![request]);
    snapshot.scope.horizon_end_ns = 120;
    assert!(matches!(
        planner(4).propose(
            &snapshot,
            &Model(|_: &WaveExecutionShape| Some(4)),
            &TestResolver,
            &mut Clock(100)
        ),
        PlanningDecision::Unknown {
            reason: PlanningUnknownReason::OutputOrResourceBlocked,
            ..
        }
    ));
    snapshot.requests[0].output_credit.available_token_commands = 3;
    let (_, witness, _) = feasible(planner(4).propose(
        &snapshot,
        &Model(|_: &WaveExecutionShape| Some(4)),
        &TestResolver,
        &mut Clock(100),
    ));
    assert!(witness.predicted_output_tokens >= 2);
}

#[test]
fn context_growth_and_global_output_bytes_remain_conservative() {
    let mut snapshot = snapshot(vec![decode(1), decode(2)]);
    snapshot.capacity.available_kv_tokens = 1;
    assert!(matches!(
        planner(2).propose(
            &snapshot,
            &Model(|_: &WaveExecutionShape| Some(5)),
            &TestResolver,
            &mut Clock(100)
        ),
        PlanningDecision::Unknown {
            reason: PlanningUnknownReason::OutputOrResourceBlocked,
            ..
        }
    ));
    snapshot.capacity.available_kv_tokens = 2;
    snapshot.capacity.available_output_bytes = 63;
    assert!(matches!(
        planner(2).propose(
            &snapshot,
            &Model(|_: &WaveExecutionShape| Some(5)),
            &TestResolver,
            &mut Clock(100)
        ),
        PlanningDecision::Unknown {
            reason: PlanningUnknownReason::OutputOrResourceBlocked,
            ..
        }
    ));
    snapshot.capacity.available_output_bytes = 64;
    feasible(planner(2).propose(
        &snapshot,
        &Model(|_: &WaveExecutionShape| Some(5)),
        &TestResolver,
        &mut Clock(100),
    ));
}

#[test]
fn blocked_and_unknown_requests_keep_their_obligations() {
    for readiness in [
        RequestReadiness::ResourceBlocked,
        RequestReadiness::OutputBlocked,
        RequestReadiness::StateBlocked,
    ] {
        let mut request = decode(2);
        request.readiness = readiness;
        let snapshot = snapshot(vec![decode(1), request]);
        assert!(matches!(
            planner(3).propose(
                &snapshot,
                &Model(|_: &WaveExecutionShape| Some(5)),
                &TestResolver,
                &mut Clock(100)
            ),
            PlanningDecision::Unknown { .. }
        ));
    }
    let mut snapshot = snapshot(vec![decode(1)]);
    snapshot.requests[0].readiness = RequestReadiness::Unknown;
    assert!(matches!(
        planner(1).propose(
            &snapshot,
            &Model(|_: &WaveExecutionShape| Some(5)),
            &TestResolver,
            &mut Clock(100)
        ),
        PlanningDecision::Unknown {
            reason: PlanningUnknownReason::UnknownReadiness,
            ..
        }
    ));
}

#[test]
fn unknown_costs_and_exhausted_search_are_not_impossibility_proofs() {
    let snapshot = snapshot(vec![decode(1)]);
    assert!(matches!(
        planner(2).propose(
            &snapshot,
            &Model(|_: &WaveExecutionShape| None),
            &TestResolver,
            &mut Clock(100)
        ),
        PlanningDecision::Unknown {
            reason: PlanningUnknownReason::CostUnavailable,
            ..
        }
    ));
    assert!(matches!(
        planner(2).propose(
            &snapshot,
            &Model(|_: &WaveExecutionShape| Some(100)),
            &TestResolver,
            &mut Clock(100)
        ),
        PlanningDecision::Unknown { .. }
    ));
}

#[test]
fn only_historical_violations_and_certified_lower_bounds_prove_impossible() {
    let mut snapshot = snapshot(vec![decode(1)]);
    snapshot.requests[0].optimistic_next_service = Some(OptimisticServiceLowerBound {
        model_version: 7,
        duration_ns: 91,
    });
    assert!(matches!(
        planner(1).propose(
            &snapshot,
            &Model(|_: &WaveExecutionShape| Some(5)),
            &TestResolver,
            &mut Clock(100)
        ),
        PlanningDecision::ProvenImpossibleUnderModel {
            reason: PlanningImpossibleReason::CertifiedOptimisticLowerBound { .. },
            ..
        }
    ));
    snapshot.requests[0]
        .optimistic_next_service
        .as_mut()
        .unwrap()
        .model_version = 6;
    feasible(planner(1).propose(
        &snapshot,
        &Model(|_: &WaveExecutionShape| Some(5)),
        &TestResolver,
        &mut Clock(100),
    ));
    snapshot.requests[0].timing.slo_failed = true;
    assert!(matches!(
        planner(1).propose(
            &snapshot,
            &Model(|_: &WaveExecutionShape| Some(5)),
            &TestResolver,
            &mut Clock(100)
        ),
        PlanningDecision::ProvenImpossibleUnderModel {
            reason: PlanningImpossibleReason::HistoricalViolation { .. },
            ..
        }
    ));
}

#[test]
fn recompute_does_not_receive_duplicate_reference_work_credit() {
    let mut request = prefill(1);
    if let RequestPhaseView::Prefill(progress) = &mut request.phase {
        progress.logical_high_water = 4;
    }
    let mut snapshot = snapshot(vec![request]);
    snapshot.capabilities.prefill_chunk_sizes = vec![n32(4)];
    snapshot.scope.horizon_end_ns = 200;
    let (_, witness, _) = feasible(planner(2).propose(
        &snapshot,
        &Model(|_: &WaveExecutionShape| Some(5)),
        &TestResolver,
        &mut Clock(100),
    ));
    assert_eq!(witness.net_prefill_reference_work_ns, 60);
    assert_eq!(witness.predicted_output_tokens, 1);
}

#[test]
fn discrete_milestones_cannot_be_satisfied_by_late_completion() {
    let mut request = prefill(1);
    if let RequestPhaseView::Prefill(progress) = &mut request.phase {
        progress.milestones = Arc::from([PrefillMilestone {
            at_ns: 104,
            required_reference_work_ns: 40,
        }]);
    }
    let mut snapshot = snapshot(vec![request]);
    snapshot.capabilities.prefill_chunk_sizes = vec![n32(4)];
    assert!(matches!(
        planner(2).propose(
            &snapshot,
            &Model(|_: &WaveExecutionShape| Some(5)),
            &TestResolver,
            &mut Clock(100)
        ),
        PlanningDecision::Unknown { .. }
    ));
    feasible(planner(2).propose(
        &snapshot,
        &Model(|_: &WaveExecutionShape| Some(4)),
        &TestResolver,
        &mut Clock(100),
    ));
    let milestones = linear_prefill_milestones(0, 100, n64(1000), &[25, 50, 100], 100).unwrap();
    assert_eq!(
        milestones
            .iter()
            .map(|point| point.required_reference_work_ns)
            .collect::<Vec<_>>(),
        [150, 400, 1000]
    );
    assert!(linear_prefill_milestones(100, 100, n64(1), &[100], 0).is_none());
}

#[test]
fn fixed_reference_work_can_prefer_a_legal_mixed_wave_without_faking_tokens() {
    let snapshot = snapshot(vec![decode(1), prefill(2)]);
    let model = Model(|shape: &WaveExecutionShape| {
        Some(match shape.kind {
            WaveKind::Decode => 6,
            WaveKind::Mixed => 10,
            _ => 20,
        })
    });
    let (first, witness, _) =
        feasible(planner(1).propose(&snapshot, &model, &TestResolver, &mut Clock(100)));
    assert_eq!(
        first.candidate.execution_shape.exact().unwrap().kind,
        WaveKind::Mixed
    );
    assert_eq!(witness.predicted_output_tokens, 2);
    assert_eq!(witness.net_prefill_reference_work_ns, 100);
    assert!((witness.proxy_score - 1.2).abs() < 1e-10);
}

#[test]
fn fairness_breaks_equal_scores_deterministically() {
    let mut a = decode(1);
    let mut b = decode(2);
    a.fairness_rank = 20;
    b.fairness_rank = 10;
    let mut snapshot = snapshot(vec![a, b]);
    snapshot.capabilities.decode_batch_sizes = vec![nz(1)];
    let (first, _, _) = feasible(planner(2).propose(
        &snapshot,
        &Model(|_: &WaveExecutionShape| Some(5)),
        &TestResolver,
        &mut Clock(100),
    ));
    assert_eq!(
        first.candidate.work[0].key.request_id,
        snapshot.requests[1].key.request_id
    );
}

#[test]
fn gamma_distinguishes_soft_debt_between_two_hard_feasible_sequences() {
    let mut request = prefill(2);
    request.context_tokens = 4;
    if let RequestPhaseView::Prefill(progress) = &mut request.phase {
        progress.offset = 4;
        progress.logical_high_water = 4;
        progress.total_prompt_tokens = n32(16);
        progress.executable_until = 16;
        progress.reference = Arc::new(PrefillReferenceWork {
            evaluation: Default::default(),
            version: 1,
            points: vec![
                ReferenceWorkPoint {
                    prompt_tokens: 0,
                    cumulative_work_ns: 0,
                },
                ReferenceWorkPoint {
                    prompt_tokens: 4,
                    cumulative_work_ns: 40,
                },
                ReferenceWorkPoint {
                    prompt_tokens: 8,
                    cumulative_work_ns: 100,
                },
                ReferenceWorkPoint {
                    prompt_tokens: 16,
                    cumulative_work_ns: 200,
                },
            ],
        });
        // Both choices already satisfy the same hard granule requirement.
        progress.milestones = Arc::from([PrefillMilestone {
            at_ns: 150,
            required_reference_work_ns: 40,
        }]);
    }
    let mut snapshot = snapshot(vec![decode(1), request]);
    snapshot.capabilities.prefill_chunk_sizes = vec![n32(4)];
    let model = Model(|shape: &WaveExecutionShape| {
        Some(if shape.kind == WaveKind::Decode {
            5
        } else {
            10
        })
    });
    let mut planner = planner(1);
    planner.settings.search.prefill_credit_beta = 0.0;
    planner.settings.search.prefill_debt_gamma = 0.0;
    let (decode_only, first_witness, _) =
        feasible(planner.propose(&snapshot, &model, &TestResolver, &mut Clock(100)));
    assert_eq!(
        decode_only.candidate.execution_shape.exact().unwrap().kind,
        WaveKind::Decode
    );
    assert_eq!(first_witness.terminal_prefill_debt_ns, 65);
    planner.settings.search.prefill_debt_gamma = 0.1;
    let (mixed, second_witness, _) =
        feasible(planner.propose(&snapshot, &model, &TestResolver, &mut Clock(100)));
    assert_eq!(
        mixed.candidate.execution_shape.exact().unwrap().kind,
        WaveKind::Mixed
    );
    assert_eq!(second_witness.terminal_prefill_debt_ns, 10);
    assert_eq!(
        first_witness.predicted_output_tokens,
        second_witness.predicted_output_tokens
    );
    assert_eq!(
        first_witness.validated_through_ns,
        second_witness.validated_through_ns
    );
    planner.settings.search.enable_prefill_milestones = false;
    let (without_hard_milestones, witness, _) =
        feasible(planner.propose(&snapshot, &model, &TestResolver, &mut Clock(100)));
    assert_eq!(
        without_hard_milestones
            .candidate
            .execution_shape
            .exact()
            .unwrap()
            .kind,
        WaveKind::Mixed
    );
    assert_eq!(witness.terminal_prefill_debt_ns, 10);
}

#[test]
fn virtual_clock_budget_and_clock_reversal_are_explicit_unknowns() {
    let snapshot = snapshot(vec![decode(1)]);
    let mut planner = planner(1);
    planner.settings.search.max_planning_us = n64(1);
    assert!(matches!(
        planner.propose(
            &snapshot,
            &Model(|_: &WaveExecutionShape| Some(5)),
            &TestResolver,
            &mut ScriptClock::new(&[100, 1100])
        ),
        PlanningDecision::Unknown {
            reason: PlanningUnknownReason::ComputeBudgetExhausted,
            ..
        }
    ));
    assert!(matches!(
        planner.propose(
            &snapshot,
            &Model(|_: &WaveExecutionShape| Some(5)),
            &TestResolver,
            &mut ScriptClock::new(&[100, 99])
        ),
        PlanningDecision::Unknown {
            reason: PlanningUnknownReason::ClockMovedBackwards,
            ..
        }
    ));
}

#[test]
fn planning_overhead_is_included_before_returning_a_first_wave() {
    struct LookupClock<'a>(&'a std::cell::Cell<u64>);
    impl PlanningClock for LookupClock<'_> {
        fn now_ns(&mut self) -> u64 {
            self.0.get()
        }
    }
    let mut request = decode(1);
    request.timing.budgets.tpot_ns = n64(14); // deadline 104
    let snapshot = snapshot(vec![request]);
    let now = std::cell::Cell::new(100);
    let model = Model(|_: &WaveExecutionShape| {
        // Search's 100 + 4 fits exactly, but one unit spent in the real lookup
        // forces fresh replay to 101 + 4 > 104. The actual clock is still before
        // the deadline, so this is sequence failure, not DeadlineAlreadyMissed.
        now.set(101);
        Some(4)
    });
    assert!(matches!(
        planner(1).propose(&snapshot, &model, &TestResolver, &mut LookupClock(&now)),
        PlanningDecision::Unknown { .. }
    ));
}

#[test]
fn configuration_caps_and_invalid_snapshot_fail_before_unbounded_work() {
    let snapshot = snapshot(vec![decode(1)]);
    let mut planner = planner(1);
    planner.settings.search.lookahead_waves = nz(17);
    assert!(matches!(
        planner.propose(
            &snapshot,
            &Model(|_: &WaveExecutionShape| Some(5)),
            &TestResolver,
            &mut Clock(100)
        ),
        PlanningDecision::Unknown {
            reason: PlanningUnknownReason::InvalidConfiguration,
            ..
        }
    ));
    let mut duplicate = snapshot.clone();
    duplicate.requests.push(duplicate.requests[0].clone());
    assert!(matches!(
        super::BoundedSloPlanner::default().propose(
            &duplicate,
            &Model(|_: &WaveExecutionShape| Some(5)),
            &TestResolver,
            &mut Clock(100)
        ),
        PlanningDecision::Unknown {
            reason: PlanningUnknownReason::InvalidSnapshot,
            ..
        }
    ));
    let mut maintenance = snapshot;
    maintenance.has_unmodeled_maintenance = true;
    assert!(matches!(
        super::BoundedSloPlanner::default().propose(
            &maintenance,
            &Model(|_: &WaveExecutionShape| Some(5)),
            &TestResolver,
            &mut Clock(100)
        ),
        PlanningDecision::Unknown {
            reason: PlanningUnknownReason::UnmodeledMaintenance,
            ..
        }
    ));
}
