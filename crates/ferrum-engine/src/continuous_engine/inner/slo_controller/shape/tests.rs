//! Real work-boundary algebra used by the production route adapter. These
//! tests do not claim that unknown future host content has a cost witness.
use super::*;

mod execution;
use ferrum_scheduler::implementations::continuous::LogicalWorkGeneration;

mod host_domain;

fn n32(value: u32) -> NonZeroU32 {
    NonZeroU32::new(value).unwrap()
}
fn n64(value: u64) -> NonZeroU64 {
    NonZeroU64::new(value).unwrap()
}

fn decode(context: u32, generated: u32, maximum: u32) -> RequestSchedulingView {
    RequestSchedulingView {
        key: RequestWorkKey {
            request_id: RequestId::new(),
            incarnation: 7,
            work_generation: LogicalWorkGeneration::default(),
        },
        timing: RequestTimingView {
            ingress_at_ns: 10,
            first_commit_at_ns: (generated > 0).then_some(20),
            last_commit_at_ns: (generated > 0).then_some(20 + u64::from(generated)),
            committed_tokens: generated,
            maximum_output_tokens: n32(maximum),
            budgets: PlannerLatencyBudgets {
                ttft_ns: n64(100),
                tpot_ns: n64(100),
                itl_ns: n64(100),
            },
            slo_failed: false,
        },
        phase: RequestPhaseView::Decode,
        readiness: RequestReadiness::Ready,
        context_tokens: context,
        recurrent_state_bytes: 0,
        output_credit: OutputCreditView {
            available_token_commands: maximum,
            byte_backing: OutputByteBacking::PrepaidLifetime {
                remaining_token_commands: maximum,
                remaining_wire_bytes: 4096,
            },
        },
        output_policy_signature: [9; 32],
        recovery_service: RecoveryServiceDebt::new(std::num::NonZeroUsize::new(256).unwrap()),
        fairness_rank: 0,
        ranking_service_cost_ns: None,
        optimistic_next_service: None,
    }
}

fn prefill(total: u32, generated: u32, high_water: u32) -> RequestSchedulingView {
    let mut request = decode(0, generated, 8);
    request.phase = RequestPhaseView::Prefill(PrefillProgressView {
        admitted_at_ns: 13,
        reference_work_at_admission_ns: 0,
        offset: 0,
        total_prompt_tokens: n32(total),
        logical_high_water: high_water,
        reference: Arc::new(PrefillReferenceWork {
            evaluation: Default::default(),
            version: 4,
            points: vec![
                ReferenceWorkPoint {
                    prompt_tokens: 0,
                    cumulative_work_ns: 0,
                },
                ReferenceWorkPoint {
                    prompt_tokens: total,
                    cumulative_work_ns: 81,
                },
            ],
        }),
        milestones: vec![PrefillMilestone {
            at_ns: 100,
            required_reference_work_ns: 81,
        }]
        .into(),
        executable_until: total,
    });
    request
}

fn action(request: &RequestSchedulingView, action: WaveAction) -> CandidateWork {
    CandidateWork {
        key: request.key.clone(),
        action,
    }
}
fn chunk(offset: u32, count: u32) -> WaveAction {
    WaveAction::Prefill {
        offset,
        count: n32(count),
    }
}
fn frontiers(requests: &[RequestSchedulingView]) -> ProjectedFrontiers {
    ProjectedFrontiers::new(requests, u32::MAX, 256, &mut || Ok(())).unwrap()
}
fn advance(frontiers: &mut ProjectedFrontiers, work: Vec<CandidateWork>) -> Vec<ActualRowWork> {
    let prepared = frontiers.prepare(&work, &mut || Ok(())).unwrap();
    let actual = prepared.rows.iter().map(|row| row.work).collect();
    frontiers.advance(prepared).unwrap();
    actual
}

#[test]
fn rollout_partial_final_and_decodes_use_each_actual_prior_frontier() {
    let original = prefill(4, 0, 0);
    let mut state = frontiers(std::slice::from_ref(&original));
    assert_eq!(
        advance(&mut state, vec![action(&original, chunk(0, 2))]),
        vec![ActualRowWork::Prefill {
            offset: 0,
            count: 2,
            total_prompt_tokens: 4
        }]
    );
    assert_eq!(
        (state.request(0).context_tokens, state.request(0).generated),
        (2, 0)
    );
    assert!(!state.request(0).host_content_changed);
    let RequestPhaseView::Prefill(progress) = &state.request(0).phase else {
        panic!("partial");
    };
    assert_eq!(
        (
            progress.offset,
            progress.logical_high_water,
            progress.admitted_at_ns
        ),
        (2, 2, 13)
    );
    assert_eq!(
        advance(&mut state, vec![action(&original, chunk(2, 2))]),
        vec![ActualRowWork::Prefill {
            offset: 2,
            count: 2,
            total_prompt_tokens: 4
        }]
    );
    assert!(matches!(state.request(0).phase, RequestPhaseView::Decode));
    assert_eq!(
        (state.request(0).context_tokens, state.request(0).generated),
        (4, 1)
    );
    assert!(
        state.request(0).host_content_changed,
        "the next host route remains Unknown"
    );
    assert_eq!(
        advance(&mut state, vec![action(&original, WaveAction::Decode)]),
        vec![ActualRowWork::Decode { kv_tokens: 4 }]
    );
    assert_eq!(
        advance(&mut state, vec![action(&original, WaveAction::Decode)]),
        vec![ActualRowWork::Decode { kv_tokens: 5 }]
    );
    assert_eq!(
        (state.request(0).context_tokens, state.request(0).generated),
        (6, 3)
    );
    assert_eq!(
        state.request(0).key,
        original.key,
        "snapshot identity is never a new execution grant"
    );
    assert_eq!(
        original.context_tokens, 0,
        "the captured request remains immutable"
    );
}

#[test]
fn rollout_interleaved_owners_advance_only_selected_rows_in_physical_order() {
    let requests = [decode(7, 2, 8), prefill(4, 0, 0), decode(10, 5, 9)];
    let mut state = frontiers(&requests);
    let work = [
        action(&requests[2], WaveAction::Decode),
        action(&requests[1], chunk(0, 2)),
    ];
    let prepared = state.prepare(&work, &mut || Ok(())).unwrap();
    assert_eq!(
        prepared
            .rows
            .iter()
            .map(|row| row.index)
            .collect::<Vec<_>>(),
        [2, 1]
    );
    assert_eq!(
        prepared.rows[0].work,
        ActualRowWork::Decode { kv_tokens: 10 }
    );
    state.advance(prepared).unwrap();
    assert!(state.request(0).matches_query(&requests[0]));
    assert!(!state.request(0).host_content_changed);
    assert_eq!(
        (state.request(2).context_tokens, state.request(2).generated),
        (11, 6)
    );
    assert!(!state.request(1).host_content_changed);
    advance(
        &mut state,
        vec![
            action(&requests[0], WaveAction::Decode),
            action(&requests[1], chunk(2, 2)),
        ],
    );
    assert_eq!(
        (state.request(0).context_tokens, state.request(0).generated),
        (8, 3)
    );
    assert_eq!(
        (state.request(1).context_tokens, state.request(1).generated),
        (4, 1)
    );
    assert_eq!(
        (state.request(2).context_tokens, state.request(2).generated),
        (11, 6)
    );
}

#[test]
fn rollout_recompute_retains_net_credit_and_original_reference_anchor() {
    let original = prefill(5, 1, 5);
    let RequestPhaseView::Prefill(initial) = &original.phase else {
        unreachable!()
    };
    let mut state = frontiers(std::slice::from_ref(&original));
    advance(&mut state, vec![action(&original, chunk(0, 2))]);
    let RequestPhaseView::Prefill(current) = &state.request(0).phase else {
        panic!("recompute partial");
    };
    assert_eq!((current.offset, current.logical_high_water), (2, 5));
    assert_eq!(current.admitted_at_ns, initial.admitted_at_ns);
    assert_eq!(
        current.reference_work_at_admission_ns,
        initial.reference_work_at_admission_ns
    );
    assert!(Arc::ptr_eq(&current.reference, &initial.reference));
    assert!(Arc::ptr_eq(&current.milestones, &initial.milestones));
    assert_eq!(state.request(0).generated, 1);
    assert!(!state.request(0).host_content_changed);
    advance(&mut state, vec![action(&original, chunk(2, 3))]);
    assert_eq!(
        (state.request(0).context_tokens, state.request(0).generated),
        (5, 2)
    );
    assert_eq!(
        advance(&mut state, vec![action(&original, WaveAction::Decode)]),
        vec![ActualRowWork::Decode { kv_tokens: 5 }]
    );
    assert_eq!(
        (state.request(0).context_tokens, state.request(0).generated),
        (6, 3)
    );
}

#[test]
fn rollout_rejects_reordered_frontiers_and_does_not_partially_advance() {
    let requests = [prefill(4, 0, 0), decode(7, 2, 8)];
    let mut state = frontiers(&requests);
    let invalid = [
        action(&requests[1], WaveAction::Decode),
        action(&requests[0], chunk(2, 2)),
    ];
    assert!(matches!(
        state.prepare(&invalid, &mut || Ok(())),
        Err(PlanningUnknownReason::InvalidShapeEvidence)
    ));
    assert!(state.request(1).matches_query(&requests[1]));
    let duplicate = [
        action(&requests[1], WaveAction::Decode),
        action(&requests[1], WaveAction::Decode),
    ];
    assert!(matches!(
        state.prepare(&duplicate, &mut || Ok(())),
        Err(PlanningUnknownReason::InvalidShapeEvidence)
    ));
    let mut other_incarnation = action(&requests[1], WaveAction::Decode);
    other_incarnation.key.incarnation += 1;
    assert!(state.prepare(&[other_incarnation], &mut || Ok(())).is_err());
    let first = state
        .prepare(&[action(&requests[0], chunk(0, 2))], &mut || Ok(()))
        .unwrap();
    let stale = state
        .prepare(&[action(&requests[1], WaveAction::Decode)], &mut || Ok(()))
        .unwrap();
    state.advance(first).unwrap();
    assert_eq!(
        state.advance(stale),
        Err(PlanningUnknownReason::InvalidShapeEvidence)
    );
    assert!(state.request(1).matches_query(&requests[1]));
}

#[test]
fn rollout_checks_context_count_limits_and_overflow_before_projection() {
    let request = decode(u32::MAX, 1, 8);
    let state = frontiers(std::slice::from_ref(&request));
    assert!(matches!(
        state.prepare(&[action(&request, WaveAction::Decode)], &mut || Ok(())),
        Err(PlanningUnknownReason::ArithmeticOverflow)
    ));
    let mut request = prefill(u32::MAX, 0, u32::MAX - 1);
    let RequestPhaseView::Prefill(progress) = &mut request.phase else {
        unreachable!()
    };
    progress.offset = u32::MAX - 1;
    request.context_tokens = u32::MAX - 1;
    let state = frontiers(std::slice::from_ref(&request));
    assert!(matches!(
        state.prepare(&[action(&request, chunk(u32::MAX - 1, 2))], &mut || Ok(())),
        Err(PlanningUnknownReason::ArithmeticOverflow)
    ));
    let request = decode(6, 1, 2);
    let state =
        ProjectedFrontiers::new(std::slice::from_ref(&request), 6, 1, &mut || Ok(())).unwrap();
    assert!(matches!(
        state.prepare(&[action(&request, WaveAction::Decode)], &mut || Ok(())),
        Err(PlanningUnknownReason::OutputOrResourceBlocked)
    ));
    let request = decode(6, 2, 2);
    let state = frontiers(std::slice::from_ref(&request));
    assert!(matches!(
        state.prepare(&[action(&request, WaveAction::Decode)], &mut || Ok(())),
        Err(PlanningUnknownReason::InvalidShapeEvidence)
    ));
}

#[test]
fn rollout_preserves_executable_ceiling_and_checks_query_against_replay() {
    let mut request = prefill(4, 0, 0);
    let RequestPhaseView::Prefill(progress) = &mut request.phase else {
        unreachable!()
    };
    progress.executable_until = 2;
    let mut state = frontiers(std::slice::from_ref(&request));
    assert!(state
        .prepare(&[action(&request, chunk(0, 4))], &mut || Ok(()))
        .is_err());
    advance(&mut state, vec![action(&request, chunk(0, 2))]);
    assert!(state
        .prepare(&[action(&request, chunk(2, 2))], &mut || Ok(()))
        .is_err());
    assert!(!state.request(0).matches_query(&request));
    let mut queried = request.clone();
    queried.context_tokens = 2;
    let RequestPhaseView::Prefill(progress) = &mut queried.phase else {
        unreachable!()
    };
    progress.offset = 2;
    progress.logical_high_water = 2;
    assert!(state.request(0).matches_query(&queried));
    queried.timing.committed_tokens = 1;
    assert!(!state.request(0).matches_query(&queried));
    queried.timing.committed_tokens = 0;
    let RequestPhaseView::Prefill(progress) = &mut queried.phase else {
        unreachable!()
    };
    progress.admitted_at_ns += 1;
    assert!(!state.request(0).matches_query(&queried));
}

#[test]
fn rollout_budget_exhaustion_and_depth_bound_cannot_advance_work() {
    let request = decode(4, 1, 64);
    let mut state = frontiers(std::slice::from_ref(&request));
    assert!(matches!(
        state.prepare(&[action(&request, WaveAction::Decode)], &mut || Err(
            PlanningUnknownReason::ComputeBudgetExhausted
        )),
        Err(PlanningUnknownReason::ComputeBudgetExhausted)
    ));
    assert!(state.request(0).matches_query(&request));
    for _ in 0..16 {
        advance(&mut state, vec![action(&request, WaveAction::Decode)]);
    }
    assert_eq!(state.request(0).generated, 17);
    assert!(state
        .prepare(&[action(&request, WaveAction::Decode)], &mut || Ok(()))
        .is_err());
}
