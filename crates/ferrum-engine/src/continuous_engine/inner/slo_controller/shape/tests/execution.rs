//! Read-view validation, not fabricated physical-route/cost evidence.
use super::super::execution::input_frontiers;
use super::*;
use ferrum_scheduler::implementations::continuous::{
    cost_model::{BatchOrderSemantics, ExecutionFingerprint, WaveExecutionPath, WaveGraphState},
    slo_planner::PlanningExecutionInput,
};

fn snapshot(requests: Vec<RequestSchedulingView>) -> SchedulerSnapshot {
    SchedulerSnapshot {
        observed_at_ns: 30,
        generation: 7,
        cost_model_version: 4,
        fingerprint: ExecutionFingerprint {
            model_weights: [1; 32],
            numerical_policy: [2; 32],
            device_runtime: [3; 32],
            execution_config: [4; 32],
        },
        requests,
        capabilities: BackendPlanningCapabilities {
            work_policy: Default::default(),
            path: WaveExecutionPath::NativeUnified,
            graph_state: WaveGraphState::Disabled,
            order: BatchOrderSemantics::Ordered,
            decode_batch_sizes: vec![NonZeroUsize::new(1).unwrap(), NonZeroUsize::new(2).unwrap()],
            prefill_batch_sizes: vec![NonZeroUsize::new(1).unwrap()],
            prefill_chunk_sizes: vec![n32(2), n32(4)],
            prefill_alignment: n32(1),
            allow_final_short_chunk: true,
            native_mixed: true,
            max_wave_rows: NonZeroUsize::new(8).unwrap(),
            max_prefill_tokens_per_wave: n64(32),
            workspace_bytes_upper_bound: 0,
        },
        capacity: CapacityReadView {
            evidence_known: false,
            available_kv_tokens: 0,
            maximum_context_tokens: n32(4096),
            available_workspace_bytes: 0,
            available_output_bytes: 0,
        },
        scope: PlanningScope {
            horizon_end_ns: 100,
            reference_decode_token_ns: n64(10),
            reference_work_version: 4,
        },
        has_unmodeled_maintenance: false,
    }
}

fn verify(
    snapshot: &SchedulerSnapshot,
    requests: &[RequestSchedulingView],
    depth: usize,
    work: &[CandidateWork],
    rows: &[PlanningShapeRow<'_>],
    kind: ActualWaveKind,
) -> std::result::Result<Vec<ProjectedRequest>, PlanningUnknownReason> {
    input_frontiers(
        snapshot,
        &PlanningExecutionInput {
            work,
            requests,
            kind,
            rows,
            recurrent_state_bytes: rows
                .iter()
                .map(|row| row.request.recurrent_state_bytes)
                .sum(),
        },
        depth,
        &mut || Ok(()),
    )
}

#[test]
fn unified_parent_partial_final_decode_uses_supplied_logical_frontier() {
    let captured = snapshot(vec![prefill(4, 0, 0), decode(9, 2, 8)]);
    let original = captured.requests.clone();
    let mut current = original.clone();
    current[0].context_tokens = 2;
    let RequestPhaseView::Prefill(progress) = &mut current[0].phase else {
        panic!()
    };
    progress.offset = 2;
    progress.logical_high_water = 2;
    let work = [action(&current[0], chunk(2, 2))];
    let rows = [PlanningShapeRow {
        request: &current[0],
        work: ActualRowWork::Prefill {
            offset: 2,
            count: 2,
            total_prompt_tokens: 4,
        },
    }];
    let view = verify(
        &captured,
        &current,
        1,
        &work,
        &rows,
        ActualWaveKind::Prefill,
    )
    .unwrap();
    assert_eq!((view[0].context_tokens, view[0].generated), (2, 0));
    assert!(!view[0].host_content_changed);
    assert!(view[1].matches_query(&original[1]));

    current[0].phase = RequestPhaseView::Decode;
    current[0].context_tokens = 4;
    current[0].timing.committed_tokens = 1;
    current[0].timing.first_commit_at_ns = Some(35);
    current[0].timing.last_commit_at_ns = Some(35);
    let work = [action(&current[0], WaveAction::Decode)];
    let rows = [PlanningShapeRow {
        request: &current[0],
        work: ActualRowWork::Decode { kv_tokens: 4 },
    }];
    let view = verify(&captured, &current, 2, &work, &rows, ActualWaveKind::Decode).unwrap();
    assert!(view[0].host_content_changed);
    assert!(!view[1].host_content_changed);
    assert_eq!((view[0].context_tokens, view[0].generated), (4, 1));
    assert_eq!(
        captured.requests, original,
        "adapter does not advance the captured logical state"
    );
}

#[test]
fn unified_parent_recompute_preserves_reference_and_net_highwater() {
    let captured = snapshot(vec![prefill(4, 1, 4)]);
    let mut current = captured.requests.clone();
    current[0].context_tokens = 2;
    let RequestPhaseView::Prefill(progress) = &mut current[0].phase else {
        panic!()
    };
    progress.offset = 2;
    let work = [action(&current[0], chunk(2, 2))];
    let row_work = ActualRowWork::Prefill {
        offset: 2,
        count: 2,
        total_prompt_tokens: 4,
    };
    let rows = [PlanningShapeRow {
        request: &current[0],
        work: row_work,
    }];
    let result = verify(
        &captured,
        &current,
        1,
        &work,
        &rows,
        ActualWaveKind::Prefill,
    )
    .unwrap();
    assert!(!result[0].host_content_changed);
    let RequestPhaseView::Prefill(progress) = &result[0].phase else {
        panic!()
    };
    assert_eq!(
        (
            progress.offset,
            progress.logical_high_water,
            progress.admitted_at_ns
        ),
        (2, 4, 13)
    );
    let mut altered = current.clone();
    let RequestPhaseView::Prefill(progress) = &mut altered[0].phase else {
        panic!()
    };
    progress.reference = Arc::new((*progress.reference).clone());
    let rows = [PlanningShapeRow {
        request: &altered[0],
        work: row_work,
    }];
    assert!(matches!(
        verify(
            &captured,
            &altered,
            1,
            &work,
            &rows,
            ActualWaveKind::Prefill
        ),
        Err(PlanningUnknownReason::InvalidShapeEvidence)
    ));
}

#[test]
fn unified_parent_rejects_missing_peer_and_detached_or_stale_rows() {
    let captured = snapshot(vec![decode(4, 1, 8), decode(9, 2, 8)]);
    let work = [action(&captured.requests[0], WaveAction::Decode)];
    let rows = [PlanningShapeRow {
        request: &captured.requests[0],
        work: ActualRowWork::Decode { kv_tokens: 4 },
    }];
    assert!(verify(
        &captured,
        &captured.requests,
        0,
        &work,
        &rows,
        ActualWaveKind::Decode
    )
    .is_ok());
    assert!(verify(
        &captured,
        &captured.requests[..1],
        0,
        &work,
        &rows,
        ActualWaveKind::Decode
    )
    .is_err());
    let mut wrong = captured.requests[0].clone();
    wrong.output_policy_signature[0] ^= 1;
    let detached = [PlanningShapeRow {
        request: &wrong,
        work: rows[0].work,
    }];
    assert!(verify(
        &captured,
        &captured.requests,
        0,
        &work,
        &detached,
        ActualWaveKind::Decode
    )
    .is_err());
    let stale = [PlanningShapeRow {
        request: &captured.requests[0],
        work: ActualRowWork::Decode { kv_tokens: 5 },
    }];
    assert!(verify(
        &captured,
        &captured.requests,
        0,
        &work,
        &stale,
        ActualWaveKind::Decode
    )
    .is_err());
    let mut peers = captured.requests.clone();
    peers[1].key.incarnation += 1;
    assert!(verify(&captured, &peers, 0, &work, &rows, ActualWaveKind::Decode).is_err());
}

#[test]
fn unified_parent_kind_recurrent_and_budget_cannot_be_omitted() {
    let mut captured = snapshot(vec![decode(4, 1, 8)]);
    captured.requests[0].recurrent_state_bytes = 64;
    let work = [action(&captured.requests[0], WaveAction::Decode)];
    let rows = [PlanningShapeRow {
        request: &captured.requests[0],
        work: ActualRowWork::Decode { kv_tokens: 4 },
    }];
    let mut input = PlanningExecutionInput {
        work: &work,
        requests: &captured.requests,
        kind: ActualWaveKind::Decode,
        rows: &rows,
        recurrent_state_bytes: 0,
    };
    assert!(matches!(
        input_frontiers(&captured, &input, 0, &mut || Ok(())),
        Err(PlanningUnknownReason::InvalidShapeEvidence)
    ));
    input.recurrent_state_bytes = 64;
    input.kind = ActualWaveKind::Mixed;
    assert!(input_frontiers(&captured, &input, 0, &mut || Ok(())).is_err());
    input.kind = ActualWaveKind::Decode;
    assert!(matches!(
        input_frontiers(&captured, &input, 0, &mut || Err(
            PlanningUnknownReason::ComputeBudgetExhausted
        )),
        Err(PlanningUnknownReason::ComputeBudgetExhausted)
    ));
    assert!(input_frontiers(&captured, &input, 0, &mut || Ok(())).is_ok());
}
