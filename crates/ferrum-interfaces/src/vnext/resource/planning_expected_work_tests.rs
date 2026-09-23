//! Completion-only work uses real captured owners, including when cost or
//! physical-fit prediction is unknown. Constructing it never acquires a Step.
use super::*;
use crate::execution_cost::{
    ActualRowWork, ActualWaveKind, CompletionOnlyReason, ExpectedExecutionWave, ExpectedWaveInput,
    ExpectedWaveWork, ExpectedWorkSelection, WaveCommitment,
};
use crate::model_executor::{LogitsReturnPolicy, PrefillChunk, TokenSelectionMask};
use ferrum_types::RequestId;

fn prefill(count: u32) -> ExpectedWorkSelection {
    ExpectedWorkSelection {
        participant_index: 0,
        request_id: RequestId::new(),
        owner_incarnation: NonZeroU64::new(1).unwrap(),
        work_generation: NonZeroU64::new(1).unwrap(),
        input: ExpectedWaveInput::Prefill {
            chunk: PrefillChunk::new(0, count as usize, 4).unwrap(),
        },
        work: ActualRowWork::Prefill {
            offset: 0,
            count,
            total_prompt_tokens: 4,
        },
        decode_policy: None,
    }
}

#[test]
fn expected_work_keeps_real_owner_when_fit_is_unknown_without_allocating_or_pinning() {
    let (harness, bucket, lane) = setup(128);
    let sequence = admitted_sequence_with_ceiling(&harness.root, "completion-work", 4);
    let session = sequence.open_session().unwrap();
    let snapshot = lane_view(&harness.root, &session, &lane);
    let before = harness.runtime.allocate_calls();
    assert!(matches!(
        harness.root.project_resource_wave_with_bucket(
            &snapshot,
            &snapshot.initial_state(),
            &[row(0, 0, 2)],
            Some(bucket.bucket_id()),
            &mut || true,
        ),
        ResourcePlanningAvailability::Unknown(ResourcePlanningUnknown::PhysicalCapacity)
    ));
    let work = ExpectedWaveWork::new(&snapshot, ActualWaveKind::Prefill, vec![prefill(2)]).unwrap();
    assert_eq!(work.query_tokens(), 2);
    assert_eq!(work.lane_id(), lane.id());
    assert_eq!(work.plan_hash(), snapshot.plan_hash());
    assert_eq!(work.coordinator_id(), snapshot.coordinator_id());
    assert!(work.participants()[0]
        .resource()
        .matches_session_identity(&session));
    assert!(!work.participants()[0].produces_token());
    let expected =
        ExpectedExecutionWave::complete_requests(work, CompletionOnlyReason::CostUnavailable);
    assert!(matches!(
        expected.commitment(),
        WaveCommitment::CompleteRequests(CompletionOnlyReason::CostUnavailable)
    ));
    assert_eq!(harness.runtime.allocate_calls(), before);
    session.try_abort_if_quiescent().unwrap();
    drop(session);
    drop(sequence);
    drop(lane);
    close_dynamic_test_root(harness.root);
    assert_eq!(
        expected.work().query_tokens(),
        2,
        "an expectation retains no physical owner"
    );
}

#[test]
fn expected_work_rejects_changed_span_duplicate_owner_and_incorrect_phase_or_fit() {
    let (harness, _, lane) = setup(256);
    let sequence = admitted_sequence_with_ceiling(&harness.root, "completion-work", 4);
    let session = sequence.open_session().unwrap();
    let snapshot = lane_view(&harness.root, &session, &lane);
    let mut wrong_span = prefill(2);
    wrong_span.work = ActualRowWork::Prefill {
        offset: 0,
        count: 1,
        total_prompt_tokens: 4,
    };
    assert!(ExpectedWaveWork::new(&snapshot, ActualWaveKind::Prefill, vec![wrong_span]).is_err());
    assert!(ExpectedWaveWork::new(&snapshot, ActualWaveKind::Decode, vec![prefill(2)]).is_err());
    assert!(ExpectedWaveWork::new(&snapshot, ActualWaveKind::Mixed, vec![prefill(2)]).is_err());
    let mut alias = prefill(2);
    alias.request_id = RequestId::new();
    assert!(
        ExpectedWaveWork::new(&snapshot, ActualWaveKind::Prefill, vec![prefill(2), alias]).is_err()
    );
    let mut decode = prefill(2);
    decode.input = ExpectedWaveInput::Decode {
        cache_id: "cache".into(),
    };
    decode.work = ActualRowWork::Decode { kv_tokens: 4 };
    decode.decode_policy = Some(LogitsReturnPolicy::FullLogits);
    assert!(
        ExpectedWaveWork::new(&snapshot, ActualWaveKind::Decode, vec![decode.clone()]).is_err()
    );
    decode.work = ActualRowWork::Decode { kv_tokens: 3 };
    let valid =
        ExpectedWaveWork::new(&snapshot, ActualWaveKind::Decode, vec![decode.clone()]).unwrap();
    assert!(valid.participants()[0].produces_token());
    decode.decode_policy = None;
    assert!(ExpectedWaveWork::new(&snapshot, ActualWaveKind::Decode, vec![decode]).is_err());
    let final_work =
        ExpectedWaveWork::new(&snapshot, ActualWaveKind::Prefill, vec![prefill(4)]).unwrap();
    assert!(final_work.participants()[0].produces_token());
    session.try_abort_if_quiescent().unwrap();
    drop(session);
    drop(sequence);
    drop(lane);
    close_dynamic_test_root(harness.root);
}

#[test]
fn captured_decode_policy_detects_mask_replacement_and_sampling_mode_change() {
    let original = LogitsReturnPolicy::GreedyArgmax {
        token_mask: Some(TokenSelectionMask::new(vec![1, 1, 0])),
        repetition_penalty: None,
    };
    let captured = original.clone();
    assert!(captured.same_captured_input(&original));
    assert!(!captured.same_captured_input(&LogitsReturnPolicy::FullLogits));
    let equal_replacement = LogitsReturnPolicy::GreedyArgmax {
        token_mask: Some(TokenSelectionMask::new(vec![1, 1, 0])),
        repetition_penalty: None,
    };
    assert!(!captured.same_captured_input(&equal_replacement));
    let mut changed = original.clone();
    let LogitsReturnPolicy::GreedyArgmax {
        token_mask: Some(mask),
        ..
    } = &mut changed
    else {
        unreachable!();
    };
    mask.set_tokens_validity(&[1], false);
    assert!(!captured.same_captured_input(&changed));
    assert!(captured.same_captured_input(&original));
}
