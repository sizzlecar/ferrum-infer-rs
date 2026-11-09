use super::*;
use crate::image_generation::continuous_trace::{
    ImageContinuousTraceLimits, ImageContinuousTraceRecorder,
};
use crate::image_generation::p_h3::expansion::{
    DiagnosticContext, PH3ImagePromptExpansionDiagnosticPolicy,
};

#[tokio::test]
async fn real_prefill_and_decode_batch_each_submit_once_and_trace_original_members() {
    let mut fixture = Fixture::new_two_answers();
    let assets = fixture.take_expansion();
    let mut requests = requests(&assets, 2);
    requests[1].tokens = vec![TokenId::new(0); 3];
    requests[1].prefill_chunk_tokens = 2;
    requests[1].engine.batching.max_num_batched_tokens = 2;
    requests[1].engine.runtime.max_model_len = Some(5);
    let prepared = assets.prepare_text_cohort(requests).unwrap();
    let (composition, runtime) = composition();
    let host = host();
    let (recorder, trace) = ImageContinuousTraceRecorder::new(
        &host,
        ImageContinuousTraceLimits {
            // Keep both complete executor-metric observations plus every
            // member's wave records within this explicit Host Output grant.
            maximum_bytes: 256 * 1024,
        },
    )
    .unwrap();
    let trace_bytes = trace.status().backing_bytes;
    let requirements = assets
        .cohort_host_requirements(&composition, &prepared, &host)
        .unwrap();
    let phase = host.reserve_scope(&requirements.claims()).unwrap();
    let request_ids = [uuid::Uuid::new_v4(), uuid::Uuid::new_v4()];
    let diagnostics = request_ids
        .iter()
        .enumerate()
        .map(|(index, request)| {
            Some(DiagnosticContext {
                policy: PH3ImagePromptExpansionDiagnosticPolicy::Summary,
                trace: trace.clone(),
                job: 101 + index as u64,
                request_id: Some(*request),
            })
        })
        .collect();
    let mut cohort = NativeExpansionCohort::initialize_with_diagnostics(
        assets,
        &composition,
        prepared,
        phase,
        diagnostics,
    )
    .unwrap();
    let original = cohort.owner.scope().unwrap();
    advance_until(&mut cohort, &[false, false], |c| {
        c.participants
            .iter()
            .all(|p| matches!(p.state, Some(ParticipantState::Tokens(_))))
    });
    // Two chunks cover the three-token prompts; the next wave decodes each
    // actual answer into EOS. A single native submission advances both rows.
    for expected_frontier in [2, 3, 4] {
        let before = original.cache_metrics_snapshot().unwrap()["counters"]["submitted_waves"]
            .as_u64()
            .unwrap();
        cohort
            .advance(&[false, false], &futures::task::noop_waker())
            .unwrap();
        let after = original.cache_metrics_snapshot().unwrap()["counters"]["submitted_waves"]
            .as_u64()
            .unwrap();
        assert_eq!(after, before + 1);
        for participant in &cohort.participants {
            let consumed = match &participant.state {
                Some(ParticipantState::Tokens(cursor)) => cursor.frontier_tokens(),
                Some(ParticipantState::Terminal) => participant
                    .completed
                    .as_ref()
                    .unwrap()
                    .frontier
                    .consumed_tokens(),
                _ => panic!("successful native batch lost a participant"),
            };
            assert_eq!(consumed, expected_frontier);
        }
    }
    assert_eq!(
        cohort.participants[0]
            .completed
            .as_ref()
            .unwrap()
            .history
            .ids(),
        &[ANSWER_ID, EOS_ID]
    );
    assert_eq!(
        cohort.participants[1]
            .completed
            .as_ref()
            .unwrap()
            .history
            .ids(),
        &[SECOND_ANSWER_ID, EOS_ID]
    );
    assert!(matches!(cohort.state, Some(CohortState::Retiring { .. })));
    drop(original);
    let results = complete(&mut cohort, &[false, false]).await.unwrap();
    assert!(results
        .iter()
        .all(|result| matches!(result, ExpansionParticipantResult::Expanded(_))));
    assert_eq!(runtime.resident_bytes(), 0);
    assert_eq!(host.usage().unwrap().committed_bytes(), trace_bytes);
    drop(cohort);
    recorder.seal();
    let status = trace.status();
    assert!(
        !status.truncated,
        "complete cohort trace required: {status:?}"
    );
    let mut bytes = Vec::new();
    trace.write_jsonl(&mut bytes).unwrap();
    let events: Vec<serde_json::Value> = bytes
        .split(|&byte| byte == b'\n')
        .filter(|line| !line.is_empty())
        .map(|line| serde_json::from_slice::<serde_json::Value>(line).unwrap()["event"].clone())
        .filter(|event| !event.is_null())
        .collect();
    let waves: Vec<_> = events
        .iter()
        .filter(|event| event["kind"] == "prompt_expansion_token_wave")
        .collect();
    assert!(!waves.is_empty());
    let owner = waves[0]["shared_owner"]["id"].clone();
    let decode: Vec<_> = waves
        .iter()
        .filter(|event| event["before_consumed_tokens"] == 3)
        .collect();
    assert_eq!(decode.len(), request_ids.len());
    assert_eq!(decode[0]["wave"], decode[1]["wave"]);
    for event in waves {
        let input_index = event["input_index"].as_u64().unwrap() as usize;
        assert_eq!(event["job"], 101 + input_index as u64);
        assert_eq!(event["request_id"], request_ids[input_index].to_string());
        assert_eq!(event["shared_owner"]["id"], owner);
        assert_eq!(event["participants"], 2);
        assert_eq!(event["outcome"], "completed");
        assert_eq!(event["native_outcome"], "completed");
    }
    let final_events: Vec<_> = events
        .iter()
        .filter(|event| event["kind"] == "prompt_expansion_observed")
        .collect();
    assert_eq!(final_events.len(), request_ids.len());
    for event in final_events {
        assert_eq!(event["shared_owner"]["id"], owner);
        assert_eq!(event["shared_owner"]["participants"], 2);
        assert_eq!(event["token_history"]["output_tokens_including_eos"], 2);
        assert_eq!(event["executor_retired"], true);
    }
    drop(trace);
    drop(recorder);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}

#[test]
fn diagnostic_extent_is_checked_before_native_initialization() {
    let mut fixture = Fixture::new_two_answers();
    let assets = fixture.take_expansion();
    let prepared = assets.prepare_text_cohort(requests(&assets, 2)).unwrap();
    let (composition, runtime) = composition();
    let host = host();
    let requirements = assets
        .cohort_host_requirements(&composition, &prepared, &host)
        .unwrap();
    let phase = host.reserve_scope(&requirements.claims()).unwrap();
    let error = NativeExpansionCohort::initialize_with_diagnostics(
        assets,
        &composition,
        prepared,
        phase,
        vec![None],
    )
    .err()
    .unwrap();
    assert!(error.cleanup().is_none());
    assert_eq!(runtime.resident_bytes(), 0);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}

#[tokio::test]
async fn a_single_participants_io_grant_cannot_authorize_a_native_batch() {
    let mut fixture = Fixture::new_two_answers();
    let assets = fixture.take_expansion();
    let prepared = assets.prepare_text_cohort(requests(&assets, 2)).unwrap();
    let (composition, runtime) = composition();
    let host = host();
    let single = assets
        .host_requirements(&composition, &prepared.requests[0], &host)
        .unwrap();
    let batch = assets
        .cohort_host_requirements(&composition, &prepared, &host)
        .unwrap();
    assert!(batch.budget.input > single.budget.input);
    let mut claims = batch.claims();
    let (_, input) = claims
        .iter_mut()
        .find(|(class, _)| *class == ferrum_runtime::HostMemoryClass::InputPreprocess)
        .unwrap();
    *input = single.budget.input;
    let phase = host.reserve_scope(&claims).unwrap();
    let mut cohort =
        NativeExpansionCohort::initialize(assets, &composition, prepared, phase).unwrap();
    let original = cohort.owner.scope().unwrap();
    advance_until(&mut cohort, &[false, false], |c| {
        c.participants
            .iter()
            .all(|p| matches!(p.state, Some(ParticipantState::Tokens(_))))
    });
    let before = original.cache_metrics_snapshot().unwrap()["counters"]["submitted_waves"].clone();
    cohort
        .advance(&[false, false], &futures::task::noop_waker())
        .unwrap();
    assert_eq!(
        original.cache_metrics_snapshot().unwrap()["counters"]["submitted_waves"],
        before
    );
    assert!(cohort.participants.iter().all(|p| matches!(
        p.failure.as_ref().map(|f| &f.error),
        Some(FerrumError::ResourceExhausted { .. })
    )));
    assert!(matches!(cohort.state, Some(CohortState::Retiring { .. })));
    assert!(host.usage().unwrap().committed_bytes() > 0);
    drop(original);
    let results = complete(&mut cohort, &[false, false]).await.unwrap();
    assert!(results
        .iter()
        .all(|result| matches!(result, ExpansionParticipantResult::Failed(_))));
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    assert_eq!(runtime.resident_bytes(), 0);
}
