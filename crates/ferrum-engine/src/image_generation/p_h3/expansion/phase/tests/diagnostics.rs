//! These execute the same real Stored Qwen program and owned phase as the
//! production consumer. The collector is its original charged Host backing.
use super::*;
use crate::image_generation::continuous_trace::{
    ImageContinuousTraceLimits, ImageContinuousTraceRecorder,
};
use crate::image_generation::p_h3::expansion::{
    DiagnosticContext, PH3ImagePromptExpansionDiagnosticPolicy as Policy,
};
use std::time::Instant;

async fn observed(
    policy: Policy,
    maximum_bytes: usize,
    max_tokens: u32,
    cancel: bool,
) -> (
    Result<ExpansionPhaseProgress>,
    Vec<serde_json::Value>,
    crate::image_generation::continuous_trace::ImageContinuousTraceStatus,
) {
    let mut fixture = Fixture::new();
    let assets = fixture.take_expansion();
    let prepared = prepared(&assets, max_tokens);
    let prompt_tokens = prepared.tokens.len();
    let (composition, runtime) = composition();
    let host = host();
    let (recorder, handle) =
        ImageContinuousTraceRecorder::new(&host, ImageContinuousTraceLimits { maximum_bytes })
            .unwrap();
    let backing = handle.status().backing_bytes;
    let requirements = assets
        .host_requirements(&composition, &prepared, &host)
        .unwrap();
    let phase = host.reserve_scope(&requirements.claims()).unwrap();
    let context = (policy != Policy::Disabled).then(|| DiagnosticContext {
        policy,
        trace: handle.clone(),
        job: 19,
        request_id: Some(uuid::Uuid::new_v4()),
    });
    let mut phase = NativeExpansionPhase::initialize_with_diagnostics(
        assets,
        &composition,
        prepared,
        phase,
        context,
    )
    .unwrap();
    let result = complete(&mut phase, cancel).await;
    assert_eq!(
        phase.cleanup().snapshot().phase(),
        ExecutorShutdownPhase::Retired
    );
    assert_eq!(runtime.resident_bytes(), 0);
    assert_eq!(
        host.usage().unwrap().committed_bytes(),
        backing,
        "only the one original trace remains charged after real retirement"
    );
    if result.is_ok() && !cancel && maximum_bytes >= 65536 && policy != Policy::Disabled {
        let metrics = phase.cohort.participants[0].observation.as_ref();
        assert!(metrics.is_none(), "terminal observation is consumed once");
    }
    drop(phase);
    recorder.seal();
    let status = handle.status();
    let mut bytes = Vec::new();
    handle.write_jsonl(&mut bytes).unwrap();
    let values: Vec<serde_json::Value> = bytes
        .split(|&b| b == b'\n')
        .filter(|line| !line.is_empty())
        .map(|line| serde_json::from_slice(line).unwrap())
        .collect();
    let events = values
        .into_iter()
        .filter_map(|mut v| v.as_object_mut()?.remove("event"))
        .collect::<Vec<_>>();
    if let Some(event) = events.first().filter(|_| !cancel && max_tokens == 2) {
        assert_eq!(event["prompt_tokens"], prompt_tokens);
        assert_eq!(event["token_history"]["output_tokens_including_eos"], 2);
        assert_eq!(
            event["executor_metrics"]["prefill_operations"],
            prompt_tokens.div_ceil(2)
        );
        assert_eq!(event["executor_metrics"]["decode_operations"], 1);
    }
    drop(handle);
    drop(recorder);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    (result, events, status)
}

#[tokio::test]
async fn prompt_expansion_diagnostics_default_summary_and_capture_preserve_the_same_real_answer() {
    for policy in [
        Policy::Disabled,
        Policy::Summary,
        Policy::CaptureEffectivePrompt {
            maximum_utf8_bytes: 4096,
        },
    ] {
        let (result, events, status) = observed(policy, 65536, 2, false).await;
        let ExpansionPhaseProgress::Expanded(text) = result.unwrap() else {
            panic!("actual EOS answer required")
        };
        assert_eq!(text, "A small red cat sits quietly.");
        assert!(!status.truncated);
        if policy == Policy::Disabled {
            assert!(events.is_empty());
            continue;
        }
        assert_eq!(events.len(), 1);
        let event = &events[0];
        assert_eq!(event["outcome"], "expanded");
        assert_eq!(event["executor_retired"], true);
        assert_eq!(event["token_history"]["last_token_id"], EOS_ID);
        assert_eq!(event["token_history"]["protocol_eos_observed"], true);
        assert_eq!(event["token_history"]["finish"], "EOS");
        assert!(event["initialization_wall_us"].is_u64());
        assert!(event["retirement_wall_us"].is_u64());
        assert!(!event.to_string().contains("</think>"));
        if matches!(policy, Policy::CaptureEffectivePrompt { .. }) {
            assert_eq!(event["capture"]["text"], text);
        } else {
            assert_eq!(event["capture"]["status"], "disabled");
            assert!(!event.to_string().contains(&text));
        }
    }
}

#[tokio::test]
async fn prompt_expansion_diagnostics_capture_ceiling_and_full_collector_never_replace_success() {
    let (result, events, status) = observed(
        Policy::CaptureEffectivePrompt {
            maximum_utf8_bytes: 1,
        },
        65536,
        2,
        false,
    )
    .await;
    assert!(matches!(result, Ok(ExpansionPhaseProgress::Expanded(_))));
    assert_eq!(events[0]["capture"]["status"], "capacity_refused");
    assert!(events[0]["capture"].get("text").is_none());
    assert!(
        !status.truncated,
        "explicit policy refusal is recorded, rather than a lost collector event"
    );
    let (result, events, status) = observed(Policy::Summary, 64, 2, false).await;
    assert!(matches!(result, Ok(ExpansionPhaseProgress::Expanded(_))));
    assert!(events.is_empty());
    assert!(status.truncated);
    assert_eq!(status.dropped, 1);
}

#[tokio::test]
async fn prompt_expansion_diagnostics_length_and_cancellation_never_capture_an_effective_answer() {
    for (limit, cancel, finish) in [(1, false, "Length"), (2, true, "Cancelled")] {
        let (result, events, status) = observed(
            Policy::CaptureEffectivePrompt {
                maximum_utf8_bytes: 4096,
            },
            65536,
            limit,
            cancel,
        )
        .await;
        if cancel {
            assert!(matches!(result, Ok(ExpansionPhaseProgress::Cancelled)));
        } else {
            assert!(result.is_err());
        }
        assert!(!status.truncated);
        assert_eq!(events.len(), 1);
        assert_eq!(events[0]["token_history"]["finish"], finish);
        assert_eq!(events[0]["token_history"]["protocol_eos_observed"], false);
        assert_eq!(events[0]["capture"]["status"], "unavailable");
        assert!(events[0]["effective_prompt_sha256"].is_null());
        if cancel {
            assert!(events[0]["executor_metrics"]["avg_prefill_time_ms"].is_null());
            assert!(events[0]["executor_metrics"]["avg_decode_time_ms"].is_null());
        }
    }
}

#[tokio::test]
async fn prompt_expansion_diagnostics_wait_for_original_retirement_before_publishing() {
    let mut fixture = Fixture::new();
    let assets = fixture.take_expansion();
    let prepared = prepared(&assets, 2);
    let (composition, _) = composition();
    let host = host();
    let (recorder, handle) = ImageContinuousTraceRecorder::new(
        &host,
        ImageContinuousTraceLimits {
            maximum_bytes: 65536,
        },
    )
    .unwrap();
    let requirements = assets
        .host_requirements(&composition, &prepared, &host)
        .unwrap();
    let phase = host.reserve_scope(&requirements.claims()).unwrap();
    let mut phase = NativeExpansionPhase::initialize_with_diagnostics(
        assets,
        &composition,
        prepared,
        phase,
        Some(DiagnosticContext {
            policy: Policy::Summary,
            trace: handle.clone(),
            job: 3,
            request_id: None,
        }),
    )
    .unwrap();
    let held_scope = phase.cohort.owner.scope().unwrap();
    let deadline = Instant::now() + std::time::Duration::from_secs(10);
    while !matches!(phase.cohort.state, Some(CohortState::Retiring { .. })) {
        assert!(Instant::now() < deadline);
        phase.advance(false, &futures::task::noop_waker()).unwrap();
    }
    assert!(matches!(
        phase.advance(false, &futures::task::noop_waker()).unwrap(),
        ExpansionPhaseProgress::Pending {
            made_progress: false
        }
    ));
    assert_eq!(handle.status().recorded, 0);
    drop(held_scope);
    assert!(matches!(
        complete(&mut phase, false).await.unwrap(),
        ExpansionPhaseProgress::Expanded(_)
    ));
    assert_eq!(handle.status().recorded, 1);
    drop(phase);
    drop(handle);
    drop(recorder);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}
