use super::super::super::PH3ImagePromptExpansionDiagnosticPolicy as Policy;
use super::*;
use crate::image_generation::continuous_trace::{
    ImageContinuousTraceHandle, ImageContinuousTraceLimits, ImageContinuousTraceRecorder,
};
use serde_json::Value;

fn contexts(
    handle: &ImageContinuousTraceHandle,
    policy: Policy,
    started: Instant,
) -> Vec<Option<(DiagnosticContext, Instant)>> {
    (0..2)
        .map(|job| {
            Some((
                DiagnosticContext {
                    policy,
                    trace: handle.clone(),
                    job,
                    request_id: Some(uuid::Uuid::new_v4()),
                },
                started,
            ))
        })
        .collect()
}
fn events(handle: &ImageContinuousTraceHandle) -> Vec<Value> {
    let mut bytes = vec![];
    handle.write_jsonl(&mut bytes).unwrap();
    bytes
        .split(|&b| b == b'\n')
        .filter(|line| !line.is_empty())
        .map(|line| serde_json::from_slice::<Value>(line).unwrap())
        .filter_map(|mut value| value.as_object_mut().unwrap().remove("event"))
        .collect()
}
fn observations(handle: &ImageContinuousTraceHandle) -> Vec<Value> {
    events(handle)
        .into_iter()
        .filter(|event| event["kind"] == "prompt_expansion_observed")
        .collect()
}

#[tokio::test]
async fn mtp_diagnostics_actual_roles_main_frontiers_and_capture_match_original_outputs() {
    let fixture = Fixture::new();
    let (composition, runtime) = composition();
    let host = host();
    let (golden, answers) = ordinary(&fixture, &host, &composition).await;
    let started = Instant::now();
    let mut machine = fixture.machine(&composition, &host);
    let (recorder, handle) = ImageContinuousTraceRecorder::new(
        &host,
        ImageContinuousTraceLimits {
            maximum_bytes: 65536,
        },
    )
    .unwrap();
    let original = contexts(
        &handle,
        Policy::CaptureEffectivePrompt {
            maximum_utf8_bytes: 4096,
        },
        started,
    );
    machine.set_diagnostics(original.clone()).unwrap();
    assert!(machine.set_diagnostics(original.clone()).is_err());
    let (_, actual) = complete(&mut machine, false).await;
    assert_eq!(actual.tokens, golden);
    assert_eq!(runtime.resident_bytes(), 0);
    assert_eq!(
        host.usage().unwrap().committed_bytes(),
        handle.status().backing_bytes
    );
    let observations = observations(&handle);
    assert_eq!(observations.len(), answers.len());
    for (index, event) in observations.iter().enumerate() {
        assert_eq!(event["job"], index);
        assert_eq!(
            event["request_id"],
            serde_json::to_value(original[index].as_ref().unwrap().0.request_id).unwrap()
        );
        assert_eq!(event["capture"]["text"], answers[index]);
        assert_eq!(event["executor_retired"], true);
        assert_eq!(
            event["token_history"]["output_tokens_including_eos"],
            golden[index].len()
        );
        assert_eq!(event["token_history"]["last_token_id"], EOS_ID);
        let metrics = &event["executor_metrics"]["mtp"];
        for (role, actual_count) in [
            ("prefill", actual.prefill),
            ("draft", actual.draft),
            ("verify", actual.verify),
            ("last", actual.last),
        ] {
            assert_eq!(metrics[role]["computed_waves"], actual_count);
            assert!(metrics[role]["poll_wall_us"].is_u64());
            assert!(metrics[role]["settlement_wall_us"].is_u64());
        }
        assert_eq!(metrics["prefill"]["maximum_participants"], 2);
        let request = &metrics["request"];
        assert_eq!(
            request["verified_proposals"].as_u64().unwrap(),
            request["accepted_proposals"].as_u64().unwrap()
                + request["rejected_proposals"].as_u64().unwrap()
        );
    }
    let mut after = [0u64; 2];
    for event in events(&handle)
        .iter()
        .filter(|event| event["kind"] == "prompt_expansion_token_wave")
    {
        let job = event["job"].as_u64().unwrap() as usize;
        let before = event["before_consumed_tokens"].as_u64().unwrap();
        assert!(
            before >= after[job],
            "draft's independent KV must not regress the main trace"
        );
        after[job] = event["after_consumed_tokens"].as_u64().unwrap();
        assert!(after[job] > before);
    }
    drop(original);
    drop(machine);
    drop(handle);
    drop(recorder);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}

#[tokio::test]
async fn mtp_diagnostics_wait_for_checked_retirement_and_keep_summary_private() {
    let fixture = Fixture::new();
    let (composition, runtime) = composition();
    let host = host();
    let mut machine = fixture.machine(&composition, &host);
    let (recorder, handle) = ImageContinuousTraceRecorder::new(
        &host,
        ImageContinuousTraceLimits {
            maximum_bytes: 65536,
        },
    )
    .unwrap();
    assert!(machine.set_diagnostics(vec![]).is_err());
    machine
        .set_diagnostics(contexts(&handle, Policy::Summary, Instant::now()))
        .unwrap();
    let held = Arc::clone(machine.programs.main().unwrap().resources());
    tokio::time::timeout(
        Duration::from_secs(10),
        std::future::poll_fn(
            |cx| match machine.advance(&[false, false], cx.waker()).unwrap() {
                ExpansionCohortProgress::Pending { made_progress } => {
                    if machine.results.iter().all(Option::is_some) {
                        return Poll::Ready(());
                    }
                    if made_progress {
                        cx.waker().wake_by_ref();
                    }
                    Poll::Pending
                }
                ExpansionCohortProgress::Retired(_) => panic!("original held plan has not retired"),
            },
        ),
    )
    .await
    .unwrap();
    assert!(observations(&handle).is_empty());
    drop(held);
    assert!(matches!(
        machine
            .advance(&[false, false], &futures::task::noop_waker())
            .unwrap(),
        ExpansionCohortProgress::Retired(_)
    ));
    let observations = observations(&handle);
    assert_eq!(observations.len(), 2);
    for event in observations {
        assert_eq!(event["executor_retired"], true);
        assert_eq!(event["capture"]["status"], "disabled");
        assert!(event["capture"].get("text").is_none());
    }
    drop(machine);
    drop(handle);
    drop(recorder);
    assert_eq!(runtime.resident_bytes(), 0);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}

#[tokio::test]
async fn mtp_diagnostics_cancellation_records_original_history_without_effective_text() {
    let fixture = Fixture::new();
    let (composition, runtime) = composition();
    let host = host();
    let mut machine = fixture.machine(&composition, &host);
    let (recorder, handle) = ImageContinuousTraceRecorder::new(
        &host,
        ImageContinuousTraceLimits {
            maximum_bytes: 65536,
        },
    )
    .unwrap();
    machine
        .set_diagnostics(contexts(
            &handle,
            Policy::CaptureEffectivePrompt {
                maximum_utf8_bytes: 1,
            },
            Instant::now(),
        ))
        .unwrap();
    let (results, _) = complete(&mut machine, true).await;
    assert!(matches!(results[0], ExpansionParticipantResult::Cancelled));
    let observations = observations(&handle);
    assert_eq!(observations[0]["outcome"], "cancelled");
    assert_eq!(observations[0]["token_history"]["finish"], "Cancelled");
    assert_eq!(observations[0]["capture"]["status"], "unavailable");
    assert!(observations[0]["effective_prompt_sha256"].is_null());
    assert_eq!(observations[1]["outcome"], "expanded");
    assert_eq!(observations[1]["capture"]["status"], "capacity_refused");
    drop(machine);
    drop(handle);
    drop(recorder);
    assert_eq!(runtime.resident_bytes(), 0);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}
