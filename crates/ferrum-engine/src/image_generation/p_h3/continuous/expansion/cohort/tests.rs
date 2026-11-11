use super::*;
use crate::image_generation::continuous_trace::{
    ImageContinuousTraceEvent, ImageContinuousTraceHandle, ImageContinuousTraceLimits,
    ImageContinuousTraceRecorder, ImageTracePhase, ImageTracePhaseOutcome,
};
use ferrum_runtime::{HostMemoryClassLimits, HostMemoryLimits};

fn trace_fixture() -> (
    HostMemoryAccount,
    ImageContinuousTraceRecorder,
    ImageContinuousTraceHandle,
) {
    let host = HostMemoryAccount::new(
        HostMemoryLimits::new(
            65536,
            HostMemoryClassLimits {
                input_preprocess_bytes: 0,
                transfer_staging_bytes: 0,
                output_bytes: 65536,
                weight_backing_bytes: 0,
            },
        )
        .unwrap(),
    );
    let (recorder, handle) = ImageContinuousTraceRecorder::new(
        &host,
        ImageContinuousTraceLimits {
            maximum_bytes: 16384,
        },
    )
    .unwrap();
    (host, recorder, handle)
}

fn traced_members(
    requests: &[ImageGenerationRequest],
    handle: &ImageContinuousTraceHandle,
) -> Vec<Member> {
    requests
        .iter()
        .enumerate()
        .map(|(index, request)| {
            let correlation = ferrum_types::ImageRequestCorrelation::new(None);
            Member {
                request_id: correlation.request_id,
                correlation: Some(correlation),
                identity: IngressIdentity::new(ImageRequestRef::Generate(request)),
                job: index as u64,
                output_credit: 0,
                ingress_backing_bytes: 0,
                prompt_trace: handle.record(&ImageContinuousTraceEvent::PhaseStarted {
                    request_id: Some(correlation.request_id),
                    request: None,
                    phase: ImageTracePhase::PromptExpansion,
                }),
            }
        })
        .collect()
}

fn trace_records(handle: &ImageContinuousTraceHandle) -> Vec<serde_json::Value> {
    let mut bytes = Vec::new();
    let status = handle.write_jsonl(&mut bytes).unwrap();
    assert!(!status.truncated && status.dropped == 0);
    bytes
        .split(|&byte| byte == b'\n')
        .filter(|line| !line.is_empty())
        .map(|line| serde_json::from_slice::<serde_json::Value>(line).unwrap())
        .filter(|record| record.get("event").is_some())
        .collect()
}

fn request() -> ImageGenerationRequest {
    ImageGenerationRequest {
        prompt: "cat".into(),
        width: 32,
        height: 32,
        seed: 7,
        n: 1,
    }
}

#[test]
fn retired_prompt_spans_share_one_boundary_before_downstream_and_keep_member_outcomes() {
    let (host, recorder, handle) = trace_fixture();
    let requests = vec![request(); 4];
    let mut members = traced_members(&requests, &handle);
    let starts = members
        .iter()
        .map(|member| member.prompt_trace.unwrap())
        .collect::<Vec<_>>();
    let views = members
        .iter()
        .zip(&requests)
        .enumerate()
        .map(
            |(index, (member, request))| ImageInitializationCohortRequest {
                request_id: member.request_id,
                request: ImageRequestRef::Generate(request),
                cancelled: index == 3,
            },
        )
        .collect::<Vec<_>>();
    let failed = || {
        ExpansionParticipantResult::Failed(ferrum_types::FerrumError::backend(
            "original expansion failure",
        ))
    };
    let outcomes = vec![
        ExpansionParticipantResult::Expanded("a cat".into()),
        ExpansionParticipantResult::Cancelled,
        failed(),
        failed(),
    ];
    let retired_at = Instant::now();
    finish_retired_prompt_spans(
        &mut members,
        &views,
        &outcomes,
        retired_at,
        |start, outcome, at| {
            crate::image_generation::p_h3::continuous::trace::record_phase_finish(
                Some(&recorder),
                start,
                outcome,
                at,
            );
        },
    );
    assert!(members.iter().all(|member| member.prompt_trace.is_none()));
    // Model the next phase's independent failure using the same real recorder.
    // It must neither extend nor reclassify a completed prompt-expansion span.
    let downstream = handle
        .record(&ImageContinuousTraceEvent::PhaseStarted {
            request_id: Some(members[0].request_id),
            request: None,
            phase: ImageTracePhase::InitializationAdvance,
        })
        .unwrap();
    let downstream_at = Instant::now();
    crate::image_generation::p_h3::continuous::trace::record_phase_finish(
        Some(&recorder),
        Some(downstream),
        ImageTracePhaseOutcome::Failed,
        downstream_at,
    );
    let count_before_retry = handle.status().recorded;
    let retry = (0..members.len()).map(|_| failed()).collect::<Vec<_>>();
    finish_retired_prompt_spans(
        &mut members,
        &views,
        &retry,
        downstream_at,
        |start, outcome, at| {
            crate::image_generation::p_h3::continuous::trace::record_phase_finish(
                Some(&recorder),
                start,
                outcome,
                at,
            );
        },
    );
    assert_eq!(handle.status().recorded, count_before_retry);
    recorder.seal();
    let records = trace_records(&handle);
    let expected = ["completed", "cancelled", "failed", "cancelled"];
    let mut boundary = None;
    for (start, expected) in starts.iter().zip(expected) {
        let finishes = records
            .iter()
            .filter(|record| record["event"]["start_sequence"].as_u64() == Some(*start))
            .collect::<Vec<_>>();
        assert_eq!(finishes.len(), 1);
        let finish = finishes[0];
        assert_eq!(finish["event"]["outcome"], expected);
        let event_ns = finish["event_ns"].as_u64().unwrap();
        assert_eq!(*boundary.get_or_insert(event_ns), event_ns);
        assert!(finish["sequence"].as_u64().unwrap() < downstream);
        let downstream_start = records
            .iter()
            .find(|record| record["sequence"].as_u64() == Some(downstream))
            .unwrap();
        assert!(event_ns <= downstream_start["event_ns"].as_u64().unwrap());
    }
    drop((recorder, handle));
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}

#[test]
fn retired_prompt_spans_fail_every_original_member_when_result_extent_is_invalid() {
    for (request_count, outcome_count) in [(2, 0), (2, 1), (2, 3), (1, 2)] {
        let (host, recorder, handle) = trace_fixture();
        let requests = vec![request(); 2];
        let mut members = traced_members(&requests, &handle);
        let starts = members
            .iter()
            .map(|member| member.prompt_trace.unwrap())
            .collect::<Vec<_>>();
        let views = members
            .iter()
            .zip(&requests)
            .take(request_count)
            .map(|(member, request)| ImageInitializationCohortRequest {
                request_id: member.request_id,
                request: ImageRequestRef::Generate(request),
                cancelled: true,
            })
            .collect::<Vec<_>>();
        let outcomes = (0..outcome_count)
            .map(|_| ExpansionParticipantResult::Expanded("unattributable result".into()))
            .collect::<Vec<_>>();
        finish_retired_prompt_spans(
            &mut members,
            &views,
            &outcomes,
            Instant::now(),
            |start, outcome, at| {
                crate::image_generation::p_h3::continuous::trace::record_phase_finish(
                    Some(&recorder),
                    start,
                    outcome,
                    at,
                );
            },
        );
        assert!(members.iter().all(|member| member.prompt_trace.is_none()));
        recorder.seal();
        let records = trace_records(&handle);
        let finishes = records
            .iter()
            .filter(|record| record["event"]["kind"] == "phase_finished")
            .collect::<Vec<_>>();
        assert_eq!(finishes.len(), starts.len());
        for (finish, start) in finishes.iter().zip(starts) {
            assert_eq!(finish["event"]["start_sequence"], start);
            assert_eq!(finish["event"]["outcome"], "failed");
            assert_eq!(finish["event_ns"], finishes[0]["event_ns"]);
        }
        drop((recorder, handle));
        assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    }
}

#[test]
fn sealed_expansion_members_reject_reorder_replacement_and_changed_ingress() {
    let pipeline = Arc::new(());
    let requests = [
        ImageGenerationRequest {
            prompt: "cat".into(),
            width: 32,
            height: 32,
            seed: 7,
            n: 1,
        },
        ImageGenerationRequest {
            prompt: "dog".into(),
            width: 32,
            height: 64,
            seed: 11,
            n: 1,
        },
    ];
    let members = requests
        .iter()
        .enumerate()
        .map(|(index, request)| Member {
            request_id: Uuid::new_v4(),
            correlation: None,
            identity: IngressIdentity::new(ImageRequestRef::Generate(request)),
            job: index as u64,
            output_credit: 0,
            ingress_backing_bytes: 0,
            prompt_trace: None,
        })
        .collect::<Vec<_>>();
    let mut views = members
        .iter()
        .zip(&requests)
        .map(|(member, request)| ImageInitializationCohortRequest {
            request_id: member.request_id,
            request: ImageRequestRef::Generate(request),
            cancelled: false,
        })
        .collect::<Vec<_>>();
    assert!(validate_members(&pipeline, &pipeline, &members, &views).is_ok());
    views[0].cancelled = true;
    assert!(validate_members(&pipeline, &pipeline, &members, &views).is_ok());
    views.swap(0, 1);
    assert!(validate_members(&pipeline, &pipeline, &members, &views).is_err());
    views.swap(0, 1);
    views[0].request_id = Uuid::new_v4();
    assert!(validate_members(&pipeline, &pipeline, &members, &views).is_err());
    views[0].request_id = members[0].request_id;
    let mut altered = requests[0].clone();
    altered.prompt = "a replaced prompt".into();
    views[0].request = ImageRequestRef::Generate(&altered);
    assert!(validate_members(&pipeline, &pipeline, &members, &views).is_err());
    views[0].request = ImageRequestRef::Generate(&requests[0]);
    assert!(validate_members(&pipeline, &Arc::new(()), &members, &views).is_err());
    assert!(validate_members(&pipeline, &pipeline, &members, &views[..1]).is_err());
}

mod controller;
