use super::*;

#[test]
fn prompt_expansion_diagnostics_policy_is_explicit_bounded_and_rejects_unknown_controls() {
    assert_eq!(
        PH3ImagePromptExpansionDiagnosticPolicy::default(),
        PH3ImagePromptExpansionDiagnosticPolicy::Disabled
    );
    for bytes in [0, PH3_IMAGE_PROMPT_EXPANSION_MAX_CAPTURE_BYTES + 1] {
        assert!(
            PH3ImagePromptExpansionDiagnosticPolicy::CaptureEffectivePrompt {
                maximum_utf8_bytes: bytes
            }
            .validate()
            .is_err()
        );
    }
    PH3ImagePromptExpansionDiagnosticPolicy::CaptureEffectivePrompt {
        maximum_utf8_bytes: PH3_IMAGE_PROMPT_EXPANSION_MAX_CAPTURE_BYTES,
    }
    .validate()
    .unwrap();
    for wire in [
        r#"{"mode":"disabled","unknown":true}"#,
        r#"{"mode":"summary","maximum_utf8_bytes":1}"#,
        r#"{"mode":"summary","thinking":true}"#,
        r#"{"mode":"capture_effective_prompt"}"#,
        r#"{"mode":"capture_effective_prompt","maximum_utf8_bytes":-1}"#,
    ] {
        assert!(serde_json::from_str::<PH3ImagePromptExpansionDiagnosticPolicy>(wire).is_err());
    }
}

#[test]
fn prompt_expansion_vision_receipts_preserve_producer_identity_in_original_trace_backing() {
    use crate::image_generation::continuous_trace::{
        ImageContinuousTraceLimits, ImageContinuousTraceRecorder,
    };
    use ferrum_runtime::{HostMemoryAccount, HostMemoryClassLimits, HostMemoryLimits};

    let host = HostMemoryAccount::new(
        HostMemoryLimits::new(
            65536,
            HostMemoryClassLimits {
                input_preprocess_bytes: 0,
                output_bytes: 65536,
                transfer_staging_bytes: 0,
                weight_backing_bytes: 0,
            },
        )
        .unwrap(),
    );
    let (recorder, handle) = ImageContinuousTraceRecorder::new(
        &host,
        ImageContinuousTraceLimits {
            maximum_bytes: 65536,
        },
    )
    .unwrap();
    let producer = Uuid::new_v4();
    let receipts = [
        None,
        Some(VisionBindingReceipt {
            reused: false,
            producer_job: 19,
            producer_request_id: Some(producer),
            output_bytes: 8192,
        }),
        Some(VisionBindingReceipt {
            reused: true,
            producer_job: 19,
            producer_request_id: Some(producer),
            output_bytes: 8192,
        }),
    ];
    for (index, receipt) in receipts.iter().copied().enumerate() {
        let mut observation = PhaseObservation::new(
            DiagnosticContext {
                policy: PH3ImagePromptExpansionDiagnosticPolicy::Summary,
                trace: handle.clone(),
                job: if index == 2 { 20 } else { 19 },
                request_id: Some(if index == 2 { Uuid::new_v4() } else { producer }),
            },
            8,
            2,
            Instant::now(),
        )
        .with_vision_binding(receipt);
        observation.tokens(&[9, 2], FinishReason::EOS);
        observation.retiring(None);
        observation.record(true, "expanded", Some("Observed answer."));
    }
    recorder.seal();
    let mut bytes = Vec::new();
    let status = handle.write_jsonl(&mut bytes).unwrap();
    assert!(!status.truncated);
    let events = bytes
        .split(|&b| b == b'\n')
        .filter(|line| !line.is_empty())
        .map(|line| serde_json::from_slice::<Value>(line).unwrap())
        .filter_map(|mut value| value.as_object_mut().unwrap().remove("event"))
        .collect::<Vec<_>>();
    assert_eq!(events.len(), receipts.len());
    assert!(events[0].get("vision_binding").is_none());
    for (event, receipt) in events[1..].iter().zip(&receipts[1..]) {
        assert_eq!(
            event["vision_binding"],
            serde_json::to_value(receipt).unwrap()
        );
    }
    assert_eq!(
        host.usage().unwrap().committed_bytes(),
        status.backing_bytes
    );
    drop(handle);
    drop(recorder);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}
