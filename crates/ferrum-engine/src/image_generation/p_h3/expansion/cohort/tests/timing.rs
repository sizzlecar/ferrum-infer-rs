//! Real stored CPU execution through the single-request adapter. These test
//! typed timing propagation and lifecycle, not CUDA event cost or performance.
use super::*;
use crate::image_generation::continuous_trace::{
    ImageContinuousTraceLimits, ImageContinuousTraceRecorder,
};
use crate::image_generation::p_h3::expansion::{
    phase::{ExpansionPhaseProgress, NativeExpansionPhase},
    DiagnosticContext, PH3ImagePromptExpansionDiagnosticPolicy,
};
use ferrum_runtime::DeviceTimingMode;

#[tokio::test]
async fn expansion_kernel_timing_single_request_preserves_answer_and_refunds() {
    for mode in [DeviceTimingMode::Off, DeviceTimingMode::Kernel] {
        let mut fixture = Fixture::new();
        let assets = fixture.take_expansion();
        let mut request = prepared(&assets, 2);
        assert_eq!(request.device_timing, DeviceTimingMode::Off);
        assets.configure_request_timing(&mut request, mode).unwrap();
        let (composition, runtime) = composition();
        let host = host();
        let (recorder, trace) = ImageContinuousTraceRecorder::new(
            &host,
            ImageContinuousTraceLimits {
                maximum_bytes: 256 << 10,
            },
        )
        .unwrap();
        let requirements = assets
            .host_requirements(&composition, &request, &host)
            .unwrap();
        let phase = host.reserve_scope(&requirements.claims()).unwrap();
        let mut owner = NativeExpansionPhase::initialize_with_diagnostics(
            assets,
            &composition,
            request,
            phase,
            Some(DiagnosticContext {
                policy: PH3ImagePromptExpansionDiagnosticPolicy::Summary,
                trace: trace.clone(),
                job: 1,
                request_id: None,
            }),
        )
        .unwrap();
        let result = tokio::time::timeout(
            std::time::Duration::from_secs(10),
            std::future::poll_fn(|cx| match owner.advance(false, cx.waker()) {
                Ok(ExpansionPhaseProgress::Pending { made_progress }) => {
                    if made_progress {
                        cx.waker().wake_by_ref();
                    }
                    Poll::Pending
                }
                other => Poll::Ready(other),
            }),
        )
        .await
        .expect("original retirement waiter must complete")
        .unwrap();
        let ExpansionPhaseProgress::Expanded(answer) = result else {
            panic!("original EOS answer must complete")
        };
        assert_eq!(answer, "A small red cat sits quietly.");
        assert_eq!(
            owner.cleanup().snapshot().phase(),
            ExecutorShutdownPhase::Retired
        );
        assert_eq!(runtime.resident_bytes(), 0);
        drop(owner);
        recorder.seal();
        let mut bytes = Vec::new();
        let status = trace.write_jsonl(&mut bytes).unwrap();
        assert!(!status.truncated && status.dropped == 0);
        let events = bytes
            .split(|&byte| byte == b'\n')
            .filter(|line| !line.is_empty())
            .map(|line| serde_json::from_slice::<serde_json::Value>(line).unwrap())
            .filter_map(|mut record| record.as_object_mut()?.remove("event"))
            .collect::<Vec<_>>();
        assert!(
            events.iter().all(|event| matches!(
                event["kind"].as_str(),
                Some("prompt_expansion_token_wave" | "prompt_expansion_observed")
            )),
            "metrics-only timing must not open an execution journal"
        );
        let final_event = events
            .iter()
            .find(|event| event["kind"] == "prompt_expansion_observed")
            .unwrap();
        assert_eq!(final_event["outcome"], "expanded");
        let metrics = &final_event["executor_metrics"];
        let samples = metrics["wave_timing_by_phase"]["decode"]["host_encode_submit_breakdown"]
            ["provider_encode_submit"]["samples"]
            .as_u64()
            .unwrap();
        assert_eq!(samples > 0, mode == DeviceTimingMode::Kernel);
        // CPU explicitly reports an unavailable device clock for requested
        // timing. Off produces no such observation, rather than a fake zero.
        let unavailable = metrics["device_timing_by_phase"]["decode"]["unavailable"]
            ["device_execution"]
            .as_u64()
            .unwrap();
        assert_eq!(unavailable > 0, mode == DeviceTimingMode::Kernel);
        drop((events, trace, recorder));
        assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    }
}

#[test]
fn expansion_kernel_timing_rejects_unsupported_and_mixed_policies_before_installation() {
    let mut fixture = Fixture::new_two_answers();
    let mut assets = fixture.take_expansion();
    let mut rows = requests(&assets, 2);
    for mode in [
        DeviceTimingMode::Completion,
        DeviceTimingMode::Replay,
        DeviceTimingMode::Verification,
    ] {
        assert!(assets.configure_request_timing(&mut rows[0], mode).is_err());
        assert_eq!(rows[0].device_timing, DeviceTimingMode::Off);
    }
    assets
        .configure_request_timing(&mut rows[0], DeviceTimingMode::Kernel)
        .unwrap();
    assert!(assets.prepare_text_cohort(rows).is_err());
    Arc::get_mut(&mut assets)
        .unwrap()
        .configure_mtp(PH3ImagePromptExpansionMtp::OneDraft);
    let mut request = prepared(&assets, 2);
    assert!(assets
        .configure_request_timing(&mut request, DeviceTimingMode::Kernel)
        .is_err());
    assert_eq!(request.device_timing, DeviceTimingMode::Off);
}
