use super::*;
use ferrum_execution::test_support::{assert_executor_closed, token_execution};
use ferrum_kernels::backend::cpu::vnext_ops::CpuVNextComposition;
use ferrum_models::test_support::native_program::{Fixture, FixtureOptions};
use ferrum_runtime::{DeviceId, DeviceTimingMode, ExecutionFactsCapture};
use ferrum_types::SamplingReadbackMode;

#[test]
fn execution_facts_are_opt_in_without_changing_the_timing_preset() {
    let directory = tempfile::tempdir().unwrap();
    for detail in [
        ObservabilityProfileDetail::Off,
        ObservabilityProfileDetail::Basic,
        ObservabilityProfileDetail::Resource,
        ObservabilityProfileDetail::Latency,
        ObservabilityProfileDetail::Kernel,
        ObservabilityProfileDetail::Debug,
        ObservabilityProfileDetail::Replay,
        ObservabilityProfileDetail::Verify,
        ObservabilityProfileDetail::Full,
    ] {
        let path = directory.path().join(detail.as_str());
        let journal = create_scheduler_trace_sink(Some(&path)).unwrap();
        let mut config = EngineConfig::default();
        config.runtime.profile_detail = detail;
        let sink =
            VNextProfileExecutionEventSink::new(journal.clone(), ProfileEntrypoint::Run, &config);
        let (facts, timing) = match detail {
            ObservabilityProfileDetail::Off
            | ObservabilityProfileDetail::Resource
            | ObservabilityProfileDetail::Latency => {
                (ExecutionFactsCapture::Off, DeviceTimingMode::Off)
            }
            ObservabilityProfileDetail::Basic => {
                (ExecutionFactsCapture::Off, DeviceTimingMode::Completion)
            }
            ObservabilityProfileDetail::Debug => {
                (ExecutionFactsCapture::On, DeviceTimingMode::Completion)
            }
            ObservabilityProfileDetail::Replay => {
                (ExecutionFactsCapture::On, DeviceTimingMode::Replay)
            }
            ObservabilityProfileDetail::Verify => {
                (ExecutionFactsCapture::On, DeviceTimingMode::Verification)
            }
            ObservabilityProfileDetail::Kernel | ObservabilityProfileDetail::Full => {
                (ExecutionFactsCapture::On, DeviceTimingMode::Kernel)
            }
        };
        assert_eq!(sink.execution_facts_capture(), facts);
        assert_eq!(sink.device_timing_mode(), timing);
        drop(sink);
        journal.close().unwrap();
    }
    assert_eq!(
        MetricsOnlyExecutionEventSink.execution_facts_capture(),
        ExecutionFactsCapture::Off
    );
}

fn read_facts(path: &Path) -> Vec<FerrumProfileEvent> {
    std::fs::read_to_string(path)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str::<FerrumProfileEvent>(line).unwrap())
        .filter(|event| event.phase == "vnext.execution_facts")
        .collect()
}

#[tokio::test]
async fn execution_facts_real_cpu_work_reaches_run_and_serve_journals() {
    for entrypoint in [ProfileEntrypoint::Run, ProfileEntrypoint::Serve] {
        for detail in [
            ObservabilityProfileDetail::Debug,
            ObservabilityProfileDetail::Basic,
        ] {
            let directory = tempfile::tempdir().unwrap();
            let profile_path = directory.path().join("profile.jsonl");
            let trace_path = directory.path().join("trace.jsonl");
            let profile = create_scheduler_trace_sink(Some(&profile_path)).unwrap();
            let trace = create_scheduler_trace_sink(Some(&trace_path)).unwrap();
            let mut config = EngineConfig::default();
            config.runtime.profile_detail = detail;
            let sink = Arc::new(VNextProfileExecutionEventSink::with_journals(
                vec![profile.clone(), trace.clone()],
                entrypoint,
                &config,
            ));
            let (runtime, registry, materializers, _, catalog) = CpuVNextComposition::create(
                DeviceId::new("device.test.execution-facts-profile").unwrap(),
                256 << 20,
            )
            .unwrap()
            .into_parts()
            .unwrap();
            let fixture = Fixture::new(
                SamplingReadbackMode::Uniform,
                Device::CPU,
                runtime,
                registry,
                materializers,
                catalog,
                FixtureOptions::default(),
            );
            fixture.executor.attach_execution_event_sink(sink.clone());
            // The existing numerical assertions execute partial prefill, final
            // prefill and decode, check full logits, and close the real owner.
            token_execution::native_tokens_cpu_partial_final_decode_and_terminal_receipt(
                &fixture.executor,
            )
            .await;
            let runtime = assert_executor_closed(fixture.executor);
            assert_eq!(runtime.resident_bytes(), 0);
            drop(sink);
            profile.close().unwrap();
            trace.close().unwrap();
            let events = read_facts(&profile_path);
            assert_eq!(events, read_facts(&trace_path));
            if detail == ObservabilityProfileDetail::Basic {
                assert!(events.is_empty());
                continue;
            }
            assert_eq!(events.len(), 3, "the three actual submitted waves");
            for (ordinal, event) in events.iter().enumerate() {
                event.validate().unwrap();
                assert_eq!(event.entrypoint, entrypoint);
                assert_eq!(event.status, ProfileStatus::Ok);
                assert!(event.duration_us.is_none(), "do not invent device time");
                assert_eq!(
                    event.attributes["completion_disposition"]["status"],
                    "succeeded"
                );
                assert_eq!(
                    event.attributes["device_facts_coverage"],
                    serde_json::json!({"status":"unavailable", "reason":"not_captured"})
                );
                let nodes = event.attributes["submitted_node_work"].as_array().unwrap();
                assert!(!nodes.is_empty());
                for (index, node) in nodes.iter().enumerate() {
                    assert_eq!(node["node_index"], index);
                    let participants = node["participants"].as_array().unwrap();
                    assert_eq!(participants.len(), 1);
                    let participant = &participants[0];
                    assert_eq!(participant["participant_index"], 0);
                    assert_eq!(participant["request_id"], event.request_id);
                    assert_eq!(
                        participant["source_tokens"],
                        serde_json::json!({"start":ordinal, "end":ordinal + 1})
                    );
                    assert_eq!(
                        participant["packed_tokens"],
                        serde_json::json!({"start":0, "end":1})
                    );
                    assert_eq!(
                        participant["full_input_tokens"],
                        if ordinal < 2 { 2 } else { 3 }
                    );
                }
            }
        }
    }
}
