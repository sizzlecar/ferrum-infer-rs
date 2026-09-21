use super::*;
use ferrum_interfaces::vnext::{
    DeviceTimingMode, EventEmissionPermit, ExecutionEventKind, ExecutionEventSink,
    ExecutionEventSinkEnablement, ExecutionEventSinkError, OperationCompletionReceipt,
};
use ferrum_interfaces::InferenceEngine;
use ferrum_types::{ObservabilityProfileDetail, ProfileEntrypoint};
use std::collections::BTreeSet;
use std::sync::atomic::{AtomicUsize, Ordering};

#[tokio::test]
async fn real_cpu_profile_wave_phase_reaches_physical_events_without_device_clocks() {
    for (detail, entrypoint) in [
        (ObservabilityProfileDetail::Kernel, ProfileEntrypoint::Run),
        (ObservabilityProfileDetail::Replay, ProfileEntrypoint::Serve),
    ] {
        let fixture = CpuFixture::new(8).await;
        let trace = fixture.directory.join("wave-profile.jsonl");
        let mut config = qwen35_fixture_component_config(&fixture.directory).engine_config;
        config.runtime.profile_detail = detail;
        config.runtime.profile_entrypoint = Some(entrypoint);
        config.runtime.profile_jsonl = Some(trace.clone());
        let engine = crate::continuous_engine::ContinuousBatchEngine::new_plan_runtime(
            config.clone(),
            Arc::new(
                ferrum_scheduler::implementations::ContinuousBatchScheduler::new(config.scheduler),
            ),
            Arc::new(ferrum_testkit::MockTokenizer::new(3)),
            Arc::new(ferrum_testkit::MockSampler),
            fixture.executor.clone(),
            Arc::new(ferrum_testkit::MockTensorFactory),
        )
        .unwrap();

        // Prefill and decode each have exactly one token. Shape inference
        // cannot distinguish them; the actual model entrypoint must do so.
        let decode = fixture.seed_decode(&[0]).await;
        let outputs = split_decode(&fixture, std::slice::from_ref(&decode)).await;
        let next = PlanRuntimeDecodeInput::new(
            decode.request_id,
            TokenId::new(1),
            Arc::clone(&outputs[0].kv_cache),
        );
        let prefill = prompt(&[1], 1);
        fixture.admit(&prefill);
        let (prefills, decodes) = match fixture
            .executor
            .plan_runtime_mixed_batch_with_capacity(
                std::slice::from_ref(&prefill),
                std::slice::from_ref(&next),
            )
            .await
            .unwrap()
        {
            PlanRuntimeMixedBatchOutcome::Completed { prefills, decodes } => (prefills, decodes),
            _ => panic!("tiny mixed wave did not execute"),
        };
        assert_eq!(decodes[0].kv_cache.num_tokens(), 3);
        assert_eq!(prefills[0].output().committed_tokens(), 1);
        fixture
            .executor
            .release_cache(&decodes[0].kv_cache.cache_id());
        fixture
            .executor
            .release_cache(&prefills[0].output().kv_cache().cache_id());
        assert_eq!(
            fixture.executor.cache_metrics_snapshot().unwrap()["active_sequences"],
            0
        );
        engine.shutdown().await.unwrap(); // Closes and flushes the actual journal.

        let events = std::fs::read_to_string(&trace)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str::<serde_json::Value>(line).unwrap())
            .filter(|event| event["phase"] == "vnext.device_physical_submission")
            .collect::<Vec<_>>();
        assert!(!events.is_empty(), "missing physical submission events");
        let phases = events
            .iter()
            .map(|event| {
                assert!(event["attributes"]["physical_submission_fingerprint"]
                    .as_str()
                    .is_some());
                // CPU has no device counter support: phase survives unavailable timing.
                assert_eq!(event["attributes"]["device_timing_status"], "unavailable");
                event["attributes"]["wave_phase"].as_str().unwrap()
            })
            .collect::<BTreeSet<_>>();
        assert_eq!(phases, BTreeSet::from(["prefill", "decode", "mixed"]));
        for phase in ["prefill", "decode"] {
            assert!(events
                .iter()
                .any(|event| event["attributes"]["wave_phase"] == phase
                    && event["shape"]["participant_count"] == 1));
        }
    }
}

#[derive(Default)]
struct LegacyAttributionSink(AtomicUsize);

impl ExecutionEventSink for LegacyAttributionSink {
    fn enablement(&self) -> ExecutionEventSinkEnablement {
        ExecutionEventSinkEnablement::None
    }
    fn is_enabled(&self, _: ExecutionEventKind) -> bool {
        false
    }
    fn device_timing_mode(&self) -> DeviceTimingMode {
        DeviceTimingMode::Kernel
    }
    fn record(&self, _: EventEmissionPermit) -> std::result::Result<(), ExecutionEventSinkError> {
        Ok(())
    }
    fn record_physical_device_submission_timing(
        &self,
        completion: &OperationCompletionReceipt,
    ) -> std::result::Result<(), ExecutionEventSinkError> {
        assert!(!completion.submission().batch_identity().nodes().is_empty());
        self.0.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }
}

#[tokio::test]
async fn real_cpu_profile_wave_phase_preserves_legacy_sink_callback() {
    let fixture = CpuFixture::new(8).await;
    let sink = Arc::new(LegacyAttributionSink::default());
    fixture.executor.attach_execution_event_sink(sink.clone());
    let decode = fixture.seed_decode(&[0]).await;
    assert_eq!(sink.0.load(Ordering::Relaxed), 1);
    fixture.executor.release_cache(&decode.kv_cache.cache_id());
}
