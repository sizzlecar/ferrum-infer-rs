use super::*;
use ferrum_interfaces::vnext::{DeviceTimingMode, ExecutionEventSinkEnablement};

#[test]
fn host_and_basic_metrics_only_attach_to_target_and_draft_without_journals() {
    for entrypoint in [ProfileEntrypoint::Run, ProfileEntrypoint::Serve] {
        for detail in [
            ObservabilityProfileDetail::Off,
            ObservabilityProfileDetail::Basic,
            ObservabilityProfileDetail::Host,
        ] {
            let mut config = EngineConfig::default();
            config.runtime.profile_detail = detail;
            config.runtime.profile_entrypoint = Some(entrypoint);
            let target = Arc::new(PlanRuntimeAdmissionTestExecutor::new(128));
            let draft = Arc::new(PlanRuntimeAdmissionTestExecutor::new(128));
            let engine = ContinuousBatchEngine::new_with_resource_composition(
                config.clone(),
                Arc::new(ContinuousBatchScheduler::new(config.scheduler)),
                Arc::new(ferrum_testkit::MockTokenizer::new(128)),
                Arc::new(ferrum_testkit::MockSampler),
                EngineResourceComposition::PlanRuntime,
                target.clone(),
                Arc::new(MockTensorFactory),
                Some(draft.clone()),
                Some(crate::speculative::SpeculativeDecodingConfig::default()),
            )
            .unwrap();
            assert!(engine.inner.profile_trace_jsonl.is_none());
            assert!(engine.inner.scheduler_trace_jsonl.is_none());
            for executor in [&target, &draft] {
                let attached = executor.event_sink.lock().unwrap();
                if detail == ObservabilityProfileDetail::Off {
                    assert!(attached.is_none());
                    continue;
                }
                let sink = attached
                    .as_ref()
                    .expect("metrics-only detail must attach timing observer");
                assert_eq!(sink.enablement(), ExecutionEventSinkEnablement::None);
                assert_eq!(
                    sink.device_timing_mode(),
                    if detail == ObservabilityProfileDetail::Host {
                        DeviceTimingMode::Off
                    } else {
                        DeviceTimingMode::Completion
                    }
                );
                assert!(sink.host_dispatch_timing_enabled());
                assert!(!sink.needs_structured_cost_sample());
                assert!(!sink.cost_observation_demand().is_required());
                assert_eq!(
                    sink.capture_policy(),
                    ExecutionEventCapturePolicy::AllFrames
                );
                assert!(!sink.records_execution_resource_maintenance());
                assert!(!sink.records_prefix_restore_decisions());
                assert!(!sink.is_enabled(VNextExecutionEventKind::RequestAccepted));
                assert!(!sink.is_enabled(VNextExecutionEventKind::NodeStarted));
            }
        }
    }
}

#[test]
fn host_metrics_only_rejects_journals_before_opening_or_attaching() {
    for journal in 0..3 {
        let path = resource_trace_temp_path("host-rejected-journal");
        let mut config = EngineConfig::default();
        config.runtime.profile_detail = ObservabilityProfileDetail::Host;
        match journal {
            0 => config.runtime.profile_jsonl = Some(path.clone()),
            1 => config.runtime.scheduler_trace_jsonl = Some(path.clone()),
            _ => config.runtime.legacy_scheduler_trace_jsonl = Some(path.clone()),
        }
        let target = Arc::new(PlanRuntimeAdmissionTestExecutor::new(128));
        let result = ContinuousBatchEngine::new_plan_runtime(
            config.clone(),
            Arc::new(ContinuousBatchScheduler::new(config.scheduler)),
            Arc::new(ferrum_testkit::MockTokenizer::new(128)),
            Arc::new(ferrum_testkit::MockSampler),
            target.clone(),
            Arc::new(MockTensorFactory),
        );
        assert!(matches!(result, Err(FerrumError::Config { .. })));
        assert!(target.event_sink.lock().unwrap().is_none());
        assert!(!path.exists());
    }
}

#[test]
fn basic_with_artifact_keeps_event_sink_and_completion_timing() {
    let path = resource_trace_temp_path("basic-artifact-sink");
    let mut config = EngineConfig::default();
    config.runtime.profile_detail = ObservabilityProfileDetail::Basic;
    config.runtime.profile_jsonl = Some(path.clone());
    let target = Arc::new(PlanRuntimeAdmissionTestExecutor::new(128));
    let engine = ContinuousBatchEngine::new_plan_runtime(
        config.clone(),
        Arc::new(ContinuousBatchScheduler::new(config.scheduler)),
        Arc::new(ferrum_testkit::MockTokenizer::new(128)),
        Arc::new(ferrum_testkit::MockSampler),
        target.clone(),
        Arc::new(MockTensorFactory),
    )
    .unwrap();
    {
        let attached = target.event_sink.lock().unwrap();
        let sink = attached.as_ref().unwrap();
        assert_eq!(sink.enablement(), ExecutionEventSinkEnablement::All);
        assert_eq!(sink.device_timing_mode(), DeviceTimingMode::Completion);
        assert!(sink.records_execution_resource_maintenance());
        assert!(!sink.cost_observation_demand().is_required());
        assert!(!sink.needs_structured_cost_sample());
    }
    assert!(engine.inner.profile_trace_jsonl.is_some());
    drop(engine);
    drop(target);
    let _ = std::fs::remove_file(path);
}
