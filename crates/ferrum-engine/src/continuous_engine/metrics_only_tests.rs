//! Assembly-only observer checks. The controlled executor supports the common
//! token protocol; no model execution or numerical timing is asserted here.

use super::*;
use ferrum_runtime::{DeviceTimingMode, ExecutionEventSinkEnablement};

#[test]
fn basic_metrics_only_attaches_to_target_without_journals() {
    for entrypoint in [ProfileEntrypoint::Run, ProfileEntrypoint::Serve] {
        for detail in [
            ObservabilityProfileDetail::Off,
            ObservabilityProfileDetail::Basic,
        ] {
            let mut config = EngineConfig::default();
            config.runtime.profile_detail = detail;
            config.runtime.profile_entrypoint = Some(entrypoint);
            let target = Arc::new(PlanRuntimeAdmissionTestExecutor::new(128));
            let engine = ContinuousBatchEngine::new_with_resource_composition(
                config.clone(),
                Arc::new(ContinuousBatchScheduler::new(config.scheduler)),
                Arc::new(ferrum_testkit::MockTokenizer::new(128)),
                Arc::new(ferrum_testkit::MockSampler),
                EngineResourceComposition::PlanRuntime,
                target.clone(),
                Arc::new(MockTensorFactory),
                None,
            )
            .unwrap();
            assert!(engine.inner.profile_trace_journal().is_none());
            assert!(engine.inner.scheduler_trace_jsonl.is_none());
            let attached = target.event_sink.lock().unwrap();
            if detail == ObservabilityProfileDetail::Off {
                assert!(attached.is_none());
            } else {
                let sink = attached
                    .as_ref()
                    .expect("basic must attach timing observer");
                assert_eq!(sink.enablement(), ExecutionEventSinkEnablement::None);
                assert_eq!(sink.device_timing_mode(), DeviceTimingMode::Completion);
                assert!(!sink.records_execution_resource_maintenance());
                assert!(!sink.records_prefix_restore_decisions());
                assert!(!sink.is_enabled(VNextExecutionEventKind::RequestAccepted));
                assert!(!sink.is_enabled(VNextExecutionEventKind::NodeStarted));
            }
        }
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
    }
    assert!(engine.inner.profile_trace_journal().is_some());
    drop(engine);
    drop(target);
    let _ = std::fs::remove_file(path);
}
