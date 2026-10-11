use super::*;
use ferrum_interfaces::{DecodeCycleOutcome, DecodeCycleStage};

fn cycle_config(entrypoint: ProfileEntrypoint, path: PathBuf) -> EngineConfig {
    let mut config = EngineConfig::default();
    config.kv_cache.max_blocks = 128;
    config.runtime.profile_detail = ObservabilityProfileDetail::Cycle;
    config.runtime.profile_entrypoint = Some(entrypoint);
    config.runtime.scheduler_trace_jsonl = Some(path);
    config
}

fn cycle_engine(
    config: EngineConfig,
) -> Result<(ContinuousBatchEngine, Arc<PlanRuntimeAdmissionTestExecutor>)> {
    let scheduler = Arc::new(ContinuousBatchScheduler::new(config.scheduler.clone()));
    let executor = Arc::new(PlanRuntimeAdmissionTestExecutor::new(128));
    let engine = ContinuousBatchEngine::new_plan_runtime(
        config,
        scheduler,
        Arc::new(ferrum_testkit::MockTokenizer::new(128)),
        Arc::new(ferrum_testkit::MockSampler),
        executor.clone(),
        Arc::new(MockTensorFactory),
    )?;
    Ok((engine, executor))
}

fn decode_input() -> PlanRuntimeDecodeInput {
    let request = RequestId::new();
    let cache = Arc::new(ferrum_testkit::MockKvCacheHandle::new(
        request.clone(),
        1,
        1,
    ));
    PlanRuntimeDecodeInput::new(request, TokenId::new(5), cache)
}

#[tokio::test]
async fn cycle_run_and_serve_attach_only_coarse_sink_and_preserve_decode_ordinals() {
    for entrypoint in [ProfileEntrypoint::Run, ProfileEntrypoint::Serve] {
        let path = resource_trace_temp_path("cycle-call-context");
        let mut config = cycle_config(entrypoint, path.clone());
        config.runtime.profile_jsonl = Some(path.clone());
        config.runtime.legacy_scheduler_trace_jsonl = Some(path.clone());
        let (engine, executor) = cycle_engine(config).unwrap();
        assert!(engine.inner.cycle_diagnostic.is_some());
        assert!(engine.inner.profile_trace_jsonl.is_none());
        assert!(engine.inner.scheduler_trace_jsonl.is_none());
        assert!(engine.inner.legacy_scheduler_trace_jsonl.is_none());
        let trace = {
            let attached = executor.event_sink.lock().unwrap();
            let sink = attached.as_ref().expect("Cycle diagnostic sink attached");
            assert_eq!(
                sink.enablement(),
                ferrum_interfaces::vnext::ExecutionEventSinkEnablement::None
            );
            assert_eq!(
                sink.device_timing_mode(),
                ferrum_interfaces::vnext::DeviceTimingMode::Off
            );
            sink.decode_cycle_trace().expect("shared Cycle clock")
        };
        let mut inputs = vec![decode_input(), decode_input()];
        assert!(inputs
            .iter()
            .all(|input| input.decode_cycle_context.is_none()));
        let call = engine
            .inner
            .decode_cycle_call_contexts(&mut inputs)
            .unwrap();
        for (ordinal, input) in inputs.iter().enumerate() {
            let context = input.decode_cycle_context.unwrap();
            assert_eq!(context.clock_id(), trace.clock().id);
            assert_eq!(context.call, call.call);
            assert_eq!(context.iteration, call.iteration);
            assert_eq!(context.caller_ordinal as usize, ordinal);
        }
        engine
            .inner
            .decode_cycle_call_mark(Some(call), DecodeCycleStage::CallStarted, None);
        engine.inner.decode_cycle_call_mark(
            Some(call),
            DecodeCycleStage::CallReturned,
            Some(DecodeCycleOutcome::Failed),
        );
        let next = engine
            .inner
            .decode_cycle_call_contexts(&mut inputs)
            .unwrap();
        assert_ne!(next.call, call.call);
        assert_eq!(next.caller_ordinal, 0);
        engine.shutdown().await.unwrap();
        let events = read_engine_profile_events(&path);
        let capture = &events[0].attributes["capture"];
        let returned = capture["events"]
            .as_array()
            .unwrap()
            .iter()
            .find(|event| event["stage"] == "call_returned")
            .expect("returned call boundary survives export");
        assert_eq!(returned["outcome"], "failed");
        std::fs::remove_file(path).unwrap();
    }
}

#[tokio::test]
async fn cycle_disabled_has_no_recorder_or_call_contexts() {
    let (engine, executor) = cycle_engine(EngineConfig::default()).unwrap();
    assert!(engine.inner.cycle_diagnostic.is_none());
    assert!(executor.event_sink.lock().unwrap().is_none());
    let mut inputs = vec![decode_input()];
    assert!(engine
        .inner
        .decode_cycle_call_contexts(&mut inputs)
        .is_none());
    assert!(inputs[0].decode_cycle_context.is_none());
    engine.shutdown().await.unwrap();
}

#[test]
fn cycle_rejects_missing_path_and_unwritable_output() {
    let mut pathless = EngineConfig::default();
    pathless.runtime.profile_detail = ObservabilityProfileDetail::Cycle;
    let error = cycle_engine(pathless)
        .err()
        .expect("Cycle requires its output path");
    assert!(error.to_string().contains("requires scheduler_trace_jsonl"));

    let file_parent = resource_trace_temp_path("cycle-file-parent");
    std::fs::write(&file_parent, b"not a directory").unwrap();
    let path = file_parent.join("capture.jsonl");
    let error = cycle_engine(cycle_config(ProfileEntrypoint::Run, path))
        .err()
        .expect("a file cannot serve as the capture output directory");
    assert!(error.to_string().contains("open decode cycle JSONL"));
    assert_eq!(std::fs::read(&file_parent).unwrap(), b"not a directory");
    std::fs::remove_file(file_parent).unwrap();
}

#[test]
fn cycle_rejects_legacy_authority_and_speculative_plan_runtime() {
    let path = resource_trace_temp_path("cycle-unsupported");
    let config = cycle_config(ProfileEntrypoint::Run, path.clone());
    let make_engine = |composition, executor: Arc<dyn ModelExecutor + Send + Sync>, draft, spec| {
        ContinuousBatchEngine::new_with_resource_composition(
            config.clone(),
            Arc::new(ContinuousBatchScheduler::new(config.scheduler.clone())),
            Arc::new(ferrum_testkit::MockTokenizer::new(128)),
            Arc::new(ferrum_testkit::MockSampler),
            composition,
            executor,
            Arc::new(MockTensorFactory),
            draft,
            spec,
        )
    };
    let legacy_error = make_engine(
        EngineResourceComposition::LegacyEngine {
            kv_cache: Arc::new(MockKvCacheManager::new(128)),
            recurrent_state_manager: None,
        },
        Arc::new(MockModelExecutor::instant(128)),
        None,
        None,
    )
    .err()
    .expect("Cycle must reject unsupported legacy execution");
    assert!(legacy_error
        .to_string()
        .contains("requires the PlanRuntime batch executor"));
    let speculative_error = make_engine(
        EngineResourceComposition::PlanRuntime,
        Arc::new(PlanRuntimeAdmissionTestExecutor::new(128)),
        Some(Arc::new(PlanRuntimeAdmissionTestExecutor::new(128))),
        Some(crate::speculative::SpeculativeDecodingConfig::default()),
    )
    .err()
    .expect("Cycle must reject speculative execution");
    assert!(speculative_error
        .to_string()
        .contains("without speculative decoding"));
    assert!(!path.exists());
}

#[tokio::test]
async fn cycle_natural_shutdown_exports_valid_loop_capture_exactly_once() {
    for entrypoint in [ProfileEntrypoint::Run, ProfileEntrypoint::Serve] {
        let path = resource_trace_temp_path("cycle-natural-shutdown");
        let (engine, _) = cycle_engine(cycle_config(entrypoint, path.clone())).unwrap();
        let mut request = policy_request();
        request.sampling_params.max_tokens = 1;
        let response = engine.infer(request).await.unwrap();
        assert_eq!(response.finish_reason, FinishReason::Length);
        engine.shutdown().await.unwrap();
        let first_bytes = std::fs::read(&path).unwrap();
        engine.shutdown().await.unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), first_bytes);
        let events = read_engine_profile_events(&path);
        assert_eq!(events.len(), 1, "the drain owns exactly one capture event");
        let event = &events[0];
        assert_eq!(event.phase, "engine.decode_cycle_capture");
        assert_eq!(event.entrypoint, entrypoint);
        assert_eq!(event.status, ProfileStatus::DiagnosticOnly);
        assert_eq!(event.attributes["clock_source"], "rust_std_instant");
        assert_eq!(event.attributes["collector_complete"], true);
        assert_eq!(
            event.attributes["boundary_coverage_requires_validation"],
            true
        );
        let capture = &event.attributes["capture"];
        assert_eq!(capture["active_producers_at_close"], 0);
        let marks = capture["events"].as_array().unwrap();
        for stage in ["loop_resumed", "iteration_end"] {
            assert!(marks.iter().any(|mark| mark["stage"] == stage));
        }
        let clock = &capture["clock"]["id"];
        assert!(marks.iter().all(|mark| &mark["at"]["clock_id"] == clock));
        // The mock executor supplies no backend submission witness; a complete
        // collector is not a claim of complete physical-cycle boundary coverage.
        assert!(marks
            .iter()
            .all(|mark| mark["stage"] != "compute_graph_launch_start"));
        std::fs::remove_file(path).unwrap();
    }
}
