//! Product construction checks actual checkpoint capability after CLI resolution.
use super::*;
use crate::continuous_engine::inner::slo_controller::tests::fixture::startup_checkpoint_components;

fn automatic_config(mode: SloMode, prefix: bool) -> EngineConfig {
    let mut config = EngineConfig::default();
    config.scheduler.slo =
        serde_json::from_value(serde_json::json!({ "mode": "enforce" })).unwrap();
    config.scheduler.slo.mode = mode;
    config.scheduler.slo.default_service_class = Some("checkpoint-startup".into());
    config.scheduler.slo.services.push(ServiceSloConfig {
        id: "checkpoint-startup".into(),
        server_token_commit: ferrum_types::SloLatencyBudgets {
            ttft_ms: NonZeroU64::new(10_000).unwrap(),
            tpot_ms: NonZeroU64::new(10_000).unwrap(),
            itl_ms: NonZeroU64::new(10_000).unwrap(),
        },
        client_visible: None,
        attainment: Default::default(),
    });
    let ferrum_types::SloLiveStructuredCalibration::AutomaticV1 { settings } = &mut config
        .scheduler
        .slo
        .cost_observation
        .live_structured_calibration
    else {
        panic!("minimal Enforce must resolve automatic calibration");
    };
    settings.reuse = ferrum_types::SloAutomaticCalibrationReuseV1::Disabled {};
    config.runtime.prefix_state_cache_enabled = prefix;
    config.scheduler.max_running_requests = 2;
    config.batching.max_batch_size = 2;
    config.scheduler.slo.validate().unwrap();
    config
}

#[tokio::test]
async fn automatic_slo_prefix_capability_accepts_actual_native_constructor() {
    for mode in [SloMode::Observe, SloMode::Enforce] {
        let (tokenizer, executor) = startup_checkpoint_components(2).await;
        assert!(executor.supports_guarded_prefix_maintenance());
        assert!(matches!(
            executor.slo_execution_capability(),
            ExecutorSloCapability::GuardedEagerWaves | ExecutorSloCapability::GuardedOnDemandWaves
        ));
        let config = automatic_config(mode, true);
        let engine = ContinuousBatchEngine::new_plan_runtime(
            config.clone(),
            Arc::new(ContinuousBatchScheduler::new(config.scheduler.clone())),
            tokenizer,
            Arc::new(crate::registry::GreedySampler),
            executor.clone(),
            Arc::new(MockTensorFactory),
        )
        .unwrap();
        assert!(engine.inner.config.runtime.prefix_state_cache_enabled);
        assert_eq!(executor.entries.load(Ordering::Acquire), 0);
        assert_eq!(executor.physical.load(Ordering::Acquire), 0);
        bounded(engine.shutdown()).await.unwrap();
        executor.abort_resource_sessions();
    }
}

#[tokio::test]
async fn automatic_slo_prefix_capability_rejects_missing_transfer_without_disabling_it() {
    let f = Fixture::new().await;
    assert!(!f.executor.supports_guarded_prefix_maintenance());
    let mut imported_enforce = f.config.clone();
    imported_enforce.runtime.prefix_state_cache_enabled = true;
    assert!(validate_execution(&imported_enforce, f.executor.as_ref(), false).is_err());
    for mode in [SloMode::Observe, SloMode::Enforce] {
        let config = automatic_config(mode, true);
        let error = match f.build(config.clone()) {
            Ok(_) => panic!("a wave-only executor must not claim guarded checkpoint support"),
            Err(error) => error,
        };
        assert!(matches!(error, FerrumError::Unsupported { .. }), "{error}");
        assert!(config.runtime.prefix_state_cache_enabled);
        assert_eq!(f.executor.entries.load(Ordering::Acquire), 0);
        assert_eq!(f.executor.physical.load(Ordering::Acquire), 0);

        let without_cache = automatic_config(mode, false);
        let engine = f.build(without_cache).unwrap();
        assert!(!engine.inner.config.runtime.prefix_state_cache_enabled);
        bounded(engine.shutdown()).await.unwrap();
    }
    let mut manual_observe = f.config.clone();
    manual_observe.scheduler.slo.mode = SloMode::Observe;
    manual_observe.runtime.prefix_state_cache_enabled = true;
    validate_execution(&manual_observe, f.executor.as_ref(), false).unwrap();
}

#[tokio::test]
async fn automatic_slo_prefix_capability_keeps_wave_and_speculation_requirements() {
    let (_, executor) = startup_checkpoint_components(2).await;
    assert!(executor.supports_guarded_prefix_maintenance());
    for mode in [SloMode::Observe, SloMode::Enforce] {
        let config = automatic_config(mode, true);
        assert!(validate_execution(&config, executor.as_ref(), true).is_err());
        *executor.startup_capability_override.lock() = Some(ExecutorSloCapability::Unavailable);
        assert!(validate_execution(&config, executor.as_ref(), false).is_err());
        *executor.startup_capability_override.lock() = None;
    }
    assert_eq!(executor.entries.load(Ordering::Acquire), 0);
    assert_eq!(executor.physical.load(Ordering::Acquire), 0);
    executor.abort_resource_sessions();
}
