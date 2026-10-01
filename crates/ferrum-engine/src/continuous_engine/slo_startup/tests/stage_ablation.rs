use super::*;
use ferrum_types::SloExperimentStageV1 as Stage;

#[tokio::test]
async fn stage_ablation_shared_startup_preserves_cost_and_execution_checks() {
    let fixture = Fixture::new().await;
    for stage in [Stage::CostCandidates, Stage::Complete] {
        let mut config = fixture.config.clone();
        config.scheduler.slo.experiment_stage = Some(stage);
        let engine = fixture.build(config.clone()).unwrap();
        assert!(engine
            .inner
            .cost_runtime
            .as_ref()
            .unwrap()
            .profile_receipt()
            .is_some());
        assert!(engine.inner.prefill_reference_runtime.is_some());
        engine.shutdown().await.unwrap();
        config.scheduler.slo.cost_profile = None;
        assert!(
            fixture.build(config).is_err(),
            "missing explicit cost remains an error"
        );
    }
    for stage in [
        Stage::ControlledAdaptiveBaseline,
        Stage::SingleWave,
        Stage::OutputIsolation,
        Stage::DeadlineOnly,
    ] {
        let mut config = fixture.config.clone();
        config.scheduler.slo.experiment_stage = Some(stage);
        config.scheduler.slo.output.transport = stage.output_transport();
        config.scheduler.slo.cost_profile = None;
        config.scheduler.slo.prefill_reference = None;
        let engine = fixture.build(config.clone()).unwrap();
        assert!(engine.inner.prefill_reference_runtime.is_none());
        let cost = engine.inner.cost_runtime.as_ref().unwrap();
        assert_eq!(cost.test_activity(), (0, 0));
        assert!(cost.profile_receipt().is_none());
        engine.shutdown().await.unwrap();
        config.scheduler.prefix_rendezvous_max_wait_ms = NonZeroU64::new(1);
        assert!(
            fixture.build(config.clone()).is_err(),
            "actual resolved prefix wait conflicts"
        );
        config.scheduler.prefix_rendezvous_max_wait_ms = None;
        assert!(validate_execution(&config, fixture.executor.as_ref(), true).is_err());
        *fixture.executor.startup_capability_override.lock() =
            Some(ExecutorSloCapability::Unavailable);
        assert!(fixture.build(config).is_err());
        *fixture.executor.startup_capability_override.lock() = None;
    }
}
