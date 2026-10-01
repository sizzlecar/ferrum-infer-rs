use super::*;
use sha2::Digest;

async fn independent_fixture() -> (CalibrationSession, Arc<ControlledExecutor>) {
    let (mut session, executor) = fixture(4).await;
    let inner = Arc::get_mut(&mut session.engine.inner).unwrap();
    let identity = inner.cost_runtime.as_ref().unwrap().identity.clone();
    let config = ferrum_types::SloCostObservationConfig::structured_whole_wave_v2();
    assert!(config.profile_export.is_none());
    assert!(config
        .profile_import
        .declared_local_clock_max_error_ns
        .is_none());
    inner.cost_runtime = Some(Arc::new(
        EngineCostRuntime::build(
            identity,
            Arc::new(EngineCostClock::default()),
            &config,
            false,
        )
        .unwrap(),
    ));
    inner.config.scheduler.slo.cost_observation = config;
    executor
        .emit_cost_observations
        .store(true, Ordering::Release);
    (session, executor)
}

async fn complete_collector(
    session: &mut CalibrationSession,
    executor: &ControlledExecutor,
) -> CalibrationReferenceCollector {
    let (plan, observations, _) = discovery(session, executor).await;
    let mut collector = freeze(session, plan, observations).await.unwrap();
    trial(
        session,
        executor,
        &mut collector,
        CalibrationReferenceTrial::Prefill {
            curve: 0,
            repetition: 0,
        },
    )
    .await;
    trial(
        session,
        executor,
        &mut collector,
        CalibrationReferenceTrial::Decode { repetition: 0 },
    )
    .await;
    collector
}

async fn capture(
    session: &mut CalibrationSession,
    collector: &CalibrationReferenceCollector,
    destination: &std::path::Path,
) -> Result<crate::continuous_engine::inner::calibration::reference::CalibrationReferenceSourceV1> {
    let runtime = session.engine.inner.cost_runtime.as_ref().unwrap().clone();
    drive_training(
        runtime,
        session.capture_reference_source_v1(collector, destination),
    )
    .await
}

#[tokio::test]
async fn independent_source_uses_actual_checkpoint_without_cost_profile_or_clock_declaration() {
    let directory = Directory::new();
    let (mut session, executor) = independent_fixture().await;
    let collector = complete_collector(&mut session, &executor).await;
    let source = capture(
        &mut session,
        &collector,
        &directory.0.join("reference-source.jsonl"),
    )
    .await
    .unwrap();
    assert!(source.accepted_ordinal() > collector.frozen_accepted_ordinal());
    assert!(source.bytes() > 0);
    assert_eq!(
        source.sha256(),
        <[u8; 32]>::from(sha2::Sha256::digest(std::fs::read(source.path()).unwrap()))
    );
    let runtime = session.engine.inner.cost_runtime.as_ref().unwrap();
    assert!(
        runtime.snapshot().is_none(),
        "reference must not require a fitted cost model"
    );
    assert!(runtime.profile_receipt().is_none());
    let artifact = collector
        .finish_from_source_v1(&source, &directory.0.join("reference.json"))
        .unwrap();
    let fingerprint = match &runtime.identity {
        ferrum_interfaces::execution_cost::ExecutorCostIdentityAvailability::Known(identity) => {
            ferrum_scheduler::implementations::continuous::cost_model::ExecutionFingerprint {
                model_weights: identity.model_weights,
                numerical_policy: identity.numerical_policy,
                device_runtime: identity.device_runtime,
                execution_config: identity.execution_config,
            }
        }
        _ => unreachable!(),
    };
    let loaded = load_prefill_reference(
        &artifact.path,
        &fingerprint,
        artifact.protocol_sha256,
        &Default::default(),
    )
    .unwrap();
    assert!(loaded.tau_ref_ns().get() > 0);
    assert_eq!(artifact.source_sha256, source.sha256());
    assert_eq!(
        artifact.training_accepted_ordinal,
        source.accepted_ordinal()
    );
    let first: serde_json::Value = serde_json::from_str(
        std::fs::read_to_string(source.path())
            .unwrap()
            .lines()
            .next()
            .unwrap(),
    )
    .unwrap();
    assert_eq!(first["header"]["artifact_type"], "ferrum.reference-source");
    assert!(first["header"].get("declared_clock_max_error_ns").is_none());
    session.shutdown().await.unwrap();
}

#[tokio::test]
async fn independent_source_rejects_changed_original_observation_or_commit() {
    for field in ["observed_at_monotonic_ns", "work_generation"] {
        let directory = Directory::new();
        let (mut session, executor) = independent_fixture().await;
        let collector = complete_collector(&mut session, &executor).await;
        let source = capture(&mut session, &collector, &directory.0.join("source.jsonl"))
            .await
            .unwrap();
        let mut records: Vec<serde_json::Value> = std::fs::read_to_string(source.path())
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        let observation = &mut records[1]["observation"];
        let target = if field == "work_generation" {
            &mut observation["commit"][field]
        } else {
            &mut observation[field]
        };
        *target = serde_json::json!(target.as_u64().unwrap() + 1);
        let changed = records
            .iter()
            .map(|record| serde_json::to_string(record).unwrap())
            .collect::<Vec<_>>()
            .join("\n")
            + "\n";
        std::fs::write(source.path(), changed).unwrap();
        let destination = directory.0.join("reference.json");
        assert!(collector
            .finish_from_source_v1(&source, &destination)
            .is_err());
        assert!(!destination.exists());
        session.shutdown().await.unwrap();
    }
}

#[tokio::test]
async fn independent_source_does_not_seal_incomplete_trials_or_another_session() {
    let directory = Directory::new();
    let (mut session, executor) = independent_fixture().await;
    let (plan, observations, _) = discovery(&mut session, &executor).await;
    let collector = freeze(&mut session, plan, observations).await.unwrap();
    let destination = directory.0.join("incomplete.jsonl");
    assert!(capture(&mut session, &collector, &destination)
        .await
        .is_err());
    assert!(!destination.exists());
    let (mut other, _) = independent_fixture().await;
    let destination = directory.0.join("other.jsonl");
    assert!(capture(&mut other, &collector, &destination).await.is_err());
    assert!(!destination.exists());
    other.shutdown().await.unwrap();
    session.shutdown().await.unwrap();
}
