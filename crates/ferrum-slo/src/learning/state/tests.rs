use super::*;
use crate::{
    learning::{LearningEstimator, LearningEstimatorOwner},
    model::{TrainingSample, WorkFeatures},
};
use ferrum_execution::{
    execution_attempt::*,
    token_execution::{ExecutedTokenWork, ExecutionRange},
};
use ferrum_scheduler::cost_prediction::*;
use ferrum_types::RequestId;

mod saved_model_planning;

fn config() -> LearningConfig {
    LearningConfig {
        training_samples: 4,
        validation_samples: 2,
        calibration_samples: 4,
        min_calibration_samples: 2,
        retrain_every: 8,
        window_samples: 16,
        calibration_ttl_ms: 10_000,
        model_ttl_ms: 100_000,
        training: crate::model::TrainingBudget {
            max_trees: 4,
            ..Default::default()
        },
        ..Default::default()
    }
}

fn identity() -> ExecutionIdentity {
    ExecutionIdentity::new("persistence/immutable-source/cpu/f32".into()).unwrap()
}

#[test]
fn width_history_state_requires_exact_policy_and_legacy_files_mean_recent() {
    let (mut config, snapshot, clock) = fixture(ModelFamily::Physical);
    let (bytes, _) = encode(&snapshot, &config, ModelFamily::Physical, clock).unwrap();
    let mut legacy: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    legacy["semantics"]
        .as_object_mut()
        .unwrap()
        .remove("training_history");
    let legacy = serde_json::to_vec(&legacy).unwrap();
    assert!(decode(&legacy, &config, &identity(), ModelFamily::Physical, clock).is_ok());
    config.training_history = super::super::TrainingHistoryPolicy::WidthStratified {
        samples_per_width: 2,
    };
    assert!(decode(&legacy, &config, &identity(), ModelFamily::Physical, clock).is_err());
    assert!(decode(&bytes, &config, &identity(), ModelFamily::Physical, clock).is_err());
    let (stratified, _) = encode(&snapshot, &config, ModelFamily::Physical, clock).unwrap();
    assert!(decode(
        &stratified,
        &config,
        &identity(),
        ModelFamily::Physical,
        clock
    )
    .is_ok());
    config.training_history = super::super::TrainingHistoryPolicy::WidthStratified {
        samples_per_width: 1,
    };
    assert!(decode(
        &stratified,
        &config,
        &identity(),
        ModelFamily::Physical,
        clock
    )
    .is_err());
    config.training_history = super::super::TrainingHistoryPolicy::Recent;
    assert!(decode(
        &stratified,
        &config,
        &identity(),
        ModelFamily::Physical,
        clock
    )
    .is_err());
}

#[test]
fn extrema_history_state_requires_its_explicit_policy_and_width_count() {
    let (mut config, snapshot, clock) = fixture(ModelFamily::Physical);
    config.training_history = super::super::TrainingHistoryPolicy::WidthAndExtrema {
        samples_per_width: 2,
    };
    let (bytes, _) = encode(&snapshot, &config, ModelFamily::Physical, clock).unwrap();
    assert!(decode(&bytes, &config, &identity(), ModelFamily::Physical, clock).is_ok());
    for policy in [
        super::super::TrainingHistoryPolicy::Recent,
        super::super::TrainingHistoryPolicy::WidthStratified {
            samples_per_width: 2,
        },
        super::super::TrainingHistoryPolicy::WidthAndExtrema {
            samples_per_width: 1,
        },
    ] {
        let mut mismatched = config.clone();
        mismatched.training_history = policy;
        assert!(decode(
            &bytes,
            &mismatched,
            &identity(),
            ModelFamily::Physical,
            clock
        )
        .is_err());
        let (other, _) = encode(&snapshot, &mismatched, ModelFamily::Physical, clock).unwrap();
        assert!(decode(&other, &config, &identity(), ModelFamily::Physical, clock).is_err());
    }
}
fn work() -> CostWork {
    CostWork {
        kind: AttemptWorkKind::Decode,
        requested: ExecutionRange::new(3, 1).unwrap(),
        context_tokens: 4,
        prefix_restore: AttemptPrefixRestore::NoneObserved,
        internally_reused_tokens: None,
    }
}

fn fixture(family: ModelFamily) -> (LearningConfig, snapshot::Snapshot, Clock) {
    let config = config();
    let clock = Clock::now();
    let features = WorkFeatures::extract(&[work()], 1).unwrap();
    let samples = [TrainingSample {
        features,
        service_seconds: 0.01,
    }; 4];
    let model = FittedModel::train(
        family,
        identity(),
        &samples[..2],
        &samples[2..],
        &config.training,
    )
    .unwrap();
    let base = model.predict(&identity(), &features).unwrap();
    let mut calibration = OnlineCalibrator::new(config.calibration_config(), 1).unwrap();
    calibration.observe(
        1,
        features.domain(),
        0,
        1.0,
        base.risk_base_seconds,
        clock.monotonic - Duration::from_secs(9),
    );
    calibration.observe(
        1,
        features.domain(),
        0,
        0.005,
        base.risk_base_seconds,
        clock.monotonic - Duration::from_secs(1),
    );
    calibration.observe(
        1,
        features.domain(),
        0,
        0.007,
        base.risk_base_seconds,
        clock.monotonic - Duration::from_secs(1),
    );
    let snapshot = snapshot::Snapshot {
        revision: 1,
        query: config.query,
        identity: Arc::new(identity()),
        boundaries: config.work_bucket_boundaries.clone().into(),
        quantile: config.training.risk_quantile,
        rejections: None,
        domains: vec![snapshot::Domain {
            model: Arc::new(model),
            model_revision: 1,
            model_expires_at: clock.monotonic + Duration::from_millis(config.model_ttl_ms),
            calibration: calibration.snapshot(clock.monotonic),
        }],
    };
    (config, snapshot, clock)
}

#[test]
fn roundtrip_restores_raw_residuals_with_offline_age_without_renewing_ttl() {
    for family in [ModelFamily::Physical, ModelFamily::Gbdt] {
        let (config, snapshot, saved_clock) = fixture(family);
        let (bytes, count) = encode(&snapshot, &config, family, saved_clock).unwrap();
        assert_eq!(count, 1);
        // A different monotonic origin is valid; only elapsed wall time crosses processes.
        let loaded_clock = Clock {
            monotonic: saved_clock.monotonic + Duration::from_secs(300),
            wall: saved_clock.wall + Duration::from_secs(3),
        };
        let restored = decode(&bytes, &config, &identity(), family, loaded_clock).unwrap();
        assert_eq!(restored.report.status, StateLoadStatus::Loaded);
        assert_eq!(restored.report.expired_residual_entries, 2);
        assert_eq!(
            restored.domains[0].model_expires_at,
            loaded_clock.monotonic + Duration::from_secs(97)
        );
        let calibration = restored.domains[0]
            .calibration
            .snapshot(loaded_clock.monotonic);
        let correction = calibration
            .correction(
                1,
                &restored.domains[0].model.domain(),
                0,
                loaded_clock.monotonic,
            )
            .unwrap();
        assert_eq!(correction.samples, 2);
        assert_eq!(
            correction.expires_at,
            loaded_clock.monotonic + Duration::from_secs(6)
        );
        assert!((0.01 + correction.signed_seconds - 0.007).abs() < 1e-12);
    }
}

#[test]
fn stale_state_is_explicit_cold_status_and_clock_rollback_is_rejected() {
    let (config, snapshot, clock) = fixture(ModelFamily::Physical);
    let (bytes, _) = encode(&snapshot, &config, ModelFamily::Physical, clock).unwrap();
    let later = Clock {
        monotonic: clock.monotonic + Duration::from_secs(20),
        wall: clock.wall + Duration::from_secs(20),
    };
    let state = decode(&bytes, &config, &identity(), ModelFamily::Physical, later).unwrap();
    assert_eq!(state.report.status, StateLoadStatus::Stale);
    assert_eq!(state.report.expired_domains, 1);
    assert!(state.domains.is_empty());
    assert!(matches!(
        decode(
            &bytes,
            &config,
            &identity(),
            ModelFamily::Physical,
            Clock {
                wall: clock.wall - Duration::from_secs(1),
                ..clock
            }
        ),
        Err(LearningStateError::ClockRollback)
    ));
}

#[test]
fn loader_checks_full_identity_family_calibration_semantics_and_revision() {
    let (config, snapshot, clock) = fixture(ModelFamily::Physical);
    let (bytes, _) = encode(&snapshot, &config, ModelFamily::Physical, clock).unwrap();
    assert!(decode(
        &bytes,
        &config,
        &ExecutionIdentity::new("other-executor".into()).unwrap(),
        ModelFamily::Physical,
        clock
    )
    .is_err());
    assert!(decode(&bytes, &config, &identity(), ModelFamily::Gbdt, clock).is_err());
    assert!(decode(
        &bytes,
        &LearningConfig {
            min_calibration_samples: 1,
            ..config.clone()
        },
        &identity(),
        ModelFamily::Physical,
        clock
    )
    .is_err());
    for field in ["schema", "feature_schema", "revision"] {
        let mut invalid: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        invalid[field] = 0.into();
        assert!(decode(
            &serde_json::to_vec(&invalid).unwrap(),
            &config,
            &identity(),
            ModelFamily::Physical,
            clock
        )
        .is_err());
    }
    let mut invalid: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    invalid["domains"][0]["calibration"]["model_revision"] = 9.into();
    assert!(decode(
        &serde_json::to_vec(&invalid).unwrap(),
        &config,
        &identity(),
        ModelFamily::Physical,
        clock
    )
    .is_err());
    invalid = serde_json::from_slice(&bytes).unwrap();
    let ordinal = invalid["domains"][0]["calibration"]["global"]["samples"][0]["ordinal"].clone();
    invalid["domains"][0]["calibration"]["global"]["samples"][1]["ordinal"] = ordinal;
    assert!(decode(
        &serde_json::to_vec(&invalid).unwrap(),
        &config,
        &identity(),
        ModelFamily::Physical,
        clock
    )
    .is_err());
    assert!(decode(
        &vec![b' '; byte_limit(&config).unwrap() + 1],
        &config,
        &identity(),
        ModelFamily::Physical,
        clock
    )
    .is_err());
}

struct TempState(std::path::PathBuf);
impl TempState {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        Self(std::env::temp_dir().join(format!(
            "ferrum-learning-state-{}-{}.json",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        )))
    }
}
impl Drop for TempState {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.0);
    }
}

fn observe(owner: &LearningEstimatorOwner, seconds: f64) {
    let completed_at = Instant::now();
    let summary = ExecutionAttemptSummary {
        started_at: completed_at - Duration::from_secs_f64(seconds),
        completed_at,
        eligibility: ExecutionAttemptEligibility::Completed,
        participants: vec![ExecutionAttemptParticipant {
            request_id: RequestId::new(),
            kind: work().kind,
            requested: work().requested,
            context_tokens: 4,
            prefix_restore: AttemptPrefixRestore::NoneObserved,
            internally_reused_tokens: None,
            outcome: AttemptParticipantOutcome::Completed {
                work: ExecutedTokenWork::Decode { position: 3 },
                frontier_committed: true,
            },
            product_failed: false,
        }],
    };
    assert_eq!(
        owner.observer().try_record(&summary),
        ExecutionAttemptDelivery::Recorded
    );
}

#[tokio::test]
async fn restored_owner_publishes_before_observation_then_continues_same_base_fifo_updates() {
    for family in [ModelFamily::Physical, ModelFamily::Gbdt] {
        let path = TempState::new();
        let (config, snapshot, clock) = fixture(family);
        let (bytes, _) = encode(&snapshot, &config, family, clock).unwrap();
        atomic_write(&path.0, &bytes).unwrap();
        let owner =
            LearningEstimator::spawn_with_state(config.clone(), identity(), family, Some(&path.0))
                .unwrap();
        assert_eq!(owner.load_report().unwrap().status, StateLoadStatus::Loaded);
        let source = owner.source();
        let query_work = [work()];
        let query = CostQuery {
            work: &query_work,
            work_is_exact: true,
            execution: CostExecution::SingleInvocation,
        };
        assert!(matches!(
            source.load().predict(&query, Instant::now()),
            CostPrediction::Known { .. }
        ));
        for _ in 0..4 {
            observe(&owner, 0.001);
        }
        let (exit, saved) = owner.shutdown_and_save(&path.0).await.unwrap();
        assert_eq!(exit.stats.model_fits, 0);
        assert_eq!(exit.stats.worker_base_prediction_known, 4);
        assert_eq!(saved, StateSaveReport::Saved { domains: 1 });
        let CostPrediction::Known {
            central_service_time,
            upper_service_time,
            ..
        } = source.load().predict(&query, Instant::now())
        else {
            panic!("lost loaded coverage");
        };
        assert_eq!(central_service_time, Duration::from_millis(10));
        assert_eq!(upper_service_time, Duration::from_millis(1));
        let reloaded =
            LearningEstimator::spawn_with_state(config, identity(), family, Some(&path.0)).unwrap();
        let CostPrediction::Known {
            upper_service_time, ..
        } = reloaded.source().load().predict(&query, Instant::now())
        else {
            panic!("saved update cannot reload");
        };
        assert_eq!(upper_service_time, Duration::from_millis(1));
        let _ = reloaded.shutdown().await.unwrap();
    }
}

#[tokio::test]
async fn missing_or_stale_state_serves_cold_and_unqualified_save_preserves_existing_file() {
    let path = TempState::new();
    let owner = LearningEstimator::spawn_with_state(
        config(),
        identity(),
        ModelFamily::Physical,
        Some(&path.0),
    )
    .unwrap();
    assert_eq!(
        owner.load_report().unwrap().status,
        StateLoadStatus::Missing
    );
    let (_, report) = owner.shutdown_and_save(&path.0).await.unwrap();
    assert_eq!(report, StateSaveReport::SkippedNoQualifiedState);
    assert!(!path.0.exists());
    let (config, snapshot, clock) = fixture(ModelFamily::Physical);
    let (bytes, _) = encode(&snapshot, &config, ModelFamily::Physical, clock).unwrap();
    let mut saved: SavedState = serde_json::from_slice(&bytes).unwrap();
    saved.saved_unix_ns -= Duration::from_secs(20).as_nanos() as u64;
    let bytes = serde_json::to_vec(&saved).unwrap();
    atomic_write(&path.0, &bytes).unwrap();
    let owner = LearningEstimator::spawn_with_state(
        config,
        identity(),
        ModelFamily::Physical,
        Some(&path.0),
    )
    .unwrap();
    assert_eq!(owner.load_report().unwrap().status, StateLoadStatus::Stale);
    let (_, report) = owner.shutdown_and_save(&path.0).await.unwrap();
    assert_eq!(report, StateSaveReport::SkippedNoQualifiedState);
    assert_eq!(fs::read(&path.0).unwrap(), bytes);
}

#[tokio::test]
async fn auto_restores_only_a_fully_validated_fresh_bound_family() {
    use crate::learning::FamilyResolution;
    for family in [ModelFamily::Physical, ModelFamily::Gbdt] {
        let path = TempState::new();
        let (config, snapshot, clock) = fixture(family);
        let (bytes, _) = encode(&snapshot, &config, family, clock).unwrap();
        atomic_write(&path.0, &bytes).unwrap();
        let owner =
            LearningEstimator::spawn_auto_with_state(config.clone(), identity(), Some(&path.0))
                .unwrap();
        assert_eq!(owner.family(), family);
        assert_eq!(
            owner.family_resolution(),
            FamilyResolution::AutoRestoredBoundState { family }
        );
        assert_eq!(owner.load_report().unwrap().status, StateLoadStatus::Loaded);
        assert_eq!(owner.source.load_typed().domains[0].model.family(), family);
        let _ = owner.shutdown().await.unwrap();
        assert!(LearningEstimator::spawn_auto_with_state(
            config.clone(),
            ExecutionIdentity::new("other actual execution".into()).unwrap(),
            Some(&path.0)
        )
        .is_err());
        let other = if family == ModelFamily::Gbdt {
            ModelFamily::Physical
        } else {
            ModelFamily::Gbdt
        };
        assert!(
            LearningEstimator::spawn_with_state(config, identity(), other, Some(&path.0)).is_err()
        );
    }
}

#[tokio::test]
async fn auto_cold_fallback_reasons_are_visible_and_stale_gbdt_cannot_select_the_family() {
    use crate::learning::{AutoFallbackReason, FamilyResolution};
    let path = TempState::new();
    for (state_path, reason) in [
        (None, AutoFallbackReason::NoBoundStateProvided),
        (Some(path.0.as_path()), AutoFallbackReason::MissingState),
    ] {
        let owner =
            LearningEstimator::spawn_auto_with_state(config(), identity(), state_path).unwrap();
        assert_eq!(owner.family(), ModelFamily::Physical);
        assert_eq!(
            owner.family_resolution(),
            FamilyResolution::AutoPhysicalFallback { reason }
        );
        let _ = owner.shutdown().await.unwrap();
    }
    let (config, snapshot, clock) = fixture(ModelFamily::Gbdt);
    let (bytes, _) = encode(&snapshot, &config, ModelFamily::Gbdt, clock).unwrap();
    let mut saved: SavedState = serde_json::from_slice(&bytes).unwrap();
    saved.saved_unix_ns -= Duration::from_secs(20).as_nanos() as u64;
    atomic_write(&path.0, &serde_json::to_vec(&saved).unwrap()).unwrap();
    let owner = LearningEstimator::spawn_auto_with_state(config.clone(), identity(), Some(&path.0))
        .unwrap();
    assert_eq!(owner.family(), ModelFamily::Physical);
    assert_eq!(
        owner.family_resolution(),
        FamilyResolution::AutoPhysicalFallback {
            reason: AutoFallbackReason::StaleState
        }
    );
    assert!(owner.source.load_typed().domains.is_empty());
    let _ = owner.shutdown().await.unwrap();
    fs::write(&path.0, b"corrupt state").unwrap();
    assert!(LearningEstimator::spawn_auto_with_state(config, identity(), Some(&path.0)).is_err());
}
