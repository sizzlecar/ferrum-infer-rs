//! Product loader receives schema 6 made from the real numbered FIFO/terminal
//! fixture. This is an engine contract test, not a hardware timing experiment.
use super::*;
use ferrum_scheduler::implementations::continuous::{
    cost_model::statistical::model::{CalibrationPartitionV1, WholeWaveSettingsV1},
    cost_profile::{self as file, statistical_v6::CostProfileFileV6},
    slo_planner::PlanningCostModel,
};
use ferrum_types::{SloCostObservationConfig, SloCostPredictor};
use std::{fs, io::Write, path::PathBuf};

struct Fixture {
    path: PathBuf,
    config: SloCostObservationConfig,
    samples: Vec<model::statistical::model::WholeWaveObservationV1>,
}
impl Fixture {
    fn new() -> Self {
        Self::with_terminal(true)
    }
    fn with_terminal(terminal: bool) -> Self {
        let config = SloCostObservationConfig {
            profile_import: ferrum_types::SloCostProfileImportConfig {
                declared_local_clock_max_error_ns: Some(0),
                ..Default::default()
            },
            ..SloCostObservationConfig::selected_whole_wave_v1()
        };
        let ids = EngineCostIds::default();
        let queue = sink(4, 32);
        let samples: Vec<_> = (0..16)
            .map(|_| {
                let (ordinal, entry, _) = recorded_case(&ids, &queue, terminal, 0, 0);
                super::super::super::trainer::whole_wave_observation(&entry, ordinal, [7; 32])
                    .unwrap()
            })
            .collect();
        let settings = WholeWaveSettingsV1::from_policy_limits(
            &super::super::super::profile::model_settings(&config.model),
        );
        let partition = CalibrationPartitionV1 {
            source_sha256: [7; 32],
            protocol_sha256: [8; 32],
            fit_through_ordinal: 8,
            residual_through_ordinal: 16,
        };
        let file = CostProfileFileV6::from_observations(
            &samples[0].fingerprint,
            &settings,
            partition,
            file::ProfileSource {
                generator: "real-engine-fifo-fixture".into(),
                generator_revision: "v1".into(),
                measurement_protocol: "independent-fit-residual-terminal-fixture".into(),
                observation_artifact_sha256: [7; 32],
            },
            21,
            1_000_000,
            0,
            &samples[..8],
            &samples[8..],
        )
        .unwrap();
        let path = std::env::temp_dir().join(format!(
            "ferrum-selected-runtime-{}.json",
            uuid::Uuid::new_v4()
        ));
        fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
            .unwrap()
            .write_all(
                &file
                    .to_bounded_bytes(config.profile_import.max_file_bytes.get())
                    .unwrap(),
            )
            .unwrap();
        Self {
            path,
            config,
            samples,
        }
    }
    fn load_clock() -> file::ProfileLoadClock {
        file::ProfileLoadClock {
            monotonic_now_ns: 100,
            wall_unix_ns: Some(1_000_010),
            wall_max_error_ns: Some(0),
        }
    }
    fn build(
        &self,
        clock: Arc<VirtualClock>,
    ) -> Result<EngineCostRuntime, ferrum_types::FerrumError> {
        EngineCostRuntime::build_with_profile(
            identity(),
            clock,
            &self.config,
            false,
            Some(&self.path),
            Some(Self::load_clock()),
        )
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
    }
}

#[test]
fn selected_runtime_import_is_immutable_and_ttl_keeps_aging() {
    let fixture = Fixture::new();
    let clock = Arc::new(VirtualClock(AtomicU64::new(100)));
    let runtime = fixture.build(clock.clone()).unwrap();
    let before = runtime.snapshot().unwrap();
    assert!(before.requires_statistical_evidence());
    let sample = &fixture.samples[0];
    let prediction = before
        .predict_selected_wave(&sample.exact, &sample.selected, 100)
        .unwrap();
    assert!(prediction.planning_ns >= 19 + fixture.config.model.drift_margin_ns);
    let receipt = runtime.profile_receipt().unwrap();
    assert_eq!(receipt.schema_version, 6);
    assert_eq!(receipt.path, fixture.path);
    assert_eq!(receipt.recorded_samples, 16);
    assert_eq!(receipt.oldest_imported_age_ns, Some(10));
    let phases = receipt.selected_whole_wave.as_ref().unwrap();
    assert_eq!((phases.fit_records, phases.residual_records), (8, 8));
    assert_eq!(
        (phases.fit_through_ordinal, phases.residual_through_ordinal),
        (8, 16)
    );
    assert_eq!(phases.protocol_sha256, [8; 32]);
    let shape = ferrum_scheduler::implementations::continuous::slo_planner::canonical_cost_shape(
        &sample.exact,
    )
    .unwrap();
    assert!(before
        .predict_with_evidence(&sample.fingerprint, &shape, None, 100)
        .is_none());
    // A genuine later terminal observation enters the runtime FIFO, but there
    // is no online trainer capable of republishing this frozen fit/residual cut.
    let (_, entry) = recorded();
    runtime.sink.offer_evidence_numbered(entry).unwrap();
    runtime.consume_samples();
    let after = runtime.snapshot().unwrap();
    assert!(Arc::ptr_eq(&before, &after));
    assert_eq!(runtime.audit_snapshot().training.publish_attempts, 0);
    assert_eq!(before.model_version(), after.model_version());
    clock.set(100 + fixture.config.model.max_sample_age_ns.get());
    assert_eq!(
        after.predict_selected_wave(&sample.exact, &sample.selected, clock.now_ns().unwrap()),
        Err(ModelUnknown::Stale)
    );
    assert!(
        Arc::ptr_eq(&before, &runtime.snapshot().unwrap()),
        "expiration changes availability, not the immutable artifact generation"
    );
}

#[test]
fn selected_runtime_requires_explicit_protocol_identity_and_clock() {
    let mut fixture = Fixture::new();
    let clock = Arc::new(VirtualClock(AtomicU64::new(100)));
    fixture.config.predictor = SloCostPredictor::LegacyFeatureModel;
    assert!(
        fixture.build(clock.clone()).is_err(),
        "legacy loader must not reinterpret schema 6"
    );
    fixture.config.predictor = SloCostPredictor::SelectedWholeWaveV1;
    let mut wrong = fixture.samples[0].fingerprint.clone();
    wrong.device_runtime = [99; 32];
    let foreign = ExecutorCostIdentityAvailability::Known(Arc::new(ExecutorCostIdentity {
        schema_version: EXECUTOR_COST_IDENTITY_SCHEMA,
        model_weights: wrong.model_weights,
        numerical_policy: wrong.numerical_policy,
        device_runtime: wrong.device_runtime,
        execution_config: wrong.execution_config,
    }));
    assert!(EngineCostRuntime::build_with_profile(
        foreign,
        clock.clone(),
        &fixture.config,
        false,
        Some(&fixture.path),
        Some(Fixture::load_clock())
    )
    .is_err());
    let mut late = Fixture::load_clock();
    late.wall_unix_ns = Some(1_000_010 + fixture.config.model.max_sample_age_ns.get());
    assert!(EngineCostRuntime::build_with_profile(
        identity(),
        clock.clone(),
        &fixture.config,
        false,
        Some(&fixture.path),
        Some(late)
    )
    .is_err());
    fixture
        .config
        .profile_import
        .declared_local_clock_max_error_ns = None;
    assert!(fixture.build(clock).is_err());
}

#[path = "runtime/serving_audit.rs"]
mod serving_audit;
