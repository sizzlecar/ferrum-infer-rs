//! Actual private call -> FIFO -> separate live phases -> real product import.
//! CPU lifecycle evidence; the selected device command is a contract fixture.
use super::*;
use ferrum_types::{SloCostObservationConfig, SloCostProfileExportConfig};
use std::{fs, path::PathBuf};

pub(in crate::continuous_engine::inner) struct Fixture {
    directory: PathBuf,
    pub(in crate::continuous_engine::inner) config: SloCostObservationConfig,
    options: SloCostProfileExportConfig,
    pub(in crate::continuous_engine::inner) clock: Arc<VirtualClock>,
    ids: EngineCostIds,
    queue: Arc<BoundedCostSampleSink>,
}
impl Fixture {
    pub(in crate::continuous_engine::inner) fn new() -> Self {
        let directory =
            std::env::temp_dir().join(format!("ferrum-selected-capture-{}", uuid::Uuid::new_v4()));
        fs::create_dir(&directory).unwrap();
        let mut config = SloCostObservationConfig::selected_whole_wave_v1();
        config.profile_import.declared_local_clock_max_error_ns = Some(1_000_000_000);
        let options = SloCostProfileExportConfig {
            path: directory.join("profile.json"),
            observations_path: directory.join("source.jsonl"),
            max_samples: NonZeroUsize::new(32).unwrap(),
            max_total_shape_rows: NonZeroUsize::new(32).unwrap(),
            declared_clock_max_error_ns: Some(1_000_000_000),
            ..Default::default()
        };
        Self {
            directory,
            config,
            options,
            clock: Arc::new(VirtualClock(AtomicU64::new(21))),
            ids: EngineCostIds::default(),
            queue: sink(4, 32),
        }
    }
    pub(in crate::continuous_engine::inner) fn fingerprint(&self) -> model::ExecutionFingerprint {
        let ExecutorCostIdentityAvailability::Known(identity) = identity() else {
            panic!()
        };
        model::ExecutionFingerprint {
            model_weights: identity.model_weights,
            numerical_policy: identity.numerical_policy,
            device_runtime: identity.device_runtime,
            execution_config: identity.execution_config,
        }
    }
    pub(in crate::continuous_engine::inner) fn begin(&self) -> SelectedCalibrationCapture {
        SelectedCalibrationCapture::new(
            self.options.clone(),
            &self.config,
            self.fingerprint(),
            [81; 32],
            self.clock.clone(),
        )
        .unwrap()
    }
    pub(in crate::continuous_engine::inner) fn runtime(&self) -> EngineCostRuntime {
        EngineCostRuntime::build_with_profile(
            identity(),
            self.clock.clone(),
            &self.config,
            false,
            None,
            None,
        )
        .unwrap()
    }
    pub(in crate::continuous_engine::inner) fn completed(
        &self,
    ) -> (u64, CostEvidenceEntry, Arc<CostCalibrationCapture>) {
        recorded_with_capture(&self.ids, &self.queue)
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.directory);
    }
}

#[test]
fn selected_live_phases_freeze_then_reload_the_real_completed_cut() {
    let fixture = Fixture::new();
    let mut capture = fixture.begin();
    assert!(fixture.options.observations_path.exists());
    assert!(!fixture.options.path.exists());
    let mut fit_cut = 0;
    for _ in 0..8 {
        let (ordinal, _, receipt) = fixture.completed();
        fit_cut = ordinal;
        capture.record(&receipt, true).unwrap();
    }
    let frozen = capture.freeze_fit(fit_cut).unwrap();
    assert_eq!(frozen.retained_fit_samples, 8);
    assert!(
        !fixture.options.path.exists(),
        "fit alone has no residual-qualified profile"
    );
    let mut residual_cut = 0;
    for _ in 0..8 {
        let (ordinal, _, receipt) = fixture.completed();
        residual_cut = ordinal;
        capture.record(&receipt, true).unwrap();
    }
    let (cut, support) = capture.finish(residual_cut).unwrap();
    assert_eq!(support["family_support"][0]["fit_samples"], 8);
    assert_eq!(support["family_support"][0]["residual_samples"], 8);
    let bytes = fs::read(&cut.source).unwrap();
    use sha2::{Digest, Sha256};
    assert_eq!(<[u8; 32]>::from(Sha256::digest(&bytes)), cut.source_digest);
    let records: Vec<serde_json::Value> = bytes
        .split(|b| *b == b'\n')
        .filter(|s| !s.is_empty())
        .map(|line| serde_json::from_slice(line).unwrap())
        .collect();
    let freeze_at = records
        .iter()
        .position(|r| r["kind"] == "fit_frozen")
        .unwrap();
    assert!(records[1..freeze_at].iter().all(|r| r["phase"] == "fit"));
    assert!(records[freeze_at + 1..records.len() - 1]
        .iter()
        .all(|r| r["phase"] == "residual"));
    assert_eq!(records.last().unwrap()["kind"], "completed_residual_cut");
    let runtime = EngineCostRuntime::build_with_profile(
        identity(),
        fixture.clock.clone(),
        &fixture.config,
        false,
        None,
        None,
    )
    .unwrap();
    let loaded = runtime
        .load_calibration_profile(&fixture.config, &cut)
        .unwrap();
    let receipt = loaded.receipt.selected_whole_wave.as_ref().unwrap();
    assert_eq!(receipt.fit_parameters_sha256, frozen.fit_parameters_sha256);
    assert_eq!(
        receipt.capture_identity_sha256,
        frozen.capture_identity_sha256
    );
    assert_ne!(receipt.capture_identity_sha256, cut.source_digest);
    assert_eq!((receipt.fit_records, receipt.residual_records), (8, 8));
    assert_eq!(
        loaded.receipt.source_observation_artifact_sha256,
        cut.source_digest
    );
    let (ordinal, entry, _) = fixture.completed();
    assert!(ordinal > cut.accepted_ordinal);
    let heldout = super::super::super::trainer::whole_wave_observation(
        &entry,
        ordinal,
        frozen.capture_identity_sha256,
    )
    .unwrap();
    assert!(loaded
        .snapshot
        .predict_selected_wave(&heldout.exact, &heldout.selected, 21)
        .is_ok());
    assert_eq!(
        fs::read(&cut.source).unwrap(),
        bytes,
        "heldout cannot enter the closed source"
    );
}

#[test]
fn unavailable_or_unreconciled_calls_do_not_become_fit_samples() {
    let fixture = Fixture::new();
    let mut capture = fixture.begin();
    capture
        .record(&CostCalibrationCapture::default(), true)
        .unwrap();
    let (ordinal, _, receipt) = fixture.completed();
    capture.record(&receipt, false).unwrap();
    assert!(capture.freeze_fit(ordinal).is_err());
    assert!(!fixture.options.path.exists());
    let raw = fs::read_to_string(&fixture.options.observations_path).unwrap();
    assert!(raw.contains("WrongSource"));
    assert!(raw.contains("InvalidSample"));
    assert!(!raw.contains("\"kind\":\"observation\""));
}

#[test]
fn capture_limit_or_unfrozen_residual_cannot_publish_a_profile() {
    let mut fixture = Fixture::new();
    fixture.options.max_samples = NonZeroUsize::new(1).unwrap();
    let mut capture = fixture.begin();
    let (_, _, first) = fixture.completed();
    capture.record(&first, true).unwrap();
    let (ordinal, _, second) = fixture.completed();
    assert!(capture.record(&second, true).is_err());
    assert!(
        capture.freeze_fit(ordinal).is_err(),
        "retention failure is sticky"
    );
    assert!(capture.finish(ordinal).is_err());
    assert!(!fixture.options.path.exists());
    // Both real observations remain as raw diagnostics. They are not a
    // completed source and cannot be mistaken for retained model support.
    let raw = fs::read_to_string(&fixture.options.observations_path).unwrap();
    assert!(!raw.contains("completed_residual_cut"));
    let other = Fixture::new();
    assert!(other.begin().finish(1).is_err());
    assert!(!other.options.path.exists());
}
