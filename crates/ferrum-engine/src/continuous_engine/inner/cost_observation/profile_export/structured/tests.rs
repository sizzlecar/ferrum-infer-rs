use super::*;
struct FixedClock;
impl CostObservationClock for FixedClock {
    fn now_ns(&self) -> Option<u64> {
        Some(100)
    }
}
struct SourcePath(PathBuf);
impl SourcePath {
    fn new() -> Self {
        Self(std::env::temp_dir().join(format!("ferrum-structured-{}.jsonl", uuid::Uuid::new_v4())))
    }
}
impl Drop for SourcePath {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}
fn options(path: &Path) -> StructuredCalibrationOptions {
    StructuredCalibrationOptions {
        observations_path: path.into(),
        protocol_sha256: [1; 32],
        scope: StructuredCalibrationScopeV1 {
            rows: NonZeroUsize::new(8).unwrap(),
            domain_signature: [2; 32],
        },
        settings: StructuredSettingsV1::default(),
        fit_members: NonZeroUsize::new(16).unwrap(),
        residual_members: NonZeroUsize::new(16).unwrap(),
        qualification_members: NonZeroUsize::new(9).unwrap(),
        maximum_offered_waves: NonZeroUsize::new(128).unwrap(),
        maximum_file_bytes: NonZeroU64::new(1 << 20).unwrap(),
    }
}
fn fingerprint() -> model::ExecutionFingerprint {
    model::ExecutionFingerprint {
        model_weights: [1; 32],
        numerical_policy: [2; 32],
        device_runtime: [3; 32],
        execution_config: [4; 32],
    }
}
#[test]
fn structured_collector_incomplete_finish_has_explicit_failed_footer_and_no_model() {
    let path = SourcePath::new();
    let collector = StructuredCalibrationCollector::new(
        options(&path.0),
        fingerprint(),
        Arc::new(FixedClock),
        10,
    )
    .unwrap();
    let result = collector.finish(10).unwrap();
    assert_eq!(result.phase, StructuredCapturePhase::Failed);
    assert!(result.model.is_none());
    assert!(result.failure.is_some());
    let bytes = std::fs::read(&path.0).unwrap();
    assert_eq!(result.source_bytes, bytes.len() as u64);
    assert_eq!(
        result.source_sha256,
        <[u8; 32]>::from(Sha256::digest(&bytes))
    );
    let records = std::str::from_utf8(&bytes)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str::<serde_json::Value>(line).unwrap())
        .collect::<Vec<_>>();
    assert_eq!(records[0]["source_record_ordinal"], 1);
    assert_eq!(records[1]["source_record_ordinal"], 2);
    assert_eq!(records[1]["record"]["kind"], "footer");
    assert_eq!(records[1]["record"]["phase"], "failed");
}
#[test]
fn structured_collector_missing_global_fifo_invalidates_artifact_instead_of_relabeling_cut() {
    let path = SourcePath::new();
    let collector = StructuredCalibrationCollector::new(
        options(&path.0),
        fingerprint(),
        Arc::new(FixedClock),
        10,
    )
    .unwrap();
    let result = collector.finish(11).unwrap();
    assert!(result.model.is_none());
    let text = std::fs::read_to_string(&path.0).unwrap();
    let footer: serde_json::Value = serde_json::from_str(text.lines().last().unwrap()).unwrap();
    assert_eq!(footer["record"]["accepted_fifo_cutoff"], 11);
    assert_eq!(footer["record"]["last_captured_fifo"], 10);
    assert_eq!(footer["record"]["fifo_audit_complete"], false);
}
#[test]
fn structured_collector_partial_raw_write_cannot_produce_completed_source_receipt() {
    let path = SourcePath::new();
    let mut source = StructuredSource::create(&path.0, 32).unwrap();
    assert!(source
        .record(&serde_json::json!({"payload":"x".repeat(128)}))
        .is_err());
    assert!(source
        .record(&serde_json::json!({"kind":"footer"}))
        .is_err());
    assert!(matches!(
        source.finish(),
        Err(ExportError::SourceOnly { .. })
    ));
    assert!(std::fs::metadata(&path.0).unwrap().len() <= 32);
}
#[test]
fn structured_collector_scope_protocol_and_numeric_capacity_are_frozen_at_opening() {
    let path = SourcePath::new();
    let original = options(&path.0);
    original.validate().unwrap();
    let mut changed = original.clone();
    changed.scope.domain_signature = [3; 32];
    assert_ne!(original.protocol_signature(), changed.protocol_signature());
    changed = original.clone();
    changed.settings.static_margin_ns += 1;
    assert_ne!(original.protocol_signature(), changed.protocol_signature());
    changed = original.clone();
    changed.qualification_members = NonZeroUsize::new(8).unwrap();
    assert!(changed.validate().is_err());
    changed = original;
    changed.settings.max_axes = 4096;
    changed.settings.max_phase_samples = 4096;
    assert!(changed.validate().is_err());
}
#[test]
fn structured_collector_failed_freeze_is_sticky_and_still_finishes_unknown() {
    let path = SourcePath::new();
    let mut collector = StructuredCalibrationCollector::new(
        options(&path.0),
        fingerprint(),
        Arc::new(FixedClock),
        10,
    )
    .unwrap();
    assert!(collector.freeze(10).is_err());
    assert!(!collector.collecting());
    let result = collector.finish(10).unwrap();
    assert_eq!(result.phase, StructuredCapturePhase::Failed);
    assert!(result.model.is_none());
}
