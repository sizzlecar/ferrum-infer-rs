//! Exercise collector isolation through the real manual driver and output flow.
use super::*;
mod multi;
mod successful;
use crate::continuous_engine::inner::cost_observation::{
    CostCalibrationCapture, StructuredCalibrationCollector, StructuredCaptureSessionBinding,
};
use ferrum_scheduler::implementations::continuous::cost_model::{
    structured::StructuredSettingsV1, ExecutionFingerprint,
};
struct SourcePath(std::path::PathBuf);
impl SourcePath {
    fn new() -> Self {
        Self(std::env::temp_dir().join(format!(
            "ferrum-manual-structured-{}.jsonl",
            uuid::Uuid::new_v4()
        )))
    }
}
impl Drop for SourcePath {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}
fn options(path: &std::path::Path) -> StructuredCalibrationOptions {
    StructuredCalibrationOptions {
        observations_path: path.into(),
        protocol_sha256: [1; 32],
        scope: StructuredCalibrationScopeV1 {
            rows: NonZeroUsize::MIN,
            domain_signature: [2; 32],
        },
        settings: StructuredSettingsV1::default(),
        fit_members: NonZeroUsize::new(16).unwrap(),
        residual_members: NonZeroUsize::new(16).unwrap(),
        qualification_members: NonZeroUsize::new(8).unwrap(),
        maximum_offered_waves: NonZeroUsize::new(128).unwrap(),
        maximum_file_bytes: NonZeroU64::new(1 << 20).unwrap(),
    }
}
fn fingerprint() -> ExecutionFingerprint {
    ExecutionFingerprint {
        model_weights: [1; 32],
        numerical_policy: [2; 32],
        device_runtime: [3; 32],
        execution_config: [4; 32],
    }
}
async fn collector_fixture() -> (CalibrationSession, Arc<ControlledExecutor>) {
    let (mut session, executor) = super::fixture(1).await;
    let inner = Arc::get_mut(&mut session.engine.inner).unwrap();
    let original = inner.cost_runtime.as_ref().unwrap();
    // The shared fixture intentionally has no training worker. These tests use
    // real asynchronous FIFO barriers, so supply their real consumer without
    // changing production behavior or the shared fixture's deterministic mode.
    inner.cost_runtime = Some(Arc::new(
        crate::continuous_engine::inner::cost_observation::EngineCostRuntime::build(
            original.identity.clone(),
            Arc::clone(&original.clock),
            &inner.config.scheduler.slo.cost_observation,
            true,
        )
        .unwrap(),
    ));
    (session, executor)
}
async fn install_test_collector(session: &mut CalibrationSession, path: &std::path::Path) {
    let cut = session
        .freeze_cost_model()
        .await
        .unwrap()
        .accepted_ordinal();
    // The controlled executor deliberately lacks qualified structured evidence.
    // Direct construction tests driver isolation without mislabeling the mock
    // as a supported live producer; the public begin gate is tested separately.
    let clock = Arc::clone(&session.engine.inner.cost_runtime.as_ref().unwrap().clock);
    session.structured_capture = Some(
        StructuredCalibrationCollector::new(options(path), fingerprint(), clock, cut).unwrap(),
    );
}
#[tokio::test]
async fn structured_collector_failure_does_not_reject_real_request_execution() {
    let (mut session, executor) = collector_fixture().await;
    let path = SourcePath::new();
    install_test_collector(&mut session, &path.0).await;
    assert!(bounded(session.freeze_structured_cost_phase())
        .await
        .is_err());
    let (id, mut output) = add(&mut session, 4).await;
    admit(&mut session).await;
    let work = frontier(&session, &id)
        .prefill_work(NonZeroU32::new(4).unwrap())
        .unwrap();
    let report = wave(&mut session, &executor, vec![work]).await;
    assert!(report.error.is_none());
    assert_eq!(
        report.submission,
        CalibrationSubmissionState::HostReconciled
    );
    drop(bounded(output.frames.next()).await.unwrap());
    let artifact = bounded(session.finish_structured_cost_calibration())
        .await
        .unwrap();
    assert_eq!(artifact.phase, StructuredCapturePhase::Failed);
    assert!(artifact.model.is_none());
    drop(output);
    session.shutdown().await.unwrap();
}
#[tokio::test]
async fn structured_collector_reserves_decode_members_before_failed_evidence_is_known() {
    let (mut session, executor) = collector_fixture().await;
    let path = SourcePath::new();
    let (id, mut output) = add(&mut session, 4).await;
    admit(&mut session).await;
    install_test_collector(&mut session, &path.0).await;
    let work = frontier(&session, &id)
        .prefill_work(NonZeroU32::new(4).unwrap())
        .unwrap();
    let report = wave(&mut session, &executor, vec![work]).await;
    assert!(report.error.is_none());
    drop(bounded(output.frames.next()).await.unwrap());
    ready(&session, &id, false).await;
    let work = frontier(&session, &id).decode_work().unwrap();
    let report = wave(&mut session, &executor, vec![work]).await;
    assert!(report.error.is_none());
    drop(bounded(output.frames.next()).await.unwrap());
    let artifact = bounded(session.finish_structured_cost_calibration())
        .await
        .unwrap();
    assert!(artifact.scope_members >= 1);
    assert!(artifact.scope_failures >= 1);
    assert!(artifact.model.is_none());
    let text = std::fs::read_to_string(&path.0).unwrap();
    let records: Vec<serde_json::Value> = text
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    let offered: Vec<_> = records
        .iter()
        .map(|r| &r["record"])
        .filter(|r| r["kind"] == "offered")
        .collect();
    assert_eq!(offered[0]["member_candidate"], false);
    assert!(offered
        .iter()
        .any(|record| record["member_candidate"] == true
            && record["rows"][0]["generated"] == 1
            && record["rows"][0]["decode"] == true));
    assert!(records
        .iter()
        .any(|record| record["record"]["kind"] == "reserved" && record["record"]["member"] == 1));
    drop(output);
    session.shutdown().await.unwrap();
}
#[tokio::test]
async fn structured_collector_disabled_capture_gate_and_immutable_receipt_binding() {
    let (mut session, executor) = collector_fixture().await;
    let path = SourcePath::new();
    assert!(session
        .begin_structured_cost_calibration(options(&path.0))
        .await
        .is_err());
    assert!(session.structured_capture.is_none());
    let (id, output) = add(&mut session, 4).await;
    admit(&mut session).await;
    let work = frontier(&session, &id)
        .prefill_work(NonZeroU32::new(4).unwrap())
        .unwrap();
    let report = wave(&mut session, &executor, vec![work]).await;
    assert!(report.error.is_none());
    let runtime = session.engine.inner.cost_runtime.as_ref().unwrap();
    let binding = Arc::new(
        StructuredCaptureSessionBinding::new([1; 32], fingerprint(), runtime.clock.as_ref())
            .unwrap(),
    );
    let capture = Arc::new(CostCalibrationCapture::for_structured_session(binding));
    let receipt = CalibrationWaveReceipt::new(report.ordered_work.clone());
    receipt
        .bind_structured_capture(Arc::clone(&capture))
        .unwrap();
    assert!(Arc::ptr_eq(receipt.capture(), &capture));
    assert!(receipt
        .bind_structured_capture(Arc::clone(&capture))
        .is_err());
    let default_receipt = CalibrationWaveReceipt::new(report.ordered_work.clone());
    let original = Arc::clone(default_receipt.capture());
    assert!(default_receipt
        .bind_structured_capture(Arc::clone(&capture))
        .is_err());
    assert!(Arc::ptr_eq(default_receipt.capture(), &original));
    let submitted = CalibrationWaveReceipt::new(report.ordered_work);
    submitted.record(CalibrationSubmissionState::Submitted);
    assert!(submitted.bind_structured_capture(capture).is_err());
    drop(output);
    session.shutdown().await.unwrap();
}
