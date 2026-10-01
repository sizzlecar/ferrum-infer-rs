//! Failed and partial service windows keep original evidence without qualifying
//! a model. Uses the original recorder, private tickets and bounded writer.
use super::*;
use sha2::{Digest, Sha256};

struct FailureFixture {
    directory: PathBuf,
    clock: Arc<VirtualClock>,
    config: SloCostObservationConfig,
}
impl FailureFixture {
    fn new(generations: usize, source_bytes: u64) -> Self {
        #[derive(serde::Serialize)]
        struct Declaration {
            schema_version: u32,
            phase_offered_waves: [usize; 3],
            maximum_window_ns: u64,
            settings: StructuredSettingsV2,
            scopes: Vec<StructuredScopeV2>,
        }
        let directory =
            std::env::temp_dir().join(format!("ferrum-live-failure-{}", uuid::Uuid::new_v4()));
        fs::create_dir(&directory).unwrap();
        let w = wave("fixture.feedback.a");
        let input = StructuredInputV2::from_actual(
            &w.prepared.exact,
            &w.prepared.selected,
            &w.prepared.recipe,
        )
        .unwrap();
        let declaration = Declaration {
            schema_version: 1,
            phase_offered_waves: [8; 3],
            maximum_window_ns: 10_000_000_000,
            settings: StructuredSettingsV2 {
                static_margin_ns: 1,
                ..Default::default()
            },
            scopes: vec![StructuredScopeV2 {
                numerical_family: None,
                owner: input.owner().clone(),
                coverage: StructuredCoverageV2 {
                    pending_eligible_positions: vec![],
                    authorized_pending_constraints: vec![HostPendingConstraintV2::AnySubset],
                    pending_counts: vec![0],
                    length_counts: vec![1],
                    pending_positions: vec![],
                    length_positions: vec![0],
                    joint_counts: vec![(0, 1)],
                },
            }],
        };
        let path = directory.join("declaration.json");
        fs::write(&path, serde_json::to_vec(&declaration).unwrap()).unwrap();
        let mut config = SloCostObservationConfig::structured_whole_wave_v2();
        config.live_structured_calibration =
            ferrum_types::SloLiveStructuredCalibration::ServiceWindowsV1 {
                declaration: path,
                evidence_directory: directory.join("evidence"),
                maximum_generations: NonZeroUsize::new(generations).unwrap(),
                maximum_source_bytes: NonZeroU64::new(source_bytes).unwrap(),
                maximum_retained_numeric_bytes: NonZeroUsize::new(16 * 1024 * 1024).unwrap(),
            };
        Self {
            directory,
            config,
            clock: Arc::new(VirtualClock(AtomicU64::new(1))),
        }
    }
    fn build(&self) -> EngineCostRuntime {
        EngineCostRuntime::build(identity(), self.clock.clone(), &self.config, false).unwrap()
    }
    fn offer(&self, runtime: &EngineCostRuntime, incarnation: u64) {
        let mut w = wave("fixture.feedback.a");
        w.actual.rows[0].owner_incarnation = incarnation;
        record_with_hooks(
            &runtime.ids,
            &runtime.sink,
            &self.clock,
            w.actual,
            w.host,
            None,
            0,
            None,
            |at| Some(runtime.reserve_live_ticket(Some(at)).unwrap()),
            |_| {},
        );
        runtime.consume_samples();
    }
    fn failed_source(&self, runtime: &EngineCostRuntime) -> (PathBuf, Vec<serde_json::Value>) {
        let receipt = runtime
            .training
            .live
            .as_ref()
            .unwrap()
            .audit()
            .last_failed_source
            .unwrap();
        assert!(receipt.footer_complete);
        assert!(!receipt.retained_unpublished);
        let path = receipt.path.unwrap();
        let bytes = fs::read(&path).unwrap();
        assert_eq!(receipt.bytes, bytes.len() as u64);
        assert_eq!(
            receipt.sha256.unwrap(),
            format!("{:x}", Sha256::digest(&bytes))
        );
        let records = bytes
            .split(|byte| *byte == b'\n')
            .filter(|line| !line.is_empty())
            .map(|line| serde_json::from_slice(line).unwrap())
            .collect();
        (path, records)
    }
}
impl Drop for FailureFixture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.directory);
    }
}

#[tokio::test]
async fn failed_window_waits_for_retirement_and_persists_unqueued_original_tickets() {
    let f = FailureFixture::new(1, 8 * 1024 * 1024);
    let runtime = f.build();
    runtime.consume_samples();
    f.offer(&runtime, 1);
    let at = f.clock.now_ns().unwrap();
    let mut held = runtime.reserve_live_ticket(Some(at)).unwrap();
    held.bind_call(900);
    let mut abandoned = runtime.reserve_live_ticket(Some(at)).unwrap();
    abandoned.bind_call(901);
    drop(abandoned);
    runtime.consume_samples();
    let audit = runtime.training.live.as_ref().unwrap().audit();
    assert_eq!((audit.population.issued, audit.population.retired), (3, 2));
    assert!(audit.last_failed_source.is_none());
    assert!(runtime.reserve_live_ticket(Some(at)).is_none());
    assert!(runtime.snapshot().is_none());
    drop(held);
    runtime.consume_samples();
    let (path, rows) = f.failed_source(&runtime);
    assert_eq!(rows.iter().filter(|r| r["kind"] == "completed").count(), 1);
    let failed: Vec<_> = rows
        .iter()
        .filter(|r| r["kind"] == "ticket_failed")
        .collect();
    assert_eq!(failed.len(), 2);
    assert_eq!(
        (failed[0]["ticket"].as_u64(), failed[0]["call_id"].as_u64()),
        (Some(2), Some(900))
    );
    assert_eq!(
        (failed[1]["ticket"].as_u64(), failed[1]["call_id"].as_u64()),
        (Some(3), Some(901))
    );
    assert!(failed.iter().all(|r| r["issued_at_ns"] == at));
    let footer = rows.last().unwrap();
    assert_eq!(footer["kind"], "footer");
    assert_eq!(footer["offered"], 3);
    assert!(footer["failure"].is_string());
    let bytes = fs::read(&path).unwrap();
    assert!(file::export_structured_profile_v13(
        &path,
        Sha256::digest(&bytes).into(),
        &f.directory.join("must-not-qualify.json"),
        0,
        &Default::default()
    )
    .is_err());
    assert!(runtime.snapshot().is_none());
    assert_eq!(
        runtime
            .training
            .live
            .as_ref()
            .unwrap()
            .audit()
            .failed_generations,
        1
    );
    runtime.shutdown().await.unwrap();
    assert_eq!(fs::read(path).unwrap(), bytes);
}

#[tokio::test]
async fn shutdown_persists_partial_successful_population_without_a_cost_publication() {
    let f = FailureFixture::new(2, 8 * 1024 * 1024);
    let runtime = f.build();
    f.offer(&runtime, 1);
    f.offer(&runtime, 2);
    runtime.shutdown().await.unwrap();
    let (_, rows) = f.failed_source(&runtime);
    assert_eq!(rows.iter().filter(|r| r["kind"] == "completed").count(), 2);
    assert!(rows.iter().all(|r| r["kind"] != "ticket_failed"));
    assert_eq!(rows.last().unwrap()["offered"], 2);
    assert!(rows.last().unwrap()["failure"]
        .as_str()
        .unwrap()
        .contains("shutdown"));
    let audit = runtime.training.live.as_ref().unwrap().audit();
    assert_eq!(audit.qualified_publications, 0);
    assert_eq!(audit.failed_generations, 1);
    assert_eq!(audit.population.generation, 1);
    assert!(runtime.snapshot().is_none());
}

#[tokio::test]
async fn bounded_source_failure_retains_written_bytes_after_runtime_drop() {
    let f = FailureFixture::new(1, 1);
    let runtime = f.build();
    runtime.consume_samples();
    let audit = runtime.training.live.as_ref().unwrap().audit();
    let receipt = audit.last_failed_source.unwrap();
    assert!(audit.publication_error.is_some());
    assert_eq!(audit.failed_generations, 1);
    assert!(receipt.retained_unpublished);
    assert!(!receipt.footer_complete);
    let path = receipt.path.unwrap();
    let bytes = fs::read(&path).unwrap();
    assert!(!bytes.is_empty() && bytes.len() <= 1);
    assert_eq!(receipt.bytes, bytes.len() as u64);
    assert_eq!(
        receipt.sha256.unwrap(),
        format!("{:x}", Sha256::digest(&bytes))
    );
    runtime.shutdown().await.unwrap();
    drop(runtime);
    assert_eq!(fs::read(&path).unwrap(), bytes);
    assert_eq!(fs::read_dir(path.parent().unwrap()).unwrap().count(), 1);
}

#[tokio::test]
async fn later_qualified_generation_clears_current_error_and_keeps_failed_source_count() {
    let f = FailureFixture::new(2, 8 * 1024 * 1024);
    let runtime = f.build();
    drop(runtime.reserve_live_ticket(f.clock.now_ns()).unwrap());
    runtime.consume_samples();
    let (failed_path, _) = f.failed_source(&runtime);
    assert!(runtime
        .training
        .live
        .as_ref()
        .unwrap()
        .audit()
        .publication_error
        .is_some());
    runtime.consume_samples();
    for incarnation in 1..=24 {
        f.offer(&runtime, incarnation);
    }
    let audit = runtime.training.live.as_ref().unwrap().audit();
    assert!(audit.publication_error.is_none(), "{audit:?}");
    assert_eq!(audit.failed_generations, 1);
    assert_eq!(audit.qualified_publications, 1);
    assert!(failed_path.exists());
    assert!(runtime.snapshot().is_some());
    runtime.shutdown().await.unwrap();
    let closed = runtime.training.live.as_ref().unwrap().audit();
    assert!(!closed.population.failed);
    assert_eq!(closed.failed_generations, 1);
    assert_eq!(closed.qualified_publications, 1);
}

#[test]
fn live_source_retention_is_opt_in_and_keeps_a_failed_publish_within_its_original_bound() {
    use crate::continuous_engine::inner::cost_observation::profile_export::StagedFile;
    use std::io::Write;
    let f = FailureFixture::new(1, 1024);
    let destination = f.directory.join("source.jsonl");
    let mut ordinary = StagedFile::create(&destination, 4).unwrap();
    ordinary.write_all(b"test").unwrap();
    let removed = ordinary.unpublished_receipt().path;
    drop(ordinary);
    assert!(!removed.exists());
    let mut live = StagedFile::create(&destination, 4).unwrap();
    live.preserve_unpublished();
    live.write_all(b"test").unwrap();
    assert!(live.write_all(b"over bound").is_err());
    let retained = live.unpublished_receipt();
    fs::write(&destination, b"existing artifact").unwrap();
    assert!(live.publish().is_err());
    assert_eq!(fs::read(retained.path).unwrap(), b"test");
    assert_eq!(fs::read(destination).unwrap(), b"existing artifact");
}
