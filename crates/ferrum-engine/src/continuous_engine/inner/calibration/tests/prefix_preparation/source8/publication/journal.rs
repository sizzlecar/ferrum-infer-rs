//! The original CPU source8 driver writes its exact collector stream. A file
//! is not a model: only independently replayed checkpoint prefixes qualify.
use super::*;
use crate::continuous_engine::inner::cost_observation::{
    PreparedSourceJournal, PreparedSourceJournalFailure, PreparedSourceJournalLimits,
    PreparedSourceJournalStage, PreparedSourceJournalStatus,
};
use ferrum_scheduler::implementations::continuous::cost_profile::{
    self as file, StructuredPreparedOwnerBlockCollectorV8 as Collector,
    StructuredPreparedOwnerBlockHeaderV8 as Header, StructuredPreparedOwnerBlockRecordV8 as Record,
};
use sha2::{Digest, Sha256};
use std::{fs, path::PathBuf};
mod automatic_reuse;
mod restart;

struct Directory(PathBuf);
impl Directory {
    fn new() -> Self {
        let path =
            std::env::temp_dir().join(format!("ferrum-source8-journal-{}", uuid::Uuid::new_v4()));
        fs::create_dir(&path).unwrap();
        Self(path)
    }
}
impl Drop for Directory {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}
fn limits() -> PreparedSourceJournalLimits {
    PreparedSourceJournalLimits {
        maximum_source_bytes: NonZeroU64::new(16 * 1024 * 1024).unwrap(),
        maximum_record_bytes: NonZeroUsize::new(1024 * 1024).unwrap(),
        maximum_checkpoints: NonZeroUsize::new(4).unwrap(),
        maximum_retained_bytes: NonZeroUsize::new(64 * 1024).unwrap(),
    }
}

fn worker_retained(runtime: &EngineCostRuntime) -> u64 {
    serde_json::to_value(runtime.sink.stats()).unwrap()["memory"]["worker_and_result_bytes"]
        .as_u64()
        .unwrap()
}

async fn collect_source(
    session: &mut CalibrationSession,
    counts: [usize; 3],
    deadline: tokio::time::Instant,
) {
    let cohorts = counts.iter().sum::<usize>();
    let mut budget = ProbeExecutionBudget::new(
        deadline,
        NonZeroUsize::new(cohorts * 2).unwrap(),
        NonZeroUsize::new(cohorts * 3).unwrap(),
    );
    let settings = ProbeCohortSettings {
        prefill_plan:
            crate::continuous_engine::inner::calibration::cohort_driver::ProbePrefillPlan::Joint,
        prefill_chunk: NonZeroU32::MIN,
        decode_route: CalibrationDecodeRoute::FullLogits,
        reset_token_policy: false,
    };
    for (pass, count) in counts.into_iter().enumerate() {
        for ordinal in 0..count {
            session.begin_prepared_owner_cohort(pass, ordinal).unwrap();
            let requests = probe_requests(session);
            let report = session
                .run_probe_cohort(requests, settings, &mut budget)
                .await
                .unwrap();
            assert_eq!(
                (report.completed_requests, report.completed_output_tokens),
                (2, 6)
            );
            assert_eq!((report.wave_attempts, report.reconciled_waves), (3, 3));
            session.end_prepared_owner_cohort().unwrap();
        }
    }
}

#[tokio::test]
async fn source8_original_journal_replays_real_cohorts_partial_tail_and_original_checkpoints() {
    let directory = Directory::new();
    let destination = directory.0.join("source.jsonl");
    let journal = PreparedSourceJournal::create(&destination, limits()).unwrap();
    let observer = journal.observer();
    let (mut session, executor) = automatic_session().await;
    let runtime = session.engine.inner.cost_runtime.as_ref().unwrap().clone();
    let deadline = tokio::time::Instant::now() + Duration::from_secs(45);
    let mut declared = full_population(&session);
    // Preserve all three original qualification phases. One extra complete
    // cohort leaves a real audit-only tail after the already complete blocks.
    let mut tail = declared.cohort_plan.phases[2][0].clone();
    tail.manifest_case = 8;
    declared.cohort_plan.phases[2].push(tail);
    let tail_prefix = declared.prefix_plan.phases[2][0].clone();
    declared.prefix_plan.phases[2].push(tail_prefix);
    declared.maximum_offered_waves = 99;
    session.begin_startup_owner_series(1, deadline).unwrap();
    session
        .begin_prepared_owner_source_with_journal(
            declared,
            CostProfileLoadLimits::default(),
            Some(journal),
        )
        .await
        .unwrap();
    collect_source(&mut session, [16, 8, 9], deadline).await;
    assert_eq!(executor.physical.load(Ordering::Acquire), 99);
    let first = session
        .prepared_owner_capture
        .as_mut()
        .unwrap()
        .checkpoint()
        .unwrap();
    assert!(first.qualified_children() > 0);
    let first_receipt = first.source_receipt();
    drop(first);
    assert!(session.activate_prepared_owner_source().await.unwrap() > 0);
    let installed = runtime.startup_series_children_for_test().unwrap();
    assert!(!installed.is_empty());
    let last_checkpoint = session
        .prepared_owner_capture
        .as_ref()
        .unwrap()
        .source_receipt();
    assert!(last_checkpoint.0 > first_receipt.0);
    assert!(matches!(
        observer.status().as_ref(),
        PreparedSourceJournalStatus::Recording
    ));
    session.retire_startup_owner_source().await.unwrap();
    let status = observer.status();
    let PreparedSourceJournalStatus::Completed(archive) = status.as_ref() else {
        panic!("original journal did not complete: {:?}", observer.status());
    };
    assert_eq!(archive.path, fs::canonicalize(&destination).unwrap());
    assert_eq!(archive.checkpoints.len(), 2);
    assert_eq!(
        (
            archive.checkpoints[0].source_bytes,
            archive.checkpoints[0].source_sha256
        ),
        first_receipt,
    );
    assert_eq!(
        (
            archive.checkpoints[1].source_bytes,
            archive.checkpoints[1].source_sha256
        ),
        last_checkpoint,
    );
    assert!(installed
        .iter()
        .all(|child| child.provenance().source_sha256 == last_checkpoint.1));
    let bytes = fs::read(&destination).unwrap();
    assert_eq!(
        (archive.bytes, archive.sha256),
        (bytes.len() as u64, Sha256::digest(&bytes).into())
    );
    assert!(
        archive.bytes > last_checkpoint.0,
        "original footer follows the selected prefix"
    );
    let load_limits = CostProfileLoadLimits::default();
    for original in &archive.checkpoints {
        let prefix = &bytes[..usize::try_from(original.source_bytes).unwrap()];
        let replayed = file::replay_structured_source_v8(prefix, &load_limits).unwrap();
        assert_eq!(
            replayed.source_receipt(),
            (original.source_bytes, original.source_sha256)
        );
        assert_eq!(replayed.qualified_children(), original.qualified_children);
        assert_eq!(replayed.qualified_children(), installed.len());
    }
    let replayed =
        file::replay_structured_source_v8(&bytes[..last_checkpoint.0 as usize], &load_limits)
            .unwrap();
    let now = file::StructuredServiceClockV7 {
        monotonic_ns: runtime.clock.now_ns().unwrap(),
        wall_unix_ns: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
            .try_into()
            .unwrap(),
    };
    let restored = replayed
        .activate_same_process_memory(now, &load_limits)
        .unwrap();
    for original in &installed {
        let matched = restored
            .children
            .iter()
            .find(|child| child.domain_signature() == original.domain_signature())
            .unwrap();
        assert_eq!(
            matched.parameters_signature(),
            original.parameters_signature()
        );
        assert_eq!(
            matched.provenance().source_sha256,
            original.provenance().source_sha256
        );
    }
    // Independently consume the entire archived stream, including its footer.
    // This catches a missed or duplicated original event even after checkpoint.
    let mut lines = bytes.split_inclusive(|b| *b == b'\n');
    let header: Header = serde_json::from_slice(lines.next().unwrap()).unwrap();
    let mut collector = Collector::new(header, load_limits.clone()).unwrap();
    let mut kinds = Vec::new();
    for line in lines {
        let original: Record = serde_json::from_slice(line).unwrap();
        let kind: serde_json::Value = serde_json::from_slice(line).unwrap();
        kinds.push(kind["kind"].as_str().unwrap().to_owned());
        collector.push(&original).unwrap();
    }
    assert_eq!(collector.source_receipt(), (archive.bytes, archive.sha256));
    assert_eq!(
        kinds
            .iter()
            .filter(|kind| kind.as_str() == "partial_tail_closed")
            .count(),
        1
    );
    assert_eq!(
        kinds
            .iter()
            .filter(|kind| kind.as_str() == "checkpoint")
            .count(),
        2
    );
    assert_eq!(kinds.last().map(String::as_str), Some("footer"));
    assert!(collector.audit().closed);
    assert!(
        file::replay_structured_source_v8(&bytes, &load_limits).is_err(),
        "footer is not a checkpoint"
    );
    assert!(file::replay_structured_source_v8(
        &bytes[..first_receipt.0 as usize - 1],
        &load_limits
    )
    .is_err());
    session.finish_startup_owner_series().unwrap();
    session.shutdown().await.unwrap();
}

#[tokio::test]
async fn source8_original_journal_capacity_failure_keeps_actual_qualification_and_requests() {
    let directory = Directory::new();
    let destination = directory.0.join("source.jsonl");
    let mut bounded = limits();
    bounded.maximum_source_bytes = NonZeroU64::MIN;
    let journal = PreparedSourceJournal::create(&destination, bounded).unwrap();
    let observer = journal.observer();
    let (mut session, executor) = automatic_session().await;
    let runtime = session.engine.inner.cost_runtime.as_ref().unwrap().clone();
    let deadline = tokio::time::Instant::now() + Duration::from_secs(45);
    session.begin_startup_owner_series(1, deadline).unwrap();
    let declared = full_population(&session);
    session
        .begin_prepared_owner_source_with_journal(
            declared,
            CostProfileLoadLimits::default(),
            Some(journal),
        )
        .await
        .unwrap();
    assert_eq!(
        observer.status().as_ref(),
        &PreparedSourceJournalStatus::Failed {
            stage: PreparedSourceJournalStage::Header,
            reason: PreparedSourceJournalFailure::EncodingOrCapacity,
        }
    );
    collect_source(&mut session, [16, 8, 8], deadline).await;
    assert_eq!(executor.physical.load(Ordering::Acquire), 96);
    assert!(session.activate_prepared_owner_source().await.unwrap() > 0);
    let before = runtime.snapshot().unwrap();
    session.retire_startup_owner_source().await.unwrap();
    assert!(Arc::ptr_eq(&before, &runtime.snapshot().unwrap()));
    assert!(!destination.exists());
    session.finish_startup_owner_series().unwrap();
    session.shutdown().await.unwrap();
}

#[tokio::test]
async fn source8_original_journal_failed_source_retains_original_failure_without_a_checkpoint() {
    let directory = Directory::new();
    let journal =
        PreparedSourceJournal::create(&directory.0.join("failed.jsonl"), limits()).unwrap();
    let observer = journal.observer();
    let (mut session, _) = automatic_session().await;
    let declared = full_population(&session);
    session
        .begin_prepared_owner_source_with_journal(
            declared,
            CostProfileLoadLimits::default(),
            Some(journal),
        )
        .await
        .unwrap();
    let source = session.prepared_owner_capture.as_mut().unwrap();
    source.invalidate("original pre-cohort failure");
    source.finish_journal();
    let expected = source.source_receipt();
    let status = observer.status();
    let PreparedSourceJournalStatus::Completed(archive) = status.as_ref() else {
        panic!(
            "failed original source must still have an exact archive: {:?}",
            observer.status()
        );
    };
    assert!(
        archive.checkpoints.is_empty(),
        "a failed source has no publication prefix"
    );
    assert!(!archive.population_complete);
    assert_eq!((archive.bytes, archive.sha256), expected);
    let bytes = fs::read(&archive.path).unwrap();
    let kinds: Vec<_> = bytes
        .split_inclusive(|b| *b == b'\n')
        .skip(1)
        .map(|line| {
            serde_json::from_slice::<serde_json::Value>(line).unwrap()["kind"]
                .as_str()
                .unwrap()
                .to_owned()
        })
        .collect();
    assert_eq!(kinds, ["failed", "footer"]);
    assert!(file::replay_structured_source_v8(&bytes, &CostProfileLoadLimits::default()).is_err());
    assert!(session
        .engine
        .inner
        .cost_runtime
        .as_ref()
        .unwrap()
        .snapshot()
        .is_none());
    session.shutdown().await.unwrap();
}

#[tokio::test]
async fn source8_original_journal_directory_default_entry_replays_publication_and_releases_memory()
{
    let directory = Directory::new();
    let policy = ferrum_types::SloAutomaticCalibrationDiagnosticsV1::Directory {
        directory: directory.0.clone(),
        maximum_source_bytes: NonZeroU64::new(16 * 1024 * 1024).unwrap(),
        maximum_total_bytes: NonZeroU64::new(64 * 1024 * 1024).unwrap(),
        maximum_retained_generations: NonZeroUsize::new(4).unwrap(),
    };
    let (mut session, executor) = automatic_session_with_diagnostics(
        2,
        Arc::new(AdvancingClock(AtomicU64::new(100))),
        policy,
    )
    .await;
    let runtime = session.engine.inner.cost_runtime.as_ref().unwrap().clone();
    let before = worker_retained(&runtime);
    let deadline = tokio::time::Instant::now() + Duration::from_secs(45);
    session.begin_startup_owner_series(1, deadline).unwrap();
    let declared = full_population(&session);
    session
        .begin_prepared_owner_source(declared, CostProfileLoadLimits::default())
        .await
        .unwrap();
    let observer = session
        .prepared_owner_capture
        .as_ref()
        .unwrap()
        .journal_observer()
        .expect("typed Directory policy activates the original source sink");
    assert!(worker_retained(&runtime) > before);
    collect_source(&mut session, [16, 8, 8], deadline).await;
    assert_eq!(executor.physical.load(Ordering::Acquire), 96);
    assert!(session.activate_prepared_owner_source().await.unwrap() > 0);
    let original_prefix = session
        .prepared_owner_capture
        .as_ref()
        .unwrap()
        .source_receipt();
    let installed = runtime.startup_series_children_for_test().unwrap();
    session.retire_startup_owner_source().await.unwrap();
    let retained = worker_retained(&runtime);
    let status = observer.status();
    let PreparedSourceJournalStatus::Completed(archive) = status.as_ref() else {
        panic!("Directory archive did not complete: {status:?}");
    };
    assert!(archive.population_complete);
    assert!(archive
        .path
        .starts_with(fs::canonicalize(directory.0.join("ferrum-automatic-v1")).unwrap()));
    assert_eq!(archive.checkpoints.len(), 1);
    assert_eq!(
        (
            archive.checkpoints[0].source_bytes,
            archive.checkpoints[0].source_sha256
        ),
        original_prefix
    );
    assert_eq!(archive.checkpoints[0].qualified_children, installed.len());
    let bytes = fs::read(&archive.path).unwrap();
    assert_eq!(
        (archive.bytes, archive.sha256),
        (bytes.len() as u64, Sha256::digest(&bytes).into())
    );
    let replayed = file::replay_structured_source_v8(
        &bytes[..original_prefix.0 as usize],
        &CostProfileLoadLimits::default(),
    )
    .unwrap();
    assert_eq!(replayed.source_receipt(), original_prefix);
    assert_eq!(replayed.qualified_children(), installed.len());
    assert!(installed
        .iter()
        .all(|child| child.provenance().source_sha256 == original_prefix.1));
    drop(observer);
    assert_eq!(worker_retained(&runtime), retained);
    drop(status);
    assert!(worker_retained(&runtime) < retained);
    session.finish_startup_owner_series().unwrap();
    session.shutdown().await.unwrap();
}

#[tokio::test]
async fn source8_original_journal_memory_only_default_has_no_archive_or_reservation() {
    let (mut session, _) = automatic_session().await;
    let runtime = session.engine.inner.cost_runtime.as_ref().unwrap().clone();
    let before = worker_retained(&runtime);
    let declared = full_population(&session);
    session
        .begin_prepared_owner_source(declared, CostProfileLoadLimits::default())
        .await
        .unwrap();
    assert!(session
        .prepared_owner_capture
        .as_ref()
        .unwrap()
        .journal_observer()
        .is_none());
    assert_eq!(worker_retained(&runtime), before);
    session.shutdown().await.unwrap();
}
