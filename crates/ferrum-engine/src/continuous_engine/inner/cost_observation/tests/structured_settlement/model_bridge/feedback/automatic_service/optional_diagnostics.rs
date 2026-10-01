//! Real Store/StagedFile failures cannot replace the numerical authority.
use super::*;

fn obstruct_store(directory: &Directory) {
    fs::create_dir_all(&directory.path).unwrap();
    // A user-owned file must never be removed or treated as our archive.
    fs::write(directory.path.join("ferrum-automatic-v1"), b"original file").unwrap();
}

fn assert_memory_model(f: &AutomaticFixture) {
    let audit = f.live().audit();
    assert_eq!(audit.qualified_publications, 1);
    assert_eq!(audit.failed_generations, 0);
    assert!(audit.publication_error.is_none());
    let receipt = f.runtime.training.published_catalog_receipt().unwrap();
    assert_eq!(receipt.storage, SloCostProfileStorage::Memory);
    assert!(receipt.path.is_none());
    assert!(receipt
        .structured_whole_wave_v2
        .as_ref()
        .unwrap()
        .children
        .iter()
        .all(|child| {
            child.storage == SloCostProfileStorage::Memory
                && child.profile_path.is_none()
                && child.source_path.is_none()
        }));
    f.runtime
        .snapshot()
        .unwrap()
        .audit_structured_query_v2(&f.query, f.clock.now_ns().unwrap())
        .unwrap();
}

fn assert_diagnostic_failure(f: &AutomaticFixture, stage: &str) {
    let audit = f.live().audit().automatic.unwrap();
    assert!(audit.last_diagnostic_publication.is_none());
    let audit = serde_json::to_value(audit).unwrap();
    let failures = &audit["diagnostic_failures"];
    assert_eq!(failures["count"], 1);
    for failure in [&failures["first"], &failures["last"]] {
        assert_eq!(failure["generation"], 1);
        assert!(failure["reason"].as_str().unwrap().chars().count() <= 1024);
        assert_eq!(failure["stage"], stage);
    }
}

#[tokio::test]
async fn automatic_optional_directory_open_failure_keeps_memory_qualification() {
    let directory = Directory::new();
    obstruct_store(&directory);
    let f = AutomaticFixture::with_diagnostics(directory.policy());
    f.start();
    f.generation(1);
    assert_memory_model(&f);
    assert_diagnostic_failure(&f, "open");
    assert_eq!(
        fs::read(directory.path.join("ferrum-automatic-v1")).unwrap(),
        b"original file"
    );
    f.runtime.shutdown().await.unwrap();
}

#[tokio::test]
async fn automatic_optional_directory_total_quota_keeps_memory_qualification() {
    let directory = Directory::new();
    let mut policy = directory.policy();
    let SloAutomaticCalibrationDiagnosticsV1::Directory {
        maximum_source_bytes,
        maximum_total_bytes,
        ..
    } = &mut policy
    else {
        unreachable!()
    };
    // The typed settings are valid, but this real Store cannot reserve both
    // temporary and published names plus control bytes inside this quota.
    *maximum_total_bytes = *maximum_source_bytes;
    policy.validate().unwrap();
    let f = AutomaticFixture::with_diagnostics(policy);
    f.start();
    f.generation(1);
    assert_memory_model(&f);
    assert_diagnostic_failure(&f, "open");
    f.runtime.shutdown().await.unwrap();
}

#[tokio::test]
async fn automatic_optional_directory_write_quota_keeps_memory_qualification() {
    // Derive a real archive budget that fits header+open but cannot hold the
    // first completed wave, using the same real fixture and writer protocol.
    let baseline_directory = Directory::new();
    let baseline = AutomaticFixture::with_diagnostics(baseline_directory.policy());
    baseline.start();
    baseline.generation(1);
    let source = baseline
        .live()
        .audit()
        .automatic
        .unwrap()
        .last_diagnostic_publication
        .unwrap()
        .source;
    let raw = fs::read(source.path).unwrap();
    let mut lines = raw.split_inclusive(|byte| *byte == b'\n');
    let header = lines.next().unwrap();
    let opening = lines.next().unwrap();
    let completed = lines.next().unwrap();
    // Only the fresh 32-byte source id varies in this otherwise identical
    // header. Bound its entire decimal JSON representation, not a measured
    // performance value or a convenient arbitrary file limit.
    let source_id_json_maximum = 2 + 32 * 3 + 31;
    assert!(completed.len() > source_id_json_maximum);
    let quota = header.len() + opening.len() + source_id_json_maximum;
    baseline.runtime.shutdown().await.unwrap();

    let directory = Directory::new();
    let mut policy = directory.policy();
    let SloAutomaticCalibrationDiagnosticsV1::Directory {
        maximum_source_bytes,
        ..
    } = &mut policy
    else {
        unreachable!()
    };
    *maximum_source_bytes = NonZeroU64::new(quota as u64).unwrap();
    let f = AutomaticFixture::with_diagnostics(policy);
    f.start();
    f.generation(1);
    assert_memory_model(&f);
    assert_diagnostic_failure(&f, "write");
    let partial = f.live().audit().last_failed_source.unwrap();
    assert!(!partial.footer_complete);
    assert!(partial.retained_unpublished);
    let raw = fs::read(partial.path.unwrap()).unwrap();
    assert!((raw.len() as u64) <= quota as u64);
    assert!(!raw
        .split(|b| *b == b'\n')
        .filter(|line| !line.is_empty())
        .any(|line| {
            serde_json::from_slice::<serde_json::Value>(line)
                .is_ok_and(|row| row["kind"] == "footer")
        }));
    f.runtime.shutdown().await.unwrap();
}

#[tokio::test]
async fn automatic_optional_directory_publish_collision_keeps_memory_qualification() {
    let directory = Directory::new();
    let f = AutomaticFixture::with_diagnostics(directory.policy());
    f.start();
    // Complete Discovery so the original numerical sidecar is actually open.
    for n in 0..8 {
        f.record(n + 1, "fixture.feedback.a", true);
    }
    assert_eq!(f.live().audit().population.phase, 0);
    let entry = fs::read_dir(directory.path.join("ferrum-automatic-v1"))
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .find(|path| path.is_dir())
        .unwrap();
    let destination = entry.join("source.jsonl");
    fs::write(&destination, b"original destination").unwrap();
    for n in 0..24 {
        f.record(n + 9, "fixture.feedback.a", true);
    }
    assert_memory_model(&f);
    assert_diagnostic_failure(&f, "publish");
    assert_eq!(fs::read(destination).unwrap(), b"original destination");
    let partial = f.live().audit().last_failed_source.unwrap();
    assert!(partial.footer_complete);
    assert!(partial.retained_unpublished);
    assert!(partial.path.unwrap().is_file());
    f.runtime.shutdown().await.unwrap();
}

#[tokio::test]
async fn automatic_optional_directory_failure_never_hides_failed_population() {
    let directory = Directory::new();
    obstruct_store(&directory);
    let f = AutomaticFixture::with_diagnostics(directory.policy());
    f.start();
    for n in 0..8 {
        f.record(n + 1, "fixture.feedback.a", true);
    }
    drop(f.runtime.reserve_live_ticket(f.clock.now_ns()).unwrap());
    f.runtime.consume_samples();
    let audit = f.live().audit();
    assert_eq!(audit.qualified_publications, 0);
    assert_eq!(audit.failed_generations, 1);
    assert!(audit.publication_error.is_some());
    assert!(f.runtime.training.published_catalog_receipt().is_none());
    assert_diagnostic_failure(&f, "open");
    f.runtime.shutdown().await.unwrap();
}

#[tokio::test]
async fn automatic_optional_directory_failure_never_qualifies_an_empty_child() {
    let directory = Directory::new();
    obstruct_store(&directory);
    let f = AutomaticFixture::with_diagnostics(directory.policy());
    f.start();
    for n in 0..8 {
        f.record(n + 1, "fixture.feedback.a", true);
    }
    for n in 0..8 {
        f.record(n + 9, "fixture.feedback.b", true);
    }
    let audit = f.live().audit();
    assert_eq!(audit.qualified_publications, 0);
    assert_eq!(audit.failed_generations, 1);
    assert!(audit
        .publication_error
        .as_ref()
        .unwrap()
        .contains("all frozen numerical children failed"));
    assert!(f.runtime.training.published_catalog_receipt().is_none());
    assert_diagnostic_failure(&f, "open");
    f.runtime.shutdown().await.unwrap();
}

#[tokio::test]
async fn automatic_optional_reference_archive_failure_does_not_block_collection() {
    let directory = Directory::new();
    obstruct_store(&directory);
    let f = AutomaticFixture::with_diagnostics(directory.policy());
    // This API only persists bytes; it grants no reference/model authority.
    // The original startup path calls it after reference qualification.
    f.runtime
        .persist_automatic_reference(b"source bytes", b"reference bytes")
        .unwrap();
    let audit = serde_json::to_value(f.live().audit().automatic.unwrap()).unwrap();
    assert_eq!(audit["diagnostic_failures"]["count"], 1);
    let failure = &audit["diagnostic_failures"]["first"];
    assert_eq!(failure["generation"], 0);
    assert_eq!(failure["stage"], "reference");
    f.start();
    f.generation(1);
    assert_memory_model(&f);
    let audit = serde_json::to_value(f.live().audit().automatic.unwrap()).unwrap();
    assert_eq!(audit["diagnostic_failures"]["count"], 2);
    f.runtime.shutdown().await.unwrap();
}

#[tokio::test]
async fn automatic_optional_directory_receipt_mismatch_keeps_only_memory_model() {
    use sha2::Digest;
    let directory = Directory::new();
    let f = AutomaticFixture::with_diagnostics(directory.policy());
    f.start();
    for n in 0..8 {
        f.record(n + 1, "fixture.feedback.a", true);
    }
    f.live().test_duplicate_phase_in_optional_source().unwrap();
    for n in 0..24 {
        f.record(n + 9, "fixture.feedback.a", true);
    }
    assert_memory_model(&f);
    assert_diagnostic_failure(&f, "receipt_mismatch");
    let failed = f.live().audit().last_failed_source.unwrap();
    assert!(failed.footer_complete);
    assert!(!failed.retained_unpublished);
    let raw = fs::read(failed.path.unwrap()).unwrap();
    assert_eq!(raw.len() as u64, failed.bytes);
    let disk_digest: [u8; 32] = sha2::Sha256::digest(&raw).into();
    let receipt = f.runtime.training.published_catalog_receipt().unwrap();
    let child = &receipt.structured_whole_wave_v2.as_ref().unwrap().children[0];
    assert_ne!(disk_digest, child.source_sha256);
    assert_ne!(failed.bytes, child.source_bytes);
    f.runtime.shutdown().await.unwrap();
}

#[tokio::test]
async fn automatic_optional_directory_failure_never_hides_collector_push_rejection() {
    let directory = Directory::new();
    obstruct_store(&directory);
    let f = AutomaticFixture::with_diagnostics(directory.policy());
    f.start();
    for n in 0..8 {
        f.record(n + 1, "fixture.feedback.a", true);
    }
    assert_diagnostic_failure(&f, "open");
    let rejection = f.live().test_duplicate_phase_in_collector().unwrap_err();
    assert!(rejection
        .to_string()
        .contains("source6 phase open or explicit gap differs"));
    // Real original offers still settle through FIFO. Closing this phase must
    // propagate the collector's poisoned state through the normal controller.
    for n in 0..8 {
        f.record(n + 9, "fixture.feedback.a", true);
    }
    let audit = f.live().audit();
    assert_eq!(audit.qualified_publications, 0);
    assert_eq!(audit.failed_generations, 1);
    assert!(audit
        .publication_error
        .as_ref()
        .unwrap()
        .contains("source6 is closed"));
    assert!(f.runtime.snapshot().is_none());
    assert!(f.runtime.training.published_catalog_receipt().is_none());
    assert_diagnostic_failure(&f, "open");
    f.runtime.shutdown().await.unwrap();
}
