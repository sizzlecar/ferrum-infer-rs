//! Exercise the real Directory writer and collector against the same original
//! CPU settlements, then independently load the resulting profile14.
use super::*;
use sha2::{Digest, Sha256};

struct Directory(PathBuf);
impl Directory {
    fn new() -> Self {
        Self(std::env::temp_dir().join(format!(
            "ferrum-owner-block-successful-archive-{}",
            uuid::Uuid::new_v4()
        )))
    }
}
impl Drop for Directory {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

#[tokio::test]
async fn automatic_family_progress_directory_success_replays_and_imports_both_families() {
    let directory = Directory::new();
    let mut f = Families::new_with_diagnostics(SloAutomaticCalibrationDiagnosticsV1::Directory {
        directory: directory.0.clone(),
        maximum_source_bytes: NonZeroU64::new(16 * 1024 * 1024).unwrap(),
        maximum_total_bytes: NonZeroU64::new(128 * 1024 * 1024).unwrap(),
        maximum_retained_generations: NonZeroUsize::new(4).unwrap(),
    });
    for algorithm in [A, A, B, B, A, A, B, B] {
        f.block(algorithm);
    }
    assert_eq!(f.recorded, 8 * OFFERS as u64);
    assert_eq!(f.live().audit().qualified_publications, 2);
    assert!(f.known(A));
    assert!(f.known(B));
    let now = f.clock.now_ns().unwrap();
    let children = f.runtime.training.live_catalog_children(now).unwrap();
    assert_eq!(children.len(), 2);
    let fingerprint = children[0].fingerprint().clone();
    let queries = [f.query(A), f.query(B)];
    let expected: Vec<_> = queries
        .iter()
        .map(|query| {
            let known: Vec<_> = children
                .iter()
                .filter_map(|child| {
                    child
                        .predict_query_local(&fingerprint, query, now)
                        .ok()
                        .map(|prediction| {
                            (
                                *child.domain_signature(),
                                child.parameters_signature(),
                                prediction,
                            )
                        })
                })
                .collect();
            assert_eq!(
                known.len(),
                1,
                "each original future query selects its own family"
            );
            known.into_iter().next().unwrap()
        })
        .collect();
    drop(children);

    // Shutdown seals the still-open diagnostic journal. It does not invent
    // an extra offer or rebind the process-local predictors to a file path.
    f.runtime.shutdown().await.unwrap();
    let audit = f.live().audit();
    assert_eq!(audit.qualified_publications, 2, "{audit:#?}");
    assert_eq!(audit.failed_generations, 0, "{audit:#?}");
    assert!(audit.publication_error.is_none(), "{audit:#?}");
    let automatic = audit.automatic.as_ref().unwrap();
    assert!(automatic.stopped);
    assert!(automatic.owner_blocks.is_none());
    assert_eq!(automatic.diagnostic_failures.count, 0, "{automatic:#?}");
    let source = &automatic
        .last_diagnostic_publication
        .as_ref()
        .expect("successful epoch publishes its original Directory journal")
        .source;
    let bytes = fs::read(&source.path).unwrap();
    let digest: [u8; 32] = Sha256::digest(&bytes).into();
    assert_eq!(source.bytes, bytes.len() as u64);
    assert_eq!(source.digest, digest);
    assert_eq!(source.sha256, format!("{:x}", Sha256::digest(&bytes)));

    let mut offset = 0usize;
    let mut checkpoint_end = None;
    let mut completed = 0usize;
    let mut closing_wall = None;
    for line in bytes.split_inclusive(|byte| *byte == b'\n') {
        offset += line.len();
        let record: serde_json::Value = serde_json::from_slice(line).unwrap();
        match record["kind"].as_str() {
            Some("completed") => completed += 1,
            Some("checkpoint") => checkpoint_end = Some(offset),
            Some("footer") => {
                assert!(!record["incomplete_block"].as_bool().unwrap());
                assert_eq!(offset, bytes.len(), "footer is the final journal record");
                closing_wall = record["closing"]["wall_unix_ns"].as_u64();
            }
            Some("failed") => panic!("successful journal contains a failed block"),
            _ => {}
        }
    }
    assert_eq!(completed, 8 * OFFERS);
    let checkpoint_end = checkpoint_end.expect("complete two-family checkpoint");
    assert!(
        checkpoint_end < bytes.len(),
        "sealed journal retains its footer"
    );
    let prefix = &bytes[..checkpoint_end];
    let prefix_digest: [u8; 32] = Sha256::digest(prefix).into();
    let limits = file::CostProfileLoadLimits::default();
    let checkpoint = file::replay_structured_source_v7(prefix, &limits).unwrap();
    assert_eq!(checkpoint.qualified_children(), 2);
    assert_eq!(
        checkpoint.source_receipt(),
        (prefix.len() as u64, prefix_digest)
    );
    drop(checkpoint);

    let profile_path = directory.0.join("qualified-profile14.json");
    let exported = file::export_structured_profile_v14(
        &source.path,
        digest,
        checkpoint_end as u64,
        &profile_path,
        0,
        &limits,
    )
    .unwrap();
    assert_eq!(exported.schema_version, 14);
    assert_eq!(exported.children.len(), 2);
    assert_eq!(exported.source_bytes, prefix.len() as u64);
    assert_eq!(exported.source_sha256, prefix_digest);
    let load_now = 1_000;
    let imported = file::load_structured_profile_v14(
        &profile_path,
        &fingerprint,
        &limits,
        file::ProfileLoadClock {
            wall_unix_ns: Some(closing_wall.unwrap() + 1),
            wall_max_error_ns: Some(0),
            monotonic_now_ns: load_now,
        },
    )
    .unwrap();
    assert_eq!(imported.journal_bytes, source.bytes);
    assert_eq!(imported.source_bytes, prefix.len() as u64);
    assert_eq!(imported.source_sha256, prefix_digest);
    assert_eq!(imported.offered_attempts, 8 * OFFERS as u64);
    assert_eq!(imported.workload_domain(), Some(&f.domain));
    assert_eq!(imported.children.len(), 2);
    for (query, (domain, parameters, before)) in queries.iter().zip(expected) {
        let child = imported
            .children
            .iter()
            .find(|child| *child.domain_signature() == domain)
            .unwrap();
        assert_eq!(child.parameters_signature(), parameters);
        assert_eq!(child.provenance().storage, SloCostProfileStorage::File);
        let after = child
            .predict_query_local(&fingerprint, query, load_now)
            .unwrap();
        assert_eq!(after.planning_ns, before.planning_ns);
        assert_eq!(after.fitted_upper_ns, before.fitted_upper_ns);
        assert_eq!(after.valid_until_ns, before.valid_until_ns);
    }
    let sink = f.runtime.audit_snapshot().sink;
    assert_eq!(sink.raw_offered, f.recorded);
    assert_eq!(sink.raw_accepted, f.recorded);
    assert_eq!(sink.raw_resolved, f.recorded);
    assert_eq!(sink.raw_resolution_failed, 0);
    assert_eq!(sink.raw_lost, 0);
    assert_eq!(sink.raw_pending, 0);
}
