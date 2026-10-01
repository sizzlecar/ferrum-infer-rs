//! Storage/lifecycle tests intentionally grant no qualified-model authority.
//! Product replay tests must additionally use real source7/8 and feedback proof.
use super::*;
use ferrum_scheduler::implementations::continuous::cost_profile::StructuredSourceRecordSinkV1;

#[test]
fn automatic_reuse_abandoned_unconstructed_source_releases_writer_and_remains_unusable() {
    let dir = Directory::new();
    let opened = CacheSession::open_at(&dir.0, identity(1), 1, limits()).unwrap();
    let journal = opened.session.original_source_raw().unwrap();
    let observer = journal.clone();
    assert_eq!(opened.session.0.ledger.lock().active_writers, 1);
    journal.abandon();
    observer.abandon();
    assert_eq!(opened.session.0.ledger.lock().active_writers, 0);
    assert_eq!(journal.failure(), Some(CacheMiss::Dirty));
    assert_eq!(journal.sealed(), Err(CacheMiss::Dirty));
    assert!(journal.checkpoint((0, Sha256::digest([]).into())).is_err());
    assert!(!journal.discard_if_unobserved().unwrap());
    drop(observer);
    assert!(journal.discard_if_unobserved().unwrap());
    assert_eq!(opened.session.0.ledger.lock().sources, 0);
    assert!(opened.session.0.root.join("dirty").exists());
    assert!(!opened.session.0.root.join("manifest.json").exists());
    let mut failed = opened.session.original_source_raw().unwrap();
    failed.append_original(b"x", (1, [0; 32]));
    assert_eq!(failed.failure(), Some(CacheMiss::Corrupt));
    failed.abandon();
    assert_eq!(failed.failure(), Some(CacheMiss::Corrupt));
    assert_eq!(opened.session.0.ledger.lock().active_writers, 0);
}

fn identity(boot: u8) -> CacheIdentity {
    CacheIdentity {
        binding: [19; 32],
        clock: CostMonotonicDomainV1::new_macos_continuous([boot; 16]).unwrap(),
    }
}
fn limits() -> CacheLimits {
    CacheLimits {
        maximum_bytes: 256 * 1024 * 1024,
        maximum_source_bytes: 1024 * 1024,
        maximum_sources: 2,
        maximum_retained_bytes: 1024 * 1024,
        maximum_samples: 20,
        maximum_rows: 40,
        maximum_duration: Duration::from_secs(30),
    }
}
struct Directory(PathBuf);
impl Directory {
    fn new() -> Self {
        let p = std::env::temp_dir().join(format!("ferrum-reuse-store-{}", uuid::Uuid::new_v4()));
        fs::create_dir(&p).unwrap();
        Self(p)
    }
}
impl Drop for Directory {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn storage_payload(session: &CacheSession) -> (ArtifactReceipt, Vec<CachedSource>) {
    let tx = session.transaction().unwrap();
    let mut source = session.original_source_raw().unwrap();
    // These are byte-ledger fixtures, not serialized model evidence.
    let bytes = b"original-header\noriginal-checkpoint\n";
    let hash: [u8; 32] = Sha256::digest(bytes).into();
    source.append_original(bytes, (bytes.len() as u64, hash));
    source.checkpoint((bytes.len() as u64, hash)).unwrap();
    let journal = source.seal((bytes.len() as u64, hash)).unwrap();
    drop(source);
    let profile = session.reserve_external(64, &tx).unwrap();
    fs::write(&profile.path, b"independent-profile-fixture").unwrap();
    let profile = profile.finish(&tx).unwrap();
    let feedback = session.reserve_external(64, &tx).unwrap();
    fs::write(&feedback.path, b"preserved-feedback-fixture").unwrap();
    let feedback = feedback.finish(&tx).unwrap();
    let sources = vec![CachedSource {
        kind: SourceKind::PreparedOwnersV8,
        encoding: JournalEncoding::IdentityV1,
        journal,
        checkpoint_bytes: journal.bytes,
        checkpoint_sha256: journal.sha256,
        profile,
        domains: vec![],
    }];
    (feedback, sources)
}
fn storage_commit(session: &CacheSession, expires: u64) -> (ArtifactReceipt, Vec<CachedSource>) {
    let (feedback, sources) = storage_payload(session);
    let tx = session.transaction().unwrap();
    session
        .commit(sources.clone(), feedback, Some(expires), 1, &tx)
        .unwrap();
    (feedback, sources)
}

#[test]
fn automatic_reuse_store_clean_candidate_then_dirty_session_never_rolls_back() {
    let dir = Directory::new();
    let first = CacheSession::open_at(&dir.0, identity(1), 1, limits()).unwrap();
    assert!(matches!(first.candidate, Err(CacheMiss::Missing)));
    let (_, sources) = storage_commit(&first.session, 100);
    drop(first);
    let second = CacheSession::open_at(&dir.0, identity(1), 10, limits()).unwrap();
    assert_eq!(second.candidate.as_ref().unwrap().sources, sources);
    assert!(second.session.0.root.join("dirty").is_file());
    assert!(matches!(
        CacheSession::open_at(&dir.0, identity(1), 10, limits()),
        Err(CacheMiss::Busy)
    ));
    // Unacknowledged Drop leaves dirty and cannot resurrect the previous clean
    // catalog with its older corrections on the next process.
    drop(second);
    let third = CacheSession::open_at(&dir.0, identity(1), 11, limits()).unwrap();
    assert!(matches!(third.candidate, Err(CacheMiss::Dirty)));
    storage_commit(&third.session, 100);
    drop(third);
    let recovered = CacheSession::open_at(&dir.0, identity(1), 12, limits()).unwrap();
    assert!(recovered.candidate.is_ok());
}

#[test]
fn automatic_reuse_private_copy_failures_preserve_committed_generation() {
    for failure in [CacheMiss::Capacity, CacheMiss::Corrupt, CacheMiss::Deadline] {
        let dir = Directory::new();
        let cold = CacheSession::open_at(&dir.0, identity(1), 1, limits()).unwrap();
        storage_commit(&cold.session, 100);
        drop(cold);
        let warm = CacheSession::open_at(&dir.0, identity(1), 2, limits()).unwrap();
        let candidate = warm.candidate.as_ref().unwrap();
        let manifest_path = warm.session.0.root.join("manifest.json");
        let original_manifest = fs::read(&manifest_path).unwrap();
        let mut source = candidate.sources[0].clone();
        let original_path = warm
            .session
            .artifact_path(candidate.generation, &source.journal);
        let original = fs::read(&original_path).unwrap();
        let mut tx = warm.session.transaction().unwrap();
        match failure {
            CacheMiss::Capacity => {
                // One byte less than the required independent physical copy.
                let available = warm.session.0.limits.maximum_bytes - warm.session.used_bytes();
                warm.session
                    .0
                    .charge(available - source.journal.bytes + 1)
                    .unwrap();
            }
            CacheMiss::Corrupt => source.journal.sha256[0] ^= 1,
            CacheMiss::Deadline => tx.deadline = Instant::now(),
            _ => unreachable!(),
        }
        assert!(matches!(
            warm.session.adopt_source(candidate.generation, &source, &tx),
            Err(reason) if reason == failure
        ));
        assert_eq!(warm.session.0.ledger.lock().active_writers, 0);
        assert_eq!(warm.session.0.ledger.lock().sources, 0);
        for receipt in candidate.artifacts() {
            store::verify_file(
                &warm.session.artifact_path(candidate.generation, receipt),
                receipt,
                &warm.session.transaction().unwrap(),
            )
            .unwrap();
        }
        let dirty_path = warm.session.0.root.join("dirty");
        drop(warm);
        assert!(dirty_path.is_file());
        assert_eq!(fs::read(&manifest_path).unwrap(), original_manifest);
        assert_eq!(fs::read(&original_path).unwrap(), original);
    }
}

#[test]
fn automatic_reuse_private_mutable_copy_keeps_lifetime_reservation_and_old_bytes() {
    let dir = Directory::new();
    let cold = CacheSession::open_at(&dir.0, identity(1), 1, limits()).unwrap();
    storage_commit(&cold.session, 100);
    drop(cold);
    let warm = CacheSession::open_at(&dir.0, identity(1), 2, limits()).unwrap();
    let candidate = warm.candidate.as_ref().unwrap();
    let committed_path = warm
        .session
        .artifact_path(candidate.generation, &candidate.feedback);
    let committed = fs::read(&committed_path).unwrap();
    let before = warm.session.used_bytes();
    let lifetime_bytes = candidate.feedback.bytes * 2 + 64;
    let private = warm
        .session
        .reserve_external(lifetime_bytes, &warm.transaction)
        .unwrap();
    private
        .copy_verified_from(&committed_path, &candidate.feedback, &warm.transaction)
        .unwrap();
    let private_path = private.path.clone();
    drop(private);
    assert_eq!(warm.session.0.ledger.lock().active_writers, 0);
    assert_eq!(warm.session.used_bytes(), before + lifetime_bytes);
    // A live feedback store may subsequently replace its private final bytes.
    // No hard link or premature finish/refund may alter the committed receipt.
    fs::write(&private_path, b"new-private-feedback-session").unwrap();
    assert_eq!(fs::read(&committed_path).unwrap(), committed);
    store::verify_file(&committed_path, &candidate.feedback, &warm.transaction).unwrap();
    assert!(warm.session.0.root.join("dirty").exists());
}

#[test]
fn automatic_reuse_full_artifact_bound_retires_only_the_stopped_private_monitor() {
    use super::super::selected_feedback::{Binding, FeedbackKind, Monitor};
    use ferrum_types::{SloAutomaticCalibrationSettingsV1, SloSelectedFeedbackStorageV1};
    let dir = Directory::new();
    let bound = CacheLimits {
        maximum_sources: 1,
        ..limits()
    };
    let cold = CacheSession::open_at(&dir.0, identity(1), 1, bound).unwrap();
    storage_commit(&cold.session, 100);
    drop(cold);
    let warm = CacheSession::open_at(&dir.0, identity(1), 2, bound).unwrap();
    let candidate = warm.candidate.as_ref().unwrap();
    let policy = SloAutomaticCalibrationSettingsV1::default().feedback;
    let binding = Binding {
        profile_sha256: [1; 32],
        source_sha256: [2; 32],
        fit_sha256: [3; 32],
        protocol_sha256: [4; 32],
        policy_sha256: [5; 32],
    };
    let private_bytes = Monitor::restart_resume_budget(&policy, &dir.0)
        .unwrap()
        .maximum_disk_bytes;
    let private = warm
        .session
        .reserve_external(private_bytes, &warm.transaction)
        .unwrap();
    let private_path = private.path.clone();
    let mut monitor = Monitor::open_bound(
        &policy,
        &SloSelectedFeedbackStorageV1::CreateNew {
            path: private_path.clone(),
        },
        binding.clone(),
        1,
        FeedbackKind::StructuredV2,
        Some(vec![[7; 32]].into()),
    )
    .unwrap();
    private.retain_resumed_feedback().unwrap();
    let journal = warm
        .session
        .adopt_source(
            candidate.generation,
            &candidate.sources[0],
            &warm.transaction,
        )
        .unwrap();
    let profile = warm
        .session
        .reserve_external(candidate.sources[0].profile.bytes, &warm.transaction)
        .unwrap();
    profile
        .copy_verified_from(
            &warm
                .session
                .artifact_path(candidate.generation, &candidate.sources[0].profile),
            &candidate.sources[0].profile,
            &warm.transaction,
        )
        .unwrap();
    let profile = profile.finish(&warm.transaction).unwrap();
    // This gate tests storage population, not declaration/model qualification.
    // The seed occupies the real optional receipt slot in the final manifest.
    let seed = warm
        .session
        .reserve_external(32, &warm.transaction)
        .unwrap();
    fs::write(&seed.path, b"bounded-input-seed-storage").unwrap();
    let seed = seed.finish(&warm.transaction).unwrap();
    assert!(matches!(
        warm.session.0.artifact(),
        Err(CacheMiss::Capacity)
    ));
    let before = warm.session.used_bytes();
    let mut foreign = Monitor::open_bound(
        &policy,
        &SloSelectedFeedbackStorageV1::MemoryOnly,
        binding,
        1,
        FeedbackKind::StructuredV2,
        Some(vec![[7; 32]].into()),
    )
    .unwrap();
    assert_eq!(
        warm.session
            .retire_resumed_feedback(&mut foreign, &warm.transaction),
        Err(CacheMiss::Corrupt)
    );
    assert!(private_path.exists());
    assert_eq!(warm.session.used_bytes(), before);
    let audit = serde_json::to_value(monitor.audit()).unwrap();
    warm.session
        .retire_resumed_feedback(&mut monitor, &warm.transaction)
        .unwrap();
    assert_eq!(serde_json::to_value(monitor.audit()).unwrap(), audit);
    assert!(!private_path.exists());
    assert_eq!(warm.session.used_bytes(), before - private_bytes);
    for receipt in candidate.artifacts() {
        store::verify_file(
            &warm.session.artifact_path(candidate.generation, receipt),
            receipt,
            &warm.transaction,
        )
        .unwrap();
    }
    let final_feedback = warm
        .session
        .reserve_external(64, &warm.transaction)
        .unwrap();
    assert_eq!(final_feedback.path, private_path);
    fs::write(&final_feedback.path, b"final-feedback-storage-fixture").unwrap();
    let final_feedback = final_feedback.finish(&warm.transaction).unwrap();
    // The closed original Store must neither persist again nor remove the
    // replacement receipt now occupying the reclaimed physical name.
    monitor.finish();
    store::verify_file(&private_path, &final_feedback, &warm.transaction).unwrap();
    let mut source = candidate.sources[0].clone();
    source.journal = journal.sealed().unwrap();
    source.profile = profile;
    warm.session
        .prepare_commit(
            vec![source],
            final_feedback,
            Some(seed),
            Some(100),
            2,
            &warm.transaction,
            || Ok(()),
        )
        .unwrap();
    warm.session
        .complete_prepared_commit(&warm.transaction, || Ok(()))
        .unwrap();
    assert!(!warm.session.0.root.join("dirty").exists());
}

#[test]
fn automatic_reuse_store_cross_boot_identity_and_original_expiry_are_misses() {
    for (kind, expected) in [
        (0, CacheMiss::CrossBoot),
        (1, CacheMiss::Identity),
        (2, CacheMiss::Expired),
    ] {
        let dir = Directory::new();
        let first = CacheSession::open_at(&dir.0, identity(1), 1, limits()).unwrap();
        storage_commit(&first.session, 100);
        drop(first);
        let mut incoming = identity(if kind == 0 { 2 } else { 1 });
        if kind == 1 {
            incoming.binding[0] ^= 1;
        }
        let next =
            CacheSession::open_at(&dir.0, incoming, if kind == 2 { 100 } else { 2 }, limits())
                .unwrap();
        assert_eq!(next.candidate.unwrap_err(), expected);
    }
}

#[test]
fn automatic_reuse_store_charges_old_commit_and_new_staging_before_append() {
    let dir = Directory::new();
    let first = CacheSession::open_at(&dir.0, identity(1), 1, limits()).unwrap();
    storage_commit(&first.session, 100);
    drop(first);
    let root = dir.0.join("cache-v1");
    let mut existing = 0;
    for e in fs::read_dir(&root).unwrap() {
        let e = e.unwrap();
        if e.file_type().unwrap().is_dir() {
            for f in fs::read_dir(e.path()).unwrap() {
                existing += f.unwrap().metadata().unwrap().len();
            }
        } else {
            existing += e.metadata().unwrap().len();
        }
    }
    let payload = b"new-original-source";
    let mut bound = limits();
    bound.maximum_bytes =
        existing + b"ferrum.automatic.reuse.active.v1\n".len() as u64 + payload.len() as u64;
    let second = CacheSession::open_at(&dir.0, identity(1), 2, bound).unwrap();
    assert!(second.candidate.is_ok());
    let mut source = second.session.original_source_raw().unwrap();
    let receipt = (payload.len() as u64, Sha256::digest(payload).into());
    source.append_original(payload, receipt);
    assert_eq!(second.session.used_bytes(), bound.maximum_bytes);
    assert_eq!(source.failure(), None);
    let mut extended = payload.to_vec();
    extended.push(b'!');
    source.append_original(
        b"!",
        (extended.len() as u64, Sha256::digest(&extended).into()),
    );
    assert_eq!(source.failure(), Some(CacheMiss::Capacity));
    assert_eq!(second.session.used_bytes(), bound.maximum_bytes);
    let artifact = second.session.0.generation_path.join(artifact_name(0));
    assert_eq!(fs::read(artifact).unwrap(), payload);
}

#[test]
fn automatic_reuse_store_deadline_and_population_limits_do_not_reset_per_source() {
    let mut tx = ColdTransaction::new(limits()).unwrap();
    tx.charge_population(12, 20).unwrap();
    assert_eq!(tx.remaining_population(), (8, 20));
    assert_eq!(tx.charge_population(9, 1), Err(CacheMiss::Capacity));
    tx.charge_population(8, 20).unwrap();
    assert_eq!(tx.remaining_population(), (0, 0));
    tx.deadline = Instant::now();
    assert_eq!(tx.charge_population(0, 0), Err(CacheMiss::Deadline));
    let dir = Directory::new();
    let opened = CacheSession::open_at(&dir.0, identity(1), 1, limits()).unwrap();
    let before = opened.session.used_bytes();
    assert!(matches!(
        opened.session.reserve_external(1, &tx),
        Err(CacheMiss::Deadline)
    ));
    assert_eq!(opened.session.used_bytes(), before);
    assert!(opened.session.0.root.join("dirty").is_file());
}

#[test]
fn automatic_reuse_store_corruption_unknown_files_and_observer_lease_are_strict() {
    let dir = Directory::new();
    let opened = CacheSession::open_at(&dir.0, identity(1), 1, limits()).unwrap();
    let source = opened.session.original_source_raw().unwrap();
    drop(opened);
    assert!(matches!(
        CacheSession::open_at(&dir.0, identity(1), 1, limits()),
        Err(CacheMiss::Busy)
    ));
    drop(source);
    let next = CacheSession::open_at(&dir.0, identity(1), 1, limits()).unwrap();
    assert!(matches!(next.candidate, Err(CacheMiss::Dirty)));
    let (_, sources) = storage_commit(&next.session, 100);
    let path = next
        .session
        .artifact_path(next.session.0.generation, &sources[0].profile);
    drop(next);
    fs::write(path, b"changed").unwrap();
    let corrupt = CacheSession::open_at(&dir.0, identity(1), 2, limits()).unwrap();
    assert!(matches!(corrupt.candidate, Err(CacheMiss::Corrupt)));
    drop(corrupt);
    let unrelated = dir.0.join("cache-v1").join("user-data");
    fs::write(&unrelated, b"must remain").unwrap();
    assert!(matches!(
        CacheSession::open_at(&dir.0, identity(1), 3, limits()),
        Err(CacheMiss::Corrupt)
    ));
    assert_eq!(fs::read(unrelated).unwrap(), b"must remain");
}

#[test]
fn automatic_reuse_store_independent_process_reads_only_clean_original_bytes() {
    let dir = Directory::new();
    let opened = CacheSession::open_at(&dir.0, identity(1), 1, limits()).unwrap();
    storage_commit(&opened.session, 100);
    drop(opened);
    let status = std::process::Command::new(std::env::current_exe().unwrap())
        .arg("automatic_reuse_store_process_child")
        .arg("--nocapture")
        .env("FERRUM_TEST_AUTOMATIC_REUSE_STORE", &dir.0)
        .status()
        .unwrap();
    assert!(status.success());
    // The child consumed the clean generation and exited without a clean-save
    // acknowledgement, so a second process is a miss, never the old commit.
    let next = CacheSession::open_at(&dir.0, identity(1), 3, limits()).unwrap();
    assert!(matches!(next.candidate, Err(CacheMiss::Dirty)));
}

#[test]
fn automatic_reuse_store_process_child() {
    let Some(path) = std::env::var_os("FERRUM_TEST_AUTOMATIC_REUSE_STORE") else {
        return;
    };
    let opened = CacheSession::open_at(Path::new(&path), identity(1), 2, limits()).unwrap();
    let manifest = opened.candidate.as_ref().unwrap();
    assert_eq!(manifest.expires_at_ns, Some(100));
    assert_eq!(manifest.identity, identity(1));
    let journal = opened
        .session
        .artifact_path(manifest.generation, &manifest.sources[0].journal);
    assert_eq!(
        fs::read(journal).unwrap(),
        b"original-header\noriginal-checkpoint\n"
    );
}

#[test]
fn automatic_reuse_worker_preparation_requires_successful_outer_shutdown_acknowledgement() {
    for outer_shutdown_succeeded in [false, true] {
        let dir = Directory::new();
        let first = CacheSession::open_at(&dir.0, identity(1), 1, limits()).unwrap();
        let (feedback, sources) = storage_payload(&first.session);
        let tx = first.session.transaction().unwrap();
        first
            .session
            .prepare_commit(sources, feedback, None, Some(100), 1, &tx, || Ok(()))
            .unwrap();
        // Complete source replay/feedback and a durable manifest are still
        // dirty until the engine has joined its worker and drained execution.
        assert!(first.session.0.root.join("manifest.json").is_file());
        assert!(first.session.0.root.join("dirty").is_file());
        assert!(first.session.original_source_raw().is_err());
        if outer_shutdown_succeeded {
            first
                .session
                .complete_prepared_commit(&tx, || Ok(()))
                .unwrap();
            assert!(!first.session.0.root.join("dirty").exists());
        } else {
            // A final stale clock/failed outer cleanup cannot authorize clean.
            assert_eq!(
                first
                    .session
                    .complete_prepared_commit(&tx, || Err(CacheMiss::Expired)),
                Err(CacheMiss::Expired)
            );
            assert!(first.session.0.root.join("dirty").is_file());
        }
        drop(first);
        let next = CacheSession::open_at(&dir.0, identity(1), 2, limits()).unwrap();
        if outer_shutdown_succeeded {
            assert!(next.candidate.is_ok());
        } else {
            assert_eq!(next.candidate.unwrap_err(), CacheMiss::Dirty);
        }
    }
}

#[cfg(unix)]
#[test]
fn automatic_reuse_last_observer_explicitly_unlocks_a_duplicated_lease() {
    let dir = Directory::new();
    let opened = CacheSession::open_at(&dir.0, identity(1), 1, limits()).unwrap();
    let inherited_description = opened.session.duplicate_lease_for_test();
    let observer = opened.session.original_source_raw().unwrap();
    drop(opened);
    // An actual original sink still owns the lease even after runtime Drop.
    assert!(matches!(
        CacheSession::open_at(&dir.0, identity(1), 2, limits()),
        Err(CacheMiss::Busy)
    ));
    drop(observer);
    // dup and fork reference the same flock. Simulate the pre-exec handle
    // remaining open, without a timing race or sleeps, and release the lease
    // exactly when its final authorized Rust observer leaves.
    let next = CacheSession::open_at(&dir.0, identity(1), 2, limits()).unwrap();
    assert!(matches!(next.candidate, Err(CacheMiss::Dirty)));
    drop(inherited_description);
    assert!(matches!(
        CacheSession::open_at(&dir.0, identity(1), 3, limits()),
        Err(CacheMiss::Busy)
    ));
}

#[test]
fn automatic_reuse_optional_algorithm_seed_is_bound_without_rewriting_legacy_manifest() {
    let dir = Directory::new();
    let opened = CacheSession::open_at(&dir.0, identity(1), 1, limits()).unwrap();
    let (feedback, sources) = storage_payload(&opened.session);
    let tx = opened.session.transaction().unwrap();
    let original = CacheManifest {
        schema_version: 1,
        generation: opened.session.0.generation,
        identity: identity(1),
        sources,
        feedback,
        startup_seed: None,
        expires_at_ns: None,
    };
    let legacy = serde_json::to_vec(&original).unwrap();
    assert!(!serde_json::from_slice::<serde_json::Value>(&legacy)
        .unwrap()
        .as_object()
        .unwrap()
        .contains_key("startup_seed"));
    let mut restored: CacheManifest = serde_json::from_slice(&legacy).unwrap();
    assert_eq!(serde_json::to_vec(&restored).unwrap(), legacy);
    restored.startup_seed = Some(feedback);
    assert_eq!(
        restored.validate(&identity(1), 1, limits()),
        Err(CacheMiss::Corrupt)
    );
    let seed = opened.session.reserve_external(32, &tx).unwrap();
    // Storage-only bytes, not a forged checked algorithm declaration.
    std::fs::write(&seed.path, b"untrusted declaration").unwrap();
    restored.startup_seed = Some(seed.finish(&tx).unwrap());
    restored.validate(&identity(1), 1, limits()).unwrap();
    assert_eq!(restored.artifacts().count(), 4);
    let seed = restored.startup_seed.unwrap();
    let path = opened.session.artifact_path(restored.generation, &seed);
    std::fs::write(&path, b"altered declaration").unwrap();
    assert_eq!(
        store::verify_file(&path, &seed, &tx),
        Err(CacheMiss::Corrupt)
    );
}

#[path = "tests/warm_peak.rs"]
mod warm_peak;

#[path = "tests/codec.rs"]
mod codec_tests;
