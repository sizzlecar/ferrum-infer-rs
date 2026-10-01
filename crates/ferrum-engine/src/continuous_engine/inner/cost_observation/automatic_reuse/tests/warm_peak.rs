//! A real store/journal ledger regression, not a qualified-model fixture.
//! Five live writers can fail at a warm peak even though rotation later
//! leaves a small final directory. No inference or replay authority is minted.
use super::*;

fn append(journal: &mut OriginalSourceJournal, content: &mut Vec<u8>, bytes: usize) {
    let next = vec![b'x'; bytes];
    content.extend_from_slice(&next);
    journal.append_original(
        &next,
        (content.len() as u64, Sha256::digest(&*content).into()),
    );
}

fn fixture_limits() -> CacheLimits {
    CacheLimits {
        maximum_bytes: 128 * 1024,
        maximum_source_bytes: 64 * 1024,
        maximum_sources: 6,
        ..limits()
    }
}

fn commit_original(session: &CacheSession) -> ArtifactReceipt {
    let mut original = session.original_source_raw().unwrap();
    let mut bytes = Vec::new();
    append(&mut original, &mut bytes, 40 * 1024);
    let receipt = (bytes.len() as u64, Sha256::digest(&bytes).into());
    original.checkpoint(receipt).unwrap();
    let journal = original.seal(receipt).unwrap();
    drop(original);
    let tx = session.transaction().unwrap();
    let profile = session.reserve_external(64, &tx).unwrap();
    fs::write(&profile.path, b"original-profile-fixture").unwrap();
    let profile = profile.finish(&tx).unwrap();
    let feedback = session.reserve_external(64, &tx).unwrap();
    fs::write(&feedback.path, b"original-feedback-fixture").unwrap();
    let feedback = feedback.finish(&tx).unwrap();
    session
        .commit(
            vec![CachedSource {
                kind: SourceKind::OwnerBlocksV7,
                encoding: JournalEncoding::IdentityV1,
                journal,
                checkpoint_bytes: journal.bytes,
                checkpoint_sha256: journal.sha256,
                profile,
                domains: Vec::new(),
            }],
            feedback,
            Some(100),
            1,
            &tx,
        )
        .unwrap();
    journal
}

#[test]
fn automatic_reuse_warm_original_copy_and_old_generation_charge_until_real_release() {
    let dir = Directory::new();
    let cold = CacheSession::open_at(&dir.0, identity(1), 1, fixture_limits()).unwrap();
    let original = commit_original(&cold.session);
    drop(cold);
    let warm = CacheSession::open_at(&dir.0, identity(1), 2, fixture_limits()).unwrap();
    let manifest = warm.candidate.as_ref().unwrap();
    let old_path = warm.session.artifact_path(manifest.generation, &original);
    let before = warm.session.0.ledger.lock().used;
    let mut copy = warm.session.original_source_raw().unwrap();
    let original_bytes = fs::read(&old_path).unwrap();
    copy.append_original(&original_bytes, (original.bytes, original.sha256));
    copy.checkpoint((original.bytes, original.sha256)).unwrap();
    copy.seal((original.bytes, original.sha256)).unwrap();
    assert_eq!(warm.session.0.ledger.lock().used, before + original.bytes);
    let catalog_custody = copy.clone();
    assert!(!copy.discard_if_unobserved().unwrap());
    drop(catalog_custody);
    assert!(copy.discard_if_unobserved().unwrap());
    assert_eq!(warm.session.0.ledger.lock().used, before);
    assert_eq!(fs::metadata(old_path).unwrap().len(), original.bytes);
    assert!(warm.session.0.root.join("dirty").exists());
}

#[test]
fn automatic_reuse_warm_peak_capacity_is_sticky_after_rotation_releases_other_sources() {
    for warm_start in [false, true] {
        let dir = Directory::new();
        if warm_start {
            let initial = CacheSession::open_at(&dir.0, identity(1), 1, fixture_limits()).unwrap();
            commit_original(&initial.session);
        }
        let opened = CacheSession::open_at(&dir.0, identity(1), 2, fixture_limits()).unwrap();
        assert_eq!(opened.candidate.is_ok(), warm_start);
        let baseline = opened.session.0.ledger.lock().used;
        let mut others = Vec::new();
        for _ in 0..4 {
            let mut journal = opened.session.original_source_raw().unwrap();
            let mut bytes = Vec::new();
            append(&mut journal, &mut bytes, 16 * 1024);
            assert_eq!(journal.failure(), None);
            others.push(journal);
        }
        let mut selected = opened.session.original_source_raw().unwrap();
        let mut bytes = Vec::new();
        append(&mut selected, &mut bytes, 16 * 1024);
        let checkpoint = (bytes.len() as u64, Sha256::digest(&bytes).into());
        selected.checkpoint(checkpoint).unwrap();
        let before = opened.session.0.ledger.lock().used;
        assert_eq!(before, baseline + 80 * 1024);
        assert_eq!(
            before + 16 * 1024 > fixture_limits().maximum_bytes,
            warm_start
        );
        append(&mut selected, &mut bytes, 16 * 1024);
        if !warm_start {
            assert_eq!(selected.failure(), None);
            selected
                .seal((bytes.len() as u64, Sha256::digest(&bytes).into()))
                .unwrap();
            continue;
        }
        assert_eq!(selected.failure(), Some(CacheMiss::Capacity));
        assert_eq!(opened.session.0.ledger.lock().used, before);
        for journal in others {
            journal.abandon();
            assert!(journal.discard_if_unobserved().unwrap());
        }
        assert_eq!(opened.session.0.ledger.lock().used, baseline + 16 * 1024);
        assert!(opened.session.0.ledger.lock().used + 16 * 1024 < fixture_limits().maximum_bytes);
        selected.finish((bytes.len() as u64, Sha256::digest(&bytes).into()));
        assert_eq!(selected.sealed(), Err(CacheMiss::Capacity));
        assert!(opened.session.0.root.join("dirty").exists());
    }
}
