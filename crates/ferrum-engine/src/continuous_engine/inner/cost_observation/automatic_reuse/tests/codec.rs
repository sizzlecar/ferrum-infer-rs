//! Physical cache codec boundaries. Model authority is tested by the separate
//! original-source replay and rolling runtime fixtures.
use super::*;

#[test]
fn automatic_reuse_codec_reads_legacy_v1_identity_without_forging_raw_receipts() {
    let dir = Directory::new();
    let first = CacheSession::open_at(&dir.0, identity(1), 1, limits()).unwrap();
    storage_commit(&first.session, 100);
    let path = first.session.0.root.join("manifest.json");
    drop(first);
    let mut old: serde_json::Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
    old["schema_version"] = 1.into();
    for source in old["sources"].as_array_mut().unwrap() {
        source.as_object_mut().unwrap().remove("encoding");
    }
    fs::write(&path, serde_json::to_vec(&old).unwrap()).unwrap();
    let warm = CacheSession::open_at(&dir.0, identity(1), 2, limits()).unwrap();
    let manifest = warm.candidate.as_ref().unwrap();
    assert_eq!(manifest.schema_version, 1);
    for source in &manifest.sources {
        assert_eq!(source.encoding, JournalEncoding::IdentityV1);
        let path = warm
            .session
            .artifact_path(manifest.generation, &source.journal);
        let raw = codec::decode(
            &warm.session,
            &path,
            source.journal,
            source.encoding,
            limits().maximum_source_bytes as usize,
            0,
            &warm.transaction,
        )
        .unwrap();
        assert_eq!(raw.len() as u64, source.journal.bytes);
        assert_eq!(
            <[u8; 32]>::from(Sha256::digest(&raw)),
            source.journal.sha256
        );
    }
}

fn raw() -> Vec<u8> {
    b"{\"record\":\"original input, never reserialized\"}\n".repeat(512)
}
fn append(journal: &mut OriginalSourceJournal, bytes: &[u8]) -> (u64, [u8; 32]) {
    let mut hash = Sha256::new();
    let mut count = 0u64;
    for chunk in bytes.chunks(193) {
        hash.update(chunk);
        count += chunk.len() as u64;
        journal.append_original(chunk, (count, hash.clone().finalize().into()));
    }
    (count, hash.finalize().into())
}
fn encode(
    session: &CacheSession,
    bytes: &[u8],
) -> (OriginalSourceJournal, ArtifactReceipt, JournalEncoding) {
    let mut journal = session.original_source().unwrap();
    assert_eq!(
        session.0.ledger.lock().codec_retained,
        codec::ENCODER_RETAINED
    );
    let raw = append(&mut journal, bytes);
    journal.checkpoint(raw).unwrap();
    let stored = journal.seal(raw).unwrap();
    let encoding = journal.encoding().unwrap();
    assert_eq!(session.0.ledger.lock().codec_retained, 0);
    (journal, stored, encoding)
}

#[test]
fn automatic_reuse_codec_charges_only_emitted_bytes_and_preserves_full_original() {
    let dir = Directory::new();
    let opened = CacheSession::open_at(&dir.0, identity(1), 1, limits()).unwrap();
    let session = &opened.session;
    let before = session.used_bytes();
    let bytes = raw();
    let (_, stored, encoding) = encode(session, &bytes);
    assert!(stored.bytes < bytes.len() as u64);
    assert_eq!(session.used_bytes() - before, stored.bytes);
    assert_eq!(
        encoding.original(stored),
        (bytes.len() as u64, Sha256::digest(&bytes).into())
    );
    let path = session.artifact_path(session.0.generation, &stored);
    let encoded = fs::read(&path).unwrap();
    assert_eq!(stored.bytes, encoded.len() as u64);
    assert_eq!(stored.sha256, <[u8; 32]>::from(Sha256::digest(&encoded)));
    let tx = session.transaction().unwrap();
    assert_eq!(
        codec::decode(session, &path, stored, encoding, bytes.len(), 0, &tx).unwrap(),
        bytes
    );
    assert_eq!(
        codec::decode(session, &path, stored, encoding, bytes.len() - 1, 0, &tx),
        Err(CacheMiss::Capacity)
    );
    let mut expired = session.transaction().unwrap();
    expired.deadline = Instant::now();
    assert_eq!(
        codec::decode(session, &path, stored, encoding, bytes.len(), 0, &expired),
        Err(CacheMiss::Deadline)
    );
}

#[test]
fn automatic_reuse_codec_footer_capacity_failure_never_writes_unpaid_drop_bytes() {
    let bytes = raw();
    let first = Directory::new();
    let full = CacheSession::open_at(&first.0, identity(1), 1, limits()).unwrap();
    let base = full.session.used_bytes();
    let (_, receipt, _) = encode(&full.session, &bytes);
    let second = Directory::new();
    let mut bound = limits();
    bound.maximum_bytes = base + receipt.bytes - 1;
    let opened = CacheSession::open_at(&second.0, identity(1), 1, bound).unwrap();
    let mut journal = opened.session.original_source().unwrap();
    let original = append(&mut journal, &bytes);
    assert_eq!(journal.failure(), None);
    journal.checkpoint(original).unwrap();
    assert_eq!(journal.seal(original), Err(CacheMiss::Capacity));
    assert_eq!(journal.failure(), Some(CacheMiss::Capacity));
    assert_eq!(opened.session.0.ledger.lock().codec_retained, 0);
    assert_eq!(opened.session.0.ledger.lock().active_writers, 0);
    let path = opened.session.0.generation_path.join(artifact_name(0));
    let before_drop = fs::read(&path).unwrap();
    assert!(before_drop.len() < receipt.bytes as usize);
    assert!(opened.session.used_bytes() <= bound.maximum_bytes);
    drop(journal);
    assert_eq!(fs::read(path).unwrap(), before_drop);
    assert!(opened.session.0.root.join("dirty").is_file());
}

#[test]
fn automatic_reuse_codec_rejects_changed_crc_truncation_trailing_member_and_raw_receipt() {
    let dir = Directory::new();
    let opened = CacheSession::open_at(&dir.0, identity(1), 1, limits()).unwrap();
    let session = &opened.session;
    let bytes = raw();
    let (_, stored, encoding) = encode(session, &bytes);
    let path = session.artifact_path(session.0.generation, &stored);
    let encoded = fs::read(&path).unwrap();
    let tx = session.transaction().unwrap();
    for case in 0..6 {
        let mut changed = encoded.clone();
        let mut claimed = encoding;
        match case {
            0 => {
                let n = changed.len();
                changed[n - 8] ^= 1;
            } // CRC
            1 => {
                changed.pop();
            } // truncated trailer
            2 => changed.push(0),
            3 => changed.extend_from_slice(&encoded), // second gzip member
            4 => changed[3] = 8,                      // optional name would otherwise allocate
            5 => {
                claimed = JournalEncoding::GzipV1 {
                    original_bytes: bytes.len() as u64,
                    original_sha256: [7; 32],
                }
            }
            _ => unreachable!(),
        }
        fs::write(&path, &changed).unwrap();
        // Recompute the physical receipt to exercise the independent gzip/raw
        // validation, not merely the preliminary stored-SHA check.
        let modified = ArtifactReceipt {
            index: stored.index,
            bytes: changed.len() as u64,
            sha256: Sha256::digest(&changed).into(),
        };
        assert_eq!(
            codec::decode(session, &path, modified, claimed, bytes.len(), 0, &tx),
            Err(CacheMiss::Corrupt),
            "case {case}"
        );
    }
}

#[test]
fn automatic_reuse_codec_memory_reservation_and_dirty_adoption_are_owned_once() {
    let dir = Directory::new();
    let mut bound = limits();
    bound.maximum_sources = 5;
    bound.maximum_retained_bytes = 2 * codec::ENCODER_RETAINED;
    let opened = CacheSession::open_at(&dir.0, identity(1), 1, bound).unwrap();
    let a = opened.session.original_source().unwrap();
    let b = opened.session.original_source().unwrap();
    assert!(matches!(
        opened.session.original_source(),
        Err(CacheMiss::Capacity)
    ));
    assert_eq!(
        opened.session.0.ledger.lock().codec_retained,
        bound.maximum_retained_bytes
    );
    a.abandon();
    assert_eq!(
        opened.session.0.ledger.lock().codec_retained,
        codec::ENCODER_RETAINED
    );
    b.abandon();
    assert_eq!(opened.session.0.ledger.lock().codec_retained, 0);
    let bytes = raw();
    let (journal, stored, encoding) = encode(&opened.session, &bytes);
    let tx = opened.session.transaction().unwrap();
    let profile = opened.session.reserve_external(32, &tx).unwrap();
    fs::write(&profile.path, b"profile-storage-fixture").unwrap();
    let profile = profile.finish(&tx).unwrap();
    let feedback = opened.session.reserve_external(32, &tx).unwrap();
    fs::write(&feedback.path, b"feedback-storage-fixture").unwrap();
    let feedback = feedback.finish(&tx).unwrap();
    let source = CachedSource {
        kind: SourceKind::OwnerBlocksV7,
        journal: stored,
        encoding,
        checkpoint_bytes: bytes.len() as u64,
        checkpoint_sha256: Sha256::digest(&bytes).into(),
        profile,
        domains: vec![],
    };
    opened
        .session
        .commit(vec![source.clone()], feedback, Some(100), 0, &tx)
        .unwrap();
    let generation = opened.session.0.generation;
    let old_path = opened.session.artifact_path(generation, &stored);
    drop((a, b, journal, opened));
    let warm = CacheSession::open_at(&dir.0, identity(1), 2, bound).unwrap();
    assert_eq!(warm.candidate.as_ref().unwrap().schema_version, 2);
    let used = warm.session.used_bytes();
    let adopted = warm
        .session
        .adopt_source(generation, &source, &warm.transaction)
        .unwrap();
    assert_eq!(
        warm.session.used_bytes(),
        used + stored.bytes,
        "committed and private journal bytes coexist inside the same quota"
    );
    store::verify_file(&old_path, &stored, &warm.transaction).unwrap();
    let transferred = adopted.sealed().unwrap();
    assert_eq!(
        (transferred.bytes, transferred.sha256),
        (stored.bytes, stored.sha256)
    );
    assert_eq!(adopted.encoding().unwrap(), encoding);
    let path = warm
        .session
        .artifact_path(warm.session.0.generation, &transferred);
    assert_eq!(
        codec::decode(
            &warm.session,
            &path,
            transferred,
            encoding,
            bytes.len(),
            0,
            &warm.transaction
        )
        .unwrap(),
        bytes
    );
    assert!(warm.session.0.root.join("dirty").is_file());
    // Expiry/pruning can release only the private copy. The original manifest
    // and its named bytes remain consistent even without a replacement commit.
    assert!(adopted.discard_if_unobserved().unwrap());
    assert!(!path.exists());
    assert_eq!(warm.session.used_bytes(), used);
    store::verify_file(&old_path, &stored, &warm.transaction).unwrap();
    let manifest_path = warm.session.0.root.join("manifest.json");
    let committed_manifest = fs::read(&manifest_path).unwrap();
    drop((adopted, warm));
    assert_eq!(fs::read(&manifest_path).unwrap(), committed_manifest);
    assert_eq!(fs::read(&old_path).unwrap().len() as u64, stored.bytes);
    let interrupted = CacheSession::open_at(&dir.0, identity(1), 3, bound).unwrap();
    assert!(matches!(interrupted.candidate, Err(CacheMiss::Dirty)));
}
