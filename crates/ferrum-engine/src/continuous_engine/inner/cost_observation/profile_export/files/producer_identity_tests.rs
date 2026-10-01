use super::*;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc, Barrier,
};

struct ExecutableFixture(PathBuf);
impl ExecutableFixture {
    fn new() -> Self {
        Self(
            std::env::temp_dir().join(format!("ferrum-producer-identity-{}", uuid::Uuid::new_v4())),
        )
    }
}
impl Drop for ExecutableFixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

#[test]
fn process_producer_identity_shared_collectors_hash_one_real_origin() {
    let fixture = ExecutableFixture::new();
    std::fs::write(&fixture.0, b"original executable bytes").unwrap();
    let store = Arc::new(ProducerIdentityStore::new());
    let reads = Arc::new(AtomicUsize::new(0));
    let start = Arc::new(Barrier::new(3));
    let workers: Vec<_> = (0..2)
        .map(|_| {
            let (store, reads, start, path) = (
                store.clone(),
                reads.clone(),
                start.clone(),
                fixture.0.clone(),
            );
            std::thread::spawn(move || {
                start.wait();
                store
                    .current(|| {
                        reads.fetch_add(1, Ordering::SeqCst);
                        ProducerIdentity::read_uncached(&path)
                    })
                    .unwrap()
            })
        })
        .collect();
    start.wait();
    let identities: Vec<_> = workers
        .into_iter()
        .map(|worker| worker.join().unwrap())
        .collect();
    assert_eq!(reads.load(Ordering::SeqCst), 1);
    assert_eq!(
        identities[0].executable_sha256,
        identities[1].executable_sha256
    );
    assert_eq!(
        identities[0].executable_path,
        fixture.0.canonicalize().unwrap()
    );
    assert_eq!(
        identities[0].executable_bytes,
        b"original executable bytes".len() as u64
    );
    assert_eq!(
        identities[0].executable_sha256,
        format!("{:x}", Sha256::digest(b"original executable bytes"))
    );

    std::fs::write(&fixture.0, b"replacement executable has different bytes").unwrap();
    let retained = store
        .current(|| {
            reads.fetch_add(1, Ordering::SeqCst);
            ProducerIdentity::read_uncached(&fixture.0)
        })
        .unwrap();
    assert_eq!(retained.executable_sha256, identities[0].executable_sha256);
    assert_eq!(reads.load(Ordering::SeqCst), 1);
    let explicit = ProducerIdentity::read(&fixture.0).unwrap();
    assert_ne!(explicit.executable_sha256, retained.executable_sha256);
    assert_eq!(
        explicit.executable_bytes,
        b"replacement executable has different bytes".len() as u64
    );
    // A new process has a fresh static store, not the previous origin.
    let fresh_process = ProducerIdentityStore::new()
        .current(|| ProducerIdentity::read_uncached(&fixture.0))
        .unwrap();
    assert_eq!(fresh_process.executable_sha256, explicit.executable_sha256);
}

#[test]
fn process_producer_identity_failed_origin_is_sticky_but_explicit_read_is_fresh() {
    let fixture = ExecutableFixture::new();
    let store = ProducerIdentityStore::new();
    let reads = AtomicUsize::new(0);
    let current = || {
        store.current(|| {
            reads.fetch_add(1, Ordering::SeqCst);
            ProducerIdentity::read_uncached(&fixture.0)
        })
    };
    let first = current().unwrap_err();
    assert!(
        matches!(&first, ExportError::Io(error) if error.kind() == std::io::ErrorKind::NotFound)
    );
    std::fs::write(&fixture.0, b"now accessible executable").unwrap();
    let second = current().unwrap_err();
    assert_eq!(first.to_string(), second.to_string());
    assert!(
        matches!(second, ExportError::Io(error) if error.kind() == std::io::ErrorKind::NotFound)
    );
    assert_eq!(reads.load(Ordering::SeqCst), 1);
    assert!(ProducerIdentity::read(&fixture.0).is_ok());
}

#[test]
fn process_producer_identity_source_rejection_preserves_original_category() {
    let fixture = ExecutableFixture::new();
    std::fs::create_dir(&fixture.0).unwrap();
    let store = ProducerIdentityStore::new();
    let result = store.current(|| ProducerIdentity::read_uncached(&fixture.0));
    std::fs::remove_dir(&fixture.0).unwrap();
    assert!(matches!(
        result,
        Err(ExportError::Source(
            "producer executable exceeds the regular-file bound"
        ))
    ));
    std::fs::write(&fixture.0, b"regular file now").unwrap();
    assert!(matches!(
        store.current(|| ProducerIdentity::read_uncached(&fixture.0)),
        Err(ExportError::Source(_))
    ));
    assert!(ProducerIdentity::read(&fixture.0).is_ok());
}
