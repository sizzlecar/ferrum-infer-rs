use super::*;
use std::num::{NonZeroU64, NonZeroUsize};

struct Fixture {
    directory: PathBuf,
    policy: SloAutomaticCalibrationDiagnosticsV1,
}
impl Fixture {
    fn new(total: u64, retained: usize) -> Self {
        let directory =
            std::env::temp_dir().join(format!("ferrum-automatic-quota-{}", uuid::Uuid::new_v4()));
        let policy = SloAutomaticCalibrationDiagnosticsV1::Directory {
            directory: directory.clone(),
            maximum_source_bytes: NonZeroU64::new(1024).unwrap(),
            maximum_total_bytes: NonZeroU64::new(total).unwrap(),
            maximum_retained_generations: NonZeroUsize::new(retained).unwrap(),
        };
        Self { directory, policy }
    }
    fn store(&self) -> Store {
        Store::new(&self.policy).unwrap()
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.directory);
    }
}

#[test]
fn busy_global_quota_returns_without_waiting_for_the_other_worker() {
    let f = Fixture::new(4096, 2);
    let store = f.store();
    let held = store.locked().unwrap();
    assert!(store.reserve(Kind::Service, 100).is_err());
    drop(held);
    assert!(store.reserve(Kind::Service, 100).is_ok());
}

#[test]
fn complete_unknown_manifest_is_not_treated_as_a_crash_fragment() {
    let f = Fixture::new(4096, 1);
    let store = f.store();
    let first = store.reserve(Kind::Service, 100).unwrap();
    let path = first.directory.clone();
    drop(first);
    fs::write(path.join("reservation.json"), b"{\"operator_data\":true}").unwrap();
    assert!(store.reserve(Kind::Service, 100).is_err());
    assert!(path.exists());
}

#[test]
fn first_owner_marker_crash_recovers_only_its_known_prefix_in_empty_root() {
    let f = Fixture::new(4096, 1);
    let store = f.store();
    fs::create_dir_all(&store.root).unwrap();
    fs::write(store.root.join("owner"), &MARKER[..8]).unwrap();
    assert!(store.reserve(Kind::Service, 100).is_ok());
    assert_eq!(fs::read(store.root.join("owner")).unwrap(), MARKER);
}

#[test]
fn active_leases_are_not_deleted_and_completed_entries_rotate() {
    let f = Fixture::new(4096, 1);
    let store = f.store();
    let active = store.reserve(Kind::Service, 512).unwrap();
    let path = active.directory.clone();
    assert!(store.reserve(Kind::Reference, 512).is_err());
    assert!(path.exists());
    drop(active);
    let next = store.reserve(Kind::Reference, 512).unwrap();
    assert!(!path.exists());
    assert!(next.directory.exists());
}

#[test]
fn restart_counts_partial_files_and_recovers_incomplete_reservations() {
    for partial in [None, Some(b"{\"version\":".as_slice())] {
        let f = Fixture::new(4096, 1);
        let store = f.store();
        let root_lock = store.locked().unwrap();
        let orphan = store.root.join(format!("entry-{}", uuid::Uuid::new_v4()));
        fs::create_dir(&orphan).unwrap();
        if let Some(bytes) = partial {
            fs::write(orphan.join("reservation.json"), bytes).unwrap();
        }
        let temporary = orphan.join(format!(".ferrum-cost-{}.tmp", uuid::Uuid::new_v4()));
        fs::write(temporary, vec![1u8; 700]).unwrap();
        drop(root_lock);
        let restarted = f.store();
        let lock = restarted.locked().unwrap();
        let retained = restarted.scan().unwrap();
        assert_eq!(retained.len(), 1);
        assert!(retained[0].bytes >= 700);
        drop(retained);
        drop(lock);
        let next = restarted.reserve(Kind::Service, 512).unwrap();
        assert!(!orphan.exists());
        assert!(next.directory.exists());
    }
}

#[test]
fn unknown_files_fail_closed_without_deleting_any_existing_entry() {
    let f = Fixture::new(4096, 1);
    let store = f.store();
    let active = store.reserve(Kind::Service, 512).unwrap();
    let path = active.directory.clone();
    fs::write(path.join("operator-data"), b"must survive").unwrap();
    drop(active);
    assert!(store.reserve(Kind::Reference, 512).is_err());
    assert_eq!(
        fs::read(path.join("operator-data")).unwrap(),
        b"must survive"
    );
    let outside = f.directory.join("unrelated");
    fs::write(&outside, b"outside scope").unwrap();
    assert_eq!(fs::read(outside).unwrap(), b"outside scope");
}

#[test]
fn reference_and_service_reservations_share_one_byte_and_retention_quota() {
    let f = Fixture::new(2048, 2);
    let store = f.store();
    store.publish_reference(&[1; 300], &[2; 100]).unwrap();
    let service = store.reserve(Kind::Service, 400).unwrap();
    let blocked = store.reserve(Kind::Service, 800);
    assert!(
        blocked.is_err(),
        "live active reservation alone leaves insufficient bytes"
    );
    assert!(service.directory.exists());
    drop(service);
    let next = f.store().reserve(Kind::Service, 600).unwrap();
    assert!(next.directory.exists());
}

#[test]
fn too_large_pair_does_not_create_a_partial_entry() {
    let f = Fixture::new(1024, 2);
    let store = f.store();
    assert!(store.publish_reference(&[0; 513], &[0; 32]).is_err());
    assert!(!store.root.exists());
}

#[test]
fn preexisting_unowned_directory_is_not_claimed_or_cleaned() {
    let f = Fixture::new(4096, 2);
    let store = f.store();
    fs::create_dir_all(&store.root).unwrap();
    fs::write(store.root.join("existing-user-file"), b"keep").unwrap();
    assert!(store.reserve(Kind::Service, 100).is_err());
    assert_eq!(
        fs::read(store.root.join("existing-user-file")).unwrap(),
        b"keep"
    );
    assert!(!store.root.join("owner").exists());
}
