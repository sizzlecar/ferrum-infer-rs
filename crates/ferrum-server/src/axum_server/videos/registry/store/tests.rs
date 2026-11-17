use super::super::tests::{pending, Directory, STORED_BYTES};
use super::*;

fn open(directory: &Directory) -> (Store, Vec<Snapshot>) {
    Store::open(&directory.0, 1024, 8).unwrap()
}

#[test]
fn unfinished_records_and_owned_copy_residue_recover_as_interrupted() {
    let directory = Directory::new();
    let (store, _) = open(&directory);
    let id = Uuid::new_v4();
    store.write(&pending(id)).unwrap();
    fs::write(store.path(id, "mp4.part"), b"unfinished encoder transfer").unwrap();
    fs::write(store.path(id, "json.next"), b"interrupted update").unwrap();
    drop(store);
    let (store, records) = open(&directory);
    assert_eq!(records.len(), 1);
    assert_eq!(records[0].status, Status::Failed);
    assert!(records[0].error.as_ref().unwrap().contains("interrupted"));
    assert!(!store.path(id, "mp4.part").exists());
    assert!(!store.path(id, "json.next").exists());
}

#[test]
fn incomplete_initial_record_does_not_block_other_recovered_jobs() {
    let directory = Directory::new();
    let (store, _) = open(&directory);
    let accepted = pending(Uuid::new_v4());
    store.write(&accepted).unwrap();
    let interrupted = Uuid::new_v4();
    fs::write(store.path(interrupted, "json.next"), b"{\"version\":").unwrap();
    drop(store);
    let (store, records) = open(&directory);
    assert_eq!(records.len(), 1);
    assert_eq!(records[0].id, accepted.id);
    assert!(!store.path(interrupted, "json.next").exists());
}

#[test]
fn interrupted_initial_marker_is_resumed_only_without_committed_data() {
    let directory = Directory::new();
    fs::create_dir(&directory.0).unwrap();
    fs::write(directory.0.join(".owner"), b"ferrum-video-").unwrap();
    let (store, records) = open(&directory);
    assert!(records.is_empty());
    store.write(&pending(Uuid::new_v4())).unwrap();
    drop(store);
    fs::write(directory.0.join(".owner"), b"ferrum-video-").unwrap();
    assert!(Store::open(&directory.0, 1024, 8).is_err());
    assert_eq!(
        fs::read(directory.0.join(".owner")).unwrap(),
        b"ferrum-video-"
    );
}

#[test]
fn interrupted_completed_publication_and_delete_tombstone_are_reclaimed() {
    let directory = Directory::new();
    let (store, _) = open(&directory);
    let mut interrupted = pending(Uuid::new_v4());
    store.write(&interrupted).unwrap();
    // MP4 was fsynced/renamed, but the completed record was not published.
    drop(
        store
            .copy(interrupted.id, STORED_BYTES.len() as u64, STORED_BYTES)
            .unwrap(),
    );
    let mut deleted = pending(Uuid::new_v4());
    store.write(&deleted).unwrap();
    drop(
        store
            .copy(deleted.id, STORED_BYTES.len() as u64, STORED_BYTES)
            .unwrap(),
    );
    deleted.delete_requested = true;
    deleted.status = Status::Completed;
    deleted.encoded_bytes = Some(STORED_BYTES.len() as u64);
    store.write(&deleted).unwrap();
    drop(store);
    let (store, records) = open(&directory);
    assert_eq!(records.len(), 1);
    assert_eq!(records[0].id, interrupted.id);
    assert_eq!(records[0].status, Status::Failed);
    assert!(!store.path(interrupted.id, "mp4").exists());
    assert!(!store.path(deleted.id, "json").exists());
    assert!(!store.path(deleted.id, "mp4").exists());
    interrupted.status = Status::Cancelled;
    store.write(&interrupted).unwrap();
}

#[test]
fn published_file_sync_failure_cannot_refund_live_disk_bytes() {
    let directory = Directory::new();
    let (store, _) = open(&directory);
    let id = Uuid::new_v4();
    store.write(&pending(id)).unwrap();
    store.0.fail_after_publish.store(true, Ordering::Release);
    assert!(store
        .copy(id, STORED_BYTES.len() as u64, STORED_BYTES)
        .is_err());
    assert!(!store.path(id, "mp4").exists());
    assert!(!store.path(id, "mp4.part").exists());
    assert_eq!(store.bytes(), 0);
}

#[test]
fn readers_hold_quota_and_exclusive_directory_ownership_until_actual_drop() {
    let directory = Directory::new();
    let (store, _) = Store::open(&directory.0, STORED_BYTES.len() as u64, 4).unwrap();
    let mut snapshot = pending(Uuid::new_v4());
    store.write(&snapshot).unwrap();
    let artifact = store
        .copy(snapshot.id, STORED_BYTES.len() as u64, STORED_BYTES)
        .unwrap();
    let mut reader = artifact.open().unwrap();
    snapshot.status = Status::Completed;
    snapshot.encoded_bytes = Some(STORED_BYTES.len() as u64);
    snapshot.delete_requested = true;
    store.write(&snapshot).unwrap();
    artifact.delete();
    drop(artifact);
    assert!(!store.available(1));
    drop(store);
    assert!(Store::open(&directory.0, 1024, 4).is_err());
    let mut bytes = Vec::new();
    reader.read_to_end(&mut bytes).unwrap();
    assert_eq!(bytes, STORED_BYTES);
    drop(reader);
    assert!(open(&directory).1.is_empty());
}

#[test]
fn unknown_files_and_corrupt_committed_records_are_preserved() {
    let directory = Directory::new();
    let (store, _) = open(&directory);
    let id = Uuid::new_v4();
    store.write(&pending(id)).unwrap();
    let unknown = directory.0.join("keep-me.txt");
    fs::write(&unknown, b"user data").unwrap();
    drop(store);
    assert!(Store::open(&directory.0, 1024, 8).is_err());
    assert_eq!(fs::read(&unknown).unwrap(), b"user data");
    fs::remove_file(unknown).unwrap();
    fs::write(
        directory.0.join(format!("{id}.json")),
        b"corrupt committed record",
    )
    .unwrap();
    assert!(Store::open(&directory.0, 1024, 8).is_err());
    assert_eq!(
        fs::read(directory.0.join(format!("{id}.json"))).unwrap(),
        b"corrupt committed record"
    );
}

#[test]
fn recovery_validates_byte_receipt_and_enforces_configured_limits() {
    let directory = Directory::new();
    let (store, _) = open(&directory);
    let mut snapshot = pending(Uuid::new_v4());
    store.write(&snapshot).unwrap();
    drop(
        store
            .copy(snapshot.id, STORED_BYTES.len() as u64, STORED_BYTES)
            .unwrap(),
    );
    snapshot.status = Status::Completed;
    snapshot.encoded_bytes = Some(STORED_BYTES.len() as u64);
    store.write(&snapshot).unwrap();
    drop(store);
    assert!(Store::open(&directory.0, STORED_BYTES.len() as u64 - 1, 8).is_err());
    fs::write(
        directory.0.join(format!("{}.mp4", snapshot.id)),
        b"truncated",
    )
    .unwrap();
    assert!(Store::open(&directory.0, 1024, 8).is_err());
}

#[cfg(unix)]
#[test]
fn new_store_is_private_and_symlinks_are_not_deleted() {
    use std::os::unix::{fs::symlink, fs::PermissionsExt};
    let directory = Directory::new();
    let (store, _) = open(&directory);
    assert_eq!(
        fs::metadata(&directory.0).unwrap().permissions().mode() & 0o777,
        0o700
    );
    let id = Uuid::new_v4();
    store.write(&pending(id)).unwrap();
    let target = directory.0.join(".owner");
    let linked = store.path(id, "mp4.part");
    symlink(&target, &linked).unwrap();
    drop(store);
    assert!(Store::open(&directory.0, 1024, 8).is_err());
    assert!(fs::symlink_metadata(linked)
        .unwrap()
        .file_type()
        .is_symlink());
    assert_eq!(fs::read_to_string(target).unwrap(), OWNER);
}
