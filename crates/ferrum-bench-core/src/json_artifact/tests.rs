use super::*;
use serde::ser::SerializeMap;
use std::sync::{
    atomic::AtomicUsize,
    mpsc::{sync_channel, Receiver, SyncSender},
    Arc, Mutex,
};
use std::time::Duration;

struct OwnedRecord {
    dropped: Arc<AtomicUsize>,
    entered: SyncSender<()>,
    release: Mutex<Receiver<()>>,
    fail: bool,
}

impl Drop for OwnedRecord {
    fn drop(&mut self) {
        self.dropped.fetch_add(1, Ordering::SeqCst);
    }
}

impl Serialize for OwnedRecord {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut map = serializer.serialize_map(Some(2))?;
        map.serialize_entry("token_count", &3)?;
        self.entered.send(()).unwrap();
        self.release.lock().unwrap().recv().unwrap();
        if self.fail {
            return Err(serde::ser::Error::custom(
                "failure after partial JSON output",
            ));
        }
        map.serialize_entry("token_ids", &[11, 12, 13])?;
        map.end()
    }
}

#[test]
fn owned_json_artifact_replaces_placeholder_and_holds_lease_through_write() {
    for fail in [false, true] {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("prompt_token_ids.json");
        let placeholder = b"{\"token_ids\":null,\"unavailable_reason\":\"awaiting completion\"}\n";
        std::fs::write(&path, placeholder).unwrap();
        let dropped = Arc::new(AtomicUsize::new(0));
        let (entered_tx, entered_rx) = sync_channel(1);
        let (release_tx, release_rx) = sync_channel(1);
        let record = OwnedRecord {
            dropped: dropped.clone(),
            entered: entered_tx,
            release: Mutex::new(release_rx),
            fail,
        };
        let target = path.clone();
        let writer = std::thread::spawn(move || write_json_owned_record(&target, record));
        entered_rx.recv_timeout(Duration::from_secs(3)).unwrap();
        assert_eq!(dropped.load(Ordering::SeqCst), 0);
        assert_eq!(std::fs::read(&path).unwrap(), placeholder);
        release_tx.send(()).unwrap();
        assert_eq!(writer.join().unwrap().is_err(), fail);
        assert_eq!(dropped.load(Ordering::SeqCst), 1);
        if fail {
            assert_eq!(std::fs::read(&path).unwrap(), placeholder);
        } else {
            let value: serde_json::Value =
                serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
            assert_eq!(
                value,
                serde_json::json!({ "token_count": 3, "token_ids": [11, 12, 13] })
            );
        }
        // Both publication and serialization failure clean the temporary file.
        assert_eq!(std::fs::read_dir(directory.path()).unwrap().count(), 1);
    }
}

#[test]
fn owned_json_artifact_publication_failure_releases_lease_and_preserves_target() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("prompt_token_ids.json");
    std::fs::create_dir(&path).unwrap();
    std::fs::write(path.join("sentinel"), b"keep").unwrap();
    let dropped = Arc::new(AtomicUsize::new(0));
    let (entered_tx, entered_rx) = sync_channel(1);
    let (release_tx, release_rx) = sync_channel(1);
    let record = OwnedRecord {
        dropped: dropped.clone(),
        entered: entered_tx,
        release: Mutex::new(release_rx),
        fail: false,
    };
    release_tx.send(()).unwrap();
    assert!(write_json_owned_record(&path, record).is_err());
    entered_rx.recv_timeout(Duration::from_secs(3)).unwrap();
    assert_eq!(dropped.load(Ordering::SeqCst), 1);
    assert_eq!(std::fs::read(path.join("sentinel")).unwrap(), b"keep");
    assert_eq!(std::fs::read_dir(directory.path()).unwrap().count(), 1);
}

#[test]
fn owned_json_artifact_new_and_repeated_writes_remain_one_value() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("prompt_token_ids.json");
    for tokens in [vec![1, 2, 3], vec![4]] {
        write_json_owned_record(&path, &tokens).unwrap();
        let actual: Vec<u32> = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        assert_eq!(actual, tokens);
    }
    assert_eq!(std::fs::read_dir(directory.path()).unwrap().count(), 1);
}
