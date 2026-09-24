use super::*;
use std::sync::atomic::AtomicUsize;
struct OwnedRecord {
    drop_count: Arc<AtomicUsize>,
    entered: std::sync::mpsc::SyncSender<()>,
    release: Mutex<std::sync::mpsc::Receiver<()>>,
    fail: bool,
}
impl Drop for OwnedRecord {
    fn drop(&mut self) {
        self.drop_count.fetch_add(1, Ordering::SeqCst);
    }
}
impl Serialize for OwnedRecord {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        self.entered.send(()).unwrap();
        self.release.lock().unwrap().recv().unwrap();
        if self.fail {
            return Err(serde::ser::Error::custom("injected serialization failure"));
        }
        s.serialize_str("borrowed complete record")
    }
}
#[test]
fn owned_jsonl_record_holds_lifetime_through_write_and_drops_on_error() {
    for fail in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("owned.jsonl");
        let (entered_tx, entered_rx) = sync_channel(1);
        let (release_tx, release_rx) = sync_channel(1);
        let dropped = Arc::new(AtomicUsize::new(0));
        let record = OwnedRecord {
            drop_count: dropped.clone(),
            entered: entered_tx,
            release: Mutex::new(release_rx),
            fail,
        };
        let target = path.clone();
        let thread = std::thread::spawn(move || write_jsonl_owned_record(&target, record));
        entered_rx.recv_timeout(Duration::from_secs(3)).unwrap();
        assert_eq!(dropped.load(Ordering::SeqCst), 0);
        release_tx.send(()).unwrap();
        let result = thread.join().unwrap();
        assert_eq!(result.is_err(), fail);
        assert_eq!(dropped.load(Ordering::SeqCst), 1);
        if !fail {
            assert_eq!(
                std::fs::read_to_string(path).unwrap(),
                "\"borrowed complete record\"\n"
            );
        }
    }
}
