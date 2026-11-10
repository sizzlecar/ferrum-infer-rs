use super::*;
use std::sync::atomic::AtomicUsize;
use std::task::Wake;

struct Signal(mpsc::SyncSender<()>);
impl Wake for Signal {
    fn wake(self: Arc<Self>) {
        let _ = self.0.try_send(());
    }
    fn wake_by_ref(self: &Arc<Self>) {
        let _ = self.0.try_send(());
    }
}

#[test]
fn downstream_preparation_bounds_work_and_wakes_after_result_publication() {
    let busy = Arc::new(AtomicBool::new(false));
    let (signal, awakened) = mpsc::sync_channel(1);
    let wake = Waker::from(Arc::new(Signal(signal)));
    let (release, blocked) = mpsc::sync_channel(0);
    let permit = PreparationPermit::try_acquire(&busy, &wake).unwrap();
    let mut task = permit
        .spawn(move || {
            blocked.recv().unwrap();
            Ok(42)
        })
        .unwrap();
    assert!(matches!(task.poll(), Poll::Pending));
    assert!(PreparationPermit::try_acquire(&busy, &wake).is_none());
    release.send(()).unwrap();
    awakened
        .recv_timeout(std::time::Duration::from_secs(5))
        .unwrap();
    assert!(matches!(task.poll(), Poll::Ready(Ok(42))));
    assert!(PreparationPermit::try_acquire(&busy, &wake).is_some());
}

struct Owner(Arc<AtomicUsize>);
impl Drop for Owner {
    fn drop(&mut self) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}

#[test]
fn downstream_preparation_cancellation_retires_work_and_result_owners() {
    let busy = Arc::new(AtomicBool::new(false));
    let drops = Arc::new(AtomicUsize::new(0));
    let owner = Owner(drops.clone());
    let (release, blocked) = mpsc::sync_channel(0);
    let task = PreparationPermit::try_acquire(&busy, Waker::noop())
        .unwrap()
        .spawn(move || {
            blocked.recv().unwrap();
            Ok(owner)
        })
        .unwrap();
    let cancelled = std::thread::spawn(move || drop(task));
    release.send(()).unwrap();
    cancelled.join().unwrap();
    assert_eq!(drops.load(Ordering::SeqCst), 1);
    assert!(!busy.load(Ordering::Acquire));
}

#[test]
fn downstream_preparation_panic_returns_error_and_releases_the_slot() {
    let busy = Arc::new(AtomicBool::new(false));
    let (signal, awakened) = mpsc::sync_channel(1);
    let wake = Waker::from(Arc::new(Signal(signal)));
    let mut task = PreparationPermit::try_acquire(&busy, &wake)
        .unwrap()
        .spawn::<()>(|| panic!("test preparation failure"))
        .unwrap();
    awakened
        .recv_timeout(std::time::Duration::from_secs(5))
        .unwrap();
    assert!(matches!(task.poll(), Poll::Ready(Err(_))));
    assert!(!busy.load(Ordering::Acquire));
}
