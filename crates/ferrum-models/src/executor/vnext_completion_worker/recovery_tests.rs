//! Deterministic recovery wakes around the real bounded worker's Tokio permits.
use super::*;
use std::sync::{mpsc as blocking, Mutex};
use std::time::Duration;

struct RecoveryGate {
    release: Option<blocking::Sender<()>>,
    recovered: Option<oneshot::Receiver<()>>,
    calls: Arc<AtomicUsize>,
}

impl RecoveryGate {
    async fn closed(worker: &VNextCompletionWorker) -> Self {
        let (entered_sender, entered_receiver) = oneshot::channel();
        let (recovered_sender, recovered_receiver) = oneshot::channel();
        let (release, released) = blocking::channel();
        let entered = Mutex::new(Some(entered_sender));
        let recovered = Mutex::new(Some(recovered_sender));
        let released = Mutex::new(released);
        let calls = Arc::new(AtomicUsize::new(0));
        let callback_calls = Arc::clone(&calls);
        worker
            .install_recovery(
                move || match callback_calls.fetch_add(1, Ordering::AcqRel) {
                    0 => {
                        let _ = entered.lock().unwrap().take().unwrap().send(());
                        let _ = released.lock().unwrap().recv();
                    }
                    1 => {
                        let _ = recovered.lock().unwrap().take().unwrap().send(());
                    }
                    _ => {}
                },
            )
            .unwrap();
        // Install Drop cleanup before awaiting, so a failed watchdog cannot
        // leave the real worker blocked while its owner tries to join it.
        let gate = Self {
            release: Some(release),
            recovered: Some(recovered_receiver),
            calls,
        };
        worker.recovery_wake().request();
        tokio::time::timeout(Duration::from_secs(3), entered_receiver)
            .await
            .expect("worker did not enter the recovery callback")
            .unwrap();
        gate
    }

    async fn open_and_wait(&mut self) {
        self.release.take().unwrap().send(()).unwrap();
        // A failure deadline, not a scheduler delay or a polling loop.
        tokio::time::timeout(Duration::from_secs(3), self.recovered.take().unwrap())
            .await
            .expect("pending recovery was not delivered")
            .unwrap();
        assert_eq!(self.calls.load(Ordering::Acquire), 2);
    }
}

impl Drop for RecoveryGate {
    fn drop(&mut self) {
        if let Some(release) = self.release.take() {
            let _ = release.send(());
        }
    }
}

#[tokio::test]
async fn recovery_reserve_cancellation_releases_capacity_before_waking_worker() {
    let worker = VNextCompletionWorker::new().unwrap();
    let mut gate = RecoveryGate::closed(&worker).await;
    let sender = worker.sender.as_ref().unwrap();

    // The receiver has consumed the initial wake and is inside its callback.
    // Occupy the only capacity permit without putting any message in the queue.
    let occupied = sender.try_reserve().unwrap();
    let ran = Arc::new(AtomicBool::new(false));
    let task_ran = Arc::clone(&ran);
    let mut abandoned = Box::pin(worker.execute(
        VNextCompletionTaskKind::WaveReadback,
        move || {
            task_ran.store(true, Ordering::Release);
        },
    ));
    assert!(futures::poll!(abandoned.as_mut()).is_pending());

    // Tokio transfers the released permit to the registered reserve waiter.
    // Do not poll it again: cancellation now drops an Acquire which owns
    // capacity, although execute has not yet received a ready Permit.
    drop(occupied);
    assert_eq!(sender.capacity(), 0);
    worker.recovery_wake().request();
    assert!(worker.recovery.pending.load(Ordering::Acquire));
    assert_eq!(
        sender.capacity(),
        0,
        "the wake cannot bypass the reserved permit"
    );
    drop(abandoned);

    // The receiver is still blocked in the first callback. No other permit
    // owner remains, so capacity zero now proves a replacement wake was queued.
    // Waking before dropping Acquire (or never retrying the wake) leaves one.
    assert_eq!(
        sender.capacity(),
        0,
        "cancel must release its permit before re-enqueuing recovery"
    );
    assert_eq!(worker.metrics_snapshot()["scheduled_tasks"], 0);
    assert!(!ran.load(Ordering::Acquire));
    gate.open_and_wait().await;
    assert!(!ran.load(Ordering::Acquire));
    assert_eq!(worker.pending_tasks(), 0);
    assert_eq!(worker.metrics_snapshot()["failed_recoveries"], 0);
}

#[tokio::test]
async fn recovery_requested_during_callback_runs_without_another_queue_message() {
    let worker = VNextCompletionWorker::new().unwrap();
    let mut gate = RecoveryGate::closed(&worker).await;
    let sender = worker.sender.as_ref().unwrap();
    let occupied = sender.try_reserve().unwrap();
    worker.recovery_wake().request();
    assert!(worker.recovery.pending.load(Ordering::Acquire));
    assert_eq!(sender.capacity(), 0);

    // Keep capacity reserved until the second callback actually runs. No task
    // or channel wake can cause it: run_pending must observe the new pending
    // request after the first callback, before returning to blocking_recv.
    gate.open_and_wait().await;
    assert_eq!(sender.capacity(), 0);
    drop(occupied);
    assert_eq!(worker.metrics_snapshot()["scheduled_tasks"], 0);
    assert_eq!(worker.metrics_snapshot()["failed_recoveries"], 0);
}
