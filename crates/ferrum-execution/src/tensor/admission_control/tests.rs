use super::*;
use std::pin::Pin;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    mpsc,
};
use std::time::Duration;

#[test]
fn parked_wait_does_not_poll_until_release_and_cancellation_wakes_it() {
    for cancel in [false, true] {
        let control = TensorAdmissionControl::new();
        let waiter = control.clone();
        let polls = Arc::new(AtomicUsize::new(0));
        let observed = polls.clone();
        let (entered_tx, entered_rx) = mpsc::channel();
        let (done_tx, done_rx) = mpsc::channel();
        let (release_tx, mut release_rx) = tokio::sync::oneshot::channel::<()>();
        let thread = thread::spawn(move || {
            let mut entered = Some(entered_tx);
            let result = waiter.wait_until(std::future::poll_fn(|cx| {
                observed.fetch_add(1, Ordering::SeqCst);
                if let Some(entered) = entered.take() {
                    entered.send(()).unwrap();
                }
                Pin::new(&mut release_rx)
                    .poll(cx)
                    .map(|value| value.map_err(|_| invalid("release sender lost")))
            }));
            done_tx.send(result).unwrap();
        });
        entered_rx.recv_timeout(Duration::from_secs(2)).unwrap();
        // Liveness, not performance: without an event there is no new reason
        // to poll the authoritative admission predicate.
        assert!(done_rx.recv_timeout(Duration::from_millis(30)).is_err());
        assert_eq!(polls.load(Ordering::SeqCst), 1);
        if cancel {
            control.cancel();
        } else {
            release_tx.send(()).unwrap();
        }
        assert_eq!(
            done_rx
                .recv_timeout(Duration::from_secs(2))
                .unwrap()
                .unwrap(),
            !cancel
        );
        thread.join().unwrap();
    }
}

#[test]
fn release_before_wait_and_cancel_before_wait_are_not_lost() {
    let control = TensorAdmissionControl::new();
    let (tx, rx) = tokio::sync::oneshot::channel::<()>();
    tx.send(()).unwrap();
    assert!(control
        .wait_until(async { rx.await.map_err(|_| invalid("release lost")) })
        .unwrap());
    control.cancel();
    assert!(!control.wait_until(std::future::pending()).unwrap());
}

#[tokio::test(flavor = "current_thread")]
async fn current_thread_runtime_rejects_blocking_wait_without_polling_it() {
    let control = TensorAdmissionControl::new();
    let result = control.wait_until(std::future::poll_fn(|_| -> Poll<Result<(), VNextError>> {
        panic!("must not block the only reactor worker")
    }));
    assert!(result.is_err());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn synchronous_wait_hands_back_the_tokio_worker() {
    let (tx, rx) = tokio::sync::oneshot::channel::<()>();
    tokio::spawn(async move {
        tx.send(()).unwrap();
    });
    assert!(TensorAdmissionControl::new()
        .wait_until(async { rx.await.map_err(|_| invalid("release lost")) })
        .unwrap());
}
