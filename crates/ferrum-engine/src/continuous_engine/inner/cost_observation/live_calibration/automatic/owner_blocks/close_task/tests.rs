use super::*;
use std::sync::atomic::AtomicUsize;
use std::time::Duration;

struct Owner {
    steps: Arc<AtomicUsize>,
    abandoned: Arc<AtomicUsize>,
}
fn owner() -> Owner {
    Owner {
        steps: Arc::new(AtomicUsize::new(0)),
        abandoned: Arc::new(AtomicUsize::new(0)),
    }
}
fn abandon(owner: &mut Owner) {
    owner.abandoned.fetch_add(1, Ordering::SeqCst);
}

fn wait<T: Send + 'static, R: Send + 'static>(task: &OwnedTask<T, R>) {
    let deadline = std::time::Instant::now() + Duration::from_secs(3);
    while !task.ready() {
        assert!(std::time::Instant::now() < deadline);
        std::thread::park_timeout(Duration::from_millis(20));
    }
}

#[test]
fn owned_close_returns_the_only_owner_once_and_notifies() {
    let owner = owner();
    let count = owner.steps.clone();
    let mut task = OwnedTask::spawn(owner, std::thread::current(), abandon, |owner| {
        owner.steps.fetch_add(1, Ordering::SeqCst);
        Ok(7)
    })
    .ok()
    .unwrap();
    wait(&task);
    let (owner, result) = task.take().unwrap();
    assert_eq!(result.unwrap(), 7);
    assert_eq!(count.load(Ordering::SeqCst), 1);
    assert!(task.take().is_none());
    drop(task);
    assert_eq!(owner.abandoned.load(Ordering::SeqCst), 0);
}

#[test]
fn owned_close_panic_returns_custody_as_failure_without_retry() {
    let owner = owner();
    let count = owner.steps.clone();
    let mut task = OwnedTask::<_, ()>::spawn(owner, std::thread::current(), abandon, |owner| {
        owner.steps.fetch_add(1, Ordering::SeqCst);
        panic!("controlled numerical panic")
    })
    .ok()
    .unwrap();
    wait(&task);
    let (_owner, result) = task.take().unwrap();
    assert!(result.is_err());
    assert_eq!(count.load(Ordering::SeqCst), 1);
    assert!(task.take().is_none());
}

#[test]
fn owned_close_cancel_rejects_result_and_drop_joins_and_abandons_once() {
    let owner = owner();
    let count = owner.steps.clone();
    let abandoned = owner.abandoned.clone();
    let gate = Arc::new(TestGate::default());
    let held = gate.clone();
    let task = OwnedTask::spawn(owner, std::thread::current(), abandon, move |owner| {
        owner.steps.fetch_add(1, Ordering::SeqCst);
        held.pause()?;
        Ok(())
    })
    .ok()
    .unwrap();
    gate.wait_started();
    task.cancel();
    gate.release();
    wait(&task);
    assert!(task.slot.lock().result.as_ref().unwrap().is_err());
    drop(task);
    assert_eq!(count.load(Ordering::SeqCst), 1);
    assert_eq!(abandoned.load(Ordering::SeqCst), 1);
}
