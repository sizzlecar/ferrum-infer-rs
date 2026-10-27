use super::*;
use std::{sync::Mutex, task::Wake};

struct ReadUsageOnWake {
    budget: OutputBudget,
    observed: Mutex<Vec<OutputUsage>>,
}

impl Wake for ReadUsageOnWake {
    fn wake(self: Arc<Self>) {
        self.wake_by_ref();
    }

    fn wake_by_ref(self: &Arc<Self>) {
        self.observed.lock().unwrap().push(self.budget.usage());
    }
}

fn budget() -> OutputBudget {
    OutputBudget::new(OutputConfig {
        max_sessions: 1,
        encoded_bytes_total: 16,
        max_event_bytes: 16,
        ..OutputConfig::default()
    })
    .unwrap()
}

fn subscribe(budget: &OutputBudget) -> (Arc<ReadUsageOnWake>, ResourceChangeSubscription) {
    let probe = Arc::new(ReadUsageOnWake {
        budget: budget.clone(),
        observed: Mutex::new(Vec::new()),
    });
    let subscription = budget.subscribe_capacity(Waker::from(probe.clone()));
    (probe, subscription)
}

#[test]
fn encoded_and_session_release_are_visible_to_synchronous_subscribers() {
    let budget = budget();
    let session = budget.try_open_session().unwrap();
    let held = session.try_reserve_encoded(16).unwrap().unwrap();
    let (probe, subscription) = subscribe(&budget);
    assert!(session.try_reserve_encoded(1).unwrap().is_none());
    assert!(session.try_reserve_encoded(17).is_err());
    drop(held);
    assert_eq!(
        probe.observed.lock().unwrap().last().unwrap().encoded_bytes,
        0
    );
    let replacement = session.try_reserve_encoded(16).unwrap().unwrap();
    drop(subscription);
    let observed = probe.observed.lock().unwrap().len();
    drop(replacement);
    assert_eq!(probe.observed.lock().unwrap().len(), observed);
    let _subscription = session.subscribe_capacity(Waker::from(probe.clone()));
    drop(session);
    assert_eq!(probe.observed.lock().unwrap().last().unwrap().sessions, 0);
}

#[tokio::test]
async fn cancelling_a_partial_async_grant_refunds_before_notifying() {
    let budget = budget();
    let session = budget.try_open_session().unwrap();
    let held = session.try_reserve_encoded(8).unwrap().unwrap();
    let mut waiting = Box::pin(session.reserve_encoded(16));
    assert!(futures::poll!(&mut waiting).is_pending());
    assert!(session.try_reserve_encoded(1).unwrap().is_none());
    let (probe, _subscription) = subscribe(&budget);
    drop(waiting);
    assert_eq!(
        probe.observed.lock().unwrap().last().unwrap().encoded_bytes,
        8
    );
    let remainder = session.try_reserve_encoded(8).unwrap().unwrap();
    drop((held, remainder));
    assert_eq!(budget.usage().encoded_bytes, 0);
}

#[tokio::test]
async fn closing_wakes_waiters_and_keeps_existing_output_charged() {
    let budget = budget();
    let session = budget.try_open_session().unwrap();
    assert!(!budget.is_closed());
    assert!(!session.is_closed());
    let held = session.try_reserve_encoded(16).unwrap().unwrap();
    let mut waiting = Box::pin(session.reserve_encoded(1));
    assert!(futures::poll!(&mut waiting).is_pending());
    let (probe, _subscription) = subscribe(&budget);
    budget.close();
    assert!(budget.is_closed());
    assert!(session.is_closed());
    assert!(!probe.observed.lock().unwrap().is_empty());
    assert!(waiting.await.is_err());
    assert!(session.try_reserve_encoded(1).is_err());
    assert!(budget.try_open_session().is_err());
    assert_eq!(budget.usage().encoded_bytes, 16);
    drop((session, held));
    assert_eq!(budget.usage().encoded_bytes, 0);
    assert_eq!(budget.usage().sessions, 0);
}
