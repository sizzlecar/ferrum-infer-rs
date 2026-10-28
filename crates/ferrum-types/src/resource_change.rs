//! Resource changes wake bounded schedulers without owning any resource credit.

use std::{
    sync::{Arc, Mutex, Weak},
    task::Waker,
};

#[derive(Debug, Default)]
struct Subscribers(Mutex<Vec<Weak<Waker>>>);

/// A notification source, independent of any async runtime or waiting policy.
/// Owners must release their accounting locks before notifying subscribers.
#[derive(Debug, Clone, Default)]
pub struct ResourceChangeNotifier(Arc<Subscribers>);

impl ResourceChangeNotifier {
    /// Subscribe before checking resource availability. The caller's wake
    /// target must record a generation so a change before parking is not lost.
    pub fn subscribe(&self, waker: Waker) -> ResourceChangeSubscription {
        let waker = Arc::new(waker);
        self.0
             .0
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .push(Arc::downgrade(&waker));
        ResourceChangeSubscription {
            subscribers: Arc::downgrade(&self.0),
            waker,
        }
    }

    /// Notification is a reason to retry admission, never an admission grant.
    /// Wakers run after the subscriber lock is released and may reenter APIs.
    pub fn notify(&self) {
        let wakers: Vec<_> = self
            .0
             .0
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .iter()
            .filter_map(Weak::upgrade)
            .collect();
        for waker in wakers {
            waker.wake_by_ref();
        }
    }
}

/// Dropping this owner unsubscribes. A notification already copied by another
/// thread may finish, so its wake target must remain safe after cancellation.
#[derive(Debug)]
pub struct ResourceChangeSubscription {
    subscribers: Weak<Subscribers>,
    waker: Arc<Waker>,
}

impl Drop for ResourceChangeSubscription {
    fn drop(&mut self) {
        if let Some(subscribers) = self.subscribers.upgrade() {
            let waker = Arc::downgrade(&self.waker);
            subscribers
                .0
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .retain(|subscriber| !subscriber.ptr_eq(&waker));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        sync::atomic::{AtomicUsize, Ordering},
        task::Wake,
    };

    struct ReentrantWake {
        notifier: ResourceChangeNotifier,
        wakes: AtomicUsize,
    }

    impl Wake for ReentrantWake {
        fn wake(self: Arc<Self>) {
            self.wake_by_ref();
        }

        fn wake_by_ref(self: &Arc<Self>) {
            self.wakes.fetch_add(1, Ordering::SeqCst);
            // Registration and cancellation inside a callback must not deadlock.
            drop(self.notifier.subscribe(Waker::from(self.clone())));
        }
    }

    #[test]
    fn subscription_cancellation_and_reentrant_notification_hold_no_lock() {
        let notifier = ResourceChangeNotifier::default();
        let wake = Arc::new(ReentrantWake {
            notifier: notifier.clone(),
            wakes: AtomicUsize::new(0),
        });
        let subscription = notifier.subscribe(Waker::from(wake.clone()));
        notifier.notify();
        assert_eq!(wake.wakes.load(Ordering::SeqCst), 1);
        drop(subscription);
        notifier.notify();
        assert_eq!(wake.wakes.load(Ordering::SeqCst), 1);
    }
}
