//! Explicit capacity waiting for synchronous tensor clients. No device work is
//! replayed: only an admission attempt which has not submitted may use this.
use super::*;
use std::{
    future::Future,
    pin::pin,
    task::{Context, Poll, Wake, Waker},
    thread,
};
use tokio::sync::watch;

/// Opt-in waiting and cancellation shared by a serial scheduler and its caller.
/// Cancellation is monotonic. Use a fresh control for the next independent job.
#[derive(Debug, Clone)]
pub struct TensorAdmissionControl {
    cancelled: watch::Sender<bool>,
}

impl Default for TensorAdmissionControl {
    fn default() -> Self {
        Self::new()
    }
}

impl TensorAdmissionControl {
    pub fn new() -> Self {
        Self {
            cancelled: watch::channel(false).0,
        }
    }

    pub fn cancel(&self) {
        self.cancelled.send_replace(true);
    }

    pub fn is_cancelled(&self) -> bool {
        *self.cancelled.borrow()
    }

    pub async fn cancelled(&self) {
        let mut rx = self.cancelled.subscribe();
        while !*rx.borrow_and_update() {
            // The control itself retains the sender throughout this future.
            if rx.changed().await.is_err() {
                return;
            }
        }
    }

    pub(super) fn wait<R: DeviceRuntime>(
        &self,
        registration: PlanCapacityWaitRegistration<R>,
    ) -> Result<bool, VNextError> {
        self.wait_until(async move { registration.wait_for_change().await.map(|_| ()) })
    }

    fn wait_until(
        &self,
        changed: impl Future<Output = Result<(), VNextError>>,
    ) -> Result<bool, VNextError> {
        let wait = async {
            tokio::select! {
                biased;
                _ = self.cancelled() => Ok(false),
                result = changed => result.map(|_| true),
            }
        };
        // Image services have a dedicated blocking worker. The synchronous CLI
        // can instead be called from a multi-thread Tokio runtime; hand its
        // reactor worker back before parking. A current-thread runtime cannot
        // safely run this synchronous contract and must use spawn_blocking.
        match tokio::runtime::Handle::try_current() {
            Ok(handle) if handle.runtime_flavor() == tokio::runtime::RuntimeFlavor::MultiThread => {
                tokio::task::block_in_place(|| park_future(wait))
            }
            Ok(_) => Err(invalid("synchronous tensor capacity waiting requires a blocking thread or a multi-thread Tokio runtime")),
            Err(_) => park_future(wait),
        }
    }
}

struct ThreadWake(thread::Thread);
impl Wake for ThreadWake {
    fn wake(self: Arc<Self>) {
        self.0.unpark();
    }
    fn wake_by_ref(self: &Arc<Self>) {
        self.0.unpark();
    }
}

fn park_future<T>(future: impl Future<Output = T>) -> T {
    let waker = Waker::from(Arc::new(ThreadWake(thread::current())));
    let mut context = Context::from_waker(&waker);
    let mut future = pin!(future);
    loop {
        match future.as_mut().poll(&mut context) {
            Poll::Ready(value) => return value,
            Poll::Pending => thread::park(),
        }
    }
}

/// Proof that this execution submitted nothing and its transient step/session
/// owners were retired (or its pre-existing request was left unchanged).
/// The private field prevents callers from constructing a cleanup receipt.
#[derive(Debug)]
pub struct TensorAdmissionCancellation {
    _private: (),
}
impl TensorAdmissionCancellation {
    pub(super) fn retired() -> Self {
        Self { _private: () }
    }
}

#[cfg(test)]
mod tests;
