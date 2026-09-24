//! A single engine-owned CPU worker. Waking it never takes the trainer lock.
use parking_lot::{Condvar, Mutex};
use std::{
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    thread::{self, JoinHandle, Thread},
};
use tokio::sync::Notify;

#[cfg(test)]
mod tests;

#[derive(Default)]
struct WorkerCompletion {
    result: Mutex<Option<bool>>,
    blocking_waiters: Condvar,
    async_waiters: Notify,
}

impl WorkerCompletion {
    fn finish(&self, succeeded: bool) {
        *self.result.lock() = Some(succeeded);
        self.blocking_waiters.notify_all();
        self.async_waiters.notify_waiters();
    }

    async fn wait(&self) -> bool {
        loop {
            let notified = self.async_waiters.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            if let Some(result) = *self.result.lock() {
                return result;
            }
            notified.await;
        }
    }

    fn wait_blocking(&self) {
        let mut result = self.result.lock();
        while result.is_none() {
            self.blocking_waiters.wait(&mut result);
        }
    }
}

pub(super) struct CostTrainingWorker {
    stop: Arc<AtomicBool>,
    thread: Thread,
    handle: Mutex<Option<JoinHandle<()>>>,
    completion: Arc<WorkerCompletion>,
}

impl CostTrainingWorker {
    /// `consume` processes one bounded batch and reports whether retained work
    /// remains. It must capture training state, never the owner of this worker.
    pub fn spawn(mut consume: impl FnMut() -> bool + Send + 'static) -> std::io::Result<Self> {
        let stop = Arc::new(AtomicBool::new(false));
        let stopped = stop.clone();
        let handle = thread::Builder::new()
            .name("ferrum-cost-training".to_owned())
            .spawn(move || loop {
                // unpark retains one permit: publication before park and a
                // shutdown racing with the last drain cannot lose their wake.
                thread::park();
                while consume() {
                    thread::yield_now();
                }
                if stopped.load(Ordering::Acquire) {
                    // Shutdown may have published a final checkpoint after
                    // the last empty observation. The acquire above sees its
                    // preceding writes; drain once more before leaving. An
                    // unpark permit alone cannot help after this loop exits.
                    while consume() {
                        thread::yield_now();
                    }
                    break;
                }
            })?;
        Ok(Self {
            thread: handle.thread().clone(),
            stop,
            handle: Mutex::new(Some(handle)),
            completion: Arc::new(WorkerCompletion::default()),
        })
    }

    pub fn wake(&self) {
        self.thread.unpark();
    }

    pub fn notification_thread(&self) -> Thread {
        self.thread.clone()
    }

    /// Stop after draining accepted samples. Engine shutdown first stops
    /// inference producers and joins this handle on a blocking executor.
    fn take_shutdown_handle(&self) -> Option<JoinHandle<()>> {
        self.stop.store(true, Ordering::Release);
        self.wake();
        self.handle.lock().take()
    }

    /// The unique join task owns a persistent completion/result. Cancelling
    /// an async shutdown waiter cannot discard that result or let a retry
    /// report completion while the training thread is still alive.
    pub async fn shutdown(&self) -> bool {
        if let Some(handle) = self.take_shutdown_handle() {
            let completion = self.completion.clone();
            tokio::task::spawn_blocking(move || completion.finish(handle.join().is_ok()));
        }
        self.completion.wait().await
    }

    #[cfg(test)]
    pub fn shutdown_started(&self) -> bool {
        self.stop.load(Ordering::Acquire)
    }
}

impl Drop for CostTrainingWorker {
    fn drop(&mut self) {
        // Constructor failure and callers that omit shutdown still release the
        // thread. Its closure owns only training state, so it cannot drop/join
        // this worker from inside the worker itself.
        if let Some(handle) = self.take_shutdown_handle() {
            self.completion.finish(handle.join().is_ok());
        }
        self.completion.wait_blocking();
    }
}
