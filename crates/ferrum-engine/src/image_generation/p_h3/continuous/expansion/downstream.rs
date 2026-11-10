//! One bounded CPU preparation, with request-owned completion and cleanup.
use anyhow::{anyhow, Result};
use std::{
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc, Arc,
    },
    task::{Poll, Waker},
    thread::JoinHandle,
};

pub(super) struct PreparationPermit {
    busy: Arc<AtomicBool>,
    wake: Waker,
}

impl PreparationPermit {
    pub(super) fn try_acquire(busy: &Arc<AtomicBool>, wake: &Waker) -> Option<Self> {
        busy.compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .ok()
            .map(|_| Self {
                busy: busy.clone(),
                wake: wake.clone(),
            })
    }

    pub(super) fn spawn<T: Send + 'static>(
        self,
        work: impl FnOnce() -> Result<T> + Send + 'static,
    ) -> Result<PreparationTask<T>> {
        let (sender, receiver) = mpsc::sync_channel(1);
        let handle = std::thread::Builder::new()
            .name("h3-plan-prepare".into())
            .spawn(move || {
                let permit = self;
                // Publish before waking, including panic/error exits, so the
                // consumer cannot miss completion between its poll and park.
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(work))
                    .unwrap_or_else(|_| Err(anyhow!("image metadata preparation worker panicked")));
                let _ = sender.send(result);
                drop(permit);
            })?;
        Ok(PreparationTask {
            receiver,
            handle: Some(handle),
        })
    }
}

impl Drop for PreparationPermit {
    fn drop(&mut self) {
        self.busy.store(false, Ordering::Release);
        self.wake.wake_by_ref();
    }
}

pub(super) struct PreparationTask<T> {
    receiver: mpsc::Receiver<Result<T>>,
    handle: Option<JoinHandle<()>>,
}

impl<T> PreparationTask<T> {
    pub(super) fn poll(&mut self) -> Poll<Result<T>> {
        match self.receiver.try_recv() {
            Ok(result) => {
                if self
                    .handle
                    .take()
                    .is_some_and(|handle| handle.join().is_err())
                {
                    return Poll::Ready(Err(anyhow!("image metadata preparation worker failed")));
                }
                Poll::Ready(result)
            }
            Err(mpsc::TryRecvError::Empty) => Poll::Pending,
            Err(mpsc::TryRecvError::Disconnected) => Poll::Ready(Err(anyhow!(
                "image metadata preparation worker disconnected"
            ))),
        }
    }
}

impl<T> Drop for PreparationTask<T> {
    fn drop(&mut self) {
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

#[cfg(test)]
mod tests;
