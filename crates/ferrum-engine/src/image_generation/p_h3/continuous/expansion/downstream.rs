//! One bounded CPU preparation, with request-owned completion and cleanup.
use anyhow::{anyhow, Result};
use futures::task::AtomicWaker;
use std::{
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc, Arc,
    },
    task::{Poll, Waker},
    thread::JoinHandle,
};

#[derive(Default)]
pub(in crate::image_generation::p_h3) struct PreparationSlot {
    busy: AtomicBool,
    wake: AtomicWaker,
}

pub(in crate::image_generation::p_h3::continuous) struct PreparationPermit {
    slot: Arc<PreparationSlot>,
}

impl PreparationPermit {
    pub(in crate::image_generation::p_h3::continuous) fn try_acquire(
        slot: &Arc<PreparationSlot>,
        wake: Option<&Waker>,
    ) -> Option<Self> {
        // Register before observing busy: completion may race this attempt.
        // An optional join is driven by an old forward and must not replace
        // the actual service waiter's notification with a noop waker.
        if let Some(wake) = wake {
            slot.wake.register(wake);
        }
        slot.busy
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .ok()
            .map(|_| Self { slot: slot.clone() })
    }

    pub(in crate::image_generation::p_h3::continuous) fn spawn<T: Send + 'static>(
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
        self.slot.busy.store(false, Ordering::Release);
        self.slot.wake.wake();
    }
}

pub(in crate::image_generation::p_h3::continuous) struct PreparationTask<T> {
    receiver: mpsc::Receiver<Result<T>>,
    handle: Option<JoinHandle<()>>,
}

impl<T> PreparationTask<T> {
    /// A mandatory completed-wave boundary may join its speculative metadata.
    pub(in crate::image_generation::p_h3::continuous) fn finish(mut self) -> Result<T> {
        let result = self
            .receiver
            .recv()
            .map_err(|_| anyhow!("image metadata preparation worker disconnected"))?;
        if self
            .handle
            .take()
            .is_some_and(|handle| handle.join().is_err())
        {
            return Err(anyhow!("image metadata preparation worker failed"));
        }
        result
    }

    pub(in crate::image_generation::p_h3::continuous) fn poll(&mut self) -> Poll<Result<T>> {
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
