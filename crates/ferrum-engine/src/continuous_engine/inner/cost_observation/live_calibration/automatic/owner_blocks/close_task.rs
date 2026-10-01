//! One owned numerical close; no runtime lock or feedback consumer is captured.
use super::*;
use std::thread::{JoinHandle, Thread};

pub(super) const STACK_BYTES: usize = 2 * 1024 * 1024;

struct Slot<T, R> {
    owner: Option<T>,
    result: Option<Result<R, FerrumError>>,
}

pub(super) struct OwnedTask<T: Send + 'static, R: Send + 'static> {
    slot: Arc<Mutex<Slot<T, R>>>,
    cancel: Arc<AtomicBool>,
    handle: Option<JoinHandle<()>>,
    abandon: fn(&mut T),
}

impl<T: Send + 'static, R: Send + 'static> OwnedTask<T, R> {
    /// Retained Rust payload only. The explicit stack is charged separately;
    /// neither measure claims to bound the OS thread's total RSS.
    pub(super) fn retained_bytes() -> usize {
        std::mem::size_of::<Self>()
            + std::mem::size_of::<Mutex<Slot<T, R>>>()
            + std::mem::size_of::<AtomicBool>()
            + 4 * std::mem::size_of::<usize>()
    }

    pub(super) fn spawn(
        owner: T,
        notification: Thread,
        abandon: fn(&mut T),
        compute: impl FnOnce(&mut T) -> Result<R, FerrumError> + Send + 'static,
    ) -> Result<Self, (T, std::io::Error)> {
        let slot = Arc::new(Mutex::new(Slot {
            owner: Some(owner),
            result: None,
        }));
        let cancel = Arc::new(AtomicBool::new(false));
        let worker_slot = slot.clone();
        let worker_cancel = cancel.clone();
        let spawned = std::thread::Builder::new()
            .name("ferrum-cost-block-close".into())
            .stack_size(STACK_BYTES)
            .spawn(move || {
                // The mutex protects transfer only, never the calculation.
                let mut owner = worker_slot.lock().owner.take().expect("unique close owner");
                let result = if worker_cancel.load(Ordering::Acquire) {
                    Err(error("owned block close cancelled"))
                } else {
                    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| compute(&mut owner)))
                        .unwrap_or_else(|_| Err(error("owned block close panicked")))
                };
                let mut slot = worker_slot.lock();
                slot.owner = Some(owner);
                slot.result = Some(if worker_cancel.load(Ordering::Acquire) {
                    Err(error("owned block close cancelled"))
                } else {
                    result
                });
                drop(slot);
                notification.unpark();
            });
        match spawned {
            Ok(handle) => Ok(Self {
                slot,
                cancel,
                handle: Some(handle),
                abandon,
            }),
            Err(reason) => {
                let owner = slot
                    .lock()
                    .owner
                    .take()
                    .expect("spawn failure retains close owner");
                Err((owner, reason))
            }
        }
    }

    pub(super) fn cancel(&self) {
        self.cancel.store(true, Ordering::Release);
    }

    pub(super) fn ready(&self) -> bool {
        self.slot.lock().result.is_some()
    }

    /// Called once by the original worker after readiness. Joining also waits
    /// for the completion notification, so no task can outlive its slot.
    pub(super) fn take(&mut self) -> Option<(T, Result<R, FerrumError>)> {
        if !self.ready() {
            return None;
        }
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
        let mut slot = self.slot.lock();
        Some((
            slot.owner.take().expect("completed close owner"),
            slot.result.take()?,
        ))
    }
}

impl<T: Send + 'static, R: Send + 'static> Drop for OwnedTask<T, R> {
    fn drop(&mut self) {
        self.cancel();
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
        if let Some(mut owner) = self.slot.lock().owner.take() {
            (self.abandon)(&mut owner);
        }
    }
}

#[cfg(test)]
mod tests;

/// Instance-local test latch: no timing sleep, global hook, or fake FIFO.
#[cfg(test)]
#[derive(Default)]
pub(in crate::continuous_engine::inner::cost_observation) struct TestGate {
    state: Mutex<(bool, bool)>,
    changed: parking_lot::Condvar,
}

#[cfg(test)]
impl TestGate {
    pub(super) fn pause(&self) -> Result<(), FerrumError> {
        let mut state = self.state.lock();
        state.0 = true;
        self.changed.notify_all();
        while !state.1 {
            if self
                .changed
                .wait_for(&mut state, std::time::Duration::from_secs(5))
                .timed_out()
            {
                return Err(error("test close gate timed out"));
            }
        }
        Ok(())
    }
    pub(in crate::continuous_engine::inner::cost_observation) fn wait_started(&self) {
        let mut state = self.state.lock();
        while !state.0 {
            assert!(!self
                .changed
                .wait_for(&mut state, std::time::Duration::from_secs(5))
                .timed_out());
        }
    }
    pub(in crate::continuous_engine::inner::cost_observation) fn release(&self) {
        self.state.lock().1 = true;
        self.changed.notify_all();
    }
}
