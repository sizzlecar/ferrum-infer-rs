//! Counts live base-allocation requests once; views and fences retain the same
//! charge. This deliberately does not claim pool reservation or GPU residency.

use std::sync::{Arc, Mutex};

#[derive(Default)]
struct State {
    bytes: u64,
    error: Option<&'static str>,
}

#[derive(Default)]
pub(crate) struct AllocationTracker {
    state: Mutex<State>,
}

impl AllocationTracker {
    pub(crate) fn charge(self: &Arc<Self>, bytes: u64) -> Result<AllocationCharge, String> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| "CUDA allocation accounting lock poisoned")?;
        if let Some(error) = state.error {
            return Err(error.to_owned());
        }
        state.bytes = match state.bytes.checked_add(bytes) {
            Some(total) => total,
            None => {
                state.error = Some("CUDA allocation accounting overflow");
                return Err(state.error.unwrap().to_owned());
            }
        };
        Ok(AllocationCharge {
            tracker: Arc::clone(self),
            bytes,
        })
    }

    pub(crate) fn current(&self) -> Result<u64, String> {
        let state = self
            .state
            .lock()
            .map_err(|_| "CUDA allocation accounting lock poisoned")?;
        match state.error {
            Some(error) => Err(error.to_owned()),
            None => Ok(state.bytes),
        }
    }
}

pub(crate) struct AllocationCharge {
    tracker: Arc<AllocationTracker>,
    bytes: u64,
}

impl Drop for AllocationCharge {
    fn drop(&mut self) {
        let mut state = self
            .tracker
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        match state.bytes.checked_sub(self.bytes) {
            Some(bytes) => state.bytes = bytes,
            None => state.error = Some("CUDA allocation accounting underflow"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cuda_memory_counts_base_allocation_once_until_last_retention() {
        let tracker = Arc::new(AllocationTracker::default());
        let base = Arc::new(tracker.charge(4096 + 255).unwrap());
        let view = Arc::clone(&base);
        let fence = Arc::clone(&base);
        let other = tracker.charge(512).unwrap();
        assert_eq!(tracker.current().unwrap(), 4863);
        drop(base);
        drop(view);
        assert_eq!(tracker.current().unwrap(), 4863);
        drop(fence);
        assert_eq!(tracker.current().unwrap(), 512);
        drop(other);
        assert_eq!(tracker.current().unwrap(), 0);
    }

    #[test]
    fn cuda_memory_accounting_overflow_cannot_become_a_valid_peak() {
        let tracker = Arc::new(AllocationTracker::default());
        let _charge = tracker.charge(u64::MAX).unwrap();
        assert!(tracker.charge(1).is_err());
        assert!(tracker.current().is_err());
    }
}
