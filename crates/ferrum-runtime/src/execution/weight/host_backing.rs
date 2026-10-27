use std::{fmt, sync::Arc};

/// Invocation-local additional physical Host WeightBacking bound, together with
/// ownership of any warm backing excluded from that bound. Retained backing
/// keeps its existing parent charge; only `additional_bytes` needs new credit.
///
/// This is not serialized plan metadata. The execution owner must keep this
/// preparation alive from budget calculation through static initialization.
#[derive(Clone)]
pub struct HostBackingPreparation {
    additional_bytes: usize,
    retention: Option<Arc<dyn Send + Sync>>,
}

impl HostBackingPreparation {
    pub fn new(additional_bytes: usize) -> Self {
        Self {
            additional_bytes,
            retention: None,
        }
    }

    /// Retain an implementation-owned warm cache lease without exposing its
    /// payload or allowing callers to synthesize allocation authority from it.
    pub fn with_retention(mut self, owner: Arc<dyn Send + Sync>) -> Self {
        self.retention = Some(owner);
        self
    }

    pub fn additional_bytes(&self) -> usize {
        self.additional_bytes
    }
}

impl fmt::Debug for HostBackingPreparation {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("HostBackingPreparation")
            .field("additional_bytes", &self.additional_bytes)
            .field("retains_warm_backing", &self.retention.is_some())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct Lease(Arc<AtomicUsize>);

    impl Drop for Lease {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }

    #[test]
    fn cloned_preparation_retains_warm_backing_until_last_execution_owner() {
        let drops = Arc::new(AtomicUsize::new(0));
        let prepared =
            HostBackingPreparation::new(64).with_retention(Arc::new(Lease(Arc::clone(&drops))));
        let execution_owner = prepared.clone();
        assert_eq!(execution_owner.additional_bytes(), 64);
        drop(prepared);
        assert_eq!(drops.load(Ordering::SeqCst), 0);
        drop(execution_owner);
        assert_eq!(drops.load(Ordering::SeqCst), 1);
    }
}
