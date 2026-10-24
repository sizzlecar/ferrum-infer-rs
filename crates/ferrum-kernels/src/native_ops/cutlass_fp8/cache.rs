//! Bounded reuse of allocation-free native planning records. Eviction does not
//! retire commands: each encoded command retains its own record and module owner.

use super::{Error, PreparedCutlassFp8, Request};
use std::collections::VecDeque;
use std::sync::Arc;

pub(crate) struct PreparedPlanCache {
    maximum_entries: usize,
    entries: VecDeque<Arc<PreparedCutlassFp8>>,
}

impl PreparedPlanCache {
    pub(crate) fn new(maximum_entries: usize) -> Result<Self, Error> {
        if maximum_entries == 0 {
            return Err(Error::InvalidArgument);
        }
        Ok(Self {
            maximum_entries,
            entries: VecDeque::new(),
        })
    }

    pub(crate) fn cached(&mut self, request: Request) -> Option<Arc<PreparedCutlassFp8>> {
        let index = self
            .entries
            .iter()
            .position(|plan| plan.plan().request == request)?;
        let plan = self.entries.remove(index)?;
        self.entries.push_front(Arc::clone(&plan));
        Some(plan)
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn get_or_prepare(
        &mut self,
        request: Request,
    ) -> Result<Arc<PreparedCutlassFp8>, Error> {
        self.get_or_prepare_with(request, PreparedCutlassFp8::new)
    }

    fn get_or_prepare_with(
        &mut self,
        request: Request,
        prepare: impl FnOnce(Request) -> Result<PreparedCutlassFp8, Error>,
    ) -> Result<Arc<PreparedCutlassFp8>, Error> {
        request.validate()?;
        if let Some(plan) = self.cached(request) {
            return Ok(plan);
        }
        // A failed preparation must not discard a usable cached record.
        let plan = Arc::new(prepare(request)?);
        if plan.plan().request != request {
            return Err(Error::Plan);
        }
        if self.entries.len() == self.maximum_entries {
            self.entries.pop_back();
        }
        self.entries.push_front(Arc::clone(&plan));
        Ok(plan)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::native_ops::cutlass_fp8::{Device, Plan};

    fn request(rows: u32) -> Request {
        Request::new(
            rows,
            16,
            32,
            Device {
                ordinal: 0,
                architecture: 1200,
                multiprocessors: 12,
                maximum_dynamic_shared_bytes: 100_000,
            },
        )
        .unwrap()
    }

    fn prepare(request: Request) -> Result<PreparedCutlassFp8, Error> {
        Ok(PreparedCutlassFp8 {
            plan: Plan::checked(request, 4096, 0)?,
        })
    }

    #[test]
    fn eviction_preserves_retained_commands_and_reconstructed_identity() {
        let mut cache = PreparedPlanCache::new(1).unwrap();
        let original = cache.get_or_prepare_with(request(1), prepare).unwrap();
        let lifetime = Arc::downgrade(&original);
        cache.get_or_prepare_with(request(2), prepare).unwrap();
        assert_eq!(cache.entries.len(), 1);
        assert!(lifetime.upgrade().is_some());
        let rebuilt = cache.get_or_prepare_with(request(1), prepare).unwrap();
        assert_eq!(cache.entries.len(), 1);
        assert!(!Arc::ptr_eq(&original, &rebuilt));
        assert_eq!(
            original.fingerprint_material(),
            rebuilt.fingerprint_material()
        );
        drop(original);
        assert!(lifetime.upgrade().is_none());
    }

    #[test]
    fn hits_refresh_recency_and_failed_misses_preserve_entries() {
        let mut cache = PreparedPlanCache::new(2).unwrap();
        cache.get_or_prepare_with(request(1), prepare).unwrap();
        cache.get_or_prepare_with(request(2), prepare).unwrap();
        cache
            .get_or_prepare_with(request(1), |_| panic!("cache hit must reuse the plan"))
            .unwrap();
        cache.get_or_prepare_with(request(3), prepare).unwrap();
        assert!(cache.entries.iter().any(|p| p.plan().request == request(1)));
        assert!(!cache.entries.iter().any(|p| p.plan().request == request(2)));
        let before: Vec<_> = cache.entries.iter().map(Arc::as_ptr).collect();
        assert!(cache
            .get_or_prepare_with(request(4), |_| Err(Error::Unavailable))
            .is_err());
        assert!(cache
            .get_or_prepare_with(request(4), |_| prepare(request(5)))
            .is_err());
        assert_eq!(
            before,
            cache.entries.iter().map(Arc::as_ptr).collect::<Vec<_>>()
        );
    }

    #[test]
    fn request_identity_includes_device_and_invalid_requests_do_not_enter_cache() {
        assert!(PreparedPlanCache::new(0).is_err());
        let mut cache = PreparedPlanCache::new(2).unwrap();
        let first = cache.get_or_prepare_with(request(1), prepare).unwrap();
        let mut another_device = request(1);
        another_device.device_ordinal = 1;
        let second = cache.get_or_prepare_with(another_device, prepare).unwrap();
        assert_ne!(first.fingerprint_material(), second.fingerprint_material());
        let mut invalid = request(1);
        invalid.rows = 0;
        assert!(cache
            .get_or_prepare_with(invalid, |_| panic!("invalid request reached preparation"))
            .is_err());
        assert_eq!(cache.entries.len(), 2);
    }
}
