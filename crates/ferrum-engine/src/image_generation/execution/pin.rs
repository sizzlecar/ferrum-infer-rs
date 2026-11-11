use super::*;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Holds the actual immutable native snapshot. This adds no allocation,
/// capacity budget, Host scope, or independent grant. It only makes eviction
/// ineligible while a request still requires future exact attachments.
pub struct ImageWeightOwnerPin<R: DeviceRuntime> {
    key: ImageWeightOwnerKey,
    weights: Option<SharedComponentWeights<R>>,
    pins: Arc<AtomicUsize>,
}
impl<R: DeviceRuntime> ImageWeightOwnerPin<R> {
    pub(in crate::image_generation) fn new(
        key: ImageWeightOwnerKey,
        weights: SharedComponentWeights<R>,
        pins: Arc<AtomicUsize>,
    ) -> anyhow::Result<Self> {
        pins.fetch_update(Ordering::AcqRel, Ordering::Acquire, |n| n.checked_add(1))
            .map_err(|_| anyhow::anyhow!("image weight pin count overflows"))?;
        Ok(Self {
            key,
            weights: Some(weights),
            pins,
        })
    }
    pub fn key(&self) -> ImageWeightOwnerKey {
        self.key
    }
    pub fn weights(&self) -> &SharedComponentWeights<R> {
        self.weights.as_ref().expect("live pin owns its snapshot")
    }
}
impl<R: DeviceRuntime> Drop for ImageWeightOwnerPin<R> {
    fn drop(&mut self) {
        drop(self.weights.take());
        let previous = self.pins.fetch_sub(1, Ordering::AcqRel);
        debug_assert!(previous > 0);
    }
}
