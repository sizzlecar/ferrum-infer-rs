use std::sync::Arc;

use crate::vnext::{
    CheckedPlanStaticSlot, DeviceRuntime, LeasedBufferView, LogicalBackingBufferView,
    ResourceAllocation, StaticProvisioningLease, VNextError,
};

/// One slot in the existing constructor-local resource table. Dynamic entries
/// retain their original per-consumer pool revalidation. Static entries share
/// only an immutable lease check; callers still validate each runtime view.
pub(super) enum SharedInvocationResource<'a, R: DeviceRuntime> {
    Vacant,
    Dynamic(Arc<LogicalBackingBufferView<'a, R::Buffer>>),
    PlanStatic(CheckedPlanStaticSlot<'a, R>),
}

impl<'a, R: DeviceRuntime> SharedInvocationResource<'a, R> {
    pub(super) fn plan_static_view(
        &mut self,
        lease: &'a StaticProvisioningLease<R>,
        slot_index: usize,
        allocation: &'a ResourceAllocation,
    ) -> Result<LeasedBufferView<'a, R::Buffer>, VNextError> {
        #[cfg(test)]
        if REFERENCE_STATIC_CHECKS.get() {
            return lease.plan_static_view(slot_index, allocation);
        }
        if let Self::PlanStatic(checked) = self {
            if let Some(view) = checked.reborrow(lease, allocation, slot_index) {
                return Ok(view);
            }
        }
        let (view, checked) = lease.checked_plan_static_view(slot_index, allocation)?;
        *self = Self::PlanStatic(checked);
        Ok(view)
    }
}

#[cfg(test)]
thread_local! {
    // This disables only static check reuse. It preserves dynamic sharing and
    // the candidate table layout; an old binary is the layout-cost reference.
    static REFERENCE_STATIC_CHECKS: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

#[cfg(test)]
pub(super) fn with_plan_static_reference<T>(reference: bool, run: impl FnOnce() -> T) -> T {
    struct Reset(bool);
    impl Drop for Reset {
        fn drop(&mut self) {
            REFERENCE_STATIC_CHECKS.set(self.0);
        }
    }
    let _reset = Reset(REFERENCE_STATIC_CHECKS.replace(reference));
    run()
}

#[cfg(test)]
pub(super) fn plan_static_reuse_slot_sizes<R: DeviceRuntime>() -> (usize, usize) {
    (
        std::mem::size_of::<Option<Arc<LogicalBackingBufferView<'_, R::Buffer>>>>(),
        std::mem::size_of::<SharedInvocationResource<'_, R>>(),
    )
}
