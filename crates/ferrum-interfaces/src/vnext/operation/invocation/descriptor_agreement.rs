use crate::vnext::DeviceDescriptor;

/// Equality over two retained immutable operands, not an assertion that a
/// runtime or plan getter always returns the same descriptor. The references
/// keep the compared objects alive and borrowed while the proof is reusable.
#[derive(Default)]
pub(super) struct DeviceDescriptorAgreement<'runtime, 'plan> {
    operands: Option<(&'runtime DeviceDescriptor, &'plan DeviceDescriptor)>,
}

impl<'runtime, 'plan> DeviceDescriptorAgreement<'runtime, 'plan> {
    pub(super) fn matches(
        &mut self,
        runtime: &'runtime DeviceDescriptor,
        plan: &'plan DeviceDescriptor,
    ) -> bool {
        #[cfg(test)]
        if REFERENCE_COMPARISON.get() {
            return runtime == plan;
        }
        if self.operands.is_some_and(|(prior_runtime, prior_plan)| {
            std::ptr::eq(prior_runtime, runtime) && std::ptr::eq(prior_plan, plan)
        }) {
            return true;
        }
        if runtime != plan {
            return false;
        }
        self.operands = Some((runtime, plan));
        true
    }
}

#[cfg(test)]
thread_local! {
    // Test-only scoped reference for paired CPU diagnostics of the actual
    // constructor. This is neither a product option nor a persistent cache.
    static REFERENCE_COMPARISON: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

#[cfg(test)]
pub(super) fn with_reference_comparison<T>(reference: bool, run: impl FnOnce() -> T) -> T {
    struct Reset(bool);
    impl Drop for Reset {
        fn drop(&mut self) {
            REFERENCE_COMPARISON.set(self.0);
        }
    }
    let _reset = Reset(REFERENCE_COMPARISON.replace(reference));
    run()
}
