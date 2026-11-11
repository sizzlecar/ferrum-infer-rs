//! Owned phase handoff around the original static-construction resource chain.
//! Native retirement authority stays with its original transaction/owner.
use crate::{ExecutorBuildError, ExecutorShutdownObserver};
use std::cell::Cell;

pub(super) struct PhaseCustody<P: Send + 'static> {
    resource: Option<P>,
    native_started: Cell<bool>,
}

impl<P: Send + 'static> PhaseCustody<P> {
    pub(super) fn new(resource: P) -> Self {
        Self {
            resource: Some(resource),
            native_started: Cell::new(false),
        }
    }

    pub(super) fn native_started(&self) -> &Cell<bool> {
        &self.native_started
    }

    pub(super) fn handoff(mut self, observer: &ExecutorShutdownObserver) {
        observer.retain_until_retired(self.resource.take().expect("phase custody hands off once"));
    }

    pub(super) fn handoff_to_plan<R: ferrum_runtime::DeviceRuntime>(
        mut self,
        resources: &ferrum_runtime::PlanRuntimeResources<R>,
    ) {
        resources.retain_until_closed(self.resource.take().expect("phase custody hands off once"));
    }

    pub(super) fn finish_error(mut self, error: &ExecutorBuildError) {
        if let Some(cleanup) = error.cleanup() {
            cleanup
                .retain_until_retired(self.resource.take().expect("phase custody hands off once"));
        } else {
            // The actual constructor returns no cleanup only before native
            // ownership, or after its checked synchronous retirement succeeds.
            drop(self.resource.take());
        }
    }
}

impl<P: Send + 'static> Drop for PhaseCustody<P> {
    fn drop(&mut self) {
        if self.native_started.get() {
            if let Some(original) = self.resource.take() {
                // A raw panic can consume a transaction inside runtime's
                // initialization method before a failure owner is returned.
                // Its existing native-abandonment fallback has no observable
                // retirement handle here. Preserve the phase grant, not a new
                // reporter or fabricated cleanup success. This one-instance
                // quarantine cannot currently be retried/reclaimed by callers.
                tracing::error!("static construction ended without phase handoff or retirement proof; original phase resource quarantined");
                std::mem::forget(original);
            }
        }
    }
}
