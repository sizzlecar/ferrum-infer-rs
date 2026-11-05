//! Opaque resources attached to the original plan lifetime. Only the actual
//! checked close path releases them. Abandonment/quarantine is not Closed.

#[derive(Default)]
pub(in crate::resource) struct PlanRuntimeCloseCustody {
    resources: Vec<Box<dyn Send + 'static>>,
}

impl PlanRuntimeCloseCustody {
    pub(super) fn retain(&mut self, resource: impl Send + 'static) {
        self.resources.push(Box::new(resource));
    }

    pub(super) fn release(mut self) {
        drop(std::mem::take(&mut self.resources));
    }
}

impl Drop for PlanRuntimeCloseCustody {
    fn drop(&mut self) {
        // Existing native abandonment can quarantine without an observable
        // close receipt. Preserve the corresponding original phase authority;
        // no extra reporter, reaper, ledger or retirement proof is created.
        if !self.resources.is_empty() {
            tracing::error!("plan resources abandoned without checked close; original caller custody quarantined");
        }
        for resource in self.resources.drain(..) {
            std::mem::forget(resource);
        }
    }
}
