use std::any::Any;
use std::sync::Arc;

/// Process-local lifetime ownership only. The fingerprint remains the semantic
/// identity; this owner cannot grant buffer, admission, or execution authority.
/// Revalidation rebuilds it through the registered estimator, never from wire.
#[derive(Clone)]
pub(crate) struct PreparedImplementationRetention {
    pub(super) fingerprint: String,
    pub(super) owner: Arc<dyn Any + Send + Sync>,
}

impl PreparedImplementationRetention {
    pub(crate) fn downcast<T: Any + Send + Sync>(&self) -> Option<Arc<T>> {
        Arc::clone(&self.owner).downcast().ok()
    }
}

impl std::fmt::Debug for PreparedImplementationRetention {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("PreparedImplementationRetention")
            .field("fingerprint", &self.fingerprint)
            .finish_non_exhaustive()
    }
}

// Independently acquired leases for the same prepared implementation have the
// same semantic identity. Ownership pointer addresses must not change a plan's
// equality or wire identity when it is rebuilt in the same runtime registry.
impl PartialEq for PreparedImplementationRetention {
    fn eq(&self, other: &Self) -> bool {
        self.fingerprint == other.fingerprint
    }
}
impl Eq for PreparedImplementationRetention {}
