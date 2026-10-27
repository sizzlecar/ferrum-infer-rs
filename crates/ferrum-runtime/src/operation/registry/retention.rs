use std::sync::Arc;

/// Process-local lifetime ownership only. The fingerprint remains the semantic
/// identity; this owner cannot grant buffer, admission, or execution authority.
/// Revalidation rebuilds it through the registered estimator, never from wire.
#[derive(Clone)]
pub(crate) struct PreparedImplementationRetention {
    pub(super) fingerprint: String,
    pub(super) _owner: Arc<dyn Send + Sync>,
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
