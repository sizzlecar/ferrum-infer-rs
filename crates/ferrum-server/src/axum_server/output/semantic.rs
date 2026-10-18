//! Private HTTP values retain the same session as the inference that produced them.
use ferrum_output::{semantic_cost, OutputSemanticLease, OutputSession};
use ferrum_types::Result;
use serde::Serialize;
use std::sync::Arc;

struct Owned<T> {
    value: T,
    lease: OutputSemanticLease,
}

pub(in crate::axum_server) struct Retained<T>(Arc<Owned<T>>);

impl<T> Clone for Retained<T> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

impl<T: Serialize> Retained<T> {
    pub(in crate::axum_server) fn new(session: &OutputSession, value: T) -> Result<Self> {
        let lease = session.try_reserve_semantic(semantic_cost(&value)?)?;
        Ok(Self(Arc::new(Owned { value, lease })))
    }
}

impl<T> Retained<T> {
    pub(in crate::axum_server) fn value(&self) -> &T {
        &self.0.value
    }
    pub(in crate::axum_server) fn output_session(&self) -> &OutputSession {
        self.0.lease.output_session()
    }
}
