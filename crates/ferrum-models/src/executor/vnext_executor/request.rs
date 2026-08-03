use std::sync::Arc;

use ferrum_interfaces::vnext::{
    AdmittedRequestResources, DeviceRuntime, RequestIdentity, SequenceSession,
};
use ferrum_types::{FerrumError, RequestId, Result};

/// Product identity and exact core Request authority shared by every child
/// sequence of one request. The root never owns a child, so the last child or
/// in-flight hold releases Request-lifetime backing without an ownership cycle.
pub(super) struct VNextRequestRoot<R: DeviceRuntime> {
    product_request_id: RequestId,
    resources: Arc<AdmittedRequestResources<R>>,
}

impl<R: DeviceRuntime> VNextRequestRoot<R> {
    pub(super) fn bind_initial(
        product_request_id: RequestId,
        request_identity: &RequestIdentity,
        session: &Arc<SequenceSession<R>>,
    ) -> Result<Arc<Self>> {
        let resources = Arc::clone(session.resources().request_resources());
        if resources.request_id() != request_identity {
            return Err(FerrumError::internal(format!(
                "vNext product request `{product_request_id}` bound core identity `{}`, expected `{request_identity}`",
                resources.request_id(),
            )));
        }
        let root = Arc::new(Self {
            product_request_id,
            resources,
        });
        root.validate_child_session(session)?;
        Ok(root)
    }

    pub(super) fn validate_child_session(&self, session: &Arc<SequenceSession<R>>) -> Result<()> {
        let sequence_resources = session.resources();
        if !Arc::ptr_eq(&self.resources, sequence_resources.request_resources())
            || self.resources.request_authority() != session.request_authority()
            || self.resources.coordinator_id() != sequence_resources.coordinator_id()
            || self.resources.run_id() != sequence_resources.run_id()
            || self.resources.request_id() != sequence_resources.request_id()
        {
            return Err(FerrumError::internal(format!(
                "vNext sequence {} does not belong to product request `{}` exact Request root",
                session.sequence_authority().sparse_id(),
                self.product_request_id
            )));
        }
        Ok(())
    }

    pub(super) fn product_request_id(&self) -> &RequestId {
        &self.product_request_id
    }
}

pub(super) fn terminalize_unsubmitted_session<R: DeviceRuntime>(
    session: &Arc<SequenceSession<R>>,
    error: FerrumError,
) -> FerrumError {
    match session.try_abort_if_quiescent() {
        Ok(_) => error,
        Err(cleanup_error) => FerrumError::backend(format!(
            "{error}; vNext unsubmitted sequence terminalization failed: {cleanup_error}"
        )),
    }
}
