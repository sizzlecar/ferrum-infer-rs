//! A final native commit guard over the exact prepared transfer metadata.
use crate::execution_cost::GuardedNotSubmittedReason;
use crate::vnext::{
    NativeCheckpointTransferCostDomain, NativeCheckpointTransferHostWork,
    NativeCheckpointTransferIdentity,
};

/// The native owner constructs this after encoding all copies and initialization.
/// Metadata cannot authorize a different session, backing or capture source.
pub struct PreparedCheckpointTransfer<'a> {
    identity: &'a NativeCheckpointTransferIdentity,
    source_capture_identity: Option<&'a NativeCheckpointTransferIdentity>,
    cost_domain: &'a NativeCheckpointTransferCostDomain,
    host_work: Option<NativeCheckpointTransferHostWork>,
}
impl<'a> PreparedCheckpointTransfer<'a> {
    pub(super) fn new(
        identity: &'a NativeCheckpointTransferIdentity,
        source_capture_identity: Option<&'a NativeCheckpointTransferIdentity>,
        cost_domain: &'a NativeCheckpointTransferCostDomain,
        host_work: Option<NativeCheckpointTransferHostWork>,
    ) -> Self {
        Self {
            identity,
            source_capture_identity,
            cost_domain,
            host_work,
        }
    }
    pub fn identity(&self) -> &NativeCheckpointTransferIdentity {
        self.identity
    }
    pub fn source_capture_identity(&self) -> Option<&NativeCheckpointTransferIdentity> {
        self.source_capture_identity
    }
    pub fn host_work(&self) -> Option<&NativeCheckpointTransferHostWork> {
        self.host_work.as_ref()
    }
    pub fn cost_domain(&self) -> &NativeCheckpointTransferCostDomain {
        self.cost_domain
    }
}

/// Called at the backend's last guarded commit boundary. Implementations must
/// be bounded and nonblocking, check their original deadline/model generation,
/// and validate this exact domain. They must not re-enter the lane or reaper.
/// Rejection is returned only after the native definitely-not-submitted rollback.
pub trait CheckpointTransferSubmissionGuard: Send + Sync {
    /// Unknown-cost completion maintenance grants no latency witness.
    fn relies_on_cost_witness(&self) -> bool {
        false
    }
    fn check(
        &self,
        prepared: &PreparedCheckpointTransfer<'_>,
    ) -> Result<(), GuardedNotSubmittedReason>;
}
