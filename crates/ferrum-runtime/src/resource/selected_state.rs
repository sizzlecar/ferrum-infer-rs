//! Owned, same-frame selection. This is not a host-authored completion proof.
use super::*;
use crate::{BatchOperationIdentity, CandidateStateLayout};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SelectedPrefix {
    All,
    One,
    Two,
}

#[must_use = "selected state must be submitted or its frame is aborted"]
pub struct PreparedSelectedStateCommit<R: DeviceRuntime> {
    pub(crate) step: Arc<StepResourceLease<R>>,
    pub(crate) identity: BatchOperationIdentity,
    pub(crate) selections: Vec<SelectedPrefix>,
    pub(crate) proof: Option<StepCompletedBoundaryProof>,
    pub(crate) retentions: Vec<DeviceBufferRetention>,
    succeeded: bool,
}
impl<R: DeviceRuntime> StepResourceLease<R> {
    pub(crate) fn candidate_layout(&self) -> Option<&CandidateStateLayout> {
        self.participants[0]
            .session
            .resources()
            .request
            .plan
            .dynamic_pools()
            .candidate_state_layout
            .as_ref()
    }
    pub fn try_prepare_selected_state_commit(
        self: &Arc<Self>,
        selections: Vec<SelectedPrefix>,
    ) -> Result<PreparedSelectedStateCommit<R>, VNextError> {
        if self.candidate_layout().is_none() {
            return Err(invalid_resource(
                "plan has no complete candidate-state effects",
            ));
        }
        let (proof, identity) = super::sequence::take_selected_proof(self, &selections)?;
        let identity = identity.for_selected_state_commit(&selections)?;
        Ok(PreparedSelectedStateCommit {
            step: Arc::clone(self),
            identity,
            selections,
            proof: Some(proof),
            retentions: Vec::new(),
            succeeded: false,
        })
    }
}
impl<R: DeviceRuntime> PreparedSelectedStateCommit<R> {
    pub(crate) fn runtime(&self) -> &Arc<R> {
        self.step.execution_lane.runtime_arc()
    }
    pub(crate) fn cleanup_domain(&self) -> DeferredDeviceCleanupDomainId {
        self.step.participants[0]
            .session
            .resources()
            .request
            .plan
            .resources
            .deferred_cleanup_domain
    }
    pub(crate) fn finish(&mut self, succeeded: bool) -> Result<(), VNextError> {
        if !succeeded {
            return Ok(());
        }
        let mut slot = self
            .step
            .completed_boundary
            .lock()
            .map_err(|_| invalid_resource("selected copy proof mutex poisoned"))?;
        if !matches!(*slot, StepCompletedBoundarySlot::CopyPending) {
            return Err(invalid_resource("selected copy proof changed"));
        }
        *slot = StepCompletedBoundarySlot::CopySucceeded(
            self.proof
                .take()
                .ok_or_else(|| invalid_resource("selected proof already consumed"))?,
        );
        self.succeeded = true;
        Ok(())
    }
}
impl<R: DeviceRuntime> Drop for PreparedSelectedStateCommit<R> {
    fn drop(&mut self) {
        if !self.succeeded {
            for p in &self.step.participants {
                super::poison_session_frame(&p.frame);
            }
            if let Ok(mut s) = self.step.completed_boundary.lock() {
                *s = StepCompletedBoundarySlot::Invalid;
            }
        }
    }
}
