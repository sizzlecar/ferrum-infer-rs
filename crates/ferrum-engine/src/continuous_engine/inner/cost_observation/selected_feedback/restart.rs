//! Cold, same-source storage handoff. Final FIFO and feedback are never reset
//! to manufacture a reusable model; the new process has its own queue ordinal.
use super::state::RestartProgressV1;
use super::*;
use crate::continuous_engine::inner::cost_observation::profile::VerifiedRestartCatalog;
use std::path::{Path, PathBuf};

#[derive(Clone, Copy, Debug)]
pub(in crate::continuous_engine::inner::cost_observation) struct RestartFeedbackBudget {
    pub maximum_transient_bytes: usize,
    pub maximum_disk_bytes: u64,
}
/// The coordinator reserves these bounds from its original cold/cache quota
/// before calling the handoff; this DTO does not raise any configured limit.
#[derive(Clone, Copy)]
pub(in crate::continuous_engine::inner::cost_observation) struct RestartFeedbackAllowance {
    pub transient_bytes: usize,
    pub disk_bytes: u64,
}
#[derive(Debug, Serialize)]
pub(in crate::continuous_engine::inner::cost_observation) struct RestartFeedbackReceipt {
    pub path: PathBuf,
    pub bytes: u64,
    pub sha256: [u8; 32],
}

fn error(value: impl std::fmt::Display) -> FerrumError {
    FerrumError::config(format!("restart feedback: {value}"))
}
impl Monitor {
    /// Stop only the exact private resume store during the already drained
    /// restart handoff. Keep the complete State and publication gate unchanged;
    /// finish_for_restart still proves bindings/FIFO and writes the final state.
    pub fn retire_restart_storage(
        &mut self,
        expected: &Path,
    ) -> Result<RestartFeedbackReceipt, FerrumError> {
        if self.kind != FeedbackKind::StructuredV2 || self.persistence_failed || self.worker_failed
        {
            return Err(error("private restart storage is not valid"));
        }
        let (path, bytes, sha256) = self.store.retire_at(&self.state, expected).map_err(error)?;
        Ok(RestartFeedbackReceipt {
            path,
            bytes,
            sha256,
        })
    }

    pub fn restart_budget(&self, path: &Path) -> Result<RestartFeedbackBudget, FerrumError> {
        let maximum = self.policy.maximum_state_bytes.get();
        // next State + Store's Receipt state clone; each owns at most the
        // existing capacities. Vec serialization grows below 2*encoded bytes;
        // the hash Vec and final Vec are not live together. Reserve 4*maximum
        // also for Resume's bounded raw buffer + decoded state + replacement.
        let transient = self
            .state
            .retained_payload_bytes()
            .and_then(|n| n.checked_mul(2))
            .and_then(|n| n.checked_add(maximum.checked_mul(4)?))
            .and_then(|n| n.checked_add(path.as_os_str().len().checked_add(32)?.checked_mul(8)?))
            .and_then(|n| n.checked_add(8192))
            .ok_or_else(|| error("memory bound overflow"))?;
        let disk = u64::try_from(maximum)
            .ok()
            .and_then(|n| n.checked_mul(2))
            .and_then(|n| n.checked_add(64))
            .ok_or_else(|| error("disk bound overflow"))?;
        Ok(RestartFeedbackBudget {
            maximum_transient_bytes: transient,
            maximum_disk_bytes: disk,
        })
    }

    pub fn finish_for_restart(
        &mut self,
        proof: &VerifiedRestartCatalog<'_>,
        destination: &Path,
        processed_fifo: u64,
        allowance: RestartFeedbackAllowance,
        read_now: impl Fn() -> Option<u64>,
    ) -> Result<RestartFeedbackReceipt, FerrumError> {
        proof.validate_freshness(read_now().ok_or_else(|| error("clock unavailable"))?)?;
        self.finish_bound_for_restart(
            proof.original_binding(),
            proof.persisted_binding(),
            proof.scopes(),
            destination,
            processed_fifo,
            allowance,
            || proof.validate_freshness(read_now().ok_or_else(|| error("clock unavailable"))?),
        )
    }

    // Only this module's real proof entry and sibling unit tests may use the
    // binding-level state operation. No outside caller can mint equivalence.
    pub(super) fn finish_bound_for_restart(
        &mut self,
        before: &Binding,
        after: &Binding,
        scopes: &Arc<[[u8; 32]]>,
        destination: &Path,
        processed_fifo: u64,
        allowance: RestartFeedbackAllowance,
        mut validate_after_write: impl FnMut() -> Result<(), FerrumError>,
    ) -> Result<RestartFeedbackReceipt, FerrumError> {
        let budget = self.restart_budget(destination)?;
        if allowance.transient_bytes < budget.maximum_transient_bytes
            || allowance.disk_bytes < budget.maximum_disk_bytes
            || self.kind != FeedbackKind::StructuredV2
            || self.state.binding != *before
            || self.declared_scopes.as_ref() != Some(scopes)
            || scopes.len() != self.capacity
            || scopes.is_empty()
            || scopes.windows(2).any(|p| p[0] >= p[1])
            || before.source_sha256 != after.source_sha256
            || before.fit_sha256 != after.fit_sha256
            || before.protocol_sha256 != after.protocol_sha256
            || before.policy_sha256 != after.policy_sha256
            || processed_fifo < self.last_ordinal
            || self.persistence_failed
            || self.worker_failed
        {
            return Err(error("unverified state, final drain, or allowance"));
        }
        self.published.gate.store(0, Ordering::Release);
        let mut next = self.state.clone();
        next.binding = after.clone();
        next.restart_progress = Some(RestartProgressV1 {
            schema_version: 1,
            previous_session: self.state.session,
            previous_processed_fifo: processed_fifo,
            previous_feedback_fifo: self.last_ordinal,
            previous_queue_drops: self.last_drops,
            outside_catalog_observations: self.outside_catalog_observations,
            outside_support_observations: self.outside_support_observations,
            outside_route_observations: self.outside_route_observations,
            no_submission_observations: self.no_submission_observations,
            outside_preparation_observations: self.outside_preparation_observations,
        });
        let mut target =
            store::Store::create_from_state(destination, &self.policy, &next, self.capacity)
                .map_err(error)?;
        validate_after_write()?;
        // Leave target dirty if either the original or destination store cannot
        // acknowledge the drain. The outer cache manifest is a separate commit.
        self.store.finish(&self.state).map_err(error)?;
        target.finish(&next).map_err(error)?;
        let (path, bytes, sha256) = target.file_receipt().map_err(error)?;
        // Durable writes and receipt hashing may themselves cross the original
        // TTL. A clean inner file alone never authorizes the outer cache commit.
        validate_after_write()?;
        Ok(RestartFeedbackReceipt {
            path,
            bytes,
            sha256,
        })
    }

    pub fn resume_for_restart(
        policy: &SloSelectedFeedbackSettingsV1,
        path: &Path,
        binding: Binding,
        scopes: Arc<[[u8; 32]]>,
        allowance: RestartFeedbackAllowance,
    ) -> Result<Self, FerrumError> {
        let budget = Self::restart_resume_budget(policy, path)?;
        if allowance.transient_bytes < budget.maximum_transient_bytes
            || allowance.disk_bytes < budget.maximum_disk_bytes
            || scopes.is_empty()
            || scopes.len() > 128
            || scopes.windows(2).any(|p| p[0] >= p[1])
        {
            return Err(error("resume allowance or declared scopes differ"));
        }
        let mut monitor = Self::open_bound(
            policy,
            &ferrum_types::SloSelectedFeedbackStorageV1::Resume {
                path: path.to_owned(),
            },
            binding,
            scopes.len(),
            FeedbackKind::StructuredV2,
            Some(scopes),
        )?;
        monitor.published.gate.store(0, Ordering::Release);
        let progress = monitor
            .state
            .restart_progress
            .ok_or_else(|| error("receipt lacks original restart drain"))?;
        monitor.state.epoch = monitor
            .state
            .epoch
            .checked_add(1)
            .ok_or_else(|| error("epoch exhausted"))?;
        monitor.outside_catalog_observations = progress.outside_catalog_observations;
        monitor.outside_support_observations = progress.outside_support_observations;
        monitor.outside_route_observations = progress.outside_route_observations;
        monitor.no_submission_observations = progress.no_submission_observations;
        monitor.outside_preparation_observations = progress.outside_preparation_observations;
        // last_ordinal/last_drops remain zero for this new sink. All preceding
        // margins, windows, sticky revocation and counters remain in State.
        monitor.store.persist(&monitor.state).map_err(error)?;
        monitor.published = Self::view(&monitor.state, monitor.published.gate.clone());
        Ok(monitor)
    }

    pub fn restart_resume_budget(
        policy: &SloSelectedFeedbackSettingsV1,
        path: &Path,
    ) -> Result<RestartFeedbackBudget, FerrumError> {
        let maximum = policy.maximum_state_bytes.get();
        // Receipt contains only fixed-size records and numeric arrays (no
        // arbitrary strings/Value trees). 16*wire covers raw Vec growth, parsed
        // Vec/VecDeque capacities, its serialization clone, hash/output buffers
        // and the <=128-entry margin map while loading. No file is read first.
        let transient = maximum
            .checked_mul(16)
            .and_then(|n| n.checked_add(path.as_os_str().len().checked_add(32)?.checked_mul(8)?))
            .and_then(|n| n.checked_add(8192))
            .ok_or_else(|| error("resume memory bound overflow"))?;
        let disk = u64::try_from(maximum)
            .ok()
            .and_then(|n| n.checked_mul(2))
            .and_then(|n| n.checked_add(64))
            .ok_or_else(|| error("resume disk bound overflow"))?;
        Ok(RestartFeedbackBudget {
            maximum_transient_bytes: transient,
            maximum_disk_bytes: disk,
        })
    }
}
