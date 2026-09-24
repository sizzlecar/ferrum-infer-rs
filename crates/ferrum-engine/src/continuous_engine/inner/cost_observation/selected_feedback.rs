//! Explicit worker-owned margin feedback; immutable base model and TTL remain.
use super::audit::{SelectedNotCompleted, SelectedServingEvaluation};
use super::profile::EngineCostSnapshot;
use ferrum_scheduler::implementations::continuous::cost_profile::statistical_v6::ImportedWholeWaveModelV1;
use ferrum_types::{FerrumError, SloSelectedFeedbackPolicy, SloSelectedFeedbackSettingsV1};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
};
mod state;
mod store;
use state::{Binding, Comparison, Revocation, State};
#[cfg(test)]
mod tests;

pub(super) struct View {
    pub epoch: u64,
    margins: BTreeMap<[u8; 32], u64>,
    gate: Arc<AtomicU64>,
    revoked: bool,
}
impl View {
    pub fn current(&self) -> bool {
        !self.revoked && self.gate.load(Ordering::Acquire) == self.epoch
    }
    pub fn margin(&self, family: &[u8; 32]) -> u64 {
        self.margins.get(family).copied().unwrap_or(0)
    }
    pub fn activate(&self) {
        if !self.revoked {
            self.gate.store(self.epoch, Ordering::Release);
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub(super) struct FeedbackAudit {
    pub protocol: &'static str,
    pub epoch: u64,
    pub session: u64,
    pub revoked: Option<Revocation>,
    pub family_count: usize,
    pub compared: u64,
    pub corrections: u64,
    pub uncomparable_observations: u64,
    pub failed_or_partial: u64,
    pub queue_drops: u64,
    pub maximum_margin_ns: u64,
    pub persistence_failed: bool,
    pub worker_failed: bool,
}

pub(super) struct Monitor {
    policy: SloSelectedFeedbackSettingsV1,
    capacity: usize,
    state: State,
    published: Arc<View>,
    store: store::Store,
    last_ordinal: u64,
    last_drops: u64,
    persistence_failed: bool,
    worker_failed: bool,
}
impl Monitor {
    pub fn open(
        policy: &SloSelectedFeedbackPolicy,
        model: &ImportedWholeWaveModelV1,
    ) -> Result<Option<Self>, FerrumError> {
        let SloSelectedFeedbackPolicy::RetrospectiveFamilyMarginV1 { policy, storage } = policy
        else {
            return Ok(None);
        };
        let policy_bytes =
            serde_json::to_vec(policy).map_err(|e| FerrumError::config(e.to_string()))?;
        let binding = Binding {
            profile_sha256: model.file_sha256,
            source_sha256: model.source_sha256,
            fit_sha256: model.fit_parameters_sha256,
            protocol_sha256: model.protocol_sha256,
            policy_sha256: Sha256::digest(policy_bytes).into(),
        };
        let capacity = model.segment_count();
        let (store, state) = store::Store::open(storage, policy, binding, capacity)
            .map_err(|e| FerrumError::config(format!("selected feedback receipt: {e}")))?;
        let gate = Arc::new(AtomicU64::new(if state.revoked.is_none() {
            state.epoch
        } else {
            0
        }));
        let published = Self::view(&state, gate);
        Ok(Some(Self {
            policy: policy.clone(),
            capacity,
            state,
            published,
            store,
            last_ordinal: 0,
            last_drops: 0,
            persistence_failed: false,
            worker_failed: false,
        }))
    }
    fn view(state: &State, gate: Arc<AtomicU64>) -> Arc<View> {
        Arc::new(View {
            epoch: state.epoch,
            margins: state
                .families
                .iter()
                .filter(|f| f.margin_ns > 0)
                .map(|f| (f.signature, f.margin_ns))
                .collect(),
            gate,
            revoked: state.revoked.is_some(),
        })
    }
    pub fn current_view(&self) -> Arc<View> {
        self.published.clone()
    }
    pub fn observe(
        &mut self,
        ordinal: u64,
        entry: &super::CostEvidenceEntry,
        evaluation: &SelectedServingEvaluation,
        previous: Option<&EngineCostSnapshot>,
    ) {
        if self.state.revoked.is_some() {
            return;
        }
        if ordinal == 0 || ordinal <= self.last_ordinal {
            self.state.revoke(Revocation::IdentityOrClock);
            self.close_changed_epoch();
            return;
        }
        self.last_ordinal = ordinal;
        match evaluation {
            SelectedServingEvaluation::NotCompleted(
                SelectedNotCompleted::NotSubmitted | SelectedNotCompleted::Deferred,
            ) => {}
            SelectedServingEvaluation::NotCompleted(_) => self.state.failed(&self.policy),
            SelectedServingEvaluation::InvalidActual(_) => {
                let stages = match entry {
                    super::CostEvidenceEntry::Training { stages, .. } => stages.as_deref(),
                    super::CostEvidenceEntry::StagesOnly { stages, .. } => Some(stages.as_ref()),
                };
                if stages.is_some_and(|s| s.completeness == super::HostStageCompleteness::Failed) {
                    self.state.failed(&self.policy);
                } else {
                    // This is unavailable complete-boundary evidence, not an
                    // assertion that the physical call completed successfully.
                    self.state.uncomparable(&self.policy);
                }
            }
            SelectedServingEvaluation::Complete {
                prediction: Some(Ok(value)),
                ..
            } => {
                let Some(snapshot) = previous.filter(|s| s.model_version() == value.model_version)
                else {
                    self.state.revoke(Revocation::IdentityOrClock);
                    self.close_changed_epoch();
                    return;
                };
                let Some(base) = value
                    .planning_ns
                    .checked_sub(snapshot.feedback_margin(&value.family_signature))
                else {
                    self.state.revoke(Revocation::Arithmetic);
                    self.close_changed_epoch();
                    return;
                };
                self.state.compare(
                    &self.policy,
                    self.capacity,
                    Comparison {
                        family: value.family_signature,
                        base_planning_ns: base,
                        actual_ns: value.actual_ns,
                        observed_at_ns: value.observed_at_ns,
                        consumed_at_ns: value.consumed_at_ns,
                    },
                );
            }
            SelectedServingEvaluation::Complete { .. } => self.state.uncomparable(&self.policy),
        }
        self.close_changed_epoch();
    }
    pub fn pending_publication(&self) -> bool {
        self.published.revoked != self.state.revoked.is_some()
            || self
                .state
                .families
                .iter()
                .any(|f| self.published.margin(&f.signature) != f.margin_ns)
    }
    fn close_changed_epoch(&self) {
        if self.pending_publication() {
            // Close at the feedback decision, before any other queued sample
            // or raw export is processed. Persistence still runs on the worker.
            self.published.gate.store(0, Ordering::Release);
        }
    }
    /// The worker calls this after one bounded drain. Counting cannot change
    /// prediction identity; only changed parameters/revocation close the gate.
    pub fn publish(&mut self, queue_drops: Option<u64>) -> Option<Arc<View>> {
        let queue_drops = queue_drops.unwrap_or_else(|| {
            self.state.revoke(Revocation::Arithmetic);
            self.last_drops
        });
        match queue_drops.checked_sub(self.last_drops) {
            Some(delta) => self.state.dropped(&self.policy, delta),
            None => self.state.revoke(Revocation::Arithmetic),
        }
        self.last_drops = queue_drops;
        if !self.pending_publication() {
            return None;
        }
        // This invalidates ALL old whole-plan epochs before disk IO. A last
        // host guard that already passed is not retroactively revoked.
        self.published.gate.store(0, Ordering::Release);
        match self.state.epoch.checked_add(1) {
            Some(epoch) => self.state.epoch = epoch,
            None => self.state.revoke(Revocation::Arithmetic),
        }
        if let Err(error) = self.store.persist(&self.state) {
            self.persistence_failed = true;
            self.state.revoke(Revocation::Persistence);
            tracing::error!(%error, "selected feedback persistence failed; artifact revoked");
        }
        self.published = Self::view(&self.state, self.published.gate.clone());
        Some(self.published.clone())
    }
    pub fn finish(&mut self) {
        // A cleanly released receipt must not leave this process authorized
        // while a replacement process resumes the persisted artifact.
        self.published.gate.store(0, Ordering::Release);
        if let Err(error) = self.store.finish(&self.state) {
            self.persistence_failed = true;
            self.state.revoke(Revocation::Persistence);
            self.published.gate.store(0, Ordering::Release);
            tracing::error!(%error, "selected feedback shutdown not durable; restart remains blocked");
        }
    }
    pub fn check_finished(&self) -> Result<(), FerrumError> {
        if self.worker_failed {
            Err(FerrumError::backend(
                "selected feedback worker stopped without a clean durable shutdown",
            ))
        } else if self.persistence_failed {
            Err(FerrumError::backend("selected feedback persistence failed"))
        } else {
            Ok(())
        }
    }
    pub fn worker_stopped_unclean(&mut self) {
        // No IO while unwinding. The already durable session marker blocks
        // restart; current in-process plans immediately lose their permission.
        self.published.gate.store(0, Ordering::Release);
        self.worker_failed = true;
        self.state.revoke(Revocation::WorkerStopped);
    }
    pub fn audit(&self) -> FeedbackAudit {
        FeedbackAudit {
            protocol: "retrospective_family_margin_v1; separate from immutable profile6 fit/residual; no q99 guarantee",
            epoch: self.state.epoch, session: self.state.session, revoked: self.state.revoked,
            family_count: self.state.families.len(), compared: self.state.compared,
            corrections: self.state.corrections,
            uncomparable_observations: self.state.uncomparable_observations,
            failed_or_partial: self.state.failed_or_partial,
            queue_drops: self.state.queue_drops,
            maximum_margin_ns: self.state.families.iter().map(|f| f.margin_ns).max().unwrap_or(0),
            persistence_failed: self.persistence_failed,
            worker_failed: self.worker_failed,
        }
    }
}
