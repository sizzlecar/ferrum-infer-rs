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
mod restart;
mod state;
mod store;
pub(super) use restart::{RestartFeedbackAllowance, RestartFeedbackBudget, RestartFeedbackReceipt};
use state::State;
pub(super) use state::{Binding, Comparison, Revocation};
#[cfg(test)]
mod tests;

pub(super) struct View {
    pub epoch: u64,
    margins: BTreeMap<[u8; 32], u64>,
    gate: Arc<AtomicU64>,
    revoked: bool,
}
/// Dynamic validity only: queued observations do not retain model margins.
pub(super) struct Validity {
    epoch: u64,
    gate: Arc<AtomicU64>,
    revoked: bool,
}
impl Validity {
    pub fn current(&self) -> bool {
        !self.revoked && self.gate.load(Ordering::Acquire) == self.epoch
    }
}
impl View {
    pub fn validity(&self) -> Validity {
        Validity {
            epoch: self.epoch,
            gate: self.gate.clone(),
            revoked: self.revoked,
        }
    }
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
    /// Validated actual owners outside the automatic catalog, including an
    /// acknowledged predecessor retained by a verified restart handoff.
    /// These are Unknown queries and never enter an error/margin population.
    pub outside_catalog_observations: u64,
    /// Complete actual inputs outside the immutable model's numerical support.
    /// Still Unknown; excluded from error/margin updates and counted separately.
    pub outside_support_observations: u64,
    /// Live-consumer-validated completed routes outside the automatic route
    /// declaration. Unknown; never an error/margin observation.
    pub outside_route_observations: u64,
    /// Privately proven V2 no-submit attempts, with original clock/lag checks.
    pub no_submission_observations: u64,
    /// Complete private calibration preparation, outside ordinary model drift.
    pub outside_preparation_observations: u64,
    pub failed_or_partial: u64,
    pub queue_drops: u64,
    pub maximum_margin_ns: u64,
    pub persistence_failed: bool,
    pub worker_failed: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub previous_process: Option<state::RestartProgressV1>,
    /// Complete declared inventory for structured feedback, including owners
    /// with no comparable observations; selected feedback retains seen scopes.
    pub scopes: Vec<ScopeAudit>,
}

#[derive(Debug, Clone, Serialize)]
pub(super) struct ScopeAudit {
    pub signature: [u8; 32],
    pub margin_ns: u64,
    pub compared: u64,
    pub underestimates: u64,
    pub maximum_base_excess_ns: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum FeedbackKind {
    Selected,
    StructuredV2,
}

pub(super) enum FeedbackObservation {
    NotSubmitted,
    FailedOrPartial,
    Uncomparable,
    /// Complete original actual evidence selected this exact qualified child,
    /// whose original TTL expired at consumption. This is lifecycle work, not
    /// a prediction error or support exclusion. FIFO/identity/lag still apply.
    ExpiredModel {
        model_epoch: u64,
        domain: [u8; 32],
        observed_at_ns: u64,
        consumed_at_ns: u64,
    },
    OutsideCatalog {
        observed_at_ns: u64,
        consumed_at_ns: u64,
    },
    OutsideSupport {
        observed_at_ns: u64,
        consumed_at_ns: u64,
    },
    /// Produced only by automatic feedback after live.consume validates an
    /// OutsideDeclaredRoute receipt. Ordinary missing actual evidence is not
    /// sufficient to construct this classification.
    OutsideRoute {
        observed_at_ns: u64,
        consumed_at_ns: u64,
    },
    ProvenNoSubmission {
        observed_at_ns: u64,
        consumed_at_ns: u64,
    },
    OutsidePreparation {
        observed_at_ns: u64,
        consumed_at_ns: u64,
    },
    InvalidIdentity,
    Compared(Comparison),
}

enum OutsideKind {
    Catalog,
    Support,
    Route,
    NoSubmission,
    Preparation,
}

pub(super) struct Monitor {
    kind: FeedbackKind,
    runtime_epoch: Option<Arc<super::structured_epoch::Epoch>>,
    declared_scopes: Option<Arc<[[u8; 32]]>>,
    policy: SloSelectedFeedbackSettingsV1,
    capacity: usize,
    state: State,
    published: Arc<View>,
    store: store::Store,
    last_ordinal: u64,
    last_drops: u64,
    outside_catalog_observations: u64,
    outside_support_observations: u64,
    outside_route_observations: u64,
    no_submission_observations: u64,
    outside_preparation_observations: u64,
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
        Self::open_bound(
            policy,
            storage,
            binding,
            capacity,
            FeedbackKind::Selected,
            None,
        )
        .map(Some)
    }
    pub fn open_bound(
        policy: &SloSelectedFeedbackSettingsV1,
        storage: &ferrum_types::SloSelectedFeedbackStorageV1,
        binding: Binding,
        capacity: usize,
        kind: FeedbackKind,
        declared_scopes: Option<Arc<[[u8; 32]]>>,
    ) -> Result<Self, FerrumError> {
        let (store, state) = store::Store::open(storage, policy, binding, capacity)
            .map_err(|e| FerrumError::config(format!("cost feedback receipt: {e}")))?;
        if declared_scopes.as_ref().is_some_and(|scopes| {
            scopes.len() != capacity
                || scopes.windows(2).any(|p| p[0] >= p[1])
                || state
                    .families
                    .iter()
                    .any(|f| scopes.binary_search(&f.signature).is_err())
        }) {
            return Err(FerrumError::config(
                "feedback receipt has an undeclared owner",
            ));
        }
        let gate = Arc::new(AtomicU64::new(if state.revoked.is_none() {
            state.epoch
        } else {
            0
        }));
        let published = Self::view(&state, gate);
        Ok(Self {
            kind,
            runtime_epoch: None,
            declared_scopes,
            policy: policy.clone(),
            capacity,
            state,
            published,
            store,
            last_ordinal: 0,
            last_drops: 0,
            outside_catalog_observations: 0,
            outside_support_observations: 0,
            outside_route_observations: 0,
            no_submission_observations: 0,
            outside_preparation_observations: 0,
            persistence_failed: false,
            worker_failed: false,
        })
    }
    pub fn attach_structured_epoch(&mut self, epoch: Arc<super::structured_epoch::Epoch>) {
        debug_assert_eq!(self.kind, FeedbackKind::StructuredV2);
        self.published = Self::view(&self.state, epoch.gate.clone());
        self.runtime_epoch = Some(epoch);
    }
    /// Bind a fresh process-local monitor to the first qualified live base.
    /// The gate remains closed until the caller atomically installs the model.
    pub fn initialize_structured_epoch(
        &mut self,
        epoch: Arc<super::structured_epoch::Epoch>,
        version: u64,
    ) -> Result<(), FerrumError> {
        if self.kind != FeedbackKind::StructuredV2
            || self.runtime_epoch.is_some()
            || version == 0
            || self.state.revoked.is_some()
            || !self.state.families.is_empty()
            || self.state.compared != 0
            || self.last_ordinal != 0
        {
            return Err(FerrumError::config(
                "initial feedback requires a fresh structured monitor",
            ));
        }
        self.state.epoch = version;
        self.store
            .persist(&self.state)
            .map_err(|e| FerrumError::config(format!("initialize live feedback: {e}")))?;
        self.attach_structured_epoch(epoch);
        Ok(())
    }
    /// Worker-only durable rebinding. No old query loses authority until the
    /// caller swaps the new immutable catalog and activates the reserved epoch.
    pub fn rebind_structured(
        &mut self,
        binding: Binding,
        scopes: Arc<[[u8; 32]]>,
        retained: &[[u8; 32]],
        epoch: u64,
        validate_after_persist: impl FnOnce() -> Result<(), FerrumError>,
    ) -> Result<Arc<View>, FerrumError> {
        if self.kind != FeedbackKind::StructuredV2
            || self.runtime_epoch.is_none()
            || scopes.is_empty()
            || scopes.windows(2).any(|v| v[0] >= v[1])
            || epoch <= self.state.epoch
            || (!retained.is_empty()
                && (self.state.revoked.is_some() || self.persistence_failed || self.worker_failed))
        {
            return Err(FerrumError::config(
                "invalid structured feedback replacement",
            ));
        }
        let mut next = if retained.is_empty() {
            State::new(binding.clone())
        } else {
            self.state.clone()
        };
        next.binding = binding;
        next.epoch = epoch;
        next.session = self.state.session;
        next.families
            .retain(|family| retained.contains(&family.signature));
        if !next.validate(&next.binding, &self.policy, scopes.len()) {
            return Err(FerrumError::config("replacement feedback capacity differs"));
        }
        self.store
            .persist(&next)
            .map_err(|e| FerrumError::backend(format!("persist replacement feedback: {e}")))?;
        // Preserve the old in-memory state/gate if the original source aged
        // out during IO. A crash cannot load an incompatible receipt silently.
        validate_after_persist()?;
        self.state = next;
        self.capacity = scopes.len();
        self.declared_scopes = Some(scopes);
        self.published = Self::view(&self.state, self.published.gate.clone());
        self.persistence_failed = false;
        self.worker_failed = false;
        Ok(self.published.clone())
    }
    pub fn policy(&self) -> &SloSelectedFeedbackSettingsV1 {
        &self.policy
    }
    pub fn kind(&self) -> FeedbackKind {
        self.kind
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
    /// Read the scalar terminal state without cloning the full audit inventory.
    pub(super) fn revocation(&self) -> Option<Revocation> {
        self.state.revoked
    }
    /// Shared bounded policy after the predictor-specific actual evidence
    /// adapter. No public numerical observation can bypass that adapter.
    pub fn observe_classified(&mut self, ordinal: u64, observation: FeedbackObservation) {
        if self.state.revoked.is_some() {
            return;
        }
        if ordinal == 0 || ordinal <= self.last_ordinal {
            self.state.revoke(Revocation::IdentityOrClock);
            self.close_changed_epoch();
            return;
        }
        self.last_ordinal = ordinal;
        match observation {
            FeedbackObservation::NotSubmitted => {}
            FeedbackObservation::FailedOrPartial => self.state.failed(&self.policy),
            FeedbackObservation::Uncomparable => self.state.uncomparable(&self.policy),
            FeedbackObservation::ExpiredModel {
                model_epoch,
                domain,
                observed_at_ns,
                consumed_at_ns,
            } => {
                if self.kind != FeedbackKind::StructuredV2
                    || model_epoch != self.state.epoch
                    || self
                        .declared_scopes
                        .as_ref()
                        .is_none_or(|scopes| scopes.binary_search(&domain).is_err())
                {
                    self.state.revoke(Revocation::IdentityOrClock);
                } else {
                    match consumed_at_ns.checked_sub(observed_at_ns) {
                        None => self.state.revoke(Revocation::IdentityOrClock),
                        Some(lag) if lag > self.policy.maximum_consumption_lag_ns.get() => {
                            self.state.revoke(Revocation::ObservationLag);
                        }
                        Some(_) => {}
                    }
                }
            }
            FeedbackObservation::OutsideCatalog {
                observed_at_ns,
                consumed_at_ns,
            } => self.observe_outside_domain(observed_at_ns, consumed_at_ns, OutsideKind::Catalog),
            FeedbackObservation::OutsideSupport {
                observed_at_ns,
                consumed_at_ns,
            } => self.observe_outside_domain(observed_at_ns, consumed_at_ns, OutsideKind::Support),
            FeedbackObservation::OutsideRoute {
                observed_at_ns,
                consumed_at_ns,
            } => self.observe_outside_domain(observed_at_ns, consumed_at_ns, OutsideKind::Route),
            FeedbackObservation::ProvenNoSubmission {
                observed_at_ns,
                consumed_at_ns,
            } => self.observe_outside_domain(
                observed_at_ns,
                consumed_at_ns,
                OutsideKind::NoSubmission,
            ),
            FeedbackObservation::InvalidIdentity => self.state.revoke(Revocation::IdentityOrClock),
            FeedbackObservation::OutsidePreparation {
                observed_at_ns,
                consumed_at_ns,
            } => self.observe_outside_domain(
                observed_at_ns,
                consumed_at_ns,
                OutsideKind::Preparation,
            ),
            FeedbackObservation::Compared(value) => {
                if self
                    .declared_scopes
                    .as_ref()
                    .is_some_and(|scopes| scopes.binary_search(&value.family).is_err())
                {
                    self.state.revoke(Revocation::IdentityOrClock);
                } else {
                    self.state.compare(&self.policy, self.capacity, value);
                }
            }
        }
        self.close_changed_epoch();
    }
    fn observe_outside_domain(
        &mut self,
        observed_at_ns: u64,
        consumed_at_ns: u64,
        kind: OutsideKind,
    ) {
        match consumed_at_ns.checked_sub(observed_at_ns) {
            None => self.state.revoke(Revocation::IdentityOrClock),
            Some(lag) if lag > self.policy.maximum_consumption_lag_ns.get() => {
                self.state.revoke(Revocation::ObservationLag);
            }
            Some(_) => {
                let counter = match kind {
                    OutsideKind::Catalog => &mut self.outside_catalog_observations,
                    OutsideKind::Support => &mut self.outside_support_observations,
                    OutsideKind::Route => &mut self.outside_route_observations,
                    OutsideKind::NoSubmission => &mut self.no_submission_observations,
                    OutsideKind::Preparation => &mut self.outside_preparation_observations,
                };
                match counter.checked_add(1) {
                    Some(count) => *counter = count,
                    None => self.state.revoke(Revocation::Arithmetic),
                }
            }
        }
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
        let next_epoch = self.runtime_epoch.as_ref().map_or_else(
            || self.state.epoch.checked_add(1),
            |clock| clock.reserve().ok(),
        );
        match next_epoch {
            Some(epoch) => self.state.epoch = epoch,
            None => self.state.revoke(Revocation::Arithmetic),
        }
        if let Err(error) = self.store.persist(&self.state) {
            self.persistence_failed = true;
            self.state.revoke(Revocation::Persistence);
            tracing::error!(%error, kind=?self.kind, "cost feedback persistence failed; artifact revoked");
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
            tracing::error!(%error, kind=?self.kind, "cost feedback shutdown not durable; restart remains blocked");
        }
    }
    pub fn check_finished(&self) -> Result<(), FerrumError> {
        if self.worker_failed {
            Err(FerrumError::backend(
                "cost feedback worker stopped without a clean durable shutdown",
            ))
        } else if self.persistence_failed {
            Err(FerrumError::backend("cost feedback persistence failed"))
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
        let observed: BTreeMap<_, _> = self
            .state
            .families
            .iter()
            .map(|f| (f.signature, f))
            .collect();
        let scopes = self
            .declared_scopes
            .as_ref()
            .map_or_else(
                || {
                    self.state
                        .families
                        .iter()
                        .map(|f| f.signature)
                        .collect::<Vec<_>>()
                },
                |scopes| scopes.to_vec(),
            )
            .into_iter()
            .map(|signature| {
                let family = observed.get(&signature);
                ScopeAudit {
                    signature,
                    margin_ns: family.map_or(0, |f| f.margin_ns),
                    compared: family.map_or(0, |f| f.compared),
                    underestimates: family.map_or(0, |f| f.underestimates),
                    maximum_base_excess_ns: family.map_or(0, |f| f.maximum_base_excess_ns),
                }
            })
            .collect();
        FeedbackAudit {
            protocol: match self.kind {
                FeedbackKind::Selected => "retrospective_family_margin_v1; separate from immutable selected fit/residual; no q99 guarantee",
                FeedbackKind::StructuredV2 => "retrospective_owner_margin_v1; complete actual host-settled waves against immutable pre-drain qualified profile10 catalog; original support and TTL; not pre-submit prediction or q99 guarantee",
            },
            epoch: self.state.epoch, session: self.state.session, revoked: self.state.revoked,
            family_count: self.state.families.len(), compared: self.state.compared,
            corrections: self.state.corrections,
            uncomparable_observations: self.state.uncomparable_observations,
            outside_catalog_observations: self.outside_catalog_observations,
            outside_support_observations: self.outside_support_observations,
            outside_route_observations: self.outside_route_observations,
            no_submission_observations: self.no_submission_observations,
            outside_preparation_observations: self.outside_preparation_observations,
            failed_or_partial: self.state.failed_or_partial,
            queue_drops: self.state.queue_drops,
            maximum_margin_ns: self.state.families.iter().map(|f| f.margin_ns).max().unwrap_or(0),
            persistence_failed: self.persistence_failed,
            worker_failed: self.worker_failed,
            previous_process: self.state.restart_progress,
            scopes,
        }
    }
}
