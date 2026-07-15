use serde::Serialize;
use std::collections::BTreeSet;

use super::{
    canonical_fingerprint, invalid_event, validate_sha256, ActiveSequenceAbortDisposition,
    ActiveSequenceAbortReceipt, ActiveSequenceCompletionReceipt, ActiveSequencePermit,
    DeviceRuntime, LogicalAdmissionCoordinatorId, LogicalBackingSliceEvidence, RequestIdentity,
    ResourceLeaseEntry, ResourceLeaseState, ResourcePoolId, ResourceTransactionIdentity, RunId,
    SequenceAuthorityId, SequenceSession, SequenceSessionEpoch, SequenceSessionFingerprint,
    SequenceSessionLiveWitness, SequenceSessionTerminalDisposition, SequenceSessionTerminalReceipt,
    TrustedPlanRuntimeEvidence, VNextError,
};

#[derive(Clone, Serialize)]
enum TrustedActiveSequenceAuthority {
    StreamActivation,
    SequenceSession {
        fingerprint: SequenceSessionFingerprint,
        #[serde(skip)]
        live_witness: SequenceSessionLiveWitness,
    },
}

#[derive(Clone, Serialize)]
pub struct TrustedActiveSequenceBinding {
    plan: TrustedPlanRuntimeEvidence,
    coordinator_id: LogicalAdmissionCoordinatorId,
    sequence_authority: SequenceAuthorityId,
    run_id: RunId,
    request_id: RequestIdentity,
    activation_epoch: u64,
    authority: TrustedActiveSequenceAuthority,
    runtime_implementation_fingerprint: String,
    static_entries: Vec<ResourceLeaseEntry>,
    #[serde(rename = "backing_slices")]
    legacy_backing_slices: Option<Vec<LogicalBackingSliceEvidence>>,
    #[serde(skip)]
    fingerprint: String,
}

impl TrustedActiveSequenceBinding {
    pub fn from_permit<R>(permit: &ActiveSequencePermit<'_, '_, R>) -> Result<Self, VNextError>
    where
        R: DeviceRuntime,
    {
        let resources = permit.resources();
        let plan = resources.plan_evidence();
        let mut static_entries = resources
            .static_provisioning()
            .map(|lease| lease.plan_static_entries().cloned().collect::<Vec<_>>())
            .unwrap_or_default();
        static_entries.sort_by(|left, right| left.resource_id().cmp(right.resource_id()));
        let mut resources = BTreeSet::new();
        if permit.activation_epoch() == 0
            || plan.coordinator_id() != permit.coordinator_id()
            || plan.runtime_implementation_fingerprint()
                != permit.runtime_implementation_fingerprint()
            || static_entries.iter().any(|entry| {
                entry.state() != ResourceLeaseState::Active
                    || !resources.insert(entry.resource_id().clone())
            })
        {
            return Err(invalid_event(
                "active permit pool, slot, epoch, admission, or lease entries are invalid",
            ));
        }
        validate_sha256(
            permit.runtime_implementation_fingerprint(),
            "runtime implementation fingerprint",
        )?;
        let mut binding = Self {
            plan,
            coordinator_id: permit.coordinator_id(),
            sequence_authority: permit.sequence_authority(),
            run_id: permit.run_id().clone(),
            request_id: permit.request_id().clone(),
            activation_epoch: permit.activation_epoch(),
            authority: TrustedActiveSequenceAuthority::StreamActivation,
            runtime_implementation_fingerprint: permit
                .runtime_implementation_fingerprint()
                .to_owned(),
            static_entries,
            legacy_backing_slices: Some(
                permit
                    .backing_slices()
                    .iter()
                    .map(|slice| slice.evidence().clone())
                    .collect(),
            ),
            fingerprint: String::new(),
        };
        binding.fingerprint = canonical_fingerprint(&binding);
        Ok(binding)
    }

    pub fn from_session<R>(session: &SequenceSession<R>) -> Result<Self, VNextError>
    where
        R: DeviceRuntime,
    {
        session.ensure_open_identity()?;
        let resources = session.resources();
        let plan = resources.plan_evidence();
        let runtime_fingerprint = plan.runtime_implementation_fingerprint().to_owned();
        validate_sha256(&runtime_fingerprint, "runtime implementation fingerprint")?;
        let mut static_entries = resources
            .static_provisioning()
            .map(|lease| lease.plan_static_entries().cloned().collect::<Vec<_>>())
            .unwrap_or_default();
        static_entries.sort_by(|left, right| left.resource_id().cmp(right.resource_id()));
        let mut resource_ids = BTreeSet::new();
        if plan.coordinator_id() != resources.coordinator_id()
            || static_entries.iter().any(|entry| {
                entry.state() != ResourceLeaseState::Active
                    || !resource_ids.insert(entry.resource_id().clone())
            })
        {
            return Err(invalid_event(
                "active session plan, coordinator, or lease entries are invalid",
            ));
        }
        let mut binding = Self {
            plan,
            coordinator_id: resources.coordinator_id(),
            sequence_authority: resources.sequence_authority(),
            run_id: resources.run_id().clone(),
            request_id: resources.request_id().clone(),
            activation_epoch: session.epoch().get(),
            authority: TrustedActiveSequenceAuthority::SequenceSession {
                fingerprint: session.fingerprint().clone(),
                live_witness: session.live_witness()?,
            },
            runtime_implementation_fingerprint: runtime_fingerprint,
            static_entries,
            // A session identity is stable across dynamic backing generations.
            // The exact physical authority belongs to each captured Step.
            legacy_backing_slices: None,
            fingerprint: String::new(),
        };
        binding.fingerprint = canonical_fingerprint(&binding);
        Ok(binding)
    }

    pub fn plan(&self) -> &TrustedPlanRuntimeEvidence {
        &self.plan
    }

    pub const fn coordinator_id(&self) -> LogicalAdmissionCoordinatorId {
        self.coordinator_id
    }

    pub const fn sequence_authority(&self) -> SequenceAuthorityId {
        self.sequence_authority
    }

    pub fn static_pool_id(&self) -> Option<ResourcePoolId> {
        self.plan.static_pool_identity().map(|pool| pool.pool_id())
    }

    pub fn static_pool_identity_fingerprint(&self) -> Option<String> {
        self.plan.static_pool_identity().map(canonical_fingerprint)
    }

    pub fn static_provisioning_identity(&self) -> Option<&ResourceTransactionIdentity> {
        self.plan.static_provisioning_identity()
    }

    pub fn run_id(&self) -> &RunId {
        &self.run_id
    }

    pub fn request_id(&self) -> &RequestIdentity {
        &self.request_id
    }

    pub const fn activation_epoch(&self) -> u64 {
        self.activation_epoch
    }

    pub(crate) fn matches_sequence_session(
        &self,
        epoch: SequenceSessionEpoch,
        fingerprint: &SequenceSessionFingerprint,
    ) -> bool {
        self.activation_epoch == epoch.get()
            && matches!(
                &self.authority,
                TrustedActiveSequenceAuthority::SequenceSession {
                    fingerprint: bound_fingerprint,
                    ..
                } if bound_fingerprint == fingerprint
            )
    }

    fn is_stream_activation(&self) -> bool {
        matches!(
            &self.authority,
            TrustedActiveSequenceAuthority::StreamActivation
        )
    }

    pub(crate) fn ensure_open_for_emission(&self) -> Result<(), VNextError> {
        match &self.authority {
            TrustedActiveSequenceAuthority::StreamActivation => Err(invalid_event(
                "node execution emission requires typed sequence-session authority",
            )),
            TrustedActiveSequenceAuthority::SequenceSession { live_witness, .. } => {
                live_witness.ensure_open()
            }
        }
    }

    pub(super) fn ensure_live_for_emission(&self) -> Result<(), VNextError> {
        match &self.authority {
            TrustedActiveSequenceAuthority::StreamActivation => Err(invalid_event(
                "operation progress emission requires typed sequence-session authority",
            )),
            TrustedActiveSequenceAuthority::SequenceSession { live_witness, .. } => {
                live_witness.ensure_live()
            }
        }
    }

    pub(super) fn matches_abort_disposition(
        &self,
        disposition: ActiveSequenceAbortDisposition,
    ) -> bool {
        matches!(
            (&self.authority, disposition),
            (
                TrustedActiveSequenceAuthority::StreamActivation,
                ActiveSequenceAbortDisposition::SynchronizedAndPoisoned,
            ) | (
                TrustedActiveSequenceAuthority::SequenceSession { .. },
                ActiveSequenceAbortDisposition::SequenceSessionTerminalized,
            )
        )
    }

    pub fn runtime_implementation_fingerprint(&self) -> &str {
        &self.runtime_implementation_fingerprint
    }

    pub fn static_entries(&self) -> &[ResourceLeaseEntry] {
        &self.static_entries
    }

    pub fn legacy_backing_slices(&self) -> Option<&[LogicalBackingSliceEvidence]> {
        self.legacy_backing_slices.as_deref()
    }

    pub fn backing_slices(&self) -> &[LogicalBackingSliceEvidence] {
        self.legacy_backing_slices.as_deref().unwrap_or_default()
    }

    pub fn fingerprint(&self) -> &str {
        &self.fingerprint
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct TrustedCompletedSequenceBinding {
    plan: TrustedPlanRuntimeEvidence,
    coordinator_id: LogicalAdmissionCoordinatorId,
    sequence_authority: SequenceAuthorityId,
    run_id: RunId,
    request_id: RequestIdentity,
    activation_epoch: u64,
    runtime_implementation_fingerprint: String,
    active_sequence_fingerprint: String,
    #[serde(skip)]
    fingerprint: String,
}

impl TrustedCompletedSequenceBinding {
    pub fn from_receipt(
        receipt: &ActiveSequenceCompletionReceipt,
        active: &TrustedActiveSequenceBinding,
    ) -> Result<Self, VNextError> {
        if !active.is_stream_activation()
            || receipt.plan() != active.plan()
            || receipt.sequence_authority() != active.sequence_authority()
            || receipt.run_id() != active.run_id()
            || receipt.request_id() != active.request_id()
            || receipt.activation_epoch() != active.activation_epoch()
            || receipt.runtime_implementation_fingerprint()
                != active.runtime_implementation_fingerprint()
        {
            return Err(invalid_event(
                "sequence completion receipt differs from the active pool, request, slot, epoch, or runtime",
            ));
        }
        let mut binding = Self {
            plan: receipt.plan().clone(),
            coordinator_id: receipt.plan().coordinator_id(),
            sequence_authority: receipt.sequence_authority(),
            run_id: receipt.run_id().clone(),
            request_id: receipt.request_id().clone(),
            activation_epoch: receipt.activation_epoch(),
            runtime_implementation_fingerprint: receipt
                .runtime_implementation_fingerprint()
                .to_owned(),
            active_sequence_fingerprint: active.fingerprint().to_owned(),
            fingerprint: String::new(),
        };
        binding.fingerprint = canonical_fingerprint(&binding);
        Ok(binding)
    }

    pub fn from_session_receipt(
        receipt: &SequenceSessionTerminalReceipt,
        active: &TrustedActiveSequenceBinding,
    ) -> Result<Self, VNextError> {
        if receipt.disposition() != SequenceSessionTerminalDisposition::Completed
            || receipt.retired_frames() == 0
            || !active.matches_sequence_session(receipt.epoch(), receipt.fingerprint())
        {
            return Err(invalid_event(
                "sequence session completion differs from the exact active session or has no retired frame",
            ));
        }
        let mut binding = Self {
            plan: active.plan().clone(),
            coordinator_id: active.coordinator_id(),
            sequence_authority: active.sequence_authority(),
            run_id: active.run_id().clone(),
            request_id: active.request_id().clone(),
            activation_epoch: active.activation_epoch(),
            runtime_implementation_fingerprint: active
                .runtime_implementation_fingerprint()
                .to_owned(),
            active_sequence_fingerprint: active.fingerprint().to_owned(),
            fingerprint: String::new(),
        };
        binding.fingerprint = canonical_fingerprint(&binding);
        Ok(binding)
    }

    pub fn plan(&self) -> &TrustedPlanRuntimeEvidence {
        &self.plan
    }

    pub const fn coordinator_id(&self) -> LogicalAdmissionCoordinatorId {
        self.coordinator_id
    }

    pub const fn sequence_authority(&self) -> SequenceAuthorityId {
        self.sequence_authority
    }

    pub fn run_id(&self) -> &RunId {
        &self.run_id
    }

    pub fn request_id(&self) -> &RequestIdentity {
        &self.request_id
    }

    pub const fn activation_epoch(&self) -> u64 {
        self.activation_epoch
    }

    pub fn runtime_implementation_fingerprint(&self) -> &str {
        &self.runtime_implementation_fingerprint
    }

    pub fn active_sequence_fingerprint(&self) -> &str {
        &self.active_sequence_fingerprint
    }

    pub fn fingerprint(&self) -> &str {
        &self.fingerprint
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct TrustedAbortedSequenceBinding {
    plan: TrustedPlanRuntimeEvidence,
    coordinator_id: LogicalAdmissionCoordinatorId,
    sequence_authority: SequenceAuthorityId,
    run_id: RunId,
    request_id: RequestIdentity,
    activation_epoch: u64,
    runtime_implementation_fingerprint: String,
    active_sequence_fingerprint: String,
    disposition: ActiveSequenceAbortDisposition,
    #[serde(skip)]
    fingerprint: String,
}

impl TrustedAbortedSequenceBinding {
    pub fn from_receipt(
        receipt: &ActiveSequenceAbortReceipt,
        active: &TrustedActiveSequenceBinding,
    ) -> Result<Self, VNextError> {
        if !active.is_stream_activation()
            || receipt.disposition() != ActiveSequenceAbortDisposition::SynchronizedAndPoisoned
            || receipt.plan() != active.plan()
            || receipt.sequence_authority() != active.sequence_authority()
            || receipt.run_id() != active.run_id()
            || receipt.request_id() != active.request_id()
            || receipt.activation_epoch() != active.activation_epoch()
            || receipt.runtime_implementation_fingerprint()
                != active.runtime_implementation_fingerprint()
        {
            return Err(invalid_event(
                "sequence abort receipt differs from the active pool, request, slot, epoch, runtime, or poison disposition",
            ));
        }
        let mut binding = Self {
            plan: receipt.plan().clone(),
            coordinator_id: receipt.plan().coordinator_id(),
            sequence_authority: receipt.sequence_authority(),
            run_id: receipt.run_id().clone(),
            request_id: receipt.request_id().clone(),
            activation_epoch: receipt.activation_epoch(),
            runtime_implementation_fingerprint: receipt
                .runtime_implementation_fingerprint()
                .to_owned(),
            active_sequence_fingerprint: active.fingerprint().to_owned(),
            disposition: receipt.disposition(),
            fingerprint: String::new(),
        };
        binding.fingerprint = canonical_fingerprint(&binding);
        Ok(binding)
    }

    pub fn from_session_receipt(
        receipt: &SequenceSessionTerminalReceipt,
        active: &TrustedActiveSequenceBinding,
    ) -> Result<Self, VNextError> {
        if receipt.disposition() != SequenceSessionTerminalDisposition::Aborted
            || !active.matches_sequence_session(receipt.epoch(), receipt.fingerprint())
        {
            return Err(invalid_event(
                "sequence session abort differs from the exact active session",
            ));
        }
        let mut binding = Self {
            plan: active.plan().clone(),
            coordinator_id: active.coordinator_id(),
            sequence_authority: active.sequence_authority(),
            run_id: active.run_id().clone(),
            request_id: active.request_id().clone(),
            activation_epoch: active.activation_epoch(),
            runtime_implementation_fingerprint: active
                .runtime_implementation_fingerprint()
                .to_owned(),
            active_sequence_fingerprint: active.fingerprint().to_owned(),
            disposition: ActiveSequenceAbortDisposition::SequenceSessionTerminalized,
            fingerprint: String::new(),
        };
        binding.fingerprint = canonical_fingerprint(&binding);
        Ok(binding)
    }

    pub fn plan(&self) -> &TrustedPlanRuntimeEvidence {
        &self.plan
    }

    pub const fn coordinator_id(&self) -> LogicalAdmissionCoordinatorId {
        self.coordinator_id
    }

    pub const fn sequence_authority(&self) -> SequenceAuthorityId {
        self.sequence_authority
    }

    pub fn run_id(&self) -> &RunId {
        &self.run_id
    }

    pub fn request_id(&self) -> &RequestIdentity {
        &self.request_id
    }

    pub const fn activation_epoch(&self) -> u64 {
        self.activation_epoch
    }

    pub fn runtime_implementation_fingerprint(&self) -> &str {
        &self.runtime_implementation_fingerprint
    }

    pub fn active_sequence_fingerprint(&self) -> &str {
        &self.active_sequence_fingerprint
    }

    pub const fn disposition(&self) -> ActiveSequenceAbortDisposition {
        self.disposition
    }

    pub fn fingerprint(&self) -> &str {
        &self.fingerprint
    }
}
