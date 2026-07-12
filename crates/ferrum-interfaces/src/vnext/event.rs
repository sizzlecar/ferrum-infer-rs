use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;

use super::{
    ActiveSequenceAbortDisposition, ActiveSequenceAbortReceipt, ActiveSequenceCompletionReceipt,
    ActiveSequencePermit, CompletionDrainReceipt, CompletionQuarantineReceipt, CompletionSlotId,
    ContractVersion, DeviceId, DeviceRuntime, ExecutionPlan, FailureDomain, FailureEnvelope,
    FailureEnvelopeWire, LogicalAdmissionCoordinatorId, LogicalBackingSliceEvidence, NodeId,
    OperationCompletionReceipt, OperationId, OperationParticipantCompletionReceipt, PlanHash,
    PlanId, PlanRuntimeCloseReceipt, PlanRuntimeQuarantineReceipt, ProviderId, RequestIdentity,
    ResolvedModelPlan, ResourceFailureId, ResourceFailureReceipt, ResourceId, ResourceLeaseEntry,
    ResourceLeaseState, ResourceLeaseTransitionReceipt, ResourceLeaseValidationContext,
    ResourceLedgerEntrySnapshot, ResourceLedgerSnapshot, ResourcePoolId,
    ResourceTransactionIdentity, ResourceTransactionState, ResourceTransitionReceipt,
    ResourceTransitionValidationContext, RunId, SequenceAuthorityId, SequenceSession,
    SequenceSessionEpoch, SequenceSessionFingerprint, SequenceSessionLiveWitness,
    SequenceSessionTerminalDisposition, SequenceSessionTerminalReceipt, SpanId,
    StaticProvisioningBinding, SubmittedOperationParticipantReceipt, SubmittedOperationReceipt,
    TransactionId, TrustedPlanRuntimeEvidence, UnvalidatedFailureEnvelope,
    UnvalidatedResourceLeaseTransitionReceipt, UnvalidatedResourceLeaseTransitionReceiptWire,
    UnvalidatedResourceTransitionReceipt, UnvalidatedResourceTransitionReceiptWire, VNextError,
};

pub const EXECUTION_IDENTITY_VERSION: ContractVersion = ContractVersion::new(3, 0);
pub const MAX_EXECUTION_EVENT_WIRE_BYTES: usize = 1024 * 1024;
pub const MAX_REPLAY_IDENTITY_WIRE_BYTES: usize = 1024 * 1024;
pub const MAX_RESOURCE_POOL_EVENT_WIRE_BYTES: usize = 16 * 1024 * 1024;

fn invalid_event(reason: impl Into<String>) -> VNextError {
    VNextError::InvalidExecutionPlan {
        reason: reason.into(),
    }
}

fn canonical_fingerprint(value: &impl Serialize) -> String {
    format!(
        "{:x}",
        Sha256::digest(serde_json::to_vec(value).expect("trusted event evidence must serialize"))
    )
}

fn sha256_bytes(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn validate_sha256(value: &str, label: &str) -> Result<(), VNextError> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(invalid_event(format!(
            "{label} must be a canonical lowercase SHA256"
        )));
    }
    Ok(())
}

macro_rules! nonzero_execution_id {
    ($name:ident, $label:literal) => {
        #[derive(
            Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
        )]
        #[serde(try_from = "u64", into = "u64")]
        pub struct $name(u64);

        impl $name {
            pub const fn get(self) -> u64 {
                self.0
            }
        }

        impl TryFrom<u64> for $name {
            type Error = VNextError;

            fn try_from(value: u64) -> Result<Self, Self::Error> {
                if value == 0 {
                    return Err(invalid_event(concat!($label, " must be non-zero")));
                }
                Ok(Self(value))
            }
        }

        impl From<$name> for u64 {
            fn from(value: $name) -> Self {
                value.0
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(formatter, "{}", self.0)
            }
        }
    };
}

nonzero_execution_id!(ExecutionFrameId, "execution frame id");
nonzero_execution_id!(BatchStepId, "batch step id");
nonzero_execution_id!(BatchInvocationId, "batch invocation id");
nonzero_execution_id!(NodeInvocationId, "node invocation id");

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExecutionEventKind {
    RequestAccepted,
    PlanBuilt,
    FrameStarted,
    NodeStarted,
    OperationSubmitted,
    NodeRetired,
    FrameCompleted,
    FailureObserved,
    SequenceCompleted,
    SequenceAborted,
    RequestCompleted,
    RequestFailed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExecutionPhase {
    Resolution,
    Planning,
    Execution,
    Completion,
}

impl ExecutionPhase {
    const fn rank(self) -> u8 {
        match self {
            Self::Resolution => 0,
            Self::Planning => 1,
            Self::Execution => 2,
            Self::Completion => 3,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MonotonicTimestamp {
    pub nanos_since_run_start: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ExecutionIdentityParts {
    pub version: ContractVersion,
    pub run_id: RunId,
    pub request_id: RequestIdentity,
    pub sequence: u64,
    pub plan_id: Option<PlanId>,
    pub plan_hash: Option<PlanHash>,
    pub frame_id: Option<ExecutionFrameId>,
    pub node_invocation_id: Option<NodeInvocationId>,
    pub node_id: Option<NodeId>,
    pub operation_id: Option<OperationId>,
    pub provider_id: Option<ProviderId>,
    pub device_id: Option<DeviceId>,
    pub resource_pool_id: Option<ResourcePoolId>,
    pub resource_pool_identity_fingerprint: Option<String>,
    pub provisioning_run_id: Option<RunId>,
    pub provisioning_request_id: Option<RequestIdentity>,
    pub transaction_id: Option<TransactionId>,
    pub active_sequence_slot: Option<u32>,
    pub admission_generation: Option<u64>,
    pub activation_epoch: Option<u64>,
    pub runtime_implementation_fingerprint: Option<String>,
    pub active_sequence_fingerprint: Option<String>,
    pub completed_sequence_fingerprint: Option<String>,
    pub aborted_sequence_fingerprint: Option<String>,
    pub resource_id: Option<ResourceId>,
    pub resource_generation: Option<u64>,
    pub resource_batch_fingerprint: Option<String>,
    pub span_id: SpanId,
    pub parent_span_id: Option<SpanId>,
    pub async_links: Vec<SpanId>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct UnvalidatedExecutionIdentityParts {
    pub version: ContractVersion,
    pub run_id: RunId,
    pub request_id: RequestIdentity,
    pub sequence: u64,
    pub plan_id: Option<PlanId>,
    pub plan_hash: Option<PlanHash>,
    pub frame_id: Option<ExecutionFrameId>,
    pub node_invocation_id: Option<NodeInvocationId>,
    pub node_id: Option<NodeId>,
    pub operation_id: Option<OperationId>,
    pub provider_id: Option<ProviderId>,
    pub device_id: Option<DeviceId>,
    pub resource_pool_id: Option<ResourcePoolId>,
    pub resource_pool_identity_fingerprint: Option<String>,
    pub provisioning_run_id: Option<RunId>,
    pub provisioning_request_id: Option<RequestIdentity>,
    pub transaction_id: Option<TransactionId>,
    pub active_sequence_slot: Option<u32>,
    pub admission_generation: Option<u64>,
    pub activation_epoch: Option<u64>,
    pub runtime_implementation_fingerprint: Option<String>,
    pub active_sequence_fingerprint: Option<String>,
    pub completed_sequence_fingerprint: Option<String>,
    pub aborted_sequence_fingerprint: Option<String>,
    pub resource_id: Option<ResourceId>,
    pub resource_generation: Option<u64>,
    pub resource_batch_fingerprint: Option<String>,
    pub span_id: SpanId,
    pub parent_span_id: Option<SpanId>,
    pub async_links: Vec<SpanId>,
}

impl From<UnvalidatedExecutionIdentityParts> for ExecutionIdentityParts {
    fn from(parts: UnvalidatedExecutionIdentityParts) -> Self {
        Self {
            version: parts.version,
            run_id: parts.run_id,
            request_id: parts.request_id,
            sequence: parts.sequence,
            plan_id: parts.plan_id,
            plan_hash: parts.plan_hash,
            frame_id: parts.frame_id,
            node_invocation_id: parts.node_invocation_id,
            node_id: parts.node_id,
            operation_id: parts.operation_id,
            provider_id: parts.provider_id,
            device_id: parts.device_id,
            resource_pool_id: parts.resource_pool_id,
            resource_pool_identity_fingerprint: parts.resource_pool_identity_fingerprint,
            provisioning_run_id: parts.provisioning_run_id,
            provisioning_request_id: parts.provisioning_request_id,
            transaction_id: parts.transaction_id,
            active_sequence_slot: parts.active_sequence_slot,
            admission_generation: parts.admission_generation,
            activation_epoch: parts.activation_epoch,
            runtime_implementation_fingerprint: parts.runtime_implementation_fingerprint,
            active_sequence_fingerprint: parts.active_sequence_fingerprint,
            completed_sequence_fingerprint: parts.completed_sequence_fingerprint,
            aborted_sequence_fingerprint: parts.aborted_sequence_fingerprint,
            resource_id: parts.resource_id,
            resource_generation: parts.resource_generation,
            resource_batch_fingerprint: parts.resource_batch_fingerprint,
            span_id: parts.span_id,
            parent_span_id: parts.parent_span_id,
            async_links: parts.async_links,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(transparent)]
pub struct ExecutionIdentityEnvelope {
    parts: ExecutionIdentityParts,
}

impl ExecutionIdentityEnvelope {
    pub fn new(parts: ExecutionIdentityParts) -> Result<Self, VNextError> {
        if parts.version != EXECUTION_IDENTITY_VERSION || parts.sequence == 0 {
            return Err(invalid_event(
                "execution identity version or sequence is invalid",
            ));
        }
        if parts.plan_id.is_some() != parts.plan_hash.is_some()
            || parts.node_invocation_id.is_some() != parts.node_id.is_some()
            || parts.node_id.is_some() && parts.frame_id.is_none()
            || parts.operation_id.is_some() != parts.provider_id.is_some()
            || parts.operation_id.is_some()
                && (parts.node_id.is_none() || parts.device_id.is_none())
            || parts.device_id.is_some() != parts.runtime_implementation_fingerprint.is_some()
        {
            return Err(invalid_event(
                "plan, frame, node invocation, operation, and provider identity shape is invalid",
            ));
        }

        let pool_present = parts.resource_pool_id.is_some();
        let pool_fields = [
            parts.resource_pool_identity_fingerprint.is_some(),
            parts.provisioning_run_id.is_some(),
            parts.provisioning_request_id.is_some(),
            parts.transaction_id.is_some(),
        ];
        if pool_fields.iter().any(|present| *present != pool_present) {
            return Err(invalid_event(
                "pool identity requires fingerprint and exact provisioning transaction",
            ));
        }
        let active_present = parts.active_sequence_slot.is_some();
        let active_fields = [
            parts.admission_generation.is_some(),
            parts.activation_epoch.is_some(),
            parts.active_sequence_fingerprint.is_some(),
        ];
        if active_fields
            .iter()
            .any(|present| *present != active_present)
            || active_present && parts.device_id.is_none()
        {
            return Err(invalid_event(
                "active identity requires slot, admission, epoch, runtime, and binding fingerprint",
            ));
        }
        if parts.admission_generation == Some(0) || parts.activation_epoch == Some(0) {
            return Err(invalid_event(
                "active admission generation and activation epoch must be non-zero",
            ));
        }
        if (parts.completed_sequence_fingerprint.is_some()
            || parts.aborted_sequence_fingerprint.is_some())
            && !active_present
            || parts.completed_sequence_fingerprint.is_some()
                && parts.aborted_sequence_fingerprint.is_some()
        {
            return Err(invalid_event(
                "sequence disposition requires one full active binding and cannot be both completed and aborted",
            ));
        }
        for (value, label) in [
            (
                parts.resource_pool_identity_fingerprint.as_deref(),
                "resource pool identity fingerprint",
            ),
            (
                parts.runtime_implementation_fingerprint.as_deref(),
                "runtime implementation fingerprint",
            ),
            (
                parts.active_sequence_fingerprint.as_deref(),
                "active sequence fingerprint",
            ),
            (
                parts.completed_sequence_fingerprint.as_deref(),
                "completed sequence fingerprint",
            ),
            (
                parts.aborted_sequence_fingerprint.as_deref(),
                "aborted sequence fingerprint",
            ),
            (
                parts.resource_batch_fingerprint.as_deref(),
                "resource batch fingerprint",
            ),
        ] {
            if let Some(value) = value {
                validate_sha256(value, label)?;
            }
        }
        if parts.resource_id.is_some() != parts.resource_generation.is_some()
            || parts.resource_generation == Some(0)
            || parts.resource_id.is_some() && parts.resource_batch_fingerprint.is_some()
            || (parts.resource_id.is_some() || parts.resource_batch_fingerprint.is_some())
                && !pool_present
        {
            return Err(invalid_event(
                "resource item/batch identity is incomplete, ambiguous, or lacks a pool",
            ));
        }
        if parts.parent_span_id.as_ref() == Some(&parts.span_id) {
            return Err(invalid_event("an execution span cannot parent itself"));
        }
        let mut links = BTreeSet::new();
        if parts.async_links.iter().any(|link| {
            link == &parts.span_id
                || parts.parent_span_id.as_ref() == Some(link)
                || !links.insert(link.clone())
        }) {
            return Err(invalid_event(
                "async links must be unique and distinct from span and parent",
            ));
        }
        Ok(Self { parts })
    }

    pub fn parts(&self) -> &ExecutionIdentityParts {
        &self.parts
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct TrustedNodeTopology {
    operation_id: OperationId,
    provider_id: ProviderId,
    dependencies: BTreeSet<NodeId>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct TrustedExecutionTopology {
    plan_id: PlanId,
    plan_hash: PlanHash,
    device_id: DeviceId,
    device_runtime_implementation_fingerprint: String,
    nodes: BTreeMap<NodeId, TrustedNodeTopology>,
    #[serde(skip)]
    fingerprint: String,
}

impl TrustedExecutionTopology {
    pub fn from_plan(plan: &ExecutionPlan) -> Result<Self, VNextError> {
        let mut nodes = BTreeMap::new();
        for node in plan.payload().nodes() {
            if nodes
                .insert(
                    node.id().clone(),
                    TrustedNodeTopology {
                        operation_id: node.operation_id().clone(),
                        provider_id: node.selection().selected_provider().clone(),
                        dependencies: node.dependencies().iter().cloned().collect(),
                    },
                )
                .is_some()
            {
                return Err(invalid_event("trusted plan has duplicate node ids"));
            }
        }
        if nodes.is_empty() {
            return Err(invalid_event("trusted execution topology is empty"));
        }
        let mut topology = Self {
            plan_id: plan.payload().plan_id().clone(),
            plan_hash: plan.plan_hash().clone(),
            device_id: plan.payload().device_id().clone(),
            device_runtime_implementation_fingerprint: plan
                .payload()
                .device_runtime_implementation_fingerprint()
                .to_owned(),
            nodes,
            fingerprint: String::new(),
        };
        validate_sha256(
            &topology.device_runtime_implementation_fingerprint,
            "topology runtime implementation fingerprint",
        )?;
        topology.fingerprint = canonical_fingerprint(&topology);
        Ok(topology)
    }

    pub fn plan_id(&self) -> &PlanId {
        &self.plan_id
    }

    pub fn plan_hash(&self) -> &PlanHash {
        &self.plan_hash
    }

    pub fn device_id(&self) -> &DeviceId {
        &self.device_id
    }

    pub fn device_runtime_implementation_fingerprint(&self) -> &str {
        &self.device_runtime_implementation_fingerprint
    }

    pub fn fingerprint(&self) -> &str {
        &self.fingerprint
    }

    pub fn node_ids(&self) -> BTreeSet<NodeId> {
        self.nodes.keys().cloned().collect()
    }
}

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
    backing_slices: Vec<LogicalBackingSliceEvidence>,
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
            backing_slices: permit
                .backing_slices()
                .iter()
                .map(|slice| slice.evidence().clone())
                .collect(),
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
            backing_slices: resources
                .backing_slices()
                .iter()
                .map(|slice| slice.evidence().clone())
                .collect(),
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

    fn ensure_open_for_emission(&self) -> Result<(), VNextError> {
        match &self.authority {
            TrustedActiveSequenceAuthority::StreamActivation => Err(invalid_event(
                "node execution emission requires typed sequence-session authority",
            )),
            TrustedActiveSequenceAuthority::SequenceSession { live_witness, .. } => {
                live_witness.ensure_open()
            }
        }
    }

    fn ensure_live_for_emission(&self) -> Result<(), VNextError> {
        match &self.authority {
            TrustedActiveSequenceAuthority::StreamActivation => Err(invalid_event(
                "operation progress emission requires typed sequence-session authority",
            )),
            TrustedActiveSequenceAuthority::SequenceSession { live_witness, .. } => {
                live_witness.ensure_live()
            }
        }
    }

    fn matches_abort_disposition(&self, disposition: ActiveSequenceAbortDisposition) -> bool {
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

    pub fn backing_slices(&self) -> &[LogicalBackingSliceEvidence] {
        &self.backing_slices
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

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct IdentifiedFailure {
    identity: ExecutionIdentityEnvelope,
    failure: FailureEnvelope,
}

impl IdentifiedFailure {
    pub fn new(
        identity: ExecutionIdentityEnvelope,
        failure: FailureEnvelope,
    ) -> Result<Self, VNextError> {
        failure.validate()?;
        validate_failure_identity(failure.domain(), identity.parts())?;
        Ok(Self { identity, failure })
    }

    pub fn identity(&self) -> &ExecutionIdentityEnvelope {
        &self.identity
    }

    pub fn failure(&self) -> &FailureEnvelope {
        &self.failure
    }

    pub fn fingerprint(&self) -> String {
        canonical_fingerprint(self)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct UnvalidatedIdentifiedFailure {
    identity: UnvalidatedExecutionIdentityParts,
    failure: UnvalidatedFailureEnvelope,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct UnvalidatedIdentifiedFailureWire {
    identity: UnvalidatedExecutionIdentityParts,
    failure: FailureEnvelopeWire,
}

impl From<UnvalidatedIdentifiedFailureWire> for UnvalidatedIdentifiedFailure {
    fn from(wire: UnvalidatedIdentifiedFailureWire) -> Self {
        Self {
            identity: wire.identity,
            failure: wire.failure.into(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ExecutionEventDetail {
    None,
    Counters { input: u64, output: u64 },
    Failure(IdentifiedFailure),
    FailureTerminal { first_failure_fingerprint: String },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum UnvalidatedExecutionEventDetail {
    None,
    Counters { input: u64, output: u64 },
    Failure(UnvalidatedIdentifiedFailure),
    FailureTerminal { first_failure_fingerprint: String },
}

#[derive(Deserialize)]
#[serde(rename_all = "snake_case")]
enum UnvalidatedExecutionEventDetailWire {
    None,
    Counters { input: u64, output: u64 },
    Failure(UnvalidatedIdentifiedFailureWire),
    FailureTerminal { first_failure_fingerprint: String },
}

impl From<UnvalidatedExecutionEventDetailWire> for UnvalidatedExecutionEventDetail {
    fn from(wire: UnvalidatedExecutionEventDetailWire) -> Self {
        match wire {
            UnvalidatedExecutionEventDetailWire::None => Self::None,
            UnvalidatedExecutionEventDetailWire::Counters { input, output } => {
                Self::Counters { input, output }
            }
            UnvalidatedExecutionEventDetailWire::Failure(failure) => Self::Failure(failure.into()),
            UnvalidatedExecutionEventDetailWire::FailureTerminal {
                first_failure_fingerprint,
            } => Self::FailureTerminal {
                first_failure_fingerprint,
            },
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ExecutionEvent {
    timestamp: MonotonicTimestamp,
    phase: ExecutionPhase,
    kind: ExecutionEventKind,
    identity: ExecutionIdentityEnvelope,
    detail: ExecutionEventDetail,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct UnvalidatedExecutionEvent {
    timestamp: MonotonicTimestamp,
    phase: ExecutionPhase,
    kind: ExecutionEventKind,
    identity: UnvalidatedExecutionIdentityParts,
    detail: UnvalidatedExecutionEventDetail,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ExecutionEventWire {
    timestamp: MonotonicTimestamp,
    phase: ExecutionPhase,
    kind: ExecutionEventKind,
    identity: UnvalidatedExecutionIdentityParts,
    detail: UnvalidatedExecutionEventDetailWire,
}

impl From<ExecutionEventWire> for UnvalidatedExecutionEvent {
    fn from(wire: ExecutionEventWire) -> Self {
        Self {
            timestamp: wire.timestamp,
            phase: wire.phase,
            kind: wire.kind,
            identity: wire.identity,
            detail: wire.detail.into(),
        }
    }
}

impl ExecutionEvent {
    pub fn new(
        timestamp: MonotonicTimestamp,
        phase: ExecutionPhase,
        kind: ExecutionEventKind,
        identity: ExecutionIdentityEnvelope,
        detail: ExecutionEventDetail,
    ) -> Result<Self, VNextError> {
        validate_event_shape(phase, kind, identity.parts(), &detail)?;
        Ok(Self {
            timestamp,
            phase,
            kind,
            identity,
            detail,
        })
    }

    pub const fn timestamp(&self) -> MonotonicTimestamp {
        self.timestamp
    }

    pub const fn phase(&self) -> ExecutionPhase {
        self.phase
    }

    pub const fn kind(&self) -> ExecutionEventKind {
        self.kind
    }

    pub fn identity(&self) -> &ExecutionIdentityEnvelope {
        &self.identity
    }

    pub fn detail(&self) -> &ExecutionEventDetail {
        &self.detail
    }

    pub fn decode_untrusted(bytes: &[u8]) -> Result<UnvalidatedExecutionEvent, VNextError> {
        if bytes.len() > MAX_EXECUTION_EVENT_WIRE_BYTES {
            return Err(invalid_event(
                "untrusted execution event exceeds the wire byte limit",
            ));
        }
        let raw = serde_json::from_slice::<serde_json::Value>(bytes).map_err(|error| {
            VNextError::Serialization {
                context: "decode untrusted execution event",
                message: error.to_string(),
            }
        })?;
        let event = serde_json::from_value::<ExecutionEventWire>(raw.clone())
            .map(UnvalidatedExecutionEvent::from)
            .map_err(|error| VNextError::Serialization {
                context: "decode untrusted execution event",
                message: error.to_string(),
            })?;
        let canonical =
            serde_json::to_value(&event).map_err(|error| VNextError::Serialization {
                context: "serialize untrusted execution event",
                message: error.to_string(),
            })?;
        if canonical != raw {
            return Err(invalid_event(
                "execution event wire contains unknown or non-canonical nested fields",
            ));
        }
        Ok(event)
    }
}

pub struct TrustedExecutionEventContext<'a> {
    run_id: &'a RunId,
    request_id: &'a RequestIdentity,
    topology: Option<&'a TrustedExecutionTopology>,
    active: Option<&'a TrustedActiveSequenceBinding>,
    completed: Option<&'a TrustedCompletedSequenceBinding>,
    aborted: Option<&'a TrustedAbortedSequenceBinding>,
    submitted_operation: Option<&'a SubmittedOperationReceipt>,
    retired_operation: Option<&'a OperationParticipantCompletionReceipt>,
    expected_failure: Option<&'a IdentifiedFailure>,
    unsubmitted_recovery_identity: Option<&'a ExecutionIdentityEnvelope>,
}

impl<'a> TrustedExecutionEventContext<'a> {
    pub fn pre_plan(run_id: &'a RunId, request_id: &'a RequestIdentity) -> Self {
        Self {
            run_id,
            request_id,
            topology: None,
            active: None,
            completed: None,
            aborted: None,
            submitted_operation: None,
            retired_operation: None,
            expected_failure: None,
            unsubmitted_recovery_identity: None,
        }
    }

    pub fn bound(
        run_id: &'a RunId,
        request_id: &'a RequestIdentity,
        topology: &'a TrustedExecutionTopology,
    ) -> Self {
        Self {
            run_id,
            request_id,
            topology: Some(topology),
            active: None,
            completed: None,
            aborted: None,
            submitted_operation: None,
            retired_operation: None,
            expected_failure: None,
            unsubmitted_recovery_identity: None,
        }
    }

    pub fn active(
        run_id: &'a RunId,
        request_id: &'a RequestIdentity,
        topology: &'a TrustedExecutionTopology,
        active: &'a TrustedActiveSequenceBinding,
    ) -> Self {
        Self {
            run_id,
            request_id,
            topology: Some(topology),
            active: Some(active),
            completed: None,
            aborted: None,
            submitted_operation: None,
            retired_operation: None,
            expected_failure: None,
            unsubmitted_recovery_identity: None,
        }
    }

    pub fn operation_submitted(
        run_id: &'a RunId,
        request_id: &'a RequestIdentity,
        topology: &'a TrustedExecutionTopology,
        active: &'a TrustedActiveSequenceBinding,
        submitted_operation: &'a SubmittedOperationReceipt,
    ) -> Self {
        Self {
            run_id,
            request_id,
            topology: Some(topology),
            active: Some(active),
            completed: None,
            aborted: None,
            submitted_operation: Some(submitted_operation),
            retired_operation: None,
            expected_failure: None,
            unsubmitted_recovery_identity: None,
        }
    }

    fn replay_operation_submitted(
        run_id: &'a RunId,
        request_id: &'a RequestIdentity,
        topology: &'a TrustedExecutionTopology,
        active: &'a TrustedActiveSequenceBinding,
        submitted_operation: &'a SubmittedOperationReceipt,
    ) -> Self {
        Self {
            run_id,
            request_id,
            topology: Some(topology),
            active: Some(active),
            completed: None,
            aborted: None,
            submitted_operation: Some(submitted_operation),
            retired_operation: None,
            expected_failure: None,
            unsubmitted_recovery_identity: None,
        }
    }

    pub fn node_retired(
        run_id: &'a RunId,
        request_id: &'a RequestIdentity,
        topology: &'a TrustedExecutionTopology,
        active: &'a TrustedActiveSequenceBinding,
        retired_operation: &'a OperationParticipantCompletionReceipt,
    ) -> Self {
        Self {
            run_id,
            request_id,
            topology: Some(topology),
            active: Some(active),
            completed: None,
            aborted: None,
            submitted_operation: None,
            retired_operation: Some(retired_operation),
            expected_failure: None,
            unsubmitted_recovery_identity: None,
        }
    }

    fn replay_node_retired(
        run_id: &'a RunId,
        request_id: &'a RequestIdentity,
        topology: &'a TrustedExecutionTopology,
        active: &'a TrustedActiveSequenceBinding,
        retired_operation: &'a OperationParticipantCompletionReceipt,
    ) -> Self {
        Self::node_retired(run_id, request_id, topology, active, retired_operation)
    }

    pub fn completed(
        run_id: &'a RunId,
        request_id: &'a RequestIdentity,
        topology: &'a TrustedExecutionTopology,
        active: &'a TrustedActiveSequenceBinding,
        completed: &'a TrustedCompletedSequenceBinding,
    ) -> Self {
        Self {
            run_id,
            request_id,
            topology: Some(topology),
            active: Some(active),
            completed: Some(completed),
            aborted: None,
            submitted_operation: None,
            retired_operation: None,
            expected_failure: None,
            unsubmitted_recovery_identity: None,
        }
    }

    pub fn aborted(
        run_id: &'a RunId,
        request_id: &'a RequestIdentity,
        topology: &'a TrustedExecutionTopology,
        active: &'a TrustedActiveSequenceBinding,
        aborted: &'a TrustedAbortedSequenceBinding,
    ) -> Self {
        Self {
            run_id,
            request_id,
            topology: Some(topology),
            active: Some(active),
            completed: None,
            aborted: Some(aborted),
            submitted_operation: None,
            retired_operation: None,
            expected_failure: None,
            unsubmitted_recovery_identity: None,
        }
    }

    pub fn failure(
        run_id: &'a RunId,
        request_id: &'a RequestIdentity,
        topology: Option<&'a TrustedExecutionTopology>,
        active: Option<&'a TrustedActiveSequenceBinding>,
        expected_failure: &'a IdentifiedFailure,
    ) -> Self {
        Self {
            run_id,
            request_id,
            topology,
            active,
            completed: None,
            aborted: None,
            submitted_operation: None,
            retired_operation: None,
            expected_failure: Some(expected_failure),
            unsubmitted_recovery_identity: None,
        }
    }

    fn replay_failure(
        run_id: &'a RunId,
        request_id: &'a RequestIdentity,
        topology: Option<&'a TrustedExecutionTopology>,
        active: Option<&'a TrustedActiveSequenceBinding>,
        expected_failure: &'a IdentifiedFailure,
        unsubmitted_recovery_identity: Option<&'a ExecutionIdentityEnvelope>,
    ) -> Self {
        Self {
            run_id,
            request_id,
            topology,
            active,
            completed: None,
            aborted: None,
            submitted_operation: None,
            retired_operation: None,
            expected_failure: Some(expected_failure),
            unsubmitted_recovery_identity,
        }
    }

    pub fn failure_with_disposition(
        run_id: &'a RunId,
        request_id: &'a RequestIdentity,
        topology: &'a TrustedExecutionTopology,
        active: &'a TrustedActiveSequenceBinding,
        completed: Option<&'a TrustedCompletedSequenceBinding>,
        aborted: Option<&'a TrustedAbortedSequenceBinding>,
        expected_failure: &'a IdentifiedFailure,
    ) -> Self {
        Self {
            run_id,
            request_id,
            topology: Some(topology),
            active: Some(active),
            completed,
            aborted,
            submitted_operation: None,
            retired_operation: None,
            expected_failure: Some(expected_failure),
            unsubmitted_recovery_identity: None,
        }
    }
}

impl UnvalidatedExecutionEvent {
    pub fn revalidate(
        self,
        context: &TrustedExecutionEventContext<'_>,
    ) -> Result<ExecutionEvent, VNextError> {
        let identity = ExecutionIdentityEnvelope::new(self.identity.into())?;
        let detail = match self.detail {
            UnvalidatedExecutionEventDetail::None => ExecutionEventDetail::None,
            UnvalidatedExecutionEventDetail::Counters { input, output } => {
                ExecutionEventDetail::Counters { input, output }
            }
            UnvalidatedExecutionEventDetail::Failure(failure) => {
                let expected = context.expected_failure.ok_or_else(|| {
                    invalid_event("wire failure lacks independent trusted failure evidence")
                })?;
                let failure_identity = ExecutionIdentityEnvelope::new(failure.identity.into())?;
                let trusted = IdentifiedFailure::new(
                    failure_identity,
                    failure.failure.revalidate(expected.failure().domain())?,
                )?;
                if &trusted != expected {
                    return Err(invalid_event(
                        "wire failure differs from independent failure evidence",
                    ));
                }
                ExecutionEventDetail::Failure(trusted)
            }
            UnvalidatedExecutionEventDetail::FailureTerminal {
                first_failure_fingerprint,
            } => {
                validate_sha256(&first_failure_fingerprint, "first failure fingerprint")?;
                let expected = context.expected_failure.ok_or_else(|| {
                    invalid_event("failure terminal lacks independent first failure evidence")
                })?;
                if first_failure_fingerprint != expected.fingerprint() {
                    return Err(invalid_event(
                        "failure terminal differs from the first observed failure",
                    ));
                }
                ExecutionEventDetail::FailureTerminal {
                    first_failure_fingerprint,
                }
            }
        };
        let event = ExecutionEvent::new(self.timestamp, self.phase, self.kind, identity, detail)?;
        validate_event_against_context(&event, context)?;
        Ok(event)
    }
}

fn has_pool(ids: &ExecutionIdentityParts) -> bool {
    ids.resource_pool_id.is_some()
}

fn has_active(ids: &ExecutionIdentityParts) -> bool {
    ids.active_sequence_slot.is_some()
}

fn has_completed(ids: &ExecutionIdentityParts) -> bool {
    ids.completed_sequence_fingerprint.is_some()
}

fn has_aborted(ids: &ExecutionIdentityParts) -> bool {
    ids.aborted_sequence_fingerprint.is_some()
}

fn no_resource_item(ids: &ExecutionIdentityParts) -> bool {
    ids.resource_id.is_none()
        && ids.resource_generation.is_none()
        && ids.resource_batch_fingerprint.is_none()
}

fn exact_plan(ids: &ExecutionIdentityParts) -> bool {
    ids.plan_id.is_some() && ids.plan_hash.is_some() && ids.device_id.is_some()
}

fn same_operation_authority_except_observation(
    observation: &ExecutionIdentityParts,
    operation: &ExecutionIdentityParts,
) -> bool {
    let mut normalized_observation = observation.clone();
    normalized_observation.sequence = operation.sequence;
    normalized_observation.span_id = operation.span_id.clone();
    normalized_observation.parent_span_id = operation.parent_span_id.clone();
    normalized_observation == *operation
}

fn validate_event_shape(
    phase: ExecutionPhase,
    kind: ExecutionEventKind,
    ids: &ExecutionIdentityParts,
    detail: &ExecutionEventDetail,
) -> Result<(), VNextError> {
    let phase_ok = match kind {
        ExecutionEventKind::RequestAccepted => phase == ExecutionPhase::Resolution,
        ExecutionEventKind::PlanBuilt => phase == ExecutionPhase::Planning,
        ExecutionEventKind::FrameStarted
        | ExecutionEventKind::NodeStarted
        | ExecutionEventKind::OperationSubmitted
        | ExecutionEventKind::NodeRetired
        | ExecutionEventKind::FrameCompleted => phase == ExecutionPhase::Execution,
        ExecutionEventKind::FailureObserved => true,
        ExecutionEventKind::SequenceCompleted
        | ExecutionEventKind::SequenceAborted
        | ExecutionEventKind::RequestCompleted => phase == ExecutionPhase::Completion,
        ExecutionEventKind::RequestFailed => true,
    };
    if !phase_ok {
        return Err(invalid_event(format!(
            "event `{kind:?}` is invalid in phase `{phase:?}`"
        )));
    }
    let no_plan = ids.plan_id.is_none() && ids.plan_hash.is_none() && ids.device_id.is_none();
    let no_frame = ids.frame_id.is_none() && ids.node_invocation_id.is_none();
    let no_node = ids.node_id.is_none() && ids.operation_id.is_none() && ids.provider_id.is_none();
    let no_pool = !has_pool(ids) && !has_active(ids) && no_resource_item(ids);
    let frame_shape = exact_plan(ids)
        && ids.frame_id.is_some()
        && ids.node_invocation_id.is_none()
        && no_node
        && has_active(ids)
        && !has_completed(ids)
        && !has_aborted(ids)
        && no_resource_item(ids);
    let node_shape = exact_plan(ids)
        && ids.frame_id.is_some()
        && ids.node_invocation_id.is_some()
        && ids.node_id.is_some()
        && ids.operation_id.is_some()
        && ids.provider_id.is_some()
        && has_active(ids)
        && !has_completed(ids)
        && !has_aborted(ids)
        && no_resource_item(ids);
    let completed_shape = exact_plan(ids)
        && no_frame
        && no_node
        && has_active(ids)
        && has_completed(ids)
        && !has_aborted(ids)
        && no_resource_item(ids);
    let aborted_shape = exact_plan(ids)
        && no_frame
        && no_node
        && has_active(ids)
        && !has_completed(ids)
        && has_aborted(ids)
        && no_resource_item(ids);
    let identity_ok = match kind {
        ExecutionEventKind::RequestAccepted => no_plan && no_frame && no_node && no_pool,
        ExecutionEventKind::PlanBuilt => exact_plan(ids) && no_frame && no_node && no_pool,
        ExecutionEventKind::FrameStarted | ExecutionEventKind::FrameCompleted => frame_shape,
        ExecutionEventKind::NodeStarted
        | ExecutionEventKind::OperationSubmitted
        | ExecutionEventKind::NodeRetired => node_shape,
        ExecutionEventKind::SequenceCompleted | ExecutionEventKind::RequestCompleted => {
            completed_shape
        }
        ExecutionEventKind::SequenceAborted => aborted_shape,
        ExecutionEventKind::FailureObserved => match detail {
            ExecutionEventDetail::Failure(failure) if has_active(ids) => {
                let failed_operation = failure.identity().parts();
                node_shape
                    && ids.sequence > failed_operation.sequence
                    && ids.parent_span_id.as_ref() == Some(&failed_operation.span_id)
                    && same_operation_authority_except_observation(ids, failed_operation)
            }
            ExecutionEventDetail::Failure(failure) => {
                failure.identity() == &ExecutionIdentityEnvelope { parts: ids.clone() }
            }
            _ => false,
        },
        ExecutionEventKind::RequestFailed => match detail {
            ExecutionEventDetail::Failure(failure) => {
                failure.identity() == &ExecutionIdentityEnvelope { parts: ids.clone() }
            }
            ExecutionEventDetail::FailureTerminal { .. } => {
                no_frame
                    && no_node
                    && no_resource_item(ids)
                    && (no_plan || exact_plan(ids))
                    && ((!has_active(ids) && !has_completed(ids) && !has_aborted(ids))
                        || (has_active(ids) && (has_completed(ids) ^ has_aborted(ids))))
            }
            _ => false,
        },
    };
    if !identity_ok {
        return Err(invalid_event(format!(
            "event `{kind:?}` has missing or extraneous identity fields"
        )));
    }
    let detail_ok = match (kind, detail) {
        (ExecutionEventKind::RequestCompleted, ExecutionEventDetail::Counters { .. }) => true,
        (ExecutionEventKind::FailureObserved, ExecutionEventDetail::Failure(_)) => true,
        (
            ExecutionEventKind::RequestFailed,
            ExecutionEventDetail::Failure(_) | ExecutionEventDetail::FailureTerminal { .. },
        ) => true,
        (
            ExecutionEventKind::RequestAccepted
            | ExecutionEventKind::PlanBuilt
            | ExecutionEventKind::FrameStarted
            | ExecutionEventKind::NodeStarted
            | ExecutionEventKind::OperationSubmitted
            | ExecutionEventKind::NodeRetired
            | ExecutionEventKind::FrameCompleted
            | ExecutionEventKind::SequenceCompleted
            | ExecutionEventKind::SequenceAborted,
            ExecutionEventDetail::None,
        ) => true,
        _ => false,
    };
    if !detail_ok {
        return Err(invalid_event(format!(
            "event `{kind:?}` has invalid structured detail"
        )));
    }
    if let ExecutionEventDetail::FailureTerminal {
        first_failure_fingerprint,
    } = detail
    {
        validate_sha256(first_failure_fingerprint, "first failure fingerprint")?;
    }
    Ok(())
}

fn validate_active_identity(
    ids: &ExecutionIdentityParts,
    active: &TrustedActiveSequenceBinding,
) -> Result<(), VNextError> {
    let provisioning = active.static_provisioning_identity();
    let pool_fingerprint = active.static_pool_identity_fingerprint();
    if &ids.run_id != active.run_id()
        || &ids.request_id != active.request_id()
        || ids.resource_pool_id != active.static_pool_id()
        || ids.resource_pool_identity_fingerprint.as_deref() != pool_fingerprint.as_deref()
        || ids.provisioning_run_id.as_ref() != provisioning.map(ResourceTransactionIdentity::run_id)
        || ids.provisioning_request_id.as_ref()
            != provisioning.map(ResourceTransactionIdentity::request_id)
        || ids.transaction_id.as_ref()
            != provisioning.map(ResourceTransactionIdentity::transaction_id)
        || ids.active_sequence_slot != Some(active.sequence_authority().sparse_id())
        || ids.admission_generation != Some(active.sequence_authority().generation())
        || ids.activation_epoch != Some(active.activation_epoch())
        || ids.runtime_implementation_fingerprint.as_deref()
            != Some(active.runtime_implementation_fingerprint())
        || ids.active_sequence_fingerprint.as_deref() != Some(active.fingerprint())
    {
        return Err(invalid_event(
            "event active identity differs from pool, epoch, runtime, or provisioning evidence",
        ));
    }
    Ok(())
}

fn validate_completed_identity(
    ids: &ExecutionIdentityParts,
    completed: &TrustedCompletedSequenceBinding,
    active: &TrustedActiveSequenceBinding,
) -> Result<(), VNextError> {
    if completed.active_sequence_fingerprint() != active.fingerprint()
        || completed.plan() != active.plan()
        || completed.coordinator_id() != active.coordinator_id()
        || completed.sequence_authority() != active.sequence_authority()
        || completed.run_id() != active.run_id()
        || completed.request_id() != active.request_id()
        || completed.activation_epoch() != active.activation_epoch()
        || completed.runtime_implementation_fingerprint()
            != active.runtime_implementation_fingerprint()
        || ids.completed_sequence_fingerprint.as_deref() != Some(completed.fingerprint())
    {
        return Err(invalid_event(
            "event completion identity differs from the synchronized active sequence receipt",
        ));
    }
    Ok(())
}

fn validate_aborted_identity(
    ids: &ExecutionIdentityParts,
    aborted: &TrustedAbortedSequenceBinding,
    active: &TrustedActiveSequenceBinding,
) -> Result<(), VNextError> {
    if !active.matches_abort_disposition(aborted.disposition())
        || aborted.active_sequence_fingerprint() != active.fingerprint()
        || aborted.plan() != active.plan()
        || aborted.coordinator_id() != active.coordinator_id()
        || aborted.sequence_authority() != active.sequence_authority()
        || aborted.run_id() != active.run_id()
        || aborted.request_id() != active.request_id()
        || aborted.activation_epoch() != active.activation_epoch()
        || aborted.runtime_implementation_fingerprint()
            != active.runtime_implementation_fingerprint()
        || ids.aborted_sequence_fingerprint.as_deref() != Some(aborted.fingerprint())
    {
        return Err(invalid_event(
            "event abort identity differs from the poisoned active sequence receipt",
        ));
    }
    Ok(())
}

fn validate_event_against_context(
    event: &ExecutionEvent,
    context: &TrustedExecutionEventContext<'_>,
) -> Result<(), VNextError> {
    let ids = event.identity.parts();
    if &ids.run_id != context.run_id || &ids.request_id != context.request_id {
        return Err(invalid_event(
            "event identity differs from trusted run/request context",
        ));
    }
    if let Some(topology) = context.topology {
        if ids.plan_id.as_ref() != Some(topology.plan_id())
            || ids.plan_hash.as_ref() != Some(topology.plan_hash())
            || ids.device_id.as_ref() != Some(topology.device_id())
            || ids.runtime_implementation_fingerprint.as_deref()
                != Some(topology.device_runtime_implementation_fingerprint())
        {
            return Err(invalid_event(
                "event plan identity differs from trusted topology",
            ));
        }
        if let Some(node_id) = &ids.node_id {
            let node = topology
                .nodes
                .get(node_id)
                .ok_or_else(|| invalid_event("event node is absent from trusted topology"))?;
            if ids.operation_id.as_ref() != Some(&node.operation_id)
                || ids.provider_id.as_ref() != Some(&node.provider_id)
            {
                return Err(invalid_event(
                    "event operation/provider differs from trusted node topology",
                ));
            }
        }
    } else if ids.plan_id.is_some() {
        return Err(invalid_event(
            "plan-bound event lacks trusted topology context",
        ));
    }
    match (has_active(ids), context.active) {
        (true, Some(active)) => {
            validate_active_identity(ids, active)?;
            let topology = context
                .topology
                .ok_or_else(|| invalid_event("active event lacks trusted execution topology"))?;
            if active.runtime_implementation_fingerprint()
                != topology.device_runtime_implementation_fingerprint()
                || active.plan().runtime_implementation_fingerprint()
                    != topology.device_runtime_implementation_fingerprint()
            {
                return Err(invalid_event(
                    "plan, admission, pool, and active runtime implementations differ",
                ));
            }
        }
        (false, None) => {}
        _ => {
            return Err(invalid_event(
                "event active identity presence differs from external active evidence",
            ));
        }
    }
    match (has_completed(ids), context.completed, context.active) {
        (true, Some(completed), Some(active)) => {
            validate_completed_identity(ids, completed, active)?;
        }
        (false, None, _) => {}
        _ => {
            return Err(invalid_event(
                "event completion identity presence differs from external synchronized receipt",
            ));
        }
    }
    match (has_aborted(ids), context.aborted, context.active) {
        (true, Some(aborted), Some(active)) => {
            validate_aborted_identity(ids, aborted, active)?;
        }
        (false, None, _) => {}
        _ => {
            return Err(invalid_event(
                "event abort identity presence differs from external poison receipt",
            ));
        }
    }
    match (event.kind, context.submitted_operation) {
        (ExecutionEventKind::OperationSubmitted, Some(submission))
            if submission
                .participants()
                .iter()
                .any(|participant| participant.identity() == event.identity()) => {}
        (ExecutionEventKind::OperationSubmitted, _) => {
            return Err(invalid_event(
                "OperationSubmitted lacks its exact external dispatch receipt",
            ));
        }
        (_, None) => {}
        (_, Some(_)) => {
            return Err(invalid_event(
                "operation submission receipt supplied for a different event kind",
            ));
        }
    }
    match (event.kind, context.retired_operation) {
        (ExecutionEventKind::NodeRetired, Some(completion))
            if same_operation_authority_except_observation(
                event.identity().parts(),
                completion.submission().identity().parts(),
            ) => {}
        (ExecutionEventKind::NodeRetired, _) => {
            return Err(invalid_event(
                "NodeRetired lacks its exact participant completion projection",
            ));
        }
        (_, None) => {}
        (_, Some(_)) => {
            return Err(invalid_event(
                "operation completion projection supplied for a different event kind",
            ));
        }
    }
    match (&event.detail, context.expected_failure) {
        (ExecutionEventDetail::Failure(failure), Some(expected)) if failure == expected => {}
        (
            ExecutionEventDetail::FailureTerminal {
                first_failure_fingerprint,
            },
            Some(expected),
        ) if first_failure_fingerprint == &expected.fingerprint() => {}
        (ExecutionEventDetail::Failure(_) | ExecutionEventDetail::FailureTerminal { .. }, _) => {
            return Err(invalid_event(
                "event failure differs from independent first failure evidence",
            ));
        }
        (_, None) => {}
        (_, Some(_)) => {
            return Err(invalid_event(
                "trusted failure evidence supplied for a non-failure event",
            ));
        }
    }
    if let Some(recovery_identity) = context.unsubmitted_recovery_identity {
        let ExecutionEventDetail::Failure(failure) = &event.detail else {
            return Err(invalid_event(
                "unsubmitted recovery identity was supplied for a non-failure event",
            ));
        };
        if failure.identity() != recovery_identity {
            return Err(invalid_event(
                "unsubmitted recovery identity differs from the exact observed failure",
            ));
        }
    }
    Ok(())
}

fn validate_failure_identity(
    domain: FailureDomain,
    ids: &ExecutionIdentityParts,
) -> Result<(), VNextError> {
    let operation_shape = exact_plan(ids)
        && ids.frame_id.is_some()
        && ids.node_invocation_id.is_some()
        && ids.node_id.is_some()
        && ids.operation_id.is_some()
        && ids.provider_id.is_some()
        && has_active(ids)
        && !has_completed(ids)
        && !has_aborted(ids)
        && no_resource_item(ids);
    let resource_shape = exact_plan(ids)
        && ids.frame_id.is_none()
        && ids.node_invocation_id.is_none()
        && ids.node_id.is_none()
        && ids.operation_id.is_none()
        && ids.provider_id.is_none()
        && has_pool(ids)
        && !has_active(ids)
        && !has_completed(ids)
        && !has_aborted(ids)
        && (ids.resource_id.is_some() ^ ids.resource_batch_fingerprint.is_some());
    let plan_shape = exact_plan(ids)
        && ids.frame_id.is_none()
        && ids.node_id.is_none()
        && !has_pool(ids)
        && !has_completed(ids)
        && !has_aborted(ids)
        && no_resource_item(ids);
    let device_only = ids.device_id.is_some()
        && ids.plan_id.is_none()
        && ids.frame_id.is_none()
        && ids.node_id.is_none()
        && !has_pool(ids)
        && !has_completed(ids)
        && !has_aborted(ids)
        && no_resource_item(ids);
    let pre_plan = ids.plan_id.is_none()
        && ids.device_id.is_none()
        && ids.frame_id.is_none()
        && ids.node_id.is_none()
        && !has_pool(ids)
        && !has_completed(ids)
        && !has_aborted(ids)
        && no_resource_item(ids);
    let valid = match domain {
        FailureDomain::Operation => operation_shape,
        FailureDomain::Resource => resource_shape,
        FailureDomain::Device => device_only || plan_shape || operation_shape || resource_shape,
        FailureDomain::Planning => plan_shape,
        FailureDomain::ModelResolution | FailureDomain::Product => pre_plan || plan_shape,
        FailureDomain::Event => pre_plan || plan_shape || operation_shape,
    };
    if !valid {
        return Err(invalid_event(format!(
            "failure domain `{domain:?}` has missing or extraneous execution identity fields"
        )));
    }
    Ok(())
}

#[derive(Debug, Clone)]
struct ActiveNodeInvocation {
    invocation_id: NodeInvocationId,
    node_span: SpanId,
    operation_submitted: bool,
}

#[derive(Debug, Clone)]
struct ActiveFrame {
    id: ExecutionFrameId,
    span_id: SpanId,
    active_nodes: BTreeMap<NodeId, ActiveNodeInvocation>,
    completed_nodes: BTreeSet<NodeId>,
}

#[derive(Debug, Clone)]
pub struct ExecutionEventCursor {
    run_id: RunId,
    request_id: RequestIdentity,
    last_sequence: u64,
    last_timestamp: Option<MonotonicTimestamp>,
    last_phase: Option<ExecutionPhase>,
    topology_fingerprint: Option<String>,
    active_fingerprint: Option<String>,
    completion_fingerprint: Option<String>,
    abort_fingerprint: Option<String>,
    observed_failure: Option<IdentifiedFailure>,
    accepted: bool,
    planned: bool,
    terminal: bool,
    root_span: Option<SpanId>,
    seen_spans: BTreeSet<SpanId>,
    next_frame: u64,
    next_invocation: u64,
    completed_frames: u64,
    frame: Option<ActiveFrame>,
}

impl ExecutionEventCursor {
    pub fn new(run_id: RunId, request_id: RequestIdentity) -> Self {
        Self {
            run_id,
            request_id,
            last_sequence: 0,
            last_timestamp: None,
            last_phase: None,
            topology_fingerprint: None,
            active_fingerprint: None,
            completion_fingerprint: None,
            abort_fingerprint: None,
            observed_failure: None,
            accepted: false,
            planned: false,
            terminal: false,
            root_span: None,
            seen_spans: BTreeSet::new(),
            next_frame: 1,
            next_invocation: 1,
            completed_frames: 0,
            frame: None,
        }
    }

    pub fn observe_against(
        &mut self,
        event: &ExecutionEvent,
        context: &TrustedExecutionEventContext<'_>,
    ) -> Result<(), VNextError> {
        let mut next = self.clone();
        next.observe_inner(event, context)?;
        *self = next;
        Ok(())
    }

    pub const fn last_sequence(&self) -> u64 {
        self.last_sequence
    }

    pub const fn is_terminal(&self) -> bool {
        self.terminal
    }

    pub const fn completed_frames(&self) -> u64 {
        self.completed_frames
    }

    fn observe_inner(
        &mut self,
        event: &ExecutionEvent,
        context: &TrustedExecutionEventContext<'_>,
    ) -> Result<(), VNextError> {
        validate_event_against_context(event, context)?;
        let ids = event.identity.parts();
        if ids.run_id != self.run_id
            || ids.request_id != self.request_id
            || ids.sequence != self.last_sequence.saturating_add(1)
            || self
                .last_timestamp
                .is_some_and(|timestamp| event.timestamp <= timestamp)
            || self
                .last_phase
                .is_some_and(|phase| event.phase.rank() < phase.rank())
            || self.terminal
        {
            return Err(invalid_event(
                "request journal run, request, sequence, timestamp, phase, or terminal boundary is invalid",
            ));
        }
        if let Some(topology) = context.topology {
            if let Some(bound) = &self.topology_fingerprint {
                if bound != topology.fingerprint() {
                    return Err(invalid_event("request changed trusted topology"));
                }
            }
        }
        if let Some(active) = context.active {
            if let Some(bound) = &self.active_fingerprint {
                if bound != active.fingerprint() {
                    return Err(invalid_event(
                        "request changed active pool/slot/epoch/runtime binding",
                    ));
                }
            }
        }
        if self.observed_failure.is_some()
            && !matches!(
                event.kind,
                ExecutionEventKind::SequenceCompleted
                    | ExecutionEventKind::SequenceAborted
                    | ExecutionEventKind::RequestFailed
            )
        {
            return Err(invalid_event(
                "only sequence disposition and terminal failure may follow FailureObserved",
            ));
        }

        match event.kind {
            ExecutionEventKind::RequestAccepted => self.accept(ids)?,
            ExecutionEventKind::PlanBuilt => {
                let topology = context
                    .topology
                    .ok_or_else(|| invalid_event("PlanBuilt lacks trusted topology"))?;
                self.plan(ids, topology)?;
            }
            ExecutionEventKind::FrameStarted => {
                let topology = self.require_topology(context)?;
                let active = self.require_active(context)?;
                self.start_frame(ids, topology, active)?;
            }
            ExecutionEventKind::NodeStarted => {
                let topology = self.require_topology(context)?;
                self.require_active(context)?;
                self.start_node(ids, topology)?;
            }
            ExecutionEventKind::OperationSubmitted => self.submit_operation(ids)?,
            ExecutionEventKind::NodeRetired => self.retire_node(ids)?,
            ExecutionEventKind::FrameCompleted => {
                let topology = self.require_topology(context)?;
                self.require_active(context)?;
                self.complete_frame(ids, topology)?;
            }
            ExecutionEventKind::FailureObserved => {
                self.observe_failure(event, context.unsubmitted_recovery_identity)?
            }
            ExecutionEventKind::SequenceCompleted => {
                self.require_topology(context)?;
                self.require_active(context)?;
                let completed = context.completed.ok_or_else(|| {
                    invalid_event("SequenceCompleted lacks synchronized completion evidence")
                })?;
                self.complete_sequence(ids, completed)?;
            }
            ExecutionEventKind::SequenceAborted => {
                self.require_topology(context)?;
                self.require_active(context)?;
                let aborted = context.aborted.ok_or_else(|| {
                    invalid_event("SequenceAborted lacks poisoned abort evidence")
                })?;
                self.abort_sequence(ids, aborted)?;
            }
            ExecutionEventKind::RequestCompleted => {
                self.require_topology(context)?;
                self.require_active(context)?;
                if context.completed.is_none() {
                    return Err(invalid_event(
                        "RequestCompleted lacks synchronized completion evidence",
                    ));
                }
                self.complete_success(ids)?;
            }
            ExecutionEventKind::RequestFailed => self.fail_request(event)?,
        }
        self.last_sequence = ids.sequence;
        self.last_timestamp = Some(event.timestamp);
        self.last_phase = Some(event.phase);
        Ok(())
    }

    fn require_topology<'a>(
        &self,
        context: &'a TrustedExecutionEventContext<'_>,
    ) -> Result<&'a TrustedExecutionTopology, VNextError> {
        if !self.planned {
            return Err(invalid_event("execution event precedes PlanBuilt"));
        }
        context
            .topology
            .ok_or_else(|| invalid_event("execution event lacks trusted topology"))
    }

    fn require_active<'a>(
        &self,
        context: &'a TrustedExecutionEventContext<'_>,
    ) -> Result<&'a TrustedActiveSequenceBinding, VNextError> {
        context
            .active
            .ok_or_else(|| invalid_event("execution event lacks active sequence evidence"))
    }

    fn accept(&mut self, ids: &ExecutionIdentityParts) -> Result<(), VNextError> {
        if self.accepted || self.last_sequence != 0 || ids.parent_span_id.is_some() {
            return Err(invalid_event(
                "RequestAccepted must open the first root span",
            ));
        }
        self.seen_spans.insert(ids.span_id.clone());
        self.root_span = Some(ids.span_id.clone());
        self.accepted = true;
        Ok(())
    }

    fn plan(
        &mut self,
        ids: &ExecutionIdentityParts,
        topology: &TrustedExecutionTopology,
    ) -> Result<(), VNextError> {
        if !self.accepted
            || self.planned
            || ids.parent_span_id.as_ref() != self.root_span.as_ref()
            || !self.seen_spans.insert(ids.span_id.clone())
        {
            return Err(invalid_event(
                "PlanBuilt must uniquely bind one topology under the request root",
            ));
        }
        self.topology_fingerprint = Some(topology.fingerprint().to_owned());
        self.planned = true;
        Ok(())
    }

    fn start_frame(
        &mut self,
        ids: &ExecutionIdentityParts,
        _topology: &TrustedExecutionTopology,
        active: &TrustedActiveSequenceBinding,
    ) -> Result<(), VNextError> {
        let frame_id = ids.frame_id.expect("frame shape validated");
        if self.frame.is_some()
            || self.completion_fingerprint.is_some()
            || self.abort_fingerprint.is_some()
            || self.observed_failure.is_some()
            || frame_id.get() != self.next_frame
            || ids.parent_span_id.as_ref() != self.root_span.as_ref()
            || !self.seen_spans.insert(ids.span_id.clone())
        {
            return Err(invalid_event(
                "frames must start once in strict contiguous order under the request root",
            ));
        }
        self.active_fingerprint
            .get_or_insert_with(|| active.fingerprint().to_owned());
        self.frame = Some(ActiveFrame {
            id: frame_id,
            span_id: ids.span_id.clone(),
            active_nodes: BTreeMap::new(),
            completed_nodes: BTreeSet::new(),
        });
        Ok(())
    }

    fn start_node(
        &mut self,
        ids: &ExecutionIdentityParts,
        topology: &TrustedExecutionTopology,
    ) -> Result<(), VNextError> {
        let node_id = ids.node_id.as_ref().expect("node shape validated");
        let invocation_id = ids
            .node_invocation_id
            .expect("node invocation shape validated");
        let frame = self
            .frame
            .as_mut()
            .ok_or_else(|| invalid_event("node started outside an active frame"))?;
        let node = topology
            .nodes
            .get(node_id)
            .ok_or_else(|| invalid_event("node is absent from trusted topology"))?;
        if ids.frame_id != Some(frame.id)
            || invocation_id.get() != self.next_invocation
            || frame.active_nodes.contains_key(node_id)
            || frame.completed_nodes.contains(node_id)
            || node
                .dependencies
                .iter()
                .any(|dependency| !frame.completed_nodes.contains(dependency))
            || ids.parent_span_id.as_ref() != Some(&frame.span_id)
            || !self.seen_spans.insert(ids.span_id.clone())
        {
            return Err(invalid_event(
                "node invocation is duplicate, non-monotonic, cross-frame, or precedes same-frame dependencies",
            ));
        }
        self.next_invocation = self
            .next_invocation
            .checked_add(1)
            .ok_or_else(|| invalid_event("node invocation id overflow"))?;
        frame.active_nodes.insert(
            node_id.clone(),
            ActiveNodeInvocation {
                invocation_id,
                node_span: ids.span_id.clone(),
                operation_submitted: false,
            },
        );
        Ok(())
    }

    fn submit_operation(&mut self, ids: &ExecutionIdentityParts) -> Result<(), VNextError> {
        let node_id = ids.node_id.as_ref().expect("operation shape validated");
        let frame = self
            .frame
            .as_mut()
            .ok_or_else(|| invalid_event("operation submitted outside an active frame"))?;
        let active = frame
            .active_nodes
            .get_mut(node_id)
            .ok_or_else(|| invalid_event("operation submitted without active node"))?;
        if ids.frame_id != Some(frame.id)
            || ids.node_invocation_id != Some(active.invocation_id)
            || active.operation_submitted
            || ids.parent_span_id.as_ref() != Some(&active.node_span)
            || !self.seen_spans.insert(ids.span_id.clone())
        {
            return Err(invalid_event(
                "operation submission does not match the active node invocation",
            ));
        }
        active.operation_submitted = true;
        Ok(())
    }

    fn retire_node(&mut self, ids: &ExecutionIdentityParts) -> Result<(), VNextError> {
        let node_id = ids.node_id.as_ref().expect("node shape validated");
        let frame = self
            .frame
            .as_mut()
            .ok_or_else(|| invalid_event("node completed outside an active frame"))?;
        let active = frame
            .active_nodes
            .get(node_id)
            .ok_or_else(|| invalid_event("node completed without active invocation"))?;
        if ids.frame_id != Some(frame.id)
            || ids.node_invocation_id != Some(active.invocation_id)
            || ids.span_id != active.node_span
            || ids.parent_span_id.as_ref() != Some(&frame.span_id)
            || !active.operation_submitted
        {
            return Err(invalid_event(
                "node completion requires its exact frame, invocation, span, and operation",
            ));
        }
        frame.active_nodes.remove(node_id);
        frame.completed_nodes.insert(node_id.clone());
        Ok(())
    }

    fn complete_sequence(
        &mut self,
        ids: &ExecutionIdentityParts,
        completed: &TrustedCompletedSequenceBinding,
    ) -> Result<(), VNextError> {
        let failure_cleanup = self.observed_failure.is_some();
        if !self.planned
            || !failure_cleanup && (self.completed_frames == 0 || self.frame.is_some())
            || failure_cleanup && self.active_fingerprint.is_none()
            || self.active_fingerprint.as_deref() != Some(completed.active_sequence_fingerprint())
            || self.completion_fingerprint.is_some()
            || self.abort_fingerprint.is_some()
            || ids.parent_span_id.as_ref() != self.root_span.as_ref()
            || !self.seen_spans.insert(ids.span_id.clone())
        {
            return Err(invalid_event(
                "SequenceCompleted requires submitted frames and one unique synchronized receipt",
            ));
        }
        if failure_cleanup {
            self.frame = None;
        }
        self.completion_fingerprint = Some(completed.fingerprint().to_owned());
        Ok(())
    }

    fn abort_sequence(
        &mut self,
        ids: &ExecutionIdentityParts,
        aborted: &TrustedAbortedSequenceBinding,
    ) -> Result<(), VNextError> {
        if self.observed_failure.is_none()
            || self.active_fingerprint.is_none()
            || self.active_fingerprint.as_deref() != Some(aborted.active_sequence_fingerprint())
            || self.completion_fingerprint.is_some()
            || self.abort_fingerprint.is_some()
            || ids.parent_span_id.as_ref() != self.root_span.as_ref()
            || !self.seen_spans.insert(ids.span_id.clone())
        {
            return Err(invalid_event(
                "SequenceAborted requires one observed failure and one unique poison receipt",
            ));
        }
        self.frame = None;
        self.abort_fingerprint = Some(aborted.fingerprint().to_owned());
        Ok(())
    }

    fn complete_frame(
        &mut self,
        ids: &ExecutionIdentityParts,
        topology: &TrustedExecutionTopology,
    ) -> Result<(), VNextError> {
        let frame = self
            .frame
            .as_ref()
            .ok_or_else(|| invalid_event("FrameCompleted lacks an active frame"))?;
        if ids.frame_id != Some(frame.id)
            || ids.span_id != frame.span_id
            || ids.parent_span_id.as_ref() != self.root_span.as_ref()
            || !frame.active_nodes.is_empty()
            || frame.completed_nodes != topology.node_ids()
        {
            return Err(invalid_event(
                "frame completion requires every trusted node exactly once and no active invocation",
            ));
        }
        self.frame = None;
        self.completed_frames += 1;
        self.next_frame = self
            .next_frame
            .checked_add(1)
            .ok_or_else(|| invalid_event("frame id overflow"))?;
        Ok(())
    }

    fn observe_failure(
        &mut self,
        event: &ExecutionEvent,
        unsubmitted_recovery_identity: Option<&ExecutionIdentityEnvelope>,
    ) -> Result<(), VNextError> {
        if !self.accepted || self.observed_failure.is_some() {
            return Err(invalid_event(
                "FailureObserved requires one accepted non-failed request",
            ));
        }
        let failure = match &event.detail {
            ExecutionEventDetail::Failure(failure) => failure,
            _ => return Err(invalid_event("FailureObserved lacks identified failure")),
        };
        let ids = event.identity.parts();
        if has_active(ids) {
            let failed_operation = failure.identity().parts();
            self.active_fingerprint
                .get_or_insert_with(|| ids.active_sequence_fingerprint.clone().unwrap());
            let frame = self.frame.as_ref().ok_or_else(|| {
                invalid_event("active operation failure lacks its execution frame")
            })?;
            let node_id = ids
                .node_id
                .as_ref()
                .ok_or_else(|| invalid_event("active operation failure lacks its node identity"))?;
            let invocation = frame.active_nodes.get(node_id).ok_or_else(|| {
                invalid_event("active operation failure lacks its node invocation")
            })?;
            let operation_span_was_submitted = self.seen_spans.contains(&failed_operation.span_id);
            let is_unsubmitted_recovery = unsubmitted_recovery_identity
                .is_some_and(|identity| identity == failure.identity());
            if ids.frame_id != Some(frame.id)
                || ids.node_invocation_id != Some(invocation.invocation_id)
                || !same_operation_authority_except_observation(ids, failed_operation)
                || failed_operation.parent_span_id.as_ref() != Some(&invocation.node_span)
                || ids.parent_span_id.as_ref() != Some(&failed_operation.span_id)
                || operation_span_was_submitted == is_unsubmitted_recovery
            {
                return Err(invalid_event(
                    "operation failure does not link one exact submitted operation to its observation span",
                ));
            }
        } else if ids.parent_span_id.as_ref() != self.root_span.as_ref()
            && !(ids.span_id == *self.root_span.as_ref().expect("accepted root")
                && ids.parent_span_id.is_none())
        {
            return Err(invalid_event(
                "non-active failure must be anchored under the request root",
            ));
        }
        if !self.seen_spans.insert(ids.span_id.clone()) {
            return Err(invalid_event("FailureObserved span was already used"));
        }
        self.observed_failure = Some(failure.clone());
        Ok(())
    }

    fn complete_success(&mut self, ids: &ExecutionIdentityParts) -> Result<(), VNextError> {
        if self.observed_failure.is_some()
            || !self.planned
            || self.completed_frames == 0
            || self.frame.is_some()
            || self.abort_fingerprint.is_some()
            || self.completion_fingerprint.as_deref()
                != ids.completed_sequence_fingerprint.as_deref()
        {
            return Err(invalid_event(
                "successful request requires submitted frames and the exact synchronized sequence receipt",
            ));
        }
        if ids.span_id != *self.root_span.as_ref().expect("accepted root")
            || ids.parent_span_id.is_some()
        {
            return Err(invalid_event(
                "terminal request event must close the exact request root",
            ));
        }
        self.terminal = true;
        Ok(())
    }

    fn fail_request(&mut self, event: &ExecutionEvent) -> Result<(), VNextError> {
        let ids = event.identity.parts();
        if !self.accepted {
            if self.last_sequence != 0
                || ids.parent_span_id.is_some()
                || !matches!(event.detail, ExecutionEventDetail::Failure(_))
            {
                return Err(invalid_event(
                    "only first-event pre-plan RequestFailed may precede acceptance",
                ));
            }
            self.terminal = true;
            return Ok(());
        }
        let observed = self
            .observed_failure
            .as_ref()
            .ok_or_else(|| invalid_event("RequestFailed lacks FailureObserved"))?;
        let terminal_fingerprint = match &event.detail {
            ExecutionEventDetail::FailureTerminal {
                first_failure_fingerprint,
            } => first_failure_fingerprint,
            _ => {
                return Err(invalid_event(
                    "post-acceptance RequestFailed requires FailureTerminal",
                ));
            }
        };
        if terminal_fingerprint != &observed.fingerprint() {
            return Err(invalid_event(
                "RequestFailed does not reference the first observed failure",
            ));
        }
        if self.active_fingerprint.is_some() {
            let completed_matches = self.completion_fingerprint.is_some()
                && self.completion_fingerprint.as_deref()
                    == ids.completed_sequence_fingerprint.as_deref();
            let aborted_matches = self.abort_fingerprint.is_some()
                && self.abort_fingerprint.as_deref() == ids.aborted_sequence_fingerprint.as_deref();
            if completed_matches == aborted_matches {
                return Err(invalid_event(
                    "active RequestFailed requires exactly one matching completion or abort disposition",
                ));
            }
            if completed_matches && ids.aborted_sequence_fingerprint.is_some()
                || aborted_matches && ids.completed_sequence_fingerprint.is_some()
            {
                return Err(invalid_event(
                    "active RequestFailed carries an unexpected opposite sequence disposition",
                ));
            }
        } else if self.completion_fingerprint.is_some()
            || self.abort_fingerprint.is_some()
            || has_completed(ids)
            || has_aborted(ids)
        {
            return Err(invalid_event(
                "non-active RequestFailed cannot carry sequence disposition",
            ));
        }
        if ids.span_id != *self.root_span.as_ref().expect("accepted root")
            || ids.parent_span_id.is_some()
        {
            return Err(invalid_event(
                "terminal request event must close the exact request root",
            ));
        }
        self.terminal = true;
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ResourcePoolEvidence {
    topology_fingerprint: String,
    pool_id: ResourcePoolId,
    pool_identity_fingerprint: String,
    admission: StaticProvisioningBinding,
    provisioning_identity: ResourceTransactionIdentity,
}

impl ResourcePoolEvidence {
    pub fn from_external(
        topology: &TrustedExecutionTopology,
        admission: &StaticProvisioningBinding,
        provisioning_identity: &ResourceTransactionIdentity,
    ) -> Result<Self, VNextError> {
        let pool_fingerprint = canonical_fingerprint(admission.pool_identity());
        if admission.plan_id() != topology.plan_id()
            || admission.plan_hash() != topology.plan_hash()
            || admission.device_id() != topology.device_id()
            || admission.device_runtime_implementation_fingerprint()
                != topology.device_runtime_implementation_fingerprint()
            || admission
                .pool_identity()
                .device_runtime_implementation_fingerprint()
                != topology.device_runtime_implementation_fingerprint()
            || provisioning_identity.pool_id() != admission.pool_id()
            || provisioning_identity.request_id() != admission.request_id()
        {
            return Err(invalid_event(
                "pool evidence differs from topology, admission, or provisioning identity",
            ));
        }
        Ok(Self {
            topology_fingerprint: topology.fingerprint().to_owned(),
            pool_id: admission.pool_id(),
            pool_identity_fingerprint: pool_fingerprint,
            admission: admission.clone(),
            provisioning_identity: provisioning_identity.clone(),
        })
    }

    pub const fn pool_id(&self) -> ResourcePoolId {
        self.pool_id
    }

    pub fn topology_fingerprint(&self) -> &str {
        &self.topology_fingerprint
    }

    pub fn pool_identity_fingerprint(&self) -> &str {
        &self.pool_identity_fingerprint
    }

    pub fn admission(&self) -> &StaticProvisioningBinding {
        &self.admission
    }

    pub fn provisioning_identity(&self) -> &ResourceTransactionIdentity {
        &self.provisioning_identity
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResourcePoolEventKind {
    ResourcePoolOpened,
    ResourceTransition,
    ResourceLeaseTransition,
    ResourceFailed,
    ResourceRecoveryCompleted,
    ResourcePoolClosed,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ResourcePoolEventIdentity {
    sequence: u64,
    pool_id: ResourcePoolId,
    pool_identity_fingerprint: String,
    provisioning_run_id: RunId,
    provisioning_request_id: RequestIdentity,
    transaction_id: TransactionId,
    resource_id: Option<ResourceId>,
    resource_generation: Option<u64>,
    resource_batch_fingerprint: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct UnvalidatedResourcePoolEventIdentity {
    sequence: u64,
    pool_id: ResourcePoolId,
    pool_identity_fingerprint: String,
    provisioning_run_id: RunId,
    provisioning_request_id: RequestIdentity,
    transaction_id: TransactionId,
    resource_id: Option<ResourceId>,
    resource_generation: Option<u64>,
    resource_batch_fingerprint: Option<String>,
}

impl ResourcePoolEventIdentity {
    fn for_evidence(
        sequence: u64,
        evidence: &ResourcePoolEvidence,
        resource_id: Option<ResourceId>,
        resource_generation: Option<u64>,
        resource_batch_fingerprint: Option<String>,
    ) -> Result<Self, VNextError> {
        if sequence == 0
            || resource_id.is_some() != resource_generation.is_some()
            || resource_generation == Some(0)
            || resource_id.is_some() && resource_batch_fingerprint.is_some()
        {
            return Err(invalid_event(
                "resource pool event identity shape is invalid",
            ));
        }
        if let Some(fingerprint) = &resource_batch_fingerprint {
            validate_sha256(fingerprint, "resource pool event batch fingerprint")?;
        }
        Ok(Self {
            sequence,
            pool_id: evidence.pool_id,
            pool_identity_fingerprint: evidence.pool_identity_fingerprint.clone(),
            provisioning_run_id: evidence.provisioning_identity.run_id().clone(),
            provisioning_request_id: evidence.provisioning_identity.request_id().clone(),
            transaction_id: evidence.provisioning_identity.transaction_id().clone(),
            resource_id,
            resource_generation,
            resource_batch_fingerprint,
        })
    }

    pub const fn sequence(&self) -> u64 {
        self.sequence
    }

    pub const fn pool_id(&self) -> ResourcePoolId {
        self.pool_id
    }

    pub fn pool_identity_fingerprint(&self) -> &str {
        &self.pool_identity_fingerprint
    }

    pub fn transaction_id(&self) -> &TransactionId {
        &self.transaction_id
    }

    pub fn provisioning_run_id(&self) -> &RunId {
        &self.provisioning_run_id
    }

    pub fn provisioning_request_id(&self) -> &RequestIdentity {
        &self.provisioning_request_id
    }

    pub fn resource_id(&self) -> Option<&ResourceId> {
        self.resource_id.as_ref()
    }

    pub const fn resource_generation(&self) -> Option<u64> {
        self.resource_generation
    }

    pub fn resource_batch_fingerprint(&self) -> Option<&str> {
        self.resource_batch_fingerprint.as_deref()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ResourcePoolEventDetail {
    Opened(StaticProvisioningBinding),
    Transition {
        receipt: ResourceTransitionReceipt,
        context: ResourceTransitionValidationContext,
    },
    LeaseTransition {
        receipt: ResourceLeaseTransitionReceipt,
        context: ResourceLeaseValidationContext,
    },
    Failure(ResourceFailureReceipt),
    Closed {
        ledger: Vec<ResourceLedgerEntrySnapshot>,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum UnvalidatedResourcePoolEventDetail {
    Opened(serde_json::Value),
    Transition {
        receipt: UnvalidatedResourceTransitionReceipt,
        context: serde_json::Value,
    },
    LeaseTransition {
        receipt: UnvalidatedResourceLeaseTransitionReceipt,
        context: serde_json::Value,
    },
    Failure(serde_json::Value),
    Closed {
        ledger: serde_json::Value,
    },
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(rename_all = "snake_case")]
enum UnvalidatedResourcePoolEventDetailWire {
    Opened(serde_json::Value),
    Transition {
        receipt: UnvalidatedResourceTransitionReceiptWire,
        context: serde_json::Value,
    },
    LeaseTransition {
        receipt: UnvalidatedResourceLeaseTransitionReceiptWire,
        context: serde_json::Value,
    },
    Failure(serde_json::Value),
    Closed {
        ledger: serde_json::Value,
    },
}

impl From<UnvalidatedResourcePoolEventDetailWire> for UnvalidatedResourcePoolEventDetail {
    fn from(wire: UnvalidatedResourcePoolEventDetailWire) -> Self {
        match wire {
            UnvalidatedResourcePoolEventDetailWire::Opened(value) => Self::Opened(value),
            UnvalidatedResourcePoolEventDetailWire::Transition { receipt, context } => {
                Self::Transition {
                    receipt: receipt.into(),
                    context,
                }
            }
            UnvalidatedResourcePoolEventDetailWire::LeaseTransition { receipt, context } => {
                Self::LeaseTransition {
                    receipt: receipt.into(),
                    context,
                }
            }
            UnvalidatedResourcePoolEventDetailWire::Failure(value) => Self::Failure(value),
            UnvalidatedResourcePoolEventDetailWire::Closed { ledger } => Self::Closed { ledger },
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ResourcePoolEvent {
    timestamp: MonotonicTimestamp,
    kind: ResourcePoolEventKind,
    identity: ResourcePoolEventIdentity,
    detail: ResourcePoolEventDetail,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct UnvalidatedResourcePoolEvent {
    timestamp: MonotonicTimestamp,
    kind: ResourcePoolEventKind,
    identity: UnvalidatedResourcePoolEventIdentity,
    detail: UnvalidatedResourcePoolEventDetail,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ResourcePoolEventWire {
    timestamp: MonotonicTimestamp,
    kind: ResourcePoolEventKind,
    identity: UnvalidatedResourcePoolEventIdentity,
    detail: UnvalidatedResourcePoolEventDetailWire,
}

impl From<ResourcePoolEventWire> for UnvalidatedResourcePoolEvent {
    fn from(wire: ResourcePoolEventWire) -> Self {
        Self {
            timestamp: wire.timestamp,
            kind: wire.kind,
            identity: wire.identity,
            detail: wire.detail.into(),
        }
    }
}

pub enum TrustedResourcePoolEventContext<'a> {
    Opened {
        evidence: &'a ResourcePoolEvidence,
    },
    Transition {
        evidence: &'a ResourcePoolEvidence,
        expected: &'a ResourceTransitionValidationContext,
    },
    LeaseTransition {
        evidence: &'a ResourcePoolEvidence,
        expected: &'a ResourceLeaseValidationContext,
    },
    Failure {
        evidence: &'a ResourcePoolEvidence,
        expected: &'a ResourceFailureReceipt,
    },
    Closed {
        evidence: &'a ResourcePoolEvidence,
        expected_snapshot: &'a ResourceLedgerSnapshot,
    },
}

impl<'a> TrustedResourcePoolEventContext<'a> {
    pub fn opened(evidence: &'a ResourcePoolEvidence) -> Self {
        Self::Opened { evidence }
    }

    pub fn transition(
        evidence: &'a ResourcePoolEvidence,
        expected: &'a ResourceTransitionValidationContext,
    ) -> Self {
        Self::Transition { evidence, expected }
    }

    pub fn lease_transition(
        evidence: &'a ResourcePoolEvidence,
        expected: &'a ResourceLeaseValidationContext,
    ) -> Self {
        Self::LeaseTransition { evidence, expected }
    }

    pub fn failure(
        evidence: &'a ResourcePoolEvidence,
        expected: &'a ResourceFailureReceipt,
    ) -> Self {
        Self::Failure { evidence, expected }
    }

    pub fn closed(
        evidence: &'a ResourcePoolEvidence,
        expected_snapshot: &'a ResourceLedgerSnapshot,
    ) -> Self {
        Self::Closed {
            evidence,
            expected_snapshot,
        }
    }
}

fn validate_pool_binding(
    evidence: &ResourcePoolEvidence,
    identity: &ResourceTransactionIdentity,
    admission: &StaticProvisioningBinding,
    label: &str,
) -> Result<(), VNextError> {
    if identity != evidence.provisioning_identity() || admission != evidence.admission() {
        return Err(invalid_event(format!(
            "resource pool {label} identity or admission differs from its exact pool evidence"
        )));
    }
    Ok(())
}

fn validate_failure_pool_binding(
    evidence: &ResourcePoolEvidence,
    failure: &ResourceFailureReceipt,
) -> Result<(), VNextError> {
    validate_pool_binding(evidence, failure.identity(), failure.admission(), "failure")?;
    if failure.failure().domain() != FailureDomain::Resource {
        return Err(invalid_event(
            "resource pool failure must carry the resource failure domain",
        ));
    }
    Ok(())
}

fn validate_terminal_snapshot(
    evidence: &ResourcePoolEvidence,
    snapshot: &ResourceLedgerSnapshot,
) -> Result<(), VNextError> {
    validate_pool_binding(
        evidence,
        snapshot.identity(),
        snapshot.admission(),
        "terminal snapshot",
    )?;
    if snapshot.entries().is_empty()
        || snapshot.entries().iter().any(|entry| {
            entry.entry().generation() != evidence.admission().admission_generation()
                || !matches!(
                    entry.transaction_state(),
                    ResourceTransactionState::RolledBack
                        | ResourceTransactionState::Released
                        | ResourceTransactionState::Quarantined
                )
                || entry.buffer_present()
        })
    {
        return Err(invalid_event(
            "resource pool terminal snapshot must be non-empty, generation-bound, terminal, and buffer-free",
        ));
    }
    Ok(())
}

fn item_or_batch_identity(
    entries: &[ResourceLedgerEntrySnapshot],
) -> (Option<ResourceId>, Option<u64>, Option<String>) {
    if entries.len() == 1 {
        (
            Some(entries[0].entry().resource_id().clone()),
            Some(entries[0].entry().generation()),
            None,
        )
    } else {
        (None, None, Some(canonical_fingerprint(&entries)))
    }
}

fn failure_item_or_batch_identity(
    failure: &ResourceFailureReceipt,
) -> (Option<ResourceId>, Option<u64>, Option<String>) {
    if let Some(point) = failure.failure_point() {
        (
            Some(point.resource_id().clone()),
            Some(point.generation()),
            None,
        )
    } else {
        (
            None,
            None,
            Some(canonical_fingerprint(&(
                failure.failure_id(),
                failure.completed(),
                failure.compensation(),
                failure.recovery_failures(),
                failure.recovery_strategy(),
                failure.ledger_before(),
                failure.ledger_after(),
            ))),
        )
    }
}

fn validate_transition_receipt_context(
    receipt: &ResourceTransitionReceipt,
    context: &ResourceTransitionValidationContext,
) -> Result<(), VNextError> {
    let rebuilt =
        UnvalidatedResourceTransitionReceipt::from(receipt).try_validate_against(context)?;
    if &rebuilt != receipt {
        return Err(invalid_event(
            "resource transition receipt does not match its exact ledger delta",
        ));
    }
    Ok(())
}

fn validate_lease_receipt_context(
    receipt: &ResourceLeaseTransitionReceipt,
    context: &ResourceLeaseValidationContext,
) -> Result<(), VNextError> {
    let rebuilt =
        UnvalidatedResourceLeaseTransitionReceipt::from(receipt).try_validate_against(context)?;
    if &rebuilt != receipt {
        return Err(invalid_event(
            "resource lease receipt does not match its exact ledger delta",
        ));
    }
    Ok(())
}

impl ResourcePoolEvent {
    pub fn opened(
        sequence: u64,
        timestamp: MonotonicTimestamp,
        evidence: &ResourcePoolEvidence,
    ) -> Result<Self, VNextError> {
        Ok(Self {
            timestamp,
            kind: ResourcePoolEventKind::ResourcePoolOpened,
            identity: ResourcePoolEventIdentity::for_evidence(
                sequence, evidence, None, None, None,
            )?,
            detail: ResourcePoolEventDetail::Opened(evidence.admission.clone()),
        })
    }

    pub fn transition(
        sequence: u64,
        timestamp: MonotonicTimestamp,
        evidence: &ResourcePoolEvidence,
        receipt: &ResourceTransitionReceipt,
        context: &ResourceTransitionValidationContext,
    ) -> Result<Self, VNextError> {
        validate_pool_binding(
            evidence,
            context.identity(),
            context.admission(),
            "transition",
        )?;
        validate_transition_receipt_context(receipt, context)?;
        let (resource, generation, batch) = item_or_batch_identity(context.after());
        Ok(Self {
            timestamp,
            kind: ResourcePoolEventKind::ResourceTransition,
            identity: ResourcePoolEventIdentity::for_evidence(
                sequence, evidence, resource, generation, batch,
            )?,
            detail: ResourcePoolEventDetail::Transition {
                receipt: receipt.clone(),
                context: context.clone(),
            },
        })
    }

    pub fn lease_transition(
        sequence: u64,
        timestamp: MonotonicTimestamp,
        evidence: &ResourcePoolEvidence,
        receipt: &ResourceLeaseTransitionReceipt,
        context: &ResourceLeaseValidationContext,
    ) -> Result<Self, VNextError> {
        validate_pool_binding(
            evidence,
            context.identity(),
            context.admission(),
            "lease transition",
        )?;
        validate_lease_receipt_context(receipt, context)?;
        let (resource, generation, batch) = item_or_batch_identity(context.after());
        Ok(Self {
            timestamp,
            kind: ResourcePoolEventKind::ResourceLeaseTransition,
            identity: ResourcePoolEventIdentity::for_evidence(
                sequence, evidence, resource, generation, batch,
            )?,
            detail: ResourcePoolEventDetail::LeaseTransition {
                receipt: receipt.clone(),
                context: context.clone(),
            },
        })
    }

    pub fn failed(
        sequence: u64,
        timestamp: MonotonicTimestamp,
        evidence: &ResourcePoolEvidence,
        failure: &ResourceFailureReceipt,
    ) -> Result<Self, VNextError> {
        validate_failure_pool_binding(evidence, failure)?;
        if failure.recovery_complete() {
            return Err(invalid_event(
                "ResourceFailed requires an incomplete recovery anchor",
            ));
        }
        let (resource, generation, batch) = failure_item_or_batch_identity(failure);
        Ok(Self {
            timestamp,
            kind: ResourcePoolEventKind::ResourceFailed,
            identity: ResourcePoolEventIdentity::for_evidence(
                sequence, evidence, resource, generation, batch,
            )?,
            detail: ResourcePoolEventDetail::Failure(failure.clone()),
        })
    }

    pub fn recovery_completed(
        sequence: u64,
        timestamp: MonotonicTimestamp,
        evidence: &ResourcePoolEvidence,
        recovery: &ResourceFailureReceipt,
    ) -> Result<Self, VNextError> {
        validate_failure_pool_binding(evidence, recovery)?;
        if !recovery.recovery_complete() {
            return Err(invalid_event(
                "ResourceRecoveryCompleted requires completed recovery evidence",
            ));
        }
        let (resource, generation, batch) = failure_item_or_batch_identity(recovery);
        Ok(Self {
            timestamp,
            kind: ResourcePoolEventKind::ResourceRecoveryCompleted,
            identity: ResourcePoolEventIdentity::for_evidence(
                sequence, evidence, resource, generation, batch,
            )?,
            detail: ResourcePoolEventDetail::Failure(recovery.clone()),
        })
    }

    pub fn closed(
        sequence: u64,
        timestamp: MonotonicTimestamp,
        evidence: &ResourcePoolEvidence,
        snapshot: &ResourceLedgerSnapshot,
    ) -> Result<Self, VNextError> {
        validate_terminal_snapshot(evidence, snapshot)?;
        Ok(Self {
            timestamp,
            kind: ResourcePoolEventKind::ResourcePoolClosed,
            identity: ResourcePoolEventIdentity::for_evidence(
                sequence, evidence, None, None, None,
            )?,
            detail: ResourcePoolEventDetail::Closed {
                ledger: snapshot.entries().to_vec(),
            },
        })
    }

    pub const fn timestamp(&self) -> MonotonicTimestamp {
        self.timestamp
    }

    pub const fn kind(&self) -> ResourcePoolEventKind {
        self.kind
    }

    pub fn identity(&self) -> &ResourcePoolEventIdentity {
        &self.identity
    }

    pub fn detail(&self) -> &ResourcePoolEventDetail {
        &self.detail
    }

    pub fn decode_untrusted(bytes: &[u8]) -> Result<UnvalidatedResourcePoolEvent, VNextError> {
        if bytes.len() > MAX_RESOURCE_POOL_EVENT_WIRE_BYTES {
            return Err(invalid_event(
                "untrusted resource pool event exceeds the wire byte limit",
            ));
        }
        let raw = serde_json::from_slice::<serde_json::Value>(bytes).map_err(|error| {
            VNextError::Serialization {
                context: "decode untrusted resource pool event",
                message: error.to_string(),
            }
        })?;
        let event = serde_json::from_value::<ResourcePoolEventWire>(raw.clone())
            .map(UnvalidatedResourcePoolEvent::from)
            .map_err(|error| VNextError::Serialization {
                context: "decode untrusted resource pool event",
                message: error.to_string(),
            })?;
        let canonical =
            serde_json::to_value(&event).map_err(|error| VNextError::Serialization {
                context: "serialize untrusted resource pool event",
                message: error.to_string(),
            })?;
        if canonical != raw {
            return Err(invalid_event(
                "resource pool event wire contains unknown or non-canonical nested fields",
            ));
        }
        Ok(event)
    }
}

impl UnvalidatedResourcePoolEventIdentity {
    fn matches(&self, expected: &ResourcePoolEventIdentity) -> bool {
        self.sequence == expected.sequence
            && self.pool_id == expected.pool_id
            && self.pool_identity_fingerprint == expected.pool_identity_fingerprint
            && self.provisioning_run_id == expected.provisioning_run_id
            && self.provisioning_request_id == expected.provisioning_request_id
            && self.transaction_id == expected.transaction_id
            && self.resource_id == expected.resource_id
            && self.resource_generation == expected.resource_generation
            && self.resource_batch_fingerprint == expected.resource_batch_fingerprint
    }
}

fn require_exact_wire_value(
    supplied: &serde_json::Value,
    expected: &(impl Serialize + ?Sized),
    label: &str,
) -> Result<(), VNextError> {
    let expected = serde_json::to_value(expected).map_err(|error| VNextError::Serialization {
        context: "serialize trusted resource pool event evidence",
        message: error.to_string(),
    })?;
    if supplied != &expected {
        return Err(invalid_event(format!(
            "untrusted resource pool {label} differs from independent evidence"
        )));
    }
    Ok(())
}

impl UnvalidatedResourcePoolEvent {
    pub fn revalidate(
        self,
        context: &TrustedResourcePoolEventContext<'_>,
    ) -> Result<ResourcePoolEvent, VNextError> {
        let rebuilt = match (self.kind, self.detail, context) {
            (
                ResourcePoolEventKind::ResourcePoolOpened,
                UnvalidatedResourcePoolEventDetail::Opened(admission),
                TrustedResourcePoolEventContext::Opened { evidence },
            ) => {
                require_exact_wire_value(&admission, evidence.admission(), "admission")?;
                ResourcePoolEvent::opened(self.identity.sequence, self.timestamp, evidence)?
            }
            (
                ResourcePoolEventKind::ResourceTransition,
                UnvalidatedResourcePoolEventDetail::Transition {
                    receipt,
                    context: supplied_context,
                },
                TrustedResourcePoolEventContext::Transition { evidence, expected },
            ) => {
                validate_pool_binding(
                    evidence,
                    expected.identity(),
                    expected.admission(),
                    "transition",
                )?;
                require_exact_wire_value(&supplied_context, expected, "transition context")?;
                let receipt = receipt.try_validate_against(expected)?;
                ResourcePoolEvent::transition(
                    self.identity.sequence,
                    self.timestamp,
                    evidence,
                    &receipt,
                    expected,
                )?
            }
            (
                ResourcePoolEventKind::ResourceLeaseTransition,
                UnvalidatedResourcePoolEventDetail::LeaseTransition {
                    receipt,
                    context: supplied_context,
                },
                TrustedResourcePoolEventContext::LeaseTransition { evidence, expected },
            ) => {
                validate_pool_binding(
                    evidence,
                    expected.identity(),
                    expected.admission(),
                    "lease transition",
                )?;
                require_exact_wire_value(&supplied_context, expected, "lease context")?;
                let receipt = receipt.try_validate_against(expected)?;
                ResourcePoolEvent::lease_transition(
                    self.identity.sequence,
                    self.timestamp,
                    evidence,
                    &receipt,
                    expected,
                )?
            }
            (
                kind @ (ResourcePoolEventKind::ResourceFailed
                | ResourcePoolEventKind::ResourceRecoveryCompleted),
                UnvalidatedResourcePoolEventDetail::Failure(supplied),
                TrustedResourcePoolEventContext::Failure { evidence, expected },
            ) => {
                validate_failure_pool_binding(evidence, expected)?;
                require_exact_wire_value(&supplied, expected, "failure receipt")?;
                match kind {
                    ResourcePoolEventKind::ResourceFailed => ResourcePoolEvent::failed(
                        self.identity.sequence,
                        self.timestamp,
                        evidence,
                        expected,
                    )?,
                    ResourcePoolEventKind::ResourceRecoveryCompleted => {
                        ResourcePoolEvent::recovery_completed(
                            self.identity.sequence,
                            self.timestamp,
                            evidence,
                            expected,
                        )?
                    }
                    _ => unreachable!(),
                }
            }
            (
                ResourcePoolEventKind::ResourcePoolClosed,
                UnvalidatedResourcePoolEventDetail::Closed { ledger },
                TrustedResourcePoolEventContext::Closed {
                    evidence,
                    expected_snapshot,
                },
            ) => {
                validate_terminal_snapshot(evidence, expected_snapshot)?;
                require_exact_wire_value(&ledger, expected_snapshot.entries(), "terminal ledger")?;
                ResourcePoolEvent::closed(
                    self.identity.sequence,
                    self.timestamp,
                    evidence,
                    expected_snapshot,
                )?
            }
            _ => {
                return Err(invalid_event(
                    "resource pool wire kind, detail, and external evidence do not match",
                ));
            }
        };
        if !self.identity.matches(&rebuilt.identity) {
            return Err(invalid_event(
                "resource pool wire identity differs from independently rebuilt evidence",
            ));
        }
        Ok(rebuilt)
    }
}

#[derive(Debug, Clone)]
pub struct ResourcePoolEventCursor {
    evidence: ResourcePoolEvidence,
    last_sequence: u64,
    last_timestamp: Option<MonotonicTimestamp>,
    opened: bool,
    closed: bool,
    ledgers: BTreeMap<TransactionId, Vec<ResourceLedgerEntrySnapshot>>,
    pending_failures: BTreeMap<TransactionId, ResourceFailureReceipt>,
    seen_failure_ids: BTreeSet<ResourceFailureId>,
    committed_resource_generations: Option<BTreeSet<(ResourceId, u64)>>,
}

impl ResourcePoolEventCursor {
    pub fn new(evidence: ResourcePoolEvidence) -> Self {
        Self {
            evidence,
            last_sequence: 0,
            last_timestamp: None,
            opened: false,
            closed: false,
            ledgers: BTreeMap::new(),
            pending_failures: BTreeMap::new(),
            seen_failure_ids: BTreeSet::new(),
            committed_resource_generations: None,
        }
    }

    pub fn observe(&mut self, event: &ResourcePoolEvent) -> Result<(), VNextError> {
        let mut next = self.clone();
        next.observe_inner(event)?;
        *self = next;
        Ok(())
    }

    pub const fn last_sequence(&self) -> u64 {
        self.last_sequence
    }

    pub const fn is_open(&self) -> bool {
        self.opened && !self.closed
    }

    pub const fn is_closed(&self) -> bool {
        self.closed
    }

    fn proves_active_binding(&self, active: &TrustedActiveSequenceBinding) -> bool {
        let active_resources = active
            .static_entries()
            .iter()
            .map(|entry| (entry.resource_id().clone(), entry.generation()))
            .collect::<BTreeSet<_>>();
        self.committed_resource_generations.as_ref() == Some(&active_resources)
    }

    fn observe_inner(&mut self, event: &ResourcePoolEvent) -> Result<(), VNextError> {
        if event.identity.sequence != self.last_sequence.saturating_add(1)
            || self
                .last_timestamp
                .is_some_and(|timestamp| event.timestamp <= timestamp)
            || self.closed
            || event.identity.pool_id != self.evidence.pool_id
            || event.identity.pool_identity_fingerprint != self.evidence.pool_identity_fingerprint
            || event.identity.provisioning_run_id != *self.evidence.provisioning_identity.run_id()
            || event.identity.provisioning_request_id
                != *self.evidence.provisioning_identity.request_id()
            || event.identity.transaction_id
                != *self.evidence.provisioning_identity.transaction_id()
        {
            return Err(invalid_event(
                "pool journal sequence, timestamp, lifecycle, or pool identity is invalid",
            ));
        }
        match (&event.kind, &event.detail) {
            (
                ResourcePoolEventKind::ResourcePoolOpened,
                ResourcePoolEventDetail::Opened(admission),
            ) => {
                if self.opened
                    || self.last_sequence != 0
                    || admission != &self.evidence.admission
                    || event.identity.resource_id.is_some()
                    || event.identity.resource_batch_fingerprint.is_some()
                {
                    return Err(invalid_event(
                        "ResourcePoolOpened must be the exact first admission event",
                    ));
                }
                self.opened = true;
            }
            (
                ResourcePoolEventKind::ResourceTransition,
                ResourcePoolEventDetail::Transition { receipt, context },
            ) => {
                self.require_open()?;
                validate_transition_receipt_context(receipt, context)?;
                self.validate_resource_evidence(
                    context.identity(),
                    context.admission(),
                    context.after(),
                    &event.identity,
                )?;
                self.apply_ledger(
                    context.identity().transaction_id(),
                    context.before(),
                    context.after(),
                )?;
            }
            (
                ResourcePoolEventKind::ResourceLeaseTransition,
                ResourcePoolEventDetail::LeaseTransition { receipt, context },
            ) => {
                self.require_open()?;
                validate_lease_receipt_context(receipt, context)?;
                self.validate_resource_evidence(
                    context.identity(),
                    context.admission(),
                    context.after(),
                    &event.identity,
                )?;
                self.apply_ledger(
                    context.identity().transaction_id(),
                    context.before(),
                    context.after(),
                )?;
            }
            (ResourcePoolEventKind::ResourceFailed, ResourcePoolEventDetail::Failure(failure)) => {
                self.require_open()?;
                self.validate_failure(failure, &event.identity)?;
                let transaction = failure.identity().transaction_id().clone();
                if failure.recovery_complete()
                    || self.pending_failures.contains_key(&transaction)
                    || !self.seen_failure_ids.insert(failure.failure_id())
                {
                    return Err(invalid_event(
                        "resource failure is already complete, duplicated, or has a pending anchor",
                    ));
                }
                if let Some(current) = self.ledgers.get(&transaction) {
                    if current != failure.ledger_before() {
                        return Err(invalid_event(
                            "resource failure ledger does not continue pool journal",
                        ));
                    }
                }
                self.ledgers
                    .insert(transaction.clone(), failure.ledger_after().to_vec());
                self.pending_failures.insert(transaction, failure.clone());
            }
            (
                ResourcePoolEventKind::ResourceRecoveryCompleted,
                ResourcePoolEventDetail::Failure(recovery),
            ) => {
                self.require_open()?;
                self.validate_failure(recovery, &event.identity)?;
                let transaction = recovery.identity().transaction_id().clone();
                let anchor = self.pending_failures.get(&transaction).ok_or_else(|| {
                    invalid_event("resource recovery has no exact pending failure anchor")
                })?;
                recovery.validate_recovery_continuation(anchor)?;
                if recovery.failure_id() != anchor.failure_id() || !recovery.recovery_complete() {
                    return Err(invalid_event(
                        "resource recovery failure id or completion flag is invalid",
                    ));
                }
                self.ledgers
                    .insert(transaction.clone(), recovery.ledger_after().to_vec());
                self.pending_failures.remove(&transaction);
            }
            (
                ResourcePoolEventKind::ResourcePoolClosed,
                ResourcePoolEventDetail::Closed { ledger },
            ) => {
                self.require_open()?;
                let current = self
                    .ledgers
                    .get(self.evidence.provisioning_identity.transaction_id())
                    .ok_or_else(|| {
                        invalid_event("pool closure lacks a complete transaction ledger")
                    })?;
                if ledger != current
                    || !self.pending_failures.is_empty()
                    || current.iter().any(|entry| {
                        !matches!(
                            entry.transaction_state(),
                            ResourceTransactionState::RolledBack
                                | ResourceTransactionState::Released
                                | ResourceTransactionState::Quarantined
                        ) || entry.buffer_present()
                    })
                {
                    return Err(invalid_event(
                        "pool closure requires the exact terminal ledger and no pending recovery",
                    ));
                }
                self.closed = true;
            }
            _ => {
                return Err(invalid_event(
                    "resource pool event kind and detail do not match",
                ));
            }
        }
        self.last_sequence = event.identity.sequence;
        self.last_timestamp = Some(event.timestamp);
        Ok(())
    }

    fn require_open(&self) -> Result<(), VNextError> {
        if !self.opened || self.closed {
            return Err(invalid_event("resource pool is not open"));
        }
        Ok(())
    }

    fn validate_resource_evidence(
        &self,
        identity: &ResourceTransactionIdentity,
        admission: &StaticProvisioningBinding,
        after: &[ResourceLedgerEntrySnapshot],
        event_identity: &ResourcePoolEventIdentity,
    ) -> Result<(), VNextError> {
        let expected = item_or_batch_identity(after);
        if identity != &self.evidence.provisioning_identity
            || admission != &self.evidence.admission
            || event_identity.resource_id != expected.0
            || event_identity.resource_generation != expected.1
            || event_identity.resource_batch_fingerprint != expected.2
        {
            return Err(invalid_event(
                "resource event identity differs from receipt/admission/transaction",
            ));
        }
        Ok(())
    }

    fn validate_failure(
        &self,
        failure: &ResourceFailureReceipt,
        event_identity: &ResourcePoolEventIdentity,
    ) -> Result<(), VNextError> {
        let expected = failure_item_or_batch_identity(failure);
        if failure.failure().domain() != FailureDomain::Resource
            || failure.identity() != &self.evidence.provisioning_identity
            || failure.admission() != &self.evidence.admission
            || event_identity.resource_id != expected.0
            || event_identity.resource_generation != expected.1
            || event_identity.resource_batch_fingerprint != expected.2
        {
            return Err(invalid_event(
                "resource failure event differs from its full external anchor",
            ));
        }
        Ok(())
    }

    fn apply_ledger(
        &mut self,
        transaction_id: &TransactionId,
        before: &[ResourceLedgerEntrySnapshot],
        after: &[ResourceLedgerEntrySnapshot],
    ) -> Result<(), VNextError> {
        if before.is_empty()
            || after.is_empty()
            || before.len() != after.len()
            || self
                .ledgers
                .get(transaction_id)
                .is_some_and(|current| current != before)
        {
            return Err(invalid_event(
                "resource ledger before/after does not continue exactly",
            ));
        }
        let mut resources = BTreeSet::new();
        if after.iter().any(|entry| {
            entry.entry().generation() != self.evidence.admission.admission_generation()
                || !resources.insert(entry.entry().resource_id().clone())
        }) {
            return Err(invalid_event(
                "resource ledger contains duplicate or wrong-generation entries",
            ));
        }
        if after.iter().all(|entry| {
            entry.transaction_state() == ResourceTransactionState::Committed
                && entry.buffer_present()
                && entry.entry().state() == ResourceLeaseState::Active
        }) {
            self.committed_resource_generations = Some(
                after
                    .iter()
                    .map(|entry| {
                        (
                            entry.entry().resource_id().clone(),
                            entry.entry().generation(),
                        )
                    })
                    .collect(),
            );
        }
        self.ledgers.insert(transaction_id.clone(), after.to_vec());
        Ok(())
    }
}

/// Independent evidence needed to rebuild a replay identity. None of these
/// values are accepted from the serialized replay envelope itself.
pub struct ReplayEvidence<'a> {
    resolved_plan: &'a ResolvedModelPlan,
    request_input: &'a [u8],
    initial_state: &'a [u8],
    random_seed: u64,
    request_journal: &'a [ExecutionEvent],
    active_binding: &'a TrustedActiveSequenceBinding,
    completed_binding: Option<&'a TrustedCompletedSequenceBinding>,
    aborted_binding: Option<&'a TrustedAbortedSequenceBinding>,
    cleanup_requirement: ReplayCleanupRequirement,
    plan_cleanup: ReplayPlanCleanupEvidence<'a>,
    operation_completions: &'a [OperationCompletionReceipt],
    operation_drains: &'a [CompletionDrainReceipt],
    // Quarantine retains invocation ownership and is never terminal replay evidence.
    operation_quarantines: &'a [CompletionQuarantineReceipt],
    pool_evidence: Option<&'a ResourcePoolEvidence>,
    pool_journal: &'a [ResourcePoolEvent],
}

impl<'a> ReplayEvidence<'a> {
    pub fn new(
        resolved_plan: &'a ResolvedModelPlan,
        request_input: &'a [u8],
        initial_state: &'a [u8],
        random_seed: u64,
        request_journal: &'a [ExecutionEvent],
        active_binding: &'a TrustedActiveSequenceBinding,
        completed_binding: Option<&'a TrustedCompletedSequenceBinding>,
        aborted_binding: Option<&'a TrustedAbortedSequenceBinding>,
        cleanup_requirement: ReplayCleanupRequirement,
        plan_cleanup: ReplayPlanCleanupEvidence<'a>,
        operation_completions: &'a [OperationCompletionReceipt],
        operation_drains: &'a [CompletionDrainReceipt],
        operation_quarantines: &'a [CompletionQuarantineReceipt],
        pool_evidence: &'a ResourcePoolEvidence,
        pool_journal: &'a [ResourcePoolEvent],
    ) -> Self {
        Self {
            resolved_plan,
            request_input,
            initial_state,
            random_seed,
            request_journal,
            active_binding,
            completed_binding,
            aborted_binding,
            cleanup_requirement,
            plan_cleanup,
            operation_completions,
            operation_drains,
            operation_quarantines,
            pool_evidence: Some(pool_evidence),
            pool_journal,
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new_no_static(
        resolved_plan: &'a ResolvedModelPlan,
        request_input: &'a [u8],
        initial_state: &'a [u8],
        random_seed: u64,
        request_journal: &'a [ExecutionEvent],
        active_binding: &'a TrustedActiveSequenceBinding,
        completed_binding: Option<&'a TrustedCompletedSequenceBinding>,
        aborted_binding: Option<&'a TrustedAbortedSequenceBinding>,
        cleanup_requirement: ReplayCleanupRequirement,
        plan_cleanup: ReplayPlanCleanupEvidence<'a>,
        operation_completions: &'a [OperationCompletionReceipt],
        operation_drains: &'a [CompletionDrainReceipt],
        operation_quarantines: &'a [CompletionQuarantineReceipt],
    ) -> Self {
        Self {
            resolved_plan,
            request_input,
            initial_state,
            random_seed,
            request_journal,
            active_binding,
            completed_binding,
            aborted_binding,
            cleanup_requirement,
            plan_cleanup,
            operation_completions,
            operation_drains,
            operation_quarantines,
            pool_evidence: None,
            pool_journal: &[],
        }
    }

    pub fn resolved_plan(&self) -> &ResolvedModelPlan {
        self.resolved_plan
    }

    pub fn request_input(&self) -> &[u8] {
        self.request_input
    }

    pub fn initial_state(&self) -> &[u8] {
        self.initial_state
    }

    pub const fn random_seed(&self) -> u64 {
        self.random_seed
    }

    pub fn request_journal(&self) -> &[ExecutionEvent] {
        self.request_journal
    }

    pub fn active_binding(&self) -> &TrustedActiveSequenceBinding {
        self.active_binding
    }

    pub fn completed_binding(&self) -> Option<&TrustedCompletedSequenceBinding> {
        self.completed_binding
    }

    pub fn aborted_binding(&self) -> Option<&TrustedAbortedSequenceBinding> {
        self.aborted_binding
    }

    pub const fn cleanup_requirement(&self) -> ReplayCleanupRequirement {
        self.cleanup_requirement
    }

    pub const fn plan_cleanup(&self) -> ReplayPlanCleanupEvidence<'a> {
        self.plan_cleanup
    }

    pub fn operation_completions(&self) -> &[OperationCompletionReceipt] {
        self.operation_completions
    }

    pub fn operation_drains(&self) -> &[CompletionDrainReceipt] {
        self.operation_drains
    }

    pub fn operation_quarantines(&self) -> &[CompletionQuarantineReceipt] {
        self.operation_quarantines
    }

    pub fn pool_evidence(&self) -> Option<&ResourcePoolEvidence> {
        self.pool_evidence
    }

    pub fn pool_journal(&self) -> &[ResourcePoolEvent] {
        self.pool_journal
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum ReplayOperationTerminalKey {
    Completion(usize),
    Drain(usize),
    Quarantine(usize),
}

#[derive(Clone, Copy)]
enum ReplayOperationTerminalRef<'a> {
    Completion(&'a OperationCompletionReceipt),
    Drain(&'a CompletionDrainReceipt),
    Quarantine(&'a CompletionQuarantineReceipt),
}

impl<'a> ReplayOperationTerminalRef<'a> {
    fn slot_id(self) -> CompletionSlotId {
        match self {
            Self::Completion(receipt) => receipt.submission().slot_id(),
            Self::Drain(receipt) => receipt.slot_id(),
            Self::Quarantine(receipt) => receipt.slot_id(),
        }
    }

    fn batch_identity(self) -> &'a super::BatchOperationIdentity {
        match self {
            Self::Completion(receipt) => receipt.submission().batch_identity(),
            Self::Drain(receipt) => receipt.batch_identity(),
            Self::Quarantine(receipt) => receipt.batch_identity(),
        }
    }

    fn participant_submission(
        self,
        identity: &ExecutionIdentityEnvelope,
    ) -> Option<&'a SubmittedOperationParticipantReceipt> {
        self.submission()?
            .participants()
            .iter()
            .find(|participant| participant.identity() == identity)
    }

    fn submission(self) -> Option<&'a SubmittedOperationReceipt> {
        match self {
            Self::Completion(receipt) => Some(receipt.submission()),
            Self::Drain(receipt) => receipt.submission(),
            Self::Quarantine(receipt) => receipt.submission(),
        }
    }

    fn participant_completion(
        self,
        identity: &ExecutionIdentityEnvelope,
    ) -> Option<&'a OperationParticipantCompletionReceipt> {
        match self {
            Self::Completion(receipt) => receipt.participants().iter().find(|participant| {
                same_operation_authority_except_observation(
                    identity.parts(),
                    participant.submission().identity().parts(),
                )
            }),
            Self::Drain(_) | Self::Quarantine(_) => None,
        }
    }

    fn contains_identity(self, identity: &ExecutionIdentityEnvelope) -> bool {
        self.batch_identity()
            .participants()
            .iter()
            .any(|participant| participant.identity() == identity)
    }

    fn had_submission_fence(self) -> bool {
        match self {
            Self::Completion(_) => true,
            Self::Drain(receipt) => receipt.had_submission_fence(),
            Self::Quarantine(receipt) => receipt.had_submission_fence(),
        }
    }

    fn exact_failed_completion(
        self,
        identity: &ExecutionIdentityEnvelope,
    ) -> Option<&'a IdentifiedFailure> {
        self.participant_completion(identity)
            .and_then(|participant| match participant.disposition() {
                super::OperationParticipantCompletionDisposition::FailedButQuiescent(failure) => {
                    Some(failure)
                }
                super::OperationParticipantCompletionDisposition::Succeeded
                | super::OperationParticipantCompletionDisposition::ContractFailedButQuiescent(
                    _,
                ) => None,
            })
    }

    fn participant_is_success(self, identity: &ExecutionIdentityEnvelope) -> bool {
        self.participant_completion(identity)
            .is_some_and(|participant| {
                matches!(
                    participant.disposition(),
                    super::OperationParticipantCompletionDisposition::Succeeded
                )
            })
    }
}

#[derive(Serialize)]
struct ReplayOperationTerminalFingerprint<'a> {
    completions: &'a [OperationCompletionReceipt],
    drains: &'a [CompletionDrainReceipt],
    quarantines: &'a [CompletionQuarantineReceipt],
}

fn replay_operation_terminals<'a>(
    evidence: &'a ReplayEvidence<'_>,
) -> Vec<(ReplayOperationTerminalKey, ReplayOperationTerminalRef<'a>)> {
    evidence
        .operation_completions
        .iter()
        .enumerate()
        .map(|(index, receipt)| {
            (
                ReplayOperationTerminalKey::Completion(index),
                ReplayOperationTerminalRef::Completion(receipt),
            )
        })
        .chain(
            evidence
                .operation_drains
                .iter()
                .enumerate()
                .map(|(index, receipt)| {
                    (
                        ReplayOperationTerminalKey::Drain(index),
                        ReplayOperationTerminalRef::Drain(receipt),
                    )
                }),
        )
        .chain(
            evidence
                .operation_quarantines
                .iter()
                .enumerate()
                .map(|(index, receipt)| {
                    (
                        ReplayOperationTerminalKey::Quarantine(index),
                        ReplayOperationTerminalRef::Quarantine(receipt),
                    )
                }),
        )
        .collect()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReplayCleanupRequirement {
    RequireClean,
    AllowPending,
}

/// Independent root-cleanup evidence supplied while rebuilding replay. Pending
/// is explicit and is accepted only when the caller allows pending cleanup.
/// The receipt variants are core-signed outputs and cannot be deserialized or
/// constructed by the replay caller.
#[derive(Clone, Copy)]
pub enum ReplayPlanCleanupEvidence<'a> {
    Pending,
    Closed(&'a PlanRuntimeCloseReceipt),
    Quarantined(&'a PlanRuntimeQuarantineReceipt),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReplayCleanupStatus {
    Completed,
    SequenceQuiescent,
    Quarantined,
    CleanupPending,
}

/// A replay identity is trusted output. Deserialization always goes through
/// `UnvalidatedReplayIdentity` and reconstruction from independent evidence.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ReplayIdentity {
    identity_version: ContractVersion,
    terminal_identity: ExecutionIdentityEnvelope,
    resolved_plan_fingerprint: String,
    execution_topology_fingerprint: String,
    request_input_fingerprint: String,
    initial_state_fingerprint: String,
    random_seed: u64,
    request_journal_event_count: u64,
    request_journal_fingerprint: String,
    active_sequence_fingerprint: String,
    completed_sequence_fingerprint: Option<String>,
    aborted_sequence_fingerprint: Option<String>,
    cleanup_status: ReplayCleanupStatus,
    operation_terminal_evidence_count: u64,
    operation_terminal_evidence_fingerprint: String,
    resource_pool_id: Option<ResourcePoolId>,
    resource_pool_identity_fingerprint: Option<String>,
    pool_journal_event_count: u64,
    pool_journal_fingerprint: Option<String>,
    plan_cleanup_fingerprint: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnvalidatedReplayIdentity {
    identity_version: ContractVersion,
    terminal_identity: UnvalidatedExecutionIdentityParts,
    resolved_plan_fingerprint: String,
    execution_topology_fingerprint: String,
    request_input_fingerprint: String,
    initial_state_fingerprint: String,
    random_seed: u64,
    request_journal_event_count: u64,
    request_journal_fingerprint: String,
    active_sequence_fingerprint: String,
    completed_sequence_fingerprint: Option<String>,
    aborted_sequence_fingerprint: Option<String>,
    cleanup_status: ReplayCleanupStatus,
    operation_terminal_evidence_count: u64,
    operation_terminal_evidence_fingerprint: String,
    resource_pool_id: Option<ResourcePoolId>,
    resource_pool_identity_fingerprint: Option<String>,
    pool_journal_event_count: u64,
    pool_journal_fingerprint: Option<String>,
    plan_cleanup_fingerprint: Option<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ReplayIdentityWire {
    identity_version: ContractVersion,
    terminal_identity: UnvalidatedExecutionIdentityParts,
    resolved_plan_fingerprint: String,
    execution_topology_fingerprint: String,
    request_input_fingerprint: String,
    initial_state_fingerprint: String,
    random_seed: u64,
    request_journal_event_count: u64,
    request_journal_fingerprint: String,
    active_sequence_fingerprint: String,
    completed_sequence_fingerprint: Option<String>,
    aborted_sequence_fingerprint: Option<String>,
    cleanup_status: ReplayCleanupStatus,
    operation_terminal_evidence_count: u64,
    operation_terminal_evidence_fingerprint: String,
    resource_pool_id: Option<ResourcePoolId>,
    resource_pool_identity_fingerprint: Option<String>,
    pool_journal_event_count: u64,
    pool_journal_fingerprint: Option<String>,
    plan_cleanup_fingerprint: Option<String>,
}

impl From<ReplayIdentityWire> for UnvalidatedReplayIdentity {
    fn from(wire: ReplayIdentityWire) -> Self {
        Self {
            identity_version: wire.identity_version,
            terminal_identity: wire.terminal_identity,
            resolved_plan_fingerprint: wire.resolved_plan_fingerprint,
            execution_topology_fingerprint: wire.execution_topology_fingerprint,
            request_input_fingerprint: wire.request_input_fingerprint,
            initial_state_fingerprint: wire.initial_state_fingerprint,
            random_seed: wire.random_seed,
            request_journal_event_count: wire.request_journal_event_count,
            request_journal_fingerprint: wire.request_journal_fingerprint,
            active_sequence_fingerprint: wire.active_sequence_fingerprint,
            completed_sequence_fingerprint: wire.completed_sequence_fingerprint,
            aborted_sequence_fingerprint: wire.aborted_sequence_fingerprint,
            cleanup_status: wire.cleanup_status,
            operation_terminal_evidence_count: wire.operation_terminal_evidence_count,
            operation_terminal_evidence_fingerprint: wire.operation_terminal_evidence_fingerprint,
            resource_pool_id: wire.resource_pool_id,
            resource_pool_identity_fingerprint: wire.resource_pool_identity_fingerprint,
            pool_journal_event_count: wire.pool_journal_event_count,
            pool_journal_fingerprint: wire.pool_journal_fingerprint,
            plan_cleanup_fingerprint: wire.plan_cleanup_fingerprint,
        }
    }
}

fn validate_replay_plan_cleanup(
    active: &TrustedActiveSequenceBinding,
    aborted: bool,
    requirement: ReplayCleanupRequirement,
    cleanup: ReplayPlanCleanupEvidence<'_>,
) -> Result<(ReplayCleanupStatus, Option<String>), VNextError> {
    let expected_static_resources = active.static_entries().len();
    match cleanup {
        ReplayPlanCleanupEvidence::Pending => {
            if requirement == ReplayCleanupRequirement::RequireClean {
                return Err(invalid_event(
                    "clean replay requires an exact plan close or quarantine receipt",
                ));
            }
            Ok((ReplayCleanupStatus::CleanupPending, None))
        }
        ReplayPlanCleanupEvidence::Closed(receipt) => {
            if receipt.evidence() != active.plan()
                || receipt.released_static_resources() != expected_static_resources
            {
                return Err(invalid_event(
                    "replay plan close receipt differs from the active plan or static resource set",
                ));
            }
            Ok((
                if aborted {
                    ReplayCleanupStatus::SequenceQuiescent
                } else {
                    ReplayCleanupStatus::Completed
                },
                Some(canonical_fingerprint(receipt)),
            ))
        }
        ReplayPlanCleanupEvidence::Quarantined(receipt) => {
            let accounted_static_resources = receipt
                .released_static_resources()
                .checked_add(receipt.quarantined_static_resources())
                .ok_or_else(|| invalid_event("replay quarantine resource count overflows usize"))?;
            if expected_static_resources == 0
                || receipt.evidence() != active.plan()
                || accounted_static_resources != expected_static_resources
            {
                return Err(invalid_event(
                    "replay quarantine receipt differs from the active static plan resource set",
                ));
            }
            Ok((
                ReplayCleanupStatus::Quarantined,
                Some(canonical_fingerprint(receipt)),
            ))
        }
    }
}

impl ReplayIdentity {
    pub fn from_evidence(evidence: &ReplayEvidence<'_>) -> Result<Self, VNextError> {
        if evidence.request_journal.is_empty() {
            return Err(invalid_event(
                "replay evidence requires a non-empty request journal",
            ));
        }
        if !evidence.operation_quarantines.is_empty() {
            if evidence
                .operation_quarantines
                .iter()
                .any(|receipt| !receipt.is_current())
            {
                return Err(invalid_event(
                    "completion quarantine replay evidence was superseded by a successful drain",
                ));
            }
            if evidence.cleanup_requirement != ReplayCleanupRequirement::AllowPending
                || !matches!(evidence.plan_cleanup, ReplayPlanCleanupEvidence::Pending)
            {
                return Err(invalid_event(
                    "current completion quarantine is pending ownership and requires explicitly pending replay cleanup",
                ));
            }
        }

        let topology =
            TrustedExecutionTopology::from_plan(evidence.resolved_plan.execution_plan())?;
        let active = evidence.active_binding;
        let completed = evidence.completed_binding;
        let aborted = evidence.aborted_binding;
        if completed.is_some() == aborted.is_some() {
            return Err(invalid_event(
                "replay requires exactly one external sequence completion or abort binding",
            ));
        }
        if active.plan().plan_id() != topology.plan_id()
            || active.plan().plan_hash() != topology.plan_hash()
            || active.plan().device_id() != topology.device_id()
            || active.plan().runtime_implementation_fingerprint()
                != topology.device_runtime_implementation_fingerprint()
            || active.runtime_implementation_fingerprint()
                != topology.device_runtime_implementation_fingerprint()
        {
            return Err(invalid_event(
                "replay plan and active binding do not share one authority",
            ));
        }
        if let Some(completed) = completed {
            if completed.active_sequence_fingerprint() != active.fingerprint()
                || completed.sequence_authority() != active.sequence_authority()
                || completed.run_id() != active.run_id()
                || completed.request_id() != active.request_id()
                || completed.activation_epoch() != active.activation_epoch()
                || completed.runtime_implementation_fingerprint()
                    != active.runtime_implementation_fingerprint()
            {
                return Err(invalid_event(
                    "replay completion evidence differs from its active sequence",
                ));
            }
        }
        if let Some(aborted) = aborted {
            if !active.matches_abort_disposition(aborted.disposition())
                || aborted.active_sequence_fingerprint() != active.fingerprint()
                || aborted.sequence_authority() != active.sequence_authority()
                || aborted.run_id() != active.run_id()
                || aborted.request_id() != active.request_id()
                || aborted.activation_epoch() != active.activation_epoch()
                || aborted.runtime_implementation_fingerprint()
                    != active.runtime_implementation_fingerprint()
            {
                return Err(invalid_event(
                    "replay abort evidence differs from its active sequence",
                ));
            }
        }

        let first = evidence
            .request_journal
            .first()
            .expect("non-empty request journal");
        let terminal = evidence
            .request_journal
            .last()
            .expect("non-empty request journal");
        let run_id = &first.identity.parts().run_id;
        let request_id = &first.identity.parts().request_id;
        if run_id != active.run_id() || request_id != active.request_id() {
            return Err(invalid_event(
                "replay request journal differs from the active run/request binding",
            ));
        }

        let operation_terminals = replay_operation_terminals(evidence);
        let mut terminal_slots = BTreeSet::new();
        if operation_terminals
            .iter()
            .any(|(_, terminal)| !terminal_slots.insert(terminal.slot_id()))
        {
            return Err(invalid_event(
                "replay operation terminal evidence reuses a completion slot across terminal types",
            ));
        }

        let mut request_cursor = ExecutionEventCursor::new(run_id.clone(), request_id.clone());
        let mut observed_active_identity = false;
        let mut used_submitted_terminals = BTreeSet::new();
        let mut first_failure: Option<IdentifiedFailure> = None;
        for event in evidence.request_journal {
            observed_active_identity |= has_active(event.identity.parts());
            let context = match event.kind {
                ExecutionEventKind::RequestAccepted => {
                    TrustedExecutionEventContext::pre_plan(run_id, request_id)
                }
                ExecutionEventKind::PlanBuilt => {
                    TrustedExecutionEventContext::bound(run_id, request_id, &topology)
                }
                ExecutionEventKind::FailureObserved => {
                    let failure = match &event.detail {
                        ExecutionEventDetail::Failure(failure) => failure,
                        _ => unreachable!("trusted FailureObserved shape was validated"),
                    };
                    if first_failure.is_some() {
                        return Err(invalid_event(
                            "replay request journal contains more than one first failure",
                        ));
                    }
                    first_failure = Some(failure.clone());
                    let unsubmitted_recoveries = operation_terminals
                        .iter()
                        .filter(|(_, terminal)| {
                            !terminal.had_submission_fence()
                                && terminal.contains_identity(failure.identity())
                        })
                        .collect::<Vec<_>>();
                    if unsubmitted_recoveries.len() > 1 {
                        return Err(invalid_event(
                            "operation failure matches multiple unsubmitted recovery receipts",
                        ));
                    }
                    TrustedExecutionEventContext::replay_failure(
                        run_id,
                        request_id,
                        event
                            .identity
                            .parts()
                            .plan_id
                            .is_some()
                            .then_some(&topology),
                        has_active(event.identity.parts()).then_some(active),
                        failure,
                        unsubmitted_recoveries.first().map(|_| failure.identity()),
                    )
                }
                ExecutionEventKind::RequestFailed => match &event.detail {
                    ExecutionEventDetail::Failure(failure) => {
                        TrustedExecutionEventContext::failure(
                            run_id,
                            request_id,
                            event
                                .identity
                                .parts()
                                .plan_id
                                .is_some()
                                .then_some(&topology),
                            has_active(event.identity.parts()).then_some(active),
                            failure,
                        )
                    }
                    ExecutionEventDetail::FailureTerminal { .. } => {
                        let failure = first_failure.as_ref().ok_or_else(|| {
                            invalid_event(
                                "terminal replay failure lacks its first FailureObserved evidence",
                            )
                        })?;
                        TrustedExecutionEventContext::failure_with_disposition(
                            run_id,
                            request_id,
                            &topology,
                            active,
                            has_completed(event.identity.parts())
                                .then_some(completed)
                                .flatten(),
                            has_aborted(event.identity.parts())
                                .then_some(aborted)
                                .flatten(),
                            failure,
                        )
                    }
                    _ => unreachable!("trusted RequestFailed shape was validated"),
                },
                ExecutionEventKind::OperationSubmitted => {
                    let matches = operation_terminals
                        .iter()
                        .filter_map(|(key, terminal)| {
                            terminal
                                .had_submission_fence()
                                .then(|| {
                                    terminal
                                        .participant_submission(event.identity())
                                        .map(|participant| (*key, participant))
                                })
                                .flatten()
                        })
                        .collect::<Vec<_>>();
                    if matches.len() != 1 || !used_submitted_terminals.insert(matches[0].0) {
                        return Err(invalid_event(
                            "replay operation event lacks one exact terminal receipt proving submission",
                        ));
                    }
                    TrustedExecutionEventContext::replay_operation_submitted(
                        run_id,
                        request_id,
                        &topology,
                        active,
                        operation_terminals
                            .iter()
                            .find(|(key, _)| *key == matches[0].0)
                            .and_then(|(_, terminal)| terminal.submission())
                            .expect("matched submitted participant has a batch receipt"),
                    )
                }
                ExecutionEventKind::NodeRetired => {
                    let matches = operation_terminals
                        .iter()
                        .filter_map(|(key, terminal)| {
                            terminal
                                .participant_completion(event.identity())
                                .map(|participant| (*key, participant))
                        })
                        .collect::<Vec<_>>();
                    if matches.len() != 1 || !used_submitted_terminals.contains(&matches[0].0) {
                        return Err(invalid_event(
                            "replay NodeRetired lacks the exact submitted batch completion projection",
                        ));
                    }
                    TrustedExecutionEventContext::replay_node_retired(
                        run_id,
                        request_id,
                        &topology,
                        active,
                        matches[0].1,
                    )
                }
                ExecutionEventKind::SequenceCompleted | ExecutionEventKind::RequestCompleted => {
                    let completed = completed.ok_or_else(|| {
                        invalid_event(
                            "successful replay journal lacks external sequence completion evidence",
                        )
                    })?;
                    TrustedExecutionEventContext::completed(
                        run_id, request_id, &topology, active, completed,
                    )
                }
                ExecutionEventKind::SequenceAborted => {
                    let aborted = aborted.ok_or_else(|| {
                        invalid_event(
                            "failed replay journal lacks external sequence abort evidence",
                        )
                    })?;
                    TrustedExecutionEventContext::aborted(
                        run_id, request_id, &topology, active, aborted,
                    )
                }
                _ => TrustedExecutionEventContext::active(run_id, request_id, &topology, active),
            };
            request_cursor.observe_against(event, &context)?;
        }
        if !request_cursor.is_terminal()
            || !observed_active_identity
            || !matches!(
                terminal.kind,
                ExecutionEventKind::RequestCompleted | ExecutionEventKind::RequestFailed
            )
        {
            return Err(invalid_event(
                "replay request journal is incomplete or lacks an exact terminal event",
            ));
        }
        let submitted_terminal_count = operation_terminals
            .iter()
            .filter(|(_, terminal)| terminal.had_submission_fence())
            .count();
        if used_submitted_terminals.len() != submitted_terminal_count {
            return Err(invalid_event(
                "replay contains unused or missing operation terminal evidence for submitted work",
            ));
        }

        let operation_failures = first_failure
            .iter()
            .filter(|failure| has_active(failure.identity().parts()))
            .collect::<Vec<_>>();
        let submitted_identities = evidence
            .request_journal
            .iter()
            .filter(|event| event.kind == ExecutionEventKind::OperationSubmitted)
            .map(ExecutionEvent::identity)
            .collect::<Vec<_>>();
        let mut used_operation_failures = BTreeSet::new();
        let request_failed = terminal.kind == ExecutionEventKind::RequestFailed;
        for (_, operation_terminal) in &operation_terminals {
            let mut relevant_identities = submitted_identities
                .iter()
                .copied()
                .filter(|identity| operation_terminal.contains_identity(identity))
                .collect::<Vec<_>>();
            if relevant_identities.is_empty() {
                relevant_identities.extend(
                    operation_failures
                        .iter()
                        .map(|failure| failure.identity())
                        .filter(|identity| operation_terminal.contains_identity(identity)),
                );
            }
            if relevant_identities.len() != 1 {
                return Err(invalid_event(
                    "operation terminal evidence has no unique participant projection in this request journal",
                ));
            }
            let operation_identity = relevant_identities[0];
            let matching_failures = operation_failures
                .iter()
                .enumerate()
                .filter(|(_, failure)| failure.identity() == operation_identity)
                .collect::<Vec<_>>();
            if operation_terminal.participant_is_success(operation_identity) {
                if !matching_failures.is_empty() {
                    return Err(invalid_event(
                        "successful operation completion coexists with a failure for the same submitted operation",
                    ));
                }
                continue;
            }
            if !request_failed || matching_failures.len() != 1 {
                return Err(invalid_event(
                    "non-success operation terminal evidence requires one exact operation failure and RequestFailed",
                ));
            }
            if let Some(expected_failure) =
                operation_terminal.exact_failed_completion(operation_identity)
            {
                if expected_failure.identity() != operation_identity
                    || *matching_failures[0].1 != expected_failure
                {
                    return Err(invalid_event(
                        "failed-but-quiescent completion differs from the exact observed operation failure",
                    ));
                }
            }
            if !used_operation_failures.insert(matching_failures[0].0) {
                return Err(invalid_event(
                    "one operation failure was reused by multiple terminal evidence receipts",
                ));
            }
        }
        if used_operation_failures.len() != operation_failures.len() {
            return Err(invalid_event(
                "operation FailureObserved lacks one exact non-success terminal evidence receipt",
            ));
        }

        let (
            resource_pool_id,
            resource_pool_identity_fingerprint,
            pool_journal_event_count,
            pool_journal_fingerprint,
        ) = match evidence.pool_evidence {
            Some(pool_evidence) => {
                if evidence.pool_journal.is_empty() {
                    return Err(invalid_event(
                        "static replay evidence requires a non-empty pool journal",
                    ));
                }
                let active_static_pool_id = active.static_pool_id().ok_or_else(|| {
                    invalid_event("static pool replay evidence was supplied for a no-static plan")
                })?;
                let active_static_fingerprint = active
                    .static_pool_identity_fingerprint()
                    .ok_or_else(|| invalid_event("static replay pool lacks identity evidence"))?;
                let active_static_provisioning =
                    active.static_provisioning_identity().ok_or_else(|| {
                        invalid_event("static replay pool lacks provisioning identity evidence")
                    })?;
                if active.static_entries().is_empty()
                    || active_static_pool_id != pool_evidence.pool_id
                    || active_static_fingerprint != pool_evidence.pool_identity_fingerprint
                    || active_static_provisioning != &pool_evidence.provisioning_identity
                    || pool_evidence.topology_fingerprint != topology.fingerprint()
                {
                    return Err(invalid_event(
                        "replay active binding and static pool evidence do not share one authority",
                    ));
                }
                let mut pool_cursor = ResourcePoolEventCursor::new(pool_evidence.clone());
                for event in evidence.pool_journal {
                    pool_cursor.observe(event)?;
                }
                if !pool_cursor.opened || !pool_cursor.proves_active_binding(active) {
                    return Err(invalid_event(
                        "replay pool journal does not prove the complete committed active lease",
                    ));
                }
                (
                    Some(active_static_pool_id),
                    Some(active_static_fingerprint),
                    u64::try_from(evidence.pool_journal.len())
                        .map_err(|_| invalid_event("pool journal length exceeds u64"))?,
                    Some(canonical_fingerprint(&evidence.pool_journal)),
                )
            }
            None => {
                if !evidence.pool_journal.is_empty()
                    || active.static_pool_id().is_some()
                    || active.static_provisioning_identity().is_some()
                    || active.plan().static_provisioning_binding().is_some()
                    || active.plan().static_pool_identity().is_some()
                    || !active.static_entries().is_empty()
                {
                    return Err(invalid_event(
                        "no-static replay evidence contains static pool identity or journal state",
                    ));
                }
                (None, None, 0, None)
            }
        };
        let (cleanup_status, plan_cleanup_fingerprint) = validate_replay_plan_cleanup(
            active,
            aborted.is_some(),
            evidence.cleanup_requirement,
            evidence.plan_cleanup,
        )?;

        let request_journal_event_count = u64::try_from(evidence.request_journal.len())
            .map_err(|_| invalid_event("request journal length exceeds u64"))?;
        let operation_terminal_evidence_len = evidence
            .operation_completions
            .len()
            .checked_add(evidence.operation_drains.len())
            .and_then(|count| count.checked_add(evidence.operation_quarantines.len()))
            .ok_or_else(|| invalid_event("operation terminal evidence count exceeds usize"))?;
        let operation_terminal_evidence_count = u64::try_from(operation_terminal_evidence_len)
            .map_err(|_| invalid_event("operation terminal evidence count exceeds u64"))?;
        let identity = Self {
            identity_version: EXECUTION_IDENTITY_VERSION,
            terminal_identity: terminal.identity.clone(),
            resolved_plan_fingerprint: evidence.resolved_plan.fingerprint().to_owned(),
            execution_topology_fingerprint: topology.fingerprint().to_owned(),
            request_input_fingerprint: sha256_bytes(evidence.request_input),
            initial_state_fingerprint: sha256_bytes(evidence.initial_state),
            random_seed: evidence.random_seed,
            request_journal_event_count,
            request_journal_fingerprint: canonical_fingerprint(&evidence.request_journal),
            active_sequence_fingerprint: active.fingerprint().to_owned(),
            completed_sequence_fingerprint: completed
                .map(|binding| binding.fingerprint().to_owned()),
            aborted_sequence_fingerprint: aborted.map(|binding| binding.fingerprint().to_owned()),
            cleanup_status,
            operation_terminal_evidence_count,
            operation_terminal_evidence_fingerprint: canonical_fingerprint(
                &ReplayOperationTerminalFingerprint {
                    completions: evidence.operation_completions,
                    drains: evidence.operation_drains,
                    quarantines: evidence.operation_quarantines,
                },
            ),
            resource_pool_id,
            resource_pool_identity_fingerprint,
            pool_journal_event_count,
            pool_journal_fingerprint,
            plan_cleanup_fingerprint,
        };
        identity.validate_fingerprint_shape()?;
        Ok(identity)
    }

    fn validate_fingerprint_shape(&self) -> Result<(), VNextError> {
        for (value, label) in [
            (&self.resolved_plan_fingerprint, "resolved plan fingerprint"),
            (
                &self.execution_topology_fingerprint,
                "execution topology fingerprint",
            ),
            (&self.request_input_fingerprint, "request input fingerprint"),
            (&self.initial_state_fingerprint, "initial state fingerprint"),
            (
                &self.request_journal_fingerprint,
                "request journal fingerprint",
            ),
            (
                &self.active_sequence_fingerprint,
                "active sequence fingerprint",
            ),
            (
                &self.operation_terminal_evidence_fingerprint,
                "operation terminal evidence fingerprint",
            ),
        ] {
            validate_sha256(value, label)?;
        }
        if let Some(fingerprint) = &self.completed_sequence_fingerprint {
            validate_sha256(fingerprint, "completed sequence fingerprint")?;
        }
        if let Some(fingerprint) = &self.aborted_sequence_fingerprint {
            validate_sha256(fingerprint, "aborted sequence fingerprint")?;
        }
        if let Some(fingerprint) = &self.resource_pool_identity_fingerprint {
            validate_sha256(fingerprint, "resource pool identity fingerprint")?;
        }
        if let Some(fingerprint) = &self.pool_journal_fingerprint {
            validate_sha256(fingerprint, "pool journal fingerprint")?;
        }
        if let Some(fingerprint) = &self.plan_cleanup_fingerprint {
            validate_sha256(fingerprint, "plan cleanup fingerprint")?;
        }
        let has_static_pool = self.resource_pool_id.is_some();
        if self.completed_sequence_fingerprint.is_some()
            == self.aborted_sequence_fingerprint.is_some()
            || self.cleanup_status == ReplayCleanupStatus::Completed
                && self.completed_sequence_fingerprint.is_none()
            || self.cleanup_status == ReplayCleanupStatus::SequenceQuiescent
                && self.aborted_sequence_fingerprint.is_none()
            || self.cleanup_status == ReplayCleanupStatus::CleanupPending
                && self.plan_cleanup_fingerprint.is_some()
            || self.cleanup_status != ReplayCleanupStatus::CleanupPending
                && self.plan_cleanup_fingerprint.is_none()
            || has_static_pool != self.resource_pool_identity_fingerprint.is_some()
            || has_static_pool != self.pool_journal_fingerprint.is_some()
            || has_static_pool != (self.pool_journal_event_count > 0)
        {
            return Err(invalid_event(
                "replay cleanup status differs from its exact sequence disposition",
            ));
        }
        Ok(())
    }

    pub fn terminal_identity(&self) -> &ExecutionIdentityEnvelope {
        &self.terminal_identity
    }

    pub fn resolved_plan_fingerprint(&self) -> &str {
        &self.resolved_plan_fingerprint
    }

    pub fn request_input_fingerprint(&self) -> &str {
        &self.request_input_fingerprint
    }

    pub fn initial_state_fingerprint(&self) -> &str {
        &self.initial_state_fingerprint
    }

    pub const fn random_seed(&self) -> u64 {
        self.random_seed
    }

    pub fn request_journal_fingerprint(&self) -> &str {
        &self.request_journal_fingerprint
    }

    pub fn pool_journal_fingerprint(&self) -> Option<&str> {
        self.pool_journal_fingerprint.as_deref()
    }

    pub fn plan_cleanup_fingerprint(&self) -> Option<&str> {
        self.plan_cleanup_fingerprint.as_deref()
    }

    pub const fn cleanup_status(&self) -> ReplayCleanupStatus {
        self.cleanup_status
    }

    pub fn decode_untrusted(bytes: &[u8]) -> Result<UnvalidatedReplayIdentity, VNextError> {
        if bytes.len() > MAX_REPLAY_IDENTITY_WIRE_BYTES {
            return Err(invalid_event(
                "untrusted replay identity exceeds the wire byte limit",
            ));
        }
        serde_json::from_slice::<ReplayIdentityWire>(bytes)
            .map(Into::into)
            .map_err(|error| VNextError::Serialization {
                context: "decode untrusted replay identity",
                message: error.to_string(),
            })
    }
}

impl UnvalidatedReplayIdentity {
    pub fn revalidate(self, evidence: &ReplayEvidence<'_>) -> Result<ReplayIdentity, VNextError> {
        let rebuilt = ReplayIdentity::from_evidence(evidence)?;
        let supplied_terminal = ExecutionIdentityEnvelope::new(self.terminal_identity.into())?;
        if self.identity_version != rebuilt.identity_version
            || supplied_terminal != rebuilt.terminal_identity
            || self.resolved_plan_fingerprint != rebuilt.resolved_plan_fingerprint
            || self.execution_topology_fingerprint != rebuilt.execution_topology_fingerprint
            || self.request_input_fingerprint != rebuilt.request_input_fingerprint
            || self.initial_state_fingerprint != rebuilt.initial_state_fingerprint
            || self.random_seed != rebuilt.random_seed
            || self.request_journal_event_count != rebuilt.request_journal_event_count
            || self.request_journal_fingerprint != rebuilt.request_journal_fingerprint
            || self.active_sequence_fingerprint != rebuilt.active_sequence_fingerprint
            || self.completed_sequence_fingerprint != rebuilt.completed_sequence_fingerprint
            || self.aborted_sequence_fingerprint != rebuilt.aborted_sequence_fingerprint
            || self.cleanup_status != rebuilt.cleanup_status
            || self.operation_terminal_evidence_count != rebuilt.operation_terminal_evidence_count
            || self.operation_terminal_evidence_fingerprint
                != rebuilt.operation_terminal_evidence_fingerprint
            || self.resource_pool_id != rebuilt.resource_pool_id
            || self.resource_pool_identity_fingerprint != rebuilt.resource_pool_identity_fingerprint
            || self.pool_journal_event_count != rebuilt.pool_journal_event_count
            || self.pool_journal_fingerprint != rebuilt.pool_journal_fingerprint
            || self.plan_cleanup_fingerprint != rebuilt.plan_cleanup_fingerprint
        {
            return Err(invalid_event(
                "serialized replay identity differs from independently rebuilt evidence",
            ));
        }
        rebuilt.validate_fingerprint_shape()?;
        Ok(rebuilt)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExecutionEventSinkError {
    message: String,
}

impl ExecutionEventSinkError {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }

    pub fn message(&self) -> &str {
        &self.message
    }
}

impl fmt::Display for ExecutionEventSinkError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl Error for ExecutionEventSinkError {}

mod event_sink_seal {
    pub struct Seal;
}

/// Capability created only after the emitter has validated the event against
/// its transactional cursor. External callers cannot construct this value.
pub struct EventEmissionPermit<'event> {
    event: &'event ExecutionEvent,
    _seal: event_sink_seal::Seal,
}

impl<'event> EventEmissionPermit<'event> {
    pub fn event(&self) -> &'event ExecutionEvent {
        self.event
    }
}

pub trait ExecutionEventSink: Send + Sync {
    fn is_enabled(&self, kind: ExecutionEventKind) -> bool;

    fn record(
        &self,
        event: &ExecutionEvent,
        permit: EventEmissionPermit<'_>,
    ) -> Result<(), ExecutionEventSinkError>;
}

pub struct ExecutionEventEmitter<'sink> {
    sink: &'sink dyn ExecutionEventSink,
    cursor: ExecutionEventCursor,
    sink_failed: bool,
}

impl<'sink> ExecutionEventEmitter<'sink> {
    pub fn new(
        sink: &'sink dyn ExecutionEventSink,
        run_id: RunId,
        request_id: RequestIdentity,
    ) -> Self {
        Self {
            sink,
            cursor: ExecutionEventCursor::new(run_id, request_id),
            sink_failed: false,
        }
    }

    pub fn emit(
        &mut self,
        event: &ExecutionEvent,
        context: &TrustedExecutionEventContext<'_>,
    ) -> Result<(), ExecutionEventSinkError> {
        if self.sink_failed {
            return Err(ExecutionEventSinkError::new(
                "execution event emitter is sealed after a sink failure",
            ));
        }
        let requires_open_sequence = matches!(
            event.kind,
            ExecutionEventKind::FrameStarted | ExecutionEventKind::NodeStarted
        );
        let requires_live_sequence = matches!(
            event.kind,
            ExecutionEventKind::OperationSubmitted
                | ExecutionEventKind::NodeRetired
                | ExecutionEventKind::FrameCompleted
        ) || event.kind == ExecutionEventKind::FailureObserved
            && has_active(event.identity.parts());
        if requires_open_sequence || requires_live_sequence {
            let active = context.active.ok_or_else(|| {
                ExecutionEventSinkError::new(
                    "active execution emission lacks live sequence evidence",
                )
            })?;
            let live = if requires_open_sequence {
                active.ensure_open_for_emission()
            } else {
                active.ensure_live_for_emission()
            };
            live.map_err(|error| ExecutionEventSinkError::new(error.to_string()))?;
        }
        let mut next_cursor = self.cursor.clone();
        next_cursor
            .observe_against(event, context)
            .map_err(|error| ExecutionEventSinkError::new(error.to_string()))?;
        if self.sink.is_enabled(event.kind()) {
            let permit = EventEmissionPermit {
                event,
                _seal: event_sink_seal::Seal,
            };
            if let Err(error) = self.sink.record(event, permit) {
                self.sink_failed = true;
                return Err(error);
            }
        }
        self.cursor = next_cursor;
        Ok(())
    }

    pub fn cursor(&self) -> &ExecutionEventCursor {
        &self.cursor
    }

    pub const fn sink_failed(&self) -> bool {
        self.sink_failed
    }
}

#[derive(Debug, Default)]
pub struct DisabledExecutionEventSink;

impl ExecutionEventSink for DisabledExecutionEventSink {
    fn is_enabled(&self, _kind: ExecutionEventKind) -> bool {
        false
    }

    fn record(
        &self,
        _event: &ExecutionEvent,
        _permit: EventEmissionPermit<'_>,
    ) -> Result<(), ExecutionEventSinkError> {
        Ok(())
    }
}
