use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::fmt;

use super::{
    AllocationKind, AllocationLifetime, BufferDescriptor, BufferUsage, CapacityDomainId,
    DeviceDescriptor, DeviceId, ElementType, FailureDomain, FailureEnvelope, NodeId, PlanHash,
    PlanId, RequestIdentity, ResourceAllocation, ResourceId, RunId, TransactionId, VNextError,
};

pub const MAX_RESOURCE_TRANSITION_RECEIPT_WIRE_BYTES: usize = 4 * 1024 * 1024;
pub const MAX_RESOURCE_LEASE_RECEIPT_WIRE_BYTES: usize = 4 * 1024 * 1024;
pub(super) const SEQUENCE_DISPATCH_POISONED_BIT: u64 = 1 << 63;

pub(super) fn invalid_resource(reason: impl Into<String>) -> VNextError {
    VNextError::InvalidExecutionPlan {
        reason: reason.into(),
    }
}

pub(super) fn core_resource_failure(
    code: &'static str,
    message: impl Into<String>,
    retryable: bool,
) -> FailureEnvelope {
    FailureEnvelope::new(FailureDomain::Resource, code, message, retryable)
        .expect("core-generated resource failure must be valid")
}

pub(crate) fn validate_runtime_descriptor_for_admission(
    descriptor: &DeviceDescriptor,
    admission: &StaticProvisioningBinding,
    context: &'static str,
) -> Result<(), VNextError> {
    descriptor.validate()?;
    if &descriptor.id != admission.device_id()
        || descriptor.runtime_implementation_fingerprint
            != admission.device_runtime_implementation_fingerprint()
        || descriptor.total_memory_bytes != admission.device_capacity_bytes()
    {
        return Err(invalid_resource(format!(
            "{context} runtime device, runtime implementation, or capacity differs from admission"
        )));
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ResourceTransactionIdentity {
    pub(super) pool_id: ResourcePoolId,
    pub(super) run_id: RunId,
    pub(super) transaction_id: TransactionId,
    pub(super) request_id: RequestIdentity,
}

impl ResourceTransactionIdentity {
    pub fn for_admission(
        admission: &StaticProvisioningBinding,
        run_id: RunId,
        transaction_id: TransactionId,
    ) -> Self {
        Self {
            pool_id: admission.pool_id(),
            run_id,
            transaction_id,
            request_id: admission.request_id().clone(),
        }
    }

    pub const fn pool_id(&self) -> ResourcePoolId {
        self.pool_id
    }

    pub fn run_id(&self) -> &RunId {
        &self.run_id
    }

    pub fn transaction_id(&self) -> &TransactionId {
        &self.transaction_id
    }

    pub fn request_id(&self) -> &RequestIdentity {
        &self.request_id
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResourceDriverFailure {
    failure: FailureEnvelope,
}

impl ResourceDriverFailure {
    pub fn new(failure: FailureEnvelope) -> Result<Self, VNextError> {
        failure.validate()?;
        if failure.domain() != FailureDomain::Resource {
            return Err(invalid_resource(
                "resource driver failure must use the resource failure domain",
            ));
        }
        Ok(Self { failure })
    }

    pub fn failure(&self) -> &FailureEnvelope {
        &self.failure
    }

    pub fn into_failure(self) -> FailureEnvelope {
        self.failure
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResourceTransactionState {
    New,
    Reserved,
    Committed,
    RolledBack,
    Released,
    Quarantined,
}

impl ResourceTransactionState {
    pub fn transition(
        self,
        resource_id: &ResourceId,
        action: ResourceTransactionAction,
    ) -> Result<Self, VNextError> {
        expected_transition(action, self).ok_or_else(|| VNextError::InvalidResourceTransition {
            resource_id: resource_id.to_string(),
            from: self.as_str(),
            action: action.as_str(),
        })
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::New => "new",
            Self::Reserved => "reserved",
            Self::Committed => "committed",
            Self::RolledBack => "rolled_back",
            Self::Released => "released",
            Self::Quarantined => "quarantined",
        }
    }

    pub(super) const fn is_live(self) -> bool {
        matches!(self, Self::New | Self::Reserved | Self::Committed)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResourceTransactionAction {
    Reserve,
    Commit,
    Rollback,
    Release,
    Quarantine,
}

impl ResourceTransactionAction {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Reserve => "reserve",
            Self::Commit => "commit",
            Self::Rollback => "rollback",
            Self::Release => "release",
            Self::Quarantine => "quarantine",
        }
    }
}

pub(super) const fn expected_transition(
    action: ResourceTransactionAction,
    before: ResourceTransactionState,
) -> Option<ResourceTransactionState> {
    match (before, action) {
        (ResourceTransactionState::New, ResourceTransactionAction::Reserve) => {
            Some(ResourceTransactionState::Reserved)
        }
        (ResourceTransactionState::Reserved, ResourceTransactionAction::Commit) => {
            Some(ResourceTransactionState::Committed)
        }
        (ResourceTransactionState::Reserved, ResourceTransactionAction::Rollback) => {
            Some(ResourceTransactionState::RolledBack)
        }
        (ResourceTransactionState::Committed, ResourceTransactionAction::Release) => {
            Some(ResourceTransactionState::Released)
        }
        (
            ResourceTransactionState::New
            | ResourceTransactionState::Reserved
            | ResourceTransactionState::Committed,
            ResourceTransactionAction::Quarantine,
        ) => Some(ResourceTransactionState::Quarantined),
        _ => None,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResourceCompensationAction {
    UndoReserve,
    UndoCommit,
}

impl ResourceCompensationAction {
    pub(super) const fn for_prepare_action(action: ResourceTransactionAction) -> Option<Self> {
        match action {
            ResourceTransactionAction::Reserve => Some(Self::UndoReserve),
            ResourceTransactionAction::Commit => Some(Self::UndoCommit),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResourceRecoveryStrategy {
    ReverseCompensation,
    ForwardCompletion,
    ReconcileOrQuarantine,
}

/// Core-owned retention policy derived from `AllocationLifetime`. A backend or
/// scheduler may decide when to act on it, but may not rewrite it after
/// admission.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResourceRetentionPolicy {
    Plan,
    Request,
    Sequence,
    Step,
    Invocation,
}

impl From<AllocationLifetime> for ResourceRetentionPolicy {
    fn from(lifetime: AllocationLifetime) -> Self {
        match lifetime {
            AllocationLifetime::Plan => Self::Plan,
            AllocationLifetime::Request => Self::Request,
            AllocationLifetime::Sequence => Self::Sequence,
            AllocationLifetime::Step => Self::Step,
            AllocationLifetime::Invocation => Self::Invocation,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResourceRetentionDecision {
    Retain,
    ReturnRequested,
}

/// Process-local identity of one provisioned resource pool. It is independent
/// from both the request that provisioned the pool and requests that later use
/// one of its active-sequence slots.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(try_from = "u64", into = "u64")]
pub struct ResourcePoolId(u64);

impl ResourcePoolId {
    pub(super) fn issue(generation: u64) -> Result<Self, VNextError> {
        Self::try_from(generation)
    }

    pub const fn get(self) -> u64 {
        self.0
    }
}

impl TryFrom<u64> for ResourcePoolId {
    type Error = VNextError;

    fn try_from(value: u64) -> Result<Self, Self::Error> {
        if value == 0 {
            return Err(invalid_resource("resource pool id must be non-zero"));
        }
        Ok(Self(value))
    }
}

impl From<ResourcePoolId> for u64 {
    fn from(value: ResourcePoolId) -> Self {
        value.0
    }
}

impl fmt::Display for ResourcePoolId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "resource-pool:{}", self.0)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ResourcePoolIdentity {
    pub(super) pool_id: ResourcePoolId,
    pub(super) plan_id: PlanId,
    pub(super) plan_hash: PlanHash,
    pub(super) device_id: DeviceId,
    pub(super) device_runtime_implementation_fingerprint: String,
    pub(super) admission_generation: u64,
}

impl ResourcePoolIdentity {
    pub const fn pool_id(&self) -> ResourcePoolId {
        self.pool_id
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

    pub const fn admission_generation(&self) -> u64 {
        self.admission_generation
    }
}

/// Immutable identity and capacity envelope signed into an admission permit.
/// This is trusted output and intentionally cannot be deserialized directly.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct StaticProvisioningBinding {
    pub(super) pool_identity: ResourcePoolIdentity,
    pub(super) plan_id: PlanId,
    pub(super) plan_hash: PlanHash,
    pub(super) request_id: RequestIdentity,
    pub(super) device_id: DeviceId,
    pub(super) device_runtime_implementation_fingerprint: String,
    pub(super) device_capacity_bytes: u64,
    pub(super) usable_capacity_bytes: u64,
    pub(super) plan_static_bytes: u64,
    pub(super) admitted_bytes: u64,
    pub(super) maximum_active_sequences: u32,
    pub(super) admission_generation: u64,
}

impl StaticProvisioningBinding {
    pub fn pool_identity(&self) -> &ResourcePoolIdentity {
        &self.pool_identity
    }

    pub const fn pool_id(&self) -> ResourcePoolId {
        self.pool_identity.pool_id
    }
    pub fn plan_id(&self) -> &PlanId {
        &self.plan_id
    }

    pub fn plan_hash(&self) -> &PlanHash {
        &self.plan_hash
    }

    pub fn request_id(&self) -> &RequestIdentity {
        &self.request_id
    }

    pub fn device_id(&self) -> &DeviceId {
        &self.device_id
    }

    pub fn device_runtime_implementation_fingerprint(&self) -> &str {
        &self.device_runtime_implementation_fingerprint
    }

    pub const fn device_capacity_bytes(&self) -> u64 {
        self.device_capacity_bytes
    }

    pub const fn usable_capacity_bytes(&self) -> u64 {
        self.usable_capacity_bytes
    }

    pub const fn plan_static_bytes(&self) -> u64 {
        self.plan_static_bytes
    }

    pub const fn admitted_bytes(&self) -> u64 {
        self.admitted_bytes
    }

    pub const fn maximum_active_sequences(&self) -> u32 {
        self.maximum_active_sequences
    }

    pub const fn admission_generation(&self) -> u64 {
        self.admission_generation
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ResourceReservation {
    pub(super) resource_id: ResourceId,
    pub(super) request_id: RequestIdentity,
    pub(super) owner_node_id: Option<NodeId>,
    pub(super) size_bytes: u64,
    pub(super) alignment_bytes: u64,
    pub(super) usage: BufferUsage,
    pub(super) element_type: ElementType,
    pub(super) retention_policy: ResourceRetentionPolicy,
    pub(super) backing_domain_id: Option<CapacityDomainId>,
    pub(super) generation: u64,
}

impl ResourceReservation {
    fn from_allocation(
        allocation: &ResourceAllocation,
        request_id: &RequestIdentity,
        generation: u64,
    ) -> Result<Self, VNextError> {
        let owner_node_id = match allocation.kind() {
            AllocationKind::Value => None,
            AllocationKind::Scratch { node_id, .. }
            | AllocationKind::Binding { node_id, .. }
            | AllocationKind::Persistent { node_id, .. } => Some(node_id.clone()),
        };
        if allocation.size_bytes() == 0
            || allocation.alignment_bytes() == 0
            || !allocation.alignment_bytes().is_power_of_two()
            || generation == 0
        {
            return Err(invalid_resource(format!(
                "allocation `{}` cannot be admitted",
                allocation.resource_id()
            )));
        }
        Ok(Self {
            resource_id: allocation.resource_id().clone(),
            request_id: request_id.clone(),
            owner_node_id,
            size_bytes: allocation.size_bytes(),
            alignment_bytes: allocation.alignment_bytes(),
            usage: allocation.usage(),
            element_type: allocation.element_type(),
            retention_policy: allocation.lifetime().into(),
            backing_domain_id: None,
            generation,
        })
    }

    pub fn resource_id(&self) -> &ResourceId {
        &self.resource_id
    }

    pub fn request_id(&self) -> &RequestIdentity {
        &self.request_id
    }

    pub fn owner_node_id(&self) -> Option<&NodeId> {
        self.owner_node_id.as_ref()
    }

    pub const fn size_bytes(&self) -> u64 {
        self.size_bytes
    }

    pub const fn alignment_bytes(&self) -> u64 {
        self.alignment_bytes
    }

    pub const fn usage(&self) -> BufferUsage {
        self.usage
    }

    pub const fn element_type(&self) -> ElementType {
        self.element_type
    }

    pub const fn retention_policy(&self) -> ResourceRetentionPolicy {
        self.retention_policy
    }

    pub const fn backing_domain_id(&self) -> Option<CapacityDomainId> {
        self.backing_domain_id
    }

    pub const fn generation(&self) -> u64 {
        self.generation
    }

    pub(super) fn matches_descriptor(&self, descriptor: &BufferDescriptor) -> bool {
        descriptor.resource_id == self.resource_id
            && descriptor.size_bytes == self.size_bytes
            && descriptor.alignment_bytes == self.alignment_bytes
            && descriptor.usage == self.usage
            && descriptor.element_type == self.element_type
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ResourceReservationBatch {
    request_id: RequestIdentity,
    pub(super) reservations: Vec<ResourceReservation>,
    plan_static_size_bytes: u64,
    total_size_bytes: u64,
}

impl ResourceReservationBatch {
    pub(super) fn from_allocations(
        request_id: &RequestIdentity,
        allocations: &[ResourceAllocation],
        generation: u64,
    ) -> Result<Self, VNextError> {
        let mut ids = BTreeSet::new();
        let mut reservations = Vec::with_capacity(allocations.len());
        let mut plan_static_size_bytes = 0_u64;
        for allocation in allocations {
            if !ids.insert(allocation.resource_id().clone()) {
                return Err(invalid_resource(format!(
                    "resource `{}` is duplicated in admission",
                    allocation.resource_id()
                )));
            }
            plan_static_size_bytes = plan_static_size_bytes
                .checked_add(allocation.size_bytes())
                .ok_or_else(|| invalid_resource("admitted resource bytes overflow u64"))?;
            reservations.push(ResourceReservation::from_allocation(
                allocation, request_id, generation,
            )?);
        }
        let total_size_bytes = plan_static_size_bytes;
        Ok(Self {
            request_id: request_id.clone(),
            reservations,
            plan_static_size_bytes,
            total_size_bytes,
        })
    }

    pub fn request_id(&self) -> &RequestIdentity {
        &self.request_id
    }

    pub fn reservations(&self) -> &[ResourceReservation] {
        &self.reservations
    }

    pub fn resource_ids(&self) -> impl Iterator<Item = &ResourceId> {
        self.reservations
            .iter()
            .map(ResourceReservation::resource_id)
    }

    pub const fn total_size_bytes(&self) -> u64 {
        self.total_size_bytes
    }

    pub const fn plan_static_size_bytes(&self) -> u64 {
        self.plan_static_size_bytes
    }
}
