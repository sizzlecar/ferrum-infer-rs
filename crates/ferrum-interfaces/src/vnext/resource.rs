use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::marker::PhantomData;
use std::mem::ManuallyDrop;
use std::num::NonZeroU64;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicU8, Ordering};
use std::sync::{Arc, Mutex, OnceLock, RwLock, RwLockReadGuard, Weak};
use tokio::sync::watch;

use super::{
    defer_device_cleanup, deferred_device_cleanup_status, maintain_deferred_device_cleanups,
    new_deferred_device_cleanup_domain, retire_deferred_device_cleanup_domain, AdmissionDecision,
    AdmissionDeferred, AdmissionDemand, AdmissionFitPolicy, AdmissionPreflightDecision,
    AdmissionPressureAction, AdmissionRejected, AllocationKind, AllocationLifetime,
    BatchCapacityClaimDecision, BatchInvocationId, BatchStepId, BufferDescriptor, BufferRequest,
    BufferUsage, CapacityDomainId, CapacityDomainSpec, CapacityEntry, CapacityEpochs,
    CapacityUnits, CapacityVector, CapacityWaitRecheck, CapacityWaitRegistration,
    DeferredDeviceCleanupDisposition, DeferredDeviceCleanupDomainId,
    DeferredDeviceCleanupMaintenanceReceipt, DeferredDeviceCleanupStatus,
    DeferredDeviceCleanupTask, DeviceDescriptor, DeviceId, DeviceRuntime, DynamicBackingPoolId,
    DynamicBackingPoolSpec, DynamicResourceDescriptor, DynamicResourceShape,
    DynamicStorageAllocator, DynamicStorageProfile, DynamicStorageView, ElementType,
    ExecutionFrameId, ExecutionPlan, FailureDomain, FailureEnvelope, LogicalAdmissionCoordinator,
    LogicalAdmissionCoordinatorId, LogicalAdmissionLease, LogicalBatchCapacityLease,
    LogicalRequestLease, NodeId, PlanHash, PlanId, PlanNode, RequestAdmissionDecision,
    RequestAuthorityId, RequestIdentity, ResourceAllocation, ResourceId, ResourceWorkShape, RunId,
    SequenceAuthorityId, StreamState, TokenSpanWork, TransactionId, VNextError,
};

pub const MAX_RESOURCE_TRANSITION_RECEIPT_WIRE_BYTES: usize = 4 * 1024 * 1024;
pub const MAX_RESOURCE_LEASE_RECEIPT_WIRE_BYTES: usize = 4 * 1024 * 1024;

fn invalid_resource(reason: impl Into<String>) -> VNextError {
    VNextError::InvalidExecutionPlan {
        reason: reason.into(),
    }
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

    const fn is_live(self) -> bool {
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

const fn expected_transition(
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
    const fn for_prepare_action(action: ResourceTransactionAction) -> Option<Self> {
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
    fn issue(generation: u64) -> Result<Self, VNextError> {
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
    pool_id: ResourcePoolId,
    plan_id: PlanId,
    plan_hash: PlanHash,
    device_id: DeviceId,
    device_runtime_implementation_fingerprint: String,
    admission_generation: u64,
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
    pool_identity: ResourcePoolIdentity,
    plan_id: PlanId,
    plan_hash: PlanHash,
    request_id: RequestIdentity,
    device_id: DeviceId,
    device_runtime_implementation_fingerprint: String,
    device_capacity_bytes: u64,
    usable_capacity_bytes: u64,
    plan_static_bytes: u64,
    admitted_bytes: u64,
    maximum_active_sequences: u32,
    admission_generation: u64,
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
    resource_id: ResourceId,
    request_id: RequestIdentity,
    owner_node_id: Option<NodeId>,
    size_bytes: u64,
    alignment_bytes: u64,
    usage: BufferUsage,
    element_type: ElementType,
    retention_policy: ResourceRetentionPolicy,
    backing_domain_id: Option<CapacityDomainId>,
    generation: u64,
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

    fn matches_descriptor(&self, descriptor: &BufferDescriptor) -> bool {
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
    reservations: Vec<ResourceReservation>,
    plan_static_size_bytes: u64,
    total_size_bytes: u64,
}

impl ResourceReservationBatch {
    fn from_allocations(
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

static NEXT_ADMISSION_GENERATION: AtomicU64 = AtomicU64::new(1);
static NEXT_BATCH_STEP_ID: AtomicU64 = AtomicU64::new(1);
static NEXT_BATCH_INVOCATION_ID: AtomicU64 = AtomicU64::new(1);

fn issue_batch_step_id() -> Result<BatchStepId, VNextError> {
    let value = NEXT_BATCH_STEP_ID
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
            current.checked_add(1)
        })
        .map_err(|_| invalid_resource("batch step id space is exhausted"))?;
    BatchStepId::try_from(value)
}

fn issue_batch_invocation_id() -> Result<BatchInvocationId, VNextError> {
    let value = NEXT_BATCH_INVOCATION_ID
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
            current.checked_add(1)
        })
        .map_err(|_| invalid_resource("batch invocation id space is exhausted"))?;
    BatchInvocationId::try_from(value)
}

#[derive(Debug)]
struct DeviceCapacityAccount {
    device_id: DeviceId,
    device_runtime_implementation_fingerprint: String,
    device_capacity_bytes: u64,
    state: Mutex<DeviceCapacityState>,
}

#[derive(Debug)]
struct DeviceCapacityBudgetRecord {
    /// A conservative device-wide ceiling contributed by this live plan. It is
    /// not an additive plan share.
    device_wide_usable_ceiling_bytes: u64,
    claimed_bytes: u64,
}

#[derive(Debug, Default)]
struct DeviceCapacityState {
    claimed_bytes: u64,
    next_budget_id: u64,
    budgets: BTreeMap<u64, DeviceCapacityBudgetRecord>,
}

impl DeviceCapacityAccount {
    fn register_budget(
        self: &Arc<Self>,
        device_wide_usable_ceiling_bytes: u64,
    ) -> Result<Arc<DeviceCapacityBudget>, VNextError> {
        if device_wide_usable_ceiling_bytes == 0
            || device_wide_usable_ceiling_bytes > self.device_capacity_bytes
        {
            return Err(invalid_resource(
                "device capacity budget is zero or exceeds raw device capacity",
            ));
        }
        let mut state = self
            .state
            .lock()
            .map_err(|_| invalid_resource("device capacity account is poisoned"))?;
        let effective = state
            .budgets
            .values()
            .map(|budget| budget.device_wide_usable_ceiling_bytes)
            .min()
            .map_or(device_wide_usable_ceiling_bytes, |existing| {
                existing.min(device_wide_usable_ceiling_bytes)
            });
        if state.claimed_bytes > effective {
            return Err(invalid_resource(format!(
                "device `{}` cannot register usable capacity {} below live physical claims {}",
                self.device_id, device_wide_usable_ceiling_bytes, state.claimed_bytes
            )));
        }
        let budget_id = state
            .next_budget_id
            .checked_add(1)
            .ok_or_else(|| invalid_resource("device capacity budget id space is exhausted"))?;
        state.next_budget_id = budget_id;
        state.budgets.insert(
            budget_id,
            DeviceCapacityBudgetRecord {
                device_wide_usable_ceiling_bytes,
                claimed_bytes: 0,
            },
        );
        Ok(Arc::new(DeviceCapacityBudget {
            account: Arc::clone(self),
            budget_id,
            device_wide_usable_ceiling_bytes,
        }))
    }

    fn claim(
        self: &Arc<Self>,
        budget: &Arc<DeviceCapacityBudget>,
        bytes: u64,
    ) -> Result<DeviceCapacityClaim, VNextError> {
        if bytes == 0 || !Arc::ptr_eq(self, &budget.account) {
            return Err(invalid_resource(
                "device capacity claim is empty or belongs to another account",
            ));
        }
        let mut state = self
            .state
            .lock()
            .map_err(|_| invalid_resource("device capacity account is poisoned"))?;
        let effective_usable_capacity = state
            .budgets
            .values()
            .map(|budget| budget.device_wide_usable_ceiling_bytes)
            .min()
            .ok_or_else(|| invalid_resource("device capacity account has no live budget"))?;
        let budget_record = state
            .budgets
            .get(&budget.budget_id)
            .ok_or_else(|| invalid_resource("device capacity budget is stale"))?;
        let next_budget_claimed = budget_record
            .claimed_bytes
            .checked_add(bytes)
            .filter(|next| *next <= budget.device_wide_usable_ceiling_bytes)
            .ok_or_else(|| {
                invalid_resource(format!(
                    "device `{}` plan budget exceeds usable capacity: claimed {}, requested {}, usable {}",
                    self.device_id,
                    budget_record.claimed_bytes,
                    bytes,
                    budget.device_wide_usable_ceiling_bytes
                ))
            })?;
        let next = state
            .claimed_bytes
            .checked_add(bytes)
            .filter(|next| *next <= effective_usable_capacity)
            .ok_or_else(|| {
                invalid_resource(format!(
                    "device `{}` resource admission exceeds live usable capacity: claimed {}, requested {}, effective usable {}, raw capacity {}",
                    self.device_id,
                    state.claimed_bytes,
                    bytes,
                    effective_usable_capacity,
                    self.device_capacity_bytes
                ))
            })?;
        state.claimed_bytes = next;
        state
            .budgets
            .get_mut(&budget.budget_id)
            .expect("validated device budget remains registered")
            .claimed_bytes = next_budget_claimed;
        Ok(DeviceCapacityClaim {
            budget: Some(Arc::clone(budget)),
            bytes,
        })
    }
}

struct DeviceCapacityBudget {
    account: Arc<DeviceCapacityAccount>,
    budget_id: u64,
    device_wide_usable_ceiling_bytes: u64,
}

impl Drop for DeviceCapacityBudget {
    fn drop(&mut self) {
        let mut state = match self.account.state.lock() {
            Ok(state) => state,
            Err(poisoned) => poisoned.into_inner(),
        };
        let record = state
            .budgets
            .remove(&self.budget_id)
            .expect("live device capacity budget remains registered");
        assert_eq!(
            record.claimed_bytes, 0,
            "device capacity budget dropped while physical grants remain live"
        );
    }
}

static DEVICE_CAPACITY_ACCOUNTS: OnceLock<Mutex<BTreeMap<DeviceId, Weak<DeviceCapacityAccount>>>> =
    OnceLock::new();

fn device_capacity_account(
    device_id: &DeviceId,
    device_runtime_implementation_fingerprint: &str,
    device_capacity_bytes: u64,
) -> Result<Arc<DeviceCapacityAccount>, VNextError> {
    let registry = DEVICE_CAPACITY_ACCOUNTS.get_or_init(|| Mutex::new(BTreeMap::new()));
    let mut registry = registry
        .lock()
        .map_err(|_| invalid_resource("device capacity registry is poisoned"))?;
    let account = match registry.get(device_id).and_then(Weak::upgrade) {
        Some(account) => {
            if account.device_runtime_implementation_fingerprint
                != device_runtime_implementation_fingerprint
                || account.device_capacity_bytes != device_capacity_bytes
            {
                return Err(invalid_resource(format!(
                    "device `{device_id}` has conflicting live runtime or capacity metadata"
                )));
            }
            account
        }
        None => {
            let account = Arc::new(DeviceCapacityAccount {
                device_id: device_id.clone(),
                device_runtime_implementation_fingerprint:
                    device_runtime_implementation_fingerprint.to_owned(),
                device_capacity_bytes,
                state: Mutex::new(DeviceCapacityState::default()),
            });
            registry.insert(device_id.clone(), Arc::downgrade(&account));
            account
        }
    };
    drop(registry);
    Ok(account)
}

/// Non-cloneable ownership of bytes claimed against the process-wide device
/// account. It follows the device resources, including quarantine ownership.
struct DeviceCapacityClaim {
    budget: Option<Arc<DeviceCapacityBudget>>,
    bytes: u64,
}

impl DeviceCapacityClaim {
    fn release(&mut self) {
        let bytes = self.bytes;
        self.release_bytes(bytes);
    }

    fn release_bytes(&mut self, bytes: u64) {
        if bytes == 0 {
            return;
        }
        assert!(
            self.bytes >= bytes,
            "device capacity claim was returned beyond its live bytes"
        );
        let budget = self
            .budget
            .as_ref()
            .expect("live device capacity bytes retain their plan budget");
        {
            let account = &budget.account;
            let mut state = account
                .state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            assert!(
                state.claimed_bytes >= bytes,
                "device capacity claim was returned more than once"
            );
            state.claimed_bytes -= bytes;
            let record = state
                .budgets
                .get_mut(&budget.budget_id)
                .expect("device capacity claim retains its plan budget");
            assert!(
                record.claimed_bytes >= bytes,
                "device plan capacity claim was returned more than once"
            );
            record.claimed_bytes -= bytes;
        }
        self.bytes -= bytes;
        if self.bytes == 0 {
            self.budget.take();
        }
    }

    fn bytes(&self) -> u64 {
        self.bytes
    }
}

struct DeviceCapacityReservation {
    claim: Option<DeviceCapacityClaim>,
}

impl DeviceCapacityReservation {
    fn reserve(budget: &Arc<DeviceCapacityBudget>, bytes: u64) -> Result<Self, VNextError> {
        Ok(Self {
            claim: Some(budget.account.claim(budget, bytes)?),
        })
    }

    fn commit_split(mut self, parts: &[u64]) -> Result<Vec<DeviceCapacityGrant>, VNextError> {
        let expected = parts.iter().try_fold(0_u64, |total, &bytes| {
            if bytes == 0 {
                return Err(invalid_resource(
                    "device capacity grant part must be non-zero",
                ));
            }
            total
                .checked_add(bytes)
                .ok_or_else(|| invalid_resource("device capacity grant parts overflow u64"))
        })?;
        let mut claim = self
            .claim
            .take()
            .ok_or_else(|| invalid_resource("device capacity reservation was already committed"))?;
        if claim.bytes != expected {
            return Err(invalid_resource(
                "device capacity grant parts differ from the atomic reservation",
            ));
        }
        let budget = claim
            .budget
            .take()
            .ok_or_else(|| invalid_resource("device capacity reservation has no live budget"))?;
        claim.bytes = 0;
        Ok(parts
            .iter()
            .map(|&bytes| DeviceCapacityGrant {
                claim: Some(DeviceCapacityClaim {
                    budget: Some(Arc::clone(&budget)),
                    bytes,
                }),
            })
            .collect())
    }
}

struct DeviceCapacityGrant {
    claim: Option<DeviceCapacityClaim>,
}

impl DeviceCapacityGrant {
    fn bytes(&self) -> u64 {
        self.claim.as_ref().map_or(0, DeviceCapacityClaim::bytes)
    }
}

impl Drop for DeviceCapacityClaim {
    fn drop(&mut self) {
        self.release();
    }
}

fn issue_generation() -> Result<u64, VNextError> {
    NEXT_ADMISSION_GENERATION
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
            current.checked_add(1)
        })
        .map_err(|_| invalid_resource("resource admission generation space is exhausted"))
}

/// One-shot plan/admission authority. It cannot be constructed, cloned, or
/// deserialized by product or backend code. `ResourceTransaction::begin`
/// consumes it, closing the old caller-built reservation bypass.
#[must_use = "an admission permit must be consumed by ResourceTransaction::begin"]
pub struct StaticProvisioningPermit<R>
where
    R: DeviceRuntime,
{
    maintenance_controller: DynamicPoolMaintenanceController<R>,
    dynamic_pools: Arc<DynamicPoolSet<R>>,
    reservations: ResourceReservationBatch,
    capacity_claim: DeviceCapacityClaim,
    binding: StaticProvisioningBinding,
    runtime: Arc<R>,
    seal: AdmissionSeal,
}

struct AdmissionSeal;

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
struct DynamicPoolDomainSpec {
    domain_id: CapacityDomainId,
    pool: DynamicBackingPoolSpec,
    descriptors: Vec<DynamicResourceDescriptor>,
}

impl DynamicPoolDomainSpec {
    pub const fn domain_id(&self) -> CapacityDomainId {
        self.domain_id
    }

    fn pool_id(&self) -> &DynamicBackingPoolId {
        self.pool.pool_id()
    }
}

fn plan_dynamic_pool_admission(
    maximum_active_sequences: u32,
    pools: &[DynamicBackingPoolSpec],
    descriptors: &[DynamicResourceDescriptor],
) -> Result<(LogicalAdmissionCoordinator, Vec<DynamicPoolDomainSpec>), VNextError> {
    let mut descriptors_by_id = descriptors
        .iter()
        .map(|descriptor| (descriptor.base_resource_id().clone(), descriptor.clone()))
        .collect::<BTreeMap<_, _>>();
    if descriptors_by_id.len() != descriptors.len() || pools.is_empty() != descriptors.is_empty() {
        return Err(invalid_resource(
            "dynamic pool catalog and descriptor membership are inconsistent",
        ));
    }
    let mut seen_pools = BTreeSet::new();
    let mut domains = Vec::with_capacity(pools.len());
    for (index, pool) in pools.iter().enumerate() {
        if !seen_pools.insert(pool.pool_id().clone()) {
            return Err(invalid_resource(
                "dynamic pool catalog contains a duplicate pool",
            ));
        }
        let mut members = Vec::with_capacity(pool.resource_ids().len());
        for resource_id in pool.resource_ids() {
            let descriptor = descriptors_by_id.remove(resource_id).ok_or_else(|| {
                invalid_resource("dynamic pool references an unknown or duplicate descriptor")
            })?;
            if descriptor.pool_id() != pool.pool_id() {
                return Err(invalid_resource(
                    "dynamic descriptor belongs to another core-derived pool",
                ));
            }
            members.push(descriptor);
        }
        members.sort_by(|left, right| left.base_resource_id().cmp(right.base_resource_id()));
        let domain_id = CapacityDomainId::new(
            u32::try_from(index + 1)
                .map_err(|_| invalid_resource("dynamic pool domain id exceeds u32"))?,
        )?;
        domains.push(DynamicPoolDomainSpec {
            domain_id,
            pool: pool.clone(),
            descriptors: members,
        });
    }
    if !descriptors_by_id.is_empty() {
        return Err(invalid_resource(
            "dynamic descriptors are missing from the core-derived pool catalog",
        ));
    }
    let coordinator_domains = domains
        .iter()
        .map(|domain| {
            Ok((
                domain.domain_id,
                CapacityDomainSpec::new(
                    CapacityUnits::ZERO,
                    CapacityUnits::new(domain.pool.provisioning().maximum_resident_bytes()),
                )?,
            ))
        })
        .collect::<Result<Vec<_>, VNextError>>()?;
    Ok((
        LogicalAdmissionCoordinator::new(coordinator_domains, maximum_active_sequences)?,
        domains,
    ))
}

impl<R> StaticProvisioningPermit<R>
where
    R: DeviceRuntime,
{
    pub fn maintenance_controller(&self) -> &DynamicPoolMaintenanceController<R> {
        &self.maintenance_controller
    }

    pub fn binding(&self) -> &StaticProvisioningBinding {
        &self.binding
    }

    pub fn reservations(&self) -> &ResourceReservationBatch {
        &self.reservations
    }
}

/// Explicit no-op result for plans that have no plan-lifetime buffers. It
/// binds the validated plan to one exact runtime without manufacturing an
/// empty reservation ledger or a zero-byte device-capacity claim.
#[must_use = "no-static provisioning must be retained while the plan runtime is live"]
pub struct NoStatic<R>
where
    R: DeviceRuntime,
{
    maintenance_controller: DynamicPoolMaintenanceController<R>,
    dynamic_pools: Arc<DynamicPoolSet<R>>,
    binding: StaticProvisioningBinding,
    runtime: Arc<R>,
}

impl<R> NoStatic<R>
where
    R: DeviceRuntime,
{
    pub fn maintenance_controller(&self) -> &DynamicPoolMaintenanceController<R> {
        &self.maintenance_controller
    }

    pub fn plan_id(&self) -> &PlanId {
        self.binding.plan_id()
    }

    pub fn plan_hash(&self) -> &PlanHash {
        self.binding.plan_hash()
    }

    pub fn device_id(&self) -> &DeviceId {
        self.binding.device_id()
    }

    pub fn device_runtime_implementation_fingerprint(&self) -> &str {
        self.binding.device_runtime_implementation_fingerprint()
    }

    pub const fn device_capacity_bytes(&self) -> u64 {
        self.binding.device_capacity_bytes()
    }

    pub const fn usable_capacity_bytes(&self) -> u64 {
        self.binding.usable_capacity_bytes()
    }

    pub const fn maximum_active_sequences(&self) -> u32 {
        self.binding.maximum_active_sequences()
    }
}

/// Static provisioning has two physically distinct outcomes. Only `Required`
/// carries transaction authority; `NoStatic` cannot be passed to
/// `ResourceTransaction::begin`.
#[must_use = "static provisioning must be retained or committed"]
pub enum StaticProvisioning<R>
where
    R: DeviceRuntime,
{
    NoStatic(NoStatic<R>),
    Required(StaticProvisioningPermit<R>),
}

/// The indivisible result of plan provisioning. Product code must consume
/// this owner through [`Self::into_parts`], which hands out the plan runtime
/// outcome and its unique maintenance controller together. There is no
/// controller-less extraction path.
#[must_use = "provisioned plan resources must be split into their runtime and maintenance owners"]
pub struct ProvisionedPlanResources<R>
where
    R: DeviceRuntime,
{
    provisioning: StaticProvisioning<R>,
}

/// Named result of consuming [`ProvisionedPlanResources`]. Keeping both
/// fields in product ownership prevents maintenance authority from being
/// silently discarded while request admission remains live.
#[must_use = "both plan provisioning and maintenance ownership must be retained"]
pub struct ProvisionedPlanParts<R>
where
    R: DeviceRuntime,
{
    pub provisioning: StaticProvisioning<R>,
}

impl<R> StaticProvisioning<R>
where
    R: DeviceRuntime,
{
    pub const fn has_static_resources(&self) -> bool {
        matches!(self, Self::Required(_))
    }
}

impl<R> ProvisionedPlanResources<R>
where
    R: DeviceRuntime,
{
    fn new(provisioning: StaticProvisioning<R>) -> Self {
        Self { provisioning }
    }

    pub fn provisioning(&self) -> &StaticProvisioning<R> {
        &self.provisioning
    }

    pub fn into_parts(self) -> ProvisionedPlanParts<R> {
        ProvisionedPlanParts {
            provisioning: self.provisioning,
        }
    }

    pub fn into_provisioning(self) -> StaticProvisioning<R> {
        self.provisioning
    }
}

impl ExecutionPlan {
    /// Provisions only plan-lifetime buffers. Dynamic sequence admission is a
    /// separate logical authority and is never implied by this result.
    pub fn provision_static<R>(
        &self,
        runtime: Arc<R>,
        request_id: RequestIdentity,
    ) -> Result<ProvisionedPlanResources<R>, VNextError>
    where
        R: DeviceRuntime,
    {
        let payload = self.payload();
        let memory = payload.memory();
        if runtime.descriptor().id != *payload.device_id()
            || runtime.descriptor().runtime_implementation_fingerprint
                != payload.device_runtime_implementation_fingerprint()
            || runtime.descriptor().total_memory_bytes != memory.device_capacity_bytes()
        {
            return Err(invalid_resource(
                "static provisioning runtime differs from the execution plan",
            ));
        }
        let (logical_admission, domains) = plan_dynamic_pool_admission(
            memory.maximum_active_sequences(),
            memory.dynamic_pools(),
            memory.dynamic_descriptors(),
        )?;
        let generation = issue_generation()?;
        let reservations = ResourceReservationBatch::from_allocations(
            &request_id,
            memory.static_allocations(),
            generation,
        )?;
        if memory.device_capacity_bytes() == 0
            || memory.usable_capacity_bytes() == 0
            || memory.usable_capacity_bytes() > memory.device_capacity_bytes()
            || memory.static_bytes() > memory.usable_capacity_bytes()
            || memory.maximum_active_sequences() == 0
            || (memory.static_bytes() == 0) != memory.static_allocations().is_empty()
            || reservations.plan_static_size_bytes() != memory.static_bytes()
            || reservations.total_size_bytes() != memory.static_bytes()
        {
            return Err(invalid_resource(
                "plan-static reservation or capacity evidence is invalid",
            ));
        }
        let pool_identity = ResourcePoolIdentity {
            pool_id: ResourcePoolId::issue(generation)?,
            plan_id: payload.plan_id().clone(),
            plan_hash: self.plan_hash().clone(),
            device_id: payload.device_id().clone(),
            device_runtime_implementation_fingerprint: payload
                .device_runtime_implementation_fingerprint()
                .to_owned(),
            admission_generation: generation,
        };
        let binding = StaticProvisioningBinding {
            pool_identity,
            plan_id: payload.plan_id().clone(),
            plan_hash: self.plan_hash().clone(),
            request_id,
            device_id: payload.device_id().clone(),
            device_runtime_implementation_fingerprint: payload
                .device_runtime_implementation_fingerprint()
                .to_owned(),
            device_capacity_bytes: memory.device_capacity_bytes(),
            usable_capacity_bytes: memory.usable_capacity_bytes(),
            plan_static_bytes: memory.static_bytes(),
            admitted_bytes: memory.static_bytes(),
            maximum_active_sequences: memory.maximum_active_sequences(),
            admission_generation: generation,
        };
        validate_runtime_descriptor_for_admission(
            runtime.descriptor(),
            &binding,
            "resource admission preflight",
        )?;
        let account = device_capacity_account(
            binding.device_id(),
            binding.device_runtime_implementation_fingerprint(),
            binding.device_capacity_bytes(),
        )?;
        let budget = account.register_budget(binding.usable_capacity_bytes())?;
        let nodes: Arc<[PlanNode]> = Arc::from(payload.nodes().to_vec());
        let dynamic_pools = Arc::new(DynamicPoolSet::new(
            Arc::clone(&runtime),
            binding.clone(),
            Arc::clone(&budget),
            logical_admission,
            domains,
            nodes,
        )?);
        validate_runtime_descriptor_for_admission(
            runtime.descriptor(),
            &binding,
            "resource admission completion",
        )?;
        let maintenance_controller =
            DynamicPoolMaintenanceController::new(Arc::clone(&dynamic_pools));
        if memory.static_allocations().is_empty() {
            return Ok(ProvisionedPlanResources::new(StaticProvisioning::NoStatic(
                NoStatic {
                    maintenance_controller,
                    dynamic_pools,
                    binding,
                    runtime,
                },
            )));
        }
        let capacity_claim = account.claim(&budget, reservations.total_size_bytes())?;
        Ok(ProvisionedPlanResources::new(StaticProvisioning::Required(
            StaticProvisioningPermit {
                maintenance_controller,
                dynamic_pools,
                reservations,
                capacity_claim,
                binding,
                runtime,
                seal: AdmissionSeal,
            },
        )))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ResourceTransactionIdentity {
    pool_id: ResourcePoolId,
    run_id: RunId,
    transaction_id: TransactionId,
    request_id: RequestIdentity,
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

#[derive(Debug, Clone, Copy)]
struct ResourceActionCursor {
    order: usize,
    action: ResourceTransactionAction,
    before: ResourceTransactionState,
    allocation_authorized: bool,
}

pub struct ResourceTransactionContext<'a, R>
where
    R: DeviceRuntime,
{
    runtime: &'a Arc<R>,
    identity: &'a ResourceTransactionIdentity,
    binding: &'a StaticProvisioningBinding,
    reservations: &'a ResourceReservationBatch,
    cursor: Option<ResourceActionCursor>,
    allocation_authority: Option<&'a AtomicBool>,
    pending_allocation: Option<&'a RefCell<Option<CoreOwnedAllocation<R::Buffer>>>>,
}

impl<'a, R> ResourceTransactionContext<'a, R>
where
    R: DeviceRuntime,
{
    pub fn identity(&self) -> &ResourceTransactionIdentity {
        self.identity
    }

    pub fn admission(&self) -> &StaticProvisioningBinding {
        self.binding
    }

    pub fn reservations(&self) -> &ResourceReservationBatch {
        self.reservations
    }

    fn allocation_permit<'permit>(
        &'permit self,
        request: &'permit BufferRequest,
    ) -> Result<DeviceAllocationPermit<'permit>, VNextError> {
        let cursor = self
            .cursor
            .filter(|cursor| {
                cursor.action == ResourceTransactionAction::Commit
                    && cursor.before == ResourceTransactionState::Reserved
                    && cursor.allocation_authorized
            })
            .ok_or_else(|| {
                invalid_resource("device allocation is authorized only during an exact commit")
            })?;
        let reservation = &self.reservations.reservations[cursor.order];
        if request.resource_id() != reservation.resource_id()
            || request.size_bytes() != reservation.size_bytes()
            || request.alignment_bytes() != reservation.alignment_bytes()
            || request.usage() != reservation.usage()
            || request.element_type() != reservation.element_type()
        {
            return Err(invalid_resource(
                "buffer request differs from the active admitted allocation",
            ));
        }
        self.allocation_authority
            .ok_or_else(|| invalid_resource("commit allocation authority is unavailable"))?
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .map_err(|_| {
                invalid_resource(
                    "the active resource action already consumed its allocation permit",
                )
            })?;
        Ok(DeviceAllocationPermit {
            identity: self.identity,
            binding: self.binding,
            reservation,
            request,
            seal: AllocationSeal,
        })
    }

    /// The only allocation path exposed to a transaction driver. A successful
    /// runtime allocation is installed in core-owned pending storage before
    /// this method returns. The receipt contains metadata only and is tied to
    /// this commit call, so dropping it cannot lose buffer ownership.
    pub fn allocate<'commit>(
        &'commit self,
        request: &BufferRequest,
    ) -> Result<DeviceAllocationReceipt<'commit>, DeviceAllocationError<R::Error>> {
        validate_runtime_descriptor_for_admission(
            self.runtime.descriptor(),
            self.binding,
            "allocation preflight",
        )
        .map_err(DeviceAllocationError::Contract)?;
        let permit = self
            .allocation_permit(request)
            .map_err(DeviceAllocationError::Contract)?;
        let resource_id = permit.resource_id().clone();
        let generation = permit.generation();
        let allocation = self
            .runtime
            .allocate(permit)
            .map_err(DeviceAllocationError::Runtime)?;
        let reservation = &self.reservations.reservations[self
            .cursor
            .expect("allocation permit requires an action cursor")
            .order];
        let pending = self.pending_allocation.ok_or_else(|| {
            DeviceAllocationError::Contract(invalid_resource(
                "core pending allocation storage is unavailable",
            ))
        })?;
        if pending.borrow().is_some() {
            return Err(DeviceAllocationError::Contract(invalid_resource(
                "core pending allocation storage is already occupied",
            )));
        }
        let expected_descriptor = BufferDescriptor {
            resource_id: reservation.resource_id.clone(),
            size_bytes: reservation.size_bytes,
            alignment_bytes: reservation.alignment_bytes,
            usage: reservation.usage,
            element_type: reservation.element_type,
        };
        pending.replace(Some(CoreOwnedAllocation {
            resource_id: resource_id.clone(),
            generation,
            descriptor: expected_descriptor,
            buffer: allocation,
        }));
        let descriptor = {
            let pending = pending.borrow();
            self.runtime.buffer_descriptor(
                &pending
                    .as_ref()
                    .expect("allocation was installed before descriptor inspection")
                    .buffer,
            )
        };
        pending
            .borrow_mut()
            .as_mut()
            .expect("allocation remains core-owned during descriptor inspection")
            .descriptor = descriptor.clone();
        validate_runtime_descriptor_for_admission(
            self.runtime.descriptor(),
            self.binding,
            "allocation completion",
        )
        .map_err(DeviceAllocationError::Contract)?;
        Ok(DeviceAllocationReceipt {
            resource_id,
            generation,
            descriptor,
            scope: PhantomData,
        })
    }
}

struct AllocationSeal;

#[must_use = "a device allocation permit must be consumed by DeviceRuntime::allocate"]
pub struct DeviceAllocationPermit<'a> {
    identity: &'a ResourceTransactionIdentity,
    binding: &'a StaticProvisioningBinding,
    reservation: &'a ResourceReservation,
    request: &'a BufferRequest,
    seal: AllocationSeal,
}

impl<'a> DeviceAllocationPermit<'a> {
    pub fn identity(&self) -> &ResourceTransactionIdentity {
        self.identity
    }

    pub fn admission(&self) -> &StaticProvisioningBinding {
        self.binding
    }

    pub fn reservation(&self) -> &ResourceReservation {
        self.reservation
    }

    pub fn request(&self) -> &BufferRequest {
        self.request
    }

    pub fn resource_id(&self) -> &ResourceId {
        self.reservation.resource_id()
    }

    pub const fn generation(&self) -> u64 {
        self.reservation.generation()
    }

    pub fn into_request(self) -> &'a BufferRequest {
        let _ = self.seal;
        self.request
    }
}

#[derive(Debug)]
pub enum DeviceAllocationError<E> {
    Contract(VNextError),
    Runtime(E),
}

impl<E> DeviceAllocationError<E> {
    pub fn contract_error(&self) -> Option<&VNextError> {
        match self {
            Self::Contract(error) => Some(error),
            Self::Runtime(_) => None,
        }
    }

    pub fn runtime_error(&self) -> Option<&E> {
        match self {
            Self::Contract(_) => None,
            Self::Runtime(error) => Some(error),
        }
    }
}

#[must_use = "an allocation receipt must be returned by the active commit call"]
pub struct DeviceAllocationReceipt<'commit> {
    resource_id: ResourceId,
    generation: u64,
    descriptor: BufferDescriptor,
    scope: PhantomData<&'commit mut ()>,
}

impl DeviceAllocationReceipt<'_> {
    pub fn resource_id(&self) -> &ResourceId {
        &self.resource_id
    }

    pub const fn generation(&self) -> u64 {
        self.generation
    }

    pub fn descriptor(&self) -> &BufferDescriptor {
        &self.descriptor
    }
}

struct DriverCommitAcknowledgement {
    resource_id: ResourceId,
    generation: u64,
    descriptor: BufferDescriptor,
}

impl DriverCommitAcknowledgement {
    fn from_receipt(receipt: &DeviceAllocationReceipt<'_>) -> Self {
        Self {
            resource_id: receipt.resource_id.clone(),
            generation: receipt.generation,
            descriptor: receipt.descriptor.clone(),
        }
    }

    fn matches<B>(&self, allocation: &CoreOwnedAllocation<B>) -> bool {
        self.resource_id == allocation.resource_id
            && self.generation == allocation.generation
            && self.descriptor == allocation.descriptor
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

struct CoreOwnedAllocation<B> {
    resource_id: ResourceId,
    generation: u64,
    descriptor: BufferDescriptor,
    buffer: B,
}

impl<B> CoreOwnedAllocation<B> {
    fn matches(&self, reservation: &ResourceReservation) -> bool {
        self.resource_id == reservation.resource_id
            && self.generation == reservation.generation
            && reservation.matches_descriptor(&self.descriptor)
    }
}

/// Borrowed view used to reconcile an invalid allocation. Core retains the
/// actual buffer regardless of the driver's return value.
pub struct ResourceCommitView<'a, B> {
    resource_id: &'a ResourceId,
    generation: u64,
    descriptor: &'a BufferDescriptor,
    buffer: &'a B,
}

impl<'a, B> ResourceCommitView<'a, B> {
    pub fn resource_id(&self) -> &ResourceId {
        self.resource_id
    }

    pub const fn generation(&self) -> u64 {
        self.generation
    }

    pub fn descriptor(&self) -> &BufferDescriptor {
        self.descriptor
    }

    pub fn buffer(&self) -> &B {
        self.buffer
    }
}

#[must_use = "owned quarantine buffers must be retained until backend cleanup"]
pub struct ResourceOwnedBuffer<B> {
    order: usize,
    expected_resource_id: ResourceId,
    actual_resource_id: ResourceId,
    expected_generation: u64,
    actual_generation: u64,
    expected_descriptor: BufferDescriptor,
    actual_descriptor: BufferDescriptor,
    buffer: B,
}

impl<B> ResourceOwnedBuffer<B> {
    pub fn resource_id(&self) -> &ResourceId {
        &self.expected_resource_id
    }

    pub const fn generation(&self) -> u64 {
        self.expected_generation
    }

    pub fn actual_resource_id(&self) -> &ResourceId {
        &self.actual_resource_id
    }

    pub const fn actual_generation(&self) -> u64 {
        self.actual_generation
    }

    pub fn expected_descriptor(&self) -> &BufferDescriptor {
        &self.expected_descriptor
    }

    pub fn actual_descriptor(&self) -> &BufferDescriptor {
        &self.actual_descriptor
    }

    pub fn buffer(&self) -> &B {
        &self.buffer
    }

    fn into_allocation(self) -> (usize, CoreOwnedAllocation<B>) {
        (
            self.order,
            CoreOwnedAllocation {
                resource_id: self.actual_resource_id,
                generation: self.actual_generation,
                descriptor: self.actual_descriptor,
                buffer: self.buffer,
            },
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResourceOwnershipReason {
    Quarantine,
    Abandon,
}

/// Ownership transferred out of core when normal cleanup cannot prove that
/// buffers and their device-capacity claim are gone. Dropping this object is
/// the durable owner's explicit cleanup point.
#[must_use = "resource ownership must remain durable until cleanup completes"]
pub struct ResourcePoolOwnership<R>
where
    R: DeviceRuntime,
{
    // Declaration order is the normal cleanup order. Device buffers must be
    // gone before their capacity becomes reusable, and both must precede the
    // backend runtime/context teardown.
    buffers: Vec<ResourceOwnedBuffer<R::Buffer>>,
    capacity_claim: Option<DeviceCapacityClaim>,
    pool_identity: ResourcePoolIdentity,
    reason: ResourceOwnershipReason,
    signal: Option<ResourceAbandonSignal>,
    runtime: Arc<R>,
}

impl<R> ResourcePoolOwnership<R>
where
    R: DeviceRuntime,
{
    pub fn runtime(&self) -> &R {
        &self.runtime
    }

    pub fn pool_identity(&self) -> &ResourcePoolIdentity {
        &self.pool_identity
    }

    pub const fn reason(&self) -> ResourceOwnershipReason {
        self.reason
    }

    pub fn abandon_signal(&self) -> Option<&ResourceAbandonSignal> {
        self.signal.as_ref()
    }

    pub fn buffers(&self) -> &[ResourceOwnedBuffer<R::Buffer>] {
        &self.buffers
    }

    pub fn claimed_bytes(&self) -> u64 {
        self.capacity_claim
            .as_ref()
            .map_or(0, DeviceCapacityClaim::bytes)
    }

    fn must_retain_on_drop(&self) -> bool {
        std::thread::panicking()
    }
}

impl<R> Drop for ResourcePoolOwnership<R>
where
    R: DeviceRuntime,
{
    fn drop(&mut self) {
        if !self.must_retain_on_drop() {
            return;
        }

        // A driver that drops unresolved ownership has violated the durable
        // cleanup contract. Preserve memory safety by retaining device buffers
        // and the capacity claim; the abandon signal and recovery registry made
        // the leak observable before this last-resort path.
        for buffer in std::mem::take(&mut self.buffers) {
            std::mem::forget(buffer);
        }
        if let Some(claim) = self.capacity_claim.take() {
            std::mem::forget(claim);
        }
        // Buffer and stream handles may borrow backend-owned device/context
        // state without expressing that lifetime in their Rust types. Retain
        // both owners as one unit; dropping either after intentionally
        // retaining an in-flight handle would invalidate the safety fallback.
        std::mem::forget(Arc::clone(&self.runtime));
    }
}

#[must_use = "a failed ownership transfer must be returned to core"]
pub struct ResourceOwnershipTransferFailure<R>
where
    R: DeviceRuntime,
{
    failure: ResourceDriverFailure,
    ownership: ResourcePoolOwnership<R>,
}

impl<R> ResourceOwnershipTransferFailure<R>
where
    R: DeviceRuntime,
{
    pub fn new(failure: ResourceDriverFailure, ownership: ResourcePoolOwnership<R>) -> Self {
        Self { failure, ownership }
    }

    pub fn failure(&self) -> &ResourceDriverFailure {
        &self.failure
    }

    pub fn ownership(&self) -> &ResourcePoolOwnership<R> {
        &self.ownership
    }

    fn into_parts(self) -> (ResourceDriverFailure, ResourcePoolOwnership<R>) {
        (self.failure, self.ownership)
    }
}

/// Backend adapter for one resource at a time. Core owns action ordering,
/// actual state, all buffers, receipts, and recovery progress. Methods must be
/// idempotent for the full identity/action/resource/generation key.
pub trait ResourceTransactionDriver: Send {
    type Buffer: Send + Sync + 'static;
    type Runtime: DeviceRuntime<Buffer = Self::Buffer>;

    fn runtime(&self) -> &Arc<Self::Runtime>;

    fn device_id(&self) -> &DeviceId;

    fn device_runtime_implementation_fingerprint(&self) -> &str;

    fn device_capacity_bytes(&self) -> u64;

    fn reserve_resource(
        &mut self,
        context: &ResourceTransactionContext<'_, Self::Runtime>,
        reservation: &ResourceReservation,
    ) -> Result<(), ResourceDriverFailure>;

    fn commit_resource<'commit>(
        &mut self,
        context: &'commit ResourceTransactionContext<'_, Self::Runtime>,
        reservation: &ResourceReservation,
    ) -> Result<DeviceAllocationReceipt<'commit>, ResourceDriverFailure>;

    fn compensate_reserve_resource(
        &mut self,
        context: &ResourceTransactionContext<'_, Self::Runtime>,
        reservation: &ResourceReservation,
    ) -> Result<(), ResourceDriverFailure>;

    fn compensate_commit_resource(
        &mut self,
        context: &ResourceTransactionContext<'_, Self::Runtime>,
        reservation: &ResourceReservation,
        buffer: &Self::Buffer,
    ) -> Result<(), ResourceDriverFailure>;

    fn rollback_resource(
        &mut self,
        context: &ResourceTransactionContext<'_, Self::Runtime>,
        reservation: &ResourceReservation,
    ) -> Result<(), ResourceDriverFailure>;

    fn release_resource(
        &mut self,
        context: &ResourceTransactionContext<'_, Self::Runtime>,
        reservation: &ResourceReservation,
        buffer: &Self::Buffer,
    ) -> Result<(), ResourceDriverFailure>;

    fn reconcile_commit_outcome(
        &mut self,
        context: &ResourceTransactionContext<'_, Self::Runtime>,
        expected: &ResourceReservation,
        actual: ResourceCommitView<'_, Self::Buffer>,
    ) -> Result<(), ResourceDriverFailure>;

    fn quarantine_transaction(
        &mut self,
        context: &ResourceTransactionContext<'_, Self::Runtime>,
        ownership: ResourcePoolOwnership<Self::Runtime>,
    ) -> Result<(), ResourceOwnershipTransferFailure<Self::Runtime>>;

    fn abandon_transaction(&mut self, ownership: ResourcePoolOwnership<Self::Runtime>);
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResourceLeaseState {
    Active,
    Deferred,
    Cancelled,
    Mixed,
}

impl ResourceLeaseState {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Active => "active",
            Self::Deferred => "deferred",
            Self::Cancelled => "cancelled",
            Self::Mixed => "mixed",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResourceLeaseAction {
    Defer,
    Resume,
    Cancel,
}

impl ResourceLeaseAction {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Defer => "defer",
            Self::Resume => "resume",
            Self::Cancel => "cancel",
        }
    }

    const fn decision(self) -> ResourceRetentionDecision {
        match self {
            Self::Defer | Self::Resume => ResourceRetentionDecision::Retain,
            Self::Cancel => ResourceRetentionDecision::ReturnRequested,
        }
    }
}

const fn expected_lease_transition(
    action: ResourceLeaseAction,
    before: ResourceLeaseState,
) -> Option<ResourceLeaseState> {
    match (before, action) {
        (ResourceLeaseState::Active, ResourceLeaseAction::Defer) => {
            Some(ResourceLeaseState::Deferred)
        }
        (ResourceLeaseState::Deferred, ResourceLeaseAction::Resume) => {
            Some(ResourceLeaseState::Active)
        }
        (
            ResourceLeaseState::Active | ResourceLeaseState::Deferred,
            ResourceLeaseAction::Cancel,
        ) => Some(ResourceLeaseState::Cancelled),
        _ => None,
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize)]
pub struct ResourceLeaseEntry {
    owner_node_id: Option<NodeId>,
    resource_id: ResourceId,
    size_bytes: u64,
    alignment_bytes: u64,
    usage: BufferUsage,
    element_type: ElementType,
    retention_policy: ResourceRetentionPolicy,
    generation: u64,
    state: ResourceLeaseState,
}

impl ResourceLeaseEntry {
    fn from_reservation(reservation: &ResourceReservation, state: ResourceLeaseState) -> Self {
        Self {
            owner_node_id: reservation.owner_node_id.clone(),
            resource_id: reservation.resource_id.clone(),
            size_bytes: reservation.size_bytes,
            alignment_bytes: reservation.alignment_bytes,
            usage: reservation.usage,
            element_type: reservation.element_type,
            retention_policy: reservation.retention_policy,
            generation: reservation.generation,
            state,
        }
    }

    pub fn owner_node_id(&self) -> Option<&NodeId> {
        self.owner_node_id.as_ref()
    }

    pub fn resource_id(&self) -> &ResourceId {
        &self.resource_id
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

    pub const fn generation(&self) -> u64 {
        self.generation
    }

    pub const fn state(&self) -> ResourceLeaseState {
        self.state
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct UnvalidatedResourceLeaseEntry {
    pub owner_node_id: Option<NodeId>,
    pub resource_id: ResourceId,
    pub size_bytes: u64,
    pub alignment_bytes: u64,
    pub usage: BufferUsage,
    pub element_type: ElementType,
    pub retention_policy: ResourceRetentionPolicy,
    pub generation: u64,
    pub state: ResourceLeaseState,
}

impl UnvalidatedResourceLeaseEntry {
    fn into_trusted(self) -> Result<ResourceLeaseEntry, VNextError> {
        if self.size_bytes == 0
            || self.alignment_bytes == 0
            || !self.alignment_bytes.is_power_of_two()
            || self.generation == 0
            || self.state == ResourceLeaseState::Mixed
        {
            return Err(invalid_resource("untrusted lease entry is malformed"));
        }
        Ok(ResourceLeaseEntry {
            owner_node_id: self.owner_node_id,
            resource_id: self.resource_id,
            size_bytes: self.size_bytes,
            alignment_bytes: self.alignment_bytes,
            usage: self.usage,
            element_type: self.element_type,
            retention_policy: self.retention_policy,
            generation: self.generation,
            state: self.state,
        })
    }
}

impl From<&ResourceLeaseEntry> for UnvalidatedResourceLeaseEntry {
    fn from(entry: &ResourceLeaseEntry) -> Self {
        Self {
            owner_node_id: entry.owner_node_id.clone(),
            resource_id: entry.resource_id.clone(),
            size_bytes: entry.size_bytes,
            alignment_bytes: entry.alignment_bytes,
            usage: entry.usage,
            element_type: entry.element_type,
            retention_policy: entry.retention_policy,
            generation: entry.generation,
            state: entry.state,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ResourceLedgerEntrySnapshot {
    entry: ResourceLeaseEntry,
    transaction_state: ResourceTransactionState,
    buffer_present: bool,
    actual_resource_id: Option<ResourceId>,
    actual_generation: Option<u64>,
    actual_descriptor: Option<BufferDescriptor>,
}

impl ResourceLedgerEntrySnapshot {
    pub fn entry(&self) -> &ResourceLeaseEntry {
        &self.entry
    }

    pub const fn transaction_state(&self) -> ResourceTransactionState {
        self.transaction_state
    }

    pub const fn buffer_present(&self) -> bool {
        self.buffer_present
    }

    pub fn actual_resource_id(&self) -> Option<&ResourceId> {
        self.actual_resource_id.as_ref()
    }

    pub const fn actual_generation(&self) -> Option<u64> {
        self.actual_generation
    }

    pub fn actual_descriptor(&self) -> Option<&BufferDescriptor> {
        self.actual_descriptor.as_ref()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ResourceLedgerSnapshot {
    identity: ResourceTransactionIdentity,
    admission: StaticProvisioningBinding,
    entries: Vec<ResourceLedgerEntrySnapshot>,
}

impl ResourceLedgerSnapshot {
    pub fn identity(&self) -> &ResourceTransactionIdentity {
        &self.identity
    }

    pub fn admission(&self) -> &StaticProvisioningBinding {
        &self.admission
    }

    pub fn entries(&self) -> &[ResourceLedgerEntrySnapshot] {
        &self.entries
    }
}

/// Trusted before/after journal supplied independently of an untrusted receipt.
/// There is no public constructor and it is Serialize-only.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ResourceTransitionValidationContext {
    identity: ResourceTransactionIdentity,
    admission: StaticProvisioningBinding,
    action: ResourceTransactionAction,
    before: Vec<ResourceLedgerEntrySnapshot>,
    after: Vec<ResourceLedgerEntrySnapshot>,
}

impl ResourceTransitionValidationContext {
    pub fn identity(&self) -> &ResourceTransactionIdentity {
        &self.identity
    }

    pub fn admission(&self) -> &StaticProvisioningBinding {
        &self.admission
    }

    pub const fn action(&self) -> ResourceTransactionAction {
        self.action
    }

    pub fn before(&self) -> &[ResourceLedgerEntrySnapshot] {
        &self.before
    }

    pub fn after(&self) -> &[ResourceLedgerEntrySnapshot] {
        &self.after
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ResourceLeaseValidationContext {
    identity: ResourceTransactionIdentity,
    admission: StaticProvisioningBinding,
    action: ResourceLeaseAction,
    before: Vec<ResourceLedgerEntrySnapshot>,
    after: Vec<ResourceLedgerEntrySnapshot>,
}

impl ResourceLeaseValidationContext {
    pub fn identity(&self) -> &ResourceTransactionIdentity {
        &self.identity
    }

    pub fn admission(&self) -> &StaticProvisioningBinding {
        &self.admission
    }

    pub const fn action(&self) -> ResourceLeaseAction {
        self.action
    }

    pub fn before(&self) -> &[ResourceLedgerEntrySnapshot] {
        &self.before
    }

    pub fn after(&self) -> &[ResourceLedgerEntrySnapshot] {
        &self.after
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ResourceTransitionRecord {
    run_id: RunId,
    transaction_id: TransactionId,
    request_id: RequestIdentity,
    plan_id: PlanId,
    plan_hash: PlanHash,
    device_id: DeviceId,
    admission_generation: u64,
    owner_node_id: Option<NodeId>,
    resource_id: ResourceId,
    generation: u64,
    retention_policy: ResourceRetentionPolicy,
    action: ResourceTransactionAction,
    before: ResourceTransactionState,
    after: ResourceTransactionState,
    order: u32,
}

impl ResourceTransitionRecord {
    fn from_reservation(
        identity: &ResourceTransactionIdentity,
        admission: &StaticProvisioningBinding,
        reservation: &ResourceReservation,
        action: ResourceTransactionAction,
        before: ResourceTransactionState,
        after: ResourceTransactionState,
        order: usize,
    ) -> Self {
        Self {
            run_id: identity.run_id.clone(),
            transaction_id: identity.transaction_id.clone(),
            request_id: identity.request_id.clone(),
            plan_id: admission.plan_id.clone(),
            plan_hash: admission.plan_hash.clone(),
            device_id: admission.device_id.clone(),
            admission_generation: admission.admission_generation,
            owner_node_id: reservation.owner_node_id.clone(),
            resource_id: reservation.resource_id.clone(),
            generation: reservation.generation,
            retention_policy: reservation.retention_policy,
            action,
            before,
            after,
            order: order as u32,
        }
    }

    pub fn validate(&self) -> Result<(), VNextError> {
        if self.generation == 0
            || self.admission_generation == 0
            || self.generation != self.admission_generation
            || expected_transition(self.action, self.before) != Some(self.after)
        {
            return Err(VNextError::InvalidResourceTransition {
                resource_id: self.resource_id.to_string(),
                from: self.before.as_str(),
                action: self.action.as_str(),
            });
        }
        Ok(())
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

    pub fn plan_id(&self) -> &PlanId {
        &self.plan_id
    }

    pub fn plan_hash(&self) -> &PlanHash {
        &self.plan_hash
    }

    pub fn device_id(&self) -> &DeviceId {
        &self.device_id
    }

    pub const fn admission_generation(&self) -> u64 {
        self.admission_generation
    }

    pub fn owner_node_id(&self) -> Option<&NodeId> {
        self.owner_node_id.as_ref()
    }

    pub fn resource_id(&self) -> &ResourceId {
        &self.resource_id
    }

    pub const fn generation(&self) -> u64 {
        self.generation
    }

    pub const fn retention_policy(&self) -> ResourceRetentionPolicy {
        self.retention_policy
    }

    pub const fn action(&self) -> ResourceTransactionAction {
        self.action
    }

    pub const fn before(&self) -> ResourceTransactionState {
        self.before
    }

    pub const fn after(&self) -> ResourceTransactionState {
        self.after
    }

    pub const fn order(&self) -> u32 {
        self.order
    }

    fn matches_identity_and_admission(
        &self,
        identity: &ResourceTransactionIdentity,
        admission: &StaticProvisioningBinding,
    ) -> bool {
        self.run_id == identity.run_id
            && self.transaction_id == identity.transaction_id
            && self.request_id == identity.request_id
            && self.plan_id == admission.plan_id
            && self.plan_hash == admission.plan_hash
            && self.device_id == admission.device_id
            && self.admission_generation == admission.admission_generation
    }

    fn matches_snapshot(&self, snapshot: &ResourceLedgerEntrySnapshot) -> bool {
        self.resource_id == *snapshot.entry.resource_id()
            && self.owner_node_id.as_ref() == snapshot.entry.owner_node_id()
            && self.generation == snapshot.entry.generation()
            && self.retention_policy == snapshot.entry.retention_policy()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct UnvalidatedResourceTransitionRecord {
    pub run_id: RunId,
    pub transaction_id: TransactionId,
    pub request_id: RequestIdentity,
    pub plan_id: PlanId,
    pub plan_hash: PlanHash,
    pub device_id: DeviceId,
    pub admission_generation: u64,
    pub owner_node_id: Option<NodeId>,
    pub resource_id: ResourceId,
    pub generation: u64,
    pub retention_policy: ResourceRetentionPolicy,
    pub action: ResourceTransactionAction,
    pub before: ResourceTransactionState,
    pub after: ResourceTransactionState,
    pub order: u32,
}

impl UnvalidatedResourceTransitionRecord {
    /// Structural conversion for event-level validation. Receipt-level trust
    /// still requires `UnvalidatedResourceTransitionReceipt::try_validate_against`.
    pub fn try_validate(self) -> Result<ResourceTransitionRecord, VNextError> {
        let record = ResourceTransitionRecord {
            run_id: self.run_id,
            transaction_id: self.transaction_id,
            request_id: self.request_id,
            plan_id: self.plan_id,
            plan_hash: self.plan_hash,
            device_id: self.device_id,
            admission_generation: self.admission_generation,
            owner_node_id: self.owner_node_id,
            resource_id: self.resource_id,
            generation: self.generation,
            retention_policy: self.retention_policy,
            action: self.action,
            before: self.before,
            after: self.after,
            order: self.order,
        };
        record.validate()?;
        Ok(record)
    }
}

impl From<&ResourceTransitionRecord> for UnvalidatedResourceTransitionRecord {
    fn from(record: &ResourceTransitionRecord) -> Self {
        Self {
            run_id: record.run_id.clone(),
            transaction_id: record.transaction_id.clone(),
            request_id: record.request_id.clone(),
            plan_id: record.plan_id.clone(),
            plan_hash: record.plan_hash.clone(),
            device_id: record.device_id.clone(),
            admission_generation: record.admission_generation,
            owner_node_id: record.owner_node_id.clone(),
            resource_id: record.resource_id.clone(),
            generation: record.generation,
            retention_policy: record.retention_policy,
            action: record.action,
            before: record.before,
            after: record.after,
            order: record.order,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ResourceCompensationRecord {
    run_id: RunId,
    transaction_id: TransactionId,
    request_id: RequestIdentity,
    plan_id: PlanId,
    plan_hash: PlanHash,
    owner_node_id: Option<NodeId>,
    resource_id: ResourceId,
    generation: u64,
    failed_action: ResourceTransactionAction,
    compensation_action: ResourceCompensationAction,
    before: ResourceTransactionState,
    after: ResourceTransactionState,
    compensation_order: u32,
}

impl ResourceCompensationRecord {
    fn from_transition(attempted: &ResourceTransitionRecord, compensation_order: usize) -> Self {
        Self {
            run_id: attempted.run_id.clone(),
            transaction_id: attempted.transaction_id.clone(),
            request_id: attempted.request_id.clone(),
            plan_id: attempted.plan_id.clone(),
            plan_hash: attempted.plan_hash.clone(),
            owner_node_id: attempted.owner_node_id.clone(),
            resource_id: attempted.resource_id.clone(),
            generation: attempted.generation,
            failed_action: attempted.action,
            compensation_action: ResourceCompensationAction::for_prepare_action(attempted.action)
                .expect("prepare transition has a compensation action"),
            before: attempted.after,
            after: attempted.before,
            compensation_order: compensation_order as u32,
        }
    }

    pub fn validate(&self) -> Result<(), VNextError> {
        if self.generation == 0
            || ResourceCompensationAction::for_prepare_action(self.failed_action)
                != Some(self.compensation_action)
            || expected_transition(self.failed_action, self.after) != Some(self.before)
        {
            return Err(VNextError::InvalidResourceTransition {
                resource_id: self.resource_id.to_string(),
                from: self.before.as_str(),
                action: "compensate",
            });
        }
        Ok(())
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

    pub fn plan_id(&self) -> &PlanId {
        &self.plan_id
    }

    pub fn plan_hash(&self) -> &PlanHash {
        &self.plan_hash
    }

    pub fn owner_node_id(&self) -> Option<&NodeId> {
        self.owner_node_id.as_ref()
    }

    pub fn resource_id(&self) -> &ResourceId {
        &self.resource_id
    }

    pub const fn generation(&self) -> u64 {
        self.generation
    }

    pub const fn failed_action(&self) -> ResourceTransactionAction {
        self.failed_action
    }

    pub const fn compensation_action(&self) -> ResourceCompensationAction {
        self.compensation_action
    }

    pub const fn before(&self) -> ResourceTransactionState {
        self.before
    }

    pub const fn after(&self) -> ResourceTransactionState {
        self.after
    }

    pub const fn compensation_order(&self) -> u32 {
        self.compensation_order
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct UnvalidatedResourceCompensationRecord {
    pub run_id: RunId,
    pub transaction_id: TransactionId,
    pub request_id: RequestIdentity,
    pub plan_id: PlanId,
    pub plan_hash: PlanHash,
    pub owner_node_id: Option<NodeId>,
    pub resource_id: ResourceId,
    pub generation: u64,
    pub failed_action: ResourceTransactionAction,
    pub compensation_action: ResourceCompensationAction,
    pub before: ResourceTransactionState,
    pub after: ResourceTransactionState,
    pub compensation_order: u32,
}

impl UnvalidatedResourceCompensationRecord {
    pub fn try_validate(self) -> Result<ResourceCompensationRecord, VNextError> {
        let record = ResourceCompensationRecord {
            run_id: self.run_id,
            transaction_id: self.transaction_id,
            request_id: self.request_id,
            plan_id: self.plan_id,
            plan_hash: self.plan_hash,
            owner_node_id: self.owner_node_id,
            resource_id: self.resource_id,
            generation: self.generation,
            failed_action: self.failed_action,
            compensation_action: self.compensation_action,
            before: self.before,
            after: self.after,
            compensation_order: self.compensation_order,
        };
        record.validate()?;
        Ok(record)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ResourceTransitionReceipt {
    identity: ResourceTransactionIdentity,
    admission: StaticProvisioningBinding,
    action: ResourceTransactionAction,
    records: Vec<ResourceTransitionRecord>,
}

impl ResourceTransitionReceipt {
    fn from_context(
        context: &ResourceTransitionValidationContext,
        records: Vec<ResourceTransitionRecord>,
    ) -> Result<Self, VNextError> {
        validate_transition_records_against_context(&records, context)?;
        Ok(Self {
            identity: context.identity.clone(),
            admission: context.admission.clone(),
            action: context.action,
            records,
        })
    }

    pub fn validate(&self) -> Result<(), VNextError> {
        if self.records.is_empty() {
            return Err(invalid_resource(
                "resource transition receipt must not be empty",
            ));
        }
        let mut previous_order = None;
        let mut resources = BTreeSet::new();
        for record in &self.records {
            record.validate()?;
            if !record.matches_identity_and_admission(&self.identity, &self.admission)
                || record.action != self.action
                || !resources.insert(record.resource_id.clone())
                || previous_order.is_some_and(|previous| record.order <= previous)
            {
                return Err(invalid_resource(
                    "resource receipt identity, admission, action, order, or set is invalid",
                ));
            }
            previous_order = Some(record.order);
        }
        Ok(())
    }

    pub fn identity(&self) -> &ResourceTransactionIdentity {
        &self.identity
    }

    pub fn admission(&self) -> &StaticProvisioningBinding {
        &self.admission
    }

    pub const fn action(&self) -> ResourceTransactionAction {
        self.action
    }

    pub fn records(&self) -> &[ResourceTransitionRecord] {
        &self.records
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct UnvalidatedResourceTransactionIdentity {
    pub pool_id: ResourcePoolId,
    pub run_id: RunId,
    pub transaction_id: TransactionId,
    pub request_id: RequestIdentity,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct UnvalidatedStaticProvisioningBinding {
    pub pool_identity: UnvalidatedResourcePoolIdentity,
    pub plan_id: PlanId,
    pub plan_hash: PlanHash,
    pub request_id: RequestIdentity,
    pub device_id: DeviceId,
    pub device_runtime_implementation_fingerprint: String,
    pub device_capacity_bytes: u64,
    pub usable_capacity_bytes: u64,
    pub plan_static_bytes: u64,
    pub admitted_bytes: u64,
    pub maximum_active_sequences: u32,
    pub admission_generation: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct UnvalidatedResourcePoolIdentity {
    pub pool_id: ResourcePoolId,
    pub plan_id: PlanId,
    pub plan_hash: PlanHash,
    pub device_id: DeviceId,
    pub device_runtime_implementation_fingerprint: String,
    pub admission_generation: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct UnvalidatedResourceTransitionReceipt {
    pub identity: UnvalidatedResourceTransactionIdentity,
    pub admission: UnvalidatedStaticProvisioningBinding,
    pub action: ResourceTransactionAction,
    pub records: Vec<UnvalidatedResourceTransitionRecord>,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct UnvalidatedResourceTransitionReceiptWire {
    identity: UnvalidatedResourceTransactionIdentity,
    admission: UnvalidatedStaticProvisioningBinding,
    action: ResourceTransactionAction,
    records: Vec<UnvalidatedResourceTransitionRecord>,
}

impl From<UnvalidatedResourceTransitionReceiptWire> for UnvalidatedResourceTransitionReceipt {
    fn from(wire: UnvalidatedResourceTransitionReceiptWire) -> Self {
        Self {
            identity: wire.identity,
            admission: wire.admission,
            action: wire.action,
            records: wire.records,
        }
    }
}

impl UnvalidatedResourceTransitionReceipt {
    pub fn decode_untrusted(bytes: &[u8]) -> Result<Self, VNextError> {
        if bytes.len() > MAX_RESOURCE_TRANSITION_RECEIPT_WIRE_BYTES {
            return Err(VNextError::Serialization {
                context: "decode untrusted resource transition receipt",
                message: format!(
                    "resource transition receipt wire size {} exceeds limit {}",
                    bytes.len(),
                    MAX_RESOURCE_TRANSITION_RECEIPT_WIRE_BYTES
                ),
            });
        }
        serde_json::from_slice::<UnvalidatedResourceTransitionReceiptWire>(bytes)
            .map(Self::from)
            .map_err(|error| VNextError::Serialization {
                context: "decode untrusted resource transition receipt",
                message: error.to_string(),
            })
    }

    /// Deliberately refuses self-validation. The expected plan/admission and
    /// before/after ledger must come from a separate trusted journal.
    pub fn try_validate(self) -> Result<ResourceTransitionReceipt, VNextError> {
        let _ = self;
        Err(invalid_resource(
            "untrusted resource receipt requires a trusted ledger context",
        ))
    }

    pub fn try_validate_against(
        self,
        expected: &ResourceTransitionValidationContext,
    ) -> Result<ResourceTransitionReceipt, VNextError> {
        if self.identity.pool_id != expected.identity.pool_id
            || self.identity.run_id != expected.identity.run_id
            || self.identity.transaction_id != expected.identity.transaction_id
            || self.identity.request_id != expected.identity.request_id
            || self.admission.pool_identity.pool_id != expected.admission.pool_identity.pool_id
            || self.admission.pool_identity.plan_id != expected.admission.pool_identity.plan_id
            || self.admission.pool_identity.plan_hash != expected.admission.pool_identity.plan_hash
            || self.admission.pool_identity.device_id != expected.admission.pool_identity.device_id
            || self
                .admission
                .pool_identity
                .device_runtime_implementation_fingerprint
                != expected
                    .admission
                    .pool_identity
                    .device_runtime_implementation_fingerprint
            || self.admission.pool_identity.admission_generation
                != expected.admission.pool_identity.admission_generation
            || self.admission.plan_id != expected.admission.plan_id
            || self.admission.plan_hash != expected.admission.plan_hash
            || self.admission.request_id != expected.admission.request_id
            || self.admission.device_id != expected.admission.device_id
            || self.admission.device_runtime_implementation_fingerprint
                != expected.admission.device_runtime_implementation_fingerprint
            || self.admission.device_capacity_bytes != expected.admission.device_capacity_bytes
            || self.admission.usable_capacity_bytes != expected.admission.usable_capacity_bytes
            || self.admission.plan_static_bytes != expected.admission.plan_static_bytes
            || self.admission.admitted_bytes != expected.admission.admitted_bytes
            || self.admission.maximum_active_sequences
                != expected.admission.maximum_active_sequences
            || self.admission.admission_generation != expected.admission.admission_generation
            || self.action != expected.action
        {
            return Err(invalid_resource(
                "untrusted resource receipt does not match expected identity or admission",
            ));
        }
        let records = self
            .records
            .into_iter()
            .map(UnvalidatedResourceTransitionRecord::try_validate)
            .collect::<Result<Vec<_>, _>>()?;
        ResourceTransitionReceipt::from_context(expected, records)
    }
}

impl From<&ResourceTransitionReceipt> for UnvalidatedResourceTransitionReceipt {
    fn from(receipt: &ResourceTransitionReceipt) -> Self {
        Self {
            identity: UnvalidatedResourceTransactionIdentity {
                pool_id: receipt.identity.pool_id,
                run_id: receipt.identity.run_id.clone(),
                transaction_id: receipt.identity.transaction_id.clone(),
                request_id: receipt.identity.request_id.clone(),
            },
            admission: UnvalidatedStaticProvisioningBinding {
                pool_identity: UnvalidatedResourcePoolIdentity {
                    pool_id: receipt.admission.pool_identity.pool_id,
                    plan_id: receipt.admission.pool_identity.plan_id.clone(),
                    plan_hash: receipt.admission.pool_identity.plan_hash.clone(),
                    device_id: receipt.admission.pool_identity.device_id.clone(),
                    device_runtime_implementation_fingerprint: receipt
                        .admission
                        .pool_identity
                        .device_runtime_implementation_fingerprint
                        .clone(),
                    admission_generation: receipt.admission.pool_identity.admission_generation,
                },
                plan_id: receipt.admission.plan_id.clone(),
                plan_hash: receipt.admission.plan_hash.clone(),
                request_id: receipt.admission.request_id.clone(),
                device_id: receipt.admission.device_id.clone(),
                device_runtime_implementation_fingerprint: receipt
                    .admission
                    .device_runtime_implementation_fingerprint
                    .clone(),
                device_capacity_bytes: receipt.admission.device_capacity_bytes,
                usable_capacity_bytes: receipt.admission.usable_capacity_bytes,
                plan_static_bytes: receipt.admission.plan_static_bytes,
                admitted_bytes: receipt.admission.admitted_bytes,
                maximum_active_sequences: receipt.admission.maximum_active_sequences,
                admission_generation: receipt.admission.admission_generation,
            },
            action: receipt.action,
            records: receipt
                .records
                .iter()
                .map(UnvalidatedResourceTransitionRecord::from)
                .collect(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ResourceLeaseTransitionReceipt {
    run_id: RunId,
    transaction_id: TransactionId,
    request_id: RequestIdentity,
    admission: StaticProvisioningBinding,
    action: ResourceLeaseAction,
    decision: ResourceRetentionDecision,
    before: ResourceLeaseState,
    after: ResourceLeaseState,
    entries: Vec<ResourceLeaseEntry>,
}

impl ResourceLeaseTransitionReceipt {
    fn from_context(
        context: &ResourceLeaseValidationContext,
        before: ResourceLeaseState,
        after: ResourceLeaseState,
        entries: Vec<ResourceLeaseEntry>,
    ) -> Result<Self, VNextError> {
        validate_lease_entries_against_context(&entries, before, after, context)?;
        Ok(Self {
            run_id: context.identity.run_id.clone(),
            transaction_id: context.identity.transaction_id.clone(),
            request_id: context.identity.request_id.clone(),
            admission: context.admission.clone(),
            action: context.action,
            decision: context.action.decision(),
            before,
            after,
            entries,
        })
    }

    pub fn validate(&self) -> Result<(), VNextError> {
        if expected_lease_transition(self.action, self.before) != Some(self.after)
            || self.entries.is_empty()
            || self.decision != self.action.decision()
        {
            return Err(VNextError::InvalidLeaseTransition {
                lease_id: self.transaction_id.to_string(),
                from: self.before.as_str(),
                action: self.action.as_str(),
            });
        }
        let mut resources = BTreeSet::new();
        if self.entries.iter().any(|entry| {
            entry.generation == 0
                || entry.state != self.after
                || !resources.insert(entry.resource_id.clone())
        }) {
            return Err(invalid_resource(
                "lease receipt contains an invalid or duplicate entry",
            ));
        }
        Ok(())
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

    pub fn admission(&self) -> &StaticProvisioningBinding {
        &self.admission
    }

    pub const fn action(&self) -> ResourceLeaseAction {
        self.action
    }

    pub const fn decision(&self) -> ResourceRetentionDecision {
        self.decision
    }

    pub const fn before(&self) -> ResourceLeaseState {
        self.before
    }

    pub const fn after(&self) -> ResourceLeaseState {
        self.after
    }

    pub fn entries(&self) -> &[ResourceLeaseEntry] {
        &self.entries
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct UnvalidatedResourceLeaseTransitionReceipt {
    pub run_id: RunId,
    pub transaction_id: TransactionId,
    pub request_id: RequestIdentity,
    pub admission: UnvalidatedStaticProvisioningBinding,
    pub action: ResourceLeaseAction,
    pub decision: ResourceRetentionDecision,
    pub before: ResourceLeaseState,
    pub after: ResourceLeaseState,
    pub entries: Vec<UnvalidatedResourceLeaseEntry>,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct UnvalidatedResourceLeaseTransitionReceiptWire {
    run_id: RunId,
    transaction_id: TransactionId,
    request_id: RequestIdentity,
    admission: UnvalidatedStaticProvisioningBinding,
    action: ResourceLeaseAction,
    decision: ResourceRetentionDecision,
    before: ResourceLeaseState,
    after: ResourceLeaseState,
    entries: Vec<UnvalidatedResourceLeaseEntry>,
}

impl From<UnvalidatedResourceLeaseTransitionReceiptWire>
    for UnvalidatedResourceLeaseTransitionReceipt
{
    fn from(wire: UnvalidatedResourceLeaseTransitionReceiptWire) -> Self {
        Self {
            run_id: wire.run_id,
            transaction_id: wire.transaction_id,
            request_id: wire.request_id,
            admission: wire.admission,
            action: wire.action,
            decision: wire.decision,
            before: wire.before,
            after: wire.after,
            entries: wire.entries,
        }
    }
}

impl UnvalidatedResourceLeaseTransitionReceipt {
    pub fn decode_untrusted(bytes: &[u8]) -> Result<Self, VNextError> {
        if bytes.len() > MAX_RESOURCE_LEASE_RECEIPT_WIRE_BYTES {
            return Err(VNextError::Serialization {
                context: "decode untrusted resource lease transition receipt",
                message: format!(
                    "resource lease transition receipt wire size {} exceeds limit {}",
                    bytes.len(),
                    MAX_RESOURCE_LEASE_RECEIPT_WIRE_BYTES
                ),
            });
        }
        serde_json::from_slice::<UnvalidatedResourceLeaseTransitionReceiptWire>(bytes)
            .map(Self::from)
            .map_err(|error| VNextError::Serialization {
                context: "decode untrusted resource lease transition receipt",
                message: error.to_string(),
            })
    }

    pub fn try_validate(self) -> Result<ResourceLeaseTransitionReceipt, VNextError> {
        let _ = self;
        Err(invalid_resource(
            "untrusted lease receipt requires a trusted ledger context",
        ))
    }

    pub fn try_validate_against(
        self,
        expected: &ResourceLeaseValidationContext,
    ) -> Result<ResourceLeaseTransitionReceipt, VNextError> {
        if self.run_id != expected.identity.run_id
            || self.transaction_id != expected.identity.transaction_id
            || self.request_id != expected.identity.request_id
            || self.admission.pool_identity.pool_id != expected.admission.pool_identity.pool_id
            || self.admission.pool_identity.plan_id != expected.admission.pool_identity.plan_id
            || self.admission.pool_identity.plan_hash != expected.admission.pool_identity.plan_hash
            || self.admission.pool_identity.device_id != expected.admission.pool_identity.device_id
            || self
                .admission
                .pool_identity
                .device_runtime_implementation_fingerprint
                != expected
                    .admission
                    .pool_identity
                    .device_runtime_implementation_fingerprint
            || self.admission.pool_identity.admission_generation
                != expected.admission.pool_identity.admission_generation
            || self.admission.plan_id != expected.admission.plan_id
            || self.admission.plan_hash != expected.admission.plan_hash
            || self.admission.request_id != expected.admission.request_id
            || self.admission.device_id != expected.admission.device_id
            || self.admission.device_runtime_implementation_fingerprint
                != expected.admission.device_runtime_implementation_fingerprint
            || self.admission.device_capacity_bytes != expected.admission.device_capacity_bytes
            || self.admission.usable_capacity_bytes != expected.admission.usable_capacity_bytes
            || self.admission.plan_static_bytes != expected.admission.plan_static_bytes
            || self.admission.admitted_bytes != expected.admission.admitted_bytes
            || self.admission.maximum_active_sequences
                != expected.admission.maximum_active_sequences
            || self.admission.admission_generation != expected.admission.admission_generation
            || self.action != expected.action
            || self.decision != self.action.decision()
        {
            return Err(invalid_resource(
                "untrusted lease receipt does not match expected identity or admission",
            ));
        }
        let entries = self
            .entries
            .into_iter()
            .map(UnvalidatedResourceLeaseEntry::into_trusted)
            .collect::<Result<Vec<_>, _>>()?;
        ResourceLeaseTransitionReceipt::from_context(expected, self.before, self.after, entries)
    }
}

impl From<&ResourceLeaseTransitionReceipt> for UnvalidatedResourceLeaseTransitionReceipt {
    fn from(receipt: &ResourceLeaseTransitionReceipt) -> Self {
        Self {
            run_id: receipt.run_id.clone(),
            transaction_id: receipt.transaction_id.clone(),
            request_id: receipt.request_id.clone(),
            admission: UnvalidatedStaticProvisioningBinding {
                pool_identity: UnvalidatedResourcePoolIdentity {
                    pool_id: receipt.admission.pool_identity.pool_id,
                    plan_id: receipt.admission.pool_identity.plan_id.clone(),
                    plan_hash: receipt.admission.pool_identity.plan_hash.clone(),
                    device_id: receipt.admission.pool_identity.device_id.clone(),
                    device_runtime_implementation_fingerprint: receipt
                        .admission
                        .pool_identity
                        .device_runtime_implementation_fingerprint
                        .clone(),
                    admission_generation: receipt.admission.pool_identity.admission_generation,
                },
                plan_id: receipt.admission.plan_id.clone(),
                plan_hash: receipt.admission.plan_hash.clone(),
                request_id: receipt.admission.request_id.clone(),
                device_id: receipt.admission.device_id.clone(),
                device_runtime_implementation_fingerprint: receipt
                    .admission
                    .device_runtime_implementation_fingerprint
                    .clone(),
                device_capacity_bytes: receipt.admission.device_capacity_bytes,
                usable_capacity_bytes: receipt.admission.usable_capacity_bytes,
                plan_static_bytes: receipt.admission.plan_static_bytes,
                admitted_bytes: receipt.admission.admitted_bytes,
                maximum_active_sequences: receipt.admission.maximum_active_sequences,
                admission_generation: receipt.admission.admission_generation,
            },
            action: receipt.action,
            decision: receipt.decision,
            before: receipt.before,
            after: receipt.after,
            entries: receipt
                .entries
                .iter()
                .map(UnvalidatedResourceLeaseEntry::from)
                .collect(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ResourceFailurePoint {
    owner_node_id: Option<NodeId>,
    resource_id: ResourceId,
    generation: u64,
    order: u32,
    actual_before: ResourceTransactionState,
}

impl ResourceFailurePoint {
    fn new(
        reservation: &ResourceReservation,
        order: usize,
        actual_before: ResourceTransactionState,
    ) -> Self {
        Self {
            owner_node_id: reservation.owner_node_id.clone(),
            resource_id: reservation.resource_id.clone(),
            generation: reservation.generation,
            order: order as u32,
            actual_before,
        }
    }

    pub fn owner_node_id(&self) -> Option<&NodeId> {
        self.owner_node_id.as_ref()
    }

    pub fn resource_id(&self) -> &ResourceId {
        &self.resource_id
    }

    pub const fn generation(&self) -> u64 {
        self.generation
    }

    pub const fn order(&self) -> u32 {
        self.order
    }

    pub const fn actual_before(&self) -> ResourceTransactionState {
        self.actual_before
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ResourceRecoveryFailure {
    failure: FailureEnvelope,
    resource: Option<ResourceFailurePoint>,
    attempt: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(try_from = "u64", into = "u64")]
pub struct ResourceFailureId(u64);

impl ResourceFailureId {
    pub const fn get(self) -> u64 {
        self.0
    }
}

impl TryFrom<u64> for ResourceFailureId {
    type Error = VNextError;

    fn try_from(value: u64) -> Result<Self, Self::Error> {
        if value == 0 {
            return Err(invalid_resource("resource failure id must be non-zero"));
        }
        Ok(Self(value))
    }
}

impl From<ResourceFailureId> for u64 {
    fn from(value: ResourceFailureId) -> Self {
        value.0
    }
}

impl ResourceRecoveryFailure {
    pub fn failure(&self) -> &FailureEnvelope {
        &self.failure
    }

    pub fn resource(&self) -> Option<&ResourceFailurePoint> {
        self.resource.as_ref()
    }

    pub const fn attempt(&self) -> u32 {
        self.attempt
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ResourceFailureReceipt {
    failure_id: ResourceFailureId,
    identity: ResourceTransactionIdentity,
    admission: StaticProvisioningBinding,
    action: ResourceTransactionAction,
    failure: FailureEnvelope,
    failure_point: Option<ResourceFailurePoint>,
    completed: Vec<ResourceTransitionRecord>,
    compensation: Vec<ResourceCompensationRecord>,
    recovery_failures: Vec<ResourceRecoveryFailure>,
    recovery_strategy: ResourceRecoveryStrategy,
    recovery_complete: bool,
    ledger_before: Vec<ResourceLedgerEntrySnapshot>,
    ledger_after: Vec<ResourceLedgerEntrySnapshot>,
}

impl ResourceFailureReceipt {
    fn new(
        failure_id: ResourceFailureId,
        identity: &ResourceTransactionIdentity,
        admission: &StaticProvisioningBinding,
        action: ResourceTransactionAction,
        failure: FailureEnvelope,
        failure_point: Option<ResourceFailurePoint>,
        completed: Vec<ResourceTransitionRecord>,
        recovery_strategy: ResourceRecoveryStrategy,
        ledger_before: Vec<ResourceLedgerEntrySnapshot>,
        ledger_after: Vec<ResourceLedgerEntrySnapshot>,
    ) -> Self {
        Self {
            failure_id,
            identity: identity.clone(),
            admission: admission.clone(),
            action,
            failure,
            failure_point,
            completed,
            compensation: Vec::new(),
            recovery_failures: Vec::new(),
            recovery_strategy,
            recovery_complete: false,
            ledger_before,
            ledger_after,
        }
    }

    pub const fn failure_id(&self) -> ResourceFailureId {
        self.failure_id
    }

    pub fn identity(&self) -> &ResourceTransactionIdentity {
        &self.identity
    }

    pub fn admission(&self) -> &StaticProvisioningBinding {
        &self.admission
    }

    pub const fn action(&self) -> ResourceTransactionAction {
        self.action
    }

    pub fn failure(&self) -> &FailureEnvelope {
        &self.failure
    }

    pub fn failure_point(&self) -> Option<&ResourceFailurePoint> {
        self.failure_point.as_ref()
    }

    pub fn completed(&self) -> &[ResourceTransitionRecord] {
        &self.completed
    }

    pub fn compensation(&self) -> &[ResourceCompensationRecord] {
        &self.compensation
    }

    pub fn recovery_failures(&self) -> &[ResourceRecoveryFailure] {
        &self.recovery_failures
    }

    pub const fn recovery_strategy(&self) -> ResourceRecoveryStrategy {
        self.recovery_strategy
    }

    pub const fn recovery_complete(&self) -> bool {
        self.recovery_complete
    }

    pub fn ledger_before(&self) -> &[ResourceLedgerEntrySnapshot] {
        &self.ledger_before
    }

    pub fn ledger_after(&self) -> &[ResourceLedgerEntrySnapshot] {
        &self.ledger_after
    }

    pub fn validate_recovery_continuation(&self, anchor: &Self) -> Result<(), VNextError> {
        let strategy_continues = self.recovery_strategy == anchor.recovery_strategy
            || (anchor.recovery_strategy == ResourceRecoveryStrategy::ReconcileOrQuarantine
                && self.recovery_strategy == ResourceRecoveryStrategy::ReverseCompensation);
        if self.failure_id != anchor.failure_id
            || self.identity != anchor.identity
            || self.admission != anchor.admission
            || self.action != anchor.action
            || self.failure != anchor.failure
            || self.failure_point != anchor.failure_point
            || !self.completed.starts_with(&anchor.completed)
            || !self.compensation.starts_with(&anchor.compensation)
            || !self
                .recovery_failures
                .starts_with(&anchor.recovery_failures)
            || !strategy_continues
            || (anchor.recovery_complete && !self.recovery_complete)
            || self.ledger_before != anchor.ledger_before
        {
            return Err(invalid_resource(
                "resource recovery receipt does not continue its exact failure anchor",
            ));
        }
        Ok(())
    }

    fn record_recovery_failure(
        &mut self,
        failure: ResourceDriverFailure,
        resource: Option<ResourceFailurePoint>,
    ) {
        self.recovery_failures.push(ResourceRecoveryFailure {
            failure: failure.into_failure(),
            resource,
            attempt: (self.recovery_failures.len() + 1) as u32,
        });
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ResourceAbandonSignal {
    identity: ResourceTransactionIdentity,
    admission: StaticProvisioningBinding,
    state: ResourceTransactionState,
    pending_action: Option<ResourceTransactionAction>,
    ledger: Vec<ResourceLedgerEntrySnapshot>,
    active_sequence_slots: Vec<u32>,
    poisoned_sequence_slots: Vec<u32>,
    undrained_sequence_slots: Vec<u32>,
    failure: Option<FailureEnvelope>,
}

impl ResourceAbandonSignal {
    pub fn identity(&self) -> &ResourceTransactionIdentity {
        &self.identity
    }

    pub fn admission(&self) -> &StaticProvisioningBinding {
        &self.admission
    }

    pub const fn state(&self) -> ResourceTransactionState {
        self.state
    }

    pub const fn pending_action(&self) -> Option<ResourceTransactionAction> {
        self.pending_action
    }

    pub fn ledger(&self) -> &[ResourceLedgerEntrySnapshot] {
        &self.ledger
    }

    pub fn active_sequence_slots(&self) -> &[u32] {
        &self.active_sequence_slots
    }

    pub fn poisoned_sequence_slots(&self) -> &[u32] {
        &self.poisoned_sequence_slots
    }

    pub fn undrained_sequence_slots(&self) -> &[u32] {
        &self.undrained_sequence_slots
    }

    pub fn live_resources(&self) -> impl Iterator<Item = &ResourceLeaseEntry> {
        self.ledger
            .iter()
            .filter(|entry| entry.transaction_state.is_live())
            .map(ResourceLedgerEntrySnapshot::entry)
    }

    pub fn failure(&self) -> Option<&FailureEnvelope> {
        self.failure.as_ref()
    }
}

struct OwnedLeaseSlot<B> {
    entry: ResourceLeaseEntry,
    actual_resource_id: Option<ResourceId>,
    actual_generation: Option<u64>,
    descriptor: Option<BufferDescriptor>,
    buffer: Option<B>,
}

impl<B> OwnedLeaseSlot<B> {
    fn new(reservation: &ResourceReservation) -> Self {
        Self {
            entry: ResourceLeaseEntry::from_reservation(reservation, ResourceLeaseState::Active),
            actual_resource_id: None,
            actual_generation: None,
            descriptor: None,
            buffer: None,
        }
    }

    fn install(&mut self, allocation: CoreOwnedAllocation<B>) {
        self.actual_resource_id = Some(allocation.resource_id);
        self.actual_generation = Some(allocation.generation);
        self.descriptor = Some(allocation.descriptor);
        self.buffer = Some(allocation.buffer);
    }

    fn clear(&mut self) {
        drop(self.buffer.take());
        self.descriptor.take();
        self.actual_resource_id.take();
        self.actual_generation.take();
    }

    fn take_allocation(&mut self) -> Option<CoreOwnedAllocation<B>> {
        Some(CoreOwnedAllocation {
            resource_id: self.actual_resource_id.take()?,
            generation: self.actual_generation.take()?,
            descriptor: self.descriptor.take()?,
            buffer: self.buffer.take()?,
        })
    }

    fn restore_allocation(&mut self, allocation: CoreOwnedAllocation<B>) {
        debug_assert!(self.buffer.is_none());
        self.install(allocation);
    }
}

/// Borrowed access to a live, active, generation-bound committed buffer.
pub struct LeasedBufferView<'a, B> {
    identity: &'a ResourceTransactionIdentity,
    admission: &'a StaticProvisioningBinding,
    resource_id: &'a ResourceId,
    generation: u64,
    descriptor: &'a BufferDescriptor,
    buffer: &'a B,
}

impl<'a, B> LeasedBufferView<'a, B> {
    pub fn identity(&self) -> &ResourceTransactionIdentity {
        self.identity
    }

    pub fn admission(&self) -> &StaticProvisioningBinding {
        self.admission
    }

    pub fn resource_id(&self) -> &ResourceId {
        self.resource_id
    }

    pub const fn generation(&self) -> u64 {
        self.generation
    }

    pub fn committed_descriptor(&self) -> &BufferDescriptor {
        self.descriptor
    }

    pub fn buffer(&self) -> &B {
        self.buffer
    }
}

#[derive(Debug)]
pub enum ExecutionStreamCreationError<E> {
    Contract(VNextError),
    Runtime(E),
}

/// A stream created and owned by one exact admitted runtime instance. Its
/// runtime and raw stream are private so execution can only proceed through an
/// active sequence permit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BoundExecutionStreamState {
    Ready,
    InUse,
    Poisoned,
}

#[derive(Debug)]
pub enum AbandonedSequenceRecoveryError<E> {
    Contract(VNextError),
    Runtime(E),
    StreamStillOwned { slot: u32, activation_epoch: u64 },
}

#[derive(Clone)]
struct AbandonedSequenceMetadata {
    plan: TrustedPlanRuntimeEvidence,
    sequence_authority: SequenceAuthorityId,
    run_id: RunId,
    request_id: RequestIdentity,
    slot: u32,
    activation_epoch: u64,
    runtime_implementation_fingerprint: String,
    state: Arc<AtomicU64>,
    sequence_dispatch_gate: Arc<AtomicU64>,
    drained: bool,
}

impl AbandonedSequenceMetadata {
    fn key(&self) -> (u32, u64) {
        (self.slot, self.activation_epoch)
    }

    fn abort_receipt(&self) -> ActiveSequenceAbortReceipt {
        ActiveSequenceAbortReceipt {
            plan: self.plan.clone(),
            sequence_authority: self.sequence_authority,
            run_id: self.run_id.clone(),
            request_id: self.request_id.clone(),
            activation_epoch: self.activation_epoch,
            runtime_implementation_fingerprint: self.runtime_implementation_fingerprint.clone(),
            disposition: ActiveSequenceAbortDisposition::SynchronizedAndPoisoned,
        }
    }
}

struct AbandonedSequenceRecord<R>
where
    R: DeviceRuntime,
{
    metadata: AbandonedSequenceMetadata,
    stream: AbandonedSequenceStream<R::Stream>,
}

enum AbandonedSequenceStream<S> {
    ExternallyOwned,
    Attached(S),
    Recovering,
}

struct SequenceRecoveryRegistry<R>
where
    R: DeviceRuntime,
{
    // Records (and their raw streams) must drop before the owning root.
    records: Mutex<BTreeMap<(u32, u64), AbandonedSequenceRecord<R>>>,
    _resources: Arc<PlanRuntimeResources<R>>,
}

impl<R> SequenceRecoveryRegistry<R>
where
    R: DeviceRuntime,
{
    fn new(resources: Arc<PlanRuntimeResources<R>>) -> Self {
        Self {
            records: Mutex::new(BTreeMap::new()),
            _resources: resources,
        }
    }

    fn lock_records(
        &self,
    ) -> std::sync::MutexGuard<'_, BTreeMap<(u32, u64), AbandonedSequenceRecord<R>>> {
        self.records
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    fn is_empty(&self) -> bool {
        self.lock_records().is_empty()
    }

    fn register(&self, metadata: AbandonedSequenceMetadata) {
        let key = metadata.key();
        let mut records = self.lock_records();
        if records.contains_key(&key) {
            debug_assert!(false, "sequence recovery epoch registered twice");
            return;
        }
        records.insert(
            key,
            AbandonedSequenceRecord {
                metadata,
                stream: AbandonedSequenceStream::ExternallyOwned,
            },
        );
    }

    fn attach_stream(&self, key: (u32, u64), stream: R::Stream) {
        let mut records = self.lock_records();
        let Some(record) = records.get_mut(&key) else {
            std::mem::forget(stream);
            return;
        };
        record
            .metadata
            .sequence_dispatch_gate
            .fetch_or(SEQUENCE_DISPATCH_POISONED_BIT, Ordering::AcqRel);
        match &record.stream {
            AbandonedSequenceStream::ExternallyOwned => {
                record.stream = AbandonedSequenceStream::Attached(stream);
            }
            AbandonedSequenceStream::Attached(_) | AbandonedSequenceStream::Recovering => {
                std::mem::forget(stream);
            }
        }
    }

    fn set_drained(&self, key: (u32, u64), drained: bool) {
        let mut records = self.lock_records();
        let record = records
            .get_mut(&key)
            .expect("active sequence recovery metadata remains registered");
        record.metadata.drained = drained;
    }

    fn clear(&self, key: (u32, u64)) {
        let removed = self.lock_records().remove(&key);
        debug_assert!(
            removed.is_some(),
            "terminal sequence lost recovery metadata"
        );
    }

    fn recover(
        &self,
        runtime: &Arc<R>,
        slot: u32,
    ) -> Result<ActiveSequenceAbortReceipt, AbandonedSequenceRecoveryError<R::Error>> {
        // Move the raw stream into an explicit Recovering state, then release
        // the registry before invoking backend code. Concurrent recovery sees
        // the state and fails closed without blocking on the backend call.
        let (key, mut stream, was_drained) = {
            let mut records = self.lock_records();
            let matching = records
                .keys()
                .filter(|(candidate, _)| *candidate == slot)
                .copied()
                .collect::<Vec<_>>();
            if matching.len() != 1 {
                return Err(AbandonedSequenceRecoveryError::Contract(invalid_resource(
                    "abandoned sequence recovery requires one exact registered slot epoch",
                )));
            }
            let key = matching[0];
            let record = records
                .get_mut(&key)
                .expect("matching recovery key remains registered");
            let stream =
                match std::mem::replace(&mut record.stream, AbandonedSequenceStream::Recovering) {
                    AbandonedSequenceStream::Attached(stream) => stream,
                    AbandonedSequenceStream::ExternallyOwned => {
                        record.stream = AbandonedSequenceStream::ExternallyOwned;
                        return Err(AbandonedSequenceRecoveryError::StreamStillOwned {
                            slot,
                            activation_epoch: record.metadata.activation_epoch,
                        });
                    }
                    AbandonedSequenceStream::Recovering => {
                        record.stream = AbandonedSequenceStream::Recovering;
                        return Err(AbandonedSequenceRecoveryError::Contract(invalid_resource(
                        "abandoned sequence recovery is already in progress for this slot epoch",
                    )));
                    }
                };
            (key, stream, record.metadata.drained)
        };

        let backend_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            if !was_drained {
                runtime.synchronize(&mut stream)?;
            }
            Ok(runtime.stream_state(&stream) == StreamState::Ready)
        }));
        let stream_ready = match backend_result {
            Ok(Ok(stream_ready)) => stream_ready,
            Ok(Err(error)) => {
                self.restore_recovery_stream(key, stream, false);
                return Err(AbandonedSequenceRecoveryError::Runtime(error));
            }
            Err(payload) => {
                self.restore_recovery_stream(key, stream, false);
                std::panic::resume_unwind(payload);
            }
        };
        if !stream_ready {
            self.restore_recovery_stream(key, stream, false);
            return Err(AbandonedSequenceRecoveryError::Contract(invalid_resource(
                "abandoned sequence synchronization did not drain its stream",
            )));
        }

        let mut records = self.lock_records();
        let Some(record) = records.get_mut(&key) else {
            std::mem::forget(stream);
            return Err(AbandonedSequenceRecoveryError::Contract(invalid_resource(
                "abandoned sequence recovery registration disappeared while synchronization was in progress",
            )));
        };
        if !matches!(record.stream, AbandonedSequenceStream::Recovering) {
            std::mem::forget(stream);
            return Err(AbandonedSequenceRecoveryError::Contract(invalid_resource(
                "abandoned sequence recovery ownership changed while synchronization was in progress",
            )));
        }
        record.metadata.drained = true;
        let expected_active = sequence_slot_active(record.metadata.activation_epoch);
        let expected_undrained = sequence_slot_poisoned_undrained(record.metadata.activation_epoch);
        let expected_drained = sequence_slot_poisoned_drained(record.metadata.activation_epoch);
        let actual = record.metadata.state.load(Ordering::Acquire);
        if actual == expected_active || actual == expected_undrained {
            if record
                .metadata
                .state
                .compare_exchange(
                    actual,
                    expected_drained,
                    Ordering::AcqRel,
                    Ordering::Acquire,
                )
                .is_err()
            {
                record.stream = AbandonedSequenceStream::Attached(stream);
                return Err(AbandonedSequenceRecoveryError::Contract(invalid_resource(
                    "abandoned sequence epoch changed during recovery",
                )));
            }
        } else if actual != expected_drained {
            record.stream = AbandonedSequenceStream::Attached(stream);
            return Err(AbandonedSequenceRecoveryError::Contract(invalid_resource(
                "abandoned sequence recovery does not own an active or poisoned registered slot epoch",
            )));
        }
        record
            .metadata
            .sequence_dispatch_gate
            .fetch_or(SEQUENCE_DISPATCH_POISONED_BIT, Ordering::AcqRel);

        let receipt = record.metadata.abort_receipt();
        let removed = records.remove(&key);
        drop(records);
        drop(removed);
        drop(stream);
        Ok(receipt)
    }

    fn restore_recovery_stream(&self, key: (u32, u64), stream: R::Stream, drained: bool) {
        let mut records = self.lock_records();
        let Some(record) = records.get_mut(&key) else {
            std::mem::forget(stream);
            return;
        };
        if matches!(record.stream, AbandonedSequenceStream::Recovering) {
            record.metadata.drained = drained;
            record.stream = AbandonedSequenceStream::Attached(stream);
        } else {
            std::mem::forget(stream);
        }
    }

    fn recover_all_for_owner_drop(
        &self,
        runtime: &Arc<R>,
    ) -> Result<(), AbandonedSequenceRecoveryError<R::Error>> {
        loop {
            let slot = {
                let records = self.lock_records();
                records.keys().next().map(|(slot, _)| *slot)
            };
            let Some(slot) = slot else {
                return Ok(());
            };
            let _ = self.recover(runtime, slot)?;
        }
    }
}

#[must_use = "an execution stream must be activated through its resource lease"]
pub struct BoundExecutionStream<R>
where
    R: DeviceRuntime,
{
    // The raw stream drops or transfers into the recovery registry before this
    // owning sequence hold can release logical capacity or backing extents.
    runtime: Arc<R>,
    coordinator_id: LogicalAdmissionCoordinatorId,
    sequence_authority: SequenceAuthorityId,
    stream: Option<R::Stream>,
    state: BoundExecutionStreamState,
    sequence_recovery: Arc<SequenceRecoveryRegistry<R>>,
    sequence_dispatch_gate: Arc<AtomicU64>,
    abandoned_sequence: Option<(u32, u64)>,
    resources: Arc<AdmittedSequenceResources<R>>,
}

impl<R> BoundExecutionStream<R>
where
    R: DeviceRuntime,
{
    fn stream(&self) -> &R::Stream {
        self.stream
            .as_ref()
            .expect("bound execution stream retains its raw stream")
    }

    fn stream_mut(&mut self) -> &mut R::Stream {
        self.stream
            .as_mut()
            .expect("bound execution stream retains its raw stream")
    }
}

impl<R> Drop for BoundExecutionStream<R>
where
    R: DeviceRuntime,
{
    fn drop(&mut self) {
        let Some(key) = self.abandoned_sequence.take() else {
            return;
        };
        self.sequence_dispatch_gate
            .fetch_or(SEQUENCE_DISPATCH_POISONED_BIT, Ordering::AcqRel);
        if let Some(stream) = self.stream.take() {
            self.sequence_recovery.attach_stream(key, stream);
        }
    }
}

static NEXT_DYNAMIC_POOL_INSTANCE_ID: AtomicU64 = AtomicU64::new(1);

fn align_up_resource(value: u64, alignment: u64) -> Result<u64, VNextError> {
    if alignment == 0 || !alignment.is_power_of_two() {
        return Err(invalid_resource(
            "dynamic pool alignment is not a non-zero power of two",
        ));
    }
    value
        .checked_add(alignment - 1)
        .map(|rounded| rounded & !(alignment - 1))
        .ok_or_else(|| invalid_resource("dynamic pool aligned bytes overflow u64"))
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize)]
pub struct BackingChunkIdentity {
    pool_id: DynamicBackingPoolId,
    ordinal: u32,
    generation: u64,
}

impl BackingChunkIdentity {
    pub fn pool_id(&self) -> &DynamicBackingPoolId {
        &self.pool_id
    }

    pub const fn ordinal(&self) -> u32 {
        self.ordinal
    }

    pub const fn generation(&self) -> u64 {
        self.generation
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct BackingSegment {
    chunk: BackingChunkIdentity,
    offset_bytes: u64,
    length_bytes: u64,
}

impl BackingSegment {
    pub(crate) fn new(
        chunk: BackingChunkIdentity,
        offset_bytes: u64,
        length_bytes: u64,
    ) -> Result<Self, VNextError> {
        Self::from_chunk(
            &chunk.pool_id,
            chunk.ordinal,
            chunk.generation,
            offset_bytes,
            length_bytes,
        )
    }

    fn from_chunk(
        pool_id: &DynamicBackingPoolId,
        chunk_ordinal: u32,
        chunk_generation: u64,
        offset_bytes: u64,
        length_bytes: u64,
    ) -> Result<Self, VNextError> {
        if chunk_ordinal == 0
            || chunk_generation == 0
            || length_bytes == 0
            || offset_bytes.checked_add(length_bytes).is_none()
        {
            return Err(invalid_resource(
                "backing segment has invalid chunk identity or physical range",
            ));
        }
        Ok(Self {
            chunk: BackingChunkIdentity {
                pool_id: pool_id.clone(),
                ordinal: chunk_ordinal,
                generation: chunk_generation,
            },
            offset_bytes,
            length_bytes,
        })
    }

    pub fn chunk(&self) -> &BackingChunkIdentity {
        &self.chunk
    }

    pub fn pool_id(&self) -> &DynamicBackingPoolId {
        self.chunk.pool_id()
    }

    pub const fn chunk_ordinal(&self) -> u32 {
        self.chunk.ordinal()
    }

    pub const fn chunk_generation(&self) -> u64 {
        self.chunk.generation()
    }

    pub const fn offset_bytes(&self) -> u64 {
        self.offset_bytes
    }

    pub const fn length_bytes(&self) -> u64 {
        self.length_bytes
    }
}

struct ResidentChunkBacking<B> {
    // Buffer must drop before its physical capacity grant is returned.
    buffer: B,
    _grant: DeviceCapacityGrant,
    identity: BackingChunkIdentity,
    descriptor: BufferDescriptor,
}

struct ResidentChunkState<B> {
    backing: Arc<ResidentChunkBacking<B>>,
    live_segments: u64,
}

#[derive(Debug, Clone, Copy)]
struct FreeExtent {
    chunk_generation: u64,
    length_bytes: u64,
}

#[derive(Debug, Default)]
struct FreeExtentIndex {
    by_offset: BTreeMap<(u32, u64), FreeExtent>,
    by_size: BTreeSet<(u64, u32, u64, u64)>,
    free_bytes: u64,
    search_probes: u64,
}

impl FreeExtentIndex {
    fn rollback_segments(&mut self, segments: &[BackingSegment]) -> Result<(), VNextError> {
        for segment in segments.iter().rev() {
            self.release(segment)?;
        }
        Ok(())
    }

    fn with_rollback_context(
        &mut self,
        segments: &[BackingSegment],
        error: VNextError,
    ) -> VNextError {
        match self.rollback_segments(segments) {
            Ok(()) => error,
            Err(rollback) => invalid_resource(format!(
                "dynamic allocator failed and its journal rollback also failed: {error}; rollback: {rollback}"
            )),
        }
    }

    fn insert_extent(
        &mut self,
        chunk_ordinal: u32,
        chunk_generation: u64,
        offset_bytes: u64,
        length_bytes: u64,
    ) -> Result<(), VNextError> {
        if chunk_ordinal == 0
            || chunk_generation == 0
            || length_bytes == 0
            || offset_bytes.checked_add(length_bytes).is_none()
            || self.by_offset.contains_key(&(chunk_ordinal, offset_bytes))
        {
            return Err(invalid_resource("free extent identity or range is invalid"));
        }
        let end = offset_bytes + length_bytes;
        if self
            .by_offset
            .range(..(chunk_ordinal, offset_bytes))
            .next_back()
            .is_some_and(|(&(ordinal, previous_offset), previous)| {
                ordinal == chunk_ordinal && previous_offset + previous.length_bytes > offset_bytes
            })
            || self
                .by_offset
                .range((chunk_ordinal, offset_bytes)..)
                .next()
                .is_some_and(|(&(ordinal, next_offset), _)| {
                    ordinal == chunk_ordinal && next_offset < end
                })
        {
            return Err(invalid_resource("free extent overlaps an existing extent"));
        }
        let next_free = self
            .free_bytes
            .checked_add(length_bytes)
            .ok_or_else(|| invalid_resource("free extent bytes overflow u64"))?;
        self.by_offset.insert(
            (chunk_ordinal, offset_bytes),
            FreeExtent {
                chunk_generation,
                length_bytes,
            },
        );
        assert!(self
            .by_size
            .insert((length_bytes, chunk_ordinal, chunk_generation, offset_bytes,)));
        self.free_bytes = next_free;
        Ok(())
    }

    fn remove_extent(
        &mut self,
        chunk_ordinal: u32,
        offset_bytes: u64,
    ) -> Result<FreeExtent, VNextError> {
        let extent = *self
            .by_offset
            .get(&(chunk_ordinal, offset_bytes))
            .ok_or_else(|| invalid_resource("free extent journal references a missing range"))?;
        let size_key = (
            extent.length_bytes,
            chunk_ordinal,
            extent.chunk_generation,
            offset_bytes,
        );
        if !self.by_size.contains(&size_key) {
            return Err(invalid_resource("free extent indexes diverged"));
        }
        let next_free_bytes = self
            .free_bytes
            .checked_sub(extent.length_bytes)
            .ok_or_else(|| invalid_resource("free extent bytes underflowed"))?;
        self.by_offset.remove(&(chunk_ordinal, offset_bytes));
        assert!(self.by_size.remove(&size_key));
        self.free_bytes = next_free_bytes;
        Ok(extent)
    }

    fn allocate_contiguous(
        &mut self,
        pool_id: &DynamicBackingPoolId,
        size_bytes: u64,
    ) -> Result<Option<BackingSegment>, VNextError> {
        self.search_probes = self.search_probes.saturating_add(1);
        let selected = self.by_size.range((size_bytes, 0, 0, 0)..).next().copied();
        let Some((length_bytes, chunk_ordinal, chunk_generation, offset_bytes)) = selected else {
            return Ok(None);
        };
        let segment = BackingSegment::from_chunk(
            pool_id,
            chunk_ordinal,
            chunk_generation,
            offset_bytes,
            size_bytes,
        )?;
        let removed = self.remove_extent(chunk_ordinal, offset_bytes)?;
        debug_assert_eq!(removed.length_bytes, length_bytes);
        debug_assert_eq!(removed.chunk_generation, chunk_generation);
        if size_bytes < length_bytes {
            if let Err(error) = self.insert_extent(
                chunk_ordinal,
                chunk_generation,
                offset_bytes + size_bytes,
                length_bytes - size_bytes,
            ) {
                let restore =
                    self.insert_extent(chunk_ordinal, chunk_generation, offset_bytes, length_bytes);
                return Err(match restore {
                    Ok(()) => error,
                    Err(rollback) => invalid_resource(format!(
                        "contiguous allocator failed and could not restore its selected extent: {error}; rollback: {rollback}"
                    )),
                });
            }
        }
        Ok(Some(segment))
    }

    fn allocate_paged(
        &mut self,
        pool_id: &DynamicBackingPoolId,
        size_bytes: u64,
        block_bytes: u64,
    ) -> Result<Option<Vec<BackingSegment>>, VNextError> {
        if size_bytes == 0 || size_bytes % block_bytes != 0 {
            return Err(invalid_resource(
                "paged backing reservation is not block aligned",
            ));
        }
        if self.free_bytes < size_bytes {
            return Ok(None);
        }
        let mut remaining = size_bytes;
        let mut segments = Vec::new();
        while remaining != 0 {
            self.search_probes = self.search_probes.saturating_add(1);
            let Some((&(chunk_ordinal, offset_bytes), &extent)) = self.by_offset.first_key_value()
            else {
                self.rollback_segments(&segments)?;
                return Ok(None);
            };
            if extent.length_bytes % block_bytes != 0 {
                let error = invalid_resource("paged free extent lost fixed-block alignment");
                return Err(self.with_rollback_context(&segments, error));
            }
            let take = extent.length_bytes.min(remaining);
            let segment = match BackingSegment::from_chunk(
                pool_id,
                chunk_ordinal,
                extent.chunk_generation,
                offset_bytes,
                take,
            ) {
                Ok(segment) => segment,
                Err(error) => return Err(self.with_rollback_context(&segments, error)),
            };
            if let Err(error) = self.remove_extent(chunk_ordinal, offset_bytes) {
                return Err(self.with_rollback_context(&segments, error));
            }
            if take < extent.length_bytes {
                if let Err(error) = self.insert_extent(
                    chunk_ordinal,
                    extent.chunk_generation,
                    offset_bytes + take,
                    extent.length_bytes - take,
                ) {
                    let restore = self.insert_extent(
                        chunk_ordinal,
                        extent.chunk_generation,
                        offset_bytes,
                        extent.length_bytes,
                    );
                    let error = match restore {
                        Ok(()) => error,
                        Err(rollback) => invalid_resource(format!(
                            "paged allocator failed and could not restore its selected extent: {error}; rollback: {rollback}"
                        )),
                    };
                    return Err(self.with_rollback_context(&segments, error));
                }
            }
            segments.push(segment);
            remaining -= take;
        }
        Ok(Some(segments))
    }

    fn release(&mut self, segment: &BackingSegment) -> Result<(), VNextError> {
        let chunk_ordinal = segment.chunk_ordinal();
        let chunk_generation = segment.chunk_generation();
        let mut offset_bytes = segment.offset_bytes();
        let mut length_bytes = segment.length_bytes();
        if let Some((&(ordinal, previous_offset), &previous)) = self
            .by_offset
            .range(..(chunk_ordinal, offset_bytes))
            .next_back()
        {
            if ordinal == chunk_ordinal
                && previous.chunk_generation == chunk_generation
                && previous_offset + previous.length_bytes == offset_bytes
            {
                self.remove_extent(ordinal, previous_offset)?;
                offset_bytes = previous_offset;
                length_bytes = length_bytes
                    .checked_add(previous.length_bytes)
                    .ok_or_else(|| invalid_resource("coalesced free extent overflows u64"))?;
            }
        }
        if let Some((&(ordinal, next_offset), &next)) =
            self.by_offset.range((chunk_ordinal, offset_bytes)..).next()
        {
            if ordinal == chunk_ordinal
                && next.chunk_generation == chunk_generation
                && offset_bytes + length_bytes == next_offset
            {
                self.remove_extent(ordinal, next_offset)?;
                length_bytes = length_bytes
                    .checked_add(next.length_bytes)
                    .ok_or_else(|| invalid_resource("coalesced free extent overflows u64"))?;
            }
        }
        self.insert_extent(chunk_ordinal, chunk_generation, offset_bytes, length_bytes)
    }

    fn largest_contiguous_bytes(&self) -> u64 {
        self.by_size
            .last()
            .map_or(0, |(length_bytes, _, _, _)| *length_bytes)
    }
}

fn rollback_free_extent_journal<B>(
    states: &mut [std::sync::MutexGuard<'_, DynamicBackingPoolState<B>>],
    journals: &[Vec<Vec<BackingSegment>>],
) -> Result<(), VNextError> {
    for group_index in (0..journals.len()).rev() {
        for segments in journals[group_index].iter().rev() {
            for segment in segments.iter().rev() {
                if let Err(error) = states[group_index].allocator.release(segment) {
                    states[group_index].poisoned = true;
                    return Err(invalid_resource(format!(
                        "dynamic backing rollback failed and poisoned its pool: {error}"
                    )));
                }
            }
        }
    }
    Ok(())
}

struct DynamicBackingPoolState<B> {
    resident_bytes: u64,
    pending_growth_bytes: u64,
    next_chunk_ordinal: u32,
    next_chunk_generation: u64,
    chunks: BTreeMap<u32, ResidentChunkState<B>>,
    allocator: FreeExtentIndex,
    quarantined: Vec<QuarantinedDynamicChunk<B>>,
    poisoned: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DynamicChunkQuarantineReason {
    DescriptorMismatch,
    PublicationRejected,
}

struct QuarantinedDynamicChunk<B> {
    backing: Arc<ResidentChunkBacking<B>>,
    reason: DynamicChunkQuarantineReason,
}

struct DynamicBackingPool<R>
where
    R: DeviceRuntime,
{
    instance_id: u64,
    domain: DynamicPoolDomainSpec,
    logical_admission: LogicalAdmissionCoordinator,
    maintenance: Mutex<()>,
    next_extent_generation: AtomicU64,
    state: Mutex<DynamicBackingPoolState<R::Buffer>>,
}

struct PendingGrowthGuard<R>
where
    R: DeviceRuntime,
{
    pool: Arc<DynamicBackingPool<R>>,
    bytes: u64,
    armed: bool,
}

impl<R> PendingGrowthGuard<R>
where
    R: DeviceRuntime,
{
    fn disarm(&mut self) {
        self.armed = false;
    }
}

impl<R> Drop for PendingGrowthGuard<R>
where
    R: DeviceRuntime,
{
    fn drop(&mut self) {
        if self.armed {
            self.pool.cancel_pending_growth(self.bytes);
        }
    }
}

trait BackingExtentOwner: Send + Sync {
    fn instance_id(&self) -> u64;
    fn release_segments(&self, segments: &[BackingSegment]);
}

struct BackingSegmentLease {
    owner: Arc<dyn BackingExtentOwner>,
    owner_instance_id: u64,
    segment_generation: u64,
    segments: Vec<BackingSegment>,
    released: bool,
}

impl Drop for BackingSegmentLease {
    fn drop(&mut self) {
        if !self.released {
            self.owner.release_segments(&self.segments);
            self.released = true;
        }
    }
}

impl<R> BackingExtentOwner for DynamicBackingPool<R>
where
    R: DeviceRuntime,
{
    fn instance_id(&self) -> u64 {
        self.instance_id
    }

    fn release_segments(&self, segments: &[BackingSegment]) {
        let mut state = match self.state.lock() {
            Ok(state) => state,
            Err(poisoned) => {
                let mut state = poisoned.into_inner();
                state.poisoned = true;
                return;
            }
        };
        if state.poisoned {
            return;
        }
        for segment in segments {
            if segment.pool_id() != self.domain.pool_id() {
                state.poisoned = true;
                return;
            }
            let Some(chunk) = state.chunks.get_mut(&segment.chunk.ordinal) else {
                state.poisoned = true;
                return;
            };
            if chunk.backing.identity != segment.chunk || chunk.live_segments == 0 {
                state.poisoned = true;
                return;
            }
        }
        for segment in segments {
            if state.allocator.release(segment).is_err() {
                state.poisoned = true;
                return;
            }
            let chunk = state
                .chunks
                .get_mut(&segment.chunk_ordinal())
                .expect("validated released chunk remains installed");
            chunk.live_segments -= 1;
        }
        drop(state);
        if self
            .logical_admission
            .notify_domain_availability_changed(self.domain.domain_id)
            .is_err()
        {
            let mut state = self
                .state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            state.poisoned = true;
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct DynamicPoolGrowthReceipt {
    pool_id: DynamicBackingPoolId,
    chunk: BackingChunkIdentity,
    chunk_bytes: u64,
    published_capacity_bytes: u64,
    capacity_epoch: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct DynamicPoolStatus {
    pool_id: DynamicBackingPoolId,
    domain_id: CapacityDomainId,
    storage_profile: DynamicStorageProfile,
    resident_bytes: u64,
    pending_growth_bytes: u64,
    free_bytes: u64,
    largest_contiguous_bytes: u64,
    resident_chunks: usize,
    live_segments: u64,
    quarantined_chunks: usize,
    quarantined_bytes: u64,
    descriptor_mismatch_chunks: usize,
    publication_rejected_chunks: usize,
    poisoned: bool,
}

impl DynamicPoolStatus {
    pub fn pool_id(&self) -> &DynamicBackingPoolId {
        &self.pool_id
    }

    pub const fn domain_id(&self) -> CapacityDomainId {
        self.domain_id
    }

    pub const fn storage_profile(&self) -> DynamicStorageProfile {
        self.storage_profile
    }

    pub const fn resident_bytes(&self) -> u64 {
        self.resident_bytes
    }

    pub const fn pending_growth_bytes(&self) -> u64 {
        self.pending_growth_bytes
    }

    pub const fn free_bytes(&self) -> u64 {
        self.free_bytes
    }

    pub const fn largest_contiguous_bytes(&self) -> u64 {
        self.largest_contiguous_bytes
    }

    pub const fn resident_chunks(&self) -> usize {
        self.resident_chunks
    }

    pub const fn live_segments(&self) -> u64 {
        self.live_segments
    }

    pub const fn quarantined_chunks(&self) -> usize {
        self.quarantined_chunks
    }

    pub const fn quarantined_bytes(&self) -> u64 {
        self.quarantined_bytes
    }

    pub const fn descriptor_mismatch_chunks(&self) -> usize {
        self.descriptor_mismatch_chunks
    }

    pub const fn publication_rejected_chunks(&self) -> usize {
        self.publication_rejected_chunks
    }

    pub const fn poisoned(&self) -> bool {
        self.poisoned
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct DynamicPoolMaintenanceStatus {
    epochs: CapacityEpochs,
    device_capacity_bytes: u64,
    effective_device_usable_ceiling_bytes: u64,
    process_claimed_bytes: u64,
    budget_device_wide_usable_ceiling_bytes: u64,
    budget_claimed_bytes: u64,
    pools: Vec<DynamicPoolStatus>,
}

impl DynamicPoolMaintenanceStatus {
    pub const fn epochs(&self) -> CapacityEpochs {
        self.epochs
    }

    pub const fn device_capacity_bytes(&self) -> u64 {
        self.device_capacity_bytes
    }

    pub const fn effective_device_usable_ceiling_bytes(&self) -> u64 {
        self.effective_device_usable_ceiling_bytes
    }

    pub const fn process_claimed_bytes(&self) -> u64 {
        self.process_claimed_bytes
    }

    pub const fn budget_device_wide_usable_ceiling_bytes(&self) -> u64 {
        self.budget_device_wide_usable_ceiling_bytes
    }

    pub const fn budget_claimed_bytes(&self) -> u64 {
        self.budget_claimed_bytes
    }

    pub fn pools(&self) -> &[DynamicPoolStatus] {
        &self.pools
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub enum DynamicDeferredMaintenanceOutcome {
    RetryWithoutGrowth { current_epochs: CapacityEpochs },
    Maintained(DynamicPoolGrowthBatchReceipt),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct DynamicPoolQuarantineRelease {
    pool_id: DynamicBackingPoolId,
    released_chunks: usize,
    released_bytes: u64,
}

impl DynamicPoolQuarantineRelease {
    pub fn pool_id(&self) -> &DynamicBackingPoolId {
        &self.pool_id
    }

    pub const fn released_chunks(&self) -> usize {
        self.released_chunks
    }

    pub const fn released_bytes(&self) -> u64 {
        self.released_bytes
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct DynamicPoolQuarantineReleaseReceipt {
    pools: Vec<DynamicPoolQuarantineRelease>,
    released_chunks: usize,
    released_bytes: u64,
}

impl DynamicPoolQuarantineReleaseReceipt {
    pub fn pools(&self) -> &[DynamicPoolQuarantineRelease] {
        &self.pools
    }

    pub const fn released_chunks(&self) -> usize {
        self.released_chunks
    }

    pub const fn released_bytes(&self) -> u64 {
        self.released_bytes
    }
}

/// Plan-owner capability for changing physical dynamic-pool residency. It is
/// created once during provisioning, is intentionally not `Clone`, and cannot
/// be derived from any request, sequence, step, invocation, or static lease.
///
/// ```compile_fail
/// use ferrum_interfaces::vnext::{AdmittedRequestResources, DeviceRuntime};
/// fn request_cannot_mint<R: DeviceRuntime>(request: &AdmittedRequestResources<R>) {
///     let _ = request.dynamic_pool_maintenance_controller();
/// }
/// ```
///
/// ```compile_fail
/// use ferrum_interfaces::vnext::{DeviceRuntime, StaticProvisioningLease};
/// fn lease_cannot_mint<R: DeviceRuntime>(lease: &StaticProvisioningLease<R>) {
///     let _ = lease.dynamic_pool_maintenance_controller();
/// }
/// ```
#[must_use = "dynamic pool maintenance controller must be retained by the plan owner"]
pub struct DynamicPoolMaintenanceController<R>
where
    R: DeviceRuntime,
{
    pools: Arc<DynamicPoolSet<R>>,
}

impl<R> DynamicPoolMaintenanceController<R>
where
    R: DeviceRuntime,
{
    fn new(pools: Arc<DynamicPoolSet<R>>) -> Self {
        Self { pools }
    }

    pub fn pool_ids(&self) -> impl ExactSizeIterator<Item = &DynamicBackingPoolId> {
        self.pools.pools.keys()
    }

    /// Returns one exact domain-to-pool and physical-capacity snapshot. The
    /// process-wide usable ceiling is the conservative minimum across all live
    /// plan budgets; plan claims remain separately visible.
    pub fn status(&self) -> Result<DynamicPoolMaintenanceStatus, VNextError> {
        let mut pools = Vec::with_capacity(self.pools.pools.len());
        for pool in self.pools.pools.values() {
            let state = pool
                .state
                .lock()
                .map_err(|_| invalid_resource("dynamic backing pool is poisoned"))?;
            let live_segments = state.chunks.values().try_fold(0_u64, |total, chunk| {
                total
                    .checked_add(chunk.live_segments)
                    .ok_or_else(|| invalid_resource("dynamic live segment count overflows u64"))
            })?;
            let quarantined_bytes = state.quarantined.iter().try_fold(0_u64, |total, chunk| {
                total
                    .checked_add(chunk.backing._grant.bytes())
                    .ok_or_else(|| invalid_resource("dynamic quarantine bytes overflow u64"))
            })?;
            pools.push(DynamicPoolStatus {
                pool_id: pool.domain.pool_id().clone(),
                domain_id: pool.domain.domain_id,
                storage_profile: pool.domain.pool.compatibility().profile(),
                resident_bytes: state.resident_bytes,
                pending_growth_bytes: state.pending_growth_bytes,
                free_bytes: state.allocator.free_bytes,
                largest_contiguous_bytes: state.allocator.largest_contiguous_bytes(),
                resident_chunks: state.chunks.len(),
                live_segments,
                quarantined_chunks: state.quarantined.len(),
                quarantined_bytes,
                descriptor_mismatch_chunks: state
                    .quarantined
                    .iter()
                    .filter(|chunk| {
                        chunk.reason == DynamicChunkQuarantineReason::DescriptorMismatch
                    })
                    .count(),
                publication_rejected_chunks: state
                    .quarantined
                    .iter()
                    .filter(|chunk| {
                        chunk.reason == DynamicChunkQuarantineReason::PublicationRejected
                    })
                    .count(),
                poisoned: state.poisoned,
            });
        }
        let account = &self.pools.budget.account;
        let state = account
            .state
            .lock()
            .map_err(|_| invalid_resource("device capacity account is poisoned"))?;
        let effective_device_usable_ceiling_bytes = state
            .budgets
            .values()
            .map(|budget| budget.device_wide_usable_ceiling_bytes)
            .min()
            .ok_or_else(|| invalid_resource("device capacity account has no live budget"))?;
        let budget_claimed_bytes = state
            .budgets
            .get(&self.pools.budget.budget_id)
            .ok_or_else(|| invalid_resource("dynamic pool plan budget is stale"))?
            .claimed_bytes;
        Ok(DynamicPoolMaintenanceStatus {
            epochs: self.pools.logical_admission.epochs()?,
            device_capacity_bytes: account.device_capacity_bytes,
            effective_device_usable_ceiling_bytes,
            process_claimed_bytes: state.claimed_bytes,
            budget_device_wide_usable_ceiling_bytes: self
                .pools
                .budget
                .device_wide_usable_ceiling_bytes,
            budget_claimed_bytes,
            pools,
        })
    }

    /// Makes the core-derived runnable minimum resident. A second call is a
    /// no-op and returns `None`; it never creates a duplicate initial chunk.
    pub fn initialize_pool(
        &self,
        pool_id: &DynamicBackingPoolId,
    ) -> Result<Option<DynamicPoolGrowthReceipt>, VNextError> {
        let mut receipt = self
            .pools
            .maintain_pools(vec![DynamicPoolGrowthIntent::Minimum(pool_id.clone())])?;
        Ok(receipt.growths.pop())
    }

    /// Initializes several pools in one canonical all-or-nothing publication.
    pub fn initialize_pools(
        &self,
        pool_ids: &[DynamicBackingPoolId],
    ) -> Result<DynamicPoolGrowthBatchReceipt, VNextError> {
        self.pools.maintain_pools(
            pool_ids
                .iter()
                .cloned()
                .map(DynamicPoolGrowthIntent::Minimum)
                .collect(),
        )
    }

    /// Adds one explicitly-sized resident chunk. The physical allocation is
    /// globally reserved before the runtime is called and published only after
    /// allocation, descriptor validation, and installation all succeed.
    pub fn grow_pool(
        &self,
        pool_id: &DynamicBackingPoolId,
        requested_bytes: u64,
    ) -> Result<DynamicPoolGrowthReceipt, VNextError> {
        let request = DynamicPoolGrowthRequest::new(pool_id.clone(), requested_bytes)?;
        let mut receipt = self.grow_pools(vec![request])?;
        receipt
            .growths
            .pop()
            .ok_or_else(|| invalid_resource("single-pool growth produced no receipt"))
    }

    /// Grows distinct pools under one global reservation and one capacity
    /// epoch publication. Input order is ignored; duplicate pools are rejected.
    pub fn grow_pools(
        &self,
        requests: Vec<DynamicPoolGrowthRequest>,
    ) -> Result<DynamicPoolGrowthBatchReceipt, VNextError> {
        self.pools.maintain_pools(
            requests
                .into_iter()
                .map(DynamicPoolGrowthIntent::Additional)
                .collect(),
        )
    }

    /// Resolves a still-current physical deferral by pool identity. If any
    /// release or capacity event happened since observation, no allocation is
    /// performed and the caller must retry admission against the new state.
    pub fn maintain_for_deferred(
        &self,
        deferred: &DynamicBackingDeferred,
    ) -> Result<DynamicDeferredMaintenanceOutcome, VNextError> {
        let current_epochs = self.pools.logical_admission.epochs()?;
        if current_epochs.coordinator_id() != deferred.epochs().coordinator_id() {
            return Err(invalid_resource(
                "dynamic backing deferral belongs to another admission coordinator",
            ));
        }
        if current_epochs != deferred.epochs() {
            return Ok(DynamicDeferredMaintenanceOutcome::RetryWithoutGrowth { current_epochs });
        }
        if deferred.blockers().is_empty() {
            return Err(invalid_resource(
                "dynamic backing deferral contains no blocking pool",
            ));
        }
        let mut requested_by_pool = BTreeMap::<DynamicBackingPoolId, u64>::new();
        for blocker in deferred.blockers() {
            if !self.pools.pools.contains_key(blocker.pool_id()) {
                return Err(invalid_resource(
                    "dynamic backing deferral belongs to another plan owner",
                ));
            }
            requested_by_pool
                .entry(blocker.pool_id().clone())
                .and_modify(|bytes| *bytes = (*bytes).max(blocker.requested_bytes()))
                .or_insert(blocker.requested_bytes());
        }
        let receipt = self.grow_pools(
            requested_by_pool
                .into_iter()
                .map(|(pool_id, bytes)| DynamicPoolGrowthRequest::new(pool_id, bytes))
                .collect::<Result<Vec<_>, _>>()?,
        )?;
        Ok(DynamicDeferredMaintenanceOutcome::Maintained(receipt))
    }

    /// Explicitly releases only unclaimable quarantined chunks. Resident
    /// chunks and logical totals are unchanged; the returned grants are
    /// dropped after all pool locks have been released.
    pub fn release_quarantined_chunks(
        &self,
    ) -> Result<DynamicPoolQuarantineReleaseReceipt, VNextError> {
        let pools = self.pools.pools.values().cloned().collect::<Vec<_>>();
        let _maintenance = pools
            .iter()
            .map(|pool| {
                pool.maintenance
                    .lock()
                    .map_err(|_| invalid_resource("dynamic pool maintenance authority is poisoned"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut states = pools
            .iter()
            .map(|pool| {
                pool.state
                    .lock()
                    .map_err(|_| invalid_resource("dynamic backing pool is poisoned"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let pool_totals = states
            .iter()
            .map(|state| {
                let bytes = state.quarantined.iter().try_fold(0_u64, |total, chunk| {
                    total
                        .checked_add(chunk.backing._grant.bytes())
                        .ok_or_else(|| invalid_resource("released quarantine bytes overflow u64"))
                })?;
                Ok((state.quarantined.len(), bytes))
            })
            .collect::<Result<Vec<_>, VNextError>>()?;
        let released_chunks = pool_totals.iter().try_fold(0_usize, |total, (count, _)| {
            total
                .checked_add(*count)
                .ok_or_else(|| invalid_resource("released quarantine count overflows usize"))
        })?;
        let released_bytes = pool_totals.iter().try_fold(0_u64, |total, (_, bytes)| {
            total
                .checked_add(*bytes)
                .ok_or_else(|| invalid_resource("released quarantine bytes overflow u64"))
        })?;
        let mut released = Vec::with_capacity(pools.len());
        let mut receipts = Vec::new();
        for ((pool, state), (pool_chunks, pool_bytes)) in
            pools.iter().zip(states.iter_mut()).zip(pool_totals)
        {
            if pool_chunks == 0 {
                continue;
            }
            let chunks = std::mem::take(&mut state.quarantined);
            debug_assert_eq!(chunks.len(), pool_chunks);
            receipts.push(DynamicPoolQuarantineRelease {
                pool_id: pool.domain.pool_id().clone(),
                released_chunks: pool_chunks,
                released_bytes: pool_bytes,
            });
            released.push(chunks);
        }
        drop(states);
        drop(released);
        Ok(DynamicPoolQuarantineReleaseReceipt {
            pools: receipts,
            released_chunks,
            released_bytes,
        })
    }
}

impl DynamicPoolGrowthReceipt {
    pub fn pool_id(&self) -> &DynamicBackingPoolId {
        &self.pool_id
    }

    pub fn chunk(&self) -> &BackingChunkIdentity {
        &self.chunk
    }

    pub const fn chunk_bytes(&self) -> u64 {
        self.chunk_bytes
    }

    pub const fn published_capacity_bytes(&self) -> u64 {
        self.published_capacity_bytes
    }

    pub const fn capacity_epoch(&self) -> u64 {
        self.capacity_epoch
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct DynamicPoolGrowthRequest {
    pool_id: DynamicBackingPoolId,
    requested_bytes: u64,
}

impl DynamicPoolGrowthRequest {
    pub fn new(pool_id: DynamicBackingPoolId, requested_bytes: u64) -> Result<Self, VNextError> {
        if requested_bytes == 0 {
            return Err(invalid_resource(
                "dynamic pool growth must request non-zero bytes",
            ));
        }
        Ok(Self {
            pool_id,
            requested_bytes,
        })
    }

    pub fn pool_id(&self) -> &DynamicBackingPoolId {
        &self.pool_id
    }

    pub const fn requested_bytes(&self) -> u64 {
        self.requested_bytes
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct DynamicPoolGrowthBatchReceipt {
    growths: Vec<DynamicPoolGrowthReceipt>,
    capacity_epoch: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DynamicBackingDeferralReason {
    GrowthRequired,
    FragmentedContiguous,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct DynamicBackingBlocker {
    pool_id: DynamicBackingPoolId,
    reason: DynamicBackingDeferralReason,
    requested_bytes: u64,
    free_bytes: u64,
    largest_contiguous_bytes: u64,
}

impl DynamicBackingBlocker {
    pub fn pool_id(&self) -> &DynamicBackingPoolId {
        &self.pool_id
    }

    pub const fn reason(&self) -> DynamicBackingDeferralReason {
        self.reason
    }

    pub const fn requested_bytes(&self) -> u64 {
        self.requested_bytes
    }

    pub const fn free_bytes(&self) -> u64 {
        self.free_bytes
    }

    pub const fn largest_contiguous_bytes(&self) -> u64 {
        self.largest_contiguous_bytes
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct DynamicBackingDeferred {
    blockers: Vec<DynamicBackingBlocker>,
    epochs: CapacityEpochs,
}

impl DynamicBackingDeferred {
    pub fn blockers(&self) -> &[DynamicBackingBlocker] {
        &self.blockers
    }

    pub const fn release_epoch(&self) -> u64 {
        self.epochs.release_epoch()
    }

    pub const fn capacity_epoch(&self) -> u64 {
        self.epochs.capacity_epoch()
    }

    pub const fn epochs(&self) -> CapacityEpochs {
        self.epochs
    }
}

impl DynamicPoolGrowthBatchReceipt {
    pub fn growths(&self) -> &[DynamicPoolGrowthReceipt] {
        &self.growths
    }

    pub const fn capacity_epoch(&self) -> u64 {
        self.capacity_epoch
    }
}

enum DynamicPoolGrowthIntent {
    Additional(DynamicPoolGrowthRequest),
    Minimum(DynamicBackingPoolId),
}

impl DynamicPoolGrowthIntent {
    fn pool_id(&self) -> &DynamicBackingPoolId {
        match self {
            Self::Additional(request) => request.pool_id(),
            Self::Minimum(pool_id) => pool_id,
        }
    }
}

struct PlannedDynamicGrowth<R>
where
    R: DeviceRuntime,
{
    pool: Arc<DynamicBackingPool<R>>,
    chunk: BackingChunkIdentity,
    expected_resource_id: ResourceId,
    chunk_bytes: u64,
}

struct AllocatedDynamicGrowth<B> {
    backing: Arc<ResidentChunkBacking<B>>,
}

struct EvaluatedBackingRequest<'a> {
    domain: &'a DynamicPoolDomainSpec,
    descriptor: &'a DynamicResourceDescriptor,
    size_bytes: u64,
}

struct PreparedBackingSlice<R>
where
    R: DeviceRuntime,
{
    pool: Arc<DynamicBackingPool<R>>,
    evidence: LogicalBackingSliceEvidence,
}

struct PreparedBackingClaim<R>
where
    R: DeviceRuntime,
{
    slices: Vec<PreparedBackingSlice<R>>,
    committed: bool,
}

impl<R> PreparedBackingClaim<R>
where
    R: DeviceRuntime,
{
    fn empty() -> Self {
        Self {
            slices: Vec::new(),
            committed: false,
        }
    }

    fn commit(mut self) -> Vec<LogicalBackingSliceAuthority> {
        let slices = std::mem::take(&mut self.slices)
            .into_iter()
            .map(|slice| {
                let owner: Arc<dyn BackingExtentOwner> = slice.pool;
                let segment_generation = slice.evidence.segment_generation;
                let segments = slice.evidence.segments.clone();
                LogicalBackingSliceAuthority {
                    evidence: slice.evidence,
                    segment_lease: BackingSegmentLease {
                        owner_instance_id: owner.instance_id(),
                        owner,
                        segment_generation,
                        segments,
                        released: false,
                    },
                }
            })
            .collect();
        self.committed = true;
        slices
    }
}

impl<R> Drop for PreparedBackingClaim<R>
where
    R: DeviceRuntime,
{
    fn drop(&mut self) {
        if self.committed {
            return;
        }
        for slice in self.slices.iter().rev() {
            slice.pool.rollback_prepared(&slice.evidence.segments);
        }
    }
}

enum BackingPrepareDecision<R>
where
    R: DeviceRuntime,
{
    Prepared(PreparedBackingClaim<R>),
    Deferred(DynamicBackingDeferred),
}

struct DynamicPoolSet<R>
where
    R: DeviceRuntime,
{
    pools: BTreeMap<DynamicBackingPoolId, Arc<DynamicBackingPool<R>>>,
    domains: Vec<DynamicPoolDomainSpec>,
    nodes: Arc<[PlanNode]>,
    logical_admission: LogicalAdmissionCoordinator,
    budget: Arc<DeviceCapacityBudget>,
    binding: StaticProvisioningBinding,
    // Backend context must outlive every resident/quarantined buffer above.
    runtime: Arc<R>,
}

impl<R> DynamicPoolSet<R>
where
    R: DeviceRuntime,
{
    fn new(
        runtime: Arc<R>,
        binding: StaticProvisioningBinding,
        budget: Arc<DeviceCapacityBudget>,
        logical_admission: LogicalAdmissionCoordinator,
        domains: Vec<DynamicPoolDomainSpec>,
        nodes: Arc<[PlanNode]>,
    ) -> Result<Self, VNextError> {
        let mut pools = BTreeMap::new();
        for domain in &domains {
            let instance_id = NEXT_DYNAMIC_POOL_INSTANCE_ID
                .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
                    current.checked_add(1)
                })
                .map_err(|_| invalid_resource("dynamic pool instance id space is exhausted"))?;
            let pool = Arc::new(DynamicBackingPool {
                instance_id,
                domain: domain.clone(),
                logical_admission: logical_admission.clone(),
                maintenance: Mutex::new(()),
                next_extent_generation: AtomicU64::new(1),
                state: Mutex::new(DynamicBackingPoolState {
                    resident_bytes: 0,
                    pending_growth_bytes: 0,
                    next_chunk_ordinal: 1,
                    next_chunk_generation: 1,
                    chunks: BTreeMap::new(),
                    allocator: FreeExtentIndex::default(),
                    quarantined: Vec::new(),
                    poisoned: false,
                }),
            });
            if pools.insert(domain.pool_id().clone(), pool).is_some() {
                return Err(invalid_resource(
                    "dynamic pool set contains a duplicate pool",
                ));
            }
        }
        Ok(Self {
            runtime,
            binding,
            budget,
            logical_admission,
            domains,
            pools,
            nodes,
        })
    }

    fn maintain_pools(
        &self,
        mut intents: Vec<DynamicPoolGrowthIntent>,
    ) -> Result<DynamicPoolGrowthBatchReceipt, VNextError> {
        intents.sort_by(|left, right| left.pool_id().cmp(right.pool_id()));
        if intents
            .windows(2)
            .any(|pair| pair[0].pool_id() == pair[1].pool_id())
        {
            return Err(invalid_resource(
                "dynamic maintenance batch contains a duplicate pool",
            ));
        }
        let pools = intents
            .iter()
            .map(|intent| {
                self.pools.get(intent.pool_id()).cloned().ok_or_else(|| {
                    invalid_resource("dynamic maintenance references an unknown pool")
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let _maintenance = pools
            .iter()
            .map(|pool| {
                pool.maintenance
                    .lock()
                    .map_err(|_| invalid_resource("dynamic pool maintenance authority is poisoned"))
            })
            .collect::<Result<Vec<_>, _>>()?;

        let mut planned = Vec::with_capacity(intents.len());
        let mut pending = Vec::with_capacity(intents.len());
        for (intent, pool) in intents.iter().zip(&pools) {
            let requested_bytes = {
                let state = pool
                    .state
                    .lock()
                    .map_err(|_| invalid_resource("dynamic backing pool is poisoned"))?;
                if state.poisoned {
                    return Err(invalid_resource("dynamic backing pool is fail-closed"));
                }
                match intent {
                    DynamicPoolGrowthIntent::Additional(request) => request.requested_bytes(),
                    DynamicPoolGrowthIntent::Minimum(_) => {
                        let current = state
                            .resident_bytes
                            .checked_add(state.pending_growth_bytes)
                            .ok_or_else(|| {
                                invalid_resource("dynamic pool residency overflows u64")
                            })?;
                        let minimum = pool.domain.pool.provisioning().minimum_resident_bytes();
                        if current >= minimum {
                            continue;
                        }
                        minimum - current
                    }
                }
            };
            let chunk_bytes = align_up_resource(requested_bytes, pool.allocation_quantum())?;
            let chunk = {
                let mut state = pool
                    .state
                    .lock()
                    .map_err(|_| invalid_resource("dynamic backing pool is poisoned"))?;
                if state.poisoned {
                    return Err(invalid_resource("dynamic backing pool is fail-closed"));
                }
                let next_residency = state
                    .resident_bytes
                    .checked_add(state.pending_growth_bytes)
                    .and_then(|bytes| bytes.checked_add(chunk_bytes))
                    .ok_or_else(|| invalid_resource("dynamic pool resident bytes overflow u64"))?;
                if next_residency > pool.domain.pool.provisioning().maximum_resident_bytes() {
                    return Err(invalid_resource(
                        "dynamic pool growth exceeds its core-derived resident maximum",
                    ));
                }
                let ordinal = state.next_chunk_ordinal;
                let generation = state.next_chunk_generation;
                let next_ordinal = ordinal
                    .checked_add(1)
                    .ok_or_else(|| invalid_resource("dynamic chunk ordinal space is exhausted"))?;
                let next_generation = generation.checked_add(1).ok_or_else(|| {
                    invalid_resource("dynamic chunk generation space is exhausted")
                })?;
                let next_pending = state
                    .pending_growth_bytes
                    .checked_add(chunk_bytes)
                    .ok_or_else(|| invalid_resource("pending dynamic growth bytes overflow u64"))?;
                state.next_chunk_ordinal = next_ordinal;
                state.next_chunk_generation = next_generation;
                state.pending_growth_bytes = next_pending;
                BackingChunkIdentity {
                    pool_id: pool.domain.pool_id().clone(),
                    ordinal,
                    generation,
                }
            };
            pending.push(PendingGrowthGuard {
                pool: Arc::clone(pool),
                bytes: chunk_bytes,
                armed: true,
            });
            let expected_resource_id = ResourceId::new(format!(
                "{}/chunk/{}/{}",
                pool.domain.pool_id().as_str(),
                chunk.ordinal,
                chunk.generation
            ))?;
            planned.push(PlannedDynamicGrowth {
                pool: Arc::clone(pool),
                chunk,
                expected_resource_id,
                chunk_bytes,
            });
        }
        if planned.is_empty() {
            return Ok(DynamicPoolGrowthBatchReceipt {
                growths: Vec::new(),
                capacity_epoch: self.logical_admission.epochs()?.capacity_epoch(),
            });
        }

        let total_bytes = planned.iter().try_fold(0_u64, |total, growth| {
            total
                .checked_add(growth.chunk_bytes)
                .ok_or_else(|| invalid_resource("dynamic maintenance batch bytes overflow u64"))
        })?;
        let reservation = DeviceCapacityReservation::reserve(&self.budget, total_bytes)?;
        let grants = reservation.commit_split(
            &planned
                .iter()
                .map(|growth| growth.chunk_bytes)
                .collect::<Vec<_>>(),
        )?;
        let mut allocated = Vec::with_capacity(planned.len());
        for (growth, grant) in planned.iter().zip(grants) {
            let transaction_identity = ResourceTransactionIdentity {
                pool_id: self.binding.pool_id(),
                run_id: RunId::new(format!("dynamic-grow-{}", growth.chunk.generation))?,
                transaction_id: TransactionId::new(format!(
                    "dynamic-grow-{}-{}",
                    growth.chunk.ordinal, growth.chunk.generation
                ))?,
                request_id: self.binding.request_id().clone(),
            };
            let reservation_evidence = ResourceReservation {
                resource_id: growth.expected_resource_id.clone(),
                request_id: self.binding.request_id().clone(),
                owner_node_id: None,
                size_bytes: growth.chunk_bytes,
                alignment_bytes: growth.pool.domain.pool.compatibility().alignment_bytes(),
                usage: growth.pool.domain.pool.compatibility().usage(),
                element_type: growth.pool.domain.pool.compatibility().element_type(),
                retention_policy: ResourceRetentionPolicy::Plan,
                backing_domain_id: Some(growth.pool.domain.domain_id),
                generation: growth.chunk.generation,
            };
            let request = BufferRequest::new(
                growth.expected_resource_id.clone(),
                growth.chunk_bytes,
                reservation_evidence.alignment_bytes,
                reservation_evidence.usage,
                reservation_evidence.element_type,
            )?;
            validate_runtime_descriptor_for_admission(
                self.runtime.descriptor(),
                &self.binding,
                "dynamic pool batch growth preflight",
            )?;
            let buffer = self
                .runtime
                .allocate(DeviceAllocationPermit {
                    identity: &transaction_identity,
                    binding: &self.binding,
                    reservation: &reservation_evidence,
                    request: &request,
                    seal: AllocationSeal,
                })
                .map_err(|error| {
                    invalid_resource(format!("dynamic pool device allocation failed: {error}"))
                })?;
            let actual_descriptor = self.runtime.buffer_descriptor(&buffer);
            validate_runtime_descriptor_for_admission(
                self.runtime.descriptor(),
                &self.binding,
                "dynamic pool batch growth completion",
            )?;
            if grant.bytes() != growth.chunk_bytes {
                return Err(invalid_resource(
                    "dynamic pool capacity grant differs from its chunk",
                ));
            }
            let backing = Arc::new(ResidentChunkBacking {
                buffer,
                _grant: grant,
                identity: growth.chunk.clone(),
                descriptor: actual_descriptor.clone(),
            });
            if !reservation_evidence.matches_descriptor(&actual_descriptor) {
                let mut state = growth
                    .pool
                    .state
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                state.quarantined.push(QuarantinedDynamicChunk {
                    backing,
                    reason: DynamicChunkQuarantineReason::DescriptorMismatch,
                });
                return Err(invalid_resource(
                    "dynamic chunk descriptor mismatch was quarantined without capacity publication",
                ));
            }
            allocated.push(AllocatedDynamicGrowth { backing });
        }

        let mut states = planned
            .iter()
            .map(|growth| {
                growth.pool.state.lock().map_err(|_| {
                    invalid_resource("dynamic backing pool is poisoned after allocation")
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut published_totals = Vec::with_capacity(planned.len());
        for ((growth, allocation), state) in planned.iter().zip(&allocated).zip(&states) {
            if state.poisoned
                || state.pending_growth_bytes < growth.chunk_bytes
                || state.chunks.contains_key(&growth.chunk.ordinal)
                || state
                    .allocator
                    .by_offset
                    .range((growth.chunk.ordinal, 0)..=(growth.chunk.ordinal, u64::MAX))
                    .next()
                    .is_some()
                || state
                    .allocator
                    .free_bytes
                    .checked_add(growth.chunk_bytes)
                    .is_none()
                || allocation.backing.identity != growth.chunk
            {
                return Err(invalid_resource(
                    "dynamic batch installation preconditions changed before publication",
                ));
            }
            published_totals.push(
                state
                    .resident_bytes
                    .checked_add(growth.chunk_bytes)
                    .ok_or_else(|| invalid_resource("published dynamic capacity overflows u64"))?,
            );
        }
        for index in 0..planned.len() {
            let growth = &planned[index];
            let state = &mut states[index];
            state.pending_growth_bytes -= growth.chunk_bytes;
            pending[index].disarm();
            state
                .allocator
                .insert_extent(
                    growth.chunk.ordinal,
                    growth.chunk.generation,
                    0,
                    growth.chunk_bytes,
                )
                .expect("validated new chunk has one disjoint free extent");
            state.chunks.insert(
                growth.chunk.ordinal,
                ResidentChunkState {
                    backing: Arc::clone(&allocated[index].backing),
                    live_segments: 0,
                },
            );
        }
        let updates = planned
            .iter()
            .zip(&published_totals)
            .map(|(growth, &total)| (growth.pool.domain.domain_id, CapacityUnits::new(total)))
            .collect::<Vec<_>>();
        let epochs = match self.logical_admission.set_domain_totals(&updates) {
            Ok(epochs) => epochs,
            Err(error) => {
                for index in 0..planned.len() {
                    states[index]
                        .allocator
                        .remove_extent(planned[index].chunk.ordinal, 0)
                        .expect("unpublished dynamic chunk free extent remains installed");
                    let removed = states[index]
                        .chunks
                        .remove(&planned[index].chunk.ordinal)
                        .expect("unpublished dynamic chunk remains installed");
                    states[index].quarantined.push(QuarantinedDynamicChunk {
                        backing: removed.backing,
                        reason: DynamicChunkQuarantineReason::PublicationRejected,
                    });
                }
                return Err(error);
            }
        };
        for (state, &published_total) in states.iter_mut().zip(&published_totals) {
            state.resident_bytes = published_total;
        }
        Ok(DynamicPoolGrowthBatchReceipt {
            growths: planned
                .iter()
                .zip(published_totals)
                .map(
                    |(growth, published_capacity_bytes)| DynamicPoolGrowthReceipt {
                        pool_id: growth.pool.domain.pool_id().clone(),
                        chunk: growth.chunk.clone(),
                        chunk_bytes: growth.chunk_bytes,
                        published_capacity_bytes,
                        capacity_epoch: epochs.capacity_epoch(),
                    },
                )
                .collect(),
            capacity_epoch: epochs.capacity_epoch(),
        })
    }

    fn prepare_claim(
        &self,
        requests: &[EvaluatedBackingRequest<'_>],
    ) -> Result<BackingPrepareDecision<R>, VNextError> {
        if requests.is_empty() {
            return Ok(BackingPrepareDecision::Prepared(
                PreparedBackingClaim::empty(),
            ));
        }
        let mut grouped =
            BTreeMap::<DynamicBackingPoolId, Vec<&EvaluatedBackingRequest<'_>>>::new();
        for request in requests {
            grouped
                .entry(request.domain.pool_id().clone())
                .or_default()
                .push(request);
        }
        let mut groups = Vec::with_capacity(grouped.len());
        for (pool_id, mut requests) in grouped {
            requests.sort_by(|left, right| {
                left.descriptor
                    .base_resource_id()
                    .cmp(right.descriptor.base_resource_id())
            });
            let pool = self.pools.get(&pool_id).cloned().ok_or_else(|| {
                invalid_resource("dynamic backing reservation references an unknown pool")
            })?;
            groups.push((pool, requests));
        }
        let segment_generations = groups
            .iter()
            .map(|(pool, requests)| {
                (0..requests.len())
                    .map(|_| {
                        pool.next_extent_generation
                            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |current| {
                                current.checked_add(1)
                            })
                            .map_err(|_| {
                                invalid_resource("dynamic extent generation space is exhausted")
                            })
                    })
                    .collect::<Result<Vec<_>, _>>()
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut states = groups
            .iter()
            .map(|(pool, _)| {
                pool.state
                    .lock()
                    .map_err(|_| invalid_resource("dynamic backing pool is poisoned"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        for (group_index, (pool, pool_requests)) in groups.iter().enumerate() {
            if states[group_index].poisoned {
                return Err(invalid_resource("dynamic backing pool is fail-closed"));
            }
            let quantum = pool.allocation_quantum();
            for request in pool_requests {
                if request.domain.pool_id() != pool.domain.pool_id()
                    || request.descriptor.pool_id() != pool.domain.pool_id()
                    || request.size_bytes == 0
                    || request.size_bytes % quantum != 0
                {
                    return Err(invalid_resource(
                        "dynamic backing request violates its exact pool or allocation quantum",
                    ));
                }
            }
        }
        let mut selections = groups
            .iter()
            .map(|_| Vec::<(&EvaluatedBackingRequest<'_>, u64, Vec<BackingSegment>)>::new())
            .collect::<Vec<_>>();
        let mut journals = groups
            .iter()
            .map(|_| Vec::<Vec<BackingSegment>>::new())
            .collect::<Vec<_>>();
        for group_index in 0..groups.len() {
            let (pool, pool_requests) = &groups[group_index];
            let profile = pool.domain.pool.compatibility().profile();
            for request in pool_requests {
                let reserved = match match profile.view() {
                    DynamicStorageView::Contiguous => states[group_index]
                        .allocator
                        .allocate_contiguous(pool.domain.pool_id(), request.size_bytes)
                        .map(|segment| segment.map(|segment| vec![segment])),
                    DynamicStorageView::PagedRegions { block_bytes } => states[group_index]
                        .allocator
                        .allocate_paged(pool.domain.pool_id(), request.size_bytes, block_bytes),
                } {
                    Ok(reserved) => reserved,
                    Err(error) => {
                        states[group_index].poisoned = true;
                        rollback_free_extent_journal(&mut states, &journals)?;
                        return Err(error);
                    }
                };
                let Some(segments) = reserved else {
                    let blocker = DynamicBackingBlocker {
                        pool_id: pool.domain.pool_id().clone(),
                        reason: if states[group_index].allocator.free_bytes < request.size_bytes {
                            DynamicBackingDeferralReason::GrowthRequired
                        } else {
                            DynamicBackingDeferralReason::FragmentedContiguous
                        },
                        requested_bytes: request.size_bytes,
                        free_bytes: states[group_index].allocator.free_bytes,
                        largest_contiguous_bytes: states[group_index]
                            .allocator
                            .largest_contiguous_bytes(),
                    };
                    rollback_free_extent_journal(&mut states, &journals)?;
                    drop(states);
                    return Ok(BackingPrepareDecision::Deferred(DynamicBackingDeferred {
                        blockers: vec![blocker],
                        epochs: self.logical_admission.epochs()?,
                    }));
                };
                journals[group_index].push(segments.clone());
                let extent_bytes = match segments.iter().try_fold(0_u64, |total, segment| {
                    total.checked_add(segment.length_bytes()).ok_or_else(|| {
                        invalid_resource("dynamic backing extent bytes overflow u64")
                    })
                }) {
                    Ok(bytes) => bytes,
                    Err(error) => {
                        rollback_free_extent_journal(&mut states, &journals)?;
                        return Err(error);
                    }
                };
                if extent_bytes != request.size_bytes {
                    rollback_free_extent_journal(&mut states, &journals)?;
                    return Err(invalid_resource(
                        "dynamic backing extents differ from their exact logical claim",
                    ));
                }
                let generation = segment_generations[group_index][selections[group_index].len()];
                selections[group_index].push((request, generation, segments));
            }
        }

        let increments = match (0..groups.len())
            .map(|group_index| {
                let mut increments = BTreeMap::<u32, u64>::new();
                for (_, _, segments) in &selections[group_index] {
                    for segment in segments {
                        let count = increments.entry(segment.chunk_ordinal()).or_default();
                        *count = count.checked_add(1).ok_or_else(|| {
                            invalid_resource("dynamic chunk live extent increment overflows u64")
                        })?;
                    }
                }
                for (&ordinal, &increment) in &increments {
                    states[group_index]
                        .chunks
                        .get(&ordinal)
                        .ok_or_else(|| invalid_resource("reserved dynamic chunk disappeared"))?
                        .live_segments
                        .checked_add(increment)
                        .ok_or_else(|| {
                            invalid_resource("dynamic chunk live extent count overflowed")
                        })?;
                }
                Ok(increments)
            })
            .collect::<Result<Vec<_>, VNextError>>()
        {
            Ok(increments) => increments,
            Err(error) => {
                rollback_free_extent_journal(&mut states, &journals)?;
                return Err(error);
            }
        };
        for (group_index, increments) in increments.into_iter().enumerate() {
            for (ordinal, increment) in increments {
                states[group_index]
                    .chunks
                    .get_mut(&ordinal)
                    .expect("validated reserved dynamic chunk remains installed")
                    .live_segments += increment;
            }
        }
        drop(states);
        let slices = groups
            .into_iter()
            .zip(selections)
            .flat_map(|((pool, _), selections)| {
                selections
                    .into_iter()
                    .map(
                        move |(request, segment_generation, segments)| PreparedBackingSlice {
                            pool: Arc::clone(&pool),
                            evidence: LogicalBackingSliceEvidence {
                                domain_id: pool.domain.domain_id,
                                pool_id: pool.domain.pool_id().clone(),
                                resource_id: request.descriptor.base_resource_id().clone(),
                                pool_instance_id: pool.instance_id,
                                segment_generation,
                                segments,
                                size_bytes: request.size_bytes,
                                alignment_bytes: request.descriptor.alignment_bytes(),
                                usage: request.descriptor.usage(),
                                element_type: request.descriptor.element_type(),
                                storage_profile: pool.domain.pool.compatibility().profile(),
                            },
                        },
                    )
            })
            .collect();
        Ok(BackingPrepareDecision::Prepared(PreparedBackingClaim {
            slices,
            committed: false,
        }))
    }

    fn view<'lease>(
        &'lease self,
        authority: &'lease LogicalBackingSliceAuthority,
    ) -> Result<LogicalBackingBufferView<'lease, R::Buffer>, VNextError> {
        let pool = self
            .pools
            .get(&authority.evidence.pool_id)
            .ok_or_else(|| invalid_resource("logical backing authority has no dynamic pool"))?;
        if pool.instance_id != authority.evidence.pool_instance_id
            || authority.segment_lease.owner_instance_id != pool.instance_id
            || authority.segment_lease.owner.instance_id() != pool.instance_id
            || authority.segment_lease.segment_generation != authority.evidence.segment_generation
        {
            return Err(invalid_resource(
                "logical backing authority belongs to another dynamic pool instance",
            ));
        }
        let state = pool
            .state
            .lock()
            .map_err(|_| invalid_resource("dynamic backing pool is poisoned"))?;
        if state.poisoned {
            return Err(invalid_resource("dynamic backing pool is fail-closed"));
        }
        let mut bindings = Vec::with_capacity(authority.evidence.segments.len());
        for segment in &authority.evidence.segments {
            let chunk = state
                .chunks
                .get(&segment.chunk_ordinal())
                .ok_or_else(|| invalid_resource("logical backing references a missing chunk"))?;
            if segment.pool_id() != &authority.evidence.pool_id
                || chunk.backing.identity != *segment.chunk()
                || segment
                    .offset_bytes()
                    .checked_add(segment.length_bytes())
                    .is_none_or(|end| end > chunk.backing.descriptor.size_bytes)
            {
                return Err(invalid_resource(
                    "logical backing references a stale or out-of-bounds chunk region",
                ));
            }
            bindings.push(LogicalBackingSegmentBinding {
                segment: segment.clone(),
                chunk: Arc::clone(&chunk.backing),
            });
        }
        drop(state);
        Ok(LogicalBackingBufferView {
            bindings,
            evidence: &authority.evidence,
        })
    }
}

impl<R> DynamicBackingPool<R>
where
    R: DeviceRuntime,
{
    fn allocation_quantum(&self) -> u64 {
        match self.domain.pool.compatibility().profile().allocator() {
            DynamicStorageAllocator::LinearArena => {
                self.domain.pool.compatibility().alignment_bytes()
            }
            DynamicStorageAllocator::FixedBlockArena { block_bytes } => {
                block_bytes.max(self.domain.pool.compatibility().alignment_bytes())
            }
        }
    }

    fn cancel_pending_growth(&self, bytes: u64) {
        let mut state = match self.state.lock() {
            Ok(state) => state,
            Err(poisoned) => poisoned.into_inner(),
        };
        if state.pending_growth_bytes < bytes {
            state.poisoned = true;
            return;
        }
        state.pending_growth_bytes -= bytes;
    }

    fn rollback_prepared(&self, segments: &[BackingSegment]) {
        let mut state = match self.state.lock() {
            Ok(state) => state,
            Err(poisoned) => {
                let mut state = poisoned.into_inner();
                state.poisoned = true;
                return;
            }
        };
        if state.poisoned {
            return;
        }
        for segment in segments.iter().rev() {
            let valid = state
                .chunks
                .get(&segment.chunk_ordinal())
                .is_some_and(|chunk| {
                    chunk.backing.identity == *segment.chunk() && chunk.live_segments != 0
                });
            if !valid || state.allocator.release(segment).is_err() {
                state.poisoned = true;
                return;
            }
            state
                .chunks
                .get_mut(&segment.chunk_ordinal())
                .expect("validated prepared chunk remains installed")
                .live_segments -= 1;
        }
        drop(state);
        if self
            .logical_admission
            .notify_domain_availability_changed(self.domain.domain_id)
            .is_err()
        {
            let mut state = self
                .state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            state.poisoned = true;
        }
    }
}

#[must_use = "a resource lease is the batch owner of committed buffers"]
pub struct StaticProvisioningLease<R>
where
    R: DeviceRuntime,
{
    slots: Vec<OwnedLeaseSlot<R::Buffer>>,
    identity: ResourceTransactionIdentity,
    admission: StaticProvisioningBinding,
    // Backend context drops after all static and dynamic buffers.
    runtime: Arc<R>,
}

impl<R> StaticProvisioningLease<R>
where
    R: DeviceRuntime,
{
    fn new(
        runtime: Arc<R>,
        identity: &ResourceTransactionIdentity,
        admission: &StaticProvisioningBinding,
        reservations: &ResourceReservationBatch,
    ) -> Self {
        Self {
            slots: reservations
                .reservations()
                .iter()
                .map(OwnedLeaseSlot::new)
                .collect(),
            identity: identity.clone(),
            admission: admission.clone(),
            runtime,
        }
    }

    pub fn identity(&self) -> &ResourceTransactionIdentity {
        &self.identity
    }

    pub fn admission(&self) -> &StaticProvisioningBinding {
        &self.admission
    }

    pub fn state(&self) -> ResourceLeaseState {
        let mut states = self
            .slots
            .iter()
            .filter(|slot| slot.buffer.is_some())
            .map(|slot| slot.entry.state);
        let Some(first) = states.next() else {
            return ResourceLeaseState::Cancelled;
        };
        if states.all(|state| state == first) {
            first
        } else {
            ResourceLeaseState::Mixed
        }
    }

    pub fn entries(&self) -> impl Iterator<Item = &ResourceLeaseEntry> {
        self.slots
            .iter()
            .filter(|slot| slot.buffer.is_some())
            .map(|slot| &slot.entry)
    }

    pub fn plan_static_entries(&self) -> impl Iterator<Item = &ResourceLeaseEntry> {
        self.slots
            .iter()
            .filter(|slot| slot.buffer.is_some())
            .map(|slot| &slot.entry)
    }

    pub(crate) fn view(
        &self,
        resource_id: &ResourceId,
        generation: u64,
    ) -> Result<LeasedBufferView<'_, R::Buffer>, VNextError> {
        let slot = self
            .slots
            .iter()
            .find(|slot| {
                slot.entry.resource_id == *resource_id && slot.entry.generation == generation
            })
            .ok_or_else(|| invalid_resource("lease does not contain that resource generation"))?;
        if slot.entry.state != ResourceLeaseState::Active {
            return Err(VNextError::InvalidLeaseTransition {
                lease_id: self.identity.transaction_id.to_string(),
                from: slot.entry.state.as_str(),
                action: "borrow_live_buffer",
            });
        }
        let descriptor = slot
            .descriptor
            .as_ref()
            .ok_or_else(|| invalid_resource("lease resource is not committed"))?;
        let buffer = slot
            .buffer
            .as_ref()
            .ok_or_else(|| invalid_resource("lease resource buffer is not live"))?;
        Ok(LeasedBufferView {
            identity: &self.identity,
            admission: &self.admission,
            resource_id: &slot.entry.resource_id,
            generation: slot.entry.generation,
            descriptor,
            buffer,
        })
    }

    fn buffer(&self, order: usize) -> Option<&R::Buffer> {
        self.slots.get(order).and_then(|slot| slot.buffer.as_ref())
    }

    fn install(&mut self, order: usize, allocation: CoreOwnedAllocation<R::Buffer>) {
        self.slots[order].install(allocation);
    }

    fn clear(&mut self, order: usize) {
        self.slots[order].clear();
    }

    fn transition_subset(
        &mut self,
        orders: &[usize],
        action: ResourceLeaseAction,
    ) -> Result<
        (
            ResourceLeaseState,
            ResourceLeaseState,
            Vec<ResourceLeaseEntry>,
        ),
        VNextError,
    > {
        if orders.is_empty() {
            return Err(invalid_resource(
                "lease transition subset must not be empty",
            ));
        }
        let mut unique = BTreeSet::new();
        let mut common_before = None;
        for &order in orders {
            if !unique.insert(order) {
                return Err(invalid_resource("lease transition subset is duplicated"));
            }
            let slot = self
                .slots
                .get(order)
                .ok_or_else(|| invalid_resource("lease transition order is out of bounds"))?;
            if slot.buffer.is_none() {
                return Err(invalid_resource(
                    "lease transition targets a non-live buffer",
                ));
            }
            let before = slot.entry.state;
            if common_before.is_some_and(|common| common != before) {
                return Err(invalid_resource(
                    "one lease receipt cannot hide heterogeneous before states",
                ));
            }
            if expected_lease_transition(action, before).is_none() {
                return Err(VNextError::InvalidLeaseTransition {
                    lease_id: self.identity.transaction_id.to_string(),
                    from: before.as_str(),
                    action: action.as_str(),
                });
            }
            common_before = Some(before);
        }
        let before = common_before.expect("non-empty subset has a before state");
        let after = expected_lease_transition(action, before)
            .expect("lease subset was preflight validated");
        for &order in orders {
            self.slots[order].entry.state = after;
        }
        Ok((
            before,
            after,
            orders
                .iter()
                .map(|&order| self.slots[order].entry.clone())
                .collect(),
        ))
    }

    fn take_owned_buffers(
        &mut self,
        reservations: &ResourceReservationBatch,
    ) -> Vec<ResourceOwnedBuffer<R::Buffer>> {
        self.slots
            .iter_mut()
            .zip(reservations.reservations())
            .enumerate()
            .filter_map(|(order, (slot, reservation))| {
                let allocation = slot.take_allocation()?;
                Some(ResourceOwnedBuffer {
                    order,
                    expected_resource_id: reservation.resource_id.clone(),
                    actual_resource_id: allocation.resource_id,
                    expected_generation: reservation.generation,
                    actual_generation: allocation.generation,
                    expected_descriptor: BufferDescriptor {
                        resource_id: reservation.resource_id.clone(),
                        size_bytes: reservation.size_bytes,
                        alignment_bytes: reservation.alignment_bytes,
                        usage: reservation.usage,
                        element_type: reservation.element_type,
                    },
                    actual_descriptor: allocation.descriptor,
                    buffer: allocation.buffer,
                })
            })
            .collect()
    }

    fn restore_owned_buffers(&mut self, buffers: Vec<ResourceOwnedBuffer<R::Buffer>>) {
        for buffer in buffers {
            let (order, allocation) = buffer.into_allocation();
            self.slots[order].restore_allocation(allocation);
        }
    }
}

macro_rules! scoped_resource_admission_request {
    ($name:ident, $single_sequence:literal) => {
        #[derive(Debug, Clone, PartialEq, Eq)]
        pub struct $name {
            work_shape: ResourceWorkShape,
            fit_policy: AdmissionFitPolicy,
            pressure_action: AdmissionPressureAction,
        }

        impl $name {
            pub fn new(
                work_shape: ResourceWorkShape,
                fit_policy: AdmissionFitPolicy,
                pressure_action: AdmissionPressureAction,
            ) -> Result<Self, VNextError> {
                if $single_sequence
                    && (work_shape.immediate_sequences() != 1 || work_shape.fit_sequences() != 1)
                {
                    return Err(invalid_resource(
                        "sequence resource admission requires a single-sequence shape",
                    ));
                }
                Ok(Self {
                    work_shape,
                    fit_policy,
                    pressure_action,
                })
            }

            pub fn work_shape(&self) -> &ResourceWorkShape {
                &self.work_shape
            }

            pub(crate) const fn immediate_shape(&self) -> DynamicResourceShape {
                self.work_shape.immediate_shape()
            }

            pub(crate) const fn fit_shape(&self) -> DynamicResourceShape {
                match self.fit_policy {
                    AdmissionFitPolicy::ImmediateOnly => self.work_shape.immediate_shape(),
                    AdmissionFitPolicy::FullInputMustFit => self.work_shape.fit_shape(),
                }
            }

            pub const fn fit_policy(&self) -> AdmissionFitPolicy {
                self.fit_policy
            }

            pub const fn pressure_action(&self) -> AdmissionPressureAction {
                self.pressure_action
            }
        }
    };
}

scoped_resource_admission_request!(RequestResourceAdmissionRequest, false);
scoped_resource_admission_request!(SequenceResourceAdmissionRequest, true);

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StepResourceAdmissionRequest {
    work_shape: BatchWorkShape,
    fit_policy: AdmissionFitPolicy,
    pressure_action: AdmissionPressureAction,
}

impl StepResourceAdmissionRequest {
    pub fn new(
        work_shape: BatchWorkShape,
        fit_policy: AdmissionFitPolicy,
        pressure_action: AdmissionPressureAction,
    ) -> Result<Self, VNextError> {
        Ok(Self {
            work_shape,
            fit_policy,
            pressure_action,
        })
    }

    pub fn work_shape(&self) -> &BatchWorkShape {
        &self.work_shape
    }

    pub(crate) const fn immediate_shape(&self) -> DynamicResourceShape {
        self.work_shape.immediate_shape()
    }

    pub(crate) const fn fit_shape(&self) -> DynamicResourceShape {
        match self.fit_policy {
            AdmissionFitPolicy::ImmediateOnly => self.work_shape.immediate_shape(),
            AdmissionFitPolicy::FullInputMustFit => self.work_shape.fit_shape(),
        }
    }

    pub const fn fit_policy(&self) -> AdmissionFitPolicy {
        self.fit_policy
    }

    pub const fn pressure_action(&self) -> AdmissionPressureAction {
        self.pressure_action
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct BatchParticipantAuthority {
    sequence_authority: SequenceAuthorityId,
    request_authority: RequestAuthorityId,
}

impl BatchParticipantAuthority {
    pub const fn new(
        sequence_authority: SequenceAuthorityId,
        request_authority: RequestAuthorityId,
    ) -> Self {
        Self {
            sequence_authority,
            request_authority,
        }
    }

    pub const fn sequence_authority(self) -> SequenceAuthorityId {
        self.sequence_authority
    }

    pub const fn request_authority(self) -> RequestAuthorityId {
        self.request_authority
    }

    const fn canonical_key(self) -> (u32, u64, u32, u64) {
        (
            self.sequence_authority.sparse_id(),
            self.sequence_authority.generation(),
            self.request_authority.sparse_id(),
            self.request_authority.generation(),
        )
    }
}

/// One participant-local node topology key in the physical batch ledger.
/// Attempt ids are deliberately absent so a fresh id cannot bypass overlap
/// detection for the same sequence/frame/node work.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize)]
pub struct ParticipantNodeKey {
    sequence_authority: SequenceAuthorityId,
    request_authority: RequestAuthorityId,
    frame_id: ExecutionFrameId,
    node_id: NodeId,
}

impl ParticipantNodeKey {
    fn new(
        participant: BatchParticipantAuthority,
        frame_id: ExecutionFrameId,
        node_id: NodeId,
    ) -> Self {
        Self {
            sequence_authority: participant.sequence_authority(),
            request_authority: participant.request_authority(),
            frame_id,
            node_id,
        }
    }

    pub const fn sequence_authority(&self) -> SequenceAuthorityId {
        self.sequence_authority
    }

    pub const fn request_authority(&self) -> RequestAuthorityId {
        self.request_authority
    }

    pub const fn frame_id(&self) -> ExecutionFrameId {
        self.frame_id
    }

    pub fn node_id(&self) -> &NodeId {
        &self.node_id
    }
}

/// Opaque association between one exact admitted participant and token work
/// derived from that participant's actual token ids.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct BatchParticipantTokenSpan {
    participant: BatchParticipantAuthority,
    token_span: TokenSpanWork,
}

impl BatchParticipantTokenSpan {
    fn new(participant: BatchParticipantAuthority, token_span: TokenSpanWork) -> Self {
        Self {
            participant,
            token_span,
        }
    }

    pub const fn participant(&self) -> BatchParticipantAuthority {
        self.participant
    }

    pub fn token_span(&self) -> &TokenSpanWork {
        &self.token_span
    }
}

/// Immutable work authority for one exact non-empty participant set. The
/// dimensions remain private so downstream claims and dispatch can only use
/// the shape that core bound to this participant topology and fingerprint.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct BatchWorkShape {
    participants: Vec<BatchParticipantAuthority>,
    participant_work: Vec<BatchParticipantTokenSpan>,
    resource_work: ResourceWorkShape,
    fingerprint: String,
}

impl BatchWorkShape {
    fn new(participant_work: Vec<BatchParticipantTokenSpan>) -> Result<Self, VNextError> {
        if participant_work.is_empty()
            || participant_work.windows(2).any(|pair| {
                pair[0].participant().canonical_key() >= pair[1].participant().canonical_key()
            })
        {
            return Err(invalid_resource(
                "batch work shape requires canonical non-empty unique participant work",
            ));
        }
        let participants = participant_work
            .iter()
            .map(BatchParticipantTokenSpan::participant)
            .collect::<Vec<_>>();
        let resource_work = ResourceWorkShape::from_token_spans(
            participant_work
                .iter()
                .map(|work| work.token_span().clone())
                .collect(),
        )?;
        if resource_work.immediate_sequences()
            != u32::try_from(participants.len())
                .map_err(|_| invalid_resource("batch work participant count exceeds u32"))?
        {
            return Err(invalid_resource(
                "batch work shape sequence count differs from participant evidence",
            ));
        }
        #[derive(Serialize)]
        struct FingerprintInput<'a> {
            domain: &'static str,
            participant_work: &'a [BatchParticipantTokenSpan],
            resource_work_fingerprint: &'a str,
        }
        let input = FingerprintInput {
            domain: "ferrum.runtime-vnext.batch-work-shape.v2",
            participant_work: &participant_work,
            resource_work_fingerprint: resource_work.fingerprint(),
        };
        let bytes = serde_json::to_vec(&input).map_err(|error| {
            invalid_resource(format!("batch work shape encode failed: {error}"))
        })?;
        Ok(Self {
            participants,
            participant_work,
            resource_work,
            fingerprint: format!("{:x}", Sha256::digest(bytes)),
        })
    }

    pub fn participants(&self) -> &[BatchParticipantAuthority] {
        &self.participants
    }

    pub fn participant_work(&self) -> &[BatchParticipantTokenSpan] {
        &self.participant_work
    }

    pub fn resource_work(&self) -> &ResourceWorkShape {
        &self.resource_work
    }

    pub const fn immediate_sequences(&self) -> u32 {
        self.resource_work.immediate_sequences()
    }

    pub const fn immediate_tokens(&self) -> u64 {
        self.resource_work.immediate_tokens()
    }

    pub const fn immediate_pages(&self) -> u64 {
        self.resource_work.immediate_pages()
    }

    pub const fn fit_sequences(&self) -> u32 {
        self.resource_work.fit_sequences()
    }

    pub const fn fit_tokens(&self) -> u64 {
        self.resource_work.fit_tokens()
    }

    pub const fn fit_pages(&self) -> u64 {
        self.resource_work.fit_pages()
    }

    pub fn fingerprint(&self) -> &str {
        &self.fingerprint
    }

    pub(crate) const fn immediate_shape(&self) -> DynamicResourceShape {
        self.resource_work.immediate_shape()
    }

    pub(crate) const fn fit_shape(&self) -> DynamicResourceShape {
        self.resource_work.fit_shape()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct StepParticipantFrameAssignment {
    participant: BatchParticipantAuthority,
    frame_id: ExecutionFrameId,
}

impl StepParticipantFrameAssignment {
    const fn new(
        sequence_authority: SequenceAuthorityId,
        request_authority: RequestAuthorityId,
        frame_id: ExecutionFrameId,
    ) -> Self {
        Self {
            participant: BatchParticipantAuthority::new(sequence_authority, request_authority),
            frame_id,
        }
    }

    pub const fn participant(self) -> BatchParticipantAuthority {
        self.participant
    }

    pub const fn sequence_authority(self) -> SequenceAuthorityId {
        self.participant.sequence_authority()
    }

    pub const fn request_authority(self) -> RequestAuthorityId {
        self.participant.request_authority()
    }

    pub const fn frame_id(self) -> ExecutionFrameId {
        self.frame_id
    }

    const fn canonical_key(self) -> (u32, u64, u32, u64) {
        self.participant.canonical_key()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InvocationResourceAdmissionRequest {
    node_id: NodeId,
    work_shape: BatchWorkShape,
    fit_policy: AdmissionFitPolicy,
    pressure_action: AdmissionPressureAction,
}

impl InvocationResourceAdmissionRequest {
    pub fn new(
        node_id: NodeId,
        work_shape: BatchWorkShape,
        fit_policy: AdmissionFitPolicy,
        pressure_action: AdmissionPressureAction,
    ) -> Result<Self, VNextError> {
        Ok(Self {
            node_id,
            work_shape,
            fit_policy,
            pressure_action,
        })
    }

    pub fn for_all_step_participants(
        node_id: NodeId,
        work_shape: BatchWorkShape,
        fit_policy: AdmissionFitPolicy,
        pressure_action: AdmissionPressureAction,
    ) -> Result<Self, VNextError> {
        Self::new(node_id, work_shape, fit_policy, pressure_action)
    }

    pub fn node_id(&self) -> &NodeId {
        &self.node_id
    }

    pub fn work_shape(&self) -> &BatchWorkShape {
        &self.work_shape
    }

    pub(crate) const fn immediate_shape(&self) -> DynamicResourceShape {
        self.work_shape.immediate_shape()
    }

    pub(crate) const fn fit_shape(&self) -> DynamicResourceShape {
        match self.fit_policy {
            AdmissionFitPolicy::ImmediateOnly => self.work_shape.immediate_shape(),
            AdmissionFitPolicy::FullInputMustFit => self.work_shape.fit_shape(),
        }
    }

    pub const fn fit_policy(&self) -> AdmissionFitPolicy {
        self.fit_policy
    }

    pub const fn pressure_action(&self) -> AdmissionPressureAction {
        self.pressure_action
    }
}

const PLAN_RUNTIME_OPEN: u8 = 0;
const PLAN_RUNTIME_CLOSING: u8 = 1;

trait ErasedPlanStaticDriver<R>: Send
where
    R: DeviceRuntime,
{
    fn release_resource(
        &mut self,
        context: &ResourceTransactionContext<'_, R>,
        reservation: &ResourceReservation,
        buffer: &R::Buffer,
    ) -> Result<(), ResourceDriverFailure>;

    fn quarantine_transaction(
        &mut self,
        context: &ResourceTransactionContext<'_, R>,
        ownership: ResourcePoolOwnership<R>,
    ) -> Result<(), ResourceOwnershipTransferFailure<R>>;

    fn abandon_transaction(&mut self, ownership: ResourcePoolOwnership<R>);
}

impl<D> ErasedPlanStaticDriver<D::Runtime> for D
where
    D: ResourceTransactionDriver,
{
    fn release_resource(
        &mut self,
        context: &ResourceTransactionContext<'_, D::Runtime>,
        reservation: &ResourceReservation,
        buffer: &D::Buffer,
    ) -> Result<(), ResourceDriverFailure> {
        ResourceTransactionDriver::release_resource(self, context, reservation, buffer)
    }

    fn quarantine_transaction(
        &mut self,
        context: &ResourceTransactionContext<'_, D::Runtime>,
        ownership: ResourcePoolOwnership<D::Runtime>,
    ) -> Result<(), ResourceOwnershipTransferFailure<D::Runtime>> {
        ResourceTransactionDriver::quarantine_transaction(self, context, ownership)
    }

    fn abandon_transaction(&mut self, ownership: ResourcePoolOwnership<D::Runtime>) {
        ResourceTransactionDriver::abandon_transaction(self, ownership);
    }
}

struct PlanStaticResources<R>
where
    R: DeviceRuntime,
{
    driver: Mutex<Option<Box<dyn ErasedPlanStaticDriver<R>>>>,
    identity: ResourceTransactionIdentity,
    admission: StaticProvisioningBinding,
    reservations: ResourceReservationBatch,
    states: Vec<ResourceTransactionState>,
    capacity_claim: Option<DeviceCapacityClaim>,
    lease: Option<StaticProvisioningLease<R>>,
    finalized: bool,
}

impl<R> PlanStaticResources<R>
where
    R: DeviceRuntime,
{
    fn ledger_snapshot_entries(&self) -> Vec<ResourceLedgerEntrySnapshot> {
        self.lease
            .as_ref()
            .expect("open plan runtime owns its static lease")
            .slots
            .iter()
            .zip(&self.states)
            .map(|(slot, &transaction_state)| ResourceLedgerEntrySnapshot {
                entry: slot.entry.clone(),
                transaction_state,
                buffer_present: slot.buffer.is_some(),
                actual_resource_id: slot.actual_resource_id.clone(),
                actual_generation: slot.actual_generation,
                actual_descriptor: slot.descriptor.clone(),
            })
            .collect()
    }

    fn release_all(&mut self) -> Result<usize, ResourceDriverFailure> {
        let mut released = 0;
        for order in 0..self.states.len() {
            if self.states[order] == ResourceTransactionState::Released {
                continue;
            }
            if self.states[order] != ResourceTransactionState::Committed {
                return Err(ResourceDriverFailure::new(core_resource_failure(
                    "plan_runtime_close_ledger_diverged",
                    "plan runtime close found a non-committed static resource",
                    false,
                ))
                .expect("core failure has resource domain"));
            }
            let lease = self
                .lease
                .as_ref()
                .expect("open plan runtime owns its static lease");
            let reservation = self.reservations.reservations[order].clone();
            let buffer = lease
                .buffer(order)
                .expect("committed plan runtime owns its static buffer");
            let context = ResourceTransactionContext {
                runtime: &lease.runtime,
                identity: &self.identity,
                binding: &self.admission,
                reservations: &self.reservations,
                cursor: Some(ResourceActionCursor {
                    order,
                    action: ResourceTransactionAction::Release,
                    before: self.states[order],
                    allocation_authorized: false,
                }),
                allocation_authority: None,
                pending_allocation: None,
            };
            let driver = match self.driver.get_mut() {
                Ok(driver) => driver,
                Err(poisoned) => poisoned.into_inner(),
            };
            driver
                .as_mut()
                .expect("open plan runtime owns its static driver")
                .release_resource(&context, &reservation, buffer)?;
            self.lease
                .as_mut()
                .expect("open plan runtime owns its static lease")
                .clear(order);
            self.states[order] = ResourceTransactionState::Released;
            self.capacity_claim
                .as_mut()
                .expect("open plan runtime owns its static capacity claim")
                .release_bytes(reservation.size_bytes());
            released += 1;
        }
        if let Some(mut claim) = self.capacity_claim.take() {
            claim.release();
        }
        self.finalized = true;
        Ok(released)
    }

    fn quarantine_remaining(&mut self) -> Result<usize, ResourceDriverFailure> {
        let quarantined = self
            .states
            .iter()
            .filter(|state| **state == ResourceTransactionState::Committed)
            .count();
        if self.states.iter().any(|state| {
            !matches!(
                state,
                ResourceTransactionState::Committed | ResourceTransactionState::Released
            )
        }) {
            return Err(ResourceDriverFailure::new(core_resource_failure(
                "plan_runtime_quarantine_ledger_diverged",
                "plan runtime quarantine found an invalid static resource state",
                false,
            ))
            .expect("core failure has resource domain"));
        }
        if quarantined == 0 {
            self.finalized = true;
            return Ok(0);
        }
        let lease = self
            .lease
            .as_mut()
            .expect("failed plan runtime close owns its static lease");
        let buffers = lease.take_owned_buffers(&self.reservations);
        let ownership = ResourcePoolOwnership {
            runtime: Arc::clone(&lease.runtime),
            pool_identity: self.admission.pool_identity.clone(),
            reason: ResourceOwnershipReason::Quarantine,
            signal: None,
            buffers,
            capacity_claim: self.capacity_claim.take(),
        };
        let result = {
            let context = ResourceTransactionContext {
                runtime: &lease.runtime,
                identity: &self.identity,
                binding: &self.admission,
                reservations: &self.reservations,
                cursor: None,
                allocation_authority: None,
                pending_allocation: None,
            };
            let driver = match self.driver.get_mut() {
                Ok(driver) => driver,
                Err(poisoned) => poisoned.into_inner(),
            };
            driver
                .as_mut()
                .expect("failed plan runtime close owns its static driver")
                .quarantine_transaction(&context, ownership)
        };
        if let Err(failure) = result {
            let (failure, mut ownership) = failure.into_parts();
            let expected_claimed_bytes = self
                .states
                .iter()
                .zip(self.reservations.reservations())
                .filter(|(state, _)| state.is_live())
                .map(|(_, reservation)| reservation.size_bytes())
                .sum::<u64>();
            if ownership.pool_identity != self.admission.pool_identity
                || ownership.claimed_bytes() != expected_claimed_bytes
            {
                std::mem::forget(ownership);
                return Err(ResourceDriverFailure::new(core_resource_failure(
                    "ownership_transfer_identity_mismatch",
                    "plan runtime quarantine failure returned foreign ownership",
                    false,
                ))
                .expect("core failure has resource domain"));
            }
            self.lease
                .as_mut()
                .expect("failed plan runtime close owns its static lease")
                .restore_owned_buffers(std::mem::take(&mut ownership.buffers));
            self.capacity_claim = ownership.capacity_claim.take();
            return Err(failure);
        }
        for state in &mut self.states {
            if *state == ResourceTransactionState::Committed {
                *state = ResourceTransactionState::Quarantined;
            }
        }
        self.finalized = true;
        Ok(quarantined)
    }
}

impl<R> Drop for PlanStaticResources<R>
where
    R: DeviceRuntime,
{
    fn drop(&mut self) {
        if self.finalized {
            return;
        }
        let signal = ResourceAbandonSignal {
            identity: self.identity.clone(),
            admission: self.admission.clone(),
            state: if self
                .states
                .iter()
                .all(|state| *state == ResourceTransactionState::Committed)
            {
                ResourceTransactionState::Committed
            } else {
                ResourceTransactionState::Released
            },
            pending_action: None,
            ledger: self.ledger_snapshot_entries(),
            active_sequence_slots: Vec::new(),
            poisoned_sequence_slots: Vec::new(),
            undrained_sequence_slots: Vec::new(),
            failure: None,
        };
        let mut lease = self
            .lease
            .take()
            .expect("open plan runtime owns its static lease");
        let buffers = lease.take_owned_buffers(&self.reservations);
        let StaticProvisioningLease {
            slots: _,
            identity: _,
            admission: _,
            runtime,
        } = lease;
        let ownership = ResourcePoolOwnership {
            runtime,
            pool_identity: self.admission.pool_identity.clone(),
            reason: ResourceOwnershipReason::Abandon,
            signal: Some(signal),
            buffers,
            capacity_claim: self.capacity_claim.take(),
        };
        let driver = match self.driver.get_mut() {
            Ok(driver) => driver,
            Err(poisoned) => poisoned.into_inner(),
        };
        if let Some(driver) = driver.as_mut() {
            let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                driver.abandon_transaction(ownership);
            }));
        } else {
            std::mem::forget(ownership);
        }
        self.finalized = true;
    }
}

enum PlanRuntimeStatic<R>
where
    R: DeviceRuntime,
{
    NoStatic { binding: StaticProvisioningBinding },
    Static(PlanStaticResources<R>),
}

/// Unique plan-lifetime owner of the runtime, dynamic pools, maintenance
/// authority, static buffers, capacity claims, and cleanup authority.
#[must_use = "plan runtime resources must be explicitly closed or safely abandoned"]
pub struct PlanRuntimeResources<R>
where
    R: DeviceRuntime,
{
    lifecycle: RwLock<()>,
    phase: AtomicU8,
    lifecycle_tx: watch::Sender<u8>,
    maintenance_controller: DynamicPoolMaintenanceController<R>,
    dynamic_pools: Arc<DynamicPoolSet<R>>,
    static_resources: PlanRuntimeStatic<R>,
    runtime: Arc<R>,
    deferred_cleanup_domain: DeferredDeviceCleanupDomainId,
}

/// Sealed owning proof that one exact plan, runtime instance, provisioning
/// outcome, and admission coordinator belong together. Every durable child
/// authority holds the same root `Arc`.
#[must_use = "a trusted plan/runtime binding must be consumed by logical admission"]
pub struct TrustedPlanRuntimeBinding<R>
where
    R: DeviceRuntime,
{
    resources: Arc<PlanRuntimeResources<R>>,
}

/// A capacity wait registration that keeps its exact plan runtime alive until
/// the waiter either observes a retry epoch or is cancelled by being dropped.
#[must_use = "capacity wait registrations must be awaited, rechecked, or dropped"]
pub struct PlanCapacityWaitRegistration<R>
where
    R: DeviceRuntime,
{
    registration: CapacityWaitRegistration,
    lifecycle_rx: watch::Receiver<u8>,
    resources: Arc<PlanRuntimeResources<R>>,
}

impl<R> PlanCapacityWaitRegistration<R>
where
    R: DeviceRuntime,
{
    pub fn recheck(&self) -> Result<CapacityWaitRecheck, VNextError> {
        let _lifecycle = self.resources.read_lifecycle("recheck a capacity waiter")?;
        self.registration.recheck()
    }

    pub async fn wait_for_change(self) -> Result<CapacityEpochs, VNextError> {
        let Self {
            registration,
            mut lifecycle_rx,
            resources,
        } = self;
        let admission_wait = registration.wait_for_change();
        tokio::pin!(admission_wait);
        tokio::select! {
            result = &mut admission_wait => result,
            changed = lifecycle_rx.changed() => {
                changed.map_err(|_| invalid_resource(
                    "plan runtime lifecycle signal closed while a capacity waiter was live",
                ))?;
                if *lifecycle_rx.borrow_and_update() == PLAN_RUNTIME_CLOSING {
                    Err(invalid_resource(
                        "closing plan runtime cancelled its capacity waiter",
                    ))
                } else {
                    drop(resources);
                    Err(invalid_resource(
                        "capacity waiter observed an invalid plan runtime lifecycle transition",
                    ))
                }
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct PlanRuntimeCloseReceipt {
    evidence: TrustedPlanRuntimeEvidence,
    released_static_resources: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct PlanRuntimeQuarantineReceipt {
    evidence: TrustedPlanRuntimeEvidence,
    released_static_resources: usize,
    quarantined_static_resources: usize,
}

impl PlanRuntimeQuarantineReceipt {
    pub fn evidence(&self) -> &TrustedPlanRuntimeEvidence {
        &self.evidence
    }

    pub const fn released_static_resources(&self) -> usize {
        self.released_static_resources
    }

    pub const fn quarantined_static_resources(&self) -> usize {
        self.quarantined_static_resources
    }
}

impl PlanRuntimeCloseReceipt {
    pub fn evidence(&self) -> &TrustedPlanRuntimeEvidence {
        &self.evidence
    }

    pub const fn released_static_resources(&self) -> usize {
        self.released_static_resources
    }
}

pub enum PlanRuntimeCloseOutcome<R>
where
    R: DeviceRuntime,
{
    Closed(PlanRuntimeCloseReceipt),
    Referenced {
        resources: Arc<PlanRuntimeResources<R>>,
        strong_count: usize,
        deferred_cleanup: DeferredDeviceCleanupStatus,
    },
}

#[must_use = "failed plan runtime close retains static cleanup ownership"]
pub struct PlanRuntimeCloseFailure<R>
where
    R: DeviceRuntime,
{
    failure: FailureEnvelope,
    evidence: TrustedPlanRuntimeEvidence,
    static_resources: Option<PlanStaticResources<R>>,
}

impl<R> PlanRuntimeCloseFailure<R>
where
    R: DeviceRuntime,
{
    pub fn failure(&self) -> &FailureEnvelope {
        &self.failure
    }

    pub fn retry(mut self) -> Result<PlanRuntimeCloseReceipt, Self> {
        let static_resources = self
            .static_resources
            .as_mut()
            .expect("plan runtime close failure owns static cleanup authority");
        let total_static_resources = static_resources.states.len();
        match static_resources.release_all() {
            Ok(_) => {
                self.static_resources.take();
                Ok(PlanRuntimeCloseReceipt {
                    evidence: self.evidence,
                    released_static_resources: total_static_resources,
                })
            }
            Err(failure) => {
                self.failure = failure.into_failure();
                Err(self)
            }
        }
    }

    pub fn quarantine(mut self) -> Result<PlanRuntimeQuarantineReceipt, Self> {
        let static_resources = self
            .static_resources
            .as_mut()
            .expect("plan runtime close failure owns static cleanup authority");
        let released_static_resources = static_resources
            .states
            .iter()
            .filter(|state| **state == ResourceTransactionState::Released)
            .count();
        match static_resources.quarantine_remaining() {
            Ok(quarantined_static_resources) => {
                self.static_resources.take();
                Ok(PlanRuntimeQuarantineReceipt {
                    evidence: self.evidence,
                    released_static_resources,
                    quarantined_static_resources,
                })
            }
            Err(failure) => {
                self.failure = failure.into_failure();
                Err(self)
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct TrustedPlanRuntimeEvidence {
    plan_id: PlanId,
    plan_hash: PlanHash,
    device_id: DeviceId,
    runtime_implementation_fingerprint: String,
    coordinator_id: LogicalAdmissionCoordinatorId,
    static_provisioning_binding: Option<StaticProvisioningBinding>,
    static_pool_identity: Option<ResourcePoolIdentity>,
    static_provisioning_identity: Option<ResourceTransactionIdentity>,
}

impl TrustedPlanRuntimeEvidence {
    pub fn plan_id(&self) -> &PlanId {
        &self.plan_id
    }

    pub fn plan_hash(&self) -> &PlanHash {
        &self.plan_hash
    }

    pub fn device_id(&self) -> &DeviceId {
        &self.device_id
    }

    pub fn runtime_implementation_fingerprint(&self) -> &str {
        &self.runtime_implementation_fingerprint
    }

    pub const fn coordinator_id(&self) -> LogicalAdmissionCoordinatorId {
        self.coordinator_id
    }

    pub fn static_pool_identity(&self) -> Option<&ResourcePoolIdentity> {
        self.static_pool_identity.as_ref()
    }

    pub fn static_provisioning_binding(&self) -> Option<&StaticProvisioningBinding> {
        self.static_provisioning_binding.as_ref()
    }

    pub fn static_provisioning_identity(&self) -> Option<&ResourceTransactionIdentity> {
        self.static_provisioning_identity.as_ref()
    }
}

impl<R> NoStatic<R>
where
    R: DeviceRuntime,
{
    pub fn into_plan_runtime(self) -> Arc<PlanRuntimeResources<R>> {
        let Self {
            maintenance_controller,
            dynamic_pools,
            binding,
            runtime,
        } = self;
        let (lifecycle_tx, _) = watch::channel(PLAN_RUNTIME_OPEN);
        Arc::new(PlanRuntimeResources {
            lifecycle: RwLock::new(()),
            phase: AtomicU8::new(PLAN_RUNTIME_OPEN),
            lifecycle_tx,
            maintenance_controller,
            dynamic_pools,
            static_resources: PlanRuntimeStatic::NoStatic { binding },
            runtime,
            deferred_cleanup_domain: new_deferred_device_cleanup_domain(),
        })
    }
}

impl<R> PlanRuntimeResources<R>
where
    R: DeviceRuntime,
{
    fn evidence(&self) -> TrustedPlanRuntimeEvidence {
        let (binding, identity) = match &self.static_resources {
            PlanRuntimeStatic::NoStatic { binding } => (binding, None),
            PlanRuntimeStatic::Static(source) => (&source.admission, Some(&source.identity)),
        };
        let has_static = matches!(&self.static_resources, PlanRuntimeStatic::Static(_));
        TrustedPlanRuntimeEvidence {
            plan_id: binding.plan_id().clone(),
            plan_hash: binding.plan_hash().clone(),
            device_id: binding.device_id().clone(),
            runtime_implementation_fingerprint: binding
                .device_runtime_implementation_fingerprint()
                .to_owned(),
            coordinator_id: self.dynamic_pools.logical_admission.id(),
            static_provisioning_binding: has_static.then(|| binding.clone()),
            static_pool_identity: has_static.then(|| binding.pool_identity().clone()),
            static_provisioning_identity: identity.cloned(),
        }
    }

    fn read_lifecycle(&self, action: &'static str) -> Result<RwLockReadGuard<'_, ()>, VNextError> {
        let lifecycle = self
            .lifecycle
            .read()
            .map_err(|_| invalid_resource("plan runtime lifecycle gate is poisoned"))?;
        if self.phase.load(Ordering::Acquire) != PLAN_RUNTIME_OPEN {
            return Err(invalid_resource(format!(
                "closing plan runtime cannot {action}"
            )));
        }
        let cleanup = self.deferred_cleanup_status();
        if cleanup.is_saturated() {
            return Err(invalid_resource(format!(
                "plan runtime cannot {action} while {} deferred device cleanup owners await recovery",
                cleanup.pending()
            )));
        }
        Ok(lifecycle)
    }

    pub fn deferred_cleanup_status(&self) -> DeferredDeviceCleanupStatus {
        deferred_device_cleanup_status(self.deferred_cleanup_domain)
    }

    /// Attempts each selected cleanup owner at most once. This may block in a
    /// backend recovery call and must therefore run on a scheduler recovery
    /// thread, never on a request, admission, or destructor path.
    pub fn maintain_deferred_cleanups(
        &self,
        maximum_tasks: usize,
    ) -> Result<DeferredDeviceCleanupMaintenanceReceipt, VNextError> {
        if maximum_tasks == 0
            || maximum_tasks > super::MAX_DEFERRED_DEVICE_CLEANUP_MAINTENANCE_TASKS
        {
            return Err(invalid_resource(format!(
                "deferred device cleanup maintenance size must be in 1..={}",
                super::MAX_DEFERRED_DEVICE_CLEANUP_MAINTENANCE_TASKS
            )));
        }
        Ok(maintain_deferred_device_cleanups(
            self.deferred_cleanup_domain,
            maximum_tasks,
        ))
    }

    pub fn trusted_runtime_binding(
        self: &Arc<Self>,
    ) -> Result<TrustedPlanRuntimeBinding<R>, VNextError> {
        let _lifecycle = self.read_lifecycle("mint a trusted runtime binding")?;
        Ok(TrustedPlanRuntimeBinding {
            resources: Arc::clone(self),
        })
    }

    /// Resolves physical backing pressure for this plan while retaining the
    /// owning root for the full maintenance operation. A close racing before
    /// the second phase check is rejected; a close racing after it observes
    /// this operation's `Arc` and cannot release the root underneath it.
    pub fn maintain_for_deferred(
        self: &Arc<Self>,
        deferred: &DynamicBackingDeferred,
    ) -> Result<DynamicDeferredMaintenanceOutcome, VNextError> {
        let _lifecycle = self.read_lifecycle("maintain deferred dynamic backing")?;
        self.maintenance_controller.maintain_for_deferred(deferred)
    }

    pub fn is_closing(&self) -> bool {
        self.phase.load(Ordering::Acquire) == PLAN_RUNTIME_CLOSING
    }

    pub fn close(
        resources: Arc<Self>,
    ) -> Result<PlanRuntimeCloseOutcome<R>, PlanRuntimeCloseFailure<R>> {
        {
            let _lifecycle = resources
                .lifecycle
                .write()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            match resources.phase.compare_exchange(
                PLAN_RUNTIME_OPEN,
                PLAN_RUNTIME_CLOSING,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => {
                    resources.lifecycle_tx.send_replace(PLAN_RUNTIME_CLOSING);
                }
                Err(PLAN_RUNTIME_CLOSING) => {}
                Err(_) => unreachable!("plan runtime phase is privately bounded"),
            }
        }
        let resources = match Arc::try_unwrap(resources) {
            Ok(resources) => resources,
            Err(resources) => {
                let strong_count = Arc::strong_count(&resources);
                let deferred_cleanup = resources.deferred_cleanup_status();
                return Ok(PlanRuntimeCloseOutcome::Referenced {
                    resources,
                    strong_count,
                    deferred_cleanup,
                });
            }
        };
        if resources.deferred_cleanup_status().pending() != 0 {
            let resources = Arc::new(resources);
            let deferred_cleanup = resources.deferred_cleanup_status();
            return Ok(PlanRuntimeCloseOutcome::Referenced {
                resources,
                strong_count: 1,
                deferred_cleanup,
            });
        }
        let evidence = resources.evidence();
        let Self {
            lifecycle: _,
            phase: _,
            lifecycle_tx,
            maintenance_controller,
            dynamic_pools,
            static_resources,
            runtime,
            deferred_cleanup_domain,
        } = resources;
        debug_assert!(retire_deferred_device_cleanup_domain(
            deferred_cleanup_domain
        ));
        drop(lifecycle_tx);
        drop(maintenance_controller);
        drop(dynamic_pools);
        match static_resources {
            PlanRuntimeStatic::NoStatic { .. } => {
                drop(runtime);
                Ok(PlanRuntimeCloseOutcome::Closed(PlanRuntimeCloseReceipt {
                    evidence,
                    released_static_resources: 0,
                }))
            }
            PlanRuntimeStatic::Static(mut static_resources) => {
                drop(runtime);
                let total_static_resources = static_resources.states.len();
                match static_resources.release_all() {
                    Ok(_) => {
                        drop(static_resources);
                        Ok(PlanRuntimeCloseOutcome::Closed(PlanRuntimeCloseReceipt {
                            evidence,
                            released_static_resources: total_static_resources,
                        }))
                    }
                    Err(failure) => Err(PlanRuntimeCloseFailure {
                        failure: failure.into_failure(),
                        evidence,
                        static_resources: Some(static_resources),
                    }),
                }
            }
        }
    }
}

impl<R> TrustedPlanRuntimeBinding<R>
where
    R: DeviceRuntime,
{
    fn runtime(&self) -> &Arc<R> {
        &self.resources.runtime
    }

    fn logical_admission(&self) -> &LogicalAdmissionCoordinator {
        &self.dynamic_pools().logical_admission
    }

    fn dynamic_pools(&self) -> &Arc<DynamicPoolSet<R>> {
        &self.resources.dynamic_pools
    }

    fn nodes(&self) -> &[PlanNode] {
        &self.dynamic_pools().nodes
    }

    pub fn plan_id(&self) -> &PlanId {
        match &self.resources.static_resources {
            PlanRuntimeStatic::NoStatic { binding } => binding.plan_id(),
            PlanRuntimeStatic::Static(source) => source.admission.plan_id(),
        }
    }

    pub fn plan_hash(&self) -> &PlanHash {
        match &self.resources.static_resources {
            PlanRuntimeStatic::NoStatic { binding } => binding.plan_hash(),
            PlanRuntimeStatic::Static(source) => source.admission.plan_hash(),
        }
    }

    pub fn device_id(&self) -> &DeviceId {
        match &self.resources.static_resources {
            PlanRuntimeStatic::NoStatic { binding } => binding.device_id(),
            PlanRuntimeStatic::Static(source) => source.admission.device_id(),
        }
    }

    pub fn runtime_implementation_fingerprint(&self) -> &str {
        match &self.resources.static_resources {
            PlanRuntimeStatic::NoStatic { binding } => {
                binding.device_runtime_implementation_fingerprint()
            }
            PlanRuntimeStatic::Static(source) => {
                source.admission.device_runtime_implementation_fingerprint()
            }
        }
    }

    pub fn coordinator_id(&self) -> LogicalAdmissionCoordinatorId {
        self.logical_admission().id()
    }

    pub fn static_provisioning(&self) -> Option<&StaticProvisioningLease<R>> {
        match &self.resources.static_resources {
            PlanRuntimeStatic::NoStatic { .. } => None,
            PlanRuntimeStatic::Static(source) => source.lease.as_ref(),
        }
    }

    pub fn evidence(&self) -> TrustedPlanRuntimeEvidence {
        self.resources.evidence()
    }

    fn scoped_demand(
        &self,
        lifetime: AllocationLifetime,
        node_id: Option<&NodeId>,
        immediate_shape: DynamicResourceShape,
        fit_shape: DynamicResourceShape,
        fit_policy: AdmissionFitPolicy,
        pressure_action: AdmissionPressureAction,
    ) -> Result<(AdmissionDemand, Vec<EvaluatedBackingRequest<'_>>), VNextError> {
        if (lifetime == AllocationLifetime::Invocation) != node_id.is_some() {
            return Err(invalid_resource(
                "invocation resource demand requires one exact node identity",
            ));
        }
        let node_resources = node_id
            .map(|node_id| {
                self.nodes()
                    .iter()
                    .find(|node| node.id() == node_id)
                    .map(PlanNode::resources)
                    .ok_or_else(|| {
                        invalid_resource("resource admission references an unknown node")
                    })
            })
            .transpose()?;
        let mut immediate_entries = Vec::new();
        let mut fit_entries = Vec::new();
        let mut requested_slices = Vec::new();
        for domain in &self.dynamic_pools().domains {
            let mut immediate_pool_bytes = 0_u64;
            let mut fit_pool_bytes = 0_u64;
            let mut matched = false;
            for descriptor in &domain.descriptors {
                if descriptor.lifetime() != lifetime
                    || node_resources
                        .is_some_and(|resources| !resources.contains(descriptor.base_resource_id()))
                {
                    continue;
                }
                matched = true;
                let size_bytes = descriptor.evaluate_request_bytes_for_shape(immediate_shape)?;
                let fit_bytes = descriptor.evaluate_request_bytes_for_shape(fit_shape)?;
                immediate_pool_bytes =
                    immediate_pool_bytes
                        .checked_add(size_bytes)
                        .ok_or_else(|| {
                            invalid_resource("dynamic pool immediate demand overflows u64")
                        })?;
                fit_pool_bytes = fit_pool_bytes
                    .checked_add(fit_bytes)
                    .ok_or_else(|| invalid_resource("dynamic pool fit demand overflows u64"))?;
                requested_slices.push(EvaluatedBackingRequest {
                    domain,
                    descriptor,
                    size_bytes,
                });
            }
            if matched {
                immediate_entries.push(CapacityEntry::new(
                    domain.domain_id(),
                    CapacityUnits::new(immediate_pool_bytes),
                )?);
                fit_entries.push(CapacityEntry::new(
                    domain.domain_id(),
                    CapacityUnits::new(fit_pool_bytes),
                )?);
            }
        }
        let immediate = if immediate_entries.is_empty() {
            CapacityVector::empty()
        } else {
            CapacityVector::new(immediate_entries)?
        };
        let fit = if fit_entries.is_empty() {
            CapacityVector::empty()
        } else {
            CapacityVector::new(fit_entries)?
        };
        Ok((
            AdmissionDemand::from_plan(immediate, fit, fit_policy, pressure_action)?,
            requested_slices,
        ))
    }

    fn prepare_backing_slices(
        &self,
        requested_slices: Vec<EvaluatedBackingRequest<'_>>,
    ) -> Result<BackingPrepareDecision<R>, VNextError> {
        self.dynamic_pools().prepare_claim(&requested_slices)
    }

    pub fn register_backing_waiter(
        &self,
        deferred: &DynamicBackingDeferred,
    ) -> Result<PlanCapacityWaitRegistration<R>, VNextError> {
        let _lifecycle = self.resources.read_lifecycle("register a backing waiter")?;
        let lifecycle_rx = self.resources.lifecycle_tx.subscribe();
        let registration = self
            .logical_admission()
            .register_waiter(deferred.epochs())?;
        Ok(PlanCapacityWaitRegistration {
            registration,
            lifecycle_rx,
            resources: Arc::clone(&self.resources),
        })
    }

    /// Request-scoped capacity is claimed exactly once before any child
    /// sequence, stream, provider encode, or device submission exists.
    pub fn try_admit_request(
        &self,
        request: RequestResourceAdmissionRequest,
        run_id: RunId,
        request_id: RequestIdentity,
    ) -> Result<RequestResourceAdmissionDecision<R>, VNextError> {
        let _lifecycle = self.resources.read_lifecycle("admit a request")?;
        let RequestResourceAdmissionRequest {
            work_shape,
            fit_policy,
            pressure_action,
        } = request;
        let immediate_shape = work_shape.immediate_shape();
        let fit_shape = match fit_policy {
            AdmissionFitPolicy::ImmediateOnly => immediate_shape,
            AdmissionFitPolicy::FullInputMustFit => work_shape.fit_shape(),
        };
        let (demand, requested_slices) = self.scoped_demand(
            AllocationLifetime::Request,
            None,
            immediate_shape,
            fit_shape,
            fit_policy,
            pressure_action,
        )?;
        let prepared = match self.prepare_backing_slices(requested_slices)? {
            BackingPrepareDecision::Prepared(prepared) => prepared,
            BackingPrepareDecision::Deferred(deferred) => {
                return Ok(RequestResourceAdmissionDecision::BackingDeferred(deferred));
            }
        };
        match self.logical_admission().try_admit_request(&demand)? {
            RequestAdmissionDecision::Admitted(logical_lease) => {
                if !self.logical_admission().owns_request(&logical_lease) {
                    return Err(invalid_resource(
                        "request admission returned authority from another coordinator",
                    ));
                }
                let slices = prepared.commit();
                Ok(RequestResourceAdmissionDecision::Admitted(Arc::new(
                    AdmittedRequestResources::new(
                        TrustedPlanRuntimeBinding {
                            resources: Arc::clone(&self.resources),
                        },
                        logical_lease,
                        slices,
                        work_shape,
                        run_id,
                        request_id,
                    )?,
                )))
            }
            RequestAdmissionDecision::Deferred(deferred) => {
                Ok(RequestResourceAdmissionDecision::Deferred(deferred))
            }
            RequestAdmissionDecision::PermanentRejected(rejected) => Ok(
                RequestResourceAdmissionDecision::PermanentRejected(rejected),
            ),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct LogicalBackingSliceEvidence {
    domain_id: CapacityDomainId,
    pool_id: DynamicBackingPoolId,
    resource_id: ResourceId,
    pool_instance_id: u64,
    segment_generation: u64,
    segments: Vec<BackingSegment>,
    size_bytes: u64,
    alignment_bytes: u64,
    usage: BufferUsage,
    element_type: ElementType,
    storage_profile: DynamicStorageProfile,
}

impl LogicalBackingSliceEvidence {
    pub const fn domain_id(&self) -> CapacityDomainId {
        self.domain_id
    }

    pub fn resource_id(&self) -> &ResourceId {
        &self.resource_id
    }

    pub fn pool_id(&self) -> &DynamicBackingPoolId {
        &self.pool_id
    }

    pub const fn pool_instance_id(&self) -> u64 {
        self.pool_instance_id
    }

    pub const fn segment_generation(&self) -> u64 {
        self.segment_generation
    }

    pub fn segments(&self) -> &[BackingSegment] {
        &self.segments
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

    pub const fn storage_profile(&self) -> DynamicStorageProfile {
        self.storage_profile
    }
}

#[must_use = "a logical backing authority owns its physical arena extents"]
pub struct LogicalBackingSliceAuthority {
    evidence: LogicalBackingSliceEvidence,
    segment_lease: BackingSegmentLease,
}

impl LogicalBackingSliceAuthority {
    pub fn evidence(&self) -> &LogicalBackingSliceEvidence {
        &self.evidence
    }

    pub const fn domain_id(&self) -> CapacityDomainId {
        self.evidence.domain_id
    }

    pub fn resource_id(&self) -> &ResourceId {
        &self.evidence.resource_id
    }

    pub const fn size_bytes(&self) -> u64 {
        self.evidence.size_bytes
    }
}

pub struct LogicalBackingBufferView<'a, B> {
    bindings: Vec<LogicalBackingSegmentBinding<B>>,
    evidence: &'a LogicalBackingSliceEvidence,
}

pub(crate) struct LogicalBackingSegmentBinding<B> {
    segment: BackingSegment,
    chunk: Arc<ResidentChunkBacking<B>>,
}

impl<B> LogicalBackingSegmentBinding<B> {
    pub(crate) fn segment(&self) -> &BackingSegment {
        &self.segment
    }

    pub(crate) fn chunk(&self) -> &BackingChunkIdentity {
        self.segment.chunk()
    }

    pub(crate) fn buffer(&self) -> &B {
        &self.chunk.buffer
    }

    pub(crate) fn descriptor(&self) -> &BufferDescriptor {
        &self.chunk.descriptor
    }
}

impl<'a, B> LogicalBackingBufferView<'a, B> {
    pub(crate) fn segment_bindings(&self) -> &[LogicalBackingSegmentBinding<B>] {
        &self.bindings
    }

    pub(crate) fn storage_profile(&self) -> DynamicStorageProfile {
        self.evidence.storage_profile()
    }

    pub fn slice(&self) -> &'a LogicalBackingSliceEvidence {
        self.evidence
    }
}

pub enum RequestResourceAdmissionDecision<R>
where
    R: DeviceRuntime,
{
    Admitted(Arc<AdmittedRequestResources<R>>),
    Deferred(AdmissionDeferred),
    BackingDeferred(DynamicBackingDeferred),
    PermanentRejected(AdmissionRejected),
}

/// Request root authority. Request-lifetime state is physically and logically
/// claimed once, then shared by exact child sequence authorities through an
/// owning `Arc` parent hold.
#[must_use = "request resources release capacity after their last child sequence"]
pub struct AdmittedRequestResources<R>
where
    R: DeviceRuntime,
{
    backing_slices: Vec<LogicalBackingSliceAuthority>,
    logical_lease: LogicalRequestLease,
    plan: TrustedPlanRuntimeBinding<R>,
    work_shape: ResourceWorkShape,
    run_id: RunId,
    request_id: RequestIdentity,
}

impl<R> AdmittedRequestResources<R>
where
    R: DeviceRuntime,
{
    fn new(
        plan: TrustedPlanRuntimeBinding<R>,
        logical_lease: LogicalRequestLease,
        backing_slices: Vec<LogicalBackingSliceAuthority>,
        work_shape: ResourceWorkShape,
        run_id: RunId,
        request_id: RequestIdentity,
    ) -> Result<Self, VNextError> {
        if !plan.logical_admission().owns_request(&logical_lease) {
            return Err(invalid_resource(
                "logical request authority belongs to another coordinator",
            ));
        }
        Ok(Self {
            backing_slices,
            logical_lease,
            plan,
            work_shape,
            run_id,
            request_id,
        })
    }

    pub const fn request_authority(&self) -> RequestAuthorityId {
        self.logical_lease.request()
    }

    pub fn coordinator_id(&self) -> LogicalAdmissionCoordinatorId {
        self.logical_lease.coordinator_id()
    }

    pub fn run_id(&self) -> &RunId {
        &self.run_id
    }

    pub fn request_id(&self) -> &RequestIdentity {
        &self.request_id
    }

    pub fn backing_slices(&self) -> &[LogicalBackingSliceAuthority] {
        &self.backing_slices
    }

    pub fn work_shape(&self) -> &ResourceWorkShape {
        &self.work_shape
    }

    pub fn static_provisioning(&self) -> Option<&StaticProvisioningLease<R>> {
        self.plan.static_provisioning()
    }

    pub fn plan_evidence(&self) -> TrustedPlanRuntimeEvidence {
        self.plan.evidence()
    }

    fn backing_view(
        &self,
        resource_id: &ResourceId,
    ) -> Result<LogicalBackingBufferView<'_, R::Buffer>, VNextError> {
        let authority = self
            .backing_slices
            .iter()
            .find(|authority| authority.resource_id() == resource_id)
            .ok_or_else(|| invalid_resource("logical request does not own that backing slice"))?;
        self.plan.dynamic_pools().view(authority)
    }

    /// Sequence-scoped capacity is charged once per exact child sequence.
    pub fn try_admit_sequence(
        self: &Arc<Self>,
        request: SequenceResourceAdmissionRequest,
    ) -> Result<SequenceResourceAdmissionDecision<R>, VNextError> {
        let _lifecycle = self
            .plan
            .resources
            .read_lifecycle("admit a child sequence")?;
        let SequenceResourceAdmissionRequest {
            work_shape,
            fit_policy,
            pressure_action,
        } = request;
        let immediate_shape = work_shape.immediate_shape();
        let fit_shape = match fit_policy {
            AdmissionFitPolicy::ImmediateOnly => immediate_shape,
            AdmissionFitPolicy::FullInputMustFit => work_shape.fit_shape(),
        };
        let (demand, requested_slices) = self.plan.scoped_demand(
            AllocationLifetime::Sequence,
            None,
            immediate_shape,
            fit_shape,
            fit_policy,
            pressure_action,
        )?;
        match self
            .plan
            .logical_admission()
            .preflight_sequence_ceiling_for_request(&self.logical_lease, &demand)?
        {
            AdmissionPreflightDecision::Eligible => {}
            AdmissionPreflightDecision::Deferred(deferred) => {
                return Ok(SequenceResourceAdmissionDecision::Deferred(deferred));
            }
            AdmissionPreflightDecision::PermanentRejected(rejected) => {
                return Ok(SequenceResourceAdmissionDecision::PermanentRejected(
                    rejected,
                ));
            }
        }
        let prepared = match self.plan.prepare_backing_slices(requested_slices)? {
            BackingPrepareDecision::Prepared(prepared) => prepared,
            BackingPrepareDecision::Deferred(deferred) => {
                return Ok(SequenceResourceAdmissionDecision::BackingDeferred(deferred));
            }
        };
        match self
            .plan
            .logical_admission()
            .try_admit_sequence_for_request(&self.logical_lease, &demand)?
        {
            AdmissionDecision::Admitted(logical_lease) => {
                if !self.plan.logical_admission().owns(&logical_lease)
                    || logical_lease.request() != self.request_authority()
                {
                    return Err(invalid_resource(
                        "sequence admission returned authority from another request",
                    ));
                }
                let slices = prepared.commit();
                Ok(SequenceResourceAdmissionDecision::Admitted(Arc::new(
                    AdmittedSequenceResources::new(
                        Arc::clone(self),
                        logical_lease,
                        slices,
                        work_shape,
                    )?,
                )))
            }
            AdmissionDecision::Deferred(deferred) => {
                Ok(SequenceResourceAdmissionDecision::Deferred(deferred))
            }
            AdmissionDecision::PermanentRejected(rejected) => Ok(
                SequenceResourceAdmissionDecision::PermanentRejected(rejected),
            ),
        }
    }
}

pub enum SequenceResourceAdmissionDecision<R>
where
    R: DeviceRuntime,
{
    Admitted(Arc<AdmittedSequenceResources<R>>),
    Deferred(AdmissionDeferred),
    BackingDeferred(DynamicBackingDeferred),
    PermanentRejected(AdmissionRejected),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
pub struct SequenceSessionEpoch(NonZeroU64);

impl SequenceSessionEpoch {
    pub const fn get(self) -> u64 {
        self.0.get()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct SequenceSessionFingerprint(String);

const SEQUENCE_SESSION_FINGERPRINT_DOMAIN: &str = "sequence-session-v1";

#[derive(Serialize)]
struct SequenceSessionFingerprintEnvelope<'a, T>
where
    T: Serialize + ?Sized,
{
    domain: &'static str,
    payload: &'a T,
}

fn sequence_session_fingerprint<T>(payload: &T) -> Result<SequenceSessionFingerprint, VNextError>
where
    T: Serialize + ?Sized,
{
    let envelope = SequenceSessionFingerprintEnvelope {
        domain: SEQUENCE_SESSION_FINGERPRINT_DOMAIN,
        payload,
    };
    let bytes = serde_json::to_vec(&envelope)
        .map_err(|_| invalid_resource("trusted sequence session identity did not serialize"))?;
    Ok(SequenceSessionFingerprint(format!(
        "{:x}",
        Sha256::digest(bytes)
    )))
}

impl SequenceSessionFingerprint {
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum SequenceSessionTerminalDisposition {
    Completed,
    Aborted,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct SequenceSessionTerminalReceipt {
    epoch: SequenceSessionEpoch,
    fingerprint: SequenceSessionFingerprint,
    disposition: SequenceSessionTerminalDisposition,
    retired_frames: u64,
}

impl SequenceSessionTerminalReceipt {
    pub const fn epoch(&self) -> SequenceSessionEpoch {
        self.epoch
    }

    pub fn fingerprint(&self) -> &SequenceSessionFingerprint {
        &self.fingerprint
    }

    pub const fn disposition(&self) -> SequenceSessionTerminalDisposition {
        self.disposition
    }

    pub const fn retired_frames(&self) -> u64 {
        self.retired_frames
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SequenceSessionPhase {
    Open,
    CancelRequested,
    Poisoned,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ActiveSequenceFrame {
    frame_id: ExecutionFrameId,
    batch_step_id: BatchStepId,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ParticipantFlightPhase {
    Prepared,
    InFlight,
}

#[derive(Debug, Clone)]
struct ActiveSequenceSessionState {
    epoch: SequenceSessionEpoch,
    fingerprint: SequenceSessionFingerprint,
    phase: SequenceSessionPhase,
    next_frame: Option<ExecutionFrameId>,
    active_frame: Option<ActiveSequenceFrame>,
    participant_flights: BTreeMap<ParticipantNodeKey, ParticipantFlightPhase>,
    retired_frames: u64,
}

#[derive(Debug, Clone)]
enum SequenceSessionSlotState {
    Dormant {
        next_epoch: Option<SequenceSessionEpoch>,
    },
    Active(ActiveSequenceSessionState),
    Terminal(SequenceSessionTerminalReceipt),
    FailClosed,
}

struct SequenceSessionSlot {
    state: Mutex<SequenceSessionSlotState>,
}

impl SequenceSessionSlot {
    fn new() -> Self {
        Self {
            state: Mutex::new(SequenceSessionSlotState::Dormant {
                next_epoch: Some(SequenceSessionEpoch(
                    NonZeroU64::new(1).expect("one is non-zero"),
                )),
            }),
        }
    }

    fn is_poisoned(&self) -> bool {
        match self.state.lock() {
            Ok(state) => matches!(
                &*state,
                SequenceSessionSlotState::Active(ActiveSequenceSessionState {
                    phase: SequenceSessionPhase::Poisoned,
                    ..
                }) | SequenceSessionSlotState::FailClosed
            ),
            Err(_) => true,
        }
    }
}

/// Process-local proof that one exact sequence session was open when checked.
/// The weak slot reference deliberately does not keep the session or its
/// resources alive, and the private fields prevent sibling modules from
/// forging a witness from copied identity values.
#[derive(Clone)]
pub(crate) struct SequenceSessionLiveWitness {
    slot: Weak<SequenceSessionSlot>,
    epoch: SequenceSessionEpoch,
    fingerprint: SequenceSessionFingerprint,
}

impl SequenceSessionLiveWitness {
    pub(crate) fn ensure_live(&self) -> Result<(), VNextError> {
        let slot = self.slot.upgrade().ok_or_else(|| {
            invalid_resource("sequence session live witness owner is no longer available")
        })?;
        let state = slot
            .state
            .lock()
            .map_err(|_| invalid_resource("sequence session state mutex is poisoned"))?;
        match &*state {
            SequenceSessionSlotState::Active(active)
                if active.epoch == self.epoch && active.fingerprint == self.fingerprint =>
            {
                Ok(())
            }
            _ => Err(invalid_resource(
                "sequence session live witness is stale or no longer active",
            )),
        }
    }

    pub(crate) fn ensure_open(&self) -> Result<(), VNextError> {
        let slot = self.slot.upgrade().ok_or_else(|| {
            invalid_resource("sequence session live witness owner is no longer available")
        })?;
        let state = slot
            .state
            .lock()
            .map_err(|_| invalid_resource("sequence session state mutex is poisoned"))?;
        match &*state {
            SequenceSessionSlotState::Active(active)
                if active.epoch == self.epoch
                    && active.fingerprint == self.fingerprint
                    && active.phase == SequenceSessionPhase::Open =>
            {
                Ok(())
            }
            _ => Err(invalid_resource(
                "sequence session live witness is stale or no longer open",
            )),
        }
    }

    pub(crate) fn ensure_identity(
        &self,
        epoch: SequenceSessionEpoch,
        fingerprint: &SequenceSessionFingerprint,
    ) -> Result<(), VNextError> {
        if self.epoch != epoch || self.fingerprint != *fingerprint {
            return Err(invalid_resource(
                "sequence session live witness differs from the expected identity",
            ));
        }
        self.ensure_open()
    }

    pub(crate) fn ensure_live_identity(
        &self,
        epoch: SequenceSessionEpoch,
        fingerprint: &SequenceSessionFingerprint,
    ) -> Result<(), VNextError> {
        if self.epoch != epoch || self.fingerprint != *fingerprint {
            return Err(invalid_resource(
                "sequence session live witness differs from the expected identity",
            ));
        }
        self.ensure_live()
    }
}

/// Core-owned logical sequence lifecycle. It owns sequence resources but no
/// device stream; scheduler-owned execution lanes may serve many sessions.
#[must_use = "a sequence session must reach an explicit terminal disposition"]
pub struct SequenceSession<R>
where
    R: DeviceRuntime,
{
    resources: Arc<AdmittedSequenceResources<R>>,
    slot: Arc<SequenceSessionSlot>,
    epoch: SequenceSessionEpoch,
    fingerprint: SequenceSessionFingerprint,
}

impl<R> SequenceSession<R>
where
    R: DeviceRuntime,
{
    pub fn resources(&self) -> &Arc<AdmittedSequenceResources<R>> {
        &self.resources
    }

    pub const fn epoch(&self) -> SequenceSessionEpoch {
        self.epoch
    }

    pub fn fingerprint(&self) -> &SequenceSessionFingerprint {
        &self.fingerprint
    }

    pub(crate) fn live_witness(&self) -> Result<SequenceSessionLiveWitness, VNextError> {
        let witness = SequenceSessionLiveWitness {
            slot: Arc::downgrade(&self.slot),
            epoch: self.epoch,
            fingerprint: self.fingerprint.clone(),
        };
        witness.ensure_identity(self.epoch, &self.fingerprint)?;
        Ok(witness)
    }

    pub(crate) fn ensure_open_identity(&self) -> Result<(), VNextError> {
        self.live_witness().map(|_| ())
    }

    pub fn sequence_authority(&self) -> SequenceAuthorityId {
        self.resources.sequence_authority()
    }

    pub fn request_authority(&self) -> RequestAuthorityId {
        self.resources.request_authority()
    }

    pub fn request_cancel(&self) -> Result<SequenceSessionCancelSnapshot, VNextError> {
        let mut state = self
            .slot
            .state
            .lock()
            .map_err(|_| invalid_resource("sequence session state mutex is poisoned"))?;
        let active = match &mut *state {
            SequenceSessionSlotState::Active(active)
                if active.epoch == self.epoch && active.fingerprint == self.fingerprint =>
            {
                active
            }
            SequenceSessionSlotState::Active(_) => {
                return Err(invalid_resource("stale sequence session authority"));
            }
            SequenceSessionSlotState::Terminal(_) => {
                return Err(invalid_resource("sequence session is already terminal"));
            }
            SequenceSessionSlotState::Dormant { .. } => {
                return Err(invalid_resource("sequence session is not active"));
            }
            SequenceSessionSlotState::FailClosed => {
                return Err(invalid_resource("sequence session is fail-closed"));
            }
        };
        match active.phase {
            SequenceSessionPhase::Open => active.phase = SequenceSessionPhase::CancelRequested,
            SequenceSessionPhase::CancelRequested => {}
            SequenceSessionPhase::Poisoned => {
                return Err(invalid_resource(
                    "poisoned sequence session cannot be cancelled",
                ));
            }
        }
        Ok(SequenceSessionCancelSnapshot {
            active_frame: active.active_frame.map(|frame| frame.frame_id),
            participant_flights: u64::try_from(active.participant_flights.len())
                .map_err(|_| invalid_resource("participant flight count exceeds u64"))?,
        })
    }

    pub fn try_complete(&self) -> Result<SequenceSessionTerminalReceipt, VNextError> {
        self.terminalize(SequenceSessionTerminalDisposition::Completed)
    }

    pub fn try_abort(&self) -> Result<SequenceSessionTerminalReceipt, VNextError> {
        self.terminalize(SequenceSessionTerminalDisposition::Aborted)
    }

    fn terminalize(
        &self,
        disposition: SequenceSessionTerminalDisposition,
    ) -> Result<SequenceSessionTerminalReceipt, VNextError> {
        let mut state = self
            .slot
            .state
            .lock()
            .map_err(|_| invalid_resource("sequence session state mutex is poisoned"))?;
        let active = match &*state {
            SequenceSessionSlotState::Active(active)
                if active.epoch == self.epoch && active.fingerprint == self.fingerprint =>
            {
                active
            }
            SequenceSessionSlotState::Active(_) => {
                return Err(invalid_resource("stale sequence session authority"));
            }
            SequenceSessionSlotState::Terminal(_) => {
                return Err(invalid_resource("sequence session is already terminal"));
            }
            SequenceSessionSlotState::Dormant { .. } => {
                return Err(invalid_resource("sequence session is not active"));
            }
            SequenceSessionSlotState::FailClosed => {
                return Err(invalid_resource("sequence session is fail-closed"));
            }
        };
        let phase_matches = match disposition {
            SequenceSessionTerminalDisposition::Completed => {
                active.phase == SequenceSessionPhase::Open && active.retired_frames > 0
            }
            SequenceSessionTerminalDisposition::Aborted => matches!(
                active.phase,
                SequenceSessionPhase::CancelRequested | SequenceSessionPhase::Poisoned
            ),
        };
        if !phase_matches || active.active_frame.is_some() || !active.participant_flights.is_empty()
        {
            return Err(invalid_resource(
                "sequence terminalization requires the matching phase, no active frame, and no participant flight",
            ));
        }
        let receipt = SequenceSessionTerminalReceipt {
            epoch: active.epoch,
            fingerprint: active.fingerprint.clone(),
            disposition,
            retired_frames: active.retired_frames,
        };
        *state = SequenceSessionSlotState::Terminal(receipt.clone());
        Ok(receipt)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SequenceSessionCancelSnapshot {
    active_frame: Option<ExecutionFrameId>,
    participant_flights: u64,
}

impl SequenceSessionCancelSnapshot {
    pub const fn active_frame(self) -> Option<ExecutionFrameId> {
        self.active_frame
    }

    pub const fn participant_flights(self) -> u64 {
        self.participant_flights
    }
}

impl<R> Drop for SequenceSession<R>
where
    R: DeviceRuntime,
{
    fn drop(&mut self) {
        let mut state = match self.slot.state.lock() {
            Ok(state) => state,
            Err(poisoned) => poisoned.into_inner(),
        };
        if matches!(
            &*state,
            SequenceSessionSlotState::Active(active)
                if active.epoch == self.epoch && active.fingerprint == self.fingerprint
        ) {
            *state = SequenceSessionSlotState::FailClosed;
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SequenceExecutionAuthoritySource {
    Unselected,
    LegacyStream,
    SequenceSession,
    FailClosed,
}

/// Sequence authority. There is exactly one state cell for the exact
/// `SequenceAuthorityId` issued by B1; no ceiling-sized slot vector and no
/// caller-selected slot allocator exist here.
#[must_use = "logical sequence resources release capacity when dropped"]
pub struct AdmittedSequenceResources<R>
where
    R: DeviceRuntime,
{
    // Recovery records own undrained raw streams. They must drop before the
    // backing slices and logical lease can make those resources reusable.
    sequence_recovery: ManuallyDrop<Arc<SequenceRecoveryRegistry<R>>>,
    backing_slices: ManuallyDrop<Vec<LogicalBackingSliceAuthority>>,
    logical_lease: ManuallyDrop<LogicalAdmissionLease>,
    request: ManuallyDrop<Arc<AdmittedRequestResources<R>>>,
    work_shape: ResourceWorkShape,
    authority_source: Mutex<SequenceExecutionAuthoritySource>,
    session_slot: Arc<SequenceSessionSlot>,
    state: Arc<AtomicU64>,
    sequence_dispatch_gate: Arc<AtomicU64>,
    next_activation_epoch: AtomicU64,
}

struct DeferredSequenceResourceCleanup<R>
where
    R: DeviceRuntime,
{
    sequence_recovery: ManuallyDrop<Arc<SequenceRecoveryRegistry<R>>>,
    backing_slices: ManuallyDrop<Vec<LogicalBackingSliceAuthority>>,
    logical_lease: ManuallyDrop<LogicalAdmissionLease>,
    request: ManuallyDrop<Arc<AdmittedRequestResources<R>>>,
    completed: bool,
}

impl<R> DeferredDeviceCleanupTask for DeferredSequenceResourceCleanup<R>
where
    R: DeviceRuntime,
{
    fn try_cleanup(&mut self) -> DeferredDeviceCleanupDisposition {
        if self.completed {
            return DeferredDeviceCleanupDisposition::Completed;
        }
        let recovery = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.sequence_recovery
                .recover_all_for_owner_drop(self.request.plan.runtime())
        }));
        if !matches!(recovery, Ok(Ok(()))) {
            return DeferredDeviceCleanupDisposition::Retryable;
        }

        // SAFETY: successful recovery removed and destroyed every raw stream.
        // This registry-owned unit is the sole owner of these fields and releases
        // them once, in dependency order.
        unsafe {
            ManuallyDrop::drop(&mut self.sequence_recovery);
            ManuallyDrop::drop(&mut self.backing_slices);
            ManuallyDrop::drop(&mut self.logical_lease);
            ManuallyDrop::drop(&mut self.request);
        }
        self.completed = true;
        DeferredDeviceCleanupDisposition::Completed
    }
}

impl<R> AdmittedSequenceResources<R>
where
    R: DeviceRuntime,
{
    fn new(
        request: Arc<AdmittedRequestResources<R>>,
        logical_lease: LogicalAdmissionLease,
        backing_slices: Vec<LogicalBackingSliceAuthority>,
        work_shape: ResourceWorkShape,
    ) -> Result<Self, VNextError> {
        if !request.plan.logical_admission().owns(&logical_lease)
            || logical_lease.request() != request.request_authority()
        {
            return Err(invalid_resource(
                "logical sequence authority belongs to another request",
            ));
        }
        let plan_resources = Arc::clone(&request.plan.resources);
        Ok(Self {
            sequence_recovery: ManuallyDrop::new(Arc::new(SequenceRecoveryRegistry::new(
                plan_resources,
            ))),
            backing_slices: ManuallyDrop::new(backing_slices),
            logical_lease: ManuallyDrop::new(logical_lease),
            request: ManuallyDrop::new(request),
            work_shape,
            authority_source: Mutex::new(SequenceExecutionAuthoritySource::Unselected),
            session_slot: Arc::new(SequenceSessionSlot::new()),
            state: Arc::new(AtomicU64::new(0)),
            sequence_dispatch_gate: Arc::new(AtomicU64::new(0)),
            next_activation_epoch: AtomicU64::new(1),
        })
    }

    pub fn sequence_authority(&self) -> SequenceAuthorityId {
        self.logical_lease.sequence()
    }

    pub fn request_authority(&self) -> RequestAuthorityId {
        self.logical_lease.request()
    }

    pub fn coordinator_id(&self) -> LogicalAdmissionCoordinatorId {
        self.logical_lease.coordinator_id()
    }

    pub fn run_id(&self) -> &RunId {
        self.request.run_id()
    }

    pub fn request_id(&self) -> &RequestIdentity {
        self.request.request_id()
    }

    pub fn backing_slices(&self) -> &[LogicalBackingSliceAuthority] {
        &self.backing_slices
    }

    pub fn request_resources(&self) -> &Arc<AdmittedRequestResources<R>> {
        &self.request
    }

    pub fn work_shape(&self) -> &ResourceWorkShape {
        &self.work_shape
    }

    pub(crate) fn backing_view(
        &self,
        resource_id: &ResourceId,
    ) -> Result<LogicalBackingBufferView<'_, R::Buffer>, VNextError> {
        if let Some(authority) = self
            .backing_slices
            .iter()
            .find(|authority| authority.resource_id() == resource_id)
        {
            return self.request.plan.dynamic_pools().view(authority);
        }
        self.request.backing_view(resource_id)
    }

    pub fn static_provisioning(&self) -> Option<&StaticProvisioningLease<R>> {
        self.request.static_provisioning()
    }

    pub fn plan_evidence(&self) -> TrustedPlanRuntimeEvidence {
        self.request.plan_evidence()
    }

    fn lock_authority_source(
        &self,
    ) -> Result<std::sync::MutexGuard<'_, SequenceExecutionAuthoritySource>, VNextError> {
        match self.authority_source.lock() {
            Ok(source) => Ok(source),
            Err(poisoned) => {
                let mut source = poisoned.into_inner();
                *source = SequenceExecutionAuthoritySource::FailClosed;
                Err(invalid_resource(
                    "logical sequence execution authority selector is fail-closed",
                ))
            }
        }
    }

    fn authority_source_is_fail_closed(&self) -> bool {
        match self.authority_source.lock() {
            Ok(source) => *source == SequenceExecutionAuthoritySource::FailClosed,
            Err(poisoned) => {
                *poisoned.into_inner() = SequenceExecutionAuthoritySource::FailClosed;
                true
            }
        }
    }

    pub fn open_session(self: &Arc<Self>) -> Result<Arc<SequenceSession<R>>, VNextError> {
        let _lifecycle = self
            .request
            .plan
            .resources
            .read_lifecycle("open a sequence session")?;
        #[derive(Serialize)]
        struct FingerprintInput<'a> {
            plan: &'a TrustedPlanRuntimeEvidence,
            coordinator_id: LogicalAdmissionCoordinatorId,
            request_authority: RequestAuthorityId,
            sequence_authority: SequenceAuthorityId,
            run_id: &'a RunId,
            request_id: &'a RequestIdentity,
            epoch: SequenceSessionEpoch,
            request_backing: Vec<&'a LogicalBackingSliceEvidence>,
            sequence_backing: Vec<&'a LogicalBackingSliceEvidence>,
        }

        let mut authority_source = self.lock_authority_source()?;
        let selecting_session = match *authority_source {
            SequenceExecutionAuthoritySource::Unselected => true,
            SequenceExecutionAuthoritySource::SequenceSession => false,
            SequenceExecutionAuthoritySource::LegacyStream => {
                return Err(invalid_resource(
                    "logical sequence execution authority is permanently selected for legacy streams",
                ));
            }
            SequenceExecutionAuthoritySource::FailClosed => {
                return Err(invalid_resource(
                    "logical sequence execution authority selector is fail-closed",
                ));
            }
        };
        let mut state = match self.session_slot.state.lock() {
            Ok(state) => state,
            Err(_) => {
                *authority_source = SequenceExecutionAuthoritySource::FailClosed;
                return Err(invalid_resource("sequence session state mutex is poisoned"));
            }
        };
        let epoch = match &*state {
            SequenceSessionSlotState::Dormant {
                next_epoch: Some(epoch),
            } if selecting_session => *epoch,
            SequenceSessionSlotState::Dormant {
                next_epoch: Some(_),
            } => {
                *authority_source = SequenceExecutionAuthoritySource::FailClosed;
                return Err(invalid_resource(
                    "sequence session authority source lost its active or terminal slot",
                ));
            }
            SequenceSessionSlotState::Dormant { next_epoch: None } => {
                *authority_source = SequenceExecutionAuthoritySource::FailClosed;
                return Err(invalid_resource(
                    "sequence session epoch space is exhausted",
                ));
            }
            SequenceSessionSlotState::Active(_) => {
                return Err(invalid_resource(
                    "logical sequence already has an active session",
                ));
            }
            SequenceSessionSlotState::Terminal(_) => {
                return Err(invalid_resource("logical sequence is already terminal"));
            }
            SequenceSessionSlotState::FailClosed => {
                return Err(invalid_resource(
                    "logical sequence session slot is fail-closed",
                ));
            }
        };
        let plan = self.plan_evidence();
        let request_backing = self
            .request
            .backing_slices
            .iter()
            .map(LogicalBackingSliceAuthority::evidence)
            .collect();
        let sequence_backing = self
            .backing_slices
            .iter()
            .map(LogicalBackingSliceAuthority::evidence)
            .collect();
        let input = FingerprintInput {
            plan: &plan,
            coordinator_id: self.coordinator_id(),
            request_authority: self.request_authority(),
            sequence_authority: self.sequence_authority(),
            run_id: self.run_id(),
            request_id: self.request_id(),
            epoch,
            request_backing,
            sequence_backing,
        };
        let fingerprint = match sequence_session_fingerprint(&input) {
            Ok(fingerprint) => fingerprint,
            Err(error) => {
                *authority_source = SequenceExecutionAuthoritySource::FailClosed;
                return Err(error);
            }
        };
        let next_epoch = epoch
            .get()
            .checked_add(1)
            .and_then(NonZeroU64::new)
            .map(SequenceSessionEpoch);
        *state = SequenceSessionSlotState::Active(ActiveSequenceSessionState {
            epoch,
            fingerprint: fingerprint.clone(),
            phase: SequenceSessionPhase::Open,
            next_frame: Some(
                ExecutionFrameId::try_from(1_u64)
                    .expect("the first execution frame id is non-zero"),
            ),
            active_frame: None,
            participant_flights: BTreeMap::new(),
            retired_frames: 0,
        });
        *authority_source = SequenceExecutionAuthoritySource::SequenceSession;
        // A logical sequence is one-shot today. Retaining the checked successor
        // makes epoch exhaustion explicit if recovery later permits reopening.
        let _ = next_epoch;
        drop(state);
        Ok(Arc::new(SequenceSession {
            resources: Arc::clone(self),
            slot: Arc::clone(&self.session_slot),
            epoch,
            fingerprint,
        }))
    }

    pub fn is_poisoned(&self) -> bool {
        self.authority_source_is_fail_closed()
            || self.session_slot.is_poisoned()
            || sequence_dispatch_is_poisoned(&self.sequence_dispatch_gate)
            || sequence_slot_is_poisoned(self.state.load(Ordering::Acquire))
    }
}

impl<R> Drop for AdmittedSequenceResources<R>
where
    R: DeviceRuntime,
{
    fn drop(&mut self) {
        if !self.sequence_recovery.is_empty() {
            self.sequence_dispatch_gate
                .fetch_or(SEQUENCE_DISPATCH_POISONED_BIT, Ordering::AcqRel);
            let cleanup_domain = self.request.plan.resources.deferred_cleanup_domain;
            // SAFETY: this Drop implementation runs once and transfers all four
            // ManuallyDrop fields into one aggregate recovery owner.
            let cleanup = unsafe {
                DeferredSequenceResourceCleanup {
                    sequence_recovery: ManuallyDrop::new(ManuallyDrop::take(
                        &mut self.sequence_recovery,
                    )),
                    backing_slices: ManuallyDrop::new(ManuallyDrop::take(&mut self.backing_slices)),
                    logical_lease: ManuallyDrop::new(ManuallyDrop::take(&mut self.logical_lease)),
                    request: ManuallyDrop::new(ManuallyDrop::take(&mut self.request)),
                    completed: false,
                }
            };
            defer_device_cleanup(cleanup_domain, cleanup);
            return;
        }

        // SAFETY: an empty registry proves there is no raw stream whose backend
        // quiescence could depend on these resources. This is pure ownership
        // teardown and performs no backend call.
        unsafe {
            ManuallyDrop::drop(&mut self.sequence_recovery);
            ManuallyDrop::drop(&mut self.backing_slices);
            ManuallyDrop::drop(&mut self.logical_lease);
            ManuallyDrop::drop(&mut self.request);
        }
    }
}

/// Canonical non-empty set selected by the scheduler for one continuous
/// batch. Membership is exact; capacity shapes may not claim a different
/// sequence count and no global concurrency ceiling is embedded here.
#[must_use = "a batch participant set is required to admit one execution frame"]
pub struct ExecutionBatchParticipants<R>
where
    R: DeviceRuntime,
{
    sessions: Vec<Arc<SequenceSession<R>>>,
    plan_evidence: TrustedPlanRuntimeEvidence,
}

fn sequence_participant_key<R: DeviceRuntime>(
    sequence: &AdmittedSequenceResources<R>,
) -> (u32, u64, u32, u64) {
    let sequence_authority = sequence.sequence_authority();
    let request_authority = sequence.request_authority();
    (
        sequence_authority.sparse_id(),
        sequence_authority.generation(),
        request_authority.sparse_id(),
        request_authority.generation(),
    )
}

fn session_participant_key<R: DeviceRuntime>(session: &SequenceSession<R>) -> (u32, u64, u32, u64) {
    sequence_participant_key(session.resources())
}

impl<R> ExecutionBatchParticipants<R>
where
    R: DeviceRuntime,
{
    pub fn new(mut sessions: Vec<Arc<SequenceSession<R>>>) -> Result<Self, VNextError> {
        sessions.sort_by_key(|session| session_participant_key(session));
        if sessions.is_empty()
            || sessions
                .windows(2)
                .any(|pair| session_participant_key(&pair[0]) == session_participant_key(&pair[1]))
        {
            return Err(invalid_resource(
                "execution batch participants must be non-empty and unique",
            ));
        }
        u32::try_from(sessions.len())
            .map_err(|_| invalid_resource("execution batch participant count exceeds u32"))?;
        let plan_evidence = sessions[0].resources().plan_evidence();
        if sessions.iter().any(|session| {
            session.resources().is_poisoned()
                || session.resources().plan_evidence() != plan_evidence
        }) {
            return Err(invalid_resource(
                "execution batch participants differ in plan, runtime, pool, coordinator, or health",
            ));
        }
        let resources = Arc::clone(&sessions[0].resources().request.plan.resources);
        if sessions
            .iter()
            .any(|session| !Arc::ptr_eq(&resources, &session.resources().request.plan.resources))
        {
            return Err(invalid_resource(
                "execution batch participants belong to distinct plan runtime roots",
            ));
        }
        let _lifecycle = resources.read_lifecycle("create execution batch participants")?;
        Ok(Self {
            sessions,
            plan_evidence,
        })
    }

    pub fn len(&self) -> u32 {
        u32::try_from(self.sessions.len())
            .expect("execution batch participant count is validated at construction")
    }

    pub fn is_empty(&self) -> bool {
        false
    }

    pub fn sessions(&self) -> &[Arc<SequenceSession<R>>] {
        &self.sessions
    }

    pub fn plan_evidence(&self) -> &TrustedPlanRuntimeEvidence {
        &self.plan_evidence
    }

    pub fn bind_work_shape(
        &self,
        token_spans: Vec<TokenSpanWork>,
    ) -> Result<BatchWorkShape, VNextError> {
        if token_spans.len() != self.sessions.len() {
            return Err(invalid_resource(
                "batch token work count differs from its exact participant set",
            ));
        }
        BatchWorkShape::new(
            self.sessions
                .iter()
                .zip(token_spans)
                .map(|(session, token_span)| {
                    BatchParticipantTokenSpan::new(
                        BatchParticipantAuthority::new(
                            session.sequence_authority(),
                            session.request_authority(),
                        ),
                        token_span,
                    )
                })
                .collect(),
        )
    }
}

#[derive(Clone)]
struct SequenceFrameCandidate {
    slot: Arc<SequenceSessionSlot>,
    epoch: SequenceSessionEpoch,
    fingerprint: SequenceSessionFingerprint,
}

struct SessionFrameHold {
    slot: Arc<SequenceSessionSlot>,
    epoch: SequenceSessionEpoch,
    fingerprint: SequenceSessionFingerprint,
    frame_id: ExecutionFrameId,
    batch_step_id: BatchStepId,
    finalized: bool,
}

#[derive(Clone)]
struct ParticipantFlightCandidate {
    slot: Arc<SequenceSessionSlot>,
    epoch: SequenceSessionEpoch,
    fingerprint: SequenceSessionFingerprint,
    frame: ActiveSequenceFrame,
    participant: BatchParticipantAuthority,
}

/// One participant-local flight owned by an exact invocation. Dropping this
/// hold removes the sequence-flight count, while the physical ledger guard
/// independently retires its topology. Only the sealed device DNF path may
/// reset an in-flight hold to Prepared for retry.
struct PreparedParticipantFlightHold {
    slot: Arc<SequenceSessionSlot>,
    epoch: SequenceSessionEpoch,
    fingerprint: SequenceSessionFingerprint,
    key: ParticipantNodeKey,
    batch_step_id: BatchStepId,
    phase: ParticipantFlightPhase,
}

impl Drop for PreparedParticipantFlightHold {
    fn drop(&mut self) {
        let mut state = match self.slot.state.lock() {
            Ok(state) => state,
            Err(poisoned) => {
                let mut state = poisoned.into_inner();
                *state = SequenceSessionSlotState::FailClosed;
                return;
            }
        };
        match &mut *state {
            SequenceSessionSlotState::Active(active)
                if active.epoch == self.epoch && active.fingerprint == self.fingerprint =>
            {
                if active.participant_flights.remove(&self.key) != Some(self.phase) {
                    active.phase = SequenceSessionPhase::Poisoned;
                }
            }
            _ => *state = SequenceSessionSlotState::FailClosed,
        }
    }
}

fn prepare_participant_flights(
    candidates: &[ParticipantFlightCandidate],
    node_id: &NodeId,
) -> Result<Vec<PreparedParticipantFlightHold>, VNextError> {
    if candidates.is_empty()
        || candidates.iter().enumerate().any(|(index, candidate)| {
            candidates[..index]
                .iter()
                .any(|prior| Arc::ptr_eq(&prior.slot, &candidate.slot))
        })
    {
        return Err(invalid_resource(
            "prepared invocation requires non-empty unique participant sessions",
        ));
    }
    let mut holds = Vec::with_capacity(candidates.len());
    let mut states = Vec::with_capacity(candidates.len());
    for candidate in candidates {
        states.push(
            candidate
                .slot
                .state
                .lock()
                .map_err(|_| invalid_resource("sequence session state mutex is poisoned"))?,
        );
    }
    for (candidate, state) in candidates.iter().zip(&states) {
        let key = ParticipantNodeKey::new(
            candidate.participant,
            candidate.frame.frame_id,
            node_id.clone(),
        );
        match &**state {
            SequenceSessionSlotState::Active(active)
                if active.epoch == candidate.epoch
                    && active.fingerprint == candidate.fingerprint
                    && active.phase == SequenceSessionPhase::Open
                    && active.active_frame == Some(candidate.frame)
                    && !active.participant_flights.contains_key(&key) => {}
            SequenceSessionSlotState::Active(active)
                if active.epoch != candidate.epoch
                    || active.fingerprint != candidate.fingerprint =>
            {
                return Err(invalid_resource(
                    "stale prepared invocation participant authority",
                ));
            }
            SequenceSessionSlotState::Active(_) => {
                return Err(invalid_resource(
                    "cancelled, poisoned, duplicate, or cross-frame participant cannot enter an invocation",
                ));
            }
            _ => {
                return Err(invalid_resource(
                    "inactive or terminal participant cannot enter an invocation",
                ));
            }
        }
    }
    let mut insertion_failure = None;
    for (index, (candidate, state)) in candidates.iter().zip(&mut states).enumerate() {
        let SequenceSessionSlotState::Active(active) = &mut **state else {
            unreachable!("all prepared invocation participants were validated");
        };
        let key = ParticipantNodeKey::new(
            candidate.participant,
            candidate.frame.frame_id,
            node_id.clone(),
        );
        if let Some(previous) = active
            .participant_flights
            .insert(key.clone(), ParticipantFlightPhase::Prepared)
        {
            active.participant_flights.insert(key, previous);
            active.phase = SequenceSessionPhase::Poisoned;
            insertion_failure = Some(index);
            break;
        }
    }
    if let Some(index) = insertion_failure {
        for rollback_index in 0..index {
            let rollback_key = ParticipantNodeKey::new(
                candidates[rollback_index].participant,
                candidates[rollback_index].frame.frame_id,
                node_id.clone(),
            );
            let SequenceSessionSlotState::Active(rollback) = &mut *states[rollback_index] else {
                continue;
            };
            rollback.participant_flights.remove(&rollback_key);
            rollback.phase = SequenceSessionPhase::Poisoned;
        }
        return Err(invalid_resource(
            "prepared participant flight changed during atomic insertion",
        ));
    }
    drop(states);
    for candidate in candidates {
        let key = ParticipantNodeKey::new(
            candidate.participant,
            candidate.frame.frame_id,
            node_id.clone(),
        );
        holds.push(PreparedParticipantFlightHold {
            slot: Arc::clone(&candidate.slot),
            epoch: candidate.epoch,
            fingerprint: candidate.fingerprint.clone(),
            key,
            batch_step_id: candidate.frame.batch_step_id,
            phase: ParticipantFlightPhase::Prepared,
        });
    }
    Ok(holds)
}

fn begin_participant_flights_dispatch(
    holds: &mut [PreparedParticipantFlightHold],
) -> Result<(), VNextError> {
    transition_participant_flights(
        holds,
        ParticipantFlightPhase::Prepared,
        ParticipantFlightPhase::InFlight,
        "begin dispatch",
    )
}

fn reset_participant_flights_after_definitely_not_submitted(
    holds: &mut [PreparedParticipantFlightHold],
) -> Result<(), VNextError> {
    transition_participant_flights(
        holds,
        ParticipantFlightPhase::InFlight,
        ParticipantFlightPhase::Prepared,
        "reset definitely-not-submitted dispatch",
    )
}

fn transition_participant_flights(
    holds: &mut [PreparedParticipantFlightHold],
    expected: ParticipantFlightPhase,
    next: ParticipantFlightPhase,
    context: &'static str,
) -> Result<(), VNextError> {
    if holds.is_empty()
        || holds.iter().enumerate().any(|(index, hold)| {
            hold.phase != expected
                || holds[..index]
                    .iter()
                    .any(|prior| Arc::ptr_eq(&prior.slot, &hold.slot))
        })
    {
        return Err(invalid_resource(format!(
            "{context} requires non-empty unique participant flights in the expected phase"
        )));
    }

    // Holds originate from the canonically ordered batch participant set. Keep
    // every session lock until all checks and transitions have completed so a
    // concurrent cancellation is ordered wholly before or after this change.
    let candidates = holds
        .iter()
        .map(|hold| {
            (
                Arc::clone(&hold.slot),
                hold.epoch,
                hold.fingerprint.clone(),
                hold.key.clone(),
                hold.batch_step_id,
            )
        })
        .collect::<Vec<_>>();
    let mut states = Vec::with_capacity(candidates.len());
    for (slot, _, _, _, _) in &candidates {
        states.push(
            slot.state
                .lock()
                .map_err(|_| invalid_resource("sequence session state mutex is poisoned"))?,
        );
    }
    for ((_, epoch, fingerprint, key, batch_step_id), state) in candidates.iter().zip(&states) {
        match &**state {
            SequenceSessionSlotState::Active(active)
                if active.epoch == *epoch
                    && active.fingerprint == *fingerprint
                    && active.phase == SequenceSessionPhase::Open
                    && active.active_frame
                        == Some(ActiveSequenceFrame {
                            frame_id: key.frame_id(),
                            batch_step_id: *batch_step_id,
                        })
                    && active.participant_flights.get(key) == Some(&expected) => {}
            SequenceSessionSlotState::Active(active)
                if active.epoch != *epoch || active.fingerprint != *fingerprint =>
            {
                return Err(invalid_resource(
                    "stale invocation participant authority during phase transition",
                ));
            }
            SequenceSessionSlotState::Active(_) => {
                return Err(invalid_resource(format!(
                    "cancelled, poisoned, wrong-phase, or cross-frame participant cannot {context}"
                )));
            }
            _ => {
                return Err(invalid_resource(format!(
                    "inactive or terminal participant cannot {context}"
                )));
            }
        }
    }
    for ((_, _, _, key, _), state) in candidates.iter().zip(&mut states) {
        let SequenceSessionSlotState::Active(active) = &mut **state else {
            unreachable!("all dispatch participants were validated while locked");
        };
        let phase = active
            .participant_flights
            .get_mut(key)
            .expect("validated participant flight remains present while locked");
        *phase = next;
    }
    for hold in holds {
        hold.phase = next;
    }
    Ok(())
}

impl Drop for SessionFrameHold {
    fn drop(&mut self) {
        if self.finalized {
            return;
        }
        let mut state = match self.slot.state.lock() {
            Ok(state) => state,
            Err(poisoned) => poisoned.into_inner(),
        };
        if let SequenceSessionSlotState::Active(active) = &mut *state {
            if active.epoch == self.epoch
                && active.fingerprint == self.fingerprint
                && active.active_frame
                    == Some(ActiveSequenceFrame {
                        frame_id: self.frame_id,
                        batch_step_id: self.batch_step_id,
                    })
            {
                active.phase = SequenceSessionPhase::Poisoned;
            }
        }
    }
}

struct AdmittedStepParticipant<R>
where
    R: DeviceRuntime,
{
    frame: SessionFrameHold,
    session: Arc<SequenceSession<R>>,
}

fn acquire_session_frames(
    candidates: &[SequenceFrameCandidate],
    batch_step_id: BatchStepId,
) -> Result<Vec<SessionFrameHold>, VNextError> {
    if candidates.is_empty()
        || candidates.iter().enumerate().any(|(index, candidate)| {
            candidates[..index]
                .iter()
                .any(|prior| Arc::ptr_eq(&prior.slot, &candidate.slot))
        })
    {
        return Err(invalid_resource(
            "step frame acquisition requires non-empty unique session slots",
        ));
    }
    let mut holds = Vec::with_capacity(candidates.len());
    let mut states = Vec::with_capacity(candidates.len());
    for candidate in candidates {
        states.push(
            candidate
                .slot
                .state
                .lock()
                .map_err(|_| invalid_resource("sequence session state mutex is poisoned"))?,
        );
    }
    for (candidate, state) in candidates.iter().zip(&states) {
        match &**state {
            SequenceSessionSlotState::Active(active)
                if active.epoch == candidate.epoch
                    && active.fingerprint == candidate.fingerprint
                    && active.phase == SequenceSessionPhase::Open
                    && active.active_frame.is_none()
                    && active.next_frame.is_some() => {}
            SequenceSessionSlotState::Active(active)
                if active.epoch != candidate.epoch
                    || active.fingerprint != candidate.fingerprint =>
            {
                return Err(invalid_resource("stale sequence session frame authority"));
            }
            SequenceSessionSlotState::Active(_) => {
                return Err(invalid_resource(
                    "sequence session cannot acquire a frame in its current phase",
                ));
            }
            _ => {
                return Err(invalid_resource(
                    "inactive or terminal sequence session cannot acquire a frame",
                ));
            }
        }
    }
    for (candidate, state) in candidates.iter().zip(&mut states) {
        let SequenceSessionSlotState::Active(active) = &mut **state else {
            unreachable!("all session frame candidates were validated");
        };
        let frame_id = active
            .next_frame
            .take()
            .expect("validated session has a next execution frame");
        active.next_frame = frame_id
            .get()
            .checked_add(1)
            .and_then(|next| ExecutionFrameId::try_from(next).ok());
        active.active_frame = Some(ActiveSequenceFrame {
            frame_id,
            batch_step_id,
        });
        holds.push(SessionFrameHold {
            slot: Arc::clone(&candidate.slot),
            epoch: candidate.epoch,
            fingerprint: candidate.fingerprint.clone(),
            frame_id,
            batch_step_id,
            finalized: false,
        });
    }
    Ok(holds)
}

fn session_frame_candidates<R: DeviceRuntime>(
    sessions: &[Arc<SequenceSession<R>>],
) -> Vec<SequenceFrameCandidate> {
    sessions
        .iter()
        .map(|session| SequenceFrameCandidate {
            slot: Arc::clone(&session.slot),
            epoch: session.epoch,
            fingerprint: session.fingerprint.clone(),
        })
        .collect()
}

fn poison_session_frame(hold: &SessionFrameHold) {
    let mut state = match hold.slot.state.lock() {
        Ok(state) => state,
        Err(poisoned) => poisoned.into_inner(),
    };
    if let SequenceSessionSlotState::Active(active) = &mut *state {
        if active.epoch == hold.epoch
            && active.fingerprint == hold.fingerprint
            && active.active_frame
                == Some(ActiveSequenceFrame {
                    frame_id: hold.frame_id,
                    batch_step_id: hold.batch_step_id,
                })
        {
            active.phase = SequenceSessionPhase::Poisoned;
        }
    }
}

fn finalize_session_frames(
    holds: &mut [&mut SessionFrameHold],
    abort: bool,
) -> Result<Vec<StepParticipantRetirementDisposition>, VNextError> {
    let slots = holds
        .iter()
        .map(|hold| Arc::clone(&hold.slot))
        .collect::<Vec<_>>();
    let mut states = Vec::with_capacity(slots.len());
    for slot in &slots {
        states.push(
            slot.state
                .lock()
                .map_err(|_| invalid_resource("sequence session state mutex is poisoned"))?,
        );
    }
    let mut dispositions = Vec::with_capacity(holds.len());
    for (hold, state) in holds.iter().zip(&states) {
        let hold = &**hold;
        let SequenceSessionSlotState::Active(active) = &**state else {
            return Err(invalid_resource(
                "step participant session is no longer active",
            ));
        };
        if active.epoch != hold.epoch
            || active.fingerprint != hold.fingerprint
            || active.active_frame
                != Some(ActiveSequenceFrame {
                    frame_id: hold.frame_id,
                    batch_step_id: hold.batch_step_id,
                })
            || !active.participant_flights.is_empty()
            || (!abort && active.phase == SequenceSessionPhase::Poisoned)
            || (!abort && active.retired_frames == u64::MAX)
        {
            return Err(invalid_resource(
                "step finalization differs from its exact session frame or has live participant work",
            ));
        }
        dispositions.push(if abort {
            StepParticipantRetirementDisposition::Aborted
        } else if active.phase == SequenceSessionPhase::CancelRequested {
            StepParticipantRetirementDisposition::DiscardedCancelled
        } else {
            StepParticipantRetirementDisposition::Committed
        });
    }
    for state in &mut states {
        let SequenceSessionSlotState::Active(active) = &mut **state else {
            unreachable!("all step participant sessions were validated");
        };
        active.active_frame = None;
        if abort {
            active.phase = SequenceSessionPhase::Poisoned;
        } else {
            active.retired_frames += 1;
        }
    }
    drop(states);
    for hold in holds {
        hold.finalized = true;
    }
    Ok(dispositions)
}

#[derive(Default)]
struct InvocationRegistryState {
    entries: BTreeMap<ParticipantNodeKey, ParticipantNodeLedgerEntry>,
    poisoned: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PhysicalInvocationPhase {
    Prepared,
    NotSubmitted,
    InFlight,
    Retired,
}

#[derive(Clone, PartialEq, Eq)]
struct ParticipantNodeLedgerEntry {
    batch_invocation_id: BatchInvocationId,
    work_fingerprint: String,
    phase: PhysicalInvocationPhase,
}

#[derive(Default)]
struct InvocationRegistry {
    state: Mutex<InvocationRegistryState>,
}

impl InvocationRegistry {
    fn enter(
        self: &Arc<Self>,
        keys: Vec<ParticipantNodeKey>,
        batch_invocation_id: BatchInvocationId,
        work_fingerprint: &str,
    ) -> Result<ActiveInvocationGuard, VNextError> {
        if keys.is_empty() || keys.windows(2).any(|pair| pair[0] >= pair[1]) {
            return Err(invalid_resource(
                "physical invocation ledger requires canonical non-empty unique participant-node keys",
            ));
        }
        let mut state = self
            .state
            .lock()
            .map_err(|_| invalid_resource("invocation registry is poisoned"))?;
        if state.poisoned {
            return Err(invalid_resource("invocation registry is fail-closed"));
        }
        if keys.iter().any(|key| state.entries.contains_key(key)) {
            return Err(invalid_resource(
                "participant/frame/node topology is already prepared, in flight, or retired in this step",
            ));
        }
        let entry = ParticipantNodeLedgerEntry {
            batch_invocation_id,
            work_fingerprint: work_fingerprint.to_owned(),
            phase: PhysicalInvocationPhase::Prepared,
        };
        for key in &keys {
            if state.entries.insert(key.clone(), entry.clone()).is_some() {
                state.poisoned = true;
                return Err(invalid_resource(
                    "physical invocation ledger changed during atomic prepare",
                ));
            }
        }
        Ok(ActiveInvocationGuard {
            registry: Arc::clone(self),
            keys,
            work_fingerprint: work_fingerprint.to_owned(),
            batch_invocation_id,
            phase: PhysicalInvocationPhase::Prepared,
        })
    }
}

struct ActiveInvocationGuard {
    registry: Arc<InvocationRegistry>,
    keys: Vec<ParticipantNodeKey>,
    work_fingerprint: String,
    batch_invocation_id: BatchInvocationId,
    phase: PhysicalInvocationPhase,
}

impl ActiveInvocationGuard {
    fn transition(
        &mut self,
        expected: PhysicalInvocationPhase,
        next: PhysicalInvocationPhase,
        next_attempt: BatchInvocationId,
    ) -> Result<(), VNextError> {
        if self.phase != expected {
            return Err(invalid_resource(
                "physical invocation guard is not in the expected phase",
            ));
        }
        let mut state = self
            .registry
            .state
            .lock()
            .map_err(|_| invalid_resource("invocation registry is poisoned"))?;
        let expected_entry = ParticipantNodeLedgerEntry {
            batch_invocation_id: self.batch_invocation_id,
            work_fingerprint: self.work_fingerprint.clone(),
            phase: expected,
        };
        if state.poisoned
            || self
                .keys
                .iter()
                .any(|key| state.entries.get(key) != Some(&expected_entry))
        {
            state.poisoned = true;
            return Err(invalid_resource(
                "physical invocation ledger differs from its exact transition authority",
            ));
        }
        for key in &self.keys {
            let entry = state
                .entries
                .get_mut(key)
                .expect("validated physical invocation key remains present");
            entry.batch_invocation_id = next_attempt;
            entry.phase = next;
        }
        self.batch_invocation_id = next_attempt;
        self.phase = next;
        Ok(())
    }

    fn mark_not_submitted(&mut self) -> Result<(), VNextError> {
        self.transition(
            PhysicalInvocationPhase::Prepared,
            PhysicalInvocationPhase::NotSubmitted,
            self.batch_invocation_id,
        )
    }

    fn prepare_retry(&mut self, fresh_attempt: BatchInvocationId) -> Result<(), VNextError> {
        if fresh_attempt == self.batch_invocation_id {
            return Err(invalid_resource(
                "definitely-not-submitted retry requires a fresh physical attempt id",
            ));
        }
        self.transition(
            PhysicalInvocationPhase::NotSubmitted,
            PhysicalInvocationPhase::Prepared,
            fresh_attempt,
        )
    }

    fn mark_in_flight(&mut self) -> Result<(), VNextError> {
        self.transition(
            PhysicalInvocationPhase::Prepared,
            PhysicalInvocationPhase::InFlight,
            self.batch_invocation_id,
        )
    }
}

impl Drop for ActiveInvocationGuard {
    fn drop(&mut self) {
        if self.phase == PhysicalInvocationPhase::Retired {
            return;
        }
        let mut state = match self.registry.state.lock() {
            Ok(state) => state,
            Err(poisoned) => {
                let mut state = poisoned.into_inner();
                state.poisoned = true;
                state
            }
        };
        let expected = ParticipantNodeLedgerEntry {
            batch_invocation_id: self.batch_invocation_id,
            work_fingerprint: self.work_fingerprint.clone(),
            phase: self.phase,
        };
        if self
            .keys
            .iter()
            .any(|key| state.entries.get(key) != Some(&expected))
        {
            state.poisoned = true;
            return;
        }
        for key in &self.keys {
            state
                .entries
                .get_mut(key)
                .expect("validated physical invocation key remains present")
                .phase = PhysicalInvocationPhase::Retired;
        }
        self.phase = PhysicalInvocationPhase::Retired;
    }
}

pub enum StepResourceAdmissionDecision<R>
where
    R: DeviceRuntime,
{
    Admitted(Arc<StepResourceLease<R>>),
    Deferred(AdmissionDeferred),
    BackingDeferred(DynamicBackingDeferred),
    PermanentRejected(AdmissionRejected),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum StepParticipantRetirementDisposition {
    Committed,
    DiscardedCancelled,
    Aborted,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct StepParticipantRetirement {
    assignment: StepParticipantFrameAssignment,
    disposition: StepParticipantRetirementDisposition,
}

impl StepParticipantRetirement {
    pub const fn assignment(&self) -> StepParticipantFrameAssignment {
        self.assignment
    }

    pub const fn disposition(&self) -> StepParticipantRetirementDisposition {
        self.disposition
    }
}

/// One atomic physical/logical backing claim bound to an immutable batch work
/// authority. Even an empty resource demand retains the work shape and claim
/// fingerprint through dispatch and fence ownership.
#[must_use = "claimed backing must remain owned through its device fence"]
pub struct ClaimedBackingTransaction {
    // Physical extents release before the logical capacity claim.
    backing_slices: Vec<LogicalBackingSliceAuthority>,
    logical_capacity: Option<LogicalBatchCapacityLease>,
    work_shape: BatchWorkShape,
    demand: AdmissionDemand,
    fingerprint: String,
}

impl ClaimedBackingTransaction {
    fn new(
        work_shape: BatchWorkShape,
        demand: AdmissionDemand,
        logical_capacity: Option<LogicalBatchCapacityLease>,
        backing_slices: Vec<LogicalBackingSliceAuthority>,
    ) -> Result<Self, VNextError> {
        let mut backing_by_domain = BTreeMap::<CapacityDomainId, u64>::new();
        for slice in &backing_slices {
            let total = backing_by_domain.entry(slice.domain_id()).or_default();
            *total = total
                .checked_add(slice.size_bytes())
                .ok_or_else(|| invalid_resource("claimed backing domain bytes overflow u64"))?;
        }
        let backing_claim = if backing_by_domain.is_empty() {
            CapacityVector::empty()
        } else {
            CapacityVector::new(
                backing_by_domain
                    .into_iter()
                    .map(|(domain, bytes)| CapacityEntry::new(domain, CapacityUnits::new(bytes)))
                    .collect::<Result<Vec<_>, _>>()?,
            )?
        };
        if backing_claim != *demand.immediate_claim() {
            return Err(invalid_resource(
                "physical backing differs from the exact evaluated immediate demand",
            ));
        }
        match &logical_capacity {
            Some(capacity)
                if capacity.claims() == demand.immediate_claim()
                    && capacity.parents().len() == work_shape.participants().len()
                    && capacity
                        .parents()
                        .iter()
                        .zip(work_shape.participants())
                        .all(|(parent, participant)| {
                            parent.sequence() == participant.sequence_authority()
                                && parent.request() == participant.request_authority()
                        }) => {}
            None if demand.immediate_claim().is_empty() && backing_slices.is_empty() => {}
            _ => {
                return Err(invalid_resource(
                    "logical backing claim differs from work participants or evaluated demand",
                ))
            }
        }
        #[derive(Serialize)]
        struct FingerprintInput<'a> {
            domain: &'static str,
            work_fingerprint: &'a str,
            demand: &'a AdmissionDemand,
            backing: Vec<&'a LogicalBackingSliceEvidence>,
            capacity_parents: Vec<(SequenceAuthorityId, RequestAuthorityId)>,
        }
        let input = FingerprintInput {
            domain: "ferrum.runtime-vnext.claimed-backing.v1",
            work_fingerprint: work_shape.fingerprint(),
            demand: &demand,
            backing: backing_slices
                .iter()
                .map(LogicalBackingSliceAuthority::evidence)
                .collect(),
            capacity_parents: logical_capacity
                .as_ref()
                .map(|capacity| {
                    capacity
                        .parents()
                        .iter()
                        .map(|parent| (parent.sequence(), parent.request()))
                        .collect()
                })
                .unwrap_or_default(),
        };
        let bytes = serde_json::to_vec(&input).map_err(|error| {
            invalid_resource(format!(
                "claimed backing fingerprint encode failed: {error}"
            ))
        })?;
        Ok(Self {
            backing_slices,
            logical_capacity,
            work_shape,
            demand,
            fingerprint: format!("{:x}", Sha256::digest(bytes)),
        })
    }

    pub fn work_shape(&self) -> &BatchWorkShape {
        &self.work_shape
    }

    pub fn backing_slices(&self) -> &[LogicalBackingSliceAuthority] {
        &self.backing_slices
    }

    pub fn logical_capacity(&self) -> Option<&LogicalBatchCapacityLease> {
        self.logical_capacity.as_ref()
    }

    pub fn fingerprint(&self) -> &str {
        &self.fingerprint
    }

    pub fn demand(&self) -> &AdmissionDemand {
        &self.demand
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct StepRetirementReceipt {
    batch_step_id: BatchStepId,
    participants: Vec<StepParticipantRetirement>,
}

impl StepRetirementReceipt {
    pub const fn batch_step_id(&self) -> BatchStepId {
        self.batch_step_id
    }

    pub fn participants(&self) -> &[StepParticipantRetirement] {
        &self.participants
    }
}

#[must_use = "failed step finalization retains the exact step authority"]
pub struct StepFinalizationFailure<R>
where
    R: DeviceRuntime,
{
    step: Arc<StepResourceLease<R>>,
    error: VNextError,
}

impl<R> fmt::Debug for StepFinalizationFailure<R>
where
    R: DeviceRuntime,
{
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("StepFinalizationFailure")
            .field("error", &self.error)
            .finish_non_exhaustive()
    }
}

impl<R> StepFinalizationFailure<R>
where
    R: DeviceRuntime,
{
    pub fn error(&self) -> &VNextError {
        &self.error
    }

    pub fn into_step(self) -> Arc<StepResourceLease<R>> {
        self.step
    }
}

/// Resources whose lifetime is one exact continuous-batch execution frame.
/// Child invocation leases retain this scope through `Arc`, so shared frame
/// capacity and every participant authority outlive asynchronous device work.
#[must_use = "step resources must live through every child invocation"]
pub struct StepResourceLease<R>
where
    R: DeviceRuntime,
{
    // The transaction releases physical extents before its logical claim,
    // then per-sequence frame guards release before their parent sessions.
    claimed_backing: ClaimedBackingTransaction,
    participants: Vec<AdmittedStepParticipant<R>>,
    invocation_registry: Arc<InvocationRegistry>,
    batch_step_id: BatchStepId,
    finalized: bool,
}

impl<R> StepResourceLease<R>
where
    R: DeviceRuntime,
{
    fn new(
        participants: Vec<AdmittedStepParticipant<R>>,
        batch_step_id: BatchStepId,
        claimed_backing: ClaimedBackingTransaction,
    ) -> Result<Self, VNextError> {
        if participants.is_empty() {
            return Err(invalid_resource(
                "step resources require a non-empty participant set",
            ));
        }
        let coordinator = participants[0]
            .session
            .resources()
            .request
            .plan
            .logical_admission();
        if claimed_backing.work_shape().participants().len() != participants.len()
            || claimed_backing
                .work_shape()
                .participants()
                .iter()
                .zip(&participants)
                .any(|(authority, participant)| {
                    authority.sequence_authority() != participant.session.sequence_authority()
                        || authority.request_authority() != participant.session.request_authority()
                })
        {
            return Err(invalid_resource(
                "step work shape differs from its exact batch participants",
            ));
        }
        if let Some(capacity) = claimed_backing.logical_capacity() {
            let parents_match = capacity
                .parents()
                .iter()
                .map(|parent| (parent.sequence(), parent.request()))
                .eq(participants.iter().map(|participant| {
                    (
                        participant.session.sequence_authority(),
                        participant.session.request_authority(),
                    )
                }));
            if !coordinator.owns_batch_capacity_claim(capacity) || !parents_match {
                return Err(invalid_resource(
                    "step capacity authority differs from its exact batch participants",
                ));
            }
        }
        Ok(Self {
            claimed_backing,
            participants,
            invocation_registry: Arc::new(InvocationRegistry::default()),
            batch_step_id,
            finalized: false,
        })
    }

    pub const fn batch_step_id(&self) -> BatchStepId {
        self.batch_step_id
    }

    pub fn try_retire_normal(
        self: Arc<Self>,
    ) -> Result<StepRetirementReceipt, StepFinalizationFailure<R>> {
        Self::try_finalize(self, false)
    }

    pub fn try_abort(self: Arc<Self>) -> Result<StepRetirementReceipt, StepFinalizationFailure<R>> {
        Self::try_finalize(self, true)
    }

    fn try_finalize(
        step: Arc<Self>,
        abort: bool,
    ) -> Result<StepRetirementReceipt, StepFinalizationFailure<R>> {
        let mut step = match Arc::try_unwrap(step) {
            Ok(step) => step,
            Err(step) => {
                return Err(StepFinalizationFailure {
                    step,
                    error: invalid_resource(
                        "step cannot finalize while an invocation or scheduler clone retains it",
                    ),
                });
            }
        };
        let dispositions = match step.finalize_participants(abort) {
            Ok(dispositions) => dispositions,
            Err(error) => {
                return Err(StepFinalizationFailure {
                    step: Arc::new(step),
                    error,
                });
            }
        };
        let participants = step
            .participants
            .iter()
            .zip(dispositions)
            .map(|(participant, disposition)| StepParticipantRetirement {
                assignment: StepParticipantFrameAssignment::new(
                    participant.session.sequence_authority(),
                    participant.session.request_authority(),
                    participant.frame.frame_id,
                ),
                disposition,
            })
            .collect();
        Ok(StepRetirementReceipt {
            batch_step_id: step.batch_step_id,
            participants,
        })
    }

    fn finalize_participants(
        &mut self,
        abort: bool,
    ) -> Result<Vec<StepParticipantRetirementDisposition>, VNextError> {
        if self.finalized {
            return Err(invalid_resource("step resources are already finalized"));
        }
        let mut holds = self
            .participants
            .iter_mut()
            .map(|participant| &mut participant.frame)
            .collect::<Vec<_>>();
        let dispositions = finalize_session_frames(&mut holds, abort)?;
        self.finalized = true;
        Ok(dispositions)
    }

    pub fn participant_count(&self) -> u32 {
        u32::try_from(self.participants.len())
            .expect("step participant count was validated before admission")
    }

    pub fn coordinator_id(&self) -> LogicalAdmissionCoordinatorId {
        self.participants[0].session.resources().coordinator_id()
    }

    pub fn participants(
        &self,
    ) -> impl ExactSizeIterator<Item = &Arc<AdmittedSequenceResources<R>>> {
        self.participants
            .iter()
            .map(|participant| participant.session.resources())
    }

    pub fn participant_frames(
        &self,
    ) -> impl ExactSizeIterator<Item = StepParticipantFrameAssignment> + '_ {
        self.participants.iter().map(|participant| {
            StepParticipantFrameAssignment::new(
                participant.session.sequence_authority(),
                participant.session.request_authority(),
                participant.frame.frame_id,
            )
        })
    }

    pub fn backing_slices(&self) -> &[LogicalBackingSliceAuthority] {
        self.claimed_backing.backing_slices()
    }

    pub fn logical_capacity(&self) -> Option<&LogicalBatchCapacityLease> {
        self.claimed_backing.logical_capacity()
    }

    pub fn work_shape(&self) -> &BatchWorkShape {
        self.claimed_backing.work_shape()
    }

    pub fn bind_invocation_work_shape(
        &self,
        mut participant_tokens: Vec<(BatchParticipantAuthority, TokenSpanWork)>,
    ) -> Result<BatchWorkShape, VNextError> {
        participant_tokens.sort_by_key(|(participant, _)| participant.canonical_key());
        if participant_tokens.is_empty()
            || participant_tokens
                .windows(2)
                .any(|pair| pair[0].0.canonical_key() == pair[1].0.canonical_key())
            || participant_tokens.iter().any(|(authority, _)| {
                self.participants
                    .binary_search_by_key(&authority.canonical_key(), |participant| {
                        session_participant_key(&participant.session)
                    })
                    .is_err()
            })
        {
            return Err(invalid_resource(
                "invocation token work must bind a unique non-empty step participant subset",
            ));
        }
        BatchWorkShape::new(
            participant_tokens
                .into_iter()
                .map(|(participant, token_span)| {
                    BatchParticipantTokenSpan::new(participant, token_span)
                })
                .collect(),
        )
    }

    pub fn bind_all_invocation_work_shape(
        &self,
        token_spans: Vec<TokenSpanWork>,
    ) -> Result<BatchWorkShape, VNextError> {
        if token_spans.len() != self.participants.len() {
            return Err(invalid_resource(
                "invocation token work count differs from all step participants",
            ));
        }
        self.bind_invocation_work_shape(
            self.participants
                .iter()
                .zip(token_spans)
                .map(|(participant, token_span)| {
                    (
                        BatchParticipantAuthority::new(
                            participant.session.sequence_authority(),
                            participant.session.request_authority(),
                        ),
                        token_span,
                    )
                })
                .collect(),
        )
    }

    pub fn claimed_backing(&self) -> &ClaimedBackingTransaction {
        &self.claimed_backing
    }

    pub fn static_provisioning(&self) -> Option<&StaticProvisioningLease<R>> {
        self.participants[0]
            .session
            .resources()
            .static_provisioning()
    }

    pub fn plan_evidence(&self) -> TrustedPlanRuntimeEvidence {
        self.participants[0].session.resources().plan_evidence()
    }

    pub(crate) fn backing_view(
        &self,
        resource_id: &ResourceId,
    ) -> Result<LogicalBackingBufferView<'_, R::Buffer>, VNextError> {
        if let Some(authority) = self
            .claimed_backing
            .backing_slices()
            .iter()
            .find(|authority| authority.resource_id() == resource_id)
        {
            return self.participants[0]
                .session
                .resources()
                .request
                .plan
                .dynamic_pools()
                .view(authority);
        }
        Err(invalid_resource(format!(
            "resource `{resource_id}` is not step-shared backing"
        )))
    }

    pub(crate) fn participant_backing_views(
        &self,
        resource_id: &ResourceId,
    ) -> Result<Vec<(SequenceAuthorityId, LogicalBackingBufferView<'_, R::Buffer>)>, VNextError>
    {
        self.participants
            .iter()
            .map(|participant| {
                Ok((
                    participant.session.sequence_authority(),
                    participant.session.resources().backing_view(resource_id)?,
                ))
            })
            .collect()
    }

    pub fn try_admit_invocation(
        self: &Arc<Self>,
        request: InvocationResourceAdmissionRequest,
    ) -> Result<InvocationResourceAdmissionDecision<R>, VNextError> {
        let _lifecycle = self.participants[0]
            .session
            .resources()
            .request
            .plan
            .resources
            .read_lifecycle("admit a step invocation")?;
        let InvocationResourceAdmissionRequest {
            node_id,
            work_shape,
            fit_policy,
            pressure_action,
        } = request;
        let immediate_shape = work_shape.immediate_shape();
        let fit_shape = match fit_policy {
            AdmissionFitPolicy::ImmediateOnly => immediate_shape,
            AdmissionFitPolicy::FullInputMustFit => work_shape.fit_shape(),
        };
        let participant_sessions = work_shape
            .participants()
            .iter()
            .map(|authority| {
                self.participants
                    .binary_search_by_key(&authority.canonical_key(), |participant| {
                        session_participant_key(&participant.session)
                    })
                    .map(|index| Arc::clone(&self.participants[index].session))
                    .map_err(|_| {
                        invalid_resource(
                            "invocation participant is not a member of its execution frame",
                        )
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut participant_frames = Vec::with_capacity(participant_sessions.len());
        let mut flight_candidates = Vec::with_capacity(participant_sessions.len());
        for participant in &participant_sessions {
            let key = session_participant_key(participant);
            let index = self
                .participants
                .binary_search_by_key(&key, |step_participant| {
                    session_participant_key(&step_participant.session)
                })
                .map_err(|_| {
                    invalid_resource("invocation participant lost its execution-frame assignment")
                })?;
            let step_participant = &self.participants[index];
            participant_frames.push(StepParticipantFrameAssignment::new(
                participant.sequence_authority(),
                participant.request_authority(),
                step_participant.frame.frame_id,
            ));
            flight_candidates.push(ParticipantFlightCandidate {
                slot: Arc::clone(&participant.slot),
                epoch: participant.epoch,
                fingerprint: participant.fingerprint.clone(),
                frame: ActiveSequenceFrame {
                    frame_id: step_participant.frame.frame_id,
                    batch_step_id: self.batch_step_id,
                },
                participant: BatchParticipantAuthority::new(
                    participant.sequence_authority(),
                    participant.request_authority(),
                ),
            });
        }
        let participants = participant_sessions
            .iter()
            .map(|participant| Arc::clone(participant.resources()))
            .collect::<Vec<_>>();
        let participant_count = u32::try_from(participants.len())
            .map_err(|_| invalid_resource("invocation participant count exceeds u32"))?;
        if participant_count == 0
            || immediate_shape.sequences() != participant_count
            || fit_shape.sequences() != participant_count
        {
            return Err(invalid_resource(
                "invocation shape sequence count differs from its exact participant set",
            ));
        }
        let plan = &participants[0].request.plan;
        let (demand, requested_slices) = plan.scoped_demand(
            AllocationLifetime::Invocation,
            Some(&node_id),
            immediate_shape,
            fit_shape,
            fit_policy,
            pressure_action,
        )?;
        let prepared = match plan.prepare_backing_slices(requested_slices)? {
            BackingPrepareDecision::Prepared(prepared) => prepared,
            BackingPrepareDecision::Deferred(deferred) => {
                return Ok(InvocationResourceAdmissionDecision::BackingDeferred(
                    deferred,
                ));
            }
        };
        let participant_authorities = participants
            .iter()
            .map(|participant| {
                BatchParticipantAuthority::new(
                    participant.sequence_authority(),
                    participant.request_authority(),
                )
            })
            .collect::<Vec<_>>();
        if work_shape.participants() != participant_authorities {
            return Err(invalid_resource(
                "invocation work authority differs from selected participants",
            ));
        }
        let logical_capacity = if demand.immediate_claim().is_empty() {
            None
        } else {
            let parents = participants
                .iter()
                .map(|participant| &*participant.logical_lease)
                .collect::<Vec<_>>();
            match plan
                .logical_admission()
                .try_claim_for_sequences(&parents, &demand)?
            {
                BatchCapacityClaimDecision::Claimed(capacity) => {
                    let parents_match = capacity
                        .parents()
                        .iter()
                        .map(|parent| (parent.sequence(), parent.request()))
                        .eq(participants.iter().map(|participant| {
                            (
                                participant.sequence_authority(),
                                participant.request_authority(),
                            )
                        }));
                    if !plan
                        .logical_admission()
                        .owns_batch_capacity_claim(&capacity)
                        || !parents_match
                    {
                        return Err(invalid_resource(
                            "invocation admission returned capacity for another participant set",
                        ));
                    }
                    Some(capacity)
                }
                BatchCapacityClaimDecision::Deferred(deferred) => {
                    return Ok(InvocationResourceAdmissionDecision::Deferred(deferred));
                }
                BatchCapacityClaimDecision::PermanentRejected(rejected) => {
                    return Ok(InvocationResourceAdmissionDecision::PermanentRejected(
                        rejected,
                    ));
                }
            }
        };
        let backing_slices = prepared.commit();
        let claimed_backing =
            ClaimedBackingTransaction::new(work_shape, demand, logical_capacity, backing_slices)?;
        let batch_invocation_id = issue_batch_invocation_id()?;
        let prepared_participant_flights =
            prepare_participant_flights(&flight_candidates, &node_id)?;
        let topology_keys = participant_frames
            .iter()
            .map(|assignment| {
                ParticipantNodeKey::new(
                    assignment.participant(),
                    assignment.frame_id(),
                    node_id.clone(),
                )
            })
            .collect();
        let active_invocation = self.invocation_registry.enter(
            topology_keys,
            batch_invocation_id,
            claimed_backing.work_shape().fingerprint(),
        )?;
        Ok(InvocationResourceAdmissionDecision::Admitted(
            InvocationResourceLease::new(
                Arc::clone(self),
                participants,
                participant_frames,
                node_id,
                batch_invocation_id,
                claimed_backing,
                prepared_participant_flights,
                active_invocation,
            )?,
        ))
    }
}

impl<R> Drop for StepResourceLease<R>
where
    R: DeviceRuntime,
{
    fn drop(&mut self) {
        if !self.finalized {
            for participant in &self.participants {
                poison_session_frame(&participant.frame);
            }
        }
    }
}

pub enum InvocationResourceAdmissionDecision<R>
where
    R: DeviceRuntime,
{
    Admitted(InvocationResourceLease<R>),
    Deferred(AdmissionDeferred),
    BackingDeferred(DynamicBackingDeferred),
    PermanentRejected(AdmissionRejected),
}

/// Exact prepared batch node/provider invocation authority. No device command
/// has been submitted at this layer; dropping it performs the typed
/// definitely-not-submitted participant-flight rollback.
#[must_use = "prepared invocation resources must be dispatched or explicitly dropped"]
pub struct InvocationResourceLease<R>
where
    R: DeviceRuntime,
{
    // Claimed backing is returned before participant-flight and parent frame
    // authorities. It retains the immutable work fingerprint even when empty.
    claimed_backing: ClaimedBackingTransaction,
    prepared_participant_flights: Vec<PreparedParticipantFlightHold>,
    active_invocation: ActiveInvocationGuard,
    participants: Vec<Arc<AdmittedSequenceResources<R>>>,
    participant_frames: Vec<StepParticipantFrameAssignment>,
    step: Arc<StepResourceLease<R>>,
    node_id: NodeId,
    batch_invocation_id: BatchInvocationId,
}

impl<R> InvocationResourceLease<R>
where
    R: DeviceRuntime,
{
    fn new(
        step: Arc<StepResourceLease<R>>,
        participants: Vec<Arc<AdmittedSequenceResources<R>>>,
        participant_frames: Vec<StepParticipantFrameAssignment>,
        node_id: NodeId,
        batch_invocation_id: BatchInvocationId,
        claimed_backing: ClaimedBackingTransaction,
        prepared_participant_flights: Vec<PreparedParticipantFlightHold>,
        active_invocation: ActiveInvocationGuard,
    ) -> Result<Self, VNextError> {
        if participants.is_empty() {
            return Err(invalid_resource(
                "invocation resources require a non-empty participant set",
            ));
        }
        if participant_frames.len() != participants.len()
            || prepared_participant_flights.len() != participants.len()
            || participant_frames
                .iter()
                .zip(&participants)
                .any(|(assignment, participant)| {
                    assignment.sequence_authority() != participant.sequence_authority()
                        || assignment.request_authority() != participant.request_authority()
                })
        {
            return Err(invalid_resource(
                "invocation frame mapping differs from its exact participant set",
            ));
        }
        if claimed_backing.work_shape().participants().len() != participants.len()
            || claimed_backing
                .work_shape()
                .participants()
                .iter()
                .zip(&participants)
                .any(|(authority, participant)| {
                    authority.sequence_authority() != participant.sequence_authority()
                        || authority.request_authority() != participant.request_authority()
                })
            || claimed_backing.work_shape().immediate_tokens()
                > step.work_shape().immediate_tokens()
            || claimed_backing.work_shape().immediate_pages() > step.work_shape().immediate_pages()
            || claimed_backing.work_shape().fit_tokens() > step.work_shape().fit_tokens()
            || claimed_backing.work_shape().fit_pages() > step.work_shape().fit_pages()
        {
            return Err(invalid_resource(
                "invocation work shape differs from participants or exceeds its step",
            ));
        }
        if let Some(capacity) = claimed_backing.logical_capacity() {
            let coordinator = participants[0].request.plan.logical_admission();
            let parents_match = capacity
                .parents()
                .iter()
                .map(|parent| (parent.sequence(), parent.request()))
                .eq(participants.iter().map(|participant| {
                    (
                        participant.sequence_authority(),
                        participant.request_authority(),
                    )
                }));
            if !coordinator.owns_batch_capacity_claim(capacity) || !parents_match {
                return Err(invalid_resource(
                    "invocation capacity authority differs from its exact participants",
                ));
            }
        }
        Ok(Self {
            claimed_backing,
            prepared_participant_flights,
            active_invocation,
            participants,
            participant_frames,
            step,
            node_id,
            batch_invocation_id,
        })
    }

    pub fn node_id(&self) -> &NodeId {
        &self.node_id
    }

    pub const fn batch_invocation_id(&self) -> BatchInvocationId {
        self.batch_invocation_id
    }

    pub fn batch_step_id(&self) -> BatchStepId {
        self.step.batch_step_id()
    }

    pub fn participant_count(&self) -> u32 {
        u32::try_from(self.participants.len())
            .expect("invocation participant count was validated before admission")
    }

    pub fn prepared_participant_count(&self) -> u32 {
        u32::try_from(self.prepared_participant_flights.len())
            .expect("prepared participant count was validated at construction")
    }

    pub(crate) fn exact_single_participant_session_identity(
        &self,
    ) -> Option<(SequenceSessionEpoch, &SequenceSessionFingerprint)> {
        match (
            self.participants.as_slice(),
            self.participant_frames.as_slice(),
            self.prepared_participant_flights.as_slice(),
        ) {
            ([_], [_], [hold]) => Some((hold.epoch, &hold.fingerprint)),
            _ => None,
        }
    }

    pub(crate) fn begin_dispatch(&mut self) -> Result<(), VNextError> {
        begin_participant_flights_dispatch(&mut self.prepared_participant_flights)
    }

    pub(crate) fn mark_submission_fence_installed(&mut self) -> Result<(), VNextError> {
        self.active_invocation.mark_in_flight()
    }

    pub(crate) fn definitely_not_submitted(
        mut self,
    ) -> Result<DefinitelyNotSubmittedRetryAuthority<R>, VNextError> {
        self.active_invocation.mark_not_submitted()?;
        reset_participant_flights_after_definitely_not_submitted(
            &mut self.prepared_participant_flights,
        )?;
        let topology_fingerprint = self.retry_topology_fingerprint()?;
        let work_fingerprint = self.work_shape().fingerprint().to_owned();
        let prior_attempt = self.batch_invocation_id;
        Ok(DefinitelyNotSubmittedRetryAuthority {
            invocation: Some(self),
            topology_fingerprint,
            work_fingerprint,
            prior_attempt,
        })
    }

    fn prepare_definitely_not_submitted_retry(
        &mut self,
        fresh_attempt: BatchInvocationId,
        topology_fingerprint: &str,
        work_fingerprint: &str,
    ) -> Result<(), VNextError> {
        if self.retry_topology_fingerprint()? != topology_fingerprint
            || self.work_shape().fingerprint() != work_fingerprint
            || self
                .prepared_participant_flights
                .iter()
                .any(|hold| hold.phase != ParticipantFlightPhase::Prepared)
        {
            return Err(invalid_resource(
                "definitely-not-submitted retry topology or work fingerprint changed",
            ));
        }
        self.active_invocation.prepare_retry(fresh_attempt)?;
        self.batch_invocation_id = fresh_attempt;
        Ok(())
    }

    fn retry_topology_fingerprint(&self) -> Result<String, VNextError> {
        #[derive(Serialize)]
        struct FingerprintInput<'a> {
            domain: &'static str,
            node_id: &'a NodeId,
            participant_frames: &'a [StepParticipantFrameAssignment],
            work_fingerprint: &'a str,
        }
        let bytes = serde_json::to_vec(&FingerprintInput {
            domain: "ferrum.runtime-vnext.invocation-retry-topology.v1",
            node_id: &self.node_id,
            participant_frames: &self.participant_frames,
            work_fingerprint: self.work_shape().fingerprint(),
        })
        .map_err(|error| {
            invalid_resource(format!(
                "invocation retry topology fingerprint encode failed: {error}"
            ))
        })?;
        Ok(format!("{:x}", Sha256::digest(bytes)))
    }

    pub fn coordinator_id(&self) -> LogicalAdmissionCoordinatorId {
        self.participants[0].coordinator_id()
    }

    pub fn participants(
        &self,
    ) -> impl ExactSizeIterator<Item = &Arc<AdmittedSequenceResources<R>>> {
        self.participants.iter()
    }

    pub fn participant_frames(&self) -> &[StepParticipantFrameAssignment] {
        &self.participant_frames
    }

    pub fn step_resources(&self) -> &Arc<StepResourceLease<R>> {
        &self.step
    }

    pub fn backing_slices(&self) -> &[LogicalBackingSliceAuthority] {
        self.claimed_backing.backing_slices()
    }

    pub fn logical_capacity(&self) -> Option<&LogicalBatchCapacityLease> {
        self.claimed_backing.logical_capacity()
    }

    pub fn work_shape(&self) -> &BatchWorkShape {
        self.claimed_backing.work_shape()
    }

    pub fn claimed_backing(&self) -> &ClaimedBackingTransaction {
        &self.claimed_backing
    }

    pub fn plan_evidence(&self) -> TrustedPlanRuntimeEvidence {
        self.step.plan_evidence()
    }

    pub(crate) fn runtime(&self) -> &Arc<R> {
        self.participants[0].request.plan.runtime()
    }

    pub(crate) fn deferred_cleanup_domain(&self) -> DeferredDeviceCleanupDomainId {
        self.participants[0]
            .request
            .plan
            .resources
            .deferred_cleanup_domain
    }

    pub(crate) fn backing_view(
        &self,
        resource_id: &ResourceId,
    ) -> Result<LogicalBackingBufferView<'_, R::Buffer>, VNextError> {
        if let Some(authority) = self
            .claimed_backing
            .backing_slices()
            .iter()
            .find(|authority| authority.resource_id() == resource_id)
        {
            return self.step.participants[0]
                .session
                .resources()
                .request
                .plan
                .dynamic_pools()
                .view(authority);
        }
        self.step.backing_view(resource_id)
    }

    pub(crate) fn participant_backing_views(
        &self,
        resource_id: &ResourceId,
    ) -> Result<Vec<(SequenceAuthorityId, LogicalBackingBufferView<'_, R::Buffer>)>, VNextError>
    {
        self.participants
            .iter()
            .map(|participant| {
                Ok((
                    participant.sequence_authority(),
                    participant.backing_view(resource_id)?,
                ))
            })
            .collect()
    }
}

/// The sole retry edge after a device runtime proves that submit did not
/// happen. It owns the exact invocation, topology and work evidence; dropping
/// it retires the ledger tombstone and cannot be relabeled as retryable later.
#[must_use = "a definitely-not-submitted retry authority must be retried or retired"]
pub struct DefinitelyNotSubmittedRetryAuthority<R>
where
    R: DeviceRuntime,
{
    invocation: Option<InvocationResourceLease<R>>,
    topology_fingerprint: String,
    work_fingerprint: String,
    prior_attempt: BatchInvocationId,
}

impl<R> fmt::Debug for DefinitelyNotSubmittedRetryAuthority<R>
where
    R: DeviceRuntime,
{
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("DefinitelyNotSubmittedRetryAuthority")
            .field("prior_attempt", &self.prior_attempt)
            .field("topology_fingerprint", &self.topology_fingerprint)
            .field("work_fingerprint", &self.work_fingerprint)
            .finish_non_exhaustive()
    }
}

impl<R> DefinitelyNotSubmittedRetryAuthority<R>
where
    R: DeviceRuntime,
{
    pub const fn prior_attempt(&self) -> BatchInvocationId {
        self.prior_attempt
    }

    pub fn topology_fingerprint(&self) -> &str {
        &self.topology_fingerprint
    }

    pub fn work_fingerprint(&self) -> &str {
        &self.work_fingerprint
    }

    pub fn retry(mut self) -> Result<InvocationResourceLease<R>, VNextError> {
        let fresh_attempt = issue_batch_invocation_id()?;
        let invocation = self
            .invocation
            .as_mut()
            .ok_or_else(|| invalid_resource("retry authority no longer owns its invocation"))?;
        invocation.prepare_definitely_not_submitted_retry(
            fresh_attempt,
            &self.topology_fingerprint,
            &self.work_fingerprint,
        )?;
        Ok(self
            .invocation
            .take()
            .expect("validated retry authority still owns its invocation"))
    }
}

impl<R> ExecutionBatchParticipants<R>
where
    R: DeviceRuntime,
{
    pub fn try_begin_step(
        &self,
        request: StepResourceAdmissionRequest,
    ) -> Result<StepResourceAdmissionDecision<R>, VNextError> {
        let _lifecycle = self.sessions[0]
            .resources()
            .request
            .plan
            .resources
            .read_lifecycle("begin an execution step")?;
        let StepResourceAdmissionRequest {
            work_shape,
            fit_policy,
            pressure_action,
        } = request;
        let expected_participants = self
            .sessions
            .iter()
            .map(|session| {
                BatchParticipantAuthority::new(
                    session.sequence_authority(),
                    session.request_authority(),
                )
            })
            .collect::<Vec<_>>();
        if work_shape.participants() != expected_participants {
            return Err(invalid_resource(
                "step work authority differs from its exact participant set",
            ));
        }
        let immediate_shape = work_shape.immediate_shape();
        let fit_shape = match fit_policy {
            AdmissionFitPolicy::ImmediateOnly => immediate_shape,
            AdmissionFitPolicy::FullInputMustFit => work_shape.fit_shape(),
        };
        let plan = &self.sessions[0].resources().request.plan;
        let (demand, requested_slices) = plan.scoped_demand(
            AllocationLifetime::Step,
            None,
            immediate_shape,
            fit_shape,
            fit_policy,
            pressure_action,
        )?;
        let prepared = match plan.prepare_backing_slices(requested_slices)? {
            BackingPrepareDecision::Prepared(prepared) => prepared,
            BackingPrepareDecision::Deferred(deferred) => {
                return Ok(StepResourceAdmissionDecision::BackingDeferred(deferred));
            }
        };
        let logical_capacity = if demand.immediate_claim().is_empty() {
            None
        } else {
            let parents = self
                .sessions
                .iter()
                .map(|session| &*session.resources().logical_lease)
                .collect::<Vec<_>>();
            match plan
                .logical_admission()
                .try_claim_for_sequences(&parents, &demand)?
            {
                BatchCapacityClaimDecision::Claimed(capacity) => {
                    let parents_match = capacity
                        .parents()
                        .iter()
                        .map(|parent| (parent.sequence(), parent.request()))
                        .eq(self.sessions.iter().map(|session| {
                            (session.sequence_authority(), session.request_authority())
                        }));
                    if !plan
                        .logical_admission()
                        .owns_batch_capacity_claim(&capacity)
                        || !parents_match
                    {
                        return Err(invalid_resource(
                            "step admission returned capacity for another participant set",
                        ));
                    }
                    Some(capacity)
                }
                BatchCapacityClaimDecision::Deferred(deferred) => {
                    return Ok(StepResourceAdmissionDecision::Deferred(deferred));
                }
                BatchCapacityClaimDecision::PermanentRejected(rejected) => {
                    return Ok(StepResourceAdmissionDecision::PermanentRejected(rejected));
                }
            }
        };
        let backing_slices = prepared.commit();
        let claimed_backing =
            ClaimedBackingTransaction::new(work_shape, demand, logical_capacity, backing_slices)?;
        let batch_step_id = issue_batch_step_id()?;
        let candidates = session_frame_candidates(&self.sessions);
        let frames = acquire_session_frames(&candidates, batch_step_id)?;
        let participants = self
            .sessions
            .iter()
            .cloned()
            .zip(frames)
            .map(|(session, frame)| AdmittedStepParticipant { frame, session })
            .collect();
        Ok(StepResourceAdmissionDecision::Admitted(Arc::new(
            StepResourceLease::new(participants, batch_step_id, claimed_backing)?,
        )))
    }
}

const fn sequence_slot_active(epoch: u64) -> u64 {
    (epoch << 2) | 1
}

const fn sequence_slot_poisoned_drained(epoch: u64) -> u64 {
    (epoch << 2) | 2
}

const fn sequence_slot_poisoned_undrained(epoch: u64) -> u64 {
    (epoch << 2) | 3
}

const fn sequence_slot_is_poisoned(state: u64) -> bool {
    matches!(state & 3, 2 | 3)
}

const SEQUENCE_DISPATCH_POISONED_BIT: u64 = 1 << 63;
const SEQUENCE_DISPATCH_COUNT_MASK: u64 = SEQUENCE_DISPATCH_POISONED_BIT - 1;

fn sequence_dispatch_is_poisoned(gate: &AtomicU64) -> bool {
    gate.load(Ordering::Acquire) & SEQUENCE_DISPATCH_POISONED_BIT != 0
}

struct SequenceDispatchGuard<'a> {
    gate: &'a AtomicU64,
}

impl Drop for SequenceDispatchGuard<'_> {
    fn drop(&mut self) {
        let previous = self.gate.fetch_sub(1, Ordering::AcqRel);
        debug_assert!(previous & SEQUENCE_DISPATCH_COUNT_MASK > 0);
    }
}

fn enter_sequence_dispatch(gate: &AtomicU64) -> Result<SequenceDispatchGuard<'_>, VNextError> {
    gate.fetch_update(Ordering::AcqRel, Ordering::Acquire, |state| {
        if state & SEQUENCE_DISPATCH_POISONED_BIT != 0
            || state & SEQUENCE_DISPATCH_COUNT_MASK == SEQUENCE_DISPATCH_COUNT_MASK
        {
            None
        } else {
            Some(state + 1)
        }
    })
    .map_err(|state| {
        if state & SEQUENCE_DISPATCH_POISONED_BIT != 0 {
            invalid_resource("poisoned resource pool cannot dispatch another operation")
        } else {
            invalid_resource("resource pool dispatch counter is exhausted")
        }
    })?;
    Ok(SequenceDispatchGuard { gate })
}

impl<R> AdmittedSequenceResources<R>
where
    R: DeviceRuntime,
{
    fn validate_runtime(&self, context: &'static str) -> Result<(), VNextError> {
        let descriptor = self.request.plan.runtime().descriptor();
        descriptor.validate()?;
        if descriptor.id != *self.request.plan.device_id()
            || descriptor.runtime_implementation_fingerprint
                != self.request.plan.runtime_implementation_fingerprint()
        {
            return Err(invalid_resource(format!(
                "{context} runtime differs from the trusted plan/runtime binding"
            )));
        }
        Ok(())
    }

    pub fn create_execution_stream(
        self: &Arc<Self>,
    ) -> Result<BoundExecutionStream<R>, ExecutionStreamCreationError<R::Error>> {
        let _lifecycle = self
            .request
            .plan
            .resources
            .read_lifecycle("create an execution stream")
            .map_err(ExecutionStreamCreationError::Contract)?;
        if self.is_poisoned() {
            return Err(ExecutionStreamCreationError::Contract(invalid_resource(
                "poisoned logical sequence cannot create an execution stream",
            )));
        }
        self.validate_runtime("execution stream creation preflight")
            .map_err(ExecutionStreamCreationError::Contract)?;
        let stream = self
            .request
            .plan
            .runtime()
            .create_stream()
            .map_err(ExecutionStreamCreationError::Runtime)?;
        self.validate_runtime("execution stream creation completion")
            .map_err(ExecutionStreamCreationError::Contract)?;
        if self.request.plan.runtime().stream_state(&stream) != StreamState::Ready {
            return Err(ExecutionStreamCreationError::Contract(invalid_resource(
                "new execution stream is not ready",
            )));
        }
        Ok(BoundExecutionStream {
            runtime: Arc::clone(self.request.plan.runtime()),
            coordinator_id: self.coordinator_id(),
            sequence_authority: self.sequence_authority(),
            stream: Some(stream),
            state: BoundExecutionStreamState::Ready,
            sequence_recovery: Arc::clone(&self.sequence_recovery),
            sequence_dispatch_gate: Arc::clone(&self.sequence_dispatch_gate),
            abandoned_sequence: None,
            resources: Arc::clone(self),
        })
    }

    pub fn activate<'resources, 'exec>(
        &'resources self,
        stream: &'exec mut BoundExecutionStream<R>,
    ) -> Result<ActiveSequencePermit<'resources, 'exec, R>, VNextError> {
        let _lifecycle = self
            .request
            .plan
            .resources
            .read_lifecycle("activate an execution stream")?;
        if self.is_poisoned() {
            return Err(invalid_resource(
                "poisoned logical sequence cannot be activated",
            ));
        }
        self.validate_runtime("logical sequence activation")?;
        if !Arc::ptr_eq(self.request.plan.runtime(), &stream.runtime)
            || !std::ptr::eq(self, Arc::as_ref(&stream.resources))
            || stream.coordinator_id != self.coordinator_id()
            || stream.sequence_authority != self.sequence_authority()
            || !Arc::ptr_eq(&self.sequence_recovery, &stream.sequence_recovery)
            || !Arc::ptr_eq(&self.sequence_dispatch_gate, &stream.sequence_dispatch_gate)
        {
            return Err(invalid_resource(
                "execution stream belongs to another logical sequence authority",
            ));
        }
        if stream.state != BoundExecutionStreamState::Ready
            || stream.abandoned_sequence.is_some()
            || self.request.plan.runtime().stream_state(stream.stream()) != StreamState::Ready
        {
            return Err(invalid_resource(
                "logical sequence activation requires one core-ready stream",
            ));
        }
        let mut authority_source = self.lock_authority_source()?;
        let selecting_legacy = match *authority_source {
            SequenceExecutionAuthoritySource::Unselected => true,
            SequenceExecutionAuthoritySource::LegacyStream => false,
            SequenceExecutionAuthoritySource::SequenceSession => {
                return Err(invalid_resource(
                    "logical sequence execution authority is permanently selected for sequence sessions",
                ));
            }
            SequenceExecutionAuthoritySource::FailClosed => {
                return Err(invalid_resource(
                    "logical sequence execution authority selector is fail-closed",
                ));
            }
        };
        let epoch = match self.next_activation_epoch.fetch_update(
            Ordering::AcqRel,
            Ordering::Acquire,
            |epoch| epoch.checked_add(1).filter(|next| *next <= (u64::MAX >> 2)),
        ) {
            Ok(epoch) => epoch,
            Err(_) => {
                *authority_source = SequenceExecutionAuthoritySource::FailClosed;
                return Err(invalid_resource("active sequence epoch space is exhausted"));
            }
        };
        let active_state = sequence_slot_active(epoch);
        if let Err(actual) =
            self.state
                .compare_exchange(0, active_state, Ordering::AcqRel, Ordering::Acquire)
        {
            if selecting_legacy {
                *authority_source = SequenceExecutionAuthoritySource::FailClosed;
            }
            return Err(if sequence_slot_is_poisoned(actual) {
                invalid_resource("logical sequence was abandoned and is poisoned")
            } else {
                invalid_resource("logical sequence already owns an active stream")
            });
        }
        let slot = self.sequence_authority().sparse_id();
        let recovery_metadata = AbandonedSequenceMetadata {
            plan: self.request.plan.evidence(),
            sequence_authority: self.sequence_authority(),
            run_id: self.run_id().clone(),
            request_id: self.request_id().clone(),
            slot,
            activation_epoch: epoch,
            runtime_implementation_fingerprint: self
                .request
                .plan
                .runtime_implementation_fingerprint()
                .to_owned(),
            state: Arc::clone(&self.state),
            sequence_dispatch_gate: Arc::clone(&self.sequence_dispatch_gate),
            drained: false,
        };
        let recovery_key = recovery_metadata.key();
        self.sequence_recovery.register(recovery_metadata);
        stream.abandoned_sequence = Some(recovery_key);
        stream.state = BoundExecutionStreamState::InUse;
        *authority_source = SequenceExecutionAuthoritySource::LegacyStream;
        Ok(ActiveSequencePermit {
            resources: self,
            epoch,
            state: Arc::clone(&self.state),
            stream,
            runtime_fingerprint: self
                .request
                .plan
                .runtime_implementation_fingerprint()
                .to_owned(),
            stream_drained: false,
            completed: false,
        })
    }

    pub fn recover_abandoned_sequence(
        &self,
    ) -> Result<ActiveSequenceAbortReceipt, AbandonedSequenceRecoveryError<R::Error>> {
        self.sequence_recovery.recover(
            self.request.plan.runtime(),
            self.sequence_authority().sparse_id(),
        )
    }
}

/// Non-cloneable guard for an admitted active-sequence slot. Dispatch borrows
/// this permit; the sequence owner retains it until all asynchronous work is
/// synchronized or cancelled.
#[must_use = "an active sequence permit must live until asynchronous work is complete"]
pub struct ActiveSequencePermit<'resources, 'exec, R>
where
    R: DeviceRuntime,
{
    resources: &'resources AdmittedSequenceResources<R>,
    epoch: u64,
    state: Arc<AtomicU64>,
    stream: &'exec mut BoundExecutionStream<R>,
    runtime_fingerprint: String,
    stream_drained: bool,
    completed: bool,
}

impl<'resources, 'exec, R> ActiveSequencePermit<'resources, 'exec, R>
where
    R: DeviceRuntime,
{
    pub fn resources(&self) -> &'resources AdmittedSequenceResources<R> {
        self.resources
    }

    pub fn run_id(&self) -> &RunId {
        self.resources.run_id()
    }

    pub fn request_id(&self) -> &RequestIdentity {
        self.resources.request_id()
    }

    pub fn sequence_authority(&self) -> SequenceAuthorityId {
        self.resources.sequence_authority()
    }

    pub fn coordinator_id(&self) -> LogicalAdmissionCoordinatorId {
        self.resources.coordinator_id()
    }

    pub fn backing_slices(&self) -> &[LogicalBackingSliceAuthority] {
        self.resources.backing_slices()
    }

    pub const fn activation_epoch(&self) -> u64 {
        self.epoch
    }

    pub fn runtime_implementation_fingerprint(&self) -> &str {
        &self.runtime_fingerprint
    }

    pub(crate) fn with_runtime_and_stream<T>(
        &mut self,
        action: impl FnOnce(&R, &mut R::Stream) -> T,
    ) -> Result<T, VNextError> {
        if self.stream.state != BoundExecutionStreamState::InUse {
            return Err(invalid_resource(
                "operation dispatch requires one core-owned in-use stream",
            ));
        }
        let _dispatch_guard = enter_sequence_dispatch(&self.resources.sequence_dispatch_gate)?;
        Ok(action(
            self.resources.request.plan.runtime(),
            self.stream.stream_mut(),
        ))
    }

    /// Consumes dispatch authority before draining the exact bound stream.
    /// Successful synchronization returns a different typestate that cannot
    /// be passed back to `OperationDispatch`.
    pub fn synchronize(
        mut self,
    ) -> Result<
        SynchronizedSequencePermit<'resources, 'exec, R>,
        SequenceSynchronizationFailure<'resources, 'exec, R>,
    > {
        let preflight = self
            .resources
            .validate_runtime("sequence synchronization preflight")
            .and_then(|()| {
                if self
                    .resources
                    .request
                    .plan
                    .runtime()
                    .descriptor()
                    .runtime_implementation_fingerprint
                    == self.runtime_fingerprint
                {
                    Ok(())
                } else {
                    Err(invalid_resource(
                        "sequence synchronization runtime differs from its activation snapshot",
                    ))
                }
            });

        // Draining is attempted even when descriptor validation fails. The
        // stream/runtime pair is privately bound, while skipping the drain
        // could make later buffer quarantine unsafe.
        let runtime_error = match self
            .resources
            .request
            .plan
            .runtime()
            .synchronize(self.stream.stream_mut())
        {
            Ok(()) => None,
            Err(error) => Some(error),
        };
        let stream_ready = self
            .resources
            .request
            .plan
            .runtime()
            .stream_state(self.stream.stream())
            == StreamState::Ready;
        self.stream_drained = runtime_error.is_none() && stream_ready;
        if self.stream_drained {
            self.stream
                .sequence_recovery
                .set_drained((self.sequence_authority().sparse_id(), self.epoch), true);
        }
        let completion = self
            .resources
            .validate_runtime("sequence synchronization completion")
            .and_then(|()| {
                if stream_ready {
                    Ok(())
                } else {
                    Err(invalid_resource(
                        "sequence synchronization did not return the bound stream to ready",
                    ))
                }
            });
        let error = preflight
            .err()
            .map(SequenceSynchronizationError::Contract)
            .or_else(|| runtime_error.map(SequenceSynchronizationError::Runtime))
            .or_else(|| completion.err().map(SequenceSynchronizationError::Contract));
        if let Some(error) = error {
            return Err(SequenceSynchronizationFailure {
                permit: Some(self),
                error,
            });
        }
        self.stream.state = BoundExecutionStreamState::Ready;
        Ok(SynchronizedSequencePermit { permit: Some(self) })
    }
}

#[derive(Debug)]
pub enum SequenceSynchronizationError<E> {
    Contract(VNextError),
    Runtime(E),
}

/// Retry owner for a failed stream drain. It intentionally does not expose
/// the active dispatch permit, so no operation can be submitted between a
/// failed synchronization attempt and its retry.
#[must_use = "failed sequence synchronization must be retried or retained"]
pub struct SequenceSynchronizationFailure<'resources, 'exec, R>
where
    R: DeviceRuntime,
{
    permit: Option<ActiveSequencePermit<'resources, 'exec, R>>,
    error: SequenceSynchronizationError<R::Error>,
}

impl<R> fmt::Debug for SequenceSynchronizationFailure<'_, '_, R>
where
    R: DeviceRuntime,
{
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("SequenceSynchronizationFailure")
            .field("error", &self.error)
            .finish_non_exhaustive()
    }
}

impl<'resources, 'exec, R> SequenceSynchronizationFailure<'resources, 'exec, R>
where
    R: DeviceRuntime,
{
    pub fn error(&self) -> &SequenceSynchronizationError<R::Error> {
        &self.error
    }

    pub fn retry(
        mut self,
    ) -> Result<
        SynchronizedSequencePermit<'resources, 'exec, R>,
        SequenceSynchronizationFailure<'resources, 'exec, R>,
    > {
        self.permit
            .take()
            .expect("synchronization failure owns its active permit")
            .synchronize()
    }
}

/// Stream-drained typestate. It has no dispatch API and must choose exactly
/// one terminal slot disposition.
#[must_use = "a synchronized sequence must be completed or aborted"]
pub struct SynchronizedSequencePermit<'resources, 'exec, R>
where
    R: DeviceRuntime,
{
    permit: Option<ActiveSequencePermit<'resources, 'exec, R>>,
}

impl<R> SynchronizedSequencePermit<'_, '_, R>
where
    R: DeviceRuntime,
{
    pub fn complete(mut self) -> Result<ActiveSequenceCompletionReceipt, VNextError> {
        let mut permit = self
            .permit
            .take()
            .expect("synchronized sequence owns its active permit");
        let sequence_poisoned =
            sequence_dispatch_is_poisoned(&permit.resources.sequence_dispatch_gate);
        let terminal_state = if sequence_poisoned {
            sequence_slot_poisoned_drained(permit.epoch)
        } else {
            0
        };
        permit
            .state
            .compare_exchange(
                sequence_slot_active(permit.epoch),
                terminal_state,
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .map_err(|_| invalid_resource("active sequence epoch is no longer completable"))?;
        permit
            .stream
            .sequence_recovery
            .clear((permit.sequence_authority().sparse_id(), permit.epoch));
        permit.stream.abandoned_sequence = None;
        permit.stream.state = BoundExecutionStreamState::Ready;
        permit.completed = true;
        if sequence_poisoned {
            return Err(invalid_resource(
                "sequence cannot complete successfully after its dispatch authority was poisoned",
            ));
        }
        Ok(ActiveSequenceCompletionReceipt {
            plan: permit.resources.request.plan.evidence(),
            sequence_authority: permit.sequence_authority(),
            run_id: permit.run_id().clone(),
            request_id: permit.request_id().clone(),
            activation_epoch: permit.epoch,
            runtime_implementation_fingerprint: permit.runtime_fingerprint.clone(),
        })
    }

    /// Produces abort evidence only after the exact bound stream was drained.
    /// Only this exact logical sequence remains poisoned after abort.
    pub fn abort(mut self) -> Result<ActiveSequenceAbortReceipt, VNextError> {
        let mut permit = self
            .permit
            .take()
            .expect("synchronized sequence owns its active permit");
        permit
            .state
            .compare_exchange(
                sequence_slot_active(permit.epoch),
                sequence_slot_poisoned_drained(permit.epoch),
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .map_err(|_| invalid_resource("active sequence epoch is no longer abortable"))?;
        permit
            .resources
            .sequence_dispatch_gate
            .fetch_or(SEQUENCE_DISPATCH_POISONED_BIT, Ordering::AcqRel);
        permit
            .stream
            .sequence_recovery
            .clear((permit.sequence_authority().sparse_id(), permit.epoch));
        permit.stream.abandoned_sequence = None;
        permit.stream.state = BoundExecutionStreamState::Ready;
        permit.completed = true;
        Ok(ActiveSequenceAbortReceipt {
            plan: permit.resources.request.plan.evidence(),
            sequence_authority: permit.sequence_authority(),
            run_id: permit.run_id().clone(),
            request_id: permit.request_id().clone(),
            activation_epoch: permit.epoch,
            runtime_implementation_fingerprint: permit.runtime_fingerprint.clone(),
            disposition: ActiveSequenceAbortDisposition::SynchronizedAndPoisoned,
        })
    }
}

/// Core-signed evidence that synchronization succeeded and the exact active
/// slot epoch was atomically cleared. It is trusted output and deliberately
/// cannot be deserialized or constructed by a caller.
#[derive(Debug, Serialize)]
#[must_use = "sequence completion evidence must be recorded by execution"]
pub struct ActiveSequenceCompletionReceipt {
    plan: TrustedPlanRuntimeEvidence,
    sequence_authority: SequenceAuthorityId,
    run_id: RunId,
    request_id: RequestIdentity,
    activation_epoch: u64,
    runtime_implementation_fingerprint: String,
}

impl ActiveSequenceCompletionReceipt {
    pub fn plan(&self) -> &TrustedPlanRuntimeEvidence {
        &self.plan
    }

    pub fn run_id(&self) -> &RunId {
        &self.run_id
    }

    pub fn request_id(&self) -> &RequestIdentity {
        &self.request_id
    }

    pub const fn sequence_authority(&self) -> SequenceAuthorityId {
        self.sequence_authority
    }

    pub const fn activation_epoch(&self) -> u64 {
        self.activation_epoch
    }

    pub fn runtime_implementation_fingerprint(&self) -> &str {
        &self.runtime_implementation_fingerprint
    }
}

/// Terminal resource disposition produced by an explicit sequence abort.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum ActiveSequenceAbortDisposition {
    SynchronizedAndPoisoned,
    SequenceSessionTerminalized,
}

/// Core-signed evidence that the exact active slot epoch was atomically
/// poisoned. This type is trusted output and has no deserialization or public
/// construction path.
#[derive(Debug, Serialize)]
#[must_use = "sequence abort evidence must be recorded by execution"]
pub struct ActiveSequenceAbortReceipt {
    plan: TrustedPlanRuntimeEvidence,
    sequence_authority: SequenceAuthorityId,
    run_id: RunId,
    request_id: RequestIdentity,
    activation_epoch: u64,
    runtime_implementation_fingerprint: String,
    disposition: ActiveSequenceAbortDisposition,
}

impl ActiveSequenceAbortReceipt {
    pub fn plan(&self) -> &TrustedPlanRuntimeEvidence {
        &self.plan
    }

    pub fn run_id(&self) -> &RunId {
        &self.run_id
    }

    pub fn request_id(&self) -> &RequestIdentity {
        &self.request_id
    }

    pub const fn sequence_authority(&self) -> SequenceAuthorityId {
        self.sequence_authority
    }

    pub const fn activation_epoch(&self) -> u64 {
        self.activation_epoch
    }

    pub fn runtime_implementation_fingerprint(&self) -> &str {
        &self.runtime_implementation_fingerprint
    }

    pub const fn disposition(&self) -> ActiveSequenceAbortDisposition {
        self.disposition
    }
}

impl<R> Drop for ActiveSequencePermit<'_, '_, R>
where
    R: DeviceRuntime,
{
    fn drop(&mut self) {
        if !self.completed {
            let poisoned_state = if self.stream_drained {
                sequence_slot_poisoned_drained(self.epoch)
            } else {
                sequence_slot_poisoned_undrained(self.epoch)
            };
            let result = self.state.compare_exchange(
                sequence_slot_active(self.epoch),
                poisoned_state,
                Ordering::AcqRel,
                Ordering::Acquire,
            );
            debug_assert!(result.is_ok(), "active sequence slot guard lost ownership");
            if result.is_ok() {
                self.resources
                    .sequence_dispatch_gate
                    .fetch_or(SEQUENCE_DISPATCH_POISONED_BIT, Ordering::AcqRel);
                self.stream.state = BoundExecutionStreamState::Poisoned;
                self.stream.sequence_recovery.set_drained(
                    (self.sequence_authority().sparse_id(), self.epoch),
                    self.stream_drained,
                );
            }
        }
    }
}

mod sealed {
    pub trait Sealed {}
}

pub trait TransactionStage: sealed::Sealed {
    const STATE: ResourceTransactionState;
    const TERMINAL: bool;
}

pub struct TransactionNew;
pub struct TransactionReserved;
pub struct TransactionCommitted;
pub struct TransactionRolledBack;
pub struct TransactionReleased;
pub struct TransactionQuarantined;

impl sealed::Sealed for TransactionNew {}
impl sealed::Sealed for TransactionReserved {}
impl sealed::Sealed for TransactionCommitted {}
impl sealed::Sealed for TransactionRolledBack {}
impl sealed::Sealed for TransactionReleased {}
impl sealed::Sealed for TransactionQuarantined {}

impl TransactionStage for TransactionNew {
    const STATE: ResourceTransactionState = ResourceTransactionState::New;
    const TERMINAL: bool = false;
}
impl TransactionStage for TransactionReserved {
    const STATE: ResourceTransactionState = ResourceTransactionState::Reserved;
    const TERMINAL: bool = false;
}
impl TransactionStage for TransactionCommitted {
    const STATE: ResourceTransactionState = ResourceTransactionState::Committed;
    const TERMINAL: bool = false;
}
impl TransactionStage for TransactionRolledBack {
    const STATE: ResourceTransactionState = ResourceTransactionState::RolledBack;
    const TERMINAL: bool = true;
}
impl TransactionStage for TransactionReleased {
    const STATE: ResourceTransactionState = ResourceTransactionState::Released;
    const TERMINAL: bool = true;
}
impl TransactionStage for TransactionQuarantined {
    const STATE: ResourceTransactionState = ResourceTransactionState::Quarantined;
    const TERMINAL: bool = true;
}

struct PendingSubsetRelease {
    target_orders: Vec<usize>,
    failure: ResourceFailureReceipt,
}

#[must_use = "dropping a nonterminal resource transaction emits an abandon signal"]
pub struct ResourceTransaction<D: ResourceTransactionDriver, S: TransactionStage> {
    driver: Option<D>,
    maintenance_controller: Option<DynamicPoolMaintenanceController<D::Runtime>>,
    dynamic_pools: Option<Arc<DynamicPoolSet<D::Runtime>>>,
    identity: ResourceTransactionIdentity,
    admission: StaticProvisioningBinding,
    reservations: ResourceReservationBatch,
    capacity_claim: Option<DeviceCapacityClaim>,
    allocation_issued: Vec<AtomicBool>,
    pending_allocations: Vec<RefCell<Option<CoreOwnedAllocation<D::Buffer>>>>,
    states: Vec<ResourceTransactionState>,
    lease: Option<StaticProvisioningLease<D::Runtime>>,
    receipts: Vec<ResourceTransitionReceipt>,
    lease_receipts: Vec<ResourceLeaseTransitionReceipt>,
    recovery_history: Vec<ResourceFailureReceipt>,
    latest_transition_context: Option<ResourceTransitionValidationContext>,
    latest_lease_context: Option<ResourceLeaseValidationContext>,
    pending_failure: Option<ResourceFailureReceipt>,
    pending_subset_release: Option<PendingSubsetRelease>,
    next_failure_id: u64,
    finalized: bool,
    stage: PhantomData<S>,
}

impl<D: ResourceTransactionDriver, S: TransactionStage> ResourceTransaction<D, S> {
    pub fn maintenance_controller(&self) -> &DynamicPoolMaintenanceController<D::Runtime> {
        self.maintenance_controller
            .as_ref()
            .expect("live resource transaction owns dynamic-pool maintenance authority")
    }

    pub fn identity(&self) -> &ResourceTransactionIdentity {
        &self.identity
    }

    pub fn admission(&self) -> &StaticProvisioningBinding {
        &self.admission
    }

    pub fn reservations(&self) -> &ResourceReservationBatch {
        &self.reservations
    }

    pub fn receipts(&self) -> &[ResourceTransitionReceipt] {
        &self.receipts
    }

    pub fn lease_receipts(&self) -> &[ResourceLeaseTransitionReceipt] {
        &self.lease_receipts
    }

    pub fn recovery_history(&self) -> &[ResourceFailureReceipt] {
        &self.recovery_history
    }

    pub const fn state(&self) -> ResourceTransactionState {
        S::STATE
    }

    pub fn actual_states(&self) -> &[ResourceTransactionState] {
        &self.states
    }

    pub fn ledger_snapshot(&self) -> ResourceLedgerSnapshot {
        ResourceLedgerSnapshot {
            identity: self.identity.clone(),
            admission: self.admission.clone(),
            entries: self.ledger_snapshot_entries(),
        }
    }

    pub fn latest_transition_validation_context(
        &self,
    ) -> Option<&ResourceTransitionValidationContext> {
        self.latest_transition_context.as_ref()
    }

    pub fn latest_lease_validation_context(&self) -> Option<&ResourceLeaseValidationContext> {
        self.latest_lease_context.as_ref()
    }

    fn ledger_snapshot_entries(&self) -> Vec<ResourceLedgerEntrySnapshot> {
        let lease = self
            .lease
            .as_ref()
            .expect("every resource transaction owns its batch lease ledger");
        lease
            .slots
            .iter()
            .zip(&self.pending_allocations)
            .zip(&self.states)
            .map(|((slot, pending), &transaction_state)| {
                let pending = pending.borrow();
                ResourceLedgerEntrySnapshot {
                    entry: slot.entry.clone(),
                    transaction_state,
                    buffer_present: slot.buffer.is_some() || pending.is_some(),
                    actual_resource_id: slot
                        .actual_resource_id
                        .clone()
                        .or_else(|| pending.as_ref().map(|value| value.resource_id.clone())),
                    actual_generation: slot
                        .actual_generation
                        .or_else(|| pending.as_ref().map(|value| value.generation)),
                    actual_descriptor: slot
                        .descriptor
                        .clone()
                        .or_else(|| pending.as_ref().map(|value| value.descriptor.clone())),
                }
            })
            .collect()
    }

    fn driver_and_context(
        &mut self,
        order: usize,
        action: ResourceTransactionAction,
    ) -> (&mut D, ResourceTransactionContext<'_, D::Runtime>) {
        let before = self.states[order];
        let driver = self
            .driver
            .as_mut()
            .expect("live transaction owns its resource driver");
        let context = ResourceTransactionContext {
            runtime: &self
                .lease
                .as_ref()
                .expect("transaction owns lease ledger")
                .runtime,
            identity: &self.identity,
            binding: &self.admission,
            reservations: &self.reservations,
            cursor: Some(ResourceActionCursor {
                order,
                action,
                before,
                allocation_authorized: action == ResourceTransactionAction::Commit,
            }),
            allocation_authority: (action == ResourceTransactionAction::Commit)
                .then_some(&self.allocation_issued[order]),
            pending_allocation: (action == ResourceTransactionAction::Commit)
                .then_some(&self.pending_allocations[order]),
        };
        (driver, context)
    }

    fn make_record(
        &self,
        order: usize,
        action: ResourceTransactionAction,
        before: ResourceTransactionState,
        after: ResourceTransactionState,
    ) -> ResourceTransitionRecord {
        ResourceTransitionRecord::from_reservation(
            &self.identity,
            &self.admission,
            &self.reservations.reservations[order],
            action,
            before,
            after,
            order,
        )
    }

    fn transition_receipt(
        &mut self,
        action: ResourceTransactionAction,
        before: Vec<ResourceLedgerEntrySnapshot>,
        records: Vec<ResourceTransitionRecord>,
    ) -> Result<ResourceTransitionReceipt, VNextError> {
        let context = ResourceTransitionValidationContext {
            identity: self.identity.clone(),
            admission: self.admission.clone(),
            action,
            before,
            after: self.ledger_snapshot_entries(),
        };
        let receipt = ResourceTransitionReceipt::from_context(&context, records)?;
        self.latest_transition_context = Some(context);
        Ok(receipt)
    }

    fn lease_transition(
        &mut self,
        resource_ids: &[ResourceId],
        action: ResourceLeaseAction,
    ) -> Result<ResourceLeaseTransitionReceipt, VNextError> {
        if self.pending_subset_release.is_some() {
            return Err(invalid_resource(
                "lease policy cannot change during pending release recovery",
            ));
        }
        let orders = self.orders_for_ids(resource_ids)?;
        if orders
            .iter()
            .any(|&order| self.states[order] != ResourceTransactionState::Committed)
        {
            return Err(invalid_resource(
                "lease policy targets a resource that is not actually committed",
            ));
        }
        let before_ledger = self.ledger_snapshot_entries();
        let (before, after, entries) = self
            .lease
            .as_mut()
            .expect("transaction owns lease ledger")
            .transition_subset(&orders, action)?;
        let context = ResourceLeaseValidationContext {
            identity: self.identity.clone(),
            admission: self.admission.clone(),
            action,
            before: before_ledger,
            after: self.ledger_snapshot_entries(),
        };
        let receipt =
            ResourceLeaseTransitionReceipt::from_context(&context, before, after, entries)?;
        self.latest_lease_context = Some(context);
        self.lease_receipts.push(receipt.clone());
        Ok(receipt)
    }

    fn orders_for_ids(&self, resource_ids: &[ResourceId]) -> Result<Vec<usize>, VNextError> {
        if resource_ids.is_empty() {
            return Err(invalid_resource("resource subset must not be empty"));
        }
        let mut unique = BTreeSet::new();
        let by_id = self
            .reservations
            .reservations()
            .iter()
            .enumerate()
            .map(|(order, reservation)| (reservation.resource_id(), order))
            .collect::<BTreeMap<_, _>>();
        let mut orders = Vec::with_capacity(resource_ids.len());
        for resource_id in resource_ids {
            if !unique.insert(resource_id) {
                return Err(invalid_resource("resource subset contains duplicates"));
            }
            orders.push(
                *by_id
                    .get(resource_id)
                    .ok_or_else(|| invalid_resource("resource subset is not admitted"))?,
            );
        }
        orders.sort_unstable();
        Ok(orders)
    }

    fn set_pending_failure(&mut self, failure: &ResourceFailureReceipt) {
        self.pending_failure = Some(failure.clone());
    }

    fn clear_pending_failure(&mut self) {
        self.pending_failure = None;
    }

    fn issue_failure_id(&mut self) -> ResourceFailureId {
        let current = self.next_failure_id;
        self.next_failure_id = current
            .checked_add(1)
            .expect("resource transaction failure id space is exhausted");
        ResourceFailureId::try_from(current).expect("transaction failure ids start at one")
    }

    fn advance<T: TransactionStage>(
        mut self,
        receipt: Option<ResourceTransitionReceipt>,
    ) -> ResourceTransaction<D, T> {
        self.finalized = true;
        if let Some(receipt) = receipt {
            self.receipts.push(receipt);
        }
        ResourceTransaction {
            driver: self.driver.take(),
            maintenance_controller: self.maintenance_controller.take(),
            dynamic_pools: self.dynamic_pools.take(),
            identity: self.identity.clone(),
            admission: self.admission.clone(),
            reservations: self.reservations.clone(),
            capacity_claim: if T::TERMINAL {
                if let Some(mut claim) = self.capacity_claim.take() {
                    claim.release();
                }
                None
            } else {
                self.capacity_claim.take()
            },
            allocation_issued: std::mem::take(&mut self.allocation_issued),
            pending_allocations: std::mem::take(&mut self.pending_allocations),
            states: std::mem::take(&mut self.states),
            lease: self.lease.take(),
            receipts: std::mem::take(&mut self.receipts),
            lease_receipts: std::mem::take(&mut self.lease_receipts),
            recovery_history: std::mem::take(&mut self.recovery_history),
            latest_transition_context: self.latest_transition_context.take(),
            latest_lease_context: self.latest_lease_context.take(),
            pending_failure: None,
            pending_subset_release: None,
            next_failure_id: self.next_failure_id,
            finalized: T::TERMINAL,
            stage: PhantomData,
        }
    }

    fn abandon_signal(&self) -> ResourceAbandonSignal {
        ResourceAbandonSignal {
            identity: self.identity.clone(),
            admission: self.admission.clone(),
            state: S::STATE,
            pending_action: self
                .pending_failure
                .as_ref()
                .map(ResourceFailureReceipt::action),
            ledger: self.ledger_snapshot_entries(),
            active_sequence_slots: Vec::new(),
            poisoned_sequence_slots: Vec::new(),
            undrained_sequence_slots: Vec::new(),
            failure: self
                .pending_failure
                .as_ref()
                .map(|receipt| receipt.failure.clone()),
        }
    }

    fn take_all_owned_buffers(&mut self) -> Vec<ResourceOwnedBuffer<D::Buffer>> {
        let mut buffers = self
            .lease
            .as_mut()
            .expect("transaction owns lease ledger")
            .take_owned_buffers(&self.reservations);
        for (order, pending) in self.pending_allocations.iter_mut().enumerate() {
            let Some(allocation) = pending.get_mut().take() else {
                continue;
            };
            let reservation = &self.reservations.reservations[order];
            buffers.push(ResourceOwnedBuffer {
                order,
                expected_resource_id: reservation.resource_id.clone(),
                actual_resource_id: allocation.resource_id,
                expected_generation: reservation.generation,
                actual_generation: allocation.generation,
                expected_descriptor: BufferDescriptor {
                    resource_id: reservation.resource_id.clone(),
                    size_bytes: reservation.size_bytes,
                    alignment_bytes: reservation.alignment_bytes,
                    usage: reservation.usage,
                    element_type: reservation.element_type,
                },
                actual_descriptor: allocation.descriptor,
                buffer: allocation.buffer,
            });
        }
        buffers.sort_by_key(|buffer| buffer.order);
        buffers
    }

    fn quarantine_live(&mut self) -> Result<Vec<ResourceTransitionRecord>, ResourceDriverFailure> {
        let buffers = self.take_all_owned_buffers();
        let ownership = ResourcePoolOwnership {
            runtime: Arc::clone(
                &self
                    .lease
                    .as_ref()
                    .expect("transaction owns lease ledger")
                    .runtime,
            ),
            pool_identity: self.admission.pool_identity.clone(),
            reason: ResourceOwnershipReason::Quarantine,
            signal: None,
            buffers,
            capacity_claim: self.capacity_claim.take(),
        };
        let result = {
            let driver = self
                .driver
                .as_mut()
                .expect("live transaction owns its resource driver");
            let context = ResourceTransactionContext {
                runtime: &self
                    .lease
                    .as_ref()
                    .expect("transaction owns lease ledger")
                    .runtime,
                identity: &self.identity,
                binding: &self.admission,
                reservations: &self.reservations,
                cursor: None,
                allocation_authority: None,
                pending_allocation: None,
            };
            driver.quarantine_transaction(&context, ownership)
        };
        if let Err(failure) = result {
            let (failure, mut ownership) = failure.into_parts();
            let expected_claimed_bytes = self
                .states
                .iter()
                .zip(self.reservations.reservations())
                .filter(|(state, _)| state.is_live())
                .map(|(_, reservation)| reservation.size_bytes())
                .sum::<u64>();
            if ownership.pool_identity != self.admission.pool_identity
                || ownership.claimed_bytes() != expected_claimed_bytes
            {
                std::mem::forget(ownership);
                return Err(ResourceDriverFailure::new(core_resource_failure(
                    "ownership_transfer_identity_mismatch",
                    "quarantine failure returned ownership for a different resource pool",
                    false,
                ))
                .expect("core failure has resource domain"));
            }
            self.lease
                .as_mut()
                .expect("transaction owns lease ledger")
                .restore_owned_buffers(std::mem::take(&mut ownership.buffers));
            self.capacity_claim = ownership.capacity_claim.take();
            return Err(failure);
        }

        let mut records = Vec::new();
        for order in 0..self.states.len() {
            let before = self.states[order];
            if before.is_live() {
                self.states[order] = ResourceTransactionState::Quarantined;
                records.push(self.make_record(
                    order,
                    ResourceTransactionAction::Quarantine,
                    before,
                    ResourceTransactionState::Quarantined,
                ));
            }
        }
        Ok(records)
    }
}

impl<D: ResourceTransactionDriver, S: TransactionStage> Drop for ResourceTransaction<D, S> {
    fn drop(&mut self) {
        if self.finalized || S::TERMINAL {
            return;
        }
        let signal = self.abandon_signal();
        drop(self.maintenance_controller.take());
        drop(self.dynamic_pools.take());
        let buffers = self.take_all_owned_buffers();
        let ownership = ResourcePoolOwnership {
            runtime: Arc::clone(
                &self
                    .lease
                    .as_ref()
                    .expect("transaction owns lease ledger")
                    .runtime,
            ),
            pool_identity: self.admission.pool_identity.clone(),
            reason: ResourceOwnershipReason::Abandon,
            signal: Some(signal),
            buffers,
            capacity_claim: self.capacity_claim.take(),
        };
        if let Some(driver) = self.driver.as_mut() {
            // Driver cleanup is an observation/transfer hook, not permission
            // to turn an existing unwind into a process abort. Ownership that
            // drops inside a panicking callback retains its backend lifetimes.
            let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                driver.abandon_transaction(ownership);
            }));
        } else {
            // A live transaction always owns its driver. Leaking is safer than
            // returning capacity while backend ownership is unknown.
            std::mem::forget(ownership);
        }
        self.finalized = true;
    }
}

fn core_resource_failure(
    code: &'static str,
    message: impl Into<String>,
    retryable: bool,
) -> FailureEnvelope {
    FailureEnvelope::new(FailureDomain::Resource, code, message, retryable)
        .expect("core-generated resource failure must be valid")
}

impl<D: ResourceTransactionDriver> ResourceTransaction<D, TransactionNew> {
    pub fn begin(
        driver: D,
        identity: ResourceTransactionIdentity,
        permit: StaticProvisioningPermit<D::Runtime>,
    ) -> Result<Self, VNextError> {
        if identity.pool_id != permit.binding.pool_id()
            || identity.request_id != permit.binding.request_id
            || permit.binding.pool_identity.plan_id != permit.binding.plan_id
            || permit.binding.pool_identity.plan_hash != permit.binding.plan_hash
            || permit.binding.pool_identity.device_id != permit.binding.device_id
            || permit
                .binding
                .pool_identity
                .device_runtime_implementation_fingerprint
                != permit.binding.device_runtime_implementation_fingerprint
            || permit.binding.pool_identity.admission_generation
                != permit.binding.admission_generation
            || permit.binding.request_id != *permit.reservations.request_id()
            || permit.binding.admitted_bytes != permit.reservations.total_size_bytes()
            || permit.binding.plan_static_bytes != permit.reservations.plan_static_size_bytes()
            || permit.binding.admitted_bytes != permit.binding.plan_static_bytes
            || permit.binding.plan_static_bytes > permit.binding.usable_capacity_bytes
            || permit.binding.usable_capacity_bytes > permit.binding.device_capacity_bytes
            || driver.device_id() != &permit.binding.device_id
            || driver.device_runtime_implementation_fingerprint()
                != permit.binding.device_runtime_implementation_fingerprint
            || driver.device_capacity_bytes() != permit.binding.device_capacity_bytes
            || !Arc::ptr_eq(driver.runtime(), &permit.runtime)
        {
            return Err(invalid_resource(
                "transaction identity, driver device, or admitted capacity does not match its permit",
            ));
        }
        let _ = permit.seal;
        if permit.capacity_claim.bytes() != permit.binding.admitted_bytes {
            return Err(invalid_resource(
                "device capacity claim differs from admitted bytes",
            ));
        }
        let states = vec![ResourceTransactionState::New; permit.reservations.reservations.len()];
        let allocation_issued = (0..states.len()).map(|_| AtomicBool::new(false)).collect();
        let pending_allocations = (0..states.len()).map(|_| RefCell::new(None)).collect();
        let lease = StaticProvisioningLease::new(
            Arc::clone(&permit.runtime),
            &identity,
            &permit.binding,
            &permit.reservations,
        );
        Ok(Self {
            driver: Some(driver),
            maintenance_controller: Some(permit.maintenance_controller),
            dynamic_pools: Some(permit.dynamic_pools),
            identity,
            admission: permit.binding,
            reservations: permit.reservations,
            capacity_claim: Some(permit.capacity_claim),
            allocation_issued,
            pending_allocations,
            states,
            lease: Some(lease),
            receipts: Vec::new(),
            lease_receipts: Vec::new(),
            recovery_history: Vec::new(),
            latest_transition_context: None,
            latest_lease_context: None,
            pending_failure: None,
            pending_subset_release: None,
            next_failure_id: 1,
            finalized: false,
            stage: PhantomData,
        })
    }

    pub fn reserve(
        mut self,
    ) -> Result<
        ResourceTransaction<D, TransactionReserved>,
        ResourcePrepareTransitionError<D, TransactionNew>,
    > {
        debug_assert!(
            self.states
                .iter()
                .all(|state| *state == ResourceTransactionState::New),
            "new transaction ledger must be uniformly new"
        );
        let ledger_before = self.ledger_snapshot_entries();
        let mut completed = Vec::new();
        for order in 0..self.states.len() {
            let result = {
                let reservation = self.reservations.reservations[order].clone();
                let (driver, context) =
                    self.driver_and_context(order, ResourceTransactionAction::Reserve);
                driver.reserve_resource(&context, &reservation)
            };
            match result {
                Ok(()) => {
                    let before = self.states[order];
                    let after = before
                        .transition(
                            &self.reservations.reservations[order].resource_id,
                            ResourceTransactionAction::Reserve,
                        )
                        .expect("reserve was preflight validated");
                    self.states[order] = after;
                    completed.push(self.make_record(
                        order,
                        ResourceTransactionAction::Reserve,
                        before,
                        after,
                    ));
                }
                Err(failure) => {
                    let failure_id = self.issue_failure_id();
                    let receipt = ResourceFailureReceipt::new(
                        failure_id,
                        &self.identity,
                        &self.admission,
                        ResourceTransactionAction::Reserve,
                        failure.into_failure(),
                        Some(ResourceFailurePoint::new(
                            &self.reservations.reservations[order],
                            order,
                            self.states[order],
                        )),
                        completed,
                        ResourceRecoveryStrategy::ReverseCompensation,
                        ledger_before,
                        self.ledger_snapshot_entries(),
                    );
                    self.set_pending_failure(&receipt);
                    return Err(ResourcePrepareTransitionError {
                        transaction: Some(self),
                        failure: receipt,
                    });
                }
            }
        }
        let receipt = self
            .transition_receipt(ResourceTransactionAction::Reserve, ledger_before, completed)
            .expect("core reserve journal must validate");
        Ok(self.advance(Some(receipt)))
    }
}

#[derive(Debug)]
pub enum ResourceCommitTransitionError<D: ResourceTransactionDriver> {
    Recoverable(ResourcePrepareTransitionError<D, TransactionReserved>),
    Poisoned(ResourcePoisonedTransaction<D>),
}

impl<D: ResourceTransactionDriver> ResourceTransaction<D, TransactionReserved> {
    pub fn commit(
        mut self,
    ) -> Result<ResourceTransaction<D, TransactionCommitted>, ResourceCommitTransitionError<D>>
    {
        debug_assert!(
            self.states
                .iter()
                .all(|state| *state == ResourceTransactionState::Reserved),
            "reserved transaction must be uniformly reserved before commit"
        );
        let ledger_before = self.ledger_snapshot_entries();
        let mut completed = Vec::new();
        for order in 0..self.states.len() {
            let driver_result = {
                let reservation = self.reservations.reservations[order].clone();
                let (driver, context) =
                    self.driver_and_context(order, ResourceTransactionAction::Commit);
                driver
                    .commit_resource(&context, &reservation)
                    .map(|receipt| DriverCommitAcknowledgement::from_receipt(&receipt))
            };
            let allocation = self.pending_allocations[order].get_mut().take();
            match (driver_result, allocation) {
                (Ok(acknowledgement), Some(allocation))
                    if acknowledgement.matches(&allocation)
                        && allocation.matches(&self.reservations.reservations[order]) =>
                {
                    self.lease
                        .as_mut()
                        .expect("transaction owns lease ledger")
                        .install(order, allocation);
                    let before = self.states[order];
                    let after = before
                        .transition(
                            &self.reservations.reservations[order].resource_id,
                            ResourceTransactionAction::Commit,
                        )
                        .expect("commit was preflight validated");
                    self.states[order] = after;
                    completed.push(self.make_record(
                        order,
                        ResourceTransactionAction::Commit,
                        before,
                        after,
                    ));
                }
                (driver_result, Some(poisoned)) => {
                    let actual_resource_id = poisoned.resource_id.clone();
                    let actual_generation = poisoned.generation;
                    let failure_envelope = match driver_result {
                        Ok(_) => core_resource_failure(
                            "invalid_commit_outcome",
                            format!(
                                "core-owned allocation `{}` generation {} does not match allocation `{}` generation {}",
                                actual_resource_id,
                                actual_generation,
                                self.reservations.reservations[order].resource_id(),
                                self.reservations.reservations[order].generation(),
                            ),
                            false,
                        ),
                        Err(failure) => failure.into_failure(),
                    };
                    self.lease
                        .as_mut()
                        .expect("transaction owns lease ledger")
                        .install(order, poisoned);
                    let failure_id = self.issue_failure_id();
                    let failure = ResourceFailureReceipt::new(
                        failure_id,
                        &self.identity,
                        &self.admission,
                        ResourceTransactionAction::Commit,
                        failure_envelope,
                        Some(ResourceFailurePoint::new(
                            &self.reservations.reservations[order],
                            order,
                            self.states[order],
                        )),
                        completed,
                        ResourceRecoveryStrategy::ReconcileOrQuarantine,
                        ledger_before,
                        self.ledger_snapshot_entries(),
                    );
                    self.set_pending_failure(&failure);
                    return Err(ResourceCommitTransitionError::Poisoned(
                        ResourcePoisonedTransaction {
                            transaction: Some(self),
                            failure,
                            poisoned_order: order,
                        },
                    ));
                }
                (driver_result, None) => {
                    let failure = match driver_result {
                        Ok(_) => core_resource_failure(
                            "commit_acknowledged_without_allocation",
                            "driver acknowledged commit without a core-owned allocation",
                            false,
                        ),
                        Err(failure) => failure.into_failure(),
                    };
                    let failure_id = self.issue_failure_id();
                    let receipt = ResourceFailureReceipt::new(
                        failure_id,
                        &self.identity,
                        &self.admission,
                        ResourceTransactionAction::Commit,
                        failure,
                        Some(ResourceFailurePoint::new(
                            &self.reservations.reservations[order],
                            order,
                            self.states[order],
                        )),
                        completed,
                        ResourceRecoveryStrategy::ReverseCompensation,
                        ledger_before,
                        self.ledger_snapshot_entries(),
                    );
                    self.set_pending_failure(&receipt);
                    return Err(ResourceCommitTransitionError::Recoverable(
                        ResourcePrepareTransitionError {
                            transaction: Some(self),
                            failure: receipt,
                        },
                    ));
                }
            }
        }
        let receipt = self
            .transition_receipt(ResourceTransactionAction::Commit, ledger_before, completed)
            .expect("core commit journal must validate");
        Ok(self.advance(Some(receipt)))
    }

    pub fn rollback(
        mut self,
    ) -> Result<ResourceTransaction<D, TransactionRolledBack>, ResourceRollbackTransitionError<D>>
    {
        debug_assert!(
            self.states
                .iter()
                .all(|state| *state == ResourceTransactionState::Reserved),
            "rollback starts from a uniformly reserved transaction"
        );
        let ledger_before = self.ledger_snapshot_entries();
        let mut completed = Vec::new();
        for order in 0..self.states.len() {
            let result = {
                let reservation = self.reservations.reservations[order].clone();
                let (driver, context) =
                    self.driver_and_context(order, ResourceTransactionAction::Rollback);
                driver.rollback_resource(&context, &reservation)
            };
            match result {
                Ok(()) => {
                    let before = self.states[order];
                    self.states[order] = ResourceTransactionState::RolledBack;
                    completed.push(self.make_record(
                        order,
                        ResourceTransactionAction::Rollback,
                        before,
                        ResourceTransactionState::RolledBack,
                    ));
                }
                Err(failure) => {
                    let failure_id = self.issue_failure_id();
                    let receipt = ResourceFailureReceipt::new(
                        failure_id,
                        &self.identity,
                        &self.admission,
                        ResourceTransactionAction::Rollback,
                        failure.into_failure(),
                        Some(ResourceFailurePoint::new(
                            &self.reservations.reservations[order],
                            order,
                            self.states[order],
                        )),
                        completed,
                        ResourceRecoveryStrategy::ForwardCompletion,
                        ledger_before,
                        self.ledger_snapshot_entries(),
                    );
                    self.set_pending_failure(&receipt);
                    return Err(ResourceRollbackTransitionError {
                        transaction: Some(self),
                        failure: receipt,
                    });
                }
            }
        }
        let receipt = self
            .transition_receipt(
                ResourceTransactionAction::Rollback,
                ledger_before,
                completed,
            )
            .expect("core rollback journal must validate");
        Ok(self.advance(Some(receipt)))
    }
}

impl<D: ResourceTransactionDriver> ResourceTransaction<D, TransactionCommitted> {
    pub fn into_plan_runtime(
        mut self,
    ) -> Result<Arc<PlanRuntimeResources<D::Runtime>>, PlanRuntimeHandoffError<D>>
    where
        D: 'static,
    {
        let valid = self
            .states
            .iter()
            .all(|state| *state == ResourceTransactionState::Committed)
            && self.pending_failure.is_none()
            && self.pending_subset_release.is_none()
            && self
                .pending_allocations
                .iter()
                .all(|allocation| allocation.borrow().is_none())
            && self.lease.as_ref().is_some_and(|lease| {
                !lease.slots.is_empty()
                    && lease.slots.iter().all(|slot| {
                        slot.buffer.is_some()
                            && slot.descriptor.is_some()
                            && slot.entry.state == ResourceLeaseState::Active
                    })
            })
            && self.driver.is_some()
            && self.maintenance_controller.is_some()
            && self.dynamic_pools.is_some()
            && self.capacity_claim.is_some();
        if !valid {
            return Err(PlanRuntimeHandoffError {
                error: invalid_resource(
                    "plan runtime handoff requires one complete active committed transaction",
                ),
                transaction: Some(self),
            });
        }
        let lease = self
            .lease
            .take()
            .expect("validated plan runtime handoff owns its static lease");
        let runtime = Arc::clone(&lease.runtime);
        let dynamic_pools = self
            .dynamic_pools
            .take()
            .expect("validated plan runtime handoff owns its dynamic pools");
        let driver: Box<dyn ErasedPlanStaticDriver<D::Runtime>> = Box::new(
            self.driver
                .take()
                .expect("validated plan runtime handoff owns its driver"),
        );
        let static_resources = PlanStaticResources {
            driver: Mutex::new(Some(driver)),
            identity: self.identity.clone(),
            admission: self.admission.clone(),
            reservations: self.reservations.clone(),
            states: std::mem::take(&mut self.states),
            capacity_claim: self.capacity_claim.take(),
            lease: Some(lease),
            finalized: false,
        };
        let maintenance_controller = self
            .maintenance_controller
            .take()
            .expect("validated plan runtime handoff owns maintenance authority");
        let (lifecycle_tx, _) = watch::channel(PLAN_RUNTIME_OPEN);
        self.finalized = true;
        Ok(Arc::new(PlanRuntimeResources {
            lifecycle: RwLock::new(()),
            phase: AtomicU8::new(PLAN_RUNTIME_OPEN),
            lifecycle_tx,
            maintenance_controller,
            dynamic_pools,
            static_resources: PlanRuntimeStatic::Static(static_resources),
            runtime,
            deferred_cleanup_domain: new_deferred_device_cleanup_domain(),
        }))
    }

    pub fn lease(&self) -> &StaticProvisioningLease<D::Runtime> {
        self.lease
            .as_ref()
            .expect("committed transaction owns its batch lease")
    }

    fn defer_lease(
        &mut self,
        resource_ids: &[ResourceId],
    ) -> Result<ResourceLeaseTransitionReceipt, VNextError> {
        self.lease_transition(resource_ids, ResourceLeaseAction::Defer)
    }

    fn resume_lease(
        &mut self,
        resource_ids: &[ResourceId],
    ) -> Result<ResourceLeaseTransitionReceipt, VNextError> {
        self.lease_transition(resource_ids, ResourceLeaseAction::Resume)
    }

    fn cancel_lease(
        &mut self,
        resource_ids: &[ResourceId],
    ) -> Result<ResourceLeaseTransitionReceipt, VNextError> {
        self.lease_transition(resource_ids, ResourceLeaseAction::Cancel)
    }

    pub fn defer_all(&mut self) -> Result<ResourceLeaseTransitionReceipt, VNextError> {
        let ids = self
            .reservations
            .resource_ids()
            .cloned()
            .collect::<Vec<_>>();
        self.defer_lease(&ids)
    }

    pub fn resume_all(&mut self) -> Result<ResourceLeaseTransitionReceipt, VNextError> {
        let ids = self
            .reservations
            .resource_ids()
            .cloned()
            .collect::<Vec<_>>();
        self.resume_lease(&ids)
    }

    pub fn cancel_all(&mut self) -> Result<ResourceLeaseTransitionReceipt, VNextError> {
        let ids = self
            .reservations
            .resource_ids()
            .cloned()
            .collect::<Vec<_>>();
        self.cancel_lease(&ids)
    }

    /// Terminal release always targets the complete pool. On partial backend
    /// failure, the actual prefix remains released in the ledger and
    /// `complete_pending_release` resumes the outstanding suffix.
    fn release_all_resources(
        &mut self,
    ) -> Result<ResourceTransitionReceipt, ResourceFailureReceipt> {
        if let Some(pending) = &self.pending_subset_release {
            return Err(pending.failure.clone());
        }
        let orders = (0..self.states.len()).collect::<Vec<_>>();
        if let Some(&order) = orders
            .iter()
            .find(|&&order| self.states[order] != ResourceTransactionState::Committed)
        {
            return Err(self.local_release_failure(format!(
                "resource `{}` is not actually committed",
                self.reservations.reservations[order].resource_id()
            )));
        }
        let ledger_before = self.ledger_snapshot_entries();
        let mut completed = Vec::new();
        for &order in &orders {
            let result = {
                let reservation = self.reservations.reservations[order].clone();
                let buffer = self
                    .lease
                    .as_ref()
                    .expect("committed transaction owns lease")
                    .buffer(order)
                    .expect("committed ledger entry owns its buffer");
                let driver = self.driver.as_mut().expect("live transaction owns driver");
                let context = ResourceTransactionContext {
                    runtime: &self.lease.as_ref().expect("transaction owns lease").runtime,
                    identity: &self.identity,
                    binding: &self.admission,
                    reservations: &self.reservations,
                    cursor: Some(ResourceActionCursor {
                        order,
                        action: ResourceTransactionAction::Release,
                        before: self.states[order],
                        allocation_authorized: false,
                    }),
                    allocation_authority: None,
                    pending_allocation: None,
                };
                driver.release_resource(&context, &reservation, buffer)
            };
            match result {
                Ok(()) => {
                    let before = self.states[order];
                    self.lease
                        .as_mut()
                        .expect("transaction owns lease ledger")
                        .clear(order);
                    self.states[order] = ResourceTransactionState::Released;
                    self.capacity_claim
                        .as_mut()
                        .expect("live transaction owns its static capacity claim")
                        .release_bytes(self.reservations.reservations[order].size_bytes());
                    completed.push(self.make_record(
                        order,
                        ResourceTransactionAction::Release,
                        before,
                        ResourceTransactionState::Released,
                    ));
                }
                Err(failure) => {
                    let failure_id = self.issue_failure_id();
                    let receipt = ResourceFailureReceipt::new(
                        failure_id,
                        &self.identity,
                        &self.admission,
                        ResourceTransactionAction::Release,
                        failure.into_failure(),
                        Some(ResourceFailurePoint::new(
                            &self.reservations.reservations[order],
                            order,
                            self.states[order],
                        )),
                        completed,
                        ResourceRecoveryStrategy::ForwardCompletion,
                        ledger_before,
                        self.ledger_snapshot_entries(),
                    );
                    self.set_pending_failure(&receipt);
                    self.pending_subset_release = Some(PendingSubsetRelease {
                        target_orders: orders,
                        failure: receipt.clone(),
                    });
                    return Err(receipt);
                }
            }
        }
        let receipt = self
            .transition_receipt(ResourceTransactionAction::Release, ledger_before, completed)
            .expect("core subset release journal must validate");
        self.receipts.push(receipt.clone());
        Ok(receipt)
    }

    pub fn complete_pending_release(
        &mut self,
    ) -> Result<ResourceTransitionReceipt, ResourceFailureReceipt> {
        let Some(mut pending) = self.pending_subset_release.take() else {
            return Err(self.local_release_failure(
                "there is no pending subset release to complete".to_owned(),
            ));
        };
        for &order in &pending.target_orders {
            match self.states[order] {
                ResourceTransactionState::Released => continue,
                ResourceTransactionState::Committed => {}
                actual => {
                    let failure = ResourceDriverFailure::new(core_resource_failure(
                        "release_ledger_diverged",
                        format!(
                            "resource `{}` has unexpected actual state {}",
                            self.reservations.reservations[order].resource_id(),
                            actual.as_str()
                        ),
                        false,
                    ))
                    .expect("core failure has resource domain");
                    pending.failure.record_recovery_failure(
                        failure,
                        Some(ResourceFailurePoint::new(
                            &self.reservations.reservations[order],
                            order,
                            actual,
                        )),
                    );
                    pending.failure.ledger_after = self.ledger_snapshot_entries();
                    self.set_pending_failure(&pending.failure);
                    self.pending_subset_release = Some(pending);
                    return Err(self
                        .pending_subset_release
                        .as_ref()
                        .expect("pending release was restored")
                        .failure
                        .clone());
                }
            }
            let result = {
                let reservation = self.reservations.reservations[order].clone();
                let buffer = self
                    .lease
                    .as_ref()
                    .expect("committed transaction owns lease")
                    .buffer(order)
                    .expect("outstanding release owns its buffer");
                let driver = self.driver.as_mut().expect("live transaction owns driver");
                let context = ResourceTransactionContext {
                    runtime: &self.lease.as_ref().expect("transaction owns lease").runtime,
                    identity: &self.identity,
                    binding: &self.admission,
                    reservations: &self.reservations,
                    cursor: Some(ResourceActionCursor {
                        order,
                        action: ResourceTransactionAction::Release,
                        before: self.states[order],
                        allocation_authorized: false,
                    }),
                    allocation_authority: None,
                    pending_allocation: None,
                };
                driver.release_resource(&context, &reservation, buffer)
            };
            match result {
                Ok(()) => {
                    let before = self.states[order];
                    self.lease
                        .as_mut()
                        .expect("transaction owns lease ledger")
                        .clear(order);
                    self.states[order] = ResourceTransactionState::Released;
                    self.capacity_claim
                        .as_mut()
                        .expect("live transaction owns its static capacity claim")
                        .release_bytes(self.reservations.reservations[order].size_bytes());
                    pending.failure.completed.push(self.make_record(
                        order,
                        ResourceTransactionAction::Release,
                        before,
                        ResourceTransactionState::Released,
                    ));
                }
                Err(failure) => {
                    pending.failure.record_recovery_failure(
                        failure,
                        Some(ResourceFailurePoint::new(
                            &self.reservations.reservations[order],
                            order,
                            self.states[order],
                        )),
                    );
                    pending.failure.ledger_after = self.ledger_snapshot_entries();
                    self.set_pending_failure(&pending.failure);
                    let snapshot = pending.failure.clone();
                    self.pending_subset_release = Some(pending);
                    return Err(snapshot);
                }
            }
        }
        pending.failure.recovery_complete = true;
        pending.failure.ledger_after = self.ledger_snapshot_entries();
        let receipt = self
            .transition_receipt(
                ResourceTransactionAction::Release,
                pending.failure.ledger_before.clone(),
                pending.failure.completed.clone(),
            )
            .expect("forward-completed release journal must validate");
        self.clear_pending_failure();
        self.recovery_history.push(pending.failure);
        self.receipts.push(receipt.clone());
        Ok(receipt)
    }

    fn local_release_failure(&mut self, message: String) -> ResourceFailureReceipt {
        let ledger = self.ledger_snapshot_entries();
        let failure_id = self.issue_failure_id();
        ResourceFailureReceipt::new(
            failure_id,
            &self.identity,
            &self.admission,
            ResourceTransactionAction::Release,
            core_resource_failure("invalid_release_request", message, false),
            None,
            Vec::new(),
            ResourceRecoveryStrategy::ForwardCompletion,
            ledger.clone(),
            ledger,
        )
    }

    pub fn release(
        mut self,
    ) -> Result<ResourceTransaction<D, TransactionReleased>, ResourceReleaseTransitionError<D>>
    {
        if self.pending_subset_release.is_some() {
            let failure = self
                .pending_subset_release
                .as_ref()
                .expect("pending release exists")
                .failure
                .clone();
            return Err(ResourceReleaseTransitionError {
                transaction: Some(self),
                failure,
            });
        }
        if self
            .states
            .iter()
            .all(|state| *state == ResourceTransactionState::Released)
        {
            return Ok(self.advance(None));
        }
        if !self
            .states
            .iter()
            .all(|state| *state == ResourceTransactionState::Committed)
        {
            let failure = self.local_release_failure(
                "transaction contains non-releasable actual states".to_owned(),
            );
            return Err(ResourceReleaseTransitionError {
                transaction: Some(self),
                failure,
            });
        }
        match self.release_all_resources() {
            Ok(_) => Ok(self.advance(None)),
            Err(failure) => Err(ResourceReleaseTransitionError {
                transaction: Some(self),
                failure,
            }),
        }
    }
}

#[must_use = "failed plan runtime handoff retains the committed transaction"]
pub struct PlanRuntimeHandoffError<D>
where
    D: ResourceTransactionDriver,
{
    error: VNextError,
    transaction: Option<ResourceTransaction<D, TransactionCommitted>>,
}

impl<D> PlanRuntimeHandoffError<D>
where
    D: ResourceTransactionDriver,
{
    pub fn error(&self) -> &VNextError {
        &self.error
    }

    pub fn into_transaction(mut self) -> ResourceTransaction<D, TransactionCommitted> {
        self.transaction
            .take()
            .expect("plan runtime handoff error owns its transaction")
    }
}

#[must_use = "reverse recovery must complete or ownership must be quarantined"]
pub struct ResourcePrepareTransitionError<D: ResourceTransactionDriver, S: TransactionStage> {
    transaction: Option<ResourceTransaction<D, S>>,
    failure: ResourceFailureReceipt,
}

impl<D: ResourceTransactionDriver, S: TransactionStage> fmt::Debug
    for ResourcePrepareTransitionError<D, S>
{
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ResourcePrepareTransitionError")
            .field("stage", &S::STATE)
            .field("failure", &self.failure)
            .finish_non_exhaustive()
    }
}

impl<D: ResourceTransactionDriver, S: TransactionStage> ResourcePrepareTransitionError<D, S> {
    pub fn failure(&self) -> &ResourceFailureReceipt {
        &self.failure
    }

    pub fn recover(mut self) -> Result<ResourceTransaction<D, S>, Self> {
        while self.failure.compensation.len() < self.failure.completed.len() {
            let record_index = self.failure.completed.len() - 1 - self.failure.compensation.len();
            let attempted = self.failure.completed[record_index].clone();
            let order = attempted.order as usize;
            let transaction = self
                .transaction
                .as_mut()
                .expect("prepare recovery owns transaction");
            if transaction.states[order] != attempted.after {
                let failure = ResourceDriverFailure::new(core_resource_failure(
                    "compensation_ledger_diverged",
                    format!(
                        "resource `{}` expected actual state {}, found {}",
                        attempted.resource_id,
                        attempted.after.as_str(),
                        transaction.states[order].as_str()
                    ),
                    false,
                ))
                .expect("core failure has resource domain");
                self.failure.record_recovery_failure(
                    failure,
                    Some(ResourceFailurePoint::new(
                        &transaction.reservations.reservations[order],
                        order,
                        transaction.states[order],
                    )),
                );
                self.failure.ledger_after = transaction.ledger_snapshot_entries();
                transaction.set_pending_failure(&self.failure);
                return Err(self);
            }
            let result = match self.failure.action {
                ResourceTransactionAction::Reserve => {
                    let reservation = transaction.reservations.reservations[order].clone();
                    let driver = transaction
                        .driver
                        .as_mut()
                        .expect("live transaction owns driver");
                    let context = ResourceTransactionContext {
                        runtime: &transaction
                            .lease
                            .as_ref()
                            .expect("transaction owns lease")
                            .runtime,
                        identity: &transaction.identity,
                        binding: &transaction.admission,
                        reservations: &transaction.reservations,
                        cursor: Some(ResourceActionCursor {
                            order,
                            action: ResourceTransactionAction::Reserve,
                            before: transaction.states[order],
                            allocation_authorized: false,
                        }),
                        allocation_authority: None,
                        pending_allocation: None,
                    };
                    driver.compensate_reserve_resource(&context, &reservation)
                }
                ResourceTransactionAction::Commit => {
                    let reservation = transaction.reservations.reservations[order].clone();
                    let buffer = transaction
                        .lease
                        .as_ref()
                        .expect("transaction owns lease ledger")
                        .buffer(order)
                        .expect("uncompensated commit owns its buffer");
                    let driver = transaction
                        .driver
                        .as_mut()
                        .expect("live transaction owns driver");
                    let context = ResourceTransactionContext {
                        runtime: &transaction
                            .lease
                            .as_ref()
                            .expect("transaction owns lease")
                            .runtime,
                        identity: &transaction.identity,
                        binding: &transaction.admission,
                        reservations: &transaction.reservations,
                        cursor: Some(ResourceActionCursor {
                            order,
                            action: ResourceTransactionAction::Commit,
                            before: transaction.states[order],
                            allocation_authorized: false,
                        }),
                        allocation_authority: None,
                        pending_allocation: None,
                    };
                    driver.compensate_commit_resource(&context, &reservation, buffer)
                }
                _ => unreachable!("prepare recovery handles only reserve and commit"),
            };
            match result {
                Ok(()) => {
                    transaction.states[order] = attempted.before;
                    if self.failure.action == ResourceTransactionAction::Commit {
                        transaction
                            .lease
                            .as_mut()
                            .expect("transaction owns lease ledger")
                            .clear(order);
                    }
                    self.failure
                        .compensation
                        .push(ResourceCompensationRecord::from_transition(
                            &attempted,
                            self.failure.compensation.len(),
                        ));
                    self.failure.ledger_after = transaction.ledger_snapshot_entries();
                }
                Err(failure) => {
                    self.failure.record_recovery_failure(
                        failure,
                        Some(ResourceFailurePoint::new(
                            &transaction.reservations.reservations[order],
                            order,
                            transaction.states[order],
                        )),
                    );
                    self.failure.ledger_after = transaction.ledger_snapshot_entries();
                    transaction.set_pending_failure(&self.failure);
                    return Err(self);
                }
            }
        }
        self.failure.recovery_complete = true;
        let mut transaction = self
            .transaction
            .take()
            .expect("prepare recovery owns transaction");
        self.failure.ledger_after = transaction.ledger_snapshot_entries();
        transaction.clear_pending_failure();
        if self.failure.action == ResourceTransactionAction::Commit {
            for authority in &transaction.allocation_issued {
                authority.store(false, Ordering::Release);
            }
        }
        transaction.recovery_history.push(self.failure);
        Ok(transaction)
    }

    pub fn quarantine(mut self) -> Result<ResourceTransaction<D, TransactionQuarantined>, Self> {
        let transaction = self
            .transaction
            .as_mut()
            .expect("prepare recovery owns transaction");
        let ledger_before = transaction.ledger_snapshot_entries();
        match transaction.quarantine_live() {
            Ok(records) => {
                self.failure.recovery_complete = true;
                self.failure.ledger_after = transaction.ledger_snapshot_entries();
                transaction.clear_pending_failure();
                transaction.recovery_history.push(self.failure.clone());
                let receipt = transaction
                    .transition_receipt(
                        ResourceTransactionAction::Quarantine,
                        ledger_before,
                        records,
                    )
                    .expect("core quarantine journal must validate");
                let transaction = self
                    .transaction
                    .take()
                    .expect("prepare recovery owns transaction");
                Ok(transaction.advance(Some(receipt)))
            }
            Err(failure) => {
                self.failure.record_recovery_failure(failure, None);
                self.failure.ledger_after = transaction.ledger_snapshot_entries();
                transaction.set_pending_failure(&self.failure);
                Err(self)
            }
        }
    }
}

#[must_use = "poisoned commit ownership must be reconciled or quarantined"]
pub struct ResourcePoisonedTransaction<D: ResourceTransactionDriver> {
    transaction: Option<ResourceTransaction<D, TransactionReserved>>,
    failure: ResourceFailureReceipt,
    poisoned_order: usize,
}

impl<D: ResourceTransactionDriver> fmt::Debug for ResourcePoisonedTransaction<D> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ResourcePoisonedTransaction")
            .field("failure", &self.failure)
            .field("poisoned_order", &self.poisoned_order)
            .finish_non_exhaustive()
    }
}

impl<D: ResourceTransactionDriver> ResourcePoisonedTransaction<D> {
    pub fn failure(&self) -> &ResourceFailureReceipt {
        &self.failure
    }

    pub fn reconcile(
        mut self,
    ) -> Result<ResourcePrepareTransitionError<D, TransactionReserved>, Self> {
        let transaction = self
            .transaction
            .as_mut()
            .expect("poisoned owner owns transaction");
        let reservation = transaction.reservations.reservations[self.poisoned_order].clone();
        let result = {
            let lease = transaction
                .lease
                .as_ref()
                .expect("poisoned transaction owns lease ledger");
            let slot = &lease.slots[self.poisoned_order];
            let actual = ResourceCommitView {
                resource_id: slot
                    .actual_resource_id
                    .as_ref()
                    .expect("poisoned allocation records actual resource identity"),
                generation: slot
                    .actual_generation
                    .expect("poisoned allocation records actual generation"),
                descriptor: slot
                    .descriptor
                    .as_ref()
                    .expect("poisoned allocation records actual descriptor"),
                buffer: slot
                    .buffer
                    .as_ref()
                    .expect("poisoned allocation remains core-owned"),
            };
            let driver = transaction
                .driver
                .as_mut()
                .expect("live transaction owns driver");
            let context = ResourceTransactionContext {
                runtime: &lease.runtime,
                identity: &transaction.identity,
                binding: &transaction.admission,
                reservations: &transaction.reservations,
                cursor: Some(ResourceActionCursor {
                    order: self.poisoned_order,
                    action: ResourceTransactionAction::Commit,
                    before: transaction.states[self.poisoned_order],
                    allocation_authorized: false,
                }),
                allocation_authority: None,
                pending_allocation: None,
            };
            driver.reconcile_commit_outcome(&context, &reservation, actual)
        };
        match result {
            Ok(()) => {
                transaction
                    .lease
                    .as_mut()
                    .expect("poisoned transaction owns lease ledger")
                    .clear(self.poisoned_order);
                self.failure.recovery_strategy = ResourceRecoveryStrategy::ReverseCompensation;
                self.failure.ledger_after = transaction.ledger_snapshot_entries();
                let transaction = self
                    .transaction
                    .take()
                    .expect("poisoned owner owns transaction");
                Ok(ResourcePrepareTransitionError {
                    transaction: Some(transaction),
                    failure: self.failure,
                })
            }
            Err(failure) => {
                self.failure.record_recovery_failure(
                    failure,
                    Some(ResourceFailurePoint::new(
                        &reservation,
                        self.poisoned_order,
                        transaction.states[self.poisoned_order],
                    )),
                );
                self.failure.ledger_after = transaction.ledger_snapshot_entries();
                transaction.set_pending_failure(&self.failure);
                Err(self)
            }
        }
    }

    pub fn quarantine(mut self) -> Result<ResourceTransaction<D, TransactionQuarantined>, Self> {
        let transaction = self
            .transaction
            .as_mut()
            .expect("poisoned owner owns transaction");
        let ledger_before = transaction.ledger_snapshot_entries();
        match transaction.quarantine_live() {
            Ok(records) => {
                self.failure.recovery_complete = true;
                self.failure.ledger_after = transaction.ledger_snapshot_entries();
                transaction.clear_pending_failure();
                transaction.recovery_history.push(self.failure.clone());
                let receipt = transaction
                    .transition_receipt(
                        ResourceTransactionAction::Quarantine,
                        ledger_before,
                        records,
                    )
                    .expect("poison quarantine journal must validate");
                let transaction = self
                    .transaction
                    .take()
                    .expect("poisoned owner owns transaction");
                Ok(transaction.advance(Some(receipt)))
            }
            Err(failure) => {
                self.failure.record_recovery_failure(failure, None);
                self.failure.ledger_after = transaction.ledger_snapshot_entries();
                transaction.set_pending_failure(&self.failure);
                Err(self)
            }
        }
    }
}

#[must_use = "rollback forward completion owns the transaction"]
pub struct ResourceRollbackTransitionError<D: ResourceTransactionDriver> {
    transaction: Option<ResourceTransaction<D, TransactionReserved>>,
    failure: ResourceFailureReceipt,
}

impl<D: ResourceTransactionDriver> fmt::Debug for ResourceRollbackTransitionError<D> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ResourceRollbackTransitionError")
            .field("failure", &self.failure)
            .finish_non_exhaustive()
    }
}

impl<D: ResourceTransactionDriver> ResourceRollbackTransitionError<D> {
    pub fn failure(&self) -> &ResourceFailureReceipt {
        &self.failure
    }

    pub fn complete(mut self) -> Result<ResourceTransaction<D, TransactionRolledBack>, Self> {
        let transaction = self
            .transaction
            .as_mut()
            .expect("rollback recovery owns transaction");
        for order in 0..transaction.states.len() {
            match transaction.states[order] {
                ResourceTransactionState::RolledBack => continue,
                ResourceTransactionState::Reserved => {}
                actual => {
                    let failure = ResourceDriverFailure::new(core_resource_failure(
                        "rollback_ledger_diverged",
                        format!("rollback found actual state {}", actual.as_str()),
                        false,
                    ))
                    .expect("core failure has resource domain");
                    self.failure.record_recovery_failure(
                        failure,
                        Some(ResourceFailurePoint::new(
                            &transaction.reservations.reservations[order],
                            order,
                            actual,
                        )),
                    );
                    self.failure.ledger_after = transaction.ledger_snapshot_entries();
                    transaction.set_pending_failure(&self.failure);
                    return Err(self);
                }
            }
            let result = {
                let reservation = transaction.reservations.reservations[order].clone();
                let driver = transaction
                    .driver
                    .as_mut()
                    .expect("live transaction owns driver");
                let context = ResourceTransactionContext {
                    runtime: &transaction
                        .lease
                        .as_ref()
                        .expect("transaction owns lease")
                        .runtime,
                    identity: &transaction.identity,
                    binding: &transaction.admission,
                    reservations: &transaction.reservations,
                    cursor: Some(ResourceActionCursor {
                        order,
                        action: ResourceTransactionAction::Rollback,
                        before: transaction.states[order],
                        allocation_authorized: false,
                    }),
                    allocation_authority: None,
                    pending_allocation: None,
                };
                driver.rollback_resource(&context, &reservation)
            };
            match result {
                Ok(()) => {
                    let before = transaction.states[order];
                    transaction.states[order] = ResourceTransactionState::RolledBack;
                    self.failure.completed.push(transaction.make_record(
                        order,
                        ResourceTransactionAction::Rollback,
                        before,
                        ResourceTransactionState::RolledBack,
                    ));
                }
                Err(failure) => {
                    self.failure.record_recovery_failure(
                        failure,
                        Some(ResourceFailurePoint::new(
                            &transaction.reservations.reservations[order],
                            order,
                            transaction.states[order],
                        )),
                    );
                    self.failure.ledger_after = transaction.ledger_snapshot_entries();
                    transaction.set_pending_failure(&self.failure);
                    return Err(self);
                }
            }
        }
        self.failure.recovery_complete = true;
        self.failure.ledger_after = transaction.ledger_snapshot_entries();
        let receipt = transaction
            .transition_receipt(
                ResourceTransactionAction::Rollback,
                self.failure.ledger_before.clone(),
                self.failure.completed.clone(),
            )
            .expect("forward-completed rollback journal must validate");
        transaction.clear_pending_failure();
        transaction.recovery_history.push(self.failure);
        let transaction = self
            .transaction
            .take()
            .expect("rollback recovery owns transaction");
        Ok(transaction.advance(Some(receipt)))
    }

    pub fn quarantine(mut self) -> Result<ResourceTransaction<D, TransactionQuarantined>, Self> {
        let transaction = self
            .transaction
            .as_mut()
            .expect("rollback recovery owns transaction");
        let ledger_before = transaction.ledger_snapshot_entries();
        match transaction.quarantine_live() {
            Ok(records) => {
                self.failure.recovery_complete = true;
                self.failure.ledger_after = transaction.ledger_snapshot_entries();
                transaction.clear_pending_failure();
                transaction.recovery_history.push(self.failure.clone());
                let receipt = transaction
                    .transition_receipt(
                        ResourceTransactionAction::Quarantine,
                        ledger_before,
                        records,
                    )
                    .expect("rollback quarantine journal must validate");
                let transaction = self
                    .transaction
                    .take()
                    .expect("rollback recovery owns transaction");
                Ok(transaction.advance(Some(receipt)))
            }
            Err(failure) => {
                self.failure.record_recovery_failure(failure, None);
                self.failure.ledger_after = transaction.ledger_snapshot_entries();
                transaction.set_pending_failure(&self.failure);
                Err(self)
            }
        }
    }
}

#[must_use = "release forward completion owns the transaction"]
pub struct ResourceReleaseTransitionError<D: ResourceTransactionDriver> {
    transaction: Option<ResourceTransaction<D, TransactionCommitted>>,
    failure: ResourceFailureReceipt,
}

impl<D: ResourceTransactionDriver> fmt::Debug for ResourceReleaseTransitionError<D> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ResourceReleaseTransitionError")
            .field("failure", &self.failure)
            .finish_non_exhaustive()
    }
}

impl<D: ResourceTransactionDriver> ResourceReleaseTransitionError<D> {
    pub fn failure(&self) -> &ResourceFailureReceipt {
        &self.failure
    }

    pub fn complete(mut self) -> Result<ResourceTransaction<D, TransactionReleased>, Self> {
        let transaction = self
            .transaction
            .as_mut()
            .expect("release recovery owns transaction");
        match transaction.complete_pending_release() {
            Ok(_) => {
                self.failure = transaction
                    .recovery_history
                    .last()
                    .cloned()
                    .unwrap_or_else(|| self.failure.clone());
                let transaction = self
                    .transaction
                    .take()
                    .expect("release recovery owns transaction");
                transaction.release().map_err(|next| Self {
                    transaction: next.transaction,
                    failure: next.failure,
                })
            }
            Err(failure) => {
                self.failure = failure;
                Err(self)
            }
        }
    }

    pub fn quarantine(mut self) -> Result<ResourceTransaction<D, TransactionQuarantined>, Self> {
        let transaction = self
            .transaction
            .as_mut()
            .expect("release recovery owns transaction");
        let ledger_before = transaction.ledger_snapshot_entries();
        match transaction.quarantine_live() {
            Ok(records) => {
                self.failure.recovery_complete = true;
                self.failure.ledger_after = transaction.ledger_snapshot_entries();
                transaction.clear_pending_failure();
                transaction.pending_subset_release = None;
                transaction.recovery_history.push(self.failure.clone());
                let receipt = transaction
                    .transition_receipt(
                        ResourceTransactionAction::Quarantine,
                        ledger_before,
                        records,
                    )
                    .expect("release quarantine journal must validate");
                let transaction = self
                    .transaction
                    .take()
                    .expect("release recovery owns transaction");
                Ok(transaction.advance(Some(receipt)))
            }
            Err(failure) => {
                self.failure.record_recovery_failure(failure, None);
                self.failure.ledger_after = transaction.ledger_snapshot_entries();
                transaction.set_pending_failure(&self.failure);
                Err(self)
            }
        }
    }
}

fn same_allocation_entry(left: &ResourceLeaseEntry, right: &ResourceLeaseEntry) -> bool {
    left.owner_node_id == right.owner_node_id
        && left.resource_id == right.resource_id
        && left.size_bytes == right.size_bytes
        && left.alignment_bytes == right.alignment_bytes
        && left.usage == right.usage
        && left.element_type == right.element_type
        && left.retention_policy == right.retention_policy
        && left.generation == right.generation
}

fn validate_context_envelope(
    identity: &ResourceTransactionIdentity,
    admission: &StaticProvisioningBinding,
    before: &[ResourceLedgerEntrySnapshot],
    after: &[ResourceLedgerEntrySnapshot],
) -> Result<(), VNextError> {
    if identity.pool_id != admission.pool_id()
        || identity.request_id != admission.request_id
        || admission.pool_identity.plan_id != admission.plan_id
        || admission.pool_identity.plan_hash != admission.plan_hash
        || admission.pool_identity.device_id != admission.device_id
        || admission
            .pool_identity
            .device_runtime_implementation_fingerprint
            != admission.device_runtime_implementation_fingerprint
        || admission.pool_identity.admission_generation != admission.admission_generation
        || admission.admission_generation == 0
        || admission.maximum_active_sequences == 0
        || admission.device_capacity_bytes == 0
        || admission.usable_capacity_bytes == 0
        || admission.usable_capacity_bytes > admission.device_capacity_bytes
        || admission.plan_static_bytes > admission.usable_capacity_bytes
        || admission.admitted_bytes == 0
        || admission.admitted_bytes != admission.plan_static_bytes
        || admission.admitted_bytes > admission.usable_capacity_bytes
        || before.is_empty()
        || before.len() != after.len()
    {
        return Err(invalid_resource(
            "trusted resource validation context has an invalid envelope",
        ));
    }
    let mut resources = BTreeSet::new();
    let mut total = 0_u64;
    for (before_entry, after_entry) in before.iter().zip(after) {
        if !same_allocation_entry(&before_entry.entry, &after_entry.entry)
            || !resources.insert(before_entry.entry.resource_id.clone())
            || before_entry.entry.generation != admission.admission_generation
        {
            return Err(invalid_resource(
                "trusted resource validation context changes allocation identity",
            ));
        }
        total = total
            .checked_add(before_entry.entry.size_bytes)
            .ok_or_else(|| invalid_resource("validation context bytes overflow u64"))?;
    }
    if total != admission.admitted_bytes {
        return Err(invalid_resource(
            "validation context does not cover the complete admitted allocation set",
        ));
    }
    Ok(())
}

fn validate_transition_records_against_context(
    records: &[ResourceTransitionRecord],
    context: &ResourceTransitionValidationContext,
) -> Result<(), VNextError> {
    validate_context_envelope(
        &context.identity,
        &context.admission,
        &context.before,
        &context.after,
    )?;
    let changed = context
        .before
        .iter()
        .zip(&context.after)
        .enumerate()
        .filter(|(_, (before, after))| {
            before.transaction_state != after.transaction_state
                || before.buffer_present != after.buffer_present
        })
        .collect::<Vec<_>>();
    if changed.is_empty() || changed.len() != records.len() {
        return Err(invalid_resource(
            "resource receipt does not cover the exact ledger delta",
        ));
    }
    for (record, (order, (before, after))) in records.iter().zip(changed) {
        record.validate()?;
        if !record.matches_identity_and_admission(&context.identity, &context.admission)
            || record.action != context.action
            || record.order as usize != order
            || !record.matches_snapshot(before)
            || !record.matches_snapshot(after)
            || record.before != before.transaction_state
            || record.after != after.transaction_state
            || expected_transition(context.action, record.before) != Some(record.after)
        {
            return Err(invalid_resource(
                "resource receipt differs from trusted allocation or ledger state",
            ));
        }
        match context.action {
            ResourceTransactionAction::Commit => {
                if before.buffer_present || !after.buffer_present {
                    return Err(invalid_resource(
                        "commit receipt does not prove buffer acquisition",
                    ));
                }
            }
            ResourceTransactionAction::Release | ResourceTransactionAction::Quarantine => {
                if before.transaction_state == ResourceTransactionState::Committed
                    && (!before.buffer_present || after.buffer_present)
                {
                    return Err(invalid_resource(
                        "cleanup receipt does not prove committed buffer return",
                    ));
                }
            }
            ResourceTransactionAction::Reserve | ResourceTransactionAction::Rollback => {
                if before.buffer_present != after.buffer_present {
                    return Err(invalid_resource(
                        "non-buffer transition changed buffer ownership",
                    ));
                }
            }
        }
    }
    Ok(())
}

fn validate_lease_entries_against_context(
    entries: &[ResourceLeaseEntry],
    before_state: ResourceLeaseState,
    after_state: ResourceLeaseState,
    context: &ResourceLeaseValidationContext,
) -> Result<(), VNextError> {
    validate_context_envelope(
        &context.identity,
        &context.admission,
        &context.before,
        &context.after,
    )?;
    if expected_lease_transition(context.action, before_state) != Some(after_state) {
        return Err(VNextError::InvalidLeaseTransition {
            lease_id: context.identity.transaction_id.to_string(),
            from: before_state.as_str(),
            action: context.action.as_str(),
        });
    }
    let changed = context
        .before
        .iter()
        .zip(&context.after)
        .filter(|(before, after)| before.entry.state != after.entry.state)
        .collect::<Vec<_>>();
    if changed.is_empty() || changed.len() != entries.len() {
        return Err(invalid_resource(
            "lease receipt does not cover the exact lease-state delta",
        ));
    }
    for (entry, (before, after)) in entries.iter().zip(changed) {
        if !same_allocation_entry(entry, &before.entry)
            || !same_allocation_entry(entry, &after.entry)
            || before.entry.state != before_state
            || after.entry.state != after_state
            || entry.state != after_state
            || before.transaction_state != after.transaction_state
            || before.transaction_state != ResourceTransactionState::Committed
            || before.buffer_present != after.buffer_present
            || !before.buffer_present
        {
            return Err(invalid_resource(
                "lease receipt differs from trusted allocation or ledger state",
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
#[path = "resource/dynamic_pool_tests.rs"]
mod dynamic_pool_tests;

#[cfg(test)]
#[path = "resource/sequence_session_frame_tests.rs"]
mod sequence_session_frame_tests;
