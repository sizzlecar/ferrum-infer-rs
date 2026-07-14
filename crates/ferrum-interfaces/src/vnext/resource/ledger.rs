use super::*;

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

pub(super) const fn expected_lease_transition(
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
    pub(super) owner_node_id: Option<NodeId>,
    pub(super) resource_id: ResourceId,
    pub(super) size_bytes: u64,
    pub(super) alignment_bytes: u64,
    pub(super) usage: BufferUsage,
    pub(super) element_type: ElementType,
    pub(super) retention_policy: ResourceRetentionPolicy,
    pub(super) generation: u64,
    pub(super) state: ResourceLeaseState,
}

impl ResourceLeaseEntry {
    pub(super) fn from_reservation(
        reservation: &ResourceReservation,
        state: ResourceLeaseState,
    ) -> Self {
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
    pub(super) entry: ResourceLeaseEntry,
    pub(super) transaction_state: ResourceTransactionState,
    pub(super) buffer_present: bool,
    pub(super) actual_resource_id: Option<ResourceId>,
    pub(super) actual_generation: Option<u64>,
    pub(super) actual_descriptor: Option<BufferDescriptor>,
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
    pub(super) identity: ResourceTransactionIdentity,
    pub(super) admission: StaticProvisioningBinding,
    pub(super) entries: Vec<ResourceLedgerEntrySnapshot>,
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
    pub(super) identity: ResourceTransactionIdentity,
    pub(super) admission: StaticProvisioningBinding,
    pub(super) action: ResourceTransactionAction,
    pub(super) before: Vec<ResourceLedgerEntrySnapshot>,
    pub(super) after: Vec<ResourceLedgerEntrySnapshot>,
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
    pub(super) identity: ResourceTransactionIdentity,
    pub(super) admission: StaticProvisioningBinding,
    pub(super) action: ResourceLeaseAction,
    pub(super) before: Vec<ResourceLedgerEntrySnapshot>,
    pub(super) after: Vec<ResourceLedgerEntrySnapshot>,
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
    pub(super) resource_id: ResourceId,
    generation: u64,
    retention_policy: ResourceRetentionPolicy,
    pub(super) action: ResourceTransactionAction,
    pub(super) before: ResourceTransactionState,
    pub(super) after: ResourceTransactionState,
    pub(super) order: u32,
}

impl ResourceTransitionRecord {
    pub(super) fn from_reservation(
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

    pub(super) fn matches_identity_and_admission(
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

    pub(super) fn matches_snapshot(&self, snapshot: &ResourceLedgerEntrySnapshot) -> bool {
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
    pub(super) fn from_transition(
        attempted: &ResourceTransitionRecord,
        compensation_order: usize,
    ) -> Self {
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
    pub(super) fn from_context(
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
    pub(super) fn from_context(
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
    pub(super) fn new(
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
    pub(super) action: ResourceTransactionAction,
    pub(super) failure: FailureEnvelope,
    failure_point: Option<ResourceFailurePoint>,
    pub(super) completed: Vec<ResourceTransitionRecord>,
    pub(super) compensation: Vec<ResourceCompensationRecord>,
    recovery_failures: Vec<ResourceRecoveryFailure>,
    pub(super) recovery_strategy: ResourceRecoveryStrategy,
    pub(super) recovery_complete: bool,
    pub(super) ledger_before: Vec<ResourceLedgerEntrySnapshot>,
    pub(super) ledger_after: Vec<ResourceLedgerEntrySnapshot>,
}

impl ResourceFailureReceipt {
    pub(super) fn new(
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

    pub(super) fn record_recovery_failure(
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
    pub(super) identity: ResourceTransactionIdentity,
    pub(super) admission: StaticProvisioningBinding,
    pub(super) state: ResourceTransactionState,
    pub(super) pending_action: Option<ResourceTransactionAction>,
    pub(super) ledger: Vec<ResourceLedgerEntrySnapshot>,
    pub(super) active_sequence_slots: Vec<u32>,
    pub(super) poisoned_sequence_slots: Vec<u32>,
    pub(super) undrained_sequence_slots: Vec<u32>,
    pub(super) failure: Option<FailureEnvelope>,
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
