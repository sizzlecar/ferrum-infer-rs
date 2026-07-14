use super::solver::quantize_storage_bytes;
use super::{
    AttributeId, BTreeMap, BTreeSet, BufferRequest, BufferUsage, CapabilityId, ContractVersion,
    Deserialize, Digest, DynamicStorageContract, ElementType, NodeId, OperationId, ProgramValueId,
    ProviderCompatibilityRejectReason, ProviderId, ProviderResourcePlan, ResolvedValueBinding,
    ResourceId, SemanticValue, Serialize, Sha256, StateId, TensorAccess, VNextError,
};

pub const EXECUTION_PLAN_SCHEMA: PlanSchemaVersion = PlanSchemaVersion::new(1, 0);
pub const MAX_EXECUTION_PLAN_WIRE_BYTES: usize = 16 * 1024 * 1024;
/// Maximum number of O(graph) static allocations plus dynamic descriptors.
/// This limit is independent of the concurrency ceiling.
pub const MAX_EXECUTION_PLAN_RESOURCE_ROWS: usize = 65_536;

pub(super) fn invalid_plan(reason: impl Into<String>) -> VNextError {
    VNextError::InvalidExecutionPlan {
        reason: reason.into(),
    }
}

pub(super) fn validate_active_sequence_ceiling(
    maximum_active_sequences: u32,
) -> Result<(), VNextError> {
    if maximum_active_sequences == 0 {
        return Err(invalid_plan(
            "maximum active sequences protocol ceiling must be non-zero",
        ));
    }
    Ok(())
}

pub(super) fn canonical_json(value: serde_json::Value) -> serde_json::Value {
    match value {
        serde_json::Value::Array(values) => {
            serde_json::Value::Array(values.into_iter().map(canonical_json).collect())
        }
        serde_json::Value::Object(values) => serde_json::Value::Object(
            values
                .into_iter()
                .map(|(key, value)| (key, canonical_json(value)))
                .collect::<BTreeMap<_, _>>()
                .into_iter()
                .collect(),
        ),
        scalar => scalar,
    }
}

pub(super) fn canonical_fingerprint<T: Serialize>(
    value: &T,
    context: &'static str,
) -> Result<String, VNextError> {
    let value = serde_json::to_value(value).map_err(|error| VNextError::Serialization {
        context,
        message: error.to_string(),
    })?;
    let bytes =
        serde_json::to_vec(&canonical_json(value)).map_err(|error| VNextError::Serialization {
            context,
            message: error.to_string(),
        })?;
    Ok(format!("{:x}", Sha256::digest(bytes)))
}

pub(super) fn is_canonical_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct PlanSchemaVersion {
    pub major: u16,
    pub minor: u16,
}

impl PlanSchemaVersion {
    pub const fn new(major: u16, minor: u16) -> Self {
        Self { major, minor }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
#[serde(transparent)]
pub struct PlanHash(String);

impl PlanHash {
    pub(super) fn new(value: String) -> Result<Self, VNextError> {
        if !is_canonical_sha256(&value) {
            return Err(invalid_plan("plan hash must be a lowercase SHA256"));
        }
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl<'de> Deserialize<'de> for PlanHash {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        Self::new(String::deserialize(deserializer)?).map_err(serde::de::Error::custom)
    }
}

impl std::fmt::Display for PlanHash {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.0)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PlanProviderRejectReason {
    NotRegistered,
    Incompatible(Vec<ProviderCompatibilityRejectReason>),
    StorageIncompatible { resource_ids: Vec<ResourceId> },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RejectedProvider {
    pub(super) provider_id: ProviderId,
    pub(super) reasons: PlanProviderRejectReason,
}

impl RejectedProvider {
    pub fn provider_id(&self) -> &ProviderId {
        &self.provider_id
    }

    pub fn reasons(&self) -> &PlanProviderRejectReason {
        &self.reasons
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProviderSelectionReason {
    PreferredCompatible,
    DeterministicCompatible,
    FallbackFromPreferred,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProviderSelection {
    pub(super) requested_provider: Option<ProviderId>,
    pub(super) selected_provider: ProviderId,
    pub(super) selection_reason: ProviderSelectionReason,
    pub(super) rejected_providers: Vec<RejectedProvider>,
}

impl ProviderSelection {
    pub fn requested_provider(&self) -> Option<&ProviderId> {
        self.requested_provider.as_ref()
    }

    pub fn selected_provider(&self) -> &ProviderId {
        &self.selected_provider
    }

    pub const fn selection_reason(&self) -> ProviderSelectionReason {
        self.selection_reason
    }

    pub fn rejected_providers(&self) -> &[RejectedProvider] {
        &self.rejected_providers
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PlanExactAliasKind {
    MayAlias,
    MustAlias,
}

/// Core-proven exact storage equality between one output and its declared
/// input. A MayAlias contract with distinct storage deliberately emits no
/// edge.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PlanExactAlias {
    pub(super) output_value_id: ProgramValueId,
    pub(super) output_ordinal: u32,
    pub(super) input_value_id: ProgramValueId,
    pub(super) input_ordinal: u32,
    pub(super) kind: PlanExactAliasKind,
}

impl PlanExactAlias {
    pub fn output_value_id(&self) -> &ProgramValueId {
        &self.output_value_id
    }

    pub const fn output_ordinal(&self) -> u32 {
        self.output_ordinal
    }

    pub fn input_value_id(&self) -> &ProgramValueId {
        &self.input_value_id
    }

    pub const fn input_ordinal(&self) -> u32 {
        self.input_ordinal
    }

    pub const fn kind(&self) -> PlanExactAliasKind {
        self.kind
    }
}

/// Typed state effect derived from the operation access contract for a
/// declared ModelProgram state binding.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PlanStateEffect {
    pub(super) state_id: StateId,
    pub(super) state_value_id: ProgramValueId,
    pub(super) lifetime: AllocationLifetime,
    pub(super) access: TensorAccess,
    pub(super) resource_ids: Vec<ResourceId>,
}

impl PlanStateEffect {
    pub fn state_id(&self) -> &StateId {
        &self.state_id
    }

    pub fn state_value_id(&self) -> &ProgramValueId {
        &self.state_value_id
    }

    pub const fn lifetime(&self) -> AllocationLifetime {
        self.lifetime
    }

    pub const fn access(&self) -> TensorAccess {
        self.access
    }

    pub fn resource_ids(&self) -> &[ResourceId] {
        &self.resource_ids
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct PlanNode {
    pub(super) id: NodeId,
    pub(super) dependencies: Vec<NodeId>,
    pub(super) operation_id: OperationId,
    pub(super) operation_version: ContractVersion,
    pub(super) operation_fingerprint: String,
    pub(super) provider_implementation_fingerprint: String,
    pub(super) required_capabilities: BTreeSet<CapabilityId>,
    pub(super) attributes: BTreeMap<AttributeId, SemanticValue>,
    pub(super) selection: ProviderSelection,
    pub(super) provider_resources: ProviderResourcePlan,
    pub(super) values: Vec<ResolvedValueBinding>,
    pub(super) exact_aliases: Vec<PlanExactAlias>,
    pub(super) state_effects: Vec<PlanStateEffect>,
    pub(super) scratch_resource: Option<ResourceId>,
    pub(super) persistent_resource: Option<ResourceId>,
    pub(super) resources: Vec<ResourceId>,
}

impl PlanNode {
    pub fn id(&self) -> &NodeId {
        &self.id
    }

    pub fn dependencies(&self) -> &[NodeId] {
        &self.dependencies
    }

    pub fn operation_id(&self) -> &OperationId {
        &self.operation_id
    }

    pub const fn operation_version(&self) -> ContractVersion {
        self.operation_version
    }

    pub fn operation_fingerprint(&self) -> &str {
        &self.operation_fingerprint
    }

    pub fn provider_implementation_fingerprint(&self) -> &str {
        &self.provider_implementation_fingerprint
    }

    pub fn required_capabilities(&self) -> &BTreeSet<CapabilityId> {
        &self.required_capabilities
    }

    pub fn attributes(&self) -> &BTreeMap<AttributeId, SemanticValue> {
        &self.attributes
    }

    pub fn selection(&self) -> &ProviderSelection {
        &self.selection
    }

    pub fn provider_resources(&self) -> &ProviderResourcePlan {
        &self.provider_resources
    }

    pub fn values(&self) -> &[ResolvedValueBinding] {
        &self.values
    }

    pub fn exact_aliases(&self) -> &[PlanExactAlias] {
        &self.exact_aliases
    }

    pub fn state_effects(&self) -> &[PlanStateEffect] {
        &self.state_effects
    }

    pub fn scratch_resource(&self) -> Option<&ResourceId> {
        self.scratch_resource.as_ref()
    }

    pub fn persistent_resource(&self) -> Option<&ResourceId> {
        self.persistent_resource.as_ref()
    }

    pub fn resources(&self) -> &[ResourceId] {
        &self.resources
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AllocationLifetime {
    Plan,
    Request,
    Sequence,
    Step,
    Invocation,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AllocationKind {
    Value,
    Scratch { node_id: NodeId },
    Persistent { node_id: NodeId },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResourceAllocation {
    pub(super) resource_id: ResourceId,
    pub(super) per_instance_bytes: u64,
    pub(super) instance_stride_bytes: u64,
    pub(super) instance_count: u32,
    pub(super) size_bytes: u64,
    pub(super) alignment_bytes: u64,
    pub(super) usage: BufferUsage,
    pub(super) element_type: ElementType,
    pub(super) lifetime: AllocationLifetime,
    pub(super) kind: AllocationKind,
    pub(super) storage: DynamicStorageContract,
}

impl ResourceAllocation {
    pub(super) fn new(
        resource_id: ResourceId,
        per_instance_bytes: u64,
        alignment_bytes: u64,
        usage: BufferUsage,
        element_type: ElementType,
        kind: AllocationKind,
        storage: DynamicStorageContract,
    ) -> Result<Self, VNextError> {
        if per_instance_bytes == 0 || alignment_bytes == 0 || !alignment_bytes.is_power_of_two() {
            return Err(invalid_plan(format!(
                "resource `{resource_id}` has invalid size or alignment"
            )));
        }
        let instance_stride_bytes =
            quantize_storage_bytes(per_instance_bytes, alignment_bytes, storage.profile())?;
        Ok(Self {
            resource_id,
            per_instance_bytes,
            instance_stride_bytes,
            instance_count: 1,
            size_bytes: instance_stride_bytes,
            alignment_bytes,
            usage,
            element_type,
            lifetime: AllocationLifetime::Plan,
            kind,
            storage,
        })
    }

    pub fn resource_id(&self) -> &ResourceId {
        &self.resource_id
    }

    pub const fn size_bytes(&self) -> u64 {
        self.size_bytes
    }

    pub const fn per_instance_bytes(&self) -> u64 {
        self.per_instance_bytes
    }

    pub const fn instance_stride_bytes(&self) -> u64 {
        self.instance_stride_bytes
    }

    pub const fn instance_count(&self) -> u32 {
        self.instance_count
    }

    pub fn scoped_offset_bytes(
        &self,
        base_offset_bytes: u64,
        _active_sequence_slot: u32,
    ) -> Result<u64, VNextError> {
        if base_offset_bytes >= self.per_instance_bytes {
            return Err(invalid_plan(format!(
                "resource `{}` base offset is outside its per-instance span",
                self.resource_id
            )));
        }
        Ok(base_offset_bytes)
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

    pub const fn lifetime(&self) -> AllocationLifetime {
        self.lifetime
    }

    pub fn kind(&self) -> &AllocationKind {
        &self.kind
    }

    pub fn storage(&self) -> &DynamicStorageContract {
        &self.storage
    }

    pub fn buffer_request(&self) -> Result<BufferRequest, VNextError> {
        BufferRequest::new(
            self.resource_id.clone(),
            self.size_bytes,
            self.alignment_bytes,
            self.usage,
            self.element_type,
        )
    }
}
