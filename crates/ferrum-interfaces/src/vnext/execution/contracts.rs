use super::{
    invalid_plan, is_canonical_sha256, AttributeId, BTreeMap, BTreeSet, CapabilityId,
    ContractVersion, Deserialize, NodeId, OperationId, ProgramValueId,
    ProviderCompatibilityRejectReason, ProviderId, ProviderResourcePlan, ResolvedValueBinding,
    ResolvedValueRole, ResourceId, SemanticValue, Serialize, StateId, TensorAccess, VNextError,
};

pub const EXECUTION_PLAN_SCHEMA: PlanSchemaVersion = PlanSchemaVersion::new(4, 0);
pub const MAX_EXECUTION_PLAN_WIRE_BYTES: usize = 16 * 1024 * 1024;
/// Maximum number of O(graph) static allocations plus dynamic descriptors.
/// This limit is independent of the concurrency ceiling.
pub const MAX_EXECUTION_PLAN_RESOURCE_ROWS: usize = 65_536;

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

/// Exact resolved binding projection whose one logical axis advances with the
/// core-issued packed token work shape.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct NodeTokenBindingProjection {
    pub(super) value_id: ProgramValueId,
    pub(super) role: ResolvedValueRole,
    pub(super) ordinal: u32,
    pub(super) axis: u32,
    pub(super) rank: u32,
    pub(super) canonical_extent: u64,
}

impl NodeTokenBindingProjection {
    pub fn value_id(&self) -> &ProgramValueId {
        &self.value_id
    }

    pub const fn role(&self) -> ResolvedValueRole {
        self.role
    }

    pub const fn ordinal(&self) -> u32 {
        self.ordinal
    }

    pub const fn axis(&self) -> u32 {
        self.axis
    }

    pub const fn rank(&self) -> u32 {
        self.rank
    }

    pub const fn canonical_extent(&self) -> u64 {
        self.canonical_extent
    }
}

/// Core-derived work mapping stored in the immutable execution plan. Token
/// projections are resolved from one model-declared source dimension through
/// the operation's symbolic signature, so providers never infer work from a
/// tensor element count or model family.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NodeWorkContract {
    Fixed,
    Tokens {
        source: NodeTokenBindingProjection,
        projections: Vec<NodeTokenBindingProjection>,
    },
}

impl NodeWorkContract {
    pub fn token_source(&self) -> Option<&NodeTokenBindingProjection> {
        match self {
            Self::Fixed => None,
            Self::Tokens { source, .. } => Some(source),
        }
    }

    pub fn token_projections(&self) -> &[NodeTokenBindingProjection] {
        match self {
            Self::Fixed => &[],
            Self::Tokens { projections, .. } => projections,
        }
    }

    pub fn token_projection(
        &self,
        role: ResolvedValueRole,
        ordinal: u32,
    ) -> Option<&NodeTokenBindingProjection> {
        self.token_projections()
            .iter()
            .find(|projection| projection.role == role && projection.ordinal == ordinal)
    }
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
    pub(super) work: NodeWorkContract,
    pub(super) selection: ProviderSelection,
    pub(super) provider_resources: ProviderResourcePlan,
    pub(super) values: Vec<ResolvedValueBinding>,
    pub(super) exact_aliases: Vec<PlanExactAlias>,
    pub(super) state_effects: Vec<PlanStateEffect>,
    pub(super) scratch_resource: Option<ResourceId>,
    pub(super) binding_resource: Option<ResourceId>,
    pub(super) persistent_resource: Option<ResourceId>,
    pub(super) resources: Vec<ResourceId>,
}

impl PlanNode {
    #[cfg(test)]
    pub(crate) fn resource_test_node(id: NodeId) -> Self {
        let provider_resources = ProviderResourcePlan {
            provider_id: ProviderId::new("provider/resource-test").expect("valid provider id"),
            estimator_id: "resource-test-estimator".to_owned(),
            estimator_version: ContractVersion::new(1, 0),
            estimator_implementation_fingerprint: "1".repeat(64),
            estimator_input_fingerprint: "2".repeat(64),
            estimate_fingerprint: "3".repeat(64),
            value_alignment_bytes: 16,
            scratch: None,
            binding: None,
            persistent: None,
        };
        let selected_provider = provider_resources.provider_id().clone();
        Self {
            id,
            dependencies: Vec::new(),
            operation_id: OperationId::new("operation/resource-test").expect("valid operation id"),
            operation_version: ContractVersion::new(1, 0),
            operation_fingerprint: "4".repeat(64),
            provider_implementation_fingerprint: "5".repeat(64),
            required_capabilities: BTreeSet::new(),
            attributes: BTreeMap::new(),
            work: super::NodeWorkContract::Fixed,
            selection: super::ProviderSelection {
                requested_provider: None,
                selected_provider,
                selection_reason: super::ProviderSelectionReason::DeterministicCompatible,
                rejected_providers: Vec::new(),
            },
            provider_resources,
            values: Vec::new(),
            exact_aliases: Vec::new(),
            state_effects: Vec::new(),
            scratch_resource: None,
            binding_resource: None,
            persistent_resource: None,
            resources: Vec::new(),
        }
    }

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

    pub fn work(&self) -> &NodeWorkContract {
        &self.work
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

    pub fn binding_resource(&self) -> Option<&ResourceId> {
        self.binding_resource.as_ref()
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
    Binding { node_id: NodeId },
    Persistent { node_id: NodeId },
}
