use serde::{Deserialize, Deserializer, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::ops::Range;

use super::{
    AliasPolicy, AttributeId, BlockedTensorPadding, BufferRequest, BufferUsage, CapabilityCatalog,
    CapabilityId, ContractVersion, DynamicStorageProfile, DynamicStorageRequirement, ElementType,
    ModelFamilyId, NodeId, OperationDescriptor, OperationId, OperationPlanningHandle,
    OperationPlanningRegistry, OperationProviderDescriptor, OperationRegistryAuthority,
    OperationResourceEstimate, OperationResourceEstimateRequest, PlanId, PreparedModelFamily,
    ProgramNode, ProgramTensorSpec, ProgramValueId, ProviderCompatibilityRejectReason,
    ProviderCompatibilityRequest, ProviderId, QuantizationFormatId, ResolvedTensorLayout,
    ResolvedTensorSpec, ResolvedValueBinding, ResolvedValueRole, ResolvedValueStorage, ResourceId,
    SemanticValue, StateCapacityDemand, StateId, StateLifetime, TensorAccess, VNextError,
    WeightEncoding, WeightFormatId, WeightId,
};

pub const EXECUTION_PLAN_SCHEMA: PlanSchemaVersion = PlanSchemaVersion::new(1, 0);
pub const MAX_EXECUTION_PLAN_WIRE_BYTES: usize = 16 * 1024 * 1024;
/// Maximum number of O(graph) static allocations plus dynamic descriptors.
/// This limit is independent of the concurrency ceiling.
pub const MAX_EXECUTION_PLAN_RESOURCE_ROWS: usize = 65_536;

fn invalid_plan(reason: impl Into<String>) -> VNextError {
    VNextError::InvalidExecutionPlan {
        reason: reason.into(),
    }
}

fn validate_active_sequence_ceiling(maximum_active_sequences: u32) -> Result<(), VNextError> {
    if maximum_active_sequences == 0 {
        return Err(invalid_plan(
            "maximum active sequences protocol ceiling must be non-zero",
        ));
    }
    Ok(())
}

fn canonical_json(value: serde_json::Value) -> serde_json::Value {
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

fn canonical_fingerprint<T: Serialize>(
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

fn is_canonical_sha256(value: &str) -> bool {
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
    fn new(value: String) -> Result<Self, VNextError> {
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
    provider_id: ProviderId,
    reasons: PlanProviderRejectReason,
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
    requested_provider: Option<ProviderId>,
    selected_provider: ProviderId,
    selection_reason: ProviderSelectionReason,
    rejected_providers: Vec<RejectedProvider>,
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
    output_value_id: ProgramValueId,
    output_ordinal: u32,
    input_value_id: ProgramValueId,
    input_ordinal: u32,
    kind: PlanExactAliasKind,
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
    state_id: StateId,
    state_value_id: ProgramValueId,
    lifetime: AllocationLifetime,
    access: TensorAccess,
    resource_ids: Vec<ResourceId>,
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
    id: NodeId,
    dependencies: Vec<NodeId>,
    operation_id: OperationId,
    operation_version: ContractVersion,
    operation_fingerprint: String,
    provider_implementation_fingerprint: String,
    required_capabilities: BTreeSet<CapabilityId>,
    attributes: BTreeMap<AttributeId, SemanticValue>,
    selection: ProviderSelection,
    provider_resources: ProviderResourcePlan,
    values: Vec<ResolvedValueBinding>,
    exact_aliases: Vec<PlanExactAlias>,
    state_effects: Vec<PlanStateEffect>,
    scratch_resource: Option<ResourceId>,
    persistent_resource: Option<ResourceId>,
    resources: Vec<ResourceId>,
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
    resource_id: ResourceId,
    per_instance_bytes: u64,
    instance_stride_bytes: u64,
    instance_count: u32,
    size_bytes: u64,
    alignment_bytes: u64,
    usage: BufferUsage,
    element_type: ElementType,
    lifetime: AllocationLifetime,
    kind: AllocationKind,
    storage: DynamicStorageContract,
}

impl ResourceAllocation {
    fn new(
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

pub const MAX_PROVIDER_WORKSPACE_SHAPE_BUCKETS: usize = 64;

/// Internal evaluator dimensions. Public callers supply token/page evidence;
/// only core lowers that evidence into aggregate formula inputs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct DynamicResourceShape {
    sequences: u32,
    tokens: u64,
    pages: u64,
}

impl DynamicResourceShape {
    pub(crate) fn new(sequences: u32, tokens: u64, pages: u64) -> Result<Self, VNextError> {
        if sequences == 0 || tokens == 0 || pages == 0 {
            return Err(invalid_plan(
                "dynamic resource shape dimensions must be non-zero",
            ));
        }
        Ok(Self {
            sequences,
            tokens,
            pages,
        })
    }

    pub(crate) const fn sequences(self) -> u32 {
        self.sequences
    }

    pub(crate) const fn tokens(self) -> u64 {
        self.tokens
    }

    pub(crate) const fn pages(self) -> u64 {
        self.pages
    }

    pub(crate) const fn from_validated(sequences: u32, tokens: u64, pages: u64) -> Self {
        Self {
            sequences,
            tokens,
            pages,
        }
    }
}

/// Evidence for one non-empty immediate token span inside an exact full input.
/// Counts are derived from the supplied token slice and private range rather
/// than accepted as caller-provided aggregate dimensions.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct TokenSpanWork {
    immediate_tokens: u64,
    full_input_tokens: u64,
    fingerprint: String,
}

impl TokenSpanWork {
    pub fn from_token_ids(
        full_input: &[u32],
        immediate_range: Range<usize>,
    ) -> Result<Self, VNextError> {
        if full_input.is_empty()
            || immediate_range.start >= immediate_range.end
            || immediate_range.end > full_input.len()
        {
            return Err(invalid_plan(
                "token work requires a non-empty in-bounds immediate span",
            ));
        }
        let immediate_tokens = u64::try_from(immediate_range.len())
            .map_err(|_| invalid_plan("immediate token span exceeds u64"))?;
        let full_input_tokens = u64::try_from(full_input.len())
            .map_err(|_| invalid_plan("full token input exceeds u64"))?;
        let mut digest = Sha256::new();
        digest.update(b"ferrum.runtime-vnext.token-span-work.v1\0");
        digest.update(full_input_tokens.to_le_bytes());
        digest.update(
            u64::try_from(immediate_range.start)
                .map_err(|_| invalid_plan("token span start exceeds u64"))?
                .to_le_bytes(),
        );
        digest.update(
            u64::try_from(immediate_range.end)
                .map_err(|_| invalid_plan("token span end exceeds u64"))?
                .to_le_bytes(),
        );
        for token in full_input {
            digest.update(token.to_le_bytes());
        }
        Ok(Self {
            immediate_tokens,
            full_input_tokens,
            fingerprint: format!("{:x}", digest.finalize()),
        })
    }

    pub const fn immediate_tokens(&self) -> u64 {
        self.immediate_tokens
    }

    pub const fn full_input_tokens(&self) -> u64 {
        self.full_input_tokens
    }

    pub fn fingerprint(&self) -> &str {
        &self.fingerprint
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub(crate) struct CommittedPageWork {
    pages: u64,
    fingerprint: String,
}

impl CommittedPageWork {
    pub(crate) fn new(pages: u64, fingerprint: String) -> Result<Self, VNextError> {
        if pages == 0
            || fingerprint.len() != 64
            || !fingerprint
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        {
            return Err(invalid_plan("committed page work evidence is invalid"));
        }
        Ok(Self { pages, fingerprint })
    }

    pub(crate) const fn pages(&self) -> u64 {
        self.pages
    }
}

/// Typed shape evidence shared by scoped admission and provider formula
/// evaluation. The aggregate dimensions and fingerprint are core-derived.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ResourceWorkShape {
    token_spans: Vec<TokenSpanWork>,
    committed_pages: Vec<CommittedPageWork>,
    immediate_sequences: u32,
    immediate_tokens: u64,
    immediate_pages: u64,
    fit_sequences: u32,
    fit_tokens: u64,
    fit_pages: u64,
    fingerprint: String,
}

impl ResourceWorkShape {
    pub fn from_token_spans(token_spans: Vec<TokenSpanWork>) -> Result<Self, VNextError> {
        Self::from_sources(token_spans, Vec::new())
    }

    pub fn single(token_span: TokenSpanWork) -> Result<Self, VNextError> {
        Self::from_token_spans(vec![token_span])
    }

    pub(crate) fn from_sources(
        token_spans: Vec<TokenSpanWork>,
        committed_pages: Vec<CommittedPageWork>,
    ) -> Result<Self, VNextError> {
        if token_spans.is_empty() {
            return Err(invalid_plan("resource work requires token evidence"));
        }
        let immediate_sequences = u32::try_from(token_spans.len())
            .map_err(|_| invalid_plan("resource work sequence count exceeds u32"))?;
        let immediate_tokens = token_spans.iter().try_fold(0_u64, |total, span| {
            total
                .checked_add(span.immediate_tokens())
                .ok_or_else(|| invalid_plan("resource work immediate tokens overflow u64"))
        })?;
        let fit_tokens = token_spans.iter().try_fold(0_u64, |total, span| {
            total
                .checked_add(span.full_input_tokens())
                .ok_or_else(|| invalid_plan("resource work full-input tokens overflow u64"))
        })?;
        let pages = committed_pages.iter().try_fold(0_u64, |total, page_work| {
            total
                .checked_add(page_work.pages())
                .ok_or_else(|| invalid_plan("resource work pages overflow u64"))
        })?;
        #[derive(Serialize)]
        struct FingerprintInput<'a> {
            domain: &'static str,
            token_spans: &'a [TokenSpanWork],
            committed_pages: &'a [CommittedPageWork],
        }
        let bytes = serde_json::to_vec(&FingerprintInput {
            domain: "ferrum.runtime-vnext.resource-work-shape.v1",
            token_spans: &token_spans,
            committed_pages: &committed_pages,
        })
        .map_err(|error| invalid_plan(format!("resource work encode failed: {error}")))?;
        Ok(Self {
            token_spans,
            committed_pages,
            immediate_sequences,
            immediate_tokens,
            immediate_pages: pages,
            fit_sequences: immediate_sequences,
            fit_tokens,
            fit_pages: pages,
            fingerprint: format!("{:x}", Sha256::digest(bytes)),
        })
    }

    pub const fn immediate_sequences(&self) -> u32 {
        self.immediate_sequences
    }

    pub const fn immediate_tokens(&self) -> u64 {
        self.immediate_tokens
    }

    pub const fn immediate_pages(&self) -> u64 {
        self.immediate_pages
    }

    pub const fn fit_sequences(&self) -> u32 {
        self.fit_sequences
    }

    pub const fn fit_tokens(&self) -> u64 {
        self.fit_tokens
    }

    pub const fn fit_pages(&self) -> u64 {
        self.fit_pages
    }

    pub fn fingerprint(&self) -> &str {
        &self.fingerprint
    }

    pub(crate) const fn immediate_shape(&self) -> DynamicResourceShape {
        DynamicResourceShape::from_validated(
            self.immediate_sequences,
            self.immediate_tokens,
            self.immediate_pages,
        )
    }

    pub(crate) const fn fit_shape(&self) -> DynamicResourceShape {
        DynamicResourceShape::from_validated(self.fit_sequences, self.fit_tokens, self.fit_pages)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DynamicResourceShapeBucket {
    maximum_sequences: u32,
    maximum_tokens: u64,
    maximum_pages: u64,
    bytes: u64,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct DynamicResourceShapeBucketWire {
    maximum_sequences: u32,
    maximum_tokens: u64,
    maximum_pages: u64,
    bytes: u64,
}

impl DynamicResourceShapeBucket {
    pub fn new(
        maximum_sequences: u32,
        maximum_tokens: u64,
        maximum_pages: u64,
        bytes: u64,
    ) -> Result<Self, VNextError> {
        if maximum_sequences == 0 || maximum_tokens == 0 || maximum_pages == 0 || bytes == 0 {
            return Err(invalid_plan(
                "workspace shape bucket bounds and bytes must be non-zero",
            ));
        }
        Ok(Self {
            maximum_sequences,
            maximum_tokens,
            maximum_pages,
            bytes,
        })
    }

    fn covers(&self, shape: DynamicResourceShape) -> bool {
        shape.sequences <= self.maximum_sequences
            && shape.tokens <= self.maximum_tokens
            && shape.pages <= self.maximum_pages
    }

    pub const fn maximum_sequences(&self) -> u32 {
        self.maximum_sequences
    }

    pub const fn maximum_tokens(&self) -> u64 {
        self.maximum_tokens
    }

    pub const fn maximum_pages(&self) -> u64 {
        self.maximum_pages
    }

    pub const fn bytes(&self) -> u64 {
        self.bytes
    }
}

impl<'de> Deserialize<'de> for DynamicResourceShapeBucket {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = DynamicResourceShapeBucketWire::deserialize(deserializer)?;
        Self::new(
            wire.maximum_sequences,
            wire.maximum_tokens,
            wire.maximum_pages,
            wire.bytes,
        )
        .map_err(serde::de::Error::custom)
    }
}

/// Core-validated sizing formula supplied by a provider. Its maximum shape is
/// the boundary of one provider invocation, not the scheduler's global active
/// sequence ceiling; a scheduler may split larger ready sets into batches.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DynamicResourceDemand {
    Fixed {
        bytes: u64,
    },
    ActualSequences {
        bytes_per_sequence: u64,
        maximum_sequences: u32,
    },
    Tokens {
        bytes_per_token: u64,
        maximum_tokens: u64,
    },
    Pages {
        bytes_per_page: u64,
        maximum_pages: u64,
    },
    BoundedShapeBuckets {
        buckets: Vec<DynamicResourceShapeBucket>,
    },
}

#[derive(Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
enum DynamicResourceDemandWire {
    Fixed {
        bytes: u64,
    },
    ActualSequences {
        bytes_per_sequence: u64,
        maximum_sequences: u32,
    },
    Tokens {
        bytes_per_token: u64,
        maximum_tokens: u64,
    },
    Pages {
        bytes_per_page: u64,
        maximum_pages: u64,
    },
    BoundedShapeBuckets {
        buckets: Vec<DynamicResourceShapeBucket>,
    },
}

impl DynamicResourceDemand {
    pub fn fixed(bytes: u64) -> Result<Self, VNextError> {
        Self::validated(Self::Fixed { bytes })
    }

    pub fn actual_sequences(
        bytes_per_sequence: u64,
        maximum_sequences: u32,
    ) -> Result<Self, VNextError> {
        Self::validated(Self::ActualSequences {
            bytes_per_sequence,
            maximum_sequences,
        })
    }

    pub fn tokens(bytes_per_token: u64, maximum_tokens: u64) -> Result<Self, VNextError> {
        Self::validated(Self::Tokens {
            bytes_per_token,
            maximum_tokens,
        })
    }

    pub fn pages(bytes_per_page: u64, maximum_pages: u64) -> Result<Self, VNextError> {
        Self::validated(Self::Pages {
            bytes_per_page,
            maximum_pages,
        })
    }

    pub fn bounded_shape_buckets(
        buckets: Vec<DynamicResourceShapeBucket>,
    ) -> Result<Self, VNextError> {
        Self::validated(Self::BoundedShapeBuckets { buckets })
    }

    fn validated(demand: Self) -> Result<Self, VNextError> {
        demand.validate()?;
        Ok(demand)
    }

    fn validate(&self) -> Result<(), VNextError> {
        let valid = match self {
            Self::Fixed { bytes } => *bytes > 0,
            Self::ActualSequences {
                bytes_per_sequence,
                maximum_sequences,
            } => {
                *bytes_per_sequence > 0
                    && *maximum_sequences > 0
                    && bytes_per_sequence
                        .checked_mul(u64::from(*maximum_sequences))
                        .is_some()
            }
            Self::Tokens {
                bytes_per_token,
                maximum_tokens,
            } => {
                *bytes_per_token > 0
                    && *maximum_tokens > 0
                    && bytes_per_token.checked_mul(*maximum_tokens).is_some()
            }
            Self::Pages {
                bytes_per_page,
                maximum_pages,
            } => {
                *bytes_per_page > 0
                    && *maximum_pages > 0
                    && bytes_per_page.checked_mul(*maximum_pages).is_some()
            }
            Self::BoundedShapeBuckets { buckets } => {
                !buckets.is_empty()
                    && buckets.len() <= MAX_PROVIDER_WORKSPACE_SHAPE_BUCKETS
                    && buckets.windows(2).all(|pair| {
                        let previous = &pair[0];
                        let next = &pair[1];
                        next.maximum_sequences >= previous.maximum_sequences
                            && next.maximum_tokens >= previous.maximum_tokens
                            && next.maximum_pages >= previous.maximum_pages
                            && (next.maximum_sequences > previous.maximum_sequences
                                || next.maximum_tokens > previous.maximum_tokens
                                || next.maximum_pages > previous.maximum_pages)
                            && next.bytes >= previous.bytes
                    })
            }
        };
        if !valid {
            return Err(invalid_plan(
                "dynamic resource formula is zero, overflowing, or non-canonical",
            ));
        }
        Ok(())
    }

    pub fn evaluate_bytes(&self, work: &ResourceWorkShape) -> Result<u64, VNextError> {
        self.evaluate_shape_bytes(work.immediate_shape())
    }

    pub fn evaluate_fit_bytes(&self, work: &ResourceWorkShape) -> Result<u64, VNextError> {
        self.evaluate_shape_bytes(work.fit_shape())
    }

    pub(crate) fn evaluate_shape_bytes(
        &self,
        shape: DynamicResourceShape,
    ) -> Result<u64, VNextError> {
        self.validate()?;
        let bytes = match self {
            Self::Fixed { bytes } => *bytes,
            Self::ActualSequences {
                bytes_per_sequence,
                maximum_sequences,
            } if shape.sequences <= *maximum_sequences => bytes_per_sequence
                .checked_mul(u64::from(shape.sequences))
                .ok_or_else(|| invalid_plan("sequence-scaled resource request overflows u64"))?,
            Self::Tokens {
                bytes_per_token,
                maximum_tokens,
            } if shape.tokens <= *maximum_tokens => bytes_per_token
                .checked_mul(shape.tokens)
                .ok_or_else(|| invalid_plan("token-scaled resource request overflows u64"))?,
            Self::Pages {
                bytes_per_page,
                maximum_pages,
            } if shape.pages <= *maximum_pages => bytes_per_page
                .checked_mul(shape.pages)
                .ok_or_else(|| invalid_plan("page-scaled resource request overflows u64"))?,
            Self::BoundedShapeBuckets { buckets } => buckets
                .iter()
                .find(|bucket| bucket.covers(shape))
                .map(DynamicResourceShapeBucket::bytes)
                .ok_or_else(|| invalid_plan("actual invocation shape exceeds workspace buckets"))?,
            _ => {
                return Err(invalid_plan(
                    "actual invocation shape exceeds its bounded resource formula",
                ))
            }
        };
        if bytes == 0 {
            return Err(invalid_plan(
                "dynamic resource request evaluates to zero bytes",
            ));
        }
        Ok(bytes)
    }

    pub(crate) fn minimum_shape(&self) -> DynamicResourceShape {
        DynamicResourceShape {
            sequences: 1,
            tokens: 1,
            pages: 1,
        }
    }

    pub(crate) fn theoretical_maximum_shape(&self) -> DynamicResourceShape {
        match self {
            Self::Fixed { .. } => self.minimum_shape(),
            Self::ActualSequences {
                maximum_sequences, ..
            } => DynamicResourceShape {
                sequences: *maximum_sequences,
                tokens: 1,
                pages: 1,
            },
            Self::Tokens { maximum_tokens, .. } => DynamicResourceShape {
                sequences: 1,
                tokens: *maximum_tokens,
                pages: 1,
            },
            Self::Pages { maximum_pages, .. } => DynamicResourceShape {
                sequences: 1,
                tokens: 1,
                pages: *maximum_pages,
            },
            Self::BoundedShapeBuckets { buckets } => {
                let bucket = buckets
                    .last()
                    .expect("validated bounded formula has at least one bucket");
                DynamicResourceShape {
                    sequences: bucket.maximum_sequences,
                    tokens: bucket.maximum_tokens,
                    pages: bucket.maximum_pages,
                }
            }
        }
    }

    fn is_fixed(&self) -> bool {
        matches!(self, Self::Fixed { .. })
    }

    fn is_valid_for_sequence_scope(&self) -> bool {
        match self {
            Self::Fixed { .. } | Self::Tokens { .. } | Self::Pages { .. } => true,
            Self::BoundedShapeBuckets { buckets } => {
                buckets.iter().all(|bucket| bucket.maximum_sequences == 1)
            }
            Self::ActualSequences { .. } => false,
        }
    }
}

impl<'de> Deserialize<'de> for DynamicResourceDemand {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let demand = match DynamicResourceDemandWire::deserialize(deserializer)? {
            DynamicResourceDemandWire::Fixed { bytes } => Self::Fixed { bytes },
            DynamicResourceDemandWire::ActualSequences {
                bytes_per_sequence,
                maximum_sequences,
            } => Self::ActualSequences {
                bytes_per_sequence,
                maximum_sequences,
            },
            DynamicResourceDemandWire::Tokens {
                bytes_per_token,
                maximum_tokens,
            } => Self::Tokens {
                bytes_per_token,
                maximum_tokens,
            },
            DynamicResourceDemandWire::Pages {
                bytes_per_page,
                maximum_pages,
            } => Self::Pages {
                bytes_per_page,
                maximum_pages,
            },
            DynamicResourceDemandWire::BoundedShapeBuckets { buckets } => {
                Self::BoundedShapeBuckets { buckets }
            }
        };
        Self::validated(demand).map_err(serde::de::Error::custom)
    }
}

pub type ProviderWorkspaceSizeFormula = DynamicResourceDemand;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
#[serde(transparent)]
pub struct DynamicBackingPoolId(String);

impl DynamicBackingPoolId {
    fn from_compatibility(key: &PoolCompatibilityKey) -> Result<Self, VNextError> {
        Ok(Self(format!(
            "dynamic-pool/sha256/{}",
            canonical_fingerprint(key, "fingerprint dynamic pool compatibility")?
        )))
    }

    fn validate(&self) -> Result<(), VNextError> {
        let Some(hash) = self.0.strip_prefix("dynamic-pool/sha256/") else {
            return Err(invalid_plan(
                "dynamic backing pool id has an invalid prefix",
            ));
        };
        if !is_canonical_sha256(hash) {
            return Err(invalid_plan("dynamic backing pool id has an invalid hash"));
        }
        Ok(())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl<'de> Deserialize<'de> for DynamicBackingPoolId {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let id = Self(String::deserialize(deserializer)?);
        id.validate().map_err(serde::de::Error::custom)?;
        Ok(id)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct DynamicStorageContract {
    profile: DynamicStorageProfile,
    logical_layout_fingerprint: String,
}

impl DynamicStorageContract {
    fn new(
        profile: DynamicStorageProfile,
        logical_layout_fingerprint: String,
    ) -> Result<Self, VNextError> {
        if !is_canonical_sha256(&logical_layout_fingerprint) {
            return Err(invalid_plan(
                "dynamic storage logical layout fingerprint is invalid",
            ));
        }
        Ok(Self {
            profile,
            logical_layout_fingerprint,
        })
    }

    pub const fn profile(&self) -> DynamicStorageProfile {
        self.profile
    }

    pub fn logical_layout_fingerprint(&self) -> &str {
        &self.logical_layout_fingerprint
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct DynamicStorageContractWire {
    profile: DynamicStorageProfile,
    logical_layout_fingerprint: String,
}

impl<'de> Deserialize<'de> for DynamicStorageContract {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = DynamicStorageContractWire::deserialize(deserializer)?;
        Self::new(wire.profile, wire.logical_layout_fingerprint).map_err(serde::de::Error::custom)
    }
}

#[derive(Serialize)]
#[serde(rename_all = "snake_case")]
enum TensorStorageLayoutClass<'a> {
    Contiguous,
    Strided {
        byte_strides: &'a [u64],
    },
    Blocked {
        block: &'a [u64],
        axis_order: &'a [u32],
        padding: BlockedStoragePaddingClass,
    },
}

#[derive(Serialize)]
#[serde(rename_all = "snake_case")]
enum BlockedStoragePaddingClass {
    Exact,
    ZeroFill,
}

#[derive(Serialize)]
#[serde(rename_all = "snake_case")]
enum WorkspaceStorageLayoutClass {
    OpaqueBytesV1,
}

fn tensor_storage_layout_fingerprint(layout: &ResolvedTensorLayout) -> Result<String, VNextError> {
    let class = match layout {
        ResolvedTensorLayout::Contiguous => TensorStorageLayoutClass::Contiguous,
        ResolvedTensorLayout::Strided { byte_strides } => {
            TensorStorageLayoutClass::Strided { byte_strides }
        }
        ResolvedTensorLayout::Blocked {
            block,
            axis_order,
            padding,
        } => TensorStorageLayoutClass::Blocked {
            block,
            axis_order,
            padding: match padding {
                BlockedTensorPadding::Exact => BlockedStoragePaddingClass::Exact,
                BlockedTensorPadding::ZeroFill { .. } => BlockedStoragePaddingClass::ZeroFill,
            },
        },
    };
    canonical_fingerprint(&class, "fingerprint tensor storage layout class")
}

fn workspace_storage_layout_fingerprint() -> Result<String, VNextError> {
    canonical_fingerprint(
        &WorkspaceStorageLayoutClass::OpaqueBytesV1,
        "fingerprint workspace storage layout class",
    )
}

fn static_contiguous_storage_profile() -> Result<DynamicStorageProfile, VNextError> {
    DynamicStorageProfile::new(
        super::DynamicStorageAllocator::LinearArena,
        super::DynamicStorageView::Contiguous,
    )
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct PoolCompatibilityKey {
    version: ContractVersion,
    profile: DynamicStorageProfile,
    usage: BufferUsage,
    element_type: ElementType,
    logical_layout_fingerprint: String,
    alignment_bytes: u64,
}

impl PoolCompatibilityKey {
    fn new(
        storage: &DynamicStorageContract,
        usage: BufferUsage,
        element_type: ElementType,
        alignment_bytes: u64,
    ) -> Result<Self, VNextError> {
        if alignment_bytes == 0 || !alignment_bytes.is_power_of_two() {
            return Err(invalid_plan(
                "dynamic pool compatibility alignment is invalid",
            ));
        }
        let key = Self {
            version: ContractVersion::new(1, 0),
            profile: storage.profile,
            usage,
            element_type,
            logical_layout_fingerprint: storage.logical_layout_fingerprint.clone(),
            alignment_bytes,
        };
        key.validate()?;
        Ok(key)
    }

    fn validate(&self) -> Result<(), VNextError> {
        if self.version != ContractVersion::new(1, 0)
            || !is_canonical_sha256(&self.logical_layout_fingerprint)
            || self.alignment_bytes == 0
            || !self.alignment_bytes.is_power_of_two()
        {
            return Err(invalid_plan("dynamic pool compatibility key is invalid"));
        }
        Ok(())
    }

    pub const fn profile(&self) -> DynamicStorageProfile {
        self.profile
    }

    pub const fn usage(&self) -> BufferUsage {
        self.usage
    }

    pub const fn element_type(&self) -> ElementType {
        self.element_type
    }

    pub fn logical_layout_fingerprint(&self) -> &str {
        &self.logical_layout_fingerprint
    }

    pub const fn alignment_bytes(&self) -> u64 {
        self.alignment_bytes
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct PoolCompatibilityKeyWire {
    version: ContractVersion,
    profile: DynamicStorageProfile,
    usage: BufferUsage,
    element_type: ElementType,
    logical_layout_fingerprint: String,
    alignment_bytes: u64,
}

impl<'de> Deserialize<'de> for PoolCompatibilityKey {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = PoolCompatibilityKeyWire::deserialize(deserializer)?;
        let key = Self {
            version: wire.version,
            profile: wire.profile,
            usage: wire.usage,
            element_type: wire.element_type,
            logical_layout_fingerprint: wire.logical_layout_fingerprint,
            alignment_bytes: wire.alignment_bytes,
        };
        key.validate().map_err(serde::de::Error::custom)?;
        Ok(key)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DynamicPoolProvisioningMode {
    DemandDrivenElastic,
}

/// Typed bounds for elastic residency. `minimum_resident_bytes` is the amount
/// required to make one request runnable, not an initial reservation. Pools
/// may grow on demand up to `maximum_resident_bytes`; the process-wide device
/// account remains the authority when several pools compete for that memory.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DynamicPoolProvisioningPolicy {
    mode: DynamicPoolProvisioningMode,
    minimum_resident_bytes: u64,
    maximum_resident_bytes: u64,
}

impl DynamicPoolProvisioningPolicy {
    fn demand_driven(
        minimum_resident_bytes: u64,
        maximum_resident_bytes: u64,
    ) -> Result<Self, VNextError> {
        let policy = Self {
            mode: DynamicPoolProvisioningMode::DemandDrivenElastic,
            minimum_resident_bytes,
            maximum_resident_bytes,
        };
        policy.validate()?;
        Ok(policy)
    }

    fn validate(&self) -> Result<(), VNextError> {
        if self.minimum_resident_bytes == 0
            || self.maximum_resident_bytes < self.minimum_resident_bytes
        {
            return Err(invalid_plan("dynamic pool provisioning bounds are invalid"));
        }
        Ok(())
    }

    pub const fn mode(&self) -> DynamicPoolProvisioningMode {
        self.mode
    }

    pub const fn minimum_resident_bytes(&self) -> u64 {
        self.minimum_resident_bytes
    }

    pub const fn maximum_resident_bytes(&self) -> u64 {
        self.maximum_resident_bytes
    }
}

/// One self-contained physical-compatibility class for demand-driven backing.
/// Membership, runnable minima, completion-order reuse evidence, and elastic
/// bounds are canonical plan data, so a runtime does not have to rediscover
/// pool structure by scanning unrelated descriptors.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct DynamicBackingPoolSpec {
    pool_id: DynamicBackingPoolId,
    compatibility: PoolCompatibilityKey,
    resource_ids: Vec<ResourceId>,
    minimum_request_bytes: u64,
    minimum_sequence_bytes: u64,
    minimum_step_bytes: u64,
    minimum_invocation_peak_bytes: u64,
    theoretical_ceiling_bytes: CanonicalU128,
    provisioning: DynamicPoolProvisioningPolicy,
    invocation_liveness_mode: InvocationLivenessMode,
    invocation_liveness: Vec<InvocationResourceLiveness>,
}

impl DynamicBackingPoolSpec {
    #[allow(clippy::too_many_arguments)]
    fn from_core(
        compatibility: PoolCompatibilityKey,
        resource_ids: Vec<ResourceId>,
        minimum_request_bytes: u64,
        minimum_sequence_bytes: u64,
        minimum_step_bytes: u64,
        minimum_invocation_peak_bytes: u64,
        theoretical_ceiling_bytes: u128,
        dynamic_capacity_bytes: u64,
        invocation_liveness_mode: InvocationLivenessMode,
        invocation_liveness: Vec<InvocationResourceLiveness>,
    ) -> Result<Self, VNextError> {
        compatibility.validate()?;
        let pool_id = DynamicBackingPoolId::from_compatibility(&compatibility)?;
        let minimum_resident_bytes = minimum_request_bytes
            .checked_add(minimum_sequence_bytes)
            .and_then(|bytes| bytes.checked_add(minimum_step_bytes))
            .and_then(|bytes| bytes.checked_add(minimum_invocation_peak_bytes))
            .ok_or_else(|| invalid_plan("dynamic pool runnable minimum overflows u64"))?;
        let maximum_resident_bytes =
            u64::try_from(theoretical_ceiling_bytes.min(u128::from(dynamic_capacity_bytes)))
                .map_err(|_| invalid_plan("dynamic pool resident ceiling exceeds u64"))?;
        let spec = Self {
            pool_id,
            compatibility,
            resource_ids,
            minimum_request_bytes,
            minimum_sequence_bytes,
            minimum_step_bytes,
            minimum_invocation_peak_bytes,
            theoretical_ceiling_bytes: CanonicalU128::new(theoretical_ceiling_bytes),
            provisioning: DynamicPoolProvisioningPolicy::demand_driven(
                minimum_resident_bytes,
                maximum_resident_bytes,
            )?,
            invocation_liveness_mode,
            invocation_liveness,
        };
        spec.validate_local()?;
        Ok(spec)
    }

    fn validate_local(&self) -> Result<(), VNextError> {
        self.pool_id.validate()?;
        self.compatibility.validate()?;
        self.provisioning.validate()?;
        let minimum_resident_bytes = self
            .minimum_request_bytes
            .checked_add(self.minimum_sequence_bytes)
            .and_then(|bytes| bytes.checked_add(self.minimum_step_bytes))
            .and_then(|bytes| bytes.checked_add(self.minimum_invocation_peak_bytes))
            .ok_or_else(|| invalid_plan("dynamic pool runnable minimum overflows u64"))?;
        if self.pool_id != DynamicBackingPoolId::from_compatibility(&self.compatibility)?
            || self.resource_ids.is_empty()
            || self.resource_ids.windows(2).any(|pair| pair[0] >= pair[1])
            || minimum_resident_bytes != self.provisioning.minimum_resident_bytes
            || u128::from(self.provisioning.maximum_resident_bytes)
                > self.theoretical_ceiling_bytes.get()
        {
            return Err(invalid_plan(
                "dynamic backing pool identity, membership, or bounds are invalid",
            ));
        }
        match self.invocation_liveness_mode {
            InvocationLivenessMode::NoInvocationResources => {
                if self.minimum_invocation_peak_bytes != 0 || !self.invocation_liveness.is_empty() {
                    return Err(invalid_plan(
                        "non-invocation pool carries invocation liveness evidence",
                    ));
                }
            }
            InvocationLivenessMode::TotalOrderReuse
            | InvocationLivenessMode::ConservativeConcurrent => {
                if self.minimum_invocation_peak_bytes == 0
                    || self.invocation_liveness.is_empty()
                    || self
                        .invocation_liveness
                        .windows(2)
                        .any(|pair| pair[0].node_id >= pair[1].node_id)
                {
                    return Err(invalid_plan(
                        "invocation pool liveness evidence is empty or non-canonical",
                    ));
                }
            }
        }
        Ok(())
    }

    pub fn pool_id(&self) -> &DynamicBackingPoolId {
        &self.pool_id
    }

    pub fn compatibility(&self) -> &PoolCompatibilityKey {
        &self.compatibility
    }

    pub fn resource_ids(&self) -> &[ResourceId] {
        &self.resource_ids
    }

    pub const fn minimum_request_bytes(&self) -> u64 {
        self.minimum_request_bytes
    }

    pub const fn minimum_sequence_bytes(&self) -> u64 {
        self.minimum_sequence_bytes
    }

    pub const fn minimum_step_bytes(&self) -> u64 {
        self.minimum_step_bytes
    }

    pub const fn minimum_invocation_peak_bytes(&self) -> u64 {
        self.minimum_invocation_peak_bytes
    }

    pub fn theoretical_ceiling_bytes(&self) -> u128 {
        self.theoretical_ceiling_bytes.get()
    }

    pub fn provisioning(&self) -> &DynamicPoolProvisioningPolicy {
        &self.provisioning
    }

    pub const fn invocation_liveness_mode(&self) -> InvocationLivenessMode {
        self.invocation_liveness_mode
    }

    pub fn invocation_liveness(&self) -> &[InvocationResourceLiveness] {
        &self.invocation_liveness
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct DynamicBackingPoolSpecWire {
    pool_id: DynamicBackingPoolId,
    compatibility: PoolCompatibilityKey,
    resource_ids: Vec<ResourceId>,
    minimum_request_bytes: u64,
    minimum_sequence_bytes: u64,
    minimum_step_bytes: u64,
    minimum_invocation_peak_bytes: u64,
    theoretical_ceiling_bytes: CanonicalU128,
    provisioning: DynamicPoolProvisioningPolicy,
    invocation_liveness_mode: InvocationLivenessMode,
    invocation_liveness: Vec<InvocationResourceLiveness>,
}

impl<'de> Deserialize<'de> for DynamicBackingPoolSpec {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = DynamicBackingPoolSpecWire::deserialize(deserializer)?;
        let spec = Self {
            pool_id: wire.pool_id,
            compatibility: wire.compatibility,
            resource_ids: wire.resource_ids,
            minimum_request_bytes: wire.minimum_request_bytes,
            minimum_sequence_bytes: wire.minimum_sequence_bytes,
            minimum_step_bytes: wire.minimum_step_bytes,
            minimum_invocation_peak_bytes: wire.minimum_invocation_peak_bytes,
            theoretical_ceiling_bytes: wire.theoretical_ceiling_bytes,
            provisioning: wire.provisioning,
            invocation_liveness_mode: wire.invocation_liveness_mode,
            invocation_liveness: wire.invocation_liveness,
        };
        spec.validate_local().map_err(serde::de::Error::custom)?;
        Ok(spec)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct DynamicResourceDescriptor {
    base_resource_id: ResourceId,
    demand: DynamicResourceDemand,
    alignment_bytes: u64,
    usage: BufferUsage,
    element_type: ElementType,
    lifetime: AllocationLifetime,
    kind: AllocationKind,
    storage: DynamicStorageContract,
    pool_id: DynamicBackingPoolId,
    /// Protocol-only ceiling used for checked evidence. No API may iterate,
    /// reserve, allocate, or claim this many instances.
    theoretical_maximum_instances: u32,
}

impl DynamicResourceDescriptor {
    #[allow(clippy::too_many_arguments)]
    fn new(
        base_resource_id: ResourceId,
        demand: DynamicResourceDemand,
        alignment_bytes: u64,
        usage: BufferUsage,
        element_type: ElementType,
        lifetime: AllocationLifetime,
        kind: AllocationKind,
        storage: DynamicStorageContract,
        theoretical_maximum_instances: u32,
    ) -> Result<Self, VNextError> {
        validate_active_sequence_ceiling(theoretical_maximum_instances)?;
        if alignment_bytes == 0
            || !alignment_bytes.is_power_of_two()
            || lifetime == AllocationLifetime::Plan
        {
            return Err(invalid_plan(
                "dynamic resource descriptor has invalid alignment or static lifetime",
            ));
        }
        let kind_valid = match &kind {
            AllocationKind::Scratch { .. } => {
                lifetime == AllocationLifetime::Invocation
                    && usage == BufferUsage::Scratch
                    && element_type == ElementType::U8
            }
            AllocationKind::Persistent { .. } => {
                matches!(
                    lifetime,
                    AllocationLifetime::Request
                        | AllocationLifetime::Sequence
                        | AllocationLifetime::Step
                ) && usage == BufferUsage::Persistent
                    && element_type == ElementType::U8
            }
            AllocationKind::Value => usage != BufferUsage::Weights,
        };
        if !kind_valid {
            return Err(invalid_plan(
                "dynamic resource kind, lifetime, usage, or element type is inconsistent",
            ));
        }
        demand.validate()?;
        let pool_id = DynamicBackingPoolId::from_compatibility(&PoolCompatibilityKey::new(
            &storage,
            usage,
            element_type,
            alignment_bytes,
        )?)?;
        let descriptor = Self {
            base_resource_id,
            demand,
            alignment_bytes,
            usage,
            element_type,
            lifetime,
            kind,
            storage,
            pool_id,
            theoretical_maximum_instances,
        };
        descriptor.evaluate_request_bytes_for_shape(descriptor.demand.minimum_shape())?;
        descriptor
            .evaluate_request_bytes_for_shape(descriptor.demand.theoretical_maximum_shape())?;
        Ok(descriptor)
    }

    pub fn base_resource_id(&self) -> &ResourceId {
        &self.base_resource_id
    }

    pub fn demand(&self) -> &DynamicResourceDemand {
        &self.demand
    }

    pub const fn theoretical_maximum_instances(&self) -> u32 {
        self.theoretical_maximum_instances
    }

    pub fn evaluate_logical_request_bytes(
        &self,
        work: &ResourceWorkShape,
    ) -> Result<u64, VNextError> {
        self.demand.evaluate_bytes(work)
    }

    pub(crate) fn evaluate_logical_request_bytes_for_shape(
        &self,
        shape: DynamicResourceShape,
    ) -> Result<u64, VNextError> {
        self.demand.evaluate_shape_bytes(shape)
    }

    pub fn physical_allocation_quantum_bytes(&self) -> u64 {
        match self.storage.profile().allocator() {
            super::DynamicStorageAllocator::LinearArena => self.alignment_bytes,
            super::DynamicStorageAllocator::FixedBlockArena { block_bytes } => {
                block_bytes.max(self.alignment_bytes)
            }
        }
    }

    /// Exact physical claim for one logical shape. The semantic demand stays
    /// unchanged in the plan; allocator geometry is applied only at this
    /// boundary so admission and backing cannot under-count fixed blocks.
    pub fn evaluate_request_bytes(&self, work: &ResourceWorkShape) -> Result<u64, VNextError> {
        self.evaluate_request_bytes_for_shape(work.immediate_shape())
    }

    pub fn evaluate_fit_request_bytes(&self, work: &ResourceWorkShape) -> Result<u64, VNextError> {
        self.evaluate_request_bytes_for_shape(work.fit_shape())
    }

    pub(crate) fn evaluate_request_bytes_for_shape(
        &self,
        shape: DynamicResourceShape,
    ) -> Result<u64, VNextError> {
        quantize_storage_bytes(
            self.evaluate_logical_request_bytes_for_shape(shape)?,
            self.alignment_bytes,
            self.storage.profile(),
        )
    }

    pub fn minimum_request_bytes(&self) -> Result<u64, VNextError> {
        self.evaluate_request_bytes_for_shape(self.demand.minimum_shape())
    }

    pub fn theoretical_maximum_request_bytes(&self) -> Result<u64, VNextError> {
        self.evaluate_request_bytes_for_shape(self.demand.theoretical_maximum_shape())
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

    pub fn pool_id(&self) -> &DynamicBackingPoolId {
        &self.pool_id
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct DynamicResourceDescriptorWire {
    base_resource_id: ResourceId,
    demand: DynamicResourceDemand,
    alignment_bytes: u64,
    usage: BufferUsage,
    element_type: ElementType,
    lifetime: AllocationLifetime,
    kind: AllocationKind,
    storage: DynamicStorageContract,
    pool_id: DynamicBackingPoolId,
    theoretical_maximum_instances: u32,
}

impl<'de> Deserialize<'de> for DynamicResourceDescriptor {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = DynamicResourceDescriptorWire::deserialize(deserializer)?;
        let descriptor = Self::new(
            wire.base_resource_id,
            wire.demand,
            wire.alignment_bytes,
            wire.usage,
            wire.element_type,
            wire.lifetime,
            wire.kind,
            wire.storage,
            wire.theoretical_maximum_instances,
        )
        .map_err(serde::de::Error::custom)?;
        if descriptor.pool_id != wire.pool_id {
            return Err(serde::de::Error::custom(
                "dynamic resource pool id is not core-derived from compatibility",
            ));
        }
        Ok(descriptor)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(transparent)]
struct CanonicalU128(String);

impl CanonicalU128 {
    fn new(value: u128) -> Self {
        Self(value.to_string())
    }

    fn get(&self) -> u128 {
        self.0
            .parse()
            .expect("canonical u128 is validated at construction or deserialization")
    }
}

impl<'de> Deserialize<'de> for CanonicalU128 {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        let parsed = value.parse::<u128>().map_err(serde::de::Error::custom)?;
        if parsed.to_string() != value {
            return Err(serde::de::Error::custom(
                "u128 evidence must be a canonical unsigned decimal string",
            ));
        }
        Ok(Self(value))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InvocationLivenessMode {
    NoInvocationResources,
    /// Every invocation row in this pool is ordered by a transitive node
    /// completion dependency, so one runnable request needs only the maximum
    /// row size rather than their sum.
    TotalOrderReuse,
    /// The plan cannot prove that every invocation row completes before the
    /// next one starts. The runnable minimum therefore sums member resources.
    ConservativeConcurrent,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InvocationResourceLiveness {
    node_id: NodeId,
    resource_ids: Vec<ResourceId>,
}

impl InvocationResourceLiveness {
    pub fn node_id(&self) -> &NodeId {
        &self.node_id
    }

    pub fn resource_ids(&self) -> &[ResourceId] {
        &self.resource_ids
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct MemoryPlan {
    device_capacity_bytes: u64,
    policy_capacity_bytes: u64,
    reserve_bytes: u64,
    usable_capacity_bytes: u64,
    maximum_active_sequences: u32,
    static_bytes: u64,
    minimum_request_bytes: u64,
    minimum_sequence_bytes: u64,
    minimum_step_bytes: u64,
    minimum_invocation_peak_bytes: u64,
    minimum_runnable_request_bytes: u64,
    theoretical_ceiling_bytes: CanonicalU128,
    static_allocations: Vec<ResourceAllocation>,
    dynamic_descriptors: Vec<DynamicResourceDescriptor>,
    dynamic_pools: Vec<DynamicBackingPoolSpec>,
    invocation_liveness_mode: InvocationLivenessMode,
    invocation_liveness: Vec<InvocationResourceLiveness>,
}

impl MemoryPlan {
    fn from_core(
        device_capacity_bytes: u64,
        policy_capacity_bytes: u64,
        reserve_bytes: u64,
        maximum_active_sequences: u32,
        mut static_allocations: Vec<ResourceAllocation>,
        mut dynamic_descriptors: Vec<DynamicResourceDescriptor>,
        nodes: &[PlanNode],
    ) -> Result<Self, VNextError> {
        if static_allocations.len() + dynamic_descriptors.len() > MAX_EXECUTION_PLAN_RESOURCE_ROWS {
            return Err(invalid_plan(format!(
                "execution plan resource rows exceed {MAX_EXECUTION_PLAN_RESOURCE_ROWS}"
            )));
        }
        let usable_capacity_bytes = policy_capacity_bytes
            .checked_sub(reserve_bytes)
            .ok_or_else(|| invalid_plan("memory reserve exceeds the policy capacity"))?;
        static_allocations.sort_by(|left, right| left.resource_id.cmp(&right.resource_id));
        dynamic_descriptors
            .sort_by(|left, right| left.base_resource_id.cmp(&right.base_resource_id));
        let static_bytes = static_allocations
            .iter()
            .try_fold(0_u64, |total, allocation| {
                total
                    .checked_add(allocation.size_bytes)
                    .ok_or_else(|| invalid_plan("static memory total overflows u64"))
            })?;
        let dynamic_capacity_bytes = usable_capacity_bytes
            .checked_sub(static_bytes)
            .ok_or_else(|| invalid_plan("static memory exceeds usable capacity"))?;
        let dynamic_pools =
            Self::derive_dynamic_pools(&dynamic_descriptors, nodes, dynamic_capacity_bytes)?;
        let (invocation_liveness_mode, invocation_liveness) =
            Self::summarize_pool_invocation_liveness(&dynamic_pools)?;
        let minimum_request_bytes = dynamic_pools.iter().try_fold(0_u64, |total, pool| {
            total
                .checked_add(pool.minimum_request_bytes)
                .ok_or_else(|| invalid_plan("minimum request bytes overflow u64"))
        })?;
        let minimum_sequence_bytes = dynamic_pools.iter().try_fold(0_u64, |total, pool| {
            total
                .checked_add(pool.minimum_sequence_bytes)
                .ok_or_else(|| invalid_plan("minimum sequence bytes overflow u64"))
        })?;
        let minimum_step_bytes = dynamic_pools.iter().try_fold(0_u64, |total, pool| {
            total
                .checked_add(pool.minimum_step_bytes)
                .ok_or_else(|| invalid_plan("minimum step bytes overflow u64"))
        })?;
        let minimum_invocation_peak_bytes =
            dynamic_pools.iter().try_fold(0_u64, |total, pool| {
                total
                    .checked_add(pool.minimum_invocation_peak_bytes)
                    .ok_or_else(|| invalid_plan("invocation pool minimum bytes overflow u64"))
            })?;
        let minimum_runnable_request_bytes = minimum_request_bytes
            .checked_add(minimum_sequence_bytes)
            .and_then(|bytes| bytes.checked_add(minimum_step_bytes))
            .and_then(|bytes| bytes.checked_add(minimum_invocation_peak_bytes))
            .ok_or_else(|| invalid_plan("minimum runnable request bytes overflow u64"))?;
        let theoretical_dynamic_ceiling =
            dynamic_pools.iter().try_fold(0_u128, |total, pool| {
                total
                    .checked_add(pool.theoretical_ceiling_bytes.get())
                    .ok_or_else(|| invalid_plan("dynamic theoretical total overflows u128"))
            })?;
        let theoretical_ceiling_bytes = u128::from(static_bytes)
            .checked_add(theoretical_dynamic_ceiling)
            .ok_or_else(|| invalid_plan("plan theoretical ceiling overflows u128"))?;
        let minimum_runnable_bytes = static_bytes
            .checked_add(minimum_runnable_request_bytes)
            .ok_or_else(|| invalid_plan("minimum runnable plan bytes overflow u64"))?;
        if minimum_runnable_bytes > usable_capacity_bytes {
            return Err(invalid_plan(format!(
                "minimum runnable plan requires {minimum_runnable_bytes} bytes, exceeding usable capacity {usable_capacity_bytes}"
            )));
        }
        let plan = Self {
            device_capacity_bytes,
            policy_capacity_bytes,
            reserve_bytes,
            usable_capacity_bytes,
            maximum_active_sequences,
            static_bytes,
            minimum_request_bytes,
            minimum_sequence_bytes,
            minimum_step_bytes,
            minimum_invocation_peak_bytes,
            minimum_runnable_request_bytes,
            theoretical_ceiling_bytes: CanonicalU128::new(theoretical_ceiling_bytes),
            static_allocations,
            dynamic_descriptors,
            dynamic_pools,
            invocation_liveness_mode,
            invocation_liveness,
        };
        plan.validate()?;
        Ok(plan)
    }

    fn validate(&self) -> Result<(), VNextError> {
        validate_active_sequence_ceiling(self.maximum_active_sequences)?;
        if self.static_allocations.len() + self.dynamic_descriptors.len()
            > MAX_EXECUTION_PLAN_RESOURCE_ROWS
        {
            return Err(invalid_plan(format!(
                "execution plan resource rows exceed {MAX_EXECUTION_PLAN_RESOURCE_ROWS}"
            )));
        }
        if self
            .static_allocations
            .windows(2)
            .any(|pair| pair[0].resource_id >= pair[1].resource_id)
            || self
                .dynamic_descriptors
                .windows(2)
                .any(|pair| pair[0].base_resource_id >= pair[1].base_resource_id)
            || self
                .dynamic_pools
                .windows(2)
                .any(|pair| pair[0].pool_id >= pair[1].pool_id)
        {
            return Err(invalid_plan("memory plan resource rows are not canonical"));
        }
        if self.device_capacity_bytes == 0
            || self.policy_capacity_bytes == 0
            || self.policy_capacity_bytes > self.device_capacity_bytes
            || self.reserve_bytes >= self.policy_capacity_bytes
            || self.usable_capacity_bytes
                != self
                    .policy_capacity_bytes
                    .checked_sub(self.reserve_bytes)
                    .ok_or_else(|| invalid_plan("memory reserve underflows policy capacity"))?
        {
            return Err(invalid_plan("memory plan capacity or reserve is invalid"));
        }
        let mut resources = BTreeSet::new();
        let workspace_layout_fingerprint = workspace_storage_layout_fingerprint()?;
        let static_contiguous_profile = static_contiguous_storage_profile()?;
        let actual_static =
            self.static_allocations
                .iter()
                .try_fold(0_u64, |total, allocation| {
                    if !resources.insert(allocation.resource_id.clone())
                        || allocation.per_instance_bytes == 0
                        || allocation.instance_stride_bytes
                            != quantize_storage_bytes(
                                allocation.per_instance_bytes,
                                allocation.alignment_bytes,
                                allocation.storage.profile(),
                            )?
                        || allocation.instance_count != 1
                        || allocation.size_bytes != allocation.instance_stride_bytes
                        || allocation.lifetime != AllocationLifetime::Plan
                        || match &allocation.kind {
                            AllocationKind::Value => {
                                allocation.storage.profile() != static_contiguous_profile
                            }
                            AllocationKind::Persistent { .. } => {
                                allocation.usage != BufferUsage::Persistent
                                    || allocation.element_type != ElementType::U8
                                    || allocation.storage.logical_layout_fingerprint()
                                        != workspace_layout_fingerprint
                            }
                            AllocationKind::Scratch { .. } => true,
                        }
                    {
                        return Err(invalid_plan(format!(
                            "static resource `{}` is duplicate or invalid",
                            allocation.resource_id
                        )));
                    }
                    total
                        .checked_add(allocation.size_bytes)
                        .ok_or_else(|| invalid_plan("static allocation total overflows u64"))
                })?;
        if actual_static != self.static_bytes {
            return Err(invalid_plan("static byte total is not core-derived"));
        }
        let mut actual_theoretical_dynamic = 0_u128;
        let dynamic_by_id = self
            .dynamic_descriptors
            .iter()
            .map(|descriptor| (descriptor.base_resource_id.clone(), descriptor))
            .collect::<BTreeMap<_, _>>();
        for descriptor in &self.dynamic_descriptors {
            if !resources.insert(descriptor.base_resource_id.clone()) {
                return Err(invalid_plan(format!(
                    "dynamic resource `{}` is duplicated",
                    descriptor.base_resource_id
                )));
            }
            validate_active_sequence_ceiling(descriptor.theoretical_maximum_instances)?;
            descriptor.demand.validate()?;
            if descriptor.lifetime == AllocationLifetime::Plan {
                return Err(invalid_plan(
                    "dynamic descriptor cannot have plan-static lifetime",
                ));
            }
            actual_theoretical_dynamic = actual_theoretical_dynamic
                .checked_add(
                    u128::from(descriptor.theoretical_maximum_request_bytes()?)
                        * u128::from(descriptor.theoretical_maximum_instances),
                )
                .ok_or_else(|| invalid_plan("dynamic ceiling total overflows u128"))?;
        }
        let actual_theoretical = u128::from(actual_static)
            .checked_add(actual_theoretical_dynamic)
            .ok_or_else(|| invalid_plan("theoretical plan ceiling overflows u128"))?;
        let dynamic_capacity_bytes = self
            .usable_capacity_bytes
            .checked_sub(actual_static)
            .ok_or_else(|| invalid_plan("static memory exceeds usable capacity"))?;
        let aggregate = Self::validate_dynamic_pools_structure(
            &self.dynamic_pools,
            &dynamic_by_id,
            dynamic_capacity_bytes,
        )?;
        let (actual_liveness_mode, actual_liveness) =
            Self::summarize_pool_invocation_liveness(&self.dynamic_pools)?;
        if actual_liveness_mode != self.invocation_liveness_mode
            || actual_liveness != self.invocation_liveness
        {
            return Err(invalid_plan(
                "global invocation liveness is not derived from per-pool evidence",
            ));
        }
        let actual_request_minimum = aggregate.minimum_request_bytes;
        let actual_sequence_minimum = aggregate.minimum_sequence_bytes;
        let actual_step_minimum = aggregate.minimum_step_bytes;
        let actual_invocation_peak = aggregate.minimum_invocation_peak_bytes;
        let actual_minimum = actual_request_minimum
            .checked_add(actual_sequence_minimum)
            .and_then(|bytes| bytes.checked_add(actual_step_minimum))
            .and_then(|bytes| bytes.checked_add(actual_invocation_peak))
            .ok_or_else(|| invalid_plan("minimum runnable request bytes overflow u64"))?;
        if aggregate.theoretical_ceiling_bytes != actual_theoretical_dynamic
            || actual_request_minimum != self.minimum_request_bytes
            || actual_sequence_minimum != self.minimum_sequence_bytes
            || actual_step_minimum != self.minimum_step_bytes
            || actual_invocation_peak != self.minimum_invocation_peak_bytes
            || actual_minimum != self.minimum_runnable_request_bytes
            || actual_theoretical != self.theoretical_ceiling_bytes.get()
            || actual_static
                .checked_add(actual_minimum)
                .is_none_or(|minimum| minimum > self.usable_capacity_bytes)
        {
            return Err(invalid_plan(
                "dynamic minimum or theoretical ceiling is not core-derived",
            ));
        }
        Ok(())
    }

    fn derive_dynamic_pools(
        dynamic_descriptors: &[DynamicResourceDescriptor],
        nodes: &[PlanNode],
        dynamic_capacity_bytes: u64,
    ) -> Result<Vec<DynamicBackingPoolSpec>, VNextError> {
        let mut groups = BTreeMap::<
            DynamicBackingPoolId,
            (PoolCompatibilityKey, Vec<&DynamicResourceDescriptor>),
        >::new();
        for descriptor in dynamic_descriptors {
            let compatibility = PoolCompatibilityKey::new(
                &descriptor.storage,
                descriptor.usage,
                descriptor.element_type,
                descriptor.alignment_bytes,
            )?;
            let expected_pool_id = DynamicBackingPoolId::from_compatibility(&compatibility)?;
            if descriptor.pool_id != expected_pool_id {
                return Err(invalid_plan(format!(
                    "dynamic resource `{}` has a non-derived backing pool id",
                    descriptor.base_resource_id
                )));
            }
            match groups.entry(expected_pool_id) {
                std::collections::btree_map::Entry::Vacant(entry) => {
                    entry.insert((compatibility, vec![descriptor]));
                }
                std::collections::btree_map::Entry::Occupied(mut entry) => {
                    if entry.get().0 != compatibility {
                        return Err(invalid_plan(
                            "dynamic backing pool hash collision has incompatible contracts",
                        ));
                    }
                    entry.get_mut().1.push(descriptor);
                }
            }
        }

        groups
            .into_values()
            .map(|(compatibility, mut descriptors)| {
                descriptors
                    .sort_by(|left, right| left.base_resource_id.cmp(&right.base_resource_id));
                let resource_ids = descriptors
                    .iter()
                    .map(|descriptor| descriptor.base_resource_id.clone())
                    .collect::<Vec<_>>();
                let minimum_request_bytes = minimum_for_lifetime(
                    &descriptors,
                    AllocationLifetime::Request,
                    "pool request minimum",
                )?;
                let minimum_sequence_bytes = minimum_for_lifetime(
                    &descriptors,
                    AllocationLifetime::Sequence,
                    "pool sequence minimum",
                )?;
                let minimum_step_bytes = minimum_for_lifetime(
                    &descriptors,
                    AllocationLifetime::Step,
                    "pool step minimum",
                )?;
                let theoretical_ceiling_bytes =
                    descriptors.iter().try_fold(0_u128, |total, descriptor| {
                        total
                            .checked_add(
                                u128::from(descriptor.theoretical_maximum_request_bytes()?)
                                    * u128::from(descriptor.theoretical_maximum_instances),
                            )
                            .ok_or_else(|| {
                                invalid_plan("dynamic pool theoretical ceiling overflows u128")
                            })
                    })?;
                let (invocation_liveness_mode, invocation_liveness, invocation_peak) =
                    Self::derive_pool_invocation_liveness(nodes, &descriptors)?;
                DynamicBackingPoolSpec::from_core(
                    compatibility,
                    resource_ids,
                    minimum_request_bytes,
                    minimum_sequence_bytes,
                    minimum_step_bytes,
                    invocation_peak,
                    theoretical_ceiling_bytes,
                    dynamic_capacity_bytes,
                    invocation_liveness_mode,
                    invocation_liveness,
                )
            })
            .collect()
    }

    fn derive_pool_invocation_liveness(
        nodes: &[PlanNode],
        descriptors: &[&DynamicResourceDescriptor],
    ) -> Result<(InvocationLivenessMode, Vec<InvocationResourceLiveness>, u64), VNextError> {
        let invocation_ids = descriptors
            .iter()
            .filter(|descriptor| descriptor.lifetime == AllocationLifetime::Invocation)
            .map(|descriptor| descriptor.base_resource_id.clone())
            .collect::<BTreeSet<_>>();
        if invocation_ids.is_empty() {
            return Ok((InvocationLivenessMode::NoInvocationResources, Vec::new(), 0));
        }
        let descriptor_by_id = descriptors
            .iter()
            .map(|descriptor| (descriptor.base_resource_id.clone(), *descriptor))
            .collect::<BTreeMap<_, _>>();
        let mut covered = BTreeSet::new();
        let liveness_in_execution_order = nodes
            .iter()
            .filter_map(|node| {
                let resource_ids = node
                    .resources
                    .iter()
                    .filter(|resource_id| invocation_ids.contains(*resource_id))
                    .cloned()
                    .collect::<Vec<_>>();
                covered.extend(resource_ids.iter().cloned());
                (!resource_ids.is_empty()).then(|| InvocationResourceLiveness {
                    node_id: node.id.clone(),
                    resource_ids,
                })
            })
            .collect::<Vec<_>>();
        if covered != invocation_ids {
            return Err(invalid_plan(
                "pool node liveness does not cover every invocation resource",
            ));
        }
        let nodes_by_id = nodes
            .iter()
            .map(|node| (node.id.clone(), node))
            .collect::<BTreeMap<_, _>>();
        let total_ordered =
            liveness_in_execution_order
                .windows(2)
                .try_fold(true, |ordered, pair| {
                    Ok::<bool, VNextError>(
                        ordered
                            && node_completion_precedes(
                                &nodes_by_id,
                                &pair[0].node_id,
                                &pair[1].node_id,
                            )?,
                    )
                })?;
        let row_bytes = |row: &InvocationResourceLiveness| {
            row.resource_ids
                .iter()
                .try_fold(0_u64, |total, resource_id| {
                    total
                        .checked_add(
                            descriptor_by_id
                                .get(resource_id)
                                .ok_or_else(|| {
                                    invalid_plan("pool liveness references an unknown descriptor")
                                })?
                                .minimum_request_bytes()?,
                        )
                        .ok_or_else(|| invalid_plan("pool invocation row bytes overflow u64"))
                })
        };
        let invocation_peak = if total_ordered {
            liveness_in_execution_order
                .iter()
                .try_fold(0_u64, |peak, row| {
                    Ok::<u64, VNextError>(peak.max(row_bytes(row)?))
                })?
        } else {
            liveness_in_execution_order
                .iter()
                .try_fold(0_u64, |total, row| {
                    total.checked_add(row_bytes(row)?).ok_or_else(|| {
                        invalid_plan("pool concurrent invocation bytes overflow u64")
                    })
                })?
        };
        let mut invocation_liveness = liveness_in_execution_order;
        invocation_liveness.sort_by(|left, right| left.node_id.cmp(&right.node_id));
        Ok((
            if total_ordered {
                InvocationLivenessMode::TotalOrderReuse
            } else {
                InvocationLivenessMode::ConservativeConcurrent
            },
            invocation_liveness,
            invocation_peak,
        ))
    }

    fn validate_dynamic_pools_structure(
        pools: &[DynamicBackingPoolSpec],
        descriptors: &BTreeMap<ResourceId, &DynamicResourceDescriptor>,
        dynamic_capacity_bytes: u64,
    ) -> Result<PoolAggregateEvidence, VNextError> {
        let mut expected_members =
            BTreeMap::<DynamicBackingPoolId, (PoolCompatibilityKey, Vec<ResourceId>)>::new();
        for descriptor in descriptors.values() {
            let compatibility = PoolCompatibilityKey::new(
                &descriptor.storage,
                descriptor.usage,
                descriptor.element_type,
                descriptor.alignment_bytes,
            )?;
            let expected_id = DynamicBackingPoolId::from_compatibility(&compatibility)?;
            if descriptor.pool_id != expected_id {
                return Err(invalid_plan(
                    "dynamic descriptor has a non-derived pool identity",
                ));
            }
            match expected_members.entry(expected_id) {
                std::collections::btree_map::Entry::Vacant(entry) => {
                    entry.insert((compatibility, vec![descriptor.base_resource_id.clone()]));
                }
                std::collections::btree_map::Entry::Occupied(mut entry) => {
                    if entry.get().0 != compatibility {
                        return Err(invalid_plan(
                            "dynamic pool hash collision has incompatible descriptors",
                        ));
                    }
                    entry.get_mut().1.push(descriptor.base_resource_id.clone());
                }
            }
        }
        if pools.len() != expected_members.len() {
            return Err(invalid_plan(
                "dynamic backing pool count is not derived from descriptors",
            ));
        }
        let mut aggregate = PoolAggregateEvidence::default();
        for pool in pools {
            pool.validate_local()?;
            let (compatibility, resource_ids) = expected_members
                .remove(&pool.pool_id)
                .ok_or_else(|| invalid_plan("dynamic backing pool has no descriptor members"))?;
            if pool.compatibility != compatibility || pool.resource_ids != resource_ids {
                return Err(invalid_plan(
                    "dynamic backing pool compatibility or membership is not core-derived",
                ));
            }
            let members =
                pool.resource_ids
                    .iter()
                    .map(|resource_id| {
                        descriptors.get(resource_id).copied().ok_or_else(|| {
                            invalid_plan("dynamic pool member descriptor is missing")
                        })
                    })
                    .collect::<Result<Vec<_>, VNextError>>()?;
            let request = minimum_for_lifetime(
                &members,
                AllocationLifetime::Request,
                "pool request minimum",
            )?;
            let sequence = minimum_for_lifetime(
                &members,
                AllocationLifetime::Sequence,
                "pool sequence minimum",
            )?;
            let step =
                minimum_for_lifetime(&members, AllocationLifetime::Step, "pool step minimum")?;
            let theoretical = members.iter().try_fold(0_u128, |total, descriptor| {
                total
                    .checked_add(
                        u128::from(descriptor.theoretical_maximum_request_bytes()?)
                            * u128::from(descriptor.theoretical_maximum_instances),
                    )
                    .ok_or_else(|| invalid_plan("pool theoretical ceiling overflows u128"))
            })?;
            let invocation_ids = members
                .iter()
                .filter(|descriptor| descriptor.lifetime == AllocationLifetime::Invocation)
                .map(|descriptor| descriptor.base_resource_id.clone())
                .collect::<BTreeSet<_>>();
            let invocation_peak = validate_pool_liveness_rows(
                pool.invocation_liveness_mode,
                &pool.invocation_liveness,
                &invocation_ids,
                descriptors,
            )?;
            let maximum_resident =
                u64::try_from(theoretical.min(u128::from(dynamic_capacity_bytes)))
                    .map_err(|_| invalid_plan("pool resident ceiling exceeds u64"))?;
            if pool.minimum_request_bytes != request
                || pool.minimum_sequence_bytes != sequence
                || pool.minimum_step_bytes != step
                || pool.minimum_invocation_peak_bytes != invocation_peak
                || pool.theoretical_ceiling_bytes.get() != theoretical
                || pool.provisioning.maximum_resident_bytes != maximum_resident
            {
                return Err(invalid_plan(
                    "dynamic backing pool bounds or liveness are not core-derived",
                ));
            }
            aggregate.add(pool)?;
        }
        if !expected_members.is_empty() {
            return Err(invalid_plan(
                "dynamic descriptors are missing canonical backing pools",
            ));
        }
        Ok(aggregate)
    }

    fn summarize_pool_invocation_liveness(
        pools: &[DynamicBackingPoolSpec],
    ) -> Result<(InvocationLivenessMode, Vec<InvocationResourceLiveness>), VNextError> {
        let mut by_node = BTreeMap::<NodeId, BTreeSet<ResourceId>>::new();
        let mut has_invocation = false;
        let mut all_total_ordered = true;
        for pool in pools {
            match pool.invocation_liveness_mode {
                InvocationLivenessMode::NoInvocationResources => {}
                InvocationLivenessMode::TotalOrderReuse => has_invocation = true,
                InvocationLivenessMode::ConservativeConcurrent => {
                    has_invocation = true;
                    all_total_ordered = false;
                }
            }
            for row in &pool.invocation_liveness {
                by_node
                    .entry(row.node_id.clone())
                    .or_default()
                    .extend(row.resource_ids.iter().cloned());
            }
        }
        let liveness = by_node
            .into_iter()
            .map(|(node_id, resource_ids)| InvocationResourceLiveness {
                node_id,
                resource_ids: resource_ids.into_iter().collect(),
            })
            .collect::<Vec<_>>();
        let reference_count = liveness.iter().try_fold(0_usize, |total, row| {
            total.checked_add(row.resource_ids.len())
        });
        if reference_count.is_none_or(|count| count > MAX_EXECUTION_PLAN_RESOURCE_ROWS) {
            return Err(invalid_plan(
                "invocation liveness reference count is invalid",
            ));
        }
        Ok((
            if !has_invocation {
                InvocationLivenessMode::NoInvocationResources
            } else if all_total_ordered {
                InvocationLivenessMode::TotalOrderReuse
            } else {
                InvocationLivenessMode::ConservativeConcurrent
            },
            liveness,
        ))
    }

    pub const fn device_capacity_bytes(&self) -> u64 {
        self.device_capacity_bytes
    }

    pub const fn policy_capacity_bytes(&self) -> u64 {
        self.policy_capacity_bytes
    }

    pub const fn reserve_bytes(&self) -> u64 {
        self.reserve_bytes
    }

    pub const fn usable_capacity_bytes(&self) -> u64 {
        self.usable_capacity_bytes
    }

    pub const fn maximum_active_sequences(&self) -> u32 {
        self.maximum_active_sequences
    }

    pub const fn capacity_bytes(&self) -> u64 {
        self.usable_capacity_bytes
    }

    pub const fn static_bytes(&self) -> u64 {
        self.static_bytes
    }

    pub const fn minimum_request_bytes(&self) -> u64 {
        self.minimum_request_bytes
    }

    pub const fn minimum_sequence_bytes(&self) -> u64 {
        self.minimum_sequence_bytes
    }

    pub const fn minimum_step_bytes(&self) -> u64 {
        self.minimum_step_bytes
    }

    pub const fn minimum_invocation_peak_bytes(&self) -> u64 {
        self.minimum_invocation_peak_bytes
    }

    pub const fn minimum_runnable_request_bytes(&self) -> u64 {
        self.minimum_runnable_request_bytes
    }

    pub const fn invocation_liveness_mode(&self) -> InvocationLivenessMode {
        self.invocation_liveness_mode
    }

    pub fn invocation_liveness(&self) -> &[InvocationResourceLiveness] {
        &self.invocation_liveness
    }

    /// Conservative checked evidence across provider formula maxima and the
    /// protocol ceiling. It is never a reservation, admission target, or
    /// claim. Invocation concurrency is described separately by the checked
    /// liveness mode instead of assuming every node is mutually exclusive.
    pub fn theoretical_ceiling_bytes(&self) -> u128 {
        self.theoretical_ceiling_bytes.get()
    }

    pub fn static_allocations(&self) -> &[ResourceAllocation] {
        &self.static_allocations
    }

    pub fn dynamic_descriptors(&self) -> &[DynamicResourceDescriptor] {
        &self.dynamic_descriptors
    }

    pub fn dynamic_pools(&self) -> &[DynamicBackingPoolSpec] {
        &self.dynamic_pools
    }

    pub fn static_buffer_requests(&self) -> Result<Vec<BufferRequest>, VNextError> {
        self.static_allocations
            .iter()
            .map(ResourceAllocation::buffer_request)
            .collect()
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct MemoryPlanWire {
    device_capacity_bytes: u64,
    policy_capacity_bytes: u64,
    reserve_bytes: u64,
    usable_capacity_bytes: u64,
    maximum_active_sequences: u32,
    static_bytes: u64,
    minimum_request_bytes: u64,
    minimum_sequence_bytes: u64,
    minimum_step_bytes: u64,
    minimum_invocation_peak_bytes: u64,
    minimum_runnable_request_bytes: u64,
    theoretical_ceiling_bytes: CanonicalU128,
    static_allocations: Vec<ResourceAllocation>,
    dynamic_descriptors: Vec<DynamicResourceDescriptor>,
    dynamic_pools: Vec<DynamicBackingPoolSpec>,
    invocation_liveness_mode: InvocationLivenessMode,
    invocation_liveness: Vec<InvocationResourceLiveness>,
}

impl<'de> Deserialize<'de> for MemoryPlan {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = MemoryPlanWire::deserialize(deserializer)?;
        let plan = Self {
            device_capacity_bytes: wire.device_capacity_bytes,
            policy_capacity_bytes: wire.policy_capacity_bytes,
            reserve_bytes: wire.reserve_bytes,
            usable_capacity_bytes: wire.usable_capacity_bytes,
            maximum_active_sequences: wire.maximum_active_sequences,
            static_bytes: wire.static_bytes,
            minimum_request_bytes: wire.minimum_request_bytes,
            minimum_sequence_bytes: wire.minimum_sequence_bytes,
            minimum_step_bytes: wire.minimum_step_bytes,
            minimum_invocation_peak_bytes: wire.minimum_invocation_peak_bytes,
            minimum_runnable_request_bytes: wire.minimum_runnable_request_bytes,
            theoretical_ceiling_bytes: wire.theoretical_ceiling_bytes,
            static_allocations: wire.static_allocations,
            dynamic_descriptors: wire.dynamic_descriptors,
            dynamic_pools: wire.dynamic_pools,
            invocation_liveness_mode: wire.invocation_liveness_mode,
            invocation_liveness: wire.invocation_liveness,
        };
        plan.validate().map_err(serde::de::Error::custom)?;
        Ok(plan)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProviderWorkspaceScope {
    Plan,
    Request,
    Sequence,
    Step,
    Invocation,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ProviderWorkspaceRequirement {
    size_formula: ProviderWorkspaceSizeFormula,
    alignment_bytes: u64,
    scope: ProviderWorkspaceScope,
    storage: DynamicStorageRequirement,
}

impl ProviderWorkspaceRequirement {
    /// Convenience constructor for a fixed-size workspace. Shape-dependent
    /// providers must use [`Self::from_formula`].
    pub fn new(
        fixed_bytes: u64,
        alignment_bytes: u64,
        scope: ProviderWorkspaceScope,
        storage: DynamicStorageRequirement,
    ) -> Result<Self, VNextError> {
        Self::from_formula(
            ProviderWorkspaceSizeFormula::fixed(fixed_bytes)?,
            alignment_bytes,
            scope,
            storage,
        )
    }

    pub fn from_formula(
        size_formula: ProviderWorkspaceSizeFormula,
        alignment_bytes: u64,
        scope: ProviderWorkspaceScope,
        storage: DynamicStorageRequirement,
    ) -> Result<Self, VNextError> {
        size_formula.validate()?;
        if alignment_bytes == 0
            || !alignment_bytes.is_power_of_two()
            || (scope == ProviderWorkspaceScope::Plan && !size_formula.is_fixed())
            || (scope == ProviderWorkspaceScope::Sequence
                && !size_formula.is_valid_for_sequence_scope())
        {
            return Err(invalid_plan(
                "provider workspace has invalid formula, alignment, or scope",
            ));
        }
        let requirement = Self {
            size_formula,
            alignment_bytes,
            scope,
            storage,
        };
        requirement.minimum_bytes()?;
        requirement.theoretical_maximum_bytes()?;
        Ok(requirement)
    }

    pub fn size_formula(&self) -> &ProviderWorkspaceSizeFormula {
        &self.size_formula
    }

    pub fn evaluate_bytes(&self, work: &ResourceWorkShape) -> Result<u64, VNextError> {
        self.evaluate_shape_bytes(work.immediate_shape())
    }

    pub fn evaluate_fit_bytes(&self, work: &ResourceWorkShape) -> Result<u64, VNextError> {
        self.evaluate_shape_bytes(work.fit_shape())
    }

    pub(crate) fn evaluate_shape_bytes(
        &self,
        shape: DynamicResourceShape,
    ) -> Result<u64, VNextError> {
        align_up(
            self.size_formula.evaluate_shape_bytes(shape)?,
            self.alignment_bytes,
        )
    }

    pub fn minimum_bytes(&self) -> Result<u64, VNextError> {
        self.evaluate_shape_bytes(self.size_formula.minimum_shape())
    }

    pub fn theoretical_maximum_bytes(&self) -> Result<u64, VNextError> {
        self.evaluate_shape_bytes(self.size_formula.theoretical_maximum_shape())
    }

    pub fn fixed_bytes(&self) -> Option<u64> {
        match &self.size_formula {
            DynamicResourceDemand::Fixed { bytes } => Some(*bytes),
            _ => None,
        }
    }

    pub const fn alignment_bytes(&self) -> u64 {
        self.alignment_bytes
    }

    pub const fn scope(&self) -> ProviderWorkspaceScope {
        self.scope
    }

    pub fn storage(&self) -> &DynamicStorageRequirement {
        &self.storage
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ProviderWorkspaceRequirementWire {
    size_formula: ProviderWorkspaceSizeFormula,
    alignment_bytes: u64,
    scope: ProviderWorkspaceScope,
    storage: DynamicStorageRequirement,
}

impl<'de> Deserialize<'de> for ProviderWorkspaceRequirement {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = ProviderWorkspaceRequirementWire::deserialize(deserializer)?;
        Self::from_formula(
            wire.size_formula,
            wire.alignment_bytes,
            wire.scope,
            wire.storage,
        )
        .map_err(serde::de::Error::custom)
    }
}

#[derive(Serialize)]
struct ProviderEstimateFingerprintMaterial<'a> {
    provider_id: &'a ProviderId,
    estimator_id: &'a str,
    estimator_version: ContractVersion,
    estimator_implementation_fingerprint: &'a str,
    estimator_input_fingerprint: &'a str,
    value_alignment_bytes: u64,
    scratch: &'a Option<ProviderWorkspaceRequirement>,
    persistent: &'a Option<ProviderWorkspaceRequirement>,
}

/// Trusted output from the selected provider's shape/attribute-specific
/// resource estimator. The core binds it to the exact estimator input and
/// selected provider before the values can enter an executable plan.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ProviderResourcePlan {
    provider_id: ProviderId,
    estimator_id: String,
    estimator_version: ContractVersion,
    estimator_implementation_fingerprint: String,
    estimator_input_fingerprint: String,
    estimate_fingerprint: String,
    value_alignment_bytes: u64,
    scratch: Option<ProviderWorkspaceRequirement>,
    persistent: Option<ProviderWorkspaceRequirement>,
}

impl ProviderResourcePlan {
    #[allow(clippy::too_many_arguments)]
    fn from_provider_output(
        descriptor: &OperationProviderDescriptor,
        estimator_input_fingerprint: &str,
        estimate: OperationResourceEstimate,
    ) -> Result<Self, VNextError> {
        if estimate.estimator_id() != descriptor.resource_estimator_id()
            || estimate.estimator_version() != descriptor.resource_estimator_version()
            || estimate.estimator_implementation_fingerprint()
                != descriptor.resource_estimator_implementation_fingerprint()
            || estimate.claimed_input_fingerprint() != estimator_input_fingerprint
        {
            return Err(invalid_plan(
                "provider raw resource estimate identity or input claim differs from the selected registered implementation",
            ));
        }
        let mut plan = Self {
            provider_id: descriptor.provider_id().clone(),
            estimator_id: descriptor.resource_estimator_id().to_owned(),
            estimator_version: descriptor.resource_estimator_version(),
            estimator_implementation_fingerprint: descriptor
                .resource_estimator_implementation_fingerprint()
                .to_owned(),
            estimator_input_fingerprint: estimator_input_fingerprint.to_owned(),
            estimate_fingerprint: String::new(),
            value_alignment_bytes: estimate.value_alignment_bytes(),
            scratch: estimate.scratch().cloned(),
            persistent: estimate.persistent().cloned(),
        };
        plan.validate_fields()?;
        plan.estimate_fingerprint = plan.compute_estimate_fingerprint()?;
        plan.validate_shape()?;
        Ok(plan)
    }

    fn validate_fields(&self) -> Result<(), VNextError> {
        if self.estimator_id.is_empty()
            || self.estimator_id.len() > 160
            || !self.estimator_id.bytes().all(|byte| {
                byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-' | b':' | b'/')
            })
            || self.estimator_version.major == 0
            || !is_canonical_sha256(&self.estimator_implementation_fingerprint)
            || !is_canonical_sha256(&self.estimator_input_fingerprint)
            || self.value_alignment_bytes == 0
            || !self.value_alignment_bytes.is_power_of_two()
            || self.scratch.as_ref().is_some_and(|workspace| {
                workspace.scope != ProviderWorkspaceScope::Invocation
                    || ProviderWorkspaceRequirement::from_formula(
                        workspace.size_formula.clone(),
                        workspace.alignment_bytes,
                        workspace.scope,
                        workspace.storage.clone(),
                    )
                    .is_err()
            })
            || self.persistent.as_ref().is_some_and(|workspace| {
                workspace.scope == ProviderWorkspaceScope::Invocation
                    || ProviderWorkspaceRequirement::from_formula(
                        workspace.size_formula.clone(),
                        workspace.alignment_bytes,
                        workspace.scope,
                        workspace.storage.clone(),
                    )
                    .is_err()
            })
        {
            return Err(invalid_plan(
                "provider resource estimate identity, alignment, or scope is invalid",
            ));
        }
        Ok(())
    }

    fn validate_shape(&self) -> Result<(), VNextError> {
        self.validate_fields()?;
        if !is_canonical_sha256(&self.estimate_fingerprint)
            || self.estimate_fingerprint != self.compute_estimate_fingerprint()?
        {
            return Err(invalid_plan(
                "provider resource estimate fingerprint does not match its typed fields",
            ));
        }
        Ok(())
    }

    fn compute_estimate_fingerprint(&self) -> Result<String, VNextError> {
        canonical_fingerprint(
            &ProviderEstimateFingerprintMaterial {
                provider_id: &self.provider_id,
                estimator_id: &self.estimator_id,
                estimator_version: self.estimator_version,
                estimator_implementation_fingerprint: &self.estimator_implementation_fingerprint,
                estimator_input_fingerprint: &self.estimator_input_fingerprint,
                value_alignment_bytes: self.value_alignment_bytes,
                scratch: &self.scratch,
                persistent: &self.persistent,
            },
            "fingerprint provider resource estimate",
        )
    }

    pub fn provider_id(&self) -> &ProviderId {
        &self.provider_id
    }

    pub fn estimator_id(&self) -> &str {
        &self.estimator_id
    }

    pub const fn estimator_version(&self) -> ContractVersion {
        self.estimator_version
    }

    pub fn estimator_implementation_fingerprint(&self) -> &str {
        &self.estimator_implementation_fingerprint
    }

    pub fn estimator_input_fingerprint(&self) -> &str {
        &self.estimator_input_fingerprint
    }

    pub fn estimate_fingerprint(&self) -> &str {
        &self.estimate_fingerprint
    }

    pub const fn value_alignment_bytes(&self) -> u64 {
        self.value_alignment_bytes
    }

    pub fn scratch(&self) -> Option<&ProviderWorkspaceRequirement> {
        self.scratch.as_ref()
    }

    pub fn persistent(&self) -> Option<&ProviderWorkspaceRequirement> {
        self.persistent.as_ref()
    }
}

#[derive(Serialize)]
struct ProviderEstimatorInputMaterial<'a> {
    prepared_family_fingerprint: &'a str,
    operation_fingerprint: &'a str,
    node_id: &'a NodeId,
    operation_id: &'a OperationId,
    operation_version: ContractVersion,
    attributes: &'a BTreeMap<AttributeId, SemanticValue>,
    provider_id: &'a ProviderId,
    values: &'a [ResolvedValueBinding],
    required_capabilities: &'a BTreeSet<CapabilityId>,
    required_weight_formats: &'a BTreeSet<WeightFormatId>,
    required_quantization_formats: &'a BTreeSet<QuantizationFormatId>,
}

/// Canonical input signature a provider estimator must bind into its trusted
/// result. Physical bindings remain part of the signature, but never come from
/// a serialized execution plan during validation.
pub(crate) fn provider_resource_estimator_input_fingerprint(
    family: &PreparedModelFamily,
    operation: &OperationDescriptor,
    node: &ProgramNode,
    provider_id: &ProviderId,
    values: &[ResolvedValueBinding],
    required_capabilities: &BTreeSet<CapabilityId>,
) -> Result<String, VNextError> {
    let (required_weight_formats, required_quantization_formats) =
        ExecutionPlan::node_weight_requirements(family, values)?;
    let effective_required_capabilities = operation
        .provider
        .required_capabilities
        .union(required_capabilities)
        .cloned()
        .collect::<BTreeSet<_>>();
    let prepared_family_fingerprint = family.fingerprint()?;
    let operation_fingerprint = operation.fingerprint()?;
    canonical_fingerprint(
        &ProviderEstimatorInputMaterial {
            prepared_family_fingerprint: &prepared_family_fingerprint,
            operation_fingerprint: &operation_fingerprint,
            node_id: &node.id,
            operation_id: &node.operation_id,
            operation_version: node.required_version,
            attributes: &node.attributes,
            provider_id,
            values,
            required_capabilities: &effective_required_capabilities,
            required_weight_formats: &required_weight_formats,
            required_quantization_formats: &required_quantization_formats,
        },
        "fingerprint provider resource estimator input",
    )
}

/// Per-node trusted physical resolution. It supplies physical bindings and a
/// provider estimator result, but cannot provide memory totals, compatibility
/// reports, plan identities, or hashes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlanNodeResolution {
    operation_registry_authority: OperationRegistryAuthority,
    node_id: NodeId,
    values: Vec<ResolvedValueBinding>,
    required_capabilities: BTreeSet<CapabilityId>,
    preferred_provider: Option<ProviderId>,
    provider_resource_candidates: Vec<ProviderResourcePlan>,
    provider_resolution_rejections: BTreeMap<ProviderId, PlanProviderRejectReason>,
}

impl PlanNodeResolution {
    fn from_provider_resolution(
        operation_registry_authority: OperationRegistryAuthority,
        node_id: NodeId,
        values: Vec<ResolvedValueBinding>,
        required_capabilities: BTreeSet<CapabilityId>,
        preferred_provider: Option<ProviderId>,
        mut provider_resource_candidates: Vec<ProviderResourcePlan>,
        provider_resolution_rejections: BTreeMap<ProviderId, PlanProviderRejectReason>,
    ) -> Result<Self, VNextError> {
        if values.is_empty() {
            return Err(invalid_plan(format!(
                "node `{node_id}` has no physical value resolution"
            )));
        }
        provider_resource_candidates
            .sort_by(|left, right| left.provider_id.cmp(&right.provider_id));
        if provider_resource_candidates.is_empty()
            || provider_resource_candidates
                .windows(2)
                .any(|pair| pair[0].provider_id == pair[1].provider_id)
        {
            return Err(invalid_plan(format!(
                "node `{node_id}` has empty or duplicate provider resource candidates"
            )));
        }
        for candidate in &provider_resource_candidates {
            candidate.validate_shape()?;
            if provider_resolution_rejections.contains_key(candidate.provider_id()) {
                return Err(invalid_plan(format!(
                    "provider `{}` is both a resource candidate and rejected resolution",
                    candidate.provider_id()
                )));
            }
        }
        for reason in provider_resolution_rejections.values() {
            match reason {
                PlanProviderRejectReason::StorageIncompatible { resource_ids }
                    if !resource_ids.is_empty()
                        && !resource_ids.windows(2).any(|pair| pair[0] >= pair[1]) => {}
                PlanProviderRejectReason::StorageIncompatible { .. } => {
                    return Err(invalid_plan(
                        "provider storage rejection resources are empty or non-canonical",
                    ))
                }
                PlanProviderRejectReason::NotRegistered
                | PlanProviderRejectReason::Incompatible(_) => {
                    return Err(invalid_plan(
                        "resolution-local rejection must describe storage incompatibility",
                    ))
                }
            }
        }
        Ok(Self {
            operation_registry_authority,
            node_id,
            values,
            required_capabilities,
            preferred_provider,
            provider_resource_candidates,
            provider_resolution_rejections,
        })
    }

    /// Resolves one node through the typed planning registry. Provider
    /// selection is core-owned; `preferred_provider` is only a compatibility
    /// preference and cannot inject a provider or resource estimate.
    #[allow(clippy::too_many_arguments)]
    pub fn resolve<P: RuntimePolicy>(
        family: &PreparedModelFamily,
        catalog: &CapabilityCatalog,
        policy: &P,
        registry: &OperationPlanningHandle<'_>,
        node_id: NodeId,
        values: Vec<ResolvedValueBinding>,
        required_capabilities: BTreeSet<CapabilityId>,
        preferred_provider: Option<ProviderId>,
    ) -> Result<Self, VNextError> {
        policy.validate()?;
        validate_active_sequence_ceiling(policy.maximum_active_sequences())?;
        if values.is_empty() {
            return Err(invalid_plan(format!(
                "node `{node_id}` has no physical value resolution"
            )));
        }
        let program_node = family
            .program()
            .blocks()
            .iter()
            .flat_map(|block| &block.nodes)
            .find(|node| node.id == node_id)
            .ok_or_else(|| invalid_plan(format!("program has no node `{node_id}`")))?;
        let operation = catalog.operation(&program_node.operation_id)?;

        let contracts = registry.contracts_for(&program_node.operation_id);
        if contracts.len() != 1 {
            return Err(invalid_plan(format!(
                "operation `{}` requires exactly one typed contract registration, found {}",
                program_node.operation_id,
                contracts.len()
            )));
        }
        let contract = contracts[0];
        if contract.descriptor() != operation {
            return Err(invalid_plan(format!(
                "typed contract for operation `{}` differs from the capability catalog",
                program_node.operation_id
            )));
        }
        contract.validate_signature(&operation.inputs, &operation.outputs)?;
        operation.validate_attributes(&program_node.attributes)?;
        operation.validate_resolved_bindings(&values)?;
        ExecutionPlan::validate_program_bindings(program_node, &values)?;
        for binding in &values {
            ExecutionPlan::validate_semantic_binding(family, binding)?;
        }

        let (required_weight_formats, required_quantization_formats) =
            ExecutionPlan::node_weight_requirements(family, &values)?;
        let compatibility_request = ProviderCompatibilityRequest::new(
            program_node.operation_id.clone(),
            program_node.required_version,
            operation
                .provider
                .required_capabilities
                .union(&required_capabilities)
                .cloned()
                .collect(),
            required_weight_formats,
            required_quantization_formats,
        )?;
        let report = catalog.provider_compatibility(compatibility_request)?;
        report.require_compatible(&catalog.device().id)?;
        let profile_available = |requirement: &DynamicStorageRequirement| {
            policy
                .dynamic_storage_profile_order()
                .iter()
                .any(|profile| {
                    catalog.device().dynamic_storage_profiles.contains(profile)
                        && requirement.accepts(*profile)
                })
        };
        let mut provider_resource_candidates = Vec::new();
        let mut provider_resolution_rejections = BTreeMap::new();
        for provider_id in report.compatible_provider_ids() {
            let descriptor = catalog
                .providers_for(&program_node.operation_id)?
                .iter()
                .find(|provider| provider.provider_id() == provider_id)
                .ok_or_else(|| invalid_plan("compatible provider is absent from the catalog"))?;
            let value_storage_conflicts = values
                .iter()
                .filter(|binding| binding.usage() != BufferUsage::Weights)
                .filter(|binding| {
                    descriptor
                        .dynamic_storage_for(binding.role(), binding.ordinal())
                        .is_none_or(|requirement| !profile_available(requirement))
                })
                .flat_map(|binding| binding.storage().components())
                .map(|component| component.resource_id().clone())
                .collect::<BTreeSet<_>>();
            if !value_storage_conflicts.is_empty() {
                provider_resolution_rejections.insert(
                    provider_id.clone(),
                    PlanProviderRejectReason::StorageIncompatible {
                        resource_ids: value_storage_conflicts.into_iter().collect(),
                    },
                );
                continue;
            }
            let estimators = registry.estimators_for(provider_id);
            if estimators.len() != 1 || estimators[0].descriptor() != descriptor {
                return Err(invalid_plan(format!(
                    "compatible provider `{provider_id}` lacks one exact resource estimator registration"
                )));
            }
            let estimator_input_fingerprint = provider_resource_estimator_input_fingerprint(
                family,
                operation,
                program_node,
                provider_id,
                &values,
                &required_capabilities,
            )?;
            let estimate_request = OperationResourceEstimateRequest::new(
                &program_node.id,
                operation,
                &values,
                &program_node.attributes,
                &estimator_input_fingerprint,
            )?;
            let estimate = estimators[0].estimate_resources(estimate_request)?;
            let provider_resources = ProviderResourcePlan::from_provider_output(
                descriptor,
                &estimator_input_fingerprint,
                estimate,
            )?;
            let minimum_alignment = operation.resources.minimum_value_alignment_bytes;
            if provider_resources.value_alignment_bytes() < minimum_alignment
                || provider_resources.value_alignment_bytes() % minimum_alignment != 0
                || !operation
                    .resources
                    .scratch
                    .accepts(provider_resources.scratch().is_some())
                || !operation
                    .resources
                    .persistent
                    .accepts(provider_resources.persistent().is_some())
            {
                return Err(invalid_plan(format!(
                    "compatible provider `{provider_id}` returned resources outside its operation contract"
                )));
            }
            let mut workspace_storage_conflicts = BTreeSet::new();
            for (kind, workspace) in [
                ("scratch", provider_resources.scratch()),
                ("persistent", provider_resources.persistent()),
            ] {
                if workspace.is_some_and(|workspace| !profile_available(workspace.storage())) {
                    workspace_storage_conflicts.insert(ExecutionPlan::workspace_base_id(
                        &program_node.id,
                        kind,
                        provider_resources.estimate_fingerprint(),
                    )?);
                }
            }
            if !workspace_storage_conflicts.is_empty() {
                provider_resolution_rejections.insert(
                    provider_id.clone(),
                    PlanProviderRejectReason::StorageIncompatible {
                        resource_ids: workspace_storage_conflicts.into_iter().collect(),
                    },
                );
                continue;
            }
            provider_resource_candidates.push(provider_resources);
        }
        if provider_resource_candidates.is_empty() {
            return Err(invalid_plan(format!(
                "node `{node_id}` has no provider whose binding and workspace storage requirements intersect runtime offers and policy"
            )));
        }
        Self::from_provider_resolution(
            registry.authority().clone(),
            node_id,
            values,
            required_capabilities,
            preferred_provider,
            provider_resource_candidates,
            provider_resolution_rejections,
        )
    }

    pub fn node_id(&self) -> &NodeId {
        &self.node_id
    }

    pub fn values(&self) -> &[ResolvedValueBinding] {
        &self.values
    }

    pub fn required_capabilities(&self) -> &BTreeSet<CapabilityId> {
        &self.required_capabilities
    }

    pub fn preferred_provider(&self) -> Option<&ProviderId> {
        self.preferred_provider.as_ref()
    }

    pub fn provider_resource_candidates(&self) -> &[ProviderResourcePlan] {
        &self.provider_resource_candidates
    }

    pub fn provider_resolution_rejections(
        &self,
    ) -> &BTreeMap<ProviderId, PlanProviderRejectReason> {
        &self.provider_resolution_rejections
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ExecutionPlanPayload {
    schema: PlanSchemaVersion,
    plan_id: PlanId,
    family_id: ModelFamilyId,
    device_id: super::DeviceId,
    device_runtime_implementation_fingerprint: String,
    prepared_family_fingerprint: String,
    program_fingerprint: String,
    capability_catalog_fingerprint: String,
    policy_version: ContractVersion,
    policy_fingerprint: String,
    weight_format: WeightFormatId,
    quantization_formats: BTreeSet<QuantizationFormatId>,
    nodes: Vec<PlanNode>,
    memory: MemoryPlan,
}

impl ExecutionPlanPayload {
    pub const fn schema(&self) -> PlanSchemaVersion {
        self.schema
    }

    pub fn plan_id(&self) -> &PlanId {
        &self.plan_id
    }

    pub fn family_id(&self) -> &ModelFamilyId {
        &self.family_id
    }

    pub fn device_id(&self) -> &super::DeviceId {
        &self.device_id
    }

    pub fn device_runtime_implementation_fingerprint(&self) -> &str {
        &self.device_runtime_implementation_fingerprint
    }

    pub fn prepared_family_fingerprint(&self) -> &str {
        &self.prepared_family_fingerprint
    }

    pub fn program_fingerprint(&self) -> &str {
        &self.program_fingerprint
    }

    pub fn capability_catalog_fingerprint(&self) -> &str {
        &self.capability_catalog_fingerprint
    }

    pub const fn policy_version(&self) -> ContractVersion {
        self.policy_version
    }

    pub fn policy_fingerprint(&self) -> &str {
        &self.policy_fingerprint
    }

    pub fn weight_format(&self) -> &WeightFormatId {
        &self.weight_format
    }

    pub fn quantization_formats(&self) -> &BTreeSet<QuantizationFormatId> {
        &self.quantization_formats
    }

    pub fn nodes(&self) -> &[PlanNode] {
        &self.nodes
    }

    pub fn memory(&self) -> &MemoryPlan {
        &self.memory
    }
}

#[derive(Serialize)]
struct PlanHashMaterial<'a> {
    schema: PlanSchemaVersion,
    family_id: &'a ModelFamilyId,
    device_id: &'a super::DeviceId,
    device_runtime_implementation_fingerprint: &'a str,
    prepared_family_fingerprint: &'a str,
    program_fingerprint: &'a str,
    capability_catalog_fingerprint: &'a str,
    policy_version: ContractVersion,
    policy_fingerprint: &'a str,
    weight_format: &'a WeightFormatId,
    quantization_formats: &'a BTreeSet<QuantizationFormatId>,
    nodes: &'a [PlanNode],
    memory: &'a MemoryPlan,
}

impl<'a> From<&'a ExecutionPlanPayload> for PlanHashMaterial<'a> {
    fn from(payload: &'a ExecutionPlanPayload) -> Self {
        Self {
            schema: payload.schema,
            family_id: &payload.family_id,
            device_id: &payload.device_id,
            device_runtime_implementation_fingerprint: &payload
                .device_runtime_implementation_fingerprint,
            prepared_family_fingerprint: &payload.prepared_family_fingerprint,
            program_fingerprint: &payload.program_fingerprint,
            capability_catalog_fingerprint: &payload.capability_catalog_fingerprint,
            policy_version: payload.policy_version,
            policy_fingerprint: &payload.policy_fingerprint,
            weight_format: &payload.weight_format,
            quantization_formats: &payload.quantization_formats,
            nodes: &payload.nodes,
            memory: &payload.memory,
        }
    }
}

/// A wire payload is deliberately not an executable plan. It must be rebuilt
/// against a typed model family, catalog, and runtime policy before use.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct UnvalidatedProviderResourcePlan {
    provider_id: ProviderId,
    estimator_id: String,
    estimator_version: ContractVersion,
    estimator_implementation_fingerprint: String,
    estimator_input_fingerprint: String,
    estimate_fingerprint: String,
    value_alignment_bytes: u64,
    scratch: Option<ProviderWorkspaceRequirement>,
    persistent: Option<ProviderWorkspaceRequirement>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct UnvalidatedPlanNode {
    id: NodeId,
    dependencies: Vec<NodeId>,
    operation_id: OperationId,
    operation_version: ContractVersion,
    operation_fingerprint: String,
    provider_implementation_fingerprint: String,
    required_capabilities: BTreeSet<CapabilityId>,
    attributes: BTreeMap<AttributeId, SemanticValue>,
    selection: ProviderSelection,
    provider_resources: UnvalidatedProviderResourcePlan,
    values: Vec<ResolvedValueBinding>,
    exact_aliases: Vec<PlanExactAlias>,
    state_effects: Vec<PlanStateEffect>,
    scratch_resource: Option<ResourceId>,
    persistent_resource: Option<ResourceId>,
    resources: Vec<ResourceId>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct UnvalidatedExecutionPlanPayload {
    schema: PlanSchemaVersion,
    plan_id: PlanId,
    family_id: ModelFamilyId,
    device_id: super::DeviceId,
    device_runtime_implementation_fingerprint: String,
    prepared_family_fingerprint: String,
    program_fingerprint: String,
    capability_catalog_fingerprint: String,
    policy_version: ContractVersion,
    policy_fingerprint: String,
    weight_format: WeightFormatId,
    quantization_formats: BTreeSet<QuantizationFormatId>,
    nodes: Vec<UnvalidatedPlanNode>,
    memory: MemoryPlan,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnvalidatedExecutionPlan {
    payload: UnvalidatedExecutionPlanPayload,
    plan_hash: PlanHash,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub(crate) struct UnvalidatedExecutionPlanWire {
    payload: UnvalidatedExecutionPlanPayload,
    plan_hash: PlanHash,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct UnvalidatedExecutionPlanWireFields {
    payload: UnvalidatedExecutionPlanPayload,
    plan_hash: PlanHash,
}

impl<'de> Deserialize<'de> for UnvalidatedExecutionPlanWire {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let raw = serde_json::Value::deserialize(deserializer)?;
        let fields = UnvalidatedExecutionPlanWireFields::deserialize(&raw)
            .map_err(serde::de::Error::custom)?;
        let canonical = serde_json::to_value(&fields).map_err(serde::de::Error::custom)?;
        if canonical != raw {
            return Err(serde::de::Error::custom(
                "execution plan wire contains unknown or non-canonical nested fields",
            ));
        }
        Ok(Self {
            payload: fields.payload,
            plan_hash: fields.plan_hash,
        })
    }
}

impl From<UnvalidatedExecutionPlanWire> for UnvalidatedExecutionPlan {
    fn from(wire: UnvalidatedExecutionPlanWire) -> Self {
        Self {
            payload: wire.payload,
            plan_hash: wire.plan_hash,
        }
    }
}

impl UnvalidatedExecutionPlan {
    pub fn schema(&self) -> PlanSchemaVersion {
        self.payload.schema
    }

    pub fn revalidate<P: RuntimePolicy>(
        self,
        family: &PreparedModelFamily,
        capabilities: &CapabilityCatalog,
        policy: &P,
        node_resolutions: Vec<PlanNodeResolution>,
    ) -> Result<ExecutionPlan, VNextError> {
        if self.payload.schema != EXECUTION_PLAN_SCHEMA {
            return Err(VNextError::UnsupportedPlanSchema {
                expected_major: EXECUTION_PLAN_SCHEMA.major,
                expected_minor: EXECUTION_PLAN_SCHEMA.minor,
                actual_major: self.payload.schema.major,
                actual_minor: self.payload.schema.minor,
            });
        }
        let rebuilt = ExecutionPlan::build(PlanBuildRequest::new(
            family,
            capabilities,
            policy,
            node_resolutions,
        )?)?;
        let untrusted_payload =
            serde_json::to_value(&self.payload).map_err(|error| VNextError::Serialization {
                context: "serialize unvalidated execution plan payload",
                message: error.to_string(),
            })?;
        let rebuilt_payload =
            serde_json::to_value(&rebuilt.payload).map_err(|error| VNextError::Serialization {
                context: "serialize rebuilt execution plan payload",
                message: error.to_string(),
            })?;
        if untrusted_payload != rebuilt_payload {
            return Err(invalid_plan(
                "untrusted plan differs from a semantic rebuild against current dependencies",
            ));
        }
        if rebuilt.plan_hash != self.plan_hash {
            return Err(VNextError::PlanHashMismatch {
                expected: rebuilt.plan_hash.to_string(),
                actual: self.plan_hash.to_string(),
            });
        }
        Ok(rebuilt)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ExecutionPlan {
    payload: ExecutionPlanPayload,
    plan_hash: PlanHash,
    #[serde(skip)]
    operation_registry_authority: OperationRegistryAuthority,
}

impl ExecutionPlan {
    pub fn build<P: RuntimePolicy>(request: PlanBuildRequest<'_, P>) -> Result<Self, VNextError> {
        request.policy.validate()?;
        let maximum_active_sequences = request.policy.maximum_active_sequences();
        validate_active_sequence_ceiling(maximum_active_sequences)?;
        let operation_registry_authority = request
            .node_resolutions
            .first()
            .ok_or_else(|| invalid_plan("plan build request has no node resolutions"))?
            .operation_registry_authority
            .clone();
        if request.node_resolutions.iter().any(|resolution| {
            resolution.operation_registry_authority != operation_registry_authority
        }) {
            return Err(invalid_plan(
                "node resolutions belong to different operation runtime registries",
            ));
        }
        let policy_capacity = request.policy.memory_capacity_bytes();
        let memory_reserve = request.policy.memory_reserve_bytes();
        let device_capacity = request.capabilities.device().total_memory_bytes;
        if policy_capacity == 0
            || policy_capacity > device_capacity
            || memory_reserve >= policy_capacity
        {
            return Err(invalid_plan(
                "runtime policy raw capacity, reserve, or typed admission concurrency is invalid for the device descriptor",
            ));
        }
        let family = request.family;
        let program = family.program();
        let prepared_family_fingerprint = family.fingerprint()?;
        let program_fingerprint = program.fingerprint()?;
        let capability_catalog_fingerprint = request.capabilities.fingerprint()?;
        let device_runtime_implementation_fingerprint = request
            .capabilities
            .device()
            .runtime_implementation_fingerprint
            .clone();
        let policy_fingerprint = canonical_runtime_policy_fingerprint(request.policy)?;
        let weight_format = family.weight_schema().format_id.clone();
        let quantization_formats = family.weight_schema().quantization_formats();

        let mut resolutions = BTreeMap::new();
        for resolution in request.node_resolutions {
            let node_id = resolution.node_id.clone();
            if resolutions.insert(node_id.clone(), resolution).is_some() {
                return Err(invalid_plan(format!(
                    "node `{node_id}` has duplicate physical resolutions"
                )));
            }
        }

        let program_nodes = program
            .blocks()
            .iter()
            .flat_map(|block| &block.nodes)
            .collect::<Vec<_>>();
        let joint_storage = Self::select_joint_provider_storage(
            &program_nodes,
            &resolutions,
            request.capabilities,
            request.policy,
        )?;
        let mut selected_node_resources = joint_storage.node_resources;
        let selected_resource_profiles = joint_storage.resource_profiles;
        let mut storage_rejections = joint_storage.storage_rejections;
        let producers = program_nodes
            .iter()
            .flat_map(|node| {
                node.outputs
                    .iter()
                    .map(move |output| (output.clone(), node.id.clone()))
            })
            .collect::<BTreeMap<_, _>>();
        let mut last_consumers = BTreeMap::<ProgramValueId, usize>::new();
        for (node_index, node) in program_nodes.iter().enumerate() {
            for input in &node.inputs {
                last_consumers.insert(input.clone(), node_index);
            }
        }
        let program_outputs = program.outputs().iter().cloned().collect::<BTreeSet<_>>();
        let mut canonical_values = BTreeMap::new();
        let mut bound_values = BTreeSet::new();
        let mut nodes = Vec::new();
        let mut state_dependencies = StateDependencyTracker::default();
        for (node_index, program_node) in program_nodes.into_iter().enumerate() {
            let resolution = resolutions.remove(&program_node.id).ok_or_else(|| {
                invalid_plan(format!(
                    "node `{}` has no physical resolution",
                    program_node.id
                ))
            })?;
            let provider_resources = selected_node_resources
                .remove(&program_node.id)
                .ok_or_else(|| invalid_plan("joint solver omitted one program node"))?;
            let storage_rejection = storage_rejections.remove(&program_node.id);
            let node = Self::build_node(
                family,
                program_node,
                resolution,
                provider_resources,
                storage_rejection,
                request.capabilities,
                &producers,
                node_index,
                &last_consumers,
                &program_outputs,
                &mut state_dependencies,
                &mut canonical_values,
                &mut bound_values,
            )?;
            nodes.push(node);
        }
        if !resolutions.is_empty() {
            return Err(invalid_plan(format!(
                "physical resolutions contain unknown nodes: {:?}",
                resolutions.keys().collect::<Vec<_>>()
            )));
        }
        if !selected_node_resources.is_empty() {
            return Err(invalid_plan(
                "joint solver returned resources for unknown program nodes",
            ));
        }
        if !storage_rejections.is_empty() {
            return Err(invalid_plan(
                "joint solver returned storage rejections for unknown program nodes",
            ));
        }
        Self::validate_semantic_coverage(family, &bound_values)?;
        Self::validate_global_storage_aliasing(&canonical_values, &nodes)?;
        let memory = Self::build_memory_plan(
            family,
            device_capacity,
            policy_capacity,
            memory_reserve,
            maximum_active_sequences,
            &nodes,
            &selected_resource_profiles,
        )?;

        let mut payload = ExecutionPlanPayload {
            schema: EXECUTION_PLAN_SCHEMA,
            plan_id: PlanId::new("plan/unset")?,
            family_id: family.family_id().clone(),
            device_id: request.capabilities.device().id.clone(),
            device_runtime_implementation_fingerprint,
            prepared_family_fingerprint,
            program_fingerprint,
            capability_catalog_fingerprint,
            policy_version: request.policy.version(),
            policy_fingerprint,
            weight_format,
            quantization_formats,
            nodes,
            memory,
        };
        let plan_hash = PlanHash::new(canonical_fingerprint(
            &PlanHashMaterial::from(&payload),
            "fingerprint execution plan",
        )?)?;
        payload.plan_id = Self::plan_id_for_hash(&plan_hash)?;
        let plan = Self {
            payload,
            plan_hash,
            operation_registry_authority,
        };
        plan.validate_internal()?;
        Ok(plan)
    }

    #[allow(clippy::too_many_arguments)]
    fn build_node(
        family: &PreparedModelFamily,
        program_node: &ProgramNode,
        resolution: PlanNodeResolution,
        provider_resources: ProviderResourcePlan,
        storage_rejection: Option<RejectedProvider>,
        catalog: &CapabilityCatalog,
        producers: &BTreeMap<ProgramValueId, NodeId>,
        node_index: usize,
        last_consumers: &BTreeMap<ProgramValueId, usize>,
        program_outputs: &BTreeSet<ProgramValueId>,
        state_dependencies: &mut StateDependencyTracker,
        canonical_values: &mut BTreeMap<ProgramValueId, CanonicalValueBinding>,
        bound_values: &mut BTreeSet<ProgramValueId>,
    ) -> Result<PlanNode, VNextError> {
        let operation = catalog.operation(&program_node.operation_id)?;
        if !operation.version.satisfies(program_node.required_version) {
            return Err(VNextError::IncompatibleOperationVersion {
                node_id: Some(program_node.id.to_string()),
                operation_id: program_node.operation_id.to_string(),
                required_major: program_node.required_version.major,
                required_minor: program_node.required_version.minor,
                available_major: operation.version.major,
                available_minor: operation.version.minor,
            });
        }
        operation.validate_attributes(&program_node.attributes)?;
        operation.validate_resolved_bindings(&resolution.values)?;
        Self::validate_program_bindings(program_node, &resolution.values)?;
        let exact_aliases = Self::extract_exact_aliases(operation, &resolution.values)?;
        Self::validate_alias_liveness(
            family,
            program_node,
            node_index,
            last_consumers,
            program_outputs,
            &exact_aliases,
            &resolution.values,
        )?;
        let state_effects = Self::derive_state_effects(family, &resolution.values)?;
        for binding in &resolution.values {
            Self::validate_semantic_binding(family, binding)?;
            Self::validate_cross_node_value(binding, canonical_values)?;
            bound_values.insert(binding.value_id().clone());
        }

        let (required_weight_formats, required_quantization_formats) =
            Self::node_weight_requirements(family, &resolution.values)?;
        let selection = Self::select_provider(
            program_node,
            operation,
            catalog,
            &resolution.required_capabilities,
            resolution.preferred_provider.as_ref(),
            &provider_resources.provider_id,
            storage_rejection,
            &required_weight_formats,
            &required_quantization_formats,
        )?;
        provider_resources.validate_shape()?;
        if provider_resources.provider_id != selection.selected_provider {
            return Err(invalid_plan(format!(
                "node `{}` resource estimate belongs to provider `{}` instead of selected provider `{}`",
                program_node.id,
                provider_resources.provider_id,
                selection.selected_provider
            )));
        }
        let selected_provider = catalog
            .providers_for(&program_node.operation_id)?
            .iter()
            .find(|provider| provider.provider_id() == &selection.selected_provider)
            .ok_or_else(|| {
                invalid_plan(format!(
                    "node `{}` selected provider is absent from the catalog",
                    program_node.id
                ))
            })?;
        if provider_resources.estimator_id != selected_provider.resource_estimator_id()
            || provider_resources.estimator_version
                != selected_provider.resource_estimator_version()
            || provider_resources.estimator_implementation_fingerprint
                != selected_provider.resource_estimator_implementation_fingerprint()
        {
            return Err(invalid_plan(format!(
                "node `{}` provider resource estimate is not issued by the selected catalog provider's estimator",
                program_node.id
            )));
        }
        let minimum_value_alignment = operation.resources.minimum_value_alignment_bytes;
        if provider_resources.value_alignment_bytes < minimum_value_alignment
            || provider_resources.value_alignment_bytes % minimum_value_alignment != 0
            || !operation
                .resources
                .scratch
                .accepts(provider_resources.scratch.is_some())
            || !operation
                .resources
                .persistent
                .accepts(provider_resources.persistent.is_some())
        {
            return Err(invalid_plan(format!(
                "node `{}` provider resource estimate violates the operation's alignment or workspace-presence contract",
                program_node.id
            )));
        }
        let expected_estimator_input = provider_resource_estimator_input_fingerprint(
            family,
            operation,
            program_node,
            &selection.selected_provider,
            &resolution.values,
            &resolution.required_capabilities,
        )?;
        if provider_resources.estimator_input_fingerprint != expected_estimator_input {
            return Err(invalid_plan(format!(
                "node `{}` provider resource estimate is not bound to its selected provider, shape, attributes, and bindings",
                program_node.id
            )));
        }
        let mut dependencies = program_node
            .inputs
            .iter()
            .filter_map(|input| producers.get(input))
            .cloned()
            .collect::<BTreeSet<_>>();
        Self::add_state_dependencies(
            &program_node.id,
            &state_effects,
            state_dependencies,
            &mut dependencies,
        );
        let dependencies = dependencies.into_iter().collect::<Vec<_>>();
        let scratch_resource = provider_resources
            .scratch
            .as_ref()
            .map(|_| {
                Self::workspace_base_id(
                    &program_node.id,
                    "scratch",
                    &provider_resources.estimate_fingerprint,
                )
            })
            .transpose()?;
        let persistent_resource = provider_resources
            .persistent
            .as_ref()
            .map(|_| {
                Self::workspace_base_id(
                    &program_node.id,
                    "persistent",
                    &provider_resources.estimate_fingerprint,
                )
            })
            .transpose()?;
        let resources = resolution
            .values
            .iter()
            .flat_map(|binding| binding.storage().components())
            .map(|component| component.resource_id().clone())
            .chain(scratch_resource.iter().cloned())
            .chain(persistent_resource.iter().cloned())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        Ok(PlanNode {
            id: program_node.id.clone(),
            dependencies,
            operation_id: program_node.operation_id.clone(),
            operation_version: program_node.required_version,
            operation_fingerprint: operation.fingerprint()?,
            provider_implementation_fingerprint: selected_provider
                .provider_implementation_fingerprint()
                .to_owned(),
            required_capabilities: resolution.required_capabilities,
            attributes: program_node.attributes.clone(),
            selection,
            provider_resources,
            values: resolution.values,
            exact_aliases,
            state_effects,
            scratch_resource,
            persistent_resource,
            resources,
        })
    }

    fn extract_exact_aliases(
        operation: &OperationDescriptor,
        bindings: &[ResolvedValueBinding],
    ) -> Result<Vec<PlanExactAlias>, VNextError> {
        let inputs = &bindings[..operation.inputs.len()];
        let outputs = &bindings[operation.inputs.len()..];
        let mut aliases = Vec::new();
        for (output_ordinal, output) in outputs.iter().enumerate() {
            let (input_ordinal, kind) = match output.alias() {
                AliasPolicy::NoAlias => continue,
                AliasPolicy::MayAlias { tensor_index } => {
                    (*tensor_index, PlanExactAliasKind::MayAlias)
                }
                AliasPolicy::MustAlias { tensor_index } => {
                    (*tensor_index, PlanExactAliasKind::MustAlias)
                }
            };
            let input = inputs.get(input_ordinal as usize).ok_or_else(|| {
                invalid_plan(format!(
                    "operation `{}` alias input ordinal is out of range after validation",
                    operation.id
                ))
            })?;
            if output.storage() == input.storage() {
                aliases.push(PlanExactAlias {
                    output_value_id: output.value_id().clone(),
                    output_ordinal: output_ordinal as u32,
                    input_value_id: input.value_id().clone(),
                    input_ordinal,
                    kind,
                });
            } else if kind == PlanExactAliasKind::MustAlias {
                return Err(invalid_plan(format!(
                    "operation `{}` lost its mandatory exact alias proof",
                    operation.id
                )));
            }
        }
        Ok(aliases)
    }

    #[allow(clippy::too_many_arguments)]
    fn validate_alias_liveness(
        family: &PreparedModelFamily,
        node: &ProgramNode,
        node_index: usize,
        last_consumers: &BTreeMap<ProgramValueId, usize>,
        program_outputs: &BTreeSet<ProgramValueId>,
        aliases: &[PlanExactAlias],
        bindings: &[ResolvedValueBinding],
    ) -> Result<(), VNextError> {
        for alias in aliases {
            let input = bindings
                .iter()
                .find(|binding| {
                    binding.role() == ResolvedValueRole::Input
                        && binding.ordinal() == alias.input_ordinal
                        && binding.value_id() == &alias.input_value_id
                })
                .ok_or_else(|| invalid_plan("exact alias input proof has no matching binding"))?;
            if family
                .program()
                .states()
                .iter()
                .any(|state| state.value_id == alias.input_value_id)
            {
                return Err(invalid_plan(format!(
                    "node `{}` output aliases state `{}` without a typed state transition contract",
                    node.id, alias.input_value_id
                )));
            }
            if input.usage() != BufferUsage::Activations
                || last_consumers.get(&alias.input_value_id) != Some(&node_index)
                || program_outputs.contains(&alias.input_value_id)
            {
                return Err(invalid_plan(format!(
                    "node `{}` aliases activation `{}` before its final legal consumer",
                    node.id, alias.input_value_id
                )));
            }
        }
        Ok(())
    }

    fn derive_state_effects(
        family: &PreparedModelFamily,
        bindings: &[ResolvedValueBinding],
    ) -> Result<Vec<PlanStateEffect>, VNextError> {
        let mut effects = Vec::new();
        for state in family.program().states() {
            let mut reads = false;
            let mut writes = false;
            let state_bindings = bindings
                .iter()
                .filter(|binding| binding.value_id() == &state.value_id)
                .collect::<Vec<_>>();
            for binding in &state_bindings {
                match binding.access() {
                    TensorAccess::Read => reads = true,
                    TensorAccess::Write => writes = true,
                    TensorAccess::ReadWrite => {
                        reads = true;
                        writes = true;
                    }
                }
            }
            let access = match (reads, writes) {
                (false, false) => continue,
                (true, false) => TensorAccess::Read,
                (false, true) => TensorAccess::Write,
                (true, true) => TensorAccess::ReadWrite,
            };
            let lifetime = match state.lifetime {
                StateLifetime::Request => AllocationLifetime::Request,
                StateLifetime::Sequence => AllocationLifetime::Sequence,
                StateLifetime::Step => AllocationLifetime::Step,
            };
            let resource_ids = state_bindings
                .iter()
                .flat_map(|binding| binding.storage().components())
                .map(|component| component.resource_id().clone())
                .collect::<BTreeSet<_>>()
                .into_iter()
                .collect::<Vec<_>>();
            if resource_ids.is_empty() {
                return Err(invalid_plan(format!(
                    "state `{}` effect has no physical resource closure",
                    state.id
                )));
            }
            effects.push(PlanStateEffect {
                state_id: state.id.clone(),
                state_value_id: state.value_id.clone(),
                lifetime,
                access,
                resource_ids,
            });
        }
        if effects
            .windows(2)
            .any(|pair| pair[0].state_id >= pair[1].state_id)
        {
            return Err(invalid_plan("state effects are not canonical"));
        }
        Ok(effects)
    }

    fn add_state_dependencies(
        node_id: &NodeId,
        effects: &[PlanStateEffect],
        tracker: &mut StateDependencyTracker,
        dependencies: &mut BTreeSet<NodeId>,
    ) {
        for effect in effects {
            let state_id = &effect.state_id;
            match effect.access {
                TensorAccess::Read => {
                    if let Some(writer) = tracker.last_writer.get(state_id) {
                        dependencies.insert(writer.clone());
                    }
                    tracker
                        .readers_since_write
                        .entry(state_id.clone())
                        .or_default()
                        .insert(node_id.clone());
                }
                TensorAccess::Write | TensorAccess::ReadWrite => {
                    if let Some(writer) = tracker.last_writer.get(state_id) {
                        dependencies.insert(writer.clone());
                    }
                    if let Some(readers) = tracker.readers_since_write.remove(state_id) {
                        dependencies.extend(readers);
                    }
                    tracker
                        .last_writer
                        .insert(state_id.clone(), node_id.clone());
                }
            }
        }
    }

    fn validate_program_bindings(
        node: &ProgramNode,
        bindings: &[ResolvedValueBinding],
    ) -> Result<(), VNextError> {
        let expected = node
            .inputs
            .iter()
            .enumerate()
            .map(|(ordinal, value)| (ResolvedValueRole::Input, ordinal as u32, value))
            .chain(
                node.outputs
                    .iter()
                    .enumerate()
                    .map(|(ordinal, value)| (ResolvedValueRole::Output, ordinal as u32, value)),
            )
            .collect::<Vec<_>>();
        if expected.len() != bindings.len()
            || expected.iter().zip(bindings).any(|(expected, actual)| {
                expected.0 != actual.role()
                    || expected.1 != actual.ordinal()
                    || expected.2 != actual.value_id()
            })
        {
            return Err(invalid_plan(format!(
                "node `{}` bindings do not match semantic program values",
                node.id
            )));
        }
        Ok(())
    }

    fn validate_semantic_binding(
        family: &PreparedModelFamily,
        binding: &ResolvedValueBinding,
    ) -> Result<(), VNextError> {
        if let Some(weight) = family
            .program()
            .weights()
            .iter()
            .find(|weight| &weight.value_id == binding.value_id())
        {
            if binding.usage() != BufferUsage::Weights || binding.access() != TensorAccess::Read {
                return Err(invalid_plan(format!(
                    "weight value `{}` is not backed by immutable weight memory",
                    binding.value_id()
                )));
            }
            Self::validate_program_tensor(&weight.tensor, binding.tensor(), "weight")?;
            Self::validate_weight_storage(family, &weight.weight_id, binding.storage())?;
            return Ok(());
        }
        if let Some(state) = family
            .program()
            .states()
            .iter()
            .find(|state| &state.value_id == binding.value_id())
        {
            if binding.usage() != BufferUsage::State {
                return Err(invalid_plan(format!(
                    "state value `{}` is not backed by state memory",
                    binding.value_id()
                )));
            }
            Self::validate_program_tensor(&state.tensor, binding.tensor(), "state")?;
            return Ok(());
        }
        if binding.usage() != BufferUsage::Activations {
            return Err(invalid_plan(format!(
                "semantic value `{}` must use activation memory",
                binding.value_id()
            )));
        }
        Ok(())
    }

    fn validate_program_tensor(
        expected: &ProgramTensorSpec,
        actual: &ResolvedTensorSpec,
        kind: &str,
    ) -> Result<(), VNextError> {
        if expected.dimensions != actual.dimensions()
            || expected.element_type != actual.element_type()
            || &expected.layout != actual.layout()
        {
            return Err(invalid_plan(format!(
                "{kind} binding shape, dtype, or layout differs from model semantics"
            )));
        }
        Ok(())
    }

    fn validate_weight_storage(
        family: &PreparedModelFamily,
        weight_id: &WeightId,
        storage: &ResolvedValueStorage,
    ) -> Result<(), VNextError> {
        let expected = family
            .weight_schema()
            .physical_component_refs(weight_id)?
            .into_iter()
            .map(|component| (component.id.clone(), component))
            .collect::<BTreeMap<_, _>>();
        if storage.components().len() != expected.len() {
            return Err(invalid_plan(format!(
                "weight `{weight_id}` physical component count differs from its schema"
            )));
        }
        let mut seen = BTreeSet::new();
        for component in storage.components() {
            let component_id = component.component_id().ok_or_else(|| {
                invalid_plan(format!(
                    "weight `{weight_id}` storage lacks a physical component identity"
                ))
            })?;
            let spec = expected.get(component_id).ok_or_else(|| {
                invalid_plan(format!(
                    "weight `{weight_id}` binds unknown physical component `{component_id}`"
                ))
            })?;
            let expected_element_type = match &spec.encoding {
                WeightEncoding::Dense { element_type } => *element_type,
                WeightEncoding::Quantized(_) => ElementType::U8,
            };
            if !seen.insert(component_id.clone())
                || component.length_bytes() != spec.physical_bytes()?
                || component.element_type() != expected_element_type
            {
                return Err(invalid_plan(format!(
                    "weight `{weight_id}` component `{component_id}` byte length or dtype differs from schema"
                )));
            }
        }
        if seen != expected.keys().cloned().collect() {
            return Err(invalid_plan(format!(
                "weight `{weight_id}` physical component identities are incomplete"
            )));
        }
        Ok(())
    }

    fn validate_cross_node_value(
        binding: &ResolvedValueBinding,
        values: &mut BTreeMap<ProgramValueId, CanonicalValueBinding>,
    ) -> Result<(), VNextError> {
        let canonical = CanonicalValueBinding {
            tensor: binding.tensor().clone(),
            usage: binding.usage(),
            storage: binding.storage().clone(),
        };
        match values.get(binding.value_id()) {
            Some(previous) if previous != &canonical => Err(invalid_plan(format!(
                "value `{}` changes tensor or physical storage between nodes",
                binding.value_id()
            ))),
            Some(_) => Ok(()),
            None => {
                values.insert(binding.value_id().clone(), canonical);
                Ok(())
            }
        }
    }

    fn validate_global_storage_aliasing(
        values: &BTreeMap<ProgramValueId, CanonicalValueBinding>,
        nodes: &[PlanNode],
    ) -> Result<(), VNextError> {
        let alias_classes = Self::alias_classes(nodes)?;
        let mut by_resource = BTreeMap::<ResourceId, Vec<GlobalValueRange>>::new();
        for (value_id, binding) in values {
            for component in binding.storage.components() {
                let end_bytes = component
                    .offset_bytes()
                    .checked_add(component.length_bytes())
                    .ok_or_else(|| invalid_plan("global value storage range overflows u64"))?;
                let ranges = by_resource
                    .entry(component.resource_id().clone())
                    .or_default();
                if let Some(previous) = ranges.iter().find(|previous| {
                    previous.value_id != *value_id
                        && previous.offset_bytes < end_bytes
                        && component.offset_bytes() < previous.end_bytes
                }) {
                    let same_alias_class = alias_classes.get(&previous.value_id).is_some()
                        && alias_classes.get(&previous.value_id) == alias_classes.get(value_id);
                    let previous_binding = values.get(&previous.value_id).ok_or_else(|| {
                        invalid_plan("global alias range has no canonical value binding")
                    })?;
                    if !same_alias_class || previous_binding.storage != binding.storage {
                        return Err(invalid_plan(format!(
                            "values `{}` and `{value_id}` have undeclared, partial, or non-equivalent overlap in physical resource `{}`",
                            previous.value_id,
                            component.resource_id()
                        )));
                    }
                }
                ranges.push(GlobalValueRange {
                    value_id: value_id.clone(),
                    offset_bytes: component.offset_bytes(),
                    end_bytes,
                });
            }
        }
        Ok(())
    }

    fn alias_classes(
        nodes: &[PlanNode],
    ) -> Result<BTreeMap<ProgramValueId, ProgramValueId>, VNextError> {
        let mut graph = BTreeMap::<ProgramValueId, BTreeSet<ProgramValueId>>::new();
        for node in nodes {
            let mut previous_output_ordinal = None;
            for alias in &node.exact_aliases {
                if previous_output_ordinal.is_some_and(|ordinal| ordinal >= alias.output_ordinal) {
                    return Err(invalid_plan(format!(
                        "node `{}` exact aliases are not canonical",
                        node.id
                    )));
                }
                previous_output_ordinal = Some(alias.output_ordinal);
                let input = node
                    .values
                    .iter()
                    .find(|binding| {
                        binding.role() == ResolvedValueRole::Input
                            && binding.ordinal() == alias.input_ordinal
                            && binding.value_id() == &alias.input_value_id
                    })
                    .ok_or_else(|| invalid_plan("plan exact alias input binding is missing"))?;
                let output = node
                    .values
                    .iter()
                    .find(|binding| {
                        binding.role() == ResolvedValueRole::Output
                            && binding.ordinal() == alias.output_ordinal
                            && binding.value_id() == &alias.output_value_id
                    })
                    .ok_or_else(|| invalid_plan("plan exact alias output binding is missing"))?;
                let policy_matches = matches!(
                    (output.alias(), alias.kind),
                    (
                        AliasPolicy::MayAlias { tensor_index },
                        PlanExactAliasKind::MayAlias
                    ) if *tensor_index == alias.input_ordinal
                ) || matches!(
                    (output.alias(), alias.kind),
                    (
                        AliasPolicy::MustAlias { tensor_index },
                        PlanExactAliasKind::MustAlias
                    ) if *tensor_index == alias.input_ordinal
                );
                if !policy_matches
                    || input.storage() != output.storage()
                    || input.usage() != BufferUsage::Activations
                    || output.usage() != BufferUsage::Activations
                {
                    return Err(invalid_plan(format!(
                        "node `{}` exact alias proof differs from its bindings",
                        node.id
                    )));
                }
                graph
                    .entry(alias.input_value_id.clone())
                    .or_default()
                    .insert(alias.output_value_id.clone());
                graph
                    .entry(alias.output_value_id.clone())
                    .or_default()
                    .insert(alias.input_value_id.clone());
            }
        }

        let mut classes = BTreeMap::new();
        let mut visited = BTreeSet::new();
        for start in graph.keys() {
            if visited.contains(start) {
                continue;
            }
            let mut pending = vec![start.clone()];
            let mut members = BTreeSet::new();
            while let Some(value) = pending.pop() {
                if !visited.insert(value.clone()) {
                    continue;
                }
                members.insert(value.clone());
                if let Some(neighbors) = graph.get(&value) {
                    pending.extend(neighbors.iter().cloned());
                }
            }
            let representative = members
                .first()
                .cloned()
                .ok_or_else(|| invalid_plan("empty alias equivalence class"))?;
            for member in members {
                classes.insert(member, representative.clone());
            }
        }
        Ok(classes)
    }

    fn validate_semantic_coverage(
        family: &PreparedModelFamily,
        bound: &BTreeSet<ProgramValueId>,
    ) -> Result<(), VNextError> {
        let required = family
            .program()
            .inputs()
            .iter()
            .cloned()
            .chain(
                family
                    .program()
                    .weights()
                    .iter()
                    .map(|weight| weight.value_id.clone()),
            )
            .chain(
                family
                    .program()
                    .states()
                    .iter()
                    .map(|state| state.value_id.clone()),
            )
            .chain(family.program().outputs().iter().cloned())
            .collect::<BTreeSet<_>>();
        if !required.is_subset(bound) {
            return Err(invalid_plan(format!(
                "semantic values lack physical bindings: {:?}",
                required.difference(bound).collect::<Vec<_>>()
            )));
        }
        Ok(())
    }

    fn node_weight_requirements(
        family: &PreparedModelFamily,
        bindings: &[ResolvedValueBinding],
    ) -> Result<(BTreeSet<WeightFormatId>, BTreeSet<QuantizationFormatId>), VNextError> {
        let mut weight_formats = BTreeSet::new();
        let mut quantization_formats = BTreeSet::new();
        for binding in bindings
            .iter()
            .filter(|binding| binding.usage() == BufferUsage::Weights)
        {
            let weight = family
                .program()
                .weights()
                .iter()
                .find(|weight| weight.value_id == *binding.value_id())
                .ok_or_else(|| {
                    invalid_plan(format!(
                        "weight binding `{}` is not declared by the model program",
                        binding.value_id()
                    ))
                })?;
            weight_formats.insert(family.weight_schema().format_id.clone());
            for component in family
                .weight_schema()
                .physical_component_refs(&weight.weight_id)?
            {
                if let WeightEncoding::Quantized(spec) = &component.encoding {
                    quantization_formats.insert(spec.format_id.clone());
                }
            }
        }
        Ok((weight_formats, quantization_formats))
    }

    fn available_storage_profiles<P: RuntimePolicy>(
        requirement: &DynamicStorageRequirement,
        catalog: &CapabilityCatalog,
        policy: &P,
    ) -> BTreeSet<DynamicStorageProfile> {
        policy
            .dynamic_storage_profile_order()
            .iter()
            .copied()
            .filter(|profile| {
                catalog.device().dynamic_storage_profiles.contains(profile)
                    && requirement.accepts(*profile)
            })
            .collect()
    }

    fn merge_storage_constraint(
        constraints: &mut BTreeMap<ResourceId, BTreeSet<DynamicStorageProfile>>,
        resource_id: ResourceId,
        accepted: BTreeSet<DynamicStorageProfile>,
    ) -> bool {
        if accepted.is_empty() {
            return false;
        }
        match constraints.get_mut(&resource_id) {
            Some(existing) => {
                existing.retain(|profile| accepted.contains(profile));
                !existing.is_empty()
            }
            None => {
                constraints.insert(resource_id, accepted);
                true
            }
        }
    }

    fn select_joint_provider_storage<P: RuntimePolicy>(
        program_nodes: &[&ProgramNode],
        resolutions: &BTreeMap<NodeId, PlanNodeResolution>,
        catalog: &CapabilityCatalog,
        policy: &P,
    ) -> Result<JointProviderStorageSelection, VNextError> {
        let mut candidate_sets = Vec::with_capacity(program_nodes.len());
        for node in program_nodes {
            let resolution = resolutions.get(&node.id).ok_or_else(|| {
                invalid_plan(format!("node `{}` has no physical resolution", node.id))
            })?;
            let providers = catalog.providers_for(&node.operation_id)?;
            let mut candidates = Vec::new();
            for resources in &resolution.provider_resource_candidates {
                let provider = providers
                    .iter()
                    .find(|provider| provider.provider_id() == resources.provider_id())
                    .ok_or_else(|| {
                        invalid_plan("provider resource candidate is absent from the catalog")
                    })?;
                let mut constraints = BTreeMap::new();
                let mut compatible = true;
                for binding in resolution
                    .values
                    .iter()
                    .filter(|binding| binding.usage() != BufferUsage::Weights)
                {
                    let Some(requirement) =
                        provider.dynamic_storage_for(binding.role(), binding.ordinal())
                    else {
                        compatible = false;
                        break;
                    };
                    let accepted = Self::available_storage_profiles(requirement, catalog, policy);
                    for component in binding.storage().components() {
                        if !Self::merge_storage_constraint(
                            &mut constraints,
                            component.resource_id().clone(),
                            accepted.clone(),
                        ) {
                            compatible = false;
                            break;
                        }
                    }
                    if !compatible {
                        break;
                    }
                }
                for (kind, workspace) in [
                    ("scratch", resources.scratch()),
                    ("persistent", resources.persistent()),
                ] {
                    let Some(workspace) = workspace else {
                        continue;
                    };
                    let resource_id =
                        Self::workspace_base_id(&node.id, kind, resources.estimate_fingerprint())?;
                    if !Self::merge_storage_constraint(
                        &mut constraints,
                        resource_id,
                        Self::available_storage_profiles(workspace.storage(), catalog, policy),
                    ) {
                        compatible = false;
                        break;
                    }
                }
                if compatible {
                    let is_preferred = resolution
                        .preferred_provider
                        .as_ref()
                        .is_some_and(|preferred| preferred == resources.provider_id());
                    candidates.push(JointProviderCandidate {
                        resources: resources.clone(),
                        allowed_profiles: constraints,
                        is_preferred,
                    });
                }
            }
            candidates.sort_by(|left, right| {
                let left_preferred = resolution
                    .preferred_provider
                    .as_ref()
                    .is_some_and(|preferred| preferred == left.resources.provider_id());
                let right_preferred = resolution
                    .preferred_provider
                    .as_ref()
                    .is_some_and(|preferred| preferred == right.resources.provider_id());
                right_preferred.cmp(&left_preferred).then(
                    left.resources
                        .provider_id()
                        .cmp(right.resources.provider_id()),
                )
            });
            if candidates.is_empty() {
                return Err(invalid_plan(format!(
                    "node `{}` has no provider candidate with an available storage profile",
                    node.id
                )));
            }
            candidate_sets.push(candidates);
        }

        let (chosen, resource_profiles) = Self::solve_joint_provider_candidates(
            &candidate_sets,
            policy.dynamic_storage_profile_order(),
        )?;

        let mut storage_rejections = BTreeMap::new();
        for (index, node) in program_nodes.iter().enumerate() {
            let resolution = resolutions
                .get(&node.id)
                .ok_or_else(|| invalid_plan("joint storage resolution disappeared"))?;
            let Some(preferred) = resolution.preferred_provider.as_ref() else {
                continue;
            };
            if chosen[index].provider_id() == preferred {
                continue;
            }
            let Some(preferred_candidate) = candidate_sets[index]
                .iter()
                .find(|candidate| candidate.resources.provider_id() == preferred)
            else {
                if let Some(reason) = resolution.provider_resolution_rejections.get(preferred) {
                    storage_rejections.insert(
                        node.id.clone(),
                        RejectedProvider {
                            provider_id: preferred.clone(),
                            reasons: reason.clone(),
                        },
                    );
                }
                continue;
            };
            let resource_ids =
                storage_incompatible_resource_ids(preferred_candidate, &resource_profiles);
            if resource_ids.is_empty() {
                return Err(invalid_plan(format!(
                    "preferred provider `{preferred}` was not selected for node `{}` without a storage conflict",
                    node.id
                )));
            }
            storage_rejections.insert(
                node.id.clone(),
                RejectedProvider {
                    provider_id: preferred.clone(),
                    reasons: PlanProviderRejectReason::StorageIncompatible { resource_ids },
                },
            );
        }

        let node_resources = program_nodes
            .iter()
            .zip(chosen)
            .map(|(node, resources)| (node.id.clone(), resources))
            .collect();
        Ok(JointProviderStorageSelection {
            node_resources,
            resource_profiles,
            storage_rejections,
        })
    }

    fn solve_joint_provider_candidates(
        candidate_sets: &[Vec<JointProviderCandidate>],
        profile_order: &[DynamicStorageProfile],
    ) -> Result<
        (
            Vec<ProviderResourcePlan>,
            BTreeMap<ResourceId, DynamicStorageProfile>,
        ),
        VNextError,
    > {
        if candidate_sets.is_empty() || candidate_sets.iter().any(Vec::is_empty) {
            return Err(invalid_plan(
                "joint provider/storage search has an empty candidate set",
            ));
        }
        if profile_order.is_empty() {
            return Err(invalid_plan(
                "joint provider/storage search has an empty profile order",
            ));
        }

        let components = joint_candidate_components(candidate_sets);
        let mut chosen = vec![None; candidate_sets.len()];
        let mut resource_profiles = BTreeMap::new();
        for component in components {
            let solution =
                Self::solve_joint_provider_component(&component, candidate_sets, profile_order)?;
            for (node_index, resources) in component.iter().copied().zip(solution.chosen) {
                if chosen[node_index].replace(resources).is_some() {
                    return Err(invalid_plan(
                        "joint storage component assigned one node more than once",
                    ));
                }
            }
            for (resource_id, profile) in solution.resource_profiles {
                if resource_profiles.insert(resource_id, profile).is_some() {
                    return Err(invalid_plan(
                        "joint storage components overlap one resource",
                    ));
                }
            }
        }
        let chosen = chosen
            .into_iter()
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| invalid_plan("joint storage components omitted one node"))?;
        Ok((chosen, resource_profiles))
    }

    fn solve_joint_provider_component(
        component: &[usize],
        candidate_sets: &[Vec<JointProviderCandidate>],
        profile_order: &[DynamicStorageProfile],
    ) -> Result<JointComponentSolution, VNextError> {
        let mut frontier = BTreeMap::from([(
            BTreeMap::<ResourceId, BTreeSet<DynamicStorageProfile>>::new(),
            JointPartialSelection::default(),
        )]);
        for node_index in component {
            let mut next = BTreeMap::<
                BTreeMap<ResourceId, BTreeSet<DynamicStorageProfile>>,
                JointPartialSelection,
            >::new();
            for (constraints, partial) in frontier {
                for candidate in &candidate_sets[*node_index] {
                    let mut next_constraints = constraints.clone();
                    if candidate
                        .allowed_profiles
                        .iter()
                        .any(|(resource_id, accepted)| {
                            !Self::merge_storage_constraint(
                                &mut next_constraints,
                                resource_id.clone(),
                                accepted.clone(),
                            )
                        })
                    {
                        continue;
                    }
                    let mut next_partial = partial.clone();
                    next_partial.chosen.push(candidate.resources.clone());
                    next_partial.preferred.push(candidate.is_preferred);
                    match next.entry(next_constraints) {
                        std::collections::btree_map::Entry::Vacant(entry) => {
                            entry.insert(next_partial);
                        }
                        std::collections::btree_map::Entry::Occupied(mut entry) => {
                            if joint_partial_precedes(&next_partial, entry.get()) {
                                entry.insert(next_partial);
                            }
                        }
                    }
                }
            }
            if next.is_empty() {
                return Err(invalid_plan(
                    "no joint provider/storage assignment satisfies shared resource constraints",
                ));
            }
            frontier = next;
        }

        let mut best: Option<(JointSelectionObjective, JointComponentSolution)> = None;
        for (constraints, partial) in frontier {
            let resource_profiles = constraints
                .into_iter()
                .map(|(resource_id, accepted)| {
                    let (rank, profile) = profile_order
                        .iter()
                        .copied()
                        .enumerate()
                        .find(|(_, profile)| accepted.contains(profile))
                        .ok_or_else(|| {
                            invalid_plan("joint storage solution lost policy-ordered profile")
                        })?;
                    Ok((resource_id, (rank, profile)))
                })
                .collect::<Result<BTreeMap<_, _>, VNextError>>()?;
            let objective = JointSelectionObjective::new(
                &partial,
                resource_profiles.values().map(|(rank, _)| *rank),
                profile_order.len(),
            )?;
            let solution = JointComponentSolution {
                chosen: partial.chosen,
                resource_profiles: resource_profiles
                    .into_iter()
                    .map(|(resource_id, (_, profile))| (resource_id, profile))
                    .collect(),
            };
            if best
                .as_ref()
                .is_none_or(|(current, _)| objective.precedes(current))
            {
                best = Some((objective, solution));
            }
        }
        best.map(|(_, solution)| solution).ok_or_else(|| {
            invalid_plan("no joint provider/storage assignment satisfies one component")
        })
    }

    fn select_provider(
        node: &ProgramNode,
        operation: &OperationDescriptor,
        catalog: &CapabilityCatalog,
        resolution_required_capabilities: &BTreeSet<CapabilityId>,
        preferred_provider: Option<&ProviderId>,
        storage_selected_provider: &ProviderId,
        storage_rejection: Option<RejectedProvider>,
        required_weight_formats: &BTreeSet<WeightFormatId>,
        required_quantization_formats: &BTreeSet<QuantizationFormatId>,
    ) -> Result<ProviderSelection, VNextError> {
        let required_capabilities = operation
            .provider
            .required_capabilities
            .union(resolution_required_capabilities)
            .cloned()
            .collect::<BTreeSet<_>>();
        let request = ProviderCompatibilityRequest::new(
            node.operation_id.clone(),
            node.required_version,
            required_capabilities,
            required_weight_formats.clone(),
            required_quantization_formats.clone(),
        )?;
        let report = catalog.provider_compatibility(request)?;
        report.require_compatible(&catalog.device().id)?;
        if !report
            .compatible_provider_ids()
            .contains(storage_selected_provider)
        {
            return Err(invalid_plan(format!(
                "joint storage solver selected incompatible provider `{storage_selected_provider}`"
            )));
        }
        let selected_provider = storage_selected_provider.clone();
        let selection_reason = match preferred_provider {
            Some(preferred) if preferred == &selected_provider => {
                ProviderSelectionReason::PreferredCompatible
            }
            Some(_) => ProviderSelectionReason::FallbackFromPreferred,
            None => ProviderSelectionReason::DeterministicCompatible,
        };
        let mut rejected_providers = report
            .rejected()
            .iter()
            .map(|rejection| RejectedProvider {
                provider_id: rejection.provider_id.clone(),
                reasons: PlanProviderRejectReason::Incompatible(rejection.reasons.clone()),
            })
            .collect::<Vec<_>>();
        if let Some(preferred) = preferred_provider {
            let registered = catalog
                .providers_for(&node.operation_id)?
                .iter()
                .any(|provider| provider.provider_id() == preferred);
            if !registered {
                rejected_providers.push(RejectedProvider {
                    provider_id: preferred.clone(),
                    reasons: PlanProviderRejectReason::NotRegistered,
                });
            }
        }
        if let Some(rejection) = storage_rejection {
            if rejected_providers
                .iter()
                .any(|existing| existing.provider_id == rejection.provider_id)
            {
                return Err(invalid_plan(
                    "provider has duplicate compatibility and storage rejection evidence",
                ));
            }
            rejected_providers.push(rejection);
        }
        if let Some(preferred) =
            preferred_provider.filter(|preferred| *preferred != &selected_provider)
        {
            if !rejected_providers
                .iter()
                .any(|rejection| &rejection.provider_id == preferred)
            {
                return Err(invalid_plan(format!(
                    "preferred provider `{preferred}` fallback lacks typed rejection evidence"
                )));
            }
        }
        rejected_providers.sort_by(|left, right| left.provider_id.cmp(&right.provider_id));
        Ok(ProviderSelection {
            requested_provider: preferred_provider.cloned(),
            selected_provider,
            selection_reason,
            rejected_providers,
        })
    }

    fn validate_provider_selection_evidence(
        selection: &ProviderSelection,
    ) -> Result<(), VNextError> {
        if selection
            .rejected_providers
            .windows(2)
            .any(|pair| pair[0].provider_id >= pair[1].provider_id)
            || selection
                .rejected_providers
                .iter()
                .any(|rejection| rejection.provider_id == selection.selected_provider)
            || selection.rejected_providers.iter().any(|rejection| {
                matches!(
                    &rejection.reasons,
                    PlanProviderRejectReason::StorageIncompatible { resource_ids }
                        if resource_ids.is_empty()
                            || resource_ids.windows(2).any(|pair| pair[0] >= pair[1])
                )
            })
        {
            return Err(invalid_plan(
                "provider rejection evidence is duplicate, non-canonical, or rejects the selected provider",
            ));
        }
        match (
            selection.requested_provider.as_ref(),
            selection.selection_reason,
        ) {
            (None, ProviderSelectionReason::DeterministicCompatible) => {}
            (Some(requested), ProviderSelectionReason::PreferredCompatible)
                if requested == &selection.selected_provider => {}
            (Some(requested), ProviderSelectionReason::FallbackFromPreferred)
                if requested != &selection.selected_provider
                    && selection
                        .rejected_providers
                        .iter()
                        .any(|rejection| &rejection.provider_id == requested) => {}
            _ => return Err(invalid_plan(
                "provider selection reason is inconsistent with preference and rejection evidence",
            )),
        }
        Ok(())
    }

    fn workspace_base_id(
        node_id: &NodeId,
        kind: &str,
        estimate_fingerprint: &str,
    ) -> Result<ResourceId, VNextError> {
        if !is_canonical_sha256(estimate_fingerprint) {
            return Err(invalid_plan(
                "provider workspace identity has invalid estimate fingerprint",
            ));
        }
        let digest =
            Sha256::digest(format!("{kind}\0{node_id}\0{estimate_fingerprint}").as_bytes());
        ResourceId::new(format!("resource/{kind}/sha256/{digest:x}"))
    }

    fn build_memory_plan(
        family: &PreparedModelFamily,
        device_capacity_bytes: u64,
        policy_capacity_bytes: u64,
        reserve_bytes: u64,
        maximum_active_sequences: u32,
        nodes: &[PlanNode],
        selected_resource_profiles: &BTreeMap<ResourceId, DynamicStorageProfile>,
    ) -> Result<MemoryPlan, VNextError> {
        let mut values = BTreeMap::<ResourceId, ValueAllocationAccumulator>::new();
        let mut static_allocations = Vec::new();
        let mut dynamic_descriptors = Vec::new();
        let workspace_layout_fingerprint = workspace_storage_layout_fingerprint()?;
        for node in nodes {
            let value_alignment = node.provider_resources.value_alignment_bytes;
            for binding in &node.values {
                let logical_layout_fingerprint =
                    tensor_storage_layout_fingerprint(binding.tensor().layout())?;
                for component in binding.storage().components() {
                    if component.offset_bytes() % value_alignment != 0 {
                        return Err(invalid_plan(format!(
                            "resource `{}` offset is not aligned for provider `{}`",
                            component.resource_id(),
                            node.provider_resources.provider_id
                        )));
                    }
                    let end = component
                        .offset_bytes()
                        .checked_add(component.length_bytes())
                        .ok_or_else(|| invalid_plan("resource byte range overflows u64"))?;
                    let demand = Self::value_resource_demand(family, binding.value_id(), end)?;
                    values
                        .entry(component.resource_id().clone())
                        .and_modify(|allocation| {
                            allocation.merge_result =
                                allocation.merge_result.take().and_then(|_| {
                                    allocation.merge(
                                        end,
                                        value_alignment,
                                        binding.usage(),
                                        component.element_type(),
                                        demand,
                                        logical_layout_fingerprint.clone(),
                                    )
                                });
                        })
                        .or_insert_with(|| ValueAllocationAccumulator {
                            end_bytes: end,
                            alignment_bytes: value_alignment,
                            usage: binding.usage(),
                            element_type: component.element_type(),
                            demand,
                            logical_layout_fingerprints: BTreeSet::from([
                                logical_layout_fingerprint.clone(),
                            ]),
                            merge_result: Some(()),
                        });
                }
            }
            if let Some(workspace) = &node.provider_resources.scratch {
                let resource_id = node.scratch_resource.clone().ok_or_else(|| {
                    invalid_plan(format!(
                        "node `{}` scratch base identity is missing",
                        node.id
                    ))
                })?;
                if workspace.scope != ProviderWorkspaceScope::Invocation {
                    return Err(invalid_plan(format!(
                        "node `{}` scratch workspace is not invocation scoped",
                        node.id
                    )));
                }
                let storage = DynamicStorageContract::new(
                    *selected_resource_profiles
                        .get(&resource_id)
                        .ok_or_else(|| {
                            invalid_plan(format!(
                                "scratch resource `{resource_id}` has no selected storage profile"
                            ))
                        })?,
                    workspace_layout_fingerprint.clone(),
                )?;
                dynamic_descriptors.push(DynamicResourceDescriptor::new(
                    resource_id,
                    workspace.size_formula.clone(),
                    workspace.alignment_bytes,
                    BufferUsage::Scratch,
                    ElementType::U8,
                    AllocationLifetime::Invocation,
                    AllocationKind::Scratch {
                        node_id: node.id.clone(),
                    },
                    storage,
                    maximum_active_sequences,
                )?);
            } else if node.scratch_resource.is_some() {
                return Err(invalid_plan(format!(
                    "node `{}` has scratch resources without a provider estimate",
                    node.id
                )));
            }
            if let Some(workspace) = &node.provider_resources.persistent {
                let resource_id = node.persistent_resource.clone().ok_or_else(|| {
                    invalid_plan(format!(
                        "node `{}` persistent base identity is missing",
                        node.id
                    ))
                })?;
                match workspace.scope {
                    ProviderWorkspaceScope::Plan => {
                        let bytes = workspace.fixed_bytes().ok_or_else(|| {
                            invalid_plan(format!(
                                "node `{}` plan workspace does not have a fixed formula",
                                node.id
                            ))
                        })?;
                        let storage = DynamicStorageContract::new(
                            *selected_resource_profiles
                                .get(&resource_id)
                                .ok_or_else(|| {
                                    invalid_plan(format!(
                                    "plan workspace `{resource_id}` has no selected storage profile"
                                ))
                                })?,
                            workspace_layout_fingerprint.clone(),
                        )?;
                        static_allocations.push(ResourceAllocation::new(
                            resource_id,
                            bytes,
                            workspace.alignment_bytes,
                            BufferUsage::Persistent,
                            ElementType::U8,
                            AllocationKind::Persistent {
                                node_id: node.id.clone(),
                            },
                            storage,
                        )?);
                    }
                    scope @ (ProviderWorkspaceScope::Request
                    | ProviderWorkspaceScope::Sequence
                    | ProviderWorkspaceScope::Step) => {
                        let lifetime = match scope {
                            ProviderWorkspaceScope::Request => AllocationLifetime::Request,
                            ProviderWorkspaceScope::Sequence => AllocationLifetime::Sequence,
                            ProviderWorkspaceScope::Step => AllocationLifetime::Step,
                            ProviderWorkspaceScope::Plan | ProviderWorkspaceScope::Invocation => {
                                unreachable!()
                            }
                        };
                        let storage = DynamicStorageContract::new(
                            *selected_resource_profiles.get(&resource_id).ok_or_else(|| {
                                invalid_plan(format!(
                                    "persistent resource `{resource_id}` has no selected storage profile"
                                ))
                            })?,
                            workspace_layout_fingerprint.clone(),
                        )?;
                        dynamic_descriptors.push(DynamicResourceDescriptor::new(
                            resource_id,
                            workspace.size_formula.clone(),
                            workspace.alignment_bytes,
                            BufferUsage::Persistent,
                            ElementType::U8,
                            lifetime,
                            AllocationKind::Persistent {
                                node_id: node.id.clone(),
                            },
                            storage,
                            maximum_active_sequences,
                        )?);
                    }
                    ProviderWorkspaceScope::Invocation => {
                        return Err(invalid_plan(format!(
                            "node `{}` persistent workspace cannot be invocation scoped",
                            node.id
                        )));
                    }
                }
            } else if node.persistent_resource.is_some() {
                return Err(invalid_plan(format!(
                    "node `{}` has persistent resources without a provider estimate",
                    node.id
                )));
            }
        }
        for (resource_id, accumulator) in values {
            accumulator.merge_result.ok_or_else(|| {
                invalid_plan(format!(
                    "resource `{resource_id}` has conflicting usage, dtype, lifetime, or demand"
                ))
            })?;
            let logical_layout_fingerprint = canonical_fingerprint(
                &accumulator.logical_layout_fingerprints,
                "fingerprint dynamic resource tensor layout classes",
            )?;
            match accumulator.demand {
                ValueResourceDemand::PlanStatic => {
                    let storage = DynamicStorageContract::new(
                        static_contiguous_storage_profile()?,
                        logical_layout_fingerprint,
                    )?;
                    static_allocations.push(ResourceAllocation::new(
                        resource_id,
                        accumulator.end_bytes,
                        accumulator.alignment_bytes,
                        accumulator.usage,
                        accumulator.element_type,
                        AllocationKind::Value,
                        storage,
                    )?);
                }
                demand => {
                    let storage = DynamicStorageContract::new(
                        *selected_resource_profiles.get(&resource_id).ok_or_else(|| {
                            invalid_plan(format!(
                                "dynamic value resource `{resource_id}` has no selected storage profile"
                            ))
                        })?,
                        logical_layout_fingerprint,
                    )?;
                    dynamic_descriptors.push(DynamicResourceDescriptor::new(
                        resource_id,
                        demand.dynamic_demand(accumulator.end_bytes)?,
                        accumulator.alignment_bytes,
                        accumulator.usage,
                        accumulator.element_type,
                        demand.lifetime().ok_or_else(|| {
                            invalid_plan("dynamic value demand lost its scoped lifetime")
                        })?,
                        AllocationKind::Value,
                        storage,
                        maximum_active_sequences,
                    )?);
                }
            }
        }
        MemoryPlan::from_core(
            device_capacity_bytes,
            policy_capacity_bytes,
            reserve_bytes,
            maximum_active_sequences,
            static_allocations,
            dynamic_descriptors,
            nodes,
        )
    }

    fn value_resource_demand(
        family: &PreparedModelFamily,
        value_id: &ProgramValueId,
        minimum_bytes: u64,
    ) -> Result<ValueResourceDemand, VNextError> {
        if family
            .program()
            .weights()
            .iter()
            .any(|weight| &weight.value_id == value_id)
        {
            return Ok(ValueResourceDemand::PlanStatic);
        }
        let state = family
            .program()
            .states()
            .iter()
            .find(|state| &state.value_id == value_id);
        let Some(state) = state else {
            return Ok(ValueResourceDemand::Fixed {
                lifetime: AllocationLifetime::Request,
            });
        };
        let lifetime = match state.lifetime {
            StateLifetime::Request => AllocationLifetime::Request,
            StateLifetime::Sequence => AllocationLifetime::Sequence,
            StateLifetime::Step => AllocationLifetime::Step,
        };
        state.capacity_demand.validate(state.tensor.byte_len()?)?;
        match state.capacity_demand {
            StateCapacityDemand::FixedPerScope => Ok(ValueResourceDemand::Fixed { lifetime }),
            StateCapacityDemand::TokenScaled {
                bytes_per_token,
                maximum_tokens,
            } => {
                if bytes_per_token < minimum_bytes {
                    return Err(invalid_plan(
                        "token-scaled state demand is smaller than its resolved resource range",
                    ));
                }
                Ok(ValueResourceDemand::TokenScaled {
                    lifetime,
                    bytes_per_token,
                    maximum_tokens,
                })
            }
        }
    }

    fn plan_id_for_hash(hash: &PlanHash) -> Result<PlanId, VNextError> {
        PlanId::new(format!("plan/sha256/{}", hash.as_str()))
    }

    fn validate_internal(&self) -> Result<(), VNextError> {
        if self.payload.schema != EXECUTION_PLAN_SCHEMA {
            return Err(VNextError::UnsupportedPlanSchema {
                expected_major: EXECUTION_PLAN_SCHEMA.major,
                expected_minor: EXECUTION_PLAN_SCHEMA.minor,
                actual_major: self.payload.schema.major,
                actual_minor: self.payload.schema.minor,
            });
        }
        let computed = PlanHash::new(canonical_fingerprint(
            &PlanHashMaterial::from(&self.payload),
            "validate execution plan hash",
        )?)?;
        if computed != self.plan_hash {
            return Err(VNextError::PlanHashMismatch {
                expected: computed.to_string(),
                actual: self.plan_hash.to_string(),
            });
        }
        if self.payload.plan_id != Self::plan_id_for_hash(&computed)? {
            return Err(invalid_plan(
                "plan id is not derived from the semantic plan hash",
            ));
        }
        if self.payload.nodes.is_empty()
            || !is_canonical_sha256(&self.payload.prepared_family_fingerprint)
            || !is_canonical_sha256(&self.payload.program_fingerprint)
            || !is_canonical_sha256(&self.payload.capability_catalog_fingerprint)
            || !is_canonical_sha256(&self.payload.device_runtime_implementation_fingerprint)
            || !is_canonical_sha256(&self.payload.policy_fingerprint)
        {
            return Err(invalid_plan("plan provenance or node set is invalid"));
        }
        self.payload.memory.validate()?;
        let dynamic_capacity_bytes = self
            .payload
            .memory
            .usable_capacity_bytes
            .checked_sub(self.payload.memory.static_bytes)
            .ok_or_else(|| invalid_plan("static memory exceeds usable capacity"))?;
        let expected_pools = MemoryPlan::derive_dynamic_pools(
            &self.payload.memory.dynamic_descriptors,
            &self.payload.nodes,
            dynamic_capacity_bytes,
        )?;
        if self.payload.memory.dynamic_pools != expected_pools {
            return Err(invalid_plan(
                "memory pools or invocation reuse are not derived from plan dependencies",
            ));
        }
        let static_allocations = self
            .payload
            .memory
            .static_allocations
            .iter()
            .map(|allocation| (allocation.resource_id.clone(), allocation))
            .collect::<BTreeMap<_, _>>();
        let dynamic_descriptors = self
            .payload
            .memory
            .dynamic_descriptors
            .iter()
            .map(|descriptor| (descriptor.base_resource_id.clone(), descriptor))
            .collect::<BTreeMap<_, _>>();
        let mut seen_nodes = BTreeSet::new();
        let mut canonical_values = BTreeMap::new();
        for node in &self.payload.nodes {
            node.provider_resources.validate_shape()?;
            Self::validate_provider_selection_evidence(&node.selection)?;
            if !seen_nodes.insert(node.id.clone())
                || !is_canonical_sha256(&node.provider_implementation_fingerprint)
                || node
                    .dependencies
                    .iter()
                    .any(|dependency| dependency == &node.id || !seen_nodes.contains(dependency))
                || node.dependencies.windows(2).any(|pair| pair[0] >= pair[1])
                || node
                    .state_effects
                    .windows(2)
                    .any(|pair| pair[0].state_id >= pair[1].state_id)
                || node.resources.iter().collect::<BTreeSet<_>>().len() != node.resources.len()
                || node.resources.iter().any(|resource| {
                    !static_allocations.contains_key(resource)
                        && !dynamic_descriptors.contains_key(resource)
                })
                || node.provider_resources.provider_id != node.selection.selected_provider
            {
                return Err(invalid_plan(format!(
                    "node `{}` identity, dependency, or resource closure is invalid",
                    node.id
                )));
            }
            let expected_resources = node
                .values
                .iter()
                .flat_map(|binding| binding.storage().components())
                .map(|component| component.resource_id().clone())
                .chain(node.scratch_resource.iter().cloned())
                .chain(node.persistent_resource.iter().cloned())
                .collect::<BTreeSet<_>>()
                .into_iter()
                .collect::<Vec<_>>();
            if node.resources != expected_resources {
                return Err(invalid_plan(format!(
                    "node `{}` resource closure is not canonical",
                    node.id
                )));
            }
            for effect in &node.state_effects {
                if !matches!(
                    effect.lifetime,
                    AllocationLifetime::Request
                        | AllocationLifetime::Sequence
                        | AllocationLifetime::Step
                ) || effect.resource_ids.is_empty()
                    || effect
                        .resource_ids
                        .windows(2)
                        .any(|pair| pair[0] >= pair[1])
                {
                    return Err(invalid_plan(format!(
                        "node `{}` state effect has an invalid lifetime or resource closure",
                        node.id
                    )));
                }
                let matching = node
                    .values
                    .iter()
                    .filter(|binding| binding.value_id() == &effect.state_value_id)
                    .collect::<Vec<_>>();
                let expected_effect_resources = matching
                    .iter()
                    .flat_map(|binding| binding.storage().components())
                    .map(|component| component.resource_id().clone())
                    .collect::<BTreeSet<_>>()
                    .into_iter()
                    .collect::<Vec<_>>();
                let reads = matching.iter().any(|binding| {
                    matches!(
                        binding.access(),
                        TensorAccess::Read | TensorAccess::ReadWrite
                    )
                });
                let writes = matching.iter().any(|binding| {
                    matches!(
                        binding.access(),
                        TensorAccess::Write | TensorAccess::ReadWrite
                    )
                });
                let expected_access = match (reads, writes) {
                    (true, false) => Some(TensorAccess::Read),
                    (false, true) => Some(TensorAccess::Write),
                    (true, true) => Some(TensorAccess::ReadWrite),
                    (false, false) => None,
                };
                if effect.resource_ids != expected_effect_resources
                    || expected_access != Some(effect.access)
                    || effect.resource_ids.iter().any(|resource_id| {
                        dynamic_descriptors
                            .get(resource_id)
                            .is_none_or(|descriptor| descriptor.lifetime != effect.lifetime)
                    })
                {
                    return Err(invalid_plan(format!(
                        "node `{}` state effect is not derived from its typed bindings",
                        node.id
                    )));
                }
            }
            if node.scratch_resource.is_some() != node.provider_resources.scratch.is_some()
                || node.persistent_resource.is_some()
                    != node.provider_resources.persistent.is_some()
            {
                return Err(invalid_plan(format!(
                    "node `{}` workspace base identity presence differs from its provider estimate",
                    node.id
                )));
            }
            if let Some(resource_id) = &node.scratch_resource {
                let descriptor = dynamic_descriptors.get(resource_id).ok_or_else(|| {
                    invalid_plan(format!("node `{}` scratch descriptor is missing", node.id))
                })?;
                let workspace = node.provider_resources.scratch.as_ref().ok_or_else(|| {
                    invalid_plan(format!("node `{}` scratch estimate is missing", node.id))
                })?;
                if descriptor.demand != workspace.size_formula
                    || descriptor.alignment_bytes != workspace.alignment_bytes
                    || descriptor.usage != BufferUsage::Scratch
                    || descriptor.lifetime != AllocationLifetime::Invocation
                    || descriptor.theoretical_maximum_instances
                        != self.payload.memory.maximum_active_sequences
                    || descriptor.kind
                        != (AllocationKind::Scratch {
                            node_id: node.id.clone(),
                        })
                {
                    return Err(invalid_plan(format!(
                        "node `{}` scratch descriptor differs from its provider estimate",
                        node.id
                    )));
                }
            }
            if let Some(resource_id) = &node.persistent_resource {
                let workspace = node.provider_resources.persistent.as_ref().ok_or_else(|| {
                    invalid_plan(format!("node `{}` persistent estimate is missing", node.id))
                })?;
                match workspace.scope {
                    ProviderWorkspaceScope::Plan => {
                        let allocation = static_allocations.get(resource_id).ok_or_else(|| {
                            invalid_plan(format!(
                                "node `{}` plan-static persistent allocation is missing",
                                node.id
                            ))
                        })?;
                        if Some(allocation.per_instance_bytes) != workspace.fixed_bytes()
                            || allocation.alignment_bytes != workspace.alignment_bytes
                            || allocation.usage != BufferUsage::Persistent
                            || !workspace.storage.accepts(allocation.storage.profile())
                            || allocation.storage.logical_layout_fingerprint()
                                != workspace_storage_layout_fingerprint()?
                            || allocation.kind
                                != (AllocationKind::Persistent {
                                    node_id: node.id.clone(),
                                })
                        {
                            return Err(invalid_plan(format!(
                                "node `{}` plan-static persistent allocation differs from its provider estimate",
                                node.id
                            )));
                        }
                    }
                    scope @ (ProviderWorkspaceScope::Request
                    | ProviderWorkspaceScope::Sequence
                    | ProviderWorkspaceScope::Step) => {
                        let expected_lifetime = match scope {
                            ProviderWorkspaceScope::Request => AllocationLifetime::Request,
                            ProviderWorkspaceScope::Sequence => AllocationLifetime::Sequence,
                            ProviderWorkspaceScope::Step => AllocationLifetime::Step,
                            ProviderWorkspaceScope::Plan | ProviderWorkspaceScope::Invocation => {
                                unreachable!()
                            }
                        };
                        let descriptor = dynamic_descriptors.get(resource_id).ok_or_else(|| {
                            invalid_plan(format!(
                                "node `{}` dynamic persistent descriptor is missing",
                                node.id
                            ))
                        })?;
                        if descriptor.demand != workspace.size_formula
                            || descriptor.alignment_bytes != workspace.alignment_bytes
                            || descriptor.usage != BufferUsage::Persistent
                            || descriptor.lifetime != expected_lifetime
                            || descriptor.theoretical_maximum_instances
                                != self.payload.memory.maximum_active_sequences
                            || descriptor.kind
                                != (AllocationKind::Persistent {
                                    node_id: node.id.clone(),
                                })
                        {
                            return Err(invalid_plan(format!(
                                "node `{}` dynamic persistent descriptor differs from its provider estimate",
                                node.id
                            )));
                        }
                    }
                    ProviderWorkspaceScope::Invocation => {
                        return Err(invalid_plan(format!(
                            "node `{}` persistent workspace cannot be invocation scoped",
                            node.id
                        )));
                    }
                }
            }
            for binding in &node.values {
                Self::validate_cross_node_value(binding, &mut canonical_values)?;
            }
        }
        Self::validate_global_storage_aliasing(&canonical_values, &self.payload.nodes)?;
        Ok(())
    }

    pub fn payload(&self) -> &ExecutionPlanPayload {
        &self.payload
    }

    pub fn plan_hash(&self) -> &PlanHash {
        &self.plan_hash
    }

    pub(crate) fn operation_registry_authority(&self) -> &OperationRegistryAuthority {
        &self.operation_registry_authority
    }

    pub fn to_json(&self) -> Result<Vec<u8>, VNextError> {
        serde_json::to_vec(self).map_err(|error| VNextError::Serialization {
            context: "serialize execution plan",
            message: error.to_string(),
        })
    }

    pub fn decode_untrusted(bytes: &[u8]) -> Result<UnvalidatedExecutionPlan, VNextError> {
        if bytes.len() > MAX_EXECUTION_PLAN_WIRE_BYTES {
            return Err(VNextError::Serialization {
                context: "decode untrusted execution plan",
                message: format!(
                    "execution plan wire size {} exceeds limit {}",
                    bytes.len(),
                    MAX_EXECUTION_PLAN_WIRE_BYTES
                ),
            });
        }
        serde_json::from_slice::<UnvalidatedExecutionPlanWire>(bytes)
            .map(UnvalidatedExecutionPlan::from)
            .map_err(|error| VNextError::Serialization {
                context: "decode untrusted execution plan",
                message: error.to_string(),
            })
    }

    pub fn from_json_validated<P: RuntimePolicy>(
        bytes: &[u8],
        family: &PreparedModelFamily,
        capabilities: &CapabilityCatalog,
        policy: &P,
        node_resolutions: Vec<PlanNodeResolution>,
    ) -> Result<Self, VNextError> {
        Self::decode_untrusted(bytes)?.revalidate(family, capabilities, policy, node_resolutions)
    }

    pub fn validate_against<P: RuntimePolicy>(
        &self,
        family: &PreparedModelFamily,
        capabilities: &CapabilityCatalog,
        policy: &P,
        node_resolutions: &[PlanNodeResolution],
    ) -> Result<(), VNextError> {
        let rebuilt = ExecutionPlan::build(PlanBuildRequest::new(
            family,
            capabilities,
            policy,
            node_resolutions.to_vec(),
        )?)?;
        if rebuilt.operation_registry_authority != self.operation_registry_authority {
            return Err(invalid_plan(
                "execution plan belongs to a different operation runtime registry",
            ));
        }
        if &rebuilt != self {
            return Err(invalid_plan(
                "execution plan is not identical to its semantic rebuild",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct CanonicalValueBinding {
    tensor: ResolvedTensorSpec,
    usage: BufferUsage,
    storage: ResolvedValueStorage,
}

struct GlobalValueRange {
    value_id: ProgramValueId,
    offset_bytes: u64,
    end_bytes: u64,
}

#[derive(Default)]
struct StateDependencyTracker {
    last_writer: BTreeMap<StateId, NodeId>,
    readers_since_write: BTreeMap<StateId, BTreeSet<NodeId>>,
}

#[derive(Clone)]
struct JointProviderCandidate {
    resources: ProviderResourcePlan,
    allowed_profiles: BTreeMap<ResourceId, BTreeSet<DynamicStorageProfile>>,
    is_preferred: bool,
}

struct JointProviderStorageSelection {
    node_resources: BTreeMap<NodeId, ProviderResourcePlan>,
    resource_profiles: BTreeMap<ResourceId, DynamicStorageProfile>,
    storage_rejections: BTreeMap<NodeId, RejectedProvider>,
}

#[derive(Clone, Default)]
struct JointPartialSelection {
    chosen: Vec<ProviderResourcePlan>,
    preferred: Vec<bool>,
}

struct JointComponentSolution {
    chosen: Vec<ProviderResourcePlan>,
    resource_profiles: BTreeMap<ResourceId, DynamicStorageProfile>,
}

struct JointSelectionObjective {
    preferred: Vec<bool>,
    /// Counts for the least-preferred profile through profile rank 1. Rank 0
    /// has zero penalty, so candidate-specific extra rank-0 resources cannot
    /// bias selection merely by increasing the resource count.
    profile_penalty: Vec<usize>,
    provider_ids: Vec<ProviderId>,
}

impl JointSelectionObjective {
    fn new(
        partial: &JointPartialSelection,
        profile_ranks: impl Iterator<Item = usize>,
        profile_count: usize,
    ) -> Result<Self, VNextError> {
        let mut profile_penalty = vec![0_usize; profile_count.saturating_sub(1)];
        for rank in profile_ranks {
            if rank >= profile_count {
                return Err(invalid_plan(
                    "joint storage objective contains an unknown profile rank",
                ));
            }
            if rank != 0 {
                let penalty_index = profile_count - 1 - rank;
                profile_penalty[penalty_index] = profile_penalty[penalty_index]
                    .checked_add(1)
                    .ok_or_else(|| invalid_plan("joint storage profile penalty overflows usize"))?;
            }
        }
        Ok(Self {
            preferred: partial.preferred.clone(),
            profile_penalty,
            provider_ids: partial
                .chosen
                .iter()
                .map(|resources| resources.provider_id().clone())
                .collect(),
        })
    }

    fn precedes(&self, other: &Self) -> bool {
        match compare_preferred(&self.preferred, &other.preferred) {
            std::cmp::Ordering::Less => return true,
            std::cmp::Ordering::Greater => return false,
            std::cmp::Ordering::Equal => {}
        }
        match self.profile_penalty.cmp(&other.profile_penalty) {
            std::cmp::Ordering::Less => true,
            std::cmp::Ordering::Greater => false,
            std::cmp::Ordering::Equal => self.provider_ids < other.provider_ids,
        }
    }
}

fn compare_preferred(left: &[bool], right: &[bool]) -> std::cmp::Ordering {
    for (left, right) in left.iter().zip(right) {
        match (*right as u8).cmp(&(*left as u8)) {
            std::cmp::Ordering::Equal => {}
            ordering => return ordering,
        }
    }
    left.len().cmp(&right.len())
}

fn joint_partial_precedes(left: &JointPartialSelection, right: &JointPartialSelection) -> bool {
    match compare_preferred(&left.preferred, &right.preferred) {
        std::cmp::Ordering::Less => true,
        std::cmp::Ordering::Greater => false,
        std::cmp::Ordering::Equal => left
            .chosen
            .iter()
            .map(ProviderResourcePlan::provider_id)
            .cmp(right.chosen.iter().map(ProviderResourcePlan::provider_id))
            .is_lt(),
    }
}

fn joint_candidate_components(candidate_sets: &[Vec<JointProviderCandidate>]) -> Vec<Vec<usize>> {
    let mut parents = (0..candidate_sets.len()).collect::<Vec<_>>();
    let mut ranks = vec![0_u8; candidate_sets.len()];
    let mut resource_owner = BTreeMap::<ResourceId, usize>::new();
    for (node_index, candidates) in candidate_sets.iter().enumerate() {
        let resource_ids = candidates
            .iter()
            .flat_map(|candidate| candidate.allowed_profiles.keys().cloned())
            .collect::<BTreeSet<_>>();
        for resource_id in resource_ids {
            if let Some(owner) = resource_owner.insert(resource_id, node_index) {
                joint_union(&mut parents, &mut ranks, owner, node_index);
            }
        }
    }
    let mut components = BTreeMap::<usize, Vec<usize>>::new();
    for node_index in 0..candidate_sets.len() {
        let root = joint_find(&mut parents, node_index);
        components.entry(root).or_default().push(node_index);
    }
    let mut components = components.into_values().collect::<Vec<_>>();
    components.sort_by_key(|component| component[0]);
    components
}

fn storage_incompatible_resource_ids(
    candidate: &JointProviderCandidate,
    selected_profiles: &BTreeMap<ResourceId, DynamicStorageProfile>,
) -> Vec<ResourceId> {
    candidate
        .allowed_profiles
        .iter()
        .filter(|(resource_id, accepted)| {
            selected_profiles
                .get(*resource_id)
                .is_some_and(|profile| !accepted.contains(profile))
        })
        .map(|(resource_id, _)| resource_id.clone())
        .collect()
}

fn joint_find(parents: &mut [usize], value: usize) -> usize {
    let mut root = value;
    while parents[root] != root {
        root = parents[root];
    }
    let mut current = value;
    while parents[current] != current {
        let next = parents[current];
        parents[current] = root;
        current = next;
    }
    root
}

fn joint_union(parents: &mut [usize], ranks: &mut [u8], left: usize, right: usize) {
    let left_root = joint_find(parents, left);
    let right_root = joint_find(parents, right);
    if left_root == right_root {
        return;
    }
    match ranks[left_root].cmp(&ranks[right_root]) {
        std::cmp::Ordering::Less => parents[left_root] = right_root,
        std::cmp::Ordering::Greater => parents[right_root] = left_root,
        std::cmp::Ordering::Equal => {
            parents[right_root] = left_root;
            ranks[left_root] = ranks[left_root].saturating_add(1);
        }
    }
}

#[derive(Default)]
struct PoolAggregateEvidence {
    minimum_request_bytes: u64,
    minimum_sequence_bytes: u64,
    minimum_step_bytes: u64,
    minimum_invocation_peak_bytes: u64,
    theoretical_ceiling_bytes: u128,
}

impl PoolAggregateEvidence {
    fn add(&mut self, pool: &DynamicBackingPoolSpec) -> Result<(), VNextError> {
        self.minimum_request_bytes = self
            .minimum_request_bytes
            .checked_add(pool.minimum_request_bytes)
            .ok_or_else(|| invalid_plan("aggregate pool request minimum overflows u64"))?;
        self.minimum_sequence_bytes = self
            .minimum_sequence_bytes
            .checked_add(pool.minimum_sequence_bytes)
            .ok_or_else(|| invalid_plan("aggregate pool sequence minimum overflows u64"))?;
        self.minimum_step_bytes = self
            .minimum_step_bytes
            .checked_add(pool.minimum_step_bytes)
            .ok_or_else(|| invalid_plan("aggregate pool step minimum overflows u64"))?;
        self.minimum_invocation_peak_bytes = self
            .minimum_invocation_peak_bytes
            .checked_add(pool.minimum_invocation_peak_bytes)
            .ok_or_else(|| invalid_plan("aggregate pool invocation minimum overflows u64"))?;
        self.theoretical_ceiling_bytes = self
            .theoretical_ceiling_bytes
            .checked_add(pool.theoretical_ceiling_bytes.get())
            .ok_or_else(|| invalid_plan("aggregate pool theoretical ceiling overflows u128"))?;
        Ok(())
    }
}

fn minimum_for_lifetime(
    descriptors: &[&DynamicResourceDescriptor],
    lifetime: AllocationLifetime,
    overflow_context: &'static str,
) -> Result<u64, VNextError> {
    descriptors
        .iter()
        .filter(|descriptor| descriptor.lifetime == lifetime)
        .try_fold(0_u64, |total, descriptor| {
            total
                .checked_add(descriptor.minimum_request_bytes()?)
                .ok_or_else(|| invalid_plan(format!("{overflow_context} overflows u64")))
        })
}

fn node_completion_precedes(
    nodes_by_id: &BTreeMap<NodeId, &PlanNode>,
    predecessor: &NodeId,
    successor: &NodeId,
) -> Result<bool, VNextError> {
    if predecessor == successor {
        return Ok(false);
    }
    let successor = nodes_by_id
        .get(successor)
        .ok_or_else(|| invalid_plan("completion-order successor node is missing"))?;
    let mut pending = successor.dependencies.clone();
    let mut visited = BTreeSet::new();
    while let Some(node_id) = pending.pop() {
        if &node_id == predecessor {
            return Ok(true);
        }
        if !visited.insert(node_id.clone()) {
            continue;
        }
        let node = nodes_by_id
            .get(&node_id)
            .ok_or_else(|| invalid_plan("completion-order dependency node is missing"))?;
        pending.extend(node.dependencies.iter().cloned());
    }
    Ok(false)
}

fn validate_pool_liveness_rows(
    mode: InvocationLivenessMode,
    rows: &[InvocationResourceLiveness],
    invocation_ids: &BTreeSet<ResourceId>,
    descriptors: &BTreeMap<ResourceId, &DynamicResourceDescriptor>,
) -> Result<u64, VNextError> {
    if invocation_ids.is_empty() {
        if mode != InvocationLivenessMode::NoInvocationResources || !rows.is_empty() {
            return Err(invalid_plan(
                "pool without invocation resources has liveness evidence",
            ));
        }
        return Ok(0);
    }
    if mode == InvocationLivenessMode::NoInvocationResources || rows.is_empty() {
        return Err(invalid_plan(
            "invocation pool is missing typed liveness evidence",
        ));
    }
    let mut covered = BTreeSet::new();
    let mut maximum_row = 0_u64;
    let mut concurrent_sum = 0_u64;
    for row in rows {
        if row.resource_ids.is_empty() || row.resource_ids.windows(2).any(|pair| pair[0] >= pair[1])
        {
            return Err(invalid_plan(
                "pool invocation liveness row is empty or non-canonical",
            ));
        }
        let row_bytes = row
            .resource_ids
            .iter()
            .try_fold(0_u64, |total, resource_id| {
                if !invocation_ids.contains(resource_id) {
                    return Err(invalid_plan(
                        "pool liveness references a non-member invocation resource",
                    ));
                }
                covered.insert(resource_id.clone());
                total
                    .checked_add(
                        descriptors
                            .get(resource_id)
                            .ok_or_else(|| invalid_plan("pool liveness descriptor is missing"))?
                            .minimum_request_bytes()?,
                    )
                    .ok_or_else(|| invalid_plan("pool invocation row bytes overflow u64"))
            })?;
        maximum_row = maximum_row.max(row_bytes);
        concurrent_sum = concurrent_sum
            .checked_add(row_bytes)
            .ok_or_else(|| invalid_plan("pool concurrent invocation bytes overflow u64"))?;
    }
    if covered != *invocation_ids {
        return Err(invalid_plan(
            "pool liveness does not cover every invocation member",
        ));
    }
    Ok(match mode {
        InvocationLivenessMode::NoInvocationResources => unreachable!(),
        InvocationLivenessMode::TotalOrderReuse => maximum_row,
        InvocationLivenessMode::ConservativeConcurrent => concurrent_sum,
    })
}

struct ValueAllocationAccumulator {
    end_bytes: u64,
    alignment_bytes: u64,
    usage: BufferUsage,
    element_type: ElementType,
    demand: ValueResourceDemand,
    logical_layout_fingerprints: BTreeSet<String>,
    merge_result: Option<()>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ValueResourceDemand {
    PlanStatic,
    Fixed {
        lifetime: AllocationLifetime,
    },
    TokenScaled {
        lifetime: AllocationLifetime,
        bytes_per_token: u64,
        maximum_tokens: u64,
    },
}

impl ValueResourceDemand {
    fn lifetime(self) -> Option<AllocationLifetime> {
        match self {
            Self::PlanStatic => None,
            Self::Fixed { lifetime } | Self::TokenScaled { lifetime, .. } => Some(lifetime),
        }
    }

    fn dynamic_demand(self, fixed_bytes: u64) -> Result<DynamicResourceDemand, VNextError> {
        match self {
            Self::PlanStatic => Err(invalid_plan(
                "plan-static value cannot produce a dynamic demand",
            )),
            Self::Fixed { .. } => DynamicResourceDemand::fixed(fixed_bytes),
            Self::TokenScaled {
                bytes_per_token,
                maximum_tokens,
                ..
            } => DynamicResourceDemand::tokens(bytes_per_token, maximum_tokens),
        }
    }
}

impl ValueAllocationAccumulator {
    fn merge(
        &mut self,
        end_bytes: u64,
        alignment_bytes: u64,
        usage: BufferUsage,
        element_type: ElementType,
        demand: ValueResourceDemand,
        logical_layout_fingerprint: String,
    ) -> Option<()> {
        if self.usage != usage || self.element_type != element_type || self.demand != demand {
            return None;
        }
        self.end_bytes = self.end_bytes.max(end_bytes);
        self.alignment_bytes = self.alignment_bytes.max(alignment_bytes);
        self.logical_layout_fingerprints
            .insert(logical_layout_fingerprint);
        Some(())
    }
}

fn align_up(value: u64, alignment: u64) -> Result<u64, VNextError> {
    if alignment == 0 || !alignment.is_power_of_two() {
        return Err(invalid_plan(
            "allocation alignment is not a non-zero power of two",
        ));
    }
    value
        .checked_add(alignment - 1)
        .map(|rounded| rounded & !(alignment - 1))
        .ok_or_else(|| invalid_plan("aligned allocation size overflows u64"))
}

fn quantize_storage_bytes(
    logical_bytes: u64,
    alignment_bytes: u64,
    profile: DynamicStorageProfile,
) -> Result<u64, VNextError> {
    let aligned = align_up(logical_bytes, alignment_bytes)?;
    match profile.allocator() {
        super::DynamicStorageAllocator::LinearArena => Ok(aligned),
        super::DynamicStorageAllocator::FixedBlockArena { block_bytes } => {
            align_up(aligned, block_bytes)
        }
    }
}

/// Typed policy selected before planning. Memory capacity is part of the
/// public policy contract so a plan cannot depend on an undocumented env var.
pub trait RuntimePolicy:
    Clone + Send + Sync + Serialize + serde::de::DeserializeOwned + std::fmt::Debug + 'static
{
    fn version(&self) -> ContractVersion;

    /// Policy memory ceiling before the explicit reserve is subtracted. It may
    /// be lower than, but never exceed, the raw device capacity.
    fn memory_capacity_bytes(&self) -> u64;

    fn memory_reserve_bytes(&self) -> u64;

    /// A non-zero protocol ceiling only. Planning must not reserve, claim,
    /// iterate, or materialize resources up to this value.
    fn maximum_active_sequences(&self) -> u32;

    /// Ordered, non-empty allowlist used by planning after intersecting every
    /// selected provider requirement with the concrete runtime offers.
    fn dynamic_storage_profile_order(&self) -> &[DynamicStorageProfile];

    fn validate(&self) -> Result<(), VNextError>;
}

pub fn canonical_runtime_policy_fingerprint<P: RuntimePolicy>(
    policy: &P,
) -> Result<String, VNextError> {
    policy.validate()?;
    canonical_fingerprint(policy, "fingerprint validated runtime policy")
}

pub struct PlanBuildRequest<'a, P: RuntimePolicy> {
    family: &'a PreparedModelFamily,
    capabilities: &'a CapabilityCatalog,
    policy: &'a P,
    node_resolutions: Vec<PlanNodeResolution>,
}

impl<'a, P: RuntimePolicy> PlanBuildRequest<'a, P> {
    pub fn new(
        family: &'a PreparedModelFamily,
        capabilities: &'a CapabilityCatalog,
        policy: &'a P,
        node_resolutions: Vec<PlanNodeResolution>,
    ) -> Result<Self, VNextError> {
        if node_resolutions.is_empty() {
            return Err(invalid_plan("plan build request has no node resolutions"));
        }
        policy.validate()?;
        Ok(Self {
            family,
            capabilities,
            policy,
            node_resolutions,
        })
    }

    pub fn family(&self) -> &PreparedModelFamily {
        self.family
    }

    pub fn capabilities(&self) -> &CapabilityCatalog {
        self.capabilities
    }

    pub fn policy(&self) -> &P {
        self.policy
    }
}

/// Pure planner boundary. Execution consumes the immutable plan and performs
/// no capability/backend selection in the token loop.
pub trait ExecutionPlanner: Send + Sync {
    type Policy: RuntimePolicy;

    fn build_plan(
        &self,
        request: PlanBuildRequest<'_, Self::Policy>,
    ) -> Result<ExecutionPlan, VNextError>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vnext::{DynamicStorageAllocator, DynamicStorageView};

    fn linear_profile() -> DynamicStorageProfile {
        DynamicStorageProfile::new(
            DynamicStorageAllocator::LinearArena,
            DynamicStorageView::Contiguous,
        )
        .expect("valid linear profile")
    }

    fn paged_profile() -> DynamicStorageProfile {
        fixed_block_profile(4096)
    }

    fn fixed_block_profile(block_bytes: u64) -> DynamicStorageProfile {
        DynamicStorageProfile::new(
            DynamicStorageAllocator::FixedBlockArena { block_bytes },
            DynamicStorageView::PagedRegions { block_bytes },
        )
        .expect("valid paged profile")
    }

    fn provider_resources(provider: &str) -> ProviderResourcePlan {
        ProviderResourcePlan {
            provider_id: ProviderId::new(provider).expect("valid provider id"),
            estimator_id: "test-estimator".to_owned(),
            estimator_version: ContractVersion::new(1, 0),
            estimator_implementation_fingerprint: "1".repeat(64),
            estimator_input_fingerprint: "2".repeat(64),
            estimate_fingerprint: "3".repeat(64),
            value_alignment_bytes: 16,
            scratch: None,
            persistent: None,
        }
    }

    fn joint_candidate(
        provider: &str,
        resource: &ResourceId,
        profiles: &[DynamicStorageProfile],
    ) -> JointProviderCandidate {
        JointProviderCandidate {
            resources: provider_resources(provider),
            allowed_profiles: BTreeMap::from([(
                resource.clone(),
                profiles.iter().copied().collect(),
            )]),
            is_preferred: provider == "provider/preferred",
        }
    }

    #[test]
    fn joint_storage_solver_falls_back_when_preferred_breaks_shared_intersection() {
        let resource = ResourceId::new("resource/shared-state").expect("valid resource id");
        let linear = linear_profile();
        let paged = paged_profile();
        let candidate_sets = vec![
            vec![
                joint_candidate("provider/preferred", &resource, &[linear]),
                joint_candidate("provider/fallback", &resource, &[paged]),
            ],
            vec![joint_candidate("provider/consumer", &resource, &[paged])],
        ];

        let (chosen, profiles) =
            ExecutionPlan::solve_joint_provider_candidates(&candidate_sets, &[linear, paged])
                .expect("fallback should satisfy the shared resource");

        assert_eq!(chosen[0].provider_id().as_str(), "provider/fallback");
        assert_eq!(chosen[1].provider_id().as_str(), "provider/consumer");
        assert_eq!(profiles.get(&resource), Some(&paged));
    }

    #[test]
    fn joint_storage_solver_fails_closed_without_shared_intersection() {
        let resource = ResourceId::new("resource/shared-state").expect("valid resource id");
        let linear = linear_profile();
        let paged = paged_profile();
        let candidate_sets = vec![
            vec![joint_candidate("provider/linear", &resource, &[linear])],
            vec![joint_candidate("provider/paged", &resource, &[paged])],
        ];

        let error =
            ExecutionPlan::solve_joint_provider_candidates(&candidate_sets, &[linear, paged])
                .expect_err("incompatible shared storage must be rejected");

        assert!(error
            .to_string()
            .contains("no joint provider/storage assignment"));
    }

    #[test]
    fn joint_storage_profile_order_precedes_provider_id_tie_break() {
        let resource = ResourceId::new("resource/state").expect("valid resource id");
        let linear = linear_profile();
        let paged = paged_profile();
        let candidate_sets = vec![vec![
            joint_candidate("provider/a", &resource, &[paged]),
            joint_candidate("provider/z", &resource, &[linear]),
        ]];

        let (chosen, profiles) =
            ExecutionPlan::solve_joint_provider_candidates(&candidate_sets, &[linear, paged])
                .expect("one candidate satisfies each profile");

        assert_eq!(chosen[0].provider_id().as_str(), "provider/z");
        assert_eq!(profiles.get(&resource), Some(&linear));
    }

    #[test]
    fn joint_storage_solver_decomposes_independent_resource_components() {
        let first = ResourceId::new("resource/first").expect("valid resource id");
        let second = ResourceId::new("resource/second").expect("valid resource id");
        let linear = linear_profile();
        let paged = paged_profile();
        let candidate_sets = vec![
            vec![
                joint_candidate("provider/a", &first, &[paged]),
                joint_candidate("provider/b", &first, &[linear]),
            ],
            vec![
                joint_candidate("provider/c", &second, &[paged]),
                joint_candidate("provider/d", &second, &[linear]),
            ],
        ];

        let components = joint_candidate_components(&candidate_sets);
        assert_eq!(components, vec![vec![0], vec![1]]);
        let (chosen, profiles) =
            ExecutionPlan::solve_joint_provider_candidates(&candidate_sets, &[linear, paged])
                .expect("independent components should solve independently");
        assert_eq!(chosen[0].provider_id().as_str(), "provider/b");
        assert_eq!(chosen[1].provider_id().as_str(), "provider/d");
        assert_eq!(profiles.get(&first), Some(&linear));
        assert_eq!(profiles.get(&second), Some(&linear));
    }

    #[test]
    fn tensor_layout_fingerprint_excludes_shape_but_preserves_interpretation() {
        let first = ResolvedTensorLayout::Blocked {
            block: vec![16, 16],
            axis_order: vec![1, 0],
            padding: BlockedTensorPadding::ZeroFill {
                physical_dimensions: vec![32, 48],
            },
        };
        let second = ResolvedTensorLayout::Blocked {
            block: vec![16, 16],
            axis_order: vec![1, 0],
            padding: BlockedTensorPadding::ZeroFill {
                physical_dimensions: vec![64, 80],
            },
        };
        let changed_layout = ResolvedTensorLayout::Blocked {
            block: vec![32, 16],
            axis_order: vec![1, 0],
            padding: BlockedTensorPadding::ZeroFill {
                physical_dimensions: vec![64, 80],
            },
        };

        assert_eq!(
            tensor_storage_layout_fingerprint(&first).expect("fingerprint"),
            tensor_storage_layout_fingerprint(&second).expect("fingerprint")
        );
        assert_ne!(
            tensor_storage_layout_fingerprint(&first).expect("fingerprint"),
            tensor_storage_layout_fingerprint(&changed_layout).expect("fingerprint")
        );
    }

    fn dynamic_value_descriptor(
        resource: &str,
        storage: DynamicStorageContract,
    ) -> DynamicResourceDescriptor {
        DynamicResourceDescriptor::new(
            ResourceId::new(resource).expect("valid resource id"),
            DynamicResourceDemand::fixed(64).expect("valid demand"),
            16,
            BufferUsage::State,
            ElementType::F16,
            AllocationLifetime::Sequence,
            AllocationKind::Value,
            storage,
            1024,
        )
        .expect("valid descriptor")
    }

    fn invocation_descriptor(
        resource: &str,
        node: &str,
        bytes: u64,
        storage: DynamicStorageContract,
    ) -> DynamicResourceDescriptor {
        DynamicResourceDescriptor::new(
            ResourceId::new(resource).expect("valid resource id"),
            DynamicResourceDemand::fixed(bytes).expect("valid demand"),
            16,
            BufferUsage::Scratch,
            ElementType::U8,
            AllocationLifetime::Invocation,
            AllocationKind::Scratch {
                node_id: NodeId::new(node).expect("valid node id"),
            },
            storage,
            1024,
        )
        .expect("valid invocation descriptor")
    }

    fn plan_node(id: &str, dependencies: &[&str], resources: &[&str]) -> PlanNode {
        let provider_resources = provider_resources("provider/test");
        let selected_provider = provider_resources.provider_id().clone();
        let mut resources = resources
            .iter()
            .map(|resource| ResourceId::new(*resource).expect("valid resource id"))
            .collect::<Vec<_>>();
        resources.sort();
        PlanNode {
            id: NodeId::new(id).expect("valid node id"),
            dependencies: dependencies
                .iter()
                .map(|dependency| NodeId::new(*dependency).expect("valid dependency id"))
                .collect(),
            operation_id: OperationId::new(format!("operation/{id}")).expect("operation id"),
            operation_version: ContractVersion::new(1, 0),
            operation_fingerprint: "4".repeat(64),
            provider_implementation_fingerprint: "5".repeat(64),
            required_capabilities: BTreeSet::new(),
            attributes: BTreeMap::new(),
            selection: ProviderSelection {
                requested_provider: None,
                selected_provider,
                selection_reason: ProviderSelectionReason::DeterministicCompatible,
                rejected_providers: Vec::new(),
            },
            provider_resources,
            values: Vec::new(),
            exact_aliases: Vec::new(),
            state_effects: Vec::new(),
            scratch_resource: None,
            persistent_resource: None,
            resources,
        }
    }

    #[test]
    fn fixed_block_storage_quantizes_logical_demand_and_maxima() {
        let layout = canonical_fingerprint(&"contiguous_v1", "test layout").expect("fingerprint");
        let storage =
            DynamicStorageContract::new(fixed_block_profile(4096), layout).expect("storage");
        let descriptor = DynamicResourceDescriptor::new(
            ResourceId::new("resource/quantized").expect("resource id"),
            DynamicResourceDemand::tokens(1, 4097).expect("demand"),
            16,
            BufferUsage::State,
            ElementType::U8,
            AllocationLifetime::Sequence,
            AllocationKind::Value,
            storage,
            1024,
        )
        .expect("descriptor");

        assert_eq!(
            descriptor
                .evaluate_logical_request_bytes_for_shape(
                    DynamicResourceShape::new(1, 1, 1).expect("shape"),
                )
                .expect("logical"),
            1
        );
        assert_eq!(descriptor.physical_allocation_quantum_bytes(), 4096);
        assert_eq!(descriptor.minimum_request_bytes().expect("minimum"), 4096);
        assert_eq!(
            descriptor
                .theoretical_maximum_request_bytes()
                .expect("maximum"),
            8192
        );
    }

    #[test]
    fn fixed_block_storage_uses_larger_allocator_geometry_and_rejects_overflow() {
        let layout = canonical_fingerprint(&"contiguous_v1", "test layout").expect("fingerprint");
        let storage = DynamicStorageContract::new(fixed_block_profile(8192), layout.clone())
            .expect("storage");
        let descriptor = dynamic_value_descriptor("resource/8k-state", storage.clone());
        assert_eq!(descriptor.physical_allocation_quantum_bytes(), 8192);
        assert_eq!(descriptor.minimum_request_bytes().expect("minimum"), 8192);

        let error = DynamicResourceDescriptor::new(
            ResourceId::new("resource/overflow").expect("resource id"),
            DynamicResourceDemand::fixed(u64::MAX).expect("logical demand is representable"),
            16,
            BufferUsage::State,
            ElementType::U8,
            AllocationLifetime::Sequence,
            AllocationKind::Value,
            storage,
            1024,
        )
        .expect_err("physical quantum rounding must reject overflow");
        assert!(error.to_string().contains("overflows u64"));
    }

    #[test]
    fn per_pool_total_order_uses_maximum_invocation_row() {
        let layout = canonical_fingerprint(&"opaque_v1", "test layout").expect("fingerprint");
        let storage = DynamicStorageContract::new(linear_profile(), layout).expect("storage");
        let first = invocation_descriptor("resource/invocation-a", "node/a", 64, storage.clone());
        let second = invocation_descriptor("resource/invocation-b", "node/b", 128, storage);
        let nodes = vec![
            plan_node("node/a", &[], &["resource/invocation-a"]),
            plan_node("node/middle", &["node/a"], &[]),
            plan_node("node/b", &["node/middle"], &["resource/invocation-b"]),
        ];

        let pools = MemoryPlan::derive_dynamic_pools(&[first, second], &nodes, 1 << 20)
            .expect("derive pools");
        assert_eq!(pools.len(), 1);
        assert_eq!(
            pools[0].invocation_liveness_mode(),
            InvocationLivenessMode::TotalOrderReuse
        );
        assert_eq!(pools[0].minimum_invocation_peak_bytes(), 128);
        assert_eq!(pools[0].provisioning().minimum_resident_bytes(), 128);
    }

    #[test]
    fn per_pool_unordered_invocations_use_conservative_sum() {
        let layout = canonical_fingerprint(&"opaque_v1", "test layout").expect("fingerprint");
        let storage = DynamicStorageContract::new(linear_profile(), layout).expect("storage");
        let first = invocation_descriptor("resource/invocation-a", "node/a", 64, storage.clone());
        let second = invocation_descriptor("resource/invocation-b", "node/b", 128, storage);
        let nodes = vec![
            plan_node("node/a", &[], &["resource/invocation-a"]),
            plan_node("node/b", &[], &["resource/invocation-b"]),
        ];

        let pools = MemoryPlan::derive_dynamic_pools(&[first, second], &nodes, 1 << 20)
            .expect("derive pools");
        assert_eq!(
            pools[0].invocation_liveness_mode(),
            InvocationLivenessMode::ConservativeConcurrent
        );
        assert_eq!(pools[0].minimum_invocation_peak_bytes(), 192);
    }

    #[test]
    fn dynamic_pool_derivation_is_order_independent_and_compatibility_hashed() {
        let layout = canonical_fingerprint(&"contiguous_v1", "test layout").expect("fingerprint");
        let storage = DynamicStorageContract::new(linear_profile(), layout).expect("storage");
        let first = dynamic_value_descriptor("resource/state-a", storage.clone());
        let second = dynamic_value_descriptor("resource/state-b", storage.clone());

        let forward =
            MemoryPlan::derive_dynamic_pools(&[first.clone(), second.clone()], &[], 1 << 20)
                .expect("derive pools");
        let reverse =
            MemoryPlan::derive_dynamic_pools(&[second, first], &[], 1 << 20).expect("derive pools");

        assert_eq!(forward, reverse);
        assert_eq!(forward.len(), 1);

        let changed_layout = DynamicStorageContract::new(
            linear_profile(),
            canonical_fingerprint(&"strided_v1", "test layout").expect("fingerprint"),
        )
        .expect("storage");
        let changed = dynamic_value_descriptor("resource/state-c", changed_layout);
        assert_ne!(changed.pool_id(), forward[0].pool_id());
    }

    #[test]
    fn dynamic_pool_identity_covers_every_physical_compatibility_dimension() {
        let layout = canonical_fingerprint(&"contiguous_v1", "test layout").expect("fingerprint");
        let changed_layout =
            canonical_fingerprint(&"strided_v1", "test layout").expect("fingerprint");
        let linear =
            DynamicStorageContract::new(linear_profile(), layout.clone()).expect("storage");
        let paged = DynamicStorageContract::new(paged_profile(), layout).expect("storage");
        let changed_layout =
            DynamicStorageContract::new(linear_profile(), changed_layout).expect("storage");

        let ids = [
            PoolCompatibilityKey::new(&linear, BufferUsage::State, ElementType::F16, 16),
            PoolCompatibilityKey::new(&paged, BufferUsage::State, ElementType::F16, 16),
            PoolCompatibilityKey::new(&linear, BufferUsage::Scratch, ElementType::F16, 16),
            PoolCompatibilityKey::new(&linear, BufferUsage::State, ElementType::U8, 16),
            PoolCompatibilityKey::new(&changed_layout, BufferUsage::State, ElementType::F16, 16),
            PoolCompatibilityKey::new(&linear, BufferUsage::State, ElementType::F16, 32),
        ]
        .into_iter()
        .map(|key| {
            DynamicBackingPoolId::from_compatibility(&key.expect("compatibility key"))
                .expect("pool id")
        })
        .collect::<BTreeSet<_>>();

        assert_eq!(ids.len(), 6);
    }

    #[test]
    fn memory_plan_wire_rejects_missing_core_derived_pool() {
        let layout = canonical_fingerprint(&"contiguous_v1", "test layout").expect("fingerprint");
        let storage = DynamicStorageContract::new(linear_profile(), layout).expect("storage");
        let descriptor = dynamic_value_descriptor("resource/state-a", storage);
        let plan =
            MemoryPlan::from_core(1 << 20, 1 << 20, 4096, 1024, vec![], vec![descriptor], &[])
                .expect("valid memory plan");
        let encoded = serde_json::to_vec(&plan).expect("serialize memory plan");
        let decoded: MemoryPlan = serde_json::from_slice(&encoded).expect("round trip memory plan");
        assert_eq!(decoded, plan);

        let mut tampered = serde_json::to_value(&plan).expect("serialize memory plan");
        tampered["dynamic_pools"] = serde_json::json!([]);
        let error = serde_json::from_value::<MemoryPlan>(tampered)
            .expect_err("missing derived pool must be rejected");
        assert!(error
            .to_string()
            .contains("dynamic backing pool count is not derived"));
    }

    #[test]
    fn plan_scope_allocation_round_trips_selected_storage_and_physical_quantum() {
        let storage = DynamicStorageContract::new(
            fixed_block_profile(4096),
            workspace_storage_layout_fingerprint().expect("workspace layout"),
        )
        .expect("storage");
        let allocation = ResourceAllocation::new(
            ResourceId::new("resource/plan-workspace").expect("resource id"),
            1,
            16,
            BufferUsage::Persistent,
            ElementType::U8,
            AllocationKind::Persistent {
                node_id: NodeId::new("node/plan-workspace").expect("node id"),
            },
            storage.clone(),
        )
        .expect("allocation");
        assert_eq!(allocation.per_instance_bytes(), 1);
        assert_eq!(allocation.instance_stride_bytes(), 4096);
        assert_eq!(allocation.storage(), &storage);

        let plan =
            MemoryPlan::from_core(1 << 20, 1 << 20, 4096, 1024, vec![allocation], vec![], &[])
                .expect("memory plan");
        let encoded = serde_json::to_vec(&plan).expect("serialize");
        let decoded: MemoryPlan = serde_json::from_slice(&encoded).expect("round trip");
        assert_eq!(decoded, plan);
        assert_eq!(
            decoded.static_allocations()[0].storage().profile(),
            fixed_block_profile(4096)
        );

        let mut tampered = serde_json::to_value(&plan).expect("serialize");
        tampered["static_allocations"][0]["storage"]["logical_layout_fingerprint"] =
            serde_json::json!("not-a-fingerprint");
        assert!(serde_json::from_value::<MemoryPlan>(tampered).is_err());
    }

    #[test]
    fn storage_fallback_evidence_is_nonempty_canonical_and_matches_conflicts() {
        let first = ResourceId::new("resource/a").expect("resource id");
        let second = ResourceId::new("resource/b").expect("resource id");
        let linear = linear_profile();
        let paged = paged_profile();
        let candidate = JointProviderCandidate {
            resources: provider_resources("provider/preferred"),
            allowed_profiles: BTreeMap::from([
                (first.clone(), BTreeSet::from([linear])),
                (second.clone(), BTreeSet::from([paged])),
            ]),
            is_preferred: true,
        };
        let selected_profiles = BTreeMap::from([(first, paged), (second.clone(), linear)]);
        let resource_ids = storage_incompatible_resource_ids(&candidate, &selected_profiles);
        assert_eq!(
            resource_ids,
            vec![ResourceId::new("resource/a").expect("resource id"), second]
        );

        let selection = ProviderSelection {
            requested_provider: Some(ProviderId::new("provider/preferred").expect("provider id")),
            selected_provider: ProviderId::new("provider/fallback").expect("provider id"),
            selection_reason: ProviderSelectionReason::FallbackFromPreferred,
            rejected_providers: vec![RejectedProvider {
                provider_id: ProviderId::new("provider/preferred").expect("provider id"),
                reasons: PlanProviderRejectReason::StorageIncompatible {
                    resource_ids: resource_ids.clone(),
                },
            }],
        };
        ExecutionPlan::validate_provider_selection_evidence(&selection)
            .expect("canonical fallback evidence");
        let encoded = serde_json::to_vec(&selection).expect("serialize selection");
        let decoded: ProviderSelection =
            serde_json::from_slice(&encoded).expect("round trip selection");
        assert_eq!(decoded, selection);

        let empty = ProviderSelection {
            rejected_providers: vec![RejectedProvider {
                provider_id: ProviderId::new("provider/preferred").expect("provider id"),
                reasons: PlanProviderRejectReason::StorageIncompatible {
                    resource_ids: Vec::new(),
                },
            }],
            ..selection
        };
        assert!(ExecutionPlan::validate_provider_selection_evidence(&empty).is_err());
    }
}
