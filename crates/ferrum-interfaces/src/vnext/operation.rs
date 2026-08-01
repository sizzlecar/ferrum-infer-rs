use serde::{Deserialize, Deserializer, Serialize, Serializer};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::io::{self, Write};

use super::{
    classify_device_error, AdmittedSequenceResources, AllocationLifetime, BatchInvocationId,
    BatchParticipantAuthority, BatchParticipantTokenRange, BatchStepId, BatchWorkShape,
    BufferDescriptor, BufferUsage, CanonicalRational, CapabilityId, ClaimedSubmissionWaveBacking,
    ContractVersion, DefinitelyNotSubmittedWaveRetryAuthority, DeviceBatchingForm,
    DeviceBufferRetention, DeviceCommandBatch, DeviceCommandLogicalWork, DeviceId,
    DeviceReusableAddressScope, DeviceReusableExecutionTopologyFingerprint, DeviceRuntime,
    DynamicResourceDemand, DynamicResourceShape, EncodedDeviceOperation,
    EncodedReusableExecutionBindings, ExecutablePlanView, ExecutionIdentityEnvelope,
    ExecutionIdentityParts, ExecutionLane, ExecutionLaneId, HostTransferLayout,
    InvocationResourceLease, LeasedBufferView, LogicalAdmissionCoordinatorId,
    LogicalBackingBufferView, LogicalBackingSegmentBinding, LogicalBackingSliceAuthority,
    MemoryPlan, NodeId, NodeInvocationId, NodeWorkContract, OperationId, ParticipantNodeKey,
    PlanHash, PlanId, PlanNode, PreparedStepSubmissionNode, PreparedStepSubmissionWave,
    ProgramBindingNodeBinding, ProgramValueId, ProviderId, ProviderWorkspaceRequirement,
    ProviderWorkspaceReusePolicy, QuantizationFormatId, RequestIdentity, ResolvedWeightBinding,
    ResourceId, ResourcePoolId, ResourceWorkShape, RunId, SemanticValue, SequenceBackingSnapshot,
    SequenceSessionEpoch, SequenceSessionFingerprint, SpanId, StepParticipantFrameAssignment,
    StepResourceLease, TransactionId, TrustedActiveSequenceBinding, TrustedPlanRuntimeEvidence,
    UnvalidatedExecutionIdentityParts, VNextError, WeightFormatId, WeightId,
    EXECUTION_IDENTITY_VERSION,
};

mod buffer_view;
mod catalog;
mod compiled_submission_wave;
mod determinism;
mod determinism_artifact;
mod dispatch;
mod identity;
mod invocation;
mod provider;
mod registry;

pub use buffer_view::{
    OperationBufferRegionIter, OperationBufferRegions, OperationBufferStorageKind,
    OperationBufferView, OperationPhysicalRegion,
};
pub use catalog::CapabilityCatalog;
pub use compiled_submission_wave::CompiledSubmissionWaveIdentity;
pub use determinism::{
    SubmissionWaveDeterminismEvidence, SubmissionWaveDeterminismHandle,
    SubmissionWaveDeterminismInitializationIdentity, SubmissionWaveDeterminismLogicalRange,
    SubmissionWaveDeterminismPhysicalReadback, SubmissionWaveDeterminismReadbackPlan,
    SubmissionWaveDeterminismReadbackTarget, SubmissionWaveDeterminismRestore,
    SubmissionWaveDeterminismRestoreLayout, SubmissionWaveDeterminismWitnessReadback,
};
pub use determinism_artifact::{
    SubmissionWaveDeterminismArtifactAttribution, SubmissionWaveDeterminismArtifactExecution,
    SubmissionWaveDeterminismArtifactInitializationIdentity,
    SubmissionWaveDeterminismArtifactLogicalCommand,
    SubmissionWaveDeterminismArtifactPhysicalCommand,
    SubmissionWaveDeterminismArtifactReplayedSegment, SubmissionWaveDeterminismArtifactWitness,
};
pub use dispatch::{
    BoundDeviceSubmissionAttribution, DispatchRetryAuthority, OperationDispatch,
    OperationDispatchError, ProfiledSubmissionHandle, SubmissionExecutionPolicy,
    SubmissionScratchInitialization, SubmissionWaveDispatchError, SubmissionWaveDispatchStage,
    SubmissionWaveDispatchTimingSink, SubmissionWaveInputUpload,
};
pub use identity::{
    BatchOperationIdentity, BatchOperationIdentityMaterializationSnapshot,
    BatchOperationNodeIdentity, BatchOperationParticipantIdentity,
};
pub use invocation::{BatchedOperationInvocation, OperationInvocation};
pub use provider::{
    EngineProviderDescriptor, ExecutionDeterminismRequirement, OperationFailure,
    OperationProviderDescriptor, ProviderCompatibilityRejectReason, ProviderCompatibilityRejection,
    ProviderCompatibilityReport, ProviderCompatibilityRequest,
    ProviderExecutionContractFingerprint, ProviderExecutionRepeatability,
    ProviderExecutionSemantics, ProviderReplayEquivalence, UnvalidatedOperationFailure,
    PROVIDER_EXECUTION_SEMANTICS_VERSION,
};
pub(crate) use registry::OperationRegistryAuthority;
pub use registry::{
    BoundOperationProvider, BoundOperationProviderSet, OperationPlanningHandle,
    OperationPlanningRegistry, OperationProvider, OperationResourceEstimate,
    OperationResourceEstimateRequest, OperationResourceEstimator, OperationRuntimeRegistry,
    ReusableExecutionTopology, ReusableExecutionTopologyRequest,
};

pub const MAX_OPERATION_CATALOG_ROWS: usize = 4096;
pub const MAX_OPERATION_PROVIDER_ROWS: usize = 16384;
pub const MAX_ENGINE_PROVIDER_ROWS: usize = 4096;
pub const MAX_OPERATION_FAILURE_WIRE_BYTES: usize = 16 * 1024;
pub const MAX_REFERENCE_ORACLE_DEPTH: usize = 64;

fn invalid_operation(reason: impl Into<String>) -> VNextError {
    VNextError::InvalidExecutionPlan {
        reason: reason.into(),
    }
}

fn operation_error_for_node(error: VNextError, node_id: &NodeId) -> VNextError {
    match error {
        VNextError::UnsupportedOperation {
            node_id: None,
            operation_id,
            device_id,
            reason,
        } => VNextError::UnsupportedOperation {
            node_id: Some(node_id.to_string()),
            operation_id,
            device_id,
            reason,
        },
        VNextError::IncompatibleOperationVersion {
            node_id: None,
            operation_id,
            required_major,
            required_minor,
            available_major,
            available_minor,
        } => VNextError::IncompatibleOperationVersion {
            node_id: Some(node_id.to_string()),
            operation_id,
            required_major,
            required_minor,
            available_major,
            available_minor,
        },
        error => error,
    }
}

struct OperationFingerprintWriter<'a>(&'a mut Sha256);

impl Write for OperationFingerprintWriter<'_> {
    fn write(&mut self, buffer: &[u8]) -> io::Result<usize> {
        self.0.update(buffer);
        Ok(buffer.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

fn canonical_operation_fingerprint(
    value: &impl Serialize,
    failure_context: &'static str,
) -> Result<String, VNextError> {
    let mut digest = Sha256::new();
    serde_json::to_writer(OperationFingerprintWriter(&mut digest), value)
        .map_err(|error| invalid_operation(format!("{failure_context}: {error}")))?;
    Ok(format!("{:x}", digest.finalize()))
}

fn canonical_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn is_axis_permutation(axis_order: &[u32], rank: usize) -> bool {
    axis_order.len() == rank
        && axis_order.iter().copied().collect::<BTreeSet<_>>()
            == (0..rank as u32).collect::<BTreeSet<_>>()
}

/// Stable semantic attribute identity. Attribute names are data, not ad-hoc
/// strings interpreted by an individual provider.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct AttributeId(String);

impl AttributeId {
    pub fn new(value: impl Into<String>) -> Result<Self, VNextError> {
        let value = value.into();
        if value.is_empty() || value.len() > 160 {
            return Err(VNextError::InvalidIdentity {
                kind: "operation attribute",
                value,
                reason: "identity must contain between 1 and 160 bytes",
            });
        }
        if !value.bytes().all(|byte| {
            byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-' | b':' | b'/')
        }) {
            return Err(VNextError::InvalidIdentity {
                kind: "operation attribute",
                value,
                reason: "identity contains a non-portable character",
            });
        }
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for AttributeId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl TryFrom<String> for AttributeId {
    type Error = VNextError;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        Self::new(value)
    }
}

impl From<AttributeId> for String {
    fn from(value: AttributeId) -> Self {
        value.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AttributeValueKind {
    Bool,
    Integer,
    Unsigned,
    Rational,
    Text,
    Integers,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AttributeSpec {
    pub value_kind: AttributeValueKind,
    pub required: bool,
    pub constraint: AttributeConstraint,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AttributeConstraint {
    None,
    BoolEquals(bool),
    IntegerRange {
        minimum: i64,
        maximum: i64,
    },
    UnsignedRange {
        minimum: u64,
        maximum: u64,
    },
    RationalRange {
        minimum: CanonicalRational,
        maximum: CanonicalRational,
    },
    TextChoices {
        values: BTreeSet<String>,
    },
    IntegerListLength {
        minimum: u32,
        maximum: u32,
    },
}

/// Closed attribute vocabulary for one operation contract.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct AttributeSchema {
    entries: BTreeMap<AttributeId, AttributeSpec>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct AttributeSchemaWire {
    entries: BTreeMap<AttributeId, AttributeSpec>,
}

impl<'de> Deserialize<'de> for AttributeSchema {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = AttributeSchemaWire::deserialize(deserializer)?;
        Self::new(wire.entries).map_err(serde::de::Error::custom)
    }
}

impl AttributeSchema {
    pub fn new(entries: BTreeMap<AttributeId, AttributeSpec>) -> Result<Self, VNextError> {
        for (attribute_id, spec) in &entries {
            spec.validate(attribute_id)?;
        }
        Ok(Self { entries })
    }

    pub fn empty() -> Self {
        Self {
            entries: BTreeMap::new(),
        }
    }

    pub fn entries(&self) -> &BTreeMap<AttributeId, AttributeSpec> {
        &self.entries
    }

    pub fn validate_values(
        &self,
        values: &BTreeMap<AttributeId, SemanticValue>,
        context: &str,
    ) -> Result<(), VNextError> {
        for (attribute_id, value) in values {
            let spec = self.entries.get(attribute_id).ok_or_else(|| {
                invalid_operation(format!(
                    "{context} contains unknown attribute `{attribute_id}`"
                ))
            })?;
            value.validate(context)?;
            if value.kind() != spec.value_kind {
                return Err(invalid_operation(format!(
                    "{context} attribute `{attribute_id}` has the wrong value kind"
                )));
            }
            spec.validate_value(attribute_id, value)?;
        }
        if let Some(attribute_id) = self.entries.iter().find_map(|(attribute_id, spec)| {
            (spec.required && !values.contains_key(attribute_id)).then_some(attribute_id)
        }) {
            return Err(invalid_operation(format!(
                "{context} is missing required attribute `{attribute_id}`"
            )));
        }
        Ok(())
    }
}

impl AttributeSpec {
    fn validate(&self, attribute_id: &AttributeId) -> Result<(), VNextError> {
        let compatible = match (&self.value_kind, &self.constraint) {
            (_, AttributeConstraint::None) => true,
            (AttributeValueKind::Bool, AttributeConstraint::BoolEquals(_)) => true,
            (
                AttributeValueKind::Integer,
                AttributeConstraint::IntegerRange { minimum, maximum },
            ) => minimum <= maximum,
            (
                AttributeValueKind::Unsigned,
                AttributeConstraint::UnsignedRange { minimum, maximum },
            ) => minimum <= maximum,
            (AttributeValueKind::Text, AttributeConstraint::TextChoices { values }) => {
                !values.is_empty() && values.iter().all(|value| !value.is_empty())
            }
            (
                AttributeValueKind::Integers,
                AttributeConstraint::IntegerListLength { minimum, maximum },
            ) => minimum <= maximum,
            (
                AttributeValueKind::Rational,
                AttributeConstraint::RationalRange { minimum, maximum },
            ) => {
                (minimum.numerator() as i128) * (maximum.denominator() as i128)
                    <= (maximum.numerator() as i128) * (minimum.denominator() as i128)
            }
            _ => false,
        };
        if !compatible {
            return Err(invalid_operation(format!(
                "attribute `{attribute_id}` has an incompatible or invalid constraint"
            )));
        }
        Ok(())
    }

    fn validate_value(
        &self,
        attribute_id: &AttributeId,
        value: &SemanticValue,
    ) -> Result<(), VNextError> {
        let accepted = match (&self.constraint, value) {
            (AttributeConstraint::None, _) => true,
            (AttributeConstraint::BoolEquals(expected), SemanticValue::Bool(actual)) => {
                expected == actual
            }
            (
                AttributeConstraint::IntegerRange { minimum, maximum },
                SemanticValue::Integer(actual),
            ) => minimum <= actual && actual <= maximum,
            (
                AttributeConstraint::UnsignedRange { minimum, maximum },
                SemanticValue::Unsigned(actual),
            ) => minimum <= actual && actual <= maximum,
            (
                AttributeConstraint::RationalRange { minimum, maximum },
                SemanticValue::Rational(actual),
            ) => {
                (actual.numerator() as i128) * (minimum.denominator() as i128)
                    >= (minimum.numerator() as i128) * (actual.denominator() as i128)
                    && (actual.numerator() as i128) * (maximum.denominator() as i128)
                        <= (maximum.numerator() as i128) * (actual.denominator() as i128)
            }
            (AttributeConstraint::TextChoices { values }, SemanticValue::Text(actual)) => {
                values.contains(actual)
            }
            (
                AttributeConstraint::IntegerListLength { minimum, maximum },
                SemanticValue::Integers(actual),
            ) => (*minimum as usize) <= actual.len() && actual.len() <= (*maximum as usize),
            _ => false,
        };
        if !accepted {
            return Err(invalid_operation(format!(
                "attribute `{attribute_id}` violates its declared constraint"
            )));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ElementType {
    Bool,
    U8,
    U32,
    I8,
    I32,
    F16,
    Bf16,
    F32,
}

impl ElementType {
    pub const fn size_bytes(self) -> u64 {
        match self {
            Self::Bool | Self::U8 | Self::I8 => 1,
            Self::F16 | Self::Bf16 => 2,
            Self::U32 | Self::I32 | Self::F32 => 4,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DynamicStorageAllocator {
    LinearArena,
    FixedBlockArena { block_bytes: u64 },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DynamicStorageView {
    Contiguous,
    PagedRegions { block_bytes: u64 },
}

/// Backend-neutral physical addressability offered by a runtime and accepted
/// by an operation provider. This is independent from capacity formulas.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
pub struct DynamicStorageProfile {
    allocator: DynamicStorageAllocator,
    view: DynamicStorageView,
}

impl DynamicStorageProfile {
    pub fn new(
        allocator: DynamicStorageAllocator,
        view: DynamicStorageView,
    ) -> Result<Self, VNextError> {
        let valid = match (allocator, view) {
            (DynamicStorageAllocator::LinearArena, DynamicStorageView::Contiguous) => true,
            (
                DynamicStorageAllocator::FixedBlockArena { block_bytes },
                DynamicStorageView::Contiguous,
            ) => block_bytes.is_power_of_two(),
            (
                DynamicStorageAllocator::FixedBlockArena {
                    block_bytes: allocator_block,
                },
                DynamicStorageView::PagedRegions {
                    block_bytes: view_block,
                },
            ) => allocator_block.is_power_of_two() && allocator_block == view_block,
            (DynamicStorageAllocator::LinearArena, DynamicStorageView::PagedRegions { .. }) => {
                false
            }
        };
        if !valid {
            return Err(invalid_operation(
                "dynamic storage allocator/view profile is incompatible or invalid",
            ));
        }
        Ok(Self { allocator, view })
    }

    pub const fn allocator(self) -> DynamicStorageAllocator {
        self.allocator
    }

    pub const fn view(self) -> DynamicStorageView {
        self.view
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct DynamicStorageProfileWire {
    allocator: DynamicStorageAllocator,
    view: DynamicStorageView,
}

impl<'de> Deserialize<'de> for DynamicStorageProfile {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = DynamicStorageProfileWire::deserialize(deserializer)?;
        Self::new(wire.allocator, wire.view).map_err(serde::de::Error::custom)
    }
}

/// Canonical non-empty set of profiles accepted by a provider binding or one
/// provider-owned workspace. The planner intersects this with runtime offers
/// and the ordered runtime-policy allowlist.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct DynamicStorageRequirement {
    accepted_profiles: Vec<DynamicStorageProfile>,
}

impl DynamicStorageRequirement {
    pub fn new(mut accepted_profiles: Vec<DynamicStorageProfile>) -> Result<Self, VNextError> {
        accepted_profiles.sort_unstable();
        accepted_profiles.dedup();
        if accepted_profiles.is_empty() {
            return Err(invalid_operation(
                "dynamic storage requirement has no accepted profile",
            ));
        }
        Ok(Self { accepted_profiles })
    }

    pub fn contiguous() -> Self {
        Self {
            accepted_profiles: vec![DynamicStorageProfile {
                allocator: DynamicStorageAllocator::LinearArena,
                view: DynamicStorageView::Contiguous,
            }],
        }
    }

    pub fn accepted_profiles(&self) -> &[DynamicStorageProfile] {
        &self.accepted_profiles
    }

    pub fn accepts(&self, profile: DynamicStorageProfile) -> bool {
        self.accepted_profiles.binary_search(&profile).is_ok()
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct DynamicStorageRequirementWire {
    accepted_profiles: Vec<DynamicStorageProfile>,
}

impl<'de> Deserialize<'de> for DynamicStorageRequirement {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = DynamicStorageRequirementWire::deserialize(deserializer)?;
        let original = wire.accepted_profiles.clone();
        let requirement = Self::new(wire.accepted_profiles).map_err(serde::de::Error::custom)?;
        if requirement.accepted_profiles != original {
            return Err(serde::de::Error::custom(
                "dynamic storage requirement profiles are not canonical and unique",
            ));
        }
        Ok(requirement)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DimensionConstraint {
    Exact(u64),
    Symbol(String),
    Range { minimum: u64, maximum: u64 },
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StrideConstraint {
    ExactBytes(u64),
    Symbol(String),
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LayoutConstraint {
    Contiguous,
    Strided {
        strides: Vec<StrideConstraint>,
    },
    Blocked {
        block: Vec<u64>,
        axis_order: Vec<u32>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TensorAccess {
    Read,
    Write,
    ReadWrite,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AliasPolicy {
    NoAlias,
    MayAlias { tensor_index: u32 },
    MustAlias { tensor_index: u32 },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct TensorContract {
    dimensions: Vec<DimensionConstraint>,
    element_types: BTreeSet<ElementType>,
    layouts: Vec<LayoutConstraint>,
    access: TensorAccess,
    alias: AliasPolicy,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct TensorContractWire {
    dimensions: Vec<DimensionConstraint>,
    element_types: BTreeSet<ElementType>,
    layouts: Vec<LayoutConstraint>,
    access: TensorAccess,
    alias: AliasPolicy,
}

impl<'de> Deserialize<'de> for TensorContract {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = TensorContractWire::deserialize(deserializer)?;
        Self::new(
            wire.dimensions,
            wire.element_types,
            wire.layouts,
            wire.access,
            wire.alias,
        )
        .map_err(serde::de::Error::custom)
    }
}

impl TensorContract {
    pub fn new(
        dimensions: Vec<DimensionConstraint>,
        element_types: BTreeSet<ElementType>,
        mut layouts: Vec<LayoutConstraint>,
        access: TensorAccess,
        alias: AliasPolicy,
    ) -> Result<Self, VNextError> {
        layouts.sort();
        layouts.dedup();
        let contract = Self {
            dimensions,
            element_types,
            layouts,
            access,
            alias,
        };
        contract.validate("tensor_contract")?;
        Ok(contract)
    }

    pub fn dimensions(&self) -> &[DimensionConstraint] {
        &self.dimensions
    }

    pub fn element_types(&self) -> &BTreeSet<ElementType> {
        &self.element_types
    }

    pub fn layouts(&self) -> &[LayoutConstraint] {
        &self.layouts
    }

    pub const fn access(&self) -> TensorAccess {
        self.access
    }

    pub fn alias(&self) -> &AliasPolicy {
        &self.alias
    }

    pub fn validate(&self, field: &str) -> Result<(), VNextError> {
        if self.element_types.is_empty() {
            return Err(VNextError::InvalidExecutionPlan {
                reason: format!("{field} has no allowed element type"),
            });
        }
        if self.layouts.is_empty() {
            return Err(VNextError::InvalidExecutionPlan {
                reason: format!("{field} has no allowed layout"),
            });
        }
        for (index, dimension) in self.dimensions.iter().enumerate() {
            match dimension {
                DimensionConstraint::Exact(0) | DimensionConstraint::Range { minimum: 0, .. } => {
                    return Err(VNextError::InvalidExecutionPlan {
                        reason: format!("{field}.dimensions[{index}] permits a zero extent"),
                    });
                }
                DimensionConstraint::Range { minimum, maximum } if minimum > maximum => {
                    return Err(VNextError::InvalidExecutionPlan {
                        reason: format!("{field}.dimensions[{index}] has an inverted range"),
                    });
                }
                DimensionConstraint::Symbol(symbol) if symbol.trim().is_empty() => {
                    return Err(VNextError::InvalidExecutionPlan {
                        reason: format!("{field}.dimensions[{index}] has an empty symbol"),
                    });
                }
                _ => {}
            }
        }
        for (index, layout) in self.layouts.iter().enumerate() {
            match layout {
                LayoutConstraint::Strided { strides }
                    if strides.len() != self.dimensions.len()
                        || strides.iter().any(|stride| match stride {
                            StrideConstraint::ExactBytes(bytes) => *bytes == 0,
                            StrideConstraint::Symbol(symbol) => symbol.trim().is_empty(),
                        }) =>
                {
                    return Err(VNextError::InvalidExecutionPlan {
                        reason: format!("{field}.layouts[{index}] has invalid strides"),
                    });
                }
                LayoutConstraint::Blocked { block, axis_order }
                    if block.len() != self.dimensions.len()
                        || block.iter().any(|extent| *extent == 0)
                        || !is_axis_permutation(axis_order, self.dimensions.len()) =>
                {
                    return Err(VNextError::InvalidExecutionPlan {
                        reason: format!("{field}.layouts[{index}] has an invalid block"),
                    });
                }
                _ => {}
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BlockedTensorPadding {
    Exact,
    ZeroFill { physical_dimensions: Vec<u64> },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResolvedTensorLayout {
    Contiguous,
    Strided {
        byte_strides: Vec<u64>,
    },
    Blocked {
        block: Vec<u64>,
        axis_order: Vec<u32>,
        padding: BlockedTensorPadding,
    },
}

/// Concrete tensor shape selected by planning and consumed unchanged by an
/// operation provider.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ResolvedTensorSpec {
    dimensions: Vec<u64>,
    element_type: ElementType,
    layout: ResolvedTensorLayout,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ResolvedTensorSpecWire {
    dimensions: Vec<u64>,
    element_type: ElementType,
    layout: ResolvedTensorLayout,
}

impl<'de> Deserialize<'de> for ResolvedTensorSpec {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = ResolvedTensorSpecWire::deserialize(deserializer)?;
        Self::new(wire.dimensions, wire.element_type, wire.layout).map_err(serde::de::Error::custom)
    }
}

impl ResolvedTensorSpec {
    pub fn new(
        dimensions: Vec<u64>,
        element_type: ElementType,
        layout: ResolvedTensorLayout,
    ) -> Result<Self, VNextError> {
        if dimensions.iter().any(|extent| *extent == 0) {
            return Err(invalid_operation(
                "resolved tensor dimensions must be non-zero",
            ));
        }
        match &layout {
            ResolvedTensorLayout::Strided { byte_strides }
                if byte_strides.len() != dimensions.len()
                    || byte_strides.iter().any(|stride| *stride == 0) =>
            {
                return Err(invalid_operation(
                    "resolved tensor byte strides must match rank and be non-zero",
                ));
            }
            ResolvedTensorLayout::Blocked {
                block,
                axis_order,
                padding,
            } => {
                if block.len() != dimensions.len()
                    || block.iter().any(|extent| *extent == 0)
                    || !is_axis_permutation(axis_order, dimensions.len())
                {
                    return Err(invalid_operation(
                        "resolved tensor block and axis order must form a non-zero ranked layout",
                    ));
                }
                match padding {
                    BlockedTensorPadding::Exact => {
                        if dimensions
                            .iter()
                            .zip(block)
                            .any(|(extent, block)| extent % block != 0)
                        {
                            return Err(invalid_operation(
                                "exact blocked tensors require every logical extent to be block-divisible",
                            ));
                        }
                    }
                    BlockedTensorPadding::ZeroFill {
                        physical_dimensions,
                    } => {
                        if physical_dimensions.len() != dimensions.len() {
                            return Err(invalid_operation(
                                "zero-filled blocked tensor padding must match tensor rank",
                            ));
                        }
                        let mut has_padding = false;
                        let mut padded_logical = Vec::with_capacity(dimensions.len());
                        for (logical, block) in dimensions.iter().zip(block) {
                            let expected = logical
                                .checked_add(block - 1)
                                .map(|extent| extent / block * block)
                                .ok_or_else(|| {
                                    invalid_operation(
                                        "zero-filled blocked tensor padding overflows u64",
                                    )
                                })?;
                            padded_logical.push(expected);
                            has_padding |= expected != *logical;
                        }
                        let expected_physical = axis_order
                            .iter()
                            .map(|axis| padded_logical[*axis as usize])
                            .collect::<Vec<_>>();
                        if *physical_dimensions != expected_physical {
                            return Err(invalid_operation(
                                "zero-filled blocked tensor physical shape is not the minimal block-aligned axis permutation",
                            ));
                        }
                        if !has_padding {
                            return Err(invalid_operation(
                                "zero-filled blocked tensor layout must contain actual padding; use Exact otherwise",
                            ));
                        }
                    }
                }
            }
            _ => {}
        }
        dimensions
            .iter()
            .try_fold(element_type.size_bytes(), |bytes, extent| {
                bytes.checked_mul(*extent)
            })
            .ok_or_else(|| invalid_operation("resolved tensor byte size overflows u64"))?;
        Ok(Self {
            dimensions,
            element_type,
            layout,
        })
    }

    pub fn dimensions(&self) -> &[u64] {
        &self.dimensions
    }

    pub fn element_type(&self) -> ElementType {
        self.element_type
    }

    pub fn layout(&self) -> &ResolvedTensorLayout {
        &self.layout
    }

    pub fn minimum_storage_bytes(&self) -> Result<u64, VNextError> {
        match &self.layout {
            ResolvedTensorLayout::Contiguous => self
                .dimensions
                .iter()
                .try_fold(self.element_type.size_bytes(), |bytes, extent| {
                    bytes.checked_mul(*extent)
                })
                .ok_or_else(|| invalid_operation("resolved tensor byte size overflows u64")),
            ResolvedTensorLayout::Blocked { padding, .. } => {
                let storage_dimensions = match padding {
                    BlockedTensorPadding::Exact => &self.dimensions,
                    BlockedTensorPadding::ZeroFill {
                        physical_dimensions,
                    } => physical_dimensions,
                };
                storage_dimensions
                    .iter()
                    .try_fold(self.element_type.size_bytes(), |bytes, extent| {
                        bytes.checked_mul(*extent)
                    })
                    .ok_or_else(|| {
                        invalid_operation("resolved blocked tensor byte size overflows u64")
                    })
            }
            ResolvedTensorLayout::Strided { byte_strides } => self
                .dimensions
                .iter()
                .zip(byte_strides)
                .try_fold(self.element_type.size_bytes(), |span, (extent, stride)| {
                    extent
                        .checked_sub(1)
                        .and_then(|steps| steps.checked_mul(*stride))
                        .and_then(|bytes| span.checked_add(bytes))
                })
                .ok_or_else(|| invalid_operation("resolved strided tensor span overflows u64")),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResolvedValueRole {
    Input,
    Output,
}

/// Provider-accepted physical profiles for one exact operation binding slot.
/// Role and ordinal are contract identities, not model-specific names.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ProviderStorageBindingRequirement {
    role: ResolvedValueRole,
    ordinal: u32,
    storage: DynamicStorageRequirement,
}

impl ProviderStorageBindingRequirement {
    pub fn new(role: ResolvedValueRole, ordinal: u32, storage: DynamicStorageRequirement) -> Self {
        Self {
            role,
            ordinal,
            storage,
        }
    }

    pub const fn role(&self) -> ResolvedValueRole {
        self.role
    }

    pub const fn ordinal(&self) -> u32 {
        self.ordinal
    }

    pub fn storage(&self) -> &DynamicStorageRequirement {
        &self.storage
    }

    fn canonical_key(&self) -> (ResolvedValueRole, u32) {
        (self.role, self.ordinal)
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ProviderStorageBindingRequirementWire {
    role: ResolvedValueRole,
    ordinal: u32,
    storage: DynamicStorageRequirement,
}

impl<'de> Deserialize<'de> for ProviderStorageBindingRequirement {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = ProviderStorageBindingRequirementWire::deserialize(deserializer)?;
        Ok(Self::new(wire.role, wire.ordinal, wire.storage))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ResolvedStorageComponent {
    component_id: Option<WeightId>,
    resource_id: ResourceId,
    offset_bytes: u64,
    length_bytes: u64,
    element_type: ElementType,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ResolvedStorageComponentWire {
    component_id: Option<WeightId>,
    resource_id: ResourceId,
    offset_bytes: u64,
    length_bytes: u64,
    element_type: ElementType,
}

impl<'de> Deserialize<'de> for ResolvedStorageComponent {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = ResolvedStorageComponentWire::deserialize(deserializer)?;
        Self::new(
            wire.component_id,
            wire.resource_id,
            wire.offset_bytes,
            wire.length_bytes,
            wire.element_type,
        )
        .map_err(serde::de::Error::custom)
    }
}

impl ResolvedStorageComponent {
    pub fn new(
        component_id: Option<WeightId>,
        resource_id: ResourceId,
        offset_bytes: u64,
        length_bytes: u64,
        element_type: ElementType,
    ) -> Result<Self, VNextError> {
        if length_bytes == 0
            || offset_bytes.checked_add(length_bytes).is_none()
            || offset_bytes % element_type.size_bytes() != 0
            || length_bytes % element_type.size_bytes() != 0
        {
            return Err(invalid_operation(
                "resolved storage component is empty or overflows u64",
            ));
        }
        Ok(Self {
            component_id,
            resource_id,
            offset_bytes,
            length_bytes,
            element_type,
        })
    }

    pub fn component_id(&self) -> Option<&WeightId> {
        self.component_id.as_ref()
    }

    pub fn resource_id(&self) -> &ResourceId {
        &self.resource_id
    }

    pub const fn offset_bytes(&self) -> u64 {
        self.offset_bytes
    }

    pub const fn length_bytes(&self) -> u64 {
        self.length_bytes
    }

    pub const fn element_type(&self) -> ElementType {
        self.element_type
    }
}

/// Physical resources backing one semantic value. A logical quantized weight
/// can bind packed values, scales, zero-points, and indices without pretending
/// they are one dense allocation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ResolvedValueStorage {
    components: Vec<ResolvedStorageComponent>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ResolvedValueStorageWire {
    components: Vec<ResolvedStorageComponent>,
}

impl<'de> Deserialize<'de> for ResolvedValueStorage {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = ResolvedValueStorageWire::deserialize(deserializer)?;
        Self::new(wire.components).map_err(serde::de::Error::custom)
    }
}

impl ResolvedValueStorage {
    pub fn single(
        resource_id: ResourceId,
        offset_bytes: u64,
        length_bytes: u64,
        element_type: ElementType,
    ) -> Result<Self, VNextError> {
        Self::new(vec![ResolvedStorageComponent::new(
            None,
            resource_id,
            offset_bytes,
            length_bytes,
            element_type,
        )?])
    }

    pub fn composite(components: Vec<ResolvedStorageComponent>) -> Result<Self, VNextError> {
        if components
            .iter()
            .any(|component| component.component_id.is_none())
        {
            return Err(invalid_operation(
                "composite value storage requires a physical component identity",
            ));
        }
        Self::new(components)
    }

    fn new(mut components: Vec<ResolvedStorageComponent>) -> Result<Self, VNextError> {
        if components.is_empty() {
            return Err(invalid_operation("resolved value storage is empty"));
        }
        if components.len() > 1
            && components
                .iter()
                .any(|component| component.component_id.is_none())
        {
            return Err(invalid_operation(
                "multi-component value storage requires physical component identities",
            ));
        }
        components.sort_by(|left, right| {
            left.component_id
                .cmp(&right.component_id)
                .then(left.resource_id.cmp(&right.resource_id))
                .then(left.offset_bytes.cmp(&right.offset_bytes))
        });
        let mut component_ids = BTreeSet::new();
        for (index, component) in components.iter().enumerate() {
            if component.length_bytes == 0
                || component
                    .offset_bytes
                    .checked_add(component.length_bytes)
                    .is_none()
                || component
                    .component_id
                    .as_ref()
                    .is_some_and(|component_id| !component_ids.insert(component_id.clone()))
            {
                return Err(invalid_operation(
                    "resolved value storage has invalid or duplicate components",
                ));
            }
            if components[..index].iter().any(|previous| {
                previous.resource_id == component.resource_id
                    && previous.offset_bytes
                        < component
                            .offset_bytes
                            .saturating_add(component.length_bytes)
                    && component.offset_bytes
                        < previous.offset_bytes.saturating_add(previous.length_bytes)
            }) {
                return Err(invalid_operation(
                    "resolved value storage components overlap in one resource",
                ));
            }
        }
        Ok(Self { components })
    }

    pub fn components(&self) -> &[ResolvedStorageComponent] {
        &self.components
    }

    pub fn resource_ids(&self) -> BTreeSet<&ResourceId> {
        self.components
            .iter()
            .map(|component| &component.resource_id)
            .collect()
    }

    pub fn total_physical_bytes(&self) -> Result<u64, VNextError> {
        self.components.iter().try_fold(0_u64, |total, component| {
            total
                .checked_add(component.length_bytes)
                .ok_or_else(|| invalid_operation("resolved storage byte count overflows u64"))
        })
    }
}

/// Value/resource binding shared by the execution plan and provider
/// invocation. Keeping one representation prevents a lossy translation at the
/// runtime boundary.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ResolvedValueBinding {
    value_id: ProgramValueId,
    role: ResolvedValueRole,
    ordinal: u32,
    tensor: ResolvedTensorSpec,
    access: TensorAccess,
    alias: AliasPolicy,
    usage: BufferUsage,
    weight: Option<ResolvedWeightBinding>,
    storage: ResolvedValueStorage,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ResolvedValueBindingWire {
    value_id: ProgramValueId,
    role: ResolvedValueRole,
    ordinal: u32,
    tensor: ResolvedTensorSpec,
    access: TensorAccess,
    alias: AliasPolicy,
    usage: BufferUsage,
    weight: Option<ResolvedWeightBinding>,
    storage: ResolvedValueStorage,
}

impl<'de> Deserialize<'de> for ResolvedValueBinding {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = ResolvedValueBindingWire::deserialize(deserializer)?;
        Self::new(
            wire.value_id,
            wire.role,
            wire.ordinal,
            wire.tensor,
            wire.access,
            wire.alias,
            wire.usage,
            wire.weight,
            wire.storage,
        )
        .map_err(serde::de::Error::custom)
    }
}

impl ResolvedValueBinding {
    pub fn new(
        value_id: ProgramValueId,
        role: ResolvedValueRole,
        ordinal: u32,
        tensor: ResolvedTensorSpec,
        access: TensorAccess,
        alias: AliasPolicy,
        usage: BufferUsage,
        weight: Option<ResolvedWeightBinding>,
        storage: ResolvedValueStorage,
    ) -> Result<Self, VNextError> {
        if (role == ResolvedValueRole::Input
            && !matches!(access, TensorAccess::Read | TensorAccess::ReadWrite))
            || (role == ResolvedValueRole::Output
                && !matches!(access, TensorAccess::Write | TensorAccess::ReadWrite))
            || (role == ResolvedValueRole::Input && !matches!(alias, AliasPolicy::NoAlias))
        {
            return Err(invalid_operation(
                "resolved value role, access, and alias policy are inconsistent",
            ));
        }
        if usage != BufferUsage::Weights && storage.components.len() != 1 {
            return Err(invalid_operation(
                "only a weight value may use composite physical storage",
            ));
        }
        if storage.components.len() == 1
            && storage.components[0].component_id.is_none()
            && storage.components[0].element_type != tensor.element_type
        {
            return Err(invalid_operation(
                "single-resource value dtype differs from its logical tensor dtype",
            ));
        }
        if usage != BufferUsage::Weights
            && storage.components[0].length_bytes < tensor.minimum_storage_bytes()?
        {
            return Err(invalid_operation(
                "resolved value storage is smaller than its tensor span",
            ));
        }
        match (usage, weight.as_ref()) {
            (BufferUsage::Weights, Some(weight)) => {
                weight.validate_logical(tensor.dimensions(), tensor.element_type())?;
                validate_resolved_weight_storage(weight, &storage)?;
            }
            (BufferUsage::Weights, None) => {
                return Err(invalid_operation(
                    "weight value lacks its resolved physical layout contract",
                ));
            }
            (_, Some(_)) => {
                return Err(invalid_operation(
                    "non-weight value carries a resolved weight layout contract",
                ));
            }
            (_, None) => {}
        }
        Ok(Self {
            value_id,
            role,
            ordinal,
            tensor,
            access,
            alias,
            usage,
            weight,
            storage,
        })
    }

    pub fn value_id(&self) -> &ProgramValueId {
        &self.value_id
    }

    pub fn role(&self) -> ResolvedValueRole {
        self.role
    }

    pub fn ordinal(&self) -> u32 {
        self.ordinal
    }

    pub fn tensor(&self) -> &ResolvedTensorSpec {
        &self.tensor
    }

    pub fn access(&self) -> TensorAccess {
        self.access
    }

    pub fn alias(&self) -> &AliasPolicy {
        &self.alias
    }

    pub const fn usage(&self) -> BufferUsage {
        self.usage
    }

    pub fn weight(&self) -> Option<&ResolvedWeightBinding> {
        self.weight.as_ref()
    }

    pub fn storage(&self) -> &ResolvedValueStorage {
        &self.storage
    }
}

fn validate_resolved_weight_storage(
    weight: &ResolvedWeightBinding,
    storage: &ResolvedValueStorage,
) -> Result<(), VNextError> {
    let expected = weight
        .components()
        .iter()
        .map(|component| (component.component_id(), component))
        .collect::<BTreeMap<_, _>>();
    if storage.components().len() != expected.len() {
        return Err(invalid_operation(
            "resolved weight storage component count differs from its layout contract",
        ));
    }
    let mut seen = BTreeSet::new();
    for stored in storage.components() {
        let component_id = stored.component_id().ok_or_else(|| {
            invalid_operation("resolved weight storage component lacks its physical identity")
        })?;
        let component = expected.get(component_id).ok_or_else(|| {
            invalid_operation(format!(
                "resolved weight storage contains unknown component `{component_id}`"
            ))
        })?;
        if !seen.insert(component_id)
            || stored.length_bytes() != component.physical_bytes()?
            || stored.element_type() != component.physical_element_type()
        {
            return Err(invalid_operation(format!(
                "resolved weight storage component `{component_id}` differs from its layout contract"
            )));
        }
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResourcePresenceRequirement {
    Forbidden,
    Optional,
    Required,
}

impl ResourcePresenceRequirement {
    pub const fn accepts(self, present: bool) -> bool {
        matches!(
            (self, present),
            (Self::Forbidden, false) | (Self::Optional, _) | (Self::Required, true)
        )
    }
}

/// Shape-independent resource contract. Concrete byte counts, scopes, and
/// alignment are produced by the selected provider's versioned estimator and
/// bound into the immutable execution plan.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResourceRequirements {
    pub minimum_value_alignment_bytes: u64,
    pub scratch: ResourcePresenceRequirement,
    /// Small request-shaped control workspace whose contents are written in
    /// the wave binding preamble and consumed by reusable compute.
    pub binding: ResourcePresenceRequirement,
    pub persistent: ResourcePresenceRequirement,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OracleSpec {
    Exact,
    AbsoluteTolerance {
        tolerance: CanonicalRational,
    },
    RelativeTolerance {
        tolerance: CanonicalRational,
    },
    ReferenceOperation {
        operation_id: OperationId,
        version: ContractVersion,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProfilePhase {
    Load,
    Prepare,
    /// Backend operation shared by prefill and decode. The exact request phase
    /// is derived from the bound work shape rather than changing operation
    /// identity or selecting another provider in the hot path.
    Forward,
    Prefill,
    Decode,
    Transfer,
    Synchronize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProviderRequirement {
    pub minimum_version: ContractVersion,
    pub required_capabilities: BTreeSet<CapabilityId>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OperationDescriptor {
    pub id: OperationId,
    pub version: ContractVersion,
    pub inputs: Vec<TensorContract>,
    pub outputs: Vec<TensorContract>,
    pub attributes: AttributeSchema,
    pub resources: ResourceRequirements,
    pub oracle: OracleSpec,
    pub provider: ProviderRequirement,
    pub profile_phase: ProfilePhase,
}

impl OperationDescriptor {
    pub fn validate(&self) -> Result<(), VNextError> {
        if self.version.major == 0 {
            return Err(VNextError::InvalidExecutionPlan {
                reason: format!("operation `{}` has an unstable zero major version", self.id),
            });
        }
        if self.outputs.is_empty() {
            return Err(VNextError::InvalidExecutionPlan {
                reason: format!("operation `{}` has no outputs", self.id),
            });
        }
        for (index, input) in self.inputs.iter().enumerate() {
            input.validate(&format!("operation.{}.inputs[{index}]", self.id))?;
            if !matches!(input.access, TensorAccess::Read | TensorAccess::ReadWrite)
                || !matches!(input.alias, AliasPolicy::NoAlias)
            {
                return Err(VNextError::InvalidExecutionPlan {
                    reason: format!(
                        "operation `{}` input {index} has invalid access or alias semantics",
                        self.id
                    ),
                });
            }
        }
        for (index, output) in self.outputs.iter().enumerate() {
            output.validate(&format!("operation.{}.outputs[{index}]", self.id))?;
            if !matches!(output.access, TensorAccess::Write | TensorAccess::ReadWrite) {
                return Err(VNextError::InvalidExecutionPlan {
                    reason: format!("operation `{}` output {index} is not writable", self.id),
                });
            }
            if let AliasPolicy::MayAlias { tensor_index }
            | AliasPolicy::MustAlias { tensor_index } = &output.alias
            {
                if *tensor_index as usize >= self.inputs.len() {
                    return Err(VNextError::InvalidExecutionPlan {
                        reason: format!("operation `{}` output {index} aliases no input", self.id),
                    });
                }
            }
        }
        if self.resources.minimum_value_alignment_bytes == 0
            || !self
                .resources
                .minimum_value_alignment_bytes
                .is_power_of_two()
        {
            return Err(VNextError::InvalidExecutionPlan {
                reason: format!("operation `{}` has invalid resource requirements", self.id),
            });
        }
        match self.oracle {
            OracleSpec::AbsoluteTolerance { tolerance }
            | OracleSpec::RelativeTolerance { tolerance }
                if tolerance.numerator() < 0 =>
            {
                return Err(VNextError::InvalidExecutionPlan {
                    reason: format!("operation `{}` has a negative oracle tolerance", self.id),
                });
            }
            OracleSpec::AbsoluteTolerance { .. } | OracleSpec::RelativeTolerance { .. }
                if self
                    .outputs
                    .iter()
                    .any(|output| output.element_types().contains(&ElementType::Bool)) =>
            {
                return Err(VNextError::InvalidExecutionPlan {
                    reason: format!(
                        "operation `{}` applies numeric oracle tolerance to a possible boolean output",
                        self.id
                    ),
                });
            }
            _ => {}
        }
        if self.provider.minimum_version.major == 0 {
            return Err(VNextError::InvalidExecutionPlan {
                reason: format!("operation `{}` has a zero provider major version", self.id),
            });
        }
        if self.provider.minimum_version.major != self.version.major {
            return Err(VNextError::InvalidExecutionPlan {
                reason: format!(
                    "operation `{}` version {} and provider minimum version {} have incompatible major versions",
                    self.id, self.version, self.provider.minimum_version
                ),
            });
        }
        Ok(())
    }

    pub fn fingerprint(&self) -> Result<String, VNextError> {
        self.validate()?;
        let bytes = serde_json::to_vec(self).map_err(|error| VNextError::Serialization {
            context: "serialize operation descriptor",
            message: error.to_string(),
        })?;
        Ok(format!("{:x}", Sha256::digest(bytes)))
    }

    pub fn validate_attributes(
        &self,
        values: &BTreeMap<AttributeId, SemanticValue>,
    ) -> Result<(), VNextError> {
        self.attributes
            .validate_values(values, &format!("operation.{}.attributes", self.id))
    }

    pub fn validate_resolved_bindings(
        &self,
        bindings: &[ResolvedValueBinding],
    ) -> Result<(), VNextError> {
        self.validate()?;
        if bindings.len() != self.inputs.len() + self.outputs.len() {
            return Err(invalid_operation(format!(
                "operation `{}` expects {} value bindings, received {}",
                self.id,
                self.inputs.len() + self.outputs.len(),
                bindings.len()
            )));
        }

        let mut dimensions = BTreeMap::<String, u64>::new();
        let mut strides = BTreeMap::<String, u64>::new();
        let mut positions = BTreeSet::new();
        for (index, binding) in bindings.iter().enumerate() {
            let expected_position = if index < self.inputs.len() {
                (ResolvedValueRole::Input, index as u32)
            } else {
                (
                    ResolvedValueRole::Output,
                    (index - self.inputs.len()) as u32,
                )
            };
            if (binding.role, binding.ordinal) != expected_position {
                return Err(invalid_operation(format!(
                    "operation `{}` bindings are not in canonical input/output ordinal order",
                    self.id
                )));
            }
            if !positions.insert((binding.role, binding.ordinal)) {
                return Err(invalid_operation(format!(
                    "operation `{}` contains duplicate ordinal bindings",
                    self.id
                )));
            }
            if let Some(previous) = bindings[..index]
                .iter()
                .find(|previous| previous.value_id == binding.value_id)
            {
                let repeated_readonly_input = previous.role == ResolvedValueRole::Input
                    && binding.role == ResolvedValueRole::Input
                    && previous.access == TensorAccess::Read
                    && binding.access == TensorAccess::Read
                    && previous.tensor == binding.tensor
                    && previous.storage == binding.storage
                    && previous.usage == binding.usage;
                if !repeated_readonly_input {
                    return Err(invalid_operation(format!(
                        "operation `{}` repeats a value outside identical read-only input slots",
                        self.id
                    )));
                }
            }
            let contract = match binding.role {
                ResolvedValueRole::Input => self.inputs.get(binding.ordinal as usize),
                ResolvedValueRole::Output => self.outputs.get(binding.ordinal as usize),
            }
            .ok_or_else(|| {
                invalid_operation(format!(
                    "operation `{}` binding ordinal is out of range",
                    self.id
                ))
            })?;
            if binding.access != contract.access || binding.alias != contract.alias {
                return Err(invalid_operation(format!(
                    "operation `{}` binding access or alias differs from its contract",
                    self.id
                )));
            }
            Self::validate_resolved_tensor(
                &self.id,
                contract,
                &binding.tensor,
                &mut dimensions,
                &mut strides,
            )?;
        }
        let inputs = &bindings[..self.inputs.len()];
        let outputs = &bindings[self.inputs.len()..];
        for (index, input) in inputs.iter().enumerate() {
            for previous in &inputs[..index] {
                if storage_overlaps(&input.storage, &previous.storage)
                    && (input.value_id != previous.value_id
                        || input.access != TensorAccess::Read
                        || previous.access != TensorAccess::Read)
                {
                    return Err(invalid_operation(format!(
                        "operation `{}` shares input storage between different or writable values",
                        self.id
                    )));
                }
            }
        }
        for (index, output) in outputs.iter().enumerate() {
            let aliased_inputs = inputs
                .iter()
                .enumerate()
                .filter(|(_, input)| storage_overlaps(&output.storage, &input.storage))
                .map(|(ordinal, _)| ordinal as u32)
                .collect::<Vec<_>>();
            match output.alias {
                AliasPolicy::NoAlias if !aliased_inputs.is_empty() => {
                    return Err(invalid_operation(format!(
                        "operation `{}` output {index} aliases despite a no-alias contract",
                        self.id
                    )));
                }
                AliasPolicy::MayAlias { tensor_index } => {
                    if aliased_inputs
                        .iter()
                        .any(|ordinal| *ordinal != tensor_index)
                        || (aliased_inputs.contains(&tensor_index)
                            && output.storage != inputs[tensor_index as usize].storage)
                    {
                        return Err(invalid_operation(format!(
                            "operation `{}` output {index} partially aliases or aliases the wrong input",
                            self.id
                        )));
                    }
                }
                AliasPolicy::MustAlias { tensor_index }
                    if aliased_inputs != [tensor_index]
                        || output.storage != inputs[tensor_index as usize].storage =>
                {
                    return Err(invalid_operation(format!(
                        "operation `{}` output {index} does not exactly alias its declared input",
                        self.id
                    )));
                }
                _ => {}
            }
            if outputs[..index]
                .iter()
                .any(|previous| storage_overlaps(&output.storage, &previous.storage))
            {
                return Err(invalid_operation(format!(
                    "operation `{}` output resources overlap",
                    self.id
                )));
            }
        }
        Ok(())
    }

    fn validate_resolved_tensor(
        operation_id: &OperationId,
        contract: &TensorContract,
        tensor: &ResolvedTensorSpec,
        dimensions: &mut BTreeMap<String, u64>,
        strides: &mut BTreeMap<String, u64>,
    ) -> Result<(), VNextError> {
        if tensor.dimensions.len() != contract.dimensions.len()
            || !contract.element_types.contains(&tensor.element_type)
        {
            return Err(invalid_operation(format!(
                "operation `{operation_id}` resolved tensor rank or element type is incompatible"
            )));
        }
        for (constraint, extent) in contract.dimensions.iter().zip(&tensor.dimensions) {
            let compatible = match constraint {
                DimensionConstraint::Exact(expected) => expected == extent,
                DimensionConstraint::Range { minimum, maximum } => {
                    minimum <= extent && extent <= maximum
                }
                DimensionConstraint::Symbol(symbol) => match dimensions.get(symbol) {
                    Some(expected) => expected == extent,
                    None => {
                        dimensions.insert(symbol.clone(), *extent);
                        true
                    }
                },
            };
            if !compatible {
                return Err(invalid_operation(format!(
                    "operation `{operation_id}` resolved tensor violates a dimension constraint"
                )));
            }
        }

        let mut matched_strides = None;
        let layout_matches = contract
            .layouts
            .iter()
            .any(|layout| match (layout, &tensor.layout) {
                (LayoutConstraint::Contiguous, ResolvedTensorLayout::Contiguous) => true,
                (
                    LayoutConstraint::Blocked {
                        block: expected_block,
                        axis_order: expected_axis_order,
                    },
                    ResolvedTensorLayout::Blocked {
                        block: actual_block,
                        axis_order: actual_axis_order,
                        ..
                    },
                ) => expected_block == actual_block && expected_axis_order == actual_axis_order,
                (
                    LayoutConstraint::Strided {
                        strides: constraints,
                    },
                    ResolvedTensorLayout::Strided { byte_strides },
                ) if constraints.len() == byte_strides.len() => {
                    let mut candidate = strides.clone();
                    let matches =
                        constraints
                            .iter()
                            .zip(byte_strides)
                            .all(|(constraint, actual)| match constraint {
                                StrideConstraint::ExactBytes(expected) => expected == actual,
                                StrideConstraint::Symbol(symbol) => match candidate.get(symbol) {
                                    Some(expected) => expected == actual,
                                    None => {
                                        candidate.insert(symbol.clone(), *actual);
                                        true
                                    }
                                },
                            });
                    if matches {
                        matched_strides = Some(candidate);
                    }
                    matches
                }
                _ => false,
            });
        if !layout_matches {
            return Err(invalid_operation(format!(
                "operation `{operation_id}` resolved tensor layout is incompatible"
            )));
        }
        if let Some(candidate) = matched_strides {
            *strides = candidate;
        }
        Ok(())
    }
}

fn storage_overlaps(left: &ResolvedValueStorage, right: &ResolvedValueStorage) -> bool {
    left.components.iter().any(|left| {
        right.components.iter().any(|right| {
            left.resource_id == right.resource_id
                && left.offset_bytes < right.offset_bytes.saturating_add(right.length_bytes)
                && right.offset_bytes < left.offset_bytes.saturating_add(left.length_bytes)
        })
    })
}

/// Object-safe semantic operation contract used while building a plan.
pub trait OperationContract: Send + Sync {
    fn descriptor(&self) -> &OperationDescriptor;

    fn validate_signature(
        &self,
        inputs: &[TensorContract],
        outputs: &[TensorContract],
    ) -> Result<(), VNextError>;
}
