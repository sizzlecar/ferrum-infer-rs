use serde::{Deserialize, Deserializer, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use super::{
    classify_device_error, AdmittedSequenceResources, AllocationLifetime, BatchInvocationId,
    BatchParticipantAuthority, BatchParticipantTokenRange, BatchStepId, BatchWorkShape,
    BufferUsage, CanonicalRational, CapabilityId, CompletionHandle, CompletionReaper,
    ContractVersion, DefinitelyNotSubmittedRetryAuthority,
    DefinitelyNotSubmittedWaveRetryAuthority, DeviceCommandBatch, DeviceId, DeviceRuntime,
    ExecutionIdentityEnvelope, ExecutionLane, ExecutionLaneId, IdentifiedFailure,
    IndeterminateSubmissionHandle, InvocationResourceLease, LaneSubmitOutcome, LeasedBufferView,
    LogicalAdmissionCoordinatorId, LogicalBackingBufferView, LogicalBackingSegmentBinding, NodeId,
    NodeWorkContract, OperationId, ParticipantNodeKey, PlanHash, PlanId,
    PreparedStepSubmissionNode, PreparedStepSubmissionWave, ProgramValueId, ProviderId,
    ProviderWorkspaceRequirement, QuantizationFormatId, ResolvedModelPlan, ResourceId,
    SemanticValue, SequenceBackingSnapshot, SequenceSessionEpoch, SequenceSessionFingerprint,
    StepParticipantFrameAssignment, StepResourceLease, TrustedActiveSequenceBinding,
    TrustedPlanRuntimeEvidence, UnvalidatedExecutionIdentityParts, VNextError, WeightFormatId,
    WeightId,
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
        Ok(Self {
            value_id,
            role,
            ordinal,
            tensor,
            access,
            alias,
            usage,
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

    pub fn storage(&self) -> &ResolvedValueStorage {
        &self.storage
    }
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

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct OperationProviderDescriptor {
    provider_id: ProviderId,
    operation_id: OperationId,
    operation_fingerprint: String,
    provider_implementation_fingerprint: String,
    version: ContractVersion,
    device_id: DeviceId,
    capabilities: BTreeSet<CapabilityId>,
    accepted_weight_formats: BTreeSet<WeightFormatId>,
    accepted_quantization_formats: BTreeSet<QuantizationFormatId>,
    dynamic_storage_bindings: Vec<ProviderStorageBindingRequirement>,
    resource_estimator_id: String,
    resource_estimator_version: ContractVersion,
    resource_estimator_implementation_fingerprint: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct OperationProviderDescriptorWire {
    provider_id: ProviderId,
    operation_id: OperationId,
    operation_fingerprint: String,
    provider_implementation_fingerprint: String,
    version: ContractVersion,
    device_id: DeviceId,
    capabilities: BTreeSet<CapabilityId>,
    accepted_weight_formats: BTreeSet<WeightFormatId>,
    accepted_quantization_formats: BTreeSet<QuantizationFormatId>,
    dynamic_storage_bindings: Vec<ProviderStorageBindingRequirement>,
    resource_estimator_id: String,
    resource_estimator_version: ContractVersion,
    resource_estimator_implementation_fingerprint: String,
}

impl<'de> Deserialize<'de> for OperationProviderDescriptor {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = OperationProviderDescriptorWire::deserialize(deserializer)?;
        let original_bindings = wire.dynamic_storage_bindings.clone();
        let descriptor = Self::new(
            wire.provider_id,
            wire.operation_id,
            wire.operation_fingerprint,
            wire.provider_implementation_fingerprint,
            wire.version,
            wire.device_id,
            wire.capabilities,
            wire.accepted_weight_formats,
            wire.accepted_quantization_formats,
            wire.dynamic_storage_bindings,
            wire.resource_estimator_id,
            wire.resource_estimator_version,
            wire.resource_estimator_implementation_fingerprint,
        )
        .map_err(serde::de::Error::custom)?;
        if descriptor.dynamic_storage_bindings != original_bindings {
            return Err(serde::de::Error::custom(
                "provider storage binding requirements are not canonical",
            ));
        }
        Ok(descriptor)
    }
}

impl OperationProviderDescriptor {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        provider_id: ProviderId,
        operation_id: OperationId,
        operation_fingerprint: impl Into<String>,
        provider_implementation_fingerprint: impl Into<String>,
        version: ContractVersion,
        device_id: DeviceId,
        capabilities: BTreeSet<CapabilityId>,
        accepted_weight_formats: BTreeSet<WeightFormatId>,
        accepted_quantization_formats: BTreeSet<QuantizationFormatId>,
        mut dynamic_storage_bindings: Vec<ProviderStorageBindingRequirement>,
        resource_estimator_id: impl Into<String>,
        resource_estimator_version: ContractVersion,
        resource_estimator_implementation_fingerprint: impl Into<String>,
    ) -> Result<Self, VNextError> {
        let operation_fingerprint = operation_fingerprint.into();
        let provider_implementation_fingerprint = provider_implementation_fingerprint.into();
        let resource_estimator_id = resource_estimator_id.into();
        let resource_estimator_implementation_fingerprint =
            resource_estimator_implementation_fingerprint.into();
        dynamic_storage_bindings.sort_by_key(|binding| binding.canonical_key());
        if version.major == 0
            || !canonical_sha256(&operation_fingerprint)
            || !canonical_sha256(&provider_implementation_fingerprint)
            || resource_estimator_id.is_empty()
            || resource_estimator_id.len() > 160
            || !resource_estimator_id.bytes().all(|byte| {
                byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-' | b':' | b'/')
            })
            || resource_estimator_version.major == 0
            || !canonical_sha256(&resource_estimator_implementation_fingerprint)
            || dynamic_storage_bindings.is_empty()
            || dynamic_storage_bindings
                .windows(2)
                .any(|pair| pair[0].canonical_key() == pair[1].canonical_key())
        {
            return Err(invalid_operation(
                "operation provider or resource-estimator identity is invalid",
            ));
        }
        Ok(Self {
            provider_id,
            operation_id,
            operation_fingerprint,
            provider_implementation_fingerprint,
            version,
            device_id,
            capabilities,
            accepted_weight_formats,
            accepted_quantization_formats,
            dynamic_storage_bindings,
            resource_estimator_id,
            resource_estimator_version,
            resource_estimator_implementation_fingerprint,
        })
    }

    pub fn provider_id(&self) -> &ProviderId {
        &self.provider_id
    }

    pub fn operation_id(&self) -> &OperationId {
        &self.operation_id
    }

    pub fn operation_fingerprint(&self) -> &str {
        &self.operation_fingerprint
    }

    pub fn provider_implementation_fingerprint(&self) -> &str {
        &self.provider_implementation_fingerprint
    }

    pub fn version(&self) -> ContractVersion {
        self.version
    }

    pub fn device_id(&self) -> &DeviceId {
        &self.device_id
    }

    pub fn capabilities(&self) -> &BTreeSet<CapabilityId> {
        &self.capabilities
    }

    pub fn accepted_weight_formats(&self) -> &BTreeSet<WeightFormatId> {
        &self.accepted_weight_formats
    }

    pub fn accepted_quantization_formats(&self) -> &BTreeSet<QuantizationFormatId> {
        &self.accepted_quantization_formats
    }

    pub fn dynamic_storage_bindings(&self) -> &[ProviderStorageBindingRequirement] {
        &self.dynamic_storage_bindings
    }

    pub fn dynamic_storage_for(
        &self,
        role: ResolvedValueRole,
        ordinal: u32,
    ) -> Option<&DynamicStorageRequirement> {
        self.dynamic_storage_bindings
            .binary_search_by_key(&(role, ordinal), |binding| binding.canonical_key())
            .ok()
            .map(|index| self.dynamic_storage_bindings[index].storage())
    }

    pub fn resource_estimator_id(&self) -> &str {
        &self.resource_estimator_id
    }

    pub const fn resource_estimator_version(&self) -> ContractVersion {
        self.resource_estimator_version
    }

    pub fn resource_estimator_implementation_fingerprint(&self) -> &str {
        &self.resource_estimator_implementation_fingerprint
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct EngineProviderDescriptor {
    provider_id: ProviderId,
    contract_version: ContractVersion,
    implementation_fingerprint: String,
    device_id: DeviceId,
    capabilities: BTreeSet<CapabilityId>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct EngineProviderDescriptorWire {
    provider_id: ProviderId,
    contract_version: ContractVersion,
    implementation_fingerprint: String,
    device_id: DeviceId,
    capabilities: BTreeSet<CapabilityId>,
}

impl<'de> Deserialize<'de> for EngineProviderDescriptor {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = EngineProviderDescriptorWire::deserialize(deserializer)?;
        Self::new(
            wire.provider_id,
            wire.contract_version,
            wire.implementation_fingerprint,
            wire.device_id,
            wire.capabilities,
        )
        .map_err(serde::de::Error::custom)
    }
}

impl EngineProviderDescriptor {
    pub fn new(
        provider_id: ProviderId,
        contract_version: ContractVersion,
        implementation_fingerprint: impl Into<String>,
        device_id: DeviceId,
        capabilities: BTreeSet<CapabilityId>,
    ) -> Result<Self, VNextError> {
        let implementation_fingerprint = implementation_fingerprint.into();
        if contract_version.major == 0 || !canonical_sha256(&implementation_fingerprint) {
            return Err(invalid_operation(
                "engine provider contract version or implementation fingerprint is invalid",
            ));
        }
        Ok(Self {
            provider_id,
            contract_version,
            implementation_fingerprint,
            device_id,
            capabilities,
        })
    }

    pub fn provider_id(&self) -> &ProviderId {
        &self.provider_id
    }

    pub const fn contract_version(&self) -> ContractVersion {
        self.contract_version
    }

    pub fn implementation_fingerprint(&self) -> &str {
        &self.implementation_fingerprint
    }

    pub fn device_id(&self) -> &DeviceId {
        &self.device_id
    }

    pub fn capabilities(&self) -> &BTreeSet<CapabilityId> {
        &self.capabilities
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ProviderCompatibilityRequest {
    operation_id: OperationId,
    required_version: ContractVersion,
    required_capabilities: BTreeSet<CapabilityId>,
    required_weight_formats: BTreeSet<WeightFormatId>,
    required_quantization_formats: BTreeSet<QuantizationFormatId>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ProviderCompatibilityRequestWire {
    operation_id: OperationId,
    required_version: ContractVersion,
    required_capabilities: BTreeSet<CapabilityId>,
    required_weight_formats: BTreeSet<WeightFormatId>,
    required_quantization_formats: BTreeSet<QuantizationFormatId>,
}

impl<'de> Deserialize<'de> for ProviderCompatibilityRequest {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = ProviderCompatibilityRequestWire::deserialize(deserializer)?;
        Self::new(
            wire.operation_id,
            wire.required_version,
            wire.required_capabilities,
            wire.required_weight_formats,
            wire.required_quantization_formats,
        )
        .map_err(serde::de::Error::custom)
    }
}

impl ProviderCompatibilityRequest {
    pub fn new(
        operation_id: OperationId,
        required_version: ContractVersion,
        required_capabilities: BTreeSet<CapabilityId>,
        required_weight_formats: BTreeSet<WeightFormatId>,
        required_quantization_formats: BTreeSet<QuantizationFormatId>,
    ) -> Result<Self, VNextError> {
        if required_version.major == 0 {
            return Err(invalid_operation(
                "provider compatibility request has a zero major version",
            ));
        }
        Ok(Self {
            operation_id,
            required_version,
            required_capabilities,
            required_weight_formats,
            required_quantization_formats,
        })
    }

    pub fn operation_id(&self) -> &OperationId {
        &self.operation_id
    }

    pub const fn required_version(&self) -> ContractVersion {
        self.required_version
    }

    pub fn required_capabilities(&self) -> &BTreeSet<CapabilityId> {
        &self.required_capabilities
    }

    pub fn required_weight_formats(&self) -> &BTreeSet<WeightFormatId> {
        &self.required_weight_formats
    }

    pub fn required_quantization_formats(&self) -> &BTreeSet<QuantizationFormatId> {
        &self.required_quantization_formats
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProviderCompatibilityRejectReason {
    OperationVersionMismatch {
        required: ContractVersion,
        available: ContractVersion,
    },
    ProviderVersionMismatch {
        required: ContractVersion,
        available: ContractVersion,
    },
    MissingCapabilities {
        capabilities: BTreeSet<CapabilityId>,
    },
    UnsupportedWeightFormats {
        formats: BTreeSet<WeightFormatId>,
    },
    UnsupportedQuantizationFormats {
        formats: BTreeSet<QuantizationFormatId>,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProviderCompatibilityRejection {
    pub provider_id: ProviderId,
    pub reasons: Vec<ProviderCompatibilityRejectReason>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ProviderCompatibilityReport {
    request: ProviderCompatibilityRequest,
    compatible_provider_ids: Vec<ProviderId>,
    rejected: Vec<ProviderCompatibilityRejection>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ProviderCompatibilityReportWire {
    request: ProviderCompatibilityRequest,
    compatible_provider_ids: Vec<ProviderId>,
    rejected: Vec<ProviderCompatibilityRejection>,
}

impl<'de> Deserialize<'de> for ProviderCompatibilityReport {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = ProviderCompatibilityReportWire::deserialize(deserializer)?;
        let report = Self {
            request: wire.request,
            compatible_provider_ids: wire.compatible_provider_ids,
            rejected: wire.rejected,
        };
        report.validate_shape().map_err(serde::de::Error::custom)?;
        Ok(report)
    }
}

impl ProviderCompatibilityReport {
    fn validate_shape(&self) -> Result<(), VNextError> {
        let compatible = self.compatible_provider_ids.iter().collect::<BTreeSet<_>>();
        let rejected = self
            .rejected
            .iter()
            .map(|rejection| &rejection.provider_id)
            .collect::<BTreeSet<_>>();
        if compatible.len() != self.compatible_provider_ids.len()
            || rejected.len() != self.rejected.len()
            || (compatible.is_empty() && rejected.is_empty())
            || !compatible.is_disjoint(&rejected)
            || self
                .rejected
                .iter()
                .any(|rejection| rejection.reasons.is_empty())
            || self
                .compatible_provider_ids
                .windows(2)
                .any(|pair| pair[0] >= pair[1])
            || self
                .rejected
                .windows(2)
                .any(|pair| pair[0].provider_id >= pair[1].provider_id)
        {
            return Err(invalid_operation(
                "provider compatibility report is duplicate, overlapping, empty, or non-canonical",
            ));
        }
        Ok(())
    }

    pub fn request(&self) -> &ProviderCompatibilityRequest {
        &self.request
    }

    pub fn compatible_provider_ids(&self) -> &[ProviderId] {
        &self.compatible_provider_ids
    }

    pub fn rejected(&self) -> &[ProviderCompatibilityRejection] {
        &self.rejected
    }

    pub fn require_compatible(&self, device_id: &DeviceId) -> Result<(), VNextError> {
        if self.compatible_provider_ids.is_empty() {
            return Err(VNextError::UnsupportedOperation {
                node_id: None,
                operation_id: self.request.operation_id.to_string(),
                device_id: device_id.to_string(),
                reason: "all providers were rejected; inspect the typed compatibility report"
                    .to_owned(),
            });
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct OperationFailure {
    identity: ExecutionIdentityEnvelope,
    phase: ProfilePhase,
    code: String,
    message: String,
    retryable: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnvalidatedOperationFailure {
    identity: UnvalidatedExecutionIdentityParts,
    phase: ProfilePhase,
    code: String,
    message: String,
    retryable: bool,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct UnvalidatedOperationFailureWire {
    identity: UnvalidatedExecutionIdentityParts,
    phase: ProfilePhase,
    code: String,
    message: String,
    retryable: bool,
}

impl From<UnvalidatedOperationFailureWire> for UnvalidatedOperationFailure {
    fn from(wire: UnvalidatedOperationFailureWire) -> Self {
        Self {
            identity: wire.identity,
            phase: wire.phase,
            code: wire.code,
            message: wire.message,
            retryable: wire.retryable,
        }
    }
}

impl UnvalidatedOperationFailure {
    pub fn revalidate(
        self,
        expected_identity: &ExecutionIdentityEnvelope,
        expected_phase: ProfilePhase,
    ) -> Result<OperationFailure, VNextError> {
        let identity = ExecutionIdentityEnvelope::new(self.identity.into())?;
        if &identity != expected_identity || self.phase != expected_phase {
            return Err(invalid_operation(
                "serialized operation failure differs from the expected execution context",
            ));
        }
        OperationFailure::new(
            identity,
            self.phase,
            self.code,
            self.message,
            self.retryable,
        )
    }
}

impl OperationFailure {
    pub fn new(
        identity: ExecutionIdentityEnvelope,
        phase: ProfilePhase,
        code: impl Into<String>,
        message: impl Into<String>,
        retryable: bool,
    ) -> Result<Self, VNextError> {
        let code = code.into();
        let message = message.into();
        let parts = identity.parts();
        if parts.frame_id.is_none()
            || parts.node_invocation_id.is_none()
            || parts.node_id.is_none()
            || parts.operation_id.is_none()
            || parts.provider_id.is_none()
            || parts.device_id.is_none()
            || parts.plan_id.is_none()
            || parts.plan_hash.is_none()
            || parts.transaction_id.is_none()
            || parts.resource_pool_id.is_none()
            || parts.resource_pool_identity_fingerprint.is_none()
            || parts.provisioning_run_id.is_none()
            || parts.provisioning_request_id.is_none()
            || parts.active_sequence_slot.is_none()
            || parts.admission_generation.is_none()
            || parts.activation_epoch.is_none()
            || parts.runtime_implementation_fingerprint.is_none()
            || parts.active_sequence_fingerprint.is_none()
            || parts.completed_sequence_fingerprint.is_some()
            || parts.aborted_sequence_fingerprint.is_some()
            || parts.resource_id.is_some()
            || parts.resource_generation.is_some()
            || parts.resource_batch_fingerprint.is_some()
            || code.trim().is_empty()
            || message.trim().is_empty()
            || code.len() > 64
            || message
                .bytes()
                .any(|byte| byte.is_ascii_control() && !matches!(byte, b'\n' | b'\t'))
            || message.len() > 4096
            || !code
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-'))
        {
            return Err(invalid_operation(
                "operation failure requires complete execution identity, code, and message",
            ));
        }
        Ok(Self {
            identity,
            phase,
            code,
            message,
            retryable,
        })
    }

    pub fn identity(&self) -> &ExecutionIdentityEnvelope {
        &self.identity
    }

    pub const fn phase(&self) -> ProfilePhase {
        self.phase
    }

    pub fn code(&self) -> &str {
        &self.code
    }

    pub fn message(&self) -> &str {
        &self.message
    }

    pub const fn retryable(&self) -> bool {
        self.retryable
    }

    pub fn decode_untrusted(bytes: &[u8]) -> Result<UnvalidatedOperationFailure, VNextError> {
        if bytes.len() > MAX_OPERATION_FAILURE_WIRE_BYTES {
            return Err(VNextError::Serialization {
                context: "decode untrusted operation failure",
                message: format!(
                    "operation failure wire size {} exceeds limit {}",
                    bytes.len(),
                    MAX_OPERATION_FAILURE_WIRE_BYTES
                ),
            });
        }
        serde_json::from_slice::<UnvalidatedOperationFailureWire>(bytes)
            .map(UnvalidatedOperationFailure::from)
            .map_err(|error| VNextError::Serialization {
                context: "decode untrusted operation failure",
                message: error.to_string(),
            })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OperationBufferStorageKind {
    StaticContiguous,
    DynamicContiguous,
    DynamicPaged,
}

enum OperationBufferSource<'a, B> {
    Static(LeasedBufferView<'a, B>),
    Backing(LogicalBackingBufferView<'a, B>),
}

#[derive(Clone, Copy)]
enum OperationRegionSource<'a, B> {
    Contiguous {
        buffer: &'a B,
        physical_base_offset_bytes: u64,
    },
    Paged {
        bindings: &'a [LogicalBackingSegmentBinding<B>],
    },
}

/// A checked logical range translated to physical device-buffer regions.
/// Dynamic buffers never expose an arena buffer without its physical offsets.
pub struct OperationBufferRegions<'a, B> {
    storage_kind: OperationBufferStorageKind,
    logical_offset_bytes: u64,
    logical_length_bytes: u64,
    source: OperationRegionSource<'a, B>,
}

impl<'a, B> OperationBufferRegions<'a, B> {
    pub const fn storage_kind(&self) -> OperationBufferStorageKind {
        self.storage_kind
    }

    pub const fn logical_offset_bytes(&self) -> u64 {
        self.logical_offset_bytes
    }

    pub const fn logical_length_bytes(&self) -> u64 {
        self.logical_length_bytes
    }

    pub fn iter(&self) -> OperationBufferRegionIter<'a, B> {
        let logical_end_bytes = self
            .logical_offset_bytes
            .checked_add(self.logical_length_bytes)
            .expect("validated operation logical range does not overflow");
        match self.source {
            OperationRegionSource::Contiguous {
                buffer,
                physical_base_offset_bytes,
            } => OperationBufferRegionIter {
                state: OperationBufferRegionIterState::Contiguous(Some(OperationPhysicalRegion {
                    buffer,
                    logical_offset_bytes: self.logical_offset_bytes,
                    physical_offset_bytes: physical_base_offset_bytes
                        .checked_add(self.logical_offset_bytes)
                        .expect("validated contiguous physical range does not overflow"),
                    length_bytes: self.logical_length_bytes,
                })),
            },
            OperationRegionSource::Paged { bindings } => OperationBufferRegionIter {
                state: OperationBufferRegionIterState::Paged {
                    bindings,
                    requested_start_bytes: self.logical_offset_bytes,
                    requested_end_bytes: logical_end_bytes,
                    next_segment: 0,
                    next_segment_logical_offset_bytes: 0,
                },
            },
        }
    }
}

/// One indivisible physical region. The buffer reference is intentionally
/// returned only together with the physical byte range.
pub struct OperationPhysicalRegion<'a, B> {
    buffer: &'a B,
    logical_offset_bytes: u64,
    physical_offset_bytes: u64,
    length_bytes: u64,
}

impl<'a, B> OperationPhysicalRegion<'a, B> {
    pub const fn logical_offset_bytes(&self) -> u64 {
        self.logical_offset_bytes
    }

    pub const fn length_bytes(&self) -> u64 {
        self.length_bytes
    }

    pub fn buffer_and_physical_range(&self) -> (&'a B, std::ops::Range<u64>) {
        (
            self.buffer,
            self.physical_offset_bytes
                ..self
                    .physical_offset_bytes
                    .checked_add(self.length_bytes)
                    .expect("validated physical region does not overflow"),
        )
    }
}

pub struct OperationBufferRegionIter<'a, B> {
    state: OperationBufferRegionIterState<'a, B>,
}

enum OperationBufferRegionIterState<'a, B> {
    Contiguous(Option<OperationPhysicalRegion<'a, B>>),
    Paged {
        bindings: &'a [LogicalBackingSegmentBinding<B>],
        requested_start_bytes: u64,
        requested_end_bytes: u64,
        next_segment: usize,
        next_segment_logical_offset_bytes: u64,
    },
}

impl<'a, B> Iterator for OperationBufferRegionIter<'a, B> {
    type Item = OperationPhysicalRegion<'a, B>;

    fn next(&mut self) -> Option<Self::Item> {
        match &mut self.state {
            OperationBufferRegionIterState::Contiguous(region) => region.take(),
            OperationBufferRegionIterState::Paged {
                bindings,
                requested_start_bytes,
                requested_end_bytes,
                next_segment,
                next_segment_logical_offset_bytes,
            } => {
                while let Some(binding) = bindings.get(*next_segment) {
                    *next_segment += 1;
                    let (segment_logical_end, region) = translate_paged_segment(
                        binding.buffer(),
                        binding.segment().offset_bytes(),
                        binding.segment().length_bytes(),
                        *next_segment_logical_offset_bytes,
                        *requested_start_bytes,
                        *requested_end_bytes,
                    );
                    *next_segment_logical_offset_bytes = segment_logical_end;
                    if region.is_some() {
                        return region;
                    }
                }
                None
            }
        }
    }
}

fn translate_paged_segment<'a, B>(
    buffer: &'a B,
    physical_offset_bytes: u64,
    length_bytes: u64,
    logical_start_bytes: u64,
    requested_start_bytes: u64,
    requested_end_bytes: u64,
) -> (u64, Option<OperationPhysicalRegion<'a, B>>) {
    let logical_end_bytes = logical_start_bytes
        .checked_add(length_bytes)
        .expect("validated backing segments do not overflow");
    let translated_start = logical_start_bytes.max(requested_start_bytes);
    let translated_end = logical_end_bytes.min(requested_end_bytes);
    let region = (translated_start < translated_end).then(|| OperationPhysicalRegion {
        buffer,
        logical_offset_bytes: translated_start,
        physical_offset_bytes: physical_offset_bytes
            .checked_add(translated_start - logical_start_bytes)
            .expect("validated paged physical range does not overflow"),
        length_bytes: translated_end - translated_start,
    });
    (logical_end_bytes, region)
}

const fn operation_storage_kind(view: DynamicStorageView) -> OperationBufferStorageKind {
    match view {
        DynamicStorageView::Contiguous => OperationBufferStorageKind::DynamicContiguous,
        DynamicStorageView::PagedRegions { .. } => OperationBufferStorageKind::DynamicPaged,
    }
}

fn validate_dynamic_binding_layout(
    storage_kind: OperationBufferStorageKind,
    logical_size_bytes: u64,
    mut binding_lengths: impl ExactSizeIterator<Item = u64>,
) -> Result<(), VNextError> {
    let binding_count = binding_lengths.len();
    if storage_kind == OperationBufferStorageKind::StaticContiguous {
        return Err(invalid_operation(
            "dynamic backing cannot claim static storage kind",
        ));
    }
    if binding_count == 0 {
        return Err(invalid_operation(
            "dynamic backing has no physical segment binding",
        ));
    }
    if storage_kind == OperationBufferStorageKind::DynamicContiguous && binding_count != 1 {
        return Err(invalid_operation(
            "contiguous dynamic storage requires one physical segment binding",
        ));
    }
    let covered = binding_lengths.try_fold(0_u64, |total, length_bytes| {
        total
            .checked_add(length_bytes)
            .ok_or_else(|| invalid_operation("backing segment coverage overflows u64"))
    })?;
    if covered != logical_size_bytes {
        return Err(invalid_operation(
            "dynamic backing segments do not exactly cover the logical resource",
        ));
    }
    Ok(())
}

pub struct OperationBufferView<'a, B> {
    descriptor: super::BufferDescriptor,
    source: OperationBufferSource<'a, B>,
}

impl<'a, B> OperationBufferView<'a, B> {
    pub fn resource_id(&self) -> &ResourceId {
        &self.descriptor.resource_id
    }

    pub fn descriptor(&self) -> &super::BufferDescriptor {
        &self.descriptor
    }

    pub fn storage_kind(&self) -> OperationBufferStorageKind {
        match &self.source {
            OperationBufferSource::Static(_) => OperationBufferStorageKind::StaticContiguous,
            OperationBufferSource::Backing(view) => {
                operation_storage_kind(view.storage_profile().view())
            }
        }
    }

    pub fn translate(
        &self,
        logical_offset_bytes: u64,
        logical_length_bytes: u64,
    ) -> Result<OperationBufferRegions<'_, B>, VNextError> {
        let logical_end_bytes = logical_offset_bytes
            .checked_add(logical_length_bytes)
            .ok_or_else(|| invalid_operation("operation logical buffer range overflows u64"))?;
        if logical_length_bytes == 0 || logical_end_bytes > self.descriptor.size_bytes {
            return Err(invalid_operation(
                "operation logical buffer range is empty or outside its resource",
            ));
        }
        match &self.source {
            OperationBufferSource::Static(view) => Ok(OperationBufferRegions {
                storage_kind: OperationBufferStorageKind::StaticContiguous,
                logical_offset_bytes,
                logical_length_bytes,
                source: OperationRegionSource::Contiguous {
                    buffer: view.buffer(),
                    physical_base_offset_bytes: 0,
                },
            }),
            OperationBufferSource::Backing(view) => {
                let bindings = view.segment_bindings();
                if bindings.len() != view.committed_evidence_segments().count()
                    || bindings.iter().zip(view.committed_evidence_segments()).any(
                        |(binding, segment)| {
                            binding.segment() != segment || binding.chunk() != segment.chunk()
                        },
                    )
                {
                    return Err(invalid_operation(
                        "dynamic backing bindings differ from committed segment evidence",
                    ));
                }
                let storage_kind = operation_storage_kind(view.storage_profile().view());
                validate_dynamic_binding_layout(
                    storage_kind,
                    self.descriptor.size_bytes,
                    bindings
                        .iter()
                        .map(|binding| binding.segment().length_bytes()),
                )?;
                let source = match storage_kind {
                    OperationBufferStorageKind::DynamicContiguous => {
                        let binding = &bindings[0];
                        OperationRegionSource::Contiguous {
                            buffer: binding.buffer(),
                            physical_base_offset_bytes: binding.segment().offset_bytes(),
                        }
                    }
                    OperationBufferStorageKind::DynamicPaged => {
                        OperationRegionSource::Paged { bindings }
                    }
                    OperationBufferStorageKind::StaticContiguous => unreachable!(
                        "dynamic storage kind was validated before region construction"
                    ),
                };
                Ok(OperationBufferRegions {
                    storage_kind,
                    logical_offset_bytes,
                    logical_length_bytes,
                    source,
                })
            }
        }
    }
}

#[cfg(test)]
mod operation_buffer_region_tests {
    use super::*;

    #[test]
    fn paged_translation_uses_each_chunks_exact_buffer() {
        struct MockBinding<'a> {
            buffer: &'a u8,
            physical_offset_bytes: u64,
            length_bytes: u64,
        }

        let first_buffer = 7_u8;
        let second_buffer = 11_u8;
        let bindings = [
            MockBinding {
                buffer: &first_buffer,
                physical_offset_bytes: 64,
                length_bytes: 8,
            },
            MockBinding {
                buffer: &second_buffer,
                physical_offset_bytes: 200,
                length_bytes: 12,
            },
        ];
        let mut next_logical_offset = 0;
        let translated = bindings
            .iter()
            .filter_map(|binding| {
                let (logical_end, region) = translate_paged_segment(
                    binding.buffer,
                    binding.physical_offset_bytes,
                    binding.length_bytes,
                    next_logical_offset,
                    6,
                    16,
                );
                next_logical_offset = logical_end;
                region
            })
            .collect::<Vec<_>>();

        assert_eq!(translated.len(), 2);
        let (first, first_physical) = translated[0].buffer_and_physical_range();
        assert!(std::ptr::eq(first, &first_buffer));
        assert_eq!(translated[0].logical_offset_bytes(), 6);
        assert_eq!(first_physical, 70..72);
        let (second, second_physical) = translated[1].buffer_and_physical_range();
        assert!(std::ptr::eq(second, &second_buffer));
        assert_eq!(translated[1].logical_offset_bytes(), 8);
        assert_eq!(second_physical, 200..208);
    }

    #[test]
    fn contiguous_layout_rejects_cross_chunk_bindings() {
        let first_buffer = 7_u8;
        let second_buffer = 11_u8;
        let bindings = [(&first_buffer, 8_u64), (&second_buffer, 12_u64)];

        let error = validate_dynamic_binding_layout(
            OperationBufferStorageKind::DynamicContiguous,
            20,
            bindings.iter().map(|(_, length_bytes)| *length_bytes),
        )
        .unwrap_err();

        assert!(error
            .to_string()
            .contains("contiguous dynamic storage requires one physical segment binding"));
    }

    #[test]
    fn contiguous_translation_applies_physical_base_offset() {
        let buffer = 9_u8;
        let regions = OperationBufferRegions {
            storage_kind: OperationBufferStorageKind::DynamicContiguous,
            logical_offset_bytes: 16,
            logical_length_bytes: 32,
            source: OperationRegionSource::Contiguous {
                buffer: &buffer,
                physical_base_offset_bytes: 4096,
            },
        };

        let translated = regions.iter().collect::<Vec<_>>();
        assert_eq!(translated.len(), 1);
        let (actual, physical) = translated[0].buffer_and_physical_range();
        assert_eq!(*actual, buffer);
        assert_eq!(translated[0].logical_offset_bytes(), 16);
        assert_eq!(physical, 4112..4144);
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct BatchOperationParticipantIdentity {
    participant_index: u32,
    node_key: ParticipantNodeKey,
    identity: ExecutionIdentityEnvelope,
}

impl BatchOperationParticipantIdentity {
    pub const fn participant_index(&self) -> u32 {
        self.participant_index
    }

    pub fn node_key(&self) -> &ParticipantNodeKey {
        &self.node_key
    }

    pub fn identity(&self) -> &ExecutionIdentityEnvelope {
        &self.identity
    }
}

/// One immutable-plan node inside a physical command batch. Participant
/// identities stay node-local even when several nodes share one submission.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct BatchOperationNodeIdentity {
    node_index: u32,
    node_id: NodeId,
    operation_id: OperationId,
    provider_id: ProviderId,
    work_shape_fingerprint: String,
    participants: Vec<BatchOperationParticipantIdentity>,
    fingerprint: String,
}

impl BatchOperationNodeIdentity {
    fn from_validated(
        node_index: u32,
        node_id: NodeId,
        operation_id: OperationId,
        provider_id: ProviderId,
        work_shape_fingerprint: String,
        participants: Vec<BatchOperationParticipantIdentity>,
    ) -> Result<Self, VNextError> {
        let participant_start = participants
            .first()
            .map(BatchOperationParticipantIdentity::participant_index);
        if participants.is_empty()
            || participants.iter().enumerate().any(|(index, participant)| {
                participant_start.and_then(|start| start.checked_add(index as u32))
                    != Some(participant.participant_index)
                    || participant.node_key.node_id() != &node_id
                    || participant.identity.parts().frame_id
                        != Some(participant.node_key.frame_id())
                    || participant.identity.parts().node_id.as_ref() != Some(&node_id)
                    || participant.identity.parts().operation_id.as_ref() != Some(&operation_id)
                    || participant.identity.parts().provider_id.as_ref() != Some(&provider_id)
            })
            || participants
                .windows(2)
                .any(|pair| pair[0].node_key >= pair[1].node_key)
            || !canonical_sha256(&work_shape_fingerprint)
        {
            return Err(invalid_operation(
                "batch node identity is empty, non-canonical, or differs from its participant projections",
            ));
        }
        #[derive(Serialize)]
        struct FingerprintInput<'a> {
            domain: &'static str,
            node_index: u32,
            node_id: &'a NodeId,
            operation_id: &'a OperationId,
            provider_id: &'a ProviderId,
            work_shape_fingerprint: &'a str,
            participants: &'a [BatchOperationParticipantIdentity],
        }
        let fingerprint = format!(
            "{:x}",
            Sha256::digest(
                serde_json::to_vec(&FingerprintInput {
                    domain: "ferrum.runtime-vnext.batch-operation-node-identity.v1",
                    node_index,
                    node_id: &node_id,
                    operation_id: &operation_id,
                    provider_id: &provider_id,
                    work_shape_fingerprint: &work_shape_fingerprint,
                    participants: &participants,
                })
                .map_err(|error| {
                    invalid_operation(format!("batch node identity encode failed: {error}"))
                })?
            )
        );
        Ok(Self {
            node_index,
            node_id,
            operation_id,
            provider_id,
            work_shape_fingerprint,
            participants,
            fingerprint,
        })
    }

    pub const fn node_index(&self) -> u32 {
        self.node_index
    }

    pub fn node_id(&self) -> &NodeId {
        &self.node_id
    }

    pub fn operation_id(&self) -> &OperationId {
        &self.operation_id
    }

    pub fn provider_id(&self) -> &ProviderId {
        &self.provider_id
    }

    pub fn work_shape_fingerprint(&self) -> &str {
        &self.work_shape_fingerprint
    }

    pub fn participants(&self) -> &[BatchOperationParticipantIdentity] {
        &self.participants
    }

    pub fn fingerprint(&self) -> &str {
        &self.fingerprint
    }

    fn contains_identity(&self, identity: &ExecutionIdentityEnvelope) -> bool {
        self.participants
            .iter()
            .any(|participant| participant.identity() == identity)
    }
}

/// One physical command-batch attempt identity. It may contain one operation
/// or the entire immutable-plan wave, but it always maps to one submit/fence.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct BatchOperationIdentity {
    batch_step_id: BatchStepId,
    batch_invocation_id: BatchInvocationId,
    plan_id: PlanId,
    plan_hash: PlanHash,
    device_id: DeviceId,
    runtime_implementation_fingerprint: String,
    lane_id: ExecutionLaneId,
    claimed_backing_fingerprint: String,
    nodes: Vec<BatchOperationNodeIdentity>,
    participants: Vec<BatchOperationParticipantIdentity>,
    fingerprint: String,
}

impl BatchOperationIdentity {
    #[allow(clippy::too_many_arguments)]
    fn from_validated(
        batch_step_id: BatchStepId,
        batch_invocation_id: BatchInvocationId,
        plan_id: PlanId,
        plan_hash: PlanHash,
        device_id: DeviceId,
        runtime_implementation_fingerprint: String,
        lane_id: ExecutionLaneId,
        claimed_backing_fingerprint: String,
        nodes: Vec<BatchOperationNodeIdentity>,
    ) -> Result<Self, VNextError> {
        if nodes.is_empty()
            || nodes.iter().enumerate().any(|(index, node)| {
                node.node_index as usize != index
                    || node.participants.iter().any(|participant| {
                        participant.identity.parts().plan_id.as_ref() != Some(&plan_id)
                            || participant.identity.parts().plan_hash.as_ref() != Some(&plan_hash)
                            || participant.identity.parts().device_id.as_ref() != Some(&device_id)
                            || participant
                                .identity
                                .parts()
                                .runtime_implementation_fingerprint
                                .as_deref()
                                != Some(runtime_implementation_fingerprint.as_str())
                    })
            })
            || nodes
                .iter()
                .map(BatchOperationNodeIdentity::node_id)
                .collect::<BTreeSet<_>>()
                .len()
                != nodes.len()
            || !canonical_sha256(&runtime_implementation_fingerprint)
            || !canonical_sha256(&claimed_backing_fingerprint)
        {
            return Err(invalid_operation(
                "physical batch identity is empty, non-canonical, or differs from its plan/runtime projections",
            ));
        }
        let participants = nodes
            .iter()
            .flat_map(|node| node.participants.iter().cloned())
            .collect::<Vec<_>>();
        if participants
            .iter()
            .enumerate()
            .any(|(index, participant)| participant.participant_index as usize != index)
        {
            return Err(invalid_operation(
                "physical batch participant indices are not globally contiguous",
            ));
        }
        #[derive(Serialize)]
        struct FingerprintInput<'a> {
            domain: &'static str,
            batch_step_id: BatchStepId,
            batch_invocation_id: BatchInvocationId,
            plan_id: &'a PlanId,
            plan_hash: &'a PlanHash,
            device_id: &'a DeviceId,
            runtime_implementation_fingerprint: &'a str,
            lane_id: ExecutionLaneId,
            claimed_backing_fingerprint: &'a str,
            nodes: &'a [BatchOperationNodeIdentity],
        }
        let fingerprint = format!(
            "{:x}",
            Sha256::digest(
                serde_json::to_vec(&FingerprintInput {
                    domain: "ferrum.runtime-vnext.physical-command-batch-identity.v1",
                    batch_step_id,
                    batch_invocation_id,
                    plan_id: &plan_id,
                    plan_hash: &plan_hash,
                    device_id: &device_id,
                    runtime_implementation_fingerprint: &runtime_implementation_fingerprint,
                    lane_id,
                    claimed_backing_fingerprint: &claimed_backing_fingerprint,
                    nodes: &nodes,
                })
                .map_err(|error| {
                    invalid_operation(format!("physical batch identity encode failed: {error}"))
                })?,
            )
        );
        Ok(Self {
            batch_step_id,
            batch_invocation_id,
            plan_id,
            plan_hash,
            device_id,
            runtime_implementation_fingerprint,
            lane_id,
            claimed_backing_fingerprint,
            nodes,
            participants,
            fingerprint,
        })
    }

    pub const fn batch_step_id(&self) -> BatchStepId {
        self.batch_step_id
    }

    pub const fn batch_invocation_id(&self) -> BatchInvocationId {
        self.batch_invocation_id
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

    pub fn runtime_implementation_fingerprint(&self) -> &str {
        &self.runtime_implementation_fingerprint
    }

    pub const fn lane_id(&self) -> ExecutionLaneId {
        self.lane_id
    }

    pub fn claimed_backing_fingerprint(&self) -> &str {
        &self.claimed_backing_fingerprint
    }

    pub fn nodes(&self) -> &[BatchOperationNodeIdentity] {
        &self.nodes
    }

    pub fn single_node(&self) -> Option<&BatchOperationNodeIdentity> {
        (self.nodes.len() == 1).then(|| &self.nodes[0])
    }

    pub fn participants(&self) -> &[BatchOperationParticipantIdentity] {
        &self.participants
    }

    pub fn fingerprint(&self) -> &str {
        &self.fingerprint
    }

    fn contains_identity(&self, identity: &ExecutionIdentityEnvelope) -> bool {
        self.participants
            .iter()
            .any(|participant| participant.identity() == identity)
    }
}

enum OperationInvocationResources<'a, R: DeviceRuntime> {
    Invocation(&'a InvocationResourceLease<R>),
    Wave {
        wave: &'a PreparedStepSubmissionWave<R>,
        node_index: usize,
    },
}

impl<R: DeviceRuntime> Copy for OperationInvocationResources<'_, R> {}

impl<R: DeviceRuntime> Clone for OperationInvocationResources<'_, R> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<'a, R: DeviceRuntime> OperationInvocationResources<'a, R> {
    fn wave_node(self) -> Result<&'a PreparedStepSubmissionNode<R>, VNextError> {
        match self {
            Self::Wave { wave, node_index } => wave
                .nodes()
                .get(node_index)
                .ok_or_else(|| invalid_operation("submission wave node index is out of bounds")),
            Self::Invocation(_) => Err(invalid_operation(
                "single-operation resources do not contain a wave node",
            )),
        }
    }

    fn node_id(self) -> Result<&'a NodeId, VNextError> {
        match self {
            Self::Invocation(invocation) => Ok(invocation.node_id()),
            Self::Wave { .. } => Ok(self.wave_node()?.node_id()),
        }
    }

    fn participant_count(self) -> Result<usize, VNextError> {
        match self {
            Self::Invocation(invocation) => usize::try_from(invocation.participant_count())
                .map_err(|_| invalid_operation("operation participant count exceeds usize")),
            Self::Wave { .. } => usize::try_from(self.wave_node()?.participant_count())
                .map_err(|_| invalid_operation("wave participant count exceeds usize")),
        }
    }

    fn prepared_participant_count(self) -> Result<usize, VNextError> {
        match self {
            Self::Invocation(invocation) => {
                usize::try_from(invocation.prepared_participant_count()).map_err(|_| {
                    invalid_operation("prepared operation participant count exceeds usize")
                })
            }
            Self::Wave { .. } => Ok(self.wave_node()?.participant_session_identities().len()),
        }
    }

    fn participant(
        self,
        index: usize,
    ) -> Result<&'a Arc<AdmittedSequenceResources<R>>, VNextError> {
        match self {
            Self::Invocation(invocation) => invocation
                .participants()
                .nth(index)
                .ok_or_else(|| invalid_operation("operation participant index is out of range")),
            Self::Wave { .. } => self
                .wave_node()?
                .participants()
                .nth(index)
                .ok_or_else(|| invalid_operation("wave participant index is out of range")),
        }
    }

    fn participant_backing_snapshot(
        self,
        index: usize,
    ) -> Result<&'a Arc<SequenceBackingSnapshot<R>>, VNextError> {
        let participant = self.participant(index)?;
        self.step_resources()
            .participant_backing_snapshot(BatchParticipantAuthority::new(
                participant.sequence_authority(),
                participant.request_authority(),
            ))
    }

    fn participant_backing_view(
        self,
        index: usize,
        resource_id: &ResourceId,
    ) -> Result<LogicalBackingBufferView<'a, R::Buffer>, VNextError> {
        let participant = self.participant(index)?;
        self.step_resources().participant_backing_view(
            BatchParticipantAuthority::new(
                participant.sequence_authority(),
                participant.request_authority(),
            ),
            resource_id,
        )
    }

    fn participant_frames(self) -> Result<&'a [StepParticipantFrameAssignment], VNextError> {
        match self {
            Self::Invocation(invocation) => Ok(invocation.participant_frames()),
            Self::Wave { .. } => Ok(self.wave_node()?.participant_frames()),
        }
    }

    fn participant_session_identity(
        self,
        index: usize,
    ) -> Result<(SequenceSessionEpoch, &'a SequenceSessionFingerprint), VNextError> {
        match self {
            Self::Invocation(invocation) => invocation
                .participant_session_identities()
                .nth(index)
                .ok_or_else(|| invalid_operation("operation participant session is missing")),
            Self::Wave { .. } => self
                .wave_node()?
                .participant_session_identities()
                .nth(index)
                .ok_or_else(|| invalid_operation("wave participant session is missing")),
        }
    }

    fn participant_node_keys(self) -> Result<Vec<ParticipantNodeKey>, VNextError> {
        match self {
            Self::Invocation(invocation) => Ok(invocation.participant_node_keys()),
            Self::Wave { .. } => Ok(self.wave_node()?.participant_node_keys()),
        }
    }

    fn batch_step_id(self) -> BatchStepId {
        match self {
            Self::Invocation(invocation) => invocation.batch_step_id(),
            Self::Wave { wave, .. } => wave.batch_step_id(),
        }
    }

    fn batch_invocation_id(self) -> BatchInvocationId {
        match self {
            Self::Invocation(invocation) => invocation.batch_invocation_id(),
            Self::Wave { wave, .. } => wave.batch_invocation_id(),
        }
    }

    fn coordinator_id(self) -> Result<LogicalAdmissionCoordinatorId, VNextError> {
        Ok(self.participant(0)?.coordinator_id())
    }

    fn work_shape(self) -> Result<&'a BatchWorkShape, VNextError> {
        match self {
            Self::Invocation(invocation) => Ok(invocation.work_shape()),
            Self::Wave { .. } => Ok(self.wave_node()?.work_shape()),
        }
    }

    fn step_resources(self) -> &'a Arc<StepResourceLease<R>> {
        match self {
            Self::Invocation(invocation) => invocation.step_resources(),
            Self::Wave { wave, .. } => wave.step_resources(),
        }
    }

    fn runtime(self) -> &'a Arc<R> {
        match self {
            Self::Invocation(invocation) => invocation.runtime(),
            Self::Wave { wave, .. } => wave.runtime(),
        }
    }

    fn plan_evidence(self) -> Result<TrustedPlanRuntimeEvidence, VNextError> {
        match self {
            Self::Invocation(invocation) => Ok(invocation.plan_evidence()),
            Self::Wave { .. } => Ok(self.wave_node()?.plan_evidence()),
        }
    }

    fn backing_fingerprint(self) -> &'a str {
        match self {
            Self::Invocation(invocation) => invocation.claimed_backing().fingerprint(),
            Self::Wave { wave, .. } => wave.fingerprint(),
        }
    }

    fn backing_view(
        self,
        resource_id: &ResourceId,
    ) -> Result<LogicalBackingBufferView<'a, R::Buffer>, VNextError> {
        match self {
            Self::Invocation(invocation) => invocation.backing_view(resource_id),
            Self::Wave { wave, node_index } => wave.backing_view(node_index, resource_id),
        }
    }
}

/// One participant projection inside a plan-selected physical batch. It has
/// no public constructor and does not own submission authority.
pub struct OperationInvocation<'a, B> {
    identity: &'a ExecutionIdentityEnvelope,
    operation: &'a OperationDescriptor,
    node_id: &'a NodeId,
    provider_id: &'a ProviderId,
    views: Vec<OperationBufferView<'a, B>>,
    bindings: Vec<ResolvedValueBinding>,
    attributes: &'a BTreeMap<AttributeId, SemanticValue>,
    work: &'a NodeWorkContract,
    scratch_view: Option<usize>,
    persistent_view: Option<usize>,
    work_shape: &'a BatchWorkShape,
    claimed_backing_fingerprint: &'a str,
}

impl<'a, B> OperationInvocation<'a, B> {
    #[allow(clippy::too_many_arguments)]
    fn from_resolved<R>(
        runtime: &R,
        resolved: &'a ResolvedModelPlan,
        provider: &'a OperationProviderDescriptor,
        identity: &'a ExecutionIdentityEnvelope,
        node_id: &'a NodeId,
        resources: OperationInvocationResources<'a, R>,
        active_binding: &TrustedActiveSequenceBinding,
        participant_index: usize,
    ) -> Result<Self, VNextError>
    where
        R: DeviceRuntime<Buffer = B>,
    {
        let plan = resolved.execution_plan();
        let node = plan
            .payload()
            .nodes()
            .iter()
            .find(|node| node.id() == node_id)
            .ok_or_else(|| invalid_operation(format!("plan has no node `{node_id}`")))?;
        let operation = resolved
            .parts()
            .capabilities
            .operation(node.operation_id())?;
        let parts = identity.parts();
        let participant = resources.participant(participant_index)?;
        let participant_backing = resources.participant_backing_snapshot(participant_index)?;
        let participant_frame = resources
            .participant_frames()?
            .get(participant_index)
            .ok_or_else(|| invalid_operation("operation participant frame is missing"))?;
        let participant_session = resources.participant_session_identity(participant_index)?;
        let static_lease = participant.static_provisioning();
        let lease_identity = static_lease.map(|lease| lease.identity());
        let admission = active_binding.plan().static_provisioning_binding();
        let pool_fingerprint = active_binding.static_pool_identity_fingerprint();
        let memory = plan.payload().memory();
        if resources.participant_count()? != resources.prepared_participant_count()?
            || resources.node_id()? != node_id
            || participant_frame.sequence_authority() != participant.sequence_authority()
            || participant_frame.request_authority() != participant.request_authority()
            || resources.plan_evidence()? != *active_binding.plan()
            || resources.coordinator_id()? != active_binding.coordinator_id()
            || participant.sequence_authority() != active_binding.sequence_authority()
            || participant.run_id() != active_binding.run_id()
            || participant.request_id() != active_binding.request_id()
            || !active_binding
                .matches_sequence_session(participant_session.0, participant_session.1)
            || runtime.descriptor() != &resolved.parts().device
            || runtime.descriptor() != resolved.parts().capabilities.device()
            || runtime.descriptor().runtime_implementation_fingerprint
                != plan.payload().device_runtime_implementation_fingerprint()
            || parts.plan_id.as_ref() != Some(plan.payload().plan_id())
            || parts.plan_hash.as_ref() != Some(plan.plan_hash())
            || parts.frame_id != Some(participant_frame.frame_id())
            || parts.node_invocation_id.is_none()
            || parts.node_id.as_ref() != Some(node.id())
            || parts.operation_id.as_ref() != Some(node.operation_id())
            || parts.provider_id.as_ref() != Some(node.selection().selected_provider())
            || parts.device_id.as_ref() != Some(plan.payload().device_id())
            || parts.run_id != *active_binding.run_id()
            || parts.request_id != *active_binding.request_id()
            || parts.transaction_id.as_ref()
                != lease_identity.map(|identity| identity.transaction_id())
            || parts.resource_pool_id != active_binding.static_pool_id()
            || parts.resource_pool_identity_fingerprint.as_deref() != pool_fingerprint.as_deref()
            || parts.provisioning_run_id.as_ref()
                != lease_identity.map(|identity| identity.run_id())
            || parts.provisioning_request_id.as_ref()
                != lease_identity.map(|identity| identity.request_id())
            || parts.active_sequence_slot != Some(active_binding.sequence_authority().sparse_id())
            || parts.admission_generation != Some(active_binding.sequence_authority().generation())
            || parts.activation_epoch != Some(active_binding.activation_epoch())
            || parts.runtime_implementation_fingerprint.as_deref()
                != Some(active_binding.runtime_implementation_fingerprint())
            || parts.active_sequence_fingerprint.as_deref() != Some(active_binding.fingerprint())
            || parts.completed_sequence_fingerprint.is_some()
            || parts.aborted_sequence_fingerprint.is_some()
            || active_binding.plan().plan_id() != plan.payload().plan_id()
            || active_binding.plan().plan_hash() != plan.plan_hash()
            || active_binding.plan().device_id() != plan.payload().device_id()
            || active_binding.plan().runtime_implementation_fingerprint()
                != plan.payload().device_runtime_implementation_fingerprint()
            || active_binding.runtime_implementation_fingerprint()
                != runtime.descriptor().runtime_implementation_fingerprint
            || active_binding.static_provisioning_identity() != lease_identity
            || admission != static_lease.map(|lease| lease.admission())
            || admission.is_some_and(|admission| {
                admission.device_capacity_bytes() != memory.device_capacity_bytes()
                    || admission.usable_capacity_bytes() != memory.usable_capacity_bytes()
                    || admission.plan_static_bytes() != memory.static_bytes()
                    || admission.maximum_active_sequences() != memory.maximum_active_sequences()
            })
            || parts.resource_id.is_some()
            || parts.resource_generation.is_some()
            || parts.resource_batch_fingerprint.is_some()
        {
            return Err(invalid_operation(
                "operation invocation does not close over the runtime device, selected plan, node, provider, request, and lease transaction",
            ));
        }
        let registered = resolved
            .parts()
            .capabilities
            .providers_for(node.operation_id())?
            .iter()
            .find(|candidate| candidate.provider_id() == provider.provider_id())
            .ok_or_else(|| invalid_operation("operation provider is absent from the catalog"))?;
        if registered != provider
            || provider.provider_id() != node.selection().selected_provider()
            || provider.operation_id() != node.operation_id()
            || provider.operation_fingerprint() != node.operation_fingerprint()
            || provider.provider_implementation_fingerprint()
                != node.provider_implementation_fingerprint()
            || provider.device_id() != plan.payload().device_id()
            || !provider.version().satisfies(node.operation_version())
        {
            return Err(invalid_operation(
                "operation provider is not the exact catalog entry selected by the plan",
            ));
        }
        operation.validate_attributes(node.attributes())?;
        operation.validate_resolved_bindings(node.values())?;

        let provider_resources = node.provider_resources();
        if provider_resources.provider_id() != provider.provider_id()
            || provider_resources.estimator_id() != provider.resource_estimator_id()
            || provider_resources.estimator_version() != provider.resource_estimator_version()
            || provider_resources.estimator_implementation_fingerprint()
                != provider.resource_estimator_implementation_fingerprint()
            || provider_resources.value_alignment_bytes()
                < operation.resources.minimum_value_alignment_bytes
            || provider_resources.value_alignment_bytes()
                % operation.resources.minimum_value_alignment_bytes
                != 0
            || !operation
                .resources
                .scratch
                .accepts(provider_resources.scratch().is_some())
            || !operation
                .resources
                .persistent
                .accepts(provider_resources.persistent().is_some())
        {
            return Err(invalid_operation(
                "plan provider resource estimate is not bound to the selected provider and operation contract",
            ));
        }
        let scratch_resource = select_workspace_resource(
            provider_resources.scratch(),
            node.scratch_resource(),
            "scratch",
        )?;
        let persistent_resource = select_workspace_resource(
            provider_resources.persistent(),
            node.persistent_resource(),
            "persistent",
        )?;

        let allocations = plan
            .payload()
            .memory()
            .static_allocations()
            .iter()
            .map(|allocation| (allocation.resource_id(), allocation))
            .collect::<BTreeMap<_, _>>();
        let dynamic_descriptors = memory
            .dynamic_descriptors()
            .iter()
            .map(|descriptor| (descriptor.base_resource_id(), descriptor))
            .collect::<BTreeMap<_, _>>();
        let bindings = node.values().to_vec();
        operation.validate_resolved_bindings(&bindings)?;

        let mut required_resources = bindings
            .iter()
            .flat_map(|binding| binding.storage().components())
            .map(|component| component.resource_id().clone())
            .collect::<BTreeSet<_>>();
        required_resources.extend(scratch_resource.iter().map(|resource| (*resource).clone()));
        required_resources.extend(
            persistent_resource
                .iter()
                .map(|resource| (*resource).clone()),
        );
        let lease_entries = static_lease
            .map(|lease| {
                lease
                    .plan_static_entries()
                    .map(|entry| (entry.resource_id(), entry))
                    .collect::<BTreeMap<_, _>>()
            })
            .unwrap_or_default();
        let mut views = Vec::with_capacity(required_resources.len());
        for resource_id in &required_resources {
            if let Some(allocation) = allocations.get(resource_id) {
                let lease = static_lease.ok_or_else(|| {
                    invalid_operation(format!(
                        "plan-static resource `{resource_id}` lacks static provisioning"
                    ))
                })?;
                let entry = lease_entries.get(resource_id).ok_or_else(|| {
                    invalid_operation(format!(
                        "static lease does not own plan resource `{resource_id}`"
                    ))
                })?;
                if entry.size_bytes() != allocation.size_bytes()
                    || entry.alignment_bytes() != allocation.alignment_bytes()
                    || entry.usage() != allocation.usage()
                    || entry.element_type() != allocation.element_type()
                {
                    return Err(invalid_operation(format!(
                        "static lease metadata differs from plan allocation `{resource_id}`"
                    )));
                }
                let leased = lease.view(resource_id, entry.generation())?;
                views.push(OperationBufferView {
                    descriptor: leased.committed_descriptor().clone(),
                    source: OperationBufferSource::Static(leased),
                });
            } else if let Some(descriptor) = dynamic_descriptors.get(resource_id) {
                let backing = resources.backing_view(resource_id).or_else(|_| {
                    resources.participant_backing_view(participant_index, resource_id)
                })?;
                let expected_bytes = match descriptor.lifetime() {
                    AllocationLifetime::Invocation => descriptor.evaluate_request_bytes_for_shape(
                        resources.work_shape()?.immediate_shape(),
                    )?,
                    AllocationLifetime::Step => descriptor.evaluate_request_bytes_for_shape(
                        resources.step_resources().work_shape().immediate_shape(),
                    )?,
                    AllocationLifetime::Sequence => descriptor
                        .evaluate_request_bytes_for_shape(participant_backing.committed_shape())?,
                    AllocationLifetime::Request => descriptor
                        .evaluate_fit_request_bytes(participant.request_resources().work_shape())?,
                    AllocationLifetime::Plan => {
                        return Err(invalid_operation(format!(
                            "plan-lifetime resource `{resource_id}` cannot use dynamic backing"
                        )))
                    }
                };
                let size_matches = match descriptor.lifetime() {
                    AllocationLifetime::Sequence => backing.size_bytes() >= expected_bytes,
                    _ => backing.size_bytes() == expected_bytes,
                };
                if !size_matches
                    || backing.alignment_bytes() != descriptor.alignment_bytes()
                    || backing.usage() != descriptor.usage()
                    || backing.element_type() != descriptor.element_type()
                    || backing.storage_profile() != descriptor.storage().profile()
                {
                    return Err(invalid_operation(format!(
                        "logical backing extent differs from plan descriptor `{resource_id}`"
                    )));
                }
                views.push(OperationBufferView {
                    descriptor: super::BufferDescriptor {
                        resource_id: resource_id.clone(),
                        size_bytes: backing.size_bytes(),
                        alignment_bytes: backing.alignment_bytes(),
                        usage: backing.usage(),
                        element_type: backing.element_type(),
                    },
                    source: OperationBufferSource::Backing(backing),
                });
            } else {
                return Err(invalid_operation(format!(
                    "plan has no static allocation or dynamic descriptor for `{resource_id}`"
                )));
            }
        }

        let mut descriptors = BTreeMap::new();
        for view in &views {
            match &view.source {
                OperationBufferSource::Static(static_view) => {
                    let actual = runtime.buffer_descriptor(static_view.buffer());
                    if Some(static_view.identity()) != lease_identity
                        || &actual != static_view.committed_descriptor()
                        || static_view.generation() == 0
                    {
                        return Err(invalid_operation(format!(
                            "runtime descriptor differs from committed static resource `{}`",
                            view.resource_id()
                        )));
                    }
                }
                OperationBufferSource::Backing(backing_view) => {
                    let bindings = backing_view.segment_bindings();
                    if bindings.is_empty()
                        || bindings.len() != backing_view.committed_evidence_segments().count()
                        || bindings
                            .iter()
                            .zip(backing_view.committed_evidence_segments())
                            .any(|(binding, evidence)| {
                                let actual = runtime.buffer_descriptor(binding.buffer());
                                binding.segment() != evidence
                                    || binding.chunk() != evidence.chunk()
                                    || &actual != binding.descriptor()
                                    || binding
                                        .segment()
                                        .offset_bytes()
                                        .checked_add(binding.segment().length_bytes())
                                        .is_none_or(|end| end > binding.descriptor().size_bytes)
                            })
                    {
                        return Err(invalid_operation(format!(
                            "runtime descriptor differs from a committed backing chunk for `{}`",
                            view.resource_id()
                        )));
                    }
                }
            }
            let translated = view.translate(0, view.descriptor.size_bytes)?;
            let translated_bytes = translated.iter().try_fold(0_u64, |total, region| {
                total
                    .checked_add(region.length_bytes())
                    .ok_or_else(|| invalid_operation("translated operation regions overflow u64"))
            })?;
            if translated_bytes != view.descriptor.size_bytes {
                return Err(invalid_operation(format!(
                    "operation resource `{}` is not fully backed by physical regions",
                    view.resource_id()
                )));
            }
            if descriptors
                .insert(view.resource_id().clone(), view.descriptor.clone())
                .is_some()
            {
                return Err(invalid_operation(format!(
                    "operation resource `{}` is duplicated",
                    view.resource_id()
                )));
            }
        }
        for binding in &bindings {
            for component in binding.storage().components() {
                let descriptor = descriptors.get(component.resource_id()).ok_or_else(|| {
                    invalid_operation("value binding lacks a committed resource view")
                })?;
                let required_end = component
                    .offset_bytes()
                    .checked_add(component.length_bytes())
                    .ok_or_else(|| invalid_operation("bound component range overflows u64"))?;
                if required_end > descriptor.size_bytes
                    || descriptor.usage != binding.usage()
                    || descriptor.element_type != component.element_type()
                    || descriptor.alignment_bytes < provider_resources.value_alignment_bytes()
                    || descriptor.alignment_bytes % provider_resources.value_alignment_bytes() != 0
                    || component.offset_bytes() % provider_resources.value_alignment_bytes() != 0
                {
                    return Err(invalid_operation(format!(
                        "resource `{}` differs from its value binding",
                        component.resource_id()
                    )));
                }
                let view = views
                    .iter()
                    .find(|view| view.resource_id() == component.resource_id())
                    .ok_or_else(|| {
                        invalid_operation("value binding lacks a physical region translator")
                    })?;
                let translated =
                    view.translate(component.offset_bytes(), component.length_bytes())?;
                let translated_bytes = translated.iter().try_fold(0_u64, |total, region| {
                    total.checked_add(region.length_bytes()).ok_or_else(|| {
                        invalid_operation("translated value-binding regions overflow u64")
                    })
                })?;
                if translated_bytes != component.length_bytes() {
                    return Err(invalid_operation(format!(
                        "resource `{}` does not physically cover its value binding",
                        component.resource_id()
                    )));
                }
            }
        }
        let scratch_view = scratch_resource
            .map(|resource| view_index(&views, resource, "scratch"))
            .transpose()?;
        let persistent_view = persistent_resource
            .map(|resource| view_index(&views, resource, "persistent"))
            .transpose()?;
        validate_workspace(
            &views,
            scratch_view,
            BufferUsage::Scratch,
            provider_resources.scratch(),
            "scratch",
        )?;
        validate_workspace(
            &views,
            persistent_view,
            BufferUsage::Persistent,
            provider_resources.persistent(),
            "persistent",
        )?;
        Ok(Self {
            identity,
            operation,
            node_id,
            provider_id: provider.provider_id(),
            views,
            bindings,
            attributes: node.attributes(),
            work: node.work(),
            scratch_view,
            persistent_view,
            work_shape: resources.work_shape()?,
            claimed_backing_fingerprint: resources.backing_fingerprint(),
        })
    }

    pub fn identity(&self) -> &ExecutionIdentityEnvelope {
        self.identity
    }

    pub fn operation(&self) -> &OperationDescriptor {
        self.operation
    }

    pub fn node_id(&self) -> &NodeId {
        self.node_id
    }

    pub fn provider_id(&self) -> &ProviderId {
        self.provider_id
    }

    pub fn views(&self) -> &[OperationBufferView<'a, B>] {
        &self.views
    }

    pub fn bindings(&self) -> &[ResolvedValueBinding] {
        &self.bindings
    }

    pub fn attributes(&self) -> &BTreeMap<AttributeId, SemanticValue> {
        self.attributes
    }

    pub fn work(&self) -> &NodeWorkContract {
        self.work
    }

    pub fn scratch_view(&self) -> Option<&OperationBufferView<'a, B>> {
        self.scratch_view.map(|index| &self.views[index])
    }

    pub fn persistent_view(&self) -> Option<&OperationBufferView<'a, B>> {
        self.persistent_view.map(|index| &self.views[index])
    }

    pub fn work_shape(&self) -> &BatchWorkShape {
        self.work_shape
    }

    pub fn claimed_backing_fingerprint(&self) -> &str {
        self.claimed_backing_fingerprint
    }
}

/// Borrowed provider view for exactly one physical command. Participant-local
/// resources remain separate projections while invocation/step/plan resources
/// may be shared by every projection.
pub struct BatchedOperationInvocation<'a, B> {
    batch_identity: &'a BatchOperationIdentity,
    node_identity: &'a BatchOperationNodeIdentity,
    participants: Vec<OperationInvocation<'a, B>>,
}

impl<'a, B> BatchedOperationInvocation<'a, B> {
    fn from_resolved<R>(
        runtime: &R,
        resolved: &'a ResolvedModelPlan,
        provider: &'a OperationProviderDescriptor,
        batch_identity: &'a BatchOperationIdentity,
        resources: &'a InvocationResourceLease<R>,
        active_bindings: &'a [TrustedActiveSequenceBinding],
    ) -> Result<Self, VNextError>
    where
        R: DeviceRuntime<Buffer = B>,
    {
        let node_identity = batch_identity.single_node().ok_or_else(|| {
            invalid_operation("single-operation invocation received a multi-node batch identity")
        })?;
        Self::from_resources(
            runtime,
            resolved,
            provider,
            batch_identity,
            node_identity,
            OperationInvocationResources::Invocation(resources),
            active_bindings,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn from_wave_node<R>(
        runtime: &R,
        resolved: &'a ResolvedModelPlan,
        provider: &'a OperationProviderDescriptor,
        batch_identity: &'a BatchOperationIdentity,
        node_identity: &'a BatchOperationNodeIdentity,
        wave: &'a PreparedStepSubmissionWave<R>,
        node_index: usize,
        active_bindings: &'a [TrustedActiveSequenceBinding],
    ) -> Result<Self, VNextError>
    where
        R: DeviceRuntime<Buffer = B>,
    {
        Self::from_resources(
            runtime,
            resolved,
            provider,
            batch_identity,
            node_identity,
            OperationInvocationResources::Wave { wave, node_index },
            active_bindings,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn from_resources<R>(
        runtime: &R,
        resolved: &'a ResolvedModelPlan,
        provider: &'a OperationProviderDescriptor,
        batch_identity: &'a BatchOperationIdentity,
        node_identity: &'a BatchOperationNodeIdentity,
        resources: OperationInvocationResources<'a, R>,
        active_bindings: &'a [TrustedActiveSequenceBinding],
    ) -> Result<Self, VNextError>
    where
        R: DeviceRuntime<Buffer = B>,
    {
        let participant_count = resources.participant_count()?;
        let resource_keys = resources.participant_node_keys()?;
        if participant_count == 0
            || participant_count != active_bindings.len()
            || participant_count != node_identity.participants().len()
            || participant_count != resource_keys.len()
            || batch_identity.batch_step_id() != resources.batch_step_id()
            || batch_identity.batch_invocation_id() != resources.batch_invocation_id()
            || node_identity.node_id() != resources.node_id()?
            || node_identity.work_shape_fingerprint() != resources.work_shape()?.fingerprint()
            || batch_identity.claimed_backing_fingerprint() != resources.backing_fingerprint()
            || node_identity
                .participants()
                .iter()
                .zip(&resource_keys)
                .any(|(participant, key)| participant.node_key() != key)
        {
            return Err(invalid_operation(
                "batched operation identity differs from its exact invocation resources",
            ));
        }
        let participants = node_identity
            .participants()
            .iter()
            .zip(active_bindings)
            .enumerate()
            .map(|(index, (participant, active_binding))| {
                OperationInvocation::from_resolved(
                    runtime,
                    resolved,
                    provider,
                    participant.identity(),
                    node_identity.node_id(),
                    resources,
                    active_binding,
                    index,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            batch_identity,
            node_identity,
            participants,
        })
    }

    pub fn batch_identity(&self) -> &BatchOperationIdentity {
        self.batch_identity
    }

    pub fn participants(&self) -> &[OperationInvocation<'a, B>] {
        &self.participants
    }

    pub fn operation(&self) -> &OperationDescriptor {
        self.participants[0].operation()
    }

    pub fn node_id(&self) -> &NodeId {
        self.node_identity.node_id()
    }

    pub fn provider_id(&self) -> &ProviderId {
        self.node_identity.provider_id()
    }

    pub fn work_shape(&self) -> &BatchWorkShape {
        self.participants[0].work_shape()
    }

    pub fn work_contract(&self) -> &NodeWorkContract {
        self.participants[0].work()
    }

    pub fn participant_token_ranges(&self) -> &[BatchParticipantTokenRange] {
        self.work_shape().participant_token_ranges()
    }
}

fn select_workspace_resource<'a>(
    requirement: Option<&ProviderWorkspaceRequirement>,
    resource: Option<&'a ResourceId>,
    kind: &str,
) -> Result<Option<&'a ResourceId>, VNextError> {
    let Some(requirement) = requirement else {
        if resource.is_none() {
            return Ok(None);
        }
        return Err(invalid_operation(format!(
            "plan has unrequested {kind} resources"
        )));
    };
    resource.map(Some).ok_or_else(|| {
        invalid_operation(format!(
            "{kind} workspace base identity is missing for {:?} scope",
            requirement.scope()
        ))
    })
}

fn view_index<B>(
    views: &[OperationBufferView<'_, B>],
    resource_id: &ResourceId,
    kind: &str,
) -> Result<usize, VNextError> {
    views
        .iter()
        .position(|view| view.resource_id() == resource_id)
        .ok_or_else(|| invalid_operation(format!("{kind} resource view is missing")))
}

fn validate_workspace<B>(
    views: &[OperationBufferView<'_, B>],
    index: Option<usize>,
    usage: BufferUsage,
    requirement: Option<&ProviderWorkspaceRequirement>,
    kind: &str,
) -> Result<(), VNextError> {
    match (requirement, index) {
        (None, None) => Ok(()),
        (None, Some(_)) | (Some(_), None) => Err(invalid_operation(format!(
            "{kind} workspace presence differs from the operation contract"
        ))),
        (Some(requirement), Some(index)) => {
            let descriptor = views[index].descriptor();
            let required_bytes = requirement.minimum_bytes()?;
            if descriptor.usage != usage
                || descriptor.element_type != ElementType::U8
                || descriptor.size_bytes < required_bytes
                || descriptor.alignment_bytes < requirement.alignment_bytes()
                || descriptor.alignment_bytes % requirement.alignment_bytes() != 0
            {
                return Err(invalid_operation(format!(
                    "{kind} workspace descriptor is invalid"
                )));
            }
            let translated = views[index].translate(0, required_bytes)?;
            let translated_bytes = translated.iter().try_fold(0_u64, |total, region| {
                total.checked_add(region.length_bytes()).ok_or_else(|| {
                    invalid_operation(format!("{kind} workspace region coverage overflows u64"))
                })
            })?;
            if translated_bytes != required_bytes {
                return Err(invalid_operation(format!(
                    "{kind} workspace is not fully backed by physical regions"
                )));
            }
            Ok(())
        }
    }
}

pub trait DispatchRetryAuthority: fmt::Debug {
    fn prior_attempt(&self) -> BatchInvocationId;
}

impl<R: DeviceRuntime> DispatchRetryAuthority for DefinitelyNotSubmittedRetryAuthority<R> {
    fn prior_attempt(&self) -> BatchInvocationId {
        self.prior_attempt()
    }
}

impl<R: DeviceRuntime> DispatchRetryAuthority for DefinitelyNotSubmittedWaveRetryAuthority<R> {
    fn prior_attempt(&self) -> BatchInvocationId {
        self.prior_attempt()
    }
}

pub enum OperationDispatchError<R, Retry = DefinitelyNotSubmittedRetryAuthority<R>>
where
    R: DeviceRuntime,
    Retry: DispatchRetryAuthority,
{
    Contract(VNextError),
    Provider(OperationFailure),
    DefinitelyNotSubmitted {
        failures: Vec<IdentifiedFailure>,
        retry: Retry,
    },
    SubmissionIndeterminate {
        recovery: IndeterminateSubmissionHandle<R>,
    },
    PostSubmitContract {
        error: VNextError,
        completion: CompletionHandle<R>,
    },
}

pub type SubmissionWaveDispatchError<R> =
    OperationDispatchError<R, DefinitelyNotSubmittedWaveRetryAuthority<R>>;

impl<R, Retry> fmt::Debug for OperationDispatchError<R, Retry>
where
    R: DeviceRuntime,
    Retry: DispatchRetryAuthority,
{
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Contract(error) => formatter.debug_tuple("Contract").field(error).finish(),
            Self::Provider(error) => formatter.debug_tuple("Provider").field(error).finish(),
            Self::DefinitelyNotSubmitted { failures, retry } => formatter
                .debug_struct("DefinitelyNotSubmitted")
                .field("failures", failures)
                .field("retry", retry)
                .finish(),
            Self::SubmissionIndeterminate { recovery } => formatter
                .debug_struct("SubmissionIndeterminate")
                .field("recovery", recovery)
                .finish(),
            Self::PostSubmitContract { error, completion } => formatter
                .debug_struct("PostSubmitContract")
                .field("error", error)
                .field("completion", completion)
                .finish(),
        }
    }
}

impl<R, Retry> fmt::Display for OperationDispatchError<R, Retry>
where
    R: DeviceRuntime,
    Retry: DispatchRetryAuthority,
{
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Contract(error) => {
                write!(formatter, "operation dispatch contract failed: {error}")
            }
            Self::Provider(error) => write!(
                formatter,
                "operation provider failed with {}: {}",
                error.code(),
                error.message()
            ),
            Self::DefinitelyNotSubmitted { failures, retry } => write!(
                formatter,
                "operation attempt {} with {} participants was definitely not submitted: {}",
                retry.prior_attempt(),
                failures.len(),
                failures
                    .first()
                    .map(|failure| failure.failure().message())
                    .unwrap_or("missing classified participant failure")
            ),
            Self::SubmissionIndeterminate { recovery } => write!(
                formatter,
                "operation submission may have reached the device; completion slot {} retains ownership",
                recovery.slot_id().get()
            ),
            Self::PostSubmitContract { error, completion } => write!(
                formatter,
                "operation submission reached the device but slot {} observed a contract failure: {error}",
                completion.slot_id().get()
            ),
        }
    }
}

impl<R, Retry> std::error::Error for OperationDispatchError<R, Retry>
where
    R: DeviceRuntime,
    Retry: DispatchRetryAuthority,
{
}

/// The only public path from a resolved plan to an operation kernel.
pub struct OperationDispatch;

impl OperationDispatch {
    #[allow(clippy::too_many_arguments)]
    fn bind_node_identity<R>(
        resolved: &ResolvedModelPlan,
        participant_identities: Vec<ExecutionIdentityEnvelope>,
        active_bindings: &[TrustedActiveSequenceBinding],
        resources: OperationInvocationResources<'_, R>,
        lane: &Arc<ExecutionLane<R>>,
        node_index: u32,
        participant_start: u32,
    ) -> Result<BatchOperationNodeIdentity, VNextError>
    where
        R: DeviceRuntime,
    {
        let plan = resolved.execution_plan();
        let node_id = resources.node_id()?;
        let node = plan
            .payload()
            .nodes()
            .iter()
            .find(|node| node.id() == node_id)
            .ok_or_else(|| invalid_operation(format!("plan has no node `{node_id}`")))?;
        let participant_count = resources.participant_count()?;
        let participants = (0..participant_count)
            .map(|index| resources.participant(index))
            .collect::<Result<Vec<_>, _>>()?;
        let frames = resources.participant_frames()?;
        let sessions = (0..participant_count)
            .map(|index| resources.participant_session_identity(index))
            .collect::<Result<Vec<_>, _>>()?;
        let node_keys = resources.participant_node_keys()?;
        let plan_evidence = resources.plan_evidence()?;
        if participant_identities.is_empty()
            || participant_identities.len() != participants.len()
            || participant_identities.len() != frames.len()
            || participant_identities.len() != sessions.len()
            || participant_identities.len() != node_keys.len()
            || participant_identities.len() != active_bindings.len()
            || resources.prepared_participant_count()? != participant_count
            || plan_evidence.plan_id() != plan.payload().plan_id()
            || plan_evidence.plan_hash() != plan.plan_hash()
            || plan_evidence.device_id() != plan.payload().device_id()
            || !Arc::ptr_eq(resources.runtime(), lane.runtime_arc())
            || lane.descriptor() != &resolved.parts().device
            || lane.descriptor() != resolved.parts().capabilities.device()
            || lane.descriptor().runtime_implementation_fingerprint
                != plan.payload().device_runtime_implementation_fingerprint()
        {
            return Err(invalid_operation(
                "batch node identity inputs differ from submission resources, plan, or lane",
            ));
        }
        let mut participant_projections = Vec::with_capacity(participant_identities.len());
        for (local_index, identity) in participant_identities.into_iter().enumerate() {
            let participant = participants[local_index];
            let frame = frames[local_index];
            let active = &active_bindings[local_index];
            let session = sessions[local_index];
            let key = &node_keys[local_index];
            let parts = identity.parts();
            if key.sequence_authority() != participant.sequence_authority()
                || key.request_authority() != participant.request_authority()
                || key.frame_id() != frame.frame_id()
                || active.sequence_authority() != participant.sequence_authority()
                || active.coordinator_id() != resources.coordinator_id()?
                || active.run_id() != participant.run_id()
                || active.request_id() != participant.request_id()
                || !active.matches_sequence_session(session.0, session.1)
                || active.plan().plan_id() != plan.payload().plan_id()
                || active.plan().plan_hash() != plan.plan_hash()
                || active.plan().device_id() != plan.payload().device_id()
                || active.runtime_implementation_fingerprint()
                    != plan.payload().device_runtime_implementation_fingerprint()
                || parts.run_id != *active.run_id()
                || parts.request_id != *active.request_id()
                || parts.plan_id.as_ref() != Some(plan.payload().plan_id())
                || parts.plan_hash.as_ref() != Some(plan.plan_hash())
                || parts.frame_id != Some(frame.frame_id())
                || parts.node_invocation_id.is_none()
                || parts.node_id.as_ref() != Some(node.id())
                || parts.operation_id.as_ref() != Some(node.operation_id())
                || parts.provider_id.as_ref() != Some(node.selection().selected_provider())
                || parts.device_id.as_ref() != Some(plan.payload().device_id())
                || parts.active_sequence_slot != Some(active.sequence_authority().sparse_id())
                || parts.admission_generation != Some(active.sequence_authority().generation())
                || parts.activation_epoch != Some(active.activation_epoch())
                || parts.runtime_implementation_fingerprint.as_deref()
                    != Some(active.runtime_implementation_fingerprint())
                || parts.active_sequence_fingerprint.as_deref() != Some(active.fingerprint())
                || parts.completed_sequence_fingerprint.is_some()
                || parts.aborted_sequence_fingerprint.is_some()
                || parts.resource_id.is_some()
                || parts.resource_generation.is_some()
                || parts.resource_batch_fingerprint.is_some()
            {
                return Err(invalid_operation(format!(
                    "batch node {node_index} participant {local_index} differs from its resource, frame, session, or plan identity"
                )));
            }
            let local_index = u32::try_from(local_index)
                .map_err(|_| invalid_operation("batch participant index exceeds u32"))?;
            participant_projections.push(BatchOperationParticipantIdentity {
                participant_index: participant_start.checked_add(local_index).ok_or_else(|| {
                    invalid_operation("physical batch participant index overflows u32")
                })?,
                node_key: key.clone(),
                identity,
            });
        }
        BatchOperationNodeIdentity::from_validated(
            node_index,
            node.id().clone(),
            node.operation_id().clone(),
            node.selection().selected_provider().clone(),
            resources.work_shape()?.fingerprint().to_owned(),
            participant_projections,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn bind_batch_identity<R>(
        resolved: &ResolvedModelPlan,
        participant_identities: Vec<ExecutionIdentityEnvelope>,
        active_bindings: &[TrustedActiveSequenceBinding],
        invocation_resources: &InvocationResourceLease<R>,
        lane: &Arc<ExecutionLane<R>>,
    ) -> Result<BatchOperationIdentity, VNextError>
    where
        R: DeviceRuntime,
    {
        let plan = resolved.execution_plan();
        let resources = OperationInvocationResources::Invocation(invocation_resources);
        let node_identity = Self::bind_node_identity(
            resolved,
            participant_identities,
            active_bindings,
            resources,
            lane,
            0,
            0,
        )?;
        BatchOperationIdentity::from_validated(
            resources.batch_step_id(),
            resources.batch_invocation_id(),
            plan.payload().plan_id().clone(),
            plan.plan_hash().clone(),
            plan.payload().device_id().clone(),
            plan.payload()
                .device_runtime_implementation_fingerprint()
                .to_owned(),
            lane.id(),
            resources.backing_fingerprint().to_owned(),
            vec![node_identity],
        )
    }

    pub fn bind_submission_wave_identity<R>(
        resolved: &ResolvedModelPlan,
        participant_identities: Vec<Vec<ExecutionIdentityEnvelope>>,
        active_bindings: &[Vec<TrustedActiveSequenceBinding>],
        wave: &PreparedStepSubmissionWave<R>,
        lane: &Arc<ExecutionLane<R>>,
    ) -> Result<BatchOperationIdentity, VNextError>
    where
        R: DeviceRuntime,
    {
        let plan = resolved.execution_plan();
        if wave.nodes().is_empty()
            || wave.nodes().len() != plan.payload().nodes().len()
            || participant_identities.len() != wave.nodes().len()
            || active_bindings.len() != wave.nodes().len()
            || wave
                .nodes()
                .iter()
                .zip(plan.payload().nodes())
                .any(|(prepared, planned)| prepared.node_id() != planned.id())
        {
            return Err(invalid_operation(
                "submission wave identity must cover every immutable plan node in order",
            ));
        }
        let mut participant_start = 0_u32;
        let mut nodes = Vec::with_capacity(wave.nodes().len());
        for (node_index, ((identities, bindings), _node)) in participant_identities
            .into_iter()
            .zip(active_bindings)
            .zip(wave.nodes())
            .enumerate()
        {
            let node_index = u32::try_from(node_index)
                .map_err(|_| invalid_operation("submission wave node index exceeds u32"))?;
            let participant_count = u32::try_from(identities.len())
                .map_err(|_| invalid_operation("submission wave participant count exceeds u32"))?;
            let node_identity = Self::bind_node_identity(
                resolved,
                identities,
                bindings,
                OperationInvocationResources::Wave {
                    wave,
                    node_index: node_index as usize,
                },
                lane,
                node_index,
                participant_start,
            )?;
            participant_start = participant_start
                .checked_add(participant_count)
                .ok_or_else(|| {
                    invalid_operation("submission wave participant index space overflows u32")
                })?;
            nodes.push(node_identity);
        }
        BatchOperationIdentity::from_validated(
            wave.batch_step_id(),
            wave.batch_invocation_id(),
            plan.payload().plan_id().clone(),
            plan.plan_hash().clone(),
            plan.payload().device_id().clone(),
            plan.payload()
                .device_runtime_implementation_fingerprint()
                .to_owned(),
            lane.id(),
            wave.fingerprint().to_owned(),
            nodes,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn encode_and_submit<R>(
        provider: &BoundOperationProvider<'_, R>,
        resolved: &ResolvedModelPlan,
        batch_identity: &BatchOperationIdentity,
        active_bindings: &[TrustedActiveSequenceBinding],
        mut invocation_resources: InvocationResourceLease<R>,
        lane: &Arc<ExecutionLane<R>>,
        reaper: &Arc<CompletionReaper<R>>,
    ) -> Result<CompletionHandle<R>, OperationDispatchError<R>>
    where
        R: DeviceRuntime,
    {
        let node_identity = batch_identity.single_node().ok_or_else(|| {
            OperationDispatchError::Contract(invalid_operation(
                "single-operation dispatch requires a one-node batch identity",
            ))
        })?;
        provider
            .validate_binding(resolved, node_identity.node_id())
            .map_err(OperationDispatchError::Contract)?;
        if active_bindings.is_empty()
            || active_bindings.len() != batch_identity.participants().len()
            || lane.id() != batch_identity.lane_id()
            || lane.descriptor().id != *batch_identity.device_id()
            || lane.descriptor().runtime_implementation_fingerprint
                != batch_identity.runtime_implementation_fingerprint()
        {
            return Err(OperationDispatchError::Contract(invalid_operation(
                "operation execution lane or participant set differs from batch identity",
            )));
        }
        invocation_resources
            .begin_dispatch()
            .map_err(OperationDispatchError::Contract)?;
        let mut completion = CompletionReaper::reserve(
            reaper,
            invocation_resources,
            Arc::clone(lane),
            batch_identity.clone(),
        )
        .map_err(OperationDispatchError::Contract)?;
        let runtime = lane.runtime();
        if !lane.current_descriptor_matches_snapshot() {
            return Err(OperationDispatchError::Contract(invalid_operation(
                "operation encode runtime differs from its execution lane snapshot",
            )));
        }
        let invocation = BatchedOperationInvocation::from_resolved(
            runtime,
            resolved,
            provider.provider.descriptor(),
            batch_identity,
            completion.invocation(),
            active_bindings,
        )
        .map_err(OperationDispatchError::Contract)?;
        let expected_phase = invocation.operation().profile_phase;
        let command = match provider.provider.encode_selected(invocation) {
            Ok(command) => command,
            Err(failure)
                if batch_identity.contains_identity(failure.identity())
                    && failure.phase() == expected_phase =>
            {
                return Err(OperationDispatchError::Provider(failure));
            }
            Err(_) => {
                return Err(OperationDispatchError::Contract(invalid_operation(
                    "operation provider returned a failure for a different execution identity or profile phase",
                )));
            }
        };
        if !lane.current_descriptor_matches_snapshot() {
            return Err(OperationDispatchError::Contract(invalid_operation(
                "operation encode completion runtime drifted",
            )));
        }
        let mut lane_reservation = lane
            .reserve_enqueue()
            .map_err(OperationDispatchError::Contract)?;
        completion.mark_submission_started();
        match lane_reservation.submit(DeviceCommandBatch::singleton(command)) {
            LaneSubmitOutcome::DefinitelyNotSubmitted(error) => {
                drop(lane_reservation);
                let retry = completion
                    .definitely_not_submitted()
                    .map_err(OperationDispatchError::Contract)?;
                let failures = batch_identity
                    .participants()
                    .iter()
                    .map(|participant| {
                        classify_device_error(runtime, participant.identity().clone(), &error)
                    })
                    .collect::<Result<Vec<_>, _>>()
                    .map_err(OperationDispatchError::Contract)?;
                Err(OperationDispatchError::DefinitelyNotSubmitted { failures, retry })
            }
            LaneSubmitOutcome::PossiblySubmittedPanic => {
                drop(lane_reservation);
                let recovery = completion.submission_indeterminate();
                Err(OperationDispatchError::SubmissionIndeterminate { recovery })
            }
            LaneSubmitOutcome::Submitted(fence) => {
                drop(lane_reservation);
                let completion = match completion.arm(fence) {
                    Ok(completion) => completion,
                    Err((error, completion)) => {
                        return Err(OperationDispatchError::PostSubmitContract {
                            error,
                            completion,
                        });
                    }
                };
                if !lane.current_descriptor_matches_snapshot() {
                    lane.fail_closed();
                    return Err(OperationDispatchError::PostSubmitContract {
                        error: invalid_operation("operation submit completion runtime drifted"),
                        completion,
                    });
                }
                Ok(completion)
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn encode_and_submit_wave<R>(
        providers: &[BoundOperationProvider<'_, R>],
        resolved: &ResolvedModelPlan,
        batch_identity: &BatchOperationIdentity,
        active_bindings: &[Vec<TrustedActiveSequenceBinding>],
        mut wave: PreparedStepSubmissionWave<R>,
        lane: &Arc<ExecutionLane<R>>,
        reaper: &Arc<CompletionReaper<R>>,
    ) -> Result<CompletionHandle<R>, SubmissionWaveDispatchError<R>>
    where
        R: DeviceRuntime,
    {
        let identity_participant_count = batch_identity.participants().len();
        let active_participant_count = active_bindings.iter().map(Vec::len).sum::<usize>();
        if providers.is_empty()
            || providers.len() != wave.nodes().len()
            || providers.len() != batch_identity.nodes().len()
            || providers.len() != active_bindings.len()
            || identity_participant_count != active_participant_count
            || batch_identity.batch_step_id() != wave.batch_step_id()
            || batch_identity.batch_invocation_id() != wave.batch_invocation_id()
            || batch_identity.claimed_backing_fingerprint() != wave.fingerprint()
            || lane.id() != batch_identity.lane_id()
            || lane.descriptor().id != *batch_identity.device_id()
            || lane.descriptor().runtime_implementation_fingerprint
                != batch_identity.runtime_implementation_fingerprint()
        {
            return Err(SubmissionWaveDispatchError::Contract(invalid_operation(
                "wave execution lane, resources, nodes, or participants differ from batch identity",
            )));
        }
        for ((provider, node_identity), prepared_node) in providers
            .iter()
            .zip(batch_identity.nodes())
            .zip(wave.nodes())
        {
            provider
                .validate_binding(resolved, node_identity.node_id())
                .map_err(SubmissionWaveDispatchError::Contract)?;
            if prepared_node.node_id() != node_identity.node_id()
                || prepared_node.work_shape().fingerprint()
                    != node_identity.work_shape_fingerprint()
                || provider.descriptor().provider_id() != node_identity.provider_id()
                || provider.descriptor().operation_id() != node_identity.operation_id()
                || node_identity.participants().len()
                    != usize::try_from(prepared_node.participant_count())
                        .expect("prepared wave participant count fits usize")
            {
                return Err(SubmissionWaveDispatchError::Contract(invalid_operation(
                    "wave provider or node identity differs from its prepared node",
                )));
            }
        }
        wave.begin_dispatch()
            .map_err(SubmissionWaveDispatchError::Contract)?;
        let mut completion =
            CompletionReaper::reserve_wave(reaper, wave, Arc::clone(lane), batch_identity.clone())
                .map_err(SubmissionWaveDispatchError::Contract)?;
        let runtime = lane.runtime();
        if !lane.current_descriptor_matches_snapshot() {
            return Err(SubmissionWaveDispatchError::Contract(invalid_operation(
                "wave encode runtime differs from its execution lane snapshot",
            )));
        }
        let mut commands = DeviceCommandBatch::with_capacity(providers.len());
        for (node_index, ((provider, node_identity), bindings)) in providers
            .iter()
            .zip(batch_identity.nodes())
            .zip(active_bindings)
            .enumerate()
        {
            let invocation = BatchedOperationInvocation::from_wave_node(
                runtime,
                resolved,
                provider.provider.descriptor(),
                batch_identity,
                node_identity,
                completion.wave(),
                node_index,
                bindings,
            )
            .map_err(SubmissionWaveDispatchError::Contract)?;
            let expected_phase = invocation.operation().profile_phase;
            let command = match provider.provider.encode_selected(invocation) {
                Ok(command) => command,
                Err(failure)
                    if node_identity.contains_identity(failure.identity())
                        && failure.phase() == expected_phase =>
                {
                    return Err(SubmissionWaveDispatchError::Provider(failure));
                }
                Err(_) => {
                    return Err(SubmissionWaveDispatchError::Contract(invalid_operation(
                        "wave provider returned a failure for another node identity or profile phase",
                    )));
                }
            };
            commands.push(command);
        }
        if commands.len() != batch_identity.nodes().len()
            || !lane.current_descriptor_matches_snapshot()
        {
            return Err(SubmissionWaveDispatchError::Contract(invalid_operation(
                "wave encode did not produce one command per immutable plan node",
            )));
        }
        let mut lane_reservation = lane
            .reserve_enqueue()
            .map_err(SubmissionWaveDispatchError::Contract)?;
        completion.mark_submission_started();
        match lane_reservation.submit(commands) {
            LaneSubmitOutcome::DefinitelyNotSubmitted(error) => {
                drop(lane_reservation);
                let retry = completion
                    .definitely_not_submitted_wave()
                    .map_err(SubmissionWaveDispatchError::Contract)?;
                let failures = batch_identity
                    .participants()
                    .iter()
                    .map(|participant| {
                        classify_device_error(runtime, participant.identity().clone(), &error)
                    })
                    .collect::<Result<Vec<_>, _>>()
                    .map_err(SubmissionWaveDispatchError::Contract)?;
                Err(SubmissionWaveDispatchError::DefinitelyNotSubmitted { failures, retry })
            }
            LaneSubmitOutcome::PossiblySubmittedPanic => {
                drop(lane_reservation);
                let recovery = completion.submission_indeterminate();
                Err(SubmissionWaveDispatchError::SubmissionIndeterminate { recovery })
            }
            LaneSubmitOutcome::Submitted(fence) => {
                drop(lane_reservation);
                let completion = match completion.arm(fence) {
                    Ok(completion) => completion,
                    Err((error, completion)) => {
                        return Err(SubmissionWaveDispatchError::PostSubmitContract {
                            error,
                            completion,
                        });
                    }
                };
                if !lane.current_descriptor_matches_snapshot() {
                    lane.fail_closed();
                    return Err(SubmissionWaveDispatchError::PostSubmitContract {
                        error: invalid_operation("wave submit completion runtime drifted"),
                        completion,
                    });
                }
                Ok(completion)
            }
        }
    }
}

/// Exact semantic input presented to a selected provider's resource estimator.
/// The core creates this request only after provider selection and verifies the
/// raw estimate against the same independently computed fingerprint. Global
/// admission ceilings are deliberately absent: the provider describes one
/// actual invocation and the scheduler decides how many invocations to admit.
pub struct OperationResourceEstimateRequest<'a> {
    node_id: &'a NodeId,
    operation: &'a OperationDescriptor,
    values: &'a [ResolvedValueBinding],
    attributes: &'a BTreeMap<AttributeId, SemanticValue>,
    input_fingerprint: &'a str,
}

impl<'a> OperationResourceEstimateRequest<'a> {
    pub(crate) fn new(
        node_id: &'a NodeId,
        operation: &'a OperationDescriptor,
        values: &'a [ResolvedValueBinding],
        attributes: &'a BTreeMap<AttributeId, SemanticValue>,
        input_fingerprint: &'a str,
    ) -> Result<Self, VNextError> {
        operation.validate()?;
        operation.validate_attributes(attributes)?;
        operation.validate_resolved_bindings(values)?;
        if !canonical_sha256(input_fingerprint) {
            return Err(invalid_operation(
                "resource estimator request has invalid input fingerprint",
            ));
        }
        Ok(Self {
            node_id,
            operation,
            values,
            attributes,
            input_fingerprint,
        })
    }

    pub fn node_id(&self) -> &NodeId {
        self.node_id
    }

    pub fn operation(&self) -> &OperationDescriptor {
        self.operation
    }

    pub fn values(&self) -> &[ResolvedValueBinding] {
        self.values
    }

    pub fn attributes(&self) -> &BTreeMap<AttributeId, SemanticValue> {
        self.attributes
    }

    pub fn input_fingerprint(&self) -> &str {
        self.input_fingerprint
    }
}

/// Untrusted raw output from one registered provider implementation. Identity
/// and input claims remain explicit so the core can reject a buggy or
/// malicious implementation before creating a trusted plan resource record.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OperationResourceEstimate {
    estimator_id: String,
    estimator_version: ContractVersion,
    estimator_implementation_fingerprint: String,
    claimed_input_fingerprint: String,
    value_alignment_bytes: u64,
    scratch: Option<ProviderWorkspaceRequirement>,
    persistent: Option<ProviderWorkspaceRequirement>,
}

impl OperationResourceEstimate {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        estimator_id: impl Into<String>,
        estimator_version: ContractVersion,
        estimator_implementation_fingerprint: impl Into<String>,
        claimed_input_fingerprint: impl Into<String>,
        value_alignment_bytes: u64,
        scratch: Option<ProviderWorkspaceRequirement>,
        persistent: Option<ProviderWorkspaceRequirement>,
    ) -> Self {
        Self {
            estimator_id: estimator_id.into(),
            estimator_version,
            estimator_implementation_fingerprint: estimator_implementation_fingerprint.into(),
            claimed_input_fingerprint: claimed_input_fingerprint.into(),
            value_alignment_bytes,
            scratch,
            persistent,
        }
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

    pub fn claimed_input_fingerprint(&self) -> &str {
        &self.claimed_input_fingerprint
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

/// Runtime-independent planning half of an operation provider. This remains
/// object-safe so planning can invoke the real implementation without
/// inventing a device runtime type.
pub trait OperationResourceEstimator: Send + Sync {
    fn descriptor(&self) -> &OperationProviderDescriptor;

    fn estimate_resources(
        &self,
        request: OperationResourceEstimateRequest<'_>,
    ) -> Result<OperationResourceEstimate, VNextError>;
}

/// Typed implementation registry used at the planning trust boundary. The
/// core requires exactly one matching contract and estimator; missing or
/// duplicate registrations fail closed before an executable plan is built.
pub trait OperationPlanningRegistry: Send + Sync {
    fn contracts_for(&self, operation_id: &OperationId) -> Vec<&dyn OperationContract>;

    fn estimators_for(&self, provider_id: &ProviderId) -> Vec<&dyn OperationResourceEstimator>;
}

/// Process-local authority for the composition root that supplied the exact
/// contract and provider objects used during planning. It deliberately has no
/// wire representation and is never part of a deterministic plan hash.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct OperationRegistryAuthority(u64);

impl OperationRegistryAuthority {
    fn mint() -> Result<Self, VNextError> {
        static NEXT_AUTHORITY: AtomicU64 = AtomicU64::new(1);
        let id = NEXT_AUTHORITY
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
                current.checked_add(1)
            })
            .map_err(|_| invalid_operation("operation registry authority space exhausted"))?;
        Ok(Self(id))
    }
}

/// Planning view issued only by a concrete runtime registry. Holding this
/// view proves that node resolution used the same composition root that can
/// later bind the selected runtime provider.
pub struct OperationPlanningHandle<'registry> {
    registry: &'registry dyn OperationPlanningRegistry,
    authority: OperationRegistryAuthority,
}

impl OperationPlanningHandle<'_> {
    pub(crate) fn authority(&self) -> &OperationRegistryAuthority {
        &self.authority
    }
}

impl OperationPlanningRegistry for OperationPlanningHandle<'_> {
    fn contracts_for(&self, operation_id: &OperationId) -> Vec<&dyn OperationContract> {
        self.registry.contracts_for(operation_id)
    }

    fn estimators_for(&self, provider_id: &ProviderId) -> Vec<&dyn OperationResourceEstimator> {
        self.registry.estimators_for(provider_id)
    }
}

/// A compile-time provider contract for one concrete runtime buffer type. The
/// kernel method consumes only a dispatch-created invocation.
pub trait OperationProvider<R: DeviceRuntime>: OperationResourceEstimator {
    fn encode_selected(
        &self,
        invocation: BatchedOperationInvocation<'_, R::Buffer>,
    ) -> Result<R::Command, OperationFailure>;
}

/// Composition-root registry that owns the exact provider objects used for
/// both planning and runtime dispatch. A dispatch call receives only a bound
/// handle issued by this registry, never an arbitrary provider implementation.
pub struct OperationRuntimeRegistry<R>
where
    R: DeviceRuntime,
{
    authority: OperationRegistryAuthority,
    contracts: BTreeMap<OperationId, Box<dyn OperationContract>>,
    providers: BTreeMap<ProviderId, Box<dyn OperationProvider<R>>>,
}

impl<R> OperationRuntimeRegistry<R>
where
    R: DeviceRuntime,
{
    pub fn new(
        contracts: Vec<Box<dyn OperationContract>>,
        providers: Vec<Box<dyn OperationProvider<R>>>,
    ) -> Result<Self, VNextError> {
        if contracts.is_empty() || providers.is_empty() {
            return Err(invalid_operation(
                "operation runtime registry requires contracts and providers",
            ));
        }
        let mut contract_map = BTreeMap::new();
        for contract in contracts {
            let descriptor = contract.descriptor();
            descriptor.validate()?;
            let operation_id = descriptor.id.clone();
            if contract_map
                .insert(operation_id.clone(), contract)
                .is_some()
            {
                return Err(invalid_operation(format!(
                    "operation runtime registry has duplicate contract `{operation_id}`"
                )));
            }
        }
        let mut provider_map = BTreeMap::new();
        for provider in providers {
            let descriptor = provider.descriptor();
            let contract = contract_map.get(descriptor.operation_id()).ok_or_else(|| {
                invalid_operation(format!(
                    "runtime provider `{}` has no registered operation contract",
                    descriptor.provider_id()
                ))
            })?;
            if descriptor.operation_fingerprint() != contract.descriptor().fingerprint()? {
                return Err(invalid_operation(format!(
                    "runtime provider `{}` differs from its registered operation contract",
                    descriptor.provider_id()
                )));
            }
            let provider_id = descriptor.provider_id().clone();
            if provider_map.insert(provider_id.clone(), provider).is_some() {
                return Err(invalid_operation(format!(
                    "operation runtime registry has duplicate or byte-identical provider `{provider_id}`"
                )));
            }
        }
        Ok(Self {
            authority: OperationRegistryAuthority::mint()?,
            contracts: contract_map,
            providers: provider_map,
        })
    }

    pub fn planning(&self) -> OperationPlanningHandle<'_> {
        OperationPlanningHandle {
            registry: self,
            authority: self.authority.clone(),
        }
    }

    pub fn bind<'registry>(
        &'registry self,
        resolved: &ResolvedModelPlan,
        node_id: &NodeId,
    ) -> Result<BoundOperationProvider<'registry, R>, VNextError> {
        let plan = resolved.execution_plan();
        if plan.operation_registry_authority() != &self.authority {
            return Err(invalid_operation(
                "resolved plan belongs to a different operation runtime registry",
            ));
        }
        let node = plan
            .payload()
            .nodes()
            .iter()
            .find(|node| node.id() == node_id)
            .ok_or_else(|| invalid_operation(format!("plan has no node `{node_id}`")))?;
        let provider = self
            .providers
            .get(node.selection().selected_provider())
            .ok_or_else(|| {
                invalid_operation(format!(
                    "runtime registry has no selected provider `{}`",
                    node.selection().selected_provider()
                ))
            })?;
        let catalog_provider = resolved
            .parts()
            .capabilities
            .providers_for(node.operation_id())?
            .iter()
            .find(|candidate| candidate.provider_id() == provider.descriptor().provider_id())
            .ok_or_else(|| invalid_operation("runtime provider is absent from resolved catalog"))?;
        if provider.descriptor() != catalog_provider
            || provider.descriptor().provider_id() != node.selection().selected_provider()
            || provider.descriptor().provider_implementation_fingerprint()
                != node.provider_implementation_fingerprint()
        {
            return Err(invalid_operation(
                "runtime provider is not the exact registry object selected by the resolved plan",
            ));
        }
        Ok(BoundOperationProvider {
            provider: provider.as_ref(),
            plan_id: plan.payload().plan_id().clone(),
            plan_hash: plan.plan_hash().clone(),
            node_id: node_id.clone(),
        })
    }
}

impl<R> OperationPlanningRegistry for OperationRuntimeRegistry<R>
where
    R: DeviceRuntime,
{
    fn contracts_for(&self, operation_id: &OperationId) -> Vec<&dyn OperationContract> {
        self.contracts
            .get(operation_id)
            .map(|contract| vec![contract.as_ref()])
            .unwrap_or_default()
    }

    fn estimators_for(&self, provider_id: &ProviderId) -> Vec<&dyn OperationResourceEstimator> {
        self.providers
            .get(provider_id)
            .map(|provider| vec![provider.as_ref() as &dyn OperationResourceEstimator])
            .unwrap_or_default()
    }
}

/// Unforgeable per-node provider authority. Its provider object and plan/node
/// binding are private and remain borrowed from one composition-root registry.
pub struct BoundOperationProvider<'registry, R>
where
    R: DeviceRuntime,
{
    provider: &'registry dyn OperationProvider<R>,
    plan_id: PlanId,
    plan_hash: PlanHash,
    node_id: NodeId,
}

impl<R> BoundOperationProvider<'_, R>
where
    R: DeviceRuntime,
{
    fn validate_binding(
        &self,
        resolved: &ResolvedModelPlan,
        node_id: &NodeId,
    ) -> Result<(), VNextError> {
        let plan = resolved.execution_plan();
        if self.plan_id != *plan.payload().plan_id()
            || self.plan_hash != *plan.plan_hash()
            || &self.node_id != node_id
        {
            return Err(invalid_operation(
                "bound operation provider belongs to a different plan or node",
            ));
        }
        Ok(())
    }

    pub fn descriptor(&self) -> &OperationProviderDescriptor {
        self.provider.descriptor()
    }
}

/// Deterministically ordered provider capabilities consumed once by planning.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct CapabilityCatalog {
    device: super::DeviceDescriptor,
    operations: BTreeMap<OperationId, OperationDescriptor>,
    providers: BTreeMap<OperationId, Vec<OperationProviderDescriptor>>,
    engine_providers: BTreeMap<ProviderId, EngineProviderDescriptor>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CapabilityCatalogWire {
    device: super::DeviceDescriptor,
    operations: BTreeMap<OperationId, OperationDescriptor>,
    providers: BTreeMap<OperationId, Vec<OperationProviderDescriptor>>,
    engine_providers: BTreeMap<ProviderId, EngineProviderDescriptor>,
}

impl<'de> Deserialize<'de> for CapabilityCatalog {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = CapabilityCatalogWire::deserialize(deserializer)?;
        Self::from_maps(
            wire.device,
            wire.operations,
            wire.providers,
            wire.engine_providers,
        )
        .map_err(serde::de::Error::custom)
    }
}

impl CapabilityCatalog {
    pub fn new(
        device: super::DeviceDescriptor,
        operations: Vec<OperationDescriptor>,
        providers: BTreeMap<OperationId, Vec<OperationProviderDescriptor>>,
        engine_providers: Vec<EngineProviderDescriptor>,
    ) -> Result<Self, VNextError> {
        let mut operation_map = BTreeMap::new();
        for operation in operations {
            let operation_id = operation.id.clone();
            if operation_map
                .insert(operation_id.clone(), operation)
                .is_some()
            {
                return Err(invalid_operation(format!(
                    "duplicate operation descriptor `{operation_id}`"
                )));
            }
        }
        let mut engine_map = BTreeMap::new();
        for engine in engine_providers {
            let provider_id = engine.provider_id.clone();
            if engine_map.insert(provider_id.clone(), engine).is_some() {
                return Err(invalid_operation(format!(
                    "duplicate engine provider `{provider_id}`"
                )));
            }
        }
        Self::from_maps(device, operation_map, providers, engine_map)
    }

    fn from_maps(
        device: super::DeviceDescriptor,
        operations: BTreeMap<OperationId, OperationDescriptor>,
        mut providers: BTreeMap<OperationId, Vec<OperationProviderDescriptor>>,
        engine_providers: BTreeMap<ProviderId, EngineProviderDescriptor>,
    ) -> Result<Self, VNextError> {
        device.validate()?;
        let provider_row_count = providers.values().try_fold(0_usize, |total, entries| {
            total.checked_add(entries.len()).ok_or_else(|| {
                invalid_operation("capability catalog provider row count overflows usize")
            })
        })?;
        if operations.is_empty()
            || providers.is_empty()
            || engine_providers.is_empty()
            || operations.len() > MAX_OPERATION_CATALOG_ROWS
            || provider_row_count > MAX_OPERATION_PROVIDER_ROWS
            || engine_providers.len() > MAX_ENGINE_PROVIDER_ROWS
        {
            return Err(invalid_operation(
                "capability catalog is empty or exceeds its operation/provider/engine row budget",
            ));
        }
        if operations.keys().collect::<BTreeSet<_>>() != providers.keys().collect::<BTreeSet<_>>() {
            return Err(invalid_operation(
                "capability catalog operation and provider rows do not match",
            ));
        }
        for (operation_id, operation) in &operations {
            if operation_id != &operation.id {
                return Err(invalid_operation(format!(
                    "operation descriptor `{}` is stored under `{operation_id}`",
                    operation.id
                )));
            }
            operation.validate()?;
            if !operation
                .provider
                .required_capabilities
                .is_subset(&device.capabilities)
            {
                return Err(VNextError::UnsupportedOperation {
                    node_id: None,
                    operation_id: operation_id.to_string(),
                    device_id: device.id.to_string(),
                    reason: "device does not advertise the operation's required capabilities"
                        .to_owned(),
                });
            }
        }
        validate_reference_oracle_graph(&operations)?;
        for (operation_id, entries) in &mut providers {
            if entries.is_empty() {
                return Err(VNextError::UnsupportedOperation {
                    node_id: None,
                    operation_id: operation_id.to_string(),
                    device_id: device.id.to_string(),
                    reason: "provider row is empty".to_owned(),
                });
            }
            let operation =
                operations
                    .get(operation_id)
                    .ok_or_else(|| VNextError::UnsupportedOperation {
                        node_id: None,
                        operation_id: operation_id.to_string(),
                        device_id: device.id.to_string(),
                        reason: "provider row has no operation descriptor".to_owned(),
                    })?;
            let operation_fingerprint = operation.fingerprint()?;
            for entry in entries.iter() {
                if &entry.operation_id != operation_id
                    || entry.operation_fingerprint != operation_fingerprint
                {
                    return Err(VNextError::UnsupportedOperation {
                        node_id: None,
                        operation_id: operation_id.to_string(),
                        device_id: device.id.to_string(),
                        reason: format!(
                            "provider `{}` is bound to a different operation descriptor",
                            entry.provider_id
                        ),
                    });
                }
                if entry.device_id != device.id {
                    return Err(VNextError::UnsupportedOperation {
                        node_id: None,
                        operation_id: operation_id.to_string(),
                        device_id: device.id.to_string(),
                        reason: format!(
                            "provider `{}` belongs to device `{}`",
                            entry.provider_id, entry.device_id
                        ),
                    });
                }
                if !entry.version.satisfies(operation.version)
                    || !entry.version.satisfies(operation.provider.minimum_version)
                {
                    return Err(VNextError::UnsupportedOperation {
                        node_id: None,
                        operation_id: operation_id.to_string(),
                        device_id: device.id.to_string(),
                        reason: format!(
                            "provider `{}` does not satisfy the operation version",
                            entry.provider_id
                        ),
                    });
                }
                if !entry.capabilities.is_subset(&device.capabilities)
                    || !operation
                        .provider
                        .required_capabilities
                        .is_subset(&entry.capabilities)
                {
                    return Err(VNextError::UnsupportedOperation {
                        node_id: None,
                        operation_id: operation_id.to_string(),
                        device_id: device.id.to_string(),
                        reason: format!(
                            "provider `{}` capabilities are incompatible with the device or operation",
                            entry.provider_id
                        ),
                    });
                }
            }
            entries.sort_by(|left, right| {
                left.provider_id
                    .cmp(&right.provider_id)
                    .then(left.version.cmp(&right.version))
            });
            let mut seen = BTreeSet::new();
            if entries
                .iter()
                .any(|entry| !seen.insert(entry.provider_id.clone()))
            {
                return Err(VNextError::UnsupportedOperation {
                    node_id: None,
                    operation_id: operation_id.to_string(),
                    device_id: device.id.to_string(),
                    reason: "duplicate provider identity".to_owned(),
                });
            }
        }
        for (provider_id, engine) in &engine_providers {
            if provider_id != &engine.provider_id
                || engine.device_id != device.id
                || !engine.capabilities.is_subset(&device.capabilities)
            {
                return Err(invalid_operation(format!(
                    "engine provider `{provider_id}` identity, device, or capabilities are invalid"
                )));
            }
        }
        Ok(Self {
            device,
            operations,
            providers,
            engine_providers,
        })
    }

    pub fn device(&self) -> &super::DeviceDescriptor {
        &self.device
    }

    pub fn providers_for(
        &self,
        operation_id: &OperationId,
    ) -> Result<&[OperationProviderDescriptor], VNextError> {
        self.providers
            .get(operation_id)
            .map(Vec::as_slice)
            .ok_or_else(|| VNextError::UnsupportedOperation {
                node_id: None,
                operation_id: operation_id.to_string(),
                device_id: self.device.id.to_string(),
                reason: "no provider is registered".to_owned(),
            })
    }

    pub fn operation(
        &self,
        operation_id: &OperationId,
    ) -> Result<&OperationDescriptor, VNextError> {
        self.operations
            .get(operation_id)
            .ok_or_else(|| VNextError::UnsupportedOperation {
                node_id: None,
                operation_id: operation_id.to_string(),
                device_id: self.device.id.to_string(),
                reason: "operation descriptor is not registered".to_owned(),
            })
    }

    pub fn provider_compatibility(
        &self,
        mut request: ProviderCompatibilityRequest,
    ) -> Result<ProviderCompatibilityReport, VNextError> {
        let operation = self.operation(&request.operation_id)?;
        request
            .required_capabilities
            .extend(operation.provider.required_capabilities.iter().cloned());
        let mut compatible_provider_ids = Vec::new();
        let mut rejected = Vec::new();
        for provider in self.providers_for(&request.operation_id)? {
            let mut reasons = Vec::new();
            if !operation.version.satisfies(request.required_version) {
                reasons.push(
                    ProviderCompatibilityRejectReason::OperationVersionMismatch {
                        required: request.required_version,
                        available: operation.version,
                    },
                );
            }
            if !provider.version.satisfies(request.required_version) {
                reasons.push(ProviderCompatibilityRejectReason::ProviderVersionMismatch {
                    required: request.required_version,
                    available: provider.version,
                });
            }
            let missing_capabilities = request
                .required_capabilities
                .difference(&provider.capabilities)
                .cloned()
                .collect::<BTreeSet<_>>();
            if !missing_capabilities.is_empty() {
                reasons.push(ProviderCompatibilityRejectReason::MissingCapabilities {
                    capabilities: missing_capabilities,
                });
            }
            let missing_weight_formats = request
                .required_weight_formats
                .difference(&provider.accepted_weight_formats)
                .cloned()
                .collect::<BTreeSet<_>>();
            if !missing_weight_formats.is_empty() {
                reasons.push(
                    ProviderCompatibilityRejectReason::UnsupportedWeightFormats {
                        formats: missing_weight_formats,
                    },
                );
            }
            let missing_quantization_formats = request
                .required_quantization_formats
                .difference(&provider.accepted_quantization_formats)
                .cloned()
                .collect::<BTreeSet<_>>();
            if !missing_quantization_formats.is_empty() {
                reasons.push(
                    ProviderCompatibilityRejectReason::UnsupportedQuantizationFormats {
                        formats: missing_quantization_formats,
                    },
                );
            }
            if reasons.is_empty() {
                compatible_provider_ids.push(provider.provider_id.clone());
            } else {
                rejected.push(ProviderCompatibilityRejection {
                    provider_id: provider.provider_id.clone(),
                    reasons,
                });
            }
        }
        let report = ProviderCompatibilityReport {
            request,
            compatible_provider_ids,
            rejected,
        };
        report.validate_shape()?;
        Ok(report)
    }

    pub fn operations(&self) -> &BTreeMap<OperationId, OperationDescriptor> {
        &self.operations
    }

    pub fn providers(&self) -> &BTreeMap<OperationId, Vec<OperationProviderDescriptor>> {
        &self.providers
    }

    pub fn engine_providers(&self) -> &BTreeMap<ProviderId, EngineProviderDescriptor> {
        &self.engine_providers
    }

    pub fn engine_provider(
        &self,
        provider_id: &ProviderId,
        required_version: ContractVersion,
    ) -> Result<&EngineProviderDescriptor, VNextError> {
        let provider = self.engine_providers.get(provider_id).ok_or_else(|| {
            invalid_operation(format!("engine provider `{provider_id}` is not registered"))
        })?;
        if !provider.contract_version.satisfies(required_version) {
            return Err(invalid_operation(format!(
                "engine provider `{provider_id}` version {} does not satisfy {required_version}",
                provider.contract_version
            )));
        }
        Ok(provider)
    }

    pub fn fingerprint(&self) -> Result<String, VNextError> {
        let bytes = serde_json::to_vec(self).map_err(|error| VNextError::Serialization {
            context: "serialize capability catalog",
            message: error.to_string(),
        })?;
        Ok(format!("{:x}", Sha256::digest(bytes)))
    }
}

fn validate_reference_oracle_graph(
    operations: &BTreeMap<OperationId, OperationDescriptor>,
) -> Result<(), VNextError> {
    for (operation_id, operation) in operations {
        if let OracleSpec::ReferenceOperation {
            operation_id: reference_id,
            version,
        } = &operation.oracle
        {
            let reference = operations.get(reference_id).ok_or_else(|| {
                invalid_operation(format!(
                    "operation `{operation_id}` references missing oracle `{reference_id}`"
                ))
            })?;
            if !reference.version.satisfies(*version) {
                return Err(invalid_operation(format!(
                    "operation `{operation_id}` oracle `{reference_id}` version {} does not satisfy {version}",
                    reference.version
                )));
            }
            if operation.inputs != reference.inputs
                || operation.outputs != reference.outputs
                || operation.attributes != reference.attributes
            {
                return Err(invalid_operation(format!(
                    "operation `{operation_id}` oracle `{reference_id}` has an incompatible input/output/attribute contract"
                )));
            }
        }
    }

    #[derive(Clone, Copy, PartialEq, Eq)]
    enum VisitState {
        Visiting,
        Visited,
    }

    let mut states = BTreeMap::<OperationId, VisitState>::new();
    for root in operations.keys() {
        if states.get(root) == Some(&VisitState::Visited) {
            continue;
        }
        let mut path = Vec::<OperationId>::new();
        let mut current = root.clone();
        loop {
            match states.get(&current) {
                Some(VisitState::Visited) => break,
                Some(VisitState::Visiting) => {
                    return Err(invalid_operation(format!(
                        "reference-oracle graph contains a cycle at `{current}`"
                    )));
                }
                None => {}
            }
            if path.len() >= MAX_REFERENCE_ORACLE_DEPTH {
                return Err(invalid_operation(format!(
                    "reference-oracle chain from `{root}` exceeds depth {MAX_REFERENCE_ORACLE_DEPTH}"
                )));
            }
            states.insert(current.clone(), VisitState::Visiting);
            path.push(current.clone());
            let Some(OperationDescriptor {
                oracle:
                    OracleSpec::ReferenceOperation {
                        operation_id: reference_id,
                        ..
                    },
                ..
            }) = operations.get(&current)
            else {
                break;
            };
            current = reference_id.clone();
        }
        for operation_id in path {
            states.insert(operation_id, VisitState::Visited);
        }
    }
    Ok(())
}
