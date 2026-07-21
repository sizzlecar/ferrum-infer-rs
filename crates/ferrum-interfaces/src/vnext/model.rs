use serde::de::DeserializeOwned;
use serde::{Deserialize, Deserializer, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

use super::{
    AttributeId, AttributeValueKind, ContractVersion, ElementType, ExternalModelMetadataId,
    ModelFamilyId, NodeId, OperationId, ProgramValueId, ResolvedTensorLayout, StateId, TokenizerId,
    VNextError, WeightFormatId, WeightId, WeightLayoutId,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QuantizationPacking {
    Linear,
    Interleaved,
    Tiled,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct QuantizationSpec {
    pub format_id: super::QuantizationFormatId,
    pub bits_per_weight: u8,
    pub group_size: u32,
    pub packing: QuantizationPacking,
    pub scale_type: ElementType,
    pub zero_point_type: Option<ElementType>,
}

impl QuantizationSpec {
    pub fn validate(&self) -> Result<(), VNextError> {
        if !(1..=8).contains(&self.bits_per_weight)
            || self.group_size == 0
            || !self.group_size.is_power_of_two()
            || !matches!(
                self.scale_type,
                ElementType::F16 | ElementType::Bf16 | ElementType::F32
            )
            || self.zero_point_type.is_some_and(|element_type| {
                !matches!(
                    element_type,
                    ElementType::U8 | ElementType::U32 | ElementType::I8 | ElementType::I32
                )
            })
        {
            return Err(VNextError::InvalidExecutionPlan {
                reason: format!("invalid quantization format `{}`", self.format_id),
            });
        }
        Ok(())
    }
}

/// Self-contained fixed-size quantization blocks such as GGML/GGUF Q4_K and
/// Q6_K. Per-block scales, minima, and packed values are part of the opaque
/// block ABI identified by `format_id`; providers must not reinterpret these
/// bytes as the separate-scale [`QuantizationSpec`] representation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BlockQuantizationSpec {
    pub format_id: super::QuantizationFormatId,
    pub logical_values_per_block: u32,
    pub bytes_per_block: u32,
}

impl BlockQuantizationSpec {
    pub fn validate(&self) -> Result<(), VNextError> {
        if self.logical_values_per_block == 0 || self.bytes_per_block == 0 {
            return Err(VNextError::InvalidExecutionPlan {
                reason: format!("invalid block quantization format `{}`", self.format_id),
            });
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WeightEncoding {
    Dense {
        element_type: ElementType,
    },
    /// Dense floating-point values materialized after applying
    /// `logical = physical * scale + bias` element-wise. This keeps checkpoint
    /// representation semantics in the typed weight schema rather than in a
    /// backend provider or model-name branch.
    DenseAffine {
        element_type: ElementType,
        scale: CanonicalRational,
        bias: CanonicalRational,
    },
    Quantized(QuantizationSpec),
    BlockQuantized(BlockQuantizationSpec),
}

impl WeightEncoding {
    pub const fn dense_element_type(&self) -> Option<ElementType> {
        match self {
            Self::Dense { element_type } | Self::DenseAffine { element_type, .. } => {
                Some(*element_type)
            }
            Self::Quantized(_) | Self::BlockQuantized(_) => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WeightComponentSpec {
    pub id: WeightId,
    pub role: WeightComponentRole,
    /// Ordered checkpoint tensors forming this physical component. One entry
    /// maps directly to that tensor. Multiple entries must have the component
    /// shape `[source_count, source_shape...]` and are stacked in this exact
    /// order by the format source; aliases are resolved by the model family
    /// before it constructs the typed schema.
    pub external_names: Vec<String>,
    pub dimensions: Vec<u64>,
    pub encoding: WeightEncoding,
    pub required: bool,
}

impl WeightComponentSpec {
    /// Exact bytes occupied by this physical component. Separate-component
    /// quantized dimensions are byte dimensions. Block-quantized dimensions
    /// are block-grid dimensions and are multiplied by the block ABI size.
    /// Logical element counts live on `WeightTensorSpec` and are checked
    /// against the corresponding physical layout contract.
    pub fn physical_bytes(&self) -> Result<u64, VNextError> {
        let elements =
            checked_elements(&self.dimensions).ok_or_else(|| VNextError::InvalidExecutionPlan {
                reason: format!("physical component `{}` size overflows u64", self.id),
            })?;
        match &self.encoding {
            WeightEncoding::Dense { element_type }
            | WeightEncoding::DenseAffine { element_type, .. } => elements
                .checked_mul(element_type.size_bytes())
                .ok_or_else(|| VNextError::InvalidExecutionPlan {
                    reason: format!("physical component `{}` byte size overflows u64", self.id),
                }),
            WeightEncoding::Quantized(_) => Ok(elements),
            WeightEncoding::BlockQuantized(spec) => {
                spec.validate()?;
                elements
                    .checked_mul(u64::from(spec.bytes_per_block))
                    .ok_or_else(|| VNextError::InvalidExecutionPlan {
                        reason: format!(
                            "physical block component `{}` byte size overflows u64",
                            self.id
                        ),
                    })
            }
        }
    }

    pub fn dense_element_type(&self) -> Option<ElementType> {
        self.encoding.dense_element_type()
    }

    pub fn physical_element_type(&self) -> ElementType {
        self.dense_element_type().unwrap_or(ElementType::U8)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PhysicalWeightComponentRef {
    pub component_id: WeightId,
    pub physical_dimensions: Vec<u64>,
    pub resource_bytes: u64,
    pub element_type: ElementType,
}

/// Structural role of a physical component in a weight format. The role is
/// intentionally independent of any named quantization or model family.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WeightComponentRole {
    Values,
    PackedValues,
    Scales,
    ZeroPoints,
    Indices,
    Permutation,
    Codebook,
    Metadata,
}

/// Padding is always explicit and carries the exact semantic padded shape.
/// `Exact` has no hidden storage extension. `ZeroFill` must increase at least
/// one dimension and, for tiled or grouped storage, must be the unique minimal
/// shape implied by that contract.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PhysicalWeightPadding {
    Exact,
    ZeroFill { padded_dimensions: Vec<u64> },
}

/// Storage geometry for one physical component binding. Strides are measured
/// in the component's schema storage unit: elements for dense encodings,
/// bytes for separate-component packing, and blocks for block quantization.
/// The component's declared dimensions describe its raw stored span, while
/// this geometry maps the semantic component shape onto that span without
/// inference or hidden padding.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PhysicalStorageLayout {
    Contiguous {
        padding: PhysicalWeightPadding,
    },
    Strided {
        strides_in_elements: Vec<u64>,
        padding: PhysicalWeightPadding,
    },
    Tiled {
        tile_shape: Vec<u64>,
        /// Physical tile-grid axis -> semantic component axis.
        axis_order: Vec<u32>,
        tile_strides_in_elements: Vec<u64>,
        padding: PhysicalWeightPadding,
    },
}

impl PhysicalStorageLayout {
    pub fn exact_contiguous() -> Self {
        Self::Contiguous {
            padding: PhysicalWeightPadding::Exact,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PhysicalWeightComponentBinding {
    pub component_id: WeightId,
    pub storage: PhysicalStorageLayout,
}

impl PhysicalWeightComponentBinding {
    pub fn exact_contiguous(component_id: WeightId) -> Self {
        Self {
            component_id,
            storage: PhysicalStorageLayout::exact_contiguous(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AxisWeightComponent {
    pub component: PhysicalWeightComponentBinding,
    pub axis: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompositeWeightPart {
    pub layout: Box<PhysicalWeightLayout>,
    pub logical_offsets: Vec<u64>,
    pub extents: Vec<u64>,
}

/// Hard bounds keep directly constructed and deserialized recursive schemas
/// cheap to validate. Ownership makes cycles unrepresentable; these limits
/// additionally bound adversarial depth and fan-out.
pub const MAX_PHYSICAL_WEIGHT_LAYOUT_DEPTH: usize = 16;
pub const MAX_PHYSICAL_WEIGHT_LAYOUT_NODES: usize = 4096;

/// Typed physical storage tree for one logical weight. Every leaf binds one
/// physical component exactly once. Recursive composition allows indexing or
/// expert stacking around dense, tiled, strided, or quantized values without
/// architecture-specific cases.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PhysicalWeightLayout {
    /// Exact, contiguous dense values. This common leaf deliberately remains
    /// compact; use `Stored` for explicit stride, tile, or padding geometry.
    Dense {
        component_id: WeightId,
    },
    Stored {
        component: PhysicalWeightComponentBinding,
    },
    Composite {
        parts: Vec<CompositeWeightPart>,
    },
    Quantized {
        packed_values: PhysicalWeightComponentBinding,
        /// Semantic packed-storage shape before the binding's optional
        /// stride/tile mapping. Its element product must equal the exact byte
        /// count implied by logical elements and `bits_per_weight`.
        packed_dimensions: Vec<u64>,
        scales: PhysicalWeightComponentBinding,
        zero_points: Option<PhysicalWeightComponentBinding>,
        /// Per-coordinate group assignment. Its semantic shape is the one
        /// dimensional logical axis extent, not the multi-dimensional group
        /// tensor shape.
        axis_indices: Option<AxisWeightComponent>,
        permutation: Option<AxisWeightComponent>,
        codebook: Option<PhysicalWeightComponentBinding>,
        group_axis: u32,
        group_padding: PhysicalWeightPadding,
    },
    /// One opaque, self-contained quantization block represents a fixed
    /// number of logical values along `block_axis`. The bound component shape
    /// is the padded logical shape with that axis divided by the block width.
    BlockQuantized {
        blocks: PhysicalWeightComponentBinding,
        block_axis: u32,
        block_padding: PhysicalWeightPadding,
    },
    /// A contiguous logical subrange on one axis is stored by reshaping that
    /// subrange and permuting the reshape axes. This captures checkpoint
    /// layouts such as grouped-to-tiled head order without a model flag,
    /// synthetic index tensor, or eager repack.
    AxisReshapePermutation {
        values: Box<PhysicalWeightLayout>,
        axis: u32,
        logical_offset: u64,
        extent: u64,
        reshape: Vec<u64>,
        /// Stored axis position -> reshaped logical axis, matching an
        /// n-dimensional transpose/permute order.
        stored_axis_order: Vec<u32>,
    },
    Indexed {
        indices: AxisWeightComponent,
        values: Box<PhysicalWeightLayout>,
        source_axis_extent: u64,
    },
    ExpertStack {
        experts: Vec<PhysicalWeightLayout>,
        expert_axis: u32,
    },
}

impl PhysicalWeightLayout {
    fn normalize(&mut self) {
        match self {
            Self::Composite { parts } => {
                for part in parts.iter_mut() {
                    part.layout.normalize();
                }
                // Offsets make composite placement semantic and order-free.
                // Validation subsequently proves that no two placements
                // overlap, so this order cannot reorder an ordered sequence.
                parts.sort_by(|left, right| {
                    left.logical_offsets
                        .cmp(&right.logical_offsets)
                        .then_with(|| left.extents.cmp(&right.extents))
                });
            }
            Self::AxisReshapePermutation { values, .. } | Self::Indexed { values, .. } => {
                values.normalize()
            }
            Self::ExpertStack { experts, .. } => {
                // Expert vector position is the expert index and is therefore
                // semantic. Normalize descendants without sorting the vector.
                for expert in experts {
                    expert.normalize();
                }
            }
            Self::Dense { .. }
            | Self::Stored { .. }
            | Self::Quantized { .. }
            | Self::BlockQuantized { .. } => {}
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WeightTensorSpec {
    pub id: WeightId,
    pub dimensions: Vec<u64>,
    /// Dtype observed by the semantic program after decoding any physical
    /// packing, indexing, or quantization components.
    pub logical_element_type: ElementType,
    pub physical_layout: PhysicalWeightLayout,
    pub required: bool,
}

impl WeightTensorSpec {
    pub fn logical_elements(&self) -> Result<u64, VNextError> {
        checked_elements(&self.dimensions).ok_or_else(|| VNextError::InvalidExecutionPlan {
            reason: format!("logical weight `{}` element count overflows u64", self.id),
        })
    }

    pub fn logical_bytes(&self) -> Result<u64, VNextError> {
        self.logical_elements()?
            .checked_mul(self.logical_element_type.size_bytes())
            .ok_or_else(|| VNextError::InvalidExecutionPlan {
                reason: format!("logical weight `{}` byte size overflows u64", self.id),
            })
    }
}

/// Provider-visible physical identity for one component of a resolved weight.
/// Source file names are intentionally excluded: source provenance belongs to
/// the prepared family fingerprint, while providers need shape, role, and ABI.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResolvedWeightComponentLayout {
    component_id: WeightId,
    role: WeightComponentRole,
    physical_dimensions: Vec<u64>,
    encoding: WeightEncoding,
}

impl ResolvedWeightComponentLayout {
    fn from_component(component: &WeightComponentSpec) -> Self {
        Self {
            component_id: component.id.clone(),
            role: component.role,
            physical_dimensions: component.dimensions.clone(),
            encoding: component.encoding.clone(),
        }
    }

    fn as_component_spec(&self) -> WeightComponentSpec {
        WeightComponentSpec {
            id: self.component_id.clone(),
            role: self.role,
            external_names: vec![format!("resolved.{}", self.component_id)],
            dimensions: self.physical_dimensions.clone(),
            encoding: self.encoding.clone(),
            required: true,
        }
    }

    pub fn component_id(&self) -> &WeightId {
        &self.component_id
    }

    pub const fn role(&self) -> WeightComponentRole {
        self.role
    }

    pub fn physical_dimensions(&self) -> &[u64] {
        &self.physical_dimensions
    }

    pub fn encoding(&self) -> &WeightEncoding {
        &self.encoding
    }

    pub fn physical_bytes(&self) -> Result<u64, VNextError> {
        self.as_component_spec().physical_bytes()
    }

    pub fn physical_element_type(&self) -> ElementType {
        self.encoding
            .dense_element_type()
            .unwrap_or(ElementType::U8)
    }
}

/// Immutable physical weight contract carried by an execution-plan binding.
/// This prevents the provider boundary from collapsing a quantized/composite
/// layout into only resource ranges and a synthetic `u8` dtype.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ResolvedWeightBinding {
    weight_id: WeightId,
    format_id: WeightFormatId,
    layout_id: WeightLayoutId,
    schema_version: ContractVersion,
    physical_layout: PhysicalWeightLayout,
    components: Vec<ResolvedWeightComponentLayout>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ResolvedWeightBindingWire {
    weight_id: WeightId,
    format_id: WeightFormatId,
    layout_id: WeightLayoutId,
    schema_version: ContractVersion,
    physical_layout: PhysicalWeightLayout,
    components: Vec<ResolvedWeightComponentLayout>,
}

impl<'de> Deserialize<'de> for ResolvedWeightBinding {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = ResolvedWeightBindingWire::deserialize(deserializer)?;
        let binding = Self {
            weight_id: wire.weight_id,
            format_id: wire.format_id,
            layout_id: wire.layout_id,
            schema_version: wire.schema_version,
            physical_layout: wire.physical_layout,
            components: wire.components,
        };
        binding
            .validate_structure()
            .map_err(serde::de::Error::custom)?;
        Ok(binding)
    }
}

impl ResolvedWeightBinding {
    pub fn from_schema(schema: &WeightSchema, weight_id: &WeightId) -> Result<Self, VNextError> {
        let tensor = schema
            .tensor(weight_id)
            .ok_or_else(|| VNextError::InvalidExecutionPlan {
                reason: format!("unknown logical weight `{weight_id}`"),
            })?;
        let mut components = schema
            .physical_component_refs(weight_id)?
            .into_iter()
            .map(ResolvedWeightComponentLayout::from_component)
            .collect::<Vec<_>>();
        components.sort_by(|left, right| left.component_id.cmp(&right.component_id));
        let binding = Self {
            weight_id: weight_id.clone(),
            format_id: schema.format_id.clone(),
            layout_id: schema.layout_id.clone(),
            schema_version: schema.version,
            physical_layout: tensor.physical_layout.clone(),
            components,
        };
        binding.validate_structure()?;
        binding.validate_logical(&tensor.dimensions, tensor.logical_element_type)?;
        Ok(binding)
    }

    fn validate_structure(&self) -> Result<(), VNextError> {
        validate_physical_layout_budget(&self.physical_layout).map_err(|reason| {
            VNextError::InvalidExecutionPlan {
                reason: format!("resolved weight `{}` layout: {reason}", self.weight_id),
            }
        })?;
        let referenced = physical_component_ids(&self.physical_layout).map_err(|reason| {
            VNextError::InvalidExecutionPlan {
                reason: format!("resolved weight `{}` layout: {reason}", self.weight_id),
            }
        })?;
        let component_ids = self
            .components
            .iter()
            .map(|component| component.component_id.clone())
            .collect::<BTreeSet<_>>();
        let canonical_components = self
            .components
            .windows(2)
            .all(|pair| pair[0].component_id < pair[1].component_id);
        if self.schema_version.major == 0
            || self.components.is_empty()
            || !canonical_components
            || component_ids.len() != self.components.len()
            || component_ids != referenced
            || self.components.iter().any(|component| {
                component.physical_dimensions.is_empty()
                    || component
                        .physical_dimensions
                        .iter()
                        .any(|extent| *extent == 0)
                    || component.physical_bytes().is_err()
            })
        {
            return Err(VNextError::InvalidExecutionPlan {
                reason: format!(
                    "resolved weight `{}` physical identity is invalid or non-canonical",
                    self.weight_id
                ),
            });
        }
        Ok(())
    }

    pub fn validate_logical(
        &self,
        logical_dimensions: &[u64],
        logical_element_type: ElementType,
    ) -> Result<(), VNextError> {
        self.validate_structure()?;
        let schema = WeightSchema {
            format_id: self.format_id.clone(),
            layout_id: self.layout_id.clone(),
            version: self.schema_version,
            components: self
                .components
                .iter()
                .map(ResolvedWeightComponentLayout::as_component_spec)
                .collect(),
            tensors: vec![WeightTensorSpec {
                id: self.weight_id.clone(),
                dimensions: logical_dimensions.to_vec(),
                logical_element_type,
                physical_layout: self.physical_layout.clone(),
                required: true,
            }],
        };
        schema.validate(&ModelFamilyId::new("family.resolved-weight-binding")?)
    }

    pub fn weight_id(&self) -> &WeightId {
        &self.weight_id
    }

    pub fn format_id(&self) -> &WeightFormatId {
        &self.format_id
    }

    pub fn layout_id(&self) -> &WeightLayoutId {
        &self.layout_id
    }

    pub const fn schema_version(&self) -> ContractVersion {
        self.schema_version
    }

    pub fn physical_layout(&self) -> &PhysicalWeightLayout {
        &self.physical_layout
    }

    pub fn components(&self) -> &[ResolvedWeightComponentLayout] {
        &self.components
    }

    pub fn quantization_formats(&self) -> BTreeSet<super::QuantizationFormatId> {
        self.components
            .iter()
            .filter_map(|component| match &component.encoding {
                WeightEncoding::Quantized(spec) => Some(spec.format_id.clone()),
                WeightEncoding::BlockQuantized(spec) => Some(spec.format_id.clone()),
                WeightEncoding::Dense { .. } | WeightEncoding::DenseAffine { .. } => None,
            })
            .collect()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WeightSchema {
    pub format_id: WeightFormatId,
    pub layout_id: WeightLayoutId,
    pub version: ContractVersion,
    pub components: Vec<WeightComponentSpec>,
    pub tensors: Vec<WeightTensorSpec>,
}

impl WeightSchema {
    fn normalize(&mut self) {
        self.components
            .sort_by(|left, right| left.id.cmp(&right.id));
        for tensor in &mut self.tensors {
            tensor.physical_layout.normalize();
        }
        self.tensors.sort_by(|left, right| left.id.cmp(&right.id));
    }

    pub fn validate(&self, family_id: &ModelFamilyId) -> Result<(), VNextError> {
        if self.version.major == 0 || self.components.is_empty() || self.tensors.is_empty() {
            return Err(VNextError::UnknownWeightLayout {
                family_id: family_id.to_string(),
                layout_id: self.layout_id.to_string(),
            });
        }
        for tensor in &self.tensors {
            validate_physical_layout_budget(&tensor.physical_layout).map_err(|reason| {
                VNextError::InvalidModelConfig {
                    family_id: family_id.to_string(),
                    field: format!("weight_schema.tensors.{}.physical_layout", tensor.id),
                    reason,
                }
            })?;
        }
        let mut component_ids = BTreeSet::new();
        let mut names = BTreeSet::new();
        let mut components = BTreeMap::new();
        let mut quantization_abis = BTreeMap::new();
        for component in &self.components {
            if !component_ids.insert(component.id.clone())
                || component.external_names.is_empty()
                || component.dimensions.is_empty()
                || component
                    .external_names
                    .iter()
                    .any(|name| name.trim().is_empty() || !names.insert(name.clone()))
                || component.dimensions.iter().any(|extent| *extent == 0)
            {
                return Err(VNextError::InvalidModelConfig {
                    family_id: family_id.to_string(),
                    field: "weight_schema.components".to_owned(),
                    reason: "component identities, names, and dimensions must be valid and unique"
                        .to_owned(),
                });
            }
            if let WeightEncoding::Quantized(quantization) = &component.encoding {
                quantization.validate()?;
            }
            if let WeightEncoding::BlockQuantized(quantization) = &component.encoding {
                quantization.validate()?;
            }
            let quantization_format = match &component.encoding {
                WeightEncoding::Quantized(spec) => Some(&spec.format_id),
                WeightEncoding::BlockQuantized(spec) => Some(&spec.format_id),
                WeightEncoding::Dense { .. } | WeightEncoding::DenseAffine { .. } => None,
            };
            if let Some(format_id) = quantization_format {
                if let Some(existing) = quantization_abis.get(format_id) {
                    if existing != &component.encoding {
                        return Err(VNextError::InvalidModelConfig {
                            family_id: family_id.to_string(),
                            field: "weight_schema.components.encoding".to_owned(),
                            reason: format!(
                                "quantization format `{format_id}` maps to conflicting physical ABIs"
                            ),
                        });
                    }
                } else {
                    quantization_abis.insert(format_id.clone(), component.encoding.clone());
                }
            }
            if let WeightEncoding::DenseAffine { element_type, .. } = &component.encoding {
                if component.role != WeightComponentRole::Values
                    || !matches!(
                        element_type,
                        ElementType::F16 | ElementType::Bf16 | ElementType::F32
                    )
                {
                    return Err(VNextError::InvalidModelConfig {
                        family_id: family_id.to_string(),
                        field: "weight_schema.components.encoding".to_owned(),
                        reason: format!(
                            "affine dense component `{}` must be a floating-point value component",
                            component.id
                        ),
                    });
                }
            }
            component
                .physical_bytes()
                .map_err(|error| VNextError::InvalidModelConfig {
                    family_id: family_id.to_string(),
                    field: "weight_schema.components.dimensions".to_owned(),
                    reason: error.to_string(),
                })?;
            let role_encoding_valid = match component.role {
                WeightComponentRole::Scales => matches!(
                    component.encoding,
                    WeightEncoding::Dense {
                        element_type: ElementType::F16 | ElementType::Bf16 | ElementType::F32
                    }
                ),
                WeightComponentRole::ZeroPoints
                | WeightComponentRole::Indices
                | WeightComponentRole::Permutation => matches!(
                    component.encoding,
                    WeightEncoding::Dense {
                        element_type: ElementType::U8
                            | ElementType::U32
                            | ElementType::I8
                            | ElementType::I32
                    }
                ),
                WeightComponentRole::PackedValues => {
                    matches!(
                        component.encoding,
                        WeightEncoding::Quantized(_) | WeightEncoding::BlockQuantized(_)
                    )
                }
                _ => true,
            };
            if !role_encoding_valid {
                return Err(VNextError::InvalidModelConfig {
                    family_id: family_id.to_string(),
                    field: "weight_schema.components.encoding".to_owned(),
                    reason: format!(
                        "component `{}` encoding is incompatible with its structural role",
                        component.id
                    ),
                });
            }
            components.insert(component.id.clone(), component);
        }

        let mut tensor_ids = BTreeSet::new();
        let mut referenced_components = BTreeSet::new();
        for tensor in &self.tensors {
            if !tensor_ids.insert(tensor.id.clone())
                || tensor.dimensions.is_empty()
                || tensor.dimensions.iter().any(|extent| *extent == 0)
            {
                return Err(VNextError::InvalidModelConfig {
                    family_id: family_id.to_string(),
                    field: "weight_schema.tensors".to_owned(),
                    reason: "logical weight identities and dimensions must be valid and unique"
                        .to_owned(),
                });
            }
            self.validate_physical_layout(
                family_id,
                tensor,
                &components,
                &mut referenced_components,
            )?;
        }
        if let Some(component) = self
            .components
            .iter()
            .find(|component| component.required && !referenced_components.contains(&component.id))
        {
            return Err(VNextError::InvalidModelConfig {
                family_id: family_id.to_string(),
                field: "weight_schema.components".to_owned(),
                reason: format!(
                    "required component `{}` is not referenced by a logical weight",
                    component.id
                ),
            });
        }
        Ok(())
    }

    pub fn quantization_formats(&self) -> BTreeSet<super::QuantizationFormatId> {
        self.components
            .iter()
            .filter_map(|component| match &component.encoding {
                WeightEncoding::Quantized(spec) => Some(spec.format_id.clone()),
                WeightEncoding::BlockQuantized(spec) => Some(spec.format_id.clone()),
                WeightEncoding::Dense { .. } | WeightEncoding::DenseAffine { .. } => None,
            })
            .collect()
    }

    fn validate_physical_layout(
        &self,
        family_id: &ModelFamilyId,
        tensor: &WeightTensorSpec,
        components: &BTreeMap<WeightId, &WeightComponentSpec>,
        referenced: &mut BTreeSet<WeightId>,
    ) -> Result<(), VNextError> {
        let mut validator = PhysicalLayoutValidator {
            family_id,
            tensor_id: &tensor.id,
            components,
            referenced,
            visited_nodes: 0,
        };
        validator.validate_layout(
            &tensor.physical_layout,
            &tensor.dimensions,
            tensor.logical_element_type,
            1,
        )
    }

    pub fn tensor(&self, weight_id: &WeightId) -> Option<&WeightTensorSpec> {
        self.tensors.iter().find(|tensor| &tensor.id == weight_id)
    }

    /// Ordered physical components needed to materialize one logical value.
    /// The order is the schema's component order and is therefore stable for
    /// fingerprints and resource-plan evidence.
    pub fn physical_component_refs(
        &self,
        weight_id: &WeightId,
    ) -> Result<Vec<&WeightComponentSpec>, VNextError> {
        let tensor = self
            .tensor(weight_id)
            .ok_or_else(|| VNextError::InvalidExecutionPlan {
                reason: format!("unknown logical weight `{weight_id}`"),
            })?;
        let required = physical_component_ids(&tensor.physical_layout).map_err(|reason| {
            VNextError::InvalidExecutionPlan {
                reason: format!(
                    "logical weight `{weight_id}` has invalid physical layout: {reason}"
                ),
            }
        })?;
        let result = self
            .components
            .iter()
            .filter(|component| required.contains(&component.id))
            .collect::<Vec<_>>();
        if result.len() != required.len() {
            return Err(VNextError::InvalidExecutionPlan {
                reason: format!("logical weight `{weight_id}` references an unknown component"),
            });
        }
        Ok(result)
    }

    pub fn physical_bytes(&self, weight_id: &WeightId) -> Result<u64, VNextError> {
        self.physical_component_refs(weight_id)?
            .into_iter()
            .try_fold(0_u64, |total, component| {
                total
                    .checked_add(component.physical_bytes()?)
                    .ok_or_else(|| VNextError::InvalidExecutionPlan {
                        reason: format!("logical weight `{weight_id}` physical bytes overflow u64"),
                    })
            })
    }

    pub fn physical_resource_requirements(
        &self,
        weight_id: &WeightId,
    ) -> Result<Vec<PhysicalWeightComponentRef>, VNextError> {
        self.physical_component_refs(weight_id)?
            .into_iter()
            .map(|component| {
                Ok(PhysicalWeightComponentRef {
                    component_id: component.id.clone(),
                    physical_dimensions: component.dimensions.clone(),
                    resource_bytes: component.physical_bytes()?,
                    element_type: component.physical_element_type(),
                })
            })
            .collect()
    }
}

fn push_physical_layout_child<'a>(
    stack: &mut Vec<(&'a PhysicalWeightLayout, usize)>,
    child: &'a PhysicalWeightLayout,
    child_depth: usize,
    visited: usize,
) -> Result<(), String> {
    if visited
        .checked_add(stack.len())
        .is_none_or(|pending| pending >= MAX_PHYSICAL_WEIGHT_LAYOUT_NODES)
    {
        return Err(format!(
            "physical layout node count exceeds {MAX_PHYSICAL_WEIGHT_LAYOUT_NODES}"
        ));
    }
    stack.push((child, child_depth));
    Ok(())
}

fn validate_physical_layout_budget(layout: &PhysicalWeightLayout) -> Result<(), String> {
    let mut stack = vec![(layout, 1_usize)];
    let mut visited = 0_usize;
    while let Some((node, depth)) = stack.pop() {
        if depth > MAX_PHYSICAL_WEIGHT_LAYOUT_DEPTH {
            return Err(format!(
                "physical layout depth exceeds {MAX_PHYSICAL_WEIGHT_LAYOUT_DEPTH}"
            ));
        }
        let direct_bindings = match node {
            PhysicalWeightLayout::Dense { .. } | PhysicalWeightLayout::Stored { .. } => 1,
            PhysicalWeightLayout::Quantized {
                zero_points,
                axis_indices,
                permutation,
                codebook,
                ..
            } => {
                2 + usize::from(zero_points.is_some())
                    + usize::from(axis_indices.is_some())
                    + usize::from(permutation.is_some())
                    + usize::from(codebook.is_some())
            }
            PhysicalWeightLayout::BlockQuantized { .. } => 1,
            PhysicalWeightLayout::AxisReshapePermutation { .. } => 0,
            PhysicalWeightLayout::Indexed { .. } => 1,
            PhysicalWeightLayout::Composite { .. } | PhysicalWeightLayout::ExpertStack { .. } => 0,
        };
        visited = visited
            .checked_add(1 + direct_bindings)
            .ok_or_else(|| "physical layout node count overflows usize".to_owned())?;
        if visited > MAX_PHYSICAL_WEIGHT_LAYOUT_NODES {
            return Err(format!(
                "physical layout node count exceeds {MAX_PHYSICAL_WEIGHT_LAYOUT_NODES}"
            ));
        }
        let child_depth = depth
            .checked_add(1)
            .ok_or_else(|| "physical layout depth overflows usize".to_owned())?;
        match node {
            PhysicalWeightLayout::Composite { parts } => {
                for part in parts {
                    push_physical_layout_child(&mut stack, &part.layout, child_depth, visited)?;
                }
            }
            PhysicalWeightLayout::AxisReshapePermutation { values, .. }
            | PhysicalWeightLayout::Indexed { values, .. } => {
                push_physical_layout_child(&mut stack, values, child_depth, visited)?;
            }
            PhysicalWeightLayout::ExpertStack { experts, .. } => {
                for expert in experts {
                    push_physical_layout_child(&mut stack, expert, child_depth, visited)?;
                }
            }
            PhysicalWeightLayout::Dense { .. }
            | PhysicalWeightLayout::Stored { .. }
            | PhysicalWeightLayout::Quantized { .. }
            | PhysicalWeightLayout::BlockQuantized { .. } => {}
        }
    }
    Ok(())
}

fn physical_component_ids(layout: &PhysicalWeightLayout) -> Result<BTreeSet<WeightId>, String> {
    validate_physical_layout_budget(layout)?;
    let mut ids = BTreeSet::new();
    let mut stack = vec![layout];
    while let Some(node) = stack.pop() {
        let mut insert_binding = |binding: &PhysicalWeightComponentBinding| {
            ids.insert(binding.component_id.clone());
        };
        match node {
            PhysicalWeightLayout::Dense { component_id } => {
                ids.insert(component_id.clone());
            }
            PhysicalWeightLayout::Stored { component } => insert_binding(component),
            PhysicalWeightLayout::Composite { parts } => {
                stack.extend(parts.iter().map(|part| part.layout.as_ref()));
            }
            PhysicalWeightLayout::Quantized {
                packed_values,
                scales,
                zero_points,
                axis_indices,
                permutation,
                codebook,
                ..
            } => {
                insert_binding(packed_values);
                insert_binding(scales);
                if let Some(binding) = zero_points {
                    insert_binding(binding);
                }
                if let Some(axis_component) = axis_indices {
                    insert_binding(&axis_component.component);
                }
                if let Some(axis_component) = permutation {
                    insert_binding(&axis_component.component);
                }
                if let Some(binding) = codebook {
                    insert_binding(binding);
                }
            }
            PhysicalWeightLayout::BlockQuantized { blocks, .. } => insert_binding(blocks),
            PhysicalWeightLayout::AxisReshapePermutation { values, .. } => stack.push(values),
            PhysicalWeightLayout::Indexed {
                indices, values, ..
            } => {
                insert_binding(&indices.component);
                stack.push(values);
            }
            PhysicalWeightLayout::ExpertStack { experts, .. } => {
                stack.extend(experts);
            }
        }
    }
    Ok(ids)
}

fn checked_elements(dimensions: &[u64]) -> Option<u64> {
    dimensions
        .iter()
        .try_fold(1_u64, |elements, extent| elements.checked_mul(*extent))
}

struct PhysicalLayoutValidator<'schema, 'references> {
    family_id: &'schema ModelFamilyId,
    tensor_id: &'schema WeightId,
    components: &'schema BTreeMap<WeightId, &'schema WeightComponentSpec>,
    referenced: &'references mut BTreeSet<WeightId>,
    visited_nodes: usize,
}

impl<'schema, 'references> PhysicalLayoutValidator<'schema, 'references> {
    fn invalid(&self, reason: impl Into<String>) -> VNextError {
        VNextError::InvalidModelConfig {
            family_id: self.family_id.to_string(),
            field: format!("weight_schema.tensors.{}.physical_layout", self.tensor_id),
            reason: reason.into(),
        }
    }

    fn visit_node(&mut self, depth: usize) -> Result<(), VNextError> {
        if depth > MAX_PHYSICAL_WEIGHT_LAYOUT_DEPTH {
            return Err(self.invalid(format!(
                "physical layout depth exceeds {MAX_PHYSICAL_WEIGHT_LAYOUT_DEPTH}"
            )));
        }
        self.visited_nodes = self
            .visited_nodes
            .checked_add(1)
            .ok_or_else(|| self.invalid("physical layout node count overflows usize"))?;
        if self.visited_nodes > MAX_PHYSICAL_WEIGHT_LAYOUT_NODES {
            return Err(self.invalid(format!(
                "physical layout node count exceeds {MAX_PHYSICAL_WEIGHT_LAYOUT_NODES}"
            )));
        }
        Ok(())
    }

    fn component(
        &self,
        component_id: &WeightId,
    ) -> Result<&'schema WeightComponentSpec, VNextError> {
        self.components
            .get(component_id)
            .copied()
            .ok_or_else(|| self.invalid(format!("unknown component `{component_id}`")))
    }

    fn bind_component(
        &mut self,
        binding: &PhysicalWeightComponentBinding,
        semantic_dimensions: &[u64],
        role: WeightComponentRole,
        depth: usize,
    ) -> Result<&'schema WeightComponentSpec, VNextError> {
        self.visit_node(depth)?;
        let component = self.component(&binding.component_id)?;
        if component.role != role {
            return Err(self.invalid(format!(
                "component `{}` has role {:?}, expected {:?}",
                component.id, component.role, role
            )));
        }
        self.validate_storage(component, semantic_dimensions, &binding.storage)?;
        if !self.referenced.insert(component.id.clone()) {
            return Err(self.invalid(format!(
                "component `{}` is referenced more than once in the physical layout tree",
                component.id
            )));
        }
        Ok(component)
    }

    fn validate_storage(
        &self,
        component: &WeightComponentSpec,
        semantic_dimensions: &[u64],
        storage: &PhysicalStorageLayout,
    ) -> Result<(), VNextError> {
        if semantic_dimensions.is_empty()
            || semantic_dimensions.iter().any(|extent| *extent == 0)
            || checked_elements(semantic_dimensions).is_none()
        {
            return Err(self.invalid(format!(
                "component `{}` has an invalid or overflowing semantic shape",
                component.id
            )));
        }
        let raw_elements = checked_elements(&component.dimensions).ok_or_else(|| {
            self.invalid(format!(
                "component `{}` raw storage shape overflows u64",
                component.id
            ))
        })?;
        match storage {
            PhysicalStorageLayout::Contiguous { padding } => {
                let padded = self.resolve_padding(semantic_dimensions, padding)?;
                if component.dimensions != padded {
                    return Err(self.invalid(format!(
                        "component `{}` contiguous shape {:?} differs from its explicit physical shape {:?}",
                        component.id, padded, component.dimensions
                    )));
                }
            }
            PhysicalStorageLayout::Strided {
                strides_in_elements,
                padding,
            } => {
                let padded = self.resolve_padding(semantic_dimensions, padding)?;
                let span = self.checked_strided_span(&padded, strides_in_elements, 1)?;
                if span != raw_elements {
                    return Err(self.invalid(format!(
                        "component `{}` strided span {span} differs from its raw storage element count {raw_elements}",
                        component.id
                    )));
                }
            }
            PhysicalStorageLayout::Tiled {
                tile_shape,
                axis_order,
                tile_strides_in_elements,
                padding,
            } => {
                let rank = semantic_dimensions.len();
                if tile_shape.len() != rank
                    || tile_shape.iter().any(|extent| *extent == 0)
                    || !is_axis_permutation(axis_order, rank)
                    || tile_strides_in_elements.len() != rank
                {
                    return Err(self.invalid(format!(
                        "component `{}` tile shape, axis order, or strides do not match rank",
                        component.id
                    )));
                }
                let padded = self.resolve_padding(semantic_dimensions, padding)?;
                let minimal_padded = semantic_dimensions
                    .iter()
                    .zip(tile_shape)
                    .map(|(extent, tile)| checked_round_up(*extent, *tile))
                    .collect::<Option<Vec<_>>>()
                    .ok_or_else(|| {
                        self.invalid(format!(
                            "component `{}` tile padding overflows u64",
                            component.id
                        ))
                    })?;
                match padding {
                    PhysicalWeightPadding::Exact if minimal_padded != semantic_dimensions => {
                        return Err(self.invalid(format!(
                            "component `{}` needs tile padding but declares exact storage",
                            component.id
                        )));
                    }
                    PhysicalWeightPadding::ZeroFill { .. } if padded != minimal_padded => {
                        return Err(self.invalid(format!(
                            "component `{}` tiled zero-fill shape is not the unique minimal padded shape",
                            component.id
                        )));
                    }
                    _ => {}
                }
                let semantic_grid = padded
                    .iter()
                    .zip(tile_shape)
                    .map(|(extent, tile)| extent / tile)
                    .collect::<Vec<_>>();
                let physical_grid = axis_order
                    .iter()
                    .map(|axis| semantic_grid[*axis as usize])
                    .collect::<Vec<_>>();
                let tile_elements = checked_elements(tile_shape).ok_or_else(|| {
                    self.invalid(format!(
                        "component `{}` tile size overflows u64",
                        component.id
                    ))
                })?;
                let span = self.checked_strided_span(
                    &physical_grid,
                    tile_strides_in_elements,
                    tile_elements,
                )?;
                if span != raw_elements {
                    return Err(self.invalid(format!(
                        "component `{}` tiled span {span} differs from its raw storage element count {raw_elements}",
                        component.id
                    )));
                }
            }
        }
        Ok(())
    }

    fn resolve_padding(
        &self,
        semantic_dimensions: &[u64],
        padding: &PhysicalWeightPadding,
    ) -> Result<Vec<u64>, VNextError> {
        match padding {
            PhysicalWeightPadding::Exact => Ok(semantic_dimensions.to_vec()),
            PhysicalWeightPadding::ZeroFill { padded_dimensions } => {
                if padded_dimensions.len() != semantic_dimensions.len()
                    || padded_dimensions.iter().any(|extent| *extent == 0)
                    || padded_dimensions
                        .iter()
                        .zip(semantic_dimensions)
                        .any(|(padded, semantic)| padded < semantic)
                    || padded_dimensions == semantic_dimensions
                    || checked_elements(padded_dimensions).is_none()
                {
                    return Err(self.invalid(
                        "zero-fill padding must explicitly enlarge a valid shape without shrinking any axis",
                    ));
                }
                Ok(padded_dimensions.clone())
            }
        }
    }

    fn checked_strided_span(
        &self,
        dimensions: &[u64],
        strides: &[u64],
        base_span: u64,
    ) -> Result<u64, VNextError> {
        if dimensions.is_empty()
            || dimensions.len() != strides.len()
            || dimensions.iter().any(|extent| *extent == 0)
            || strides.iter().any(|stride| *stride == 0)
            || base_span == 0
        {
            return Err(self.invalid("strided storage dimensions and strides are invalid"));
        }
        let mut axes = dimensions
            .iter()
            .copied()
            .zip(strides.iter().copied())
            .filter(|(extent, _)| *extent > 1)
            .collect::<Vec<_>>();
        axes.sort_by_key(|(_, stride)| *stride);
        let mut span = base_span;
        for (extent, stride) in axes {
            if stride < span {
                return Err(
                    self.invalid("strided storage aliases coordinates or overlaps physical tiles")
                );
            }
            span = extent
                .checked_sub(1)
                .and_then(|count| count.checked_mul(stride))
                .and_then(|addition| span.checked_add(addition))
                .ok_or_else(|| self.invalid("strided storage span overflows u64"))?;
        }
        Ok(span)
    }

    fn grouped_dimensions(
        &self,
        semantic_dimensions: &[u64],
        padding: &PhysicalWeightPadding,
        group_axis: usize,
        group_size: u64,
    ) -> Result<Vec<u64>, VNextError> {
        let axis_extent = semantic_dimensions[group_axis];
        let minimal_axis = checked_round_up(axis_extent, group_size)
            .ok_or_else(|| self.invalid("quantization group padding overflows u64"))?;
        match padding {
            PhysicalWeightPadding::Exact => {
                if minimal_axis != axis_extent {
                    return Err(self.invalid(
                        "quantization groups require padding but exact storage was declared",
                    ));
                }
                Ok(semantic_dimensions.to_vec())
            }
            PhysicalWeightPadding::ZeroFill { padded_dimensions } => {
                if minimal_axis == axis_extent
                    || padded_dimensions.len() != semantic_dimensions.len()
                    || padded_dimensions.iter().enumerate().any(|(axis, extent)| {
                        if axis == group_axis {
                            *extent != minimal_axis
                        } else {
                            *extent != semantic_dimensions[axis]
                        }
                    })
                {
                    return Err(self.invalid(
                        "quantization zero-fill must pad only the group axis to its unique minimal extent",
                    ));
                }
                checked_elements(padded_dimensions)
                    .is_some()
                    .then(|| padded_dimensions.clone())
                    .ok_or_else(|| self.invalid("quantization padded shape overflows u64"))
            }
        }
    }

    fn validate_dense_values(
        &mut self,
        binding: &PhysicalWeightComponentBinding,
        semantic_dimensions: &[u64],
        logical_element_type: ElementType,
        depth: usize,
    ) -> Result<(), VNextError> {
        let component = self.bind_component(
            binding,
            semantic_dimensions,
            WeightComponentRole::Values,
            depth,
        )?;
        if component.dense_element_type() != Some(logical_element_type) {
            return Err(self.invalid(format!(
                "values component `{}` dtype differs from the logical tensor",
                component.id
            )));
        }
        Ok(())
    }

    fn validate_axis_component(
        &mut self,
        axis_component: &AxisWeightComponent,
        semantic_dimensions: &[u64],
        expected_axis: usize,
        role: WeightComponentRole,
        allow_narrow_integer: bool,
        depth: usize,
    ) -> Result<(), VNextError> {
        if axis_component.axis as usize != expected_axis {
            return Err(self.invalid(format!(
                "axis component `{}` targets axis {}, expected {expected_axis}",
                axis_component.component.component_id, axis_component.axis
            )));
        }
        let axis_shape = [semantic_dimensions[expected_axis]];
        let component = self.bind_component(&axis_component.component, &axis_shape, role, depth)?;
        let integer_type_valid = component.dense_element_type().is_some_and(|element_type| {
            if allow_narrow_integer {
                matches!(
                    element_type,
                    ElementType::U8 | ElementType::U32 | ElementType::I8 | ElementType::I32
                )
            } else {
                matches!(element_type, ElementType::U32 | ElementType::I32)
            }
        });
        if !integer_type_valid {
            return Err(self.invalid(format!(
                "axis component `{}` must use an integer encoding valid for {:?}",
                component.id, role
            )));
        }
        Ok(())
    }

    fn validate_layout(
        &mut self,
        layout: &PhysicalWeightLayout,
        semantic_dimensions: &[u64],
        logical_element_type: ElementType,
        depth: usize,
    ) -> Result<(), VNextError> {
        self.visit_node(depth)?;
        if semantic_dimensions.is_empty()
            || semantic_dimensions.iter().any(|extent| *extent == 0)
            || checked_elements(semantic_dimensions).is_none()
        {
            return Err(self.invalid("logical layout shape is empty, zero, or overflowing"));
        }
        match layout {
            PhysicalWeightLayout::Dense { component_id } => {
                let binding =
                    PhysicalWeightComponentBinding::exact_contiguous(component_id.clone());
                self.validate_dense_values(
                    &binding,
                    semantic_dimensions,
                    logical_element_type,
                    depth,
                )?;
            }
            PhysicalWeightLayout::Stored { component } => {
                self.validate_dense_values(
                    component,
                    semantic_dimensions,
                    logical_element_type,
                    depth,
                )?;
            }
            PhysicalWeightLayout::Composite { parts } => {
                if parts.is_empty() {
                    return Err(self.invalid("composite layout has no parts"));
                }
                let rank = semantic_dimensions.len();
                let mut covered_elements = 0_u64;
                for (index, part) in parts.iter().enumerate() {
                    if part.logical_offsets.len() != rank
                        || part.extents.len() != rank
                        || part.extents.iter().any(|extent| *extent == 0)
                        || part
                            .logical_offsets
                            .iter()
                            .zip(&part.extents)
                            .zip(semantic_dimensions)
                            .any(|((offset, extent), logical)| {
                                offset.checked_add(*extent).is_none_or(|end| end > *logical)
                            })
                    {
                        return Err(self.invalid(format!(
                            "composite part {index} has invalid semantic offsets or extents"
                        )));
                    }
                    for previous in &parts[..index] {
                        let overlaps = part
                            .logical_offsets
                            .iter()
                            .zip(&part.extents)
                            .zip(previous.logical_offsets.iter().zip(&previous.extents))
                            .all(|((offset, extent), (other_offset, other_extent))| {
                                offset.checked_add(*extent).is_some_and(|end| {
                                    other_offset.checked_add(*other_extent).is_some_and(
                                        |other_end| *offset < other_end && *other_offset < end,
                                    )
                                })
                            });
                        if overlaps {
                            return Err(self.invalid("composite semantic placements overlap"));
                        }
                    }
                    let part_elements = checked_elements(&part.extents)
                        .ok_or_else(|| self.invalid("composite part size overflows u64"))?;
                    covered_elements = covered_elements
                        .checked_add(part_elements)
                        .ok_or_else(|| self.invalid("composite coverage overflows u64"))?;
                    self.validate_layout(
                        &part.layout,
                        &part.extents,
                        logical_element_type,
                        depth + 1,
                    )?;
                }
                if covered_elements != checked_elements(semantic_dimensions).unwrap() {
                    return Err(self.invalid(
                        "composite semantic placements do not cover the logical tensor exactly",
                    ));
                }
            }
            PhysicalWeightLayout::Quantized {
                packed_values,
                packed_dimensions,
                scales,
                zero_points,
                axis_indices,
                permutation,
                codebook,
                group_axis,
                group_padding,
            } => {
                if !matches!(
                    logical_element_type,
                    ElementType::F16 | ElementType::Bf16 | ElementType::F32
                ) {
                    return Err(
                        self.invalid("quantized logical weight dtype must be floating point")
                    );
                }
                let axis = *group_axis as usize;
                if axis >= semantic_dimensions.len() {
                    return Err(self.invalid("quantization group axis is out of range"));
                }
                let quantization = {
                    let component = self.component(&packed_values.component_id)?;
                    let WeightEncoding::Quantized(spec) = &component.encoding else {
                        return Err(self.invalid(
                            "packed-values component does not carry a quantization spec",
                        ));
                    };
                    spec.clone()
                };
                let grouped_dimensions = self.grouped_dimensions(
                    semantic_dimensions,
                    group_padding,
                    axis,
                    u64::from(quantization.group_size),
                )?;
                let packed_bytes = checked_elements(&grouped_dimensions)
                    .and_then(|elements| {
                        elements.checked_mul(u64::from(quantization.bits_per_weight))
                    })
                    .and_then(|bits| bits.checked_add(7))
                    .map(|bits| bits / 8)
                    .ok_or_else(|| self.invalid("packed-values size overflows u64"))?;
                if checked_elements(packed_dimensions) != Some(packed_bytes) {
                    return Err(self.invalid(format!(
                        "packed-values semantic shape contains {} storage bytes, expected {packed_bytes}",
                        checked_elements(packed_dimensions)
                            .map_or_else(|| "an overflowing number of".to_owned(), |value| value.to_string())
                    )));
                }
                let packed = self.bind_component(
                    packed_values,
                    packed_dimensions,
                    WeightComponentRole::PackedValues,
                    depth,
                )?;
                if packed.encoding != WeightEncoding::Quantized(quantization.clone()) {
                    return Err(self.invalid(
                        "packed-values encoding changed while validating the quantized tree",
                    ));
                }

                let mut group_shape = grouped_dimensions;
                group_shape[axis] /= u64::from(quantization.group_size);
                let scales_component =
                    self.bind_component(scales, &group_shape, WeightComponentRole::Scales, depth)?;
                if scales_component.dense_element_type() != Some(quantization.scale_type) {
                    return Err(
                        self.invalid("scale component dtype differs from the quantization spec")
                    );
                }
                match (quantization.zero_point_type, zero_points) {
                    (Some(expected_type), Some(binding)) => {
                        let component = self.bind_component(
                            binding,
                            &group_shape,
                            WeightComponentRole::ZeroPoints,
                            depth,
                        )?;
                        if component.dense_element_type() != Some(expected_type) {
                            return Err(self.invalid(
                                "zero-point component dtype differs from the quantization spec",
                            ));
                        }
                    }
                    (None, None) => {}
                    _ => {
                        return Err(self.invalid(
                            "zero-point component presence differs from the quantization spec",
                        ));
                    }
                }
                if let Some(axis_indices) = axis_indices {
                    self.validate_axis_component(
                        axis_indices,
                        semantic_dimensions,
                        axis,
                        WeightComponentRole::Indices,
                        true,
                        depth,
                    )?;
                }
                if let Some(permutation) = permutation {
                    self.validate_axis_component(
                        permutation,
                        semantic_dimensions,
                        axis,
                        WeightComponentRole::Permutation,
                        false,
                        depth,
                    )?;
                }
                if let Some(codebook) = codebook {
                    let entries = 1_u64
                        .checked_shl(u32::from(quantization.bits_per_weight))
                        .ok_or_else(|| self.invalid("codebook size overflows u64"))?;
                    let component = self.bind_component(
                        codebook,
                        &[entries],
                        WeightComponentRole::Codebook,
                        depth,
                    )?;
                    if component.dense_element_type() != Some(logical_element_type) {
                        return Err(
                            self.invalid("codebook dtype differs from the logical tensor dtype")
                        );
                    }
                }
            }
            PhysicalWeightLayout::BlockQuantized {
                blocks,
                block_axis,
                block_padding,
            } => {
                if !matches!(
                    logical_element_type,
                    ElementType::F16 | ElementType::Bf16 | ElementType::F32
                ) {
                    return Err(
                        self.invalid("block-quantized logical weight dtype must be floating point")
                    );
                }
                let axis = *block_axis as usize;
                if axis >= semantic_dimensions.len() {
                    return Err(self.invalid("block quantization axis is out of range"));
                }
                let quantization = {
                    let component = self.component(&blocks.component_id)?;
                    let WeightEncoding::BlockQuantized(spec) = &component.encoding else {
                        return Err(self
                            .invalid("block component does not carry a block quantization spec"));
                    };
                    spec.clone()
                };
                let mut block_dimensions = self.grouped_dimensions(
                    semantic_dimensions,
                    block_padding,
                    axis,
                    u64::from(quantization.logical_values_per_block),
                )?;
                block_dimensions[axis] /= u64::from(quantization.logical_values_per_block);
                let component = self.bind_component(
                    blocks,
                    &block_dimensions,
                    WeightComponentRole::PackedValues,
                    depth,
                )?;
                if component.encoding != WeightEncoding::BlockQuantized(quantization) {
                    return Err(
                        self.invalid("block encoding changed while validating the physical layout")
                    );
                }
            }
            PhysicalWeightLayout::AxisReshapePermutation {
                values,
                axis,
                logical_offset,
                extent,
                reshape,
                stored_axis_order,
            } => {
                let axis = *axis as usize;
                let end = logical_offset.checked_add(*extent);
                let reshape_rank = reshape.len();
                let order_is_permutation = stored_axis_order.len() == reshape_rank
                    && stored_axis_order
                        .iter()
                        .all(|axis| (*axis as usize) < reshape_rank)
                    && stored_axis_order
                        .iter()
                        .copied()
                        .collect::<BTreeSet<_>>()
                        .len()
                        == reshape_rank;
                let order_is_identity = stored_axis_order
                    .iter()
                    .filter(|stored| reshape[**stored as usize] > 1)
                    .copied()
                    .eq(reshape
                        .iter()
                        .enumerate()
                        .filter_map(|(axis, extent)| (*extent > 1).then_some(axis as u32)));
                if axis >= semantic_dimensions.len()
                    || *extent == 0
                    || end.is_none_or(|end| end > semantic_dimensions[axis])
                    || reshape_rank < 2
                    || reshape.iter().any(|dimension| *dimension == 0)
                    || checked_elements(reshape) != Some(*extent)
                    || !order_is_permutation
                    || order_is_identity
                {
                    return Err(self.invalid(
                        "axis reshape permutation has invalid range, shape, or stored axis order",
                    ));
                }
                self.validate_layout(values, semantic_dimensions, logical_element_type, depth + 1)?;
            }
            PhysicalWeightLayout::Indexed {
                indices,
                values,
                source_axis_extent,
            } => {
                let axis = indices.axis as usize;
                if axis >= semantic_dimensions.len() || *source_axis_extent == 0 {
                    return Err(self.invalid("indexed layout axis or source extent is invalid"));
                }
                self.validate_axis_component(
                    indices,
                    semantic_dimensions,
                    axis,
                    WeightComponentRole::Indices,
                    true,
                    depth,
                )?;
                let mut source_dimensions = semantic_dimensions.to_vec();
                source_dimensions[axis] = *source_axis_extent;
                checked_elements(&source_dimensions)
                    .ok_or_else(|| self.invalid("indexed source semantic shape overflows u64"))?;
                self.validate_layout(values, &source_dimensions, logical_element_type, depth + 1)?;
            }
            PhysicalWeightLayout::ExpertStack {
                experts,
                expert_axis,
            } => {
                let axis = *expert_axis as usize;
                if axis >= semantic_dimensions.len() {
                    return Err(self.invalid("expert stack axis is out of range"));
                }
                let expected_count = usize::try_from(semantic_dimensions[axis]).map_err(|_| {
                    self.invalid("expert stack count does not fit the platform usize")
                })?;
                if experts.is_empty() || experts.len() != expected_count {
                    return Err(self
                        .invalid("expert stack child count differs from its logical expert axis"));
                }
                let mut expert_dimensions = semantic_dimensions.to_vec();
                expert_dimensions.remove(axis);
                if expert_dimensions.is_empty() {
                    return Err(self.invalid(
                        "expert stack children must retain at least one tensor dimension",
                    ));
                }
                for expert in experts {
                    self.validate_layout(
                        expert,
                        &expert_dimensions,
                        logical_element_type,
                        depth + 1,
                    )?;
                }
            }
        }
        Ok(())
    }
}

fn is_axis_permutation(axis_order: &[u32], rank: usize) -> bool {
    axis_order.len() == rank
        && axis_order.iter().all(|axis| (*axis as usize) < rank)
        && axis_order.iter().copied().collect::<BTreeSet<_>>().len() == rank
}

fn checked_round_up(extent: u64, multiple: u64) -> Option<u64> {
    if extent == 0 || multiple == 0 {
        return None;
    }
    extent
        .checked_add(multiple.checked_sub(1)?)
        .map(|rounded| rounded / multiple * multiple)
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProgramTensorSpec {
    pub dimensions: Vec<u64>,
    pub element_type: ElementType,
    pub layout: ResolvedTensorLayout,
}

impl ProgramTensorSpec {
    pub fn validate(&self, field: &str) -> Result<(), VNextError> {
        super::ResolvedTensorSpec::new(
            self.dimensions.clone(),
            self.element_type,
            self.layout.clone(),
        )
        .map(|_| ())
        .map_err(|error| VNextError::InvalidExecutionPlan {
            reason: format!("{field} is invalid: {error}"),
        })
    }

    pub fn byte_len(&self) -> Result<u64, VNextError> {
        checked_elements(&self.dimensions)
            .and_then(|elements| elements.checked_mul(self.element_type.size_bytes()))
            .ok_or_else(|| VNextError::InvalidExecutionPlan {
                reason: "program tensor byte size overflows u64".to_owned(),
            })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WeightReference {
    pub weight_id: WeightId,
    pub value_id: ProgramValueId,
    pub tensor: ProgramTensorSpec,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct StateSpec {
    pub id: StateId,
    pub value_id: ProgramValueId,
    /// Logical tensor view consumed by one operation. It does not select a
    /// physical allocator, addressability profile, or backing pool.
    pub tensor: ProgramTensorSpec,
    pub lifetime: StateLifetime,
    pub capacity_demand: StateCapacityDemand,
    /// Initial contents required when a new logical state scope acquires
    /// physical backing. This is semantic model state, not an allocator hint.
    pub initialization: StateInitialization,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct StateSpecWire {
    id: StateId,
    value_id: ProgramValueId,
    tensor: ProgramTensorSpec,
    lifetime: StateLifetime,
    capacity_demand: StateCapacityDemand,
    initialization: StateInitialization,
}

impl<'de> Deserialize<'de> for StateSpec {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = StateSpecWire::deserialize(deserializer)?;
        wire.tensor
            .validate("state_spec.tensor")
            .and_then(|()| wire.capacity_demand.validate(wire.tensor.byte_len()?))
            .map_err(serde::de::Error::custom)?;
        Ok(Self {
            id: wire.id,
            value_id: wire.value_id,
            tensor: wire.tensor,
            lifetime: wire.lifetime,
            capacity_demand: wire.capacity_demand,
            initialization: wire.initialization,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StateInitialization {
    /// Core does not initialize the backing. The first consumer must fully
    /// define every byte before it is read.
    None,
    /// Core zeroes each exact Sequence backing acquisition before its first
    /// state consumer in the same ordered submission. Request-scope zeroing is
    /// reserved until initialization dependencies can defer sibling sequences.
    Zero,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StateLifetime {
    Request,
    Sequence,
    Step,
}

/// Backend-neutral capacity formula for semantic state. This deliberately says
/// nothing about pages, blocks, allocator kind, or provider-visible regions.
/// Concrete physical storage is selected only while building an execution
/// plan from provider requirements, runtime offers, and typed policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum StateCapacityDemand {
    FixedPerScope,
    TokenScaled {
        bytes_per_token: u64,
        maximum_tokens: u64,
    },
}

#[derive(Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
enum StateCapacityDemandWire {
    FixedPerScope,
    TokenScaled {
        bytes_per_token: u64,
        maximum_tokens: u64,
    },
}

impl<'de> Deserialize<'de> for StateCapacityDemand {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let demand = match StateCapacityDemandWire::deserialize(deserializer)? {
            StateCapacityDemandWire::FixedPerScope => Self::FixedPerScope,
            StateCapacityDemandWire::TokenScaled {
                bytes_per_token,
                maximum_tokens,
            } => Self::TokenScaled {
                bytes_per_token,
                maximum_tokens,
            },
        };
        demand.validate(1).map_err(serde::de::Error::custom)?;
        Ok(demand)
    }
}

impl StateCapacityDemand {
    pub fn validate(self, tensor_minimum_bytes: u64) -> Result<(), VNextError> {
        let valid = match self {
            Self::FixedPerScope => tensor_minimum_bytes > 0,
            Self::TokenScaled {
                bytes_per_token,
                maximum_tokens,
            } => {
                bytes_per_token >= tensor_minimum_bytes
                    && maximum_tokens > 0
                    && bytes_per_token.checked_mul(maximum_tokens).is_some()
            }
        };
        if !valid {
            return Err(VNextError::InvalidExecutionPlan {
                reason: "state resource demand is zero, smaller than its tensor, or overflows u64"
                    .to_owned(),
            });
        }
        Ok(())
    }

    pub fn minimum_bytes(self, tensor_minimum_bytes: u64) -> Result<u64, VNextError> {
        self.validate(tensor_minimum_bytes)?;
        Ok(match self {
            Self::FixedPerScope => tensor_minimum_bytes,
            Self::TokenScaled {
                bytes_per_token, ..
            } => bytes_per_token,
        })
    }

    pub fn theoretical_bytes(self, tensor_minimum_bytes: u64) -> Result<u64, VNextError> {
        self.validate(tensor_minimum_bytes)?;
        match self {
            Self::FixedPerScope => Ok(tensor_minimum_bytes),
            Self::TokenScaled {
                bytes_per_token,
                maximum_tokens,
            } => bytes_per_token.checked_mul(maximum_tokens).ok_or_else(|| {
                VNextError::InvalidExecutionPlan {
                    reason: "token-scaled state demand overflows u64".to_owned(),
                }
            }),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
pub struct CanonicalRational {
    numerator: i64,
    denominator: u64,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CanonicalRationalWire {
    numerator: i64,
    denominator: u64,
}

impl<'de> Deserialize<'de> for CanonicalRational {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = CanonicalRationalWire::deserialize(deserializer)?;
        Self::new(wire.numerator, wire.denominator).map_err(serde::de::Error::custom)
    }
}

impl CanonicalRational {
    pub fn new(numerator: i64, denominator: u64) -> Result<Self, VNextError> {
        if denominator == 0 {
            return Err(VNextError::InvalidExecutionPlan {
                reason: "rational denominator must be non-zero".to_owned(),
            });
        }
        let divisor = gcd_u64(numerator.unsigned_abs(), denominator);
        let denominator = denominator / divisor;
        let reduced = i128::from(numerator) / i128::from(divisor);
        let numerator = i64::try_from(reduced).map_err(|_| VNextError::InvalidExecutionPlan {
            reason: "canonical rational numerator overflows i64".to_owned(),
        })?;
        Ok(Self {
            numerator,
            denominator,
        })
    }

    /// Parses a finite base-10 decimal or scientific-notation value without
    /// routing through binary floating point.
    pub fn from_decimal_str(raw: &str) -> Result<Self, VNextError> {
        let normalized = raw.to_ascii_lowercase();
        let (mantissa, exponent) = match normalized.split_once('e') {
            Some((mantissa, exponent)) => (
                mantissa,
                exponent.parse::<i32>().map_err(|error| {
                    invalid_decimal_rational(format!("invalid decimal exponent: {error}"))
                })?,
            ),
            None => (normalized.as_str(), 0),
        };
        let (negative, mantissa) = if let Some(unsigned) = mantissa.strip_prefix('-') {
            (true, unsigned)
        } else {
            (false, mantissa.strip_prefix('+').unwrap_or(mantissa))
        };
        let (whole, fraction) = mantissa.split_once('.').unwrap_or((mantissa, ""));
        if whole.is_empty()
            || !whole.bytes().all(|byte| byte.is_ascii_digit())
            || !fraction.bytes().all(|byte| byte.is_ascii_digit())
        {
            return Err(invalid_decimal_rational(format!(
                "invalid decimal rational {raw:?}"
            )));
        }

        let digits = format!("{whole}{fraction}");
        let mut magnitude = digits.parse::<u128>().map_err(|error| {
            invalid_decimal_rational(format!("decimal numerator overflows: {error}"))
        })?;
        let fractional_digits = i32::try_from(fraction.len()).map_err(|_| {
            invalid_decimal_rational("decimal rational has too many fractional digits")
        })?;
        let scale = fractional_digits
            .checked_sub(exponent)
            .ok_or_else(|| invalid_decimal_rational("decimal rational exponent overflows"))?;
        let denominator = if scale >= 0 {
            10_u128
                .checked_pow(scale as u32)
                .ok_or_else(|| invalid_decimal_rational("decimal rational denominator overflows"))?
        } else {
            magnitude = magnitude
                .checked_mul(10_u128.checked_pow(scale.unsigned_abs()).ok_or_else(|| {
                    invalid_decimal_rational("decimal rational numerator scale overflows")
                })?)
                .ok_or_else(|| invalid_decimal_rational("decimal rational numerator overflows"))?;
            1
        };
        let signed = if negative {
            -(i128::try_from(magnitude)
                .map_err(|_| invalid_decimal_rational("decimal rational numerator exceeds i128"))?)
        } else {
            i128::try_from(magnitude)
                .map_err(|_| invalid_decimal_rational("decimal rational numerator exceeds i128"))?
        };
        let numerator = i64::try_from(signed)
            .map_err(|_| invalid_decimal_rational("decimal rational numerator exceeds i64"))?;
        let denominator = u64::try_from(denominator)
            .map_err(|_| invalid_decimal_rational("decimal rational denominator exceeds u64"))?;
        Self::new(numerator, denominator)
    }

    pub const fn numerator(self) -> i64 {
        self.numerator
    }

    pub const fn denominator(self) -> u64 {
        self.denominator
    }
}

fn invalid_decimal_rational(reason: impl Into<String>) -> VNextError {
    VNextError::InvalidExecutionPlan {
        reason: reason.into(),
    }
}

fn gcd_u64(mut left: u64, mut right: u64) -> u64 {
    while right != 0 {
        let remainder = left % right;
        left = right;
        right = remainder;
    }
    left.max(1)
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SemanticValue {
    Bool(bool),
    Integer(i64),
    Unsigned(u64),
    Rational(CanonicalRational),
    Text(String),
    Integers(Vec<i64>),
}

impl SemanticValue {
    pub const fn kind(&self) -> AttributeValueKind {
        match self {
            Self::Bool(_) => AttributeValueKind::Bool,
            Self::Integer(_) => AttributeValueKind::Integer,
            Self::Unsigned(_) => AttributeValueKind::Unsigned,
            Self::Rational(_) => AttributeValueKind::Rational,
            Self::Text(_) => AttributeValueKind::Text,
            Self::Integers(_) => AttributeValueKind::Integers,
        }
    }

    pub fn validate(&self, context: &str) -> Result<(), VNextError> {
        match self {
            Self::Text(value) if value.is_empty() => Err(VNextError::InvalidExecutionPlan {
                reason: format!("{context} contains an empty text attribute"),
            }),
            Self::Integers(values) if values.is_empty() => Err(VNextError::InvalidExecutionPlan {
                reason: format!("{context} contains an empty integer-list attribute"),
            }),
            _ => Ok(()),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProgramNodeWorkSpec {
    Fixed,
    Tokens { value_id: ProgramValueId, axis: u32 },
}

impl ProgramNodeWorkSpec {
    pub fn tokens(value_id: ProgramValueId, axis: u32) -> Self {
        Self::Tokens { value_id, axis }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProgramNode {
    pub id: NodeId,
    pub operation_id: OperationId,
    pub required_version: ContractVersion,
    pub work: ProgramNodeWorkSpec,
    pub inputs: Vec<ProgramValueId>,
    pub outputs: Vec<ProgramValueId>,
    pub attributes: BTreeMap<AttributeId, SemanticValue>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProgramBlock {
    pub id: String,
    pub nodes: Vec<ProgramNode>,
}

/// Backend-free semantic program for a model family.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ModelProgram {
    family_id: ModelFamilyId,
    inputs: Vec<ProgramValueId>,
    blocks: Vec<ProgramBlock>,
    states: Vec<StateSpec>,
    weights: Vec<WeightReference>,
    outputs: Vec<ProgramValueId>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ModelProgramWire {
    family_id: ModelFamilyId,
    inputs: Vec<ProgramValueId>,
    blocks: Vec<ProgramBlock>,
    states: Vec<StateSpec>,
    weights: Vec<WeightReference>,
    outputs: Vec<ProgramValueId>,
}

impl<'de> Deserialize<'de> for ModelProgram {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = ModelProgramWire::deserialize(deserializer)?;
        Self::new(
            wire.family_id,
            wire.inputs,
            wire.blocks,
            wire.states,
            wire.weights,
            wire.outputs,
        )
        .map_err(serde::de::Error::custom)
    }
}

impl ModelProgram {
    pub fn new(
        family_id: ModelFamilyId,
        inputs: Vec<ProgramValueId>,
        blocks: Vec<ProgramBlock>,
        mut states: Vec<StateSpec>,
        mut weights: Vec<WeightReference>,
        outputs: Vec<ProgramValueId>,
    ) -> Result<Self, VNextError> {
        if blocks.is_empty() {
            return Err(VNextError::InvalidModelConfig {
                family_id: family_id.to_string(),
                field: "program.blocks".to_owned(),
                reason: "at least one block is required".to_owned(),
            });
        }
        let mut known_values = BTreeSet::new();
        if inputs.is_empty()
            || inputs
                .iter()
                .any(|input| !known_values.insert(input.clone()))
        {
            return Err(VNextError::InvalidModelConfig {
                family_id: family_id.to_string(),
                field: "program.inputs".to_owned(),
                reason: "input identities must be non-empty and unique".to_owned(),
            });
        }
        let mut block_ids = BTreeSet::new();
        let mut node_ids = BTreeSet::new();
        for state in &states {
            let tensor_valid = state
                .tensor
                .validate(&format!("program.states.{}.tensor", state.id))
                .and_then(|()| state.capacity_demand.validate(state.tensor.byte_len()?));
            if tensor_valid.is_err() || !known_values.insert(state.value_id.clone()) {
                return Err(VNextError::InvalidModelConfig {
                    family_id: family_id.to_string(),
                    field: "program.states.value_id".to_owned(),
                    reason: format!("duplicate value `{}`", state.value_id),
                });
            }
        }
        let mut weight_ids = BTreeSet::new();
        for weight in &weights {
            if !weight_ids.insert(weight.weight_id.clone())
                || !known_values.insert(weight.value_id.clone())
            {
                return Err(VNextError::InvalidModelConfig {
                    family_id: family_id.to_string(),
                    field: "program.weights".to_owned(),
                    reason: format!(
                        "duplicate weight `{}` or value `{}`",
                        weight.weight_id, weight.value_id
                    ),
                });
            }
            weight
                .tensor
                .validate(&format!("program.weights.{}.tensor", weight.weight_id))?;
        }
        for block in &blocks {
            if block.id.is_empty() || block.nodes.is_empty() || !block_ids.insert(block.id.clone())
            {
                return Err(VNextError::InvalidModelConfig {
                    family_id: family_id.to_string(),
                    field: "program.blocks.id".to_owned(),
                    reason: "block identities must be non-empty and unique".to_owned(),
                });
            }
            for node in &block.nodes {
                if node.required_version.major == 0 || node.outputs.is_empty() {
                    return Err(VNextError::InvalidModelConfig {
                        family_id: family_id.to_string(),
                        field: "program.nodes.contract".to_owned(),
                        reason: format!("node `{}` has an invalid version or no outputs", node.id),
                    });
                }
                for value in node.attributes.values() {
                    value.validate(&format!("program node `{}` attributes", node.id))?;
                }
                if !node_ids.insert(node.id.clone()) {
                    return Err(VNextError::InvalidModelConfig {
                        family_id: family_id.to_string(),
                        field: "program.nodes.id".to_owned(),
                        reason: format!("duplicate node `{}`", node.id),
                    });
                }
                if let ProgramNodeWorkSpec::Tokens { value_id, .. } = &node.work {
                    let source_count = node
                        .inputs
                        .iter()
                        .chain(&node.outputs)
                        .filter(|candidate| *candidate == value_id)
                        .count();
                    let is_state_or_weight = states.iter().any(|state| state.value_id == *value_id)
                        || weights.iter().any(|weight| weight.value_id == *value_id);
                    if source_count != 1 || is_state_or_weight {
                        return Err(VNextError::InvalidModelConfig {
                            family_id: family_id.to_string(),
                            field: "program.nodes.work".to_owned(),
                            reason: format!(
                                "node `{}` token work source must identify one activation binding",
                                node.id
                            ),
                        });
                    }
                }
                if node
                    .inputs
                    .iter()
                    .any(|input| !known_values.contains(input))
                {
                    return Err(VNextError::InvalidModelConfig {
                        family_id: family_id.to_string(),
                        field: "program.nodes.inputs".to_owned(),
                        reason: format!("node `{}` references an unknown input", node.id),
                    });
                }
                for output in &node.outputs {
                    if !known_values.insert(output.clone()) {
                        return Err(VNextError::InvalidModelConfig {
                            family_id: family_id.to_string(),
                            field: "program.nodes.outputs".to_owned(),
                            reason: format!("value `{output}` has multiple producers"),
                        });
                    }
                }
            }
        }
        let mut state_ids = BTreeSet::new();
        if states
            .iter()
            .any(|state| !state_ids.insert(state.id.clone()))
        {
            return Err(VNextError::InvalidModelConfig {
                family_id: family_id.to_string(),
                field: "program.states.id".to_owned(),
                reason: "state identities must be unique".to_owned(),
            });
        }
        let mut output_ids = BTreeSet::new();
        if outputs.is_empty()
            || outputs
                .iter()
                .any(|output| !known_values.contains(output) || !output_ids.insert(output.clone()))
        {
            return Err(VNextError::InvalidModelConfig {
                family_id: family_id.to_string(),
                field: "program.outputs".to_owned(),
                reason: "program outputs must be non-empty, known, and unique".to_owned(),
            });
        }
        states.sort_by(|left, right| left.id.cmp(&right.id));
        weights.sort_by(|left, right| left.weight_id.cmp(&right.weight_id));
        Ok(Self {
            family_id,
            inputs,
            blocks,
            states,
            weights,
            outputs,
        })
    }

    pub fn family_id(&self) -> &ModelFamilyId {
        &self.family_id
    }

    pub fn inputs(&self) -> &[ProgramValueId] {
        &self.inputs
    }

    pub fn blocks(&self) -> &[ProgramBlock] {
        &self.blocks
    }

    pub fn states(&self) -> &[StateSpec] {
        &self.states
    }

    pub fn weights(&self) -> &[WeightReference] {
        &self.weights
    }

    pub fn outputs(&self) -> &[ProgramValueId] {
        &self.outputs
    }

    pub fn fingerprint(&self) -> Result<String, VNextError> {
        let bytes = serde_json::to_vec(self).map_err(|error| VNextError::Serialization {
            context: "serialize model program",
            message: error.to_string(),
        })?;
        Ok(format!("{:x}", Sha256::digest(bytes)))
    }
}

impl WeightSchema {
    pub fn validate_program_references(
        &self,
        family_id: &ModelFamilyId,
        program: &ModelProgram,
    ) -> Result<(), VNextError> {
        if program.family_id() != family_id {
            return Err(VNextError::InvalidModelConfig {
                family_id: family_id.to_string(),
                field: "program.family_id".to_owned(),
                reason: "program family does not match the weight schema owner".to_owned(),
            });
        }
        let schema_weights = self
            .tensors
            .iter()
            .map(|tensor| (&tensor.id, tensor.required))
            .collect::<BTreeMap<_, _>>();
        let referenced_weights = program
            .weights()
            .iter()
            .map(|reference| &reference.weight_id)
            .collect::<BTreeSet<_>>();
        if let Some(weight_id) = referenced_weights
            .iter()
            .find(|weight_id| !schema_weights.contains_key(**weight_id))
        {
            return Err(VNextError::InvalidModelConfig {
                family_id: family_id.to_string(),
                field: "program.weights".to_owned(),
                reason: format!("program references unknown weight `{weight_id}`"),
            });
        }
        if let Some(weight_id) = schema_weights.iter().find_map(|(weight_id, required)| {
            (*required && !referenced_weights.contains(weight_id)).then_some(*weight_id)
        }) {
            return Err(VNextError::InvalidModelConfig {
                family_id: family_id.to_string(),
                field: "program.weights".to_owned(),
                reason: format!("program does not reference required weight `{weight_id}`"),
            });
        }
        for reference in program.weights() {
            let tensor = self.tensor(&reference.weight_id).ok_or_else(|| {
                VNextError::InvalidModelConfig {
                    family_id: family_id.to_string(),
                    field: "program.weights".to_owned(),
                    reason: format!(
                        "program references unknown weight `{}`",
                        reference.weight_id
                    ),
                }
            })?;
            if reference.tensor.dimensions != tensor.dimensions
                || reference.tensor.element_type != tensor.logical_element_type
            {
                return Err(VNextError::InvalidModelConfig {
                    family_id: family_id.to_string(),
                    field: format!("program.weights.{}.tensor", reference.weight_id),
                    reason: "program value shape or dtype differs from the logical weight schema"
                        .to_owned(),
                });
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TemplateMetadata {
    pub template: String,
    pub source_file: String,
    pub sha256: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SpecialTokenRole {
    Bos,
    Eos,
    Pad,
    Stop,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize)]
pub struct SpecialTokenCollision {
    first: SpecialTokenRole,
    second: SpecialTokenRole,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SpecialTokenCollisionWire {
    first: SpecialTokenRole,
    second: SpecialTokenRole,
}

impl<'de> Deserialize<'de> for SpecialTokenCollision {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = SpecialTokenCollisionWire::deserialize(deserializer)?;
        Self::new(wire.first, wire.second).map_err(serde::de::Error::custom)
    }
}

impl SpecialTokenCollision {
    pub fn new(first: SpecialTokenRole, second: SpecialTokenRole) -> Result<Self, VNextError> {
        if first == second {
            return Err(VNextError::InvalidExecutionPlan {
                reason: "a special-token collision must name two different roles".to_owned(),
            });
        }
        let (first, second) = if first < second {
            (first, second)
        } else {
            (second, first)
        };
        Ok(Self { first, second })
    }

    pub const fn first(&self) -> SpecialTokenRole {
        self.first
    }

    pub const fn second(&self) -> SpecialTokenRole {
        self.second
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct SpecialTokenCollisionPolicy {
    allowed: BTreeSet<SpecialTokenCollision>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SpecialTokenCollisionPolicyWire {
    allowed: BTreeSet<SpecialTokenCollision>,
}

impl<'de> Deserialize<'de> for SpecialTokenCollisionPolicy {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = SpecialTokenCollisionPolicyWire::deserialize(deserializer)?;
        Ok(Self::new(wire.allowed))
    }
}

impl SpecialTokenCollisionPolicy {
    pub fn new(allowed: BTreeSet<SpecialTokenCollision>) -> Self {
        Self { allowed }
    }

    pub fn require_distinct() -> Self {
        Self {
            allowed: BTreeSet::new(),
        }
    }

    pub fn allows(&self, left: SpecialTokenRole, right: SpecialTokenRole) -> bool {
        SpecialTokenCollision::new(left, right)
            .is_ok_and(|collision| self.allowed.contains(&collision))
    }

    pub fn allowed(&self) -> &BTreeSet<SpecialTokenCollision> {
        &self.allowed
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SpecialTokenMetadata {
    pub bos_token_id: Option<u32>,
    pub eos_token_ids: BTreeSet<u32>,
    pub pad_token_id: Option<u32>,
    pub collision_policy: SpecialTokenCollisionPolicy,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelSemanticMetadata {
    pub template: TemplateMetadata,
    pub special_tokens: SpecialTokenMetadata,
}

/// Compile-time model family provider with a typed, validated configuration.
pub trait ModelFamilyProvider: Send + Sync {
    type Config: Clone + Send + Sync + Serialize + DeserializeOwned + 'static;

    fn family_id(&self) -> &ModelFamilyId;

    fn external_metadata_ids(&self) -> BTreeSet<ExternalModelMetadataId>;

    fn validate_config_identity(
        &self,
        raw: &serde_json::Value,
        config: &Self::Config,
    ) -> Result<(), VNextError>;

    /// Returns the exact external metadata identity represented by `config`.
    /// Every provider must make this selection explicit; core never assumes a
    /// singleton catalog row is the intended typed identity.
    fn validated_external_metadata_id(
        &self,
        raw: &serde_json::Value,
        config: &Self::Config,
    ) -> Result<ExternalModelMetadataId, VNextError>;

    fn parse_config(&self, raw: &serde_json::Value) -> Result<Self::Config, VNextError>;

    fn weight_schema(&self, config: &Self::Config) -> Result<WeightSchema, VNextError>;

    fn semantic_program(&self, config: &Self::Config) -> Result<ModelProgram, VNextError>;

    fn semantic_metadata(&self, config: &Self::Config)
        -> Result<ModelSemanticMetadata, VNextError>;
}

/// Maximum raw JSON bytes accepted before decoding a prepared family package.
pub const MAX_PREPARED_MODEL_FAMILY_WIRE_BYTES: usize = 16 * 1024 * 1024;

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct PreparedModelFamily {
    family_id: ModelFamilyId,
    external_metadata_id: ExternalModelMetadataId,
    canonical_config: serde_json::Value,
    config_fingerprint: String,
    weight_schema: WeightSchema,
    program: ModelProgram,
    metadata: ModelSemanticMetadata,
}

/// Serialized prepared packages are evidence, not trusted runtime objects.
/// Rehydration must resolve the typed provider again and reproduce every field.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct UnvalidatedPreparedModelFamily {
    family_id: ModelFamilyId,
    external_metadata_id: ExternalModelMetadataId,
    canonical_config: serde_json::Value,
    config_fingerprint: String,
    weight_schema: WeightSchema,
    program: ModelProgram,
    metadata: ModelSemanticMetadata,
}

/// Crate-private serde shape used by both the top-level decoder and nested
/// resolved-plan wire. Public code can only obtain the explicit unvalidated
/// package through a byte-bounded decoder.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub(crate) struct PreparedModelFamilyWire {
    family_id: ModelFamilyId,
    external_metadata_id: ExternalModelMetadataId,
    canonical_config: serde_json::Value,
    config_fingerprint: String,
    weight_schema: WeightSchema,
    program: ModelProgram,
    metadata: ModelSemanticMetadata,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct PreparedModelFamilyWireFields {
    family_id: ModelFamilyId,
    external_metadata_id: ExternalModelMetadataId,
    canonical_config: serde_json::Value,
    config_fingerprint: String,
    weight_schema: WeightSchema,
    program: ModelProgram,
    metadata: ModelSemanticMetadata,
}

impl<'de> Deserialize<'de> for PreparedModelFamilyWire {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let raw = serde_json::Value::deserialize(deserializer)?;
        let fields =
            PreparedModelFamilyWireFields::deserialize(&raw).map_err(serde::de::Error::custom)?;
        let canonical = serde_json::to_value(&fields).map_err(serde::de::Error::custom)?;
        if canonical != raw {
            return Err(serde::de::Error::custom(
                "prepared model family wire contains unknown or non-canonical nested fields",
            ));
        }
        Ok(Self {
            family_id: fields.family_id,
            external_metadata_id: fields.external_metadata_id,
            canonical_config: fields.canonical_config,
            config_fingerprint: fields.config_fingerprint,
            weight_schema: fields.weight_schema,
            program: fields.program,
            metadata: fields.metadata,
        })
    }
}

impl From<PreparedModelFamilyWire> for UnvalidatedPreparedModelFamily {
    fn from(wire: PreparedModelFamilyWire) -> Self {
        Self {
            family_id: wire.family_id,
            external_metadata_id: wire.external_metadata_id,
            canonical_config: wire.canonical_config,
            config_fingerprint: wire.config_fingerprint,
            weight_schema: wire.weight_schema,
            program: wire.program,
            metadata: wire.metadata,
        }
    }
}

impl UnvalidatedPreparedModelFamily {
    pub fn revalidate(
        self,
        registry: &dyn ModelFamilyRegistry,
    ) -> Result<PreparedModelFamily, VNextError> {
        let registration = registry.resolve(&self.family_id)?;
        if registration.family_id() != &self.family_id {
            return Err(VNextError::InvalidModelConfig {
                family_id: self.family_id.to_string(),
                field: "registration.family_id".to_owned(),
                reason: "registry returned a registration for a different family".to_owned(),
            });
        }
        let metadata_registration = registry.resolve_external(&self.external_metadata_id)?;
        if !std::ptr::eq(registration, metadata_registration) {
            return Err(VNextError::InvalidModelConfig {
                family_id: self.family_id.to_string(),
                field: "external_metadata_id".to_owned(),
                reason: "external metadata identity resolves to a different family registration"
                    .to_owned(),
            });
        }
        let rebuilt = registration.prepare(&self.canonical_config)?;
        let exact_match = rebuilt.family_id == self.family_id
            && rebuilt.external_metadata_id == self.external_metadata_id
            && rebuilt.canonical_config == self.canonical_config
            && rebuilt.config_fingerprint == self.config_fingerprint
            && rebuilt.weight_schema == self.weight_schema
            && rebuilt.program == self.program
            && rebuilt.metadata == self.metadata;
        if !exact_match {
            return Err(VNextError::InvalidModelConfig {
                family_id: self.family_id.to_string(),
                field: "prepared_package".to_owned(),
                reason: "serialized package differs from the typed provider reconstruction"
                    .to_owned(),
            });
        }
        Ok(rebuilt)
    }
}

impl PreparedModelFamily {
    fn from_canonical_config(
        family_id: ModelFamilyId,
        external_metadata_id: ExternalModelMetadataId,
        canonical_config: serde_json::Value,
        mut weight_schema: WeightSchema,
        program: ModelProgram,
        metadata: ModelSemanticMetadata,
    ) -> Result<Self, VNextError> {
        if !canonical_config.is_object()
            || canonicalize_json(canonical_config.clone()) != canonical_config
        {
            return Err(VNextError::InvalidModelConfig {
                family_id: family_id.to_string(),
                field: "config".to_owned(),
                reason: "prepared config must be a canonical JSON object".to_owned(),
            });
        }
        let config_bytes =
            serde_json::to_vec(&canonical_config).map_err(|error| VNextError::Serialization {
                context: "serialize canonical model family config",
                message: error.to_string(),
            })?;
        let config_fingerprint = format!("{:x}", Sha256::digest(config_bytes));
        weight_schema.validate(&family_id)?;
        weight_schema.normalize();
        weight_schema.validate(&family_id)?;
        if program.family_id() != &family_id {
            return Err(VNextError::InvalidModelConfig {
                family_id: family_id.to_string(),
                field: "program.family_id".to_owned(),
                reason: "program family does not match prepared family".to_owned(),
            });
        }
        weight_schema.validate_program_references(&family_id, &program)?;
        Self::validate_metadata(&family_id, &metadata)?;
        Ok(Self {
            family_id,
            external_metadata_id,
            canonical_config,
            config_fingerprint,
            weight_schema,
            program,
            metadata,
        })
    }

    fn validate_metadata(
        family_id: &ModelFamilyId,
        metadata: &ModelSemanticMetadata,
    ) -> Result<(), VNextError> {
        let source = metadata.template.source_file.as_str();
        let valid_source = !source.is_empty()
            && !source.starts_with('/')
            && !source.contains('\\')
            && source
                .split('/')
                .all(|component| !matches!(component, "" | "." | ".."));
        if metadata.template.template.is_empty()
            || !valid_source
            || !is_canonical_sha256(&metadata.template.sha256)
            || metadata.special_tokens.eos_token_ids.is_empty()
        {
            return Err(VNextError::InvalidModelConfig {
                family_id: family_id.to_string(),
                field: "semantic_metadata".to_owned(),
                reason: "template, source, checksum, and end tokens must be explicit and valid"
                    .to_owned(),
            });
        }
        Ok(())
    }

    pub fn family_id(&self) -> &ModelFamilyId {
        &self.family_id
    }

    pub fn external_metadata_id(&self) -> &ExternalModelMetadataId {
        &self.external_metadata_id
    }

    pub fn canonical_config(&self) -> &serde_json::Value {
        &self.canonical_config
    }

    pub fn config_fingerprint(&self) -> &str {
        &self.config_fingerprint
    }

    pub fn weight_schema(&self) -> &WeightSchema {
        &self.weight_schema
    }

    pub fn program(&self) -> &ModelProgram {
        &self.program
    }

    pub fn metadata(&self) -> &ModelSemanticMetadata {
        &self.metadata
    }

    pub fn fingerprint(&self) -> Result<String, VNextError> {
        let bytes = serde_json::to_vec(self).map_err(|error| VNextError::Serialization {
            context: "serialize prepared model family",
            message: error.to_string(),
        })?;
        Ok(format!("{:x}", Sha256::digest(bytes)))
    }

    pub fn decode_untrusted(bytes: &[u8]) -> Result<UnvalidatedPreparedModelFamily, VNextError> {
        if bytes.len() > MAX_PREPARED_MODEL_FAMILY_WIRE_BYTES {
            return Err(VNextError::Serialization {
                context: "decode untrusted prepared model family",
                message: format!(
                    "payload has {} bytes; maximum is {MAX_PREPARED_MODEL_FAMILY_WIRE_BYTES}",
                    bytes.len()
                ),
            });
        }
        serde_json::from_slice::<PreparedModelFamilyWire>(bytes)
            .map(Into::into)
            .map_err(|error| VNextError::Serialization {
                context: "decode untrusted prepared model family",
                message: error.to_string(),
            })
    }

    pub fn from_json_validated(
        bytes: &[u8],
        registry: &dyn ModelFamilyRegistry,
    ) -> Result<Self, VNextError> {
        Self::decode_untrusted(bytes)?.revalidate(registry)
    }
}

fn canonicalize_json(value: serde_json::Value) -> serde_json::Value {
    match value {
        serde_json::Value::Array(values) => {
            serde_json::Value::Array(values.into_iter().map(canonicalize_json).collect())
        }
        serde_json::Value::Object(values) => {
            let sorted = values
                .into_iter()
                .map(|(key, value)| (key, canonicalize_json(value)))
                .collect::<BTreeMap<_, _>>();
            serde_json::Value::Object(sorted.into_iter().collect())
        }
        other => other,
    }
}

fn validate_raw_config_consumed(
    family_id: &ModelFamilyId,
    raw: &serde_json::Value,
    typed: &serde_json::Value,
) -> Result<(), VNextError> {
    // Typed serialization may add explicit provider defaults. Every caller-
    // supplied value must still survive at the same path with the same JSON
    // value; serde's default unknown-field behavior cannot erase input here.
    fn walk(raw: &serde_json::Value, typed: &serde_json::Value, path: &str) -> Option<String> {
        match (raw, typed) {
            (serde_json::Value::Object(raw), serde_json::Value::Object(typed)) => {
                for (key, raw_value) in raw {
                    let next = if path.is_empty() {
                        format!("/{key}")
                    } else {
                        format!("{path}/{key}")
                    };
                    let Some(typed_value) = typed.get(key) else {
                        return Some(next);
                    };
                    if let Some(rejected) = walk(raw_value, typed_value, &next) {
                        return Some(rejected);
                    }
                }
                None
            }
            (serde_json::Value::Array(raw), serde_json::Value::Array(typed))
                if raw.len() == typed.len() =>
            {
                raw.iter()
                    .zip(typed)
                    .enumerate()
                    .find_map(|(index, (raw, typed))| walk(raw, typed, &format!("{path}/{index}")))
            }
            _ if raw == typed => None,
            _ => Some(path.to_owned()),
        }
    }

    if !raw.is_object() || !typed.is_object() {
        return Err(VNextError::InvalidModelConfig {
            family_id: family_id.to_string(),
            field: "config".to_owned(),
            reason: "raw and typed model configurations must be JSON objects".to_owned(),
        });
    }
    if let Some(path) = walk(raw, typed, "") {
        return Err(VNextError::InvalidModelConfig {
            family_id: family_id.to_string(),
            field: if path.is_empty() {
                "config".to_owned()
            } else {
                path
            },
            reason: "raw configuration field was ignored or changed by typed parsing".to_owned(),
        });
    }
    Ok(())
}

fn is_canonical_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

/// Object-safe loading-time trampoline for a heterogeneous family catalog.
/// The raw JSON exists only at the configuration boundary; a typed provider
/// validates it before producing the backend-free package.
pub trait ModelFamilyRegistration: Send + Sync {
    fn family_id(&self) -> &ModelFamilyId;

    fn external_metadata_ids(&self) -> BTreeSet<ExternalModelMetadataId>;

    fn prepare(&self, raw_config: &serde_json::Value) -> Result<PreparedModelFamily, VNextError>;
}

pub struct TypedFamilyRegistration<P> {
    provider: P,
}

impl<P> TypedFamilyRegistration<P> {
    pub fn new(provider: P) -> Self {
        Self { provider }
    }
}

impl<P: ModelFamilyProvider> ModelFamilyRegistration for TypedFamilyRegistration<P> {
    fn family_id(&self) -> &ModelFamilyId {
        self.provider.family_id()
    }

    fn external_metadata_ids(&self) -> BTreeSet<ExternalModelMetadataId> {
        self.provider.external_metadata_ids()
    }

    fn prepare(&self, raw_config: &serde_json::Value) -> Result<PreparedModelFamily, VNextError> {
        let external_metadata_ids = self.provider.external_metadata_ids();
        if external_metadata_ids.is_empty() {
            return Err(VNextError::InvalidModelConfig {
                family_id: self.provider.family_id().to_string(),
                field: "external_metadata_ids".to_owned(),
                reason: "model family must declare at least one external metadata identity"
                    .to_owned(),
            });
        }
        let config = self.provider.parse_config(raw_config)?;
        let external_metadata_id = self
            .provider
            .validated_external_metadata_id(raw_config, &config)?;
        if !external_metadata_ids.contains(&external_metadata_id) {
            return Err(VNextError::InvalidModelConfig {
                family_id: self.provider.family_id().to_string(),
                field: "external_metadata_id".to_owned(),
                reason: format!(
                    "provider selected undeclared external metadata identity `{external_metadata_id}`"
                ),
            });
        }
        let typed_config = canonicalize_json(serde_json::to_value(&config).map_err(|error| {
            VNextError::Serialization {
                context: "serialize typed model configuration",
                message: error.to_string(),
            }
        })?);
        validate_raw_config_consumed(self.provider.family_id(), raw_config, &typed_config)?;
        let weight_schema = self.provider.weight_schema(&config)?;
        let program = self.provider.semantic_program(&config)?;
        let metadata = self.provider.semantic_metadata(&config)?;
        PreparedModelFamily::from_canonical_config(
            self.provider.family_id().clone(),
            external_metadata_id,
            typed_config,
            weight_schema,
            program,
            metadata,
        )
    }
}

pub trait ModelFamilyRegistry: Send + Sync {
    /// Returns the complete trusted catalog. Core owns all identity lookup so a
    /// registry cannot silently fall back or hide ambiguous registrations.
    fn registrations(&self) -> Vec<&dyn ModelFamilyRegistration>;
}

impl dyn ModelFamilyRegistry + '_ {
    pub fn resolve(
        &self,
        family_id: &ModelFamilyId,
    ) -> Result<&dyn ModelFamilyRegistration, VNextError> {
        let matches = self
            .registrations()
            .into_iter()
            .filter(|registration| registration.family_id() == family_id)
            .collect::<Vec<_>>();
        match matches.as_slice() {
            [] => Err(VNextError::UnknownModelFamily {
                family_id: family_id.to_string(),
            }),
            [registration] => Ok(*registration),
            _ => Err(VNextError::AmbiguousModelFamilyRegistration {
                identity_kind: "internal family",
                identity: family_id.to_string(),
                matches: matches.len(),
            }),
        }
    }

    pub fn resolve_external(
        &self,
        metadata_id: &ExternalModelMetadataId,
    ) -> Result<&dyn ModelFamilyRegistration, VNextError> {
        let matches = self
            .registrations()
            .into_iter()
            .filter(|registration| registration.external_metadata_ids().contains(metadata_id))
            .collect::<Vec<_>>();
        match matches.as_slice() {
            [] => Err(VNextError::UnknownExternalModelMetadata {
                metadata_id: metadata_id.to_string(),
            }),
            [registration] => Ok(*registration),
            _ => Err(VNextError::AmbiguousModelFamilyRegistration {
                identity_kind: "external metadata",
                identity: metadata_id.to_string(),
                matches: matches.len(),
            }),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TokenizerDescriptor {
    pub tokenizer_id: TokenizerId,
    pub source_file: String,
    pub sha256: String,
    pub vocabulary_size: u64,
}
