use std::collections::BTreeSet;
use std::num::NonZeroU32;

use serde::{Deserialize, Deserializer, Serialize};

use super::super::{
    CanonicalRational, ContractVersion, QuantizationFormatId, VNextError, WeightFormatId, WeightId,
    WeightLayoutId,
};
use super::ElementType;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QuantizationPacking {
    Linear,
    Interleaved,
    Tiled,
}

/// How values are partitioned by a quantized physical layout.
///
/// `WholeAxis` is shape-relative by design: all values on the group axis
/// share one scale. This represents channelwise quantization without making a
/// matrix dimension part of the otherwise stable quantization-format ABI.
/// `Block2d` carries the exact rectangular source block shape; the physical
/// layout separately binds its two dimensions to ordered logical axes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum QuantizationGrouping {
    Fixed { size: u32 },
    WholeAxis,
    Block2d { block_shape: [NonZeroU32; 2] },
}

impl QuantizationGrouping {
    pub const fn fixed(size: u32) -> Self {
        Self::Fixed { size }
    }

    pub const fn fixed_size(self) -> Option<u32> {
        match self {
            Self::Fixed { size } => Some(size),
            Self::WholeAxis | Self::Block2d { .. } => None,
        }
    }

    pub const fn block_2d(block_shape: [NonZeroU32; 2]) -> Self {
        Self::Block2d { block_shape }
    }

    pub const fn block_shape_2d(self) -> Option<[NonZeroU32; 2]> {
        match self {
            Self::Block2d { block_shape } => Some(block_shape),
            Self::Fixed { .. } | Self::WholeAxis => None,
        }
    }

    pub const fn resolved_size(self, axis_extent: u64) -> u64 {
        match self {
            Self::Fixed { size } => size as u64,
            Self::WholeAxis => axis_extent,
            // Multidimensional grouping has no meaningful single-axis size.
            // Zero makes accidental use by a legacy single-axis validator
            // fail closed instead of silently choosing one block dimension.
            Self::Block2d { .. } => 0,
        }
    }

    const fn is_valid(self) -> bool {
        match self {
            Self::Fixed { size } => size != 0 && size.is_power_of_two(),
            Self::WholeAxis => true,
            Self::Block2d { block_shape } => {
                block_shape[0].get().is_power_of_two() && block_shape[1].get().is_power_of_two()
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct QuantizationSpec {
    pub format_id: QuantizationFormatId,
    pub bits_per_weight: u8,
    pub grouping: QuantizationGrouping,
    pub packing: QuantizationPacking,
    pub scale_type: ElementType,
    pub zero_point_type: Option<ElementType>,
}

impl QuantizationSpec {
    pub fn validate(&self) -> Result<(), VNextError> {
        if !(1..=8).contains(&self.bits_per_weight)
            || !self.grouping.is_valid()
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
    pub format_id: QuantizationFormatId,
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

    pub(crate) fn physical_bytes(
        &self,
        dimensions: &[u64],
        component_id: &WeightId,
    ) -> Result<u64, VNextError> {
        let elements =
            checked_elements(dimensions).ok_or_else(|| VNextError::InvalidExecutionPlan {
                reason: format!("physical component `{component_id}` size overflows u64"),
            })?;
        match self {
            Self::Dense { element_type } | Self::DenseAffine { element_type, .. } => elements
                .checked_mul(element_type.size_bytes())
                .ok_or_else(|| VNextError::InvalidExecutionPlan {
                    reason: format!("physical component `{component_id}` byte size overflows u64"),
                }),
            Self::Quantized(_) => Ok(elements),
            Self::BlockQuantized(spec) => {
                spec.validate()?;
                elements
                    .checked_mul(u64::from(spec.bytes_per_block))
                    .ok_or_else(|| VNextError::InvalidExecutionPlan {
                        reason: format!(
                            "physical block component `{component_id}` byte size overflows u64"
                        ),
                    })
            }
        }
    }
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
        /// Optional packed storage shape for asymmetric zero points. When
        /// absent, one dense zero-point scalar is stored per quantization
        /// group. When present, it must contain exactly `bits_per_weight`
        /// bits per group and the bound component dtype describes the packed
        /// storage word (for example I32 words containing eight INT4 values).
        zero_point_packed_dimensions: Option<Vec<u64>>,
        /// Per-coordinate group assignment. Its semantic shape is the one
        /// dimensional logical axis extent, not the multi-dimensional group
        /// tensor shape.
        axis_indices: Option<AxisWeightComponent>,
        permutation: Option<AxisWeightComponent>,
        codebook: Option<PhysicalWeightComponentBinding>,
        group_axis: u32,
        group_padding: PhysicalWeightPadding,
    },
    /// Separate-scale quantization over a rectangular two-dimensional block
    /// grid. `block_axes` are ordered and canonical; the corresponding block
    /// sizes live losslessly in [`QuantizationGrouping::Block2d`]. Boundary
    /// blocks may be partial, so scale extents use ceiling division without
    /// implying padded packed-value storage.
    QuantizedBlockGrid {
        packed_values: PhysicalWeightComponentBinding,
        /// Semantic packed-storage shape. Its element product must equal the
        /// exact byte count implied by the unpadded logical tensor.
        packed_dimensions: Vec<u64>,
        scales: PhysicalWeightComponentBinding,
        block_axes: [u32; 2],
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
    pub(crate) fn normalize(&mut self) {
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
            | Self::QuantizedBlockGrid { .. }
            | Self::BlockQuantized { .. } => {}
        }
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
    pub(crate) fn from_parts(
        component_id: WeightId,
        role: WeightComponentRole,
        physical_dimensions: Vec<u64>,
        encoding: WeightEncoding,
    ) -> Self {
        Self {
            component_id,
            role,
            physical_dimensions,
            encoding,
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
        self.encoding
            .physical_bytes(&self.physical_dimensions, &self.component_id)
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
///
/// `schema_format_id` identifies the enclosing source or materialized schema.
/// It is a planning compatibility key, not the physical ABI of every component
/// referenced by this binding. Providers must decode components from
/// `physical_layout` and `components`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ResolvedWeightBinding {
    weight_id: WeightId,
    #[serde(rename = "format_id")]
    schema_format_id: WeightFormatId,
    layout_id: WeightLayoutId,
    schema_version: ContractVersion,
    physical_layout: PhysicalWeightLayout,
    components: Vec<ResolvedWeightComponentLayout>,
}

/// Operation-owned capability for validating a resolved physical binding
/// against the logical tensor contract supplied by the model layer.
///
/// The physical ABI stays independent of model schemas; the model owner
/// provides the schema-aware implementation without introducing an
/// operation-to-model dependency.
pub(crate) trait ResolvedWeightLogicalValidation {
    fn validate_logical_contract(
        &self,
        logical_dimensions: &[u64],
        logical_element_type: ElementType,
    ) -> Result<(), VNextError>;
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
        Self::from_parts(
            wire.weight_id,
            wire.format_id,
            wire.layout_id,
            wire.schema_version,
            wire.physical_layout,
            wire.components,
        )
        .map_err(serde::de::Error::custom)
    }
}

impl ResolvedWeightBinding {
    pub(crate) fn from_parts(
        weight_id: WeightId,
        schema_format_id: WeightFormatId,
        layout_id: WeightLayoutId,
        schema_version: ContractVersion,
        physical_layout: PhysicalWeightLayout,
        components: Vec<ResolvedWeightComponentLayout>,
    ) -> Result<Self, VNextError> {
        let binding = Self {
            weight_id,
            schema_format_id,
            layout_id,
            schema_version,
            physical_layout,
            components,
        };
        binding.validate_structure()?;
        Ok(binding)
    }

    pub(crate) fn validate_structure(&self) -> Result<(), VNextError> {
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

    pub fn weight_id(&self) -> &WeightId {
        &self.weight_id
    }

    /// Enclosing schema/container identity used by provider selection.
    ///
    /// This must not be used to infer a bound component's physical encoding.
    pub(crate) fn schema_format_id(&self) -> &WeightFormatId {
        &self.schema_format_id
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

    pub fn quantization_formats(&self) -> BTreeSet<QuantizationFormatId> {
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

pub(crate) fn validate_physical_layout_budget(layout: &PhysicalWeightLayout) -> Result<(), String> {
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
            PhysicalWeightLayout::QuantizedBlockGrid { .. } => 2,
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
            | PhysicalWeightLayout::QuantizedBlockGrid { .. }
            | PhysicalWeightLayout::BlockQuantized { .. } => {}
        }
    }
    Ok(())
}

pub(crate) fn physical_component_ids(
    layout: &PhysicalWeightLayout,
) -> Result<BTreeSet<WeightId>, String> {
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
            PhysicalWeightLayout::QuantizedBlockGrid {
                packed_values,
                scales,
                ..
            } => {
                insert_binding(packed_values);
                insert_binding(scales);
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

pub(crate) fn checked_elements(dimensions: &[u64]) -> Option<u64> {
    dimensions
        .iter()
        .try_fold(1_u64, |elements, extent| elements.checked_mul(*extent))
}
