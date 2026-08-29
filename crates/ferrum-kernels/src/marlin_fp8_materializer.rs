//! Cold-path execution-weight preparation for CUDA Marlin FP8 W8A16.
//!
//! Eligibility comes from stable operation contracts and physical weight
//! shapes. Model names, device names, memory tiers, and request state are
//! deliberately absent from this boundary.

use std::borrow::Cow;
use std::collections::{BTreeMap, BTreeSet};

use ferrum_interfaces::vnext::{
    ApproximateWeightQualityContract, CanonicalRational, CapabilityId, ContractVersion,
    DeviceDescriptor, ElementType, PhysicalStorageLayout, PhysicalWeightComponentBinding,
    PhysicalWeightLayout, PhysicalWeightPadding, PreparedModelFamily, QuantizationFormatId,
    QuantizationGrouping, QuantizationPacking, QuantizationSpec, VNextError,
    WeightComponentPayload, WeightComponentRole, WeightComponentSource, WeightComponentSpec,
    WeightEncoding, WeightFormatId, WeightId, WeightLayoutId, WeightMaterializationFidelity,
    WeightMaterializer, WeightMaterializerDescriptor, WeightMaterializerId, WeightSchema,
    CAUSAL_PAGED_ATTENTION_OPERATION_ID, DENSE_LINEAR_OPERATION_ID, DENSE_SWIGLU_OPERATION_ID,
    GATED_DELTA_RECURRENT_ATTENTION_OPERATION_ID,
};
use sha2::{Digest, Sha256};

use crate::marlin_repack::{
    fp8_marlin_shape_supported, prepare_block_fp8_weight_for_fp8_marlin,
    prepare_f16_weight_for_fp8_marlin, Fp8MarlinWeight,
};

pub const MARLIN_FP8_WEIGHT_MATERIALIZER_ID: &str = "weight-materializer.cuda.marlin-fp8-w8a16";
pub const BLOCK_FP8_TO_MARLIN_FP8_WEIGHT_MATERIALIZER_ID: &str =
    "weight-materializer.cuda.block-fp8-to-marlin-fp8-w8a16";
pub const MARLIN_FP8_CAPABILITY_ID: &str = "capability.kernel.cuda.marlin.fp8-w8a16";
pub const MARLIN_FP8_WEIGHT_FORMAT_ID: &str = "weight-format.execution.cuda.marlin-fp8-w8a16-mixed";
pub const MARLIN_FP8_WEIGHT_LAYOUT_ID: &str = "weight-layout.execution.cuda.marlin-fp8-w8a16-mixed";
pub const MARLIN_FP8_QUANTIZATION_FORMAT_ID: &str = "quantization.marlin.fp8-e4m3fn-channelwise";

const MATERIALIZER_VERSION: ContractVersion = ContractVersion::new(2, 0);
const BLOCK_FP8_MATERIALIZER_VERSION: ContractVersion = ContractVersion::new(1, 0);
const DERIVED_COMPONENT_PREFIX: &str = "component.execution.marlin-fp8";
const BLOCK_FP8_DERIVED_COMPONENT_PREFIX: &str = "component.execution.block-fp8-marlin-fp8";
const BLOCK_FP8_SOURCE_QUANTIZATION_FORMAT_ID: &str =
    "quantization.safetensors.fp8-e4m3-block-grid-inverse-scale";
const BLOCK_FP8_EXECUTION_CONTRACT_FINGERPRINT: &str =
    "882bc49ca312875a12a5290319f6c8294386a5960c2065cbda3f3dff2d55598e";
const BLOCK_FP8_QUALITY_VECTOR_DIGEST: &str =
    "4c8b44a6a6e2ca803f6a3916b033a50a8a007cb2452a0e9246ed6c7f3cacbb51";
const BLOCK_FP8_BLOCK_SHAPE: [usize; 2] = [128, 128];

pub fn marlin_fp8_weight_materializer() -> Result<Box<dyn WeightMaterializer>, VNextError> {
    Ok(Box::new(MarlinFp8WeightMaterializer::new()?))
}

pub fn block_fp8_to_marlin_fp8_weight_materializer(
) -> Result<Box<dyn WeightMaterializer>, VNextError> {
    Ok(Box::new(BlockFp8ToMarlinFp8WeightMaterializer::new()?))
}

struct MarlinFp8WeightMaterializer {
    descriptor: WeightMaterializerDescriptor,
}

impl MarlinFp8WeightMaterializer {
    fn new() -> Result<Self, VNextError> {
        let fingerprint = implementation_fingerprint(&[
            include_str!("marlin_fp8_materializer.rs").as_bytes(),
            include_str!("marlin_repack.rs").as_bytes(),
            MARLIN_FP8_WEIGHT_MATERIALIZER_ID.as_bytes(),
        ]);
        Ok(Self {
            descriptor: WeightMaterializerDescriptor::new(
                WeightMaterializerId::new(MARLIN_FP8_WEIGHT_MATERIALIZER_ID)?,
                MATERIALIZER_VERSION,
                fingerprint,
                WeightMaterializationFidelity::Approximate,
                BTreeSet::from([CapabilityId::new(MARLIN_FP8_CAPABILITY_ID)?]),
            )?,
        })
    }

    fn candidates(family: &PreparedModelFamily) -> Result<Vec<MarlinFp8Candidate>, VNextError> {
        marlin_fp8_candidates(family)
    }

    fn materialize_group<'source>(
        &self,
        source: &'source dyn WeightComponentSource,
        source_components: &[&WeightComponentSpec],
        execution_components: &[&WeightComponentSpec],
    ) -> Result<Vec<WeightComponentPayload<'source>>, VNextError> {
        let [source_component] = source_components else {
            return Err(invalid_plan(
                "Marlin FP8 materialization requires exactly one F16 source component",
            ));
        };
        if let [execution_component] = execution_components {
            if *source_component == *execution_component {
                return source
                    .component(source_component)
                    .map(|payload| vec![payload]);
            }
        }
        if execution_components.is_empty() || execution_components.len() > 2 {
            return Err(invalid_plan(
                "Marlin FP8 materialization requires one or both derived components",
            ));
        }
        let [n, k] = source_component.dimensions.as_slice() else {
            return Err(invalid_plan(
                "Marlin FP8 source component must be a two-dimensional matrix",
            ));
        };
        let n = usize::try_from(*n)
            .map_err(|_| invalid_plan("Marlin FP8 output width exceeds usize"))?;
        let k = usize::try_from(*k)
            .map_err(|_| invalid_plan("Marlin FP8 input width exceeds usize"))?;
        if source_component.role != WeightComponentRole::Values
            || source_component.encoding
                != (WeightEncoding::Dense {
                    element_type: ElementType::F16,
                })
            || !marlin_fp8_projection_shape_supported(n, k)
        {
            return Err(invalid_plan(format!(
                "source component `{}` is not an eligible Marlin FP8 F16 matrix",
                source_component.id
            )));
        }

        let packed_id = derived_component_id(&source_component.id, DerivedComponentKind::Packed)?;
        let scales_id = derived_component_id(&source_component.id, DerivedComponentKind::Scales)?;
        let mut requested_ids = BTreeSet::new();
        for component in execution_components {
            if !requested_ids.insert(component.id.clone())
                || (component.id != packed_id && component.id != scales_id)
            {
                return Err(invalid_plan(format!(
                    "execution component `{}` is not derived from source `{}`",
                    component.id, source_component.id
                )));
            }
        }

        let source_payload = source.component(source_component)?;
        let source_files = source_payload.source_files().to_vec();
        let prepared =
            prepare_f16_weight_for_fp8_marlin(source_payload.bytes(), n, k).map_err(|error| {
                invalid_plan(format!(
                    "prepare Marlin FP8 component `{}`: {error}",
                    source_component.id
                ))
            })?;
        execution_components
            .iter()
            .map(|component| {
                derived_payload(
                    component,
                    source_files.clone(),
                    &packed_id,
                    &scales_id,
                    &prepared,
                )
            })
            .collect()
    }
}

impl WeightMaterializer for MarlinFp8WeightMaterializer {
    fn descriptor(&self) -> &WeightMaterializerDescriptor {
        &self.descriptor
    }

    fn execution_schema(
        &self,
        family: &PreparedModelFamily,
        _device: &DeviceDescriptor,
    ) -> Result<WeightSchema, VNextError> {
        let candidates = Self::candidates(family)?;
        if candidates.is_empty() {
            return Ok(family.weight_schema().clone());
        }

        let mut schema = family.weight_schema().clone();
        schema.format_id = WeightFormatId::new(MARLIN_FP8_WEIGHT_FORMAT_ID)?;
        schema.layout_id = WeightLayoutId::new(MARLIN_FP8_WEIGHT_LAYOUT_ID)?;
        let candidate_by_weight = candidates
            .iter()
            .map(|candidate| (&candidate.weight_id, candidate))
            .collect::<BTreeMap<_, _>>();
        let removed_sources = candidates
            .iter()
            .map(|candidate| &candidate.source_component_id)
            .collect::<BTreeSet<_>>();
        schema
            .components
            .retain(|component| !removed_sources.contains(&component.id));
        for candidate in &candidates {
            schema.components.push(candidate.packed_component.clone());
            schema.components.push(candidate.scales_component.clone());
        }
        for tensor in &mut schema.tensors {
            let Some(candidate) = candidate_by_weight.get(&tensor.id) else {
                continue;
            };
            tensor.physical_layout = PhysicalWeightLayout::Quantized {
                packed_values: PhysicalWeightComponentBinding::exact_contiguous(
                    candidate.packed_component.id.clone(),
                ),
                packed_dimensions: candidate.logical_dimensions.clone(),
                scales: PhysicalWeightComponentBinding::exact_contiguous(
                    candidate.scales_component.id.clone(),
                ),
                zero_points: None,
                zero_point_packed_dimensions: None,
                axis_indices: None,
                permutation: None,
                codebook: None,
                group_axis: 1,
                group_padding: PhysicalWeightPadding::Exact,
            };
        }
        Ok(schema)
    }

    fn component_sources(
        &self,
        family: &PreparedModelFamily,
        execution_schema: &WeightSchema,
    ) -> Result<BTreeMap<WeightId, Vec<WeightId>>, VNextError> {
        let mut derived_sources = BTreeMap::new();
        for candidate in Self::candidates(family)? {
            derived_sources.insert(
                candidate.packed_component.id,
                candidate.source_component_id.clone(),
            );
            derived_sources.insert(candidate.scales_component.id, candidate.source_component_id);
        }
        execution_schema
            .components
            .iter()
            .map(|component| {
                let source_id = derived_sources
                    .get(&component.id)
                    .cloned()
                    .unwrap_or_else(|| component.id.clone());
                Ok((component.id.clone(), vec![source_id]))
            })
            .collect()
    }

    fn materialize_component<'source>(
        &self,
        source: &'source dyn WeightComponentSource,
        source_components: &[&WeightComponentSpec],
        execution_component: &WeightComponentSpec,
    ) -> Result<WeightComponentPayload<'source>, VNextError> {
        self.materialize_group(source, source_components, &[execution_component])?
            .pop()
            .ok_or_else(|| invalid_plan("Marlin FP8 materializer returned no component"))
    }

    fn materialize_components<'source>(
        &self,
        source: &'source dyn WeightComponentSource,
        source_components: &[&WeightComponentSpec],
        execution_components: &[&WeightComponentSpec],
    ) -> Result<Vec<WeightComponentPayload<'source>>, VNextError> {
        self.materialize_group(source, source_components, execution_components)
    }
}

struct MarlinFp8Candidate {
    weight_id: WeightId,
    source_component_id: WeightId,
    logical_dimensions: Vec<u64>,
    packed_component: WeightComponentSpec,
    scales_component: WeightComponentSpec,
}

fn marlin_fp8_candidates(
    family: &PreparedModelFamily,
) -> Result<Vec<MarlinFp8Candidate>, VNextError> {
    let schema = family.weight_schema();
    let mut component_references = BTreeMap::<WeightId, usize>::new();
    for tensor in &schema.tensors {
        for component in schema.physical_component_refs(&tensor.id)? {
            *component_references
                .entry(component.id.clone())
                .or_default() += 1;
        }
    }

    let mut candidates = Vec::new();
    let mut source_ids = BTreeSet::new();
    for reference in family.program().weights() {
        let uses = family
            .program()
            .blocks()
            .iter()
            .flat_map(|block| &block.nodes)
            .flat_map(|node| {
                node.inputs
                    .iter()
                    .enumerate()
                    .filter(move |(_, input)| **input == reference.value_id)
                    .map(move |(ordinal, _)| (node.operation_id.as_str(), ordinal))
            })
            .collect::<Vec<_>>();
        if uses.is_empty()
            || uses
                .iter()
                .any(|(operation_id, ordinal)| !eligible_projection_use(operation_id, *ordinal))
        {
            continue;
        }
        let Some(tensor) = schema.tensor(&reference.weight_id) else {
            return Err(invalid_plan(format!(
                "program weight `{}` has no source tensor",
                reference.weight_id
            )));
        };
        let [n, k] = tensor.dimensions.as_slice() else {
            continue;
        };
        if tensor.logical_element_type != ElementType::F16 {
            continue;
        }
        let (n_usize, k_usize) = match (usize::try_from(*n), usize::try_from(*k)) {
            (Ok(n), Ok(k)) => (n, k),
            _ => continue,
        };
        if !marlin_fp8_projection_shape_supported(n_usize, k_usize) {
            continue;
        }
        let PhysicalWeightLayout::Dense {
            component_id: source_component_id,
        } = &tensor.physical_layout
        else {
            continue;
        };
        if component_references.get(source_component_id) != Some(&1)
            || !source_ids.insert(source_component_id.clone())
        {
            continue;
        }
        let source_component = schema
            .components
            .iter()
            .find(|component| component.id == *source_component_id)
            .ok_or_else(|| {
                invalid_plan(format!(
                    "source tensor `{}` references absent component `{source_component_id}`",
                    tensor.id
                ))
            })?;
        if source_component.role != WeightComponentRole::Values
            || source_component.dimensions != tensor.dimensions
            || source_component.encoding
                != (WeightEncoding::Dense {
                    element_type: ElementType::F16,
                })
        {
            continue;
        }

        let packed_id = derived_component_id(source_component_id, DerivedComponentKind::Packed)?;
        let scales_id = derived_component_id(source_component_id, DerivedComponentKind::Scales)?;
        let quantization = marlin_fp8_quantization_spec()?;
        quantization.validate()?;
        candidates.push(MarlinFp8Candidate {
            weight_id: reference.weight_id.clone(),
            source_component_id: source_component_id.clone(),
            logical_dimensions: tensor.dimensions.clone(),
            packed_component: WeightComponentSpec {
                id: packed_id,
                role: WeightComponentRole::PackedValues,
                external_names: derived_external_names(
                    source_component_id,
                    source_component.external_names.len(),
                    DerivedComponentKind::Packed,
                ),
                dimensions: tensor.dimensions.clone(),
                encoding: WeightEncoding::Quantized(quantization),
                required: source_component.required,
            },
            scales_component: WeightComponentSpec {
                id: scales_id,
                role: WeightComponentRole::Scales,
                external_names: derived_external_names(
                    source_component_id,
                    source_component.external_names.len(),
                    DerivedComponentKind::Scales,
                ),
                dimensions: vec![*n, 1],
                encoding: WeightEncoding::Dense {
                    element_type: ElementType::F16,
                },
                required: source_component.required,
            },
        });
    }
    Ok(candidates)
}

fn marlin_fp8_quantization_spec() -> Result<QuantizationSpec, VNextError> {
    Ok(QuantizationSpec {
        format_id: QuantizationFormatId::new(MARLIN_FP8_QUANTIZATION_FORMAT_ID)?,
        bits_per_weight: 8,
        grouping: QuantizationGrouping::WholeAxis,
        packing: QuantizationPacking::Tiled,
        scale_type: ElementType::F16,
        zero_point_type: None,
    })
}

/// The shared execution provider must support every admitted token count.
///
/// Marlin accepts narrower output tiles for small row counts, but its automatic
/// configuration rejects them once the row count grows beyond the narrow-tile
/// path. Keep those projections in F16 until the execution plan can represent
/// an explicit row-chunking strategy.
pub(crate) const fn marlin_fp8_projection_shape_supported(n: usize, k: usize) -> bool {
    fp8_marlin_shape_supported(n, k) && n.is_multiple_of(256)
}

fn eligible_projection_use(operation_id: &str, ordinal: usize) -> bool {
    (operation_id == DENSE_LINEAR_OPERATION_ID && ordinal == 1)
        || (operation_id == GATED_DELTA_RECURRENT_ATTENTION_OPERATION_ID
            && matches!(ordinal, 2 | 7))
}

#[derive(Clone, Copy)]
enum DerivedComponentKind {
    Packed,
    Scales,
}

impl DerivedComponentKind {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Packed => "packed",
            Self::Scales => "scales",
        }
    }
}

fn derived_component_id(
    source_id: &WeightId,
    kind: DerivedComponentKind,
) -> Result<WeightId, VNextError> {
    let digest = Sha256::digest(source_id.as_str().as_bytes());
    WeightId::new(format!(
        "{DERIVED_COMPONENT_PREFIX}.{:x}.{}",
        digest,
        kind.as_str()
    ))
}

fn derived_external_names(
    source_id: &WeightId,
    count: usize,
    kind: DerivedComponentKind,
) -> Vec<String> {
    let digest = Sha256::digest(source_id.as_str().as_bytes());
    (0..count)
        .map(|index| {
            format!(
                "execution.marlin-fp8.{:x}.{}.{index}",
                digest,
                kind.as_str()
            )
        })
        .collect()
}

fn derived_payload<'source>(
    component: &WeightComponentSpec,
    source_files: Vec<String>,
    packed_id: &WeightId,
    scales_id: &WeightId,
    prepared: &Fp8MarlinWeight,
) -> Result<WeightComponentPayload<'source>, VNextError> {
    let bytes = if component.id == *packed_id {
        Cow::Owned(prepared.packed_values().to_vec())
    } else if component.id == *scales_id {
        Cow::Owned(
            prepared
                .scales()
                .iter()
                .flat_map(|scale| scale.to_le_bytes())
                .collect(),
        )
    } else {
        return Err(invalid_plan(format!(
            "unknown Marlin FP8 execution component `{}`",
            component.id
        )));
    };
    WeightComponentPayload::from_ordered_sources(
        component,
        component.external_names.clone(),
        source_files,
        component.dimensions.clone(),
        component.physical_element_type(),
        bytes,
    )
}

struct BlockFp8ToMarlinFp8WeightMaterializer {
    descriptor: WeightMaterializerDescriptor,
}

impl BlockFp8ToMarlinFp8WeightMaterializer {
    fn new() -> Result<Self, VNextError> {
        let fingerprint = implementation_fingerprint(&[
            include_str!("marlin_fp8_materializer.rs").as_bytes(),
            include_str!("marlin_repack.rs").as_bytes(),
            BLOCK_FP8_TO_MARLIN_FP8_WEIGHT_MATERIALIZER_ID.as_bytes(),
        ]);
        let quality_contract = ApproximateWeightQualityContract::new(
            BLOCK_FP8_EXECUTION_CONTRACT_FINGERPRINT,
            BLOCK_FP8_QUALITY_VECTOR_DIGEST,
            4,
            CanonicalRational::new(1, 20)?,
            0,
            0,
        )?;
        let descriptor = WeightMaterializerDescriptor::new(
            WeightMaterializerId::new(BLOCK_FP8_TO_MARLIN_FP8_WEIGHT_MATERIALIZER_ID)?,
            BLOCK_FP8_MATERIALIZER_VERSION,
            fingerprint,
            WeightMaterializationFidelity::Approximate,
            BTreeSet::from([CapabilityId::new(MARLIN_FP8_CAPABILITY_ID)?]),
        )?
        .with_approximate_quality_contract(quality_contract)?;
        Ok(Self { descriptor })
    }

    fn candidates(family: &PreparedModelFamily) -> Result<Vec<BlockFp8Candidate>, VNextError> {
        block_fp8_candidates(family)
    }

    fn materialize_group<'source>(
        &self,
        source: &'source dyn WeightComponentSource,
        source_components: &[&WeightComponentSpec],
        execution_components: &[&WeightComponentSpec],
    ) -> Result<Vec<WeightComponentPayload<'source>>, VNextError> {
        if let ([source_component], [execution_component]) =
            (source_components, execution_components)
        {
            if *source_component == *execution_component {
                return source
                    .component(source_component)
                    .map(|payload| vec![payload]);
            }
        }

        let [source_values, source_scales] = source_components else {
            return Err(invalid_plan(
                "block-FP8 to Marlin FP8 materialization requires ordered values and inverse-scale source components",
            ));
        };
        if execution_components.is_empty() || execution_components.len() > 2 {
            return Err(invalid_plan(
                "block-FP8 to Marlin FP8 materialization requires one or both derived components",
            ));
        }
        let logical_dimensions =
            block_fp8_source_component_dimensions(source_values, source_scales).ok_or_else(
                || {
                    invalid_plan(format!(
                    "source components `{}` and `{}` are not an eligible 128x128 block-FP8 pair",
                    source_values.id, source_scales.id
                ))
                },
            )?;
        let [n, k] = logical_dimensions[logical_dimensions.len() - 2..] else {
            unreachable!("block-FP8 source validation requires at least two dimensions")
        };
        let n =
            usize::try_from(n).map_err(|_| invalid_plan("block-FP8 output width exceeds usize"))?;
        let k =
            usize::try_from(k).map_err(|_| invalid_plan("block-FP8 input width exceeds usize"))?;
        let (expected_packed, expected_scales) =
            block_fp8_derived_components(source_values, source_scales, &logical_dimensions)?;
        let mut requested = BTreeSet::new();
        for component in execution_components {
            if !requested.insert(component.id.clone())
                || (**component != expected_packed && **component != expected_scales)
            {
                return Err(invalid_plan(format!(
                    "execution component `{}` is not derived from block-FP8 source pair `{}`, `{}`",
                    component.id, source_values.id, source_scales.id
                )));
            }
        }

        let values_payload = source.component(source_values)?;
        let scales_payload = source.component(source_scales)?;
        let [values_source_file] = values_payload.source_files() else {
            return Err(invalid_plan(format!(
                "block-FP8 values component `{}` must resolve from exactly one checkpoint tensor",
                source_values.id
            )));
        };
        let [scales_source_file] = scales_payload.source_files() else {
            return Err(invalid_plan(format!(
                "block-FP8 inverse-scale component `{}` must resolve from exactly one checkpoint tensor",
                source_scales.id
            )));
        };
        let prepared = prepare_block_fp8_weight_for_fp8_marlin(
            values_payload.bytes(),
            scales_payload.bytes(),
            n,
            k,
            BLOCK_FP8_BLOCK_SHAPE,
        )
        .map_err(|error| {
            invalid_plan(format!(
                "prepare Marlin FP8 components from `{}` and `{}`: {error}",
                source_values.id, source_scales.id
            ))
        })?;
        let (packed_values, scales) = prepared.into_parts();
        let scales = scales
            .into_iter()
            .flat_map(|scale| scale.to_le_bytes())
            .collect::<Vec<_>>();
        let mut packed_values = Some(packed_values);
        let mut scales = Some(scales);
        let mut source_files = vec![values_source_file.clone(), scales_source_file.clone()];
        let requested_count = execution_components.len();
        execution_components
            .iter()
            .enumerate()
            .map(|(index, component)| {
                let bytes = if component.id == expected_packed.id {
                    packed_values
                        .take()
                        .ok_or_else(|| invalid_plan("duplicate Marlin FP8 packed output"))?
                } else {
                    scales
                        .take()
                        .ok_or_else(|| invalid_plan("duplicate Marlin FP8 scale output"))?
                };
                let component_source_files = if index + 1 == requested_count {
                    std::mem::take(&mut source_files)
                } else {
                    source_files.clone()
                };
                WeightComponentPayload::from_ordered_sources(
                    component,
                    component.external_names.clone(),
                    component_source_files,
                    component.dimensions.clone(),
                    component.physical_element_type(),
                    Cow::Owned(bytes),
                )
            })
            .collect()
    }
}

impl WeightMaterializer for BlockFp8ToMarlinFp8WeightMaterializer {
    fn descriptor(&self) -> &WeightMaterializerDescriptor {
        &self.descriptor
    }

    fn execution_schema(
        &self,
        family: &PreparedModelFamily,
        _device: &DeviceDescriptor,
    ) -> Result<WeightSchema, VNextError> {
        let candidates = Self::candidates(family)?;
        if candidates.is_empty() {
            return Ok(family.weight_schema().clone());
        }

        let mut schema = family.weight_schema().clone();
        schema.format_id = WeightFormatId::new(MARLIN_FP8_WEIGHT_FORMAT_ID)?;
        schema.layout_id = WeightLayoutId::new(MARLIN_FP8_WEIGHT_LAYOUT_ID)?;
        let candidate_by_source = candidates
            .iter()
            .map(|candidate| (candidate.source_values_id.clone(), candidate))
            .collect::<BTreeMap<_, _>>();
        let removed_sources = candidates
            .iter()
            .flat_map(|candidate| {
                [
                    candidate.source_values_id.clone(),
                    candidate.source_scales_id.clone(),
                ]
            })
            .collect::<BTreeSet<_>>();
        schema
            .components
            .retain(|component| !removed_sources.contains(&component.id));
        for candidate in &candidates {
            schema.components.push(candidate.packed_component.clone());
            schema.components.push(candidate.scales_component.clone());
        }
        for tensor in &mut schema.tensors {
            tensor.physical_layout =
                replace_block_fp8_leaves(&tensor.physical_layout, &candidate_by_source);
        }
        Ok(schema)
    }

    fn component_sources(
        &self,
        family: &PreparedModelFamily,
        execution_schema: &WeightSchema,
    ) -> Result<BTreeMap<WeightId, Vec<WeightId>>, VNextError> {
        let mut derived_sources = BTreeMap::new();
        for candidate in Self::candidates(family)? {
            let sources = vec![
                candidate.source_values_id.clone(),
                candidate.source_scales_id.clone(),
            ];
            derived_sources.insert(candidate.packed_component.id, sources.clone());
            derived_sources.insert(candidate.scales_component.id, sources);
        }
        execution_schema
            .components
            .iter()
            .map(|component| {
                let sources = derived_sources
                    .get(&component.id)
                    .cloned()
                    .unwrap_or_else(|| vec![component.id.clone()]);
                Ok((component.id.clone(), sources))
            })
            .collect()
    }

    fn materialize_component<'source>(
        &self,
        source: &'source dyn WeightComponentSource,
        source_components: &[&WeightComponentSpec],
        execution_component: &WeightComponentSpec,
    ) -> Result<WeightComponentPayload<'source>, VNextError> {
        self.materialize_group(source, source_components, &[execution_component])?
            .pop()
            .ok_or_else(|| invalid_plan("block-FP8 materializer returned no component"))
    }

    fn materialize_components<'source>(
        &self,
        source: &'source dyn WeightComponentSource,
        source_components: &[&WeightComponentSpec],
        execution_components: &[&WeightComponentSpec],
    ) -> Result<Vec<WeightComponentPayload<'source>>, VNextError> {
        self.materialize_group(source, source_components, execution_components)
    }
}

struct BlockFp8Candidate {
    source_values_id: WeightId,
    source_scales_id: WeightId,
    logical_dimensions: Vec<u64>,
    packed_component: WeightComponentSpec,
    scales_component: WeightComponentSpec,
}

impl BlockFp8Candidate {
    fn execution_layout(&self) -> PhysicalWeightLayout {
        PhysicalWeightLayout::Quantized {
            packed_values: PhysicalWeightComponentBinding::exact_contiguous(
                self.packed_component.id.clone(),
            ),
            packed_dimensions: self.logical_dimensions.clone(),
            scales: PhysicalWeightComponentBinding::exact_contiguous(
                self.scales_component.id.clone(),
            ),
            zero_points: None,
            zero_point_packed_dimensions: None,
            axis_indices: None,
            permutation: None,
            codebook: None,
            group_axis: u32::try_from(self.logical_dimensions.len() - 1)
                .expect("validated block-FP8 rank fits u32"),
            group_padding: PhysicalWeightPadding::Exact,
        }
    }
}

fn block_fp8_candidates(
    family: &PreparedModelFamily,
) -> Result<Vec<BlockFp8Candidate>, VNextError> {
    let schema = family.weight_schema();
    let mut component_references = BTreeMap::<WeightId, usize>::new();
    for tensor in &schema.tensors {
        for component in schema.physical_component_refs(&tensor.id)? {
            *component_references
                .entry(component.id.clone())
                .or_default() += 1;
        }
    }

    let mut candidates = Vec::new();
    let mut selected_sources = BTreeSet::new();
    let mut derived_ids = schema
        .components
        .iter()
        .map(|component| component.id.clone())
        .collect::<BTreeSet<_>>();
    for reference in family.program().weights() {
        let tensor = schema.tensor(&reference.weight_id).ok_or_else(|| {
            invalid_plan(format!(
                "program weight `{}` has no source tensor",
                reference.weight_id
            ))
        })?;
        let mut leaves = Vec::new();
        collect_block_fp8_leaves(
            &tensor.physical_layout,
            &tensor.dimensions,
            schema,
            &mut leaves,
        )?;
        if leaves.is_empty() {
            continue;
        }
        let uses = family
            .program()
            .blocks()
            .iter()
            .flat_map(|block| &block.nodes)
            .flat_map(|node| {
                node.inputs
                    .iter()
                    .enumerate()
                    .filter(move |(_, input)| **input == reference.value_id)
                    .map(move |(ordinal, _)| (node.operation_id.as_str(), ordinal))
            })
            .collect::<Vec<_>>();
        if uses.is_empty()
            || uses.iter().any(|(operation_id, ordinal)| {
                !eligible_block_fp8_projection_use(operation_id, *ordinal)
            })
        {
            return Err(invalid_plan(format!(
                "block-FP8 program weight `{}` is used outside the Marlin FP8 projection contract",
                reference.weight_id
            )));
        }
        if tensor.logical_element_type != ElementType::F16 {
            return Err(invalid_plan(format!(
                "block-FP8 program weight `{}` has logical type {:?}, but Marlin W8A16 requires F16",
                reference.weight_id, tensor.logical_element_type
            )));
        }
        for leaf in leaves {
            if component_references.get(&leaf.values.id) != Some(&1)
                || component_references.get(&leaf.scales.id) != Some(&1)
            {
                return Err(invalid_plan(format!(
                    "block-FP8 source pair `{}`, `{}` is shared across physical tensors",
                    leaf.values.id, leaf.scales.id
                )));
            }
            if selected_sources.contains(&leaf.values.id)
                || selected_sources.contains(&leaf.scales.id)
            {
                return Err(invalid_plan(format!(
                    "block-FP8 source pair `{}`, `{}` is selected more than once",
                    leaf.values.id, leaf.scales.id
                )));
            }
            selected_sources.insert(leaf.values.id.clone());
            selected_sources.insert(leaf.scales.id.clone());
            let (packed_component, scales_component) =
                block_fp8_derived_components(leaf.values, leaf.scales, &leaf.logical_dimensions)?;
            if !derived_ids.insert(packed_component.id.clone())
                || !derived_ids.insert(scales_component.id.clone())
            {
                return Err(invalid_plan(format!(
                    "block-FP8 derived component identity collides for `{}` and `{}`",
                    leaf.values.id, leaf.scales.id
                )));
            }
            candidates.push(BlockFp8Candidate {
                source_values_id: leaf.values.id.clone(),
                source_scales_id: leaf.scales.id.clone(),
                logical_dimensions: leaf.logical_dimensions,
                packed_component,
                scales_component,
            });
        }
    }
    Ok(candidates)
}

struct BlockFp8SourceLeaf<'schema> {
    values: &'schema WeightComponentSpec,
    scales: &'schema WeightComponentSpec,
    logical_dimensions: Vec<u64>,
}

fn collect_block_fp8_leaves<'schema>(
    layout: &PhysicalWeightLayout,
    logical_dimensions: &[u64],
    schema: &'schema WeightSchema,
    leaves: &mut Vec<BlockFp8SourceLeaf<'schema>>,
) -> Result<(), VNextError> {
    match layout {
        PhysicalWeightLayout::Composite { parts } => {
            for part in parts {
                collect_block_fp8_leaves(&part.layout, &part.extents, schema, leaves)?;
            }
        }
        PhysicalWeightLayout::QuantizedBlockGrid {
            packed_values,
            packed_dimensions,
            scales,
            block_axes,
        } => {
            let values = schema
                .components
                .iter()
                .find(|component| component.id == packed_values.component_id)
                .ok_or_else(|| {
                    invalid_plan(format!(
                        "block-FP8 layout references absent values component `{}`",
                        packed_values.component_id
                    ))
                })?;
            if !is_block_fp8_source_format(&values.encoding) {
                return Ok(());
            }
            let scale_component = schema
                .components
                .iter()
                .find(|component| component.id == scales.component_id)
                .ok_or_else(|| {
                    invalid_plan(format!(
                        "block-FP8 layout references absent scale component `{}`",
                        scales.component_id
                    ))
                })?;
            if logical_dimensions.len() < 2 || logical_dimensions.len() > 3 {
                return Err(invalid_plan(format!(
                    "block-FP8 source pair `{}`, `{}` has unsupported logical rank {}",
                    values.id,
                    scale_component.id,
                    logical_dimensions.len()
                )));
            }
            let expected_block_axes = [
                u32::try_from(logical_dimensions.len() - 2)
                    .map_err(|_| invalid_plan("block-FP8 rank exceeds u32"))?,
                u32::try_from(logical_dimensions.len() - 1)
                    .map_err(|_| invalid_plan("block-FP8 rank exceeds u32"))?,
            ];
            if !is_exact_contiguous(&packed_values.storage)
                || !is_exact_contiguous(&scales.storage)
                || packed_dimensions != logical_dimensions
                || logical_dimensions[..logical_dimensions.len() - 2]
                    .iter()
                    .any(|extent| *extent != 1)
                || *block_axes != expected_block_axes
                || block_fp8_source_component_dimensions(values, scale_component).as_deref()
                    != Some(logical_dimensions)
            {
                return Err(invalid_plan(format!(
                    "block-FP8 source pair `{}`, `{}` differs from the exact contiguous 128x128 source contract",
                    values.id, scale_component.id
                )));
            }
            leaves.push(BlockFp8SourceLeaf {
                values,
                scales: scale_component,
                logical_dimensions: logical_dimensions.to_vec(),
            });
        }
        _ => {}
    }
    Ok(())
}

fn replace_block_fp8_leaves(
    layout: &PhysicalWeightLayout,
    candidate_by_source: &BTreeMap<WeightId, &BlockFp8Candidate>,
) -> PhysicalWeightLayout {
    match layout {
        PhysicalWeightLayout::Composite { parts } => PhysicalWeightLayout::Composite {
            parts: parts
                .iter()
                .map(|part| ferrum_interfaces::vnext::CompositeWeightPart {
                    layout: Box::new(replace_block_fp8_leaves(&part.layout, candidate_by_source)),
                    logical_offsets: part.logical_offsets.clone(),
                    extents: part.extents.clone(),
                })
                .collect(),
        },
        PhysicalWeightLayout::QuantizedBlockGrid { packed_values, .. } => candidate_by_source
            .get(&packed_values.component_id)
            .map_or_else(|| layout.clone(), |candidate| candidate.execution_layout()),
        _ => layout.clone(),
    }
}

fn block_fp8_source_component_dimensions(
    values: &WeightComponentSpec,
    scales: &WeightComponentSpec,
) -> Option<Vec<u64>> {
    let dimensions = &values.dimensions;
    if dimensions.len() < 2
        || dimensions.len() > 3
        || dimensions[..dimensions.len() - 2]
            .iter()
            .any(|extent| *extent != 1)
        || values.role != WeightComponentRole::PackedValues
        || values.external_names.len() != 1
        || scales.role != WeightComponentRole::Scales
        || scales.external_names.len() != 1
        || values.required != scales.required
        || !block_fp8_source_quantization_matches(&values.encoding)
        || scales.encoding
            != (WeightEncoding::Dense {
                element_type: ElementType::Bf16,
            })
    {
        return None;
    }
    let n = usize::try_from(dimensions[dimensions.len() - 2]).ok()?;
    let k = usize::try_from(dimensions[dimensions.len() - 1]).ok()?;
    if !marlin_fp8_projection_shape_supported(n, k) {
        return None;
    }
    let mut expected_scale_dimensions = dimensions.clone();
    let rank = expected_scale_dimensions.len();
    expected_scale_dimensions[rank - 2] = expected_scale_dimensions[rank - 2].div_ceil(128);
    expected_scale_dimensions[rank - 1] = expected_scale_dimensions[rank - 1].div_ceil(128);
    (scales.dimensions == expected_scale_dimensions).then(|| dimensions.clone())
}

fn block_fp8_source_quantization_matches(encoding: &WeightEncoding) -> bool {
    let WeightEncoding::Quantized(quantization) = encoding else {
        return false;
    };
    let Some(block_shape) = quantization.grouping.block_shape_2d() else {
        return false;
    };
    quantization.format_id.as_str() == BLOCK_FP8_SOURCE_QUANTIZATION_FORMAT_ID
        && quantization.bits_per_weight == 8
        && block_shape.map(|extent| extent.get()) == [128, 128]
        && quantization.packing == QuantizationPacking::Linear
        && quantization.scale_type == ElementType::Bf16
        && quantization.zero_point_type.is_none()
}

fn is_block_fp8_source_format(encoding: &WeightEncoding) -> bool {
    matches!(
        encoding,
        WeightEncoding::Quantized(quantization)
            if quantization.format_id.as_str() == BLOCK_FP8_SOURCE_QUANTIZATION_FORMAT_ID
    )
}

fn block_fp8_derived_components(
    source_values: &WeightComponentSpec,
    source_scales: &WeightComponentSpec,
    logical_dimensions: &[u64],
) -> Result<(WeightComponentSpec, WeightComponentSpec), VNextError> {
    let packed_id = block_fp8_derived_component_id(
        &source_values.id,
        &source_scales.id,
        DerivedComponentKind::Packed,
    )?;
    let scales_id = block_fp8_derived_component_id(
        &source_values.id,
        &source_scales.id,
        DerivedComponentKind::Scales,
    )?;
    let mut scales_dimensions = logical_dimensions.to_vec();
    let group_axis = scales_dimensions
        .len()
        .checked_sub(1)
        .ok_or_else(|| invalid_plan("block-FP8 logical shape is empty"))?;
    scales_dimensions[group_axis] = 1;
    let required = source_values.required && source_scales.required;
    let quantization = marlin_fp8_quantization_spec()?;
    quantization.validate()?;
    Ok((
        WeightComponentSpec {
            id: packed_id,
            role: WeightComponentRole::PackedValues,
            external_names: block_fp8_derived_external_names(
                &source_values.id,
                &source_scales.id,
                DerivedComponentKind::Packed,
            ),
            dimensions: logical_dimensions.to_vec(),
            encoding: WeightEncoding::Quantized(quantization),
            required,
        },
        WeightComponentSpec {
            id: scales_id,
            role: WeightComponentRole::Scales,
            external_names: block_fp8_derived_external_names(
                &source_values.id,
                &source_scales.id,
                DerivedComponentKind::Scales,
            ),
            dimensions: scales_dimensions,
            encoding: WeightEncoding::Dense {
                element_type: ElementType::F16,
            },
            required,
        },
    ))
}

fn block_fp8_derived_component_id(
    source_values_id: &WeightId,
    source_scales_id: &WeightId,
    kind: DerivedComponentKind,
) -> Result<WeightId, VNextError> {
    let digest = block_fp8_source_pair_digest(source_values_id, source_scales_id);
    WeightId::new(format!(
        "{BLOCK_FP8_DERIVED_COMPONENT_PREFIX}.{digest}.{}",
        kind.as_str()
    ))
}

fn block_fp8_derived_external_names(
    source_values_id: &WeightId,
    source_scales_id: &WeightId,
    kind: DerivedComponentKind,
) -> Vec<String> {
    let digest = block_fp8_source_pair_digest(source_values_id, source_scales_id);
    (0..2)
        .map(|index| {
            format!(
                "execution.block-fp8-marlin-fp8.{digest}.{}.{index}",
                kind.as_str()
            )
        })
        .collect()
}

fn block_fp8_source_pair_digest(
    source_values_id: &WeightId,
    source_scales_id: &WeightId,
) -> String {
    let mut hash = Sha256::new();
    for source_id in [source_values_id, source_scales_id] {
        hash.update((source_id.as_str().len() as u64).to_le_bytes());
        hash.update(source_id.as_str().as_bytes());
    }
    format!("{:x}", hash.finalize())
}

fn is_exact_contiguous(storage: &PhysicalStorageLayout) -> bool {
    matches!(
        storage,
        PhysicalStorageLayout::Contiguous {
            padding: PhysicalWeightPadding::Exact
        }
    )
}

fn eligible_block_fp8_projection_use(operation_id: &str, ordinal: usize) -> bool {
    (operation_id == DENSE_LINEAR_OPERATION_ID && ordinal == 1)
        || (operation_id == GATED_DELTA_RECURRENT_ATTENTION_OPERATION_ID
            && matches!(ordinal, 2 | 7))
        || (operation_id == CAUSAL_PAGED_ATTENTION_OPERATION_ID && matches!(ordinal, 2 | 3 | 4 | 5))
        || (operation_id == DENSE_SWIGLU_OPERATION_ID && matches!(ordinal, 1 | 2))
}

fn implementation_fingerprint(parts: &[&[u8]]) -> String {
    let mut hash = Sha256::new();
    for part in parts {
        hash.update((part.len() as u64).to_le_bytes());
        hash.update(part);
    }
    format!("{:x}", hash.finalize())
}

fn invalid_plan(reason: impl Into<String>) -> VNextError {
    VNextError::InvalidExecutionPlan {
        reason: reason.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::num::NonZeroU32;

    struct ZeroWeightSource;

    impl WeightComponentSource for ZeroWeightSource {
        fn component<'source>(
            &'source self,
            component: &WeightComponentSpec,
        ) -> Result<WeightComponentPayload<'source>, VNextError> {
            let byte_len = usize::try_from(component.physical_bytes()?)
                .map_err(|_| invalid_plan("test component exceeds address space"))?;
            WeightComponentPayload::from_ordered_sources(
                component,
                component.external_names.clone(),
                vec!["model.safetensors".to_owned(); component.external_names.len()],
                component.dimensions.clone(),
                component.physical_element_type(),
                vec![0_u8; byte_len],
            )
        }
    }

    struct BlockFp8TestSource {
        values_id: WeightId,
        scales_id: WeightId,
        values: Vec<u8>,
        scales: Vec<u8>,
    }

    impl WeightComponentSource for BlockFp8TestSource {
        fn component<'source>(
            &'source self,
            component: &WeightComponentSpec,
        ) -> Result<WeightComponentPayload<'source>, VNextError> {
            let (bytes, source_file) = if component.id == self.values_id {
                (self.values.as_slice(), "values.safetensors")
            } else if component.id == self.scales_id {
                (self.scales.as_slice(), "scales.safetensors")
            } else {
                return Err(invalid_plan(format!(
                    "unknown block-FP8 test component `{}`",
                    component.id
                )));
            };
            WeightComponentPayload::from_ordered_sources(
                component,
                component.external_names.clone(),
                vec![source_file.to_owned()],
                component.dimensions.clone(),
                component.physical_element_type(),
                Cow::Borrowed(bytes),
            )
        }
    }

    fn test_block_fp8_components() -> (WeightComponentSpec, WeightComponentSpec) {
        let values = WeightComponentSpec {
            id: WeightId::new("component.test.block_fp8.values").unwrap(),
            role: WeightComponentRole::PackedValues,
            external_names: vec!["model.layers.0.proj.weight".to_owned()],
            dimensions: vec![256, 128],
            encoding: WeightEncoding::Quantized(QuantizationSpec {
                format_id: QuantizationFormatId::new(BLOCK_FP8_SOURCE_QUANTIZATION_FORMAT_ID)
                    .unwrap(),
                bits_per_weight: 8,
                grouping: QuantizationGrouping::block_2d([
                    NonZeroU32::new(128).unwrap(),
                    NonZeroU32::new(128).unwrap(),
                ]),
                packing: QuantizationPacking::Linear,
                scale_type: ElementType::Bf16,
                zero_point_type: None,
            }),
            required: true,
        };
        let scales = WeightComponentSpec {
            id: WeightId::new("component.test.block_fp8.inverse_scales").unwrap(),
            role: WeightComponentRole::Scales,
            external_names: vec!["model.layers.0.proj.weight_scale_inv".to_owned()],
            dimensions: vec![2, 1],
            encoding: WeightEncoding::Dense {
                element_type: ElementType::Bf16,
            },
            required: true,
        };
        (values, scales)
    }

    #[test]
    fn eligibility_is_operation_and_ordinal_driven() {
        assert!(eligible_projection_use(DENSE_LINEAR_OPERATION_ID, 1));
        assert!(eligible_projection_use(
            GATED_DELTA_RECURRENT_ATTENTION_OPERATION_ID,
            2
        ));
        assert!(eligible_projection_use(
            GATED_DELTA_RECURRENT_ATTENTION_OPERATION_ID,
            7
        ));
        assert!(!eligible_projection_use(DENSE_LINEAR_OPERATION_ID, 0));
        assert!(!eligible_projection_use(
            GATED_DELTA_RECURRENT_ATTENTION_OPERATION_ID,
            3
        ));
        assert!(!eligible_projection_use(
            GATED_DELTA_RECURRENT_ATTENTION_OPERATION_ID,
            8
        ));
        assert!(!eligible_projection_use(
            GATED_DELTA_RECURRENT_ATTENTION_OPERATION_ID,
            4
        ));
    }

    #[test]
    fn derived_id_is_bounded_and_independent_of_source_length() {
        let source = WeightId::new(format!("component.{}", "x".repeat(140))).unwrap();
        let packed = derived_component_id(&source, DerivedComponentKind::Packed).unwrap();
        let scales = derived_component_id(&source, DerivedComponentKind::Scales).unwrap();
        assert_ne!(packed, scales);
        assert!(packed.as_str().len() <= 160);
        assert!(scales.as_str().len() <= 160);
    }

    #[test]
    fn projection_shape_is_safe_for_all_admitted_token_counts() {
        assert!(marlin_fp8_projection_shape_supported(256, 128));
        assert!(marlin_fp8_projection_shape_supported(2_048, 4_096));
        assert!(!marlin_fp8_projection_shape_supported(64, 2_048));
        assert!(!marlin_fp8_projection_shape_supported(128, 2_048));
        assert!(!marlin_fp8_projection_shape_supported(256, 96));
    }

    #[test]
    fn channelwise_quantization_abi_is_shape_relative() {
        let quantization = marlin_fp8_quantization_spec().unwrap();
        assert_eq!(quantization.grouping, QuantizationGrouping::WholeAxis);
        assert_eq!(quantization.grouping.resolved_size(2_048), 2_048);
        assert_eq!(quantization.grouping.resolved_size(4_096), 4_096);
        quantization.validate().unwrap();
    }

    #[test]
    fn unchanged_components_borrow_their_source_payload() {
        let component = WeightComponentSpec {
            id: WeightId::new("component.global.embed_tokens").unwrap(),
            role: WeightComponentRole::Values,
            external_names: vec!["model.embed_tokens.weight".to_owned()],
            dimensions: vec![2, 4],
            encoding: WeightEncoding::Dense {
                element_type: ElementType::F16,
            },
            required: true,
        };
        let materializer = MarlinFp8WeightMaterializer::new().unwrap();
        let payloads = materializer
            .materialize_components(&ZeroWeightSource, &[&component], &[&component])
            .unwrap();

        assert_eq!(payloads.len(), 1);
        assert_eq!(payloads[0].component_id(), &component.id);
        assert_eq!(payloads[0].bytes().len(), 16);
    }

    #[test]
    fn block_fp8_descriptor_carries_fixed_quality_contract() {
        let materializer = BlockFp8ToMarlinFp8WeightMaterializer::new().unwrap();
        let descriptor = materializer.descriptor();
        assert_eq!(
            descriptor.id().as_str(),
            BLOCK_FP8_TO_MARLIN_FP8_WEIGHT_MATERIALIZER_ID
        );
        assert_eq!(descriptor.version(), ContractVersion::new(1, 0));
        assert_eq!(
            descriptor.fidelity(),
            WeightMaterializationFidelity::Approximate
        );
        assert_eq!(
            descriptor.required_capabilities(),
            &BTreeSet::from([CapabilityId::new(MARLIN_FP8_CAPABILITY_ID).unwrap()])
        );
        let quality = descriptor
            .approximate_quality_contract()
            .expect("approximate materializer quality contract");
        assert_eq!(
            quality.execution_contract_fingerprint(),
            BLOCK_FP8_EXECUTION_CONTRACT_FINGERPRINT
        );
        assert_eq!(
            quality.quality_vector_digest(),
            BLOCK_FP8_QUALITY_VECTOR_DIGEST
        );
        assert_eq!(quality.required_case_count(), 4);
        assert_eq!(
            quality.relative_l2_max(),
            CanonicalRational::new(1, 20).unwrap()
        );
        assert_eq!(quality.nan_count_max(), 0);
        assert_eq!(quality.inf_count_max(), 0);
    }

    #[test]
    fn block_fp8_eligibility_covers_all_projection_contracts() {
        assert!(eligible_block_fp8_projection_use(
            DENSE_LINEAR_OPERATION_ID,
            1
        ));
        for ordinal in [2, 7] {
            assert!(eligible_block_fp8_projection_use(
                GATED_DELTA_RECURRENT_ATTENTION_OPERATION_ID,
                ordinal
            ));
        }
        for ordinal in [2, 3, 4, 5] {
            assert!(eligible_block_fp8_projection_use(
                CAUSAL_PAGED_ATTENTION_OPERATION_ID,
                ordinal
            ));
        }
        for ordinal in [1, 2] {
            assert!(eligible_block_fp8_projection_use(
                DENSE_SWIGLU_OPERATION_ID,
                ordinal
            ));
        }
        assert!(!eligible_block_fp8_projection_use(
            DENSE_LINEAR_OPERATION_ID,
            0
        ));
        assert!(!eligible_block_fp8_projection_use(
            GATED_DELTA_RECURRENT_ATTENTION_OPERATION_ID,
            3
        ));
        assert!(!eligible_block_fp8_projection_use(
            CAUSAL_PAGED_ATTENTION_OPERATION_ID,
            1
        ));
        assert!(!eligible_block_fp8_projection_use(
            DENSE_SWIGLU_OPERATION_ID,
            3
        ));
    }

    #[test]
    fn block_fp8_source_accepts_rank_two_and_unit_prefix_rank_three_only() {
        let (values, inverse_scales) = test_block_fp8_components();
        assert_eq!(
            block_fp8_source_component_dimensions(&values, &inverse_scales),
            Some(vec![256, 128])
        );

        let mut stacked_values = values.clone();
        stacked_values.dimensions = vec![1, 256, 128];
        let mut stacked_scales = inverse_scales.clone();
        stacked_scales.dimensions = vec![1, 2, 1];
        assert_eq!(
            block_fp8_source_component_dimensions(&stacked_values, &stacked_scales),
            Some(vec![1, 256, 128])
        );
        let (packed, scales) = block_fp8_derived_components(
            &stacked_values,
            &stacked_scales,
            &stacked_values.dimensions,
        )
        .unwrap();
        assert_eq!(packed.dimensions, [1, 256, 128]);
        assert_eq!(scales.dimensions, [1, 256, 1]);
        let candidate = BlockFp8Candidate {
            source_values_id: stacked_values.id.clone(),
            source_scales_id: stacked_scales.id.clone(),
            logical_dimensions: stacked_values.dimensions.clone(),
            packed_component: packed,
            scales_component: scales,
        };
        let PhysicalWeightLayout::Quantized { group_axis, .. } = candidate.execution_layout()
        else {
            panic!("rank-three source must produce quantized execution layout")
        };
        assert_eq!(group_axis, 2);

        stacked_values.dimensions[0] = 2;
        assert!(block_fp8_source_component_dimensions(&stacked_values, &stacked_scales).is_none());
    }

    #[test]
    fn block_fp8_materialization_consumes_ordered_pair_once_for_both_outputs() {
        let (values, inverse_scales) = test_block_fp8_components();
        let (packed, scales) =
            block_fp8_derived_components(&values, &inverse_scales, &values.dimensions).unwrap();
        assert_eq!(packed.external_names.len(), 2);
        assert_eq!(scales.external_names.len(), 2);
        let synthetic_names = packed
            .external_names
            .iter()
            .chain(&scales.external_names)
            .collect::<BTreeSet<_>>();
        assert_eq!(synthetic_names.len(), 4);

        let source = BlockFp8TestSource {
            values_id: values.id.clone(),
            scales_id: inverse_scales.id.clone(),
            values: vec![0_u8; 256 * 128],
            scales: [half::bf16::from_f32(1.0).to_le_bytes(); 2].concat(),
        };
        let materializer = BlockFp8ToMarlinFp8WeightMaterializer::new().unwrap();
        let payloads = materializer
            .materialize_components(&source, &[&values, &inverse_scales], &[&packed, &scales])
            .unwrap();

        assert_eq!(payloads.len(), 2);
        assert_eq!(payloads[0].component_id(), &packed.id);
        assert_eq!(payloads[0].bytes().len(), 256 * 128);
        assert_eq!(payloads[1].component_id(), &scales.id);
        assert_eq!(payloads[1].bytes().len(), 256 * 2);
        for payload in payloads {
            assert_eq!(
                payload.source_files(),
                ["values.safetensors", "scales.safetensors"]
            );
        }
    }

    #[test]
    fn recursive_rewrite_preserves_composite_dense_leaf_and_offsets() {
        let (values, inverse_scales) = test_block_fp8_components();
        let (packed_component, scales_component) =
            block_fp8_derived_components(&values, &inverse_scales, &values.dimensions).unwrap();
        let candidate = BlockFp8Candidate {
            source_values_id: values.id.clone(),
            source_scales_id: inverse_scales.id.clone(),
            logical_dimensions: values.dimensions.clone(),
            packed_component,
            scales_component,
        };
        let dense_id = WeightId::new("component.test.dense").unwrap();
        let original = PhysicalWeightLayout::Composite {
            parts: vec![
                ferrum_interfaces::vnext::CompositeWeightPart {
                    layout: Box::new(PhysicalWeightLayout::QuantizedBlockGrid {
                        packed_values: PhysicalWeightComponentBinding::exact_contiguous(
                            values.id.clone(),
                        ),
                        packed_dimensions: values.dimensions.clone(),
                        scales: PhysicalWeightComponentBinding::exact_contiguous(
                            inverse_scales.id.clone(),
                        ),
                        block_axes: [0, 1],
                    }),
                    logical_offsets: vec![0, 0],
                    extents: vec![256, 128],
                },
                ferrum_interfaces::vnext::CompositeWeightPart {
                    layout: Box::new(PhysicalWeightLayout::Dense {
                        component_id: dense_id.clone(),
                    }),
                    logical_offsets: vec![256, 0],
                    extents: vec![256, 128],
                },
            ],
        };
        let rewritten = replace_block_fp8_leaves(
            &original,
            &BTreeMap::from([(values.id.clone(), &candidate)]),
        );
        let PhysicalWeightLayout::Composite { parts } = rewritten else {
            panic!("composite layout must remain composite")
        };
        assert_eq!(parts[0].logical_offsets, [0, 0]);
        assert_eq!(parts[0].extents, [256, 128]);
        let PhysicalWeightLayout::Quantized {
            packed_values,
            scales,
            group_axis,
            ..
        } = parts[0].layout.as_ref()
        else {
            panic!("block-FP8 leaf must become Marlin FP8")
        };
        assert_eq!(packed_values.component_id, candidate.packed_component.id);
        assert_eq!(scales.component_id, candidate.scales_component.id);
        assert_eq!(*group_axis, 1);
        assert_eq!(parts[1].logical_offsets, [256, 0]);
        assert_eq!(parts[1].extents, [256, 128]);
        assert_eq!(
            parts[1].layout.as_ref(),
            &PhysicalWeightLayout::Dense {
                component_id: dense_id
            }
        );
    }
}
