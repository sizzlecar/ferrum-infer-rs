//! Exact GPT-OSS MXFP4 to CUDA Marlin execution-weight preparation.
//!
//! The source E2M1 nibbles and E8M0 scale bytes remain quantized throughout
//! this boundary. Each expert is staged independently: packed bytes are viewed
//! as GPTQ-compatible little-endian I32 words and transposed before the native
//! Marlin repack, while scale bytes are transposed and permuted into the vLLM
//! MXFP4 ABI. No dense or BF16 expert matrix is ever constructed.

use std::collections::{BTreeMap, BTreeSet};

use ferrum_interfaces::vnext::{
    CapabilityId, ContractVersion, DeviceDescriptor, ElementType, ModelFamilyId,
    PhysicalStorageLayout, PhysicalWeightComponentBinding, PhysicalWeightLayout,
    PhysicalWeightPadding, PreparedModelFamily, QuantizationFormatId, QuantizationGrouping,
    QuantizationPacking, QuantizationSpec, StaticWeightTransformPlan, VNextError,
    WeightComponentPayload, WeightComponentRole, WeightComponentSource, WeightComponentSpec,
    WeightEncoding, WeightFormatId, WeightId, WeightLayoutId, WeightMaterializationFidelity,
    WeightMaterializer, WeightMaterializerDescriptor, WeightMaterializerId, WeightSchema,
    GPT_OSS_ROUTED_CLAMPED_SWIGLU_MOE_MXFP4_BF16_CAPABILITY_ID,
};
use sha2::{Digest, Sha256};

#[cfg(test)]
use crate::marlin_repack::repack_scales_to_marlin;

pub const GPT_OSS_MXFP4_TO_MARLIN_WEIGHT_MATERIALIZER_ID: &str =
    "weight-materializer.cuda.gpt-oss-mxfp4-to-marlin";
pub const GPT_OSS_MXFP4_SOURCE_WEIGHT_FORMAT_ID: &str =
    "weight-format.safetensors.gpt-oss-mxfp4-source";
pub const GPT_OSS_MXFP4_SOURCE_WEIGHT_LAYOUT_ID: &str =
    "weight-layout.gpt_oss.mxfp4.e2m1_e8m0.group32.expert_major";
pub const GPT_OSS_MXFP4_SOURCE_QUANTIZATION_FORMAT_ID: &str =
    "quantization.safetensors.mxfp4-e2m1-e8m0-group32-lsb-even";
pub const GPT_OSS_MXFP4_MARLIN_WEIGHT_FORMAT_ID: &str =
    "weight-format.execution.cuda.gpt-oss-mxfp4-marlin";
pub const GPT_OSS_MXFP4_MARLIN_WEIGHT_LAYOUT_ID: &str =
    "weight-layout.execution.cuda.gpt-oss-mxfp4-marlin.expert-major";
pub const GPT_OSS_MXFP4_MARLIN_QUANTIZATION_FORMAT_ID: &str =
    "quantization.marlin.mxfp4-e2m1-e8m0-group32";

const MATERIALIZER_VERSION: ContractVersion = ContractVersion::new(1, 0);
const SOURCE_SCHEMA_VERSION: ContractVersion = ContractVersion::new(1, 0);
const MXFP4_GROUP_SIZE: usize = 32;
const MXFP4_PACKED_BYTES_PER_GROUP: usize = 16;
const MARLIN_OUTPUT_TILE: usize = 64;
const DERIVED_COMPONENT_PREFIX: &str = "component.execution.gpt-oss-mxfp4-marlin";

pub fn gpt_oss_mxfp4_to_marlin_weight_materializer(
) -> Result<Box<dyn WeightMaterializer>, VNextError> {
    Ok(Box::new(GptOssMxfp4ToMarlinWeightMaterializer::new()?))
}

struct GptOssMxfp4ToMarlinWeightMaterializer {
    descriptor: WeightMaterializerDescriptor,
}

impl GptOssMxfp4ToMarlinWeightMaterializer {
    fn new() -> Result<Self, VNextError> {
        let fingerprint = implementation_fingerprint(&[
            include_str!("mxfp4_marlin_materializer.rs").as_bytes(),
            include_str!("marlin_repack.rs").as_bytes(),
            GPT_OSS_MXFP4_TO_MARLIN_WEIGHT_MATERIALIZER_ID.as_bytes(),
        ]);
        Ok(Self {
            descriptor: WeightMaterializerDescriptor::new(
                WeightMaterializerId::new(GPT_OSS_MXFP4_TO_MARLIN_WEIGHT_MATERIALIZER_ID)?,
                MATERIALIZER_VERSION,
                fingerprint,
                WeightMaterializationFidelity::Exact,
                BTreeSet::from([CapabilityId::new(
                    GPT_OSS_ROUTED_CLAMPED_SWIGLU_MOE_MXFP4_BF16_CAPABILITY_ID,
                )?]),
            )?,
        })
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

        let [source_blocks, source_scales] = source_components else {
            return Err(invalid_plan(
                "GPT-OSS MXFP4 Marlin materialization requires ordered blocks and E8M0 scale source components",
            ));
        };
        let candidate = candidate_from_source_pair(source_blocks, source_scales)?;
        if execution_components.is_empty() || execution_components.len() > 2 {
            return Err(invalid_plan(
                "GPT-OSS MXFP4 Marlin materialization requires one or both derived components",
            ));
        }
        let mut requested = BTreeSet::new();
        for component in execution_components {
            if !requested.insert(component.id.clone())
                || (**component != candidate.packed_component
                    && **component != candidate.scales_component)
            {
                return Err(invalid_plan(format!(
                    "execution component `{}` is not derived from GPT-OSS MXFP4 source pair `{}`, `{}`",
                    component.id, source_blocks.id, source_scales.id
                )));
            }
        }

        Err(invalid_plan(format!(
            "GPT-OSS MXFP4 execution components derived from `{}` and `{}` require the typed per-expert MXFP4-to-Marlin static device transform; host materialization is forbidden",
            source_blocks.id, source_scales.id
        )))
    }
}

impl WeightMaterializer for GptOssMxfp4ToMarlinWeightMaterializer {
    fn descriptor(&self) -> &WeightMaterializerDescriptor {
        &self.descriptor
    }

    fn execution_schema(
        &self,
        family: &PreparedModelFamily,
        _device: &DeviceDescriptor,
    ) -> Result<WeightSchema, VNextError> {
        derive_execution_schema(family.weight_schema(), family.family_id())
    }

    fn component_sources(
        &self,
        family: &PreparedModelFamily,
        execution_schema: &WeightSchema,
    ) -> Result<BTreeMap<WeightId, Vec<WeightId>>, VNextError> {
        let expected = derive_execution_schema(family.weight_schema(), family.family_id())?;
        if &expected != execution_schema {
            return Err(invalid_plan(
                "GPT-OSS MXFP4 execution schema differs from the exact materializer output",
            ));
        }
        let candidates = collect_candidates(family.weight_schema())?;
        let mut derived_sources = BTreeMap::new();
        for candidate in candidates {
            let sources = vec![
                candidate.source_blocks_id.clone(),
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

    fn static_weight_transforms(
        &self,
        family: &PreparedModelFamily,
        execution_schema: &WeightSchema,
    ) -> Result<Vec<StaticWeightTransformPlan>, VNextError> {
        mxfp4_marlin_transform_descriptors(family, execution_schema)?
            .into_iter()
            .map(|descriptor| {
                Ok(StaticWeightTransformPlan::GptOssMxfp4ToMarlin {
                    source_blocks_id: descriptor.source_blocks_id,
                    source_scales_id: descriptor.source_scales_id,
                    packed_values_id: descriptor.packed_values_id,
                    scales_id: descriptor.scales_id,
                    logical_dimensions: descriptor.logical_dimensions,
                })
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
            .ok_or_else(|| invalid_plan("GPT-OSS MXFP4 materializer returned no component"))
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Mxfp4MarlinTransformDescriptor {
    pub(crate) source_blocks_id: WeightId,
    pub(crate) source_scales_id: WeightId,
    pub(crate) packed_values_id: WeightId,
    pub(crate) scales_id: WeightId,
    pub(crate) logical_dimensions: Vec<u64>,
}

/// Exact fields needed by the pending typed static-device transform plan.
pub(crate) fn mxfp4_marlin_transform_descriptors(
    family: &PreparedModelFamily,
    execution_schema: &WeightSchema,
) -> Result<Vec<Mxfp4MarlinTransformDescriptor>, VNextError> {
    let expected = derive_execution_schema(family.weight_schema(), family.family_id())?;
    if &expected != execution_schema {
        return Err(invalid_plan(
            "GPT-OSS MXFP4 execution schema differs from the exact materializer output",
        ));
    }
    collect_candidates(family.weight_schema())?
        .into_iter()
        .map(|candidate| Ok(candidate.transform_descriptor()))
        .collect()
}

#[derive(Debug, Clone)]
struct Mxfp4MarlinCandidate {
    weight_id: WeightId,
    source_blocks_id: WeightId,
    source_scales_id: WeightId,
    logical_dimensions: Vec<u64>,
    packed_component: WeightComponentSpec,
    scales_component: WeightComponentSpec,
}

impl Mxfp4MarlinCandidate {
    fn execution_layout(&self) -> PhysicalWeightLayout {
        PhysicalWeightLayout::Quantized {
            packed_values: PhysicalWeightComponentBinding::exact_contiguous(
                self.packed_component.id.clone(),
            ),
            packed_dimensions: self.packed_component.dimensions.clone(),
            scales: PhysicalWeightComponentBinding::exact_contiguous(
                self.scales_component.id.clone(),
            ),
            zero_points: None,
            zero_point_packed_dimensions: None,
            axis_indices: None,
            permutation: None,
            codebook: None,
            group_axis: 2,
            group_padding: PhysicalWeightPadding::Exact,
        }
    }

    fn transform_descriptor(&self) -> Mxfp4MarlinTransformDescriptor {
        Mxfp4MarlinTransformDescriptor {
            source_blocks_id: self.source_blocks_id.clone(),
            source_scales_id: self.source_scales_id.clone(),
            packed_values_id: self.packed_component.id.clone(),
            scales_id: self.scales_component.id.clone(),
            logical_dimensions: self.logical_dimensions.clone(),
        }
    }
}

fn derive_execution_schema(
    source_schema: &WeightSchema,
    family_id: &ModelFamilyId,
) -> Result<WeightSchema, VNextError> {
    let candidates = collect_candidates(source_schema)?;
    let by_weight = candidates
        .iter()
        .map(|candidate| (&candidate.weight_id, candidate))
        .collect::<BTreeMap<_, _>>();
    let removed = candidates
        .iter()
        .flat_map(|candidate| {
            [
                candidate.source_blocks_id.clone(),
                candidate.source_scales_id.clone(),
            ]
        })
        .collect::<BTreeSet<_>>();

    let mut schema = source_schema.clone();
    schema.format_id = WeightFormatId::new(GPT_OSS_MXFP4_MARLIN_WEIGHT_FORMAT_ID)?;
    schema.layout_id = WeightLayoutId::new(GPT_OSS_MXFP4_MARLIN_WEIGHT_LAYOUT_ID)?;
    schema.version = MATERIALIZER_VERSION;
    schema
        .components
        .retain(|component| !removed.contains(&component.id));
    for candidate in &candidates {
        schema.components.push(candidate.packed_component.clone());
        schema.components.push(candidate.scales_component.clone());
    }
    for tensor in &mut schema.tensors {
        if let Some(candidate) = by_weight.get(&tensor.id) {
            tensor.physical_layout = candidate.execution_layout();
        }
    }
    schema.validate(family_id)?;
    Ok(schema)
}

fn collect_candidates(
    source_schema: &WeightSchema,
) -> Result<Vec<Mxfp4MarlinCandidate>, VNextError> {
    if source_schema.format_id.as_str() != GPT_OSS_MXFP4_SOURCE_WEIGHT_FORMAT_ID
        || source_schema.layout_id.as_str() != GPT_OSS_MXFP4_SOURCE_WEIGHT_LAYOUT_ID
        || source_schema.version != SOURCE_SCHEMA_VERSION
    {
        return Err(invalid_plan(
            "GPT-OSS MXFP4 materializer requires the locked GPT-OSS source weight format, layout, and version",
        ));
    }

    let components = source_schema
        .components
        .iter()
        .map(|component| (&component.id, component))
        .collect::<BTreeMap<_, _>>();
    let mut candidates = Vec::new();
    let mut selected = BTreeSet::new();
    for tensor in &source_schema.tensors {
        let PhysicalWeightLayout::Quantized {
            packed_values,
            packed_dimensions,
            scales,
            zero_points,
            zero_point_packed_dimensions,
            axis_indices,
            permutation,
            codebook,
            group_axis,
            group_padding,
        } = &tensor.physical_layout
        else {
            continue;
        };
        let source_blocks = components.get(&packed_values.component_id).ok_or_else(|| {
            invalid_plan(format!(
                "GPT-OSS MXFP4 tensor `{}` references absent blocks component `{}`",
                tensor.id, packed_values.component_id
            ))
        })?;
        let source_scales = components.get(&scales.component_id).ok_or_else(|| {
            invalid_plan(format!(
                "GPT-OSS MXFP4 tensor `{}` references absent scales component `{}`",
                tensor.id, scales.component_id
            ))
        })?;

        if !is_source_mxfp4_encoding(&source_blocks.encoding) {
            return Err(invalid_plan(format!(
                "quantized GPT-OSS tensor `{}` does not use the locked native MXFP4 source encoding",
                tensor.id
            )));
        }
        if packed_dimensions != &source_blocks.dimensions
            || !is_exact_contiguous(&packed_values.storage)
            || !is_exact_contiguous(&scales.storage)
            || zero_points.is_some()
            || zero_point_packed_dimensions.is_some()
            || axis_indices.is_some()
            || permutation.is_some()
            || codebook.is_some()
            || *group_axis != 2
            || group_padding != &PhysicalWeightPadding::Exact
            || tensor.logical_element_type != ElementType::Bf16
        {
            return Err(invalid_plan(format!(
                "GPT-OSS MXFP4 tensor `{}` has unsupported physical layout or logical type",
                tensor.id
            )));
        }
        if !selected.insert(source_blocks.id.clone()) || !selected.insert(source_scales.id.clone())
        {
            return Err(invalid_plan(format!(
                "GPT-OSS MXFP4 source components for `{}` are shared or repeated",
                tensor.id
            )));
        }
        let mut candidate = candidate_from_source_pair(source_blocks, source_scales)?;
        if candidate.logical_dimensions != tensor.dimensions
            || tensor.required != (source_blocks.required && source_scales.required)
        {
            return Err(invalid_plan(format!(
                "GPT-OSS MXFP4 tensor `{}` differs from its source component shape or requiredness",
                tensor.id
            )));
        }
        candidate.weight_id = tensor.id.clone();
        candidates.push(candidate);
    }

    let source_mxfp4_components = source_schema
        .components
        .iter()
        .filter(|component| is_source_mxfp4_encoding(&component.encoding))
        .count();
    if candidates.is_empty()
        || source_mxfp4_components != candidates.len()
        || selected.len() != candidates.len() * 2
    {
        return Err(invalid_plan(
            "GPT-OSS MXFP4 source schema contains no complete one-to-one blocks/scale pairs",
        ));
    }
    Ok(candidates)
}

fn candidate_from_source_pair(
    source_blocks: &WeightComponentSpec,
    source_scales: &WeightComponentSpec,
) -> Result<Mxfp4MarlinCandidate, VNextError> {
    let [experts, rows, groups, packed_bytes] = source_blocks.dimensions.as_slice() else {
        return Err(invalid_plan(format!(
            "GPT-OSS MXFP4 blocks component `{}` must have [E,N,K/32,16] shape",
            source_blocks.id
        )));
    };
    if *experts == 0
        || *rows == 0
        || *groups < 2
        || !groups.is_multiple_of(2)
        || *packed_bytes != MXFP4_PACKED_BYTES_PER_GROUP as u64
        || !rows.is_multiple_of(MARLIN_OUTPUT_TILE as u64)
        || !is_source_mxfp4_encoding(&source_blocks.encoding)
        || source_blocks.role != WeightComponentRole::PackedValues
        || source_blocks.external_names.len() != 1
    {
        return Err(invalid_plan(format!(
            "GPT-OSS MXFP4 blocks component `{}` violates group-32, low-nibble-first, or Marlin shape requirements",
            source_blocks.id
        )));
    }
    if source_scales.role != WeightComponentRole::Scales
        || source_scales.dimensions != [*experts, *rows, *groups]
        || source_scales.encoding
            != (WeightEncoding::Dense {
                element_type: ElementType::U8,
            })
        || source_scales.external_names.len() != 1
        || source_blocks.required != source_scales.required
    {
        return Err(invalid_plan(format!(
            "GPT-OSS MXFP4 scales component `{}` does not match blocks component `{}`",
            source_scales.id, source_blocks.id
        )));
    }
    let columns = groups
        .checked_mul(MXFP4_GROUP_SIZE as u64)
        .ok_or_else(|| invalid_plan("GPT-OSS MXFP4 input width overflows u64"))?;
    let logical_dimensions = vec![*experts, *rows, columns];
    let packed_id = derived_component_id(source_blocks, source_scales, DerivedKind::Packed)?;
    let scales_id = derived_component_id(source_blocks, source_scales, DerivedKind::Scales)?;
    let required = source_blocks.required && source_scales.required;
    let quantization = execution_quantization_spec()?;
    quantization.validate()?;
    Ok(Mxfp4MarlinCandidate {
        weight_id: WeightId::new("weight.pending.gpt-oss-mxfp4")?,
        source_blocks_id: source_blocks.id.clone(),
        source_scales_id: source_scales.id.clone(),
        logical_dimensions,
        packed_component: WeightComponentSpec {
            id: packed_id,
            role: WeightComponentRole::PackedValues,
            external_names: vec![derived_external_name(
                source_blocks,
                source_scales,
                DerivedKind::Packed,
            )],
            dimensions: vec![*experts, *rows, columns / 2],
            encoding: WeightEncoding::Quantized(quantization),
            required,
        },
        scales_component: WeightComponentSpec {
            id: scales_id,
            role: WeightComponentRole::Scales,
            external_names: vec![derived_external_name(
                source_blocks,
                source_scales,
                DerivedKind::Scales,
            )],
            dimensions: vec![*experts, *rows, *groups],
            encoding: WeightEncoding::Dense {
                element_type: ElementType::U8,
            },
            required,
        },
    })
}

fn execution_quantization_spec() -> Result<QuantizationSpec, VNextError> {
    Ok(QuantizationSpec {
        format_id: QuantizationFormatId::new(GPT_OSS_MXFP4_MARLIN_QUANTIZATION_FORMAT_ID)?,
        bits_per_weight: 4,
        grouping: QuantizationGrouping::fixed(MXFP4_GROUP_SIZE as u32),
        packing: QuantizationPacking::Tiled,
        scale_type: ElementType::U8,
        zero_point_type: None,
    })
}

fn is_source_mxfp4_encoding(encoding: &WeightEncoding) -> bool {
    matches!(
        encoding,
        WeightEncoding::Quantized(spec)
            if spec.format_id.as_str() == GPT_OSS_MXFP4_SOURCE_QUANTIZATION_FORMAT_ID
                && spec.bits_per_weight == 4
                && spec.grouping.fixed_size() == Some(MXFP4_GROUP_SIZE as u32)
                && spec.packing == QuantizationPacking::Interleaved
                && spec.scale_type == ElementType::U8
                && spec.zero_point_type.is_none()
    )
}

fn is_exact_contiguous(storage: &PhysicalStorageLayout) -> bool {
    matches!(
        storage,
        PhysicalStorageLayout::Contiguous {
            padding: PhysicalWeightPadding::Exact
        }
    )
}

#[derive(Debug, Clone, Copy)]
enum DerivedKind {
    Packed,
    Scales,
}

impl DerivedKind {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Packed => "packed",
            Self::Scales => "scales",
        }
    }
}

fn derived_component_id(
    source_blocks: &WeightComponentSpec,
    source_scales: &WeightComponentSpec,
    kind: DerivedKind,
) -> Result<WeightId, VNextError> {
    WeightId::new(format!(
        "{DERIVED_COMPONENT_PREFIX}.{}.{}",
        source_pair_digest(source_blocks, source_scales),
        kind.as_str()
    ))
}

fn derived_external_name(
    source_blocks: &WeightComponentSpec,
    source_scales: &WeightComponentSpec,
    kind: DerivedKind,
) -> String {
    format!(
        "execution.gpt-oss-mxfp4-marlin.{}.{}",
        source_pair_digest(source_blocks, source_scales),
        kind.as_str()
    )
}

fn source_pair_digest(
    source_blocks: &WeightComponentSpec,
    source_scales: &WeightComponentSpec,
) -> String {
    let mut hash = Sha256::new();
    for source_id in [&source_blocks.id, &source_scales.id] {
        hash.update((source_id.as_str().len() as u64).to_le_bytes());
        hash.update(source_id.as_str().as_bytes());
    }
    format!("{:x}", hash.finalize())
}

/// View one raw `[N,K/32,16]` expert as little-endian I32 `[N,K/8]`
/// without decoding nibbles, then emit its contiguous transpose `[K/8,N]`.
#[cfg(test)]
pub(crate) fn transpose_mxfp4_expert_blocks_to_gptq_words(
    raw_blocks: &[u8],
    rows: usize,
    columns: usize,
) -> Result<Vec<i32>, VNextError> {
    if rows == 0 || columns == 0 || !columns.is_multiple_of(MXFP4_GROUP_SIZE) {
        return Err(invalid_plan(
            "MXFP4 expert block transpose requires non-zero N and group-32 K",
        ));
    }
    let packed_words_per_row = columns / 8;
    let expected_bytes = rows
        .checked_mul(columns / 2)
        .ok_or_else(|| invalid_plan("MXFP4 expert block byte count overflows usize"))?;
    if raw_blocks.len() != expected_bytes {
        return Err(invalid_plan(format!(
            "MXFP4 expert blocks contain {} bytes, expected {expected_bytes}",
            raw_blocks.len()
        )));
    }
    let word_count = rows
        .checked_mul(packed_words_per_row)
        .ok_or_else(|| invalid_plan("MXFP4 GPTQ word count overflows usize"))?;
    let mut transposed = Vec::new();
    transposed
        .try_reserve_exact(word_count)
        .map_err(|_| invalid_plan("could not reserve one MXFP4 expert GPTQ staging buffer"))?;
    transposed.resize(word_count, 0_i32);
    for packed_column in 0..packed_words_per_row {
        for row in 0..rows {
            let source = row
                .checked_mul(columns / 2)
                .and_then(|offset| offset.checked_add(packed_column * 4))
                .ok_or_else(|| invalid_plan("MXFP4 block transpose offset overflows usize"))?;
            let bytes: [u8; 4] = raw_blocks[source..source + 4]
                .try_into()
                .expect("validated MXFP4 source range has four bytes");
            transposed[packed_column * rows + row] = i32::from_le_bytes(bytes);
        }
    }
    Ok(transposed)
}

/// Transform one expert's E8M0 `[N,K/32]` bytes to the exact Marlin-MXFP4
/// scale order. Values remain opaque bytes; `f16::from_bits` is used only as a
/// byte-preserving carrier through the shared P64 permutation implementation.
#[cfg(test)]
pub(crate) fn prepare_mxfp4_expert_scales_for_marlin(
    source_scales: &[u8],
    rows: usize,
    columns: usize,
) -> Result<Vec<u8>, VNextError> {
    if rows == 0
        || !rows.is_multiple_of(MARLIN_OUTPUT_TILE)
        || columns == 0
        || !columns.is_multiple_of(MXFP4_GROUP_SIZE * 2)
    {
        return Err(invalid_plan(
            "MXFP4 Marlin scales require N divisible by 64 and K divisible by 64",
        ));
    }
    let group_rows = columns / MXFP4_GROUP_SIZE;
    let expected = rows
        .checked_mul(group_rows)
        .ok_or_else(|| invalid_plan("MXFP4 expert scale count overflows usize"))?;
    if source_scales.len() != expected {
        return Err(invalid_plan(format!(
            "MXFP4 expert scales contain {} bytes, expected {expected}",
            source_scales.len()
        )));
    }

    let mut transposed = Vec::new();
    transposed
        .try_reserve_exact(expected)
        .map_err(|_| invalid_plan("could not reserve one MXFP4 expert scale staging buffer"))?;
    for group in 0..group_rows {
        for row in 0..rows {
            transposed.push(half::f16::from_bits(u16::from(
                source_scales[row * group_rows + group],
            )));
        }
    }
    let p64 = repack_scales_to_marlin(&transposed, columns, rows, MXFP4_GROUP_SIZE);
    let bytes = p64
        .into_iter()
        .map(|value| value.to_bits() as u8)
        .collect::<Vec<_>>();
    vllm_mxfp4_scale_byte_permutation(&bytes, group_rows, rows)
}

#[cfg(test)]
fn vllm_mxfp4_scale_byte_permutation(
    p64_scales: &[u8],
    rows: usize,
    columns: usize,
) -> Result<Vec<u8>, VNextError> {
    if rows == 0
        || !rows.is_multiple_of(2)
        || columns == 0
        || !columns.is_multiple_of(8)
        || p64_scales.len() != rows.saturating_mul(columns)
    {
        return Err(invalid_plan(
            "vLLM MXFP4 byte permutation requires an exact even-row matrix with columns divisible by eight",
        ));
    }
    let mut first = vec![0_u8; p64_scales.len()];
    let column_blocks = columns / 8;
    for row_pair in 0..rows / 2 {
        for column_block in 0..column_blocks {
            for pair_row in 0..2 {
                let source = (row_pair * 2 + pair_row) * columns + column_block * 8;
                let destination = ((row_pair * column_blocks + column_block) * 2 + pair_row) * 8;
                first[destination..destination + 8]
                    .copy_from_slice(&p64_scales[source..source + 8]);
            }
        }
    }
    for chunk in first.chunks_exact_mut(4) {
        chunk.swap(1, 2);
    }
    Ok(first)
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
    use ferrum_interfaces::vnext::{WeightComponentPayload, WeightTensorSpec};

    fn family_id() -> ModelFamilyId {
        ModelFamilyId::new("family.test.gpt-oss-mxfp4").unwrap()
    }

    fn source_quantization() -> QuantizationSpec {
        QuantizationSpec {
            format_id: QuantizationFormatId::new(GPT_OSS_MXFP4_SOURCE_QUANTIZATION_FORMAT_ID)
                .unwrap(),
            bits_per_weight: 4,
            grouping: QuantizationGrouping::fixed(32),
            packing: QuantizationPacking::Interleaved,
            scale_type: ElementType::U8,
            zero_point_type: None,
        }
    }

    fn source_schema() -> WeightSchema {
        let dense_id = WeightId::new("component.test.router").unwrap();
        let blocks_id = WeightId::new("component.test.experts.blocks").unwrap();
        let scales_id = WeightId::new("component.test.experts.scales").unwrap();
        let schema = WeightSchema {
            format_id: WeightFormatId::new(GPT_OSS_MXFP4_SOURCE_WEIGHT_FORMAT_ID).unwrap(),
            layout_id: WeightLayoutId::new(GPT_OSS_MXFP4_SOURCE_WEIGHT_LAYOUT_ID).unwrap(),
            version: SOURCE_SCHEMA_VERSION,
            components: vec![
                WeightComponentSpec {
                    id: dense_id.clone(),
                    role: WeightComponentRole::Values,
                    external_names: vec!["model.layers.0.mlp.router.weight".to_owned()],
                    dimensions: vec![2, 64],
                    encoding: WeightEncoding::Dense {
                        element_type: ElementType::Bf16,
                    },
                    required: true,
                },
                WeightComponentSpec {
                    id: blocks_id.clone(),
                    role: WeightComponentRole::PackedValues,
                    external_names: vec![
                        "model.layers.0.mlp.experts.gate_up_proj_blocks".to_owned()
                    ],
                    dimensions: vec![2, 64, 2, 16],
                    encoding: WeightEncoding::Quantized(source_quantization()),
                    required: true,
                },
                WeightComponentSpec {
                    id: scales_id.clone(),
                    role: WeightComponentRole::Scales,
                    external_names: vec![
                        "model.layers.0.mlp.experts.gate_up_proj_scales".to_owned()
                    ],
                    dimensions: vec![2, 64, 2],
                    encoding: WeightEncoding::Dense {
                        element_type: ElementType::U8,
                    },
                    required: true,
                },
            ],
            tensors: vec![
                WeightTensorSpec {
                    id: WeightId::new("weight.layer.0.router").unwrap(),
                    dimensions: vec![2, 64],
                    logical_element_type: ElementType::Bf16,
                    physical_layout: PhysicalWeightLayout::Dense {
                        component_id: dense_id,
                    },
                    required: true,
                },
                WeightTensorSpec {
                    id: WeightId::new("weight.layer.0.routed_gate_up").unwrap(),
                    dimensions: vec![2, 64, 64],
                    logical_element_type: ElementType::Bf16,
                    physical_layout: PhysicalWeightLayout::Quantized {
                        packed_values: PhysicalWeightComponentBinding::exact_contiguous(blocks_id),
                        packed_dimensions: vec![2, 64, 2, 16],
                        scales: PhysicalWeightComponentBinding::exact_contiguous(scales_id),
                        zero_points: None,
                        zero_point_packed_dimensions: None,
                        axis_indices: None,
                        permutation: None,
                        codebook: None,
                        group_axis: 2,
                        group_padding: PhysicalWeightPadding::Exact,
                    },
                    required: true,
                },
            ],
        };
        schema.validate(&family_id()).unwrap();
        schema
    }

    #[test]
    fn descriptor_is_exact_and_bound_to_the_gpt_oss_mxfp4_provider() {
        let materializer = GptOssMxfp4ToMarlinWeightMaterializer::new().unwrap();
        let descriptor = materializer.descriptor();
        assert_eq!(
            descriptor.id().as_str(),
            GPT_OSS_MXFP4_TO_MARLIN_WEIGHT_MATERIALIZER_ID
        );
        assert_eq!(descriptor.version(), ContractVersion::new(1, 0));
        assert_eq!(descriptor.fidelity(), WeightMaterializationFidelity::Exact);
        assert_eq!(
            descriptor.required_capabilities(),
            &BTreeSet::from([CapabilityId::new(
                GPT_OSS_ROUTED_CLAMPED_SWIGLU_MOE_MXFP4_BF16_CAPABILITY_ID
            )
            .unwrap()])
        );
        assert!(descriptor.approximate_quality_contract().is_none());
        assert_eq!(descriptor.implementation_fingerprint().len(), 64);
    }

    #[test]
    fn execution_schema_rewrites_only_native_mxfp4_pairs() {
        let source = source_schema();
        let execution = derive_execution_schema(&source, &family_id()).unwrap();
        execution.validate(&family_id()).unwrap();

        assert_eq!(
            execution.format_id.as_str(),
            GPT_OSS_MXFP4_MARLIN_WEIGHT_FORMAT_ID
        );
        assert_eq!(
            execution.layout_id.as_str(),
            GPT_OSS_MXFP4_MARLIN_WEIGHT_LAYOUT_ID
        );
        assert!(execution
            .components
            .iter()
            .any(|component| component.id.as_str() == "component.test.router"));
        assert!(!execution.components.iter().any(|component| {
            matches!(
                component.id.as_str(),
                "component.test.experts.blocks" | "component.test.experts.scales"
            )
        }));

        let packed = execution
            .components
            .iter()
            .find(|component| component.role == WeightComponentRole::PackedValues)
            .unwrap();
        let scales = execution
            .components
            .iter()
            .find(|component| component.role == WeightComponentRole::Scales)
            .unwrap();
        assert_eq!(packed.dimensions, [2, 64, 32]);
        assert_eq!(scales.dimensions, [2, 64, 2]);
        let WeightEncoding::Quantized(quantization) = &packed.encoding else {
            panic!("Marlin packed component must retain typed quantization")
        };
        assert_eq!(
            quantization.format_id.as_str(),
            GPT_OSS_MXFP4_MARLIN_QUANTIZATION_FORMAT_ID
        );
        assert_eq!(quantization.bits_per_weight, 4);
        assert_eq!(quantization.grouping, QuantizationGrouping::fixed(32));
        assert_eq!(quantization.packing, QuantizationPacking::Tiled);
        assert_eq!(quantization.scale_type, ElementType::U8);
        assert!(quantization.zero_point_type.is_none());

        let candidate = collect_candidates(&source).unwrap().pop().unwrap();
        let transform = candidate.transform_descriptor();
        assert_eq!(transform.logical_dimensions, [2, 64, 64]);
        assert_eq!(
            transform.source_blocks_id.as_str(),
            "component.test.experts.blocks"
        );
        assert_eq!(
            transform.source_scales_id.as_str(),
            "component.test.experts.scales"
        );
        assert_eq!(transform.packed_values_id, packed.id);
        assert_eq!(transform.scales_id, scales.id);
        let plan = StaticWeightTransformPlan::GptOssMxfp4ToMarlin {
            source_blocks_id: transform.source_blocks_id,
            source_scales_id: transform.source_scales_id,
            packed_values_id: transform.packed_values_id,
            scales_id: transform.scales_id,
            logical_dimensions: transform.logical_dimensions,
        };
        assert_eq!(plan.logical_dimensions(), [2, 64, 64]);
        assert_eq!(plan.matrices_per_output(), 1);
        assert_eq!(plan.scratch_bytes().unwrap(), 64 * 64 / 2);
    }

    #[test]
    fn expert_block_staging_is_a_byte_exact_i32_transpose() {
        let raw = (0_u8..32).collect::<Vec<_>>();
        let words = transpose_mxfp4_expert_blocks_to_gptq_words(&raw, 2, 32).unwrap();
        let actual = words
            .into_iter()
            .flat_map(i32::to_le_bytes)
            .collect::<Vec<_>>();
        assert_eq!(
            actual,
            vec![
                0, 1, 2, 3, 16, 17, 18, 19, 4, 5, 6, 7, 20, 21, 22, 23, 8, 9, 10, 11, 24, 25, 26,
                27, 12, 13, 14, 15, 28, 29, 30, 31,
            ]
        );
        assert!(transpose_mxfp4_expert_blocks_to_gptq_words(&raw[..31], 2, 32).is_err());
    }

    #[test]
    fn scale_staging_applies_transpose_p64_and_both_vllm_byte_permutations() {
        let rows = 64;
        let columns = 64;
        let group_rows = columns / 32;
        let source = (0..rows * group_rows)
            .map(|index| index as u8)
            .collect::<Vec<_>>();

        let actual = prepare_mxfp4_expert_scales_for_marlin(&source, rows, columns).unwrap();

        let transposed = (0..group_rows)
            .flat_map(|group| {
                let source = &source;
                (0..rows).map(move |row| source[row * group_rows + group])
            })
            .collect::<Vec<_>>();
        let p64_permutation = (0..8)
            .flat_map(|row| (0..8).map(move |column| row + 8 * column))
            .collect::<Vec<_>>();
        let mut p64 = vec![0_u8; transposed.len()];
        for chunk_start in (0..transposed.len()).step_by(64) {
            for (destination, source_offset) in p64_permutation.iter().copied().enumerate() {
                p64[chunk_start + destination] = transposed[chunk_start + source_offset];
            }
        }
        let mut expected = vec![0_u8; p64.len()];
        for column_block in 0..rows / 8 {
            for pair_row in 0..2 {
                let source_offset = pair_row * rows + column_block * 8;
                let destination = (column_block * 2 + pair_row) * 8;
                expected[destination..destination + 8]
                    .copy_from_slice(&p64[source_offset..source_offset + 8]);
            }
        }
        for chunk in expected.chunks_exact_mut(4) {
            chunk.swap(1, 2);
        }
        assert_eq!(actual, expected);
        let mut actual_multiset = actual.clone();
        let mut source_multiset = source.clone();
        actual_multiset.sort_unstable();
        source_multiset.sort_unstable();
        assert_eq!(actual_multiset, source_multiset);
    }

    #[test]
    fn direct_gpu_scale_index_formula_matches_the_rust_marlin_oracle() {
        fn direct_source_index(destination: usize, rows: usize, columns: usize) -> usize {
            let groups = columns / 32;
            let chunk_base = destination & !3;
            let first = chunk_base
                + match destination & 3 {
                    1 => 2,
                    2 => 1,
                    lane => lane,
                };
            let lane8 = first & 7;
            let mut packed = first >> 3;
            let pair_row = packed & 1;
            packed >>= 1;
            let column_blocks = rows / 8;
            let column_block = packed % column_blocks;
            let row_pair = packed / column_blocks;
            let p64 = (row_pair * 2 + pair_row) * rows + column_block * 8 + lane8;
            let p64_chunk = p64 & !63;
            let p64_lane = p64 & 63;
            let transposed = p64_chunk + (p64_lane >> 3) + 8 * (p64_lane & 7);
            let group = transposed / rows;
            let row = transposed % rows;
            row * groups + group
        }

        for (rows, columns) in [(64, 64), (64, 128), (128, 64), (128, 256)] {
            let source = (0..rows * (columns / 32))
                .map(|index| ((index * 37 + 11) % 251) as u8)
                .collect::<Vec<_>>();
            let oracle = prepare_mxfp4_expert_scales_for_marlin(&source, rows, columns).unwrap();
            let direct = (0..oracle.len())
                .map(|destination| source[direct_source_index(destination, rows, columns)])
                .collect::<Vec<_>>();
            assert_eq!(direct, oracle, "N={rows}, K={columns}");
        }
    }

    #[test]
    fn source_recipe_and_shapes_fail_closed() {
        let mut wrong_format = source_schema();
        wrong_format.format_id = WeightFormatId::new("weight-format.test.other").unwrap();
        let error = derive_execution_schema(&wrong_format, &family_id()).unwrap_err();
        assert!(
            error.to_string().contains("locked GPT-OSS source"),
            "{error}"
        );

        let mut wrong_recipe = source_schema();
        let blocks = wrong_recipe
            .components
            .iter_mut()
            .find(|component| component.role == WeightComponentRole::PackedValues)
            .unwrap();
        let WeightEncoding::Quantized(quantization) = &mut blocks.encoding else {
            unreachable!()
        };
        quantization.packing = QuantizationPacking::Linear;
        let error = derive_execution_schema(&wrong_recipe, &family_id()).unwrap_err();
        assert!(error.to_string().contains("locked native MXFP4"), "{error}");

        let mut wrong_rows = source_schema();
        let blocks = wrong_rows
            .components
            .iter_mut()
            .find(|component| component.role == WeightComponentRole::PackedValues)
            .unwrap();
        blocks.dimensions[1] = 32;
        let scales = wrong_rows
            .components
            .iter_mut()
            .find(|component| component.role == WeightComponentRole::Scales)
            .unwrap();
        scales.dimensions[1] = 32;
        wrong_rows.tensors[1].dimensions[1] = 32;
        let PhysicalWeightLayout::Quantized {
            packed_dimensions, ..
        } = &mut wrong_rows.tensors[1].physical_layout
        else {
            unreachable!()
        };
        packed_dimensions[1] = 32;
        let error = derive_execution_schema(&wrong_rows, &family_id()).unwrap_err();
        assert!(error.to_string().contains("Marlin shape"), "{error}");
    }

    struct PanicSource;

    impl WeightComponentSource for PanicSource {
        fn component<'source>(
            &'source self,
            _component: &WeightComponentSpec,
        ) -> Result<WeightComponentPayload<'source>, VNextError> {
            panic!("derived host materialization must fail before reading source bytes")
        }
    }

    #[test]
    fn derived_host_materialization_is_forbidden_for_the_typed_device_transform() {
        let source = source_schema();
        let blocks = source
            .components
            .iter()
            .find(|component| component.role == WeightComponentRole::PackedValues)
            .unwrap();
        let scales = source
            .components
            .iter()
            .find(|component| component.role == WeightComponentRole::Scales)
            .unwrap();
        let candidate = candidate_from_source_pair(blocks, scales).unwrap();
        let materializer = GptOssMxfp4ToMarlinWeightMaterializer::new().unwrap();
        let error = match materializer.materialize_components(
            &PanicSource,
            &[blocks, scales],
            &[&candidate.packed_component, &candidate.scales_component],
        ) {
            Ok(_) => panic!("derived host materialization must fail closed"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("per-expert"), "{error}");
        assert!(
            error
                .to_string()
                .contains("host materialization is forbidden"),
            "{error}"
        );
    }
}
