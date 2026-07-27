//! Cold-path execution-weight preparation for CUDA Marlin FP8 W8A16.
//!
//! Eligibility comes from stable operation contracts and physical weight
//! shapes. Model names, device names, memory tiers, and request state are
//! deliberately absent from this boundary.

use std::borrow::Cow;
use std::collections::{BTreeMap, BTreeSet};

use ferrum_interfaces::vnext::{
    CapabilityId, ContractVersion, DeviceDescriptor, ElementType, PhysicalWeightComponentBinding,
    PhysicalWeightLayout, PhysicalWeightPadding, PreparedModelFamily, QuantizationFormatId,
    QuantizationGrouping, QuantizationPacking, QuantizationSpec, VNextError,
    WeightComponentPayload, WeightComponentRole, WeightComponentSource, WeightComponentSpec,
    WeightEncoding, WeightFormatId, WeightId, WeightLayoutId, WeightMaterializationFidelity,
    WeightMaterializer, WeightMaterializerDescriptor, WeightMaterializerId, WeightSchema,
    DENSE_LINEAR_OPERATION_ID, GATED_DELTA_RECURRENT_ATTENTION_OPERATION_ID,
};
use sha2::{Digest, Sha256};

use crate::marlin_repack::{
    fp8_marlin_shape_supported, prepare_f16_weight_for_fp8_marlin, Fp8MarlinWeight,
};

pub const MARLIN_FP8_WEIGHT_MATERIALIZER_ID: &str = "weight-materializer.cuda.marlin-fp8-w8a16";
pub const MARLIN_FP8_CAPABILITY_ID: &str = "capability.kernel.cuda.marlin.fp8-w8a16";
pub const MARLIN_FP8_WEIGHT_FORMAT_ID: &str = "weight-format.execution.cuda.marlin-fp8-w8a16-mixed";
pub const MARLIN_FP8_WEIGHT_LAYOUT_ID: &str = "weight-layout.execution.cuda.marlin-fp8-w8a16-mixed";
pub const MARLIN_FP8_QUANTIZATION_FORMAT_ID: &str = "quantization.marlin.fp8-e4m3fn-channelwise";

const MATERIALIZER_VERSION: ContractVersion = ContractVersion::new(2, 0);
const DERIVED_COMPONENT_PREFIX: &str = "component.execution.marlin-fp8";

pub fn marlin_fp8_weight_materializer() -> Result<Box<dyn WeightMaterializer>, VNextError> {
    Ok(Box::new(MarlinFp8WeightMaterializer::new()?))
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
const fn marlin_fp8_projection_shape_supported(n: usize, k: usize) -> bool {
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
}
