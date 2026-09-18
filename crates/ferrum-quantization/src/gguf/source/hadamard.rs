//! Explicit metadata-component registry and complete source/layout binding.

use std::collections::{BTreeMap, BTreeSet};
use std::num::NonZeroU32;

use ferrum_interfaces::vnext::{
    ElementType, GroupedFeatureTranspose, HadamardApplication, HadamardSigns,
    HadamardTransformSpec, ModelFamilyId, PhysicalWeightComponentBinding, PhysicalWeightLayout,
    VNextError, WeightComponentRole, WeightComponentSpec, WeightEncoding, WeightId, WeightSchema,
};

use crate::gguf::{GgufHadamard, GgufHadamardDirection, GgufHadamardSigns, NativeGgufFile};

pub fn hadamard_sign_component(width: u64) -> Result<WeightComponentSpec, VNextError> {
    if width == 0 {
        return Err(invalid("sign width must be nonzero"));
    }
    Ok(WeightComponentSpec {
        id: WeightId::new(format!("gguf.hadamard.signs.width-{width}"))?,
        role: WeightComponentRole::TransformSigns,
        external_names: vec![format!("gguf.metadata.prism.hadamard.signs.{width}")],
        dimensions: vec![width],
        encoding: WeightEncoding::Dense {
            element_type: ElementType::F32,
        },
        required: true,
    })
}

pub fn hadamard_transform_spec(
    metadata: &GgufHadamard,
    name: &str,
) -> Result<Option<HadamardTransformSpec>, VNextError> {
    let Some(weight) = metadata.weight(name) else {
        return Ok(None);
    };
    let signs = match metadata.signs() {
        GgufHadamardSigns::Identity => HadamardSigns::Identity,
        GgufHadamardSigns::Explicit(_) => {
            HadamardSigns::Explicit(PhysicalWeightComponentBinding::exact_contiguous(
                hadamard_sign_component(weight.input_width())?.id,
            ))
        }
    };
    let application = match weight.direction() {
        GgufHadamardDirection::BeforeMatmul => HadamardApplication::BeforeMatmul {
            input_permutation: weight.gdn_permutation().map(|permutation| {
                GroupedFeatureTranspose {
                    inner_extent: permutation.head_dim,
                    first_outer_extent: permutation.key_heads,
                    second_outer_extent: permutation.repeats,
                }
            }),
        },
        GgufHadamardDirection::AfterEmbeddingLookup => HadamardApplication::AfterEmbeddingLookup,
    };
    Ok(Some(HadamardTransformSpec {
        block_size: NonZeroU32::new(metadata.block_size())
            .ok_or_else(|| invalid("zero block size"))?,
        signs,
        application,
    }))
}

#[derive(Default)]
pub(super) struct HadamardRegistry {
    pub(super) components: BTreeMap<WeightId, (WeightComponentSpec, Vec<u8>)>,
}

impl HadamardRegistry {
    pub(super) fn bind(file: &NativeGgufFile, schema: &WeightSchema) -> Result<Self, VNextError> {
        schema.validate(&ModelFamilyId::new("source.gguf.hadamard")?)?;
        let components: BTreeMap<_, _> = schema
            .components
            .iter()
            .map(|component| (component.id.clone(), component))
            .collect();
        let mut consumed = BTreeSet::new();
        for tensor in &schema.tensors {
            bind_layout(
                file,
                &components,
                &tensor.physical_layout,
                None,
                &mut consumed,
            )?;
        }
        let expected: BTreeSet<_> = file
            .hadamard()
            .map(|metadata| metadata.weights().keys().map(String::as_str).collect())
            .unwrap_or_default();
        if consumed != expected {
            return Err(invalid(
                "weight schema does not consume every declared source transform",
            ));
        }
        let mut registry = Self::default();
        if let Some(metadata) = file.hadamard() {
            if let GgufHadamardSigns::Explicit(signs) = metadata.signs() {
                for weight in metadata.weights().values() {
                    let width = weight.input_width();
                    let spec = hadamard_sign_component(width)?;
                    if components.get(&spec.id).copied() != Some(&spec) {
                        return Err(invalid(
                            "sign component differs from its validated source declaration",
                        ));
                    }
                    if registry.components.contains_key(&spec.id) {
                        continue;
                    }
                    let values = signs
                        .get(&width)
                        .ok_or_else(|| invalid("missing source signs"))?;
                    let bytes = values
                        .iter()
                        .flat_map(|sign| f32::from(*sign).to_le_bytes())
                        .collect();
                    registry.components.insert(spec.id.clone(), (spec, bytes));
                }
            }
        }
        for component in &schema.components {
            if component.role == WeightComponentRole::TransformSigns
                && !registry.components.contains_key(&component.id)
            {
                return Err(invalid(
                    "sign component has no source metadata registry entry",
                ));
            }
        }
        Ok(registry)
    }
}

fn bind_layout<'a>(
    file: &'a NativeGgufFile,
    components: &BTreeMap<WeightId, &WeightComponentSpec>,
    layout: &PhysicalWeightLayout,
    transform: Option<&HadamardTransformSpec>,
    consumed: &mut BTreeSet<&'a str>,
) -> Result<(), VNextError> {
    match layout {
        PhysicalWeightLayout::Hadamard {
            values,
            transform: declared,
        } => {
            if transform.is_some() {
                return Err(invalid("nested source transforms are unsupported"));
            }
            bind_layout(file, components, values, Some(declared), consumed)
        }
        PhysicalWeightLayout::Composite { parts } if transform.is_none() => {
            for part in parts {
                bind_layout(file, components, &part.layout, None, consumed)?;
            }
            Ok(())
        }
        PhysicalWeightLayout::Dense { component_id } => {
            bind_leaf(file, components, component_id, transform, consumed)
        }
        PhysicalWeightLayout::Stored { component } => bind_leaf(
            file,
            components,
            &component.component_id,
            transform,
            consumed,
        ),
        PhysicalWeightLayout::BlockQuantized { blocks, .. } => {
            bind_leaf(file, components, &blocks.component_id, transform, consumed)
        }
        _ => Err(invalid(
            "GGUF transform binding requires per-projection dense or block leaves",
        )),
    }
}

fn bind_leaf<'a>(
    file: &'a NativeGgufFile,
    components: &BTreeMap<WeightId, &WeightComponentSpec>,
    component_id: &WeightId,
    transform: Option<&HadamardTransformSpec>,
    consumed: &mut BTreeSet<&'a str>,
) -> Result<(), VNextError> {
    let component = components
        .get(component_id)
        .ok_or_else(|| invalid("missing weight component"))?;
    let [name] = component.external_names.as_slice() else {
        return Err(invalid("GGUF leaf must bind one physical tensor"));
    };
    let expected = file
        .hadamard()
        .map(|metadata| hadamard_transform_spec(metadata, name))
        .transpose()?
        .flatten();
    if transform != expected.as_ref() {
        return Err(invalid(format!(
            "source transform for {name:?} is missing or differs from its declared layout"
        )));
    }
    if expected.is_some() {
        let (name, _) = file
            .hadamard()
            .unwrap()
            .weights()
            .get_key_value(name)
            .unwrap();
        if !consumed.insert(name.as_str()) {
            return Err(invalid("source transform is consumed more than once"));
        }
    }
    Ok(())
}

fn invalid(reason: impl std::fmt::Display) -> VNextError {
    VNextError::InvalidExecutionPlan {
        reason: format!("GGUF Hadamard source binding: {reason}"),
    }
}
