use super::*;
use crate::gguf_blocks::GgufBlockFormat;
use ferrum_interfaces::vnext::{
    AliasPolicy, BlockQuantizationSpec, BufferUsage, CompositeWeightPart, ContractVersion,
    PhysicalStorageLayout, PhysicalWeightComponentBinding, PhysicalWeightLayout,
    PhysicalWeightPadding, ProgramValueId, QuantizationFormatId, ResolvedStorageComponent,
    ResolvedTensorSpec, ResolvedValueStorage, ResolvedWeightBinding, ResourceId, TensorAccess,
    WeightComponentRole, WeightEncoding, WeightId,
};
use serde_json::json;

fn projection_binding(
    shape: AttentionShape,
    element_type: ElementType,
    storage: impl Fn(u64, u64) -> PhysicalStorageLayout,
) -> Result<ResolvedValueBinding, VNextError> {
    let formats = [
        GgufBlockFormat::Q5K,
        GgufBlockFormat::Q4K,
        GgufBlockFormat::Q8_0,
        GgufBlockFormat::Q8_0,
    ];
    let rows = [
        shape.qkv_features,
        shape.value_features,
        shape.value_heads,
        shape.value_heads,
    ];
    let mut parts = Vec::new();
    let mut components = Vec::new();
    let mut offset = 0;
    for (index, (format, rows)) in formats.into_iter().zip(rows).enumerate() {
        let component = WeightId::new(format!("gdn-part-{index}"))?;
        let spec = BlockQuantizationSpec {
            format_id: QuantizationFormatId::new(format.format_id())?,
            logical_values_per_block: format.block_values() as u32,
            bytes_per_block: format.block_bytes() as u32,
        };
        let blocks = shape.hidden_size / u64::from(spec.logical_values_per_block);
        parts.push(CompositeWeightPart {
            logical_offsets: vec![offset, 0],
            extents: vec![rows, shape.hidden_size],
            layout: Box::new(PhysicalWeightLayout::BlockQuantized {
                blocks: PhysicalWeightComponentBinding {
                    component_id: component.clone(),
                    storage: storage(rows, blocks),
                },
                block_axis: 1,
                block_padding: PhysicalWeightPadding::Exact,
            }),
        });
        components.push(json!({
            "component_id": component, "role": WeightComponentRole::PackedValues,
            "physical_dimensions": [rows, blocks],
            "encoding": WeightEncoding::BlockQuantized(spec),
        }));
        offset += rows;
    }
    let weight: ResolvedWeightBinding = serde_json::from_value(json!({
        "weight_id": "gdn-projections", "format_id": "weight-format.gguf.native-block",
        "layout_id": "layout.gguf.native-block", "schema_version": ContractVersion::new(1, 0),
        "physical_layout": PhysicalWeightLayout::Composite {parts}, "components": components,
    }))
    .unwrap();
    let tensor = ResolvedTensorSpec::new(
        vec![offset, shape.hidden_size],
        element_type,
        ResolvedTensorLayout::Contiguous,
    )?;
    let storage = ResolvedValueStorage::composite(
        weight
            .components()
            .iter()
            .map(|component| {
                ResolvedStorageComponent::new(
                    Some(component.component_id().clone()),
                    ResourceId::new(format!("resource.{}", component.component_id()))?,
                    0,
                    component.physical_bytes()?,
                    component.physical_element_type(),
                )
            })
            .collect::<Result<Vec<_>, VNextError>>()?,
    )?;
    ResolvedValueBinding::new(
        ProgramValueId::new("gdn-projections")?,
        ResolvedValueRole::Input,
        2,
        tensor,
        TensorAccess::Read,
        AliasPolicy::NoAlias,
        BufferUsage::Weights,
        Some(weight),
        storage,
    )
}

#[test]
fn staged_gated_delta_resolved_binding_matches_resource_layout() {
    let mut shape = qwen35_attention_shape();
    shape.hidden_size = 4096;
    for hidden in [4096, 8192] {
        shape.hidden_size = hidden;
        for strided in [false, true] {
            let value = projection_binding(shape, ElementType::F16, |_, columns| {
                if strided {
                    PhysicalStorageLayout::Strided {
                        strides_in_elements: vec![columns, 1],
                        padding: PhysicalWeightPadding::Exact,
                    }
                } else {
                    PhysicalStorageLayout::exact_contiguous()
                }
            })
            .unwrap();
            // Goes through the actual GDN input-2 resolver, not just leaf metadata.
            let bytes = input_projection_workspace(&[value], shape).unwrap();
            assert_eq!(bytes, shape.qkv_features * hidden * 2);
            for tokens in [3, 768, 3072] {
                let base = ScratchLayout::new(shape, tokens).unwrap();
                let staged = base.with_projection_workspace(bytes).unwrap();
                assert_eq!(staged.projection_workspace, base.required_bytes);
                assert_eq!(
                    staged.required_bytes,
                    shape.fixed_scratch_bytes().unwrap()
                        + bytes
                        + tokens * shape.scratch_bytes_per_token().unwrap()
                );
                assert_eq!(staged.normalized, base.normalized);
                assert_eq!(staged.core, base.core);
            }
        }
    }
    // The 64 MiB value comes from the real leaf shape, not a generic byte cap.
    shape.hidden_size = 4096;
    let value = projection_binding(shape, ElementType::F16, |_, _| {
        PhysicalStorageLayout::exact_contiguous()
    })
    .unwrap();
    assert_eq!(
        input_projection_workspace(&[value], shape).unwrap(),
        64 * 1024 * 1024
    );
    let f32 = projection_binding(shape, ElementType::F32, |_, _| {
        PhysicalStorageLayout::exact_contiguous()
    })
    .unwrap();
    assert_eq!(input_projection_workspace(&[f32], shape).unwrap(), 0);
}

#[test]
fn staged_gated_delta_rejects_non_row_major_physical_bindings() {
    let mut shape = qwen35_attention_shape();
    shape.hidden_size = 4096;
    for layout in 0..4 {
        let binding = projection_binding(shape, ElementType::F16, |rows, columns| match layout {
            0 => PhysicalStorageLayout::Strided {
                strides_in_elements: vec![1, rows],
                padding: PhysicalWeightPadding::Exact,
            },
            1 => PhysicalStorageLayout::Tiled {
                tile_shape: vec![8, columns],
                axis_order: vec![0, 1],
                tile_strides_in_elements: vec![8 * columns, 8 * columns],
                padding: PhysicalWeightPadding::Exact,
            },
            2 => PhysicalStorageLayout::Strided {
                strides_in_elements: vec![columns + 1, 1],
                padding: PhysicalWeightPadding::Exact,
            },
            _ => PhysicalStorageLayout::Strided {
                strides_in_elements: vec![columns, 1],
                padding: PhysicalWeightPadding::ZeroFill {
                    padded_dimensions: vec![rows + 1, columns],
                },
            },
        });
        if layout < 2 {
            assert_eq!(
                input_projection_workspace(&[binding.unwrap()], shape).unwrap(),
                0
            );
        } else {
            assert!(
                binding.is_err(),
                "holes/padding exceed the physical component span"
            );
        }
    }
    let mixed = projection_binding(shape, ElementType::F16, |rows, columns| {
        if rows == shape.value_features {
            PhysicalStorageLayout::Strided {
                strides_in_elements: vec![1, rows],
                padding: PhysicalWeightPadding::Exact,
            }
        } else {
            PhysicalStorageLayout::Strided {
                strides_in_elements: vec![columns, 1],
                padding: PhysicalWeightPadding::Exact,
            }
        }
    })
    .unwrap();
    assert_eq!(
        input_projection_workspace(&[mixed], shape).unwrap(),
        0,
        "a valid QKV leaf must not enable staging for an unproven Z layout"
    );
}
