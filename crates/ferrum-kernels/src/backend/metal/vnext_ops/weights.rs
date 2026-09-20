//! Typed translation from a resolved logical weight to retained Metal regions.
//!
//! Providers consume this representation instead of guessing a physical ABI
//! from a model name, source file, component ordering, or byte length.

use ferrum_interfaces::vnext::{
    AxisWeightComponent, BlockQuantizationSpec, CompositeWeightPart, ElementType,
    HadamardApplication, HadamardSigns, OperationInvocation, PhysicalWeightLayout,
    PhysicalWeightPadding, ResolvedValueBinding, ResolvedWeightBinding,
    ResolvedWeightComponentLayout, WeightEncoding, WeightId,
};

use super::super::vnext_runtime::{MetalBufferRegion, MetalDeviceBuffer};
use super::hadamard::{GroupedFeatureTranspose, HadamardTransform};

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct MetalResolvedWeightComponent {
    physical_dimensions: Vec<u64>,
    encoding: WeightEncoding,
}

impl MetalResolvedWeightComponent {
    pub(crate) fn physical_dimensions(&self) -> &[u64] {
        &self.physical_dimensions
    }

    pub(crate) fn encoding(&self) -> &WeightEncoding {
        &self.encoding
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct MetalResolvedCompositePart {
    pub(crate) layout: MetalResolvedWeightLayout,
    pub(crate) logical_offsets: Vec<u64>,
    pub(crate) extents: Vec<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct MetalResolvedAxisComponent {
    pub(crate) component: usize,
    pub(crate) axis: u32,
}

/// Provider-visible physical tree. Leaf indices address the sibling regions
/// returned by [`MetalResolvedWeight::into_command_parts`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum MetalResolvedWeightLayout {
    Dense {
        component: usize,
    },
    Stored {
        component: usize,
    },
    Composite {
        parts: Vec<MetalResolvedCompositePart>,
    },
    Quantized {
        packed_values: usize,
        packed_dimensions: Vec<u64>,
        scales: usize,
        zero_points: Option<usize>,
        zero_point_packed_dimensions: Option<Vec<u64>>,
        axis_indices: Option<MetalResolvedAxisComponent>,
        permutation: Option<MetalResolvedAxisComponent>,
        codebook: Option<usize>,
        group_axis: u32,
        group_padding: PhysicalWeightPadding,
    },
    BlockQuantized {
        component: usize,
        spec: BlockQuantizationSpec,
        block_axis: u32,
        block_padding: PhysicalWeightPadding,
    },
    Hadamard {
        values: Box<MetalResolvedWeightLayout>,
        transform: HadamardTransform,
    },
    AxisReshapePermutation {
        values: Box<MetalResolvedWeightLayout>,
        axis: u32,
        logical_offset: u64,
        extent: u64,
        reshape: Vec<u64>,
        stored_axis_order: Vec<u32>,
    },
    Indexed {
        indices: MetalResolvedAxisComponent,
        values: Box<MetalResolvedWeightLayout>,
        source_axis_extent: u64,
    },
    ExpertStack {
        experts: Vec<MetalResolvedWeightLayout>,
        expert_axis: u32,
    },
}

pub(crate) struct MetalResolvedWeight {
    logical_dimensions: Vec<u64>,
    logical_element_type: ElementType,
    components: Vec<MetalResolvedWeightComponent>,
    layout: MetalResolvedWeightLayout,
    regions: Vec<MetalBufferRegion>,
}

impl MetalResolvedWeight {
    pub(crate) fn logical_dimensions(&self) -> &[u64] {
        &self.logical_dimensions
    }

    pub(crate) const fn logical_element_type(&self) -> ElementType {
        self.logical_element_type
    }

    pub(crate) fn components(&self) -> &[MetalResolvedWeightComponent] {
        &self.components
    }

    pub(crate) fn layout(&self) -> &MetalResolvedWeightLayout {
        &self.layout
    }

    pub(crate) fn regions(&self) -> &[MetalBufferRegion] {
        &self.regions
    }

    pub(crate) fn into_command_parts(
        self,
    ) -> (
        Vec<MetalBufferRegion>,
        Vec<MetalResolvedWeightComponent>,
        MetalResolvedWeightLayout,
    ) {
        (self.regions, self.components, self.layout)
    }
}

pub(super) fn validate_hadamard_transform(
    transform: HadamardTransform,
    width: u64,
    components: &[MetalResolvedWeightComponent],
    regions: &[MetalBufferRegion],
) -> Result<(), String> {
    transform.validate(width)?;
    if let Some(index) = transform.signs_region {
        let component = components
            .get(index)
            .ok_or_else(|| "Metal Hadamard signs metadata is absent".to_owned())?;
        let region = regions
            .get(index)
            .ok_or_else(|| "Metal Hadamard signs storage is absent".to_owned())?;
        if component.encoding()
            != &(WeightEncoding::Dense {
                element_type: ElementType::F32,
            })
            || component.physical_dimensions() != [width]
            || region.element_type() != ElementType::F32
            || width.checked_mul(4) != Some(region.length_bytes())
        {
            return Err("Metal Hadamard signs must be a complete F32 feature row".to_owned());
        }
    }
    Ok(())
}

pub(crate) fn resolve_weight(
    participant: &OperationInvocation<'_, MetalDeviceBuffer>,
    binding: &ResolvedValueBinding,
) -> Result<MetalResolvedWeight, String> {
    let weight = binding
        .weight()
        .ok_or_else(|| "Metal weight binding lacks its typed physical layout".to_owned())?;
    let stored_components = binding.storage().components();
    if stored_components.len() != weight.components().len() {
        return Err("Metal weight component identities are duplicated or incomplete".to_owned());
    }

    let mut components = Vec::with_capacity(weight.components().len());
    let mut regions = Vec::with_capacity(weight.components().len());
    // Both contracts require canonical component IDs and reject duplicates. Check
    // their correspondence without rebuilding an index for every invocation.
    for (component, stored) in weight.components().iter().zip(stored_components) {
        let stored_id = stored
            .component_id()
            .ok_or_else(|| "Metal weight component lacks its physical identity".to_owned())?;
        if stored_id != component.component_id() {
            return Err(format!(
                "Metal weight component `{}` has no resolved storage",
                component.component_id()
            ));
        }
        if stored.element_type() != component.physical_element_type()
            || stored.length_bytes()
                != component
                    .physical_bytes()
                    .map_err(|error| error.to_string())?
        {
            return Err(format!(
                "Metal weight component `{}` differs from its typed physical ABI",
                component.component_id()
            ));
        }
        let view = participant
            .views()
            .iter()
            .find(|view| view.resource_id() == stored.resource_id())
            .ok_or_else(|| {
                format!(
                    "Metal weight component `{}` has no committed resource view",
                    component.component_id()
                )
            })?;
        let translated = view
            .translate(stored.offset_bytes(), stored.length_bytes())
            .map_err(|error| error.to_string())?;
        let mut physical_regions = translated.iter();
        let physical = physical_regions.next().ok_or_else(|| {
            format!(
                "Metal weight component `{}` translated to no physical region",
                component.component_id()
            )
        })?;
        if physical_regions.next().is_some() {
            return Err(format!(
                "Metal weight component `{}` is not physically contiguous",
                component.component_id()
            ));
        }
        let (buffer, range, retention) = physical.buffer_and_physical_range();
        let region = buffer
            .retained_region(range, retention)
            .map_err(|error| error.to_string())?;
        if region.element_type() != stored.element_type()
            || region.length_bytes() != stored.length_bytes()
        {
            return Err(format!(
                "Metal weight component `{}` retained the wrong physical range",
                component.component_id()
            ));
        }
        components.push(component_metadata(component));
        regions.push(region);
    }
    let layout = resolve_layout(weight)?;
    Ok(MetalResolvedWeight {
        logical_dimensions: binding.tensor().dimensions().to_vec(),
        logical_element_type: binding.tensor().element_type(),
        components,
        layout,
        regions,
    })
}

fn component_metadata(component: &ResolvedWeightComponentLayout) -> MetalResolvedWeightComponent {
    MetalResolvedWeightComponent {
        physical_dimensions: component.physical_dimensions().to_vec(),
        encoding: component.encoding().clone(),
    }
}

fn resolve_layout(weight: &ResolvedWeightBinding) -> Result<MetalResolvedWeightLayout, String> {
    resolve_layout_node(weight.physical_layout(), weight)
}

fn resolve_layout_node(
    layout: &PhysicalWeightLayout,
    weight: &ResolvedWeightBinding,
) -> Result<MetalResolvedWeightLayout, String> {
    let component_index = |component_id: &WeightId| {
        // ResolvedWeightBinding validates strict ID order; retained regions
        // follow this same component order.
        weight
            .components()
            .binary_search_by(|component| component.component_id().cmp(component_id))
            .map_err(|_| {
                format!("Metal physical layout references absent component `{component_id}`")
            })
    };
    let axis_component = |component: &AxisWeightComponent| -> Result<_, String> {
        Ok(MetalResolvedAxisComponent {
            component: component_index(&component.component.component_id)?,
            axis: component.axis,
        })
    };
    match layout {
        PhysicalWeightLayout::Dense { component_id } => Ok(MetalResolvedWeightLayout::Dense {
            component: component_index(component_id)?,
        }),
        PhysicalWeightLayout::Stored { component } => Ok(MetalResolvedWeightLayout::Stored {
            component: component_index(&component.component_id)?,
        }),
        PhysicalWeightLayout::Composite { parts } => Ok(MetalResolvedWeightLayout::Composite {
            parts: parts
                .iter()
                .map(|part: &CompositeWeightPart| {
                    Ok(MetalResolvedCompositePart {
                        layout: resolve_layout_node(&part.layout, weight)?,
                        logical_offsets: part.logical_offsets.clone(),
                        extents: part.extents.clone(),
                    })
                })
                .collect::<Result<Vec<_>, String>>()?,
        }),
        PhysicalWeightLayout::Quantized {
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
        } => Ok(MetalResolvedWeightLayout::Quantized {
            packed_values: component_index(&packed_values.component_id)?,
            packed_dimensions: packed_dimensions.clone(),
            scales: component_index(&scales.component_id)?,
            zero_points: zero_points
                .as_ref()
                .map(|component| component_index(&component.component_id))
                .transpose()?,
            zero_point_packed_dimensions: zero_point_packed_dimensions.clone(),
            axis_indices: axis_indices.as_ref().map(axis_component).transpose()?,
            permutation: permutation.as_ref().map(axis_component).transpose()?,
            codebook: codebook
                .as_ref()
                .map(|component| component_index(&component.component_id))
                .transpose()?,
            group_axis: *group_axis,
            group_padding: group_padding.clone(),
        }),
        PhysicalWeightLayout::QuantizedBlockGrid { .. } => {
            Err("Metal does not support quantized block-grid physical weight layouts".to_owned())
        }
        PhysicalWeightLayout::BlockQuantized {
            blocks,
            block_axis,
            block_padding,
        } => {
            let component = component_index(&blocks.component_id)?;
            let metadata = weight
                .components()
                .get(component)
                .ok_or_else(|| "Metal block component index is out of range".to_owned())?;
            let WeightEncoding::BlockQuantized(spec) = metadata.encoding() else {
                return Err("Metal block layout component lacks its block ABI".to_owned());
            };
            Ok(MetalResolvedWeightLayout::BlockQuantized {
                component,
                spec: spec.clone(),
                block_axis: *block_axis,
                block_padding: block_padding.clone(),
            })
        }
        PhysicalWeightLayout::Hadamard { values, transform } => {
            let signs_region = match &transform.signs {
                HadamardSigns::Identity => None,
                HadamardSigns::Explicit(signs) => Some(component_index(&signs.component_id)?),
            };
            let (inverse, permutation) = match &transform.application {
                HadamardApplication::AfterEmbeddingLookup => (true, None),
                HadamardApplication::BeforeMatmul { input_permutation } => {
                    let permutation = input_permutation
                        .as_ref()
                        .map(|value| {
                            let extent = |value| {
                                u32::try_from(value).map_err(|_| {
                                    "Metal Hadamard permutation extent exceeds u32".to_owned()
                                })
                            };
                            Ok::<_, String>(GroupedFeatureTranspose {
                                inner_extent: extent(value.inner_extent)?,
                                first_outer_extent: extent(value.first_outer_extent)?,
                                second_outer_extent: extent(value.second_outer_extent)?,
                            })
                        })
                        .transpose()?;
                    (false, permutation)
                }
            };
            Ok(MetalResolvedWeightLayout::Hadamard {
                values: Box::new(resolve_layout_node(values, weight)?),
                transform: HadamardTransform {
                    block_size: transform.block_size.get(),
                    signs_region,
                    inverse,
                    permutation,
                },
            })
        }
        PhysicalWeightLayout::AxisReshapePermutation {
            values,
            axis,
            logical_offset,
            extent,
            reshape,
            stored_axis_order,
        } => Ok(MetalResolvedWeightLayout::AxisReshapePermutation {
            values: Box::new(resolve_layout_node(values, weight)?),
            axis: *axis,
            logical_offset: *logical_offset,
            extent: *extent,
            reshape: reshape.clone(),
            stored_axis_order: stored_axis_order.clone(),
        }),
        PhysicalWeightLayout::Indexed {
            indices,
            values,
            source_axis_extent,
        } => Ok(MetalResolvedWeightLayout::Indexed {
            indices: axis_component(indices)?,
            values: Box::new(resolve_layout_node(values, weight)?),
            source_axis_extent: *source_axis_extent,
        }),
        PhysicalWeightLayout::ExpertStack {
            experts,
            expert_axis,
        } => Ok(MetalResolvedWeightLayout::ExpertStack {
            experts: experts
                .iter()
                .map(|expert| resolve_layout_node(expert, weight))
                .collect::<Result<Vec<_>, _>>()?,
            expert_axis: *expert_axis,
        }),
    }
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroU32;

    use super::*;
    use ferrum_interfaces::vnext::{
        BlockQuantizationSpec, CompositeWeightPart, ContractVersion, HadamardTransformSpec,
        ModelFamilyId, PhysicalStorageLayout, PhysicalWeightComponentBinding, QuantizationFormatId,
        QuantizationGrouping, QuantizationPacking, QuantizationSpec, WeightComponentRole,
        WeightComponentSpec, WeightFormatId, WeightLayoutId, WeightSchema, WeightTensorSpec,
    };

    fn id(value: &str) -> WeightId {
        WeightId::new(value).unwrap()
    }

    fn block_component(value: &str) -> WeightComponentSpec {
        WeightComponentSpec {
            id: id(value),
            role: WeightComponentRole::PackedValues,
            external_names: vec![format!("{value}.weight")],
            dimensions: vec![1, 4, 1],
            encoding: WeightEncoding::BlockQuantized(BlockQuantizationSpec {
                format_id: QuantizationFormatId::new("quantization.gguf.q6-k").unwrap(),
                logical_values_per_block: 256,
                bytes_per_block: 210,
            }),
            required: true,
        }
    }

    #[test]
    fn composite_layout_uses_semantic_offsets_not_component_sort_order() {
        let gate = id("component.z_gate");
        let up = id("component.a_up");
        let schema = WeightSchema {
            format_id: WeightFormatId::new("weight-format.gguf.native-block").unwrap(),
            layout_id: WeightLayoutId::new("weight-layout.test.composite").unwrap(),
            version: ContractVersion::new(1, 0),
            components: vec![block_component(gate.as_str()), block_component(up.as_str())],
            tensors: vec![WeightTensorSpec {
                id: id("weight.gate_up"),
                dimensions: vec![2, 4, 256],
                logical_element_type: ElementType::F16,
                physical_layout: PhysicalWeightLayout::Composite {
                    parts: vec![
                        CompositeWeightPart {
                            layout: Box::new(PhysicalWeightLayout::BlockQuantized {
                                blocks: PhysicalWeightComponentBinding {
                                    component_id: gate.clone(),
                                    storage: PhysicalStorageLayout::exact_contiguous(),
                                },
                                block_axis: 2,
                                block_padding: PhysicalWeightPadding::Exact,
                            }),
                            logical_offsets: vec![0, 0, 0],
                            extents: vec![1, 4, 256],
                        },
                        CompositeWeightPart {
                            layout: Box::new(PhysicalWeightLayout::BlockQuantized {
                                blocks: PhysicalWeightComponentBinding {
                                    component_id: up.clone(),
                                    storage: PhysicalStorageLayout::exact_contiguous(),
                                },
                                block_axis: 2,
                                block_padding: PhysicalWeightPadding::Exact,
                            }),
                            logical_offsets: vec![1, 0, 0],
                            extents: vec![1, 4, 256],
                        },
                    ],
                },
                required: true,
            }],
        };
        schema
            .validate(&ModelFamilyId::new("family.test").unwrap())
            .unwrap();
        let weight = ResolvedWeightBinding::from_schema(&schema, &id("weight.gate_up")).unwrap();
        assert_eq!(weight.components()[0].component_id(), &up);
        assert_eq!(weight.components()[1].component_id(), &gate);
        let MetalResolvedWeightLayout::Composite { parts } = resolve_layout(&weight).unwrap()
        else {
            panic!("expected composite layout");
        };
        let MetalResolvedWeightLayout::BlockQuantized {
            component: first, ..
        } = parts[0].layout
        else {
            panic!("expected gate block");
        };
        let MetalResolvedWeightLayout::BlockQuantized {
            component: second, ..
        } = parts[1].layout
        else {
            panic!("expected up block");
        };
        assert_eq!(parts[0].logical_offsets, [0, 0, 0]);
        assert_eq!(parts[1].logical_offsets, [1, 0, 0]);
        assert_eq!(weight.components()[first].component_id(), &gate);
        assert_eq!(weight.components()[second].component_id(), &up);
    }

    #[test]
    fn hadamard_layout_keeps_signs_distinct_from_block_values_in_either_id_order() {
        for (values_name, signs_name) in [
            ("component.a_values", "component.z_signs"),
            ("component.z_values", "component.a_signs"),
        ] {
            let values = id(values_name);
            let signs = id(signs_name);
            let mut blocks = block_component(values_name);
            blocks.dimensions = vec![4, 1];
            let schema = WeightSchema {
                format_id: WeightFormatId::new("weight-format.gguf.native-block").unwrap(),
                layout_id: WeightLayoutId::new("weight-layout.test.hadamard").unwrap(),
                version: ContractVersion::new(1, 0),
                components: vec![
                    blocks,
                    WeightComponentSpec {
                        id: signs.clone(),
                        role: WeightComponentRole::TransformSigns,
                        external_names: vec!["signs".to_owned()],
                        dimensions: vec![256],
                        encoding: WeightEncoding::Dense {
                            element_type: ElementType::F32,
                        },
                        required: true,
                    },
                ],
                tensors: vec![WeightTensorSpec {
                    id: id("weight.projection"),
                    dimensions: vec![4, 256],
                    logical_element_type: ElementType::F16,
                    physical_layout: PhysicalWeightLayout::Hadamard {
                        values: Box::new(PhysicalWeightLayout::BlockQuantized {
                            blocks: PhysicalWeightComponentBinding::exact_contiguous(
                                values.clone(),
                            ),
                            block_axis: 1,
                            block_padding: PhysicalWeightPadding::Exact,
                        }),
                        transform: HadamardTransformSpec {
                            block_size: NonZeroU32::new(128).unwrap(),
                            signs: HadamardSigns::Explicit(
                                PhysicalWeightComponentBinding::exact_contiguous(signs.clone()),
                            ),
                            application: HadamardApplication::BeforeMatmul {
                                input_permutation: None,
                            },
                        },
                    },
                    required: true,
                }],
            };
            schema
                .validate(&ModelFamilyId::new("family.test").unwrap())
                .unwrap();
            let weight =
                ResolvedWeightBinding::from_schema(&schema, &id("weight.projection")).unwrap();
            let MetalResolvedWeightLayout::Hadamard {
                values: layout,
                transform,
            } = resolve_layout(&weight).unwrap()
            else {
                panic!("expected Hadamard layout");
            };
            let MetalResolvedWeightLayout::BlockQuantized { component, .. } = *layout else {
                panic!("expected block values");
            };
            let signs_region = transform.signs_region.expect("explicit transform signs");
            assert_ne!(component, signs_region);
            assert_eq!(weight.components()[component].component_id(), &values);
            assert_eq!(weight.components()[signs_region].component_id(), &signs);
        }
    }

    #[test]
    fn quantized_block_grid_fails_closed_until_metal_supports_its_abi() {
        let values = id("component.fp8-values");
        let scales = id("component.fp8-scales");
        let quantization = QuantizationSpec {
            format_id: QuantizationFormatId::new("quantization.fp8-e4m3.block-grid").unwrap(),
            bits_per_weight: 8,
            grouping: QuantizationGrouping::block_2d([
                NonZeroU32::new(128).unwrap(),
                NonZeroU32::new(128).unwrap(),
            ]),
            packing: QuantizationPacking::Linear,
            scale_type: ElementType::Bf16,
            zero_point_type: None,
        };
        let schema = WeightSchema {
            format_id: WeightFormatId::new("weight-format.fp8-block-grid").unwrap(),
            layout_id: WeightLayoutId::new("weight-layout.fp8-block-grid").unwrap(),
            version: ContractVersion::new(1, 0),
            components: vec![
                WeightComponentSpec {
                    id: values.clone(),
                    role: WeightComponentRole::PackedValues,
                    external_names: vec!["weight".to_owned()],
                    dimensions: vec![130, 257],
                    encoding: WeightEncoding::Quantized(quantization),
                    required: true,
                },
                WeightComponentSpec {
                    id: scales.clone(),
                    role: WeightComponentRole::Scales,
                    external_names: vec!["weight_scale_inv".to_owned()],
                    dimensions: vec![2, 3],
                    encoding: WeightEncoding::Dense {
                        element_type: ElementType::Bf16,
                    },
                    required: true,
                },
            ],
            tensors: vec![WeightTensorSpec {
                id: id("weight.projection"),
                dimensions: vec![130, 257],
                logical_element_type: ElementType::Bf16,
                physical_layout: PhysicalWeightLayout::QuantizedBlockGrid {
                    packed_values: PhysicalWeightComponentBinding::exact_contiguous(values),
                    packed_dimensions: vec![130, 257],
                    scales: PhysicalWeightComponentBinding::exact_contiguous(scales),
                    block_axes: [0, 1],
                },
                required: true,
            }],
        };
        schema
            .validate(&ModelFamilyId::new("family.test").unwrap())
            .unwrap();
        let weight = ResolvedWeightBinding::from_schema(&schema, &id("weight.projection")).unwrap();
        assert_eq!(
            resolve_layout(&weight).unwrap_err(),
            "Metal does not support quantized block-grid physical weight layouts"
        );
    }
}
