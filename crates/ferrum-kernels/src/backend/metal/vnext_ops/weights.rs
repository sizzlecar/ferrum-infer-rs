//! Typed translation from a resolved logical weight to retained Metal regions.
//!
//! Providers consume this representation instead of guessing a physical ABI
//! from a model name, source file, component ordering, or byte length.

use std::collections::BTreeMap;

use ferrum_interfaces::vnext::{
    AxisWeightComponent, BlockQuantizationSpec, CompositeWeightPart, ElementType,
    OperationInvocation, PhysicalWeightLayout, PhysicalWeightPadding, ResolvedValueBinding,
    ResolvedWeightBinding, ResolvedWeightComponentLayout, WeightEncoding, WeightFormatId, WeightId,
};

use super::super::vnext_runtime::{MetalBufferRegion, MetalDeviceBuffer};

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
    format_id: WeightFormatId,
    logical_dimensions: Vec<u64>,
    logical_element_type: ElementType,
    components: Vec<MetalResolvedWeightComponent>,
    layout: MetalResolvedWeightLayout,
    regions: Vec<MetalBufferRegion>,
}

impl MetalResolvedWeight {
    pub(crate) fn format_id(&self) -> &WeightFormatId {
        &self.format_id
    }

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

pub(crate) fn resolve_weight(
    participant: &OperationInvocation<'_, MetalDeviceBuffer>,
    binding: &ResolvedValueBinding,
) -> Result<MetalResolvedWeight, String> {
    let weight = binding
        .weight()
        .ok_or_else(|| "Metal weight binding lacks its typed physical layout".to_owned())?;
    let stored_by_id = binding
        .storage()
        .components()
        .iter()
        .map(|component| {
            component
                .component_id()
                .map(|id| (id, component))
                .ok_or_else(|| "Metal weight component lacks its physical identity".to_owned())
        })
        .collect::<Result<BTreeMap<_, _>, _>>()?;

    let mut components = Vec::with_capacity(weight.components().len());
    let mut regions = Vec::with_capacity(weight.components().len());
    let mut index_by_id = BTreeMap::new();
    for component in weight.components() {
        let stored = stored_by_id.get(component.component_id()).ok_or_else(|| {
            format!(
                "Metal weight component `{}` has no resolved storage",
                component.component_id()
            )
        })?;
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
        let index = regions.len();
        index_by_id.insert(component.component_id().clone(), index);
        components.push(component_metadata(component));
        regions.push(region);
    }
    if index_by_id.len() != weight.components().len()
        || stored_by_id.len() != weight.components().len()
    {
        return Err("Metal weight component identities are duplicated or incomplete".to_owned());
    }
    let layout = resolve_layout(weight, &index_by_id)?;
    Ok(MetalResolvedWeight {
        format_id: weight.format_id().clone(),
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

fn resolve_layout(
    weight: &ResolvedWeightBinding,
    index_by_id: &BTreeMap<WeightId, usize>,
) -> Result<MetalResolvedWeightLayout, String> {
    resolve_layout_node(weight.physical_layout(), weight, index_by_id)
}

fn resolve_layout_node(
    layout: &PhysicalWeightLayout,
    weight: &ResolvedWeightBinding,
    index_by_id: &BTreeMap<WeightId, usize>,
) -> Result<MetalResolvedWeightLayout, String> {
    let component_index = |component_id: &WeightId| {
        index_by_id.get(component_id).copied().ok_or_else(|| {
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
                        layout: resolve_layout_node(&part.layout, weight, index_by_id)?,
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
            axis_indices: axis_indices.as_ref().map(axis_component).transpose()?,
            permutation: permutation.as_ref().map(axis_component).transpose()?,
            codebook: codebook
                .as_ref()
                .map(|component| component_index(&component.component_id))
                .transpose()?,
            group_axis: *group_axis,
            group_padding: group_padding.clone(),
        }),
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
        PhysicalWeightLayout::AxisReshapePermutation {
            values,
            axis,
            logical_offset,
            extent,
            reshape,
            stored_axis_order,
        } => Ok(MetalResolvedWeightLayout::AxisReshapePermutation {
            values: Box::new(resolve_layout_node(values, weight, index_by_id)?),
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
            values: Box::new(resolve_layout_node(values, weight, index_by_id)?),
            source_axis_extent: *source_axis_extent,
        }),
        PhysicalWeightLayout::ExpertStack {
            experts,
            expert_axis,
        } => Ok(MetalResolvedWeightLayout::ExpertStack {
            experts: experts
                .iter()
                .map(|expert| resolve_layout_node(expert, weight, index_by_id))
                .collect::<Result<Vec<_>, _>>()?,
            expert_axis: *expert_axis,
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferrum_interfaces::vnext::{
        BlockQuantizationSpec, CompositeWeightPart, ContractVersion, ModelFamilyId,
        PhysicalStorageLayout, PhysicalWeightComponentBinding, QuantizationFormatId,
        WeightComponentRole, WeightComponentSpec, WeightLayoutId, WeightSchema, WeightTensorSpec,
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
        let indexes = weight
            .components()
            .iter()
            .enumerate()
            .map(|(index, component)| (component.component_id().clone(), index))
            .collect();
        let MetalResolvedWeightLayout::Composite { parts } =
            resolve_layout(&weight, &indexes).unwrap()
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
}
