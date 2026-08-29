//! Strict execution-weight resolver for CUDA Marlin FP8 W8A16.

use std::collections::BTreeMap;

use ferrum_interfaces::vnext::{
    ElementType, OperationInvocation, PhysicalStorageLayout, PhysicalWeightLayout,
    PhysicalWeightPadding, QuantizationGrouping, QuantizationPacking, ResolvedStorageComponent,
    ResolvedValueBinding, ResolvedWeightBinding, ResolvedWeightComponentLayout,
    WeightComponentRole, WeightEncoding, WeightId,
};

use crate::backend::cuda::vnext_runtime::{CudaBufferRegion, CudaDeviceBuffer};
use crate::marlin_fp8_materializer::{
    marlin_fp8_projection_shape_supported, MARLIN_FP8_QUANTIZATION_FORMAT_ID,
};

const MARLIN_REGION_ALIGNMENT_BYTES: u64 = 16;
pub(super) const MARLIN_FP8_CHANNELWISE_GROUP_SIZE: i32 = -1;

pub(super) struct CudaMarlinFp8Weight {
    packed_region: CudaBufferRegion,
    scales_region: CudaBufferRegion,
    output_features: u64,
    input_features: u64,
}

impl CudaMarlinFp8Weight {
    pub(super) fn packed_region(&self) -> &CudaBufferRegion {
        &self.packed_region
    }

    pub(super) fn scales_region(&self) -> &CudaBufferRegion {
        &self.scales_region
    }

    pub(super) const fn output_features(&self) -> u64 {
        self.output_features
    }

    pub(super) const fn input_features(&self) -> u64 {
        self.input_features
    }

    pub(super) fn into_regions(self) -> [CudaBufferRegion; 2] {
        [self.packed_region, self.scales_region]
    }
}

#[derive(Debug)]
struct MarlinFp8Metadata {
    packed_component_id: WeightId,
    scales_component_id: WeightId,
    output_features: u64,
    input_features: u64,
    packed_bytes: u64,
    scales_bytes: u64,
}

pub(super) fn resolve_marlin_fp8_weight(
    participant: &OperationInvocation<'_, CudaDeviceBuffer>,
    binding: &ResolvedValueBinding,
    logical_dimensions: &[u64],
) -> Result<CudaMarlinFp8Weight, String> {
    let weight = binding
        .weight()
        .ok_or_else(|| "CUDA Marlin FP8 weight lacks its typed physical layout".to_owned())?;
    validate_marlin_fp8_bound_contract(
        weight,
        binding.tensor().dimensions(),
        binding.tensor().element_type(),
        logical_dimensions,
    )?;
    resolve_marlin_fp8_layout(
        participant,
        binding,
        weight.physical_layout(),
        logical_dimensions,
    )
}

/// Resolve one exact Marlin FP8 quantized leaf. The binding may own storage
/// for sibling leaves in a composite layout; component identities select only
/// the packed values and scales belonging to `selected_layout`.
pub(super) fn resolve_marlin_fp8_layout(
    participant: &OperationInvocation<'_, CudaDeviceBuffer>,
    binding: &ResolvedValueBinding,
    selected_layout: &PhysicalWeightLayout,
    logical_dimensions: &[u64],
) -> Result<CudaMarlinFp8Weight, String> {
    let weight = binding
        .weight()
        .ok_or_else(|| "CUDA Marlin FP8 weight lacks its typed physical layout".to_owned())?;
    let metadata = validate_marlin_fp8_layout_contract(
        weight,
        binding.tensor().element_type(),
        selected_layout,
        logical_dimensions,
    )?;
    let (packed_stored, scales_stored) = select_marlin_fp8_storage(
        binding.storage().components(),
        &metadata.packed_component_id,
        &metadata.scales_component_id,
    )?;

    let packed_region = retain_component_region(
        participant,
        &metadata.packed_component_id,
        packed_stored,
        ElementType::U8,
        metadata.packed_bytes,
    )?;
    let scales_region = retain_component_region(
        participant,
        &metadata.scales_component_id,
        scales_stored,
        ElementType::F16,
        metadata.scales_bytes,
    )?;
    Ok(CudaMarlinFp8Weight {
        packed_region,
        scales_region,
        output_features: metadata.output_features,
        input_features: metadata.input_features,
    })
}

fn validate_marlin_fp8_bound_contract(
    weight: &ResolvedWeightBinding,
    bound_logical_dimensions: &[u64],
    logical_element_type: ElementType,
    caller_logical_dimensions: &[u64],
) -> Result<(), String> {
    if caller_logical_dimensions != bound_logical_dimensions {
        return Err(format!(
            "CUDA Marlin FP8 caller shape {caller_logical_dimensions:?} differs from bound shape {bound_logical_dimensions:?}"
        ));
    }
    weight
        .validate_logical(bound_logical_dimensions, logical_element_type)
        .map_err(|error| format!("CUDA Marlin FP8 logical contract is invalid: {error}"))
}

fn validate_marlin_fp8_layout_contract(
    weight: &ResolvedWeightBinding,
    logical_element_type: ElementType,
    selected_layout: &PhysicalWeightLayout,
    logical_dimensions: &[u64],
) -> Result<MarlinFp8Metadata, String> {
    let (output_features, input_features) = match logical_dimensions {
        [output_features, input_features] => (*output_features, *input_features),
        [1, output_features, input_features] => (*output_features, *input_features),
        _ => {
            return Err(
                "CUDA Marlin FP8 logical weight must have shape [N, K] or [1, N, K]".to_owned(),
            )
        }
    };
    if output_features == 0 || input_features == 0 || logical_element_type != ElementType::F16 {
        return Err(
            "CUDA Marlin FP8 logical weight must be a non-empty F16 matrix with an optional unit prefix"
                .to_owned(),
        );
    }
    let output_features_usize = usize::try_from(output_features)
        .map_err(|_| "CUDA Marlin FP8 output width exceeds usize".to_owned())?;
    let input_features_usize = usize::try_from(input_features)
        .map_err(|_| "CUDA Marlin FP8 input width exceeds usize".to_owned())?;
    if !marlin_fp8_projection_shape_supported(output_features_usize, input_features_usize) {
        return Err(format!(
            "CUDA Marlin FP8 projection shape [{output_features}, {input_features}] is not supported by the shared execution provider"
        ));
    }
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
    } = selected_layout
    else {
        return Err("CUDA Marlin FP8 requires a quantized leaf layout".to_owned());
    };
    if zero_points.is_some()
        || zero_point_packed_dimensions.is_some()
        || axis_indices.is_some()
        || permutation.is_some()
        || codebook.is_some()
    {
        return Err(
            "CUDA Marlin FP8 forbids zero-point, index, permutation, and codebook components"
                .to_owned(),
        );
    }
    if usize::try_from(*group_axis).ok() != packed_dimensions.len().checked_sub(1)
        || !matches!(group_padding, PhysicalWeightPadding::Exact)
    {
        return Err(
            "CUDA Marlin FP8 requires exact channelwise groups on the final input axis".to_owned(),
        );
    }
    if !is_exact_contiguous(&packed_values.storage) || !is_exact_contiguous(&scales.storage) {
        return Err(
            "CUDA Marlin FP8 packed values and scales must use exact contiguous storage".to_owned(),
        );
    }
    if packed_values.component_id == scales.component_id {
        return Err(
            "CUDA Marlin FP8 packed values and scales must have distinct identities".to_owned(),
        );
    }

    let mut component_by_id = BTreeMap::new();
    for component in weight.components() {
        if component_by_id
            .insert(component.component_id().clone(), component)
            .is_some()
        {
            return Err(format!(
                "CUDA Marlin FP8 layout duplicates component `{}`",
                component.component_id()
            ));
        }
    }
    let packed_component = required_component(
        &component_by_id,
        &packed_values.component_id,
        WeightComponentRole::PackedValues,
        "packed values",
    )?;
    let scales_component = required_component(
        &component_by_id,
        &scales.component_id,
        WeightComponentRole::Scales,
        "scales",
    )?;
    let WeightEncoding::Quantized(quantization) = packed_component.encoding() else {
        return Err("CUDA Marlin FP8 packed component is not quantized".to_owned());
    };
    quantization
        .validate()
        .map_err(|error| format!("CUDA Marlin FP8 quantization ABI is invalid: {error}"))?;
    if quantization.format_id.as_str() != MARLIN_FP8_QUANTIZATION_FORMAT_ID
        || quantization.bits_per_weight != 8
        || quantization.grouping != QuantizationGrouping::WholeAxis
        || quantization.packing != QuantizationPacking::Tiled
        || quantization.scale_type != ElementType::F16
        || quantization.zero_point_type.is_some()
    {
        return Err(format!(
            "CUDA Marlin FP8 component `{}` is not channelwise E4M3 tiled W8A16",
            packed_component.component_id()
        ));
    }
    if !matches!(
        scales_component.encoding(),
        WeightEncoding::Dense {
            element_type: ElementType::F16
        }
    ) {
        return Err("CUDA Marlin FP8 scales component must be dense F16".to_owned());
    }

    let expected_scales_dimensions = match packed_dimensions.as_slice() {
        [packed_output, packed_input]
            if *packed_output == output_features && *packed_input == input_features =>
        {
            vec![output_features, 1]
        }
        [1, packed_output, packed_input]
            if *packed_output == output_features && *packed_input == input_features =>
        {
            vec![1, output_features, 1]
        }
        _ => {
            return Err(format!(
                "CUDA Marlin FP8 packed shape must be [{output_features}, {input_features}] or [1, {output_features}, {input_features}]"
            ))
        }
    };
    if packed_component.physical_dimensions() != packed_dimensions.as_slice()
        || scales_component.physical_dimensions() != expected_scales_dimensions.as_slice()
    {
        return Err(format!(
            "CUDA Marlin FP8 component shapes must be packed {packed_dimensions:?} and scales {expected_scales_dimensions:?}"
        ));
    }
    let packed_bytes = checked_physical_bytes(packed_dimensions, 1, "packed")?;
    let scales_bytes = checked_physical_bytes(
        &expected_scales_dimensions,
        ElementType::F16.size_bytes(),
        "scales",
    )?;
    if packed_component
        .physical_bytes()
        .map_err(|error| error.to_string())?
        != packed_bytes
        || scales_component
            .physical_bytes()
            .map_err(|error| error.to_string())?
            != scales_bytes
    {
        return Err("CUDA Marlin FP8 component byte counts differ from the typed ABI".to_owned());
    }
    if !packed_bytes.is_multiple_of(MARLIN_REGION_ALIGNMENT_BYTES)
        || !scales_bytes.is_multiple_of(MARLIN_REGION_ALIGNMENT_BYTES)
    {
        return Err("CUDA Marlin FP8 physical byte counts are not 16-byte aligned".to_owned());
    }

    Ok(MarlinFp8Metadata {
        packed_component_id: packed_values.component_id.clone(),
        scales_component_id: scales.component_id.clone(),
        output_features,
        input_features,
        packed_bytes,
        scales_bytes,
    })
}

fn checked_physical_bytes(
    dimensions: &[u64],
    element_bytes: u64,
    label: &str,
) -> Result<u64, String> {
    dimensions
        .iter()
        .try_fold(element_bytes, |bytes, extent| bytes.checked_mul(*extent))
        .ok_or_else(|| format!("CUDA Marlin FP8 {label} byte count overflows u64"))
}

fn select_marlin_fp8_storage<'a>(
    storage: &'a [ResolvedStorageComponent],
    packed_component_id: &WeightId,
    scales_component_id: &WeightId,
) -> Result<(&'a ResolvedStorageComponent, &'a ResolvedStorageComponent), String> {
    let mut stored_by_id = BTreeMap::new();
    for stored in storage {
        let component_id = stored.component_id().ok_or_else(|| {
            "CUDA Marlin FP8 storage component lacks its physical identity".to_owned()
        })?;
        if stored_by_id.insert(component_id.clone(), stored).is_some() {
            return Err(format!(
                "CUDA Marlin FP8 storage duplicates component `{component_id}`"
            ));
        }
    }
    let packed_stored = stored_by_id
        .get(packed_component_id)
        .copied()
        .ok_or_else(|| {
            format!(
                "CUDA Marlin FP8 packed component `{packed_component_id}` has no resolved storage"
            )
        })?;
    let scales_stored = stored_by_id
        .get(scales_component_id)
        .copied()
        .ok_or_else(|| {
            format!(
                "CUDA Marlin FP8 scales component `{scales_component_id}` has no resolved storage"
            )
        })?;
    Ok((packed_stored, scales_stored))
}

fn required_component<'a>(
    component_by_id: &'a BTreeMap<WeightId, &'a ResolvedWeightComponentLayout>,
    component_id: &WeightId,
    expected_role: WeightComponentRole,
    label: &str,
) -> Result<&'a ResolvedWeightComponentLayout, String> {
    let component = component_by_id
        .get(component_id)
        .copied()
        .ok_or_else(|| format!("CUDA Marlin FP8 {label} component `{component_id}` is absent"))?;
    if component.role() != expected_role {
        return Err(format!(
            "CUDA Marlin FP8 component `{component_id}` has role {:?}, expected {expected_role:?}",
            component.role()
        ));
    }
    Ok(component)
}

fn is_exact_contiguous(storage: &PhysicalStorageLayout) -> bool {
    matches!(
        storage,
        PhysicalStorageLayout::Contiguous {
            padding: PhysicalWeightPadding::Exact
        }
    )
}

fn retain_component_region(
    participant: &OperationInvocation<'_, CudaDeviceBuffer>,
    component_id: &WeightId,
    stored: &ResolvedStorageComponent,
    expected_element_type: ElementType,
    expected_length_bytes: u64,
) -> Result<CudaBufferRegion, String> {
    if stored.element_type() != expected_element_type
        || stored.length_bytes() != expected_length_bytes
    {
        return Err(format!(
            "CUDA Marlin FP8 component `{component_id}` differs from its typed physical ABI"
        ));
    }
    let mut matching_views = participant
        .views()
        .iter()
        .filter(|view| view.resource_id() == stored.resource_id());
    let view = matching_views.next().ok_or_else(|| {
        format!("CUDA Marlin FP8 component `{component_id}` has no committed resource view")
    })?;
    if matching_views.next().is_some() {
        return Err(format!(
            "CUDA Marlin FP8 component `{component_id}` has ambiguous committed resource views"
        ));
    }
    let translated = view
        .translate(stored.offset_bytes(), stored.length_bytes())
        .map_err(|error| error.to_string())?;
    let mut physical_regions = translated.iter();
    let physical = physical_regions.next().ok_or_else(|| {
        format!("CUDA Marlin FP8 component `{component_id}` translated to no physical region")
    })?;
    if physical_regions.next().is_some() {
        return Err(format!(
            "CUDA Marlin FP8 component `{component_id}` is not physically contiguous"
        ));
    }
    let (buffer, range, retention) = physical.buffer_and_physical_range();
    let region = buffer
        .retained_region(range, retention)
        .map_err(|error| error.to_string())?;
    if region.element_type() != expected_element_type
        || region.length_bytes() != expected_length_bytes
        || !region
            .device_ptr()
            .is_multiple_of(MARLIN_REGION_ALIGNMENT_BYTES)
    {
        return Err(format!(
            "CUDA Marlin FP8 component `{component_id}` retained the wrong or unaligned physical range"
        ));
    }
    Ok(region)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::marlin_fp8_materializer::MARLIN_FP8_WEIGHT_FORMAT_ID;
    use ferrum_interfaces::vnext::{
        CompositeWeightPart, ContractVersion, PhysicalWeightComponentBinding, QuantizationFormatId,
        QuantizationSpec, ResourceId, WeightComponentSpec, WeightFormatId, WeightLayoutId,
        WeightSchema, WeightTensorSpec,
    };

    fn id(value: &str) -> WeightId {
        WeightId::new(value).unwrap()
    }

    fn packed_component(component_id: &WeightId, dimensions: &[u64]) -> WeightComponentSpec {
        WeightComponentSpec {
            id: component_id.clone(),
            role: WeightComponentRole::PackedValues,
            external_names: vec![format!("{component_id}.external")],
            dimensions: dimensions.to_vec(),
            encoding: WeightEncoding::Quantized(QuantizationSpec {
                format_id: QuantizationFormatId::new(MARLIN_FP8_QUANTIZATION_FORMAT_ID).unwrap(),
                bits_per_weight: 8,
                grouping: QuantizationGrouping::WholeAxis,
                packing: QuantizationPacking::Tiled,
                scale_type: ElementType::F16,
                zero_point_type: None,
            }),
            required: true,
        }
    }

    fn scales_component(component_id: &WeightId, dimensions: &[u64]) -> WeightComponentSpec {
        WeightComponentSpec {
            id: component_id.clone(),
            role: WeightComponentRole::Scales,
            external_names: vec![format!("{component_id}.external")],
            dimensions: dimensions.to_vec(),
            encoding: WeightEncoding::Dense {
                element_type: ElementType::F16,
            },
            required: true,
        }
    }

    fn dense_component(component_id: &WeightId, dimensions: &[u64]) -> WeightComponentSpec {
        WeightComponentSpec {
            id: component_id.clone(),
            role: WeightComponentRole::Values,
            external_names: vec![format!("{component_id}.external")],
            dimensions: dimensions.to_vec(),
            encoding: WeightEncoding::Dense {
                element_type: ElementType::F16,
            },
            required: true,
        }
    }

    fn marlin_layout(
        packed_id: &WeightId,
        scales_id: &WeightId,
        logical_dimensions: &[u64],
    ) -> PhysicalWeightLayout {
        PhysicalWeightLayout::Quantized {
            packed_values: PhysicalWeightComponentBinding::exact_contiguous(packed_id.clone()),
            packed_dimensions: logical_dimensions.to_vec(),
            scales: PhysicalWeightComponentBinding::exact_contiguous(scales_id.clone()),
            zero_points: None,
            zero_point_packed_dimensions: None,
            axis_indices: None,
            permutation: None,
            codebook: None,
            group_axis: u32::try_from(logical_dimensions.len() - 1).unwrap(),
            group_padding: PhysicalWeightPadding::Exact,
        }
    }

    fn valid_schema(schema_format_id: &str) -> WeightSchema {
        let packed_id = id("component.fp8.packed");
        let scales_id = id("component.fp8.scales");
        WeightSchema {
            format_id: WeightFormatId::new(schema_format_id).unwrap(),
            layout_id: WeightLayoutId::new("weight-layout.test.mixed-fp8").unwrap(),
            version: ContractVersion::new(1, 0),
            components: vec![
                WeightComponentSpec {
                    id: packed_id.clone(),
                    role: WeightComponentRole::PackedValues,
                    external_names: vec!["projection.fp8".to_owned()],
                    dimensions: vec![256, 128],
                    encoding: WeightEncoding::Quantized(QuantizationSpec {
                        format_id: QuantizationFormatId::new(MARLIN_FP8_QUANTIZATION_FORMAT_ID)
                            .unwrap(),
                        bits_per_weight: 8,
                        grouping: QuantizationGrouping::WholeAxis,
                        packing: QuantizationPacking::Tiled,
                        scale_type: ElementType::F16,
                        zero_point_type: None,
                    }),
                    required: true,
                },
                WeightComponentSpec {
                    id: scales_id.clone(),
                    role: WeightComponentRole::Scales,
                    external_names: vec!["projection.scale".to_owned()],
                    dimensions: vec![256, 1],
                    encoding: WeightEncoding::Dense {
                        element_type: ElementType::F16,
                    },
                    required: true,
                },
            ],
            tensors: vec![WeightTensorSpec {
                id: id("weight.projection"),
                dimensions: vec![256, 128],
                logical_element_type: ElementType::F16,
                physical_layout: PhysicalWeightLayout::Quantized {
                    packed_values: PhysicalWeightComponentBinding::exact_contiguous(packed_id),
                    packed_dimensions: vec![256, 128],
                    scales: PhysicalWeightComponentBinding::exact_contiguous(scales_id),
                    zero_points: None,
                    zero_point_packed_dimensions: None,
                    axis_indices: None,
                    permutation: None,
                    codebook: None,
                    group_axis: 1,
                    group_padding: PhysicalWeightPadding::Exact,
                },
                required: true,
            }],
        }
    }

    fn composite_schema() -> WeightSchema {
        let qkv_packed = id("component.qkv.fp8.packed");
        let qkv_scales = id("component.qkv.fp8.scales");
        let z_packed = id("component.z.fp8.packed");
        let z_scales = id("component.z.fp8.scales");
        let b_values = id("component.b.values");
        let a_values = id("component.a.values");
        WeightSchema {
            format_id: WeightFormatId::new("weight-format.execution.test.gda-composite").unwrap(),
            layout_id: WeightLayoutId::new("weight-layout.test.gda-composite").unwrap(),
            version: ContractVersion::new(1, 0),
            components: vec![
                packed_component(&qkv_packed, &[256, 128]),
                scales_component(&qkv_scales, &[256, 1]),
                packed_component(&z_packed, &[128, 128]),
                scales_component(&z_scales, &[128, 1]),
                dense_component(&b_values, &[32, 128]),
                dense_component(&a_values, &[32, 128]),
            ],
            tensors: vec![WeightTensorSpec {
                id: id("weight.gda.composite"),
                dimensions: vec![448, 128],
                logical_element_type: ElementType::F16,
                physical_layout: PhysicalWeightLayout::Composite {
                    parts: vec![
                        CompositeWeightPart {
                            layout: Box::new(marlin_layout(&qkv_packed, &qkv_scales, &[256, 128])),
                            logical_offsets: vec![0, 0],
                            extents: vec![256, 128],
                        },
                        CompositeWeightPart {
                            layout: Box::new(marlin_layout(&z_packed, &z_scales, &[128, 128])),
                            logical_offsets: vec![256, 0],
                            extents: vec![128, 128],
                        },
                        CompositeWeightPart {
                            layout: Box::new(PhysicalWeightLayout::Dense {
                                component_id: b_values,
                            }),
                            logical_offsets: vec![384, 0],
                            extents: vec![32, 128],
                        },
                        CompositeWeightPart {
                            layout: Box::new(PhysicalWeightLayout::Dense {
                                component_id: a_values,
                            }),
                            logical_offsets: vec![416, 0],
                            extents: vec![32, 128],
                        },
                    ],
                },
                required: true,
            }],
        }
    }

    fn unit_prefix_composite_schema() -> WeightSchema {
        let first_packed = id("component.first.fp8.packed");
        let first_scales = id("component.first.fp8.scales");
        let second_packed = id("component.second.fp8.packed");
        let second_scales = id("component.second.fp8.scales");
        WeightSchema {
            format_id: WeightFormatId::new("weight-format.execution.test.unit-prefix").unwrap(),
            layout_id: WeightLayoutId::new("weight-layout.test.unit-prefix").unwrap(),
            version: ContractVersion::new(1, 0),
            components: vec![
                packed_component(&first_packed, &[1, 256, 128]),
                scales_component(&first_scales, &[1, 256, 1]),
                packed_component(&second_packed, &[1, 256, 128]),
                scales_component(&second_scales, &[1, 256, 1]),
            ],
            tensors: vec![WeightTensorSpec {
                id: id("weight.unit-prefix.composite"),
                dimensions: vec![2, 256, 128],
                logical_element_type: ElementType::F16,
                physical_layout: PhysicalWeightLayout::Composite {
                    parts: vec![
                        CompositeWeightPart {
                            layout: Box::new(marlin_layout(
                                &first_packed,
                                &first_scales,
                                &[1, 256, 128],
                            )),
                            logical_offsets: vec![0, 0, 0],
                            extents: vec![1, 256, 128],
                        },
                        CompositeWeightPart {
                            layout: Box::new(marlin_layout(
                                &second_packed,
                                &second_scales,
                                &[1, 256, 128],
                            )),
                            logical_offsets: vec![1, 0, 0],
                            extents: vec![1, 256, 128],
                        },
                    ],
                },
                required: true,
            }],
        }
    }

    fn validate(schema: &WeightSchema) -> Result<MarlinFp8Metadata, String> {
        let weight = ResolvedWeightBinding::from_schema(schema, &schema.tensors[0].id).unwrap();
        validate_marlin_fp8_bound_contract(
            &weight,
            &schema.tensors[0].dimensions,
            schema.tensors[0].logical_element_type,
            &schema.tensors[0].dimensions,
        )?;
        validate_marlin_fp8_layout_contract(
            &weight,
            schema.tensors[0].logical_element_type,
            weight.physical_layout(),
            &schema.tensors[0].dimensions,
        )
    }

    #[test]
    fn component_abi_does_not_depend_on_the_enclosing_schema_format() {
        for schema_format_id in [
            MARLIN_FP8_WEIGHT_FORMAT_ID,
            "weight-format.execution.test.next-mixed-container",
        ] {
            let metadata = validate(&valid_schema(schema_format_id)).unwrap();
            assert_eq!(metadata.output_features, 256);
            assert_eq!(metadata.input_features, 128);
        }
    }

    #[test]
    fn rejects_a_different_component_quantization_abi() {
        let mut schema = valid_schema(MARLIN_FP8_WEIGHT_FORMAT_ID);
        let WeightEncoding::Quantized(spec) = &mut schema.components[0].encoding else {
            unreachable!();
        };
        spec.format_id = QuantizationFormatId::new("quantization.test.other").unwrap();
        let error = validate(&schema).unwrap_err();
        assert!(error.contains("not channelwise E4M3"), "{error}");
    }

    #[test]
    fn selects_one_quantized_leaf_with_composite_storage_siblings() {
        let schema = composite_schema();
        let weight = ResolvedWeightBinding::from_schema(&schema, &schema.tensors[0].id).unwrap();
        assert_eq!(weight.components().len(), 6);
        let PhysicalWeightLayout::Composite { parts } = weight.physical_layout() else {
            unreachable!();
        };
        let metadata = validate_marlin_fp8_layout_contract(
            &weight,
            ElementType::F16,
            parts[0].layout.as_ref(),
            &parts[0].extents,
        )
        .unwrap();
        assert_eq!(metadata.output_features, 256);
        assert_eq!(metadata.input_features, 128);

        let storage = weight
            .components()
            .iter()
            .enumerate()
            .map(|(index, component)| {
                ResolvedStorageComponent::new(
                    Some(component.component_id().clone()),
                    ResourceId::new(format!("resource/test/{index}")).unwrap(),
                    0,
                    component.physical_bytes().unwrap(),
                    component.physical_element_type(),
                )
                .unwrap()
            })
            .collect::<Vec<_>>();
        let (packed, scales) = select_marlin_fp8_storage(
            &storage,
            &metadata.packed_component_id,
            &metadata.scales_component_id,
        )
        .unwrap();
        assert_eq!(packed.component_id(), Some(&id("component.qkv.fp8.packed")));
        assert_eq!(scales.component_id(), Some(&id("component.qkv.fp8.scales")));
    }

    #[test]
    fn accepts_a_unit_prefix_leaf_and_scale_prefix() {
        let schema = unit_prefix_composite_schema();
        let weight = ResolvedWeightBinding::from_schema(&schema, &schema.tensors[0].id).unwrap();
        let PhysicalWeightLayout::Composite { parts } = weight.physical_layout() else {
            unreachable!();
        };
        for part in parts {
            let metadata = validate_marlin_fp8_layout_contract(
                &weight,
                ElementType::F16,
                part.layout.as_ref(),
                &[256, 128],
            )
            .unwrap();
            assert_eq!(metadata.output_features, 256);
            assert_eq!(metadata.input_features, 128);
            assert_eq!(metadata.packed_bytes, 256 * 128);
            assert_eq!(metadata.scales_bytes, 256 * ElementType::F16.size_bytes());
        }
    }

    #[test]
    fn rejects_non_quantized_and_non_unit_prefix_leaves() {
        let schema = composite_schema();
        let weight = ResolvedWeightBinding::from_schema(&schema, &schema.tensors[0].id).unwrap();
        let PhysicalWeightLayout::Composite { parts } = weight.physical_layout() else {
            unreachable!();
        };
        let error = validate_marlin_fp8_layout_contract(
            &weight,
            ElementType::F16,
            parts[2].layout.as_ref(),
            &parts[2].extents,
        )
        .unwrap_err();
        assert!(error.contains("quantized leaf"), "{error}");

        let prefix_schema = unit_prefix_composite_schema();
        let prefix_weight =
            ResolvedWeightBinding::from_schema(&prefix_schema, &prefix_schema.tensors[0].id)
                .unwrap();
        let PhysicalWeightLayout::Composite { parts } = prefix_weight.physical_layout() else {
            unreachable!();
        };
        let error = validate_marlin_fp8_layout_contract(
            &prefix_weight,
            ElementType::F16,
            parts[0].layout.as_ref(),
            &[2, 256, 128],
        )
        .unwrap_err();
        assert!(error.contains("[1, N, K]"), "{error}");
    }

    #[test]
    fn rejects_shapes_the_shared_marlin_provider_cannot_dispatch() {
        let mut schema = valid_schema(MARLIN_FP8_WEIGHT_FORMAT_ID);
        schema.components[0].dimensions = vec![64, 128];
        schema.components[1].dimensions = vec![64, 1];
        schema.tensors[0].dimensions = vec![64, 128];
        let PhysicalWeightLayout::Quantized {
            packed_dimensions, ..
        } = &mut schema.tensors[0].physical_layout
        else {
            unreachable!();
        };
        *packed_dimensions = vec![64, 128];

        let error = validate(&schema).unwrap_err();
        assert!(error.contains("not supported"), "{error}");
    }

    #[test]
    fn rejects_a_non_final_group_axis_and_non_f16_logical_dtype() {
        let schema = unit_prefix_composite_schema();
        let weight = ResolvedWeightBinding::from_schema(&schema, &schema.tensors[0].id).unwrap();
        let PhysicalWeightLayout::Composite { parts } = weight.physical_layout() else {
            unreachable!();
        };
        let mut wrong_axis = parts[0].layout.as_ref().clone();
        let PhysicalWeightLayout::Quantized { group_axis, .. } = &mut wrong_axis else {
            unreachable!();
        };
        *group_axis = 1;
        let error = validate_marlin_fp8_layout_contract(
            &weight,
            ElementType::F16,
            &wrong_axis,
            &[256, 128],
        )
        .unwrap_err();
        assert!(error.contains("final input axis"), "{error}");

        let error = validate_marlin_fp8_layout_contract(
            &weight,
            ElementType::Bf16,
            parts[0].layout.as_ref(),
            &[256, 128],
        )
        .unwrap_err();
        assert!(error.contains("F16"), "{error}");
    }
}
