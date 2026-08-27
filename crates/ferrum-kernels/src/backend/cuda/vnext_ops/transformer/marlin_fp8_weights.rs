//! Strict execution-weight resolver for CUDA Marlin FP8 W8A16.

use std::collections::BTreeMap;

use ferrum_interfaces::vnext::{
    ElementType, OperationInvocation, PhysicalStorageLayout, PhysicalWeightLayout,
    PhysicalWeightPadding, QuantizationGrouping, QuantizationPacking, ResolvedStorageComponent,
    ResolvedValueBinding, ResolvedWeightBinding, ResolvedWeightComponentLayout,
    WeightComponentRole, WeightEncoding, WeightId,
};

use crate::backend::cuda::vnext_runtime::{CudaBufferRegion, CudaDeviceBuffer};
use crate::marlin_fp8_materializer::MARLIN_FP8_QUANTIZATION_FORMAT_ID;

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
    let metadata = validate_marlin_fp8_contract(
        weight,
        binding.tensor().dimensions(),
        binding.tensor().element_type(),
        logical_dimensions,
    )?;

    let mut stored_by_id = BTreeMap::new();
    for stored in binding.storage().components() {
        let component_id = stored.component_id().ok_or_else(|| {
            "CUDA Marlin FP8 storage component lacks its physical identity".to_owned()
        })?;
        if stored_by_id.insert(component_id.clone(), stored).is_some() {
            return Err(format!(
                "CUDA Marlin FP8 storage duplicates component `{component_id}`"
            ));
        }
    }
    if stored_by_id.len() != 2 {
        return Err(
            "CUDA Marlin FP8 storage must contain exactly packed values and scales".to_owned(),
        );
    }
    let packed_stored = stored_by_id
        .remove(&metadata.packed_component_id)
        .ok_or_else(|| {
            format!(
                "CUDA Marlin FP8 packed component `{}` has no resolved storage",
                metadata.packed_component_id
            )
        })?;
    let scales_stored = stored_by_id
        .remove(&metadata.scales_component_id)
        .ok_or_else(|| {
            format!(
                "CUDA Marlin FP8 scales component `{}` has no resolved storage",
                metadata.scales_component_id
            )
        })?;
    if !stored_by_id.is_empty() {
        return Err("CUDA Marlin FP8 storage contains an unreferenced component".to_owned());
    }

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

fn validate_marlin_fp8_contract(
    weight: &ResolvedWeightBinding,
    bound_logical_dimensions: &[u64],
    logical_element_type: ElementType,
    caller_logical_dimensions: &[u64],
) -> Result<MarlinFp8Metadata, String> {
    if caller_logical_dimensions != bound_logical_dimensions {
        return Err(format!(
            "CUDA Marlin FP8 caller shape {caller_logical_dimensions:?} differs from bound shape {bound_logical_dimensions:?}"
        ));
    }
    let [output_features, input_features] = bound_logical_dimensions else {
        return Err("CUDA Marlin FP8 logical weight must be a two-dimensional matrix".to_owned());
    };
    if *output_features == 0 || *input_features == 0 || logical_element_type != ElementType::F16 {
        return Err("CUDA Marlin FP8 logical weight must be a non-empty F16 matrix".to_owned());
    }
    weight
        .validate_logical(bound_logical_dimensions, logical_element_type)
        .map_err(|error| format!("CUDA Marlin FP8 logical contract is invalid: {error}"))?;
    let PhysicalWeightLayout::Quantized {
        packed_values,
        packed_dimensions,
        scales,
        zero_points,
        axis_indices,
        permutation,
        codebook,
        group_axis,
        group_padding,
    } = weight.physical_layout()
    else {
        return Err("CUDA Marlin FP8 requires one whole quantized layout".to_owned());
    };
    if zero_points.is_some()
        || axis_indices.is_some()
        || permutation.is_some()
        || codebook.is_some()
    {
        return Err(
            "CUDA Marlin FP8 forbids zero-point, index, permutation, and codebook components"
                .to_owned(),
        );
    }
    if *group_axis != 1 || !matches!(group_padding, PhysicalWeightPadding::Exact) {
        return Err(
            "CUDA Marlin FP8 requires exact channelwise groups on the input axis".to_owned(),
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
    if component_by_id.len() != 2 {
        return Err(
            "CUDA Marlin FP8 layout must contain exactly packed values and scales".to_owned(),
        );
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

    let expected_packed_dimensions = vec![*output_features, *input_features];
    let expected_scales_dimensions = vec![*output_features, 1];
    if packed_dimensions != &expected_packed_dimensions
        || packed_component.physical_dimensions() != expected_packed_dimensions
        || scales_component.physical_dimensions() != expected_scales_dimensions
    {
        return Err(format!(
            "CUDA Marlin FP8 physical shapes must be packed [{output_features}, {input_features}] and scales [{output_features}, 1]"
        ));
    }
    let packed_bytes = output_features
        .checked_mul(*input_features)
        .ok_or_else(|| "CUDA Marlin FP8 packed byte count overflows u64".to_owned())?;
    let scales_bytes = output_features
        .checked_mul(ElementType::F16.size_bytes())
        .ok_or_else(|| "CUDA Marlin FP8 scale byte count overflows u64".to_owned())?;
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
        output_features: *output_features,
        input_features: *input_features,
        packed_bytes,
        scales_bytes,
    })
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
        ContractVersion, PhysicalWeightComponentBinding, QuantizationFormatId, QuantizationSpec,
        WeightComponentSpec, WeightFormatId, WeightLayoutId, WeightSchema, WeightTensorSpec,
    };

    fn id(value: &str) -> WeightId {
        WeightId::new(value).unwrap()
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

    fn validate(schema: &WeightSchema) -> Result<MarlinFp8Metadata, String> {
        validate_marlin_fp8_contract(
            &ResolvedWeightBinding::from_schema(schema, &schema.tensors[0].id).unwrap(),
            &schema.tensors[0].dimensions,
            schema.tensors[0].logical_element_type,
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
}
