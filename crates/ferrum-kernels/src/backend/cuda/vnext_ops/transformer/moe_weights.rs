//! Typed GPTQ-Marlin MoE weight translation for CUDA vNext providers.
//!
//! This boundary accepts one exact physical ABI. It deliberately does not
//! infer component meaning from sorted ids, model names, or allocation sizes.

use std::collections::BTreeMap;

use ferrum_interfaces::vnext::{
    ElementType, OperationInvocation, PhysicalStorageLayout, PhysicalWeightLayout,
    PhysicalWeightPadding, QuantizationPacking, ResolvedValueBinding, ResolvedWeightBinding,
    ResolvedWeightComponentLayout, WeightComponentRole, WeightEncoding, WeightId,
};

use crate::backend::cuda::vnext_runtime::{CudaBufferRegion, CudaDeviceBuffer};

pub(super) const GPTQ_MARLIN_WEIGHT_FORMAT_ID: &str = "weight-format.safetensors.gptq-marlin-int4";
pub(super) const GPTQ_MARLIN_QUANTIZATION_FORMAT_ID: &str =
    "quantization.marlin.gptq-int4-symmetric";
const MARLIN_REGION_ALIGNMENT_BYTES: u64 = 16;

/// Retained, expert-major physical regions accepted by the CUDA Marlin-MoE
/// launch path.
pub(super) struct CudaMarlinMoeWeight {
    packed_region: CudaBufferRegion,
    scales_region: CudaBufferRegion,
    logical_dimensions: Vec<u64>,
    packed_physical_dimensions: Vec<u64>,
    scales_physical_dimensions: Vec<u64>,
    expert_count: u64,
    packed_expert_stride_bytes: u64,
    scales_expert_stride_bytes: u64,
    group_size: u32,
}

impl CudaMarlinMoeWeight {
    pub(super) fn packed_region(&self) -> &CudaBufferRegion {
        &self.packed_region
    }

    pub(super) fn scales_region(&self) -> &CudaBufferRegion {
        &self.scales_region
    }

    pub(super) fn logical_dimensions(&self) -> &[u64] {
        &self.logical_dimensions
    }

    pub(super) fn packed_physical_dimensions(&self) -> &[u64] {
        &self.packed_physical_dimensions
    }

    pub(super) fn scales_physical_dimensions(&self) -> &[u64] {
        &self.scales_physical_dimensions
    }

    pub(super) const fn expert_count(&self) -> u64 {
        self.expert_count
    }

    pub(super) const fn packed_expert_stride_bytes(&self) -> u64 {
        self.packed_expert_stride_bytes
    }

    pub(super) const fn scales_expert_stride_bytes(&self) -> u64 {
        self.scales_expert_stride_bytes
    }

    pub(super) const fn group_size(&self) -> u32 {
        self.group_size
    }

    pub(super) fn into_regions(self) -> [CudaBufferRegion; 2] {
        [self.packed_region, self.scales_region]
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct MarlinMoeWeightMetadata {
    packed_component_id: WeightId,
    scales_component_id: WeightId,
    logical_dimensions: Vec<u64>,
    packed_physical_dimensions: Vec<u64>,
    scales_physical_dimensions: Vec<u64>,
    expert_count: u64,
    packed_bytes: u64,
    scales_bytes: u64,
    packed_expert_stride_bytes: u64,
    scales_expert_stride_bytes: u64,
    group_size: u32,
}

/// Resolve a whole, expert-major GPTQ-Marlin INT4 weight into two retained
/// CUDA regions. `logical_dimensions` is supplied by the operation provider;
/// it must exactly equal the immutable logical shape on the binding.
pub(super) fn resolve_gptq_marlin_moe_weight(
    participant: &OperationInvocation<'_, CudaDeviceBuffer>,
    binding: &ResolvedValueBinding,
    logical_dimensions: &[u64],
) -> Result<CudaMarlinMoeWeight, String> {
    let weight = binding
        .weight()
        .ok_or_else(|| "CUDA Marlin-MoE weight lacks its typed physical layout".to_owned())?;
    let metadata = validate_gptq_marlin_moe_contract(
        weight,
        binding.tensor().dimensions(),
        binding.tensor().element_type(),
        logical_dimensions,
    )?;

    let mut stored_by_id = BTreeMap::new();
    for stored in binding.storage().components() {
        let component_id = stored.component_id().ok_or_else(|| {
            "CUDA Marlin-MoE storage component lacks its physical identity".to_owned()
        })?;
        if stored_by_id.insert(component_id.clone(), stored).is_some() {
            return Err(format!(
                "CUDA Marlin-MoE storage duplicates component `{component_id}`"
            ));
        }
    }
    if stored_by_id.len() != 2 {
        return Err(
            "CUDA Marlin-MoE storage must contain exactly packed values and scales".to_owned(),
        );
    }

    let packed_stored = stored_by_id
        .remove(&metadata.packed_component_id)
        .ok_or_else(|| {
            format!(
                "CUDA Marlin-MoE packed component `{}` has no resolved storage",
                metadata.packed_component_id
            )
        })?;
    let scales_stored = stored_by_id
        .remove(&metadata.scales_component_id)
        .ok_or_else(|| {
            format!(
                "CUDA Marlin-MoE scales component `{}` has no resolved storage",
                metadata.scales_component_id
            )
        })?;
    if !stored_by_id.is_empty() {
        return Err("CUDA Marlin-MoE storage contains an unreferenced component".to_owned());
    }

    let packed_region = retain_component_region(
        participant,
        &metadata.packed_component_id,
        packed_stored,
        ElementType::U8,
        metadata.packed_bytes,
        metadata.packed_expert_stride_bytes,
    )?;
    let scales_region = retain_component_region(
        participant,
        &metadata.scales_component_id,
        scales_stored,
        ElementType::F16,
        metadata.scales_bytes,
        metadata.scales_expert_stride_bytes,
    )?;

    Ok(CudaMarlinMoeWeight {
        packed_region,
        scales_region,
        logical_dimensions: metadata.logical_dimensions,
        packed_physical_dimensions: metadata.packed_physical_dimensions,
        scales_physical_dimensions: metadata.scales_physical_dimensions,
        expert_count: metadata.expert_count,
        packed_expert_stride_bytes: metadata.packed_expert_stride_bytes,
        scales_expert_stride_bytes: metadata.scales_expert_stride_bytes,
        group_size: metadata.group_size,
    })
}

fn validate_gptq_marlin_moe_contract(
    weight: &ResolvedWeightBinding,
    bound_logical_dimensions: &[u64],
    logical_element_type: ElementType,
    caller_logical_dimensions: &[u64],
) -> Result<MarlinMoeWeightMetadata, String> {
    if caller_logical_dimensions != bound_logical_dimensions {
        return Err(format!(
            "CUDA Marlin-MoE caller shape {caller_logical_dimensions:?} differs from bound shape {bound_logical_dimensions:?}"
        ));
    }
    if bound_logical_dimensions.len() < 3
        || bound_logical_dimensions.iter().any(|extent| *extent == 0)
    {
        return Err(
            "CUDA Marlin-MoE logical shape must be a non-empty expert-major matrix stack"
                .to_owned(),
        );
    }
    if logical_element_type != ElementType::F16 {
        return Err(format!(
            "CUDA Marlin-MoE logical element type must be F16, got {logical_element_type:?}"
        ));
    }
    weight
        .validate_logical(bound_logical_dimensions, logical_element_type)
        .map_err(|error| format!("CUDA Marlin-MoE logical contract is invalid: {error}"))?;
    if weight.format_id().as_str() != GPTQ_MARLIN_WEIGHT_FORMAT_ID {
        return Err(format!(
            "CUDA Marlin-MoE requires weight format `{GPTQ_MARLIN_WEIGHT_FORMAT_ID}`, got `{}`",
            weight.format_id()
        ));
    }

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
        return Err(
            "CUDA Marlin-MoE requires one whole quantized physical weight layout".to_owned(),
        );
    };
    if zero_points.is_some()
        || axis_indices.is_some()
        || permutation.is_some()
        || codebook.is_some()
    {
        return Err(
            "CUDA Marlin-MoE symmetric INT4 forbids zero-point, index, permutation, and codebook components"
                .to_owned(),
        );
    }
    if !matches!(group_padding, PhysicalWeightPadding::Exact) {
        return Err("CUDA Marlin-MoE group padding must be exact".to_owned());
    }
    if !is_exact_contiguous(&packed_values.storage) || !is_exact_contiguous(&scales.storage) {
        return Err(
            "CUDA Marlin-MoE packed values and scales must use exact contiguous storage".to_owned(),
        );
    }
    if packed_values.component_id == scales.component_id {
        return Err(
            "CUDA Marlin-MoE packed values and scales must have distinct component identities"
                .to_owned(),
        );
    }
    let last_axis = bound_logical_dimensions.len() - 1;
    if usize::try_from(*group_axis).ok() != Some(last_axis) {
        return Err(format!(
            "CUDA Marlin-MoE group axis {group_axis} must be the final logical axis {last_axis}"
        ));
    }

    let mut component_by_id = BTreeMap::new();
    for component in weight.components() {
        if component_by_id
            .insert(component.component_id().clone(), component)
            .is_some()
        {
            return Err(format!(
                "CUDA Marlin-MoE layout duplicates component `{}`",
                component.component_id()
            ));
        }
    }
    if component_by_id.len() != 2 {
        return Err(
            "CUDA Marlin-MoE layout must contain exactly packed values and scales".to_owned(),
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
        return Err("CUDA Marlin-MoE packed component must carry a quantized encoding".to_owned());
    };
    quantization
        .validate()
        .map_err(|error| format!("CUDA Marlin-MoE quantization ABI is invalid: {error}"))?;
    if quantization.format_id.as_str() != GPTQ_MARLIN_QUANTIZATION_FORMAT_ID
        || quantization.bits_per_weight != 4
        || quantization.group_size == 0
        || quantization.packing != QuantizationPacking::Tiled
        || quantization.scale_type != ElementType::F16
        || quantization.zero_point_type.is_some()
    {
        return Err(format!(
            "CUDA Marlin-MoE packed component `{}` is not symmetric tiled GPTQ-Marlin INT4 with F16 scales",
            packed_component.component_id()
        ));
    }
    if !matches!(
        scales_component.encoding(),
        WeightEncoding::Dense {
            element_type: ElementType::F16
        }
    ) {
        return Err(format!(
            "CUDA Marlin-MoE scales component `{}` must be dense F16",
            scales_component.component_id()
        ));
    }

    let expert_count = bound_logical_dimensions[0];
    let mut expected_packed_dimensions = bound_logical_dimensions.to_vec();
    if !expected_packed_dimensions[last_axis].is_multiple_of(2) {
        return Err(
            "CUDA Marlin-MoE final logical axis must contain an even number of INT4 values"
                .to_owned(),
        );
    }
    expected_packed_dimensions[last_axis] /= 2;
    if packed_dimensions != &expected_packed_dimensions
        || packed_component.physical_dimensions() != expected_packed_dimensions
    {
        return Err(format!(
            "CUDA Marlin-MoE packed physical shape must be {expected_packed_dimensions:?}"
        ));
    }

    let group_size = u64::from(quantization.group_size);
    let mut expected_scales_dimensions = bound_logical_dimensions.to_vec();
    if !expected_scales_dimensions[last_axis].is_multiple_of(group_size) {
        return Err(format!(
            "CUDA Marlin-MoE final logical axis {} is not divisible by group size {group_size}",
            expected_scales_dimensions[last_axis]
        ));
    }
    expected_scales_dimensions[last_axis] /= group_size;
    if scales_component.physical_dimensions() != expected_scales_dimensions {
        return Err(format!(
            "CUDA Marlin-MoE scales physical shape must be {expected_scales_dimensions:?}"
        ));
    }
    if expected_packed_dimensions[0] != expert_count
        || expected_scales_dimensions[0] != expert_count
    {
        return Err("CUDA Marlin-MoE first physical axis must equal the expert count".to_owned());
    }

    let packed_bytes = checked_physical_bytes(&expected_packed_dimensions, 1, "packed")?;
    let scales_bytes = checked_physical_bytes(
        &expected_scales_dimensions,
        ElementType::F16.size_bytes(),
        "scales",
    )?;
    if packed_component
        .physical_bytes()
        .map_err(|error| error.to_string())?
        != packed_bytes
    {
        return Err(format!(
            "CUDA Marlin-MoE packed component `{}` byte count differs from its physical shape",
            packed_component.component_id()
        ));
    }
    if scales_component
        .physical_bytes()
        .map_err(|error| error.to_string())?
        != scales_bytes
    {
        return Err(format!(
            "CUDA Marlin-MoE scales component `{}` byte count differs from its physical shape",
            scales_component.component_id()
        ));
    }

    let packed_expert_stride_bytes = expert_stride_bytes("packed", packed_bytes, expert_count)?;
    let scales_expert_stride_bytes = expert_stride_bytes("scales", scales_bytes, expert_count)?;

    Ok(MarlinMoeWeightMetadata {
        packed_component_id: packed_values.component_id.clone(),
        scales_component_id: scales.component_id.clone(),
        logical_dimensions: bound_logical_dimensions.to_vec(),
        packed_physical_dimensions: expected_packed_dimensions,
        scales_physical_dimensions: expected_scales_dimensions,
        expert_count,
        packed_bytes,
        scales_bytes,
        packed_expert_stride_bytes,
        scales_expert_stride_bytes,
        group_size: quantization.group_size,
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
        .ok_or_else(|| format!("CUDA Marlin-MoE {label} component `{component_id}` is absent"))?;
    if component.role() != expected_role {
        return Err(format!(
            "CUDA Marlin-MoE component `{component_id}` has role {:?}, expected {expected_role:?}",
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

fn checked_physical_bytes(
    dimensions: &[u64],
    bytes_per_element: u64,
    label: &str,
) -> Result<u64, String> {
    dimensions
        .iter()
        .try_fold(1_u64, |elements, extent| elements.checked_mul(*extent))
        .and_then(|elements| elements.checked_mul(bytes_per_element))
        .ok_or_else(|| format!("CUDA Marlin-MoE {label} byte count overflows u64"))
}

fn expert_stride_bytes(label: &str, length_bytes: u64, expert_count: u64) -> Result<u64, String> {
    if expert_count == 0 || length_bytes == 0 || !length_bytes.is_multiple_of(expert_count) {
        return Err(format!(
            "CUDA Marlin-MoE {label} byte length {length_bytes} is not exactly divisible by expert count {expert_count}"
        ));
    }
    let stride_bytes = length_bytes / expert_count;
    if !length_bytes.is_multiple_of(MARLIN_REGION_ALIGNMENT_BYTES)
        || !stride_bytes.is_multiple_of(MARLIN_REGION_ALIGNMENT_BYTES)
    {
        return Err(format!(
            "CUDA Marlin-MoE {label} length {length_bytes} and per-expert stride {stride_bytes} must be aligned to {MARLIN_REGION_ALIGNMENT_BYTES} bytes"
        ));
    }
    Ok(stride_bytes)
}

fn retain_component_region(
    participant: &OperationInvocation<'_, CudaDeviceBuffer>,
    component_id: &WeightId,
    stored: &ferrum_interfaces::vnext::ResolvedStorageComponent,
    expected_element_type: ElementType,
    expected_length_bytes: u64,
    expert_stride_bytes: u64,
) -> Result<CudaBufferRegion, String> {
    if stored.element_type() != expected_element_type
        || stored.length_bytes() != expected_length_bytes
    {
        return Err(format!(
            "CUDA Marlin-MoE component `{component_id}` differs from its typed physical ABI"
        ));
    }
    let mut matching_views = participant
        .views()
        .iter()
        .filter(|view| view.resource_id() == stored.resource_id());
    let view = matching_views.next().ok_or_else(|| {
        format!("CUDA Marlin-MoE component `{component_id}` has no committed resource view")
    })?;
    if matching_views.next().is_some() {
        return Err(format!(
            "CUDA Marlin-MoE component `{component_id}` has ambiguous committed resource views"
        ));
    }
    let translated = view
        .translate(stored.offset_bytes(), stored.length_bytes())
        .map_err(|error| error.to_string())?;
    let mut physical_regions = translated.iter();
    let physical = physical_regions.next().ok_or_else(|| {
        format!("CUDA Marlin-MoE component `{component_id}` translated to no physical region")
    })?;
    if physical_regions.next().is_some() {
        return Err(format!(
            "CUDA Marlin-MoE component `{component_id}` is not physically contiguous"
        ));
    }
    let (buffer, range, retention) = physical.buffer_and_physical_range();
    let region = buffer
        .retained_region(range, retention)
        .map_err(|error| error.to_string())?;
    if region.element_type() != expected_element_type
        || region.length_bytes() != expected_length_bytes
    {
        return Err(format!(
            "CUDA Marlin-MoE component `{component_id}` retained the wrong physical range"
        ));
    }
    validate_region_alignment(
        component_id.as_str(),
        region.device_ptr(),
        region.length_bytes(),
        expert_stride_bytes,
    )?;
    Ok(region)
}

fn validate_region_alignment(
    label: &str,
    device_ptr: u64,
    length_bytes: u64,
    expert_stride_bytes: u64,
) -> Result<(), String> {
    if device_ptr == 0 || !device_ptr.is_multiple_of(MARLIN_REGION_ALIGNMENT_BYTES) {
        return Err(format!(
            "CUDA Marlin-MoE {label} address 0x{device_ptr:x} must be non-null and aligned to {MARLIN_REGION_ALIGNMENT_BYTES} bytes"
        ));
    }
    if length_bytes == 0
        || !length_bytes.is_multiple_of(MARLIN_REGION_ALIGNMENT_BYTES)
        || expert_stride_bytes == 0
        || !expert_stride_bytes.is_multiple_of(MARLIN_REGION_ALIGNMENT_BYTES)
        || !length_bytes.is_multiple_of(expert_stride_bytes)
    {
        return Err(format!(
            "CUDA Marlin-MoE {label} length {length_bytes} and expert stride {expert_stride_bytes} are not aligned contiguous geometry"
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferrum_interfaces::vnext::{
        ContractVersion, PhysicalWeightComponentBinding, QuantizationFormatId, QuantizationSpec,
        WeightComponentSpec, WeightFormatId, WeightLayoutId, WeightSchema, WeightTensorSpec,
    };

    fn id(value: &str) -> WeightId {
        WeightId::new(value).unwrap()
    }

    fn valid_schema() -> WeightSchema {
        let packed_id = id("component.z_packed");
        let scales_id = id("component.a_scales");
        WeightSchema {
            format_id: WeightFormatId::new(GPTQ_MARLIN_WEIGHT_FORMAT_ID).unwrap(),
            layout_id: WeightLayoutId::new("weight-layout.test.marlin-moe").unwrap(),
            version: ContractVersion::new(1, 0),
            components: vec![
                WeightComponentSpec {
                    id: packed_id.clone(),
                    role: WeightComponentRole::PackedValues,
                    external_names: vec!["experts.qweight".to_owned()],
                    dimensions: vec![2, 64, 64],
                    encoding: WeightEncoding::Quantized(QuantizationSpec {
                        format_id: QuantizationFormatId::new(GPTQ_MARLIN_QUANTIZATION_FORMAT_ID)
                            .unwrap(),
                        bits_per_weight: 4,
                        group_size: 128,
                        packing: QuantizationPacking::Tiled,
                        scale_type: ElementType::F16,
                        zero_point_type: None,
                    }),
                    required: true,
                },
                WeightComponentSpec {
                    id: scales_id.clone(),
                    role: WeightComponentRole::Scales,
                    external_names: vec!["experts.scales".to_owned()],
                    dimensions: vec![2, 64, 1],
                    encoding: WeightEncoding::Dense {
                        element_type: ElementType::F16,
                    },
                    required: true,
                },
            ],
            tensors: vec![WeightTensorSpec {
                id: id("weight.experts"),
                dimensions: vec![2, 64, 128],
                logical_element_type: ElementType::F16,
                physical_layout: PhysicalWeightLayout::Quantized {
                    packed_values: PhysicalWeightComponentBinding::exact_contiguous(packed_id),
                    packed_dimensions: vec![2, 64, 64],
                    scales: PhysicalWeightComponentBinding::exact_contiguous(scales_id),
                    zero_points: None,
                    axis_indices: None,
                    permutation: None,
                    codebook: None,
                    group_axis: 2,
                    group_padding: PhysicalWeightPadding::Exact,
                },
                required: true,
            }],
        }
    }

    fn resolved(schema: &WeightSchema) -> ResolvedWeightBinding {
        ResolvedWeightBinding::from_schema(schema, &schema.tensors[0].id).unwrap()
    }

    fn validate(schema: &WeightSchema) -> Result<MarlinMoeWeightMetadata, String> {
        validate_gptq_marlin_moe_contract(
            &resolved(schema),
            &schema.tensors[0].dimensions,
            schema.tensors[0].logical_element_type,
            &schema.tensors[0].dimensions,
        )
    }

    #[test]
    fn accepts_component_identity_mapping_without_sorted_role_assumptions() {
        let schema = valid_schema();
        let weight = resolved(&schema);
        assert_eq!(weight.components()[0].role(), WeightComponentRole::Scales);

        let metadata = validate(&schema).unwrap();
        assert_eq!(metadata.packed_component_id, id("component.z_packed"));
        assert_eq!(metadata.scales_component_id, id("component.a_scales"));
        assert_eq!(metadata.logical_dimensions, [2, 64, 128]);
        assert_eq!(metadata.packed_physical_dimensions, [2, 64, 64]);
        assert_eq!(metadata.scales_physical_dimensions, [2, 64, 1]);
        assert_eq!(metadata.expert_count, 2);
        assert_eq!(metadata.packed_expert_stride_bytes, 4096);
        assert_eq!(metadata.scales_expert_stride_bytes, 128);
        assert_eq!(metadata.group_size, 128);
    }

    #[test]
    fn rejects_caller_shape_drift() {
        let schema = valid_schema();
        let error = validate_gptq_marlin_moe_contract(
            &resolved(&schema),
            &schema.tensors[0].dimensions,
            ElementType::F16,
            &[2, 32, 128],
        )
        .unwrap_err();
        assert!(error.contains("caller shape"), "{error}");
    }

    #[test]
    fn rejects_non_f16_logical_weights() {
        let schema = valid_schema();
        let error = validate_gptq_marlin_moe_contract(
            &resolved(&schema),
            &schema.tensors[0].dimensions,
            ElementType::Bf16,
            &schema.tensors[0].dimensions,
        )
        .unwrap_err();
        assert!(error.contains("must be F16"), "{error}");
    }

    #[test]
    fn rejects_another_weight_or_quantization_format() {
        let mut schema = valid_schema();
        schema.format_id = WeightFormatId::new("weight-format.test.other").unwrap();
        let error = validate(&schema).unwrap_err();
        assert!(error.contains("requires weight format"), "{error}");

        let mut schema = valid_schema();
        let WeightEncoding::Quantized(spec) = &mut schema.components[0].encoding else {
            unreachable!();
        };
        spec.format_id = QuantizationFormatId::new("quantization.test.other").unwrap();
        let error = validate(&schema).unwrap_err();
        assert!(error.contains("not symmetric tiled"), "{error}");
    }

    #[test]
    fn rejects_non_marlin_packing_or_non_f16_scales() {
        let mut schema = valid_schema();
        let WeightEncoding::Quantized(spec) = &mut schema.components[0].encoding else {
            unreachable!();
        };
        spec.packing = QuantizationPacking::Linear;
        let error = validate(&schema).unwrap_err();
        assert!(error.contains("not symmetric tiled"), "{error}");

        let mut schema = valid_schema();
        let WeightEncoding::Quantized(spec) = &mut schema.components[0].encoding else {
            unreachable!();
        };
        spec.scale_type = ElementType::Bf16;
        schema.components[1].encoding = WeightEncoding::Dense {
            element_type: ElementType::Bf16,
        };
        let error = validate(&schema).unwrap_err();
        assert!(error.contains("not symmetric tiled"), "{error}");
    }

    #[test]
    fn rejects_zero_point_sidecar() {
        let mut schema = valid_schema();
        let zero_points_id = id("component.zero_points");
        let WeightEncoding::Quantized(spec) = &mut schema.components[0].encoding else {
            unreachable!();
        };
        spec.zero_point_type = Some(ElementType::I32);
        schema.components.push(WeightComponentSpec {
            id: zero_points_id.clone(),
            role: WeightComponentRole::ZeroPoints,
            external_names: vec!["experts.qzeros".to_owned()],
            dimensions: vec![2, 64, 1],
            encoding: WeightEncoding::Dense {
                element_type: ElementType::I32,
            },
            required: true,
        });
        let PhysicalWeightLayout::Quantized { zero_points, .. } =
            &mut schema.tensors[0].physical_layout
        else {
            unreachable!();
        };
        *zero_points = Some(PhysicalWeightComponentBinding::exact_contiguous(
            zero_points_id,
        ));

        let error = validate(&schema).unwrap_err();
        assert!(error.contains("forbids zero-point"), "{error}");
    }

    #[test]
    fn rejects_non_contiguous_component_storage() {
        let mut schema = valid_schema();
        let PhysicalWeightLayout::Quantized { packed_values, .. } =
            &mut schema.tensors[0].physical_layout
        else {
            unreachable!();
        };
        packed_values.storage = PhysicalStorageLayout::Strided {
            strides_in_elements: vec![4096, 64, 1],
            padding: PhysicalWeightPadding::Exact,
        };

        let error = validate(&schema).unwrap_err();
        assert!(error.contains("exact contiguous"), "{error}");
    }

    #[test]
    fn rejects_unaligned_component_lengths_and_addresses() {
        let mut schema = valid_schema();
        schema.tensors[0].dimensions = vec![2, 1, 128];
        schema.components[0].dimensions = vec![2, 1, 64];
        schema.components[1].dimensions = vec![2, 1, 1];
        let PhysicalWeightLayout::Quantized {
            packed_dimensions, ..
        } = &mut schema.tensors[0].physical_layout
        else {
            unreachable!();
        };
        *packed_dimensions = vec![2, 1, 64];
        let error = validate(&schema).unwrap_err();
        assert!(error.contains("must be aligned to 16 bytes"), "{error}");

        let error = validate_region_alignment("packed", 0x1008, 4096, 2048).unwrap_err();
        assert!(error.contains("address"), "{error}");
        let error = validate_region_alignment("packed", 0x1000, 4095, 2048).unwrap_err();
        assert!(error.contains("not aligned contiguous geometry"), "{error}");
    }
}
