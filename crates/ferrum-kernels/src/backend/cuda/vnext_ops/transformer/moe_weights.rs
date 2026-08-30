//! Typed Marlin weight translation for CUDA vNext providers.
//!
//! This boundary accepts one exact physical ABI. It deliberately does not
//! infer component meaning from sorted ids, model names, or allocation sizes.

use std::collections::BTreeMap;

use ferrum_interfaces::vnext::{
    ElementType, OperationInvocation, PhysicalStorageLayout, PhysicalWeightLayout,
    PhysicalWeightPadding, QuantizationGrouping, QuantizationPacking, ResolvedValueBinding,
    ResolvedWeightBinding, ResolvedWeightComponentLayout, WeightComponentRole, WeightEncoding,
    WeightId,
};

use crate::backend::cuda::marlin::MarlinMoeF16WeightType;
use crate::backend::cuda::vnext_runtime::{CudaBufferRegion, CudaDeviceBuffer};
use crate::marlin_fp8_materializer::MARLIN_FP8_QUANTIZATION_FORMAT_ID;

pub(super) const GPTQ_MARLIN_WEIGHT_FORMAT_ID: &str = "weight-format.safetensors.gptq-marlin-int4";
pub(super) const GPTQ_MARLIN_QUANTIZATION_FORMAT_ID: &str =
    "quantization.marlin.gptq-int4-symmetric";
pub(super) const COMPRESSED_TENSORS_MARLIN_WEIGHT_FORMAT_ID: &str =
    "weight-format.safetensors.compressed-tensors-marlin-int4";
pub(super) const COMPRESSED_TENSORS_MARLIN_QUANTIZATION_FORMAT_ID: &str =
    "quantization.marlin.compressed-tensors-int4-asymmetric";
pub(in crate::backend::cuda::vnext_ops) const GPTQ_MARLIN_CAPABILITY_ID: &str =
    "capability.kernel.cuda.marlin.gptq-int4-w4a16";
pub(in crate::backend::cuda::vnext_ops) const COMPRESSED_TENSORS_MARLIN_CAPABILITY_ID: &str =
    "capability.kernel.cuda.marlin.compressed-tensors-int4-asymmetric-w4a16";
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
    group_size: i32,
    weight_type: MarlinMoeF16WeightType,
}

pub(super) type CudaMarlinGptqMatrixWeight = CudaMarlinMoeWeight;

pub(super) struct CudaMarlinCompressedTensorsMatrixWeight {
    packed_region: CudaBufferRegion,
    scales_region: CudaBufferRegion,
    zero_points_region: CudaBufferRegion,
    logical_dimensions: Vec<u64>,
    packed_physical_dimensions: Vec<u64>,
    scales_physical_dimensions: Vec<u64>,
    zero_points_physical_dimensions: Vec<u64>,
    group_size: u32,
}

impl CudaMarlinCompressedTensorsMatrixWeight {
    pub(super) fn packed_region(&self) -> &CudaBufferRegion {
        &self.packed_region
    }

    pub(super) fn scales_region(&self) -> &CudaBufferRegion {
        &self.scales_region
    }

    pub(super) fn zero_points_region(&self) -> &CudaBufferRegion {
        &self.zero_points_region
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

    pub(super) fn zero_points_physical_dimensions(&self) -> &[u64] {
        &self.zero_points_physical_dimensions
    }

    pub(super) const fn group_size(&self) -> u32 {
        self.group_size
    }

    pub(super) fn into_regions(self) -> [CudaBufferRegion; 3] {
        [
            self.packed_region,
            self.scales_region,
            self.zero_points_region,
        ]
    }
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

    pub(super) const fn group_size(&self) -> i32 {
        self.group_size
    }

    pub(super) const fn weight_type(&self) -> MarlinMoeF16WeightType {
        self.weight_type
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
    group_size: i32,
    weight_type: MarlinMoeF16WeightType,
}

/// Resolve a whole, expert-major GPTQ-Marlin INT4 weight into two retained
/// CUDA regions. `logical_dimensions` is supplied by the operation provider;
/// it must exactly equal the immutable logical shape on the binding.
pub(super) fn resolve_gptq_marlin_moe_weight(
    participant: &OperationInvocation<'_, CudaDeviceBuffer>,
    binding: &ResolvedValueBinding,
    logical_dimensions: &[u64],
) -> Result<CudaMarlinMoeWeight, String> {
    if logical_dimensions.len() < 3 {
        return Err(
            "CUDA Marlin-MoE logical shape must be a non-empty expert-major matrix stack"
                .to_owned(),
        );
    }
    let projection_axes = &logical_dimensions[1..logical_dimensions.len() - 2];
    let projections_per_expert = projection_axes
        .iter()
        .try_fold(1_u64, |count, extent| count.checked_mul(*extent))
        .ok_or_else(|| "CUDA Marlin-MoE projection count overflows".to_owned())?;
    let output_features = logical_dimensions[logical_dimensions.len() - 2]
        .checked_mul(projections_per_expert)
        .ok_or_else(|| "CUDA Marlin-MoE fused output width overflows".to_owned())?;
    let input_features = logical_dimensions[logical_dimensions.len() - 1];
    validate_marlin_thread_tile(output_features, input_features, "CUDA Marlin-MoE")?;
    resolve_gptq_marlin_weight(participant, binding, logical_dimensions)
}

/// Resolve one whole expert-major channelwise E4M3 Marlin stack. Routed
/// gate/up is `[E, 2, N, K]`; routed down is `[E, N, K]`. Splitting experts or
/// projections into unrelated physical regions is deliberately rejected.
pub(super) fn resolve_marlin_fp8_moe_weight(
    participant: &OperationInvocation<'_, CudaDeviceBuffer>,
    binding: &ResolvedValueBinding,
    logical_dimensions: &[u64],
) -> Result<CudaMarlinMoeWeight, String> {
    let (output_features, input_features) = match logical_dimensions {
        [expert_count, output_features, input_features] if *expert_count > 0 => {
            (*output_features, *input_features)
        }
        [expert_count, 2, output_features, input_features] if *expert_count > 0 => (
            output_features
                .checked_mul(2)
                .ok_or_else(|| "CUDA Marlin FP8 MoE fused gate/up width overflows".to_owned())?,
            *input_features,
        ),
        _ => {
            return Err(
                "CUDA Marlin FP8 MoE logical shape must be [E, N, K] or [E, 2, N, K]".to_owned(),
            )
        }
    };
    validate_marlin_thread_tile(output_features, input_features, "CUDA Marlin FP8 MoE")?;
    let weight = binding
        .weight()
        .ok_or_else(|| "CUDA Marlin FP8 MoE weight lacks its typed physical layout".to_owned())?;
    let metadata = validate_marlin_fp8_moe_contract(
        weight,
        binding.tensor().dimensions(),
        binding.tensor().element_type(),
        logical_dimensions,
    )?;
    resolve_marlin_moe_weight_from_metadata(participant, binding, metadata)
}

/// Resolve one exact rank-2 GPTQ-Marlin projection matrix `[N, K]`.
pub(super) fn resolve_gptq_marlin_matrix_weight(
    participant: &OperationInvocation<'_, CudaDeviceBuffer>,
    binding: &ResolvedValueBinding,
    logical_dimensions: &[u64],
) -> Result<CudaMarlinGptqMatrixWeight, String> {
    let [output_features, input_features] = logical_dimensions else {
        return Err(
            "CUDA GPTQ-Marlin projection must have exactly two logical dimensions [N, K]"
                .to_owned(),
        );
    };
    validate_marlin_thread_tile(
        *output_features,
        *input_features,
        "CUDA GPTQ-Marlin projection",
    )?;
    resolve_gptq_marlin_weight(participant, binding, logical_dimensions)
}

/// Resolve one exact rank-2 compressed-tensors asymmetric INT4 projection.
pub(super) fn resolve_compressed_tensors_marlin_matrix_weight(
    participant: &OperationInvocation<'_, CudaDeviceBuffer>,
    binding: &ResolvedValueBinding,
    logical_dimensions: &[u64],
) -> Result<CudaMarlinCompressedTensorsMatrixWeight, String> {
    let weight = binding.weight().ok_or_else(|| {
        "CUDA compressed-tensors Marlin weight lacks its typed physical layout".to_owned()
    })?;
    resolve_compressed_tensors_marlin_layout(
        participant,
        binding,
        weight.physical_layout(),
        logical_dimensions,
    )
}

/// Resolve a quantized leaf inside a composite projection. The full binding
/// owns storage for every leaf; component identities keep the selection exact.
pub(super) fn resolve_compressed_tensors_marlin_layout(
    participant: &OperationInvocation<'_, CudaDeviceBuffer>,
    binding: &ResolvedValueBinding,
    layout: &PhysicalWeightLayout,
    logical_dimensions: &[u64],
) -> Result<CudaMarlinCompressedTensorsMatrixWeight, String> {
    let [output_features, input_features] = logical_dimensions else {
        return Err("CUDA compressed-tensors Marlin projection must have shape [N, K]".to_owned());
    };
    validate_marlin_thread_tile(
        *output_features,
        *input_features,
        "CUDA compressed-tensors Marlin projection",
    )?;
    if binding.tensor().element_type() != ElementType::F16 {
        return Err("CUDA compressed-tensors Marlin logical dtype must be F16".to_owned());
    }
    let PhysicalWeightLayout::Quantized {
        packed_values,
        packed_dimensions,
        scales,
        zero_points: Some(zero_points),
        zero_point_packed_dimensions: Some(zero_point_dimensions),
        axis_indices,
        permutation,
        codebook,
        group_axis,
        group_padding,
    } = layout
    else {
        return Err(
            "CUDA compressed-tensors Marlin requires a quantized layout with packed zero points"
                .to_owned(),
        );
    };
    if axis_indices.is_some() || permutation.is_some() || codebook.is_some() {
        return Err(
            "CUDA compressed-tensors Marlin forbids index, permutation, and codebook components"
                .to_owned(),
        );
    }
    if usize::try_from(*group_axis).ok() != packed_dimensions.len().checked_sub(1)
        || !matches!(group_padding, PhysicalWeightPadding::Exact)
    {
        return Err(
            "CUDA compressed-tensors Marlin requires exact grouping on the final matrix axis"
                .to_owned(),
        );
    }
    if !is_exact_contiguous(&packed_values.storage)
        || !is_exact_contiguous(&scales.storage)
        || !is_exact_contiguous(&zero_points.storage)
    {
        return Err(
            "CUDA compressed-tensors Marlin components must use exact contiguous storage"
                .to_owned(),
        );
    }
    let mut component_by_id = BTreeMap::new();
    for component in binding
        .weight()
        .expect("weight presence was checked")
        .components()
    {
        component_by_id.insert(component.component_id().clone(), component);
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
    let zero_points_component = required_component(
        &component_by_id,
        &zero_points.component_id,
        WeightComponentRole::ZeroPoints,
        "zero points",
    )?;
    let WeightEncoding::Quantized(quantization) = packed_component.encoding() else {
        return Err(
            "CUDA compressed-tensors packed component lacks quantization metadata".to_owned(),
        );
    };
    let group_size = quantization.grouping.fixed_size().ok_or_else(|| {
        "CUDA compressed-tensors Marlin requires fixed quantization groups".to_owned()
    })?;
    if quantization.format_id.as_str() != COMPRESSED_TENSORS_MARLIN_QUANTIZATION_FORMAT_ID
        || quantization.bits_per_weight != 4
        || quantization.packing != QuantizationPacking::Tiled
        || quantization.scale_type != ElementType::F16
        || quantization.zero_point_type != Some(ElementType::I32)
        || group_size != 32
        || !matches!(
            scales_component.encoding(),
            WeightEncoding::Dense {
                element_type: ElementType::F16
            }
        )
        || !matches!(
            zero_points_component.encoding(),
            WeightEncoding::Dense {
                element_type: ElementType::I32
            }
        )
    {
        return Err(
            "CUDA compressed-tensors Marlin requires tiled asymmetric INT4/group32, F16 scales, and packed I32 zero points"
                .to_owned(),
        );
    }
    let packed_tail = [*output_features, *input_features / 2];
    let scales_tail = [*output_features, *input_features / u64::from(group_size)];
    let zero_points_tail = [
        *input_features / u64::from(group_size),
        *output_features / 8,
    ];
    if !has_unit_prefix_and_tail(packed_dimensions, &packed_tail)
        || packed_component.physical_dimensions() != packed_dimensions
        || !has_unit_prefix_and_tail(scales_component.physical_dimensions(), &scales_tail)
        || !has_unit_prefix_and_tail(zero_point_dimensions, &zero_points_tail)
        || zero_points_component.physical_dimensions() != zero_point_dimensions
    {
        return Err(format!(
            "CUDA compressed-tensors Marlin physical shapes differ from [N={output_features}, K={input_features}]"
        ));
    }
    let packed_bytes = checked_physical_bytes(packed_dimensions, 1, "packed")?;
    let scales_bytes = checked_physical_bytes(
        scales_component.physical_dimensions(),
        ElementType::F16.size_bytes(),
        "scales",
    )?;
    let zero_points_bytes = checked_physical_bytes(
        zero_point_dimensions,
        ElementType::I32.size_bytes(),
        "zero points",
    )?;
    let stored_by_id = binding
        .storage()
        .components()
        .iter()
        .filter_map(|stored| stored.component_id().map(|id| (id.clone(), stored)))
        .collect::<BTreeMap<_, _>>();
    let packed_stored = stored_by_id
        .get(&packed_values.component_id)
        .ok_or_else(|| {
            format!(
                "CUDA compressed-tensors packed component `{}` has no storage",
                packed_values.component_id
            )
        })?;
    let scales_stored = stored_by_id.get(&scales.component_id).ok_or_else(|| {
        format!(
            "CUDA compressed-tensors scales component `{}` has no storage",
            scales.component_id
        )
    })?;
    let zero_points_stored = stored_by_id.get(&zero_points.component_id).ok_or_else(|| {
        format!(
            "CUDA compressed-tensors zero-point component `{}` has no storage",
            zero_points.component_id
        )
    })?;
    Ok(CudaMarlinCompressedTensorsMatrixWeight {
        packed_region: retain_component_region(
            participant,
            &packed_values.component_id,
            packed_stored,
            ElementType::U8,
            packed_bytes,
            packed_bytes,
        )?,
        scales_region: retain_component_region(
            participant,
            &scales.component_id,
            scales_stored,
            ElementType::F16,
            scales_bytes,
            scales_bytes,
        )?,
        zero_points_region: retain_component_region(
            participant,
            &zero_points.component_id,
            zero_points_stored,
            ElementType::I32,
            zero_points_bytes,
            zero_points_bytes,
        )?,
        logical_dimensions: logical_dimensions.to_vec(),
        packed_physical_dimensions: packed_dimensions.clone(),
        scales_physical_dimensions: scales_component.physical_dimensions().to_vec(),
        zero_points_physical_dimensions: zero_point_dimensions.clone(),
        group_size,
    })
}

fn has_unit_prefix_and_tail(dimensions: &[u64], tail: &[u64; 2]) -> bool {
    dimensions.len() >= 2
        && dimensions[dimensions.len() - 2..] == *tail
        && dimensions[..dimensions.len() - 2]
            .iter()
            .all(|extent| *extent == 1)
}

pub(super) fn resolve_dense_f16_layout_region(
    participant: &OperationInvocation<'_, CudaDeviceBuffer>,
    binding: &ResolvedValueBinding,
    layout: &PhysicalWeightLayout,
    logical_dimensions: &[u64],
) -> Result<CudaBufferRegion, String> {
    let component_id = match layout {
        PhysicalWeightLayout::Dense { component_id } => component_id,
        PhysicalWeightLayout::Stored { component } if is_exact_contiguous(&component.storage) => {
            &component.component_id
        }
        _ => {
            return Err(
                "CUDA segmented projection dense leaf must use exact contiguous storage".to_owned(),
            )
        }
    };
    let weight = binding
        .weight()
        .ok_or_else(|| "CUDA segmented projection lacks a physical weight binding".to_owned())?;
    let component = weight
        .components()
        .iter()
        .find(|component| component.component_id() == component_id)
        .ok_or_else(|| format!("CUDA dense component `{component_id}` is absent"))?;
    if component.role() != WeightComponentRole::Values
        || !matches!(
            component.encoding(),
            WeightEncoding::Dense {
                element_type: ElementType::F16
            }
        )
        || component.physical_dimensions() != logical_dimensions
    {
        return Err(format!(
            "CUDA dense component `{component_id}` differs from its F16 matrix contract"
        ));
    }
    let stored = binding
        .storage()
        .components()
        .iter()
        .find(|stored| stored.component_id() == Some(component_id))
        .ok_or_else(|| format!("CUDA dense component `{component_id}` has no storage"))?;
    let bytes = checked_physical_bytes(
        logical_dimensions,
        ElementType::F16.size_bytes(),
        "dense component",
    )?;
    retain_component_region(
        participant,
        component_id,
        stored,
        ElementType::F16,
        bytes,
        bytes,
    )
}

fn resolve_gptq_marlin_weight(
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

    resolve_marlin_moe_weight_from_metadata(participant, binding, metadata)
}

fn resolve_marlin_moe_weight_from_metadata(
    participant: &OperationInvocation<'_, CudaDeviceBuffer>,
    binding: &ResolvedValueBinding,
    metadata: MarlinMoeWeightMetadata,
) -> Result<CudaMarlinMoeWeight, String> {
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
        weight_type: metadata.weight_type,
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
    if bound_logical_dimensions.len() < 2
        || bound_logical_dimensions.iter().any(|extent| *extent == 0)
    {
        return Err("CUDA GPTQ-Marlin logical shape must end in a non-empty matrix".to_owned());
    }
    if logical_element_type != ElementType::F16 {
        return Err(format!(
            "CUDA Marlin-MoE logical element type must be F16, got {logical_element_type:?}"
        ));
    }
    weight
        .validate_logical(bound_logical_dimensions, logical_element_type)
        .map_err(|error| format!("CUDA Marlin-MoE logical contract is invalid: {error}"))?;
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
    } = weight.physical_layout()
    else {
        return Err(
            "CUDA Marlin-MoE requires one whole quantized physical weight layout".to_owned(),
        );
    };
    if zero_points.is_some()
        || zero_point_packed_dimensions.is_some()
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
    let group_size = quantization
        .grouping
        .fixed_size()
        .ok_or_else(|| "CUDA Marlin-MoE requires fixed-size GPTQ quantization groups".to_owned())?;
    if quantization.format_id.as_str() != GPTQ_MARLIN_QUANTIZATION_FORMAT_ID
        || quantization.bits_per_weight != 4
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

    let expert_count = if bound_logical_dimensions.len() == 2 {
        1
    } else {
        bound_logical_dimensions[0]
    };
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

    let group_size_u64 = u64::from(group_size);
    let mut expected_scales_dimensions = bound_logical_dimensions.to_vec();
    if !expected_scales_dimensions[last_axis].is_multiple_of(group_size_u64) {
        return Err(format!(
            "CUDA Marlin-MoE final logical axis {} is not divisible by group size {group_size_u64}",
            expected_scales_dimensions[last_axis]
        ));
    }
    expected_scales_dimensions[last_axis] /= group_size_u64;
    if scales_component.physical_dimensions() != expected_scales_dimensions {
        return Err(format!(
            "CUDA Marlin-MoE scales physical shape must be {expected_scales_dimensions:?}"
        ));
    }
    if bound_logical_dimensions.len() >= 3
        && (expected_packed_dimensions[0] != expert_count
            || expected_scales_dimensions[0] != expert_count)
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
        group_size: i32::try_from(group_size)
            .map_err(|_| "CUDA Marlin-MoE group size exceeds i32".to_owned())?,
        weight_type: MarlinMoeF16WeightType::U4B8,
    })
}

fn validate_marlin_fp8_moe_contract(
    weight: &ResolvedWeightBinding,
    bound_logical_dimensions: &[u64],
    logical_element_type: ElementType,
    caller_logical_dimensions: &[u64],
) -> Result<MarlinMoeWeightMetadata, String> {
    if caller_logical_dimensions != bound_logical_dimensions {
        return Err(format!(
            "CUDA Marlin FP8 MoE caller shape {caller_logical_dimensions:?} differs from bound shape {bound_logical_dimensions:?}"
        ));
    }
    match bound_logical_dimensions {
        [expert_count, output_features, input_features]
            if *expert_count > 0 && *output_features > 0 && *input_features > 0 => {}
        [expert_count, 2, output_features, input_features]
            if *expert_count > 0 && *output_features > 0 && *input_features > 0 => {}
        _ => {
            return Err(
                "CUDA Marlin FP8 MoE logical shape must be [E, N, K] or [E, 2, N, K]".to_owned(),
            )
        }
    }
    if logical_element_type != ElementType::F16 {
        return Err(format!(
            "CUDA Marlin FP8 MoE logical element type must be F16, got {logical_element_type:?}"
        ));
    }
    weight
        .validate_logical(bound_logical_dimensions, logical_element_type)
        .map_err(|error| format!("CUDA Marlin FP8 MoE logical contract is invalid: {error}"))?;
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
    } = weight.physical_layout()
    else {
        return Err(
            "CUDA Marlin FP8 MoE requires one whole quantized expert-major layout".to_owned(),
        );
    };
    if zero_points.is_some()
        || zero_point_packed_dimensions.is_some()
        || axis_indices.is_some()
        || permutation.is_some()
        || codebook.is_some()
    {
        return Err(
            "CUDA Marlin FP8 MoE forbids zero-point, index, permutation, and codebook components"
                .to_owned(),
        );
    }
    let last_axis = bound_logical_dimensions.len() - 1;
    if usize::try_from(*group_axis).ok() != Some(last_axis)
        || !matches!(group_padding, PhysicalWeightPadding::Exact)
    {
        return Err(
            "CUDA Marlin FP8 MoE requires exact channelwise grouping on the final input axis"
                .to_owned(),
        );
    }
    if !is_exact_contiguous(&packed_values.storage) || !is_exact_contiguous(&scales.storage) {
        return Err(
            "CUDA Marlin FP8 MoE packed values and scales must use exact contiguous storage"
                .to_owned(),
        );
    }
    if packed_values.component_id == scales.component_id {
        return Err(
            "CUDA Marlin FP8 MoE packed values and scales must have distinct identities".to_owned(),
        );
    }

    let mut component_by_id = BTreeMap::new();
    for component in weight.components() {
        if component_by_id
            .insert(component.component_id().clone(), component)
            .is_some()
        {
            return Err(format!(
                "CUDA Marlin FP8 MoE layout duplicates component `{}`",
                component.component_id()
            ));
        }
    }
    if component_by_id.len() != 2 {
        return Err(
            "CUDA Marlin FP8 MoE layout must contain exactly packed values and scales".to_owned(),
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
        return Err("CUDA Marlin FP8 MoE packed component must be quantized".to_owned());
    };
    quantization
        .validate()
        .map_err(|error| format!("CUDA Marlin FP8 MoE quantization ABI is invalid: {error}"))?;
    if quantization.format_id.as_str() != MARLIN_FP8_QUANTIZATION_FORMAT_ID
        || quantization.bits_per_weight != 8
        || quantization.grouping != QuantizationGrouping::WholeAxis
        || quantization.packing != QuantizationPacking::Tiled
        || quantization.scale_type != ElementType::F16
        || quantization.zero_point_type.is_some()
    {
        return Err(format!(
            "CUDA Marlin FP8 MoE component `{}` is not channelwise E4M3 tiled W8A16",
            packed_component.component_id()
        ));
    }
    if !matches!(
        scales_component.encoding(),
        WeightEncoding::Dense {
            element_type: ElementType::F16
        }
    ) {
        return Err("CUDA Marlin FP8 MoE scales component must be dense F16".to_owned());
    }

    let expected_packed_dimensions = bound_logical_dimensions.to_vec();
    if packed_dimensions != &expected_packed_dimensions
        || packed_component.physical_dimensions() != expected_packed_dimensions
    {
        return Err(format!(
            "CUDA Marlin FP8 MoE packed physical shape must be {expected_packed_dimensions:?}"
        ));
    }
    let mut expected_scales_dimensions = bound_logical_dimensions.to_vec();
    expected_scales_dimensions[last_axis] = 1;
    if scales_component.physical_dimensions() != expected_scales_dimensions {
        return Err(format!(
            "CUDA Marlin FP8 MoE scales physical shape must be {expected_scales_dimensions:?}"
        ));
    }

    let expert_count = bound_logical_dimensions[0];
    let packed_bytes = checked_physical_bytes(&expected_packed_dimensions, 1, "FP8 packed")?;
    let scales_bytes = checked_physical_bytes(
        &expected_scales_dimensions,
        ElementType::F16.size_bytes(),
        "FP8 scales",
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
        return Err(
            "CUDA Marlin FP8 MoE component byte counts differ from the typed ABI".to_owned(),
        );
    }

    Ok(MarlinMoeWeightMetadata {
        packed_component_id: packed_values.component_id.clone(),
        scales_component_id: scales.component_id.clone(),
        logical_dimensions: bound_logical_dimensions.to_vec(),
        packed_physical_dimensions: expected_packed_dimensions,
        scales_physical_dimensions: expected_scales_dimensions,
        expert_count,
        packed_bytes,
        scales_bytes,
        packed_expert_stride_bytes: expert_stride_bytes("FP8 packed", packed_bytes, expert_count)?,
        scales_expert_stride_bytes: expert_stride_bytes("FP8 scales", scales_bytes, expert_count)?,
        group_size: -1,
        weight_type: MarlinMoeF16WeightType::E4M3,
    })
}

fn validate_marlin_thread_tile(
    output_features: u64,
    input_features: u64,
    label: &str,
) -> Result<(), String> {
    let supported = (output_features.is_multiple_of(64) && input_features.is_multiple_of(128))
        || (output_features.is_multiple_of(128) && input_features.is_multiple_of(64));
    if output_features == 0 || input_features == 0 || !supported {
        return Err(format!(
            "{label} shape N={output_features}, K={input_features} does not satisfy a Marlin 64x128 or 128x64 thread tile"
        ));
    }
    Ok(())
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
    use crate::marlin_fp8_materializer::MARLIN_FP8_WEIGHT_FORMAT_ID;
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
                        grouping: QuantizationGrouping::fixed(128),
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
                    zero_point_packed_dimensions: None,
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

    fn valid_matrix_schema() -> WeightSchema {
        let mut schema = valid_schema();
        schema.layout_id = WeightLayoutId::new("weight-layout.test.marlin-matrix").unwrap();
        schema.components[0].dimensions = vec![64, 64];
        schema.components[0].external_names = vec![
            "projection.qweight".to_owned(),
            "projection.qzeros".to_owned(),
            "projection.g_idx".to_owned(),
        ];
        schema.components[1].dimensions = vec![64, 1];
        schema.components[1].external_names = vec!["projection.scales".to_owned()];
        schema.tensors[0].id = id("weight.projection");
        schema.tensors[0].dimensions = vec![64, 128];
        let PhysicalWeightLayout::Quantized {
            packed_dimensions,
            group_axis,
            ..
        } = &mut schema.tensors[0].physical_layout
        else {
            unreachable!();
        };
        *packed_dimensions = vec![64, 64];
        *group_axis = 1;
        schema
    }

    fn valid_fp8_schema(gate_up: bool) -> WeightSchema {
        let packed_id = id("component.fp8_packed");
        let scales_id = id("component.fp8_scales");
        let (dimensions, scales_dimensions, group_axis) = if gate_up {
            (vec![2, 2, 64, 128], vec![2, 2, 64, 1], 3)
        } else {
            (vec![2, 128, 64], vec![2, 128, 1], 2)
        };
        WeightSchema {
            format_id: WeightFormatId::new(MARLIN_FP8_WEIGHT_FORMAT_ID).unwrap(),
            layout_id: WeightLayoutId::new("weight-layout.test.marlin-fp8-moe").unwrap(),
            version: ContractVersion::new(1, 0),
            components: vec![
                WeightComponentSpec {
                    id: packed_id.clone(),
                    role: WeightComponentRole::PackedValues,
                    external_names: vec!["experts.fp8".to_owned()],
                    dimensions: dimensions.clone(),
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
                    external_names: vec!["experts.fp8_scales".to_owned()],
                    dimensions: scales_dimensions,
                    encoding: WeightEncoding::Dense {
                        element_type: ElementType::F16,
                    },
                    required: true,
                },
            ],
            tensors: vec![WeightTensorSpec {
                id: id("weight.fp8_experts"),
                dimensions: dimensions.clone(),
                logical_element_type: ElementType::F16,
                physical_layout: PhysicalWeightLayout::Quantized {
                    packed_values: PhysicalWeightComponentBinding::exact_contiguous(packed_id),
                    packed_dimensions: dimensions,
                    scales: PhysicalWeightComponentBinding::exact_contiguous(scales_id),
                    zero_points: None,
                    zero_point_packed_dimensions: None,
                    axis_indices: None,
                    permutation: None,
                    codebook: None,
                    group_axis,
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

    fn validate_fp8(schema: &WeightSchema) -> Result<MarlinMoeWeightMetadata, String> {
        validate_marlin_fp8_moe_contract(
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
        assert_eq!(metadata.weight_type, MarlinMoeF16WeightType::U4B8);
    }

    #[test]
    fn accepts_whole_expert_major_marlin_fp8_gate_up_and_down_stacks() {
        let gate_up = validate_fp8(&valid_fp8_schema(true)).unwrap();
        assert_eq!(gate_up.logical_dimensions, [2, 2, 64, 128]);
        assert_eq!(gate_up.packed_physical_dimensions, [2, 2, 64, 128]);
        assert_eq!(gate_up.scales_physical_dimensions, [2, 2, 64, 1]);
        assert_eq!(gate_up.expert_count, 2);
        assert_eq!(gate_up.packed_expert_stride_bytes, 2 * 64 * 128);
        assert_eq!(gate_up.scales_expert_stride_bytes, 2 * 64 * 2);
        assert_eq!(gate_up.group_size, -1);
        assert_eq!(gate_up.weight_type, MarlinMoeF16WeightType::E4M3);

        let down = validate_fp8(&valid_fp8_schema(false)).unwrap();
        assert_eq!(down.logical_dimensions, [2, 128, 64]);
        assert_eq!(down.packed_physical_dimensions, [2, 128, 64]);
        assert_eq!(down.scales_physical_dimensions, [2, 128, 1]);
        assert_eq!(down.packed_expert_stride_bytes, 128 * 64);
        assert_eq!(down.scales_expert_stride_bytes, 128 * 2);
        assert_eq!(down.weight_type, MarlinMoeF16WeightType::E4M3);
    }

    #[test]
    fn rejects_marlin_fp8_moe_shape_or_channelwise_contract_drift() {
        let valid = valid_fp8_schema(true);
        let bad_shape = [2, 3, 64, 128];
        let error = validate_marlin_fp8_moe_contract(
            &resolved(&valid),
            &bad_shape,
            ElementType::F16,
            &bad_shape,
        )
        .unwrap_err();
        assert!(error.contains("[E, N, K] or [E, 2, N, K]"), "{error}");

        let mut wrong_format = valid_fp8_schema(false);
        let WeightEncoding::Quantized(spec) = &mut wrong_format.components[0].encoding else {
            unreachable!();
        };
        spec.format_id = QuantizationFormatId::new("quantization.test.fp8-other").unwrap();
        let error = validate_fp8(&wrong_format).unwrap_err();
        assert!(error.contains("not channelwise E4M3"), "{error}");

        let mut fragmented = valid_fp8_schema(false);
        let PhysicalWeightLayout::Quantized { packed_values, .. } =
            &mut fragmented.tensors[0].physical_layout
        else {
            unreachable!();
        };
        packed_values.storage = PhysicalStorageLayout::Strided {
            strides_in_elements: vec![8192, 64, 1],
            padding: PhysicalWeightPadding::Exact,
        };
        let error = validate_fp8(&fragmented).unwrap_err();
        assert!(error.contains("exact contiguous"), "{error}");
    }

    #[test]
    fn accepts_exact_rank_two_projection_contract() {
        let schema = valid_matrix_schema();
        let metadata = validate(&schema).unwrap();
        validate_marlin_thread_tile(64, 128, "test projection").unwrap();
        assert_eq!(metadata.logical_dimensions, [64, 128]);
        assert_eq!(metadata.packed_physical_dimensions, [64, 64]);
        assert_eq!(metadata.scales_physical_dimensions, [64, 1]);
        assert_eq!(metadata.expert_count, 1);
        assert_eq!(metadata.packed_expert_stride_bytes, 4096);
        assert_eq!(metadata.scales_expert_stride_bytes, 128);
    }

    #[test]
    fn rejects_rank_two_projection_outside_marlin_thread_tiles() {
        let error = validate_marlin_thread_tile(48, 128, "test projection").unwrap_err();
        assert!(error.contains("64x128 or 128x64"), "{error}");
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
    fn accepts_gptq_component_abi_inside_a_mixed_execution_schema() {
        let mut schema = valid_schema();
        schema.format_id = WeightFormatId::new(MARLIN_FP8_WEIGHT_FORMAT_ID).unwrap();
        let metadata = validate(&schema).unwrap();
        assert_eq!(metadata.group_size, 128);
        assert_eq!(metadata.logical_dimensions, [2, 64, 128]);
    }

    #[test]
    fn rejects_another_quantization_format() {
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
