//! Bounded, normalized Sylvester transforms over retained activation regions.

use std::ffi::c_void;

use ferrum_interfaces::vnext::{
    DynamicStorageRequirement, ElementType, OperationProviderDescriptor, OperationResourceEstimate,
    OperationResourceEstimateRequest, PhysicalWeightLayout, ProviderWorkspaceRequirement,
    ProviderWorkspaceReusePolicy, ProviderWorkspaceScope, ProviderWorkspaceSizeFormula,
    ResolvedValueBinding, VNextError,
};
use metal::{
    BufferRef, CompileOptions, ComputeCommandEncoderRef, ComputePipelineState, Device, MTLSize,
};

use super::super::vnext_runtime::{MetalBufferRegion, MetalDeviceRuntimeError};

pub(super) const FINGERPRINT_SOURCE: &str =
    concat!(include_str!("hadamard.rs"), include_str!("hadamard.metal"),);

pub(super) fn workspace_bytes_per_token(bindings: &[ResolvedValueBinding]) -> Result<u64, String> {
    fn width(layout: &PhysicalWeightLayout, dimensions: &[u64]) -> Result<u64, String> {
        match layout {
            PhysicalWeightLayout::Hadamard { transform, .. } => {
                let width = dimensions
                    .last()
                    .copied()
                    .ok_or_else(|| "Metal Hadamard weight has no feature axis".to_owned())?;
                transform
                    .validate(width)
                    .map_err(|error| error.to_string())?;
                Ok(width)
            }
            PhysicalWeightLayout::Composite { parts } => {
                parts.iter().try_fold(0_u64, |maximum, part| {
                    Ok(maximum.max(width(&part.layout, &part.extents)?))
                })
            }
            _ => Ok(0),
        }
    }
    bindings
        .iter()
        .filter_map(|binding| binding.weight().map(|weight| (binding, weight)))
        .try_fold(0_u64, |maximum, (binding, weight)| {
            let bytes = width(weight.physical_layout(), binding.tensor().dimensions())?
                .checked_mul(ElementType::F32.size_bytes())
                .and_then(|value| value.checked_add(15))
                .map(|value| value & !15)
                .ok_or_else(|| "Metal Hadamard workspace row size overflows".to_owned())?;
            Ok(maximum.max(bytes))
        })
}

pub(super) fn estimate_token_workspace(
    descriptor: &OperationProviderDescriptor,
    request: &OperationResourceEstimateRequest<'_>,
    operation_id: &str,
) -> Result<OperationResourceEstimate, VNextError> {
    let baseline = super::estimate_without_workspace(descriptor, request, operation_id)?;
    let bytes = workspace_bytes_per_token(request.values()).map_err(super::invalid_plan)?;
    if bytes == 0 {
        return Ok(baseline);
    }
    let workspace = ProviderWorkspaceRequirement::from_formula(
        ProviderWorkspaceSizeFormula::affine(0, 0, bytes)?,
        super::VALUE_ALIGNMENT_BYTES,
        ProviderWorkspaceScope::Invocation,
        ProviderWorkspaceReusePolicy::OverwriteBeforeRead,
        DynamicStorageRequirement::contiguous(),
    )?;
    Ok(OperationResourceEstimate::new(
        descriptor.resource_estimator_id(),
        descriptor.resource_estimator_version(),
        descriptor.resource_estimator_implementation_fingerprint(),
        request.input_fingerprint(),
        super::VALUE_ALIGNMENT_BYTES,
        Some(workspace),
        None,
    ))
}

pub(super) fn append_workspace(
    required: &mut u64,
    bindings: &[ResolvedValueBinding],
    tokens: u64,
) -> Result<u64, String> {
    let bytes = workspace_bytes_per_token(bindings)?;
    let offset = if bytes == 0 {
        *required
    } else {
        required
            .checked_add(15)
            .map(|bytes| bytes & !15)
            .ok_or_else(|| "Metal Hadamard workspace alignment overflows".to_owned())?
    };
    *required = bytes
        .checked_mul(tokens)
        .and_then(|bytes| offset.checked_add(bytes))
        .ok_or_else(|| "Metal Hadamard workspace size overflows".to_owned())?;
    Ok(offset)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct GroupedFeatureTranspose {
    pub(super) inner_extent: u32,
    pub(super) first_outer_extent: u32,
    pub(super) second_outer_extent: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct HadamardTransform {
    pub(super) block_size: u32,
    pub(super) signs_region: Option<usize>,
    pub(super) inverse: bool,
    pub(super) permutation: Option<GroupedFeatureTranspose>,
}

impl HadamardTransform {
    pub(super) fn validate(self, width: u64) -> Result<(), String> {
        if !self.block_size.is_power_of_two()
            || width == 0
            || !width.is_multiple_of(u64::from(self.block_size))
        {
            return Err("Metal Hadamard requires complete power-of-two blocks".to_owned());
        }
        if let Some(permutation) = self.permutation {
            let extent = u64::from(permutation.inner_extent)
                .checked_mul(u64::from(permutation.first_outer_extent))
                .and_then(|value| value.checked_mul(u64::from(permutation.second_outer_extent)));
            if self.inverse
                || permutation.inner_extent == 0
                || permutation.first_outer_extent == 0
                || permutation.second_outer_extent == 0
                || extent != Some(width)
            {
                return Err(
                    "Metal Hadamard grouped permutation differs from its input width".to_owned(),
                );
            }
        }
        Ok(())
    }

    pub(super) fn relocate(self, region_base: usize) -> Result<Self, String> {
        Ok(Self {
            signs_region: self
                .signs_region
                .map(|index| {
                    index
                        .checked_add(region_base)
                        .ok_or_else(|| "Metal Hadamard signs region index overflows".to_owned())
                })
                .transpose()?,
            ..self
        })
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct HadamardParams {
    rows: u32,
    width: u32,
    block_size: u32,
    input_stride: u32,
    output_stride: u32,
    has_signs: u32,
    inverse: u32,
    inner_extent: u32,
    first_outer_extent: u32,
    second_outer_extent: u32,
}

pub(super) struct MetalHadamardPipelines {
    f16_f32: ComputePipelineState,
    f32_f32: ComputePipelineState,
    f32_f16: ComputePipelineState,
    specialized_1024: Option<Hadamard1024Pipelines>,
    maximum_threadgroup_bytes: u64,
}

struct Hadamard1024Pipelines {
    f16_f32: ComputePipelineState,
    f32_f32: ComputePipelineState,
    f32_f16: ComputePipelineState,
}

fn supports_hadamard_1024(
    simd_width: u64,
    pipeline_threads: u64,
    static_bytes: u64,
    device_threads: u64,
    device_bytes: u64,
) -> bool {
    simd_width == 32
        && pipeline_threads >= 256
        && device_threads >= 256
        && static_bytes
            .checked_add(4096)
            .is_some_and(|required| required <= device_bytes)
}

impl MetalHadamardPipelines {
    pub(super) fn new(device: &Device) -> Result<Self, MetalDeviceRuntimeError> {
        let options = CompileOptions::new();
        options.set_fast_math_enabled(false);
        let library = device
            .new_library_with_source(include_str!("hadamard.metal"), &options)
            .map_err(|error| {
                MetalDeviceRuntimeError::contract(format!("compile Metal Hadamard: {error}"))
            })?;
        let pipeline = |name| {
            let function = library
                .get_function(name, None)
                .map_err(MetalDeviceRuntimeError::contract)?;
            device
                .new_compute_pipeline_state_with_function(&function)
                .map_err(MetalDeviceRuntimeError::contract)
        };
        let specialized_pipeline = |name| {
            pipeline(name).ok().filter(|pipeline| {
                supports_hadamard_1024(
                    pipeline.thread_execution_width(),
                    pipeline.max_total_threads_per_threadgroup(),
                    pipeline.static_threadgroup_memory_length(),
                    device.max_threads_per_threadgroup().width,
                    device.max_threadgroup_memory_length(),
                )
            })
        };
        let specialized_1024 = (|| {
            Some(Hadamard1024Pipelines {
                f16_f32: specialized_pipeline("vnext_hadamard_1024_f16_f32")?,
                f32_f32: specialized_pipeline("vnext_hadamard_1024_f32_f32")?,
                f32_f16: specialized_pipeline("vnext_hadamard_1024_f32_f16")?,
            })
        })();
        Ok(Self {
            f16_f32: pipeline("vnext_hadamard_f16_f32")?,
            f32_f32: pipeline("vnext_hadamard_f32_f32")?,
            f32_f16: pipeline("vnext_hadamard_f32_f16")?,
            specialized_1024,
            maximum_threadgroup_bytes: device.max_threadgroup_memory_length(),
        })
    }

    fn pipeline(
        &self,
        input: ElementType,
        output: ElementType,
    ) -> Result<&ComputePipelineState, String> {
        match (input, output) {
            (ElementType::F16, ElementType::F32) => Ok(&self.f16_f32),
            (ElementType::F32, ElementType::F32) => Ok(&self.f32_f32),
            (ElementType::F32, ElementType::F16) => Ok(&self.f32_f16),
            _ => Err("Metal Hadamard requires F32 intermediate values".to_owned()),
        }
    }

    fn selected_pipeline(
        &self,
        transform: HadamardTransform,
        input: ElementType,
        output: ElementType,
    ) -> Result<&ComputePipelineState, String> {
        if transform.block_size == 1024 {
            if let Some(pipelines) = &self.specialized_1024 {
                return match (input, output) {
                    (ElementType::F16, ElementType::F32) => Ok(&pipelines.f16_f32),
                    (ElementType::F32, ElementType::F32) => Ok(&pipelines.f32_f32),
                    (ElementType::F32, ElementType::F16) => Ok(&pipelines.f32_f16),
                    _ => Err("Metal Hadamard requires F32 intermediate values".to_owned()),
                };
            }
        }
        self.pipeline(input, output)
    }

    pub(super) fn validate_dispatch(
        &self,
        transform: HadamardTransform,
        width: u32,
        input: ElementType,
        output: ElementType,
    ) -> Result<(), String> {
        transform.validate(u64::from(width))?;
        if (!transform.inverse && output != ElementType::F32)
            || (transform.inverse && input != ElementType::F32)
        {
            return Err("Metal Hadamard cannot narrow its intermediate transform".to_owned());
        }
        let pipeline = self.selected_pipeline(transform, input, output)?;
        let dynamic_bytes = (u64::from(transform.block_size) * 4).next_multiple_of(16);
        let required = dynamic_bytes + pipeline.static_threadgroup_memory_length();
        if required > self.maximum_threadgroup_bytes
            || pipeline.max_total_threads_per_threadgroup() == 0
        {
            return Err("Metal Hadamard block exceeds device threadgroup limits".to_owned());
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn dispatch(
        &self,
        encoder: &ComputeCommandEncoderRef,
        transform: HadamardTransform,
        input: &MetalBufferRegion,
        input_offset: u64,
        input_type: ElementType,
        output: &MetalBufferRegion,
        output_offset: u64,
        output_type: ElementType,
        regions: &[MetalBufferRegion],
        rows: u32,
        width: u32,
    ) {
        let signs = transform.signs_region.map(|index| (&regions[index], 0_u64));
        self.dispatch_raw(
            encoder,
            transform,
            input.buffer(),
            input.offset_bytes() + input_offset,
            input_type,
            output.buffer(),
            output.offset_bytes() + output_offset,
            output_type,
            signs.map(|(region, offset)| (region.buffer(), region.offset_bytes() + offset)),
            rows,
            width,
            width,
            width,
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn dispatch_raw(
        &self,
        encoder: &ComputeCommandEncoderRef,
        transform: HadamardTransform,
        input: &BufferRef,
        input_offset: u64,
        input_type: ElementType,
        output: &BufferRef,
        output_offset: u64,
        output_type: ElementType,
        signs: Option<(&BufferRef, u64)>,
        rows: u32,
        width: u32,
        input_stride: u32,
        output_stride: u32,
    ) {
        let pipeline = self
            .selected_pipeline(transform, input_type, output_type)
            .expect("validated Metal Hadamard types");
        let permutation = transform.permutation.unwrap_or(GroupedFeatureTranspose {
            inner_extent: 0,
            first_outer_extent: 0,
            second_outer_extent: 0,
        });
        let params = HadamardParams {
            rows,
            width,
            block_size: transform.block_size,
            input_stride,
            output_stride,
            has_signs: u32::from(signs.is_some()),
            inverse: u32::from(transform.inverse),
            inner_extent: permutation.inner_extent,
            first_outer_extent: permutation.first_outer_extent,
            second_outer_extent: permutation.second_outer_extent,
        };
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(input), input_offset);
        let (signs_buffer, signs_offset) = signs.unwrap_or((input, input_offset));
        encoder.set_buffer(1, Some(signs_buffer), signs_offset);
        encoder.set_buffer(2, Some(output), output_offset);
        encoder.set_bytes(
            3,
            std::mem::size_of::<HadamardParams>() as u64,
            &params as *const _ as *const c_void,
        );
        encoder.set_threadgroup_memory_length(
            0,
            (u64::from(transform.block_size) * 4).next_multiple_of(16),
        );
        encoder.dispatch_thread_groups(
            MTLSize::new(u64::from(width / transform.block_size), u64::from(rows), 1),
            MTLSize::new(pipeline.max_total_threads_per_threadgroup().min(256), 1, 1),
        );
    }
}

#[cfg(test)]
mod tests;
