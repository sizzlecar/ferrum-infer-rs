//! Invocation-scoped dequantization for selected native projections.
//! The planner owns all backing; no materialized weights survive as model state.

use super::*;
use ferrum_interfaces::vnext::{
    PhysicalStorageLayout, PhysicalWeightLayout, ResolvedValueBinding, ResolvedWeightBinding,
};

// Reserve staging for larger matrices to limit dispatch/global-write overhead.
// This total-matrix work gate is independent of either logical matrix axis.
const MIN_STAGED_WEIGHT_ELEMENTS: u64 = 1 << 20;

/// Projection owners select their measured or candidate route explicitly;
/// adding a format for one operation must not enable it for another.
#[derive(Debug, Clone, Copy)]
pub(in super::super) enum StagingPolicy {
    SwiGlu,
    GatedDelta,
}

impl StagingPolicy {
    fn minimum_rows(self, format: LinearPhysicalFormat) -> Option<u32> {
        match (self, format) {
            (Self::SwiGlu, LinearPhysicalFormat::Q4K) => Some(768),
            (Self::SwiGlu, LinearPhysicalFormat::Q6K) => Some(256),
            // Candidate large-prefill policy; the recurrent core is unchanged.
            (Self::GatedDelta, LinearPhysicalFormat::Q4K | LinearPhysicalFormat::Q5K) => Some(768),
            _ => None,
        }
    }
}

fn leaf_format(
    weight: &ResolvedWeightBinding,
    layout: &PhysicalWeightLayout,
    logical_dimensions: &[u64],
    expected_axis: u32,
) -> Option<LinearPhysicalFormat> {
    let format = quantized_leaf_format(weight, layout, logical_dimensions, expected_axis)?;
    StagingPolicy::SwiGlu.minimum_rows(format).map(|_| format)
}

fn quantized_leaf_format(
    weight: &ResolvedWeightBinding,
    layout: &PhysicalWeightLayout,
    logical_dimensions: &[u64],
    expected_axis: u32,
) -> Option<LinearPhysicalFormat> {
    let PhysicalWeightLayout::BlockQuantized {
        blocks,
        block_axis,
        block_padding,
    } = layout
    else {
        return None;
    };
    if *block_axis != expected_axis || *block_padding != PhysicalWeightPadding::Exact {
        return None;
    }
    let component = weight
        .components()
        .iter()
        .find(|component| component.component_id() == &blocks.component_id)?;
    let WeightEncoding::BlockQuantized(spec) = component.encoding() else {
        return None;
    };
    // A composite can add a singleton axis to a native matrix without moving
    // its blocks. That shape-only reshape is explicitly Strided, although the
    // matrix still has exactly the row-major storage consumed by staging.
    let mut block_dimensions = logical_dimensions.to_vec();
    let axis = block_dimensions.get_mut(expected_axis as usize)?;
    let block_width = u64::from(spec.logical_values_per_block);
    if block_width == 0 || !axis.is_multiple_of(block_width) {
        return None;
    }
    *axis /= block_width;
    if !exact_row_major(&blocks.storage, &block_dimensions) {
        return None;
    }
    match GgufBlockFormat::from_spec(spec).ok()? {
        GgufBlockFormat::Q4K => Some(LinearPhysicalFormat::Q4K),
        GgufBlockFormat::Q5K => Some(LinearPhysicalFormat::Q5K),
        GgufBlockFormat::Q6K => Some(LinearPhysicalFormat::Q6K),
        GgufBlockFormat::Q8_0 => Some(LinearPhysicalFormat::Q8_0),
        _ => None,
    }
}

fn exact_row_major(storage: &PhysicalStorageLayout, dimensions: &[u64]) -> bool {
    match storage {
        PhysicalStorageLayout::Contiguous {
            padding: PhysicalWeightPadding::Exact,
        } => true,
        PhysicalStorageLayout::Strided {
            strides_in_elements,
            padding: PhysicalWeightPadding::Exact,
        } if strides_in_elements.len() == dimensions.len() => {
            let mut expected_stride = 1_u64;
            for (&extent, &stride) in dimensions.iter().zip(strides_in_elements).rev() {
                // A singleton axis is never advanced, so its stride cannot
                // introduce either a hole or a permutation into the matrix.
                if extent == 0 || (extent > 1 && stride != expected_stride) {
                    return false;
                }
                let Some(next) = expected_stride.checked_mul(extent) else {
                    return false;
                };
                expected_stride = next;
            }
            true
        }
        _ => false,
    }
}

/// One logical projection, reused for gate, up, and down. The affine estimator
/// conservatively reserves these fixed bytes even below the staging threshold.
pub(super) fn workspace_bytes(
    bindings: &[ResolvedValueBinding],
    hidden: u64,
    intermediate: u64,
) -> Result<u64, String> {
    let gate_up = binding(bindings, ResolvedValueRole::Input, 1)?;
    let down = binding(bindings, ResolvedValueRole::Input, 2)?;
    let (Some(gate_weight), Some(down_weight)) = (gate_up.weight(), down.weight()) else {
        return Ok(0);
    };
    if gate_up.tensor().element_type() != ElementType::F16
        || down.tensor().element_type() != ElementType::F16
    {
        return Ok(0);
    }
    weight_workspace_bytes(gate_weight, down_weight, hidden, intermediate)
}

fn weight_workspace_bytes(
    gate_weight: &ResolvedWeightBinding,
    down_weight: &ResolvedWeightBinding,
    hidden: u64,
    intermediate: u64,
) -> Result<u64, String> {
    let PhysicalWeightLayout::Composite { parts } = gate_weight.physical_layout() else {
        return Ok(0);
    };
    if parts.len() != 2
        || ![0, 1].into_iter().all(|part_index| {
            parts.iter().any(|part| {
                part.logical_offsets == [part_index, 0, 0]
                    && part.extents == [1, intermediate, hidden]
                    && leaf_format(gate_weight, &part.layout, &part.extents, 2).is_some()
            })
        })
        || leaf_format(
            down_weight,
            down_weight.physical_layout(),
            &[hidden, intermediate],
            1,
        )
        .is_none()
    {
        return Ok(0);
    }
    let elements = hidden
        .checked_mul(intermediate)
        .ok_or_else(|| "Metal staged SwiGLU matrix size overflows".to_owned())?;
    if elements < MIN_STAGED_WEIGHT_ELEMENTS {
        return Ok(0);
    }
    elements
        .checked_mul(ElementType::F16.size_bytes())
        .ok_or_else(|| "Metal staged SwiGLU workspace size overflows".to_owned())
}

/// One shared expansion for the largest eligible leaf of a row-partitioned
/// projection. Fixed workspace is conservatively declared even for small waves;
/// runtime admission, rather than a model-specific byte cap, owns its backing.
pub(in super::super) fn partitioned_workspace_bytes(
    value: &ResolvedValueBinding,
    out_features: u64,
    in_features: u64,
    policy: StagingPolicy,
) -> Result<u64, String> {
    if value.tensor().element_type() != ElementType::F16
        || value.tensor().dimensions() != [out_features, in_features]
    {
        return Ok(0);
    }
    let Some(weight) = value.weight() else {
        return Ok(0);
    };
    let PhysicalWeightLayout::Composite { parts } = weight.physical_layout() else {
        return Ok(0);
    };
    let mut ordered = parts.iter().collect::<Vec<_>>();
    ordered.sort_by(|left, right| left.logical_offsets.cmp(&right.logical_offsets));
    let mut next_row = 0_u64;
    let mut largest = 0_u64;
    for part in ordered {
        if part.logical_offsets != [next_row, 0]
            || part.extents.len() != 2
            || part.extents[0] == 0
            || part.extents[1] != in_features
        {
            return Ok(0);
        }
        next_row = next_row
            .checked_add(part.extents[0])
            .ok_or_else(|| "Metal staged projection row extent overflows".to_owned())?;
        let Some(format) = quantized_leaf_format(weight, &part.layout, &part.extents, 1) else {
            // Launches retain the quantization format, not physical strides.
            // Keep the whole projection on its existing route if any leaf's
            // row-major storage is unproven; another eligible leaf's scratch
            // must not accidentally enable staging for that unknown layout.
            return Ok(0);
        };
        if policy.minimum_rows(format).is_none() {
            continue;
        }
        let elements = in_features
            .checked_mul(part.extents[0])
            .ok_or_else(|| "Metal staged projection matrix size overflows".to_owned())?;
        if elements < MIN_STAGED_WEIGHT_ELEMENTS || elements / 16 > u64::from(u32::MAX) {
            continue;
        }
        largest = largest.max(
            elements
                .checked_mul(ElementType::F16.size_bytes())
                .ok_or_else(|| "Metal staged projection workspace size overflows".to_owned())?,
        );
    }
    Ok(if next_row == out_features { largest } else { 0 })
}

pub(super) fn selected(launch: LinearLaunch) -> bool {
    selected_for(launch, StagingPolicy::SwiGlu)
}

fn selected_for(launch: LinearLaunch, policy: StagingPolicy) -> bool {
    let Some(minimum_rows) = policy.minimum_rows(launch.format) else {
        return false;
    };
    let elements = u64::from(launch.params.in_features) * u64::from(launch.params.out_features);
    launch.activation_type == ElementType::F16
        && launch.params.rows >= minimum_rows
        && launch.params.in_features.is_multiple_of(256)
        && elements >= MIN_STAGED_WEIGHT_ELEMENTS
        // The stage kernel indexes one 16-coefficient tile with a uint.
        && elements / 16 <= u64::from(u32::MAX)
}

#[derive(Debug, Clone, Copy)]
pub(in super::super) struct Workspace {
    region: usize,
    offset: u64,
    policy: StagingPolicy,
}

pub(super) struct Sequence {
    pub(super) gate_up: Vec<LinearLaunch>,
    pub(super) down: LinearLaunch,
    pub(super) activation: SwiGluLaunch,
    pub(super) scratch_region: usize,
    pub(super) workspace: Option<Workspace>,
}

impl Sequence {
    pub(super) fn dispatch_count(&self) -> u64 {
        let staged = if self.workspace.is_some() {
            self.gate_up
                .iter()
                .copied()
                .chain([self.down])
                .filter(|launch| selected(*launch))
                .count() as u64
        } else {
            0
        };
        self.gate_up.len() as u64 + 2 + staged
    }

    pub(super) fn encode(
        &self,
        pipelines: &MetalLinearPipelines,
        encoder: &ComputeCommandEncoderRef,
        regions: &[MetalBufferRegion],
    ) {
        for launch in &self.gate_up {
            dispatch(pipelines, encoder, regions, *launch, self.workspace);
        }
        dispatch_swiglu(
            pipelines,
            encoder,
            &regions[self.scratch_region],
            self.activation,
        );
        dispatch(pipelines, encoder, regions, self.down, self.workspace);
    }
}

impl Workspace {
    pub(super) fn new(
        regions: &[MetalBufferRegion],
        region: usize,
        offset: u64,
        bytes: u64,
        launches: impl IntoIterator<Item = LinearLaunch>,
    ) -> Result<Option<Self>, String> {
        Self::with_policy(
            regions,
            region,
            offset,
            bytes,
            launches,
            StagingPolicy::SwiGlu,
        )
    }

    pub(in super::super) fn with_policy(
        regions: &[MetalBufferRegion],
        region: usize,
        offset: u64,
        bytes: u64,
        launches: impl IntoIterator<Item = LinearLaunch>,
        policy: StagingPolicy,
    ) -> Result<Option<Self>, String> {
        if bytes == 0 {
            return Ok(None);
        }
        let scratch = regions
            .get(region)
            .ok_or_else(|| "Metal staged SwiGLU scratch region is absent".to_owned())?;
        if scratch.element_type() != ElementType::U8 {
            return Err("Metal staged SwiGLU requires declared byte workspace".to_owned());
        }
        validate_region_span(scratch, offset, bytes, "Metal staged SwiGLU scratch")?;
        if !offset.is_multiple_of(ElementType::F16.size_bytes()) {
            return Err("Metal staged SwiGLU scratch is not F16 aligned".to_owned());
        }
        for launch in launches {
            if !selected_for(launch, policy) {
                continue;
            }
            let elements =
                u64::from(launch.params.in_features) * u64::from(launch.params.out_features);
            if elements * 2 > bytes {
                return Err("Metal staged SwiGLU projection exceeds its workspace".to_owned());
            }
            let format = match launch.format {
                LinearPhysicalFormat::Q4K => GgufBlockFormat::Q4K,
                LinearPhysicalFormat::Q5K => GgufBlockFormat::Q5K,
                LinearPhysicalFormat::Q6K => GgufBlockFormat::Q6K,
                _ => unreachable!("selected staging format"),
            };
            validate_region_span(
                &regions[launch.weight_region],
                0,
                elements / 256 * format.block_bytes() as u64,
                "Metal staged SwiGLU quantized weight",
            )?;
            let rows = u64::from(launch.params.rows);
            if (launch.input_region == region
                && launch.input_offset_bytes + rows * u64::from(launch.params.in_features) * 2
                    > offset)
                || (launch.output_region == region
                    && launch.output_offset_bytes
                        + rows * u64::from(launch.params.output_stride) * 2
                        > offset)
            {
                return Err("Metal staged SwiGLU workspace overlaps live activations".to_owned());
            }
        }
        Ok(Some(Self {
            region,
            offset,
            policy,
        }))
    }
}

pub(in super::super) fn dispatch_count(launch: LinearLaunch, workspace: Option<Workspace>) -> u64 {
    1 + u64::from(workspace.is_some_and(|workspace| selected_for(launch, workspace.policy)))
}

pub(in super::super) fn dispatch(
    pipelines: &MetalLinearPipelines,
    encoder: &ComputeCommandEncoderRef,
    regions: &[MetalBufferRegion],
    launch: LinearLaunch,
    workspace: Option<Workspace>,
) {
    let Some(workspace) = workspace.filter(|workspace| selected_for(launch, workspace.policy))
    else {
        return dispatch_linear(pipelines, encoder, regions, launch);
    };
    let blocks =
        (u64::from(launch.params.in_features) * u64::from(launch.params.out_features) / 256) as u32;
    encoder.set_compute_pipeline_state(match launch.format {
        LinearPhysicalFormat::Q4K => &pipelines.k_quant_gemm.stage_q4_k,
        LinearPhysicalFormat::Q5K => &pipelines.k_quant_gemm.stage_q5_k,
        LinearPhysicalFormat::Q6K => &pipelines.k_quant_gemm.stage_q6_k,
        _ => unreachable!("selected staging format"),
    });
    encoder.set_threadgroup_memory_length(0, 0);
    set_region_offset(encoder, 0, &regions[launch.weight_region], 0);
    set_region_offset(encoder, 1, &regions[workspace.region], workspace.offset);
    encoder.set_bytes(2, 4, (&blocks as *const u32).cast());
    encoder.dispatch_thread_groups(
        MTLSize::new(u64::from(blocks).div_ceil(8), 1, 1),
        MTLSize::new(128, 1, 1),
    );
    // Sequential tracked-resource dispatches preserve stage -> GEMM -> reuse
    // ordering within the provider's existing compute encoder.
    encoder.set_compute_pipeline_state(&pipelines.k_quant_gemm.staged_f16);
    set_region_offset(
        encoder,
        0,
        &regions[launch.input_region],
        launch.input_offset_bytes,
    );
    set_region_offset(encoder, 1, &regions[workspace.region], workspace.offset);
    set_region_offset(
        encoder,
        2,
        &regions[launch.output_region],
        launch.output_offset_bytes,
    );
    bind_linear_params(encoder, launch.params, launch.format, ElementType::F16);
    dispatch_linear_grid(encoder, launch.params, LinearDispatchKind::TiledGemm);
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod gated_delta_tests;
