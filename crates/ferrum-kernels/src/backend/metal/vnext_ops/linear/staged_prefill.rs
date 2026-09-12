//! Invocation-scoped dequantization for independent SwiGLU projections.
//! The planner owns all backing; no materialized weights survive as model state.

use super::*;
use ferrum_interfaces::vnext::{
    PhysicalStorageLayout, PhysicalWeightLayout, ResolvedValueBinding, ResolvedWeightBinding,
};

// Reserve staging for larger matrices to limit dispatch/global-write overhead.
// This total-matrix work gate is independent of either logical matrix axis.
const MIN_STAGED_WEIGHT_ELEMENTS: u64 = 1 << 20;

fn leaf_format(
    weight: &ResolvedWeightBinding,
    layout: &PhysicalWeightLayout,
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
    if *block_axis != expected_axis
        || *block_padding != PhysicalWeightPadding::Exact
        || blocks.storage != PhysicalStorageLayout::exact_contiguous()
    {
        return None;
    }
    let component = weight
        .components()
        .iter()
        .find(|component| component.component_id() == &blocks.component_id)?;
    let WeightEncoding::BlockQuantized(spec) = component.encoding() else {
        return None;
    };
    match GgufBlockFormat::from_spec(spec).ok()? {
        GgufBlockFormat::Q4K => Some(LinearPhysicalFormat::Q4K),
        GgufBlockFormat::Q6K => Some(LinearPhysicalFormat::Q6K),
        _ => None,
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
                    && leaf_format(gate_weight, &part.layout, 2).is_some()
            })
        })
        || leaf_format(down_weight, down_weight.physical_layout(), 1).is_none()
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

pub(super) fn selected(launch: LinearLaunch) -> bool {
    let minimum_rows = match launch.format {
        LinearPhysicalFormat::Q4K => 768,
        LinearPhysicalFormat::Q6K => 256,
        _ => return false,
    };
    let elements = u64::from(launch.params.in_features) * u64::from(launch.params.out_features);
    launch.activation_type == ElementType::F16
        && launch.params.rows >= minimum_rows
        && launch.params.in_features.is_multiple_of(256)
        && elements >= MIN_STAGED_WEIGHT_ELEMENTS
        // The stage kernel indexes one 16-coefficient tile with a uint.
        && elements / 16 <= u64::from(u32::MAX)
}

#[derive(Clone, Copy)]
pub(super) struct Workspace {
    region: usize,
    offset: u64,
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
            if !selected(launch) {
                continue;
            }
            let elements =
                u64::from(launch.params.in_features) * u64::from(launch.params.out_features);
            if elements * 2 > bytes {
                return Err("Metal staged SwiGLU projection exceeds its workspace".to_owned());
            }
            let format = match launch.format {
                LinearPhysicalFormat::Q4K => GgufBlockFormat::Q4K,
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
        Ok(Some(Self { region, offset }))
    }
}

pub(super) fn dispatch(
    pipelines: &MetalLinearPipelines,
    encoder: &ComputeCommandEncoderRef,
    regions: &[MetalBufferRegion],
    launch: LinearLaunch,
    workspace: Option<Workspace>,
) {
    let Some(workspace) = workspace.filter(|_| selected(launch)) else {
        return dispatch_linear(pipelines, encoder, regions, launch);
    };
    let blocks =
        (u64::from(launch.params.in_features) * u64::from(launch.params.out_features) / 256) as u32;
    encoder.set_compute_pipeline_state(match launch.format {
        LinearPhysicalFormat::Q4K => &pipelines.k_quant_gemm.stage_q4_k,
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
