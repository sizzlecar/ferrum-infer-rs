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
    fn minimum_rows(self, format: LinearPhysicalFormat, input_features: u64) -> Option<u32> {
        match (self, format) {
            // Short prefills amortize staging once there are at least two
            // blocks per output. Retain Q4's prior threshold for narrower K;
            // Q5 has no measured staged route at that width.
            (Self::SwiGlu, LinearPhysicalFormat::Q4K | LinearPhysicalFormat::Q5K)
                if input_features >= 512 =>
            {
                Some(256)
            }
            (Self::SwiGlu, LinearPhysicalFormat::Q4K) => Some(768),
            (Self::SwiGlu, LinearPhysicalFormat::Q6K) => Some(256),
            // Candidate large-prefill policy; the recurrent core is unchanged.
            (Self::GatedDelta, LinearPhysicalFormat::Q4K | LinearPhysicalFormat::Q5K) => Some(768),
            _ => None,
        }
    }
}

fn quantized_leaf_format(
    weight: &ResolvedWeightBinding,
    layout: &PhysicalWeightLayout,
    logical_dimensions: &[u64],
    expected_axis: u32,
) -> Option<LinearPhysicalFormat> {
    // Preserve the existing partitioned GatedDelta cohort. SwiGLU separately
    // proves all native leaves before admitting any eligible projection.
    let format = leaf_format(weight, layout, logical_dimensions, expected_axis)?;
    (!matches!(format, LinearPhysicalFormat::Native(_))).then_some(format)
}

fn leaf_format(
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
        native => Some(LinearPhysicalFormat::Native(native)),
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
    if parts.len() != 2 {
        return Ok(0);
    }
    let mut eligible = false;
    for part_index in [0, 1] {
        let Some(part) = parts.iter().find(|part| {
            part.logical_offsets == [part_index, 0, 0] && part.extents == [1, intermediate, hidden]
        }) else {
            return Ok(0);
        };
        let Some(format) = leaf_format(gate_weight, &part.layout, &part.extents, 2) else {
            // Launches do not retain physical strides. Prove every leaf's
            // layout before another leaf's scratch can enable staging.
            return Ok(0);
        };
        eligible |= StagingPolicy::SwiGlu.minimum_rows(format, hidden).is_some();
    }
    let Some(down_format) = leaf_format(
        down_weight,
        down_weight.physical_layout(),
        &[hidden, intermediate],
        1,
    ) else {
        return Ok(0);
    };
    eligible |= StagingPolicy::SwiGlu
        .minimum_rows(down_format, intermediate)
        .is_some();
    if !eligible {
        return Ok(0);
    }
    let elements = hidden
        .checked_mul(intermediate)
        .ok_or_else(|| "Metal staged SwiGLU matrix size overflows".to_owned())?;
    if elements < MIN_STAGED_WEIGHT_ELEMENTS || elements / 16 > u64::from(u32::MAX) {
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
        if policy.minimum_rows(format, in_features).is_none() {
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

#[cfg(test)]
fn selected(launch: LinearLaunch) -> bool {
    selected_for(launch, StagingPolicy::SwiGlu)
}

fn selected_for(launch: LinearLaunch, policy: StagingPolicy) -> bool {
    if launch.transform.is_some() {
        return false;
    }
    let Some(minimum_rows) =
        policy.minimum_rows(launch.format, u64::from(launch.params.in_features))
    else {
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

// Only adjacent projections may share their invocation-local transformed input.
// Keep dispatch accounting and encoding on the same checked reuse decision.
pub(in super::super) struct ProjectionStep {
    launch: LinearLaunch,
    reuse_hadamard: bool,
}

impl ProjectionStep {
    pub(in super::super) fn dispatch_count(&self, workspace: Option<Workspace>) -> u64 {
        dispatch_count(self.launch, workspace) - u64::from(self.reuse_hadamard)
    }

    pub(in super::super) fn encode(
        &self,
        pipelines: &MetalLinearPipelines,
        encoder: &ComputeCommandEncoderRef,
        regions: &[MetalBufferRegion],
        workspace: Option<Workspace>,
    ) {
        if self.reuse_hadamard {
            dispatch_transformed_linear(pipelines, encoder, regions, self.launch);
        } else {
            dispatch(pipelines, encoder, regions, self.launch, workspace);
        }
    }
}

pub(in super::super) fn projection_steps<'a>(
    launches: &'a [LinearLaunch],
    regions: &'a [MetalBufferRegion],
) -> impl Iterator<Item = ProjectionStep> + 'a {
    launches.iter().enumerate().map(move |(index, &launch)| {
        let reuse_hadamard = index
            .checked_sub(1)
            .is_some_and(|previous| can_reuse_hadamard(launches[previous], launch, regions));
        ProjectionStep {
            launch,
            reuse_hadamard,
        }
    })
}

impl Sequence {
    pub(super) fn dispatch_count(&self, regions: &[MetalBufferRegion]) -> u64 {
        projection_steps(&self.gate_up, regions)
            .map(|step| step.dispatch_count(self.workspace))
            .sum::<u64>()
            + dispatch_count(self.down, self.workspace)
            + 1
    }

    // The caller can give each stage a profiling encoder boundary while tests
    // and unprofiled execution retain the same dispatch sequence.
    pub(super) fn encode(
        &self,
        pipelines: &MetalLinearPipelines,
        regions: &[MetalBufferRegion],
        mut with_encoder: impl FnMut(&'static str, &dyn Fn(&ComputeCommandEncoderRef)),
    ) {
        for step in projection_steps(&self.gate_up, regions) {
            with_encoder("dense_swiglu.gate_up_projection", &|encoder| {
                step.encode(pipelines, encoder, regions, self.workspace);
            });
        }
        with_encoder("dense_swiglu.activation", &|encoder| {
            dispatch_swiglu(
                pipelines,
                encoder,
                &regions[self.scratch_region],
                self.activation,
            );
        });
        with_encoder("dense_swiglu.down_projection", &|encoder| {
            dispatch(pipelines, encoder, regions, self.down, self.workspace);
        });
    }
}

#[derive(Clone, Copy)]
struct PhysicalSpan<'a> {
    buffer: &'a metal::BufferRef,
    start: u64,
    end: u64,
}

impl PhysicalSpan<'_> {
    fn same(self, other: Self) -> bool {
        std::ptr::eq(self.buffer, other.buffer)
            && self.start == other.start
            && self.end == other.end
    }

    fn overlaps(self, other: Self) -> bool {
        std::ptr::eq(self.buffer, other.buffer) && self.start < other.end && other.start < self.end
    }
}

fn physical_span(
    regions: &[MetalBufferRegion],
    index: usize,
    offset: u64,
    bytes: u64,
) -> Option<PhysicalSpan<'_>> {
    let region = regions.get(index)?;
    if bytes == 0 || offset.checked_add(bytes)? > region.length_bytes() {
        return None;
    }
    let start = region.offset_bytes().checked_add(offset)?;
    Some(PhysicalSpan {
        buffer: region.buffer(),
        start,
        end: start.checked_add(bytes)?,
    })
}

fn can_reuse_hadamard(
    previous: LinearLaunch,
    current: LinearLaunch,
    regions: &[MetalBufferRegion],
) -> bool {
    let compatible = || -> Option<bool> {
        let before = previous.transform?;
        let after = current.transform?;
        if before.block_size != after.block_size
            || before.inverse != after.inverse
            || before.permutation != after.permutation
            || previous.activation_type != current.activation_type
            || previous.params.rows != current.params.rows
            || previous.params.in_features != current.params.in_features
        {
            return Some(false);
        }
        let elements =
            u64::from(current.params.rows).checked_mul(u64::from(current.params.in_features))?;
        let input_bytes = elements.checked_mul(current.activation_type.size_bytes())?;
        let input = physical_span(
            regions,
            current.input_region,
            current.input_offset_bytes,
            input_bytes,
        )?;
        let previous_input = physical_span(
            regions,
            previous.input_region,
            previous.input_offset_bytes,
            input_bytes,
        )?;
        let (before_region, before_offset) = previous.transform_workspace?;
        let (after_region, after_offset) = current.transform_workspace?;
        let workspace_bytes = elements.checked_mul(4)?;
        let workspace = physical_span(regions, after_region, after_offset, workspace_bytes)?;
        let previous_workspace =
            physical_span(regions, before_region, before_offset, workspace_bytes)?;
        if !input.same(previous_input)
            || !workspace.same(previous_workspace)
            || workspace.overlaps(input)
        {
            return Some(false);
        }
        let signs = match (before.signs_region, after.signs_region) {
            (None, None) => None,
            (Some(before), Some(after)) => {
                let a = physical_span(regions, before, 0, regions.get(before)?.length_bytes())?;
                let b = physical_span(regions, after, 0, regions.get(after)?.length_bytes())?;
                if !a.same(b) || workspace.overlaps(b) {
                    return Some(false);
                }
                Some(b)
            }
            _ => return Some(false),
        };
        // Conservatively cover whole output rows, including stride padding.
        // Neither the preceding transform nor its projection may mutate data
        // that another execution of this transform would read or overwrite.
        let output_bytes = u64::from(previous.params.rows)
            .checked_mul(u64::from(previous.params.output_stride))?
            .checked_mul(previous.activation_type.size_bytes())?;
        let output = physical_span(
            regions,
            previous.output_region,
            previous.output_offset_bytes,
            output_bytes,
        )?;
        Some(
            !output.overlaps(input)
                && !output.overlaps(workspace)
                && signs.is_none_or(|signs| !output.overlaps(signs)),
        )
    };
    compatible().unwrap_or(false)
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
    if workspace.is_some_and(|workspace| selected_for(launch, workspace.policy)) {
        // Staging bypasses the plain projection plan: one dequantization and
        // one GEMM cover the entire matrix, including any partial token tile.
        2
    } else {
        launch.dispatch_count()
    }
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

#[cfg(test)]
mod hadamard_reuse_tests;
