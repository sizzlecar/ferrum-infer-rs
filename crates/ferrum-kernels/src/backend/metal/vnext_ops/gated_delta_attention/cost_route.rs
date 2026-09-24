//! Future eager GDN route, from this provider's actual static decisions.
//! No state values, physical addresses, resources or historical trace required.
use super::super::linear::{prepare_leaf_encoding, prepare_matrix_partition, PreparedLinearPart};
use super::*;
use ferrum_interfaces::vnext::{
    DeviceCommandPhase, OperationCostCommand, OperationCostRoute, OperationCostRouteRequest,
    OperationCostWorkRow, PhysicalWeightLayout, ResolvedWeightBinding,
};

// Bound pre-submission tree translation even when the executable itself can
// describe a larger composite. Unsupported projection coverage remains None.
const MAX_PLAIN_PARTS: usize = 16;

#[derive(Clone, Copy)]
pub(super) struct ProjectionDispatches {
    pub input: u64,
    pub output: u64,
}

pub(super) struct RowDispatches {
    pub projections: ProjectionDispatches,
    pub delta: u64,
    pub chunked: bool,
}

/// Both the actual encoder and future query use this command accounting.
/// Projection counts come from their respective prepared numerical launches;
/// the actual side still accounts for live Hadamard reuse before calling here.
pub(super) fn command(
    hidden_type: ElementType,
    tokens: u64,
    packed: Option<ProjectionDispatches>,
    rows: impl ExactSizeIterator<Item = Result<RowDispatches, String>>,
) -> Result<OperationCostCommand, String> {
    let participants = checked_u32(rows.len() as u64, "Metal gated-delta participants")?;
    if participants == 0
        || tokens == 0
        || !matches!(hidden_type, ElementType::F16 | ElementType::F32)
    {
        return Err("Metal gated-delta cost command has empty or invalid work".into());
    }
    let add = |a: u64, b: u64| {
        a.checked_add(b)
            .ok_or_else(|| "Metal gated-delta dispatch count overflows".to_owned())
    };
    let mut dispatches = if let Some(projection) = packed {
        add(add(5, projection.input)?, projection.output)?
    } else {
        0
    };
    let mut chunked = 0;
    for row in rows {
        let row = row?;
        chunked += usize::from(row.chunked);
        let local = if packed.is_some() {
            3
        } else {
            add(add(8, row.projections.input)?, row.projections.output)?
        };
        dispatches = add(dispatches, add(local, row.delta)?)?;
    }
    let operation = match (hidden_type, chunked, participants as usize) {
        (ElementType::F32, count, total) if count == total => {
            "vnext_gated_delta_chunked_attention_f32_master"
        }
        (ElementType::F32, 0, _) => "vnext_gated_delta_recurrent_attention_f32_master",
        (ElementType::F32, _, _) => "vnext_gated_delta_mixed_attention_f32_master",
        (_, count, total) if count == total => "vnext_gated_delta_chunked_attention",
        (_, 0, _) => "vnext_gated_delta_recurrent_attention",
        _ => "vnext_gated_delta_mixed_attention",
    };
    OperationCostCommand::new(
        operation,
        DeviceCommandPhase::Compute,
        if packed.is_some() {
            DeviceBatchingForm::Packed
        } else if participants == 1 {
            DeviceBatchingForm::Scalar
        } else {
            DeviceBatchingForm::ParticipantLoop
        },
        0,
        participants,
        tokens,
        dispatches,
        0,
    )
    .map_err(|error| error.to_string())
}

pub(super) fn eager_route(
    request: OperationCostRouteRequest<'_>,
    operation_id: &str,
    hidden_type: ElementType,
    capabilities: GatedDeltaExecutionCapabilities,
    cost_model: MetalGatedDeltaExecutionCostModel,
    attention: &MetalGatedDeltaPipelines,
    linear: &MetalLinearPipelines,
    primitives: &MetalPrimitivePipelines,
) -> Result<Option<OperationCostRoute>, VNextError> {
    if request.operation_id().as_str() != operation_id {
        return Ok(None);
    }
    let calculate = || -> Result<Option<OperationCostCommand>, String> {
        let shape = AttentionShape::from_attributes(request.attributes())?;
        validate_bindings(request.bindings(), shape, hidden_type)?;
        let input_binding = binding(request.bindings(), ResolvedValueRole::Input, 2)?;
        let output_binding = binding(request.bindings(), ResolvedValueRole::Input, 7)?;
        let Some(input) = projection(
            input_binding,
            shape.qkvzba_features,
            shape.hidden_size,
            true,
        )?
        else {
            return Ok(None);
        };
        let Some(output) = projection(
            output_binding,
            shape.hidden_size,
            shape.value_features,
            false,
        )?
        else {
            return Ok(None);
        };
        let [output] = output.as_slice() else {
            return Err("Metal GDN output projection is not one matrix".into());
        };
        let staging_bytes = input_projection_workspace(request.bindings(), shape)?;
        let layout = ScratchLayout::new(shape, request.immediate_tokens())?
            .with_projection_workspace(staging_bytes)?;
        let packed = request.rows().len() > 1
            && request
                .binding_uses_packed_batch_coordinates(ResolvedValueRole::Input, 0)
                .map_err(|e| e.to_string())?
            && request
                .binding_uses_packed_batch_coordinates(ResolvedValueRole::Output, 0)
                .map_err(|e| e.to_string())?;
        let route = project(
            shape,
            hidden_type,
            request.rows(),
            request.immediate_tokens(),
            packed,
            &input,
            *output,
            layout,
            staging_bytes,
            capabilities,
            cost_model,
        )?;
        let statistics = project_statistics(
            shape,
            hidden_type,
            request.rows(),
            request.immediate_tokens(),
            packed,
            &input,
            *output,
            layout,
            staging_bytes,
            capabilities,
            cost_model,
            attention,
            linear,
            primitives,
        );
        Ok(Some(match statistics {
            Some(evidence) => route
                .clone()
                .with_statistical_evidence(evidence)
                .unwrap_or(route),
            None => route,
        }))
    };
    calculate()
        .map_err(invalid_plan)?
        .map(|command| OperationCostRoute::new(vec![command]))
        .transpose()
}

fn projection(
    binding: &ResolvedValueBinding,
    output: u64,
    input: u64,
    partitioned: bool,
) -> Result<Option<Vec<PreparedLinearPart>>, String> {
    let weight = binding
        .weight()
        .ok_or_else(|| "Metal GDN projection has no weight ABI".to_owned())?;
    if weight.components().len() > MAX_PLAIN_PARTS {
        return Ok(None);
    }
    if weight.components().len() != binding.storage().components().len() {
        return Err("Metal GDN weight storage differs from its components".into());
    }
    for (component, stored) in weight
        .components()
        .iter()
        .zip(binding.storage().components())
    {
        if stored.component_id() != Some(component.component_id())
            || stored.element_type() != component.physical_element_type()
            || stored.length_bytes() != component.physical_bytes().map_err(|e| e.to_string())?
        {
            return Err("Metal GDN projection component differs from its physical ABI".into());
        }
    }
    plain_projection(weight, output, input, partitioned)
}

fn plain_projection(
    weight: &ResolvedWeightBinding,
    output: u64,
    input: u64,
    partitioned: bool,
) -> Result<Option<Vec<PreparedLinearPart>>, String> {
    let leaf = |layout: &PhysicalWeightLayout| {
        matches!(
            layout,
            PhysicalWeightLayout::Dense { .. }
                | PhysicalWeightLayout::Stored { .. }
                | PhysicalWeightLayout::BlockQuantized { .. }
        )
    };
    if let PhysicalWeightLayout::Composite { parts } = weight.physical_layout() {
        if !partitioned
            || parts.is_empty()
            || parts.len() > MAX_PLAIN_PARTS
            || parts.iter().any(|part| {
                part.logical_offsets.len() != 2 || part.extents.len() != 2 || !leaf(&part.layout)
            })
        {
            return Ok(None);
        }
    } else if !leaf(weight.physical_layout()) {
        return Ok(None);
    }
    if weight.components().is_empty()
        || weight.components().len() > if partitioned { MAX_PLAIN_PARTS } else { 1 }
        || weight
            .components()
            .iter()
            .any(|component| component.physical_dimensions().len() > 3)
    {
        return Ok(None);
    }
    let mut components = Vec::new();
    components
        .try_reserve_exact(weight.components().len())
        .map_err(|_| "Metal GDN metadata capacity unavailable")?;
    components.extend(
        weight
            .components()
            .iter()
            .map(super::super::weights::component_metadata),
    );
    let layout = super::super::weights::resolve_layout(weight)?;
    prepare_matrix_partition(&layout, output, input, |layout, leaf_output, offset| {
        prepare_leaf_encoding(&components, layout, leaf_output, input, 1, offset)
    })
    .map(Some)
}

#[allow(clippy::too_many_arguments)]
fn project(
    shape: AttentionShape,
    hidden_type: ElementType,
    rows: &[OperationCostWorkRow],
    total: u64,
    packed: bool,
    input: &[PreparedLinearPart],
    output: PreparedLinearPart,
    layout: ScratchLayout,
    staging_bytes: u64,
    capabilities: GatedDeltaExecutionCapabilities,
    cost_model: MetalGatedDeltaExecutionCostModel,
) -> Result<OperationCostCommand, String> {
    let policy = (staging_bytes != 0).then_some(staged_prefill::StagingPolicy::GatedDelta);
    let packed_projection = if packed {
        shape.validate_launch_extents(total)?;
        Some(project_dispatches(
            shape, input, output, layout, 0, total, policy,
        )?)
    } else {
        None
    };
    let mut projected = Vec::new();
    projected
        .try_reserve_exact(rows.len())
        .map_err(|_| "Metal GDN row metadata capacity unavailable")?;
    let mut start = 0;
    for row in rows {
        let tokens = row.count.get();
        let form = shape.execution_form(tokens, capabilities, cost_model)?;
        let projections = if packed {
            ProjectionDispatches {
                input: 0,
                output: 0,
            }
        } else {
            project_dispatches(shape, input, output, layout, start, tokens, policy)?
        };
        projected.push(RowDispatches {
            projections,
            delta: delta_dispatch_count(form, &shape.params(tokens)?),
            chunked: matches!(form, GatedDeltaExecutionForm::ChunkedScan(_)),
        });
        start = start
            .checked_add(tokens)
            .ok_or("Metal GDN projected token count overflows")?;
    }
    if start != total {
        return Err("Metal GDN projected rows differ from total tokens".into());
    }
    command(
        hidden_type,
        total,
        packed_projection,
        projected.into_iter().map(Ok),
    )
}

#[allow(clippy::too_many_arguments)]
fn project_dispatches(
    shape: AttentionShape,
    input: &[PreparedLinearPart],
    output: PreparedLinearPart,
    layout: ScratchLayout,
    start: u64,
    tokens: u64,
    policy: Option<staged_prefill::StagingPolicy>,
) -> Result<ProjectionDispatches, String> {
    let (input, output) = project_launches(shape, input, output, layout, start, tokens)?;
    let input = input.into_iter().try_fold(0_u64, |count, launch| {
        count
            .checked_add(staged_prefill::policy_dispatch_count(launch, policy))
            .ok_or_else(|| "Metal GDN input dispatch count overflows".to_owned())
    })?;
    Ok(ProjectionDispatches {
        input,
        output: output.dispatch_count(),
    })
}

fn project_launches(
    shape: AttentionShape,
    input: &[PreparedLinearPart],
    output: PreparedLinearPart,
    layout: ScratchLayout,
    start: u64,
    tokens: u64,
) -> Result<(Vec<LinearLaunch>, LinearLaunch), String> {
    let end_token = start
        .checked_add(tokens)
        .ok_or("Metal GDN projection row range overflows")?;
    for (base, width, limit) in [
        (layout.normalized, shape.hidden_size, layout.qkvzba),
        (layout.qkvzba, shape.qkvzba_features, layout.z),
    ] {
        if layout.token_offset(base, end_token, width, ElementType::F16)? > limit {
            return Err("Metal GDN projection crosses an activation scratch segment".into());
        }
    }
    let normalized = layout.token_offset(
        layout.normalized,
        start,
        shape.hidden_size,
        ElementType::F16,
    )?;
    let qkvzba = layout.token_offset(
        layout.qkvzba,
        start,
        shape.qkvzba_features,
        ElementType::F16,
    )?;
    let checked_launch = |part, width, stride, input_offset, output_offset| {
        let launch = linear_launch(
            part,
            0,
            0,
            tokens,
            width,
            stride,
            input_offset,
            output_offset,
        )?;
        let (input_bytes, output_bytes) = launch.activation_bytes()?;
        if input_offset
            .checked_add(input_bytes)
            .is_none_or(|end| end > layout.projection_workspace)
            || output_offset
                .checked_add(output_bytes)
                .is_none_or(|end| end > layout.projection_workspace)
        {
            return Err("Metal GDN projection exceeds numerical scratch layout".into());
        }
        Ok::<_, String>(launch)
    };
    let input = input
        .iter()
        .map(|part| {
            checked_launch(
                *part,
                shape.hidden_size,
                shape.qkvzba_features,
                normalized,
                qkvzba,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    let output = checked_launch(
        output,
        shape.value_features,
        shape.hidden_size,
        qkvzba,
        normalized,
    )?;
    Ok((input, output))
}

#[allow(clippy::too_many_arguments)]
fn project_statistics(
    shape: AttentionShape,
    hidden: ElementType,
    rows: &[OperationCostWorkRow],
    total: u64,
    packed: bool,
    input: &[PreparedLinearPart],
    output: PreparedLinearPart,
    layout: ScratchLayout,
    staging_bytes: u64,
    capabilities: GatedDeltaExecutionCapabilities,
    cost_model: MetalGatedDeltaExecutionCostModel,
    attention: &MetalGatedDeltaPipelines,
    linear: &MetalLinearPipelines,
    primitives: &MetalPrimitivePipelines,
) -> Option<ferrum_interfaces::execution_cost::SelectedCommandCostEvidenceV1> {
    let mut projections = Vec::new();
    projections.try_reserve_exact(rows.len()).ok()?;
    let mut start = 0_u64;
    for row in rows {
        let count = row.count.get();
        let form = shape.execution_form(count, capabilities, cost_model).ok()?;
        let (input, output) = project_launches(shape, input, output, layout, start, count).ok()?;
        projections.push((shape.params(count).ok()?, form, input, output));
        start = start.checked_add(count)?;
    }
    let packed_projection = if packed {
        Some(project_launches(shape, input, output, layout, 0, total).ok()?)
    } else {
        None
    };
    let packed = if let Some((input, output)) = packed_projection.as_ref() {
        Some(selected::Projection {
            params: shape.params(total).ok()?,
            input,
            output: *output,
            staged: staging_bytes != 0,
        })
    } else {
        None
    };
    selected::evidence(
        attention,
        linear,
        primitives,
        hidden,
        total,
        layout.required_bytes,
        packed,
        projections
            .iter()
            .map(|(params, form, input, output)| selected::Row {
                projection: selected::Projection {
                    params: *params,
                    input,
                    output: *output,
                    staged: staging_bytes != 0,
                },
                form: *form,
            }),
    )
}

#[cfg(test)]
mod tests;
