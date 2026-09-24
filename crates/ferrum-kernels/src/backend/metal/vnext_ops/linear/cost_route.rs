//! Dense linear route metadata shares ABI checks and row partition selection
//! with the real encoder. Resource-dependent transform routes remain unknown.
use super::*;
use ferrum_interfaces::execution_cost::SelectedCommandCostEvidenceV1;
use ferrum_interfaces::vnext::{
    DeviceCommandPhase, OperationCostCommand, OperationCostRoute, OperationCostRouteRequest,
    PhysicalWeightLayout,
};

/// Bounded, resource-free translation using the same ABI metadata translator
/// as resolve_weight. Only plain leaves or a two-part plain composite qualify.
pub(super) fn plain_weight(
    binding: &ferrum_interfaces::vnext::ResolvedValueBinding,
    composite: bool,
) -> Result<Option<(Vec<MetalResolvedWeightComponent>, MetalResolvedWeightLayout)>, String> {
    let weight = binding
        .weight()
        .ok_or("linear cost route lacks weight ABI")?;
    let plain_leaf = |layout: &PhysicalWeightLayout| {
        matches!(
            layout,
            PhysicalWeightLayout::Dense { .. }
                | PhysicalWeightLayout::Stored { .. }
                | PhysicalWeightLayout::BlockQuantized { .. }
        )
    };
    if let PhysicalWeightLayout::Composite { parts } = weight.physical_layout() {
        if !composite
            || parts.len() != 2
            || parts.iter().any(|p| {
                p.logical_offsets.len() != 3 || p.extents.len() != 3 || !plain_leaf(&p.layout)
            })
        {
            return Ok(None);
        }
    } else if !plain_leaf(weight.physical_layout()) {
        return Ok(None);
    }
    let limit = if composite { 2 } else { 1 };
    if weight.components().is_empty()
        || weight.components().len() > limit
        || weight.components().len() != binding.storage().components().len()
    {
        return Ok(None);
    }
    let mut metadata = Vec::new();
    metadata
        .try_reserve_exact(weight.components().len())
        .map_err(|_| "linear cost metadata capacity unavailable")?;
    for (component, stored) in weight
        .components()
        .iter()
        .zip(binding.storage().components())
    {
        if component.physical_dimensions().len() > 3
            || stored.component_id() != Some(component.component_id())
            || stored.element_type() != component.physical_element_type()
            || stored.length_bytes() != component.physical_bytes().map_err(|e| e.to_string())?
        {
            return Err("linear cost component differs from its physical ABI".into());
        }
        metadata.push(super::super::weights::component_metadata(component));
    }
    Ok(Some((
        metadata,
        super::super::weights::resolve_layout(weight)?,
    )))
}

pub(super) fn swiglu_route(
    pipelines: &MetalLinearPipelines,
    request: OperationCostRouteRequest<'_>,
) -> Result<Option<OperationCostRoute>, VNextError> {
    if request.operation_id().as_str() != DENSE_SWIGLU_OPERATION_ID {
        return Ok(None);
    }
    // The real encoder uses one shared activation region for the whole wave.
    if !request.binding_uses_packed_batch_coordinates(ResolvedValueRole::Input, 0)?
        || !request.binding_uses_packed_batch_coordinates(ResolvedValueRole::Output, 0)?
    {
        return Ok(None);
    }
    let calculate = || -> Result<Option<(u64, Option<SelectedCommandCostEvidenceV1>)>, String> {
        let hidden = unsigned_attribute(request.attributes(), "hidden_size")?;
        let intermediate = unsigned_attribute(request.attributes(), "intermediate_size")?;
        if hidden == 0 || intermediate == 0 {
            return Err("SwiGLU cost has empty features".into());
        }
        validate_swiglu_bindings(request.bindings(), hidden, intermediate)?;
        let Some((gate_components, gate_layout)) = plain_weight(
            binding(request.bindings(), ResolvedValueRole::Input, 1)?,
            true,
        )?
        else {
            return Ok(None);
        };
        let Some((down_components, down_layout)) = plain_weight(
            binding(request.bindings(), ResolvedValueRole::Input, 2)?,
            false,
        )?
        else {
            return Ok(None);
        };
        let packed = intermediate
            .checked_mul(2)
            .ok_or("SwiGLU packed width overflows")?;
        let gate_parts = match &gate_layout {
            MetalResolvedWeightLayout::Composite { parts } => {
                prepare_gate_up_partition(parts, intermediate, hidden, |layout, offset| {
                    prepare_leaf_encoding(&gate_components, layout, intermediate, hidden, 2, offset)
                })?
            }
            layout => vec![prepare_leaf_encoding(
                &gate_components,
                layout,
                packed,
                hidden,
                2,
                0,
            )?],
        };
        let down =
            prepare_leaf_encoding(&down_components, &down_layout, hidden, intermediate, 1, 0)?;
        let tokens = request.immediate_tokens();
        let elements = tokens
            .checked_mul(intermediate)
            .ok_or("SwiGLU activation elements overflow")?;
        let activation_bytes = elements
            .checked_mul(6)
            .ok_or("SwiGLU activation bytes overflow")?;
        let staging_bytes =
            staged_prefill::workspace_bytes(request.bindings(), hidden, intermediate)?;
        let scratch_bytes = activation_bytes
            .checked_add(staging_bytes)
            .ok_or("SwiGLU total scratch bytes overflow")?;
        let activation = swiglu_launch(
            0,
            elements
                .checked_mul(4)
                .ok_or("SwiGLU gate/up bytes overflow")?,
            tokens,
            intermediate,
            packed,
        )?;
        let policy = (staging_bytes > 0).then_some(staged_prefill::StagingPolicy::SwiGlu);
        let mut launches = Vec::with_capacity(gate_parts.len());
        let mut dispatches = 1_u64; // Pointwise activation, shared by all rows.
        for part in gate_parts {
            let launch = linear_launch(part, 0, 0, tokens, hidden, packed, 0, 0)?;
            launch.activation_bytes()?;
            dispatches += staged_prefill::policy_dispatch_count(launch, policy);
            launches.push(launch);
        }
        let launch = linear_launch(down, 0, 0, tokens, intermediate, hidden, 0, 0)?;
        launch.activation_bytes()?;
        dispatches += staged_prefill::policy_dispatch_count(launch, policy);
        let statistics = selected::swiglu(
            pipelines,
            &launches,
            launch,
            activation,
            policy,
            tokens,
            scratch_bytes,
        );
        Ok(Some((dispatches, statistics)))
    };
    let Some((dispatches, statistics)) = calculate().map_err(invalid_plan)? else {
        return Ok(None);
    };
    let participants = request.rows().len() as u32;
    let command = OperationCostCommand::new(
        "vnext_dense_swiglu",
        DeviceCommandPhase::Compute,
        if participants == 1 {
            DeviceBatchingForm::Scalar
        } else {
            DeviceBatchingForm::Packed
        },
        0,
        participants,
        request.immediate_tokens(),
        dispatches,
        0,
    )?;
    let command = if let Some(evidence) = statistics {
        command
            .clone()
            .with_statistical_evidence(evidence)
            .unwrap_or(command)
    } else {
        command
    };
    OperationCostRoute::new(vec![command]).map(Some)
}

pub(super) fn dense_command_selected(
    pipelines: &MetalLinearPipelines,
    participants: u32,
    tokens: u64,
    launches: &[LinearLaunch],
) -> Result<OperationCostCommand, VNextError> {
    let command = dense_command(participants, tokens, launches)?;
    match selected::dense(pipelines, launches, tokens) {
        Some(evidence) => Ok(command
            .clone()
            .with_statistical_evidence(evidence)
            .unwrap_or(command)),
        None => Ok(command),
    }
}

pub(super) fn dense_command(
    participants: u32,
    tokens: u64,
    launches: &[LinearLaunch],
) -> Result<OperationCostCommand, VNextError> {
    if launches.is_empty() || (launches.len() != 1 && launches.len() != participants as usize) {
        return Err(invalid_plan(
            "dense cost route has incomplete launch coverage",
        ));
    }
    let dispatches = launches.iter().try_fold(0_u64, |total, launch| {
        total
            .checked_add(launch.dispatch_count())
            .ok_or_else(|| invalid_plan("dense cost route dispatch count overflows"))
    })?;
    OperationCostCommand::new(
        "vnext_dense_linear",
        DeviceCommandPhase::Compute,
        if participants == 1 {
            DeviceBatchingForm::Scalar
        } else if launches.len() == 1 {
            DeviceBatchingForm::Packed
        } else {
            DeviceBatchingForm::ParticipantLoop
        },
        0,
        participants,
        tokens,
        dispatches,
        0,
    )
}

pub(super) fn dense_route(
    pipelines: &MetalLinearPipelines,
    request: OperationCostRouteRequest<'_>,
) -> Result<Option<OperationCostRoute>, VNextError> {
    if request.operation_id().as_str() != DENSE_LINEAR_OPERATION_ID {
        return Ok(None);
    }
    let get_part = || -> Result<Option<PreparedLinearPart>, String> {
        let input = unsigned_attribute(request.attributes(), "in_features")?;
        let output = unsigned_attribute(request.attributes(), "out_features")?;
        if input == 0 || output == 0 {
            return Err("dense cost route has empty features".into());
        }
        checked_u32(input, "dense cost route input width")?;
        checked_u32(output, "dense cost route output width")?;
        validate_dense_linear_bindings(request.bindings(), input, output)?;
        let binding = binding(request.bindings(), ResolvedValueRole::Input, 1)?;
        let Some((metadata, layout)) = plain_weight(binding, false)? else {
            return Ok(None);
        };
        prepare_leaf_encoding(&metadata, &layout, output, input, 1, 0).map(Some)
    };
    let Some(part) = get_part().map_err(invalid_plan)? else {
        return Ok(None);
    };
    let input_packed =
        request.binding_uses_packed_batch_coordinates(ResolvedValueRole::Input, 0)?;
    let output_packed =
        request.binding_uses_packed_batch_coordinates(ResolvedValueRole::Output, 0)?;
    let packed = input_packed && output_packed;
    let mut launches = Vec::new();
    launches
        .try_reserve_exact(if packed { 1 } else { request.rows().len() })
        .map_err(|_| invalid_plan("dense cost route launch capacity unavailable"))?;
    let input = unsigned_attribute(request.attributes(), "in_features").map_err(invalid_plan)?;
    let output = unsigned_attribute(request.attributes(), "out_features").map_err(invalid_plan)?;
    let mut append = |tokens: u64| -> Result<(), VNextError> {
        // Launch contains numeric offsets only. No buffers, invocation, scratch
        // reservation, or device encoding is created by this calculation.
        let launch =
            linear_launch(part, 0, 0, tokens, input, output, 0, 0).map_err(invalid_plan)?;
        launch.activation_bytes().map_err(invalid_plan)?;
        launches.push(launch);
        Ok(())
    };
    if packed {
        append(request.immediate_tokens())?;
    } else {
        for row in request.rows() {
            append(row.count.get())?;
        }
    }
    OperationCostRoute::new(vec![dense_command_selected(
        pipelines,
        request.rows().len() as u32,
        request.immediate_tokens(),
        &launches,
    )?])
    .map(Some)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn launch(format: LinearPhysicalFormat, rows: u64) -> LinearLaunch {
        linear_launch(
            PreparedLinearPart {
                region: 0,
                format,
                output_offset: 0,
                out_features: 4096,
                transform: None,
            },
            0,
            0,
            rows,
            4096,
            4096,
            0,
            0,
        )
        .unwrap()
    }

    #[test]
    fn dense_route_counts_real_plain_partition_and_batch_shape() {
        let single = dense_command(1, 33, &[launch(LinearPhysicalFormat::Q4K, 33)]).unwrap();
        assert_eq!(single.compute_dispatch_count(), 2);
        assert_eq!(single.batching(), DeviceBatchingForm::Scalar);
        let packed = dense_command(8, 8, &[launch(LinearPhysicalFormat::Q4K, 8)]).unwrap();
        assert_eq!(packed.compute_dispatch_count(), 2);
        assert_eq!(packed.batching(), DeviceBatchingForm::Packed);
        let separate = dense_command(8, 8, &vec![launch(LinearPhysicalFormat::Q4K, 1); 8]).unwrap();
        assert_eq!(separate.compute_dispatch_count(), 8);
        assert_eq!(separate.batching(), DeviceBatchingForm::ParticipantLoop);
        assert!(dense_command(8, 8, &[]).is_err());
        assert!(dense_command(8, 8, &[launch(LinearPhysicalFormat::Q4K, 1); 2]).is_err());
    }

    #[test]
    fn projected_activation_extent_rejects_byte_overflow_despite_valid_shader_dimensions() {
        let part = PreparedLinearPart {
            region: 0,
            format: LinearPhysicalFormat::DenseF16,
            output_offset: 0,
            out_features: u32::MAX,
            transform: None,
        };
        let launch = linear_launch(
            part,
            0,
            0,
            u64::from(u32::MAX),
            u64::from(u32::MAX),
            u64::from(u32::MAX),
            0,
            0,
        )
        .unwrap();
        assert!(launch.activation_bytes().is_err());
    }
}
