//! One admitted FFN invocation; all geometry is fixed before graph capture.
use super::super::replay_encoding::{Encoding, EncodingTarget};
use super::*;
use crate::backend::cuda::vnext_ops::native_blocks::upstream_linear::{
    PreparedUpstreamLeaf, WeightValidation,
};
use crate::native_ops::upstream_linear::DeviceSpan;
use ferrum_interfaces::vnext::{
    EncodedRetainedPlanDependency, EncodedReusableExecutionBindings, ProjectionRole,
    UpstreamProjectionLayout, UpstreamProjectionWaveFacts,
};

struct Leaf {
    weight: usize,
    part: weights::MatrixPart,
    stage: Arc<PreparedUpstreamLeaf>,
    validation: Option<Arc<WeightValidation>>,
}

struct Launch {
    input: usize,
    output: usize,
    gate: usize,
    activation: usize,
    rows: u32,
    start: u64,
    g32: bool,
    gate_leaves: Vec<Leaf>,
    down_leaves: Vec<Leaf>,
}

pub(super) fn encode(
    provider: &CudaUpstreamSwiGluProvider,
    invocation: BatchedOperationInvocation<'_, CudaDeviceBuffer>,
    target: EncodingTarget,
) -> Result<Encoding<CudaDeviceCommand>, String> {
    ensure_invocation(&invocation, provider.profile.operation_id())?;
    let first = &invocation.participants()[0];
    let prepared = first
        .projection_numerics()
        .ok_or("upstream FFN lacks retained numerics")?;
    if prepared.contract() != &provider.arithmetic
        || invocation
            .participants()
            .iter()
            .any(|p| p.projection_numerics() != Some(prepared))
    {
        return Err("upstream FFN participants disagree on their numerical contract".into());
    }
    let hidden = unsigned_attribute(first.attributes(), "hidden_size")?;
    let intermediate = unsigned_attribute(first.attributes(), "intermediate_size")?;
    let hidden_u32 = checked_u32(hidden, "upstream hidden")?;
    let intermediate_i32 =
        i32::try_from(intermediate).map_err(|_| "upstream intermediate exceeds i32")?;
    let doubled = intermediate
        .checked_mul(2)
        .ok_or("upstream gate/up width overflows")?;
    let doubled_u32 = checked_u32(doubled, "upstream gate/up width")?;
    let tokens = invocation.work_shape().immediate_tokens();
    let activation_elements = mul(tokens, intermediate)?;
    let gate_bytes = mul(activation_elements, 4)?;
    if gate_bytes / 2 > i32::MAX as u64 {
        return Err("upstream SiLU indexing exceeds i32".into());
    }
    let transform_bytes = mul(
        super::super::super::native_blocks::hadamard::workspace_bytes_per_token(first.bindings())?,
        tokens,
    )?;
    let base_bytes = add(transform_bytes, mul(activation_elements, 6)?)?;
    let extra_offset = add(base_bytes, 15)? & !15;
    let extra_bytes = provider.scratch_bytes(prepared)?.bytes(tokens)?;
    let required = add(extra_offset, extra_bytes)?;
    let mut regions = Vec::new();
    let gate_up =
        native_matrix::resolve_shared(&mut regions, &invocation, 1, &[2, intermediate, hidden])?;
    let down =
        native_matrix::resolve_shared(&mut regions, &invocation, 2, &[hidden, intermediate])?;
    for (role, matrix) in [
        (ProjectionRole::SwiGluGateUp, &gate_up),
        (ProjectionRole::SwiGluDown, &down),
    ] {
        validate_parts(
            provider.profile,
            prepared
                .projection(role)
                .ok_or("upstream projection absent")?,
            &matrix.parts,
        )?;
    }
    let scratch = shared_scratch_region(&invocation, required)?;
    let persistent = persistent_region(&invocation, flag_bytes(prepared)?)?;
    let scratch_index = regions.len();
    regions.push(scratch.clone());
    regions.push(persistent.clone());
    let input_packed = token_binding_is_packed(&invocation, ResolvedValueRole::Input, 0)?;
    let output_packed = token_binding_is_packed(&invocation, ResolvedValueRole::Output, 0)?;
    let packed =
        native_matrix::single_launch_rows(tokens).filter(|_| input_packed && output_packed);
    let mut coordinates = Vec::new();
    if invocation.participant_token_ranges().len() != invocation.participants().len() {
        return Err("upstream FFN participant ranges are incomplete".into());
    }
    for (participant, range) in invocation
        .participants()
        .iter()
        .zip(invocation.participant_token_ranges())
    {
        if unsigned_attribute(participant.attributes(), "hidden_size")? != hidden
            || unsigned_attribute(participant.attributes(), "intermediate_size")? != intermediate
        {
            return Err("upstream FFN participant dimensions differ".into());
        }
        let input = binding(participant.bindings(), ResolvedValueRole::Input, 0)?;
        let output = binding(participant.bindings(), ResolvedValueRole::Output, 0)?;
        validate_dense_swiglu(
            input,
            binding(participant.bindings(), ResolvedValueRole::Input, 1)?,
            binding(participant.bindings(), ResolvedValueRole::Input, 2)?,
            output,
            hidden,
            intermediate,
        )?;
        let rows = native_matrix::single_launch_rows(range.immediate_tokens())
            .ok_or("upstream FFN local row extent unsupported")?;
        let target = range.immediate_token_range();
        if target.end > tokens {
            return Err("upstream FFN participant exceeds scratch rows".into());
        }
        if packed.is_some() {
            continue;
        }
        let source = range.source_token_range();
        let input_index = regions.len();
        regions.push(contiguous_token_region(
            participant,
            input,
            ElementType::F16,
            if input_packed {
                target.start
            } else {
                source.start
            },
            u64::from(rows),
        )?);
        let output_index = regions.len();
        regions.push(contiguous_token_region(
            participant,
            output,
            ElementType::F16,
            if output_packed {
                target.start
            } else {
                source.start
            },
            u64::from(rows),
        )?);
        coordinates.push((input_index, output_index, rows, target.start));
    }
    if let Some(rows) = packed {
        let input = regions.len();
        regions.push(shared_token_region(
            &invocation,
            ResolvedValueRole::Input,
            0,
            ElementType::F16,
            tokens,
        )?);
        let output = regions.len();
        regions.push(shared_token_region(
            &invocation,
            ResolvedValueRole::Output,
            0,
            ElementType::F16,
            tokens,
        )?);
        coordinates.push((input, output, rows, 0));
    }
    let participants = checked_u32(
        invocation.participants().len() as u64,
        "upstream participants",
    )?;
    let mut launches = Vec::new();
    let mut bindings = Vec::new();
    let mut dispatches = 0_u64;
    let mut transfers = 0_u64;
    for (input, output, rows, start) in coordinates {
        let gate = regions.len();
        regions.push(
            scratch
                .subregion(
                    add(transform_bytes, mul(mul(start, intermediate)?, 4)?)?,
                    mul(mul(u64::from(rows), intermediate)?, 4)?,
                )
                .map_err(|e| e.to_string())?,
        );
        let activation = regions.len();
        regions.push(
            scratch
                .subregion(
                    add(
                        add(transform_bytes, gate_bytes)?,
                        mul(mul(start, intermediate)?, 2)?,
                    )?,
                    mul(mul(u64::from(rows), intermediate)?, 2)?,
                )
                .map_err(|e| e.to_string())?,
        );
        let gate_leaves = prepare_leaves(
            target,
            &invocation,
            provider,
            prepared,
            ProjectionRole::SwiGluGateUp,
            &gate_up,
            &regions,
            input,
            gate,
            rows,
            &persistent,
            0,
            &mut bindings,
        )?;
        let down_leaves = prepare_leaves(
            target,
            &invocation,
            provider,
            prepared,
            ProjectionRole::SwiGluDown,
            &down,
            &regions,
            activation,
            output,
            rows,
            &persistent,
            gate_up.parts.len() as u64,
            &mut bindings,
        )?;
        if target == EncodingTarget::BindingsOnly {
            // All live matrix, scratch, range and dependency checks above are
            // shared with full encoding; no retained compute is rebuilt.
            continue;
        }
        let g32 = provider.profile.uses_g32(rows);
        if g32 {
            dispatches = add(
                dispatches,
                u64::from(
                    prepared
                        .projection(ProjectionRole::SwiGluGateUp)
                        .ok_or("missing gate")?
                        .has_staged_leaf(),
                ) + u64::from(
                    prepared
                        .projection(ProjectionRole::SwiGluDown)
                        .ok_or("missing down")?
                        .has_staged_leaf(),
                ),
            )?;
        }
        for leaf in gate_leaves.iter().chain(&down_leaves) {
            if let Some(native) = &leaf.stage.native {
                let p = native.geometry();
                dispatches = add(dispatches, 4 + u64::from(p.fixup != 0))?;
                transfers = add(transfers, 1 + u64::from(p.guard_blocks != 0))?;
            } else {
                dispatches = add(dispatches, 1 + u64::from(leaf.part.transform.is_some()))?;
            }
        }
        dispatches = add(dispatches, 1)?; // SiLU multiplication.
        launches.push(Launch {
            input,
            output,
            gate,
            activation,
            rows,
            start,
            g32,
            gate_leaves,
            down_leaves,
        });
    }
    if target == EncodingTarget::BindingsOnly {
        let bindings = bindings.into_iter().fold(
            EncodedReusableExecutionBindings::empty(),
            |encoded, dependency| encoded.with_retained_plan_dependency(dependency),
        );
        return Ok(Encoding::Bindings(bindings));
    }
    let mut key = CudaCommandReplayKeyBuilder::new(
        provider.descriptor.provider_implementation_fingerprint(),
        "upstream_marker_v2_swiglu",
    )
    .bytes(prepared.fingerprint().as_bytes())
    .u64(hidden)
    .u64(intermediate)
    .u64(tokens)
    .u64(transform_bytes)
    .u64(extra_offset)
    .u64(extra_bytes)
    .boolean(packed.is_some());
    key = weights::key(weights::key(key, &gate_up.parts), &down.parts);
    for launch in &launches {
        key = key
            .u64(launch.input as u64)
            .u64(launch.output as u64)
            .u64(launch.gate as u64)
            .u64(launch.activation as u64)
            .u32(launch.rows)
            .u64(launch.start);
        for leaf in launch.gate_leaves.iter().chain(&launch.down_leaves) {
            key = key.bytes(leaf.stage.replay_fingerprint()?.as_bytes());
        }
    }
    let kernels = provider.native.clone();
    let g32_kernels = provider.g32.clone();
    let g32_gate = prepared
        .projection(ProjectionRole::SwiGluGateUp)
        .ok_or("missing gate")?
        .clone();
    let g32_down = prepared
        .projection(ProjectionRole::SwiGluDown)
        .ok_or("missing down")?
        .clone();
    let silu = provider.silu.clone();
    let compute = CudaDeviceCommand::replayable_operation(
        "upstream_marker_v2_swiglu",
        regions,
        key.finish(),
        move |stream, regions| {
            let workspace = DeviceSpan {
                address: regions[scratch_index]
                    .device_ptr()
                    .checked_add(extra_offset)
                    .ok_or_else(|| {
                        CudaDeviceRuntimeError::contract("upstream workspace address overflows")
                    })?,
                bytes: extra_bytes,
            };
            let transform = if transform_bytes == 0 {
                0
            } else {
                regions[scratch_index].device_ptr()
            };
            for launch in &launches {
                launch_leaves(
                    launch
                        .g32
                        .then_some((&g32_kernels, &g32_gate, gate_up.parts.as_ref())),
                    &kernels,
                    stream,
                    regions,
                    &launch.gate_leaves,
                    launch.input,
                    launch.gate,
                    launch.rows,
                    doubled_u32,
                    workspace,
                    transform,
                    gate_up.first_region,
                )?;
                launch_silu_mul(
                    stream,
                    &silu,
                    regions[launch.gate].device_ptr(),
                    regions[launch.activation].device_ptr(),
                    intermediate_i32,
                    u64::from(launch.rows) * intermediate,
                )?;
                launch_leaves(
                    launch
                        .g32
                        .then_some((&g32_kernels, &g32_down, down.parts.as_ref())),
                    &kernels,
                    stream,
                    regions,
                    &launch.down_leaves,
                    launch.activation,
                    launch.output,
                    launch.rows,
                    hidden_u32,
                    workspace,
                    transform,
                    down.first_region,
                )?;
            }
            Ok(())
        },
    )
    .and_then(|command| {
        command.with_work_attribution(
            if participants == 1 {
                DeviceBatchingForm::Scalar
            } else if packed.is_some() {
                DeviceBatchingForm::Packed
            } else {
                DeviceBatchingForm::ParticipantLoop
            },
            participants,
            tokens,
            dispatches,
            transfers,
        )
    })
    .map_err(|e| e.to_string())?;
    let mut operation = EncodedDeviceOperation::compute(compute);
    for binding in bindings {
        operation = operation.with_retained_plan_dependency(binding);
    }
    Ok(Encoding::Full(operation))
}

#[allow(clippy::too_many_arguments)]
fn prepare_leaves(
    target: EncodingTarget,
    invocation: &BatchedOperationInvocation<'_, CudaDeviceBuffer>,
    provider: &CudaUpstreamSwiGluProvider,
    prepared: &PreparedProjectionNumerics,
    role: ProjectionRole,
    matrix: &native_matrix::SharedNativeMatrix,
    regions: &[CudaBufferRegion],
    input: usize,
    output: usize,
    rows: u32,
    persistent: &CudaBufferRegion,
    first_leaf: u64,
    bindings: &mut Vec<EncodedRetainedPlanDependency<CudaDeviceCommand>>,
) -> Result<Vec<Leaf>, String> {
    let projection = prepared
        .projection(role)
        .ok_or("missing upstream retained role")?;
    matrix
        .parts
        .iter()
        .enumerate()
        .map(|(index, part)| {
            let weight = matrix.first_region + index;
            let facts = UpstreamProjectionWaveFacts {
                role,
                component_id: part.component_id.clone(),
                local_rows: rows,
                layout: UpstreamProjectionLayout::Columns,
                input_stride: projection.input_features(),
                output_stride: projection.output_features(),
                input_byte_offset: regions[input].backing_byte_offset(),
                output_byte_offset: regions[output].backing_byte_offset(),
                weight_byte_offset: regions[weight].backing_byte_offset(),
                input_available_bytes: regions[input].length_bytes(),
                output_available_bytes: regions[output].length_bytes(),
                weight_available_bytes: regions[weight].length_bytes(),
                retained_zero_padded_weight_rows: u64::from(part.rows),
            };
            let stage =
                provider
                    .plans
                    .prepare_for(prepared, &facts, target.projection_preparation())?;
            let validation = if let Some(native) = &stage.native {
                let bank = if native.geometry().algorithm == 1 {
                    0
                } else {
                    4
                };
                let offset = add(mul(add(first_leaf, index as u64)?, 8)?, bank)?;
                let flag = persistent.subregion(offset, 4).map_err(|e| e.to_string())?;
                let state = provider
                    .validation
                    .prepare(
                        native.clone(),
                        provider.descriptor.provider_implementation_fingerprint(),
                        regions[weight].clone(),
                        flag,
                    )
                    .map_err(|e| e.to_string())?;
                bindings.push(
                    state
                        .retained_dependency(
                            invocation,
                            projection.weight_input_ordinal(),
                            &part.component_id,
                            offset,
                        )
                        .map_err(|e| e.to_string())?,
                );
                Some(state)
            } else {
                None
            };
            Ok(Leaf {
                weight,
                part: part.clone(),
                stage,
                validation,
            })
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn launch_leaves(
    g32: Option<(
        &Option<Q8ActKernels>,
        &ferrum_interfaces::vnext::PreparedProjection,
        &[weights::MatrixPart],
    )>,
    kernels: &CudaNativeBlockKernels,
    stream: &CudaStream,
    regions: &[CudaBufferRegion],
    leaves: &[Leaf],
    input: usize,
    output: usize,
    rows: u32,
    stride: u32,
    workspace: DeviceSpan,
    transform: u64,
    first_weight: usize,
) -> Result<(), CudaDeviceRuntimeError> {
    if let Some((g32, projection, parts)) = g32 {
        let g32 = g32
            .as_ref()
            .ok_or_else(|| CudaDeviceRuntimeError::contract("hybrid G32 kernels absent"))?;
        let pointers = regions[first_weight..first_weight + weights::region_count(parts)]
            .iter()
            .map(CudaBufferRegion::device_ptr)
            .collect::<Vec<_>>();
        return g32.launch(
            kernels,
            stream,
            projection,
            parts,
            &pointers,
            regions[input].device_ptr(),
            regions[output].device_ptr(),
            rows,
            stride,
            workspace.address,
            workspace.bytes,
            transform,
        );
    }
    let span = |index: usize| DeviceSpan {
        address: regions[index].device_ptr(),
        bytes: regions[index].length_bytes(),
    };
    for leaf in leaves {
        if let Some(validation) = &leaf.validation {
            // The Arc is retained unconditionally by this captured command.
            // A dynamic prelude always orders the real device scan/event.
            unsafe {
                leaf.stage.launch(
                    stream,
                    span(input),
                    span(leaf.weight),
                    span(output),
                    workspace,
                    validation.flag_span(),
                )?;
            }
        } else {
            kernels.transformed_linear(
                stream,
                regions[input].device_ptr(),
                regions[leaf.weight].device_ptr(),
                regions[output].device_ptr(),
                &leaf.part,
                rows,
                stride,
                ElementType::F16,
                leaf.part
                    .signs_region
                    .map_or(0, |index| regions[first_weight + index].device_ptr()),
                transform,
            )?;
        }
    }
    Ok(())
}

fn persistent_region(
    invocation: &BatchedOperationInvocation<'_, CudaDeviceBuffer>,
    bytes: u64,
) -> Result<CudaBufferRegion, String> {
    let resolve = |participant: &OperationInvocation<'_, CudaDeviceBuffer>| {
        let view = participant
            .persistent_view()
            .ok_or("upstream FFN has no admitted validation storage")?;
        if view.descriptor().element_type != ElementType::U8 || view.descriptor().size_bytes < bytes
        {
            return Err(format!(
                "upstream validation storage needs {bytes} U8 bytes, admitted {:?} {} bytes",
                view.descriptor().element_type,
                view.descriptor().size_bytes,
            ));
        }
        // Static plan storage includes the selected allocator's alignment or
        // block quantum. Only the required prefix belongs to these flags.
        let parts = view.translate(0, bytes).map_err(|e| e.to_string())?;
        let mut physical = parts.iter();
        let part = physical
            .next()
            .ok_or("upstream validation storage is absent")?;
        if physical.next().is_some() {
            return Err("upstream validation storage must be contiguous".into());
        }
        let (buffer, range, retention) = part.buffer_and_physical_range();
        buffer
            .retained_region(range, retention)
            .map_err(|e| e.to_string())
    };
    let first = resolve(&invocation.participants()[0])?;
    for participant in &invocation.participants()[1..] {
        if !same_physical_region(&first, &resolve(participant)?) {
            return Err("upstream validation storage is not shared by the plan".into());
        }
    }
    Ok(first)
}
fn mul(a: u64, b: u64) -> Result<u64, String> {
    a.checked_mul(b)
        .ok_or_else(|| "upstream extent overflows".into())
}
fn add(a: u64, b: u64) -> Result<u64, String> {
    a.checked_add(b)
        .ok_or_else(|| "upstream extent overflows".into())
}
