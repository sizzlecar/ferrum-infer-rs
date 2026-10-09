use super::super::native_blocks::upstream_linear::WeightValidation;
use super::*;

pub(super) fn encode(
    provider: &CudaQ6MmqF32LastTokenProvider,
    invocation: BatchedOperationInvocation<'_, CudaDeviceBuffer>,
) -> Result<EncodedDeviceOperation<CudaDeviceCommand>, String> {
    let precision = TokenPrecision::F32;
    transformer::ensure_invocation(&invocation, OPERATION)?;
    let first = &invocation.participants()[0];
    let hidden = unsigned_attribute(first.attributes(), "hidden_size")?;
    let outputs = unsigned_attribute(first.attributes(), "out_features")?;
    let weight = native_io::retain_shared_weight(&invocation, &[outputs, hidden])?;
    let input_packed =
        transformer::token_binding_is_packed(&invocation, ResolvedValueRole::Input, 0)?;
    let mut key = native_io::matrix_key(
        provider.descriptor.provider_implementation_fingerprint(),
        "vnext_q6_f32_last_token_linear",
        &weight,
    );
    let mut regions = weight.regions;
    let weight_region_count = regions.len();
    let scratch = native_blocks::hadamard::retain_workspace(&invocation, &mut regions)?;
    let mut launches = Vec::new();
    for (participant, range) in invocation
        .participants()
        .iter()
        .zip(invocation.participant_token_ranges())
    {
        let input = binding(participant.bindings(), ResolvedValueRole::Input, 0)?;
        let table = binding(participant.bindings(), ResolvedValueRole::Input, 1)?;
        let output = binding(participant.bindings(), ResolvedValueRole::Output, 0)?;
        if unsigned_attribute(participant.attributes(), "hidden_size")? != hidden
            || unsigned_attribute(participant.attributes(), "out_features")? != outputs
        {
            return Err("CUDA native projection participant dimensions disagree".into());
        }
        validate_last_token_dense_linear_signature(
            input,
            table,
            output,
            hidden,
            outputs,
            precision.element(),
        )?;
        let selected = if input_packed {
            range.immediate_token_range()
        } else {
            range.source_token_range()
        };
        let last = native_io::last_token(selected)?;
        let input_index = regions.len();
        regions.push(contiguous_token_region(
            participant,
            input,
            precision.element(),
            last,
            1,
        )?);
        let output_index = regions.len();
        regions.push(contiguous_region(participant, output, precision.element())?);
        launches.push((input_index, output_index, 1_u32));
        key = key.u64(input_index as u64).u64(output_index as u64);
    }
    let participants =
        u32::try_from(launches.len()).map_err(|_| "too many projection participants")?;
    let weight_ranges = regions[..weight_region_count]
        .iter()
        .map(|region| {
            let start = region.device_ptr();
            Some(start..start.checked_add(region.length_bytes())?)
        })
        .collect::<Option<Vec<_>>>();
    let packed_rows = weight_ranges.as_ref().and_then(|weight_ranges| {
        native_io::packed_projection_rows(
            precision,
            input_packed,
            hidden,
            outputs,
            &weight.parts,
            invocation
                .participant_token_ranges()
                .iter()
                .map(|range| range.immediate_token_range()),
            launches
                .iter()
                .map(|&(input, output, _)| native_io::NativeProjectionRow {
                    input: regions[input].device_ptr(),
                    input_bytes: regions[input].length_bytes(),
                    output: regions[output].device_ptr(),
                    output_bytes: regions[output].length_bytes(),
                }),
            weight_ranges,
        )
    });
    if let Some(rows) = packed_rows {
        // Every original region remains retained by the command, including
        // rows addressed relative to the first participant's pointer.
        launches.truncate(1);
        launches[0].2 = rows;
    }
    key = key
        .boolean(packed_rows.is_some())
        .u32(packed_rows.unwrap_or(1));
    let stride = u32::try_from(outputs).map_err(|_| "native projection output stride overflows")?;
    let flags_bytes = (weight.parts.len() as u64)
        .checked_mul(4)
        .ok_or("Q6 flag extent overflow")?
        .max(4);
    let persistent = persistent_region(&invocation, flags_bytes)?;
    let workspace_index = regions.len();
    let required = provider.maximum_scratch(&weight.parts)?;
    regions.push(transformer::shared_scratch_region(&invocation, required)?);
    // Preserve the complete Plan lease even for currently strict geometry.
    regions.push(persistent.clone());
    let mut dependency_parts = BTreeSet::new();
    let mut dependencies = Vec::new();
    let mut plans: Vec<Vec<Option<(Arc<PreparedQ6F32Linear>, Arc<WeightValidation>)>>> = Vec::new();
    let mut dispatches = 0u64;
    let mut transfers = 0u64;
    for &(_, _, rows) in &launches {
        let mut selected = Vec::new();
        for (index, part) in weight.parts.iter().enumerate() {
            // `resolve` has already checked the complete live physical view.
            // Alignment is a declared layout limit, never a test of liveness.
            let route = provider
                .policy
                .route(rows, eligible(part) && regions[index].device_ptr() % 4 == 0)?;
            key = key.u32(rows).boolean(route == Q6MmqF32Route::Mmq);
            if route == Q6MmqF32Route::Mmq {
                let plan = provider.plan(rows, part)?;
                let offset = (index as u64)
                    .checked_mul(4)
                    .ok_or("Q6 flag offset overflow")?;
                let state = provider
                    .validation
                    .prepare_q6(
                        plan.clone(),
                        provider.descriptor.provider_implementation_fingerprint(),
                        regions[index].clone(),
                        persistent.subregion(offset, 4).map_err(|e| e.to_string())?,
                    )
                    .map_err(|e| e.to_string())?;
                if dependency_parts.insert(index) {
                    dependencies.push(
                        state
                            .retained_dependency(&invocation, 1, &part.component_id, offset)
                            .map_err(|e| e.to_string())?,
                    );
                }
                let g = plan.geometry();
                key = key
                    .u32(g.j)
                    .u32(g.blocks)
                    .u32(g.fixup)
                    .u32(g.guard_blocks)
                    .u64(g.packed_bytes)
                    .u64(g.fixup_bytes)
                    .u64(plan.workspace_bytes());
                // copy/pad + quantize + main + optional fixup + publish;
                // rowflag reset and optional guard reset are CUDA memsets.
                dispatches += 4 + u64::from(g.fixup != 0);
                transfers += 1 + u64::from(g.guard_blocks != 0);
                selected.push(Some((plan, state)));
            } else {
                dispatches += 1 + u64::from(part.transform.is_some());
                selected.push(None);
            }
        }
        plans.push(selected);
    }
    let kernels = provider.strict.clone();
    let command = CudaDeviceCommand::replayable_operation(
        "vnext_q6_f32_last_token_linear",
        regions,
        key.finish(),
        move |stream, regions| {
            for ((input, output, rows), plans) in launches.iter().copied().zip(&plans) {
                for (index, part) in weight.parts.iter().enumerate() {
                    if let Some((plan, state)) = &plans[index] {
                        // All merged row leases remain in `regions`. Their exact
                        // adjacency and extents were proved before truncating launches.
                        let input_bytes = u64::from(rows)
                            .checked_mul(u64::from(part.columns))
                            .and_then(|n| n.checked_mul(4))
                            .ok_or_else(|| {
                                CudaDeviceRuntimeError::contract("Q6 input extent overflow")
                            })?;
                        let output_offset = u64::from(part.output_offset) * 4;
                        let output_bytes = (u64::from(rows) - 1)
                            .checked_mul(u64::from(stride))
                            .and_then(|n| n.checked_add(u64::from(part.rows)))
                            .and_then(|n| n.checked_mul(4))
                            .ok_or_else(|| {
                                CudaDeviceRuntimeError::contract("Q6 output extent overflow")
                            })?;
                        let output_address = regions[output]
                            .device_ptr()
                            .checked_add(output_offset)
                            .ok_or_else(|| {
                                CudaDeviceRuntimeError::contract("Q6 output offset overflow")
                            })?;
                        unsafe {
                            plan.launch(
                                DeviceSpan {
                                    address: regions[input].device_ptr(),
                                    bytes: input_bytes,
                                },
                                part.columns,
                                DeviceSpan {
                                    address: regions[index].device_ptr(),
                                    bytes: regions[index].length_bytes(),
                                },
                                DeviceSpan {
                                    address: output_address,
                                    bytes: output_bytes,
                                },
                                stride,
                                DeviceSpan {
                                    address: regions[workspace_index].device_ptr(),
                                    bytes: regions[workspace_index].length_bytes(),
                                },
                                state.flag_span(),
                                stream.cu_stream().cast(),
                            )
                        }
                        .map_err(|e| CudaDeviceRuntimeError::contract(e.to_string()))?;
                    } else {
                        kernels.transformed_linear(
                            stream,
                            regions[input].device_ptr(),
                            regions[index].device_ptr(),
                            regions[output].device_ptr(),
                            part,
                            rows,
                            stride,
                            ElementType::F32,
                            part.signs_region.map_or(0, |i| regions[i].device_ptr()),
                            scratch.map_or(0, |i| regions[i].device_ptr()),
                        )?;
                    }
                }
            }
            Ok(())
        },
    )
    .and_then(|command| {
        command.with_work_attribution(
            if packed_rows.is_some() {
                DeviceBatchingForm::Packed
            } else {
                DeviceBatchingForm::ParticipantLoop
            },
            participants,
            u64::from(participants),
            dispatches,
            transfers,
        )
    })
    .map_err(|e| e.to_string())?;
    let mut operation = EncodedDeviceOperation::compute(command);
    for dependency in dependencies {
        operation = operation.with_retained_plan_dependency(dependency);
    }
    Ok(operation)
}

fn persistent_region(
    invocation: &BatchedOperationInvocation<'_, CudaDeviceBuffer>,
    bytes: u64,
) -> Result<CudaBufferRegion, String> {
    let resolve = |participant: &OperationInvocation<'_, CudaDeviceBuffer>| {
        let view = participant
            .persistent_view()
            .ok_or("Q6 head has no admitted validation storage")?;
        if view.descriptor().element_type != ElementType::U8 || view.descriptor().size_bytes < bytes
        {
            return Err(format!(
                "Q6 validation storage needs {bytes} U8 bytes, admitted {:?} {} bytes",
                view.descriptor().element_type,
                view.descriptor().size_bytes,
            ));
        }
        // Static plan storage includes the selected allocator's alignment or
        // block quantum. Only the required prefix belongs to these flags.
        let parts = view.translate(0, bytes).map_err(|e| e.to_string())?;
        let mut physical = parts.iter();
        let part = physical.next().ok_or("Q6 validation storage is absent")?;
        if physical.next().is_some() {
            return Err("Q6 validation storage must be contiguous".into());
        }
        let (buffer, range, retention) = part.buffer_and_physical_range();
        buffer
            .retained_region(range, retention)
            .map_err(|e| e.to_string())
    };
    let first = resolve(&invocation.participants()[0])?;
    for participant in &invocation.participants()[1..] {
        if !same_physical_region(&first, &resolve(participant)?) {
            return Err("Q6 validation storage is not shared by the plan".into());
        }
    }
    Ok(first)
}
