//! FP16 KV eager work from the production path and workspace selectors.
//! Only native or explicitly declared RN-F16 library projections are supported.
use super::*;
use ferrum_interfaces::vnext::{
    DeviceCommandPhase, OperationCostCommand, OperationCostRoute, OperationCostRouteRequest,
    OperationCostSelection, OperationCostTopologyRequirement, ProgramBindingCostWrite,
};

mod prepared;
pub(super) use prepared::PreparedCostData;

pub(super) fn route(
    request: OperationCostRouteRequest<'_>,
    policy: AttentionExecutionPolicy,
    semantics: CausalAttentionSemantics,
    precision: CausalPrecision,
    capture: SloStructuredCostCapture,
    operation: &str,
    library_identity: Option<super::super::cublas_api::CublasHandleApiIdentity>,
    #[cfg(feature = "vllm-marlin")] projection_runtime: MarlinProjectionRuntime,
) -> Result<Option<OperationCostRoute>, VNextError> {
    selection(
        request,
        OperationCostTopologyRequirement::NotRequested,
        &mut || Ok(()),
        policy,
        semantics,
        precision,
        capture,
        operation,
        library_identity,
        #[cfg(feature = "vllm-marlin")]
        projection_runtime,
    )
    .map(|selected| selected.map(OperationCostSelection::into_route))
}

pub(super) fn selection(
    request: OperationCostRouteRequest<'_>,
    topology: OperationCostTopologyRequirement,
    poll: &mut dyn FnMut() -> Result<(), VNextError>,
    policy: AttentionExecutionPolicy,
    semantics: CausalAttentionSemantics,
    precision: CausalPrecision,
    capture: SloStructuredCostCapture,
    operation: &str,
    library_identity: Option<super::super::cublas_api::CublasHandleApiIdentity>,
    #[cfg(feature = "vllm-marlin")] projection_runtime: MarlinProjectionRuntime,
) -> Result<Option<OperationCostSelection>, VNextError> {
    let mut checked = || -> Result<Option<OperationCostSelection>, String> {
        if request.operation_id().as_str() != operation {
            return Err("CUDA causal cost operation mismatch".into());
        }
        let fallback;
        let prepared = match request.prepared_cost_data::<PreparedCostData>() {
            Some(prepared) => prepared,
            None => {
                let Some(value) = PreparedCostData::new(
                    request.operation_id(),
                    request.bindings(),
                    request.attributes(),
                    semantics,
                    precision,
                    #[cfg(feature = "vllm-marlin")]
                    projection_runtime,
                )?
                else {
                    return Ok(None);
                };
                fallback = value;
                &fallback
            }
        };
        let rounded = prepared.rounded;
        if rounded && (capture.is_disabled() || library_identity.is_none()) {
            return Ok(None);
        }
        let shape = prepared.shape;
        let tokens = request.immediate_tokens();
        let geometry = prepared
            .template
            .geometry(policy, tokens, request.rows().len())?;
        let layout = geometry.bindings;
        let packed = request.rows().len() > 1
            && request
                .binding_uses_packed_batch_coordinates(ResolvedValueRole::Input, 0)
                .map_err(|error| error.to_string())?
            && request
                .binding_uses_packed_batch_coordinates(ResolvedValueRole::Output, 0)
                .map_err(|error| error.to_string())?;
        if packed {
            checked_i32(tokens, "packed causal attention token count")?;
        }
        let mut rows = Vec::with_capacity(request.rows().len());
        let mut writes = Vec::with_capacity(request.rows().len());
        let mut packed_start = 0_u64;
        for (index, row) in request.rows().iter().enumerate() {
            let end = row
                .offset
                .checked_add(row.count.get())
                .ok_or("CUDA causal cost token end overflow")?;
            if end > row.full_input_tokens.get()
                || row.full_input_tokens.get() > shape.maximum_context_tokens
            {
                return Err("causal attention token range exceeds its admitted context".into());
            }
            shape.physical_state_bytes_for_source_frontier(end, row.full_input_tokens.get())?;
            checked_i32(row.offset, "causal attention position start")?;
            checked_i32(row.count.get(), "causal attention active tokens")?;
            checked_i32(end, "causal attention sequence tokens")?;
            checked_i32(packed_start, "causal attention packed token start")?;
            let entries = shape.table_entries(end)?;
            checked_i32(entries, "causal attention table entries")?;
            let bytes = entries
                .checked_mul(POINTER_BYTES)
                .and_then(|bytes| bytes.checked_add(BINDING_CONTROL_BYTES))
                .ok_or("causal attention binding payload overflow")?;
            if bytes > layout.slot_bytes {
                return Err("causal attention binding payload escapes its slot".into());
            }
            writes.push(
                ProgramBindingCostWrite::new(layout.binding_offset(index)?, bytes)
                    .map_err(|error| error.to_string())?,
            );
            rows.push(
                selected::Row::new(shape, policy, row.count.get(), end, false)
                    .ok_or("causal query row selection is invalid")?,
            );
            packed_start = packed_start
                .checked_add(row.count.get())
                .ok_or("causal attention packed extent overflow")?;
        }
        let parts = &prepared.parts;
        let operation = rows
            .first()
            .map(|row| row.path)
            .filter(|first| rows.iter().all(|row| row.path == *first))
            .map(CausalAttentionKernelPath::operation)
            .unwrap_or(COMPUTE_MIXED_OPERATION);
        let participants = u32::try_from(request.rows().len())
            .map_err(|_| "CUDA causal participant count exceeds u32")?;
        let projections = if rounded {
            selected::ProjectionWork::DenseF16(
                library_identity.ok_or("causal library identity absent")?,
            )
        } else {
            selected::ProjectionWork::Native([&parts[0], &parts[1], &parts[2], &parts[3]])
        };
        let alias_checked = (|| {
            if capture.is_disabled() {
                return Some(());
            }
            for (index, (row, selected)) in request.rows().iter().zip(&mut rows).enumerate() {
                selected.inplace = if precision.inplace_residual_kernel().is_some() {
                    let end = row.offset.checked_add(row.count.get())?;
                    let stride = shape
                        .hidden_size
                        .checked_mul(precision.hidden().size_bytes())?;
                    let range = row.offset.checked_mul(stride)?..end.checked_mul(stride)?;
                    let input = request
                        .binding_physical_range(
                            ResolvedValueRole::Input,
                            0,
                            0,
                            index,
                            range.clone(),
                        )
                        .ok()??;
                    let output = request
                        .binding_physical_range(ResolvedValueRole::Output, 0, 0, index, range)
                        .ok()??;
                    if input.allocation_id().is_some() != output.allocation_id().is_some() {
                        return None;
                    }
                    input == output
                } else {
                    false
                };
            }
            Some(())
        })()
        .is_some();
        let query = geometry
            .finish(&rows, packed, projections)
            .ok_or("causal query geometry or projection work is invalid")?;
        let dispatches = query.dispatches;
        let selected_compute = if alias_checked {
            query.compute(capture)
        } else {
            None
        };
        if rounded && selected_compute.is_none() {
            return Ok(None);
        }
        let selected_binding = selected::binding_evidence(
            writes.iter().map(|write| write.length_bytes()),
            tokens,
            capture,
        );
        let binding = OperationCostCommand::new(
            "vnext_causal_paged_attention_bindings",
            DeviceCommandPhase::DynamicBinding,
            DeviceBatchingForm::ParticipantLoop,
            0,
            participants,
            tokens,
            0,
            u64::from(participants),
        )
        .map_err(|error| error.to_string())?;
        let compute = OperationCostCommand::new(
            operation,
            DeviceCommandPhase::Compute,
            if packed {
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
        .map_err(|error| error.to_string())?;
        let binding = selected::attach(binding, selected_binding);
        let compute = selected::attach(compute, selected_compute);
        let route = OperationCostRoute::new(vec![binding, compute])
            .and_then(|route| route.with_relocatable_binding(0))
            .and_then(|route| route.with_program_binding_writes(writes))
            .map_err(|error| error.to_string())?;
        let topology = match topology {
            OperationCostTopologyRequirement::NotRequested => None,
            OperationCostTopologyRequirement::Required => {
                poll().map_err(|error| error.to_string())?;
                // The cost route already selected every kernel path from these
                // exact rows. Reuse that numerical choice inside this query,
                // but still prove every address captured by the graph.
                Some(
                    topology_from_prepared_selection(&request, semantics, shape, &rows)
                        .map_err(|error| error.to_string())?,
                )
            }
        };
        Ok(Some(OperationCostSelection::new(route, topology)))
    };
    checked().map_err(invalid_plan)
}

/// The prepared path shares both selection and envelope construction, while
/// every captured address still has to satisfy the current physical scope.
fn topology_from_prepared_selection(
    request: &impl ReusableExecutionTopologyView,
    semantics: CausalAttentionSemantics,
    shape: CausalAttentionShape,
    rows: &[selected::Row],
) -> Result<ReusableExecutionTopology, VNextError> {
    if rows.len() != request.participant_count() {
        return Err(invalid_plan(
            "CUDA causal prepared topology row count changed",
        ));
    }
    if !reusable_attention_address_scope(request, semantics)? {
        return Ok(ReusableExecutionTopology::EagerBoundary);
    }
    reusable_attention_topology_from_envelopes(
        shape,
        rows.len(),
        rows.iter().enumerate().map(|(index, selected)| {
            let row = request
                .token_row(index)
                .ok_or("CUDA causal topology row missing")?;
            let end = row
                .offset
                .checked_add(row.count.get())
                .ok_or("CUDA causal topology end overflow")?;
            if selected.tokens != row.count.get() || selected.sequence != end {
                return Err("CUDA causal prepared topology query changed".to_owned());
            }
            Ok((
                CausalAttentionTopologyRow {
                    active_tokens: selected.tokens,
                    sequence_tokens: selected.sequence,
                },
                selected.path,
                selected.envelope,
            ))
        }),
    )
    .map_err(invalid_plan)
}

/// Private to this provider's synchronous selection. `paths` are the choices
/// already checked while constructing this query's eager commands; no caller
/// can supply paths through the public cost/topology interface.
#[cfg(test)]
fn topology_from_cost_selection(
    request: &impl ReusableExecutionTopologyView,
    semantics: CausalAttentionSemantics,
    shape: CausalAttentionShape,
    paths: &[CausalAttentionKernelPath],
) -> Result<ReusableExecutionTopology, VNextError> {
    if paths.len() != request.participant_count() {
        return Err(invalid_plan(
            "CUDA causal selected topology row count changed",
        ));
    }
    if !reusable_attention_address_scope(request, semantics)? {
        return Ok(ReusableExecutionTopology::EagerBoundary);
    }
    reusable_attention_topology_from_selected(
        shape,
        paths.len(),
        paths.iter().copied().enumerate().map(|(index, path)| {
            let row = request
                .token_row(index)
                .ok_or("CUDA causal topology row is missing")?;
            Ok((
                CausalAttentionTopologyRow {
                    active_tokens: row.count.get(),
                    sequence_tokens: row
                        .offset
                        .checked_add(row.count.get())
                        .ok_or("CUDA causal topology token end overflow")?,
                },
                path,
            ))
        }),
    )
    .map_err(invalid_plan)
}
