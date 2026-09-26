//! FP16 KV eager work from the production path and workspace selectors.
//! Only native or explicitly declared RN-F16 library projections are supported.
use super::*;
use ferrum_interfaces::vnext::{
    DeviceCommandPhase, OperationCostCommand, OperationCostRoute, OperationCostRouteRequest,
    ProgramBindingCostWrite,
};

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
    let checked = || -> Result<Option<OperationCostRoute>, String> {
        if request.operation_id().as_str() != operation {
            return Err("CUDA causal cost operation mismatch".into());
        }
        let rounded = super::super::gguf_f16_projection::is_operation(request.operation_id());
        super::super::gguf_f16_projection::validate_values(
            request.operation_id(),
            request.bindings(),
        )?;
        if rounded && (capture.is_disabled() || library_identity.is_none()) {
            return Ok(None);
        }
        let shape = CausalAttentionShape::from_attributes_for(request.attributes(), semantics)?;
        validate_signature_values(request.bindings(), shape, semantics, precision)?;
        if shape.int8_kv {
            return Ok(None);
        }
        let projection = CausalProjection::from_values(
            request.bindings(),
            #[cfg(feature = "vllm-marlin")]
            projection_runtime,
        )?;
        if !(matches!(projection, CausalProjection::Native { .. }) && !rounded
            || matches!(projection, CausalProjection::F16) && rounded)
        {
            return Ok(None);
        }
        let tokens = request.immediate_tokens();
        ScratchLayout::for_participants(shape, tokens, request.rows().len(), projection, policy)?;
        shape.cuda_shape()?;
        let layout = BindingLayout::new(shape, request.rows().len())?;
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
        let mut paths = Vec::with_capacity(request.rows().len());
        let mut writes = Vec::with_capacity(request.rows().len());
        let mut packed_start = 0_u64;
        let mut envelopes = Vec::with_capacity(request.rows().len());
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
            let path = CausalAttentionKernelPath::select(policy, shape, row.count.get(), end)?;
            let envelope = CausalAttentionReplayEnvelope::new(shape, path, end)?;
            envelopes.push(envelope.sequence_capacity_tokens);
            paths.push(path);
            packed_start = packed_start
                .checked_add(row.count.get())
                .ok_or("causal attention packed extent overflow")?;
        }
        if batch_decode::eligible(paths.iter().copied(), packed) {
            for range in batch_decode::group_ranges(&paths) {
                let binding = BindingLayout {
                    slot_bytes: layout.slot_bytes,
                    required_bytes: layout
                        .slot_bytes
                        .checked_mul(range.len() as u64)
                        .ok_or("causal cost group binding extent overflows")?,
                };
                batch_decode::BatchDecode::dimensions(
                    range.len(),
                    binding,
                    *envelopes[range]
                        .iter()
                        .max()
                        .ok_or("empty causal cost group")?,
                    shape.query_heads,
                    shape.head_dim,
                )?;
            }
        }
        let parts = if rounded {
            Vec::new()
        } else {
            (2..=5)
                .map(|ordinal| {
                    let value = binding(request.bindings(), ResolvedValueRole::Input, ordinal)?;
                    weights::matrix_parts(
                        value.weight().ok_or("causal projection metadata absent")?,
                        value.tensor().dimensions(),
                    )
                })
                .collect::<Result<Vec<_>, String>>()?
        };
        let extra = |rows| {
            parts.iter().try_fold(0_u64, |total, parts| {
                total
                    .checked_add(native_matrix_extra_dispatches(parts, rows)?)
                    .ok_or_else(|| "causal projection dispatch count overflows".to_owned())
            })
        };
        let extra = if packed {
            extra(tokens)?
        } else {
            request.rows().iter().try_fold(0_u64, |total, row| {
                total
                    .checked_add(extra(row.count.get())?)
                    .ok_or_else(|| "causal projection dispatch count overflows".to_owned())
            })?
        };
        let dispatches = physical_dispatch_count(
            paths.iter().copied(),
            shape.output_gate,
            shape.post_attention_norm,
            packed,
        )
        .checked_add(extra)
        .ok_or("causal compute dispatch count overflows")?;
        let operation = paths
            .first()
            .copied()
            .filter(|first| paths.iter().all(|path| path == first))
            .map(CausalAttentionKernelPath::operation)
            .unwrap_or(COMPUTE_MIXED_OPERATION);
        let participants = u32::try_from(request.rows().len())
            .map_err(|_| "CUDA causal participant count exceeds u32")?;
        let selected_compute = (|| {
            if capture == SloStructuredCostCapture::Disabled {
                return None;
            }
            let rows = request
                .rows()
                .iter()
                .enumerate()
                .map(|(index, row)| {
                    let end = row.offset.checked_add(row.count.get())?;
                    let inplace = if precision.inplace_residual_kernel().is_some() {
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
                    selected::Row::new(shape, policy, row.count.get(), end, inplace)
                })
                .collect::<Option<Vec<_>>>()?;
            selected::compute(
                shape,
                precision,
                projection,
                policy,
                if rounded {
                    selected::ProjectionWork::DenseF16(library_identity?)
                } else {
                    selected::ProjectionWork::Native([&parts[0], &parts[1], &parts[2], &parts[3]])
                },
                &rows,
                tokens,
                packed,
                capture,
            )
        })();
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
        OperationCostRoute::new(vec![binding, compute])
            .and_then(|route| route.with_relocatable_binding(0))
            .and_then(|route| route.with_program_binding_writes(writes))
            .map(Some)
            .map_err(|error| error.to_string())
    };
    checked().map_err(invalid_plan)
}
