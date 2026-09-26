//! Pure recurrent route from the actual native/library, shape and scan selectors.
use super::*;
use ferrum_interfaces::vnext::{
    DeviceCommandPhase, OperationCostCommand, OperationCostRoute, OperationCostRouteRequest,
    ProgramBindingCostWrite,
};

pub(super) fn native_projection_dispatches(
    parts: &[weights::MatrixPart],
    rows: u64,
    quantized: bool,
) -> Result<u64, String> {
    if quantized {
        native_projection::q8_dispatch_count(parts, rows)
    } else {
        native_projection::strict_dispatch_count(parts, rows)
    }
}

pub(super) fn route(
    request: OperationCostRouteRequest<'_>,
    precision: AttentionPrecision,
    capabilities: GatedDeltaExecutionCapabilities,
    capture: ferrum_types::SloStructuredCostCapture,
    library_identity: Option<super::super::cublas_api::CublasHandleApiIdentity>,
    #[cfg(feature = "vllm-marlin")] projection_runtime: MarlinProjectionRuntime,
) -> Result<Option<OperationCostRoute>, VNextError> {
    let checked = || -> Result<Option<OperationCostRoute>, String> {
        if request.operation_id().as_str() != precision.operation() {
            return Err("CUDA recurrent cost operation mismatch".into());
        }
        let is_library = matches!(precision, AttentionPrecision::F32MasterGgufF16Projections);
        if is_library
            && (capture == ferrum_types::SloStructuredCostCapture::Disabled
                || library_identity.is_none())
        {
            return Ok(None);
        }
        super::super::gguf_f16_projection::validate_values(
            request.operation_id(),
            request.bindings(),
        )?;
        let shape = AttentionShape::from_attributes(request.attributes())?;
        validate_signature_values(request.bindings(), shape, precision)?;
        let projection = AttentionProjection::from_values(
            request.bindings(),
            precision,
            #[cfg(feature = "vllm-marlin")]
            projection_runtime,
        )?;
        if is_library {
            if !matches!(projection, AttentionProjection::F16) {
                return Ok(None);
            }
        } else if !matches!(
            projection,
            AttentionProjection::Native { .. } | AttentionProjection::NativeQ8 { .. }
        ) {
            return Ok(None);
        }
        let tokens = request.immediate_tokens();
        shape.validate_launch_extents(tokens)?;
        ScratchLayout::new(shape, tokens, request.rows().len(), projection)?;
        let binding_layout = StateBindingLayout::new(request.rows().len())?;
        let parts = |ordinal| -> Result<_, String> {
            let value = binding(request.bindings(), ResolvedValueRole::Input, ordinal)?;
            weights::matrix_parts(
                value
                    .weight()
                    .ok_or("CUDA recurrent projection metadata absent")?,
                value.tensor().dimensions(),
            )
        };
        let matrices = if is_library {
            None
        } else {
            Some((parts(2)?, parts(7)?))
        };
        let evidence = match &matrices {
            Some((input, output)) => selected::ProjectionEvidence::Native { input, output },
            None => selected::ProjectionEvidence::Library(
                library_identity.ok_or("RN-F16 cuBLAS handle identity is unavailable")?,
            ),
        };
        let packed = request.rows().len() > 1
            && request
                .binding_uses_packed_batch_coordinates(ResolvedValueRole::Input, 0)
                .map_err(|error| error.to_string())?
            && request
                .binding_uses_packed_batch_coordinates(ResolvedValueRole::Output, 0)
                .map_err(|error| error.to_string())?;
        let mut packed_form = None;
        for row in request.rows() {
            shape.validate_launch_extents(row.count.get())?;
            let form = capabilities
                .select(
                    row.count.get(),
                    GatedDeltaExecutionPreference::RecurrentScan,
                )
                .map_err(|error| error.to_string())?;
            if matches!(form, GatedDeltaExecutionForm::ChunkedScan(_)) {
                return Ok(None);
            }
            if packed
                && packed_form
                    .replace(form)
                    .is_some_and(|previous| previous != form)
            {
                return Ok(None);
            }
        }
        let native_count = |parts: &[weights::MatrixPart], count| {
            native_projection_dispatches(parts, count, precision.quantizes_projections())
        };
        let launch_count = if packed {
            1
        } else {
            request.rows().len() as u64
        };
        let launch_work = |count| {
            match &matrices {
                Some((input, output)) => combine_attention_dispatches(
                    native_count(input, count)?,
                    native_count(output, count)?,
                ),
                // Logical API calls, not an asserted vendor kernel launch count.
                None => combine_attention_dispatches(1, 1),
            }
        };
        let dispatches = if packed {
            launch_work(tokens)?
        } else {
            request.rows().iter().try_fold(0_u64, |total, row| {
                total
                    .checked_add(launch_work(row.count.get())?)
                    .ok_or_else(|| "CUDA recurrent cost dispatch count overflows".to_owned())
            })?
        };
        let transfers = launch_count
            .checked_mul(combine_attention_transfers(0, 0)?)
            .ok_or("CUDA recurrent cost transfers overflow")?;
        let participants = u32::try_from(request.rows().len())
            .map_err(|_| "CUDA recurrent cost participants exceed u32")?;
        let binding_command = OperationCostCommand::new(
            "vnext_gated_delta_recurrent_attention_bindings",
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
            "vnext_gated_delta_recurrent_attention",
            DeviceCommandPhase::Compute,
            if packed {
                DeviceBatchingForm::Packed
            } else {
                DeviceBatchingForm::ParticipantLoop
            },
            0,
            participants,
            tokens,
            dispatches,
            transfers,
        )
        .map_err(|error| error.to_string())?;
        let binding_command = selected::attach(
            binding_command,
            selected::bindings(request.rows().len(), tokens, capture),
        );
        let leaves = std::iter::once((tokens, participants, true))
            .take(usize::from(packed))
            .chain(
                request
                    .rows()
                    .iter()
                    .filter(move |_| !packed)
                    .map(|row| (row.count.get(), 1, false)),
            );
        let selected = selected::compute(
            shape,
            precision,
            projection,
            evidence,
            leaves,
            tokens,
            participants as usize,
            true,
            capture,
        );
        // A library route is complete or Unknown. Never emit a partial native
        // prefix while the required library selection/parameters are missing.
        if is_library && selected.is_none() {
            return Ok(None);
        }
        let compute = selected::attach(compute, selected);
        let writes = (0..request.rows().len())
            .map(|index| {
                ProgramBindingCostWrite::new(
                    binding_layout.offset(index).map_err(invalid_plan)?,
                    STATE_BINDING_SLOT_BYTES,
                )
            })
            .collect::<Result<Vec<_>, _>>()
            .map_err(|error| error.to_string())?;
        OperationCostRoute::new(vec![binding_command, compute])
            .and_then(|route| route.with_relocatable_binding(0))
            .and_then(|route| route.with_program_binding_writes(writes))
            .map(Some)
            .map_err(|error| error.to_string())
    };
    checked().map_err(invalid_plan)
}
