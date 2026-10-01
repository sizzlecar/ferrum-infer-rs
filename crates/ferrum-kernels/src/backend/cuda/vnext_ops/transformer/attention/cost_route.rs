//! Pure recurrent route from the actual native/library, shape and scan selectors.
use super::*;
use ferrum_interfaces::vnext::{
    DeviceCommandPhase, OperationCostCommand, OperationCostRoute, OperationCostRouteRequest,
    OperationCostSelection, OperationCostTopologyRequirement, OperationId, ProgramBindingCostWrite,
};

/// Per-node model geometry and weight metadata. No current token extent,
/// address, library-handle observation or selected dynamic work is retained.
pub(super) struct PreparedCostData {
    template: selected::PreparedCostTemplate,
    matrices: Option<(Vec<weights::MatrixPart>, Vec<weights::MatrixPart>)>,
}

impl PreparedCostData {
    pub(super) fn new(
        operation_id: &OperationId,
        bindings: &[ResolvedValueBinding],
        attributes: &BTreeMap<AttributeId, SemanticValue>,
        precision: AttentionPrecision,
        #[cfg(feature = "vllm-marlin")] projection_runtime: MarlinProjectionRuntime,
    ) -> Result<Option<Self>, String> {
        super::super::gguf_f16_projection::validate_values(operation_id, bindings)?;
        let shape = AttentionShape::from_attributes(attributes)?;
        validate_signature_values(bindings, shape, precision)?;
        let projection = AttentionProjection::from_values(
            bindings,
            precision,
            #[cfg(feature = "vllm-marlin")]
            projection_runtime,
        )?;
        let is_library = matches!(precision, AttentionPrecision::F32MasterGgufF16Projections);
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
        let parts = |ordinal| -> Result<_, String> {
            let value = binding(bindings, ResolvedValueRole::Input, ordinal)?;
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
        Ok(Some(Self {
            template: selected::PreparedCostTemplate::new(shape, precision, projection)?,
            matrices,
        }))
    }
}

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
    selection(
        request,
        OperationCostTopologyRequirement::NotRequested,
        &mut || Ok(()),
        precision,
        capabilities,
        capture,
        library_identity,
        #[cfg(feature = "vllm-marlin")]
        projection_runtime,
    )
    .map(|value| value.map(OperationCostSelection::into_route))
}

#[allow(clippy::too_many_arguments)]
pub(super) fn selection(
    request: OperationCostRouteRequest<'_>,
    topology: OperationCostTopologyRequirement,
    poll: &mut dyn FnMut() -> Result<(), VNextError>,
    precision: AttentionPrecision,
    capabilities: GatedDeltaExecutionCapabilities,
    capture: ferrum_types::SloStructuredCostCapture,
    library_identity: Option<super::super::cublas_api::CublasHandleApiIdentity>,
    #[cfg(feature = "vllm-marlin")] projection_runtime: MarlinProjectionRuntime,
) -> Result<Option<OperationCostSelection>, VNextError> {
    let mut checked = || -> Result<Option<OperationCostSelection>, String> {
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
        let fallback;
        let prepared = match request.prepared_cost_data::<PreparedCostData>() {
            Some(prepared) => prepared,
            None => {
                let Some(value) = PreparedCostData::new(
                    request.operation_id(),
                    request.bindings(),
                    request.attributes(),
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
        let tokens = request.immediate_tokens();
        let matrices = &prepared.matrices;
        let evidence = match matrices {
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
        let Some(query) = prepared.template.query(
            request.rows(),
            tokens,
            packed,
            capabilities,
            Some(evidence),
            topology == OperationCostTopologyRequirement::Required,
        )?
        else {
            return Ok(None);
        };
        let participants = query.participants();
        let dispatches = query.dispatches();
        let transfers = query.transfers();
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
        let selected = query.compute(capture);
        // A library route is complete or Unknown. Never emit a partial native
        // prefix while the required library selection/parameters are missing.
        if is_library && selected.is_none() {
            return Ok(None);
        }
        let compute = selected::attach(compute, selected);
        let writes = (0..request.rows().len())
            .map(|index| {
                ProgramBindingCostWrite::new(
                    query.binding_offset(index).map_err(invalid_plan)?,
                    STATE_BINDING_SLOT_BYTES,
                )
            })
            .collect::<Result<Vec<_>, _>>()
            .map_err(|error| error.to_string())?;
        let route = OperationCostRoute::new(vec![binding_command, compute])
            .and_then(|route| route.with_relocatable_binding(0))
            .and_then(|route| route.with_program_binding_writes(writes))
            .map_err(|error| error.to_string())?;
        let topology = match topology {
            OperationCostTopologyRequirement::NotRequested => None,
            OperationCostTopologyRequirement::Required => {
                poll().map_err(|error| error.to_string())?;
                Some(
                    query
                        .topology(&request)
                        .map_err(|error| error.to_string())?,
                )
            }
        };
        Ok(Some(OperationCostSelection::new(route, topology)))
    };
    checked().map_err(invalid_plan)
}
