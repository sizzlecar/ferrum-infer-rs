//! The packed-head predicate consumes actual projected physical ranges; a
//! plan-only Bn decode query cannot infer contiguity or non-aliasing.
use super::*;
use ferrum_interfaces::vnext::{
    OperationCostRoute, OperationCostRouteRequest, PhysicalWeightLayout,
};

pub(in crate::backend::cuda::vnext_ops) fn projection(
    request: OperationCostRouteRequest<'_>,
    precision: TokenPrecision,
    capture: ferrum_types::SloStructuredCostCapture,
) -> Result<Option<OperationCostRoute>, VNextError> {
    let checked = || -> Result<Option<OperationCostRoute>, String> {
        if request.operation_id().as_str() != precision.projection_operation() {
            return Err("CUDA head cost operation mismatch".into());
        }
        let hidden = unsigned_attribute(request.attributes(), "hidden_size")?;
        let outputs = unsigned_attribute(request.attributes(), "out_features")?;
        let input = binding(request.bindings(), ResolvedValueRole::Input, 0)?;
        let table = binding(request.bindings(), ResolvedValueRole::Input, 1)?;
        let output = binding(request.bindings(), ResolvedValueRole::Output, 0)?;
        validate_last_token_dense_linear_signature(
            input,
            table,
            output,
            hidden,
            outputs,
            precision.element(),
        )?;
        let weight = table.weight().ok_or("CUDA head weight metadata absent")?;
        if precision != TokenPrecision::F32
            && matches!(weight.physical_layout(), PhysicalWeightLayout::Dense { .. })
        {
            return Ok(None);
        }
        let parts = weights::matrix_parts(weight, &[outputs, hidden])?;
        u32::try_from(outputs).map_err(|_| "native projection output stride overflows")?;
        let input_packed = request
            .binding_uses_packed_batch_coordinates(ResolvedValueRole::Input, 0)
            .map_err(|error| error.to_string())?;
        let mut end = 0;
        let token_ranges = request
            .rows()
            .iter()
            .map(|row| {
                let start = end;
                end += row.count.get();
                start..end
            })
            .collect::<Vec<_>>();
        let may_pack = packed_projection_semantics(
            input_packed,
            &parts,
            token_ranges.len(),
            token_ranges
                .iter()
                .enumerate()
                .all(|(index, range)| *range == (index as u64..index as u64 + 1)),
        );
        let packed = if may_pack {
            let mut weights = Vec::with_capacity(table.storage().components().len());
            for (index, component) in table.storage().components().iter().enumerate() {
                let end = component
                    .offset_bytes()
                    .checked_add(component.length_bytes())
                    .ok_or("CUDA head weight extent overflows")?;
                let Some(range) = request
                    .binding_physical_range(
                        ResolvedValueRole::Input,
                        1,
                        index,
                        0,
                        component.offset_bytes()..end,
                    )
                    .map_err(|error| error.to_string())?
                else {
                    return Ok(None);
                };
                weights.push(range.start()..range.end());
            }
            let [input_component] = input.storage().components() else {
                return Ok(None);
            };
            let [output_component] = output.storage().components() else {
                return Ok(None);
            };
            let bytes = hidden
                .checked_mul(precision.element().size_bytes())
                .ok_or("CUDA head input row overflows")?;
            if input_component.offset_bytes() != 0 {
                return Ok(None);
            }
            let mut rows = Vec::with_capacity(request.rows().len());
            for (index, row) in request.rows().iter().enumerate() {
                let start = (row.count.get() - 1)
                    .checked_mul(bytes)
                    .ok_or("CUDA head row offset overflows")?;
                let Some(input) = request
                    .binding_physical_range(
                        ResolvedValueRole::Input,
                        0,
                        0,
                        index,
                        start
                            ..start
                                .checked_add(bytes)
                                .ok_or("CUDA head row end overflows")?,
                    )
                    .map_err(|error| error.to_string())?
                else {
                    return Ok(None);
                };
                let start = output_component.offset_bytes();
                let Some(output) = request
                    .binding_physical_range(
                        ResolvedValueRole::Output,
                        0,
                        0,
                        index,
                        start
                            ..start
                                .checked_add(output_component.length_bytes())
                                .ok_or("CUDA head output extent overflows")?,
                    )
                    .map_err(|error| error.to_string())?
                else {
                    return Ok(None);
                };
                rows.push(NativeProjectionRow {
                    input: input.start(),
                    input_bytes: input.length(),
                    output: output.start(),
                    output_bytes: output.length(),
                });
            }
            packed_projection_rows(
                precision,
                input_packed,
                hidden,
                outputs,
                &parts,
                token_ranges.into_iter(),
                rows.into_iter(),
                &weights,
            )
            .is_some()
        } else {
            false
        };
        let participants =
            u32::try_from(request.rows().len()).map_err(|_| "CUDA head participants overflow")?;
        let dispatches = weights::dispatches(&parts)
            .checked_mul(if packed { 1 } else { u64::from(participants) })
            .ok_or("native projection dispatch count overflows")?;
        let command = super::super::cost_route::compute(
            "vnext_native_last_token_linear",
            if packed {
                DeviceBatchingForm::Packed
            } else {
                DeviceBatchingForm::ParticipantLoop
            },
            participants,
            u64::from(participants),
            dispatches,
        )
        .map_err(|error| error.to_string())?;
        let selected = if capture == ferrum_types::SloStructuredCostCapture::Disabled {
            None
        } else {
            super::super::native_blocks::selected::linear(
                &parts,
                (0..if packed { 1 } else { participants }).map(|_| {
                    if packed {
                        participants
                    } else {
                        1
                    }
                }),
                u64::from(participants),
                outputs as u32, // checked above before physical projection
                precision.element(),
                super::super::native_blocks::hadamard::workspace_bytes_per_token(
                    request.bindings(),
                )?
                .checked_mul(request.immediate_tokens())
                .ok_or("CUDA head transform scratch overflows")?,
                capture,
            )
        };
        let command = match selected {
            Some(evidence) => command
                .clone()
                .with_statistical_evidence(evidence)
                .unwrap_or(command),
            None => command,
        };
        OperationCostRoute::new(vec![command])
            .map(Some)
            .map_err(|error| error.to_string())
    };
    checked().map_err(|reason| VNextError::InvalidExecutionPlan { reason })
}
