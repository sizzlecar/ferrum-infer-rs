//! Eager routes use the same selection and occupancy-derived workspace bound
//! as actual native encoding. The original full-Q8 operation stays separate.
use super::*;
use ferrum_interfaces::vnext::{OperationCostRoute, OperationCostRouteRequest};

pub(in crate::backend::cuda::vnext_ops::transformer) fn route(
    request: OperationCostRouteRequest<'_>,
    capture: ferrum_types::SloStructuredCostCapture,
) -> Result<Option<OperationCostRoute>, VNextError> {
    route_with_policy(request, None, None, capture)
}

pub(in crate::backend::cuda::vnext_ops::transformer) fn stream_mmq_route(
    request: OperationCostRouteRequest<'_>,
    mmq: &StreamMmq,
    capture: ferrum_types::SloStructuredCostCapture,
) -> Result<Option<OperationCostRoute>, VNextError> {
    route_with_policy(request, Some(mmq), None, capture)
}

pub(in crate::backend::cuda::vnext_ops::transformer) fn q8_route(
    request: OperationCostRouteRequest<'_>,
    policy: Q8SumPolicy,
    capture: ferrum_types::SloStructuredCostCapture,
) -> Result<Option<OperationCostRoute>, VNextError> {
    route_with_policy(request, None, Some(policy), capture)
}

fn route_with_policy(
    request: OperationCostRouteRequest<'_>,
    mmq: Option<&StreamMmq>,
    q8: Option<Q8SumPolicy>,
    capture: ferrum_types::SloStructuredCostCapture,
) -> Result<Option<OperationCostRoute>, VNextError> {
    if !uses_native(request.bindings()) {
        return Ok(None);
    }
    let checked = || -> Result<_, String> {
        let operation = if let Some(mmq) = mmq {
            mmq.operation_id()
        } else if let Some(policy) = q8 {
            policy.operation_id()
        } else {
            DENSE_SWIGLU_OPERATION_ID
        };
        if request.operation_id().as_str() != operation {
            return Err("CUDA native SwiGLU cost operation mismatch".into());
        }
        let hidden = unsigned_attribute(request.attributes(), "hidden_size")?;
        let intermediate = unsigned_attribute(request.attributes(), "intermediate_size")?;
        let input = binding(request.bindings(), ResolvedValueRole::Input, 0)?;
        let gate_up = binding(request.bindings(), ResolvedValueRole::Input, 1)?;
        let down = binding(request.bindings(), ResolvedValueRole::Input, 2)?;
        let output = binding(request.bindings(), ResolvedValueRole::Output, 0)?;
        validate_dense_swiglu(input, gate_up, down, output, hidden, intermediate)?;
        let parts = |value: &ResolvedValueBinding, shape: &[u64]| {
            weights::matrix_parts(
                value
                    .weight()
                    .ok_or("CUDA SwiGLU matrix metadata missing")?,
                shape,
            )
        };
        let gate_up = parts(gate_up, &[2, intermediate, hidden])?;
        let down = parts(down, &[hidden, intermediate])?;
        let tokens = request.immediate_tokens();
        let scratch = ScratchLayout::new(tokens, intermediate)?;
        let transform_bytes =
            super::super::super::native_blocks::hadamard::workspace_bytes_per_token(
                request.bindings(),
            )?
            .checked_mul(tokens)
            .ok_or("CUDA SwiGLU transform scratch overflows")?;
        let mmq_bytes = if let Some(mmq) = mmq {
            stream_mmq_workspace(
                request.bindings(),
                checked_u32(hidden, "Stream-MMQ hidden")?,
                checked_u32(intermediate, "Stream-MMQ intermediate")?,
                mmq,
            )?
        } else {
            0
        };
        let q8_bytes = if let Some(policy) = q8 {
            q8_f32scale::matrix_plan_from_parts(&gate_up, tokens, policy)?
                .workspace_bytes(tokens)?
                .max(
                    q8_f32scale::matrix_plan_from_parts(&down, tokens, policy)?
                        .workspace_bytes(tokens)?,
                )
        } else {
            0
        };
        scratch
            .total_bytes
            .checked_add(transform_bytes)
            .and_then(|bytes| bytes.checked_add(mmq_bytes))
            .and_then(|bytes| bytes.checked_add(q8_bytes))
            .ok_or("CUDA SwiGLU scratch overflows")?;
        for row in request.rows() {
            let count = checked_u32(row.count.get(), "native SwiGLU tokens")?;
            if count == 0 || count > u16::MAX as u32 {
                return Err("CUDA native SwiGLU span exceeds launch extent".into());
            }
        }
        checked_u32(hidden, "native SwiGLU hidden")?;
        checked_u32(intermediate, "native SwiGLU intermediate")?;
        let input_packed = request
            .binding_uses_packed_batch_coordinates(ResolvedValueRole::Input, 0)
            .map_err(|error| error.to_string())?;
        let output_packed = request
            .binding_uses_packed_batch_coordinates(ResolvedValueRole::Output, 0)
            .map_err(|error| error.to_string())?;
        let participants = checked_u32(request.rows().len() as u64, "native SwiGLU participants")?;
        let selection = route_selection::select(
            &gate_up,
            &down,
            hidden,
            intermediate,
            tokens,
            participants,
            input_packed,
            output_packed,
            if let Some(mmq) = mmq {
                if mmq.is_residual2() {
                    route_selection::Arithmetic::Residual2M2To8
                } else {
                    route_selection::Arithmetic::StreamMmq
                }
            } else if q8.is_some() {
                route_selection::Arithmetic::Q8
            } else {
                route_selection::Arithmetic::Strict
            },
        )?;
        let rows = selection.packed_rows.into_iter().chain(
            request
                .rows()
                .iter()
                .filter(|_| selection.packed_rows.is_none())
                .map(|row| row.count.get() as u32),
        );
        let evidence = selected::swiglu(
            &gate_up,
            &down,
            rows,
            tokens,
            checked_u32(hidden, "native SwiGLU hidden")?,
            checked_u32(intermediate, "native SwiGLU intermediate")?,
            transform_bytes,
            q8_bytes,
            q8,
            mmq,
            selection.mmq_hit,
            capture,
        );
        Ok(selected::attach(selection.command, evidence))
    };
    Ok(Some(OperationCostRoute::new(vec![checked().map_err(
        |reason| VNextError::InvalidExecutionPlan { reason },
    )?])?))
}
