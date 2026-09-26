//! Complete dense FFN API sequence. Actual, future and replay evidence share
//! the checked dimensions and the original GemmEx/SiLU launch helpers.
use super::*;
use cublas_api::{CublasHandleApiIdentity, GemmF16ApiPlan};
use ferrum_interfaces::execution_cost::SelectedCommandCostEvidenceV1;
use ferrum_interfaces::vnext::{
    DeviceCommandPhase, OperationCostCommand, OperationCostRoute, OperationCostRouteRequest,
};
use ferrum_types::SloStructuredCostCapture;

#[derive(Clone, Copy)]
struct Shape {
    tokens: u64,
    rows: i32,
    hidden: i32,
    intermediate: i32,
    activation_elements: u64,
    gate_up_bytes: u64,
    scratch_bytes: u64,
    gate: GemmF16ApiPlan,
    down: GemmF16ApiPlan,
}

impl Shape {
    fn new(tokens: u64, hidden: u64, intermediate: u64) -> Result<Self, String> {
        let activation_elements = tokens
            .checked_mul(intermediate)
            .ok_or("dense SwiGLU activation element count overflows")?;
        let gate_up_bytes = activation_elements
            .checked_mul(4)
            .ok_or("dense SwiGLU gate/up scratch size overflows")?;
        let scratch_bytes = activation_elements
            .checked_mul(2)
            .and_then(|bytes| bytes.checked_add(gate_up_bytes))
            .ok_or("dense SwiGLU total scratch size overflows")?;
        let rows = checked_i32(tokens, "dense SwiGLU token count")?;
        let hidden = checked_i32(hidden, "dense SwiGLU hidden size")?;
        let intermediate = checked_i32(intermediate, "dense SwiGLU intermediate size")?;
        let width = intermediate
            .checked_mul(2)
            .ok_or("dense SwiGLU packed width overflows i32")?;
        // The same bounds used by the real launch, before any cost can be Known.
        silu_mul_launch_config(activation_elements).map_err(|e| e.to_string())?;
        Ok(Self {
            tokens,
            rows,
            hidden,
            intermediate,
            activation_elements,
            gate_up_bytes,
            scratch_bytes,
            gate: GemmF16ApiPlan::new(rows, width, hidden).map_err(|e| e.to_string())?,
            down: GemmF16ApiPlan::new(rows, hidden, intermediate).map_err(|e| e.to_string())?,
        })
    }

    fn from_values(
        operation: &ferrum_interfaces::vnext::OperationId,
        values: &[ResolvedValueBinding],
        attributes: &BTreeMap<AttributeId, SemanticValue>,
        tokens: u64,
    ) -> Result<Self, String> {
        gguf_f16_projection::validate_values(operation, values)?;
        let hidden = unsigned_attribute(attributes, "hidden_size")?;
        let intermediate = unsigned_attribute(attributes, "intermediate_size")?;
        validate_dense_swiglu(
            binding(values, ResolvedValueRole::Input, 0)?,
            binding(values, ResolvedValueRole::Input, 1)?,
            binding(values, ResolvedValueRole::Input, 2)?,
            binding(values, ResolvedValueRole::Output, 0)?,
            hidden,
            intermediate,
        )?;
        Self::new(tokens, hidden, intermediate)
    }

    fn selected(
        self,
        capture: SloStructuredCostCapture,
        identity: Option<CublasHandleApiIdentity>,
    ) -> Option<SelectedCommandCostEvidenceV1> {
        // Disabled creates no builder, table, hash or per-command allocation.
        let identity = identity?;
        let mut builder =
            crate::backend::cuda::vnext_runtime::selected_cost::builder(capture, self.tokens)?;
        self.gate.append_selected(&mut builder, identity).ok()?;
        native_swiglu::append_silu(
            &mut builder,
            self.rows as u32,
            self.intermediate as u32,
            self.scratch_bytes,
        )?;
        self.down.append_selected(&mut builder, identity).ok()?;
        builder.finish().ok()
    }
}

struct Prepared {
    shape: Shape,
    participants: u32,
    regions: Vec<CudaBufferRegion>,
}

fn prepare(
    operation: &str,
    invocation: &BatchedOperationInvocation<'_, CudaDeviceBuffer>,
) -> Result<Prepared, String> {
    ensure_invocation(invocation, operation)?;
    let first = &invocation.participants()[0];
    let shape = Shape::from_values(
        &invocation.operation().id,
        first.bindings(),
        first.attributes(),
        invocation.work_shape().immediate_tokens(),
    )?;
    for participant in &invocation.participants()[1..] {
        let other = Shape::from_values(
            &invocation.operation().id,
            participant.bindings(),
            participant.attributes(),
            shape.tokens,
        )?;
        if other.hidden != shape.hidden || other.intermediate != shape.intermediate {
            return Err("CUDA dense SwiGLU participant attributes disagree".into());
        }
    }
    let scratch = shared_scratch_region(invocation, shape.scratch_bytes)?;
    let regions = vec![
        shared_token_region(
            invocation,
            ResolvedValueRole::Input,
            0,
            ElementType::F16,
            shape.tokens,
        )?,
        shared_full_region(invocation, ResolvedValueRole::Input, 1, ElementType::F16)?,
        shared_full_region(invocation, ResolvedValueRole::Input, 2, ElementType::F16)?,
        shared_token_region(
            invocation,
            ResolvedValueRole::Output,
            0,
            ElementType::F16,
            shape.tokens,
        )?,
        scratch,
    ];
    Ok(Prepared {
        shape,
        regions,
        participants: checked_u32(
            invocation.participants().len() as u64,
            "dense SwiGLU participant count",
        )?,
    })
}

pub(super) fn replay_evidence(
    operation: &str,
    capture: SloStructuredCostCapture,
    identity: Option<CublasHandleApiIdentity>,
    invocation: &BatchedOperationInvocation<'_, CudaDeviceBuffer>,
) -> Result<Option<SelectedCommandCostEvidenceV1>, VNextError> {
    if capture.is_disabled() || identity.is_none() {
        return Ok(None);
    }
    let prepared = prepare(operation, invocation).map_err(invalid_plan)?;
    Ok(prepared.shape.selected(capture, identity))
}

pub(super) fn route(
    request: &OperationCostRouteRequest<'_>,
    capture: SloStructuredCostCapture,
    identity: Option<CublasHandleApiIdentity>,
) -> Result<Option<OperationCostRoute>, VNextError> {
    if capture.is_disabled() || identity.is_none() {
        return Ok(None);
    }
    if request.operation_id().as_str()
        != ferrum_interfaces::vnext::DENSE_SWIGLU_GGUF_F16_WEIGHTS_OPERATION_ID
    {
        return Ok(None);
    }
    let shape = Shape::from_values(
        request.operation_id(),
        request.bindings(),
        request.attributes(),
        request.immediate_tokens(),
    )
    .map_err(invalid_plan)?;
    // Match the actual shared_token_region path; participant-local multi-row
    // storage does not get a guessed packed library route.
    if request.rows().len() > 1
        && (!request.binding_uses_packed_batch_coordinates(ResolvedValueRole::Input, 0)?
            || !request.binding_uses_packed_batch_coordinates(ResolvedValueRole::Output, 0)?)
    {
        return Ok(None);
    }
    let Some(evidence) = shape.selected(capture, identity) else {
        return Ok(None);
    };
    let command = OperationCostCommand::new(
        "vnext_dense_swiglu",
        DeviceCommandPhase::Compute,
        DeviceBatchingForm::Packed,
        0,
        checked_u32(request.rows().len() as u64, "dense SwiGLU participants")
            .map_err(invalid_plan)?,
        shape.tokens,
        3,
        0,
    )?
    .with_statistical_evidence(evidence)
    .map_err(|reason| invalid_plan(format!("CUDA dense SwiGLU cost evidence: {reason:?}")))?;
    Ok(Some(OperationCostRoute::new(vec![command])?))
}

pub(super) fn encode(
    operation: &str,
    fingerprint: &str,
    silu_mul: &CudaFunction,
    capture: SloStructuredCostCapture,
    identity: Option<CublasHandleApiIdentity>,
    invocation: BatchedOperationInvocation<'_, CudaDeviceBuffer>,
) -> Result<CudaDeviceCommand, String> {
    let rounded = gguf_f16_projection::is_operation(&invocation.operation().id);
    let Prepared {
        shape,
        participants,
        regions,
    } = prepare(operation, &invocation)?;
    let selected = rounded.then(|| shape.selected(capture, identity)).flatten();
    let silu_mul = silu_mul.clone();
    let replay_key = CudaCommandReplayKeyBuilder::new(fingerprint, "vnext_dense_swiglu")
        .i32(shape.rows)
        .i32(shape.hidden)
        .i32(shape.intermediate)
        .u64(shape.gate_up_bytes)
        .u64(shape.scratch_bytes)
        .finish();
    let command = CudaDeviceCommand::replayable_operation_with_blas(
        "vnext_dense_swiglu",
        regions,
        replay_key,
        move |stream, blas, regions| {
            let scratch = &regions[4];
            if scratch.length_bytes() < shape.scratch_bytes {
                return Err(CudaDeviceRuntimeError::contract(
                    "vNext dense SwiGLU scratch is smaller than its admitted estimate",
                ));
            }
            let gate_up_output = scratch.device_ptr();
            let activation = gate_up_output
                .checked_add(shape.gate_up_bytes)
                .ok_or_else(|| {
                    CudaDeviceRuntimeError::contract(
                        "vNext dense SwiGLU activation pointer overflows",
                    )
                })?;
            shape.gate.launch(
                blas,
                regions[0].device_ptr(),
                regions[1].device_ptr(),
                gate_up_output,
                "vNext dense SwiGLU gate/up GEMM",
            )?;
            launch_silu_mul(
                stream,
                &silu_mul,
                gate_up_output,
                activation,
                shape.intermediate,
                shape.activation_elements,
            )?;
            shape.down.launch(
                blas,
                activation,
                regions[2].device_ptr(),
                regions[3].device_ptr(),
                "vNext dense SwiGLU down GEMM",
            )?;
            Ok(())
        },
    )
    .and_then(|command| {
        command.with_work_attribution(DeviceBatchingForm::Packed, participants, shape.tokens, 3, 0)
    })
    .map_err(|e| e.to_string())?;
    Ok(if rounded {
        command
            .with_statistical_evidence(selected)
            .with_cublas_cost_requirement(identity)
    } else {
        command
    })
}

#[cfg(test)]
mod tests;
