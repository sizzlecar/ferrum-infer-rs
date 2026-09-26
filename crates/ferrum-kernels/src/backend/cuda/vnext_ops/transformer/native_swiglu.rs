//! Mixed native gate/up/down matrices with bounded F16 activation scratch.

use super::super::native_blocks::q8_f32scale::{self, PackLayout, Q8F32ScaleKernels, Q8SumPolicy};
use super::super::native_blocks::{
    stream_mmq::{self, StreamMmq},
    weights, CudaNativeBlockKernels,
};
use super::*;

mod cost_route;
mod prepared;
pub(super) mod replay_cost;
mod route_selection;
mod selected;
pub(super) use cost_route::{
    q8_route as eager_q8_cost_route, route as eager_cost_route,
    stream_mmq_route as eager_stream_mmq_cost_route,
};
pub(super) use selected::append_silu;

fn dispatches_per_launch(
    gate_up: &[weights::MatrixPart],
    down: &[weights::MatrixPart],
    q8: bool,
) -> Result<u64, String> {
    if !q8 {
        return Ok(weights::dispatches(gate_up) + weights::dispatches(down) + 1);
    }
    // Sum policy changes coefficients and scratch, never this launch count.
    let gate = q8_f32scale::matrix_plan_from_parts(gate_up, 1, Q8SumPolicy::Quantized)?;
    let down = q8_f32scale::matrix_plan_from_parts(down, 1, Q8SumPolicy::Quantized)?;
    gate.dispatches(1)?
        .checked_add(down.dispatches(1)?)
        .and_then(|count| count.checked_add(1))
        .ok_or_else(|| "Q8 SwiGLU dispatch count overflows".into())
}

pub(super) fn q8_workspace_per_token(values: &[ResolvedValueBinding]) -> Result<u64, String> {
    q8_workspace_per_token_with_policy(values, Q8SumPolicy::Quantized)
}

pub(super) fn q8_workspace_per_token_with_policy(
    values: &[ResolvedValueBinding],
    policy: Q8SumPolicy,
) -> Result<u64, String> {
    let mut bytes = 0;
    for ordinal in [1, 2] {
        let value = binding(values, ResolvedValueRole::Input, ordinal)?;
        let weight = value
            .weight()
            .ok_or("Q8 SwiGLU lacks physical matrix metadata")?;
        let parts = weights::matrix_parts(weight, value.tensor().dimensions())?;
        bytes = bytes.max(q8_part_workspace_per_token_with_policy(&parts, policy)?);
    }
    Ok(bytes)
}

#[cfg(test)]
fn q8_part_workspace_per_token(parts: &[weights::MatrixPart]) -> Result<u64, String> {
    q8_part_workspace_per_token_with_policy(parts, Q8SumPolicy::Quantized)
}

fn q8_part_workspace_per_token_with_policy(
    parts: &[weights::MatrixPart],
    policy: Q8SumPolicy,
) -> Result<u64, String> {
    Ok(q8_f32scale::matrix_plan_from_parts(parts, 1, policy)?.pack_bytes_per_row())
}

pub(super) fn uses_native(values: &[ResolvedValueBinding]) -> bool {
    values.iter().any(|value| {
        value.role() == ResolvedValueRole::Input
            && matches!(value.ordinal(), 1 | 2)
            && value.weight().is_some_and(|weight| {
                !matches!(
                    weight.physical_layout(),
                    ferrum_interfaces::vnext::PhysicalWeightLayout::Dense { .. }
                ) && weights::matrix_parts(weight, value.tensor().dimensions()).is_ok()
            })
    })
}

#[derive(Clone, Copy, Debug)]
struct ScratchLayout {
    gate_up_bytes: u64,
    activation_elements: u64,
    total_bytes: u64,
}

impl ScratchLayout {
    fn packed_rows(self, tokens: u64) -> Option<u32> {
        // The linear grid bounds rows, while SiLU and gate/up indexing use i32.
        // Keep larger, otherwise valid participant batches on the existing path.
        native_matrix::single_launch_rows(tokens)
            .filter(|_| self.gate_up_bytes / 2 <= i32::MAX as u64)
    }

    fn new(tokens: u64, intermediate: u64) -> Result<Self, String> {
        let activation_elements = tokens
            .checked_mul(intermediate)
            .filter(|&elements| elements != 0)
            .ok_or("native SwiGLU activation extent is zero or overflows")?;
        let gate_up_bytes = activation_elements
            .checked_mul(4)
            .ok_or("native SwiGLU gate/up bytes overflow")?;
        let total_bytes = activation_elements
            .checked_mul(6)
            .ok_or("native SwiGLU scratch bytes overflow")?;
        Ok(Self {
            gate_up_bytes,
            activation_elements,
            total_bytes,
        })
    }
}

pub(super) fn encode(
    fingerprint: &str,
    kernels: &CudaNativeBlockKernels,
    silu: &CudaFunction,
    capture: ferrum_types::SloStructuredCostCapture,
    invocation: BatchedOperationInvocation<'_, CudaDeviceBuffer>,
) -> Result<CudaDeviceCommand, String> {
    encode_with_q8(fingerprint, kernels, silu, None, capture, invocation)
}

pub(super) fn encode_with_q8(
    fingerprint: &str,
    kernels: &CudaNativeBlockKernels,
    silu: &CudaFunction,
    q8: Option<&Q8F32ScaleKernels>,
    capture: ferrum_types::SloStructuredCostCapture,
    invocation: BatchedOperationInvocation<'_, CudaDeviceBuffer>,
) -> Result<CudaDeviceCommand, String> {
    encode_with_policy(fingerprint, kernels, silu, q8, None, capture, invocation)
}

pub(super) fn encode_with_stream_mmq(
    fingerprint: &str,
    kernels: &CudaNativeBlockKernels,
    silu: &CudaFunction,
    mmq: &StreamMmq,
    capture: ferrum_types::SloStructuredCostCapture,
    invocation: BatchedOperationInvocation<'_, CudaDeviceBuffer>,
) -> Result<CudaDeviceCommand, String> {
    encode_with_policy(
        fingerprint,
        kernels,
        silu,
        None,
        Some(mmq),
        capture,
        invocation,
    )
}

pub(super) fn stream_mmq_workspace(
    values: &[ResolvedValueBinding],
    hidden: u32,
    intermediate: u32,
    mmq: &StreamMmq,
) -> Result<u64, String> {
    let value = binding(values, ResolvedValueRole::Input, 1)?;
    let weight = value
        .weight()
        .ok_or("Stream-MMQ requires native physical metadata")?;
    let parts = weights::matrix_parts(weight, value.tensor().dimensions())?;
    if stream_mmq::eligible(&parts, 8, hidden, intermediate) {
        let mut bytes = mmq.workspace(hidden, intermediate)?.total_bytes;
        if mmq.is_residual2() {
            let down = binding(values, ResolvedValueRole::Input, 2)?;
            let parts = weights::matrix_parts(
                down.weight().ok_or("Stream-MMQ down metadata missing")?,
                down.tensor().dimensions(),
            )?;
            if stream_mmq::eligible_q4_down(&parts, 8, intermediate, hidden) {
                bytes = bytes.max(mmq.workspace(intermediate, hidden)?.total_bytes);
            }
        }
        Ok(bytes)
    } else {
        Ok(0)
    }
}

fn encode_with_policy(
    fingerprint: &str,
    kernels: &CudaNativeBlockKernels,
    silu: &CudaFunction,
    q8: Option<&Q8F32ScaleKernels>,
    mmq: Option<&StreamMmq>,
    capture: ferrum_types::SloStructuredCostCapture,
    invocation: BatchedOperationInvocation<'_, CudaDeviceBuffer>,
) -> Result<CudaDeviceCommand, String> {
    let prepared = prepared::prepare(
        fingerprint,
        q8.map(Q8F32ScaleKernels::policy),
        mmq,
        capture,
        &invocation,
    )?;
    let recipe = if capture.is_disabled() {
        None
    } else {
        replay_cost::Recipe::from_prepared(&prepared, q8.map(Q8F32ScaleKernels::policy), mmq)
            .and_then(|numeric| {
                super::replay_cost::CudaReplayCostRecipe::native(&invocation, numeric)
            })
    };
    let prepared::Prepared {
        regions,
        gate_up,
        down,
        launches,
        scratch_index,
        scratch_layout,
        hidden,
        intermediate,
        transform_bytes,
        q8_bytes,
        mmq_bytes,
        selection,
        key,
    } = prepared;
    let native_name = selection.command.native_operation();
    let mmq_hit = selection.mmq_hit;
    let kernels = kernels.clone();
    let q8 = q8.cloned();
    let mmq = mmq.filter(|_| mmq_hit).cloned();
    let silu = silu.clone();
    CudaDeviceCommand::replayable_operation(native_name, regions, key, move |stream, regions| {
        let weights = regions[..scratch_index]
            .iter()
            .map(CudaBufferRegion::device_ptr)
            .collect::<Vec<_>>();
        let transform_scratch = regions[scratch_index].device_ptr();
        let scratch = transform_scratch
            .checked_add(transform_bytes)
            .and_then(|pointer| pointer.checked_add(q8_bytes))
            .and_then(|pointer| pointer.checked_add(mmq_bytes))
            .ok_or_else(|| CudaDeviceRuntimeError::contract("SwiGLU scratch pointer overflows"))?;
        for &(input, output, count, packed_start) in &launches {
            let gate_offset = packed_start
                .checked_mul(u64::from(intermediate))
                .and_then(|n| n.checked_mul(4))
                .ok_or_else(|| {
                    CudaDeviceRuntimeError::contract("native SwiGLU scratch offset overflows")
                })?;
            let activation_offset = packed_start
                .checked_mul(u64::from(intermediate))
                .and_then(|n| n.checked_mul(2))
                .and_then(|n| n.checked_add(scratch_layout.gate_up_bytes))
                .ok_or_else(|| {
                    CudaDeviceRuntimeError::contract("native SwiGLU activation offset overflows")
                })?;
            launch_with_policy(
                &kernels,
                &silu,
                stream,
                &gate_up,
                &down,
                &weights,
                regions[input].device_ptr(),
                regions[output].device_ptr(),
                scratch.checked_add(gate_offset).ok_or_else(|| {
                    CudaDeviceRuntimeError::contract("native gate/up pointer overflows")
                })?,
                scratch.checked_add(activation_offset).ok_or_else(|| {
                    CudaDeviceRuntimeError::contract("native activation pointer overflows")
                })?,
                count,
                hidden,
                intermediate,
                if transform_bytes > 0 {
                    transform_scratch
                } else {
                    0
                },
                q8.as_ref(),
                transform_scratch,
                mmq.as_ref(),
                transform_scratch
                    .checked_add(transform_bytes)
                    .and_then(|p| p.checked_add(q8_bytes))
                    .ok_or_else(|| {
                        CudaDeviceRuntimeError::contract("Stream-MMQ scratch pointer overflows")
                    })?,
            )?;
        }
        Ok(())
    })
    .and_then(|command| super::super::cost_route::apply(command, selection.command))
    .map(|command| command.with_replay_cost_recipe(recipe))
    .map_err(|error| error.to_string())
}

pub(super) fn replay_evidence(
    fingerprint: &str,
    q8: Option<Q8SumPolicy>,
    mmq: Option<&StreamMmq>,
    capture: ferrum_types::SloStructuredCostCapture,
    invocation: &BatchedOperationInvocation<'_, CudaDeviceBuffer>,
) -> Result<Option<ferrum_interfaces::execution_cost::SelectedCommandCostEvidenceV1>, VNextError> {
    if capture == ferrum_types::SloStructuredCostCapture::Disabled
        || !invocation
            .participants()
            .iter()
            .any(|p| uses_native(p.bindings()))
    {
        return Ok(None);
    }
    let prepared =
        prepared::prepare(fingerprint, q8, mmq, capture, invocation).map_err(invalid_plan)?;
    Ok(prepared.selection.command.statistical_evidence().cloned())
}

#[cfg(test)]
#[allow(clippy::too_many_arguments)]
fn launch(
    kernels: &CudaNativeBlockKernels,
    silu: &CudaFunction,
    stream: &CudaStream,
    gate_up: &[weights::MatrixPart],
    down: &[weights::MatrixPart],
    weights: &[u64],
    input: u64,
    output: u64,
    gate_up_output: u64,
    activation: u64,
    tokens: u32,
    hidden: u32,
    intermediate: u32,
    transform_scratch: u64,
) -> Result<(), CudaDeviceRuntimeError> {
    launch_with_policy(
        kernels,
        silu,
        stream,
        gate_up,
        down,
        weights,
        input,
        output,
        gate_up_output,
        activation,
        tokens,
        hidden,
        intermediate,
        transform_scratch,
        None,
        0,
        None,
        0,
    )
}

#[cfg(test)]
#[allow(clippy::too_many_arguments)]
fn launch_with_q8(
    kernels: &CudaNativeBlockKernels,
    silu: &CudaFunction,
    stream: &CudaStream,
    gate_up: &[weights::MatrixPart],
    down: &[weights::MatrixPart],
    weights: &[u64],
    input: u64,
    output: u64,
    gate_up_output: u64,
    activation: u64,
    tokens: u32,
    hidden: u32,
    intermediate: u32,
    transform_scratch: u64,
    q8: Option<&Q8F32ScaleKernels>,
    q8_scratch: u64,
) -> Result<(), CudaDeviceRuntimeError> {
    launch_with_policy(
        kernels,
        silu,
        stream,
        gate_up,
        down,
        weights,
        input,
        output,
        gate_up_output,
        activation,
        tokens,
        hidden,
        intermediate,
        transform_scratch,
        q8,
        q8_scratch,
        None,
        0,
    )
}

#[allow(clippy::too_many_arguments)]
fn launch_with_policy(
    kernels: &CudaNativeBlockKernels,
    silu: &CudaFunction,
    stream: &CudaStream,
    gate_up: &[weights::MatrixPart],
    down: &[weights::MatrixPart],
    weights: &[u64],
    input: u64,
    output: u64,
    gate_up_output: u64,
    activation: u64,
    tokens: u32,
    hidden: u32,
    intermediate: u32,
    transform_scratch: u64,
    q8: Option<&Q8F32ScaleKernels>,
    q8_scratch: u64,
    mmq: Option<&StreamMmq>,
    mmq_scratch: u64,
) -> Result<(), CudaDeviceRuntimeError> {
    let layout = ScratchLayout::new(u64::from(tokens), u64::from(intermediate))
        .map_err(CudaDeviceRuntimeError::contract)?;
    let doubled = intermediate
        .checked_mul(2)
        .ok_or_else(|| CudaDeviceRuntimeError::contract("native SwiGLU gate/up width overflows"))?;
    let intermediate_i32 =
        checked_i32_runtime(u64::from(intermediate), "native SwiGLU intermediate")?;
    checked_i32_runtime(
        layout.activation_elements,
        "native SwiGLU activation extent",
    )?;
    checked_i32_runtime(
        layout.gate_up_bytes / 2,
        "native SwiGLU gate/up indexing extent",
    )?;
    let down_start = weights::region_count(gate_up);
    if weights.len() != down_start + weights::region_count(down) {
        return Err(CudaDeviceRuntimeError::contract(
            "native SwiGLU matrix pointer inventory differs",
        ));
    }
    if let Some(mmq) = mmq {
        let launch = if mmq.is_residual2() {
            StreamMmq::launch_residual_gate_up
        } else {
            StreamMmq::launch_gate_up
        };
        launch(
            mmq,
            stream,
            gate_up,
            &weights[..down_start],
            input,
            gate_up_output,
            tokens,
            hidden,
            intermediate,
            mmq_scratch,
        )?;
    } else if let Some(q8) = q8 {
        q8.launch(
            kernels,
            stream,
            gate_up,
            &weights[..down_start],
            input,
            gate_up_output,
            tokens,
            hidden,
            doubled,
            q8_scratch,
        )?;
    } else {
        for (index, part) in gate_up.iter().enumerate() {
            kernels.transformed_linear(
                stream,
                input,
                weights[index],
                gate_up_output,
                part,
                tokens,
                doubled,
                ElementType::F16,
                part.signs_region.map_or(0, |index| weights[index]),
                transform_scratch,
            )?;
        }
    }
    launch_silu_mul(
        stream,
        silu,
        gate_up_output,
        activation,
        intermediate_i32,
        layout.activation_elements,
    )?;
    if let Some(mmq) = mmq.filter(|mmq| {
        mmq.is_residual2() && stream_mmq::eligible_q4_down(down, 8, intermediate, hidden)
    }) {
        mmq.launch_residual_q4_down(
            stream,
            down,
            &weights[down_start..],
            activation,
            output,
            tokens,
            intermediate,
            hidden,
            mmq_scratch,
        )?;
    } else if let Some(q8) = q8 {
        q8.launch(
            kernels,
            stream,
            down,
            &weights[down_start..],
            activation,
            output,
            tokens,
            intermediate,
            hidden,
            q8_scratch,
        )?;
    } else {
        for (index, part) in down.iter().enumerate() {
            kernels.transformed_linear(
                stream,
                activation,
                weights[down_start + index],
                output,
                part,
                tokens,
                hidden,
                ElementType::F16,
                part.signs_region
                    .map_or(0, |index| weights[down_start + index]),
                transform_scratch,
            )?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests;
