//! Mixed native gate/up/down matrices with bounded F16 activation scratch.

use super::super::native_blocks::{weights, CudaNativeBlockKernels};
use super::*;

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
    invocation: BatchedOperationInvocation<'_, CudaDeviceBuffer>,
) -> Result<CudaDeviceCommand, String> {
    ensure_invocation(&invocation, DENSE_SWIGLU_OPERATION_ID)?;
    let first = &invocation.participants()[0];
    let hidden = unsigned_attribute(first.attributes(), "hidden_size")?;
    let intermediate = unsigned_attribute(first.attributes(), "intermediate_size")?;
    let mut matrices = Vec::new();
    for (ordinal, shape) in [
        (1, vec![2, intermediate, hidden]),
        (2, vec![hidden, intermediate]),
    ] {
        let matrix = weights::resolve(
            first,
            binding(first.bindings(), ResolvedValueRole::Input, ordinal)?,
            &shape,
        )?;
        for participant in &invocation.participants()[1..] {
            let candidate = weights::resolve(
                participant,
                binding(participant.bindings(), ResolvedValueRole::Input, ordinal)?,
                &shape,
            )?;
            if candidate.parts != matrix.parts
                || candidate.regions.len() != matrix.regions.len()
                || !candidate
                    .regions
                    .iter()
                    .zip(&matrix.regions)
                    .all(|(a, b)| same_physical_region(a, b))
            {
                return Err("CUDA native SwiGLU participants do not share physical weights".into());
            }
        }
        matrices.push(matrix);
    }
    let mut regions = Vec::new();
    let mut parts = Vec::new();
    let mut key = CudaCommandReplayKeyBuilder::new(fingerprint, "vnext_native_swiglu")
        .u64(hidden)
        .u64(intermediate);
    for matrix in matrices {
        key = key.u64(matrix.parts.len() as u64);
        for part in &matrix.parts {
            key = key.u32(part.rows).u32(part.columns).u32(part.output_offset);
            for parameter in part.format.parameters() {
                key = key.u32(parameter);
            }
        }
        regions.extend(matrix.regions);
        parts.push(matrix.parts);
    }
    let mut parts = parts.into_iter();
    let gate_up = parts.next().ok_or("missing native gate/up matrix")?;
    let down = parts.next().ok_or("missing native down matrix")?;
    let tokens = invocation.work_shape().immediate_tokens();
    let scratch_layout = ScratchLayout::new(tokens, intermediate)?;
    let scratch_index = regions.len();
    regions.push(shared_scratch_region(
        &invocation,
        scratch_layout.total_bytes,
    )?);
    let input_packed = token_binding_is_packed(&invocation, ResolvedValueRole::Input, 0)?;
    let output_packed = token_binding_is_packed(&invocation, ResolvedValueRole::Output, 0)?;
    if invocation.participant_token_ranges().len() != invocation.participants().len() {
        return Err("CUDA native SwiGLU participant ranges are incomplete".into());
    }
    let mut launches = Vec::new();
    for (participant, range) in invocation
        .participants()
        .iter()
        .zip(invocation.participant_token_ranges())
    {
        let input = binding(participant.bindings(), ResolvedValueRole::Input, 0)?;
        let output = binding(participant.bindings(), ResolvedValueRole::Output, 0)?;
        if unsigned_attribute(participant.attributes(), "hidden_size")? != hidden
            || unsigned_attribute(participant.attributes(), "intermediate_size")? != intermediate
        {
            return Err("CUDA native SwiGLU participant dimensions disagree".into());
        }
        validate_dense_swiglu(
            input,
            binding(participant.bindings(), ResolvedValueRole::Input, 1)?,
            binding(participant.bindings(), ResolvedValueRole::Input, 2)?,
            output,
            hidden,
            intermediate,
        )?;
        let count = checked_u32(range.immediate_tokens(), "native SwiGLU tokens")?;
        if count == 0 || count > u16::MAX as u32 {
            return Err("CUDA native SwiGLU token count exceeds its launch extent".into());
        }
        let packed = range.immediate_token_range();
        let source = range.source_token_range();
        if packed.end > tokens {
            return Err("native SwiGLU packed span exceeds scratch".into());
        }
        let input_index = regions.len();
        regions.push(contiguous_token_region(
            participant,
            input,
            ElementType::F16,
            if input_packed {
                packed.start
            } else {
                source.start
            },
            u64::from(count),
        )?);
        let output_index = regions.len();
        regions.push(contiguous_token_region(
            participant,
            output,
            ElementType::F16,
            if output_packed {
                packed.start
            } else {
                source.start
            },
            u64::from(count),
        )?);
        launches.push((input_index, output_index, count, packed.start));
        key = key
            .u64(input_index as u64)
            .u64(output_index as u64)
            .u32(count)
            .u64(packed.start);
    }
    let participants = checked_u32(launches.len() as u64, "native SwiGLU participants")?;
    let dispatches = (launches.len() as u64)
        .checked_mul((gate_up.len() + down.len() + 1) as u64)
        .ok_or("native SwiGLU dispatch count overflows")?;
    let hidden = checked_u32(hidden, "native SwiGLU hidden")?;
    let intermediate = checked_u32(intermediate, "native SwiGLU intermediate")?;
    key = key
        .u64(scratch_layout.gate_up_bytes)
        .u64(scratch_layout.total_bytes);
    let kernels = kernels.clone();
    let silu = silu.clone();
    CudaDeviceCommand::replayable_operation(
        "vnext_native_swiglu",
        regions,
        key.finish(),
        move |stream, regions| {
            let weights = regions[..scratch_index]
                .iter()
                .map(CudaBufferRegion::device_ptr)
                .collect::<Vec<_>>();
            let scratch = regions[scratch_index].device_ptr();
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
                        CudaDeviceRuntimeError::contract(
                            "native SwiGLU activation offset overflows",
                        )
                    })?;
                launch(
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
                )?;
            }
            Ok(())
        },
    )
    .and_then(|command| {
        command.with_work_attribution(
            DeviceBatchingForm::ParticipantLoop,
            participants,
            tokens,
            dispatches,
            0,
        )
    })
    .map_err(|error| error.to_string())
}

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
    if weights.len() != gate_up.len() + down.len() {
        return Err(CudaDeviceRuntimeError::contract(
            "native SwiGLU matrix pointer inventory differs",
        ));
    }
    for (index, part) in gate_up.iter().enumerate() {
        kernels.linear(
            stream,
            input,
            weights[index],
            gate_up_output,
            part,
            tokens,
            doubled,
            ElementType::F16,
        )?;
    }
    launch_silu_mul(
        stream,
        silu,
        gate_up_output,
        activation,
        intermediate_i32,
        layout.activation_elements,
    )?;
    for (index, part) in down.iter().enumerate() {
        kernels.linear(
            stream,
            activation,
            weights[gate_up.len() + index],
            output,
            part,
            tokens,
            hidden,
            ElementType::F16,
        )?;
    }
    Ok(())
}

#[cfg(test)]
mod tests;
