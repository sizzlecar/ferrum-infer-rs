//! Retained actual FFN preparation shared by eager encoding and direct replay.
use super::*;

pub(super) struct Prepared {
    pub regions: Vec<CudaBufferRegion>,
    pub gate_up: Vec<weights::MatrixPart>,
    pub down: Vec<weights::MatrixPart>,
    pub launches: Vec<(usize, usize, u32, u64)>,
    pub scratch_index: usize,
    pub scratch_layout: ScratchLayout,
    pub hidden: u32,
    pub intermediate: u32,
    pub transform_bytes: u64,
    pub q8_bytes: u64,
    pub mmq_bytes: u64,
    pub selection: route_selection::Selection,
    pub key: crate::backend::cuda::vnext_replay::CudaCommandReplayKey,
}

pub(super) fn prepare(
    fingerprint: &str,
    q8: Option<Q8SumPolicy>,
    mmq: Option<&StreamMmq>,
    capture: ferrum_types::SloStructuredCostCapture,
    invocation: &BatchedOperationInvocation<'_, CudaDeviceBuffer>,
) -> Result<Prepared, String> {
    let operation = if let Some(mmq) = mmq {
        mmq.operation_id()
    } else if let Some(q8) = q8 {
        q8.operation_id()
    } else {
        DENSE_SWIGLU_OPERATION_ID
    };
    ensure_invocation(invocation, operation)?;
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
        .bytes(operation.as_bytes())
        .u64(hidden)
        .u64(intermediate);
    for matrix in matrices {
        key = weights::key(key, &matrix.parts);
        regions.extend(matrix.regions);
        parts.push(matrix.parts);
    }
    let mut parts = parts.into_iter();
    let gate_up = parts.next().ok_or("missing native gate/up matrix")?;
    let down = parts.next().ok_or("missing native down matrix")?;
    let tokens = invocation.work_shape().immediate_tokens();
    let scratch_layout = ScratchLayout::new(tokens, intermediate)?;
    let q8_bytes = if let Some(q8) = q8 {
        q8_part_workspace_per_token_with_policy(&gate_up, q8)?
            .max(q8_part_workspace_per_token_with_policy(&down, q8)?)
            .checked_mul(tokens)
            .ok_or("Q8 SwiGLU scratch overflows")?
    } else {
        0
    };
    let mmq_bytes = if let Some(mmq) = mmq {
        stream_mmq_workspace(
            first.bindings(),
            checked_u32(hidden, "Stream-MMQ hidden")?,
            checked_u32(intermediate, "Stream-MMQ intermediate")?,
            mmq,
        )?
    } else {
        0
    };
    let transform_bytes =
        super::super::super::native_blocks::hadamard::workspace_bytes_per_token(first.bindings())?
            .checked_mul(tokens)
            .ok_or("native SwiGLU Hadamard scratch overflows")?;
    let required_bytes = scratch_layout
        .total_bytes
        .checked_add(transform_bytes)
        .and_then(|bytes| bytes.checked_add(q8_bytes))
        .and_then(|bytes| bytes.checked_add(mmq_bytes))
        .ok_or("native SwiGLU scratch overflows")?;
    let scratch_index = regions.len();
    regions.push(shared_scratch_region(invocation, required_bytes)?);
    let input_packed = token_binding_is_packed(invocation, ResolvedValueRole::Input, 0)?;
    let output_packed = token_binding_is_packed(invocation, ResolvedValueRole::Output, 0)?;
    if invocation.participant_token_ranges().len() != invocation.participants().len() {
        return Err("CUDA native SwiGLU participant ranges are incomplete".into());
    }
    let participants = checked_u32(
        invocation.participants().len() as u64,
        "native SwiGLU participants",
    )?;
    let mut selection = route_selection::select(
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
    let packed_rows = selection.packed_rows;
    let mmq_hit = selection.mmq_hit;
    key = key
        .boolean(packed_rows.is_some())
        .boolean(mmq_hit)
        .boolean(selection.mmq_down_hit)
        .u64(mmq_bytes);
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
        if packed_rows.is_some() {
            continue;
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
    if let Some(rows) = packed_rows {
        let input = regions.len();
        regions.push(shared_token_region(
            invocation,
            ResolvedValueRole::Input,
            0,
            ElementType::F16,
            tokens,
        )?);
        let output = regions.len();
        regions.push(shared_token_region(
            invocation,
            ResolvedValueRole::Output,
            0,
            ElementType::F16,
            tokens,
        )?);
        launches.push((input, output, rows, 0));
        key = key.u64(input as u64).u64(output as u64).u32(rows).u64(0);
    }
    if launches.len() as u64 != selection.launches {
        return Err("native SwiGLU actual launch inventory differs from selection".into());
    }
    let hidden = checked_u32(hidden, "native SwiGLU hidden")?;
    let intermediate = checked_u32(intermediate, "native SwiGLU intermediate")?;
    key = key
        .u64(scratch_layout.gate_up_bytes)
        .u64(required_bytes)
        .u64(transform_bytes)
        .u64(q8_bytes);
    selection.command = selected::attach(
        selection.command,
        selected::swiglu(
            &gate_up,
            &down,
            launches.iter().map(|row| row.2),
            tokens,
            hidden,
            intermediate,
            transform_bytes,
            q8_bytes,
            q8,
            mmq,
            mmq_hit,
            capture,
        ),
    );
    Ok(Prepared {
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
        key: key.finish(),
    })
}
