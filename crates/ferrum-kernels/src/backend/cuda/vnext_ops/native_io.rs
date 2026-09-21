//! Token lookup and final-row projection over retained native matrix layouts.

use super::native_blocks::{weights, CudaNativeBlockKernels};
use super::*;

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum TokenPrecision {
    F16,
    F32,
}

impl TokenPrecision {
    pub(super) fn element(self) -> ElementType {
        match self {
            Self::F16 => ElementType::F16,
            Self::F32 => ElementType::F32,
        }
    }

    pub(super) fn embedding_operation(self) -> &'static str {
        match self {
            Self::F16 => TOKEN_EMBEDDING_OPERATION_ID,
            Self::F32 => ferrum_interfaces::vnext::TOKEN_EMBEDDING_F32_MASTER_OPERATION_ID,
        }
    }

    pub(super) fn projection_operation(self) -> &'static str {
        match self {
            Self::F16 => LAST_TOKEN_DENSE_LINEAR_OPERATION_ID,
            Self::F32 => ferrum_interfaces::vnext::LAST_TOKEN_DENSE_LINEAR_F32_OPERATION_ID,
        }
    }
}

pub(super) fn descriptor(
    runtime: &CudaDeviceRuntime,
    contract: &dyn OperationContract,
    provider: &str,
    capability: &str,
    estimator: &str,
) -> Result<OperationProviderDescriptor, CudaDeviceRuntimeError> {
    use crate::gguf_blocks::GgufBlockFormat::*;
    transformer::provider_descriptor_with_formats(
        runtime,
        contract,
        provider,
        capability,
        estimator,
        contiguous_bindings(),
        BTreeSet::from([
            WeightFormatId::new(DENSE_SAFETENSORS_FORMAT_ID).map_err(contract_error)?,
            WeightFormatId::new("weight-format.gguf.native-block").map_err(contract_error)?,
        ]),
        [Pq2_0, Q3K, Q4K, Q5K, Q6K, Q8_0, Iq3S, Iq4Nl, Iq4Xs]
            .into_iter()
            .map(|format| QuantizationFormatId::new(format.format_id()))
            .collect::<Result<_, _>>()
            .map_err(contract_error)?,
        implementation_fingerprint(&[
            include_str!("native_io.rs").as_bytes(),
            include_str!("../vnext_ops.rs").as_bytes(),
            include_str!("native_blocks.rs").as_bytes(),
            include_str!("native_blocks/hadamard.rs").as_bytes(),
            include_str!("native_blocks/weights.rs").as_bytes(),
            crate::ptx::EMBEDDING_LOOKUP.as_bytes(),
            crate::ptx::VNEXT_GGUF.as_bytes(),
            provider.as_bytes(),
        ]),
    )
}

pub(super) fn requires_native(
    invocation: &BatchedOperationInvocation<'_, CudaDeviceBuffer>,
) -> bool {
    invocation.participants().iter().any(|participant| {
        participant.bindings().iter().any(|value| {
            value.role() == ResolvedValueRole::Input
                && value.ordinal() == 1
                && value.weight().is_some_and(|weight| {
                    !matches!(
                        weight.physical_layout(),
                        ferrum_interfaces::vnext::PhysicalWeightLayout::Dense { .. }
                    )
                })
        })
    })
}

fn retain_shared_weight(
    invocation: &BatchedOperationInvocation<'_, CudaDeviceBuffer>,
    shape: &[u64; 2],
) -> Result<weights::MatrixWeight, String> {
    let first = invocation
        .participants()
        .first()
        .ok_or("empty native token I/O")?;
    if invocation.participant_token_ranges().len() != invocation.participants().len() {
        return Err("CUDA native token I/O participant ranges are incomplete".into());
    }
    let weight = weights::resolve(
        first,
        binding(first.bindings(), ResolvedValueRole::Input, 1)?,
        shape,
    )?;
    for participant in &invocation.participants()[1..] {
        let candidate = weights::resolve(
            participant,
            binding(participant.bindings(), ResolvedValueRole::Input, 1)?,
            shape,
        )?;
        if candidate.parts != weight.parts
            || candidate.regions.len() != weight.regions.len()
            || !candidate
                .regions
                .iter()
                .zip(&weight.regions)
                .all(|(left, right)| same_physical_region(left, right))
        {
            return Err("CUDA native token I/O participants do not share physical weights".into());
        }
    }
    Ok(weight)
}

fn matrix_key(
    fingerprint: &str,
    operation: &'static str,
    weight: &weights::MatrixWeight,
) -> CudaCommandReplayKeyBuilder {
    weights::key(
        CudaCommandReplayKeyBuilder::new(fingerprint, operation),
        &weight.parts,
    )
}

pub(super) fn encode_embedding(
    fingerprint: &str,
    kernels: &CudaNativeBlockKernels,
    precision: TokenPrecision,
    invocation: BatchedOperationInvocation<'_, CudaDeviceBuffer>,
) -> Result<CudaDeviceCommand, String> {
    transformer::ensure_invocation(&invocation, precision.embedding_operation())?;
    let first = &invocation.participants()[0];
    let hidden = unsigned_attribute(first.attributes(), "hidden_size")?;
    let vocabulary = unsigned_attribute(first.attributes(), "vocab_size")?;
    let weight = retain_shared_weight(&invocation, &[vocabulary, hidden])?;
    // Vocabulary partitions require ID routing before lookup; this kernel
    // consumes one complete table and must not silently address the first part.
    let [part] = weight.parts.as_slice() else {
        return Err("CUDA native embedding requires one complete vocabulary table".into());
    };
    let chunk_limit = embedding_chunk_limit(part)?;
    let input_packed =
        transformer::token_binding_is_packed(&invocation, ResolvedValueRole::Input, 0)?;
    let output_packed =
        transformer::token_binding_is_packed(&invocation, ResolvedValueRole::Output, 0)?;
    let mut key = matrix_key(fingerprint, "vnext_native_embedding", &weight);
    let mut regions = weight.regions;
    let scratch = super::native_blocks::hadamard::retain_workspace(&invocation, &mut regions)?;
    let mut launches = Vec::new();
    for (participant, range) in invocation
        .participants()
        .iter()
        .zip(invocation.participant_token_ranges())
    {
        let input = binding(participant.bindings(), ResolvedValueRole::Input, 0)?;
        let table = binding(participant.bindings(), ResolvedValueRole::Input, 1)?;
        let output = binding(participant.bindings(), ResolvedValueRole::Output, 0)?;
        if unsigned_attribute(participant.attributes(), "hidden_size")? != hidden
            || unsigned_attribute(participant.attributes(), "vocab_size")? != vocabulary
        {
            return Err("CUDA native embedding participant dimensions disagree".into());
        }
        validate_signature(
            input,
            table,
            output,
            vocabulary,
            hidden,
            precision.element(),
        )?;
        let count = range.immediate_tokens();
        if count == 0 {
            return Err("CUDA native embedding cannot launch an empty token span".into());
        }
        let source = range.source_token_range();
        let packed = range.immediate_token_range();
        let input_index = regions.len();
        regions.push(contiguous_token_region(
            participant,
            input,
            ElementType::U32,
            if input_packed {
                packed.start
            } else {
                source.start
            },
            count,
        )?);
        let output_index = regions.len();
        regions.push(contiguous_token_region(
            participant,
            output,
            precision.element(),
            if output_packed {
                packed.start
            } else {
                source.start
            },
            count,
        )?);
        launches.push((input_index, output_index, count));
        key = key
            .u64(input_index as u64)
            .u64(output_index as u64)
            .u64(count);
    }
    let participants =
        u32::try_from(launches.len()).map_err(|_| "too many embedding participants")?;
    let tokens = invocation.work_shape().immediate_tokens();
    let dispatches = launches.iter().try_fold(0_u64, |sum, &(_, _, count)| {
        sum.checked_add(count.div_ceil(chunk_limit) * (1 + u64::from(part.transform.is_some())))
            .ok_or("embedding dispatch count overflows")
    })?;
    let part = part.clone();
    let kernels = kernels.clone();
    CudaDeviceCommand::replayable_operation(
        "vnext_native_embedding",
        regions,
        key.finish(),
        move |stream, regions| {
            for &(input, output, count) in &launches {
                let mut offset = 0_u64;
                while offset < count {
                    let chunk = (count - offset).min(chunk_limit) as u32;
                    let token_ptr = checked_pointer_offset(
                        regions[input].device_ptr(),
                        offset,
                        4,
                        "native embedding token",
                    )?;
                    let element_offset =
                        offset.checked_mul(u64::from(part.columns)).ok_or_else(|| {
                            CudaDeviceRuntimeError::contract("embedding output offset overflows")
                        })?;
                    let output_ptr = checked_pointer_offset(
                        regions[output].device_ptr(),
                        element_offset,
                        precision.element().size_bytes(),
                        "native embedding output",
                    )?;
                    kernels.transformed_embedding(
                        stream,
                        token_ptr,
                        regions[0].device_ptr(),
                        output_ptr,
                        &part,
                        chunk,
                        precision.element(),
                        part.signs_region
                            .map_or(0, |index| regions[index].device_ptr()),
                        scratch.map_or(0, |index| regions[index].device_ptr()),
                    )?;
                    offset += u64::from(chunk);
                }
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

fn embedding_chunk_limit(part: &weights::MatrixPart) -> Result<u64, String> {
    super::native_blocks::embedding_elements(part, 1).map_err(|error| error.to_string())?;
    Ok(u64::from(u32::MAX / part.columns).min(MAXIMUM_TOKENS_PER_LAUNCH))
}

#[derive(Clone, Copy, Debug)]
struct NativeProjectionRow {
    input: u64,
    input_bytes: u64,
    output: u64,
    output_bytes: u64,
}

/// Only coalesce canonical decode rows whose retained windows form two exact
/// physical matrices. Shared backing or logical token order alone is not proof:
/// participant output windows can include alignment padding or be reordered.
fn packed_projection_rows(
    precision: TokenPrecision,
    input_packed: bool,
    hidden: u64,
    outputs: u64,
    parts: &[weights::MatrixPart],
    token_ranges: impl ExactSizeIterator<Item = std::ops::Range<u64>>,
    rows: impl ExactSizeIterator<Item = NativeProjectionRow>,
    weight_ranges: &[std::ops::Range<u64>],
) -> Option<u32> {
    let count = token_ranges.len();
    if !input_packed
        || count < 2
        || rows.len() != count
        || parts.is_empty()
        || parts.iter().any(|part| part.transform.is_some())
        || weight_ranges.is_empty()
        || weight_ranges.iter().any(|range| range.is_empty())
    {
        return None;
    }
    let count = u32::try_from(count)
        .ok()
        .filter(|&n| n <= u16::MAX as u32)?;
    let element_bytes = precision.element().size_bytes();
    let input_bytes = hidden.checked_mul(element_bytes).filter(|&n| n > 0)?;
    let output_bytes = outputs.checked_mul(element_bytes).filter(|&n| n > 0)?;
    let mut input_start = None;
    let mut output_start = None;
    let mut input_end = None;
    let mut output_end = None;
    for (index, (range, row)) in token_ranges.zip(rows).enumerate() {
        let index = u64::try_from(index).ok()?;
        if range != (index..index.checked_add(1)?)
            || row.input == 0
            || row.output == 0
            || !row.input.is_multiple_of(element_bytes)
            || !row.output.is_multiple_of(element_bytes)
            || row.input_bytes != input_bytes
            || row.output_bytes != output_bytes
            || input_end.is_some_and(|end| row.input != end)
            || output_end.is_some_and(|end| row.output != end)
        {
            return None;
        }
        input_start.get_or_insert(row.input);
        output_start.get_or_insert(row.output);
        input_end = Some(row.input.checked_add(input_bytes)?);
        output_end = Some(row.output.checked_add(output_bytes)?);
    }
    let input = input_start?..input_end?;
    let output = output_start?..output_end?;
    let overlaps = |read: &std::ops::Range<u64>| output.start < read.end && read.start < output.end;
    if overlaps(&input) || weight_ranges.iter().any(overlaps) {
        return None;
    }
    Some(count)
}

pub(super) fn encode_projection(
    fingerprint: &str,
    kernels: &CudaNativeBlockKernels,
    precision: TokenPrecision,
    invocation: BatchedOperationInvocation<'_, CudaDeviceBuffer>,
) -> Result<CudaDeviceCommand, String> {
    transformer::ensure_invocation(&invocation, precision.projection_operation())?;
    let first = &invocation.participants()[0];
    let hidden = unsigned_attribute(first.attributes(), "hidden_size")?;
    let outputs = unsigned_attribute(first.attributes(), "out_features")?;
    let weight = retain_shared_weight(&invocation, &[outputs, hidden])?;
    let input_packed =
        transformer::token_binding_is_packed(&invocation, ResolvedValueRole::Input, 0)?;
    let mut key = matrix_key(fingerprint, "vnext_native_last_token_linear", &weight);
    let mut regions = weight.regions;
    let weight_region_count = regions.len();
    let scratch = super::native_blocks::hadamard::retain_workspace(&invocation, &mut regions)?;
    let mut launches = Vec::new();
    for (participant, range) in invocation
        .participants()
        .iter()
        .zip(invocation.participant_token_ranges())
    {
        let input = binding(participant.bindings(), ResolvedValueRole::Input, 0)?;
        let table = binding(participant.bindings(), ResolvedValueRole::Input, 1)?;
        let output = binding(participant.bindings(), ResolvedValueRole::Output, 0)?;
        if unsigned_attribute(participant.attributes(), "hidden_size")? != hidden
            || unsigned_attribute(participant.attributes(), "out_features")? != outputs
        {
            return Err("CUDA native projection participant dimensions disagree".into());
        }
        validate_last_token_dense_linear_signature(
            input,
            table,
            output,
            hidden,
            outputs,
            precision.element(),
        )?;
        let selected = if input_packed {
            range.immediate_token_range()
        } else {
            range.source_token_range()
        };
        let last = last_token(selected)?;
        let input_index = regions.len();
        regions.push(contiguous_token_region(
            participant,
            input,
            precision.element(),
            last,
            1,
        )?);
        let output_index = regions.len();
        regions.push(contiguous_region(participant, output, precision.element())?);
        launches.push((input_index, output_index, 1_u32));
        key = key.u64(input_index as u64).u64(output_index as u64);
    }
    let participants =
        u32::try_from(launches.len()).map_err(|_| "too many projection participants")?;
    let weight_ranges = regions[..weight_region_count]
        .iter()
        .map(|region| {
            let start = region.device_ptr();
            Some(start..start.checked_add(region.length_bytes())?)
        })
        .collect::<Option<Vec<_>>>();
    let packed_rows = weight_ranges.as_ref().and_then(|weight_ranges| {
        packed_projection_rows(
            precision,
            input_packed,
            hidden,
            outputs,
            &weight.parts,
            invocation
                .participant_token_ranges()
                .iter()
                .map(|range| range.immediate_token_range()),
            launches
                .iter()
                .map(|&(input, output, _)| NativeProjectionRow {
                    input: regions[input].device_ptr(),
                    input_bytes: regions[input].length_bytes(),
                    output: regions[output].device_ptr(),
                    output_bytes: regions[output].length_bytes(),
                }),
            weight_ranges,
        )
    });
    if let Some(rows) = packed_rows {
        // Every original region remains retained by the command, including
        // rows addressed relative to the first participant's pointer.
        launches.truncate(1);
        launches[0].2 = rows;
    }
    key = key
        .boolean(packed_rows.is_some())
        .u32(packed_rows.unwrap_or(1));
    let stride = u32::try_from(outputs).map_err(|_| "native projection output stride overflows")?;
    let dispatches = (launches.len() as u64)
        .checked_mul(weights::dispatches(&weight.parts))
        .ok_or("native projection dispatch count overflows")?;
    let kernels = kernels.clone();
    CudaDeviceCommand::replayable_operation(
        "vnext_native_last_token_linear",
        regions,
        key.finish(),
        move |stream, regions| {
            for &(input, output, rows) in &launches {
                for (index, part) in weight.parts.iter().enumerate() {
                    kernels.transformed_linear(
                        stream,
                        regions[input].device_ptr(),
                        regions[index].device_ptr(),
                        regions[output].device_ptr(),
                        part,
                        rows,
                        stride,
                        precision.element(),
                        part.signs_region
                            .map_or(0, |index| regions[index].device_ptr()),
                        scratch.map_or(0, |index| regions[index].device_ptr()),
                    )?;
                }
            }
            Ok(())
        },
    )
    .and_then(|command| {
        command.with_work_attribution(
            if packed_rows.is_some() {
                DeviceBatchingForm::Packed
            } else {
                DeviceBatchingForm::ParticipantLoop
            },
            participants,
            u64::from(participants),
            dispatches,
            0,
        )
    })
    .map_err(|error| error.to_string())
}

fn last_token(range: std::ops::Range<u64>) -> Result<u64, String> {
    if range.is_empty() {
        return Err("CUDA last-token projection cannot select from an empty span".into());
    }
    Ok(range.end - 1)
}

#[cfg(test)]
mod tests;
