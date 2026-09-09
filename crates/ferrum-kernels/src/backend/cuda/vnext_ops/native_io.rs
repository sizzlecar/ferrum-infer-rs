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
        [Q3K, Q4K, Q5K, Q6K, Q8_0, Iq3S, Iq4Nl, Iq4Xs]
            .into_iter()
            .map(|format| QuantizationFormatId::new(format.format_id()))
            .collect::<Result<_, _>>()
            .map_err(contract_error)?,
        implementation_fingerprint(&[
            include_str!("native_io.rs").as_bytes(),
            include_str!("../vnext_ops.rs").as_bytes(),
            include_str!("native_blocks.rs").as_bytes(),
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
    let mut key =
        CudaCommandReplayKeyBuilder::new(fingerprint, operation).u64(weight.parts.len() as u64);
    for part in &weight.parts {
        key = key.u32(part.rows).u32(part.columns).u32(part.output_offset);
        for value in part.format.parameters() {
            key = key.u32(value);
        }
    }
    key
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
        sum.checked_add(count.div_ceil(chunk_limit))
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
                    kernels.embedding(
                        stream,
                        token_ptr,
                        regions[0].device_ptr(),
                        output_ptr,
                        &part,
                        chunk,
                        precision.element(),
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
        launches.push((input_index, output_index));
        key = key.u64(input_index as u64).u64(output_index as u64);
    }
    let participants =
        u32::try_from(launches.len()).map_err(|_| "too many projection participants")?;
    let stride = u32::try_from(outputs).map_err(|_| "native projection output stride overflows")?;
    let dispatches = (launches.len() as u64)
        .checked_mul(weight.parts.len() as u64)
        .ok_or("native projection dispatch count overflows")?;
    let kernels = kernels.clone();
    CudaDeviceCommand::replayable_operation(
        "vnext_native_last_token_linear",
        regions,
        key.finish(),
        move |stream, regions| {
            for &(input, output) in &launches {
                for (index, part) in weight.parts.iter().enumerate() {
                    kernels.linear(
                        stream,
                        regions[input].device_ptr(),
                        regions[index].device_ptr(),
                        regions[output].device_ptr(),
                        part,
                        1,
                        stride,
                        precision.element(),
                    )?;
                }
            }
            Ok(())
        },
    )
    .and_then(|command| {
        command.with_work_attribution(
            DeviceBatchingForm::ParticipantLoop,
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
