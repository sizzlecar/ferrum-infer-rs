//! Native compressed matrix projection through the ordinary vNext command path.

use super::super::native_blocks::{weights, CudaNativeBlockKernels};
use super::*;
use crate::gguf_blocks::GgufBlockFormat;

pub(super) fn quantization_formats() -> Result<BTreeSet<QuantizationFormatId>, VNextError> {
    use GgufBlockFormat::*;
    [Q3K, Q4K, Q5K, Q6K, Q8_0, Iq3S, Iq4Nl, Iq4Xs]
        .into_iter()
        .map(|format| QuantizationFormatId::new(format.format_id()))
        .collect()
}

pub(super) fn encode(
    fingerprint: &str,
    kernels: &CudaNativeBlockKernels,
    invocation: BatchedOperationInvocation<'_, CudaDeviceBuffer>,
) -> Result<CudaDeviceCommand, String> {
    ensure_invocation(&invocation, DENSE_LINEAR_OPERATION_ID)?;
    let first = &invocation.participants()[0];
    let columns = unsigned_attribute(first.attributes(), "in_features")?;
    let rows = unsigned_attribute(first.attributes(), "out_features")?;
    let weight = weights::resolve(
        first,
        binding(first.bindings(), ResolvedValueRole::Input, 1)?,
        &[rows, columns],
    )?;
    let input_packed = token_binding_is_packed(&invocation, ResolvedValueRole::Input, 0)?;
    let output_packed = token_binding_is_packed(&invocation, ResolvedValueRole::Output, 0)?;
    if invocation.participant_token_ranges().len() != invocation.participants().len() {
        return Err("CUDA native linear participant ranges are incomplete".into());
    }
    let mut regions = weight.regions;
    let mut launches = Vec::new();
    let mut key = CudaCommandReplayKeyBuilder::new(fingerprint, "vnext_native_dense_linear")
        .u64(rows)
        .u64(columns)
        .u64(weight.parts.len() as u64);
    for part in &weight.parts {
        key = key.u32(part.rows).u32(part.columns).u32(part.output_offset);
        for parameter in part.format.parameters() {
            key = key.u32(parameter);
        }
    }
    for (index, (participant, range)) in invocation
        .participants()
        .iter()
        .zip(invocation.participant_token_ranges())
        .enumerate()
    {
        let input = binding(participant.bindings(), ResolvedValueRole::Input, 0)?;
        let value = binding(participant.bindings(), ResolvedValueRole::Input, 1)?;
        let output = binding(participant.bindings(), ResolvedValueRole::Output, 0)?;
        if unsigned_attribute(participant.attributes(), "in_features")? != columns
            || unsigned_attribute(participant.attributes(), "out_features")? != rows
        {
            return Err("CUDA native linear participant dimensions disagree".into());
        }
        validate_dense_linear(input, value, output, columns, rows)?;
        if index != 0 {
            let candidate = weights::resolve(participant, value, &[rows, columns])?;
            if candidate.parts != weight.parts
                || candidate.regions.len() != weight.parts.len()
                || !candidate
                    .regions
                    .iter()
                    .zip(&regions)
                    .all(|(left, right)| same_physical_region(left, right))
            {
                return Err(
                    "CUDA native linear participants do not share their physical weights".into(),
                );
            }
        }
        let tokens = checked_u32(range.immediate_tokens(), "native linear tokens")?;
        if tokens == 0 || tokens > u16::MAX as u32 {
            return Err("CUDA native linear token count exceeds its launch extent".into());
        }
        let packed = range.immediate_token_range();
        let source = range.source_token_range();
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
            u64::from(tokens),
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
            u64::from(tokens),
        )?);
        launches.push((input_index, output_index, tokens));
        key = key
            .u64(input_index as u64)
            .u64(output_index as u64)
            .u32(tokens);
    }
    let participants = checked_u32(
        invocation.participants().len() as u64,
        "native linear participants",
    )?;
    let tokens = invocation.work_shape().immediate_tokens();
    let output_stride = checked_u32(rows, "native linear output stride")?;
    let dispatches = (launches.len() as u64)
        .checked_mul(weight.parts.len() as u64)
        .ok_or("CUDA native linear dispatch count overflows")?;
    let kernels = kernels.clone();
    CudaDeviceCommand::replayable_operation(
        "vnext_native_dense_linear",
        regions,
        key.finish(),
        move |stream, regions| {
            for &(input, output, rows) in &launches {
                for (index, part) in weight.parts.iter().enumerate() {
                    kernels.linear(
                        stream,
                        regions[input].device_ptr(),
                        regions[index].device_ptr(),
                        regions[output].device_ptr(),
                        part,
                        rows,
                        output_stride,
                        ElementType::F16,
                    )?;
                }
            }
            Ok(())
        },
    )
    .and_then(|command| {
        command.with_work_attribution(
            if participants == 1 {
                DeviceBatchingForm::Scalar
            } else {
                DeviceBatchingForm::ParticipantLoop
            },
            participants,
            tokens,
            dispatches,
            0,
        )
    })
    .map_err(|error| error.to_string())
}
