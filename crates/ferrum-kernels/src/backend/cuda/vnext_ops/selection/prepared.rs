//! Retained argmax metadata shared by eager encoding and passive replay.
use super::*;
use ferrum_interfaces::execution_cost::SelectedCommandCostEvidenceV1;
use ferrum_interfaces::vnext::OperationCostCommand;
use ferrum_types::SloStructuredCostCapture;

#[derive(Debug, Clone, Copy)]
struct MaskedArgmaxLaunch {
    first_region: usize,
    scratch_offset_bytes: u64,
    vocabulary_size: i32,
    repetition_capacity: i32,
}

struct PreparedMaskedArgmax {
    regions: Vec<CudaBufferRegion>,
    launches: Vec<MaskedArgmaxLaunch>,
    scratch_region: usize,
    parallel: bool,
    replay_key: crate::backend::cuda::vnext_replay::CudaCommandReplayKey,
    cost_work: OperationCostCommand,
}

fn prepare(
    provider_fingerprint: &str,
    precision: ArgmaxPrecision,
    capture: SloStructuredCostCapture,
    invocation: &BatchedOperationInvocation<'_, CudaDeviceBuffer>,
) -> Result<PreparedMaskedArgmax, String> {
    if invocation.operation().id.as_str() != precision.operation()
        || invocation.participants().is_empty()
    {
        return Err("CUDA masked argmax received another or empty operation".to_owned());
    }

    let first_vocabulary_size =
        unsigned_attribute(invocation.participants()[0].attributes(), "vocab_size")?;
    let scratch_stride = masked_argmax_scratch_stride(first_vocabulary_size, precision.element())?;
    let required_scratch_bytes = scratch_stride
        .checked_mul(invocation.participants().len() as u64)
        .ok_or_else(|| "CUDA masked argmax scratch size overflows".to_owned())?;
    let mut regions = Vec::with_capacity(invocation.participants().len() * 6 + 1);
    let mut launches = Vec::with_capacity(invocation.participants().len());
    for (participant_index, participant) in invocation.participants().iter().enumerate() {
        let logits = binding(participant.bindings(), ResolvedValueRole::Input, 0)?;
        let valid_mask = binding(participant.bindings(), ResolvedValueRole::Input, 1)?;
        let repetition_token_ids = binding(participant.bindings(), ResolvedValueRole::Input, 2)?;
        let repetition_offsets = binding(participant.bindings(), ResolvedValueRole::Input, 3)?;
        let repetition_penalty = binding(participant.bindings(), ResolvedValueRole::Input, 4)?;
        let output = binding(participant.bindings(), ResolvedValueRole::Output, 0)?;
        let vocabulary_size = unsigned_attribute(participant.attributes(), "vocab_size")?;
        if vocabulary_size != first_vocabulary_size {
            return Err("CUDA masked argmax participants disagree on vocabulary size".to_owned());
        }
        let repetition_capacity = validate_masked_argmax_signature(
            logits,
            valid_mask,
            repetition_token_ids,
            repetition_offsets,
            repetition_penalty,
            output,
            vocabulary_size,
            precision.element(),
        )?;

        let first_region = regions.len();
        regions.push(contiguous_region(participant, logits, precision.element())?);
        regions.push(contiguous_region(participant, valid_mask, ElementType::U8)?);
        regions.push(contiguous_region(
            participant,
            repetition_token_ids,
            ElementType::U32,
        )?);
        regions.push(contiguous_region(
            participant,
            repetition_offsets,
            ElementType::U32,
        )?);
        regions.push(contiguous_region(
            participant,
            repetition_penalty,
            ElementType::F32,
        )?);
        regions.push(contiguous_region(participant, output, ElementType::U32)?);
        launches.push(MaskedArgmaxLaunch {
            first_region,
            scratch_offset_bytes: scratch_stride
                .checked_mul(participant_index as u64)
                .ok_or_else(|| "CUDA masked argmax scratch offset overflows".to_owned())?,
            vocabulary_size: i32::try_from(vocabulary_size)
                .map_err(|_| "masked argmax vocabulary exceeds i32".to_owned())?,
            repetition_capacity,
        });
    }
    let scratch_region = regions.len();
    regions.push(transformer::shared_scratch_region(
        invocation,
        required_scratch_bytes,
    )?);

    let participant_count = u32::try_from(invocation.participants().len())
        .map_err(|_| "masked argmax participant count exceeds u32".to_owned())?;
    let dispatches_per_participant = argmax_dispatches(launches[0].vocabulary_size);
    let parallel = dispatches_per_participant == 2;
    let mut replay_key =
        CudaCommandReplayKeyBuilder::new(provider_fingerprint, "vnext_last_token_masked_argmax")
            .u64(dispatches_per_participant)
            .u64(launches.len() as u64);
    for launch in &launches {
        replay_key = replay_key
            .u64(launch.first_region as u64)
            .u64(launch.scratch_offset_bytes)
            .i32(launch.vocabulary_size)
            .i32(launch.repetition_capacity);
    }
    let cost_work = cost_route::argmax_command(participant_count, launches[0].vocabulary_size)
        .map_err(|error| error.to_string())?;
    let cost_work = match selected::evidence(
        precision,
        participant_count,
        launches
            .iter()
            .map(|row| (row.vocabulary_size, row.repetition_capacity)),
        capture,
    ) {
        Some(evidence) => cost_work
            .with_statistical_evidence(evidence)
            .map_err(|error| format!("{error:?}"))?,
        None => cost_work,
    };
    Ok(PreparedMaskedArgmax {
        regions,
        launches,
        scratch_region,
        parallel,
        replay_key: replay_key.finish(),
        cost_work,
    })
}

pub(in crate::backend::cuda::vnext_ops) fn replay_evidence(
    invocation: &BatchedOperationInvocation<'_, CudaDeviceBuffer>,
    precision: ArgmaxPrecision,
    capture: SloStructuredCostCapture,
) -> Result<Option<SelectedCommandCostEvidenceV1>, VNextError> {
    if capture == SloStructuredCostCapture::Disabled {
        return Ok(None);
    }
    prepare("passive-replay-query", precision, capture, invocation)
        .map(|prepared| prepared.cost_work.statistical_evidence().cloned())
        .map_err(|reason| VNextError::InvalidExecutionPlan { reason })
}

pub(in crate::backend::cuda::vnext_ops) fn encode(
    functions: &ArgmaxFunctions,
    provider_fingerprint: &str,
    precision: ArgmaxPrecision,
    capture: SloStructuredCostCapture,
    invocation: BatchedOperationInvocation<'_, CudaDeviceBuffer>,
) -> Result<CudaDeviceCommand, String> {
    let PreparedMaskedArgmax {
        regions,
        launches,
        scratch_region,
        parallel,
        replay_key,
        cost_work,
    } = prepare(provider_fingerprint, precision, capture, &invocation)?;
    let functions = functions.clone();
    CudaDeviceCommand::replayable_operation(
        "vnext_last_token_masked_argmax",
        regions,
        replay_key,
        move |stream, regions| {
            for launch in &launches {
                let logits = regions[launch.first_region].device_ptr();
                let valid_mask = regions[launch.first_region + 1].device_ptr();
                let repetition_token_ids = regions[launch.first_region + 2].device_ptr();
                let repetition_offsets = regions[launch.first_region + 3].device_ptr();
                let repetition_penalty = regions[launch.first_region + 4].device_ptr();
                let output = regions[launch.first_region + 5].device_ptr();
                let scratch = regions[scratch_region]
                    .device_ptr()
                    .checked_add(launch.scratch_offset_bytes)
                    .ok_or_else(|| {
                        CudaDeviceRuntimeError::contract(
                            "vNext masked argmax scratch pointer overflows",
                        )
                    })?;
                functions.launch(
                    stream,
                    ArgmaxArguments {
                        logits,
                        scratch,
                        valid_mask,
                        repetition_offsets,
                        repetition_token_ids,
                        repetition_penalty,
                        output,
                        vocabulary_size: launch.vocabulary_size,
                        repetition_capacity: launch.repetition_capacity,
                    },
                    parallel,
                )?;
            }
            Ok(())
        },
    )
    .and_then(|command| cost_route::apply(command, cost_work))
    .map_err(|error| error.to_string())
}
