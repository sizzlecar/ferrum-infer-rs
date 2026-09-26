//! Token lookup and final-row projection over retained native matrix layouts.

use super::native_blocks::{weights, CudaNativeBlockKernels};
use super::*;
mod cost_route;
pub(super) mod embedding;
pub(super) use cost_route::projection as projection_cost_route;

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
            include_bytes!("native_io/cost_route.rs"),
            include_bytes!("embedding.rs"),
            include_bytes!("native_io/embedding.rs"),
            include_bytes!("native_blocks/embedding.rs"),
            include_bytes!("native_blocks/linear_launch.rs"),
            include_bytes!("native_blocks/selected.rs"),
            include_str!("cost_route.rs").as_bytes(),
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
    structured_capture: ferrum_types::SloStructuredCostCapture,
    invocation: BatchedOperationInvocation<'_, CudaDeviceBuffer>,
) -> Result<CudaDeviceCommand, String> {
    let prepared = embedding::prepare(fingerprint, precision, &invocation)?;
    let selected = prepared.selected(precision, structured_capture);
    let embedding::Prepared {
        part,
        regions,
        scratch,
        launches,
        key,
        chunk_limit,
        participants,
        tokens,
        dispatches,
    } = prepared;
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
    .map(|command| command.with_statistical_evidence(selected))
    .map_err(|error| error.to_string())
}

fn embedding_chunk_limit(part: &weights::MatrixPart) -> Result<u64, String> {
    super::native_blocks::embedding_elements(part, 1).map_err(|error| error.to_string())?;
    Ok(u64::from(u32::MAX / part.columns).min(MAXIMUM_TOKENS_PER_LAUNCH))
}

pub(super) fn embedding_dispatches(
    part: &weights::MatrixPart,
    counts: impl IntoIterator<Item = u64>,
) -> Result<u64, String> {
    let limit = embedding_chunk_limit(part)?;
    counts.into_iter().try_fold(0_u64, |sum, count| {
        if count == 0 {
            return Err("CUDA embedding cost span is empty".into());
        }
        count
            .div_ceil(limit)
            .checked_mul(1 + u64::from(part.transform.is_some()))
            .and_then(|dispatches| sum.checked_add(dispatches))
            .ok_or_else(|| "CUDA embedding dispatch count overflows".into())
    })
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
    if !packed_projection_semantics(input_packed, parts, count, true)
        || rows.len() != count
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

fn packed_projection_semantics(
    input_packed: bool,
    parts: &[weights::MatrixPart],
    count: usize,
    canonical_unit_rows: bool,
) -> bool {
    input_packed
        && (2..=u16::MAX as usize).contains(&count)
        && !parts.is_empty()
        && parts.iter().all(|part| part.transform.is_none())
        && canonical_unit_rows
}

struct PreparedNativeProjection {
    regions: Vec<CudaBufferRegion>,
    parts: Vec<weights::MatrixPart>,
    launches: Vec<(usize, usize, u32)>,
    scratch: Option<usize>,
    participants: u32,
    packed_rows: Option<u32>,
    stride: u32,
    dispatches: u64,
    key: crate::backend::cuda::vnext_replay::CudaCommandReplayKey,
    selected: Option<ferrum_interfaces::execution_cost::SelectedCommandCostEvidenceV1>,
}

// This prepares retained row metadata only. It neither encodes nor submits
// kernels. Eager and direct graph replay therefore prove the same actual
// packed/participant layout before attaching selected work.
fn prepare_projection(
    fingerprint: &str,
    precision: TokenPrecision,
    capture: ferrum_types::SloStructuredCostCapture,
    invocation: &BatchedOperationInvocation<'_, CudaDeviceBuffer>,
) -> Result<PreparedNativeProjection, String> {
    transformer::ensure_invocation(invocation, precision.projection_operation())?;
    let first = &invocation.participants()[0];
    let hidden = unsigned_attribute(first.attributes(), "hidden_size")?;
    let outputs = unsigned_attribute(first.attributes(), "out_features")?;
    let weight = retain_shared_weight(invocation, &[outputs, hidden])?;
    let input_packed =
        transformer::token_binding_is_packed(invocation, ResolvedValueRole::Input, 0)?;
    let mut key = matrix_key(fingerprint, "vnext_native_last_token_linear", &weight);
    let mut regions = weight.regions;
    let weight_region_count = regions.len();
    let scratch = super::native_blocks::hadamard::retain_workspace(invocation, &mut regions)?;
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
    let selected = super::native_blocks::selected::linear(
        &weight.parts,
        launches.iter().map(|&(_, _, rows)| rows),
        u64::from(participants),
        stride,
        precision.element(),
        scratch.map_or(0, |index| regions[index].length_bytes()),
        capture,
    );
    Ok(PreparedNativeProjection {
        regions,
        parts: weight.parts,
        launches,
        scratch,
        participants,
        packed_rows,
        stride,
        dispatches,
        key: key.finish(),
        selected,
    })
}

pub(super) fn projection_replay_evidence(
    invocation: &BatchedOperationInvocation<'_, CudaDeviceBuffer>,
    precision: TokenPrecision,
    capture: ferrum_types::SloStructuredCostCapture,
) -> Result<Option<ferrum_interfaces::execution_cost::SelectedCommandCostEvidenceV1>, VNextError> {
    if capture == ferrum_types::SloStructuredCostCapture::Disabled {
        return Ok(None);
    }
    // Dense F16 dispatches through the separate cuBLAS/gather encoder; this
    // native helper cannot declare that unimplemented provider branch complete.
    if precision != TokenPrecision::F32 && !requires_native(invocation) {
        return Ok(None);
    }
    prepare_projection("passive-replay-query", precision, capture, invocation)
        .map(|prepared| prepared.selected)
        .map_err(|reason| VNextError::InvalidExecutionPlan { reason })
}

pub(super) fn encode_projection(
    fingerprint: &str,
    kernels: &CudaNativeBlockKernels,
    precision: TokenPrecision,
    capture: ferrum_types::SloStructuredCostCapture,
    invocation: BatchedOperationInvocation<'_, CudaDeviceBuffer>,
) -> Result<CudaDeviceCommand, String> {
    let PreparedNativeProjection {
        regions,
        parts,
        launches,
        scratch,
        participants,
        packed_rows,
        stride,
        dispatches,
        key,
        selected,
    } = prepare_projection(fingerprint, precision, capture, &invocation)?;
    let kernels = kernels.clone();
    CudaDeviceCommand::replayable_operation(
        "vnext_native_last_token_linear",
        regions,
        key,
        move |stream, regions| {
            for &(input, output, rows) in &launches {
                for (index, part) in parts.iter().enumerate() {
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
    .map(|command| command.with_statistical_evidence(selected))
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
