//! Bounded terminal head. One admitted logits row is reused on one CUDA stream.
//! No token-length logits output, policy buffers, persistent writes or replay.
use super::*;
use cudarc::cublas::CudaBlas;
use cudarc::driver::CudaStream;

const OP: &str = ferrum_runtime::TAIL_TWO_GREEDY_F16_OPERATION_ID;
const KERNEL: &str = "tail_two_greedy_argmax_f16";

pub(super) struct CudaTailTwoGreedyProvider {
    descriptor: OperationProviderDescriptor,
    function: CudaFunction,
}

impl CudaTailTwoGreedyProvider {
    pub(super) fn new(runtime: &CudaDeviceRuntime) -> Result<Self, CudaDeviceRuntimeError> {
        let contract = ferrum_runtime::tail_two_greedy_f16_contract().map_err(contract_error)?;
        let capability = CapabilityId::new(ferrum_runtime::TAIL_TWO_GREEDY_F16_CAPABILITY_ID)
            .map_err(contract_error)?;
        if !runtime.descriptor().capabilities.contains(&capability) {
            return Err(CudaDeviceRuntimeError::contract(
                "CUDA tail head capability is absent",
            ));
        }
        let fingerprint = implementation_fingerprint(&[
            include_bytes!("tail_two_greedy.rs"),
            include_bytes!("transformer.rs"),
            crate::ptx::ARGMAX_ROWS.as_bytes(),
        ]);
        let mut bindings = transformer::contiguous_bindings(2);
        bindings.push(ProviderStorageBindingRequirement::new(
            ResolvedValueRole::Output,
            1,
            DynamicStorageRequirement::contiguous(),
        ));
        let descriptor = OperationProviderDescriptor::new(
            ProviderId::new("provider.cuda.tail_two_greedy.f16").map_err(contract_error)?,
            contract.descriptor().id.clone(),
            contract
                .descriptor()
                .fingerprint()
                .map_err(contract_error)?,
            fingerprint.clone(),
            ferrum_runtime::ProviderExecutionSemantics::bitwise_eager_only(),
            contract.descriptor().version,
            runtime.descriptor().id.clone(),
            BTreeSet::from([capability]),
            BTreeSet::from([
                WeightFormatId::new(DENSE_SAFETENSORS_FORMAT_ID).map_err(contract_error)?
            ]),
            BTreeSet::new(),
            bindings,
            "resource-estimator.cuda.tail_two_greedy.f16",
            ContractVersion::new(1, 0),
            fingerprint,
        )
        .map_err(contract_error)?
        .with_candidate_state_effects(ferrum_runtime::ProviderCandidateStateEffects::TerminalOnly);
        let function = load(runtime.context())?;
        Ok(Self {
            descriptor,
            function,
        })
    }
}

fn load(
    context: &Arc<cudarc::driver::CudaContext>,
) -> Result<CudaFunction, CudaDeviceRuntimeError> {
    context
        .load_module(Ptx::from_src(crate::ptx::ARGMAX_ROWS))
        .map_err(|e| CudaDeviceRuntimeError::driver("tail head module", e))?
        .load_function(KERNEL)
        .map_err(|e| CudaDeviceRuntimeError::driver("tail head reduction", e))
}

fn dimensions(hidden: u64, vocabulary: u64) -> Result<(i32, i32, u64), String> {
    if hidden == 0 || vocabulary == 0 {
        return Err("tail head dimensions must be positive".into());
    }
    let h = i32::try_from(hidden).map_err(|_| "tail head hidden width exceeds i32")?;
    let v = i32::try_from(vocabulary).map_err(|_| "tail head vocabulary exceeds i32")?;
    // The unchanged scalar reduction increments a signed int by its block
    // width after the final element; reject the otherwise overflowing edge.
    if vocabulary > i32::MAX as u64 - u64::from(THREADS_PER_BLOCK - 1) {
        return Err("tail head vocabulary exceeds safe reduction index range".into());
    }
    let row = vocabulary
        .checked_mul(2)
        .ok_or("tail head scratch overflow")?;
    let scratch = row
        .checked_add(VALUE_ALIGNMENT_BYTES - 1)
        .map(|n| n / VALUE_ALIGNMENT_BYTES * VALUE_ALIGNMENT_BYTES)
        .ok_or("tail head scratch alignment overflow")?;
    hidden
        .checked_mul(vocabulary)
        .and_then(|n| n.checked_mul(2))
        .filter(|n| *n <= isize::MAX as u64)
        .ok_or("tail head weights extent exceeds isize")?;
    Ok((h, v, scratch))
}

impl OperationResourceEstimator for CudaTailTwoGreedyProvider {
    fn descriptor(&self) -> &OperationProviderDescriptor {
        &self.descriptor
    }
    fn estimate_resources(
        &self,
        request: OperationResourceEstimateRequest<'_>,
    ) -> Result<OperationResourceEstimate, VNextError> {
        transformer::ensure_estimator_request(&self.descriptor, &request, OP)?;
        let bad = |reason| VNextError::InvalidExecutionPlan { reason };
        let (_, _, bytes) = dimensions(
            unsigned_attribute(request.attributes(), "hidden_size").map_err(bad)?,
            unsigned_attribute(request.attributes(), "vocab_size").map_err(bad)?,
        )
        .map_err(bad)?;
        // Every participant and tail row uses this one scratch row in stream
        // order. Its bound depends on V, never N or the number of requests.
        let scratch = ProviderWorkspaceRequirement::from_formula(
            ProviderWorkspaceSizeFormula::fixed(bytes)?,
            VALUE_ALIGNMENT_BYTES,
            ProviderWorkspaceScope::Invocation,
            ProviderWorkspaceReusePolicy::OverwriteBeforeRead,
            DynamicStorageRequirement::contiguous(),
        )?;
        Ok(transformer::estimate(
            &self.descriptor,
            request.input_fingerprint(),
            Some(scratch),
        ))
    }
}

impl OperationProvider<CudaDeviceRuntime> for CudaTailTwoGreedyProvider {
    fn reusable_execution_topology(
        &self,
        _: ReusableExecutionTopologyRequest<'_>,
    ) -> Result<ReusableExecutionTopology, VNextError> {
        Ok(ReusableExecutionTopology::EagerBoundary)
    }
    fn encode_selected(
        &self,
        invocation: BatchedOperationInvocation<'_, CudaDeviceBuffer>,
    ) -> Result<EncodedDeviceOperation<CudaDeviceCommand>, OperationFailure> {
        let identity = invocation.participants()[0].identity().clone();
        encode(&self.function, &invocation)
            .map(EncodedDeviceOperation::compute)
            .map_err(|e| provider_failure(identity, "cuda.tail_two_greedy.encode", e))
    }
}

#[derive(Clone, Copy)]
struct TailLaunch {
    input: u64,
    weight: u64,
    decisions: u64,
    carrier: u64,
    rows: u64,
}

fn encode(
    function: &CudaFunction,
    invocation: &BatchedOperationInvocation<'_, CudaDeviceBuffer>,
) -> Result<CudaDeviceCommand, String> {
    if invocation.operation().id.as_str() != OP
        || invocation.participants().is_empty()
        || invocation.participant_token_ranges().len() != invocation.participants().len()
    {
        return Err("tail head requires a nonempty selected token wave".into());
    }
    let first = &invocation.participants()[0];
    let hidden = unsigned_attribute(first.attributes(), "hidden_size")?;
    let vocabulary = unsigned_attribute(first.attributes(), "vocab_size")?;
    let (h, v, scratch_bytes) = dimensions(hidden, vocabulary)?;
    let packed = transformer::token_binding_is_packed(invocation, ResolvedValueRole::Input, 0)?;
    let mut regions = Vec::new();
    let mut reads = Vec::new();
    let mut writes = Vec::new();
    let mut launches = Vec::new();
    for (participant, range) in invocation
        .participants()
        .iter()
        .zip(invocation.participant_token_ranges())
    {
        if participant.attributes() != first.attributes() {
            return Err("tail head participant attributes differ".into());
        }
        let input = binding(participant.bindings(), ResolvedValueRole::Input, 0)?;
        let weight = binding(participant.bindings(), ResolvedValueRole::Input, 1)?;
        let decisions = binding(participant.bindings(), ResolvedValueRole::Output, 0)?;
        let carrier = binding(participant.bindings(), ResolvedValueRole::Output, 1)?;
        let dims = input.tensor().dimensions();
        if dims.len() != 2
            || dims[0] == 0
            || dims[1] != hidden
            || input.tensor().element_type() != ElementType::F16
            || input.tensor().layout() != &ResolvedTensorLayout::Contiguous
            || weight.tensor().dimensions() != [vocabulary, hidden]
            || decisions.tensor().dimensions() != [2]
            || carrier.tensor().dimensions() != [2, hidden]
        {
            return Err("tail head resolved shapes differ from contract".into());
        }
        let selected = if packed {
            range.immediate_token_range()
        } else {
            range.source_token_range()
        };
        if selected.is_empty() || selected.end > dims[0] {
            return Err("tail head selected range is empty or outside input".into());
        }
        let base = regions.len();
        let actual_rows = selected.end - selected.start;
        let input_region = contiguous_token_region(
            participant,
            input,
            ElementType::F16,
            selected.start,
            actual_rows,
        )?;
        let input_bytes = actual_rows
            .checked_mul(hidden)
            .and_then(|n| n.checked_mul(2))
            .ok_or("tail head input byte extent overflows")?;
        if input_region.length_bytes() != input_bytes {
            return Err("tail head physical input differs from actual token extent".into());
        }
        reads.push(byte_range(
            input_region.device_ptr(),
            input_region.length_bytes(),
        )?);
        regions.push(input_region);
        for (b, dtype, writable, expected) in [
            (weight, ElementType::F16, false, vocabulary * hidden * 2),
            (decisions, ElementType::U32, true, 8),
            (carrier, ElementType::F16, true, hidden * 4),
        ] {
            if b.tensor().element_type() != dtype
                || b.tensor().layout() != &ResolvedTensorLayout::Contiguous
            {
                return Err("tail head requires typed contiguous values".into());
            }
            let r = contiguous_region(participant, b, dtype)?;
            if r.length_bytes() != expected {
                return Err("tail head physical output/weight extent differs from contract".into());
            }
            let span = byte_range(r.device_ptr(), r.length_bytes())?;
            if writable {
                writes.push(span);
            } else {
                reads.push(span);
            }
            regions.push(r);
        }
        let rows = actual_rows.min(2);
        let offset = (actual_rows - rows)
            .checked_mul(hidden)
            .and_then(|n| n.checked_mul(2))
            .ok_or("tail head input offset overflow")?;
        launches.push(TailLaunch {
            input: regions[base]
                .device_ptr()
                .checked_add(offset)
                .ok_or("tail input address overflow")?,
            weight: regions[base + 1].device_ptr(),
            decisions: regions[base + 2].device_ptr(),
            carrier: regions[base + 3].device_ptr(),
            rows,
        });
    }
    let scratch = transformer::shared_scratch_region(invocation, scratch_bytes)?;
    writes.push(byte_range(scratch.device_ptr(), scratch.length_bytes())?);
    validate_disjoint(&reads, &writes)?;
    let scratch_pointer = scratch.device_ptr();
    regions.push(scratch);
    let count = u32::try_from(launches.len()).map_err(|_| "tail head batch exceeds u32")?;
    let rows = launches.iter().map(|x| x.rows).sum::<u64>();
    let copies = 2 * u64::from(count) + launches.iter().filter(|x| x.rows == 1).count() as u64;
    let tokens = invocation.work_shape().immediate_tokens();
    let function = function.clone();
    CudaDeviceCommand::operation_with_blas(
        "vnext_tail_two_greedy_f16",
        regions,
        move |stream, blas, _regions| {
            launch_batch(stream, blas, &function, h, v, scratch_pointer, &launches)
        },
    )
    .and_then(|c| {
        c.with_work_attribution(
            if count == 1 {
                DeviceBatchingForm::Scalar
            } else {
                DeviceBatchingForm::ParticipantLoop
            },
            count,
            tokens,
            rows * 2,
            copies,
        )
    })
    .map_err(|e| e.to_string())
}

fn byte_range(pointer: u64, bytes: u64) -> Result<std::ops::Range<u64>, String> {
    if pointer == 0 || bytes == 0 {
        return Err("tail head has empty/null physical region".into());
    }
    Ok(pointer
        ..pointer
            .checked_add(bytes)
            .ok_or("tail head region address overflow")?)
}

fn validate_disjoint(
    reads: &[std::ops::Range<u64>],
    writes: &[std::ops::Range<u64>],
) -> Result<(), String> {
    for (i, w) in writes.iter().enumerate() {
        for other in reads.iter().chain(&writes[..i]) {
            if w.start < other.end && other.start < w.end {
                return Err("tail head write aliases a read/write in the selected wave".into());
            }
        }
    }
    Ok(())
}

fn launch_argmax(
    stream: &CudaStream,
    function: &CudaFunction,
    logits: u64,
    vocabulary: i32,
    output: u64,
) -> Result<(), CudaDeviceRuntimeError> {
    let mut b = stream.launch_builder(function);
    b.arg(&logits).arg(&vocabulary).arg(&output);
    // Encoder retains a full F16 logits row and one U32 decision. The same
    // 256-thread reduction as the ordinary no-penalty head owns this output.
    unsafe {
        b.launch(LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (THREADS_PER_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        })
    }
    .map(|_| ())
    .map_err(|e| CudaDeviceRuntimeError::driver("tail greedy reduction", e))
}

fn launch_batch(
    stream: &CudaStream,
    blas: &CudaBlas,
    function: &CudaFunction,
    h: i32,
    v: i32,
    scratch: u64,
    launches: &[TailLaunch],
) -> Result<(), CudaDeviceRuntimeError> {
    let row_bytes = h as usize * 2;
    for p in launches {
        for slot in 0..p.rows {
            let input = p.input + slot * row_bytes as u64;
            transformer::launch_gemm_f16(
                blas,
                input,
                p.weight,
                scratch,
                1,
                v,
                h,
                "tail head ordinary F16 GEMM",
            )?;
            launch_argmax(stream, function, scratch, v, p.decisions + slot * 4)?;
        }
        // Original hidden bits, without conversion. N=1 projects once, then
        // copies its sole decision and hidden row to the second fixed slot.
        copy_last_token_rows_f16(
            stream,
            p.carrier,
            row_bytes,
            [p.input, p.input + (p.rows - 1) * row_bytes as u64].into_iter(),
        )?;
        if p.rows == 1 {
            unsafe {
                cudarc::driver::result::memcpy_dtod_async(
                    p.decisions + 4,
                    p.decisions,
                    4,
                    stream.cu_stream(),
                )
            }
            .map_err(|e| CudaDeviceRuntimeError::driver("tail head duplicate decision", e))?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests;
