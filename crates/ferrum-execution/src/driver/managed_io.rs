//! Explicit managed greedy I/O over the ordinary causal execution authorities.
//! Full prefill rows remain real Output allocations until scanning/retirement;
//! only the sampling product is reduced to an owned scalar token.
use super::*;
use crate::token_execution::ExecutionWork;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ManagedCausalIoProfile {
    /// F32 logits, finite total-order greedy selection, lowest-index ties.
    /// No token masks, repetition processors or diagnostic output captures.
    ManagedGreedyF32V1,
    /// The same existing native greedy comparator over real F16 head output.
    ManagedGreedyF16V1,
    /// Finite total-order greedy selection over real BF16 head output.
    ManagedGreedyBf16V1,
}

impl ManagedCausalIoProfile {
    pub const fn logits_element_type(self) -> ElementType {
        match self {
            Self::ManagedGreedyF32V1 => ElementType::F32,
            Self::ManagedGreedyF16V1 => ElementType::F16,
            Self::ManagedGreedyBf16V1 => ElementType::Bf16,
        }
    }

    pub(super) fn validate_logits_type(self, actual: ElementType) -> Result<()> {
        let expected = self.logits_element_type();
        if actual != expected {
            return Err(FerrumError::config(format!(
                "managed causal I/O profile {self:?} requires compiled logits {expected:?}, got {actual:?}",
            )));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ManagedCausalReadback {
    F32Logits,
    F16Logits,
    Bf16Logits,
    GreedyToken,
}

/// Planning input only. Actual waves derive and reserve their own exact demand.
#[derive(Debug, Clone, Copy)]
pub struct ManagedCausalIoWork {
    pub immediate_tokens: usize,
    /// This version reserves a cold mask owner even if its upload later hits
    /// a native slot cache. False is rejected rather than understating demand.
    pub mask_upload: bool,
    pub output: ManagedCausalReadback,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ManagedCausalIoRequirements {
    pub input_bytes: usize,
    pub output_bytes: usize,
    pub total_bytes: usize,
}

#[derive(Clone)]
pub(super) struct ManagedCausalIo {
    pub(super) host: HostMemoryAccount,
    pub(super) profile: ManagedCausalIoProfile,
}

struct ManagedRowInputs {
    causal: Vec<causal_inputs::PreparedInput>,
    tokens: ManagedHostBytes,
    token_start: u64,
    mask: ManagedHostBytes,
    offsets: ManagedHostBytes,
    penalty: ManagedHostBytes,
}

/// Actual allocations, not a nested scope or a second reservation ledger.
/// The caller's account may itself be a previously admitted phase-scope view.
pub(super) struct ManagedWaveIo {
    rows: Vec<ManagedRowInputs>,
    outputs: Option<Vec<(CompletionReadbackRequest, ManagedHostBuffer)>>,
}

impl ManagedCausalIo {
    pub(super) fn validate_config<R: DeviceRuntime>(
        &self,
        runtime: &R,
        io: &VNextIoBinding,
        config: &ProgramExecutorConfig,
        checkpoint: bool,
    ) -> Result<()> {
        self.profile.validate_logits_type(io.output_element_type)?;
        if !runtime.supports_managed_tensor_io() {
            return Err(FerrumError::config(format!(
                "managed causal I/O profile {:?} requires managed uploads/readback; compiled logits are {:?}",
                self.profile, io.output_element_type,
            )));
        }
        if checkpoint
            || config.diagnostic_output_capture.is_some()
            || config.diagnostic_fault.is_some()
        {
            return Err(FerrumError::config(
                "managed causal greedy I/O does not support diagnostic capture/fault",
            ));
        }
        Ok(())
    }

    pub(super) fn validate_policy(&self, policy: &LogitsReturnPolicy) -> Result<()> {
        match (self.profile, policy) {
            (
                ManagedCausalIoProfile::ManagedGreedyF32V1
                | ManagedCausalIoProfile::ManagedGreedyF16V1
                | ManagedCausalIoProfile::ManagedGreedyBf16V1,
                LogitsReturnPolicy::GreedyArgmax {
                    token_mask: None,
                    repetition_penalty: None,
                },
            ) => Ok(()),
            _ => Err(FerrumError::unsupported(
                "managed causal greedy I/O requires unmasked, unpenalized greedy token work",
            )),
        }
    }
}

#[cfg(test)]
pub(super) fn requirements(
    host: &HostMemoryAccount,
    vocabulary: usize,
    profile: ManagedCausalIoProfile,
    work: &[ManagedCausalIoWork],
) -> Result<ManagedCausalIoRequirements> {
    requirements_with_inputs(host, vocabulary, profile, work, None)
}

pub(super) fn requirements_with_inputs(
    host: &HostMemoryAccount,
    vocabulary: usize,
    profile: ManagedCausalIoProfile,
    work: &[ManagedCausalIoWork],
    inputs: Option<&CausalInputSchema>,
) -> Result<ManagedCausalIoRequirements> {
    if work.is_empty() || vocabulary == 0 {
        return Err(FerrumError::config(
            "managed I/O requires nonempty work and vocabulary",
        ));
    }
    let mut input_bytes = 0usize;
    let mut output_bytes = 0usize;
    for row in work {
        if !row.mask_upload {
            return Err(FerrumError::unsupported(
                "managed causal greedy v1 requires the cold mask allocation bound",
            ));
        }
        if row.immediate_tokens == 0 {
            return Err(FerrumError::config("managed token upload must be nonempty"));
        }
        let token_bytes = row
            .immediate_tokens
            .checked_mul(4)
            .ok_or_else(|| FerrumError::config("managed token bytes overflow"))?;
        for logical in [
            Some(token_bytes),
            row.mask_upload.then_some(vocabulary),
            Some(8),
            Some(4),
        ]
        .into_iter()
        .flatten()
        {
            input_bytes = input_bytes
                .checked_add(
                    host.backing_bytes(logical)
                        .map_err(|e| FerrumError::resource_exhausted(e.to_string()))?,
                )
                .ok_or_else(|| FerrumError::config("managed Input demand overflow"))?;
        }
        if let Some(schema) = inputs {
            input_bytes = input_bytes
                .checked_add(schema.additional_bytes(host, row.immediate_tokens)?)
                .ok_or_else(|| FerrumError::config("causal wave Input demand overflows"))?;
        }
        let logical = match row.output {
            ManagedCausalReadback::F32Logits
            | ManagedCausalReadback::F16Logits
            | ManagedCausalReadback::Bf16Logits => {
                profile.validate_logits_type(match row.output {
                    ManagedCausalReadback::F32Logits => ElementType::F32,
                    ManagedCausalReadback::F16Logits => ElementType::F16,
                    ManagedCausalReadback::Bf16Logits => ElementType::Bf16,
                    ManagedCausalReadback::GreedyToken => unreachable!(),
                })?;
                vocabulary
                    .checked_mul(profile.logits_element_type().size_bytes() as usize)
                    .ok_or_else(|| FerrumError::config("managed logits bytes overflow"))?
            }
            ManagedCausalReadback::GreedyToken => 4,
        };
        output_bytes = output_bytes
            .checked_add(
                host.backing_bytes(logical)
                    .map_err(|e| FerrumError::resource_exhausted(e.to_string()))?,
            )
            .ok_or_else(|| FerrumError::config("managed Output demand overflow"))?;
    }
    Ok(ManagedCausalIoRequirements {
        input_bytes,
        output_bytes,
        total_bytes: input_bytes
            .checked_add(output_bytes)
            .ok_or_else(|| FerrumError::config("managed I/O total demand overflow"))?,
    })
}

/// Owns the actual readback destinations across the completion-worker boundary.
/// If delivery is abandoned, only the original completion fact is retained;
/// synchronous managed readback guarantees these buffers have no pending DMA.
pub(super) struct ManagedWaveObservation {
    pub(super) completion: CompletionObservation,
    pub(super) outputs: Option<std::result::Result<Vec<ManagedHostBytes>, VNextError>>,
}
impl ManagedWaveObservation {
    pub(super) fn from_runtime(observation: ManagedCompletionReadbackObservation) -> Self {
        let (completion, outputs) = observation.into_parts();
        Self {
            completion,
            outputs,
        }
    }
}
impl sequence_settlement::RetainWaveObservation
    for std::result::Result<ManagedWaveObservation, VNextError>
{
    fn into_retained(self) -> sequence_settlement::RetainedWaveObservation {
        sequence_settlement::RetainedWaveObservation::Completion(self.map(|o| o.completion))
    }
}

fn select_finite_logits(
    bytes: &[u8],
    vocabulary: usize,
    element: ElementType,
) -> Result<ExecutorSamplingOutput> {
    let width = match element {
        ElementType::F32 => 4,
        ElementType::F16 | ElementType::Bf16 => 2,
        other => {
            return Err(FerrumError::backend(format!(
                "unsupported managed greedy logits {other:?}"
            )))
        }
    };
    if vocabulary == 0 || vocabulary.checked_mul(width) != Some(bytes.len()) {
        return Err(FerrumError::backend(
            "managed logits byte extent differs from compiled dtype/vocabulary",
        ));
    }
    let mut best: Option<(usize, f32)> = None;
    for (index, chunk) in bytes.chunks_exact(width).enumerate() {
        let value = match element {
            ElementType::F32 => f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]),
            ElementType::F16 => {
                half::f16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]])).to_f32()
            }
            ElementType::Bf16 => {
                half::bf16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]])).to_f32()
            }
            _ => unreachable!(),
        };
        if value.is_finite() && best.is_none_or(|(_, previous)| value.total_cmp(&previous).is_gt())
        {
            best = Some((index, value));
        }
    }
    match best {
        Some((index, _)) => Ok(ExecutorSamplingOutput::greedy_token(TokenId::new(
            u32::try_from(index).map_err(|_| FerrumError::backend("greedy index exceeds U32"))?,
        ))),
        None => Ok(ExecutorSamplingOutput::NoSelectableFiniteLogit),
    }
}

fn managed_product_mode(
    kind: VNextExecutionWaveKind,
    work: &[crate::token_execution::ExecutedTokenWork],
) -> VNextProductOutputMode {
    let roles = work
        .iter()
        .map(|work| match *work {
            crate::token_execution::ExecutedTokenWork::Prefill { chunk, .. } => {
                VNextParticipantOutputRole::prefill(chunk, LogitsReturnPolicy::FullLogits)
            }
            crate::token_execution::ExecutedTokenWork::Decode { .. } => {
                VNextParticipantOutputRole::Decode(LogitsReturnPolicy::GreedyArgmax {
                    token_mask: None,
                    repetition_penalty: None,
                })
            }
        })
        .collect::<Vec<_>>();
    // Use the ordinary driver's exact role rule: an intermediate prefill does
    // not force full logits on a simultaneous greedy decode; a final one does.
    product_output_mode_for_roles(kind, roles.iter())
}

impl<R: DeviceRuntime> ProgramExecutor<R> {
    pub fn managed_io_requirements(
        &self,
        work: &[ManagedCausalIoWork],
    ) -> Result<ManagedCausalIoRequirements> {
        let io = self
            .managed_io
            .as_ref()
            .ok_or_else(|| FerrumError::config("executor has no managed I/O profile"))?;
        io.profile
            .validate_logits_type(self.io.output_element_type)?;
        requirements_with_inputs(
            &io.host,
            self.io.output_elements,
            io.profile,
            work,
            self.io.causal_inputs.as_ref().map(|i| &i.schema),
        )
    }

    pub(super) fn validate_managed_work(&self, work: &[ExecutionWork<'_>]) -> Result<()> {
        if let Some(io) = &self.managed_io {
            for row in work {
                io.validate_policy(row.logits_policy())?;
            }
        }
        Ok(())
    }

    pub(super) fn prepare_managed_wave_with_inputs(
        &self,
        tokens: &[Vec<u32>],
        work: &[crate::token_execution::ExecutedTokenWork],
        kind: VNextExecutionWaveKind,
        inputs: &[Option<&CausalRequestInputs>],
    ) -> Result<Option<ManagedWaveIo>> {
        let Some(io) = &self.managed_io else {
            return Ok(None);
        };
        if tokens.is_empty() || tokens.len() != work.len() || work.len() != inputs.len() {
            return Err(FerrumError::backend(
                "managed wave work/token count differs",
            ));
        }
        let output_mode = managed_product_mode(kind, work);
        let batch = self.product_readback_batch(work.len(), output_mode)?;
        let mut demand = Vec::with_capacity(work.len() * 5);
        for (((tokens, work), request), inputs) in
            tokens.iter().zip(work).zip(batch.requests()).zip(inputs)
        {
            let range = work.range()?.as_range();
            if range.end > tokens.len()
                || tokens[range.clone()]
                    .iter()
                    .any(|&t| t as usize >= self.io.output_elements)
            {
                return Err(FerrumError::request_validation(
                    "managed token range or vocabulary differs",
                ));
            }
            let logical_tokens = range
                .len()
                .checked_mul(4)
                .ok_or_else(|| FerrumError::config("managed token bytes overflow"))?;
            demand.extend([
                (HostMemoryClass::InputPreprocess, logical_tokens),
                (HostMemoryClass::InputPreprocess, self.io.output_elements),
                (HostMemoryClass::InputPreprocess, 8),
                (HostMemoryClass::InputPreprocess, 4),
                (
                    HostMemoryClass::Output,
                    usize::try_from(
                        request
                            .output_layout()
                            .byte_len()
                            .map_err(|e| FerrumError::backend(e.to_string()))?,
                    )
                    .map_err(|_| FerrumError::config("managed Output exceeds address space"))?,
                ),
            ]);
            match (&self.io.causal_inputs, inputs) {
                (None, None) => (),
                (Some(binding), Some(inputs))
                    if binding.schema == *inputs.schema()
                        && range.end <= inputs.maximum_tokens() =>
                {
                    for bytes in binding.schema.zero_demands(range.len())? {
                        demand.push((HostMemoryClass::InputPreprocess, bytes));
                    }
                }
                _ => {
                    return Err(FerrumError::request_validation(
                        "wave causal inputs differ from the retained compiled request",
                    ))
                }
            }
        }
        // Atomically grant all real destinations before any native preparation
        // or frontier mutation. This works on an original or scoped account.
        let mut claims = io
            .host
            .reserve_many(&demand)
            .map_err(|e| FerrumError::resource_exhausted(e.to_string()))?
            .into_iter();
        let mut allocate = || {
            claims
                .next()
                .expect("one claim per validated allocation")
                .allocate_zeroed()
                .map_err(|e| FerrumError::resource_exhausted(e.to_string()))
        };
        let mut rows = Vec::with_capacity(work.len());
        let mut outputs = Vec::with_capacity(work.len());
        for (((tokens, work), request), inputs) in
            tokens.iter().zip(work).zip(batch.requests()).zip(inputs)
        {
            let range = work.range()?.as_range();
            let mut ids = allocate()?;
            for (slot, token) in ids.as_mut().chunks_exact_mut(4).zip(&tokens[range.clone()]) {
                slot.copy_from_slice(&token.to_le_bytes());
            }
            let mut mask = allocate()?;
            mask.as_mut().fill(1);
            let offsets = allocate()?; // sparse repetition count is exactly zero
            let mut penalty = allocate()?;
            penalty.as_mut().copy_from_slice(&1.0f32.to_le_bytes());
            let output = allocate()?;
            let causal = match inputs {
                Some(inputs) => inputs.prepare(range.clone(), &mut allocate)?,
                None => Vec::new(),
            };
            rows.push(ManagedRowInputs {
                causal,
                tokens: ids.freeze(),
                token_start: range.start as u64,
                mask: mask.freeze(),
                offsets: offsets.freeze(),
                penalty: penalty.freeze(),
            });
            outputs.push((request.clone(), output));
        }
        Ok(Some(ManagedWaveIo {
            rows,
            outputs: Some(outputs),
        }))
    }

    pub(super) fn managed_uploads(
        &self,
        wave: &ManagedWaveIo,
        participants: &[VNextExecutionParticipant<'_, R>],
        plans: &[VNextProductTokenMaskSubmissionPlan],
    ) -> Result<Vec<SubmissionWaveInputUpload>> {
        if wave.rows.len() != participants.len() || plans.len() != participants.len() {
            return Err(FerrumError::backend(
                "managed upload membership differs from preparation",
            ));
        }
        let layout = |kind, count| {
            HostTransferLayout::new(kind, count).map_err(|e| FerrumError::backend(e.to_string()))
        };
        let mut uploads = Vec::with_capacity(participants.len() * 4);
        for (index, (row, participant)) in wave.rows.iter().zip(participants).enumerate() {
            let range = participant.span.immediate_token_range();
            let length = usize::try_from(participant.span.immediate_tokens())
                .ok()
                .and_then(|n| n.checked_mul(4))
                .ok_or_else(|| FerrumError::backend("managed token extent overflow"))?;
            if range.start != row.token_start || length > row.tokens.len() {
                return Err(FerrumError::backend(
                    "managed prefill narrowing changed its initial frontier",
                ));
            }
            uploads.push(
                SubmissionWaveInputUpload::managed_range(
                    self.io.input_node_id.clone(),
                    index as u32,
                    self.io.input_ordinal,
                    range
                        .start
                        .checked_mul(4)
                        .ok_or_else(|| FerrumError::backend("managed token offset overflow"))?,
                    layout(ElementType::U32, participant.span.immediate_tokens())?,
                    row.tokens
                        .slice(0..length)
                        .map_err(|e| FerrumError::backend(e.to_string()))?,
                )
                .map_err(|e| FerrumError::backend(e.to_string()))?,
            );
            if let Some(binding) = &self.io.causal_inputs {
                uploads.extend(
                    binding.uploads(
                        &row.causal,
                        range,
                        u32::try_from(index)
                            .map_err(|_| FerrumError::backend("causal participant exceeds u32"))?,
                    )?,
                );
            } else if !row.causal.is_empty() {
                return Err(FerrumError::backend(
                    "wave retained undeclared causal inputs",
                ));
            }
        }
        for (index, (row, plan)) in wave.rows.iter().zip(plans).enumerate() {
            if plan.upload_required {
                uploads.push(
                    SubmissionWaveInputUpload::managed_range(
                        self.io.token_mask_input_node_id.clone(),
                        index as u32,
                        self.io.token_mask_input_ordinal,
                        0,
                        layout(ElementType::U8, self.io.output_elements as u64)?,
                        row.mask.clone(),
                    )
                    .map_err(|e| FerrumError::backend(e.to_string()))?,
                );
            }
        }
        for (node, ordinal, kind, count, select) in [
            (
                &self.io.repetition_offsets_input_node_id,
                self.io.repetition_offsets_input_ordinal,
                ElementType::U32,
                2,
                false,
            ),
            (
                &self.io.repetition_penalty_input_node_id,
                self.io.repetition_penalty_input_ordinal,
                ElementType::F32,
                1,
                true,
            ),
        ] {
            for (index, row) in wave.rows.iter().enumerate() {
                uploads.push(
                    SubmissionWaveInputUpload::managed_range(
                        node.clone(),
                        index as u32,
                        ordinal,
                        0,
                        layout(kind, count)?,
                        if select {
                            row.penalty.clone()
                        } else {
                            row.offsets.clone()
                        },
                    )
                    .map_err(|e| FerrumError::backend(e.to_string()))?,
                );
            }
        }
        Ok(uploads)
    }

    pub(super) fn managed_readbacks(
        &self,
        readbacks: &VNextTerminalReadbacks,
        wave: &mut Option<ManagedWaveIo>,
    ) -> Result<Option<Vec<(CompletionReadbackRequest, ManagedHostBuffer)>>> {
        if self.managed_io.is_none() {
            return Ok(None);
        };
        let VNextTerminalReadbacks::Batch(batch) = readbacks else {
            return Err(FerrumError::unsupported(
                "managed greedy readback requires one uniform product batch",
            ));
        };
        let prepared = wave
            .as_mut()
            .ok_or_else(|| FerrumError::backend("managed wave has no original Host owners"))?;
        let outputs = prepared
            .outputs
            .take()
            .ok_or_else(|| FerrumError::backend("managed outputs were already consumed"))?;
        if outputs.iter().map(|(r, _)| r).ne(batch.requests().iter()) {
            return Err(FerrumError::backend(
                "managed Output requests differ from original preparation",
            ));
        }
        Ok(Some(outputs))
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) async fn finish_managed_participants<'a>(
        &'a self,
        mut guard: PreparedWaveGuard<'a, R>,
        completion: CompletionHandle<R>,
        participants: &'a [VNextExecutionParticipant<'a, R>],
        kind: VNextExecutionWaveKind,
        output_modes: ProductReadbackModes,
        mut mask_residency: VNextProductTokenMaskResidencyTransaction<'a>,
        readbacks: Vec<(CompletionReadbackRequest, ManagedHostBuffer)>,
        mut event_error: Option<String>,
        resident_install: Option<Arc<ResidentDecodeProgram<R>>>,
    ) -> std::result::Result<RetiredWave<Vec<ExecutorSamplingOutput>, R>, WaveFailure> {
        let requests = readbacks
            .iter()
            .map(|(request, _)| request.clone())
            .collect::<Vec<_>>();
        let segment = completion.take_segment_binding_publication();
        let route = guard.worker_route();
        let reaper = Arc::clone(&self.reaper);
        let observation = self
            .completion_worker
            .execute(VNextCompletionTaskKind::WaveReadback, move || {
                let observed = completion
                    .wait_with_managed_readbacks(readbacks)
                    .map(ManagedWaveObservation::from_runtime);
                drop(reaper);
                route.wrap(observed)
            })
            .await
            .map_err(|error| {
                guard.fail(FerrumError::backend(format!(
                    "managed completion task failed: {error}"
                )))
            })?
            .into_inner(&mut guard);
        let ManagedWaveObservation {
            completion,
            outputs,
        } = match observation {
            Ok(observation) => observation,
            Err(error) => {
                let message = error.to_string();
                guard.accept_observation(Err(error));
                return Err(guard.fail(FerrumError::backend(message)));
            }
        };
        let terminal = matches!(&completion, CompletionObservation::Terminal(_));
        guard.accept_observation(Ok(completion));
        if !terminal {
            mask_residency.invalidate_before_slot_release();
            return Err(guard.fail(FerrumError::backend(
                "managed readback remained nonterminal",
            )));
        }
        let receipt = guard.completion_receipt().clone();
        if !matches!(
            receipt.disposition(),
            OperationCompletionDisposition::Succeeded
        ) {
            mask_residency.invalidate_before_slot_release();
            return Err(guard.fail(FerrumError::backend(
                "managed causal compute did not succeed",
            )));
        }
        let outputs = outputs
            .ok_or_else(|| guard.fail(FerrumError::backend("managed completion omitted outputs")))?
            .map_err(|e| guard.fail(FerrumError::backend(e.to_string())))?;
        if outputs.len() != participants.len() || outputs.len() != requests.len() {
            return Err(guard.fail(FerrumError::backend(
                "managed product row count differs from participants",
            )));
        }
        self.refresh_on_demand_reusable_execution_catalog()
            .map_err(|e| guard.fail(e))?;
        self.metrics.device_timing.record_completion(&receipt);
        self.metrics
            .device_timing_for(kind)
            .record_completion(&receipt);
        for participant in participants {
            if let Some(events) = &participant.sequence.events {
                if let Err(error) = events.lock().completed(&receipt) {
                    event_error.get_or_insert_with(|| error.to_string());
                }
            }
        }
        mask_residency.publish();
        mask_residency.settle_success();
        let mut products = Vec::with_capacity(outputs.len());
        let mut bytes = 0u64;
        for (index, (request, owner)) in requests.iter().zip(&outputs).enumerate() {
            let io = self
                .managed_io
                .as_ref()
                .expect("managed path retains its account");
            if !io.host.owns(owner)
                || owner.class() != HostMemoryClass::Output
                || request.participant_index() as usize != index
            {
                return Err(guard.fail(FerrumError::backend(
                    "managed Output provenance/order differs",
                )));
            }
            request
                .output_layout()
                .validate_bytes(owner.len())
                .map_err(|e| guard.fail(FerrumError::backend(e.to_string())))?;
            bytes += owner.len() as u64;
            let product = match output_modes.at(index) {
                Some(VNextProductOutputMode::FullLogits) => select_finite_logits(
                    owner.bytes(),
                    self.io.output_elements,
                    self.io.output_element_type,
                ),
                Some(VNextProductOutputMode::GreedyToken)
                    if owner.bytes() == u32::MAX.to_le_bytes() =>
                {
                    Ok(ExecutorSamplingOutput::NoSelectableFiniteLogit)
                }
                Some(VNextProductOutputMode::GreedyToken) => {
                    decode_selected_token(owner.bytes(), self.io.output_elements)
                        .map(ExecutorSamplingOutput::greedy_token)
                }
                None => Err(FerrumError::backend(
                    "managed output has no declared product mode",
                )),
            }
            .map_err(|e| guard.fail(e))?;
            products.push(product);
        }
        self.metrics
            .readback_bytes
            .fetch_add(bytes, Ordering::Relaxed);
        self.metrics
            .product_readback_bytes
            .fetch_add(bytes, Ordering::Relaxed);
        self.metrics
            .no_selectable_finite_logit_participants
            .fetch_add(
                products
                    .iter()
                    .filter(|p| matches!(p, ExecutorSamplingOutput::NoSelectableFiniteLogit))
                    .count() as u64,
                Ordering::Relaxed,
            );
        let full = output_modes.full_participants() as u64;
        self.metrics
            .full_logits_readback_participants
            .fetch_add(full, Ordering::Relaxed);
        self.metrics
            .greedy_token_readback_participants
            .fetch_add(participants.len() as u64 - full, Ordering::Relaxed);
        match output_modes.uniform() {
            Some(VNextProductOutputMode::FullLogits) => &self.metrics.full_logits_readback_waves,
            Some(VNextProductOutputMode::GreedyToken) => &self.metrics.greedy_token_readback_waves,
            None => &self.metrics.hybrid_readback_waves,
        }
        .fetch_add(1, Ordering::Relaxed);
        for plan in mask_residency.plans() {
            if plan.upload_required {
                &self.metrics.token_mask_upload_participants
            } else {
                &self.metrics.token_mask_cache_hit_participants
            }
            .fetch_add(1, Ordering::Relaxed);
        }
        let ready_segment = segment.and_then(|ticket| ticket.bind_completion(&receipt));
        let retired = guard.retire_normal(products)?;
        // The actual prefill/full-row consumer outlives scanning and checked
        // Step retirement. No unaccounted f32 Vec escaped this boundary.
        drop(outputs);
        self.metrics.completed_waves.fetch_add(1, Ordering::Relaxed);
        if let Some(error) = guard.observation_error() {
            event_error.get_or_insert_with(|| error.to_string());
        }
        if let Some(error) = event_error {
            return Err(retired.retirement.fail(FerrumError::backend(error)));
        }
        if let Some(ready) = ready_segment {
            let _ = ready.publish(retired.retirement.receipt());
        }
        if let Some(program) = resident_install {
            self.install_resident_decode(program);
        }
        Ok(retired)
    }
}

#[cfg(test)]
mod tests;
