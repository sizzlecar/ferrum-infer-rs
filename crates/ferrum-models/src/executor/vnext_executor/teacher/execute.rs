use super::*;

struct CaptureGuard<'a, R: DeviceRuntime> {
    executor: &'a VNextModelExecutor<R>,
    requests: Vec<RequestId>,
    caches: Vec<String>,
}

impl<R: DeviceRuntime> Drop for CaptureGuard<'_, R> {
    fn drop(&mut self) {
        for request in &self.requests {
            self.executor.sequences.lock().cancel_prefill(request);
        }
        for cache in &self.caches {
            self.executor.release_cache(cache);
        }
        self.executor.teacher_wave_capture.lock().take();
        self.executor
            .teacher_capture_active
            .store(false, Ordering::Release);
    }
}

struct OwnerState {
    request_id: RequestId,
    cache: Arc<dyn KvCacheHandle>,
    history: Vec<u32>,
}

impl<R: DeviceRuntime> VNextModelExecutor<R> {
    async fn admit_teacher_owner(
        &self,
        request_id: &RequestId,
        prompt: &[TokenId],
        ceiling: usize,
    ) -> Result<()> {
        let mut attempts = 0;
        let mut pressure_maintenance = admission::TeacherAdmissionPressureMaintenance::default();
        loop {
            match self.try_admit_prefill(ExecutorPrefillAdmission::for_diagnostic(
                request_id, prompt, ceiling,
            ))? {
                ExecutorPrefillAdmissionDecision::Admitted(receipt) => {
                    if receipt.request_id != *request_id {
                        return Err(FerrumError::internal("teacher admission changed its owner"));
                    }
                    return Ok(());
                }
                ExecutorPrefillAdmissionDecision::MaintenanceDeferred(_) => {
                    if attempts >= MAX_BACKING_MAINTENANCE_ATTEMPTS {
                        return Err(FerrumError::resource_exhausted(
                            "teacher admission backing did not converge",
                        ));
                    }
                    attempts += 1;
                    match self.maintain_prefill_backing(request_id)? {
                        ExecutorPrefillMaintenanceOutcome::Maintained { .. }
                        | ExecutorPrefillMaintenanceOutcome::RetryAdmission { .. } => {}
                        ExecutorPrefillMaintenanceOutcome::WaitForRelease { .. } => {
                            return Err(FerrumError::resource_exhausted(
                                "teacher admission requires capacity release",
                            ));
                        }
                        ExecutorPrefillMaintenanceOutcome::NoLongerPending => {
                            return Err(FerrumError::internal("teacher admission lost its owner"));
                        }
                    }
                }
                ExecutorPrefillAdmissionDecision::Deferred(deferred) => {
                    if pressure_maintenance.try_maintain(&self.plan_resources, &deferred)? {
                        continue;
                    }
                    return Err(Self::deferred("teacher admission", &deferred));
                }
                ExecutorPrefillAdmissionDecision::PermanentRejected(rejected) => {
                    return Err(FerrumError::resource_exhausted(format!(
                        "teacher admission exceeds the real runtime capacity: {rejected:?}",
                    )));
                }
            }
        }
    }

    /// A dedicated diagnostic owns this executor for the complete capture.
    /// The same product executor methods perform all state allocation, model
    /// operations, capacity checks, completion observation and retirement.
    pub async fn collect_teacher_history(
        &self,
        spec: &VNextTeacherExecutionSpec,
        sink: &mut dyn VNextTeacherEvidenceSink,
    ) -> Result<VNextTeacherCaptureSummary> {
        spec.validate(self.io.output_elements, self.maximum_model_tokens)?;
        if self.checkpoint_capture.is_some() || self.sequences.lock().total_len() != 0 {
            return Err(FerrumError::request_validation(
                "real-history capture requires a dedicated idle executor without another checkpoint capture",
            ));
        }
        if spec.owners.len() > self.policy.memory().maximum_active_sequences as usize {
            return Err(FerrumError::resource_exhausted(
                "teacher owner count exceeds the resolved runtime slot capacity",
            ));
        }
        self.prepare_startup().await?;
        self.teacher_capture_active
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .map_err(|_| FerrumError::already_exists("a teacher capture is already active"))?;
        let mut guard = CaptureGuard {
            executor: self,
            requests: Vec::new(),
            caches: Vec::new(),
        };
        let mut states = Vec::with_capacity(spec.owners.len());
        let mut wave_index = 0;
        let mut decision_count = 0;
        // Real prefill is identical for both execution modes. Only subsequent
        // decode physical width differs; all owners retain their actual state.
        for owner in &spec.owners {
            let request_id = RequestId::new();
            guard.requests.push(request_id.clone());
            let prompt: Arc<[TokenId]> = owner
                .prompt_token_ids
                .iter()
                .copied()
                .map(TokenId::new)
                .collect::<Vec<_>>()
                .into();
            let sequence_ceiling = spec.owner_sequence_ceiling(owner)?;
            self.admit_teacher_owner(&request_id, &prompt, sequence_ceiling)
                .await?;
            let mut processed = 0;
            let mut current_cache: Option<Arc<dyn KvCacheHandle>> = None;
            while processed < prompt.len() {
                let length = spec.prefill_chunk_tokens.min(prompt.len() - processed);
                let chunk = PrefillChunk::new(processed, length, prompt.len())?;
                let input = PlanRuntimePrefillInput::new(
                    request_id.clone(),
                    Arc::clone(&prompt),
                    sequence_ceiling,
                    chunk,
                )?;
                self.begin_teacher_wave(
                    wave_index,
                    VNextExecutionWaveKind::Prefill,
                    vec![TeacherExpectedParticipant {
                        owner_id: owner.owner_id.clone(),
                        request_id: request_id.clone(),
                        cache_id: current_cache.as_ref().map(|cache| cache.cache_id()),
                        history: owner.prompt_token_ids[..chunk.end()].to_vec(),
                        immediate_start: processed,
                    }],
                )?;
                let completion = match self.plan_runtime_prefill_with_capacity(&input).await? {
                    PlanRuntimePrefillOutcome::Completed(completion) => completion,
                    PlanRuntimePrefillOutcome::Deferred(deferred) => {
                        return Err(FerrumError::resource_exhausted(format!(
                            "teacher prefill deferred before submission at {:?}",
                            deferred.stage(),
                        )));
                    }
                };
                completion.validate_for(&request_id, chunk, self.io.output_elements)?;
                let (output, _, completed, _) = completion.into_parts();
                let observed = self.finish_teacher_wave()?;
                let wave = &observed.evidence;
                if wave.participants[0].immediate_end != completed.end() {
                    return Err(FerrumError::internal(
                        "teacher prefill receipt and cache frontier differ",
                    ));
                }
                let cache = Arc::clone(output.kv_cache());
                if current_cache.is_none() {
                    guard.caches.push(cache.cache_id());
                }
                current_cache = Some(Arc::clone(&cache));
                processed = completed.end();
                sink.wave(wave, &observed.raw_readbacks, &observed.completion_receipt)?;
                wave_index += 1;
                if let PlanRuntimePrefillProduct::FinalLogits(logits) = output.product() {
                    validate_logits(logits, self.io.output_elements)?;
                    let decision = decision_evidence(
                        owner,
                        0,
                        &owner.prompt_token_ids,
                        &request_id,
                        cache.as_ref(),
                        wave,
                    )?;
                    sink.decision(&decision, logits)?;
                    decision_count += 1;
                }
            }
            let cache = current_cache
                .ok_or_else(|| FerrumError::internal("teacher prefill returned no state"))?;
            states.push(OwnerState {
                request_id,
                cache,
                history: owner.prompt_token_ids.clone(),
            });
        }
        for decision_index in 1..spec.owners[0].teacher_token_ids.len() {
            let width = match spec.mode {
                VNextTeacherMode::Serial => 1,
                VNextTeacherMode::Batched => states.len(),
            };
            for first in (0..states.len()).step_by(width) {
                let indices = first..first + width;
                let mut expected = Vec::with_capacity(width);
                let mut inputs = Vec::with_capacity(width);
                for index in indices.clone() {
                    let owner = &spec.owners[index];
                    let state = &states[index];
                    let next_token = owner.teacher_token_ids[decision_index - 1];
                    let mut history = state.history.clone();
                    history.push(next_token);
                    expected.push(TeacherExpectedParticipant {
                        owner_id: owner.owner_id.clone(),
                        request_id: state.request_id.clone(),
                        cache_id: Some(state.cache.cache_id()),
                        history,
                        immediate_start: state.history.len(),
                    });
                    inputs.push(PlanRuntimeDecodeInput::new(
                        state.request_id.clone(),
                        TokenId::new(next_token),
                        Arc::clone(&state.cache),
                    ));
                }
                self.begin_teacher_wave(wave_index, VNextExecutionWaveKind::Decode, expected)?;
                let outputs = match self
                    .plan_runtime_batch_decode_with_capacity(&inputs)
                    .await?
                {
                    PlanRuntimeBatchDecodeOutcome::Completed(outputs) => outputs,
                    PlanRuntimeBatchDecodeOutcome::Deferred(deferred) => {
                        return Err(FerrumError::resource_exhausted(format!(
                            "teacher decode width {width} deferred before submission at {:?}",
                            deferred.stage(),
                        )));
                    }
                };
                if outputs.len() != width {
                    return Err(FerrumError::internal(
                        "teacher decode lost or duplicated an owner output",
                    ));
                }
                let observed = self.finish_teacher_wave()?;
                let wave = &observed.evidence;
                sink.wave(wave, &observed.raw_readbacks, &observed.completion_receipt)?;
                wave_index += 1;
                for (index, output) in indices.zip(outputs) {
                    let owner = &spec.owners[index];
                    let state = &mut states[index];
                    if output.kv_cache.cache_id() != state.cache.cache_id() {
                        return Err(FerrumError::internal(
                            "teacher decode reordered or replaced owner caches",
                        ));
                    }
                    let logits = output.sampling_output.into_full_logits()?;
                    validate_logits(&logits, self.io.output_elements)?;
                    state
                        .history
                        .push(owner.teacher_token_ids[decision_index - 1]);
                    state.cache = output.kv_cache;
                    let decision = decision_evidence(
                        owner,
                        decision_index,
                        &state.history,
                        &state.request_id,
                        state.cache.as_ref(),
                        wave,
                    )?;
                    sink.decision(&decision, &logits)?;
                    decision_count += 1;
                }
            }
        }
        if decision_count
            != spec
                .owners
                .iter()
                .map(|owner| owner.teacher_token_ids.len())
                .sum::<usize>()
        {
            return Err(FerrumError::internal(
                "teacher capture omitted a canonical decision",
            ));
        }
        for (owner, state) in spec.owners.iter().zip(states) {
            self.complete_cache(ExecutorSequenceCompletion::new(
                state.request_id,
                state.cache.cache_id(),
                owner.prompt_token_ids.len(),
                owner.teacher_token_ids.len(),
            )?)
            .await?;
        }
        if self.sequences.lock().total_len() != 0 {
            return Err(FerrumError::internal(
                "teacher capture retained sequence resources after completion",
            ));
        }
        Ok(VNextTeacherCaptureSummary {
            mode: spec.mode,
            owner_count: spec.owners.len(),
            vocabulary_size: self.io.output_elements,
            decision_count,
            physical_wave_count: wave_index,
            resolved_plan_fingerprint: self.resolved_plan.fingerprint().to_owned(),
            family_fingerprint: self.family_fingerprint.clone(),
            program_fingerprint: self.program_fingerprint.clone(),
            output_policy: "unmodified_full_logits_before_sampling",
        })
    }
}
