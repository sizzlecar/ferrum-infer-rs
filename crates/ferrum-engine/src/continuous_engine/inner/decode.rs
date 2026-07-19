use super::*;

#[derive(Debug)]
pub(in crate::continuous_engine) enum PlanRuntimeDecodeBatchOutcome {
    Completed,
    Deferred {
        request_ids: Vec<RequestId>,
        deferral: ExecutorExecutionCapacityDeferral,
    },
}

impl EngineInner {
    // ── batch decode ──────────────────────────────────────────────────

    /// Executes one PlanRuntime decode cohort through the typed batch API.
    /// Output cardinality and cache identity are validated before any request
    /// publishes a sampled token or updated physical-resource handle.
    pub(in crate::continuous_engine) async fn run_plan_runtime_batch_decode(
        &self,
        request_ids: &[RequestId],
    ) -> Result<PlanRuntimeDecodeBatchOutcome> {
        let mut rids = Vec::with_capacity(request_ids.len());
        let mut inputs = Vec::with_capacity(request_ids.len());
        {
            let sequences = self.sequences.read();
            for rid in request_ids {
                let Some(sequence) = sequences.get(rid) else {
                    continue;
                };
                let Some(resources) = sequence.ready_decode_resources(rid) else {
                    continue;
                };
                let tensor = self.tokens_to_tensor(&[resources.last_token.get()])?;
                let input =
                    ferrum_interfaces::model_executor::DecodeInput::new(tensor, resources.kv_cache)
                        .with_request_id(rid.clone())
                        .with_metadata(sequence.model_decode_metadata())
                        .with_logits_policy(sequence.model_decode_logits_policy());
                inputs.push(if let Some(state) = resources.recurrent_state {
                    input.with_recurrent_state(state)
                } else {
                    input
                });
                rids.push(rid.clone());
            }
        }
        if inputs.is_empty() {
            return Ok(PlanRuntimeDecodeBatchOutcome::Completed);
        }

        let input_cache_ids = inputs
            .iter()
            .map(|input| input.kv_cache.cache_id())
            .collect::<Vec<_>>();
        let input_recurrent_states = inputs
            .iter()
            .map(|input| input.recurrent_state.clone())
            .collect::<Vec<_>>();
        let workspace_lease = self.acquire_backend_workspace_lease(
            rids.clone(),
            "engine_plan_runtime_batch_decode_workspace",
            "engine_plan_runtime_batch_decode_workspace_release",
        );
        let outputs = match self
            .model_executor
            .batch_decode_with_capacity(&inputs)
            .await
        {
            Ok(ExecutorBatchDecodeOutcome::Completed(outputs)) => {
                workspace_lease.release();
                outputs
            }
            Ok(ExecutorBatchDecodeOutcome::Deferred(deferral)) => {
                workspace_lease.release();
                return Ok(PlanRuntimeDecodeBatchOutcome::Deferred {
                    request_ids: rids,
                    deferral,
                });
            }
            Err(error) => {
                drop(workspace_lease);
                return Err(error);
            }
        };
        if outputs.len() != rids.len() {
            return Err(FerrumError::internal(format!(
                "PlanRuntime batch_decode returned {} outputs for {} requests",
                outputs.len(),
                rids.len()
            )));
        }
        for (index, (output, expected_cache_id)) in outputs.iter().zip(&input_cache_ids).enumerate()
        {
            let actual_cache_id = output.kv_cache.cache_id();
            if &actual_cache_id != expected_cache_id {
                return Err(FerrumError::internal(format!(
                    "PlanRuntime batch_decode output {index} returned cache `{actual_cache_id}`, expected `{expected_cache_id}`"
                )));
            }
        }
        let logits = outputs
            .iter()
            .map(|output| output.logits.to_vec_f32())
            .collect::<Result<Vec<_>>>()?;

        for (((rid, output), input_recurrent_state), mut logits) in rids
            .iter()
            .zip(outputs)
            .zip(input_recurrent_states)
            .zip(logits)
        {
            let next_token_result = {
                let mut sequences = self.sequences.write();
                sequences.get_mut(rid).map(|sequence| {
                    let token = if logits.len() == 1 {
                        let token = TokenId::new(logits[0] as u32);
                        sequence.accept_model_greedy_argmax_token(
                            Some(self.tokenizer.as_ref()),
                            token,
                        )?;
                        token
                    } else {
                        sequence.sample_with_processors_with_tokenizer(
                            &mut logits,
                            Some(self.tokenizer.as_ref()),
                        )?
                    };
                    sequence.generated_tokens.push(token);
                    sequence.commit_decode_step_physical_resources(output.kv_cache.clone())?;
                    sequence.commit_decode_recurrent_state(
                        output.recurrent_state.clone().or(input_recurrent_state),
                    );
                    Ok::<TokenId, FerrumError>(token)
                })
            };
            let Some(next_token_result) = next_token_result else {
                continue;
            };
            let next_token = match next_token_result {
                Ok(token) => token,
                Err(error) => {
                    warn!("PlanRuntime batch decode post-process failed for {rid}: {error}");
                    self.complete_request_with_error(rid, error).await?;
                    continue;
                }
            };

            let generated_count = self
                .sequences
                .read()
                .get(rid)
                .map(|sequence| sequence.generated_tokens.len())
                .unwrap_or(0);
            self.scheduler.update_decode_progress(rid, generated_count);
            self.total_decode_tokens.fetch_add(1, Ordering::Relaxed);
            counter!("ferrum.engine.decode_tokens_total").increment(1);

            let stop_reason = self.stop_reason_for_request(rid);
            if self.should_stream_generated_token(stop_reason) {
                self.send_stream_update(rid, next_token).await;
            }
            if let Some(reason) = stop_reason {
                self.complete_request(rid, reason).await?;
            }
        }
        Ok(PlanRuntimeDecodeBatchOutcome::Completed)
    }

    pub(in crate::continuous_engine) async fn run_plan_runtime_batch_decode_adaptive(
        &self,
        request_ids: &[RequestId],
    ) -> Result<()> {
        let mut stack = vec![self.decode_ready_request_ids(request_ids)];
        while let Some(chunk) = stack.pop() {
            let chunk = self.decode_ready_request_ids(&chunk);
            if chunk.is_empty() {
                continue;
            }
            let outcome = match self.run_plan_runtime_batch_decode(&chunk).await {
                Ok(outcome) => outcome,
                Err(error) => {
                    warn!(
                        "Plan-runtime adaptive decode failed for {} exact request(s): {}",
                        chunk.len(),
                        error
                    );
                    self.write_scheduler_trace_event(serde_json::json!({
                        "event": "engine_plan_runtime_decode_subcohort_failure",
                        "request_ids": chunk,
                        "error": error.to_string(),
                        "scheduler": self.scheduler.trace_snapshot(),
                    }));
                    for request_id in &chunk {
                        self.complete_request(request_id, FinishReason::Error)
                            .await?;
                    }
                    continue;
                }
            };
            match outcome {
                PlanRuntimeDecodeBatchOutcome::Completed => {}
                PlanRuntimeDecodeBatchOutcome::Deferred {
                    request_ids,
                    deferral,
                } if request_ids.len() > 1 => {
                    self.trace_executor_decode_capacity_decision(
                        &request_ids,
                        &deferral,
                        "split_cohort",
                        None,
                    );
                    self.scheduler
                        .record_decode_capacity_pressure(request_ids.len(), None);
                    let mid = request_ids.len() / 2;
                    stack.push(request_ids[mid..].to_vec());
                    stack.push(request_ids[..mid].to_vec());
                }
                PlanRuntimeDecodeBatchOutcome::Deferred {
                    request_ids,
                    deferral,
                } => {
                    let observed = deferral.observed();
                    let scheduler_deferral = AdmissionDeferral::new(
                        DeferredAction::WaitForRelease,
                        AdmissionWakeEpochs::new(
                            observed.coordinator_id,
                            observed.release_epoch,
                            observed.capacity_epoch,
                            0,
                        ),
                        deferral.wait_condition().clone(),
                    );
                    let release_snapshot = self.execution_capacity_release_snapshot()?;
                    match self.scheduler.defer_decode_for_execution_capacity(
                        &request_ids,
                        scheduler_deferral,
                        &release_snapshot,
                    )? {
                        ExecutionCapacityAction::Deferred { count } => {
                            if count != request_ids.len() {
                                return Err(FerrumError::scheduler(format!(
                                    "PlanRuntime decode deferral retained {count} of {} scheduler entries",
                                    request_ids.len()
                                )));
                            }
                            self.trace_executor_decode_capacity_decision(
                                &request_ids,
                                &deferral,
                                "wait_for_release",
                                None,
                            );
                            self.write_scheduler_trace_event(serde_json::json!({
                                "event": "scheduler_execution_capacity_defer",
                                "request_ids": request_ids,
                                "stage": deferral.stage(),
                                "observed": observed,
                                "wait_condition": deferral.wait_condition(),
                                "scheduler": self.scheduler.trace_snapshot(),
                            }));
                        }
                        ExecutionCapacityAction::YieldPlanned { transaction } => {
                            let victim_id = transaction.victim_request_id().clone();
                            let progress_owner_id = transaction.progress_owner_id().clone();
                            let progress_baseline = transaction.progress_baseline().get();
                            self.trace_executor_decode_capacity_decision(
                                &request_ids,
                                &deferral,
                                "pressure_yield_planned",
                                Some(&transaction),
                            );
                            let progress_owner_resumable = self
                                .execute_capacity_yield(
                                    &transaction,
                                    request_ids.len().max(1),
                                    None,
                                )
                                .await?;
                            self.write_scheduler_trace_event(serde_json::json!({
                                "event": "scheduler_execution_capacity_yield_planned",
                                "request_ids": &request_ids,
                                "episode_id": transaction.episode_id().get(),
                                "handoff_generation": transaction.handoff_generation(),
                                "yield_kind": transaction.kind().as_str(),
                                "planned_transition_ordinal": transaction.planned_ordinal().get(),
                                "victim_request_id": victim_id,
                                "progress_owner_id": progress_owner_id,
                                "progress_baseline": progress_baseline,
                                "stage": deferral.stage(),
                                "observed": observed,
                                "wait_condition": deferral.wait_condition(),
                                "scheduler": self.scheduler.trace_snapshot(),
                            }));
                            // The yielded request no longer owns runnable physical
                            // resources. Resume the frontier named by the completed
                            // release transaction instead of retrying the victim.
                            if progress_owner_resumable {
                                stack.push(vec![progress_owner_id]);
                            }
                        }
                        ExecutionCapacityAction::InvariantViolation { violation } => {
                            return Err(FerrumError::internal(format!(
                                "execution-capacity pressure episode {} violated {:?}",
                                violation.episode_id().get(),
                                violation.class()
                            )));
                        }
                    }
                }
            }
        }
        Ok(())
    }

    pub(super) async fn run_batch_decode_adaptive(&self, request_ids: &[RequestId]) -> Result<()> {
        self.run_batch_decode_adaptive_inner(request_ids, true)
            .await
    }

    pub(super) async fn run_batch_decode_adaptive_no_preempt(
        &self,
        request_ids: &[RequestId],
    ) -> Result<()> {
        self.run_batch_decode_adaptive_inner(request_ids, false)
            .await
    }

    async fn run_batch_decode_adaptive_inner(
        &self,
        request_ids: &[RequestId],
        allow_preempt: bool,
    ) -> Result<()> {
        let pressure_width = request_ids.len().max(1);
        let mut stack = vec![(self.decode_ready_request_ids(request_ids), pressure_width)];
        while let Some((chunk, pressure_width)) = stack.pop() {
            let chunk = self.decode_ready_request_ids(&chunk);
            if chunk.is_empty() {
                continue;
            }
            match self.run_batch_decode(&chunk).await {
                Ok(()) => {}
                Err(e) if is_resource_exhausted_error(&e) && chunk.len() > 1 => {
                    let pressure = paged_kv_admission_pressure(&e);
                    self.scheduler.record_decode_capacity_pressure(
                        pressure_width,
                        pressure.map(|pressure| pressure.free_blocks),
                    );
                    let mid = chunk.len() / 2;
                    stack.push((chunk[mid..].to_vec(), pressure_width));
                    stack.push((chunk[..mid].to_vec(), pressure_width));
                }
                Err(e) if is_resource_exhausted_error(&e) => {
                    if !allow_preempt {
                        let pressure = paged_kv_admission_pressure(&e);
                        if let Some(pressure) = pressure {
                            self.scheduler
                                .defer_capacity_deferred_mixed_recompute_until_kv_capacity(
                                    Some(pressure.admission_blocks),
                                    Some(pressure.free_blocks),
                                    Some(chunk.len().max(1)),
                                );
                        } else {
                            self.scheduler
                                .defer_capacity_deferred_mixed_recompute_until_release();
                        }
                        for rid in &chunk {
                            if !self
                                .defer_decode_for_capacity_recompute(
                                    rid,
                                    pressure_width,
                                    pressure.map(|pressure| pressure.free_blocks),
                                )
                                .await
                            {
                                warn!(
                                    "Batch decode deferred for request {}: capacity pressure with no scheduler decode entry",
                                    rid
                                );
                            }
                        }
                        continue;
                    }
                    let exclude: std::collections::HashSet<RequestId> =
                        chunk.iter().cloned().collect();
                    if self.preempt_victim_excluding(&exclude).await {
                        stack.push((chunk, pressure_width));
                    } else {
                        warn!(
                            "Batch decode deferred for {} request(s): no preempt victim",
                            chunk.len()
                        );
                        continue;
                    }
                }
                Err(e) => return Err(e),
            }
        }
        Ok(())
    }

    /// Run batch decode for multiple requests in a single forward pass.
    ///
    /// Dispatches via the unified-batch API: we build a `UnifiedBatch`
    /// of decode-only items (each `q_len = 1`, `is_final_chunk = true`)
    /// and call `model_executor.unified_decode(...)`. The default fallback
    /// in `LlmModelExecutor` recognises the all-decode shape and reroutes
    /// to the existing batched decode path; once `LlmFamilyModel` ships
    /// a real unified forward (Step 5), the same call benefits from the
    /// chunked-prefill kernel work without further engine changes.
    pub(super) async fn run_batch_decode(&self, request_ids: &[RequestId]) -> Result<()> {
        use ferrum_interfaces::model_executor::{UnifiedBatch, UnifiedBatchItem};

        let mut rids: Vec<RequestId> = Vec::new();

        // Build the unified batch from sequence state.
        let mut batch = UnifiedBatch::new();
        {
            let sequences = self.sequences.read();
            for rid in request_ids {
                let Some(seq) = sequences.get(rid) else {
                    continue;
                };
                let Some(resources) = seq.ready_decode_resources(rid) else {
                    continue;
                };
                batch.items.push(UnifiedBatchItem {
                    seq_id: resources.seq_id,
                    q_tokens: vec![resources.last_token.get()],
                    kv_cache: resources.kv_cache,
                    recurrent_state: resources.recurrent_state,
                    pos_offset: resources.pos_offset,
                    is_final_chunk: true,
                    metadata: seq.model_decode_metadata(),
                    logits_policy: seq.model_decode_logits_policy(),
                });
                rids.push(rid.clone());
            }
        }
        if batch.items.is_empty() {
            return Ok(());
        }

        let kv_requests = kv_slot_requests_for_unified_batch(&batch);
        self.model_executor.reserve_kv_slots(&kv_requests)?;

        let prof = self.runtime_config.rbd_prof;
        let t_decode = if prof { Some(Instant::now()) } else { None };
        let workspace_lease = self.acquire_backend_workspace_lease(
            rids.clone(),
            "engine_batch_decode_workspace",
            "engine_batch_decode_workspace_release",
        );
        let results = match self.model_executor.unified_decode(&batch).await {
            Ok(results) => {
                workspace_lease.release();
                results
            }
            Err(error) => {
                drop(workspace_lease);
                return Err(error);
            }
        };
        if results.len() != rids.len() {
            return Err(FerrumError::internal(format!(
                "unified_decode returned {} results for {} requests",
                results.len(),
                rids.len(),
            )));
        }
        let t_decode_done = if prof { Some(Instant::now()) } else { None };
        let mut t_sample_us: u64 = 0;
        let mut t_sched_us: u64 = 0;
        let mut t_stream_us: u64 = 0;
        let mut t_stop_us: u64 = 0;
        let mut t_complete_us: u64 = 0;

        // Per-item post-processing: sample, update sequence state, stream
        // the new token, check stop conditions. Decode-only items always
        // produce Some(logits); a None here would indicate a backend bug.
        for (rid, logits_opt) in rids.iter().zip(results.into_iter()) {
            let mut logits = logits_opt.ok_or_else(|| {
                FerrumError::internal(format!(
                    "unified_decode returned None for decode item (rid={rid})"
                ))
            })?;

            let t0_sample = if prof { Some(Instant::now()) } else { None };
            let next_token_result = {
                let mut sequences = self.sequences.write();
                let seq = sequences
                    .get_mut(rid)
                    .ok_or_else(|| FerrumError::internal("Sequence not found"))?;
                // Greedy fast path: the model did GPU argmax and emitted one
                // f32 carrying the token id. The sequence still validates that
                // this request is eligible for model-side argmax and that the
                // returned token satisfies the same hard token-quality masks.
                let token = if logits.len() == 1 {
                    let token = TokenId::new(logits[0] as u32);
                    seq.accept_model_greedy_argmax_token(Some(self.tokenizer.as_ref()), token)?;
                    token
                } else {
                    seq.sample_with_processors_with_tokenizer(
                        &mut logits,
                        Some(self.tokenizer.as_ref()),
                    )?
                };
                seq.generated_tokens.push(token);
                let cache_id = seq.decode_model_cache_id_or_request_id(rid);
                let kv_len = seq.decode_model_kv_len_after_last_generated_token();
                let model_kv = self.make_model_kv_handle_with_seq(cache_id, kv_len);
                seq.commit_decode_step_physical_resources(model_kv)?;
                // pos_offset is sourced from SequenceState bookkeeping
                // (see process_batch_unified). The engine-side KV handle's
                // sequence_length is not used for position tracking
                // anymore — production handles (Paged/Default) don't
                // update it across iterations.
                Ok::<TokenId, FerrumError>(token)
            };
            let next_token = match next_token_result {
                Ok(token) => token,
                Err(e) => {
                    warn!("Batch decode post-process failed for {}: {}", rid, e);
                    self.complete_request_with_error(rid, e).await?;
                    continue;
                }
            };

            if let Some(t0) = t0_sample {
                t_sample_us += t0.elapsed().as_micros() as u64;
            }

            let t0_sched = if prof { Some(Instant::now()) } else { None };
            let generated_count = {
                let sequences = self.sequences.read();
                sequences
                    .get(rid)
                    .map(|s| s.generated_tokens.len())
                    .unwrap_or(0)
            };
            self.scheduler.update_decode_progress(rid, generated_count);
            self.total_decode_tokens.fetch_add(1, Ordering::Relaxed);
            counter!("ferrum.engine.decode_tokens_total").increment(1);
            if let Some(t0) = t0_sched {
                t_sched_us += t0.elapsed().as_micros() as u64;
            }

            let stop_reason = self.stop_reason_for_request(rid);
            let t0_stream = if prof { Some(Instant::now()) } else { None };
            if self.should_stream_generated_token(stop_reason) {
                self.send_stream_update(rid, next_token).await;
            }
            if let Some(t0) = t0_stream {
                t_stream_us += t0.elapsed().as_micros() as u64;
            }

            let t0_stop = if prof { Some(Instant::now()) } else { None };
            if let Some(t0) = t0_stop {
                t_stop_us += t0.elapsed().as_micros() as u64;
            }
            if let Some(reason) = stop_reason {
                let t0_comp = if prof { Some(Instant::now()) } else { None };
                self.complete_request(rid, reason).await?;
                if let Some(t0) = t0_comp {
                    t_complete_us += t0.elapsed().as_micros() as u64;
                }
            }
        }

        if let (Some(t0), Some(t1)) = (t_decode, t_decode_done) {
            use std::sync::atomic::AtomicU64;
            static N: AtomicU64 = AtomicU64::new(0);
            let n = N.fetch_add(1, Ordering::Relaxed);
            if n.is_multiple_of(32) {
                let decode_us = t1.duration_since(t0).as_micros() as u64;
                let total_post_us =
                    t_sample_us + t_sched_us + t_stream_us + t_stop_us + t_complete_us;
                eprintln!(
                    "[rbd-prof] iter#{} m={} decode={}us post={}us | sample={} sched={} stream={} stop={} complete={} (us)",
                    n,
                    rids.len(),
                    decode_us,
                    total_post_us,
                    t_sample_us,
                    t_sched_us,
                    t_stream_us,
                    t_stop_us,
                    t_complete_us,
                );
            }
        }
        Ok(())
    }
    // ── decode step ────────────────────────────────────────────────────

    pub(super) async fn run_decode_step(&self, request_id: &RequestId) -> Result<()> {
        // Speculative decoding path: when both a draft executor and
        // config are set, delegate to the runner and push the accepted
        // tokens onto the sequence in one shot.
        // The legacy speculative runner can express only raw greedy sampling.
        // Any host processor, grammar, or token-validity mask must stay on the
        // ordinary decode path so request semantics are never silently skipped.
        let can_use_speculative_decode = self
            .sequences
            .read()
            .get(request_id)
            .is_some_and(SequenceState::supports_raw_speculative_decode);
        if self.draft_executor.is_some() && self.spec_config.is_some() && can_use_speculative_decode
        {
            return self.run_decode_step_speculative(request_id).await;
        }

        let decode_input = {
            let sequences = self.sequences.read();
            let seq = sequences
                .get(request_id)
                .ok_or_else(|| FerrumError::internal("Sequence not found"))?;
            let resources = seq
                .decode_resources(request_id)
                .ok_or_else(|| FerrumError::internal("No KV cache"))?;
            let tensor = self.tokens_to_tensor(&[resources.last_token.get()])?;
            let input =
                ferrum_interfaces::model_executor::DecodeInput::new(tensor, resources.kv_cache)
                    .with_request_id(request_id.clone())
                    .with_metadata(seq.model_decode_metadata())
                    .with_logits_policy(seq.model_decode_logits_policy());
            if let Some(state) = resources.recurrent_state {
                input.with_recurrent_state(state)
            } else {
                input
            }
        };

        let input_recurrent_state = decode_input.recurrent_state.clone();
        let workspace_lease = self.acquire_backend_workspace_lease(
            vec![request_id.clone()],
            "engine_decode_workspace",
            "engine_decode_workspace_release",
        );
        let decode_output = match self.model_executor.decode(&decode_input).await {
            Ok(output) => {
                workspace_lease.release();
                output
            }
            Err(error) => {
                drop(workspace_lease);
                return Err(error);
            }
        };
        let logits_vec = decode_output.logits.to_vec_f32()?;

        let next_token_result = (|| {
            let mut sequences = self.sequences.write();
            let seq = sequences
                .get_mut(request_id)
                .ok_or_else(|| FerrumError::internal("Sequence not found"))?;
            let mut logits = logits_vec;
            let token = if logits.len() == 1 {
                let token = TokenId::new(logits[0] as u32);
                seq.accept_model_greedy_argmax_token(Some(self.tokenizer.as_ref()), token)?;
                token
            } else {
                seq.sample_with_processors_with_tokenizer(
                    &mut logits,
                    Some(self.tokenizer.as_ref()),
                )?
            };
            seq.generated_tokens.push(token);
            seq.commit_decode_step_physical_resources(decode_output.kv_cache.clone())?;
            seq.commit_decode_recurrent_state(
                decode_output
                    .recurrent_state
                    .clone()
                    .or(input_recurrent_state),
            );
            Ok::<TokenId, FerrumError>(token)
        })();
        let next_token = match next_token_result {
            Ok(token) => token,
            Err(error) => {
                warn!("Decode post-process failed for {}: {}", request_id, error);
                self.complete_request_with_error(request_id, error).await?;
                return Ok(());
            }
        };

        let generated_count = {
            let sequences = self.sequences.read();
            sequences
                .get(request_id)
                .map(|s| s.generated_tokens.len())
                .unwrap_or(0)
        };
        self.scheduler
            .update_decode_progress(request_id, generated_count);
        self.total_decode_tokens.fetch_add(1, Ordering::Relaxed);
        counter!("ferrum.engine.decode_tokens_total").increment(1);

        let stop_reason = self.stop_reason_for_request(request_id);
        if self.should_stream_generated_token(stop_reason) {
            self.send_stream_update(request_id, next_token).await;
        }

        if let Some(reason) = stop_reason {
            self.complete_request(request_id, reason).await?;
        }

        Ok(())
    }

    /// Speculative-decoding variant of `run_decode_step`. Lazily prefills
    /// the draft model's KV cache on first call (same prompt as target).
    /// Then each iteration produces 1..=N+1 tokens via `SpeculativeRunner`.
    async fn run_decode_step_speculative(&self, request_id: &RequestId) -> Result<()> {
        use ferrum_interfaces::model_executor::PrefillInput;

        let (draft_exec, cfg_base) = match (&self.draft_executor, &self.spec_config) {
            (Some(d), Some(c)) => (d.clone(), c.clone()),
            _ => unreachable!("speculative gate checked in run_decode_step"),
        };

        // Use the caller's sampling temperature for accept/reject, not the
        // engine-default from spec_config. Otherwise a greedy request
        // (temperature=0) runs through a T=1 verifier and ULP-level fp32
        // noise between draft/target causes stochastic rejections → KV
        // misalignment → output drift.
        let per_request_temperature = {
            let sequences = self.sequences.read();
            sequences
                .get(request_id)
                .map(|s| s.sampling_params.temperature)
                .unwrap_or(cfg_base.temperature)
        };
        let cfg = crate::speculative::SpeculativeDecodingConfig {
            num_speculative_tokens: cfg_base.num_speculative_tokens,
            temperature: per_request_temperature,
        };

        // ── 1. Ensure draft KV is prefilled once with the full prompt ────
        let draft_kv_ready = {
            let sequences = self.sequences.read();
            sequences
                .get(request_id)
                .and_then(SequenceState::draft_kv_cache_handle)
        };
        let draft_kv = if let Some(kv) = draft_kv_ready {
            kv
        } else {
            // First spec iteration — prefill the draft on the prompt only.
            // The already-sampled `last_token` is passed into the runner in
            // step() below (as `last_token`) — the runner's first draft
            // decode consumes it, writing KV at position prompt_len exactly
            // as target's decode path does. DO NOT pre-consume it here.
            let prompt_u32s = {
                let sequences = self.sequences.read();
                let seq = sequences
                    .get(request_id)
                    .ok_or_else(|| FerrumError::internal("Sequence not found"))?;
                seq.input_tokens.iter().map(|t| t.get()).collect::<Vec<_>>()
            };
            let model_info = draft_exec.info();
            let draft_kv_request_id = RequestId::new();
            let alloc_request = AllocationRequest {
                request_id: draft_kv_request_id.clone(),
                initial_tokens: prompt_u32s.len(),
                max_sequence_length: model_info.max_sequence_length,
                num_layers: model_info.num_layers,
                num_heads: model_info.num_kv_heads,
                head_dim: model_info.hidden_size / model_info.num_heads.max(1),
                device: self.config.backend.device.clone(),
                dtype: model_info.dtype,
                priority: Priority::Normal,
            };
            let draft_kv_lease = self
                .allocate_kv_lease(
                    request_id,
                    draft_kv_request_id.clone(),
                    &alloc_request,
                    prompt_u32s.len(),
                )
                .await?;
            let draft_kv_handle = draft_kv_lease.handle();
            let prompt_tensor = match self.tokens_to_tensor(&prompt_u32s) {
                Ok(tensor) => tensor,
                Err(e) => {
                    draft_kv_lease.release(self).await;
                    return Err(e);
                }
            };
            let pfx = PrefillInput::new(prompt_tensor).with_kv_cache(draft_kv_handle);
            let pfx_out = match draft_exec.prefill(&pfx).await {
                Ok(output) => output,
                Err(e) => {
                    draft_kv_lease.release(self).await;
                    return Err(e);
                }
            };
            let kv = pfx_out.kv_cache.clone();
            let (draft_kv_request_id, draft_kv_resource_blocks) =
                draft_kv_lease.into_committed_parts();
            {
                let mut sequences = self.sequences.write();
                if let Some(s) = sequences.get_mut(request_id) {
                    s.commit_draft_kv_allocation(
                        kv.clone(),
                        draft_kv_request_id,
                        draft_kv_resource_blocks,
                    );
                }
            }
            kv
        };

        // ── 2. Pull current state + run one spec step ────────────────────
        let (target_kv, last_token) = {
            let sequences = self.sequences.read();
            let seq = sequences
                .get(request_id)
                .ok_or_else(|| FerrumError::internal("Sequence not found"))?;
            let kv = seq
                .kv_cache_handle()
                .ok_or_else(|| FerrumError::internal("No target KV"))?;
            let last = seq
                .generated_tokens
                .last()
                .copied()
                .unwrap_or(TokenId::new(0));
            (kv, last)
        };

        let runner = crate::speculative::SpeculativeRunner {
            draft: draft_exec.as_ref(),
            target: self.model_executor.as_ref(),
            tensor_factory: self.tensor_factory.clone(),
            cfg,
        };

        // Snapshot the RNG out of the sequence so the async step can borrow
        // the mutable RNG; reinstall on the way back.
        let rng_seed = {
            let sequences = self.sequences.read();
            let seed = sequences
                .get(request_id)
                .and_then(|s| s.sampling_params.seed)
                .unwrap_or(42);
            // Mix in iteration count for non-deterministic draws across
            // speculative rounds that share the same seed.
            seed.wrapping_add(self.iteration_count.load(Ordering::Relaxed))
        };
        let mut rng = rand::rngs::StdRng::from_seed({
            let mut seed = [0u8; 32];
            seed[..8].copy_from_slice(&rng_seed.to_le_bytes());
            seed
        });

        // Capture entry seq_len so we can compute the correct rollback
        // length on partial rejection below. Both handles here are
        // `GenericKvCacheHandle` (constructed by `LlmExecutor::prefill` /
        // `decode`, NOT the engine `KvCacheManager`-allocated Paged/Default
        // handles used on the non-spec path), so `sequence_length` is
        // correctly updated each step via `with_sequence_length(new_seq)`.
        // Verified 2026-05-14 with FERRUM_DEBUG_SPEC_POS instrumentation:
        // entry_target_seq grows by N+1 per step, matching the actual
        // model writes (see make_kv_handle_with_seq at L1827-1828 for the
        // update site on the partial-reject path).
        let entry_target_seq = target_kv.block_table().sequence_length;
        let entry_draft_seq = draft_kv.block_table().sequence_length;

        let outcome = runner
            .step(last_token, draft_kv.clone(), target_kv, &mut rng)
            .await?;

        // KV-cache reconciliation post-step.
        //
        //   full accept: target wrote N+1 new positions, draft only N.
        //     Draft lags by one. Feed `draft_catchup_token`
        //     (= draft_tokens[N-1]) into draft so it catches up.
        //
        //   partial reject at k < N: target wrote N+1 positions including
        //     `N-k` that were conditioned on rejected drafts. Truncate
        //     BOTH to entry_seq + k + 1 (keep last_token + k accepted
        //     drafts), then feed the replacement token so both advance
        //     to entry_seq + k + 2 in lockstep.
        let (draft_kv_aligned, target_kv_aligned) = if let Some(catchup) =
            outcome.draft_catchup_token
        {
            let tensor = self.tokens_to_tensor(&[catchup.get()])?;
            let input = ferrum_interfaces::model_executor::DecodeInput::new(
                tensor,
                outcome.draft_kv.clone(),
            );
            let feed_out = draft_exec.decode(&input).await?;
            (feed_out.kv_cache.clone(), outcome.target_kv.clone())
        } else {
            // Partial reject path. k = rejected_at accepted drafts + 1
            // replacement → emitted.len() = k+1. Target wrote positions
            // [entry..entry+N], draft wrote [entry..entry+N-1] during the
            // runner step; only the first k writes are valid. Truncate
            // BOTH caches back to entry+k+1 (keep last_token + k accepted
            // drafts). Do NOT feed replacement here — the next iter's
            // runner will consume replacement as its new last_token and
            // write it at the correct position automatically, mirroring
            // how target self-corrects on the bonus token in full-accept.
            let k = outcome.rejected_at;
            let kept_target = entry_target_seq + k + 1;
            let kept_draft = entry_draft_seq + k + 1;

            draft_exec
                .truncate_kv(&outcome.draft_kv, kept_draft)
                .await?;
            self.model_executor
                .truncate_kv(&outcome.target_kv, kept_target)
                .await?;

            let truncated_draft = self.make_kv_handle_with_seq(&outcome.draft_kv, kept_draft);
            let truncated_target = self.make_kv_handle_with_seq(&outcome.target_kv, kept_target);
            (truncated_draft, truncated_target)
        };

        // ── 3. Install accepted tokens; check stop after each ───────────
        let mut last_emitted = last_token;
        for &tok in &outcome.tokens {
            {
                let mut sequences = self.sequences.write();
                if let Some(seq) = sequences.get_mut(request_id) {
                    seq.generated_tokens.push(tok);
                    *seq.token_frequencies.entry(tok).or_insert(0) += 1;
                    seq.tokens_this_iteration += 1;
                }
            }
            last_emitted = tok;
            self.total_decode_tokens.fetch_add(1, Ordering::Relaxed);
            counter!("ferrum.engine.decode_tokens_total").increment(1);

            let stop_reason = self.stop_reason_for_request(request_id);
            if self.should_stream_generated_token(stop_reason) {
                self.send_stream_update(request_id, tok).await;
            }
            if let Some(reason) = stop_reason {
                self.complete_request(request_id, reason).await?;
                return Ok(());
            }
        }

        // ── 4. Persist updated KV handles ───────────────────────────────
        {
            let mut sequences = self.sequences.write();
            if let Some(seq) = sequences.get_mut(request_id) {
                seq.commit_speculative_decode_physical_resources(
                    target_kv_aligned,
                    draft_kv_aligned,
                )?;
            }
        }
        let _ = last_emitted;
        let generated_count = {
            let sequences = self.sequences.read();
            sequences
                .get(request_id)
                .map(|s| s.generated_tokens.len())
                .unwrap_or(0)
        };
        self.scheduler
            .update_decode_progress(request_id, generated_count);

        Ok(())
    }
}
