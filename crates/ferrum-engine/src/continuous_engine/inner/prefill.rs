use super::*;

impl EngineInner {
    // ── prefill ────────────────────────────────────────────────────────

    pub(super) async fn run_prefill(&self, request_id: &RequestId) -> Result<()> {
        let prefill_prof = self.runtime_config.batch_decode_prof;
        let prefill_t0 = if prefill_prof {
            Some(std::time::Instant::now())
        } else {
            None
        };
        let res = self.run_prefill_inner(request_id).await;
        if let Some(t0) = prefill_t0 {
            static PREFILL_PROF_CALLS: std::sync::atomic::AtomicU64 =
                std::sync::atomic::AtomicU64::new(0);
            let n = PREFILL_PROF_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let elapsed = t0.elapsed().as_micros();
            eprintln!(
                "[prefill-prof] call#{} req={} elapsed={}us ok={}",
                n,
                request_id,
                elapsed,
                res.is_ok()
            );
        }
        res
    }

    async fn run_prefill_inner(&self, request_id: &RequestId) -> Result<()> {
        let (context_tokens, num_tokens, can_use_prefix_cache) = {
            let sequences = self.sequences.read();
            let seq = sequences
                .get(request_id)
                .ok_or_else(|| FerrumError::internal("Sequence not found"))?;
            (
                seq.prefill_context_tokens(),
                seq.prefill_context_len(),
                seq.generated_tokens.is_empty(),
            )
        };

        // ── Check prefix cache ──────────────────────────────────────────
        // Exact-match only: on hit, skip executor prefill entirely by cloning
        // the cached KV handle and sampling from the stored last-token logits.
        // Partial matches (stored prefix is a proper prefix of input) fall
        // through to full prefill — supporting them needs incremental prefill
        // on top of a cloned KV, not yet exposed by the executor contract.
        //
        // CUDA + CPU + Metal: prefix cache defaults OFF. The `clone_handle`
        // path in ferrum-kv flags blocks as COW but the write path doesn't
        // fork before mutating, so cache hits share decode-time mutations
        // back into the cached prefix — first request differs from
        // subsequent ones (reproduced 2026-05-19, see gaps memo). Opt in
        // via env `FERRUM_PREFIX_CACHE=1` once the CoW fix lands.
        // Prefix cache defaults OFF on every backend. The `clone_handle`
        // path in `crates/ferrum-kv/src/managers/paged.rs` is COW-by-flag
        // but the engine write path doesn't fork blocks on first write,
        // so a second request that hits the cache shares mutated KV from
        // the first request's decode and diverges deterministically
        // (request 1 ≠ request 2 == request 3). Reproduced 2026-05-19;
        // see `~/.claude/projects/*/memory/project_http_server_gaps_2026_05_19.md`.
        // Opt in via `FERRUM_PREFIX_CACHE=1` once the CoW fix lands.
        let recurrent_state_spec = self
            .model_executor
            .recurrent_state_spec(request_id, &context_tokens)?;
        let skip_prefix_cache =
            !self.runtime_config.prefix_cache_enabled || recurrent_state_spec.is_some();
        if !skip_prefix_cache && can_use_prefix_cache {
            let hit = self
                .prefix_cache
                .find_prefix(&context_tokens)
                .filter(|(prefix_id, _, _)| prefix_id.len() == context_tokens.len());
            if let Some((_prefix_id, cached_kv, cached_logits)) = hit {
                debug!(
                    "Prefix cache hit for {}: reusing {} cached tokens",
                    request_id, num_tokens,
                );

                let cloned_kv = cached_kv.clone_handle()?;

                let (first_token, model_cache_update) = {
                    let mut sequences = self.sequences.write();
                    let seq = sequences
                        .get_mut(request_id)
                        .ok_or_else(|| FerrumError::internal("Sequence not found"))?;
                    seq.reset_guided_processors()?;
                    let mut logits = cached_logits;
                    let token = seq.sample_with_processors_with_tokenizer(
                        &mut logits,
                        Some(self.tokenizer.as_ref()),
                    )?;
                    seq.generated_tokens.push(token);
                    let model_cache_update =
                        seq.commit_cached_prefill_physical_resources(cloned_kv, num_tokens);
                    (token, model_cache_update)
                };
                self.apply_model_cache_ref_update(request_id, model_cache_update);

                self.scheduler.mark_prefill_complete(request_id, num_tokens);
                self.prefix_cache_hits.fetch_add(1, Ordering::Relaxed);
                counter!("ferrum.engine.prefix_cache_hits").increment(1);

                debug!(
                    "Prefix cache prefill for {}: first generated: {}",
                    request_id,
                    first_token.get()
                );

                let stop_reason = self.stop_reason_for_request(request_id);
                if self.should_stream_generated_token(stop_reason) {
                    self.send_stream_update(request_id, first_token).await;
                }
                if let Some(reason) = stop_reason {
                    self.complete_request(request_id, reason).await?;
                }

                return Ok(());
            }
        } // skip_prefix_cache

        // ── Cache miss (or prefix cache skipped) — full prefill ─────────
        let mut recurrent_admission = self
            .prepare_recurrent_state(request_id, recurrent_state_spec)
            .await?;
        let initial_recurrent_state = recurrent_admission.handle();
        let model_info = self.model_executor.info();
        let alloc_request = AllocationRequest {
            request_id: request_id.clone(),
            initial_tokens: num_tokens,
            max_sequence_length: model_info.max_sequence_length,
            num_layers: model_info.num_layers,
            num_heads: model_info.num_kv_heads,
            head_dim: model_info.hidden_size / model_info.num_heads.max(1),
            device: self.config.backend.device.clone(),
            dtype: model_info.dtype,
            priority: Priority::Normal,
        };

        // Try allocation, preempting if necessary. The lease owns the backend
        // allocation until the sequence state accepts it below.
        let kv_lease = match self
            .allocate_kv_lease(request_id, request_id.clone(), &alloc_request, num_tokens)
            .await
        {
            Ok(lease) => lease,
            Err(_) => {
                // OOM — try to free blocks by preempting a victim
                if self.preempt_victim(request_id).await {
                    // Retry after preemption
                    match self
                        .allocate_kv_lease(
                            request_id,
                            request_id.clone(),
                            &alloc_request,
                            num_tokens,
                        )
                        .await
                    {
                        Ok(lease) => lease,
                        Err(e) => {
                            recurrent_admission.release_fresh(self).await;
                            return Err(e);
                        }
                    }
                } else {
                    recurrent_admission.release_fresh(self).await;
                    return Err(FerrumError::resource_exhausted(
                        "No blocks available and no request to preempt",
                    ));
                }
            }
        };
        let kv_handle = kv_lease.handle();
        let kv_resource_blocks = kv_lease.blocks();

        // Opt-in chunked prefill: `FERRUM_CHUNKED_PREFILL=<chunk_size>` splits
        // the prompt into sequential chunks and runs `prefill` per chunk.
        // Reduces peak activation memory for long prompts; also informs the
        // scheduler so its metrics reflect actual progress. True cross-
        // iteration interleaving with decode is a follow-up refactor.
        let chunk_size = self.runtime_config.chunked_prefill_size_for(num_tokens);
        let request_metadata = {
            let sequences = self.sequences.read();
            sequences
                .get(request_id)
                .map(|seq| seq.model_decode_metadata())
                .unwrap_or_default()
        };
        let prefill_output = if let Some(csz) = chunk_size {
            let mut current_kv = kv_handle;
            let mut current_recurrent_state = initial_recurrent_state.clone();
            let mut final_output: Option<ferrum_interfaces::model_executor::PrefillOutput> = None;
            let mut processed = 0usize;
            while processed < num_tokens {
                let end = (processed + csz).min(num_tokens);
                let chunk_ids: Vec<u32> = context_tokens[processed..end]
                    .iter()
                    .map(|t| t.get())
                    .collect();
                let chunk_tensor = match self.tokens_to_tensor(&chunk_ids) {
                    Ok(tensor) => tensor,
                    Err(e) => {
                        kv_lease.release(self).await;
                        recurrent_admission.release_fresh(self).await;
                        return Err(e);
                    }
                };
                let mut input = ferrum_interfaces::model_executor::PrefillInput::new(chunk_tensor)
                    .with_kv_cache(current_kv.clone())
                    .with_metadata(request_metadata.clone());
                if let Some(state) = current_recurrent_state.clone() {
                    input = input.with_recurrent_state(state);
                }
                let workspace_lease = self.acquire_backend_workspace_lease(
                    vec![request_id.clone()],
                    "engine_prefill_workspace",
                    "engine_prefill_workspace_release",
                );
                let out = match self.model_executor.prefill(&input).await {
                    Ok(out) => {
                        workspace_lease.release();
                        out
                    }
                    Err(e) => {
                        drop(workspace_lease);
                        kv_lease.release(self).await;
                        recurrent_admission.release_fresh(self).await;
                        return Err(e);
                    }
                };
                current_kv = out.kv_cache.clone();
                current_recurrent_state = out.recurrent_state.clone();

                self.scheduler.mark_prefill_chunk_processed(
                    request_id,
                    num_tokens,
                    end - processed,
                );

                processed = end;
                if processed >= num_tokens {
                    final_output = Some(out);
                }
            }
            final_output.expect("at least one chunk must run")
        } else {
            let input_tensor = {
                let token_u32s: Vec<u32> = context_tokens.iter().map(|t| t.get()).collect();
                match self.tokens_to_tensor(&token_u32s) {
                    Ok(tensor) => tensor,
                    Err(e) => {
                        kv_lease.release(self).await;
                        recurrent_admission.release_fresh(self).await;
                        return Err(e);
                    }
                }
            };
            let prefill_input = ferrum_interfaces::model_executor::PrefillInput::new(input_tensor)
                .with_kv_cache(kv_handle)
                .with_metadata(request_metadata);
            let prefill_input = if let Some(state) = initial_recurrent_state.clone() {
                prefill_input.with_recurrent_state(state)
            } else {
                prefill_input
            };
            let workspace_lease = self.acquire_backend_workspace_lease(
                vec![request_id.clone()],
                "engine_prefill_workspace",
                "engine_prefill_workspace_release",
            );
            match self.model_executor.prefill(&prefill_input).await {
                Ok(out) => {
                    workspace_lease.release();
                    out
                }
                Err(e) => {
                    drop(workspace_lease);
                    kv_lease.release(self).await;
                    recurrent_admission.release_fresh(self).await;
                    return Err(e);
                }
            }
        };

        let first_token_result = (|| {
            let last_logits = prefill_output.last_token_logits()?;
            let logits_vec = last_logits.to_vec_f32()?;

            // Store only prompt-only prefills. Replay prefills include already
            // generated output and would be low-value, request-specific entries.
            if can_use_prefix_cache {
                let _ = self.prefix_cache.store_prefix(
                    &context_tokens,
                    prefill_output.kv_cache.clone(),
                    logits_vec.clone(),
                );
            }

            let mut sequences = self.sequences.write();
            let seq = sequences
                .get_mut(request_id)
                .ok_or_else(|| FerrumError::internal("Sequence not found"))?;
            seq.reset_guided_processors()?;
            let mut logits = logits_vec;
            let token = seq.sample_with_processors_with_tokenizer(
                &mut logits,
                Some(self.tokenizer.as_ref()),
            )?;
            seq.generated_tokens.push(token);
            let recurrent_state = prefill_output
                .recurrent_state
                .clone()
                .or_else(|| recurrent_admission.handle());
            let model_cache_update = seq.commit_prefill_physical_resources(
                prefill_output.kv_cache.clone(),
                kv_resource_blocks,
                recurrent_state,
                recurrent_admission.fresh_slots(),
            );
            Ok::<(TokenId, ModelCacheRefUpdate), FerrumError>((token, model_cache_update))
        })();
        let (first_token, model_cache_update) = match first_token_result {
            Ok(value) => value,
            Err(e) => {
                kv_lease.release(self).await;
                recurrent_admission.release_fresh(self).await;
                return Err(e);
            }
        };
        self.apply_model_cache_ref_update(request_id, model_cache_update);
        let (_committed_request_id, committed_kv_resource_blocks) = kv_lease.into_committed_parts();
        debug_assert_eq!(committed_kv_resource_blocks, kv_resource_blocks);
        recurrent_admission.commit_fresh();

        self.scheduler.mark_prefill_complete(request_id, num_tokens);
        self.total_prefill_tokens
            .fetch_add(num_tokens as u64, Ordering::Relaxed);
        counter!("ferrum.engine.prefill_tokens_total").increment(num_tokens as u64);
        counter!("ferrum.engine.prefills_total").increment(1);

        debug!(
            "Prefill complete for {}: {} prompt tokens, first generated: {}",
            request_id,
            num_tokens,
            first_token.get()
        );

        let stop_reason = self.stop_reason_for_request(request_id);
        if self.should_stream_generated_token(stop_reason) {
            self.send_stream_update(request_id, first_token).await;
        }
        if let Some(reason) = stop_reason {
            self.complete_request(request_id, reason).await?;
        }

        Ok(())
    }

    // ── batch prefill ─────────────────────────────────────────────────

    pub(super) async fn run_plan_runtime_prefill(
        &self,
        scheduled: &ferrum_interfaces::scheduler::ScheduledRequest,
    ) -> Result<()> {
        use ferrum_interfaces::model_executor::{
            ExecutorPrefillOutcome, PrefillChunk, PrefillInput,
        };

        let request_id = &scheduled.request.id;

        let Some((input_tokens, maximum_sequence_tokens, metadata)) =
            self.sequences.read().get(request_id).map(|seq| {
                (
                    seq.prefill_context_tokens(),
                    seq.model_maximum_sequence_tokens(),
                    seq.model_decode_metadata(),
                )
            })
        else {
            return Ok(());
        };
        let chunk = PrefillChunk::new(
            scheduled.tokens_processed,
            scheduled.tokens_to_process.ok_or_else(|| {
                FerrumError::scheduler(format!(
                    "PlanRuntime prefill for {request_id} has no scheduled token budget"
                ))
            })?,
            input_tokens.len(),
        )?;
        let token_ids = input_tokens
            .iter()
            .map(|token| token.get())
            .collect::<Vec<_>>();
        let input = PrefillInput::new(self.tokens_to_tensor(&token_ids)?)
            .with_request_context(request_id.clone(), maximum_sequence_tokens)
            .with_chunk(chunk)
            .with_metadata(metadata);

        let workspace_lease = self.acquire_backend_workspace_lease(
            vec![request_id.clone()],
            "plan_runtime_prefill_workspace",
            "plan_runtime_prefill_workspace_release",
        );
        let output = match self.model_executor.prefill_with_capacity(&input).await? {
            ExecutorPrefillOutcome::Completed(output) => {
                workspace_lease.release();
                output
            }
            ExecutorPrefillOutcome::Deferred(deferral) => {
                drop(workspace_lease);
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
                if !self
                    .scheduler
                    .defer_prefill_for_execution_capacity(request_id, scheduler_deferral)?
                {
                    return Err(FerrumError::scheduler(format!(
                        "PlanRuntime prefill deferral lost scheduler entry {request_id}"
                    )));
                }
                self.write_scheduler_trace_event(serde_json::json!({
                    "event": "scheduler_prefill_execution_capacity_defer",
                    "request_id": request_id,
                    "tokens_processed": chunk.tokens_processed(),
                    "tokens_to_process": chunk.tokens_to_process(),
                    "stage": deferral.stage(),
                    "observed": observed,
                    "wait_condition": deferral.wait_condition(),
                    "scheduler": self.scheduler.trace_snapshot(),
                }));
                return Ok(());
            }
        };
        let cache_id = output.kv_cache.cache_id();
        if !chunk.is_final() {
            let model_cache_update = {
                let mut sequences = self.sequences.write();
                let Some(seq) = sequences.get_mut(request_id) else {
                    self.model_executor.cancel_prefill_admission(request_id);
                    return Ok(());
                };
                seq.commit_plan_runtime_prefill_chunk_resources(output.kv_cache, chunk.end(), false)
            };
            self.apply_model_cache_ref_update(request_id, model_cache_update);
            if self.scheduler.mark_prefill_chunk_processed(
                request_id,
                input_tokens.len(),
                chunk.tokens_to_process(),
            ) {
                return Err(FerrumError::scheduler(format!(
                    "non-final PlanRuntime prefill chunk promoted {request_id} to decode"
                )));
            }
            self.total_prefill_tokens
                .fetch_add(chunk.tokens_to_process() as u64, Ordering::Relaxed);
            counter!("ferrum.engine.prefill_tokens_total")
                .increment(chunk.tokens_to_process() as u64);
            return Ok(());
        }

        let commit_result = (|| {
            let last_logits = output.last_token_logits()?;
            let mut logits = last_logits.to_vec_f32()?;
            let mut sequences = self.sequences.write();
            let Some(seq) = sequences.get_mut(request_id) else {
                return Ok(None);
            };
            seq.reset_guided_processors()?;
            let token = seq.sample_with_processors_with_tokenizer(
                &mut logits,
                Some(self.tokenizer.as_ref()),
            )?;
            seq.generated_tokens.push(token);
            let update = seq.commit_plan_runtime_prefill_chunk_resources(
                output.kv_cache.clone(),
                chunk.end(),
                true,
            );
            Ok::<Option<(TokenId, ModelCacheRefUpdate)>, FerrumError>(Some((token, update)))
        })();
        let Some((first_token, model_cache_update)) = (match commit_result {
            Ok(value) => value,
            Err(error) => {
                self.model_executor.release_cache(&cache_id);
                return Err(error);
            }
        }) else {
            self.model_executor.release_cache(&cache_id);
            return Ok(());
        };

        self.apply_model_cache_ref_update(request_id, model_cache_update);
        if !self.scheduler.mark_prefill_chunk_processed(
            request_id,
            input_tokens.len(),
            chunk.tokens_to_process(),
        ) {
            self.model_executor.release_cache(&cache_id);
            return Err(FerrumError::scheduler(format!(
                "final PlanRuntime prefill chunk did not promote {request_id} to decode"
            )));
        }
        self.total_prefill_tokens
            .fetch_add(chunk.tokens_to_process() as u64, Ordering::Relaxed);
        counter!("ferrum.engine.prefill_tokens_total").increment(chunk.tokens_to_process() as u64);
        counter!("ferrum.engine.prefills_total").increment(1);

        let stop_reason = self.stop_reason_for_request(request_id);
        if self.should_stream_generated_token(stop_reason) {
            self.send_stream_update(request_id, first_token).await;
        }
        if let Some(reason) = stop_reason {
            self.complete_request(request_id, reason).await?;
        }
        Ok(())
    }

    /// Run prefill for multiple requests as ONE batched forward pass.
    ///
    /// Replaces the serial `for rid in prefill_ids { run_prefill }` loop
    /// in `process_batch`. Per-request setup (prefix cache check + KV
    /// allocation + tokenization) still happens individually; the GPU
    /// call coalesces into one `model_executor.batch_prefill` invocation.
    ///
    /// Falls back to serial `run_prefill` per request when chunked prefill
    /// is enabled (`FERRUM_CHUNKED_PREFILL=N`) — those paths have
    /// multi-call semantics that the batched path doesn't model yet.
    /// Phase 2 will lift this restriction.
    pub(super) async fn run_batch_prefill(&self, request_ids: &[RequestId]) -> Result<()> {
        use ferrum_interfaces::model_executor::PrefillInput;

        if request_ids.is_empty() {
            return Ok(());
        }

        // Chunked-prefill opt-in path: fall back to serial.
        if self.runtime_config.chunked_prefill_present {
            for rid in request_ids {
                if let Err(e) = self.run_prefill(rid).await {
                    warn!("Prefill failed for {}: {}", rid, e);
                    if is_resource_exhausted_error(&e) {
                        continue;
                    }
                    self.complete_request(rid, FinishReason::Error).await?;
                }
            }
            return Ok(());
        }

        // ── Phase 1a: per-request setup (prefix cache → tokens → kv alloc) ──
        // After this loop, `to_prefill` holds only requests that need a real
        // model call. Prefix cache hits + immediate stops are handled inline.
        let mut to_prefill: Vec<PendingBatchPrefill> = Vec::new();

        let model_info = self.model_executor.info();
        // Prefix cache defaults OFF on every backend. The `clone_handle`
        // path in `crates/ferrum-kv/src/managers/paged.rs` is COW-by-flag
        // but the engine write path doesn't fork blocks on first write,
        // so a second request that hits the cache shares mutated KV from
        // the first request's decode and diverges deterministically
        // (request 1 ≠ request 2 == request 3). Reproduced 2026-05-19;
        // see `~/.claude/projects/*/memory/project_http_server_gaps_2026_05_19.md`.
        // Opt in via `FERRUM_PREFIX_CACHE=1` once the CoW fix lands.
        let skip_prefix_cache = !self.runtime_config.prefix_cache_enabled;

        for rid in request_ids {
            let (input_tokens, num_tokens, metadata, can_use_prefix_cache) = {
                let sequences = self.sequences.read();
                let Some(seq) = sequences.get(rid) else {
                    continue; // request gone (cancelled mid-batch)
                };
                (
                    seq.prefill_context_tokens(),
                    seq.prefill_context_len(),
                    seq.model_decode_metadata(),
                    seq.generated_tokens.is_empty(),
                )
            };
            let recurrent_state_spec = self
                .model_executor
                .recurrent_state_spec(rid, &input_tokens)?;

            // Prefix cache hit short-circuit (mirrors run_prefill_inner).
            if !skip_prefix_cache && can_use_prefix_cache && recurrent_state_spec.is_none() {
                let hit = self
                    .prefix_cache
                    .find_prefix(&input_tokens)
                    .filter(|(prefix_id, _, _)| prefix_id.len() == input_tokens.len());
                if let Some((_, cached_kv, cached_logits)) = hit {
                    let cloned_kv = cached_kv.clone_handle()?;
                    let (first_token, model_cache_update) = {
                        let mut sequences = self.sequences.write();
                        let Some(seq) = sequences.get_mut(rid) else {
                            continue;
                        };
                        seq.reset_guided_processors()?;
                        let mut logits = cached_logits;
                        let token = seq.sample_with_processors_with_tokenizer(
                            &mut logits,
                            Some(self.tokenizer.as_ref()),
                        )?;
                        seq.generated_tokens.push(token);
                        let model_cache_update =
                            seq.commit_cached_prefill_physical_resources(cloned_kv, num_tokens);
                        (token, model_cache_update)
                    };
                    self.apply_model_cache_ref_update(rid, model_cache_update);
                    self.scheduler.mark_prefill_complete(rid, num_tokens);
                    self.prefix_cache_hits.fetch_add(1, Ordering::Relaxed);
                    counter!("ferrum.engine.prefix_cache_hits").increment(1);
                    let stop_reason = self.stop_reason_for_request(rid);
                    if self.should_stream_generated_token(stop_reason) {
                        self.send_stream_update(rid, first_token).await;
                    }
                    if let Some(reason) = stop_reason {
                        self.complete_request(rid, reason).await?;
                    }
                    continue;
                }
            }

            // Cache miss — allocate KV pages.
            let alloc_request = AllocationRequest {
                request_id: rid.clone(),
                initial_tokens: num_tokens,
                max_sequence_length: model_info.max_sequence_length,
                num_layers: model_info.num_layers,
                num_heads: model_info.num_kv_heads,
                head_dim: model_info.hidden_size / model_info.num_heads.max(1),
                device: self.config.backend.device.clone(),
                dtype: model_info.dtype,
                priority: Priority::Normal,
            };
            let kv_lease = match self
                .allocate_kv_lease(rid, rid.clone(), &alloc_request, num_tokens)
                .await
            {
                Ok(lease) => lease,
                Err(_) => {
                    if self.preempt_victim(rid).await {
                        match self
                            .allocate_kv_lease(rid, rid.clone(), &alloc_request, num_tokens)
                            .await
                        {
                            Ok(lease) => lease,
                            Err(e) => {
                                warn!("Prefill alloc deferred for {} after preempt: {}", rid, e);
                                continue;
                            }
                        }
                    } else {
                        warn!("Prefill alloc deferred for {}: no preempt victim", rid);
                        continue;
                    }
                }
            };
            let recurrent_state = match self
                .prepare_recurrent_state(rid, recurrent_state_spec)
                .await
            {
                Ok(state) => state,
                Err(e) => {
                    warn!("Recurrent-state alloc failed for {}: {}", rid, e);
                    kv_lease.release(self).await;
                    if is_resource_exhausted_error(&e) {
                        continue;
                    }
                    self.complete_request(rid, FinishReason::Error).await?;
                    continue;
                }
            };
            to_prefill.push(PendingBatchPrefill::new(
                rid.clone(),
                input_tokens,
                kv_lease,
                recurrent_state,
                metadata,
                can_use_prefix_cache,
            ));
        }

        if to_prefill.is_empty() {
            return Ok(());
        }

        let workspace_request_ids: Vec<RequestId> = to_prefill
            .iter()
            .map(|pending| pending.request_id.clone())
            .collect();
        // ── Phase 1b: ONE batched model_executor.batch_prefill call ──
        let mut inputs: Vec<PrefillInput> = Vec::with_capacity(to_prefill.len());
        for pending in &to_prefill {
            let token_u32s: Vec<u32> = pending.input_tokens.iter().map(|t| t.get()).collect();
            let tensor = match self.tokens_to_tensor(&token_u32s) {
                Ok(tensor) => tensor,
                Err(e) => {
                    for pending in &mut to_prefill {
                        pending.release_resources(self).await;
                    }
                    return Err(e);
                }
            };
            let kv = pending.kv_handle()?;
            let input = PrefillInput::new(tensor)
                .with_kv_cache(kv)
                .with_metadata(pending.metadata.clone());
            inputs.push(if let Some(state) = pending.recurrent_state.handle() {
                input.with_recurrent_state(state)
            } else {
                input
            });
        }

        let workspace_lease = self.acquire_backend_workspace_lease(
            workspace_request_ids,
            "engine_batch_prefill_workspace",
            "engine_batch_prefill_workspace_release",
        );
        let outputs = match self.model_executor.batch_prefill(&inputs).await {
            Ok(outputs) => {
                workspace_lease.release();
                outputs
            }
            Err(e) => {
                drop(workspace_lease);
                for pending in &mut to_prefill {
                    pending.release_resources(self).await;
                }
                return Err(e);
            }
        };
        if outputs.len() != to_prefill.len() {
            for pending in &mut to_prefill {
                pending.release_resources(self).await;
            }
            return Err(FerrumError::internal(format!(
                "batch_prefill returned {} outputs for {} inputs",
                outputs.len(),
                to_prefill.len(),
            )));
        }

        // ── Phase 1c: per-item post-process (sample, update seq, stream, stop) ──
        for (pending, prefill_output) in to_prefill.iter_mut().zip(outputs.iter()) {
            let rid = pending.request_id.clone();
            let kv_resource_blocks = pending.kv_resource_blocks()?;
            let first_token_result = (|| {
                let last_logits = prefill_output.last_token_logits()?;
                let logits_vec = last_logits.to_vec_f32()?;
                if pending.can_use_prefix_cache {
                    let _ = self.prefix_cache.store_prefix(
                        &pending.input_tokens,
                        prefill_output.kv_cache.clone(),
                        logits_vec.clone(),
                    );
                }
                let mut sequences = self.sequences.write();
                let Some(seq) = sequences.get_mut(&rid) else {
                    return Ok(None);
                };
                seq.reset_guided_processors()?;
                let mut logits = logits_vec;
                let token = seq.sample_with_processors_with_tokenizer(
                    &mut logits,
                    Some(self.tokenizer.as_ref()),
                )?;
                seq.generated_tokens.push(token);
                let recurrent_state = prefill_output
                    .recurrent_state
                    .clone()
                    .or_else(|| pending.recurrent_state.handle());
                let model_cache_update = seq.commit_prefill_physical_resources(
                    prefill_output.kv_cache.clone(),
                    kv_resource_blocks,
                    recurrent_state,
                    pending.recurrent_state.fresh_slots(),
                );
                Ok::<Option<(TokenId, ModelCacheRefUpdate)>, FerrumError>(Some((
                    token,
                    model_cache_update,
                )))
            })();
            let (first_token, model_cache_update) = match first_token_result {
                Ok(Some(value)) => value,
                Ok(None) => {
                    pending.release_resources(self).await;
                    continue;
                }
                Err(e) => {
                    warn!("Batch prefill post-process failed for {}: {}", rid, e);
                    pending.release_resources(self).await;
                    self.complete_request(&rid, FinishReason::Error).await?;
                    continue;
                }
            };
            self.apply_model_cache_ref_update(&rid, model_cache_update);
            let committed_kv_resource_blocks = pending.commit_kv()?;
            debug_assert_eq!(committed_kv_resource_blocks, kv_resource_blocks);
            pending.recurrent_state.commit_fresh();
            let num_tokens = pending.input_tokens.len();
            self.scheduler.mark_prefill_complete(&rid, num_tokens);
            self.total_prefill_tokens
                .fetch_add(num_tokens as u64, Ordering::Relaxed);
            counter!("ferrum.engine.prefill_tokens_total").increment(num_tokens as u64);
            counter!("ferrum.engine.prefills_total").increment(1);
            let stop_reason = self.stop_reason_for_request(&rid);
            if self.should_stream_generated_token(stop_reason) {
                self.send_stream_update(&rid, first_token).await;
            }
            if let Some(reason) = stop_reason {
                self.complete_request(&rid, reason).await?;
            }
        }
        Ok(())
    }
}
