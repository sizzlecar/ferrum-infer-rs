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

                let first_token = {
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
                    seq.model_cache_id = Some(cloned_kv.cache_id());
                    seq.kv_cache = Some(cloned_kv);
                    seq.prefill_complete = true;
                    seq.phase = RequestPhase::Decoding;
                    token
                };

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
        let initial_recurrent_state = self
            .ensure_recurrent_state(request_id, recurrent_state_spec)
            .await?;
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

        // Try allocation, preempting if necessary
        let kv_handle = match self.kv_cache.allocate(&alloc_request).await {
            Ok(h) => h,
            Err(_) => {
                // OOM — try to free blocks by preempting a victim
                if self.preempt_victim(request_id).await {
                    // Retry after preemption
                    match self.kv_cache.allocate(&alloc_request).await {
                        Ok(h) => h,
                        Err(e) => {
                            self.release_recurrent_state(request_id).await;
                            return Err(e);
                        }
                    }
                } else {
                    self.release_recurrent_state(request_id).await;
                    return Err(FerrumError::resource_exhausted(
                        "No blocks available and no request to preempt",
                    ));
                }
            }
        };

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
                let chunk_tensor = self.tokens_to_tensor(&chunk_ids)?;
                let mut input = ferrum_interfaces::model_executor::PrefillInput::new(chunk_tensor)
                    .with_kv_cache(current_kv.clone())
                    .with_metadata(request_metadata.clone());
                if let Some(state) = current_recurrent_state.clone() {
                    input = input.with_recurrent_state(state);
                }
                let out = match self.model_executor.prefill(&input).await {
                    Ok(out) => out,
                    Err(e) => {
                        let _ = self.kv_cache.deallocate(request_id.clone()).await;
                        self.release_recurrent_state(request_id).await;
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
                self.tokens_to_tensor(&token_u32s)?
            };
            let prefill_input = ferrum_interfaces::model_executor::PrefillInput::new(input_tensor)
                .with_kv_cache(kv_handle)
                .with_metadata(request_metadata);
            let prefill_input = if let Some(state) = initial_recurrent_state.clone() {
                prefill_input.with_recurrent_state(state)
            } else {
                prefill_input
            };
            match self.model_executor.prefill(&prefill_input).await {
                Ok(out) => out,
                Err(e) => {
                    let _ = self.kv_cache.deallocate(request_id.clone()).await;
                    self.release_recurrent_state(request_id).await;
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
            seq.model_cache_id = Some(prefill_output.kv_cache.cache_id());
            seq.kv_cache = Some(prefill_output.kv_cache.clone());
            seq.recurrent_state = prefill_output
                .recurrent_state
                .clone()
                .or_else(|| initial_recurrent_state.clone());
            seq.prefill_complete = true;
            seq.phase = RequestPhase::Decoding;
            Ok::<TokenId, FerrumError>(token)
        })();
        let first_token = match first_token_result {
            Ok(token) => token,
            Err(e) => {
                let _ = self.kv_cache.deallocate(request_id.clone()).await;
                self.release_recurrent_state(request_id).await;
                return Err(e);
            }
        };

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
        let mut to_prefill: Vec<(
            RequestId,
            Vec<TokenId>,
            Arc<dyn ferrum_interfaces::KvCacheHandle>,
            Option<Arc<dyn ferrum_interfaces::RecurrentStateHandle>>,
            std::collections::HashMap<String, serde_json::Value>,
            bool,
        )> = Vec::new();

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
                    let first_token = {
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
                        seq.model_cache_id = Some(cloned_kv.cache_id());
                        seq.kv_cache = Some(cloned_kv);
                        seq.prefill_complete = true;
                        seq.phase = RequestPhase::Decoding;
                        token
                    };
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
            let kv_handle = match self.kv_cache.allocate(&alloc_request).await {
                Ok(h) => h,
                Err(_) => {
                    if self.preempt_victim(rid).await {
                        match self.kv_cache.allocate(&alloc_request).await {
                            Ok(h) => h,
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
            let recurrent_state = match self.ensure_recurrent_state(rid, recurrent_state_spec).await
            {
                Ok(state) => state,
                Err(e) => {
                    warn!("Recurrent-state alloc failed for {}: {}", rid, e);
                    let _ = self.kv_cache.deallocate(rid.clone()).await;
                    if is_resource_exhausted_error(&e) {
                        continue;
                    }
                    self.complete_request(rid, FinishReason::Error).await?;
                    continue;
                }
            };
            to_prefill.push((
                rid.clone(),
                input_tokens,
                kv_handle,
                recurrent_state,
                metadata,
                can_use_prefix_cache,
            ));
        }

        if to_prefill.is_empty() {
            return Ok(());
        }

        // ── Phase 1b: ONE batched model_executor.batch_prefill call ──
        let inputs: Vec<PrefillInput> = to_prefill
            .iter()
            .map(|(_, tokens, kv, recurrent_state, metadata, _)| {
                let token_u32s: Vec<u32> = tokens.iter().map(|t| t.get()).collect();
                let tensor = self.tokens_to_tensor(&token_u32s)?;
                let input = PrefillInput::new(tensor)
                    .with_kv_cache(kv.clone())
                    .with_metadata(metadata.clone());
                Ok(if let Some(state) = recurrent_state.clone() {
                    input.with_recurrent_state(state)
                } else {
                    input
                })
            })
            .collect::<Result<Vec<_>>>()?;

        let outputs = match self.model_executor.batch_prefill(&inputs).await {
            Ok(outputs) => outputs,
            Err(e) => {
                for (rid, _, _, _, _, _) in &to_prefill {
                    let _ = self.kv_cache.deallocate(rid.clone()).await;
                    self.release_recurrent_state(rid).await;
                }
                return Err(e);
            }
        };
        if outputs.len() != to_prefill.len() {
            for (rid, _, _, _, _, _) in &to_prefill {
                let _ = self.kv_cache.deallocate(rid.clone()).await;
                self.release_recurrent_state(rid).await;
            }
            return Err(FerrumError::internal(format!(
                "batch_prefill returned {} outputs for {} inputs",
                outputs.len(),
                to_prefill.len(),
            )));
        }

        // ── Phase 1c: per-item post-process (sample, update seq, stream, stop) ──
        for ((rid, input_tokens, _, recurrent_state, _, can_use_prefix_cache), prefill_output) in
            to_prefill.iter().zip(outputs.iter())
        {
            let first_token_result = (|| {
                let last_logits = prefill_output.last_token_logits()?;
                let logits_vec = last_logits.to_vec_f32()?;
                if *can_use_prefix_cache {
                    let _ = self.prefix_cache.store_prefix(
                        input_tokens,
                        prefill_output.kv_cache.clone(),
                        logits_vec.clone(),
                    );
                }
                let mut sequences = self.sequences.write();
                let Some(seq) = sequences.get_mut(rid) else {
                    return Ok(None);
                };
                seq.reset_guided_processors()?;
                let mut logits = logits_vec;
                let token = seq.sample_with_processors_with_tokenizer(
                    &mut logits,
                    Some(self.tokenizer.as_ref()),
                )?;
                seq.generated_tokens.push(token);
                seq.model_cache_id = Some(prefill_output.kv_cache.cache_id());
                seq.kv_cache = Some(prefill_output.kv_cache.clone());
                seq.recurrent_state = prefill_output
                    .recurrent_state
                    .clone()
                    .or_else(|| recurrent_state.clone());
                seq.prefill_complete = true;
                seq.phase = RequestPhase::Decoding;
                Ok::<Option<TokenId>, FerrumError>(Some(token))
            })();
            let first_token = match first_token_result {
                Ok(Some(token)) => token,
                Ok(None) => continue,
                Err(e) => {
                    warn!("Batch prefill post-process failed for {}: {}", rid, e);
                    let _ = self.kv_cache.deallocate(rid.clone()).await;
                    self.release_recurrent_state(rid).await;
                    self.complete_request(rid, FinishReason::Error).await?;
                    continue;
                }
            };
            let num_tokens = input_tokens.len();
            self.scheduler.mark_prefill_complete(rid, num_tokens);
            self.total_prefill_tokens
                .fetch_add(num_tokens as u64, Ordering::Relaxed);
            counter!("ferrum.engine.prefill_tokens_total").increment(num_tokens as u64);
            counter!("ferrum.engine.prefills_total").increment(1);
            let stop_reason = self.stop_reason_for_request(rid);
            if self.should_stream_generated_token(stop_reason) {
                self.send_stream_update(rid, first_token).await;
            }
            if let Some(reason) = stop_reason {
                self.complete_request(rid, reason).await?;
            }
        }
        Ok(())
    }
}
