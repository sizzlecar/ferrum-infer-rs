use super::*;

impl EngineInner {
    // ── batch decode ──────────────────────────────────────────────────

    pub(super) async fn run_batch_decode_adaptive(&self, request_ids: &[RequestId]) -> Result<()> {
        let mut stack = vec![request_ids.to_vec()];
        while let Some(chunk) = stack.pop() {
            if chunk.is_empty() {
                continue;
            }
            match self.run_batch_decode(&chunk).await {
                Ok(()) => {}
                Err(e) if is_resource_exhausted_error(&e) && chunk.len() > 1 => {
                    let mid = chunk.len() / 2;
                    stack.push(chunk[mid..].to_vec());
                    stack.push(chunk[..mid].to_vec());
                }
                Err(e) if is_resource_exhausted_error(&e) => {
                    let exclude: std::collections::HashSet<RequestId> =
                        chunk.iter().cloned().collect();
                    if self.preempt_victim_excluding(&exclude).await {
                        stack.push(chunk);
                    } else {
                        return Err(e);
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

        let rids: Vec<RequestId> = request_ids.to_vec();

        // Build the unified batch from sequence state.
        let mut batch = UnifiedBatch::new();
        {
            let sequences = self.sequences.read();
            for rid in &rids {
                let seq = sequences
                    .get(rid)
                    .ok_or_else(|| FerrumError::internal("Sequence not found"))?;
                let kv_cache = seq
                    .kv_cache
                    .as_ref()
                    .ok_or_else(|| FerrumError::internal("No KV cache"))?
                    .clone();
                let last_token = seq
                    .generated_tokens
                    .last()
                    .copied()
                    .unwrap_or(TokenId::new(0));
                // pos_offset = position of the NEW token. Compute from
                // engine bookkeeping (see process_batch_unified for why
                // `kv_cache.block_table().sequence_length` is not reliable
                // — Paged/Default handles never increment it).
                let pos_offset = seq.input_tokens.len() + seq.generated_tokens.len() - 1;
                // Use the model-side cache_id (set in `run_prefill_inner`
                // from `prefill_output.kv_cache.cache_id()`), NOT the
                // engine's request_id. The model's `kv_caches` is keyed
                // by the executor-generated id (e.g. "llm-cache-N"); using
                // the request_id (UUID) makes `ensure_kv` allocate a
                // fresh cache + 128 paged blocks for every decode iter,
                // exhausting the pool within ~60 prompts.
                let seq_id = seq
                    .model_cache_id
                    .clone()
                    .unwrap_or_else(|| rid.to_string());
                batch.items.push(UnifiedBatchItem {
                    seq_id,
                    q_tokens: vec![last_token.get()],
                    kv_cache,
                    recurrent_state: seq.recurrent_state.clone(),
                    pos_offset,
                    is_final_chunk: true,
                    metadata: seq.model_decode_metadata(),
                    logits_policy: seq.model_decode_logits_policy(),
                });
            }
        }

        let kv_requests = kv_slot_requests_for_unified_batch(&batch);
        self.model_executor.reserve_kv_slots(&kv_requests)?;

        let prof = self.runtime_config.rbd_prof;
        let t_decode = if prof { Some(Instant::now()) } else { None };
        let results = self.model_executor.unified_decode(&batch).await?;
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
            let next_token = {
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
                seq.tokens_this_iteration += 1;
                // pos_offset is sourced from SequenceState bookkeeping
                // (see process_batch_unified). The engine-side KV handle's
                // sequence_length is not used for position tracking
                // anymore — production handles (Paged/Default) don't
                // update it across iterations.
                token
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
        if self.draft_executor.is_some() && self.spec_config.is_some() {
            return self.run_decode_step_speculative(request_id).await;
        }

        let decode_input = {
            let sequences = self.sequences.read();
            let seq = sequences
                .get(request_id)
                .ok_or_else(|| FerrumError::internal("Sequence not found"))?;
            let kv_cache = seq
                .kv_cache
                .as_ref()
                .ok_or_else(|| FerrumError::internal("No KV cache"))?
                .clone();
            let recurrent_state = seq.recurrent_state.clone();
            let last_token = seq
                .generated_tokens
                .last()
                .copied()
                .unwrap_or(TokenId::new(0));
            let tensor = self.tokens_to_tensor(&[last_token.get()])?;
            let input = ferrum_interfaces::model_executor::DecodeInput::new(tensor, kv_cache)
                .with_metadata(seq.model_decode_metadata());
            if let Some(state) = recurrent_state {
                input.with_recurrent_state(state)
            } else {
                input
            }
        };

        let input_recurrent_state = decode_input.recurrent_state.clone();
        let decode_output = self.model_executor.decode(&decode_input).await?;
        let logits_vec = decode_output.logits.to_vec_f32()?;

        let next_token = {
            let mut sequences = self.sequences.write();
            let seq = sequences
                .get_mut(request_id)
                .ok_or_else(|| FerrumError::internal("Sequence not found"))?;
            let mut logits = logits_vec;
            let token = seq.sample_with_processors_with_tokenizer(
                &mut logits,
                Some(self.tokenizer.as_ref()),
            )?;
            seq.generated_tokens.push(token);
            seq.kv_cache = Some(decode_output.kv_cache.clone());
            seq.recurrent_state = decode_output
                .recurrent_state
                .clone()
                .or(input_recurrent_state);
            seq.tokens_this_iteration += 1;
            token
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
                .and_then(|s| s.draft_kv_cache.clone())
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
            let alloc_request = AllocationRequest {
                request_id: request_id.clone(),
                initial_tokens: prompt_u32s.len(),
                max_sequence_length: model_info.max_sequence_length,
                num_layers: model_info.num_layers,
                num_heads: model_info.num_kv_heads,
                head_dim: model_info.hidden_size / model_info.num_heads.max(1),
                device: self.config.backend.device.clone(),
                dtype: model_info.dtype,
                priority: Priority::Normal,
            };
            let draft_kv_handle = self.kv_cache.allocate(&alloc_request).await?;
            let prompt_tensor = self.tokens_to_tensor(&prompt_u32s)?;
            let pfx = PrefillInput::new(prompt_tensor).with_kv_cache(draft_kv_handle);
            let pfx_out = draft_exec.prefill(&pfx).await?;
            let kv = pfx_out.kv_cache.clone();
            {
                let mut sequences = self.sequences.write();
                if let Some(s) = sequences.get_mut(request_id) {
                    s.draft_kv_cache = Some(kv.clone());
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
                .kv_cache
                .as_ref()
                .ok_or_else(|| FerrumError::internal("No target KV"))?
                .clone();
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
                seq.kv_cache = Some(target_kv_aligned);
                seq.draft_kv_cache = Some(draft_kv_aligned);
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
