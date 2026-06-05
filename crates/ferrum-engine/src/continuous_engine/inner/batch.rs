use super::*;

impl EngineInner {
    // ── batch processing ───────────────────────────────────────────────

    pub(super) async fn process_batch(&self, batch: &ferrum_interfaces::BatchPlan) -> Result<()> {
        // Single-shot unified path: prefill + decode items go through ONE
        // `model_executor.unified_decode` call. Phase-2/3 redesign goal —
        // prefill chunks and decode tokens are co-batched at the kernel
        // level, eliminating the cohort gap (decode_queue=0 ↔ 32
        // alternation that consumed half the bench wall on apples).
        //
        // For chunked-prefill (FERRUM_CHUNKED_PREFILL=N) or speculative
        // decoding, fall back to the legacy split path. Phase 3 will
        // extend chunked-prefill into the unified mode.
        let chunked_or_spec =
            self.runtime_config.chunked_prefill_present || self.spec_config.is_some();
        if chunked_or_spec {
            return self.process_batch_legacy_split(batch).await;
        }
        let info = self.model_executor.info();
        let is_qwen3_moe = info
            .metadata
            .get("ferrum_arch")
            .and_then(|value| value.as_str())
            == Some("qwen3moe");
        let backend_lacks_native_unified = matches!(info.device, ferrum_types::Device::CPU) || {
            #[cfg(any(target_os = "macos", target_os = "ios"))]
            {
                matches!(info.device, ferrum_types::Device::Metal)
            }
            #[cfg(not(any(target_os = "macos", target_os = "ios")))]
            {
                false
            }
        };
        if is_qwen3_moe && backend_lacks_native_unified {
            return self.process_batch_legacy_split(batch).await;
        }
        self.process_batch_unified(batch).await
    }

    /// Legacy split path: separate prefill (batched via run_batch_prefill)
    /// then decode (batched via run_batch_decode). Used for chunked-prefill
    /// and speculative-decoding flows that the unified path doesn't model yet.
    async fn process_batch_legacy_split(&self, batch: &ferrum_interfaces::BatchPlan) -> Result<()> {
        let mut prefill_ids = Vec::new();
        let mut decode_ids = Vec::new();
        {
            let mut sequences = self.sequences.write();
            for scheduled_req in &batch.requests {
                let rid = &scheduled_req.request.id;
                let seq = sequences.entry(rid.clone()).or_insert_with(|| {
                    let input_tokens = self
                        .tokenizer
                        .encode(&scheduled_req.request.prompt, true)
                        .unwrap_or_else(|_| vec![TokenId::new(0)]);
                    SequenceState::new_with_tokenizer(
                        scheduled_req.request.clone(),
                        input_tokens,
                        Some(self.tokenizer.clone()),
                    )
                });
                if !seq.prefill_complete {
                    prefill_ids.push(rid.clone());
                } else {
                    decode_ids.push(rid.clone());
                }
            }
        }
        if !prefill_ids.is_empty() {
            if let Err(e) = self.run_batch_prefill(&prefill_ids).await {
                warn!("Batch prefill failed: {}; falling back to per-request", e);
                for rid in &prefill_ids {
                    if let Err(e) = self.run_prefill(rid).await {
                        warn!("Prefill failed for {}: {}", rid, e);
                        self.complete_request(rid, FinishReason::Error).await?;
                    }
                }
            }
        }
        if decode_ids.len() > 1 {
            if let Err(e) = self.run_batch_decode(&decode_ids).await {
                warn!("Batch decode failed, falling back to per-request: {}", e);
                for rid in &decode_ids {
                    if let Err(e) = self.run_decode_step(rid).await {
                        warn!("Decode failed for {}: {}", rid, e);
                        self.complete_request(rid, FinishReason::Error).await?;
                    }
                }
            }
        } else {
            for rid in &decode_ids {
                if let Err(e) = self.run_decode_step(rid).await {
                    warn!("Decode failed for {}: {}", rid, e);
                    self.complete_request(rid, FinishReason::Error).await?;
                }
            }
        }
        Ok(())
    }

    /// Unified path: build ONE `UnifiedBatch` from all requests in the plan
    /// (prefill items get full input_tokens at pos_offset=0; decode items
    /// get [last_token] at pos_offset=current_kv_len), then a single
    /// `model_executor.unified_decode` call drives the entire forward.
    ///
    /// When the model's `unified_forward` returns Unsupported (Qwen3Moe
    /// today, until Phase 2 native), `LlmExecutor.unified_decode`'s fallback
    /// partitions the batch by item shape and serializes prefills under one
    /// model lock — behavior-preserving but no perf gain. Llama already
    /// supports unified_forward, so M2 immediately co-batches prefill+decode.
    async fn process_batch_unified(&self, batch: &ferrum_interfaces::BatchPlan) -> Result<()> {
        use ferrum_interfaces::model_executor::{UnifiedBatch, UnifiedBatchItem};
        use ferrum_interfaces::KvCacheHandle;

        // ── 0. Materialize SequenceState for every request, classify ──
        let mut prefill_ids: Vec<RequestId> = Vec::new();
        let mut decode_ids: Vec<RequestId> = Vec::new();
        {
            let mut sequences = self.sequences.write();
            for scheduled_req in &batch.requests {
                let rid = &scheduled_req.request.id;
                let seq = sequences.entry(rid.clone()).or_insert_with(|| {
                    let input_tokens = self
                        .tokenizer
                        .encode(&scheduled_req.request.prompt, true)
                        .unwrap_or_else(|_| vec![TokenId::new(0)]);
                    SequenceState::new_with_tokenizer(
                        scheduled_req.request.clone(),
                        input_tokens,
                        Some(self.tokenizer.clone()),
                    )
                });
                if !seq.prefill_complete {
                    prefill_ids.push(rid.clone());
                } else {
                    decode_ids.push(rid.clone());
                }
            }
        }

        // ── 1. Per-prefill setup: prefix-cache check, KV alloc, gather tokens ──
        // Prefix-cache hits short-circuit through the legacy single-prompt
        // path (they don't enter the unified batch — they have no model
        // call to make). The remainder are added to `unified_prefills`.
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
        struct UnifiedPrefillWork {
            rid: RequestId,
            input_tokens: Vec<TokenId>,
            kv_handle: Arc<dyn KvCacheHandle>,
            metadata: std::collections::HashMap<String, serde_json::Value>,
            fresh_kv: bool,
            chunk_start: usize,
            chunk_len: usize,
            is_final_chunk: bool,
        }

        let active_prefill_chunk_size = self.runtime_config.active_decode_prefill_chunk;
        let has_decode_items = !decode_ids.is_empty();
        let mut unified_prefills: Vec<UnifiedPrefillWork> = Vec::new();
        for rid in &prefill_ids {
            let (input_tokens, num_tokens, existing_kv, chunk_start, metadata) = {
                let sequences = self.sequences.read();
                let Some(seq) = sequences.get(rid) else {
                    continue;
                };
                (
                    seq.input_tokens.clone(),
                    seq.input_tokens.len(),
                    seq.kv_cache.clone(),
                    seq.prefill_tokens_processed,
                    seq.original_request.metadata.clone(),
                )
            };
            if chunk_start >= num_tokens {
                continue;
            }
            if chunk_start == 0 && !skip_prefix_cache {
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
                        if let Some(ref jp) = seq.json_processor {
                            jp.reset();
                        }
                        let mut logits = cached_logits;
                        let token = seq.sample_with_processors(&mut logits)?;
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
                    self.send_stream_update(rid, first_token).await;
                    let should_stop = {
                        let sequences = self.sequences.read();
                        sequences.get(rid).is_none_or(|s| s.should_stop())
                    };
                    if should_stop {
                        self.complete_request(rid, FinishReason::EOS).await?;
                    }
                    continue;
                }
            }

            let (kv_handle, fresh_kv) = if let Some(kv) = existing_kv {
                (kv, false)
            } else {
                // Allocate KV pages (with preempt fallback) for fresh prefill.
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
                let allocated = match self.kv_cache.allocate(&alloc_request).await {
                    Ok(h) => h,
                    Err(_) => {
                        if self.preempt_victim(rid).await {
                            match self.kv_cache.allocate(&alloc_request).await {
                                Ok(h) => h,
                                Err(e) => {
                                    warn!("Unified prefill alloc failed for {}: {}", rid, e);
                                    self.complete_request(rid, FinishReason::Error).await?;
                                    continue;
                                }
                            }
                        } else {
                            warn!("Unified prefill alloc failed for {}: no victim", rid);
                            self.complete_request(rid, FinishReason::Error).await?;
                            continue;
                        }
                    }
                };
                (allocated, true)
            };
            let remaining = num_tokens - chunk_start;
            let chunk_len = match active_prefill_chunk_size {
                Some(chunk) if has_decode_items || chunk_start > 0 => chunk.min(remaining),
                _ => remaining,
            };
            let is_final_chunk = chunk_start + chunk_len >= num_tokens;
            unified_prefills.push(UnifiedPrefillWork {
                rid: rid.clone(),
                input_tokens,
                kv_handle,
                metadata,
                fresh_kv,
                chunk_start,
                chunk_len,
                is_final_chunk,
            });
        }

        // ── 2. Build the UnifiedBatch (prefill chunks + decode tokens) ──
        let mut unified = UnifiedBatch::new();
        let mut prefill_meta: Vec<UnifiedPrefillWork> = Vec::new();
        let mut decode_meta: Vec<RequestId> = Vec::new();
        for work in unified_prefills {
            let chunk_end = work.chunk_start + work.chunk_len;
            let q_tokens: Vec<u32> = work.input_tokens[work.chunk_start..chunk_end]
                .iter()
                .map(|t| t.get())
                .collect();
            let seq_id = work.kv_handle.cache_id();
            unified.items.push(UnifiedBatchItem {
                seq_id,
                q_tokens,
                kv_cache: work.kv_handle.clone(),
                pos_offset: work.chunk_start,
                is_final_chunk: work.is_final_chunk,
                metadata: work.metadata.clone(),
            });
            prefill_meta.push(work);
        }
        {
            let sequences = self.sequences.read();
            for rid in &decode_ids {
                let Some(seq) = sequences.get(rid) else {
                    continue;
                };
                let Some(kv) = seq.kv_cache.clone() else {
                    continue;
                };
                let last_token = seq
                    .generated_tokens
                    .last()
                    .copied()
                    .unwrap_or(TokenId::new(0));
                // pos_offset = position of the NEW token in the K/V cache.
                // It must increment by 1 every decode step. Source of truth
                // is the engine's own bookkeeping: input prompt + tokens
                // generated so far (the last one is the one we're about to
                // decode, so its slot is `len - 1` past the prompt). NOT
                // `kv.block_table().sequence_length` — that field is set
                // once at allocate() time and `make_kv_handle_with_seq`
                // doesn't actually update Paged/Default handles, so reading
                // it leaves every decode step at the same position.
                let pos_offset = seq.input_tokens.len() + seq.generated_tokens.len() - 1;
                let seq_id = seq
                    .model_cache_id
                    .clone()
                    .unwrap_or_else(|| rid.to_string());
                unified.items.push(UnifiedBatchItem {
                    seq_id,
                    q_tokens: vec![last_token.get()],
                    kv_cache: kv,
                    pos_offset,
                    is_final_chunk: true,
                    metadata: seq.original_request.metadata.clone(),
                });
                decode_meta.push(rid.clone());
            }
        }

        if unified.items.is_empty() {
            return Ok(());
        }

        // ── 3. ONE unified forward call ──
        let unified_prof = self.runtime_config.unified_post_prof;
        let t_unified_model = if unified_prof {
            Some(Instant::now())
        } else {
            None
        };
        let results = match self.model_executor.unified_decode(&unified).await {
            Ok(r) => r,
            Err(e) => {
                warn!("Unified forward failed: {}; falling back to split", e);
                // Release the KV cache slots we just allocated for the
                // unified-path prefills — otherwise the legacy split
                // path's `run_batch_prefill` re-allocates for the same
                // request_id, double-counting `active_caches` (only one
                // of the two pairs ever gets deallocated by
                // `complete_request`). Found via paged_attention_test.
                for work in &prefill_meta {
                    if work.fresh_kv {
                        let _ = self.kv_cache.deallocate(work.rid.clone()).await;
                    }
                }
                return self.process_batch_legacy_split(batch).await;
            }
        };
        let t_unified_model_done = if unified_prof {
            Some(Instant::now())
        } else {
            None
        };
        if results.len() != unified.items.len() {
            return Err(FerrumError::internal(format!(
                "unified_decode returned {} results for {} items",
                results.len(),
                unified.items.len(),
            )));
        }

        // ── 4. Per-item post-process — split by category ──
        // Prefill items come first (in the order added), then decode items.
        let prefill_count = prefill_meta.len();
        let decode_count = decode_meta.len();
        let item_count = unified.items.len();
        let mut t_decode_sample_us: u64 = 0;
        let mut t_decode_sched_us: u64 = 0;
        let mut t_decode_stream_us: u64 = 0;
        let mut t_decode_stop_us: u64 = 0;
        let mut t_decode_complete_us: u64 = 0;
        for (i, work) in prefill_meta.into_iter().enumerate() {
            let logits_vec = match &results[i] {
                Some(l) => l.clone(),
                None if !work.is_final_chunk => {
                    let kv_handle = unified.items[i].kv_cache.clone();
                    {
                        let mut sequences = self.sequences.write();
                        if let Some(seq) = sequences.get_mut(&work.rid) {
                            seq.model_cache_id = Some(kv_handle.cache_id());
                            seq.kv_cache = Some(kv_handle);
                            seq.prefill_tokens_processed =
                                work.chunk_start.saturating_add(work.chunk_len);
                            seq.phase = RequestPhase::Prefilling;
                        }
                    }
                    self.scheduler.mark_prefill_chunk_processed(
                        &work.rid,
                        work.input_tokens.len(),
                        work.chunk_len,
                    );
                    continue;
                }
                None => {
                    warn!("Unified prefill result missing for {}", work.rid);
                    continue;
                }
            };
            let num_tokens = work.input_tokens.len();
            let kv_handle = unified.items[i].kv_cache.clone();
            if !skip_prefix_cache && logits_vec.len() > 1 {
                // Store in prefix cache (best-effort). Greedy-argmax results
                // are single-token sentinels, not reusable full logits.
                let _ = self.prefix_cache.store_prefix(
                    &work.input_tokens,
                    kv_handle.clone(),
                    logits_vec.clone(),
                );
            }
            let first_token = {
                let mut sequences = self.sequences.write();
                let Some(seq) = sequences.get_mut(&work.rid) else {
                    continue;
                };
                if let Some(ref jp) = seq.json_processor {
                    jp.reset();
                }
                let mut logits = logits_vec;
                let token = if logits.len() == 1 {
                    TokenId::new(logits[0] as u32)
                } else {
                    seq.sample_with_processors(&mut logits)?
                };
                seq.generated_tokens.push(token);
                seq.model_cache_id = Some(kv_handle.cache_id());
                seq.kv_cache = Some(kv_handle);
                seq.prefill_tokens_processed = num_tokens;
                seq.prefill_complete = true;
                seq.phase = RequestPhase::Decoding;
                token
            };
            self.scheduler.mark_prefill_complete(&work.rid, num_tokens);
            self.total_prefill_tokens
                .fetch_add(num_tokens as u64, Ordering::Relaxed);
            counter!("ferrum.engine.prefill_tokens_total").increment(num_tokens as u64);
            counter!("ferrum.engine.prefills_total").increment(1);
            self.send_stream_update(&work.rid, first_token).await;
            let should_stop = {
                let sequences = self.sequences.read();
                sequences.get(&work.rid).is_none_or(|s| s.should_stop())
            };
            if should_stop {
                self.complete_request(&work.rid, FinishReason::EOS).await?;
            }
        }
        for (j, rid) in decode_meta.into_iter().enumerate() {
            let i = prefill_count + j;
            let t0_sample = if unified_prof {
                Some(Instant::now())
            } else {
                None
            };
            let logits_vec = match &results[i] {
                Some(l) => l.clone(),
                None => continue,
            };
            let next_token = {
                let mut sequences = self.sequences.write();
                let Some(seq) = sequences.get_mut(&rid) else {
                    continue;
                };
                let mut logits = logits_vec;
                let token = if logits.len() == 1 {
                    TokenId::new(logits[0] as u32)
                } else {
                    seq.sample_with_processors(&mut logits)?
                };
                seq.generated_tokens.push(token);
                seq.tokens_this_iteration += 1;
                // pos_offset is sourced from SequenceState bookkeeping above
                // (`input_tokens.len() + generated_tokens.len() - 1`); the
                // engine-side KV handle's `sequence_length` field is no
                // longer load-bearing here. Resource handles like
                // PagedKvCacheHandle don't update the field anyway (the
                // model's internal paged_pool is what actually grows), so
                // the previous `make_kv_handle_with_seq` write was a
                // silent no-op for production handles.
                token
            };
            if let Some(t0) = t0_sample {
                t_decode_sample_us += t0.elapsed().as_micros() as u64;
            }
            let t0_sched = if unified_prof {
                Some(Instant::now())
            } else {
                None
            };
            let generated_count = {
                let sequences = self.sequences.read();
                sequences
                    .get(&rid)
                    .map(|s| s.generated_tokens.len())
                    .unwrap_or(0)
            };
            self.scheduler.update_decode_progress(&rid, generated_count);
            self.total_decode_tokens.fetch_add(1, Ordering::Relaxed);
            counter!("ferrum.engine.decode_tokens_total").increment(1);
            if let Some(t0) = t0_sched {
                t_decode_sched_us += t0.elapsed().as_micros() as u64;
            }
            let t0_stream = if unified_prof {
                Some(Instant::now())
            } else {
                None
            };
            self.send_stream_update(&rid, next_token).await;
            if let Some(t0) = t0_stream {
                t_decode_stream_us += t0.elapsed().as_micros() as u64;
            }
            let t0_stop = if unified_prof {
                Some(Instant::now())
            } else {
                None
            };
            let should_stop = {
                let sequences = self.sequences.read();
                sequences.get(&rid).is_none_or(|s| s.should_stop())
            };
            if let Some(t0) = t0_stop {
                t_decode_stop_us += t0.elapsed().as_micros() as u64;
            }
            if should_stop {
                let t0_complete = if unified_prof {
                    Some(Instant::now())
                } else {
                    None
                };
                let finish_reason = {
                    let sequences = self.sequences.read();
                    match sequences.get(&rid) {
                        Some(seq)
                            if seq.generated_tokens.len() >= seq.sampling_params.max_tokens =>
                        {
                            FinishReason::Length
                        }
                        Some(_) => FinishReason::EOS,
                        None => FinishReason::Error,
                    }
                };
                self.complete_request(&rid, finish_reason).await?;
                if let Some(t0) = t0_complete {
                    t_decode_complete_us += t0.elapsed().as_micros() as u64;
                }
            }
        }
        if let (Some(t0), Some(t1)) = (t_unified_model, t_unified_model_done) {
            use std::sync::atomic::AtomicU64;
            static UNIFIED_PROF_N: AtomicU64 = AtomicU64::new(0);
            let n = UNIFIED_PROF_N.fetch_add(1, Ordering::Relaxed);
            if n < 64 || n.is_multiple_of(32) {
                let model_us = t1.duration_since(t0).as_micros() as u64;
                let total_us = t0.elapsed().as_micros() as u64;
                let decode_post_us = t_decode_sample_us
                    + t_decode_sched_us
                    + t_decode_stream_us
                    + t_decode_stop_us
                    + t_decode_complete_us;
                eprintln!(
                    "[unified-prof] iter#{} items={} prefill={} decode={} total={}us model={}us decode_post={}us | sample={} sched={} stream={} stop={} complete={} (us)",
                    n,
                    item_count,
                    prefill_count,
                    decode_count,
                    total_us,
                    model_us,
                    decode_post_us,
                    t_decode_sample_us,
                    t_decode_sched_us,
                    t_decode_stream_us,
                    t_decode_stop_us,
                    t_decode_complete_us,
                );
                let profile = global_profile();
                if profile.is_enabled() {
                    let _ = profile.push_event(
                        "unified_prof",
                        profile_fields_from_json(serde_json::json!({
                            "iter": n,
                            "items": item_count,
                            "prefill": prefill_count,
                            "decode": decode_count,
                        })),
                        profile_fields_from_json(serde_json::json!({
                            "total": total_us,
                            "model": model_us,
                            "decode_post": decode_post_us,
                            "sample": t_decode_sample_us,
                            "sched": t_decode_sched_us,
                            "stream": t_decode_stream_us,
                            "stop": t_decode_stop_us,
                            "complete": t_decode_complete_us,
                        })),
                        false,
                    );
                }
            }
        }

        Ok(())
    }

    // ── preemption ──────────────────────────────────────────────────────

    /// Try to preempt a decoding request to free KV cache blocks.
    ///
    /// Picks the lowest-priority victim (ties broken by fewest generated
    /// tokens — least work lost).  Frees the victim's KV cache, resets
    /// its sequence state, and re-submits it to the scheduler so it will
    /// be re-prefilled in a later iteration.
    ///
    /// Returns `true` if a victim was preempted.
    pub(super) async fn preempt_victim(&self, exclude_id: &RequestId) -> bool {
        // Select victim: any decoding sequence except the requester
        let victim_id = {
            let sequences = self.sequences.read();
            sequences
                .iter()
                .filter(|(id, s)| *id != exclude_id && s.prefill_complete && s.kv_cache.is_some())
                .min_by(|(_, a), (_, b)| {
                    // Lowest priority first, then fewest generated tokens
                    a.sampling_params
                        .max_tokens // proxy for priority (TODO: use real priority)
                        .cmp(&b.sampling_params.max_tokens)
                        .then_with(|| a.generated_tokens.len().cmp(&b.generated_tokens.len()))
                })
                .map(|(id, _)| id.clone())
        };

        let victim_id = match victim_id {
            Some(id) => id,
            None => return false,
        };

        info!("Preempting request {} to free KV blocks", victim_id);

        // Free model executor's KV cache for this sequence
        {
            let sequences = self.sequences.read();
            if let Some(seq) = sequences.get(&victim_id) {
                if let Some(ref cache_id) = seq.model_cache_id {
                    self.model_executor.release_cache(cache_id);
                }
            }
        }

        // Free KV cache manager blocks
        let _ = self.kv_cache.deallocate(victim_id.clone()).await;

        // Reset sequence state — keep response/stream channels intact
        {
            let mut sequences = self.sequences.write();
            if let Some(seq) = sequences.get_mut(&victim_id) {
                seq.kv_cache = None;
                seq.model_cache_id = None;
                seq.generated_tokens.clear();
                seq.prefill_complete = false;
                seq.prefill_tokens_processed = 0;
                seq.phase = RequestPhase::Waiting;
                seq.tokens_this_iteration = 0;
                seq.preemption_count += 1;
                // Reset RNG to original seed for deterministic re-generation
                let seed = seq.sampling_params.seed.unwrap_or(42);
                seq.rng = StdRng::seed_from_u64(seed);
            }
        }

        // Cancel in scheduler and re-submit so it goes back to waiting queue
        let _ = self.scheduler.cancel(victim_id.clone()).await;
        let request = {
            let sequences = self.sequences.read();
            sequences
                .get(&victim_id)
                .map(|s| s.original_request.clone())
        };
        if let Some(req) = request {
            let _ = self.scheduler.submit(req).await;
        }

        self.total_preemptions.fetch_add(1, Ordering::Relaxed);
        counter!("ferrum.engine.preemptions_total").increment(1);
        true
    }
}
