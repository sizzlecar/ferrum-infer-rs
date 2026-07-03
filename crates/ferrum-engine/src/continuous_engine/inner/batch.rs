use super::*;

impl EngineInner {
    // ── batch processing ───────────────────────────────────────────────

    pub(in crate::continuous_engine) async fn process_batch(
        &self,
        batch: &ferrum_interfaces::BatchPlan,
    ) -> Result<()> {
        // Single-shot unified path: prefill + decode items go through ONE
        // `model_executor.unified_decode` call. Phase-2/3 redesign goal —
        // prefill chunks and decode tokens are co-batched at the kernel
        // level, eliminating the cohort gap (decode_queue=0 ↔ 32
        // alternation that consumed half the bench wall on apples).
        //
        // Speculative decoding still owns a separate multi-token verify path.
        // Typed chunked prefill can stay on the unified path: the unified
        // producer below emits non-final prefill chunks and can co-batch them
        // with decode work when the executor has a native unified forward.
        if self.spec_config.is_some() {
            return self.process_batch_legacy_split(batch).await;
        }
        let info = self.model_executor.info();
        let is_qwen3_moe = info
            .metadata
            .get("ferrum_arch")
            .and_then(|value| value.as_str())
            == Some("qwen3moe");
        let backend_lacks_native_unified = !self.model_executor.supports_native_unified_decode();
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
                    SequenceState::new_with_tokenizer_and_model_vocab_size(
                        scheduled_req.request.clone(),
                        input_tokens,
                        Some(self.tokenizer.clone()),
                        Some(self.model_executor.info().vocab_size),
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
                        if is_resource_exhausted_error(&e) {
                            continue;
                        }
                        self.complete_request(rid, FinishReason::Error).await?;
                    }
                }
            }
        }
        let decode_ids = self.decode_ready_request_ids(&decode_ids);
        if decode_ids.len() > 1 {
            if let Err(e) = self.run_batch_decode_adaptive(&decode_ids).await {
                warn!("Batch decode failed, falling back to per-request: {}", e);
                for rid in self.decode_ready_request_ids(&decode_ids) {
                    if let Err(e) = self.run_decode_step(&rid).await {
                        warn!("Decode failed for {}: {}", rid, e);
                        if is_resource_exhausted_error(&e) {
                            continue;
                        }
                        self.complete_request(&rid, FinishReason::Error).await?;
                    }
                }
            }
        } else {
            for rid in self.decode_ready_request_ids(&decode_ids) {
                if let Err(e) = self.run_decode_step(&rid).await {
                    warn!("Decode failed for {}: {}", rid, e);
                    if is_resource_exhausted_error(&e) {
                        continue;
                    }
                    self.complete_request(&rid, FinishReason::Error).await?;
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
        use ferrum_interfaces::model_executor::{
            LogitsReturnPolicy, UnifiedBatch, UnifiedBatchItem,
        };
        use ferrum_interfaces::KvCacheHandle;

        // ── 0. Materialize SequenceState for every request, classify ──
        let mut prefill_ids: Vec<RequestId> = Vec::new();
        let mut decode_ids: Vec<RequestId> = Vec::new();
        let scheduled_tokens_by_id: HashMap<RequestId, usize> = batch
            .requests
            .iter()
            .filter_map(|scheduled_req| {
                scheduled_req
                    .tokens_to_process
                    .map(|tokens| (scheduled_req.request.id.clone(), tokens))
            })
            .collect();
        {
            let mut sequences = self.sequences.write();
            for scheduled_req in &batch.requests {
                let rid = &scheduled_req.request.id;
                let seq = sequences.entry(rid.clone()).or_insert_with(|| {
                    let input_tokens = self
                        .tokenizer
                        .encode(&scheduled_req.request.prompt, true)
                        .unwrap_or_else(|_| vec![TokenId::new(0)]);
                    SequenceState::new_with_tokenizer_and_model_vocab_size(
                        scheduled_req.request.clone(),
                        input_tokens,
                        Some(self.tokenizer.clone()),
                        Some(self.model_executor.info().vocab_size),
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
            recurrent_state: Option<Arc<dyn ferrum_interfaces::RecurrentStateHandle>>,
            metadata: std::collections::HashMap<String, serde_json::Value>,
            logits_policy: LogitsReturnPolicy,
            can_use_prefix_cache: bool,
            fresh_kv: bool,
            kv_resource_blocks: Option<usize>,
            fresh_recurrent: bool,
            chunk_start: usize,
            chunk_len: usize,
            is_final_chunk: bool,
        }

        let explicit_active_prefill_chunk_size = self.runtime_config.active_decode_prefill_chunk;
        let has_decode_items = !decode_ids.is_empty();
        let mut unified_prefills: Vec<UnifiedPrefillWork> = Vec::new();
        for rid in &prefill_ids {
            let (
                input_tokens,
                num_tokens,
                existing_kv,
                existing_kv_resource_blocks,
                existing_recurrent_state,
                chunk_start,
                mut metadata,
                logits_policy,
                can_use_prefix_cache,
            ) = {
                let sequences = self.sequences.read();
                let Some(seq) = sequences.get(rid) else {
                    continue;
                };
                (
                    seq.prefill_context_tokens(),
                    seq.prefill_context_len(),
                    seq.kv_cache.clone(),
                    seq.kv_resource_blocks,
                    seq.recurrent_state.clone(),
                    seq.prefill_tokens_processed,
                    seq.model_decode_metadata(),
                    if skip_prefix_cache {
                        seq.model_decode_logits_policy()
                    } else {
                        LogitsReturnPolicy::FullLogits
                    },
                    seq.generated_tokens.is_empty(),
                )
            };
            if chunk_start >= num_tokens {
                continue;
            }
            let recurrent_state_spec = self
                .model_executor
                .recurrent_state_spec(rid, &input_tokens)?;
            let skip_request_prefix_cache =
                skip_prefix_cache || !can_use_prefix_cache || recurrent_state_spec.is_some();
            if chunk_start == 0 && !skip_request_prefix_cache {
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

            if existing_recurrent_state.is_none() {
                if let (Some(spec), Some(manager)) =
                    (recurrent_state_spec.as_ref(), &self.recurrent_state_manager)
                {
                    if !manager.can_allocate(spec) {
                        warn!(
                            "Unified prefill recurrent-state alloc deferred for {}: insufficient capacity",
                            rid
                        );
                        self.defer_prefill_for_capacity(rid).await;
                        continue;
                    }
                }
            }

            let had_recurrent_state = existing_recurrent_state.is_some();
            let recurrent_state =
                match self.ensure_recurrent_state(rid, recurrent_state_spec).await {
                    Ok(state) => state,
                    Err(FerrumError::ResourceExhausted { message }) => {
                        warn!(
                            "Unified prefill recurrent-state alloc deferred for {}: {}",
                            rid, message
                        );
                        self.defer_prefill_for_capacity(rid).await;
                        continue;
                    }
                    Err(e) => {
                        warn!(
                            "Unified prefill recurrent-state alloc failed for {}: {}",
                            rid, e
                        );
                        self.complete_request(rid, FinishReason::Error).await?;
                        continue;
                    }
                }
                .or(existing_recurrent_state);
            let fresh_recurrent = !had_recurrent_state && recurrent_state.is_some();

            let (kv_handle, fresh_kv, kv_resource_blocks) = if let Some(kv) = existing_kv {
                (kv, false, existing_kv_resource_blocks)
            } else {
                // Allocate KV pages for a fresh prefill. This is waiting-request
                // admission, so capacity failure should defer the prefill rather
                // than preempt running decode work.
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
                let lease = match self
                    .allocate_kv_lease(rid, rid.clone(), &alloc_request, num_tokens)
                    .await
                {
                    Ok(lease) => lease,
                    Err(e) => {
                        warn!("Unified prefill alloc deferred for {}: {}", rid, e);
                        self.release_recurrent_state(rid).await;
                        self.defer_prefill_for_capacity(rid).await;
                        continue;
                    }
                };
                let allocated = lease.handle();
                let (_allocation_request_id, blocks) = lease.into_committed_parts();
                (allocated, true, Some(blocks))
            };
            let remaining = num_tokens - chunk_start;
            let chunk_len = [
                scheduled_tokens_by_id.get(rid).copied(),
                match explicit_active_prefill_chunk_size {
                    Some(chunk) if has_decode_items || chunk_start > 0 => Some(chunk),
                    _ => None,
                },
                self.runtime_config.chunked_prefill_size_for(num_tokens),
            ]
            .into_iter()
            .flatten()
            .min()
            .map(|chunk| chunk.min(remaining).max(1))
            .unwrap_or(remaining);
            let is_final_chunk = chunk_start + chunk_len >= num_tokens;
            if has_decode_items && !is_final_chunk {
                metadata.remove(KV_ADMISSION_TARGET_LEN_METADATA_KEY);
            }
            unified_prefills.push(UnifiedPrefillWork {
                rid: rid.clone(),
                input_tokens,
                kv_handle,
                recurrent_state,
                metadata,
                logits_policy,
                can_use_prefix_cache,
                fresh_kv,
                kv_resource_blocks,
                fresh_recurrent,
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
                recurrent_state: work.recurrent_state.clone(),
                pos_offset: work.chunk_start,
                is_final_chunk: work.is_final_chunk,
                metadata: work.metadata.clone(),
                logits_policy: work.logits_policy.clone(),
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
                    recurrent_state: seq.recurrent_state.clone(),
                    pos_offset,
                    is_final_chunk: true,
                    metadata: seq.model_decode_metadata(),
                    logits_policy: seq.model_decode_logits_policy(),
                });
                decode_meta.push(rid.clone());
            }
        }

        if unified.items.is_empty() {
            return Ok(());
        }

        // ── 3. ONE unified forward call ──
        let unified_prof = self.runtime_config.unified_post_prof;
        let first_token_prof = self.runtime_config.batch_decode_prof || unified_prof;
        let prefill_model_start_wait_us: std::collections::HashMap<RequestId, u64> =
            if first_token_prof {
                let sequences = self.sequences.read();
                prefill_meta
                    .iter()
                    .filter_map(|work| {
                        sequences.get(&work.rid).map(|seq| {
                            (
                                work.rid.clone(),
                                seq.start_time.elapsed().as_micros() as u64,
                            )
                        })
                    })
                    .collect()
            } else {
                std::collections::HashMap::new()
            };
        let t_unified_model = if unified_prof {
            Some(Instant::now())
        } else {
            None
        };
        let kv_requests = kv_slot_requests_for_unified_batch(&unified);
        if let Err(e) = self.model_executor.reserve_kv_slots(&kv_requests) {
            warn!("Unified KV admission failed: {}", e);
            if is_resource_exhausted_error(&e) {
                let pressure = paged_kv_admission_pressure(&e);
                let mixed_decode_prefill = !decode_meta.is_empty() && !prefill_meta.is_empty();
                if !decode_meta.is_empty() && prefill_meta.is_empty() {
                    if let Some(pressure) = pressure {
                        self.scheduler.record_decode_capacity_pressure(
                            decode_meta.len(),
                            Some(pressure.free_blocks),
                        );
                        self.scheduler
                            .defer_capacity_deferred_mixed_recompute_until_kv_capacity(
                                Some(pressure.admission_blocks),
                                Some(pressure.free_blocks),
                                Some(decode_meta.len()),
                            );
                    } else {
                        self.scheduler
                            .defer_capacity_deferred_mixed_recompute_until_release();
                    }
                }
                for work in &prefill_meta {
                    if work.fresh_kv {
                        self.release_kv_allocation(
                            &work.rid,
                            work.rid.clone(),
                            work.kv_resource_blocks,
                        )
                        .await;
                    }
                    self.defer_prefill_for_capacity(&work.rid).await;
                }
                if mixed_decode_prefill {
                    self.scheduler
                        .defer_capacity_deferred_mixed_recompute_until_kv_capacity(
                            pressure.map(|pressure| pressure.admission_blocks),
                            pressure.map(|pressure| pressure.free_blocks),
                            Some(prefill_meta.len()),
                        );
                }
                if !decode_meta.is_empty() {
                    return self
                        .run_batch_decode_adaptive_no_preempt(&decode_meta)
                        .await;
                }
                return Ok(());
            }
            for work in &prefill_meta {
                if work.fresh_kv {
                    self.release_kv_allocation(
                        &work.rid,
                        work.rid.clone(),
                        work.kv_resource_blocks,
                    )
                    .await;
                }
                if work.fresh_recurrent {
                    self.release_recurrent_state(&work.rid).await;
                }
            }
            return self.process_batch_legacy_split(batch).await;
        }
        let results = match self.model_executor.unified_decode(&unified).await {
            Ok(r) => r,
            Err(e) => {
                if is_resource_exhausted_error(&e) {
                    warn!(
                        "Unified forward resource exhausted: {}; deferring prefills",
                        e
                    );
                    for work in &prefill_meta {
                        if work.fresh_kv {
                            self.release_kv_allocation(
                                &work.rid,
                                work.rid.clone(),
                                work.kv_resource_blocks,
                            )
                            .await;
                        }
                        self.defer_prefill_for_capacity(&work.rid).await;
                    }
                    if !decode_meta.is_empty() {
                        return self
                            .run_batch_decode_adaptive_no_preempt(&decode_meta)
                            .await;
                    }
                    return Ok(());
                }
                warn!("Unified forward failed: {}; falling back to split", e);
                // Release the KV cache slots we just allocated for the
                // unified-path prefills — otherwise the legacy split
                // path's `run_batch_prefill` re-allocates for the same
                // request_id, double-counting `active_caches` (only one
                // of the two pairs ever gets deallocated by
                // `complete_request`). Found via paged_attention_test.
                for work in &prefill_meta {
                    if work.fresh_kv {
                        self.release_kv_allocation(
                            &work.rid,
                            work.rid.clone(),
                            work.kv_resource_blocks,
                        )
                        .await;
                    }
                    if work.fresh_recurrent {
                        self.release_recurrent_state(&work.rid).await;
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
        let unified_model_us = t_unified_model
            .zip(t_unified_model_done)
            .map(|(t0, t1)| t1.duration_since(t0).as_micros() as u64);
        if results.len() != unified.items.len() {
            for work in &prefill_meta {
                if work.fresh_kv {
                    self.release_kv_allocation(
                        &work.rid,
                        work.rid.clone(),
                        work.kv_resource_blocks,
                    )
                    .await;
                }
                if work.fresh_recurrent {
                    self.release_recurrent_state(&work.rid).await;
                }
            }
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
                    let cache_id = unified.items[i].seq_id.clone();
                    let kv_len = work.chunk_start.saturating_add(work.chunk_len);
                    let model_kv = self.make_model_kv_handle_with_seq(cache_id.clone(), kv_len);
                    {
                        let mut sequences = self.sequences.write();
                        if let Some(seq) = sequences.get_mut(&work.rid) {
                            seq.model_cache_id = Some(cache_id);
                            seq.kv_cache = Some(model_kv);
                            seq.kv_resource_blocks = work.kv_resource_blocks;
                            seq.recurrent_state = unified.items[i].recurrent_state.clone();
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
                    if work.fresh_kv {
                        self.release_kv_allocation(
                            &work.rid,
                            work.rid.clone(),
                            work.kv_resource_blocks,
                        )
                        .await;
                    }
                    if work.fresh_recurrent {
                        self.release_recurrent_state(&work.rid).await;
                    }
                    self.complete_request(&work.rid, FinishReason::Error)
                        .await?;
                    continue;
                }
            };
            let num_tokens = work.input_tokens.len();
            let cache_id = unified.items[i].seq_id.clone();
            let model_kv = self.make_model_kv_handle_with_seq(cache_id.clone(), num_tokens);
            if !skip_prefix_cache && work.can_use_prefix_cache && logits_vec.len() > 1 {
                // Store in prefix cache (best-effort). Greedy-argmax results
                // are single-token sentinels, not reusable full logits.
                let _ = self.prefix_cache.store_prefix(
                    &work.input_tokens,
                    model_kv.clone(),
                    logits_vec.clone(),
                );
            }
            let first_token_result = (|| {
                let mut sequences = self.sequences.write();
                let Some(seq) = sequences.get_mut(&work.rid) else {
                    return Ok(None);
                };
                seq.reset_guided_processors()?;
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
                seq.model_cache_id = Some(cache_id.clone());
                seq.kv_cache = Some(model_kv);
                seq.kv_resource_blocks = work.kv_resource_blocks;
                seq.recurrent_state = unified.items[i].recurrent_state.clone();
                seq.prefill_tokens_processed = num_tokens;
                seq.prefill_complete = true;
                seq.phase = RequestPhase::Decoding;
                Ok::<Option<(TokenId, u64)>, FerrumError>(Some((
                    token,
                    seq.start_time.elapsed().as_micros() as u64,
                )))
            })();
            let (first_token, queue_to_first_token_us) = match first_token_result {
                Ok(Some(value)) => value,
                Ok(None) => continue,
                Err(e) => {
                    warn!(
                        "Unified prefill post-process failed for {}: {}",
                        work.rid, e
                    );
                    if work.fresh_kv {
                        self.release_kv_allocation(
                            &work.rid,
                            work.rid.clone(),
                            work.kv_resource_blocks,
                        )
                        .await;
                    }
                    if work.fresh_recurrent {
                        self.release_recurrent_state(&work.rid).await;
                    }
                    self.complete_request(&work.rid, FinishReason::Error)
                        .await?;
                    continue;
                }
            };
            self.scheduler.mark_prefill_complete(&work.rid, num_tokens);
            self.total_prefill_tokens
                .fetch_add(num_tokens as u64, Ordering::Relaxed);
            counter!("ferrum.engine.prefill_tokens_total").increment(num_tokens as u64);
            counter!("ferrum.engine.prefills_total").increment(1);
            if first_token_prof {
                let queue_to_model_start_us = prefill_model_start_wait_us
                    .get(&work.rid)
                    .copied()
                    .unwrap_or(0);
                let model_batch_us = unified_model_us.unwrap_or(0);
                eprintln!(
                    "[first-token-prof] req={} source=unified_prefill prompt_tokens={} chunk_start={} chunk_len={} queue_to_model_start={}us model_batch={}us queue_to_first_token={}us",
                    work.rid,
                    num_tokens,
                    work.chunk_start,
                    work.chunk_len,
                    queue_to_model_start_us,
                    model_batch_us,
                    queue_to_first_token_us,
                );
                let profile = global_profile();
                if profile.is_enabled() {
                    let _ = profile.push_event(
                        "first_token_prof",
                        profile_fields_from_json(serde_json::json!({
                            "source": "unified_prefill",
                            "request_id": work.rid.to_string(),
                            "prompt_tokens": num_tokens,
                            "chunk_start": work.chunk_start,
                            "chunk_len": work.chunk_len,
                        })),
                        profile_fields_from_json(serde_json::json!({
                            "queue_to_model_start": queue_to_model_start_us,
                            "model_batch": model_batch_us,
                            "queue_to_first_token": queue_to_first_token_us,
                        })),
                        false,
                    );
                }
            }
            let stop_reason = self.stop_reason_for_request(&work.rid);
            if self.should_stream_generated_token(stop_reason) {
                self.send_stream_update(&work.rid, first_token).await;
            }
            if let Some(reason) = stop_reason {
                self.complete_request(&work.rid, reason).await?;
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
                None => {
                    warn!("Unified decode result missing for {}", rid);
                    self.complete_request(&rid, FinishReason::Error).await?;
                    continue;
                }
            };
            let next_token_result = (|| {
                let mut sequences = self.sequences.write();
                let Some(seq) = sequences.get_mut(&rid) else {
                    return Ok(None);
                };
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
                let cache_id = seq
                    .model_cache_id
                    .clone()
                    .unwrap_or_else(|| rid.to_string());
                let kv_len = seq
                    .input_tokens
                    .len()
                    .saturating_add(seq.generated_tokens.len())
                    .saturating_sub(1);
                seq.kv_cache = Some(self.make_model_kv_handle_with_seq(cache_id, kv_len));
                seq.tokens_this_iteration += 1;
                // pos_offset is sourced from SequenceState bookkeeping above
                // (`input_tokens.len() + generated_tokens.len() - 1`); the
                // engine-side KV handle's `sequence_length` field is no
                // longer load-bearing here. Resource handles like
                // PagedKvCacheHandle don't update the field anyway (the
                // model's internal paged_pool is what actually grows), so
                // the previous `make_kv_handle_with_seq` write was a
                // silent no-op for production handles.
                Ok::<Option<TokenId>, FerrumError>(Some(token))
            })();
            let next_token = match next_token_result {
                Ok(Some(token)) => token,
                Ok(None) => continue,
                Err(e) => {
                    warn!("Unified decode post-process failed for {}: {}", rid, e);
                    self.complete_request(&rid, FinishReason::Error).await?;
                    continue;
                }
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
            let stop_reason = self.stop_reason_for_request(&rid);
            if self.should_stream_generated_token(stop_reason) {
                self.send_stream_update(&rid, next_token).await;
            }
            if let Some(t0) = t0_stream {
                t_decode_stream_us += t0.elapsed().as_micros() as u64;
            }
            let t0_stop = if unified_prof {
                Some(Instant::now())
            } else {
                None
            };
            if let Some(t0) = t0_stop {
                t_decode_stop_us += t0.elapsed().as_micros() as u64;
            }
            if let Some(reason) = stop_reason {
                let t0_complete = if unified_prof {
                    Some(Instant::now())
                } else {
                    None
                };
                self.complete_request(&rid, reason).await?;
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

    async fn defer_prefill_for_capacity(&self, request_id: &RequestId) {
        let (
            had_kv_cache,
            kv_resource_blocks,
            draft_kv_request_id,
            draft_kv_resource_blocks,
            had_recurrent_state,
            model_cache_id,
        ) = {
            let mut sequences = self.sequences.write();
            if let Some(seq) = sequences.get_mut(request_id) {
                let had_kv_cache = seq.kv_cache.is_some();
                let kv_resource_blocks = seq.kv_resource_blocks.take();
                let draft_kv_request_id = seq.draft_kv_request_id.clone();
                let draft_kv_resource_blocks = seq.draft_kv_resource_blocks.take();
                let had_recurrent_state = seq.recurrent_state.is_some();
                let model_cache_id = seq.model_cache_id.clone();
                seq.kv_cache = None;
                seq.draft_kv_cache = None;
                seq.draft_kv_request_id = None;
                seq.recurrent_state = None;
                seq.model_cache_id = None;
                seq.prefill_complete = false;
                seq.prefill_tokens_processed = 0;
                seq.phase = RequestPhase::Waiting;
                seq.tokens_this_iteration = 0;
                (
                    had_kv_cache,
                    kv_resource_blocks,
                    draft_kv_request_id,
                    draft_kv_resource_blocks,
                    had_recurrent_state,
                    model_cache_id,
                )
            } else {
                (false, None, None, None, false, None)
            }
        };

        if let Some(cache_id) = model_cache_id {
            self.model_executor.release_cache(&cache_id);
        }
        if had_kv_cache {
            self.release_kv_allocation(request_id, request_id.clone(), kv_resource_blocks)
                .await;
        }
        if let Some(draft_request_id) = draft_kv_request_id {
            self.release_kv_allocation(request_id, draft_request_id, draft_kv_resource_blocks)
                .await;
        }
        if had_recurrent_state {
            self.release_recurrent_state(request_id).await;
        }
        self.scheduler.defer_prefill_to_waiting(request_id);
        self.trace_scheduler_defer(
            request_id,
            "engine_scheduler_prefill_capacity_defer",
            "prefill capacity deferred until KV/recurrent resources are released",
        );
        self.work_notify.notify_waiters();
    }

    pub(super) async fn defer_decode_for_capacity_recompute(
        &self,
        request_id: &RequestId,
        attempted_decode_width: usize,
        observed_free_blocks: Option<usize>,
    ) -> bool {
        let (
            found,
            had_kv_cache,
            kv_resource_blocks,
            draft_kv_request_id,
            draft_kv_resource_blocks,
            had_recurrent_state,
            model_cache_id,
        ) = {
            let mut sequences = self.sequences.write();
            if let Some(seq) = sequences.get_mut(request_id) {
                let had_kv_cache = seq.kv_cache.is_some();
                let kv_resource_blocks = seq.kv_resource_blocks.take();
                let draft_kv_request_id = seq.draft_kv_request_id.clone();
                let draft_kv_resource_blocks = seq.draft_kv_resource_blocks.take();
                let had_recurrent_state = seq.recurrent_state.is_some();
                let model_cache_id = seq.model_cache_id.clone();
                seq.kv_cache = None;
                seq.draft_kv_cache = None;
                seq.draft_kv_request_id = None;
                seq.recurrent_state = None;
                seq.model_cache_id = None;
                seq.prefill_complete = false;
                seq.prefill_tokens_processed = 0;
                seq.phase = RequestPhase::Waiting;
                seq.tokens_this_iteration = 0;
                seq.preemption_count += 1;
                (
                    true,
                    had_kv_cache,
                    kv_resource_blocks,
                    draft_kv_request_id,
                    draft_kv_resource_blocks,
                    had_recurrent_state,
                    model_cache_id,
                )
            } else {
                (false, false, None, None, None, false, None)
            }
        };

        if !found {
            return false;
        }

        if let Some(cache_id) = model_cache_id {
            self.model_executor.release_cache(&cache_id);
        }
        if had_kv_cache {
            self.release_kv_allocation(request_id, request_id.clone(), kv_resource_blocks)
                .await;
        }
        if let Some(draft_request_id) = draft_kv_request_id {
            self.release_kv_allocation(request_id, draft_request_id, draft_kv_resource_blocks)
                .await;
        }
        if had_recurrent_state {
            self.release_recurrent_state(request_id).await;
        }

        let moved = self
            .scheduler
            .defer_decode_to_waiting_for_capacity_with_pressure(
                request_id,
                attempted_decode_width,
                observed_free_blocks,
            );
        if moved {
            info!(
                "Capacity-deferred decode request {} for KV recompute after failed width {}",
                request_id, attempted_decode_width
            );
            self.trace_scheduler_defer(
                request_id,
                "engine_scheduler_decode_capacity_defer",
                "decode capacity deferred for KV recompute",
            );
            self.work_notify.notify_waiters();
        }
        moved
    }

    /// Try to preempt a decoding request to free KV cache blocks.
    ///
    /// Picks the lowest-priority victim (ties broken by fewest generated
    /// tokens — least work lost). Frees the victim's physical KV cache and
    /// re-submits it to the scheduler. The logical output state is preserved:
    /// the next prefill rebuilds KV from prompt + already-generated tokens,
    /// mirroring vLLM-style recompute preemption instead of duplicating or
    /// dropping streamed output.
    ///
    /// Returns `true` if a victim was preempted.
    pub(super) async fn preempt_victim(&self, exclude_id: &RequestId) -> bool {
        let exclude = std::collections::HashSet::from([exclude_id.clone()]);
        self.preempt_victim_excluding(&exclude).await
    }

    /// Try to preempt a decoding request outside `exclude_ids`.
    pub(super) async fn preempt_victim_excluding(
        &self,
        exclude_ids: &std::collections::HashSet<RequestId>,
    ) -> bool {
        // Select victim: any decoding sequence except the requester
        let victim_id = {
            let sequences = self.sequences.read();
            sequences
                .iter()
                .filter(|(id, s)| {
                    !exclude_ids.contains(*id) && s.prefill_complete && s.kv_cache.is_some()
                })
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
        let (kv_resource_blocks, draft_kv_request_id, draft_kv_resource_blocks, had_recurrent) = {
            let sequences = self.sequences.read();
            if let Some(seq) = sequences.get(&victim_id) {
                (
                    seq.kv_resource_blocks,
                    seq.draft_kv_request_id.clone(),
                    seq.draft_kv_resource_blocks,
                    seq.recurrent_state.is_some(),
                )
            } else {
                (None, None, None, false)
            }
        };

        self.release_kv_allocation(&victim_id, victim_id.clone(), kv_resource_blocks)
            .await;
        if let Some(draft_request_id) = draft_kv_request_id {
            self.release_kv_allocation(&victim_id, draft_request_id, draft_kv_resource_blocks)
                .await;
        }

        if had_recurrent {
            self.release_recurrent_state(&victim_id).await;
        }

        // Reset only physical model state. Keep generated_tokens, RNG,
        // token-frequency, and stream offsets intact; those are logical
        // request state and are needed to continue without replaying output
        // to the client after KV recompute.
        {
            let mut sequences = self.sequences.write();
            if let Some(seq) = sequences.get_mut(&victim_id) {
                seq.kv_cache = None;
                seq.kv_resource_blocks = None;
                seq.draft_kv_cache = None;
                seq.draft_kv_request_id = None;
                seq.draft_kv_resource_blocks = None;
                seq.recurrent_state = None;
                seq.recurrent_state_slots = None;
                seq.model_cache_id = None;
                seq.prefill_complete = false;
                seq.prefill_tokens_processed = 0;
                seq.phase = RequestPhase::Waiting;
                seq.tokens_this_iteration = 0;
                seq.preemption_count += 1;
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
