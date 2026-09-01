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

                let prefix_hit_result = (|| {
                    let cloned_kv = cached_kv.clone_handle()?;
                    let mut sequences = self.sequences.write();
                    let seq = sequences
                        .get_mut(request_id)
                        .ok_or_else(|| FerrumError::internal("Sequence not found"))?;
                    seq.reset_guided_processors()?;
                    let mut logits = cached_logits;
                    let token = seq.sample_and_commit_with_processors_and_tokenizer(
                        &mut logits,
                        Some(self.tokenizer.as_ref()),
                    )?;
                    let model_cache_update =
                        seq.commit_cached_prefill_physical_resources(cloned_kv, num_tokens);
                    seq.record_generated_token_commit();
                    Ok::<(TokenId, ModelCacheRefUpdate), FerrumError>((token, model_cache_update))
                })();
                let (first_token, model_cache_update) = match prefix_hit_result {
                    Ok(value) => value,
                    Err(error) => {
                        warn!(
                            "Prefix-cache prefill post-process failed for {}: {}",
                            request_id, error
                        );
                        self.complete_request_with_error(request_id, error).await?;
                        return Ok(());
                    }
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
                if self.should_stream_generated_token(request_id, first_token, stop_reason) {
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
                let workspace_lease = self.acquire_legacy_backend_workspace_trace_lease(
                    vec![request_id.clone()],
                    "engine_prefill_workspace",
                    "engine_prefill_workspace_release",
                )?;
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
            let workspace_lease = self.acquire_legacy_backend_workspace_trace_lease(
                vec![request_id.clone()],
                "engine_prefill_workspace",
                "engine_prefill_workspace_release",
            )?;
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
            let token = seq.sample_and_commit_with_processors_and_tokenizer(
                &mut logits,
                Some(self.tokenizer.as_ref()),
            )?;
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
            seq.record_generated_token_commit();
            Ok::<(TokenId, ModelCacheRefUpdate), FerrumError>((token, model_cache_update))
        })();
        let (first_token, model_cache_update) = match first_token_result {
            Ok(value) => value,
            Err(e) => {
                kv_lease.release(self).await;
                recurrent_admission.release_fresh(self).await;
                self.complete_request_with_error(request_id, e).await?;
                return Ok(());
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
        if self.should_stream_generated_token(request_id, first_token, stop_reason) {
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
        use ferrum_interfaces::model_executor::PrefillChunk;

        let request_id = &scheduled.request.id;

        let Some((input_tokens, maximum_sequence_tokens)) =
            self.sequences.read().get(request_id).map(|seq| {
                (
                    seq.prefill_context_tokens(),
                    seq.model_maximum_sequence_tokens(),
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
        let input_token_count = input_tokens.len();
        let input = PlanRuntimePrefillInput::new(
            request_id.clone(),
            input_tokens,
            maximum_sequence_tokens,
            chunk,
        )?;

        match self
            .model_executor
            .plan_runtime_prefill_with_capacity(&input)
            .await?
        {
            PlanRuntimePrefillOutcome::Completed(completion) => {
                return self
                    .commit_plan_runtime_prefill_completion(
                        request_id,
                        input_token_count,
                        chunk,
                        completion,
                    )
                    .await;
            }
            PlanRuntimePrefillOutcome::Deferred(ExecutorExecutionDeferral::RequestState(
                deferral,
            )) => {
                if deferral.request_ids() != std::slice::from_ref(request_id) {
                    return Err(FerrumError::internal(
                        "single prefill Request-state deferral names another frontier",
                    ));
                }
                self.defer_for_request_state_readiness(deferral).await?;
                return Ok(());
            }
            PlanRuntimePrefillOutcome::Deferred(ExecutorExecutionDeferral::Capacity(deferral)) => {
                if let Some(retry) =
                    deferral.validated_maintenance_retry_scope(std::slice::from_ref(request_id))?
                {
                    let receipt = self
                        .scheduler
                        .defer_retry_after_execution_maintenance(retry)?;
                    if receipt.deferred_count() != 1 {
                        return Err(FerrumError::scheduler(format!(
                            "PlanRuntime prefill maintenance retry retained {} scheduler entries for {request_id}",
                            receipt.deferred_count()
                        )));
                    }
                    self.write_scheduler_trace_event(serde_json::json!({
                        "event": "scheduler_prefill_execution_maintenance_retry",
                        "request_id": request_id,
                        "tokens_processed": chunk.tokens_processed(),
                        "tokens_to_process": chunk.tokens_to_process(),
                        "stage": deferral.stage(),
                        "observed": deferral.observed(),
                        "wait_condition": deferral.wait_condition(),
                        "shortfalls": deferral.shortfalls(),
                        "backing_blockers": deferral.backing_blockers(),
                        "typed_evidence": deferral.evidence(),
                        "maintenance_retry": retry,
                        "not_before_iteration": receipt.not_before_iteration(),
                        "latest_capacity_epoch": receipt.latest_capacity_epoch(),
                        "scheduler": self.scheduler.trace_snapshot(),
                    }));
                    return Ok(());
                }
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
                if chunk.tokens_to_process() == 1
                    && !release_snapshot
                        .has_external_releaser(request_id, deferral.wait_condition())
                {
                    self.write_scheduler_trace_event(serde_json::json!({
                        "event": "scheduler_prefill_execution_capacity_impossible",
                        "request_id": request_id,
                        "tokens_processed": chunk.tokens_processed(),
                        "tokens_to_process": chunk.tokens_to_process(),
                        "stage": deferral.stage(),
                        "wait_condition": deferral.wait_condition(),
                        "shortfalls": deferral.shortfalls(),
                        "backing_blockers": deferral.backing_blockers(),
                        "typed_evidence": deferral.evidence(),
                        "reason": "minimum_frontier_has_no_external_releaser",
                        "scheduler": self.scheduler.trace_snapshot(),
                    }));
                    return Err(FerrumError::resource_exhausted(format!(
                        "PlanRuntime prefill for {request_id} has no runnable one-token frontier and no external capacity releaser"
                    )));
                }
                match self.scheduler.defer_prefill_for_execution_capacity(
                    request_id,
                    scheduler_deferral,
                    &release_snapshot,
                )? {
                    ExecutionCapacityAction::Deferred { count: 1 } => {}
                    ExecutionCapacityAction::Deferred { count } => {
                        return Err(FerrumError::scheduler(format!(
                            "PlanRuntime prefill deferral retained {count} scheduler entries for {request_id}"
                        )));
                    }
                    ExecutionCapacityAction::YieldPlanned { transaction } => {
                        let _progress_owner_resumable =
                            self.execute_capacity_yield(&transaction, 1, None).await?;
                        self.write_scheduler_trace_event(serde_json::json!({
                            "event": "scheduler_prefill_execution_capacity_yield_planned",
                            "request_id": request_id,
                            "episode_id": transaction.episode_id().get(),
                            "handoff_generation": transaction.handoff_generation(),
                            "yield_kind": transaction.kind().as_str(),
                            "planned_transition_ordinal": transaction.planned_ordinal().get(),
                            "victim_request_id": transaction.victim_request_id(),
                            "progress_owner_id": transaction.progress_owner_id(),
                            "rotated_from_progress_owner_id": transaction.rotated_from_progress_owner_id(),
                            "rotated_from_progress_baseline": transaction.rotated_from_progress_baseline().map(|generation| generation.get()),
                            "rotated_from_progress_current": transaction.rotated_from_progress_current().map(|generation| generation.get()),
                            "scheduler": self.scheduler.trace_snapshot(),
                        }));
                    }
                    ExecutionCapacityAction::InvariantViolation { violation } => {
                        return Err(FerrumError::internal(format!(
                            "prefill execution-capacity pressure episode {} violated {:?}",
                            violation.episode_id().get(),
                            violation.class()
                        )));
                    }
                }
                self.write_scheduler_trace_event(serde_json::json!({
                    "event": "scheduler_prefill_execution_capacity_defer",
                    "request_id": request_id,
                    "tokens_processed": chunk.tokens_processed(),
                    "tokens_to_process": chunk.tokens_to_process(),
                    "stage": deferral.stage(),
                    "observed": observed,
                    "wait_condition": deferral.wait_condition(),
                    "shortfalls": deferral.shortfalls(),
                    "backing_blockers": deferral.backing_blockers(),
                    "typed_evidence": deferral.evidence(),
                    "scheduler": self.scheduler.trace_snapshot(),
                }));
                return Ok(());
            }
        }
    }

    async fn commit_plan_runtime_prefill_completion(
        &self,
        request_id: &RequestId,
        input_token_count: usize,
        planned_chunk: ferrum_interfaces::model_executor::PrefillChunk,
        completion: ferrum_interfaces::model_executor::PlanRuntimePrefillCompletion,
    ) -> Result<()> {
        if let Err(error) = completion.validate_for(
            request_id,
            planned_chunk,
            self.model_executor.info().vocab_size,
        ) {
            self.discard_plan_runtime_prefill_completion(completion)?;
            return Err(error);
        }
        let (output, executor_planned_chunk, completed_chunk, capacity_probe_count) =
            completion.into_parts();
        debug_assert_eq!(executor_planned_chunk, planned_chunk);
        let chunk = completed_chunk;
        if chunk != planned_chunk {
            self.write_scheduler_trace_event(serde_json::json!({
                "event": "scheduler_prefill_execution_frontier_narrowed",
                "request_id": request_id,
                "tokens_processed": chunk.tokens_processed(),
                "planned_tokens": planned_chunk.tokens_to_process(),
                "completed_tokens": chunk.tokens_to_process(),
                "capacity_probe_count": capacity_probe_count,
                "scheduler": self.scheduler.trace_snapshot(),
            }));
        }
        let (authority, product) = output.into_parts();
        if !chunk.is_final() {
            debug_assert!(matches!(product, PlanRuntimePrefillProduct::Intermediate));
            let model_cache_update = {
                let mut sequences = self.sequences.write();
                sequences.get_mut(request_id).map(|seq| {
                    seq.commit_plan_runtime_prefill_chunk_resources(
                        Arc::clone(authority.kv_cache()),
                        chunk.end(),
                        false,
                    )
                })
            };
            let Some(model_cache_update) = model_cache_update else {
                self.model_executor
                    .discard_plan_runtime_prefill(authority)?;
                return Ok(());
            };
            self.apply_model_cache_ref_update(request_id, model_cache_update);
            let scheduler_result = self
                .scheduler
                .mark_prefill_chunk_processed_with_capacity_feedback(
                    request_id,
                    input_token_count,
                    planned_chunk.tokens_to_process(),
                    chunk.tokens_to_process(),
                );
            let scheduler_error = match scheduler_result {
                Ok(false) => None,
                Ok(true) => Some(FerrumError::scheduler(format!(
                    "non-final PlanRuntime prefill chunk promoted {request_id} to decode"
                ))),
                Err(error) => Some(error),
            };
            if let Some(error) = scheduler_error {
                self.model_executor
                    .discard_plan_runtime_prefill(authority)?;
                self.complete_request_with_error(request_id, error).await?;
                return Ok(());
            }
            self.total_prefill_tokens
                .fetch_add(chunk.tokens_to_process() as u64, Ordering::Relaxed);
            counter!("ferrum.engine.prefill_tokens_total")
                .increment(chunk.tokens_to_process() as u64);
            return Ok(());
        }

        let commit_result = (|| {
            let PlanRuntimePrefillProduct::FinalLogits(mut logits) = product else {
                return Err(FerrumError::internal(
                    "validated final PlanRuntime prefill lost its logits",
                ));
            };
            let mut sequences = self.sequences.write();
            let Some(seq) = sequences.get_mut(request_id) else {
                return Ok(None);
            };
            seq.reset_guided_processors()?;
            let token = seq.sample_and_commit_with_processors_and_tokenizer(
                &mut logits,
                Some(self.tokenizer.as_ref()),
            )?;
            let update = seq.commit_plan_runtime_prefill_chunk_resources(
                Arc::clone(authority.kv_cache()),
                chunk.end(),
                true,
            );
            seq.record_generated_token_commit();
            Ok::<Option<(TokenId, ModelCacheRefUpdate)>, FerrumError>(Some((token, update)))
        })();
        let Some((first_token, model_cache_update)) = (match commit_result {
            Ok(value) => value,
            Err(error) => {
                self.model_executor
                    .discard_plan_runtime_prefill(authority)?;
                self.complete_request_with_error(request_id, error).await?;
                return Ok(());
            }
        }) else {
            self.model_executor
                .discard_plan_runtime_prefill(authority)?;
            return Ok(());
        };

        self.apply_model_cache_ref_update(request_id, model_cache_update);
        let scheduler_result = self
            .scheduler
            .mark_prefill_chunk_processed_with_capacity_feedback(
                request_id,
                input_token_count,
                planned_chunk.tokens_to_process(),
                chunk.tokens_to_process(),
            );
        let scheduler_error = match scheduler_result {
            Ok(true) => None,
            Ok(false) => Some(FerrumError::scheduler(format!(
                "final PlanRuntime prefill chunk did not promote {request_id} to decode"
            ))),
            Err(error) => Some(error),
        };
        if let Some(error) = scheduler_error {
            self.model_executor
                .discard_plan_runtime_prefill(authority)?;
            self.complete_request_with_error(request_id, error).await?;
            return Ok(());
        }
        self.total_prefill_tokens
            .fetch_add(chunk.tokens_to_process() as u64, Ordering::Relaxed);
        counter!("ferrum.engine.prefill_tokens_total").increment(chunk.tokens_to_process() as u64);
        counter!("ferrum.engine.prefills_total").increment(1);

        let stop_reason = self.stop_reason_for_request(request_id);
        if self.should_stream_generated_token(request_id, first_token, stop_reason) {
            self.send_stream_update(request_id, first_token).await;
        }
        if let Some(reason) = stop_reason {
            self.complete_request(request_id, reason).await?;
        }
        Ok(())
    }

    fn discard_plan_runtime_prefill_completion(
        &self,
        completion: ferrum_interfaces::model_executor::PlanRuntimePrefillCompletion,
    ) -> Result<()> {
        let (output, _, _, _) = completion.into_parts();
        let (authority, _) = output.into_parts();
        self.model_executor.discard_plan_runtime_prefill(authority)
    }

    fn discard_plan_runtime_prefill_completions(
        &self,
        completions: impl IntoIterator<
            Item = ferrum_interfaces::model_executor::PlanRuntimePrefillCompletion,
        >,
    ) -> Result<()> {
        let mut failures = Vec::new();
        for completion in completions {
            if let Err(error) = self.discard_plan_runtime_prefill_completion(completion) {
                failures.push(error.to_string());
            }
        }
        if failures.is_empty() {
            Ok(())
        } else {
            Err(FerrumError::internal(format!(
                "failed to discard {} PlanRuntime prefill authorities: {}",
                failures.len(),
                failures.join("; ")
            )))
        }
    }

    /// Returns the exact participants that may fall back to the per-request
    /// path in this scheduler iteration. A maintenance retry removes its bound
    /// affected frontiers from that set after installing their scheduler
    /// ticket, so they cannot be retried in the same iteration.
    pub(super) async fn run_plan_runtime_batch_prefill(
        &self,
        scheduled: &[&ferrum_interfaces::scheduler::ScheduledRequest],
    ) -> Result<PlanRuntimeBatchPrefillDisposition> {
        use ferrum_interfaces::model_executor::PrefillChunk;

        struct Work {
            request_id: RequestId,
            input_token_count: usize,
            planned_chunk: PrefillChunk,
        }

        let mut work = Vec::with_capacity(scheduled.len());
        let mut inputs = Vec::with_capacity(scheduled.len());
        for scheduled in scheduled {
            let request_id = &scheduled.request.id;
            let Some((input_tokens, maximum_sequence_tokens)) =
                self.sequences.read().get(request_id).map(|sequence| {
                    (
                        sequence.prefill_context_tokens(),
                        sequence.model_maximum_sequence_tokens(),
                    )
                })
            else {
                continue;
            };
            let planned_chunk = PrefillChunk::new(
                scheduled.tokens_processed,
                scheduled.tokens_to_process.ok_or_else(|| {
                    FerrumError::scheduler(format!(
                        "PlanRuntime prefill for {request_id} has no scheduled token budget"
                    ))
                })?,
                input_tokens.len(),
            )?;
            let input_token_count = input_tokens.len();
            inputs.push(PlanRuntimePrefillInput::new(
                request_id.clone(),
                input_tokens,
                maximum_sequence_tokens,
                planned_chunk,
            )?);
            work.push(Work {
                request_id: request_id.clone(),
                input_token_count,
                planned_chunk,
            });
        }
        if inputs.len() < 2 {
            return Ok(PlanRuntimeBatchPrefillDisposition::PerRequestFallback(
                work.into_iter().map(|item| item.request_id).collect(),
            ));
        }

        let completions = match self
            .model_executor
            .plan_runtime_batch_prefill_with_capacity(&inputs)
            .await?
        {
            PlanRuntimeBatchPrefillOutcome::Completed(completions) => completions,
            PlanRuntimeBatchPrefillOutcome::NotSubmitted(
                ExecutorExecutionDeferral::RequestState(deferral),
            ) => {
                let request_ids = work
                    .iter()
                    .map(|item| item.request_id.clone())
                    .collect::<Vec<_>>();
                if deferral
                    .request_ids()
                    .iter()
                    .any(|request_id| !request_ids.contains(request_id))
                {
                    return Err(FerrumError::internal(
                        "batch prefill Request-state deferral names another frontier",
                    ));
                }
                let fallback_request_ids =
                    unaffected_maintenance_retry_frontiers(&request_ids, deferral.request_ids());
                self.defer_for_request_state_readiness(deferral).await?;
                return Ok(PlanRuntimeBatchPrefillDisposition::PerRequestFallback(
                    fallback_request_ids,
                ));
            }
            PlanRuntimeBatchPrefillOutcome::NotSubmitted(ExecutorExecutionDeferral::Capacity(
                deferral,
            )) => {
                let request_ids = work
                    .iter()
                    .map(|item| item.request_id.clone())
                    .collect::<Vec<_>>();
                if let Some(retry) = deferral.validated_maintenance_retry_scope(&request_ids)? {
                    let affected_request_ids = retry.affected_request_ids();
                    let fallback_request_ids =
                        unaffected_maintenance_retry_frontiers(&request_ids, affected_request_ids);
                    let receipt = self
                        .scheduler
                        .defer_retry_after_execution_maintenance(retry)?;
                    if receipt.deferred_count() != affected_request_ids.len() {
                        return Err(FerrumError::scheduler(format!(
                            "PlanRuntime batch prefill maintenance retry retained {} of {} scheduler entries",
                            receipt.deferred_count(),
                            affected_request_ids.len()
                        )));
                    }
                    self.write_scheduler_trace_event(serde_json::json!({
                        "event": "scheduler_batch_prefill_execution_maintenance_retry",
                        "request_ids": affected_request_ids,
                        "input_cohort_request_ids": request_ids,
                        "stage": deferral.stage(),
                        "observed": deferral.observed(),
                        "wait_condition": deferral.wait_condition(),
                        "shortfalls": deferral.shortfalls(),
                        "backing_blockers": deferral.backing_blockers(),
                        "typed_evidence": deferral.evidence(),
                        "maintenance_retry": retry,
                        "not_before_iteration": receipt.not_before_iteration(),
                        "latest_capacity_epoch": receipt.latest_capacity_epoch(),
                        "scheduler": self.scheduler.trace_snapshot(),
                    }));
                    return Ok(PlanRuntimeBatchPrefillDisposition::PerRequestFallback(
                        fallback_request_ids,
                    ));
                }
                self.write_scheduler_trace_event(serde_json::json!({
                    "event": "scheduler_batch_prefill_execution_capacity_defer",
                    "request_ids": request_ids,
                    "stage": deferral.stage(),
                    "observed": deferral.observed(),
                    "wait_condition": deferral.wait_condition(),
                    "shortfalls": deferral.shortfalls(),
                    "backing_blockers": deferral.backing_blockers(),
                    "typed_evidence": deferral.evidence(),
                    "scheduler": self.scheduler.trace_snapshot(),
                }));
                return Ok(PlanRuntimeBatchPrefillDisposition::PerRequestFallback(
                    work.into_iter().map(|item| item.request_id).collect(),
                ));
            }
            PlanRuntimeBatchPrefillOutcome::Unsupported => {
                return Ok(PlanRuntimeBatchPrefillDisposition::PerRequestFallback(
                    work.into_iter().map(|item| item.request_id).collect(),
                ));
            }
        };
        if completions.len() != work.len() {
            let completion_count = completions.len();
            let cleanup_result = self.discard_plan_runtime_prefill_completions(completions);
            for item in &work {
                self.model_executor
                    .cancel_prefill_admission(&item.request_id);
            }
            let error = FerrumError::internal(format!(
                "PlanRuntime batch prefill returned a different participant count: expected {}, got {}",
                work.len(),
                completion_count
            ));
            if let Err(cleanup_error) = cleanup_result {
                return Err(FerrumError::internal(format!(
                    "{error}; exact-authority cleanup also failed: {cleanup_error}"
                )));
            }
            return Err(error);
        }
        let validation = work
            .iter()
            .zip(&completions)
            .try_for_each(|(item, completion)| {
                completion.validate_for(
                    &item.request_id,
                    item.planned_chunk,
                    self.model_executor.info().vocab_size,
                )
            });
        if let Err(error) = validation {
            if let Err(cleanup_error) = self.discard_plan_runtime_prefill_completions(completions) {
                return Err(FerrumError::internal(format!(
                    "{error}; exact-authority cleanup also failed: {cleanup_error}"
                )));
            }
            return Err(error);
        }

        let mut completions = completions.into_iter();
        for item in work {
            let completion = completions.next().ok_or_else(|| {
                FerrumError::internal("PlanRuntime batch prefill completion disappeared")
            })?;
            if let Err(error) = self
                .commit_plan_runtime_prefill_completion(
                    &item.request_id,
                    item.input_token_count,
                    item.planned_chunk,
                    completion,
                )
                .await
            {
                if let Err(cleanup_error) =
                    self.discard_plan_runtime_prefill_completions(completions)
                {
                    return Err(FerrumError::internal(format!(
                        "{error}; remaining exact-authority cleanup also failed: {cleanup_error}"
                    )));
                }
                return Err(error);
            }
        }
        Ok(PlanRuntimeBatchPrefillDisposition::Completed)
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
                    let prefix_hit_result = (|| {
                        let cloned_kv = cached_kv.clone_handle()?;
                        let mut sequences = self.sequences.write();
                        let Some(seq) = sequences.get_mut(rid) else {
                            return Ok(None);
                        };
                        seq.reset_guided_processors()?;
                        let mut logits = cached_logits;
                        let token = seq.sample_and_commit_with_processors_and_tokenizer(
                            &mut logits,
                            Some(self.tokenizer.as_ref()),
                        )?;
                        let model_cache_update =
                            seq.commit_cached_prefill_physical_resources(cloned_kv, num_tokens);
                        seq.record_generated_token_commit();
                        Ok::<Option<(TokenId, ModelCacheRefUpdate)>, FerrumError>(Some((
                            token,
                            model_cache_update,
                        )))
                    })();
                    let prefix_hit = match prefix_hit_result {
                        Ok(value) => value,
                        Err(error) => {
                            warn!(
                                "Batch prefix-cache prefill post-process failed for {}: {}",
                                rid, error
                            );
                            self.complete_request_with_error(rid, error).await?;
                            continue;
                        }
                    };
                    let Some((first_token, model_cache_update)) = prefix_hit else {
                        continue;
                    };
                    self.apply_model_cache_ref_update(rid, model_cache_update);
                    self.scheduler.mark_prefill_complete(rid, num_tokens);
                    self.prefix_cache_hits.fetch_add(1, Ordering::Relaxed);
                    counter!("ferrum.engine.prefix_cache_hits").increment(1);
                    let stop_reason = self.stop_reason_for_request(rid);
                    if self.should_stream_generated_token(rid, first_token, stop_reason) {
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

        let workspace_lease = self.acquire_legacy_backend_workspace_trace_lease(
            workspace_request_ids,
            "engine_batch_prefill_workspace",
            "engine_batch_prefill_workspace_release",
        )?;
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
                let token = seq.sample_and_commit_with_processors_and_tokenizer(
                    &mut logits,
                    Some(self.tokenizer.as_ref()),
                )?;
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
                seq.record_generated_token_commit();
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
                    self.complete_request_with_error(&rid, e).await?;
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
            if self.should_stream_generated_token(&rid, first_token, stop_reason) {
                self.send_stream_update(&rid, first_token).await;
            }
            if let Some(reason) = stop_reason {
                self.complete_request(&rid, reason).await?;
            }
        }
        Ok(())
    }
}
