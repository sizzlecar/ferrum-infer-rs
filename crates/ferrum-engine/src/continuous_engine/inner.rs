//! EngineInner iteration and request-processing implementation.

use super::*;

mod batch;
mod completion;
mod decode;
mod prefill;

pub(super) fn is_resource_exhausted_error(error: &FerrumError) -> bool {
    matches!(error, FerrumError::ResourceExhausted { .. })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct PagedKvAdmissionPressure {
    pub(super) admission_blocks: usize,
    pub(super) immediate_blocks: usize,
    pub(super) free_blocks: usize,
}

pub(super) fn paged_kv_admission_pressure(error: &FerrumError) -> Option<PagedKvAdmissionPressure> {
    let FerrumError::ResourceExhausted { message } = error else {
        return None;
    };
    let (_, after_need) = message.split_once("paged KV admission: need ")?;
    let (admission_blocks, rest) = parse_usize_prefix(after_need)?;
    let rest = rest.strip_prefix(" admission blocks (")?;
    let (immediate_blocks, rest) = parse_usize_prefix(rest)?;
    let rest = rest.strip_prefix(" immediate) but only ")?;
    let (free_blocks, rest) = parse_usize_prefix(rest)?;
    if rest != " free" {
        return None;
    }
    Some(PagedKvAdmissionPressure {
        admission_blocks,
        immediate_blocks,
        free_blocks,
    })
}

fn parse_usize_prefix(input: &str) -> Option<(usize, &str)> {
    let end = input
        .char_indices()
        .find_map(|(index, ch)| (!ch.is_ascii_digit()).then_some(index))
        .unwrap_or(input.len());
    if end == 0 {
        return None;
    }
    let value = input[..end].parse().ok()?;
    Some((value, &input[end..]))
}

pub(super) fn kv_slot_requests_for_unified_batch(
    batch: &ferrum_interfaces::model_executor::UnifiedBatch,
) -> Vec<KvSlotRequest> {
    batch
        .items
        .iter()
        .map(|item| {
            let target_len = item.pos_offset.saturating_add(item.q_tokens.len());
            KvSlotRequest {
                cache_id: item.seq_id.clone(),
                target_len,
                admission_target_len: metadata_kv_admission_target_len(&item.metadata)
                    .map(|len| len.max(target_len)),
            }
        })
        .collect()
}

pub(super) fn metadata_kv_admission_target_len(
    metadata: &HashMap<String, serde_json::Value>,
) -> Option<usize> {
    metadata
        .get(KV_ADMISSION_TARGET_LEN_METADATA_KEY)
        .and_then(|value| value.as_u64())
        .map(|value| value as usize)
        .filter(|&value| value > 0)
}

#[derive(Debug, Default, serde::Serialize)]
pub(super) struct SchedulerTraceDistribution {
    pub(super) count: usize,
    pub(super) min: Option<usize>,
    pub(super) p50: Option<usize>,
    pub(super) max: Option<usize>,
}

#[derive(Debug, Default, serde::Serialize)]
pub(super) struct SchedulerTracePlanStats {
    pub(super) batch_size: usize,
    pub(super) prefill_items: usize,
    pub(super) decode_items: usize,
    pub(super) waiting_items: usize,
    pub(super) preempted_items: usize,
    pub(super) unknown_items: usize,
    pub(super) scheduled_tokens_total: usize,
    pub(super) prefill_tokens: usize,
    pub(super) decode_tokens: usize,
    pub(super) tokens_to_process_missing: usize,
    pub(super) decode_generated_tokens: SchedulerTraceDistribution,
    pub(super) prefill_prompt_tokens: SchedulerTraceDistribution,
    pub(super) requests: Vec<SchedulerTraceRequestStats>,
}

#[derive(Debug, serde::Serialize)]
pub(super) struct SchedulerTraceRequestStats {
    pub(super) request_id: String,
    pub(super) phase: Option<String>,
    pub(super) scheduled_tokens: usize,
    pub(super) tokens_to_process_missing: bool,
    pub(super) prompt_tokens: Option<usize>,
    pub(super) generated_tokens: Option<usize>,
    pub(super) prefill_tokens_processed: Option<usize>,
    pub(super) prefill_tokens_remaining_before: Option<usize>,
    pub(super) is_final_prefill_chunk: Option<bool>,
}

fn scheduler_trace_distribution(mut values: Vec<usize>) -> SchedulerTraceDistribution {
    if values.is_empty() {
        return SchedulerTraceDistribution::default();
    }
    values.sort_unstable();
    SchedulerTraceDistribution {
        count: values.len(),
        min: values.first().copied(),
        p50: values.get(values.len() / 2).copied(),
        max: values.last().copied(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn paged_kv_admission_pressure_parses_qwen35_resource_error() {
        let error = FerrumError::resource_exhausted(
            "Qwen3.5 paged KV admission: need 24 admission blocks (4 immediate) but only 3 free",
        );

        assert_eq!(
            paged_kv_admission_pressure(&error),
            Some(PagedKvAdmissionPressure {
                admission_blocks: 24,
                immediate_blocks: 4,
                free_blocks: 3,
            })
        );
    }

    #[test]
    fn paged_kv_admission_pressure_parses_generic_llama_resource_error() {
        let error = FerrumError::resource_exhausted(
            "paged KV admission: need 1 admission blocks (1 immediate) but only 0 free",
        );

        assert_eq!(
            paged_kv_admission_pressure(&error),
            Some(PagedKvAdmissionPressure {
                admission_blocks: 1,
                immediate_blocks: 1,
                free_blocks: 0,
            })
        );
    }

    #[test]
    fn paged_kv_admission_pressure_ignores_unrelated_resource_errors() {
        let error = FerrumError::resource_exhausted("synthetic unified reserve failure");

        assert_eq!(paged_kv_admission_pressure(&error), None);
    }
}

impl EngineInner {
    // ── tensor helper ──────────────────────────────────────────────────

    pub(super) fn tokens_to_tensor(&self, token_ids: &[u32]) -> Result<TensorRef> {
        let f32_data: Vec<f32> = token_ids.iter().map(|&v| v as f32).collect();
        let len = f32_data.len();
        self.tensor_factory
            .from_slice(&f32_data, &[1, len], DataType::FP32, Device::CPU)
    }

    /// Rebuild a KvCacheHandle with a corrected sequence_length.
    ///
    /// Only meaningful for `GenericKvCacheHandle`, which is what the LLM
    /// executor (`LlmExecutor::prefill` / `decode`) constructs and threads
    /// through speculative decoding. Resource handles minted by
    /// `KvCacheManager` impls (Paged / Default) are returned as a plain
    /// clone — those handles don't track per-iter position (the model's
    /// internal paged_pool does), and the engine no longer reads
    /// `sequence_length` from them for position purposes (see
    /// `process_batch_unified` for the SequenceState-sourced pos_offset).
    pub(super) fn make_kv_handle_with_seq(
        &self,
        h: &std::sync::Arc<dyn ferrum_interfaces::KvCacheHandle>,
        new_seq: usize,
    ) -> std::sync::Arc<dyn ferrum_interfaces::KvCacheHandle> {
        if let Some(g) = h
            .as_any()
            .downcast_ref::<ferrum_models::executor::common::GenericKvCacheHandle>()
        {
            std::sync::Arc::new(g.with_sequence_length(new_seq))
        } else {
            h.clone()
        }
    }

    /// Build the model-executor KV handle used by `LlmExecutor::decode`.
    ///
    /// The continuous engine has two KV identities:
    /// - `KvCacheManager` allocations, keyed by request id, track resource
    ///   lifetime and are deallocated on preemption/completion.
    /// - model/executor cache ids track the actual model-side KV contents.
    ///
    /// `SequenceState` model-KV state must carry the second identity because the
    /// fallback single-request decode path downcasts it to
    /// `GenericKvCacheHandle`. Unified CUDA paths don't read the handle body,
    /// but keeping this invariant prevents resource-pressure fallbacks from
    /// feeding a manager handle into `LlmExecutor::decode`.
    pub(super) fn make_model_kv_handle_with_seq(
        &self,
        cache_id: String,
        seq_len: usize,
    ) -> std::sync::Arc<dyn ferrum_interfaces::KvCacheHandle> {
        let info = self.model_executor.info();
        let head_dim = if info.num_heads == 0 {
            info.hidden_size.max(1)
        } else {
            (info.hidden_size / info.num_heads).max(1)
        };
        let num_kv_heads = if info.num_kv_heads == 0 {
            info.num_heads.max(1)
        } else {
            info.num_kv_heads
        };
        std::sync::Arc::new(ferrum_models::executor::common::GenericKvCacheHandle::new(
            info.num_layers,
            num_kv_heads,
            head_dim,
            candle_core::Device::Cpu,
            seq_len,
            cache_id,
        ))
    }

    pub(super) fn decode_ready_request_ids(&self, request_ids: &[RequestId]) -> Vec<RequestId> {
        let sequences = self.sequences.read();
        request_ids
            .iter()
            .filter(|rid| {
                sequences.get(*rid).is_some_and(|seq| {
                    seq.prefill_complete
                        && seq.kv_cache_handle().is_some()
                        && !seq.generated_tokens.is_empty()
                })
            })
            .cloned()
            .collect()
    }

    fn scheduler_trace_enabled(&self) -> bool {
        self.legacy_scheduler_trace_jsonl.is_some()
    }

    pub(super) fn scheduler_trace_plan_stats(
        &self,
        batch: &ferrum_interfaces::BatchPlan,
    ) -> SchedulerTracePlanStats {
        let mut stats = SchedulerTracePlanStats {
            batch_size: batch.size(),
            ..SchedulerTracePlanStats::default()
        };
        let mut decode_generated_tokens = Vec::new();
        let mut prefill_prompt_tokens = Vec::new();
        let sequences = self.sequences.read();

        for scheduled_req in &batch.requests {
            let request_id = &scheduled_req.request.id;
            let scheduled_tokens = scheduled_req.tokens_to_process.unwrap_or(0);
            if scheduled_req.tokens_to_process.is_none() {
                stats.tokens_to_process_missing += 1;
            }
            stats.scheduled_tokens_total += scheduled_tokens;

            let phase = self.scheduler.trace_phase(request_id);
            let seq = sequences.get(request_id);
            let prompt_tokens = seq.map(|seq| seq.prefill_context_len());
            let generated_tokens = seq.map(|seq| seq.generated_tokens.len());
            let prefill_tokens_processed = seq.map(|seq| seq.prefill_tokens_processed);
            let prefill_tokens_remaining_before = prompt_tokens
                .zip(prefill_tokens_processed)
                .map(|(prompt, processed)| prompt.saturating_sub(processed));
            let is_final_prefill_chunk = match (phase, prefill_tokens_remaining_before) {
                (Some(RequestPhase::Prefilling), Some(remaining)) => {
                    Some(scheduled_tokens >= remaining)
                }
                _ => None,
            };

            match phase {
                Some(RequestPhase::Decoding) => {
                    stats.decode_items += 1;
                    stats.decode_tokens += scheduled_tokens;
                    if let Some(generated_tokens) = generated_tokens {
                        decode_generated_tokens.push(generated_tokens);
                    }
                }
                Some(RequestPhase::Prefilling) => {
                    stats.prefill_items += 1;
                    stats.prefill_tokens += scheduled_tokens;
                    if let Some(prompt_tokens) = prompt_tokens {
                        prefill_prompt_tokens.push(prompt_tokens);
                    }
                }
                Some(RequestPhase::Waiting) => {
                    stats.waiting_items += 1;
                    stats.prefill_tokens += scheduled_tokens;
                }
                Some(RequestPhase::Preempted) => {
                    stats.preempted_items += 1;
                }
                Some(RequestPhase::Completed | RequestPhase::Cancelled) | None => {
                    stats.unknown_items += 1;
                }
            }

            stats.requests.push(SchedulerTraceRequestStats {
                request_id: request_id.to_string(),
                phase: phase.map(|phase| format!("{phase:?}")),
                scheduled_tokens,
                tokens_to_process_missing: scheduled_req.tokens_to_process.is_none(),
                prompt_tokens,
                generated_tokens,
                prefill_tokens_processed,
                prefill_tokens_remaining_before,
                is_final_prefill_chunk,
            });
        }

        stats.decode_generated_tokens = scheduler_trace_distribution(decode_generated_tokens);
        stats.prefill_prompt_tokens = scheduler_trace_distribution(prefill_prompt_tokens);
        stats
    }

    fn write_scheduler_trace_event(&self, event: serde_json::Value) {
        let Some(file) = &self.legacy_scheduler_trace_jsonl else {
            return;
        };
        let mut file = file.lock();
        if let Err(error) = serde_json::to_writer(&mut *file, &event) {
            warn!("Failed to write scheduler trace event: {}", error);
            return;
        }
        if let Err(error) = file.write_all(b"\n") {
            warn!("Failed to terminate scheduler trace event: {}", error);
        }
    }

    // ── iteration loop ─────────────────────────────────────────────────

    /// Run one iteration: ask the scheduler for a batch, then process it.
    pub(super) async fn run_iteration(&self) -> Result<()> {
        self.cancel_abandoned_requests().await?;

        let iteration = self.iteration_count.fetch_add(1, Ordering::Relaxed);
        counter!("ferrum.engine.iterations_total").increment(1);
        let prof = self.runtime_config.batch_decode_prof;
        let t_iter_start = if prof { Some(Instant::now()) } else { None };
        let trace_enabled = self.scheduler_trace_enabled();
        let trace_scheduler_before = trace_enabled.then(|| self.scheduler.trace_snapshot());
        let trace_prefill_tokens_before = if trace_enabled {
            Some(self.total_prefill_tokens.load(Ordering::Relaxed))
        } else {
            None
        };
        let trace_decode_tokens_before = if trace_enabled {
            Some(self.total_decode_tokens.load(Ordering::Relaxed))
        } else {
            None
        };

        // Phase 3 token-budget hint: scheduler emits a mixed batch
        // summing to at most `max_num_batched_tokens` Q tokens. This
        // replaces the prior `max_batch_size * 2048` heuristic which
        // never bit and left scheduler-side prefill admission capped
        // at `max_prefill_batch=8`. Defaults to 4096 (autosizer can
        // override via `FERRUM_MAX_BATCHED_TOKENS`).
        let hint = ferrum_interfaces::BatchHint {
            max_batch_size: self.config.batching.max_batch_size,
            max_tokens: self.config.batching.max_num_batched_tokens,
            target_latency_ms: None,
            available_memory: None,
            resource_constraints: ferrum_interfaces::scheduler::ResourceConstraints::default(),
        };
        let hint_max_batch_size = hint.max_batch_size;
        let hint_max_tokens = hint.max_tokens;

        // FERRUM_NEXT_BATCH_PROF=1: count Some/None returns to root-cause
        // the apples HTTP-serve 17 ms inter-batch-iter gap. Prints every
        // 1024 run_iteration calls. Set None_size to 0 explicitly so the
        // bg-loop tight-spin theory can be confirmed.
        let nb_prof = self.runtime_config.next_batch_prof;
        let sched_t0 = Instant::now();
        let nb_t0 = if nb_prof { Some(Instant::now()) } else { None };
        let nb_result = self.scheduler.next_batch(hint).await;
        let sched_elapsed = sched_t0.elapsed();
        self.record_scheduling_time(sched_elapsed);
        if let Some(t0) = nb_t0 {
            use std::sync::atomic::AtomicU64;
            static SOME_N: AtomicU64 = AtomicU64::new(0);
            static NONE_N: AtomicU64 = AtomicU64::new(0);
            static SOME_US: AtomicU64 = AtomicU64::new(0);
            static NONE_US: AtomicU64 = AtomicU64::new(0);
            let us = t0.elapsed().as_micros() as u64;
            let is_some = nb_result.is_some();
            let batch_size = nb_result.as_ref().map_or(0, |b| b.size());
            if is_some {
                SOME_N.fetch_add(1, Ordering::Relaxed);
                SOME_US.fetch_add(us, Ordering::Relaxed);
            } else {
                NONE_N.fetch_add(1, Ordering::Relaxed);
                NONE_US.fetch_add(us, Ordering::Relaxed);
            }
            let total = SOME_N.load(Ordering::Relaxed) + NONE_N.load(Ordering::Relaxed);
            if total.is_multiple_of(1024) {
                let s_n = SOME_N.load(Ordering::Relaxed);
                let n_n = NONE_N.load(Ordering::Relaxed);
                let s_us = SOME_US.load(Ordering::Relaxed);
                let n_us = NONE_US.load(Ordering::Relaxed);
                eprintln!(
                    "[nb-prof] total={} some={} none={} ratio={:.3} | some_avg={}us none_avg={}us last_batch_size={} last_was_some={}",
                    total,
                    s_n,
                    n_n,
                    s_n as f64 / total as f64,
                    if s_n > 0 { s_us / s_n } else { 0 },
                    if n_n > 0 { n_us / n_n } else { 0 },
                    batch_size,
                    is_some,
                );
            }
        }

        let batch = match nb_result {
            Some(b) => b,
            None => {
                if trace_enabled {
                    let none_streak = self
                        .scheduler_trace_none_streak
                        .fetch_add(1, Ordering::Relaxed)
                        + 1;
                    if none_streak <= 4 || none_streak.is_multiple_of(128) {
                        self.write_scheduler_trace_event(serde_json::json!({
                            "event": "scheduler_iteration",
                            "iteration": iteration,
                            "result": "none",
                            "none_streak": none_streak,
                            "hint": {
                                "max_batch_size": hint_max_batch_size,
                                "max_tokens": hint_max_tokens,
                            },
                            "scheduler_before": trace_scheduler_before.as_ref(),
                            "scheduler_after_schedule": self.scheduler.trace_snapshot(),
                            "timing_us": {
                                "schedule": duration_to_us(sched_elapsed),
                            },
                        }));
                    }
                }
                tokio::time::sleep(Duration::from_millis(1)).await;
                return Ok(());
            }
        };
        let trace_none_since_last_some = if trace_enabled {
            Some(self.scheduler_trace_none_streak.swap(0, Ordering::Relaxed))
        } else {
            None
        };
        let trace_scheduler_after_schedule = trace_enabled.then(|| self.scheduler.trace_snapshot());
        let trace_plan = trace_enabled.then(|| self.scheduler_trace_plan_stats(&batch));
        let t_after_sched = if prof { Some(Instant::now()) } else { None };

        debug!(
            "Iteration {}: batch with {} requests",
            iteration,
            batch.size()
        );

        let process_t0 = Instant::now();
        let r = self.process_batch(&batch).await;
        let process_elapsed = process_t0.elapsed();
        self.record_model_execution_time(process_elapsed);
        if trace_enabled {
            let prefill_tokens_after = self.total_prefill_tokens.load(Ordering::Relaxed);
            let decode_tokens_after = self.total_decode_tokens.load(Ordering::Relaxed);
            self.write_scheduler_trace_event(serde_json::json!({
                "event": "scheduler_iteration",
                "iteration": iteration,
                "result": if r.is_ok() { "some_ok" } else { "some_error" },
                "error": r.as_ref().err().map(|error| error.to_string()),
                "none_since_last_some": trace_none_since_last_some,
                "hint": {
                    "max_batch_size": hint_max_batch_size,
                    "max_tokens": hint_max_tokens,
                },
                "scheduler_before": trace_scheduler_before.as_ref(),
                "scheduler_after_schedule": trace_scheduler_after_schedule.as_ref(),
                "scheduler_after_process": self.scheduler.trace_snapshot(),
                "plan": trace_plan.as_ref(),
                "engine_counters": {
                    "prefill_tokens_before": trace_prefill_tokens_before,
                    "prefill_tokens_after": prefill_tokens_after,
                    "prefill_tokens_delta": prefill_tokens_after
                        .saturating_sub(trace_prefill_tokens_before.unwrap_or(prefill_tokens_after)),
                    "decode_tokens_before": trace_decode_tokens_before,
                    "decode_tokens_after": decode_tokens_after,
                    "decode_tokens_delta": decode_tokens_after
                        .saturating_sub(trace_decode_tokens_before.unwrap_or(decode_tokens_after)),
                },
                "timing_us": {
                    "schedule": duration_to_us(sched_elapsed),
                    "process": duration_to_us(process_elapsed),
                    "total_since_schedule_start": duration_to_us(sched_t0.elapsed()),
                },
            }));
        }
        if let (Some(t0), Some(ts)) = (t_iter_start, t_after_sched) {
            let n = self.iteration_count.load(Ordering::Relaxed);
            if n < 64 || n.is_multiple_of(32) {
                let total = t0.elapsed().as_micros();
                let sched = ts.duration_since(t0).as_micros();
                let proc = ts.elapsed().as_micros();
                eprintln!(
                    "[iter-prof] iter#{} total={}us sched={}us process={}us batch_size={}",
                    iteration,
                    total,
                    sched,
                    proc,
                    batch.size()
                );
                let profile = global_profile();
                if profile.is_enabled() {
                    let _ = profile.push_event(
                        "iter_prof",
                        profile_fields_from_json(serde_json::json!({
                            "iter": iteration,
                            "batch_size": batch.size(),
                        })),
                        profile_fields_from_json(serde_json::json!({
                            "total": total,
                            "sched": sched,
                            "process": proc,
                        })),
                        false,
                    );
                }
            }
        }
        r
    }
}
