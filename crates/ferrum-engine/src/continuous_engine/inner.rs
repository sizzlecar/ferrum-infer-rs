//! EngineInner iteration and request-processing implementation.

use super::*;

mod batch;
mod completion;
mod decode;
mod prefill;

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

    // ── iteration loop ─────────────────────────────────────────────────

    /// Run one iteration: ask the scheduler for a batch, then process it.
    pub(super) async fn run_iteration(&self) -> Result<()> {
        let iteration = self.iteration_count.fetch_add(1, Ordering::Relaxed);
        counter!("ferrum.engine.iterations_total").increment(1);
        let prof = self.runtime_config.batch_decode_prof;
        let t_iter_start = if prof { Some(Instant::now()) } else { None };

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

        // FERRUM_NEXT_BATCH_PROF=1: count Some/None returns to root-cause
        // the apples HTTP-serve 17 ms inter-batch-iter gap. Prints every
        // 1024 run_iteration calls. Set None_size to 0 explicitly so the
        // bg-loop tight-spin theory can be confirmed.
        let nb_prof = self.runtime_config.next_batch_prof;
        let nb_t0 = if nb_prof { Some(Instant::now()) } else { None };
        let nb_result = self.scheduler.next_batch(hint).await;
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
                tokio::time::sleep(Duration::from_millis(1)).await;
                return Ok(());
            }
        };
        let t_after_sched = if prof { Some(Instant::now()) } else { None };

        debug!(
            "Iteration {}: batch with {} requests",
            iteration,
            batch.size()
        );

        let r = self.process_batch(&batch).await;
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
