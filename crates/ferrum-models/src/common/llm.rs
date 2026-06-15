//! `DecoderOnlyLLM` trait — the "model family" interface that every
//! decoder-only language model (Qwen3 / Llama / Mistral / DeepSeek / ...)
//! implements, independent of backend and weight format.
//!
//! `LlmExecutor` (living in `ferrum-engine`) holds a `Box<dyn DecoderOnlyLLM>`
//! and adapts it to the `ModelExecutor` trait that the scheduler calls.

use ferrum_interfaces::model_executor::LogitsReturnPolicy;

/// Runtime configuration every decoder-only LLM must expose.
///
/// This is the *execution-facing* config — the bare minimum the surrounding
/// engine needs (KV cache sizing, sampler vocab bounds, scheduler quotas).
/// It deliberately does not include architecture details like `num_heads`
/// or `intermediate_size`; those stay private to the model implementation.
#[derive(Clone, Debug)]
pub struct LlmRuntimeConfig {
    pub hidden_size: usize,
    pub num_layers: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub max_seq_len: usize,
}

/// A decoder-only language model.
///
/// Contract:
/// - `prefill` processes a batch of prompt tokens and returns logits for the
///   *last* token, along with initializing whatever KV cache the model
///   maintains internally (keyed by `cache_id`).
/// - `decode` processes a single generated token at position `pos` and
///   returns logits for the next step.
/// - `release` frees the KV cache for a completed sequence.
///
/// Today the model owns its KV cache. Integration with `ferrum-kv`'s paged
/// KV manager is a Phase D concern; the trait is kept minimal so it can
/// evolve then without a full refactor.
pub trait DecoderOnlyLLM: Send + Sync {
    /// Runtime-facing configuration.
    fn config(&self) -> &LlmRuntimeConfig;

    /// Optional model-level cache metrics.
    ///
    /// Models with real paged-KV prefix reuse override this so the executor
    /// and HTTP server can distinguish true KV reuse from product-level
    /// prompt observability.
    fn cache_metrics_snapshot(&self) -> Option<serde_json::Value> {
        None
    }

    /// Optional runtime LoRA metrics.
    fn lora_metrics_snapshot(&self) -> Option<serde_json::Value> {
        None
    }

    /// Bind or clear a startup LoRA adapter for a model-side KV cache id.
    ///
    /// The executor calls this before prefill/decode based on request
    /// metadata. Models that implement real LoRA inference override it and
    /// keep the adapter scoped to `cache_id`; unsupported models return an
    /// explicit error instead of silently serving the base model.
    fn set_lora_adapter_for_cache(
        &mut self,
        cache_id: &str,
        adapter: Option<crate::lora::ActiveLoraAdapter>,
    ) -> std::result::Result<(), ferrum_types::FerrumError> {
        let _ = cache_id;
        if let Some(adapter) = adapter {
            return Err(ferrum_types::FerrumError::unsupported(format!(
                "LoRA inference is not supported by this model/backend for adapter {} at {}",
                adapter.name,
                adapter.path.display()
            )));
        }
        Ok(())
    }

    /// Hint that an upcoming `prefill` / `decode` sequence on
    /// `cache_id` will have at most `max_tokens` tokens per call. Lets
    /// the model eagerly grow its internal scratch buffers AND allocate
    /// the KV cache for `cache_id` so the first real `prefill` doesn't
    /// have to allocate them on the hot path.
    ///
    /// Without this, on Qwen3-MoE's first prefill the timer captures:
    ///   • ~25 scratch MTLBuffers (residual / qkv / head-major / MoE
    ///     staging / batch-logits) — ~80-150 ms total alloc
    ///   • ~96 KV-cache MTLBuffers (K and V × 48 layers) — another
    ///     ~100-500 ms total alloc
    ///
    /// Combined that's the ~350 ms fixed overhead that made pp50 numbers
    /// look 40% slower than pp512 for the same per-token compute.
    ///
    /// Default no-op — backends without resizable buffers ignore it.
    fn prepare(&mut self, cache_id: &str, max_tokens: usize) {
        let _ = (cache_id, max_tokens);
    }

    /// Hint that `cache_id` will need at most `capacity_hint` KV positions
    /// for the whole request. This is separate from [`prepare`], whose
    /// `max_tokens` parameter sizes per-call scratch buffers.
    fn prepare_kv_capacity(&mut self, cache_id: &str, capacity_hint: usize) {
        let _ = (cache_id, capacity_hint);
    }

    /// Per-cache KV capacity in tokens — the maximum sequence length any
    /// single `cache_id` can grow to before `prefill` / `decode` would
    /// overflow the pre-allocated K/V buffers.
    ///
    /// Honours `FERRUM_KV_CAPACITY` and clamps to the model's declared
    /// `max_seq_len`. Callers (REPL, HTTP server, schedulers) should
    /// pre-check this before extending a sequence; the model panics on
    /// append-side overflow rather than silently corrupt the cache.
    ///
    /// Default returns `config().max_seq_len`. Models that allocate a
    /// smaller window (most do, capped by `FERRUM_KV_CAPACITY` or the
    /// 4096 default in `ensure_kv`) override this to surface the real
    /// budget.
    fn kv_capacity(&self) -> usize {
        self.config().max_seq_len
    }

    /// Prefill the model with a prompt. Returns `[vocab_size]` logits for
    /// the last prompt token.
    fn prefill(&mut self, cache_id: &str, tokens: &[u32]) -> Vec<f32>;

    /// Advance the model by one generated token. `pos` is the position of
    /// `token` in the sequence (number of tokens already consumed so far).
    /// Returns `[vocab_size]` logits for the next step.
    fn decode(&mut self, cache_id: &str, token: u32, pos: u32) -> Vec<f32>;

    /// Decode multiple concurrent requests in a single forward pass.
    ///
    /// Each entry is `(cache_id, token, pos)` — per-request state. Returns
    /// one `[vocab_size]` logits vec per request in the SAME order.
    ///
    /// Default implementation loops `decode` sequentially. Backends that
    /// implement true batched decode (one GEMM with m=batch, per-item
    /// attention loop) override for concurrency speedup.
    fn decode_batch(&mut self, batch: &[(String, u32, u32)]) -> Vec<Vec<f32>> {
        batch
            .iter()
            .map(|(cid, tok, p)| self.decode(cid, *tok, *p))
            .collect()
    }

    fn decode_batch_with_full_logits(
        &mut self,
        batch: &[(String, u32, u32)],
        _force_full_logits: bool,
    ) -> Vec<Vec<f32>> {
        self.decode_batch(batch)
    }

    fn decode_batch_with_logits_policy(
        &mut self,
        batch: &[(String, u32, u32)],
        _policies: &[LogitsReturnPolicy],
    ) -> Vec<Vec<f32>> {
        self.decode_batch_with_full_logits(batch, true)
    }

    /// Multi-position decode-verify: run a single forward over `tokens`
    /// starting at the current KV end, append their K/V in place, and
    /// return `seq_len * vocab_size` logits (row-major, position-first).
    ///
    /// Used by speculative decoding to collect N+1 verification logits
    /// in one target pass instead of N+1 sequential decodes.
    ///
    /// Default falls back to a decode loop — slower but correct, lets
    /// minor backends not reimplement the primitive.
    fn forward_verify(&mut self, cache_id: &str, tokens: &[u32]) -> Vec<f32> {
        let mut out = Vec::with_capacity(tokens.len() * self.config().vocab_size);
        // cache.len before any decode in this batch — we can derive per-token
        // position from it. Backends override this default for real batching.
        let start_pos = 0u32; // placeholder; real impls know their own state
        for (i, &tok) in tokens.iter().enumerate() {
            out.extend_from_slice(&self.decode(cache_id, tok, start_pos + i as u32));
        }
        out
    }

    /// Unified mixed-batch forward (chunked-prefill API).
    ///
    /// Accepts a heterogeneous batch where each item is `(cache_id,
    /// q_tokens, pos_offset, is_final_chunk)`:
    /// - `q_tokens.len() == 1` & `is_final_chunk == true` → decode step
    /// - `q_tokens.len() >= 1` & `is_final_chunk == true` → final
    ///   prefill chunk (returns logits for sampling)
    /// - `q_tokens.len() >= 1` & `is_final_chunk == false` → intermediate
    ///   prefill chunk (advances KV state, returns None)
    ///
    /// `pos_offset` is the absolute KV position of the first q-token
    /// for that sequence (0 for fresh prefill, prior `kv_len` for
    /// continuing chunks or decode steps).
    ///
    /// Returns one entry per `items[i]`: `Some(logits)` iff
    /// `is_final_chunk == true`, else `None`.
    ///
    /// Default implementation: returns `Err(unsupported)`. Concrete
    /// models that support a true unified forward (single forward pass
    /// over the concatenated `[M_total, hidden]` tensor + varlen
    /// attention) override this. The engine's caller (`LlmExecutor`)
    /// recognises the unsupported error and falls back to splitting
    /// the batch into per-item `prefill()` and a single `decode_batch()`
    /// — behaviour-preserving but doesn't get the chunked-prefill perf
    /// win until the model exposes a real unified path.
    #[allow(clippy::type_complexity)]
    fn unified_forward(
        &mut self,
        _items: &[(String, Vec<u32>, usize, bool)],
    ) -> std::result::Result<Vec<Option<Vec<f32>>>, ferrum_types::FerrumError> {
        Err(ferrum_types::FerrumError::unsupported(
            "unified_forward not implemented for this model",
        ))
    }

    /// Release the KV cache for a completed sequence.
    fn release(&mut self, cache_id: &str);

    /// Truncate the KV cache for `cache_id` back to `new_len` positions.
    /// Used by speculative decoding on rejection — roll draft/target KV
    /// back to the last accepted position before the next iteration.
    ///
    /// Default implementation is a panic so backends that don't support
    /// rollback fail loudly; implementations override this.
    fn truncate_kv(&mut self, cache_id: &str, new_len: usize) {
        let _ = (cache_id, new_len);
        panic!("truncate_kv not implemented for this DecoderOnlyLLM");
    }

    /// Drop all cached state (useful for tests and hot-reload).
    fn reset(&mut self) {}
}
