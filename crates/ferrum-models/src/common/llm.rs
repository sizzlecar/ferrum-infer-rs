//! `DecoderOnlyLLM` trait — the "model family" interface that every
//! decoder-only language model (Qwen3 / Llama / Mistral / DeepSeek / ...)
//! implements, independent of backend and weight format.
//!
//! `LlmExecutor` (living in `ferrum-engine`) holds a `Box<dyn DecoderOnlyLLM>`
//! and adapts it to the `ModelExecutor` trait that the scheduler calls.

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
