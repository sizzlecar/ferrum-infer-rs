//! Decode backend abstraction.
//!
//! Different backends (CUDA, Metal, CPU/Candle) implement `DecodeBackend`
//! to provide optimized decode execution. The `GenericDecodeExecutor` uses
//! a DecodeBackend to execute the decode hot path.

use crate::tensor::TensorRef;
use ferrum_types::Result;

/// Decode-phase execution backend.
///
/// Implements the actual computation for single-token decode steps.
/// Different backends optimize for different hardware:
/// - `CudaDecodeBackend`: cuBLAS + custom CUDA kernels, pre-allocated buffers
/// - `MetalDecodeBackend`: Metal compute shaders
/// - `CandleDecodeBackend`: candle tensor ops (CPU/fallback)
///
/// The backend is initialized with model weights and manages its own
/// internal state (KV cache, buffers, cuBLAS handles, etc.).
pub trait DecodeBackend: Send + Sync {
    /// Execute a single decode step: one token in, logits out.
    ///
    /// - `token_id`: the input token
    /// - `position`: sequence position (for RoPE)
    /// - `cache_key`: identifies the sequence's KV cache
    ///
    /// Returns logits as a TensorRef [1, 1, vocab_size].
    fn decode_step(&mut self, token_id: u32, position: usize, cache_key: &str)
        -> Result<TensorRef>;

    /// Initialize KV cache for a new sequence from prefill data.
    ///
    /// Called after prefill (which runs through the model's forward pass)
    /// to hand off the KV cache to the decode backend.
    ///
    /// `kv_data`: per-layer (K, V) tensor pairs from the prefill pass.
    /// `prefill_len`: number of tokens in the prefill.
    fn init_kv_cache(
        &mut self,
        cache_key: &str,
        kv_data: Vec<(TensorRef, TensorRef)>,
        prefill_len: usize,
    ) -> Result<()>;

    /// Check if KV cache exists for a sequence.
    fn has_kv_cache(&self, cache_key: &str) -> bool;

    /// Release KV cache for a completed sequence.
    fn release_kv_cache(&mut self, cache_key: &str);

    /// Human-readable backend name (for logging).
    fn name(&self) -> &str;
}
