//! Model execution interface with clear prefill/decode separation
//!
//! This module provides the ModelExecutor trait that replaces the "fat" Model
//! interface, focusing purely on tensor operations without tokenization or sampling.

use crate::{KvCacheHandle, RecurrentStateHandle, RecurrentStateSpec, TensorRef};
use async_trait::async_trait;
use ferrum_types::{ModelInfo, RequestId, Result, TokenId};
use serde::{Deserialize, Serialize};
use std::{
    collections::{hash_map::DefaultHasher, HashMap},
    hash::{Hash, Hasher},
    sync::Arc,
};

/// One model-owned KV slot reservation request.
///
/// `cache_id` is the executor/model cache key attached to a sequence. `target_len`
/// is the sequence length that must be writable before the next forward runs.
/// `admission_target_len`, when present, is a larger known-context bound used
/// only for admission fit checks. Paged models must not allocate future blocks
/// for it; it mirrors vLLM's chunked-prefill full-context fit gate.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KvSlotRequest {
    pub cache_id: String,
    pub target_len: usize,
    pub admission_target_len: Option<usize>,
}

/// Per-cache outcome from a KV slot reservation attempt.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KvSlotAllocation {
    pub cache_id: String,
    pub blocks_before: usize,
    pub blocks_after: usize,
    pub new_blocks: usize,
}

/// Model-owned paged-KV reservation evidence.
///
/// Executors that own a vLLM-style physical KV block pool return this after
/// reserving all requested slots. Executors without model-owned paged KV return
/// `None` from `reserve_kv_slots`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KvSlotReservation {
    pub block_size: usize,
    pub total_blocks: usize,
    pub free_blocks_before: usize,
    pub free_blocks_after: usize,
    pub allocations: Vec<KvSlotAllocation>,
}

/// Point-in-time model-owned paged-KV capacity snapshot.
///
/// This is intentionally smaller than [`KvSlotReservation`]: it lets the
/// engine observe whether physical block capacity has actually changed after a
/// release, without allocating speculative slots or depending on model-family
/// names.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KvSlotCapacitySnapshot {
    pub block_size: usize,
    pub total_blocks: usize,
    pub free_blocks: usize,
}

/// Token-validity mask for model-side greedy argmax.
///
/// `valid_token_mask[id] != 0` means token `id` may be selected. Tokens at or
/// above `valid_token_mask.len()` are invalid. The fingerprint lets model
/// backends cache an uploaded device mask without comparing the full vector on
/// every decode step.
#[derive(Clone)]
pub struct TokenSelectionMask {
    pub fingerprint: u64,
    pub valid_token_mask: Arc<[i8]>,
}

impl TokenSelectionMask {
    pub fn new(valid_token_mask: Vec<i8>) -> Self {
        let mut hasher = DefaultHasher::new();
        valid_token_mask.hash(&mut hasher);
        Self {
            fingerprint: hasher.finish(),
            valid_token_mask: Arc::from(valid_token_mask),
        }
    }

    pub fn len(&self) -> usize {
        self.valid_token_mask.len()
    }

    pub fn is_empty(&self) -> bool {
        self.valid_token_mask.is_empty()
    }
}

impl std::fmt::Debug for TokenSelectionMask {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let valid_count = self.valid_token_mask.iter().filter(|&&v| v != 0).count();
        f.debug_struct("TokenSelectionMask")
            .field("fingerprint", &self.fingerprint)
            .field("len", &self.valid_token_mask.len())
            .field("valid_count", &valid_count)
            .finish()
    }
}

#[derive(Clone, Debug)]
pub enum LogitsReturnPolicy {
    FullLogits,
    GreedyArgmax {
        token_mask: Option<TokenSelectionMask>,
        repetition_penalty: Option<GreedyRepetitionPenalty>,
    },
}

impl Default for LogitsReturnPolicy {
    fn default() -> Self {
        Self::FullLogits
    }
}

impl LogitsReturnPolicy {
    pub fn requires_full_logits(&self) -> bool {
        matches!(self, Self::FullLogits)
    }
}

/// Sparse repetition-penalty metadata for model-side greedy argmax.
///
/// The token list is request-local and de-duplicated. Applying the penalty
/// before GPU argmax avoids downloading full `[batch, vocab]` logits for the
/// common greedy chat path while preserving repeat avoidance.
#[derive(Clone, Debug)]
pub struct GreedyRepetitionPenalty {
    pub penalty: f32,
    pub token_ids: Arc<[u32]>,
}

impl GreedyRepetitionPenalty {
    pub fn new(penalty: f32, token_ids: Vec<u32>) -> Self {
        Self {
            penalty,
            token_ids: Arc::from(token_ids),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.token_ids.is_empty() || self.penalty == 1.0
    }
}

/// Input for prefill phase (processing the initial prompt)
#[derive(Debug, Clone)]
pub struct PrefillInput {
    /// Stable product request identity for executor-owned runtime resources.
    pub request_id: Option<RequestId>,
    /// Maximum sequence extent this request may reach, including the prompt.
    /// Executors use this for fit validation without allocating future pages.
    pub maximum_sequence_tokens: Option<usize>,
    /// Input token IDs [batch_size, sequence_length]
    pub input_ids: TensorRef,
    /// Attention mask [batch_size, sequence_length] (optional)
    pub attention_mask: Option<TensorRef>,
    /// Position IDs [batch_size, sequence_length] (optional, for RoPE)
    pub position_ids: Option<TensorRef>,
    /// Pre-allocated KV cache handle (optional, for paged attention)
    pub kv_cache: Option<Arc<dyn KvCacheHandle>>,
    /// Pre-allocated recurrent-state handle (optional, for state-space layers)
    pub recurrent_state: Option<Arc<dyn RecurrentStateHandle>>,
    /// Request metadata that can affect model execution.
    pub metadata: HashMap<String, serde_json::Value>,
}

impl PrefillInput {
    /// Create new prefill input
    pub fn new(input_ids: TensorRef) -> Self {
        Self {
            request_id: None,
            maximum_sequence_tokens: None,
            input_ids,
            attention_mask: None,
            position_ids: None,
            kv_cache: None,
            recurrent_state: None,
            metadata: HashMap::new(),
        }
    }

    /// Attach the typed request boundary consumed by executor-owned runtimes.
    pub fn with_request_context(
        mut self,
        request_id: RequestId,
        maximum_sequence_tokens: usize,
    ) -> Self {
        self.request_id = Some(request_id);
        self.maximum_sequence_tokens = Some(maximum_sequence_tokens);
        self
    }

    /// Create prefill input with a pre-allocated KV cache handle.
    pub fn with_kv_cache(mut self, kv_cache: Arc<dyn KvCacheHandle>) -> Self {
        self.kv_cache = Some(kv_cache);
        self
    }

    /// Create prefill input with a pre-allocated recurrent-state handle.
    pub fn with_recurrent_state(mut self, recurrent_state: Arc<dyn RecurrentStateHandle>) -> Self {
        self.recurrent_state = Some(recurrent_state);
        self
    }

    /// Attach request metadata.
    pub fn with_metadata(mut self, metadata: HashMap<String, serde_json::Value>) -> Self {
        self.metadata = metadata;
        self
    }

    /// Add attention mask
    pub fn with_attention_mask(mut self, mask: TensorRef) -> Self {
        self.attention_mask = Some(mask);
        self
    }

    /// Add position IDs
    pub fn with_position_ids(mut self, positions: TensorRef) -> Self {
        self.position_ids = Some(positions);
        self
    }

    /// Get batch size
    pub fn batch_size(&self) -> usize {
        self.input_ids.shape()[0]
    }

    /// Get sequence length
    pub fn sequence_length(&self) -> usize {
        if self.input_ids.shape().len() >= 2 {
            self.input_ids.shape()[1]
        } else {
            1
        }
    }
}

/// Output from prefill phase
#[derive(Debug, Clone)]
pub struct PrefillOutput {
    /// Logits for all positions [batch_size, sequence_length, vocab_size]
    pub logits: TensorRef,
    /// KV cache handle populated with prompt states
    pub kv_cache: Arc<dyn KvCacheHandle>,
    /// Recurrent-state handle populated with prompt state, when used.
    pub recurrent_state: Option<Arc<dyn RecurrentStateHandle>>,
    /// Hidden states at each layer (optional, for analysis)
    pub hidden_states: Option<Vec<TensorRef>>,
    /// Attention weights (optional, for analysis)
    pub attention_weights: Option<Vec<TensorRef>>,
}

impl PrefillOutput {
    /// Create new prefill output
    pub fn new(logits: TensorRef, kv_cache: Arc<dyn KvCacheHandle>) -> Self {
        Self {
            logits,
            kv_cache,
            recurrent_state: None,
            hidden_states: None,
            attention_weights: None,
        }
    }

    /// Attach updated recurrent state to the prefill output.
    pub fn with_recurrent_state(mut self, recurrent_state: Arc<dyn RecurrentStateHandle>) -> Self {
        self.recurrent_state = Some(recurrent_state);
        self
    }

    /// Get logits for last position (for next token generation)
    pub fn last_token_logits(&self) -> Result<TensorRef> {
        let shape = self.logits.shape();
        if shape.len() != 3 {
            return Err(ferrum_types::FerrumError::backend(
                "Expected 3D logits tensor [batch, seq, vocab]",
            ));
        }

        let seq_len = shape[1];
        if seq_len == 0 {
            return Err(ferrum_types::FerrumError::backend("Empty sequence"));
        }

        // Extract last position: [batch, seq-1:seq, vocab] -> [batch, vocab]
        self.logits
            .view(&[0, seq_len - 1, 0], &[shape[0], seq_len, shape[2]])
    }
}

/// Input for decode phase (generating one token at a time)
#[derive(Debug, Clone)]
pub struct DecodeInput {
    /// Stable product request identity for executor-owned runtime resources.
    pub request_id: Option<RequestId>,
    /// Input token ID for current step [batch_size, 1]
    pub input_ids: TensorRef,
    /// Existing KV cache from previous steps
    pub kv_cache: Arc<dyn KvCacheHandle>,
    /// Existing recurrent state from previous steps, when used.
    pub recurrent_state: Option<Arc<dyn RecurrentStateHandle>>,
    /// Position IDs for current step [batch_size, 1] (optional)
    pub position_ids: Option<TensorRef>,
    /// Request metadata that can affect model execution.
    pub metadata: HashMap<String, serde_json::Value>,
    /// How the model may return final-position logits for this request.
    pub logits_policy: LogitsReturnPolicy,
}

impl DecodeInput {
    /// Create new decode input
    pub fn new(input_ids: TensorRef, kv_cache: Arc<dyn KvCacheHandle>) -> Self {
        Self {
            request_id: None,
            input_ids,
            kv_cache,
            recurrent_state: None,
            position_ids: None,
            metadata: HashMap::new(),
            logits_policy: LogitsReturnPolicy::FullLogits,
        }
    }

    /// Attach the product request identity to this decode step.
    pub fn with_request_id(mut self, request_id: RequestId) -> Self {
        self.request_id = Some(request_id);
        self
    }

    /// Add position IDs
    pub fn with_position_ids(mut self, positions: TensorRef) -> Self {
        self.position_ids = Some(positions);
        self
    }

    /// Attach recurrent state for state-space or hybrid layers.
    pub fn with_recurrent_state(mut self, recurrent_state: Arc<dyn RecurrentStateHandle>) -> Self {
        self.recurrent_state = Some(recurrent_state);
        self
    }

    /// Attach request metadata.
    pub fn with_metadata(mut self, metadata: HashMap<String, serde_json::Value>) -> Self {
        self.metadata = metadata;
        self
    }

    pub fn with_logits_policy(mut self, policy: LogitsReturnPolicy) -> Self {
        self.logits_policy = policy;
        self
    }

    /// Get batch size
    pub fn batch_size(&self) -> usize {
        self.input_ids.shape()[0]
    }
}

/// One sequence's contribution to a unified mixed-batch forward.
///
/// A unified batch lets a single model forward pass process a mix of
/// per-sequence work units: a prefill chunk (q_tokens.len() ≥ 1, possibly
/// continuing from `pos_offset > 0` for chunked prefill) and a decode step
/// (q_tokens.len() == 1, `pos_offset` = current cache length) coexist in
/// the same call. The model layer concatenates all `q_tokens` into one
/// [M_total, hidden] tensor and runs all GEMMs / norms once; only the
/// attention kernel sees per-item segmentation.
///
/// This is the abstraction that enables vLLM-style chunked prefill where
/// decode tokens for already-running sequences are produced in the same
/// iter as a prefill chunk for a newly-arriving sequence.
#[derive(Clone)]
pub struct UnifiedBatchItem {
    /// Identifier matching the sequence's KV cache (model-side keying).
    pub seq_id: String,
    /// Tokens to process this iter. For decode this is exactly 1 token;
    /// for prefill (chunked or whole) this is the chunk's tokens.
    pub q_tokens: Vec<u32>,
    /// KV cache handle for this sequence.
    pub kv_cache: Arc<dyn KvCacheHandle>,
    /// Recurrent-state handle for this sequence, when used.
    pub recurrent_state: Option<Arc<dyn RecurrentStateHandle>>,
    /// Starting absolute position for the FIRST token in `q_tokens`.
    /// 0 for a fresh prefill, `kv_len` for a decode step or a continuing
    /// chunked-prefill slice.
    pub pos_offset: usize,
    /// True iff this item completes the request's prefill (or is a decode
    /// item) — i.e. logits at the last token of `q_tokens` should be
    /// returned for sampling. Intermediate prefill chunks set this false
    /// to skip the lm_head + sampling path.
    pub is_final_chunk: bool,
    /// Request metadata that can affect model execution.
    pub metadata: HashMap<String, serde_json::Value>,
    /// How the model may return final-position logits for this item.
    pub logits_policy: LogitsReturnPolicy,
}

impl std::fmt::Debug for UnifiedBatchItem {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("UnifiedBatchItem")
            .field("seq_id", &self.seq_id)
            .field("q_len", &self.q_tokens.len())
            .field("has_recurrent_state", &self.recurrent_state.is_some())
            .field("pos_offset", &self.pos_offset)
            .field("is_final_chunk", &self.is_final_chunk)
            .finish()
    }
}

/// A mixed-batch forward request: any combination of in-progress prefill
/// chunks and decode steps. See [`UnifiedBatchItem`] for the per-item
/// semantics. The producer (engine) groups all sequences active in this
/// iter into a single batch; the consumer (model) runs one forward and
/// returns per-item logits (only for items with `is_final_chunk = true`,
/// in the order they appear in `items`).
#[derive(Debug, Clone, Default)]
pub struct UnifiedBatch {
    pub items: Vec<UnifiedBatchItem>,
}

impl UnifiedBatch {
    pub fn new() -> Self {
        Self::default()
    }

    /// Total query tokens across all items — corresponds to the M dim of
    /// the model's per-layer GEMMs in the unified forward.
    pub fn total_q_tokens(&self) -> usize {
        self.items.iter().map(|it| it.q_tokens.len()).sum()
    }

    /// Number of items that will produce a logits vector (decode items
    /// always; prefill items only on their final chunk).
    pub fn num_sampled_items(&self) -> usize {
        self.items.iter().filter(|it| it.is_final_chunk).count()
    }
}

/// Output from decode phase
#[derive(Debug, Clone)]
pub struct DecodeOutput {
    /// Logits for next token [batch_size, vocab_size]
    pub logits: TensorRef,
    /// Updated KV cache with new token state
    pub kv_cache: Arc<dyn KvCacheHandle>,
    /// Updated recurrent state, when used.
    pub recurrent_state: Option<Arc<dyn RecurrentStateHandle>>,
    /// Hidden state for current token (optional)
    pub hidden_state: Option<TensorRef>,
    /// Attention weights for current token (optional)
    pub attention_weights: Option<Vec<TensorRef>>,
}

impl DecodeOutput {
    /// Create new decode output
    pub fn new(logits: TensorRef, kv_cache: Arc<dyn KvCacheHandle>) -> Self {
        Self {
            logits,
            kv_cache,
            recurrent_state: None,
            hidden_state: None,
            attention_weights: None,
        }
    }

    /// Attach updated recurrent state to the decode output.
    pub fn with_recurrent_state(mut self, recurrent_state: Arc<dyn RecurrentStateHandle>) -> Self {
        self.recurrent_state = Some(recurrent_state);
        self
    }
}

/// Declares which side owns request-lifetime accelerator resources.
///
/// This is a lifecycle boundary, not a capacity limit. Executor-managed
/// runtimes still make dynamic admission decisions from then-live capacity;
/// the engine must not reserve a second KV or recurrent-state allocation for
/// the same request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExecutionResourceOwnership {
    EngineManaged,
    ExecutorManaged,
}

/// Core model executor trait focusing on tensor operations
#[async_trait]
pub trait ModelExecutor: Send + Sync {
    /// Get model information and metadata
    fn info(&self) -> &ModelInfo;

    /// Selects the single authority for request-lifetime model resources.
    /// Existing executors remain engine-managed by default. A runtime that
    /// returns `ExecutorManaged` must return its own opaque cache handle from
    /// prefill/decode and release that authority from `release_cache`.
    fn execution_resource_ownership(&self) -> ExecutionResourceOwnership {
        ExecutionResourceOwnership::EngineManaged
    }

    /// Whether this executor's backend can run the unified mixed prefill+decode
    /// forward natively. When false, the engine routes Qwen3-MoE batches through
    /// the legacy split path. Reported by the (backend-aware) executor so the
    /// engine stays backend-agnostic — replaces a `cfg(target_os)` branch that
    /// previously hard-coded "Metal/CPU lack native unified" in the hot path.
    ///
    /// Default false (conservative legacy path); accelerators with a native
    /// unified forward override to true.
    fn supports_native_unified_decode(&self) -> bool {
        false
    }

    /// Per-request KV capacity in tokens when the executor owns a smaller
    /// runtime cache window than the model's declared context length.
    fn kv_capacity(&self) -> Option<usize> {
        None
    }

    /// Reserve model-owned KV slots before a forward is dispatched.
    ///
    /// This is the executor-level admission hook for vLLM-style paged KV. The
    /// engine calls it at the batch boundary so a request that cannot grow its
    /// KV cache is delayed or preempted before kernel launch instead of
    /// panicking inside attention.
    fn reserve_kv_slots(&self, _requests: &[KvSlotRequest]) -> Result<Option<KvSlotReservation>> {
        Ok(None)
    }

    /// Snapshot model-owned paged-KV capacity without allocating slots.
    ///
    /// Executors without model-owned paged KV return `None`.
    fn kv_slot_capacity_snapshot(&self) -> Option<KvSlotCapacitySnapshot> {
        None
    }

    /// Recurrent-state allocation spec for this request, when the model has
    /// state-space or hybrid layers that need per-request recurrent state.
    ///
    /// Attention-only models return `None`. If this returns `Some`, the engine
    /// must allocate a recurrent-state handle before prefill and pass it through
    /// prefill/decode inputs. The default keeps existing executors KV-only.
    fn recurrent_state_spec(
        &self,
        _request_id: &RequestId,
        _input_tokens: &[TokenId],
    ) -> Result<Option<RecurrentStateSpec>> {
        Ok(None)
    }

    /// Execute prefill phase (process initial prompt)
    async fn prefill(&self, input: &PrefillInput) -> Result<PrefillOutput>;

    /// Batch prefill: process multiple prompts' prefill in ONE forward pass.
    ///
    /// Default implementation falls back to per-request `prefill()` (serial,
    /// which is the current behavior the engine sees today). Executors that
    /// support unified mixed-batch forward (e.g. via `model.unified_forward`
    /// over a varlen QKV path) should override this to amortize launch /
    /// kernel-overhead across all `inputs` items in one call.
    ///
    /// Used by the continuous-batching engine to coalesce a cohort of new
    /// prefills (apples M3 c=32 sees 32 simultaneous prefills as one logical
    /// batch; the serial fallback runs each in ~47 ms while a true batched
    /// path runs all 32 in ~100 ms).
    async fn batch_prefill(&self, inputs: &[PrefillInput]) -> Result<Vec<PrefillOutput>> {
        let mut outputs = Vec::with_capacity(inputs.len());
        for input in inputs {
            outputs.push(self.prefill(input).await?);
        }
        Ok(outputs)
    }

    /// Execute decode phase (generate next token)
    async fn decode(&self, input: &DecodeInput) -> Result<DecodeOutput>;

    /// Batch decode: process multiple sequences in one forward pass.
    ///
    /// Default implementation falls back to per-request `decode()`.
    /// Executors with batched CUDA runners should override this.
    async fn batch_decode(&self, inputs: &[DecodeInput]) -> Result<Vec<DecodeOutput>> {
        let mut outputs = Vec::with_capacity(inputs.len());
        for input in inputs {
            outputs.push(self.decode(input).await?);
        }
        Ok(outputs)
    }

    /// Unified mixed-batch forward: process a [`UnifiedBatch`] containing
    /// any combination of prefill chunks (one or more `q_tokens` per item,
    /// possibly continuing from `pos_offset > 0`) and decode steps
    /// (`q_tokens.len() == 1`, `is_final_chunk = true`) in a single model
    /// forward pass.
    ///
    /// Returns one element per `batch.items[i]`:
    /// - `Some(logits)` for items with `is_final_chunk = true` (the
    ///   request's final-position logits, ready for sampling)
    /// - `None` for intermediate prefill chunks (no lm_head executed —
    ///   model only updates KV state)
    ///
    /// Default implementation returns `Err(unsupported)`. Concrete LLM
    /// executors should override with either:
    /// - A behavioral fallback that dispatches each chunk via existing
    ///   `prefill()` and groups decode items into `batch_decode()` (this
    ///   preserves current behavior; no perf change), OR
    /// - A real unified-forward path that runs all items through one
    ///   `[M_total, hidden]` GEMM chain with a varlen attention kernel
    ///   (this is the chunked-prefill perf unlock).
    async fn unified_decode(&self, _batch: &UnifiedBatch) -> Result<Vec<Option<Vec<f32>>>> {
        Err(ferrum_types::FerrumError::unsupported(
            "unified_decode not implemented for this executor",
        ))
    }

    /// Optional: full forward pass (for non-autoregressive use cases)
    async fn forward(&self, _input: &TensorRef) -> Result<TensorRef> {
        // Default implementation not supported
        Err(ferrum_types::FerrumError::unsupported(
            "Full forward pass not supported by this executor",
        ))
    }

    /// Roll the KV cache for this executor's sequence back to `new_len`.
    /// Used by speculative decoding on partial rejection so the next
    /// iteration sees a KV prefix that matches the accepted token stream.
    /// Default: Ok(()) — executors that don't cache per-sequence state
    /// (stub, mock) are inherently tolerant; real LLM executors override.
    async fn truncate_kv(
        &self,
        _kv_cache: &std::sync::Arc<dyn crate::KvCacheHandle>,
        _new_len: usize,
    ) -> Result<()> {
        Ok(())
    }

    /// Multi-position decode-verify: one forward over `N+1` tokens,
    /// producing one logits row per position. Used by speculative
    /// decoding's target path so we don't pay N+1 sequential forwards.
    ///
    /// Default falls back to N+1 sequential `decode()` calls — correct
    /// but slow; real LLM executors override.
    ///
    /// Returns a `Vec<DecodeOutput>` of length `inputs.len()` with the
    /// final KV handle attached to the last element.
    async fn forward_verify(&self, inputs: &[DecodeInput]) -> Result<Vec<DecodeOutput>> {
        let mut out = Vec::with_capacity(inputs.len());
        for input in inputs {
            out.push(self.decode(input).await?);
        }
        Ok(out)
    }

    /// Get executor capabilities
    fn capabilities(&self) -> ExecutorCapabilities;

    /// Get current executor status
    fn status(&self) -> ExecutorStatus;

    /// Optional model/executor cache metrics.
    ///
    /// Concrete LLM executors use this for model-level paged KV prefix reuse
    /// counters. Default implementations keep non-autoregressive executors
    /// and tests from needing cache-specific plumbing.
    fn cache_metrics_snapshot(&self) -> Option<serde_json::Value> {
        None
    }

    /// Optional LoRA runtime metrics.
    fn lora_metrics_snapshot(&self) -> Option<serde_json::Value> {
        None
    }

    /// Warm up executor (load model, allocate memory, etc.)
    async fn warmup(&mut self) -> Result<()> {
        // Default no-op implementation
        Ok(())
    }

    /// Shutdown executor gracefully
    async fn shutdown(&mut self) -> Result<()> {
        // Default no-op implementation
        Ok(())
    }

    /// Release KV cache and state for a completed sequence.
    ///
    /// Called by the engine when a request finishes (success or error) to free
    /// GPU memory held by the sequence's KV cache. The `cache_id` matches the
    /// value embedded in the `KvCacheHandle` returned by prefill/decode.
    fn release_cache(&self, _cache_id: &str) {
        // Default no-op — executors that manage per-sequence KV caches should override.
    }
}

/// Executor capabilities and configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutorCapabilities {
    /// Maximum supported batch size
    pub max_batch_size: usize,
    /// Maximum sequence length
    pub max_sequence_length: usize,
    /// Supported attention mechanisms
    pub attention_mechanisms: Vec<AttentionType>,
    /// Whether executor supports dynamic batching
    pub supports_dynamic_batching: bool,
    /// Whether executor supports continuous batching
    pub supports_continuous_batching: bool,
    /// Whether executor supports speculative decoding
    pub supports_speculative_decoding: bool,
    /// Whether executor supports tensor parallelism
    pub supports_tensor_parallelism: bool,
    /// Whether executor supports pipeline parallelism
    pub supports_pipeline_parallelism: bool,
    /// Supported data types
    pub supported_dtypes: Vec<ferrum_types::DataType>,
    /// Supported devices
    pub supported_devices: Vec<ferrum_types::Device>,
    /// Memory requirements estimation
    pub memory_requirements: MemoryRequirements,
}

/// Attention mechanism types
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum AttentionType {
    /// Standard multi-head attention
    MultiHead,
    /// Multi-query attention (MQA)
    MultiQuery,
    /// Grouped-query attention (GQA)
    GroupedQuery,
    /// Flash attention
    Flash,
    /// Paged attention
    Paged,
    /// Sliding window attention
    SlidingWindow,
}

/// Memory requirements for model execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryRequirements {
    /// Model parameter memory in bytes
    pub parameter_memory: u64,
    /// Minimum activation memory per token
    pub activation_memory_per_token: usize,
    /// KV cache memory per token per layer
    pub kv_cache_memory_per_token: usize,
    /// Additional overhead memory
    pub overhead_memory: u64,
}

impl MemoryRequirements {
    /// Calculate total memory for given configuration
    pub fn calculate_total_memory(
        &self,
        batch_size: usize,
        sequence_length: usize,
        num_layers: usize,
    ) -> u64 {
        let activation_mem =
            (self.activation_memory_per_token * batch_size * sequence_length) as u64;
        let kv_cache_mem =
            (self.kv_cache_memory_per_token * batch_size * sequence_length * num_layers) as u64;

        self.parameter_memory + activation_mem + kv_cache_mem + self.overhead_memory
    }
}

/// Executor status information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutorStatus {
    /// Current executor state
    pub state: ExecutorState,
    /// Whether executor is ready to accept requests
    pub is_ready: bool,
    /// Current batch size being processed
    pub current_batch_size: usize,
    /// Number of prefill operations completed
    pub prefill_operations: u64,
    /// Number of decode operations completed
    pub decode_operations: u64,
    /// Average prefill time in milliseconds
    pub avg_prefill_time_ms: f64,
    /// Average decode time in milliseconds
    pub avg_decode_time_ms: f64,
    /// Memory usage statistics
    pub memory_usage: ExecutorMemoryUsage,
    /// Last operation timestamp
    #[serde(skip)]
    pub last_operation: Option<std::time::Instant>,
}

/// Executor state
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExecutorState {
    /// Executor is initializing
    Initializing,
    /// Executor is ready to accept requests
    Ready,
    /// Executor is processing requests
    Busy,
    /// Executor encountered an error
    Error,
    /// Executor is shutting down
    Shutdown,
}

/// Executor memory usage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutorMemoryUsage {
    /// Total allocated memory in bytes
    pub allocated_bytes: usize,
    /// Currently used memory in bytes
    pub used_bytes: usize,
    /// Peak memory usage
    pub peak_bytes: usize,
    /// Memory utilization percentage
    pub utilization_percent: f32,
}

/// Batch model executor for processing multiple requests efficiently
#[async_trait]
pub trait BatchModelExecutor: ModelExecutor {
    /// Execute batch prefill for multiple sequences
    async fn batch_prefill(&self, inputs: &[PrefillInput]) -> Result<Vec<PrefillOutput>>;

    /// Execute batch decode for multiple sequences
    async fn batch_decode(&self, inputs: &[DecodeInput]) -> Result<Vec<DecodeOutput>>;

    /// Get optimal batch size for current conditions
    fn optimal_batch_size(&self) -> usize;

    /// Check if batch size is supported
    fn supports_batch_size(&self, batch_size: usize) -> bool;
}

/// Speculative execution support
#[async_trait]
pub trait SpeculativeExecutor: ModelExecutor {
    /// Execute speculative decoding with draft model
    async fn speculative_decode(
        &self,
        input: &DecodeInput,
        draft_tokens: &[ferrum_types::TokenId],
        acceptance_threshold: f32,
    ) -> Result<SpeculativeDecodeOutput>;
}

/// Output from speculative decoding
#[derive(Debug, Clone)]
pub struct SpeculativeDecodeOutput {
    /// Accepted tokens (subset of draft tokens)
    pub accepted_tokens: Vec<ferrum_types::TokenId>,
    /// Logits for the next token after last accepted
    pub next_logits: TensorRef,
    /// Updated KV cache
    pub kv_cache: Arc<dyn KvCacheHandle>,
    /// Number of draft tokens accepted
    pub acceptance_count: usize,
}

/// Model executor factory
#[async_trait]
pub trait ModelExecutorFactory: Send + Sync {
    /// Create executor from model configuration
    async fn create_executor(&self, config: &ExecutorConfig) -> Result<Box<dyn ModelExecutor>>;

    /// Create batch executor
    async fn create_batch_executor(
        &self,
        config: &ExecutorConfig,
    ) -> Result<Box<dyn BatchModelExecutor>>;

    /// Get supported executor types
    fn supported_types(&self) -> Vec<ExecutorType>;

    /// Validate configuration
    fn validate_config(&self, config: &ExecutorConfig) -> Result<()>;
}

/// Executor configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutorConfig {
    /// Model information
    pub model_info: ModelInfo,
    /// Target device
    pub device: ferrum_types::Device,
    /// Data type for computation
    pub dtype: ferrum_types::DataType,
    /// Maximum batch size
    pub max_batch_size: usize,
    /// Maximum sequence length
    pub max_sequence_length: usize,
    /// Attention configuration
    pub attention_config: ExecutorAttentionConfig,
    /// Memory configuration
    pub memory_config: ExecutorMemoryConfig,
    /// Optimization settings
    pub optimization_config: OptimizationConfig,
    /// Additional executor-specific options
    pub executor_options: HashMap<String, serde_json::Value>,
}

/// Runtime attention configuration for model executor
///
/// Note: This is different from ferrum_types::AttentionConfig which describes
/// the model architecture's attention configuration from config.json.
/// This type describes the runtime execution settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutorAttentionConfig {
    /// Type of attention to use
    pub attention_type: AttentionType,
    /// Enable flash attention if available
    pub enable_flash_attention: bool,
    /// Enable paged attention
    pub enable_paged_attention: bool,
    /// Block size for paged attention
    pub block_size: Option<usize>,
    /// Sliding window size (if using sliding window attention)
    pub sliding_window_size: Option<usize>,
}

/// Memory configuration for executor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutorMemoryConfig {
    /// Enable memory pooling
    pub enable_memory_pooling: bool,
    /// Memory pool size in bytes (None for auto)
    pub memory_pool_size: Option<usize>,
    /// Enable KV cache sharing
    pub enable_kv_cache_sharing: bool,
    /// Maximum memory usage percentage
    pub max_memory_usage: f32,
}

/// Optimization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationConfig {
    /// Enable CUDA graphs (if supported)
    pub enable_cuda_graphs: bool,
    /// Enable kernel fusion
    pub enable_kernel_fusion: bool,
    /// Enable mixed precision
    pub enable_mixed_precision: bool,
    /// Optimization level (0-3)
    pub optimization_level: u8,
    /// Custom optimization flags
    pub custom_flags: HashMap<String, bool>,
}

/// Supported executor types
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ExecutorType {
    /// Standard sequential executor
    Sequential,
    /// Batch executor for parallel processing
    Batch,
    /// Continuous batching executor
    ContinuousBatch,
    /// Speculative decoding executor
    Speculative,
    /// Pipeline parallel executor
    PipelineParallel,
    /// Tensor parallel executor
    TensorParallel,
}

/// Executor performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutorMetrics {
    /// Total operations executed
    pub total_operations: u64,
    /// Prefill operations
    pub prefill_operations: u64,
    /// Decode operations
    pub decode_operations: u64,
    /// Average prefill latency (ms)
    pub avg_prefill_latency: f64,
    /// Average decode latency (ms)
    pub avg_decode_latency: f64,
    /// P95 prefill latency (ms)
    pub p95_prefill_latency: f64,
    /// P95 decode latency (ms)
    pub p95_decode_latency: f64,
    /// Throughput (tokens per second)
    pub throughput_tps: f64,
    /// Memory efficiency (used/allocated)
    pub memory_efficiency: f32,
    /// Batch utilization
    pub batch_utilization: f32,
}

/// Executor registry for managing multiple executors
pub trait ExecutorRegistry: Send + Sync {
    /// Register executor with name
    fn register(&mut self, name: &str, executor: Box<dyn ModelExecutor>) -> Result<()>;

    /// Get executor by name
    fn get(&self, name: &str) -> Option<&dyn ModelExecutor>;

    /// Remove executor by name
    fn remove(&mut self, name: &str) -> Option<Box<dyn ModelExecutor>>;

    /// List registered executor names
    fn list_names(&self) -> Vec<String>;

    /// Get executor metrics
    fn get_metrics(&self, name: &str) -> Option<ExecutorMetrics>;
}
