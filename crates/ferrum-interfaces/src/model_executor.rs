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
    future::Future,
    hash::{Hash, Hasher},
    num::NonZeroU64,
    ops::Range,
    pin::Pin,
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
    /// Stable product request identity for plan-runtime resources.
    pub request_id: Option<RequestId>,
    /// Maximum sequence extent this request may reach, including the prompt.
    /// Executors use this for fit validation without allocating future pages.
    pub maximum_sequence_tokens: Option<usize>,
    /// Exact scheduler-owned prompt chunk for this invocation.
    ///
    /// The input tensor still contains the full prompt so token identity and
    /// global offsets remain stable. Plan runtimes execute only this range.
    pub chunk: Option<PrefillChunk>,
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
            chunk: None,
            input_ids,
            attention_mask: None,
            position_ids: None,
            kv_cache: None,
            recurrent_state: None,
            metadata: HashMap::new(),
        }
    }

    /// Attach the typed request boundary consumed by plan runtimes.
    pub fn with_request_context(
        mut self,
        request_id: RequestId,
        maximum_sequence_tokens: usize,
    ) -> Self {
        self.request_id = Some(request_id);
        self.maximum_sequence_tokens = Some(maximum_sequence_tokens);
        self
    }

    /// Attach the exact scheduler-published prompt chunk.
    pub fn with_chunk(mut self, chunk: PrefillChunk) -> Self {
        self.chunk = Some(chunk);
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

/// Exact, validated prompt progress assigned to one prefill invocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct PrefillChunk {
    tokens_processed: usize,
    tokens_to_process: usize,
    total_prompt_tokens: usize,
}

#[cfg(test)]
mod prefill_chunk_tests {
    use super::PrefillChunk;

    #[test]
    fn validates_exact_progress_and_finality() {
        let first = PrefillChunk::new(0, 3, 8).unwrap();
        assert_eq!(first.range(), 0..3);
        assert_eq!(first.end(), 3);
        assert!(!first.is_final());

        let final_chunk = PrefillChunk::new(3, 5, 8).unwrap();
        assert_eq!(final_chunk.range(), 3..8);
        assert!(final_chunk.is_final());
    }

    #[test]
    fn rejects_empty_out_of_bounds_and_overflowing_progress() {
        assert!(PrefillChunk::new(0, 0, 8).is_err());
        assert!(PrefillChunk::new(0, 1, 0).is_err());
        assert!(PrefillChunk::new(7, 2, 8).is_err());
        assert!(PrefillChunk::new(usize::MAX, 1, usize::MAX).is_err());
    }
}

impl PrefillChunk {
    pub fn new(
        tokens_processed: usize,
        tokens_to_process: usize,
        total_prompt_tokens: usize,
    ) -> Result<Self> {
        let end = tokens_processed
            .checked_add(tokens_to_process)
            .ok_or_else(|| {
                ferrum_types::FerrumError::request_validation("prefill chunk overflows")
            })?;
        if tokens_to_process == 0 || total_prompt_tokens == 0 || end > total_prompt_tokens {
            return Err(ferrum_types::FerrumError::request_validation(
                "prefill chunk must be non-empty and within the full prompt",
            ));
        }
        Ok(Self {
            tokens_processed,
            tokens_to_process,
            total_prompt_tokens,
        })
    }

    pub const fn tokens_processed(self) -> usize {
        self.tokens_processed
    }

    pub const fn tokens_to_process(self) -> usize {
        self.tokens_to_process
    }

    pub const fn total_prompt_tokens(self) -> usize {
        self.total_prompt_tokens
    }

    pub fn range(self) -> Range<usize> {
        self.tokens_processed..self.tokens_processed + self.tokens_to_process
    }

    pub const fn end(self) -> usize {
        self.tokens_processed + self.tokens_to_process
    }

    pub const fn is_final(self) -> bool {
        self.end() == self.total_prompt_tokens
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
    /// Stable product request identity for plan-runtime resources.
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

/// Declares the authoritative runtime for request-lifetime accelerator resources.
///
/// This is a lifecycle boundary, not a capacity limit. `PlanRuntime` means the
/// shared execution runtime owns admission, allocation, fences, and release;
/// a model executor may adapt those operations but is not their owner. The
/// engine must not reserve a second KV or recurrent-state allocation for the
/// same request. `LegacyEngine` exists only while old executors are migrated.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExecutionResourceAuthority {
    LegacyEngine,
    PlanRuntime,
}

/// Point-in-time memory evidence emitted by the shared plan runtime.
///
/// Static model allocations are separated from dynamic request resources so
/// product telemetry never reports model weights as KV or recurrent-state
/// usage. Process-wide claims are included because another live plan can
/// consume capacity visible to this runtime. Dynamic free bytes remain
/// reusable by this plan; quarantined and other claimed bytes do not.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PlanRuntimeResourceSnapshot {
    device_capacity_bytes: u64,
    usable_capacity_bytes: u64,
    process_claimed_bytes: u64,
    plan_claimed_bytes: u64,
    static_bytes: u64,
    dynamic_resident_bytes: u64,
    dynamic_free_bytes: u64,
    pending_growth_bytes: u64,
    quarantined_bytes: u64,
}

impl PlanRuntimeResourceSnapshot {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        device_capacity_bytes: u64,
        usable_capacity_bytes: u64,
        process_claimed_bytes: u64,
        plan_claimed_bytes: u64,
        static_bytes: u64,
        dynamic_resident_bytes: u64,
        dynamic_free_bytes: u64,
        pending_growth_bytes: u64,
        quarantined_bytes: u64,
    ) -> Result<Self> {
        if usable_capacity_bytes > device_capacity_bytes {
            return Err(ferrum_types::FerrumError::internal(format!(
                "plan runtime usable capacity {usable_capacity_bytes} exceeds device capacity {device_capacity_bytes}"
            )));
        }
        if process_claimed_bytes > usable_capacity_bytes {
            return Err(ferrum_types::FerrumError::internal(format!(
                "plan runtime process claims {process_claimed_bytes} exceed usable capacity {usable_capacity_bytes}"
            )));
        }
        if plan_claimed_bytes > process_claimed_bytes {
            return Err(ferrum_types::FerrumError::internal(format!(
                "plan runtime plan claims {plan_claimed_bytes} exceed process claims {process_claimed_bytes}"
            )));
        }
        if dynamic_free_bytes > dynamic_resident_bytes {
            return Err(ferrum_types::FerrumError::internal(format!(
                "plan runtime dynamic free bytes {dynamic_free_bytes} exceed resident bytes {dynamic_resident_bytes}"
            )));
        }
        let minimum_plan_claim = static_bytes
            .checked_add(dynamic_resident_bytes)
            .and_then(|bytes| bytes.checked_add(quarantined_bytes))
            .ok_or_else(|| {
                ferrum_types::FerrumError::internal(
                    "plan runtime static, resident, and quarantined bytes overflow u64",
                )
            })?;
        if minimum_plan_claim > plan_claimed_bytes {
            return Err(ferrum_types::FerrumError::internal(format!(
                "plan runtime accounted plan bytes {minimum_plan_claim} exceed plan claims {plan_claimed_bytes}"
            )));
        }
        Ok(Self {
            device_capacity_bytes,
            usable_capacity_bytes,
            process_claimed_bytes,
            plan_claimed_bytes,
            static_bytes,
            dynamic_resident_bytes,
            dynamic_free_bytes,
            pending_growth_bytes,
            quarantined_bytes,
        })
    }

    pub const fn device_capacity_bytes(&self) -> u64 {
        self.device_capacity_bytes
    }

    pub const fn usable_capacity_bytes(&self) -> u64 {
        self.usable_capacity_bytes
    }

    pub const fn process_claimed_bytes(&self) -> u64 {
        self.process_claimed_bytes
    }

    pub const fn plan_claimed_bytes(&self) -> u64 {
        self.plan_claimed_bytes
    }

    pub const fn static_bytes(&self) -> u64 {
        self.static_bytes
    }

    pub const fn dynamic_resident_bytes(&self) -> u64 {
        self.dynamic_resident_bytes
    }

    pub const fn dynamic_free_bytes(&self) -> u64 {
        self.dynamic_free_bytes
    }

    pub const fn dynamic_used_bytes(&self) -> u64 {
        self.dynamic_resident_bytes - self.dynamic_free_bytes
    }

    pub const fn pending_growth_bytes(&self) -> u64 {
        self.pending_growth_bytes
    }

    pub const fn quarantined_bytes(&self) -> u64 {
        self.quarantined_bytes
    }

    /// Capacity immediately reusable by this plan without reclaiming another
    /// plan: process-wide unclaimed bytes plus free extents already resident
    /// in this plan's dynamic pools.
    pub fn available_bytes(&self) -> Result<u64> {
        self.usable_capacity_bytes
            .checked_sub(self.process_claimed_bytes)
            .and_then(|bytes| bytes.checked_add(self.dynamic_free_bytes))
            .ok_or_else(|| {
                ferrum_types::FerrumError::internal(
                    "plan runtime available capacity calculation overflowed",
                )
            })
    }

    pub fn used_bytes(&self) -> Result<u64> {
        self.available_bytes().and_then(|available| {
            self.usable_capacity_bytes
                .checked_sub(available)
                .ok_or_else(|| {
                    ferrum_types::FerrumError::internal(
                        "plan runtime available bytes exceed usable capacity",
                    )
                })
        })
    }
}

#[cfg(test)]
mod plan_runtime_resource_snapshot_tests {
    use super::PlanRuntimeResourceSnapshot;

    #[test]
    fn separates_static_and_dynamic_usage() {
        let snapshot =
            PlanRuntimeResourceSnapshot::new(1_000, 900, 710, 710, 400, 300, 200, 20, 10).unwrap();

        assert_eq!(snapshot.available_bytes().unwrap(), 390);
        assert_eq!(snapshot.used_bytes().unwrap(), 510);
        assert_eq!(snapshot.dynamic_resident_bytes(), 300);
        assert_eq!(snapshot.dynamic_used_bytes(), 100);
        assert_eq!(snapshot.dynamic_free_bytes(), 200);
        assert_eq!(snapshot.pending_growth_bytes(), 20);
        assert_eq!(snapshot.quarantined_bytes(), 10);
    }

    #[test]
    fn rejects_incoherent_capacity_evidence() {
        assert!(PlanRuntimeResourceSnapshot::new(1_000, 1_001, 0, 0, 0, 0, 0, 0, 0).is_err());
        assert!(PlanRuntimeResourceSnapshot::new(1_000, 900, 901, 0, 0, 0, 0, 0, 0).is_err());
        assert!(PlanRuntimeResourceSnapshot::new(1_000, 900, 500, 501, 0, 0, 0, 0, 0).is_err());
        assert!(PlanRuntimeResourceSnapshot::new(1_000, 900, 100, 100, 0, 100, 101, 0, 0).is_err());
        assert!(PlanRuntimeResourceSnapshot::new(1_000, 900, 500, 500, 400, 100, 0, 0, 1).is_err());
    }
}

/// Borrowed, already-tokenized input used to probe plan-runtime prefill
/// admission before the request can enter a device submission batch.
///
/// This carries semantic token identity rather than an aggregate token count:
/// vNext derives the exact resource work shape and its fingerprint from this
/// boundary. The request remains owned by the scheduler while the executor
/// retains any admitted authority internally until [`ModelExecutor::prefill`]
/// consumes it or cancellation releases it.
#[derive(Debug, Clone, Copy)]
pub struct ExecutorPrefillAdmission<'a> {
    pub request_id: &'a RequestId,
    pub input_tokens: &'a [TokenId],
    pub maximum_sequence_tokens: usize,
}

impl<'a> ExecutorPrefillAdmission<'a> {
    pub const fn new(
        request_id: &'a RequestId,
        input_tokens: &'a [TokenId],
        maximum_sequence_tokens: usize,
    ) -> Self {
        Self {
            request_id,
            input_tokens,
            maximum_sequence_tokens,
        }
    }
}

/// Scheduler-visible proof that an executor retained request and sequence
/// authority for future scheduler-owned prefill chunks.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ExecutorPrefillAdmissionReceipt {
    pub request_id: RequestId,
}

/// Stable scheduler-facing projection of one plan-runtime capacity domain.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct ExecutorAdmissionEpochs {
    pub coordinator_id: NonZeroU64,
    pub release_epoch: u64,
    pub capacity_epoch: u64,
}

impl ExecutorAdmissionEpochs {
    pub const fn new(coordinator_id: NonZeroU64, release_epoch: u64, capacity_epoch: u64) -> Self {
        Self {
            coordinator_id,
            release_epoch,
            capacity_epoch,
        }
    }

    pub fn from_capacity(epochs: crate::vnext::CapacityEpochs) -> Self {
        Self::new(
            NonZeroU64::new(epochs.coordinator_id().get())
                .expect("core-issued admission coordinator ids are non-zero"),
            epochs.release_epoch(),
            epochs.capacity_epoch(),
        )
    }
}

type ExecutorCapacityWaitFuture =
    Pin<Box<dyn Future<Output = Result<ExecutorAdmissionEpochs>> + Send + 'static>>;

/// Type-erased, single-use registration for one plan-runtime capacity wait.
///
/// Registration is created synchronously so the executor can subscribe before
/// the engine releases its iteration lock. Awaiting it never grants resources;
/// it only returns fresh epochs that permit another authoritative admission
/// probe.
#[must_use = "capacity wait registrations must be awaited or explicitly dropped"]
pub struct ExecutorCapacityWaitRegistration {
    future: ExecutorCapacityWaitFuture,
}

impl ExecutorCapacityWaitRegistration {
    pub fn new<F>(future: F) -> Self
    where
        F: Future<Output = Result<ExecutorAdmissionEpochs>> + Send + 'static,
    {
        Self {
            future: Box::pin(future),
        }
    }

    pub async fn wait_for_change(self) -> Result<ExecutorAdmissionEpochs> {
        self.future.await
    }
}

/// Pre-submit runtime stage that could not acquire its exact dynamic capacity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ExecutorExecutionCapacityStage {
    SequenceExtension,
    StepAdmission,
    SubmissionWave,
}

/// Scheduler-visible proof that an execution attempt was not submitted and
/// must not be retried until one of its exact capacity sources changes.
///
/// This value owns no resource authority. The executor retains the committed
/// request/sequence authority and has already retired any unsubmitted step or
/// submission-wave authority before returning it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ExecutorExecutionCapacityDeferral {
    observed: ExecutorAdmissionEpochs,
    wait_condition: crate::vnext::CapacityWaitCondition,
    stage: ExecutorExecutionCapacityStage,
}

impl ExecutorExecutionCapacityDeferral {
    pub fn new(
        observed: ExecutorAdmissionEpochs,
        wait_condition: crate::vnext::CapacityWaitCondition,
        stage: ExecutorExecutionCapacityStage,
    ) -> Result<Self> {
        if wait_condition.coordinator_id().get() != observed.coordinator_id.get() {
            return Err(ferrum_types::FerrumError::request_validation(
                "executor execution deferral belongs to a different capacity coordinator",
            ));
        }
        Ok(Self {
            observed,
            wait_condition,
            stage,
        })
    }

    pub fn from_admission(
        deferred: &crate::vnext::AdmissionDeferred,
        stage: ExecutorExecutionCapacityStage,
    ) -> Result<Self> {
        if deferred.action() != crate::vnext::DeferredAction::WaitForRelease {
            return Err(ferrum_types::FerrumError::internal(
                "execution capacity deferral must be reduced to WaitForRelease before export",
            ));
        }
        Self::new(
            ExecutorAdmissionEpochs::from_capacity(deferred.epochs()),
            deferred.wait_condition().clone(),
            stage,
        )
    }

    pub const fn observed(&self) -> ExecutorAdmissionEpochs {
        self.observed
    }

    pub fn wait_condition(&self) -> &crate::vnext::CapacityWaitCondition {
        &self.wait_condition
    }

    pub const fn stage(&self) -> ExecutorExecutionCapacityStage {
        self.stage
    }
}

/// Capacity-aware batch decode result.
///
/// `Deferred` is only legal before provider encode or device submission. All
/// possibly-submitted failures remain ordinary errors and retain their typed
/// fence/recovery authority inside the executor.
pub enum ExecutorBatchDecodeOutcome {
    Completed(Vec<DecodeOutput>),
    Deferred(ExecutorExecutionCapacityDeferral),
}

/// Capacity-aware result for one exact prefill chunk.
///
/// `Deferred` is only legal before provider encode or device submission. The
/// executor retains request/sequence authority so the scheduler can park the
/// same chunk until one of the exact capacity sources changes.
pub enum ExecutorPrefillOutcome {
    Completed(PrefillOutput),
    Deferred(ExecutorExecutionCapacityDeferral),
}

/// Stage that must advance before a plan-runtime prefill can be admitted.
///
/// This is scheduler evidence, not allocator authority. The executor retains
/// the sealed logical or physical deferral that authorizes maintenance.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ExecutorPrefillMaintenanceStage {
    LogicalCapacity,
    PhysicalBacking,
}

/// Scheduler-visible reason that a prefill needs plan-runtime backing
/// maintenance. These values are projections only and cannot allocate memory.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "source", rename_all = "snake_case")]
pub enum ExecutorPrefillMaintenanceBlocker {
    Capacity {
        domain_id: Option<u32>,
        kind: crate::vnext::CapacityShortfallKind,
        requested: u64,
        available: u64,
        current_total: u64,
        maximum_total: u64,
    },
    Backing {
        pool_id: String,
        domain_id: u32,
        lifetime: crate::vnext::AllocationLifetime,
        reason: crate::vnext::DynamicBackingDeferralReason,
        requested_bytes: u64,
        free_bytes: u64,
        largest_contiguous_bytes: u64,
    },
}

/// Non-authoritative projection of plan-runtime maintenance work.
///
/// The request id is the only handle returned to the engine. Implementations
/// must retain the sealed deferral internally and validate it again when
/// [`ModelExecutor::maintain_prefill_backing`] is called.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ExecutorPrefillMaintenanceDeferral {
    request_id: RequestId,
    observed: ExecutorAdmissionEpochs,
    wait_condition: crate::vnext::CapacityWaitCondition,
    stage: ExecutorPrefillMaintenanceStage,
    blockers: Vec<ExecutorPrefillMaintenanceBlocker>,
}

impl ExecutorPrefillMaintenanceDeferral {
    pub fn new(
        request_id: RequestId,
        observed: ExecutorAdmissionEpochs,
        wait_condition: crate::vnext::CapacityWaitCondition,
        stage: ExecutorPrefillMaintenanceStage,
        blockers: Vec<ExecutorPrefillMaintenanceBlocker>,
    ) -> Result<Self> {
        if blockers.is_empty() {
            return Err(ferrum_types::FerrumError::request_validation(
                "executor prefill maintenance deferral requires at least one blocker",
            ));
        }
        if wait_condition.coordinator_id().get() != observed.coordinator_id.get() {
            return Err(ferrum_types::FerrumError::request_validation(
                "executor prefill maintenance wait condition belongs to a different coordinator",
            ));
        }
        Ok(Self {
            request_id,
            observed,
            wait_condition,
            stage,
            blockers,
        })
    }

    pub fn from_admission(
        request_id: &RequestId,
        deferred: &crate::vnext::AdmissionDeferred,
    ) -> Result<Self> {
        if deferred.action() != crate::vnext::DeferredAction::AwaitBackingGrowth {
            return Err(ferrum_types::FerrumError::internal(
                "logical prefill maintenance projection requires AwaitBackingGrowth",
            ));
        }
        let blockers = deferred
            .blockers()
            .iter()
            .map(|blocker| ExecutorPrefillMaintenanceBlocker::Capacity {
                domain_id: blocker.domain().map(|domain| domain.get()),
                kind: blocker.kind(),
                requested: blocker.requested().get(),
                available: blocker.available().get(),
                current_total: blocker.current_total().get(),
                maximum_total: blocker.maximum_total().get(),
            })
            .collect();
        Self::new(
            request_id.clone(),
            ExecutorAdmissionEpochs::from_capacity(deferred.epochs()),
            deferred.wait_condition().clone(),
            ExecutorPrefillMaintenanceStage::LogicalCapacity,
            blockers,
        )
    }

    pub fn from_backing(
        request_id: &RequestId,
        deferred: &crate::vnext::DynamicBackingDeferred,
    ) -> Result<Self> {
        let blockers = deferred
            .blockers()
            .iter()
            .map(|blocker| ExecutorPrefillMaintenanceBlocker::Backing {
                pool_id: blocker.pool_id().as_str().to_string(),
                domain_id: blocker.domain_id().get(),
                lifetime: deferred.lifetime(),
                reason: blocker.reason(),
                requested_bytes: blocker.requested_bytes(),
                free_bytes: blocker.free_bytes(),
                largest_contiguous_bytes: blocker.largest_contiguous_bytes(),
            })
            .collect();
        Self::new(
            request_id.clone(),
            ExecutorAdmissionEpochs::from_capacity(deferred.epochs()),
            deferred.wait_condition().clone(),
            ExecutorPrefillMaintenanceStage::PhysicalBacking,
            blockers,
        )
    }

    pub fn request_id(&self) -> &RequestId {
        &self.request_id
    }

    pub const fn observed(&self) -> ExecutorAdmissionEpochs {
        self.observed
    }

    pub fn wait_condition(&self) -> &crate::vnext::CapacityWaitCondition {
        &self.wait_condition
    }

    pub const fn stage(&self) -> ExecutorPrefillMaintenanceStage {
        self.stage
    }

    pub fn blockers(&self) -> &[ExecutorPrefillMaintenanceBlocker] {
        &self.blockers
    }
}

/// Result of one bounded plan-runtime backing maintenance attempt.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "outcome", rename_all = "snake_case")]
pub enum ExecutorPrefillMaintenanceOutcome {
    /// Cancellation won the race before the maintenance task consumed the
    /// retained deferral.
    NoLongerPending,
    /// The physical allocator changed while maintenance was installing its
    /// wait predicate. The scheduler must clear the old backing deferral and
    /// perform one authoritative admission probe, even if publication of the
    /// corresponding capacity epoch is still in flight.
    RetryAdmission { current: ExecutorAdmissionEpochs },
    /// The requested backing is valid but cannot be installed while current
    /// device claims remain live. The scheduler must wait for release evidence
    /// rather than completing the request as an error.
    WaitForRelease {
        current: ExecutorAdmissionEpochs,
        wait_condition: crate::vnext::CapacityWaitCondition,
        pressure: crate::vnext::DeviceCapacityPressure,
    },
    /// The executor installed real backing and published the resulting
    /// capacity epoch.
    Maintained {
        current: ExecutorAdmissionEpochs,
        pools_grown: usize,
        allocated_bytes: u64,
    },
}

/// Typed result of probing plan-runtime prefill capacity.
///
/// `Deferred` and `MaintenanceDeferred` preserve the capacity domains and
/// epochs required by the plan-local dynamic admission queue. They must never
/// be flattened into a generic resource error at the scheduler boundary.
#[derive(Debug, Clone)]
pub enum ExecutorPrefillAdmissionDecision {
    Admitted(ExecutorPrefillAdmissionReceipt),
    Deferred(crate::vnext::AdmissionDeferred),
    MaintenanceDeferred(ExecutorPrefillMaintenanceDeferral),
    PermanentRejected(crate::vnext::AdmissionRejected),
}

/// Core model executor trait focusing on tensor operations
#[async_trait]
pub trait ModelExecutor: Send + Sync {
    /// Get model information and metadata
    fn info(&self) -> &ModelInfo;

    /// Selects the single authority for request-lifetime model resources.
    /// Existing executors remain on the transitional legacy-engine path by
    /// default. A runtime that returns `PlanRuntime` must return the shared
    /// runtime's opaque cache handle from prefill/decode and delegate release
    /// of that authority from `release_cache`.
    fn execution_resource_authority(&self) -> ExecutionResourceAuthority {
        ExecutionResourceAuthority::LegacyEngine
    }

    /// Returns the authoritative memory breakdown for a shared plan runtime.
    /// `LegacyEngine` executors return `None`; `PlanRuntime` executors must
    /// return `Some` while they are ready.
    fn plan_runtime_resource_snapshot(&self) -> Result<Option<PlanRuntimeResourceSnapshot>> {
        Ok(None)
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

    /// Installs the product-owned execution event sink before requests start.
    ///
    /// Legacy executors have no typed execution journal and keep the default
    /// no-op. Executors backed by the vNext runtime retain this sink with each
    /// admitted request so node/operation events share the product artifact.
    fn attach_execution_event_sink(&self, _sink: Arc<dyn crate::vnext::ExecutionEventSink>) {}

    /// Current plan-local capacity evidence for scheduler wake suppression.
    /// Legacy-engine executors return `None`; an executor declaring
    /// [`ExecutionResourceAuthority::PlanRuntime`] must return `Some`.
    fn execution_capacity_epochs(&self) -> Result<Option<ExecutorAdmissionEpochs>> {
        Ok(None)
    }

    /// Writes the canonical per-source availability generations into
    /// caller-owned storage and returns the matching global audit epochs.
    /// Executors with typed dynamic admission override this to avoid allocating
    /// on steady scheduler ticks.
    fn write_execution_capacity_snapshot(
        &self,
        availability: &mut Vec<crate::vnext::CapacityAvailabilityEpoch>,
    ) -> Result<Option<ExecutorAdmissionEpochs>> {
        availability.clear();
        self.execution_capacity_epochs()
    }

    /// Synchronously subscribes to every source named by one passive capacity
    /// wait. The returned registration must remain alive until it is awaited or
    /// deliberately cancelled by being dropped.
    ///
    /// Legacy-engine executors return `None`. An executor declaring
    /// [`ExecutionResourceAuthority::PlanRuntime`] must return `Some` for a
    /// wait condition issued by its own admission coordinator.
    fn register_execution_capacity_waiter(
        &self,
        _observed: &crate::vnext::CapacityWaitCondition,
    ) -> Result<Option<ExecutorCapacityWaitRegistration>> {
        Ok(None)
    }

    /// Probe and retain the exact request/sequence authority needed by a
    /// future prefill. No provider encode, kernel launch, or device submit may
    /// occur in this method.
    fn try_admit_prefill(
        &self,
        _input: ExecutorPrefillAdmission<'_>,
    ) -> Result<ExecutorPrefillAdmissionDecision> {
        Err(ferrum_types::FerrumError::unsupported(
            "plan-runtime prefill admission is not implemented",
        ))
    }

    /// Release an admitted but not yet active prefill authority.
    ///
    /// Returns true only when a retained authority was found and released.
    fn cancel_prefill_admission(&self, _request_id: &RequestId) -> bool {
        false
    }

    /// Consume one retained logical/physical backing deferral after the
    /// scheduler waiting lock has been released. Implementations must perform
    /// at most one bounded maintenance attempt and publish capacity epochs only
    /// after real backing is installed.
    fn maintain_prefill_backing(
        &self,
        _request_id: &RequestId,
    ) -> Result<ExecutorPrefillMaintenanceOutcome> {
        Err(ferrum_types::FerrumError::unsupported(
            "plan-runtime prefill backing maintenance is not implemented",
        ))
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

    /// Execute one exact prefill chunk with an explicit pre-submit capacity
    /// deferral edge. Legacy executors inherit full-prefill behavior.
    async fn prefill_with_capacity(&self, input: &PrefillInput) -> Result<ExecutorPrefillOutcome> {
        self.prefill(input)
            .await
            .map(ExecutorPrefillOutcome::Completed)
    }

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
    /// A successful result must contain exactly one output per input, in the
    /// original input order, and each output cache must retain the identity of
    /// its corresponding input cache. Implementations must not expose partial
    /// success as a shorter or reordered vector.
    ///
    /// The default implementation falls back to serial per-request `decode()`.
    /// Executors with a typed batch submission path should override this so one
    /// call maps to one resource step and one terminal submission fence.
    async fn batch_decode(&self, inputs: &[DecodeInput]) -> Result<Vec<DecodeOutput>> {
        let mut outputs = Vec::with_capacity(inputs.len());
        for input in inputs {
            outputs.push(self.decode(input).await?);
        }
        Ok(outputs)
    }

    /// Batch decode with an explicit pre-submit capacity deferral edge.
    ///
    /// Legacy executors inherit the successful/error-only behavior. A runtime
    /// with typed resource authority overrides this method so temporary
    /// capacity pressure is never flattened into a stringly resource error.
    async fn batch_decode_with_capacity(
        &self,
        inputs: &[DecodeInput],
    ) -> Result<ExecutorBatchDecodeOutcome> {
        self.batch_decode(inputs)
            .await
            .map(ExecutorBatchDecodeOutcome::Completed)
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
