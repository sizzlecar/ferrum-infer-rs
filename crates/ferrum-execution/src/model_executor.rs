//! Product execution through admitted token work and exact sequence lifecycles.
//!
//! Model implementations own numerical execution and physical resources; callers
//! use one admission, execution and close protocol without tensor/cache adapters.

use async_trait::async_trait;
use ferrum_interfaces::RecurrentStateSpec;
pub use ferrum_runtime::{ExecutorRequestOrigin, PlanRuntimeResourceSnapshot};
use ferrum_types::{ExecutorAdmissionLimits, FerrumError, ModelInfo, RequestId, Result, TokenId};
use serde::{Deserialize, Serialize};
use std::{
    collections::{hash_map::DefaultHasher, HashMap, HashSet},
    future::Future,
    hash::{Hash, Hasher},
    num::NonZeroU64,
    ops::Range,
    pin::Pin,
    sync::Arc,
};

mod prefill_shape;
mod prefix_capture;
mod prefix_restore;
pub use prefill_shape::{PrefillShapePlan, PrefillShapePolicy};
pub use prefix_capture::{
    PrefixCaptureBoundary, PrefixCaptureLease, PrefixCapturePlan, PrefixCaptureRequest,
    PrefixCaptureStatus,
};
pub use prefix_restore::{
    PlanRuntimePrefixRestoreDeferral, PlanRuntimePrefixRestoreInput,
    PlanRuntimePrefixRestoreOutcome, PlanRuntimePrefixRestoreOutput, PrefixRestoreAcknowledgement,
    PrefixRestoreDecision, PrefixRestoreObservation, PrefixRestoreSource,
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
        let fingerprint = Self::fingerprint(&valid_token_mask);
        Self {
            fingerprint,
            valid_token_mask: Arc::from(valid_token_mask),
        }
    }

    fn fingerprint(valid_token_mask: &[i8]) -> u64 {
        let mut hasher = DefaultHasher::new();
        valid_token_mask.hash(&mut hasher);
        hasher.finish()
    }

    /// Change a small set of token-validity slots and refresh the cache key
    /// once. `Arc::make_mut` keeps the common unshared request mask in place
    /// and preserves safety when an in-flight backend policy still owns a
    /// clone.
    pub fn set_tokens_validity(&mut self, token_ids: &[u32], valid: bool) -> bool {
        let value = i8::from(valid);
        let slots = Arc::make_mut(&mut self.valid_token_mask);
        let mut changed = false;
        for &token_id in token_ids {
            if let Some(slot) = slots.get_mut(token_id as usize) {
                if *slot != value {
                    *slot = value;
                    changed = true;
                }
            }
        }
        if changed {
            self.fingerprint = Self::fingerprint(slots);
        }
        changed
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

#[cfg(test)]
mod token_selection_mask_tests {
    use super::TokenSelectionMask;

    #[test]
    fn response_completion_mask_is_copy_on_write_and_restores_fingerprint() {
        let mut mask = TokenSelectionMask::new(vec![1, 1, 1]);
        let original = mask.clone();
        let original_fingerprint = mask.fingerprint;

        assert!(mask.set_tokens_validity(&[1], false));
        assert_eq!(mask.valid_token_mask.as_ref(), &[1, 0, 1]);
        assert_eq!(original.valid_token_mask.as_ref(), &[1, 1, 1]);
        assert_ne!(mask.fingerprint, original_fingerprint);

        let masked_fingerprint = mask.fingerprint;
        assert!(!mask.set_tokens_validity(&[1], false));
        assert_eq!(mask.fingerprint, masked_fingerprint);

        assert!(mask.set_tokens_validity(&[1], true));
        assert_eq!(mask.fingerprint, original_fingerprint);
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

/// Typed product output returned by a plan-runtime execution wave.
///
/// A selected token is not a one-element logits vector. Keeping the variants
/// distinct prevents the engine from inferring product semantics from tensor
/// shape and lets full-logits requests retain every host-side sampler,
/// grammar, and structured-output processor.
#[derive(Debug, Clone, PartialEq)]
pub enum ExecutorSamplingOutput {
    FullLogits(Vec<f32>),
    GreedyToken(TokenId),
    /// An authorized greedy selection found no finite logit among eligible
    /// tokens. This is a request-local sampling failure, not a token ID or a
    /// malformed execution-wave output.
    NoSelectableFiniteLogit,
}

impl ExecutorSamplingOutput {
    pub fn full_logits(logits: Vec<f32>) -> Result<Self> {
        if logits.is_empty() {
            return Err(FerrumError::backend(
                "plan-runtime sampling output requires non-empty logits",
            ));
        }
        Ok(Self::FullLogits(logits))
    }

    pub const fn greedy_token(token: TokenId) -> Self {
        Self::GreedyToken(token)
    }

    /// Validate that a device-selected token was explicitly authorized.
    ///
    /// Returning full logits for a greedy policy remains legal: heterogeneous
    /// batches may deliberately fall back to host sampling. The inverse would
    /// skip required product processors and therefore fails closed.
    pub fn validate_for_policy(
        &self,
        policy: &LogitsReturnPolicy,
        vocabulary_size: usize,
    ) -> Result<()> {
        match self {
            Self::FullLogits(logits) if logits.len() != vocabulary_size => {
                return Err(FerrumError::backend(format!(
                    "plan runtime returned {} logits for vocabulary {vocabulary_size}",
                    logits.len()
                )));
            }
            Self::GreedyToken(_) if policy.requires_full_logits() => {
                return Err(FerrumError::backend(
                    "plan runtime returned a greedy token for a full-logits request",
                ));
            }
            Self::NoSelectableFiniteLogit if policy.requires_full_logits() => {
                return Err(FerrumError::backend(
                    "plan runtime returned a greedy sampling failure for a full-logits request",
                ));
            }
            Self::GreedyToken(token)
                if usize::try_from(token.get())
                    .ok()
                    .is_none_or(|token| token >= vocabulary_size) =>
            {
                return Err(FerrumError::backend(format!(
                    "plan runtime returned token {} outside vocabulary {vocabulary_size}",
                    token.get()
                )));
            }
            _ => {}
        }
        Ok(())
    }

    pub fn into_full_logits(self) -> Result<Vec<f32>> {
        match self {
            Self::FullLogits(logits) => Ok(logits),
            Self::GreedyToken(_) => Err(FerrumError::backend(
                "plan-runtime prefill unexpectedly returned a selected token",
            )),
            Self::NoSelectableFiniteLogit => Err(FerrumError::backend(
                "plan-runtime prefill unexpectedly returned a greedy sampling failure",
            )),
        }
    }
}

#[cfg(test)]
mod executor_sampling_output_tests {
    use super::{ExecutorSamplingOutput, LogitsReturnPolicy};
    use ferrum_types::TokenId;

    #[test]
    fn full_logits_require_exact_vocabulary_width() {
        let output = ExecutorSamplingOutput::full_logits(vec![0.0; 4]).unwrap();
        assert!(output
            .validate_for_policy(&LogitsReturnPolicy::FullLogits, 4)
            .is_ok());
        assert!(output
            .validate_for_policy(&LogitsReturnPolicy::FullLogits, 5)
            .is_err());
    }

    #[test]
    fn greedy_token_requires_greedy_policy_and_in_vocabulary_token() {
        let allowed = LogitsReturnPolicy::GreedyArgmax {
            token_mask: None,
            repetition_penalty: None,
        };
        let output = ExecutorSamplingOutput::greedy_token(TokenId::new(3));
        assert!(output.validate_for_policy(&allowed, 4).is_ok());
        assert!(output.validate_for_policy(&allowed, 3).is_err());
        assert!(output
            .validate_for_policy(&LogitsReturnPolicy::FullLogits, 4)
            .is_err());
    }

    #[test]
    fn full_logits_are_a_legal_greedy_batch_fallback() {
        let policy = LogitsReturnPolicy::GreedyArgmax {
            token_mask: None,
            repetition_penalty: None,
        };
        let output = ExecutorSamplingOutput::full_logits(vec![0.0; 4]).unwrap();
        assert!(output.validate_for_policy(&policy, 4).is_ok());
    }

    #[test]
    fn no_selectable_finite_logit_requires_greedy_authorization() {
        let policy = LogitsReturnPolicy::GreedyArgmax {
            token_mask: None,
            repetition_penalty: None,
        };
        let output = ExecutorSamplingOutput::NoSelectableFiniteLogit;
        assert!(output.validate_for_policy(&policy, 4).is_ok());
        assert!(output
            .validate_for_policy(&LogitsReturnPolicy::FullLogits, 4)
            .is_err());
        assert!(output.into_full_logits().is_err());
    }
}

/// Sparse repetition-penalty metadata for model-side greedy argmax.
///
/// The token list is request-local and de-duplicated. Applying the penalty
/// before GPU argmax avoids downloading full `[batch, vocab]` logits for the
/// common greedy chat path while preserving repeat avoidance.
#[derive(Clone, Debug)]
pub struct GreedyRepetitionPenalty {
    penalty: f32,
    token_ids: Arc<[u32]>,
}

impl GreedyRepetitionPenalty {
    pub fn new(penalty: f32, mut token_ids: Vec<u32>) -> Self {
        let mut seen = HashSet::with_capacity(token_ids.len());
        token_ids.retain(|token| seen.insert(*token));
        Self {
            penalty,
            token_ids: Arc::from(token_ids),
        }
    }

    pub const fn penalty(&self) -> f32 {
        self.penalty
    }

    pub fn token_ids(&self) -> &[u32] {
        &self.token_ids
    }

    pub fn is_empty(&self) -> bool {
        self.token_ids.is_empty() || self.penalty == 1.0
    }
}

#[cfg(test)]
mod greedy_repetition_penalty_tests {
    use super::GreedyRepetitionPenalty;

    #[test]
    fn constructor_preserves_first_seen_order_and_removes_duplicates() {
        let repetition = GreedyRepetitionPenalty::new(1.1, vec![7, 3, 7, 9, 3]);
        assert_eq!(repetition.penalty(), 1.1);
        assert_eq!(repetition.token_ids(), [7, 3, 9]);
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

/// Product-authoritative evidence for a successfully completed sequence.
///
/// Physical cache release is not completion evidence: cancellation, failure,
/// recompute, and successful generation all release the same cache authority.
/// The engine constructs this receipt only after it has finalized user-visible
/// token usage, allowing plan runtimes to reconcile terminal events without
/// inferring output counts from execution frames.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExecutorSequenceCompletion {
    request_id: RequestId,
    cache_id: String,
    input_tokens: u64,
    output_tokens: u64,
}

impl ExecutorSequenceCompletion {
    pub fn new(
        request_id: RequestId,
        cache_id: String,
        input_tokens: usize,
        output_tokens: usize,
    ) -> Result<Self> {
        if cache_id.is_empty() {
            return Err(FerrumError::request_validation(
                "executor sequence completion requires a cache identity",
            ));
        }
        let input_tokens = u64::try_from(input_tokens).map_err(|_| {
            FerrumError::request_validation("executor completion input token count exceeds u64")
        })?;
        let output_tokens = u64::try_from(output_tokens).map_err(|_| {
            FerrumError::request_validation("executor completion output token count exceeds u64")
        })?;
        Ok(Self {
            request_id,
            cache_id,
            input_tokens,
            output_tokens,
        })
    }

    pub fn request_id(&self) -> &RequestId {
        &self.request_id
    }

    pub fn cache_id(&self) -> &str {
        &self.cache_id
    }

    pub const fn input_tokens(&self) -> u64 {
        self.input_tokens
    }

    pub const fn output_tokens(&self) -> u64 {
        self.output_tokens
    }
}

pub use ferrum_types::ExecutionResourceAuthority;

/// Request-scoped authority selected for a capacity-pressure preemption.
///
/// The cache identity prevents the engine from releasing a newer sequence
/// incarnation after a stale scheduler decision. Implementations must retire
/// retained prefill and active decode authority through the same operation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExecutorExecutionCapacityPreemption {
    request_id: RequestId,
    cache_id: String,
}

impl ExecutorExecutionCapacityPreemption {
    pub fn new(request_id: RequestId, cache_id: String) -> Result<Self> {
        if cache_id.is_empty() {
            return Err(FerrumError::request_validation(
                "execution-capacity preemption requires a cache identity",
            ));
        }
        Ok(Self {
            request_id,
            cache_id,
        })
    }

    pub fn request_id(&self) -> &RequestId {
        &self.request_id
    }

    pub fn cache_id(&self) -> &str {
        &self.cache_id
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExecutorExecutionCapacityPreemptionAuthority {
    RetainedPrefill,
    ActiveSequence,
}

/// Proof that the executor retired one exact request authority to a terminal
/// state. Source-generation advancement remains independently verified by the
/// engine before the scheduler may resume another frontier.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExecutorExecutionCapacityPreemptionReceipt {
    request_id: RequestId,
    cache_id: String,
    authority: ExecutorExecutionCapacityPreemptionAuthority,
}

impl ExecutorExecutionCapacityPreemptionReceipt {
    pub fn new(
        request_id: RequestId,
        cache_id: String,
        authority: ExecutorExecutionCapacityPreemptionAuthority,
    ) -> Self {
        Self {
            request_id,
            cache_id,
            authority,
        }
    }

    pub fn request_id(&self) -> &RequestId {
        &self.request_id
    }

    pub fn cache_id(&self) -> &str {
        &self.cache_id
    }

    pub const fn authority(&self) -> ExecutorExecutionCapacityPreemptionAuthority {
        self.authority
    }
}

/// Borrowed, already-tokenized input used to probe plan-runtime prefill
/// admission before the request can enter a device submission batch.
///
/// This carries semantic token identity rather than an aggregate token count:
/// vNext derives the exact resource work shape and its fingerprint from this
/// boundary. The request remains owned by the scheduler while the executor
/// retains the admitted sequence internally for [`ModelExecutor::execute_tokens`]
/// until its exact token lifecycle reaches a terminal state.
#[derive(Debug, Clone, Copy)]
pub struct ExecutorPrefillAdmission<'a> {
    pub request_id: &'a RequestId,
    pub input_tokens: &'a [TokenId],
    pub maximum_sequence_tokens: usize,
    /// Prompt tokens reported in the final product usage.
    pub product_prompt_tokens: usize,
    /// Already-committed output tokens replayed as part of a recompute input.
    pub replayed_output_tokens: usize,
    pub request_origin: ExecutorRequestOrigin,
}

impl<'a> ExecutorPrefillAdmission<'a> {
    pub const fn for_startup(
        request_id: &'a RequestId,
        input_tokens: &'a [TokenId],
        maximum_sequence_tokens: usize,
    ) -> Self {
        Self {
            request_id,
            input_tokens,
            maximum_sequence_tokens,
            product_prompt_tokens: input_tokens.len(),
            replayed_output_tokens: 0,
            request_origin: ExecutorRequestOrigin::Startup,
        }
    }

    pub const fn for_diagnostic(
        request_id: &'a RequestId,
        input_tokens: &'a [TokenId],
        maximum_sequence_tokens: usize,
    ) -> Self {
        Self {
            request_id,
            input_tokens,
            maximum_sequence_tokens,
            product_prompt_tokens: input_tokens.len(),
            replayed_output_tokens: 0,
            request_origin: ExecutorRequestOrigin::Diagnostic,
        }
    }

    /// Construct admission for a product request whose execution context may
    /// include output tokens replayed after preemption.
    pub fn for_product_request(
        request_id: &'a RequestId,
        input_tokens: &'a [TokenId],
        maximum_sequence_tokens: usize,
        product_prompt_tokens: usize,
        replayed_output_tokens: usize,
    ) -> Result<Self> {
        let admission = Self {
            request_id,
            input_tokens,
            maximum_sequence_tokens,
            product_prompt_tokens,
            replayed_output_tokens,
            request_origin: ExecutorRequestOrigin::Product,
        };
        admission.validate()?;
        Ok(admission)
    }

    pub fn validate(&self) -> Result<()> {
        if self.input_tokens.is_empty() {
            return Err(FerrumError::request_validation(
                "executor prefill admission requires at least one execution-context token",
            ));
        }
        if self.product_prompt_tokens == 0 {
            return Err(FerrumError::request_validation(
                "executor prefill admission requires at least one product prompt token",
            ));
        }
        let execution_context_tokens = self
            .product_prompt_tokens
            .checked_add(self.replayed_output_tokens)
            .ok_or_else(|| {
                FerrumError::request_validation(
                    "executor prefill product token accounting exceeds usize",
                )
            })?;
        if execution_context_tokens != self.input_tokens.len() {
            return Err(FerrumError::request_validation(format!(
                "executor prefill execution context has {} tokens but product accounting declares {} prompt + {} replayed output",
                self.input_tokens.len(),
                self.product_prompt_tokens,
                self.replayed_output_tokens
            )));
        }
        if self.maximum_sequence_tokens < execution_context_tokens {
            return Err(FerrumError::request_validation(format!(
                "executor prefill sequence ceiling {} does not cover execution context {execution_context_tokens}",
                self.maximum_sequence_tokens
            )));
        }
        Ok(())
    }
}

#[cfg(test)]
mod executor_prefill_admission_tests {
    use super::{ExecutorPrefillAdmission, ExecutorRequestOrigin};
    use ferrum_types::{RequestId, TokenId};

    #[test]
    fn product_accounting_distinguishes_replayed_output_from_prompt() {
        let request_id = RequestId::new();
        let tokens = [1, 2, 3, 4, 5]
            .into_iter()
            .map(TokenId::new)
            .collect::<Vec<_>>();

        let admission =
            ExecutorPrefillAdmission::for_product_request(&request_id, &tokens, 8, 3, 2)
                .expect("recompute accounting must be accepted");

        assert_eq!(admission.product_prompt_tokens, 3);
        assert_eq!(admission.replayed_output_tokens, 2);
        assert_eq!(admission.request_origin, ExecutorRequestOrigin::Product);
        assert_eq!(
            ExecutorPrefillAdmission::for_startup(&request_id, &tokens, 8).request_origin,
            ExecutorRequestOrigin::Startup
        );
        assert_eq!(
            ExecutorPrefillAdmission::for_diagnostic(&request_id, &tokens, 8).request_origin,
            ExecutorRequestOrigin::Diagnostic
        );
    }

    #[test]
    fn product_accounting_rejects_context_drift_and_short_ceiling() {
        let request_id = RequestId::new();
        let tokens = [1, 2, 3].into_iter().map(TokenId::new).collect::<Vec<_>>();

        assert!(
            ExecutorPrefillAdmission::for_product_request(&request_id, &tokens, 3, 2, 0).is_err()
        );
        assert!(
            ExecutorPrefillAdmission::for_product_request(&request_id, &tokens, 2, 2, 1).is_err()
        );
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

    pub fn from_capacity(epochs: ferrum_runtime::CapacityEpochs) -> Self {
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

/// One exact physical pool mutation completed while an executor was trying to
/// make an unsubmitted frontier runnable.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ExecutorExecutionMaintenanceMutation {
    pool_id: ferrum_runtime::DynamicBackingPoolId,
    domain_id: ferrum_runtime::CapacityDomainId,
    chunk: ferrum_runtime::BackingChunkIdentity,
    chunk_bytes: u64,
    published_capacity_bytes: u64,
    capacity_epoch: u64,
}

impl ExecutorExecutionMaintenanceMutation {
    pub fn pool_id(&self) -> &ferrum_runtime::DynamicBackingPoolId {
        &self.pool_id
    }

    pub const fn domain_id(&self) -> ferrum_runtime::CapacityDomainId {
        self.domain_id
    }

    pub fn chunk(&self) -> &ferrum_runtime::BackingChunkIdentity {
        &self.chunk
    }

    pub const fn chunk_bytes(&self) -> u64 {
        self.chunk_bytes
    }

    pub const fn published_capacity_bytes(&self) -> u64 {
        self.published_capacity_bytes
    }

    pub const fn capacity_epoch(&self) -> u64 {
        self.capacity_epoch
    }
}

/// Typed proof that a bounded executor call committed relevant backing growth
/// before yielding the same logical frontier back to the scheduler.
///
/// This is deliberately stronger than observing a global capacity epoch. Every
/// mutation is reconstructed from a real growth receipt and bound back to the
/// pool's exact capacity domain. The scheduler may grant one fairness-bounded
/// retry only when this proof is present.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ExecutorExecutionMaintenanceProgress {
    attempts: u32,
    coordinator_id: NonZeroU64,
    mutations: Vec<ExecutorExecutionMaintenanceMutation>,
    latest_capacity_epoch: u64,
}

impl ExecutorExecutionMaintenanceProgress {
    pub fn from_growth_receipts(
        attempts: u32,
        observed: ExecutorAdmissionEpochs,
        receipts: &[ferrum_runtime::DynamicPoolGrowthBatchReceipt],
        pools: &[ferrum_runtime::DynamicPoolStatus],
    ) -> Result<Self> {
        if attempts == 0 || receipts.is_empty() || receipts.len() > attempts as usize {
            return Err(FerrumError::internal(
                "execution maintenance retry requires bounded, non-empty growth receipts",
            ));
        }

        let mut mutations = Vec::new();
        let mut previous_capacity_epoch = None;
        for receipt in receipts {
            if receipt.coordinator_id().get() != observed.coordinator_id.get() {
                return Err(FerrumError::internal(
                    "execution maintenance receipt belongs to another capacity coordinator",
                ));
            }
            if receipt.growths().is_empty()
                || previous_capacity_epoch
                    .is_some_and(|previous| receipt.capacity_epoch() <= previous)
            {
                return Err(FerrumError::internal(
                    "execution maintenance receipts contain no new ordered capacity mutation",
                ));
            }
            previous_capacity_epoch = Some(receipt.capacity_epoch());

            for growth in receipt.growths() {
                let pool = pools
                    .iter()
                    .find(|pool| pool.pool_id() == growth.pool_id())
                    .ok_or_else(|| {
                        FerrumError::internal(
                            "execution maintenance receipt references an unknown dynamic pool",
                        )
                    })?;
                if growth.chunk().pool_id() != growth.pool_id()
                    || growth.chunk_bytes() == 0
                    || growth.published_capacity_bytes() == 0
                    || growth.capacity_epoch() != receipt.capacity_epoch()
                {
                    return Err(FerrumError::internal(
                        "execution maintenance receipt contains an invalid pool mutation",
                    ));
                }
                if mutations
                    .iter()
                    .any(|mutation: &ExecutorExecutionMaintenanceMutation| {
                        mutation.pool_id() == growth.pool_id() && mutation.chunk() == growth.chunk()
                    })
                {
                    return Err(FerrumError::internal(
                        "execution maintenance receipts repeat one physical pool mutation",
                    ));
                }
                mutations.push(ExecutorExecutionMaintenanceMutation {
                    pool_id: growth.pool_id().clone(),
                    domain_id: pool.domain_id(),
                    chunk: growth.chunk().clone(),
                    chunk_bytes: growth.chunk_bytes(),
                    published_capacity_bytes: growth.published_capacity_bytes(),
                    capacity_epoch: growth.capacity_epoch(),
                });
            }
        }

        let latest_capacity_epoch = previous_capacity_epoch.expect("receipts are non-empty");
        if latest_capacity_epoch > observed.capacity_epoch {
            return Err(FerrumError::internal(
                "execution maintenance receipt is newer than the exported capacity observation",
            ));
        }
        mutations.sort_by(|left, right| {
            (
                left.capacity_epoch,
                left.pool_id.as_str(),
                left.chunk.ordinal(),
                left.chunk.generation(),
            )
                .cmp(&(
                    right.capacity_epoch,
                    right.pool_id.as_str(),
                    right.chunk.ordinal(),
                    right.chunk.generation(),
                ))
        });
        Ok(Self {
            attempts,
            coordinator_id: observed.coordinator_id,
            mutations,
            latest_capacity_epoch,
        })
    }

    pub const fn attempts(&self) -> u32 {
        self.attempts
    }

    pub const fn coordinator_id(&self) -> NonZeroU64 {
        self.coordinator_id
    }

    pub fn mutations(&self) -> &[ExecutorExecutionMaintenanceMutation] {
        &self.mutations
    }

    pub const fn latest_capacity_epoch(&self) -> u64 {
        self.latest_capacity_epoch
    }
}

/// Scheduler-consumable proof binding physical maintenance to the exact
/// logical frontiers whose unsubmitted execution attempt observed it.
///
/// Keeping the request scope inside the proof prevents a caller from attaching
/// one sequence's growth receipt to an unrelated decode cohort.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ExecutorExecutionMaintenanceRetry {
    affected_request_ids: Vec<RequestId>,
    progress: ExecutorExecutionMaintenanceProgress,
}

impl ExecutorExecutionMaintenanceRetry {
    fn new(
        affected_request_ids: Vec<RequestId>,
        progress: ExecutorExecutionMaintenanceProgress,
    ) -> Result<Self> {
        let unique = affected_request_ids.iter().collect::<HashSet<_>>();
        if affected_request_ids.is_empty() || unique.len() != affected_request_ids.len() {
            return Err(FerrumError::internal(
                "execution maintenance retry requires unique affected requests",
            ));
        }
        if progress.mutations().is_empty() {
            return Err(FerrumError::internal(
                "execution maintenance retry requires physical mutations",
            ));
        }
        Ok(Self {
            affected_request_ids,
            progress,
        })
    }

    pub fn affected_request_ids(&self) -> &[RequestId] {
        &self.affected_request_ids
    }

    pub const fn progress(&self) -> &ExecutorExecutionMaintenanceProgress {
        &self.progress
    }
}

/// Scheduler-visible proof that an execution attempt was not submitted and
/// must not be retried until one of its exact capacity sources changes.
///
/// This value owns no resource authority. The executor retains the committed
/// request/sequence authority and has already retired any unsubmitted step or
/// submission-wave authority before returning it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ExecutorExecutionCapacityEvidenceOwner {
    Logical,
    Backing,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum ExecutorExecutionCapacityEvidenceKind {
    Logical {
        shortfalls: Vec<ferrum_runtime::CapacityShortfall>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pressure: Option<ferrum_runtime::DynamicBackingPressure>,
    },
    BackingDeferred {
        blockers: Vec<ferrum_runtime::DynamicBackingBlocker>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pressure: Option<ferrum_runtime::DynamicBackingPressure>,
    },
    BackingPressure {
        pressure: ferrum_runtime::DynamicBackingPressure,
    },
}

/// Exactly one typed owner for an execution-capacity deferral.
///
/// Its fields are private so callers cannot construct an empty or ambiguous
/// logical/backing evidence combination.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ExecutorExecutionCapacityEvidence {
    owner: ExecutorExecutionCapacityEvidenceOwner,
    #[serde(flatten)]
    kind: ExecutorExecutionCapacityEvidenceKind,
    #[serde(skip_serializing_if = "Option::is_none")]
    maintenance_boundary: Option<ferrum_runtime::DynamicPoolMaintenanceBoundaryReceipt>,
}

impl ExecutorExecutionCapacityEvidence {
    fn logical(shortfalls: Vec<ferrum_runtime::CapacityShortfall>) -> Result<Self> {
        Self::logical_with_pressure(shortfalls, None)
    }

    fn logical_with_pressure(
        shortfalls: Vec<ferrum_runtime::CapacityShortfall>,
        pressure: Option<ferrum_runtime::DynamicBackingPressure>,
    ) -> Result<Self> {
        if shortfalls.is_empty() {
            return Err(FerrumError::internal(
                "logical execution deferral requires at least one shortfall",
            ));
        }
        Ok(Self {
            owner: ExecutorExecutionCapacityEvidenceOwner::Logical,
            kind: ExecutorExecutionCapacityEvidenceKind::Logical {
                shortfalls,
                pressure,
            },
            maintenance_boundary: None,
        })
    }

    fn backing_deferred(blockers: Vec<ferrum_runtime::DynamicBackingBlocker>) -> Result<Self> {
        Self::backing_deferred_with_pressure(blockers, None)
    }

    fn backing_deferred_with_pressure(
        blockers: Vec<ferrum_runtime::DynamicBackingBlocker>,
        pressure: Option<ferrum_runtime::DynamicBackingPressure>,
    ) -> Result<Self> {
        if blockers.is_empty() {
            return Err(FerrumError::internal(
                "physical execution deferral requires at least one backing blocker",
            ));
        }
        Ok(Self {
            owner: ExecutorExecutionCapacityEvidenceOwner::Backing,
            kind: ExecutorExecutionCapacityEvidenceKind::BackingDeferred { blockers, pressure },
            maintenance_boundary: None,
        })
    }

    fn direct_backing_pressure(pressure: ferrum_runtime::DynamicBackingPressure) -> Self {
        Self {
            owner: ExecutorExecutionCapacityEvidenceOwner::Backing,
            kind: ExecutorExecutionCapacityEvidenceKind::BackingPressure { pressure },
            maintenance_boundary: None,
        }
    }

    pub const fn owner(&self) -> ExecutorExecutionCapacityEvidenceOwner {
        self.owner
    }

    pub fn shortfalls(&self) -> &[ferrum_runtime::CapacityShortfall] {
        match &self.kind {
            ExecutorExecutionCapacityEvidenceKind::Logical { shortfalls, .. } => shortfalls,
            ExecutorExecutionCapacityEvidenceKind::BackingDeferred { .. }
            | ExecutorExecutionCapacityEvidenceKind::BackingPressure { .. } => &[],
        }
    }

    pub fn backing_blockers(&self) -> &[ferrum_runtime::DynamicBackingBlocker] {
        match &self.kind {
            ExecutorExecutionCapacityEvidenceKind::BackingDeferred { blockers, .. } => blockers,
            ExecutorExecutionCapacityEvidenceKind::Logical { .. }
            | ExecutorExecutionCapacityEvidenceKind::BackingPressure { .. } => &[],
        }
    }

    pub const fn backing_pressure(&self) -> Option<&ferrum_runtime::DynamicBackingPressure> {
        match &self.kind {
            ExecutorExecutionCapacityEvidenceKind::Logical { pressure, .. }
            | ExecutorExecutionCapacityEvidenceKind::BackingDeferred { pressure, .. } => {
                pressure.as_ref()
            }
            ExecutorExecutionCapacityEvidenceKind::BackingPressure { pressure } => Some(pressure),
        }
    }

    pub const fn maintenance_boundary(
        &self,
    ) -> Option<&ferrum_runtime::DynamicPoolMaintenanceBoundaryReceipt> {
        self.maintenance_boundary.as_ref()
    }

    fn with_maintenance_boundary(
        mut self,
        boundary: Option<ferrum_runtime::DynamicPoolMaintenanceBoundaryReceipt>,
    ) -> Result<Self> {
        match (self.backing_pressure(), boundary.as_ref()) {
            (Some(pressure), Some(boundary))
                if pressure == boundary.pressure() && !boundary.reclaim_sufficient() => {}
            (Some(ferrum_runtime::DynamicBackingPressure::PoolResident(_)), None)
            | (None, None) => {}
            (Some(ferrum_runtime::DynamicBackingPressure::DeviceCapacity(_)), None) => {
                return Err(FerrumError::internal(
                    "device-capacity execution maintenance lost its boundary receipt",
                ));
            }
            _ => {
                return Err(FerrumError::internal(
                    "execution maintenance boundary differs from its blocked pressure",
                ));
            }
        }
        self.maintenance_boundary = boundary;
        Ok(self)
    }

    fn has_relevant_mutation(&self, mutation: &ExecutorExecutionMaintenanceMutation) -> bool {
        let logical_matches = |shortfalls: &[ferrum_runtime::CapacityShortfall]| {
            shortfalls.iter().any(|shortfall| {
                shortfall.kind() == ferrum_runtime::CapacityShortfallKind::BackingGrowthRequired
                    && shortfall.domain() == Some(mutation.domain_id())
            })
        };
        let backing_matches = |blockers: &[ferrum_runtime::DynamicBackingBlocker]| {
            blockers.iter().any(|blocker| {
                blocker.pool_id() == mutation.pool_id()
                    && blocker.domain_id() == mutation.domain_id()
            })
        };
        match &self.kind {
            ExecutorExecutionCapacityEvidenceKind::Logical { shortfalls, .. } => {
                logical_matches(shortfalls)
            }
            ExecutorExecutionCapacityEvidenceKind::BackingDeferred { blockers, .. } => {
                backing_matches(blockers)
            }
            ExecutorExecutionCapacityEvidenceKind::BackingPressure { .. } => false,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ExecutorExecutionCapacityDeferral {
    observed: ExecutorAdmissionEpochs,
    wait_condition: ferrum_runtime::CapacityWaitCondition,
    stage: ExecutorExecutionCapacityStage,
    evidence: ExecutorExecutionCapacityEvidence,
    #[serde(skip_serializing_if = "Option::is_none")]
    maintenance_retry: Option<ExecutorExecutionMaintenanceRetry>,
}

impl ExecutorExecutionCapacityDeferral {
    fn with_evidence(
        observed: ExecutorAdmissionEpochs,
        wait_condition: ferrum_runtime::CapacityWaitCondition,
        stage: ExecutorExecutionCapacityStage,
        evidence: ExecutorExecutionCapacityEvidence,
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
            evidence,
            maintenance_retry: None,
        })
    }

    /// Construct a physical deferral for an executor that observes device or
    /// pool pressure directly rather than through Ferrum's backing allocator.
    pub fn from_backing_pressure(
        observed: ExecutorAdmissionEpochs,
        wait_condition: ferrum_runtime::CapacityWaitCondition,
        pressure: ferrum_runtime::DynamicBackingPressure,
        stage: ExecutorExecutionCapacityStage,
    ) -> Result<Self> {
        Self::with_evidence(
            observed,
            wait_condition,
            stage,
            ExecutorExecutionCapacityEvidence::direct_backing_pressure(pressure),
        )
    }

    pub fn from_admission(
        deferred: &ferrum_runtime::AdmissionDeferred,
        stage: ExecutorExecutionCapacityStage,
    ) -> Result<Self> {
        if deferred.action() != ferrum_runtime::DeferredAction::WaitForRelease {
            return Err(ferrum_types::FerrumError::internal(
                "execution capacity deferral must be reduced to WaitForRelease before export",
            ));
        }
        let evidence = ExecutorExecutionCapacityEvidence::logical(deferred.blockers().to_vec())?;
        Self::with_evidence(
            ExecutorAdmissionEpochs::from_capacity(deferred.epochs()),
            deferred.wait_condition().clone(),
            stage,
            evidence,
        )
    }

    /// Export an unresolved logical backing-growth decision after the
    /// executor's bounded in-call maintenance attempts are exhausted.
    ///
    /// This remains a pre-submit scheduling deferral. The bound controls only
    /// how much allocator maintenance one executor call may perform; it must
    /// not turn temporary capacity pressure into a terminal request failure.
    pub fn from_pending_maintenance(
        deferred: &ferrum_runtime::AdmissionDeferred,
        stage: ExecutorExecutionCapacityStage,
    ) -> Result<Self> {
        if deferred.action() != ferrum_runtime::DeferredAction::AwaitBackingGrowth {
            return Err(ferrum_types::FerrumError::internal(
                "pending execution maintenance must await backing growth",
            ));
        }
        let evidence = ExecutorExecutionCapacityEvidence::logical(deferred.blockers().to_vec())?;
        Self::with_evidence(
            ExecutorAdmissionEpochs::from_capacity(deferred.epochs()),
            deferred.wait_condition().clone(),
            stage,
            evidence,
        )
    }

    /// Export unresolved physical backing pressure without discarding its
    /// exact pool/domain evidence.
    pub fn from_backing(
        deferred: &ferrum_runtime::DynamicBackingDeferred,
        stage: ExecutorExecutionCapacityStage,
    ) -> Result<Self> {
        let evidence =
            ExecutorExecutionCapacityEvidence::backing_deferred(deferred.blockers().to_vec())?;
        Self::with_evidence(
            ExecutorAdmissionEpochs::from_capacity(deferred.epochs()),
            deferred.wait_condition().clone(),
            stage,
            evidence,
        )
    }

    /// Attach a scheduler retry only when this call produced at least one real
    /// physical mutation relevant to the final blocker. An empty or unrelated
    /// receipt set is an ordinary typed wait, not an internal error.
    pub fn with_relevant_maintenance_retry(
        mut self,
        attempts: u32,
        receipts: &[ferrum_runtime::DynamicPoolGrowthBatchReceipt],
        pools: &[ferrum_runtime::DynamicPoolStatus],
        affected_request_ids: Vec<RequestId>,
    ) -> Result<Self> {
        if receipts.is_empty() {
            return Ok(self);
        }
        let progress = ExecutorExecutionMaintenanceProgress::from_growth_receipts(
            attempts,
            self.observed,
            receipts,
            pools,
        )?;
        if progress.coordinator_id() != self.observed.coordinator_id
            || progress.latest_capacity_epoch() > self.observed.capacity_epoch
            || progress.mutations().is_empty()
        {
            return Err(FerrumError::internal(
                "execution maintenance progress does not match the exported deferral",
            ));
        }
        let relevant_mutation = progress
            .mutations()
            .iter()
            .any(|mutation| self.evidence.has_relevant_mutation(mutation));
        if !relevant_mutation {
            return Ok(self);
        }
        let retry = ExecutorExecutionMaintenanceRetry::new(affected_request_ids, progress)?;
        if self.stage == ExecutorExecutionCapacityStage::SequenceExtension
            && retry.affected_request_ids().len() != 1
        {
            return Err(FerrumError::internal(
                "sequence-extension maintenance retry must affect exactly one request",
            ));
        }
        self.maintenance_retry = Some(retry);
        Ok(self)
    }

    pub fn from_admission_maintenance(
        source: &ferrum_runtime::AdmissionDeferred,
        observed: ExecutorAdmissionEpochs,
        wait_condition: ferrum_runtime::CapacityWaitCondition,
        pressure: ferrum_runtime::DynamicBackingPressure,
        maintenance_boundary: Option<ferrum_runtime::DynamicPoolMaintenanceBoundaryReceipt>,
        stage: ExecutorExecutionCapacityStage,
    ) -> Result<Self> {
        if source.action() != ferrum_runtime::DeferredAction::AwaitBackingGrowth {
            return Err(ferrum_types::FerrumError::internal(
                "execution maintenance source must await backing growth",
            ));
        }
        let evidence = ExecutorExecutionCapacityEvidence::logical_with_pressure(
            source.blockers().to_vec(),
            Some(pressure),
        )?
        .with_maintenance_boundary(maintenance_boundary)?;
        if evidence.maintenance_boundary().is_some_and(|boundary| {
            boundary.coordinator_id().get() != observed.coordinator_id.get()
        }) {
            return Err(FerrumError::internal(
                "execution maintenance boundary belongs to another coordinator",
            ));
        }
        Self::with_evidence(observed, wait_condition, stage, evidence)
    }

    pub fn from_backing_maintenance(
        source: &ferrum_runtime::DynamicBackingDeferred,
        observed: ExecutorAdmissionEpochs,
        wait_condition: ferrum_runtime::CapacityWaitCondition,
        pressure: ferrum_runtime::DynamicBackingPressure,
        maintenance_boundary: Option<ferrum_runtime::DynamicPoolMaintenanceBoundaryReceipt>,
        stage: ExecutorExecutionCapacityStage,
    ) -> Result<Self> {
        let evidence = ExecutorExecutionCapacityEvidence::backing_deferred_with_pressure(
            source.blockers().to_vec(),
            Some(pressure),
        )?
        .with_maintenance_boundary(maintenance_boundary)?;
        if evidence.maintenance_boundary().is_some_and(|boundary| {
            boundary.coordinator_id().get() != observed.coordinator_id.get()
        }) {
            return Err(FerrumError::internal(
                "execution maintenance boundary belongs to another coordinator",
            ));
        }
        Self::with_evidence(observed, wait_condition, stage, evidence)
    }

    pub const fn observed(&self) -> ExecutorAdmissionEpochs {
        self.observed
    }

    pub fn wait_condition(&self) -> &ferrum_runtime::CapacityWaitCondition {
        &self.wait_condition
    }

    pub const fn stage(&self) -> ExecutorExecutionCapacityStage {
        self.stage
    }

    pub const fn evidence(&self) -> &ExecutorExecutionCapacityEvidence {
        &self.evidence
    }

    pub fn shortfalls(&self) -> &[ferrum_runtime::CapacityShortfall] {
        self.evidence.shortfalls()
    }

    pub fn backing_blockers(&self) -> &[ferrum_runtime::DynamicBackingBlocker] {
        self.evidence.backing_blockers()
    }

    pub const fn backing_pressure(&self) -> Option<&ferrum_runtime::DynamicBackingPressure> {
        self.evidence.backing_pressure()
    }

    pub const fn maintenance_boundary(
        &self,
    ) -> Option<&ferrum_runtime::DynamicPoolMaintenanceBoundaryReceipt> {
        self.evidence.maintenance_boundary()
    }

    pub fn maintenance_retry(&self) -> Option<&ExecutorExecutionMaintenanceRetry> {
        self.maintenance_retry.as_ref()
    }

    /// Validate the bound retry scope against the authoritative logical
    /// frontiers in the current engine call.
    pub fn validated_maintenance_retry_scope(
        &self,
        current_request_ids: &[RequestId],
    ) -> Result<Option<&ExecutorExecutionMaintenanceRetry>> {
        let Some(retry) = self.maintenance_retry.as_ref() else {
            return Ok(None);
        };
        let current = current_request_ids.iter().collect::<HashSet<_>>();
        if current_request_ids.is_empty() || current.len() != current_request_ids.len() {
            return Err(FerrumError::internal(
                "execution maintenance retry received an invalid current request cohort",
            ));
        }
        let affected = retry.affected_request_ids().iter().collect::<HashSet<_>>();
        if !affected.is_subset(&current) {
            return Err(FerrumError::internal(
                "execution maintenance retry affects a request outside the current cohort",
            ));
        }
        match self.stage {
            ExecutorExecutionCapacityStage::SequenceExtension => {
                if affected.len() != 1 {
                    return Err(FerrumError::internal(
                        "sequence-extension maintenance retry must affect one current request",
                    ));
                }
            }
            ExecutorExecutionCapacityStage::StepAdmission
            | ExecutorExecutionCapacityStage::SubmissionWave => {
                if affected != current {
                    return Err(FerrumError::internal(
                        "cohort maintenance retry must cover the complete current cohort",
                    ));
                }
            }
        }
        Ok(Some(retry))
    }

    /// Return a strictly smaller, capacity-informed prefill width.
    ///
    /// This is a cold pressure-path hint, not allocator authority. The caller
    /// must probe the returned prefix through normal typed admission before any
    /// provider encode or device submission. A bounded reduction prevents a
    /// large frontier from producing an unbounded sequence of near-identical
    /// probes when the shortfall is small.
    pub fn narrower_prefill_tokens(&self, attempted_tokens: usize) -> Option<usize> {
        if attempted_tokens <= 1 {
            return None;
        }
        let maximum_next = attempted_tokens
            .saturating_sub(attempted_tokens.div_ceil(4))
            .max(1);
        let proportional = self
            .shortfalls()
            .iter()
            .filter_map(|shortfall| {
                let requested = shortfall.requested().get();
                let available = shortfall.available().get();
                (requested > available).then(|| {
                    let scaled = (attempted_tokens as u128).saturating_mul(available as u128)
                        / requested as u128;
                    usize::try_from(scaled)
                        .unwrap_or(usize::MAX)
                        .clamp(1, attempted_tokens - 1)
                })
            })
            .min();
        Some(
            proportional
                .unwrap_or_else(|| attempted_tokens.div_ceil(2))
                .min(maximum_next)
                .max(1),
        )
    }
}

/// Scheduler-visible proof that execution is temporarily blocked by a
/// Request-lifetime state hazard rather than by physical capacity.
///
/// The embedded hazard evidence retains the exact plan-local coordinator and
/// supports subscribe-before-recheck waiter registration. The request cohort
/// is the product identity projection used by the scheduler; it must stay
/// separate from the allocator's internal request-authority ids.
#[derive(Debug, Clone, Serialize)]
pub struct ExecutorRequestStateDeferral {
    stage: ExecutorExecutionCapacityStage,
    request_ids: Vec<RequestId>,
    hazard: ferrum_runtime::RequestStateHazardDeferral,
}

impl ExecutorRequestStateDeferral {
    pub fn new(
        stage: ExecutorExecutionCapacityStage,
        request_ids: Vec<RequestId>,
        hazard: ferrum_runtime::RequestStateHazardDeferral,
    ) -> Result<Self> {
        let unique = request_ids.iter().collect::<HashSet<_>>();
        if request_ids.is_empty() || unique.len() != request_ids.len() {
            return Err(FerrumError::internal(
                "request-state execution deferral requires a non-empty unique product cohort",
            ));
        }
        if hazard.blockers().is_empty() {
            return Err(FerrumError::internal(
                "request-state execution deferral requires exact blockers",
            ));
        }
        Ok(Self {
            stage,
            request_ids,
            hazard,
        })
    }

    pub const fn stage(&self) -> ExecutorExecutionCapacityStage {
        self.stage
    }

    pub fn request_ids(&self) -> &[RequestId] {
        &self.request_ids
    }

    pub const fn hazard(&self) -> &ferrum_runtime::RequestStateHazardDeferral {
        &self.hazard
    }

    pub fn register_waiter(&self) -> Result<ferrum_runtime::RequestStateHazardWaitRegistration> {
        self.hazard
            .register_waiter()
            .map_err(|error| FerrumError::backend(error.to_string()))
    }
}

/// A pre-submit execution frontier can be blocked by independently evolving
/// sources. Capacity waits participate in scheduler pressure/yield policy;
/// Request-state waits never do and resume only from their exact hazard
/// coordinator.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "reason", content = "evidence", rename_all = "snake_case")]
pub enum ExecutorExecutionDeferral {
    Capacity(ExecutorExecutionCapacityDeferral),
    /// A real model-owned KV block shortfall without native coordinator
    /// generations. This observation is neither a reservation nor a release
    /// capability; the original allocator must approve any later retry.
    AllocatorCapacity(crate::token_execution::TokenAdmissionCapacityDeferral),
    RequestState(ExecutorRequestStateDeferral),
}

impl ExecutorExecutionDeferral {
    pub const fn stage(&self) -> ExecutorExecutionCapacityStage {
        match self {
            Self::Capacity(deferral) => deferral.stage(),
            Self::AllocatorCapacity(_) => ExecutorExecutionCapacityStage::SequenceExtension,
            Self::RequestState(deferral) => deferral.stage(),
        }
    }

    /// Native coordinator evidence only. Allocator observations have no
    /// synthetic native source, wait condition or epoch.
    pub const fn as_capacity(&self) -> Option<&ExecutorExecutionCapacityDeferral> {
        match self {
            Self::Capacity(deferral) => Some(deferral),
            Self::AllocatorCapacity(_) | Self::RequestState(_) => None,
        }
    }

    pub const fn as_allocator_capacity(
        &self,
    ) -> Option<&crate::token_execution::TokenAdmissionCapacityDeferral> {
        match self {
            Self::AllocatorCapacity(deferral) => Some(deferral),
            Self::Capacity(_) | Self::RequestState(_) => None,
        }
    }

    pub const fn as_request_state(&self) -> Option<&ExecutorRequestStateDeferral> {
        match self {
            Self::Capacity(_) | Self::AllocatorCapacity(_) => None,
            Self::RequestState(deferral) => Some(deferral),
        }
    }
}

impl From<ExecutorExecutionCapacityDeferral> for ExecutorExecutionDeferral {
    fn from(deferral: ExecutorExecutionCapacityDeferral) -> Self {
        Self::Capacity(deferral)
    }
}

impl From<ExecutorRequestStateDeferral> for ExecutorExecutionDeferral {
    fn from(deferral: ExecutorRequestStateDeferral) -> Self {
        Self::RequestState(deferral)
    }
}

impl From<crate::token_execution::TokenAdmissionCapacityDeferral> for ExecutorExecutionDeferral {
    fn from(deferral: crate::token_execution::TokenAdmissionCapacityDeferral) -> Self {
        Self::AllocatorCapacity(deferral)
    }
}

#[cfg(test)]
mod execution_capacity_deferral_tests {
    use super::{
        ExecutorAdmissionEpochs, ExecutorExecutionCapacityDeferral,
        ExecutorExecutionCapacityEvidenceOwner, ExecutorExecutionCapacityStage,
        ExecutorExecutionMaintenanceProgress, ExecutorExecutionMaintenanceRetry,
    };
    use ferrum_runtime::{
        CapacityAvailabilityEpoch, CapacityAvailabilitySource, CapacityWaitCondition,
        DeviceCapacityPressure, DeviceCapacityPressureScope, DynamicBackingPressure,
    };
    use ferrum_types::RequestId;
    use std::num::NonZeroU64;

    fn test_progress() -> ExecutorExecutionMaintenanceProgress {
        ExecutorExecutionMaintenanceProgress {
            attempts: 1,
            coordinator_id: NonZeroU64::new(19).unwrap(),
            mutations: Vec::new(),
            latest_capacity_epoch: 5,
        }
    }

    fn test_deferral(stage: ExecutorExecutionCapacityStage) -> ExecutorExecutionCapacityDeferral {
        let observed =
            CapacityAvailabilityEpoch::new(CapacityAvailabilitySource::ActiveSequenceSlots, 7)
                .unwrap();
        let condition = CapacityWaitCondition::from_observation(19, vec![observed]).unwrap();
        ExecutorExecutionCapacityDeferral::from_backing_pressure(
            ExecutorAdmissionEpochs::new(NonZeroU64::new(19).unwrap(), 3, 5),
            condition,
            test_pressure(),
            stage,
        )
        .unwrap()
    }

    fn test_pressure() -> DynamicBackingPressure {
        DeviceCapacityPressure::new(
            DeviceCapacityPressureScope::PlanBudget,
            "device.execution-capacity-test".to_owned(),
            1,
            1,
            1,
            1,
            1,
        )
        .unwrap()
        .into()
    }

    #[test]
    fn prefill_narrowing_is_strict_bounded_and_stops_at_one_token() {
        let observed =
            CapacityAvailabilityEpoch::new(CapacityAvailabilitySource::ActiveSequenceSlots, 7)
                .unwrap();
        let condition = CapacityWaitCondition::from_observation(19, vec![observed]).unwrap();
        let deferred = ExecutorExecutionCapacityDeferral::from_backing_pressure(
            ExecutorAdmissionEpochs::new(NonZeroU64::new(19).unwrap(), 3, 5),
            condition,
            test_pressure(),
            ExecutorExecutionCapacityStage::StepAdmission,
        )
        .unwrap();

        assert_eq!(deferred.narrower_prefill_tokens(342), Some(171));
        assert_eq!(deferred.narrower_prefill_tokens(2), Some(1));
        assert_eq!(deferred.narrower_prefill_tokens(1), None);
    }

    #[test]
    fn backing_pressure_serializes_one_typed_evidence_owner() {
        let deferred = test_deferral(ExecutorExecutionCapacityStage::SequenceExtension);
        let serialized = serde_json::to_value(&deferred).unwrap();

        assert_eq!(
            deferred.evidence().owner(),
            ExecutorExecutionCapacityEvidenceOwner::Backing
        );
        assert!(deferred.shortfalls().is_empty());
        assert!(deferred.backing_blockers().is_empty());
        assert!(deferred.backing_pressure().is_some());
        assert_eq!(serialized["evidence"]["owner"], "backing");
        assert_eq!(serialized["evidence"]["kind"], "backing_pressure");
        assert!(serialized["evidence"]["pressure"].is_object());
    }

    #[test]
    fn empty_maintenance_receipts_remain_an_ordinary_typed_deferral() {
        let request_id = RequestId::new();
        let deferred = test_deferral(ExecutorExecutionCapacityStage::SequenceExtension)
            .with_relevant_maintenance_retry(2, &[], &[], vec![request_id])
            .unwrap();

        assert!(deferred.maintenance_retry().is_none());
    }

    #[test]
    fn maintenance_retry_rejects_empty_duplicate_or_unproven_scope() {
        let request_id = RequestId::new();
        assert!(ExecutorExecutionMaintenanceRetry::new(Vec::new(), test_progress()).is_err());
        assert!(ExecutorExecutionMaintenanceRetry::new(
            vec![request_id.clone(), request_id.clone()],
            test_progress(),
        )
        .is_err());
        assert!(ExecutorExecutionMaintenanceRetry::new(vec![request_id], test_progress()).is_err());
    }

    #[test]
    fn maintenance_retry_scope_is_fail_closed_for_sequence_and_cohort_stages() {
        let first = RequestId::new();
        let second = RequestId::new();
        let retry = |affected_request_ids| ExecutorExecutionMaintenanceRetry {
            affected_request_ids,
            progress: test_progress(),
        };

        let mut sequence = test_deferral(ExecutorExecutionCapacityStage::SequenceExtension);
        sequence.maintenance_retry = Some(retry(vec![second.clone()]));
        assert_eq!(
            sequence
                .validated_maintenance_retry_scope(&[first.clone(), second.clone()])
                .unwrap()
                .unwrap()
                .affected_request_ids(),
            [second.clone()]
        );
        sequence.maintenance_retry = Some(retry(vec![first.clone(), second.clone()]));
        assert!(sequence
            .validated_maintenance_retry_scope(&[first.clone(), second.clone()])
            .is_err());

        let mut cohort = test_deferral(ExecutorExecutionCapacityStage::SubmissionWave);
        cohort.maintenance_retry = Some(retry(vec![second.clone()]));
        assert!(cohort
            .validated_maintenance_retry_scope(&[first.clone(), second.clone()])
            .is_err());
        cohort.maintenance_retry = Some(retry(vec![first.clone(), second.clone()]));
        assert!(cohort
            .validated_maintenance_retry_scope(&[first, second])
            .unwrap()
            .is_some());
    }
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
        kind: ferrum_runtime::CapacityShortfallKind,
        requested: u64,
        available: u64,
        current_total: u64,
        maximum_total: u64,
    },
    Backing {
        pool_id: String,
        domain_id: u32,
        lifetime: ferrum_runtime::DynamicBackingClaimScope,
        reason: ferrum_runtime::DynamicBackingDeferralReason,
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
    wait_condition: ferrum_runtime::CapacityWaitCondition,
    stage: ExecutorPrefillMaintenanceStage,
    blockers: Vec<ExecutorPrefillMaintenanceBlocker>,
}

impl ExecutorPrefillMaintenanceDeferral {
    pub fn new(
        request_id: RequestId,
        observed: ExecutorAdmissionEpochs,
        wait_condition: ferrum_runtime::CapacityWaitCondition,
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
        deferred: &ferrum_runtime::AdmissionDeferred,
    ) -> Result<Self> {
        if deferred.action() != ferrum_runtime::DeferredAction::AwaitBackingGrowth {
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
        deferred: &ferrum_runtime::DynamicBackingDeferred,
    ) -> Result<Self> {
        let blockers = deferred
            .blockers()
            .iter()
            .map(|blocker| ExecutorPrefillMaintenanceBlocker::Backing {
                pool_id: blocker.pool_id().as_str().to_string(),
                domain_id: blocker.domain_id().get(),
                lifetime: deferred.scope(),
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

    pub fn wait_condition(&self) -> &ferrum_runtime::CapacityWaitCondition {
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
        wait_condition: ferrum_runtime::CapacityWaitCondition,
        pressure: ferrum_runtime::DynamicBackingPressure,
    },
    /// The executor installed real backing and published the resulting
    /// capacity epoch.
    Maintained {
        current: ExecutorAdmissionEpochs,
        pools_grown: usize,
        allocated_bytes: u64,
        pools_reclaimed: usize,
        chunks_reclaimed: usize,
        reclaimed_bytes: u64,
        /// Exact allocator-issued rebalance receipt. The aggregate counters
        /// above remain for stable metrics and must reconcile with this value.
        rebalance: Option<ferrum_runtime::DynamicPoolRebalanceReceipt>,
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
    Deferred(ferrum_runtime::AdmissionDeferred),
    MaintenanceDeferred(ExecutorPrefillMaintenanceDeferral),
    PermanentRejected(ferrum_runtime::AdmissionRejected),
}

/// Core model executor trait focusing on tensor operations
#[async_trait]
pub trait ModelExecutor: Send + Sync + 'static {
    /// Admit immutable token work using this executor's actual resource owner.
    /// Deferred probes do not issue an executable incarnation. An implementation
    /// that queues device preparation retains its actual resources independently
    /// of this future: dropping it must reclaim the unpublished incarnation.
    /// Returning `Admitted` transfers that exact frontier to the caller, which
    /// must publish it or close it without an intervening ownership gap.
    async fn admit_tokens(
        &self,
        _input: crate::token_execution::ExecutionAdmission<'_>,
    ) -> Result<crate::token_execution::TokenAdmissionOutcome> {
        Err(FerrumError::unsupported(
            "executor has not implemented token admission",
        ))
    }

    /// Execute at exact, live frontiers. A bare error is permitted only before
    /// preparation starts. Once work may have advanced, preserve each member's
    /// actual progress and continuation in the returned batch outcome.
    async fn execute_tokens(
        &self,
        _work: &[crate::token_execution::ExecutionWork<'_>],
    ) -> Result<crate::token_execution::ExecutionBatchOutcome> {
        Err(FerrumError::unsupported(
            "executor has not implemented token execution",
        ))
    }

    /// Close an exact incarnation. Held preserves recovery independently of
    /// the returned observer; only checked terminal ownership permits Ended.
    async fn finish_tokens(
        &self,
        _sequence: &crate::token_execution::ExecutorSequence,
        _reason: crate::token_execution::ExecutionEndReason,
    ) -> Result<crate::token_execution::ExecutionCloseOutcome> {
        Err(FerrumError::unsupported(
            "executor has not implemented token closure",
        ))
    }

    /// Request cancellation without awaiting while a scheduler lock is held.
    /// The caller tracks a returned Held observer outside its critical section.
    fn cancel_tokens(
        &self,
        _sequence: &crate::token_execution::ExecutorSequence,
    ) -> Result<crate::token_execution::ExecutionCloseOutcome> {
        Err(FerrumError::unsupported(
            "executor has not implemented token cancellation",
        ))
    }

    /// Immutable product-range planning for this executor instance. Planning
    /// never grants capacity, capture or submission authority. Both selection
    /// and execution preparation use the same pure rule with current progress.
    /// Unknown preserves the supplied chunk without qualifying it for exact
    /// advance prediction; wrappers must explicitly forward their inner rule.
    fn prefill_shape_policy(&self) -> PrefillShapePolicy {
        PrefillShapePolicy::Unknown
    }

    /// Pure boundary planning for optional sharing. None leaves normal scheduling
    /// unchanged; a returned boundary never grants request or resource authority.
    fn plan_prefix_capture_boundary(
        &self,
        _input: PrefixCaptureBoundary<'_>,
    ) -> Option<PrefixCapturePlan> {
        None
    }

    /// Arm interest only against an already-admitted, exact source incarnation.
    /// No device allocation, capacity reservation, provider encoding, or device
    /// submission is allowed here.
    fn retain_prefix_capture_interest(
        &self,
        _input: PrefixCaptureRequest<'_>,
    ) -> Result<Option<Arc<dyn PrefixCaptureLease>>> {
        Ok(None)
    }

    /// Get model information and metadata
    fn info(&self) -> &ModelInfo;

    /// Declares the resource planning model used by this implementation.
    /// Both variants execute and close through the same exact token sequence
    /// protocol. `PlanRuntime` additionally exposes its immutable resolved plan
    /// and shared runtime's capacity evidence; this tag grants no authority.
    fn execution_resource_authority(&self) -> ExecutionResourceAuthority {
        ExecutionResourceAuthority::LegacyEngine
    }

    /// Immutable admission limits compiled into this executor's runtime plan.
    /// A PlanRuntime executor must return `Some`; legacy executors may defer to
    /// the engine-owned scheduler and recurrent-state limits.
    fn admission_limits(&self) -> Result<Option<ExecutorAdmissionLimits>> {
        Ok(None)
    }

    /// Returns the immutable product plan that owns planning, provider
    /// selection, and resource authority for a plan-runtime executor.
    /// Legacy executors return `None`; `PlanRuntime` executors must expose the
    /// exact plan used for provisioning and dispatch.
    fn resolved_model_plan(&self) -> Option<&ferrum_runtime::ResolvedModelPlan> {
        None
    }

    /// Reports the immutable sources bound to this executor. The default
    /// borrows the sources from the exact resolved plan, when one is exposed.
    /// This is a trusted implementation's consistency report, not runtime
    /// authorization to allocate resources, bind requests, or submit work.
    fn bound_resolved_sources(&self) -> Option<&ferrum_runtime::ResolvedModelSources> {
        self.resolved_model_plan()
            .map(|plan| &plan.parts().resolved_sources)
    }

    /// Returns the authoritative memory breakdown for a shared plan runtime.
    /// `LegacyEngine` executors return `None`; `PlanRuntime` executors must
    /// return `Some` while they are ready.
    fn plan_runtime_resource_snapshot(&self) -> Result<Option<PlanRuntimeResourceSnapshot>> {
        Ok(None)
    }

    /// Per-request KV capacity in tokens when the executor owns a smaller
    /// runtime cache window than the model's declared context length.
    fn kv_capacity(&self) -> Option<usize> {
        None
    }

    /// Device-aware limits selected from the exact compiled memory plan before
    /// static model upload. Legacy executors do not provide this evidence.
    fn startup_memory_plan(&self) -> Option<&ferrum_types::StartupMemoryPlan> {
        None
    }

    /// Installs the product-owned execution event sink before requests start.
    ///
    /// Legacy executors have no typed execution journal and keep the default
    /// no-op. Executors backed by the vNext runtime retain this sink with each
    /// admitted request so node/operation events share the product artifact.
    fn attach_execution_event_sink(&self, _sink: Arc<dyn crate::ProductExecutionEventSink>) {}

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
        availability: &mut Vec<ferrum_runtime::CapacityAvailabilityEpoch>,
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
        _observed: &ferrum_runtime::CapacityWaitCondition,
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

    /// Whether this exact plan can retain and restore independent prefix state.
    /// This must include resolved model/provider support and product policy.
    fn supports_plan_runtime_prefix_restore(&self) -> bool {
        false
    }

    /// Restore a proper prefix into a freshly admitted request before the
    /// scheduler publishes any work for it. A miss submits no device work and
    /// preserves the admitted target. Successful restoration remains gated
    /// until the engine publishes the matching scheduler/executor progress and
    /// acknowledges the returned owner. Cancellation must retain unknown native
    /// writes until the existing completion path proves quiescence.
    async fn try_restore_plan_runtime_prefix(
        &self,
        _input: PlanRuntimePrefixRestoreInput<'_>,
    ) -> Result<PlanRuntimePrefixRestoreOutcome> {
        Ok(PlanRuntimePrefixRestoreOutcome::Unavailable)
    }

    /// Writes the exact availability sources advanced when this request
    /// authority is preempted for recompute.
    ///
    /// `true` proves that `preemption` still identifies a live, quiescently
    /// releasable authority and that `sources` is its complete release
    /// footprint. Callers must treat `false` as not releasable; a generic
    /// "owns some cache" observation is not evidence that another request can
    /// advance the source on which the current frontier is blocked.
    fn write_execution_capacity_release_sources(
        &self,
        _preemption: &ExecutorExecutionCapacityPreemption,
        sources: &mut Vec<ferrum_runtime::CapacityAvailabilitySource>,
    ) -> Result<bool> {
        sources.clear();
        Ok(false)
    }

    /// Query the real release footprint of this executor's exact live token
    /// incarnation. This is scheduling evidence, not permission to release or
    /// proof of quiescence: callers must still observe `cancel_tokens` reaching
    /// `Ended`. Executors without native source generations return `false`.
    fn write_token_capacity_release_sources(
        &self,
        _sequence: &crate::token_execution::ExecutorSequence,
        sources: &mut Vec<ferrum_runtime::CapacityAvailabilitySource>,
    ) -> Result<bool> {
        sources.clear();
        Ok(false)
    }

    /// Observe the real allocator behind a typed capacity shortfall. The
    /// blocked frontier must still identify the original live owner. External
    /// candidates are distinct exact incarnations, including retained closes;
    /// missing already-retired candidates contribute no resources. `None`
    /// means this implementation cannot observe the specified physical domain.
    /// Only the original close protocol can establish actual release.
    fn observe_token_allocator_reclaim(
        &self,
        _blocked: &crate::token_execution::ExecutionFrontier,
        _domain: &ferrum_types::AllocatorCapacityDomain,
        _candidates: &[crate::token_execution::ExecutorSequence],
    ) -> Result<Option<crate::token_execution::TokenAllocatorReclaim>> {
        Ok(None)
    }

    /// Retire one exact request-scoped runtime authority for recompute.
    ///
    /// Success is a terminal release fence: all provider/device work that can
    /// access the authority is quiescent and the request can be admitted as a
    /// new sequence incarnation. Implementations must fail closed on identity
    /// mismatch or an in-flight authority they cannot terminalize.
    async fn preempt_execution_capacity(
        &self,
        _preemption: ExecutorExecutionCapacityPreemption,
    ) -> Result<ExecutorExecutionCapacityPreemptionReceipt> {
        Err(FerrumError::unsupported(
            "request-scoped execution-capacity preemption is not implemented",
        ))
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
    /// This capability observes the model's existing allocator. Normal token
    /// execution prepares storage through its admitted sequence owner before
    /// kernel launch; a separate reservation is not permission to execute.
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
    /// Attention-only models return `None`. This describes storage, not an
    /// allocation authority; the admitted sequence owner prepares and retains
    /// recurrent state through the same token lifecycle as its KV state.
    fn recurrent_state_spec(
        &self,
        _request_id: &RequestId,
        _input_tokens: &[TokenId],
    ) -> Result<Option<RecurrentStateSpec>> {
        Ok(None)
    }

    /// Get executor capabilities
    fn capabilities(&self) -> ExecutorCapabilities;

    /// Actual adapter execution, independently of model architecture and resource
    /// authority. Overrides must apply the adapter and isolate sequence state.
    fn adapter_execution_support(&self) -> ferrum_interfaces::adapter::AdapterExecutionSupport {
        ferrum_interfaces::adapter::AdapterExecutionSupport::Unsupported
    }

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

    /// Optional compact provider-attribution witness emitted by executors
    /// whose immutable plan can bind quantized source tensors to selected
    /// operation providers without exposing per-tensor logs.
    fn execution_attribution_snapshot(&self) -> Option<serde_json::Value> {
        None
    }

    /// Optional LoRA runtime metrics.
    fn lora_metrics_snapshot(&self) -> Option<serde_json::Value> {
        None
    }

    /// Complete executor-owned startup preparation before either product
    /// entrypoint can accept a request.
    ///
    /// Implementations use this cold-path hook for work that requires the
    /// fully constructed executor but must not be charged to a user's first
    /// request, such as compiling reusable execution shapes. The default is a
    /// no-op so existing executors remain source compatible.
    async fn prepare_startup(&self) -> Result<()> {
        Ok(())
    }

    /// Warm up executor (load model, allocate memory, etc.)
    async fn warmup(&mut self) -> Result<()> {
        // Default no-op implementation
        Ok(())
    }

    /// Begin final retirement under the original assembly owner's authority.
    /// A per-engine request drain does not grant this permission. Implementations
    /// must retain unresolved resources independently of an awaiting caller.
    /// The owner transfers its last execution reference after all use scopes
    /// end. Retry observes or wakes this same retirement; it must not rebuild
    /// the executor, replay inference, or release an original owner twice.
    fn begin_shutdown(
        self: Arc<Self>,
        permit: &crate::executor_lifecycle::ExecutorShutdownPermit,
    ) -> crate::executor_lifecycle::ExecutorShutdownObserver {
        let error = permit.check(self.as_ref()).err().unwrap_or_else(|| {
            FerrumError::unsupported("executor has not implemented checked whole-model shutdown")
        });
        crate::executor_lifecycle::ExecutorShutdownObserver::retained(Arc::new(self), error)
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

/// Logical state bytes across the complete model, before physical page
/// alignment, pool residency, checkpoint copies or operation workspace.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct TypedSequenceStateMemory {
    /// K/V payload and quantization scale bytes for one token in one sequence.
    pub kv_bytes_per_token: u64,
    /// Other sequence states whose size scales with token count.
    pub other_token_scaled_bytes_per_token: u64,
    /// Fixed sequence state, including recurrent accumulators and windows.
    pub fixed_bytes_per_sequence: u64,
}

/// Memory requirements for model execution. This describes logical model
/// requirements, not measured device allocation or an admission guarantee.
#[derive(Debug, Clone, Deserialize)]
pub struct MemoryRequirements {
    /// Model parameter memory in bytes
    pub parameter_memory: u64,
    /// Minimum activation memory per token
    pub activation_memory_per_token: usize,
    /// Legacy KV cache bytes per token per layer. Ignored and omitted from
    /// serialization when typed_sequence_state supplies complete-model bytes.
    #[serde(default)]
    pub kv_cache_memory_per_token: usize,
    /// Exact logical sequence state from a typed model plan. Old wire values
    /// without this field retain the legacy per-layer interpretation.
    #[serde(default)]
    pub typed_sequence_state: Option<TypedSequenceStateMemory>,
    /// Additional overhead memory
    pub overhead_memory: u64,
}

impl Serialize for MemoryRequirements {
    fn serialize<S: serde::Serializer>(
        &self,
        serializer: S,
    ) -> std::result::Result<S::Ok, S::Error> {
        use serde::ser::SerializeStruct;
        let mut value = serializer.serialize_struct("MemoryRequirements", 4)?;
        value.serialize_field("parameter_memory", &self.parameter_memory)?;
        value.serialize_field(
            "activation_memory_per_token",
            &self.activation_memory_per_token,
        )?;
        match &self.typed_sequence_state {
            Some(state) => value.serialize_field("typed_sequence_state", state)?,
            None => value
                .serialize_field("kv_cache_memory_per_token", &self.kv_cache_memory_per_token)?,
        }
        value.serialize_field("overhead_memory", &self.overhead_memory)?;
        value.end()
    }
}

impl MemoryRequirements {
    /// Calculate a logical memory estimate, returning None on arithmetic
    /// overflow. Typed sequence state is already summed across all layers.
    pub fn checked_calculate_total_memory(
        &self,
        batch_size: usize,
        sequence_length: usize,
        num_layers: usize,
    ) -> Option<u64> {
        let batch = u64::try_from(batch_size).ok()?;
        let tokens = u64::try_from(sequence_length).ok()?;
        let token_count = batch.checked_mul(tokens)?;
        let activation_mem = u64::try_from(self.activation_memory_per_token)
            .ok()?
            .checked_mul(token_count)?;
        let state_mem = match self.typed_sequence_state {
            Some(state) => state
                .kv_bytes_per_token
                .checked_add(state.other_token_scaled_bytes_per_token)?
                .checked_mul(token_count)?
                .checked_add(state.fixed_bytes_per_sequence.checked_mul(batch)?)?,
            None => u64::try_from(self.kv_cache_memory_per_token)
                .ok()?
                .checked_mul(token_count)?
                .checked_mul(u64::try_from(num_layers).ok()?)?,
        };
        self.parameter_memory
            .checked_add(activation_mem)?
            .checked_add(state_mem)?
            .checked_add(self.overhead_memory)
    }

    /// Calculate a logical memory estimate. Overflow saturates to u64::MAX;
    /// callers needing an explicit overflow result can use the checked method.
    pub fn calculate_total_memory(
        &self,
        batch_size: usize,
        sequence_length: usize,
        num_layers: usize,
    ) -> u64 {
        self.checked_calculate_total_memory(batch_size, sequence_length, num_layers)
            .unwrap_or(u64::MAX)
    }
}

#[cfg(test)]
mod memory_requirements_tests {
    use super::{MemoryRequirements, TypedSequenceStateMemory};

    #[test]
    fn typed_sequence_memory_counts_scales_and_fixed_state_once_per_sequence() {
        let memory = MemoryRequirements {
            parameter_memory: 100,
            activation_memory_per_token: 4,
            kv_cache_memory_per_token: 999,
            typed_sequence_state: Some(TypedSequenceStateMemory {
                kv_bytes_per_token: 528,
                other_token_scaled_bytes_per_token: 8,
                fixed_bytes_per_sequence: 64,
            }),
            overhead_memory: 20,
        };
        let expected = 100 + 4 * 2 * 3 + (528 + 8) * 2 * 3 + 64 * 2 + 20;
        for layers in [1, 3, 32] {
            assert_eq!(
                memory.checked_calculate_total_memory(2, 3, layers),
                Some(expected)
            );
        }
        let wire = serde_json::to_value(&memory).unwrap();
        assert!(wire.get("kv_cache_memory_per_token").is_none());
        assert_eq!(wire["typed_sequence_state"]["kv_bytes_per_token"], 528);
        let decoded: MemoryRequirements = serde_json::from_value(wire).unwrap();
        assert_eq!(decoded.calculate_total_memory(2, 3, 32), expected);
    }

    #[test]
    fn legacy_memory_wire_keeps_per_layer_calculation_and_rejects_overflow() {
        let wire = serde_json::json!({
            "parameter_memory": 100,
            "activation_memory_per_token": 4,
            "kv_cache_memory_per_token": 16,
            "overhead_memory": 20,
        });
        let mut memory: MemoryRequirements = serde_json::from_value(wire.clone()).unwrap();
        assert!(memory.typed_sequence_state.is_none());
        assert_eq!(
            memory.calculate_total_memory(2, 3, 5),
            100 + 4 * 2 * 3 + 16 * 2 * 3 * 5 + 20
        );
        assert_eq!(serde_json::to_value(&memory).unwrap(), wire);
        memory.parameter_memory = u64::MAX;
        assert_eq!(memory.checked_calculate_total_memory(1, 1, 1), None);
        assert_eq!(memory.calculate_total_memory(1, 1, 1), u64::MAX);
        memory.parameter_memory = 0;
        memory.typed_sequence_state = Some(TypedSequenceStateMemory {
            kv_bytes_per_token: u64::MAX,
            other_token_scaled_bytes_per_token: 1,
            fixed_bytes_per_sequence: 0,
        });
        assert_eq!(memory.checked_calculate_total_memory(1, 1, 1), None);
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

/// Model executor factory
#[async_trait]
pub trait ModelExecutorFactory: Send + Sync {
    /// Create executor from model configuration
    async fn create_executor(&self, config: &ExecutorConfig) -> Result<Box<dyn ModelExecutor>>;

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
