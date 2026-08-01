//! Continuous Batching Engine
//!
//! Iteration-level continuous batching: each step processes a mixed batch of
//! prefill and decode requests selected by the scheduler.  Multiple callers
//! can submit requests concurrently. An `iteration_lock` makes publication
//! atomic with planning, while the single background driver owns device-wave
//! execution.

use crate::resource_lifecycle::{
    ResourceLedgerTransition, ResourceLifecycleLedger, ResourceOwnerCloseSummary,
};
use async_trait::async_trait;
use ferrum_bench_core::{
    global_profile, profile_fields_from_json, JsonlJournal, JsonlJournalError,
};
use ferrum_interfaces::{
    engine::{InferenceEngine, LlmInferenceEngine},
    kv_cache::AllocationRequest,
    model_executor::{
        ExecutionResourceAuthority, ExecutorAdmissionEpochs, ExecutorCapacityWaitRegistration,
        ExecutorExecutionCapacityDeferral, ExecutorExecutionCapacityPreemption,
        ExecutorPrefillAdmission, ExecutorPrefillAdmissionDecision,
        ExecutorPrefillAdmissionReceipt, ExecutorPrefillMaintenanceDeferral,
        ExecutorPrefillMaintenanceOutcome, ExecutorRequestOrigin, ExecutorSamplingOutput,
        ExecutorSequenceCompletion, GreedyRepetitionPenalty, KvSlotRequest, LogitsReturnPolicy,
        PlanRuntimeBatchDecodeOutcome, PlanRuntimeBatchPrefillOutcome, PlanRuntimeDecodeInput,
        PlanRuntimePrefillInput, PlanRuntimePrefillOutcome, PlanRuntimePrefillProduct,
        TokenSelectionMask,
    },
    sampler::{SamplingConfig as TokenSamplingPlan, SamplingRng},
    vnext::{
        AdmissionDeferred, AdmissionRejected, BoundDeviceSubmissionAttribution,
        BoundExecutionResourceMaintenance, CapacityAvailabilityEpoch, DeferredAction,
        DeviceCapacityPressureScope, DeviceExecutionSpanKind, DeviceExecutionSpanMeasurement,
        DeviceSubmissionExecutionSpan, DeviceSubmissionExecutionTiming, DeviceTimingMeasurement,
        DeviceTimingUnavailableReason, EventBatchEmissionPermit, EventEmissionPermit,
        ExecutionEvent, ExecutionEventCapturePolicy, ExecutionEventDetail,
        ExecutionEventKind as VNextExecutionEventKind, ExecutionEventSink, ExecutionEventSinkError,
        OperationCompletionReceipt,
    },
    KvCacheHandle, KvCacheManager, ModelExecutor, RecurrentStateHandle, RecurrentStateManager,
    Sampler, SchedulerInterface as Scheduler, TensorFactory, TensorRef, Tokenizer,
};
use ferrum_kv::cache::prefix::PrefixCache;
use ferrum_sampler::structured_output::{StructuredOutputFactory, StructuredOutputProcessor};
use ferrum_scheduler::implementations::{
    ContinuousBatchScheduler, ExecutionCapacityAction, ExecutionCapacityReleaseSnapshot,
    ExecutorAdmissionProbeOutcome, ExecutorAdmissionQueueObservation, PressureYieldTransaction,
    RequestPhase,
};
#[cfg(test)]
use ferrum_scheduler::implementations::{PressureTransitionKind, PressureYieldKind};
use ferrum_scheduler::vnext::{
    AdmissionDeferral, AdmissionProbeOutcome, AdmissionWakeEpochs, AdmissionWakeSnapshot,
};
use ferrum_types::{
    DataType, Device, EngineConfig, EngineStatus, FerrumError, FerrumProfileEvent, FinishReason,
    InferenceExecutionEvidence, InferenceRequest, InferenceResponse, ObservabilityProfileDetail,
    Priority, ProfileEntrypoint, ProfileError, ProfileEventKind, ProfileStatus, RequestId,
    ResourceAction, ResourceTraceEvent, ResponseCompletionBoundary, Result, SamplingParams,
    StreamChunk, TokenId, TokenUsage, DEFAULT_MAX_TOKENS_METADATA_KEY,
    ENGINE_RUNTIME_TRACE_PRESET_HASH, OBSERVABILITY_PROFILE_SCHEMA_VERSION,
    PROMPT_TOKENS_METADATA_KEY,
};
use futures::stream::Stream;
use metrics::{counter, gauge, histogram};
use parking_lot::{Mutex, RwLock};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::OnceLock;
use std::sync::{Arc, Weak};
use std::task::{Context, Poll};
use std::time::{Duration, Instant};
use tokio::sync::{mpsc, Notify};
use tracing::{debug, info, warn};

// Env-name constants + `from_env_vars` are retained as test-only parse
// helpers: production resolves these knobs via EngineConfig.runtime
// (apply_runtime_config_snapshot), not env. The unit tests still exercise the
// env-name → field mapping.
#[cfg(test)]
const BATCH_DECODE_PROF_ENV: &str = "FERRUM_BATCH_DECODE_PROF";
#[cfg(test)]
const CHUNKED_PREFILL_ENV: &str = "FERRUM_CHUNKED_PREFILL";
#[cfg(test)]
const KV_CAPACITY_ENV: &str = "FERRUM_KV_CAPACITY";
#[cfg(test)]
const MAX_MODEL_LEN_ENV: &str = "FERRUM_MAX_MODEL_LEN";
#[cfg(test)]
const NEXT_BATCH_PROF_ENV: &str = "FERRUM_NEXT_BATCH_PROF";
#[cfg(test)]
const WHOLE_PROMPT_PREFIX_CACHE_ENV: &str = "FERRUM_WHOLE_PROMPT_PREFIX_CACHE";
#[cfg(test)]
const RBD_PROF_ENV: &str = "FERRUM_RBD_PROF";
#[cfg(test)]
const UNIFIED_POST_PROF_ENV: &str = "FERRUM_UNIFIED_POST_PROF";
const GENERATION_POLICY_SCAN_LIMIT: usize = 262_144;
const FORBIDDEN_DECODE_RESAMPLE_LIMIT: usize = 64;
const TOKEN_TRACE_PROMPT_PREFIX_LIMIT: usize = 64;
const TOKEN_TRACE_PROMPT_TAIL_LIMIT: usize = 128;
const TOKEN_TRACE_GENERATED_PREFIX_LIMIT: usize = 256;
const TOKEN_TRACE_GENERATED_TAIL_LIMIT: usize = 32;
const KV_ADMISSION_TARGET_LEN_METADATA_KEY: &str = "ferrum_kv_admission_target_len";
const GENERATED_CONTROL_TOKEN_TEXTS: &[&str] = &[
    "<think>",
    "</think>",
    "<|im_end|>",
    "<|endoftext|>",
    "<|eot_id|>",
    "<|eom_id|>",
    "</s>",
];

struct TokenPolicyCacheEntry {
    tokenizer: Weak<dyn Tokenizer + Send + Sync>,
    forbidden: HashSet<u32>,
}

static TOKEN_POLICY_CACHE: OnceLock<
    std::sync::Mutex<HashMap<(usize, usize), TokenPolicyCacheEntry>>,
> = OnceLock::new();

#[derive(Debug, Clone, Default, PartialEq, Eq)]
struct ContinuousEngineRuntimeConfig {
    active_decode_prefill_chunk: Option<usize>,
    batch_decode_prof: bool,
    chunked_prefill_present: bool,
    chunked_prefill_size: Option<usize>,
    kv_capacity: Option<usize>,
    max_model_len: Option<usize>,
    next_batch_prof: bool,
    profile_entrypoint: Option<ProfileEntrypoint>,
    profile_jsonl: Option<PathBuf>,
    prefix_cache_enabled: bool,
    rbd_prof: bool,
    scheduler_trace_jsonl: Option<PathBuf>,
    legacy_scheduler_trace_jsonl: Option<PathBuf>,
    unified_post_prof: bool,
}

impl ContinuousEngineRuntimeConfig {
    /// Build from the typed `EngineConfig.runtime` knobs (resolved by the CLI/
    /// autosizer via the runtime-config snapshot). Reads no environment — the
    /// env bridge stays at the composition root.
    fn from_engine_config(config: &EngineConfig) -> Self {
        let r = &config.runtime;
        Self {
            active_decode_prefill_chunk: config.scheduler.active_decode_prefill_chunk,
            batch_decode_prof: r.batch_decode_prof,
            chunked_prefill_present: r.chunked_prefill_size.is_some(),
            chunked_prefill_size: r.chunked_prefill_size,
            kv_capacity: r.kv_capacity,
            max_model_len: r.max_model_len,
            next_batch_prof: r.next_batch_prof,
            profile_entrypoint: r.profile_entrypoint,
            profile_jsonl: r.profile_jsonl.clone(),
            prefix_cache_enabled: r.prefix_cache_enabled,
            rbd_prof: r.rbd_prof,
            scheduler_trace_jsonl: r.scheduler_trace_jsonl.clone(),
            legacy_scheduler_trace_jsonl: r.legacy_scheduler_trace_jsonl.clone(),
            unified_post_prof: r.unified_post_prof,
        }
    }

    #[cfg(test)]
    fn from_env_vars<I, K, V>(active_decode_prefill_chunk: Option<usize>, vars: I) -> Self
    where
        I: IntoIterator<Item = (K, V)>,
        K: Into<String>,
        V: Into<String>,
    {
        let vars: HashMap<String, String> = vars
            .into_iter()
            .map(|(key, value)| (key.into(), value.into()))
            .collect();
        Self {
            active_decode_prefill_chunk,
            batch_decode_prof: vars.contains_key(BATCH_DECODE_PROF_ENV),
            chunked_prefill_present: vars.contains_key(CHUNKED_PREFILL_ENV),
            chunked_prefill_size: parse_positive_usize_env(&vars, CHUNKED_PREFILL_ENV),
            kv_capacity: parse_positive_usize_env(&vars, KV_CAPACITY_ENV),
            max_model_len: parse_positive_usize_env(&vars, MAX_MODEL_LEN_ENV),
            next_batch_prof: vars.contains_key(NEXT_BATCH_PROF_ENV),
            profile_entrypoint: vars
                .get("FERRUM_PROFILE_ENTRYPOINT")
                .and_then(|value| ProfileEntrypoint::parse(value)),
            profile_jsonl: vars
                .get("FERRUM_PROFILE_JSONL")
                .and_then(|value| ferrum_types::parse_path_env_value(value).ok()),
            prefix_cache_enabled: vars
                .get(WHOLE_PROMPT_PREFIX_CACHE_ENV)
                .is_some_and(|v| v == "1"),
            rbd_prof: vars.contains_key(RBD_PROF_ENV),
            scheduler_trace_jsonl: vars
                .get("FERRUM_SCHEDULER_TRACE_JSONL")
                .and_then(|value| ferrum_types::parse_path_env_value(value).ok()),
            legacy_scheduler_trace_jsonl: vars
                .get("FERRUM_LEGACY_SCHEDULER_TRACE_JSONL")
                .and_then(|value| ferrum_types::parse_path_env_value(value).ok()),
            unified_post_prof: vars.contains_key(UNIFIED_POST_PROF_ENV),
        }
    }

    fn chunked_prefill_size_for(&self, num_tokens: usize) -> Option<usize> {
        self.chunked_prefill_size.filter(|&n| n < num_tokens)
    }
}

#[cfg(test)]
fn parse_positive_usize_env(vars: &HashMap<String, String>, name: &str) -> Option<usize> {
    vars.get(name)
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&v| v > 0)
}

fn effective_request_context_capacity(
    config: &EngineConfig,
    runtime_config: &ContinuousEngineRuntimeConfig,
    executor_kv_capacity: Option<usize>,
) -> Option<usize> {
    let kv_capacity = runtime_config
        .kv_capacity
        .or(executor_kv_capacity)
        .or_else(|| (config.kv_cache.max_blocks > 0).then_some(config.kv_cache.max_blocks));
    let max_model_len = runtime_config.max_model_len.or_else(|| {
        config
            .model
            .model_info
            .as_ref()
            .map(|info| info.max_sequence_length)
            .filter(|&len| len > 0)
    });

    match (kv_capacity, max_model_len) {
        (Some(kv), Some(model)) => Some(kv.min(model)),
        (Some(kv), None) => Some(kv),
        (None, Some(model)) => Some(model),
        (None, None) => None,
    }
}

fn validate_request_context_budget(
    request: &InferenceRequest,
    input_tokens: usize,
    config: &EngineConfig,
    runtime_config: &ContinuousEngineRuntimeConfig,
    executor_kv_capacity: Option<usize>,
) -> Result<()> {
    let Some(capacity) =
        effective_request_context_capacity(config, runtime_config, executor_kv_capacity)
    else {
        return Ok(());
    };
    let output_tokens = request.sampling_params.max_tokens;
    if input_tokens.saturating_add(output_tokens) <= capacity {
        return Ok(());
    }

    Err(FerrumError::request_validation(format!(
        "This model context is limited to {capacity} tokens, but this request needs {input_tokens} input tokens + {output_tokens} output tokens. Reduce max_tokens or shorten the messages."
    )))
}

fn clamp_default_max_tokens_to_context(
    request: &mut InferenceRequest,
    input_tokens: usize,
    config: &EngineConfig,
    runtime_config: &ContinuousEngineRuntimeConfig,
    executor_kv_capacity: Option<usize>,
) {
    let default_max_tokens = request
        .metadata
        .get(DEFAULT_MAX_TOKENS_METADATA_KEY)
        .and_then(|value| value.as_bool())
        .unwrap_or(false);
    if !default_max_tokens {
        return;
    }
    let Some(capacity) =
        effective_request_context_capacity(config, runtime_config, executor_kv_capacity)
    else {
        return;
    };
    let available_output_tokens = capacity.saturating_sub(input_tokens);
    if available_output_tokens == 0 {
        return;
    }
    let current = request.sampling_params.max_tokens;
    let clamped = current.min(available_output_tokens);
    if clamped < current {
        warn!(
            "Clamping default max_tokens from {} to {} for context budget: input_tokens={}, capacity={}",
            current, clamped, input_tokens, capacity
        );
        request.sampling_params.max_tokens = clamped;
    }
}

/// Resolve per-request stop conditions into (single-token-ids, multi-token-texts).
///
/// Combines:
/// 1. Model EOS reported by the tokenizer (`special_tokens().eos_token`).
/// 2. Common chat-EOS literal names looked up in the tokenizer's vocab —
///    `<|im_end|>`, `<|endoftext|>`, `<|eot_id|>`, `</s>`. Each lookup is
///    model-specific (only IDs that actually exist in this vocab get added),
///    so there's no risk of inserting an unrelated token id from a hard-coded
///    fallback list (e.g. `2` is `</s>` for LLaMA but `!` for Qwen3).
/// 3. User-supplied `stop_sequences` — each is encoded with `add_special=false`;
///    one-token results land in `stop_token_ids` for the fast path, and all
///    user stop strings remain in `stop_text_seqs` so tokens that contain the
///    stop text as a substring still stop.
fn resolve_stop_conditions(
    params: &SamplingParams,
    tokenizer: Option<&(dyn Tokenizer + Send + Sync)>,
    ignore_eos: bool,
) -> (Vec<u32>, HashSet<u32>, Vec<String>) {
    let mut model_eos_token_ids = Vec::new();
    let mut text_seqs: Vec<String> = Vec::new();

    if let Some(tok) = tokenizer {
        if !ignore_eos {
            if let Some(eos) = tok.special_tokens().eos_token {
                model_eos_token_ids.push(eos.get());
            }
            for extra in &tok.special_tokens().extra_eos_tokens {
                model_eos_token_ids.push(extra.get());
            }
            for name in ["<|im_end|>", "<|endoftext|>", "<|eot_id|>", "</s>"] {
                if let Some(t) = tok.token_id(name) {
                    model_eos_token_ids.push(t.get());
                }
            }
        }
    }
    model_eos_token_ids.sort_unstable();
    model_eos_token_ids.dedup();
    let mut ids: HashSet<u32> = model_eos_token_ids.iter().copied().collect();

    if let Some(tok) = tokenizer {
        for stop_seq in &params.stop_sequences {
            if !stop_seq.is_empty() {
                text_seqs.push(stop_seq.clone());
            }
            match tok.encode(stop_seq, false) {
                Ok(toks) if toks.len() == 1 => {
                    ids.insert(toks[0].get());
                }
                _ => {}
            }
        }
    } else {
        for stop_seq in &params.stop_sequences {
            text_seqs.push(stop_seq.clone());
        }
    }
    (model_eos_token_ids, ids, text_seqs)
}

#[derive(Debug)]
struct TokenSequenceMatcher {
    tokens: Vec<u32>,
    failure: Vec<usize>,
    matched: usize,
}

impl TokenSequenceMatcher {
    fn new(tokens: Vec<u32>, label: &str) -> Result<Self> {
        if tokens.is_empty() {
            return Err(FerrumError::config(format!(
                "{label} requires at least one token"
            )));
        }
        let failure = delimiter_failure_table(&tokens);
        Ok(Self {
            tokens,
            failure,
            matched: 0,
        })
    }

    fn observe(&mut self, token_id: u32) -> bool {
        if self.matched == self.tokens.len() {
            self.matched = self.failure[self.matched - 1];
        }
        while self.matched > 0 && self.tokens[self.matched] != token_id {
            self.matched = self.failure[self.matched - 1];
        }
        if self.tokens[self.matched] == token_id {
            self.matched += 1;
        }
        let completed = self.matched == self.tokens.len();
        if completed {
            self.matched = self.failure[self.matched - 1];
        }
        completed
    }

    fn reset(&mut self) {
        self.matched = 0;
    }

    fn is_at_boundary(&self) -> bool {
        self.matched == 0
    }

    fn is_partial(&self) -> bool {
        self.matched > 0
    }
}

#[derive(Debug)]
enum DelimitedPayloadCompletionState {
    AwaitingDelimiter(TokenSequenceMatcher),
    AwaitingPayload,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EnvelopePrefixEffect {
    Clear,
    Pending,
    Rejected,
}

impl DelimitedPayloadCompletionState {
    fn observe(
        &mut self,
        previous_tokens: &[TokenId],
        token: TokenId,
        tokenizer: Option<&(dyn Tokenizer + Send + Sync)>,
        envelope_prefix: EnvelopePrefixEffect,
    ) -> Result<bool> {
        match self {
            Self::AwaitingDelimiter(matcher) => {
                if matcher.observe(token.get()) {
                    *self = Self::AwaitingPayload;
                }
                Ok(false)
            }
            Self::AwaitingPayload => {
                match envelope_prefix {
                    EnvelopePrefixEffect::Pending => return Ok(false),
                    EnvelopePrefixEffect::Rejected => return Ok(true),
                    EnvelopePrefixEffect::Clear => {}
                }
                let tokenizer = tokenizer.ok_or_else(|| {
                    FerrumError::config("response completion boundary lost its tokenizer")
                })?;
                let delta = tokenizer.decode_incremental(previous_tokens, token)?;
                Ok(delta
                    .chars()
                    .any(|character| !character.is_whitespace() && character != '\u{FFFD}'))
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ResponseEnvelopePhase {
    AwaitingOpen,
    AwaitingClose,
}

#[derive(Debug)]
struct ResponseEnvelopeCompletionState {
    open: TokenSequenceMatcher,
    close: TokenSequenceMatcher,
    phase: ResponseEnvelopePhase,
    completed_envelopes: usize,
    max_envelopes: usize,
}

impl ResponseEnvelopeCompletionState {
    fn observe(&mut self, token_id: u32) -> Result<()> {
        match self.phase {
            ResponseEnvelopePhase::AwaitingOpen => {
                if self.open.observe(token_id) {
                    if self.completed_envelopes == self.max_envelopes {
                        return Err(FerrumError::invalid_format(format!(
                            "generated response exceeded its {}-envelope protocol limit",
                            self.max_envelopes
                        )));
                    }
                    self.phase = ResponseEnvelopePhase::AwaitingClose;
                    self.close.reset();
                }
            }
            ResponseEnvelopePhase::AwaitingClose => {
                if self.close.observe(token_id) {
                    self.completed_envelopes += 1;
                    self.phase = ResponseEnvelopePhase::AwaitingOpen;
                    self.open.reset();
                    self.close.reset();
                }
            }
        }
        Ok(())
    }

    fn allows_model_eos(&self) -> bool {
        self.completed_envelopes > 0
            && self.phase == ResponseEnvelopePhase::AwaitingOpen
            && self.open.is_at_boundary()
    }

    fn has_committed_to_envelope_path(&self) -> bool {
        self.phase == ResponseEnvelopePhase::AwaitingClose || self.completed_envelopes > 0
    }

    fn opener_is_partial(&self) -> bool {
        self.phase == ResponseEnvelopePhase::AwaitingOpen && self.open.is_partial()
    }
}

#[derive(Debug)]
enum ResponseCompletionState {
    Satisfied,
    Pending {
        delimited_payload: DelimitedPayloadCompletionState,
        alternate_envelope: Option<ResponseEnvelopeCompletionState>,
    },
}

impl ResponseCompletionState {
    fn compile_token_sequence(
        text: &str,
        label: &str,
        tokenizer: &(dyn Tokenizer + Send + Sync),
        model_eos_token_ids: &[u32],
    ) -> Result<Vec<u32>> {
        let tokens = if let Some(token) = tokenizer.token_id(text) {
            vec![token.get()]
        } else {
            tokenizer
                .encode(text, false)?
                .into_iter()
                .map(TokenId::get)
                .collect::<Vec<_>>()
        };
        if tokens.is_empty() {
            return Err(FerrumError::invalid_request(format!(
                "{label} {text:?} did not tokenize"
            )));
        }
        if let Some(token) = tokens
            .iter()
            .find(|token| model_eos_token_ids.contains(token))
        {
            return Err(FerrumError::invalid_request(format!(
                "{label} token {token} conflicts with model EOS"
            )));
        }
        Ok(tokens)
    }

    fn compile(
        boundary: &ResponseCompletionBoundary,
        tokenizer: Option<&(dyn Tokenizer + Send + Sync)>,
        model_eos_token_ids: &[u32],
        max_tokens: usize,
    ) -> Result<Self> {
        let ResponseCompletionBoundary::AfterDelimiterAndPayload {
            delimiter,
            alternate_envelope,
        } = boundary
        else {
            return Ok(Self::Satisfied);
        };
        if model_eos_token_ids.is_empty() {
            return Ok(Self::Satisfied);
        }
        let tokenizer = tokenizer.ok_or_else(|| {
            FerrumError::config("response completion boundary requires a tokenizer")
        })?;
        let delimiter_tokens = Self::compile_token_sequence(
            delimiter,
            "response completion delimiter",
            tokenizer,
            model_eos_token_ids,
        )?;
        if max_tokens <= delimiter_tokens.len() {
            return Err(FerrumError::invalid_request(format!(
                "response completion requires max_tokens greater than its {}-token delimiter",
                delimiter_tokens.len()
            )));
        }
        let alternate_envelope = alternate_envelope
            .as_ref()
            .map(|envelope| -> Result<ResponseEnvelopeCompletionState> {
                if envelope.max_envelopes == 0 {
                    return Err(FerrumError::invalid_request(
                        "response completion envelope limit must be greater than zero",
                    ));
                }
                Ok(ResponseEnvelopeCompletionState {
                    open: TokenSequenceMatcher::new(
                        Self::compile_token_sequence(
                            &envelope.open_token_text,
                            "response completion envelope opener",
                            tokenizer,
                            model_eos_token_ids,
                        )?,
                        "response completion envelope opener",
                    )?,
                    close: TokenSequenceMatcher::new(
                        Self::compile_token_sequence(
                            &envelope.close_token_text,
                            "response completion envelope closer",
                            tokenizer,
                            model_eos_token_ids,
                        )?,
                        "response completion envelope closer",
                    )?,
                    phase: ResponseEnvelopePhase::AwaitingOpen,
                    completed_envelopes: 0,
                    max_envelopes: envelope.max_envelopes,
                })
            })
            .transpose()?;
        Ok(Self::Pending {
            delimited_payload: DelimitedPayloadCompletionState::AwaitingDelimiter(
                TokenSequenceMatcher::new(delimiter_tokens, "response completion delimiter")?,
            ),
            alternate_envelope,
        })
    }

    fn allows_model_eos(&self) -> bool {
        match self {
            Self::Satisfied => true,
            Self::Pending {
                alternate_envelope, ..
            } => alternate_envelope
                .as_ref()
                .is_some_and(ResponseEnvelopeCompletionState::allows_model_eos),
        }
    }

    /// Advance the completion protocol with one accepted token. Matchers are
    /// allocation-free after request construction.
    fn observe(
        &mut self,
        previous_tokens: &[TokenId],
        token: TokenId,
        tokenizer: Option<&(dyn Tokenizer + Send + Sync)>,
    ) -> Result<Option<bool>> {
        let allowed_before = self.allows_model_eos();
        let payload_completed = match self {
            Self::Satisfied => return Ok(None),
            Self::Pending {
                delimited_payload,
                alternate_envelope,
            } => {
                if let Some(envelope) = alternate_envelope {
                    let committed_before = envelope.has_committed_to_envelope_path();
                    let opener_was_partial = envelope.opener_is_partial();
                    envelope.observe(token.get())?;
                    let committed_after = envelope.has_committed_to_envelope_path();
                    let opener_is_partial = envelope.opener_is_partial();

                    if committed_before || committed_after {
                        false
                    } else {
                        let envelope_prefix = if opener_is_partial {
                            EnvelopePrefixEffect::Pending
                        } else if opener_was_partial {
                            EnvelopePrefixEffect::Rejected
                        } else {
                            EnvelopePrefixEffect::Clear
                        };
                        delimited_payload.observe(
                            previous_tokens,
                            token,
                            tokenizer,
                            envelope_prefix,
                        )?
                    }
                } else {
                    delimited_payload.observe(
                        previous_tokens,
                        token,
                        tokenizer,
                        EnvelopePrefixEffect::Clear,
                    )?
                }
            }
        };
        if payload_completed {
            *self = Self::Satisfied;
        }
        let allowed_after = self.allows_model_eos();
        Ok((allowed_before != allowed_after).then_some(allowed_after))
    }
}

fn delimiter_failure_table(tokens: &[u32]) -> Vec<usize> {
    let mut failure = vec![0usize; tokens.len()];
    let mut matched = 0usize;
    for index in 1..tokens.len() {
        while matched > 0 && tokens[matched] != tokens[index] {
            matched = failure[matched - 1];
        }
        if tokens[matched] == tokens[index] {
            matched += 1;
        }
        failure[index] = matched;
    }
    failure
}

fn resolve_sampling_token_constraints(
    tokenizer: Option<&Arc<dyn Tokenizer + Send + Sync>>,
    stop_token_ids: &HashSet<u32>,
    request_generated_control_token_texts: &[&str],
) -> (HashSet<u32>, Option<usize>, HashSet<u32>) {
    let mut allowed_extended = stop_token_ids.clone();
    let Some(tok) = tokenizer else {
        return (HashSet::new(), None, allowed_extended);
    };

    if let Some(eos) = tok.special_tokens().eos_token {
        allowed_extended.insert(eos.get());
    }
    for extra in &tok.special_tokens().extra_eos_tokens {
        allowed_extended.insert(extra.get());
    }
    for text in GENERATED_CONTROL_TOKEN_TEXTS {
        if let Some(token) = tok.token_id(text) {
            allowed_extended.insert(token.get());
        }
    }
    for text in request_generated_control_token_texts {
        if let Some(token) = tok.token_id(text) {
            allowed_extended.insert(token.get());
        }
    }

    let forbidden = cached_forbidden_generation_tokens(tok, &allowed_extended);

    (forbidden, Some(tok.vocab_size()), allowed_extended)
}

fn build_argmax_token_mask(
    tok: &(dyn Tokenizer + Send + Sync),
    model_vocab_size: Option<usize>,
    forbidden_token_ids: &HashSet<u32>,
    initial_forbidden_token_ids: &HashSet<u32>,
    stop_token_ids: &HashSet<u32>,
    allowed_extended_token_ids: &HashSet<u32>,
) -> TokenSelectionMask {
    let tokenizer_vocab_size = tok.vocab_size();
    let max_allowed_id = allowed_extended_token_ids
        .iter()
        .chain(stop_token_ids.iter())
        .copied()
        .max()
        .map(|id| id as usize + 1)
        .unwrap_or(0);
    let mask_len = model_vocab_size
        .unwrap_or(tokenizer_vocab_size)
        .max(tokenizer_vocab_size)
        .max(max_allowed_id);
    let mut valid = vec![1i8; mask_len];
    for &token_id in forbidden_token_ids
        .iter()
        .chain(initial_forbidden_token_ids.iter())
    {
        if let Some(slot) = valid.get_mut(token_id as usize) {
            *slot = 0;
        }
    }
    for token_id in tokenizer_vocab_size..mask_len {
        if !allowed_extended_token_ids.contains(&(token_id as u32)) {
            valid[token_id] = 0;
        }
    }
    for &token_id in allowed_extended_token_ids {
        if stop_token_ids.contains(&token_id) {
            continue;
        }
        let token = TokenId::new(token_id);
        let should_mask = tok
            .decode(&[token], true)
            .map(|text| decoded_delta_has_forbidden_quality(&text, 0, false, true))
            .unwrap_or(true);
        if should_mask {
            if let Some(slot) = valid.get_mut(token_id as usize) {
                *slot = 0;
            }
        }
    }
    TokenSelectionMask::new(valid)
}

fn cached_forbidden_generation_tokens(
    tok: &Arc<dyn Tokenizer + Send + Sync>,
    allowed_generated_controls: &HashSet<u32>,
) -> HashSet<u32> {
    let key = (tokenizer_cache_key(tok), tok.vocab_size());
    let cache = TOKEN_POLICY_CACHE.get_or_init(|| std::sync::Mutex::new(HashMap::new()));
    let tokenizer_identity = Arc::downgrade(tok);
    {
        let mut cache = cache.lock().expect("token policy cache poisoned");
        if let Some(cached) = cache.get(&key) {
            if cached.tokenizer.ptr_eq(&tokenizer_identity) && cached.tokenizer.strong_count() > 0 {
                let mut forbidden = cached.forbidden.clone();
                forbidden.retain(|token_id| !allowed_generated_controls.contains(token_id));
                return forbidden;
            }
        }
        cache.remove(&key);
    }

    let mut forbidden = HashSet::new();
    let scan_limit = tok.vocab_size().min(GENERATION_POLICY_SCAN_LIMIT);
    let has_reverse_vocab =
        (0..scan_limit).any(|token_id| tok.token_text(TokenId::new(token_id as u32)).is_some());
    let special = tok.special_tokens();
    for token in [
        special.bos_token,
        special.unk_token,
        special.pad_token,
        special.sep_token,
        special.cls_token,
        special.mask_token,
    ]
    .into_iter()
    .flatten()
    {
        forbidden.insert(token.get());
    }

    for text in [
        "<unk",
        "<unk>",
        "[UNK]",
        "<pad>",
        "[PAD]",
        "<|pad|>",
        "<mask>",
        "[MASK]",
        "\u{00ef}\u{00bf}\u{00bd}",
    ] {
        if let Some(token) = tok.token_id(text) {
            forbidden.insert(token.get());
        }
    }
    for token_id in 0..scan_limit {
        let id = token_id as u32;
        let token = TokenId::new(id);
        let raw_text = tok.token_text(token);
        let missing_token_text = has_reverse_vocab && raw_text.is_none();
        let raw_text_forbidden = raw_text.is_some_and(is_forbidden_generation_token_text);
        let decoded_text_forbidden = tok
            .decode(&[token], true)
            .map(|text| decoded_token_is_statically_forbidden(tok.as_ref(), token, &text))
            .unwrap_or(true);
        if missing_token_text || raw_text_forbidden || decoded_text_forbidden {
            forbidden.insert(id);
        }
    }

    let mut cache = cache.lock().expect("token policy cache poisoned");
    cache.retain(|_, entry| entry.tokenizer.strong_count() > 0);
    cache.insert(
        key,
        TokenPolicyCacheEntry {
            tokenizer: tokenizer_identity,
            forbidden: forbidden.clone(),
        },
    );
    drop(cache);
    forbidden.retain(|token_id| !allowed_generated_controls.contains(token_id));
    forbidden
}

fn tokenizer_cache_key(tok: &Arc<dyn Tokenizer + Send + Sync>) -> usize {
    Arc::as_ptr(tok).cast::<()>() as usize
}

fn maybe_trace_prompt_tokens(
    tok: &(dyn Tokenizer + Send + Sync),
    request_id: &RequestId,
    prompt: &str,
) {
    if std::env::var_os("FERRUM_TRACE_PROMPT_TOKENS").is_none() {
        return;
    }

    let prompt_json = serde_json::to_string(prompt).unwrap_or_else(|_| "<json-error>".to_string());
    eprintln!("[prompt-tokens] request_id={request_id} prompt={prompt_json}");
    for add_special in [true, false] {
        match tok.encode(prompt, add_special) {
            Ok(tokens) => {
                let ids: Vec<u32> = tokens.iter().map(|token| token.get()).collect();
                let head_ids: Vec<u32> = ids.iter().copied().take(96).collect();
                let mut tail_ids: Vec<u32> = ids.iter().rev().copied().take(32).collect();
                tail_ids.reverse();
                let head_texts: Vec<String> = tokens
                    .iter()
                    .take(24)
                    .map(|token| {
                        tok.decode(&[*token], false)
                            .unwrap_or_else(|_| "<decode-error>".to_string())
                    })
                    .collect();
                eprintln!(
                    "[prompt-tokens] request_id={request_id} add_special={add_special} len={} head_ids={:?} tail_ids={:?} head_texts={:?}",
                    tokens.len(),
                    head_ids,
                    tail_ids,
                    head_texts,
                );
            }
            Err(err) => {
                eprintln!(
                    "[prompt-tokens] request_id={request_id} add_special={add_special} error={err}"
                );
            }
        }
    }
}

fn is_forbidden_generation_token_text(text: &str) -> bool {
    let text = text.trim();
    if text.is_empty() {
        return false;
    }
    if text.contains('\u{FFFD}') {
        return true;
    }
    if contains_replacement_char_mojibake(text) {
        return true;
    }

    let lower = text.to_ascii_lowercase();
    let lower = lower.as_str();
    if matches!(
        lower,
        "<unk" | "<unk>" | "[unk]" | "<pad>" | "[pad]" | "<|pad|>" | "<mask>" | "[mask]"
    ) {
        return true;
    }

    let looks_like_special = (lower.starts_with('<') && lower.ends_with('>'))
        || (lower.starts_with('[') && lower.ends_with(']'));
    if !looks_like_special {
        return false;
    }

    lower.contains("unk")
        || lower.contains("pad")
        || lower.contains("mask")
        || lower.contains("reserved")
        || lower.contains("unused")
}

fn decoded_token_is_statically_forbidden(
    tokenizer: &(dyn Tokenizer + Send + Sync),
    token: TokenId,
    decoded: &str,
) -> bool {
    if !decoded.contains('\u{FFFD}') {
        return is_forbidden_generation_token_text(decoded);
    }
    if contains_replacement_char_mojibake(decoded) {
        return true;
    }

    let Some(bytes) = tokenizer.token_bytes(token) else {
        return true;
    };
    if bytes.is_empty() || std::str::from_utf8(&bytes).is_ok() {
        return true;
    }

    // Byte-level vocabularies can split one UTF-8 scalar across tokens. A
    // one-token string decode must render such a fragment as U+FFFD, but the
    // raw bytes remain legal in context and are owned by candidate decoding.
    !is_potential_utf8_fragment(&bytes)
}

fn is_potential_utf8_fragment(bytes: &[u8]) -> bool {
    if bytes.is_empty() {
        return false;
    }

    let mut offset = 0usize;
    while offset < bytes.len() && is_utf8_continuation(bytes[offset]) {
        offset += 1;
    }
    if offset > 3 {
        return false;
    }

    while offset < bytes.len() {
        let lead = bytes[offset];
        if lead.is_ascii() {
            offset += 1;
            continue;
        }

        let continuation_count = match lead {
            0xC2..=0xDF => 1,
            0xE0..=0xEF => 2,
            0xF0..=0xF4 => 3,
            _ => return false,
        };
        let available = bytes.len() - offset - 1;
        let present = available.min(continuation_count);
        for index in 0..present {
            let byte = bytes[offset + index + 1];
            if !is_utf8_continuation(byte) {
                return false;
            }
            if index == 0
                && matches!(
                    (lead, byte),
                    (0xE0, 0x80..=0x9F)
                        | (0xED, 0xA0..=0xBF)
                        | (0xF0, 0x80..=0x8F)
                        | (0xF4, 0x90..=0xBF)
                )
            {
                return false;
            }
        }
        if present < continuation_count {
            return true;
        }
        offset += continuation_count + 1;
    }

    true
}

fn is_utf8_continuation(byte: u8) -> bool {
    matches!(byte, 0x80..=0xBF)
}

fn decoded_delta_has_forbidden_quality(
    full_text: &str,
    previous_text_len: usize,
    candidate_is_stop: bool,
    candidate_is_non_stop_control: bool,
) -> bool {
    if previous_text_len > full_text.len() || !full_text.is_char_boundary(previous_text_len) {
        return true;
    }
    let delta = &full_text[previous_text_len..];
    if delta.is_empty() {
        return candidate_is_non_stop_control;
    }
    if contains_replacement_char_mojibake(delta) {
        return true;
    }
    if delta.contains('\u{FFFD}') && (candidate_is_stop || !full_text.ends_with('\u{FFFD}')) {
        return true;
    }
    false
}

fn contains_replacement_char_mojibake(text: &str) -> bool {
    let mut chars = text.chars();
    let mut a = chars.next();
    let mut b = chars.next();
    let mut c = chars.next();
    loop {
        if matches!(
            (a, b, c),
            (Some('\u{00ef}'), Some('\u{00bf}'), Some('\u{00bd}'))
        ) {
            return true;
        }
        if c.is_none() {
            return false;
        }
        a = b;
        b = c;
        c = chars.next();
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Sequence state
// ────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Serialize)]
struct SequenceTokenTraceEvidence {
    schema_version: u32,
    token_encoding: &'static str,
    prompt_token_count: usize,
    prompt_token_sha256: String,
    prompt_token_prefix: Vec<u32>,
    prompt_token_tail: Vec<u32>,
    generated_token_count: usize,
    generated_token_sha256: String,
    generated_token_prefix: Vec<u32>,
    generated_token_tail: Vec<u32>,
    sampling_rng_algorithm: &'static str,
    sampling_seed: Option<u64>,
    sampler: String,
    processors: Vec<String>,
    max_tokens: usize,
    temperature: f32,
    top_p: f32,
    top_k: Option<usize>,
    repetition_penalty: f32,
    presence_penalty: f32,
    frequency_penalty: f32,
}

impl SequenceTokenTraceEvidence {
    fn capture(sequence: &SequenceState) -> Self {
        Self {
            schema_version: OBSERVABILITY_PROFILE_SCHEMA_VERSION,
            token_encoding: "u32-le-v1",
            prompt_token_count: sequence.input_tokens.len(),
            prompt_token_sha256: token_ids_sha256(&sequence.input_tokens),
            prompt_token_prefix: token_id_prefix(
                &sequence.input_tokens,
                TOKEN_TRACE_PROMPT_PREFIX_LIMIT,
            ),
            prompt_token_tail: token_id_tail(&sequence.input_tokens, TOKEN_TRACE_PROMPT_TAIL_LIMIT),
            generated_token_count: sequence.generated_tokens.len(),
            generated_token_sha256: token_ids_sha256(&sequence.generated_tokens),
            generated_token_prefix: token_id_prefix(
                &sequence.generated_tokens,
                TOKEN_TRACE_GENERATED_PREFIX_LIMIT,
            ),
            generated_token_tail: token_id_tail(
                &sequence.generated_tokens,
                TOKEN_TRACE_GENERATED_TAIL_LIMIT,
            ),
            sampling_rng_algorithm: SamplingRng::algorithm_id(),
            sampling_seed: sequence.sampling_params.seed,
            sampler: sequence.sampling_plan.sampler.name().to_string(),
            processors: sequence
                .sampling_plan
                .processor_chain
                .processor_names()
                .into_iter()
                .map(str::to_string)
                .collect(),
            max_tokens: sequence.sampling_params.max_tokens,
            temperature: sequence.sampling_params.temperature,
            top_p: sequence.sampling_params.top_p,
            top_k: sequence.sampling_params.top_k,
            repetition_penalty: sequence.sampling_params.repetition_penalty,
            presence_penalty: sequence.sampling_params.presence_penalty,
            frequency_penalty: sequence.sampling_params.frequency_penalty,
        }
    }
}

fn token_ids_sha256(tokens: &[TokenId]) -> String {
    let mut digest = Sha256::new();
    digest.update(b"ferrum-token-ids:u32-le-v1\0");
    for token in tokens {
        digest.update(token.get().to_le_bytes());
    }
    format!("sha256:{:x}", digest.finalize())
}

fn token_id_prefix(tokens: &[TokenId], limit: usize) -> Vec<u32> {
    tokens.iter().take(limit).map(|token| token.get()).collect()
}

fn token_id_tail(tokens: &[TokenId], limit: usize) -> Vec<u32> {
    let start = tokens.len().saturating_sub(limit);
    tokens[start..].iter().map(|token| token.get()).collect()
}

mod sequence;
pub use sequence::SequenceState;
use sequence::*;

enum EngineIterationOutcome {
    Progressed,
    Idle,
    CapacityBlocked(ExecutorCapacityWaitRegistration),
}

enum EngineResourceComposition {
    LegacyEngine {
        kv_cache: Arc<dyn KvCacheManager + Send + Sync>,
        recurrent_state_manager: Option<Arc<dyn RecurrentStateManager + Send + Sync>>,
    },
    PlanRuntime,
}

impl EngineResourceComposition {
    const fn authority(&self) -> ExecutionResourceAuthority {
        match self {
            Self::LegacyEngine { .. } => ExecutionResourceAuthority::LegacyEngine,
            Self::PlanRuntime => ExecutionResourceAuthority::PlanRuntime,
        }
    }

    fn kv_cache(&self) -> Option<&Arc<dyn KvCacheManager + Send + Sync>> {
        match self {
            Self::LegacyEngine { kv_cache, .. } => Some(kv_cache),
            Self::PlanRuntime => None,
        }
    }

    fn recurrent_state_manager(&self) -> Option<&Arc<dyn RecurrentStateManager + Send + Sync>> {
        match self {
            Self::LegacyEngine {
                recurrent_state_manager,
                ..
            } => recurrent_state_manager.as_ref(),
            Self::PlanRuntime => None,
        }
    }
}

struct EngineInner {
    config: EngineConfig,
    scheduler: Arc<ContinuousBatchScheduler>,
    tokenizer: Arc<dyn Tokenizer + Send + Sync>,
    /// Lazily built once because most requests are plain text. Structured
    /// requests reuse its tokenizer trie and compiled grammar templates.
    structured_output_factory: OnceLock<std::result::Result<Arc<StructuredOutputFactory>, String>>,
    #[allow(dead_code)]
    // Retained for constructor API; sampling now uses per-request SamplingConfig
    sampler: Arc<dyn Sampler + Send + Sync>,
    resource_composition: EngineResourceComposition,
    model_executor: Arc<dyn ModelExecutor + Send + Sync>,
    /// Optional draft executor for speculative decoding. When set alongside
    /// `spec_config`, `run_single_decode` routes through `SpeculativeRunner`.
    draft_executor: Option<Arc<dyn ModelExecutor + Send + Sync>>,
    /// Speculative decoding parameters (N, temperature). `None` = disabled.
    spec_config: Option<crate::speculative::SpeculativeDecodingConfig>,
    tensor_factory: Arc<dyn TensorFactory>,
    sequences: RwLock<HashMap<RequestId, SequenceState>>,
    is_running: AtomicBool,
    shutdown_notify: Arc<Notify>,
    /// Serializes request publication with cancellation and BatchPlan
    /// construction. Device execution deliberately runs after this guard is
    /// released so new user requests can enter while a wave is in flight.
    iteration_lock: tokio::sync::Mutex<()>,
    /// Wakes callers or a background loop when new work is submitted.
    work_notify: Arc<Notify>,
    /// Prefix cache: shares KV blocks across requests with common prompts.
    prefix_cache: PrefixCache,
    runtime_config: ContinuousEngineRuntimeConfig,
    profile_trace_jsonl: Option<SchedulerTraceJournal>,
    scheduler_trace_jsonl: Option<SchedulerTraceJournal>,
    legacy_scheduler_trace_jsonl: Option<Arc<Mutex<std::fs::File>>>,
    scheduler_trace_none_streak: AtomicU64,
    resource_lifecycle: Mutex<ResourceLifecycleLedger>,
    resource_trace_event_counter: AtomicU64,
    dynamic_admission_availability: Mutex<Vec<CapacityAvailabilityEpoch>>,
    // stats
    iteration_count: AtomicU64,
    total_prefill_tokens: AtomicU64,
    total_decode_tokens: AtomicU64,
    total_preemptions: AtomicU64,
    prefix_cache_hits: AtomicU64,
    total_iteration_lock_wait_us: AtomicU64,
    iteration_lock_wait_samples: AtomicU64,
    total_scheduling_time_us: AtomicU64,
    scheduling_time_samples: AtomicU64,
    total_model_execution_time_us: AtomicU64,
    model_execution_time_samples: AtomicU64,
    /// Set true the first time `ensure_bg_loop` runs, so per-request
    /// `infer_stream` callers don't each spawn their own competing
    /// driver task (16 streaming requests = 16 drivers thrashing on
    /// `iteration_lock`, ~5ms/iter of tokio scheduling overhead).
    bg_loop_spawned: AtomicBool,
    shutdown_started: AtomicBool,
    shutdown_lock: tokio::sync::Mutex<()>,
    background_loop: Mutex<Option<tokio::task::JoinHandle<()>>>,
}

struct ClientReceiverDropWake {
    work_notify: Arc<Notify>,
    armed: bool,
}

impl EngineInner {
    fn signal_shutdown(&self) {
        self.shutdown_started.store(true, Ordering::Release);
        self.is_running.store(false, Ordering::SeqCst);

        // The background loop can be between its state check and registering
        // the async wait. `notify_one` retains a permit across that window;
        // `notify_waiters` would lose the shutdown signal when no waiter is
        // registered yet.
        self.shutdown_notify.notify_one();
        self.work_notify.notify_one();
    }

    fn structured_output_factory(&self) -> Result<Arc<StructuredOutputFactory>> {
        match self.structured_output_factory.get_or_init(|| {
            StructuredOutputFactory::new_with_model_vocab_size(
                Arc::clone(&self.tokenizer),
                Some(self.model_executor.info().vocab_size),
            )
            .map(Arc::new)
            .map_err(|error| error.to_string())
        }) {
            Ok(factory) => Ok(Arc::clone(factory)),
            Err(message) => Err(FerrumError::config(format!(
                "structured-output runtime unavailable: {message}"
            ))),
        }
    }
}

impl ClientReceiverDropWake {
    fn new(work_notify: Arc<Notify>) -> Self {
        Self {
            work_notify,
            armed: true,
        }
    }

    fn disarm(&mut self) {
        self.armed = false;
    }
}

impl Drop for ClientReceiverDropWake {
    fn drop(&mut self) {
        if self.armed {
            self.work_notify.notify_one();
        }
    }
}

struct CancellationAwareResponseStream {
    receiver: tokio_stream::wrappers::ReceiverStream<Result<StreamChunk>>,
    receiver_drop_wake: ClientReceiverDropWake,
}

impl Stream for CancellationAwareResponseStream {
    type Item = Result<StreamChunk>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let result = Pin::new(&mut self.receiver).poll_next(cx);
        if matches!(&result, Poll::Ready(None)) {
            self.receiver_drop_wake.disarm();
        }
        result
    }
}

impl EngineInner {
    fn engine_managed_kv_cache(&self) -> Result<&Arc<dyn KvCacheManager + Send + Sync>> {
        self.resource_composition.kv_cache().ok_or_else(|| {
            FerrumError::internal(
                "plan runtime attempted to use the legacy engine KV-cache manager",
            )
        })
    }

    fn recurrent_state_manager(&self) -> Option<&Arc<dyn RecurrentStateManager + Send + Sync>> {
        self.resource_composition.recurrent_state_manager()
    }

    fn record_iteration_lock_wait(&self, duration: Duration) {
        self.total_iteration_lock_wait_us
            .fetch_add(duration_to_us(duration), Ordering::Relaxed);
        self.iteration_lock_wait_samples
            .fetch_add(1, Ordering::Relaxed);
    }

    fn record_scheduling_time(&self, duration: Duration) {
        self.total_scheduling_time_us
            .fetch_add(duration_to_us(duration), Ordering::Relaxed);
        self.scheduling_time_samples.fetch_add(1, Ordering::Relaxed);
    }

    fn record_model_execution_time(&self, duration: Duration) {
        self.total_model_execution_time_us
            .fetch_add(duration_to_us(duration), Ordering::Relaxed);
        self.model_execution_time_samples
            .fetch_add(1, Ordering::Relaxed);
    }

    fn trace_entrypoint(&self) -> ProfileEntrypoint {
        self.runtime_config
            .profile_entrypoint
            .unwrap_or(ProfileEntrypoint::Synthetic)
    }

    fn extend_scheduler_timeline_attributes(
        &self,
        attributes: &mut BTreeMap<String, serde_json::Value>,
    ) {
        let snapshot = self.scheduler.trace_snapshot();
        attributes.extend([
            (
                "active_sequence_count".to_string(),
                serde_json::json!(snapshot.active_len),
            ),
            (
                "monotonic_nanos".to_string(),
                serde_json::json!(inner::scheduler_trace_monotonic_nanos()),
            ),
            (
                "scheduler_snapshot".to_string(),
                serde_json::to_value(snapshot).unwrap_or(serde_json::Value::Null),
            ),
        ]);
    }

    #[allow(clippy::too_many_arguments)]
    fn trace_resource_event(
        &self,
        request_id: &RequestId,
        owner_kind: &str,
        owner_id: &str,
        resource_kind: &str,
        phase: &str,
        action: ResourceAction,
        amount: Option<i64>,
        before: Option<i64>,
        after: Option<i64>,
        capacity: Option<i64>,
        reason: Option<String>,
    ) {
        let Some(sink) = &self.scheduler_trace_jsonl else {
            return;
        };
        let entrypoint = self.trace_entrypoint();
        let event_num = self
            .resource_trace_event_counter
            .fetch_add(1, Ordering::Relaxed);
        let mut attributes = BTreeMap::from([
            (
                "actual_model_smoke".to_string(),
                serde_json::json!(matches!(
                    entrypoint,
                    ProfileEntrypoint::Run | ProfileEntrypoint::Serve
                )),
            ),
            (
                "backend_device".to_string(),
                serde_json::json!(format!("{:?}", self.config.backend.device)),
            ),
            (
                "backend_type".to_string(),
                serde_json::json!(format!("{:?}", self.config.backend.backend_type)),
            ),
            (
                "diagnostic_only".to_string(),
                serde_json::json!(self.config.runtime.profile_detail.diagnostic_only()),
            ),
            ("l0_only".to_string(), serde_json::json!(false)),
            (
                "profile_detail".to_string(),
                serde_json::json!(self.config.runtime.profile_detail.as_str()),
            ),
            (
                "resource_trace_source".to_string(),
                serde_json::json!("engine"),
            ),
        ]);
        if let Some(reason) = reason.as_deref() {
            attributes.insert("resource_reason".to_string(), serde_json::json!(reason));
        }
        let underflow_amount = match (action, amount, before) {
            (ResourceAction::Release | ResourceAction::Rollback, Some(amount), Some(before))
                if amount > before =>
            {
                Some(amount.saturating_sub(before))
            }
            _ => None,
        };
        if let Some(underflow_amount) = underflow_amount {
            attributes.insert(
                "resource_underflow_amount".to_string(),
                serde_json::json!(underflow_amount),
            );
        }
        if matches!(
            action,
            ResourceAction::RequestOpen
                | ResourceAction::RequestClose
                | ResourceAction::Defer
                | ResourceAction::Reject
        ) {
            self.extend_scheduler_timeline_attributes(&mut attributes);
        }
        let timestamp = chrono::Utc::now();
        let mut shape =
            BTreeMap::from([("resource_amount".to_string(), serde_json::json!(amount))]);
        if let Some(capacity) = capacity {
            shape.insert("resource_capacity".to_string(), serde_json::json!(capacity));
        }
        let event = FerrumProfileEvent {
            schema_version: OBSERVABILITY_PROFILE_SCHEMA_VERSION,
            ts_unix_nanos: timestamp
                .timestamp_nanos_opt()
                .unwrap_or_else(|| timestamp.timestamp_micros() * 1_000),
            event_id: format!("evt-engine-resource-{event_num}"),
            request_id: request_id.to_string(),
            correlation_id: Some(request_id.to_string()),
            entrypoint,
            backend: "actual".to_string(),
            runtime_preset_hash: ENGINE_RUNTIME_TRACE_PRESET_HASH.to_string(),
            phase: phase.to_string(),
            event_kind: ProfileEventKind::Resource,
            timestamp,
            status: ProfileStatus::Ok,
            model: Some(self.config.model.model_id.to_string()),
            duration_us: None,
            memory: None,
            resource: Some(ResourceTraceEvent {
                owner_kind: owner_kind.to_string(),
                owner_id: owner_id.to_string(),
                resource_kind: resource_kind.to_string(),
                action,
                amount,
                before,
                after,
                capacity,
                underflow_amount,
                reason,
                error_kind: None,
                message: None,
                resource_error_kind: None,
            }),
            error: None,
            replay: None,
            shape,
            backend_detail: Some(BTreeMap::from([
                (
                    "backend_device".to_string(),
                    serde_json::json!(format!("{:?}", self.config.backend.device)),
                ),
                (
                    "backend_type".to_string(),
                    serde_json::json!(format!("{:?}", self.config.backend.backend_type)),
                ),
            ])),
            attributes,
        };
        if let Err(error) = event.validate() {
            warn!("Skipping invalid engine resource trace event: {}", error);
            return;
        }
        if let Err(error) = sink.enqueue(event) {
            warn!("Failed to enqueue engine resource trace event: {}", error);
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn trace_resource_event_with_close_summary(
        &self,
        request_id: &RequestId,
        owner_kind: &str,
        owner_id: &str,
        resource_kind: &str,
        phase: &str,
        action: ResourceAction,
        close_summary: &[ResourceOwnerCloseSummary],
        status: ProfileStatus,
        message: Option<String>,
    ) {
        let Some(sink) = &self.scheduler_trace_jsonl else {
            return;
        };
        let entrypoint = self.trace_entrypoint();
        let event_num = self
            .resource_trace_event_counter
            .fetch_add(1, Ordering::Relaxed);
        let outstanding: Vec<_> = close_summary
            .iter()
            .filter(|item| item.outstanding_reserved > 0 || item.outstanding_committed > 0)
            .collect();
        let close_summary_json: Vec<_> = close_summary
            .iter()
            .map(|item| {
                serde_json::json!({
                    "resource_kind": item.resource_kind,
                    "reserved": item.reserved,
                    "committed": item.committed,
                    "released": item.released,
                    "rolled_back": item.rolled_back,
                    "outstanding_reserved": item.outstanding_reserved,
                    "outstanding_committed": item.outstanding_committed,
                    "capacity": item.capacity,
                })
            })
            .collect();
        let outstanding_kinds: Vec<_> = outstanding
            .iter()
            .map(|item| item.resource_kind.clone())
            .collect();
        let mut attributes = BTreeMap::from([
            (
                "actual_model_smoke".to_string(),
                serde_json::json!(matches!(
                    entrypoint,
                    ProfileEntrypoint::Run | ProfileEntrypoint::Serve
                )),
            ),
            (
                "backend_device".to_string(),
                serde_json::json!(format!("{:?}", self.config.backend.device)),
            ),
            (
                "backend_type".to_string(),
                serde_json::json!(format!("{:?}", self.config.backend.backend_type)),
            ),
            (
                "diagnostic_only".to_string(),
                serde_json::json!(self.config.runtime.profile_detail.diagnostic_only()),
            ),
            ("l0_only".to_string(), serde_json::json!(false)),
            (
                "profile_detail".to_string(),
                serde_json::json!(self.config.runtime.profile_detail.as_str()),
            ),
            (
                "resource_owner_close_summary".to_string(),
                serde_json::Value::Array(close_summary_json),
            ),
            (
                "resource_owner_outstanding_count".to_string(),
                serde_json::json!(outstanding.len()),
            ),
            (
                "resource_owner_outstanding_kinds".to_string(),
                serde_json::json!(outstanding_kinds),
            ),
            (
                "resource_trace_source".to_string(),
                serde_json::json!("engine"),
            ),
        ]);
        if let Some(message) = message.as_deref() {
            attributes.insert(
                "resource_close_error".to_string(),
                serde_json::json!(message),
            );
        }
        self.extend_scheduler_timeline_attributes(&mut attributes);
        let timestamp = chrono::Utc::now();
        let error = message.as_ref().map(|message| ProfileError {
            kind: "resource_owner_close_outstanding".to_string(),
            message: message.clone(),
            blocking: true,
        });
        let resource_error_kind = error.as_ref().map(|_| "resource_leak".to_string());
        let mut shape = BTreeMap::from([("resource_amount".to_string(), serde_json::Value::Null)]);
        shape.insert(
            "resource_owner_outstanding_count".to_string(),
            serde_json::json!(outstanding.len()),
        );
        let event = FerrumProfileEvent {
            schema_version: OBSERVABILITY_PROFILE_SCHEMA_VERSION,
            ts_unix_nanos: timestamp
                .timestamp_nanos_opt()
                .unwrap_or_else(|| timestamp.timestamp_micros() * 1_000),
            event_id: format!("evt-engine-resource-{event_num}"),
            request_id: request_id.to_string(),
            correlation_id: Some(request_id.to_string()),
            entrypoint,
            backend: "actual".to_string(),
            runtime_preset_hash: ENGINE_RUNTIME_TRACE_PRESET_HASH.to_string(),
            phase: phase.to_string(),
            event_kind: ProfileEventKind::Resource,
            timestamp,
            status,
            model: Some(self.config.model.model_id.to_string()),
            duration_us: None,
            memory: None,
            resource: Some(ResourceTraceEvent {
                owner_kind: owner_kind.to_string(),
                owner_id: owner_id.to_string(),
                resource_kind: resource_kind.to_string(),
                action,
                amount: None,
                before: None,
                after: None,
                capacity: None,
                underflow_amount: None,
                reason: None,
                error_kind: error.as_ref().map(|error| error.kind.clone()),
                message: error.as_ref().map(|error| error.message.clone()),
                resource_error_kind,
            }),
            error,
            replay: None,
            shape,
            backend_detail: Some(BTreeMap::from([
                (
                    "backend_device".to_string(),
                    serde_json::json!(format!("{:?}", self.config.backend.device)),
                ),
                (
                    "backend_type".to_string(),
                    serde_json::json!(format!("{:?}", self.config.backend.backend_type)),
                ),
            ])),
            attributes,
        };
        if let Err(error) = event.validate() {
            warn!(
                "Skipping invalid engine resource close trace event: {}",
                error
            );
            return;
        }
        if let Err(error) = sink.enqueue(event) {
            warn!(
                "Failed to enqueue engine resource close trace event: {}",
                error
            );
        }
    }

    fn resource_amount_i64(amount: usize) -> i64 {
        amount.min(i64::MAX as usize) as i64
    }

    fn trace_lifecycle_resource_event(
        &self,
        request_id: &RequestId,
        owner_kind: &str,
        owner_id: &str,
        resource_kind: &str,
        phase: &str,
        action: ResourceAction,
        amount: i64,
        transition: ResourceLedgerTransition,
    ) {
        self.trace_resource_event(
            request_id,
            owner_kind,
            owner_id,
            resource_kind,
            phase,
            action,
            Some(amount),
            Some(transition.before),
            Some(transition.after),
            transition.capacity,
            None,
        );
    }

    fn trace_request_open(&self, request_id: &RequestId) {
        self.trace_resource_event(
            request_id,
            "request",
            &request_id.to_string(),
            "request_slot",
            "engine_request_open",
            ResourceAction::RequestOpen,
            None,
            None,
            None,
            None,
            None,
        );
    }

    fn trace_request_admitted(&self, request_id: &RequestId) {
        self.trace_resource_reserve_commit(
            request_id,
            "request",
            &request_id.to_string(),
            "request_slot",
            "engine_request_slot",
            1,
            None,
        );
    }

    fn trace_request_rejected(&self, request_id: &RequestId, reason: String) {
        self.trace_resource_event(
            request_id,
            "request",
            &request_id.to_string(),
            "request_slot",
            "engine_request_reject",
            ResourceAction::Reject,
            None,
            None,
            None,
            Some(Self::resource_amount_i64(
                self.config.scheduler.max_waiting_requests,
            )),
            Some(reason),
        );
        self.trace_request_owner_close(request_id);
    }

    fn trace_request_close(&self, request_id: &RequestId) {
        self.trace_resource_release(
            request_id,
            "request",
            &request_id.to_string(),
            "request_slot",
            "engine_request_slot_release",
            1,
            None,
        );
        self.trace_request_owner_close(request_id);
    }

    fn trace_request_owner_close(&self, request_id: &RequestId) {
        let owner_id = request_id.to_string();
        if self.scheduler_trace_jsonl.is_none() {
            self.trace_resource_event(
                request_id,
                "request",
                &owner_id,
                "request_slot",
                "engine_request_close",
                ResourceAction::RequestClose,
                None,
                None,
                None,
                None,
                None,
            );
            return;
        }

        let summary = {
            let mut lifecycle = self.resource_lifecycle.lock();
            let summary = lifecycle.owner_close_summary("request", &owner_id);
            lifecycle.close_owner("request", &owner_id);
            summary
        };
        self.trace_request_owner_close_with_summary(request_id, &summary);
    }

    fn trace_request_owner_close_with_summary(
        &self,
        request_id: &RequestId,
        summary: &[ResourceOwnerCloseSummary],
    ) {
        let outstanding: Vec<_> = summary
            .iter()
            .filter(|item| item.outstanding_reserved > 0 || item.outstanding_committed > 0)
            .collect();
        let close_status = if outstanding.is_empty() {
            ProfileStatus::Ok
        } else {
            ProfileStatus::Failure
        };
        let message = if outstanding.is_empty() {
            None
        } else {
            Some(format!(
                "request closed with outstanding resources: {}",
                outstanding
                    .iter()
                    .map(|item| format!(
                        "{} reserved={} committed={}",
                        item.resource_kind, item.outstanding_reserved, item.outstanding_committed
                    ))
                    .collect::<Vec<_>>()
                    .join(", ")
            ))
        };
        self.trace_resource_event_with_close_summary(
            request_id,
            "request",
            &request_id.to_string(),
            "request_slot",
            "engine_request_close",
            ResourceAction::RequestClose,
            summary,
            close_status,
            message,
        );
    }

    fn trace_scheduler_defer(&self, request_id: &RequestId, phase: &str, reason: &str) {
        self.trace_resource_event(
            request_id,
            "request",
            &request_id.to_string(),
            "scheduler_capacity",
            phase,
            ResourceAction::Defer,
            None,
            None,
            None,
            Some(Self::resource_amount_i64(
                self.config.scheduler.max_running_requests.max(1),
            )),
            Some(reason.to_string()),
        );
    }

    fn trace_resource_reserve_commit(
        &self,
        request_id: &RequestId,
        owner_kind: &str,
        owner_id: &str,
        resource_kind: &str,
        phase_prefix: &str,
        amount: usize,
        capacity: Option<usize>,
    ) {
        if self.scheduler_trace_jsonl.is_none() {
            return;
        }
        let amount = Self::resource_amount_i64(amount.max(1));
        let capacity_i64 = capacity.map(Self::resource_amount_i64);
        let (reserve, commit) = {
            let mut lifecycle = self.resource_lifecycle.lock();
            let reserve =
                lifecycle.reserve(owner_kind, owner_id, resource_kind, amount, capacity_i64);
            let commit =
                lifecycle.commit(owner_kind, owner_id, resource_kind, amount, capacity_i64);
            (reserve, commit)
        };
        self.trace_lifecycle_resource_event(
            request_id,
            owner_kind,
            owner_id,
            resource_kind,
            &format!("{phase_prefix}_reserve"),
            ResourceAction::Reserve,
            amount,
            reserve,
        );
        self.trace_lifecycle_resource_event(
            request_id,
            owner_kind,
            owner_id,
            resource_kind,
            &format!("{phase_prefix}_commit"),
            ResourceAction::Commit,
            amount,
            commit,
        );
    }

    fn trace_resource_release(
        &self,
        request_id: &RequestId,
        owner_kind: &str,
        owner_id: &str,
        resource_kind: &str,
        phase: &str,
        amount: usize,
        capacity: Option<usize>,
    ) {
        if self.scheduler_trace_jsonl.is_none() {
            return;
        }
        let amount = Self::resource_amount_i64(amount.max(1));
        let transition = self.resource_lifecycle.lock().release(
            owner_kind,
            owner_id,
            resource_kind,
            amount,
            capacity.map(Self::resource_amount_i64),
        );
        self.trace_lifecycle_resource_event(
            request_id,
            owner_kind,
            owner_id,
            resource_kind,
            phase,
            ResourceAction::Release,
            amount,
            transition,
        );
    }

    fn trace_resource_release_failure(
        &self,
        request_id: &RequestId,
        resource_kind: &str,
        phase: &str,
        capacity: Option<usize>,
        reason: String,
    ) {
        self.trace_resource_event(
            request_id,
            "request",
            &request_id.to_string(),
            resource_kind,
            phase,
            ResourceAction::Reject,
            None,
            None,
            None,
            capacity.map(Self::resource_amount_i64),
            Some(reason),
        );
    }

    fn kv_resource_blocks_for_tokens(&self, tokens: usize) -> usize {
        tokens
            .div_ceil(self.config.kv_cache.block_size.max(1))
            .max(1)
    }

    fn trace_kv_allocate(&self, request_id: &RequestId, blocks: usize) {
        self.trace_resource_reserve_commit(
            request_id,
            "request",
            &request_id.to_string(),
            "kv_block",
            "engine_kv_block",
            blocks,
            Some(self.config.kv_cache.max_blocks),
        );
    }

    async fn allocate_kv_lease(
        &self,
        owner_request_id: &RequestId,
        allocation_request_id: RequestId,
        request: &AllocationRequest,
        tokens: usize,
    ) -> Result<KvAllocationLease> {
        debug_assert_eq!(allocation_request_id, request.request_id);
        let handle = self.engine_managed_kv_cache()?.allocate(request).await?;
        let blocks = self.kv_resource_blocks_for_tokens(tokens);
        self.trace_kv_allocate(owner_request_id, blocks);
        Ok(KvAllocationLease::new(
            owner_request_id.clone(),
            allocation_request_id,
            handle,
            blocks,
        ))
    }

    fn trace_kv_release(&self, request_id: &RequestId, blocks: usize) {
        self.trace_resource_release(
            request_id,
            "request",
            &request_id.to_string(),
            "kv_block",
            "engine_kv_block_release",
            blocks,
            Some(self.config.kv_cache.max_blocks),
        );
    }

    fn trace_model_cache_ref_acquire(&self, request_id: &RequestId) {
        self.trace_resource_reserve_commit(
            request_id,
            "request",
            &request_id.to_string(),
            "model_cache_ref",
            "engine_model_cache_ref",
            1,
            None,
        );
    }

    fn trace_model_cache_ref_release(&self, request_id: &RequestId) {
        self.trace_resource_release(
            request_id,
            "request",
            &request_id.to_string(),
            "model_cache_ref",
            "engine_model_cache_ref_release",
            1,
            None,
        );
    }

    fn backend_workspace_capacity(&self) -> Option<usize> {
        Some(self.config.scheduler.max_running_requests.max(1))
    }

    fn trace_backend_workspace_acquire(&self, request_id: &RequestId, phase_prefix: &str) {
        self.trace_resource_reserve_commit(
            request_id,
            "request",
            &request_id.to_string(),
            "backend_workspace",
            phase_prefix,
            1,
            self.backend_workspace_capacity(),
        );
    }

    fn trace_backend_workspace_release(&self, request_id: &RequestId, phase: &str) {
        self.trace_resource_release(
            request_id,
            "request",
            &request_id.to_string(),
            "backend_workspace",
            phase,
            1,
            self.backend_workspace_capacity(),
        );
    }

    fn trace_backend_workspace_acquire_many(&self, request_ids: &[RequestId], phase_prefix: &str) {
        for request_id in request_ids {
            self.trace_backend_workspace_acquire(request_id, phase_prefix);
        }
    }

    fn trace_backend_workspace_release_many(&self, request_ids: &[RequestId], phase: &str) {
        for request_id in request_ids {
            self.trace_backend_workspace_release(request_id, phase);
        }
    }

    fn acquire_backend_workspace_lease(
        &self,
        request_ids: Vec<RequestId>,
        phase_prefix: &'static str,
        release_phase: &'static str,
    ) -> BackendWorkspaceLease<'_> {
        BackendWorkspaceLease::new(self, request_ids, phase_prefix, release_phase)
    }

    fn apply_model_cache_ref_update(&self, request_id: &RequestId, update: ModelCacheRefUpdate) {
        if let Some(cache_id) = update.released {
            self.model_executor.release_cache(&cache_id);
            self.trace_model_cache_ref_release(request_id);
        }
        if update.acquired.is_some() {
            self.trace_model_cache_ref_acquire(request_id);
        }
    }

    fn release_model_cache_ref(&self, request_id: &RequestId, cache_id: &str) {
        self.model_executor.release_cache(cache_id);
        self.trace_model_cache_ref_release(request_id);
    }

    async fn release_kv_allocation(
        &self,
        owner_request_id: &RequestId,
        allocation_request_id: RequestId,
        blocks: usize,
    ) {
        let kv_cache = match self.engine_managed_kv_cache() {
            Ok(kv_cache) => kv_cache,
            Err(error) => {
                warn!(
                    owner_request_id = %owner_request_id,
                    allocation_request_id = %allocation_request_id,
                    error = %error,
                    "Legacy engine KV allocation reached a plan-runtime composition"
                );
                return;
            }
        };
        match kv_cache.deallocate(allocation_request_id.clone()).await {
            Ok(()) => {
                self.trace_kv_release(owner_request_id, blocks);
            }
            Err(error) => {
                warn!(
                    owner_request_id = %owner_request_id,
                    allocation_request_id = %allocation_request_id,
                    error = %error,
                    "KV allocation release failed"
                );
                self.trace_resource_release_failure(
                    owner_request_id,
                    "kv_block",
                    "engine_kv_block_release_failed",
                    Some(self.config.kv_cache.max_blocks),
                    format!("kv release failed for {allocation_request_id}: {error}"),
                );
            }
        }
    }

    async fn release_sequence_physical_resources(
        &self,
        request_id: &RequestId,
        resources: SequencePhysicalResources,
    ) {
        if let Some(cache_id) = resources.model_cache_id {
            self.release_model_cache_ref(request_id, &cache_id);
        }
        if let Some(kv_allocation) = resources.legacy_kv_allocation {
            self.release_kv_allocation(request_id, kv_allocation.request_id, kv_allocation.blocks)
                .await;
        }
        if let Some(draft_kv_allocation) = resources.legacy_draft_kv_allocation {
            self.release_kv_allocation(
                request_id,
                draft_kv_allocation.request_id,
                draft_kv_allocation.blocks,
            )
            .await;
        }
        if let Some(recurrent_allocation) = resources.recurrent_state_allocation {
            self.release_recurrent_allocation(request_id, recurrent_allocation.slots)
                .await;
        }
    }

    async fn complete_sequence_physical_resources(
        &self,
        request_id: &RequestId,
        mut resources: SequencePhysicalResources,
        usage: &TokenUsage,
    ) -> Result<()> {
        let completion_result = if let Some(cache_id) = resources.model_cache_id.take() {
            let completion = ExecutorSequenceCompletion::new(
                request_id.clone(),
                cache_id.clone(),
                usage.prompt_tokens,
                usage.completion_tokens,
            );
            let result = match completion {
                Ok(completion) => self.model_executor.complete_cache(completion),
                Err(error) => {
                    self.model_executor.release_cache(&cache_id);
                    Err(error)
                }
            };
            self.trace_model_cache_ref_release(request_id);
            result
        } else {
            Ok(())
        };

        self.release_sequence_physical_resources(request_id, resources)
            .await;
        completion_result
    }

    fn trace_recurrent_allocate(
        &self,
        request_id: &RequestId,
        slots: usize,
        capacity: Option<usize>,
    ) {
        self.trace_resource_reserve_commit(
            request_id,
            "request",
            &request_id.to_string(),
            "recurrent_state_slot",
            "engine_recurrent_state_slot",
            slots,
            capacity,
        );
    }

    fn trace_recurrent_release(
        &self,
        request_id: &RequestId,
        slots: usize,
        capacity: Option<usize>,
    ) {
        self.trace_resource_release(
            request_id,
            "request",
            &request_id.to_string(),
            "recurrent_state_slot",
            "engine_recurrent_state_slot_release",
            slots,
            capacity,
        );
    }

    async fn release_recurrent_allocation(&self, request_id: &RequestId, slots: Option<usize>) {
        if let Some(manager) = self.recurrent_state_manager() {
            let capacity = manager.stats().total_batch_slots;
            match manager.deallocate(request_id.clone()).await {
                Ok(()) => {
                    if let Some(slots) = slots {
                        self.trace_recurrent_release(request_id, slots, Some(capacity));
                    }
                }
                Err(error) => {
                    warn!(
                        request_id = %request_id,
                        error = %error,
                        "Recurrent-state release failed"
                    );
                    if slots.is_some() {
                        self.trace_resource_release_failure(
                            request_id,
                            "recurrent_state_slot",
                            "engine_recurrent_state_slot_release_failed",
                            Some(capacity),
                            format!("recurrent-state release failed for {request_id}: {error}"),
                        );
                    }
                }
            }
        }
    }

    async fn prepare_recurrent_state(
        &self,
        request_id: &RequestId,
        spec: Option<ferrum_interfaces::RecurrentStateSpec>,
    ) -> Result<RecurrentStateAdmission> {
        if let Some(existing) = self
            .sequences
            .read()
            .get(request_id)
            .and_then(SequenceState::recurrent_state_handle)
        {
            return Ok(RecurrentStateAdmission::existing(existing));
        }

        let Some(spec) = spec else {
            return Ok(RecurrentStateAdmission::none());
        };

        debug_assert_eq!(&spec.request_id, request_id);
        let Some(manager) = self.recurrent_state_manager() else {
            return Err(FerrumError::config(format!(
                "model '{}' requires recurrent state for request {}, but no recurrent-state manager is configured",
                self.model_executor.info().model_id, request_id
            )));
        };

        let before_stats = manager.stats();
        let slots = spec.max_batch_slots.max(1);
        let handle = match manager.allocate(&spec).await {
            Ok(handle) => handle,
            Err(error) => {
                self.trace_resource_event(
                    request_id,
                    "request",
                    &request_id.to_string(),
                    "recurrent_state_slot",
                    "engine_recurrent_state_slot_reject",
                    ResourceAction::Reject,
                    None,
                    None,
                    None,
                    Some(Self::resource_amount_i64(before_stats.total_batch_slots)),
                    Some(error.to_string()),
                );
                return Err(error);
            }
        };
        let after_stats = manager.stats();
        self.trace_recurrent_allocate(request_id, slots, Some(after_stats.total_batch_slots));
        Ok(RecurrentStateAdmission::fresh(RecurrentStateLease::new(
            request_id.clone(),
            handle,
            slots,
            Some(after_stats.total_batch_slots),
        )))
    }

    async fn ensure_recurrent_state(
        &self,
        request_id: &RequestId,
        spec: Option<ferrum_interfaces::RecurrentStateSpec>,
    ) -> Result<Option<Arc<dyn RecurrentStateHandle>>> {
        let mut admission = self.prepare_recurrent_state(request_id, spec).await?;
        let handle = admission.handle();
        if let Some(slots) = admission.fresh_slots() {
            let Some(handle) = handle.clone() else {
                admission.release_fresh(self).await;
                return Err(FerrumError::internal(format!(
                    "missing recurrent state handle while committing recurrent slots for {request_id}"
                )));
            };
            let mut found = false;
            {
                let mut sequences = self.sequences.write();
                if let Some(seq) = sequences.get_mut(request_id) {
                    seq.commit_recurrent_state_admission(handle, slots);
                    found = true;
                }
            }
            if found {
                admission.commit_fresh();
            } else {
                admission.release_fresh(self).await;
                return Err(FerrumError::internal(format!(
                    "sequence not found while committing recurrent state for {request_id}"
                )));
            }
        }

        Ok(handle)
    }

    fn performance_breakdown(&self) -> ferrum_types::PerformanceBreakdown {
        ferrum_types::PerformanceBreakdown {
            scheduling_time_ms: avg_duration_ms(
                self.total_scheduling_time_us.load(Ordering::Relaxed),
                self.scheduling_time_samples.load(Ordering::Relaxed),
            ),
            model_execution_time_ms: avg_duration_ms(
                self.total_model_execution_time_us.load(Ordering::Relaxed),
                self.model_execution_time_samples.load(Ordering::Relaxed),
            ),
            other_overhead_time_ms: avg_duration_ms(
                self.total_iteration_lock_wait_us.load(Ordering::Relaxed),
                self.iteration_lock_wait_samples.load(Ordering::Relaxed),
            ),
            ..Default::default()
        }
    }
}

fn duration_to_us(duration: Duration) -> u64 {
    duration.as_micros().min(u64::MAX as u128) as u64
}

fn avg_duration_ms(total_us: u64, samples: u64) -> f64 {
    if samples == 0 {
        0.0
    } else {
        total_us as f64 / samples as f64 / 1000.0
    }
}

mod profile;
use profile::*;

mod inner;

// ────────────────────────────────────────────────────────────────────────────
// Public engine wrapper
// ────────────────────────────────────────────────────────────────────────────

/// Continuous batching inference engine.
///
/// Wraps an `Arc<EngineInner>` so it can be cloned and shared freely.
/// Multiple concurrent `infer()` / `infer_stream()` calls are safe —
/// an internal `iteration_lock` serializes engine steps while allowing
/// all pending requests to be processed in each iteration's batch.
pub struct ContinuousBatchEngine {
    inner: Arc<EngineInner>,
}

impl ContinuousBatchEngine {
    pub fn new(
        config: EngineConfig,
        scheduler: Arc<ContinuousBatchScheduler>,
        tokenizer: Arc<dyn Tokenizer + Send + Sync>,
        sampler: Arc<dyn Sampler + Send + Sync>,
        kv_cache: Arc<dyn KvCacheManager + Send + Sync>,
        model_executor: Arc<dyn ModelExecutor + Send + Sync>,
        tensor_factory: Arc<dyn TensorFactory>,
    ) -> Result<Self> {
        Self::new_with_speculation(
            config,
            scheduler,
            tokenizer,
            sampler,
            kv_cache,
            model_executor,
            tensor_factory,
            None,
            None,
        )
    }

    /// Build an engine with optional speculative decoding. Pass both the
    /// draft executor AND the config together — either both or neither.
    #[allow(clippy::too_many_arguments)]
    pub fn new_with_speculation(
        config: EngineConfig,
        scheduler: Arc<ContinuousBatchScheduler>,
        tokenizer: Arc<dyn Tokenizer + Send + Sync>,
        sampler: Arc<dyn Sampler + Send + Sync>,
        kv_cache: Arc<dyn KvCacheManager + Send + Sync>,
        model_executor: Arc<dyn ModelExecutor + Send + Sync>,
        tensor_factory: Arc<dyn TensorFactory>,
        draft_executor: Option<Arc<dyn ModelExecutor + Send + Sync>>,
        spec_config: Option<crate::speculative::SpeculativeDecodingConfig>,
    ) -> Result<Self> {
        Self::new_with_speculation_and_recurrent_state_manager(
            config,
            scheduler,
            tokenizer,
            sampler,
            kv_cache,
            model_executor,
            tensor_factory,
            draft_executor,
            spec_config,
            None,
        )
    }

    /// Build an engine with optional speculative decoding and an optional
    /// recurrent-state manager for state-space / hybrid models.
    #[allow(clippy::too_many_arguments)]
    pub fn new_with_speculation_and_recurrent_state_manager(
        config: EngineConfig,
        scheduler: Arc<ContinuousBatchScheduler>,
        tokenizer: Arc<dyn Tokenizer + Send + Sync>,
        sampler: Arc<dyn Sampler + Send + Sync>,
        kv_cache: Arc<dyn KvCacheManager + Send + Sync>,
        model_executor: Arc<dyn ModelExecutor + Send + Sync>,
        tensor_factory: Arc<dyn TensorFactory>,
        draft_executor: Option<Arc<dyn ModelExecutor + Send + Sync>>,
        spec_config: Option<crate::speculative::SpeculativeDecodingConfig>,
        recurrent_state_manager: Option<Arc<dyn RecurrentStateManager + Send + Sync>>,
    ) -> Result<Self> {
        Self::new_with_resource_composition(
            config,
            scheduler,
            tokenizer,
            sampler,
            EngineResourceComposition::LegacyEngine {
                kv_cache,
                recurrent_state_manager,
            },
            model_executor,
            tensor_factory,
            draft_executor,
            spec_config,
        )
    }

    /// Build an engine bound to the shared plan runtime, which is the sole
    /// owner of request-lifetime KV, recurrent state, and backing capacity.
    /// The model executor adapts that runtime but does not own a second
    /// resource manager; no legacy engine manager is created or retained.
    pub fn new_plan_runtime(
        config: EngineConfig,
        scheduler: Arc<ContinuousBatchScheduler>,
        tokenizer: Arc<dyn Tokenizer + Send + Sync>,
        sampler: Arc<dyn Sampler + Send + Sync>,
        model_executor: Arc<dyn ModelExecutor + Send + Sync>,
        tensor_factory: Arc<dyn TensorFactory>,
    ) -> Result<Self> {
        Self::new_with_resource_composition(
            config,
            scheduler,
            tokenizer,
            sampler,
            EngineResourceComposition::PlanRuntime,
            model_executor,
            tensor_factory,
            None,
            None,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn new_with_resource_composition(
        config: EngineConfig,
        scheduler: Arc<ContinuousBatchScheduler>,
        tokenizer: Arc<dyn Tokenizer + Send + Sync>,
        sampler: Arc<dyn Sampler + Send + Sync>,
        resource_composition: EngineResourceComposition,
        model_executor: Arc<dyn ModelExecutor + Send + Sync>,
        tensor_factory: Arc<dyn TensorFactory>,
        draft_executor: Option<Arc<dyn ModelExecutor + Send + Sync>>,
        spec_config: Option<crate::speculative::SpeculativeDecodingConfig>,
    ) -> Result<Self> {
        let executor_authority = model_executor.execution_resource_authority();
        if draft_executor.is_some() != spec_config.is_some() {
            return Err(FerrumError::config(
                "speculative decoding requires both a draft executor and its configuration",
            ));
        }
        if let Some(draft_executor) = draft_executor.as_ref() {
            let draft_authority = draft_executor.execution_resource_authority();
            if draft_authority != executor_authority {
                return Err(FerrumError::config(format!(
                    "draft executor authority {draft_authority:?} does not match target authority {executor_authority:?}"
                )));
            }
        }
        if resource_composition.authority() != executor_authority {
            return Err(FerrumError::config(format!(
                "engine resource composition {:?} does not match executor authority {:?}",
                resource_composition.authority(),
                executor_authority
            )));
        }
        let recurrent_state_manager = resource_composition.recurrent_state_manager().is_some();
        info!(
            ?executor_authority,
            "Creating ContinuousBatchEngine (speculative_decoding={}, recurrent_state_manager={})",
            draft_executor.is_some() && spec_config.is_some(),
            recurrent_state_manager
        );
        let runtime_config = ContinuousEngineRuntimeConfig::from_engine_config(&config);
        let profile_trace_jsonl = runtime_config
            .profile_jsonl
            .as_ref()
            .map(|path| {
                SchedulerTraceJournal::create(path.clone()).map_err(|error| {
                    FerrumError::io(format!(
                        "open product profile JSONL {}: {error}",
                        path.display()
                    ))
                })
            })
            .transpose()?;
        let scheduler_trace_jsonl = match runtime_config.scheduler_trace_jsonl.as_deref() {
            Some(path)
                if profile_trace_jsonl
                    .as_ref()
                    .is_some_and(|journal| journal.path() == path) =>
            {
                profile_trace_jsonl.clone()
            }
            path => create_scheduler_trace_sink(path),
        };
        let legacy_scheduler_trace_jsonl = create_legacy_scheduler_trace_sink(
            runtime_config.legacy_scheduler_trace_jsonl.as_deref(),
        );
        let mut execution_profile_journals = Vec::with_capacity(2);
        if let Some(journal) = profile_trace_jsonl.as_ref() {
            execution_profile_journals.push(journal.clone());
        }
        if let Some(journal) = scheduler_trace_jsonl.as_ref() {
            if execution_profile_journals
                .iter()
                .all(|existing| existing.path() != journal.path())
            {
                execution_profile_journals.push(journal.clone());
            }
        }
        if !execution_profile_journals.is_empty() {
            let sink: Arc<dyn ExecutionEventSink> =
                Arc::new(VNextProfileExecutionEventSink::with_journals(
                    execution_profile_journals,
                    runtime_config
                        .profile_entrypoint
                        .unwrap_or(ProfileEntrypoint::Synthetic),
                    &config,
                ));
            model_executor.attach_execution_event_sink(Arc::clone(&sink));
            if let Some(draft_executor) = draft_executor.as_ref() {
                draft_executor.attach_execution_event_sink(sink);
            }
        }

        Ok(Self {
            inner: Arc::new(EngineInner {
                config,
                scheduler,
                tokenizer,
                structured_output_factory: OnceLock::new(),
                sampler,
                resource_composition,
                model_executor,
                draft_executor,
                spec_config,
                tensor_factory,
                sequences: RwLock::new(HashMap::new()),
                is_running: AtomicBool::new(false),
                shutdown_notify: Arc::new(Notify::new()),
                iteration_lock: tokio::sync::Mutex::new(()),
                work_notify: Arc::new(Notify::new()),
                iteration_count: AtomicU64::new(0),
                prefix_cache: PrefixCache::new(256, 2),
                runtime_config,
                profile_trace_jsonl,
                scheduler_trace_jsonl,
                legacy_scheduler_trace_jsonl,
                scheduler_trace_none_streak: AtomicU64::new(0),
                resource_lifecycle: Mutex::new(ResourceLifecycleLedger::default()),
                resource_trace_event_counter: AtomicU64::new(0),
                dynamic_admission_availability: Mutex::new(Vec::with_capacity(16)),
                total_prefill_tokens: AtomicU64::new(0),
                total_decode_tokens: AtomicU64::new(0),
                total_preemptions: AtomicU64::new(0),
                prefix_cache_hits: AtomicU64::new(0),
                total_iteration_lock_wait_us: AtomicU64::new(0),
                iteration_lock_wait_samples: AtomicU64::new(0),
                total_scheduling_time_us: AtomicU64::new(0),
                scheduling_time_samples: AtomicU64::new(0),
                total_model_execution_time_us: AtomicU64::new(0),
                model_execution_time_samples: AtomicU64::new(0),
                bg_loop_spawned: AtomicBool::new(false),
                shutdown_started: AtomicBool::new(false),
                shutdown_lock: tokio::sync::Mutex::new(()),
                background_loop: Mutex::new(None),
            }),
        })
    }

    /// Spawn the background iteration loop on first request. Without this,
    /// every concurrent infer/infer_stream call spawned its own
    /// drive_to_completion task → 16 streaming requests = 16 tasks all
    /// racing for `iteration_lock` (thundering herd, observed as ~5ms of
    /// per-iter tokio scheduling overhead at c=16). With one bg loop +
    /// per-request tasks just consuming their channel, lock is uncontested.
    fn ensure_bg_loop(&self) {
        if self.inner.bg_loop_spawned.load(Ordering::Acquire) {
            return;
        }
        let mut background_loop = self.inner.background_loop.lock();
        if self.inner.shutdown_started.load(Ordering::Acquire) {
            return;
        }
        if !self.inner.bg_loop_spawned.swap(true, Ordering::AcqRel) {
            *background_loop = Some(self.start_loop());
        }
    }

    /// Hit count since engine construction (prefix cache). Exposed for
    /// tests + /metrics endpoint; monotonic, Relaxed-ordered.
    pub fn prefix_cache_hits(&self) -> u64 {
        self.inner.prefix_cache_hits.load(Ordering::Relaxed)
    }

    /// Snapshot of prefix cache stats (hits/misses/evictions/active entries).
    pub fn prefix_cache_stats(&self) -> ferrum_kv::cache::prefix::PrefixCacheStats {
        self.inner.prefix_cache.stats()
    }

    /// Start a background iteration loop.  Returns a `JoinHandle` that
    /// runs until `shutdown()` is called.  When a background loop is
    /// active, `infer()` / `infer_stream()` simply submit and wait.
    pub fn start_loop(&self) -> tokio::task::JoinHandle<()> {
        let inner = self.inner.clone();
        inner.is_running.store(true, Ordering::SeqCst);
        tokio::spawn(async move {
            info!("Background iteration loop started");
            let prof = inner.runtime_config.batch_decode_prof;
            let mut last_iter_end: Option<std::time::Instant> = None;
            static GAP_PROF_CALLS: std::sync::atomic::AtomicU64 =
                std::sync::atomic::AtomicU64::new(0);
            loop {
                if !inner.is_running.load(Ordering::SeqCst) {
                    break;
                }
                let inter_iter_us = if let Some(prev) = last_iter_end {
                    Some(prev.elapsed().as_micros() as u64)
                } else {
                    None
                };
                let outcome = match inner.run_iteration().await {
                    Ok(outcome) => outcome,
                    Err(error) => {
                        warn!("Iteration error: {}", error);
                        EngineIterationOutcome::Progressed
                    }
                };
                if prof {
                    let n = GAP_PROF_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    if n.is_multiple_of(8) {
                        if let Some(gap_us) = inter_iter_us {
                            eprintln!("[bg-loop-gap] call#{} inter_iter={}us", n, gap_us);
                        }
                    }
                }
                last_iter_end = Some(std::time::Instant::now());
                if !inner.is_running.load(Ordering::SeqCst) {
                    break;
                }
                match outcome {
                    EngineIterationOutcome::Progressed => tokio::task::yield_now().await,
                    EngineIterationOutcome::Idle => {
                        tokio::select! {
                            _ = inner.shutdown_notify.notified() => {}
                            _ = inner.work_notify.notified() => {}
                        }
                    }
                    EngineIterationOutcome::CapacityBlocked(registration) => {
                        tokio::select! {
                            _ = inner.shutdown_notify.notified() => {}
                            _ = inner.work_notify.notified() => {}
                            result = registration.wait_for_change() => {
                                if let Err(error) = result {
                                    warn!("Executor capacity wait error: {}", error);
                                }
                            }
                        }
                    }
                }
            }
            info!("Background iteration loop stopped");
        })
    }
}

#[async_trait]
impl LlmInferenceEngine for ContinuousBatchEngine {
    async fn infer(&self, mut request: InferenceRequest) -> Result<InferenceResponse> {
        let request_id = request.id.clone();
        let infer_start = Instant::now();
        counter!("ferrum.engine.requests_total").increment(1);

        maybe_trace_prompt_tokens(&*self.inner.tokenizer, &request_id, &request.prompt);
        let input_tokens = self.inner.tokenizer.encode(&request.prompt, true)?;
        clamp_default_max_tokens_to_context(
            &mut request,
            input_tokens.len(),
            &self.inner.config,
            &self.inner.runtime_config,
            self.inner.model_executor.kv_capacity(),
        );
        validate_request_context_budget(
            &request,
            input_tokens.len(),
            &self.inner.config,
            &self.inner.runtime_config,
            self.inner.model_executor.kv_capacity(),
        )?;
        request.metadata.insert(
            PROMPT_TOKENS_METADATA_KEY.to_string(),
            serde_json::Value::from(input_tokens.len() as u64),
        );

        // Publish the tokenized sequence and scheduler item atomically with
        // respect to the iteration driver. Typed admission must never observe
        // one without the other.
        let (resp_tx, resp_rx) = tokio::sync::oneshot::channel();
        let mut receiver_drop_wake =
            ClientReceiverDropWake::new(Arc::clone(&self.inner.work_notify));
        let structured_factory = if matches!(
            &request.sampling_params.response_format,
            ferrum_types::ResponseFormat::Text
        ) {
            None
        } else {
            Some(self.inner.structured_output_factory()?)
        };
        let mut seq_state =
            SequenceState::try_new_with_tokenizer_model_vocab_and_structured_factory(
                request.clone(),
                input_tokens,
                Some(self.inner.tokenizer.clone()),
                Some(self.inner.model_executor.info().vocab_size),
                structured_factory.as_deref(),
            )?;
        gauge!("ferrum.engine.active_requests").increment(1.0);
        let request_slot = RequestSlotLease::open(&self.inner, request_id.clone());
        seq_state.response_sender = Some(resp_tx);
        seq_state.request_slot = Some(request_slot);
        {
            let _iteration = self.inner.iteration_lock.lock().await;
            {
                let mut sequences = self.inner.sequences.write();
                if sequences.contains_key(&request_id) {
                    let error = FerrumError::already_exists(format!(
                        "request {} is already active",
                        request_id
                    ));
                    if let Some(request_slot) = seq_state.request_slot.take() {
                        request_slot.reject(&self.inner, error.to_string());
                    }
                    gauge!("ferrum.engine.active_requests").decrement(1.0);
                    return Err(error);
                }
                sequences.insert(request_id.clone(), seq_state);
            }
            if let Err(error) = self.inner.scheduler.submit(request).await {
                let mut sequence = self
                    .inner
                    .sequences
                    .write()
                    .remove(&request_id)
                    .expect("just-published sequence remains present after submit failure");
                if let Some(request_slot) = sequence.request_slot.take() {
                    request_slot.reject(&self.inner, error.to_string());
                }
                gauge!("ferrum.engine.active_requests").decrement(1.0);
                return Err(error);
            }
            self.inner
                .sequences
                .write()
                .get_mut(&request_id)
                .and_then(|sequence| sequence.request_slot.as_mut())
                .expect("submitted sequence retains its request slot")
                .admit(&self.inner);
        }

        // Make sure the single shared bg loop is running, then just wait
        // for our oneshot to fire. Avoids per-request drive_to_completion
        // contention on iteration_lock.
        self.ensure_bg_loop();
        self.inner.work_notify.notify_one();

        let result = resp_rx.await.unwrap_or_else(|_| {
            Err(FerrumError::internal(
                "Response channel closed before response was sent",
            ))
        });
        receiver_drop_wake.disarm();

        gauge!("ferrum.engine.active_requests").decrement(1.0);
        let elapsed_ms = infer_start.elapsed().as_secs_f64() * 1000.0;
        histogram!("ferrum.engine.request_duration_ms").record(elapsed_ms);

        if let Ok(ref resp) = result {
            counter!("ferrum.engine.requests_completed").increment(1);
            counter!("ferrum.engine.tokens_generated_total").increment(resp.tokens.len() as u64);
            // NOTE: real TTFT lives in `send_stream_update` —
            // emitted as `ferrum.engine.ttft_seconds`. The sync `infer`
            // path returns the whole response at once, so there's no
            // observable first-token moment to record here.
        } else {
            counter!("ferrum.engine.requests_failed").increment(1);
        }

        result
    }

    async fn infer_stream(
        &self,
        mut request: InferenceRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk>> + Send>>> {
        let (tx, rx) = mpsc::channel(100);
        let receiver_drop_wake = ClientReceiverDropWake::new(Arc::clone(&self.inner.work_notify));
        let request_id = request.id.clone();

        maybe_trace_prompt_tokens(&*self.inner.tokenizer, &request_id, &request.prompt);
        let input_tokens = self.inner.tokenizer.encode(&request.prompt, true)?;
        clamp_default_max_tokens_to_context(
            &mut request,
            input_tokens.len(),
            &self.inner.config,
            &self.inner.runtime_config,
            self.inner.model_executor.kv_capacity(),
        );
        validate_request_context_budget(
            &request,
            input_tokens.len(),
            &self.inner.config,
            &self.inner.runtime_config,
            self.inner.model_executor.kv_capacity(),
        )?;
        request.metadata.insert(
            PROMPT_TOKENS_METADATA_KEY.to_string(),
            serde_json::Value::from(input_tokens.len() as u64),
        );

        // Publish tokenized state and the scheduler item under the same
        // iteration boundary; see the non-streaming path above.
        let structured_factory = if matches!(
            &request.sampling_params.response_format,
            ferrum_types::ResponseFormat::Text
        ) {
            None
        } else {
            Some(self.inner.structured_output_factory()?)
        };
        let mut seq_state =
            SequenceState::try_new_with_tokenizer_model_vocab_and_structured_factory(
                request.clone(),
                input_tokens,
                Some(self.inner.tokenizer.clone()),
                Some(self.inner.model_executor.info().vocab_size),
                structured_factory.as_deref(),
            )?;
        let request_slot = RequestSlotLease::open(&self.inner, request_id.clone());
        seq_state.stream_sender = Some(tx);
        seq_state.request_slot = Some(request_slot);
        {
            let _iteration = self.inner.iteration_lock.lock().await;
            {
                let mut sequences = self.inner.sequences.write();
                if sequences.contains_key(&request_id) {
                    let error = FerrumError::already_exists(format!(
                        "request {} is already active",
                        request_id
                    ));
                    if let Some(request_slot) = seq_state.request_slot.take() {
                        request_slot.reject(&self.inner, error.to_string());
                    }
                    return Err(error);
                }
                sequences.insert(request_id.clone(), seq_state);
            }
            if let Err(error) = self.inner.scheduler.submit(request).await {
                let mut sequence = self
                    .inner
                    .sequences
                    .write()
                    .remove(&request_id)
                    .expect("just-published sequence remains present after submit failure");
                if let Some(request_slot) = sequence.request_slot.take() {
                    request_slot.reject(&self.inner, error.to_string());
                }
                return Err(error);
            }
            self.inner
                .sequences
                .write()
                .get_mut(&request_id)
                .and_then(|sequence| sequence.request_slot.as_mut())
                .expect("submitted sequence retains its request slot")
                .admit(&self.inner);
        }

        // Single shared bg loop drives iters; per-request stream just
        // consumes from `rx`. Used to spawn a per-request drive_to_completion
        // task here, but with c=N concurrent streams that produced N
        // tasks all racing for `iteration_lock` — measured ~5ms/iter of
        // tokio thundering-herd overhead at c=16.
        let _ = request_id;
        self.ensure_bg_loop();
        self.inner.work_notify.notify_one();

        Ok(Box::pin(CancellationAwareResponseStream {
            receiver: tokio_stream::wrappers::ReceiverStream::new(rx),
            receiver_drop_wake,
        }))
    }
}

#[async_trait]
impl InferenceEngine for ContinuousBatchEngine {
    async fn status(&self) -> EngineStatus {
        let metrics = self.inner.scheduler.metrics();
        let (total_bytes, used_bytes, cache_memory_bytes, resource_status_ready) =
            match &self.inner.resource_composition {
                EngineResourceComposition::LegacyEngine { kv_cache, .. } => {
                    let kv_stats = kv_cache.stats();
                    (
                        kv_stats.total_memory_bytes,
                        kv_stats.used_memory_bytes,
                        kv_stats.used_memory_bytes,
                        true,
                    )
                }
                EngineResourceComposition::PlanRuntime => {
                    match self.inner.model_executor.plan_runtime_resource_snapshot() {
                        Ok(Some(snapshot)) => {
                            let total_bytes = usize::try_from(snapshot.usable_capacity_bytes())
                                .unwrap_or(usize::MAX);
                            let used_bytes = snapshot
                                .used_bytes()
                                .ok()
                                .and_then(|bytes| usize::try_from(bytes).ok())
                                .unwrap_or(usize::MAX);
                            let dynamic_used_bytes = usize::try_from(snapshot.dynamic_used_bytes())
                                .unwrap_or(usize::MAX);
                            (total_bytes, used_bytes, dynamic_used_bytes, true)
                        }
                        Ok(None) => {
                            warn!("Plan runtime did not expose its required resource snapshot");
                            (0, 0, 0, false)
                        }
                        Err(error) => {
                            warn!(error = %error, "Plan-runtime resource snapshot failed");
                            (0, 0, 0, false)
                        }
                    }
                }
            };
        let free_bytes = total_bytes.saturating_sub(used_bytes);
        let mut memory_usage = ferrum_types::MemoryUsage {
            total_bytes,
            used_bytes,
            free_bytes,
            gpu_memory_bytes: self
                .inner
                .config
                .backend
                .device
                .is_gpu()
                .then_some(used_bytes),
            cpu_memory_bytes: matches!(self.inner.config.backend.device, Device::CPU)
                .then_some(used_bytes),
            cache_memory_bytes,
            utilization_percent: 0.0,
        };
        memory_usage.calculate_utilization();
        EngineStatus {
            is_ready: resource_status_ready && self.inner.is_running.load(Ordering::SeqCst),
            loaded_models: vec![self.inner.config.model.model_id.clone()],
            active_requests: metrics.running_requests,
            queued_requests: metrics.waiting_requests,
            memory_usage,
            uptime_seconds: 0,
            last_heartbeat: chrono::Utc::now(),
            version: env!("CARGO_PKG_VERSION").to_string(),
        }
    }

    async fn shutdown(&self) -> Result<()> {
        let _shutdown_guard = self.inner.shutdown_lock.lock().await;
        info!("Shutting down continuous batch engine");
        let background_loop = {
            let mut background_loop = self.inner.background_loop.lock();
            self.inner.signal_shutdown();
            background_loop.take()
        };

        let loop_result = match background_loop {
            Some(background_loop) => background_loop.await.map_err(|error| {
                FerrumError::internal(format!("background iteration loop failed: {error}"))
            }),
            None => Ok(()),
        };

        let mut trace_journals = Vec::with_capacity(2);
        if let Some(journal) = self.inner.profile_trace_jsonl.clone() {
            trace_journals.push(journal);
        }
        if let Some(journal) = self.inner.scheduler_trace_jsonl.clone() {
            if trace_journals
                .iter()
                .all(|existing| existing.path() != journal.path())
            {
                trace_journals.push(journal);
            }
        }
        let trace_result = if trace_journals.is_empty() {
            Ok(())
        } else {
            tokio::task::spawn_blocking(move || {
                for journal in trace_journals {
                    journal.close()?;
                }
                Ok::<(), JsonlJournalError>(())
            })
            .await
            .map_err(|error| {
                FerrumError::internal(format!("scheduler trace close task failed: {error}"))
            })?
            .map_err(|error| {
                FerrumError::internal(format!("scheduler trace close failed: {error}"))
            })
        };

        loop_result?;
        trace_result
    }

    fn config(&self) -> &EngineConfig {
        &self.inner.config
    }

    fn metrics(&self) -> ferrum_types::EngineMetrics {
        let sm = self.inner.scheduler.metrics();
        ferrum_types::EngineMetrics {
            total_requests: sm.completed_requests + sm.failed_requests,
            successful_requests: sm.completed_requests,
            failed_requests: sm.failed_requests,
            avg_request_latency_ms: 0.0,
            p95_request_latency_ms: 0.0,
            p99_request_latency_ms: 0.0,
            throughput_rps: sm.throughput_rps as f32,
            tokens_per_second: 0.0,
            queue_metrics: ferrum_types::QueueMetrics {
                current_queue_length: sm.waiting_requests,
                avg_queue_wait_time_ms: sm.avg_wait_time_ms,
                queue_throughput_rps: sm.throughput_rps as f32,
                queue_rejection_rate: 0.0,
            },
            resource_utilization: Default::default(),
            error_stats: Default::default(),
            performance_breakdown: self.inner.performance_breakdown(),
        }
    }

    fn cache_metrics_snapshot(&self) -> Option<serde_json::Value> {
        if let Some(snapshot) = self.inner.model_executor.cache_metrics_snapshot() {
            return Some(snapshot);
        }

        let stats = self.inner.prefix_cache.stats();
        Some(serde_json::json!({
            "position": "engine-whole-prompt-debug-cache",
            "source": "continuous-engine-whole-prompt-prefix-cache",
            "enabled": self.inner.runtime_config.prefix_cache_enabled,
            "hits": stats.hits as u64,
            "misses": stats.misses as u64,
            "evictions": stats.evictions as u64,
            "saved_prefill_tokens": self.inner.prefix_cache_hits.load(Ordering::Relaxed),
            "entries": stats.active_prefixes as u64,
            "bytes": 0u64,
            "cached_tokens": stats.total_cached_tokens as u64,
            "hit_rate": stats.hit_rate,
        }))
    }

    fn admission_snapshot(
        &self,
    ) -> ferrum_types::Result<Option<ferrum_types::ExecutorAdmissionSnapshot>> {
        let scheduler = self.inner.scheduler.admission_phase_counts();
        let authority = self.inner.resource_composition.authority();
        let limits = match authority {
            ExecutionResourceAuthority::PlanRuntime => self
                .inner
                .model_executor
                .admission_limits()?
                .ok_or_else(|| {
                    FerrumError::internal(
                        "PlanRuntime executor did not expose its resolved admission limits",
                    )
                })?,
            ExecutionResourceAuthority::LegacyEngine => {
                let scheduler_limit = self.inner.config.scheduler.max_running_requests;
                let maximum_active_sequences = self
                    .inner
                    .recurrent_state_manager()
                    .map(|manager| scheduler_limit.min(manager.stats().total_batch_slots))
                    .unwrap_or(scheduler_limit);
                ferrum_types::ExecutorAdmissionLimits::new(
                    u32::try_from(maximum_active_sequences).map_err(|_| {
                        FerrumError::internal("legacy admission sequence limit exceeds u32")
                    })?,
                    u64::try_from(self.inner.config.batching.max_num_batched_tokens).map_err(
                        |_| FerrumError::internal("legacy scheduled-token limit exceeds u64"),
                    )?,
                )
                .map_err(|reason| {
                    FerrumError::internal(format!(
                        "legacy admission limits violated their typed contract: {reason}"
                    ))
                })?
            }
        };
        ferrum_types::ExecutorAdmissionSnapshot::new(
            authority,
            limits,
            u32::try_from(scheduler.waiting_requests)
                .map_err(|_| FerrumError::internal("waiting request count exceeds u32"))?,
            u32::try_from(scheduler.active_prefill_sequences)
                .map_err(|_| FerrumError::internal("active prefill count exceeds u32"))?,
            u32::try_from(scheduler.active_decode_sequences)
                .map_err(|_| FerrumError::internal("active decode count exceeds u32"))?,
            None,
            None,
        )
        .map(Some)
        .map_err(|reason| {
            FerrumError::internal(format!(
                "runtime admission snapshot violated its typed contract: {reason}"
            ))
        })
    }

    fn lora_metrics_snapshot(&self) -> Option<serde_json::Value> {
        self.inner.model_executor.lora_metrics_snapshot()
    }

    async fn health_check(&self) -> ferrum_types::HealthStatus {
        if self.inner.is_running.load(Ordering::SeqCst) {
            ferrum_types::HealthStatus::healthy()
        } else {
            ferrum_types::HealthStatus {
                status: ferrum_types::HealthStatusType::Unhealthy,
                component_status: ferrum_types::ComponentStatus::healthy(),
                last_check: chrono::Utc::now(),
            }
        }
    }
}

impl std::fmt::Debug for ContinuousBatchEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ContinuousBatchEngine")
            .field("is_running", &self.inner.is_running.load(Ordering::SeqCst))
            .field(
                "iteration_count",
                &self.inner.iteration_count.load(Ordering::SeqCst),
            )
            .field("active_sequences", &self.inner.sequences.read().len())
            .finish()
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Unit tests
// ────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests;
