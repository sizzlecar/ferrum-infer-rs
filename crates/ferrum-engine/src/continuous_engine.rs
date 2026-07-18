//! Continuous Batching Engine
//!
//! Iteration-level continuous batching: each step processes a mixed batch of
//! prefill and decode requests selected by the scheduler.  Multiple callers
//! can submit requests concurrently — an `iteration_lock` serializes the
//! actual engine steps so each batch is processed exactly once.

use crate::resource_lifecycle::{
    ResourceLedgerTransition, ResourceLifecycleLedger, ResourceOwnerCloseSummary,
};
use async_trait::async_trait;
use ferrum_bench_core::{global_profile, profile_fields_from_json};
use ferrum_interfaces::{
    engine::{InferenceEngine, LlmInferenceEngine},
    kv_cache::AllocationRequest,
    model_executor::{
        ExecutionResourceAuthority, ExecutorAdmissionEpochs, ExecutorBatchDecodeOutcome,
        ExecutorCapacityWaitRegistration, ExecutorExecutionCapacityDeferral,
        ExecutorExecutionCapacityPreemption, ExecutorPrefillAdmission,
        ExecutorPrefillAdmissionDecision, ExecutorPrefillAdmissionReceipt,
        ExecutorPrefillMaintenanceDeferral, ExecutorPrefillMaintenanceOutcome,
        ExecutorSequenceCompletion, GreedyRepetitionPenalty, KvSlotRequest, LogitsReturnPolicy,
        TokenSelectionMask,
    },
    vnext::{
        AdmissionDeferred, AdmissionRejected, CapacityAvailabilityEpoch, DeferredAction,
        DeviceCapacityPressureScope, EventEmissionPermit, ExecutionEvent,
        ExecutionEventCapturePolicy, ExecutionEventDetail,
        ExecutionEventKind as VNextExecutionEventKind, ExecutionEventSink, ExecutionEventSinkError,
    },
    KvCacheHandle, KvCacheManager, ModelExecutor, RecurrentStateHandle, RecurrentStateManager,
    Sampler, SchedulerInterface as Scheduler, TensorFactory, TensorRef, Tokenizer,
};
use ferrum_kv::cache::prefix::PrefixCache;
use ferrum_sampler::json_mode::JsonModeProcessor;
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
    InferenceRequest, InferenceResponse, Priority, ProfileEntrypoint, ProfileError,
    ProfileEventKind, ProfileStatus, RequestId, ResourceAction, ResourceTraceEvent, Result,
    SamplingParams, StreamChunk, TokenId, TokenUsage, DEFAULT_MAX_TOKENS_METADATA_KEY,
    ENGINE_RUNTIME_TRACE_PRESET_HASH, OBSERVABILITY_PROFILE_SCHEMA_VERSION,
    PROMPT_TOKENS_METADATA_KEY,
};
use futures::stream::Stream;
use metrics::{counter, gauge, histogram};
use parking_lot::{Mutex, RwLock};
use rand::{rngs::StdRng, SeedableRng};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::sync::OnceLock;
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

static TOKEN_POLICY_CACHE: OnceLock<std::sync::Mutex<HashMap<(usize, usize), HashSet<u32>>>> =
    OnceLock::new();

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
) -> (HashSet<u32>, Vec<String>) {
    let mut ids: HashSet<u32> = HashSet::new();
    let mut text_seqs: Vec<String> = Vec::new();

    if let Some(tok) = tokenizer {
        if !ignore_eos {
            if let Some(eos) = tok.special_tokens().eos_token {
                ids.insert(eos.get());
            }
            for extra in &tok.special_tokens().extra_eos_tokens {
                ids.insert(extra.get());
            }
            for name in ["<|im_end|>", "<|endoftext|>", "<|eot_id|>", "</s>"] {
                if let Some(t) = tok.token_id(name) {
                    ids.insert(t.get());
                }
            }
        }
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
    (ids, text_seqs)
}

fn resolve_sampling_token_constraints(
    tokenizer: Option<&(dyn Tokenizer + Send + Sync)>,
    stop_token_ids: &HashSet<u32>,
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
    tok: &(dyn Tokenizer + Send + Sync),
    allowed_generated_controls: &HashSet<u32>,
) -> HashSet<u32> {
    let key = (tokenizer_cache_key(tok), tok.vocab_size());
    let cache = TOKEN_POLICY_CACHE.get_or_init(|| std::sync::Mutex::new(HashMap::new()));
    if let Some(cached) = cache.lock().expect("token policy cache poisoned").get(&key) {
        return cached.clone();
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
        if !allowed_generated_controls.contains(&token.get()) {
            forbidden.insert(token.get());
        }
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
            if !allowed_generated_controls.contains(&token.get()) {
                forbidden.insert(token.get());
            }
        }
    }
    for token_id in 0..scan_limit {
        let id = token_id as u32;
        if allowed_generated_controls.contains(&id) {
            continue;
        }
        let token = TokenId::new(id);
        let raw_text = tok.token_text(token);
        let missing_token_text = has_reverse_vocab && raw_text.is_none();
        let raw_text_forbidden = raw_text.is_some_and(is_forbidden_generation_token_text);
        let decoded_text_forbidden = tok
            .decode(&[token], true)
            .map(|text| is_forbidden_generation_token_text(&text))
            .unwrap_or(true);
        if missing_token_text || raw_text_forbidden || decoded_text_forbidden {
            forbidden.insert(id);
        }
    }

    cache
        .lock()
        .expect("token policy cache poisoned")
        .insert(key, forbidden.clone());
    forbidden
}

fn tokenizer_cache_key(tok: &(dyn Tokenizer + Send + Sync)) -> usize {
    let ptr = tok as *const (dyn Tokenizer + Send + Sync);
    ptr.cast::<()>() as usize
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

/// State of a running sequence in the continuous batch.
#[derive(Debug)]
pub struct SequenceState {
    pub request_id: RequestId,
    /// Original request — kept for re-submission after preemption.
    pub original_request: InferenceRequest,
    pub input_tokens: Vec<TokenId>,
    pub generated_tokens: Vec<TokenId>,
    model_kv: Option<SequenceModelKvState>,
    recurrent_state: Option<SequenceRecurrentState>,
    pub sampling_params: SamplingParams,
    pub phase: RequestPhase,
    pub rng: StdRng,
    pub prefill_complete: bool,
    /// Number of prompt tokens already written into the model KV cache by
    /// opt-in unified chunked prefill. Zero for the normal full-prefill path.
    pub prefill_tokens_processed: usize,
    pub stream_sender: Option<mpsc::Sender<Result<StreamChunk>>>,
    pub response_sender: Option<tokio::sync::oneshot::Sender<InferenceResponse>>,
    request_slot: Option<RequestSlotLease>,
    pub start_time: Instant,
    /// Wall-clock `Instant` at which the first SSE chunk was actually
    /// sent to the client stream. Populated lazily by `send_stream_update`
    /// the first time a non-empty delta is emitted (multi-byte UTF-8
    /// buffering can defer that past the first scheduler-completed token).
    /// Used to record `ferrum.engine.ttft_seconds` and as the start point
    /// of the TPOT window.
    pub first_emit_at: Option<Instant>,
    /// Wall-clock `Instant` at the most recent successfully-sent chunk.
    /// Used to compute per-token ITL deltas (`ferrum.engine.itl_seconds`).
    pub last_emit_at: Option<Instant>,
    /// Count of stream chunks successfully sent to the client. Lags
    /// `generated_tokens.len()` by the number of tokens currently buffered
    /// for a multi-byte UTF-8 sequence (so a Chinese char split across
    /// 2 BPE tokens emits once, increments the count by 1).
    pub emitted_chunks: u32,
    pub tokens_this_iteration: usize,
    /// Number of times this request has been preempted.
    pub preemption_count: usize,
    /// JSON mode logits processor (active when response_format is JsonObject).
    pub json_processor: Option<Arc<JsonModeProcessor>>,
    /// Regex-guided hard-mask processor (active when response_format is Regex).
    pub regex_processor: Option<Arc<ferrum_sampler::guided::RegexGuidedProcessor>>,
    draft_kv: Option<SequenceDraftKvState>,
    /// Token frequency counts for repetition penalty.
    pub token_frequencies: HashMap<TokenId, usize>,
    /// Single-token stop ids: model's EOS + any `stop_sequences` that encode to
    /// exactly one token. Checked against the last generated token each step
    /// — replaces the old "token id near top of vocab = EOS" placeholder. Built
    /// from `tokenizer.eos_token`, a common-EOS fallback list (`</s>`,
    /// `<|im_end|>`, `<|endoftext|>`, `<|eot_id|>`), and one-token encodings of
    /// `sampling_params.stop_sequences`.
    pub stop_token_ids: HashSet<u32>,
    /// Token IDs that should never be sampled as normal output. Used for
    /// tokenizer/model vocab holes such as Qwen3's reserved tail IDs and
    /// literal `<unk` / `<unk>` pieces.
    pub forbidden_token_ids: HashSet<u32>,
    /// Token IDs masked only before the first generated token.
    pub initial_forbidden_token_ids: HashSet<u32>,
    /// Base tokenizer vocabulary size. IDs above this are allowed only when
    /// they are explicitly whitelisted in `allowed_extended_token_ids`.
    pub tokenizer_base_vocab_size: Option<usize>,
    pub allowed_extended_token_ids: HashSet<u32>,
    /// Multi-token text stop sequences (`stop_sequences` entries that don't
    /// resolve to a single token). Checked via accumulated decoded text.
    pub stop_text_seqs: Vec<String>,
    /// Base token-validity mask for model-side greedy argmax.
    pub argmax_token_mask: Option<TokenSelectionMask>,
    /// First-token variant that also applies `initial_forbidden_token_ids`.
    pub initial_argmax_token_mask: Option<TokenSelectionMask>,
    /// Bytes of decoded `generated_tokens` already flushed via the stream
    /// channel. Used by `send_stream_update` to compute per-call delta from
    /// the full-history decode, so multi-byte UTF-8 sequences (Chinese chars,
    /// emoji) that span several BPE tokens don't get rendered as
    /// `\u{FFFD}` replacement chars when decoded one token at a time.
    pub streamed_text_len: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct SequenceKvAllocation {
    request_id: RequestId,
    blocks: usize,
}

impl SequenceKvAllocation {
    fn new(request_id: RequestId, blocks: usize) -> Self {
        Self {
            request_id,
            blocks: blocks.max(1),
        }
    }
}

#[derive(Debug, Clone)]
enum SequenceKvRelease {
    /// The model runtime is the only release authority. This covers vNext
    /// plan-runtime leases and cloned prefix-cache references; neither has a
    /// second allocation in the legacy engine KV manager.
    RuntimeManaged,
    /// Transitional legacy composition: release the model cache reference and
    /// the exact engine KV-manager allocation together.
    LegacyAllocated(SequenceKvAllocation),
}

#[derive(Debug, Clone)]
struct SequenceModelKvState {
    cache: Arc<dyn KvCacheHandle>,
    model_cache_id: String,
    release: SequenceKvRelease,
}

impl SequenceModelKvState {
    fn runtime_managed(cache: Arc<dyn KvCacheHandle>) -> Self {
        let model_cache_id = cache.cache_id();
        Self {
            cache,
            model_cache_id,
            release: SequenceKvRelease::RuntimeManaged,
        }
    }

    fn legacy_allocated(cache: Arc<dyn KvCacheHandle>, allocation: SequenceKvAllocation) -> Self {
        let model_cache_id = cache.cache_id();
        Self {
            cache,
            model_cache_id,
            release: SequenceKvRelease::LegacyAllocated(allocation),
        }
    }

    fn handle(&self) -> Arc<dyn KvCacheHandle> {
        self.cache.clone()
    }

    fn legacy_allocation(&self) -> Option<&SequenceKvAllocation> {
        match &self.release {
            SequenceKvRelease::RuntimeManaged => None,
            SequenceKvRelease::LegacyAllocated(allocation) => Some(allocation),
        }
    }

    fn model_cache_id(&self) -> &str {
        &self.model_cache_id
    }

    fn validate_replacement_cache(&self, cache: &Arc<dyn KvCacheHandle>) -> Result<()> {
        let replacement_cache_id = cache.cache_id();
        if replacement_cache_id != self.model_cache_id() {
            return Err(FerrumError::internal(format!(
                "decode replaced model cache authority {} with {}",
                self.model_cache_id(),
                replacement_cache_id
            )));
        }
        Ok(())
    }

    fn replace_cache_handle(&mut self, cache: Arc<dyn KvCacheHandle>) -> Result<()> {
        self.validate_replacement_cache(&cache)?;
        self.cache = cache;
        Ok(())
    }

    fn into_physical_resources(self) -> (Option<SequenceKvAllocation>, String) {
        let legacy_allocation = match self.release {
            SequenceKvRelease::RuntimeManaged => None,
            SequenceKvRelease::LegacyAllocated(allocation) => Some(allocation),
        };
        (legacy_allocation, self.model_cache_id)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SequenceRecurrentAllocation {
    slots: Option<usize>,
}

impl SequenceRecurrentAllocation {
    fn new(slots: Option<usize>) -> Self {
        Self {
            slots: slots.map(|slots| slots.max(1)),
        }
    }
}

#[derive(Debug, Clone)]
struct SequenceRecurrentState {
    handle: Arc<dyn RecurrentStateHandle>,
    slots: Option<usize>,
}

impl SequenceRecurrentState {
    fn new(handle: Arc<dyn RecurrentStateHandle>, slots: Option<usize>) -> Self {
        Self {
            handle,
            slots: slots.map(|slots| slots.max(1)),
        }
    }

    fn handle(&self) -> Arc<dyn RecurrentStateHandle> {
        self.handle.clone()
    }

    fn allocation(self) -> SequenceRecurrentAllocation {
        SequenceRecurrentAllocation::new(self.slots)
    }
}

#[derive(Debug, Clone)]
struct SequenceDraftKvState {
    cache: Arc<dyn KvCacheHandle>,
    request_id: RequestId,
    resource_blocks: usize,
}

impl SequenceDraftKvState {
    fn new(cache: Arc<dyn KvCacheHandle>, request_id: RequestId, resource_blocks: usize) -> Self {
        Self {
            cache,
            request_id,
            resource_blocks: resource_blocks.max(1),
        }
    }

    fn allocation(self) -> SequenceKvAllocation {
        SequenceKvAllocation::new(self.request_id, self.resource_blocks)
    }
}

#[derive(Debug, Default)]
struct SequencePhysicalResources {
    legacy_kv_allocation: Option<SequenceKvAllocation>,
    legacy_draft_kv_allocation: Option<SequenceKvAllocation>,
    recurrent_state_allocation: Option<SequenceRecurrentAllocation>,
    model_cache_id: Option<String>,
}

#[cfg(test)]
impl SequencePhysicalResources {
    fn model_cache_id(&self) -> Option<&str> {
        self.model_cache_id.as_deref()
    }
}

#[derive(Debug, Default)]
struct SequenceCompletionResources {
    physical: SequencePhysicalResources,
    request_slot: Option<RequestSlotLease>,
}

#[derive(Debug, Default)]
#[must_use = "unified prefill owned resources must be released or committed"]
struct UnifiedPrefillOwnedResources {
    legacy_kv_allocation: Option<SequenceKvAllocation>,
    recurrent_state_allocation: Option<SequenceRecurrentAllocation>,
}

impl UnifiedPrefillOwnedResources {
    fn with_fresh_kv(mut self, allocation: SequenceKvAllocation) -> Self {
        self.legacy_kv_allocation = Some(allocation);
        self
    }

    fn with_fresh_recurrent_state(mut self, slots: usize) -> Self {
        self.recurrent_state_allocation = Some(SequenceRecurrentAllocation::new(Some(slots)));
        self
    }

    fn commit(mut self) {
        self.legacy_kv_allocation = None;
        self.recurrent_state_allocation = None;
    }

    fn is_empty(&self) -> bool {
        self.legacy_kv_allocation.is_none() && self.recurrent_state_allocation.is_none()
    }

    async fn release(mut self, engine: &EngineInner, owner_request_id: &RequestId) {
        if let Some(kv_allocation) = self.legacy_kv_allocation.take() {
            engine
                .release_kv_allocation(
                    owner_request_id,
                    kv_allocation.request_id,
                    kv_allocation.blocks,
                )
                .await;
        }
        if let Some(recurrent_allocation) = self.recurrent_state_allocation.take() {
            let sequence_slots = engine
                .sequences
                .write()
                .get_mut(owner_request_id)
                .and_then(SequenceState::take_recurrent_state_allocation);
            if sequence_slots != recurrent_allocation.slots {
                warn!(
                    request_id = %owner_request_id,
                    sequence_slots = ?sequence_slots,
                    owned_slots = ?recurrent_allocation.slots,
                    "unified prefill recurrent ownership metadata differed from sequence state"
                );
            }
            engine
                .release_recurrent_allocation(
                    owner_request_id,
                    recurrent_allocation.slots.or(sequence_slots),
                )
                .await;
        }
    }
}

impl Drop for UnifiedPrefillOwnedResources {
    fn drop(&mut self) {
        if self.is_empty() {
            return;
        }
        let message = "unified prefill resources dropped without explicit release or commit";
        warn!(
            legacy_kv_allocation = ?self.legacy_kv_allocation,
            recurrent_state_allocation = ?self.recurrent_state_allocation,
            "{message}"
        );
        #[cfg(test)]
        if !std::thread::panicking() {
            panic!("{message}");
        }
    }
}

#[derive(Debug, Clone)]
struct SequenceDecodeResources {
    seq_id: String,
    kv_cache: Arc<dyn KvCacheHandle>,
    recurrent_state: Option<Arc<dyn RecurrentStateHandle>>,
    last_token: TokenId,
    pos_offset: usize,
}

#[derive(Debug, Clone)]
struct SequencePrefillResources {
    kv_cache: Option<Arc<dyn KvCacheHandle>>,
    legacy_kv_allocation: Option<SequenceKvAllocation>,
    recurrent_state: Option<Arc<dyn RecurrentStateHandle>>,
    prefill_tokens_processed: usize,
}

#[cfg(test)]
impl SequencePrefillResources {
    fn kv_cache_handle(&self) -> Option<Arc<dyn KvCacheHandle>> {
        self.kv_cache.clone()
    }

    fn kv_resource_blocks(&self) -> Option<usize> {
        self.legacy_kv_allocation
            .as_ref()
            .map(|allocation| allocation.blocks)
    }
}

#[derive(Debug, Default, PartialEq, Eq)]
struct ModelCacheRefUpdate {
    released: Option<String>,
    acquired: Option<String>,
}

impl SequenceState {
    pub fn new(request: InferenceRequest, input_tokens: Vec<TokenId>) -> Self {
        Self::new_with_tokenizer(request, input_tokens, None)
    }

    /// Build sequence state, optionally wiring a tokenizer for guided-decoding
    /// processors (`ResponseFormat::Regex` needs vocab access to compile a
    /// token-level mask). Falling back to `None` preserves the old behaviour
    /// for call sites that don't have a tokenizer to hand (smoke tests).
    pub fn new_with_tokenizer(
        request: InferenceRequest,
        input_tokens: Vec<TokenId>,
        tokenizer: Option<Arc<dyn Tokenizer + Send + Sync>>,
    ) -> Self {
        Self::new_with_tokenizer_and_model_vocab_size(request, input_tokens, tokenizer, None)
    }

    pub fn new_with_tokenizer_and_model_vocab_size(
        request: InferenceRequest,
        input_tokens: Vec<TokenId>,
        tokenizer: Option<Arc<dyn Tokenizer + Send + Sync>>,
        model_vocab_size: Option<usize>,
    ) -> Self {
        use ferrum_types::ResponseFormat;
        let rng = request
            .sampling_params
            .seed
            .map(StdRng::seed_from_u64)
            .unwrap_or_else(|| {
                let mut rng = rand::rng();
                StdRng::from_rng(&mut rng)
            });
        let json_processor = match &request.sampling_params.response_format {
            ResponseFormat::JsonObject => Some(Arc::new(JsonModeProcessor::new())),
            _ => None,
        };
        let regex_processor = match (&request.sampling_params.response_format, &tokenizer) {
            (ResponseFormat::JsonSchema(schema), Some(tok)) => {
                // OpenAI-style structured output: compile the JSON Schema
                // to a regex then drive the DFA-guided processor with it.
                match ferrum_sampler::schema_to_regex::schema_to_regex(schema) {
                    Ok(pattern) => {
                        let eos = tok.special_tokens().eos_token;
                        match ferrum_sampler::guided::RegexGuidedProcessor::new(
                            &pattern,
                            tok.clone(),
                            eos,
                        ) {
                            Ok(p) => Some(Arc::new(p)),
                            Err(e) => {
                                tracing::warn!(
                                    "json_schema guided decode disabled (regex build): {e}"
                                );
                                None
                            }
                        }
                    }
                    Err(e) => {
                        tracing::warn!(
                            "json_schema guided decode disabled (schema translation): {e}"
                        );
                        None
                    }
                }
            }
            _ => None,
        };
        let ignore_eos = request
            .metadata
            .get("ferrum_ignore_eos")
            .and_then(|value| value.as_bool())
            .unwrap_or(false);
        let (stop_token_ids, stop_text_seqs) =
            resolve_stop_conditions(&request.sampling_params, tokenizer.as_deref(), ignore_eos);
        let (forbidden_token_ids, tokenizer_base_vocab_size, allowed_extended_token_ids) =
            resolve_sampling_token_constraints(tokenizer.as_deref(), &stop_token_ids);
        let mut initial_forbidden_token_ids = HashSet::new();
        let initial_forbidden_token_texts = request
            .metadata
            .get("ferrum_initial_forbidden_token_texts")
            .and_then(|value| value.as_array());
        if let (Some(texts), Some(tok)) = (initial_forbidden_token_texts, tokenizer.as_deref()) {
            for text in texts.iter().filter_map(|value| value.as_str()) {
                if let Some(token) = tok.token_id(text) {
                    initial_forbidden_token_ids.insert(token.get());
                }
            }
        }
        let empty_initial_forbidden = HashSet::new();
        let argmax_token_mask = tokenizer.as_deref().map(|tok| {
            build_argmax_token_mask(
                tok,
                model_vocab_size,
                &forbidden_token_ids,
                &empty_initial_forbidden,
                &stop_token_ids,
                &allowed_extended_token_ids,
            )
        });
        let initial_argmax_token_mask = if initial_forbidden_token_ids.is_empty() {
            None
        } else {
            tokenizer.as_deref().map(|tok| {
                build_argmax_token_mask(
                    tok,
                    model_vocab_size,
                    &forbidden_token_ids,
                    &initial_forbidden_token_ids,
                    &stop_token_ids,
                    &allowed_extended_token_ids,
                )
            })
        };
        Self {
            request_id: request.id.clone(),
            original_request: request.clone(),
            input_tokens,
            generated_tokens: Vec::new(),
            model_kv: None,
            recurrent_state: None,
            sampling_params: request.sampling_params,
            phase: RequestPhase::Waiting,
            rng,
            prefill_complete: false,
            prefill_tokens_processed: 0,
            stream_sender: None,
            response_sender: None,
            request_slot: None,
            start_time: Instant::now(),
            first_emit_at: None,
            last_emit_at: None,
            emitted_chunks: 0,
            tokens_this_iteration: 0,
            preemption_count: 0,
            json_processor,
            regex_processor,
            draft_kv: None,
            token_frequencies: HashMap::new(),
            stop_token_ids,
            forbidden_token_ids,
            initial_forbidden_token_ids,
            tokenizer_base_vocab_size,
            allowed_extended_token_ids,
            stop_text_seqs,
            argmax_token_mask,
            initial_argmax_token_mask,
            streamed_text_len: 0,
        }
    }

    pub fn total_tokens(&self) -> usize {
        self.input_tokens.len() + self.generated_tokens.len()
    }

    pub fn prefill_context_tokens(&self) -> Vec<TokenId> {
        if self.generated_tokens.is_empty() {
            return self.input_tokens.clone();
        }
        let mut tokens = Vec::with_capacity(self.input_tokens.len() + self.generated_tokens.len());
        tokens.extend_from_slice(&self.input_tokens);
        tokens.extend_from_slice(&self.generated_tokens);
        tokens
    }

    pub fn prefill_context_len(&self) -> usize {
        self.input_tokens.len() + self.generated_tokens.len()
    }

    pub fn model_decode_metadata(&self) -> HashMap<String, serde_json::Value> {
        let mut metadata = self.original_request.metadata.clone();
        if self.requires_engine_full_logits_for_sampling() {
            metadata.insert(
                "ferrum_require_full_logits".to_string(),
                serde_json::json!(true),
            );
        }
        metadata.insert(
            "ferrum_kv_capacity_hint".to_string(),
            serde_json::json!(self
                .prefill_context_len()
                .max(self.input_tokens.len() + self.sampling_params.max_tokens.saturating_sub(1))),
        );
        metadata.insert(
            KV_ADMISSION_TARGET_LEN_METADATA_KEY.to_string(),
            serde_json::json!(self.prefill_context_len()),
        );
        metadata
    }

    fn model_maximum_sequence_tokens(&self) -> usize {
        self.prefill_context_len().max(
            self.input_tokens
                .len()
                .saturating_add(self.sampling_params.max_tokens.saturating_sub(1)),
        )
    }

    fn take_physical_resources(&mut self) -> SequencePhysicalResources {
        let (legacy_kv_allocation, model_cache_id) = self
            .model_kv
            .take()
            .map(|state| {
                let (legacy_kv_allocation, model_cache_id) = state.into_physical_resources();
                (legacy_kv_allocation, Some(model_cache_id))
            })
            .unwrap_or((None, None));
        let legacy_draft_kv_allocation = self.draft_kv.take().map(SequenceDraftKvState::allocation);
        let resources = SequencePhysicalResources {
            legacy_kv_allocation,
            legacy_draft_kv_allocation,
            recurrent_state_allocation: self
                .recurrent_state
                .take()
                .map(SequenceRecurrentState::allocation),
            model_cache_id,
        };
        resources
    }

    fn take_physical_resources_for_recompute(&mut self) -> SequencePhysicalResources {
        let resources = self.take_physical_resources();
        self.prefill_complete = false;
        self.prefill_tokens_processed = 0;
        self.phase = RequestPhase::Waiting;
        self.tokens_this_iteration = 0;
        resources
    }

    fn take_completion_resources(&mut self) -> SequenceCompletionResources {
        SequenceCompletionResources {
            physical: self.take_physical_resources(),
            request_slot: self.request_slot.take(),
        }
    }

    fn model_cache_ref_update_for(&self, cache_id: &str) -> ModelCacheRefUpdate {
        if self
            .model_kv
            .as_ref()
            .is_some_and(|state| state.model_cache_id() == cache_id)
        {
            return ModelCacheRefUpdate::default();
        }
        let released = self
            .model_kv
            .as_ref()
            .map(|state| state.model_cache_id().to_string());
        ModelCacheRefUpdate {
            released,
            acquired: Some(cache_id.to_string()),
        }
    }

    fn install_model_kv_state(&mut self, state: SequenceModelKvState) -> ModelCacheRefUpdate {
        let model_cache_id = state.model_cache_id().to_string();
        let model_cache_update = self.model_cache_ref_update_for(&model_cache_id);
        self.model_kv = Some(state);
        model_cache_update
    }

    fn install_runtime_managed_model_kv(
        &mut self,
        kv_cache: Arc<dyn KvCacheHandle>,
    ) -> ModelCacheRefUpdate {
        self.install_model_kv_state(SequenceModelKvState::runtime_managed(kv_cache))
    }

    fn install_legacy_allocated_model_kv(
        &mut self,
        kv_cache: Arc<dyn KvCacheHandle>,
        allocation: SequenceKvAllocation,
    ) -> ModelCacheRefUpdate {
        self.install_model_kv_state(SequenceModelKvState::legacy_allocated(kv_cache, allocation))
    }

    fn commit_cached_prefill_physical_resources(
        &mut self,
        kv_cache: Arc<dyn KvCacheHandle>,
        prefill_tokens_processed: usize,
    ) -> ModelCacheRefUpdate {
        let model_cache_update = self.install_runtime_managed_model_kv(kv_cache);
        self.prefill_tokens_processed = prefill_tokens_processed;
        self.prefill_complete = true;
        self.phase = RequestPhase::Decoding;
        model_cache_update
    }

    fn commit_prefill_physical_resources(
        &mut self,
        kv_cache: Arc<dyn KvCacheHandle>,
        kv_resource_blocks: usize,
        recurrent_state: Option<Arc<dyn RecurrentStateHandle>>,
        recurrent_state_slots: Option<usize>,
    ) -> ModelCacheRefUpdate {
        let allocation = SequenceKvAllocation::new(self.request_id.clone(), kv_resource_blocks);
        let model_cache_update = self.install_legacy_allocated_model_kv(kv_cache, allocation);
        self.recurrent_state =
            recurrent_state.map(|state| SequenceRecurrentState::new(state, recurrent_state_slots));
        self.prefill_complete = true;
        self.phase = RequestPhase::Decoding;
        model_cache_update
    }

    fn commit_plan_runtime_prefill_chunk_resources(
        &mut self,
        kv_cache: Arc<dyn KvCacheHandle>,
        prefill_tokens_processed: usize,
        is_final_chunk: bool,
    ) -> ModelCacheRefUpdate {
        let model_cache_update = self.install_runtime_managed_model_kv(kv_cache);
        self.recurrent_state = None;
        self.prefill_tokens_processed = prefill_tokens_processed;
        self.prefill_complete = is_final_chunk;
        self.phase = if is_final_chunk {
            RequestPhase::Decoding
        } else {
            RequestPhase::Prefilling
        };
        model_cache_update
    }

    fn commit_prefill_chunk_physical_resources(
        &mut self,
        kv_cache: Arc<dyn KvCacheHandle>,
        legacy_kv_allocation: SequenceKvAllocation,
        recurrent_state: Option<Arc<dyn RecurrentStateHandle>>,
        prefill_tokens_processed: usize,
        is_final_chunk: bool,
    ) -> ModelCacheRefUpdate {
        let model_cache_update =
            self.install_legacy_allocated_model_kv(kv_cache, legacy_kv_allocation);
        let existing_slots = self.recurrent_state.as_ref().and_then(|state| state.slots);
        self.recurrent_state =
            recurrent_state.map(|state| SequenceRecurrentState::new(state, existing_slots));
        self.prefill_tokens_processed = prefill_tokens_processed;
        self.prefill_complete = is_final_chunk;
        self.phase = if is_final_chunk {
            RequestPhase::Decoding
        } else {
            RequestPhase::Prefilling
        };
        model_cache_update
    }

    fn decode_model_cache_id_or_request_id(&self, request_id: &RequestId) -> String {
        self.model_cache_id()
            .map(str::to_string)
            .unwrap_or_else(|| request_id.to_string())
    }

    fn decode_model_kv_len_after_last_generated_token(&self) -> usize {
        self.input_tokens
            .len()
            .saturating_add(self.generated_tokens.len())
            .saturating_sub(1)
    }

    fn decode_resources(&self, request_id: &RequestId) -> Option<SequenceDecodeResources> {
        Some(SequenceDecodeResources {
            seq_id: self.decode_model_cache_id_or_request_id(request_id),
            kv_cache: self.model_kv.as_ref()?.handle(),
            recurrent_state: self
                .recurrent_state
                .as_ref()
                .map(SequenceRecurrentState::handle),
            last_token: self
                .generated_tokens
                .last()
                .copied()
                .unwrap_or(TokenId::new(0)),
            pos_offset: self.decode_model_kv_len_after_last_generated_token(),
        })
    }

    fn ready_decode_resources(&self, request_id: &RequestId) -> Option<SequenceDecodeResources> {
        if !self.prefill_complete || self.generated_tokens.is_empty() {
            return None;
        }
        self.decode_resources(request_id)
    }

    fn is_preemptible_decode_candidate(&self) -> bool {
        self.prefill_complete && self.model_kv.is_some()
    }

    fn prefill_resources(&self) -> SequencePrefillResources {
        SequencePrefillResources {
            kv_cache: self.model_kv.as_ref().map(SequenceModelKvState::handle),
            legacy_kv_allocation: self
                .model_kv
                .as_ref()
                .and_then(SequenceModelKvState::legacy_allocation)
                .cloned(),
            recurrent_state: self
                .recurrent_state
                .as_ref()
                .map(SequenceRecurrentState::handle),
            prefill_tokens_processed: self.prefill_tokens_processed,
        }
    }

    fn recurrent_state_handle(&self) -> Option<Arc<dyn RecurrentStateHandle>> {
        self.recurrent_state
            .as_ref()
            .map(SequenceRecurrentState::handle)
    }

    fn recurrent_state_slots(&self) -> Option<usize> {
        self.recurrent_state.as_ref().and_then(|state| state.slots)
    }

    fn draft_kv_cache_handle(&self) -> Option<Arc<dyn KvCacheHandle>> {
        self.draft_kv.as_ref().map(|draft| draft.cache.clone())
    }

    fn kv_cache_handle(&self) -> Option<Arc<dyn KvCacheHandle>> {
        self.model_kv.as_ref().map(SequenceModelKvState::handle)
    }

    fn kv_resource_blocks(&self) -> Option<usize> {
        self.model_kv
            .as_ref()
            .and_then(SequenceModelKvState::legacy_allocation)
            .map(|allocation| allocation.blocks)
    }

    fn model_cache_id(&self) -> Option<&str> {
        self.model_kv
            .as_ref()
            .map(SequenceModelKvState::model_cache_id)
    }

    #[cfg(test)]
    fn clear_model_kv_for_test(&mut self) {
        self.model_kv = None;
    }

    fn commit_decode_step_physical_resources(
        &mut self,
        kv_cache: Arc<dyn KvCacheHandle>,
    ) -> Result<()> {
        self.model_kv
            .as_mut()
            .ok_or_else(|| {
                FerrumError::internal("decode completed without an active model KV lease")
            })?
            .replace_cache_handle(kv_cache)?;
        self.tokens_this_iteration += 1;
        Ok(())
    }

    fn commit_decode_recurrent_state(
        &mut self,
        recurrent_state: Option<Arc<dyn RecurrentStateHandle>>,
    ) {
        let existing_slots = self.recurrent_state.as_ref().and_then(|state| state.slots);
        self.recurrent_state =
            recurrent_state.map(|state| SequenceRecurrentState::new(state, existing_slots));
    }

    fn commit_recurrent_state_admission(
        &mut self,
        recurrent_state: Arc<dyn RecurrentStateHandle>,
        slots: usize,
    ) {
        self.recurrent_state = Some(SequenceRecurrentState::new(recurrent_state, Some(slots)));
    }

    fn take_recurrent_state_allocation(&mut self) -> Option<usize> {
        self.recurrent_state.take().and_then(|state| state.slots)
    }

    fn commit_speculative_decode_physical_resources(
        &mut self,
        target_kv_cache: Arc<dyn KvCacheHandle>,
        draft_kv_cache: Arc<dyn KvCacheHandle>,
    ) -> Result<()> {
        self.model_kv
            .as_ref()
            .ok_or_else(|| {
                FerrumError::internal(
                    "speculative decode completed without an active target KV lease",
                )
            })?
            .validate_replacement_cache(&target_kv_cache)?;
        if let Some(draft) = &self.draft_kv {
            let replacement_cache_id = draft_kv_cache.cache_id();
            if replacement_cache_id != draft.cache.cache_id() {
                return Err(FerrumError::internal(format!(
                    "speculative decode replaced draft cache authority {} with {}",
                    draft.cache.cache_id(),
                    replacement_cache_id
                )));
            }
        } else {
            return Err(FerrumError::internal(
                "draft KV cache updated without owned allocation metadata",
            ));
        }
        self.model_kv
            .as_mut()
            .expect("validated target KV lease remains installed")
            .replace_cache_handle(target_kv_cache)?;
        self.draft_kv
            .as_mut()
            .expect("validated draft KV lease remains installed")
            .cache = draft_kv_cache;
        Ok(())
    }

    fn commit_draft_kv_allocation(
        &mut self,
        draft_kv_cache: Arc<dyn KvCacheHandle>,
        draft_request_id: RequestId,
        draft_resource_blocks: usize,
    ) {
        self.draft_kv = Some(SequenceDraftKvState::new(
            draft_kv_cache,
            draft_request_id,
            draft_resource_blocks,
        ));
    }

    pub fn model_decode_logits_policy(&self) -> LogitsReturnPolicy {
        if !self.can_use_model_greedy_argmax() {
            return LogitsReturnPolicy::FullLogits;
        }
        let token_mask =
            if self.generated_tokens.is_empty() && self.initial_argmax_token_mask.is_some() {
                self.initial_argmax_token_mask.clone()
            } else {
                self.argmax_token_mask.clone()
            };
        LogitsReturnPolicy::GreedyArgmax {
            token_mask,
            repetition_penalty: self.model_decode_repetition_penalty(),
        }
    }

    fn can_use_model_greedy_argmax(&self) -> bool {
        use ferrum_types::ResponseFormat;

        let params = &self.sampling_params;
        params.temperature == 0.0
            && params.top_p == 1.0
            && params.top_k.is_none()
            && params.repetition_penalty > 0.0
            && params.presence_penalty == 0.0
            && params.frequency_penalty == 0.0
            && params.min_p.is_none()
            && params.tfs.is_none()
            && params.typical_p.is_none()
            && params.mirostat.is_none()
            && self.json_processor.is_none()
            && self.regex_processor.is_none()
            && matches!(params.response_format, ResponseFormat::Text)
    }

    fn model_decode_repetition_penalty(&self) -> Option<GreedyRepetitionPenalty> {
        let penalty = self.sampling_params.repetition_penalty;
        if penalty == 1.0 || self.generated_tokens.is_empty() {
            return None;
        }
        let mut seen = HashSet::new();
        let mut token_ids = Vec::new();
        for token in &self.generated_tokens {
            if seen.insert(token.get()) {
                token_ids.push(token.get());
            }
        }
        if token_ids.is_empty() {
            None
        } else {
            Some(GreedyRepetitionPenalty::new(penalty, token_ids))
        }
    }

    pub fn accept_model_greedy_argmax_token(
        &self,
        tokenizer: Option<&(dyn Tokenizer + Send + Sync)>,
        token: TokenId,
    ) -> Result<()> {
        let token_detail = || self.describe_model_greedy_argmax_token(tokenizer, token);
        if !self.can_use_model_greedy_argmax() {
            return Err(FerrumError::model(format!(
                "model returned greedy token sentinel for request requiring full logits ({})",
                token_detail()
            )));
        }

        let token_id = token.get();
        if self.forbidden_token_ids.contains(&token_id) {
            return Err(FerrumError::model(format!(
                "model greedy argmax returned a forbidden token ({})",
                token_detail()
            )));
        }
        if self.generated_tokens.is_empty() && self.initial_forbidden_token_ids.contains(&token_id)
        {
            return Err(FerrumError::model(format!(
                "model greedy argmax returned an initially forbidden token ({})",
                token_detail()
            )));
        }
        if self
            .tokenizer_base_vocab_size
            .is_some_and(|base| token_id as usize >= base)
            && !self.allowed_extended_token_ids.contains(&token_id)
        {
            return Err(FerrumError::model(format!(
                "model greedy argmax returned a disallowed extended-vocab token ({})",
                token_detail()
            )));
        }
        if self.sample_candidate_decodes_to_forbidden_output(
            tokenizer,
            self.streamed_text_len,
            token,
        ) {
            return Err(FerrumError::model(format!(
                "model greedy argmax token decoded to forbidden output ({})",
                token_detail()
            )));
        }

        Ok(())
    }

    fn describe_model_greedy_argmax_token(
        &self,
        tokenizer: Option<&(dyn Tokenizer + Send + Sync)>,
        token: TokenId,
    ) -> String {
        let token_text = tokenizer
            .and_then(|tokenizer| tokenizer.token_text(token))
            .map(|text| format!("{text:?}"))
            .unwrap_or_else(|| "None".to_string());
        let decoded_delta = tokenizer
            .map(|tokenizer| tokenizer.decode_incremental(&self.generated_tokens, token))
            .map(|result| match result {
                Ok(text) => format!("{text:?}"),
                Err(err) => format!("decode_error:{err}"),
            })
            .unwrap_or_else(|| "None".to_string());
        format!(
            "token_id={}, token_text={}, decoded_delta={}, generated_tokens={}, \
             forbidden_count={}, initial_forbidden_count={}, base_vocab_size={:?}, \
             allowed_extended_count={}, argmax_mask={}, initial_argmax_mask={}",
            token.get(),
            token_text,
            decoded_delta,
            self.generated_tokens.len(),
            self.forbidden_token_ids.len(),
            self.initial_forbidden_token_ids.len(),
            self.tokenizer_base_vocab_size,
            self.allowed_extended_token_ids.len(),
            Self::describe_argmax_mask_value(self.argmax_token_mask.as_ref(), token),
            Self::describe_argmax_mask_value(self.initial_argmax_token_mask.as_ref(), token)
        )
    }

    fn describe_argmax_mask_value(mask: Option<&TokenSelectionMask>, token: TokenId) -> String {
        match mask {
            Some(mask) => {
                let value = mask
                    .valid_token_mask
                    .get(token.get() as usize)
                    .copied()
                    .map(|value| value.to_string())
                    .unwrap_or_else(|| "out_of_range".to_string());
                format!(
                    "fingerprint={},len={},value={}",
                    mask.fingerprint,
                    mask.len(),
                    value
                )
            }
            None => "none".to_string(),
        }
    }

    pub fn requires_engine_full_logits_for_sampling(&self) -> bool {
        use ferrum_types::ResponseFormat;

        self.json_processor.is_some()
            || self.regex_processor.is_some()
            || matches!(
                self.sampling_params.response_format,
                ResponseFormat::JsonSchema(_)
            )
    }

    pub fn requires_full_logits_for_sampling(&self) -> bool {
        self.requires_engine_full_logits_for_sampling()
            || self.argmax_token_mask.is_some()
            || (self.generated_tokens.is_empty() && self.initial_argmax_token_mask.is_some())
    }

    pub fn reset_guided_processors(&self) -> Result<()> {
        if let Some(ref jp) = self.json_processor {
            jp.reset();
        }
        if let Some(ref rp) = self.regex_processor {
            rp.reset()?;
        }
        Ok(())
    }

    /// Return the reason this sequence should stop, if any.
    ///
    /// Checks: (1) last generated token is in the resolved `stop_token_ids`
    /// set (model EOS + any single-token `stop_sequences`), (2) decoded text
    /// contains a multi-token user stop sequence, (3) max-tokens budget is
    /// exhausted. Text-stop decoding only runs for requests that supplied a
    /// multi-token stop string, so the common EOS path stays cheap.
    pub fn stop_reason(
        &self,
        tokenizer: Option<&(dyn Tokenizer + Send + Sync)>,
    ) -> Option<FinishReason> {
        if let Some(&last_token) = self.generated_tokens.last() {
            if self.stop_token_ids.contains(&last_token.get()) {
                return Some(FinishReason::Stop);
            }
        }
        if !self.stop_text_seqs.is_empty() {
            if let Some(tok) = tokenizer {
                if let Ok(text) = tok.decode(&self.generated_tokens, true) {
                    if self
                        .stop_text_seqs
                        .iter()
                        .any(|stop| !stop.is_empty() && text.contains(stop))
                    {
                        return Some(FinishReason::Stop);
                    }
                }
            }
        }
        if self.generated_tokens.len() >= self.sampling_params.max_tokens {
            return Some(FinishReason::Length);
        }
        None
    }

    /// Cheap stop check for tests and callers that do not have tokenizer
    /// access. Engine hot paths use `stop_reason` through `EngineInner`.
    pub fn should_stop(&self) -> bool {
        self.stop_reason(None).is_some()
    }

    /// Sample next token with full processor chain (temperature, top-k/p,
    /// repetition penalty, JSON mode, regex-guided mask).
    pub fn sample_with_processors(&mut self, logits: &mut [f32]) -> Result<TokenId> {
        self.sample_with_processors_with_tokenizer(logits, None)
    }

    pub fn sample_with_processors_with_tokenizer(
        &mut self,
        logits: &mut [f32],
        tokenizer: Option<&(dyn Tokenizer + Send + Sync)>,
    ) -> Result<TokenId> {
        use ferrum_interfaces::sampler::{SamplingConfig, SamplingContext};

        // Regex-guided mask runs FIRST: it's a hard constraint (sets invalid
        // tokens to -inf). Subsequent temperature / top-k / top-p stay
        // correct because -inf tokens can't make it through any softmax.
        if let Some(ref rp) = self.regex_processor {
            let best_non_control =
                best_finite_excluding_tokens(logits, &self.allowed_extended_token_ids);
            rp.advance_with_tokens_public(&self.generated_tokens);
            rp.mask_logits(logits);
            mask_non_stop_control_token_logits(
                logits,
                &self.allowed_extended_token_ids,
                &self.stop_token_ids,
            );
            if !rp.can_accept() {
                mask_stop_token_logits(logits, &self.stop_token_ids);
                if !logits.iter().any(|logit| logit.is_finite()) {
                    if let Some(token) = best_non_control {
                        force_only_token(logits, token);
                    }
                }
            }
        }

        // Apply JSON mode biases before the standard processor chain
        if let Some(ref jp) = self.json_processor {
            let generated: String = self
                .generated_tokens
                .iter()
                .filter_map(|t| {
                    let v = t.get();
                    if v < 128 {
                        Some(v as u8 as char)
                    } else {
                        None
                    }
                })
                .collect();
            jp.apply_biases(logits, &generated);
        }

        for &token_id in &self.forbidden_token_ids {
            if let Some(logit) = logits.get_mut(token_id as usize) {
                *logit = f32::NEG_INFINITY;
            }
        }
        if self.generated_tokens.is_empty() {
            for &token_id in &self.initial_forbidden_token_ids {
                if let Some(logit) = logits.get_mut(token_id as usize) {
                    *logit = f32::NEG_INFINITY;
                }
            }
        }
        if let Some(base_vocab_size) = self.tokenizer_base_vocab_size {
            if logits.len() > base_vocab_size {
                for (token_id, logit) in logits.iter_mut().enumerate().skip(base_vocab_size) {
                    if !self.allowed_extended_token_ids.contains(&(token_id as u32)) {
                        *logit = f32::NEG_INFINITY;
                    }
                }
            }
        }

        // Build SamplingConfig from this request's params (includes temperature, top-k/p, repetition penalty)
        let config = SamplingConfig::from_params(&self.sampling_params);
        let step = self.generated_tokens.len();
        let vocab_size = logits.len();
        let previous_streamed_text_len = self.streamed_text_len;
        let token = {
            let mut ctx = SamplingContext::new(
                step,
                &self.sampling_params,
                logits,
                &self.generated_tokens,
                &self.token_frequencies,
                vocab_size,
            );
            config.processor_chain.process(&mut ctx)?;
            let mut attempts = 0usize;
            let mut rejected_tokens = Vec::new();
            loop {
                let token = config.sampler.sample_with_context(&ctx, &mut self.rng)?;
                if !self.sample_candidate_decodes_to_forbidden_output(
                    tokenizer,
                    previous_streamed_text_len,
                    token,
                ) {
                    break token;
                }
                if rejected_tokens.len() < 8 {
                    rejected_tokens.push(token);
                }
                if let Some(logit) = ctx.logits.get_mut(usize::from(token)) {
                    *logit = f32::NEG_INFINITY;
                }
                attempts += 1;
                if attempts >= FORBIDDEN_DECODE_RESAMPLE_LIMIT {
                    self.log_forbidden_decode_resample_failure(
                        tokenizer,
                        previous_streamed_text_len,
                        &rejected_tokens,
                    );
                    return Err(FerrumError::model(
                        "sampling candidates decoded to forbidden output",
                    ));
                }
            }
        };

        // Update frequency tracking
        *self.token_frequencies.entry(token).or_insert(0) += 1;

        Ok(token)
    }

    fn sample_candidate_decodes_to_forbidden_output(
        &self,
        tokenizer: Option<&(dyn Tokenizer + Send + Sync)>,
        previous_streamed_text_len: usize,
        token: TokenId,
    ) -> bool {
        let Some(tokenizer) = tokenizer else {
            return false;
        };
        let mut tokens = Vec::with_capacity(self.generated_tokens.len() + 1);
        tokens.extend_from_slice(&self.generated_tokens);
        tokens.push(token);
        let candidate_is_stop = self.stop_token_ids.contains(&token.get());
        let candidate_is_non_stop_control =
            self.allowed_extended_token_ids.contains(&token.get()) && !candidate_is_stop;
        tokenizer
            .decode(&tokens, true)
            .map(|text| {
                decoded_delta_has_forbidden_quality(
                    &text,
                    previous_streamed_text_len,
                    candidate_is_stop,
                    candidate_is_non_stop_control,
                )
            })
            .unwrap_or(true)
    }

    fn log_forbidden_decode_resample_failure(
        &self,
        tokenizer: Option<&(dyn Tokenizer + Send + Sync)>,
        previous_streamed_text_len: usize,
        rejected_tokens: &[TokenId],
    ) {
        let generated_tail: Vec<String> = self
            .generated_tokens
            .iter()
            .rev()
            .take(8)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .map(|token| describe_token_for_log(tokenizer, *token))
            .collect();
        let rejected: Vec<String> = rejected_tokens
            .iter()
            .map(|token| describe_token_for_log(tokenizer, *token))
            .collect();
        warn!(
            request_id = %self.request_id,
            generated_len = self.generated_tokens.len(),
            previous_streamed_text_len,
            generated_tail = ?generated_tail,
            rejected_candidates = ?rejected,
            "sampling candidates decoded to forbidden output"
        );
    }
}

impl Drop for SequenceState {
    fn drop(&mut self) {
        if self.request_slot.is_some() {
            let message = "sequence state dropped with owned request slot";
            warn!(
                request_id = %self.request_id,
                has_kv_cache = self.model_kv.is_some(),
                kv_resource_blocks = ?self.kv_resource_blocks(),
                has_recurrent_state = self.recurrent_state.is_some(),
                recurrent_state_slots = ?self.recurrent_state_slots(),
                has_draft_kv = self.draft_kv.is_some(),
                draft_kv_resource_blocks = ?self.draft_kv.as_ref().map(|draft| draft.resource_blocks),
                "{message}"
            );
            #[cfg(test)]
            if !std::thread::panicking() {
                panic!("{message}");
            }
        }
    }
}

fn describe_token_for_log(
    tokenizer: Option<&(dyn Tokenizer + Send + Sync)>,
    token: TokenId,
) -> String {
    let Some(tokenizer) = tokenizer else {
        return token.get().to_string();
    };
    let raw = tokenizer.token_text(token).unwrap_or("<missing>");
    let decoded = tokenizer
        .decode(&[token], true)
        .unwrap_or_else(|_| "<decode-error>".to_string());
    format!("{} raw={:?} decoded={:?}", token.get(), raw, decoded)
}

fn mask_stop_token_logits(logits: &mut [f32], stop_token_ids: &HashSet<u32>) {
    for &token_id in stop_token_ids {
        if let Some(logit) = logits.get_mut(token_id as usize) {
            *logit = f32::NEG_INFINITY;
        }
    }
}

fn mask_non_stop_control_token_logits(
    logits: &mut [f32],
    control_token_ids: &HashSet<u32>,
    stop_token_ids: &HashSet<u32>,
) {
    for &token_id in control_token_ids {
        if stop_token_ids.contains(&token_id) {
            continue;
        }
        if let Some(logit) = logits.get_mut(token_id as usize) {
            *logit = f32::NEG_INFINITY;
        }
    }
}

fn best_finite_excluding_tokens(
    logits: &[f32],
    excluded_token_ids: &HashSet<u32>,
) -> Option<usize> {
    logits
        .iter()
        .enumerate()
        .filter(|(idx, logit)| logit.is_finite() && !excluded_token_ids.contains(&(*idx as u32)))
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(idx, _)| idx)
}

fn force_only_token(logits: &mut [f32], token: usize) {
    for (idx, logit) in logits.iter_mut().enumerate() {
        *logit = if idx == token { 0.0 } else { f32::NEG_INFINITY };
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Engine inner – shared via Arc so we can spawn tasks
// ────────────────────────────────────────────────────────────────────────────

#[derive(Debug)]
#[must_use = "request slot leases must be consumed by reject() or close()"]
struct RequestSlotLease {
    request_id: RequestId,
    admitted: bool,
    armed: bool,
}

impl RequestSlotLease {
    fn open(engine: &EngineInner, request_id: RequestId) -> Self {
        engine.trace_request_open(&request_id);
        Self {
            request_id,
            admitted: false,
            armed: true,
        }
    }

    fn admit(&mut self, engine: &EngineInner) {
        if !self.admitted {
            engine.trace_request_admitted(&self.request_id);
            self.admitted = true;
        }
    }

    fn reject(mut self, engine: &EngineInner, reason: String) {
        engine.trace_request_rejected(&self.request_id, reason);
        self.armed = false;
    }

    fn close(mut self, engine: &EngineInner) {
        if self.admitted {
            engine.trace_request_close(&self.request_id);
        } else {
            engine.trace_request_owner_close(&self.request_id);
        }
        self.armed = false;
    }
}

impl Drop for RequestSlotLease {
    fn drop(&mut self) {
        if self.armed {
            let message = "request slot lease dropped without explicit reject or close";
            warn!(
                request_id = %self.request_id,
                admitted = self.admitted,
                "{message}"
            );
            #[cfg(test)]
            if !std::thread::panicking() {
                panic!("{message}");
            }
        }
    }
}

#[must_use = "KV allocation leases must be consumed by release().await or into_committed_parts()"]
struct KvAllocationLease {
    owner_request_id: RequestId,
    allocation_request_id: RequestId,
    handle: Arc<dyn KvCacheHandle>,
    blocks: usize,
    armed: bool,
}

impl KvAllocationLease {
    fn new(
        owner_request_id: RequestId,
        allocation_request_id: RequestId,
        handle: Arc<dyn KvCacheHandle>,
        blocks: usize,
    ) -> Self {
        Self {
            owner_request_id,
            allocation_request_id,
            handle,
            blocks,
            armed: true,
        }
    }

    fn handle(&self) -> Arc<dyn KvCacheHandle> {
        self.handle.clone()
    }

    fn blocks(&self) -> usize {
        self.blocks
    }

    async fn release(mut self, engine: &EngineInner) {
        engine
            .release_kv_allocation(
                &self.owner_request_id,
                self.allocation_request_id.clone(),
                self.blocks,
            )
            .await;
        self.armed = false;
    }

    fn into_committed_parts(mut self) -> (RequestId, usize) {
        self.armed = false;
        (self.allocation_request_id.clone(), self.blocks)
    }
}

impl Drop for KvAllocationLease {
    fn drop(&mut self) {
        if self.armed {
            let message = "KV allocation lease dropped without explicit commit or async release";
            warn!(
                owner_request_id = %self.owner_request_id,
                allocation_request_id = %self.allocation_request_id,
                blocks = self.blocks,
                "{message}"
            );
            #[cfg(test)]
            if !std::thread::panicking() {
                panic!("{message}");
            }
        }
    }
}

#[must_use = "recurrent-state leases must be consumed by release().await or commit()"]
struct RecurrentStateLease {
    request_id: RequestId,
    handle: Arc<dyn RecurrentStateHandle>,
    slots: usize,
    capacity: Option<usize>,
    armed: bool,
}

impl RecurrentStateLease {
    fn new(
        request_id: RequestId,
        handle: Arc<dyn RecurrentStateHandle>,
        slots: usize,
        capacity: Option<usize>,
    ) -> Self {
        Self {
            request_id,
            handle,
            slots,
            capacity,
            armed: true,
        }
    }

    fn handle(&self) -> Arc<dyn RecurrentStateHandle> {
        self.handle.clone()
    }

    fn slots(&self) -> usize {
        self.slots
    }

    async fn release(mut self, engine: &EngineInner) {
        engine
            .release_recurrent_allocation(&self.request_id, Some(self.slots))
            .await;
        self.armed = false;
    }

    fn commit(mut self) -> usize {
        self.armed = false;
        self.slots
    }
}

impl Drop for RecurrentStateLease {
    fn drop(&mut self) {
        if self.armed {
            let message = "recurrent-state lease dropped without explicit commit or async release";
            warn!(
                request_id = %self.request_id,
                slots = self.slots,
                capacity = ?self.capacity,
                "{message}"
            );
            #[cfg(test)]
            if !std::thread::panicking() {
                panic!("{message}");
            }
        }
    }
}

struct RecurrentStateAdmission {
    handle: Option<Arc<dyn RecurrentStateHandle>>,
    lease: Option<RecurrentStateLease>,
}

impl RecurrentStateAdmission {
    fn none() -> Self {
        Self {
            handle: None,
            lease: None,
        }
    }

    fn existing(handle: Arc<dyn RecurrentStateHandle>) -> Self {
        Self {
            handle: Some(handle),
            lease: None,
        }
    }

    fn fresh(lease: RecurrentStateLease) -> Self {
        Self {
            handle: Some(lease.handle()),
            lease: Some(lease),
        }
    }

    fn handle(&self) -> Option<Arc<dyn RecurrentStateHandle>> {
        self.handle.clone()
    }

    fn fresh_slots(&self) -> Option<usize> {
        self.lease.as_ref().map(RecurrentStateLease::slots)
    }

    fn commit_fresh(&mut self) -> Option<usize> {
        self.lease.take().map(RecurrentStateLease::commit)
    }

    async fn release_fresh(&mut self, engine: &EngineInner) {
        if let Some(lease) = self.lease.take() {
            lease.release(engine).await;
        }
    }
}

#[must_use = "backend workspace leases must be released or dropped to close the trace lifecycle"]
struct BackendWorkspaceLease<'a> {
    engine: &'a EngineInner,
    request_ids: Vec<RequestId>,
    release_phase: &'static str,
    armed: bool,
}

impl<'a> BackendWorkspaceLease<'a> {
    fn new(
        engine: &'a EngineInner,
        request_ids: Vec<RequestId>,
        phase_prefix: &'static str,
        release_phase: &'static str,
    ) -> Self {
        engine.trace_backend_workspace_acquire_many(&request_ids, phase_prefix);
        Self {
            engine,
            request_ids,
            release_phase,
            armed: true,
        }
    }

    fn release(mut self) {
        self.release_now();
        self.armed = false;
    }

    fn release_now(&self) {
        self.engine
            .trace_backend_workspace_release_many(&self.request_ids, self.release_phase);
    }
}

impl Drop for BackendWorkspaceLease<'_> {
    fn drop(&mut self) {
        if self.armed {
            self.release_now();
        }
    }
}

struct PendingBatchPrefill {
    request_id: RequestId,
    input_tokens: Vec<TokenId>,
    kv_lease: Option<KvAllocationLease>,
    recurrent_state: RecurrentStateAdmission,
    metadata: HashMap<String, serde_json::Value>,
    can_use_prefix_cache: bool,
}

impl PendingBatchPrefill {
    fn new(
        request_id: RequestId,
        input_tokens: Vec<TokenId>,
        kv_lease: KvAllocationLease,
        recurrent_state: RecurrentStateAdmission,
        metadata: HashMap<String, serde_json::Value>,
        can_use_prefix_cache: bool,
    ) -> Self {
        Self {
            request_id,
            input_tokens,
            kv_lease: Some(kv_lease),
            recurrent_state,
            metadata,
            can_use_prefix_cache,
        }
    }

    fn kv_handle(&self) -> Result<Arc<dyn KvCacheHandle>> {
        self.kv_lease
            .as_ref()
            .map(KvAllocationLease::handle)
            .ok_or_else(|| FerrumError::internal("batch prefill KV lease already consumed"))
    }

    fn kv_resource_blocks(&self) -> Result<usize> {
        self.kv_lease
            .as_ref()
            .map(KvAllocationLease::blocks)
            .ok_or_else(|| FerrumError::internal("batch prefill KV lease already consumed"))
    }

    fn commit_kv(&mut self) -> Result<usize> {
        let lease = self
            .kv_lease
            .take()
            .ok_or_else(|| FerrumError::internal("batch prefill KV lease already consumed"))?;
        let (_allocation_request_id, blocks) = lease.into_committed_parts();
        Ok(blocks)
    }

    async fn release_resources(&mut self, engine: &EngineInner) {
        if let Some(lease) = self.kv_lease.take() {
            lease.release(engine).await;
        }
        self.recurrent_state.release_fresh(engine).await;
    }
}

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
    /// Ensures only one iteration step runs at a time.
    iteration_lock: tokio::sync::Mutex<()>,
    /// Wakes callers or a background loop when new work is submitted.
    work_notify: Notify,
    /// Prefix cache: shares KV blocks across requests with common prompts.
    prefix_cache: PrefixCache,
    runtime_config: ContinuousEngineRuntimeConfig,
    scheduler_trace_jsonl: Option<Arc<Mutex<std::fs::File>>>,
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
            ("diagnostic_only".to_string(), serde_json::json!(false)),
            ("l0_only".to_string(), serde_json::json!(false)),
            ("profile_detail".to_string(), serde_json::json!("basic")),
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
        if matches!(action, ResourceAction::Defer) {
            attributes.insert(
                "scheduler_snapshot".to_string(),
                serde_json::to_value(self.scheduler.trace_snapshot())
                    .unwrap_or(serde_json::Value::Null),
            );
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
        let mut line = match serde_json::to_string(&event) {
            Ok(line) => line,
            Err(error) => {
                warn!("Failed to serialize engine resource trace event: {}", error);
                return;
            }
        };
        line.push('\n');
        let mut file = sink.lock();
        if let Err(error) = file.write_all(line.as_bytes()) {
            warn!("Failed to write engine resource trace event: {}", error);
            return;
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
            ("diagnostic_only".to_string(), serde_json::json!(false)),
            ("l0_only".to_string(), serde_json::json!(false)),
            ("profile_detail".to_string(), serde_json::json!("basic")),
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
        let mut line = match serde_json::to_string(&event) {
            Ok(line) => line,
            Err(error) => {
                warn!(
                    "Failed to serialize engine resource close trace event: {}",
                    error
                );
                return;
            }
        };
        line.push('\n');
        let mut file = sink.lock();
        if let Err(error) = file.write_all(line.as_bytes()) {
            warn!(
                "Failed to write engine resource close trace event: {}",
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

fn vnext_execution_event_name(kind: VNextExecutionEventKind) -> &'static str {
    match kind {
        VNextExecutionEventKind::RequestAccepted => "request_accepted",
        VNextExecutionEventKind::PlanBuilt => "plan_built",
        VNextExecutionEventKind::FrameStarted => "frame_started",
        VNextExecutionEventKind::NodeStarted => "node_started",
        VNextExecutionEventKind::OperationSubmitted => "operation_submitted",
        VNextExecutionEventKind::NodeRetired => "node_retired",
        VNextExecutionEventKind::FrameCompleted => "frame_completed",
        VNextExecutionEventKind::FailureObserved => "failure_observed",
        VNextExecutionEventKind::SequenceCompleted => "sequence_completed",
        VNextExecutionEventKind::SequenceAborted => "sequence_aborted",
        VNextExecutionEventKind::RequestCompleted => "request_completed",
        VNextExecutionEventKind::RequestFailed => "request_failed",
    }
}

struct VNextProfileExecutionEventSink {
    file: Arc<Mutex<std::fs::File>>,
    entrypoint: ProfileEntrypoint,
    model: String,
    backend_device: String,
    backend_type: String,
}

impl VNextProfileExecutionEventSink {
    fn new(
        file: Arc<Mutex<std::fs::File>>,
        entrypoint: ProfileEntrypoint,
        config: &EngineConfig,
    ) -> Self {
        Self {
            file,
            entrypoint,
            model: config.model.model_id.to_string(),
            backend_device: format!("{:?}", config.backend.device),
            backend_type: format!("{:?}", config.backend.backend_type),
        }
    }
}

impl ExecutionEventSink for VNextProfileExecutionEventSink {
    fn is_enabled(&self, _kind: VNextExecutionEventKind) -> bool {
        true
    }

    fn capture_policy(&self) -> ExecutionEventCapturePolicy {
        ExecutionEventCapturePolicy::FirstFramePerRequest
    }

    fn record(
        &self,
        event: &ExecutionEvent,
        _permit: EventEmissionPermit<'_>,
    ) -> std::result::Result<(), ExecutionEventSinkError> {
        let identity = event.identity().parts();
        let event_name = vnext_execution_event_name(event.kind());
        let timestamp = chrono::Utc::now();
        let failure = match event.detail() {
            ExecutionEventDetail::Failure(failure) => Some(ProfileError {
                kind: failure.failure().code().to_string(),
                message: failure.failure().message().to_string(),
                blocking: true,
            }),
            ExecutionEventDetail::FailureTerminal {
                first_failure_fingerprint,
            } => Some(ProfileError {
                kind: "vnext_request_failed".to_string(),
                message: format!("request terminated after failure {first_failure_fingerprint}"),
                blocking: true,
            }),
            _ => None,
        };
        let status = if failure.is_some() {
            ProfileStatus::Failure
        } else {
            ProfileStatus::Ok
        };
        let mut shape = BTreeMap::from([(
            "execution_sequence".to_string(),
            serde_json::json!(identity.sequence),
        )]);
        if let Some(frame_id) = identity.frame_id {
            shape.insert("frame_id".to_string(), serde_json::json!(frame_id.get()));
        }
        if let Some(invocation_id) = identity.node_invocation_id {
            shape.insert(
                "node_invocation_id".to_string(),
                serde_json::json!(invocation_id.get()),
            );
        }
        if let ExecutionEventDetail::Counters { input, output } = event.detail() {
            shape.insert("event_input_count".to_string(), serde_json::json!(input));
            shape.insert("event_output_count".to_string(), serde_json::json!(output));
        }
        let mut attributes = BTreeMap::from([
            (
                "actual_model_smoke".to_string(),
                serde_json::json!(matches!(
                    self.entrypoint,
                    ProfileEntrypoint::Run | ProfileEntrypoint::Serve
                )),
            ),
            (
                "backend_device".to_string(),
                serde_json::json!(self.backend_device),
            ),
            (
                "backend_type".to_string(),
                serde_json::json!(self.backend_type),
            ),
            ("diagnostic_only".to_string(), serde_json::json!(false)),
            ("l0_only".to_string(), serde_json::json!(false)),
            ("profile_detail".to_string(), serde_json::json!("basic")),
            (
                "execution_capture_policy".to_string(),
                serde_json::json!(self.capture_policy().as_str()),
            ),
            (
                "execution_event_kind".to_string(),
                serde_json::json!(event_name),
            ),
            (
                "execution_phase".to_string(),
                serde_json::json!(format!("{:?}", event.phase()).to_ascii_lowercase()),
            ),
            (
                "execution_trace_source".to_string(),
                serde_json::json!("vnext"),
            ),
            (
                "monotonic_nanos_since_run_start".to_string(),
                serde_json::json!(event.timestamp().nanos_since_run_start),
            ),
            (
                "run_id".to_string(),
                serde_json::json!(identity.run_id.to_string()),
            ),
            (
                "span_id".to_string(),
                serde_json::json!(identity.span_id.to_string()),
            ),
        ]);
        for (key, value) in [
            (
                "plan_id",
                identity.plan_id.as_ref().map(ToString::to_string),
            ),
            (
                "plan_hash",
                identity.plan_hash.as_ref().map(ToString::to_string),
            ),
            (
                "node_id",
                identity.node_id.as_ref().map(ToString::to_string),
            ),
            (
                "operation_id",
                identity.operation_id.as_ref().map(ToString::to_string),
            ),
            (
                "provider_id",
                identity.provider_id.as_ref().map(ToString::to_string),
            ),
            (
                "device_id",
                identity.device_id.as_ref().map(ToString::to_string),
            ),
            (
                "parent_span_id",
                identity.parent_span_id.as_ref().map(ToString::to_string),
            ),
        ] {
            if let Some(value) = value {
                attributes.insert(key.to_string(), serde_json::json!(value));
            }
        }
        let profile = FerrumProfileEvent {
            schema_version: OBSERVABILITY_PROFILE_SCHEMA_VERSION,
            ts_unix_nanos: timestamp
                .timestamp_nanos_opt()
                .unwrap_or_else(|| timestamp.timestamp_micros() * 1_000),
            event_id: format!(
                "evt-vnext-{}-{}-{}",
                identity.run_id, identity.request_id, identity.sequence
            ),
            request_id: identity.request_id.to_string(),
            correlation_id: Some(identity.request_id.to_string()),
            entrypoint: self.entrypoint,
            backend: "actual".to_string(),
            runtime_preset_hash: ENGINE_RUNTIME_TRACE_PRESET_HASH.to_string(),
            phase: format!("vnext.{event_name}"),
            event_kind: if failure.is_some() {
                ProfileEventKind::Error
            } else {
                ProfileEventKind::Instant
            },
            timestamp,
            status,
            model: Some(self.model.clone()),
            duration_us: None,
            memory: None,
            resource: None,
            error: failure,
            replay: None,
            shape,
            backend_detail: Some(BTreeMap::from([
                (
                    "backend_device".to_string(),
                    serde_json::json!(self.backend_device),
                ),
                (
                    "backend_type".to_string(),
                    serde_json::json!(self.backend_type),
                ),
            ])),
            attributes,
        };
        profile.validate().map_err(|error| {
            ExecutionEventSinkError::new(format!("invalid vNext profile event: {error}"))
        })?;
        let mut line = serde_json::to_string(&profile).map_err(|error| {
            ExecutionEventSinkError::new(format!("serialize vNext profile event: {error}"))
        })?;
        line.push('\n');
        self.file
            .lock()
            .write_all(line.as_bytes())
            .map_err(|error| {
                ExecutionEventSinkError::new(format!("write vNext profile event: {error}"))
            })
    }
}

fn create_scheduler_trace_sink(path: Option<&Path>) -> Option<Arc<Mutex<std::fs::File>>> {
    let path = path?;
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            if let Err(error) = std::fs::create_dir_all(parent) {
                warn!(
                    "Failed to create scheduler trace directory {}: {}",
                    parent.display(),
                    error
                );
                return None;
            }
        }
    }
    if let Err(error) = std::fs::remove_file(path) {
        if error.kind() != std::io::ErrorKind::NotFound {
            warn!(
                "Failed to clear scheduler trace JSONL {}: {}",
                path.display(),
                error
            );
            return None;
        }
    }
    match std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
    {
        Ok(file) => Some(Arc::new(Mutex::new(file))),
        Err(error) => {
            warn!(
                "Failed to open scheduler trace JSONL {}: {}",
                path.display(),
                error
            );
            None
        }
    }
}

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
        let scheduler_trace_jsonl =
            create_scheduler_trace_sink(runtime_config.scheduler_trace_jsonl.as_deref());
        let legacy_scheduler_trace_jsonl =
            create_scheduler_trace_sink(runtime_config.legacy_scheduler_trace_jsonl.as_deref());
        if let Some(file) = scheduler_trace_jsonl.as_ref() {
            let sink: Arc<dyn ExecutionEventSink> = Arc::new(VNextProfileExecutionEventSink::new(
                Arc::clone(file),
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
                work_notify: Notify::new(),
                iteration_count: AtomicU64::new(0),
                prefix_cache: PrefixCache::new(256, 2),
                runtime_config,
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
        if !self.inner.bg_loop_spawned.swap(true, Ordering::SeqCst) {
            let _ = self.start_loop();
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
                let outcome = {
                    let lock_wait_start = Instant::now();
                    let _guard = inner.iteration_lock.lock().await;
                    inner.record_iteration_lock_wait(lock_wait_start.elapsed());
                    match inner.run_iteration().await {
                        Ok(outcome) => outcome,
                        Err(error) => {
                            warn!("Iteration error: {}", error);
                            EngineIterationOutcome::Progressed
                        }
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
        gauge!("ferrum.engine.active_requests").increment(1.0);

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
        let request_slot = RequestSlotLease::open(&self.inner, request_id.clone());
        let mut seq_state = SequenceState::new_with_tokenizer_and_model_vocab_size(
            request.clone(),
            input_tokens,
            Some(self.inner.tokenizer.clone()),
            Some(self.inner.model_executor.info().vocab_size),
        );
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

        let result = resp_rx
            .await
            .map_err(|_| FerrumError::internal("Response channel closed before response was sent"));

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
        let request_slot = RequestSlotLease::open(&self.inner, request_id.clone());
        let mut seq_state = SequenceState::new_with_tokenizer_and_model_vocab_size(
            request.clone(),
            input_tokens,
            Some(self.inner.tokenizer.clone()),
            Some(self.inner.model_executor.info().vocab_size),
        );
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

        Ok(Box::pin(tokio_stream::wrappers::ReceiverStream::new(rx)))
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
        info!("Shutting down continuous batch engine");
        self.inner.is_running.store(false, Ordering::SeqCst);
        self.inner.shutdown_notify.notify_one();
        self.inner.work_notify.notify_one();
        Ok(())
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
