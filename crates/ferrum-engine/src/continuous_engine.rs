//! Continuous Batching Engine
//!
//! Iteration-level continuous batching: each step processes a mixed batch of
//! prefill and decode requests selected by the scheduler.  Multiple callers
//! can submit requests concurrently — an `iteration_lock` serializes the
//! actual engine steps so each batch is processed exactly once.

use async_trait::async_trait;
use ferrum_bench_core::{global_profile, profile_fields_from_json};
use ferrum_interfaces::{
    engine::{InferenceEngine, LlmInferenceEngine},
    kv_cache::AllocationRequest,
    model_executor::{
        GreedyRepetitionPenalty, KvSlotRequest, LogitsReturnPolicy, TokenSelectionMask,
    },
    KvCacheHandle, KvCacheManager, ModelExecutor, RecurrentStateHandle, RecurrentStateManager,
    Sampler, SchedulerInterface as Scheduler, TensorFactory, TensorRef, Tokenizer,
};
use ferrum_kv::cache::prefix::PrefixCache;
use ferrum_sampler::json_mode::JsonModeProcessor;
use ferrum_scheduler::implementations::{ContinuousBatchScheduler, RequestPhase};
use ferrum_types::{
    DataType, Device, EngineConfig, EngineStatus, FerrumError, FinishReason, InferenceRequest,
    InferenceResponse, Priority, RequestId, Result, SamplingParams, StreamChunk, TokenId,
    TokenUsage, DEFAULT_MAX_TOKENS_METADATA_KEY, PROMPT_TOKENS_METADATA_KEY,
};
use futures::stream::Stream;
use metrics::{counter, gauge, histogram};
use parking_lot::RwLock;
use rand::{rngs::StdRng, SeedableRng};
use std::collections::{HashMap, HashSet};
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
    prefix_cache_enabled: bool,
    rbd_prof: bool,
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
            prefix_cache_enabled: r.prefix_cache_enabled,
            rbd_prof: r.rbd_prof,
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
            prefix_cache_enabled: vars
                .get(WHOLE_PROMPT_PREFIX_CACHE_ENV)
                .is_some_and(|v| v == "1"),
            rbd_prof: vars.contains_key(RBD_PROF_ENV),
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
    forbidden_token_ids: &HashSet<u32>,
    initial_forbidden_token_ids: &HashSet<u32>,
    stop_token_ids: &HashSet<u32>,
    allowed_extended_token_ids: &HashSet<u32>,
) -> TokenSelectionMask {
    let mut valid = vec![1i8; tok.vocab_size()];
    for &token_id in forbidden_token_ids
        .iter()
        .chain(initial_forbidden_token_ids.iter())
    {
        if let Some(slot) = valid.get_mut(token_id as usize) {
            *slot = 0;
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
    pub kv_cache: Option<Arc<dyn KvCacheHandle>>,
    pub recurrent_state: Option<Arc<dyn RecurrentStateHandle>>,
    pub sampling_params: SamplingParams,
    pub phase: RequestPhase,
    pub rng: StdRng,
    pub prefill_complete: bool,
    /// Number of prompt tokens already written into the model KV cache by
    /// opt-in unified chunked prefill. Zero for the normal full-prefill path.
    pub prefill_tokens_processed: usize,
    pub stream_sender: Option<mpsc::Sender<Result<StreamChunk>>>,
    pub response_sender: Option<tokio::sync::oneshot::Sender<InferenceResponse>>,
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
    /// Draft-model KV cache (only populated when engine has speculative
    /// decoding enabled). Allocated + prefilled lazily on the first decode.
    pub draft_kv_cache: Option<Arc<dyn KvCacheHandle>>,
    /// Token frequency counts for repetition penalty.
    pub token_frequencies: HashMap<TokenId, usize>,
    /// Model executor's KV cache key for this sequence (for cleanup on completion).
    pub model_cache_id: Option<String>,
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
            kv_cache: None,
            recurrent_state: None,
            sampling_params: request.sampling_params,
            phase: RequestPhase::Waiting,
            rng,
            prefill_complete: false,
            prefill_tokens_processed: 0,
            stream_sender: None,
            response_sender: None,
            start_time: Instant::now(),
            first_emit_at: None,
            last_emit_at: None,
            emitted_chunks: 0,
            tokens_this_iteration: 0,
            preemption_count: 0,
            json_processor,
            regex_processor,
            draft_kv_cache: None,
            token_frequencies: HashMap::new(),
            model_cache_id: None,
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
        metadata
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

struct EngineInner {
    config: EngineConfig,
    scheduler: Arc<ContinuousBatchScheduler>,
    tokenizer: Arc<dyn Tokenizer + Send + Sync>,
    #[allow(dead_code)]
    // Retained for constructor API; sampling now uses per-request SamplingConfig
    sampler: Arc<dyn Sampler + Send + Sync>,
    kv_cache: Arc<dyn KvCacheManager + Send + Sync>,
    recurrent_state_manager: Option<Arc<dyn RecurrentStateManager + Send + Sync>>,
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

    async fn ensure_recurrent_state(
        &self,
        request_id: &RequestId,
        spec: Option<ferrum_interfaces::RecurrentStateSpec>,
    ) -> Result<Option<Arc<dyn RecurrentStateHandle>>> {
        if let Some(existing) = self
            .sequences
            .read()
            .get(request_id)
            .and_then(|seq| seq.recurrent_state.clone())
        {
            return Ok(Some(existing));
        }

        let Some(spec) = spec else {
            return Ok(None);
        };

        debug_assert_eq!(&spec.request_id, request_id);
        let Some(manager) = &self.recurrent_state_manager else {
            return Err(FerrumError::config(format!(
                "model '{}' requires recurrent state for request {}, but no recurrent-state manager is configured",
                self.model_executor.info().model_id, request_id
            )));
        };

        let handle = manager.allocate(&spec).await?;

        if let Some(seq) = self.sequences.write().get_mut(request_id) {
            seq.recurrent_state = Some(handle.clone());
        }

        Ok(Some(handle))
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
    ) -> Self {
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
    ) -> Self {
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
    ) -> Self {
        info!(
            "Creating ContinuousBatchEngine (speculative_decoding={}, recurrent_state_manager={})",
            draft_executor.is_some() && spec_config.is_some(),
            recurrent_state_manager.is_some()
        );
        let runtime_config = ContinuousEngineRuntimeConfig::from_engine_config(&config);

        Self {
            inner: Arc::new(EngineInner {
                config,
                scheduler,
                tokenizer,
                sampler,
                kv_cache,
                recurrent_state_manager,
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
        }
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
                {
                    let lock_wait_start = Instant::now();
                    let _guard = inner.iteration_lock.lock().await;
                    inner.record_iteration_lock_wait(lock_wait_start.elapsed());
                    if let Err(e) = inner.run_iteration().await {
                        warn!("Iteration error: {}", e);
                    }
                }
                if prof {
                    let n = GAP_PROF_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    if n.is_multiple_of(8) {
                        if let Some(gap_us) = inter_iter_us {
                            eprintln!("[bg-loop-gap] call#{} inter_iter={}us", n, gap_us);
                        }
                    }
                }
                last_iter_end = Some(std::time::Instant::now());
                tokio::task::yield_now().await;
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

        // Submit to scheduler
        self.inner.scheduler.submit(request.clone()).await?;

        // Create sequence state with oneshot channel
        let (resp_tx, resp_rx) = tokio::sync::oneshot::channel();
        let mut seq_state = SequenceState::new_with_tokenizer(
            request,
            input_tokens,
            Some(self.inner.tokenizer.clone()),
        );
        seq_state.response_sender = Some(resp_tx);
        self.inner
            .sequences
            .write()
            .insert(request_id.clone(), seq_state);

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

        // Submit to scheduler
        self.inner.scheduler.submit(request.clone()).await?;

        // Create sequence state with stream sender
        let mut seq_state = SequenceState::new_with_tokenizer(
            request,
            input_tokens,
            Some(self.inner.tokenizer.clone()),
        );
        seq_state.stream_sender = Some(tx);
        self.inner
            .sequences
            .write()
            .insert(request_id.clone(), seq_state);

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
        EngineStatus {
            is_ready: self.inner.is_running.load(Ordering::SeqCst),
            loaded_models: vec![self.inner.config.model.model_id.clone()],
            active_requests: metrics.running_requests,
            queued_requests: metrics.waiting_requests,
            memory_usage: ferrum_types::MemoryUsage {
                total_bytes: 0,
                used_bytes: 0,
                free_bytes: 0,
                gpu_memory_bytes: None,
                cpu_memory_bytes: None,
                cache_memory_bytes: 0,
                utilization_percent: 0.0,
            },
            uptime_seconds: 0,
            last_heartbeat: chrono::Utc::now(),
            version: env!("CARGO_PKG_VERSION").to_string(),
        }
    }

    async fn shutdown(&self) -> Result<()> {
        info!("Shutting down continuous batch engine");
        self.inner.is_running.store(false, Ordering::SeqCst);
        self.inner.shutdown_notify.notify_waiters();
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
