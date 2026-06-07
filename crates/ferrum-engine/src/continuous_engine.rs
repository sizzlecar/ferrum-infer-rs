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
    KvCacheHandle, KvCacheManager, ModelExecutor, Sampler, SchedulerInterface as Scheduler,
    TensorFactory, TensorRef, Tokenizer,
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

const BATCH_DECODE_PROF_ENV: &str = "FERRUM_BATCH_DECODE_PROF";
const CHUNKED_PREFILL_ENV: &str = "FERRUM_CHUNKED_PREFILL";
const KV_CAPACITY_ENV: &str = "FERRUM_KV_CAPACITY";
const MAX_MODEL_LEN_ENV: &str = "FERRUM_MAX_MODEL_LEN";
const NEXT_BATCH_PROF_ENV: &str = "FERRUM_NEXT_BATCH_PROF";
const WHOLE_PROMPT_PREFIX_CACHE_ENV: &str = "FERRUM_WHOLE_PROMPT_PREFIX_CACHE";
const RBD_PROF_ENV: &str = "FERRUM_RBD_PROF";
const UNIFIED_POST_PROF_ENV: &str = "FERRUM_UNIFIED_POST_PROF";
const GENERATION_POLICY_SCAN_LIMIT: usize = 262_144;
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
    fn from_engine_config_and_env(config: &EngineConfig) -> Self {
        Self::from_env_vars(
            config.scheduler.active_decode_prefill_chunk,
            std::env::vars(),
        )
    }

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
///    one-token results land in `stop_token_ids`, multi-token results go to
///    `stop_text_seqs` for text-match fallback.
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
            for name in ["<|im_end|>", "<|endoftext|>", "<|eot_id|>", "</s>"] {
                if let Some(t) = tok.token_id(name) {
                    ids.insert(t.get());
                }
            }
        }
        for stop_seq in &params.stop_sequences {
            match tok.encode(stop_seq, false) {
                Ok(toks) if toks.len() == 1 => {
                    ids.insert(toks[0].get());
                }
                _ => text_seqs.push(stop_seq.clone()),
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
    for text in GENERATED_CONTROL_TOKEN_TEXTS {
        if let Some(token) = tok.token_id(text) {
            allowed_extended.insert(token.get());
        }
    }

    let forbidden = cached_forbidden_generation_tokens(tok, &allowed_extended);

    (forbidden, Some(tok.vocab_size()), allowed_extended)
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
        "<unk", "<unk>", "[UNK]", "<pad>", "[PAD]", "<|pad|>", "<mask>", "[MASK]",
    ] {
        if let Some(token) = tok.token_id(text) {
            if !allowed_generated_controls.contains(&token.get()) {
                forbidden.insert(token.get());
            }
        }
    }
    let scan_limit = tok.vocab_size().min(GENERATION_POLICY_SCAN_LIMIT);
    for token_id in 0..scan_limit {
        let id = token_id as u32;
        if allowed_generated_controls.contains(&id) {
            continue;
        }
        let token = TokenId::new(id);
        let raw_text_forbidden = tok
            .token_text(token)
            .is_some_and(is_forbidden_generation_token_text);
        let decoded_text_forbidden = tok
            .decode(&[token], true)
            .map(|text| is_forbidden_generation_token_text(&text))
            .unwrap_or(true);
        if raw_text_forbidden || decoded_text_forbidden {
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
        Self {
            request_id: request.id.clone(),
            original_request: request.clone(),
            input_tokens,
            generated_tokens: Vec::new(),
            kv_cache: None,
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
            streamed_text_len: 0,
        }
    }

    pub fn total_tokens(&self) -> usize {
        self.input_tokens.len() + self.generated_tokens.len()
    }

    pub fn model_decode_metadata(&self) -> HashMap<String, serde_json::Value> {
        let mut metadata = self.original_request.metadata.clone();
        let needs_sampling_masks = !self.forbidden_token_ids.is_empty()
            || (self.generated_tokens.is_empty() && !self.initial_forbidden_token_ids.is_empty());
        if self.json_processor.is_some() || self.regex_processor.is_some() || needs_sampling_masks {
            metadata.insert(
                "ferrum_require_full_logits".to_string(),
                serde_json::json!(true),
            );
        }
        metadata
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
        use ferrum_interfaces::sampler::{SamplingConfig, SamplingContext};

        // Regex-guided mask runs FIRST: it's a hard constraint (sets invalid
        // tokens to -inf). Subsequent temperature / top-k / top-p stay
        // correct because -inf tokens can't make it through any softmax.
        if let Some(ref rp) = self.regex_processor {
            rp.advance_with_tokens_public(&self.generated_tokens);
            rp.mask_logits(logits);
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
        let ctx = SamplingContext::new(
            step,
            &self.sampling_params,
            logits,
            &self.generated_tokens,
            &self.token_frequencies,
            vocab_size,
        );
        let token = config.sample(ctx, &mut self.rng)?;

        // Update frequency tracking
        *self.token_frequencies.entry(token).or_insert(0) += 1;

        Ok(token)
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
    /// Set true the first time `ensure_bg_loop` runs, so per-request
    /// `infer_stream` callers don't each spawn their own competing
    /// driver task (16 streaming requests = 16 drivers thrashing on
    /// `iteration_lock`, ~5ms/iter of tokio scheduling overhead).
    bg_loop_spawned: AtomicBool,
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
        info!(
            "Creating ContinuousBatchEngine (speculative_decoding={})",
            draft_executor.is_some() && spec_config.is_some()
        );
        let runtime_config = ContinuousEngineRuntimeConfig::from_engine_config_and_env(&config);

        Self {
            inner: Arc::new(EngineInner {
                config,
                scheduler,
                tokenizer,
                sampler,
                kv_cache,
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
                    let _guard = inner.iteration_lock.lock().await;
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
            queue_metrics: Default::default(),
            resource_utilization: Default::default(),
            error_stats: Default::default(),
            performance_breakdown: Default::default(),
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
mod tests {
    use super::*;
    use ferrum_interfaces::tokenizer::{TokenizerInfo, TokenizerType};

    struct PolicyTokenizer {
        vocab_size: usize,
        special: ferrum_types::SpecialTokens,
        ids: HashMap<String, TokenId>,
        texts: Vec<Option<String>>,
    }

    impl PolicyTokenizer {
        fn new(vocab_size: usize, pairs: &[(&str, u32)]) -> Self {
            let max_id = pairs.iter().map(|(_, id)| *id as usize).max().unwrap_or(0);
            let mut texts = vec![None; max_id + 1];
            let mut ids = HashMap::new();
            for (text, id) in pairs {
                ids.insert((*text).to_string(), TokenId::new(*id));
                texts[*id as usize] = Some((*text).to_string());
            }
            Self {
                vocab_size,
                special: ferrum_types::SpecialTokens {
                    bos_token: Some(TokenId::new(1)),
                    eos_token: Some(TokenId::new(3)),
                    unk_token: Some(TokenId::new(2)),
                    pad_token: Some(TokenId::new(4)),
                    sep_token: None,
                    cls_token: None,
                    mask_token: None,
                },
                ids,
                texts,
            }
        }
    }

    impl Tokenizer for PolicyTokenizer {
        fn encode(&self, _text: &str, _add_special: bool) -> Result<Vec<TokenId>> {
            Ok(vec![TokenId::new(0)])
        }

        fn decode(&self, tokens: &[TokenId], _skip_special: bool) -> Result<String> {
            Ok(tokens
                .iter()
                .filter_map(|token| self.token_text(*token))
                .map(|text| {
                    if text == "byte-fallback" {
                        "\u{FFFD}"
                    } else {
                        text
                    }
                })
                .collect::<Vec<_>>()
                .join(""))
        }

        fn decode_incremental(&self, _prev: &[TokenId], next: TokenId) -> Result<String> {
            Ok(self.token_text(next).unwrap_or_default().to_string())
        }

        fn vocab_size(&self) -> usize {
            self.vocab_size
        }

        fn special_tokens(&self) -> &ferrum_types::SpecialTokens {
            &self.special
        }

        fn token_id(&self, text: &str) -> Option<TokenId> {
            self.ids.get(text).copied()
        }

        fn token_text(&self, token_id: TokenId) -> Option<&str> {
            self.texts
                .get(token_id.get() as usize)
                .and_then(|text| text.as_deref())
        }

        fn info(&self) -> TokenizerInfo {
            TokenizerInfo {
                tokenizer_type: TokenizerType::Custom,
                vocab_size: self.vocab_size,
                special_tokens: self.special.clone(),
                supports_incremental: true,
                supports_chat_template: false,
                max_token_length: None,
                model_name: Some("policy-tokenizer-test".to_string()),
            }
        }
    }

    fn policy_request() -> InferenceRequest {
        InferenceRequest {
            id: RequestId::new(),
            prompt: "test".to_string(),
            model_id: ferrum_types::ModelId::new("test"),
            sampling_params: SamplingParams::greedy(),
            stream: false,
            priority: Priority::Normal,
            client_id: None,
            session_id: None,
            created_at: chrono::Utc::now(),
            api_request: None,
            metadata: HashMap::new(),
        }
    }

    #[test]
    fn continuous_engine_runtime_config_parses_env_snapshot() {
        let cfg = ContinuousEngineRuntimeConfig::from_env_vars(
            Some(64),
            [
                (BATCH_DECODE_PROF_ENV, "1"),
                (CHUNKED_PREFILL_ENV, "128"),
                (KV_CAPACITY_ENV, "2048"),
                (MAX_MODEL_LEN_ENV, "4096"),
                (NEXT_BATCH_PROF_ENV, "1"),
                (WHOLE_PROMPT_PREFIX_CACHE_ENV, "1"),
                (RBD_PROF_ENV, "1"),
                (UNIFIED_POST_PROF_ENV, "1"),
            ],
        );

        assert_eq!(cfg.active_decode_prefill_chunk, Some(64));
        assert!(cfg.batch_decode_prof);
        assert!(cfg.chunked_prefill_present);
        assert_eq!(cfg.chunked_prefill_size, Some(128));
        assert_eq!(cfg.chunked_prefill_size_for(200), Some(128));
        assert_eq!(cfg.chunked_prefill_size_for(128), None);
        assert_eq!(cfg.kv_capacity, Some(2048));
        assert_eq!(cfg.max_model_len, Some(4096));
        assert!(cfg.next_batch_prof);
        assert!(cfg.prefix_cache_enabled);
        assert!(cfg.rbd_prof);
        assert!(cfg.unified_post_prof);
    }

    #[test]
    fn continuous_engine_runtime_config_keeps_invalid_chunk_presence() {
        let cfg = ContinuousEngineRuntimeConfig::from_env_vars(
            None,
            [
                (CHUNKED_PREFILL_ENV, "invalid"),
                (WHOLE_PROMPT_PREFIX_CACHE_ENV, "0"),
            ],
        );

        assert!(cfg.chunked_prefill_present);
        assert_eq!(cfg.chunked_prefill_size, None);
        assert_eq!(cfg.chunked_prefill_size_for(200), None);
        assert!(!cfg.prefix_cache_enabled);
    }

    #[test]
    fn request_context_capacity_uses_executor_kv_capacity_when_smaller() {
        let mut config = EngineConfig::default();
        config.kv_cache.max_blocks = 2048;
        let runtime =
            ContinuousEngineRuntimeConfig::from_env_vars(None, Vec::<(&str, &str)>::new());

        assert_eq!(
            effective_request_context_capacity(&config, &runtime, Some(512)),
            Some(512)
        );
    }

    #[test]
    fn test_sequence_state() {
        let request = InferenceRequest {
            id: RequestId::new(),
            prompt: "test".to_string(),
            model_id: ferrum_types::ModelId::new("test"),
            sampling_params: SamplingParams::default(),
            stream: false,
            priority: Priority::Normal,
            client_id: None,
            session_id: None,
            created_at: chrono::Utc::now(),
            api_request: None,
            metadata: HashMap::new(),
        };

        let tokens = vec![TokenId::new(1), TokenId::new(2)];
        let state = SequenceState::new(request, tokens);

        assert_eq!(state.phase, RequestPhase::Waiting);
        assert_eq!(state.total_tokens(), 2);
        assert!(!state.prefill_complete);
    }

    #[test]
    fn sequence_state_detects_text_stop_before_length() {
        let tokenizer = PolicyTokenizer::new(8, &[("OK", 5), ("<END>", 6), ("TAIL", 7)]);
        let mut request = policy_request();
        request.sampling_params.max_tokens = 3;
        let mut state = SequenceState::new(request, vec![TokenId::new(0)]);
        state.generated_tokens = vec![TokenId::new(5), TokenId::new(6), TokenId::new(7)];
        state.stop_text_seqs = vec!["<END>".to_string()];

        assert_eq!(
            state.stop_reason(Some(&tokenizer)),
            Some(FinishReason::Stop)
        );
    }

    #[test]
    fn model_decode_metadata_marks_structured_requests_for_full_logits() {
        let plain = SequenceState::new(policy_request(), vec![TokenId::new(0)]);
        assert_eq!(
            plain
                .model_decode_metadata()
                .get("ferrum_require_full_logits")
                .and_then(|value| value.as_bool()),
            None
        );

        let mut request = policy_request();
        request.sampling_params.response_format = ferrum_types::ResponseFormat::JsonObject;
        let structured = SequenceState::new(request, vec![TokenId::new(0)]);
        assert_eq!(
            structured
                .model_decode_metadata()
                .get("ferrum_require_full_logits")
                .and_then(|value| value.as_bool()),
            Some(true)
        );
    }

    #[test]
    fn model_decode_metadata_marks_sampling_masks_for_full_logits() {
        let tokenizer: Arc<dyn Tokenizer + Send + Sync> = Arc::new(PolicyTokenizer::new(
            4,
            &[("normal", 0), ("<s>", 1), ("<unk>", 2), ("ok", 3)],
        ));
        let state = SequenceState::new_with_tokenizer(
            policy_request(),
            vec![TokenId::new(0)],
            Some(tokenizer),
        );

        assert_eq!(
            state
                .model_decode_metadata()
                .get("ferrum_require_full_logits")
                .and_then(|value| value.as_bool()),
            Some(true)
        );
    }

    #[test]
    fn sample_masks_unknown_pad_reserved_and_bos_tokens() {
        let tokenizer: Arc<dyn Tokenizer + Send + Sync> = Arc::new(PolicyTokenizer::new(
            10,
            &[
                ("normal", 0),
                ("<s>", 1),
                ("<unk>", 2),
                ("</s>", 3),
                ("[PAD151935]", 4),
                ("<|reserved_special_token_0|>", 5),
                ("ok", 6),
                ("other", 7),
                ("byte-fallback", 8),
                ("\u{00ef}\u{00bf}\u{00bd}", 9),
            ],
        ));
        let mut state = SequenceState::new_with_tokenizer(
            policy_request(),
            vec![TokenId::new(0)],
            Some(tokenizer),
        );
        let mut logits = vec![0.0f32; 10];
        logits[1] = 100.0;
        logits[2] = 99.0;
        logits[4] = 98.0;
        logits[5] = 97.0;
        logits[8] = 96.0;
        logits[9] = 95.0;
        logits[6] = 1.0;

        let token = state.sample_with_processors(&mut logits).unwrap();

        assert_eq!(token.get(), 6);
        for token_id in [1usize, 2, 4, 5, 8, 9] {
            assert_eq!(logits[token_id], f32::NEG_INFINITY);
        }
    }

    #[test]
    fn sample_allows_generated_control_tokens_above_base_vocab() {
        let tokenizer: Arc<dyn Tokenizer + Send + Sync> = Arc::new(PolicyTokenizer::new(
            6,
            &[
                ("normal", 0),
                ("<s>", 1),
                ("<unk>", 2),
                ("</s>", 3),
                ("ok", 4),
                ("</think>", 5),
                ("[PAD151935]", 6),
            ],
        ));
        let mut state = SequenceState::new_with_tokenizer(
            policy_request(),
            vec![TokenId::new(0)],
            Some(tokenizer),
        );
        let mut logits = vec![0.0f32; 7];
        logits[4] = 1.0;
        logits[5] = 90.0;
        logits[6] = 100.0;

        let token = state.sample_with_processors(&mut logits).unwrap();

        assert_eq!(token.get(), 5);
        assert_eq!(logits[5], 90.0);
        assert_eq!(logits[6], f32::NEG_INFINITY);
    }

    #[test]
    fn sample_masks_metadata_initial_token_text_only_before_first_generation() {
        let tokenizer: Arc<dyn Tokenizer + Send + Sync> = Arc::new(PolicyTokenizer::new(
            6,
            &[
                ("normal", 0),
                ("<s>", 1),
                ("<unk>", 2),
                ("</s>", 3),
                ("ok", 4),
                ("</think>", 5),
            ],
        ));
        let mut request = policy_request();
        request.metadata.insert(
            "ferrum_initial_forbidden_token_texts".to_string(),
            serde_json::json!(["</think>"]),
        );
        let mut state =
            SequenceState::new_with_tokenizer(request, vec![TokenId::new(0)], Some(tokenizer));

        let mut first_logits = vec![0.0f32; 6];
        first_logits[0] = 1.0;
        first_logits[5] = 100.0;
        let first = state.sample_with_processors(&mut first_logits).unwrap();
        assert_eq!(first.get(), 0);
        assert_eq!(first_logits[5], f32::NEG_INFINITY);

        state.generated_tokens.push(first);
        let mut next_logits = vec![0.0f32; 6];
        next_logits[0] = 1.0;
        next_logits[5] = 100.0;
        let next = state.sample_with_processors(&mut next_logits).unwrap();
        assert_eq!(next.get(), 5);
        assert_eq!(next_logits[5], 100.0);
    }
}
