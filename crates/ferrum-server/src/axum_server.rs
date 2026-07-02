//! Axum-based HTTP server implementation for Ferrum
//!
//! This module provides a concrete implementation of the HttpServer trait
//! using the Axum web framework, with OpenAI-shaped endpoint compatibility.

use crate::{
    chat_template::{
        render_chat_prompt_with_model_template_options,
        render_chat_prompt_with_tools_and_model_template, ChatTemplateOptions, ModelChatTemplate,
    },
    openai::*,
    traits::HttpServer,
    types::*,
};
use async_trait::async_trait;
use axum::{
    extract::{multipart::MultipartRejection, rejection::JsonRejection, State},
    http::{HeaderMap, StatusCode as AxumStatusCode},
    response::{sse::Event, IntoResponse, Response, Sse},
    routing::{get, post},
    Json, Router,
};
use ferrum_interfaces::engine::{EmbedEngine, LlmInferenceEngine, TranscribeEngine, TtsEngine};
use ferrum_types::{
    EngineMetrics, EngineStatus, FerrumConfigBuilder, FerrumError as Error, FinishReason,
    InferenceRequest, InferenceResponse, ModelId, Priority, RequestId, ResolvedFerrumConfig,
    RuntimeConfigSnapshot, SamplingParams, TokenUsage, DEFAULT_CHAT_REPETITION_PENALTY,
    DEFAULT_MAX_TOKENS_METADATA_KEY, OBSERVABILITY_PROFILE_SCHEMA_VERSION,
};
use sha2::{Digest, Sha256};
use std::{
    collections::HashMap,
    fs,
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
};
use tokio::sync::mpsc;
use tokio_stream::StreamExt;
use tower::ServiceBuilder;
use tower_http::{cors::CorsLayer, trace::TraceLayer};
use tracing::{debug, error, info, span, warn, Level};
use uuid::Uuid;

const DEFAULT_SAMPLING_TEMPERATURE: f32 = 0.0;
const DEFAULT_SAMPLING_TOP_P: f32 = 1.0;
const DEFAULT_COMPLETION_MAX_TOKENS: u32 = 512;
const INITIAL_FORBIDDEN_TOKEN_TEXTS_METADATA_KEY: &str = "ferrum_initial_forbidden_token_texts";
const THINK_START_TAG: &str = "<think>";
const THINK_END_TAG: &str = "</think>";
const DEFAULT_GUIDED_TOOL_ARGUMENT_STRING_MAX_LENGTH: u64 = 128;
const INITIAL_STRUCTURED_CALL_FORBIDDEN_TOKEN_TEXTS: &[&str] =
    &["<|im_end|>", "<|endoftext|>", "<|eot_id|>", "</s>"];
const FERRUM_SESSION_HEADER: &str = "x-ferrum-session";

#[derive(Debug, Clone)]
struct CachePolicy {
    prefix_cache_enabled: bool,
    session_cache_mode: String,
    session_cache_max_entries: usize,
    session_cache_max_tokens: usize,
}

impl CachePolicy {
    fn current() -> Self {
        Self {
            prefix_cache_enabled: env_bool("FERRUM_PREFIX_CACHE_PRODUCT")
                .or_else(|| env_bool("FERRUM_PREFIX_CACHE_REQUESTED"))
                .or_else(|| env_bool("FERRUM_PREFIX_CACHE"))
                .unwrap_or(false),
            session_cache_mode: std::env::var("FERRUM_SESSION_CACHE")
                .unwrap_or_else(|_| "off".to_string())
                .to_ascii_lowercase(),
            session_cache_max_entries: env_usize("FERRUM_SESSION_CACHE_MAX_ENTRIES").unwrap_or(128),
            session_cache_max_tokens: env_usize("FERRUM_SESSION_CACHE_MAX_TOKENS").unwrap_or(4096),
        }
    }

    fn session_memory_enabled(&self) -> bool {
        self.session_cache_mode == "memory"
    }
}

fn env_bool(key: &str) -> Option<bool> {
    match std::env::var(key).ok()?.to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => Some(true),
        "0" | "false" | "no" | "off" => Some(false),
        _ => None,
    }
}

fn env_usize(key: &str) -> Option<usize> {
    std::env::var(key).ok()?.parse().ok()
}

/// Shared Prometheus recorder handle for rendering metrics.
static PROM_HANDLE: std::sync::OnceLock<metrics_exporter_prometheus::PrometheusHandle> =
    std::sync::OnceLock::new();

/// Initialize the Prometheus metrics recorder.
///
/// Must be called once before any `metrics::counter!()` / `histogram!()` calls.
/// Safe to call multiple times — subsequent calls are no-ops.
pub fn init_prometheus_recorder() {
    PROM_HANDLE.get_or_init(|| {
        let builder = metrics_exporter_prometheus::PrometheusBuilder::new();
        let handle = builder
            .install_recorder()
            .expect("Failed to install Prometheus recorder");
        info!("Prometheus metrics recorder installed");
        handle
    });
}

/// Axum-based server implementation.
///
/// The server is built around [`AppState`], which holds an optional
/// engine per modality. Handlers fault to 503 when the modality they
/// need isn't loaded, instead of running stub error logic.
pub struct AxumServer {
    state: AppState,
    config: ServerConfig,
}

impl AxumServer {
    /// Create a server with a fully populated AppState.
    pub fn from_state(state: AppState) -> Self {
        Self {
            state,
            config: ServerConfig::default(),
        }
    }

    /// Convenience constructor for an LLM-only server (chat / completions).
    pub fn from_llm(engine: Arc<dyn LlmInferenceEngine + Send + Sync>) -> Self {
        Self::from_state(AppState::default().with_llm(engine))
    }

    /// Convenience constructor for an embedding-only server (`/v1/embeddings`).
    pub fn from_embed(engine: Arc<dyn EmbedEngine + Send + Sync>) -> Self {
        Self::from_state(AppState::default().with_embed(engine))
    }

    /// Convenience constructor for a transcription-only server
    /// (`/v1/audio/transcriptions`).
    pub fn from_transcribe(engine: Arc<dyn TranscribeEngine + Send + Sync>) -> Self {
        Self::from_state(AppState::default().with_transcribe(engine))
    }

    /// Convenience constructor for a TTS-only server (`/v1/audio/speech`).
    pub fn from_tts(engine: Arc<dyn TtsEngine + Send + Sync>) -> Self {
        Self::from_state(AppState::default().with_tts(engine))
    }

    /// Attach the startup auto-configuration decision trace exposed by
    /// `/health`. Constructors keep this optional so tests and non-LLM
    /// deployments can use the server without a model-specific resolver.
    pub fn with_auto_config(mut self, auto_config: ResolvedFerrumConfig) -> Self {
        self.state = self.state.with_auto_config(auto_config);
        self
    }

    /// Attach the loaded model's prompt template, if available. This keeps
    /// OpenAI request aliases from driving prompt-family selection.
    pub fn with_prompt_template(mut self, prompt_template: Option<ModelChatTemplate>) -> Self {
        self.state = self.state.with_prompt_template(prompt_template);
        self
    }

    /// Attach startup-loaded LoRA adapter model ids.
    pub fn with_lora_adapters(
        mut self,
        base_model_id: impl Into<String>,
        adapters: Vec<LoraAdapterModel>,
    ) -> Self {
        self.state = self.state.with_lora_adapters(base_model_id, adapters);
        self
    }

    /// Build the router with all routes
    #[allow(dead_code)]
    fn build_router(&self) -> Router {
        self.build_router_with_state(self.state.clone())
    }

    fn build_router_with_state(&self, app_state: AppState) -> Router {
        Router::new()
            // OpenAI API routes
            .route("/v1/chat/completions", post(chat_completions_handler))
            .route("/v1/completions", post(completions_handler))
            .route("/v1/embeddings", post(embeddings_handler))
            .route("/v1/audio/transcriptions", post(transcriptions_handler))
            .route("/v1/audio/speech", post(speech_handler))
            .route("/v1/models", get(models_handler))
            // Health & observability
            .route("/health", get(health_handler))
            .route("/metrics", get(metrics_handler))
            .route("/", get(root_handler))
            // Apply middleware
            .layer(
                ServiceBuilder::new()
                    .layer(TraceLayer::new_for_http())
                    .layer(CorsLayer::permissive()), // For MVP, allow all origins
            )
            .with_state(app_state)
    }
}

/// Application state shared across handlers — one optional engine per
/// modality. Handlers reach into the field they need and 503 when it's
/// not loaded.
#[derive(Clone, Default)]
pub struct AppState {
    pub llm: Option<Arc<dyn LlmInferenceEngine + Send + Sync>>,
    pub embed: Option<Arc<dyn EmbedEngine + Send + Sync>>,
    pub transcribe: Option<Arc<dyn TranscribeEngine + Send + Sync>>,
    pub tts: Option<Arc<dyn TtsEngine + Send + Sync>>,
    pub auto_config: Option<ResolvedFerrumConfig>,
    pub prompt_template: Option<Arc<ModelChatTemplate>>,
    pub lora_registry: Arc<LoraModelRegistry>,
    pub request_dump_dir: Option<Arc<PathBuf>>,
    cache: Arc<CacheRuntimeState>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LoraAdapterModel {
    pub name: String,
    pub model_id: String,
    pub path: String,
}

impl LoraAdapterModel {
    pub fn new(
        name: impl Into<String>,
        model_id: impl Into<String>,
        path: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            model_id: model_id.into(),
            path: path.into(),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct LoraModelRegistry {
    base_model_id: Option<String>,
    adapters: Vec<LoraAdapterModel>,
}

#[derive(Clone, Debug)]
struct LoraModelResolution {
    base_model_id: String,
    adapter: Option<LoraAdapterModel>,
}

impl LoraModelRegistry {
    pub fn new(base_model_id: impl Into<String>, adapters: Vec<LoraAdapterModel>) -> Self {
        Self {
            base_model_id: Some(base_model_id.into()),
            adapters,
        }
    }

    pub fn is_enabled(&self) -> bool {
        !self.adapters.is_empty()
    }

    fn adapter_models(&self) -> &[LoraAdapterModel] {
        &self.adapters
    }

    fn resolve(
        &self,
        request_model: &str,
        loaded_models: &[ModelId],
    ) -> std::result::Result<Option<LoraModelResolution>, ServerError> {
        if !self.is_enabled() {
            return Ok(None);
        }
        let base = self
            .base_model_id
            .clone()
            .or_else(|| loaded_models.first().map(ToString::to_string))
            .unwrap_or_default();
        if request_model == base || loaded_models.iter().any(|model| model.0 == request_model) {
            return Ok(Some(LoraModelResolution {
                base_model_id: request_model.to_string(),
                adapter: None,
            }));
        }
        if let Some(adapter) = self
            .adapters
            .iter()
            .find(|adapter| adapter.model_id == request_model)
        {
            return Ok(Some(LoraModelResolution {
                base_model_id: base,
                adapter: Some(adapter.clone()),
            }));
        }
        Err(ServerError::invalid_request(
            format!("unknown LoRA adapter model: {request_model}"),
            Some("model"),
        ))
    }
}

impl AppState {
    pub fn with_llm(mut self, engine: Arc<dyn LlmInferenceEngine + Send + Sync>) -> Self {
        self.llm = Some(engine);
        self
    }
    pub fn with_embed(mut self, engine: Arc<dyn EmbedEngine + Send + Sync>) -> Self {
        self.embed = Some(engine);
        self
    }
    pub fn with_transcribe(mut self, engine: Arc<dyn TranscribeEngine + Send + Sync>) -> Self {
        self.transcribe = Some(engine);
        self
    }
    pub fn with_tts(mut self, engine: Arc<dyn TtsEngine + Send + Sync>) -> Self {
        self.tts = Some(engine);
        self
    }

    pub fn with_auto_config(mut self, auto_config: ResolvedFerrumConfig) -> Self {
        self.auto_config = Some(auto_config);
        self
    }

    pub fn with_prompt_template(mut self, prompt_template: Option<ModelChatTemplate>) -> Self {
        self.prompt_template = prompt_template.map(Arc::new);
        self
    }

    pub fn with_lora_adapters(
        mut self,
        base_model_id: impl Into<String>,
        adapters: Vec<LoraAdapterModel>,
    ) -> Self {
        self.lora_registry = Arc::new(LoraModelRegistry::new(base_model_id, adapters));
        self
    }

    pub fn with_request_dump_dir(mut self, request_dump_dir: Option<PathBuf>) -> Self {
        self.request_dump_dir = request_dump_dir.map(Arc::new);
        self
    }

    /// Async aggregated status across whichever modality is loaded.
    /// In single-modality deployments (current CLI), exactly one is Some.
    async fn status(&self) -> EngineStatus {
        if let Some(e) = &self.llm {
            return e.status().await;
        }
        if let Some(e) = &self.embed {
            return e.status().await;
        }
        if let Some(e) = &self.transcribe {
            return e.status().await;
        }
        if let Some(e) = &self.tts {
            return e.status().await;
        }
        EngineStatus {
            is_ready: false,
            loaded_models: vec![],
            active_requests: 0,
            queued_requests: 0,
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

    fn metrics(&self) -> EngineMetrics {
        if let Some(e) = &self.llm {
            return e.metrics();
        }
        if let Some(e) = &self.embed {
            return e.metrics();
        }
        if let Some(e) = &self.transcribe {
            return e.metrics();
        }
        if let Some(e) = &self.tts {
            return e.metrics();
        }
        EngineMetrics {
            total_requests: 0,
            successful_requests: 0,
            failed_requests: 0,
            avg_request_latency_ms: 0.0,
            p95_request_latency_ms: 0.0,
            p99_request_latency_ms: 0.0,
            throughput_rps: 0.0,
            tokens_per_second: 0.0,
            queue_metrics: Default::default(),
            resource_utilization: Default::default(),
            error_stats: Default::default(),
            performance_breakdown: Default::default(),
        }
    }
}

#[derive(Default)]
struct CacheRuntimeState {
    stats: Mutex<CacheStats>,
    prefix_prompts: Mutex<HashMap<String, usize>>,
    sessions: Mutex<HashMap<String, Vec<ChatMessage>>>,
}

#[derive(Debug, Clone, Default)]
struct CacheStats {
    prefix_hits: u64,
    prefix_misses: u64,
    prefix_evictions: u64,
    prefix_saved_prefill_tokens: u64,
    prefix_entries: u64,
    prefix_bytes: u64,
    session_hits: u64,
    session_misses: u64,
    session_evictions: u64,
    session_entries: u64,
    session_tokens: u64,
}

#[derive(Clone)]
struct SessionContext {
    id: String,
    prior_messages: Vec<ChatMessage>,
    incoming_messages: Vec<ChatMessage>,
}

impl CacheRuntimeState {
    fn record_prefix_prompt(&self, prompt: &str, policy: &CachePolicy) {
        if !policy.prefix_cache_enabled {
            return;
        }

        let prompt_tokens = approx_tokens(prompt);
        let mut prompts = self.prefix_prompts.lock().expect("prefix cache lock");
        let saved_tokens = prompts
            .keys()
            .map(|seen| approx_tokens_for_chars(longest_common_prefix_chars(seen, prompt)))
            .max()
            .unwrap_or(0);

        let mut stats = self.stats.lock().expect("cache stats lock");
        if saved_tokens > 0 {
            stats.prefix_hits += 1;
            stats.prefix_saved_prefill_tokens += saved_tokens as u64;
        } else {
            stats.prefix_misses += 1;
        }

        let max_entries = policy.session_cache_max_entries.max(1);
        if !prompts.contains_key(prompt) && prompts.len() >= max_entries {
            if let Some(key) = prompts.keys().next().cloned() {
                prompts.remove(&key);
                stats.prefix_evictions += 1;
            }
        }
        prompts.insert(prompt.to_string(), prompt_tokens);
        stats.prefix_entries = prompts.len() as u64;
        stats.prefix_bytes = prompts.keys().map(|key| key.len() as u64).sum();
    }

    fn prepare_session_request(
        &self,
        request: &mut ChatCompletionsRequest,
        headers: &HeaderMap,
        policy: &CachePolicy,
    ) -> Option<SessionContext> {
        let session_id = request_session_id(headers, request)?;
        if !policy.session_memory_enabled() {
            return None;
        }

        let incoming_messages = request.messages.clone();
        let prior_messages = {
            let sessions = self.sessions.lock().expect("session cache lock");
            sessions.get(&session_id).cloned().unwrap_or_default()
        };
        {
            let mut stats = self.stats.lock().expect("cache stats lock");
            if prior_messages.is_empty() {
                stats.session_misses += 1;
            } else {
                stats.session_hits += 1;
                let mut merged = prior_messages.clone();
                merged.extend(request.messages.clone());
                request.messages = merged;
            }
        }

        Some(SessionContext {
            id: session_id,
            prior_messages,
            incoming_messages,
        })
    }

    fn update_session(
        &self,
        context: Option<SessionContext>,
        assistant_message: ChatMessage,
        policy: &CachePolicy,
    ) {
        let Some(context) = context else {
            return;
        };
        if !policy.session_memory_enabled() {
            return;
        }

        let mut history = context.prior_messages;
        history.extend(context.incoming_messages);
        history.push(assistant_message);
        trim_messages_to_token_budget(&mut history, policy.session_cache_max_tokens);

        let mut sessions = self.sessions.lock().expect("session cache lock");
        if !sessions.contains_key(&context.id)
            && sessions.len() >= policy.session_cache_max_entries.max(1)
        {
            if let Some(evict_key) = sessions.keys().next().cloned() {
                sessions.remove(&evict_key);
                self.stats
                    .lock()
                    .expect("cache stats lock")
                    .session_evictions += 1;
            }
        }
        sessions.insert(context.id, history);

        let entries = sessions.len() as u64;
        let tokens = sessions
            .values()
            .map(|messages| {
                messages
                    .iter()
                    .map(|msg| approx_tokens(&msg.content))
                    .sum::<usize>()
            })
            .sum::<usize>() as u64;
        let mut stats = self.stats.lock().expect("cache stats lock");
        stats.session_entries = entries;
        stats.session_tokens = tokens;
    }

    fn stats(&self) -> CacheStats {
        let mut stats = self.stats.lock().expect("cache stats lock").clone();
        stats.prefix_entries = self.prefix_prompts.lock().expect("prefix cache lock").len() as u64;
        let sessions = self.sessions.lock().expect("session cache lock");
        stats.session_entries = sessions.len() as u64;
        stats.session_tokens = sessions
            .values()
            .map(|messages| {
                messages
                    .iter()
                    .map(|msg| approx_tokens(&msg.content))
                    .sum::<usize>()
            })
            .sum::<usize>() as u64;
        stats
    }

    fn health_json(
        &self,
        policy: &CachePolicy,
        engine_prefix_cache: Option<&serde_json::Value>,
    ) -> serde_json::Value {
        let stats = self.stats();
        let prefix_hits = engine_u64(engine_prefix_cache, "hits").unwrap_or(stats.prefix_hits);
        let prefix_misses =
            engine_u64(engine_prefix_cache, "misses").unwrap_or(stats.prefix_misses);
        let prefix_evictions =
            engine_u64(engine_prefix_cache, "evictions").unwrap_or(stats.prefix_evictions);
        let prefix_saved = engine_u64(engine_prefix_cache, "saved_prefill_tokens")
            .unwrap_or(stats.prefix_saved_prefill_tokens);
        let prefix_entries =
            engine_u64(engine_prefix_cache, "entries").unwrap_or(stats.prefix_entries);
        let prefix_bytes = engine_u64(engine_prefix_cache, "bytes").unwrap_or(stats.prefix_bytes);
        let mut prefix_cache = serde_json::json!({
            "enabled": engine_bool(engine_prefix_cache, "enabled").unwrap_or(policy.prefix_cache_enabled),
            "position": engine_str(engine_prefix_cache, "position").unwrap_or("product-observability"),
            "source": engine_str(engine_prefix_cache, "source").unwrap_or("server-prompt-lcp-observability"),
            "entries": prefix_entries,
            "hits": prefix_hits,
            "misses": prefix_misses,
            "evictions": prefix_evictions,
            "saved_prefill_tokens": prefix_saved,
            "bytes": prefix_bytes,
            "block_size": engine_u64(engine_prefix_cache, "block_size"),
            "kv_dtype": engine_str(engine_prefix_cache, "kv_dtype"),
        });
        if let (Some(engine), Some(prefix)) = (
            engine_prefix_cache.and_then(|value| value.as_object()),
            prefix_cache.as_object_mut(),
        ) {
            for (key, value) in engine {
                prefix.entry(key.clone()).or_insert_with(|| value.clone());
            }
        }
        serde_json::json!({
            "prefix_cache": prefix_cache,
            "session_cache": {
                "mode": policy.session_cache_mode,
                "entries": stats.session_entries,
                "hits": stats.session_hits,
                "misses": stats.session_misses,
                "evictions": stats.session_evictions,
                "tokens": stats.session_tokens,
                "max_entries": policy.session_cache_max_entries,
                "max_tokens": policy.session_cache_max_tokens,
            }
        })
    }

    fn prometheus_metrics(&self, engine_prefix_cache: Option<&serde_json::Value>) -> String {
        let stats = self.stats();
        let prefix_hits = engine_u64(engine_prefix_cache, "hits").unwrap_or(stats.prefix_hits);
        let prefix_misses =
            engine_u64(engine_prefix_cache, "misses").unwrap_or(stats.prefix_misses);
        let prefix_evictions =
            engine_u64(engine_prefix_cache, "evictions").unwrap_or(stats.prefix_evictions);
        let prefix_saved = engine_u64(engine_prefix_cache, "saved_prefill_tokens")
            .unwrap_or(stats.prefix_saved_prefill_tokens);
        let prefix_entries =
            engine_u64(engine_prefix_cache, "entries").unwrap_or(stats.prefix_entries);
        let prefix_bytes = engine_u64(engine_prefix_cache, "bytes").unwrap_or(stats.prefix_bytes);
        format!(
            concat!(
                "ferrum_prefix_cache_hits_total {}\n",
                "ferrum_prefix_cache_misses_total {}\n",
                "ferrum_prefix_cache_evictions_total {}\n",
                "ferrum_prefix_cache_saved_prefill_tokens_total {}\n",
                "ferrum_prefix_cache_entries {}\n",
                "ferrum_prefix_cache_bytes {}\n",
                "ferrum_session_cache_hits_total {}\n",
                "ferrum_session_cache_misses_total {}\n",
                "ferrum_session_cache_evictions_total {}\n",
                "ferrum_session_cache_entries {}\n",
                "ferrum_session_cache_tokens {}\n"
            ),
            prefix_hits,
            prefix_misses,
            prefix_evictions,
            prefix_saved,
            prefix_entries,
            prefix_bytes,
            stats.session_hits,
            stats.session_misses,
            stats.session_evictions,
            stats.session_entries,
            stats.session_tokens,
        )
    }
}

fn engine_u64(snapshot: Option<&serde_json::Value>, key: &str) -> Option<u64> {
    snapshot?.get(key)?.as_u64()
}

fn engine_bool(snapshot: Option<&serde_json::Value>, key: &str) -> Option<bool> {
    snapshot?.get(key)?.as_bool()
}

fn engine_str<'a>(snapshot: Option<&'a serde_json::Value>, key: &str) -> Option<&'a str> {
    snapshot?.get(key)?.as_str()
}

fn auto_config_health_value(auto_config: Option<&ResolvedFerrumConfig>) -> serde_json::Value {
    match auto_config {
        Some(auto_config) => auto_config.effective_config_document(),
        None => {
            match FerrumConfigBuilder::new(RuntimeConfigSnapshot::capture_current()).resolve() {
                Ok(auto_config) => auto_config.effective_config_document(),
                Err(err) => serde_json::json!({
                    "schema_version": 1,
                    "error": err.to_string(),
                }),
            }
        }
    }
}

fn admission_health_json(
    engine_status: &EngineStatus,
    scheduler_metrics: &EngineMetrics,
    auto_config: &serde_json::Value,
) -> serde_json::Value {
    let configured = auto_config
        .get("admission")
        .and_then(|value| value.as_object());
    let effective_max_concurrent = configured
        .and_then(|value| value.get("effective_max_concurrent"))
        .and_then(|value| value.as_u64())
        .unwrap_or_else(|| {
            (engine_status.active_requests + engine_status.queued_requests)
                .max(1)
                .try_into()
                .unwrap_or(u64::MAX)
        });
    serde_json::json!({
        "schema_version": 1,
        "source": "startup_auto_config_and_engine_status",
        "effective_max_concurrent": effective_max_concurrent,
        "queue_depth": engine_status.queued_requests as u64,
        "active_prefill": 0u64,
        "active_decode": engine_status.active_requests as u64,
        "current_batch_size": engine_status.active_requests as u64,
        "rejected_requests_total": 0u64,
        "failed_requests_total": scheduler_metrics.failed_requests,
        "completed_requests_total": scheduler_metrics.successful_requests,
        "avg_queue_wait_time_ms": scheduler_metrics.queue_metrics.avg_queue_wait_time_ms,
        "scheduler_policy": configured
            .and_then(|value| value.get("scheduler_policy"))
            .and_then(|value| value.as_str())
            .unwrap_or("unknown"),
        "phase_detail_source": "engine_status_does_not_split_prefill_decode",
    })
}

fn admission_prometheus_metrics(admission: &serde_json::Value) -> String {
    let value = |key: &str| {
        admission
            .get(key)
            .and_then(|value| value.as_u64())
            .unwrap_or(0)
    };
    format!(
        concat!(
            "ferrum_admission_effective_max_concurrent {}\n",
            "ferrum_admission_queue_depth {}\n",
            "ferrum_admission_active_prefill {}\n",
            "ferrum_admission_active_decode {}\n",
            "ferrum_admission_current_batch_size {}\n",
            "ferrum_admission_rejected_requests_total {}\n",
            "ferrum_admission_failed_requests_total {}\n",
            "ferrum_admission_completed_requests_total {}\n"
        ),
        value("effective_max_concurrent"),
        value("queue_depth"),
        value("active_prefill"),
        value("active_decode"),
        value("current_batch_size"),
        value("rejected_requests_total"),
        value("failed_requests_total"),
        value("completed_requests_total"),
    )
}

fn request_session_id(headers: &HeaderMap, request: &ChatCompletionsRequest) -> Option<String> {
    headers
        .get(FERRUM_SESSION_HEADER)
        .and_then(|value| value.to_str().ok())
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
        .or_else(|| {
            request
                .metadata
                .as_ref()
                .and_then(|metadata| metadata.get("ferrum_session_id"))
                .and_then(|value| value.as_str())
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .map(str::to_string)
        })
}

fn approx_tokens(text: &str) -> usize {
    approx_tokens_for_chars(text.chars().count())
}

fn approx_tokens_for_chars(chars: usize) -> usize {
    (chars / 4).max(1)
}

fn longest_common_prefix_chars(a: &str, b: &str) -> usize {
    a.chars().zip(b.chars()).take_while(|(a, b)| a == b).count()
}

fn trim_messages_to_token_budget(messages: &mut Vec<ChatMessage>, max_tokens: usize) {
    let max_tokens = max_tokens.max(1);
    while messages.len() > 1
        && messages
            .iter()
            .map(|msg| approx_tokens(&msg.content))
            .sum::<usize>()
            > max_tokens
    {
        messages.remove(0);
    }
}

#[async_trait]
impl HttpServer for AxumServer {
    async fn start(&self, config: &ServerConfig) -> ferrum_types::Result<()> {
        let addr = format!("{}:{}", config.host, config.port);
        info!("Starting Axum server on {}", addr);

        let app = self.build_router_with_state(
            self.state
                .clone()
                .with_request_dump_dir(config.request_dump_dir.clone()),
        );
        let listener = tokio::net::TcpListener::bind(&addr)
            .await
            .map_err(|e| Error::internal(format!("Failed to bind to {}: {}", addr, e)))?;

        info!("Server listening on {}", addr);

        axum::serve(listener, app)
            .await
            .map_err(|e| Error::internal(format!("Server error: {}", e)))?;

        Ok(())
    }

    async fn stop(&self, _timeout: std::time::Duration) -> ferrum_types::Result<()> {
        info!("Stopping Axum server");
        // Axum doesn't have explicit stop - server stops when task is cancelled
        Ok(())
    }

    fn is_running(&self) -> bool {
        // For MVP, always return true when server object exists
        true
    }

    fn address(&self) -> Option<std::net::SocketAddr> {
        // For MVP, return configured address
        format!("{}:{}", self.config.host, self.config.port)
            .parse()
            .ok()
    }

    fn register_handler(
        &mut self,
        _path: &str,
        _method: HttpMethod,
        _handler: Box<dyn crate::traits::RequestHandler>,
    ) {
        // For MVP, routes are static
        unimplemented!("Dynamic handler registration not implemented in MVP")
    }

    fn register_middleware(&mut self, _middleware: Box<dyn crate::traits::Middleware>) {
        // For MVP, middleware is static
        unimplemented!("Dynamic middleware registration not implemented in MVP")
    }

    fn get_metrics(&self) -> ServerMetrics {
        // Return empty metrics for MVP
        ServerMetrics {
            total_requests: 0,
            requests_by_endpoint: std::collections::HashMap::new(),
            requests_by_status: std::collections::HashMap::new(),
            avg_response_time_ms: 0.0,
            p95_response_time_ms: 0.0,
            p99_response_time_ms: 0.0,
            active_connections: 0,
            bytes_sent: 0,
            bytes_received: 0,
            error_rate: 0.0,
            uptime_seconds: 0,
        }
    }

    async fn health_check(&self) -> HealthStatus {
        HealthStatus::Healthy
    }
}

/// Main chat completions handler
async fn chat_completions_handler(
    State(state): State<AppState>,
    headers: HeaderMap,
    request: std::result::Result<Json<ChatCompletionsRequest>, JsonRejection>,
) -> std::result::Result<Response, ServerError> {
    let Json(mut request) = request.map_err(|e| {
        ServerError::invalid_request(format!("invalid chat completions request: {e}"), None)
    })?;
    let cache_policy = CachePolicy::current();
    let session_context =
        state
            .cache
            .prepare_session_request(&mut request, &headers, &cache_policy);

    let span = span!(Level::INFO, "chat_completions", model = %request.model);
    let _enter = span.enter();

    info!(
        "Received chat completions request for model: {}",
        request.model
    );
    debug!("Request: {:?}", request);

    // OpenAI spec requires at least one message. Reject empty arrays at
    // the boundary rather than synthesising a fake prompt downstream.
    validate_chat_request(&request)?;
    let loaded_models = state.status().await.loaded_models;
    let lora_resolution = state
        .lora_registry
        .resolve(&request.model, &loaded_models)?;

    // Convert OpenAI request to internal format
    let template_model_id = lora_resolution
        .as_ref()
        .map(|resolution| resolution.base_model_id.clone())
        .or_else(|| loaded_models.first().map(ToString::to_string))
        .unwrap_or_else(|| request.model.clone());
    let mut inference_request = convert_chat_request_with_template_model(
        &request,
        &template_model_id,
        state.prompt_template.as_deref(),
    )
    .map_err(server_error_from_ferrum_error)?;
    apply_lora_resolution(&mut inference_request, lora_resolution.as_ref());
    state
        .cache
        .record_prefix_prompt(&inference_request.prompt, &cache_policy);
    if let Err(err) =
        write_chat_request_replay_bundle(&state, &headers, &request, &inference_request)
    {
        warn!("failed to write chat request replay bundle: {}", err);
    }

    // Check if streaming is requested
    if request.stream.unwrap_or(false) {
        handle_chat_completions_stream(state, request, inference_request).await
    } else {
        handle_chat_completions_sync(state, request, inference_request, session_context).await
    }
}

fn write_chat_request_replay_bundle(
    state: &AppState,
    headers: &HeaderMap,
    openai_request: &ChatCompletionsRequest,
    inference_request: &InferenceRequest,
) -> std::result::Result<(), String> {
    let Some(root) = state.request_dump_dir.as_ref() else {
        return Ok(());
    };
    let request_id = inference_request.id.to_string();
    let bundle_dir = root.join(&request_id);
    fs::create_dir_all(&bundle_dir).map_err(|err| err.to_string())?;

    let sanitized_body = sanitized_chat_request_body(openai_request);
    let replay_body_path = bundle_dir.join("replay_body.json");
    write_json_value(&replay_body_path, &sanitized_body)?;

    let request = serde_json::json!({
        "schema_version": OBSERVABILITY_PROFILE_SCHEMA_VERSION,
        "entrypoint": "serve",
        "request_id": request_id,
        "model": openai_request.model.clone(),
        "backend": "actual",
        "endpoint": "/v1/chat/completions",
        "method": "POST",
        "stream": openai_request.stream.unwrap_or(false),
        "actual_model_smoke": true,
        "sanitized": true,
        "http": {
            "method": "POST",
            "path": "/v1/chat/completions",
            "headers": sanitized_replay_headers(headers),
            "body": sanitized_body
        }
    });
    let files = [
        ("request.json", request),
        (
            "prompt_token_ids.json",
            serde_json::json!({
                "schema_version": OBSERVABILITY_PROFILE_SCHEMA_VERSION,
                "request_id": request_id,
                "model": openai_request.model.clone(),
                "tokenizer_or_model": openai_request.model.clone(),
                "token_ids": null,
                "token_count": null,
                "unavailable_reason": "server request replay captures the OpenAI body before prompt token ids are retained",
                "sanitized": true
            }),
        ),
        (
            "sampling_params.json",
            serde_json::json!({
                "schema_version": OBSERVABILITY_PROFILE_SCHEMA_VERSION,
                "request_id": request_id,
                "sampling_params": inference_request.sampling_params.clone(),
                "unavailable_reason": null
            }),
        ),
        (
            "runtime_effective_config.json",
            serde_json::json!({
                "schema_version": OBSERVABILITY_PROFILE_SCHEMA_VERSION,
                "request_id": request_id,
                "entrypoint": "serve",
                "endpoint": "/v1/chat/completions",
                "stream": openai_request.stream.unwrap_or(false),
                "request_dump_dir": root.to_string_lossy(),
                "sanitized": true
            }),
        ),
        (
            "backend_selection.json",
            serde_json::json!({
                "schema_version": OBSERVABILITY_PROFILE_SCHEMA_VERSION,
                "request_id": request_id,
                "backend": "actual",
                "model": openai_request.model.clone(),
                "actual_model_smoke": true
            }),
        ),
        (
            "output_token_ids.json",
            serde_json::json!({
                "schema_version": OBSERVABILITY_PROFILE_SCHEMA_VERSION,
                "request_id": request_id,
                "token_ids": [],
                "token_count": 0,
                "finish_reason": null,
                "unavailable_reason": "server request replay bundle is emitted at request admission in this WP9 slice"
            }),
        ),
        (
            "bad_output_scan.json",
            serde_json::json!({
                "schema_version": OBSERVABILITY_PROFILE_SCHEMA_VERSION,
                "request_id": request_id,
                "bad_output": false,
                "bad_text_count": 0,
                "reasons": [],
                "first_bad_text_span": null,
                "failure_kind": null,
                "output_chars": 0,
                "output_sha256": sha256_hex(b"")
            }),
        ),
        (
            "replay.command.json",
            serde_json::json!({
                "schema_version": OBSERVABILITY_PROFILE_SCHEMA_VERSION,
                "request_id": request_id,
                "entrypoint": "serve",
                "command": replay_curl_command(&bundle_dir),
                "argv": replay_curl_argv(&bundle_dir),
                "bundle_dir": bundle_dir.to_string_lossy(),
                "requires_running_server": true,
                "sanitized": true
            }),
        ),
    ];
    for (name, value) in files {
        write_json_value(&bundle_dir.join(name), &value)?;
    }
    fs::write(
        bundle_dir.join("output_text.txt"),
        format!(
            "[server request replay emitted before response]\nsha256={}\nchars=0\n",
            sha256_hex(b"")
        ),
    )
    .map_err(|err| err.to_string())?;
    Ok(())
}

fn sanitized_replay_headers(headers: &HeaderMap) -> serde_json::Value {
    let mut result = serde_json::Map::new();
    for key in ["content-type", "traceparent", "tracestate"] {
        if let Some(value) = headers.get(key).and_then(|value| value.to_str().ok()) {
            result.insert(key.to_string(), serde_json::json!(value));
        }
    }
    result.insert("authorization".to_string(), serde_json::json!("[redacted]"));
    result.insert("cookie".to_string(), serde_json::json!("[redacted]"));
    serde_json::Value::Object(result)
}

fn sanitized_chat_request_body(request: &ChatCompletionsRequest) -> serde_json::Value {
    let mut value = serde_json::to_value(request).unwrap_or_else(|_| {
        serde_json::json!({
            "model": request.model.clone(),
            "stream": request.stream.unwrap_or(false),
            "messages": []
        })
    });
    redact_json_value(&mut value, None);
    value
}

fn redact_json_value(value: &mut serde_json::Value, key: Option<&str>) {
    if key.is_some_and(is_secret_key) {
        *value = serde_json::json!("[redacted]");
        return;
    }
    if matches!(key, Some("content" | "arguments")) && value.is_string() {
        *value = serde_json::json!("[redacted]");
        return;
    }
    match value {
        serde_json::Value::Object(map) => {
            for field in ["content", "arguments"] {
                if let Some(chars) = map
                    .get(field)
                    .and_then(|child| child.as_str())
                    .map(|text| text.chars().count())
                {
                    map.insert(field.to_string(), serde_json::json!("[redacted]"));
                    map.insert(format!("{field}_redacted"), serde_json::json!(true));
                    map.insert(format!("{field}_chars"), serde_json::json!(chars));
                }
            }
            for (child_key, child) in map.iter_mut() {
                redact_json_value(child, Some(child_key.as_str()));
            }
        }
        serde_json::Value::Array(items) => {
            for child in items {
                redact_json_value(child, None);
            }
        }
        _ => {}
    }
}

fn is_secret_key(key: &str) -> bool {
    let normalized = key
        .chars()
        .filter(|ch| *ch != '-' && *ch != '_')
        .flat_map(char::to_lowercase)
        .collect::<String>();
    matches!(
        normalized.as_str(),
        "authorization"
            | "cookie"
            | "secret"
            | "apikey"
            | "password"
            | "accesstoken"
            | "refreshtoken"
            | "idtoken"
    )
}

fn replay_curl_argv(bundle_dir: &Path) -> Vec<String> {
    vec![
        "curl".to_string(),
        "-sS".to_string(),
        "-X".to_string(),
        "POST".to_string(),
        "http://127.0.0.1:8000/v1/chat/completions".to_string(),
        "-H".to_string(),
        "content-type: application/json".to_string(),
        "--data-binary".to_string(),
        format!("@{}", bundle_dir.join("replay_body.json").display()),
    ]
}

fn replay_curl_command(bundle_dir: &Path) -> String {
    replay_curl_argv(bundle_dir)
        .iter()
        .map(|part| shell_quote(part))
        .collect::<Vec<_>>()
        .join(" ")
}

fn shell_quote(value: &str) -> String {
    if value
        .chars()
        .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '/' | '.' | '_' | '-' | ':' | '@'))
    {
        value.to_string()
    } else {
        format!("'{}'", value.replace('\'', "'\\''"))
    }
}

fn write_json_value(path: &Path, value: &serde_json::Value) -> std::result::Result<(), String> {
    let bytes = serde_json::to_vec_pretty(value).map_err(|err| err.to_string())?;
    fs::write(path, [bytes, b"\n".to_vec()].concat()).map_err(|err| err.to_string())
}

fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

/// Handle streaming chat completions
async fn handle_chat_completions_stream(
    state: AppState,
    openai_request: ChatCompletionsRequest,
    inference_request: InferenceRequest,
) -> std::result::Result<Response, ServerError> {
    let (tx, rx) = mpsc::unbounded_channel::<std::result::Result<Event, axum::Error>>();

    // Spawn task to generate tokens
    let engine = state.llm.clone().ok_or_else(|| {
        ServerError::ServiceUnavailable("LLM engine not loaded; chat unavailable".into())
    })?;
    let request_id = Uuid::new_v4().to_string();
    let include_stream_usage = openai_request
        .stream_options
        .as_ref()
        .and_then(|opts| opts.include_usage)
        .unwrap_or(false);
    let buffer_json_object_stream = response_format_is_json_object(&openai_request);
    let buffer_strict_json_schema_stream = strict_json_schema_string(&openai_request)?.is_some();
    let stream_api_request = match inference_request.api_request.as_ref() {
        Some(ferrum_types::ApiRequest::Chat(request)) => request.clone(),
        _ => api_chat_request(&openai_request, openai_request.tool_choice.as_ref()),
    };
    let buffer_structured_api_stream =
        ferrum_types::chat_api_may_emit_tool_or_function_call(&stream_api_request);
    let buffer_stream_output = buffer_json_object_stream
        || buffer_strict_json_schema_stream
        || buffer_structured_api_stream;
    // R1-distill-style templates open the think block inside the prompt;
    // the parser must know generation starts mid-think.
    let started_in_think = has_unclosed_thinking_block(&inference_request.prompt);
    let mut stream = match engine.infer_stream(inference_request).await {
        Ok(stream) => stream,
        Err(e) => {
            error!("Stream generation failed before first chunk: {}", e);
            let _ = tx.send(Ok(openai_error_sse_event(
                e.to_string(),
                "internal_server_error",
                None,
            )));
            let _ = tx.send(Ok(Event::default().data("[DONE]")));
            let stream = tokio_stream::wrappers::UnboundedReceiverStream::new(rx);
            return Ok(Sse::new(stream).into_response());
        }
    };

    tokio::spawn(async move {
        let mut current_text = String::new();
        let mut sent_reasoning_len = 0usize;
        let mut sent_content_len = 0usize;

        while let Some(result) = stream.next().await {
            match result {
                Ok(chunk) => {
                    if !chunk.text.is_empty() {
                        current_text.push_str(&chunk.text);

                        if !buffer_stream_output {
                            if should_defer_reasoning_stream_delta(&current_text) {
                                continue;
                            }
                            let parsed = if started_in_think {
                                parse_reasoning_response_started_in_think(&current_text)
                            } else {
                                parse_reasoning_response(&current_text)
                            };
                            let full_reasoning = parsed.reasoning.as_deref().unwrap_or("");
                            let reasoning_delta =
                                stream_text_delta(full_reasoning, &mut sent_reasoning_len);
                            let content_delta =
                                stream_text_delta(&parsed.content, &mut sent_content_len);
                            if reasoning_delta.is_empty() && content_delta.is_empty() {
                                continue;
                            }
                            // Create streaming response chunk
                            let response_chunk = ChatCompletionsResponse {
                                id: request_id.clone(),
                                object: "chat.completion.chunk".to_string(),
                                created: chrono::Utc::now().timestamp() as u64,
                                model: openai_request.model.clone(),
                                choices: vec![ChatChoice {
                                    index: 0,
                                    message: None,
                                    delta: Some(ChatMessage {
                                        role: MessageRole::Assistant,
                                        content: content_delta,
                                        reasoning: (!reasoning_delta.is_empty())
                                            .then_some(reasoning_delta),
                                        name: None,
                                        tool_calls: None,
                                        tool_call_id: None,
                                        function_call: None,
                                    }),
                                    finish_reason: None,
                                }],
                                usage: None,
                            };

                            let sse_event = Event::default()
                                .json_data(&response_chunk)
                                .unwrap_or_else(|_| Event::default().data("error"));
                            if tx.send(Ok(sse_event)).is_err() {
                                break;
                            }
                        }
                    }

                    if chunk.finish_reason.is_some() {
                        let usage = chunk.usage.as_ref().map(openai_usage_from_token_usage);
                        let mut parsed_final = if started_in_think {
                            parse_reasoning_response_started_in_think(&current_text)
                        } else {
                            parse_reasoning_response(&current_text)
                        };
                        parsed_final.content = normalize_structured_response_content(
                            &openai_request,
                            &parsed_final.content,
                        );
                        if let Err(e) = validate_strict_json_schema_response(
                            &openai_request,
                            &parsed_final.content,
                        ) {
                            let error_event = openai_error_sse_event(
                                strict_stream_validation_error_message(e),
                                "internal_server_error",
                                Some("response_format.json_schema"),
                            );
                            let _ = tx.send(Ok(error_event));
                            let _ = tx.send(Ok(Event::default().data("[DONE]")));
                            break;
                        }
                        let structured_chat_response = match chunk.api_response.as_ref() {
                            Some(ferrum_types::ApiResponse::Chat(response)) => {
                                Some(response.clone())
                            }
                            _ if buffer_structured_api_stream => {
                                chat_api_response_from_parsed_generated_text(
                                    &stream_api_request,
                                    &parsed_final,
                                )
                            }
                            _ => None,
                        };

                        if let Some(chat_response) = structured_chat_response.as_ref() {
                            let mut delta = openai_chat_delta_from_api(&chat_response.message);
                            if delta.reasoning.is_none() {
                                delta.reasoning = parsed_final.reasoning.clone();
                            }
                            let response_chunk = ChatCompletionsResponse {
                                id: request_id.clone(),
                                object: "chat.completion.chunk".to_string(),
                                created: chrono::Utc::now().timestamp() as u64,
                                model: openai_request.model.clone(),
                                choices: vec![ChatChoice {
                                    index: 0,
                                    message: None,
                                    delta: Some(delta),
                                    finish_reason: None,
                                }],
                                usage: None,
                            };

                            let sse_event = Event::default()
                                .json_data(&response_chunk)
                                .unwrap_or_else(|_| Event::default().data("error"));
                            if tx.send(Ok(sse_event)).is_err() {
                                break;
                            }
                        } else if tool_choice_required(&openai_request) {
                            log_required_tool_choice_failure(
                                &openai_request,
                                &parsed_final.content,
                                parsed_final.reasoning.as_deref(),
                            );
                            let error_event = openai_error_sse_event(
                                "model output did not satisfy required tool_choice",
                                "invalid_request_error",
                                Some("tool_choice"),
                            );
                            let _ = tx.send(Ok(error_event));
                            let _ = tx.send(Ok(Event::default().data("[DONE]")));
                            break;
                        } else if buffer_structured_api_stream
                            && parsed_final.content.trim().is_empty()
                        {
                            let error_event = openai_error_sse_event(
                                "model output did not satisfy tool/function call request",
                                "internal_server_error",
                                Some("tool_choice"),
                            );
                            let _ = tx.send(Ok(error_event));
                            let _ = tx.send(Ok(Event::default().data("[DONE]")));
                            break;
                        } else if buffer_stream_output && !current_text.is_empty() {
                            let response_chunk = ChatCompletionsResponse {
                                id: request_id.clone(),
                                object: "chat.completion.chunk".to_string(),
                                created: chrono::Utc::now().timestamp() as u64,
                                model: openai_request.model.clone(),
                                choices: vec![ChatChoice {
                                    index: 0,
                                    message: None,
                                    delta: Some(ChatMessage {
                                        role: MessageRole::Assistant,
                                        content: parsed_final.content.clone(),
                                        reasoning: parsed_final.reasoning.clone(),
                                        name: None,
                                        tool_calls: None,
                                        tool_call_id: None,
                                        function_call: None,
                                    }),
                                    finish_reason: None,
                                }],
                                usage: None,
                            };

                            let sse_event = Event::default()
                                .json_data(&response_chunk)
                                .unwrap_or_else(|_| Event::default().data("error"));
                            if tx.send(Ok(sse_event)).is_err() {
                                break;
                            }
                        }
                        // Send final chunk. OpenAI-style streaming
                        // clients (e.g. `vllm bench serve`) blindly
                        // read `choices[0]["delta"]` on every chunk
                        // that has any `choices` entries, so the
                        // last chunk must include `delta` even when
                        // empty. Skipping it triggers
                        // `KeyError: 'delta'` on the client side
                        // and the request is reported as failed
                        // despite returning a 200 with content.
                        let final_chunk = ChatCompletionsResponse {
                            id: request_id.clone(),
                            object: "chat.completion.chunk".to_string(),
                            created: chrono::Utc::now().timestamp() as u64,
                            model: openai_request.model.clone(),
                            choices: vec![ChatChoice {
                                index: 0,
                                message: None,
                                delta: Some(ChatMessage {
                                    role: MessageRole::Assistant,
                                    content: String::new(),
                                    reasoning: None,
                                    name: None,
                                    tool_calls: None,
                                    tool_call_id: None,
                                    function_call: None,
                                }),
                                finish_reason: structured_chat_response
                                    .as_ref()
                                    .and_then(|response| response.finish_reason.clone())
                                    .or_else(|| {
                                        chunk.finish_reason.as_ref().map(finish_reason_to_string)
                                    })
                                    .or(Some("length".to_string())),
                            }],
                            usage: None,
                        };

                        let final_event = Event::default()
                            .json_data(&final_chunk)
                            .unwrap_or_else(|_| Event::default().data("error"));
                        let _ = tx.send(Ok(final_event));
                        if include_stream_usage && usage.is_some() {
                            let usage_chunk = ChatCompletionsResponse {
                                id: request_id.clone(),
                                object: "chat.completion.chunk".to_string(),
                                created: chrono::Utc::now().timestamp() as u64,
                                model: openai_request.model.clone(),
                                choices: vec![],
                                usage,
                            };
                            let usage_event = Event::default()
                                .json_data(&usage_chunk)
                                .unwrap_or_else(|_| Event::default().data("error"));
                            let _ = tx.send(Ok(usage_event));
                        }
                        let _ = tx.send(Ok(Event::default().data("[DONE]")));
                        break;
                    }
                }
                Err(e) => {
                    error!("Stream generation error: {}", e);
                    let _ = tx.send(Ok(openai_error_sse_event(
                        e.to_string(),
                        "internal_server_error",
                        None,
                    )));
                    let _ = tx.send(Ok(Event::default().data("[DONE]")));
                    break;
                }
            }
        }
    });

    let stream = tokio_stream::wrappers::UnboundedReceiverStream::new(rx);
    let sse_stream = Sse::new(stream);

    Ok(sse_stream.into_response())
}

/// Handle non-streaming chat completions
async fn handle_chat_completions_sync(
    state: AppState,
    openai_request: ChatCompletionsRequest,
    inference_request: InferenceRequest,
    session_context: Option<SessionContext>,
) -> std::result::Result<Response, ServerError> {
    info!("Processing non-streaming chat completion");

    let engine = state.llm.clone().ok_or_else(|| {
        ServerError::ServiceUnavailable("LLM engine not loaded; chat unavailable".into())
    })?;
    let request_chat_api = inference_request
        .api_request
        .as_ref()
        .and_then(|api_request| match api_request {
            ferrum_types::ApiRequest::Chat(chat_request) => {
                ferrum_types::chat_api_may_emit_tool_or_function_call(chat_request)
                    .then(|| chat_request.clone())
            }
            _ => None,
        });
    // R1-distill-style templates open the think block inside the prompt.
    let started_in_think = has_unclosed_thinking_block(&inference_request.prompt);
    match engine.infer(inference_request).await {
        Ok(output) => {
            let InferenceResponse {
                text: output_text,
                finish_reason,
                usage,
                api_response,
                ..
            } = output;

            // Post-process the completion text in two passes:
            //   1. Strip a markdown fence when `response_format = json_object`
            //      — JsonModeProcessor's soft biases don't hard-mask the
            //      fence the model wants to emit. Strict json_schema does
            //      not get this cleanup; success must come from hard masking
            //      and final validation, not markdown repair.
            //   2. Strip a trailing user-supplied `stop` sentinel — OpenAI
            //      convention is that stop strings mark a boundary and are
            //      NOT included in the returned completion.
            // Order matters: fence-strip first reveals the actual JSON,
            // then any stop sentinel inside that JSON gets trimmed.
            let after_fence = match &openai_request.response_format {
                Some(rf) if rf.format_type == "json_object" => {
                    strip_markdown_json_fence(&output_text)
                }
                _ => output_text,
            };
            let stop_sequences = openai_request.stop.clone().unwrap_or_default();
            let content = strip_after_stop(&after_fence, &stop_sequences);
            let parsed = if started_in_think {
                parse_reasoning_response_started_in_think(&content)
            } else {
                parse_reasoning_response(&content)
            };
            let visible_content =
                normalize_structured_response_content(&openai_request, &parsed.content);
            let mut message = ChatMessage {
                role: MessageRole::Assistant,
                content: visible_content,
                reasoning: parsed.reasoning.clone(),
                name: None,
                tool_calls: None,
                tool_call_id: None,
                function_call: None,
            };
            let mut openai_finish_reason = finish_reason_to_string(&finish_reason);
            let structured_chat_response = match api_response.as_ref() {
                Some(ferrum_types::ApiResponse::Chat(chat_response)) => Some(chat_response.clone()),
                _ => match request_chat_api.as_ref() {
                    Some(chat_request) => {
                        chat_api_response_from_parsed_generated_text(chat_request, &parsed)
                    }
                    _ => None,
                },
            };
            if let Some(chat_response) = structured_chat_response.as_ref() {
                message = openai_chat_message_from_api(&chat_response.message);
                if message.reasoning.is_none() {
                    message.reasoning = parsed.reasoning.clone();
                }
                if let Some(reason) = &chat_response.finish_reason {
                    openai_finish_reason = reason.clone();
                }
            } else if tool_choice_required(&openai_request) {
                log_required_tool_choice_failure(
                    &openai_request,
                    &parsed.content,
                    parsed.reasoning.as_deref(),
                );
                return Err(ServerError::invalid_request(
                    "model output did not satisfy required tool_choice",
                    Some("tool_choice"),
                ));
            }
            validate_strict_json_schema_response(&openai_request, &message.content)?;
            state
                .cache
                .update_session(session_context, message.clone(), &CachePolicy::current());
            let response = ChatCompletionsResponse {
                id: Uuid::new_v4().to_string(),
                object: "chat.completion".to_string(),
                created: chrono::Utc::now().timestamp() as u64,
                model: openai_request.model,
                choices: vec![ChatChoice {
                    index: 0,
                    message: Some(message),
                    delta: None,
                    finish_reason: Some(openai_finish_reason),
                }],
                usage: Some(openai_usage_from_token_usage(&usage)),
            };

            Ok(Json(response).into_response())
        }
        Err(e) => {
            error!("Generation failed: {}", e);
            Err(server_error_from_ferrum_error(e))
        }
    }
}

/// Convert OpenAI chat request to internal inference request
#[allow(dead_code)]
fn convert_chat_request(
    request: &ChatCompletionsRequest,
) -> ferrum_types::Result<InferenceRequest> {
    convert_chat_request_with_template_model(request, &request.model, None)
}

/// Convert OpenAI chat request to internal inference request.
///
/// `template_model_id` is the loaded model id used for prompt-template family
/// detection. The request `model` field may be an OpenAI-compatible alias such
/// as "ferrum"; using it for template selection can feed a fallback prompt to
/// a Qwen/Llama model.
fn convert_chat_request_with_template_model(
    request: &ChatCompletionsRequest,
    template_model_id: &str,
    model_template: Option<&ModelChatTemplate>,
) -> ferrum_types::Result<InferenceRequest> {
    let no_tools: &[ChatTool] = &[];
    let tools = if tool_choice_none_hides_tools(request.tool_choice.as_ref(), model_template) {
        no_tools
    } else {
        request.tools.as_deref().unwrap_or_default()
    };
    let default_tool_choice =
        default_auto_tool_choice_for_tools(tools, request.tool_choice.as_ref());
    let effective_tool_choice = request
        .tool_choice
        .as_ref()
        .or(default_tool_choice.as_ref());
    let functions = request.functions.as_deref().unwrap_or_default();
    let forced_response_format = forced_tool_choice_response_format(request);
    let requested_response_format = requested_response_format_for_sampling(request)?;
    let render_messages =
        render_messages_with_response_format_instruction(request, forced_response_format.as_ref());
    let chat_template_options = chat_template_options_for_request(request, model_template)?;
    let prompt = if tools.is_empty() && functions.is_empty() {
        render_chat_prompt_with_model_template_options(
            &render_messages,
            template_model_id,
            model_template,
            &chat_template_options,
        )?
    } else {
        render_chat_prompt_with_tools_and_model_template(
            &render_messages,
            template_model_id,
            model_template,
            &chat_template_options,
            tools,
            effective_tool_choice,
            functions,
            request.function_call.as_ref(),
        )?
    };
    let api_chat = api_chat_request(request, effective_tool_choice);
    let may_emit_structured_call = ferrum_types::chat_api_may_emit_tool_or_function_call(&api_chat);
    let mut metadata = HashMap::new();
    metadata.insert(
        "openai_messages".to_string(),
        serde_json::to_value(&request.messages)?,
    );
    if let Some(tools) = &request.tools {
        metadata.insert("openai_tools".to_string(), serde_json::to_value(tools)?);
    }
    if let Some(tool_choice) = effective_tool_choice {
        metadata.insert(
            "openai_tool_choice".to_string(),
            serde_json::to_value(tool_choice)?,
        );
    }
    if let Some(functions) = &request.functions {
        metadata.insert(
            "openai_legacy_functions".to_string(),
            serde_json::to_value(functions)?,
        );
    }
    if let Some(function_call) = &request.function_call {
        metadata.insert(
            "openai_legacy_function_call".to_string(),
            serde_json::to_value(function_call)?,
        );
    }
    if request.ignore_eos.unwrap_or(false) {
        metadata.insert("ferrum_ignore_eos".to_string(), serde_json::json!(true));
    }
    if request.max_completion_tokens.is_none() && request.max_tokens.is_none() {
        metadata.insert(
            DEFAULT_MAX_TOKENS_METADATA_KEY.to_string(),
            serde_json::json!(true),
        );
    }
    if !has_unclosed_thinking_block(&prompt) {
        let mut forbidden = vec![THINK_END_TAG.to_string()];
        if may_emit_structured_call {
            for token_text in INITIAL_STRUCTURED_CALL_FORBIDDEN_TOKEN_TEXTS {
                push_unique_forbidden_token_text(&mut forbidden, token_text);
            }
            if let Some(eos) = model_template.as_ref().and_then(|template| {
                template
                    .eos_token
                    .as_deref()
                    .filter(|token| !token.is_empty())
            }) {
                push_unique_forbidden_token_text(&mut forbidden, eos);
            }
        }
        if chat_template_options.enable_thinking == Some(false) {
            push_unique_forbidden_token_text(&mut forbidden, THINK_START_TAG);
        }
        metadata.insert(
            INITIAL_FORBIDDEN_TOKEN_TEXTS_METADATA_KEY.to_string(),
            serde_json::json!(forbidden),
        );
    }

    Ok(InferenceRequest {
        id: RequestId(Uuid::new_v4()),
        model_id: ModelId(request.model.clone()),
        prompt,
        sampling_params: SamplingParams {
            max_tokens: chat_completion_max_tokens(request) as usize,
            temperature: request.temperature.unwrap_or(DEFAULT_SAMPLING_TEMPERATURE),
            top_p: request.top_p.unwrap_or(DEFAULT_SAMPLING_TOP_P),
            top_k: None, // OpenAI doesn't use top-k
            repetition_penalty: DEFAULT_CHAT_REPETITION_PENALTY,
            presence_penalty: request.presence_penalty.unwrap_or(0.0),
            frequency_penalty: request.frequency_penalty.unwrap_or(0.0),
            stop_sequences: request.stop.clone().unwrap_or_default(),
            seed: request.seed,
            min_p: None,
            tfs: None,
            typical_p: None,
            mirostat: None,
            response_format: forced_response_format
                .clone()
                .or(requested_response_format)
                .or_else(|| inferred_auto_tool_response_format(request))
                .unwrap_or(ferrum_types::ResponseFormat::Text),
        },
        stream: request.stream.unwrap_or(false),
        priority: Priority::Normal, // Default priority
        client_id: None,
        session_id: None,
        created_at: chrono::Utc::now(),
        api_request: Some(ferrum_types::ApiRequest::Chat(api_chat)),
        metadata,
    })
}

fn push_unique_forbidden_token_text(tokens: &mut Vec<String>, token: &str) {
    if !token.is_empty() && !tokens.iter().any(|existing| existing == token) {
        tokens.push(token.to_string());
    }
}

fn default_auto_tool_choice_for_tools(
    tools: &[ChatTool],
    choice: Option<&ToolChoice>,
) -> Option<ToolChoice> {
    if choice.is_none() && !tools.is_empty() {
        Some(ToolChoice::Mode("auto".to_string()))
    } else {
        None
    }
}

fn tool_choice_none(choice: Option<&ToolChoice>) -> bool {
    matches!(choice, Some(ToolChoice::Mode(mode)) if mode.eq_ignore_ascii_case("none"))
}

fn tool_choice_none_hides_tools(
    choice: Option<&ToolChoice>,
    model_template: Option<&ModelChatTemplate>,
) -> bool {
    tool_choice_none(choice)
        && model_template
            .map(|template| template.template.contains("tools_in_user_message"))
            .unwrap_or(false)
}

fn chat_template_options_for_request(
    request: &ChatCompletionsRequest,
    model_template: Option<&ModelChatTemplate>,
) -> ferrum_types::Result<ChatTemplateOptions> {
    let mut options = ChatTemplateOptions::default_for_template(model_template);
    let Some(kwargs) = request.chat_template_kwargs.as_ref() else {
        return Ok(options);
    };
    let Some(value) = kwargs.get("enable_thinking") else {
        return Ok(options);
    };
    let Some(enable_thinking) = value.as_bool() else {
        return Err(Error::invalid_request(
            "chat_template_kwargs.enable_thinking must be a boolean",
        ));
    };
    options.enable_thinking = Some(enable_thinking);
    Ok(options)
}

fn render_messages_with_response_format_instruction(
    request: &ChatCompletionsRequest,
    forced_response_format: Option<&ferrum_types::ResponseFormat>,
) -> Vec<ChatMessage> {
    let Some(instruction) = response_format_prompt_instruction(request, forced_response_format)
    else {
        return request.messages.clone();
    };
    let mut messages = Vec::with_capacity(request.messages.len() + 1);
    messages.push(ChatMessage {
        role: MessageRole::System,
        content: instruction,
        reasoning: None,
        name: None,
        tool_calls: None,
        tool_call_id: None,
        function_call: None,
    });
    messages.extend(request.messages.clone());
    messages
}

fn response_format_prompt_instruction(
    request: &ChatCompletionsRequest,
    _forced_response_format: Option<&ferrum_types::ResponseFormat>,
) -> Option<String> {
    if let Some(format) = request.response_format.as_ref() {
        return match format.format_type.as_str() {
            "json_object" => Some(
                "The response_format requires a single valid JSON object. Output only JSON, with no markdown fences, no explanation, no chain-of-thought, and no extra text."
                    .to_string(),
            ),
            "json_schema" => {
                let schema = format.json_schema.as_ref()?.schema.as_ref()?;
                let schema_text = serde_json::to_string(schema).ok()?;
                Some(format!(
                    "The response_format requires a single valid JSON object satisfying this JSON Schema. Output only JSON, with no markdown fences, no explanation, no chain-of-thought, and no extra text. Schema: {schema_text}"
                ))
            }
            _ => None,
        };
    }
    None
}

fn forced_tool_choice_response_format(
    request: &ChatCompletionsRequest,
) -> Option<ferrum_types::ResponseFormat> {
    let selected_tool = selected_tool_for_forced_tool_choice(request)?;
    let schema = guided_tool_arguments_schema(selected_tool.function.parameters.as_ref())?;
    serde_json::to_string(&schema)
        .ok()
        .map(ferrum_types::ResponseFormat::JsonSchema)
}

fn requested_response_format_for_sampling(
    request: &ChatCompletionsRequest,
) -> ferrum_types::Result<Option<ferrum_types::ResponseFormat>> {
    let Some(format) = request.response_format.as_ref() else {
        return Ok(None);
    };
    match format.format_type.as_str() {
        "json_schema" => {
            let Some(schema) = format.json_schema.as_ref() else {
                return Err(Error::invalid_request(
                    "response_format.json_schema.schema is required",
                ));
            };
            if !schema.strict.unwrap_or(false) {
                return Ok(None);
            }
            let Some(schema_value) = schema.schema.as_ref() else {
                return Err(Error::invalid_request(
                    "response_format.json_schema.schema is required",
                ));
            };
            serde_json::to_string(schema_value)
                .map(|schema| Some(ferrum_types::ResponseFormat::JsonSchema(schema)))
                .map_err(|err| Error::invalid_request(err.to_string()))
        }
        _ => Ok(None),
    }
}

fn inferred_auto_tool_response_format(
    request: &ChatCompletionsRequest,
) -> Option<ferrum_types::ResponseFormat> {
    if !tool_choice_auto_or_omitted(request.tool_choice.as_ref()) {
        return None;
    }
    let tool = single_function_tool(request.tools.as_deref()?)?;
    let prompt = latest_user_text(request)?;
    if !text_mentions_tool(&prompt, &tool.function) {
        return None;
    }
    let schema = serde_json::to_string(&guided_tool_arguments_schema(
        tool.function.parameters.as_ref(),
    )?)
    .ok()?;
    Some(ferrum_types::ResponseFormat::JsonSchema(schema))
}

fn selected_tool_for_forced_tool_choice(request: &ChatCompletionsRequest) -> Option<&ChatTool> {
    match request.tool_choice.as_ref()? {
        ToolChoice::Function {
            tool_type,
            function,
        } if tool_type == "function" => request
            .tools
            .as_ref()?
            .iter()
            .find(|tool| tool.function.name == function.name),
        ToolChoice::Mode(mode) if mode.eq_ignore_ascii_case("required") => {
            request.tools.as_ref()?.first()
        }
        _ => None,
    }
}

fn guided_tool_arguments_schema(
    parameters: Option<&serde_json::Value>,
) -> Option<serde_json::Value> {
    let mut schema = parameters?.clone();
    bound_unconstrained_tool_argument_strings(
        &mut schema,
        DEFAULT_GUIDED_TOOL_ARGUMENT_STRING_MAX_LENGTH,
    );
    Some(schema)
}

fn bound_unconstrained_tool_argument_strings(value: &mut serde_json::Value, default_max: u64) {
    match value {
        serde_json::Value::Object(map) => {
            let is_string = map
                .get("type")
                .and_then(serde_json::Value::as_str)
                .is_some_and(|ty| ty == "string");
            let has_finite_string_shape = map.contains_key("enum") || map.contains_key("maxLength");
            if is_string && !has_finite_string_shape {
                map.insert(
                    "maxLength".to_string(),
                    serde_json::Value::Number(default_max.into()),
                );
            }
            if let Some(properties) = map
                .get_mut("properties")
                .and_then(serde_json::Value::as_object_mut)
            {
                for property in properties.values_mut() {
                    bound_unconstrained_tool_argument_strings(property, default_max);
                }
            }
            if let Some(items) = map.get_mut("items") {
                bound_unconstrained_tool_argument_strings(items, default_max);
            }
        }
        serde_json::Value::Array(items) => {
            for item in items {
                bound_unconstrained_tool_argument_strings(item, default_max);
            }
        }
        _ => {}
    }
}

fn tool_choice_auto_or_omitted(choice: Option<&ToolChoice>) -> bool {
    match choice {
        None => true,
        Some(ToolChoice::Mode(mode)) => mode.eq_ignore_ascii_case("auto"),
        _ => false,
    }
}

fn single_function_tool(tools: &[ChatTool]) -> Option<&ChatTool> {
    let mut function_tools = tools.iter().filter(|tool| tool.tool_type == "function");
    let tool = function_tools.next()?;
    function_tools.next().is_none().then_some(tool)
}

fn latest_user_text(request: &ChatCompletionsRequest) -> Option<String> {
    request
        .messages
        .iter()
        .rev()
        .find(|message| matches!(message.role, MessageRole::User))
        .map(|message| message.content.clone())
        .filter(|content| !content.trim().is_empty())
}

fn text_mentions_tool(text: &str, function: &ChatFunction) -> bool {
    let text_lower = text.to_lowercase();
    for word in ascii_words(&function.name) {
        if text_lower.contains(&word) {
            return true;
        }
    }
    if let Some(description) = &function.description {
        for word in ascii_words(description) {
            if text_lower.contains(&word) {
                return true;
            }
        }
        for bigram in cjk_bigrams(description) {
            if text.contains(&bigram) {
                return true;
            }
        }
    }
    false
}

fn ascii_words(text: &str) -> Vec<String> {
    text.split(|ch: char| !ch.is_ascii_alphanumeric())
        .filter_map(|word| {
            let word = word.to_ascii_lowercase();
            (word.len() >= 3).then_some(word)
        })
        .collect()
}

fn cjk_bigrams(text: &str) -> Vec<String> {
    let chars = text
        .chars()
        .filter(|ch| matches!(*ch as u32, 0x3400..=0x9fff | 0xf900..=0xfaff))
        .collect::<Vec<_>>();
    chars
        .windows(2)
        .map(|window| window.iter().collect::<String>())
        .collect()
}

fn has_unclosed_thinking_block(prompt: &str) -> bool {
    match (prompt.rfind(THINK_START_TAG), prompt.rfind(THINK_END_TAG)) {
        (Some(start), Some(end)) => start > end,
        (Some(_), None) => true,
        _ => false,
    }
}

fn should_defer_reasoning_stream_delta(text: &str) -> bool {
    let candidate = text.trim_start_matches(['\r', '\n']);
    if candidate.is_empty() {
        return true;
    }
    THINK_START_TAG.starts_with(candidate) || THINK_END_TAG.starts_with(candidate)
}

fn stream_text_delta(text: &str, sent_len: &mut usize) -> String {
    if *sent_len <= text.len() && text.is_char_boundary(*sent_len) {
        let delta = text[*sent_len..].to_string();
        *sent_len = text.len();
        return delta;
    }
    *sent_len = text.len();
    String::new()
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ParsedReasoningResponse {
    content: String,
    reasoning: Option<String>,
}

/// Parse generated text whose think block was OPENED BY THE PROMPT —
/// R1-distill-style templates append `<think>\n` to the rendered prompt,
/// so the generation never contains the start tag. Without this, the
/// in-flight thinking streams as content deltas until `</think>` arrives.
fn parse_reasoning_response_started_in_think(text: &str) -> ParsedReasoningResponse {
    if text.contains(THINK_START_TAG) {
        // Model re-opened a think block itself — defer to the normal parse.
        return parse_reasoning_response(text);
    }
    let Some(end) = text.find(THINK_END_TAG) else {
        return ParsedReasoningResponse {
            content: String::new(),
            reasoning: (!text.is_empty()).then(|| text.to_string()),
        };
    };
    let reasoning = text[..end].to_string();
    let content = text[end + THINK_END_TAG.len()..]
        .trim_start_matches(['\r', '\n'])
        .to_string();
    ParsedReasoningResponse {
        content,
        reasoning: (!reasoning.is_empty()).then_some(reasoning),
    }
}

fn parse_reasoning_response(text: &str) -> ParsedReasoningResponse {
    let Some(start) = text.find(THINK_START_TAG) else {
        if let Some(end) = text.find(THINK_END_TAG) {
            let reasoning = text[..end].to_string();
            let content = text[end + THINK_END_TAG.len()..]
                .trim_start_matches(['\r', '\n'])
                .to_string();
            return ParsedReasoningResponse {
                content,
                reasoning: (!reasoning.is_empty()).then_some(reasoning),
            };
        }
        return ParsedReasoningResponse {
            content: text.to_string(),
            reasoning: None,
        };
    };

    let before = &text[..start];
    let after_start = &text[start + THINK_START_TAG.len()..];
    let Some(end) = after_start.find(THINK_END_TAG) else {
        return ParsedReasoningResponse {
            content: before.to_string(),
            reasoning: Some(after_start.to_string()),
        };
    };

    let reasoning = after_start[..end].to_string();
    let after_end = &after_start[end + THINK_END_TAG.len()..];
    let mut content = String::new();
    content.push_str(before);
    content.push_str(after_end.trim_start_matches(['\r', '\n']));

    ParsedReasoningResponse {
        content,
        reasoning: (!reasoning.is_empty()).then_some(reasoning),
    }
}

fn chat_api_response_from_parsed_generated_text(
    chat_request: &ferrum_types::ApiChatRequest,
    parsed: &ParsedReasoningResponse,
) -> Option<ferrum_types::ApiChatResponse> {
    parsed
        .reasoning
        .as_deref()
        .and_then(|reasoning| {
            ferrum_types::chat_api_response_from_generated_text(chat_request, reasoning)
        })
        .or_else(|| {
            ferrum_types::chat_api_response_from_generated_text(chat_request, &parsed.content)
        })
}

fn log_required_tool_choice_failure(
    request: &ChatCompletionsRequest,
    content: &str,
    reasoning: Option<&str>,
) {
    warn!(
        model = %request.model,
        content_len = content.len(),
        content_head = %log_excerpt(content, 512),
        reasoning_len = reasoning.map(str::len).unwrap_or(0),
        reasoning_head = %reasoning.map(|value| log_excerpt(value, 512)).unwrap_or_default(),
        "model output did not satisfy required tool_choice"
    );
}

fn log_excerpt(value: &str, max_chars: usize) -> String {
    let mut out = value.chars().take(max_chars).collect::<String>();
    if value.chars().count() > max_chars {
        out.push_str("...");
    }
    out
}

fn normalize_structured_response_content(
    request: &ChatCompletionsRequest,
    content: &str,
) -> String {
    let Some(response_format) = request.response_format.as_ref() else {
        return content.to_string();
    };
    match response_format.format_type.as_str() {
        "json_object" => extract_json_object_text(content)
            .unwrap_or_else(|| strip_markdown_json_fence(content).to_string()),
        "json_schema"
            if !response_format
                .json_schema
                .as_ref()
                .and_then(|schema| schema.strict)
                .unwrap_or(false) =>
        {
            extract_json_object_text(content)
                .unwrap_or_else(|| strip_markdown_json_fence(content).to_string())
        }
        _ => content.to_string(),
    }
}

fn response_format_is_json_object(request: &ChatCompletionsRequest) -> bool {
    request
        .response_format
        .as_ref()
        .is_some_and(|format| format.format_type == "json_object")
}

fn extract_json_object_text(text: &str) -> Option<String> {
    let text = strip_markdown_json_fence(text.trim());
    if serde_json::from_str::<serde_json::Value>(&text)
        .ok()
        .filter(|value| value.is_object())
        .is_some()
    {
        return Some(text.to_string());
    }

    let start = text.find('{')?;
    let mut depth = 0usize;
    let mut in_string = false;
    let mut escaped = false;
    for (offset, ch) in text[start..].char_indices() {
        if in_string {
            if escaped {
                escaped = false;
            } else if ch == '\\' {
                escaped = true;
            } else if ch == '"' {
                in_string = false;
            }
            continue;
        }
        match ch {
            '"' => in_string = true,
            '{' => depth += 1,
            '}' => {
                depth = depth.saturating_sub(1);
                if depth == 0 {
                    let end = start + offset + ch.len_utf8();
                    let candidate = &text[start..end];
                    if serde_json::from_str::<serde_json::Value>(candidate)
                        .ok()
                        .filter(|value| value.is_object())
                        .is_some()
                    {
                        return Some(candidate.to_string());
                    }
                }
            }
            _ => {}
        }
    }
    None
}

fn api_chat_request(
    request: &ChatCompletionsRequest,
    effective_tool_choice: Option<&ToolChoice>,
) -> ferrum_types::ApiChatRequest {
    ferrum_types::ApiChatRequest {
        messages: request.messages.iter().map(api_chat_message).collect(),
        tools: request
            .tools
            .as_deref()
            .unwrap_or_default()
            .iter()
            .map(api_tool)
            .collect(),
        tool_choice: effective_tool_choice.map(api_tool_choice),
        legacy_functions: request
            .functions
            .as_deref()
            .unwrap_or_default()
            .iter()
            .map(api_function)
            .collect(),
        legacy_function_call: request.function_call.as_ref().map(api_function_call_choice),
        response_format: request.response_format.as_ref().map(api_response_format),
        stream_options: request.stream_options.as_ref().map(|opts| {
            ferrum_types::ApiStreamOptions {
                include_usage: opts.include_usage,
            }
        }),
    }
}

fn chat_completion_max_tokens(request: &ChatCompletionsRequest) -> u32 {
    request
        .max_completion_tokens
        .or(request.max_tokens)
        .unwrap_or(DEFAULT_COMPLETION_MAX_TOKENS)
}

fn api_chat_message(message: &ChatMessage) -> ferrum_types::ApiChatMessage {
    ferrum_types::ApiChatMessage {
        role: match message.role {
            MessageRole::System => ferrum_types::ApiMessageRole::System,
            MessageRole::User => ferrum_types::ApiMessageRole::User,
            MessageRole::Assistant => ferrum_types::ApiMessageRole::Assistant,
            MessageRole::Function => ferrum_types::ApiMessageRole::Function,
            MessageRole::Tool => ferrum_types::ApiMessageRole::Tool,
        },
        content: message.content.clone(),
        name: message.name.clone(),
        tool_calls: message
            .tool_calls
            .as_deref()
            .unwrap_or_default()
            .iter()
            .map(api_tool_call)
            .collect(),
        tool_call_id: message.tool_call_id.clone(),
        function_call: message.function_call.as_ref().map(api_function_call),
    }
}

fn api_tool(tool: &ChatTool) -> ferrum_types::ApiTool {
    ferrum_types::ApiTool {
        tool_type: tool.tool_type.clone(),
        function: api_function(&tool.function),
    }
}

fn api_function(function: &ChatFunction) -> ferrum_types::ApiFunction {
    ferrum_types::ApiFunction {
        name: function.name.clone(),
        description: function.description.clone(),
        parameters: function.parameters.clone(),
        strict: function.strict,
    }
}

fn api_tool_choice(choice: &ToolChoice) -> ferrum_types::ApiToolChoice {
    match choice {
        ToolChoice::Mode(mode) => ferrum_types::ApiToolChoice::Mode(mode.clone()),
        ToolChoice::Function {
            tool_type,
            function,
        } => ferrum_types::ApiToolChoice::Function {
            tool_type: tool_type.clone(),
            function: ferrum_types::ApiToolChoiceFunction {
                name: function.name.clone(),
            },
        },
    }
}

fn api_function_call_choice(choice: &FunctionCallChoice) -> ferrum_types::ApiFunctionCallChoice {
    match choice {
        FunctionCallChoice::Mode(mode) => ferrum_types::ApiFunctionCallChoice::Mode(mode.clone()),
        FunctionCallChoice::Function { name } => {
            ferrum_types::ApiFunctionCallChoice::Function { name: name.clone() }
        }
    }
}

fn api_tool_call(tool_call: &ChatToolCall) -> ferrum_types::ApiToolCall {
    ferrum_types::ApiToolCall {
        id: tool_call.id.clone(),
        tool_type: tool_call.tool_type.clone(),
        function: api_function_call(&tool_call.function),
    }
}

fn api_function_call(function_call: &ChatFunctionCall) -> ferrum_types::ApiFunctionCall {
    ferrum_types::ApiFunctionCall {
        name: function_call.name.clone(),
        arguments: function_call.arguments.clone(),
    }
}

fn openai_chat_message_from_api(message: &ferrum_types::ApiChatMessage) -> ChatMessage {
    ChatMessage {
        role: openai_message_role_from_api(message.role),
        content: message.content.clone(),
        reasoning: None,
        name: message.name.clone(),
        tool_calls: if message.tool_calls.is_empty() {
            None
        } else {
            Some(
                message
                    .tool_calls
                    .iter()
                    .map(openai_tool_call_from_api)
                    .collect(),
            )
        },
        tool_call_id: message.tool_call_id.clone(),
        function_call: message
            .function_call
            .as_ref()
            .map(openai_function_call_from_api),
    }
}

fn openai_message_role_from_api(role: ferrum_types::ApiMessageRole) -> MessageRole {
    match role {
        ferrum_types::ApiMessageRole::System => MessageRole::System,
        ferrum_types::ApiMessageRole::User => MessageRole::User,
        ferrum_types::ApiMessageRole::Assistant => MessageRole::Assistant,
        ferrum_types::ApiMessageRole::Function => MessageRole::Function,
        ferrum_types::ApiMessageRole::Tool => MessageRole::Tool,
    }
}

fn openai_tool_call_from_api(tool_call: &ferrum_types::ApiToolCall) -> ChatToolCall {
    ChatToolCall {
        index: None,
        id: tool_call.id.clone(),
        tool_type: tool_call.tool_type.clone(),
        function: openai_function_call_from_api(&tool_call.function),
    }
}

fn openai_tool_call_delta_from_api(
    index: usize,
    tool_call: &ferrum_types::ApiToolCall,
) -> ChatToolCall {
    ChatToolCall {
        index: Some(usize_to_u32_saturating(index)),
        id: tool_call.id.clone(),
        tool_type: tool_call.tool_type.clone(),
        function: openai_function_call_from_api(&tool_call.function),
    }
}

fn openai_chat_delta_from_api(message: &ferrum_types::ApiChatMessage) -> ChatMessage {
    let mut delta = openai_chat_message_from_api(message);
    if !message.tool_calls.is_empty() {
        delta.tool_calls = Some(
            message
                .tool_calls
                .iter()
                .enumerate()
                .map(|(index, call)| openai_tool_call_delta_from_api(index, call))
                .collect(),
        );
    }
    delta
}

fn openai_function_call_from_api(
    function_call: &ferrum_types::ApiFunctionCall,
) -> ChatFunctionCall {
    ChatFunctionCall {
        name: function_call.name.clone(),
        arguments: function_call.arguments.clone(),
    }
}

fn api_response_format(format: &OpenAiResponseFormat) -> ferrum_types::ApiResponseFormat {
    ferrum_types::ApiResponseFormat {
        format_type: format.format_type.clone(),
        json_schema: format
            .json_schema
            .as_ref()
            .map(|schema| ferrum_types::ApiJsonSchema {
                name: schema.name.clone(),
                schema: schema.schema.clone().unwrap_or(serde_json::Value::Null),
                strict: schema.strict,
            }),
    }
}

fn validate_chat_request(request: &ChatCompletionsRequest) -> std::result::Result<(), ServerError> {
    if request.messages.is_empty() {
        return Err(ServerError::invalid_request(
            "messages array must not be empty",
            Some("messages"),
        ));
    }

    if let Some(n) = request.n {
        if n != 1 {
            return Err(ServerError::unsupported_feature(
                "only n=1 is supported for chat completions",
                Some("n"),
            ));
        }
    }

    if request
        .logit_bias
        .as_ref()
        .is_some_and(|bias| !bias.is_empty())
    {
        return Err(ServerError::unsupported_feature(
            "logit_bias is not supported",
            Some("logit_bias"),
        ));
    }
    if request.logprobs.unwrap_or(false) {
        return Err(ServerError::unsupported_feature(
            "logprobs is not supported",
            Some("logprobs"),
        ));
    }
    if request.top_logprobs.unwrap_or(0) > 0 {
        return Err(ServerError::unsupported_feature(
            "top_logprobs is not supported",
            Some("top_logprobs"),
        ));
    }

    if request.stream_options.is_some() && !request.stream.unwrap_or(false) {
        return Err(ServerError::invalid_request(
            "stream_options is only valid when stream=true",
            Some("stream_options"),
        ));
    }
    ensure_response_format_supported(request)?;

    if let Some(tools) = &request.tools {
        for tool in tools {
            if tool.tool_type != "function" {
                return Err(ServerError::unsupported_feature(
                    "only function tools are supported",
                    Some("tools"),
                ));
            }
        }
    }

    if let Some(choice) = &request.tool_choice {
        match choice {
            ToolChoice::Mode(mode)
                if mode.eq_ignore_ascii_case("auto") || mode.eq_ignore_ascii_case("none") => {}
            ToolChoice::Mode(mode) if mode.eq_ignore_ascii_case("required") => {
                if request.tools.as_deref().unwrap_or_default().is_empty() {
                    return Err(ServerError::invalid_request(
                        "tool_choice=required requires at least one function tool",
                        Some("tool_choice"),
                    ));
                }
            }
            ToolChoice::Mode(_) => {
                return Err(ServerError::unsupported_feature(
                    "unsupported tool_choice mode",
                    Some("tool_choice"),
                ));
            }
            ToolChoice::Function {
                tool_type,
                function,
            } => {
                if tool_type != "function" {
                    return Err(ServerError::unsupported_feature(
                        "only function tool_choice is supported",
                        Some("tool_choice"),
                    ));
                }
                let declared = request
                    .tools
                    .as_deref()
                    .unwrap_or_default()
                    .iter()
                    .any(|tool| tool.function.name == function.name);
                if !declared {
                    return Err(ServerError::invalid_request(
                        "tool_choice selects a function that is not declared in tools",
                        Some("tool_choice"),
                    ));
                }
            }
        }
    }

    if let Some(choice) = &request.function_call {
        match choice {
            FunctionCallChoice::Mode(mode)
                if mode.eq_ignore_ascii_case("auto") || mode.eq_ignore_ascii_case("none") => {}
            FunctionCallChoice::Mode(_) => {
                return Err(ServerError::unsupported_feature(
                    "unsupported function_call mode",
                    Some("function_call"),
                ));
            }
            FunctionCallChoice::Function { name } => {
                let declared = request
                    .functions
                    .as_deref()
                    .unwrap_or_default()
                    .iter()
                    .any(|function| function.name == *name);
                if !declared {
                    return Err(ServerError::invalid_request(
                        "function_call selects a function that is not declared in functions",
                        Some("function_call"),
                    ));
                }
            }
        }
    }

    Ok(())
}

fn tool_choice_required(request: &ChatCompletionsRequest) -> bool {
    match request.tool_choice.as_ref() {
        Some(ToolChoice::Mode(mode)) if mode.eq_ignore_ascii_case("required") => true,
        Some(ToolChoice::Function {
            tool_type,
            function,
        }) => {
            tool_type == "function"
                && request
                    .tools
                    .as_deref()
                    .unwrap_or_default()
                    .iter()
                    .any(|tool| tool.function.name == function.name)
        }
        _ => false,
    }
}

fn openai_usage_from_token_usage(usage: &TokenUsage) -> Usage {
    let prompt_tokens = usize_to_u32_saturating(usage.prompt_tokens);
    let completion_tokens = usize_to_u32_saturating(usage.completion_tokens);
    let total_tokens = usize_to_u32_saturating(usage.total_tokens);
    Usage {
        prompt_tokens,
        completion_tokens,
        total_tokens,
    }
}

fn usize_to_u32_saturating(value: usize) -> u32 {
    u32::try_from(value).unwrap_or(u32::MAX)
}

fn ensure_response_format_supported(
    request: &ChatCompletionsRequest,
) -> std::result::Result<(), ServerError> {
    if let Some(rf) = &request.response_format {
        match rf.format_type.as_str() {
            "text" | "json_object" => {}
            "json_schema" => {
                let Some(schema_json) = strict_json_schema_string(request)? else {
                    if rf.json_schema.is_none() {
                        return Err(ServerError::invalid_request(
                            "response_format.json_schema.schema is required",
                            Some("response_format.json_schema"),
                        ));
                    }
                    return Ok(());
                };
                ferrum_sampler::schema_to_regex::schema_to_regex(&schema_json).map_err(|e| {
                    ServerError::unsupported_feature(
                        format!("unsupported strict json_schema: {e}"),
                        Some("response_format.json_schema"),
                    )
                })?;
            }
            _ => {
                return Err(ServerError::invalid_request(
                    "unsupported response_format.type",
                    Some("response_format.type"),
                ));
            }
        }
    }
    Ok(())
}

fn strict_json_schema_string(
    request: &ChatCompletionsRequest,
) -> std::result::Result<Option<String>, ServerError> {
    let Some(rf) = &request.response_format else {
        return Ok(None);
    };
    if rf.format_type != "json_schema" {
        return Ok(None);
    }
    let Some(schema) = &rf.json_schema else {
        return Err(ServerError::invalid_request(
            "response_format.json_schema.schema is required",
            Some("response_format.json_schema"),
        ));
    };
    let Some(schema_value) = schema.schema.as_ref() else {
        return Err(ServerError::invalid_request(
            "response_format.json_schema.schema is required",
            Some("response_format.json_schema"),
        ));
    };
    if !schema.strict.unwrap_or(false) {
        return Ok(None);
    }
    serde_json::to_string(schema_value).map(Some).map_err(|e| {
        ServerError::invalid_request(e.to_string(), Some("response_format.json_schema"))
    })
}

fn validate_strict_json_schema_response(
    request: &ChatCompletionsRequest,
    content: &str,
) -> std::result::Result<(), ServerError> {
    let Some(schema_json) = strict_json_schema_string(request)? else {
        return Ok(());
    };
    let _parsed_json: serde_json::Value = serde_json::from_str(content).map_err(|e| {
        ServerError::InternalError(format!(
            "model output did not satisfy response_format.json_schema.strict: invalid JSON: {e}"
        ))
    })?;
    let pattern = ferrum_sampler::schema_to_regex::schema_to_regex(&schema_json).map_err(|e| {
        ServerError::InternalError(format!(
            "strict json_schema translator failed after validation: {e}"
        ))
    })?;
    let regex = regex_lite::Regex::new(&format!("^(?:{pattern})$")).map_err(|e| {
        ServerError::InternalError(format!("strict json_schema validator build failed: {e}"))
    })?;
    if !regex.is_match(content) {
        return Err(ServerError::InternalError(
            "model output did not satisfy response_format.json_schema.strict".to_string(),
        ));
    }
    Ok(())
}

fn strict_stream_validation_error_message(error: ServerError) -> String {
    match error {
        ServerError::InternalError(message)
        | ServerError::NotImplemented(message)
        | ServerError::ServiceUnavailable(message)
        | ServerError::InvalidRequest { message, .. }
        | ServerError::UnsupportedFeature { message, .. } => message,
    }
}

fn server_error_from_ferrum_error(error: Error) -> ServerError {
    match error {
        Error::RequestValidation { message } => ServerError::invalid_request(message, None),
        Error::ResourceExhausted { message } => ServerError::ServiceUnavailable(message),
        other => ServerError::InternalError(other.to_string()),
    }
}

fn stream_error_payload(
    message: impl Into<String>,
    error_type: &str,
    param: Option<&str>,
) -> OpenAiError {
    OpenAiError {
        error: OpenAiErrorDetail {
            message: message.into(),
            error_type: error_type.to_string(),
            param: param.map(str::to_string),
            code: None,
        },
    }
}

fn openai_error_sse_event(
    message: impl Into<String>,
    error_type: &str,
    param: Option<&str>,
) -> Event {
    Event::default()
        .json_data(&stream_error_payload(message, error_type, param))
        .unwrap_or_else(|_| Event::default().data("error"))
}

fn convert_completion_request(request: &CompletionsRequest) -> InferenceRequest {
    let prompt = request
        .prompt
        .as_text()
        .expect("completion prompt validated before conversion");
    InferenceRequest {
        id: RequestId(Uuid::new_v4()),
        model_id: ModelId(request.model.clone()),
        prompt: prompt.to_string(),
        sampling_params: SamplingParams {
            max_tokens: request.max_tokens.unwrap_or(DEFAULT_COMPLETION_MAX_TOKENS) as usize,
            temperature: request.temperature.unwrap_or(DEFAULT_SAMPLING_TEMPERATURE),
            top_p: request.top_p.unwrap_or(DEFAULT_SAMPLING_TOP_P),
            top_k: None,
            repetition_penalty: 1.0,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            stop_sequences: request.stop.clone().unwrap_or_default(),
            seed: None,
            min_p: None,
            tfs: None,
            typical_p: None,
            mirostat: None,
            response_format: ferrum_types::ResponseFormat::Text,
        },
        stream: request.stream.unwrap_or(false),
        priority: Priority::Normal,
        client_id: None,
        session_id: None,
        created_at: chrono::Utc::now(),
        api_request: Some(ferrum_types::ApiRequest::Completion(
            ferrum_types::ApiCompletionRequest {
                prompt: prompt.to_string(),
                response_format: None,
            },
        )),
        metadata: if request.max_tokens.is_none() {
            HashMap::from([(
                DEFAULT_MAX_TOKENS_METADATA_KEY.to_string(),
                serde_json::json!(true),
            )])
        } else {
            HashMap::new()
        },
    }
}

fn apply_lora_resolution(
    inference_request: &mut InferenceRequest,
    resolution: Option<&LoraModelResolution>,
) {
    let Some(resolution) = resolution else {
        return;
    };
    if let Some(adapter) = &resolution.adapter {
        inference_request.model_id = ModelId(resolution.base_model_id.clone());
        inference_request.metadata.insert(
            "ferrum_lora_adapter".to_string(),
            serde_json::json!(adapter.name),
        );
        inference_request.metadata.insert(
            "ferrum_lora_model_id".to_string(),
            serde_json::json!(adapter.model_id),
        );
        inference_request.metadata.insert(
            "ferrum_lora_path".to_string(),
            serde_json::json!(adapter.path),
        );
    }
}

async fn handle_completions_sync(
    state: AppState,
    openai_request: CompletionsRequest,
    inference_request: InferenceRequest,
) -> std::result::Result<Response, ServerError> {
    let engine = state.llm.clone().ok_or_else(|| {
        ServerError::ServiceUnavailable("LLM engine not loaded; completions unavailable".into())
    })?;
    match engine.infer(inference_request).await {
        Ok(output) => {
            let InferenceResponse {
                text: output_text,
                finish_reason,
                usage,
                api_response,
                ..
            } = output;
            let stop_sequences = openai_request.stop.clone().unwrap_or_default();
            let mut text = strip_after_stop(&output_text, &stop_sequences);
            let mut openai_finish_reason = finish_reason_to_string(&finish_reason);
            if let Some(ferrum_types::ApiResponse::Completion(completion_response)) =
                api_response.as_ref()
            {
                text = strip_after_stop(&completion_response.text, &stop_sequences);
                if let Some(reason) = &completion_response.finish_reason {
                    openai_finish_reason = reason.clone();
                }
            }
            let response = CompletionsResponse {
                id: Uuid::new_v4().to_string(),
                object: "text_completion".to_string(),
                created: chrono::Utc::now().timestamp() as u64,
                model: openai_request.model,
                choices: vec![CompletionChoice {
                    text,
                    index: 0,
                    finish_reason: Some(openai_finish_reason),
                }],
                usage: Some(openai_usage_from_token_usage(&usage)),
            };
            Ok(Json(response).into_response())
        }
        Err(e) => {
            error!("Completion generation failed: {}", e);
            Err(ServerError::InternalError(e.to_string()))
        }
    }
}

async fn handle_completions_stream(
    state: AppState,
    openai_request: CompletionsRequest,
    inference_request: InferenceRequest,
) -> std::result::Result<Response, ServerError> {
    let (tx, rx) = mpsc::unbounded_channel::<std::result::Result<Event, axum::Error>>();
    let engine = state.llm.clone().ok_or_else(|| {
        ServerError::ServiceUnavailable("LLM engine not loaded; completions unavailable".into())
    })?;
    let request_id = Uuid::new_v4().to_string();

    tokio::spawn(async move {
        match engine.infer_stream(inference_request).await {
            Ok(mut stream) => {
                while let Some(result) = stream.next().await {
                    match result {
                        Ok(chunk) => {
                            let response_chunk = CompletionsResponse {
                                id: request_id.clone(),
                                object: "text_completion".to_string(),
                                created: chrono::Utc::now().timestamp() as u64,
                                model: openai_request.model.clone(),
                                choices: vec![CompletionChoice {
                                    text: chunk.text.clone(),
                                    index: 0,
                                    finish_reason: chunk
                                        .finish_reason
                                        .as_ref()
                                        .map(finish_reason_to_string),
                                }],
                                usage: None,
                            };
                            let event = Event::default()
                                .json_data(&response_chunk)
                                .unwrap_or_else(|_| Event::default().data("error"));
                            if tx.send(Ok(event)).is_err() {
                                break;
                            }
                            if chunk.finish_reason.is_some() {
                                if let Some(usage) =
                                    chunk.usage.as_ref().map(openai_usage_from_token_usage)
                                {
                                    let final_chunk = CompletionsResponse {
                                        id: request_id.clone(),
                                        object: "text_completion".to_string(),
                                        created: chrono::Utc::now().timestamp() as u64,
                                        model: openai_request.model.clone(),
                                        choices: vec![],
                                        usage: Some(usage),
                                    };
                                    let event = Event::default()
                                        .json_data(&final_chunk)
                                        .unwrap_or_else(|_| Event::default().data("error"));
                                    let _ = tx.send(Ok(event));
                                }
                                let _ = tx.send(Ok(Event::default().data("[DONE]")));
                                break;
                            }
                        }
                        Err(e) => {
                            error!("Completion stream generation error: {}", e);
                            let _ = tx.send(Ok(openai_error_sse_event(
                                e.to_string(),
                                "internal_server_error",
                                None,
                            )));
                            let _ = tx.send(Ok(Event::default().data("[DONE]")));
                            break;
                        }
                    }
                }
            }
            Err(e) => {
                error!("Failed to start completion stream: {}", e);
                let _ = tx.send(Ok(openai_error_sse_event(
                    e.to_string(),
                    "internal_server_error",
                    None,
                )));
                let _ = tx.send(Ok(Event::default().data("[DONE]")));
            }
        }
    });

    let stream = tokio_stream::wrappers::UnboundedReceiverStream::new(rx);
    Ok(Sse::new(stream).into_response())
}

/// Other handlers
async fn completions_handler(
    State(state): State<AppState>,
    request: std::result::Result<Json<CompletionsRequest>, JsonRejection>,
) -> std::result::Result<Response, ServerError> {
    let Json(request) = request.map_err(|e| {
        ServerError::invalid_request(format!("invalid completions request: {e}"), None)
    })?;
    validate_completion_request(&request)?;
    let loaded_models = state.status().await.loaded_models;
    let lora_resolution = state
        .lora_registry
        .resolve(&request.model, &loaded_models)?;
    let mut inference_request = convert_completion_request(&request);
    apply_lora_resolution(&mut inference_request, lora_resolution.as_ref());
    if request.stream.unwrap_or(false) {
        handle_completions_stream(state, request, inference_request).await
    } else {
        handle_completions_sync(state, request, inference_request).await
    }
}

fn validate_completion_request(
    request: &CompletionsRequest,
) -> std::result::Result<(), ServerError> {
    if request.prompt.as_text().is_none() {
        return Err(ServerError::invalid_request(
            "only string prompt is supported for completions",
            Some("prompt"),
        ));
    }
    if let Some(n) = request.n {
        if n != 1 {
            return Err(ServerError::unsupported_feature(
                "only n=1 is supported for completions",
                Some("n"),
            ));
        }
    }
    if request.logprobs.is_some() {
        return Err(ServerError::unsupported_feature(
            "logprobs is not supported for completions",
            Some("logprobs"),
        ));
    }
    if request
        .logit_bias
        .as_ref()
        .is_some_and(|bias| !bias.is_empty())
    {
        return Err(ServerError::unsupported_feature(
            "logit_bias is not supported",
            Some("logit_bias"),
        ));
    }
    Ok(())
}

/// Embeddings handler — text and image embedding via OpenAI-compatible API.
async fn embeddings_handler(
    State(state): State<AppState>,
    request: std::result::Result<Json<EmbeddingsRequest>, JsonRejection>,
) -> std::result::Result<Response, ServerError> {
    let Json(request) = request.map_err(|e| {
        ServerError::invalid_request(format!("invalid embeddings request: {e}"), None)
    })?;

    let span = span!(Level::INFO, "embeddings", model = %request.model);
    let _enter = span.enter();

    validate_embeddings_request(&request)?;

    // Flatten input into individual items
    let items: Vec<EmbeddingItem> = match request.input {
        EmbeddingInput::Single(text) => vec![EmbeddingItem {
            text: Some(text),
            image: None,
        }],
        EmbeddingInput::Batch(texts) => texts
            .into_iter()
            .map(|t| EmbeddingItem {
                text: Some(t),
                image: None,
            })
            .collect(),
        EmbeddingInput::SingleObject(item) => vec![item],
        EmbeddingInput::BatchObjects(items) => items,
    };

    if items.is_empty() {
        return Err(ServerError::invalid_request(
            "input must not be empty",
            Some("input"),
        ));
    }

    let mut data = Vec::with_capacity(items.len());
    let mut total_tokens = 0u32;

    let engine = state.embed.as_ref().ok_or_else(|| {
        ServerError::NotImplemented("Embed engine not loaded; embeddings unavailable".into())
    })?;
    for (idx, item) in items.iter().enumerate() {
        let embedding = if let Some(ref image) = item.image {
            engine
                .embed_image(image)
                .await
                .map_err(|e| ServerError::InternalError(format!("embed_image: {e}")))?
        } else if let Some(ref text) = item.text {
            total_tokens += text.len() as u32;
            engine
                .embed_text(text)
                .await
                .map_err(|e| ServerError::InternalError(format!("embed_text: {e}")))?
        } else {
            return Err(ServerError::invalid_request(
                "each input item must have either text or image",
                Some("input"),
            ));
        };

        data.push(EmbeddingData {
            object: "embedding".to_string(),
            embedding,
            index: idx,
        });
    }

    let response = EmbeddingsResponse {
        object: "list".to_string(),
        data,
        model: request.model,
        usage: EmbeddingUsage {
            prompt_tokens: total_tokens,
            total_tokens,
        },
    };

    Ok(Json(response).into_response())
}

fn validate_embeddings_request(
    request: &EmbeddingsRequest,
) -> std::result::Result<(), ServerError> {
    if let Some(format) = request.encoding_format.as_deref() {
        if !format.eq_ignore_ascii_case("float") {
            return Err(ServerError::unsupported_feature(
                "only encoding_format=float is supported for embeddings",
                Some("encoding_format"),
            ));
        }
    }
    Ok(())
}

/// Audio transcription handler (OpenAI-compatible multipart form).
async fn transcriptions_handler(
    State(state): State<AppState>,
    multipart: std::result::Result<axum::extract::Multipart, MultipartRejection>,
) -> std::result::Result<Response, ServerError> {
    let mut multipart = multipart.map_err(|e| {
        ServerError::invalid_request(format!("invalid transcriptions request: {e}"), None)
    })?;

    let span = span!(Level::INFO, "transcription");
    let _enter = span.enter();

    let mut file_data: Option<Vec<u8>> = None;
    let mut language: Option<String> = None;
    let mut response_format: Option<String> = None;

    while let Some(field) = multipart
        .next_field()
        .await
        .map_err(|e| ServerError::invalid_request(format!("multipart: {e}"), None))?
    {
        let name = field.name().unwrap_or("").to_string();
        match name.as_str() {
            "file" => {
                file_data = Some(
                    field
                        .bytes()
                        .await
                        .map_err(|e| {
                            ServerError::invalid_request(format!("read file: {e}"), Some("file"))
                        })?
                        .to_vec(),
                );
            }
            "language" => {
                language = field.text().await.ok().filter(|s| !s.is_empty());
            }
            "response_format" => {
                response_format = field.text().await.ok().filter(|s| !s.is_empty());
            }
            _ => {} // ignore model and other optional multipart fields for now
        }
    }

    validate_transcription_response_format(response_format.as_deref())?;

    let data = file_data
        .ok_or_else(|| ServerError::invalid_request("missing file field", Some("file")))?;

    let engine = state.transcribe.as_ref().ok_or_else(|| {
        ServerError::NotImplemented("Transcribe engine not loaded; ASR unavailable".into())
    })?;
    let text = engine
        .transcribe_bytes(&data, language.as_deref())
        .await
        .map_err(|e| ServerError::InternalError(format!("transcribe: {e}")))?;

    Ok(Json(TranscriptionResponse { text }).into_response())
}

fn validate_transcription_response_format(
    response_format: Option<&str>,
) -> std::result::Result<(), ServerError> {
    if let Some(format) = response_format {
        if !format.eq_ignore_ascii_case("json") {
            return Err(ServerError::unsupported_feature(
                "only response_format=json is supported for transcriptions",
                Some("response_format"),
            ));
        }
    }
    Ok(())
}

/// TTS speech synthesis handler (OpenAI-compatible /v1/audio/speech)
async fn speech_handler(
    State(state): State<AppState>,
    request: std::result::Result<Json<SpeechRequest>, JsonRejection>,
) -> std::result::Result<Response, ServerError> {
    let Json(request) = request
        .map_err(|e| ServerError::invalid_request(format!("invalid speech request: {e}"), None))?;

    let response_format = speech_output_format(&request)?;

    let span = span!(Level::INFO, "speech");
    let _guard = span.enter();

    let language = if request.language.is_empty() || request.language == "auto" {
        None
    } else {
        Some(request.language.as_str())
    };

    let chunk_frames = 10usize;
    let tts = state.tts.as_ref().ok_or_else(|| {
        ServerError::NotImplemented("TTS engine not loaded; speech unavailable".into())
    })?;
    let sample_rate = tts.tts_sample_rate();

    if request.stream {
        // Streaming: chunked transfer encoding with WAV audio
        let (tx, rx) =
            mpsc::unbounded_channel::<std::result::Result<axum::body::Bytes, std::io::Error>>();

        let engine = tts.clone();
        let text = request.input.clone();
        let lang = request.language.clone();

        tokio::task::spawn_blocking(move || {
            let lang_opt = if lang.is_empty() || lang == "auto" {
                None
            } else {
                Some(lang.as_str())
            };
            let rt = tokio::runtime::Handle::current();

            match rt.block_on(engine.synthesize_speech(&text, lang_opt, chunk_frames)) {
                Ok(chunks) => {
                    for chunk in &chunks {
                        let audio_bytes = encode_speech_audio(chunk, sample_rate, response_format);
                        let _ = tx.send(Ok(axum::body::Bytes::from(audio_bytes)));
                    }
                }
                Err(e) => {
                    error!("TTS error: {e}");
                }
            }
        });

        let stream = tokio_stream::wrappers::UnboundedReceiverStream::new(rx);
        let body = axum::body::Body::from_stream(stream);
        Ok(Response::builder()
            .status(200)
            .header("content-type", speech_content_type(response_format))
            .header("transfer-encoding", "chunked")
            .body(body)
            .unwrap())
    } else {
        // Non-streaming: return complete WAV
        let chunks = tts
            .synthesize_speech(&request.input, language, chunk_frames)
            .await
            .map_err(|e| ServerError::InternalError(format!("TTS: {e}")))?;

        let all_samples: Vec<f32> = chunks.into_iter().flatten().collect();
        let audio_bytes = encode_speech_audio(&all_samples, sample_rate, response_format);

        Ok(Response::builder()
            .status(200)
            .header("content-type", speech_content_type(response_format))
            .header("content-length", audio_bytes.len().to_string())
            .body(axum::body::Body::from(audio_bytes))
            .unwrap())
    }
}

#[derive(Clone, Copy)]
enum SpeechOutputFormat {
    Wav,
    Pcm,
}

fn speech_output_format(
    request: &SpeechRequest,
) -> std::result::Result<SpeechOutputFormat, ServerError> {
    if request.response_format.eq_ignore_ascii_case("wav") {
        Ok(SpeechOutputFormat::Wav)
    } else if request.response_format.eq_ignore_ascii_case("pcm") {
        Ok(SpeechOutputFormat::Pcm)
    } else {
        Err(ServerError::unsupported_feature(
            "only response_format=wav or response_format=pcm is supported for speech",
            Some("response_format"),
        ))
    }
}

fn speech_content_type(format: SpeechOutputFormat) -> &'static str {
    match format {
        SpeechOutputFormat::Wav => "audio/wav",
        SpeechOutputFormat::Pcm => "audio/pcm",
    }
}

fn encode_speech_audio(samples: &[f32], sample_rate: u32, format: SpeechOutputFormat) -> Vec<u8> {
    match format {
        SpeechOutputFormat::Wav => pcm_to_wav_bytes(samples, sample_rate),
        SpeechOutputFormat::Pcm => pcm_to_s16le_bytes(samples),
    }
}

/// Convert PCM f32 samples to WAV bytes (16-bit, mono).
fn pcm_to_wav_bytes(samples: &[f32], sample_rate: u32) -> Vec<u8> {
    let num_samples = samples.len();
    let data_size = num_samples * 2; // 16-bit = 2 bytes per sample
    let file_size = 44 + data_size;

    let mut buf = Vec::with_capacity(file_size);
    // RIFF header
    buf.extend_from_slice(b"RIFF");
    buf.extend_from_slice(&((file_size - 8) as u32).to_le_bytes());
    buf.extend_from_slice(b"WAVE");
    // fmt chunk
    buf.extend_from_slice(b"fmt ");
    buf.extend_from_slice(&16u32.to_le_bytes()); // chunk size
    buf.extend_from_slice(&1u16.to_le_bytes()); // PCM
    buf.extend_from_slice(&1u16.to_le_bytes()); // mono
    buf.extend_from_slice(&sample_rate.to_le_bytes());
    buf.extend_from_slice(&(sample_rate * 2).to_le_bytes()); // byte rate
    buf.extend_from_slice(&2u16.to_le_bytes()); // block align
    buf.extend_from_slice(&16u16.to_le_bytes()); // bits per sample
                                                 // data chunk
    buf.extend_from_slice(b"data");
    buf.extend_from_slice(&(data_size as u32).to_le_bytes());
    buf.extend_from_slice(&pcm_to_s16le_bytes(samples));
    buf
}

fn pcm_to_s16le_bytes(samples: &[f32]) -> Vec<u8> {
    let mut buf = Vec::with_capacity(samples.len() * 2);
    for &s in samples {
        let i16_val = (s.clamp(-1.0, 1.0) * 32767.0) as i16;
        buf.extend_from_slice(&i16_val.to_le_bytes());
    }
    buf
}

async fn models_handler(
    State(state): State<AppState>,
) -> std::result::Result<Response, ServerError> {
    let status = state.status().await;
    let now = chrono::Utc::now().timestamp() as u64;
    let mut data: Vec<_> = status
        .loaded_models
        .into_iter()
        .map(|model_id| crate::openai::ModelInfo {
            id: model_id.to_string(),
            object: "model".to_string(),
            created: now,
            owned_by: "ferrum".to_string(),
            permission: vec![],
            root: None,
            parent: None,
        })
        .collect();
    data.extend(state.lora_registry.adapter_models().iter().map(|adapter| {
        crate::openai::ModelInfo {
            id: adapter.model_id.clone(),
            object: "model".to_string(),
            created: now,
            owned_by: "ferrum".to_string(),
            permission: vec![],
            root: state.lora_registry.base_model_id.as_ref().cloned(),
            parent: state.lora_registry.base_model_id.as_ref().cloned(),
        }
    }));

    let models = ModelListResponse {
        object: "list".to_string(),
        data,
    };

    Ok(Json(models).into_response())
}

async fn health_handler(
    State(state): State<AppState>,
) -> std::result::Result<Response, ServerError> {
    let engine_status = state.status().await;
    let scheduler_metrics = state.metrics();
    let runtime_config = RuntimeConfigSnapshot::capture_current();
    let cache_policy = CachePolicy::current();
    let engine_cache = state
        .llm
        .as_ref()
        .and_then(|engine| engine.cache_metrics_snapshot());
    let engine_lora = state
        .llm
        .as_ref()
        .and_then(|engine| engine.lora_metrics_snapshot());
    let auto_config = auto_config_health_value(state.auto_config.as_ref());
    let admission = admission_health_json(&engine_status, &scheduler_metrics, &auto_config);

    let health = serde_json::json!({
        "status": "healthy",
        "timestamp": chrono::Utc::now().to_rfc3339(),
        "version": env!("CARGO_PKG_VERSION"),
        "engine": {
            "active_requests": engine_status.active_requests,
            "queued_requests": engine_status.queued_requests,
        },
        "scheduler": {
            "total_requests": scheduler_metrics.total_requests,
            "successful_requests": scheduler_metrics.successful_requests,
            "failed_requests": scheduler_metrics.failed_requests,
            "throughput_rps": scheduler_metrics.throughput_rps,
            "avg_wait_time_ms": scheduler_metrics.queue_metrics.avg_queue_wait_time_ms,
            "scheduling_time_ms": scheduler_metrics.performance_breakdown.scheduling_time_ms,
            "model_execution_time_ms": scheduler_metrics
                .performance_breakdown
                .model_execution_time_ms,
            "iteration_lock_wait_time_ms": scheduler_metrics
                .performance_breakdown
                .other_overhead_time_ms,
        },
        "config": runtime_config,
        "auto_config": auto_config,
        "admission": admission,
        "cache": state.cache.health_json(&cache_policy, engine_cache.as_ref()),
        "lora": engine_lora.unwrap_or_else(|| serde_json::json!({
            "enabled": state.lora_registry.is_enabled(),
            "adapter_count": state.lora_registry.adapter_models().len() as u64,
            "active_cache_bindings": 0u64,
            "projection_applications": 0u64,
            "position": "startup-routing",
            "source": "server-lora-registry",
        })),
    });

    Ok(Json(health).into_response())
}

/// Prometheus metrics endpoint — returns metrics in Prometheus text format.
async fn metrics_handler(
    State(state): State<AppState>,
) -> std::result::Result<Response, ServerError> {
    let mut body = match PROM_HANDLE.get() {
        Some(handle) => handle.render(),
        None => "# Prometheus recorder not initialized\n".to_string(),
    };
    if !body.ends_with('\n') {
        body.push('\n');
    }
    let engine_cache = state
        .llm
        .as_ref()
        .and_then(|engine| engine.cache_metrics_snapshot());
    body.push_str(&state.cache.prometheus_metrics(engine_cache.as_ref()));
    let engine_status = state.status().await;
    let scheduler_metrics = state.metrics();
    let auto_config = auto_config_health_value(state.auto_config.as_ref());
    let admission = admission_health_json(&engine_status, &scheduler_metrics, &auto_config);
    body.push_str(&admission_prometheus_metrics(&admission));

    Ok((
        [(
            axum::http::header::CONTENT_TYPE,
            "text/plain; version=0.0.4; charset=utf-8",
        )],
        body,
    )
        .into_response())
}

async fn root_handler() -> std::result::Result<Response, ServerError> {
    let info = serde_json::json!({
        "name": "Ferrum Inference Server",
        "version": env!("CARGO_PKG_VERSION"),
        "api_version": "v1",
        "status": "running"
    });

    Ok(Json(info).into_response())
}

/// Server error type for HTTP responses
#[derive(Debug)]
enum ServerError {
    InvalidRequest {
        message: String,
        param: Option<String>,
    },
    UnsupportedFeature {
        message: String,
        param: Option<String>,
    },
    InternalError(String),
    NotImplemented(String),
    ServiceUnavailable(String),
}

impl ServerError {
    fn invalid_request(message: impl Into<String>, param: Option<&str>) -> Self {
        Self::InvalidRequest {
            message: message.into(),
            param: param.map(str::to_string),
        }
    }

    fn unsupported_feature(message: impl Into<String>, param: Option<&str>) -> Self {
        Self::UnsupportedFeature {
            message: message.into(),
            param: param.map(str::to_string),
        }
    }
}

impl IntoResponse for ServerError {
    fn into_response(self) -> Response {
        let (status, message, error_type, param) = match self {
            ServerError::InvalidRequest { message, param } => (
                AxumStatusCode::BAD_REQUEST,
                message,
                "invalid_request_error",
                param,
            ),
            ServerError::UnsupportedFeature { message, param } => (
                AxumStatusCode::BAD_REQUEST,
                message,
                "invalid_request_error",
                param,
            ),
            ServerError::InternalError(msg) => (
                AxumStatusCode::INTERNAL_SERVER_ERROR,
                msg,
                "internal_server_error",
                None,
            ),
            ServerError::NotImplemented(msg) => (
                AxumStatusCode::SERVICE_UNAVAILABLE,
                msg,
                "service_unavailable_error",
                None,
            ),
            ServerError::ServiceUnavailable(msg) => (
                AxumStatusCode::SERVICE_UNAVAILABLE,
                msg,
                "service_unavailable_error",
                None,
            ),
        };

        let error = OpenAiError {
            error: OpenAiErrorDetail {
                message,
                error_type: error_type.to_string(),
                param,
                code: None,
            },
        };

        (status, Json(error)).into_response()
    }
}

impl std::fmt::Display for MessageRole {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MessageRole::System => write!(f, "system"),
            MessageRole::User => write!(f, "user"),
            MessageRole::Assistant => write!(f, "assistant"),
            MessageRole::Function => write!(f, "function"),
            MessageRole::Tool => write!(f, "tool"),
        }
    }
}

/// Strip model output at the first user-supplied stop sequence.
/// OpenAI-compatible `stop` strings are generation boundaries and must not
/// be returned to the caller, even if the model continued after the boundary.
fn strip_after_stop(text: &str, stops: &[String]) -> String {
    let mut first: Option<usize> = None;
    for stop in stops {
        if stop.is_empty() {
            continue;
        }
        if let Some(idx) = text.find(stop.as_str()) {
            first = Some(first.map_or(idx, |current| current.min(idx)));
        }
    }
    match first {
        Some(idx) => text[..idx].to_string(),
        None => text.to_string(),
    }
}

/// When `response_format = json_object` is set, the model is meant to
/// emit valid JSON only. Qwen / Llama instruct models frequently wrap
/// the JSON in markdown fences anyway (```` ```json ... ``` ````)
/// because that's how they were trained. `JsonModeProcessor` only
/// applies soft logit biases, not a hard mask, so the fence slips
/// through. Strip a single outermost ` ```json ... ``` ` /
/// ` ``` ... ``` ` wrapper. Preserves inner JSON exactly. Returns the
/// input unchanged if no fence is present.
fn strip_markdown_json_fence(text: &str) -> String {
    let trimmed = text.trim();
    // Try the most specific marker first.
    for prefix in ["```json\n", "```json", "```\n", "```"] {
        if let Some(rest) = trimmed.strip_prefix(prefix) {
            if let Some(inner) = rest.strip_suffix("```") {
                return inner.trim().to_string();
            }
        }
    }
    text.to_string()
}

/// Convert FinishReason to OpenAI API string
fn finish_reason_to_string(reason: &FinishReason) -> String {
    match reason {
        FinishReason::Length => "length".to_string(),
        FinishReason::Stop => "stop".to_string(),
        FinishReason::EOS => "stop".to_string(),
        FinishReason::Cancelled => "cancelled".to_string(),
        FinishReason::Error => "error".to_string(),
        FinishReason::ContentFilter => "content_filter".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use axum::{
        body::{to_bytes, Body},
        http::{header, Request},
        response::Response,
    };
    use ferrum_interfaces::engine::{
        EmbedEngine, InferenceEngine, LlmInferenceEngine, TranscribeEngine, TtsEngine,
    };
    use ferrum_types::{
        EngineConfig, EngineMetrics, EngineStatus, FinishReason,
        HealthStatus as EngineHealthStatus, InferenceRequest, InferenceResponse, MemoryUsage,
        ModelId, StreamChunk, TokenId, TokenUsage,
    };
    use futures::{stream, Stream};
    use serde_json::{json, Value};
    use std::{
        collections::HashMap,
        pin::Pin,
        sync::{Arc, Mutex},
    };
    use tower::ServiceExt;

    #[test]
    fn strip_after_stop_removes_first_boundary() {
        assert_eq!(
            strip_after_stop(
                "KS0214Z\nS0225\nEND0214Z0214Z\nS0225\n",
                &["END0214Z".to_string()]
            ),
            "KS0214Z\nS0225\n"
        );
    }

    struct StubLlm {
        config: EngineConfig,
        text: String,
        stream_chunks: Option<Vec<String>>,
        stream_final_chunk_separate: bool,
        stream_usage: Option<TokenUsage>,
        api_response: Option<ferrum_types::ApiResponse>,
        lora_metrics: Option<Value>,
    }

    impl StubLlm {
        fn new(text: &str) -> Self {
            let mut config = EngineConfig::default();
            config.model.model_id = ModelId::new("stub-model");
            Self {
                config,
                text: text.to_string(),
                stream_chunks: None,
                stream_final_chunk_separate: false,
                stream_usage: Some(TokenUsage::new(5, 1)),
                api_response: None,
                lora_metrics: None,
            }
        }

        fn without_stream_usage(text: &str) -> Self {
            Self {
                stream_usage: None,
                ..Self::new(text)
            }
        }

        fn with_stream_chunks(chunks: &[&str]) -> Self {
            Self {
                text: chunks.concat(),
                stream_chunks: Some(chunks.iter().map(|chunk| (*chunk).to_string()).collect()),
                ..Self::new("")
            }
        }

        fn with_separate_final_stream_chunk(chunks: &[&str]) -> Self {
            Self {
                text: chunks.concat(),
                stream_chunks: Some(chunks.iter().map(|chunk| (*chunk).to_string()).collect()),
                stream_final_chunk_separate: true,
                ..Self::new("")
            }
        }

        fn with_api_response(text: &str, api_response: ferrum_types::ApiResponse) -> Self {
            Self {
                api_response: Some(api_response),
                ..Self::new(text)
            }
        }

        fn with_lora_metrics(text: &str, lora_metrics: Value) -> Self {
            Self {
                lora_metrics: Some(lora_metrics),
                ..Self::new(text)
            }
        }
    }

    struct StubEmbed {
        config: EngineConfig,
    }

    impl StubEmbed {
        fn new() -> Self {
            let mut config = EngineConfig::default();
            config.model.model_id = ModelId::new("stub-embed");
            Self { config }
        }
    }

    struct StubTranscribe {
        config: EngineConfig,
    }

    impl StubTranscribe {
        fn new() -> Self {
            let mut config = EngineConfig::default();
            config.model.model_id = ModelId::new("stub-transcribe");
            Self { config }
        }
    }

    struct StubTts {
        config: EngineConfig,
    }

    impl StubTts {
        fn new() -> Self {
            let mut config = EngineConfig::default();
            config.model.model_id = ModelId::new("stub-tts");
            Self { config }
        }
    }

    struct FailingLlm {
        config: EngineConfig,
        fail_after_stream_start: bool,
    }

    impl FailingLlm {
        fn new() -> Self {
            let mut config = EngineConfig::default();
            config.model.model_id = ModelId::new("failing-model");
            Self {
                config,
                fail_after_stream_start: false,
            }
        }

        fn after_stream_start() -> Self {
            Self {
                fail_after_stream_start: true,
                ..Self::new()
            }
        }
    }

    struct CapturingLlm {
        config: EngineConfig,
        last_request: Mutex<Option<InferenceRequest>>,
    }

    impl CapturingLlm {
        fn new() -> Self {
            let mut config = EngineConfig::default();
            config.model.model_id = ModelId::new("qwen3");
            Self {
                config,
                last_request: Mutex::new(None),
            }
        }

        fn last_request(&self) -> InferenceRequest {
            self.last_request
                .lock()
                .expect("capture lock")
                .clone()
                .expect("request captured")
        }
    }

    #[async_trait]
    impl InferenceEngine for StubLlm {
        async fn status(&self) -> EngineStatus {
            EngineStatus {
                is_ready: true,
                loaded_models: vec![self.config.model.model_id.clone()],
                active_requests: 0,
                queued_requests: 0,
                memory_usage: MemoryUsage {
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
                version: "test".to_string(),
            }
        }

        async fn shutdown(&self) -> ferrum_types::Result<()> {
            Ok(())
        }

        fn config(&self) -> &EngineConfig {
            &self.config
        }

        fn metrics(&self) -> EngineMetrics {
            EngineMetrics::default()
        }

        async fn health_check(&self) -> EngineHealthStatus {
            EngineHealthStatus::healthy()
        }

        fn lora_metrics_snapshot(&self) -> Option<Value> {
            self.lora_metrics.clone()
        }
    }

    #[async_trait]
    impl InferenceEngine for StubEmbed {
        async fn status(&self) -> EngineStatus {
            EngineStatus {
                is_ready: true,
                loaded_models: vec![self.config.model.model_id.clone()],
                active_requests: 0,
                queued_requests: 0,
                memory_usage: MemoryUsage {
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
                version: "test".to_string(),
            }
        }

        async fn shutdown(&self) -> ferrum_types::Result<()> {
            Ok(())
        }

        fn config(&self) -> &EngineConfig {
            &self.config
        }

        fn metrics(&self) -> EngineMetrics {
            EngineMetrics::default()
        }

        async fn health_check(&self) -> EngineHealthStatus {
            EngineHealthStatus::healthy()
        }
    }

    #[async_trait]
    impl EmbedEngine for StubEmbed {
        async fn embed_text(&self, text: &str) -> ferrum_types::Result<Vec<f32>> {
            Ok(vec![text.len() as f32, 1.0, 0.0])
        }

        async fn embed_image(&self, image: &str) -> ferrum_types::Result<Vec<f32>> {
            Ok(vec![image.len() as f32, 0.0, 1.0])
        }

        fn embedding_dim(&self) -> usize {
            3
        }
    }

    #[async_trait]
    impl InferenceEngine for StubTranscribe {
        async fn status(&self) -> EngineStatus {
            EngineStatus {
                is_ready: true,
                loaded_models: vec![self.config.model.model_id.clone()],
                active_requests: 0,
                queued_requests: 0,
                memory_usage: MemoryUsage {
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
                version: "test".to_string(),
            }
        }

        async fn shutdown(&self) -> ferrum_types::Result<()> {
            Ok(())
        }

        fn config(&self) -> &EngineConfig {
            &self.config
        }

        fn metrics(&self) -> EngineMetrics {
            EngineMetrics::default()
        }

        async fn health_check(&self) -> EngineHealthStatus {
            EngineHealthStatus::healthy()
        }
    }

    #[async_trait]
    impl TranscribeEngine for StubTranscribe {
        async fn transcribe_file(
            &self,
            path: &str,
            language: Option<&str>,
        ) -> ferrum_types::Result<String> {
            Ok(format!("file:{path}:{}", language.unwrap_or("auto")))
        }

        async fn transcribe_bytes(
            &self,
            data: &[u8],
            language: Option<&str>,
        ) -> ferrum_types::Result<String> {
            Ok(format!(
                "bytes:{}:{}",
                data.len(),
                language.unwrap_or("auto")
            ))
        }
    }

    #[async_trait]
    impl InferenceEngine for StubTts {
        async fn status(&self) -> EngineStatus {
            EngineStatus {
                is_ready: true,
                loaded_models: vec![self.config.model.model_id.clone()],
                active_requests: 0,
                queued_requests: 0,
                memory_usage: MemoryUsage {
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
                version: "test".to_string(),
            }
        }

        async fn shutdown(&self) -> ferrum_types::Result<()> {
            Ok(())
        }

        fn config(&self) -> &EngineConfig {
            &self.config
        }

        fn metrics(&self) -> EngineMetrics {
            EngineMetrics::default()
        }

        async fn health_check(&self) -> EngineHealthStatus {
            EngineHealthStatus::healthy()
        }
    }

    #[async_trait]
    impl TtsEngine for StubTts {
        async fn synthesize_speech(
            &self,
            _text: &str,
            _language: Option<&str>,
            _chunk_frames: usize,
        ) -> ferrum_types::Result<Vec<Vec<f32>>> {
            Ok(vec![vec![0.0, 0.5, -0.5]])
        }

        fn tts_sample_rate(&self) -> u32 {
            16_000
        }
    }

    #[async_trait]
    impl InferenceEngine for FailingLlm {
        async fn status(&self) -> EngineStatus {
            EngineStatus {
                is_ready: true,
                loaded_models: vec![self.config.model.model_id.clone()],
                active_requests: 0,
                queued_requests: 0,
                memory_usage: MemoryUsage {
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
                version: "test".to_string(),
            }
        }

        async fn shutdown(&self) -> ferrum_types::Result<()> {
            Ok(())
        }

        fn config(&self) -> &EngineConfig {
            &self.config
        }

        fn metrics(&self) -> EngineMetrics {
            EngineMetrics::default()
        }

        async fn health_check(&self) -> EngineHealthStatus {
            EngineHealthStatus::healthy()
        }
    }

    #[async_trait]
    impl InferenceEngine for CapturingLlm {
        async fn status(&self) -> EngineStatus {
            EngineStatus {
                is_ready: true,
                loaded_models: vec![self.config.model.model_id.clone()],
                active_requests: 0,
                queued_requests: 0,
                memory_usage: MemoryUsage {
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
                version: "test".to_string(),
            }
        }

        async fn shutdown(&self) -> ferrum_types::Result<()> {
            Ok(())
        }

        fn config(&self) -> &EngineConfig {
            &self.config
        }

        fn metrics(&self) -> EngineMetrics {
            EngineMetrics::default()
        }

        async fn health_check(&self) -> EngineHealthStatus {
            EngineHealthStatus::healthy()
        }
    }

    #[async_trait]
    impl LlmInferenceEngine for StubLlm {
        async fn infer(
            &self,
            request: InferenceRequest,
        ) -> ferrum_types::Result<InferenceResponse> {
            Ok(InferenceResponse {
                request_id: request.id,
                text: self.text.clone(),
                tokens: vec![TokenId::new(11), TokenId::new(12)],
                finish_reason: FinishReason::Stop,
                usage: TokenUsage::new(7, 2),
                latency_ms: 1,
                created_at: chrono::Utc::now(),
                metadata: HashMap::new(),
                api_response: self.api_response.clone(),
            })
        }

        async fn infer_stream(
            &self,
            request: InferenceRequest,
        ) -> ferrum_types::Result<
            Pin<Box<dyn Stream<Item = ferrum_types::Result<StreamChunk>> + Send>>,
        > {
            if let Some(chunks) = &self.stream_chunks {
                let request_id = request.id;
                let mut stream_chunks = Vec::with_capacity(
                    chunks.len() + usize::from(self.stream_final_chunk_separate),
                );
                let last = chunks.len().saturating_sub(1);
                for (index, text) in chunks.iter().enumerate() {
                    let is_final_text_chunk = index == last && !self.stream_final_chunk_separate;
                    stream_chunks.push(Ok(StreamChunk {
                        request_id: request_id.clone(),
                        text: text.clone(),
                        token: Some(TokenId::new(11 + index as u32)),
                        finish_reason: is_final_text_chunk.then_some(FinishReason::Stop),
                        usage: is_final_text_chunk
                            .then(|| self.stream_usage.clone())
                            .flatten(),
                        created_at: chrono::Utc::now(),
                        metadata: HashMap::new(),
                        api_response: is_final_text_chunk
                            .then(|| self.api_response.clone())
                            .flatten(),
                    }));
                }
                if self.stream_final_chunk_separate {
                    stream_chunks.push(Ok(StreamChunk {
                        request_id,
                        text: String::new(),
                        token: None,
                        finish_reason: Some(FinishReason::Stop),
                        usage: self.stream_usage.clone(),
                        created_at: chrono::Utc::now(),
                        metadata: HashMap::new(),
                        api_response: self.api_response.clone(),
                    }));
                }
                return Ok(Box::pin(stream::iter(stream_chunks)));
            }

            let chunk = StreamChunk {
                request_id: request.id,
                text: self.text.clone(),
                token: Some(TokenId::new(11)),
                finish_reason: Some(FinishReason::Stop),
                usage: self.stream_usage.clone(),
                created_at: chrono::Utc::now(),
                metadata: HashMap::new(),
                api_response: self.api_response.clone(),
            };
            Ok(Box::pin(stream::iter(vec![Ok(chunk)])))
        }
    }

    #[async_trait]
    impl LlmInferenceEngine for FailingLlm {
        async fn infer(
            &self,
            _request: InferenceRequest,
        ) -> ferrum_types::Result<InferenceResponse> {
            Err(ferrum_types::FerrumError::internal(
                "stub generation failed",
            ))
        }

        async fn infer_stream(
            &self,
            request: InferenceRequest,
        ) -> ferrum_types::Result<
            Pin<Box<dyn Stream<Item = ferrum_types::Result<StreamChunk>> + Send>>,
        > {
            if self.fail_after_stream_start {
                let _request_id = request.id;
                return Ok(Box::pin(stream::iter(vec![Err(
                    ferrum_types::FerrumError::internal("stub stream chunk failed"),
                )])));
            }
            Err(ferrum_types::FerrumError::internal("stub stream failed"))
        }
    }

    #[async_trait]
    impl LlmInferenceEngine for CapturingLlm {
        async fn infer(
            &self,
            request: InferenceRequest,
        ) -> ferrum_types::Result<InferenceResponse> {
            *self.last_request.lock().expect("capture lock") = Some(request.clone());
            Ok(InferenceResponse {
                request_id: request.id,
                text: "captured".to_string(),
                tokens: vec![TokenId::new(21)],
                finish_reason: FinishReason::Stop,
                usage: TokenUsage::new(9, 1),
                latency_ms: 1,
                created_at: chrono::Utc::now(),
                metadata: HashMap::new(),
                api_response: None,
            })
        }

        async fn infer_stream(
            &self,
            request: InferenceRequest,
        ) -> ferrum_types::Result<
            Pin<Box<dyn Stream<Item = ferrum_types::Result<StreamChunk>> + Send>>,
        > {
            *self.last_request.lock().expect("capture lock") = Some(request.clone());
            let chunk = StreamChunk {
                request_id: request.id,
                text: "captured".to_string(),
                token: Some(TokenId::new(21)),
                finish_reason: Some(FinishReason::Stop),
                usage: Some(TokenUsage::new(9, 1)),
                created_at: chrono::Utc::now(),
                metadata: HashMap::new(),
                api_response: None,
            };
            Ok(Box::pin(stream::iter(vec![Ok(chunk)])))
        }
    }

    fn state_with_stub(text: &str) -> AppState {
        AppState::default().with_llm(Arc::new(StubLlm::new(text)))
    }

    fn router_with_stub(text: &str) -> Router {
        AxumServer::from_llm(Arc::new(StubLlm::new(text))).build_router()
    }

    fn router_with_stub_stream_chunks(chunks: &[&str]) -> Router {
        AxumServer::from_llm(Arc::new(StubLlm::with_stream_chunks(chunks))).build_router()
    }

    fn router_with_stub_separate_final_stream_chunk(chunks: &[&str]) -> Router {
        AxumServer::from_llm(Arc::new(StubLlm::with_separate_final_stream_chunk(chunks)))
            .build_router()
    }

    fn router_with_stub_api_response(
        text: &str,
        api_response: ferrum_types::ApiResponse,
    ) -> Router {
        AxumServer::from_llm(Arc::new(StubLlm::with_api_response(text, api_response)))
            .build_router()
    }

    fn router_with_stub_without_stream_usage(text: &str) -> Router {
        AxumServer::from_llm(Arc::new(StubLlm::without_stream_usage(text))).build_router()
    }

    fn router_without_llm() -> Router {
        AxumServer::from_state(AppState::default()).build_router()
    }

    fn router_with_failing_llm() -> Router {
        AxumServer::from_llm(Arc::new(FailingLlm::new())).build_router()
    }

    fn router_with_stream_chunk_failing_llm() -> Router {
        AxumServer::from_llm(Arc::new(FailingLlm::after_stream_start())).build_router()
    }

    fn router_with_capturing_llm() -> (Router, Arc<CapturingLlm>) {
        let engine = Arc::new(CapturingLlm::new());
        let router = AxumServer::from_llm(engine.clone()).build_router();
        (router, engine)
    }

    fn router_with_capturing_llm_and_template(
        template: ModelChatTemplate,
    ) -> (Router, Arc<CapturingLlm>) {
        let engine = Arc::new(CapturingLlm::new());
        let router = AxumServer::from_llm(engine.clone())
            .with_prompt_template(Some(template))
            .build_router();
        (router, engine)
    }

    fn router_with_capturing_lora_llm() -> (Router, Arc<CapturingLlm>) {
        let engine = Arc::new(CapturingLlm::new());
        let router = AxumServer::from_llm(engine.clone())
            .with_lora_adapters(
                "qwen3",
                vec![LoraAdapterModel::new(
                    "sql",
                    "qwen3:sql",
                    "/tmp/sql-adapter",
                )],
            )
            .build_router();
        (router, engine)
    }

    fn router_with_stub_embed() -> Router {
        AxumServer::from_embed(Arc::new(StubEmbed::new())).build_router()
    }

    fn router_with_stub_transcribe() -> Router {
        AxumServer::from_transcribe(Arc::new(StubTranscribe::new())).build_router()
    }

    fn router_with_stub_tts() -> Router {
        AxumServer::from_tts(Arc::new(StubTts::new())).build_router()
    }

    async fn post_json(app: Router, path: &str, body: Value) -> Response {
        app.oneshot(
            Request::builder()
                .method("POST")
                .uri(path)
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(body.to_string()))
                .expect("request"),
        )
        .await
        .expect("route response")
    }

    async fn post_raw_json(app: Router, path: &str, body: &str) -> Response {
        app.oneshot(
            Request::builder()
                .method("POST")
                .uri(path)
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(body.to_string()))
                .expect("request"),
        )
        .await
        .expect("route response")
    }

    async fn post_multipart(app: Router, path: &str, boundary: &str, body: &str) -> Response {
        app.oneshot(
            Request::builder()
                .method("POST")
                .uri(path)
                .header(
                    header::CONTENT_TYPE,
                    format!("multipart/form-data; boundary={boundary}"),
                )
                .body(Body::from(body.to_string()))
                .expect("request"),
        )
        .await
        .expect("route response")
    }

    async fn get(app: Router, path: &str) -> Response {
        app.oneshot(
            Request::builder()
                .method("GET")
                .uri(path)
                .body(Body::empty())
                .expect("request"),
        )
        .await
        .expect("route response")
    }

    async fn response_json(response: Response) -> Value {
        let bytes = to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("body bytes");
        serde_json::from_slice(&bytes).expect("json body")
    }

    async fn response_text(response: Response) -> String {
        let bytes = to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("body bytes");
        String::from_utf8(bytes.to_vec()).expect("utf8 body")
    }

    async fn response_bytes(response: Response) -> Vec<u8> {
        to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("body bytes")
            .to_vec()
    }

    async fn error_json(error: ServerError) -> (AxumStatusCode, Value) {
        let response = error.into_response();
        let status = response.status();
        (status, response_json(response).await)
    }

    fn assert_openai_stream_error(body: &str, expected_message: &str) {
        assert!(body.contains("data: [DONE]"), "missing DONE: {body}");
        assert!(
            body.contains("\"error\":{\"message\":\""),
            "stream failure should emit OpenAI error envelope: {body}"
        );
        assert!(
            body.contains(expected_message),
            "stream failure should include engine error message {expected_message:?}: {body}"
        );
        assert!(
            body.contains("\"type\":\"internal_server_error\""),
            "stream failure should use internal_server_error: {body}"
        );
        assert!(
            !body.contains("{\"error\":\""),
            "stream failure must not use legacy bare error payload: {body}"
        );
    }

    fn chat_request(extra: Value) -> ChatCompletionsRequest {
        let mut value = json!({
            "model": "stub-model",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 8
        });
        let obj = value.as_object_mut().unwrap();
        for (k, v) in extra.as_object().unwrap() {
            obj.insert(k.clone(), v.clone());
        }
        serde_json::from_value(value).expect("chat request")
    }

    #[test]
    fn sanitized_chat_request_body_redacts_user_text_and_secret_metadata() {
        let request = chat_request(json!({
            "messages": [{"role": "user", "content": "private prompt"}],
            "metadata": {"api_key": "should-not-survive"},
            "stream": true
        }));
        let body = sanitized_chat_request_body(&request);
        assert_eq!(body["model"], "stub-model");
        assert_eq!(body["stream"], true);
        assert_eq!(body["messages"][0]["role"], "user");
        assert_eq!(body["messages"][0]["content"], "[redacted]");
        assert_eq!(body["messages"][0]["content_redacted"], true);
        assert_eq!(body["messages"][0]["content_chars"], 14);
        assert_eq!(body["metadata"]["api_key"], "[redacted]");
    }

    #[tokio::test]
    async fn route_health_includes_runtime_config_snapshot() {
        let response = get(router_with_stub("ok"), "/health").await;
        assert_eq!(response.status(), AxumStatusCode::OK);
        let body = response_json(response).await;
        assert_eq!(body["status"], "healthy");
        assert!(body["config"]["entries"].is_array(), "body: {body}");
        assert_eq!(body["auto_config"]["schema_version"], 1);
        assert!(body["auto_config"]["entries"].is_array(), "body: {body}");
        assert!(body["auto_config"]["admission"].is_object(), "body: {body}");
        assert_eq!(body["admission"]["schema_version"], 1);
        assert!(body["admission"]["effective_max_concurrent"].is_number());
        assert!(body["admission"]["queue_depth"].is_number());
        assert!(body["admission"]["active_prefill"].is_number());
        assert!(body["admission"]["active_decode"].is_number());
        assert!(body["admission"]["current_batch_size"].is_number());
        assert!(body["admission"]["rejected_requests_total"].is_number());
        assert!(body["admission"]["failed_requests_total"].is_number());
        assert!(body["admission"]["completed_requests_total"].is_number());
        assert!(body["admission"]["avg_queue_wait_time_ms"].is_number());
        assert!(body["scheduler"]["avg_wait_time_ms"].is_number());
        assert!(body["scheduler"]["scheduling_time_ms"].is_number());
        assert!(body["scheduler"]["model_execution_time_ms"].is_number());
        assert!(body["scheduler"]["iteration_lock_wait_time_ms"].is_number());
        assert!(
            body["auto_config"]["decisions"].is_array() || body["auto_config"]["error"].is_string(),
            "body: {body}"
        );
    }

    #[tokio::test]
    async fn route_metrics_includes_admission_counters() {
        let response = get(router_with_stub("ok"), "/metrics").await;
        assert_eq!(response.status(), AxumStatusCode::OK);
        let body = response_text(response).await;
        for metric in [
            "ferrum_admission_effective_max_concurrent",
            "ferrum_admission_queue_depth",
            "ferrum_admission_active_prefill",
            "ferrum_admission_active_decode",
            "ferrum_admission_current_batch_size",
            "ferrum_admission_rejected_requests_total",
            "ferrum_admission_failed_requests_total",
            "ferrum_admission_completed_requests_total",
        ] {
            assert!(body.contains(metric), "missing {metric}:\n{body}");
        }
    }

    #[tokio::test]
    async fn route_health_includes_engine_lora_metrics_snapshot() {
        let router = AxumServer::from_llm(Arc::new(StubLlm::with_lora_metrics(
            "ok",
            json!({
                "enabled": true,
                "adapter_count": 1,
                "active_cache_bindings": 0,
                "projection_applications": 7,
                "position": "real-inference",
                "source": "test-lora",
            }),
        )))
        .with_lora_adapters(
            "stub-model",
            vec![LoraAdapterModel::new(
                "sql",
                "stub-model:sql",
                "/tmp/sql-adapter",
            )],
        )
        .build_router();
        let response = get(router, "/health").await;
        assert_eq!(response.status(), AxumStatusCode::OK);
        let body = response_json(response).await;
        assert_eq!(body["lora"]["enabled"], true);
        assert_eq!(body["lora"]["adapter_count"], 1);
        assert_eq!(body["lora"]["projection_applications"], 7);
        assert_eq!(body["lora"]["position"], "real-inference");
        assert_eq!(body["lora"]["source"], "test-lora");
    }

    #[tokio::test]
    async fn route_models_lists_loaded_stub_model() {
        let response = get(router_with_stub("ok"), "/v1/models").await;
        assert_eq!(response.status(), AxumStatusCode::OK);
        let body = response_json(response).await;
        assert_eq!(body["object"], "list");
        let data = body["data"].as_array().expect("models data array");
        assert_eq!(data.len(), 1, "body: {body}");
        assert_eq!(data[0]["id"], "stub-model");
        assert_eq!(data[0]["object"], "model");
        assert_eq!(data[0]["owned_by"], "ferrum");
        assert!(data[0]["created"].as_u64().unwrap_or_default() > 0);
        assert!(data[0]["permission"].as_array().unwrap().is_empty());
        assert!(data[0]["root"].is_null());
        assert!(data[0]["parent"].is_null());
    }

    #[tokio::test]
    async fn route_models_lists_startup_lora_adapters() {
        let router = AxumServer::from_llm(Arc::new(StubLlm::new("ok")))
            .with_lora_adapters(
                "stub-model",
                vec![LoraAdapterModel::new(
                    "sql",
                    "stub-model:sql",
                    "/tmp/sql-adapter",
                )],
            )
            .build_router();
        let response = get(router, "/v1/models").await;
        assert_eq!(response.status(), AxumStatusCode::OK);
        let body = response_json(response).await;
        let data = body["data"].as_array().expect("models data array");
        let ids: Vec<_> = data
            .iter()
            .map(|item| item["id"].as_str().unwrap_or_default())
            .collect();
        assert!(ids.contains(&"stub-model"), "body: {body}");
        assert!(ids.contains(&"stub-model:sql"), "body: {body}");
        let adapter = data
            .iter()
            .find(|item| item["id"] == "stub-model:sql")
            .expect("adapter model");
        assert_eq!(adapter["root"], "stub-model");
        assert_eq!(adapter["parent"], "stub-model");
    }

    #[tokio::test]
    async fn route_chat_lora_adapter_maps_internal_request_to_base_model() {
        let (router, engine) = router_with_capturing_lora_llm();
        let response = post_json(
            router,
            "/v1/chat/completions",
            json!({
                "model": "qwen3:sql",
                "messages": [{"role": "user", "content": "Say hi"}],
                "max_tokens": 8,
                "temperature": 0.0
            }),
        )
        .await;
        assert_eq!(response.status(), AxumStatusCode::OK);
        let body = response_json(response).await;
        assert_eq!(body["model"], "qwen3:sql");
        let captured = engine.last_request();
        assert_eq!(captured.model_id, ModelId::new("qwen3"));
        assert_eq!(captured.metadata["ferrum_lora_adapter"], "sql");
        assert_eq!(captured.metadata["ferrum_lora_model_id"], "qwen3:sql");
    }

    #[tokio::test]
    async fn route_chat_base_model_still_uses_base_path_with_lora_loaded() {
        let (router, engine) = router_with_capturing_lora_llm();
        let response = post_json(
            router,
            "/v1/chat/completions",
            json!({
                "model": "qwen3",
                "messages": [{"role": "user", "content": "Say hi"}],
                "max_tokens": 8,
                "temperature": 0.0
            }),
        )
        .await;
        assert_eq!(response.status(), AxumStatusCode::OK);
        let captured = engine.last_request();
        assert_eq!(captured.model_id, ModelId::new("qwen3"));
        assert!(!captured.metadata.contains_key("ferrum_lora_adapter"));
    }

    #[tokio::test]
    async fn route_chat_unknown_lora_adapter_returns_openai_model_error() {
        let (router, _) = router_with_capturing_lora_llm();
        let response = post_json(
            router,
            "/v1/chat/completions",
            json!({
                "model": "qwen3:missing",
                "messages": [{"role": "user", "content": "Say hi"}],
                "max_tokens": 8
            }),
        )
        .await;
        assert_eq!(response.status(), AxumStatusCode::BAD_REQUEST);
        let body = response_json(response).await;
        assert_eq!(body["error"]["type"], "invalid_request_error");
        assert_eq!(body["error"]["param"], "model");
        assert!(
            body["error"]["message"]
                .as_str()
                .unwrap_or_default()
                .contains("unknown LoRA adapter model"),
            "body: {body}"
        );
    }

    #[tokio::test]
    async fn route_models_without_engine_returns_empty_list() {
        let response = get(router_without_llm(), "/v1/models").await;
        assert_eq!(response.status(), AxumStatusCode::OK);
        let body = response_json(response).await;
        assert_eq!(body["object"], "list");
        assert!(body["data"].as_array().unwrap().is_empty(), "body: {body}");
    }

    #[tokio::test]
    async fn route_basic_chat_contract_uses_stub_engine() {
        let response = post_json(
            router_with_stub("hello"),
            "/v1/chat/completions",
            json!({
                "model": "stub-model",
                "messages": [{"role": "user", "content": "Say hi"}],
                "max_tokens": 8,
                "temperature": 0.0
            }),
        )
        .await;
        assert_eq!(response.status(), AxumStatusCode::OK);
        let body = response_json(response).await;
        assert_eq!(body["object"], "chat.completion");
        assert_eq!(body["choices"][0]["message"]["role"], "assistant");
        assert_eq!(body["choices"][0]["message"]["content"], "hello");
        assert_eq!(body["usage"]["prompt_tokens"], 7);
        assert_eq!(body["usage"]["completion_tokens"], 2);
    }

    #[tokio::test]
    async fn route_chat_serializes_structured_tool_call_response() {
        let response = post_json(
            router_with_stub_api_response(
                "",
                ferrum_types::ApiResponse::Chat(ferrum_types::ApiChatResponse {
                    message: ferrum_types::ApiChatMessage {
                        role: ferrum_types::ApiMessageRole::Assistant,
                        content: String::new(),
                        name: None,
                        tool_calls: vec![ferrum_types::ApiToolCall {
                            id: "call_1".to_string(),
                            tool_type: "function".to_string(),
                            function: ferrum_types::ApiFunctionCall {
                                name: "weather".to_string(),
                                arguments: "{\"city\":\"Paris\"}".to_string(),
                            },
                        }],
                        tool_call_id: None,
                        function_call: None,
                    },
                    finish_reason: Some("tool_calls".to_string()),
                }),
            ),
            "/v1/chat/completions",
            json!({
                "model": "stub-model",
                "messages": [{"role": "user", "content": "Use the weather tool."}],
                "tools": [{
                    "type": "function",
                    "function": {
                        "name": "weather",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                            "required": ["city"]
                        }
                    }
                }],
                "tool_choice": "auto"
            }),
        )
        .await;
        assert_eq!(response.status(), AxumStatusCode::OK);
        let body = response_json(response).await;
        assert_eq!(body["choices"][0]["finish_reason"], "tool_calls");
        assert_eq!(
            body["choices"][0]["message"]["tool_calls"][0]["function"]["name"],
            "weather"
        );
        assert_eq!(
            body["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"],
            "{\"city\":\"Paris\"}"
        );
    }

    #[tokio::test]
    async fn route_chat_serializes_generated_tool_call_json_when_engine_returns_text_only() {
        let response = post_json(
            router_with_stub(
                r#"{"tool_calls":[{"id":"call_1","type":"function","function":{"name":"weather","arguments":{"city":"Paris"}}}]}"#,
            ),
            "/v1/chat/completions",
            json!({
                "model": "stub-model",
                "messages": [{"role": "user", "content": "Use the weather tool."}],
                "tools": [{
                    "type": "function",
                    "function": {
                        "name": "weather",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                            "required": ["city"]
                        }
                    }
                }],
                "tool_choice": "auto"
            }),
        )
        .await;
        assert_eq!(response.status(), AxumStatusCode::OK);
        let body = response_json(response).await;
        assert_eq!(body["choices"][0]["finish_reason"], "tool_calls");
        assert_eq!(body["choices"][0]["message"]["content"], "");
        assert_eq!(
            body["choices"][0]["message"]["tool_calls"][0]["id"],
            "call_1"
        );
        assert_eq!(
            body["choices"][0]["message"]["tool_calls"][0]["function"]["name"],
            "weather"
        );
        assert_eq!(
            body["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"],
            "{\"city\":\"Paris\"}"
        );
    }

    #[tokio::test]
    async fn route_chat_serializes_qwen3_function_parameters_tool_json() {
        let response = post_json(
            router_with_stub(
                r#"{"function":"get_weather","parameters":{"city":"北京","unit":"c"}}"#,
            ),
            "/v1/chat/completions",
            json!({
                "model": "stub-model",
                "messages": [{"role": "user", "content": "北京现在天气怎么样？"}],
                "tools": [{
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "city": {"type": "string"},
                                "unit": {"type": "string", "enum": ["c", "f"]}
                            },
                            "required": ["city"]
                        }
                    }
                }],
                "tool_choice": "auto"
            }),
        )
        .await;
        assert_eq!(response.status(), AxumStatusCode::OK);
        let body = response_json(response).await;
        assert_eq!(body["choices"][0]["finish_reason"], "tool_calls");
        assert_eq!(body["choices"][0]["message"]["content"], "");
        assert_eq!(
            body["choices"][0]["message"]["tool_calls"][0]["function"]["name"],
            "get_weather"
        );
        assert_eq!(
            body["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"],
            "{\"city\":\"北京\",\"unit\":\"c\"}"
        );
    }

    #[tokio::test]
    async fn route_chat_parses_tool_call_from_reasoning_before_fake_tool_result_content() {
        let response = post_json(
            router_with_stub(
                "kaza\n\
                 {\"name\":\"get_weather\",\"arguments\":{\"city\":\"北京\",\"unit\":\"celsius\"}}\n\
                 </think>\n\
                 {\"name\":\"get_weather\",\"content\":{\"temperature\":25,\"condition\":\"晴\"}}\n\
                 {\"temperature\":25,\"condition\":\"晴\"}",
            ),
            "/v1/chat/completions",
            json!({
                "model": "stub-model",
                "messages": [{"role": "user", "content": "北京现在天气怎么样？请先调用工具。"}],
                "tools": [{
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "city": {"type": "string"},
                                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                            },
                            "required": ["city"]
                        }
                    }
                }]
            }),
        )
        .await;
        assert_eq!(response.status(), AxumStatusCode::OK);
        let body = response_json(response).await;
        assert_eq!(body["choices"][0]["finish_reason"], "tool_calls");
        assert_eq!(body["choices"][0]["message"]["content"], "");
        assert_eq!(
            body["choices"][0]["message"]["tool_calls"][0]["function"]["name"],
            "get_weather"
        );
        assert_eq!(
            body["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"],
            "{\"city\":\"北京\",\"unit\":\"celsius\"}"
        );
    }

    #[tokio::test]
    async fn route_chat_prefers_reasoning_tool_call_over_empty_visible_arguments() {
        let response = post_json(
            router_with_stub(
                "{\"name\":\"get_weather\",\"arguments\":{\"city\":\"北京\",\"unit\":\"celsius\"}}\n\
                 </think>\n\
                 {\"name\":\"get_weather\",\"arguments\":{}}",
            ),
            "/v1/chat/completions",
            json!({
                "model": "stub-model",
                "messages": [{"role": "user", "content": "北京现在天气怎么样？请先调用 get_weather 工具。"}],
                "tools": [{
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "city": {"type": "string"},
                                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                            },
                            "required": ["city"]
                        }
                    }
                }]
            }),
        )
        .await;
        assert_eq!(response.status(), AxumStatusCode::OK);
        let body = response_json(response).await;
        assert_eq!(body["choices"][0]["finish_reason"], "tool_calls");
        assert_eq!(
            body["choices"][0]["message"]["tool_calls"][0]["function"]["name"],
            "get_weather"
        );
        assert_eq!(
            body["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"],
            "{\"city\":\"北京\",\"unit\":\"celsius\"}"
        );
    }

    #[tokio::test]
    async fn route_chat_honors_specific_tool_choice_for_generated_tool_call_json() {
        let response = post_json(
            router_with_stub(r#"{"name":"weather","arguments":{"city":"Paris"}}"#),
            "/v1/chat/completions",
            json!({
                "model": "stub-model",
                "messages": [{"role": "user", "content": "Use the selected tool."}],
                "tools": [
                    {
                        "type": "function",
                        "function": {"name": "weather", "parameters": {"type": "object"}}
                    },
                    {
                        "type": "function",
                        "function": {"name": "calendar", "parameters": {"type": "object"}}
                    }
                ],
                "tool_choice": {
                    "type": "function",
                    "function": {"name": "weather"}
                }
            }),
        )
        .await;
        assert_eq!(response.status(), AxumStatusCode::OK);
        let body = response_json(response).await;
        assert_eq!(body["choices"][0]["finish_reason"], "tool_calls");
        assert_eq!(
            body["choices"][0]["message"]["tool_calls"][0]["function"]["name"],
            "weather"
        );

        let response = post_json(
            router_with_stub(r#"{"name":"calendar","arguments":{}}"#),
            "/v1/chat/completions",
            json!({
                "model": "stub-model",
                "messages": [{"role": "user", "content": "Use the selected tool."}],
                "tools": [
                    {
                        "type": "function",
                        "function": {"name": "weather", "parameters": {"type": "object"}}
                    },
                    {
                        "type": "function",
                        "function": {"name": "calendar", "parameters": {"type": "object"}}
                    }
                ],
                "tool_choice": {
                    "type": "function",
                    "function": {"name": "weather"}
                }
            }),
        )
        .await;
        assert_eq!(response.status(), AxumStatusCode::BAD_REQUEST);
        let body = response_json(response).await;
        assert_eq!(body["error"]["param"], "tool_choice");
        assert_eq!(body["error"]["type"], "invalid_request_error");
    }

    #[tokio::test]
    async fn route_chat_specific_tool_choice_wraps_generated_arguments() {
        let response = post_json(
            router_with_stub(r#"{"city":"Paris"}"#),
            "/v1/chat/completions",
            json!({
                "model": "stub-model",
                "messages": [{"role": "user", "content": "Use the selected tool."}],
                "tools": [{
                    "type": "function",
                    "function": {
                        "name": "weather",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                            "required": ["city"]
                        }
                    }
                }],
                "tool_choice": {
                    "type": "function",
                    "function": {"name": "weather"}
                }
            }),
        )
        .await;
        assert_eq!(response.status(), AxumStatusCode::OK);
        let body = response_json(response).await;
        assert_eq!(body["choices"][0]["finish_reason"], "tool_calls");
        assert_eq!(body["choices"][0]["message"]["content"], "");
        assert_eq!(
            body["choices"][0]["message"]["tool_calls"][0]["function"]["name"],
            "weather"
        );
        assert_eq!(
            body["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"],
            "{\"city\":\"Paris\"}"
        );
    }

    #[tokio::test]
    async fn route_chat_tool_choice_none_keeps_generated_tool_json_as_content() {
        let generated = r#"{"name":"weather","arguments":{"city":"Paris"}}"#;
        let response = post_json(
            router_with_stub(generated),
            "/v1/chat/completions",
            json!({
                "model": "stub-model",
                "messages": [{"role": "user", "content": "Do not use tools."}],
                "tools": [{
                    "type": "function",
                    "function": {"name": "weather", "parameters": {"type": "object"}}
                }],
                "tool_choice": "none"
            }),
        )
        .await;
        assert_eq!(response.status(), AxumStatusCode::OK);
        let body = response_json(response).await;
        assert_eq!(body["choices"][0]["finish_reason"], "stop");
        assert_eq!(body["choices"][0]["message"]["content"], generated);
        assert!(body["choices"][0]["message"]["tool_calls"].is_null());
    }

    #[tokio::test]
    async fn route_chat_tool_choice_required_wraps_generated_arguments() {
        let response = post_json(
            router_with_stub(r#"{"city":"Paris"}"#),
            "/v1/chat/completions",
            json!({
                "model": "stub-model",
                "messages": [{"role": "user", "content": "Use a tool."}],
                "tools": [{
                    "type": "function",
                    "function": {
                        "name": "weather",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                            "required": ["city"]
                        }
                    }
                }],
                "tool_choice": "required"
            }),
        )
        .await;
        assert_eq!(response.status(), AxumStatusCode::OK);
        let body = response_json(response).await;
        assert_eq!(body["choices"][0]["finish_reason"], "tool_calls");
        assert_eq!(body["choices"][0]["message"]["content"], "");
        assert_eq!(
            body["choices"][0]["message"]["tool_calls"][0]["function"]["name"],
            "weather"
        );
        assert_eq!(
            body["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"],
            "{\"city\":\"Paris\"}"
        );
    }

    #[tokio::test]
    async fn route_chat_tool_choice_required_errors_without_valid_tool_call() {
        let response = post_json(
            router_with_stub("plain answer"),
            "/v1/chat/completions",
            json!({
                "model": "stub-model",
                "messages": [{"role": "user", "content": "Use a tool."}],
                "tools": [{
                    "type": "function",
                    "function": {"name": "weather", "parameters": {"type": "object"}}
                }],
                "tool_choice": "required"
            }),
        )
        .await;
        assert_eq!(response.status(), AxumStatusCode::BAD_REQUEST);
        let body = response_json(response).await;
        assert_eq!(body["error"]["type"], "invalid_request_error");
        assert_eq!(body["error"]["param"], "tool_choice");
        assert!(
            body["error"]["message"]
                .as_str()
                .is_some_and(|message| message.contains("required tool_choice")),
            "body: {body}"
        );
    }

    #[tokio::test]
    async fn route_streaming_chat_serializes_generated_tool_call_delta() {
        let response = post_json(
            router_with_stub(
                r#"{"tool_calls":[{"id":"call_1","type":"function","function":{"name":"weather","arguments":{"city":"Paris"}}}]}"#,
            ),
            "/v1/chat/completions",
            json!({
                "model": "stub-model",
                "messages": [{"role": "user", "content": "Use the weather tool."}],
                "stream": true,
                "tools": [{
                    "type": "function",
                    "function": {
                        "name": "weather",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                            "required": ["city"]
                        }
                    }
                }],
                "tool_choice": "auto"
            }),
        )
        .await;
        assert_eq!(response.status(), AxumStatusCode::OK);
        let body = response_text(response).await;
        assert!(body.contains("data: [DONE]"), "missing DONE: {body}");
        assert!(
            body.contains(r#""finish_reason":"tool_calls""#),
            "stream should finish with tool_calls: {body}"
        );
        assert!(
            body.contains(r#""tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"weather""#),
            "stream should emit OpenAI tool_calls delta with index: {body}"
        );
        assert!(
            body.contains(r#""arguments":"{\"city\":\"Paris\"}""#),
            "tool arguments should be serialized as JSON string: {body}"
        );
        assert!(
            !body.contains(r#""content":"{\"tool_calls\""#),
            "raw tool-call JSON should not be streamed as assistant content: {body}"
        );
    }

    #[tokio::test]
    async fn route_streaming_chat_serializes_qwen3_function_parameters_tool_delta() {
        let response = post_json(
            router_with_stub(
                r#"{"function":"get_weather","parameters":{"city":"深圳","unit":"c"}}"#,
            ),
            "/v1/chat/completions",
            json!({
                "model": "stub-model",
                "messages": [{"role": "user", "content": "深圳天气？"}],
                "stream": true,
                "tools": [{
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "city": {"type": "string"},
                                "unit": {"type": "string", "enum": ["c", "f"]}
                            },
                            "required": ["city"]
                        }
                    }
                }],
                "tool_choice": "auto"
            }),
        )
        .await;
        assert_eq!(response.status(), AxumStatusCode::OK);
        let body = response_text(response).await;
        assert!(body.contains("data: [DONE]"), "missing DONE: {body}");
        assert!(
            body.contains(r#""finish_reason":"tool_calls""#),
            "stream should finish with tool_calls: {body}"
        );
        assert!(
            body.contains(r#""function":{"name":"get_weather","arguments":"{\"city\":\"深圳\",\"unit\":\"c\"}"}"#),
            "stream should emit parsed Qwen3 function parameters as tool args: {body}"
        );
        assert!(
            !body.contains(r#""content":"{\"function\""#),
            "raw Qwen3 tool JSON should not leak as assistant content: {body}"
        );
    }

    #[tokio::test]
    async fn route_streaming_chat_honors_specific_tool_choice_for_generated_tool_call_delta() {
        let request = |generated: &'static str| {
            post_json(
                router_with_stub(generated),
                "/v1/chat/completions",
                json!({
                    "model": "stub-model",
                    "messages": [{"role": "user", "content": "Use the selected tool."}],
                    "stream": true,
                    "tools": [
                        {
                            "type": "function",
                            "function": {"name": "weather", "parameters": {"type": "object"}}
                        },
                        {
                            "type": "function",
                            "function": {"name": "calendar", "parameters": {"type": "object"}}
                        }
                    ],
                    "tool_choice": {
                        "type": "function",
                        "function": {"name": "weather"}
                    }
                }),
            )
        };

        let response = request(r#"{"name":"weather","arguments":{"city":"Paris"}}"#).await;
        assert_eq!(response.status(), AxumStatusCode::OK);
        let body = response_text(response).await;
        assert!(body.contains("data: [DONE]"), "missing DONE: {body}");
        assert!(
            body.contains(r#""finish_reason":"tool_calls""#),
            "selected tool should finish with tool_calls: {body}"
        );
        assert!(
            body.contains(r#""function":{"name":"weather","arguments":"{\"city\":\"Paris\"}"}"#),
            "selected tool should stream as tool_calls delta: {body}"
        );

        let response = request(r#"{"name":"calendar","arguments":{}}"#).await;
        assert_eq!(response.status(), AxumStatusCode::OK);
        let body = response_text(response).await;
        assert!(
            body.contains(
                r#""error":{"message":"model output did not satisfy required tool_choice""#
            ),
            "selected-tool stream should reject unselected tool output: {body}"
        );
        assert!(
            !body.contains(r#""finish_reason":"tool_calls""#),
            "unselected tool JSON must not become tool_calls: {body}"
        );
    }

    #[tokio::test]
    async fn route_streaming_chat_prefers_chunk_api_response_for_tool_delta() {
        let response = post_json(
            router_with_stub_api_response(
                "raw text that should not stream",
                ferrum_types::ApiResponse::Chat(ferrum_types::ApiChatResponse {
                    message: ferrum_types::ApiChatMessage {
                        role: ferrum_types::ApiMessageRole::Assistant,
                        content: String::new(),
                        name: None,
                        tool_calls: vec![ferrum_types::ApiToolCall {
                            id: "call_1".to_string(),
                            tool_type: "function".to_string(),
                            function: ferrum_types::ApiFunctionCall {
                                name: "weather".to_string(),
                                arguments: "{\"city\":\"Paris\"}".to_string(),
                            },
                        }],
                        tool_call_id: None,
                        function_call: None,
                    },
                    finish_reason: Some("tool_calls".to_string()),
                }),
            ),
            "/v1/chat/completions",
            json!({
                "model": "stub-model",
                "messages": [{"role": "user", "content": "Use the weather tool."}],
                "stream": true,
                "tools": [{
                    "type": "function",
                    "function": {"name": "weather", "parameters": {"type": "object"}}
                }],
                "tool_choice": "auto"
            }),
        )
        .await;
        assert_eq!(response.status(), AxumStatusCode::OK);
        let body = response_text(response).await;
        assert!(body.contains("data: [DONE]"), "missing DONE: {body}");
        assert!(
            body.contains(r#""finish_reason":"tool_calls""#),
            "stream should finish with tool_calls: {body}"
        );
        assert!(
            body.contains(r#""tool_calls":[{"index":0,"id":"call_1""#),
            "stream should emit tool_calls from chunk api_response: {body}"
        );
        assert!(
            !body.contains("raw text that should not stream"),
            "structured api_response should suppress raw generated text in tool-call stream: {body}"
        );
    }

    #[tokio::test]
    async fn route_streaming_chat_tool_choice_required_errors_without_leaking_content() {
        let response = post_json(
            router_with_stub("plain answer"),
            "/v1/chat/completions",
            json!({
                "model": "stub-model",
                "messages": [{"role": "user", "content": "Use a tool."}],
                "stream": true,
                "tools": [{
                    "type": "function",
                    "function": {"name": "weather", "parameters": {"type": "object"}}
                }],
                "tool_choice": "required"
            }),
        )
        .await;
        assert_eq!(response.status(), AxumStatusCode::OK);
        let body = response_text(response).await;
        assert_eq!(body.matches("data: [DONE]").count(), 1, "body: {body}");
        assert!(
            body.contains(
                r#""error":{"message":"model output did not satisfy required tool_choice""#
            ),
            "stream should emit OpenAI error envelope: {body}"
        );
        assert!(
            body.contains(r#""type":"invalid_request_error""#),
            "stream should use invalid_request_error: {body}"
        );
        assert!(
            body.contains(r#""param":"tool_choice""#),
            "stream should include tool_choice param: {body}"
        );
        assert!(
            !body.contains(r#""content":"plain answer""#),
            "required stream must not leak invalid content before validation: {body}"
        );
    }

    #[tokio::test]
    async fn route_streaming_chat_tool_request_falls_back_to_content_when_no_tool_call() {
        let response = post_json(
            router_with_stub("plain answer"),
            "/v1/chat/completions",
            json!({
                "model": "stub-model",
                "messages": [{"role": "user", "content": "Use the weather tool if needed."}],
                "stream": true,
                "tools": [{
                    "type": "function",
                    "function": {"name": "weather", "parameters": {"type": "object"}}
                }],
                "tool_choice": "auto"
            }),
        )
        .await;
        assert_eq!(response.status(), AxumStatusCode::OK);
        let body = response_text(response).await;
        assert!(
            body.contains(r#""content":"plain answer""#),
            "plain content should still stream when no tool call is generated: {body}"
        );
        assert!(
            body.contains(r#""finish_reason":"stop""#),
            "plain content should keep normal finish reason: {body}"
        );
        assert!(
            !body.contains(r#""tool_calls""#),
            "fallback content should not synthesize tool_calls: {body}"
        );
    }

    #[tokio::test]
    async fn route_streaming_chat_serializes_generated_legacy_function_call_delta() {
        let response = post_json(
            router_with_stub(
                r#"{"function_call":{"name":"weather","arguments":{"city":"Paris"}}}"#,
            ),
            "/v1/chat/completions",
            json!({
                "model": "stub-model",
                "messages": [{"role": "user", "content": "Use the weather function."}],
                "stream": true,
                "functions": [{
                    "name": "weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"]
                    }
                }],
                "function_call": "auto"
            }),
        )
        .await;
        assert_eq!(response.status(), AxumStatusCode::OK);
        let body = response_text(response).await;
        assert!(body.contains("data: [DONE]"), "missing DONE: {body}");
        assert!(
            body.contains(r#""finish_reason":"function_call""#),
            "stream should finish with function_call: {body}"
        );
        assert!(
            body.contains(
                r#""function_call":{"name":"weather","arguments":"{\"city\":\"Paris\"}"}"#
            ),
            "stream should emit OpenAI legacy function_call delta: {body}"
        );
        assert!(
            !body.contains(r#""content":"{\"function_call\""#),
            "raw function-call JSON should not be streamed as assistant content: {body}"
        );
    }

    #[tokio::test]
    async fn route_streaming_chat_honors_specific_legacy_function_call_delta() {
        let request = |generated: &'static str| {
            post_json(
                router_with_stub(generated),
                "/v1/chat/completions",
                json!({
                    "model": "stub-model",
                    "messages": [{"role": "user", "content": "Use the selected function."}],
                    "stream": true,
                    "functions": [
                        {"name": "weather", "parameters": {"type": "object"}},
                        {"name": "calendar", "parameters": {"type": "object"}}
                    ],
                    "function_call": {"name": "weather"}
                }),
            )
        };

        let response =
            request(r#"{"function_call":{"name":"weather","arguments":{"city":"Paris"}}}"#).await;
        assert_eq!(response.status(), AxumStatusCode::OK);
        let body = response_text(response).await;
        assert!(body.contains("data: [DONE]"), "missing DONE: {body}");
        assert!(
            body.contains(r#""finish_reason":"function_call""#),
            "selected function should finish with function_call: {body}"
        );
        assert!(
            body.contains(
                r#""function_call":{"name":"weather","arguments":"{\"city\":\"Paris\"}"}"#
            ),
            "selected function should stream as function_call delta: {body}"
        );

        let response = request(r#"{"function_call":{"name":"calendar","arguments":{}}}"#).await;
        assert_eq!(response.status(), AxumStatusCode::OK);
        let body = response_text(response).await;
        assert!(
            body.contains(
                r#""content":"{\"function_call\":{\"name\":\"calendar\",\"arguments\":{}}}""#
            ),
            "unselected function JSON should stream as ordinary content: {body}"
        );
        assert!(
            body.contains(r#""finish_reason":"stop""#),
            "unselected function JSON should keep normal stop finish: {body}"
        );
        assert!(
            !body.contains(r#""finish_reason":"function_call""#),
            "unselected function JSON must not become function_call: {body}"
        );
    }

    #[tokio::test]
    async fn route_chat_serializes_generated_legacy_function_call_when_engine_returns_text_only() {
        let response = post_json(
            router_with_stub(
                r#"{"function_call":{"name":"weather","arguments":{"city":"Paris"}}}"#,
            ),
            "/v1/chat/completions",
            json!({
                "model": "stub-model",
                "messages": [{"role": "user", "content": "Use the weather function."}],
                "functions": [{
                    "name": "weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"]
                    }
                }],
                "function_call": "auto"
            }),
        )
        .await;
        assert_eq!(response.status(), AxumStatusCode::OK);
        let body = response_json(response).await;
        assert_eq!(body["choices"][0]["finish_reason"], "function_call");
        assert_eq!(body["choices"][0]["message"]["content"], "");
        assert_eq!(
            body["choices"][0]["message"]["function_call"]["name"],
            "weather"
        );
        assert_eq!(
            body["choices"][0]["message"]["function_call"]["arguments"],
            "{\"city\":\"Paris\"}"
        );
    }

    #[tokio::test]
    async fn route_chat_serializes_legacy_function_call_response() {
        let response = post_json(
            router_with_stub_api_response(
                "",
                ferrum_types::ApiResponse::Chat(ferrum_types::ApiChatResponse {
                    message: ferrum_types::ApiChatMessage {
                        role: ferrum_types::ApiMessageRole::Assistant,
                        content: String::new(),
                        name: None,
                        tool_calls: vec![],
                        tool_call_id: None,
                        function_call: Some(ferrum_types::ApiFunctionCall {
                            name: "weather".to_string(),
                            arguments: "{\"city\":\"Paris\"}".to_string(),
                        }),
                    },
                    finish_reason: Some("function_call".to_string()),
                }),
            ),
            "/v1/chat/completions",
            json!({
                "model": "stub-model",
                "messages": [{"role": "user", "content": "Use the weather function."}],
                "functions": [{
                    "name": "weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"]
                    }
                }],
                "function_call": "auto"
            }),
        )
        .await;
        assert_eq!(response.status(), AxumStatusCode::OK);
        let body = response_json(response).await;
        assert_eq!(body["choices"][0]["finish_reason"], "function_call");
        assert_eq!(
            body["choices"][0]["message"]["function_call"]["name"],
            "weather"
        );
        assert_eq!(
            body["choices"][0]["message"]["function_call"]["arguments"],
            "{\"city\":\"Paris\"}"
        );
    }

    #[tokio::test]
    async fn route_streaming_chat_include_usage_contract() {
        let response = post_json(
            router_with_stub("ok"),
            "/v1/chat/completions",
            json!({
                "model": "stub-model",
                "messages": [{"role": "user", "content": "Say ok"}],
                "stream": true,
                "stream_options": {"include_usage": true}
            }),
        )
        .await;
        assert_eq!(response.status(), AxumStatusCode::OK);
        let body = response_text(response).await;
        assert!(body.contains("data: [DONE]"), "missing DONE: {body}");
        assert!(
            body.contains("\"object\":\"chat.completion.chunk\""),
            "missing chat chunk: {body}"
        );
        assert!(
            body.contains("\"usage\":{\"prompt_tokens\""),
            "missing final usage chunk: {body}"
        );
        assert!(
            body.contains("\"choices\":[],\"usage\""),
            "usage should be emitted as a separate chunk: {body}"
        );
        assert!(
            body.contains("\"prompt_tokens\":5"),
            "stream usage should come from engine token usage: {body}"
        );
    }

    #[tokio::test]
    async fn route_streaming_chat_waits_for_separate_final_usage_at_max_tokens() {
        let response = post_json(
            router_with_stub_separate_final_stream_chunk(&["he", "llo"]),
            "/v1/chat/completions",
            json!({
                "model": "stub-model",
                "messages": [{"role": "user", "content": "Say hello"}],
                "max_tokens": 2,
                "stream": true,
                "stream_options": {"include_usage": true}
            }),
        )
        .await;
        assert_eq!(response.status(), AxumStatusCode::OK);
        let body = response_text(response).await;
        assert_eq!(body.matches("data: [DONE]").count(), 1, "body: {body}");
        assert!(
            body.contains("\"content\":\"he\""),
            "missing first chunk: {body}"
        );
        assert!(
            body.contains("\"content\":\"llo\""),
            "missing second chunk: {body}"
        );
        assert!(
            body.contains("\"choices\":[],\"usage\""),
            "missing separate usage chunk from final engine chunk: {body}"
        );
        assert!(
            body.contains("\"prompt_tokens\":5"),
            "stream usage should come from engine final usage: {body}"
        );
    }

    #[tokio::test]
    async fn route_rejects_multimodal_content_with_400() {
        let response = post_json(
            router_with_stub("unused"),
            "/v1/chat/completions",
            json!({
                "model": "stub-model",
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "describe this"},
                        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}}
                    ]
                }]
            }),
        )
        .await;
        assert_eq!(response.status(), AxumStatusCode::BAD_REQUEST);
        let body = response_json(response).await;
        assert_eq!(body["error"]["type"], "invalid_request_error");
        assert!(body["error"]["message"]
            .as_str()
            .unwrap()
            .contains("invalid chat completions request"));
    }

    #[tokio::test]
    async fn route_chat_invalid_json_maps_to_openai_error() {
        let response = post_raw_json(router_with_stub("unused"), "/v1/chat/completions", "{").await;
        assert_eq!(response.status(), AxumStatusCode::BAD_REQUEST);
        let body = response_json(response).await;
        assert_eq!(body["error"]["type"], "invalid_request_error");
        assert_eq!(body["error"]["param"], Value::Null);
        assert!(body["error"]["message"]
            .as_str()
            .unwrap()
            .contains("invalid chat completions request"));
    }

    #[tokio::test]
    async fn route_rejects_logit_bias_with_openai_error_param() {
        let response = post_json(
            router_with_stub("unused"),
            "/v1/chat/completions",
            json!({
                "model": "stub-model",
                "messages": [{"role": "user", "content": "hello"}],
                "logit_bias": {"1": 42.0}
            }),
        )
        .await;
        assert_eq!(response.status(), AxumStatusCode::BAD_REQUEST);
        let body = response_json(response).await;
        assert_eq!(body["error"]["type"], "invalid_request_error");
        assert_eq!(body["error"]["param"], "logit_bias");
    }

    #[tokio::test]
    async fn route_tool_request_reaches_engine_structured_boundary() {
        let (router, engine) = router_with_capturing_llm();
        let response = post_json(
            router,
            "/v1/chat/completions",
            json!({
                "model": "qwen3",
                "messages": [
                    {"role": "user", "content": "Use the weather tool."},
                    {
                        "role": "assistant",
                        "content": null,
                        "tool_calls": [{
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "weather", "arguments": "{\"city\":\"Paris\"}"}
                        }]
                    },
                    {"role": "tool", "tool_call_id": "call_1", "content": "sunny"}
                ],
                "tools": [{
                    "type": "function",
                    "function": {
                        "name": "weather",
                        "description": "Get weather",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                            "required": ["city"]
                        }
                    }
                }],
                "tool_choice": "auto",
                "functions": [{
                    "name": "legacy_weather",
                    "parameters": {"type": "object", "properties": {}}
                }],
                "function_call": "auto"
            }),
        )
        .await;
        assert_eq!(response.status(), AxumStatusCode::OK);

        let request = engine.last_request();
        assert!(request.prompt.contains("\"tools\":[{"));
        assert!(request.prompt.contains("\"type\":\"function\""));
        assert!(request.prompt.contains("\"name\":\"weather\""));
        assert!(request.prompt.contains("<|im_start|>assistant\n{"));
        assert!(request.prompt.contains("\"tool_calls\":[{"));
        assert!(request.prompt.contains("\"id\":\"call_1\""));
        assert!(request.prompt.contains("<|im_start|>tool\nsunny<|im_end|>"));
        assert_eq!(
            request.metadata["openai_tools"][0]["function"]["name"],
            "weather"
        );
        assert_eq!(request.metadata["openai_tool_choice"], "auto");
        assert_eq!(
            request.metadata["openai_legacy_functions"][0]["name"],
            "legacy_weather"
        );
        assert_eq!(request.metadata["openai_legacy_function_call"], "auto");
        let Some(ferrum_types::ApiRequest::Chat(api)) = request.api_request.as_ref() else {
            panic!("expected structured chat api_request");
        };
        assert_eq!(api.messages.len(), 3);
        assert_eq!(api.messages[2].role, ferrum_types::ApiMessageRole::Tool);
        assert_eq!(api.messages[2].tool_call_id.as_deref(), Some("call_1"));
        assert_eq!(api.tools[0].function.name, "weather");
        assert_eq!(api.legacy_functions[0].name, "legacy_weather");
        assert_eq!(
            api.messages[1].tool_calls[0].function.arguments,
            "{\"city\":\"Paris\"}"
        );
    }

    #[tokio::test]
    async fn route_tool_request_prefers_model_chat_template() {
        let template = ModelChatTemplate::new(
            "{% if tools %}<tools>{% for tool in tools %}{{ tool.function.name }}{% endfor %}</tools>{% endif %}{% for message in messages %}[{{ message.role }}]{{ message.content }}{% if message.tool_calls %}{% for tool_call in message.tool_calls %}<tool_call>{{ tool_call.function.name }}:{{ tool_call.function.arguments }}</tool_call>{% endfor %}{% endif %}{% if message.tool_call_id %}<tool_response id=\"{{ message.tool_call_id }}\">{{ message.content }}</tool_response>{% endif %}{% endfor %}{% if add_generation_prompt %}[assistant]{% endif %}",
            "tool-template",
        );
        let (router, engine) = router_with_capturing_llm_and_template(template);
        let response = post_json(
            router,
            "/v1/chat/completions",
            json!({
                "model": "served-alias",
                "messages": [
                    {"role": "user", "content": "Use the weather tool."},
                    {
                        "role": "assistant",
                        "content": null,
                        "tool_calls": [{
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "weather", "arguments": "{\"city\":\"Paris\"}"}
                        }]
                    },
                    {"role": "tool", "tool_call_id": "call_1", "content": "sunny"}
                ],
                "tools": [{
                    "type": "function",
                    "function": {
                        "name": "weather",
                        "description": "Get weather",
                        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}
                    }
                }],
                "tool_choice": "auto"
            }),
        )
        .await;
        assert_eq!(response.status(), AxumStatusCode::OK);

        let request = engine.last_request();
        assert!(request.prompt.contains("<tools>weather</tools>"));
        assert!(
            request.prompt.contains("<tool_call>weather:"),
            "{}",
            request.prompt
        );
        assert!(request.prompt.contains("\"city\""), "{}", request.prompt);
        assert!(request.prompt.contains("Paris"), "{}", request.prompt);
        assert!(request
            .prompt
            .contains("<tool_response id=\"call_1\">sunny</tool_response>"));
        assert!(
            !request.prompt.contains("<|assistant|>"),
            "model-template tool prompt should not use generic fallback: {}",
            request.prompt
        );
        assert!(
            !request.prompt.contains("When a tool is needed"),
            "model-template tool prompt should not inject fallback tool instructions: {}",
            request.prompt
        );
    }

    #[tokio::test]
    async fn chat_accepts_stop_string_and_max_completion_tokens() {
        let (router, engine) = router_with_capturing_llm();
        let response = post_json(
            router,
            "/v1/chat/completions",
            json!({
                "model": "stub-model",
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 99,
                "max_completion_tokens": 3,
                "stop": "<END>"
            }),
        )
        .await;
        assert_eq!(response.status(), AxumStatusCode::OK);

        let request = engine.last_request();
        assert_eq!(request.sampling_params.max_tokens, 3);
        assert_eq!(
            request.sampling_params.temperature,
            DEFAULT_SAMPLING_TEMPERATURE
        );
        assert_eq!(
            request.sampling_params.repetition_penalty,
            DEFAULT_CHAT_REPETITION_PENALTY
        );
        assert_eq!(request.sampling_params.stop_sequences, vec!["<END>"]);
    }

    #[tokio::test]
    async fn chat_request_forbids_initial_think_close_token() {
        let engine = Arc::new(CapturingLlm::new());
        let router = AxumServer::from_llm(engine.clone()).build_router();
        let response = post_json(
            router,
            "/v1/chat/completions",
            json!({
                "model": "stub-model",
                "messages": [{"role": "user", "content": "hello"}]
            }),
        )
        .await;
        assert_eq!(response.status(), AxumStatusCode::OK);

        let request = engine.last_request();
        assert_eq!(
            request
                .metadata
                .get(INITIAL_FORBIDDEN_TOKEN_TEXTS_METADATA_KEY),
            Some(&serde_json::json!([THINK_END_TAG]))
        );
    }

    #[tokio::test]
    async fn chat_template_enable_thinking_default_is_template_controlled() {
        let template = ModelChatTemplate::new(
            "{% for message in messages %}{{ '<|im_start|>' ~ message.role ~ '\n' ~ message.content ~ '<|im_end|>\n' }}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% if enable_thinking is defined and enable_thinking is false %}{{ '<think>\n\n</think>\n\n' }}{% endif %}{% endif %}",
            "test-template",
        );
        let (router, engine) = router_with_capturing_llm_and_template(template);
        let response = post_json(
            router,
            "/v1/chat/completions",
            json!({
                "model": "served-alias",
                "messages": [{"role": "user", "content": "hello"}]
            }),
        )
        .await;
        assert_eq!(response.status(), AxumStatusCode::OK);

        let request = engine.last_request();
        assert!(request
            .prompt
            .ends_with("<|im_start|>assistant\n<think>\n\n</think>\n\n"));
        assert_eq!(
            request
                .metadata
                .get(INITIAL_FORBIDDEN_TOKEN_TEXTS_METADATA_KEY),
            Some(&serde_json::json!([THINK_END_TAG, THINK_START_TAG]))
        );
    }

    #[tokio::test]
    async fn chat_template_enable_thinking_true_overrides_default() {
        let template = ModelChatTemplate::new(
            "{% if add_generation_prompt %}<assistant>{% if enable_thinking is defined and enable_thinking is false %}<think>\n\n</think>\n\n{% endif %}{% endif %}",
            "test-template",
        );
        let (router, engine) = router_with_capturing_llm_and_template(template);
        let response = post_json(
            router,
            "/v1/chat/completions",
            json!({
                "model": "served-alias",
                "messages": [{"role": "user", "content": "hello"}],
                "chat_template_kwargs": {"enable_thinking": true}
            }),
        )
        .await;
        assert_eq!(response.status(), AxumStatusCode::OK);

        let request = engine.last_request();
        assert_eq!(request.prompt, "<assistant>");
    }

    #[tokio::test]
    async fn chat_template_enable_thinking_rejects_non_bool() {
        let template = ModelChatTemplate::new(
            "{% if add_generation_prompt %}<assistant>{% endif %}",
            "test-template",
        );
        let (router, _) = router_with_capturing_llm_and_template(template);
        let response = post_json(
            router,
            "/v1/chat/completions",
            json!({
                "model": "served-alias",
                "messages": [{"role": "user", "content": "hello"}],
                "chat_template_kwargs": {"enable_thinking": "false"}
            }),
        )
        .await;
        assert_eq!(response.status(), AxumStatusCode::BAD_REQUEST);
        let body = response_json(response).await;
        assert_eq!(body["error"]["type"], "invalid_request_error");
        assert!(body["error"]["message"]
            .as_str()
            .unwrap_or_default()
            .contains("chat_template_kwargs.enable_thinking must be a boolean"));
    }

    #[tokio::test]
    async fn stop_string_strips_chat_and_completion_suffixes() {
        let chat = post_json(
            router_with_stub("hello<END>"),
            "/v1/chat/completions",
            json!({
                "model": "stub-model",
                "messages": [{"role": "user", "content": "hello"}],
                "stop": "<END>"
            }),
        )
        .await;
        assert_eq!(chat.status(), AxumStatusCode::OK);
        let chat_body = response_json(chat).await;
        assert_eq!(chat_body["choices"][0]["message"]["content"], "hello");

        let completion = post_json(
            router_with_stub("done<END>"),
            "/v1/completions",
            json!({
                "model": "stub-model",
                "prompt": "complete",
                "stop": "<END>"
            }),
        )
        .await;
        assert_eq!(completion.status(), AxumStatusCode::OK);
        let completion_body = response_json(completion).await;
        assert_eq!(completion_body["choices"][0]["text"], "done");
    }

    #[test]
    fn started_in_think_parse_streams_reasoning_before_end_tag() {
        // R1-distill templates open `<think>` inside the prompt, so the
        // generated text never contains the start tag. Mid-think text must
        // be reasoning, not content (this leaked as content deltas before).
        let parsed = parse_reasoning_response_started_in_think("Okay, the user wants");
        assert_eq!(parsed.reasoning.as_deref(), Some("Okay, the user wants"));
        assert_eq!(parsed.content, "");

        let parsed = parse_reasoning_response_started_in_think("thinking...</think>\nanswer");
        assert_eq!(parsed.reasoning.as_deref(), Some("thinking..."));
        assert_eq!(parsed.content, "answer");

        // Model re-opening its own think block defers to the normal parse.
        let parsed = parse_reasoning_response_started_in_think("<think>\nx\n</think>\n\nanswer");
        assert_eq!(parsed.reasoning.as_deref(), Some("\nx\n"));
        assert_eq!(parsed.content, "answer");
    }

    #[tokio::test]
    async fn chat_response_splits_reasoning_from_content() {
        let response = post_json(
            router_with_stub("<think>\nreasoning\n</think>\n\nfinal answer"),
            "/v1/chat/completions",
            json!({
                "model": "stub-model",
                "messages": [{"role": "user", "content": "hello"}]
            }),
        )
        .await;
        assert_eq!(response.status(), AxumStatusCode::OK);

        let body = response_json(response).await;
        let message = &body["choices"][0]["message"];
        assert_eq!(message["content"], "final answer");
        assert_eq!(message["reasoning"], "\nreasoning\n");
        assert!(message.get("reasoning_content").is_none());
    }

    #[tokio::test]
    async fn streaming_chat_reasoning_prefix_chunks_do_not_panic_or_leak_content() {
        let response = post_json(
            router_with_stub_stream_chunks(&["<", "think", ">\nreason", "\n</think>\n\nfinal"]),
            "/v1/chat/completions",
            json!({
                "model": "stub-model",
                "messages": [{"role": "user", "content": "think then answer"}],
                "stream": true
            }),
        )
        .await;
        assert_eq!(response.status(), AxumStatusCode::OK);
        let body = response_text(response).await;
        assert!(body.contains("data: [DONE]"), "missing DONE: {body}");
        assert!(
            body.contains(r#""reasoning":"\nreason"#),
            "stream should emit reasoning delta after full think prefix: {body}"
        );
        assert!(
            body.contains(r#""content":"final""#),
            "stream should emit visible content after think close: {body}"
        );
        assert!(
            !body.contains(r#""content":"<"#),
            "partial think prefix must not leak as content: {body}"
        );
    }

    #[tokio::test]
    async fn route_rejects_unsupported_tool_and_function_selection() {
        for (extra, param) in [
            (
                json!({
                    "tools": [{
                        "type": "function",
                        "function": {"name": "weather", "parameters": {"type": "object"}}
                    }],
                    "tool_choice": {
                        "type": "function",
                        "function": {"name": "calendar"}
                    }
                }),
                "tool_choice",
            ),
            (
                json!({
                    "functions": [{"name": "weather", "parameters": {"type": "object"}}],
                    "function_call": {"name": "calendar"}
                }),
                "function_call",
            ),
        ] {
            let mut body = json!({
                "model": "stub-model",
                "messages": [{"role": "user", "content": "hello"}]
            });
            body.as_object_mut()
                .expect("object")
                .extend(extra.as_object().expect("extra object").clone());
            let response =
                post_json(router_with_stub("unused"), "/v1/chat/completions", body).await;
            assert_eq!(response.status(), AxumStatusCode::BAD_REQUEST);
            let body = response_json(response).await;
            assert_eq!(body["error"]["type"], "invalid_request_error");
            assert_eq!(body["error"]["param"], param);
        }
    }

    #[tokio::test]
    async fn route_rejects_non_function_tools_with_openai_error_param() {
        let response = post_json(
            router_with_stub("unused"),
            "/v1/chat/completions",
            json!({
                "model": "stub-model",
                "messages": [{"role": "user", "content": "hello"}],
                "tools": [{
                    "type": "retrieval",
                    "function": {"name": "search", "parameters": {"type": "object"}}
                }]
            }),
        )
        .await;
        assert_eq!(response.status(), AxumStatusCode::BAD_REQUEST);
        let body = response_json(response).await;
        assert_eq!(body["error"]["type"], "invalid_request_error");
        assert_eq!(body["error"]["param"], "tools");
    }

    #[tokio::test]
    async fn route_rejects_tool_choice_required_without_tools() {
        let response = post_json(
            router_with_stub("unused"),
            "/v1/chat/completions",
            json!({
                "model": "stub-model",
                "messages": [{"role": "user", "content": "hello"}],
                "tool_choice": "required"
            }),
        )
        .await;
        assert_eq!(response.status(), AxumStatusCode::BAD_REQUEST);
        let body = response_json(response).await;
        assert_eq!(body["error"]["type"], "invalid_request_error");
        assert_eq!(body["error"]["param"], "tool_choice");
    }

    #[tokio::test]
    async fn route_rejects_unknown_response_format_type_with_openai_error_param() {
        let response = post_json(
            router_with_stub("unused"),
            "/v1/chat/completions",
            json!({
                "model": "stub-model",
                "messages": [{"role": "user", "content": "hello"}],
                "response_format": {"type": "xml"}
            }),
        )
        .await;
        assert_eq!(response.status(), AxumStatusCode::BAD_REQUEST);
        let body = response_json(response).await;
        assert_eq!(body["error"]["type"], "invalid_request_error");
        assert_eq!(body["error"]["param"], "response_format.type");
    }

    #[tokio::test]
    async fn route_chat_engine_unavailable_maps_to_503() {
        let response = post_json(
            router_without_llm(),
            "/v1/chat/completions",
            json!({
                "model": "stub-model",
                "messages": [{"role": "user", "content": "hello"}]
            }),
        )
        .await;
        assert_eq!(response.status(), AxumStatusCode::SERVICE_UNAVAILABLE);
        let body = response_json(response).await;
        assert_eq!(body["error"]["type"], "service_unavailable_error");
        assert_eq!(body["error"]["param"], Value::Null);
    }

    #[tokio::test]
    async fn route_chat_generation_failure_maps_to_500() {
        let response = post_json(
            router_with_failing_llm(),
            "/v1/chat/completions",
            json!({
                "model": "failing-model",
                "messages": [{"role": "user", "content": "hello"}]
            }),
        )
        .await;
        assert_eq!(response.status(), AxumStatusCode::INTERNAL_SERVER_ERROR);
        let body = response_json(response).await;
        assert_eq!(body["error"]["type"], "internal_server_error");
        assert!(body["error"]["message"]
            .as_str()
            .unwrap()
            .contains("stub generation failed"));
    }

    #[tokio::test]
    async fn route_chat_stream_generation_failure_emits_openai_error_event() {
        let response = post_json(
            router_with_failing_llm(),
            "/v1/chat/completions",
            json!({
                "model": "failing-model",
                "messages": [{"role": "user", "content": "hello"}],
                "stream": true
            }),
        )
        .await;
        assert_eq!(response.status(), AxumStatusCode::OK);
        let body = response_text(response).await;
        assert_openai_stream_error(&body, "stub stream failed");
    }

    #[tokio::test]
    async fn route_chat_stream_chunk_failure_emits_openai_error_event() {
        let response = post_json(
            router_with_stream_chunk_failing_llm(),
            "/v1/chat/completions",
            json!({
                "model": "failing-model",
                "messages": [{"role": "user", "content": "hello"}],
                "stream": true
            }),
        )
        .await;
        assert_eq!(response.status(), AxumStatusCode::OK);
        let body = response_text(response).await;
        assert_openai_stream_error(&body, "stub stream chunk failed");
    }

    #[tokio::test]
    async fn route_completions_engine_unavailable_maps_to_503() {
        let response = post_json(
            router_without_llm(),
            "/v1/completions",
            json!({
                "model": "stub-model",
                "prompt": "complete me"
            }),
        )
        .await;
        assert_eq!(response.status(), AxumStatusCode::SERVICE_UNAVAILABLE);
        let body = response_json(response).await;
        assert_eq!(body["error"]["type"], "service_unavailable_error");
        assert_eq!(body["error"]["param"], Value::Null);
    }

    #[tokio::test]
    async fn route_embeddings_engine_unavailable_maps_to_503() {
        let response = post_json(
            router_without_llm(),
            "/v1/embeddings",
            json!({
                "model": "embed-model",
                "input": "hello"
            }),
        )
        .await;
        assert_eq!(response.status(), AxumStatusCode::SERVICE_UNAVAILABLE);
        let body = response_json(response).await;
        assert_eq!(body["error"]["type"], "service_unavailable_error");
        assert_eq!(body["error"]["param"], Value::Null);
    }

    #[tokio::test]
    async fn route_embeddings_contract_uses_stub_engine() {
        let response = post_json(
            router_with_stub_embed(),
            "/v1/embeddings",
            json!({
                "model": "stub-embed",
                "input": ["hi", "world"],
                "encoding_format": "float"
            }),
        )
        .await;
        assert_eq!(response.status(), AxumStatusCode::OK);
        let body = response_json(response).await;
        assert_eq!(body["object"], "list");
        assert_eq!(body["model"], "stub-embed");
        assert_eq!(body["usage"]["prompt_tokens"], 7);
        assert_eq!(body["usage"]["total_tokens"], 7);

        let data = body["data"].as_array().expect("embedding data");
        assert_eq!(data.len(), 2, "body: {body}");
        assert_eq!(data[0]["object"], "embedding");
        assert_eq!(data[0]["index"], 0);
        assert_eq!(data[0]["embedding"].as_array().unwrap().len(), 3);
        assert_eq!(data[0]["embedding"][0].as_f64().unwrap(), 2.0);
        assert_eq!(data[1]["index"], 1);
        assert_eq!(data[1]["embedding"][0].as_f64().unwrap(), 5.0);
    }

    #[tokio::test]
    async fn route_embeddings_rejects_unsupported_encoding_format() {
        let response = post_json(
            router_with_stub_embed(),
            "/v1/embeddings",
            json!({
                "model": "stub-embed",
                "input": "hi",
                "encoding_format": "base64"
            }),
        )
        .await;
        assert_eq!(response.status(), AxumStatusCode::BAD_REQUEST);
        let body = response_json(response).await;
        assert_eq!(body["error"]["type"], "invalid_request_error");
        assert_eq!(body["error"]["param"], "encoding_format");
    }

    #[tokio::test]
    async fn route_embeddings_rejects_empty_input_with_field_param() {
        let response = post_json(
            router_with_stub_embed(),
            "/v1/embeddings",
            json!({
                "model": "stub-embed",
                "input": []
            }),
        )
        .await;
        assert_eq!(response.status(), AxumStatusCode::BAD_REQUEST);
        let body = response_json(response).await;
        assert_eq!(body["error"]["type"], "invalid_request_error");
        assert_eq!(body["error"]["param"], "input");
    }

    #[tokio::test]
    async fn route_embeddings_rejects_empty_item_with_field_param() {
        let response = post_json(
            router_with_stub_embed(),
            "/v1/embeddings",
            json!({
                "model": "stub-embed",
                "input": [{}]
            }),
        )
        .await;
        assert_eq!(response.status(), AxumStatusCode::BAD_REQUEST);
        let body = response_json(response).await;
        assert_eq!(body["error"]["type"], "invalid_request_error");
        assert_eq!(body["error"]["param"], "input");
    }

    #[tokio::test]
    async fn route_embeddings_invalid_json_maps_to_openai_error() {
        let response = post_raw_json(router_with_stub_embed(), "/v1/embeddings", "{").await;
        assert_eq!(response.status(), AxumStatusCode::BAD_REQUEST);
        let body = response_json(response).await;
        assert_eq!(body["error"]["type"], "invalid_request_error");
        assert_eq!(body["error"]["param"], Value::Null);
        assert!(body["error"]["message"]
            .as_str()
            .unwrap()
            .contains("invalid embeddings request"));
    }

    #[tokio::test]
    async fn route_transcriptions_engine_unavailable_maps_to_503() {
        let boundary = "ferrum-test-boundary";
        let body = concat!(
            "--ferrum-test-boundary\r\n",
            "Content-Disposition: form-data; name=\"file\"; filename=\"audio.wav\"\r\n",
            "Content-Type: audio/wav\r\n",
            "\r\n",
            "RIFFtest\r\n",
            "--ferrum-test-boundary--\r\n"
        );
        let response = post_multipart(
            router_without_llm(),
            "/v1/audio/transcriptions",
            boundary,
            body,
        )
        .await;
        assert_eq!(response.status(), AxumStatusCode::SERVICE_UNAVAILABLE);
        let body = response_json(response).await;
        assert_eq!(body["error"]["type"], "service_unavailable_error");
        assert_eq!(body["error"]["param"], Value::Null);
    }

    #[tokio::test]
    async fn route_transcriptions_contract_uses_stub_engine() {
        let boundary = "ferrum-test-boundary";
        let body = concat!(
            "--ferrum-test-boundary\r\n",
            "Content-Disposition: form-data; name=\"file\"; filename=\"audio.wav\"\r\n",
            "Content-Type: audio/wav\r\n",
            "\r\n",
            "RIFFtest\r\n",
            "--ferrum-test-boundary\r\n",
            "Content-Disposition: form-data; name=\"language\"\r\n",
            "\r\n",
            "en\r\n",
            "--ferrum-test-boundary\r\n",
            "Content-Disposition: form-data; name=\"response_format\"\r\n",
            "\r\n",
            "json\r\n",
            "--ferrum-test-boundary--\r\n"
        );
        let response = post_multipart(
            router_with_stub_transcribe(),
            "/v1/audio/transcriptions",
            boundary,
            body,
        )
        .await;
        assert_eq!(response.status(), AxumStatusCode::OK);
        let body = response_json(response).await;
        assert_eq!(body["text"], "bytes:8:en");
    }

    #[tokio::test]
    async fn route_transcriptions_rejects_unsupported_response_format() {
        let boundary = "ferrum-test-boundary";
        let body = concat!(
            "--ferrum-test-boundary\r\n",
            "Content-Disposition: form-data; name=\"file\"; filename=\"audio.wav\"\r\n",
            "Content-Type: audio/wav\r\n",
            "\r\n",
            "RIFFtest\r\n",
            "--ferrum-test-boundary\r\n",
            "Content-Disposition: form-data; name=\"response_format\"\r\n",
            "\r\n",
            "text\r\n",
            "--ferrum-test-boundary--\r\n"
        );
        let response = post_multipart(
            router_with_stub_transcribe(),
            "/v1/audio/transcriptions",
            boundary,
            body,
        )
        .await;
        assert_eq!(response.status(), AxumStatusCode::BAD_REQUEST);
        let body = response_json(response).await;
        assert_eq!(body["error"]["type"], "invalid_request_error");
        assert_eq!(body["error"]["param"], "response_format");
    }

    #[tokio::test]
    async fn route_transcriptions_rejects_missing_file_with_field_param() {
        let boundary = "ferrum-test-boundary";
        let body = concat!(
            "--ferrum-test-boundary\r\n",
            "Content-Disposition: form-data; name=\"language\"\r\n",
            "\r\n",
            "en\r\n",
            "--ferrum-test-boundary--\r\n"
        );
        let response = post_multipart(
            router_with_stub_transcribe(),
            "/v1/audio/transcriptions",
            boundary,
            body,
        )
        .await;
        assert_eq!(response.status(), AxumStatusCode::BAD_REQUEST);
        let body = response_json(response).await;
        assert_eq!(body["error"]["type"], "invalid_request_error");
        assert_eq!(body["error"]["param"], "file");
    }

    #[tokio::test]
    async fn route_transcriptions_invalid_multipart_maps_to_openai_error() {
        let response = post_json(
            router_with_stub_transcribe(),
            "/v1/audio/transcriptions",
            json!({}),
        )
        .await;
        assert_eq!(response.status(), AxumStatusCode::BAD_REQUEST);
        let body = response_json(response).await;
        assert_eq!(body["error"]["type"], "invalid_request_error");
        assert_eq!(body["error"]["param"], Value::Null);
        assert!(body["error"]["message"]
            .as_str()
            .unwrap()
            .contains("invalid transcriptions request"));
    }

    #[tokio::test]
    async fn route_speech_engine_unavailable_maps_to_503() {
        let response = post_json(
            router_without_llm(),
            "/v1/audio/speech",
            json!({
                "model": "tts-model",
                "input": "hello",
                "voice": "default"
            }),
        )
        .await;
        assert_eq!(response.status(), AxumStatusCode::SERVICE_UNAVAILABLE);
        let body = response_json(response).await;
        assert_eq!(body["error"]["type"], "service_unavailable_error");
        assert_eq!(body["error"]["param"], Value::Null);
    }

    #[tokio::test]
    async fn route_speech_contract_uses_stub_engine() {
        let response = post_json(
            router_with_stub_tts(),
            "/v1/audio/speech",
            json!({
                "model": "stub-tts",
                "input": "hello",
                "voice": "default",
                "response_format": "wav",
                "language": "english"
            }),
        )
        .await;
        assert_eq!(response.status(), AxumStatusCode::OK);
        assert_eq!(
            response.headers().get(header::CONTENT_TYPE).unwrap(),
            "audio/wav"
        );
        let body = response_bytes(response).await;
        assert!(body.len() > 44, "WAV should include header and PCM data");
        assert_eq!(&body[0..4], b"RIFF");
        assert_eq!(&body[8..12], b"WAVE");
    }

    #[tokio::test]
    async fn route_speech_streaming_contract_uses_stub_engine() {
        let response = post_json(
            router_with_stub_tts(),
            "/v1/audio/speech",
            json!({
                "model": "stub-tts",
                "input": "hello",
                "voice": "default",
                "response_format": "wav",
                "stream": true
            }),
        )
        .await;
        assert_eq!(response.status(), AxumStatusCode::OK);
        assert_eq!(
            response.headers().get(header::CONTENT_TYPE).unwrap(),
            "audio/wav"
        );
        assert_eq!(
            response.headers().get(header::TRANSFER_ENCODING).unwrap(),
            "chunked"
        );
        let body = response_bytes(response).await;
        assert!(body.len() > 44, "streaming WAV should include audio bytes");
        assert_eq!(&body[0..4], b"RIFF");
        assert_eq!(&body[8..12], b"WAVE");
    }

    #[tokio::test]
    async fn route_speech_pcm_response_format_returns_raw_pcm() {
        let response = post_json(
            router_with_stub_tts(),
            "/v1/audio/speech",
            json!({
                "model": "stub-tts",
                "input": "hello",
                "voice": "default",
                "response_format": "pcm"
            }),
        )
        .await;
        assert_eq!(response.status(), AxumStatusCode::OK);
        assert_eq!(
            response.headers().get(header::CONTENT_TYPE).unwrap(),
            "audio/pcm"
        );
        let body = response_bytes(response).await;
        assert_eq!(body.len(), 6, "three f32 samples should encode as s16le");
        assert_eq!(&body[0..2], &[0, 0]);
        assert_ne!(&body[0..4], b"RIFF");
    }

    #[tokio::test]
    async fn route_speech_rejects_unsupported_response_format() {
        let response = post_json(
            router_with_stub_tts(),
            "/v1/audio/speech",
            json!({
                "model": "stub-tts",
                "input": "hello",
                "voice": "default",
                "response_format": "mp3"
            }),
        )
        .await;
        assert_eq!(response.status(), AxumStatusCode::BAD_REQUEST);
        let body = response_json(response).await;
        assert_eq!(body["error"]["type"], "invalid_request_error");
        assert_eq!(body["error"]["param"], "response_format");
    }

    #[tokio::test]
    async fn route_speech_invalid_json_maps_to_openai_error() {
        let response = post_raw_json(router_with_stub_tts(), "/v1/audio/speech", "{").await;
        assert_eq!(response.status(), AxumStatusCode::BAD_REQUEST);
        let body = response_json(response).await;
        assert_eq!(body["error"]["type"], "invalid_request_error");
        assert_eq!(body["error"]["param"], Value::Null);
        assert!(body["error"]["message"]
            .as_str()
            .unwrap()
            .contains("invalid speech request"));
    }

    #[tokio::test]
    async fn route_completions_generation_failure_maps_to_500() {
        let response = post_json(
            router_with_failing_llm(),
            "/v1/completions",
            json!({
                "model": "failing-model",
                "prompt": "complete me"
            }),
        )
        .await;
        assert_eq!(response.status(), AxumStatusCode::INTERNAL_SERVER_ERROR);
        let body = response_json(response).await;
        assert_eq!(body["error"]["type"], "internal_server_error");
        assert!(body["error"]["message"]
            .as_str()
            .unwrap()
            .contains("stub generation failed"));
    }

    #[tokio::test]
    async fn route_completions_stream_generation_failure_emits_openai_error_event() {
        let response = post_json(
            router_with_failing_llm(),
            "/v1/completions",
            json!({
                "model": "failing-model",
                "prompt": "complete me",
                "stream": true
            }),
        )
        .await;
        assert_eq!(response.status(), AxumStatusCode::OK);
        let body = response_text(response).await;
        assert_openai_stream_error(&body, "stub stream failed");
    }

    #[tokio::test]
    async fn route_completions_stream_chunk_failure_emits_openai_error_event() {
        let response = post_json(
            router_with_stream_chunk_failing_llm(),
            "/v1/completions",
            json!({
                "model": "failing-model",
                "prompt": "complete me",
                "stream": true
            }),
        )
        .await;
        assert_eq!(response.status(), AxumStatusCode::OK);
        let body = response_text(response).await;
        assert_openai_stream_error(&body, "stub stream chunk failed");
    }

    #[tokio::test]
    async fn route_completions_contract_uses_stub_engine() {
        let response = post_json(
            router_with_stub("done"),
            "/v1/completions",
            json!({
                "model": "stub-model",
                "prompt": "complete me",
                "max_tokens": 8,
                "temperature": 0.0
            }),
        )
        .await;
        assert_eq!(response.status(), AxumStatusCode::OK);
        let body = response_json(response).await;
        assert_eq!(body["object"], "text_completion");
        assert_eq!(body["choices"][0]["text"], "done");
        assert_eq!(body["usage"]["prompt_tokens"], 7);
        assert_eq!(body["usage"]["completion_tokens"], 2);
    }

    #[tokio::test]
    async fn route_completions_streaming_contract_uses_stub_engine() {
        let response = post_json(
            router_with_stub("done"),
            "/v1/completions",
            json!({
                "model": "stub-model",
                "prompt": "complete me",
                "max_tokens": 8,
                "temperature": 0.0,
                "stream": true
            }),
        )
        .await;
        assert_eq!(response.status(), AxumStatusCode::OK);
        let body = response_text(response).await;
        assert!(body.contains("data: [DONE]"), "missing DONE: {body}");
        assert!(
            body.contains("\"object\":\"text_completion\""),
            "missing completion chunk: {body}"
        );
        assert!(body.contains("\"text\":\"done\""), "missing text: {body}");
        assert!(
            body.contains("\"choices\":[],\"usage\""),
            "missing separate usage chunk: {body}"
        );
        assert!(
            body.contains("\"prompt_tokens\":5"),
            "stream usage should come from engine token usage: {body}"
        );
        assert!(
            body.contains("\"completion_tokens\":1"),
            "stream completion usage should come from engine token usage: {body}"
        );
    }

    #[tokio::test]
    async fn route_completions_stream_waits_for_separate_final_usage_at_max_tokens() {
        let response = post_json(
            router_with_stub_separate_final_stream_chunk(&["do", "ne"]),
            "/v1/completions",
            json!({
                "model": "stub-model",
                "prompt": "complete me",
                "max_tokens": 2,
                "temperature": 0.0,
                "stream": true
            }),
        )
        .await;
        assert_eq!(response.status(), AxumStatusCode::OK);
        let body = response_text(response).await;
        assert_eq!(body.matches("data: [DONE]").count(), 1, "body: {body}");
        assert!(
            body.contains("\"text\":\"do\""),
            "missing first chunk: {body}"
        );
        assert!(
            body.contains("\"text\":\"ne\""),
            "missing second chunk: {body}"
        );
        assert!(
            body.contains("\"choices\":[],\"usage\""),
            "missing separate usage chunk from final engine chunk: {body}"
        );
        assert!(
            body.contains("\"prompt_tokens\":5"),
            "stream usage should come from engine final usage: {body}"
        );
    }

    #[tokio::test]
    async fn route_completions_invalid_json_maps_to_openai_error() {
        let response = post_raw_json(router_with_stub("unused"), "/v1/completions", "{").await;
        assert_eq!(response.status(), AxumStatusCode::BAD_REQUEST);
        let body = response_json(response).await;
        assert_eq!(body["error"]["type"], "invalid_request_error");
        assert_eq!(body["error"]["param"], Value::Null);
        assert!(body["error"]["message"]
            .as_str()
            .unwrap()
            .contains("invalid completions request"));
    }

    #[tokio::test]
    async fn route_completions_rejects_unsupported_fields_explicitly() {
        for (extra, param) in [
            (json!({"n": 2}), "n"),
            (json!({"logprobs": 3}), "logprobs"),
            (json!({"logit_bias": {"42": 1.0}}), "logit_bias"),
        ] {
            let mut body = json!({
                "model": "stub-model",
                "prompt": "complete me"
            });
            body.as_object_mut()
                .expect("object")
                .extend(extra.as_object().expect("extra object").clone());
            let response = post_json(router_with_stub("unused"), "/v1/completions", body).await;
            assert_eq!(response.status(), AxumStatusCode::BAD_REQUEST);
            let body = response_json(response).await;
            assert_eq!(body["error"]["type"], "invalid_request_error");
            assert_eq!(body["error"]["param"], param);
        }
    }

    #[tokio::test]
    async fn streaming_completions_do_not_synthesize_whitespace_usage() {
        let response = post_json(
            router_with_stub_without_stream_usage("done"),
            "/v1/completions",
            json!({
                "model": "stub-model",
                "prompt": "one two three four",
                "stream": true
            }),
        )
        .await;
        assert_eq!(response.status(), AxumStatusCode::OK);
        let body = response_text(response).await;
        assert!(body.contains("data: [DONE]"), "missing DONE: {body}");
        assert!(
            !body.contains("\"usage\":{\"prompt_tokens\""),
            "server must not synthesize whitespace-count completion usage: {body}"
        );
    }

    #[tokio::test]
    async fn chat_rejects_n_not_one_with_openai_error_param() {
        let request = chat_request(json!({"n": 2}));
        let err = chat_completions_handler(
            State(state_with_stub("unused")),
            HeaderMap::new(),
            Ok(Json(request)),
        )
        .await
        .expect_err("n=2 should reject");
        let (status, body) = error_json(err).await;
        assert_eq!(status, AxumStatusCode::BAD_REQUEST);
        assert_eq!(body["error"]["type"], "invalid_request_error");
        assert_eq!(body["error"]["param"], "n");
    }

    #[tokio::test]
    async fn chat_rejects_logit_bias_and_logprobs_explicitly() {
        for (extra, param) in [
            (json!({"logit_bias": {"1": 100.0}}), "logit_bias"),
            (json!({"logprobs": true}), "logprobs"),
            (json!({"top_logprobs": 2}), "top_logprobs"),
        ] {
            let request = chat_request(extra);
            let err = chat_completions_handler(
                State(state_with_stub("unused")),
                HeaderMap::new(),
                Ok(Json(request)),
            )
            .await
            .expect_err("unsupported field should reject");
            let (status, body) = error_json(err).await;
            assert_eq!(status, AxumStatusCode::BAD_REQUEST);
            assert_eq!(body["error"]["param"], param);
            assert_eq!(body["error"]["type"], "invalid_request_error");
        }
    }

    #[tokio::test]
    async fn chat_stream_options_include_usage_controls_stream_usage() {
        let request = chat_request(json!({
            "stream": true,
            "stream_options": {"include_usage": true}
        }));
        let response = chat_completions_handler(
            State(state_with_stub("ok")),
            HeaderMap::new(),
            Ok(Json(request)),
        )
        .await
        .expect("stream response");
        assert_eq!(response.status(), AxumStatusCode::OK);
        let body = response_text(response).await;
        assert!(body.contains("data: [DONE]"), "missing DONE: {body}");
        assert!(
            body.contains("\"usage\"") && body.contains("\"completion_tokens\":1"),
            "include_usage=true should emit stream usage: {body}"
        );
        assert!(
            body.contains("\"choices\":[],\"usage\""),
            "include_usage=true should use a separate usage chunk: {body}"
        );
        assert!(
            body.contains("\"prompt_tokens\":5"),
            "stream usage should come from engine token usage: {body}"
        );

        let request = chat_request(json!({"stream": true}));
        let response = chat_completions_handler(
            State(state_with_stub("ok")),
            HeaderMap::new(),
            Ok(Json(request)),
        )
        .await
        .expect("stream response");
        let body = response_text(response).await;
        assert!(
            !body.contains("\"usage\":{\"prompt_tokens\""),
            "stream usage should be omitted unless requested: {body}"
        );
    }

    #[tokio::test]
    async fn streaming_chat_does_not_synthesize_whitespace_usage() {
        let response = post_json(
            router_with_stub_without_stream_usage("ok"),
            "/v1/chat/completions",
            json!({
                "model": "stub-model",
                "messages": [{"role": "user", "content": "one two three four"}],
                "stream": true,
                "stream_options": {"include_usage": true}
            }),
        )
        .await;
        assert_eq!(response.status(), AxumStatusCode::OK);
        let body = response_text(response).await;
        assert!(body.contains("data: [DONE]"), "missing DONE: {body}");
        assert!(
            !body.contains("\"usage\":{\"prompt_tokens\""),
            "server must not synthesize whitespace-count usage when engine stream omits usage: {body}"
        );
    }

    #[test]
    fn tool_requests_and_tool_messages_parse_into_structured_api_request() {
        let request: ChatCompletionsRequest = serde_json::from_value(json!({
            "model": "qwen3",
            "messages": [
                {"role": "user", "content": "Use the weather tool."},
                {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "weather", "arguments": "{\"city\":\"Paris\"}"}
                    }]
                },
                {"role": "tool", "tool_call_id": "call_1", "content": "sunny"}
            ],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "weather",
                    "description": "Get weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"]
                    }
                }
            }],
            "tool_choice": "auto"
        }))
        .expect("tool request parses");

        validate_chat_request(&request).expect("tool request validates");
        let internal = convert_chat_request(&request).expect("convert");
        assert!(internal.prompt.contains("\"tools\":[{"));
        assert!(internal.prompt.contains("\"type\":\"function\""));
        assert!(internal.prompt.contains("\"name\":\"weather\""));
        assert!(internal.prompt.contains("<|im_start|>assistant\n{"));
        assert!(internal.prompt.contains("\"tool_calls\":[{"));
        assert!(internal.prompt.contains("\"id\":\"call_1\""));
        assert!(internal
            .prompt
            .contains("<|im_start|>tool\nsunny<|im_end|>"));
        assert_eq!(
            internal.metadata["openai_tools"][0]["function"]["name"],
            "weather"
        );
        assert_eq!(internal.metadata["openai_tool_choice"], "auto");
        let Some(ferrum_types::ApiRequest::Chat(api)) = internal.api_request.as_ref() else {
            panic!("expected structured chat api_request");
        };
        assert_eq!(api.messages.len(), 3);
        assert_eq!(api.messages[2].role, ferrum_types::ApiMessageRole::Tool);
        assert_eq!(api.messages[2].content, "sunny");
        assert_eq!(api.messages[2].tool_call_id.as_deref(), Some("call_1"));
        assert_eq!(api.tools[0].function.name, "weather");
        assert_eq!(
            api.tool_choice,
            Some(ferrum_types::ApiToolChoice::Mode("auto".into()))
        );
        assert_eq!(
            api.messages[1].tool_calls[0].function.arguments,
            "{\"city\":\"Paris\"}"
        );
    }

    #[test]
    fn omitted_tool_choice_defaults_to_auto_when_tools_are_present() {
        let request: ChatCompletionsRequest = serde_json::from_value(json!({
            "model": "served-alias",
            "messages": [{"role": "user", "content": "Use the weather tool."}],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "weather",
                    "description": "Get weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"]
                    }
                }
            }]
        }))
        .expect("tool request parses");

        validate_chat_request(&request).expect("tool request validates");
        let internal = convert_chat_request(&request).expect("convert");
        assert!(internal.prompt.contains("\"tools\":[{"));
        assert!(internal.prompt.contains("\"tool_choice\":\"auto\""));
        assert_eq!(internal.metadata["openai_tool_choice"], "auto");
        let initial_forbidden = internal.metadata[INITIAL_FORBIDDEN_TOKEN_TEXTS_METADATA_KEY]
            .as_array()
            .expect("initial forbidden token list");
        for token in [
            THINK_END_TAG,
            "<|im_end|>",
            "<|endoftext|>",
            "<|eot_id|>",
            "</s>",
        ] {
            assert!(
                initial_forbidden.iter().any(|value| value == token),
                "missing initial forbidden token {token}: {initial_forbidden:?}"
            );
        }
        let Some(ferrum_types::ApiRequest::Chat(api)) = internal.api_request.as_ref() else {
            panic!("expected structured chat api_request");
        };
        assert_eq!(
            api.tool_choice,
            Some(ferrum_types::ApiToolChoice::Mode("auto".into()))
        );
    }

    #[test]
    fn omitted_single_matching_tool_uses_tool_schema_response_format() {
        let request: ChatCompletionsRequest = serde_json::from_value(json!({
            "model": "served-alias",
            "messages": [{"role": "user", "content": "北京现在天气怎么样?用摄氏度。"}],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "查询指定城市的当前天气",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {"type": "string"},
                            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                        },
                        "required": ["city"]
                    }
                }
            }]
        }))
        .expect("tool request parses");

        validate_chat_request(&request).expect("tool request validates");
        let internal = convert_chat_request(&request).expect("convert");
        assert_eq!(internal.metadata["openai_tool_choice"], "auto");
        match internal.sampling_params.response_format {
            ferrum_types::ResponseFormat::JsonSchema(ref schema) => {
                assert!(schema.contains(r#""required":["city"]"#), "{schema}");
            }
            ref other => panic!("expected inferred tool json schema, got {other:?}"),
        }
    }

    #[test]
    fn tool_schema_response_format_bounds_unconstrained_strings() {
        let request: ChatCompletionsRequest = serde_json::from_value(json!({
            "model": "served-alias",
            "messages": [{"role": "user", "content": "Use the selected tool."}],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {"type": "string"},
                            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                        },
                        "required": ["city"]
                    }
                }
            }],
            "tool_choice": {
                "type": "function",
                "function": {"name": "get_weather"}
            }
        }))
        .expect("tool request parses");

        validate_chat_request(&request).expect("tool request validates");
        let internal = convert_chat_request(&request).expect("convert");
        match internal.sampling_params.response_format {
            ferrum_types::ResponseFormat::JsonSchema(ref schema) => {
                let value: serde_json::Value =
                    serde_json::from_str(schema).expect("schema should be JSON");
                assert_eq!(
                    value["properties"]["city"]["maxLength"],
                    DEFAULT_GUIDED_TOOL_ARGUMENT_STRING_MAX_LENGTH
                );
                assert_eq!(
                    value["properties"]["unit"]["enum"],
                    json!(["celsius", "fahrenheit"])
                );
                assert!(
                    value["properties"]["unit"]["maxLength"].is_null(),
                    "enum string should remain finite via enum instead of maxLength: {value}"
                );
            }
            ref other => panic!("expected forced tool json schema, got {other:?}"),
        }
    }

    #[test]
    fn required_tool_choice_uses_tool_schema_response_format_without_extra_prompt_instruction() {
        let request: ChatCompletionsRequest = serde_json::from_value(json!({
            "model": "served-alias",
            "messages": [{"role": "user", "content": "Call capture_quality_marker."}],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "capture_quality_marker",
                    "description": "Record one marker.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "marker": {"type": "string", "enum": ["ferrum0401"]},
                            "checksum": {"type": "string", "enum": ["S0004"]}
                        },
                        "required": ["marker", "checksum"]
                    }
                }
            }],
            "tool_choice": "required"
        }))
        .expect("tool request parses");

        validate_chat_request(&request).expect("tool request validates");
        let internal = convert_chat_request(&request).expect("convert");

        assert!(
            !internal.prompt.contains(
                "Output only a single JSON object containing the selected function arguments"
            ),
            "{}",
            internal.prompt
        );
        assert!(
            internal.prompt.contains("\"tool_choice\":\"required\""),
            "{}",
            internal.prompt
        );
        assert_eq!(internal.metadata["openai_tool_choice"], "required");
        match internal.sampling_params.response_format {
            ferrum_types::ResponseFormat::JsonSchema(ref schema) => {
                assert!(schema.contains(r#""enum":["ferrum0401"]"#), "{schema}");
                assert!(schema.contains(r#""enum":["S0004"]"#), "{schema}");
            }
            ref other => panic!("expected forced tool json schema, got {other:?}"),
        }
    }

    #[test]
    fn omitted_single_unrelated_tool_keeps_text_response_format() {
        let request: ChatCompletionsRequest = serde_json::from_value(json!({
            "model": "served-alias",
            "messages": [{"role": "user", "content": "讲一个短笑话。"}],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "查询指定城市的当前天气",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"]
                    }
                }
            }]
        }))
        .expect("tool request parses");

        validate_chat_request(&request).expect("tool request validates");
        let internal = convert_chat_request(&request).expect("convert");
        assert_eq!(
            internal.sampling_params.response_format,
            ferrum_types::ResponseFormat::Text
        );
    }

    #[test]
    fn tool_choice_none_omits_tools_from_model_template_prompt() {
        let request: ChatCompletionsRequest = serde_json::from_value(json!({
            "model": "served-alias",
            "messages": [
                {"role": "user", "content": "Use the weather tool if needed."},
                {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "weather", "arguments": "{\"city\":\"Paris\"}"}
                    }]
                },
                {"role": "tool", "tool_call_id": "call_1", "content": "{\"temp\":22}"}
            ],
            "tools": [{
                "type": "function",
                "function": {"name": "weather", "parameters": {"type": "object"}}
            }],
            "tool_choice": "none"
        }))
        .expect("tool_choice none request parses");
        let template = ModelChatTemplate::new(
            "{% set tools_in_user_message = true %}{% if tools %}<tools>{% for tool in tools %}{{ tool.function.name }}{% endfor %}</tools>{% endif %}{% for message in messages %}[{{ message.role }}]{{ message.content }}{% endfor %}{% if add_generation_prompt %}[assistant]{% endif %}",
            "tool-choice-none-template",
        );

        validate_chat_request(&request).expect("tool_choice none request validates");
        let internal = convert_chat_request_with_template_model(
            &request,
            "served-template-model",
            Some(&template),
        )
        .expect("convert");
        assert!(
            !internal.prompt.contains("<tools>"),
            "tool_choice none must not expose tools to the model template: {}",
            internal.prompt
        );
        assert!(internal.prompt.contains("[tool]"), "{}", internal.prompt);
        assert_eq!(
            internal.metadata["openai_tools"][0]["function"]["name"],
            "weather"
        );
        assert_eq!(internal.metadata["openai_tool_choice"], "none");
        let Some(ferrum_types::ApiRequest::Chat(api)) = internal.api_request.as_ref() else {
            panic!("expected structured chat api_request");
        };
        assert_eq!(api.tools[0].function.name, "weather");
        assert_eq!(
            api.tool_choice,
            Some(ferrum_types::ApiToolChoice::Mode("none".into()))
        );
    }

    #[test]
    fn specific_tool_choice_parses_into_structured_api_request() {
        let request: ChatCompletionsRequest = serde_json::from_value(json!({
            "model": "qwen3",
            "messages": [{"role": "user", "content": "Use the selected tool."}],
            "tools": [
                {
                    "type": "function",
                    "function": {"name": "weather", "parameters": {"type": "object"}}
                },
                {
                    "type": "function",
                    "function": {"name": "calendar", "parameters": {"type": "object"}}
                }
            ],
            "tool_choice": {
                "type": "function",
                "function": {"name": "weather"}
            }
        }))
        .expect("specific tool_choice request parses");

        validate_chat_request(&request).expect("specific tool_choice validates");
        let internal = convert_chat_request(&request).expect("convert");
        assert!(internal.prompt.contains("\"tool_choice\":{"));
        assert!(internal.prompt.contains("\"name\":\"weather\""));
        let Some(ferrum_types::ApiRequest::Chat(api)) = internal.api_request.as_ref() else {
            panic!("expected structured chat api_request");
        };
        assert_eq!(
            api.tool_choice,
            Some(ferrum_types::ApiToolChoice::Function {
                tool_type: "function".to_string(),
                function: ferrum_types::ApiToolChoiceFunction {
                    name: "weather".to_string()
                },
            })
        );

        let invalid: ChatCompletionsRequest = serde_json::from_value(json!({
            "model": "qwen3",
            "messages": [{"role": "user", "content": "Use the selected tool."}],
            "tools": [{
                "type": "function",
                "function": {"name": "weather", "parameters": {"type": "object"}}
            }],
            "tool_choice": {
                "type": "function",
                "function": {"name": "calendar"}
            }
        }))
        .expect("invalid specific tool_choice request parses");
        let err = validate_chat_request(&invalid).expect_err("undeclared tool should reject");
        match err {
            ServerError::InvalidRequest { param, .. } => {
                assert_eq!(param.as_deref(), Some("tool_choice"));
            }
            other => panic!("expected invalid_request_error for tool_choice, got {other:?}"),
        }
    }

    #[test]
    fn legacy_function_role_messages_parse_into_structured_api_request() {
        let request: ChatCompletionsRequest = serde_json::from_value(json!({
            "model": "mystery-model",
            "messages": [
                {"role": "user", "content": "Call weather."},
                {
                    "role": "assistant",
                    "content": null,
                    "function_call": {"name": "weather", "arguments": "{\"city\":\"Paris\"}"}
                },
                {"role": "function", "name": "weather", "content": "{\"forecast\":\"sunny\"}"}
            ],
            "functions": [{
                "name": "weather",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"]
                }
            }],
            "function_call": "auto"
        }))
        .expect("legacy function request parses");

        validate_chat_request(&request).expect("legacy function request validates");
        let internal = convert_chat_request(&request).expect("convert");
        assert!(
            internal
                .prompt
                .contains("<|function|>\n{\"forecast\":\"sunny\"}</s>"),
            "legacy function role should be preserved in fallback template: {}",
            internal.prompt
        );
        assert_eq!(
            internal.metadata["openai_legacy_functions"][0]["name"],
            "weather"
        );
        assert_eq!(internal.metadata["openai_legacy_function_call"], "auto");
        let Some(ferrum_types::ApiRequest::Chat(api)) = internal.api_request.as_ref() else {
            panic!("expected structured chat api_request");
        };
        assert_eq!(api.messages.len(), 3);
        assert_eq!(api.messages[2].role, ferrum_types::ApiMessageRole::Function);
        assert_eq!(api.messages[2].name.as_deref(), Some("weather"));
        assert_eq!(
            api.messages[1]
                .function_call
                .as_ref()
                .map(|call| call.name.as_str()),
            Some("weather")
        );
        assert_eq!(api.legacy_functions[0].name, "weather");
        assert_eq!(
            api.legacy_function_call,
            Some(ferrum_types::ApiFunctionCallChoice::Mode("auto".into()))
        );
    }

    #[test]
    fn specific_legacy_function_call_parses_into_structured_api_request() {
        let request: ChatCompletionsRequest = serde_json::from_value(json!({
            "model": "mystery-model",
            "messages": [{"role": "user", "content": "Use the selected function."}],
            "functions": [
                {"name": "weather", "parameters": {"type": "object"}},
                {"name": "calendar", "parameters": {"type": "object"}}
            ],
            "function_call": {"name": "weather"}
        }))
        .expect("specific function_call request parses");

        validate_chat_request(&request).expect("specific function_call validates");
        let internal = convert_chat_request(&request).expect("convert");
        assert!(internal.prompt.contains("\"function_call\":{"));
        assert!(internal.prompt.contains("\"name\":\"weather\""));
        let Some(ferrum_types::ApiRequest::Chat(api)) = internal.api_request.as_ref() else {
            panic!("expected structured chat api_request");
        };
        assert_eq!(
            api.legacy_function_call,
            Some(ferrum_types::ApiFunctionCallChoice::Function {
                name: "weather".to_string(),
            })
        );

        let invalid: ChatCompletionsRequest = serde_json::from_value(json!({
            "model": "mystery-model",
            "messages": [{"role": "user", "content": "Use the selected function."}],
            "functions": [{"name": "weather", "parameters": {"type": "object"}}],
            "function_call": {"name": "calendar"}
        }))
        .expect("invalid specific function_call request parses");
        let err = validate_chat_request(&invalid).expect_err("undeclared function should reject");
        match err {
            ServerError::InvalidRequest { param, .. } => {
                assert_eq!(param.as_deref(), Some("function_call"));
            }
            other => panic!("expected invalid_request_error for function_call, got {other:?}"),
        }
    }

    #[test]
    fn stream_text_delta_handles_unicode_boundaries() {
        let mut sent_len = 0usize;
        assert_eq!(stream_text_delta("你好", &mut sent_len), "你好");
        assert_eq!(sent_len, "你好".len());
        assert_eq!(stream_text_delta("你好世界", &mut sent_len), "世界");
        assert_eq!(sent_len, "你好世界".len());
    }

    #[test]
    fn stream_text_delta_recovers_from_non_boundary_offset() {
        let mut sent_len = 1usize;
        assert_eq!(stream_text_delta("你好", &mut sent_len), "");
        assert_eq!(sent_len, "你好".len());
    }

    #[test]
    fn assistant_tool_call_serializes_openai_shape() {
        let message = ChatMessage {
            role: MessageRole::Assistant,
            content: String::new(),
            reasoning: None,
            name: None,
            tool_calls: Some(vec![ChatToolCall {
                index: None,
                id: "call_1".to_string(),
                tool_type: "function".to_string(),
                function: ChatFunctionCall {
                    name: "weather".to_string(),
                    arguments: "{\"city\":\"Paris\"}".to_string(),
                },
            }]),
            tool_call_id: None,
            function_call: None,
        };
        let value = serde_json::to_value(message).expect("serialize");
        assert_eq!(value["role"], "assistant");
        assert_eq!(value["tool_calls"][0]["type"], "function");
        assert_eq!(value["tool_calls"][0]["function"]["name"], "weather");
    }

    #[test]
    fn unsupported_multimodal_content_is_not_silently_dropped() {
        let err = serde_json::from_value::<ChatCompletionsRequest>(json!({
            "model": "stub-model",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "describe this"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}}
                ]
            }]
        }))
        .expect_err("unsupported content part should fail parsing");
        assert!(
            err.to_string()
                .contains("unsupported message content part type"),
            "unexpected error: {err}"
        );
    }

    #[tokio::test]
    async fn completions_endpoint_uses_stub_engine() {
        let request = CompletionsRequest {
            model: "stub-model".to_string(),
            prompt: CompletionPrompt::Text("complete me".to_string()),
            max_tokens: Some(8),
            temperature: Some(0.0),
            top_p: None,
            n: None,
            stream: None,
            stop: None,
            logprobs: None,
            logit_bias: None,
        };
        let response = completions_handler(State(state_with_stub("done")), Ok(Json(request)))
            .await
            .expect("completion response");
        assert_eq!(response.status(), AxumStatusCode::OK);
        let body = response_json(response).await;
        assert_eq!(body["object"], "text_completion");
        assert_eq!(body["choices"][0]["text"], "done");
        assert_eq!(body["usage"]["prompt_tokens"], 7);
        assert_eq!(body["usage"]["completion_tokens"], 2);
    }

    #[tokio::test]
    async fn route_completions_rejects_non_string_prompt_with_field_param() {
        for prompt in [
            json!(["a", "b"]),
            json!({"text": "complete me"}),
            Value::Null,
        ] {
            let response = post_json(
                router_with_stub("unused"),
                "/v1/completions",
                json!({
                    "model": "stub-model",
                    "prompt": prompt
                }),
            )
            .await;
            assert_eq!(response.status(), AxumStatusCode::BAD_REQUEST);
            let body = response_json(response).await;
            assert_eq!(body["error"]["type"], "invalid_request_error");
            assert_eq!(body["error"]["param"], "prompt");
        }

        let response = post_json(
            router_with_stub("unused"),
            "/v1/completions",
            json!({"model": "stub-model"}),
        )
        .await;
        assert_eq!(response.status(), AxumStatusCode::BAD_REQUEST);
        let body = response_json(response).await;
        assert_eq!(body["error"]["type"], "invalid_request_error");
        assert_eq!(body["error"]["param"], "prompt");
    }

    #[tokio::test]
    async fn stream_options_without_stream_is_invalid() {
        let request = chat_request(json!({"stream_options": {"include_usage": true}}));
        let err = chat_completions_handler(
            State(state_with_stub("unused")),
            HeaderMap::new(),
            Ok(Json(request)),
        )
        .await
        .expect_err("stream_options without stream should reject");
        let (status, body) = error_json(err).await;
        assert_eq!(status, AxumStatusCode::BAD_REQUEST);
        assert_eq!(body["error"]["param"], "stream_options");
        assert_eq!(body["error"]["type"], "invalid_request_error");
    }

    #[tokio::test]
    async fn json_object_strips_single_markdown_fence_as_best_effort_repair() {
        let request = chat_request(json!({
            "response_format": {"type": "json_object"}
        }));
        let response = chat_completions_handler(
            State(state_with_stub("```json\n{\"answer\":\"yes\"}\n```")),
            HeaderMap::new(),
            Ok(Json(request)),
        )
        .await
        .expect("json_object response");
        assert_eq!(response.status(), AxumStatusCode::OK);
        let body = response_json(response).await;
        let content = body["choices"][0]["message"]["content"]
            .as_str()
            .expect("content string");
        assert_eq!(content, "{\"answer\":\"yes\"}");
        let parsed: serde_json::Value = serde_json::from_str(content).expect("parse json_object");
        assert_eq!(parsed["answer"], "yes");
    }

    #[tokio::test]
    async fn streaming_json_object_buffers_thinking_and_emits_clean_json_content() {
        let response = post_json(
            router_with_stub_stream_chunks(&[
                "<think>\n好的，我需要输出 JSON。",
                "\n</think>\n\n",
                "{\"name\":\"李四\",\"age\":30}",
            ]),
            "/v1/chat/completions",
            json!({
                "model": "stub-model",
                "messages": [{"role": "user", "content": "输出JSON(name,age)：李四,30岁"}],
                "stream": true,
                "response_format": {"type": "json_object"}
            }),
        )
        .await;
        assert_eq!(response.status(), AxumStatusCode::OK);
        let body = response_text(response).await;
        assert!(body.contains("data: [DONE]"), "missing DONE: {body}");
        assert!(
            body.contains(r#""content":"{\"name\":\"李四\",\"age\":30}""#),
            "stream should emit clean JSON content: {body}"
        );
        assert!(
            body.contains(r#""reasoning":"\n好的，我需要输出 JSON。\n""#),
            "stream should keep thinking in reasoning field: {body}"
        );
        assert!(
            !body.contains(r#""content":"<think"#)
                && !body.contains(r#""content":"好的"#)
                && !body.contains(r#""content":"我需要"#),
            "thinking text must not leak as streamed content: {body}"
        );
    }

    #[tokio::test]
    async fn json_object_remains_best_effort_not_strict_validation() {
        let request = chat_request(json!({
            "response_format": {"type": "json_object"}
        }));
        let response = chat_completions_handler(
            State(state_with_stub("not json")),
            HeaderMap::new(),
            Ok(Json(request)),
        )
        .await
        .expect("json_object remains best-effort");
        assert_eq!(response.status(), AxumStatusCode::OK);
        let body = response_json(response).await;
        assert_eq!(body["choices"][0]["message"]["content"], "not json");
    }

    #[tokio::test]
    async fn unsupported_strict_json_schema_is_rejected_at_boundary() {
        let request = chat_request(json!({
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "unsupported",
                    "strict": true,
                    "schema": {"oneOf": [{"type": "string"}, {"type": "integer"}]}
                }
            }
        }));
        let err = chat_completions_handler(
            State(state_with_stub("unused")),
            HeaderMap::new(),
            Ok(Json(request)),
        )
        .await
        .expect_err("unsupported strict schema should reject");
        let (status, body) = error_json(err).await;
        assert_eq!(status, AxumStatusCode::BAD_REQUEST);
        assert_eq!(body["error"]["param"], "response_format.json_schema");
        assert_eq!(body["error"]["type"], "invalid_request_error");
    }

    #[tokio::test]
    async fn missing_json_schema_schema_rejects_with_field_param() {
        let request = chat_request(json!({
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "missing_schema",
                    "strict": true
                }
            }
        }));
        let err = chat_completions_handler(
            State(state_with_stub("unused")),
            HeaderMap::new(),
            Ok(Json(request)),
        )
        .await
        .expect_err("missing strict schema should reject");
        let (status, body) = error_json(err).await;
        assert_eq!(status, AxumStatusCode::BAD_REQUEST);
        assert_eq!(body["error"]["param"], "response_format.json_schema");
        assert_eq!(body["error"]["type"], "invalid_request_error");
        assert!(body["error"]["message"]
            .as_str()
            .unwrap()
            .contains("schema is required"));
    }

    #[test]
    fn non_strict_json_schema_is_preserved_but_not_hard_masked() {
        let request = chat_request(json!({
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "best_effort",
                    "strict": false,
                    "schema": {"oneOf": [{"type": "string"}, {"type": "integer"}]}
                }
            }
        }));

        validate_chat_request(&request).expect("non-strict schema should not boundary reject");
        let internal = convert_chat_request(&request).expect("convert non-strict schema");
        assert!(
            internal
                .prompt
                .contains("response_format requires a single valid JSON object"),
            "response_format instruction should reach the model prompt: {}",
            internal.prompt
        );
        assert!(
            internal.prompt.contains("\"oneOf\""),
            "schema should reach the model prompt: {}",
            internal.prompt
        );
        assert_eq!(
            internal.sampling_params.response_format,
            ferrum_types::ResponseFormat::Text,
            "non-strict json_schema must stay best-effort instead of enabling hard guided decode"
        );
        let Some(ferrum_types::ApiRequest::Chat(api)) = internal.api_request.as_ref() else {
            panic!("expected structured chat api_request");
        };
        assert_eq!(
            api.response_format
                .as_ref()
                .and_then(|format| format.json_schema.as_ref())
                .and_then(|schema| schema.strict),
            Some(false)
        );
    }

    #[test]
    fn json_object_response_format_instruction_reaches_model_prompt() {
        let request = chat_request(json!({
            "response_format": {"type": "json_object"}
        }));

        let internal = convert_chat_request(&request).expect("convert json_object");
        assert!(
            internal
                .prompt
                .contains("response_format requires a single valid JSON object"),
            "response_format instruction should reach the model prompt: {}",
            internal.prompt
        );
        assert!(
            internal.prompt.contains("Output only JSON"),
            "JSON-only instruction should reach the model prompt: {}",
            internal.prompt
        );
        assert_eq!(
            internal.sampling_params.response_format,
            ferrum_types::ResponseFormat::Text,
            "json_object response_format should use prompt instruction and final JSON cleanup, not engine-wide JSON soft bias"
        );
    }

    #[test]
    fn strict_json_schema_response_format_uses_guided_sampling_mode() {
        let request = chat_request(json!({
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "answer",
                    "strict": true,
                    "schema": {
                        "type": "object",
                        "properties": {"answer": {"type": "string"}},
                        "required": ["answer"]
                    }
                }
            }
        }));

        let internal = convert_chat_request(&request).expect("convert strict json_schema");
        assert!(
            internal
                .prompt
                .contains("response_format requires a single valid JSON object"),
            "response_format instruction should reach the model prompt: {}",
            internal.prompt
        );
        let ferrum_types::ResponseFormat::JsonSchema(schema) =
            internal.sampling_params.response_format
        else {
            panic!(
                "strict json_schema must reach guided decoding, got {:?}",
                internal.sampling_params.response_format
            );
        };
        let schema: serde_json::Value = serde_json::from_str(&schema).unwrap();
        assert_eq!(schema["type"], "object");
        assert_eq!(schema["properties"]["answer"]["type"], "string");
        assert_eq!(schema["required"], json!(["answer"]));
    }

    #[tokio::test]
    async fn strict_json_schema_validates_non_streaming_response() {
        let request = chat_request(json!({
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "answer",
                    "strict": true,
                    "schema": {
                        "type": "object",
                        "properties": {"answer": {"type": "string"}},
                        "required": ["answer"]
                    }
                }
            }
        }));
        let response = chat_completions_handler(
            State(state_with_stub("{\"answer\":\"yes\"}")),
            HeaderMap::new(),
            Ok(Json(request)),
        )
        .await
        .expect("strict response");
        assert_eq!(response.status(), AxumStatusCode::OK);
        let body = response_json(response).await;
        assert_eq!(
            body["choices"][0]["message"]["content"],
            "{\"answer\":\"yes\"}"
        );
    }

    #[tokio::test]
    async fn strict_json_schema_validates_non_streaming_response_after_reasoning_block() {
        let request = chat_request(json!({
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "answer",
                    "strict": true,
                    "schema": {
                        "type": "object",
                        "properties": {"answer": {"type": "string"}},
                        "required": ["answer"]
                    }
                }
            }
        }));
        let response = chat_completions_handler(
            State(state_with_stub(
                "<think>\nreasoning\n</think>\n\n{\"answer\":\"yes\"}",
            )),
            HeaderMap::new(),
            Ok(Json(request)),
        )
        .await
        .expect("strict response with reasoning");
        assert_eq!(response.status(), AxumStatusCode::OK);
        let body = response_json(response).await;
        assert_eq!(
            body["choices"][0]["message"]["content"],
            "{\"answer\":\"yes\"}"
        );
        assert_eq!(body["choices"][0]["message"]["reasoning"], "\nreasoning\n");
    }

    #[tokio::test]
    async fn strict_json_schema_validates_streaming_final_response() {
        let response = post_json(
            router_with_stub("{\"answer\":\"yes\"}"),
            "/v1/chat/completions",
            json!({
                "model": "stub-model",
                "messages": [{"role": "user", "content": "Return an answer object."}],
                "stream": true,
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "answer",
                        "strict": true,
                        "schema": {
                            "type": "object",
                            "properties": {"answer": {"type": "string"}},
                            "required": ["answer"]
                        }
                    }
                }
            }),
        )
        .await;
        assert_eq!(response.status(), AxumStatusCode::OK);
        let body = response_text(response).await;
        assert!(body.contains("data: [DONE]"), "missing DONE: {body}");
        assert!(
            body.contains("\\\"answer\\\":\\\"yes\\\""),
            "strict streaming content missing: {body}"
        );
        assert!(
            !body.contains("\"error\""),
            "valid strict streaming response should not emit error: {body}"
        );
    }

    #[tokio::test]
    async fn strict_json_schema_validates_streaming_final_response_after_reasoning_block() {
        let response = post_json(
            router_with_stub_stream_chunks(&[
                "<think>\nreasoning",
                "\n</think>\n\n",
                "{\"answer\":\"yes\"}",
            ]),
            "/v1/chat/completions",
            json!({
                "model": "stub-model",
                "messages": [{"role": "user", "content": "Return an answer object."}],
                "stream": true,
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "answer",
                        "strict": true,
                        "schema": {
                            "type": "object",
                            "properties": {"answer": {"type": "string"}},
                            "required": ["answer"]
                        }
                    }
                }
            }),
        )
        .await;
        assert_eq!(response.status(), AxumStatusCode::OK);
        let body = response_text(response).await;
        assert!(body.contains("data: [DONE]"), "missing DONE: {body}");
        assert!(
            body.contains("\\\"answer\\\":\\\"yes\\\""),
            "strict streaming content missing: {body}"
        );
        assert!(
            body.contains(r#""reasoning":"\nreasoning\n""#),
            "strict streaming should keep reasoning separate: {body}"
        );
        assert!(
            !body.contains("\"error\""),
            "valid strict streaming response should not emit error: {body}"
        );
    }

    #[tokio::test]
    async fn strict_json_schema_invalid_streaming_output_emits_error_event() {
        let response = post_json(
            router_with_stub("not json"),
            "/v1/chat/completions",
            json!({
                "model": "stub-model",
                "messages": [{"role": "user", "content": "Return an answer object."}],
                "stream": true,
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "answer",
                        "strict": true,
                        "schema": {
                            "type": "object",
                            "properties": {"answer": {"type": "string"}},
                            "required": ["answer"]
                        }
                    }
                }
            }),
        )
        .await;
        assert_eq!(response.status(), AxumStatusCode::OK);
        let body = response_text(response).await;
        assert!(body.contains("data: [DONE]"), "missing DONE: {body}");
        assert!(
            body.contains("\"type\":\"internal_server_error\""),
            "strict streaming validation failure should emit OpenAI error: {body}"
        );
        assert!(
            body.contains("\"param\":\"response_format.json_schema\""),
            "strict streaming validation error should identify schema param: {body}"
        );
        assert!(
            body.contains("invalid JSON"),
            "strict streaming validation should report invalid JSON: {body}"
        );
        assert!(
            !body.contains("not json"),
            "strict streaming must not emit invalid partial deltas before validation failure: {body}"
        );
    }

    #[tokio::test]
    async fn route_strict_json_schema_supported_schema_passes_100_runs() {
        let request_body = json!({
            "model": "stub-model",
            "messages": [{"role": "user", "content": "Return an answer object."}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "answer",
                    "strict": true,
                    "schema": {
                        "type": "object",
                        "properties": {"answer": {"type": "string"}},
                        "required": ["answer"]
                    }
                }
            }
        });
        let router = router_with_stub("{\"answer\":\"yes\"}");
        for run in 0..100 {
            let response =
                post_json(router.clone(), "/v1/chat/completions", request_body.clone()).await;
            assert_eq!(
                response.status(),
                AxumStatusCode::OK,
                "strict schema run {run} returned non-200"
            );
            let body = response_json(response).await;
            let content = body["choices"][0]["message"]["content"]
                .as_str()
                .unwrap_or("");
            assert_eq!(
                content, "{\"answer\":\"yes\"}",
                "strict schema run {run} returned unexpected content"
            );
            let parsed: serde_json::Value =
                serde_json::from_str(content).expect("strict content JSON");
            assert_eq!(parsed["answer"], "yes");
        }
    }

    #[test]
    fn cache_metrics_use_engine_real_kv_snapshot_when_available() {
        let cache = CacheRuntimeState::default();
        let policy = CachePolicy {
            prefix_cache_enabled: true,
            session_cache_mode: "memory".to_string(),
            session_cache_max_entries: 128,
            session_cache_max_tokens: 4096,
        };
        cache.record_prefix_prompt("alpha beta gamma", &policy);
        cache.record_prefix_prompt("alpha beta delta", &policy);

        let engine_snapshot = json!({
            "position": "real-kv-reuse",
            "source": "llama-family-paged-block-prefix-cache",
            "enabled": true,
            "hits": 7,
            "misses": 3,
            "evictions": 1,
            "saved_prefill_tokens": 64,
            "entries": 5,
            "bytes": 8192,
            "block_size": 16,
            "kv_dtype": "fp16",
            "selected_pipeline_mode": "batch",
            "selected_stage_bridge": "host",
            "stage_count": 2,
        });

        let health = cache.health_json(&policy, Some(&engine_snapshot));
        let prefix = &health["prefix_cache"];
        assert_eq!(prefix["position"], "real-kv-reuse");
        assert_eq!(prefix["source"], "llama-family-paged-block-prefix-cache");
        assert_eq!(prefix["hits"], 7);
        assert_eq!(prefix["misses"], 3);
        assert_eq!(prefix["evictions"], 1);
        assert_eq!(prefix["saved_prefill_tokens"], 64);
        assert_eq!(prefix["entries"], 5);
        assert_eq!(prefix["bytes"], 8192);
        assert_eq!(prefix["block_size"], 16);
        assert_eq!(prefix["kv_dtype"], "fp16");
        assert_eq!(prefix["selected_pipeline_mode"], "batch");
        assert_eq!(prefix["selected_stage_bridge"], "host");
        assert_eq!(prefix["stage_count"], 2);

        let metrics = cache.prometheus_metrics(Some(&engine_snapshot));
        assert!(metrics.contains("ferrum_prefix_cache_hits_total 7\n"));
        assert!(metrics.contains("ferrum_prefix_cache_misses_total 3\n"));
        assert!(metrics.contains("ferrum_prefix_cache_saved_prefill_tokens_total 64\n"));
        assert!(metrics.contains("ferrum_prefix_cache_entries 5\n"));
        assert!(metrics.contains("ferrum_prefix_cache_bytes 8192\n"));
    }

    #[tokio::test]
    async fn strict_json_schema_invalid_model_output_fails_before_response() {
        let request = chat_request(json!({
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "answer",
                    "strict": true,
                    "schema": {
                        "type": "object",
                        "properties": {"answer": {"type": "string"}},
                        "required": ["answer"]
                    }
                }
            }
        }));
        let err = chat_completions_handler(
            State(state_with_stub("not json")),
            HeaderMap::new(),
            Ok(Json(request)),
        )
        .await
        .expect_err("invalid strict response should fail");
        let (status, body) = error_json(err).await;
        assert_eq!(status, AxumStatusCode::INTERNAL_SERVER_ERROR);
        assert_eq!(body["error"]["type"], "internal_server_error");
        assert!(body["error"]["message"]
            .as_str()
            .unwrap()
            .contains("json_schema.strict"));
    }

    #[tokio::test]
    async fn strict_json_schema_does_not_rely_on_markdown_fence_stripping() {
        let request = chat_request(json!({
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "answer",
                    "strict": true,
                    "schema": {
                        "type": "object",
                        "properties": {"answer": {"type": "string"}},
                        "required": ["answer"]
                    }
                }
            }
        }));
        let err = chat_completions_handler(
            State(state_with_stub("```json\n{\"answer\":\"yes\"}\n```")),
            HeaderMap::new(),
            Ok(Json(request)),
        )
        .await
        .expect_err("strict schema should fail fenced JSON instead of repairing it");
        let (status, body) = error_json(err).await;
        assert_eq!(status, AxumStatusCode::INTERNAL_SERVER_ERROR);
        assert_eq!(body["error"]["type"], "internal_server_error");
        assert!(body["error"]["message"]
            .as_str()
            .unwrap()
            .contains("json_schema.strict: invalid JSON"));
    }
}
