//! Axum-based HTTP server implementation for Ferrum
//!
//! This module provides a concrete implementation of the HttpServer trait
//! using the Axum web framework, with OpenAI-shaped endpoint compatibility.

use crate::{
    chat_template::{
        render_chat_prompt_with_model_template, render_chat_prompt_with_tools, ModelChatTemplate,
    },
    openai::*,
    traits::HttpServer,
    types::*,
};
use async_trait::async_trait;
use axum::{
    extract::{multipart::MultipartRejection, rejection::JsonRejection, State},
    http::StatusCode as AxumStatusCode,
    response::{sse::Event, IntoResponse, Response, Sse},
    routing::{get, post},
    Json, Router,
};
use ferrum_interfaces::engine::{EmbedEngine, LlmInferenceEngine, TranscribeEngine, TtsEngine};
use ferrum_types::{
    EngineMetrics, EngineStatus, FerrumConfigBuilder, FerrumError as Error, FinishReason,
    InferenceRequest, InferenceResponse, ModelId, Priority, RequestId, ResolvedFerrumConfig,
    RuntimeConfigSnapshot, SamplingParams, TokenUsage,
};
use std::{collections::HashMap, sync::Arc};
use tokio::sync::mpsc;
use tokio_stream::StreamExt;
use tower::ServiceBuilder;
use tower_http::{cors::CorsLayer, trace::TraceLayer};
use tracing::{debug, error, info, span, Level};
use uuid::Uuid;

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

    /// Build the router with all routes
    fn build_router(&self) -> Router {
        let app_state = self.state.clone();

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

#[async_trait]
impl HttpServer for AxumServer {
    async fn start(&self, config: &ServerConfig) -> ferrum_types::Result<()> {
        let addr = format!("{}:{}", config.host, config.port);
        info!("Starting Axum server on {}", addr);

        let app = self.build_router();
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
    request: std::result::Result<Json<ChatCompletionsRequest>, JsonRejection>,
) -> std::result::Result<Response, ServerError> {
    let Json(request) = request.map_err(|e| {
        ServerError::invalid_request(format!("invalid chat completions request: {e}"), None)
    })?;

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

    // Convert OpenAI request to internal format
    let template_model_id = state
        .status()
        .await
        .loaded_models
        .first()
        .map(ToString::to_string)
        .unwrap_or_else(|| request.model.clone());
    let inference_request = convert_chat_request_with_template_model(
        &request,
        &template_model_id,
        state.prompt_template.as_deref(),
    )
    .map_err(|e| ServerError::BadRequest(e.to_string()))?;

    // Check if streaming is requested
    if request.stream.unwrap_or(false) {
        handle_chat_completions_stream(state, request, inference_request).await
    } else {
        handle_chat_completions_sync(state, request, inference_request).await
    }
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
    let buffer_strict_json_schema_stream = strict_json_schema_string(&openai_request)?.is_some();
    let stream_api_request = api_chat_request(&openai_request);
    let buffer_structured_api_stream =
        ferrum_types::chat_api_may_emit_tool_or_function_call(&stream_api_request);
    let buffer_stream_output = buffer_strict_json_schema_stream || buffer_structured_api_stream;

    tokio::spawn(async move {
        let mut current_text = String::new();
        let mut token_count = 0;
        let max_tokens = chat_completion_max_tokens(&openai_request);

        match engine.infer_stream(inference_request).await {
            Ok(mut stream) => {
                while let Some(result) = stream.next().await {
                    match result {
                        Ok(chunk) => {
                            if chunk.token.is_some() {
                                token_count += 1;
                            }
                            if !chunk.text.is_empty() {
                                current_text.push_str(&chunk.text);

                                if !buffer_stream_output {
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
                                                content: chunk.text.clone(),
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

                            if token_count >= max_tokens || chunk.finish_reason.is_some() {
                                let usage = chunk.usage.as_ref().map(openai_usage_from_token_usage);
                                if let Err(e) = validate_strict_json_schema_response(
                                    &openai_request,
                                    &current_text,
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
                                        ferrum_types::chat_api_response_from_generated_text(
                                            &stream_api_request,
                                            &current_text,
                                        )
                                    }
                                    _ => None,
                                };

                                if let Some(chat_response) = structured_chat_response.as_ref() {
                                    let response_chunk = ChatCompletionsResponse {
                                        id: request_id.clone(),
                                        object: "chat.completion.chunk".to_string(),
                                        created: chrono::Utc::now().timestamp() as u64,
                                        model: openai_request.model.clone(),
                                        choices: vec![ChatChoice {
                                            index: 0,
                                            message: None,
                                            delta: Some(openai_chat_delta_from_api(
                                                &chat_response.message,
                                            )),
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
                                                content: current_text.clone(),
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
                                            name: None,
                                            tool_calls: None,
                                            tool_call_id: None,
                                            function_call: None,
                                        }),
                                        finish_reason: structured_chat_response
                                            .as_ref()
                                            .and_then(|response| response.finish_reason.clone())
                                            .or_else(|| {
                                                chunk
                                                    .finish_reason
                                                    .as_ref()
                                                    .map(finish_reason_to_string)
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
            }
            Err(e) => {
                error!("Failed to start streaming: {}", e);
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
    let sse_stream = Sse::new(stream);

    Ok(sse_stream.into_response())
}

/// Handle non-streaming chat completions
async fn handle_chat_completions_sync(
    state: AppState,
    openai_request: ChatCompletionsRequest,
    inference_request: InferenceRequest,
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
            let content = strip_trailing_stop(&after_fence, &stop_sequences);
            let mut message = ChatMessage {
                role: MessageRole::Assistant,
                content,
                name: None,
                tool_calls: None,
                tool_call_id: None,
                function_call: None,
            };
            let mut openai_finish_reason = finish_reason_to_string(&finish_reason);
            let structured_chat_response = match api_response.as_ref() {
                Some(ferrum_types::ApiResponse::Chat(chat_response)) => Some(chat_response.clone()),
                _ => match request_chat_api.as_ref() {
                    Some(chat_request) => ferrum_types::chat_api_response_from_generated_text(
                        chat_request,
                        &message.content,
                    ),
                    _ => None,
                },
            };
            if let Some(chat_response) = structured_chat_response.as_ref() {
                message = openai_chat_message_from_api(&chat_response.message);
                if let Some(reason) = &chat_response.finish_reason {
                    openai_finish_reason = reason.clone();
                }
            }
            validate_strict_json_schema_response(&openai_request, &message.content)?;
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
            Err(ServerError::InternalError(e.to_string()))
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
    let tools = request.tools.as_deref().unwrap_or_default();
    let functions = request.functions.as_deref().unwrap_or_default();
    let prompt = if tools.is_empty() && functions.is_empty() {
        render_chat_prompt_with_model_template(&request.messages, template_model_id, model_template)
    } else {
        render_chat_prompt_with_tools(
            &request.messages,
            template_model_id,
            tools,
            request.tool_choice.as_ref(),
            functions,
            request.function_call.as_ref(),
        )
    };
    let mut metadata = HashMap::new();
    metadata.insert(
        "openai_messages".to_string(),
        serde_json::to_value(&request.messages)?,
    );
    if let Some(tools) = &request.tools {
        metadata.insert("openai_tools".to_string(), serde_json::to_value(tools)?);
    }
    if let Some(tool_choice) = &request.tool_choice {
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

    Ok(InferenceRequest {
        id: RequestId(Uuid::new_v4()),
        model_id: ModelId(request.model.clone()),
        prompt,
        sampling_params: SamplingParams {
            max_tokens: chat_completion_max_tokens(request) as usize,
            temperature: request.temperature.unwrap_or(1.0),
            top_p: request.top_p.unwrap_or(1.0),
            top_k: None, // OpenAI doesn't use top-k
            repetition_penalty: 1.0,
            presence_penalty: request.presence_penalty.unwrap_or(0.0),
            frequency_penalty: request.frequency_penalty.unwrap_or(0.0),
            stop_sequences: request.stop.clone().unwrap_or_default(),
            seed: request.seed,
            min_p: None,
            tfs: None,
            typical_p: None,
            mirostat: None,
            response_format: match &request.response_format {
                Some(rf) if rf.format_type == "json_object" => {
                    ferrum_types::ResponseFormat::JsonObject
                }
                Some(rf)
                    if rf.format_type == "json_schema"
                        && rf
                            .json_schema
                            .as_ref()
                            .and_then(|schema| schema.strict)
                            .unwrap_or(false) =>
                {
                    // OpenAI wraps the actual schema one level deeper.
                    match rf.json_schema.as_ref().and_then(|js| js.schema.as_ref()) {
                        Some(schema) => match serde_json::to_string(schema) {
                            Ok(s) => ferrum_types::ResponseFormat::JsonSchema(s),
                            Err(_) => ferrum_types::ResponseFormat::Text,
                        },
                        None => ferrum_types::ResponseFormat::Text,
                    }
                }
                _ => ferrum_types::ResponseFormat::Text,
            },
        },
        stream: request.stream.unwrap_or(false),
        priority: Priority::Normal, // Default priority
        client_id: None,
        session_id: None,
        created_at: chrono::Utc::now(),
        api_request: Some(ferrum_types::ApiRequest::Chat(api_chat_request(request))),
        metadata,
    })
}

fn api_chat_request(request: &ChatCompletionsRequest) -> ferrum_types::ApiChatRequest {
    ferrum_types::ApiChatRequest {
        messages: request.messages.iter().map(api_chat_message).collect(),
        tools: request
            .tools
            .as_deref()
            .unwrap_or_default()
            .iter()
            .map(api_tool)
            .collect(),
        tool_choice: request.tool_choice.as_ref().map(api_tool_choice),
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
        .unwrap_or(100)
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
                return Err(ServerError::unsupported_feature(
                    "required tool_choice is not supported",
                    Some("tool_choice"),
                ));
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
        | ServerError::BadRequest(message)
        | ServerError::NotImplemented(message)
        | ServerError::ServiceUnavailable(message)
        | ServerError::InvalidRequest { message, .. }
        | ServerError::UnsupportedFeature { message, .. } => message,
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
            max_tokens: request.max_tokens.unwrap_or(100) as usize,
            temperature: request.temperature.unwrap_or(1.0),
            top_p: request.top_p.unwrap_or(1.0),
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
        metadata: HashMap::new(),
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
            let mut text = strip_trailing_stop(&output_text, &stop_sequences);
            let mut openai_finish_reason = finish_reason_to_string(&finish_reason);
            if let Some(ferrum_types::ApiResponse::Completion(completion_response)) =
                api_response.as_ref()
            {
                text = strip_trailing_stop(&completion_response.text, &stop_sequences);
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
        let mut token_count = 0;
        let max_tokens = openai_request.max_tokens.unwrap_or(100);
        match engine.infer_stream(inference_request).await {
            Ok(mut stream) => {
                while let Some(result) = stream.next().await {
                    match result {
                        Ok(chunk) => {
                            if chunk.token.is_some() {
                                token_count += 1;
                            }
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
                            if token_count >= max_tokens || chunk.finish_reason.is_some() {
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
    let inference_request = convert_completion_request(&request);
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
    let data = status
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
    let auto_config = match state.auto_config.clone() {
        Some(auto_config) => serde_json::to_value(auto_config).unwrap_or_else(|err| {
            serde_json::json!({
                "schema_version": 1,
                "error": format!("failed to serialize startup auto config: {err}"),
            })
        }),
        None => match FerrumConfigBuilder::new(runtime_config.clone()).resolve() {
            Ok(auto_config) => serde_json::to_value(auto_config).unwrap_or_else(|err| {
                serde_json::json!({
                    "schema_version": 1,
                    "error": format!("failed to serialize runtime auto config: {err}"),
                })
            }),
            Err(err) => serde_json::json!({
                "schema_version": 1,
                "error": err.to_string(),
            }),
        },
    };

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
        },
        "config": runtime_config,
        "auto_config": auto_config,
    });

    Ok(Json(health).into_response())
}

/// Prometheus metrics endpoint — returns metrics in Prometheus text format.
async fn metrics_handler() -> std::result::Result<Response, ServerError> {
    let body = match PROM_HANDLE.get() {
        Some(handle) => handle.render(),
        None => "# Prometheus recorder not initialized\n".to_string(),
    };

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
    BadRequest(String),
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
            ServerError::BadRequest(msg) => (
                AxumStatusCode::BAD_REQUEST,
                msg,
                "invalid_request_error",
                None,
            ),
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

/// Strip a single trailing occurrence of any user-supplied stop sequence
/// from `text`. Mirrors OpenAI's convention that `stop` strings act as
/// boundaries that are NOT included in the returned completion. We only
/// strip from the suffix (not all occurrences) because a stop sequence
/// the model produced legitimately mid-response shouldn't be redacted.
fn strip_trailing_stop(text: &str, stops: &[String]) -> String {
    let mut out = text.to_string();
    let trimmed = out.trim_end();
    for stop in stops {
        if stop.is_empty() {
            continue;
        }
        if let Some(prefix) = trimmed.strip_suffix(stop.as_str()) {
            out = prefix.to_string();
            break;
        }
    }
    out
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

    struct StubLlm {
        config: EngineConfig,
        text: String,
        stream_usage: Option<TokenUsage>,
        api_response: Option<ferrum_types::ApiResponse>,
    }

    impl StubLlm {
        fn new(text: &str) -> Self {
            let mut config = EngineConfig::default();
            config.model.model_id = ModelId::new("stub-model");
            Self {
                config,
                text: text.to_string(),
                stream_usage: Some(TokenUsage::new(5, 1)),
                api_response: None,
            }
        }

        fn without_stream_usage(text: &str) -> Self {
            Self {
                stream_usage: None,
                ..Self::new(text)
            }
        }

        fn with_api_response(text: &str, api_response: ferrum_types::ApiResponse) -> Self {
            Self {
                api_response: Some(api_response),
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
            config.model.model_id = ModelId::new("capture-model");
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

    #[tokio::test]
    async fn route_health_includes_runtime_config_snapshot() {
        let response = get(router_with_stub("ok"), "/health").await;
        assert_eq!(response.status(), AxumStatusCode::OK);
        let body = response_json(response).await;
        assert_eq!(body["status"], "healthy");
        assert!(body["config"]["entries"].is_array(), "body: {body}");
        assert_eq!(body["auto_config"]["schema_version"], 1);
        assert!(
            body["auto_config"]["decisions"].is_array() || body["auto_config"]["error"].is_string(),
            "body: {body}"
        );
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
        assert_eq!(response.status(), AxumStatusCode::OK);
        let body = response_json(response).await;
        assert_eq!(body["choices"][0]["finish_reason"], "stop");
        assert_eq!(
            body["choices"][0]["message"]["content"],
            r#"{"name":"calendar","arguments":{}}"#
        );
        assert!(body["choices"][0]["message"]["tool_calls"].is_null());
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
            body.contains(r#""content":"{\"name\":\"calendar\",\"arguments\":{}}""#),
            "unselected tool JSON should stream as ordinary content: {body}"
        );
        assert!(
            body.contains(r#""finish_reason":"stop""#),
            "unselected tool JSON should keep normal stop finish: {body}"
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
        assert_eq!(request.sampling_params.stop_sequences, vec!["<END>"]);
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

    #[tokio::test]
    async fn route_rejects_unsupported_tool_and_function_selection() {
        for (extra, param) in [
            (
                json!({
                    "tools": [{
                        "type": "function",
                        "function": {"name": "weather", "parameters": {"type": "object"}}
                    }],
                    "tool_choice": "required"
                }),
                "tool_choice",
            ),
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
        let err = chat_completions_handler(State(state_with_stub("unused")), Ok(Json(request)))
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
            let err = chat_completions_handler(State(state_with_stub("unused")), Ok(Json(request)))
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
        let response = chat_completions_handler(State(state_with_stub("ok")), Ok(Json(request)))
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
        let response = chat_completions_handler(State(state_with_stub("ok")), Ok(Json(request)))
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
    fn assistant_tool_call_serializes_openai_shape() {
        let message = ChatMessage {
            role: MessageRole::Assistant,
            content: String::new(),
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
        let err = chat_completions_handler(State(state_with_stub("unused")), Ok(Json(request)))
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
    async fn json_object_remains_best_effort_not_strict_validation() {
        let request = chat_request(json!({
            "response_format": {"type": "json_object"}
        }));
        let response =
            chat_completions_handler(State(state_with_stub("not json")), Ok(Json(request)))
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
        let err = chat_completions_handler(State(state_with_stub("unused")), Ok(Json(request)))
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
        let err = chat_completions_handler(State(state_with_stub("unused")), Ok(Json(request)))
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
        let err = chat_completions_handler(State(state_with_stub("not json")), Ok(Json(request)))
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
