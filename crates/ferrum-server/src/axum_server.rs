//! Axum-based HTTP server implementation for Ferrum
//!
//! This module provides a concrete implementation of the HttpServer trait
//! using the Axum web framework, with full OpenAI API compatibility.

use crate::{openai::*, traits::HttpServer, types::*};
use async_trait::async_trait;
use axum::{
    extract::State,
    http::StatusCode as AxumStatusCode,
    response::{sse::Event, IntoResponse, Response, Sse},
    routing::{get, post},
    Json, Router,
};
use ferrum_interfaces::engine::InferenceEngine;
use ferrum_types::{
    FerrumError as Error, FinishReason, InferenceRequest, ModelId, Priority, RequestId,
    SamplingParams,
};
use std::sync::Arc;
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

/// Axum-based server implementation
pub struct AxumServer {
    engine: Arc<dyn InferenceEngine + Send + Sync>,
    config: ServerConfig,
}

impl AxumServer {
    /// Create a new Axum server
    pub fn new(engine: Arc<dyn InferenceEngine + Send + Sync>) -> Self {
        Self {
            engine,
            config: ServerConfig::default(),
        }
    }

    /// Build the router with all routes
    fn build_router(&self) -> Router {
        let app_state = AppState {
            engine: self.engine.clone(),
        };

        Router::new()
            // OpenAI API routes
            .route("/v1/chat/completions", post(chat_completions_handler))
            .route("/v1/completions", post(completions_handler))
            .route("/v1/embeddings", post(embeddings_handler))
            .route("/v1/audio/transcriptions", post(transcriptions_handler))
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

/// Application state shared across handlers
#[derive(Clone)]
struct AppState {
    engine: Arc<dyn InferenceEngine + Send + Sync>,
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
    Json(request): Json<ChatCompletionsRequest>,
) -> std::result::Result<Response, ServerError> {
    let span = span!(Level::INFO, "chat_completions", model = %request.model);
    let _enter = span.enter();

    info!(
        "Received chat completions request for model: {}",
        request.model
    );
    debug!("Request: {:?}", request);

    // Convert OpenAI request to internal format
    let inference_request =
        convert_chat_request(&request).map_err(|e| ServerError::BadRequest(e.to_string()))?;

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
    let engine = state.engine.clone();
    let request_id = Uuid::new_v4().to_string();
    let prompt_tokens = inference_request.prompt.split_whitespace().count() as u32;

    tokio::spawn(async move {
        let mut current_text = String::new();
        let mut token_count = 0;
        let max_tokens = openai_request.max_tokens.unwrap_or(100);

        match engine.infer_stream(inference_request).await {
            Ok(mut stream) => {
                while let Some(result) = stream.next().await {
                    match result {
                        Ok(chunk) => {
                            current_text.push_str(&chunk.text);
                            token_count += 1;

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

                            // Check stopping conditions
                            if token_count >= max_tokens || chunk.finish_reason.is_some() {
                                // Send final chunk
                                let final_chunk = ChatCompletionsResponse {
                                    id: request_id.clone(),
                                    object: "chat.completion.chunk".to_string(),
                                    created: chrono::Utc::now().timestamp() as u64,
                                    model: openai_request.model.clone(),
                                    choices: vec![ChatChoice {
                                        index: 0,
                                        message: None,
                                        delta: None,
                                        finish_reason: chunk
                                            .finish_reason
                                            .as_ref()
                                            .map(finish_reason_to_string),
                                    }],
                                    usage: Some(Usage {
                                        prompt_tokens: prompt_tokens,
                                        completion_tokens: token_count,
                                        total_tokens: prompt_tokens + token_count,
                                    }),
                                };

                                let final_event = Event::default()
                                    .json_data(&final_chunk)
                                    .unwrap_or_else(|_| Event::default().data("error"));
                                let _ = tx.send(Ok(final_event));
                                let _ = tx.send(Ok(Event::default().data("[DONE]")));
                                break;
                            }
                        }
                        Err(e) => {
                            error!("Stream generation error: {}", e);
                            let _ = tx.send(Ok(
                                Event::default().data(&format!("{{\"error\": \"{}\"}}", e))
                            ));
                            break;
                        }
                    }
                }
            }
            Err(e) => {
                error!("Failed to start streaming: {}", e);
                let _ = tx.send(Ok(
                    Event::default().data(&format!("{{\"error\": \"{}\"}}", e))
                ));
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

    let prompt_tokens = inference_request.prompt.split_whitespace().count() as u32;
    match state.engine.infer(inference_request).await {
        Ok(output) => {
            let response = ChatCompletionsResponse {
                id: Uuid::new_v4().to_string(),
                object: "chat.completion".to_string(),
                created: chrono::Utc::now().timestamp() as u64,
                model: openai_request.model,
                choices: vec![ChatChoice {
                    index: 0,
                    message: Some(ChatMessage {
                        role: MessageRole::Assistant,
                        content: output.text,
                        name: None,
                    }),
                    delta: None,
                    finish_reason: Some(finish_reason_to_string(&output.finish_reason)),
                }],
                usage: Some(Usage {
                    prompt_tokens: prompt_tokens,
                    completion_tokens: output.tokens.len() as u32,
                    total_tokens: prompt_tokens + output.tokens.len() as u32,
                }),
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
fn convert_chat_request(
    request: &ChatCompletionsRequest,
) -> ferrum_types::Result<InferenceRequest> {
    // Combine all messages into a single prompt for MVP
    let prompt = request
        .messages
        .iter()
        .map(|msg| format!("{}: {}", msg.role.to_string(), msg.content))
        .collect::<Vec<_>>()
        .join("\n");

    Ok(InferenceRequest {
        id: RequestId(Uuid::new_v4()),
        model_id: ModelId(request.model.clone()),
        prompt,
        sampling_params: SamplingParams {
            max_tokens: request.max_tokens.unwrap_or(100) as usize,
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
                _ => ferrum_types::ResponseFormat::Text,
            },
        },
        stream: request.stream.unwrap_or(false),
        priority: Priority::Normal, // Default priority
        client_id: None,
        session_id: None,
        created_at: chrono::Utc::now(),
        metadata: std::collections::HashMap::new(),
    })
}

/// Other handlers
async fn completions_handler(
    State(_state): State<AppState>,
    Json(_request): Json<CompletionsRequest>,
) -> std::result::Result<Response, ServerError> {
    // TODO: Implement legacy completions endpoint
    Err(ServerError::NotImplemented(
        "Legacy completions not implemented in MVP".to_string(),
    ))
}

/// Embeddings handler — text and image embedding via OpenAI-compatible API.
async fn embeddings_handler(
    State(state): State<AppState>,
    Json(request): Json<EmbeddingsRequest>,
) -> std::result::Result<Response, ServerError> {
    let span = span!(Level::INFO, "embeddings", model = %request.model);
    let _enter = span.enter();

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
        return Err(ServerError::BadRequest("Empty input".to_string()));
    }

    let mut data = Vec::with_capacity(items.len());
    let mut total_tokens = 0u32;

    for (idx, item) in items.iter().enumerate() {
        let embedding = if let Some(ref image) = item.image {
            state
                .engine
                .embed_image(image)
                .await
                .map_err(|e| ServerError::InternalError(format!("embed_image: {e}")))?
        } else if let Some(ref text) = item.text {
            total_tokens += text.len() as u32;
            state
                .engine
                .embed_text(text)
                .await
                .map_err(|e| ServerError::InternalError(format!("embed_text: {e}")))?
        } else {
            return Err(ServerError::BadRequest(
                "Each input must have either 'text' or 'image'".to_string(),
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

/// Audio transcription handler (OpenAI-compatible multipart form).
async fn transcriptions_handler(
    State(state): State<AppState>,
    mut multipart: axum::extract::Multipart,
) -> std::result::Result<Response, ServerError> {
    let span = span!(Level::INFO, "transcription");
    let _enter = span.enter();

    let mut file_data: Option<Vec<u8>> = None;
    let mut language: Option<String> = None;

    while let Some(field) = multipart
        .next_field()
        .await
        .map_err(|e| ServerError::BadRequest(format!("multipart: {e}")))?
    {
        let name = field.name().unwrap_or("").to_string();
        match name.as_str() {
            "file" => {
                file_data = Some(
                    field
                        .bytes()
                        .await
                        .map_err(|e| ServerError::BadRequest(format!("read file: {e}")))?
                        .to_vec(),
                );
            }
            "language" => {
                language = field.text().await.ok().filter(|s| !s.is_empty());
            }
            _ => {} // ignore model, response_format, etc. for now
        }
    }

    let data = file_data.ok_or_else(|| ServerError::BadRequest("missing 'file' field".into()))?;

    let text = state
        .engine
        .transcribe_bytes(&data, language.as_deref())
        .await
        .map_err(|e| ServerError::InternalError(format!("transcribe: {e}")))?;

    Ok(Json(TranscriptionResponse { text }).into_response())
}

async fn models_handler(
    State(state): State<AppState>,
) -> std::result::Result<Response, ServerError> {
    let status = state.engine.status().await;
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
    let engine_status = state.engine.status().await;
    let scheduler_metrics = state.engine.metrics();

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
        }
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
    InternalError(String),
    NotImplemented(String),
}

impl IntoResponse for ServerError {
    fn into_response(self) -> Response {
        let (status, message) = match self {
            ServerError::BadRequest(msg) => (AxumStatusCode::BAD_REQUEST, msg),
            ServerError::InternalError(msg) => (AxumStatusCode::INTERNAL_SERVER_ERROR, msg),
            ServerError::NotImplemented(msg) => (AxumStatusCode::NOT_IMPLEMENTED, msg),
        };

        let error = OpenAiError {
            error: OpenAiErrorDetail {
                message,
                error_type: "server_error".to_string(),
                param: None,
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
        }
    }
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
