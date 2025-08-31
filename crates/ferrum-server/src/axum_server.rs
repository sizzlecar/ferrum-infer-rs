//! Axum-based HTTP server implementation for Ferrum
//!
//! This module provides a concrete implementation of the HttpServer trait
//! using the Axum web framework, with full OpenAI API compatibility.

use crate::{
    openai::*,
    traits::{HttpServer, StreamSender, StreamingHandler},
    types::*,
};
use async_trait::async_trait;
use axum::{
    extract::State,
    http::StatusCode as AxumStatusCode,
    response::{sse::Event, IntoResponse, Response, Sse},
    routing::{get, post},
    Json, Router,
};
use ferrum_core::{
    Error, FinishReason, InferenceEngine, InferenceRequest, ModelId, Priority, RequestId,
    SamplingParams,
};
use ferrum_engine::Engine;
use std::sync::Arc;
use tokio::sync::mpsc;
use tokio_stream::{wrappers::UnboundedReceiverStream, StreamExt};
use tower::ServiceBuilder;
use tower_http::{cors::CorsLayer, trace::TraceLayer};
use tracing::{debug, error, info, span, Level};
use uuid::Uuid;

/// Axum-based server implementation
pub struct AxumServer {
    engine: Arc<Engine>,
    config: ServerConfig,
    app: Option<Router>,
}

impl AxumServer {
    /// Create a new Axum server
    pub fn new(engine: Arc<Engine>) -> Self {
        Self {
            engine,
            config: ServerConfig::default(),
            app: None,
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
            .route("/v1/models", get(models_handler))
            // Health check
            .route("/health", get(health_handler))
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
    engine: Arc<Engine>,
}

/// SSE stream sender implementation
struct SseSender {
    sender: mpsc::UnboundedSender<std::result::Result<Event, axum::Error>>,
}

#[async_trait]
impl StreamSender for SseSender {
    async fn send_chunk(&self, chunk: &str) -> ferrum_core::Result<()> {
        self.sender
            .send(Ok(Event::default().data(chunk)))
            .map_err(|_| Error::internal("Stream closed"))?;
        Ok(())
    }

    async fn send_json(&self, data: &serde_json::Value) -> ferrum_core::Result<()> {
        let chunk = serde_json::to_string(data)
            .map_err(|e| Error::internal(format!("JSON serialization failed: {}", e)))?;
        self.send_chunk(&chunk).await
    }

    async fn close(&self) -> ferrum_core::Result<()> {
        // Channel will be closed when sender is dropped
        Ok(())
    }

    fn is_closed(&self) -> bool {
        self.sender.is_closed()
    }
}

#[async_trait]
impl HttpServer for AxumServer {
    async fn start(&self, config: &ServerConfig) -> ferrum_core::Result<()> {
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

    async fn stop(&self, _timeout: std::time::Duration) -> ferrum_core::Result<()> {
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
                            let chunk = ChatCompletionsResponse {
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
                                .json_data(&chunk)
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
fn convert_chat_request(request: &ChatCompletionsRequest) -> ferrum_core::Result<InferenceRequest> {
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
        },
        stream: request.stream.unwrap_or(false),
        priority: Priority::Normal, // Default priority
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

async fn models_handler(
    State(_state): State<AppState>,
) -> std::result::Result<Response, ServerError> {
    let models = ModelListResponse {
        object: "list".to_string(),
        data: vec![crate::openai::ModelInfo {
            id: "TinyLlama-1.1B-Chat-v1.0".to_string(),
            object: "model".to_string(),
            created: chrono::Utc::now().timestamp() as u64,
            owned_by: "ferrum".to_string(),
            permission: vec![],
            root: None,
            parent: None,
        }],
    };

    Ok(Json(models).into_response())
}

async fn health_handler() -> std::result::Result<Response, ServerError> {
    let health = serde_json::json!({
        "status": "healthy",
        "timestamp": chrono::Utc::now().to_rfc3339(),
        "version": env!("CARGO_PKG_VERSION")
    });

    Ok(Json(health).into_response())
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
    NotFound(String),
    InternalError(String),
    NotImplemented(String),
}

impl IntoResponse for ServerError {
    fn into_response(self) -> Response {
        let (status, message) = match self {
            ServerError::BadRequest(msg) => (AxumStatusCode::BAD_REQUEST, msg),
            ServerError::NotFound(msg) => (AxumStatusCode::NOT_FOUND, msg),
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
    }
}
