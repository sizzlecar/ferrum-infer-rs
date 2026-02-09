//! # Ferrum Server
//!
//! HTTP API server abstractions for LLM inference services.
//!
//! ## Overview
//!
//! This module defines the core traits for implementing HTTP API servers
//! that can serve LLM inference requests with OpenAI API compatibility.
//!
//! ## Design Principles
//!
//! - **Framework Agnostic**: Abstract interfaces that work with any HTTP framework
//! - **OpenAI Compatible**: Support for OpenAI Chat Completions API
//! - **Middleware Support**: Pluggable middleware for auth, logging, rate limiting
//! - **Streaming Support**: Server-Sent Events for streaming responses
//! - **Monitoring**: Built-in metrics and health checks

pub mod axum_server;
pub mod middleware;
pub mod openai;
pub mod traits;
pub mod types;

// Re-exports
pub use traits::{
    AuthProvider, HttpServer, MiddlewareStack, RateLimiter as ServerRateLimiter, RequestHandler,
    ResponseBuilder, StreamingHandler,
};

pub use types::{
    ApiVersion, Headers, HealthStatus, HttpMethod, HttpRequest, HttpResponse, RequestContext,
    ServerConfig, ServerMetrics, StatusCode,
};

pub use openai::{
    ChatCompletionsRequest, ChatCompletionsResponse, ChatMessage, CompletionsRequest,
    CompletionsResponse, ModelListResponse, OpenAiError, OpenAiErrorType,
};

pub use middleware::{
    AuthConfig, CompressionConfig, CorsConfig, LoggingConfig, MiddlewareConfig,
    RateLimitConfig as MiddlewareRateLimitConfig,
};

pub use axum_server::AxumServer;
