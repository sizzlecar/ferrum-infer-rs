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

pub mod traits;
pub mod types;
pub mod openai;
pub mod middleware;

// Re-exports
pub use traits::{
    HttpServer, RequestHandler, ResponseBuilder, StreamingHandler,
    MiddlewareStack, AuthProvider, RateLimiter as ServerRateLimiter
};

pub use types::{
    ServerConfig, HttpRequest, HttpResponse, HttpMethod,
    StatusCode, Headers, RequestContext, ServerMetrics,
    HealthStatus, ApiVersion
};

pub use openai::{
    ChatCompletionsRequest, ChatCompletionsResponse, ChatMessage,
    CompletionsRequest, CompletionsResponse, ModelListResponse,
    OpenAiError, OpenAiErrorType
};

pub use middleware::{
    MiddlewareConfig, AuthConfig, CorsConfig, LoggingConfig,
    CompressionConfig, RateLimitConfig as MiddlewareRateLimitConfig
};
