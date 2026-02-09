//! Core server traits
//!
//! This module defines the abstract interfaces for HTTP server implementation,
//! middleware management, and request handling.

use crate::types::*;
use async_trait::async_trait;
use ferrum_types::{InferenceRequest, Result};
use std::time::Duration;

/// Main HTTP server trait
#[async_trait]
pub trait HttpServer: Send + Sync {
    /// Start the server
    async fn start(&self, config: &ServerConfig) -> Result<()>;

    /// Stop the server gracefully
    async fn stop(&self, timeout: Duration) -> Result<()>;

    /// Check if server is running
    fn is_running(&self) -> bool;

    /// Get server address
    fn address(&self) -> Option<std::net::SocketAddr>;

    /// Register a request handler
    fn register_handler(
        &mut self,
        path: &str,
        method: HttpMethod,
        handler: Box<dyn RequestHandler>,
    );

    /// Register middleware
    fn register_middleware(&mut self, middleware: Box<dyn Middleware>);

    /// Get server metrics
    fn get_metrics(&self) -> ServerMetrics;

    /// Health check
    async fn health_check(&self) -> HealthStatus;
}

/// Request handler trait for processing HTTP requests
#[async_trait]
pub trait RequestHandler: Send + Sync {
    /// Handle an HTTP request
    async fn handle(&self, request: HttpRequest, context: RequestContext) -> Result<HttpResponse>;

    /// Get handler name for debugging
    fn name(&self) -> &str;

    /// Check if handler supports the request
    fn can_handle(&self, request: &HttpRequest) -> bool;
}

/// Response builder trait for constructing HTTP responses
pub trait ResponseBuilder: Send + Sync {
    /// Create a successful response
    fn ok(&self, body: serde_json::Value) -> HttpResponse;

    /// Create an error response
    fn error(&self, code: StatusCode, message: &str) -> HttpResponse;

    /// Create a streaming response
    fn streaming(&self, content_type: &str) -> HttpResponse;

    /// Create a response with custom headers
    fn with_headers(&self, body: serde_json::Value, headers: Headers) -> HttpResponse;

    /// Create a redirect response
    fn redirect(&self, location: &str, permanent: bool) -> HttpResponse;
}

/// Streaming response handler
#[async_trait]
pub trait StreamingHandler: Send + Sync {
    /// Handle streaming inference request
    async fn handle_stream(
        &self,
        request: InferenceRequest,
        sender: Box<dyn StreamSender>,
    ) -> Result<()>;

    /// Get streaming configuration
    fn stream_config(&self) -> &StreamConfig;
}

/// Stream sender for sending chunks
#[async_trait]
pub trait StreamSender: Send + Sync {
    /// Send a chunk
    async fn send_chunk(&self, chunk: &str) -> Result<()>;

    /// Send JSON chunk
    async fn send_json(&self, data: &serde_json::Value) -> Result<()>;

    /// Close the stream
    async fn close(&self) -> Result<()>;

    /// Check if stream is closed
    fn is_closed(&self) -> bool;
}

/// Middleware trait for request/response processing
#[async_trait]
pub trait Middleware: Send + Sync {
    /// Process request before handler
    async fn before_request(
        &self,
        request: &mut HttpRequest,
        context: &mut RequestContext,
    ) -> Result<()>;

    /// Process response after handler
    async fn after_response(
        &self,
        request: &HttpRequest,
        response: &mut HttpResponse,
        context: &RequestContext,
    ) -> Result<()>;

    /// Handle middleware errors
    async fn on_error(
        &self,
        error: &ferrum_types::FerrumError,
        context: &RequestContext,
    ) -> Option<HttpResponse>;

    /// Get middleware name
    fn name(&self) -> &str;

    /// Get middleware priority (lower numbers run first)
    fn priority(&self) -> i32;
}

/// Middleware stack management
pub trait MiddlewareStack: Send + Sync {
    /// Add middleware to stack
    fn add(&mut self, middleware: Box<dyn Middleware>);

    /// Remove middleware by name
    fn remove(&mut self, name: &str) -> bool;

    /// Get middleware by name
    fn get(&self, name: &str) -> Option<&dyn Middleware>;

    /// Clear all middleware
    fn clear(&mut self);

    /// Get middleware count
    fn len(&self) -> usize;
}

/// Authentication provider trait
#[async_trait]
pub trait AuthProvider: Send + Sync {
    /// Authenticate a request
    async fn authenticate(&self, request: &HttpRequest) -> Result<AuthResult>;

    /// Validate API key
    async fn validate_api_key(&self, api_key: &str) -> Result<ClientInfo>;

    /// Validate JWT token
    async fn validate_jwt(&self, token: &str) -> Result<TokenClaims>;

    /// Get authentication scheme
    fn scheme(&self) -> AuthScheme;
}

/// Rate limiter trait for server-side rate limiting
#[async_trait]
pub trait RateLimiter: Send + Sync {
    /// Check if request is within limits
    async fn check_limit(&self, client_id: &str, endpoint: &str) -> Result<RateLimitResult>;

    /// Record request for rate limiting
    async fn record_request(&self, client_id: &str, endpoint: &str) -> Result<()>;

    /// Get rate limit status
    async fn get_status(&self, client_id: &str) -> Result<RateLimitStatus>;

    /// Reset rate limits for client
    async fn reset_limits(&self, client_id: &str) -> Result<()>;
}

/// Request validator trait
pub trait RequestValidator: Send + Sync {
    /// Validate inference request
    fn validate_inference_request(&self, request: &InferenceRequest) -> Result<()>;

    /// Validate OpenAI chat request
    fn validate_chat_request(&self, request: &crate::openai::ChatCompletionsRequest) -> Result<()>;

    /// Validate request parameters
    fn validate_parameters(&self, params: &serde_json::Value) -> Result<()>;

    /// Get validation rules
    fn get_rules(&self) -> &ValidationRules;
}

/// Health check provider
#[async_trait]
pub trait HealthChecker: Send + Sync {
    /// Perform health check
    async fn check_health(&self) -> HealthStatus;

    /// Check specific component
    async fn check_component(&self, component: &str) -> ComponentHealth;

    /// Get health check configuration
    fn config(&self) -> &HealthCheckConfig;
}

/// Metrics collector for server
#[async_trait]
pub trait MetricsCollector: Send + Sync {
    /// Record request metrics
    async fn record_request(
        &self,
        request: &HttpRequest,
        response: &HttpResponse,
        duration: Duration,
    );

    /// Record error metrics
    async fn record_error(&self, error: &ferrum_types::FerrumError, endpoint: &str);

    /// Get current metrics
    fn get_metrics(&self) -> ServerMetrics;

    /// Reset metrics
    async fn reset_metrics(&self) -> Result<()>;
}

/// Server lifecycle manager
#[async_trait]
pub trait ServerLifecycle: Send + Sync {
    /// Initialize server components
    async fn initialize(&self) -> Result<()>;

    /// Start all services
    async fn start_services(&self) -> Result<()>;

    /// Stop all services
    async fn stop_services(&self, timeout: Duration) -> Result<()>;

    /// Handle graceful shutdown
    async fn graceful_shutdown(&self, signal: ShutdownSignal) -> Result<()>;

    /// Get lifecycle state
    fn get_state(&self) -> LifecycleState;
}
