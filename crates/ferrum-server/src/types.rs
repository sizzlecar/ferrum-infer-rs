//! Type definitions for HTTP server
//!
//! This module defines the core types used throughout the server system.

use ferrum_core::{RequestId, ModelId};
use crate::middleware::{AuthConfig, CorsConfig, CompressionConfig};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};
use chrono::{DateTime, Utc};

/// HTTP request representation
#[derive(Debug, Clone)]
pub struct HttpRequest {
    /// HTTP method
    pub method: HttpMethod,
    
    /// Request path
    pub path: String,
    
    /// Query parameters
    pub query: HashMap<String, String>,
    
    /// Request headers
    pub headers: Headers,
    
    /// Request body
    pub body: Vec<u8>,
    
    /// Client IP address
    pub client_ip: Option<std::net::IpAddr>,
    
    /// Request timestamp
    pub timestamp: DateTime<Utc>,
    
    /// Request ID for tracking
    pub request_id: RequestId,
}

/// HTTP response representation
#[derive(Debug, Clone)]
pub struct HttpResponse {
    /// Status code
    pub status: StatusCode,
    
    /// Response headers
    pub headers: Headers,
    
    /// Response body
    pub body: Vec<u8>,
    
    /// Content type
    pub content_type: String,
}

/// HTTP methods
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum HttpMethod {
    GET,
    POST,
    PUT,
    DELETE,
    PATCH,
    HEAD,
    OPTIONS,
}

/// HTTP status codes
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StatusCode {
    OK = 200,
    Created = 201,
    NoContent = 204,
    BadRequest = 400,
    Unauthorized = 401,
    Forbidden = 403,
    NotFound = 404,
    MethodNotAllowed = 405,
    TooManyRequests = 429,
    InternalServerError = 500,
    BadGateway = 502,
    ServiceUnavailable = 503,
    GatewayTimeout = 504,
}

/// HTTP headers
pub type Headers = HashMap<String, String>;

/// Request context for passing data between middleware
#[derive(Debug, Clone)]
pub struct RequestContext {
    /// Client information
    pub client_info: Option<ClientInfo>,
    
    /// Authentication result
    pub auth_result: Option<AuthResult>,
    
    /// Request start time
    pub start_time: Instant,
    
    /// Custom context data
    pub data: HashMap<String, serde_json::Value>,
    
    /// Request tracing ID
    pub trace_id: String,
}

impl Default for RequestContext {
    fn default() -> Self {
        Self {
            client_info: None,
            auth_result: None,
            start_time: Instant::now(),
            data: HashMap::new(),
            trace_id: uuid::Uuid::new_v4().to_string(),
        }
    }
}

/// Server configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    /// Server host
    pub host: String,
    
    /// Server port
    pub port: u16,
    
    /// Maximum concurrent connections
    pub max_connections: usize,
    
    /// Request timeout
    pub request_timeout: Duration,
    
    /// Keep-alive timeout
    pub keep_alive_timeout: Duration,
    
    /// Enable TLS
    pub enable_tls: bool,
    
    /// TLS certificate path
    pub tls_cert_path: Option<String>,
    
    /// TLS private key path
    pub tls_key_path: Option<String>,
    
    /// CORS configuration
    pub cors: Option<CorsConfig>,
    
    /// Compression configuration
    pub compression: Option<CompressionConfig>,
    
    /// Authentication configuration
    pub auth: Option<AuthConfig>,
    
    /// API versioning
    pub api_version: ApiVersion,
}

/// API version
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ApiVersion {
    V1,
    V2,
}

/// Server metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ServerMetrics {
    /// Total requests handled
    pub total_requests: u64,
    
    /// Requests by endpoint
    pub requests_by_endpoint: HashMap<String, u64>,
    
    /// Requests by status code
    pub requests_by_status: HashMap<u16, u64>,
    
    /// Average response time in milliseconds
    pub avg_response_time_ms: f64,
    
    /// 95th percentile response time
    pub p95_response_time_ms: f64,
    
    /// 99th percentile response time
    pub p99_response_time_ms: f64,
    
    /// Current active connections
    pub active_connections: usize,
    
    /// Total bytes sent
    pub bytes_sent: u64,
    
    /// Total bytes received
    pub bytes_received: u64,
    
    /// Error rate (0.0 - 1.0)
    pub error_rate: f32,
    
    /// Uptime in seconds
    pub uptime_seconds: u64,
}

/// Health status
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum HealthStatus {
    Healthy,
    Degraded,
    Unhealthy,
    Unknown,
}

/// Component health
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentHealth {
    pub name: String,
    pub status: HealthStatus,
    pub message: Option<String>,
    pub last_check: DateTime<Utc>,
    pub response_time_ms: Option<u64>,
}

/// Authentication result
#[derive(Debug, Clone)]
pub struct AuthResult {
    pub success: bool,
    pub client_info: Option<ClientInfo>,
    pub token_claims: Option<TokenClaims>,
    pub error: Option<String>,
}

/// Client information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClientInfo {
    pub client_id: String,
    pub api_key: Option<String>,
    pub organization_id: Option<String>,
    pub rate_limit_tier: RateLimitTier,
    pub permissions: Vec<String>,
}

/// JWT token claims
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenClaims {
    pub sub: String,
    pub exp: u64,
    pub iat: u64,
    pub iss: String,
    pub aud: String,
    pub permissions: Vec<String>,
}

/// Authentication schemes
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AuthScheme {
    ApiKey,
    Bearer,
    Basic,
    Custom(String),
}

/// Rate limit tiers
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RateLimitTier {
    Free,
    Pro,
    Enterprise,
    Custom(String),
}

/// Rate limit result
#[derive(Debug, Clone)]
pub struct RateLimitResult {
    pub allowed: bool,
    pub limit: u32,
    pub remaining: u32,
    pub reset_time: DateTime<Utc>,
    pub retry_after: Option<Duration>,
}

/// Rate limit status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimitStatus {
    pub requests_per_minute: u32,
    pub requests_per_hour: u32,
    pub tokens_per_minute: u32,
    pub current_usage: RateLimitUsage,
    pub reset_times: RateLimitResetTimes,
}

/// Rate limit usage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimitUsage {
    pub requests_this_minute: u32,
    pub requests_this_hour: u32,
    pub tokens_this_minute: u32,
}

/// Rate limit reset times
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimitResetTimes {
    pub requests_reset_minute: DateTime<Utc>,
    pub requests_reset_hour: DateTime<Utc>,
    pub tokens_reset_minute: DateTime<Utc>,
}

/// Validation rules
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationRules {
    /// Maximum prompt length
    pub max_prompt_length: usize,
    
    /// Maximum completion tokens
    pub max_completion_tokens: usize,
    
    /// Allowed models
    pub allowed_models: Option<Vec<String>>,
    
    /// Required parameters
    pub required_params: Vec<String>,
    
    /// Parameter constraints
    pub param_constraints: HashMap<String, ParameterConstraint>,
}

/// Parameter constraint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ParameterConstraint {
    Range { min: f64, max: f64 },
    OneOf(Vec<serde_json::Value>),
    Regex(String),
    Custom(String),
}

/// Health check configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthCheckConfig {
    /// Health check interval
    pub interval: Duration,
    
    /// Components to check
    pub components: Vec<String>,
    
    /// Timeout for health checks
    pub timeout: Duration,
    
    /// Number of retries
    pub retries: u32,
}

/// Stream configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamConfig {
    /// Chunk size for streaming
    pub chunk_size: usize,
    
    /// Buffer size
    pub buffer_size: usize,
    
    /// Flush interval
    pub flush_interval: Duration,
    
    /// Enable compression
    pub enable_compression: bool,
    
    /// Stream timeout
    pub timeout: Duration,
}

/// Shutdown signal types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShutdownSignal {
    SIGTERM,
    SIGINT,
    SIGQUIT,
    Custom,
}

/// Server lifecycle state
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LifecycleState {
    Starting,
    Running,
    Stopping,
    Stopped,
    Error,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: "0.0.0.0".to_string(),
            port: 8000,
            max_connections: 10000,
            request_timeout: Duration::from_secs(30),
            keep_alive_timeout: Duration::from_secs(60),
            enable_tls: false,
            tls_cert_path: None,
            tls_key_path: None,
            cors: None,
            compression: None,
            auth: None,
            api_version: ApiVersion::V1,
        }
    }
}

impl Default for StreamConfig {
    fn default() -> Self {
        Self {
            chunk_size: 1024,
            buffer_size: 8192,
            flush_interval: Duration::from_millis(50),
            enable_compression: false,
            timeout: Duration::from_secs(300),
        }
    }
}

impl Default for HealthCheckConfig {
    fn default() -> Self {
        Self {
            interval: Duration::from_secs(30),
            components: vec![
                "inference_engine".to_string(),
                "scheduler".to_string(),
                "cache".to_string(),
            ],
            timeout: Duration::from_secs(5),
            retries: 3,
        }
    }
}
