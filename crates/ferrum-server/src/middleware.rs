//! Middleware configuration types
//!
//! This module defines configuration types for various middleware components.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

/// Middleware configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MiddlewareConfig {
    /// Authentication middleware
    pub auth: Option<AuthConfig>,

    /// CORS middleware
    pub cors: Option<CorsConfig>,

    /// Logging middleware
    pub logging: Option<LoggingConfig>,

    /// Compression middleware
    pub compression: Option<CompressionConfig>,

    /// Rate limiting middleware
    pub rate_limit: Option<RateLimitConfig>,

    /// Timeout middleware
    pub timeout: Option<TimeoutConfig>,

    /// Custom middleware configurations
    pub custom: HashMap<String, serde_json::Value>,
}

/// Authentication configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthConfig {
    /// Enable authentication
    pub enabled: bool,

    /// JWT secret key
    pub jwt_secret: Option<String>,

    /// JWT issuer
    pub jwt_issuer: Option<String>,

    /// JWT audience
    pub jwt_audience: Option<String>,

    /// Token expiration time
    pub token_expiration: Duration,

    /// API key validation endpoint
    pub api_key_endpoint: Option<String>,

    /// Default permissions
    pub default_permissions: Vec<String>,

    /// Admin API keys
    pub admin_keys: Vec<String>,
}

/// CORS configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorsConfig {
    /// Enable CORS
    pub enabled: bool,

    /// Allowed origins
    pub allowed_origins: Vec<String>,

    /// Allowed methods
    pub allowed_methods: Vec<String>,

    /// Allowed headers
    pub allowed_headers: Vec<String>,

    /// Exposed headers
    pub exposed_headers: Vec<String>,

    /// Allow credentials
    pub allow_credentials: bool,

    /// Max age for preflight requests
    pub max_age: Duration,
}

/// Logging configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoggingConfig {
    /// Enable request logging
    pub enabled: bool,

    /// Log level
    pub level: LogLevel,

    /// Include request body
    pub include_body: bool,

    /// Include response body
    pub include_response: bool,

    /// Include headers
    pub include_headers: bool,

    /// Exclude paths from logging
    pub exclude_paths: Vec<String>,

    /// Log format
    pub format: LogFormat,
}

/// Log levels
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum LogLevel {
    Trace,
    Debug,
    Info,
    Warn,
    Error,
}

/// Log formats
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum LogFormat {
    Json,
    Text,
    Combined,
    Common,
}

/// Compression configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionConfig {
    /// Enable compression
    pub enabled: bool,

    /// Compression algorithms
    pub algorithms: Vec<CompressionAlgorithm>,

    /// Minimum response size to compress
    pub min_size: usize,

    /// Compression level (0-9)
    pub level: u32,

    /// Content types to compress
    pub content_types: Vec<String>,
}

/// Compression algorithms
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CompressionAlgorithm {
    Gzip,
    Deflate,
    Brotli,
    Zstd,
}

/// Rate limiting configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimitConfig {
    /// Enable rate limiting
    pub enabled: bool,

    /// Default rate limits
    pub default_limits: RateLimits,

    /// Per-client rate limits
    pub client_limits: HashMap<String, RateLimits>,

    /// Rate limit storage backend
    pub storage: RateLimitStorage,

    /// Rate limit headers
    pub include_headers: bool,
}

/// Rate limits specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimits {
    /// Requests per minute
    pub requests_per_minute: u32,

    /// Requests per hour
    pub requests_per_hour: u32,

    /// Tokens per minute
    pub tokens_per_minute: u32,

    /// Tokens per hour
    pub tokens_per_hour: u32,

    /// Concurrent requests
    pub concurrent_requests: u32,
}

/// Rate limit storage backends
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RateLimitStorage {
    Memory,
    Redis,
    Database,
}

/// Timeout configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeoutConfig {
    /// Enable timeout middleware
    pub enabled: bool,

    /// Request timeout
    pub request_timeout: Duration,

    /// Keep-alive timeout
    pub keep_alive_timeout: Duration,

    /// Read timeout
    pub read_timeout: Duration,

    /// Write timeout
    pub write_timeout: Duration,
}

impl Default for AuthConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            jwt_secret: None,
            jwt_issuer: Some("ferrum-infer".to_string()),
            jwt_audience: Some("ferrum-api".to_string()),
            token_expiration: Duration::from_secs(3600),
            api_key_endpoint: None,
            default_permissions: vec![],
            admin_keys: vec![],
        }
    }
}

impl Default for CorsConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            allowed_origins: vec!["*".to_string()],
            allowed_methods: vec!["GET".to_string(), "POST".to_string(), "OPTIONS".to_string()],
            allowed_headers: vec![
                "Content-Type".to_string(),
                "Authorization".to_string(),
                "X-Requested-With".to_string(),
            ],
            exposed_headers: vec![],
            allow_credentials: false,
            max_age: Duration::from_secs(86400),
        }
    }
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            level: LogLevel::Info,
            include_body: false,
            include_response: false,
            include_headers: false,
            exclude_paths: vec!["/health".to_string(), "/metrics".to_string()],
            format: LogFormat::Json,
        }
    }
}

impl Default for CompressionConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            algorithms: vec![CompressionAlgorithm::Gzip, CompressionAlgorithm::Deflate],
            min_size: 1024,
            level: 6,
            content_types: vec![
                "application/json".to_string(),
                "text/plain".to_string(),
                "text/html".to_string(),
            ],
        }
    }
}

impl Default for RateLimitConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            default_limits: RateLimits::default(),
            client_limits: HashMap::new(),
            storage: RateLimitStorage::Memory,
            include_headers: true,
        }
    }
}

impl Default for RateLimits {
    fn default() -> Self {
        Self {
            requests_per_minute: 60,
            requests_per_hour: 1000,
            tokens_per_minute: 10000,
            tokens_per_hour: 100000,
            concurrent_requests: 10,
        }
    }
}

impl Default for TimeoutConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            request_timeout: Duration::from_secs(30),
            keep_alive_timeout: Duration::from_secs(60),
            read_timeout: Duration::from_secs(10),
            write_timeout: Duration::from_secs(10),
        }
    }
}
