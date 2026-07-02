//! Error types for Ferrum inference framework

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Main error type for Ferrum operations
#[derive(Debug, Error, Clone, Serialize, Deserialize)]
pub enum FerrumError {
    /// Configuration errors
    #[error("Configuration error: {message}")]
    Config { message: String },

    /// Model loading/initialization errors
    #[error("Model error: {message}")]
    Model { message: String },

    /// Tokenizer errors
    #[error("Tokenizer error: {message}")]
    Tokenizer { message: String },

    /// Backend/runtime errors
    #[error("Backend error: {message}")]
    Backend { message: String },

    /// Device/memory errors
    #[error("Device error: {message}")]
    Device { message: String },

    /// Scheduling/queue errors
    #[error("Scheduler error: {message}")]
    Scheduler { message: String },

    /// Request validation errors
    #[error("Request validation error: {message}")]
    RequestValidation { message: String },

    /// Resource exhaustion errors
    #[error("Resource exhausted: {message}")]
    ResourceExhausted { message: String },

    /// Timeout errors
    #[error("Operation timed out: {message}")]
    Timeout { message: String },

    /// Authentication/authorization errors
    #[error("Authentication error: {message}")]
    Auth { message: String },

    /// Rate limiting errors
    #[error("Rate limit exceeded: {message}")]
    RateLimit { message: String },

    /// I/O errors
    #[error("I/O error: {message}")]
    IO { message: String },

    /// Serialization/deserialization errors
    #[error("Serialization error: {message}")]
    Serialization { message: String },

    /// Network errors
    #[error("Network error: {message}")]
    Network { message: String },

    /// Internal errors (should not happen in normal operation)
    #[error("Internal error: {message}")]
    Internal { message: String },

    /// Request was cancelled
    #[error("Request cancelled: {message}")]
    Cancelled { message: String },

    /// Not found errors
    #[error("Not found: {message}")]
    NotFound { message: String },

    /// Already exists errors
    #[error("Already exists: {message}")]
    AlreadyExists { message: String },

    /// Permission denied errors
    #[error("Permission denied: {message}")]
    PermissionDenied { message: String },

    /// Unsupported operation errors
    #[error("Unsupported operation: {message}")]
    Unsupported { message: String },

    /// Invalid format errors (parsing, schema mismatches)
    #[error("Invalid format: {message}")]
    InvalidFormat { message: String },

    /// Invalid parameters or configuration values
    #[error("Invalid parameter: {message}")]
    InvalidParameter { message: String },
}

impl FerrumError {
    /// Create a configuration error
    pub fn config(message: impl Into<String>) -> Self {
        Self::Config {
            message: message.into(),
        }
    }

    /// Create a model error
    pub fn model(message: impl Into<String>) -> Self {
        Self::Model {
            message: message.into(),
        }
    }

    /// Create a tokenizer error
    pub fn tokenizer(message: impl Into<String>) -> Self {
        Self::Tokenizer {
            message: message.into(),
        }
    }

    /// Create a backend error
    pub fn backend(message: impl Into<String>) -> Self {
        Self::Backend {
            message: message.into(),
        }
    }

    /// Create a device error
    pub fn device(message: impl Into<String>) -> Self {
        Self::Device {
            message: message.into(),
        }
    }

    /// Create a scheduler error
    pub fn scheduler(message: impl Into<String>) -> Self {
        Self::Scheduler {
            message: message.into(),
        }
    }

    /// Create a request validation error
    pub fn request_validation(message: impl Into<String>) -> Self {
        Self::RequestValidation {
            message: message.into(),
        }
    }

    /// Create a resource exhausted error
    pub fn resource_exhausted(message: impl Into<String>) -> Self {
        Self::ResourceExhausted {
            message: message.into(),
        }
    }

    /// Create a timeout error
    pub fn timeout(message: impl Into<String>) -> Self {
        Self::Timeout {
            message: message.into(),
        }
    }

    /// Create an auth error
    pub fn auth(message: impl Into<String>) -> Self {
        Self::Auth {
            message: message.into(),
        }
    }

    /// Create a rate limit error
    pub fn rate_limit(message: impl Into<String>) -> Self {
        Self::RateLimit {
            message: message.into(),
        }
    }

    /// Create an I/O error
    pub fn io(message: impl Into<String>) -> Self {
        Self::IO {
            message: message.into(),
        }
    }

    /// Create a serialization error
    pub fn serialization(message: impl Into<String>) -> Self {
        Self::Serialization {
            message: message.into(),
        }
    }

    /// Create a network error
    pub fn network(message: impl Into<String>) -> Self {
        Self::Network {
            message: message.into(),
        }
    }

    /// Create an internal error
    pub fn internal(message: impl Into<String>) -> Self {
        Self::Internal {
            message: message.into(),
        }
    }

    /// Create a cancelled error
    pub fn cancelled(message: impl Into<String>) -> Self {
        Self::Cancelled {
            message: message.into(),
        }
    }

    /// Create a not found error
    pub fn not_found(message: impl Into<String>) -> Self {
        Self::NotFound {
            message: message.into(),
        }
    }

    /// Create an already exists error
    pub fn already_exists(message: impl Into<String>) -> Self {
        Self::AlreadyExists {
            message: message.into(),
        }
    }

    /// Create a permission denied error
    pub fn permission_denied(message: impl Into<String>) -> Self {
        Self::PermissionDenied {
            message: message.into(),
        }
    }

    /// Create an unsupported operation error
    pub fn unsupported(message: impl Into<String>) -> Self {
        Self::Unsupported {
            message: message.into(),
        }
    }

    /// Create an invalid format error (parsing/schema mismatch)
    pub fn invalid_format(message: impl Into<String>) -> Self {
        Self::InvalidFormat {
            message: message.into(),
        }
    }

    /// Create an invalid parameter error
    pub fn invalid_parameter(message: impl Into<String>) -> Self {
        Self::InvalidParameter {
            message: message.into(),
        }
    }

    // Alias methods for compatibility

    /// Alias for io() - Create an I/O error from string
    pub fn io_str(message: impl Into<String>) -> Self {
        Self::io(message)
    }

    /// Alias for config() - Create a configuration error
    pub fn configuration(message: impl Into<String>) -> Self {
        Self::config(message)
    }

    /// Alias for serialization() - Create a deserialization error
    pub fn deserialization(message: impl Into<String>) -> Self {
        Self::serialization(message)
    }

    /// Alias for request_validation() - Create an invalid request error
    pub fn invalid_request(message: impl Into<String>) -> Self {
        Self::request_validation(message)
    }

    /// Check if this is a retryable error
    pub fn is_retryable(&self) -> bool {
        matches!(
            self,
            Self::ResourceExhausted { .. } | Self::Timeout { .. } | Self::Network { .. }
        )
    }

    /// Check if this is a client error (4xx equivalent)
    pub fn is_client_error(&self) -> bool {
        matches!(
            self,
            Self::RequestValidation { .. }
                | Self::Auth { .. }
                | Self::RateLimit { .. }
                | Self::NotFound { .. }
                | Self::AlreadyExists { .. }
                | Self::PermissionDenied { .. }
                | Self::Unsupported { .. }
        )
    }

    /// Check if this is a server error (5xx equivalent)
    pub fn is_server_error(&self) -> bool {
        matches!(
            self,
            Self::Model { .. }
                | Self::Backend { .. }
                | Self::Device { .. }
                | Self::Scheduler { .. }
                | Self::ResourceExhausted { .. }
                | Self::Timeout { .. }
                | Self::Internal { .. }
        )
    }

    /// Stable failure class used by product observability replay bundles.
    ///
    /// This intentionally stays smaller than the full error enum because the
    /// replay validator has different evidence requirements for resource
    /// failures and generic panic/error failures.
    pub fn observability_failure_kind(&self) -> &'static str {
        match self {
            Self::ResourceExhausted { .. } => "oom_admission",
            Self::Device { message } if looks_like_oom(message) => "oom",
            Self::Scheduler { message } if looks_like_admission(message) => "admission",
            Self::Backend { message } if looks_like_oom(message) => "oom",
            _ => "error",
        }
    }

    /// Stable error-kind label for first-failure-event diagnostics.
    pub fn observability_error_kind(&self) -> &'static str {
        match self {
            Self::Config { .. } => "config",
            Self::Model { .. } => "model",
            Self::Tokenizer { .. } => "tokenizer",
            Self::Backend { .. } => "backend",
            Self::Device { .. } => "device",
            Self::Scheduler { .. } => "scheduler",
            Self::RequestValidation { .. } => "request_validation",
            Self::ResourceExhausted { .. } => "resource_exhausted",
            Self::Timeout { .. } => "timeout",
            Self::Auth { .. } => "auth",
            Self::RateLimit { .. } => "rate_limit",
            Self::IO { .. } => "io",
            Self::Serialization { .. } => "serialization",
            Self::Network { .. } => "network",
            Self::Internal { .. } => "internal",
            Self::Cancelled { .. } => "cancelled",
            Self::NotFound { .. } => "not_found",
            Self::AlreadyExists { .. } => "already_exists",
            Self::PermissionDenied { .. } => "permission_denied",
            Self::Unsupported { .. } => "unsupported",
            Self::InvalidFormat { .. } => "invalid_format",
            Self::InvalidParameter { .. } => "invalid_parameter",
        }
    }
}

fn looks_like_oom(message: &str) -> bool {
    let normalized = message.to_ascii_lowercase();
    normalized.contains("out of memory")
        || normalized.contains("oom")
        || normalized.contains("cuda error 2")
        || normalized.contains("cudaerroroutofmemory")
        || normalized.contains("metal out of memory")
}

fn looks_like_admission(message: &str) -> bool {
    let normalized = message.to_ascii_lowercase();
    normalized.contains("admission")
        || normalized.contains("capacity")
        || normalized.contains("resource")
        || normalized.contains("queue")
}

/// Conversion from std::io::Error
impl From<std::io::Error> for FerrumError {
    fn from(err: std::io::Error) -> Self {
        Self::io(format!("{}", err))
    }
}

/// Conversion from serde_json::Error
impl From<serde_json::Error> for FerrumError {
    fn from(err: serde_json::Error) -> Self {
        Self::serialization(format!("{}", err))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn observability_failure_kind_classifies_resource_and_oom() {
        assert_eq!(
            FerrumError::resource_exhausted("kv capacity exhausted").observability_failure_kind(),
            "oom_admission"
        );
        assert_eq!(
            FerrumError::device("CUDA out of memory while allocating KV")
                .observability_failure_kind(),
            "oom"
        );
        assert_eq!(
            FerrumError::scheduler("admission capacity rejected request")
                .observability_failure_kind(),
            "admission"
        );
        assert_eq!(
            FerrumError::internal("stub generation failed").observability_failure_kind(),
            "error"
        );
    }

    #[test]
    fn observability_error_kind_uses_stable_variant_labels() {
        assert_eq!(
            FerrumError::resource_exhausted("slots exhausted").observability_error_kind(),
            "resource_exhausted"
        );
        assert_eq!(
            FerrumError::invalid_parameter("bad").observability_error_kind(),
            "invalid_parameter"
        );
    }
}
