//! Error handling for the LLM inference engine
//!
//! This module provides centralized error handling with structured error types
//! that can be easily converted to HTTP responses and logged appropriately.

use std::fmt;
use actix_web::{HttpResponse, ResponseError};
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Main result type used throughout the engine
pub type Result<T> = std::result::Result<T, EngineError>;

/// Main error type for the inference engine
#[derive(Error, Debug)]
pub enum EngineError {
    /// Configuration related errors
    #[error("Configuration error: {message}")]
    Config { message: String },

    /// Model loading and management errors
    #[error("Model error: {message}")]
    Model { message: String },

    /// Inference computation errors
    #[error("Inference error: {message}")]
    Inference { message: String },

    /// Cache related errors
    #[error("Cache error: {message}")]
    Cache { message: String },

    /// API request validation errors
    #[error("Invalid request: {message}")]
    InvalidRequest { message: String },

    /// Resource management errors (memory, GPU, etc.)
    #[error("Resource error: {message}")]
    Resource { message: String },

    /// Internal server errors
    #[error("Internal error: {message}")]
    Internal { message: String },

    /// IO errors
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// Serialization errors
    #[error("Serialization error: {0}")]
    Serde(#[from] serde_json::Error),

    /// Candle framework errors
    #[error("Candle error: {0}")]
    Candle(#[from] candle_core::Error),

    /// Tokenizer errors
    #[error("Tokenizer error: {0}")]
    Tokenizer(#[from] tokenizers::Error),
}

/// Error response structure for API endpoints
#[derive(Debug, Serialize, Deserialize)]
pub struct ErrorResponse {
    pub error: ErrorDetail,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ErrorDetail {
    pub message: String,
    pub r#type: String,
    pub code: String,
}

impl EngineError {
    /// Create a configuration error
    pub fn config<S: Into<String>>(message: S) -> Self {
        Self::Config {
            message: message.into(),
        }
    }

    /// Create a model error
    pub fn model<S: Into<String>>(message: S) -> Self {
        Self::Model {
            message: message.into(),
        }
    }

    /// Create an inference error
    pub fn inference<S: Into<String>>(message: S) -> Self {
        Self::Inference {
            message: message.into(),
        }
    }

    /// Create a cache error
    pub fn cache<S: Into<String>>(message: S) -> Self {
        Self::Cache {
            message: message.into(),
        }
    }

    /// Create an invalid request error
    pub fn invalid_request<S: Into<String>>(message: S) -> Self {
        Self::InvalidRequest {
            message: message.into(),
        }
    }

    /// Create a resource error
    pub fn resource<S: Into<String>>(message: S) -> Self {
        Self::Resource {
            message: message.into(),
        }
    }

    /// Create an internal error
    pub fn internal<S: Into<String>>(message: S) -> Self {
        Self::Internal {
            message: message.into(),
        }
    }

    /// Get the error type as a string
    pub fn error_type(&self) -> &'static str {
        match self {
            Self::Config { .. } => "configuration_error",
            Self::Model { .. } => "model_error",
            Self::Inference { .. } => "inference_error",
            Self::Cache { .. } => "cache_error",
            Self::InvalidRequest { .. } => "invalid_request_error",
            Self::Resource { .. } => "resource_error",
            Self::Internal { .. } => "internal_error",
            Self::Io(_) => "io_error",
            Self::Serde(_) => "serialization_error",
            Self::Candle(_) => "candle_error",
            Self::Tokenizer(_) => "tokenizer_error",
        }
    }

    /// Get the error code for API responses
    pub fn error_code(&self) -> &'static str {
        match self {
            Self::Config { .. } => "CONFIG_ERROR",
            Self::Model { .. } => "MODEL_ERROR",
            Self::Inference { .. } => "INFERENCE_ERROR",
            Self::Cache { .. } => "CACHE_ERROR",
            Self::InvalidRequest { .. } => "INVALID_REQUEST",
            Self::Resource { .. } => "RESOURCE_ERROR",
            Self::Internal { .. } => "INTERNAL_ERROR",
            Self::Io(_) => "IO_ERROR",
            Self::Serde(_) => "SERIALIZATION_ERROR",
            Self::Candle(_) => "CANDLE_ERROR",
            Self::Tokenizer(_) => "TOKENIZER_ERROR",
        }
    }

    /// Convert to ErrorResponse for API responses
    pub fn to_error_response(&self) -> ErrorResponse {
        ErrorResponse {
            error: ErrorDetail {
                message: self.to_string(),
                r#type: self.error_type().to_string(),
                code: self.error_code().to_string(),
            },
        }
    }
}

impl ResponseError for EngineError {
    fn error_response(&self) -> HttpResponse {
        let status = match self {
            Self::InvalidRequest { .. } => actix_web::http::StatusCode::BAD_REQUEST,
            Self::Model { .. } => actix_web::http::StatusCode::SERVICE_UNAVAILABLE,
            Self::Resource { .. } => actix_web::http::StatusCode::INSUFFICIENT_STORAGE,
            _ => actix_web::http::StatusCode::INTERNAL_SERVER_ERROR,
        };

        HttpResponse::build(status).json(self.to_error_response())
    }
}

impl fmt::Display for ErrorResponse {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.error.code, self.error.message)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_creation() {
        let err = EngineError::config("Invalid config file");
        assert_eq!(err.error_type(), "configuration_error");
        assert_eq!(err.error_code(), "CONFIG_ERROR");
    }

    #[test]
    fn test_error_response() {
        let err = EngineError::invalid_request("Missing required parameter");
        let response = err.to_error_response();
        assert_eq!(response.error.r#type, "invalid_request_error");
        assert_eq!(response.error.code, "INVALID_REQUEST");
    }
}