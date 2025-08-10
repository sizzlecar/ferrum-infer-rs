//! Error handling for the LLM inference engine
//!
//! This module provides a unified error handling system with proper error mapping
//! to HTTP status codes and structured error responses.

use actix_web::{HttpResponse, ResponseError};
use serde::{Deserialize, Serialize};
use std::fmt;
use thiserror::Error;

/// Main error type for the inference engine
#[derive(Error, Debug)]
pub enum EngineError {
    /// Configuration errors
    #[error("Configuration error: {message}")]
    Config { message: String },

    /// Model loading/inference errors
    #[error("Model error: {message}")]
    Model { message: String },

    /// Inference computation errors
    #[error("Inference error: {message}")]
    Inference { message: String },

    /// Cache operation errors
    #[error("Cache error: {message}")]
    Cache { message: String },

    /// Invalid request errors
    #[error("Invalid request: {message}")]
    InvalidRequest { message: String },

    /// Resource management errors
    #[error("Resource error: {message}")]
    Resource { message: String },

    /// Internal server errors
    #[error("Internal error: {message}")]
    Internal { message: String },

    /// I/O errors
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// Serialization errors
    #[error("Serialization error: {0}")]
    Serde(#[from] serde_json::Error),

    /// ML framework errors (only when ml feature is enabled)
    #[cfg(feature = "ml")]
    #[error("Candle error: {0}")]
    Candle(#[from] candle_core::Error),

    /// Tokenizer errors (only when ml feature is enabled)
    #[cfg(feature = "ml")]
    #[error("Tokenizer error: {0}")]
    Tokenizer(#[from] tokenizers::Error),
}

/// Result type alias for convenience
pub type Result<T> = std::result::Result<T, EngineError>;

/// Error response structure for API responses
#[derive(Debug, Serialize, Deserialize)]
pub struct ErrorResponse {
    pub error: ErrorDetails,
}

/// Detailed error information
#[derive(Debug, Serialize, Deserialize)]
pub struct ErrorDetails {
    pub message: String,
    pub error_type: String,
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

    /// Convert to an error response for API
    pub fn to_error_response(&self) -> ErrorResponse {
        let (error_type, code) = match self {
            EngineError::Config { .. } => ("config_error", "CONFIG_ERROR"),
            EngineError::Model { .. } => ("model_error", "MODEL_ERROR"),
            EngineError::Inference { .. } => ("inference_error", "INFERENCE_ERROR"),
            EngineError::Cache { .. } => ("cache_error", "CACHE_ERROR"),
            EngineError::InvalidRequest { .. } => ("invalid_request_error", "INVALID_REQUEST"),
            EngineError::Resource { .. } => ("resource_error", "RESOURCE_ERROR"),
            EngineError::Internal { .. } => ("internal_error", "INTERNAL_ERROR"),
            EngineError::Io(_) => ("io_error", "IO_ERROR"),
            EngineError::Serde(_) => ("serialization_error", "SERIALIZATION_ERROR"),
            #[cfg(feature = "ml")]
            EngineError::Candle(_) => ("ml_error", "ML_ERROR"),
            #[cfg(feature = "ml")]
            EngineError::Tokenizer(_) => ("tokenizer_error", "TOKENIZER_ERROR"),
        };

        ErrorResponse {
            error: ErrorDetails {
                message: self.to_string(),
                error_type: error_type.to_string(),
                code: code.to_string(),
            },
        }
    }
}

impl ResponseError for EngineError {
    fn error_response(&self) -> HttpResponse {
        let status = match self {
            EngineError::InvalidRequest { .. } => actix_web::http::StatusCode::BAD_REQUEST,
            EngineError::Model { .. } => actix_web::http::StatusCode::SERVICE_UNAVAILABLE,
            EngineError::Resource { .. } => actix_web::http::StatusCode::INSUFFICIENT_STORAGE,
            EngineError::Config { .. } 
            | EngineError::Inference { .. }
            | EngineError::Cache { .. }
            | EngineError::Internal { .. }
            | EngineError::Io(_)
            | EngineError::Serde(_) => actix_web::http::StatusCode::INTERNAL_SERVER_ERROR,
            #[cfg(feature = "ml")]
            EngineError::Candle(_) | EngineError::Tokenizer(_) => {
                actix_web::http::StatusCode::INTERNAL_SERVER_ERROR
            }
        };

        HttpResponse::build(status).json(self.to_error_response())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_creation() {
        let error = EngineError::config("Test config error");
        assert!(error.to_string().contains("Test config error"));

        let error = EngineError::invalid_request("Invalid parameter");
        assert!(error.to_string().contains("Invalid parameter"));
    }

    #[test]
    fn test_error_response() {
        let error = EngineError::invalid_request("Test error");
        let response = error.to_error_response();
        
        assert_eq!(response.error.error_type, "invalid_request_error");
        assert_eq!(response.error.code, "INVALID_REQUEST");
        assert!(response.error.message.contains("Test error"));
    }

    #[test]
    fn test_http_response() {
        let error = EngineError::invalid_request("Test error");
        let http_response = error.error_response();
        
        assert_eq!(http_response.status(), 400);
    }
}
