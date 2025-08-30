//! Error types for the Ferrum inference framework

use thiserror::Error;

/// Main error type for Ferrum
#[derive(Error, Debug)]
pub enum Error {
    /// IO errors
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// Model loading errors
    #[error("Model loading error: {0}")]
    ModelLoading(String),

    /// Model execution errors
    #[error("Model execution error: {0}")]
    ModelExecution(String),

    /// Scheduler errors
    #[error("Scheduler error: {0}")]
    Scheduler(String),

    /// Cache management errors
    #[error("Cache error: {0}")]
    Cache(String),

    /// Memory allocation errors
    #[error("Memory allocation error: {0}")]
    MemoryAllocation(String),

    /// Out of memory
    #[error("Out of memory: {0}")]
    OutOfMemory(String),

    /// Invalid request
    #[error("Invalid request: {0}")]
    InvalidRequest(String),

    /// Timeout
    #[error("Operation timed out: {0}")]
    Timeout(String),

    /// Resource not found
    #[error("Resource not found: {0}")]
    NotFound(String),

    /// Resource already exists
    #[error("Resource already exists: {0}")]
    AlreadyExists(String),

    /// Configuration error
    #[error("Configuration error: {0}")]
    Configuration(String),

    /// Unsupported operation
    #[error("Unsupported operation: {0}")]
    Unsupported(String),

    /// Internal error
    #[error("Internal error: {0}")]
    Internal(String),

    /// Other errors
    #[error("{0}")]
    Other(#[from] anyhow::Error),
}

/// Result type alias for Ferrum operations
pub type Result<T> = std::result::Result<T, Error>;

impl Error {
    /// Create a model loading error
    pub fn model_loading<S: Into<String>>(msg: S) -> Self {
        Error::ModelLoading(msg.into())
    }

    /// Create a model execution error
    pub fn model_execution<S: Into<String>>(msg: S) -> Self {
        Error::ModelExecution(msg.into())
    }

    /// Create a scheduler error
    pub fn scheduler<S: Into<String>>(msg: S) -> Self {
        Error::Scheduler(msg.into())
    }

    /// Create a cache error
    pub fn cache<S: Into<String>>(msg: S) -> Self {
        Error::Cache(msg.into())
    }

    /// Create a memory allocation error
    pub fn memory<S: Into<String>>(msg: S) -> Self {
        Error::MemoryAllocation(msg.into())
    }

    /// Create an out of memory error
    pub fn oom<S: Into<String>>(msg: S) -> Self {
        Error::OutOfMemory(msg.into())
    }

    /// Create an invalid request error
    pub fn invalid_request<S: Into<String>>(msg: S) -> Self {
        Error::InvalidRequest(msg.into())
    }

    /// Create a timeout error
    pub fn timeout<S: Into<String>>(msg: S) -> Self {
        Error::Timeout(msg.into())
    }

    /// Create a not found error
    pub fn not_found<S: Into<String>>(msg: S) -> Self {
        Error::NotFound(msg.into())
    }

    /// Create an already exists error
    pub fn already_exists<S: Into<String>>(msg: S) -> Self {
        Error::AlreadyExists(msg.into())
    }

    /// Create a configuration error
    pub fn configuration<S: Into<String>>(msg: S) -> Self {
        Error::Configuration(msg.into())
    }

    /// Create an unsupported operation error
    pub fn unsupported<S: Into<String>>(msg: S) -> Self {
        Error::Unsupported(msg.into())
    }

    /// Create an internal error
    pub fn internal<S: Into<String>>(msg: S) -> Self {
        Error::Internal(msg.into())
    }

    /// Create an IO error from string
    pub fn io_str<S: Into<String>>(msg: S) -> Self {
        Error::Io(std::io::Error::new(std::io::ErrorKind::Other, msg.into()))
    }

    /// Create a serialization error
    pub fn serialization<S: Into<String>>(msg: S) -> Self {
        Error::Internal(format!("Serialization error: {}", msg.into()))
    }

    /// Create a deserialization error
    pub fn deserialization<S: Into<String>>(msg: S) -> Self {
        Error::Internal(format!("Deserialization error: {}", msg.into()))
    }

    /// Create a network error
    pub fn network<S: Into<String>>(msg: S) -> Self {
        Error::Internal(format!("Network error: {}", msg.into()))
    }
}
