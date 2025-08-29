//! # Ferrum Infer
//!
//! A high-performance Rust-based LLM inference engine MVP designed for
//! single-node deployment with OpenAI-compatible API.
//!
//! ## Architecture Overview
//!
//! The engine consists of several core modules:
//! - `api`: RESTful API layer with OpenAI compatibility
//! - `inference`: Core inference engine and model management
//! - `cache`: KV caching system for performance optimization
//! - `config`: Configuration management
//! - `error`: Centralized error handling
//! - `metrics`: Performance monitoring and telemetry

pub mod api;
pub mod cache;
pub mod config;
pub mod error;
pub mod inference;
pub mod metrics;
pub mod models;
pub mod utils;

#[cfg(test)]
pub mod test_utils;

// Re-export commonly used types
pub use crate::config::Config;
pub use crate::error::{EngineError, Result};
pub use crate::inference::{InferenceEngine, InferenceRequest, InferenceResponse};
pub use crate::models::{ModelLoader, ModelManager};

/// Engine version information
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
pub const NAME: &str = env!("CARGO_PKG_NAME");

/// Initialize the inference engine with default configuration
pub async fn init_engine() -> Result<InferenceEngine> {
    let config = Config::from_env()?;
    InferenceEngine::new(config).await
}

/// Initialize the inference engine with custom configuration
pub async fn init_engine_with_config(config: Config) -> Result<InferenceEngine> {
    InferenceEngine::new(config).await
}
