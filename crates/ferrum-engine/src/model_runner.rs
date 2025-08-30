//! Model runner abstraction
//!
//! This module provides an abstract model runner that delegates to the backend
//! for actual model execution.

use ferrum_core::{Backend, Model, ModelConfig, Result};
use std::sync::Arc;

/// Generic model runner that uses a backend
pub struct ModelRunner {
    backend: Arc<dyn Backend>,
}

impl ModelRunner {
    /// Create a new model runner with the given backend
    pub fn new(backend: Arc<dyn Backend>) -> Self {
        Self { backend }
    }

    /// Get the backend
    pub fn backend(&self) -> &Arc<dyn Backend> {
        &self.backend
    }

    /// Load a model using the backend
    pub async fn load_model(&self, config: &ModelConfig) -> Result<Box<dyn Model>> {
        self.backend
            .load_weights(&config.model_path, config.dtype, &config.device)
            .await
    }
}
