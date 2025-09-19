//! Model configuration management

use ferrum_types::{Result, ModelId};
use std::collections::HashMap;

/// Configuration manager for models
#[derive(Debug, Clone, Default)]
pub struct ConfigManager {
    configs: HashMap<ModelId, serde_json::Value>,
}

impl ConfigManager {
    pub fn new() -> Self {
        Self::default()
    }
}
