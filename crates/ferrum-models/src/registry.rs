//! 模型注册表占位实现

use std::collections::HashMap;

use ferrum_types::{ModelId, Result};

#[derive(Debug, Default)]
pub struct ModelRegistry {
    aliases: HashMap<String, String>,
}

impl ModelRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register_alias(&mut self, alias: impl Into<String>, target: impl Into<String>) {
        self.aliases.insert(alias.into(), target.into());
    }

    pub fn resolve_model_id(&self, name: &str) -> String {
        self.aliases
            .get(name)
            .cloned()
            .unwrap_or_else(|| name.to_string())
    }

    pub async fn discover_models(&mut self, _root: &std::path::Path) -> Result<Vec<ModelDiscoveryEntry>> {
        Ok(vec![])
    }
}

#[derive(Debug, Clone)]
pub struct ModelDiscoveryEntry {
    pub model_id: ModelId,
    pub path: std::path::PathBuf,
    pub format: String,
}
