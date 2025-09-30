//! 模型定义与配置解析占位实现

use ferrum_types::{ModelInfo, ModelType, Result};

/// 占位的模型定义结构
#[derive(Debug, Clone, Default)]
pub struct ModelDefinition {
    pub info: Option<ModelInfo>,
}

/// 配置管理器（占位）
#[derive(Debug, Default)]
pub struct ConfigManager;

impl ConfigManager {
    pub fn new() -> Self {
        Self
    }

    pub async fn load_from_path(&mut self, _path: &std::path::Path) -> Result<ModelDefinition> {
        Ok(ModelDefinition::default())
    }

    pub fn infer_model_type(&self, _definition: &ModelDefinition) -> ModelType {
        ModelType::Custom("placeholder".into())
    }
}
