//! 默认模型构建器工厂，占位实现

use std::sync::Arc;

use async_trait::async_trait;
use ferrum_interfaces::{ModelBuilder, ModelExecutor, WeightLoader};
use ferrum_types::{ModelConfig, ModelInfo, ModelType, Result};

/// 简单的占位模型构建器
#[derive(Debug, Default)]
pub struct SimpleModelBuilder;

#[async_trait]
impl ModelBuilder for SimpleModelBuilder {
    async fn build_model(
        &self,
        _config: &ModelConfig,
        _compute_backend: Arc<dyn ferrum_interfaces::ComputeBackend>,
        _weight_loader: Arc<dyn WeightLoader>,
    ) -> Result<Box<dyn ModelExecutor>> {
        Err(ferrum_types::FerrumError::not_implemented(
            "Model builder placeholder",
        ))
    }

    async fn build_from_source(
        &self,
        _source: &ferrum_types::ModelSource,
        _compute_backend: Arc<dyn ferrum_interfaces::ComputeBackend>,
        _weight_loader: Arc<dyn WeightLoader>,
        _build_options: &ferrum_interfaces::BuildOptions,
    ) -> Result<Box<dyn ModelExecutor>> {
        Err(ferrum_types::FerrumError::not_implemented(
            "Model builder placeholder",
        ))
    }

    fn validate_config(&self, _config: &ModelConfig) -> Result<Vec<ferrum_interfaces::ValidationIssue>> {
        Ok(vec![])
    }

    fn supported_model_types(&self) -> Vec<ModelType> {
        vec![ModelType::Custom("placeholder".into())]
    }

    async fn estimate_build_time(&self, _config: &ModelConfig) -> Result<ferrum_interfaces::BuildTimeEstimate> {
        Err(ferrum_types::FerrumError::not_implemented(
            "Build time estimation placeholder",
        ))
    }

    fn builder_info(&self) -> ferrum_interfaces::BuilderInfo {
        ferrum_interfaces::BuilderInfo {
            name: "placeholder".into(),
            version: "0.0.0".into(),
            supported_architectures: vec![],
            supported_weight_formats: vec![],
            supported_optimizations: vec![],
            capabilities: ferrum_interfaces::BuilderCapabilities::default(),
        }
    }
}

/// 默认模型构建器工厂
#[derive(Debug, Default)]
pub struct DefaultModelBuilderFactory;

impl DefaultModelBuilderFactory {
    pub fn new() -> Self {
        Self
    }

    pub fn create(&self) -> Arc<dyn ModelBuilder + Send + Sync> {
        Arc::new(SimpleModelBuilder)
    }
}
