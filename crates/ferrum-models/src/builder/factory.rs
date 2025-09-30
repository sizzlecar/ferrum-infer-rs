//! Model builder factory - MVP with working stub executor

use std::sync::Arc;

use async_trait::async_trait;
use ferrum_interfaces::{
    backend::WeightFormat,
    model_builder::{
        BuildOptions, BuildTimeBreakdown, BuildTimeEstimate, BuilderCapabilities, BuilderInfo,
        ModelArchitecture, ModelArchitectureFamily, ValidationIssue, ValidationSeverity,
    },
    ComputeBackend, ModelBuilder, ModelExecutor, WeightLoader,
};
use ferrum_types::{ModelConfig, ModelSource, ModelType, Result};
use tracing::debug;

use crate::executor::StubModelExecutor;

/// Simple model builder - MVP implementation
#[derive(Debug, Default)]
pub struct SimpleModelBuilder;

#[async_trait]
impl ModelBuilder for SimpleModelBuilder {
    async fn build_model(
        &self,
        config: &ModelConfig,
        compute_backend: Arc<dyn ComputeBackend>,
        _weight_loader: Arc<dyn WeightLoader>,
    ) -> Result<Box<dyn ModelExecutor>> {
        debug!(
            "Building stub model: model_id={:?}, type={:?}",
            config.model_id, config.model_type
        );

        let vocab_size = config
            .extra_config
            .get("vocab_size")
            .and_then(|v| v.as_u64())
            .unwrap_or(32000) as usize;

        let executor = StubModelExecutor::new(config.model_id.clone(), vocab_size, compute_backend);

        debug!("Built stub model executor");
        Ok(Box::new(executor))
    }

    async fn build_from_source(
        &self,
        source: &ModelSource,
        compute_backend: Arc<dyn ComputeBackend>,
        _weight_loader: Arc<dyn WeightLoader>,
        _build_options: &BuildOptions,
    ) -> Result<Box<dyn ModelExecutor>> {
        debug!("Building model from source: {:?}", source);

        let model_id = match source {
            ModelSource::Local(path) => path.clone(),
            ModelSource::HuggingFace { repo_id, .. } => repo_id.clone(),
            ModelSource::Url { url, .. } => url.clone(),
            ModelSource::S3 { key, .. } => key.clone(),
        };

        let executor = StubModelExecutor::new(model_id, 32000, compute_backend);
        Ok(Box::new(executor))
    }

    fn validate_config(&self, config: &ModelConfig) -> Result<Vec<ValidationIssue>> {
        let mut issues = Vec::new();

        if config.model_path.is_empty() {
            issues.push(ValidationIssue {
                severity: ValidationSeverity::Error,
                category: "configuration".into(),
                description: "Model path cannot be empty".into(),
                suggested_fix: Some("Provide a valid model path".into()),
                config_path: "model_path".into(),
            });
        }

        Ok(issues)
    }

    fn supported_model_types(&self) -> Vec<ModelType> {
        vec![
            ModelType::Llama,
            ModelType::Mistral,
            ModelType::Qwen,
            ModelType::Phi,
            ModelType::Custom("stub".into()),
        ]
    }

    async fn estimate_build_time(&self, _config: &ModelConfig) -> Result<BuildTimeEstimate> {
        Ok(BuildTimeEstimate {
            min_time_seconds: 1,
            max_time_seconds: 10,
            expected_time_seconds: 3,
            time_breakdown: BuildTimeBreakdown {
                weight_loading_seconds: 1,
                model_init_seconds: 1,
                optimization_seconds: 0,
                validation_seconds: 1,
                overhead_seconds: 0,
            },
            factors: vec![],
        })
    }

    fn builder_info(&self) -> BuilderInfo {
        BuilderInfo {
            name: "SimpleModelBuilder".into(),
            version: env!("CARGO_PKG_VERSION").into(),
            supported_architectures: vec![
                ModelArchitecture {
                    name: "Llama".into(),
                    family: ModelArchitectureFamily::Transformer,
                    variants: vec!["llama-7b".into(), "llama-13b".into()],
                    required_features: vec![],
                },
                ModelArchitecture {
                    name: "Mistral".into(),
                    family: ModelArchitectureFamily::Transformer,
                    variants: vec!["mistral-7b".into()],
                    required_features: vec![],
                },
            ],
            supported_weight_formats: vec![WeightFormat::SafeTensors],
            supported_optimizations: vec![],
            capabilities: BuilderCapabilities {
                max_model_size: Some(70_000_000_000),
                supports_dynamic_shapes: false,
                supports_custom_ops: false,
                supports_mixed_precision: true,
                supports_model_parallelism: false,
                supports_parallel_build: false,
                supports_incremental_build: false,
            },
        }
    }
}

/// Default model builder factory
#[derive(Debug, Default, Clone)]
pub struct DefaultModelBuilderFactory;

impl DefaultModelBuilderFactory {
    pub fn new() -> Self {
        Self
    }

    pub fn create(&self) -> Arc<dyn ModelBuilder + Send + Sync> {
        Arc::new(SimpleModelBuilder)
    }

    pub fn create_for_model_type(
        &self,
        _model_type: &ModelType,
    ) -> Arc<dyn ModelBuilder + Send + Sync> {
        Arc::new(SimpleModelBuilder)
    }
}