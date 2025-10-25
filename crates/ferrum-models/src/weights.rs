//! Weight loading - MVP stub implementation

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use ferrum_interfaces::{
    backend::{
        TensorSpec, TransformationType, WeightFormat, WeightLoaderCapabilities, WeightMetadata,
        WeightSource, WeightSourceType,
    },
    TensorFactory, TensorRef, WeightLoader,
};
use ferrum_types::{DataType, Result};
use tracing::debug;

/// Weight loader handle wrapping a trait object
#[derive(Clone)]
pub struct WeightLoaderHandle(pub Arc<dyn WeightLoader + Send + Sync>);

impl std::fmt::Debug for WeightLoaderHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WeightLoaderHandle").finish()
    }
}

/// Create default stub weight loader
pub fn default_weight_loader() -> Result<WeightLoaderHandle> {
    debug!("Creating default stub weight loader");
    Ok(WeightLoaderHandle(Arc::new(StubWeightLoader::new())))
}

/// Stub weight loader - MVP implementation
pub struct StubWeightLoader {
    factory: Option<Arc<dyn TensorFactory>>,
}

impl StubWeightLoader {
    pub fn new() -> Self {
        Self { factory: None }
    }

    pub fn with_factory(factory: Arc<dyn TensorFactory>) -> Self {
        Self {
            factory: Some(factory),
        }
    }
}

impl Default for StubWeightLoader {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for StubWeightLoader {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StubWeightLoader")
            .field("has_factory", &self.factory.is_some())
            .finish()
    }
}

#[async_trait]
impl WeightLoader for StubWeightLoader {
    async fn load_tensor(&self, spec: &TensorSpec) -> Result<TensorRef> {
        debug!(
            "StubWeightLoader: creating zeros for '{}' {:?}",
            spec.name, spec.shape
        );

        if let Some(factory) = &self.factory {
            factory.zeros(&spec.shape, spec.dtype, &spec.device)
        } else {
            Err(ferrum_types::FerrumError::model(
                "No tensor factory configured in stub weight loader",
            ))
        }
    }

    async fn load_tensors(&self, specs: &[TensorSpec]) -> Result<Vec<TensorRef>> {
        let mut tensors = Vec::with_capacity(specs.len());
        for spec in specs {
            tensors.push(self.load_tensor(spec).await?);
        }
        Ok(tensors)
    }

    async fn is_available(&self, _source: &WeightSource) -> bool {
        true
    }

    async fn get_metadata(&self, _source: &WeightSource) -> Result<WeightMetadata> {
        Ok(WeightMetadata {
            tensors: HashMap::new(),
            format: WeightFormat::SafeTensors,
            total_size_bytes: 1024 * 1024,
            dtypes: vec![DataType::FP16],
            extra: HashMap::new(),
        })
    }

    async fn preload(&self, _source: &WeightSource) -> Result<()> {
        Ok(())
    }

    fn capabilities(&self) -> WeightLoaderCapabilities {
        WeightLoaderCapabilities {
            supported_formats: vec![WeightFormat::SafeTensors],
            supported_sources: vec![WeightSourceType::File, WeightSourceType::HuggingFace],
            max_tensor_size: 10 * 1024 * 1024 * 1024, // 10GB
            supports_streaming: false,
            supports_concurrent: false,
            supported_transformations: vec![
                TransformationType::Transpose,
                TransformationType::Reshape,
                TransformationType::Cast,
            ],
        }
    }
}

/// Placeholder SafeTensors loader
pub struct SafeTensorsLoader;

/// Placeholder GGUF loader
pub struct GGUFLoader;
