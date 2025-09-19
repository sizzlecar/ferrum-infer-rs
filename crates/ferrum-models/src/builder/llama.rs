//! Llama family model builder

use ferrum_interfaces::{ModelBuilder, ModelInfo, ModelConfig, ModelCapabilities, ModelExecutor};
use ferrum_interfaces::{ComputeBackend, WeightLoader};
use ferrum_types::{Result, Architecture, DataType, FerrumError};
use std::sync::Arc;
use tracing::{debug, info};
use async_trait::async_trait;

/// Llama model builder
#[derive(Debug, Clone)]
pub struct LlamaModelBuilder {
    capabilities: ModelCapabilities,
}

impl LlamaModelBuilder {
    /// Create new Llama model builder
    pub fn new() -> Self {
        let capabilities = ModelCapabilities {
            supports_batch_inference: true,
            supports_streaming: true,
            supports_chat_template: true,
            supports_function_calling: false,
            supports_vision: false,
            max_sequence_length: 4096, // Default, can be overridden by config
            supported_dtypes: vec![DataType::F16, DataType::F32, DataType::BF16],
            supported_architectures: vec![
                Architecture::Llama,
                Architecture::Llama2,
                Architecture::CodeLlama,
            ],
        };

        Self { capabilities }
    }

    /// Validate model configuration
    fn validate_config(&self, config: &ModelConfig) -> Result<()> {
        // Check architecture compatibility
        if !self.capabilities.supported_architectures.contains(&config.architecture) {
            return Err(FerrumError::invalid_parameter(
                format!("Architecture {:?} not supported by LlamaModelBuilder", config.architecture)
            ));
        }

        // Validate required parameters
        if config.hidden_size == 0 {
            return Err(FerrumError::invalid_parameter("hidden_size must be positive"));
        }
        if config.num_layers == 0 {
            return Err(FerrumError::invalid_parameter("num_layers must be positive"));
        }
        if config.num_attention_heads == 0 {
            return Err(FerrumError::invalid_parameter("num_attention_heads must be positive"));
        }
        if config.vocab_size == 0 {
            return Err(FerrumError::invalid_parameter("vocab_size must be positive"));
        }

        // Validate attention head configuration
        if config.hidden_size % config.num_attention_heads != 0 {
            return Err(FerrumError::invalid_parameter(
                "hidden_size must be divisible by num_attention_heads"
            ));
        }

        debug!("Llama config validation passed");
        Ok(())
    }

    /// Create model info from config
    fn create_model_info(&self, config: &ModelConfig) -> ModelInfo {
        let head_dim = config.hidden_size / config.num_attention_heads;
        let total_params = self.estimate_parameters(config);

        ModelInfo {
            model_id: config.model_id.clone(),
            architecture: config.architecture,
            vocab_size: config.vocab_size,
            hidden_size: config.hidden_size,
            num_layers: config.num_layers,
            num_attention_heads: config.num_attention_heads,
            num_key_value_heads: config.num_key_value_heads.unwrap_or(config.num_attention_heads),
            head_dim,
            intermediate_size: config.intermediate_size.unwrap_or(config.hidden_size * 4),
            max_sequence_length: config.max_position_embeddings.unwrap_or(2048),
            rope_theta: config.rope_theta.unwrap_or(10000.0),
            rope_scaling: config.rope_scaling.clone(),
            norm_eps: config.rms_norm_eps.unwrap_or(1e-6),
            data_type: config.data_type.unwrap_or(DataType::F16),
            total_parameters: Some(total_params),
            capabilities: self.capabilities.clone(),
        }
    }

    /// Estimate total parameters
    fn estimate_parameters(&self, config: &ModelConfig) -> u64 {
        let vocab_size = config.vocab_size as u64;
        let hidden_size = config.hidden_size as u64;
        let num_layers = config.num_layers as u64;
        let intermediate_size = config.intermediate_size.unwrap_or(config.hidden_size * 4) as u64;

        // Embedding layer
        let embed_params = vocab_size * hidden_size;

        // Per-layer parameters
        let attention_params = 4 * hidden_size * hidden_size; // q, k, v, o projections
        let mlp_params = 3 * hidden_size * intermediate_size; // gate, up, down projections
        let norm_params = 2 * hidden_size; // attention norm + mlp norm
        let layer_params = attention_params + mlp_params + norm_params;

        // Output layer
        let output_params = hidden_size * vocab_size;

        // Final norm
        let final_norm_params = hidden_size;

        embed_params + (layer_params * num_layers) + output_params + final_norm_params
    }
}

impl Default for LlamaModelBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl ModelBuilder for LlamaModelBuilder {
    async fn build(
        &self,
        config: &ModelConfig,
        backend: Arc<dyn ComputeBackend + Send + Sync>,
        weight_loader: Arc<dyn WeightLoader + Send + Sync>,
    ) -> Result<Box<dyn ModelExecutor + Send + Sync>> {
        info!("Building Llama model: {:?}", config.model_id);
        
        // Validate configuration
        self.validate_config(config)?;

        // Create model info
        let model_info = self.create_model_info(config);
        debug!("Created model info: {} parameters", model_info.total_parameters.unwrap_or(0));

        // Create model executor (placeholder implementation)
        let executor = PlaceholderModelExecutor::new(model_info, backend, weight_loader);
        
        info!("Successfully built Llama model");
        Ok(Box::new(executor))
    }

    fn model_info(&self, config: &ModelConfig) -> Result<ModelInfo> {
        self.validate_config(config)?;
        Ok(self.create_model_info(config))
    }

    fn capabilities(&self) -> ModelCapabilities {
        self.capabilities.clone()
    }

    fn supported_architectures(&self) -> Vec<Architecture> {
        self.capabilities.supported_architectures.clone()
    }
}

/// Placeholder model executor for Llama
/// In a real implementation, this would contain the actual model layers
#[derive(Debug)]
struct PlaceholderModelExecutor {
    model_info: ModelInfo,
    backend: Arc<dyn ComputeBackend + Send + Sync>,
    weight_loader: Arc<dyn WeightLoader + Send + Sync>,
}

impl PlaceholderModelExecutor {
    fn new(
        model_info: ModelInfo,
        backend: Arc<dyn ComputeBackend + Send + Sync>,
        weight_loader: Arc<dyn WeightLoader + Send + Sync>,
    ) -> Self {
        Self {
            model_info,
            backend,
            weight_loader,
        }
    }
}

#[async_trait]
impl ModelExecutor for PlaceholderModelExecutor {
    fn info(&self) -> &ModelInfo {
        &self.model_info
    }

    async fn prefill(
        &self,
        input: &ferrum_interfaces::PrefillInput,
    ) -> Result<ferrum_interfaces::PrefillOutput> {
        // Placeholder implementation
        debug!("Llama prefill: processing {} tokens", input.input_ids.shape()[1]);
        
        // In real implementation, this would:
        // 1. Run embedding layer
        // 2. Run transformer layers
        // 3. Generate logits
        // 4. Create/update KV cache
        
        Err(FerrumError::not_implemented("Llama prefill not yet implemented"))
    }

    async fn decode(
        &self,
        input: &ferrum_interfaces::DecodeInput,
    ) -> Result<ferrum_interfaces::DecodeOutput> {
        // Placeholder implementation
        debug!("Llama decode: processing single token");
        
        // In real implementation, this would:
        // 1. Run embedding for new token
        // 2. Run transformer layers with KV cache
        // 3. Generate logits
        // 4. Update KV cache
        
        Err(FerrumError::not_implemented("Llama decode not yet implemented"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferrum_types::ModelId;

    fn create_test_config() -> ModelConfig {
        ModelConfig {
            model_id: ModelId::new("test-llama"),
            architecture: Architecture::Llama,
            vocab_size: 32000,
            hidden_size: 4096,
            num_layers: 32,
            num_attention_heads: 32,
            num_key_value_heads: Some(32),
            intermediate_size: Some(11008),
            max_position_embeddings: Some(2048),
            rope_theta: Some(10000.0),
            rope_scaling: None,
            rms_norm_eps: Some(1e-6),
            data_type: Some(DataType::F16),
        }
    }

    #[test]
    fn test_builder_creation() {
        let builder = LlamaModelBuilder::new();
        let archs = builder.supported_architectures();
        
        assert!(archs.contains(&Architecture::Llama));
        assert!(archs.contains(&Architecture::Llama2));
        assert!(archs.contains(&Architecture::CodeLlama));
    }

    #[test]
    fn test_config_validation() {
        let builder = LlamaModelBuilder::new();
        let config = create_test_config();
        
        assert!(builder.validate_config(&config).is_ok());
    }

    #[test]
    fn test_invalid_config() {
        let builder = LlamaModelBuilder::new();
        
        // Test zero hidden_size
        let mut config = create_test_config();
        config.hidden_size = 0;
        assert!(builder.validate_config(&config).is_err());
        
        // Test misaligned attention heads
        let mut config = create_test_config();
        config.hidden_size = 4095; // Not divisible by 32
        assert!(builder.validate_config(&config).is_err());
    }

    #[test]
    fn test_model_info_creation() {
        let builder = LlamaModelBuilder::new();
        let config = create_test_config();
        
        let info = builder.model_info(&config).unwrap();
        
        assert_eq!(info.architecture, Architecture::Llama);
        assert_eq!(info.vocab_size, 32000);
        assert_eq!(info.hidden_size, 4096);
        assert_eq!(info.num_layers, 32);
        assert_eq!(info.head_dim, 128); // 4096 / 32
        assert!(info.total_parameters.is_some());
    }

    #[test]
    fn test_parameter_estimation() {
        let builder = LlamaModelBuilder::new();
        let config = create_test_config();
        
        let params = builder.estimate_parameters(&config);
        
        // Should be a reasonable number of parameters for this config
        assert!(params > 1_000_000); // At least 1M parameters
        assert!(params < 100_000_000_000); // Less than 100B parameters
    }
}
