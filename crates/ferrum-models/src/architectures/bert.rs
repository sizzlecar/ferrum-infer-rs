//! BERT architecture using Candle's built-in implementation
//! BERT is an encoder model used for embeddings and classification tasks

use candle_core::{DType, Device as CandleDevice, Tensor, D};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config as BertConfig, HiddenAct};
use ferrum_types::{FerrumError, Result};
use parking_lot::Mutex;
use tracing::{debug, info};

/// BERT model wrapper for embeddings
pub struct BertModelWrapper {
    model: Mutex<BertModel>,
    config: BertConfig,
    device: CandleDevice,
    dtype: DType,
}

impl BertModelWrapper {
    /// Create from VarBuilder and config
    pub fn from_varbuilder(
        vb: VarBuilder,
        config: &crate::definition::ModelDefinition,
        device: CandleDevice,
        dtype: DType,
    ) -> Result<Self> {
        info!("🔨 Creating BERT model from weights...");

        // Build Candle's BERT config
        let bert_config = BertConfig {
            vocab_size: config.vocab_size,
            hidden_size: config.hidden_size,
            num_hidden_layers: config.num_hidden_layers,
            num_attention_heads: config.num_attention_heads,
            intermediate_size: config.intermediate_size,
            hidden_act: HiddenAct::Gelu,
            hidden_dropout_prob: 0.1,
            max_position_embeddings: config.max_position_embeddings,
            type_vocab_size: 2,
            initializer_range: 0.02,
            layer_norm_eps: config.norm_eps,
            pad_token_id: 0,
            position_embedding_type: candle_transformers::models::bert::PositionEmbeddingType::Absolute,
            use_cache: true,
            classifier_dropout: None,
            model_type: Some("bert".to_string()),
        };

        debug!(
            "BERT config: hidden={}, layers={}, heads={}",
            bert_config.hidden_size,
            bert_config.num_hidden_layers,
            bert_config.num_attention_heads,
        );

        // Load model
        let model = BertModel::load(vb, &bert_config)
            .map_err(|e| FerrumError::model(format!("Failed to create BERT model: {}", e)))?;

        info!("✅ BERT model created successfully");

        Ok(Self {
            model: Mutex::new(model),
            config: bert_config,
            device,
            dtype,
        })
    }

    /// Load from config.json path
    pub fn from_config_json(
        vb: VarBuilder,
        config_path: &std::path::Path,
        device: CandleDevice,
        dtype: DType,
    ) -> Result<Self> {
        info!("🔨 Loading BERT model from config: {:?}", config_path);

        let config_content = std::fs::read_to_string(config_path)
            .map_err(|e| FerrumError::model(format!("Failed to read config: {}", e)))?;
        
        let bert_config: BertConfig = serde_json::from_str(&config_content)
            .map_err(|e| FerrumError::model(format!("Failed to parse BERT config: {}", e)))?;

        debug!(
            "BERT config: hidden={}, layers={}, heads={}",
            bert_config.hidden_size,
            bert_config.num_hidden_layers,
            bert_config.num_attention_heads,
        );

        let model = BertModel::load(vb, &bert_config)
            .map_err(|e| FerrumError::model(format!("Failed to create BERT model: {}", e)))?;

        info!("✅ BERT model created successfully");

        Ok(Self {
            model: Mutex::new(model),
            config: bert_config,
            device,
            dtype,
        })
    }

    /// Forward pass to get embeddings
    /// Returns the pooled output (CLS token representation) for sentence embeddings
    pub fn forward(&self, input_ids: &Tensor, token_type_ids: &Tensor) -> Result<Tensor> {
        let model = self.model.lock();

        let output = model
            .forward(input_ids, token_type_ids, None)
            .map_err(|e| FerrumError::model(format!("BERT forward failed: {}", e)))?;

        Ok(output)
    }

    /// Get sentence embedding (mean pooling over sequence)
    pub fn get_sentence_embedding(
        &self,
        input_ids: &Tensor,
        token_type_ids: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let hidden_states = self.forward(input_ids, token_type_ids)?;
        
        // Mean pooling over sequence dimension (dim 1)
        let embedding = if let Some(mask) = attention_mask {
            // Expand mask to hidden size
            let mask = mask.unsqueeze(2)
                .map_err(|e| FerrumError::model(format!("unsqueeze failed: {}", e)))?
                .broadcast_as(hidden_states.shape())
                .map_err(|e| FerrumError::model(format!("broadcast_as failed: {}", e)))?
                .to_dtype(hidden_states.dtype())
                .map_err(|e| FerrumError::model(format!("to_dtype failed: {}", e)))?;
            
            // Masked mean
            let masked = hidden_states.broadcast_mul(&mask)
                .map_err(|e| FerrumError::model(format!("broadcast_mul failed: {}", e)))?;
            let sum = masked.sum(1)
                .map_err(|e| FerrumError::model(format!("sum failed: {}", e)))?;
            let count = mask.sum(1)
                .map_err(|e| FerrumError::model(format!("mask sum failed: {}", e)))?
                .clamp(1e-9, f64::MAX)
                .map_err(|e| FerrumError::model(format!("clamp failed: {}", e)))?;
            sum.broadcast_div(&count)
                .map_err(|e| FerrumError::model(format!("broadcast_div failed: {}", e)))?
        } else {
            // Simple mean over sequence dimension
            hidden_states.mean(1)
                .map_err(|e| FerrumError::model(format!("mean failed: {}", e)))?
        };

        Ok(embedding)
    }

    /// Get CLS token embedding
    pub fn get_cls_embedding(
        &self,
        input_ids: &Tensor,
        token_type_ids: &Tensor,
    ) -> Result<Tensor> {
        let hidden_states = self.forward(input_ids, token_type_ids)?;
        
        // Get first token (CLS) - shape [batch, seq, hidden] -> [batch, hidden]
        hidden_states
            .narrow(1, 0, 1)
            .map_err(|e| FerrumError::model(format!("Failed to narrow: {}", e)))?
            .squeeze(1)
            .map_err(|e| FerrumError::model(format!("Failed to squeeze: {}", e)))
    }

    /// Get config reference
    pub fn config(&self) -> &BertConfig {
        &self.config
    }

    /// Get device
    pub fn device(&self) -> &CandleDevice {
        &self.device
    }

    /// Get dtype
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Get hidden size
    pub fn hidden_size(&self) -> usize {
        self.config.hidden_size
    }
}

