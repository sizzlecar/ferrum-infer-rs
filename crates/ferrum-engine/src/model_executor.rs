//! ModelExecutor implementation for Candle backend
//!
//! This module provides the ModelExecutor implementation that separates
//! prefill and decode phases, following the ferrum-interfaces design.

use ferrum_interfaces::{
    ModelExecutor, PrefillInput, PrefillOutput, DecodeInput, DecodeOutput,
    TensorRef, KvCacheHandle, AllocationRequest, BlockTable,
};
use ferrum_types::{Result, FerrumError, ModelInfo, TokenId, Device, DataType};
use crate::candle_backend::{CandleModel, CandleCacheSnapshot};
use std::sync::Arc;
use async_trait::async_trait;
use tracing::{debug, instrument};

/// KV Cache handle implementation for Candle
/// This is a simplified implementation that will be replaced with proper handles
#[derive(Debug, Clone)]
pub struct CandleKvCacheHandle {
    block_table: BlockTable,
    device: Device,
    num_layers: usize,
    num_heads: usize,
    head_dim: usize,
    dtype: DataType,
    key_cache: Vec<Option<TensorRef>>,
    value_cache: Vec<Option<TensorRef>>,
    cache_id: String,
}

impl KvCacheHandle for CandleKvCacheHandle {
    fn block_table(&self) -> &BlockTable {
        &self.block_table
    }

    fn block_table_mut(&mut self) -> &mut BlockTable {
        &mut self.block_table
    }

    fn device(&self) -> Device {
        self.device
    }

    fn num_layers(&self) -> usize {
        self.num_layers
    }

    fn num_heads(&self) -> usize {
        self.num_heads
    }

    fn head_dim(&self) -> usize {
        self.head_dim
    }

    fn key_cache(&self, layer: usize) -> Result<Option<TensorRef>> {
        Ok(self.key_cache.get(layer).cloned().unwrap_or(None))
    }

    fn value_cache(&self, layer: usize) -> Result<Option<TensorRef>> {
        Ok(self.value_cache.get(layer).cloned().unwrap_or(None))
    }

    fn clone_handle(&self) -> Result<Arc<dyn KvCacheHandle>> {
        Ok(Arc::new(self.clone()) as Arc<dyn KvCacheHandle>)
    }

    fn stats(&self) -> ferrum_interfaces::CacheHandleStats {
        ferrum_interfaces::CacheHandleStats {
            memory_bytes: 0,
            blocks_allocated: self.block_table.num_blocks(),
            tokens_stored: self.block_table.sequence_length,
            utilization: 1.0,
            last_access: std::time::Instant::now(),
        }
    }

    fn is_valid(&self) -> bool {
        true
    }

    fn cache_id(&self) -> String {
        self.cache_id.clone()
    }
}

/// ModelExecutor adapter for CandleModel
pub struct CandleModelExecutor {
    model: Arc<CandleModel>,
}

impl CandleModelExecutor {
    pub fn new(model: Arc<CandleModel>) -> Self {
        Self { model }
    }

    fn build_kv_handle(
        snapshot: CandleCacheSnapshot,
        num_layers: usize,
        num_heads: usize,
        head_dim: usize,
        device: Device,
    ) -> Result<CandleKvCacheHandle> {
        let mut block_table = BlockTable::new(head_dim);
        block_table.sequence_length = snapshot.sequence_length;

        let key_cache = snapshot
            .key_cache
            .into_iter()
            .map(|layer| {
                // Pack placeholder data into tensor
                let tensor = ferrum_runtime::TensorFactoryHandle::default()
                    .from_slice(&layer, &[snapshot.sequence_length, head_dim], device)
                    .map_err(|e| FerrumError::backend(format!("Failed to build key cache tensor: {}", e)))?;
                Ok(Some(tensor))
            })
            .collect::<Result<Vec<_>>>()?;

        let value_cache = snapshot
            .value_cache
            .into_iter()
            .map(|layer| {
                let tensor = ferrum_runtime::TensorFactoryHandle::default()
                    .from_slice(&layer, &[snapshot.sequence_length, head_dim], device)
                    .map_err(|e| FerrumError::backend(format!("Failed to build value cache tensor: {}", e)))?;
                Ok(Some(tensor))
            })
            .collect::<Result<Vec<_>>>()?;

        Ok(CandleKvCacheHandle {
            block_table,
            device,
            num_layers,
            num_heads,
            head_dim,
            dtype: DataType::F32,
            key_cache,
            value_cache,
            cache_id: uuid::Uuid::new_v4().to_string(),
            snapshot: Some(snapshot),
        })
    }
}

#[async_trait]
impl ModelExecutor for CandleModelExecutor {
    fn info(&self) -> &ModelInfo {
        self.model.info()
    }

    #[instrument(skip(self, input), fields(batch_size = input.input_ids.shape().get(0).unwrap_or(&0)))]
    async fn prefill(&self, input: &PrefillInput) -> Result<PrefillOutput> {
        debug!("Running prefill with input shape: {:?}", input.input_ids.shape());
        
        // Convert TensorRef to token IDs for Candle model
        let tensor_data = input
            .input_ids
            .data_f32()
            .ok_or_else(|| FerrumError::backend("Tensor data must be accessible as f32 slice"))?;

        // Convert f32 tensor data to TokenId (u32)
        let token_ids: Vec<TokenId> = tensor_data.iter()
            .map(|&f| f as TokenId)
            .collect();

        debug!("Prefill with {} tokens", token_ids.len());

        // Use the model's forward_logits method for prefill
        let (logits_tensor, kv_cache) = self.model.forward_logits(&token_ids, None).await?;

        let logits_ref = self
            .model
            .tensor_factory()
            .as_ref()
            .from_slice(&logits_tensor.data, &logits_tensor.shape, self.model.device())
            .map_err(|e| FerrumError::backend(format!("Failed to build logits tensor: {}", e)))?;

        // Convert KV cache snapshot to handle
        let kv_handle = Arc::new(Self::build_kv_handle(
            kv_cache,
            self.model.info().num_layers,
            self.model.info().num_heads,
            self.model.info().hidden_size / self.model.info().num_heads,
            self.model.device(),
        )?) as Arc<dyn KvCacheHandle>;

        Ok(PrefillOutput {
            logits: logits_ref,
            kv: kv_handle,
        })
    }

    #[instrument(skip(self, input), fields(batch_size = input.input_ids.shape().get(0).unwrap_or(&0)))]
    async fn decode(&self, input: &DecodeInput) -> Result<DecodeOutput> {
        debug!("Running decode with input shape: {:?}", input.input_ids.shape());

        // Convert TensorRef to token IDs
        let tensor_data = input
            .input_ids
            .data_f32()
            .ok_or_else(|| FerrumError::backend("Tensor data must be accessible as f32 slice"))?;

        let token_ids: Vec<TokenId> = tensor_data.iter()
            .map(|&f| f as TokenId)
            .collect();

        debug!("Decode with {} tokens", token_ids.len());

        // Convert KV cache handle to legacy format for now
        let kv_cache_snapshot = input
            .kv
            .as_any()
            .downcast_ref::<CandleKvCacheHandle>()
            .and_then(|handle| handle.snapshot.clone());

        // Use the model's forward_logits method for decode
        let (logits_tensor, new_kv_cache) = self.model.forward_logits(&token_ids, kv_cache_snapshot.as_ref()).await?;

        let logits_ref = self
            .model
            .tensor_factory()
            .as_ref()
            .from_slice(&logits_tensor.data, &logits_tensor.shape, self.model.device())
            .map_err(|e| FerrumError::backend(format!("Failed to build logits tensor: {}", e)))?;

        // Update KV cache handle
        let kv_handle = Arc::new(Self::build_kv_handle(
            new_kv_cache,
            self.model.info().num_layers,
            self.model.info().num_heads,
            self.model.info().hidden_size / self.model.info().num_heads,
            self.model.device(),
        )?) as Arc<dyn KvCacheHandle>;

        Ok(DecodeOutput {
            logits: logits_ref,
            kv: kv_handle,
        })
    }

    #[instrument(skip(self, input))]
    async fn forward(&self, input: &TensorRef) -> Result<TensorRef> {
        debug!("Running forward pass with input shape: {:?}", input.shape());
        let tensor_data = input
            .data_f32()
            .ok_or_else(|| FerrumError::backend("Tensor data must be accessible as f32 slice for forward"))?;

        let shape = input.shape().to_vec();
        let legacy_tensor = ferrum_core::Tensor {
            data: tensor_data.to_vec(),
            shape,
            dtype: ferrum_core::DataType::FP32,
        };

        let result = self.model.forward(&legacy_tensor).await?;

        let result_ref = self
            .model
            .tensor_factory()
            .as_ref()
            .from_slice(&result.data, &result.shape, self.model.device())
            .map_err(|e| FerrumError::backend(format!("Failed to convert output tensor: {}", e)))?;

        Ok(result_ref)
    }
}
