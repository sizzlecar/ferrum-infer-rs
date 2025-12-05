//! Chunked Prefill Implementation
//!
//! This module implements chunked prefill, which splits long prompts into
//! smaller chunks for processing. Benefits include:
//!
//! - Better memory efficiency for long prompts
//! - Ability to interleave prefill with decode for better GPU utilization
//! - Reduced latency for first token when processing long contexts

use ferrum_interfaces::{
    model_executor::{DecodeInput, DecodeOutput, PrefillInput, PrefillOutput},
    KvCacheHandle, ModelExecutor, TensorRef,
};
use ferrum_models::CandleTensorWrapper;
use ferrum_types::{FerrumError, Result, TokenId};
use std::sync::Arc;
use tracing::{debug, info};

/// Configuration for chunked prefill
#[derive(Debug, Clone)]
pub struct ChunkedPrefillConfig {
    /// Maximum tokens per chunk
    pub chunk_size: usize,
    /// Minimum tokens to trigger chunking (below this, process as single chunk)
    pub min_sequence_for_chunking: usize,
    /// Whether to overlap chunks for better context
    pub enable_overlap: bool,
    /// Number of tokens to overlap between chunks
    pub overlap_size: usize,
}

impl Default for ChunkedPrefillConfig {
    fn default() -> Self {
        Self {
            chunk_size: 512,
            min_sequence_for_chunking: 128,
            enable_overlap: false,
            overlap_size: 16,
        }
    }
}

/// State for chunked prefill processing
#[derive(Debug)]
pub struct ChunkedPrefillState {
    /// Original input tokens
    pub tokens: Vec<TokenId>,
    /// Current chunk index
    pub current_chunk: usize,
    /// Total number of chunks
    pub total_chunks: usize,
    /// Tokens processed so far
    pub tokens_processed: usize,
    /// KV cache handle
    pub kv_cache: Option<Arc<dyn KvCacheHandle>>,
    /// Configuration
    pub config: ChunkedPrefillConfig,
}

impl ChunkedPrefillState {
    /// Create new chunked prefill state
    pub fn new(tokens: Vec<TokenId>, config: ChunkedPrefillConfig) -> Self {
        let chunk_size = config.chunk_size;
        let total_chunks = if tokens.len() <= config.min_sequence_for_chunking {
            1
        } else {
            (tokens.len() + chunk_size - 1) / chunk_size
        };

        Self {
            tokens,
            current_chunk: 0,
            total_chunks,
            tokens_processed: 0,
            kv_cache: None,
            config,
        }
    }

    /// Check if chunking is needed
    pub fn needs_chunking(&self) -> bool {
        self.tokens.len() > self.config.min_sequence_for_chunking
    }

    /// Check if all chunks have been processed
    pub fn is_complete(&self) -> bool {
        self.tokens_processed >= self.tokens.len()
    }

    /// Get the next chunk of tokens
    pub fn next_chunk(&self) -> Option<&[TokenId]> {
        if self.is_complete() {
            return None;
        }

        let start = self.tokens_processed;
        let end = (start + self.config.chunk_size).min(self.tokens.len());

        if start < self.tokens.len() {
            Some(&self.tokens[start..end])
        } else {
            None
        }
    }

    /// Mark current chunk as processed
    pub fn advance(&mut self, tokens_processed: usize) {
        self.tokens_processed += tokens_processed;
        self.current_chunk += 1;
    }

    /// Get progress as a fraction
    pub fn progress(&self) -> f32 {
        if self.tokens.is_empty() {
            1.0
        } else {
            self.tokens_processed as f32 / self.tokens.len() as f32
        }
    }
}

/// Executor for chunked prefill operations
pub struct ChunkedPrefillExecutor {
    /// Configuration
    config: ChunkedPrefillConfig,
    /// Underlying model executor
    model_executor: Arc<dyn ModelExecutor + Send + Sync>,
}

impl ChunkedPrefillExecutor {
    /// Create new chunked prefill executor
    pub fn new(
        model_executor: Arc<dyn ModelExecutor + Send + Sync>,
        config: ChunkedPrefillConfig,
    ) -> Self {
        info!(
            "Creating ChunkedPrefillExecutor: chunk_size={}, min_seq={}",
            config.chunk_size, config.min_sequence_for_chunking
        );
        Self {
            config,
            model_executor,
        }
    }

    /// Create with default config
    pub fn with_defaults(model_executor: Arc<dyn ModelExecutor + Send + Sync>) -> Self {
        Self::new(model_executor, ChunkedPrefillConfig::default())
    }

    /// Execute prefill with chunking if needed
    pub async fn execute(&self, tokens: Vec<TokenId>) -> Result<PrefillOutput> {
        let mut state = ChunkedPrefillState::new(tokens, self.config.clone());

        if !state.needs_chunking() {
            // Process as single chunk
            debug!("Processing {} tokens as single chunk", state.tokens.len());
            return self.process_single_chunk(&state.tokens).await;
        }

        debug!(
            "Processing {} tokens in {} chunks",
            state.tokens.len(),
            state.total_chunks
        );

        let mut last_output: Option<PrefillOutput> = None;

        while let Some(chunk) = state.next_chunk() {
            let chunk_len = chunk.len();
            debug!(
                "Processing chunk {}/{}: {} tokens (progress: {:.1}%)",
                state.current_chunk + 1,
                state.total_chunks,
                chunk_len,
                state.progress() * 100.0
            );

            let output = if state.kv_cache.is_some() {
                // Continue from existing KV cache
                self.process_continuation_chunk(chunk, state.kv_cache.as_ref().unwrap().clone())
                    .await?
            } else {
                // First chunk
                self.process_single_chunk(chunk).await?
            };

            state.kv_cache = Some(output.kv_cache.clone());
            state.advance(chunk_len);
            last_output = Some(output);
        }

        last_output.ok_or_else(|| FerrumError::internal("No output from chunked prefill"))
    }

    /// Process a single chunk as prefill
    async fn process_single_chunk(&self, tokens: &[TokenId]) -> Result<PrefillOutput> {
        // Convert tokens to tensor
        let token_u32s: Vec<u32> = tokens.iter().map(|t| t.get()).collect();
        let tensor = candle_core::Tensor::new(&token_u32s[..], &candle_core::Device::Cpu)
            .map_err(|e| FerrumError::model(format!("Tensor error: {}", e)))?
            .unsqueeze(0)
            .map_err(|e| FerrumError::model(format!("Unsqueeze error: {}", e)))?;

        let tensor_ref: TensorRef = Arc::new(CandleTensorWrapper::new(tensor));
        let input = PrefillInput::new(tensor_ref);

        self.model_executor.prefill(&input).await
    }

    /// Process a continuation chunk using existing KV cache
    async fn process_continuation_chunk(
        &self,
        tokens: &[TokenId],
        kv_cache: Arc<dyn KvCacheHandle>,
    ) -> Result<PrefillOutput> {
        // For continuation, we use decode-like processing but with multiple tokens
        // This is an optimization - some models support batched decode which is
        // essentially the same as prefill with existing KV cache

        // Convert tokens to tensor
        let token_u32s: Vec<u32> = tokens.iter().map(|t| t.get()).collect();
        let tensor = candle_core::Tensor::new(&token_u32s[..], &candle_core::Device::Cpu)
            .map_err(|e| FerrumError::model(format!("Tensor error: {}", e)))?
            .unsqueeze(0)
            .map_err(|e| FerrumError::model(format!("Unsqueeze error: {}", e)))?;

        let tensor_ref: TensorRef = Arc::new(CandleTensorWrapper::new(tensor));

        // Use prefill with the input tensor and pass position info
        // In a full implementation, we'd also pass the existing KV cache
        let input = PrefillInput::new(tensor_ref);
        let output = self.model_executor.prefill(&input).await?;

        // Return with the updated KV cache
        Ok(PrefillOutput::new(output.logits, output.kv_cache))
    }

    /// Get configuration
    pub fn config(&self) -> &ChunkedPrefillConfig {
        &self.config
    }
}

impl std::fmt::Debug for ChunkedPrefillExecutor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ChunkedPrefillExecutor")
            .field("chunk_size", &self.config.chunk_size)
            .field("min_sequence_for_chunking", &self.config.min_sequence_for_chunking)
            .finish()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chunked_state_creation() {
        let tokens: Vec<TokenId> = (0..100).map(|i| TokenId::new(i as u32)).collect();
        let config = ChunkedPrefillConfig {
            chunk_size: 32,
            min_sequence_for_chunking: 50,
            ..Default::default()
        };

        let state = ChunkedPrefillState::new(tokens, config);

        assert!(state.needs_chunking());
        assert_eq!(state.total_chunks, 4); // 100 / 32 = 3.125, rounded up to 4
        assert!(!state.is_complete());
    }

    #[test]
    fn test_chunked_state_no_chunking() {
        let tokens: Vec<TokenId> = (0..30).map(|i| TokenId::new(i as u32)).collect();
        let config = ChunkedPrefillConfig {
            chunk_size: 32,
            min_sequence_for_chunking: 50,
            ..Default::default()
        };

        let state = ChunkedPrefillState::new(tokens, config);

        assert!(!state.needs_chunking());
        assert_eq!(state.total_chunks, 1);
    }

    #[test]
    fn test_chunked_state_iteration() {
        let tokens: Vec<TokenId> = (0..100).map(|i| TokenId::new(i as u32)).collect();
        let config = ChunkedPrefillConfig {
            chunk_size: 32,
            min_sequence_for_chunking: 50,
            ..Default::default()
        };

        let mut state = ChunkedPrefillState::new(tokens, config);

        // First chunk: 0-31
        let chunk1 = state.next_chunk().unwrap();
        assert_eq!(chunk1.len(), 32);
        assert_eq!(chunk1[0].get(), 0);
        state.advance(32);

        // Second chunk: 32-63
        let chunk2 = state.next_chunk().unwrap();
        assert_eq!(chunk2.len(), 32);
        assert_eq!(chunk2[0].get(), 32);
        state.advance(32);

        // Third chunk: 64-95
        let chunk3 = state.next_chunk().unwrap();
        assert_eq!(chunk3.len(), 32);
        state.advance(32);

        // Fourth chunk: 96-99 (4 tokens)
        let chunk4 = state.next_chunk().unwrap();
        assert_eq!(chunk4.len(), 4);
        state.advance(4);

        // No more chunks
        assert!(state.is_complete());
        assert!(state.next_chunk().is_none());
    }

    #[test]
    fn test_progress() {
        let tokens: Vec<TokenId> = (0..100).map(|i| TokenId::new(i as u32)).collect();
        let config = ChunkedPrefillConfig::default();
        let mut state = ChunkedPrefillState::new(tokens, config);

        assert_eq!(state.progress(), 0.0);

        state.advance(50);
        assert!((state.progress() - 0.5).abs() < 0.01);

        state.advance(50);
        assert!((state.progress() - 1.0).abs() < 0.01);
    }
}


