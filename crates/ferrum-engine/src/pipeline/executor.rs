//! Pipeline Executor
//!
//! This module provides a high-level pipeline executor that manages
//! the separation between prefill and decode phases for optimal efficiency.

use super::chunked_prefill::{ChunkedPrefillConfig, ChunkedPrefillExecutor};
use ferrum_interfaces::{
    model_executor::DecodeInput, KvCacheHandle, ModelExecutor, Sampler, TensorRef,
};
use ferrum_models::CandleTensorWrapper;
use ferrum_types::{FerrumError, Priority, Result, SamplingParams, TokenId};
use parking_lot::RwLock;
use rand::{rngs::StdRng, SeedableRng};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;
use tracing::{debug, info};

/// Current execution phase
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionPhase {
    /// Processing initial prompt
    Prefill,
    /// Generating tokens
    Decode,
    /// Completed
    Done,
}

/// Pipeline configuration
#[derive(Debug, Clone)]
pub struct PipelineConfig {
    /// Chunked prefill configuration
    pub chunked_prefill: ChunkedPrefillConfig,
    /// Maximum decode batch size
    pub max_decode_batch: usize,
    /// Maximum prefill batch size
    pub max_prefill_batch: usize,
    /// Target tokens per second for scheduling
    pub target_tps: Option<f32>,
    /// Enable prefill-decode interleaving
    pub enable_interleaving: bool,
    /// Prefill priority ratio (0.0-1.0)
    pub prefill_priority: f32,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            chunked_prefill: ChunkedPrefillConfig::default(),
            max_decode_batch: 256,
            max_prefill_batch: 8,
            target_tps: None,
            enable_interleaving: true,
            prefill_priority: 0.5,
        }
    }
}

/// Sequence being processed in the pipeline
#[derive(Debug)]
pub struct PipelineSequence {
    /// Unique sequence ID
    pub id: u64,
    /// Current phase
    pub phase: ExecutionPhase,
    /// Input tokens (for prefill)
    pub input_tokens: Vec<TokenId>,
    /// Generated tokens
    pub generated_tokens: Vec<TokenId>,
    /// KV cache handle
    pub kv_cache: Option<Arc<dyn KvCacheHandle>>,
    /// Sampling parameters
    pub sampling_params: SamplingParams,
    /// Priority
    pub priority: Priority,
    /// Random number generator
    pub rng: StdRng,
    /// Prefill start time
    pub prefill_start: Option<Instant>,
    /// First token time (TTFT)
    pub first_token_time: Option<Instant>,
    /// Total time to complete
    pub completion_time: Option<Instant>,
}

impl PipelineSequence {
    /// Create new sequence
    pub fn new(
        id: u64,
        input_tokens: Vec<TokenId>,
        sampling_params: SamplingParams,
        priority: Priority,
    ) -> Self {
        let seed = sampling_params.seed.unwrap_or(42);
        Self {
            id,
            phase: ExecutionPhase::Prefill,
            input_tokens,
            generated_tokens: Vec::new(),
            kv_cache: None,
            sampling_params,
            priority,
            rng: StdRng::seed_from_u64(seed),
            prefill_start: None,
            first_token_time: None,
            completion_time: None,
        }
    }

    /// Total tokens in sequence
    pub fn total_tokens(&self) -> usize {
        self.input_tokens.len() + self.generated_tokens.len()
    }

    /// Check if generation should stop
    pub fn should_stop(&self, vocab_size: usize) -> bool {
        // Check max tokens
        if self.generated_tokens.len() >= self.sampling_params.max_tokens {
            return true;
        }

        // Check for EOS token (last ~10 tokens are usually special)
        if let Some(&last_token) = self.generated_tokens.last() {
            if last_token.get() >= (vocab_size.saturating_sub(10)) as u32 {
                return true;
            }
        }

        false
    }

    /// Get time to first token in milliseconds
    pub fn ttft_ms(&self) -> Option<u64> {
        match (self.prefill_start, self.first_token_time) {
            (Some(start), Some(first)) => Some(first.duration_since(start).as_millis() as u64),
            _ => None,
        }
    }
}

/// Pipeline executor for efficient prefill/decode separation
pub struct PipelineExecutor {
    /// Configuration
    config: PipelineConfig,
    /// Model executor
    model_executor: Arc<dyn ModelExecutor + Send + Sync>,
    /// Sampler
    sampler: Arc<dyn Sampler + Send + Sync>,
    /// Chunked prefill executor
    chunked_prefill: ChunkedPrefillExecutor,
    /// Active sequences
    sequences: RwLock<HashMap<u64, PipelineSequence>>,
    /// Next sequence ID
    next_id: AtomicU64,
    /// Statistics
    stats: PipelineStats,
}

/// Pipeline statistics
#[derive(Debug, Default)]
struct PipelineStats {
    total_prefill_tokens: AtomicU64,
    total_decode_tokens: AtomicU64,
    total_prefill_time_us: AtomicU64,
    total_decode_time_us: AtomicU64,
    completed_sequences: AtomicU64,
}

impl PipelineExecutor {
    /// Create new pipeline executor
    pub fn new(
        model_executor: Arc<dyn ModelExecutor + Send + Sync>,
        sampler: Arc<dyn Sampler + Send + Sync>,
        config: PipelineConfig,
    ) -> Self {
        info!(
            "Creating PipelineExecutor: max_decode_batch={}, max_prefill_batch={}",
            config.max_decode_batch, config.max_prefill_batch
        );

        let chunked_prefill =
            ChunkedPrefillExecutor::new(model_executor.clone(), config.chunked_prefill.clone());

        Self {
            config,
            model_executor,
            sampler,
            chunked_prefill,
            sequences: RwLock::new(HashMap::new()),
            next_id: AtomicU64::new(0),
            stats: PipelineStats::default(),
        }
    }

    /// Submit a new sequence for processing
    pub fn submit(
        &self,
        input_tokens: Vec<TokenId>,
        sampling_params: SamplingParams,
        priority: Priority,
    ) -> u64 {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let sequence = PipelineSequence::new(id, input_tokens, sampling_params, priority);

        self.sequences.write().insert(id, sequence);
        debug!("Submitted sequence {} for processing", id);

        id
    }

    /// Execute prefill for a sequence
    pub async fn run_prefill(&self, sequence_id: u64) -> Result<()> {
        let tokens = {
            let mut sequences = self.sequences.write();
            let seq = sequences
                .get_mut(&sequence_id)
                .ok_or_else(|| FerrumError::internal("Sequence not found"))?;

            seq.phase = ExecutionPhase::Prefill;
            seq.prefill_start = Some(Instant::now());
            seq.input_tokens.clone()
        };

        let start = Instant::now();
        let output = self.chunked_prefill.execute(tokens.clone()).await?;
        let prefill_time = start.elapsed();

        // Sample first token
        let logits_vec = output.logits.to_vec_f32()?;
        let first_token = {
            let mut sequences = self.sequences.write();
            let seq = sequences
                .get_mut(&sequence_id)
                .ok_or_else(|| FerrumError::internal("Sequence not found"))?;

            let token = self.sampler.sample(&logits_vec, &mut seq.rng)?;
            seq.generated_tokens.push(token);
            seq.kv_cache = Some(output.kv_cache);
            seq.phase = ExecutionPhase::Decode;
            seq.first_token_time = Some(Instant::now());

            token
        };

        // Update stats
        self.stats
            .total_prefill_tokens
            .fetch_add(tokens.len() as u64, Ordering::Relaxed);
        self.stats
            .total_prefill_time_us
            .fetch_add(prefill_time.as_micros() as u64, Ordering::Relaxed);

        debug!(
            "Prefill complete for seq {}: {} tokens in {:?}, first token: {}",
            sequence_id,
            tokens.len(),
            prefill_time,
            first_token.get()
        );

        Ok(())
    }

    /// Execute a single decode step for a sequence
    pub async fn run_decode_step(&self, sequence_id: u64) -> Result<Option<TokenId>> {
        let (last_token, kv_cache) = {
            let sequences = self.sequences.read();
            let seq = sequences
                .get(&sequence_id)
                .ok_or_else(|| FerrumError::internal("Sequence not found"))?;

            if seq.phase != ExecutionPhase::Decode {
                return Err(FerrumError::internal("Sequence not in decode phase"));
            }

            let last_token = seq
                .generated_tokens
                .last()
                .copied()
                .ok_or_else(|| FerrumError::internal("No tokens generated"))?;

            let kv_cache = seq
                .kv_cache
                .as_ref()
                .ok_or_else(|| FerrumError::internal("No KV cache"))?
                .clone();

            (last_token, kv_cache)
        };

        let start = Instant::now();

        // Create decode input
        let tensor = candle_core::Tensor::new(&[last_token.get()], &candle_core::Device::Cpu)
            .map_err(|e| FerrumError::model(format!("Tensor error: {}", e)))?
            .unsqueeze(0)
            .map_err(|e| FerrumError::model(format!("Unsqueeze error: {}", e)))?;

        let tensor_ref: TensorRef = Arc::new(CandleTensorWrapper::new(tensor));
        let decode_input = DecodeInput::new(tensor_ref, kv_cache);

        // Run decode
        let output = self.model_executor.decode(&decode_input).await?;
        let decode_time = start.elapsed();

        // Sample next token
        let logits_vec = output.logits.to_vec_f32()?;
        let (next_token, should_stop) = {
            let mut sequences = self.sequences.write();
            let seq = sequences
                .get_mut(&sequence_id)
                .ok_or_else(|| FerrumError::internal("Sequence not found"))?;

            let token = self.sampler.sample(&logits_vec, &mut seq.rng)?;
            seq.generated_tokens.push(token);
            seq.kv_cache = Some(output.kv_cache);

            let vocab_size = self.model_executor.info().vocab_size;
            let stop = seq.should_stop(vocab_size);

            if stop {
                seq.phase = ExecutionPhase::Done;
                seq.completion_time = Some(Instant::now());
            }

            (token, stop)
        };

        // Update stats
        self.stats
            .total_decode_tokens
            .fetch_add(1, Ordering::Relaxed);
        self.stats
            .total_decode_time_us
            .fetch_add(decode_time.as_micros() as u64, Ordering::Relaxed);

        if should_stop {
            self.stats
                .completed_sequences
                .fetch_add(1, Ordering::Relaxed);
            debug!(
                "Decode complete for seq {}: generated {} tokens",
                sequence_id,
                {
                    let sequences = self.sequences.read();
                    sequences
                        .get(&sequence_id)
                        .map(|s| s.generated_tokens.len())
                        .unwrap_or(0)
                }
            );
            return Ok(None);
        }

        Ok(Some(next_token))
    }

    /// Run a full generation for a sequence
    pub async fn generate(&self, sequence_id: u64) -> Result<Vec<TokenId>> {
        // Run prefill
        self.run_prefill(sequence_id).await?;

        // Run decode loop
        loop {
            match self.run_decode_step(sequence_id).await? {
                Some(_token) => continue,
                None => break,
            }
        }

        // Get generated tokens
        let sequences = self.sequences.read();
        let seq = sequences
            .get(&sequence_id)
            .ok_or_else(|| FerrumError::internal("Sequence not found"))?;

        Ok(seq.generated_tokens.clone())
    }

    /// Get sequence by ID
    pub fn get_sequence(&self, sequence_id: u64) -> Option<PipelineSequence> {
        self.sequences.read().get(&sequence_id).cloned()
    }

    /// Remove completed sequence
    pub fn remove_sequence(&self, sequence_id: u64) -> Option<PipelineSequence> {
        self.sequences.write().remove(&sequence_id)
    }

    /// Get number of active sequences
    pub fn active_count(&self) -> usize {
        self.sequences.read().len()
    }

    /// Get sequences in prefill phase
    pub fn prefill_count(&self) -> usize {
        self.sequences
            .read()
            .values()
            .filter(|s| s.phase == ExecutionPhase::Prefill)
            .count()
    }

    /// Get sequences in decode phase
    pub fn decode_count(&self) -> usize {
        self.sequences
            .read()
            .values()
            .filter(|s| s.phase == ExecutionPhase::Decode)
            .count()
    }

    /// Get statistics
    pub fn get_stats(&self) -> PipelineStatsSnapshot {
        let total_prefill = self.stats.total_prefill_tokens.load(Ordering::Relaxed);
        let total_decode = self.stats.total_decode_tokens.load(Ordering::Relaxed);
        let prefill_time_us = self.stats.total_prefill_time_us.load(Ordering::Relaxed);
        let decode_time_us = self.stats.total_decode_time_us.load(Ordering::Relaxed);
        let completed = self.stats.completed_sequences.load(Ordering::Relaxed);

        PipelineStatsSnapshot {
            total_prefill_tokens: total_prefill,
            total_decode_tokens: total_decode,
            avg_prefill_tokens_per_sec: if prefill_time_us > 0 {
                (total_prefill as f64 * 1_000_000.0) / prefill_time_us as f64
            } else {
                0.0
            },
            avg_decode_tokens_per_sec: if decode_time_us > 0 {
                (total_decode as f64 * 1_000_000.0) / decode_time_us as f64
            } else {
                0.0
            },
            completed_sequences: completed,
            active_sequences: self.active_count() as u64,
        }
    }

    /// Get configuration
    pub fn config(&self) -> &PipelineConfig {
        &self.config
    }
}

/// Snapshot of pipeline statistics
#[derive(Debug, Clone)]
pub struct PipelineStatsSnapshot {
    pub total_prefill_tokens: u64,
    pub total_decode_tokens: u64,
    pub avg_prefill_tokens_per_sec: f64,
    pub avg_decode_tokens_per_sec: f64,
    pub completed_sequences: u64,
    pub active_sequences: u64,
}

impl std::fmt::Debug for PipelineExecutor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PipelineExecutor")
            .field("active_sequences", &self.active_count())
            .field("prefill_count", &self.prefill_count())
            .field("decode_count", &self.decode_count())
            .finish()
    }
}

// Implement Clone for PipelineSequence
impl Clone for PipelineSequence {
    fn clone(&self) -> Self {
        Self {
            id: self.id,
            phase: self.phase,
            input_tokens: self.input_tokens.clone(),
            generated_tokens: self.generated_tokens.clone(),
            kv_cache: self.kv_cache.clone(),
            sampling_params: self.sampling_params.clone(),
            priority: self.priority,
            rng: StdRng::seed_from_u64(self.sampling_params.seed.unwrap_or(42)),
            prefill_start: self.prefill_start,
            first_token_time: self.first_token_time,
            completion_time: self.completion_time,
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pipeline_sequence_creation() {
        let tokens: Vec<TokenId> = (0..10).map(|i| TokenId::new(i as u32)).collect();
        let params = SamplingParams::default();
        let seq = PipelineSequence::new(1, tokens, params, Priority::Normal);

        assert_eq!(seq.id, 1);
        assert_eq!(seq.phase, ExecutionPhase::Prefill);
        assert_eq!(seq.input_tokens.len(), 10);
        assert!(seq.generated_tokens.is_empty());
    }

    #[test]
    fn test_pipeline_sequence_stop_conditions() {
        let tokens: Vec<TokenId> = vec![TokenId::new(0)];
        let mut params = SamplingParams::default();
        params.max_tokens = 5;

        let mut seq = PipelineSequence::new(1, tokens, params, Priority::Normal);

        // Not yet at max tokens
        seq.generated_tokens.push(TokenId::new(1));
        seq.generated_tokens.push(TokenId::new(2));
        assert!(!seq.should_stop(32000));

        // At max tokens
        seq.generated_tokens.push(TokenId::new(3));
        seq.generated_tokens.push(TokenId::new(4));
        seq.generated_tokens.push(TokenId::new(5));
        assert!(seq.should_stop(32000));
    }

    #[test]
    fn test_pipeline_config_defaults() {
        let config = PipelineConfig::default();

        assert_eq!(config.max_decode_batch, 256);
        assert_eq!(config.max_prefill_batch, 8);
        assert!(config.enable_interleaving);
    }

    #[test]
    fn test_execution_phase() {
        assert_eq!(ExecutionPhase::Prefill, ExecutionPhase::Prefill);
        assert_ne!(ExecutionPhase::Prefill, ExecutionPhase::Decode);
    }
}
