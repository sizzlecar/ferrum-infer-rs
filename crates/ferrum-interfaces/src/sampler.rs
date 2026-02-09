//! Sampling and logits processing interfaces
//!
//! This module provides abstractions for sampling tokens from model outputs,
//! including various sampling strategies and logits processors. These are
//! completely separate from model execution to allow for flexible composition.

use ferrum_types::{Result, SamplingParams, TokenId};
use rand::RngCore;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Sampling context passed to logits processors and samplers
#[derive(Debug)]
pub struct SamplingContext<'a> {
    /// Current generation step (0-based)
    pub step: usize,
    /// Request-specific sampling parameters
    pub sampling_params: &'a SamplingParams,
    /// Current logits (mutable for processing)
    pub logits: &'a mut [f32],
    /// Previous token IDs in sequence  
    pub previous_tokens: &'a [TokenId],
    /// Token frequencies for repetition penalty
    pub token_frequencies: &'a HashMap<TokenId, usize>,
    /// Vocabulary size
    pub vocab_size: usize,
    /// Additional metadata
    pub metadata: HashMap<String, f32>,
}

impl<'a> SamplingContext<'a> {
    /// Create new sampling context
    pub fn new(
        step: usize,
        sampling_params: &'a SamplingParams,
        logits: &'a mut [f32],
        previous_tokens: &'a [TokenId],
        token_frequencies: &'a HashMap<TokenId, usize>,
        vocab_size: usize,
    ) -> Self {
        Self {
            step,
            sampling_params,
            logits,
            previous_tokens,
            token_frequencies,
            vocab_size,
            metadata: HashMap::new(),
        }
    }

    /// Get logit value for specific token
    pub fn get_logit(&self, token_id: TokenId) -> Option<f32> {
        if usize::from(token_id) < self.logits.len() {
            Some(self.logits[usize::from(token_id)])
        } else {
            None
        }
    }

    /// Set logit value for specific token
    pub fn set_logit(&mut self, token_id: TokenId, value: f32) -> bool {
        if usize::from(token_id) < self.logits.len() {
            self.logits[usize::from(token_id)] = value;
            true
        } else {
            false
        }
    }

    /// Mask (set to negative infinity) specific tokens
    pub fn mask_tokens(&mut self, token_ids: &[TokenId]) {
        for &token_id in token_ids {
            if usize::from(token_id) < self.logits.len() {
                self.logits[usize::from(token_id)] = f32::NEG_INFINITY;
            }
        }
    }
}

/// Logits processor trait for modifying raw model outputs
pub trait LogitsProcessor: Send + Sync {
    /// Process logits in-place
    fn process(&self, ctx: &mut SamplingContext) -> Result<()>;

    /// Get processor name for debugging/logging
    fn name(&self) -> &str;

    /// Whether this processor should be applied before others
    fn priority(&self) -> ProcessorPriority {
        ProcessorPriority::Normal
    }
}

/// Priority levels for logits processors
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ProcessorPriority {
    /// Applied first (e.g., hard constraints, token masking)
    High = 3,
    /// Normal processing order
    Normal = 2,
    /// Applied later (e.g., temperature scaling)
    Low = 1,
}

/// Token sampler trait for selecting next token from processed logits
pub trait Sampler: Send + Sync {
    /// Sample next token from logits
    fn sample(&self, logits: &[f32], rng: &mut dyn RngCore) -> Result<TokenId>;

    /// Sample with additional context (default implementation ignores context)
    fn sample_with_context(&self, ctx: &SamplingContext, rng: &mut dyn RngCore) -> Result<TokenId> {
        self.sample(ctx.logits, rng)
    }

    /// Get sampler name
    fn name(&self) -> &str;

    /// Whether this sampler is deterministic
    fn is_deterministic(&self) -> bool;
}

/// Multi-sample capability for beam search and parallel sampling
pub trait MultiSampler: Sampler {
    /// Sample multiple tokens at once
    fn sample_multiple(
        &self,
        logits: &[f32],
        num_samples: usize,
        rng: &mut dyn RngCore,
    ) -> Result<Vec<TokenId>>;

    /// Sample with probabilities for each token
    fn sample_with_probabilities(
        &self,
        logits: &[f32],
        rng: &mut dyn RngCore,
    ) -> Result<(TokenId, Vec<f32>)>;
}

/// Logits processor chain for composing multiple processors
pub struct LogitsProcessorChain {
    processors: Vec<Box<dyn LogitsProcessor>>,
}

impl LogitsProcessorChain {
    /// Create new processor chain
    pub fn new() -> Self {
        Self {
            processors: Vec::new(),
        }
    }

    /// Add processor to chain
    pub fn add_processor(mut self, processor: Box<dyn LogitsProcessor>) -> Self {
        self.processors.push(processor);
        // Sort by priority (high to low)
        self.processors
            .sort_by(|a, b| b.priority().cmp(&a.priority()));
        self
    }

    /// Process logits through entire chain
    pub fn process(&self, ctx: &mut SamplingContext) -> Result<()> {
        for processor in &self.processors {
            processor.process(ctx)?;
        }
        Ok(())
    }

    /// Get all processor names in order
    pub fn processor_names(&self) -> Vec<&str> {
        self.processors.iter().map(|p| p.name()).collect()
    }
}

impl Default for LogitsProcessorChain {
    fn default() -> Self {
        Self::new()
    }
}

/// Common logits processors

/// Temperature scaling processor
pub struct TemperatureProcessor {
    pub temperature: f32,
}

impl TemperatureProcessor {
    pub fn new(temperature: f32) -> Self {
        Self { temperature }
    }
}

impl LogitsProcessor for TemperatureProcessor {
    fn process(&self, ctx: &mut SamplingContext) -> Result<()> {
        if self.temperature > 0.0 && self.temperature != 1.0 {
            for logit in ctx.logits.iter_mut() {
                *logit /= self.temperature;
            }
        }
        Ok(())
    }

    fn name(&self) -> &str {
        "temperature"
    }

    fn priority(&self) -> ProcessorPriority {
        ProcessorPriority::Low // Apply temperature scaling last
    }
}

/// Top-k filtering processor
pub struct TopKProcessor {
    pub k: usize,
}

impl TopKProcessor {
    pub fn new(k: usize) -> Self {
        Self { k }
    }
}

impl LogitsProcessor for TopKProcessor {
    fn process(&self, ctx: &mut SamplingContext) -> Result<()> {
        if self.k > 0 && self.k < ctx.logits.len() {
            // Find k-th largest logit
            let mut indices: Vec<usize> = (0..ctx.logits.len()).collect();
            indices.sort_by(|&a, &b| {
                ctx.logits[b]
                    .partial_cmp(&ctx.logits[a])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            let threshold = ctx.logits[indices[self.k - 1]];

            // Mask tokens below threshold
            for logit in ctx.logits.iter_mut() {
                if *logit < threshold {
                    *logit = f32::NEG_INFINITY;
                }
            }
        }
        Ok(())
    }

    fn name(&self) -> &str {
        "top_k"
    }
}

/// Top-p (nucleus) filtering processor
pub struct TopPProcessor {
    pub p: f32,
}

impl TopPProcessor {
    pub fn new(p: f32) -> Self {
        Self { p }
    }
}

impl LogitsProcessor for TopPProcessor {
    fn process(&self, ctx: &mut SamplingContext) -> Result<()> {
        if self.p < 1.0 && self.p > 0.0 {
            // Convert logits to probabilities
            let max_logit = ctx.logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let mut probs: Vec<f32> = ctx
                .logits
                .iter()
                .map(|&logit| (logit - max_logit).exp())
                .collect();

            let sum: f32 = probs.iter().sum();
            for prob in probs.iter_mut() {
                *prob /= sum;
            }

            // Sort by probability
            let mut indices: Vec<usize> = (0..probs.len()).collect();
            indices.sort_by(|&a, &b| {
                probs[b]
                    .partial_cmp(&probs[a])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            // Find cumulative probability threshold
            let mut cum_prob = 0.0;
            let mut cutoff_idx = probs.len();

            for (i, &idx) in indices.iter().enumerate() {
                cum_prob += probs[idx];
                if cum_prob > self.p {
                    cutoff_idx = i + 1;
                    break;
                }
            }

            // Mask tokens beyond cutoff
            for (i, &idx) in indices.iter().enumerate() {
                if i >= cutoff_idx {
                    ctx.logits[idx] = f32::NEG_INFINITY;
                }
            }
        }
        Ok(())
    }

    fn name(&self) -> &str {
        "top_p"
    }
}

/// Repetition penalty processor
pub struct RepetitionPenaltyProcessor {
    pub penalty: f32,
}

impl RepetitionPenaltyProcessor {
    pub fn new(penalty: f32) -> Self {
        Self { penalty }
    }
}

impl LogitsProcessor for RepetitionPenaltyProcessor {
    fn process(&self, ctx: &mut SamplingContext) -> Result<()> {
        if self.penalty != 1.0 {
            for &token_id in ctx.previous_tokens {
                if let Some(freq) = ctx.token_frequencies.get(&token_id) {
                    if usize::from(token_id) < ctx.logits.len() {
                        let idx = usize::from(token_id);
                        let current_logit = ctx.logits[idx];
                        let penalty_factor = self.penalty.powi(*freq as i32);

                        if current_logit > 0.0 {
                            ctx.logits[idx] = current_logit / penalty_factor;
                        } else {
                            ctx.logits[idx] = current_logit * penalty_factor;
                        }
                    }
                }
            }
        }
        Ok(())
    }

    fn name(&self) -> &str {
        "repetition_penalty"
    }

    fn priority(&self) -> ProcessorPriority {
        ProcessorPriority::High // Apply penalties early
    }
}

/// Common samplers

/// Greedy sampler (always picks highest probability token)
pub struct GreedySampler;

impl Sampler for GreedySampler {
    fn sample(&self, logits: &[f32], _rng: &mut dyn RngCore) -> Result<TokenId> {
        let max_idx = logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx)
            .ok_or_else(|| ferrum_types::FerrumError::backend("Empty logits for sampling"))?;

        Ok(TokenId::new(max_idx as u32))
    }

    fn name(&self) -> &str {
        "greedy"
    }

    fn is_deterministic(&self) -> bool {
        true
    }
}

/// Multinomial sampler for probabilistic sampling
pub struct MultinomialSampler;

impl Sampler for MultinomialSampler {
    fn sample(&self, logits: &[f32], rng: &mut dyn RngCore) -> Result<TokenId> {
        // Convert logits to probabilities
        let max_logit = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

        let mut probs: Vec<f32> = logits
            .iter()
            .map(|&logit| {
                if logit.is_finite() && logit > f32::NEG_INFINITY {
                    (logit - max_logit).exp()
                } else {
                    0.0
                }
            })
            .collect();

        let sum: f32 = probs.iter().sum();
        if sum <= 0.0 {
            return Err(ferrum_types::FerrumError::backend(
                "No valid tokens for sampling",
            ));
        }

        for prob in probs.iter_mut() {
            *prob /= sum;
        }

        // Sample from categorical distribution
        let threshold = rng.next_u32() as f32 / u32::MAX as f32;
        let mut cumulative = 0.0;

        for (idx, prob) in probs.iter().enumerate() {
            cumulative += prob;
            if cumulative >= threshold {
                return Ok(TokenId::new(idx as u32));
            }
        }

        // Fallback to last token (shouldn't happen with proper normalization)
        Ok(TokenId::new((probs.len() - 1) as u32))
    }

    fn name(&self) -> &str {
        "multinomial"
    }

    fn is_deterministic(&self) -> bool {
        false
    }
}

/// Sampling configuration builder
pub struct SamplingConfigBuilder {
    processors: Vec<Box<dyn LogitsProcessor>>,
    sampler: Option<Box<dyn Sampler>>,
}

impl SamplingConfigBuilder {
    /// Create new builder
    pub fn new() -> Self {
        Self {
            processors: Vec::new(),
            sampler: None,
        }
    }

    /// Add temperature scaling
    pub fn with_temperature(mut self, temperature: f32) -> Self {
        if temperature > 0.0 && temperature != 1.0 {
            self.processors
                .push(Box::new(TemperatureProcessor::new(temperature)));
        }
        self
    }

    /// Add top-k filtering
    pub fn with_top_k(mut self, k: usize) -> Self {
        if k > 0 {
            self.processors.push(Box::new(TopKProcessor::new(k)));
        }
        self
    }

    /// Add top-p filtering
    pub fn with_top_p(mut self, p: f32) -> Self {
        if p > 0.0 && p < 1.0 {
            self.processors.push(Box::new(TopPProcessor::new(p)));
        }
        self
    }

    /// Add repetition penalty
    pub fn with_repetition_penalty(mut self, penalty: f32) -> Self {
        if penalty != 1.0 {
            self.processors
                .push(Box::new(RepetitionPenaltyProcessor::new(penalty)));
        }
        self
    }

    /// Set sampler (greedy vs multinomial)
    pub fn with_sampler(mut self, sampler: Box<dyn Sampler>) -> Self {
        self.sampler = Some(sampler);
        self
    }

    /// Build sampling configuration
    pub fn build(self) -> SamplingConfig {
        let mut chain = LogitsProcessorChain::new();
        for processor in self.processors {
            chain = chain.add_processor(processor);
        }

        let sampler = self.sampler.unwrap_or_else(|| Box::new(MultinomialSampler));

        SamplingConfig {
            processor_chain: chain,
            sampler,
        }
    }
}

impl Default for SamplingConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Complete sampling configuration
pub struct SamplingConfig {
    pub processor_chain: LogitsProcessorChain,
    pub sampler: Box<dyn Sampler>,
}

impl SamplingConfig {
    /// Create from sampling parameters
    pub fn from_params(params: &SamplingParams) -> Self {
        let mut builder = SamplingConfigBuilder::new()
            .with_temperature(params.temperature)
            .with_repetition_penalty(params.repetition_penalty);

        if let Some(top_k) = params.top_k {
            builder = builder.with_top_k(top_k);
        }

        if params.top_p < 1.0 {
            builder = builder.with_top_p(params.top_p);
        }

        // Choose sampler based on temperature
        let sampler: Box<dyn Sampler> = if params.temperature == 0.0 {
            Box::new(GreedySampler)
        } else {
            Box::new(MultinomialSampler)
        };

        builder.with_sampler(sampler).build()
    }

    /// Process logits and sample token
    pub fn sample(&self, mut ctx: SamplingContext, rng: &mut dyn RngCore) -> Result<TokenId> {
        // Apply all logits processors
        self.processor_chain.process(&mut ctx)?;

        // Sample token
        self.sampler.sample_with_context(&ctx, rng)
    }
}

/// Sampling statistics for monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SamplingStats {
    /// Total sampling operations
    pub total_samples: u64,
    /// Average sampling time in microseconds
    pub avg_sample_time_us: f64,
    /// Distribution of sampled tokens
    pub token_distribution: HashMap<TokenId, u64>,
    /// Effective temperature (entropy-based measure)
    pub effective_temperature: f32,
    /// Processor execution times
    pub processor_times: HashMap<String, f64>,
}
