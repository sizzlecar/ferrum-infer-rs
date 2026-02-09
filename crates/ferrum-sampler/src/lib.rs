//! # Ferrum Sampler
//!
//! MVP sampler implementation for Ferrum inference stack.
//!
//! This crate provides a thin wrapper around the sampling interfaces defined in
//! `ferrum-interfaces`, offering convenient factory functions and utilities for
//! building sampling pipelines from `SamplingParams`.
//!
//! ## Design
//!
//! - **Re-export Interface Types**: All core types from `ferrum-interfaces::sampler`
//! - **Factory Pattern**: Simple factory for creating samplers and configs
//! - **Zero Overhead**: Direct delegation to interface implementations
//!
//! ## Usage
//!
//! ```no_run
//! use ferrum_sampler::{build_sampling_config, sampler_from_params};
//! use ferrum_types::SamplingParams;
//!
//! let params = SamplingParams::default();
//! let config = build_sampling_config(&params);
//! let sampler = sampler_from_params(&params);
//! ```

// Re-export all sampler types from ferrum-interfaces
pub use ferrum_interfaces::sampler::{
    GreedySampler, LogitsProcessor, LogitsProcessorChain, MultiSampler, MultinomialSampler,
    ProcessorPriority, RepetitionPenaltyProcessor, Sampler, SamplingConfig, SamplingConfigBuilder,
    SamplingContext, SamplingStats, TemperatureProcessor, TopKProcessor, TopPProcessor,
};

// Re-export types from ferrum-types
pub use ferrum_types::{Result, SamplingParams, TokenId};

use rand::RngCore;
use std::collections::HashMap;

/// Default sampler factory for creating samplers and configurations.
#[derive(Debug, Clone, Default)]
pub struct DefaultSamplerFactory;

impl DefaultSamplerFactory {
    /// Create new factory instance
    pub fn new() -> Self {
        Self
    }

    /// Build sampling configuration from parameters
    pub fn build_config(&self, params: &SamplingParams) -> SamplingConfig {
        SamplingConfig::from_params(params)
    }

    /// Create sampler instance based on temperature
    /// - temperature == 0.0 → GreedySampler (deterministic)
    /// - temperature > 0.0 → MultinomialSampler (stochastic)
    pub fn create_sampler(&self, params: &SamplingParams) -> Box<dyn Sampler + Send + Sync> {
        if params.temperature == 0.0 {
            Box::new(GreedySampler)
        } else {
            Box::new(MultinomialSampler)
        }
    }

    /// Create sampling pipeline with config and sampler
    pub fn build_pipeline(&self, params: &SamplingParams) -> SamplingPipeline {
        let config = self.build_config(params);
        SamplingPipeline { config }
    }
}

/// Sampling pipeline that combines config and execution logic.
///
/// This struct holds a `SamplingConfig` and provides a convenient interface
/// for sampling tokens with context.
pub struct SamplingPipeline {
    config: SamplingConfig,
}

impl SamplingPipeline {
    /// Create new pipeline from parameters
    pub fn new(params: &SamplingParams) -> Self {
        let config = SamplingConfig::from_params(params);
        Self { config }
    }

    /// Get reference to sampling config
    pub fn config(&self) -> &SamplingConfig {
        &self.config
    }

    /// Sample next token with full context
    ///
    /// # Arguments
    /// * `step` - Current generation step (0-based)
    /// * `logits` - Mutable logits array to process
    /// * `previous_tokens` - Previously generated tokens
    /// * `token_frequencies` - Token frequency map for penalties
    /// * `sampling_params` - Sampling parameters for this step
    /// * `rng` - Random number generator
    pub fn sample_next(
        &self,
        step: usize,
        logits: &mut [f32],
        previous_tokens: &[TokenId],
        token_frequencies: &HashMap<TokenId, usize>,
        sampling_params: &SamplingParams,
        rng: &mut dyn RngCore,
    ) -> Result<TokenId> {
        let vocab_size = logits.len();
        let ctx = SamplingContext::new(
            step,
            sampling_params,
            logits,
            previous_tokens,
            token_frequencies,
            vocab_size,
        );
        self.config.sample(ctx, rng)
    }

    /// Simple sampling without context (uses default params)
    pub fn sample_simple(&self, logits: &mut [f32], rng: &mut dyn RngCore) -> Result<TokenId> {
        let params = SamplingParams::default();
        let empty_tokens = Vec::new();
        let empty_freqs = HashMap::new();
        self.sample_next(0, logits, &empty_tokens, &empty_freqs, &params, rng)
    }
}

// ============================================================================
// Convenience Functions
// ============================================================================

/// Build sampling configuration from parameters.
///
/// This is the primary entry point for creating a `SamplingConfig`.
pub fn build_sampling_config(params: &SamplingParams) -> SamplingConfig {
    SamplingConfig::from_params(params)
}

/// Create sampler instance from parameters.
///
/// Returns a boxed `Sampler` trait object based on the temperature setting.
pub fn sampler_from_params(params: &SamplingParams) -> Box<dyn Sampler + Send + Sync> {
    DefaultSamplerFactory::new().create_sampler(params)
}

/// Build complete sampling pipeline from parameters.
pub fn pipeline_from_params(params: &SamplingParams) -> SamplingPipeline {
    DefaultSamplerFactory::new().build_pipeline(params)
}

/// Create a greedy sampler (always picks highest logit).
pub fn greedy_sampler() -> Box<dyn Sampler + Send + Sync> {
    Box::new(GreedySampler)
}

/// Create a multinomial sampler (probabilistic sampling).
pub fn multinomial_sampler() -> Box<dyn Sampler + Send + Sync> {
    Box::new(MultinomialSampler)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn test_factory_creates_greedy_for_zero_temp() {
        let factory = DefaultSamplerFactory::new();
        let params = SamplingParams {
            temperature: 0.0,
            ..Default::default()
        };
        let sampler = factory.create_sampler(&params);
        assert!(sampler.is_deterministic());
    }

    #[test]
    fn test_factory_creates_multinomial_for_nonzero_temp() {
        let factory = DefaultSamplerFactory::new();
        let params = SamplingParams {
            temperature: 1.0,
            ..Default::default()
        };
        let sampler = factory.create_sampler(&params);
        assert!(!sampler.is_deterministic());
    }

    #[test]
    fn test_build_sampling_config() {
        let params = SamplingParams {
            temperature: 0.8,
            top_k: Some(50),
            top_p: 0.95,
            repetition_penalty: 1.1,
            ..Default::default()
        };
        let config = build_sampling_config(&params);
        // Config should be created successfully
        // Should have: temperature, top_k, top_p, repetition_penalty processors
        assert_eq!(config.processor_chain.processor_names().len(), 4);
    }

    #[test]
    fn test_pipeline_sample_simple() {
        let params = SamplingParams::greedy();
        let pipeline = pipeline_from_params(&params);
        let mut rng = StdRng::seed_from_u64(42);

        let mut logits = vec![1.0, 5.0, 2.0, 0.5];
        let token = pipeline.sample_simple(&mut logits, &mut rng).unwrap();

        // Should select index 1 (highest logit)
        assert_eq!(token.get(), 1);
    }

    #[test]
    fn test_greedy_sampler_deterministic() {
        let sampler = greedy_sampler();
        assert!(sampler.is_deterministic());
        assert_eq!(sampler.name(), "greedy");
    }

    #[test]
    fn test_multinomial_sampler_stochastic() {
        let sampler = multinomial_sampler();
        assert!(!sampler.is_deterministic());
        assert_eq!(sampler.name(), "multinomial");
    }

    #[test]
    fn test_pipeline_with_context() {
        let params = SamplingParams {
            temperature: 1.0,
            repetition_penalty: 1.2,
            ..Default::default()
        };
        let pipeline = SamplingPipeline::new(&params);
        let mut rng = StdRng::seed_from_u64(42);

        let mut logits = vec![1.0, 2.0, 3.0, 2.0];
        let previous_tokens = vec![TokenId::new(2)]; // Token 2 was generated before
        let mut freqs = HashMap::new();
        freqs.insert(TokenId::new(2), 1);

        let token = pipeline
            .sample_next(0, &mut logits, &previous_tokens, &freqs, &params, &mut rng)
            .unwrap();

        // Token should be valid
        assert!(token.get() < 4);
    }
}
