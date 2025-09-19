//! Default sampler factory implementation

use crate::{SamplerFactoryInterface, SamplerInterface, SamplingConfig, SamplingMode};
use crate::implementations::{GreedySampler, MultinomialSampler};
use crate::processors::{ProcessorChain, TemperatureProcessor, TopKProcessor, TopPProcessor};
use crate::processors::{RepetitionPenaltyProcessor, PresencePenaltyProcessor, FrequencyPenaltyProcessor};
use ferrum_types::Result;
use std::sync::Arc;

/// Default sampler factory
#[derive(Debug, Clone, Default)]
pub struct DefaultSamplerFactory;

impl DefaultSamplerFactory {
    /// Create new factory
    pub fn new() -> Self {
        Self
    }

    /// Create processor chain from config
    pub fn create_processor_chain(&self, config: &SamplingConfig) -> Result<ProcessorChain> {
        let mut chain = ProcessorChain::new();

        // Add repetition penalty if specified
        if let Some(repetition_penalty) = config.repetition_penalty {
            if (repetition_penalty.value() - 1.0).abs() > f32::EPSILON {
                chain.add(RepetitionPenaltyProcessor::new(repetition_penalty));
            }
        }

        // Add presence penalty if specified
        if let Some(presence_penalty) = config.presence_penalty {
            if presence_penalty.value().abs() > f32::EPSILON {
                chain.add(PresencePenaltyProcessor::new(presence_penalty));
            }
        }

        // Add frequency penalty if specified  
        if let Some(frequency_penalty) = config.frequency_penalty {
            if frequency_penalty.value().abs() > f32::EPSILON {
                chain.add(FrequencyPenaltyProcessor::new(frequency_penalty));
            }
        }

        // Add temperature scaling if specified
        if let Some(temperature) = config.temperature {
            if (temperature.value() - 1.0).abs() > f32::EPSILON {
                chain.add(TemperatureProcessor::new(temperature));
            }
        }

        // Add top-k filtering if specified
        if let Some(top_k) = config.top_k {
            if top_k.value() > 0 {
                chain.add(TopKProcessor::new(top_k));
            }
        }

        // Add top-p filtering if specified
        if let Some(top_p) = config.top_p {
            if (top_p.value() - 1.0).abs() > f32::EPSILON {
                chain.add(TopPProcessor::new(top_p));
            }
        }

        Ok(chain)
    }
}

impl SamplerFactoryInterface for DefaultSamplerFactory {
    fn create_sampler(&self, config: &SamplingConfig) -> Result<Box<dyn SamplerInterface + Send + Sync>> {
        let sampler: Box<dyn SamplerInterface + Send + Sync> = match config.mode {
            SamplingMode::Greedy => Box::new(GreedySampler::new()),
            SamplingMode::Multinomial => Box::new(MultinomialSampler::new()),
            SamplingMode::MultinomialStable => Box::new(MultinomialSampler::new_stable()),
        };

        Ok(sampler)
    }

    fn create_processor_chain(&self, config: &SamplingConfig) -> Result<Box<dyn crate::LogitsProcessorInterface + Send + Sync>> {
        let chain = self.create_processor_chain(config)?;
        Ok(Box::new(chain))
    }

    fn supported_modes(&self) -> Vec<SamplingMode> {
        vec![
            SamplingMode::Greedy,
            SamplingMode::Multinomial,
            SamplingMode::MultinomialStable,
        ]
    }

    fn name(&self) -> &str {
        "default"
    }
}

/// Combined sampler that includes both processor chain and sampler
#[derive(Debug)]
pub struct CombinedSampler {
    processor_chain: ProcessorChain,
    sampler: Box<dyn SamplerInterface + Send + Sync>,
    name: String,
}

impl CombinedSampler {
    /// Create new combined sampler
    pub fn new(
        processor_chain: ProcessorChain,
        sampler: Box<dyn SamplerInterface + Send + Sync>,
    ) -> Self {
        let name = format!("combined[{}+{}]", 
            if processor_chain.is_empty() { "none" } else { "processors" },
            sampler.name()
        );

        Self {
            processor_chain,
            sampler,
            name,
        }
    }

    /// Create from config using factory
    pub fn from_config(config: &SamplingConfig) -> Result<Self> {
        let factory = DefaultSamplerFactory::new();
        let processor_chain = factory.create_processor_chain(config)?;
        let sampler = factory.create_sampler(config)?;
        
        Ok(Self::new(processor_chain, sampler))
    }

    /// Process logits and sample token
    pub fn process_and_sample(&self, logits: &mut [f32], rng: &mut dyn rand::RngCore) -> Result<ferrum_types::TokenId> {
        // Apply processor chain
        self.processor_chain.process(logits)?;
        
        // Sample token
        self.sampler.sample(logits, rng)
    }

    /// Process logits and sample multiple tokens
    pub fn process_and_sample_multiple(
        &self, 
        logits: &mut [f32], 
        n: usize, 
        rng: &mut dyn rand::RngCore
    ) -> Result<Vec<ferrum_types::TokenId>> {
        // Apply processor chain
        self.processor_chain.process(logits)?;
        
        // Sample tokens
        self.sampler.sample_multiple(logits, n, rng)
    }

    /// Get processor chain
    pub fn processor_chain(&self) -> &ProcessorChain {
        &self.processor_chain
    }

    /// Get processor chain (mutable)
    pub fn processor_chain_mut(&mut self) -> &mut ProcessorChain {
        &mut self.processor_chain
    }

    /// Get sampler
    pub fn sampler(&self) -> &dyn SamplerInterface {
        self.sampler.as_ref()
    }
}

impl SamplerInterface for CombinedSampler {
    fn sample(&self, logits: &[f32], rng: &mut dyn rand::RngCore) -> Result<ferrum_types::TokenId> {
        let mut logits_copy = logits.to_vec();
        self.process_and_sample(&mut logits_copy, rng)
    }

    fn sample_multiple(&self, logits: &[f32], n: usize, rng: &mut dyn rand::RngCore) -> Result<Vec<ferrum_types::TokenId>> {
        let mut logits_copy = logits.to_vec();
        self.process_and_sample_multiple(&mut logits_copy, n, rng)
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn is_deterministic(&self) -> bool {
        self.sampler.is_deterministic()
    }

    fn supports_multiple(&self) -> bool {
        self.sampler.supports_multiple()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferrum_types::{Temperature, TopK, TopP};
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn test_factory_greedy_sampler() {
        let factory = DefaultSamplerFactory::new();
        let config = SamplingConfig {
            mode: SamplingMode::Greedy,
            temperature: None,
            top_k: None,
            top_p: None,
            repetition_penalty: None,
            presence_penalty: None,
            frequency_penalty: None,
        };

        let sampler = factory.create_sampler(&config).unwrap();
        assert_eq!(sampler.name(), "greedy");
        assert!(sampler.is_deterministic());
    }

    #[test]
    fn test_factory_multinomial_sampler() {
        let factory = DefaultSamplerFactory::new();
        let config = SamplingConfig {
            mode: SamplingMode::Multinomial,
            temperature: None,
            top_k: None,
            top_p: None,
            repetition_penalty: None,
            presence_penalty: None,
            frequency_penalty: None,
        };

        let sampler = factory.create_sampler(&config).unwrap();
        assert_eq!(sampler.name(), "multinomial");
        assert!(!sampler.is_deterministic());
    }

    #[test]
    fn test_processor_chain_creation() {
        let factory = DefaultSamplerFactory::new();
        let config = SamplingConfig {
            mode: SamplingMode::Greedy,
            temperature: Some(Temperature::new(2.0)),
            top_k: Some(TopK::new(10)),
            top_p: Some(TopP::new(0.9)),
            repetition_penalty: None,
            presence_penalty: None,
            frequency_penalty: None,
        };

        let chain = factory.create_processor_chain(&config).unwrap();
        
        // Should have temperature, top-k, and top-p processors
        assert_eq!(chain.len(), 3);
        let names = chain.processor_names();
        assert!(names.contains(&"temperature"));
        assert!(names.contains(&"top_k"));
        assert!(names.contains(&"top_p"));
    }

    #[test]
    fn test_combined_sampler() {
        let config = SamplingConfig {
            mode: SamplingMode::Greedy,
            temperature: Some(Temperature::new(2.0)),
            top_k: None,
            top_p: None,
            repetition_penalty: None,
            presence_penalty: None,
            frequency_penalty: None,
        };

        let combined = CombinedSampler::from_config(&config).unwrap();
        let mut rng = StdRng::seed_from_u64(42);
        
        let logits = vec![2.0, 4.0, 6.0];
        let token = combined.sample(&logits, &mut rng).unwrap();
        
        // Should select highest logit (index 2) after temperature processing
        assert_eq!(token.value(), 2);
    }
}
