//! Sampling components for token generation
//!
//! This module implements the LogitsProcessor and Sampler traits from ferrum-interfaces,
//! moving sampling logic from model implementations to the engine layer.

use ferrum_interfaces::{LogitsProcessor, Sampler, SamplingContext, SamplingConfig};
use ferrum_types::{Result, FerrumError, TokenId};
use rand::Rng;
use tracing::{debug, trace};

/// Temperature-based logits processor
#[derive(Debug, Clone)]
pub struct TemperatureProcessor {
    pub temperature: f32,
}

impl TemperatureProcessor {
    pub fn new(temperature: f32) -> Self {
        Self { temperature }
    }
}

impl LogitsProcessor for TemperatureProcessor {
    fn process(&self, ctx: &mut SamplingContext) {
        if self.temperature > 0.0 && self.temperature != 1.0 {
            for logit in ctx.logits.iter_mut() {
                *logit /= self.temperature;
            }
        }
    }
}

/// Top-k logits processor
#[derive(Debug, Clone)]
pub struct TopKProcessor {
    pub k: Option<usize>,
}

impl TopKProcessor {
    pub fn new(k: Option<usize>) -> Self {
        Self { k }
    }
}

impl LogitsProcessor for TopKProcessor {
    fn process(&self, ctx: &mut SamplingContext) {
        if let Some(top_k) = self.k {
            if top_k > 0 && top_k < ctx.logits.len() {
                let mut indexed_logits: Vec<(usize, f32)> =
                    ctx.logits.iter().enumerate().map(|(i, &v)| (i, v)).collect();
                
                // Sort by logit value (descending)
                indexed_logits.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

                // Set logits outside top-k to negative infinity
                let threshold = indexed_logits[top_k].1;
                for (i, logit) in ctx.logits.iter_mut().enumerate() {
                    if !indexed_logits
                        .iter()
                        .take(top_k)
                        .any(|(idx, _)| *idx == i)
                        && *logit < threshold
                    {
                        *logit = f32::NEG_INFINITY;
                    }
                }
            }
        }
    }
}

/// Top-p (nucleus) logits processor
#[derive(Debug, Clone)]
pub struct TopPProcessor {
    pub p: f32,
}

impl TopPProcessor {
    pub fn new(p: f32) -> Self {
        Self { p }
    }
}

impl LogitsProcessor for TopPProcessor {
    fn process(&self, ctx: &mut SamplingContext) {
        if self.p < 1.0 && self.p > 0.0 {
            let mut indexed_logits: Vec<(usize, f32)> =
                ctx.logits.iter().enumerate().map(|(i, &v)| (i, v)).collect();
            
            // Sort by logit value (descending)
            indexed_logits.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            // Apply softmax to get probabilities
            let max_logit = indexed_logits[0].1;
            let exp_logits: Vec<f32> = indexed_logits
                .iter()
                .map(|(_, logit)| (logit - max_logit).exp())
                .collect();
            let sum_exp: f32 = exp_logits.iter().sum();
            
            if sum_exp > 0.0 {
                let probs: Vec<f32> = exp_logits
                    .iter()
                    .map(|exp_logit| exp_logit / sum_exp)
                    .collect();

                // Find cumulative probability cutoff
                let mut cumulative_prob = 0.0;
                let mut cutoff_idx = probs.len();

                for (i, &prob) in probs.iter().enumerate() {
                    cumulative_prob += prob;
                    if cumulative_prob >= self.p {
                        cutoff_idx = i + 1;
                        break;
                    }
                }

                // Set tokens outside top-p to negative infinity
                for (i, logit) in ctx.logits.iter_mut().enumerate() {
                    if !indexed_logits
                        .iter()
                        .take(cutoff_idx)
                        .any(|(idx, _)| *idx == i)
                    {
                        *logit = f32::NEG_INFINITY;
                    }
                }
            }
        }
    }
}

/// Repetition penalty processor
#[derive(Debug, Clone)]
pub struct RepetitionPenaltyProcessor {
    pub penalty: f32,
    pub generated_tokens: Vec<TokenId>,
}

impl RepetitionPenaltyProcessor {
    pub fn new(penalty: f32, generated_tokens: Vec<TokenId>) -> Self {
        Self { penalty, generated_tokens }
    }
}

impl LogitsProcessor for RepetitionPenaltyProcessor {
    fn process(&self, ctx: &mut SamplingContext) {
        if self.penalty != 1.0 && !self.generated_tokens.is_empty() {
            // Apply repetition penalty to previously generated tokens
            for &token_id in &self.generated_tokens {
                let idx = token_id as usize;
                if idx < ctx.logits.len() {
                    if ctx.logits[idx] > 0.0 {
                        ctx.logits[idx] /= self.penalty;
                    } else {
                        ctx.logits[idx] *= self.penalty;
                    }
                }
            }
        }
    }
}

/// Multinomial sampler
#[derive(Debug, Clone)]
pub struct MultinomialSampler {
    pub temperature: f32,
}

impl MultinomialSampler {
    pub fn new(temperature: f32) -> Self {
        Self { temperature }
    }
}

impl Sampler for MultinomialSampler {
    fn sample(&self, logits: &[f32], rng: &mut dyn rand::RngCore) -> TokenId {
        if self.temperature <= 0.0 {
            // Greedy sampling
            return logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i as TokenId)
                .unwrap_or(0);
        }

        // Convert logits to probabilities using stable softmax
        let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exp_logits: Vec<f32> = logits.iter().map(|&x| (x - max_logit).exp()).collect();
        let sum_exp: f32 = exp_logits.iter().sum();

        if sum_exp <= 0.0 {
            return 0; // Fallback to first token
        }

        let probs: Vec<f32> = exp_logits.iter().map(|&x| x / sum_exp).collect();

        // Sample from the distribution
        let mut cumulative = 0.0;
        let random_value: f32 = rng.gen_range(0.0..1.0);

        for (i, &prob) in probs.iter().enumerate() {
            cumulative += prob;
            if random_value <= cumulative {
                return i as TokenId;
            }
        }

        // Fallback
        (logits.len() - 1) as TokenId
    }
}

/// Greedy sampler (always selects highest probability token)
#[derive(Debug, Clone)]
pub struct GreedySampler;

impl Sampler for GreedySampler {
    fn sample(&self, logits: &[f32], _rng: &mut dyn rand::RngCore) -> TokenId {
        logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i as TokenId)
            .unwrap_or(0)
    }
}

/// Unified sampling pipeline that applies processors and then samples
#[derive(Debug)]
pub struct SamplingPipeline {
    pub processors: Vec<Box<dyn LogitsProcessor>>,
    pub sampler: Box<dyn Sampler>,
    pub rng: Box<dyn rand::RngCore + Send>,
}

impl SamplingPipeline {
    pub fn new(
        processors: Vec<Box<dyn LogitsProcessor>>,
        sampler: Box<dyn Sampler>,
        rng: Box<dyn rand::RngCore + Send>,
    ) -> Self {
        Self {
            processors,
            sampler,
            rng,
        }
    }

    /// Create a default pipeline from sampling parameters
    pub fn from_params(
        temperature: f32,
        top_k: Option<usize>,
        top_p: f32,
        repetition_penalty: f32,
        generated_tokens: Vec<TokenId>,
    ) -> Result<Self> {
        let mut processors: Vec<Box<dyn LogitsProcessor>> = Vec::new();
        
        // Add temperature processor
        if temperature > 0.0 && temperature != 1.0 {
            processors.push(Box::new(TemperatureProcessor::new(temperature)));
        }
        
        // Add top-k processor
        if top_k.is_some() {
            processors.push(Box::new(TopKProcessor::new(top_k)));
        }
        
        // Add top-p processor
        if top_p < 1.0 && top_p > 0.0 {
            processors.push(Box::new(TopPProcessor::new(top_p)));
        }
        
        // Add repetition penalty processor
        if repetition_penalty != 1.0 {
            processors.push(Box::new(RepetitionPenaltyProcessor::new(
                repetition_penalty,
                generated_tokens,
            )));
        }

        // Choose sampler based on temperature
        let sampler: Box<dyn Sampler> = if temperature <= 0.0 {
            Box::new(GreedySampler)
        } else {
            Box::new(MultinomialSampler::new(temperature))
        };

        let rng = Box::new(rand::rngs::ThreadRng::default());

        Ok(Self::new(processors, sampler, rng))
    }

    /// Apply all processors and sample a token
    pub fn process_and_sample(&mut self, logits: &[f32]) -> Result<TokenId> {
        let mut logits_copy = logits.to_vec();
        
        let mut ctx = SamplingContext {
            step: 0, // This would be set by the caller
            temperature: 1.0, // Processors handle their own temperature
            top_p: 1.0,
            top_k: None,
            repetition_penalty: 1.0,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            logits: &mut logits_copy,
        };

        // Apply all processors
        for processor in &self.processors {
            processor.process(&mut ctx);
        }

        // Sample from processed logits
        let token = self.sampler.sample(&logits_copy, &mut *self.rng);
        
        debug!("Sampled token {} from {} logits", token, logits.len());
        Ok(token)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_temperature_processor() {
        let processor = TemperatureProcessor::new(2.0);
        let mut logits = vec![1.0, 2.0, 3.0, 4.0];
        let mut ctx = SamplingContext {
            step: 0,
            temperature: 2.0,
            top_p: 1.0,
            top_k: None,
            repetition_penalty: 1.0,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            logits: &mut logits,
        };
        
        processor.process(&mut ctx);
        
        assert_eq!(ctx.logits, vec![0.5, 1.0, 1.5, 2.0]);
    }

    #[test]
    fn test_greedy_sampler() {
        let sampler = GreedySampler;
        let logits = vec![1.0, 3.0, 2.0, 0.5];
        let mut rng = rand::rngs::ThreadRng::default();
        
        let token = sampler.sample(&logits, &mut rng);
        assert_eq!(token, 1); // Index of maximum value (3.0)
    }

    #[test] 
    fn test_sampling_pipeline() {
        let mut pipeline = SamplingPipeline::from_params(
            1.0, // temperature
            Some(2), // top_k
            0.9, // top_p
            1.0, // repetition_penalty
            vec![], // generated_tokens
        ).unwrap();
        
        let logits = vec![1.0, 2.0, 3.0, 0.5];
        let token = pipeline.process_and_sample(&logits).unwrap();
        
        // Token should be one of the top-2 (indices 1 or 2)
        assert!(token == 1 || token == 2);
    }
}
