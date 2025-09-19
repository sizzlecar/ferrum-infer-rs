//! Multinomial sampler implementation

use crate::SamplerInterface;
use ferrum_types::{Result, TokenId};
use rand::RngCore;
use rand_distr::{Distribution, WeightedIndex};

/// Multinomial sampler
/// 
/// Samples tokens according to their probability distribution.
/// This introduces randomness and variety in outputs.
#[derive(Debug, Clone)]
pub struct MultinomialSampler {
    /// Whether to use stable sampling (consistent ordering for ties)
    stable: bool,
}

impl MultinomialSampler {
    /// Create new multinomial sampler
    pub fn new() -> Self {
        Self { stable: false }
    }

    /// Create new stable multinomial sampler
    pub fn new_stable() -> Self {
        Self { stable: true }
    }

    /// Set stability mode
    pub fn set_stable(&mut self, stable: bool) {
        self.stable = stable;
    }

    /// Get stability mode
    pub fn is_stable(&self) -> bool {
        self.stable
    }

    /// Convert logits to probabilities using softmax
    fn softmax(&self, logits: &[f32]) -> Vec<f32> {
        if logits.is_empty() {
            return Vec::new();
        }

        // Find max logit for numerical stability
        let max_logit = logits.iter().fold(f32::NEG_INFINITY, |a, &b| {
            if b.is_finite() { a.max(b) } else { a }
        });

        // If all logits are -inf, return uniform distribution
        if max_logit == f32::NEG_INFINITY {
            let uniform_prob = 1.0 / logits.len() as f32;
            return vec![uniform_prob; logits.len()];
        }

        // Compute exp(logit - max_logit)
        let mut exp_logits: Vec<f32> = logits
            .iter()
            .map(|&x| if x.is_finite() { (x - max_logit).exp() } else { 0.0 })
            .collect();

        // Normalize
        let sum: f32 = exp_logits.iter().sum();
        if sum > 0.0 {
            for exp_logit in exp_logits.iter_mut() {
                *exp_logit /= sum;
            }
        } else {
            // Fallback to uniform if all weights are zero
            let uniform_prob = 1.0 / logits.len() as f32;
            exp_logits.fill(uniform_prob);
        }

        exp_logits
    }
}

impl Default for MultinomialSampler {
    fn default() -> Self {
        Self::new()
    }
}

impl SamplerInterface for MultinomialSampler {
    fn sample(&self, logits: &[f32], rng: &mut dyn RngCore) -> Result<TokenId> {
        if logits.is_empty() {
            return Err(ferrum_types::FerrumError::invalid_parameter(
                "Cannot sample from empty logits",
            ));
        }

        let probs = self.softmax(logits);
        
        // Create weighted distribution
        let dist = WeightedIndex::new(&probs)
            .map_err(|e| ferrum_types::FerrumError::sampling_error(
                format!("Failed to create weighted distribution: {}", e)
            ))?;

        // Sample from distribution
        let idx = dist.sample(rng);
        Ok(TokenId::new(idx as u32))
    }

    fn sample_multiple(&self, logits: &[f32], n: usize, rng: &mut dyn RngCore) -> Result<Vec<TokenId>> {
        if n == 0 {
            return Ok(Vec::new());
        }

        let probs = self.softmax(logits);
        
        // Create weighted distribution
        let dist = WeightedIndex::new(&probs)
            .map_err(|e| ferrum_types::FerrumError::sampling_error(
                format!("Failed to create weighted distribution: {}", e)
            ))?;

        // Sample multiple times
        let mut tokens = Vec::with_capacity(n);
        for _ in 0..n {
            let idx = dist.sample(rng);
            tokens.push(TokenId::new(idx as u32));
        }

        Ok(tokens)
    }

    fn name(&self) -> &str {
        if self.stable {
            "multinomial_stable"
        } else {
            "multinomial"
        }
    }

    fn is_deterministic(&self) -> bool {
        false
    }

    fn supports_multiple(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;
    use std::collections::HashMap;

    #[test]
    fn test_multinomial_sampling() {
        let sampler = MultinomialSampler::new();
        let mut rng = StdRng::seed_from_u64(42);
        
        // High probability for token 1
        let logits = vec![-10.0, 10.0, -10.0];
        
        // Sample multiple times to check distribution
        let mut counts = HashMap::new();
        for _ in 0..100 {
            let token = sampler.sample(&logits, &mut rng).unwrap();
            *counts.entry(token.value()).or_insert(0) += 1;
        }
        
        // Token 1 should be sampled most frequently
        let token1_count = counts.get(&1).copied().unwrap_or(0);
        assert!(token1_count > 80); // Should be very likely
    }

    #[test]
    fn test_multinomial_uniform() {
        let sampler = MultinomialSampler::new();
        let mut rng = StdRng::seed_from_u64(42);
        
        // Uniform logits
        let logits = vec![0.0, 0.0, 0.0];
        
        // Sample multiple times
        let mut counts = HashMap::new();
        for _ in 0..300 {
            let token = sampler.sample(&logits, &mut rng).unwrap();
            *counts.entry(token.value()).or_insert(0) += 1;
        }
        
        // Each token should appear roughly equally
        for i in 0..3 {
            let count = counts.get(&i).copied().unwrap_or(0);
            assert!(count > 50 && count < 150); // Roughly uniform
        }
    }

    #[test]
    fn test_multinomial_multiple_sampling() {
        let sampler = MultinomialSampler::new();
        let mut rng = StdRng::seed_from_u64(42);
        
        let logits = vec![1.0, 2.0, 1.0];
        let tokens = sampler.sample_multiple(&logits, 5, &mut rng).unwrap();
        
        assert_eq!(tokens.len(), 5);
        // All tokens should be valid indices
        assert!(tokens.iter().all(|t| t.value() < 3));
    }

    #[test]
    fn test_softmax_computation() {
        let sampler = MultinomialSampler::new();
        
        let logits = vec![0.0, 1.0, 0.0];
        let probs = sampler.softmax(&logits);
        
        // Middle token should have highest probability
        assert!(probs[1] > probs[0]);
        assert!(probs[1] > probs[2]);
        
        // Probabilities should sum to 1
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_softmax_with_negative_infinity() {
        let sampler = MultinomialSampler::new();
        
        let logits = vec![f32::NEG_INFINITY, 0.0, f32::NEG_INFINITY];
        let probs = sampler.softmax(&logits);
        
        // Only middle token should have non-zero probability
        assert_eq!(probs[0], 0.0);
        assert_eq!(probs[1], 1.0);
        assert_eq!(probs[2], 0.0);
    }

    #[test]
    fn test_multinomial_empty_logits() {
        let sampler = MultinomialSampler::new();
        let mut rng = StdRng::seed_from_u64(42);
        
        let logits = vec![];
        let result = sampler.sample(&logits, &mut rng);
        
        assert!(result.is_err());
    }
}
