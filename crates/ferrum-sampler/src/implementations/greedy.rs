//! Greedy sampler implementation

use crate::SamplerInterface;
use ferrum_types::{Result, TokenId};
use rand::RngCore;

/// Greedy sampler
///
/// Always selects the token with the highest logit value.
/// This is deterministic and produces consistent outputs.
#[derive(Debug, Clone, Default)]
pub struct GreedySampler;

impl GreedySampler {
    /// Create new greedy sampler
    pub fn new() -> Self {
        Self
    }
}

impl SamplerInterface for GreedySampler {
    fn sample(&self, logits: &[f32], _rng: &mut dyn RngCore) -> Result<TokenId> {
        if logits.is_empty() {
            return Err(ferrum_types::FerrumError::config(
                "Cannot sample from empty logits",
            ));
        }

        // Find the index with the highest logit
        let mut max_idx = 0;
        let mut max_logit = logits[0];

        for (idx, &logit) in logits.iter().enumerate().skip(1) {
            if logit > max_logit || (logit == max_logit && logit.is_finite()) {
                max_logit = logit;
                max_idx = idx;
            }
        }

        // Check if all logits are negative infinity (impossible to sample)
        if max_logit == f32::NEG_INFINITY {
            return Err(ferrum_types::FerrumError::sampling_error(
                "All logits are negative infinity",
            ));
        }

        Ok(TokenId::new(max_idx as u32))
    }

    fn sample_multiple(
        &self,
        logits: &[f32],
        n: usize,
        _rng: &mut dyn RngCore,
    ) -> Result<Vec<TokenId>> {
        if n == 0 {
            return Ok(Vec::new());
        }

        // For greedy sampling, all samples are the same
        let token = self.sample(logits, _rng)?;
        Ok(vec![token; n])
    }

    fn name(&self) -> &str {
        "greedy"
    }

    fn is_deterministic(&self) -> bool {
        true
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

    #[test]
    fn test_greedy_sampling() {
        let sampler = GreedySampler::new();
        let mut rng = StdRng::seed_from_u64(42);

        let logits = vec![1.0, 3.0, 2.0, 0.5];
        let token = sampler.sample(&logits, &mut rng).unwrap();

        // Should select index 1 (highest logit = 3.0)
        assert_eq!(token.get(), 1);
    }

    #[test]
    fn test_greedy_with_ties() {
        let sampler = GreedySampler::new();
        let mut rng = StdRng::seed_from_u64(42);

        let logits = vec![2.0, 3.0, 3.0, 1.0]; // Tie between indices 1 and 2
        let token = sampler.sample(&logits, &mut rng).unwrap();

        // Should select the first occurrence (index 1)
        assert_eq!(token.get(), 1);
    }

    #[test]
    fn test_greedy_multiple_sampling() {
        let sampler = GreedySampler::new();
        let mut rng = StdRng::seed_from_u64(42);

        let logits = vec![1.0, 3.0, 2.0];
        let tokens = sampler.sample_multiple(&logits, 3, &mut rng).unwrap();

        // All samples should be the same (greedy)
        assert_eq!(tokens.len(), 3);
        assert!(tokens.iter().all(|&t| t.get() == 1));
    }

    #[test]
    fn test_greedy_empty_logits() {
        let sampler = GreedySampler::new();
        let mut rng = StdRng::seed_from_u64(42);

        let logits = vec![];
        let result = sampler.sample(&logits, &mut rng);

        assert!(result.is_err());
    }

    #[test]
    fn test_greedy_negative_infinity() {
        let sampler = GreedySampler::new();
        let mut rng = StdRng::seed_from_u64(42);

        let logits = vec![f32::NEG_INFINITY, f32::NEG_INFINITY];
        let result = sampler.sample(&logits, &mut rng);

        assert!(result.is_err());
    }

    #[test]
    fn test_greedy_with_some_negative_infinity() {
        let sampler = GreedySampler::new();
        let mut rng = StdRng::seed_from_u64(42);

        let logits = vec![f32::NEG_INFINITY, 1.0, f32::NEG_INFINITY];
        let token = sampler.sample(&logits, &mut rng).unwrap();

        // Should select index 1 (only finite logit)
        assert_eq!(token.get(), 1);
    }
}
