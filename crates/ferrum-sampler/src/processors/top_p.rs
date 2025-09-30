//! Top-P (Nucleus Sampling) processor

use crate::LogitsProcessorInterface;
use ferrum_types::{Result, TopP};

/// Top-P (Nucleus Sampling) processor
///
/// Keeps only tokens whose cumulative probability mass is within top_p.
/// This dynamically adjusts the vocabulary size based on probability distribution.
#[derive(Debug, Clone)]
pub struct TopPProcessor {
    p: f32,
}

impl TopPProcessor {
    /// Create new top-p processor
    pub fn new(top_p: TopP) -> Self {
        Self { p: top_p.value() }
    }

    /// Create from raw value
    pub fn from_value(p: f32) -> Result<Self> {
        if p <= 0.0 || p > 1.0 {
            return Err(ferrum_types::FerrumError::config("Top-p must be in (0, 1]"));
        }
        Ok(Self { p })
    }

    /// Get p value
    pub fn p(&self) -> f32 {
        self.p
    }

    /// Set p value
    pub fn set_p(&mut self, p: f32) -> Result<()> {
        if p <= 0.0 || p > 1.0 {
            return Err(ferrum_types::FerrumError::config("Top-p must be in (0, 1]"));
        }
        self.p = p;
        Ok(())
    }

    /// Compute softmax probabilities
    fn softmax(logits: &[f32]) -> Vec<f32> {
        let max_logit = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

        let mut exp_logits: Vec<f32> = logits.iter().map(|&x| (x - max_logit).exp()).collect();

        let sum: f32 = exp_logits.iter().sum();

        if sum > 0.0 {
            for exp_logit in exp_logits.iter_mut() {
                *exp_logit /= sum;
            }
        }

        exp_logits
    }
}

impl LogitsProcessorInterface for TopPProcessor {
    fn process(&self, logits: &mut [f32]) -> Result<()> {
        if self.p >= 1.0 {
            // No filtering needed
            return Ok(());
        }

        // Compute probabilities
        let probs = Self::softmax(logits);

        // Create indices sorted by probability (descending)
        let mut indices: Vec<usize> = (0..probs.len()).collect();
        indices.sort_by(|&a, &b| {
            probs[b]
                .partial_cmp(&probs[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Find cutoff point where cumulative probability exceeds p
        let mut cumulative_prob = 0.0;
        let mut cutoff_idx = probs.len();

        for (i, &idx) in indices.iter().enumerate() {
            cumulative_prob += probs[idx];
            if cumulative_prob >= self.p {
                cutoff_idx = i + 1;
                break;
            }
        }

        // Set logits to negative infinity for tokens beyond cutoff
        for (i, logit) in logits.iter_mut().enumerate() {
            let is_in_nucleus = indices[..cutoff_idx].contains(&i);
            if !is_in_nucleus {
                *logit = f32::NEG_INFINITY;
            }
        }

        Ok(())
    }

    fn name(&self) -> &str {
        "top_p"
    }

    fn is_stateful(&self) -> bool {
        false
    }

    fn reset(&mut self) -> Result<()> {
        // Top-P processor is stateless
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_top_p_filtering() {
        let mut processor = TopPProcessor::from_value(0.8).unwrap();
        // Create logits that will have clear probability ranking
        let mut logits = vec![0.0, 2.0, 1.0, -1.0]; // After softmax: ~[0.12, 0.67, 0.25, 0.04]

        processor.process(&mut logits).unwrap();

        // The highest probability tokens should be kept
        assert_ne!(logits[1], f32::NEG_INFINITY); // Highest prob
        assert_ne!(logits[2], f32::NEG_INFINITY); // Second highest prob
                                                  // Lowest probability token should be filtered
        assert_eq!(logits[3], f32::NEG_INFINITY);
    }

    #[test]
    fn test_top_p_no_filtering() {
        let mut processor = TopPProcessor::from_value(1.0).unwrap();
        let original = vec![1.0, 2.0, 3.0];
        let mut logits = original.clone();

        processor.process(&mut logits).unwrap();

        // No filtering should occur when p = 1.0
        assert_eq!(logits, original);
    }

    #[test]
    fn test_invalid_top_p() {
        assert!(TopPProcessor::from_value(0.0).is_err());
        assert!(TopPProcessor::from_value(-0.1).is_err());
        assert!(TopPProcessor::from_value(1.1).is_err());
    }
}
