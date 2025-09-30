//! Top-K processor

use crate::LogitsProcessorInterface;
use ferrum_types::{Result, TopK};

/// Top-K processor
///
/// Keeps only the top-k highest logits, sets others to negative infinity.
/// This reduces the vocabulary to the k most likely tokens.
#[derive(Debug, Clone)]
pub struct TopKProcessor {
    k: usize,
}

impl TopKProcessor {
    /// Create new top-k processor
    pub fn new(top_k: TopK) -> Self {
        Self { k: top_k.value() }
    }

    /// Create from raw value
    pub fn from_value(k: usize) -> Self {
        Self { k }
    }

    /// Get k value
    pub fn k(&self) -> usize {
        self.k
    }

    /// Set k value
    pub fn set_k(&mut self, k: usize) {
        self.k = k;
    }
}

impl LogitsProcessorInterface for TopKProcessor {
    fn process(&self, logits: &mut [f32]) -> Result<()> {
        if self.k == 0 || self.k >= logits.len() {
            // No filtering needed
            return Ok(());
        }

        // Find the k-th largest value using partial sort
        let mut indices: Vec<usize> = (0..logits.len()).collect();

        // Partial sort to get top-k indices
        indices.select_nth_unstable_by(self.k, |&a, &b| {
            logits[b]
                .partial_cmp(&logits[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Find the threshold value (k-th largest)
        let threshold = logits[indices[self.k]];

        // Set all logits below threshold to negative infinity
        for (i, logit) in logits.iter_mut().enumerate() {
            if *logit < threshold {
                *logit = f32::NEG_INFINITY;
            }
        }

        Ok(())
    }

    fn name(&self) -> &str {
        "top_k"
    }

    fn is_stateful(&self) -> bool {
        false
    }

    fn reset(&mut self) -> Result<()> {
        // Top-K processor is stateless
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_top_k_filtering() {
        let mut processor = TopKProcessor::from_value(2);
        let mut logits = vec![1.0, 4.0, 2.0, 3.0, 0.5];

        processor.process(&mut logits).unwrap();

        // Only top 2 values (4.0 and 3.0) should remain
        assert_eq!(logits[1], 4.0); // Highest
        assert_eq!(logits[3], 3.0); // Second highest
        assert_eq!(logits[0], f32::NEG_INFINITY); // Filtered
        assert_eq!(logits[2], f32::NEG_INFINITY); // Filtered
        assert_eq!(logits[4], f32::NEG_INFINITY); // Filtered
    }

    #[test]
    fn test_top_k_no_filtering() {
        let mut processor = TopKProcessor::from_value(10);
        let original = vec![1.0, 2.0, 3.0];
        let mut logits = original.clone();

        processor.process(&mut logits).unwrap();

        // No filtering should occur when k >= vocab size
        assert_eq!(logits, original);
    }

    #[test]
    fn test_top_k_zero() {
        let mut processor = TopKProcessor::from_value(0);
        let original = vec![1.0, 2.0, 3.0];
        let mut logits = original.clone();

        processor.process(&mut logits).unwrap();

        // No filtering should occur when k = 0
        assert_eq!(logits, original);
    }
}
