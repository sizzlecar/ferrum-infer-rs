//! Penalty processors for repetition control

use crate::LogitsProcessorInterface;
use ferrum_types::{FrequencyPenalty, PresencePenalty, RepetitionPenalty, Result, TokenId};
use std::collections::HashMap;

/// Repetition penalty processor
///
/// Applies penalty to tokens that have already appeared in the sequence.
/// Penalty > 1.0 reduces likelihood of repetition.
/// Penalty < 1.0 increases likelihood of repetition.
#[derive(Debug, Clone)]
pub struct RepetitionPenaltyProcessor {
    penalty: f32,
    previous_tokens: Vec<TokenId>,
}

impl RepetitionPenaltyProcessor {
    /// Create new repetition penalty processor
    pub fn new(penalty: RepetitionPenalty) -> Self {
        Self {
            penalty: penalty.value(),
            previous_tokens: Vec::new(),
        }
    }

    /// Create from raw value
    pub fn from_value(penalty: f32) -> Result<Self> {
        if penalty <= 0.0 {
            return Err(ferrum_types::FerrumError::config(
                "Repetition penalty must be positive",
            ));
        }
        Ok(Self {
            penalty,
            previous_tokens: Vec::new(),
        })
    }

    /// Add token to history
    pub fn add_token(&mut self, token: TokenId) {
        self.previous_tokens.push(token);
    }

    /// Set token history
    pub fn set_tokens(&mut self, tokens: Vec<TokenId>) {
        self.previous_tokens = tokens;
    }

    /// Get penalty value
    pub fn penalty(&self) -> f32 {
        self.penalty
    }

    /// Set penalty value
    pub fn set_penalty(&mut self, penalty: f32) -> Result<()> {
        if penalty <= 0.0 {
            return Err(ferrum_types::FerrumError::config(
                "Repetition penalty must be positive",
            ));
        }
        self.penalty = penalty;
        Ok(())
    }
}

impl LogitsProcessorInterface for RepetitionPenaltyProcessor {
    fn process(&self, logits: &mut [f32]) -> Result<()> {
        if (self.penalty - 1.0).abs() < f32::EPSILON {
            // No penalty to apply
            return Ok(());
        }

        // Apply penalty to tokens that have appeared before
        for &token in &self.previous_tokens {
            let token_id = token.get() as usize;
            if token_id < logits.len() {
                let logit = &mut logits[token_id];
                if *logit > 0.0 {
                    *logit /= self.penalty;
                } else {
                    *logit *= self.penalty;
                }
            }
        }

        Ok(())
    }

    fn name(&self) -> &str {
        "repetition_penalty"
    }

    fn is_stateful(&self) -> bool {
        true
    }

    fn reset(&mut self) -> Result<()> {
        self.previous_tokens.clear();
        Ok(())
    }
}

/// Presence penalty processor
///
/// Applies a fixed penalty to tokens that have appeared in the sequence.
/// Unlike repetition penalty, this doesn't depend on the logit value.
#[derive(Debug, Clone)]
pub struct PresencePenaltyProcessor {
    penalty: f32,
    present_tokens: std::collections::HashSet<TokenId>,
}

impl PresencePenaltyProcessor {
    /// Create new presence penalty processor
    pub fn new(penalty: PresencePenalty) -> Self {
        Self {
            penalty: penalty.value(),
            present_tokens: std::collections::HashSet::new(),
        }
    }

    /// Create from raw value
    pub fn from_value(penalty: f32) -> Self {
        Self {
            penalty,
            present_tokens: std::collections::HashSet::new(),
        }
    }

    /// Add token to presence set
    pub fn add_token(&mut self, token: TokenId) {
        self.present_tokens.insert(token);
    }

    /// Set present tokens
    pub fn set_tokens(&mut self, tokens: &[TokenId]) {
        self.present_tokens = tokens.iter().cloned().collect();
    }

    /// Get penalty value
    pub fn penalty(&self) -> f32 {
        self.penalty
    }

    /// Set penalty value
    pub fn set_penalty(&mut self, penalty: f32) {
        self.penalty = penalty;
    }
}

impl LogitsProcessorInterface for PresencePenaltyProcessor {
    fn process(&self, logits: &mut [f32]) -> Result<()> {
        if self.penalty.abs() < f32::EPSILON {
            // No penalty to apply
            return Ok(());
        }

        // Apply fixed penalty to present tokens
        for &token in &self.present_tokens {
            let token_id = token.get() as usize;
            if token_id < logits.len() {
                logits[token_id] -= self.penalty;
            }
        }

        Ok(())
    }

    fn name(&self) -> &str {
        "presence_penalty"
    }

    fn is_stateful(&self) -> bool {
        true
    }

    fn reset(&mut self) -> Result<()> {
        self.present_tokens.clear();
        Ok(())
    }
}

/// Frequency penalty processor
///
/// Applies penalty based on how frequently tokens have appeared.
/// Penalty is proportional to token frequency.
#[derive(Debug, Clone)]
pub struct FrequencyPenaltyProcessor {
    penalty: f32,
    token_counts: HashMap<TokenId, usize>,
}

impl FrequencyPenaltyProcessor {
    /// Create new frequency penalty processor
    pub fn new(penalty: FrequencyPenalty) -> Self {
        Self {
            penalty: penalty.value(),
            token_counts: HashMap::new(),
        }
    }

    /// Create from raw value
    pub fn from_value(penalty: f32) -> Self {
        Self {
            penalty,
            token_counts: HashMap::new(),
        }
    }

    /// Add token occurrence
    pub fn add_token(&mut self, token: TokenId) {
        *self.token_counts.entry(token).or_insert(0) += 1;
    }

    /// Set token counts
    pub fn set_token_counts(&mut self, counts: HashMap<TokenId, usize>) {
        self.token_counts = counts;
    }

    /// Get penalty value
    pub fn penalty(&self) -> f32 {
        self.penalty
    }

    /// Set penalty value
    pub fn set_penalty(&mut self, penalty: f32) {
        self.penalty = penalty;
    }
}

impl LogitsProcessorInterface for FrequencyPenaltyProcessor {
    fn process(&self, logits: &mut [f32]) -> Result<()> {
        if self.penalty.abs() < f32::EPSILON {
            // No penalty to apply
            return Ok(());
        }

        // Apply penalty based on frequency
        for (&token, &count) in &self.token_counts {
            let token_id = token.get() as usize;
            if token_id < logits.len() {
                let frequency_penalty = self.penalty * count as f32;
                logits[token_id] -= frequency_penalty;
            }
        }

        Ok(())
    }

    fn name(&self) -> &str {
        "frequency_penalty"
    }

    fn is_stateful(&self) -> bool {
        true
    }

    fn reset(&mut self) -> Result<()> {
        self.token_counts.clear();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_repetition_penalty() {
        let mut processor = RepetitionPenaltyProcessor::from_value(1.2).unwrap();
        processor.add_token(TokenId::new(1));

        let mut logits = vec![1.0, 2.0, 1.0]; // Token 1 has positive logit
        processor.process(&mut logits).unwrap();

        // Token 1 should be penalized (divided by 1.2)
        assert_eq!(logits[1], 2.0 / 1.2);
        assert_eq!(logits[0], 1.0); // Unchanged
        assert_eq!(logits[2], 1.0); // Unchanged
    }

    #[test]
    fn test_presence_penalty() {
        let mut processor = PresencePenaltyProcessor::from_value(0.5);
        processor.add_token(TokenId::new(1));

        let mut logits = vec![1.0, 2.0, 1.0];
        processor.process(&mut logits).unwrap();

        // Token 1 should have penalty subtracted
        assert_eq!(logits[1], 2.0 - 0.5);
        assert_eq!(logits[0], 1.0); // Unchanged
        assert_eq!(logits[2], 1.0); // Unchanged
    }

    #[test]
    fn test_frequency_penalty() {
        let mut processor = FrequencyPenaltyProcessor::from_value(0.1);
        processor.add_token(TokenId::new(1));
        processor.add_token(TokenId::new(1)); // Token 1 appears twice

        let mut logits = vec![1.0, 2.0, 1.0];
        processor.process(&mut logits).unwrap();

        // Token 1 should have penalty based on frequency (0.1 * 2 = 0.2)
        assert_eq!(logits[1], 2.0 - 0.2);
        assert_eq!(logits[0], 1.0); // Unchanged
        assert_eq!(logits[2], 1.0); // Unchanged
    }
}
