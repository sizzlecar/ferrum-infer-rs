//! Mock sampler: greedy argmax over logits.

use ferrum_interfaces::Sampler;
use ferrum_types::{Result, TokenId};
use rand::RngCore;

/// Greedy sampler — always picks the token with highest logit.
/// Deterministic, no temperature or top-k.
pub struct MockSampler;

impl Sampler for MockSampler {
    fn sample(&self, logits: &[f32], _rng: &mut dyn RngCore) -> Result<TokenId> {
        let (idx, _) = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .ok_or_else(|| ferrum_types::FerrumError::internal("Empty logits"))?;
        Ok(TokenId::new(idx as u32))
    }

    fn name(&self) -> &str {
        "mock-greedy"
    }

    fn is_deterministic(&self) -> bool {
        true
    }
}
