//! Exact matcher construction after the request owns its projection grant.
use super::*;
use ferrum_interfaces::output_flow::ResponseCompletionPlan;

impl ResponseCompletionState {
    pub(in crate::continuous_engine) fn compile_prepared(
        plan: &ResponseCompletionPlan,
        boundary: &ResponseCompletionBoundary,
        tokenizer: Option<&(dyn Tokenizer + Send + Sync)>,
        model_eos_token_ids: &[u32],
        max_tokens: usize,
    ) -> Result<Self> {
        Self::compile_with_tokens(boundary, model_eos_token_ids, max_tokens, |text, label| {
            let tokenizer = tokenizer.ok_or_else(|| {
                FerrumError::config("response completion boundary requires a tokenizer")
            })?;
            let tokens = plan
                .resolve_marker(tokenizer, text)
                .map_err(|error| FerrumError::invalid_request(error.to_string()))?;
            let tokens = tokens.as_slice();
            if let Some(token) = tokens
                .iter()
                .find(|token| model_eos_token_ids.contains(&token.get()))
            {
                return Err(FerrumError::invalid_request(format!(
                    "{label} token {} conflicts with model EOS",
                    token.get()
                )));
            }
            // The plan accounts this Vec and the failure Vec built by the
            // shared compiler, without a transient encoding/token-ID copy.
            let mut ids = Vec::with_capacity(tokens.len());
            if ids.capacity() != tokens.len() {
                return Err(FerrumError::internal(
                    "completion matcher exceeded its admitted capacity",
                ));
            }
            ids.extend(tokens.iter().map(|token| token.get()));
            Ok(ids)
        })
    }
}
