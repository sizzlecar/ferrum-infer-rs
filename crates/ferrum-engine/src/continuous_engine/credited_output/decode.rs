//! Decode work authorized by a live request's retained projection budget.
use super::*;
use ferrum_interfaces::output_flow::OutputCodecDescriptor;
use ferrum_interfaces::tokenizer::{BoundedDecodeBound, BoundedIncrementalDecodePolicy};

#[derive(Debug, Clone, Copy)]
pub(in crate::continuous_engine) struct CreditedDecodePolicy {
    bound: BoundedDecodeBound,
    token_bytes_bound: usize,
    max_tokens: usize,
    codec: Option<OutputCodecDescriptor>,
    incremental: Option<BoundedIncrementalDecodePolicy>,
}

impl CreditedDecodePolicy {
    pub(in crate::continuous_engine) fn cost_incremental_policy(self) -> Option<&'static str> {
        self.incremental.map(|policy| match policy {
            BoundedIncrementalDecodePolicy::StrictDecodedPrefix => "strict_decoded_prefix.v1",
        })
    }
    pub(in crate::continuous_engine) fn from_plan(plan: &RequestOutputPlan) -> Self {
        Self {
            bound: plan.bounded_decoder(),
            token_bytes_bound: plan.max_token_bytes(),
            max_tokens: plan.effective_max_tokens(),
            codec: plan.codec_descriptor().ok(),
            incremental: plan.bounded_incremental_policy(),
        }
    }

    /// Immutable values of the actually admitted output path. A missing codec
    /// descriptor disables cost training, without changing inference behavior.
    pub(in crate::continuous_engine) fn cost_policy(
        self,
    ) -> Option<(BoundedDecodeBound, usize, usize, OutputCodecDescriptor)> {
        Some((
            self.bound,
            self.token_bytes_bound,
            self.max_tokens,
            self.codec?,
        ))
    }

    /// The caller retains the projection owner for both these temporary
    /// buffers and the returned text. No legacy decoder fallback is permitted.
    pub(in crate::continuous_engine) fn decode(
        self,
        tokenizer: &dyn Tokenizer,
        tokens: &[TokenId],
    ) -> Result<String> {
        if tokens.len() > self.max_tokens || tokenizer.bounded_decode_bound() != Some(self.bound) {
            return Err(FerrumError::internal(
                "credited decode does not match its admitted token/workspace bound",
            ));
        }
        let required = self
            .bound
            .requirements(tokens.len())
            .map_err(|error| FerrumError::internal(error.to_string()))?;
        let mut scratch = vec![0; required.scratch_bytes];
        let mut text = String::with_capacity(required.text_bytes);
        // Account the actual exposed capacities, never just the decoded len.
        if scratch.capacity() != required.scratch_bytes || text.capacity() != required.text_bytes {
            return Err(FerrumError::internal(
                "credited decoder allocation exceeded its reserved capacity",
            ));
        }
        tokenizer
            .decode_bounded_into(tokens, true, &mut scratch, &mut text)
            .map_err(|error| FerrumError::model(error.to_string()))?;
        Ok(text)
    }

    pub(in crate::continuous_engine) fn token_bytes(
        self,
        tokenizer: &dyn Tokenizer,
        token: TokenId,
    ) -> Result<Option<Vec<u8>>> {
        if tokenizer.bounded_token_bytes_bound().map(NonZeroUsize::get)
            != Some(self.token_bytes_bound)
        {
            return Err(FerrumError::internal(
                "credited token bytes do not match their admitted bound",
            ));
        }
        let mut bytes = vec![0; self.token_bytes_bound];
        if bytes.capacity() != self.token_bytes_bound {
            return Err(FerrumError::internal(
                "credited token byte allocation exceeded its reserved capacity",
            ));
        }
        match tokenizer
            .token_bytes_bounded_into(token, &mut bytes)
            .map_err(|error| FerrumError::model(error.to_string()))?
        {
            None => Ok(None),
            Some(length) if length <= bytes.len() => {
                bytes.truncate(length);
                Ok(Some(bytes))
            }
            Some(_) => Err(FerrumError::internal(
                "credited token byte surface exceeded its declared bound",
            )),
        }
    }
}

impl CreditedDecodePolicy {
    /// Preserve the tokenizer's incremental contract, including an error on a
    /// rewritten UTF-8 replacement prefix. Delayed plans reserve both text
    /// buffers and the combined token copy in addition to normal projection.
    pub(in crate::continuous_engine) fn incremental_has_payload(
        self,
        tokenizer: &dyn Tokenizer,
        previous: &[TokenId],
        token: TokenId,
    ) -> Result<bool> {
        let policy = self.incremental.ok_or_else(|| {
            FerrumError::internal("credited completion has no bounded incremental policy")
        })?;
        if tokenizer.bounded_incremental_decode_policy() != Some(policy) {
            return Err(FerrumError::internal("credited incremental policy changed"));
        }
        let count = previous
            .len()
            .checked_add(1)
            .filter(|count| *count <= self.max_tokens)
            .ok_or_else(|| FerrumError::internal("credited completion exceeded its token bound"))?;
        let mut combined = Vec::with_capacity(count);
        if combined.capacity() != count {
            return Err(FerrumError::internal(
                "credited completion token scratch exceeded its grant",
            ));
        }
        combined.extend_from_slice(previous);
        combined.push(token);
        // Each decode drops raw workspace before returning. The two calls
        // share the existing maximum scratch grant, never two live workspaces.
        let previous_text = self.decode(tokenizer, previous)?;
        let full_text = self.decode(tokenizer, &combined)?;
        let delta = policy.delta(&previous_text, &full_text).ok_or_else(|| {
            FerrumError::tokenizer("Incremental decode changed the previously emitted text prefix")
        })?;
        Ok(delta
            .chars()
            .any(|character| !character.is_whitespace() && character != '\u{FFFD}'))
    }
}

impl SequenceState {
    pub(in crate::continuous_engine) fn token_bytes_for_output(
        &self,
        tokenizer: &dyn Tokenizer,
        token: TokenId,
    ) -> Result<Option<Vec<u8>>> {
        match self.credited_output.as_ref() {
            Some(output) => output.decoder.token_bytes(tokenizer, token),
            None => Ok(tokenizer.token_bytes(token)),
        }
    }

    pub(in crate::continuous_engine) fn decode_owned_output(
        &self,
        tokenizer: &dyn Tokenizer,
        tokens: &[TokenId],
    ) -> Result<String> {
        match self.credited_output.as_ref() {
            Some(output) => output.decoder.decode(tokenizer, tokens),
            None => tokenizer.decode(tokens, true),
        }
    }
}
