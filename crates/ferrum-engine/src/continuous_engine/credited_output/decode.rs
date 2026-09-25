//! Decode work authorized by a live request's retained projection budget.
use super::*;
use ferrum_interfaces::output_flow::OutputCodecDescriptor;
use ferrum_interfaces::tokenizer::{BoundedDecodeBound, BoundedIncrementalDecodePolicy};

/// Close a real incomplete byte suffix without emitting the decoder's lossy
/// replacement. This is not a text sanitizer: literal U+FFFD in the complete
/// prefix is retained, and a decoder/byte mismatch is an error. With no tracked
/// pending bytes, a trailing U+FFFD still needs proof that it is literal. False
/// leaves that legacy/unknown output withheld; it is not permission to emit it.
pub(in crate::continuous_engine) fn finish_incomplete_utf8(
    tokenizer: &dyn Tokenizer,
    tokens: &[TokenId],
    pending: &[u8],
    decoder: Option<CreditedDecodePolicy>,
    text: &mut String,
) -> Result<bool> {
    if pending.is_empty() && !text.ends_with('\u{FFFD}') {
        return Ok(true);
    }
    let incomplete = std::str::from_utf8(pending)
        .is_err_and(|error| error.valid_up_to() == 0 && error.error_len().is_none());
    if pending.len() > 3 || (!pending.is_empty() && !incomplete) {
        return Err(FerrumError::internal(
            "invalid terminal UTF-8 pending proof",
        ));
    }
    let prefix = if pending.is_empty() {
        text.as_str()
    } else {
        text.strip_suffix('\u{FFFD}').ok_or_else(|| {
            FerrumError::tokenizer("terminal decoder does not expose its incomplete UTF-8 suffix")
        })?
    };
    let prefix_len = prefix.len();
    // Compare the original bytes to the entire decoded prefix plus the actual
    // pending suffix, not merely to the final character. At most one admitted
    // token-byte buffer is live, regardless of the generated history length.
    let proof = (|| -> Result<()> {
        let mut expected = prefix.as_bytes().iter().chain(pending).copied();
        let mut compare = |bytes: &[u8]| -> Result<()> {
            for &byte in bytes {
                if expected.next() != Some(byte) {
                    return Err(FerrumError::tokenizer(
                        "terminal token bytes do not match the decoded UTF-8 prefix",
                    ));
                }
            }
            Ok(())
        };
        if let Some(policy) = decoder {
            if tokens.len() > policy.max_tokens
                || tokenizer.bounded_decode_bound() != Some(policy.bound)
                || tokenizer.bounded_token_bytes_bound().map(NonZeroUsize::get)
                    != Some(policy.token_bytes_bound)
            {
                return Err(FerrumError::internal(
                    "terminal byte proof exceeds its admitted policy",
                ));
            }
            let mut scratch = vec![0; policy.token_bytes_bound];
            if scratch.capacity() != policy.token_bytes_bound {
                return Err(FerrumError::internal(
                    "terminal byte proof exceeded its scratch grant",
                ));
            }
            for &token in tokens {
                let length = tokenizer
                    .token_bytes_bounded_into(token, &mut scratch)
                    .map_err(|error| FerrumError::tokenizer(error.to_string()))?
                    .filter(|length| *length <= scratch.len())
                    .ok_or_else(|| FerrumError::tokenizer("terminal byte proof is unavailable"))?;
                compare(&scratch[..length])?;
            }
        } else {
            for &token in tokens {
                let bytes = tokenizer
                    .token_bytes(token)
                    .ok_or_else(|| FerrumError::tokenizer("terminal byte proof is unavailable"))?;
                compare(&bytes)?;
            }
        }
        if expected.next().is_some() {
            return Err(FerrumError::tokenizer(
                "terminal byte proof has an unmatched suffix",
            ));
        }
        Ok(())
    })();
    match proof {
        Ok(()) => {
            text.truncate(prefix_len);
            Ok(true)
        }
        Err(_) if pending.is_empty() => Ok(false),
        Err(error) => Err(error),
    }
}

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
