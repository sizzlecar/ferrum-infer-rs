//! Allocation-free admission description of exact completion marker tokens.
use super::*;
use ferrum_types::TokenId;
use sha2::{Digest, Sha256};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Source {
    Atomic,
    Prepared,
}

#[derive(Debug, Clone, Copy)]
struct MarkerPlan {
    text_identity: [u8; 32],
    token_identity: [u8; 32],
    count: usize,
    source: Source,
}

/// A borrowed cold token sequence or one stack-owned atomic token. Querying
/// this view allocates nothing; copy only after the request owns its grant.
pub enum CompletionTokenIds<'a> {
    Atomic([TokenId; 1]),
    Prepared(&'a [TokenId]),
}
impl CompletionTokenIds<'_> {
    pub fn as_slice(&self) -> &[TokenId] {
        match self {
            Self::Atomic(token) => token,
            Self::Prepared(tokens) => tokens,
        }
    }
    fn source(&self) -> Source {
        match self {
            Self::Atomic(_) => Source::Atomic,
            Self::Prepared(_) => Source::Prepared,
        }
    }
}

fn lookup<'a>(
    tokenizer: &'a dyn Tokenizer,
    text: &str,
) -> Result<CompletionTokenIds<'a>, OutputFlowError> {
    if let Some(token) = tokenizer.token_id(text) {
        return Ok(CompletionTokenIds::Atomic([token]));
    }
    let tokens = tokenizer
        .prepared_completion_tokens(text)
        .filter(|tokens| !tokens.is_empty())
        .ok_or(OutputFlowError::Unsupported(
            "completion marker has no immutable prepared token sequence",
        ))?;
    Ok(CompletionTokenIds::Prepared(tokens))
}
fn text_identity(text: &str) -> [u8; 32] {
    Sha256::digest(text.as_bytes()).into()
}
fn token_identity(tokens: &[TokenId]) -> [u8; 32] {
    let mut digest = Sha256::new();
    digest.update(b"ferrum.completion-marker.tokens.v1\0");
    for token in tokens {
        digest.update(token.get().to_le_bytes());
    }
    digest.finalize().into()
}
impl MarkerPlan {
    fn derive(tokenizer: &dyn Tokenizer, text: &str) -> Result<Self, OutputFlowError> {
        if text.is_empty() {
            return Err(OutputFlowError::Unsupported("empty completion marker"));
        }
        let tokens = lookup(tokenizer, text)?;
        Ok(Self {
            text_identity: text_identity(text),
            token_identity: token_identity(tokens.as_slice()),
            count: tokens.as_slice().len(),
            source: tokens.source(),
        })
    }
    fn resolve<'a>(
        &self,
        tokenizer: &'a dyn Tokenizer,
        text: &str,
    ) -> Result<CompletionTokenIds<'a>, OutputFlowError> {
        let tokens = lookup(tokenizer, text)?;
        if tokens.source() != self.source
            || tokens.as_slice().len() != self.count
            || token_identity(tokens.as_slice()) != self.token_identity
        {
            return Err(OutputFlowError::BoundExceeded);
        }
        Ok(tokens)
    }
}

/// Fixed-size descriptor: no token/history/string copy occurs before output
/// admission. KMP token and failure arrays are charged from exact lengths,
/// and their immutable source is revalidated after acquiring that storage.
#[derive(Debug, Clone, Copy)]
pub struct ResponseCompletionPlan {
    markers: [Option<MarkerPlan>; 3],
    storage_bytes: usize,
}
impl ResponseCompletionPlan {
    pub fn derive(
        tokenizer: &dyn Tokenizer,
        boundary: &ResponseCompletionBoundary,
    ) -> Result<Self, OutputFlowError> {
        let mut plan = Self {
            markers: [None; 3],
            storage_bytes: 0,
        };
        let ResponseCompletionBoundary::AfterDelimiterAndPayload {
            delimiter,
            alternate_envelope,
        } = boundary
        else {
            return Ok(plan);
        };
        plan.add(tokenizer, 0, delimiter)?;
        if let Some(envelope) = alternate_envelope {
            if envelope.max_envelopes == 0 {
                return Err(OutputFlowError::Unsupported(
                    "zero completion envelope limit",
                ));
            }
            plan.add(tokenizer, 1, &envelope.open_token_text)?;
            plan.add(tokenizer, 2, &envelope.close_token_text)?;
        }
        Ok(plan)
    }
    fn add(
        &mut self,
        tokenizer: &dyn Tokenizer,
        index: usize,
        text: &str,
    ) -> Result<(), OutputFlowError> {
        let marker = MarkerPlan::derive(tokenizer, text)?;
        self.storage_bytes = self
            .storage_bytes
            .checked_add(marker_storage(marker.count, text.len())?)
            .ok_or(OutputFlowError::Overflow)?;
        self.markers[index] = Some(marker);
        Ok(())
    }
    pub fn is_delayed(&self) -> bool {
        self.markers[0].is_some()
    }
    pub fn retained_storage_bytes(&self) -> usize {
        self.storage_bytes
    }
    pub fn resolve_marker<'a>(
        &self,
        tokenizer: &'a dyn Tokenizer,
        text: &str,
    ) -> Result<CompletionTokenIds<'a>, OutputFlowError> {
        let identity = text_identity(text);
        self.markers
            .iter()
            .flatten()
            .find(|marker| marker.text_identity == identity)
            .ok_or(OutputFlowError::BoundExceeded)?
            .resolve(tokenizer, text)
    }
}
fn marker_storage(tokens: usize, text: usize) -> Result<usize, OutputFlowError> {
    tokens
        .checked_mul(std::mem::size_of::<u32>() + std::mem::size_of::<usize>())
        // Request/scheduler, sequence sampling policy and retained projection
        // snapshots may hold marker spellings. Tokenizer-owned cold table is
        // a separate persistent allocation domain, never charged per request.
        .and_then(|bytes| text.checked_mul(3).and_then(|text| bytes.checked_add(text)))
        .ok_or(OutputFlowError::Overflow)
}

#[cfg(test)]
mod tests;
