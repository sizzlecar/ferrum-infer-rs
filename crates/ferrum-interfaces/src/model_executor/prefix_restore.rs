//! Publication of restored state before constructing a prefill batch.

use super::{KvCacheHandle, PlanRuntimePrefillAuthority};
use ferrum_types::{FerrumError, RequestId, Result, TokenId};
use std::{fmt, sync::Arc};

/// The exact input already admitted by the plan runtime. Restoration does not
/// replace admission or authorize a different request incarnation.
#[derive(Debug, Clone, Copy)]
pub struct PlanRuntimePrefixRestoreInput<'a> {
    pub request_id: &'a RequestId,
    pub input_tokens: &'a [TokenId],
    pub maximum_sequence_tokens: usize,
}

/// Independently restored state whose execution gate remains closed while the
/// engine installs its physical cache authority and advances scheduler progress.
///
/// The executor callback owns the native publication guard. Dropping this value
/// must drop that guard and cancel its exact target; the callback must never
/// retain only a request id or a later registry lookup. No logits are cached:
/// at least one remaining prompt token must execute before sampling.
#[must_use = "publish matching progress and acknowledge, or drop to cancel the restored target"]
pub struct PlanRuntimePrefixRestoreOutput {
    authority: PlanRuntimePrefillAuthority,
    prompt_tokens: usize,
    acknowledge: Box<dyn FnOnce() -> Result<()> + Send>,
}

impl PlanRuntimePrefixRestoreOutput {
    /// Implementer constructor. The callback must own and acknowledge the
    /// exact native restoration, revalidating cancellation and target identity.
    /// Rejected construction drops the callback and its retained guard.
    pub fn new(
        request_id: RequestId,
        restored_tokens: usize,
        prompt_tokens: usize,
        kv_cache: Arc<dyn KvCacheHandle>,
        acknowledge: impl FnOnce() -> Result<()> + Send + 'static,
    ) -> Result<Self> {
        let output = Self {
            authority: PlanRuntimePrefillAuthority {
                request_id,
                committed_tokens: restored_tokens,
                kv_cache,
            },
            prompt_tokens,
            acknowledge: Box::new(acknowledge),
        };
        output.validate_for(output.request_id(), prompt_tokens)?;
        Ok(output)
    }

    pub fn request_id(&self) -> &RequestId {
        self.authority.request_id()
    }

    pub fn restored_tokens(&self) -> usize {
        self.authority.committed_tokens()
    }

    pub fn kv_cache(&self) -> &Arc<dyn KvCacheHandle> {
        self.authority.kv_cache()
    }

    pub fn validate_for(&self, request_id: &RequestId, prompt_tokens: usize) -> Result<()> {
        if self.request_id() != request_id || self.prompt_tokens != prompt_tokens {
            return Err(FerrumError::backend(
                "prefix restore publication does not match the admitted request",
            ));
        }
        let restored = self.restored_tokens();
        if restored == 0 || restored >= prompt_tokens {
            return Err(FerrumError::backend(
                "prefix restore must leave a nonempty prompt suffix for execution",
            ));
        }
        if self.kv_cache().num_tokens() != restored || !self.kv_cache().is_valid() {
            return Err(FerrumError::backend(
                "prefix restore cache authority does not match the restored extent",
            ));
        }
        Ok(())
    }

    /// Opens the native execution gate only after outer progress is installed.
    /// An error requires cancellation of the corresponding outer request; it
    /// does not authorize falling back to execution on the restored target.
    pub fn acknowledge(self) -> Result<PlanRuntimePrefillAuthority> {
        let Self {
            authority,
            acknowledge,
            ..
        } = self;
        acknowledge()?;
        Ok(authority)
    }
}

impl fmt::Debug for PlanRuntimePrefixRestoreOutput {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PlanRuntimePrefixRestoreOutput")
            .field("authority", &self.authority)
            .field("prompt_tokens", &self.prompt_tokens)
            .finish_non_exhaustive()
    }
}
