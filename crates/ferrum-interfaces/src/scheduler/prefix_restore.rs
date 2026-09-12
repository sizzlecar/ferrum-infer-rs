//! Single-use scheduler preparation for an independent state restore stage.

use std::any::Any;

use ferrum_types::{FerrumError, RequestId, Result};

/// A scheduler-owned preparation, created before asynchronous device work.
///
/// The implementation proof is intentionally opaque to the engine. It must
/// bind the exact admission incarnation and logical work generation, rather
/// than authorizing a later lookup by request id alone. Its destructor releases
/// any scheduling hold without committing progress. This value is not Clone
/// or serializable and does not prove that device restoration has completed.
#[must_use = "commit after successful restore publication, or drop to abandon it"]
pub struct PreparedPrefixRestore {
    request_id: RequestId,
    expected_offset: usize,
    prompt_tokens: usize,
    proof: Box<dyn Any + Send + Sync>,
}

impl PreparedPrefixRestore {
    /// Implementer constructor. The receiving scheduler must validate the
    /// private proof type and its identity on every commit.
    pub fn new<T: Any + Send + Sync>(
        request_id: RequestId,
        expected_offset: usize,
        prompt_tokens: usize,
        proof: T,
    ) -> Self {
        Self {
            request_id,
            expected_offset,
            prompt_tokens,
            proof: Box::new(proof),
        }
    }

    pub fn request_id(&self) -> &RequestId {
        &self.request_id
    }

    pub const fn expected_offset(&self) -> usize {
        self.expected_offset
    }

    pub const fn prompt_tokens(&self) -> usize {
        self.prompt_tokens
    }

    /// Consume the implementation's private proof. A foreign scheduler or
    /// proof type cannot silently become authority for a matching request id.
    pub fn into_proof<T: Any + Send + Sync>(self) -> Result<T> {
        self.proof.downcast::<T>().map(|proof| *proof).map_err(|_| {
            FerrumError::scheduler("Prefix restore preparation belongs to another scheduler type")
        })
    }
}

impl std::fmt::Debug for PreparedPrefixRestore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PreparedPrefixRestore")
            .field("request_id", &self.request_id)
            .field("expected_offset", &self.expected_offset)
            .field("prompt_tokens", &self.prompt_tokens)
            .finish_non_exhaustive()
    }
}
