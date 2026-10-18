//! Model-side token work, without Tensor transport or executor identities.

use ferrum_types::{FerrumError, Result};

#[derive(Debug, Clone, Copy)]
pub enum ModelTokenWork<'a> {
    Prefill { tokens: &'a [u32], position: usize },
    Decode { token: u32, position: u32 },
}

/// Actual execution status supplied by the model implementation. In
/// particular, a returned error alone does not establish zero submission.
#[derive(Debug)]
pub enum ModelTokenOutcome {
    NotSubmitted(FerrumError),
    /// The implementation observed synchronous completion (or an actual
    /// terminal failure). Logit validation is a later product operation.
    Terminal(Result<Vec<f32>>),
}

/// Only callers with an actual synchronous-host implementation may use this.
/// A device enqueue panic cannot be made terminal by catching it here.
pub(crate) fn catch_host_execution<T>(work: impl FnOnce() -> T) -> Result<T> {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(work)).map_err(|payload| {
        let detail = payload
            .downcast_ref::<String>()
            .map(String::as_str)
            .or_else(|| payload.downcast_ref::<&str>().copied())
            .unwrap_or("non-string panic payload");
        FerrumError::backend(format!("synchronous model execution panicked: {detail}"))
    })
}
