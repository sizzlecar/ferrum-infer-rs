//! Model-side token work, without Tensor transport or executor identities.

use ferrum_types::{FerrumError, Result};

/// An observation made by the model's actual KV allocator. A deferred fit
/// check has not reserved the requested blocks and grants no execution right.
#[derive(Debug)]
pub enum ModelKvReservation {
    Reserved(Option<ferrum_execution::model_executor::KvSlotReservation>),
    CapacityDeferred(ferrum_execution::token_execution::TokenAdmissionCapacityDeferral),
}

#[derive(Debug, Clone, Copy)]
pub enum ModelTokenWork<'a> {
    Prefill { tokens: &'a [u32], position: usize },
    Decode { token: u32, position: u32 },
}

/// Actual execution status supplied by the model implementation. In
/// particular, a returned error alone does not establish zero submission.
#[derive(Debug)]
pub enum ModelTokenOutcome<T = Vec<f32>> {
    NotSubmitted(FerrumError),
    /// Actual allocator fit rejection before token execution. The original
    /// model KV frontier remains usable; this observation is not a reservation.
    CapacityDeferred(ferrum_execution::token_execution::TokenAdmissionCapacityDeferral),
    /// The implementation observed synchronous completion (or an actual
    /// terminal failure). Logit validation is a later product operation.
    Terminal(Result<T>),
    /// The original model still owns an unobserved physical frame. A caller
    /// must not retry this work or release its cache on the strength of Err.
    Held(FerrumError),
}

/// Borrowed numerical inputs. Executor identities and queue claims stay above
/// this boundary; the model retains its own physical KV and context owners.
pub struct ModelTokenInput<'a> {
    pub cache_id: &'a str,
    pub work: ModelTokenWork<'a>,
    pub logits_policy: &'a ferrum_execution::model_executor::LogitsReturnPolicy,
    pub sample: bool,
}

pub type ModelTokenProduct = Option<ferrum_execution::model_executor::ExecutorSamplingOutput>;

/// A physical cohort has one terminal observation. Serial fallbacks retain
/// each actual result, including successful peers before a later failure.
pub enum ModelTokenBatchOutcome {
    Cohort(ModelTokenOutcome<Vec<ModelTokenProduct>>),
    Serial(Vec<ModelTokenOutcome<ModelTokenProduct>>),
}

/// Closing an incarnation is independent of whether its operation succeeded.
/// Held retains the actual model-owned context and cache for later recovery.
pub enum ModelTokenCloseOutcome {
    Ended { error: Option<FerrumError> },
    Held { error: FerrumError },
}

/// Observation of the original shared context, not a sequence release.
pub enum ModelExecutionRecovery {
    Quiescent { error: Option<FerrumError> },
    Held { error: FerrumError },
}

/// Only callers with an actual synchronous-host implementation may use this.
/// A device enqueue panic cannot be made terminal by catching it here.
pub(crate) fn catch_host_execution<T>(work: impl FnOnce() -> T) -> Result<T> {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(work)).map_err(|payload| {
        if let Some(error) = payload.downcast_ref::<FerrumError>() {
            return error.clone();
        }
        let detail = payload
            .downcast_ref::<String>()
            .map(String::as_str)
            .or_else(|| payload.downcast_ref::<&str>().copied())
            .unwrap_or("non-string panic payload");
        FerrumError::backend(format!("synchronous model execution panicked: {detail}"))
    })
}
