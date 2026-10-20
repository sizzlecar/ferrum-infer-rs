//! Completion evidence from the context that owns the submitted device work.

use ferrum_types::{FerrumError, Result};

/// A device error and an unknown submission lifetime are different facts.
/// Callers must retain the original context and its resources for `Unresolved`;
/// neither this enum nor a synchronization attempt transfers those owners.
#[derive(Debug)]
#[must_use]
pub enum BackendCompletion {
    Succeeded,
    /// Device execution is terminal, but execution or checked resource cleanup
    /// failed. A cleanup failure can retain native handles for a later retry;
    /// terminal execution alone does not certify complete resource retirement.
    Failed(FerrumError),
    /// No terminal observation was obtained. This includes unsupported checks.
    Unresolved(FerrumError),
}

impl BackendCompletion {
    pub fn is_terminal(&self) -> bool {
        matches!(self, Self::Succeeded | Self::Failed(_))
    }

    /// Compatibility for callers requiring success only. Lifecycle code must
    /// inspect the variant before using this conversion: `Err` alone does not
    /// say whether it is safe to release the execution owners.
    pub fn into_result(self) -> Result<()> {
        match self {
            Self::Succeeded => Ok(()),
            Self::Failed(error) | Self::Unresolved(error) => Err(error),
        }
    }
}
