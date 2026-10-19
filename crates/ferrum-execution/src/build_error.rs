//! Construction failure is a value plus an observation of any original
//! resources still being retired. It must not be flattened into a wire error.

use crate::ExecutorShutdownObserver;
use ferrum_types::FerrumError;

pub type ExecutorBuildResult<T> = std::result::Result<T, ExecutorBuildError>;

#[derive(Clone)]
#[must_use = "observe retained initialization cleanup before discarding a construction failure"]
pub struct ExecutorBuildError {
    cause: FerrumError,
    cleanup: Option<ExecutorShutdownObserver>,
}

impl ExecutorBuildError {
    pub fn cause(&self) -> &FerrumError {
        &self.cause
    }

    pub fn cleanup(&self) -> Option<&ExecutorShutdownObserver> {
        self.cleanup.as_ref()
    }

    /// For an actual constructor's original cleanup service. The observer
    /// cannot release resources or manufacture a successful close receipt.
    pub fn with_cleanup(cause: FerrumError, cleanup: ExecutorShutdownObserver) -> Self {
        Self {
            cause,
            cleanup: Some(cleanup),
        }
    }
}

impl From<FerrumError> for ExecutorBuildError {
    fn from(cause: FerrumError) -> Self {
        Self {
            cause,
            cleanup: None,
        }
    }
}

impl std::fmt::Display for ExecutorBuildError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.cause, f)
    }
}

impl std::fmt::Debug for ExecutorBuildError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ExecutorBuildError")
            .field("cause", &self.cause)
            .field(
                "cleanup",
                &self.cleanup.as_ref().map(|observer| observer.snapshot()),
            )
            .finish()
    }
}

impl std::error::Error for ExecutorBuildError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.cause)
    }
}
