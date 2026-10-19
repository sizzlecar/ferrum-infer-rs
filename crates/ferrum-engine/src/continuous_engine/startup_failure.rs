//! Preserve the unpublished engine's original cleanup observation on failure.

use super::*;
use ferrum_execution::ExecutorBuildError;

impl ContinuousBatchEngine {
    pub(crate) async fn into_startup_failure(self, cause: FerrumError) -> ExecutorBuildError {
        let runtime = match tokio::runtime::Handle::try_current() {
            Ok(runtime) => runtime,
            Err(error) => {
                return ExecutorBuildError::with_cleanup(
                    cause,
                    ExecutorShutdownObserver::retained(
                        Arc::new(self),
                        FerrumError::internal(format!(
                            "startup failure retains engine cleanup without an async runtime: {error}"
                        )),
                    ),
                );
            }
        };
        let original = match self.driver.shutdown_active() {
            Ok(original) => original,
            Err(error) => {
                return ExecutorBuildError::with_cleanup(
                    cause,
                    ExecutorShutdownObserver::retained(Arc::new(self), error),
                );
            }
        };
        // Keep the existing startup-error timing: await the original driver's
        // first report. Cancelling this wait does not abort its managed task.
        let report = original.wait().await;
        let (reporter, observer) = ExecutorShutdownObserver::channel(Arc::new(original));
        let retry_report = Arc::clone(&report);
        let observer = observer.with_retry(Arc::new(move || {
            retry_report.retry_shutdown();
        }));
        if report.retains_unproved_owners() {
            reporter.retained(report.errors().first().cloned().unwrap_or_else(|| {
                FerrumError::internal("startup cleanup retains original engine owners")
            }));
            // This task only observes the existing service and forwards facts;
            // it issues no request close, executor close or device operation.
            runtime.spawn(async move {
                let result = report.wait_for_shutdown_completion().await;
                reporter.retired(result.err());
            });
        } else {
            // Executor retirement alone cannot release an engine-held scope
            // or prove its journals closed. Use the original complete boundary.
            let result = report.wait_for_shutdown_completion().await;
            reporter.retired(result.err());
        }
        ExecutorBuildError::with_cleanup(cause, observer)
    }
}
