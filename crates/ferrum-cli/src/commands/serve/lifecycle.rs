//! A command may report a timeout while its original lifecycle still owns cleanup.
use ferrum_execution::ExecutorBuildResult as Result;
use ferrum_types::FerrumError;
use std::future::Future;
use tokio::sync::oneshot;

/// A deadline limits observation, never the lifetime of the cleanup future.
/// All cleanup stages share one deadline, including stages after HTTP drain.
pub(super) struct ShutdownObservation {
    timeout: std::time::Duration,
    deadline: Option<tokio::time::Instant>,
    report: Option<oneshot::Sender<Result<()>>>,
    timed_out: Option<FerrumError>,
    first_failure: Option<ferrum_execution::ExecutorBuildError>,
}

impl ShutdownObservation {
    pub(super) fn new(timeout: std::time::Duration, report: oneshot::Sender<Result<()>>) -> Self {
        Self {
            timeout,
            deadline: None,
            report: Some(report),
            timed_out: None,
            first_failure: None,
        }
    }

    /// A known construction failure precedes cleanup's observation timeout.
    /// Retain its original typed recovery observation in every public report.
    pub(super) fn record_failure(&mut self, error: &ferrum_execution::ExecutorBuildError) {
        if self.first_failure.is_none() {
            self.first_failure = Some(error.clone());
        }
    }

    pub(super) async fn wait<F: Future>(&mut self, cleanup: F) -> F::Output {
        let deadline = *self
            .deadline
            .get_or_insert_with(|| tokio::time::Instant::now() + self.timeout);
        tokio::pin!(cleanup);
        if self.timed_out.is_none() {
            match tokio::time::timeout_at(deadline, &mut cleanup).await {
                Ok(result) => return result,
                Err(_) => {
                    let error = FerrumError::internal(format!(
                        "serve shutdown did not finish within {} ms; original cleanup remains pending",
                        self.timeout.as_millis()
                    ));
                    tracing::warn!(%error, "retaining original serve cleanup custody");
                    if let Some(report) = self.report.take() {
                        let reported = self
                            .first_failure
                            .clone()
                            .unwrap_or_else(|| error.clone().into());
                        let _ = report.send(Err(reported));
                    }
                    self.timed_out = Some(error);
                }
            }
        }
        // Re-poll this exact future, not a fresh stop attempt or cleanup task.
        cleanup.await
    }

    pub(super) fn finish(self, actual: Result<()>) -> Result<()> {
        if let Some(first) = self.first_failure {
            return Err(first);
        }
        match self.timed_out {
            Some(timeout) => {
                if let Err(error) = actual {
                    tracing::warn!(%error, "serve cleanup later failed after reported timeout");
                }
                Err(timeout.into())
            }
            None => actual,
        }
    }
}

pub(super) struct ServeTaskObserver {
    task: tokio::task::JoinHandle<Result<()>>,
    reported: Option<oneshot::Receiver<Result<()>>>,
    _cancel: oneshot::Sender<()>,
    terminal: Option<Result<()>>,
}
impl ServeTaskObserver {
    pub(super) fn spawn<F, Fut>(run: F) -> Self
    where
        F: FnOnce(oneshot::Receiver<()>, oneshot::Sender<Result<()>>) -> Fut,
        Fut: Future<Output = Result<()>> + Send + 'static,
    {
        let (cancel, cancellation) = oneshot::channel();
        let (report, reported) = oneshot::channel();
        Self {
            task: tokio::spawn(run(cancellation, report)),
            reported: Some(reported),
            _cancel: cancel,
            terminal: None,
        }
    }
    pub(super) async fn result(&mut self) -> Result<()> {
        if let Some(result) = &self.terminal {
            return result.clone();
        }
        let Some(reported) = self.reported.as_mut() else {
            return self.join().await;
        };
        tokio::select! {
            joined = &mut self.task => {
                let result = joined.unwrap_or_else(|error| Err(FerrumError::internal(format!("serve lifecycle task failed: {error}")).into()));
                self.terminal = Some(result.clone()); result
            }
            report = reported => {
                self.reported = None;
                match report { Ok(result) => result, Err(_) => self.join().await }
            }
        }
    }
    pub(super) async fn join(&mut self) -> Result<()> {
        if let Some(result) = &self.terminal {
            return result.clone();
        }
        let result = (&mut self.task).await.unwrap_or_else(|error| {
            Err(FerrumError::internal(format!("serve lifecycle task failed: {error}")).into())
        });
        self.terminal = Some(result.clone());
        result
    }
}
// Dropping the observer closes cancellation; dropping JoinHandle detaches the
// same task. That task retains its actual server/engine/journal owners.
