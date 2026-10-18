//! A command may report a timeout while its original lifecycle still owns cleanup.
use ferrum_types::{FerrumError, Result};
use std::future::Future;
use tokio::sync::oneshot;

/// A deadline limits observation, never the lifetime of the cleanup future.
/// All cleanup stages share one deadline, including stages after HTTP drain.
pub(super) struct ShutdownObservation {
    timeout: std::time::Duration,
    deadline: Option<tokio::time::Instant>,
    report: Option<oneshot::Sender<Result<()>>>,
    timed_out: Option<FerrumError>,
}

impl ShutdownObservation {
    pub(super) fn new(timeout: std::time::Duration, report: oneshot::Sender<Result<()>>) -> Self {
        Self {
            timeout,
            deadline: None,
            report: Some(report),
            timed_out: None,
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
                        let _ = report.send(Err(error.clone()));
                    }
                    self.timed_out = Some(error);
                }
            }
        }
        // Re-poll this exact future, not a fresh stop attempt or cleanup task.
        cleanup.await
    }

    pub(super) fn finish(self, actual: Result<()>) -> Result<()> {
        match self.timed_out {
            Some(timeout) => {
                if let Err(error) = actual {
                    tracing::warn!(%error, "serve cleanup later failed after reported timeout");
                }
                Err(timeout)
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
                let result = joined.unwrap_or_else(|error| Err(FerrumError::internal(format!("serve lifecycle task failed: {error}"))));
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
            Err(FerrumError::internal(format!(
                "serve lifecycle task failed: {error}"
            )))
        });
        self.terminal = Some(result.clone());
        result
    }
}
// Dropping the observer closes cancellation; dropping JoinHandle detaches the
// same task. That task retains its actual server/engine/journal owners.
