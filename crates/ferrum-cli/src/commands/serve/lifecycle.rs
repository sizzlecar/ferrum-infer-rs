//! A command may report a timeout while its original lifecycle still owns cleanup.
use ferrum_types::{FerrumError, Result};
use std::future::Future;
use tokio::sync::oneshot;

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
