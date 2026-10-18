use super::*;

struct DeliveryTask {
    join: tokio::sync::Mutex<Option<tokio::task::JoinHandle<()>>>,
}

struct RegistryState {
    next_id: u64,
    closing: bool,
    tasks: HashMap<u64, Arc<DeliveryTask>>,
    error: Option<FerrumError>,
    pending_control_error: Option<FerrumError>,
    control_report: Option<FerrumError>,
}

/// Only client delivery tasks live here; physical owners stay in execution_closes.
pub(in crate::continuous_engine) struct OutputTaskRegistry {
    state: Mutex<RegistryState>,
    stop: tokio::sync::watch::Sender<bool>,
}

impl Default for OutputTaskRegistry {
    fn default() -> Self {
        Self {
            state: Mutex::new(RegistryState {
                next_id: 0,
                closing: false,
                tasks: HashMap::new(),
                error: None,
                pending_control_error: None,
                control_report: None,
            }),
            stop: tokio::sync::watch::channel(false).0,
        }
    }
}

impl OutputTaskRegistry {
    pub(in crate::continuous_engine) fn record_control_error(&self, error: FerrumError) {
        // The driver consumes this independently of delivery/reader progress.
        // Keep the first pending failure; every further failure is logged too.
        warn!(error = %error, "request completion control failed");
        let mut state = self.state.lock();
        state.pending_control_error.get_or_insert(error);
    }

    pub(in crate::continuous_engine) fn record_interrupted_error(&self, error: FerrumError) {
        warn!(error = %error, "shutdown interrupted known terminal error delivery");
        self.state.lock().control_report.get_or_insert(error);
    }

    pub(in crate::continuous_engine) fn take_control_error(&self) -> Option<FerrumError> {
        self.state.lock().pending_control_error.take()
    }

    pub(in crate::continuous_engine) fn stop_receiver(&self) -> tokio::sync::watch::Receiver<bool> {
        self.stop.subscribe()
    }

    pub(in crate::continuous_engine) fn spawn(
        self: &Arc<Self>,
        future: impl std::future::Future<Output = Result<()>> + Send + 'static,
    ) -> Result<tokio::sync::oneshot::Receiver<Result<()>>> {
        let mut state = self.state.lock();
        if state.closing {
            return Err(FerrumError::scheduler("output delivery is stopping"));
        }
        let id = state
            .next_id
            .checked_add(1)
            .ok_or_else(|| FerrumError::internal("output task identity space exhausted"))?;
        state.next_id = id;
        let runtime = tokio::runtime::Handle::try_current()
            .map_err(|error| FerrumError::internal(error.to_string()))?;
        let registry = Arc::downgrade(self);
        let (start, started) = tokio::sync::oneshot::channel();
        let (result, receiver) = tokio::sync::oneshot::channel();
        let handle = runtime.spawn(async move {
            if started.await.is_err() {
                return;
            }
            let outcome = std::panic::AssertUnwindSafe(future)
                .catch_unwind()
                .await
                .unwrap_or_else(|_| Err(FerrumError::internal("output delivery task panicked")));
            if let Some(registry) = registry.upgrade() {
                let mut state = registry.state.lock();
                if let Err(error) = &outcome {
                    state.error.get_or_insert_with(|| error.clone());
                }
                state.tasks.remove(&id);
            }
            let _ = result.send(outcome);
        });
        let task = Arc::new(DeliveryTask {
            join: tokio::sync::Mutex::new(Some(handle)),
        });
        state.tasks.insert(id, task);
        let _ = start.send(());
        Ok(receiver)
    }

    #[cfg(test)]
    pub(in crate::continuous_engine) async fn join_current(&self) {
        let tasks: Vec<_> = self.state.lock().tasks.values().cloned().collect();
        for task in tasks {
            let mut join = task.join.lock().await;
            if let Some(handle) = join.as_mut() {
                handle.await.expect("managed output task panicked");
                *join = None;
            }
        }
    }

    pub(in crate::continuous_engine) async fn stop_and_join(&self) -> Result<()> {
        let tasks = {
            let mut state = self.state.lock();
            state.closing = true;
            self.stop.send_replace(true);
            state.tasks.values().cloned().collect::<Vec<_>>()
        };
        for task in tasks {
            let mut join = task.join.lock().await;
            if let Some(handle) = join.as_mut() {
                if let Err(error) = handle.await {
                    self.state.lock().error.get_or_insert_with(|| {
                        FerrumError::internal(format!("output task join failed: {error}"))
                    });
                }
                *join = None;
            }
        }
        let mut state = self.state.lock();
        let pending = state.pending_control_error.take();
        state
            .error
            .clone()
            .or_else(|| state.control_report.clone())
            .or(pending)
            .map_or(Ok(()), Err)
    }
}
