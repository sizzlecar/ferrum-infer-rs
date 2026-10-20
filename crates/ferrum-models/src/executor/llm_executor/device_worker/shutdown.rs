//! Cold join custody for the original model worker. No per-request thread.

use super::*;
use ferrum_execution::executor_lifecycle::{ExecutorShutdownObserver, ExecutorShutdownReporter};
use parking_lot::Mutex;

pub(super) enum WorkerExit {
    Retired(Option<FerrumError>),
    Retained(FerrumError),
}

pub(super) struct WorkerShutdown {
    reporter: Mutex<Option<ExecutorShutdownReporter>>,
    observer: ExecutorShutdownObserver,
    handle: Mutex<Option<thread::JoinHandle<WorkerExit>>>,
    monitor_started: Mutex<bool>,
}

impl WorkerShutdown {
    pub(super) fn new(state: Arc<TokenWorkerState>) -> Arc<Self> {
        let (reporter, observer) = ExecutorShutdownObserver::channel(state);
        Arc::new(Self {
            reporter: Mutex::new(Some(reporter)),
            observer,
            handle: Mutex::new(None),
            monitor_started: Mutex::new(false),
        })
    }

    pub(super) fn install(&self, handle: thread::JoinHandle<WorkerExit>) {
        *self.handle.lock() = Some(handle);
    }

    pub(super) fn reporter(&self) -> Result<ExecutorShutdownReporter> {
        self.reporter
            .lock()
            .clone()
            .ok_or_else(|| FerrumError::internal("model worker reporter has already retired"))
    }

    pub(super) fn observe(self: &Arc<Self>) -> Result<ExecutorShutdownObserver> {
        self.start_monitor();
        // The base observer stored inside this object has no retry callback.
        // Only the outgoing clone holds this custody, so there is no self-cycle.
        let custody = Arc::clone(self);
        Ok(self
            .observer
            .clone()
            .with_retry(Arc::new(move || custody.start_monitor())))
    }

    fn start_monitor(self: &Arc<Self>) {
        let mut started = self.monitor_started.lock();
        if !*started {
            let custody = Arc::clone(self);
            let result = thread::Builder::new()
                .name("ferrum-llm-device-close".to_owned())
                .spawn(move || custody.join_original());
            if let Err(error) = result {
                let error = FerrumError::backend(format!(
                    "cannot observe original model worker exit: {error}"
                ));
                if let Some(reporter) = self.reporter.lock().as_ref() {
                    reporter.retained(error);
                }
                return;
            }
            *started = true;
        }
    }

    fn join_original(&self) {
        // Only this successfully spawned cold monitor can take the handle.
        // Dropping any async waiter cannot detach this ownership or the model.
        let Some(handle) = self.handle.lock().take() else {
            if let Some(reporter) = self.reporter.lock().take() {
                reporter.retained(FerrumError::internal(
                    "original model worker join handle is missing",
                ));
            }
            return;
        };
        let result = handle.join();
        let Some(reporter) = self.reporter.lock().take() else {
            return;
        };
        match result {
            Ok(WorkerExit::Retired(error)) => reporter.retired(error),
            Ok(WorkerExit::Retained(error)) => reporter.retained(error),
            Err(_) => reporter.retained(FerrumError::backend(
                "original model worker panicked before checked retirement",
            )),
        }
    }
}
