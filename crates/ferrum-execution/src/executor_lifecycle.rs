//! Whole-executor retirement is distinct from a request or an Engine scope.
//!
//! The assembly that consumes the original executor owns final-close authority.
//! Observers retain custody but cannot manufacture that authority or cancel work.

use std::any::Any;
use std::sync::{Arc, Condvar, Mutex};

use ferrum_types::{FerrumError, Result};
use tokio::sync::watch;

use crate::ModelExecutor;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExecutorShutdownPhase {
    Draining,
    Retained,
    Retired,
}

#[derive(Clone, Debug)]
pub struct ExecutorShutdownReport {
    phase: ExecutorShutdownPhase,
    error: Option<FerrumError>,
}

impl ExecutorShutdownReport {
    pub fn phase(&self) -> ExecutorShutdownPhase {
        self.phase
    }

    pub fn error(&self) -> Option<&FerrumError> {
        self.error.as_ref()
    }

    /// An old execution error remains an error even after safe retirement.
    /// Conversely, absence of an error while draining is not a close receipt.
    pub fn result(&self) -> Result<()> {
        if let Some(error) = &self.error {
            return Err(error.clone());
        }
        if self.phase != ExecutorShutdownPhase::Retired {
            return Err(FerrumError::backend("executor retirement is not proved"));
        }
        Ok(())
    }
}

struct ReportPublisher {
    sender: watch::Sender<ExecutorShutdownReport>,
}

impl Drop for ReportPublisher {
    fn drop(&mut self) {
        self.sender.send_modify(|report| {
            if report.phase != ExecutorShutdownPhase::Retired {
                report.phase = ExecutorShutdownPhase::Retained;
                report.error.get_or_insert_with(|| {
                    FerrumError::internal("executor shutdown publisher ended without retirement")
                });
            }
        });
    }
}

/// Kept by the original cleanup service, never by a cancellable wait future.
#[derive(Clone)]
pub struct ExecutorShutdownReporter {
    inner: Arc<ReportPublisher>,
}

impl ExecutorShutdownReporter {
    pub fn retained(&self, error: FerrumError) {
        self.update(ExecutorShutdownPhase::Retained, Some(error));
    }

    /// Publish only after checked retirement and all owned workers have joined.
    /// Keeping an inert Rust handle is permitted; no device work may refer to it.
    pub fn retired(&self, error: Option<FerrumError>) {
        self.update(ExecutorShutdownPhase::Retired, error);
    }

    fn update(&self, phase: ExecutorShutdownPhase, error: Option<FerrumError>) {
        self.inner.sender.send_if_modified(|report| {
            if report.phase == ExecutorShutdownPhase::Retired {
                return false;
            }
            report.phase = phase;
            if report.error.is_none() {
                report.error = error;
            }
            true
        });
    }
}

/// A durable observation of the same close operation. Dropping or cancelling a
/// waiter never requests cancellation of the cleanup service.
#[derive(Clone)]
pub struct ExecutorShutdownObserver {
    receiver: watch::Receiver<ExecutorShutdownReport>,
    _custody: Arc<dyn Any + Send + Sync>,
    retry: Option<Arc<dyn Fn() + Send + Sync>>,
}

impl ExecutorShutdownObserver {
    pub fn channel(custody: Arc<dyn Any + Send + Sync>) -> (ExecutorShutdownReporter, Self) {
        Self::channel_with_error(custody, None)
    }

    pub(crate) fn channel_with_error(
        custody: Arc<dyn Any + Send + Sync>,
        error: Option<FerrumError>,
    ) -> (ExecutorShutdownReporter, Self) {
        let (sender, receiver) = watch::channel(ExecutorShutdownReport {
            phase: ExecutorShutdownPhase::Draining,
            error,
        });
        (
            ExecutorShutdownReporter {
                inner: Arc::new(ReportPublisher { sender }),
            },
            Self {
                receiver,
                _custody: custody,
                retry: None,
            },
        )
    }

    pub fn retained(custody: Arc<dyn Any + Send + Sync>, error: FerrumError) -> Self {
        let (reporter, observer) = Self::channel(custody);
        reporter.retained(error);
        observer
    }

    /// A cold-path wake of the original recovery service, never a new model or
    /// numerical replay. Keep callbacks out of their own custody graph.
    pub fn with_retry(mut self, retry: Arc<dyn Fn() + Send + Sync>) -> Self {
        self.retry = Some(retry);
        self
    }

    pub fn retry(&self) {
        if let Some(retry) = &self.retry {
            retry();
        }
    }

    pub fn snapshot(&self) -> ExecutorShutdownReport {
        self.receiver.borrow().clone()
    }

    /// Return the first result that is no longer simply draining. A retained
    /// owner is observable immediately rather than hidden behind infinite wait.
    pub async fn wait(&self) -> ExecutorShutdownReport {
        let mut observation = self.clone();
        loop {
            let report = observation.receiver.borrow_and_update().clone();
            if report.phase != ExecutorShutdownPhase::Draining {
                return report;
            }
            if observation.receiver.changed().await.is_err() {
                return observation.snapshot();
            }
        }
    }

    pub async fn changed(&mut self) -> ExecutorShutdownReport {
        // ReportPublisher publishes Retained before disconnecting. Do not turn
        // channel closure into a successful retirement receipt or a busy loop.
        // watch delivers the last unseen value before reporting disconnection.
        if self.receiver.changed().await.is_err() {
            let report = self.receiver.borrow_and_update().clone();
            if report.phase == ExecutorShutdownPhase::Retired {
                return report;
            }
            return std::future::pending().await;
        }
        self.receiver.borrow_and_update().clone()
    }
}

/// Unforgeable final-close permission tied to the original live allocation.
/// Address comparison confers no custody; the owning assembly retains the Arc
/// for the whole lifetime of this permission, preventing address reuse.
pub struct ExecutorShutdownPermit {
    executor_address: usize,
}

impl ExecutorShutdownPermit {
    pub fn check<T: ?Sized>(&self, executor: &T) -> Result<()> {
        if (executor as *const T).cast::<()>() as usize != self.executor_address {
            return Err(FerrumError::request_validation(
                "shutdown permission belongs to a different executor",
            ));
        }
        Ok(())
    }
}

struct OwnerGate {
    scopes: usize,
    sealed: bool,
    coordinator_running: bool,
    retry_generation: u64,
}

/// Separate custody from control to avoid a state -> observer -> state cycle.
/// The original executor Arc is moved, not cloned, into final retirement.
struct OwnerCustody {
    executor: Mutex<Option<Arc<dyn ModelExecutor>>>,
    shutdown: Mutex<Option<ExecutorShutdownObserver>>,
}

struct OwnerState {
    custody: Arc<OwnerCustody>,
    permit: ExecutorShutdownPermit,
    gate: Mutex<OwnerGate>,
    changed: Condvar,
    retry: watch::Sender<u64>,
    reporter: ExecutorShutdownReporter,
    observer: ExecutorShutdownObserver,
}

/// Sole final-close authority. Construction consumes an owned implementation;
/// an arbitrary shared Arc cannot be promoted into another owner.
pub struct ExecutorOwner {
    state: Arc<OwnerState>,
}

impl ExecutorOwner {
    pub fn new(executor: Box<dyn ModelExecutor>) -> Self {
        let executor: Arc<dyn ModelExecutor> = Arc::from(executor);
        let permit = ExecutorShutdownPermit {
            executor_address: Arc::as_ptr(&executor).cast::<()>() as usize,
        };
        let custody = Arc::new(OwnerCustody {
            executor: Mutex::new(Some(executor)),
            shutdown: Mutex::new(None),
        });
        let (reporter, observer) = ExecutorShutdownObserver::channel(custody.clone());
        let (retry, _) = watch::channel(0);
        Self {
            state: Arc::new(OwnerState {
                custody,
                permit,
                gate: Mutex::new(OwnerGate {
                    scopes: 0,
                    sealed: false,
                    coordinator_running: false,
                    retry_generation: 0,
                }),
                changed: Condvar::new(),
                retry,
                reporter,
                observer,
            }),
        }
    }

    /// Each engine or direct user takes its own non-cloneable use scope.
    pub fn scope(&self) -> Result<ExecutorScope> {
        let mut gate = self
            .state
            .gate
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        if gate.sealed {
            return Err(FerrumError::request_validation("executor owner is closing"));
        }
        let executor = self
            .state
            .custody
            .executor
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .as_ref()
            .cloned()
            .ok_or_else(|| FerrumError::internal("open executor owner lost its implementation"))?;
        gate.scopes = gate
            .scopes
            .checked_add(1)
            .ok_or_else(|| FerrumError::internal("executor scope count overflow"))?;
        Ok(ExecutorScope {
            state: Arc::clone(&self.state),
            executor: Some(executor),
        })
    }

    /// Seal scope registration and arrange cleanup independently of the caller.
    /// A failed start remains observable and can be retried by this same owner.
    pub fn begin_shutdown(&self) -> ExecutorShutdownObserver {
        begin_owner_shutdown(&self.state);
        self.shutdown_observer()
    }

    /// Observe this original owner's future retirement without sealing its
    /// scopes or starting cleanup. Retry only wakes an already-authorized close.
    pub fn shutdown_observer(&self) -> ExecutorShutdownObserver {
        let state = Arc::downgrade(&self.state);
        self.state.observer.clone().with_retry(Arc::new(move || {
            if let Some(state) = state.upgrade() {
                let sealed = state
                    .gate
                    .lock()
                    .unwrap_or_else(|error| error.into_inner())
                    .sealed;
                if sealed {
                    begin_owner_shutdown(&state);
                }
            }
        }))
    }
}

impl Drop for ExecutorOwner {
    fn drop(&mut self) {
        begin_owner_shutdown(&self.state);
        let gate = self
            .state
            .gate
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        if !gate.coordinator_running
            && self.state.observer.snapshot().phase != ExecutorShutdownPhase::Retired
        {
            // OS thread creation failed even on the final owner's last try.
            // There is no recovery service to receive custody. Quarantine this
            // one original instance rather than release unknown device work.
            // This is a reported failed shutdown, never a successful receipt.
            tracing::error!("executor close service could not start; original owner quarantined");
            std::mem::forget(Arc::clone(&self.state));
        }
    }
}

/// Borrowed execution use, with no final-close authority. Drop this only after
/// its actual requests and any borrowed operation futures have finished.
/// Rust borrows prevent moving/dropping the scope while its direct work is live.
pub struct ExecutorScope {
    state: Arc<OwnerState>,
    executor: Option<Arc<dyn ModelExecutor>>,
}

impl std::ops::Deref for ExecutorScope {
    type Target = dyn ModelExecutor;

    fn deref(&self) -> &Self::Target {
        self.executor
            .as_deref()
            .expect("a live scope retains its executor")
    }
}

impl AsRef<dyn ModelExecutor> for ExecutorScope {
    fn as_ref(&self) -> &(dyn ModelExecutor + 'static) {
        std::ops::Deref::deref(self)
    }
}

impl Drop for ExecutorScope {
    fn drop(&mut self) {
        // Drop the real use reference before waking the final owner. Merely
        // decrementing the count first races Arc::try_unwrap at retirement.
        drop(self.executor.take());
        let mut gate = self
            .state
            .gate
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        gate.scopes -= 1;
        self.state.changed.notify_all();
    }
}

fn begin_owner_shutdown(state: &Arc<OwnerState>) {
    let mut gate = state.gate.lock().unwrap_or_else(|error| error.into_inner());
    gate.sealed = true;
    if state.observer.snapshot().phase == ExecutorShutdownPhase::Retired {
        return;
    }
    gate.retry_generation = gate.retry_generation.wrapping_add(1);
    state.retry.send_replace(gate.retry_generation);
    state.changed.notify_all();
    if gate.coordinator_running {
        return;
    }
    gate.coordinator_running = true;
    let owned = Arc::clone(state);
    let spawn = std::thread::Builder::new()
        .name("ferrum-executor-close".to_owned())
        .spawn(move || run_owner_shutdown(owned));
    if let Err(error) = spawn {
        gate.coordinator_running = false;
        state.reporter.retained(FerrumError::internal(format!(
            "could not start executor close: {error}"
        )));
    }
}

fn wait_for_retry(state: &OwnerState, generation: u64) {
    let mut gate = state.gate.lock().unwrap_or_else(|error| error.into_inner());
    while gate.retry_generation == generation {
        gate = state
            .changed
            .wait(gate)
            .unwrap_or_else(|error| error.into_inner());
    }
}

fn run_owner_shutdown(state: Arc<OwnerState>) {
    let mut gate = state.gate.lock().unwrap_or_else(|error| error.into_inner());
    while gate.scopes != 0 {
        gate = state
            .changed
            .wait(gate)
            .unwrap_or_else(|error| error.into_inner());
    }
    drop(gate);
    // This one cold service retains custody even if every observer and the
    // public owner disappear. Failure parks on an explicit wake, never spins.
    loop {
        let generation = *state.retry.borrow();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let runtime = tokio::runtime::Builder::new_current_thread()
                .build()
                .map_err(|error| {
                    FerrumError::internal(format!("close observer runtime: {error}"))
                })?;
            let existing = state
                .custody
                .shutdown
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .clone();
            let mut observer = match existing {
                Some(observer) => observer,
                None => {
                    let executor = state
                        .custody
                        .executor
                        .lock()
                        .unwrap_or_else(|error| error.into_inner())
                        .take()
                        .ok_or_else(|| {
                            FerrumError::internal("executor close lost its original handoff")
                        })?;
                    let observer = executor.begin_shutdown(&state.permit);
                    *state
                        .custody
                        .shutdown
                        .lock()
                        .unwrap_or_else(|error| error.into_inner()) = Some(observer.clone());
                    observer
                }
            };
            let mut retries = state.retry.subscribe();
            runtime.block_on(async {
                let mut report = observer.snapshot();
                let mut disconnected = false;
                loop {
                    match report.phase {
                        ExecutorShutdownPhase::Draining => {}
                        ExecutorShutdownPhase::Retained => {
                            if let Some(error) = report.error.clone() {
                                state.reporter.retained(error);
                            }
                        }
                        ExecutorShutdownPhase::Retired => {
                            state.reporter.retired(report.error);
                            return Ok::<(), FerrumError>(());
                        }
                    }
                    tokio::select! {
                        result = observer.receiver.changed(), if !disconnected => {
                            disconnected = result.is_err();
                            // Consume the final value even when its publisher
                            // disconnected immediately after sending Retired.
                            report = observer.receiver.borrow_and_update().clone();
                        }
                        _ = retries.changed() => {
                            observer.retry();
                            report = observer.snapshot();
                        }
                    }
                    // A disconnected Retained receipt is not permission to
                    // drop its custody. Keep this original service parked;
                    // an explicit retry may wake its original cleanup hook.
                }
            })
        }));
        match result {
            Ok(Ok(())) => {
                state
                    .gate
                    .lock()
                    .unwrap_or_else(|error| error.into_inner())
                    .coordinator_running = false;
                return;
            }
            Ok(Err(error)) => state.reporter.retained(error),
            Err(_) => state
                .reporter
                .retained(FerrumError::internal("executor close coordinator panicked")),
        }
        wait_for_retry(&state, generation);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model_executor::{ExecutorCapabilities, ExecutorStatus};
    use ferrum_types::ModelInfo;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    struct CloseProbe {
        calls: Arc<AtomicUsize>,
    }

    #[async_trait::async_trait]
    impl ModelExecutor for CloseProbe {
        fn info(&self) -> &ModelInfo {
            unreachable!("the close protocol does not inspect model metadata")
        }
        fn capabilities(&self) -> ExecutorCapabilities {
            unreachable!("the close protocol does not perform admission")
        }
        fn status(&self) -> ExecutorStatus {
            unreachable!("the close protocol uses its original receipt")
        }
        fn begin_shutdown(
            self: Arc<Self>,
            permit: &ExecutorShutdownPermit,
        ) -> ExecutorShutdownObserver {
            permit.check(self.as_ref()).unwrap();
            // The factory's sole original reference must actually be moved.
            // Scope counts alone cannot prove that use references have gone.
            let original = Arc::try_unwrap(self)
                .ok()
                .expect("original executor has no remaining use references");
            original.calls.fetch_add(1, Ordering::Relaxed);
            let (reporter, observer) = ExecutorShutdownObserver::channel(Arc::new(()));
            reporter.retired(None);
            observer
        }
    }

    #[tokio::test]
    async fn shutdown_observation_cannot_seal_or_start_original_owner() {
        let calls = Arc::new(AtomicUsize::new(0));
        let owner = ExecutorOwner::new(Box::new(CloseProbe {
            calls: calls.clone(),
        }));
        let observer = owner.shutdown_observer();
        observer.retry();
        let scope = owner.scope().expect("observation cannot seal registration");
        assert_eq!(calls.load(Ordering::Relaxed), 0);
        drop(owner);
        assert_eq!(observer.snapshot().phase(), ExecutorShutdownPhase::Draining);
        drop(scope);
        tokio::time::timeout(Duration::from_secs(5), observer.wait())
            .await
            .unwrap()
            .result()
            .unwrap();
        assert_eq!(calls.load(Ordering::Relaxed), 1);
    }

    #[tokio::test]
    async fn final_owner_waits_for_every_real_scope_and_transfers_original_once() {
        let calls = Arc::new(AtomicUsize::new(0));
        let owner = ExecutorOwner::new(Box::new(CloseProbe {
            calls: calls.clone(),
        }));
        let first = owner.scope().unwrap();
        let second = owner.scope().unwrap();
        drop(first);
        let observer = owner.begin_shutdown();
        assert!(owner.scope().is_err());
        assert_eq!(observer.snapshot().phase(), ExecutorShutdownPhase::Draining);
        assert_eq!(calls.load(Ordering::Relaxed), 0);
        // A peer's live scope still has its ordinary executor access.
        assert_eq!(second.kv_capacity(), None);
        drop(second);
        let report = tokio::time::timeout(Duration::from_secs(5), observer.wait())
            .await
            .unwrap();
        assert_eq!(report.phase(), ExecutorShutdownPhase::Retired);
        report.result().unwrap();
        owner.begin_shutdown().wait().await.result().unwrap();
        assert_eq!(calls.load(Ordering::Relaxed), 1);
    }

    #[tokio::test]
    async fn owner_drop_and_cancelled_wait_do_not_cancel_original_close() {
        let calls = Arc::new(AtomicUsize::new(0));
        let owner = ExecutorOwner::new(Box::new(CloseProbe {
            calls: calls.clone(),
        }));
        let scope = owner.scope().unwrap();
        let observer = owner.begin_shutdown();
        let mut waiting = Box::pin(observer.wait());
        assert!(futures::poll!(&mut waiting).is_pending());
        drop(waiting);
        drop(owner);
        assert_eq!(calls.load(Ordering::Relaxed), 0);
        drop(scope);
        tokio::time::timeout(Duration::from_secs(5), observer.wait())
            .await
            .unwrap()
            .result()
            .unwrap();
        assert_eq!(calls.load(Ordering::Relaxed), 1);
    }

    #[tokio::test]
    async fn retired_with_prior_error_is_distinct_from_unproved_retirement() {
        let (reporter, observer) = ExecutorShutdownObserver::channel(Arc::new(()));
        reporter.retained(FerrumError::backend("original operation failed"));
        let retained = observer.wait().await;
        assert_eq!(retained.phase(), ExecutorShutdownPhase::Retained);
        reporter.retained(FerrumError::backend("later cleanup failed"));
        reporter.retired(None);
        reporter.retained(FerrumError::backend("late notification"));
        drop(reporter);
        let retired = observer.wait().await;
        assert_eq!(retired.phase(), ExecutorShutdownPhase::Retired);
        assert_eq!(
            retired.result().unwrap_err().to_string(),
            retained.result().unwrap_err().to_string()
        );
    }

    #[tokio::test]
    async fn disconnected_publisher_does_not_manufacture_retirement_or_drop_observer_custody() {
        let custody = Arc::new(());
        let weak = Arc::downgrade(&custody);
        let (reporter, observer) = ExecutorShutdownObserver::channel(custody);
        drop(reporter);
        let result = observer.wait().await;
        assert_eq!(result.phase(), ExecutorShutdownPhase::Retained);
        assert!(result.result().is_err());
        assert!(weak.upgrade().is_some());
        let mut changes = observer.clone();
        assert_eq!(
            changes.changed().await.phase(),
            ExecutorShutdownPhase::Retained
        );
        // Once the final retained report was observed, a closed publisher must
        // not wake a background cleanup loop continuously with the same value.
        let mut next = Box::pin(changes.changed());
        assert!(futures::poll!(&mut next).is_pending());
        drop(next);
        drop(changes);
        drop(observer);
        assert!(weak.upgrade().is_none());
    }
}
