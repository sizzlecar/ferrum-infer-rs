//! One task owns iteration and shutdown cleanup. Public handles only observe it.

use super::*;

#[derive(Default)]
enum DriverState {
    #[default]
    Idle,
    Running(Arc<DriverTask>),
    Closing(Arc<DriverTask>),
    Finished(Arc<DriverTask>),
}

struct DriverTask {
    join: tokio::sync::Mutex<JoinOwner>,
    // A task-level panic must not discard the last product owner before its
    // JoinError is observed. EngineInner has only a Weak control reference.
    custody: Mutex<Option<Arc<EngineInner>>>,
    journal_custody: Mutex<Option<Arc<Mutex<EngineJournals>>>>,
    executor_shutdown: Mutex<Option<ExecutorShutdownObserver>>,
    journal_policy: ferrum_runtime::ExecutionEventDeliveryPolicy,
}

enum JoinOwner {
    Task(tokio::task::JoinHandle<Arc<DriverExitReport>>),
    Joined(Arc<DriverExitReport>),
}

/// Shared observation of the engine's one managed task; dropping it never aborts work.
#[derive(Clone)]
pub struct DriverObserver {
    task: Arc<DriverTask>,
}

impl std::fmt::Debug for DriverObserver {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DriverObserver").finish_non_exhaustive()
    }
}

/// Driver, delivery and cleanup facts. Diagnostics do not confer resource custody.
pub struct DriverExitReport {
    errors: Vec<FerrumError>,
    telemetry_degradations: Vec<FerrumError>,
    journal_deliveries: Vec<DriverJournalDelivery>,
    optional_observation_failures: u64,
    retained: Option<Arc<RetainedInnerCleanup>>,
    cleanup_unproved: bool,
    executor_shutdown: Option<ExecutorShutdownObserver>,
    retained_journals: Option<Arc<Mutex<EngineJournals>>>,
}

/// Final facts from the original ordered journal, including earlier dropped records.
#[derive(Debug, Clone)]
pub struct DriverJournalDelivery {
    pub path: PathBuf,
    /// Dynamic whole-journal health, including other assemblies' producers.
    /// Scope completion is proved by `scope_receipt`, not these counters.
    pub stats: ferrum_journal::jsonl_journal::JsonlJournalDeliveryStats,
    pub scope_receipt: Option<ferrum_journal::jsonl_journal::JsonlFlushReceipt>,
    /// Present only when this standalone engine owns final-close authority.
    pub final_receipt: Option<ferrum_journal::jsonl_journal::JsonlFinalReceipt>,
}

impl DriverExitReport {
    pub fn result(&self) -> Result<()> {
        self.errors.first().cloned().map_or(Ok(()), Err)
    }

    pub fn errors(&self) -> &[FerrumError] {
        &self.errors
    }

    pub fn telemetry_degradations(&self) -> &[FerrumError] {
        &self.telemetry_degradations
    }

    pub fn journal_deliveries(&self) -> &[DriverJournalDelivery] {
        &self.journal_deliveries
    }

    /// Failures reported directly by Engine, including journal open failures.
    /// Native sink delivery is counted by each original journal instead; use
    /// `journal_deliveries()` and `telemetry_degradations()` for that loss.
    pub fn optional_observation_failures(&self) -> u64 {
        self.optional_observation_failures
    }

    pub async fn wait_for_shutdown_completion(&self) -> Result<()> {
        // No later observation can erase an unwind or an unreclaimed request
        // owner. Do not turn its diagnostic error into permission to close the
        // product's final journal owner.
        let retained_result = if let Some(cleanup) = &self.retained {
            cleanup.wait().await
        } else if self.cleanup_unproved {
            return std::future::pending().await;
        } else {
            Ok(())
        };
        if self
            .retained_journals
            .as_ref()
            .is_some_and(|journals| journals.lock().has_unfinished_custody())
        {
            return std::future::pending().await;
        }
        self.result().and(retained_result)
    }

    #[cfg(test)]
    pub(super) fn cleanup_completion(&self) -> tokio::sync::watch::Receiver<bool> {
        self.retained
            .as_ref()
            .expect("retained cleanup service")
            .completed
            .subscribe()
    }

    pub fn executor_shutdown(&self) -> Option<&ExecutorShutdownObserver> {
        self.executor_shutdown.as_ref()
    }

    /// Reobserve the same retained cleanup after its original owner publishes
    /// new evidence; this never issues another physical request close.
    pub fn retry_shutdown(&self) {
        if let Some(cleanup) = &self.retained {
            cleanup.retry.notify_one();
        }
    }

    pub fn retains_unproved_owners(&self) -> bool {
        (self.cleanup_unproved && self.retained.is_none())
            || self
                .retained_journals
                .as_ref()
                .is_some_and(|journals| journals.lock().has_unfinished_custody())
            || self
                .retained
                .as_ref()
                .is_some_and(|cleanup| !*cleanup.completed.borrow())
    }
}

impl std::fmt::Debug for DriverExitReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DriverExitReport")
            .field("errors", &self.errors)
            .field("telemetry_degradations", &self.telemetry_degradations)
            .field("journal_deliveries", &self.journal_deliveries)
            .field(
                "optional_observation_failures",
                &self.optional_observation_failures,
            )
            .field("retains_unproved_owners", &self.retains_unproved_owners())
            .finish()
    }
}

impl DriverObserver {
    pub async fn wait(&self) -> Arc<DriverExitReport> {
        let mut join = self.task.join.lock().await;
        let result = match &mut *join {
            JoinOwner::Joined(report) => return Arc::clone(report),
            // Borrow the registered handle: cancellation of this wait must not
            // detach it or leave another observer with an empty owner slot.
            JoinOwner::Task(handle) => handle.await,
        };
        let report = match result {
            Ok(report) => report,
            Err(error) => Arc::new(DriverExitReport {
                errors: vec![FerrumError::internal(format!(
                    "managed engine task failed: {error}"
                ))],
                retained: self
                    .task
                    .custody
                    .lock()
                    .take()
                    .map(|inner| RetainedInnerCleanup::start(inner, None)),
                telemetry_degradations: Vec::new(),
                journal_deliveries: Vec::new(),
                optional_observation_failures: 0,
                cleanup_unproved: true,
                executor_shutdown: None,
                retained_journals: self.task.journal_custody.lock().take(),
            }),
        };
        self.task.custody.lock().take();
        self.task.journal_custody.lock().take();
        *join = JoinOwner::Joined(Arc::clone(&report));
        report
    }
}

#[derive(Default)]
pub(super) struct ManagedDriver {
    pub(super) active: Mutex<Option<Arc<EngineInner>>>,
    pub(super) final_owner: Mutex<Option<ExecutorOwner>>,
    pub(super) final_memory: Mutex<Option<ferrum_types::MemoryUsage>>,
    pub(super) final_performance: Mutex<ferrum_types::PerformanceBreakdown>,
    pub(super) final_prefix_hits: AtomicU64,
    state: Mutex<DriverState>,
    // Read-only, monotonic mirror: recompute queries this while holding
    // sequences, whereas publication holds gate -> sequences. Taking the
    // gate from that query would invert the lock order. Only the gate grants
    // admission; this latch never authorizes starting or publishing work.
    closing: AtomicBool,
    #[cfg(test)]
    pub(super) panic_next_iteration: AtomicBool,
    #[cfg(test)]
    starts: AtomicU64,
}

impl ManagedDriver {
    pub(super) fn shutdown_active(self: &Arc<Self>) -> Result<DriverObserver> {
        let active = self.active.lock().clone();
        if let Some(inner) = active {
            return self.shutdown(&inner);
        }
        let state = self.state.lock();
        match &*state {
            DriverState::Running(task)
            | DriverState::Closing(task)
            | DriverState::Finished(task) => Ok(DriverObserver {
                task: Arc::clone(task),
            }),
            DriverState::Idle => Err(Self::admission_closed()),
        }
    }

    fn spawn_locked(
        self: &Arc<Self>,
        inner: &Arc<EngineInner>,
        state: &mut DriverState,
        run_iterations: bool,
    ) -> Result<Arc<DriverTask>> {
        let runtime = tokio::runtime::Handle::try_current().map_err(|error| {
            FerrumError::internal(format!(
                "starting engine requires an async runtime: {error}"
            ))
        })?;
        let driver = Arc::clone(self);
        let task_inner = Arc::clone(inner);
        let handle = runtime.spawn(async move {
            match std::panic::AssertUnwindSafe(Arc::clone(&driver).run(task_inner, run_iterations))
                .catch_unwind()
                .await
            {
                Ok(report) => report,
                Err(_) => driver.retain_failed_task(FerrumError::internal(
                    "managed engine task panicked outside cleanup",
                )),
            }
        });
        let task = Arc::new(DriverTask {
            join: tokio::sync::Mutex::new(JoinOwner::Task(handle)),
            custody: Mutex::new(Some(Arc::clone(inner))),
            journal_custody: Mutex::new(Some(Arc::clone(&inner.journal_scopes))),
            executor_shutdown: Mutex::new(None),
            journal_policy: observation_delivery_policy(inner.config.runtime.profile_detail),
        });
        if !run_iterations {
            self.closing.store(true, Ordering::Release);
        }
        *state = if run_iterations {
            DriverState::Running(Arc::clone(&task))
        } else {
            DriverState::Closing(Arc::clone(&task))
        };
        #[cfg(test)]
        self.starts.fetch_add(1, Ordering::Relaxed);
        Ok(task)
    }

    pub(super) fn start(self: &Arc<Self>, inner: &Arc<EngineInner>) -> Result<DriverObserver> {
        let mut state = self.state.lock();
        let task = match &*state {
            DriverState::Idle => self.spawn_locked(inner, &mut state, true)?,
            DriverState::Running(task) => Arc::clone(task),
            DriverState::Closing(_) | DriverState::Finished(_) => {
                return Err(Self::admission_closed());
            }
        };
        Ok(DriverObserver { task })
    }

    /// Called under iteration_lock; publication has no await while this gate is held.
    pub(super) fn publish<T>(
        self: &Arc<Self>,
        inner: &Arc<EngineInner>,
        publish: impl FnOnce() -> Result<T>,
    ) -> Result<T> {
        let mut state = self.state.lock();
        match &*state {
            DriverState::Idle => {
                self.spawn_locked(inner, &mut state, true)?;
            }
            DriverState::Running(_) => {}
            DriverState::Closing(_) | DriverState::Finished(_) => {
                return Err(Self::admission_closed());
            }
        }
        publish()
    }

    pub(super) fn shutdown(self: &Arc<Self>, inner: &Arc<EngineInner>) -> Result<DriverObserver> {
        let task = {
            let mut state = self.state.lock();
            match &*state {
                DriverState::Idle => self.spawn_locked(inner, &mut state, false)?,
                DriverState::Running(task) => {
                    let task = Arc::clone(task);
                    self.closing.store(true, Ordering::Release);
                    *state = DriverState::Closing(Arc::clone(&task));
                    task
                }
                DriverState::Closing(task) | DriverState::Finished(task) => Arc::clone(task),
            }
        };
        // notify_one preserves the wake when the loop has not registered yet.
        inner.shutdown_notify.notify_one();
        inner.work_notify.notify_one();
        Ok(DriverObserver { task })
    }

    pub(super) fn is_running(&self) -> bool {
        matches!(*self.state.lock(), DriverState::Running(_))
    }

    pub(super) fn is_closing(&self) -> bool {
        self.closing.load(Ordering::Acquire)
    }

    #[cfg(test)]
    pub(super) fn start_count(&self) -> u64 {
        self.starts.load(Ordering::Relaxed)
    }

    fn admission_closed() -> FerrumError {
        FerrumError::scheduler("engine is closing and no longer accepts requests")
    }

    fn mark_closing(&self) {
        // New callers can no longer acquire an execution borrow. Existing
        // callers and the driver retain their original Inner until they exit.
        self.active.lock().take();
        let mut state = self.state.lock();
        self.closing.store(true, Ordering::Release);
        if let DriverState::Running(task) = &*state {
            *state = DriverState::Closing(Arc::clone(task));
        }
    }

    fn mark_finished(&self) {
        let mut state = self.state.lock();
        self.closing.store(true, Ordering::Release);
        if let DriverState::Running(task) | DriverState::Closing(task) = &*state {
            *state = DriverState::Finished(Arc::clone(task));
        }
    }

    async fn run(
        self: Arc<Self>,
        inner: Arc<EngineInner>,
        run_iterations: bool,
    ) -> Arc<DriverExitReport> {
        let mut exit = DriverExitGuard {
            driver: &self,
            armed: true,
        };
        let mut errors = Vec::new();
        if run_iterations {
            if std::panic::AssertUnwindSafe(self.run_iterations(&inner))
                .catch_unwind()
                .await
                .is_err()
            {
                errors.push(FerrumError::internal(
                    "managed engine iteration task panicked",
                ));
            }
        }
        // A body panic must close admission and still execute the independent
        // cleanup future. One catch around both would skip that cleanup.
        self.mark_closing();
        let (cleanup, cleanup_unproved) =
            match std::panic::AssertUnwindSafe(inner.shutdown_cleanup())
                .catch_unwind()
                .await
            {
                Ok(cleanup) => (cleanup, false),
                Err(_) => {
                    errors.push(FerrumError::internal("managed engine cleanup panicked"));
                    (ShutdownCleanupReport::default(), true)
                }
            };
        let mut cleanup = cleanup;
        let has_owners = !inner.sequences.is_empty()
            || inner.execution_closes.unreclaimed_count() != 0
            || inner.execution_readiness_waiters.pending_count() != 0;
        *self.final_memory.lock() = Some(inner.memory_snapshot().0);
        *self.final_performance.lock() = inner.performance_breakdown();
        self.final_prefix_hits.store(
            inner.prefix_cache_hits.load(Ordering::Relaxed),
            Ordering::Relaxed,
        );
        let journals = Arc::clone(&inner.journal_scopes);
        let policy = observation_delivery_policy(inner.config.runtime.profile_detail);
        let mut retained = None;
        let mut executor_shutdown = None;
        if cleanup_unproved || has_owners {
            // Transfer the task fallback before starting recovery: neither a
            // report nor the completed task may keep an extra execution scope.
            if let DriverState::Closing(task) | DriverState::Running(task) = &*self.state.lock() {
                task.custody.lock().take();
            }
            retained = Some(RetainedInnerCleanup::start(
                inner,
                self.final_owner.lock().take(),
            ));
        } else {
            // The task's panic fallback is another real execution borrow.
            // Release it before waiting for the assembly's final close.
            if let DriverState::Closing(task) | DriverState::Running(task) = &*self.state.lock() {
                task.custody.lock().take();
            }
            drop(inner);
            let owner = self.final_owner.lock().take();
            if let Some(owner) = owner {
                let observer = owner.begin_shutdown();
                if let DriverState::Closing(task) | DriverState::Running(task) = &*self.state.lock()
                {
                    *task.executor_shutdown.lock() = Some(observer.clone());
                }
                let close = observer.wait().await;
                if let Err(error) = close.result() {
                    cleanup.errors.push(error);
                }
                executor_shutdown = Some(observer.clone());
                if close.phase() != ExecutorShutdownPhase::Retired {
                    retained = Some(RetainedInnerCleanup::start_journals(
                        Arc::clone(&journals),
                        policy,
                        Some(owner),
                        Some(observer),
                    ));
                    cleanup.journal_deliveries = journals.lock().pending_deliveries();
                } else {
                    finish_journals(&journals, policy, &mut cleanup).await;
                }
            } else {
                // A shared Engine finishes only its own producer scope. Its
                // external executor owner decides when all peer scopes end.
                finish_journals(&journals, policy, &mut cleanup).await;
            }
        }
        let retained_journals = journals
            .lock()
            .has_unfinished_custody()
            .then(|| Arc::clone(&journals));
        if retained.is_none() && retained_journals.is_some() {
            // Model and request retirement are already proved; an unproved
            // journal close still needs its own durable, retryable custody.
            retained = Some(RetainedInnerCleanup::start_journals(
                Arc::clone(&journals),
                policy,
                None,
                None,
            ));
        }
        errors.extend(cleanup.errors);
        let report = Arc::new(DriverExitReport {
            retained,
            errors,
            telemetry_degradations: cleanup.telemetry_degradations,
            journal_deliveries: cleanup.journal_deliveries,
            optional_observation_failures: cleanup.optional_observation_failures,
            cleanup_unproved,
            executor_shutdown,
            retained_journals,
        });
        self.mark_finished();
        exit.armed = false;
        // The completed report now owns any unresolved custody. Keep the
        // fallback only for exits without a report; an observer that never
        // joins must not retain a clean EngineInner through the task handle.
        // No await follows this transfer.
        if let DriverState::Finished(task) = &*self.state.lock() {
            task.custody.lock().take();
            task.journal_custody.lock().take();
            task.executor_shutdown.lock().take();
        }
        report
    }

    fn retain_failed_task(&self, error: FerrumError) -> Arc<DriverExitReport> {
        self.mark_closing();
        let (inner, journals, executor_shutdown, policy) = match &*self.state.lock() {
            DriverState::Running(task)
            | DriverState::Closing(task)
            | DriverState::Finished(task) => (
                task.custody.lock().take(),
                task.journal_custody.lock().take(),
                task.executor_shutdown.lock().take(),
                Some(task.journal_policy),
            ),
            DriverState::Idle => (None, None, None, None),
        };
        let owner = self.final_owner.lock().take();
        let retained = if let Some(inner) = inner {
            Some(RetainedInnerCleanup::start(inner, owner))
        } else if let (Some(journals), Some(policy)) = (&journals, policy) {
            Some(RetainedInnerCleanup::start_journals(
                Arc::clone(journals),
                policy,
                owner,
                executor_shutdown.clone(),
            ))
        } else {
            None
        };
        let report = Arc::new(DriverExitReport {
            errors: vec![error],
            telemetry_degradations: Vec::new(),
            journal_deliveries: journals
                .as_ref()
                .map(|journals| journals.lock().pending_deliveries())
                .unwrap_or_default(),
            optional_observation_failures: 0,
            retained,
            cleanup_unproved: true,
            executor_shutdown,
            retained_journals: journals,
        });
        self.mark_finished();
        report
    }

    async fn run_iterations(&self, inner: &EngineInner) {
        info!("Background iteration loop started");
        // Subscribe before the first planning load. A publication after the
        // load but before the wait remains pending in this original receiver.
        let mut prediction_updates = inner
            .cost_snapshot_source
            .as_ref()
            .map(|source| source.subscribe());
        while self.is_running() {
            if let Some(updates) = prediction_updates.as_mut() {
                drop(updates.borrow_and_update());
            }
            #[cfg(test)]
            if self.panic_next_iteration.swap(false, Ordering::AcqRel) {
                panic!("injected managed driver iteration panic");
            }
            let outcome = match inner.run_iteration().await {
                Ok(outcome) => outcome,
                Err(error) => {
                    warn!("Iteration error: {}", error);
                    EngineIterationOutcome::Progressed
                }
            };
            if !self.is_running() {
                break;
            }
            match outcome {
                EngineIterationOutcome::Progressed => tokio::task::yield_now().await,
                EngineIterationOutcome::Waiting { capacity, retry_at } => {
                    wait_for_iteration_wake(inner, capacity, retry_at, &mut prediction_updates)
                        .await;
                }
            }
        }
        info!("Background iteration loop stopped");
    }
}

pub(super) async fn wait_for_iteration_wake(
    inner: &EngineInner,
    capacity: Option<ExecutorCapacityWaitRegistration>,
    retry_at: Option<Instant>,
    prediction_updates: &mut Option<tokio::sync::watch::Receiver<u64>>,
) {
    let deadline = inner.prefix_deadline().into_iter().chain(retry_at).min();
    tokio::select! {
        _ = wait_for_deadline(deadline) => {}
        _ = inner.shutdown_notify.notified() => {}
        _ = inner.work_notify.notified() => {}
        _ = wait_for_prediction_update(prediction_updates) => {}
        result = async {
            match capacity {
                Some(registration) => registration.wait_for_change().await,
                None => std::future::pending().await,
            }
        } => {
            if let Err(error) = result {
                warn!("Executor capacity wait error: {}", error);
            }
        }
    }
}

pub(super) async fn wait_for_deadline(deadline: Option<Instant>) {
    match deadline {
        Some(deadline) => tokio::time::sleep_until(tokio::time::Instant::from_std(deadline)).await,
        None => std::future::pending().await,
    }
}

async fn wait_for_prediction_update(updates: &mut Option<tokio::sync::watch::Receiver<u64>>) {
    if let Some(receiver) = updates.as_mut() {
        if receiver.changed().await.is_ok() {
            return;
        }
        // Closing a watch is not new work. Disable this branch permanently;
        // the original deadline/shutdown/work/capacity waits still apply.
        *updates = None;
    }
    std::future::pending().await
}

struct DriverExitGuard<'a> {
    driver: &'a ManagedDriver,
    armed: bool,
}

impl Drop for DriverExitGuard<'_> {
    fn drop(&mut self) {
        if self.armed {
            self.driver.mark_finished();
        }
    }
}

#[derive(Default)]
struct ShutdownCleanupReport {
    errors: Vec<FerrumError>,
    telemetry_degradations: Vec<FerrumError>,
    journal_deliveries: Vec<DriverJournalDelivery>,
    optional_observation_failures: u64,
}

impl EngineInner {
    async fn shutdown_cleanup(&self) -> ShutdownCleanupReport {
        let mut report = ShutdownCleanupReport {
            errors: self.handoff_shutdown_requests(),
            ..ShutdownCleanupReport::default()
        };
        if let Err(error) = self.output_tasks.stop_and_join().await {
            report.errors.push(error);
        }
        if let Err(error) = self.execution_readiness_waiters.abort_and_join().await {
            report.errors.push(error);
        }
        self.execution_closes.join_observers().await;
        if let Err(error) = self.complete_failed_execution_recomputes().await {
            report.errors.push(error);
        }
        if let Err(error) = self.drain_execution_closes() {
            report.errors.push(error);
        }
        let unreclaimed = self.execution_closes.unreclaimed_count();
        let producers_unproved = unreclaimed > 0
            || !self.sequences.is_empty()
            || self.execution_readiness_waiters.pending_count() != 0;
        if producers_unproved {
            report.errors.push(FerrumError::internal(format!(
                "engine shutdown retains {unreclaimed} unproved execution closures and {} requests",
                self.sequences.len()
            )));
        }
        let (required, optional, optional_failures) = self.observation_health.snapshot();
        report.errors.extend(required);
        report.telemetry_degradations.extend(optional);
        report.optional_observation_failures = optional_failures;
        // Error reporting and proof of producer termination are independent.
        // A strict evidence error does not keep a stopped producer live;
        // unreclaimed execution or a cleanup unwind does. The latter never
        // reaches scope.finish, retaining its original scope in EngineInner.
        if producers_unproved {
            report.journal_deliveries = self.journal_scopes.lock().pending_deliveries();
            return report;
        }
        report
    }
}

/// The service owns the original Inner independently of all public reports.
/// Its observation contains no execution scope, so a report cannot block final
/// whole-model retirement once the original request obligations are consumed.
struct RetainedInnerCleanup {
    completed: tokio::sync::watch::Sender<bool>,
    retry: Arc<Notify>,
    error: Mutex<Option<FerrumError>>,
}

impl RetainedInnerCleanup {
    fn start(inner: Arc<EngineInner>, owner: Option<ExecutorOwner>) -> Arc<Self> {
        let (completed, _) = tokio::sync::watch::channel(false);
        let cleanup = Arc::new(Self {
            completed,
            retry: Arc::clone(&inner.work_notify),
            error: Mutex::new(None),
        });
        let service = Arc::clone(&cleanup);
        tokio::spawn(async move {
            loop {
                // Enable before checking the original state; a terminal
                // publication between the check and await preserves its wake.
                let notified = inner.work_notify.notified();
                tokio::pin!(notified);
                notified.as_mut().enable();
                let result = std::panic::AssertUnwindSafe(inner.shutdown_cleanup())
                    .catch_unwind()
                    .await;
                if let Ok(report) = result {
                    if let Some(error) = report.errors.into_iter().next() {
                        service.error.lock().get_or_insert(error);
                    }
                    if inner.sequences.is_empty()
                        && inner.execution_closes.unreclaimed_count() == 0
                        && inner.execution_readiness_waiters.pending_count() == 0
                    {
                        break;
                    }
                } else {
                    service.error.lock().get_or_insert_with(|| {
                        FerrumError::internal("retained engine cleanup panicked")
                    });
                }
                notified.await;
            }
            let journals = Arc::clone(&inner.journal_scopes);
            let policy = observation_delivery_policy(inner.config.runtime.profile_detail);
            drop(inner);
            service.finish(journals, policy, owner, None).await;
        });
        cleanup
    }

    fn start_journals(
        journals: Arc<Mutex<EngineJournals>>,
        policy: ferrum_runtime::ExecutionEventDeliveryPolicy,
        owner: Option<ExecutorOwner>,
        observer: Option<ExecutorShutdownObserver>,
    ) -> Arc<Self> {
        let (completed, _) = tokio::sync::watch::channel(false);
        let cleanup = Arc::new(Self {
            completed,
            retry: Arc::new(Notify::new()),
            error: Mutex::new(None),
        });
        let service = Arc::clone(&cleanup);
        tokio::spawn(async move {
            service.finish(journals, policy, owner, observer).await;
        });
        cleanup
    }

    async fn finish(
        &self,
        journals: Arc<Mutex<EngineJournals>>,
        policy: ferrum_runtime::ExecutionEventDeliveryPolicy,
        owner: Option<ExecutorOwner>,
        observer: Option<ExecutorShutdownObserver>,
    ) {
        let observer = observer.or_else(|| owner.as_ref().map(ExecutorOwner::begin_shutdown));
        if let Some(mut observer) = observer {
            loop {
                let notified = self.retry.notified();
                tokio::pin!(notified);
                notified.as_mut().enable();
                let report = observer.snapshot();
                if let Some(error) = report.error() {
                    self.error.lock().get_or_insert_with(|| error.clone());
                }
                if report.phase() == ExecutorShutdownPhase::Retired {
                    break;
                }
                tokio::select! {
                    _ = observer.changed() => {},
                    _ = notified => observer.retry(),
                }
            }
        }
        drop(owner);
        loop {
            let notified = self.retry.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            let mut report = ShutdownCleanupReport::default();
            finish_journals(&journals, policy, &mut report).await;
            if let Some(error) = report.errors.into_iter().next() {
                self.error.lock().get_or_insert(error);
            }
            if !journals.lock().has_unfinished_custody() {
                break;
            }
            notified.await;
        }
        self.completed.send_replace(true);
    }

    async fn wait(&self) -> Result<()> {
        let mut completed = self.completed.subscribe();
        loop {
            if *completed.borrow_and_update() {
                break;
            }
            if completed.changed().await.is_err() {
                return std::future::pending().await;
            }
        }
        self.error.lock().clone().map_or(Ok(()), Err)
    }
}

async fn finish_journals(
    journals: &Arc<Mutex<EngineJournals>>,
    policy: ferrum_runtime::ExecutionEventDeliveryPolicy,
    report: &mut ShutdownCleanupReport,
) {
    if !journals.lock().is_empty() {
        let journals = Arc::clone(journals);
        match tokio::task::spawn_blocking(move || journals.lock().finish()).await {
            Ok(results) => {
                for (delivery, error) in results {
                    report.journal_deliveries.push(delivery);
                    if let Some(error) = error {
                        let failure = FerrumError::internal(format!(
                            "execution observation close failed: {error}"
                        ));
                        if error.is_optional_transport(policy) {
                            // One managed close and one result per deduplicated
                            // destination: visible even to callers reading only result().
                            warn!(error = %error, "optional execution journal scope finished with incomplete telemetry");
                            report.telemetry_degradations.push(failure);
                        } else {
                            report.errors.push(failure);
                        }
                    }
                }
            }
            Err(error) => {
                let failure =
                    FerrumError::internal(format!("execution journal close task failed: {error}"));
                // A task unwind is an unknown internal failure, not I/O.
                report.errors.push(failure);
            }
        }
    }
}
