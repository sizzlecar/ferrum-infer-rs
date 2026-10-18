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
    retained: Option<Arc<EngineInner>>,
    cleanup_unproved: bool,
}

/// Final facts from the original ordered journal, including earlier dropped records.
#[derive(Debug, Clone)]
pub struct DriverJournalDelivery {
    pub path: PathBuf,
    pub stats: ferrum_bench_core::jsonl_journal::JsonlJournalDeliveryStats,
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

    pub fn retains_unproved_owners(&self) -> bool {
        self.retained.as_ref().is_some_and(|inner| {
            self.cleanup_unproved
                || !inner.sequences.is_empty()
                || inner.execution_closes.unreclaimed_count() != 0
                || inner.execution_readiness_waiters.pending_count() != 0
        })
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
                retained: self.task.custody.lock().take(),
                telemetry_degradations: Vec::new(),
                journal_deliveries: Vec::new(),
                optional_observation_failures: 0,
                cleanup_unproved: true,
            }),
        };
        self.task.custody.lock().take();
        *join = JoinOwner::Joined(Arc::clone(&report));
        report
    }
}

#[derive(Default)]
pub(super) struct ManagedDriver {
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
        let handle = runtime.spawn(async move { driver.run(task_inner, run_iterations).await });
        let task = Arc::new(DriverTask {
            join: tokio::sync::Mutex::new(JoinOwner::Task(handle)),
            custody: Mutex::new(Some(Arc::clone(inner))),
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
        errors.extend(cleanup.errors);
        let has_owners = !inner.sequences.is_empty()
            || inner.execution_closes.unreclaimed_count() != 0
            || inner.execution_readiness_waiters.pending_count() != 0;
        let report = Arc::new(DriverExitReport {
            // An evidence error does not mean GPU work is still live. A panic
            // during cleanup, however, cannot be cleared by empty counters.
            retained: (cleanup_unproved || has_owners).then(|| Arc::clone(&inner)),
            errors,
            telemetry_degradations: cleanup.telemetry_degradations,
            journal_deliveries: cleanup.journal_deliveries,
            optional_observation_failures: cleanup.optional_observation_failures,
            cleanup_unproved,
        });
        self.mark_finished();
        exit.armed = false;
        // The completed report now owns any unresolved custody. Keep the
        // fallback only for exits without a report; an observer that never
        // joins must not retain a clean EngineInner through the task handle.
        // No await follows this transfer.
        if let DriverState::Finished(task) = &*self.state.lock() {
            task.custody.lock().take();
        }
        report
    }

    async fn run_iterations(&self, inner: &EngineInner) {
        info!("Background iteration loop started");
        let prof = inner.runtime_config.batch_decode_prof;
        let mut last_iter_end: Option<Instant> = None;
        static GAP_PROF_CALLS: AtomicU64 = AtomicU64::new(0);
        while self.is_running() {
            #[cfg(test)]
            if self.panic_next_iteration.swap(false, Ordering::AcqRel) {
                panic!("injected managed driver iteration panic");
            }
            let inter_iter_us = last_iter_end.map(|previous| previous.elapsed().as_micros() as u64);
            let outcome = match inner.run_iteration().await {
                Ok(outcome) => outcome,
                Err(error) => {
                    warn!("Iteration error: {}", error);
                    EngineIterationOutcome::Progressed
                }
            };
            if prof {
                let count = GAP_PROF_CALLS.fetch_add(1, Ordering::Relaxed);
                if count.is_multiple_of(8) {
                    if let Some(gap_us) = inter_iter_us {
                        eprintln!("[bg-loop-gap] call#{} inter_iter={}us", count, gap_us);
                    }
                }
            }
            last_iter_end = Some(Instant::now());
            if !self.is_running() {
                break;
            }
            match outcome {
                EngineIterationOutcome::Progressed => tokio::task::yield_now().await,
                EngineIterationOutcome::Idle => {
                    tokio::select! {
                        _ = inner.wait_for_prefix_deadline() => {}
                        _ = inner.shutdown_notify.notified() => {}
                        _ = inner.work_notify.notified() => {}
                    }
                }
                EngineIterationOutcome::CapacityBlocked(registration) => {
                    tokio::select! {
                        _ = inner.wait_for_prefix_deadline() => {}
                        _ = inner.shutdown_notify.notified() => {}
                        _ = inner.work_notify.notified() => {}
                        result = registration.wait_for_change() => {
                            if let Err(error) = result {
                                warn!("Executor capacity wait error: {}", error);
                            }
                        }
                    }
                }
            }
        }
        info!("Background iteration loop stopped");
    }
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

enum ClosingJournal {
    Profile(SchedulerTraceJournal),
    Legacy(JsonlJournal<serde_json::Value>),
}

impl ClosingJournal {
    fn path(&self) -> &Path {
        match self {
            Self::Profile(journal) => journal.path(),
            Self::Legacy(journal) => journal.path(),
        }
    }

    fn close(&self) -> std::result::Result<(), JsonlJournalError> {
        match self {
            Self::Profile(journal) => journal.close(),
            Self::Legacy(journal) => journal.close(),
        }
    }

    fn stats(&self) -> ferrum_bench_core::jsonl_journal::JsonlJournalDeliveryStats {
        match self {
            Self::Profile(journal) => journal.delivery_stats(),
            Self::Legacy(journal) => journal.delivery_stats(),
        }
    }
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
        if unreclaimed > 0 || !self.sequences.is_empty() {
            report.errors.push(FerrumError::internal(format!(
                "engine shutdown retains {unreclaimed} unproved execution closures and {} requests",
                self.sequences.len()
            )));
        }
        let (required, optional, optional_failures) = self.observation_health.snapshot();
        report.errors.extend(required);
        report.telemetry_degradations.extend(optional);
        report.optional_observation_failures = optional_failures;
        let mut journals = Vec::with_capacity(3);
        for journal in [
            self.profile_trace_jsonl.clone(),
            self.scheduler_trace_jsonl.clone(),
        ]
        .into_iter()
        .flatten()
        {
            if journals
                .iter()
                .all(|existing: &ClosingJournal| existing.path() != journal.path())
            {
                journals.push(ClosingJournal::Profile(journal));
            }
        }
        if let Some(journal) = self.legacy_scheduler_trace_jsonl.clone() {
            if journals
                .iter()
                .all(|existing| existing.path() != journal.path())
            {
                journals.push(ClosingJournal::Legacy(journal));
            }
        }
        let policy = observation_delivery_policy(self.config.runtime.profile_detail);
        if !journals.is_empty() {
            match tokio::task::spawn_blocking(move || {
                let mut results = Vec::with_capacity(journals.len());
                for journal in journals {
                    // A bad diagnostic destination must not skip the others.
                    let close = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| journal.close()));
                    let stats = journal.stats();
                    let incomplete = stats.dropped != 0 || stats.failed != 0
                        || stats.flushed != stats.accepted || stats.flush_pending
                        || stats.pending_payload_bytes != 0 || !stats.worker_finished
                        || stats.failure_kind.is_some();
                    let error = match close {
                        Ok(Err(error)) => Some(journal_observation_error(error)),
                        Err(_) => Some(ExecutionEventSinkError::contract("execution journal close panicked")),
                        Ok(Ok(())) if incomplete => Some(ExecutionEventSinkError::transport(format!(
                            "execution journal {} is incomplete: accepted={}, dropped={}, failed={}, flushed={}",
                            journal.path().display(), stats.accepted, stats.dropped, stats.failed, stats.flushed,
                        ))),
                        Ok(Ok(())) => None,
                    };
                    results.push((DriverJournalDelivery { path: journal.path().to_path_buf(), stats }, error));
                }
                results
            }).await {
                Ok(results) => for (delivery, error) in results {
                    report.journal_deliveries.push(delivery);
                    if let Some(error) = error {
                        let failure = FerrumError::internal(format!("execution observation close failed: {error}"));
                        if error.is_optional_transport(policy) {
                            // One managed close and one result per deduplicated
                            // destination: visible even to callers reading only result().
                            warn!(error = %error, "optional execution journal closed with incomplete telemetry");
                            report.telemetry_degradations.push(failure);
                        } else {
                            report.errors.push(failure);
                        }
                    }
                },
                Err(error) => {
                    let failure = FerrumError::internal(format!("execution journal close task failed: {error}"));
                    // A task unwind is an unknown internal failure, not I/O.
                    report.errors.push(failure);
                }
            }
        }
        report
    }
}
