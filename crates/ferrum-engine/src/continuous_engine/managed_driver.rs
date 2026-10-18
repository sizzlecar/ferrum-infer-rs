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

/// Actual driver and cleanup results. A failed exit retains its original product owners.
pub struct DriverExitReport {
    errors: Vec<FerrumError>,
    retained: Option<Arc<EngineInner>>,
}

impl DriverExitReport {
    pub fn result(&self) -> Result<()> {
        self.errors.first().cloned().map_or(Ok(()), Err)
    }

    pub fn errors(&self) -> &[FerrumError] {
        &self.errors
    }

    pub fn retains_unproved_owners(&self) -> bool {
        self.retained.as_ref().is_some_and(|inner| {
            !inner.sequences.is_empty()
                || inner.execution_closes.unreclaimed_count() != 0
                || inner.execution_readiness_waiters.pending_count() != 0
        })
    }
}

impl std::fmt::Debug for DriverExitReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DriverExitReport")
            .field("errors", &self.errors)
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
        match std::panic::AssertUnwindSafe(inner.shutdown_cleanup())
            .catch_unwind()
            .await
        {
            Ok(cleanup_errors) => errors.extend(cleanup_errors),
            Err(_) => errors.push(FerrumError::internal("managed engine cleanup panicked")),
        }
        let report = Arc::new(DriverExitReport {
            retained: (!errors.is_empty()).then(|| Arc::clone(&inner)),
            errors,
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

impl EngineInner {
    async fn shutdown_cleanup(&self) -> Vec<FerrumError> {
        let mut errors = self.handoff_shutdown_requests();
        if let Err(error) = self.output_tasks.stop_and_join().await {
            errors.push(error);
        }
        if let Err(error) = self.execution_readiness_waiters.abort_and_join().await {
            errors.push(error);
        }
        self.execution_closes.join_observers().await;
        if let Err(error) = self.complete_failed_execution_recomputes().await {
            errors.push(error);
        }
        if let Err(error) = self.drain_execution_closes() {
            errors.push(error);
        }
        let unreclaimed = self.execution_closes.unreclaimed_count();
        if unreclaimed > 0 || !self.sequences.is_empty() {
            errors.push(FerrumError::internal(format!(
                "engine shutdown retains {unreclaimed} unproved execution closures and {} requests",
                self.sequences.len()
            )));
            return errors;
        }
        let mut trace_journals = Vec::with_capacity(2);
        if let Some(journal) = self.profile_trace_jsonl.clone() {
            trace_journals.push(journal);
        }
        if let Some(journal) = self.scheduler_trace_jsonl.clone() {
            if trace_journals
                .iter()
                .all(|existing| existing.path() != journal.path())
            {
                trace_journals.push(journal);
            }
        }
        if !trace_journals.is_empty() {
            match tokio::task::spawn_blocking(move || {
                let mut errors = Vec::new();
                for journal in trace_journals {
                    if let Err(error) = journal.close() {
                        errors.push(FerrumError::internal(format!(
                            "scheduler trace close failed: {error}"
                        )));
                    }
                }
                errors
            })
            .await
            {
                Ok(trace_errors) => errors.extend(trace_errors),
                Err(error) => errors.push(FerrumError::internal(format!(
                    "scheduler trace close task failed: {error}"
                ))),
            }
        }
        errors
    }
}
