//! Final retirement of the original driver, its worker, and its plan resources.
//! The observer only wakes this service; neither a cancelled wait nor a failed
//! close destroys an owner or reconstructs an executor.

use super::*;
use crate::executor_lifecycle::{
    ExecutorShutdownObserver, ExecutorShutdownPermit, ExecutorShutdownReporter,
};
use parking_lot::Condvar;
use std::panic::{catch_unwind, AssertUnwindSafe};

enum Owner<R: DeviceRuntime> {
    Preparation(Option<ferrum_runtime::ResourcePreparationRetirement<VNextDriver<R>>>),
    Initialization(Option<ferrum_runtime::StaticInitializationFailure<VNextDriver<R>>>),
    Shared(Option<Arc<ProgramExecutor<R>>>),
    Driver(Option<Box<ProgramExecutor<R>>>),
    Resources(Option<Arc<PlanRuntimeResources<R>>>),
    CloseFailure(Option<PlanRuntimeCloseFailure<R>>),
    Retired,
}

struct State<R: DeviceRuntime> {
    owner: Owner<R>,
    error: Option<FerrumError>,
}

#[derive(Default)]
struct Wake {
    requested: bool,
    running: bool,
}

struct Custody<R: DeviceRuntime> {
    state: Mutex<State<R>>,
    wake: Mutex<Wake>,
    changed: Condvar,
    // Only OS-thread-start failure uses this explicit per-instance quarantine.
    // A later successful start transfers custody to the original service and
    // clears this hold; there is no global registry or repeated leaked Arc.
    failed_start_hold: Mutex<Option<Arc<Self>>>,
}

pub(super) fn begin<R: DeviceRuntime>(
    executor: Arc<ProgramExecutor<R>>,
    permit: &ExecutorShutdownPermit,
) -> ExecutorShutdownObserver {
    if let Err(error) = permit.check(executor.as_ref()) {
        return ExecutorShutdownObserver::retained(Arc::new(executor), error);
    }
    let owner = match Arc::try_unwrap(executor) {
        Ok(executor) => Owner::Driver(Some(Box::new(executor))),
        Err(executor) => Owner::Shared(Some(executor)),
    };
    begin_owned(owner, None)
}

pub(super) fn initialization_error<R: DeviceRuntime>(
    mut failure: ferrum_runtime::StaticInitializationFailure<VNextDriver<R>>,
    cause: FerrumError,
) -> crate::ExecutorBuildError {
    if !failure.is_indeterminate() && failure.retire_in_place().is_ok() {
        return cause.into();
    }
    let cleanup = begin_initialization(failure, cause.clone());
    crate::ExecutorBuildError::with_cleanup(cause, cleanup)
}

pub(super) fn begin_initialization<R: DeviceRuntime>(
    failure: ferrum_runtime::StaticInitializationFailure<VNextDriver<R>>,
    cause: FerrumError,
) -> ExecutorShutdownObserver {
    begin_owned(Owner::Initialization(Some(failure)), Some(cause))
}

pub(super) fn preparation_error<R: DeviceRuntime>(
    mut original: ferrum_runtime::ResourcePreparationRetirement<VNextDriver<R>>,
    cause: FerrumError,
) -> crate::ExecutorBuildError {
    if original.retire_in_place().is_ok() {
        return cause.into();
    }
    let cleanup = begin_owned(Owner::Preparation(Some(original)), Some(cause.clone()));
    crate::ExecutorBuildError::with_cleanup(cause, cleanup)
}

fn begin_owned<R: DeviceRuntime>(
    owner: Owner<R>,
    error: Option<FerrumError>,
) -> ExecutorShutdownObserver {
    let custody = Arc::new(Custody {
        state: Mutex::new(State {
            owner,
            error: error.clone(),
        }),
        wake: Mutex::new(Wake::default()),
        changed: Condvar::new(),
        failed_start_hold: Mutex::new(None),
    });
    let (reporter, observer) = ExecutorShutdownObserver::channel_with_error(custody.clone(), error);
    let weak = Arc::downgrade(&custody);
    let retry_reporter = reporter.clone();
    let observer = observer.with_retry(Arc::new(move || {
        if let Some(custody) = weak.upgrade() {
            custody.request(retry_reporter.clone());
        }
    }));
    custody.request(reporter);
    observer
}

impl<R: DeviceRuntime> Custody<R> {
    fn request(self: &Arc<Self>, reporter: ExecutorShutdownReporter) {
        let mut wake = self.wake.lock();
        wake.requested = true;
        if wake.running {
            self.changed.notify_one();
            return;
        }
        if matches!(self.state.lock().owner, Owner::Retired) {
            return;
        }
        wake.running = true;
        let custody = Arc::clone(self);
        let failed_start = reporter.clone();
        let spawned = std::thread::Builder::new()
            .name("ferrum-program-close".into())
            .spawn(move || custody.run(reporter));
        if let Err(error) = spawned {
            wake.running = false;
            // Preserve the one original custody even if the public build error
            // and every observer are dropped before an OS thread can start.
            let mut hold = self.failed_start_hold.lock();
            if hold.is_none() {
                *hold = Some(Arc::clone(self));
            }
            failed_start.retained(FerrumError::backend(format!(
                "could not start program retirement service: {error}",
            )));
        } else {
            // The spawned service now owns the same instance independently.
            self.failed_start_hold.lock().take();
        }
    }

    fn run(self: Arc<Self>, reporter: ExecutorShutdownReporter) {
        loop {
            {
                let mut wake = self.wake.lock();
                while !wake.requested {
                    self.changed.wait(&mut wake);
                }
                wake.requested = false;
            }
            let mut state = self.state.lock();
            match catch_unwind(AssertUnwindSafe(|| state.advance())) {
                Ok(Ok(())) => {
                    reporter.retired(state.error.clone());
                    drop(state);
                    self.wake.lock().running = false;
                    return;
                }
                Ok(Err(error)) => reporter.retained(error),
                Err(_) => reporter.retained(FerrumError::internal(
                    "program retirement panicked; terminal retirement was not proved",
                )),
            }
            // Retained is observable immediately. The original service keeps
            // custody even if every public waiter is dropped, without spinning
            // on an unchanged backend failure or an outstanding external owner.
        }
    }
}

impl<R: DeviceRuntime> State<R> {
    fn advance(&mut self) -> Result<()> {
        if let Owner::Preparation(original) = &mut self.owner {
            let failure = original
                .as_mut()
                .expect("failed preparation retains original owner");
            failure.retire_in_place().map_err(|error| {
                FerrumError::backend(format!(
                    "static preparation retirement: {}",
                    error.message()
                ))
            })?;
            debug_assert!(failure.is_retired());
            self.owner = Owner::Retired;
        }
        if let Owner::Initialization(original) = &mut self.owner {
            let failure = original
                .as_mut()
                .expect("failed initialization retains original owner");
            failure.retire_in_place().map_err(|error| {
                FerrumError::backend(format!(
                    "static initialization retirement: {}",
                    error.message()
                ))
            })?;
            debug_assert!(failure.is_retired());
            self.owner = Owner::Retired;
        }
        if let Owner::Shared(executor) = &mut self.owner {
            let original = executor
                .take()
                .expect("shared shutdown owns its original Arc");
            match Arc::try_unwrap(original) {
                Ok(executor) => self.owner = Owner::Driver(Some(Box::new(executor))),
                Err(original) => {
                    let references = Arc::strong_count(&original);
                    *executor = Some(original);
                    return Err(FerrumError::backend(format!(
                        "program shutdown retains {references} executor references",
                    )));
                }
            }
        }
        if let Owner::Driver(executor) = &mut self.owner {
            let original = executor
                .as_mut()
                .expect("shutdown owns its original driver");
            let sequences = original.sequences.lock().retained_sequences();
            for sequence in sequences {
                sequence.abort();
            }
            let requests = original
                .sequences
                .lock()
                .prefills
                .keys()
                .cloned()
                .collect::<Vec<_>>();
            for request in requests {
                original.sequences.lock().cancel_prefill(&request);
            }
            if let Err(error) = original.completion_worker.stop_and_join() {
                self.error.get_or_insert_with(|| {
                    FerrumError::backend(format!("program completion worker retirement: {error}",))
                });
            }
            loop {
                let before = (
                    original.sequences.lock().total_len(),
                    original.reaper.retained_count(),
                    original.lane.in_flight_count(),
                );
                original
                    .completion_worker
                    .recover_after_join()
                    .map_err(|error| {
                        FerrumError::backend(format!("program retained-sequence recovery: {error}"))
                    })?;
                let sweep = original
                    .reaper
                    .poll_bounded(MAX_COMPLETION_SWEEP_SLOTS)
                    .map_err(|error| FerrumError::backend(error.to_string()))?;
                for entry in sweep.entries() {
                    if let ferrum_runtime::CompletionSweepObservation::Failed(error) =
                        entry.observation()
                    {
                        self.error.get_or_insert_with(|| {
                            FerrumError::backend(format!("program completion sweep: {error}",))
                        });
                    }
                }
                let sequences = original.sequences.lock().total_len();
                let completions = sweep.retained_after();
                let flights = original.lane.in_flight_count();
                if sequences == 0 && completions == 0 && flights == 0 {
                    break;
                }
                // The original worker is joined. Continue a cold bounded sweep
                // only while its actual retained-owner counts strictly decline;
                // an unchanged/quarantined owner requires an explicit retry.
                let progressed = sequences <= before.0
                    && completions <= before.1
                    && flights <= before.2
                    && (sequences < before.0 || completions < before.1 || flights < before.2);
                if !progressed {
                    return Err(FerrumError::backend(format!(
                        "program shutdown retains {sequences} sequences, {completions} completion owners ({} quarantined), and {flights} lane flights",
                        sweep.quarantined_after(),
                    )));
                }
            }
            // The worker has joined and the native receipts prove all active
            // work ended. Drop this driver's prefix/checkpoint, resident/startup
            // catalogs, lane and provider references before closing its plan.
            // Install custody of that same plan before any child destructor.
            let resources = Arc::clone(&original.plan_resources);
            let original = executor.take();
            self.owner = Owner::Resources(Some(resources));
            drop(original);
        }
        if let Owner::Resources(resources) = &mut self.owner {
            let original = resources.as_ref().ok_or_else(|| {
                FerrumError::internal(
                "consuming plan close unwound without a close receipt; retirement remains unproved",
            )
            })?;
            original
                .maintain_deferred_cleanups(MAX_DEFERRED_DEVICE_CLEANUP_MAINTENANCE_TASKS)
                .map_err(|error| FerrumError::backend(error.to_string()))?;
            let original = resources
                .take()
                .expect("plan close consumes original resources once");
            match PlanRuntimeResources::close(original) {
                Ok(PlanRuntimeCloseOutcome::Closed(_)) => self.owner = Owner::Retired,
                Ok(PlanRuntimeCloseOutcome::Referenced {
                    resources: original,
                    strong_count,
                    deferred_cleanup,
                }) => {
                    *resources = Some(original);
                    return Err(FerrumError::backend(format!(
                        "program plan retains {strong_count} references and deferred cleanup {deferred_cleanup:?}",
                    )));
                }
                Err(failure) => {
                    let error = FerrumError::backend(format!(
                        "program plan close: {:?}",
                        failure.failure()
                    ));
                    self.owner = Owner::CloseFailure(Some(failure));
                    return Err(error);
                }
            }
        }
        if let Owner::CloseFailure(failure) = &mut self.owner {
            let original = failure.take().ok_or_else(|| FerrumError::internal(
                "consuming plan close retry unwound without a close receipt; retirement remains unproved",
            ))?;
            match original.retry() {
                Ok(_) => self.owner = Owner::Retired,
                Err(original) => {
                    let error = FerrumError::backend(format!(
                        "program plan close retry: {:?}",
                        original.failure()
                    ));
                    *failure = Some(original);
                    return Err(error);
                }
            }
        }
        if matches!(self.owner, Owner::Retired) {
            Ok(())
        } else {
            Err(FerrumError::internal(
                "program shutdown lost its close authority",
            ))
        }
    }
}

#[cfg(test)]
mod initialization_tests {
    use super::*;
    use crate::{ExecutorBuildError, ExecutorShutdownPhase};
    use ferrum_runtime::test_support::resource_support::failed_static_initialization;
    use std::sync::Barrier;
    use std::time::Duration;

    async fn retained(observer: &mut ExecutorShutdownObserver) {
        tokio::time::timeout(Duration::from_secs(5), async {
            while observer.snapshot().phase() != ExecutorShutdownPhase::Retained {
                observer.changed().await;
            }
        })
        .await
        .expect("original first retirement attempt must be observable");
    }

    #[tokio::test]
    async fn failed_static_initialization_keeps_original_owner_and_cause_across_retry() {
        let (failure, probe) = failed_static_initialization(1);
        let cause = FerrumError::device(failure.failure().message());
        let initial = probe.snapshot();
        let mut observer = begin_initialization(failure, cause.clone());
        let error = ExecutorBuildError::with_cleanup(cause.clone(), observer.clone());
        retained(&mut observer).await;
        assert_eq!(
            observer.snapshot().error().unwrap().to_string(),
            cause.to_string()
        );
        assert_eq!(probe.snapshot().buffer_drops, 0);
        assert!(probe.runtime_alive());
        // Cancel a wait and discard the constructor's error. Neither owns the
        // original transaction; the same service still does.
        let mut cancelled = error.cleanup().unwrap().clone();
        // Consume the already-published Retained revision, then cancel a wait
        // for a new result from the same service.
        cancelled.wait().await;
        let pending = async {
            loop {
                if cancelled.changed().await.phase() == ExecutorShutdownPhase::Retired {
                    break;
                }
            }
        };
        assert!(tokio::time::timeout(Duration::ZERO, pending).await.is_err());
        drop(error);
        assert_eq!(probe.snapshot().synchronizations, 1);
        observer.retry();
        let final_state = tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                let report = observer.snapshot();
                if report.phase() == ExecutorShutdownPhase::Retired {
                    break report;
                }
                observer.changed().await;
            }
        })
        .await
        .unwrap();
        assert_eq!(final_state.phase(), ExecutorShutdownPhase::Retired);
        assert_eq!(final_state.error().unwrap().to_string(), cause.to_string());
        assert!(
            final_state.result().is_err(),
            "retirement must not erase initialization failure"
        );
        probe.wait_for_all_buffer_drops();
        let retired = probe.snapshot();
        assert_eq!(retired.synchronizations, 2);
        assert_eq!(retired.uploads, initial.uploads);
        assert_eq!(retired.submissions, initial.submissions);
        assert!(!probe.runtime_alive());
        observer.retry();
        assert_eq!(
            probe.snapshot().synchronizations,
            2,
            "terminal retry cannot repeat release"
        );
    }

    #[test]
    fn failed_static_initialization_service_survives_all_public_observers() {
        let (failure, probe) = failed_static_initialization(0);
        let entered = Arc::new(Barrier::new(2));
        let release = Arc::new(Barrier::new(2));
        probe.block_next_synchronize(entered.clone(), release.clone());
        let observer = begin_initialization(
            failure,
            FerrumError::device("original initialization failed"),
        );
        entered.wait();
        struct Release(Arc<Barrier>);
        impl Drop for Release {
            fn drop(&mut self) {
                self.0.wait();
            }
        }
        let release = Release(release);
        assert!(probe.runtime_alive());
        assert_eq!(probe.snapshot().buffer_drops, 0);
        drop(observer);
        drop(release);
        probe.wait_for_all_buffer_drops();
        assert_eq!(probe.snapshot().submissions, 1);
        assert_eq!(probe.snapshot().synchronizations, 1);
    }
}
