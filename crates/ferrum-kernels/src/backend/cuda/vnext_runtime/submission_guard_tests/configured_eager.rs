use super::*;

struct EagerGuard {
    reject: bool,
    calls: AtomicU64,
    enqueues: Arc<AtomicU64>,
    actual: Mutex<Option<DeviceSubmissionAttribution>>,
}

impl EagerGuard {
    fn new(f: &Fixture, reject: bool) -> Self {
        Self {
            reject,
            calls: AtomicU64::new(0),
            enqueues: Arc::clone(&f.enqueues),
            actual: Mutex::new(None),
        }
    }
}

impl PreparedWaveSubmissionGuard for EagerGuard {
    fn submission_mode(&self) -> GuardedSubmissionMode {
        GuardedSubmissionMode::ExactAdaptiveRoute
    }
    fn check(
        &self,
        actual: &DeviceSubmissionAttribution,
        readback: CoreReadbackRoute,
    ) -> Result<(), GuardedNotSubmittedReason> {
        assert_eq!(self.enqueues.load(Ordering::Relaxed), 0);
        assert_eq!(readback, CoreReadbackRoute::SubmissionStaged);
        let proof = actual.graph_evidence().unwrap();
        assert!(proof.proves_configured_eager_observation());
        assert!(!proof.proves_unconfigured_eager());
        assert!(!proof.proves_warm_direct_replay());
        assert!(actual.replayed_segments().is_empty());
        assert!(actual.commands().iter().all(|command| {
            command.execution_path() == DeviceExecutionPath::Eager
                && command.reusable_graph_node_count().is_none()
        }));
        assert_eq!(self.calls.fetch_add(1, Ordering::Relaxed), 0);
        *self.actual.lock().unwrap() = Some(actual.clone());
        if self.reject {
            Err(GuardedNotSubmittedReason::HostRejected(
                HostSubmissionRejection::WitnessExpired,
            ))
        } else {
            Ok(())
        }
    }
}

#[test]
#[ignore = "requires an actual CUDA device and installed native operator artifacts"]
fn guarded_cuda_configured_eager_final_guard_retries_without_preparation() {
    let f = Fixture::new_configured_eager();
    let before_preparation = f.lane.reusable_executable_preparation().unwrap();
    let before_catalog = f.lane.reusable_execution_catalog().unwrap().into_parts();
    let (step, wave) = f.prepare();
    let reject = EagerGuard::new(&f, true);
    let pending = match f.dispatch(wave, &reject) {
        GuardedWaveSubmissionOutcome::NotSubmitted(value) => value,
        _ => panic!("expired host authority must reject before any enqueue"),
    };
    assert_eq!(reject.calls.load(Ordering::Relaxed), 1);
    assert_eq!(f.enqueues.load(Ordering::Relaxed), 0);
    pending
        .reconcile_step(step)
        .unwrap_or_else(|(e, _)| panic!("rollback: {e}"));
    assert_eq!(
        f.lane.reusable_executable_preparation().unwrap(),
        before_preparation
    );
    assert_eq!(
        f.lane.reusable_execution_catalog().unwrap().into_parts(),
        before_catalog
    );
    let (step, wave) = f.prepare();
    let guard = EagerGuard::new(&f, false);
    let handle = accepted(f.dispatch(wave, &guard));
    assert_eq!(f.enqueues.load(Ordering::Relaxed), 1);
    assert_eq!(guard.calls.load(Ordering::Relaxed), 1);
    warm_graph::finish(&f, handle);
    step.try_retire_normal().unwrap();
    assert_eq!(
        f.lane.reusable_executable_preparation().unwrap(),
        before_preparation
    );
    assert_eq!(
        f.lane.reusable_execution_catalog().unwrap().into_parts(),
        before_catalog
    );
    f.close(true);
}

#[test]
#[ignore = "requires an actual CUDA device and installed native operator artifacts"]
fn guarded_cuda_configured_eager_never_promotes_an_actual_capture_candidate() {
    let f = Fixture::new_graph();
    let before_preparation = f.lane.reusable_executable_preparation().unwrap();
    let before_catalog = f.lane.reusable_execution_catalog().unwrap().into_parts();
    let (step, wave) = f.prepare();
    let guard = EagerGuard::new(&f, false);
    let pending = match f.dispatch(wave, &guard) {
        GuardedWaveSubmissionOutcome::NotSubmitted(value) => value,
        _ => panic!("actual capture intent is never a configured eager proof"),
    };
    assert_eq!(guard.calls.load(Ordering::Relaxed), 0);
    assert_eq!(f.enqueues.load(Ordering::Relaxed), 0);
    pending
        .reconcile_step(step)
        .unwrap_or_else(|(e, _)| panic!("rollback: {e}"));
    assert_eq!(
        f.lane.reusable_executable_preparation().unwrap(),
        before_preparation
    );
    assert_eq!(
        f.lane.reusable_execution_catalog().unwrap().into_parts(),
        before_catalog
    );
    f.close(false);
}
