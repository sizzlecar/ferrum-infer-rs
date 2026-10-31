use super::*;
use std::sync::{atomic::AtomicUsize, mpsc};

struct Fixture {
    calls: Arc<Mutex<Vec<Vec<u64>>>>,
    entered: Option<mpsc::Sender<usize>>,
    release: Option<mpsc::Receiver<()>>,
    fail: Option<Arc<AtomicUsize>>,
    rejection: Option<String>,
    panic: Option<Arc<AtomicUsize>>,
    wrong_count: bool,
    dropped: Arc<AtomicUsize>,
    shutdown_calls: Arc<AtomicUsize>,
    shutdown_failure: Option<Arc<AtomicUsize>>,
}
impl BatchWorker for Fixture {
    type Output = u64;
    fn shutdown(&mut self) -> Result<(), GeneratorError> {
        self.shutdown_calls.fetch_add(1, Ordering::SeqCst);
        match self.shutdown_failure.take() {
            Some(owner) => Err(Box::new(OriginalFailure(owner))),
            None => Ok(()),
        }
    }
    fn compatible(&self, a: &ImageGenerationRequest, b: &ImageGenerationRequest) -> bool {
        // A typed-generator-specific key may further partition same-size jobs.
        a.prompt.starts_with('a') == b.prompt.starts_with('a')
    }
    fn generate_batch(
        &mut self,
        requests: Vec<ImageGenerationRequest>,
    ) -> Result<Vec<u64>, ImageBatchError> {
        let seeds = requests.iter().map(|r| r.seed).collect::<Vec<_>>();
        self.calls.lock().unwrap().push(seeds.clone());
        if let Some(entered) = &self.entered {
            entered.send(requests.len()).unwrap();
        }
        if let Some(release) = &self.release {
            release.recv().unwrap();
        }
        if let Some(dropped) = self.panic.take() {
            std::panic::panic_any(OriginalFailure(dropped));
        }
        if let Some(dropped) = self.fail.take() {
            return Err(ImageBatchError::Fatal(Box::new(OriginalFailure(dropped))));
        }
        if let Some(message) = self.rejection.take() {
            return Err(ImageBatchError::Rejected(message));
        }
        if self.wrong_count {
            return Ok(Vec::new());
        }
        Ok(seeds)
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        self.dropped.fetch_add(1, Ordering::SeqCst);
    }
}

#[derive(Debug)]
struct OriginalFailure(Arc<AtomicUsize>);
impl fmt::Display for OriginalFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("typed test failure")
    }
}
impl Error for OriginalFailure {}
impl Drop for OriginalFailure {
    fn drop(&mut self) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}

fn fixture() -> Fixture {
    Fixture {
        calls: Arc::new(Mutex::new(Vec::new())),
        entered: None,
        release: None,
        fail: None,
        rejection: None,
        panic: None,
        wrong_count: false,
        dropped: Arc::new(AtomicUsize::new(0)),
        shutdown_calls: Arc::new(AtomicUsize::new(0)),
        shutdown_failure: None,
    }
}
fn request(seed: u64) -> ImageGenerationRequest {
    ImageGenerationRequest {
        prompt: "a prompt".into(),
        width: 32,
        height: 32,
        seed,
        n: 1,
    }
}
fn config(batch: usize, wait: u64, pending: usize) -> ImageServiceConfig {
    ImageServiceConfig {
        batching: ferrum_types::ImageBatchPolicy::Configured,
        maximum_pending: pending,
        maximum_pending_bytes: 1024 * 1024,
        maximum_prompt_bytes: 1024,
        max_num_seqs: batch,
        request_batch_max_wait_ms: wait,
    }
}
fn caps(batch: usize) -> ImageGeneratorCapabilities {
    ImageGeneratorCapabilities {
        maximum_batch_size: batch,
        maximum_width: 1024,
        maximum_height: 1024,
        dimension_multiple: 32,
    }
}
fn start(fixture: Fixture, config: ImageServiceConfig) -> ServiceCore<u64> {
    validate_config(config, caps(config.max_num_seqs)).unwrap();
    ServiceCore::start(Box::new(fixture), config).unwrap()
}
fn release_retirement(mut retired: Retirement<u64>) {
    // Explicitly take original owners, as production shutdown.into_parts does.
    drop(retired.failure.take());
    drop(retired.worker.take());
}

#[tokio::test]
async fn shutdown_failure_preserves_original_error_and_worker_owner() {
    let mut f = fixture();
    let error_dropped = Arc::new(AtomicUsize::new(0));
    f.shutdown_failure = Some(error_dropped.clone());
    let worker_dropped = f.dropped.clone();
    let calls = f.shutdown_calls.clone();
    let service = start(f, config(1, 0, 1));
    assert_eq!(
        service
            .try_submit(request(7))
            .unwrap()
            .wait()
            .await
            .unwrap(),
        7
    );
    let retired = service.shutdown().await.unwrap();
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    let Some(ImageWorkerFailure::Generation(error)) = &retired.failure else {
        panic!("original shutdown failure")
    };
    assert!(error.downcast_ref::<OriginalFailure>().is_some());
    assert_eq!(error_dropped.load(Ordering::SeqCst), 0);
    assert_eq!(worker_dropped.load(Ordering::SeqCst), 0);
    release_retirement(retired);
    assert_eq!(error_dropped.load(Ordering::SeqCst), 1);
    assert_eq!(worker_dropped.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn fatal_generation_does_not_invoke_shutdown_on_unrecovered_native_owners() {
    let mut f = fixture();
    let owner = Arc::new(AtomicUsize::new(0));
    f.fail = Some(owner.clone());
    let calls = f.shutdown_calls.clone();
    let service = start(f, config(1, 0, 1));
    assert!(service
        .try_submit(request(3))
        .unwrap()
        .wait()
        .await
        .is_err());
    let retired = service.shutdown().await.unwrap();
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    assert_eq!(owner.load(Ordering::SeqCst), 0);
    let Some(ImageWorkerFailure::Generation(error)) = &retired.failure else {
        panic!("original execution failure")
    };
    assert!(error.downcast_ref::<OriginalFailure>().is_some());
    release_retirement(retired);
    assert_eq!(owner.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn compatible_requests_are_one_actual_batch_with_ordered_independent_results() {
    let f = fixture();
    let calls = f.calls.clone();
    let dropped = f.dropped.clone();
    let shutdown_calls = f.shutdown_calls.clone();
    let service = start(f, config(2, 30_000, 2));
    let first = service.try_submit(request(11)).unwrap();
    let second = service.try_submit(request(29)).unwrap();
    assert_eq!(first.wait().await.unwrap(), 11);
    assert_eq!(second.wait().await.unwrap(), 29);
    assert_eq!(*calls.lock().unwrap(), vec![vec![11, 29]]);
    let retired = service.shutdown().await.unwrap();
    assert!(retired.failure.is_none());
    assert_eq!(shutdown_calls.load(Ordering::SeqCst), 1);
    drop(retired);
    assert_eq!(dropped.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn fifo_shape_and_generator_compatibility_partition_batches() {
    let f = fixture();
    let calls = f.calls.clone();
    let service = start(f, config(3, 1, 3));
    let first = service.try_submit(request(1)).unwrap();
    let mut second_request = request(2);
    second_request.width = 64;
    let second = service.try_submit(second_request).unwrap();
    let mut third_request = request(3);
    third_request.width = 64;
    third_request.prompt = "different generator key".into();
    let third = service.try_submit(third_request).unwrap();
    assert_eq!(first.wait().await.unwrap(), 1);
    assert_eq!(second.wait().await.unwrap(), 2);
    assert_eq!(third.wait().await.unwrap(), 3);
    assert_eq!(*calls.lock().unwrap(), vec![vec![1], vec![2], vec![3]]);
    drop(service.shutdown().await.unwrap());
}

#[tokio::test]
async fn partial_batch_wait_deadline_dispatches_without_requiring_a_full_batch() {
    let f = fixture();
    let calls = f.calls.clone();
    let service = start(f, config(4, 1, 4));
    assert_eq!(
        service
            .try_submit(request(7))
            .unwrap()
            .wait()
            .await
            .unwrap(),
        7
    );
    assert_eq!(*calls.lock().unwrap(), vec![vec![7]]);
    drop(service.shutdown().await.unwrap());
}

#[tokio::test]
async fn owner_free_rejection_preserves_queued_and_later_batches() {
    let (entered_tx, entered_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel();
    let mut f = fixture();
    f.entered = Some(entered_tx);
    f.release = Some(release_rx);
    f.rejection = Some("request cannot be prepared".into());
    let calls = f.calls.clone();
    let dropped = f.dropped.clone();
    let service = start(f, config(2, 30_000, 2));
    let first = service.try_submit(request(1)).unwrap();
    let second = service.try_submit(request(2)).unwrap();
    assert_eq!(entered_rx.recv().unwrap(), 2);
    let queued_first = service.try_submit(request(3)).unwrap();
    let queued_second = service.try_submit(request(4)).unwrap();
    release_tx.send(()).unwrap();
    assert_eq!(
        first.wait().await.unwrap_err().kind(),
        ImageServiceErrorKind::Generation
    );
    assert_eq!(
        second.wait().await.unwrap_err().kind(),
        ImageServiceErrorKind::Generation
    );
    assert!(service.snapshot().accepting);
    assert_eq!(entered_rx.recv().unwrap(), 2);
    release_tx.send(()).unwrap();
    assert_eq!(queued_first.wait().await.unwrap(), 3);
    assert_eq!(queued_second.wait().await.unwrap(), 4);
    let later_first = service.try_submit(request(5)).unwrap();
    let later_second = service.try_submit(request(6)).unwrap();
    assert_eq!(entered_rx.recv().unwrap(), 2);
    release_tx.send(()).unwrap();
    assert_eq!(later_first.wait().await.unwrap(), 5);
    assert_eq!(later_second.wait().await.unwrap(), 6);
    assert_eq!(
        *calls.lock().unwrap(),
        vec![vec![1, 2], vec![3, 4], vec![5, 6]]
    );
    let retired = service.shutdown().await.unwrap();
    assert!(retired.failure.is_none());
    drop(retired);
    assert_eq!(dropped.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn rejected_active_batch_does_not_undo_cancellation_or_shutdown() {
    let (entered_tx, entered_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel();
    let mut f = fixture();
    f.entered = Some(entered_tx);
    f.release = Some(release_rx);
    f.rejection = Some("owner-free preflight failure".into());
    let calls = f.calls.clone();
    let service = start(f, config(1, 0, 1));
    let active = service.try_submit(request(1)).unwrap();
    assert_eq!(entered_rx.recv().unwrap(), 1);
    let pending = service.try_submit(request(2)).unwrap();
    active.cancel();
    service.shared.stop();
    release_tx.send(()).unwrap();
    assert_eq!(
        active.wait().await.unwrap_err().kind(),
        ImageServiceErrorKind::Cancelled
    );
    assert_eq!(
        pending.wait().await.unwrap_err().kind(),
        ImageServiceErrorKind::Shutdown
    );
    assert!(!service.snapshot().accepting);
    assert!(
        matches!(service.try_submit(request(3)), Err(e) if e.kind() == ImageServiceErrorKind::Stopped)
    );
    assert_eq!(*calls.lock().unwrap(), vec![vec![1]]);
    let retired = service.shutdown().await.unwrap();
    assert!(retired.failure.is_none());
    drop(retired);
}

#[test]
fn unclassified_errors_are_fatal_and_stopped_preflight_is_not_recoverable() {
    use super::super::{ImageGenerationError, ImageGenerationFailure};
    use ferrum_kernels::backend::reference::ReferenceDeviceRuntime;

    let dropped = Arc::new(AtomicUsize::new(0));
    let source: GeneratorError = Box::new(OriginalFailure(dropped.clone()));
    let error = ImageBatchError::from(source);
    assert!(matches!(&error, ImageBatchError::Fatal(source) if source.is::<OriginalFailure>()));
    assert_eq!(dropped.load(Ordering::SeqCst), 0);
    drop(error);
    assert_eq!(dropped.load(Ordering::SeqCst), 1);

    let preflight = || ImageGenerationError::<ReferenceDeviceRuntime> {
        failure: ImageGenerationFailure::Preflight(anyhow::anyhow!("preflight")),
        completed_stages: Vec::new(),
        completed_steps: Vec::new(),
    };
    assert!(matches!(
        qwen_batch_error(false, preflight()),
        ImageBatchError::Rejected(_)
    ));
    let error = qwen_batch_error(true, preflight());
    assert!(
        matches!(&error, ImageBatchError::Fatal(source) if source.is::<ImageGenerationError<ReferenceDeviceRuntime>>())
    );
    let mut with_receipt = preflight();
    with_receipt
        .completed_steps
        .push(super::super::ImageDenoiseStepReceipt {
            index: 0,
            sigma: 1.0,
            next_sigma: 0.5,
            scheduler_timestep: 1000.0,
        });
    assert!(matches!(
        qwen_batch_error(false, with_receipt),
        ImageBatchError::Fatal(_)
    ));
    let unavailable = ImageGenerationError::<ReferenceDeviceRuntime> {
        failure: ImageGenerationFailure::Unavailable,
        completed_stages: Vec::new(),
        completed_steps: Vec::new(),
    };
    assert!(matches!(
        qwen_batch_error(false, unavailable),
        ImageBatchError::Fatal(_)
    ));
}

#[test]
fn rolled_back_allocation_rejection_does_not_stop_a_healthy_worker() {
    use super::super::{ImageGenerationError, ImageGenerationFailure, ImageStage};
    use ferrum_kernels::backend::reference::ReferenceDeviceRuntime;
    let failure = || ImageGenerationError::<ReferenceDeviceRuntime> {
        failure: ImageGenerationFailure::AllocationRejected {
            stage: ImageStage::TextEncoder,
            failure: ferrum_runtime::FailureEnvelope::new(
                ferrum_runtime::FailureDomain::Resource,
                "device_allocation",
                "allocation unavailable",
                true,
            )
            .unwrap(),
        },
        completed_stages: Vec::new(),
        completed_steps: Vec::new(),
    };
    assert!(matches!(
        qwen_batch_error(false, failure()),
        ImageBatchError::Rejected(_)
    ));
    assert!(matches!(
        qwen_batch_error(true, failure()),
        ImageBatchError::Fatal(_)
    ));
}

#[tokio::test]
async fn pending_capacity_cancel_and_active_cancel_preserve_worker_until_completion() {
    let (entered_tx, entered_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel();
    let mut f = fixture();
    f.entered = Some(entered_tx);
    f.release = Some(release_rx);
    let dropped = f.dropped.clone();
    let calls = f.calls.clone();
    let service = start(f, config(1, 0, 1));
    let active = service.try_submit(request(1)).unwrap();
    assert_eq!(entered_rx.recv().unwrap(), 1);
    let queued = service.try_submit(request(2)).unwrap();
    assert!(
        matches!(service.try_submit(request(3)), Err(error) if error.kind() == ImageServiceErrorKind::QueueFull)
    );
    queued.cancel();
    let replacement = service.try_submit(request(3)).unwrap();
    assert_eq!(
        queued.wait().await.unwrap_err().kind(),
        ImageServiceErrorKind::Cancelled
    );
    active.cancel();
    assert_eq!(service.snapshot().active_requests, 1);
    assert_eq!(dropped.load(Ordering::SeqCst), 0);
    // Shutdown removes only pending work and cannot interrupt the active call.
    service.shared.stop();
    release_tx.send(()).unwrap();
    assert_eq!(
        active.wait().await.unwrap_err().kind(),
        ImageServiceErrorKind::Cancelled
    );
    assert_eq!(
        replacement.wait().await.unwrap_err().kind(),
        ImageServiceErrorKind::Shutdown
    );
    assert_eq!(*calls.lock().unwrap(), vec![vec![1]]);
    drop(service.shutdown().await.unwrap());
    assert_eq!(dropped.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn original_failure_stops_admission_and_survives_join_until_explicit_recovery() {
    let (entered_tx, entered_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel();
    let original_dropped = Arc::new(AtomicUsize::new(0));
    let mut f = fixture();
    f.entered = Some(entered_tx);
    f.release = Some(release_rx);
    f.fail = Some(original_dropped.clone());
    let generator_dropped = f.dropped.clone();
    let service = start(f, config(1, 0, 1));
    let active = service.try_submit(request(1)).unwrap();
    entered_rx.recv().unwrap();
    let pending = service.try_submit(request(2)).unwrap();
    release_tx.send(()).unwrap();
    assert_eq!(
        active.wait().await.unwrap_err().kind(),
        ImageServiceErrorKind::Generation
    );
    assert_eq!(
        pending.wait().await.unwrap_err().kind(),
        ImageServiceErrorKind::Stopped
    );
    assert!(!service.snapshot().accepting);
    assert!(
        matches!(service.try_submit(request(3)), Err(e) if e.kind() == ImageServiceErrorKind::Stopped)
    );
    let retired = service.shutdown().await.unwrap();
    match retired.failure.as_ref().unwrap() {
        ImageWorkerFailure::Generation(source) => {
            assert!(source.downcast_ref::<OriginalFailure>().is_some())
        }
        _ => panic!("original typed error must remain available"),
    }
    assert_eq!(original_dropped.load(Ordering::SeqCst), 0);
    assert_eq!(generator_dropped.load(Ordering::SeqCst), 0);
    release_retirement(retired);
    assert_eq!(original_dropped.load(Ordering::SeqCst), 1);
    assert_eq!(generator_dropped.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn panic_payload_and_generator_are_retained_and_output_count_mismatch_fails_closed() {
    let original_dropped = Arc::new(AtomicUsize::new(0));
    let mut f = fixture();
    f.panic = Some(original_dropped.clone());
    let generator_dropped = f.dropped.clone();
    let service = start(f, config(1, 0, 1));
    assert_eq!(
        service
            .try_submit(request(1))
            .unwrap()
            .wait()
            .await
            .unwrap_err()
            .kind(),
        ImageServiceErrorKind::WorkerPanic
    );
    let retired = service.shutdown().await.unwrap();
    assert!(
        matches!(retired.failure.as_ref(), Some(ImageWorkerFailure::Panic(payload)) if payload.is::<OriginalFailure>())
    );
    assert_eq!(original_dropped.load(Ordering::SeqCst), 0);
    release_retirement(retired);
    assert_eq!(original_dropped.load(Ordering::SeqCst), 1);
    assert_eq!(generator_dropped.load(Ordering::SeqCst), 1);

    let mut f = fixture();
    f.wrong_count = true;
    let service = start(f, config(1, 0, 1));
    assert_eq!(
        service
            .try_submit(request(1))
            .unwrap()
            .wait()
            .await
            .unwrap_err()
            .kind(),
        ImageServiceErrorKind::Generation
    );
    assert!(!service.snapshot().accepting);
    release_retirement(service.shutdown().await.unwrap());
}

#[test]
fn explicit_capacity_and_batch_capability_are_checked_before_worker_creation() {
    validate_config(config(1, 0, 1), caps(1)).unwrap();
    assert!(validate_config(config(8, 200, 8), caps(1)).is_err());
    assert!(validate_config(config(0, 0, 1), caps(8)).is_err());
    assert!(validate_config(config(1, 0, 0), caps(1)).is_err());
    let mut large = config(1, 0, usize::MAX);
    assert!(validate_config(large, caps(1)).is_err());
    large = config(1, 0, 1);
    large.maximum_prompt_bytes = usize::MAX;
    assert!(validate_config(large, caps(1)).is_err());
    large.maximum_prompt_bytes = 0;
    assert!(validate_config(large, caps(1)).is_err());
    let mut invalid_caps = caps(1);
    invalid_caps.dimension_multiple = 0;
    assert!(validate_config(config(1, 0, 1), invalid_caps).is_err());
}

#[derive(Clone)]
struct PipelinedFixture {
    planning: bool,
    prepared: Arc<(Mutex<usize>, Condvar)>,
    active: Arc<AtomicUsize>,
    overlap: Arc<AtomicBool>,
    entered: mpsc::Sender<()>,
    release: Option<Arc<Mutex<mpsc::Receiver<()>>>>,
    failures: Option<(Arc<AtomicUsize>, Arc<AtomicUsize>)>,
}
impl BatchWorker for PipelinedFixture {
    type Output = u64;
    fn preparation_worker(&self) -> Option<Box<dyn BatchWorker<Output = u64>>> {
        Some(Box::new(Self {
            planning: true,
            ..self.clone()
        }))
    }
    fn compatible(&self, _: &ImageGenerationRequest, _: &ImageGenerationRequest) -> bool {
        true
    }
    fn prepare(
        &mut self,
        _: &mut [&mut QueuedRequest],
    ) -> Result<ImageBatchPreparation, ImageBatchError> {
        let (lock, changed) = &*self.prepared;
        let mut n = lock.lock().unwrap();
        *n += 1;
        if self.planning {
            self.overlap
                .store(self.active.load(Ordering::Acquire) == 1, Ordering::Release);
        }
        changed.notify_all();
        drop(n);
        if self.planning {
            if let Some(release) = &self.release {
                let _ = release.lock().unwrap().recv();
            }
            if let Some((_, owner)) = &self.failures {
                return Err(ImageBatchError::Fatal(Box::new(OriginalFailure(
                    owner.clone(),
                ))));
            }
        }
        Ok(ImageBatchPreparation::default())
    }
    fn try_admit_prepared(
        &mut self,
        _: &mut ImageBatchPreparation,
        _: ImagePreparationResources,
    ) -> Result<bool, ImageBatchError> {
        assert!(!self.planning, "preparer cannot admit execution resources");
        Ok(true)
    }
    fn generate_batch(
        &mut self,
        requests: Vec<ImageGenerationRequest>,
    ) -> Result<Vec<u64>, ImageBatchError> {
        assert!(!self.planning, "preparer cannot execute");
        assert_eq!(self.active.fetch_add(1, Ordering::AcqRel), 0);
        if requests[0].seed == 0 {
            self.entered.send(()).unwrap();
            let (lock, changed) = &*self.prepared;
            let (n, _) = changed
                .wait_timeout_while(lock.lock().unwrap(), Duration::from_secs(3), |n| *n < 2)
                .unwrap();
            assert!(
                *n >= 2,
                "next plan was not prepared during active execution"
            );
        }
        self.active.fetch_sub(1, Ordering::AcqRel);
        if let Some((owner, _)) = &self.failures {
            return Err(ImageBatchError::Fatal(Box::new(OriginalFailure(
                owner.clone(),
            ))));
        }
        Ok(requests.iter().map(|r| r.seed).collect())
    }
}

#[tokio::test]
async fn next_plan_overlaps_one_execution_and_does_not_delay_completed_response() {
    let (entered, rx) = mpsc::channel();
    let (release, pending) = mpsc::channel();
    let overlap = Arc::new(AtomicBool::new(false));
    let service = ServiceCore::start(
        Box::new(PipelinedFixture {
            planning: false,
            prepared: Arc::new((Mutex::new(0), Condvar::new())),
            active: Arc::new(AtomicUsize::new(0)),
            overlap: overlap.clone(),
            entered,
            release: Some(Arc::new(Mutex::new(pending))),
            failures: None,
        }),
        ImageServiceConfig {
            batching: ImageBatchPolicy::Auto,
            ..config(1, 0, 4)
        },
    )
    .unwrap();
    let first = service.try_submit(request(0)).unwrap();
    rx.recv_timeout(Duration::from_secs(2)).unwrap();
    let second = service.try_submit(request(1)).unwrap();
    assert_eq!(
        tokio::time::timeout(Duration::from_secs(2), first.wait())
            .await
            .unwrap()
            .unwrap(),
        0
    );
    assert!(overlap.load(Ordering::Acquire));
    // Compilation of the next batch is still blocked until after this reply.
    release.send(()).unwrap();
    assert_eq!(
        tokio::time::timeout(Duration::from_secs(2), second.wait())
            .await
            .unwrap()
            .unwrap(),
        1
    );
    let retired = service.shutdown().await.unwrap();
    assert!(retired.failure.is_none());
    release_retirement(retired);
}

#[tokio::test]
async fn simultaneous_execution_and_preparation_failure_retain_both_original_owners() {
    let (entered, rx) = mpsc::channel();
    let execution_owner = Arc::new(AtomicUsize::new(0));
    let preparation_owner = Arc::new(AtomicUsize::new(0));
    let service = ServiceCore::start(
        Box::new(PipelinedFixture {
            planning: false,
            prepared: Arc::new((Mutex::new(0), Condvar::new())),
            active: Arc::new(AtomicUsize::new(0)),
            overlap: Arc::new(AtomicBool::new(false)),
            entered,
            release: None,
            failures: Some((execution_owner.clone(), preparation_owner.clone())),
        }),
        ImageServiceConfig {
            batching: ImageBatchPolicy::Auto,
            ..config(1, 0, 4)
        },
    )
    .unwrap();
    let first = service.try_submit(request(0)).unwrap();
    rx.recv_timeout(Duration::from_secs(2)).unwrap();
    let second = service.try_submit(request(1)).unwrap();
    assert!(first.wait().await.is_err());
    assert!(second.wait().await.is_err());
    let retired = service.shutdown().await.unwrap();
    assert!(matches!(
        retired.failure,
        Some(ImageWorkerFailure::Concurrent { .. })
    ));
    assert_eq!(execution_owner.load(Ordering::Acquire), 0);
    assert_eq!(preparation_owner.load(Ordering::Acquire), 0);
    release_retirement(retired);
    assert_eq!(execution_owner.load(Ordering::Acquire), 1);
    assert_eq!(preparation_owner.load(Ordering::Acquire), 1);
}

#[tokio::test]
async fn cancelling_a_prepared_future_batch_never_executes_it() {
    let (entered, rx) = mpsc::channel();
    let (release, pending) = mpsc::channel();
    let service = ServiceCore::start(
        Box::new(PipelinedFixture {
            planning: false,
            prepared: Arc::new((Mutex::new(0), Condvar::new())),
            active: Arc::new(AtomicUsize::new(0)),
            overlap: Arc::new(AtomicBool::new(false)),
            entered,
            release: Some(Arc::new(Mutex::new(pending))),
            failures: None,
        }),
        ImageServiceConfig {
            batching: ImageBatchPolicy::Auto,
            ..config(1, 0, 4)
        },
    )
    .unwrap();
    let first = service.try_submit(request(0)).unwrap();
    rx.recv_timeout(Duration::from_secs(2)).unwrap();
    let second = service.try_submit(request(1)).unwrap();
    assert_eq!(
        tokio::time::timeout(Duration::from_secs(2), first.wait())
            .await
            .unwrap()
            .unwrap(),
        0
    );
    second.cancel();
    release.send(()).unwrap();
    assert!(second.wait().await.is_err());
    assert_eq!(service.snapshot().executed_requests, 1);
    let retired = service.shutdown().await.unwrap();
    assert!(retired.failure.is_none());
    release_retirement(retired);
}
