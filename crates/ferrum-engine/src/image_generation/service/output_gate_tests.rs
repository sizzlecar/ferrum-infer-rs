use super::*;
use ferrum_runtime::{HostMemoryClassLimits, HostMemoryError, HostMemoryLimits};
use std::{sync::mpsc, task::Waker};

struct OutputGate {
    host: HostMemoryAccount,
    bytes: usize,
    blocked: mpsc::Sender<()>,
}
impl ImageOutputGate for OutputGate {
    fn host_output_bytes(&self, host: &HostMemoryAccount) -> usize {
        if self.host.same_parent(host) {
            self.bytes
        } else {
            0
        }
    }
    fn try_acquire(&self) -> Result<Option<ImageOutputLease>, ImageOutputGateError> {
        match self.host.reserve(HostMemoryClass::Output, self.bytes) {
            Ok(owner) => Ok(Some(ImageOutputLease::new(owner))),
            Err(HostMemoryError::Capacity {
                requested, limit, ..
            }) if requested <= limit => {
                let _ = self.blocked.send(());
                Ok(None)
            }
            Err(error) => Err(ImageOutputGateError::Capacity(error.to_string())),
        }
    }
    fn subscribe_capacity(&self, waker: Waker) -> ImageOutputLease {
        ImageOutputLease::new(self.host.subscribe_capacity(waker))
    }
}

struct Worker {
    entered: mpsc::Sender<Vec<u64>>,
    release: mpsc::Receiver<()>,
}
impl BatchWorker for Worker {
    type Output = u64;
    fn compatible(&self, _: &ImageGenerationRequest, _: &ImageGenerationRequest) -> bool {
        true
    }
    fn generate_batch(
        &mut self,
        requests: Vec<ImageGenerationRequest>,
    ) -> Result<Vec<u64>, ImageBatchError> {
        let seeds: Vec<_> = requests.into_iter().map(|request| request.seed).collect();
        self.entered.send(seeds.clone()).unwrap();
        self.release.recv().unwrap();
        Ok(seeds)
    }
}
fn request(seed: u64) -> ImageGenerationRequest {
    ImageGenerationRequest {
        prompt: "image".into(),
        width: 32,
        height: 32,
        seed,
        n: 1,
    }
}
fn config() -> ImageServiceConfig {
    ImageServiceConfig {
        batching: ImageBatchPolicy::Auto,
        maximum_pending: 4,
        maximum_pending_bytes: 16 * 1024,
        maximum_prompt_bytes: 1024,
        max_num_seqs: 4,
        request_batch_max_wait_ms: 0,
    }
}
fn host() -> HostMemoryAccount {
    HostMemoryAccount::new(
        HostMemoryLimits::new(
            128,
            HostMemoryClassLimits {
                input_preprocess_bytes: 0,
                transfer_staging_bytes: 0,
                output_bytes: 64,
                weight_backing_bytes: 0,
            },
        )
        .unwrap(),
    )
}
fn receive<T>(channel: &mpsc::Receiver<T>) -> T {
    channel
        .recv_timeout(Duration::from_secs(10))
        .expect("worker made progress")
}
fn submit(
    service: &ServiceCore<u64>,
    seed: u64,
    gate: Arc<dyn ImageOutputGate>,
) -> (Job<u64>, oneshot::Receiver<ImageOutputLease>) {
    let (response, receiver) = oneshot::channel();
    let job = service
        .try_submit_request_with_output(
            QueuedRequest::Generate(request(seed)),
            Some(QueuedOutput {
                gate,
                lease: None,
                response,
            }),
        )
        .unwrap();
    (job, receiver)
}

#[tokio::test]
async fn output_credit_dispatches_fifo_prefix_and_slow_owner_release_resumes_without_rejection() {
    let (entered, execution) = mpsc::channel();
    let (release, permit) = mpsc::channel();
    let service = ServiceCore::start(
        Box::new(Worker {
            entered,
            release: permit,
        }),
        config(),
    )
    .unwrap();
    let initial = service.try_submit(request(0)).unwrap();
    assert_eq!(receive(&execution), [0]);
    let host = host();
    let (blocked, blocked_events) = mpsc::channel();
    let gate: Arc<dyn ImageOutputGate> = Arc::new(OutputGate {
        host: host.clone(),
        bytes: 32,
        blocked,
    });
    let (one, one_credit) = submit(&service, 1, gate.clone());
    let (two, two_credit) = submit(&service, 2, gate.clone());
    let (three, three_credit) = submit(&service, 3, gate.clone());
    let (four, four_credit) = submit(&service, 4, gate);
    let arrivals: Vec<_> = service
        .shared
        .lock()
        .queue
        .iter()
        .map(|job| job.submitted)
        .collect();
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    release.send(()).unwrap();
    assert_eq!(initial.wait().await.unwrap(), 0);
    assert_eq!(receive(&execution), [1, 2]);
    assert_eq!(host.usage().unwrap().committed_bytes(), 64);
    assert_eq!(
        service
            .shared
            .lock()
            .queue
            .iter()
            .map(|job| job.submitted)
            .collect::<Vec<_>>(),
        arrivals[2..]
    );
    release.send(()).unwrap();
    assert_eq!(one.wait().await.unwrap(), 1);
    assert_eq!(two.wait().await.unwrap(), 2);
    let one_credit = one_credit.await.unwrap();
    let two_credit = two_credit.await.unwrap();
    // The earlier attempt stopped at request three; the next attempt waits
    // on the same FIFO head while both completed responses retain real bytes.
    receive(&blocked_events);
    receive(&blocked_events);
    assert!(matches!(
        execution.try_recv(),
        Err(mpsc::TryRecvError::Empty)
    ));
    assert_eq!(service.snapshot().pending_requests, 2);
    assert_eq!(service.snapshot().rejected_requests, 0);
    assert_eq!(service.snapshot().capacity_reductions, 0);
    drop(one_credit);
    assert_eq!(receive(&execution), [3]);
    release.send(()).unwrap();
    assert_eq!(three.wait().await.unwrap(), 3);
    drop(three_credit.await.unwrap());
    assert_eq!(receive(&execution), [4]);
    release.send(()).unwrap();
    assert_eq!(four.wait().await.unwrap(), 4);
    drop(four_credit.await.unwrap());
    drop(two_credit);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    assert!(!service.shutdown().await.unwrap().failure.is_some());
}

#[tokio::test]
async fn output_wait_is_cancellable_and_shutdown_does_not_wait_for_slow_client() {
    let (entered, execution) = mpsc::channel();
    let (_release, permit) = mpsc::channel();
    let service = ServiceCore::start(
        Box::new(Worker {
            entered,
            release: permit,
        }),
        config(),
    )
    .unwrap();
    let host = host();
    let slow_owner = host.reserve(HostMemoryClass::Output, 64).unwrap();
    let (blocked, blocked_events) = mpsc::channel();
    let gate: Arc<dyn ImageOutputGate> = Arc::new(OutputGate {
        host: host.clone(),
        bytes: 32,
        blocked,
    });
    let (first, first_credit) = submit(&service, 1, gate.clone());
    receive(&blocked_events);
    first.cancel();
    assert_eq!(
        first.wait().await.unwrap_err().kind(),
        ImageServiceErrorKind::Cancelled
    );
    assert!(first_credit.await.is_err());
    let (second, second_credit) = submit(&service, 2, gate);
    receive(&blocked_events);
    let retirement = service.shutdown().await.unwrap();
    assert!(retirement.failure.is_none());
    assert_eq!(
        second.wait().await.unwrap_err().kind(),
        ImageServiceErrorKind::Shutdown
    );
    assert!(second_credit.await.is_err());
    assert!(matches!(
        execution.try_recv(),
        Err(mpsc::TryRecvError::Empty)
    ));
    assert_eq!(service.snapshot().pending_requests, 0);
    assert_eq!(host.usage().unwrap().committed_bytes(), 64);
    drop(slow_owner);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}

#[tokio::test]
async fn impossible_single_output_rejects_without_execution_and_active_cancel_refunds_grant() {
    let (entered, execution) = mpsc::channel();
    let (release, permit) = mpsc::channel();
    let service = ServiceCore::start(
        Box::new(Worker {
            entered,
            release: permit,
        }),
        config(),
    )
    .unwrap();
    let host = host();
    let (blocked, _) = mpsc::channel();
    let impossible = Arc::new(OutputGate {
        host: host.clone(),
        bytes: 128,
        blocked: blocked.clone(),
    });
    let (invalid, invalid_credit) = submit(&service, 1, impossible);
    assert_eq!(
        invalid.wait().await.unwrap_err().kind(),
        ImageServiceErrorKind::QueueFull
    );
    assert!(invalid_credit.await.is_err());
    assert_eq!(service.snapshot().executed_requests, 0);
    let gate = Arc::new(OutputGate {
        host: host.clone(),
        bytes: 32,
        blocked,
    });
    let (active, credit) = submit(&service, 2, gate);
    assert_eq!(receive(&execution), [2]);
    assert_eq!(host.usage().unwrap().committed_bytes(), 32);
    active.cancel();
    release.send(()).unwrap();
    assert_eq!(
        active.wait().await.unwrap_err().kind(),
        ImageServiceErrorKind::Cancelled
    );
    assert!(credit.await.is_err());
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    assert_eq!(service.snapshot().rejected_requests, 1);
    assert_eq!(service.snapshot().capacity_reductions, 0);
    assert!(service.shutdown().await.unwrap().failure.is_none());
}

struct AdmissionWorker {
    worker: Worker,
    host: HostMemoryAccount,
    preparations: Arc<std::sync::atomic::AtomicUsize>,
    blocked: mpsc::Sender<()>,
}
impl BatchWorker for AdmissionWorker {
    type Output = u64;
    fn compatible(&self, _: &ImageGenerationRequest, _: &ImageGenerationRequest) -> bool {
        true
    }
    fn prepare(
        &mut self,
        _: &mut [&mut QueuedRequest],
    ) -> Result<ImageBatchPreparation, ImageBatchError> {
        self.preparations.fetch_add(1, Ordering::SeqCst);
        Ok(ImageBatchPreparation::new(None::<HostMemoryReservation>))
    }
    fn try_admit_prepared(
        &mut self,
        prepared: &mut ImageBatchPreparation,
        resources: ImagePreparationResources,
    ) -> Result<bool, ImageBatchError> {
        assert_eq!(resources.output_credit_bytes, 16);
        match self.host.reserve(HostMemoryClass::Output, 32) {
            Ok(owner) => {
                *prepared
                    .0
                    .as_mut()
                    .unwrap()
                    .downcast_mut::<Option<HostMemoryReservation>>()
                    .unwrap() = Some(owner);
                Ok(true)
            }
            Err(error) if error.is_temporary_capacity() => {
                self.blocked.send(()).unwrap();
                Ok(false)
            }
            Err(error) => Err(ImageBatchError::Rejected(error.to_string())),
        }
    }
    fn execute(
        &mut self,
        prepared: ImageBatchPreparation,
        requests: Vec<QueuedRequest>,
    ) -> Result<Vec<u64>, ImageBatchError> {
        let grant = prepared.into_inner::<Option<HostMemoryReservation>>()?;
        assert!(grant.is_some());
        let requests = requests
            .into_iter()
            .map(|request| match request {
                QueuedRequest::Generate(request) => request,
                _ => unreachable!(),
            })
            .collect();
        self.worker.generate_batch(requests)
    }
    fn generate_batch(
        &mut self,
        _: Vec<ImageGenerationRequest>,
    ) -> Result<Vec<u64>, ImageBatchError> {
        panic!("prepared owner must be executed directly")
    }
}

#[tokio::test]
async fn temporary_pipeline_output_pressure_retains_prepared_token_until_refund() {
    let (entered, execution) = mpsc::channel();
    let (release, permit) = mpsc::channel();
    let (blocked, blocked_events) = mpsc::channel();
    let host = HostMemoryAccount::new(
        HostMemoryLimits::new(
            8192,
            HostMemoryClassLimits {
                input_preprocess_bytes: 4096,
                transfer_staging_bytes: 0,
                output_bytes: 64,
                weight_backing_bytes: 0,
            },
        )
        .unwrap(),
    );
    let slow_owner = host.reserve(HostMemoryClass::Output, 48).unwrap();
    let preparations = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let worker = AdmissionWorker {
        worker: Worker {
            entered,
            release: permit,
        },
        host: host.clone(),
        preparations: preparations.clone(),
        blocked,
    };
    let service =
        ServiceCore::start_with_host(Box::new(worker), config(), Some(host.clone())).unwrap();
    let (blocked, _) = mpsc::channel();
    let gate = Arc::new(OutputGate {
        host: host.clone(),
        bytes: 16,
        blocked,
    });
    let (job, credit) = submit(&service, 1, gate);
    receive(&blocked_events);
    assert_eq!(preparations.load(Ordering::SeqCst), 1);
    assert_eq!(service.snapshot().active_requests, 0);
    assert_eq!(service.snapshot().preparing_requests, 1);
    assert!(matches!(
        execution.try_recv(),
        Err(mpsc::TryRecvError::Empty)
    ));
    drop(slow_owner);
    assert_eq!(receive(&execution), [1]);
    assert_eq!(preparations.load(Ordering::SeqCst), 1);
    release.send(()).unwrap();
    assert_eq!(job.wait().await.unwrap(), 1);
    drop(credit.await.unwrap());
    assert!(service.shutdown().await.unwrap().failure.is_none());
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}

#[tokio::test]
async fn unread_completed_output_does_not_hold_execution_slot_when_capacity_remains() {
    let (entered, execution) = mpsc::channel();
    let (release, permit) = mpsc::channel();
    let service = ServiceCore::start(
        Box::new(Worker {
            entered,
            release: permit,
        }),
        config(),
    )
    .unwrap();
    let host = host();
    let (blocked, _) = mpsc::channel();
    let gate: Arc<dyn ImageOutputGate> = Arc::new(OutputGate {
        host: host.clone(),
        bytes: 32,
        blocked,
    });
    let (first, first_credit) = submit(&service, 1, gate.clone());
    assert_eq!(receive(&execution), [1]);
    let (second, second_credit) = submit(&service, 2, gate);
    release.send(()).unwrap();
    // Neither completion nor its output owner has been read by the first
    // client. The next inference still acquires the remaining real credit.
    assert_eq!(receive(&execution), [2]);
    assert_eq!(host.usage().unwrap().committed_bytes(), 64);
    assert_eq!(service.snapshot().executed_requests, 2);
    release.send(()).unwrap();
    assert_eq!(first.wait().await.unwrap(), 1);
    assert_eq!(second.wait().await.unwrap(), 2);
    drop(first_credit.await.unwrap());
    drop(second_credit.await.unwrap());
    assert!(service.shutdown().await.unwrap().failure.is_none());
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}
