//! Real common-worker jobs and wake/retirement paths. Synthetic timelines
//! exercise scheduling contracts; no native numerical/performance claim.
use super::*;
use ferrum_runtime::{HostMemoryClassLimits, HostMemoryLimits};
use std::sync::atomic::AtomicUsize;

enum Refusal {
    Plan,
    RolledBackDynamic,
    Fatal,
    Host {
        account: HostMemoryAccount,
        attempted: mpsc::Sender<()>,
    },
}
struct CapacityWorker {
    timeline: TimelineWorker,
    refusal: Refusal,
    attempts: Arc<AtomicUsize>,
}
impl BatchWorker for CapacityWorker {
    type Request = TimelineRequest;
    type Output = TimelineOutput;
    fn lifecycle_observer(&self) -> Option<LifecycleObserverHandle> {
        self.timeline.lifecycle_observer()
    }
    fn denoising_generator(
        &mut self,
    ) -> Option<&mut dyn DenoisingGenerator<Request = Self::Request, Output = Self::Output>> {
        Some(self)
    }
    fn execute(
        &mut self,
        _: BatchPreparation,
        _: Vec<Self::Request>,
    ) -> Result<Vec<Self::Output>, BatchError> {
        panic!("iterative protocol required")
    }
    fn compatible(&self, a: &Self::Request, b: &Self::Request) -> bool {
        self.timeline.compatible(a, b)
    }
    fn maximum_batch_size(&self, r: &Self::Request, configured: usize) -> usize {
        self.timeline.maximum_batch_size(r, configured)
    }
}
impl DenoisingGenerator for CapacityWorker {
    type Request = TimelineRequest;
    type Output = TimelineOutput;
    fn supports_request(&mut self, _: &Self::Request) -> bool {
        true
    }
    fn prepare_request(&mut self, r: &Self::Request) -> Result<BatchPreparation, BatchError> {
        self.timeline.prepare_request(r)
    }
    fn try_admit_request(
        &mut self,
        p: &mut BatchPreparation,
        r: PreparationResources,
    ) -> Result<bool, BatchError> {
        self.timeline.try_admit_request(p, r)
    }
    fn start_request(
        &mut self,
        p: BatchPreparation,
        r: &Self::Request,
    ) -> Result<DenoisingRequest, BatchError> {
        self.timeline.start_request(p, r)
    }
    fn try_prepare_batch(
        &mut self,
        r: &mut [&mut DenoisingRequest],
        previous: Option<&mut DenoisingBatch>,
    ) -> Result<Option<DenoisingBatch>, BatchError> {
        if let Refusal::Host { account, attempted } = &self.refusal {
            match account.reserve_scope(&[(HostMemoryClass::Output, 128)]) {
                Ok(scope) => drop(scope),
                Err(pressure) => {
                    assert!(pressure.is_temporary_capacity());
                    self.attempts.fetch_add(1, Ordering::Relaxed);
                    attempted.send(()).unwrap();
                    return Err(BatchError::AwaitHostCapacity {
                        account: account.clone(),
                        pressure,
                    });
                }
            }
        } else if r.len() == 2 {
            self.attempts.fetch_add(1, Ordering::Relaxed);
            // A candidate must keep both its unadvanced continuation and the
            // previous exact B1 owner. The worker may only advance that B1.
            assert_eq!(r[1].get::<Lane>()?.cursor, 0);
            assert_eq!(
                previous.unwrap().get::<Vec<u64>>()?,
                &[r[0].get::<Lane>()?.output.id],
            );
            return match self.refusal {
                Refusal::Plan => {
                    let proof = DeviceCapacityReport {
                        complete_batch: false,
                        required_bytes: 1025,
                        capacity_bytes: 1024,
                        retained_static_bytes: 0,
                        active_stage_peak_bytes: 1025,
                        retained_dynamic_allowance_bytes: 0,
                        native_allowance_bytes: 0,
                    }
                    .validate()
                    .unwrap_err();
                    Err(BatchError::Capacity {
                        diagnostic: proof.to_string(),
                        proof,
                    })
                }
                Refusal::RolledBackDynamic => Ok(None),
                Refusal::Fatal => Err(BatchError::Fatal(Box::new(OriginalFailure))),
                Refusal::Host { .. } => unreachable!(),
            };
        }
        self.timeline.try_prepare_batch(r, previous)
    }
    fn pause_batch(
        &mut self,
        b: &mut DenoisingBatch,
        r: &mut [&mut DenoisingRequest],
    ) -> Result<(), BatchError> {
        self.timeline.pause_batch(b, r)
    }
    fn advance_batch(
        &mut self,
        b: &mut DenoisingBatch,
        r: &mut [&mut DenoisingRequest],
    ) -> Result<Vec<bool>, BatchError> {
        assert_eq!(r.len(), 1, "capacity refusal must not publish a B2");
        self.timeline.advance_batch(b, r)
    }
    fn finish_request(
        &mut self,
        r: &mut DenoisingRequest,
        b: &mut DenoisingBatch,
    ) -> Result<Self::Output, BatchError> {
        self.timeline.finish_request(r, b)
    }
}
#[derive(Debug)]
struct OriginalFailure;
impl fmt::Display for OriginalFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("original non-capacity failure")
    }
}
impl Error for OriginalFailure {}

fn config() -> ServiceConfig {
    ServiceConfig {
        batching: BatchPolicy::Configured,
        maximum_pending: 2,
        maximum_pending_bytes: 4096,
        max_num_seqs: 2,
        request_batch_max_wait_ms: 0,
    }
}
fn request(id: u64) -> TimelineRequest {
    TimelineRequest {
        id,
        frames: vec![id * 10],
        transitions: 3,
    }
}
fn worker(
    mode: Refusal,
    first: Option<mpsc::Sender<()>>,
    release: mpsc::Receiver<()>,
) -> CapacityWorker {
    CapacityWorker {
        timeline: TimelineWorker {
            first_boundary: first,
            release,
            memberships: Arc::new(Mutex::new(Vec::new())),
            observed: Arc::new(Observed::default()),
            cancellation_seen: None,
        },
        refusal: mode,
        attempts: Arc::new(AtomicUsize::new(0)),
    }
}

#[tokio::test]
async fn accepted_b2_capacity_refusals_serialize_both_jobs_without_error_or_lost_progress() {
    for mode in [Refusal::Plan, Refusal::RolledBackDynamic] {
        let (first, reached) = mpsc::channel();
        let (release, released) = mpsc::channel();
        let worker = worker(mode, Some(first), released);
        let memberships = worker.timeline.memberships.clone();
        let attempts = worker.attempts.clone();
        let service = ServiceCore::start(Box::new(worker), config()).unwrap();
        let first = service.try_submit(request(1)).unwrap();
        reached.recv_timeout(Duration::from_secs(10)).unwrap();
        let second = service.try_submit(request(2)).unwrap();
        release.send(()).unwrap();
        let (first, second) = tokio::time::timeout(Duration::from_secs(10), async {
            (first.wait().await, second.wait().await)
        })
        .await
        .expect("both accepted jobs must reach a terminal response");
        assert_eq!(
            first.unwrap(),
            TimelineOutput {
                id: 1,
                frames: vec![16]
            }
        );
        assert_eq!(
            second.unwrap(),
            TimelineOutput {
                id: 2,
                frames: vec![26]
            }
        );
        assert!(attempts.load(Ordering::Relaxed) > 0);
        assert_eq!(
            *memberships.lock().unwrap(),
            vec![
                vec![(1, 0)],
                vec![(1, 1)],
                vec![(1, 2)],
                vec![(2, 0)],
                vec![(2, 1)],
                vec![(2, 2)],
            ]
        );
        let snapshot = service.snapshot();
        assert_eq!(snapshot.executed_requests, 2);
        assert_eq!(snapshot.rejected_requests, 0);
        assert_eq!(
            snapshot.active_requests + snapshot.pending_requests + snapshot.preparing_requests,
            0
        );
        assert!(service.shutdown().await.unwrap().failure.is_none());
    }
}

#[tokio::test]
async fn accepted_ready_request_waits_for_real_host_refund_without_a_running_batch() {
    let host = HostMemoryAccount::new(
        HostMemoryLimits::new(
            1 << 20,
            HostMemoryClassLimits {
                input_preprocess_bytes: 1 << 19,
                output_bytes: 1 << 19,
                transfer_staging_bytes: 0,
                weight_backing_bytes: 0,
            },
        )
        .unwrap(),
    );
    // An independent actual Output owner models an encoder/codec result still
    // alive after the previous model job left the running set.
    let output = host
        .allocate_zeroed(HostMemoryClass::Output, 1 << 19)
        .unwrap();
    let (attempted, waiting) = mpsc::channel();
    let (_, released) = mpsc::channel();
    let worker = worker(
        Refusal::Host {
            account: host.clone(),
            attempted,
        },
        None,
        released,
    );
    let service =
        ServiceCore::start_with_host(Box::new(worker), config(), Some(host.clone())).unwrap();
    let job = service.try_submit(request(3)).unwrap();
    waiting.recv_timeout(Duration::from_secs(10)).unwrap();
    assert_eq!(service.snapshot().active_requests, 0);
    assert_eq!(service.snapshot().preparing_requests, 1);
    drop(output); // Original Host subscription is the only capacity wake.
    let result = tokio::time::timeout(Duration::from_secs(10), job.wait())
        .await
        .unwrap()
        .unwrap();
    assert_eq!(
        result,
        TimelineOutput {
            id: 3,
            frames: vec![36]
        }
    );
    assert!(service.shutdown().await.unwrap().failure.is_none());
    drop(service);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}

#[tokio::test]
async fn capacity_fallback_does_not_downgrade_original_fatal_owners() {
    let (first, reached) = mpsc::channel();
    let (release, released) = mpsc::channel();
    let service = ServiceCore::start(
        Box::new(worker(Refusal::Fatal, Some(first), released)),
        config(),
    )
    .unwrap();
    let first = service.try_submit(request(1)).unwrap();
    reached.recv_timeout(Duration::from_secs(10)).unwrap();
    let second = service.try_submit(request(2)).unwrap();
    release.send(()).unwrap();
    let (first, second) = tokio::time::timeout(Duration::from_secs(10), async {
        (first.wait().await, second.wait().await)
    })
    .await
    .unwrap();
    assert_eq!(first.unwrap_err().kind(), ServiceErrorKind::Generation);
    assert_eq!(second.unwrap_err().kind(), ServiceErrorKind::Generation);
    let mut retired = service.shutdown().await.unwrap();
    let Some(WorkerFailure::Generation(error)) = retired.failure.take() else {
        panic!("original failure must remain fatal")
    };
    let owners = error.downcast::<steps::FailedOwners>().unwrap();
    assert!(owners.source.downcast_ref::<OriginalFailure>().is_some());
    assert_eq!(owners._requests.len(), 2);
    // This synthetic failure submitted no native work. Explicitly consume its
    // original owners here; production recovery obligations are unchanged.
    drop((owners, retired.worker.take()));
}
