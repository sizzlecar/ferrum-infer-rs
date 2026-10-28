use super::*;
use std::sync::mpsc;

type Seeds = Vec<u64>;
struct PreparedFixture {
    capacity: usize,
    preparation: mpsc::Sender<Seeds>,
    prepare_release: mpsc::Receiver<()>,
    execution: mpsc::Sender<Seeds>,
    execute_release: mpsc::Receiver<()>,
    reject: bool,
    fail_execution: bool,
}
impl BatchWorker for PreparedFixture {
    type Output = u64;
    fn compatible(&self, a: &ImageGenerationRequest, b: &ImageGenerationRequest) -> bool {
        a.prompt == b.prompt
    }
    fn prepare(
        &mut self,
        requests: &mut [&mut QueuedRequest],
    ) -> Result<ImageBatchPreparation, ImageBatchError> {
        let seeds: Seeds = requests.iter().map(|r| r.generation().seed).collect();
        self.preparation.send(seeds.clone()).unwrap();
        self.prepare_release
            .recv_timeout(Duration::from_secs(10))
            .unwrap();
        if self.reject {
            return Err(ImageBatchError::Rejected("unsupported operator".into()));
        }
        if requests.len() > self.capacity {
            return Err(capacity_error(requests.len(), self.capacity));
        }
        Ok(ImageBatchPreparation::new(seeds))
    }
    fn execute(
        &mut self,
        prepared: ImageBatchPreparation,
        requests: Vec<QueuedRequest>,
    ) -> Result<Vec<u64>, ImageBatchError> {
        let seeds = prepared.into_inner::<Seeds>()?;
        assert_eq!(
            seeds,
            requests
                .iter()
                .map(|r| r.generation().seed)
                .collect::<Seeds>()
        );
        self.execution.send(seeds.clone()).unwrap();
        self.execute_release
            .recv_timeout(Duration::from_secs(10))
            .unwrap();
        if self.fail_execution {
            return Err(capacity_error(requests.len(), 0));
        }
        Ok(seeds)
    }
    fn generate_batch(
        &mut self,
        _: Vec<ImageGenerationRequest>,
    ) -> Result<Vec<u64>, ImageBatchError> {
        panic!("a successful preparation must execute directly")
    }
}
fn capacity_error(required: usize, capacity: usize) -> ImageBatchError {
    let proof = super::super::ImageDeviceCapacityExceeded {
        report: super::super::ImageDeviceCapacityReport {
            complete_batch: true,
            required_bytes: required as u64,
            capacity_bytes: capacity as u64,
            retained_static_bytes: 0,
            active_stage_peak_bytes: required as u64,
            retained_dynamic_allowance_bytes: 0,
            native_allowance_bytes: 0,
        },
    };
    ImageBatchError::Capacity {
        diagnostic: proof.to_string(),
        proof,
    }
}
struct Controls {
    preparation: mpsc::Receiver<Seeds>,
    prepare_release: mpsc::Sender<()>,
    execution: mpsc::Receiver<Seeds>,
    execute_release: mpsc::Sender<()>,
}
fn fixture(capacity: usize) -> (PreparedFixture, Controls) {
    let (preparation, p) = mpsc::channel();
    let (pr, prepare_release) = mpsc::channel();
    let (execution, e) = mpsc::channel();
    let (er, execute_release) = mpsc::channel();
    (
        PreparedFixture {
            capacity,
            preparation,
            prepare_release,
            execution,
            execute_release,
            reject: false,
            fail_execution: false,
        },
        Controls {
            preparation: p,
            prepare_release: pr,
            execution: e,
            execute_release: er,
        },
    )
}
fn recv(channel: &mpsc::Receiver<Seeds>) -> Seeds {
    channel
        .recv_timeout(Duration::from_secs(10))
        .expect("worker made progress")
}
fn config() -> ImageServiceConfig {
    ImageServiceConfig {
        batching: ImageBatchPolicy::Auto,
        maximum_pending: 4,
        maximum_pending_bytes: 4096,
        maximum_prompt_bytes: 128,
        max_num_seqs: 4,
        request_batch_max_wait_ms: 0,
    }
}
fn request(seed: u64) -> ImageGenerationRequest {
    ImageGenerationRequest {
        prompt: "a".into(),
        width: 32,
        height: 32,
        seed,
        n: 1,
    }
}

#[tokio::test]
async fn auto_dispatches_idle_one_and_shrinks_only_before_execution_preserving_fifo() {
    let (worker, c) = fixture(2);
    let service = ServiceCore::start(Box::new(worker), config()).unwrap();
    let first = service.try_submit(request(0)).unwrap();
    assert_eq!(recv(&c.preparation), [0]);
    c.prepare_release.send(()).unwrap();
    assert_eq!(recv(&c.execution), [0]);
    let mut jobs = Vec::new();
    for seed in 1..=4 {
        jobs.push(service.try_submit(request(seed)).unwrap());
    }
    let arrivals = service
        .shared
        .lock()
        .queue
        .iter()
        .map(|job| job.submitted)
        .collect::<Vec<_>>();
    c.execute_release.send(()).unwrap();
    assert_eq!(first.wait().await.unwrap(), 0);
    assert_eq!(recv(&c.preparation), [1, 2, 3, 4]);
    let snap = service.snapshot();
    assert_eq!(
        (
            snap.active_requests,
            snap.pending_requests,
            snap.preparing_requests
        ),
        (0, 4, 4)
    );
    assert_eq!(
        service.try_submit(request(5)).err().unwrap().kind(),
        ImageServiceErrorKind::QueueFull
    );
    c.prepare_release.send(()).unwrap();
    assert_eq!(recv(&c.preparation), [1, 2]);
    assert_eq!(service.snapshot().pending_requests, 4);
    c.prepare_release.send(()).unwrap();
    assert_eq!(recv(&c.preparation), [1, 2, 3]);
    c.prepare_release.send(()).unwrap();
    assert_eq!(recv(&c.execution), [1, 2]);
    {
        let state = service.shared.lock();
        assert_eq!(state.queue.front().unwrap().request.generation().seed, 3);
        assert_eq!(state.queue.front().unwrap().submitted, arrivals[2]);
        assert_eq!(state.queue.back().unwrap().submitted, arrivals[3]);
    }
    let snap = service.snapshot();
    assert_eq!(
        (
            snap.active_requests,
            snap.pending_requests,
            snap.capacity_reductions
        ),
        (2, 2, 2)
    );
    c.execute_release.send(()).unwrap();
    assert_eq!(recv(&c.preparation), [3, 4]);
    c.prepare_release.send(()).unwrap();
    assert_eq!(recv(&c.execution), [3, 4]);
    c.execute_release.send(()).unwrap();
    for (index, job) in jobs.into_iter().enumerate() {
        assert_eq!(job.wait().await.unwrap(), index as u64 + 1);
    }
    let snap = service.snapshot();
    assert_eq!(
        (
            snap.rejected_requests,
            snap.executed_batches,
            snap.executed_requests
        ),
        (1, 3, 5)
    );
    assert!(service.shutdown().await.unwrap().failure.is_none());
}

#[tokio::test]
async fn single_image_capacity_and_non_capacity_errors_never_execute_or_retry() {
    for unsupported in [false, true] {
        let (mut worker, c) = fixture(0);
        worker.reject = unsupported;
        let service = ServiceCore::start(Box::new(worker), config()).unwrap();
        let job = service.try_submit(request(7)).unwrap();
        assert_eq!(recv(&c.preparation), [7]);
        c.prepare_release.send(()).unwrap();
        assert_eq!(
            job.wait().await.unwrap_err().kind(),
            ImageServiceErrorKind::Generation
        );
        assert!(c.execution.try_recv().is_err());
        assert_eq!(service.snapshot().capacity_reductions, 0);
        assert_eq!(service.snapshot().executed_batches, 0);
        assert!(service.shutdown().await.unwrap().failure.is_none());
    }
}

#[tokio::test]
async fn preparation_cancel_and_shutdown_do_not_submit_device_work() {
    for closing in [false, true] {
        let (worker, c) = fixture(4);
        let service = ServiceCore::start(Box::new(worker), config()).unwrap();
        let job = service.try_submit(request(1)).unwrap();
        assert_eq!(recv(&c.preparation), [1]);
        if closing {
            service.shared.stop();
        } else {
            job.cancel();
        }
        c.prepare_release.send(()).unwrap();
        assert_eq!(
            job.wait().await.unwrap_err().kind(),
            if closing {
                ImageServiceErrorKind::Shutdown
            } else {
                ImageServiceErrorKind::Cancelled
            }
        );
        assert!(service.shutdown().await.unwrap().failure.is_none());
        assert!(c.execution.try_recv().is_err());
        assert_eq!(service.snapshot().pending_requests, 0);
    }
}

#[tokio::test]
async fn execution_capacity_failure_is_fatal_and_never_retried() {
    let (mut worker, c) = fixture(4);
    worker.fail_execution = true;
    let service = ServiceCore::start(Box::new(worker), config()).unwrap();
    let job = service.try_submit(request(1)).unwrap();
    assert_eq!(recv(&c.preparation), [1]);
    c.prepare_release.send(()).unwrap();
    assert_eq!(recv(&c.execution), [1]);
    c.execute_release.send(()).unwrap();
    assert!(job.wait().await.is_err());
    assert!(!service.snapshot().accepting);
    assert_eq!(service.snapshot().capacity_reductions, 0);
    let mut retired = service.shutdown().await.unwrap();
    assert!(retired.failure.take().is_some());
    drop(retired.worker.take());
}

#[tokio::test]
async fn byte_budget_counts_preparation_and_host_reservation_until_execution_finishes() {
    let (worker, c) = fixture(4);
    let mut config = config();
    config.maximum_pending_bytes = QueuedRequest::Generate(request(1)).pending_bytes();
    let host = HostMemoryAccount::new(
        ferrum_runtime::HostMemoryLimits::new(
            8192,
            ferrum_runtime::HostMemoryClassLimits {
                input_preprocess_bytes: 8192,
                transfer_staging_bytes: 0,
                output_bytes: 0,
                weight_backing_bytes: 0,
            },
        )
        .unwrap(),
    );
    let service =
        ServiceCore::start_with_host(Box::new(worker), config, Some(host.clone())).unwrap();
    assert_eq!(service.shared.lock().queue.capacity(), 0);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    let job = service.try_submit(request(1)).unwrap();
    assert_eq!(recv(&c.preparation), [1]);
    let queue_bytes = {
        let state = service.shared.lock();
        assert!(state.queue.is_empty());
        let bytes = state.queue.capacity() * std::mem::size_of::<Queued<u64>>();
        assert_eq!(state.queue_backing.as_ref().unwrap().bytes(), bytes);
        bytes
    };
    let request_bytes = host
        .backing_bytes(ImageService::minimum_pending_request_bytes() + request(1).prompt.capacity())
        .unwrap();
    assert_eq!(
        host.usage().unwrap().committed_bytes(),
        host.backing_bytes(queue_bytes).unwrap() + request_bytes
    );
    assert_eq!(
        service.snapshot().pending_bytes,
        config.maximum_pending_bytes
    );
    assert_eq!(
        service.try_submit(request(2)).err().unwrap().kind(),
        ImageServiceErrorKind::QueueFull
    );
    assert!(host.usage().unwrap().committed_bytes() > 0);
    c.prepare_release.send(()).unwrap();
    assert_eq!(recv(&c.execution), [1]);
    assert_eq!(service.snapshot().pending_bytes, 0);
    assert!(host.usage().unwrap().committed_bytes() > 0);
    c.execute_release.send(()).unwrap();
    assert_eq!(job.wait().await.unwrap(), 1);
    // Empty logical queue retains its allocation and its separate Host charge.
    // The per-job metadata grant survives preparation/execution, then refunds.
    assert!(host.usage().unwrap().committed_bytes() >= queue_bytes);
    assert!(service.shutdown().await.unwrap().failure.is_none());
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}

#[tokio::test]
async fn queue_growth_keeps_prepared_return_slots_and_accounts_relocation_peak() {
    let (worker, c) = fixture(4);
    let slot_bytes = std::mem::size_of::<Queued<u64>>();
    let job_bytes = ImageService::minimum_pending_request_bytes() + request(1).prompt.capacity();
    // Two job owners plus both the old one-slot and new two-slot allocations.
    let peak_bytes = 2 * job_bytes + 3 * slot_bytes;
    let host = HostMemoryAccount::new(
        ferrum_runtime::HostMemoryLimits::new(
            peak_bytes,
            ferrum_runtime::HostMemoryClassLimits {
                input_preprocess_bytes: peak_bytes,
                transfer_staging_bytes: 0,
                output_bytes: 0,
                weight_backing_bytes: 0,
            },
        )
        .unwrap(),
    );
    let pressure = host.reserve(HostMemoryClass::InputPreprocess, 1).unwrap();
    let service =
        ServiceCore::start_with_host(Box::new(worker), config(), Some(host.clone())).unwrap();
    let first = service.try_submit(request(1)).unwrap();
    assert_eq!(recv(&c.preparation), [1]);
    let before_growth = host.usage().unwrap().committed_bytes();
    assert_eq!(
        service.try_submit(request(2)).err().unwrap().kind(),
        ImageServiceErrorKind::QueueFull
    );
    assert_eq!(host.usage().unwrap().committed_bytes(), before_growth);
    assert_eq!(service.shared.lock().queue.capacity(), 1);
    drop(pressure);
    let second = service.try_submit(request(2)).unwrap();
    {
        let state = service.shared.lock();
        assert_eq!((state.queue.len(), state.preparing), (1, 1));
        assert_eq!(state.queue.capacity(), 2);
        assert_eq!(
            state.queue_backing.as_ref().unwrap().bytes(),
            2 * slot_bytes
        );
    }
    assert_eq!(
        host.usage().unwrap().committed_bytes(),
        2 * job_bytes + 2 * slot_bytes
    );
    c.prepare_release.send(()).unwrap();
    assert_eq!(recv(&c.execution), [1]);
    c.execute_release.send(()).unwrap();
    assert_eq!(first.wait().await.unwrap(), 1);
    assert_eq!(recv(&c.preparation), [2]);
    c.prepare_release.send(()).unwrap();
    assert_eq!(recv(&c.execution), [2]);
    c.execute_release.send(()).unwrap();
    assert_eq!(second.wait().await.unwrap(), 2);
    assert!(service.shutdown().await.unwrap().failure.is_none());
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}
