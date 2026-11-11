//! Actual shared Host owners exercise the service's hypothetical-release
//! proof. These are capacity/protocol tests, without a model or fake GPU work.
use super::*;
use crate::image_generation::continuous_trace::{
    ImageContinuousTraceHandle, ImageContinuousTraceLimits, ImageContinuousTraceRecorder,
};
use ferrum_runtime::{HostMemoryClassLimits, HostMemoryLimits};
use std::sync::{atomic::AtomicUsize, mpsc};

struct RetainedWorker {
    host: HostMemoryAccount,
    _recorder: ImageContinuousTraceRecorder,
    handle: ImageContinuousTraceHandle,
    engine_peak: Option<usize>,
    probed: mpsc::Sender<()>,
    entered: mpsc::Sender<usize>,
}
impl BatchWorker for RetainedWorker {
    type Output = u64;
    fn compatible(&self, a: &ImageGenerationRequest, b: &ImageGenerationRequest) -> bool {
        a.width == b.width && a.height == b.height
    }
    fn retained_output_bytes(&self) -> u64 {
        self.handle.status().backing_bytes as u64
    }
    fn execution_output_bytes(
        &self,
        _: &[&QueuedRequest],
    ) -> Result<Option<usize>, ImageBatchError> {
        let _ = self.probed.send(());
        Ok(self.engine_peak)
    }
    fn generate_batch(
        &mut self,
        requests: Vec<ImageGenerationRequest>,
    ) -> Result<Vec<u64>, ImageBatchError> {
        let _output = self
            .engine_peak
            .map(|bytes| self.host.allocate_zeroed(HostMemoryClass::Output, bytes))
            .transpose()
            .map_err(|error| preparation_error(error.into()))?;
        let _ = self.entered.send(
            self.host
                .usage()
                .unwrap()
                .class_committed_bytes(HostMemoryClass::Output),
        );
        Ok(requests.into_iter().map(|request| request.seed).collect())
    }
}
struct Gate {
    host: HostMemoryAccount,
    bytes: usize,
    attempts: Arc<AtomicUsize>,
}
impl ImageOutputGate for Gate {
    fn host_output_bytes(&self, host: &HostMemoryAccount) -> usize {
        if self.host.same_parent(host) {
            self.bytes
        } else {
            0
        }
    }
    fn try_acquire(&self) -> Result<Option<ImageOutputLease>, ImageOutputGateError> {
        self.attempts.fetch_add(1, Ordering::SeqCst);
        match self.host.reserve(HostMemoryClass::Output, self.bytes) {
            Ok(owner) => Ok(Some(ImageOutputLease::new(owner))),
            Err(error) if error.is_temporary_capacity() => Ok(None),
            Err(error) => Err(ImageOutputGateError::Capacity(error.to_string())),
        }
    }
    fn subscribe_capacity(&self, waker: std::task::Waker) -> ImageOutputLease {
        ImageOutputLease::new(self.host.subscribe_capacity(waker))
    }
}
fn host(output_bytes: usize) -> HostMemoryAccount {
    HostMemoryAccount::new(
        HostMemoryLimits::new(
            16 << 10,
            HostMemoryClassLimits {
                input_preprocess_bytes: 8 << 10,
                output_bytes,
                transfer_staging_bytes: 0,
                weight_backing_bytes: 0,
            },
        )
        .unwrap(),
    )
}
fn config() -> ImageServiceConfig {
    ImageServiceConfig {
        batching: ImageBatchPolicy::Auto,
        maximum_pending: 2,
        maximum_pending_bytes: 8 << 10,
        maximum_prompt_bytes: 1024,
        max_num_seqs: 1,
        request_batch_max_wait_ms: 0,
    }
}
fn worker(
    host: &HostMemoryAccount,
    trace_bytes: usize,
    engine_peak: Option<usize>,
) -> (
    RetainedWorker,
    ImageContinuousTraceHandle,
    mpsc::Receiver<()>,
    mpsc::Receiver<usize>,
) {
    let (recorder, handle) = ImageContinuousTraceRecorder::new(
        host,
        ImageContinuousTraceLimits {
            maximum_bytes: trace_bytes,
        },
    )
    .unwrap();
    let (probed, probe) = mpsc::channel();
    let (entered, entry) = mpsc::channel();
    (
        RetainedWorker {
            host: host.clone(),
            _recorder: recorder,
            handle: handle.clone(),
            engine_peak,
            probed,
            entered,
        },
        handle,
        probe,
        entry,
    )
}
fn submit(
    service: &ServiceCore<u64>,
    gate: Arc<Gate>,
) -> (Job<u64>, oneshot::Receiver<ImageOutputLease>) {
    let (response, credit) = oneshot::channel();
    let job = service
        .try_submit_request_with_output(
            QueuedRequest::Generate(ImageGenerationRequest {
                prompt: "cat".into(),
                width: 32,
                height: 32,
                seed: 7,
                n: 1,
            }),
            Some(QueuedOutput {
                gate,
                lease: None,
                response,
            }),
        )
        .unwrap();
    (job, credit)
}
async fn shutdown(
    service: ServiceCore<u64>,
    handle: ImageContinuousTraceHandle,
    host: &HostMemoryAccount,
) {
    let retired = service.shutdown().await.unwrap();
    assert!(retired.failure.is_none());
    drop(retired);
    drop(service);
    assert_eq!(
        host.usage()
            .unwrap()
            .class_allocated_bytes(HostMemoryClass::Output),
        handle.status().backing_bytes
    );
    drop(handle);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}

#[tokio::test]
async fn retained_output_rejects_an_impossible_gate_before_waiting_with_or_without_engine_geometry()
{
    for (engine_peak, gate_bytes) in [(None, 48), (Some(32), 16)] {
        let host = host(96);
        let (worker, handle, _probe, entered) = worker(&host, 64, engine_peak);
        let service =
            ServiceCore::start_with_host(Box::new(worker), config(), Some(host.clone())).unwrap();
        let attempts = Arc::new(AtomicUsize::new(0));
        let (job, credit) = submit(
            &service,
            Arc::new(Gate {
                host: host.clone(),
                bytes: gate_bytes,
                attempts: attempts.clone(),
            }),
        );
        let result = tokio::time::timeout(Duration::from_secs(10), job.wait())
            .await
            .expect("hard-capacity rejection cannot await the service's own diagnostic retirement");
        assert_eq!(
            result
                .err()
                .expect("single request cannot coexist with fixed backing")
                .kind(),
            ImageServiceErrorKind::QueueFull
        );
        assert!(credit.await.is_err());
        assert_eq!(attempts.load(Ordering::SeqCst), 0);
        assert!(entered.try_recv().is_err());
        assert_eq!(
            host.usage()
                .unwrap()
                .class_committed_bytes(HostMemoryClass::Output),
            64
        );
        shutdown(service, handle, &host).await;
    }
}

#[tokio::test]
async fn retained_output_preserves_real_caller_output_release_retry_without_double_charging() {
    let host = host(128);
    let (worker, handle, probe, entered) = worker(&host, 48, Some(32));
    let caller = host.allocate_zeroed(HostMemoryClass::Output, 64).unwrap();
    let service =
        ServiceCore::start_with_host(Box::new(worker), config(), Some(host.clone())).unwrap();
    let attempts = Arc::new(AtomicUsize::new(0));
    let (job, credit) = submit(
        &service,
        Arc::new(Gate {
            host: host.clone(),
            bytes: 32,
            attempts: attempts.clone(),
        }),
    );
    probe
        .recv_timeout(Duration::from_secs(10))
        .expect("worker examines actual output geometry");
    // The fixed 48-byte trace cannot retire; the separate 64-byte caller can.
    // No gate or execution is admitted while both original owners remain.
    assert_eq!(attempts.load(Ordering::SeqCst), 0);
    assert!(entered.try_recv().is_err());
    assert_eq!(
        host.usage()
            .unwrap()
            .class_allocated_bytes(HostMemoryClass::Output),
        112
    );
    drop(caller);
    let result = tokio::time::timeout(Duration::from_secs(10), job.wait())
        .await
        .expect("original Host refund wakes the retained FIFO request")
        .unwrap();
    assert_eq!(result, 7);
    let credit = credit.await.unwrap();
    // Fixed backing + actual gate reservation + actual engine allocation;
    // the diagnostic is never granted a second time as execution demand.
    assert_eq!(entered.recv_timeout(Duration::from_secs(10)).unwrap(), 112);
    assert!(attempts.load(Ordering::SeqCst) > 0);
    drop(credit);
    shutdown(service, handle, &host).await;
}
