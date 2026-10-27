//! Cancel a service-active request before its real Metal execution, not a GPU
//! kernel in flight. The service must still execute and retire that request.
use super::*;
use crate::image_generation::service::{
    ImageBatchError, ImageGenerator, ImageGeneratorCapabilities, ImageService, ImageServiceConfig,
    ImageServiceErrorKind,
};
use ferrum_kernels::backend::metal::vnext_runtime::MetalDeviceRuntime;
use std::{
    future::{poll_fn, Future},
    sync::{mpsc, Mutex},
    task::Poll,
    time::Duration,
};

type GeneratorError = Box<dyn std::error::Error + Send + 'static>;

struct GatedMetalGenerator {
    pipeline: QwenImage21Pipeline<MetalDeviceRuntime>,
    entered: Option<mpsc::Sender<()>>,
    release: mpsc::Receiver<()>,
    completed: Arc<Mutex<Vec<ImageGenerationReceipt>>>,
}

impl ImageGenerator for GatedMetalGenerator {
    fn host_account(&self) -> &HostMemoryAccount {
        self.pipeline.host_account()
    }

    fn capabilities(&self) -> Result<ImageGeneratorCapabilities, GeneratorError> {
        ImageGenerator::capabilities(&self.pipeline)
    }

    fn generate_batch(
        &mut self,
        requests: Vec<ImageGenerationRequest>,
    ) -> Result<Vec<ImageGenerationOutput>, ImageBatchError> {
        if let Some(entered) = self.entered.take() {
            let _ = entered.send(());
            // The drop guard releases this gate on assertion failure; this
            // timeout is a second bound, never a scheduling delay.
            self.release
                .recv_timeout(Duration::from_secs(30))
                .map_err(|error| ImageBatchError::Rejected(format!("test gate: {error}")))?;
        }
        let outputs = ImageGenerator::generate_batch(&mut self.pipeline, requests)?;
        self.completed
            .lock()
            .unwrap()
            .extend(outputs.iter().map(|output| output.receipt.clone()));
        Ok(outputs)
    }

    fn shutdown(&mut self) -> Result<(), GeneratorError> {
        ImageGenerator::shutdown(&mut self.pipeline)
    }
}

struct ReleaseOnDrop(Option<mpsc::Sender<()>>);
impl ReleaseOnDrop {
    fn release(&mut self) {
        if let Some(sender) = self.0.take() {
            let _ = sender.send(());
        }
    }
}
impl Drop for ReleaseOnDrop {
    fn drop(&mut self) {
        self.release();
    }
}

#[tokio::test]
async fn metal_service_cancellation_executes_active_work_then_recovers_and_releases_owners() {
    let _guard = METAL_FACTORY_TEST.lock().unwrap_or_else(|e| e.into_inner());
    let (assets, _directory) = assets();
    let mut config = metal_config();
    config.pipeline.residency = ImageResidencyPolicy::AllStages;
    let pipeline = config.create_pipeline(assets).unwrap();
    let host = pipeline.host_account().clone();
    let runtime = Arc::downgrade(&pipeline.runtime);
    let ceiling = pipeline.config.device_capacity_bytes;
    // This observer owns only a budget identity, not the native runtime. A
    // whole-ceiling reservation after shutdown proves all prior claims retired.
    let capacity = DeviceCapacityOwner::open(pipeline.runtime.descriptor(), ceiling).unwrap();
    let completed = Arc::new(Mutex::new(Vec::new()));
    let (entered_tx, entered_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel();
    let service = ImageService::new(
        GatedMetalGenerator {
            pipeline,
            entered: Some(entered_tx),
            release: release_rx,
            completed: completed.clone(),
        },
        ImageServiceConfig {
            batching: ferrum_types::ImageBatchPolicy::Configured,
            maximum_pending: 1,
            maximum_pending_bytes: 1024 * 1024,
            maximum_prompt_bytes: 1024,
            max_num_seqs: 1,
            request_batch_max_wait_ms: 0,
        },
    )
    .unwrap();
    // Declared after the service so unwinding releases the gate before the
    // service's synchronous worker join. No detached worker on failed asserts.
    let mut gate = ReleaseOnDrop(Some(release_tx));
    let request = |seed| ImageGenerationRequest {
        prompt: "blue cat".into(),
        seed,
        width: 32,
        height: 32,
        n: 1,
    };
    let active = service.try_submit(request(7)).unwrap();
    entered_rx.recv_timeout(Duration::from_secs(10)).unwrap();
    assert_eq!(service.snapshot().active_requests, 1);
    let pending = service.try_submit(request(8)).unwrap();
    assert_eq!(service.snapshot().pending_requests, 1);
    assert!(matches!(
        service.try_submit(request(9)),
        Err(error) if error.kind() == ImageServiceErrorKind::QueueFull
    ));
    pending.cancel();
    // Admission prunes the cancelled queued request and reuses its only slot
    // while the worker is still deterministically held at the first call.
    let healthy = service.try_submit(request(9)).unwrap();
    assert_eq!(service.snapshot().pending_requests, 1);
    assert_eq!(
        tokio::time::timeout(Duration::from_secs(10), pending.wait())
            .await
            .unwrap()
            .err()
            .unwrap()
            .kind(),
        ImageServiceErrorKind::Cancelled
    );
    active.cancel();
    let mut active_wait = Box::pin(active.wait());
    assert!(poll_fn(|cx| Poll::Ready(active_wait.as_mut().poll(cx).is_pending())).await);
    assert_eq!(service.snapshot().active_requests, 1);
    assert!(completed.lock().unwrap().is_empty());
    assert!(runtime.upgrade().is_some());
    assert!(matches!(
        capacity.reserve(DeviceOpaqueMemoryKind::Bootstrap, ceiling),
        Err(VNextError::DeviceCapacityUnavailable(_))
    ));
    gate.release();
    assert_eq!(
        tokio::time::timeout(Duration::from_secs(120), active_wait)
            .await
            .unwrap()
            .err()
            .unwrap()
            .kind(),
        ImageServiceErrorKind::Cancelled
    );
    let output = tokio::time::timeout(Duration::from_secs(120), healthy.wait())
        .await
        .unwrap()
        .unwrap();
    assert_eq!(output.receipt.seed, 9);
    assert_eq!(output.rgba.tensor().dimensions(), [1, 4, 32, 32]);
    assert!(output.rgba.bytes().bytes().chunks_exact(2).all(|bytes| {
        half::bf16::from_bits(u16::from_le_bytes(bytes.try_into().unwrap())).is_finite()
    }));
    assert!(service.is_accepting());
    {
        let receipts = completed.lock().unwrap();
        assert_eq!(receipts.iter().map(|r| r.seed).collect::<Vec<_>>(), [7, 9]);
        for receipt in receipts.iter() {
            assert!(!receipt.steps.is_empty());
            assert!(receipt.native_memory.unwrap().observed_peak_allocated_bytes > 0);
            for stage in [
                ImageStage::TextEncoder,
                ImageStage::Denoiser,
                ImageStage::Decoder,
            ] {
                assert!(receipt
                    .stages
                    .iter()
                    .any(|s| s.execution.plan.stage == stage));
            }
        }
    }
    let retired = service.shutdown().await.unwrap();
    assert!(!retired.has_failure());
    assert_eq!(service.snapshot().active_requests, 0);
    assert_eq!(service.snapshot().pending_requests, 0);
    drop(retired);
    drop(service);
    assert!(host.usage().unwrap().committed_bytes() > 0); // Healthy output still owns its backing.
    drop(output);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    assert!(runtime.upgrade().is_none());
    let all_capacity = capacity
        .reserve(DeviceOpaqueMemoryKind::Bootstrap, ceiling)
        .unwrap();
    assert!(capacity.reserve(DeviceOpaqueMemoryKind::Stream, 1).is_err());
    drop(all_capacity);
}
