//! Ready candidates use real Stored continuations and native admission. The
//! initialization fixture is preconditioned; this is a service observation
//! regression, not full H3/CUDA or prompt-initialization qualification.
use super::*;
use crate::image_generation::{QwenImage21T2iLayout, QwenImage21TransformerConfig};
use ferrum_kernels::backend::reference::{
    reference_vnext_operation_registry, ReferenceVNextComposition,
};
use ferrum_runtime::{DeviceId, WeightMaterializerRegistry};
use std::sync::atomic::AtomicU64;

type Exits = Arc<Mutex<Vec<(u64, usize, ImageDenoisingExitReason)>>>;
struct ObservedReady {
    inner: Preconditioned,
    exits: Exits,
    ready: Arc<Mutex<Vec<u64>>>,
}
impl ImageGenerator for ObservedReady {
    fn denoising_generator(
        &mut self,
    ) -> Option<&mut dyn ImageDenoisingGenerator<Output = ImageGenerationOutput>> {
        Some(self)
    }
    fn host_account(&self) -> &HostMemoryAccount {
        self.inner.host_account()
    }
    fn capabilities(&self) -> Result<ImageGeneratorCapabilities, GeneratorError> {
        self.inner.capabilities()
    }
    fn generate_batch(
        &mut self,
        _: Vec<ImageGenerationRequest>,
    ) -> Result<Vec<ImageGenerationOutput>, ImageBatchError> {
        panic!("Ready withdrawal must use the actual step worker")
    }
    fn shutdown(&mut self) -> Result<(), GeneratorError> {
        self.inner.shutdown()
    }
}
impl ImageDenoisingGenerator for ObservedReady {
    type Output = ImageGenerationOutput;
    fn record_request_exit(&self, r: &ImageDenoisingRequest, reason: ImageDenoisingExitReason) {
        let (seed, step, _) = r
            .get::<ContinuousRequest<Runtime>>()
            .unwrap()
            .fixture_snapshot();
        self.exits.lock().unwrap().push((seed, step, reason));
    }
    fn prepare_request(
        &mut self,
        r: &ImageGenerationRequest,
    ) -> Result<ImageBatchPreparation, ImageBatchError> {
        self.inner.prepare_request(r)
    }
    fn try_admit_request(
        &mut self,
        p: &mut ImageBatchPreparation,
        r: ImagePreparationResources,
    ) -> Result<bool, ImageBatchError> {
        self.inner.try_admit_request(p, r)
    }
    fn start_request(
        &mut self,
        p: ImageBatchPreparation,
        r: &ImageGenerationRequest,
    ) -> Result<ImageDenoisingRequest, ImageBatchError> {
        let ready = self.inner.start_request(p, r)?;
        assert_eq!(
            ready
                .get::<ContinuousRequest<Runtime>>()?
                .fixture_snapshot()
                .1,
            0
        );
        self.ready.lock().unwrap().push(r.seed);
        Ok(ready)
    }
    fn try_prepare_batch(
        &mut self,
        r: &mut [&mut ImageDenoisingRequest],
        p: Option<&mut ImageDenoisingBatch>,
    ) -> Result<Option<ImageDenoisingBatch>, ImageBatchError> {
        self.inner.try_prepare_batch(r, p)
    }
    fn pause_batch(
        &mut self,
        b: &mut ImageDenoisingBatch,
        r: &mut [&mut ImageDenoisingRequest],
    ) -> Result<(), ImageBatchError> {
        self.inner.pause_batch(b, r)
    }
    fn advance_batch(
        &mut self,
        b: &mut ImageDenoisingBatch,
        r: &mut [&mut ImageDenoisingRequest],
    ) -> Result<Vec<bool>, ImageBatchError> {
        self.inner.advance_batch(b, r)
    }
    fn finish_request(
        &mut self,
        r: &mut ImageDenoisingRequest,
        b: &mut ImageDenoisingBatch,
    ) -> Result<Self::Output, ImageBatchError> {
        self.inner.finish_request(r, b)
    }
}
struct Fixture {
    service: ImageService,
    boundary: Arc<Boundary>,
    trace: Arc<Mutex<Trace>>,
    runtime: Arc<Runtime>,
    exits: Exits,
    ready: Arc<Mutex<Vec<u64>>>,
    denied: Arc<AtomicBool>,
    pressure: native_pressure::NativePressure,
}
fn start_ready(first_cohort_refused: bool) -> Fixture {
    let (original, source) = fixture_pipeline();
    // Process-wide admission intentionally shares a ledger for equal device
    // identities. These independent fixtures need independent real devices,
    // including when the cancellation/shutdown cases run in parallel.
    static NEXT_DEVICE: AtomicU64 = AtomicU64::new(0);
    let instance = NEXT_DEVICE.fetch_add(1, Ordering::Relaxed);
    let composition = ReferenceVNextComposition::create_with_capacity(
        DeviceId::new(format!("device.service.ready_exit.{instance}")).unwrap(),
        original.execution.config.device_capacity_bytes,
    )
    .unwrap();
    let QwenImage21Pipeline {
        execution,
        profiles,
        ..
    } = original;
    let crate::image_generation::ImagePipelineExecution {
        host,
        config,
        sources,
        ..
    } = execution;
    let assets =
        Arc::try_unwrap(sources.assets).unwrap_or_else(|_| panic!("one fresh fixture asset owner"));
    let pipeline = QwenImage21Pipeline::new(
        composition.runtime().clone(),
        host,
        reference_vnext_operation_registry(composition.runtime()).unwrap(),
        composition.catalog().clone(),
        WeightMaterializerRegistry::identity_only().unwrap(),
        assets,
        profiles,
        config,
    )
    .unwrap();
    let needs = [vec![2], vec![2, 1]]
        .into_iter()
        .map(|lengths| {
            let batch = lengths.len();
            let c = QwenImage21TransformerConfig::new(
                pipeline.execution.sources.assets.transformer.clone(),
                QwenImage21T2iLayout::new(2, 2, 2, lengths).unwrap(),
            )
            .unwrap();
            let stage = pipeline.prepare_denoiser(&c, batch).unwrap();
            let m = stage
                .component
                .executable()
                .execution_plan()
                .payload()
                .memory();
            m.static_bytes() + m.minimum_runnable_request_bytes()
        })
        .collect::<Vec<_>>();
    let available = if first_cohort_refused {
        needs[0].saturating_sub(256) / 256 * 256
    } else {
        (needs[0] + needs[1]) / 2 / 256 * 256
    };
    assert!(available > 0 && available < needs[if first_cohort_refused { 0 } else { 1 }]);
    if !first_cohort_refused {
        assert!(available >= needs[0]);
    }
    let pressure = native_pressure::NativePressure::new(
        &pipeline,
        pipeline.execution.config.device_capacity_bytes - available,
    );
    let runtime = pipeline.execution.runtime.clone();
    let boundary = Arc::new(Boundary::default());
    let trace = Arc::new(Mutex::new(Vec::new()));
    let exits = Arc::new(Mutex::new(Vec::new()));
    let ready = Arc::new(Mutex::new(Vec::new()));
    let denied = Arc::new(AtomicBool::new(false));
    let service = ImageService::new(
        ObservedReady {
            inner: Preconditioned {
                pipeline,
                _source: source,
                boundary: boundary.clone(),
                trace: trace.clone(),
                denied_native_join: denied.clone(),
            },
            exits: exits.clone(),
            ready: ready.clone(),
        },
        ImageServiceConfig {
            batching: ImageBatchPolicy::Auto,
            maximum_pending: 4,
            maximum_pending_bytes: 1 << 18,
            maximum_prompt_bytes: 1024,
            max_num_seqs: 2,
            request_batch_max_wait_ms: 0,
        },
    )
    .unwrap();
    Fixture {
        service,
        boundary,
        trace,
        runtime,
        exits,
        ready,
        denied,
        pressure,
    }
}
fn assert_one_exit(exits: &Exits, seed: u64, step: usize, reason: ImageDenoisingExitReason) {
    let exits = exits.lock().unwrap();
    assert_eq!(
        exits
            .iter()
            .filter(|(s, _, _)| *s == seed)
            .copied()
            .collect::<Vec<_>>(),
        [(seed, step, reason)]
    );
}

async fn withdraw_ready(closing: bool) {
    let Fixture {
        service,
        boundary,
        trace,
        runtime,
        exits,
        ready,
        denied,
        pressure,
    } = start_ready(false);
    let _release = ReleaseOnDrop(boundary.clone());
    let a = service.try_submit(request(0)).unwrap();
    boundary.wait(1).await;
    let b = service.try_submit(request(1)).unwrap();
    boundary.release(1);
    boundary.wait(2).await;
    assert!(
        denied.load(Ordering::Acquire),
        "the original native B2 admission must actually refuse"
    );
    assert_eq!(*ready.lock().unwrap(), [0, 1]);
    assert!(
        trace
            .lock()
            .unwrap()
            .iter()
            .all(|wave| wave.iter().all(|(seed, _, _)| *seed == 0)),
        "Ready B must never enter a physical forward"
    );
    assert_eq!(service.snapshot().preparing_requests, 1);
    let shutdown = if closing {
        let mut shutdown = Box::pin(service.shutdown());
        assert!(futures::poll!(shutdown.as_mut()).is_pending());
        assert!(!service.is_accepting());
        Some(shutdown)
    } else {
        b.cancel();
        None
    };
    // Only the independent native owner is released. The actual pending state
    // and its Host/output credit remain exclusively owned by the worker.
    pressure.close();
    boundary.release(2);
    let error = b.wait().await.err().expect("Ready candidate is withdrawn");
    assert_eq!(
        error.kind(),
        if closing {
            ImageServiceErrorKind::Shutdown
        } else {
            ImageServiceErrorKind::Cancelled
        }
    );
    assert_one_exit(&exits, 1, 0, ImageDenoisingExitReason::Cancelled);
    let output = a.wait().await.unwrap();
    assert_eq!(output.receipt.seed, 0);
    drop(output);
    check(
        &trace.lock().unwrap(),
        &[
            fixture_trajectory(&[0], &[0]),
            fixture_trajectory(&[1], &[0]),
        ],
    );
    if let Some(shutdown) = shutdown {
        let retired = shutdown.await.unwrap();
        assert!(!retired.has_failure(), "{:?}", retired.failure());
        drop(retired);
        assert_eq!(runtime.snapshot().live_allocations, 0);
        assert_eq!(service.host_account().usage().unwrap().committed_bytes(), 0);
    } else {
        finish(&service, &runtime).await;
    }
    assert_one_exit(&exits, 0, 2, ImageDenoisingExitReason::Completed);
}
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn ready_exit_observation_cancels_native_refused_candidate_before_first_forward() {
    withdraw_ready(false).await;
}
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn ready_exit_observation_shutdown_retires_native_refused_candidate_once() {
    withdraw_ready(true).await;
}
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn ready_exit_observation_first_native_refusal_is_finite_failed_and_worker_recovers() {
    let Fixture {
        service,
        boundary,
        trace,
        runtime,
        exits,
        ready,
        pressure,
        ..
    } = start_ready(true);
    let _release = ReleaseOnDrop(boundary.clone());
    let error = service
        .try_submit(request(0))
        .unwrap()
        .wait()
        .await
        .err()
        .expect("actual first-cohort native refusal");
    assert_eq!(error.kind(), ImageServiceErrorKind::QueueFull);
    assert_eq!(*ready.lock().unwrap(), [0]);
    assert!(trace.lock().unwrap().is_empty());
    assert_one_exit(&exits, 0, 0, ImageDenoisingExitReason::Failed);
    assert_eq!(service.snapshot().preparing_requests, 0);
    assert!(service.is_accepting());
    pressure.close();
    boundary.release(2);
    let output = service
        .try_submit(request(2))
        .unwrap()
        .wait()
        .await
        .unwrap();
    assert_eq!(output.receipt.seed, 2);
    drop(output);
    check(
        &trace.lock().unwrap(),
        &[
            fixture_trajectory(&[0], &[0]),
            fixture_trajectory(&[1], &[0]),
        ],
    );
    finish(&service, &runtime).await;
    assert_one_exit(&exits, 2, 2, ImageDenoisingExitReason::Completed);
}
