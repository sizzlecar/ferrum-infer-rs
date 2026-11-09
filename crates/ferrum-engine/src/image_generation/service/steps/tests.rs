//! Real service consumer with deterministic preconditioned latents/text. Only
//! initialization is supplied by the fixture: membership admission, Stored DiT
//! execution, request-local Euler and actual VAE/output ownership are production
//! paths. Reference does not implement production Philox/TE initialization.
use super::*;
use crate::image_generation::denoising::{fixture_pipeline, fixture_request, fixture_trajectory};
use ferrum_kernels::backend::reference::ReferenceDeviceRuntime as Runtime;
use ferrum_runtime::ExecutablePlanView;

type Trace = Vec<Vec<(u64, usize, Vec<u8>)>>;
#[derive(Default)]
struct BoundaryState {
    reached: usize,
    released: usize,
    all: bool,
}
#[derive(Default)]
struct Boundary {
    state: Mutex<BoundaryState>,
    changed: Condvar,
    arrived: tokio::sync::Notify,
}
impl Boundary {
    fn completed(&self) {
        let mut s = self.state.lock().unwrap();
        s.reached += 1;
        let ordinal = s.reached;
        self.arrived.notify_one();
        // These waits only expose the first two completed device fences to
        // the client. No timer or artificial workload selects batch membership.
        while ordinal <= 2 && s.released < ordinal && !s.all {
            s = self.changed.wait(s).unwrap();
        }
    }
    async fn wait(&self, ordinal: usize) {
        tokio::time::timeout(Duration::from_secs(15), async {
            loop {
                let ready = self.arrived.notified();
                if self.state.lock().unwrap().reached >= ordinal {
                    break;
                }
                ready.await;
            }
        })
        .await
        .expect("actual step reaches its completed fence");
    }
    fn release(&self, ordinal: usize) {
        self.state.lock().unwrap().released = ordinal;
        self.changed.notify_all();
    }
}
struct ReleaseOnDrop(Arc<Boundary>);
impl Drop for ReleaseOnDrop {
    fn drop(&mut self) {
        self.0.state.lock().unwrap_or_else(|p| p.into_inner()).all = true;
        self.0.changed.notify_all();
    }
}

struct Preconditioned {
    pipeline: QwenImage21Pipeline<Runtime>,
    _source: tempfile::TempDir,
    boundary: Arc<Boundary>,
    trace: Arc<Mutex<Trace>>,
    denied_native_join: Arc<AtomicBool>,
}
impl ImageGenerator for Preconditioned {
    fn denoising_generator(
        &mut self,
    ) -> Option<&mut dyn ImageDenoisingGenerator<Output = ImageGenerationOutput>> {
        Some(self)
    }
    fn host_account(&self) -> &HostMemoryAccount {
        self.pipeline.host_account()
    }
    fn capabilities(&self) -> Result<ImageGeneratorCapabilities, GeneratorError> {
        Ok(ImageGeneratorCapabilities {
            maximum_batch_size: 2,
            maximum_width: 32,
            maximum_height: 32,
            dimension_multiple: 32,
        })
    }
    fn generate_batch(
        &mut self,
        _: Vec<ImageGenerationRequest>,
    ) -> Result<Vec<ImageGenerationOutput>, ImageBatchError> {
        panic!("declared step generator must reach the production step consumer")
    }
    fn shutdown(&mut self) -> Result<(), GeneratorError> {
        ImageGenerator::shutdown(&mut self.pipeline)
    }
}
impl ImageDenoisingGenerator for Preconditioned {
    type Output = ImageGenerationOutput;
    fn prepare_request(
        &mut self,
        request: &ImageGenerationRequest,
    ) -> Result<ImageBatchPreparation, ImageBatchError> {
        Ok(ImageBatchPreparation::new(request.seed))
    }
    fn try_admit_request(
        &mut self,
        _: &mut ImageBatchPreparation,
        _: ImagePreparationResources,
    ) -> Result<bool, ImageBatchError> {
        Ok(true)
    }
    fn start_request(
        &mut self,
        prepared: ImageBatchPreparation,
        request: &ImageGenerationRequest,
    ) -> Result<ImageDenoisingRequest, ImageBatchError> {
        assert_eq!(prepared.into_inner::<u64>()?, request.seed);
        Ok(ImageDenoisingRequest::new(fixture_request(
            &mut self.pipeline,
            request.seed,
        )))
    }
    fn try_prepare_batch(
        &mut self,
        requests: &mut [&mut ImageDenoisingRequest],
        previous: Option<&mut ImageDenoisingBatch>,
    ) -> Result<Option<ImageDenoisingBatch>, ImageBatchError> {
        let outcome =
            ImageDenoisingGenerator::try_prepare_batch(&mut self.pipeline, requests, previous);
        if requests.len() > 1 && matches!(outcome, Ok(None)) {
            self.denied_native_join.store(true, Ordering::Release);
        }
        outcome
    }
    fn pause_batch(
        &mut self,
        batch: &mut ImageDenoisingBatch,
        requests: &mut [&mut ImageDenoisingRequest],
    ) -> Result<(), ImageBatchError> {
        ImageDenoisingGenerator::pause_batch(&mut self.pipeline, batch, requests)
    }
    fn advance_batch(
        &mut self,
        batch: &mut ImageDenoisingBatch,
        requests: &mut [&mut ImageDenoisingRequest],
    ) -> Result<Vec<bool>, ImageBatchError> {
        let flags = ImageDenoisingGenerator::advance_batch(&mut self.pipeline, batch, requests)?;
        self.trace.lock().unwrap().push(
            requests
                .iter()
                .map(|r| {
                    r.get::<ContinuousRequest<Runtime>>()
                        .unwrap()
                        .fixture_snapshot()
                })
                .collect(),
        );
        self.boundary.completed();
        Ok(flags)
    }
    fn finish_request(
        &mut self,
        request: &mut ImageDenoisingRequest,
        batch: &mut ImageDenoisingBatch,
    ) -> Result<ImageGenerationOutput, ImageBatchError> {
        ImageDenoisingGenerator::finish_request(&mut self.pipeline, request, batch)
    }
}

fn request(seed: u64) -> ImageGenerationRequest {
    ImageGenerationRequest {
        prompt: "cat".into(),
        width: 32,
        height: 32,
        seed,
        n: 1,
    }
}
fn start() -> (ImageService, Arc<Boundary>, Arc<Mutex<Trace>>, Arc<Runtime>) {
    let (pipeline, source) = fixture_pipeline();
    let (service, boundary, trace, runtime, _) = start_pipeline(pipeline, source);
    (service, boundary, trace, runtime)
}
fn start_pipeline(
    pipeline: QwenImage21Pipeline<Runtime>,
    source: tempfile::TempDir,
) -> (
    ImageService,
    Arc<Boundary>,
    Arc<Mutex<Trace>>,
    Arc<Runtime>,
    Arc<AtomicBool>,
) {
    let runtime = pipeline.execution.runtime.clone();
    let boundary = Arc::new(Boundary::default());
    let trace = Arc::new(Mutex::new(Vec::new()));
    let denied = Arc::new(AtomicBool::new(false));
    let service = ImageService::new(
        Preconditioned {
            pipeline,
            _source: source,
            boundary: boundary.clone(),
            trace: trace.clone(),
            denied_native_join: denied.clone(),
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
    (service, boundary, trace, runtime, denied)
}
fn check(trace: &Trace, independent: &[Vec<Vec<Vec<u8>>>; 2]) {
    for cohort in trace {
        for (seed, progress, sample) in cohort {
            assert_eq!(
                sample,
                &independent[(*seed % 2) as usize][*progress][0],
                "seed {seed}, independent B1 progress {progress}"
            );
        }
    }
}
async fn finish(service: &ImageService, runtime: &Runtime) {
    let retired = service.shutdown().await.unwrap();
    assert!(!retired.has_failure(), "{:?}", retired.failure());
    drop(retired);
    assert_eq!(runtime.snapshot().live_allocations, 0);
    assert_eq!(service.host_account().usage().unwrap().committed_bytes(), 0);
    let snapshot = service.snapshot();
    assert_eq!(snapshot.active_requests, 0);
    assert_eq!(snapshot.pending_requests, 0);
    assert_eq!(snapshot.preparing_requests, 0);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn service_real_dit_mixed_progress_cancel_refill_and_actual_decode_match_b1() {
    let independent = [
        fixture_trajectory(&[0], &[0]),
        fixture_trajectory(&[1], &[0]),
    ];
    let (service, boundary, trace, runtime) = start();
    let _release = ReleaseOnDrop(boundary.clone());
    let a = service.try_submit(request(0)).unwrap();
    boundary.wait(1).await;
    let b = service.try_submit(request(1)).unwrap();
    boundary.release(1);
    boundary.wait(2).await;
    a.cancel();
    let c = service.try_submit(request(2)).unwrap();
    boundary.release(2);
    let cancelled = match a.wait().await {
        Err(error) => error,
        Ok(_) => panic!("withdrawn request must not deliver a completed image"),
    };
    assert_eq!(cancelled.kind(), ImageServiceErrorKind::Cancelled);
    let b = b.wait().await.unwrap();
    let c = c.wait().await.unwrap();
    assert_eq!([b.receipt.seed, c.receipt.seed], [1, 2]);
    for image in [&b, &c] {
        assert_eq!(image.receipt.steps.len(), independent[0].len() - 1);
        assert!(image
            .receipt
            .stages
            .iter()
            .any(|s| s.execution.plan.stage == super::super::super::ImageStage::Decoder));
        assert_eq!(image.rgba.bytes().len(), 32 * 32 * 4 * 2);
    }
    let trace = trace.lock().unwrap().clone();
    assert_eq!(
        trace
            .iter()
            .map(|v| v.iter().map(|(s, p, _)| (*s, *p)).collect::<Vec<_>>())
            .collect::<Vec<_>>(),
        [
            vec![(0, 1)],
            vec![(0, 2), (1, 1)],
            vec![(1, 2), (2, 1)],
            vec![(2, 2)]
        ]
    );
    check(&trace, &independent);
    drop((b, c));
    finish(&service, &runtime).await;
}

struct HostOutputGate {
    host: HostMemoryAccount,
    logical: usize,
}
impl ImageOutputGate for HostOutputGate {
    fn host_output_bytes(&self, _: &HostMemoryAccount) -> usize {
        self.host.backing_bytes(self.logical).unwrap()
    }
    fn try_acquire(&self) -> Result<Option<ImageOutputLease>, ImageOutputGateError> {
        match self.host.reserve(HostMemoryClass::Output, self.logical) {
            Ok(reservation) => Ok(Some(ImageOutputLease::new(reservation))),
            Err(e) if e.is_temporary_capacity() => Ok(None),
            Err(e) => Err(ImageOutputGateError::Capacity(e.to_string())),
        }
    }
    fn subscribe_capacity(&self, waker: std::task::Waker) -> ImageOutputLease {
        ImageOutputLease::new(self.host.subscribe_capacity(waker))
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn service_resource_blocked_join_does_not_starve_real_active_dit() {
    let independent = [
        fixture_trajectory(&[0], &[0]),
        fixture_trajectory(&[1], &[0]),
    ];
    let (service, boundary, trace, runtime) = start();
    let _release = ReleaseOnDrop(boundary.clone());
    let a = service.try_submit(request(0)).unwrap();
    boundary.wait(1).await;
    // This is an actual independently owned Output grant in the same parent,
    // not a synthetic capacity response or a post-admission limit mutation.
    let external = service
        .host_account()
        .reserve(HostMemoryClass::Output, 192 * 1024)
        .unwrap();
    let gate = Arc::new(HostOutputGate {
        host: service.host_account().clone(),
        logical: 64 * 1024,
    });
    let b = service.try_submit_with_output(request(1), gate).unwrap();
    boundary.release(1);
    boundary.wait(2).await;
    assert_eq!(
        trace.lock().unwrap()[1]
            .iter()
            .map(|(s, p, _)| (*s, *p))
            .collect::<Vec<_>>(),
        [(0, 2)]
    );
    assert_eq!(service.snapshot().active_requests, 1);
    assert_eq!(service.snapshot().preparing_requests, 1);
    drop(external);
    boundary.release(2);
    let a = a.wait().await.unwrap();
    let (b, output_grant) = b.wait_with_output().await.unwrap();
    assert_eq!([a.receipt.seed, b.receipt.seed], [0, 1]);
    check(&trace.lock().unwrap(), &independent);
    drop((a, b, output_grant));
    finish(&service, &runtime).await;
}

mod auxiliary_suspension;
mod cohorts;
mod completion;
mod exit_observations;
mod fairness;
mod initialization;
mod native_completion;
mod native_pressure;
mod ready_exits;
mod trace_producer;
mod typed_edit;

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn service_actual_native_b2_pool_pressure_rolls_back_then_b1_keeps_advancing() {
    use crate::image_generation::{QwenImage21T2iLayout, QwenImage21TransformerConfig};
    use ferrum_kernels::backend::reference::{
        reference_vnext_operation_registry, ReferenceVNextComposition,
    };
    use ferrum_runtime::{DeviceId, WeightMaterializerRegistry};
    let independent = [
        fixture_trajectory(&[0], &[0]),
        fixture_trajectory(&[1], &[0]),
    ];
    let (original, source) = fixture_pipeline();
    // A separate declared Reference device isolates this genuine 64 MiB
    // account boundary from unrelated tests sharing the ordinary tiny fixture.
    let c = ReferenceVNextComposition::create_with_capacity(
        DeviceId::new("device.service.actual.native.pressure").unwrap(),
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
    let assets = sources.assets;
    let assets = match Arc::try_unwrap(assets) {
        Ok(assets) => assets,
        Err(_) => panic!("fresh fixture has one immutable asset owner"),
    };
    let mut pipeline = QwenImage21Pipeline::new(
        c.runtime().clone(),
        host,
        reference_vnext_operation_registry(c.runtime()).unwrap(),
        c.catalog().clone(),
        WeightMaterializerRegistry::identity_only().unwrap(),
        assets,
        profiles,
        config,
    )
    .unwrap();
    let requirements = [vec![2], vec![2, 1]]
        .into_iter()
        .map(|lengths| {
            let rows = lengths.len();
            let config = QwenImage21TransformerConfig::new(
                pipeline.execution.sources.assets.transformer.clone(),
                QwenImage21T2iLayout::new(2, 2, 2, lengths).unwrap(),
            )
            .unwrap();
            let stage = pipeline.prepare_denoiser(&config, rows).unwrap();
            let memory = stage
                .component
                .executable()
                .execution_plan()
                .payload()
                .memory();
            memory.static_bytes() + memory.minimum_runnable_request_bytes()
        })
        .collect::<Vec<_>>();
    let available = ((requirements[0] + requirements[1]) / 2 / 256) * 256;
    assert!(requirements[0] <= available && available < requirements[1]);
    let external = native_pressure::NativePressure::new(
        &pipeline,
        pipeline.execution.config.device_capacity_bytes - available,
    );
    assert!(
        pipeline.execution.runtime.snapshot().live_allocations > 0,
        "external native buffer exists"
    );
    let (service, boundary, trace, runtime, denied) = start_pipeline(pipeline, source);
    let _release = ReleaseOnDrop(boundary.clone());
    let a = service.try_submit(request(0)).unwrap();
    boundary.wait(1).await;
    let b = service.try_submit(request(1)).unwrap();
    boundary.release(1);
    boundary.wait(2).await;
    assert!(
        denied.load(Ordering::Acquire),
        "real B2 native admission was attempted and declined"
    );
    assert_eq!(
        trace.lock().unwrap()[1]
            .iter()
            .map(|(s, p, _)| (*s, *p))
            .collect::<Vec<_>>(),
        [(0, 2)]
    );
    assert_eq!(service.snapshot().preparing_requests, 1);
    // Return the actual independent native owner before decode/next admission.
    external.close();
    boundary.release(2);
    let a = a.wait().await.unwrap();
    let b = b.wait().await.unwrap();
    assert_eq!([a.receipt.seed, b.receipt.seed], [0, 1]);
    check(&trace.lock().unwrap(), &independent);
    drop((a, b));
    finish(&service, &runtime).await;
}
