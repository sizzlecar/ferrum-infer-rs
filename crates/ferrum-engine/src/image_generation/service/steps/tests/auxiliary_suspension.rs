//! The generic worker protocol around real Reference Stored DiT work and an
//! independently provisioned native auxiliary owner. This is not H3/causal
//! numerical qualification; no provider or successful forward is fabricated.
use super::*;
use ferrum_kernels::backend::reference::{
    reference_vnext_operation_registry, ReferenceVNextComposition,
};
use std::sync::atomic::AtomicU64;

#[derive(Default)]
struct Control {
    entered: Mutex<bool>,
    released: Mutex<bool>,
    changed: Condvar,
    arrived: tokio::sync::Notify,
    events: Mutex<Vec<&'static str>>,
    restore_still_exclusive: bool,
}
impl Control {
    fn enter(&self) {
        *self.entered.lock().unwrap() = true;
        self.events.lock().unwrap().push("auxiliary_native_live");
        self.arrived.notify_one();
        let mut released = self.released.lock().unwrap();
        while !*released {
            released = self.changed.wait(released).unwrap();
        }
    }
    async fn wait(&self) {
        tokio::time::timeout(Duration::from_secs(15), async {
            loop {
                let arrived = self.arrived.notified();
                if *self.entered.lock().unwrap() {
                    return;
                }
                arrived.await;
            }
        })
        .await
        .expect("actual auxiliary owner becomes native-live");
    }
    fn release(&self) {
        *self.released.lock().unwrap_or_else(|p| p.into_inner()) = true;
        self.changed.notify_all();
    }
}
struct Release(Arc<Control>);
impl Drop for Release {
    fn drop(&mut self) {
        self.0.release();
    }
}
struct Phase {
    seed: u64,
    native: Option<native_pressure::NativePressure>,
}
struct Generator {
    inner: Preconditioned,
    control: Arc<Control>,
    suspended: bool,
    native: Option<native_pressure::NativePressure>,
}
impl ImageGenerator for Generator {
    fn denoising_generator(
        &mut self,
    ) -> Option<&mut dyn ImageDenoisingGenerator<Output = ImageGenerationOutput>> {
        Some(self)
    }
    fn host_account(&self) -> &HostMemoryAccount {
        self.inner.host_account()
    }
    fn capabilities(&self) -> Result<ImageGeneratorCapabilities, GeneratorError> {
        let mut capabilities = self.inner.capabilities()?;
        capabilities.maximum_batch_size = self.inner.pipeline.execution.config.maximum_batch_size;
        Ok(capabilities)
    }
    fn generate_batch(
        &mut self,
        _: Vec<ImageGenerationRequest>,
    ) -> Result<Vec<ImageGenerationOutput>, ImageBatchError> {
        unreachable!()
    }
    fn shutdown(&mut self) -> Result<(), GeneratorError> {
        ImageGenerator::shutdown(&mut self.inner)
    }
}
impl ImageDenoisingGenerator for Generator {
    type Output = ImageGenerationOutput;
    fn prepare_request(
        &mut self,
        request: &ImageGenerationRequest,
    ) -> Result<ImageBatchPreparation, ImageBatchError> {
        self.inner.prepare_request(request)
    }
    fn try_admit_request(
        &mut self,
        prepared: &mut ImageBatchPreparation,
        resources: ImagePreparationResources,
    ) -> Result<bool, ImageBatchError> {
        self.inner.try_admit_request(prepared, resources)
    }
    fn try_admit_request_at_boundary(
        &mut self,
        prepared: &mut ImageBatchPreparation,
        resources: ImagePreparationResources,
        batch: Option<&mut ImageDenoisingBatch>,
        requests: &mut [&mut ImageDenoisingRequest],
    ) -> Result<bool, ImageBatchError> {
        let seed = *prepared.get_mut::<u64>()?;
        if seed == 1 && batch.is_some() && !requests.is_empty() {
            assert!(!self.suspended);
            assert!(self
                .inner
                .pipeline
                .execution
                .resident_stages
                .has_weights(crate::image_generation::ImageStage::Denoiser));
            self.inner
                .pipeline
                .execution
                .release_weight_owner(
                    crate::image_generation::ImageWeightOwnerKey::Denoiser,
                    &mut vec![],
                )
                .unwrap_or_else(|e| panic!("real idle DiT close: {e}"));
            assert_eq!(
                self.inner
                    .pipeline
                    .execution
                    .runtime
                    .snapshot()
                    .live_allocations,
                0
            );
            self.native = Some(native_pressure::NativePressure::new(
                &self.inner.pipeline,
                4096,
            ));
            self.suspended = true;
        }
        self.try_admit_request(prepared, resources)
    }
    fn batch_is_suspended(&self, _: &ImageDenoisingBatch) -> bool {
        self.suspended
    }
    fn initialization_blocks_denoising(&self, initialization: &ImageRequestInitialization) -> bool {
        initialization
            .get::<Phase>()
            .is_ok_and(|phase| phase.native.is_some())
    }
    fn restore_suspended_batch(
        &mut self,
        batch: &mut ImageDenoisingBatch,
        requests: &mut [&mut ImageDenoisingRequest],
    ) -> Result<(), ImageBatchError> {
        assert!(self.suspended && self.native.is_none());
        let next = self
            .inner
            .try_prepare_batch(requests, Some(batch))?
            .expect("original cohort really readmits after auxiliary Closed");
        *batch = next;
        self.suspended = self.control.restore_still_exclusive;
        self.control
            .events
            .lock()
            .unwrap()
            .push("restored_native_cohort");
        Ok(())
    }
    fn start_request(
        &mut self,
        prepared: ImageBatchPreparation,
        request: &ImageGenerationRequest,
    ) -> Result<ImageDenoisingRequest, ImageBatchError> {
        self.inner.start_request(prepared, request)
    }
    fn begin_image_initialization(
        &mut self,
        prepared: ImageBatchPreparation,
        request: ImageRequestRef<'_>,
    ) -> Result<ImageInitializationOutcome, ImageBatchError> {
        let ImageRequestRef::Generate(request) = request else {
            unreachable!()
        };
        if request.seed != 1 {
            return self
                .start_request(prepared, request)
                .map(ImageInitializationOutcome::Ready);
        }
        assert_eq!(prepared.into_inner::<u64>()?, request.seed);
        let phase = Phase {
            seed: request.seed,
            native: self.native.take(),
        };
        assert!(phase.native.is_some());
        self.control.enter();
        Ok(ImageInitializationOutcome::Pending(
            ImageRequestInitialization::new(phase),
        ))
    }
    fn advance_image_initialization(
        &mut self,
        initialization: &mut ImageRequestInitialization,
        request: ImageRequestRef<'_>,
        cancelled: bool,
        _: &std::task::Waker,
    ) -> Result<ImageInitializationProgress, ImageBatchError> {
        let phase = initialization.get_mut::<Phase>()?;
        let ImageRequestRef::Generate(request) = request else {
            unreachable!()
        };
        assert_eq!(phase.seed, request.seed);
        phase
            .native
            .take()
            .expect("one original native owner")
            .close();
        self.control.events.lock().unwrap().push("auxiliary_closed");
        if cancelled {
            return Ok(ImageInitializationProgress::Cancelled);
        }
        self.inner
            .start_request(ImageBatchPreparation::new(phase.seed), request)
            .map(ImageInitializationProgress::Ready)
    }
    fn try_prepare_batch(
        &mut self,
        requests: &mut [&mut ImageDenoisingRequest],
        previous: Option<&mut ImageDenoisingBatch>,
    ) -> Result<Option<ImageDenoisingBatch>, ImageBatchError> {
        self.inner.try_prepare_batch(requests, previous)
    }
    fn pause_batch(
        &mut self,
        batch: &mut ImageDenoisingBatch,
        requests: &mut [&mut ImageDenoisingRequest],
    ) -> Result<(), ImageBatchError> {
        self.inner.pause_batch(batch, requests)
    }
    fn advance_batch(
        &mut self,
        batch: &mut ImageDenoisingBatch,
        requests: &mut [&mut ImageDenoisingRequest],
    ) -> Result<Vec<bool>, ImageBatchError> {
        assert!(
            !self.suspended,
            "worker must not fake or submit a suspended forward"
        );
        self.control.events.lock().unwrap().push("actual_forward");
        self.inner.advance_batch(batch, requests)
    }
    fn finish_request(
        &mut self,
        request: &mut ImageDenoisingRequest,
        batch: &mut ImageDenoisingBatch,
    ) -> Result<Self::Output, ImageBatchError> {
        self.inner.finish_request(request, batch)
    }
}

fn generator(inconsistent_restore: bool, maximum_batch_size: usize) -> Generator {
    let (original, source) = fixture_pipeline();
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let composition = ReferenceVNextComposition::create_with_capacity(
        ferrum_runtime::DeviceId::new(format!(
            "device.worker.suspended.{}",
            NEXT.fetch_add(1, Ordering::Relaxed)
        ))
        .unwrap(),
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
        sources,
        mut config,
        ..
    } = execution;
    config.residency = crate::image_generation::ImageResidencyPolicy::Adaptive;
    config.maximum_batch_size = maximum_batch_size;
    let assets =
        Arc::try_unwrap(sources.assets).unwrap_or_else(|_| panic!("one fresh source owner"));
    let pipeline = QwenImage21Pipeline::new(
        composition.runtime().clone(),
        host,
        reference_vnext_operation_registry(composition.runtime()).unwrap(),
        composition.catalog().clone(),
        ferrum_runtime::WeightMaterializerRegistry::identity_only().unwrap(),
        assets,
        profiles,
        config,
    )
    .unwrap();
    let boundary = Arc::new(Boundary::default());
    let trace = Arc::new(Mutex::new(Vec::new()));
    let control = Arc::new(Control {
        restore_still_exclusive: inconsistent_restore,
        ..Control::default()
    });
    Generator {
        inner: Preconditioned {
            pipeline,
            _source: source,
            boundary,
            trace,
            denied_native_join: Arc::new(AtomicBool::new(false)),
        },
        control,
        suspended: false,
        native: None,
    }
}

async fn trajectory(cancel: bool, inconsistent_restore: bool) {
    let independent = [
        fixture_trajectory(&[0], &[0]),
        fixture_trajectory(&[1], &[0]),
    ];
    let generator = generator(inconsistent_restore, 2);
    let runtime = generator.inner.pipeline.execution.runtime.clone();
    let boundary = generator.inner.boundary.clone();
    let trace = generator.inner.trace.clone();
    let control = generator.control.clone();
    let _release = (Release(control.clone()), ReleaseOnDrop(boundary.clone()));
    let service = ImageService::new(
        generator,
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
    let a = service.try_submit(request(0)).unwrap();
    boundary.wait(1).await;
    let b = service.try_submit(request(1)).unwrap();
    boundary.release(1);
    control.wait().await;
    assert_eq!(trace.lock().unwrap().len(), 1);
    if cancel {
        b.cancel();
    }
    control.release();
    if inconsistent_restore {
        assert_eq!(
            a.wait()
                .await
                .err()
                .expect("invalid restore must fail finitely")
                .kind(),
            ImageServiceErrorKind::Generation
        );
        assert_eq!(
            b.wait()
                .await
                .err()
                .expect("pending request retains the same failure")
                .kind(),
            ImageServiceErrorKind::Generation
        );
        let retired = service.shutdown().await.unwrap();
        assert!(retired.has_failure());
        assert_eq!(
            trace.lock().unwrap().len(),
            1,
            "no successful forward is invented"
        );
        assert!(runtime.snapshot().live_allocations > 0);
        assert!(service.host_account().usage().unwrap().committed_bytes() > 0);
        let (mut generator, failure) = retired.into_parts();
        let Some(ImageWorkerFailure::Generation(error)) = failure else {
            panic!("typed original failure owner required");
        };
        let mut owners = error.downcast::<super::super::FailedOwners>().unwrap();
        assert!(owners.source.downcast_ref::<ImageServiceError>().is_some());
        assert_eq!(owners._requests.len(), 2);
        let mut batch = owners
            ._batch
            .take()
            .expect("actual admitted native batch stays recoverable");
        let a = owners
            ._requests
            .iter_mut()
            .find(|request| {
                request
                    .get::<ContinuousRequest<Runtime>>()
                    .unwrap()
                    .fixture_snapshot()
                    .0
                    == 0
            })
            .unwrap();
        generator
            .as_mut()
            .unwrap()
            ._worker
            .denoising_generator()
            .unwrap()
            .pause_batch(&mut batch, &mut [a])
            .unwrap();
        drop((batch, owners));
        generator.as_mut().unwrap()._worker.shutdown().unwrap();
        drop(generator);
        assert_eq!(runtime.snapshot().live_allocations, 0);
        assert_eq!(service.host_account().usage().unwrap().committed_bytes(), 0);
        return;
    }
    boundary.wait(2).await;
    boundary.release(2);
    let b = b.wait().await;
    if cancel {
        assert_eq!(b.err().unwrap().kind(), ImageServiceErrorKind::Cancelled);
    } else {
        drop(b.unwrap());
    }
    drop(a.wait().await.unwrap());
    let events = control.events.lock().unwrap().clone();
    let start = events
        .iter()
        .position(|event| *event == "auxiliary_native_live")
        .unwrap();
    let closed = events
        .iter()
        .position(|event| *event == "auxiliary_closed")
        .unwrap();
    let restored = events
        .iter()
        .position(|event| *event == "restored_native_cohort")
        .unwrap();
    assert!(start < closed && closed < restored);
    assert!(!events[start..restored].contains(&"actual_forward"));
    check(&trace.lock().unwrap(), &independent);
    if !cancel {
        assert!(trace.lock().unwrap().iter().any(|members| members
            .iter()
            .map(|(seed, step, _)| (*seed, *step))
            .collect::<Vec<_>>()
            == [(0, 2), (1, 1)]));
    }
    finish(&service, &runtime).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn suspended_worker_auxiliary_close_precedes_real_restore_and_mixed_forward() {
    trajectory(false, false).await;
}
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn suspended_worker_auxiliary_cancel_retires_native_owner_before_survivor_forward() {
    trajectory(true, false).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn suspended_worker_auxiliary_inconsistent_restore_retains_real_native_and_host_owners() {
    trajectory(false, true).await;
}

mod ready_native;
