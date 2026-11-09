//! Genuine Reference native-account pressure and an actual readonly DiT pin.
//! The worker must advance its remaining real cursor before retrying the VAE;
//! with no advancing cursor it must reject without waiting on a native event.
use super::*;
use crate::image_generation::{
    p_h3::clean_decoder_admission, ImageGenerationFailure, ImagePipelineExecution,
    ImageResidencyPolicy, ImageStage, ImageStagePlanRole, ImageWeightOwnerKey, ImageWeightOwnerPin,
    PreparedImageStage, QwenImage21T2iLayout, QwenImage21TransformerConfig,
    QwenImage21VaeDecoderFamily,
};
use ferrum_kernels::backend::reference::{
    reference_vnext_operation_registry, ReferenceVNextComposition,
};
use ferrum_runtime::{DeviceId, WeightMaterializerRegistry};

mod stage_progress;

struct NativeCompletion {
    inner: Preconditioned,
    pressure: Option<super::native_pressure::NativePressure>,
    pin: Option<ImageWeightOwnerPin<Runtime>>,
    release_on_complete: bool,
    denied: Arc<AtomicBool>,
}
impl ImageGenerator for NativeCompletion {
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
        unreachable!("declared step consumer")
    }
    fn shutdown(&mut self) -> Result<(), GeneratorError> {
        drop(self.pin.take());
        if let Some(pressure) = self.pressure.take() {
            pressure.close();
        }
        ImageGenerator::shutdown(&mut self.inner)
    }
}
impl ImageDenoisingGenerator for NativeCompletion {
    type Output = ImageGenerationOutput;
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
        self.inner.start_request(p, r)
    }
    fn try_prepare_batch(
        &mut self,
        r: &mut [&mut ImageDenoisingRequest],
        p: Option<&mut ImageDenoisingBatch>,
    ) -> Result<Option<ImageDenoisingBatch>, ImageBatchError> {
        let batch = self.inner.try_prepare_batch(r, p)?;
        if batch.is_some() && self.pin.is_none() {
            self.pin = Some(
                self.inner
                    .pipeline
                    .execution
                    .pin_weight_owner(ImageWeightOwnerKey::Denoiser)
                    .unwrap(),
            );
        }
        Ok(batch)
    }
    fn pause_batch(
        &mut self,
        b: &mut ImageDenoisingBatch,
        r: &mut [&mut ImageDenoisingRequest],
    ) -> Result<(), ImageBatchError> {
        let complete = r
            .iter()
            .all(|r| r.get::<ContinuousRequest<Runtime>>().unwrap().is_complete());
        self.inner.pause_batch(b, r)?;
        if complete && self.release_on_complete {
            drop(self.pin.take());
        }
        Ok(())
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
        self.try_finish_request(r, b)?
            .ok_or_else(|| ImageBatchError::Rejected("unexpected Host completion wait".into()))
    }
    fn try_finish_request(
        &mut self,
        r: &mut ImageDenoisingRequest,
        b: &mut ImageDenoisingBatch,
    ) -> Result<Option<Self::Output>, ImageBatchError> {
        // Call the actual typed engine consumer so this fixture never derives
        // a capacity classification from a formatted ImageBatchError.
        let result = self.inner.pipeline.finish_continuous_request_borrowed(
            r.get_mut::<ContinuousRequest<Runtime>>()?,
            b.get::<ContinuousBatch<Runtime>>()?,
        );
        match result {
            Ok(output) => Ok(Some(output)),
            Err(failure) => {
                let diagnostic = failure.to_string();
                match clean_decoder_admission(failure) {
                    Ok(()) => {
                        self.denied.store(true, Ordering::Release);
                        Err(ImageBatchError::AwaitDenoisingProgress {
                            stage: ImageStage::Decoder,
                            diagnostic,
                        })
                    }
                    Err(failure) => Err(ImageBatchError::Fatal(Box::new(
                        crate::image_generation::ImageGenerationError {
                            failure,
                            completed_stages: Vec::new(),
                            completed_steps: Vec::new(),
                        },
                    ))),
                }
            }
        }
    }
}

fn native_pipeline(device_id: &str) -> (QwenImage21Pipeline<Runtime>, tempfile::TempDir) {
    let (original, source) = fixture_pipeline();
    let composition = ReferenceVNextComposition::create_with_capacity(
        DeviceId::new(device_id).unwrap(),
        original.execution.config.device_capacity_bytes,
    )
    .unwrap();
    let QwenImage21Pipeline {
        execution,
        profiles,
        ..
    } = original;
    let ImagePipelineExecution {
        host,
        mut config,
        sources,
        ..
    } = execution;
    let assets =
        Arc::try_unwrap(sources.assets).unwrap_or_else(|_| panic!("one fresh asset owner"));
    config.residency = ImageResidencyPolicy::Adaptive;
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
    (pipeline, source)
}

fn start_native(
    release_on_complete: bool,
) -> (
    ImageService,
    Arc<Boundary>,
    Arc<Mutex<Trace>>,
    Arc<Runtime>,
    Arc<AtomicBool>,
) {
    let (pipeline, source) = native_pipeline(if release_on_complete {
        "device.service.decoder.live_progress"
    } else {
        "device.service.decoder.no_progress"
    });
    let dit = (1..=2)
        .map(|batch| {
            let c = QwenImage21TransformerConfig::new(
                pipeline.execution.sources.assets.transformer.clone(),
                QwenImage21T2iLayout::new(2, 2, 2, vec![2; batch]).unwrap(),
            )
            .unwrap();
            pipeline.prepare_denoiser(&c, batch).unwrap()
        })
        .collect::<Vec<_>>();
    let decoder = pipeline
        .execution
        .sources
        .assets
        .decoder_batch(1, 2, 2)
        .unwrap();
    let family = pipeline
        .prepare_family(
            ImageStage::Decoder,
            ImageStagePlanRole::WholeStage,
            1,
            QwenImage21VaeDecoderFamily::new().unwrap(),
            &decoder,
            &pipeline.profiles.decoder,
        )
        .unwrap();
    let decode = PreparedImageStage::compile(
        ImageStage::Decoder,
        1,
        family,
        &decoder.compile_options().unwrap(),
        &pipeline,
    )
    .unwrap();
    let demand = |s: &PreparedImageStage<Runtime>| {
        let m = s.executable().execution_plan().payload().memory();
        m.static_bytes() + m.minimum_runnable_request_bytes()
    };
    let available = dit
        .iter()
        .map(&demand)
        .chain([demand(&decode)])
        .max()
        .unwrap()
        .div_ceil(256)
        * 256;
    assert!(
        available < dit[0].plan.static_bytes + demand(&decode),
        "one model fits, but its decoder cannot coexist with the actual pinned DiT"
    );
    let pressure = super::native_pressure::NativePressure::new(
        &pipeline,
        pipeline.execution.config.device_capacity_bytes - available,
    );
    let runtime = pipeline.execution.runtime.clone();
    let boundary = Arc::new(Boundary::default());
    let trace = Arc::new(Mutex::new(Vec::new()));
    let denied = Arc::new(AtomicBool::new(false));
    let service = ImageService::new(
        NativeCompletion {
            inner: Preconditioned {
                pipeline,
                _source: source,
                boundary: boundary.clone(),
                trace: trace.clone(),
                denied_native_join: Arc::new(AtomicBool::new(false)),
            },
            pressure: Some(pressure),
            pin: None,
            release_on_complete,
            denied: denied.clone(),
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

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn actual_decoder_native_refusal_advances_survivor_then_retries_after_pin_retirement() {
    let (service, boundary, trace, runtime, denied) = start_native(true);
    let _release = ReleaseOnDrop(boundary.clone());
    let a = service.try_submit(request(0)).unwrap();
    boundary.wait(1).await;
    let b = service.try_submit(request(1)).unwrap();
    boundary.release(1);
    boundary.wait(2).await;
    boundary.release(2);
    let a = tokio::time::timeout(Duration::from_secs(15), a.wait())
        .await
        .unwrap()
        .unwrap();
    let b = tokio::time::timeout(Duration::from_secs(15), b.wait())
        .await
        .unwrap()
        .unwrap();
    assert!(
        denied.load(Ordering::Acquire),
        "actual native VAE admission was rejected while the real DiT pin remained live"
    );
    assert_eq!(
        trace
            .lock()
            .unwrap()
            .last()
            .unwrap()
            .iter()
            .map(|(seed, step, _)| (*seed, *step))
            .collect::<Vec<_>>(),
        [(1, 2)]
    );
    check(
        &trace.lock().unwrap(),
        &[
            fixture_trajectory(&[0], &[0]),
            fixture_trajectory(&[1], &[0]),
        ],
    );
    drop((a, b));
    finish(&service, &runtime).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn actual_decoder_native_refusal_without_a_live_survivor_returns_finitely() {
    let (service, boundary, _trace, runtime, denied) = start_native(false);
    let _release = ReleaseOnDrop(boundary.clone());
    let a = service.try_submit(request(0)).unwrap();
    boundary.wait(1).await;
    boundary.release(1);
    boundary.wait(2).await;
    boundary.release(2);
    let failure = tokio::time::timeout(Duration::from_secs(15), a.wait())
        .await
        .expect("no native-capacity event is required")
        .err()
        .expect("held pin leaves no decoder capacity");
    assert_eq!(failure.kind(), ImageServiceErrorKind::QueueFull);
    assert!(denied.load(Ordering::Acquire));
    assert!(
        service.snapshot().accepting,
        "clean admission refusal does not poison the worker"
    );
    finish(&service, &runtime).await;
}
