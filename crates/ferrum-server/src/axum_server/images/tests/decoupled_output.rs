//! Real Reference VAE execution through HTTP, with a deliberately occupied
//! Tokio blocking pool. This proves lifecycle separation, not model quality.
use super::*;
use ferrum_engine::image_generation::service::{
    ImageBatchError, ImageGenerator, ImageGeneratorCapabilities, ImageServiceConfig,
};
use ferrum_engine::image_generation::{
    ImageAssetReceipt, ImageDecoderBatchPolicy, ImageGenerationOutput, ImageGenerationReceipt,
    ImageResidencyPolicy,
};
use ferrum_execution::{
    component::{
        CompiledTensorComponent, ComponentInitializationConfig, ComponentWeightSource,
        TensorComponentPolicy, TensorComponentSession,
    },
    TensorExecutorConfig,
};
use ferrum_kernels::backend::reference::{ReferenceDeviceRuntime, ReferenceVNextComposition};
use ferrum_models::qwen_image21::{QwenImage21VaeDecoderConfig, QwenImage21VaeDecoderFamily};
use ferrum_runtime::*;
use std::{
    borrow::Cow,
    collections::BTreeMap,
    sync::{Condvar, Mutex},
    time::Duration,
};

fn id<T: TryFrom<String>>(value: impl Into<String>) -> T
where
    T::Error: std::fmt::Debug,
{
    T::try_from(value.into()).unwrap()
}

struct Weights(BTreeMap<WeightId, Vec<u8>>);
impl WeightComponentSource for Weights {
    fn component<'a>(
        &'a self,
        spec: &WeightComponentSpec,
    ) -> Result<WeightComponentPayload<'a>, VNextError> {
        WeightComponentPayload::new(
            spec,
            spec.external_names[0].clone(),
            "zero-bf16-lifecycle-fixture",
            spec.dimensions.clone(),
            spec.physical_element_type(),
            Cow::Borrowed(self.0[&spec.id].as_slice()),
        )
    }
}

struct Decoder {
    host: HostMemoryAccount,
    session: Option<TensorComponentSession<ReferenceDeviceRuntime>>,
    input: ResolvedTensorSpec,
    calls: Arc<AtomicUsize>,
}
impl Decoder {
    fn new(host: HostMemoryAccount, calls: Arc<AtomicUsize>) -> Self {
        let decoder = QwenImage21VaeDecoderConfig::new(
            64,
            1,
            vec![1, 2, 4, 8, 8],
            2,
            vec![true, true, true, false],
            1,
            2,
            2,
        )
        .unwrap();
        let family = TypedFamilyRegistration::new(QwenImage21VaeDecoderFamily::new().unwrap())
            .prepare_with_profile(
                &serde_json::to_value(&decoder).unwrap(),
                &id(QwenImage21VaeDecoderFamily::REFERENCE_PROFILE_ID),
            )
            .unwrap();
        let weights = Weights(
            family
                .stored_weight_schema()
                .unwrap()
                .components
                .iter()
                .map(|spec| {
                    (
                        spec.id.clone(),
                        vec![0; spec.physical_bytes().unwrap() as usize],
                    )
                })
                .collect(),
        );
        let composition = ReferenceVNextComposition::create_with_capacity(
            id("device.http-output-decoupling"),
            64 * 1024 * 1024,
        )
        .unwrap();
        let policy = TensorComponentPolicy::new(
            64 * 1024 * 1024,
            composition
                .runtime()
                .descriptor()
                .dynamic_storage_profiles
                .iter()
                .copied()
                .collect(),
        )
        .unwrap();
        let component = CompiledTensorComponent::compile(
            family,
            composition.registry(),
            composition.catalog(),
            &policy,
            composition.weight_materializers(),
            &decoder.compile_options().unwrap(),
        )
        .unwrap();
        let session = component
            .initialize(
                composition.runtime().clone(),
                host.clone(),
                ComponentWeightSource::Stored(&weights),
                ComponentInitializationConfig {
                    request_id: id("request.http-output.init"),
                    run_id: id("run.http-output.init"),
                    transaction_id: id("transaction.http-output.init"),
                    static_policy: StaticInitializationPolicy::new(1024 * 1024, 64).unwrap(),
                    executor: TensorExecutorConfig::default(),
                },
            )
            .unwrap_or_else(|_| panic!("Reference decoder initialization failed"));
        Self {
            host,
            session: Some(session),
            input: decoder.input_tensor().unwrap(),
            calls,
        }
    }
}
impl ImageGenerator for Decoder {
    fn host_account(&self) -> &HostMemoryAccount {
        &self.host
    }
    fn capabilities(
        &self,
    ) -> Result<ImageGeneratorCapabilities, Box<dyn std::error::Error + Send>> {
        Ok(ImageGeneratorCapabilities {
            maximum_batch_size: 1,
            maximum_width: 32,
            maximum_height: 32,
            dimension_multiple: 32,
        })
    }
    fn generate_batch(
        &mut self,
        requests: Vec<ImageGenerationRequest>,
    ) -> Result<Vec<ImageGenerationOutput>, ImageBatchError> {
        assert_eq!(requests.len(), 1);
        let request = &requests[0];
        let job = self.calls.fetch_add(1, Ordering::SeqCst) + 1;
        let bytes = self
            .host
            .allocate_zeroed(
                HostMemoryClass::InputPreprocess,
                self.input.minimum_storage_bytes().unwrap() as usize,
            )
            .unwrap()
            .freeze();
        let input = ManagedHostTensor::new(
            id(QwenImage21VaeDecoderFamily::INPUT_ID),
            self.input.clone(),
            bytes,
        )
        .unwrap();
        let mut outputs = self
            .session
            .as_mut()
            .unwrap()
            .execute(
                id(format!("run.http-output.{job}")),
                id(format!("request.http-output.{job}")),
                vec![input],
            )
            .unwrap();
        assert_eq!(outputs.len(), 1);
        let rgba = outputs.remove(0);
        assert_eq!(rgba.tensor().dimensions(), [1, 4, 32, 32]);
        Ok(vec![ImageGenerationOutput {
            rgba,
            receipt: ImageGenerationReceipt {
                job: job as u64,
                batch_size: 1,
                batch_index: 0,
                decoder_batch_policy: ImageDecoderBatchPolicy::SingleImage,
                residency: ImageResidencyPolicy::AllModelComponents,
                width: request.width,
                height: request.height,
                seed: request.seed,
                stages: vec![],
                steps: vec![],
                runtime_fingerprint: "actual-reference-vae-lifecycle-fixture".into(),
                native_memory: None,
                assets: ImageAssetReceipt {
                    metadata_sha256: BTreeMap::new(),
                    required_weight_bytes: [0; 3],
                    payload_verification: "zero BF16 test weights; no image quality claim",
                },
                euler_profile: "not-run-decoder-only-fixture",
            },
        }])
    }
    fn shutdown(&mut self) -> Result<(), Box<dyn std::error::Error + Send>> {
        match self
            .session
            .take()
            .unwrap()
            .close()
            .unwrap_or_else(|_| panic!("Reference decoder failed to close"))
        {
            PlanRuntimeCloseOutcome::Closed(_) => Ok(()),
            PlanRuntimeCloseOutcome::Referenced { .. } => {
                panic!("Reference session unexpectedly retained")
            }
        }
    }
}

#[derive(Default)]
struct BlockingPoolGate {
    entered: tokio::sync::Notify,
    released: Mutex<bool>,
    changed: Condvar,
}
struct ReleasePool(Arc<BlockingPoolGate>);
impl Drop for ReleasePool {
    fn drop(&mut self) {
        *self.0.released.lock().unwrap() = true;
        self.0.changed.notify_all();
    }
}
fn request(seed: u64) -> axum::http::Request<Body> {
    axum::http::Request::builder().method("POST").uri("/v1/images/generations").header("content-type", "application/json")
        .body(Body::from(serde_json::json!({"model":"image","prompt":"lifecycle fixture","size":"32x32","seed":seed}).to_string())).unwrap()
}

#[test]
fn next_inference_runs_while_encoding_is_queued_and_an_earlier_body_is_unread() {
    let host = HostMemoryAccount::new(
        HostMemoryLimits::new(
            8 * 1024 * 1024,
            HostMemoryClassLimits {
                input_preprocess_bytes: 1024 * 1024,
                output_bytes: 4 * 1024 * 1024,
                transfer_staging_bytes: 2 * 1024 * 1024,
                weight_backing_bytes: 0,
            },
        )
        .unwrap(),
    );
    let calls = Arc::new(AtomicUsize::new(0));
    let service = ImageService::new(
        Decoder::new(host.clone(), calls.clone()),
        ImageServiceConfig {
            batching: ferrum_types::ImageBatchPolicy::Auto,
            maximum_pending: 4,
            maximum_pending_bytes: 4096,
            maximum_prompt_bytes: 64,
            max_num_seqs: 1,
            request_batch_max_wait_ms: 0,
        },
    )
    .unwrap();
    let mut config = config();
    config.default_width = 32;
    config.default_height = 32;
    config.output.max_sessions = 3;
    let images = ImageHttpService::new("image", service, config).unwrap();
    let router = super::super::super::AxumServer::from_state(
        AppState::default().with_images(images.clone()),
    )
    .build_router()
    .unwrap();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .max_blocking_threads(1)
        .build()
        .unwrap();
    runtime.block_on(async {
        let gate = Arc::new(BlockingPoolGate::default());
        let release = ReleasePool(gate.clone()); // Unblocks the pool even if a test assertion fails.
        let blocked = gate.clone();
        let blocker = tokio::task::spawn_blocking(move || {
            blocked.entered.notify_one();
            let mut released = blocked.released.lock().unwrap();
            while !*released {
                released = blocked.changed.wait(released).unwrap();
            }
        });
        gate.entered.notified().await;
        let first = tokio::spawn(router.clone().oneshot(request(1)));
        let second = tokio::spawn(router.clone().oneshot(request(2)));
        tokio::time::timeout(Duration::from_secs(30), async {
            while images.service().snapshot().executed_requests < 2
                || images.service().snapshot().active_requests != 0
            {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
        // Both actual decodes completed on the service worker. HTTP encoding is
        // still queued in the occupied blocking pool, not running on this reactor.
        assert_eq!(calls.load(Ordering::SeqCst), 2);
        assert_eq!(images.service().snapshot().active_requests, 0);
        assert!(!first.is_finished() && !second.is_finished());
        assert_eq!(images.output_usage().sessions, 2);
        assert!(images.output_usage().encoded_bytes > 0);
        drop(release);
        blocker.await.unwrap();
        let first = tokio::time::timeout(Duration::from_secs(30), first)
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        let second = tokio::time::timeout(Duration::from_secs(30), second)
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        assert!(first.status().is_success() && second.status().is_success());
        drop(second);
        // Keep every byte of the first HTTP body unread. Spare output capacity
        // permits a later inference and encoding without consuming that body.
        assert_eq!(images.output_usage().sessions, 1);
        let third = tokio::time::timeout(Duration::from_secs(30), router.oneshot(request(3)))
            .await
            .unwrap()
            .unwrap();
        assert!(third.status().is_success());
        assert_eq!(calls.load(Ordering::SeqCst), 3);
        let bytes = axum::body::to_bytes(third.into_body(), 64 * 1024)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        let png = base64::engine::general_purpose::STANDARD
            .decode(json["data"][0]["b64_json"].as_str().unwrap())
            .unwrap();
        let mut decoder = png::Decoder::new(Cursor::new(png)).read_info().unwrap();
        let mut pixels = vec![0; 32 * 32 * 4];
        let frame = decoder.next_frame(&mut pixels).unwrap();
        assert_eq!((frame.width, frame.height), (32, 32));
        drop((bytes, first));
        images.shutdown().await.unwrap();
        assert!(!images.take_shutdown().await.unwrap().has_failure());
        assert_eq!(images.output_usage().sessions, 0);
        assert_eq!(images.output_usage().encoded_bytes, 0);
        assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    });
}
