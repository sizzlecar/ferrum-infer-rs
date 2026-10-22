use super::*;
use axum::body::Body;
use base64::Engine;
use ferrum_runtime::{
    ElementType, HostMemoryAccount, HostMemoryClass, HostMemoryClassLimits, HostMemoryLimits,
    ManagedHostBytes, ResolvedTensorLayout, ResolvedTensorSpec,
};
use futures::StreamExt;
use std::io::Cursor;
use std::sync::atomic::{AtomicUsize, Ordering};
use tower::ServiceExt;

fn config() -> ImageHttpConfig {
    ImageHttpConfig {
        output: OutputConfig {
            max_sessions: 1,
            ..OutputConfig::default()
        },
        default_width: 512,
        default_height: 512,
        default_seed: 42,
        rgba_scratch_bytes: 8,
        png_limits: PngLimits::default(),
    }
}

fn host_account(total: usize) -> HostMemoryAccount {
    HostMemoryAccount::new(
        HostMemoryLimits::new(
            total,
            HostMemoryClassLimits {
                input_preprocess_bytes: 0,
                output_bytes: total,
                transfer_staging_bytes: 0,
                weight_backing_bytes: 0,
            },
        )
        .unwrap(),
    )
}

fn rgba(host: &HostMemoryAccount) -> (ResolvedTensorSpec, ManagedHostBytes) {
    let mut bytes = host.allocate_zeroed(HostMemoryClass::Output, 32).unwrap();
    let values: [u16; 16] = [
        0x3f80, 0x3f80, 0x3f80, 0x3f80, // R
        0xbf80, 0xbf80, 0xbf80, 0xbf80, // G
        0, 0, 0, 0, // B
        0xbf80, 0x3f80, 0xbf80, 0x3f80, // A
    ];
    for (chunk, bits) in bytes.as_mut().chunks_exact_mut(2).zip(values) {
        chunk.copy_from_slice(&bits.to_le_bytes());
    }
    (
        ResolvedTensorSpec::new(
            vec![1, 4, 2, 2],
            ElementType::Bf16,
            ResolvedTensorLayout::Contiguous,
        )
        .unwrap(),
        bytes.freeze(),
    )
}

#[test]
fn image_wire_has_explicit_defaults_and_rejects_unsupported_controls() {
    let default: ImageRequest =
        serde_json::from_value(serde_json::json!({"model":"image","prompt":"cloud"})).unwrap();
    let request = default.convert("image", &config()).unwrap();
    assert_eq!(
        (request.width, request.height, request.seed, request.n),
        (512, 512, 42, 1)
    );
    let explicit: ImageRequest = serde_json::from_value(serde_json::json!({
        "model":"image","prompt":"cloud","n":1,"size":"640x512","seed":u64::MAX,"response_format":"b64_json"
    })).unwrap();
    let explicit = explicit.convert("image", &config()).unwrap();
    assert_eq!(
        (explicit.width, explicit.height, explicit.seed),
        (640, 512, u64::MAX)
    );
    for (key, value) in [
        ("negative_prompt", serde_json::json!("rain")),
        ("steps", serde_json::json!(8)),
        ("guidance_scale", serde_json::json!(1)),
        ("stream", serde_json::json!(true)),
    ] {
        let mut input = serde_json::json!({"model":"image","prompt":"cloud"});
        input[key] = value;
        assert!(serde_json::from_value::<ImageRequest>(input).is_err());
    }
    for patch in [
        serde_json::json!({"n":0}),
        serde_json::json!({"n":2}),
        serde_json::json!({"response_format":"url"}),
        serde_json::json!({"model":"other"}),
        serde_json::json!({"size":"0x512"}),
        serde_json::json!({"size":"+512x512"}),
        serde_json::json!({"size":"512X512"}),
        serde_json::json!({"size":"512x512x1"}),
        serde_json::json!({"size":"4294967296x512"}),
        serde_json::json!({"size":"16385x1"}),
    ] {
        let mut input = serde_json::json!({"model":"image","prompt":"cloud"});
        input
            .as_object_mut()
            .unwrap()
            .extend(patch.as_object().unwrap().clone());
        let request: ImageRequest = serde_json::from_value(input).unwrap();
        assert!(request.convert("image", &config()).is_err());
    }
}

#[tokio::test]
async fn png_base64_body_retains_host_and_output_credit_until_last_byte_owner() {
    let host = host_account(4096);
    let (tensor, raw) = rgba(&host);
    let budget = OutputBudget::new(config().output).unwrap();
    let session = budget.try_open_session().unwrap();
    let bytes = response::encode(
        session.clone(),
        host.clone(),
        tensor,
        raw.clone(),
        PngLimits::default(),
        8,
    )
    .await
    .unwrap();
    let length = bytes.len();
    assert_eq!(host.usage().unwrap().allocated_bytes(), raw.len() + length);
    let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    let png = base64::engine::general_purpose::STANDARD
        .decode(json["data"][0]["b64_json"].as_str().unwrap())
        .unwrap();
    let mut reader = png::Decoder::new(Cursor::new(png)).read_info().unwrap();
    let mut pixels = [0; 16];
    let info = reader.next_frame(&mut pixels).unwrap();
    assert_eq!(
        (info.width, info.height, info.color_type),
        (2, 2, png::ColorType::Rgba)
    );
    assert_eq!(
        pixels,
        [255, 0, 128, 0, 255, 0, 128, 255, 255, 0, 128, 0, 255, 0, 128, 255]
    );
    drop(raw);
    let mut body =
        super::super::output::retained_body(session, Body::from(bytes)).into_data_stream();
    let frame = body.next().await.unwrap().unwrap();
    let cloned = frame.clone();
    let sliced = frame.slice(1..4);
    assert!(body.next().await.is_none());
    assert!(budget.try_open_session().is_err());
    drop(body);
    drop(frame);
    drop(cloned);
    assert_eq!(host.usage().unwrap().allocated_bytes(), length);
    assert_eq!(budget.usage().encoded_bytes, length);
    assert!(budget.try_open_session().is_err());
    drop(sliced);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    assert_eq!(budget.usage().encoded_bytes, 0);
    assert_eq!(budget.usage().sessions, 0);
}

#[tokio::test]
async fn encoding_capacity_failure_returns_all_intermediate_owners() {
    // Enough Host memory for raw pixels and scratch, but not the final JSON.
    let host = host_account(40);
    let (tensor, raw) = rgba(&host);
    let budget = OutputBudget::new(config().output).unwrap();
    let result = response::encode(
        budget.try_open_session().unwrap(),
        host.clone(),
        tensor,
        raw.clone(),
        PngLimits::default(),
        8,
    )
    .await;
    assert!(matches!(result, Err(ServerError::CapacityExceeded(_))));
    assert_eq!(host.usage().unwrap().allocated_bytes(), 32);
    assert_eq!(budget.usage().encoded_bytes, 0);
    assert_eq!(budget.usage().sessions, 0);
    drop(raw);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}

#[tokio::test]
async fn encoded_limit_and_nonfinite_or_foreign_pixels_reject_without_leaking() {
    let host = host_account(4096);
    let (tensor, raw) = rgba(&host);
    let budget = OutputBudget::new(OutputConfig {
        encoded_bytes_total: 32,
        max_event_bytes: 32,
        ..config().output
    })
    .unwrap();
    assert!(matches!(
        response::encode(
            budget.try_open_session().unwrap(),
            host.clone(),
            tensor.clone(),
            raw.clone(),
            PngLimits::default(),
            8
        )
        .await,
        Err(ServerError::CapacityExceeded(_))
    ));
    assert_eq!(host.usage().unwrap().allocated_bytes(), 32);
    assert_eq!(budget.usage().sessions, 0);
    let foreign = host_account(4096);
    assert!(response::encode(
        budget.try_open_session().unwrap(),
        foreign.clone(),
        tensor.clone(),
        raw.clone(),
        PngLimits::default(),
        8
    )
    .await
    .is_err());
    assert_eq!(foreign.usage().unwrap().committed_bytes(), 0);
    let mut invalid = host.allocate_zeroed(HostMemoryClass::Output, 32).unwrap();
    invalid.as_mut()[..2].copy_from_slice(&0x7fc0_u16.to_le_bytes());
    assert!(response::encode(
        budget.try_open_session().unwrap(),
        host.clone(),
        tensor,
        invalid.freeze(),
        PngLimits::default(),
        8
    )
    .await
    .is_err());
    assert_eq!(host.usage().unwrap().allocated_bytes(), 32);
    assert_eq!(budget.usage().sessions, 0);
}

#[tokio::test]
async fn absent_image_service_is_unavailable_and_bad_json_is_rejected() {
    let router = super::super::AxumServer::from_state(AppState::default())
        .build_router()
        .unwrap();
    for (body, status) in [
        (
            serde_json::json!({"model":"image","prompt":"cloud"}),
            axum::http::StatusCode::SERVICE_UNAVAILABLE,
        ),
        (
            serde_json::json!({"model":"image","prompt":"cloud","steps":8}),
            axum::http::StatusCode::BAD_REQUEST,
        ),
    ] {
        let response = router
            .clone()
            .oneshot(
                axum::http::Request::builder()
                    .method("POST")
                    .uri("/v1/images/generations")
                    .header("content-type", "application/json")
                    .body(Body::from(body.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), status);
    }
}

#[test]
fn image_registry_does_not_enable_lora_or_text_routes() {
    let registry = ServedModelRegistry::try_new(
        "image",
        ServedModelKind::ImageGeneration,
        vec!["image".into()],
        vec![],
    )
    .unwrap();
    assert!(registry.resolve("image", ServedModelKind::Llm).is_none());
    assert_eq!(
        registry
            .resolve("image", ServedModelKind::ImageGeneration)
            .unwrap()
            .kind()
            .modalities(),
        &["image"]
    );
    let adapter = crate::model_registry::LoraAdapterModel::new(
        "adapter",
        "adapter-model",
        "/weights/adapter",
    );
    assert!(ServedModelRegistry::try_new(
        "image",
        ServedModelKind::ImageGeneration,
        vec!["image".into()],
        vec![adapter.clone()]
    )
    .is_err());
    assert!(registry
        .try_with_lora_adapters("image", vec![adapter])
        .is_err());
}

// This backend intentionally never succeeds; these are HTTP admission, status
// and failure-retirement tests, not GPU/model-generation evidence.
struct FailingGenerator {
    host: HostMemoryAccount,
    calls: Arc<AtomicUsize>,
    gate: Option<Arc<GenerationGate>>,
}
#[derive(Default)]
struct GenerationGate {
    started: tokio::sync::Notify,
    released: std::sync::Mutex<bool>,
    changed: std::sync::Condvar,
}
impl ferrum_engine::image_generation::service::ImageGenerator for FailingGenerator {
    fn host_account(&self) -> &HostMemoryAccount {
        &self.host
    }
    fn capabilities(
        &self,
    ) -> Result<
        ferrum_engine::image_generation::service::ImageGeneratorCapabilities,
        Box<dyn std::error::Error + Send>,
    > {
        Ok(
            ferrum_engine::image_generation::service::ImageGeneratorCapabilities {
                maximum_batch_size: 1,
                maximum_width: 1024,
                maximum_height: 1024,
                dimension_multiple: 64,
            },
        )
    }
    fn generate_batch(
        &mut self,
        _: Vec<ImageGenerationRequest>,
    ) -> Result<
        Vec<ferrum_engine::image_generation::ImageGenerationOutput>,
        ferrum_engine::image_generation::service::ImageBatchError,
    > {
        self.calls.fetch_add(1, Ordering::SeqCst);
        if let Some(gate) = &self.gate {
            gate.started.notify_one();
            let mut released = gate.released.lock().unwrap();
            while !*released {
                released = gate.changed.wait(released).unwrap();
            }
        }
        Err(
            ferrum_engine::image_generation::service::ImageBatchError::Fatal(Box::new(
                std::io::Error::other("intentional CPU generator failure"),
            )),
        )
    }
}

fn image_service() -> (ImageHttpService, Arc<AtomicUsize>) {
    image_service_with_gate(None)
}

fn image_service_with_gate(
    gate: Option<Arc<GenerationGate>>,
) -> (ImageHttpService, Arc<AtomicUsize>) {
    let calls = Arc::new(AtomicUsize::new(0));
    let service = ImageService::new(
        FailingGenerator {
            host: host_account(4096),
            calls: calls.clone(),
            gate,
        },
        ferrum_engine::image_generation::service::ImageServiceConfig {
            maximum_pending: 2,
            maximum_prompt_bytes: 64,
            max_num_seqs: 1,
            request_batch_max_wait_ms: 0,
        },
    )
    .unwrap();
    (
        ImageHttpService::new("image", service, config()).unwrap(),
        calls,
    )
}

async fn json_get(router: axum::Router, path: &str) -> serde_json::Value {
    let response = router
        .oneshot(
            axum::http::Request::builder()
                .uri(path)
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert!(response.status().is_success());
    serde_json::from_slice(
        &axum::body::to_bytes(response.into_body(), 64 * 1024)
            .await
            .unwrap(),
    )
    .unwrap()
}

#[tokio::test]
async fn image_only_health_models_and_validation_do_not_claim_a_ready_text_engine() {
    let (images, calls) = image_service();
    let state = AppState::default().with_images(images.clone());
    let router = super::super::AxumServer::from_state(state)
        .build_router()
        .unwrap();
    let health = json_get(router.clone(), "/health").await;
    assert_eq!(health["engine"]["text_loaded"], false);
    assert_eq!(health["engine"]["text_ready"], false);
    assert_eq!(health["images"]["ready"], true);
    assert_eq!(health["images"]["scheduler"]["max_num_seqs"], 1);
    assert_eq!(health["images"]["host"]["allocated_bytes"], 0);
    assert_eq!(health["images"]["host"]["committed_bytes"], 0);
    // An image-only service must not present fallback text configuration as
    // evidence about its actual backend or resource authority. Images retain
    // their separate observed queue/Host state, without invented CUDA fields.
    for text_only in [
        "config",
        "auto_config",
        "admission",
        "scheduler",
        "numerical_execution",
        "kv_storage",
        "cache",
        "execution_attribution",
        "lora",
        "reasoning_protocol",
    ] {
        assert!(
            health.get(text_only).is_none(),
            "unexpected {text_only}: {health}"
        );
    }
    let models = json_get(router.clone(), "/v1/models").await;
    assert_eq!(models["data"][0]["id"], "image");
    assert_eq!(
        models["data"][0]["modalities"],
        serde_json::json!(["image"])
    );
    assert!(models["data"][0].get("max_model_len").is_none());
    assert!(models["data"][0].get("reasoning").is_none());
    let held = images.output.try_open_session().unwrap();
    let response = router
        .clone()
        .oneshot(
            axum::http::Request::builder()
                .method("POST")
                .uri("/v1/images/generations")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"model":"image","prompt":"x"}"#))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), axum::http::StatusCode::TOO_MANY_REQUESTS);
    drop(held);
    for body in [
        serde_json::json!({"model":"image","prompt":"x","size":"513x512"}),
        serde_json::json!({"model":"image","prompt":"x","size":"2048x512"}),
        serde_json::json!({"model":"image","prompt":"x","n":2}),
        serde_json::json!({"model":"image","prompt":"x","response_format":"url"}),
        serde_json::json!({"model":"image","prompt":"x".repeat(65)}),
    ] {
        let response = router
            .clone()
            .oneshot(
                axum::http::Request::builder()
                    .method("POST")
                    .uri("/v1/images/generations")
                    .header("content-type", "application/json")
                    .body(Body::from(body.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), axum::http::StatusCode::BAD_REQUEST);
    }
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    assert_eq!(images.output_usage().sessions, 0);
    images.shutdown().await.unwrap();
    assert!(images.take_shutdown().await.is_some());
    let health = json_get(router, "/health").await;
    assert_eq!(health["images"]["ready"], false);
    assert_eq!(health["status"], "unhealthy");
}

#[tokio::test]
async fn generation_failure_keeps_original_owner_through_http_shutdown() {
    let (images, calls) = image_service();
    let router =
        super::super::AxumServer::from_state(AppState::default().with_images(images.clone()))
            .build_router()
            .unwrap();
    let response = router
        .oneshot(
            axum::http::Request::builder()
                .method("POST")
                .uri("/v1/images/generations")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"model":"image","prompt":"x"}"#))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(
        response.status(),
        axum::http::StatusCode::INTERNAL_SERVER_ERROR
    );
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert_eq!(images.output_usage().sessions, 0);
    assert!(images.shutdown().await.is_err());
    assert!(images.shutdown().await.is_err());
    let retained = images.take_shutdown().await.unwrap();
    assert!(retained.has_failure());
    let (generator, failure) = retained.into_parts();
    let Some(ferrum_engine::image_generation::service::ImageWorkerFailure::Generation(error)) =
        failure
    else {
        panic!("original generation error was not retained");
    };
    assert!(error.downcast_ref::<std::io::Error>().is_some());
    drop(error);
    drop(generator);
    assert!(images.take_shutdown().await.is_none());
    assert_eq!(
        images
            .service()
            .host_account()
            .usage()
            .unwrap()
            .committed_bytes(),
        0
    );
}

#[tokio::test]
async fn incompatible_existing_registry_is_a_startup_error() {
    let (images, _) = image_service();
    let text_registry =
        ServedModelRegistry::try_new("text", ServedModelKind::Llm, vec!["text".into()], vec![])
            .unwrap();
    let state = AppState::default()
        .with_served_model_registry(text_registry)
        .with_images(images.clone());
    assert!(super::super::AxumServer::from_state(state)
        .build_router()
        .is_err());
    images.shutdown().await.unwrap();
    drop(images.take_shutdown().await);
}

#[tokio::test]
async fn cancelling_shutdown_observation_keeps_original_join_and_failure_owner() {
    let gate = Arc::new(GenerationGate::default());
    let (images, _) = image_service_with_gate(Some(gate.clone()));
    let job = images
        .service()
        .try_submit(ImageGenerationRequest {
            prompt: "x".into(),
            width: 512,
            height: 512,
            n: 1,
            seed: 0,
        })
        .unwrap();
    gate.started.notified().await;
    {
        let observation = images.shutdown();
        futures::pin_mut!(observation);
        assert!(futures::poll!(observation).is_pending());
    }
    *gate.released.lock().unwrap() = true;
    gate.changed.notify_all();
    assert!(job.wait().await.is_err());
    assert!(images.shutdown().await.is_err());
    let owner = images
        .take_shutdown()
        .await
        .expect("original join outcome remains accessible");
    let (generator, failure) = owner.into_parts();
    assert!(matches!(
        &failure,
        Some(ferrum_engine::image_generation::service::ImageWorkerFailure::Generation(_))
    ));
    drop(failure);
    drop(generator);
}
