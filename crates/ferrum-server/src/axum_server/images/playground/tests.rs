use super::*;
use axum::{
    body::Body,
    http::{Request, StatusCode},
    Router,
};
use ferrum_engine::image_generation::service::{
    ImageBatchError, ImageGenerator, ImageGeneratorCapabilities, ImageService, ImageServiceConfig,
};
use ferrum_runtime::{HostMemoryAccount, HostMemoryClassLimits, HostMemoryLimits};
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};
use tower::ServiceExt;

struct MetadataGenerator {
    host: HostMemoryAccount,
    edits: bool,
    maximum_reference_images: usize,
    calls: Arc<AtomicUsize>,
}

impl ImageGenerator for MetadataGenerator {
    fn host_account(&self) -> &HostMemoryAccount {
        &self.host
    }
    fn supports_edits(&self) -> bool {
        self.edits
    }
    fn maximum_reference_images(&self) -> usize {
        self.maximum_reference_images
    }
    fn capabilities(
        &self,
    ) -> Result<ImageGeneratorCapabilities, Box<dyn std::error::Error + Send>> {
        Ok(ImageGeneratorCapabilities {
            maximum_batch_size: 1,
            maximum_width: 768,
            maximum_height: 1024,
            dimension_multiple: 128,
        })
    }
    fn generate_batch(
        &mut self,
        _: Vec<ImageGenerationRequest>,
    ) -> Result<Vec<ferrum_engine::image_generation::ImageGenerationOutput>, ImageBatchError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        Err(ImageBatchError::Rejected(
            "metadata-only fixture cannot generate".into(),
        ))
    }
}

fn router(state: AppState) -> Router {
    crate::axum_server::AxumServer::from_state(state)
        .build_router()
        .unwrap()
}

async fn get(router: &Router, path: &str) -> Response {
    router
        .clone()
        .oneshot(Request::builder().uri(path).body(Body::empty()).unwrap())
        .await
        .unwrap()
}

#[tokio::test]
async fn playground_routes_require_a_real_configured_image_service() {
    let router = router(AppState::default());
    for path in [
        "/playground",
        "/playground/config",
        "/playground/app.js",
        "/playground/style.css",
    ] {
        let response = get(&router, path).await;
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        let body = axum::body::to_bytes(response.into_body(), 4096)
            .await
            .unwrap();
        let error: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(error["error"]["type"], "service_unavailable_error");
    }
}

#[tokio::test]
async fn playground_config_uses_declared_capabilities_without_queueing_inference() {
    for maximum_reference_images in [0, 1, 3, 8, 10] {
        let edits = maximum_reference_images != 0;
        let host = HostMemoryAccount::new(
            HostMemoryLimits::new(
                4096,
                HostMemoryClassLimits {
                    input_preprocess_bytes: 4096,
                    output_bytes: 4096,
                    transfer_staging_bytes: 0,
                    weight_backing_bytes: 0,
                },
            )
            .unwrap(),
        );
        let calls = Arc::new(AtomicUsize::new(0));
        let service = ImageService::new(
            MetadataGenerator {
                host: host.clone(),
                edits,
                maximum_reference_images,
                calls: calls.clone(),
            },
            ImageServiceConfig {
                batching: ferrum_types::ImageBatchPolicy::Configured,
                maximum_pending: 2,
                maximum_pending_bytes: 4096,
                maximum_prompt_bytes: 64,
                max_num_seqs: 1,
                request_batch_max_wait_ms: 0,
            },
        )
        .unwrap();
        let mut config = super::super::tests::config();
        config.default_width = 640;
        config.default_height = 512;
        config.default_seed = u64::MAX;
        let images = ImageHttpService::new("declared-image-alias", service, config).unwrap();
        let router = router(AppState::default().with_images(images.clone()));
        let response = get(&router, "/playground/config").await;
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(response.headers()[header::CACHE_CONTROL], "no-store");
        let body = axum::body::to_bytes(response.into_body(), 64 * 1024)
            .await
            .unwrap();
        let value: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(value["model"], "declared-image-alias");
        assert_eq!(value["default_width"], 640);
        assert_eq!(value["default_height"], 512);
        assert_eq!(value["default_seed"].as_u64(), Some(u64::MAX));
        assert_eq!(value["supports_edits"], edits);
        assert_eq!(value["maximum_reference_images"], maximum_reference_images);
        assert_eq!(
            value["maximum_reference_image_bytes"],
            ImageInputLimits::default().maximum_encoded_bytes()
        );
        let sizes = value["sizes"].as_array().unwrap();
        let dimensions: Vec<(u32, u32)> = sizes
            .iter()
            .map(|size| {
                (
                    size["width"].as_u64().unwrap() as u32,
                    size["height"].as_u64().unwrap() as u32,
                )
            })
            .collect();
        assert!(dimensions.contains(&(640, 512)));
        assert!(dimensions.contains(&(512, 384)));
        assert!(dimensions.contains(&(768, 1024)));
        assert!(!dimensions.contains(&(768, 576)));
        assert!(!dimensions.contains(&(1024, 1024)));
        for &(width, height) in &dimensions {
            images
                .service
                .validate_request(&ImageGenerationRequest {
                    prompt: String::new(),
                    width,
                    height,
                    seed: 0,
                    n: 1,
                })
                .unwrap();
        }
        for (path, content_type) in [
            ("/playground", "text/html"),
            ("/playground/app.js", "text/javascript"),
            ("/playground/style.css", "text/css"),
        ] {
            let response = get(&router, path).await;
            assert_eq!(response.status(), StatusCode::OK);
            assert!(response.headers()[header::CONTENT_TYPE]
                .to_str()
                .unwrap()
                .starts_with(content_type));
            assert!(!axum::body::to_bytes(response.into_body(), 1024 * 1024)
                .await
                .unwrap()
                .is_empty());
        }
        assert_eq!(calls.load(Ordering::SeqCst), 0);
        assert_eq!(images.service.snapshot().executed_batches, 0);
        images.shutdown().await.unwrap();
        assert!(!images.take_shutdown().await.unwrap().has_failure());
        drop((router, images));
        assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    }
}
