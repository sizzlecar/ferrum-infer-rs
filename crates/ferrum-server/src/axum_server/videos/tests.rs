use super::*;
use axum::{body::Body, http::Request};
use ferrum_runtime::{HostMemoryAccount, HostMemoryClassLimits, HostMemoryLimits};
use tower::ServiceExt;

pub(super) fn host() -> HostMemoryAccount {
    HostMemoryAccount::new(
        HostMemoryLimits::new(
            1024 * 1024,
            HostMemoryClassLimits {
                input_preprocess_bytes: 1024 * 1024,
                output_bytes: 1024 * 1024,
                transfer_staging_bytes: 0,
                weight_backing_bytes: 0,
            },
        )
        .unwrap(),
    )
}
pub(super) fn config() -> VideoHttpConfig {
    VideoHttpConfig {
        maximum_jobs: 2,
        maximum_body_bytes: 16 * 1024,
        maximum_prompt_bytes: 1024,
        references: VideoReferenceLimits {
            minimum_visual: 1,
            maximum_images: 1,
            maximum_audio: 1,
            maximum_videos: 0,
            maximum_total: 2,
        },
        image_limits: ImageInputLimits::new(4096, 64, 8, 8, 16 * 1024).unwrap(),
        audio_limits: AudioInputLimits {
            maximum_encoded_bytes: 4096,
            maximum_decoded_bytes: 16 * 1024,
        },
        maximum_readers: 1,
        read_buffer_bytes: 256,
    }
}

#[tokio::test]
async fn actual_router_returns_unavailable_without_a_video_factory() {
    let router = super::super::AxumServer::from_state(AppState::default())
        .build_router()
        .unwrap();
    let id = uuid::Uuid::new_v4();
    for (method, path) in [
        ("POST", "/v1/videos".to_owned()),
        ("GET", format!("/v1/videos/{id}")),
        ("GET", format!("/v1/videos/{id}/content")),
        ("DELETE", format!("/v1/videos/{id}")),
    ] {
        let response = router
            .clone()
            .oneshot(
                Request::builder()
                    .method(method)
                    .uri(path)
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        let body = axum::body::to_bytes(response.into_body(), 4096)
            .await
            .unwrap();
        assert!(String::from_utf8_lossy(&body).contains("video generator not loaded"));
    }
}

#[test]
fn video_registry_requires_an_actual_service_and_declares_both_outputs() {
    assert_eq!(
        ServedModelKind::VideoGeneration.modalities(),
        &["video", "audio"]
    );
    let registry = ServedModelRegistry::try_new(
        "video",
        ServedModelKind::VideoGeneration,
        vec!["video".into()],
        vec![],
    )
    .unwrap();
    let state = AppState::default().with_served_model_registry(registry);
    assert!(super::super::AxumServer::from_state(state)
        .build_router()
        .is_err());
    let mut config = config();
    config.maximum_jobs = 0;
    assert!(config.validate().is_err());
}
