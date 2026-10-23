use super::*;
use axum::{
    body::{Body, Bytes},
    extract::FromRequest,
    http::Request,
};
use ferrum_runtime::{HostMemoryClassLimits, HostMemoryLimits};

fn account(bytes: usize) -> HostMemoryAccount {
    HostMemoryAccount::new(
        HostMemoryLimits::new(
            bytes,
            HostMemoryClassLimits {
                input_preprocess_bytes: bytes,
                output_bytes: 0,
                transfer_staging_bytes: 0,
                weight_backing_bytes: 0,
            },
        )
        .unwrap(),
    )
}
fn wire(fields: &[(&str, &[u8], bool)]) -> Vec<u8> {
    let mut body = Vec::new();
    for &(name, bytes, file) in fields {
        body.extend_from_slice(
            format!(
                "--bound\r\nContent-Disposition: form-data; name=\"{name}\"{}\r\n{}\r\n",
                if file {
                    "; filename=\"../../unused.png\""
                } else {
                    ""
                },
                if file {
                    "Content-Type: image/png\r\n"
                } else {
                    ""
                }
            )
            .as_bytes(),
        );
        body.extend_from_slice(bytes);
        body.extend_from_slice(b"\r\n");
    }
    body.extend_from_slice(b"--bound--\r\n");
    body
}
async fn multipart(bytes: Vec<u8>) -> Multipart {
    // Exercise boundaries and UTF-8 split across arbitrary transport chunks.
    let chunks = bytes
        .chunks(7)
        .map(Bytes::copy_from_slice)
        .collect::<Vec<_>>();
    let body = Body::from_stream(futures::stream::iter(
        chunks.into_iter().map(Ok::<_, std::io::Error>),
    ));
    Multipart::from_request(
        Request::builder()
            .header("content-type", "multipart/form-data; boundary=bound")
            .body(body)
            .unwrap(),
        &(),
    )
    .await
    .unwrap()
}

#[tokio::test]
async fn streaming_fields_preserve_png_owner_and_use_common_image_controls() {
    let account = account(32 * 1024 * 1024);
    let png = b"\x89PNG\r\n\x1a\ntransport-fixture";
    let body = wire(&[
        ("image", png, true),
        ("prompt", "改成蓝色".as_bytes(), false),
        ("model", b"image", false),
        ("size", b"640x512", false),
        ("seed", b"42", false),
    ]);
    let (wire, image) = parse(multipart(body).await, &account).await.unwrap();
    let request = wire
        .convert("image", &super::super::tests::config())
        .unwrap();
    assert_eq!(request.prompt, "改成蓝色");
    assert_eq!(
        (request.width, request.height, request.seed),
        (640, 512, 42)
    );
    assert_eq!(image.bytes(), png);
    assert!(account.owns(&image));
    assert_eq!(image.class(), HostMemoryClass::InputPreprocess);
    // The exact range is exposed; the admitted maximum backing remains charged.
    assert_eq!(account.usage().unwrap().allocated_bytes(), 16 * 1024 * 1024);
    drop(image);
    assert_eq!(account.usage().unwrap().committed_bytes(), 0);
}

#[tokio::test]
async fn duplicate_files_masks_unknown_controls_and_bad_fields_release_ingress() {
    let account = account(32 * 1024 * 1024);
    let png = b"\x89PNG\r\n\x1a\n";
    let common = [
        ("model", b"image".as_slice(), false),
        ("prompt", b"edit".as_slice(), false),
        ("image", png.as_slice(), true),
    ];
    for extra in [
        ("image[]", png.as_slice(), true),
        ("mask", png.as_slice(), true),
        ("prompt", b"again".as_slice(), false),
        ("strength", b"0.5".as_slice(), false),
        ("seed", b"-1".as_slice(), false),
        ("n", b"+1".as_slice(), false),
    ] {
        let mut fields = common.to_vec();
        fields.push(extra);
        assert!(parse(multipart(wire(&fields)).await, &account)
            .await
            .is_err());
        assert_eq!(account.usage().unwrap().committed_bytes(), 0);
    }
    for fields in [
        vec![("model", b"image".as_slice(), false)],
        vec![
            ("model", b"image".as_slice(), false),
            ("prompt", b"edit".as_slice(), false),
            ("image", b"notpng".as_slice(), true),
        ],
    ] {
        assert!(parse(multipart(wire(&fields)).await, &account)
            .await
            .is_err());
        assert_eq!(account.usage().unwrap().committed_bytes(), 0);
    }
}

#[tokio::test]
async fn ingress_capacity_and_text_bounds_fail_before_worker_submission() {
    let small = account(1024);
    assert!(parse(
        multipart(wire(&[("image", b"\x89PNG\r\n\x1a\n", true)])).await,
        &small
    )
    .await
    .is_err());
    assert_eq!(small.usage().unwrap().committed_bytes(), 0);
    let account = account(32 * 1024 * 1024);
    let oversized = vec![b'a'; 1025];
    assert!(parse(
        multipart(wire(&[("model", &oversized, false)])).await,
        &account
    )
    .await
    .is_err());
    assert_eq!(account.usage().unwrap().committed_bytes(), 0);
    let bad_utf8 = [0xff, 0xfe];
    assert!(parse(
        multipart(wire(&[("prompt", &bad_utf8, false)])).await,
        &account
    )
    .await
    .is_err());
}

struct DecoderOnly {
    host: HostMemoryAccount,
    calls: Arc<std::sync::atomic::AtomicUsize>,
}
impl ferrum_engine::image_generation::service::ImageGenerator for DecoderOnly {
    fn host_account(&self) -> &HostMemoryAccount {
        &self.host
    }
    fn supports_edits(&self) -> bool {
        true
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
                dimension_multiple: 32,
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
        Err(
            ferrum_engine::image_generation::service::ImageBatchError::Rejected(
                "fixture has no model executor".into(),
            ),
        )
    }
    fn edit(
        &mut self,
        request: ImageEditRequest,
    ) -> Result<
        ferrum_engine::image_generation::ImageGenerationOutput,
        ferrum_engine::image_generation::service::ImageBatchError,
    > {
        use ferrum_engine::image_generation::service::ImageBatchError;
        use ferrum_models::qwen_image21::image_input::{ImageInputError, ManagedRgba8Image};
        self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        match ManagedRgba8Image::decode_png(&self.host, &request.image, ImageInputLimits::default())
        {
            Err(
                error @ (ImageInputError::Invalid(_)
                | ImageInputError::Unsupported(_)
                | ImageInputError::Limit(_)
                | ImageInputError::Png(_)),
            ) => Err(ImageBatchError::InvalidInput(error.to_string())),
            Err(error) => Err(ImageBatchError::Rejected(error.to_string())),
            // This test verifies HTTP/queue/codec admission, not fabricated
            // native image execution or a forged TensorExecutionOutput.
            Ok(_) => Err(ImageBatchError::Rejected(
                "PNG decoded; fixture has no model executor".into(),
            )),
        }
    }
}

#[tokio::test]
async fn invalid_png_crc_and_pixel_count_return_400_without_stopping_the_worker() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use tower::ServiceExt;
    let host = account(32 * 1024 * 1024);
    let calls = Arc::new(AtomicUsize::new(0));
    let service = ImageService::new(
        DecoderOnly {
            host: host.clone(),
            calls: calls.clone(),
        },
        ferrum_engine::image_generation::service::ImageServiceConfig {
            maximum_pending: 2,
            maximum_prompt_bytes: 1024,
            max_num_seqs: 1,
            request_batch_max_wait_ms: 0,
        },
    )
    .unwrap();
    let images = ImageHttpService::new("image", service, super::super::tests::config()).unwrap();
    assert_eq!(
        images.health()["capabilities"]["maximum_edit_batch_size"],
        1
    );
    let router = super::super::super::AxumServer::from_state(
        AppState::default().with_images(images.clone()),
    )
    .build_router()
    .unwrap();
    let mut png = Vec::new();
    ferrum_output::png::write_rgba8_png(
        &mut png,
        PngDimensions::new(1, 1).unwrap(),
        PngLimits::default(),
        |writer| writer.write_all(&[1, 2, 3, 255]),
    )
    .unwrap();
    let mut crc = png.clone();
    crc[29] ^= 1;
    let mut pixels = png.clone();
    pixels[16..20].copy_from_slice(&8192u32.to_be_bytes());
    pixels[20..24].copy_from_slice(&8192u32.to_be_bytes());
    for (index, bytes) in [crc, pixels, png].into_iter().enumerate() {
        let body = wire(&[
            ("model", b"image", false),
            ("prompt", b"edit", false),
            ("image", &bytes, true),
        ]);
        let response = router
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/images/edits")
                    .header("content-type", "multipart/form-data; boundary=bound")
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(
            response.status(),
            if index < 2 {
                axum::http::StatusCode::BAD_REQUEST
            } else {
                axum::http::StatusCode::INTERNAL_SERVER_ERROR
            }
        );
        drop(response);
        assert!(images.service().is_accepting());
        assert_eq!(host.usage().unwrap().committed_bytes(), 0);
        assert_eq!(calls.load(Ordering::SeqCst), index + 1);
    }
    images.shutdown().await.unwrap();
    assert!(!images.take_shutdown().await.unwrap().has_failure());
}
