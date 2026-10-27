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
    assert_eq!(
        account.usage().unwrap().allocated_bytes(),
        ImageInputLimits::default().maximum_encoded_bytes()
    );
    assert_eq!(
        image.backing_bytes(),
        ImageInputLimits::default().maximum_encoded_bytes()
    );
    assert!(image.backing_bytes() > image.len());
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
    decoded: Arc<std::sync::Mutex<Option<(u32, u32, usize)>>>,
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
            Ok(image) => {
                *self.decoded.lock().unwrap() =
                    Some((image.width(), image.height(), image.bytes().len()));
                Err(ImageBatchError::Rejected(
                    "PNG decoded; fixture has no model executor".into(),
                ))
            }
        }
    }
}

#[tokio::test]
async fn two_k_png_round_trip_and_invalid_inputs_use_the_actual_router() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use tower::ServiceExt;
    // The encoded range retains its maximum backing while decoded RGBA and
    // the admitted prompt/queue metadata are live. This test exercises codec
    // limits; queue Host pressure has a separate HTTP 429 regression.
    let encoded_limit = ImageInputLimits::default().maximum_encoded_bytes();
    let rgba_bytes = 2048 * 2048 * 4;
    let queue_bytes = 4096;
    let output_bytes = 2 * 1024 * 1024;
    let host = HostMemoryAccount::new(
        HostMemoryLimits::new(
            encoded_limit + rgba_bytes + queue_bytes + output_bytes,
            HostMemoryClassLimits {
                input_preprocess_bytes: encoded_limit + rgba_bytes + queue_bytes,
                output_bytes,
                transfer_staging_bytes: 0,
                weight_backing_bytes: 0,
            },
        )
        .unwrap(),
    );
    let calls = Arc::new(AtomicUsize::new(0));
    let decoded = Arc::new(std::sync::Mutex::new(None));
    let service = ImageService::new(
        DecoderOnly {
            host: host.clone(),
            calls: calls.clone(),
            decoded: decoded.clone(),
        },
        ferrum_engine::image_generation::service::ImageServiceConfig {
            batching: ferrum_types::ImageBatchPolicy::Configured,
            maximum_pending: 2,
            maximum_pending_bytes: encoded_limit + queue_bytes,
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
        PngDimensions::new(2048, 2048).unwrap(),
        PngLimits::default(),
        |writer| {
            let row = [1, 2, 3, 255].repeat(2048);
            for _ in 0..2048 {
                writer.write_all(&row)?;
            }
            Ok(())
        },
    )
    .unwrap();
    assert!(png.len() > rgba_bytes);
    assert!(png.len() <= encoded_limit);
    let held_output = host.reserve(HostMemoryClass::Output, output_bytes).unwrap();
    let waiting = tokio::spawn(
        router.clone().oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/images/edits")
                .header("content-type", "multipart/form-data; boundary=bound")
                .body(Body::from(wire(&[
                    ("model", b"image", false),
                    ("prompt", b"edit", false),
                    ("image", &png, true),
                ])))
                .unwrap(),
        ),
    );
    tokio::time::timeout(std::time::Duration::from_secs(5), async {
        while images.service().snapshot().pending_requests == 0 {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    assert!(!waiting.is_finished());
    assert_eq!(images.output_usage().sessions, 1);
    assert_eq!(images.output_usage().encoded_bytes, 0);
    // Cancellation removes this cheap pending job even while output stays full.
    waiting.abort();
    assert!(waiting.await.unwrap_err().is_cancelled());
    tokio::time::timeout(std::time::Duration::from_secs(5), async {
        while images.service().snapshot().pending_requests != 0 {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    assert_eq!(images.output_usage().sessions, 0);
    assert_eq!(host.usage().unwrap().committed_bytes(), output_bytes);
    drop(held_output);
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
        assert_eq!(
            *decoded.lock().unwrap(),
            (index == 2).then_some((2048, 2048, rgba_bytes))
        );
    }
    // Both transport and encoded-image caps reject before worker submission.
    // Large fixtures use one Body chunk, avoiding millions of tiny fragments.
    let envelope_bytes = wire(&[
        ("model", b"image", false),
        ("prompt", b"edit", false),
        ("image", b"", true),
    ])
    .len();
    for encoded_rejection in [true, false] {
        let payload_bytes = if encoded_rejection {
            encoded_limit + 1
        } else {
            EDIT_BODY_LIMIT + 1 - envelope_bytes
        };
        let mut bytes = vec![0; payload_bytes];
        bytes[..8].copy_from_slice(b"\x89PNG\r\n\x1a\n");
        let body = wire(&[
            ("model", b"image", false),
            ("prompt", b"edit", false),
            ("image", &bytes, true),
        ]);
        if !encoded_rejection {
            assert_eq!(body.len(), EDIT_BODY_LIMIT + 1);
        }
        assert_eq!(body.len() <= EDIT_BODY_LIMIT, encoded_rejection);
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
        assert!(response.status().is_client_error());
        let response = axum::body::to_bytes(response.into_body(), 64 * 1024)
            .await
            .unwrap();
        let message = String::from_utf8(response.to_vec()).unwrap();
        assert_eq!(
            message.contains(&format!("encoded image exceeds {encoded_limit} bytes")),
            encoded_rejection,
            "{message}"
        );
        assert_eq!(calls.load(Ordering::SeqCst), 3);
        assert!(images.service().is_accepting());
        assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    }
    images.shutdown().await.unwrap();
    assert!(!images.take_shutdown().await.unwrap().has_failure());
}
