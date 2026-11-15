use super::*;

fn png() -> Vec<u8> {
    let mut bytes = Vec::new();
    {
        let mut encoder = png::Encoder::new(&mut bytes, 2, 2);
        encoder.set_color(png::ColorType::Rgb);
        encoder.set_depth(png::BitDepth::Eight);
        let mut writer = encoder.write_header().unwrap();
        writer
            .write_image_data(&[255, 0, 0, 0, 255, 0, 0, 0, 255, 127, 128, 129])
            .unwrap();
    }
    bytes
}
fn wire(image: &[u8]) -> serde_json::Value {
    serde_json::json!({"model":"video","prompt":"A quiet lake","seed":42,"width":832,"height":480,"num_frames":124,"fps":24,"image_reference":[{"image_url":format!("data:image/png;base64,{}",STANDARD.encode(image))}]})
}
fn parse_value(
    value: serde_json::Value,
    host: &HostMemoryAccount,
    config: &VideoHttpConfig,
) -> Result<VideoGenerationRequest, ServerError> {
    let value = serde_json::to_vec(&value).unwrap();
    let mut body = host
        .allocate_zeroed(HostMemoryClass::InputPreprocess, value.len())
        .unwrap();
    body.as_mut().copy_from_slice(&value);
    parse(body.freeze(), host, config, "video")
}

fn wave(samples: &[f32]) -> Vec<u8> {
    let mut b = Vec::new();
    b.extend(b"RIFF");
    b.extend((36 + samples.len() as u32 * 4).to_le_bytes());
    b.extend(b"WAVEfmt ");
    b.extend(16u32.to_le_bytes());
    b.extend(3u16.to_le_bytes());
    b.extend(1u16.to_le_bytes());
    b.extend(32000u32.to_le_bytes());
    b.extend(128000u32.to_le_bytes());
    b.extend(4u16.to_le_bytes());
    b.extend(32u16.to_le_bytes());
    b.extend(b"data");
    b.extend((samples.len() as u32 * 4).to_le_bytes());
    for sample in samples {
        b.extend(sample.to_le_bytes());
    }
    b
}

#[test]
fn mixed_media_preserves_order_and_rejects_ambiguous_or_invalid_audio_before_queueing() {
    let host = super::super::tests::host();
    let config = super::super::tests::config();
    let image = png();
    let sound = wave(&[0., -0.25, 0.75]);
    let mut value = wire(&image);
    value.as_object_mut().unwrap().remove("image_reference");
    value["references"] = serde_json::json!([
        {"type":"audio","data_url":format!("data:audio/wav;base64,{}", STANDARD.encode(&sound))},
        {"type":"image","data_url":format!("data:image/png;base64,{}", STANDARD.encode(&image))}
    ]);
    let request = parse_value(value.clone(), &host, &config).unwrap();
    assert_eq!(
        request
            .references
            .iter()
            .map(VideoReference::kind)
            .collect::<Vec<_>>(),
        [VideoReferenceKind::Audio, VideoReferenceKind::Image]
    );
    assert_eq!(request.references[0].bytes().bytes(), sound);
    assert_eq!(request.references[1].bytes().bytes(), image);
    drop(request);
    let mut ambiguous = value.clone();
    ambiguous["image_reference"] = wire(&image)["image_reference"].clone();
    assert!(parse_value(ambiguous, &host, &config).is_err());
    let mut only_audio = value.clone();
    only_audio["references"].as_array_mut().unwrap().pop();
    assert!(parse_value(only_audio, &host, &config).is_err());
    for bad in [wave(&[f32::NAN]), vec![0; 64], image] {
        let mut invalid = value.clone();
        invalid["references"][0]["data_url"] =
            serde_json::json!(format!("data:audio/wav;base64,{}", STANDARD.encode(bad)));
        assert!(parse_value(invalid, &host, &config).is_err());
    }
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}
#[test]
fn png_data_url_keeps_original_admitted_owner_and_exact_video_fields() {
    let host = super::super::tests::host();
    let config = super::super::tests::config();
    let png = png();
    let request = parse_value(wire(&png), &host, &config).unwrap();
    assert_eq!(request.parameters.seed, 42);
    assert_eq!(request.parameters.num_frames, 124);
    assert_eq!(
        request.parameters.frame_rate,
        VideoFrameRate {
            numerator: 24,
            denominator: 1
        }
    );
    assert_eq!(request.references[0].bytes().bytes(), png);
    assert!(host.owns(request.references[0].bytes()));
    assert_eq!(
        host.usage().unwrap().allocated_bytes(),
        request.references[0].bytes().backing_bytes()
    );
    drop(request);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}

#[test]
fn video_transport_preserves_typed_bytes_order_and_declared_capacity() {
    let host = super::super::tests::host();
    let mut config = super::super::tests::config();
    config.references.maximum_videos = 1;
    config.references.maximum_total = 3;
    let image = png();
    // Container validity belongs to the worker's codec. This test exercises the
    // wire boundary and exact original payload ownership, not video decoding.
    let container = b"opaque container bytes for the codec";
    let mut value = wire(&image);
    value.as_object_mut().unwrap().remove("image_reference");
    value["references"] = serde_json::json!([
        {"type":"video","data_url":format!("data:video/mp4;base64,{}", STANDARD.encode(container))},
        {"type":"image","data_url":format!("data:image/png;base64,{}", STANDARD.encode(image))}
    ]);
    let request = parse_value(value.clone(), &host, &config).unwrap();
    assert_eq!(request.references[0].kind(), VideoReferenceKind::Video);
    assert_eq!(request.references[0].bytes().bytes(), container);
    assert!(host.owns(request.references[0].bytes()));
    assert_eq!(request.references[1].kind(), VideoReferenceKind::Image);
    drop(request);
    for invalid in [
        "https://example.invalid/video.mp4".to_owned(),
        format!("data:audio/wav;base64,{}", STANDARD.encode(container)),
        format!("data:video/mp4;base64,{}", STANDARD.encode(vec![0; 4097])),
    ] {
        let mut bad = value.clone();
        bad["references"][0]["data_url"] = invalid.into();
        assert!(parse_value(bad, &host, &config).is_err());
    }
    config.references.maximum_videos = 0;
    assert!(parse_value(value, &host, &config).is_err());
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}
#[test]
fn corrupt_oversized_remote_and_unknown_inputs_fail_without_retained_payload() {
    let host = super::super::tests::host();
    let config = super::super::tests::config();
    let mut corrupt = png();
    let n = corrupt.len();
    corrupt[n - 1] ^= 1;
    assert!(parse_value(wire(&corrupt), &host, &config).is_err());
    let mut value = wire(&png());
    value["image_reference"][0]["image_url"] = serde_json::json!("https://example.com/image.png");
    assert!(parse_value(value, &host, &config).is_err());
    let mut value = wire(&png());
    value["unimplemented_control"] = serde_json::json!(true);
    assert!(parse_value(value, &host, &config).is_err());
    let mut value = wire(&png());
    value["prompt"] = serde_json::json!("x".repeat(config.maximum_prompt_bytes + 1));
    assert!(parse_value(value, &host, &config).is_err());
    let mut bounded = config;
    bounded.image_limits = ImageInputLimits::new(4096, 1, 1, 1, 16 * 1024).unwrap();
    assert!(parse_value(wire(&png()), &host, &bounded).is_err());
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}

#[tokio::test]
async fn chunked_body_enforces_limit_and_shutdown_releases_partial_ingress() {
    use axum::body::{Body, Bytes};
    let host = super::super::tests::host();
    let (signal, closing) = tokio::sync::watch::channel(false);
    let chunks = futures::stream::iter([
        Ok::<_, std::io::Error>(Bytes::from_static(b"12345")),
        Ok(Bytes::from_static(b"6789")),
    ]);
    let request = Request::builder()
        .header("content-type", "application/json")
        .body(Body::from_stream(chunks))
        .unwrap();
    assert!(read_body(request, &host, 8, closing).await.is_err());
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    let (tx, rx) = tokio::sync::mpsc::channel::<Result<Bytes, std::io::Error>>(1);
    tx.send(Ok(Bytes::from_static(b"{\"prompt\":")))
        .await
        .unwrap();
    let request = Request::builder()
        .header("content-type", "application/json")
        .body(Body::from_stream(
            tokio_stream::wrappers::ReceiverStream::new(rx),
        ))
        .unwrap();
    let read = read_body(request, &host, 1024, signal.subscribe());
    tokio::pin!(read);
    assert!(futures::poll!(&mut read).is_pending());
    assert!(host.usage().unwrap().committed_bytes() > 0);
    signal.send_replace(true);
    assert!(read.await.is_err());
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}
