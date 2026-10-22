use super::*;
use base64::{engine::general_purpose::STANDARD, Engine};

fn png(color: png::ColorType, pixels: &[u8]) -> Vec<u8> {
    let mut bytes = Vec::new();
    {
        let mut encoder = png::Encoder::new(&mut bytes, 2, 1);
        encoder.set_color(color);
        encoder.set_depth(png::BitDepth::Eight);
        let mut writer = encoder.write_header().unwrap();
        writer.write_image_data(pixels).unwrap();
        writer.finish().unwrap();
    }
    bytes
}

fn response(bytes: &[u8]) -> Vec<u8> {
    serde_json::to_vec(&serde_json::json!({"data":[{"b64_json":STANDARD.encode(bytes)}]})).unwrap()
}

#[test]
fn decodes_actual_rgb_rgba_pixels_and_hashes_common_rgba8() {
    let limits = ImageResponseLimits::default();
    let rgb = validate_image_response(
        &response(&png(png::ColorType::Rgb, &[0, 127, 255, 1, 2, 3])),
        2,
        1,
        &limits,
    )
    .unwrap();
    let rgba = [0, 127, 255, 255, 1, 2, 3, 255];
    let decoded =
        validate_image_response(&response(&png(png::ColorType::Rgba, &rgba)), 2, 1, &limits)
            .unwrap();
    assert_eq!(rgb.rgba8_sha256, sha256(&rgba));
    assert_eq!(rgb.rgba8_sha256, decoded.rgba8_sha256);
    assert_ne!(rgb.png_sha256, decoded.png_sha256);
    assert_eq!(decoded.rgba8_bytes, 8);
}

#[test]
fn rejects_corruption_trailing_data_wrong_dimensions_and_capacity() {
    let limits = ImageResponseLimits::default();
    let bytes = png(png::ColorType::Rgba, &[0, 1, 2, 3, 255, 254, 253, 252]);
    assert!(validate_image_response(&response(&bytes), 1, 2, &limits).is_err());
    let mut truncated = bytes.clone();
    truncated.pop();
    assert!(validate_image_response(&response(&truncated), 2, 1, &limits).is_err());
    let mut trailing = bytes.clone();
    trailing.push(0);
    assert!(validate_image_response(&response(&trailing), 2, 1, &limits).is_err());
    let mut corrupt = bytes.clone();
    // IHDR CRC, without changing dimensions or relying on DEFLATE internals.
    corrupt[29] ^= 1;
    assert!(validate_image_response(&response(&corrupt), 2, 1, &limits).is_err());
    let mut small = limits.clone();
    small.maximum_pixels = 1;
    assert!(validate_image_response(&response(&bytes), 2, 1, &small).is_err());
    small = limits.clone();
    small.maximum_png_bytes = bytes.len() - 1;
    assert!(validate_image_response(&response(&bytes), 2, 1, &small).is_err());
    small = limits;
    small.maximum_response_bytes = response(&bytes).len() - 1;
    assert!(validate_image_response(&response(&bytes), 2, 1, &small).is_err());
}

#[test]
fn rejects_noncanonical_wire_and_animation_without_exposing_payload() {
    let limits = ImageResponseLimits::default();
    for body in [
        br#"{"data":[]}"#.as_slice(),
        br#"{"data":[{"b64_json":"secret-invalid!"}]}"#,
        br#"{"data":[{"b64_json":"AAAA","url":"http://unused"}]}"#,
        br#"{"data":[{"b64_json":"AAAA"},{"b64_json":"AAAA"}]}"#,
    ] {
        let error = validate_image_response(body, 2, 1, &limits).unwrap_err();
        assert!(!error.to_string().contains("secret-invalid"));
    }
    let mut animated = Vec::new();
    {
        let mut encoder = png::Encoder::new(&mut animated, 2, 1);
        encoder.set_color(png::ColorType::Rgba);
        encoder.set_depth(png::BitDepth::Eight);
        encoder.set_animated(1, 0).unwrap();
        let mut writer = encoder.write_header().unwrap();
        writer.write_image_data(&[0; 8]).unwrap();
        writer.finish().unwrap();
    }
    assert!(validate_image_response(&response(&animated), 2, 1, &limits).is_err());
}

#[test]
fn workload_identity_is_order_seed_and_geometry_sensitive() {
    let sample = ImageSample {
        id: "one".into(),
        prompt: "a red kite".into(),
        seed: 7,
        width: 2,
        height: 1,
        num_inference_steps: None,
    };
    let mut dataset = ImageDataset {
        schema_version: 1,
        samples: vec![sample.clone()],
    };
    dataset.validate(&ImageResponseLimits::default()).unwrap();
    let original = dataset.fingerprint().unwrap();
    dataset.samples[0].seed += 1;
    assert_ne!(original, dataset.fingerprint().unwrap());
    let request = serde_json::to_value(ImageHttpRequest::new("model", &sample)).unwrap();
    assert_eq!(request["n"], 1);
    assert_eq!(request["response_format"], "b64_json");
    assert_eq!(request["size"], "2x1");
    assert!(request.get("num_inference_steps").is_none());
    dataset.samples.push(dataset.samples[0].clone());
    assert!(dataset.validate(&ImageResponseLimits::default()).is_err());
    assert!(serde_json::from_str::<ImageDataset>(
        r#"{"schema_version":1,"samples":[],"shuffle":true}"#
    )
    .is_err());
}

#[test]
fn metrics_do_not_hide_errors_or_invent_missing_slo() {
    let image = validate_image_response(
        &response(&png(png::ColorType::Rgb, &[0; 6])),
        2,
        1,
        &ImageResponseLimits::default(),
    )
    .unwrap();
    let mut record = ImageRequestRecord {
        sample_index: 0,
        sample_id: "one".into(),
        phase: ImageRequestPhase::Measured,
        repetition: 0,
        dispatch_index: 0,
        http_status: Some(200),
        e2e_ms: 10.0,
        validation_ms: 2.0,
        response_bytes: 100,
        image: Some(image),
        failure: None,
        error: None,
    };
    let report = ImageMetrics::from_records(2.0, &[record.clone()], None);
    assert_eq!(report.images_per_second, Some(0.5));
    assert_eq!(report.latency_slo_status, ImageSloStatus::NotEvaluated);
    assert_eq!(
        ImageMetrics::from_records(2.0, &[record.clone()], Some(10.0)).latency_slo_status,
        ImageSloStatus::Met
    );
    record.failure = Some(ImageFailureKind::HttpStatus);
    record.image = None;
    let failed = ImageMetrics::from_records(2.0, &[record], Some(10.0));
    assert_eq!(failed.errors, 1);
    assert_eq!(failed.e2e_p99_ms, None);
    assert_eq!(failed.latency_slo_status, ImageSloStatus::Violated);
}
