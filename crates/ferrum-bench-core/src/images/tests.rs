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
fn retained_png_is_the_validated_original_encoding() {
    let limits = ImageResponseLimits::default();
    let original = png(png::ColorType::Rgb, &[0, 127, 255, 1, 2, 3]);
    let wire = response(&original);
    let (image, retained) = validate_image_response_with_png(&wire, 2, 1, &limits)
        .unwrap()
        .into_parts();
    assert_eq!(retained, original);
    assert_eq!(image.png_sha256, sha256(&retained));
    let summary = validate_image_response(&wire, 2, 1, &limits).unwrap();
    assert_eq!(image.rgba8_sha256, summary.rgba8_sha256);
    assert_eq!(image.png_bytes, summary.png_bytes);
    let mut invalid_png = original;
    invalid_png.push(0);
    assert!(validate_image_response_with_png(&response(&invalid_png), 2, 1, &limits).is_err());
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
        mode: None,
    };
    let mut dataset = ImageDataset {
        schema_version: 1,
        mode: ImageRequestMode::Generate,
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
    assert_eq!(report.e2e_p95_ms, Some(10.0));
    assert_eq!(report.dispatched, 1);
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

#[test]
fn rejection_and_unsent_arrivals_remain_failures_without_success_percentiles() {
    let record = |index, status, failure| ImageRequestRecord {
        sample_index: index,
        sample_id: format!("sample-{index}"),
        phase: ImageRequestPhase::Measured,
        repetition: 0,
        dispatch_index: index,
        http_status: status,
        e2e_ms: 1.0,
        validation_ms: 0.0,
        response_bytes: 0,
        image: None,
        failure: Some(failure),
        error: None,
    };
    let records = [
        record(0, Some(429), ImageFailureKind::HttpStatus),
        record(1, None, ImageFailureKind::ClientCapacity),
        record(2, Some(500), ImageFailureKind::HttpStatus),
    ];
    let metrics = ImageMetrics::from_records(2.0, &records, Some(10.0));
    assert_eq!(
        (metrics.attempted, metrics.dispatched, metrics.errors),
        (3, 2, 3)
    );
    assert_eq!(
        (
            metrics.rejected,
            metrics.client_not_dispatched,
            metrics.other_errors
        ),
        (1, 1, 1)
    );
    assert_eq!(metrics.images_per_second, Some(0.0));
    assert_eq!(metrics.e2e_p95_ms, None);
    assert_eq!(metrics.latency_slo_status, ImageSloStatus::Violated);
}

#[test]
fn generation_v1_wire_is_unchanged_and_edit_requires_explicit_v2() {
    let old = serde_json::json!({
        "schema_version":1,
        "samples":[{"id":"one","prompt":"blue teapot","seed":9,"width":2,"height":1}]
    });
    let mut data: ImageDataset = serde_json::from_value(old.clone()).unwrap();
    assert!(data.mode.is_generate());
    assert_eq!(serde_json::to_value(&data).unwrap(), old);
    let generation_identity = data.fingerprint().unwrap();
    data.mode = ImageRequestMode::Edit {
        image: ImageReference {
            path: "reference.png".into(),
            encoded_sha256: None,
        },
    };
    assert!(data.validate(&ImageResponseLimits::default()).is_err());
    data.schema_version = 2;
    data.validate(&ImageResponseLimits::default()).unwrap();
    assert_ne!(generation_identity, data.fingerprint().unwrap());
    let before = data.fingerprint().unwrap();
    let ImageRequestMode::Edit { image } = &mut data.mode else {
        unreachable!()
    };
    image.encoded_sha256 = Some("a".repeat(64));
    assert_ne!(before, data.fingerprint().unwrap());
    let ImageRequestMode::Edit { image } = &mut data.mode else {
        unreachable!()
    };
    image.encoded_sha256 = Some("A".repeat(64));
    assert!(data.validate(&ImageResponseLimits::default()).is_err());
    assert!(
        serde_json::from_value::<ImageRequestMode>(serde_json::json!({
            "kind":"edit", "image":{"path":"ref.png"}, "mask":"mask.png"
        }))
        .is_err()
    );
}

#[test]
fn mixed_workloads_require_v3_and_partition_failures_by_request_kind_and_size() {
    let wire = serde_json::json!({"schema_version":3,"samples":[
        {"id":"generation","prompt":"red kite","seed":1,"width":512,"height":512},
        {"id":"edit","prompt":"make blue","seed":2,"width":1024,"height":1024,
         "mode":{"kind":"edit","image":{"path":"reference.png"}}}
    ]});
    let mut data: ImageDataset = serde_json::from_value(wire).unwrap();
    data.validate(&ImageResponseLimits::default()).unwrap();
    let original = data.fingerprint().unwrap();
    data.samples[1].mode = Some(ImageRequestMode::Generate);
    assert_ne!(data.fingerprint().unwrap(), original);
    data.samples[1].mode = Some(ImageRequestMode::Edit {
        image: ImageReference {
            path: "reference.png".into(),
            encoded_sha256: None,
        },
    });
    data.schema_version = 2;
    assert!(data.validate(&ImageResponseLimits::default()).is_err());
    data.schema_version = 3;
    let records = data
        .samples
        .iter()
        .enumerate()
        .map(|(index, sample)| ImageRequestRecord {
            sample_index: index,
            sample_id: sample.id.clone(),
            phase: ImageRequestPhase::Measured,
            repetition: 0,
            dispatch_index: index,
            http_status: if index == 0 { Some(429) } else { None },
            e2e_ms: 0.0,
            validation_ms: 0.0,
            response_bytes: 0,
            image: None,
            failure: Some(if index == 0 {
                ImageFailureKind::HttpStatus
            } else {
                ImageFailureKind::ClientCapacity
            }),
            error: None,
        })
        .collect::<Vec<_>>();
    let groups = image_workload_metrics(2.0, &records, &data, Some(100.0)).unwrap();
    assert_eq!(groups.len(), 2);
    let edit = groups
        .iter()
        .find(|group| group.request_kind == "edit")
        .unwrap();
    assert_eq!((edit.width, edit.height), (1024, 1024));
    assert_eq!(edit.metrics.client_not_dispatched, 1);
    assert_eq!(edit.metrics.latency_slo_status, ImageSloStatus::Violated);
    let generation = groups
        .iter()
        .find(|group| group.request_kind == "generate")
        .unwrap();
    assert_eq!((generation.width, generation.height), (512, 512));
    assert_eq!(generation.metrics.rejected, 1);
    let mut corrupt = records;
    corrupt[0].sample_id = "wrong-owner".into();
    assert!(image_workload_metrics(2.0, &corrupt, &data, None).is_err());
}

#[test]
fn reference_png_validation_reuses_full_decoder_without_base64_or_assumed_output_size() {
    let bytes = png(png::ColorType::Rgba, &[10, 20, 30, 0, 40, 50, 60, 255]);
    let limits = ImageResponseLimits::default();
    let actual = validate_reference_png(&bytes, &limits).unwrap();
    let response = validate_image_response(&response(&bytes), 2, 1, &limits).unwrap();
    assert_eq!((actual.width, actual.height), (2, 1));
    assert_eq!(actual.png_sha256, sha256(&bytes));
    assert_eq!(actual.rgba8_sha256, response.rgba8_sha256);
    let mut short = limits.clone();
    short.maximum_png_bytes = bytes.len() - 1;
    assert!(validate_reference_png(&bytes, &short).is_err());
    short = limits.clone();
    short.maximum_pixels = 1;
    assert!(validate_reference_png(&bytes, &short).is_err());
    let mut bad = bytes.clone();
    bad[29] ^= 1;
    assert!(validate_reference_png(&bad, &limits).is_err());
    let mut trailing = bytes;
    trailing.push(0);
    assert!(validate_reference_png(&trailing, &limits).is_err());
}
