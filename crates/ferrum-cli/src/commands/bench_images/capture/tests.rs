use super::*;
use ferrum_bench_core::images::validate_image_response_with_png;

fn payload() -> (ValidatedImage, Vec<u8>) {
    validate_image_response_with_png(
        super::super::tests::valid_body().as_bytes(),
        1,
        1,
        &ImageResponseLimits::default(),
    )
    .unwrap()
    .into_parts()
}

#[test]
fn create_new_rejects_existing_directory_and_existing_png_without_overwriting() {
    let parent = tempfile::tempdir().unwrap();
    let (image, png) = payload();
    let config = PngCaptureConfig {
        directory: parent.path().join("capture"),
        maximum_capture_bytes: NonZeroU64::new(png.len() as u64 * 2).unwrap(),
    };
    let sink = CaptureSink::create(&config).unwrap();
    assert!(CaptureSink::create(&config).is_err());
    let saved = sink.save(0, ImageRequestPhase::Measured, 0, 0, &image, &png);
    let PngCaptureResult::Saved { relative_path, .. } = saved else {
        panic!("capture failed")
    };
    let before = std::fs::read(config.directory.join(&relative_path)).unwrap();
    assert!(matches!(
        sink.save(0, ImageRequestPhase::Measured, 0, 0, &image, &png),
        PngCaptureResult::Failed {
            kind: PngCaptureFailureKind::Io,
            ..
        }
    ));
    assert_eq!(
        std::fs::read(config.directory.join(relative_path)).unwrap(),
        before
    );
}

#[test]
fn concurrent_capture_cannot_exceed_the_shared_byte_allowance() {
    let parent = tempfile::tempdir().unwrap();
    let (image, png) = payload();
    let config = PngCaptureConfig {
        directory: parent.path().join("capture"),
        maximum_capture_bytes: NonZeroU64::new(png.len() as u64 * 2).unwrap(),
    };
    let sink = CaptureSink::create(&config).unwrap();
    let barrier = std::sync::Barrier::new(4);
    let results = std::thread::scope(|scope| {
        let jobs: Vec<_> = (0..4)
            .map(|index| {
                let sink = &sink;
                let barrier = &barrier;
                let image = &image;
                let png = &png;
                scope.spawn(move || {
                    barrier.wait();
                    sink.save(0, ImageRequestPhase::Measured, 0, index, image, png)
                })
            })
            .collect();
        jobs.into_iter()
            .map(|job| job.join().unwrap())
            .collect::<Vec<_>>()
    });
    assert_eq!(
        results
            .iter()
            .filter(|r| matches!(r, PngCaptureResult::Saved { .. }))
            .count(),
        2
    );
    assert_eq!(
        results
            .iter()
            .filter(|r| matches!(
                r,
                PngCaptureResult::Failed {
                    kind: PngCaptureFailureKind::Capacity,
                    ..
                }
            ))
            .count(),
        2
    );
    assert_eq!(sink.reserved.load(Ordering::Relaxed), png.len() as u64 * 2);
    assert_eq!(std::fs::read_dir(&config.directory).unwrap().count(), 2);
}

#[test]
fn save_panic_retains_validated_image_and_reports_unknown_partial_capture() {
    let parent = tempfile::tempdir().unwrap();
    let partial_path = parent.path().join("partial.png");
    let response = validate_image_response_with_png(
        super::super::tests::valid_body().as_bytes(),
        1,
        1,
        &ImageResponseLimits::default(),
    )
    .unwrap();
    let (expected, png) = payload();
    let (image, captured) = capture_validated_png(response, |_, png| {
        std::fs::write(&partial_path, &png[..8]).unwrap();
        panic!("injected capture failure after partial write");
    });
    assert_eq!(image.png_sha256, expected.png_sha256);
    assert_eq!(image.rgba8_sha256, expected.rgba8_sha256);
    assert_eq!(std::fs::read(partial_path).unwrap(), png[..8]);
    match captured {
        PngCaptureResult::Failed {
            kind: PngCaptureFailureKind::ClientTask,
            relative_path: None,
            capture_ms: Some(elapsed),
            ..
        } => assert!(elapsed.is_finite() && elapsed >= 0.0),
        other => panic!("expected independent capture failure, got {other:?}"),
    }
}
