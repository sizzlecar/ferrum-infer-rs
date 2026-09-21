use super::*;

#[test]
fn metal_counter_page_budget_stays_bounded_and_fails_closed() {
    let device = metal::Device::system_default().expect("Metal test device");
    let MetalTimestampCounterSupport::Supported(set) = timestamp_counter_support(&device) else {
        return;
    };
    let stats = Arc::new(CounterReadbackStats::new(0, device.name()));
    let mut capture = MetalCounterCaptureBuilder::new(device.clone(), set, stats).unwrap();
    assert!(
        capture.pages.is_empty(),
        "counter pages must allocate on demand"
    );
    let maximum_intervals = METAL_COUNTER_MAX_PAGES as u64 * METAL_COUNTER_SAMPLES_PER_PAGE / 2;
    for _ in 0..maximum_intervals {
        assert!(capture
            .reserve(0, DeviceExecutionIntervalKind::Transfer, None)
            .is_some());
    }
    assert_eq!(capture.pages.len(), METAL_COUNTER_MAX_PAGES);
    let maximum_payload = capture
        .pages
        .iter()
        .map(|page| {
            page.sample_buffer.sample_count() * std::mem::size_of::<u64>() as u64
                + page.resolved.length()
        })
        .sum::<u64>();
    assert_eq!(maximum_payload, 256 * 1024);
    assert!(capture
        .reserve(0, DeviceExecutionIntervalKind::Transfer, None)
        .is_none());
    assert!(capture.unavailable);
    assert!(capture
        .reserve(1, DeviceExecutionIntervalKind::Transfer, None)
        .is_none());
    assert_eq!(capture.pages.len(), METAL_COUNTER_MAX_PAGES);
    let queue = device.new_command_queue();
    let command_buffer = queue.new_command_buffer();
    let captured = capture.finish(command_buffer, 2);
    // No incomplete prefix of an exhausted capture may look like full timing.
    assert!(matches!(
        captured.resolve(),
        DeviceTimingMeasurement::Unavailable(
            DeviceTimingUnavailableReason::BackendMeasurementFailed
        )
    ));
}

#[test]
fn metal_counter_crosses_old_page_limit_with_real_ordered_transfers() {
    let runtime = super::tests::runtime();
    let destination = runtime
        .allocate_request(&super::tests::buffer_request(
            "resource/counter-capacity-transfer",
        ))
        .unwrap();
    let region = destination.region(0..8).unwrap();
    // Exceed the previous 2048-interval ceiling and cross multiple larger
    // pages, as a C32 per-participant workload does. Every interval does work.
    const INTERVALS: usize = 4097;
    let command = MetalDeviceCommand::operation(
        "test.counter_capacity_transfers",
        vec![region.clone()],
        |encoder, regions| {
            for index in 0..INTERVALS {
                encoder.with_blit(|blit| {
                    blit.fill_buffer(
                        regions[0].buffer(),
                        NSRange::new(regions[0].offset_bytes(), 8),
                        (index % 251) as u8,
                    )
                });
            }
            Ok(())
        },
    )
    .unwrap();
    let mut stream = runtime.create_stream().unwrap();
    let fence = runtime
        .submit_commands(
            &mut stream,
            vec![(DeviceCommandPhase::Compute, None, command)],
            DeviceTimingMode::Kernel,
            &DisabledDeviceSubmissionTimingSink,
        )
        .unwrap();
    let terminal = runtime.wait_fence(&fence).unwrap();
    assert!(terminal.terminal().is_succeeded());
    assert_eq!(
        super::tests::region_bytes(&region),
        &[((INTERVALS - 1) % 251) as u8; 8]
    );
    let timing = match terminal.submission_timing() {
        DeviceTimingMeasurement::Measured(timing) => timing,
        DeviceTimingMeasurement::Unavailable(DeviceTimingUnavailableReason::BackendUnsupported) => {
            return
        }
        other => panic!("cross-page Metal counter timing failed: {other:?}"),
    };
    assert_eq!(timing.command_count(), 1);
    let intervals = timing.spans()[0].measurement().intervals().unwrap();
    assert_eq!(intervals.len(), INTERVALS);
    assert!(intervals.iter().all(|interval| interval.kind()
        == DeviceExecutionIntervalKind::Transfer
        && interval.end_offset_ns() > interval.start_offset_ns()));
    assert!(intervals
        .windows(2)
        .all(|pair| pair[0].end_offset_ns() <= pair[1].start_offset_ns()));
}
