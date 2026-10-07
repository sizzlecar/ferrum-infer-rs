//! Model-free boundary tests for physical encoder timestamp capacity.

use super::*;
use ferrum_interfaces::vnext::{DynamicStorageAllocator, DynamicStorageView, ResourceId};
use metal::objc::rc::autoreleasepool;

struct CounterCapacityFixture {
    runtime: MetalDeviceRuntime,
    stream: MetalDeviceStream,
    destination: MetalDeviceBuffer,
}

impl CounterCapacityFixture {
    fn new() -> Option<Self> {
        let runtime = MetalDeviceRuntime::new(MetalDeviceRuntimeConfig {
            device_id: DeviceId::new("device/metal/counter-capacity-test").unwrap(),
            runtime_implementation_fingerprint: "a".repeat(64),
            capabilities: BTreeSet::new(),
            dynamic_storage_profiles: BTreeSet::from([DynamicStorageProfile::new(
                DynamicStorageAllocator::LinearArena,
                DynamicStorageView::Contiguous,
            )
            .unwrap()]),
        })
        .expect("Metal runtime");
        if matches!(
            runtime.timestamp_counter_support(),
            MetalTimestampCounterSupport::Unsupported
        ) {
            eprintln!("Metal counter capacity test skipped: timestamp counters unsupported");
            return None;
        }
        let request = BufferRequest::new(
            ResourceId::new("resource/counter-capacity-test").unwrap(),
            8,
            64,
            BufferUsage::Transfer,
            ElementType::U8,
        )
        .unwrap();
        let destination = runtime.allocate_request(&request).expect("allocation");
        let stream = runtime.create_stream().expect("stream");
        Some(Self {
            runtime,
            stream,
            destination,
        })
    }

    fn check_submission(&mut self, interval_count: usize) {
        assert!(interval_count > 0);
        let intervals_per_page = METAL_COUNTER_SAMPLES_PER_PAGE as usize / 2;
        let capacity = intervals_per_page * METAL_COUNTER_MAX_PAGES;
        let captured_intervals = interval_count.min(capacity);
        let over_capacity = interval_count > capacity;
        let fence = autoreleasepool(|| {
            // One logical command deliberately owns many physical encoders:
            // counter capacity must track intervals, not logical command count.
            let command = MetalDeviceCommand::operation(
                "test.counter_capacity",
                vec![self.destination.region(0..8).expect("region")],
                move |encoder, regions| {
                    for index in 0..interval_count {
                        encoder.with_blit_commands(1, |blit| {
                            blit.fill_buffer(
                                regions[0].buffer(),
                                NSRange::new(regions[0].offset_bytes(), 8),
                                payload_byte(index),
                            );
                        });
                    }
                    Ok(())
                },
            )
            .expect("command");
            self.runtime
                .submit_commands(
                    &mut self.stream,
                    vec![(DeviceCommandPhase::Compute, None, command)],
                    DeviceTimingMode::Kernel,
                    &DisabledDeviceSubmissionTimingSink,
                )
                .expect("profiled submission")
        });
        // The submission's temporary Objective-C owners have drained; the
        // fence must retain every counter page until completion and resolution.
        let terminal = self.runtime.wait_fence(&fence).expect("terminal fence");
        assert!(terminal.terminal().is_succeeded());
        assert_eq!(self.stream.pending.len(), 0);
        let output = self
            .runtime
            .readback(
                &mut self.stream,
                &self.destination,
                CopyRegion::new(0, 0, 8).unwrap(),
                HostTransferLayout::new(ElementType::U8, 8).unwrap(),
            )
            .expect("readback");
        assert_eq!(output, [payload_byte(interval_count - 1); 8]);

        let MetalFenceCommandTiming::Captured { capture, .. } = &fence.command_timing else {
            panic!("timestamp support must produce a capture")
        };
        assert_eq!(capture.unavailable, over_capacity);
        assert_eq!(capture.command_count, 1);
        assert_eq!(capture.mappings.len(), captured_intervals);
        assert_eq!(
            capture.pages.len(),
            captured_intervals.div_ceil(intervals_per_page)
        );
        for (index, page) in capture.pages.iter().enumerate() {
            let used_intervals =
                (captured_intervals - index * intervals_per_page).min(intervals_per_page);
            assert_eq!(page.used_samples, (used_intervals * 2) as u64);
        }
        for (index, mapping) in capture.mappings.iter().enumerate() {
            assert_eq!(mapping.command_index, 0);
            assert_eq!(mapping.kind, DeviceExecutionIntervalKind::Transfer);
            assert_eq!(mapping.page_index, index / intervals_per_page);
            assert_eq!(
                mapping.start_sample_index,
                ((index % intervals_per_page) * 2) as u64
            );
            assert_eq!(mapping.end_sample_index, mapping.start_sample_index + 1);
        }
        let attribution = self
            .runtime
            .submission_attribution(&fence)
            .expect("native attribution");
        assert_eq!(attribution.commands().len(), 1);
        assert_eq!(
            attribution.commands()[0].transfer_command_count(),
            interval_count as u64
        );
        if over_capacity {
            assert!(matches!(
                terminal.submission_timing(),
                DeviceTimingMeasurement::Unavailable(
                    DeviceTimingUnavailableReason::BackendMeasurementFailed
                )
            ));
        } else {
            let DeviceTimingMeasurement::Measured(timing) = terminal.submission_timing() else {
                panic!(
                    "within-capacity capture failed: {:?}",
                    terminal.submission_timing()
                )
            };
            assert_eq!(timing.command_count(), 1);
            let [span] = timing.spans() else {
                panic!("one logical command must have one timing span")
            };
            let intervals = span.measurement().intervals().expect("measured intervals");
            assert_eq!(intervals.len(), interval_count);
            assert!(intervals
                .iter()
                .all(|interval| interval.kind() == DeviceExecutionIntervalKind::Transfer));
        }
        let repeated = self.runtime.wait_fence(&fence).expect("repeat wait");
        assert_eq!(terminal.submission_timing(), repeated.submission_timing());
    }
}

fn payload_byte(index: usize) -> u8 {
    // Consecutive encoders write different nonzero bytes, including after the
    // counter limit, so a silently skipped final encoder cannot pass readback.
    (index % 251 + 1) as u8
}

#[test]
fn counter_capture_page_and_capacity_boundaries_preserve_intervals() {
    autoreleasepool(|| {
        let Some(mut fixture) = CounterCapacityFixture::new() else {
            return;
        };
        let intervals_per_page = METAL_COUNTER_SAMPLES_PER_PAGE as usize / 2;
        for interval_count in [
            intervals_per_page - 1,
            intervals_per_page,
            intervals_per_page + 1,
            intervals_per_page * METAL_COUNTER_MAX_PAGES,
        ] {
            autoreleasepool(|| fixture.check_submission(interval_count));
        }
    });
}

#[test]
fn counter_capture_overflow_preserves_work_and_next_submission_recovers() {
    autoreleasepool(|| {
        let Some(mut fixture) = CounterCapacityFixture::new() else {
            return;
        };
        let intervals_per_page = METAL_COUNTER_SAMPLES_PER_PAGE as usize / 2;
        let capacity = intervals_per_page * METAL_COUNTER_MAX_PAGES;
        // More than one over-limit encoder also checks that allocation stops
        // after the first failure while all native work continues to execute.
        autoreleasepool(|| fixture.check_submission(capacity + 2));
        autoreleasepool(|| fixture.check_submission(intervals_per_page + 1));
    });
}
