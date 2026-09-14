//! Opt-in, model-free counter-buffer lifetime diagnostics.
//!
//! Run each ignored test in its own test process. Both scopes use the same
//! submissions and payloads; the only difference is when Objective-C
//! autoreleased submission objects are drained. These tests report missing
//! measurements explicitly and never replace them with zero durations.

use super::*;
use ferrum_interfaces::vnext::{DynamicStorageAllocator, DynamicStorageView, ResourceId};
use metal::objc::rc::autoreleasepool;

#[derive(Clone, Copy, Debug)]
enum PoolScope {
    WholeRun,
    EachSubmission,
}

#[derive(Clone, Copy)]
struct ProbeWorkload {
    submissions: usize,
    commands_per_submission: usize,
}

const COUNTER_LIFETIME_WORKLOAD: ProbeWorkload = ProbeWorkload {
    // A normal observed decode submission owns about four 256-sample pages.
    // Repetition exposes retained resources; no particular failure ordinal
    // or platform-specific resource limit is an acceptance requirement.
    submissions: 16,
    commands_per_submission: 504,
};

fn probe_runtime() -> MetalDeviceRuntime {
    MetalDeviceRuntime::new(MetalDeviceRuntimeConfig {
        device_id: DeviceId::new("device/metal/counter-lifetime-probe").unwrap(),
        runtime_implementation_fingerprint: "a".repeat(64),
        capabilities: BTreeSet::new(),
        dynamic_storage_profiles: BTreeSet::from([DynamicStorageProfile::new(
            DynamicStorageAllocator::LinearArena,
            DynamicStorageView::Contiguous,
        )
        .unwrap()]),
    })
    .expect("Metal runtime")
}

fn objective_c_error(error: *mut Object) -> serde_json::Value {
    if error.is_null() {
        return serde_json::Value::Null;
    }
    unsafe {
        let domain: *mut Object = msg_send![error, domain];
        let description: *mut Object = msg_send![error, localizedDescription];
        let code: metal::NSInteger = msg_send![error, code];
        let owned_text = |object: *mut Object| {
            if object.is_null() {
                return None;
            }
            let text: *const std::ffi::c_char = msg_send![object, UTF8String];
            (!text.is_null()).then(|| CStr::from_ptr(text).to_string_lossy().into_owned())
        };
        serde_json::json!({
            "domain": owned_text(domain),
            "code": code,
            "description": owned_text(description),
        })
    }
}

/// An extra, clearly identified allocation after the first observed failure.
/// It captures NSError without changing production measurement handling.
fn probe_counter_creation(runtime: &MetalDeviceRuntime) -> serde_json::Value {
    let MetalTimestampCounterSupport::Supported(counter_set) = runtime.timestamp_counter_support()
    else {
        return serde_json::json!({"status": "unsupported"});
    };
    let descriptor = CounterSampleBufferDescriptor::new();
    descriptor.set_counter_set(&counter_set);
    descriptor.set_storage_mode(MTLStorageMode::Shared);
    descriptor.set_sample_count(METAL_COUNTER_SAMPLES_PER_PAGE);
    descriptor.set_label("ferrum.test.counter_lifetime.allocation_probe");
    let mut error: *mut Object = std::ptr::null_mut();
    let pointer: *mut MTLCounterSampleBuffer = unsafe {
        msg_send![&*runtime.device,
            newCounterSampleBufferWithDescriptor: &*descriptor
            error: &mut error
        ]
    };
    let allocated = !pointer.is_null();
    let error = objective_c_error(error);
    if allocated {
        // new returns an owned reference. Do not retain it a second time.
        drop(unsafe { CounterSampleBuffer::from_ptr(pointer) });
    }
    serde_json::json!({"allocated": allocated, "error": error})
}

fn capture_state(fence: &MetalDeviceFence) -> serde_json::Value {
    match &fence.command_timing {
        MetalFenceCommandTiming::NotRequested => {
            serde_json::json!({"status": "not_requested"})
        }
        MetalFenceCommandTiming::Unavailable(reason) => {
            serde_json::json!({"status": "unavailable", "reason": format!("{reason:?}")})
        }
        MetalFenceCommandTiming::Captured { capture, .. } => {
            serde_json::json!({
                "status": "captured",
                "builder_unavailable": capture.unavailable,
                "pages": capture.pages.len(),
                "used_samples": capture.pages.iter().map(|page| page.used_samples).collect::<Vec<_>>(),
                "mappings": capture.mappings.len(),
                "command_count": capture.command_count,
                "cpu_anchor_start": capture.cpu_anchor_start,
                "gpu_anchor_start": capture.gpu_anchor_start,
            })
        }
    }
}

fn run_submission(
    runtime: &MetalDeviceRuntime,
    stream: &mut MetalDeviceStream,
    destination: &MetalDeviceBuffer,
    command_count: usize,
    ordinal: usize,
    probe_allocation_on_failure: bool,
) -> bool {
    assert!(command_count > 1);
    let source = [0xA5_u8; 8];
    let mut commands = Vec::with_capacity(command_count);
    commands.push(
        runtime
            .encode_upload(
                &source,
                HostTransferLayout::new(ElementType::U8, 8).unwrap(),
                destination,
                0,
            )
            .expect("upload"),
    );
    for _ in 1..command_count {
        commands.push(runtime.encode_zero(destination, 0, 8).expect("zero"));
    }
    let entries = commands
        .into_iter()
        .map(|command| (DeviceCommandPhase::Compute, None, command))
        .collect();
    let fence = runtime
        .submit_commands(
            stream,
            entries,
            DeviceTimingMode::Kernel,
            &DisabledDeviceSubmissionTimingSink,
        )
        .expect("profiled submission");
    let terminal = runtime.wait_fence(&fence).expect("terminal fence");
    assert!(terminal.terminal().is_succeeded());
    assert_eq!(
        stream.pending.len(),
        0,
        "completed native command buffer remains pending"
    );
    let output = runtime
        .readback(
            stream,
            destination,
            CopyRegion::new(0, 0, 8).unwrap(),
            HostTransferLayout::new(ElementType::U8, 8).unwrap(),
        )
        .expect("readback");
    assert_eq!(output, [0; 8], "GPU payload changed");

    let (complete, measured, unavailable, failure) = match terminal.submission_timing() {
        DeviceTimingMeasurement::Measured(timing) => {
            assert_eq!(timing.command_count() as usize, command_count);
            let measured = timing
                .spans()
                .iter()
                .filter(|span| span.measurement().elapsed_ns().is_some())
                .count();
            let unavailable = timing.spans().len() - measured;
            (unavailable == 0, measured, unavailable, None)
        }
        DeviceTimingMeasurement::Unavailable(reason) => {
            (false, 0, command_count, Some(format!("{reason:?}")))
        }
        DeviceTimingMeasurement::NotRequested => {
            (false, 0, command_count, Some("NotRequested".to_owned()))
        }
    };
    let creation_probe = if !complete && probe_allocation_on_failure {
        Some(probe_counter_creation(runtime))
    } else {
        None
    };
    eprintln!(
        "{}",
        serde_json::json!({
            "event": "metal_counter_lifetime_submission",
            "ordinal": ordinal,
            "payload_valid": true,
            "complete_measurement": complete,
            "measured_commands": measured,
            "unavailable_commands": unavailable,
            "unavailable_reason": failure,
            "capture": capture_state(&fence),
            "post_failure_creation_probe": creation_probe,
        })
    );
    // Explicitly drop both the Rust fence and its copied numeric receipt before
    // the surrounding per-submission pool drains.
    drop(terminal);
    drop(fence);
    complete
}

fn counter_lifetime_probe(scope: PoolScope, workload: ProbeWorkload) {
    autoreleasepool(|| {
        let runtime = probe_runtime();
        if matches!(
            runtime.timestamp_counter_support(),
            MetalTimestampCounterSupport::Unsupported
        ) {
            eprintln!(
                "{}",
                serde_json::json!({
                    "event": "metal_counter_lifetime_summary",
                    "status": "unsupported",
                    "pool_scope": format!("{scope:?}"),
                })
            );
            return;
        }
        let request = BufferRequest::new(
            ResourceId::new("resource/counter-lifetime-probe").unwrap(),
            8,
            64,
            BufferUsage::Transfer,
            ElementType::U8,
        )
        .unwrap();
        let destination = runtime.allocate_request(&request).expect("allocation");
        let mut stream = runtime.create_stream().expect("stream");
        let mut first_unavailable = None;
        let mut complete_submissions = 0;
        for ordinal in 0..workload.submissions {
            let mut action = || {
                run_submission(
                    &runtime,
                    &mut stream,
                    &destination,
                    workload.commands_per_submission,
                    ordinal,
                    first_unavailable.is_none(),
                )
            };
            let complete = match scope {
                PoolScope::WholeRun => action(),
                PoolScope::EachSubmission => autoreleasepool(action),
            };
            if complete {
                complete_submissions += 1;
            } else if first_unavailable.is_none() {
                first_unavailable = Some(ordinal);
            }
        }
        eprintln!(
            "{}",
            serde_json::json!({
                "event": "metal_counter_lifetime_summary",
                "status": "completed",
                "pool_scope": format!("{scope:?}"),
                "submissions": workload.submissions,
                "commands_per_submission": workload.commands_per_submission,
                "complete_submissions": complete_submissions,
                "first_unavailable": first_unavailable,
                "payload_valid": true,
            })
        );
    });
}

#[test]
#[ignore = "model-free GPU lifecycle diagnostic; coordinate an idle GPU and run this test alone"]
fn counter_capture_whole_run_pool_diagnostic() {
    counter_lifetime_probe(PoolScope::WholeRun, COUNTER_LIFETIME_WORKLOAD);
}

#[test]
#[ignore = "model-free GPU lifecycle diagnostic; coordinate an idle GPU and run this test alone"]
fn counter_capture_per_submission_pool_diagnostic() {
    counter_lifetime_probe(PoolScope::EachSubmission, COUNTER_LIFETIME_WORKLOAD);
}

fn verify_fence_outlives_submit_pool(timing_mode: DeviceTimingMode) {
    autoreleasepool(|| {
        let runtime = probe_runtime();
        let mut stream = runtime.create_stream().expect("stream");
        let (fence, destination) = autoreleasepool(|| {
            let request = BufferRequest::new(
                ResourceId::new("resource/fence-outlives-submit-pool").unwrap(),
                8,
                64,
                BufferUsage::Transfer,
                ElementType::U8,
            )
            .unwrap();
            let destination = runtime.allocate_request(&request).expect("allocation");
            // This payload and all autoreleased encoding objects leave scope
            // before the fence is waited. Only the returned owners survive.
            let payload = vec![1_u8, 2, 3, 4, 5, 6, 7, 8];
            let commands = vec![
                runtime
                    .encode_upload(
                        &payload,
                        HostTransferLayout::new(ElementType::U8, 8).unwrap(),
                        &destination,
                        0,
                    )
                    .expect("upload"),
                runtime.encode_zero(&destination, 2, 3).expect("zero range"),
            ];
            let entries = commands
                .into_iter()
                .map(|command| (DeviceCommandPhase::Compute, None, command))
                .collect();
            let fence = runtime
                .submit_commands(
                    &mut stream,
                    entries,
                    timing_mode,
                    &DisabledDeviceSubmissionTimingSink,
                )
                .expect("submission");
            (fence, destination)
        });

        let terminal = runtime.wait_fence(&fence).expect("wait after pool drain");
        assert!(terminal.terminal().is_succeeded());
        assert_eq!(stream.pending.len(), 0);
        let output = runtime
            .readback(
                &mut stream,
                &destination,
                CopyRegion::new(0, 0, 8).unwrap(),
                HostTransferLayout::new(ElementType::U8, 8).unwrap(),
            )
            .expect("readback after pool drain");
        assert_eq!(output, [1, 2, 0, 0, 0, 6, 7, 8]);
        match timing_mode {
            DeviceTimingMode::Off => {
                assert!(matches!(
                    terminal.execution_timing(),
                    DeviceTimingMeasurement::NotRequested
                ));
                assert!(matches!(
                    terminal.submission_timing(),
                    DeviceTimingMeasurement::NotRequested
                ));
            }
            DeviceTimingMode::Kernel => {
                assert!(matches!(
                    terminal.execution_timing(),
                    DeviceTimingMeasurement::Measured(_)
                ));
                if matches!(
                    runtime.timestamp_counter_support(),
                    MetalTimestampCounterSupport::Supported(_)
                ) {
                    let DeviceTimingMeasurement::Measured(timing) = terminal.submission_timing()
                    else {
                        panic!(
                            "counter timing unavailable after pool drain: {:?}",
                            terminal.submission_timing()
                        );
                    };
                    assert_eq!(timing.command_count(), 2);
                    for span in timing.spans() {
                        assert!(span
                            .measurement()
                            .elapsed_ns()
                            .is_some_and(|elapsed| elapsed > 0));
                    }
                } else {
                    assert!(matches!(
                        terminal.submission_timing(),
                        DeviceTimingMeasurement::Unavailable(
                            DeviceTimingUnavailableReason::BackendUnsupported
                        )
                    ));
                }
            }
            _ => unreachable!("test only exercises profiled and unprofiled lifetimes"),
        }
    });
}

#[test]
fn profiled_fence_and_buffers_survive_submit_pool_drain() {
    verify_fence_outlives_submit_pool(DeviceTimingMode::Kernel);
}

#[test]
fn unprofiled_fence_and_buffers_survive_submit_pool_drain() {
    verify_fence_outlives_submit_pool(DeviceTimingMode::Off);
}
