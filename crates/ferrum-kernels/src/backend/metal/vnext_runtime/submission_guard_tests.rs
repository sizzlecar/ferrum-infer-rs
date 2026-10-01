use super::*;
use ferrum_interfaces::execution_cost::{GuardedNotSubmittedReason, HostSubmissionRejection};

struct Guard {
    reject: bool,
    calls: AtomicU64,
}
impl DeviceSubmissionGuard for Guard {
    fn check(
        &self,
        evidence: Option<&DeviceSubmissionAttribution>,
    ) -> Result<(), GuardedNotSubmittedReason> {
        let evidence = evidence.expect("gate must receive actual native encoding evidence");
        assert_eq!(evidence.commands().len(), 1);
        assert_eq!(evidence.commands()[0].transfer_command_count(), 1);
        assert_eq!(evidence.commands()[0].compute_dispatch_count(), 0);
        self.calls.fetch_add(1, Ordering::Relaxed);
        if self.reject {
            Err(GuardedNotSubmittedReason::HostRejected(
                HostSubmissionRejection::WitnessExpired,
            ))
        } else {
            Ok(())
        }
    }
}

#[test]
fn guarded_native_encode_rejection_keeps_zero_commits_and_lane_reusable() {
    guarded_native_encode_rejection_and_retry(DeviceTimingMode::Off);
}

#[test]
fn guarded_native_completion_timing_keeps_zero_rejected_commits_and_measures_retry() {
    guarded_native_encode_rejection_and_retry(DeviceTimingMode::Completion);
}

fn guarded_native_encode_rejection_and_retry(timing_mode: DeviceTimingMode) {
    let runtime = tests::runtime();
    let buffer = runtime
        .allocate_request(&tests::buffer_request("guarded.destination"))
        .unwrap();
    let mut stream = runtime.create_stream().unwrap();
    let layout = HostTransferLayout::new(ElementType::U8, 8).unwrap();
    let upload = runtime.encode_upload(&[9; 8], layout, &buffer, 0).unwrap();
    let initial = runtime
        .submit_commands(
            &mut stream,
            vec![(DeviceCommandPhase::DynamicBinding, None, upload)],
            DeviceTimingMode::Off,
            &DisabledDeviceSubmissionTimingSink,
        )
        .unwrap();
    assert!(runtime
        .wait_fence(&initial)
        .unwrap()
        .terminal()
        .is_succeeded());
    let rejected = Guard {
        reject: true,
        calls: AtomicU64::new(0),
    };
    let command = runtime.encode_zero(&buffer, 0, 8).unwrap();
    let result = runtime.submit_commands_inner(
        &mut stream,
        vec![(DeviceCommandPhase::Initialization, None, command)],
        timing_mode,
        true,
        DeviceCostObservationDemand::Required,
        &DisabledDeviceSubmissionTimingSink,
        Some(&rejected),
    );
    assert!(matches!(
        result,
        Err(GuardedDeviceSubmissionError::Rejected(
            GuardedNotSubmittedReason::HostRejected(HostSubmissionRejection::WitnessExpired)
        ))
    ));
    assert_eq!(rejected.calls.load(Ordering::Relaxed), 1);
    assert_eq!(runtime.stream_state(&stream), StreamState::Ready);
    assert_eq!(stream.pending.len(), 0);
    assert_eq!(
        runtime
            .readback(
                &mut stream,
                &buffer,
                CopyRegion::new(0, 0, 8).unwrap(),
                layout
            )
            .unwrap(),
        [9; 8]
    );

    // The same stream and real allocation execute normally after rejection;
    // no abort/recreate operation can hide a stuck Recording state.
    let accepted = Guard {
        reject: false,
        calls: AtomicU64::new(0),
    };
    let command = runtime.encode_zero(&buffer, 0, 8).unwrap();
    let fence = runtime
        .submit_commands_inner(
            &mut stream,
            vec![(DeviceCommandPhase::Initialization, None, command)],
            timing_mode,
            true,
            DeviceCostObservationDemand::Required,
            &DisabledDeviceSubmissionTimingSink,
            Some(&accepted),
        )
        .unwrap();
    assert_eq!(accepted.calls.load(Ordering::Relaxed), 1);
    let terminal = runtime.wait_fence(&fence).unwrap();
    assert!(terminal.terminal().is_succeeded());
    match timing_mode {
        DeviceTimingMode::Completion => assert!(matches!(
            terminal.execution_timing(),
            DeviceTimingMeasurement::Measured(_)
        )),
        DeviceTimingMode::Off => assert!(matches!(
            terminal.execution_timing(),
            DeviceTimingMeasurement::NotRequested
        )),
        _ => unreachable!(),
    }
    assert!(matches!(
        terminal.submission_timing(),
        DeviceTimingMeasurement::NotRequested
    ));
    assert_eq!(
        runtime
            .readback(
                &mut stream,
                &buffer,
                CopyRegion::new(0, 0, 8).unwrap(),
                layout
            )
            .unwrap(),
        [0; 8]
    );
    assert_eq!(runtime.stream_state(&stream), StreamState::Ready);
    assert_eq!(stream.pending.len(), 0);
}
