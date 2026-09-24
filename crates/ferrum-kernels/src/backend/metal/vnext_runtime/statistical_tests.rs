use super::*;
use ferrum_interfaces::execution_cost::{
    SelectedAlgorithmClassV1, SelectedCommandCostBuilderV1, StatisticalTransferKindV1,
};
use half::f16;

fn bytes(region: &MetalBufferRegion) -> Vec<u8> {
    unsafe {
        std::slice::from_raw_parts(
            region
                .buffer()
                .contents()
                .cast::<u8>()
                .add(region.offset_bytes() as usize),
            region.length_bytes() as usize,
        )
        .to_vec()
    }
}

#[test]
fn real_linear_native_statistics_match_future_and_bad_extension_preserves_exact_execution() {
    let runtime = tests::runtime();
    let mut stream = runtime.create_stream().unwrap();
    for invalid in [false, true] {
        let (mut command, expected, regions) =
            super::super::vnext_ops::selected_runtime_fixture(&runtime);
        let input = bytes(&regions[0]);
        let weight = bytes(&regions[1]);
        if invalid {
            let mut wrong = SelectedCommandCostBuilderV1::new(8);
            wrong
                .transfer(
                    SelectedAlgorithmClassV1::new("fixture.invalid.transfer", 1, [1; 32], [2; 32])
                        .unwrap(),
                    StatisticalTransferKindV1::Fill,
                    4,
                )
                .unwrap();
            command = command.with_statistical_evidence(Some(wrong.finish().unwrap()));
        }
        let fence = runtime
            .submit_commands_with_attribution(
                &mut stream,
                vec![(DeviceCommandPhase::Compute, Some(0), command)],
                DeviceTimingMode::Off,
                true,
                &DisabledDeviceSubmissionTimingSink,
            )
            .unwrap();
        assert!(runtime
            .wait_fence(&fence)
            .unwrap()
            .terminal()
            .is_succeeded());
        let attribution = runtime.submission_attribution(&fence).unwrap();
        assert_eq!(attribution.commands().len(), 1);
        let actual = &attribution.commands()[0];
        assert_eq!(
            actual.compute_dispatch_count(),
            expected.compute_dispatch_count()
        );
        assert_eq!(actual.transfer_command_count(), 0);
        assert_eq!(actual.token_count(), 8);
        if invalid {
            assert!(actual.statistical_evidence().is_none());
        } else {
            assert_eq!(
                actual.statistical_evidence(),
                expected.statistical_evidence()
            );
        }
        assert_eq!(bytes(&regions[0]), input);
        assert_eq!(bytes(&regions[1]), weight);
        let output = bytes(&regions[2]);
        let words: Vec<_> = output
            .chunks_exact(2)
            .map(|v| u16::from_le_bytes([v[0], v[1]]))
            .collect();
        let sentinel = f16::from_f32(333.0).to_bits();
        assert!(words[..4].iter().all(|&v| v == sentinel));
        assert!(words[4 + 8 * 1027..].iter().all(|&v| v == sentinel));
        for row in words[4..4 + 8 * 1027].chunks_exact(1027) {
            assert_eq!(row[0], sentinel);
            assert!(row[1..1025].iter().all(|&v| v == f16::ZERO.to_bits()));
            assert!(row[1025..].iter().all(|&v| v == sentinel));
        }
        assert_eq!(runtime.stream_state(&stream), StreamState::Ready);
    }
}

#[test]
fn real_core_transfer_statistics_match_proven_spans_and_rebound_scratch_work() {
    use StatisticalTransferKindV1 as K;
    let runtime = tests::runtime();
    let source = runtime
        .allocate_request(&tests::buffer_request("resource/stat-source"))
        .unwrap();
    let destination = runtime
        .allocate_request(&tests::buffer_request("resource/stat-target"))
        .unwrap();
    let mut stream = runtime.create_stream().unwrap();
    let payload = [1, 2, 3, 4, 5, 6, 7, 8];
    let upload = runtime
        .encode_upload(
            &payload,
            HostTransferLayout::new(ElementType::U8, 8).unwrap(),
            &source,
            0,
        )
        .unwrap();
    let copy = runtime
        .encode_copy(&source, &destination, CopyRegion::new(0, 0, 8).unwrap())
        .unwrap();
    let zero = runtime
        .encode_zero(&destination, 2, 3)
        .unwrap()
        .bind_core_logical_work(
            DeviceCommandLogicalWork::new(DeviceBatchingForm::Packed, 2, 7).unwrap(),
        )
        .unwrap();
    let expected = [
        (K::HostToDevice, 8, 0),
        (K::DeviceToDevice, 8, 0),
        (K::Fill, 3, 7),
    ]
    .map(|(kind, size, tokens)| {
        runtime
            .cost_core_transfer_evidence(kind, size, tokens)
            .unwrap()
    });
    let fence = runtime
        .submit_commands_with_attribution(
            &mut stream,
            vec![
                (DeviceCommandPhase::DynamicBinding, None, upload),
                (DeviceCommandPhase::DynamicBinding, None, copy),
                (DeviceCommandPhase::Initialization, Some(0), zero),
            ],
            DeviceTimingMode::Off,
            true,
            &DisabledDeviceSubmissionTimingSink,
        )
        .unwrap();
    assert!(runtime
        .wait_fence(&fence)
        .unwrap()
        .terminal()
        .is_succeeded());
    let attribution = runtime.submission_attribution(&fence).unwrap();
    assert_eq!(attribution.commands().len(), 3);
    for (actual, expected) in attribution.commands().iter().zip(&expected) {
        assert_eq!(actual.statistical_evidence(), Some(expected));
        assert_eq!(actual.compute_dispatch_count(), 0);
        assert_eq!(actual.transfer_command_count(), 1);
    }
    assert_eq!(expected[0].work().host_to_device_bytes, 8);
    assert_eq!(expected[1].work().device_to_device_bytes, 8);
    assert_eq!(expected[2].work().fill_bytes, 3);
    let read = |buffer: &MetalDeviceBuffer| bytes(&buffer.region(0..8).unwrap());
    assert_eq!(read(&source), payload);
    assert_eq!(read(&destination), [1, 2, 0, 0, 0, 6, 7, 8]);
    assert!(runtime.cost_core_transfer_evidence(K::Fill, 0, 0).is_none());
    assert!(runtime
        .cost_core_transfer_evidence(K::DeviceToHost, 8, 0)
        .is_none());
    assert_eq!(runtime.stream_state(&stream), StreamState::Ready);
}

#[test]
fn real_primitive_native_statistics_match_selected_psos_and_preserve_guarded_outputs() {
    let runtime = tests::runtime();
    let mut stream = runtime.create_stream().unwrap();
    for fixture in super::super::vnext_ops::selected_primitive_runtime_fixtures(&runtime) {
        let fence = runtime
            .submit_commands_with_attribution(
                &mut stream,
                vec![(DeviceCommandPhase::Compute, Some(0), fixture.command)],
                DeviceTimingMode::Off,
                true,
                &DisabledDeviceSubmissionTimingSink,
            )
            .unwrap();
        assert!(runtime
            .wait_fence(&fence)
            .unwrap()
            .terminal()
            .is_succeeded());
        let attribution = runtime.submission_attribution(&fence).unwrap();
        let [actual] = attribution.commands() else {
            panic!("one actual physical primitive command")
        };
        assert_eq!(
            actual.statistical_evidence(),
            fixture.projected.statistical_evidence()
        );
        assert!(actual.statistical_evidence().is_some());
        assert_eq!(
            actual.compute_dispatch_count(),
            fixture.projected.compute_dispatch_count()
        );
        assert_eq!(actual.transfer_command_count(), 0);
        assert_eq!(actual.token_count(), fixture.projected.token_count());
        for (region, expected) in fixture.checks {
            assert_eq!(bytes(&region), expected);
        }
        for (region, payload_size) in fixture.scratch_guards {
            let actual = bytes(&region);
            assert!(actual[..64].iter().all(|&v| v == 0xa5));
            assert!(actual[64 + payload_size..].iter().all(|&v| v == 0xa5));
        }
        assert_eq!(runtime.stream_state(&stream), StreamState::Ready);
    }
}
