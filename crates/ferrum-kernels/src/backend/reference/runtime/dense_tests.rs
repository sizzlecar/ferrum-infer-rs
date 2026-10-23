//! Actual reference arithmetic and the submission preflight boundary. Test
//! buffers bypass admission only to construct malformed private commands; the
//! integration fixtures exercise real runtime admission and managed IO.

use super::tests::{buffer, config};
use super::*;
use ferrum_runtime::DeviceLogicalWork;
use std::sync::{mpsc, Barrier};
use std::time::Duration;

fn encode(values: &[f32], scalar: ReferenceDenseScalar) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| scalar.encode(*value).to_le_bytes())
        .collect()
}

fn fixture(
    runtime: &ReferenceDeviceRuntime,
    scalar: ReferenceDenseScalar,
) -> ReferenceDenseLinearLaunch {
    let input = buffer(
        runtime,
        "resource.dense.input",
        &encode(&[1., 2., 3., -1., 0., 2.], scalar),
        16,
        scalar.element_type(),
    );
    let weight = buffer(
        runtime,
        "resource.dense.weight",
        &encode(&[2., -1., 1., 1., 3., -2.], scalar),
        16,
        scalar.element_type(),
    );
    let output = buffer(
        runtime,
        "resource.dense.output",
        &[0xa5; 8],
        16,
        scalar.element_type(),
    );
    ReferenceDenseLinearLaunch {
        input: input.region(0..12).unwrap(),
        weight: weight.region(0..12).unwrap(),
        output: output.region(0..8).unwrap(),
        rows: 2,
        in_features: 3,
        out_features: 2,
        scalar,
    }
}

fn submit(
    runtime: &ReferenceDeviceRuntime,
    commands: Vec<ReferenceDeviceCommand>,
) -> Result<ReferenceDeviceFence, DefinitelyNotSubmitted<ReferenceDeviceRuntimeError>> {
    let mut stream = runtime.create_stream().unwrap();
    runtime.submit_entries(
        &mut stream,
        commands
            .into_iter()
            .map(|command| (DeviceCommandPhase::Compute, Some(0), command))
            .collect(),
        false,
    )
}

#[test]
fn dense_direct_storage_covers_all_five_allocation_partitions_and_preserves_sentinels() {
    for scalar in [ReferenceDenseScalar::F16, ReferenceDenseScalar::Bf16] {
        // I/W/O all distinct; I=W; I=O; W=O; I=W=O. Equal allocations
        // still contain distinct, adjacent operand ranges in this test.
        for roles in [[0, 1, 2], [0, 0, 1], [0, 1, 0], [0, 1, 1], [0, 0, 0]] {
            let runtime = ReferenceDeviceRuntime::new(config()).unwrap();
            let buffers = ["resource.dense.a", "resource.dense.b", "resource.dense.c"]
                .map(|id| buffer(&runtime, id, &[0xa5; 40], 16, scalar.element_type()));
            let input = buffers[roles[0]].region(2..14).unwrap();
            let weight = buffers[roles[1]].region(14..26).unwrap();
            let output = buffers[roles[2]].region(26..34).unwrap();
            input.write(&encode(&[1., 2., 3., -1., 0., 2.], scalar));
            weight.write(&encode(&[2., -1., 1., 1., 3., -2.], scalar));
            let before = buffers
                .each_ref()
                .map(|buffer| buffer.region(0..40).unwrap().read());
            let launch = ReferenceDenseLinearLaunch {
                input,
                weight,
                output: output.clone(),
                rows: 2,
                in_features: 3,
                out_features: 2,
                scalar,
            };
            submit(
                &runtime,
                vec![ReferenceDeviceCommand::dense_tensor(launch).unwrap()],
            )
            .unwrap();
            assert_eq!(output.read(), encode(&[3., 1., 0., -5.], scalar));
            for (index, buffer) in buffers.iter().enumerate() {
                let actual = buffer.region(0..40).unwrap().read();
                for byte in 0..40 {
                    if index != roles[2] || !(26..34).contains(&byte) {
                        assert_eq!(
                            actual[byte], before[index][byte],
                            "{scalar:?}, roles={roles:?}, byte={byte}"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn dense_allows_identical_and_partially_overlapping_read_operands() {
    let runtime = ReferenceDeviceRuntime::new(config()).unwrap();
    for (weight_range, expected) in [(2..8, 14.), (4..10, 20.)] {
        let storage = buffer(
            &runtime,
            "resource.dense.read-alias",
            &[0; 14],
            16,
            ElementType::F16,
        );
        storage
            .region(2..10)
            .unwrap()
            .write(&encode(&[1., 2., 3., 4.], ReferenceDenseScalar::F16));
        let output = storage.region(10..12).unwrap();
        let launch = ReferenceDenseLinearLaunch {
            input: storage.region(2..8).unwrap(),
            weight: storage.region(weight_range).unwrap(),
            output: output.clone(),
            rows: 1,
            in_features: 3,
            out_features: 1,
            scalar: ReferenceDenseScalar::F16,
        };
        submit(
            &runtime,
            vec![ReferenceDeviceCommand::dense_tensor(launch).unwrap()],
        )
        .unwrap();
        assert_eq!(
            output.read(),
            encode(&[expected], ReferenceDenseScalar::F16)
        );
    }
}

#[test]
fn dense_bf16_uses_fp32_accumulation_and_final_ties_even_without_f16_conversion() {
    let runtime = ReferenceDeviceRuntime::new(config()).unwrap();
    let scalar = ReferenceDenseScalar::Bf16;
    let input = buffer(
        &runtime,
        "resource.bf16.input",
        &encode(&[1., 1., 1.], scalar),
        16,
        ElementType::Bf16,
    );
    let weight = buffer(
        &runtime,
        "resource.bf16.weight",
        &encode(
            &[
                1.,
                1. / 256.,
                0.,
                1. + 1. / 128.,
                1. / 256.,
                0.,
                256.,
                1.,
                -256.,
                65536.,
                0.,
                0.,
                -1.,
                -1. / 256.,
                0.,
            ],
            scalar,
        ),
        16,
        ElementType::Bf16,
    );
    let output = buffer(
        &runtime,
        "resource.bf16.output",
        &[0; 10],
        16,
        ElementType::Bf16,
    );
    let launch = ReferenceDenseLinearLaunch {
        input: input.region(0..6).unwrap(),
        weight: weight.region(0..30).unwrap(),
        output: output.region(0..10).unwrap(),
        rows: 1,
        in_features: 3,
        out_features: 5,
        scalar,
    };
    let command = ReferenceDeviceCommand::dense_tensor(launch).unwrap();
    let attribution = command
        .attribution(0, Some(0), DeviceCommandPhase::Compute)
        .unwrap();
    assert_eq!(attribution.token_count(), None);
    assert_eq!(
        attribution.logical_work(),
        DeviceLogicalWork::Tensor {
            element_count: 5,
            arithmetic_operations: 30
        }
    );
    submit(&runtime, vec![command]).unwrap();
    // Hand-derived BF16 encodings: midpoint rounds down/up to even,
    // cancellation retains the FP32 +1, and 65536 remains finite.
    let expected: Vec<u8> = [0x3f80_u16, 0x3f82, 0x3f80, 0x4780, 0xbf80]
        .into_iter()
        .flat_map(u16::to_le_bytes)
        .collect();
    assert_eq!(output.region(0..10).unwrap().read(), expected);
}

#[test]
fn dense_rejects_output_aliases_mixed_precision_zero_overflow_and_out_of_range_before_execution() {
    let runtime = ReferenceDeviceRuntime::new(config()).unwrap();
    for mutation in 0..13 {
        let mut launch = fixture(&runtime, ReferenceDenseScalar::F16);
        let output = launch.output.clone();
        match mutation {
            0 => {
                launch.output = launch.input.clone();
                launch.rows = 2;
                launch.out_features = 3;
            }
            1 => {
                launch.output = launch.input.clone();
                launch.output.length_bytes = 8;
            }
            2 => {
                launch.output = launch.weight.clone();
                launch.output.offset_bytes += 2;
                launch.output.length_bytes = 8;
            }
            3 => launch.weight.element_type = ElementType::Bf16,
            4 => launch.scalar = ReferenceDenseScalar::Bf16,
            5 => launch.rows = 0,
            6 => launch.in_features = usize::MAX,
            7 => launch.rows = usize::MAX / 2 + 1,
            8 => launch.input.offset_bytes = usize::MAX,
            9 => launch.input.offset_bytes += 1,
            10 => launch.output.length_bytes -= 2,
            11 => launch.input.element_type = ElementType::Bf16,
            12 => launch.output.element_type = ElementType::Bf16,
            _ => unreachable!(),
        }
        assert!(
            ReferenceDeviceCommand::dense_tensor(launch).is_err(),
            "mutation={mutation}"
        );
        assert_eq!(output.read(), [0xa5; 8]);
    }
    assert_eq!(runtime.snapshot().dense_linear_launches, 0);
    assert_eq!(runtime.snapshot().submissions, 0);
}

#[test]
fn dense_rejects_exact_and_partial_output_overlap_with_either_read_operand() {
    let runtime = ReferenceDeviceRuntime::new(config()).unwrap();
    let storage = buffer(
        &runtime,
        "resource.dense.output-alias",
        &[0; 24],
        16,
        ElementType::F16,
    );
    for output in [0..8, 2..10, 8..16, 10..18] {
        let launch = ReferenceDenseLinearLaunch {
            input: storage.region(0..8).unwrap(),
            weight: storage.region(8..16).unwrap(),
            output: storage.region(output).unwrap(),
            rows: 2,
            in_features: 2,
            out_features: 2,
            scalar: ReferenceDenseScalar::F16,
        };
        assert!(ReferenceDeviceCommand::dense_tensor(launch).is_err());
    }
    assert_eq!(storage.region(0..24).unwrap().read(), [0; 24]);
}

#[test]
fn invalid_late_launch_or_command_prevents_all_earlier_device_effects() {
    let runtime = ReferenceDeviceRuntime::new(config()).unwrap();
    for invalid_inside_first_command in [true, false] {
        let good = fixture(&runtime, ReferenceDenseScalar::F16);
        let good_output = good.output.clone();
        let bad = fixture(&runtime, ReferenceDenseScalar::F16);
        let bad_output = bad.output.clone();
        let mut command = ReferenceDeviceCommand::dense_tensor(bad).unwrap();
        let ReferenceCommandKind::DenseLinear { launches } = &mut command.kind else {
            unreachable!()
        };
        launches[0].weight.element_type = ElementType::Bf16;
        let commands = if invalid_inside_first_command {
            let ReferenceCommandKind::DenseLinear { launches } = command.kind else {
                unreachable!()
            };
            command.kind = ReferenceCommandKind::DenseLinear {
                launches: std::iter::once(good).chain(launches).collect(),
            };
            vec![command]
        } else {
            vec![ReferenceDeviceCommand::dense_tensor(good).unwrap(), command]
        };
        let sentinel = buffer(
            &runtime,
            "resource.preflight.sentinel",
            &[0xa5; 4],
            16,
            ElementType::U8,
        );
        let zero = runtime.encode_zero(&sentinel, 0, 4).unwrap();
        let mut entries = vec![(DeviceCommandPhase::Compute, Some(0), zero)];
        entries.extend(
            commands
                .into_iter()
                .map(|command| (DeviceCommandPhase::Compute, Some(0), command)),
        );
        let mut stream = runtime.create_stream().unwrap();
        let error = runtime
            .submit_entries(&mut stream, entries, false)
            .unwrap_err();
        let _: ReferenceDeviceRuntimeError = error.into_error();
        assert_eq!(runtime.stream_state(&stream), StreamState::Ready);
        assert_eq!(sentinel.region(0..4).unwrap().read(), [0xa5; 4]);
        assert_eq!(good_output.read(), [0xa5; 8]);
        assert_eq!(bad_output.read(), [0xa5; 8]);
        assert_eq!(runtime.snapshot().commands, 0);
        assert_eq!(runtime.snapshot().dense_linear_launches, 0);
        assert_eq!(runtime.snapshot().submissions, 0);
    }
}

#[test]
fn reverse_role_order_concurrent_dense_commands_share_one_global_lock_order() {
    let runtime = Arc::new(ReferenceDeviceRuntime::new(config()).unwrap());
    let a = buffer(&runtime, "resource.lock.a", &[0; 12], 16, ElementType::F16);
    let b = buffer(&runtime, "resource.lock.b", &[0; 12], 16, ElementType::F16);
    a.region(0..6)
        .unwrap()
        .write(&encode(&[1., 2., 3.], ReferenceDenseScalar::F16));
    b.region(0..6)
        .unwrap()
        .write(&encode(&[2., 3., 4.], ReferenceDenseScalar::F16));
    let launches = [(&a, &b), (&b, &a)].map(|(input, weight)| ReferenceDenseLinearLaunch {
        input: input.region(0..6).unwrap(),
        weight: weight.region(0..6).unwrap(),
        output: weight.region(8..10).unwrap(),
        rows: 1,
        in_features: 3,
        out_features: 1,
        scalar: ReferenceDenseScalar::F16,
    });
    let slots = |launch: &ReferenceDenseLinearLaunch| {
        AllocationLocks::new([&launch.input, &launch.weight, &launch.output]).slots
    };
    let first_slots = slots(&launches[0]);
    let second_slots = slots(&launches[1]);
    assert_ne!(first_slots[0], first_slots[1]);
    assert_eq!(first_slots[0], second_slots[1]);
    assert_eq!(first_slots[1], second_slots[0]);
    assert_eq!(first_slots[1], first_slots[2]);
    assert_eq!(second_slots[1], second_slots[2]);
    let barrier = Arc::new(Barrier::new(3));
    let (sent, received) = mpsc::channel();
    let handles = launches.map(|launch| {
        let runtime = runtime.clone();
        let barrier = barrier.clone();
        let sent = sent.clone();
        std::thread::spawn(move || {
            let command = ReferenceDeviceCommand::dense_tensor(launch).unwrap();
            barrier.wait();
            submit(&runtime, vec![command]).unwrap();
            sent.send(()).unwrap();
        })
    });
    barrier.wait();
    for _ in 0..2 {
        received
            .recv_timeout(Duration::from_secs(10))
            .expect("opposite operand roles must not deadlock");
    }
    for handle in handles {
        handle.join().unwrap();
    }
    for buffer in [&a, &b] {
        assert_eq!(
            buffer.region(8..10).unwrap().read(),
            encode(&[20.], ReferenceDenseScalar::F16)
        );
    }
}
