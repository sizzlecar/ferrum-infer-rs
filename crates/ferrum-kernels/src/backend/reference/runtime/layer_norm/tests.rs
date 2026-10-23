//! Real Reference storage/arithmetic and malformed private submission tests.
//! Managed input admission is exercised by the separate executor integration.

use super::super::tests::{buffer, config};
use super::super::{
    DefinitelyNotSubmitted, DeviceCommandPhase, DeviceRuntime, ReferenceCommandKind,
    ReferenceDeviceCommand, ReferenceDeviceFence, ReferenceDeviceRuntime, StreamState,
};
use super::*;
use ferrum_runtime::DeviceLogicalWork;
use std::sync::{mpsc, Arc, Barrier};
use std::time::Duration;

fn encoded(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| bf16::from_f32(*value).to_bits().to_le_bytes())
        .collect()
}

fn bits(region: &ReferenceBufferRegion) -> Vec<u16> {
    region
        .read()
        .chunks_exact(2)
        .map(|chunk| u16::from_le_bytes([chunk[0], chunk[1]]))
        .collect()
}

fn fixture(
    runtime: &ReferenceDeviceRuntime,
    input: &[f32],
    width: usize,
    epsilon: CanonicalRational,
) -> ReferenceLayerNormLaunch {
    assert_eq!(input.len() % width, 0);
    let bytes = encoded(input);
    let input = buffer(
        runtime,
        "resource.norm.input",
        &bytes,
        16,
        ElementType::Bf16,
    );
    let output = buffer(
        runtime,
        "resource.norm.output",
        &vec![0xa5; bytes.len()],
        16,
        ElementType::Bf16,
    );
    ReferenceLayerNormLaunch {
        input: input.region(0..bytes.len() as u64).unwrap(),
        output: output.region(0..bytes.len() as u64).unwrap(),
        rows: bytes.len() / 2 / width,
        hidden_size: width,
        epsilon,
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
fn exact_population_norm_fixtures_cover_constant_shift_and_small_variance() {
    let runtime = ReferenceDeviceRuntime::new(config()).unwrap();
    let cases = [
        (vec![4., 4., 4., 4.], 4, (3, 4), vec![0; 4]),
        (vec![0., -0., 9., -9.], 1, (1, 1), vec![0; 4]),
        (
            vec![1., 1., 2., 2., 128., 128., 129., 129.],
            4,
            (3, 4),
            vec![
                0xbf00, 0xbf00, 0x3f00, 0x3f00, 0xbf00, 0xbf00, 0x3f00, 0x3f00,
            ],
        ),
        (
            vec![-1. / 1024., 1. / 1024.],
            2,
            (3, 1 << 20),
            vec![0xbf00, 0x3f00],
        ),
        (
            vec![-1., -1., -1., 3., 1., 1., 1., 5.],
            4,
            (1, 1),
            vec![
                0xbf00, 0xbf00, 0xbf00, 0x3fc0, 0xbf00, 0xbf00, 0xbf00, 0x3fc0,
            ],
        ),
        // Non-power-of-two width: variance 2, sqrt(variance + 1/4) = 1.5.
        (vec![-1., -1., 2.], 3, (1, 4), vec![0xbf2b, 0xbf2b, 0x3fab]),
    ];
    for (input, width, (numerator, denominator), expected) in cases {
        let launch = fixture(
            &runtime,
            &input,
            width,
            CanonicalRational::new(numerator, denominator).unwrap(),
        );
        let output = launch.output.clone();
        let command = ReferenceDeviceCommand::layer_norm_tensor(launch).unwrap();
        let attribution = command
            .attribution(0, Some(0), DeviceCommandPhase::Compute)
            .unwrap();
        assert_eq!(attribution.token_count(), None);
        assert_eq!(
            attribution.logical_work(),
            DeviceLogicalWork::Tensor {
                element_count: input.len() as u64,
                arithmetic_operations: (input.len() / width * (6 * width + 3)) as u64,
            }
        );
        submit(&runtime, vec![command]).unwrap();
        assert_eq!(bits(&output), expected);
    }
    assert_eq!(runtime.snapshot().layer_norm_launches, 6);
    assert_eq!(runtime.snapshot().dense_linear_launches, 0);
}

#[test]
fn width_4096_near_constant_row_has_independently_derived_exact_output() {
    let runtime = ReferenceDeviceRuntime::new(config()).unwrap();
    let mut input = vec![1_f32; 4096];
    input[4095] += 1. / 128.;
    // mu=1+2^-19 and variance=4095/2^38. Effective epsilon is exact;
    // variance+epsilon=2^-20, hence inverse sqrt=1024.
    let launch = fixture(
        &runtime,
        &input,
        4096,
        CanonicalRational::new(258049, 274877906944).unwrap(),
    );
    let output = launch.output.clone();
    submit(
        &runtime,
        vec![ReferenceDeviceCommand::layer_norm_tensor(launch).unwrap()],
    )
    .unwrap();
    let result = bits(&output);
    assert!(result[..4095].iter().all(|bits| *bits == 0xbb00));
    assert_eq!(result[4095], 0x4100);
}

#[test]
fn odd_pairwise_tree_is_floor_split_and_differs_from_left_fold() {
    let input = [16_777_216_f32, 1., -16_777_216.];
    assert_eq!(pairwise_sum(input.len(), 0, &|index| input[index]), 1.);
    assert_eq!(input.into_iter().fold(0_f32, |sum, value| sum + value), 0.);
    let five = [16_777_216_f32, 1., 0., -16_777_216., 1.];
    // floor split: (2^24 + 1) + (0 + (-2^24 + 1)) = 1.
    assert_eq!(pairwise_sum(five.len(), 0, &|index| five[index]), 1.);
}

#[test]
fn direct_storage_handles_both_allocation_partitions_nonzero_offsets_and_sentinels() {
    let runtime = ReferenceDeviceRuntime::new(config()).unwrap();
    for same_allocation in [false, true] {
        let a = buffer(
            &runtime,
            "resource.norm.a",
            &[0xa5; 20],
            16,
            ElementType::Bf16,
        );
        let b = buffer(
            &runtime,
            "resource.norm.b",
            &[0xa5; 20],
            16,
            ElementType::Bf16,
        );
        let input = a.region(2..10).unwrap();
        let output = if same_allocation { &a } else { &b }
            .region(10..18)
            .unwrap();
        input.write(&encoded(&[1., 1., 2., 2.]));
        let before_a = a.region(0..20).unwrap().read();
        let before_b = b.region(0..20).unwrap().read();
        let launch = ReferenceLayerNormLaunch {
            input,
            output: output.clone(),
            rows: 1,
            hidden_size: 4,
            epsilon: CanonicalRational::new(3, 4).unwrap(),
        };
        submit(
            &runtime,
            vec![ReferenceDeviceCommand::layer_norm_tensor(launch).unwrap()],
        )
        .unwrap();
        assert_eq!(bits(&output), [0xbf00, 0xbf00, 0x3f00, 0x3f00]);
        for (index, (buffer, before)) in [(&a, before_a), (&b, before_b)].into_iter().enumerate() {
            let after = buffer.region(0..20).unwrap().read();
            for byte in 0..20 {
                let output_buffer = if same_allocation { 0 } else { 1 };
                if index != output_buffer || !(10..18).contains(&byte) {
                    assert_eq!(after[byte], before[byte]);
                }
            }
        }
    }
}

#[test]
fn launch_rejects_exact_and_partial_output_aliases_before_writing() {
    let runtime = ReferenceDeviceRuntime::new(config()).unwrap();
    let storage = buffer(
        &runtime,
        "resource.norm.alias",
        &[0xa5; 24],
        16,
        ElementType::Bf16,
    );
    for output in [2..10, 0..8, 4..12, 8..16] {
        let launch = ReferenceLayerNormLaunch {
            input: storage.region(2..10).unwrap(),
            output: storage.region(output).unwrap(),
            rows: 1,
            hidden_size: 4,
            epsilon: CanonicalRational::new(3, 4).unwrap(),
        };
        assert!(ReferenceDeviceCommand::layer_norm_tensor(launch).is_err());
    }
    assert_eq!(storage.region(0..24).unwrap().read(), [0xa5; 24]);
}

#[test]
fn structural_limits_epsilon_and_byte_extents_are_checked_without_large_allocations() {
    let epsilon = CanonicalRational::new(1, 1).unwrap();
    assert_eq!(validate_parameters(1, 1, epsilon).unwrap(), 2);
    assert_eq!(validate_parameters(1, 1 << 24, epsilon).unwrap(), 1 << 25);
    for (rows, width) in [(0, 1), (1, 0), (1, (1 << 24) + 1), (usize::MAX, 1)] {
        assert!(validate_parameters(rows, width, epsilon).is_err());
    }
    for (numerator, denominator, valid) in [
        (1, 1_000_000_000_000, true),
        (1, 1, true),
        (0, 1, false),
        (-1, 1, false),
        (2, 1, false),
        (1, 1_000_000_000_001, false),
    ] {
        assert_eq!(
            validate_parameters(
                1,
                1,
                CanonicalRational::new(numerator, denominator).unwrap()
            )
            .is_ok(),
            valid
        );
    }
    let runtime = ReferenceDeviceRuntime::new(config()).unwrap();
    for mutation in 0..5 {
        let mut launch = fixture(&runtime, &[1., 1., 2., 2.], 4, epsilon);
        match mutation {
            0 => launch.output.element_type = ElementType::F16,
            1 => launch.input.element_type = ElementType::F16,
            2 => launch.input.offset_bytes += 1,
            3 => launch.output.length_bytes -= 2,
            4 => launch.rows = 2,
            _ => unreachable!(),
        }
        assert!(ReferenceDeviceCommand::layer_norm_tensor(launch).is_err());
    }
    let mut launch = fixture(&runtime, &[1.], 1, epsilon);
    launch.rows = usize::MAX;
    assert!(launch.logical_work().is_err());
}

#[test]
fn late_invalid_norm_command_preserves_earlier_commands_outputs_and_stream() {
    let runtime = ReferenceDeviceRuntime::new(config()).unwrap();
    let foreign = ReferenceDeviceRuntime::new(config()).unwrap();
    for foreign_input in [false, true] {
        let first = fixture(
            &runtime,
            &[1., 1., 2., 2.],
            4,
            CanonicalRational::new(3, 4).unwrap(),
        );
        let first_output = first.output.clone();
        let later = fixture(
            &runtime,
            &[1., 1., 2., 2.],
            4,
            CanonicalRational::new(3, 4).unwrap(),
        );
        let later_output = later.output.clone();
        let mut bad = ReferenceDeviceCommand::layer_norm_tensor(later).unwrap();
        let ReferenceCommandKind::LayerNorm { launch } = &mut bad.kind else {
            unreachable!()
        };
        if foreign_input {
            launch.input = fixture(
                &foreign,
                &[1., 1., 2., 2.],
                4,
                CanonicalRational::new(3, 4).unwrap(),
            )
            .input;
        } else {
            launch.epsilon = CanonicalRational::new(-1, 1).unwrap();
        }
        let sentinel = buffer(
            &runtime,
            "resource.norm.preflight",
            &[0xa5; 4],
            16,
            ElementType::U8,
        );
        let zero = runtime.encode_zero(&sentinel, 0, 4).unwrap();
        let entries = vec![
            zero,
            ReferenceDeviceCommand::layer_norm_tensor(first).unwrap(),
            bad,
        ]
        .into_iter()
        .map(|command| (DeviceCommandPhase::Compute, Some(0), command))
        .collect();
        let mut stream = runtime.create_stream().unwrap();
        let before = runtime.snapshot();
        let error = runtime
            .submit_entries(&mut stream, entries, false)
            .unwrap_err();
        let _: ReferenceDeviceRuntimeError = error.into_error();
        assert_eq!(runtime.stream_state(&stream), StreamState::Ready);
        let after = runtime.snapshot();
        assert_eq!(after.submissions, before.submissions);
        assert_eq!(after.commands, before.commands);
        assert_eq!(after.uploaded_bytes, before.uploaded_bytes);
        assert_eq!(after.dense_linear_launches, before.dense_linear_launches);
        assert_eq!(after.layer_norm_launches, before.layer_norm_launches);
        assert_eq!(sentinel.region(0..4).unwrap().read(), [0xa5; 4]);
        assert_eq!(first_output.read(), [0xa5; 8]);
        assert_eq!(later_output.read(), [0xa5; 8]);
    }
}

#[test]
fn numeric_nonfinite_and_overflow_follow_ieee_not_structural_rejection() {
    let runtime = ReferenceDeviceRuntime::new(config()).unwrap();
    for value in [f32::NAN, f32::INFINITY, bf16::MAX.to_f32()] {
        // Equal maximum finite inputs overflow the mean tree; they are outside
        // the finite oracle domain even though structural submission succeeds.
        let launch = fixture(
            &runtime,
            &[value, value],
            2,
            CanonicalRational::new(1, 1).unwrap(),
        );
        let output = launch.output.clone();
        submit(
            &runtime,
            vec![ReferenceDeviceCommand::layer_norm_tensor(launch).unwrap()],
        )
        .unwrap();
        assert!(bits(&output)
            .into_iter()
            .all(|value| bf16::from_bits(value).is_nan()));
    }
    assert_eq!(runtime.snapshot().submissions, 3);
    assert_eq!(runtime.snapshot().layer_norm_launches, 3);
}

#[test]
fn opposite_input_output_allocation_roles_share_global_lock_order() {
    let runtime = Arc::new(ReferenceDeviceRuntime::new(config()).unwrap());
    let a = buffer(
        &runtime,
        "resource.norm.concurrent-a",
        &[0xa5; 16],
        16,
        ElementType::Bf16,
    );
    let b = buffer(
        &runtime,
        "resource.norm.concurrent-b",
        &[0xa5; 16],
        16,
        ElementType::Bf16,
    );
    for source in [&a, &b] {
        source
            .region(0..8)
            .unwrap()
            .write(&encoded(&[1., 1., 2., 2.]));
    }
    let launches = [(&a, &b), (&b, &a)].map(|(input, output)| ReferenceLayerNormLaunch {
        input: input.region(0..8).unwrap(),
        output: output.region(8..16).unwrap(),
        rows: 1,
        hidden_size: 4,
        epsilon: CanonicalRational::new(3, 4).unwrap(),
    });
    let first_slots = AllocationLocks::new([&launches[0].input, &launches[0].output]).slots;
    let second_slots = AllocationLocks::new([&launches[1].input, &launches[1].output]).slots;
    assert_eq!(first_slots, [second_slots[1], second_slots[0]]);
    let barrier = Arc::new(Barrier::new(3));
    let (sent, received) = mpsc::channel();
    let handles = launches.map(|launch| {
        let runtime = runtime.clone();
        let barrier = barrier.clone();
        let sent = sent.clone();
        std::thread::spawn(move || {
            let command = ReferenceDeviceCommand::layer_norm_tensor(launch).unwrap();
            barrier.wait();
            submit(&runtime, vec![command]).unwrap();
            sent.send(()).unwrap();
        })
    });
    barrier.wait();
    for _ in 0..2 {
        received
            .recv_timeout(Duration::from_secs(10))
            .expect("opposite roles must not deadlock");
    }
    for handle in handles {
        handle.join().unwrap();
    }
    for output in [&a, &b] {
        assert_eq!(
            bits(&output.region(8..16).unwrap()),
            [0xbf00, 0xbf00, 0x3f00, 0x3f00]
        );
    }
}
