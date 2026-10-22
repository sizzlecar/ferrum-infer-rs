use super::super::tests::{buffer, config};
use super::super::{
    DefinitelyNotSubmitted, DeviceCommandPhase, DeviceRuntime, ReferenceCommandKind,
    ReferenceDeviceCommand, ReferenceDeviceFence, ReferenceDeviceRuntime, StreamState,
};
use super::*;
use ferrum_runtime::DeviceLogicalWork;

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
    columns: usize,
    weight: &[f32],
    epsilon: CanonicalRational,
) -> ReferenceZeroCenteredRmsNormLaunch {
    assert_eq!(input.len() % columns, 0);
    let input_bytes = encoded(input);
    let weight_bytes = encoded(weight);
    let x = buffer(
        runtime,
        "resource.zero-centered-rms.x",
        &input_bytes,
        16,
        ElementType::Bf16,
    );
    let w = buffer(
        runtime,
        "resource.zero-centered-rms.weight",
        &weight_bytes,
        16,
        ElementType::Bf16,
    );
    let y = buffer(
        runtime,
        "resource.zero-centered-rms.y",
        &vec![0xa5; input_bytes.len()],
        16,
        ElementType::Bf16,
    );
    ReferenceZeroCenteredRmsNormLaunch {
        input: x.region(0..input_bytes.len() as u64).unwrap(),
        weight: w.region(0..weight_bytes.len() as u64).unwrap(),
        output: y.region(0..input_bytes.len() as u64).unwrap(),
        shape: ReferenceZeroCenteredRmsNormShape {
            rows: input.len() / columns,
            columns,
            epsilon,
        },
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
fn single_final_bf16_boundary_differs_from_head_rms_and_has_exact_tensor_work() {
    let runtime = ReferenceDeviceRuntime::new(config()).unwrap();
    // Effective mean+epsilon=0x3fe38e39, sqrt=0x3faaaaab, reciprocal=3/4.
    // normalized=387/512 and weight+1=129/128. With a single final B,
    // B(49923/65536)=195/256 (3f43). An inserted B(normalized) gives 3f44.
    let epsilon = CanonicalRational::new(112375, 147456).unwrap();
    let effective = (epsilon.numerator() as f64 / epsilon.denominator() as f64) as f32;
    let mean = (129.0_f32 / 128.0) * (129.0_f32 / 128.0);
    assert_eq!((mean + effective).to_bits(), 0x3fe38e39);
    assert_eq!((1.0_f32 / (mean + effective).sqrt()).to_bits(), 0x3f400000);
    let launch = fixture(
        &runtime,
        &[129.0 / 128.0; 128],
        128,
        &[1.0 / 128.0; 128],
        epsilon,
    );
    let output = launch.output.clone();
    let command = ReferenceDeviceCommand::zero_centered_rms_norm_tensor(launch).unwrap();
    let work = command
        .attribution(0, Some(0), DeviceCommandPhase::Compute)
        .unwrap();
    assert_eq!(work.token_count(), None);
    assert_eq!(
        work.logical_work(),
        DeviceLogicalWork::Tensor {
            element_count: 128,
            arithmetic_operations: 643
        }
    );
    let before = runtime.snapshot();
    submit(&runtime, vec![command]).unwrap();
    assert_eq!(bits(&output), vec![0x3f43; 128]);
    let after = runtime.snapshot();
    assert_eq!(
        after.zero_centered_rms_norm_launches,
        before.zero_centered_rms_norm_launches + 1
    );
    assert_eq!(after.allocations, before.allocations);
    assert_eq!(after.head_rms_norm_launches, 0);
}

#[test]
fn learned_scale_offset_remains_fp32_until_final_product() {
    let runtime = ReferenceDeviceRuntime::new(config()).unwrap();
    // x=1, epsilon=7/9 gives reciprocal=3/4. The original BF16 weight
    // 1/256 becomes FP32 257/256; B((3/4)*(257/256))=193/256 (3f41).
    // Rounding weight+1 to BF16 prematurely would instead produce 3f40.
    let epsilon = CanonicalRational::new(7, 9).unwrap();
    let effective = (7.0_f64 / 9.0) as f32;
    assert_eq!(
        (1.0_f32 / (1.0_f32 + effective).sqrt()).to_bits(),
        0x3f400000
    );
    let launch = fixture(&runtime, &[1.0], 1, &[1.0 / 256.0], epsilon);
    let output = launch.output.clone();
    submit(
        &runtime,
        vec![ReferenceDeviceCommand::zero_centered_rms_norm_tensor(launch).unwrap()],
    )
    .unwrap();
    assert_eq!(bits(&output), [0x3f41]);
}

#[test]
fn whole_rows_and_odd_width_use_original_weight_without_head_repetition() {
    let runtime = ReferenceDeviceRuntime::new(config()).unwrap();
    let launch = fixture(
        &runtime,
        &[0.5, 0.5, -0.5, -0.5, -0.5, -0.5, 0.5, 0.5],
        4,
        &[1.0, 3.0, 5.0, 7.0],
        CanonicalRational::new(3, 4).unwrap(),
    );
    let output = launch.output.clone();
    submit(
        &runtime,
        vec![ReferenceDeviceCommand::zero_centered_rms_norm_tensor(launch).unwrap()],
    )
    .unwrap();
    assert_eq!(
        bits(&output),
        [0x3f80, 0x4000, 0xc040, 0xc080, 0xbf80, 0xc000, 0x4040, 0x4080]
    );
    let odd = fixture(
        &runtime,
        &[0.5; 3],
        3,
        &[0.0; 3],
        CanonicalRational::new(3, 4).unwrap(),
    );
    let output = odd.output.clone();
    submit(
        &runtime,
        vec![ReferenceDeviceCommand::zero_centered_rms_norm_tensor(odd).unwrap()],
    )
    .unwrap();
    assert_eq!(bits(&output), [0x3f00; 3]);
}

#[test]
fn exact_regions_reject_shape_dtype_alignment_and_alias_before_writes() {
    let runtime = ReferenceDeviceRuntime::new(config()).unwrap();
    for mutation in 0..8 {
        let mut launch = fixture(
            &runtime,
            &[0.5; 4],
            4,
            &[0.0; 4],
            CanonicalRational::new(3, 4).unwrap(),
        );
        let output = launch.output.clone();
        match mutation {
            0 => launch.shape.rows = 0,
            1 => launch.shape.epsilon = CanonicalRational::new(-1, 1).unwrap(),
            2 => launch.weight.element_type = ElementType::F16,
            3 => launch.weight.length_bytes -= 2,
            4 => launch.input.offset_bytes += 1,
            5 => launch.output.length_bytes -= 2,
            6 => launch.output = launch.input.clone(),
            7 => launch.output = launch.weight.clone(),
            _ => unreachable!(),
        }
        assert!(ReferenceDeviceCommand::zero_centered_rms_norm_tensor(launch).is_err());
        assert_eq!(output.read(), [0xa5; 8]);
    }
    let shape = ReferenceZeroCenteredRmsNormShape {
        rows: 1,
        columns: 1,
        epsilon: CanonicalRational::new(1, 1).unwrap(),
    };
    assert_eq!(shape.byte_extents().unwrap(), (2, 2));
    assert!(ReferenceZeroCenteredRmsNormShape {
        rows: usize::MAX,
        ..shape
    }
    .byte_extents()
    .is_err());
    assert!(ReferenceZeroCenteredRmsNormShape {
        columns: (1 << 24) + 1,
        ..shape
    }
    .byte_extents()
    .is_err());
    if let Ok(rows) = usize::try_from(u64::MAX / 8 + 1) {
        assert!(ReferenceZeroCenteredRmsNormShape { rows, ..shape }
            .logical_work()
            .is_err());
    }
}

#[test]
fn later_foreign_or_short_weight_prevents_entire_batch_execution() {
    let runtime = ReferenceDeviceRuntime::new(config()).unwrap();
    let foreign = ReferenceDeviceRuntime::new(config()).unwrap();
    for bad_foreign in [false, true] {
        let make = |runtime| {
            fixture(
                runtime,
                &[0.5; 4],
                4,
                &[0.0; 4],
                CanonicalRational::new(3, 4).unwrap(),
            )
        };
        let first = make(&runtime);
        let output = first.output.clone();
        let mut second =
            ReferenceDeviceCommand::zero_centered_rms_norm_tensor(make(&runtime)).unwrap();
        let ReferenceCommandKind::ZeroCenteredRmsNorm { launch } = &mut second.kind else {
            unreachable!()
        };
        if bad_foreign {
            launch.weight = make(&foreign).weight;
        } else {
            launch.weight.length_bytes -= 2;
        }
        let sentinel = buffer(
            &runtime,
            "resource.zero-centered-rms.sentinel",
            &[0xa5; 4],
            16,
            ElementType::U8,
        );
        let commands = vec![
            runtime.encode_zero(&sentinel, 0, 4).unwrap(),
            ReferenceDeviceCommand::zero_centered_rms_norm_tensor(first).unwrap(),
            second,
        ];
        let mut stream = runtime.create_stream().unwrap();
        let before = runtime.snapshot();
        assert!(runtime
            .submit_entries(
                &mut stream,
                commands
                    .into_iter()
                    .map(|c| (DeviceCommandPhase::Compute, Some(0), c))
                    .collect(),
                false
            )
            .is_err());
        assert_eq!(runtime.stream_state(&stream), StreamState::Ready);
        let after = runtime.snapshot();
        assert_eq!(after.submissions, before.submissions);
        assert_eq!(after.commands, before.commands);
        assert_eq!(
            after.zero_centered_rms_norm_launches,
            before.zero_centered_rms_norm_launches
        );
        assert_eq!(sentinel.region(0..4).unwrap().read(), [0xa5; 4]);
        assert_eq!(output.read(), [0xa5; 8]);
    }
}

#[test]
fn ieee_propagation_is_not_replaced_by_a_finite_payload_scan() {
    let runtime = ReferenceDeviceRuntime::new(config()).unwrap();
    for input in [[f32::INFINITY, 1.0], [f32::NAN, 1.0]] {
        let launch = fixture(
            &runtime,
            &input,
            2,
            &[0.0; 2],
            CanonicalRational::new(1, 1).unwrap(),
        );
        let output = launch.output.clone();
        submit(
            &runtime,
            vec![ReferenceDeviceCommand::zero_centered_rms_norm_tensor(launch).unwrap()],
        )
        .unwrap();
        let result = bits(&output);
        assert!(bf16::from_bits(result[0]).is_nan());
        if input[0].is_infinite() {
            assert_eq!(result[1], 0);
        } else {
            assert!(bf16::from_bits(result[1]).is_nan());
        }
    }
    let launch = fixture(
        &runtime,
        &[0.0, -0.0, 0.0, -0.0],
        4,
        &[0.0, 0.0, -2.0, -2.0],
        CanonicalRational::new(1, 1).unwrap(),
    );
    let output = launch.output.clone();
    submit(
        &runtime,
        vec![ReferenceDeviceCommand::zero_centered_rms_norm_tensor(launch).unwrap()],
    )
    .unwrap();
    assert_eq!(bits(&output), [0, 0x8000, 0x8000, 0]);
}
