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
    gamma: &[f32],
    epsilon: CanonicalRational,
) -> ReferenceHeadRmsNormLaunch {
    assert_eq!(input.len() % columns, 0);
    let input_bytes = encoded(input);
    let gamma_bytes = encoded(gamma);
    let x = buffer(
        runtime,
        "resource.head-rms.x",
        &input_bytes,
        16,
        ElementType::Bf16,
    );
    let g = buffer(
        runtime,
        "resource.head-rms.gamma",
        &gamma_bytes,
        16,
        ElementType::Bf16,
    );
    let y = buffer(
        runtime,
        "resource.head-rms.y",
        &vec![0xa5; input_bytes.len()],
        16,
        ElementType::Bf16,
    );
    ReferenceHeadRmsNormLaunch {
        input: x.region(0..input_bytes.len() as u64).unwrap(),
        gamma: g.region(0..gamma_bytes.len() as u64).unwrap(),
        output: y.region(0..input_bytes.len() as u64).unwrap(),
        shape: ReferenceHeadRmsNormShape {
            rows: input.len() / columns,
            columns,
            head_dim: gamma.len(),
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
fn d128_first_bf16_boundary_has_exact_ties_even_counterexample_and_tensor_work() {
    let runtime = ReferenceDeviceRuntime::new(config()).unwrap();
    let input = vec![129.0 / 128.0; 128];
    // epsilon = 16/9 - (129/128)^2. The effective-FP32 sum is
    // 0x3fe38e39; sqrt rounds to 0x3faaaaab and reciprocal to exact 3/4.
    // x*3/4 = 387/512 ties to 194/256 at the FIRST BF16 boundary.
    // B((194/256)*(129/128)) = 196/256 = 0x3f44. Omitting that
    // boundary gives B((387/512)*(129/128)) = 195/256 = 0x3f43.
    let epsilon = CanonicalRational::new(112375, 147456).unwrap();
    let effective = (epsilon.numerator() as f64 / epsilon.denominator() as f64) as f32;
    let mean = (129.0_f32 / 128.0) * (129.0_f32 / 128.0);
    assert_eq!((mean + effective).to_bits(), 0x3fe38e39);
    assert_eq!((mean + effective).sqrt().to_bits(), 0x3faaaaab);
    assert_eq!((1.0_f32 / (mean + effective).sqrt()).to_bits(), 0x3f400000);
    let launch = fixture(&runtime, &input, 128, &input, epsilon);
    let output = launch.output.clone();
    let command = ReferenceDeviceCommand::head_rms_norm_tensor(launch).unwrap();
    let work = command
        .attribution(0, Some(0), DeviceCommandPhase::Compute)
        .unwrap();
    assert_eq!(work.token_count(), None);
    assert_eq!(
        work.logical_work(),
        DeviceLogicalWork::Tensor {
            element_count: 128,
            arithmetic_operations: 515
        }
    );
    let before = runtime.snapshot();
    submit(&runtime, vec![command]).unwrap();
    assert_eq!(bits(&output), vec![0x3f44; 128]);
    let after = runtime.snapshot();
    assert_eq!(
        after.head_rms_norm_launches,
        before.head_rms_norm_launches + 1
    );
    assert_eq!(after.allocations, before.allocations);
    assert_eq!(after.dense_linear_launches, 0);
}

#[test]
fn heads_and_rows_are_isolated_and_gamma_restarts_at_each_head() {
    let runtime = ReferenceDeviceRuntime::new(config()).unwrap();
    let launch = fixture(
        &runtime,
        &[0.5, -0.5, 1.375, -1.375, 0.0, -0.0, -0.5, 0.5],
        4,
        &[2.0, -1.0],
        CanonicalRational::new(3, 4).unwrap(),
    );
    let output = launch.output.clone();
    // Head RMS denominators are 1, 13/8, sqrt(3/4), and 1.
    // 11/13 rounds to BF16 217/256 (0x3f59), before gamma.
    submit(
        &runtime,
        vec![ReferenceDeviceCommand::head_rms_norm_tensor(launch).unwrap()],
    )
    .unwrap();
    assert_eq!(
        bits(&output),
        [0x3f80, 0x3f00, 0x3fd9, 0x3f59, 0, 0, 0xbf80, 0xbf00]
    );
    let odd = fixture(
        &runtime,
        &[1.0, 2.0, 2.0, -1.0, -2.0, -2.0],
        6,
        &[1.0, 2.0, -1.0],
        CanonicalRational::new(1, 1).unwrap(),
    );
    let odd_output = odd.output.clone();
    submit(
        &runtime,
        vec![ReferenceDeviceCommand::head_rms_norm_tensor(odd).unwrap()],
    )
    .unwrap();
    assert_eq!(
        bits(&odd_output),
        [0x3f00, 0x4000, 0xbf80, 0xbf00, 0xc000, 0x3f80]
    );
}

#[test]
fn shared_backing_nonzero_regions_preserve_sentinels_and_exact_byte_extents() {
    let runtime = ReferenceDeviceRuntime::new(config()).unwrap();
    let backing = buffer(
        &runtime,
        "resource.head-rms.shared",
        &[0xa5; 32],
        16,
        ElementType::Bf16,
    );
    let input = backing.region(2..10).unwrap();
    let gamma = backing.region(12..16).unwrap();
    let output = backing.region(20..28).unwrap();
    input.write(&encoded(&[0.5, -0.5, -0.5, 0.5]));
    gamma.write(&encoded(&[2.0, -1.0]));
    let before = backing.region(0..32).unwrap().read();
    let launch = ReferenceHeadRmsNormLaunch {
        input,
        gamma,
        output: output.clone(),
        shape: ReferenceHeadRmsNormShape {
            rows: 1,
            columns: 4,
            head_dim: 2,
            epsilon: CanonicalRational::new(3, 4).unwrap(),
        },
    };
    submit(
        &runtime,
        vec![ReferenceDeviceCommand::head_rms_norm_tensor(launch).unwrap()],
    )
    .unwrap();
    assert_eq!(bits(&output), [0x3f80, 0x3f00, 0xbf80, 0xbf00]);
    let after = backing.region(0..32).unwrap().read();
    for index in 0..32 {
        if !(20..28).contains(&index) {
            assert_eq!(after[index], before[index]);
        }
    }
}

#[test]
fn invalid_dimensions_regions_and_output_alias_are_rejected_without_writes() {
    let runtime = ReferenceDeviceRuntime::new(config()).unwrap();
    for mutation in 0..9 {
        let mut launch = fixture(
            &runtime,
            &[1.0; 4],
            4,
            &[1.0; 2],
            CanonicalRational::new(1, 1).unwrap(),
        );
        let output = launch.output.clone();
        match mutation {
            0 => launch.shape.head_dim = 3,
            1 => launch.shape.rows = 0,
            2 => launch.shape.epsilon = CanonicalRational::new(-1, 1).unwrap(),
            3 => launch.gamma.element_type = ElementType::F16,
            4 => launch.gamma.length_bytes -= 2,
            5 => launch.input.offset_bytes += 1,
            6 => launch.output.length_bytes -= 2,
            7 => launch.output = launch.input.clone(),
            8 => {
                launch.shape.columns = 2;
                launch.input.length_bytes = 4;
                launch.output = launch.gamma.clone();
            }
            _ => unreachable!(),
        }
        assert!(ReferenceDeviceCommand::head_rms_norm_tensor(launch).is_err());
        assert_eq!(output.read(), [0xa5; 8]);
    }
    let shape = ReferenceHeadRmsNormShape {
        rows: 1,
        columns: 1,
        head_dim: 1,
        epsilon: CanonicalRational::new(1, 1).unwrap(),
    };
    assert_eq!(shape.byte_extents().unwrap(), (2, 2));
    assert!(ReferenceHeadRmsNormShape {
        rows: usize::MAX,
        ..shape
    }
    .byte_extents()
    .is_err());
    assert!(ReferenceHeadRmsNormShape {
        columns: (1 << 24) + 1,
        head_dim: (1 << 24) + 1,
        ..shape
    }
    .byte_extents()
    .is_err());
    if let Ok(rows) = usize::try_from(u64::MAX / 7 + 1) {
        let huge = ReferenceHeadRmsNormShape { rows, ..shape };
        assert!(huge.logical_work().is_err());
    }
}

#[test]
fn later_invalid_gamma_preflight_prevents_all_prior_commands_and_stream_changes() {
    let runtime = ReferenceDeviceRuntime::new(config()).unwrap();
    let foreign = ReferenceDeviceRuntime::new(config()).unwrap();
    for bad_foreign in [false, true] {
        let first = fixture(
            &runtime,
            &[0.5; 4],
            4,
            &[1.0; 2],
            CanonicalRational::new(3, 4).unwrap(),
        );
        let first_output = first.output.clone();
        let second = fixture(
            &runtime,
            &[0.5; 4],
            4,
            &[1.0; 2],
            CanonicalRational::new(3, 4).unwrap(),
        );
        let second_output = second.output.clone();
        let mut second = ReferenceDeviceCommand::head_rms_norm_tensor(second).unwrap();
        let ReferenceCommandKind::HeadRmsNorm { launch } = &mut second.kind else {
            unreachable!()
        };
        if bad_foreign {
            launch.gamma = fixture(
                &foreign,
                &[0.5; 4],
                4,
                &[1.0; 2],
                CanonicalRational::new(3, 4).unwrap(),
            )
            .gamma;
        } else {
            launch.gamma.length_bytes -= 2;
        }
        let sentinel = buffer(
            &runtime,
            "resource.head-rms.sentinel",
            &[0xa5; 4],
            16,
            ElementType::U8,
        );
        let commands = vec![
            runtime.encode_zero(&sentinel, 0, 4).unwrap(),
            ReferenceDeviceCommand::head_rms_norm_tensor(first).unwrap(),
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
        assert_eq!(after.head_rms_norm_launches, before.head_rms_norm_launches);
        assert_eq!(after.uploaded_bytes, before.uploaded_bytes);
        assert_eq!(sentinel.region(0..4).unwrap().read(), [0xa5; 4]);
        assert_eq!(first_output.read(), [0xa5; 8]);
        assert_eq!(second_output.read(), [0xa5; 8]);
    }
}

#[test]
fn nonfinite_values_propagate_without_claiming_a_finite_oracle_domain() {
    let runtime = ReferenceDeviceRuntime::new(config()).unwrap();
    for value in [f32::NAN, f32::INFINITY] {
        let launch = fixture(
            &runtime,
            &[value; 2],
            2,
            &[1.0; 2],
            CanonicalRational::new(1, 1).unwrap(),
        );
        let output = launch.output.clone();
        submit(
            &runtime,
            vec![ReferenceDeviceCommand::head_rms_norm_tensor(launch).unwrap()],
        )
        .unwrap();
        assert!(bits(&output)
            .into_iter()
            .all(|bits| bf16::from_bits(bits).is_nan()));
    }
}
