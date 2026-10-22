use super::super::tests::{buffer, config};
use super::super::{
    DeviceCommandPhase, DeviceRuntime, ReferenceDeviceCommand, ReferenceDeviceRuntime,
};
use super::*;
use ferrum_runtime::DeviceLogicalWork;

fn encoded(x: &[f32]) -> Vec<u8> {
    x.iter()
        .flat_map(|v| bf16::from_f32(*v).to_bits().to_le_bytes())
        .collect()
}
fn bits(region: &ReferenceBufferRegion) -> Vec<u16> {
    region
        .read()
        .chunks_exact(2)
        .map(|b| u16::from_le_bytes([b[0], b[1]]))
        .collect()
}
fn fixture(
    runtime: &ReferenceDeviceRuntime,
    x: &[f32],
    columns: usize,
    cos: &[f32],
    sin: &[f32],
    d: usize,
) -> ReferenceHalfSplitRopeLaunch {
    let x = encoded(x);
    let cos = encoded(cos);
    let sin = encoded(sin);
    ReferenceHalfSplitRopeLaunch {
        input: buffer(runtime, "resource.half-rope.x", &x, 16, ElementType::Bf16)
            .region(0..x.len() as u64)
            .unwrap(),
        cosine: buffer(
            runtime,
            "resource.half-rope.cos",
            &cos,
            16,
            ElementType::Bf16,
        )
        .region(0..cos.len() as u64)
        .unwrap(),
        sine: buffer(
            runtime,
            "resource.half-rope.sin",
            &sin,
            16,
            ElementType::Bf16,
        )
        .region(0..sin.len() as u64)
        .unwrap(),
        output: buffer(
            runtime,
            "resource.half-rope.y",
            &vec![0xa5; x.len()],
            16,
            ElementType::Bf16,
        )
        .region(0..x.len() as u64)
        .unwrap(),
        shape: ReferenceHalfSplitRopeShape {
            rows: x.len() / 2 / columns,
            columns,
            head_dim: d,
        },
    }
}
fn submit(runtime: &ReferenceDeviceRuntime, command: ReferenceDeviceCommand) {
    let mut stream = runtime.create_stream().unwrap();
    runtime
        .submit_entries(
            &mut stream,
            vec![(DeviceCommandPhase::Compute, Some(0), command)],
            false,
        )
        .unwrap();
}

#[test]
fn product_bf16_boundary_is_observable_and_not_an_adjacent_fma_profile() {
    let runtime = ReferenceDeviceRuntime::new(config()).unwrap();
    let a = 129.0_f32 / 128.0;
    let b = 130.0_f32 / 128.0;
    // a*a = 16641/16384; B(a*a)=130/128=b. The required sum is +0.
    // Omitting the product BF16 boundary yields 1/16384, BF16 0x3880.
    assert_eq!(bf16::from_f32(a * a - b).to_bits(), 0x3880);
    let launch = fixture(&runtime, &[a, b], 2, &[a, 0.], &[1., 0.], 2);
    let output = launch.output.clone();
    let command = ReferenceDeviceCommand::half_split_rope_tensor(launch).unwrap();
    assert_eq!(
        command
            .attribution(0, Some(0), DeviceCommandPhase::Compute)
            .unwrap()
            .logical_work(),
        DeviceLogicalWork::Tensor {
            element_count: 2,
            arithmetic_operations: 7
        }
    );
    let before = runtime.snapshot();
    submit(&runtime, command);
    assert_eq!(bits(&output), [0, 0]);
    assert_eq!(runtime.snapshot().allocations, before.allocations);
    assert_eq!(
        runtime.snapshot().half_split_rope_launches,
        before.half_split_rope_launches + 1
    );
    // The other product must round too: B(1-B((127/128)*(129/128)))=0,
    // whereas keeping that product in FP32 yields +2^-14.
    let launch = fixture(
        &runtime,
        &[1., 127.0 / 128.0],
        2,
        &[1., 0.],
        &[129.0 / 128.0, 0.],
        2,
    );
    let output = launch.output.clone();
    submit(
        &runtime,
        ReferenceDeviceCommand::half_split_rope_tensor(launch).unwrap(),
    );
    assert_eq!(bits(&output), [0, 0]);
}

#[test]
fn frequencies_are_per_row_and_shared_across_heads_not_between_halves() {
    let runtime = ReferenceDeviceRuntime::new(config()).unwrap();
    let x = [1., 2., 3., 4., 5., 6., 7., 8.].repeat(2);
    let launch = fixture(
        &runtime,
        &x,
        8,
        &[1., 0., 0., 1., 0., 0., 0., 0.],
        &[0., 1., 1., 0., -1., -1., -1., -1.],
        4,
    );
    let output = launch.output.clone();
    submit(
        &runtime,
        ReferenceDeviceCommand::half_split_rope_tensor(launch).unwrap(),
    );
    assert_eq!(
        output.read(),
        encoded(&[1., -4., 1., 4., 5., -8., 5., 8., 3., 4., -1., -2., 7., 8., -5., -6.])
    );
}

#[test]
fn staged_ieee_arithmetic_preserves_zero_rules_and_propagates_nonfinite() {
    let runtime = ReferenceDeviceRuntime::new(config()).unwrap();
    let launch = fixture(&runtime, &[-0., 0.], 2, &[1., 1.], &[0., 0.], 2);
    let output = launch.output.clone();
    submit(
        &runtime,
        ReferenceDeviceCommand::half_split_rope_tensor(launch).unwrap(),
    );
    assert_eq!(bits(&output), [0x8000, 0]);
    let launch = fixture(&runtime, &[f32::INFINITY, 1.], 2, &[0., 0.], &[1., 1.], 2);
    let output = launch.output.clone();
    submit(
        &runtime,
        ReferenceDeviceCommand::half_split_rope_tensor(launch).unwrap(),
    );
    let b = bits(&output);
    assert!(bf16::from_bits(b[0]).is_nan());
    assert_eq!(b[1], 0x7f80);
}

#[test]
fn exact_extents_scalar_alignment_noalias_and_work_overflow_fail_before_execution() {
    let runtime = ReferenceDeviceRuntime::new(config()).unwrap();
    for mutation in 0..9 {
        let mut launch = fixture(&runtime, &[1., 2., 3., 4.], 4, &[1.; 4], &[0.; 4], 4);
        let output = launch.output.clone();
        match mutation {
            0 => launch.shape.head_dim = 3,
            1 => launch.shape.rows = 2,
            2 => launch.cosine.element_type = ElementType::F32,
            3 => launch.sine.length_bytes -= 2,
            4 => launch.cosine.offset_bytes += 1,
            5 => launch.output = launch.input.clone(),
            6 => launch.output = launch.sine.clone(),
            7 => launch.input.length_bytes -= 2,
            8 => launch.shape.columns = 0,
            _ => unreachable!(),
        }
        assert!(ReferenceDeviceCommand::half_split_rope_tensor(launch).is_err());
        assert_eq!(output.read(), [0xa5; 8]);
    }
    let shape = ReferenceHalfSplitRopeShape {
        rows: 1,
        columns: 2,
        head_dim: 2,
    };
    assert_eq!(shape.byte_extents().unwrap(), (4, 4));
    assert!(ReferenceHalfSplitRopeShape {
        rows: usize::MAX,
        ..shape
    }
    .byte_extents()
    .is_err());
    if let Ok(rows) = usize::try_from(u64::MAX / 7 + 1) {
        assert!(ReferenceHalfSplitRopeShape { rows, ..shape }
            .logical_work()
            .is_err());
    }
}

#[test]
fn shared_backing_nonzero_regions_and_late_foreign_frequency_respect_preflight() {
    let runtime = ReferenceDeviceRuntime::new(config()).unwrap();
    let all = buffer(
        &runtime,
        "resource.half-rope.shared",
        &[0xa5; 32],
        16,
        ElementType::Bf16,
    );
    let input = all.region(2..6).unwrap();
    input.write(&encoded(&[1., 2.]));
    let cosine = all.region(8..12).unwrap();
    cosine.write(&encoded(&[0., 0.]));
    let sine = all.region(14..18).unwrap();
    sine.write(&encoded(&[1., 1.]));
    let output = all.region(22..26).unwrap();
    let before = all.region(0..32).unwrap().read();
    let launch = ReferenceHalfSplitRopeLaunch {
        input,
        cosine,
        sine,
        output: output.clone(),
        shape: ReferenceHalfSplitRopeShape {
            rows: 1,
            columns: 2,
            head_dim: 2,
        },
    };
    submit(
        &runtime,
        ReferenceDeviceCommand::half_split_rope_tensor(launch).unwrap(),
    );
    assert_eq!(bits(&output), [0xc000, 0x3f80]);
    let after = all.region(0..32).unwrap().read();
    for i in 0..32 {
        if !(22..26).contains(&i) {
            assert_eq!(after[i], before[i]);
        }
    }
    let foreign = ReferenceDeviceRuntime::new(config()).unwrap();
    let first = fixture(&runtime, &[1., 2.], 2, &[0.; 2], &[1.; 2], 2);
    let first_output = first.output.clone();
    let mut second = fixture(&runtime, &[1., 2.], 2, &[0.; 2], &[1.; 2], 2);
    second.cosine = fixture(&foreign, &[1., 2.], 2, &[0.; 2], &[1.; 2], 2).cosine;
    let mut stream = runtime.create_stream().unwrap();
    let before = runtime.snapshot();
    assert!(runtime
        .submit_entries(
            &mut stream,
            vec![
                (
                    DeviceCommandPhase::Compute,
                    Some(0),
                    ReferenceDeviceCommand::half_split_rope_tensor(first).unwrap()
                ),
                (
                    DeviceCommandPhase::Compute,
                    Some(1),
                    ReferenceDeviceCommand::half_split_rope_tensor(second).unwrap()
                )
            ],
            false
        )
        .is_err());
    assert_eq!(first_output.read(), [0xa5; 4]);
    assert_eq!(runtime.snapshot().submissions, before.submissions);
}
