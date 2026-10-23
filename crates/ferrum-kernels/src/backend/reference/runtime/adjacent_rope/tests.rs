use super::super::tests::{buffer, config};
use super::super::{
    DeviceCommandPhase, DeviceRuntime, ReferenceCommandKind, ReferenceDeviceCommand,
    ReferenceDeviceRuntime, StreamState,
};
use super::*;
use ferrum_runtime::DeviceLogicalWork;
fn encoded(x: &[f32]) -> Vec<u8> {
    x.iter()
        .flat_map(|x| bf16::from_f32(*x).to_bits().to_le_bytes())
        .collect()
}
fn frequencies(x: &[f32]) -> Vec<u8> {
    x.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn bits(x: &ReferenceBufferRegion) -> Vec<u16> {
    x.read()
        .chunks_exact(2)
        .map(|b| u16::from_le_bytes([b[0], b[1]]))
        .collect()
}
fn fixture(
    runtime: &ReferenceDeviceRuntime,
    x: &[f32],
    columns: usize,
    f: &[f32],
    d: usize,
) -> ReferenceAdjacentRopeLaunch {
    let xbytes = encoded(x);
    let fbytes = frequencies(f);
    let input = buffer(runtime, "resource.rope.x", &xbytes, 16, ElementType::Bf16);
    let freq = buffer(runtime, "resource.rope.freq", &fbytes, 16, ElementType::F32);
    let output = buffer(
        runtime,
        "resource.rope.y",
        &vec![0xa5; xbytes.len()],
        16,
        ElementType::Bf16,
    );
    ReferenceAdjacentRopeLaunch {
        input: input.region(0..xbytes.len() as u64).unwrap(),
        frequencies: freq.region(0..fbytes.len() as u64).unwrap(),
        output: output.region(0..xbytes.len() as u64).unwrap(),
        shape: ReferenceAdjacentRopeShape {
            rows: x.len() / columns,
            columns,
            sequence: f.len() / d,
            head_dim: d,
        },
    }
}
fn submit(runtime: &ReferenceDeviceRuntime, command: ReferenceDeviceCommand) {
    let mut s = runtime.create_stream().unwrap();
    runtime
        .submit_entries(
            &mut s,
            vec![(DeviceCommandPhase::Compute, Some(0), command)],
            false,
        )
        .unwrap();
}

#[test]
fn explicit_fma_counterexample_and_logical_work_do_not_match_split_multiply_add() {
    let runtime = ReferenceDeviceRuntime::new(config()).unwrap();
    let a = 129.0_f32 / 128.0;
    let b = 1.0_f32;
    let c = f32::from_bits(0x3f7fffff);
    let s = a;
    // a*c-a = -129*2^-31 is exact (BF16 b381). Rounding a*c
    // first instead gives a-2^-23, hence the incorrect BF16 b400.
    let separately_rounded = a * c;
    assert_eq!(bf16::from_f32(separately_rounded - b * s).to_bits(), 0xb400);
    let launch = fixture(&runtime, &[a, b], 2, &[c, s], 2);
    let output = launch.output.clone();
    let command = ReferenceDeviceCommand::adjacent_rope_tensor(launch).unwrap();
    let work = command
        .attribution(0, Some(0), DeviceCommandPhase::Compute)
        .unwrap();
    assert_eq!(work.token_count(), None);
    assert_eq!(
        work.logical_work(),
        DeviceLogicalWork::Tensor {
            element_count: 2,
            arithmetic_operations: 7
        }
    );
    let before = runtime.snapshot();
    submit(&runtime, command);
    assert_eq!(bits(&output), [0xb381, 0x4001]);
    let after = runtime.snapshot();
    assert_eq!(
        after.adjacent_rope_launches,
        before.adjacent_rope_launches + 1
    );
    assert_eq!(after.allocations, before.allocations);
}

#[test]
fn quadrant_frequencies_repeat_across_batch_and_heads_with_adjacent_pairs() {
    let runtime = ReferenceDeviceRuntime::new(config()).unwrap();
    let x = [1., 2., 3., 4., 5., 6., 7., 8.].repeat(4);
    let launch = fixture(&runtime, &x, 8, &[1., 0., 0., 1., -1., 0., 0., -1.], 4);
    let output = launch.output.clone();
    submit(
        &runtime,
        ReferenceDeviceCommand::adjacent_rope_tensor(launch).unwrap(),
    );
    let expected = [
        1., 2., -4., 3., 5., 6., -8., 7., -1., -2., 4., -3., -5., -6., 8., -7.,
    ]
    .repeat(2);
    assert_eq!(output.read(), encoded(&expected));
}

#[test]
fn explicit_fma_signed_zero_and_nonfinite_propagation_are_not_identity_copy() {
    let runtime = ReferenceDeviceRuntime::new(config()).unwrap();
    let launch = fixture(&runtime, &[0., -0.], 2, &[1., 0.], 2);
    let output = launch.output.clone();
    submit(
        &runtime,
        ReferenceDeviceCommand::adjacent_rope_tensor(launch).unwrap(),
    );
    // Imaginary component is FMA(-0,1,+0), which is +0 under RNE.
    assert_eq!(bits(&output), [0, 0]);
    let launch = fixture(&runtime, &[f32::INFINITY, 1.], 2, &[0., 1.], 2);
    let output = launch.output.clone();
    submit(
        &runtime,
        ReferenceDeviceCommand::adjacent_rope_tensor(launch).unwrap(),
    );
    let result = bits(&output);
    assert!(bf16::from_bits(result[0]).is_nan());
    assert_eq!(result[1], 0x7f80);
}

#[test]
fn shared_backing_mixed_scalar_widths_and_nonzero_offsets_preserve_other_bytes() {
    let runtime = ReferenceDeviceRuntime::new(config()).unwrap();
    let all = buffer(
        &runtime,
        "resource.rope.shared",
        &[0xa5; 32],
        16,
        ElementType::Bf16,
    );
    let input = all.region(2..6).unwrap();
    input.write(&encoded(&[1., 2.]));
    let mut freq = all.region(8..16).unwrap();
    freq.element_type = ElementType::F32;
    freq.write(&frequencies(&[0., 1.]));
    let output = all.region(22..26).unwrap();
    let before = all.region(0..32).unwrap().read();
    let launch = ReferenceAdjacentRopeLaunch {
        input,
        frequencies: freq,
        output: output.clone(),
        shape: ReferenceAdjacentRopeShape {
            rows: 1,
            columns: 2,
            sequence: 1,
            head_dim: 2,
        },
    };
    submit(
        &runtime,
        ReferenceDeviceCommand::adjacent_rope_tensor(launch).unwrap(),
    );
    assert_eq!(bits(&output), [0xc000, 0x3f80]);
    let after = all.region(0..32).unwrap().read();
    for i in 0..32 {
        if !(22..26).contains(&i) {
            assert_eq!(after[i], before[i]);
        }
    }
}

#[test]
fn exact_ranges_scalar_alignment_noalias_and_geometry_fail_before_execution() {
    let runtime = ReferenceDeviceRuntime::new(config()).unwrap();
    for mutation in 0..10 {
        let mut launch = fixture(&runtime, &[1., 2., 3., 4.], 4, &[1., 0., 0., 1.], 4);
        let output = launch.output.clone();
        match mutation {
            0 => launch.shape.head_dim = 3,
            1 => launch.shape.sequence = 2,
            2 => launch.shape.rows = 0,
            3 => launch.frequencies.element_type = ElementType::Bf16,
            4 => launch.frequencies.length_bytes -= 4,
            5 => launch.frequencies.offset_bytes += 2,
            6 => launch.output.length_bytes -= 2,
            7 => launch.output = launch.input.clone(),
            8 => {
                launch.output = launch.frequencies.clone();
                launch.output.element_type = ElementType::Bf16;
                launch.output.length_bytes = 8;
            }
            9 => launch.input.offset_bytes += 1,
            _ => unreachable!(),
        }
        assert!(ReferenceDeviceCommand::adjacent_rope_tensor(launch).is_err());
        assert_eq!(output.read(), [0xa5; 8]);
    }
    let shape = ReferenceAdjacentRopeShape {
        rows: 1,
        columns: 2,
        sequence: 1,
        head_dim: 2,
    };
    assert_eq!(shape.byte_extents().unwrap(), (4, 8));
    assert!(ReferenceAdjacentRopeShape {
        rows: usize::MAX,
        ..shape
    }
    .byte_extents()
    .is_err());
    if let Ok(rows) = usize::try_from(u64::MAX / 7 + 1) {
        assert!(ReferenceAdjacentRopeShape { rows, ..shape }
            .logical_work()
            .is_err());
    }
}

#[test]
fn late_foreign_or_short_frequency_preflight_prevents_all_prior_writes() {
    let runtime = ReferenceDeviceRuntime::new(config()).unwrap();
    let foreign = ReferenceDeviceRuntime::new(config()).unwrap();
    for foreign_frequency in [false, true] {
        let first = fixture(&runtime, &[1., 2.], 2, &[0., 1.], 2);
        let first_output = first.output.clone();
        let second = fixture(&runtime, &[3., 4.], 2, &[0., 1.], 2);
        let second_output = second.output.clone();
        let mut second = ReferenceDeviceCommand::adjacent_rope_tensor(second).unwrap();
        let ReferenceCommandKind::AdjacentRope { launch } = &mut second.kind else {
            unreachable!()
        };
        if foreign_frequency {
            launch.frequencies = fixture(&foreign, &[1., 2.], 2, &[0., 1.], 2).frequencies;
        } else {
            launch.frequencies.length_bytes -= 4;
        }
        let sentinel = buffer(
            &runtime,
            "resource.rope.sentinel",
            &[0xa5; 4],
            16,
            ElementType::U8,
        );
        let commands = vec![
            runtime.encode_zero(&sentinel, 0, 4).unwrap(),
            ReferenceDeviceCommand::adjacent_rope_tensor(first).unwrap(),
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
        let after = runtime.snapshot();
        assert_eq!(after.submissions, before.submissions);
        assert_eq!(after.commands, before.commands);
        assert_eq!(after.adjacent_rope_launches, before.adjacent_rope_launches);
        assert_eq!(runtime.stream_state(&stream), StreamState::Ready);
        assert_eq!(sentinel.region(0..4).unwrap().read(), [0xa5; 4]);
        assert_eq!(first_output.read(), [0xa5; 4]);
        assert_eq!(second_output.read(), [0xa5; 4]);
    }
}
