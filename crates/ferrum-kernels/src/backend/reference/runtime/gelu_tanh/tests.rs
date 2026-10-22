use super::super::tests::{buffer, config};
use super::super::{
    DeviceCommandPhase, DeviceRuntime, ElementType, ReferenceActivationShape, ReferenceCommandKind,
    ReferenceDeviceCommand, ReferenceDeviceRuntime, ReferenceUnaryKind, ReferenceUnaryLaunch,
    StreamState,
};
use super::*;
use ferrum_runtime::DeviceLogicalWork;

// Pure interval arithmetic/Taylor remainder proof, with no candidate tanh call.
#[path = "../../transcendental/oracle/interval.rs"]
mod interval;

fn launch(runtime: &ReferenceDeviceRuntime, input: &[u16]) -> ReferenceUnaryLaunch {
    let source: Vec<_> = input.iter().flat_map(|v| v.to_le_bytes()).collect();
    let x = buffer(runtime, "resource.gelu.x", &source, 16, ElementType::Bf16);
    let y = buffer(
        runtime,
        "resource.gelu.y",
        &vec![0xa5; source.len()],
        16,
        ElementType::Bf16,
    );
    ReferenceUnaryLaunch {
        kind: ReferenceUnaryKind::GeluTanh,
        input: x.region(0..source.len() as u64).unwrap(),
        output: y.region(0..source.len() as u64).unwrap(),
        shape: ReferenceActivationShape {
            rows: 1,
            width: input.len(),
        },
    }
}

#[test]
fn independent_taylor_cell_proves_no_intermediate_bf16_activation_boundary() {
    // For x=1 the declared FP32 polynomial argument has these fixed bits.
    // This expectation is independent of libm/candidate tanh.
    let argument = f32::from_bits(0x3f556452);
    assert_eq!((BETA * (1.0_f32 + KAPPA)).to_bits(), argument.to_bits());
    let enclosure = interval::tanh(f64::from(argument)).unwrap();
    // There is a wide margin even after the FP32 tanh/add/mul rounding.
    // These strict rational inequalities place B(.5*(1+tanh)) in 3f57;
    // B(tanh) itself is 175/256, which would instead give midpoint 3f58.
    assert!(enclosure.lo > 349.0 / 512.0 + 1.0 / 1_048_576.0);
    assert!(enclosure.hi < 175.0 / 256.0 - 1.0 / 1_048_576.0);
    // The negative-input cell also has margin from its lower midpoint.
    assert!(enclosure.hi < 699.0 / 1024.0 - 1.0 / 1_048_576.0);
    assert_eq!(gelu_tanh_bf16_bits(0x3f80), 0x3f57);
    assert_eq!(gelu_tanh_bf16_bits(0xbf80), 0xbe23);
    // This is a selected analytic fixture, not general FP32 tanh qualification.
}

#[test]
fn signed_zero_underflow_and_ieee_nonfinite_paths_are_explicit() {
    for (input, expected) in [
        (0x0000, 0x0000),
        (0x8000, 0x8000),
        (0x0001, 0x0000),
        (0x8001, 0x8000),
        (0x0003, 0x0002),
        (0x8003, 0x8002),
        (0x7f80, 0x7f80),
        (0x7f7f, 0x7f7f),
        (0xff7f, 0x8000),
    ] {
        assert_eq!(gelu_tanh_bf16_bits(input), expected, "input {input:04x}");
    }
    for input in [0xff80, 0x7fc1, 0xffc1] {
        assert!(bf16::from_bits(gelu_tanh_bf16_bits(input)).is_nan());
    }
}

#[test]
fn actual_managed_unary_dispatch_records_gelu_work_and_preserves_regions() {
    let runtime = ReferenceDeviceRuntime::new(config()).unwrap();
    let backing = buffer(
        &runtime,
        "resource.gelu.shared",
        &[0xa5; 32],
        16,
        ElementType::Bf16,
    );
    let input = backing.region(2..10).unwrap();
    input.write(
        &[0x3f80_u16, 0xbf80, 0, 0x8000]
            .into_iter()
            .flat_map(u16::to_le_bytes)
            .collect::<Vec<_>>(),
    );
    let output = backing.region(20..28).unwrap();
    let before_bytes = backing.region(0..32).unwrap().read();
    let command = ReferenceDeviceCommand::unary_activation_tensor(ReferenceUnaryLaunch {
        kind: ReferenceUnaryKind::GeluTanh,
        input,
        output: output.clone(),
        shape: ReferenceActivationShape { rows: 2, width: 2 },
    })
    .unwrap();
    assert_eq!(
        command
            .attribution(0, Some(0), DeviceCommandPhase::Compute)
            .unwrap()
            .logical_work(),
        DeviceLogicalWork::Tensor {
            element_count: 4,
            arithmetic_operations: 36
        }
    );
    let before = runtime.snapshot();
    let mut stream = runtime.create_stream().unwrap();
    runtime
        .submit_entries(
            &mut stream,
            vec![(DeviceCommandPhase::Compute, Some(0), command)],
            false,
        )
        .unwrap();
    let actual: Vec<_> = output
        .read()
        .chunks_exact(2)
        .map(|b| u16::from_le_bytes([b[0], b[1]]))
        .collect();
    assert_eq!(actual, [0x3f57, 0xbe23, 0, 0x8000]);
    assert_eq!(
        runtime.snapshot().gelu_tanh_launches,
        before.gelu_tanh_launches + 1
    );
    assert_eq!(runtime.snapshot().allocations, before.allocations);
    assert_eq!(runtime.snapshot().tanh_launches, before.tanh_launches);
    let after_bytes = backing.region(0..32).unwrap().read();
    for i in 0..32 {
        if !(20..28).contains(&i) {
            assert_eq!(after_bytes[i], before_bytes[i]);
        }
    }
}

#[test]
fn invalid_regions_and_late_foreign_input_fail_before_any_command_runs() {
    let runtime = ReferenceDeviceRuntime::new(config()).unwrap();
    for mutation in 0..5 {
        let mut selected = launch(&runtime, &[0x3f80; 2]);
        let output = selected.output.clone();
        match mutation {
            0 => selected.output = selected.input.clone(),
            1 => selected.input.element_type = ElementType::F16,
            2 => selected.output.length_bytes -= 2,
            3 => selected.input.offset_bytes += 1,
            4 => selected.shape.width = 0,
            _ => unreachable!(),
        }
        assert!(ReferenceDeviceCommand::unary_activation_tensor(selected).is_err());
        assert_eq!(output.read(), [0xa5; 4]);
    }
    let foreign = ReferenceDeviceRuntime::new(config()).unwrap();
    let first = launch(&runtime, &[0x3f80; 2]);
    let output = first.output.clone();
    let mut second =
        ReferenceDeviceCommand::unary_activation_tensor(launch(&runtime, &[0x3f80; 2])).unwrap();
    let ReferenceCommandKind::UnaryActivation { launch: selected } = &mut second.kind else {
        unreachable!()
    };
    selected.input = launch(&foreign, &[0x3f80; 2]).input;
    let mut stream = runtime.create_stream().unwrap();
    let before = runtime.snapshot();
    assert!(runtime
        .submit_entries(
            &mut stream,
            vec![
                (
                    DeviceCommandPhase::Compute,
                    Some(0),
                    ReferenceDeviceCommand::unary_activation_tensor(first).unwrap()
                ),
                (DeviceCommandPhase::Compute, Some(1), second),
            ],
            false
        )
        .is_err());
    assert_eq!(runtime.stream_state(&stream), StreamState::Ready);
    assert_eq!(runtime.snapshot().submissions, before.submissions);
    assert_eq!(
        runtime.snapshot().gelu_tanh_launches,
        before.gelu_tanh_launches
    );
    assert_eq!(output.read(), [0xa5; 4]);
}
