use super::super::tests::{buffer, config};
use super::super::{
    DeviceCommandPhase, DeviceRuntime, ReferenceDeviceCommand, ReferenceDeviceRuntime,
};
use super::*;
use ferrum_runtime::DeviceLogicalWork;
fn fixture(
    runtime: &ReferenceDeviceRuntime,
    x: &[u16],
    u: &[u16],
) -> ReferencePlainResidualAddLaunch {
    let bytes = |bits: &[u16]| {
        bits.iter()
            .flat_map(|b| b.to_le_bytes())
            .collect::<Vec<_>>()
    };
    let n = x.len() * 2;
    ReferencePlainResidualAddLaunch {
        input: buffer(runtime, "resource.add.x", &bytes(x), 16, ElementType::Bf16)
            .region(0..n as u64)
            .unwrap(),
        update: buffer(runtime, "resource.add.u", &bytes(u), 16, ElementType::Bf16)
            .region(0..(u.len() * 2) as u64)
            .unwrap(),
        output: buffer(
            runtime,
            "resource.add.y",
            &vec![0xa5; n],
            16,
            ElementType::Bf16,
        )
        .region(0..n as u64)
        .unwrap(),
        shape: ReferenceActivationShape {
            rows: 1,
            width: x.len(),
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
fn plain_add_rne_signed_zero_overflow_and_nan_use_no_gate_or_extra_allocations() {
    let runtime = ReferenceDeviceRuntime::new(config()).unwrap();
    let launch = fixture(
        &runtime,
        &[0x3f80, 0x3f81, 0x8000, 0x8000, 0x7f7f, 0x7f80],
        &[0x3b80, 0x3b80, 0x8000, 0, 0x7f7f, 0xff80],
    );
    let output = launch.output.clone();
    let command = ReferenceDeviceCommand::plain_residual_add_tensor(launch).unwrap();
    assert_eq!(
        command
            .attribution(0, Some(0), DeviceCommandPhase::Compute)
            .unwrap()
            .logical_work(),
        DeviceLogicalWork::Tensor {
            element_count: 6,
            arithmetic_operations: 6
        }
    );
    let before = runtime.snapshot();
    submit(&runtime, command);
    let bits: Vec<_> = output
        .read()
        .chunks_exact(2)
        .map(|b| u16::from_le_bytes([b[0], b[1]]))
        .collect();
    assert_eq!(&bits[..5], &[0x3f80, 0x3f82, 0x8000, 0, 0x7f80]);
    assert!(bf16::from_bits(bits[5]).is_nan());
    assert_eq!(runtime.snapshot().allocations, before.allocations);
    assert_eq!(
        runtime.snapshot().plain_residual_add_launches,
        before.plain_residual_add_launches + 1
    );
}
#[test]
fn repeated_read_storage_is_legal_but_output_alias_short_storage_and_dtype_are_rejected() {
    let runtime = ReferenceDeviceRuntime::new(config()).unwrap();
    let mut launch = fixture(&runtime, &[0x3f80, 0x4000], &[0x4040, 0x4080]);
    launch.update = launch.input.clone();
    let output = launch.output.clone();
    submit(
        &runtime,
        ReferenceDeviceCommand::plain_residual_add_tensor(launch).unwrap(),
    );
    assert_eq!(output.read(), [0, 0x40, 0x80, 0x40]);
    for variant in 0..5 {
        let mut launch = fixture(&runtime, &[0x3f80, 0x4000], &[0x4040, 0x4080]);
        match variant {
            0 => launch.output = launch.input.clone(),
            1 => launch.update.length_bytes -= 2,
            2 => launch.update.element_type = ElementType::F16,
            3 => launch.input.offset_bytes += 1,
            4 => launch.shape.rows = 0,
            _ => unreachable!(),
        }
        assert!(ReferenceDeviceCommand::plain_residual_add_tensor(launch).is_err());
    }
}
#[test]
fn foreign_late_operand_is_rejected_before_any_earlier_command_writes() {
    let runtime = ReferenceDeviceRuntime::new(config()).unwrap();
    let foreign = ReferenceDeviceRuntime::new(config()).unwrap();
    let first = fixture(&runtime, &[0x3f80], &[0x3f80]);
    let output = first.output.clone();
    let mut second = fixture(&runtime, &[0x3f80], &[0x3f80]);
    second.update = fixture(&foreign, &[0x3f80], &[0x3f80]).update;
    let mut stream = runtime.create_stream().unwrap();
    let before = runtime.snapshot();
    assert!(runtime
        .submit_entries(
            &mut stream,
            vec![
                (
                    DeviceCommandPhase::Compute,
                    Some(0),
                    ReferenceDeviceCommand::plain_residual_add_tensor(first).unwrap()
                ),
                (
                    DeviceCommandPhase::Compute,
                    Some(1),
                    ReferenceDeviceCommand::plain_residual_add_tensor(second).unwrap()
                )
            ],
            false
        )
        .is_err());
    assert_eq!(output.read(), [0xa5; 2]);
    assert_eq!(runtime.snapshot().submissions, before.submissions);
}
