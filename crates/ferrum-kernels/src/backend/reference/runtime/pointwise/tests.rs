//! Actual Reference allocations and commands; private malformed launches test
//! the backend boundary. Public managed admission is covered by model tests.

use super::super::tests::{buffer, config};
use super::super::{
    DefinitelyNotSubmitted, DeviceCommandPhase, DeviceRuntime, ReferenceCommandKind,
    ReferenceDeviceCommand, ReferenceDeviceFence, ReferenceDeviceRuntime, StreamState,
};
use super::*;
use crate::backend::reference::pointwise::ReferencePointwiseKind as Kind;
use ferrum_runtime::DeviceLogicalWork;

const KINDS: [Kind; 2] = [Kind::ModulationScale, Kind::GatedResidual];

#[derive(Clone)]
struct Fixture {
    kind: Kind,
    input: ReferenceBufferRegion,
    parameter: ReferenceBufferRegion,
    update: ReferenceBufferRegion,
    output: ReferenceBufferRegion,
    shape: ReferencePointwiseShape,
}

impl Fixture {
    fn command(&self) -> Result<ReferenceDeviceCommand, ReferenceDeviceRuntimeError> {
        match self.kind {
            Kind::ModulationScale => {
                ReferenceDeviceCommand::modulation_scale_tensor(ReferenceModulationScaleLaunch {
                    input: self.input.clone(),
                    scale: self.parameter.clone(),
                    output: self.output.clone(),
                    shape: self.shape,
                })
            }
            Kind::GatedResidual => {
                ReferenceDeviceCommand::gated_residual_tensor(ReferenceGatedResidualLaunch {
                    input: self.input.clone(),
                    gate: self.parameter.clone(),
                    update: self.update.clone(),
                    output: self.output.clone(),
                    shape: self.shape,
                })
            }
        }
    }
}

fn encoded(values: &[f32]) -> Vec<u8> {
    values
        .iter()
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
    kind: Kind,
    width: usize,
    mode: RowBroadcast,
    x: &[f32],
    parameter: &[f32],
    update: &[f32],
) -> Fixture {
    let region = |name, values: &[f32]| {
        let bytes = encoded(values);
        buffer(runtime, name, &bytes, 16, ElementType::Bf16)
            .region(0..bytes.len() as u64)
            .unwrap()
    };
    let input = region("resource.pointwise.input", x);
    // The unused update owner for modulation is an alias, not a dummy resource.
    let update = if kind == Kind::GatedResidual {
        region("resource.pointwise.update", update)
    } else {
        input.clone()
    };
    Fixture {
        kind,
        input,
        parameter: region("resource.pointwise.parameter", parameter),
        update,
        output: buffer(
            runtime,
            "resource.pointwise.output",
            &vec![0xa5; x.len() * 2],
            16,
            ElementType::Bf16,
        )
        .region(0..(x.len() * 2) as u64)
        .unwrap(),
        shape: ReferencePointwiseShape {
            rows: x.len() / width,
            width,
            broadcast: mode,
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
fn intermediate_bf16_rounding_is_observable_and_work_is_tensor_only() {
    let runtime = ReferenceDeviceRuntime::new(config()).unwrap();
    for mode in RowBroadcast::ALL {
        let modulation = fixture(
            &runtime,
            Kind::ModulationScale,
            1,
            mode,
            &[1.5],
            &[1. / 256.],
            &[],
        );
        let gated = fixture(
            &runtime,
            Kind::GatedResidual,
            1,
            mode,
            &[-1.],
            &[127. / 128.],
            &[129. / 128.],
        );
        for (case, expected) in [(modulation, 0x3fc0), (gated, 0x0000)] {
            let command = case.command().unwrap();
            let attribution = command
                .attribution(0, Some(0), DeviceCommandPhase::Compute)
                .unwrap();
            assert_eq!(attribution.token_count(), None);
            assert_eq!(
                attribution.logical_work(),
                DeviceLogicalWork::Tensor {
                    element_count: 1,
                    arithmetic_operations: 2,
                }
            );
            submit(&runtime, vec![command]).unwrap();
            assert_eq!(bits(&case.output), [expected]);
        }
    }
    // Omitting either inner round gives a different, independently derived bit pattern.
    assert_eq!(bf16::from_f32(1.5 * (1. + 1. / 256.)).to_bits(), 0x3fc1);
    assert_eq!(
        bf16::from_f32(-1. + (127. / 128.) * (129. / 128.)).to_bits(),
        0xb880
    );
    assert_eq!(runtime.snapshot().modulation_scale_launches, 2);
    assert_eq!(runtime.snapshot().gated_residual_launches, 2);
    assert_eq!(runtime.snapshot().dense_linear_launches, 0);
    assert_eq!(runtime.snapshot().layer_norm_launches, 0);
}

#[test]
fn explicit_row_and_elementwise_modes_apply_different_multiline_parameters() {
    let runtime = ReferenceDeviceRuntime::new(config()).unwrap();
    for (kind, mode, parameter, expected) in [
        (
            Kind::ModulationScale,
            RowBroadcast::Row,
            vec![0., 1.],
            vec![1., 4., 3., 8.],
        ),
        (
            Kind::ModulationScale,
            RowBroadcast::Elementwise,
            vec![0., 1., 2., -1.],
            vec![1., 4., 9., 0.],
        ),
        (
            Kind::GatedResidual,
            RowBroadcast::Row,
            vec![0., 1.],
            vec![1., 6., 3., 12.],
        ),
        (
            Kind::GatedResidual,
            RowBroadcast::Elementwise,
            vec![0., 1., 2., -1.],
            vec![1., 6., 15., -4.],
        ),
    ] {
        let case = fixture(
            &runtime,
            kind,
            2,
            mode,
            &[1., 2., 3., 4.],
            &parameter,
            &[2., 4., 6., 8.],
        );
        submit(&runtime, vec![case.command().unwrap()]).unwrap();
        assert_eq!(case.output.read(), encoded(&expected));
    }
}

#[test]
fn signed_zero_and_gradual_bf16_subnormals_preserve_declared_rounding() {
    let runtime = ReferenceDeviceRuntime::new(config()).unwrap();
    let tiny = bf16::from_bits(1).to_f32();
    let cases = [
        (Kind::ModulationScale, -0., 0., 0., 0x8000),
        (Kind::ModulationScale, -1., -1., 0., 0x8000),
        (Kind::ModulationScale, 1., -1., 0., 0x0000),
        (Kind::ModulationScale, tiny, 0., 0., 0x0001),
        (Kind::ModulationScale, -tiny, 0., 0., 0x8001),
        (Kind::ModulationScale, tiny, -0.5, 0., 0x0000),
        (Kind::ModulationScale, -tiny, -0.5, 0., 0x8000),
        (Kind::GatedResidual, -0., -0., 1., 0x8000),
        (Kind::GatedResidual, 0., -0., 1., 0x0000),
        (Kind::GatedResidual, 0., tiny, 1., 0x0001),
        (Kind::GatedResidual, -0., -tiny, 1., 0x8001),
        (Kind::GatedResidual, 0., tiny, 0.5, 0x0000),
    ];
    for (kind, x, parameter, update, expected) in cases {
        let case = fixture(
            &runtime,
            kind,
            1,
            RowBroadcast::Row,
            &[x],
            &[parameter],
            &[update],
        );
        submit(&runtime, vec![case.command().unwrap()]).unwrap();
        assert_eq!(bits(&case.output), [expected]);
    }
}

#[test]
fn bf16_range_and_inner_overflow_differ_from_f16_and_fused_arithmetic() {
    let runtime = ReferenceDeviceRuntime::new(config()).unwrap();
    let finite = fixture(
        &runtime,
        Kind::GatedResidual,
        1,
        RowBroadcast::Row,
        &[0.],
        &[256.],
        &[256.],
    );
    submit(&runtime, vec![finite.command().unwrap()]).unwrap();
    assert_eq!(bits(&finite.output), [0x4780]); // BF16 65536 is finite.
    let gate = bf16::from_bits(0x3ffe).to_f32();
    let update = bf16::from_bits(0x7f01).to_f32();
    assert!((gate * update).is_finite());
    let overflow = fixture(
        &runtime,
        Kind::GatedResidual,
        1,
        RowBroadcast::Elementwise,
        &[-bf16::MAX.to_f32()],
        &[gate],
        &[update],
    );
    submit(&runtime, vec![overflow.command().unwrap()]).unwrap();
    assert_eq!(bits(&overflow.output), [0x7f80]);
    assert!((-bf16::MAX.to_f32() + gate * update).is_finite());
}

#[test]
fn ieee_nonfinite_values_are_submitted_not_reported_as_structural_failures() {
    let runtime = ReferenceDeviceRuntime::new(config()).unwrap();
    for (kind, x, parameter, update) in [
        (Kind::ModulationScale, 1., f32::NAN, 0.),
        (Kind::ModulationScale, 0., f32::INFINITY, 0.),
        (Kind::GatedResidual, 1., f32::NAN, 2.),
        (Kind::GatedResidual, 1., f32::INFINITY, 0.),
    ] {
        let case = fixture(
            &runtime,
            kind,
            1,
            RowBroadcast::Elementwise,
            &[x],
            &[parameter],
            &[update],
        );
        submit(&runtime, vec![case.command().unwrap()]).unwrap();
        assert!(bf16::from_bits(bits(&case.output)[0]).is_nan());
    }
    assert_eq!(runtime.snapshot().submissions, 4);
}

#[test]
fn shape_byte_extent_and_all_operand_dtypes_are_checked_without_large_allocations() {
    for mode in RowBroadcast::ALL {
        for (rows, width) in [(0, 1), (1, 0), (usize::MAX, 1), (2, usize::MAX / 2)] {
            assert!(ReferencePointwiseShape {
                rows,
                width,
                broadcast: mode
            }
            .logical_work()
            .is_err());
        }
        // Pointwise has no LayerNorm reduction-width limit.
        assert!(ReferencePointwiseShape {
            rows: 1,
            width: (1 << 24) + 1,
            broadcast: mode
        }
        .logical_work()
        .is_ok());
    }
    let runtime = ReferenceDeviceRuntime::new(config()).unwrap();
    for kind in KINDS {
        let case = fixture(
            &runtime,
            kind,
            2,
            RowBroadcast::Elementwise,
            &[1., 2., 3., 4.],
            &[0.; 4],
            &[1.; 4],
        );
        for role in 0..if kind == Kind::GatedResidual { 4 } else { 3 } {
            let mut wrong = case.clone();
            match role {
                0 => wrong.input.element_type = ElementType::F16,
                1 => wrong.parameter.element_type = ElementType::F16,
                2 => wrong.output.element_type = ElementType::F16,
                3 => wrong.update.element_type = ElementType::F16,
                _ => unreachable!(),
            }
            assert!(wrong.command().is_err());
        }
        let mut wrong = case.clone();
        wrong.shape.broadcast = RowBroadcast::Row;
        assert!(wrong.command().is_err());
        let mut wrong = case.clone();
        wrong.input.offset_bytes += 1;
        assert!(wrong.command().is_err());
        let mut wrong = case.clone();
        wrong.output.length_bytes -= 2;
        assert!(wrong.command().is_err());
        assert_eq!(case.output.read(), [0xa5; 8]);
    }
}

#[test]
fn late_invalid_command_preserves_all_earlier_outputs_counters_and_ready_stream() {
    let runtime = ReferenceDeviceRuntime::new(config()).unwrap();
    let foreign = ReferenceDeviceRuntime::new(config()).unwrap();
    for kind in KINDS {
        for fault in 0..3 {
            let first = fixture(
                &runtime,
                Kind::ModulationScale,
                2,
                RowBroadcast::Row,
                &[1.; 4],
                &[0.; 2],
                &[],
            );
            let second = fixture(
                &runtime,
                Kind::GatedResidual,
                2,
                RowBroadcast::Row,
                &[1.; 4],
                &[1.; 2],
                &[1.; 4],
            );
            let later = fixture(
                &runtime,
                kind,
                2,
                RowBroadcast::Row,
                &[1.; 4],
                &[0.; 2],
                &[1.; 4],
            );
            let mut bad = later.command().unwrap();
            let (shape, input, output) = match &mut bad.kind {
                ReferenceCommandKind::ModulationScale { launch } => {
                    (&mut launch.shape, &mut launch.input, &mut launch.output)
                }
                ReferenceCommandKind::GatedResidual { launch } => {
                    (&mut launch.shape, &mut launch.input, &mut launch.output)
                }
                _ => unreachable!(),
            };
            match fault {
                0 => shape.rows = 0,
                1 => {
                    *input = fixture(
                        &foreign,
                        kind,
                        2,
                        RowBroadcast::Row,
                        &[1.; 4],
                        &[0.; 2],
                        &[1.; 4],
                    )
                    .input
                }
                2 => *output = input.clone(),
                _ => unreachable!(),
            }
            let sentinel = buffer(
                &runtime,
                "resource.pointwise.preflight",
                &[0xa5; 4],
                16,
                ElementType::U8,
            );
            let zero = runtime.encode_zero(&sentinel, 0, 4).unwrap();
            let entries = vec![
                zero,
                first.command().unwrap(),
                second.command().unwrap(),
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
            assert_eq!(
                after.modulation_scale_launches,
                before.modulation_scale_launches
            );
            assert_eq!(
                after.gated_residual_launches,
                before.gated_residual_launches
            );
            assert_eq!(sentinel.region(0..4).unwrap().read(), [0xa5; 4]);
            for output in [&first.output, &second.output, &later.output] {
                assert_eq!(output.read(), [0xa5; 8]);
            }
        }
    }
}

mod regions;
