//! Actual Reference storage/dispatch tests. Exhaustive scalar qualification is
//! independent and lives in reference::transcendental::oracle.

use super::super::tests::{buffer, config};
use super::super::{
    DefinitelyNotSubmitted, DeviceCommandPhase, DeviceRuntime, ReferenceCommandKind,
    ReferenceDeviceCommand, ReferenceDeviceFence, ReferenceDeviceRuntime, StreamState,
};
use super::*;
use ferrum_runtime::{
    DeviceLogicalWork, HostMemoryAccount, HostMemoryClass, HostMemoryClassLimits, HostMemoryLimits,
    HostTransferLayout,
};

#[derive(Clone, Copy, Debug)]
enum Kind {
    Silu,
    SiluMul,
    Tanh,
}
const KINDS: [Kind; 3] = [Kind::Silu, Kind::SiluMul, Kind::Tanh];

#[derive(Clone)]
struct Fixture {
    kind: Kind,
    input: ReferenceBufferRegion,
    up: ReferenceBufferRegion,
    output: ReferenceBufferRegion,
    shape: ReferenceActivationShape,
}

impl Fixture {
    fn command(&self) -> Result<ReferenceDeviceCommand, ReferenceDeviceRuntimeError> {
        match self.kind {
            Kind::Silu | Kind::Tanh => {
                ReferenceDeviceCommand::unary_activation_tensor(ReferenceUnaryLaunch {
                    kind: if matches!(self.kind, Kind::Silu) {
                        ReferenceUnaryKind::Silu
                    } else {
                        ReferenceUnaryKind::Tanh
                    },
                    input: self.input.clone(),
                    output: self.output.clone(),
                    shape: self.shape,
                })
            }
            Kind::SiluMul => ReferenceDeviceCommand::silu_mul_tensor(ReferenceSiluMulLaunch {
                gate: self.input.clone(),
                up: self.up.clone(),
                output: self.output.clone(),
                shape: self.shape,
            }),
        }
    }
}

fn bytes(bits: &[u16]) -> Vec<u8> {
    bits.iter().flat_map(|b| b.to_le_bytes()).collect()
}

fn check(region: &ReferenceBufferRegion, expected: &[u16]) {
    let output = region.read();
    assert_eq!(output.len(), expected.len() * 2);
    for (chunk, expected) in output.chunks_exact(2).zip(expected) {
        let actual = u16::from_le_bytes([chunk[0], chunk[1]]);
        if expected & 0x7fff > 0x7f80 {
            assert!(actual & 0x7fff > 0x7f80);
        } else {
            assert_eq!(actual, *expected);
        }
    }
}

fn fixture(
    runtime: &ReferenceDeviceRuntime,
    kind: Kind,
    width: usize,
    input: &[u16],
    up: &[u16],
) -> Fixture {
    let region = |name, values| {
        let data = bytes(values);
        buffer(runtime, name, &data, 16, ElementType::Bf16)
            .region(0..data.len() as u64)
            .unwrap()
    };
    let source = region("resource.activation.input", input);
    Fixture {
        kind,
        up: if matches!(kind, Kind::SiluMul) {
            region("resource.activation.up", up)
        } else {
            source.clone()
        },
        input: source,
        output: buffer(
            runtime,
            "resource.activation.output",
            &vec![0xa5; input.len() * 2],
            16,
            ElementType::Bf16,
        )
        .region(0..(input.len() * 2) as u64)
        .unwrap(),
        shape: ReferenceActivationShape {
            rows: input.len() / width,
            width,
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
fn real_commands_preserve_analytic_boundaries_and_tensor_logical_work() {
    let runtime = ReferenceDeviceRuntime::new(config()).unwrap();
    for (kind, input, up, expected, per_element) in [
        (
            Kind::Silu,
            vec![0x3f80, 0xc2b4, 1, 0x8001],
            vec![],
            vec![0x3f3b, 0x8000, 0, 0x8000],
            4,
        ),
        (
            Kind::SiluMul,
            vec![0x3f80, 0xc2b4, 0, 0],
            vec![0x3f81, 0x7f80, 0x7f80, 0xbf80],
            vec![0x3f3c, 0x7fc0, 0x7fc0, 0x8000],
            5,
        ),
        (
            Kind::Tanh,
            vec![0x3f80, 0xbf80, 1, 0x8001],
            vec![],
            vec![0x3f43, 0xbf43, 1, 0x8001],
            1,
        ),
    ] {
        let case = fixture(&runtime, kind, 2, &input, &up);
        let command = case.command().unwrap();
        let attribution = command
            .attribution(0, Some(0), DeviceCommandPhase::Compute)
            .unwrap();
        assert_eq!(attribution.token_count(), None);
        assert_eq!(
            attribution.logical_work(),
            DeviceLogicalWork::Tensor {
                element_count: 4,
                arithmetic_operations: 4 * per_element
            }
        );
        submit(&runtime, vec![command]).unwrap();
        check(&case.output, &expected);
    }
    let snapshot = runtime.snapshot();
    assert_eq!(
        (
            snapshot.silu_launches,
            snapshot.silu_mul_launches,
            snapshot.tanh_launches
        ),
        (1, 1, 1)
    );
    assert_eq!(
        (snapshot.dense_linear_launches, snapshot.layer_norm_launches),
        (0, 0)
    );
}

#[test]
fn numerical_specials_execute_with_ieee_classification_and_signed_zero() {
    let runtime = ReferenceDeviceRuntime::new(config()).unwrap();
    for (kind, input, up, expected) in [
        (
            Kind::Silu,
            vec![0, 0x8000, 0x7f80, 0xff80, 0x7fc1, 0xff81],
            vec![],
            vec![0, 0x8000, 0x7f80, 0x7fc0, 0x7fc0, 0x7fc0],
        ),
        (
            Kind::Tanh,
            vec![0, 0x8000, 0x7f80, 0xff80, 0x7fc1, 0xff81],
            vec![],
            vec![0, 0x8000, 0x3f80, 0xbf80, 0x7fc0, 0x7fc0],
        ),
        (
            Kind::SiluMul,
            vec![0x7f7f, 0xc2b4, 0xff80, 0x3f80],
            vec![0x7f7f, 0xbf80, 0, 0x7fc1],
            vec![0x7f80, 0, 0x7fc0, 0x7fc0],
        ),
    ] {
        let case = fixture(&runtime, kind, input.len(), &input, &up);
        submit(&runtime, vec![case.command().unwrap()]).unwrap();
        check(&case.output, &expected);
    }
    assert_eq!(runtime.snapshot().submissions, 3);
}

#[test]
fn structural_extent_dtype_and_logical_count_overflow_are_rejected() {
    for (rows, width) in [(0, 1), (1, 0), (usize::MAX, 1), (2, usize::MAX / 2)] {
        assert!(ReferenceActivationShape { rows, width }
            .byte_extent()
            .is_err());
    }
    assert!(ReferenceActivationShape {
        rows: 1,
        width: (1 << 24) + 1
    }
    .logical_work(5)
    .is_ok());
    let extreme = ReferenceActivationShape {
        rows: 1,
        width: usize::MAX / 2,
    };
    // On 64-bit hosts the byte extent fits while five operations per scalar do not.
    if let Ok(width) = usize::try_from(u64::MAX / 5 + 1) {
        let counted = ReferenceActivationShape { rows: 1, width };
        assert!(counted.logical_work(5).is_err());
    }
    assert!(extreme.byte_extent().is_ok());
    let runtime = ReferenceDeviceRuntime::new(config()).unwrap();
    for kind in KINDS {
        let case = fixture(&runtime, kind, 2, &[0x3f80; 4], &[0x3f80; 4]);
        for role in 0..if matches!(kind, Kind::SiluMul) { 3 } else { 2 } {
            let mut wrong = case.clone();
            match role {
                0 => wrong.input.element_type = ElementType::F16,
                1 => wrong.output.element_type = ElementType::F16,
                2 => wrong.up.element_type = ElementType::F16,
                _ => unreachable!(),
            }
            assert!(wrong.command().is_err());
        }
        let mut wrong = case.clone();
        wrong.output.length_bytes -= 2;
        assert!(wrong.command().is_err());
        let mut wrong = case.clone();
        wrong.shape.rows = 3;
        assert!(wrong.command().is_err());
        if matches!(kind, Kind::SiluMul) {
            let mut wrong = case.clone();
            wrong.up.length_bytes = 4; // No row broadcast.
            assert!(wrong.command().is_err());
        }
        assert_eq!(case.output.read(), [0xa5; 8]);
    }
}

#[test]
fn late_invalid_command_prevents_all_earlier_writes_and_dispatches() {
    let runtime = ReferenceDeviceRuntime::new(config()).unwrap();
    let foreign = ReferenceDeviceRuntime::new(config()).unwrap();
    for kind in KINDS {
        for fault in 0..3 {
            let earlier = KINDS.map(|k| fixture(&runtime, k, 2, &[0x3f80; 4], &[0x3f81; 4]));
            let later = fixture(&runtime, kind, 2, &[0x3f80; 4], &[0x3f81; 4]);
            let mut bad = later.command().unwrap();
            let (shape, input, output) = match &mut bad.kind {
                ReferenceCommandKind::UnaryActivation { launch } => {
                    (&mut launch.shape, &mut launch.input, &mut launch.output)
                }
                ReferenceCommandKind::SiluMul { launch } => {
                    (&mut launch.shape, &mut launch.gate, &mut launch.output)
                }
                _ => unreachable!(),
            };
            match fault {
                0 => shape.width = 0,
                1 => *input = fixture(&foreign, kind, 2, &[0x3f80; 4], &[0x3f81; 4]).input,
                2 => *output = input.clone(),
                _ => unreachable!(),
            }
            let sentinel = buffer(
                &runtime,
                "resource.activation.preflight",
                &[0xa5; 4],
                16,
                ElementType::U8,
            );
            let host = HostMemoryAccount::new(
                HostMemoryLimits::new(
                    4,
                    HostMemoryClassLimits {
                        input_preprocess_bytes: 4,
                        transfer_staging_bytes: 0,
                        output_bytes: 0,
                        weight_backing_bytes: 0,
                    },
                )
                .unwrap(),
            );
            let upload = host
                .allocate_zeroed(HostMemoryClass::InputPreprocess, 4)
                .unwrap()
                .freeze();
            let mut commands = vec![runtime
                .encode_managed_upload(
                    upload,
                    HostTransferLayout::new(ElementType::U8, 4).unwrap(),
                    &sentinel,
                    0,
                )
                .unwrap()
                .unwrap()];
            commands.extend(earlier.iter().map(|f| f.command().unwrap()));
            commands.push(bad);
            let entries = commands
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
            assert_eq!(
                (after.submissions, after.commands, after.uploaded_bytes),
                (before.submissions, before.commands, before.uploaded_bytes)
            );
            assert_eq!(
                (
                    after.silu_launches,
                    after.silu_mul_launches,
                    after.tanh_launches
                ),
                (
                    before.silu_launches,
                    before.silu_mul_launches,
                    before.tanh_launches
                )
            );
            assert_eq!(sentinel.region(0..4).unwrap().read(), [0xa5; 4]);
            assert_eq!(host.usage().unwrap().committed_bytes(), 0);
            for case in earlier.iter().chain(std::iter::once(&later)) {
                assert_eq!(case.output.read(), [0xa5; 8]);
            }
        }
    }
}

mod regions;
