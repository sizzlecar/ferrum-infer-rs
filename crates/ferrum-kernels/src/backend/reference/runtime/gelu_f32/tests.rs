use super::super::tests::{buffer, config};
use super::*;
use ferrum_runtime::{DeviceLogicalWork, ResolvedTensorLayout, ResolvedTensorSpec};

fn geometry(rows: u64, width: u64) -> GeluF32Geometry {
    let t = ResolvedTensorSpec::new(
        vec![rows, width],
        ElementType::F32,
        ResolvedTensorLayout::Contiguous,
    )
    .unwrap();
    GeluF32Geometry::new(&t, &t).unwrap()
}
fn launch(runtime: &ReferenceDeviceRuntime, variant: GeluF32Variant) -> ReferenceGeluF32Launch {
    let data = [1.000244140625_f32, -1.]
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect::<Vec<_>>();
    ReferenceGeluF32Launch {
        input: buffer(runtime, "resource.gelu_f32.x", &data, 16, ElementType::F32)
            .region(0..8)
            .unwrap(),
        output: buffer(
            runtime,
            "resource.gelu_f32.y",
            &[0xa5; 8],
            16,
            ElementType::F32,
        )
        .region(0..8)
        .unwrap(),
        geometry: geometry(1, 2),
        variant,
    }
}

#[test]
fn both_variants_agree_with_independent_f64_formulas_on_unsaturated_cells() {
    for x in [-5.0_f32, -3.5, -2.5, -1.0, -0.25, 0.25, 1., 2.5, 3.5, 5.] {
        let d = f64::from(x);
        for variant in [GeluF32Variant::Tanh, GeluF32Variant::Erf] {
            // Independent F64 ideal-expression oracle: no candidate F32
            // constants, polynomial intermediates or tanhf/erff calls.
            let expected = match variant {
                GeluF32Variant::Tanh => {
                    0.5 * d * (1. + libm::tanh(0.7978845608028654 * (d + 0.044715 * d * d * d)))
                }
                GeluF32Variant::Erf => {
                    0.5 * d * (1. + libm::erf(d * std::f64::consts::FRAC_1_SQRT_2))
                }
            };
            let actual = f64::from(scalar(x, variant));
            assert!(
                (actual - expected).abs() < 1e-6,
                "{variant:?} at {x}: {actual} vs {expected}"
            );
        }
    }
    // Tanh is the approximation; the separately typed erf route must not
    // silently alias it, even though both are called GELU by upstream APIs.
    assert!((scalar(1., GeluF32Variant::Tanh) - scalar(1., GeluF32Variant::Erf)).abs() > 1e-4);
}
#[test]
fn signed_zero_gradual_underflow_cancellation_and_nonfinite_paths_are_retained() {
    for variant in [GeluF32Variant::Tanh, GeluF32Variant::Erf] {
        for (raw, expected) in [
            (0x00000000, 0x00000000),
            (0x80000000, 0x80000000),
            (0x00000001, 0x00000000),
            (0x80000001, 0x80000000),
            (0x00000003, 0x00000002),
            (0x80000003, 0x80000002),
            (0x00800000, 0x00400000),
            (0x80800000, 0x80400000),
            (0x7f7fffff, 0x7f7fffff),
            (0xff7fffff, 0x80000000),
            (0x7f800000, 0x7f800000),
        ] {
            assert_eq!(
                scalar(f32::from_bits(raw), variant).to_bits(),
                expected,
                "{variant:?} input {raw:08x}"
            );
        }
        assert_eq!(scalar(-10., variant).to_bits(), (-0.0_f32).to_bits());
        for raw in [0xff800000, 0x7fc00001, 0xffc12345] {
            assert!(scalar(f32::from_bits(raw), variant).is_nan());
        }
    }
}
#[test]
fn fractional_f32_input_and_output_bits_survive_without_a_half_boundary() {
    let x = 1.000244140625_f32;
    assert_eq!(half::bf16::from_f32(x).to_f32(), 1.);
    for variant in [GeluF32Variant::Tanh, GeluF32Variant::Erf] {
        let y = scalar(x, variant);
        assert_ne!(y.to_bits(), scalar(1., variant).to_bits());
        assert_ne!(y.to_bits(), half::bf16::from_f32(y).to_f32().to_bits());
    }
}
#[test]
fn actual_unary_commands_process_both_rows_and_preserve_guard_bytes_without_scratch() {
    let runtime = ReferenceDeviceRuntime::new(config()).unwrap();
    let values = [
        0.25_f32,
        -0.25,
        1.,
        -1.,
        2.5,
        -2.5,
        1.000244140625,
        -3.5,
        0.,
        -0.,
        f32::from_bits(3),
        -10.,
        3.5,
        5.,
    ];
    for variant in [GeluF32Variant::Tanh, GeluF32Variant::Erf] {
        let backing = buffer(
            &runtime,
            "resource.gelu_f32.guard",
            &[0xa5; 160],
            16,
            ElementType::F32,
        );
        let input = backing.region(16..72).unwrap();
        input.write(
            &values
                .into_iter()
                .flat_map(f32::to_le_bytes)
                .collect::<Vec<_>>(),
        );
        let output = backing.region(88..144).unwrap();
        let guards = backing.region(0..160).unwrap();
        let before = guards.read();
        let command = ReferenceDeviceCommand::gelu_f32_tensor(ReferenceGeluF32Launch {
            input,
            output: output.clone(),
            geometry: geometry(2, 7),
            variant,
        })
        .unwrap();
        assert_eq!(
            command
                .attribution(0, Some(0), DeviceCommandPhase::Compute)
                .unwrap()
                .logical_work(),
            DeviceLogicalWork::Tensor {
                element_count: 14,
                arithmetic_operations: 14 * variant.logical_operations_per_element()
            }
        );
        drop(backing); // Command and readback views retain the original owner.
        let allocations = runtime.snapshot().allocations;
        let mut stream = runtime.create_stream().unwrap();
        runtime
            .submit_entries(
                &mut stream,
                vec![(DeviceCommandPhase::Compute, Some(0), command)],
                false,
            )
            .unwrap();
        for (raw, x) in output.read().chunks_exact(4).zip(values) {
            assert_eq!(
                u32::from_le_bytes(raw.try_into().unwrap()),
                scalar(x, variant).to_bits()
            );
        }
        let after = guards.read();
        for i in 0..160 {
            if !(88..144).contains(&i) {
                assert_eq!(after[i], before[i]);
            }
        }
        assert_eq!(runtime.snapshot().allocations, allocations);
    }
    assert_eq!(runtime.snapshot().live_allocations, 0);
}
#[test]
fn physical_dtype_alignment_alias_and_extent_are_rejected_before_writes() {
    let runtime = ReferenceDeviceRuntime::new(config()).unwrap();
    for mutation in 0..5 {
        let mut selected = launch(&runtime, GeluF32Variant::Tanh);
        let output = selected.output.clone();
        match mutation {
            0 => selected.output = selected.input.clone(),
            1 => selected.input.element_type = ElementType::Bf16,
            2 => selected.output.length_bytes -= 4,
            3 => selected.input.offset_bytes += 1,
            4 => selected.geometry = geometry(1, 3),
            _ => unreachable!(),
        }
        assert!(ReferenceDeviceCommand::gelu_f32_tensor(selected).is_err());
        assert_eq!(output.read(), [0xa5; 8]);
    }
    assert_eq!(runtime.snapshot().submissions, 0);
}
#[test]
fn late_foreign_input_preflights_the_whole_submission_before_any_output_is_written() {
    let runtime = ReferenceDeviceRuntime::new(config()).unwrap();
    let foreign = ReferenceDeviceRuntime::new(config()).unwrap();
    let first = launch(&runtime, GeluF32Variant::Erf);
    let output = first.output.clone();
    let mut second = launch(&runtime, GeluF32Variant::Tanh);
    second.input = launch(&foreign, GeluF32Variant::Tanh).input;
    let mut stream = runtime.create_stream().unwrap();
    assert!(runtime
        .submit_entries(
            &mut stream,
            vec![
                (
                    DeviceCommandPhase::Compute,
                    Some(0),
                    ReferenceDeviceCommand::gelu_f32_tensor(first).unwrap()
                ),
                (
                    DeviceCommandPhase::Compute,
                    Some(1),
                    ReferenceDeviceCommand::gelu_f32_tensor(second).unwrap()
                ),
            ],
            false
        )
        .is_err());
    assert_eq!(output.read(), [0xa5; 8]);
    assert_eq!(runtime.snapshot().submissions, 0);
}
