use super::super::tests::{buffer, config};
use super::*;
use ferrum_runtime::{AttributeId, ResolvedTensorLayout, ResolvedTensorSpec, SemanticValue};
use std::collections::BTreeMap;
fn launch(runtime: &ReferenceDeviceRuntime) -> ReferenceRowSelectedModulationLaunch {
    let t = |shape: Vec<u64>, dtype| {
        ResolvedTensorSpec::new(shape, dtype, ResolvedTensorLayout::Contiguous).unwrap()
    };
    let x = t(vec![3, 1], ElementType::Bf16);
    let p = t(vec![2, 2], ElementType::F32);
    let y = t(vec![3, 1], ElementType::F32);
    let attrs = BTreeMap::from([
        (
            AttributeId::new("run_lengths").unwrap(),
            SemanticValue::Integers(vec![1, 2]),
        ),
        (
            AttributeId::new("parameter_rows").unwrap(),
            SemanticValue::Integers(vec![1, 0]),
        ),
    ]);
    let geometry = RowSelectedModulationBf16F32Geometry::new(&x, &p, &y, &attrs).unwrap();
    let input: Vec<_> = [1.5_f32, 2., -2.]
        .into_iter()
        .flat_map(|x| bf16::from_f32(x).to_bits().to_le_bytes())
        .collect();
    let parameters: Vec<_> = [0.25_f32, 0.1, -1.5, 2.0_f32.powi(-23)]
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect();
    ReferenceRowSelectedModulationLaunch {
        input: buffer(runtime, "resource.row_mod.x", &input, 16, ElementType::Bf16)
            .region(0..6)
            .unwrap(),
        parameters: buffer(
            runtime,
            "resource.row_mod.p",
            &parameters,
            16,
            ElementType::F32,
        )
        .region(0..16)
        .unwrap(),
        output: buffer(
            runtime,
            "resource.row_mod.y",
            &[0xa5; 12],
            16,
            ElementType::F32,
        )
        .region(0..12)
        .unwrap(),
        geometry,
    }
}
#[test]
fn selected_f32_rows_preserve_separate_rounding_and_do_not_expand_parameters() {
    let runtime = ReferenceDeviceRuntime::new(config()).unwrap();
    let l = launch(&runtime);
    let output = l.output.clone();
    let before = runtime.snapshot().allocations;
    let command = ReferenceDeviceCommand::row_selected_modulation_tensor(l).unwrap();
    let mut stream = runtime.create_stream().unwrap();
    runtime
        .submit_entries(
            &mut stream,
            vec![(DeviceCommandPhase::Compute, Some(0), command)],
            false,
        )
        .unwrap();
    let got: Vec<_> = output
        .read()
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    assert_eq!(got, [2.0_f32.powi(-22), 2.45, -1.95]);
    assert_ne!(got[1], 2.0 * (1.0 + bf16::from_f32(0.1).to_f32()) + 0.25);
    assert_ne!(got[0], 1.5_f32.mul_add(1.0 + 2.0_f32.powi(-23), -1.5));
    assert_eq!(runtime.snapshot().allocations, before);
    drop(output);
    drop(stream);
    assert_eq!(runtime.snapshot().live_allocations, 0);
}
#[test]
fn bad_physical_ports_and_foreign_late_parameters_fail_before_output_writes() {
    let runtime = ReferenceDeviceRuntime::new(config()).unwrap();
    for case in 0..5 {
        let mut l = launch(&runtime);
        let output = l.output.clone();
        match case {
            0 => l.input.element_type = ElementType::F32,
            1 => l.parameters.element_type = ElementType::Bf16,
            2 => l.parameters.length_bytes -= 4,
            3 => l.output = l.parameters.clone(),
            4 => l.output.length_bytes -= 4,
            _ => unreachable!(),
        }
        assert!(ReferenceDeviceCommand::row_selected_modulation_tensor(l).is_err());
        assert_eq!(output.read(), [0xa5; 12]);
    }
    let foreign = ReferenceDeviceRuntime::new(config()).unwrap();
    let first = launch(&runtime);
    let output = first.output.clone();
    let mut second = launch(&runtime);
    second.parameters = launch(&foreign).parameters;
    let mut stream = runtime.create_stream().unwrap();
    assert!(runtime
        .submit_entries(
            &mut stream,
            vec![
                (
                    DeviceCommandPhase::Compute,
                    Some(0),
                    ReferenceDeviceCommand::row_selected_modulation_tensor(first).unwrap()
                ),
                (
                    DeviceCommandPhase::Compute,
                    Some(1),
                    ReferenceDeviceCommand::row_selected_modulation_tensor(second).unwrap()
                ),
            ],
            false
        )
        .is_err());
    assert_eq!(output.read(), [0xa5; 12]);
    assert_eq!(runtime.snapshot().submissions, 0);
    drop(output);
    drop(stream);
    assert_eq!(runtime.snapshot().live_allocations, 0);
    assert_eq!(foreign.snapshot().live_allocations, 0);
}
