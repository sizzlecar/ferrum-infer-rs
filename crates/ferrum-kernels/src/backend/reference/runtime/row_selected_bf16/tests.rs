use super::super::tests::{buffer, config};
use super::*;
use ferrum_runtime::{AttributeId, ResolvedTensorLayout, ResolvedTensorSpec, SemanticValue};
use std::collections::BTreeMap;
fn launch(
    runtime: &ReferenceDeviceRuntime,
    kind: RowSelectedBf16Kind,
) -> ReferenceRowSelectedBf16Launch {
    let tensor = |shape| {
        ResolvedTensorSpec::new(shape, ElementType::Bf16, ResolvedTensorLayout::Contiguous).unwrap()
    };
    let x = tensor(vec![2, 2]);
    let p = tensor(vec![2, 8]);
    let mut attrs = BTreeMap::from([
        (
            AttributeId::new("run_lengths").unwrap(),
            SemanticValue::Integers(vec![1, 1]),
        ),
        (
            AttributeId::new("parameter_rows").unwrap(),
            SemanticValue::Integers(vec![1, 0]),
        ),
    ]);
    for (name, column) in if kind == RowSelectedBf16Kind::Modulation {
        vec![("shift_column", 1), ("scale_column", 3)]
    } else {
        vec![("gate_column", 5)]
    } {
        attrs.insert(
            AttributeId::new(name).unwrap(),
            SemanticValue::Unsigned(column),
        );
    }
    let geometry = RowSelectedBf16Geometry::new(
        kind,
        &if kind == RowSelectedBf16Kind::Modulation {
            vec![&x, &p]
        } else {
            vec![&x, &p, &x]
        },
        &x,
        &attrs,
    )
    .unwrap();
    let input = [0.5f32, -1., 1., 2.]
        .into_iter()
        .flat_map(|v| bf16::from_f32(v).to_bits().to_le_bytes())
        .collect::<Vec<_>>();
    let parameters = (0..16)
        .flat_map(|i| {
            bf16::from_f32(i as f32 / 16. - 0.25)
                .to_bits()
                .to_le_bytes()
        })
        .collect::<Vec<_>>();
    ReferenceRowSelectedBf16Launch {
        input: buffer(runtime, "resource.compact.x", &input, 16, ElementType::Bf16)
            .region(0..8)
            .unwrap(),
        parameters: buffer(
            runtime,
            "resource.compact.p",
            &parameters,
            16,
            ElementType::Bf16,
        )
        .region(0..32)
        .unwrap(),
        update: if kind == RowSelectedBf16Kind::Addcmul {
            Some(
                buffer(runtime, "resource.compact.u", &input, 16, ElementType::Bf16)
                    .region(0..8)
                    .unwrap(),
            )
        } else {
            None
        },
        output: buffer(
            runtime,
            "resource.compact.y",
            &[0xa5; 8],
            16,
            ElementType::Bf16,
        )
        .region(0..8)
        .unwrap(),
        geometry,
    }
}
#[test]
fn row_selected_bf16_rejects_wrong_physical_ports_aliases_and_update_presence() {
    let runtime = ReferenceDeviceRuntime::new(config()).unwrap();
    for kind in [
        RowSelectedBf16Kind::Modulation,
        RowSelectedBf16Kind::Addcmul,
    ] {
        for case in 0..5 {
            let mut l = launch(&runtime, kind);
            let output = l.output.clone();
            match case {
                0 => l.parameters.element_type = ElementType::F32,
                1 => l.parameters.length_bytes -= 2,
                2 => l.output = l.input.clone(),
                3 => l.output.length_bytes -= 2,
                _ => {
                    l.update = if kind == RowSelectedBf16Kind::Modulation {
                        Some(l.input.clone())
                    } else {
                        None
                    }
                }
            };
            assert!(ReferenceDeviceCommand::row_selected_bf16_tensor(l).is_err());
            assert_eq!(output.read(), [0xa5; 8]);
        }
    }
    assert_eq!(runtime.snapshot().live_allocations, 0);
}
#[test]
fn row_selected_bf16_foreign_late_table_preserves_all_outputs_before_submission() {
    let runtime = ReferenceDeviceRuntime::new(config()).unwrap();
    let foreign = ReferenceDeviceRuntime::new(config()).unwrap();
    let first = launch(&runtime, RowSelectedBf16Kind::Modulation);
    let output = first.output.clone();
    let mut second = launch(&runtime, RowSelectedBf16Kind::Addcmul);
    second.parameters = launch(&foreign, RowSelectedBf16Kind::Addcmul).parameters;
    let mut stream = runtime.create_stream().unwrap();
    assert!(runtime
        .submit_entries(
            &mut stream,
            vec![
                (
                    DeviceCommandPhase::Compute,
                    Some(0),
                    ReferenceDeviceCommand::row_selected_bf16_tensor(first).unwrap()
                ),
                (
                    DeviceCommandPhase::Compute,
                    Some(1),
                    ReferenceDeviceCommand::row_selected_bf16_tensor(second).unwrap()
                )
            ],
            false
        )
        .is_err());
    assert_eq!(output.read(), [0xa5; 8]);
    drop(output);
    drop(stream);
    assert_eq!(runtime.snapshot().live_allocations, 0);
    assert_eq!(foreign.snapshot().live_allocations, 0);
}
