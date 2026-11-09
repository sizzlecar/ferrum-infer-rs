use super::super::tests::{buffer, config};
use super::*;
use ferrum_runtime::{
    AttributeId, ResolvedTensorLayout, ResolvedTensorSpec, RowScatterF32Geometry, RowScatterF32Op,
    SemanticValue,
};
use std::collections::BTreeMap;
fn launch(runtime: &ReferenceDeviceRuntime, kind: RowScatterF32Op) -> ReferenceRowScatterF32Launch {
    let x = ResolvedTensorSpec::new(
        vec![6, 4],
        ElementType::F32,
        ResolvedTensorLayout::Contiguous,
    )
    .unwrap();
    let u = ResolvedTensorSpec::new(
        vec![3, 4],
        ElementType::F32,
        ResolvedTensorLayout::Contiguous,
    )
    .unwrap();
    let attrs = BTreeMap::from([
        (
            AttributeId::new("destination_starts").unwrap(),
            SemanticValue::Integers(vec![1, 5]),
        ),
        (
            AttributeId::new("run_lengths").unwrap(),
            SemanticValue::Integers(vec![2, 1]),
        ),
    ]);
    let input = [
        0x7f800001, 0xffc12345, 0x80000000, 0x00000001, 0x3f800800, 0x3f800000, 0x7f800000,
        0x7fc12345, 0x3f800000, 0x80000000, 0x00000001, 0xff800000, 0x7fc98765, 0x80000000,
        0x00000003, 0x3f810000, 0x00000000, 0x00800000, 0x80000001, 0xbf800000, 0x3f800000,
        0x3f800000, 0x3f800000, 0x3f800000,
    ];
    let updates = [
        0xbf800000, 0x34400000, 0xff800000, 0x3f800000, 0x7f800001, 0x80000000, 0x00000001,
        0x3f800000, 0x3f800800, 0x7fc12345, 0x80000000, 0x00000001,
    ];
    ReferenceRowScatterF32Launch {
        input: buffer(
            runtime,
            "resource.scatter.input",
            &input
                .into_iter()
                .flat_map(u32::to_le_bytes)
                .collect::<Vec<_>>(),
            16,
            ElementType::F32,
        )
        .region(0..96)
        .unwrap(),
        updates: buffer(
            runtime,
            "resource.scatter.updates",
            &updates
                .into_iter()
                .flat_map(u32::to_le_bytes)
                .collect::<Vec<_>>(),
            16,
            ElementType::F32,
        )
        .region(0..48)
        .unwrap(),
        output: buffer(
            runtime,
            "resource.scatter.output",
            &[0xa5; 96],
            16,
            ElementType::F32,
        )
        .region(0..96)
        .unwrap(),
        geometry: RowScatterF32Geometry::new(&x, &u, &x, &attrs).unwrap(),
        kind,
    }
}
#[test]
fn real_commands_preserve_untouched_bits_and_replace_nonfinite_placeholders_without_reading_them() {
    let runtime = ReferenceDeviceRuntime::new(config()).unwrap();
    for kind in [RowScatterF32Op::Replace, RowScatterF32Op::Add] {
        let l = launch(&runtime, kind);
        let base = l.input.read();
        let updates = l.updates.read();
        let output = l.output.clone();
        let geometry = l.geometry.clone();
        let command = ReferenceDeviceCommand::row_scatter_f32_tensor(l).unwrap();
        let mut stream = runtime.create_stream().unwrap();
        runtime
            .submit_entries(
                &mut stream,
                vec![(DeviceCommandPhase::Compute, Some(0), command)],
                false,
            )
            .unwrap();
        let actual = output.read();
        for row in 0usize..6 {
            for col in 0..4 {
                let i = (row * 4 + col) * 4;
                let actual = u32::from_le_bytes(actual[i..i + 4].try_into().unwrap());
                if let Some(update) = geometry.update_row(row as u64) {
                    let j = (update as usize * 4 + col) * 4;
                    let update = u32::from_le_bytes(updates[j..j + 4].try_into().unwrap());
                    if kind == RowScatterF32Op::Replace {
                        assert_eq!(actual, update);
                    } else {
                        let expected =
                            (f64::from(f32::from_le_bytes(base[i..i + 4].try_into().unwrap()))
                                + f64::from(f32::from_bits(update)))
                                as f32;
                        if expected.is_nan() {
                            assert!(f32::from_bits(actual).is_nan());
                        } else {
                            assert_eq!(actual, expected.to_bits());
                        }
                    }
                } else {
                    assert_eq!(&actual.to_le_bytes(), &base[i..i + 4]);
                }
            }
        }
    }
    assert_eq!(runtime.snapshot().live_allocations, 0);
}
#[test]
fn original_dtype_extents_alias_and_foreign_updates_fail_before_any_submission_write() {
    let runtime = ReferenceDeviceRuntime::new(config()).unwrap();
    for mutation in 0..5 {
        let mut l = launch(&runtime, RowScatterF32Op::Replace);
        let original = l.output.clone();
        match mutation {
            0 => l.input.element_type = ElementType::Bf16,
            1 => l.updates.length_bytes -= 4,
            2 => l.output = l.input.clone(),
            3 => l.output = l.updates.clone(),
            4 => l.updates.offset_bytes += 1,
            _ => unreachable!(),
        };
        assert!(ReferenceDeviceCommand::row_scatter_f32_tensor(l).is_err());
        assert_eq!(original.read(), [0xa5; 96]);
    }
    let foreign = ReferenceDeviceRuntime::new(config()).unwrap();
    let first = launch(&runtime, RowScatterF32Op::Add);
    let out = first.output.clone();
    let mut second = launch(&runtime, RowScatterF32Op::Replace);
    second.updates = launch(&foreign, RowScatterF32Op::Replace).updates;
    let mut stream = runtime.create_stream().unwrap();
    assert!(runtime
        .submit_entries(
            &mut stream,
            vec![
                (
                    DeviceCommandPhase::Compute,
                    Some(0),
                    ReferenceDeviceCommand::row_scatter_f32_tensor(first).unwrap()
                ),
                (
                    DeviceCommandPhase::Compute,
                    Some(1),
                    ReferenceDeviceCommand::row_scatter_f32_tensor(second).unwrap()
                )
            ],
            false
        )
        .is_err());
    assert_eq!(out.read(), [0xa5; 96]);
    assert_eq!(runtime.snapshot().submissions, 0);
}
