use super::super::tests::{buffer, config};
use super::*;
use ferrum_runtime::{ResolvedTensorLayout, ResolvedTensorSpec};
fn launch(runtime: &ReferenceDeviceRuntime) -> ReferenceBatchedSuffixAppendF32Launch {
    let t = |d: Vec<u64>| {
        ResolvedTensorSpec::new(d, ElementType::F32, ResolvedTensorLayout::Contiguous).unwrap()
    };
    let bits = |v: &[u32]| v.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>();
    let input = bits(&[
        0x3f800001, 0x80000000, 0x40000000, 0x7f800001, 0xbf800000, 1, 0x7f800000, 0xff800000,
    ]);
    let suffix = bits(&[0x7fa12345, 0x80000000, 0x3f000001, 0x80000001]);
    ReferenceBatchedSuffixAppendF32Launch {
        input: buffer(runtime, "resource.suffix.x", &input, 16, ElementType::F32)
            .region(0..32)
            .unwrap(),
        suffix: buffer(
            runtime,
            "resource.suffix.shared",
            &suffix,
            16,
            ElementType::F32,
        )
        .region(0..16)
        .unwrap(),
        output: buffer(
            runtime,
            "resource.suffix.y",
            &[0xa5; 80],
            16,
            ElementType::F32,
        )
        .region(0..80)
        .unwrap(),
        geometry: BatchedSuffixAppendF32Geometry::new(
            &t(vec![4, 2]),
            &t(vec![2, 2]),
            &t(vec![10, 2]),
            2,
            1,
        )
        .unwrap(),
    }
}
#[test]
fn suffix_and_zero_rows_are_per_request_exact_bits_without_expanded_storage() {
    let runtime = ReferenceDeviceRuntime::new(config()).unwrap();
    let l = launch(&runtime);
    let output = l.output.clone();
    let allocations = runtime.snapshot().allocations;
    let mut stream = runtime.create_stream().unwrap();
    runtime
        .submit_entries(
            &mut stream,
            vec![(
                DeviceCommandPhase::Compute,
                Some(0),
                ReferenceDeviceCommand::batched_suffix_append_f32_tensor(l).unwrap(),
            )],
            false,
        )
        .unwrap();
    let bits: Vec<_> = output
        .read()
        .chunks_exact(4)
        .map(|v| u32::from_le_bytes(v.try_into().unwrap()))
        .collect();
    assert_eq!(
        bits,
        [
            0x3f800001, 0x80000000, 0x40000000, 0x7f800001, 0x7fa12345, 0x80000000, 0x3f000001,
            0x80000001, 0, 0, 0xbf800000, 1, 0x7f800000, 0xff800000, 0x7fa12345, 0x80000000,
            0x3f000001, 0x80000001, 0, 0
        ]
    );
    assert_eq!(runtime.snapshot().allocations, allocations);
    drop(output);
    drop(stream);
    assert_eq!(runtime.snapshot().live_allocations, 0);
}
#[test]
fn physical_ports_and_foreign_suffix_reject_before_any_batch_write() {
    let runtime = ReferenceDeviceRuntime::new(config()).unwrap();
    for case in 0..4 {
        let mut l = launch(&runtime);
        let output = l.output.clone();
        match case {
            0 => l.suffix.element_type = ElementType::Bf16,
            1 => l.input.length_bytes -= 4,
            2 => l.output = l.input.clone(),
            3 => l.output.length_bytes -= 4,
            _ => unreachable!(),
        };
        assert!(ReferenceDeviceCommand::batched_suffix_append_f32_tensor(l).is_err());
        assert_eq!(output.read(), [0xa5; 80]);
    }
    let foreign = ReferenceDeviceRuntime::new(config()).unwrap();
    let first = launch(&runtime);
    let output = first.output.clone();
    let mut last = launch(&runtime);
    last.suffix = launch(&foreign).suffix;
    let mut stream = runtime.create_stream().unwrap();
    assert!(runtime
        .submit_entries(
            &mut stream,
            vec![
                (
                    DeviceCommandPhase::Compute,
                    Some(0),
                    ReferenceDeviceCommand::batched_suffix_append_f32_tensor(first).unwrap()
                ),
                (
                    DeviceCommandPhase::Compute,
                    Some(1),
                    ReferenceDeviceCommand::batched_suffix_append_f32_tensor(last).unwrap()
                )
            ],
            false
        )
        .is_err());
    assert_eq!(output.read(), [0xa5; 80]);
    assert_eq!(runtime.snapshot().submissions, 0);
    drop(output);
    drop(stream);
    assert_eq!(runtime.snapshot().live_allocations, 0);
    assert_eq!(foreign.snapshot().live_allocations, 0);
}
