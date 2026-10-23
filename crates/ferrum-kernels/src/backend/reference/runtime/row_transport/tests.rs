use super::super::tests::{buffer, config};
use super::super::{
    DeviceCommandPhase, DeviceRuntime, ReferenceCommandKind, ReferenceDeviceCommand,
    ReferenceDeviceRuntime, StreamState,
};
use super::*;
use ferrum_runtime::{
    AttributeId, DeviceLogicalWork, ResolvedTensorLayout, ResolvedTensorSpec, SemanticValue,
};
use std::collections::BTreeMap;

fn tensor(rows: u64, columns: u64) -> ResolvedTensorSpec {
    ResolvedTensorSpec::new(
        vec![rows, columns],
        ElementType::Bf16,
        ResolvedTensorLayout::Contiguous,
    )
    .unwrap()
}
fn attrs(
    batch: u64,
    start: u64,
    columns: u64,
    runs: &[[i64; 4]],
) -> BTreeMap<AttributeId, SemanticValue> {
    let mut a = BTreeMap::from([
        (
            AttributeId::new("batch").unwrap(),
            SemanticValue::Unsigned(batch),
        ),
        (
            AttributeId::new("column_start").unwrap(),
            SemanticValue::Unsigned(start),
        ),
        (
            AttributeId::new("column_count").unwrap(),
            SemanticValue::Unsigned(columns),
        ),
    ]);
    for (i, name) in [
        "run_lengths",
        "source_starts",
        "source_batch_strides",
        "source_row_strides",
    ]
    .into_iter()
    .enumerate()
    {
        a.insert(
            AttributeId::new(name).unwrap(),
            SemanticValue::Integers(runs.iter().map(|r| r[i]).collect()),
        );
    }
    a
}
fn bytes(values: &[u16]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}
fn bits(region: &ReferenceBufferRegion) -> Vec<u16> {
    region
        .read()
        .chunks_exact(2)
        .map(|b| u16::from_le_bytes([b[0], b[1]]))
        .collect()
}
fn region(runtime: &ReferenceDeviceRuntime, name: &str, values: &[u16]) -> ReferenceBufferRegion {
    let raw = bytes(values);
    buffer(runtime, name, &raw, 16, ElementType::Bf16)
        .region(0..raw.len() as u64)
        .unwrap()
}
fn gather(runtime: &ReferenceDeviceRuntime) -> ReferenceRowGatherLaunch {
    ReferenceRowGatherLaunch {
        source: region(
            runtime,
            "resource.gather.source",
            &[
                0x3f80, 0x8000, 0x7f81, 0x4000, 0x4040, 0xff81, 0x7fc1, 0x4080, 0x40a0, 0x7f80,
                0xff80, 0x40c0,
            ],
        ),
        output: region(runtime, "resource.gather.output", &[0xa5a5; 12]),
        geometry: RowGatherBf16Geometry::new(
            &tensor(3, 4),
            &tensor(6, 2),
            &attrs(2, 1, 2, &[[2, 2, 0, 0], [1, 0, 1, 0]]),
        )
        .unwrap(),
    }
}
fn concat(runtime: &ReferenceDeviceRuntime) -> ReferenceBatchedRowConcatLaunch {
    ReferenceBatchedRowConcatLaunch {
        left: region(
            runtime,
            "resource.concat.left",
            &[0, 0x8000, 0x7f81, 0x7fc1, 4, 5, 6, 7],
        ),
        right: region(runtime, "resource.concat.right", &[0xff80, 0x7f80, 10, 11]),
        output: region(runtime, "resource.concat.output", &[0xa5a5; 12]),
        geometry: BatchedRowConcatBf16Geometry::new(&tensor(4, 2), &tensor(2, 2), &tensor(6, 2), 2)
            .unwrap(),
    }
}
fn submit(runtime: &ReferenceDeviceRuntime, commands: Vec<ReferenceDeviceCommand>) {
    let mut stream = runtime.create_stream().unwrap();
    runtime
        .submit_entries(
            &mut stream,
            commands
                .into_iter()
                .map(|c| (DeviceCommandPhase::Compute, Some(0), c))
                .collect(),
            false,
        )
        .unwrap();
}
#[test]
fn gather_reads_shared_and_per_batch_rows_and_preserves_nonfinite_payload_bits() {
    let runtime = ReferenceDeviceRuntime::new(config()).unwrap();
    let launch = gather(&runtime);
    let output = launch.output.clone();
    let command = ReferenceDeviceCommand::row_gather_tensor(launch).unwrap();
    let attribution = command
        .attribution(0, Some(0), DeviceCommandPhase::Compute)
        .unwrap();
    assert_eq!(attribution.token_count(), None);
    assert_eq!(
        attribution.logical_work(),
        DeviceLogicalWork::Tensor {
            element_count: 12,
            arithmetic_operations: 0
        }
    );
    let before = runtime.snapshot();
    submit(&runtime, vec![command]);
    assert_eq!(
        bits(&output),
        [
            0x7f80, 0xff80, 0x7f80, 0xff80, 0x8000, 0x7f81, 0x7f80, 0xff80, 0x7f80, 0xff80, 0xff81,
            0x7fc1
        ]
    );
    assert_eq!(
        runtime.snapshot().row_gather_launches,
        before.row_gather_launches + 1
    );
    assert_eq!(runtime.snapshot().allocations, before.allocations);
}
#[test]
fn concat_interleaves_whole_sample_segments_instead_of_concatenating_flat_matrices() {
    let runtime = ReferenceDeviceRuntime::new(config()).unwrap();
    let launch = concat(&runtime);
    let output = launch.output.clone();
    let command = ReferenceDeviceCommand::batched_row_concat_tensor(launch).unwrap();
    assert_eq!(
        command
            .attribution(0, Some(0), DeviceCommandPhase::Compute)
            .unwrap()
            .logical_work(),
        DeviceLogicalWork::Tensor {
            element_count: 12,
            arithmetic_operations: 0
        }
    );
    let before = runtime.snapshot();
    submit(&runtime, vec![command]);
    assert_eq!(
        bits(&output),
        [0, 0x8000, 0x7f81, 0x7fc1, 0xff80, 0x7f80, 4, 5, 6, 7, 10, 11]
    );
    assert_eq!(
        runtime.snapshot().batched_row_concat_launches,
        before.batched_row_concat_launches + 1
    );
    assert_eq!(runtime.snapshot().allocations, before.allocations);
    // Private backend reads may alias. Core's logical-value NoAlias checks
    // still govern which public invocation may carry these identical sources.
    let source = region(&runtime, "resource.concat.shared-read", &[1, 2, 3, 4]);
    let output = region(&runtime, "resource.concat.read-alias-output", &[0; 8]);
    let alias = ReferenceBatchedRowConcatLaunch {
        left: source.clone(),
        right: source,
        output: output.clone(),
        geometry: BatchedRowConcatBf16Geometry::new(&tensor(2, 2), &tensor(2, 2), &tensor(4, 2), 2)
            .unwrap(),
    };
    submit(
        &runtime,
        vec![ReferenceDeviceCommand::batched_row_concat_tensor(alias).unwrap()],
    );
    assert_eq!(bits(&output), [1, 2, 1, 2, 3, 4, 3, 4]);
}
#[test]
fn disjoint_shared_backing_and_nonzero_regions_preserve_sentinels() {
    let runtime = ReferenceDeviceRuntime::new(config()).unwrap();
    let backing = buffer(
        &runtime,
        "resource.gather.shared",
        &[0xa5; 64],
        16,
        ElementType::Bf16,
    );
    let source = backing.region(2..34).unwrap();
    source.write(&bytes(&(0_u16..16).collect::<Vec<_>>()));
    let output = backing.region(42..58).unwrap();
    let before = backing.region(0..64).unwrap().read();
    let launch = ReferenceRowGatherLaunch {
        source,
        output: output.clone(),
        geometry: RowGatherBf16Geometry::new(
            &tensor(8, 2),
            &tensor(4, 2),
            &attrs(2, 0, 2, &[[2, 1, 4, 1]]),
        )
        .unwrap(),
    };
    submit(
        &runtime,
        vec![ReferenceDeviceCommand::row_gather_tensor(launch).unwrap()],
    );
    assert_eq!(bits(&output), [2, 3, 4, 5, 10, 11, 12, 13]);
    let after = backing.region(0..64).unwrap().read();
    for i in 0..64 {
        if !(42..58).contains(&i) {
            assert_eq!(after[i], before[i]);
        }
    }
}
#[test]
fn exact_extents_dtype_alignment_and_noalias_are_checked_before_execution() {
    let runtime = ReferenceDeviceRuntime::new(config()).unwrap();
    for mutation in 0..4 {
        let mut g = gather(&runtime);
        let output = g.output.clone();
        match mutation {
            0 => g.source.element_type = ElementType::F16,
            1 => g.source.offset_bytes += 1,
            2 => g.output.length_bytes -= 2,
            3 => g.output = g.source.clone(),
            _ => unreachable!(),
        }
        assert!(ReferenceDeviceCommand::row_gather_tensor(g).is_err());
        assert_eq!(bits(&output), [0xa5a5; 12]);
        let mut c = concat(&runtime);
        let output = c.output.clone();
        match mutation {
            0 => c.right.element_type = ElementType::F16,
            1 => c.left.offset_bytes += 1,
            2 => c.right.length_bytes -= 2,
            3 => c.output = c.left.clone(),
            _ => unreachable!(),
        }
        assert!(ReferenceDeviceCommand::batched_row_concat_tensor(c).is_err());
        assert_eq!(bits(&output), [0xa5a5; 12]);
    }
}
#[test]
fn late_invalid_or_foreign_source_preflight_prevents_all_prior_copies() {
    let runtime = ReferenceDeviceRuntime::new(config()).unwrap();
    let foreign = ReferenceDeviceRuntime::new(config()).unwrap();
    for foreign_source in [false, true] {
        let first = gather(&runtime);
        let output = first.output.clone();
        let mut second =
            ReferenceDeviceCommand::batched_row_concat_tensor(concat(&runtime)).unwrap();
        let ReferenceCommandKind::BatchedRowConcat { launch } = &mut second.kind else {
            unreachable!()
        };
        if foreign_source {
            launch.right = concat(&foreign).right;
        } else {
            launch.right.length_bytes -= 2;
        }
        let mut stream = runtime.create_stream().unwrap();
        let before = runtime.snapshot();
        assert!(runtime
            .submit_entries(
                &mut stream,
                vec![
                    (
                        DeviceCommandPhase::Compute,
                        Some(0),
                        ReferenceDeviceCommand::row_gather_tensor(first).unwrap()
                    ),
                    (DeviceCommandPhase::Compute, Some(1), second)
                ],
                false
            )
            .is_err());
        assert_eq!(runtime.stream_state(&stream), StreamState::Ready);
        assert_eq!(runtime.snapshot().commands, before.commands);
        assert_eq!(runtime.snapshot().submissions, before.submissions);
        assert_eq!(
            runtime.snapshot().row_gather_launches,
            before.row_gather_launches
        );
        assert_eq!(bits(&output), [0xa5a5; 12]);
    }
}
