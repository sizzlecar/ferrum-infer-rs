use super::super::tests::{buffer, config};
use super::super::{
    DeviceCommandPhase, DeviceRuntime, ReferenceDeviceCommand, ReferenceDeviceRuntime,
};
use super::*;
use ferrum_runtime::{
    AttributeId, CanonicalRational, DeviceLogicalWork, ResolvedTensorLayout, ResolvedTensorSpec,
    SemanticValue,
};
use std::collections::BTreeMap;
fn encoded(x: &[f32]) -> Vec<u8> {
    x.iter().flat_map(|x| x.to_le_bytes()).collect()
}
#[test]
fn nonpower_dot_carries_unmatched_subtrees_and_preserves_power_two_order() {
    let negative_zero = dot(3, 0, &|_| (-0., 1.));
    assert_eq!(negative_zero.to_bits(), (-0.0f32).to_bits());
    assert_eq!(dot(3, 0, &|i| ([33554432., -33554432., 1.][i], 1.)), 1.);
    assert_eq!(
        super::super::structured_attention::dot(3, 0, &|i| ([33554432., -33554432., 1.][i], 1.)),
        0.
    );
    let mut tail = [0.0f32; 72];
    tail[64..67].copy_from_slice(&[33554432., -33554432., 1.]);
    assert_eq!(dot(72, 0, &|i| (tail[i], 1.)), 1.);
    assert_eq!(
        super::super::structured_attention::dot(72, 0, &|i| (tail[i], 1.)),
        0.
    );
    for width in [2, 4, 8, 16, 32, 64, 128, 256] {
        let read = |i| (((i * 11 % 17) as f32 - 8.) / 16., (i % 3) as f32 - 1.);
        assert_eq!(
            dot(width, 0, &read).to_bits(),
            super::super::structured_attention::dot(width, 0, &read).to_bits()
        );
    }
}
fn values(x: &ReferenceBufferRegion) -> Vec<f32> {
    x.read()
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect()
}
fn geometry(
    rows: u64,
    columns: u64,
    kv_columns: u64,
    s: u64,
    d: u64,
    segments: &[i64],
    bidir: &[i64],
    counts: &[i64],
    runs: &[i64],
    valid: &[i64],
) -> StructuredGqaAttentionF32Geometry {
    let t = ResolvedTensorSpec::new(
        vec![rows, columns],
        ElementType::F32,
        ResolvedTensorLayout::Contiguous,
    )
    .unwrap();
    let mut a = BTreeMap::from([
        (
            AttributeId::new("sequence_length").unwrap(),
            SemanticValue::Unsigned(s),
        ),
        (
            AttributeId::new("head_dim").unwrap(),
            SemanticValue::Unsigned(d),
        ),
        (
            AttributeId::new("scale").unwrap(),
            SemanticValue::Rational(CanonicalRational::new(1, 1).unwrap()),
        ),
    ]);
    for (name, x) in [
        ("segment_lengths", segments),
        ("segment_bidirectional", bidir),
        ("key_run_counts", counts),
        ("key_run_lengths", runs),
        ("key_run_valid", valid),
    ] {
        a.insert(
            AttributeId::new(name).unwrap(),
            SemanticValue::Integers(x.to_vec()),
        );
    }
    let kv = ResolvedTensorSpec::new(
        vec![rows, kv_columns],
        ElementType::F32,
        ResolvedTensorLayout::Contiguous,
    )
    .unwrap();
    StructuredGqaAttentionF32Geometry::new(&t, &kv, &kv, &t, &a).unwrap()
}
fn fixture(
    runtime: &ReferenceDeviceRuntime,
    g: StructuredGqaAttentionF32Geometry,
    q: &[f32],
    k: &[f32],
    v: &[f32],
) -> ReferenceStructuredGqaAttentionF32Launch {
    let regions: [ReferenceBufferRegion; 3] = std::array::from_fn(|i| {
        let x = encoded([q, k, v][i]);
        buffer(
            runtime,
            &format!("resource.attention.input{i}"),
            &x,
            16,
            ElementType::F32,
        )
        .region(0..x.len() as u64)
        .unwrap()
    });
    let bytes = usize::try_from(g.element_count() * 4).unwrap();
    let output = buffer(
        runtime,
        "resource.attention.y",
        &vec![0xa5; bytes],
        16,
        ElementType::F32,
    )
    .region(0..bytes as u64)
    .unwrap();
    let [query, key, value] = regions;
    ReferenceStructuredGqaAttentionF32Launch {
        query,
        key,
        value,
        output,
        geometry: g,
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
fn compact_kv_maps_to_query_groups_across_batches_with_key_only_padding() {
    let runtime = ReferenceDeviceRuntime::new(config()).unwrap();
    let g = geometry(
        6,
        8,
        4,
        3,
        2,
        &[3],
        &[0],
        &[3, 2],
        &[1, 1, 1, 1, 2],
        &[1, 0, 1, 0, 1],
    );
    assert_eq!(
        (
            g.visible_pairs(),
            g.nonempty_rows(),
            g.logical_arithmetic_operations()
        ),
        (7, 5, 268)
    );
    let mut q = vec![0.; 48];
    q[24..32].fill(f32::NAN); // Empty batch-1/query-0.
    let mut k = vec![0.; 24];
    k[4..8].fill(f32::NAN);
    k[12..16].fill(f32::INFINITY);
    let v = [
        2.,
        4.,
        10.,
        20.,
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
        6.,
        8.,
        30.,
        40.,
        f32::NAN,
        f32::NAN,
        f32::NAN,
        f32::NAN,
        8.,
        12.,
        40.,
        60.,
        12.,
        16.,
        60.,
        80.,
    ];
    let launch = fixture(&runtime, g, &q, &k, &v);
    let output = launch.output.clone();
    assert_eq!(launch.query.length_bytes, 192);
    assert_eq!(launch.key.length_bytes, 96);
    let command = ReferenceDeviceCommand::structured_gqa_attention_f32_tensor(launch).unwrap();
    assert_eq!(
        command
            .attribution(0, Some(0), DeviceCommandPhase::Compute)
            .unwrap()
            .logical_work(),
        DeviceLogicalWork::Tensor {
            element_count: 48,
            arithmetic_operations: 268
        }
    );
    let before = runtime.snapshot();
    submit(&runtime, command);
    assert_eq!(
        output.read(),
        encoded(&[
            2., 4., 2., 4., 10., 20., 10., 20., 2., 4., 2., 4., 10., 20., 10., 20., 4., 6., 4., 6.,
            20., 30., 20., 30., 0., 0., 0., 0., 0., 0., 0., 0., 8., 12., 8., 12., 40., 60., 40.,
            60., 10., 14., 10., 14., 50., 70., 50., 70.
        ])
    );
    assert_eq!(runtime.snapshot().allocations, before.allocations);
    assert_eq!(
        runtime.snapshot().structured_gqa_attention_launches,
        before.structured_gqa_attention_launches + 1
    );
}
#[test]
fn independent_nonuniform_two_key_cells_distinguish_query_and_kv_head_indices() {
    let runtime = ReferenceDeviceRuntime::new(config()).unwrap();
    let g = geometry(2, 8, 4, 2, 2, &[2], &[1], &[1], &[2], &[1]);
    let q = [1., 0., -1., 0., 1., 0., -1., 0.].repeat(2);
    let k = [0., 0., 0., 0., 1., 0., -1., 0.];
    let v = [0., 0., 0., 0., 1., 1., 2., -2.];
    let launch = fixture(&runtime, g, &q, &k, &v);
    let output = launch.output.clone();
    submit(
        &runtime,
        ReferenceDeviceCommand::structured_gqa_attention_f32_tensor(launch).unwrap(),
    );
    let p = 1.0_f64 / (1.0 + (-1.0_f64).exp());
    let expected = [
        p,
        p,
        1.0 - p,
        1.0 - p,
        2.0 * (1.0 - p),
        -2.0 * (1.0 - p),
        2.0 * p,
        -2.0 * p,
    ]
    .repeat(2);
    for (got, expected) in values(&output).iter().zip(expected) {
        assert!((f64::from(*got) - expected).abs() < 2.0e-7);
    }
    assert!(values(&output)
        .iter()
        .any(|x| half::bf16::from_f32(*x).to_f32() != *x));
}
fn small(runtime: &ReferenceDeviceRuntime) -> ReferenceStructuredGqaAttentionF32Launch {
    fixture(
        runtime,
        geometry(2, 4, 2, 2, 2, &[2], &[0], &[1], &[2], &[1]),
        &[0.; 8],
        &[0.; 4],
        &[1.; 4],
    )
}
#[test]
fn visible_nonfinite_score_poisons_only_the_mapped_query_group() {
    let runtime = ReferenceDeviceRuntime::new(config()).unwrap();
    let g = geometry(1, 8, 4, 1, 2, &[1], &[0], &[1], &[1], &[1]);
    let launch = fixture(
        &runtime,
        g,
        &[0.; 8],
        &[f32::INFINITY, 0., 0., 0.],
        &[1., 2., 3., 4.],
    );
    let output = launch.output.clone();
    submit(
        &runtime,
        ReferenceDeviceCommand::structured_gqa_attention_f32_tensor(launch).unwrap(),
    );
    let b = values(&output);
    assert!(b[..4].iter().all(|x| x.is_nan()));
    assert_eq!(&b[4..], &[3.0, 4.0, 3.0, 4.0]);
}
#[test]
fn exact_compact_kv_extents_noalias_and_foreign_late_read_fail_before_writes() {
    let runtime = ReferenceDeviceRuntime::new(config()).unwrap();
    for mutation in 0..6 {
        let mut launch = small(&runtime);
        let output = launch.output.clone();
        match mutation {
            0 => launch.key.length_bytes -= 4,
            1 => launch.value = launch.query.clone(),
            2 => launch.key.element_type = ElementType::F16,
            3 => launch.value.offset_bytes += 1,
            4 => launch.output = launch.query.clone(),
            5 => launch.output.length_bytes -= 4,
            _ => unreachable!(),
        }
        assert!(ReferenceDeviceCommand::structured_gqa_attention_f32_tensor(launch).is_err());
        assert_eq!(output.read(), [0xa5; 32]);
    }
    let foreign = ReferenceDeviceRuntime::new(config()).unwrap();
    let first = small(&runtime);
    let output = first.output.clone();
    let mut second = small(&runtime);
    second.key = small(&foreign).key;
    let mut stream = runtime.create_stream().unwrap();
    let before = runtime.snapshot();
    assert!(runtime
        .submit_entries(
            &mut stream,
            vec![
                (
                    DeviceCommandPhase::Compute,
                    Some(0),
                    ReferenceDeviceCommand::structured_gqa_attention_f32_tensor(first).unwrap()
                ),
                (
                    DeviceCommandPhase::Compute,
                    Some(1),
                    ReferenceDeviceCommand::structured_gqa_attention_f32_tensor(second).unwrap()
                )
            ],
            false
        )
        .is_err());
    assert_eq!(output.read(), [0xa5; 32]);
    assert_eq!(runtime.snapshot().submissions, before.submissions);
}

#[test]
fn nonpower_d72_physical_extents_reject_truncation_and_alias_before_writes() {
    let runtime = ReferenceDeviceRuntime::new(config()).unwrap();
    for mutation in 0..4 {
        let g = geometry(2, 4 * 72, 2 * 72, 2, 72, &[2], &[1], &[1], &[2], &[1]);
        let mut launch = fixture(
            &runtime,
            g,
            &vec![0.; 2 * 4 * 72],
            &vec![0.; 2 * 2 * 72],
            &vec![1.; 2 * 2 * 72],
        );
        let output = launch.output.clone();
        match mutation {
            0 => launch.key.length_bytes -= 1,
            1 => launch.value.length_bytes -= 4,
            2 => launch.output = launch.query.clone(),
            _ => launch.output.length_bytes -= 1,
        }
        assert!(ReferenceDeviceCommand::structured_gqa_attention_f32_tensor(launch).is_err());
        assert_eq!(output.read(), vec![0xa5; 2 * 4 * 72 * 4]);
    }
    assert_eq!(runtime.snapshot().live_allocations, 0);
}
