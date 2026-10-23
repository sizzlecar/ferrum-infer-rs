use super::super::tests::{buffer, config};
use super::super::{
    DeviceCommandPhase, DeviceRuntime, ReferenceDeviceCommand, ReferenceDeviceRuntime,
};
use super::*;
use ferrum_runtime::{
    AttributeId, CanonicalRational, ResolvedTensorLayout, ResolvedTensorSpec, SemanticValue,
};
use std::collections::BTreeMap;

fn geometry(
    batch: u64,
    q: u64,
    kv: u64,
    d: u64,
    counts: &[i64],
    lengths: &[i64],
    valid: &[i64],
) -> CrossAttentionBf16Geometry {
    geometry_with_heads([batch, q, kv, d, 1], counts, lengths, valid)
}
fn geometry_with_heads(
    [batch, q, kv, d, heads]: [u64; 5],
    counts: &[i64],
    lengths: &[i64],
    valid: &[i64],
) -> CrossAttentionBf16Geometry {
    let t = |rows| {
        ResolvedTensorSpec::new(
            vec![rows, heads * d],
            ElementType::Bf16,
            ResolvedTensorLayout::Contiguous,
        )
        .unwrap()
    };
    let a: BTreeMap<_, _> = [
        ("batch", SemanticValue::Unsigned(batch)),
        ("head_dim", SemanticValue::Unsigned(d)),
        (
            "scale",
            SemanticValue::Rational(CanonicalRational::new(1, 1).unwrap()),
        ),
        ("key_run_counts", SemanticValue::Integers(counts.to_vec())),
        ("key_run_lengths", SemanticValue::Integers(lengths.to_vec())),
        ("key_run_valid", SemanticValue::Integers(valid.to_vec())),
    ]
    .into_iter()
    .map(|(k, v)| (AttributeId::new(k).unwrap(), v))
    .collect();
    CrossAttentionBf16Geometry::new(
        &t(batch * q),
        &t(batch * kv),
        &t(batch * kv),
        &t(batch * q),
        &a,
    )
    .unwrap()
}
fn fixture(
    runtime: &ReferenceDeviceRuntime,
    g: CrossAttentionBf16Geometry,
    q: &[f32],
    k: &[f32],
    v: &[f32],
) -> ReferenceCrossAttentionLaunch {
    let input = |name: &str, x: &[f32]| {
        let bytes: Vec<_> = x
            .iter()
            .flat_map(|x| bf16::from_f32(*x).to_bits().to_le_bytes())
            .collect();
        buffer(runtime, name, &bytes, 16, ElementType::Bf16)
            .region(0..bytes.len() as u64)
            .unwrap()
    };
    let output = buffer(
        runtime,
        "resource.cross.y",
        &vec![0xa5; g.query_bytes() as usize],
        16,
        ElementType::Bf16,
    )
    .region(0..g.query_bytes())
    .unwrap();
    ReferenceCrossAttentionLaunch {
        query: input("resource.cross.q", q),
        key: input("resource.cross.k", k),
        value: input("resource.cross.v", v),
        output,
        geometry: g,
    }
}
fn bits(r: &ReferenceBufferRegion) -> Vec<u16> {
    r.read()
        .chunks_exact(2)
        .map(|b| u16::from_le_bytes([b[0], b[1]]))
        .collect()
}
fn submit(runtime: &ReferenceDeviceRuntime, launch: ReferenceCrossAttentionLaunch) {
    let mut stream = runtime.create_stream().unwrap();
    runtime
        .submit_entries(
            &mut stream,
            vec![(
                DeviceCommandPhase::Compute,
                Some(0),
                ReferenceDeviceCommand::cross_attention_tensor(launch).unwrap(),
            )],
            false,
        )
        .unwrap();
}
#[test]
fn unequal_lengths_preserve_batches_holes_and_empty_queries_without_reading_poison() {
    let runtime = ReferenceDeviceRuntime::new(config()).unwrap();
    let g = geometry(2, 2, 3, 2, &[3, 1], &[1, 1, 1, 3], &[1, 0, 1, 0]);
    let launch = fixture(
        &runtime,
        g,
        &[0., 0., 0., 0., f32::NAN, f32::INFINITY, f32::NAN, f32::NAN],
        &[
            0.,
            0.,
            f32::NAN,
            f32::INFINITY,
            0.,
            0.,
            f32::NAN,
            f32::NAN,
            f32::NAN,
            f32::NAN,
            f32::NAN,
            f32::NAN,
        ],
        &[
            2.,
            4.,
            f32::NAN,
            f32::NEG_INFINITY,
            6.,
            8.,
            f32::NAN,
            f32::NAN,
            f32::NAN,
            f32::NAN,
            f32::NAN,
            f32::NAN,
        ],
    );
    let output = launch.output.clone();
    let before = runtime.snapshot();
    submit(&runtime, launch);
    assert_eq!(bits(&output), [0x4080, 0x40c0, 0x4080, 0x40c0, 0, 0, 0, 0]);
    let after = runtime.snapshot();
    assert_eq!(after.allocations, before.allocations);
    assert_eq!(
        after.cross_attention_launches,
        before.cross_attention_launches + 1
    );
}
#[test]
fn rectangular_nonempty_batches_and_heads_use_distinct_value_rows() {
    let runtime = ReferenceDeviceRuntime::new(config()).unwrap();
    let launch = fixture(
        &runtime,
        geometry_with_heads([2, 2, 3, 2, 2], &[1, 1], &[3, 3], &[1, 1]),
        &[0.; 16],
        &[0.; 24],
        &[
            2., 4., -2., -4., 4., 8., -4., -8., 6., 12., -6., -12., 10., 20., -10., -20., 20., 40.,
            -20., -40., 30., 60., -30., -60.,
        ],
    );
    let output = launch.output.clone();
    submit(&runtime, launch);
    // Uniform scores give the per-head arithmetic mean of each batch's three
    // rows; Q=2 and KV=3 deliberately have different batch strides.
    assert_eq!(
        bits(&output),
        [
            0x4080, 0x4100, 0xc080, 0xc100, 0x4080, 0x4100, 0xc080, 0xc100, 0x41a0, 0x4220, 0xc1a0,
            0xc220, 0x41a0, 0x4220, 0xc1a0, 0xc220,
        ]
    );
}
#[test]
fn rectangular_nonuniform_softmax_has_independent_bf16_expected() {
    let runtime = ReferenceDeviceRuntime::new(config()).unwrap();
    let launch = fixture(
        &runtime,
        geometry(1, 1, 2, 2, &[1], &[2], &[1]),
        &[1., 0.],
        &[0., 0., 1., 0.],
        &[0., 0., 1., 1.],
    );
    let output = launch.output.clone();
    submit(&runtime, launch);
    // exp(1) in (2.71828,2.71829) bounds sigmoid(1) inside this BF16 cell.
    assert_eq!(bits(&output), [0x3f3b, 0x3f3b]);
}
#[test]
fn d72_uses_every_column_and_visible_nonfinite_score_poisons_only_its_head() {
    let runtime = ReferenceDeviceRuntime::new(config()).unwrap();
    let mut q = vec![0.; 72];
    q[71] = 1.;
    let mut k = vec![0.; 144];
    k[143] = 1.;
    let mut v = vec![0.; 144];
    v[72..].fill(1.);
    let launch = fixture(
        &runtime,
        geometry(1, 1, 2, 72, &[1], &[2], &[1]),
        &q,
        &k,
        &v,
    );
    let output = launch.output.clone();
    submit(&runtime, launch);
    assert_eq!(bits(&output), vec![0x3f3b; 72]);
    q[0] = f32::INFINITY;
    let launch = fixture(
        &runtime,
        geometry(1, 1, 2, 72, &[1], &[2], &[1]),
        &q,
        &k,
        &v,
    );
    let output = launch.output.clone();
    submit(&runtime, launch);
    assert!(bits(&output)
        .into_iter()
        .all(|b| bf16::from_bits(b).is_nan()));
}
#[test]
fn exact_ranges_runtime_binding_and_no_alias_validate_before_writes() {
    let runtime = ReferenceDeviceRuntime::new(config()).unwrap();
    let mut launch = fixture(
        &runtime,
        geometry(1, 1, 2, 2, &[1], &[2], &[1]),
        &[1., 0.],
        &[0., 0., 1., 0.],
        &[0., 0., 1., 1.],
    );
    let saved = launch.output.clone();
    let before = bits(&saved);
    launch.key = launch.query.clone();
    assert!(launch.validate().is_err());
    assert_eq!(bits(&saved), before);
    let mut launch = fixture(
        &runtime,
        geometry(1, 1, 2, 2, &[1], &[2], &[1]),
        &[1., 0.],
        &[0., 0., 1., 0.],
        &[0., 0., 1., 1.],
    );
    launch.value = launch.key.clone();
    launch.validate().unwrap(); // Read/read aliases are permitted.
    launch.output = launch.query.clone();
    assert!(launch.validate().is_err());
    let other = ReferenceDeviceRuntime::new(config()).unwrap();
    let launch = fixture(
        &runtime,
        geometry(1, 1, 2, 2, &[1], &[2], &[1]),
        &[1., 0.],
        &[0., 0., 1., 0.],
        &[0., 0., 1., 1.],
    );
    let mut stream = other.create_stream().unwrap();
    assert!(other
        .submit_entries(
            &mut stream,
            vec![(
                DeviceCommandPhase::Compute,
                Some(0),
                ReferenceDeviceCommand::cross_attention_tensor(launch).unwrap()
            )],
            false
        )
        .is_err());
}
