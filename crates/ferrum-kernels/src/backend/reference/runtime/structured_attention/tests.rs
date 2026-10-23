use super::super::tests::{buffer, config};
use super::super::{
    DeviceCommandPhase, DeviceRuntime, ReferenceCommandKind, ReferenceDeviceCommand,
    ReferenceDeviceRuntime, StreamState,
};
use super::*;
use ferrum_runtime::{
    AttributeId, CanonicalRational, DeviceLogicalWork, ResolvedTensorLayout, ResolvedTensorSpec,
    SemanticValue,
};
use std::collections::BTreeMap;
fn encoded(x: &[f32]) -> Vec<u8> {
    x.iter()
        .flat_map(|x| bf16::from_f32(*x).to_bits().to_le_bytes())
        .collect()
}
fn bits(x: &ReferenceBufferRegion) -> Vec<u16> {
    x.read()
        .chunks_exact(2)
        .map(|b| u16::from_le_bytes([b[0], b[1]]))
        .collect()
}
fn geometry(
    rows: u64,
    columns: u64,
    s: u64,
    d: u64,
    segments: &[i64],
    bidir: &[i64],
    counts: &[i64],
    runs: &[i64],
    valid: &[i64],
) -> StructuredAttentionBf16Geometry {
    let t = ResolvedTensorSpec::new(
        vec![rows, columns],
        ElementType::Bf16,
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
    StructuredAttentionBf16Geometry::new(&t, &t, &t, &t, &a).unwrap()
}
fn fixture(
    runtime: &ReferenceDeviceRuntime,
    g: StructuredAttentionBf16Geometry,
    q: &[f32],
    k: &[f32],
    v: &[f32],
) -> ReferenceStructuredAttentionLaunch {
    let regions: [ReferenceBufferRegion; 3] = std::array::from_fn(|i| {
        let x = encoded([q, k, v][i]);
        buffer(
            runtime,
            &format!("resource.attention.input{i}"),
            &x,
            16,
            ElementType::Bf16,
        )
        .region(0..x.len() as u64)
        .unwrap()
    });
    let bytes = usize::try_from(g.element_count() * 2).unwrap();
    let output = buffer(
        runtime,
        "resource.attention.y",
        &vec![0xa5; bytes],
        16,
        ElementType::Bf16,
    )
    .region(0..bytes as u64)
    .unwrap();
    let [query, key, value] = regions;
    ReferenceStructuredAttentionLaunch {
        query,
        key,
        value,
        output,
        geometry: g,
    }
}
fn small(runtime: &ReferenceDeviceRuntime) -> ReferenceStructuredAttentionLaunch {
    fixture(
        runtime,
        geometry(2, 2, 2, 2, &[2], &[1], &[1], &[2], &[1]),
        &[1., 0., 1., 0.],
        &[0., 0., 1., 0.],
        &[0., 0., 1., 1.],
    )
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
fn uniform_multibatch_heads_obey_distinct_blocks_internal_padding_and_query_visibility() {
    let runtime = ReferenceDeviceRuntime::new(config()).unwrap();
    let g = geometry(
        20,
        4,
        10,
        2,
        &[2, 3, 2, 3],
        &[0, 1, 1, 1],
        &[3, 5],
        &[3, 1, 6, 3, 1, 4, 1, 1],
        &[1, 0, 1, 1, 0, 1, 0, 1],
    );
    let q = vec![0.; 80];
    let mut k = q.clone();
    let mut v = Vec::new();
    let values = [0., 1., 2., 0., 4., 5., 6., 7., 8., 12.];
    for b in 0..2 {
        for (key, &x) in values.iter().enumerate() {
            if key == 3 || (b == 1 && key == 8) {
                v.extend([f32::NAN, f32::INFINITY, f32::NEG_INFINITY, f32::NAN]);
                k[(b * 10 + key) * 4..(b * 10 + key + 1) * 4].fill(f32::NAN);
            } else {
                v.extend([x, x, 2. * x, -x]);
            }
        }
    }
    let work = g.logical_arithmetic_operations();
    let launch = fixture(&runtime, g, &q, &k, &v);
    let output = launch.output.clone();
    let command = ReferenceDeviceCommand::structured_attention_tensor(launch).unwrap();
    let attribution = command
        .attribution(0, Some(0), DeviceCommandPhase::Compute)
        .unwrap();
    assert_eq!(attribution.token_count(), None);
    assert_eq!(
        attribution.logical_work(),
        DeviceLogicalWork::Tensor {
            element_count: 80,
            arithmetic_operations: work
        }
    );
    let before = runtime.snapshot();
    submit(&runtime, command);
    let mut expected = Vec::new();
    for b in 0..2 {
        for x in [
            0.,
            0.5,
            1.75,
            1.75,
            1.75,
            3.,
            3.,
            if b == 0 { 5. } else { 4.625 },
            if b == 0 { 5. } else { 4.625 },
            if b == 0 { 5. } else { 4.625 },
        ] {
            expected.extend([x, x, 2. * x, -x]);
        }
    }
    // The first -0 value becomes +0 through the final +0 additions; compare
    // numerical zero separately, while every nonzero BF16 value is exact.
    let actual = bits(&output);
    for (a, e) in actual.into_iter().zip(expected) {
        if e == 0. {
            assert_eq!(bf16::from_bits(a).to_f32(), 0.);
        } else {
            assert_eq!(a, bf16::from_f32(e).to_bits());
        }
    }
    let after = runtime.snapshot();
    assert_eq!(
        after.structured_attention_launches,
        before.structured_attention_launches + 1
    );
    assert_eq!(after.allocations, before.allocations);
}
#[test]
fn nonuniform_two_key_softmax_has_independent_analytic_bf16_cell() {
    let runtime = ReferenceDeviceRuntime::new(config()).unwrap();
    let launch = small(&runtime);
    let output = launch.output.clone();
    submit(
        &runtime,
        ReferenceDeviceCommand::structured_attention_tensor(launch).unwrap(),
    );
    // e in (2.71828,2.71829) proves e/(1+e) lies strictly inside
    // the RNE BF16 cell around 187/256, between 373/512 and 375/512.
    // This fixture is not a global libm expf accuracy qualification.
    assert_eq!(bits(&output), [0x3f3b; 4]);
}
#[test]
fn empty_rows_skip_all_nonfinite_inputs_and_visible_nonfinite_scores_poison_queries() {
    let runtime = ReferenceDeviceRuntime::new(config()).unwrap();
    let g = geometry(2, 2, 2, 2, &[2], &[1], &[1], &[2], &[0]);
    assert_eq!(g.logical_arithmetic_operations(), 0);
    let launch = fixture(
        &runtime,
        g,
        &[f32::NAN; 4],
        &[f32::INFINITY; 4],
        &[f32::NAN; 4],
    );
    let output = launch.output.clone();
    submit(
        &runtime,
        ReferenceDeviceCommand::structured_attention_tensor(launch).unwrap(),
    );
    assert_eq!(bits(&output), [0; 4]);
    for score in [f32::NAN, f32::INFINITY] {
        let launch = small(&runtime);
        launch.key.write(&encoded(&[score, 0., 1., 0.]));
        let output = launch.output.clone();
        submit(
            &runtime,
            ReferenceDeviceCommand::structured_attention_tensor(launch).unwrap(),
        );
        assert!(bits(&output)
            .into_iter()
            .all(|x| bf16::from_bits(x).is_nan()));
    }
    let launch = small(&runtime);
    launch.value.write(&encoded(&[f32::INFINITY, 0., 1., 1.]));
    let output = launch.output.clone();
    submit(
        &runtime,
        ReferenceDeviceCommand::structured_attention_tensor(launch).unwrap(),
    );
    let result = bits(&output);
    assert_eq!(result, [0x7f80, 0x3f3b, 0x7f80, 0x3f3b]);
}
#[test]
fn exact_ranges_dtype_scalar_alignment_and_output_alias_are_checked() {
    let runtime = ReferenceDeviceRuntime::new(config()).unwrap();
    for mutation in 0..7 {
        let mut launch = small(&runtime);
        let output = launch.output.clone();
        match mutation {
            0 => launch.query.length_bytes -= 2,
            1 => launch.key.element_type = ElementType::F16,
            2 => launch.value.offset_bytes += 1,
            3 => launch.output.length_bytes -= 2,
            4 => launch.output = launch.query.clone(),
            5 => launch.output = launch.key.clone(),
            6 => launch.output = launch.value.clone(),
            _ => unreachable!(),
        };
        assert!(ReferenceDeviceCommand::structured_attention_tensor(launch).is_err());
        assert_eq!(output.read(), [0xa5; 8]);
    }
}
#[test]
fn readonly_aliases_and_disjoint_offsets_share_one_allocation_without_clobber() {
    let runtime = ReferenceDeviceRuntime::new(config()).unwrap();
    let all = buffer(
        &runtime,
        "resource.attention.shared",
        &[0xa5; 32],
        16,
        ElementType::Bf16,
    );
    let input = all.region(2..6).unwrap();
    input.write(&encoded(&[1., 2.]));
    let output = all.region(22..26).unwrap();
    let before = all.region(0..32).unwrap().read();
    let launch = ReferenceStructuredAttentionLaunch {
        query: input.clone(),
        key: input.clone(),
        value: input,
        output: output.clone(),
        geometry: geometry(1, 2, 1, 2, &[1], &[1], &[1], &[1], &[1]),
    };
    submit(
        &runtime,
        ReferenceDeviceCommand::structured_attention_tensor(launch).unwrap(),
    );
    assert_eq!(bits(&output), [0x3f80, 0x4000]);
    let after = all.region(0..32).unwrap().read();
    for i in 0..32 {
        if !(22..26).contains(&i) {
            assert_eq!(after[i], before[i]);
        }
    }
}
#[test]
fn whole_batch_preflight_rejects_late_foreign_or_short_value_before_any_write() {
    let runtime = ReferenceDeviceRuntime::new(config()).unwrap();
    let foreign = ReferenceDeviceRuntime::new(config()).unwrap();
    for foreign_value in [false, true] {
        let first = small(&runtime);
        let first_output = first.output.clone();
        let second = small(&runtime);
        let second_output = second.output.clone();
        let mut second = ReferenceDeviceCommand::structured_attention_tensor(second).unwrap();
        let ReferenceCommandKind::StructuredAttention { launch } = &mut second.kind else {
            unreachable!()
        };
        if foreign_value {
            launch.value = small(&foreign).value;
        } else {
            launch.value.length_bytes -= 2;
        }
        let sentinel = buffer(
            &runtime,
            "resource.attention.sentinel",
            &[0xa5; 4],
            16,
            ElementType::U8,
        );
        let commands = vec![
            runtime.encode_zero(&sentinel, 0, 4).unwrap(),
            ReferenceDeviceCommand::structured_attention_tensor(first).unwrap(),
            second,
        ];
        let mut stream = runtime.create_stream().unwrap();
        let before = runtime.snapshot();
        assert!(runtime
            .submit_entries(
                &mut stream,
                commands
                    .into_iter()
                    .map(|c| (DeviceCommandPhase::Compute, Some(0), c))
                    .collect(),
                false
            )
            .is_err());
        let after = runtime.snapshot();
        assert_eq!(
            (
                after.submissions,
                after.commands,
                after.structured_attention_launches
            ),
            (
                before.submissions,
                before.commands,
                before.structured_attention_launches
            )
        );
        assert_eq!(runtime.stream_state(&stream), StreamState::Ready);
        assert_eq!(sentinel.region(0..4).unwrap().read(), [0xa5; 4]);
        assert_eq!(first_output.read(), [0xa5; 8]);
        assert_eq!(second_output.read(), [0xa5; 8]);
    }
}
