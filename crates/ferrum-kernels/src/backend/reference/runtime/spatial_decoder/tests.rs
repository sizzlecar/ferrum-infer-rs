use super::super::tests::{buffer, config};
use super::super::{
    DeviceCommandPhase, ReferenceChannelL2NormLaunch, ReferenceCommandKind, ReferenceDeviceCommand,
    ReferenceDeviceRuntime,
};
use super::*;
use ferrum_runtime::{
    AttributeId, CanonicalRational, ChannelL2NormImageBf16Geometry, DeviceLogicalWork,
    DeviceRuntime, ResolvedTensorLayout, ResolvedTensorSpec, SemanticValue, StreamState,
};
use std::collections::BTreeMap;
fn t(d: &[u64]) -> ResolvedTensorSpec {
    ResolvedTensorSpec::new(
        d.to_vec(),
        ElementType::Bf16,
        ResolvedTensorLayout::Contiguous,
    )
    .unwrap()
}
fn data(r: &ReferenceDeviceRuntime, name: &str, bits: &[u16]) -> ReferenceBufferRegion {
    let bytes = bits
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect::<Vec<_>>();
    buffer(r, name, &bytes, 16, ElementType::Bf16)
        .region(0..bytes.len() as u64)
        .unwrap()
}
fn floats(r: &ReferenceDeviceRuntime, name: &str, x: &[f32]) -> ReferenceBufferRegion {
    data(
        r,
        name,
        &x.iter()
            .map(|x| bf16::from_f32(*x).to_bits())
            .collect::<Vec<_>>(),
    )
}
fn blank(r: &ReferenceDeviceRuntime, n: usize) -> ReferenceBufferRegion {
    data(r, "resource.spatial_decoder.output", &vec![0xa5a5; n])
}
fn bits(r: &ReferenceBufferRegion) -> Vec<u16> {
    r.read()
        .chunks_exact(2)
        .map(|v| u16::from_le_bytes([v[0], v[1]]))
        .collect()
}
fn attrs() -> BTreeMap<AttributeId, SemanticValue> {
    BTreeMap::from([(
        AttributeId::new("scale").unwrap(),
        SemanticValue::Rational(CanonicalRational::new(1, 1).unwrap()),
    )])
}
fn attention(
    r: &ReferenceDeviceRuntime,
    n: u64,
    c: u64,
    h: u64,
    w: u64,
    input: &[f32],
) -> ReferenceSpatialDecoderLaunch {
    ReferenceSpatialDecoderLaunch {
        input: floats(r, "resource.spatial_decoder.qkv", input),
        output: blank(r, (n * c * h * w) as usize),
        shape: ReferenceSpatialDecoderGeometry::Attention(
            SpatialAttentionBf16Geometry::new(&t(&[n, 3 * c, h, w]), &t(&[n, c, h, w]), &attrs())
                .unwrap(),
        ),
    }
}
fn gather(
    r: &ReferenceDeviceRuntime,
    input: &[u16],
    offsets: Vec<i64>,
) -> ReferenceSpatialDecoderLaunch {
    let a = BTreeMap::from([
        (
            AttributeId::new("scale_h").unwrap(),
            SemanticValue::Unsigned(2),
        ),
        (
            AttributeId::new("scale_w").unwrap(),
            SemanticValue::Unsigned(2),
        ),
        (
            AttributeId::new("channel_stride").unwrap(),
            SemanticValue::Unsigned(2),
        ),
        (
            AttributeId::new("channel_offsets").unwrap(),
            SemanticValue::Integers(offsets),
        ),
    ]);
    ReferenceSpatialDecoderLaunch {
        input: data(r, "resource.spatial_decoder.gather", input),
        output: blank(r, 16),
        shape: ReferenceSpatialDecoderGeometry::Upsample(
            SpatialUpsampleGatherBf16Geometry::new(&t(&[2, 2, 1, 2]), &t(&[2, 1, 2, 4]), &a)
                .unwrap(),
        ),
    }
}
fn run(r: &ReferenceDeviceRuntime, c: ReferenceDeviceCommand) {
    let mut s = r.create_stream().unwrap();
    r.submit_entries(
        &mut s,
        vec![(DeviceCommandPhase::Compute, Some(0), c)],
        false,
    )
    .unwrap();
}
#[test]
fn nonzero_packed_qkv_is_one_full_spatial_head_and_keeps_batches_separate() {
    let r = ReferenceDeviceRuntime::new(config()).unwrap();
    // Per sample Q=(1,1,0), K0=(1,0,0), K1=(0,1,0), all scores=1.
    // The channel contributions must be added before softmax, not split heads.
    let x = [
        1., 1., 1., 1., 0., 0., 1., 0., 0., 1., 0., 0., 0., 2., 2., 4., 4., 8., 1., 1., 1., 1., 0.,
        0., 1., 0., 0., 1., 0., 0., -2., 0., -4., -2., 8., 12.,
    ];
    let l = attention(&r, 2, 3, 1, 2, &x);
    let out = l.output.clone();
    let c = ReferenceDeviceCommand::spatial_decoder_tensor(l).unwrap();
    assert_eq!(
        c.attribution(0, Some(0), DeviceCommandPhase::Compute)
            .unwrap()
            .logical_work(),
        DeviceLogicalWork::Tensor {
            element_count: 12,
            arithmetic_operations: 132
        }
    );
    run(&r, c);
    assert_eq!(
        bits(&out),
        [
            0x3f80, 0x3f80, 0x4040, 0x4040, 0x40c0, 0x40c0, 0xbf80, 0xbf80, 0xc040, 0xc040, 0x4120,
            0x4120
        ]
    );
    assert_eq!(r.snapshot().spatial_attention_launches, 1);
}
#[test]
fn nonuniform_softmax_and_wide_1152_head_have_independent_expected_outputs() {
    let r = ReferenceDeviceRuntime::new(config()).unwrap();
    let l = attention(&r, 1, 1, 1, 2, &[1., 0., 0., 1., 0., 1.]);
    let out = l.output.clone();
    run(
        &r,
        ReferenceDeviceCommand::spatial_decoder_tensor(l).unwrap(),
    );
    // Logistic(1) is strictly inside the BF16 cell 0x3f3b; q=0 is mean .5.
    assert_eq!(bits(&out), [0x3f3b, 0x3f00]);
    let c = 1152usize;
    let mut x = vec![0.0; 3 * c * 2];
    for p in 0..2 {
        x[p] = 1.;
        x[(c - 1) * 2 + p] = 1.;
    }
    x[c * 2] = 1.;
    x[(2 * c - 1) * 2 + 1] = 1.;
    for ch in 0..c {
        x[(2 * c + ch) * 2] = 2.;
        x[(2 * c + ch) * 2 + 1] = 6.;
    }
    let l = attention(&r, 1, c as u64, 1, 2, &x);
    let out = l.output.clone();
    run(
        &r,
        ReferenceDeviceCommand::spatial_decoder_tensor(l).unwrap(),
    );
    assert!(bits(&out).iter().all(|b| *b == 0x4080));
}
#[test]
fn score_nonfinite_poison_is_per_query_while_value_ieee_is_per_channel() {
    let r = ReferenceDeviceRuntime::new(config()).unwrap();
    let l = attention(&r, 1, 1, 1, 2, &[f32::INFINITY, 0., 1., 1., 2., 4.]);
    let out = l.output.clone();
    run(
        &r,
        ReferenceDeviceCommand::spatial_decoder_tensor(l).unwrap(),
    );
    let y = bits(&out);
    assert!(bf16::from_bits(y[0]).is_nan());
    assert_eq!(y[1], 0x4040);
    let l = attention(
        &r,
        1,
        2,
        1,
        2,
        &[0., 0., 0., 0., 1., 1., 1., 1., f32::NAN, 1., -0., -0.],
    );
    let out = l.output.clone();
    run(
        &r,
        ReferenceDeviceCommand::spatial_decoder_tensor(l).unwrap(),
    );
    let y = bits(&out);
    assert!(bf16::from_bits(y[0]).is_nan() && bf16::from_bits(y[1]).is_nan());
    assert_eq!(&y[2..], &[0x8000, 0x8000]);
}
#[test]
fn upsample_channel_tiles_are_bit_exact_including_nan_payload_and_signed_zero() {
    let r = ReferenceDeviceRuntime::new(config()).unwrap();
    let input = [
        0x8000, 0x7fa1, 0x1234, 0x5678, 0xabcd, 0xef01, 0x7fc3, 0x0000,
    ];
    for (offsets, expected) in [
        (
            vec![0, 0, 1, 1],
            vec![
                0x8000, 0x8000, 0x7fa1, 0x7fa1, 0x1234, 0x1234, 0x5678, 0x5678, 0xabcd, 0xabcd,
                0xef01, 0xef01, 0x7fc3, 0x7fc3, 0, 0,
            ],
        ),
        (
            vec![1, 1, 1, 1],
            vec![
                0x1234, 0x1234, 0x5678, 0x5678, 0x1234, 0x1234, 0x5678, 0x5678, 0x7fc3, 0x7fc3, 0,
                0, 0x7fc3, 0x7fc3, 0, 0,
            ],
        ),
    ] {
        let l = gather(&r, &input, offsets);
        let out = l.output.clone();
        let c = ReferenceDeviceCommand::spatial_decoder_tensor(l).unwrap();
        assert_eq!(
            c.attribution(0, Some(0), DeviceCommandPhase::Compute)
                .unwrap()
                .logical_work(),
            DeviceLogicalWork::Tensor {
                element_count: 16,
                arithmetic_operations: 0
            }
        );
        run(&r, c);
        assert_eq!(bits(&out), expected);
    }
    assert_eq!(r.snapshot().spatial_upsample_gather_launches, 2);
}
#[test]
fn nearest_repeat_preserves_each_channel_and_rectangular_spatial_order() {
    let r = ReferenceDeviceRuntime::new(config()).unwrap();
    let a = BTreeMap::from([
        (
            AttributeId::new("scale_h").unwrap(),
            SemanticValue::Unsigned(2),
        ),
        (
            AttributeId::new("scale_w").unwrap(),
            SemanticValue::Unsigned(2),
        ),
        (
            AttributeId::new("channel_stride").unwrap(),
            SemanticValue::Unsigned(1),
        ),
        (
            AttributeId::new("channel_offsets").unwrap(),
            SemanticValue::Integers(vec![0; 4]),
        ),
    ]);
    let output = blank(&r, 16);
    let launch = ReferenceSpatialDecoderLaunch {
        input: data(&r, "resource.nearest.input", &[1, 2, 3, 4]),
        output: output.clone(),
        shape: ReferenceSpatialDecoderGeometry::Upsample(
            SpatialUpsampleGatherBf16Geometry::new(&t(&[1, 2, 1, 2]), &t(&[1, 2, 2, 4]), &a)
                .unwrap(),
        ),
    };
    run(
        &r,
        ReferenceDeviceCommand::spatial_decoder_tensor(launch).unwrap(),
    );
    assert_eq!(
        bits(&output),
        [1, 1, 2, 2, 1, 1, 2, 2, 3, 3, 4, 4, 3, 3, 4, 4]
    );
}
#[test]
fn image_gamma_uses_the_existing_staged_arithmetic_without_reinterpreting_abi() {
    let r = ReferenceDeviceRuntime::new(config()).unwrap();
    let mut a = attrs();
    a.insert(
        AttributeId::new("epsilon").unwrap(),
        SemanticValue::Rational(CanonicalRational::new(1, 1_000_000_000_000).unwrap()),
    );
    let g = ChannelL2NormImageBf16Geometry::new(
        &t(&[1, 1, 1, 3]),
        &t(&[1, 1, 1]),
        &t(&[1, 1, 1, 3]),
        &a,
    )
    .unwrap();
    let output = blank(&r, 3);
    let l = ReferenceChannelL2NormLaunch {
        input: floats(&r, "resource.image_norm.input", &[-1., 0., 1.]),
        gamma: floats(&r, "resource.image_norm.gamma", &[2.]),
        output: output.clone(),
        shape: g.arithmetic_geometry(),
    };
    run(
        &r,
        ReferenceDeviceCommand::channel_l2_norm_tensor(l).unwrap(),
    );
    assert_eq!(bits(&output), [0xc000, 0, 0x4000]);
}
#[test]
fn whole_batch_preflight_rejects_foreign_runtime_and_alias_before_any_write() {
    let r = ReferenceDeviceRuntime::new(config()).unwrap();
    let other = ReferenceDeviceRuntime::new(config()).unwrap();
    let first = attention(&r, 1, 1, 1, 1, &[1., 1., 3.]);
    let out = first.output.clone();
    let mut invalid = attention(&r, 1, 1, 1, 1, &[1., 1., 5.]);
    invalid.input = floats(&other, "resource.foreign.qkv", &[1., 1., 5.]);
    let mut stream = r.create_stream().unwrap();
    assert!(r
        .submit_entries(
            &mut stream,
            vec![
                (
                    DeviceCommandPhase::Compute,
                    Some(0),
                    ReferenceDeviceCommand::spatial_decoder_tensor(first).unwrap()
                ),
                (
                    DeviceCommandPhase::Compute,
                    Some(1),
                    ReferenceDeviceCommand::spatial_decoder_tensor(invalid).unwrap()
                )
            ],
            false
        )
        .is_err());
    assert_eq!(bits(&out), [0xa5a5]);
    assert_eq!(r.stream_state(&stream), StreamState::Ready);
    assert_eq!(r.snapshot().spatial_attention_launches, 0);
    let mut l = attention(&r, 1, 1, 1, 1, &[1., 1., 3.]);
    l.output = l.input.clone();
    l.output.length_bytes = 2;
    assert!(ReferenceDeviceCommand::spatial_decoder_tensor(l).is_err());
    // An internal corruption after constructor validation is checked at submit.
    let l = attention(&r, 1, 1, 1, 1, &[1., 1., 3.]);
    let mut cmd = ReferenceDeviceCommand::spatial_decoder_tensor(l).unwrap();
    if let ReferenceCommandKind::SpatialDecoder { launch } = &mut cmd.kind {
        launch.output.length_bytes = 1;
    }
    assert!(r
        .submit_entries(
            &mut stream,
            vec![(DeviceCommandPhase::Compute, Some(0), cmd)],
            false
        )
        .is_err());
}
