use super::super::tests::{buffer, config};
use super::super::{
    DeviceCommandPhase, ReferenceCommandKind, ReferenceDeviceCommand, ReferenceDeviceRuntime,
    StreamState,
};
use super::*;
use ferrum_runtime::{
    AttributeId, CanonicalRational, DeviceLogicalWork, DeviceRuntime, ResolvedTensorLayout,
    ResolvedTensorSpec, SemanticValue,
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
fn data(r: &ReferenceDeviceRuntime, name: &str, x: &[f32]) -> ReferenceBufferRegion {
    let bytes = x
        .iter()
        .flat_map(|f| bf16::from_f32(*f).to_bits().to_le_bytes())
        .collect::<Vec<_>>();
    buffer(r, name, &bytes, 16, ElementType::Bf16)
        .region(0..bytes.len() as u64)
        .unwrap()
}
fn blank(r: &ReferenceDeviceRuntime, count: usize) -> ReferenceBufferRegion {
    buffer(
        r,
        "resource.spatial.output",
        &vec![0xa5; count * 2],
        16,
        ElementType::Bf16,
    )
    .region(0..count as u64 * 2)
    .unwrap()
}
fn bits(r: &ReferenceBufferRegion) -> Vec<u16> {
    r.read()
        .chunks_exact(2)
        .map(|v| u16::from_le_bytes([v[0], v[1]]))
        .collect()
}
fn conv_attrs(groups: u64, padding: Vec<i64>) -> BTreeMap<AttributeId, SemanticValue> {
    BTreeMap::from([
        (
            AttributeId::new("padding").unwrap(),
            SemanticValue::Integers(padding),
        ),
        (
            AttributeId::new("stride").unwrap(),
            SemanticValue::Integers(vec![1, 1]),
        ),
        (
            AttributeId::new("dilation").unwrap(),
            SemanticValue::Integers(vec![1, 1]),
        ),
        (
            AttributeId::new("groups").unwrap(),
            SemanticValue::Unsigned(groups),
        ),
        (
            AttributeId::new("padding_mode").unwrap(),
            SemanticValue::Text("explicit_zero".into()),
        ),
    ])
}
fn norm_attrs(scale: CanonicalRational) -> BTreeMap<AttributeId, SemanticValue> {
    BTreeMap::from([
        (
            AttributeId::new("epsilon").unwrap(),
            SemanticValue::Rational(CanonicalRational::new(1, 1_000_000_000_000).unwrap()),
        ),
        (
            AttributeId::new("scale").unwrap(),
            SemanticValue::Rational(scale),
        ),
    ])
}
fn conv(
    r: &ReferenceDeviceRuntime,
    xd: &[u64],
    x: &[f32],
    wd: &[u64],
    w: &[f32],
    bias: &[f32],
    yd: &[u64],
    attrs: &BTreeMap<AttributeId, SemanticValue>,
) -> ReferenceConv2dLaunch {
    ReferenceConv2dLaunch {
        input: data(r, "resource.spatial.x", x),
        weight: data(r, "resource.spatial.w", w),
        bias: data(r, "resource.spatial.bias", bias),
        output: blank(r, yd.iter().product::<u64>() as usize),
        shape: Conv2dBf16Geometry::new(&t(xd), &t(wd), &t(&[bias.len() as u64]), &t(yd), attrs)
            .unwrap(),
    }
}
fn norm(
    r: &ReferenceDeviceRuntime,
    dims: &[u64],
    x: &[f32],
    gamma: &[f32],
    scale: CanonicalRational,
) -> ReferenceChannelL2NormLaunch {
    ReferenceChannelL2NormLaunch {
        input: data(r, "resource.spatial.x", x),
        gamma: data(r, "resource.spatial.gamma", gamma),
        output: blank(r, x.len()),
        shape: ChannelL2NormBf16Geometry::new(
            &t(dims),
            &t(&[gamma.len() as u64, 1, 1, 1]),
            &t(dims),
            &norm_attrs(scale),
        )
        .unwrap(),
    }
}
fn run(r: &ReferenceDeviceRuntime, c: ReferenceDeviceCommand) {
    let mut stream = r.create_stream().unwrap();
    r.submit_entries(
        &mut stream,
        vec![(DeviceCommandPhase::Compute, Some(0), c)],
        false,
    )
    .unwrap();
}
#[test]
fn convolution_bias_has_a_required_pre_add_bf16_boundary() {
    let r = ReferenceDeviceRuntime::new(config()).unwrap();
    let l = conv(
        &r,
        &[1, 1, 1, 1],
        &[129.0 / 128.0],
        &[1, 1, 1, 1],
        &[129.0 / 128.0],
        &[-130.0 / 128.0],
        &[1, 1, 1, 1],
        &conv_attrs(1, vec![0; 4]),
    );
    let output = l.output.clone();
    let c = ReferenceDeviceCommand::conv2d_tensor(l).unwrap();
    let attr = c
        .attribution(0, Some(0), DeviceCommandPhase::Compute)
        .unwrap();
    assert_eq!(
        attr.logical_work(),
        DeviceLogicalWork::Tensor {
            element_count: 1,
            arithmetic_operations: 2
        }
    );
    // Exact product is 16641/16384. B(product)=130/128, so bias cancels.
    // Fusing bias before B would leave 1/16384 = BF16 0x3880.
    let before = r.snapshot();
    run(&r, c);
    assert_eq!(bits(&output), [0]);
    assert_eq!(r.snapshot().conv2d_launches, before.conv2d_launches + 1);
    assert_eq!(r.snapshot().allocations, before.allocations);
}
#[test]
fn convolution_preserves_nchw_batch_groups_and_oi_hw_indexing() {
    let r = ReferenceDeviceRuntime::new(config()).unwrap();
    let x = [
        1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., -1., -2., -3., -4., -5., -6., 1., 3.,
        6., 2., 6., 12.,
    ];
    let l = conv(
        &r,
        &[2, 2, 2, 3],
        &x,
        &[2, 1, 1, 2],
        &[1., 2., -1., 1.],
        &[1., -1.],
        &[2, 2, 2, 2],
        &conv_attrs(2, vec![0; 4]),
    );
    let y = l.output.clone();
    run(&r, ReferenceDeviceCommand::conv2d_tensor(l).unwrap());
    let expected = [
        6., 9., 15., 18., 0., 0., 0., 0., -4., -7., -13., -16., 1., 2., 3., 5.,
    ];
    assert_eq!(bits(&y), expected.map(|x| bf16::from_f32(x).to_bits()));
}
#[test]
fn virtual_zero_padding_keeps_ieee_zero_times_infinity_and_kernel_orientation() {
    let r = ReferenceDeviceRuntime::new(config()).unwrap();
    let l = conv(
        &r,
        &[1, 1, 1, 2],
        &[2., 3.],
        &[1, 1, 1, 2],
        &[5., 7.],
        &[0.],
        &[1, 1, 1, 3],
        &conv_attrs(1, vec![0, 0, 1, 1]),
    );
    let y = l.output.clone();
    run(&r, ReferenceDeviceCommand::conv2d_tensor(l).unwrap());
    assert_eq!(
        bits(&y),
        [14., 31., 15.].map(|v| bf16::from_f32(v).to_bits())
    );
    let l = conv(
        &r,
        &[1, 1, 1, 1],
        &[1.],
        &[1, 1, 1, 3],
        &[f32::INFINITY, 1., 0.],
        &[0.],
        &[1, 1, 1, 1],
        &conv_attrs(1, vec![0, 0, 1, 1]),
    );
    let y = l.output.clone();
    run(&r, ReferenceDeviceCommand::conv2d_tensor(l).unwrap());
    assert!(bf16::from_bits(bits(&y)[0]).is_nan());
}
#[test]
fn channel_l2_first_rounding_and_channel_strided_access_are_independent() {
    let r = ReferenceDeviceRuntime::new(config()).unwrap();
    let l = norm(
        &r,
        &[1, 2, 1, 2],
        &[3., 0., 4., 5.],
        &[131.0 / 128.0, 1.],
        CanonicalRational::new(1, 1).unwrap(),
    );
    let y = l.output.clone();
    let c = ReferenceDeviceCommand::channel_l2_norm_tensor(l).unwrap();
    assert_eq!(
        c.attribution(0, Some(0), DeviceCommandPhase::Compute)
            .unwrap()
            .logical_work(),
        DeviceLogicalWork::Tensor {
            element_count: 4,
            arithmetic_operations: 24
        }
    );
    run(&r, c);
    // L=5; B(3/5)=77/128. Its product with 131/128 rounds to
    // 158/256 (3f1e), whereas omitting that B yields 157/256 (3f1d).
    assert_eq!(bits(&y), [0x3f1e, 0, 0x3f4d, 0x3f80]);
}
#[test]
fn channel_l2_scale_gamma_and_positive_zero_epilogue_keep_their_boundaries() {
    let r = ReferenceDeviceRuntime::new(config()).unwrap();
    let l = norm(
        &r,
        &[2, 2, 1, 1],
        &[3., 4., -0., 0.],
        &[2., -1.],
        CanonicalRational::new(5, 1).unwrap(),
    );
    let y = l.output.clone();
    run(
        &r,
        ReferenceDeviceCommand::channel_l2_norm_tensor(l).unwrap(),
    );
    assert_eq!(bits(&y), [0x40c0, 0xc080, 0, 0]);
    let l = norm(
        &r,
        &[1, 2, 1, 2],
        &[f32::INFINITY, f32::NAN, 1., 1.],
        &[1., 1.],
        CanonicalRational::new(1, 1).unwrap(),
    );
    let y = l.output.clone();
    run(
        &r,
        ReferenceDeviceCommand::channel_l2_norm_tensor(l).unwrap(),
    );
    let y = bits(&y);
    assert!(bf16::from_bits(y[0]).is_nan());
    assert!(bf16::from_bits(y[1]).is_nan());
    assert_eq!(y[2], 0);
    assert!(bf16::from_bits(y[3]).is_nan());
}
#[test]
fn whole_submission_preflight_rejects_alias_or_short_region_before_any_spatial_launch() {
    let r = ReferenceDeviceRuntime::new(config()).unwrap();
    let good = norm(
        &r,
        &[1, 1, 1, 1],
        &[1.],
        &[1.],
        CanonicalRational::new(1, 1).unwrap(),
    );
    let output = good.output.clone();
    let mut bad = norm(
        &r,
        &[1, 1, 1, 1],
        &[1.],
        &[1.],
        CanonicalRational::new(1, 1).unwrap(),
    );
    bad.output = bad.input.clone();
    assert!(bad.validate().is_err());
    bad.output = blank(&r, 1);
    bad.output.length_bytes = 0;
    let c = ReferenceDeviceCommand {
        operation: "vnext_channel_l2_norm_bf16",
        logical_work: Some(bad.logical_work().unwrap()),
        compute_dispatch_count: 1,
        transfer_command_count: 0,
        kind: ReferenceCommandKind::ChannelL2Norm { launch: bad },
    };
    let mut stream = r.create_stream().unwrap();
    let before = r.snapshot();
    assert!(r
        .submit_entries(
            &mut stream,
            vec![
                (
                    DeviceCommandPhase::Compute,
                    Some(0),
                    ReferenceDeviceCommand::channel_l2_norm_tensor(good).unwrap()
                ),
                (DeviceCommandPhase::Compute, Some(1), c)
            ],
            false
        )
        .is_err());
    assert_eq!(bits(&output), [0xa5a5]);
    assert_eq!(
        r.snapshot().channel_l2_norm_launches,
        before.channel_l2_norm_launches
    );
    assert_eq!(r.stream_state(&stream), StreamState::Ready);
}
