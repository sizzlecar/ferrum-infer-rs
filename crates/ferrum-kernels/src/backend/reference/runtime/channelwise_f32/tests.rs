use super::super::tests::{buffer, config};
use super::*;
use ferrum_runtime::{AttributeId, ResolvedTensorLayout, ResolvedTensorSpec, SemanticValue};
use std::collections::BTreeMap;
fn spec(shape: &[u64]) -> ResolvedTensorSpec {
    ResolvedTensorSpec::new(
        shape.to_vec(),
        ElementType::F32,
        ResolvedTensorLayout::Contiguous,
    )
    .unwrap()
}
fn launch(
    runtime: &ReferenceDeviceRuntime,
    kind: F32ChannelwiseOp,
) -> ReferenceF32ChannelwiseLaunch {
    let x = spec(&[
        2,
        if matches!(
            kind,
            F32ChannelwiseOp::GroupedColumnSlice | F32ChannelwiseOp::GroupedColumnMean
        ) {
            12
        } else {
            4
        },
    ]);
    let y = spec(&[2, 4]);
    let vector = spec(&[4]);
    let attrs = if kind == F32ChannelwiseOp::GroupedColumnSlice {
        [("group_width", 6), ("column_start", 2), ("column_count", 2)]
            .into_iter()
            .map(|(k, v)| (AttributeId::new(k).unwrap(), SemanticValue::Unsigned(v)))
            .collect()
    } else if kind == F32ChannelwiseOp::GroupedColumnMean {
        BTreeMap::from([(
            AttributeId::new("group_width").unwrap(),
            SemanticValue::Unsigned(4),
        )])
    } else {
        BTreeMap::new()
    };
    let inputs = match kind {
        F32ChannelwiseOp::ScaledResidual | F32ChannelwiseOp::ScaledResidualSeparate => {
            vec![&x, &x, &vector]
        }
        F32ChannelwiseOp::BiasAdd => vec![&x, &vector],
        _ => vec![&x],
    };
    let geometry = F32ChannelwiseGeometry::new(kind, &inputs, &y, &attrs).unwrap();
    let inputs = inputs
        .iter()
        .enumerate()
        .map(|(role, spec)| {
            let data: Vec<_> = (0..spec.minimum_storage_bytes().unwrap() / 4)
                .map(|i| (i + role as u64) as f32)
                .flat_map(f32::to_le_bytes)
                .collect();
            buffer(
                runtime,
                &format!("resource.channelwise.input{role}"),
                &data,
                16,
                ElementType::F32,
            )
            .region(0..data.len() as u64)
            .unwrap()
        })
        .collect();
    let output = buffer(
        runtime,
        "resource.channelwise.y",
        &[0xa5; 32],
        16,
        ElementType::F32,
    )
    .region(0..32)
    .unwrap();
    ReferenceF32ChannelwiseLaunch {
        inputs,
        output,
        geometry,
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
fn grouped_columns_preserve_raw_payload_bits_across_rows_and_heads() {
    let runtime = ReferenceDeviceRuntime::new(config()).unwrap();
    let mut l = launch(&runtime, F32ChannelwiseOp::GroupedColumnSlice);
    let mut bits: Vec<u32> = (0..24).map(|i| 0x3f800000 + i).collect();
    for (i, value) in [
        (2, 0x80000000),
        (3, 0x7f800001),
        (8, 0x00000001),
        (9, 0xffc12345),
        (14, 0x7f800000),
        (15, 0xff800000),
    ] {
        bits[i] = value;
    }
    let bytes: Vec<_> = bits.iter().flat_map(|v| v.to_le_bytes()).collect();
    l.inputs[0] = buffer(
        &runtime,
        "resource.channelwise.bits",
        &bytes,
        16,
        ElementType::F32,
    )
    .region(0..bytes.len() as u64)
    .unwrap();
    let output = l.output.clone();
    submit(
        &runtime,
        ReferenceDeviceCommand::channelwise_f32_tensor(l).unwrap(),
    );
    let got: Vec<u32> = output
        .read()
        .chunks_exact(4)
        .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    assert_eq!(
        got,
        [bits[2], bits[3], bits[8], bits[9], bits[14], bits[15], bits[20], bits[21]]
    );
    drop(output);
    assert_eq!(runtime.snapshot().live_allocations, 0);
}
#[test]
fn grouped_mean_and_adaptive_bins_preserve_rows_and_two_distinct_reductions() {
    let runtime = ReferenceDeviceRuntime::new(config()).unwrap();
    let l = launch(&runtime, F32ChannelwiseOp::GroupedColumnMean);
    let output = l.output.clone();
    submit(
        &runtime,
        ReferenceDeviceCommand::channelwise_f32_tensor(l).unwrap(),
    );
    let got: Vec<f32> = output
        .read()
        .chunks_exact(4)
        .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
        .collect();
    assert_eq!(got, [4., 5., 6., 7., 16., 17., 18., 19.]);
    // Non-divisible bins overlap; upsampling repeats or averages adjacent
    // values. These are independent expected values, not the index formula.
    for (columns, expected) in [
        (3, vec![2., 6., 10., -2., -6., -10.]),
        (
            7,
            vec![
                0., 2., 4., 6., 8., 10., 12., 0., -2., -4., -6., -8., -10., -12.,
            ],
        ),
    ] {
        let data: Vec<u8> = [0f32, 4., 8., 12., 0., -4., -8., -12.]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect();
        let input = buffer(&runtime, "resource.pool.x", &data, 16, ElementType::F32)
            .region(0..data.len() as u64)
            .unwrap();
        let out = buffer(
            &runtime,
            "resource.pool.y",
            &vec![0; expected.len() * 4],
            16,
            ElementType::F32,
        )
        .region(0..expected.len() as u64 * 4)
        .unwrap();
        let geometry = F32ChannelwiseGeometry::new(
            F32ChannelwiseOp::AdaptiveAvgPool,
            &[&spec(&[2, 4])],
            &spec(&[2, columns]),
            &BTreeMap::new(),
        )
        .unwrap();
        submit(
            &runtime,
            ReferenceDeviceCommand::channelwise_f32_tensor(ReferenceF32ChannelwiseLaunch {
                inputs: vec![input],
                output: out.clone(),
                geometry,
            })
            .unwrap(),
        );
        let values: Vec<f32> = out
            .read()
            .chunks_exact(4)
            .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
            .collect();
        assert_eq!(values, expected);
    }
    assert!(F32ChannelwiseGeometry::new(
        F32ChannelwiseOp::GroupedColumnMean,
        &[&spec(&[2, 13])],
        &spec(&[2, 4]),
        &BTreeMap::from([(
            AttributeId::new("group_width").unwrap(),
            SemanticValue::Unsigned(4)
        )])
    )
    .is_err());
    drop(output);
    assert_eq!(runtime.snapshot().live_allocations, 0);
}
#[test]
fn exact_ranges_aliases_and_foreign_late_scale_fail_before_any_output_write() {
    let runtime = ReferenceDeviceRuntime::new(config()).unwrap();
    for kind in F32ChannelwiseOp::ALL {
        for mutation in 0..5 {
            let mut l = launch(&runtime, kind);
            let output = l.output.clone();
            match mutation {
                0 => l.inputs[0].element_type = ElementType::Bf16,
                1 => l.inputs[0].length_bytes -= 4,
                2 => l.output.length_bytes -= 4,
                3 => l.output = l.inputs[0].clone(),
                4 => {
                    let length = l.inputs[0].length_bytes;
                    l.inputs[0] = buffer(
                        &runtime,
                        "resource.channelwise.misaligned",
                        &vec![0; length + 4],
                        16,
                        ElementType::F32,
                    )
                    .region(1..length as u64 + 1)
                    .unwrap();
                }
                _ => unreachable!(),
            }
            assert!(ReferenceDeviceCommand::channelwise_f32_tensor(l).is_err());
            assert_eq!(output.read(), [0xa5; 32]);
        }
    }
    let foreign = ReferenceDeviceRuntime::new(config()).unwrap();
    let first = launch(&runtime, F32ChannelwiseOp::BiasAdd);
    let output = first.output.clone();
    let mut second = launch(&runtime, F32ChannelwiseOp::ScaledResidual);
    second.inputs[2] = launch(&foreign, F32ChannelwiseOp::ScaledResidual)
        .inputs
        .pop()
        .unwrap();
    let mut stream = runtime.create_stream().unwrap();
    let before = runtime.snapshot();
    assert!(runtime
        .submit_entries(
            &mut stream,
            vec![
                (
                    DeviceCommandPhase::Compute,
                    Some(0),
                    ReferenceDeviceCommand::channelwise_f32_tensor(first).unwrap()
                ),
                (
                    DeviceCommandPhase::Compute,
                    Some(1),
                    ReferenceDeviceCommand::channelwise_f32_tensor(second).unwrap()
                ),
            ],
            false
        )
        .is_err());
    assert_eq!(output.read(), [0xa5; 32]);
    assert_eq!(runtime.snapshot().submissions, before.submissions);
    drop(output);
    assert_eq!(runtime.snapshot().live_allocations, 0);
    assert_eq!(foreign.snapshot().live_allocations, 0);
}

#[test]
fn separate_residual_rounds_product_before_addition_including_overflow_and_underflow() {
    let runtime = ReferenceDeviceRuntime::new(config()).unwrap();
    let base = [
        -f32::MAX,
        -1.0,
        -0.0,
        f32::from_bits(1),
        -f32::MAX,
        -1.0,
        -0.0,
        f32::from_bits(1),
    ];
    let branch = [
        f32::MAX,
        f32::from_bits(0x3f800001),
        -0.0,
        f32::from_bits(1),
        f32::MAX,
        f32::from_bits(0x3f800001),
        -0.0,
        f32::from_bits(1),
    ];
    let scale = [2.0, f32::from_bits(0x3f7ffffe), 1.0, 0.5];
    let mut outputs = Vec::new();
    for kind in [
        F32ChannelwiseOp::ScaledResidualSeparate,
        F32ChannelwiseOp::ScaledResidual,
    ] {
        let mut l = launch(&runtime, kind);
        for (i, values) in [&base[..], &branch[..], &scale[..]].into_iter().enumerate() {
            let bytes: Vec<_> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
            l.inputs[i] = buffer(
                &runtime,
                &format!("resource.channelwise.witness{i}"),
                &bytes,
                16,
                ElementType::F32,
            )
            .region(0..bytes.len() as u64)
            .unwrap();
        }
        let output = l.output.clone();
        submit(
            &runtime,
            ReferenceDeviceCommand::channelwise_f32_tensor(l).unwrap(),
        );
        outputs.push(
            output
                .read()
                .chunks_exact(4)
                .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
                .collect::<Vec<_>>(),
        );
    }
    let expected: Vec<_> = (0..8)
        .map(|i| {
            let product = (f64::from(branch[i]) * f64::from(scale[i % 4])) as f32;
            (f64::from(base[i]) + f64::from(product)) as f32
        })
        .map(f32::to_bits)
        .collect();
    assert_eq!(outputs[0], expected);
    assert_eq!(
        outputs[0][..4],
        [f32::INFINITY.to_bits(), 0, (-0.0f32).to_bits(), 1]
    );
    assert_eq!(outputs[1][0], f32::MAX.to_bits());
    assert_ne!(outputs[1][1], outputs[0][1]);
    assert_eq!(outputs[1][3], 2);
    assert_eq!(runtime.snapshot().live_allocations, 0);
}
