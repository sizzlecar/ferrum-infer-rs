use super::*;
use ferrum_execution::component::TensorComponentPolicy;
use transforms::{fixture::Fixture, run_case, spec};
#[path = "spatial/oracle.rs"]
mod oracle;
pub(super) use oracle::check;

fn data(spec: &ProgramTensorSpec, seed: usize) -> Vec<f32> {
    (0..spec.dimensions.iter().product::<u64>() as usize)
        .map(|i| rounds(((i * 13 + seed * 7) % 31) as f32 / 64.0 - 0.25))
        .collect()
}
fn attrs(entries: Vec<(&str, SemanticValue)>) -> BTreeMap<AttributeId, SemanticValue> {
    entries
        .into_iter()
        .map(|(name, value)| (id(name), value))
        .collect()
}
pub(super) fn verify(
    runtime: &Arc<MetalDeviceRuntime>,
    registry: &OperationRuntimeRegistry<MetalDeviceRuntime>,
    catalog: &CapabilityCatalog,
    host: &HostMemoryAccount,
) {
    use MetalTensorOperation::*;
    let policy = TensorComponentPolicy::new(
        CAPACITY,
        runtime
            .descriptor()
            .dynamic_storage_profiles
            .iter()
            .copied()
            .collect(),
    )
    .unwrap();
    let run = |kind,
               inputs: Vec<ProgramTensorSpec>,
               output: ProgramTensorSpec,
               attributes: BTreeMap<AttributeId, SemanticValue>,
               values: Vec<Vec<f32>>| {
        let mut fixture = Fixture::new(kind, inputs, attributes);
        fixture.output = Some(output.clone());
        run_case(
            runtime,
            registry,
            catalog,
            &policy,
            host,
            &fixture,
            &values,
            |actual| {
                check(
                    kind,
                    &fixture.attributes,
                    &fixture
                        .inputs
                        .iter()
                        .zip(&values)
                        .map(|(s, v)| (s, v.as_slice()))
                        .collect::<Vec<_>>(),
                    &output,
                    actual,
                );
            },
        );
    };
    // Grouped cross-correlation, non-square kernels, dilation, asymmetric
    // padding, and matrix tiles crossing 32 spatial / 64 channel boundaries.
    for (batch, ic, oc, h, w, kh, kw, groups, pad, stride, dilation) in [
        (2, 4, 6, 5, 7, 3, 2, 2, [1, 2, 0, 1], [2, 1], [1, 2]),
        (1, 3, 65, 3, 11, 1, 1, 1, [0, 0, 0, 0], [1, 1], [1, 1]),
        (1, 2, 3, 7, 9, 3, 3, 1, [2, 1, 1, 0], [1, 2], [2, 1]),
        (1, 1152, 3, 1, 1, 1, 1, 1, [0, 0, 0, 0], [1, 1], [1, 1]),
        (2, 6, 66, 7, 13, 3, 3, 2, [1, 2, 0, 3], [1, 1], [1, 1]),
        (1, 5, 65, 5, 13, 3, 3, 1, [0, 0, 0, 0], [1, 1], [1, 1]),
        (2, 6, 66, 7, 13, 1, 1, 2, [1, 2, 0, 3], [1, 1], [1, 1]),
        (2, 4, 6, 1, 2, 3, 3, 2, [3, 2, 2, 4], [1, 1], [1, 1]),
    ] {
        let inputs = vec![
            spec(vec![batch, ic, h, w], ElementType::Bf16),
            spec(vec![oc, ic / groups, kh, kw], ElementType::Bf16),
            spec(vec![oc], ElementType::Bf16),
        ];
        let output = spec(
            vec![
                batch,
                oc,
                (h + pad[0] + pad[1] - (kh - 1) * dilation[0] - 1) / stride[0] + 1,
                (w + pad[2] + pad[3] - (kw - 1) * dilation[1] - 1) / stride[1] + 1,
            ],
            ElementType::Bf16,
        );
        for mode in ["explicit_zero", "implicit_zero"] {
            run(
                Conv2d,
                inputs.clone(),
                output.clone(),
                attrs(vec![
                    (
                        "padding",
                        SemanticValue::Integers(pad.map(|x| x as i64).to_vec()),
                    ),
                    (
                        "stride",
                        SemanticValue::Integers(stride.map(|x| x as i64).to_vec()),
                    ),
                    (
                        "dilation",
                        SemanticValue::Integers(dilation.map(|x| x as i64).to_vec()),
                    ),
                    ("groups", SemanticValue::Unsigned(groups)),
                    ("padding_mode", SemanticValue::Text(mode.into())),
                ]),
                inputs.iter().enumerate().map(|(i, s)| data(s, i)).collect(),
            );
        }
    }
    // This result changes if bias is fused before the convolution BF16 boundary.
    run(
        Conv2d,
        vec![
            spec(vec![1, 1, 1, 1], ElementType::Bf16),
            spec(vec![1, 1, 1, 1], ElementType::Bf16),
            spec(vec![1], ElementType::Bf16),
        ],
        spec(vec![1, 1, 1, 1], ElementType::Bf16),
        attrs(vec![
            ("padding", SemanticValue::Integers(vec![0; 4])),
            ("stride", SemanticValue::Integers(vec![1; 2])),
            ("dilation", SemanticValue::Integers(vec![1; 2])),
            ("groups", SemanticValue::Unsigned(1)),
            ("padding_mode", SemanticValue::Text("implicit_zero".into())),
        ]),
        vec![vec![1.0078125], vec![1.5], vec![-1.5]],
    );
    verify_convolution_padding_special_values(runtime, registry, catalog, &policy, host);
    for kind in [ChannelL2Norm, ChannelL2NormImage] {
        for c in [1, 3, 1152] {
            let input = spec(vec![2, c, 3, 5], ElementType::Bf16);
            let gamma = spec(
                if kind == ChannelL2Norm {
                    vec![c, 1, 1, 1]
                } else {
                    vec![c, 1, 1]
                },
                ElementType::Bf16,
            );
            let mut values = data(&input, 1);
            for ch in 0..c as usize {
                values[ch * 15] = -0.0;
                values[ch * 15 + 1] = rounds(1e-15);
            }
            values[2] = f32::INFINITY;
            values[3] = f32::NAN;
            run(
                kind,
                vec![input.clone(), gamma.clone()],
                input,
                attrs(vec![
                    (
                        "epsilon",
                        SemanticValue::Rational(
                            CanonicalRational::new(1, 1_000_000_000_000).unwrap(),
                        ),
                    ),
                    (
                        "scale",
                        SemanticValue::Rational(CanonicalRational::new(3, 2).unwrap()),
                    ),
                ]),
                vec![values, data(&gamma, 3)],
            );
        }
    }
    for c in [1, 7, 257, 1152, 2048] {
        let input = spec(vec![2, 3 * c, 3, 11], ElementType::Bf16);
        run(
            SpatialAttention,
            vec![input.clone()],
            spec(vec![2, c, 3, 11], ElementType::Bf16),
            attrs(vec![(
                "scale",
                SemanticValue::Rational(CanonicalRational::new(1, 32).unwrap()),
            )]),
            vec![data(&input, 2)],
        );
    }
    for role in 0..3 {
        let input = spec(vec![2, 3 * 17, 1, 33], ElementType::Bf16);
        let mut values = data(&input, 2);
        values[(role * 17 + 3) * 33 + 32] = if role == 2 { f32::INFINITY } else { f32::NAN };
        run(
            SpatialAttention,
            vec![input],
            spec(vec![2, 17, 1, 33], ElementType::Bf16),
            attrs(vec![(
                "scale",
                SemanticValue::Rational(CanonicalRational::new(1, 8).unwrap()),
            )]),
            vec![values],
        );
    }
    for (sh, sw, stride, offsets, ic) in [
        (8, 8, 64, (0..64).rev().collect::<Vec<i64>>(), 128),
        (2, 2, 1, vec![0; 4], 2),
    ] {
        let input = spec(vec![2, ic, 2, 3], ElementType::Bf16);
        let words = [
            0, 0x8000, 0x0001, 0x007f, 0x7fc1, 0xffd5, 0x7f80, 0xff80, 0x3f80,
        ];
        let values = (0..input.dimensions.iter().product::<u64>())
            .map(|i| bf16::from_bits(words[i as usize % words.len()]).to_f32())
            .collect();
        run(
            SpatialUpsampleGather,
            vec![input],
            spec(vec![2, 2, 2 * sh, 3 * sw], ElementType::Bf16),
            attrs(vec![
                ("scale_h", SemanticValue::Unsigned(sh)),
                ("scale_w", SemanticValue::Unsigned(sw)),
                ("channel_stride", SemanticValue::Unsigned(stride)),
                ("channel_offsets", SemanticValue::Integers(offsets)),
            ]),
            vec![values],
        );
    }
    for (oc, sh, sw, frames) in [
        (3, 2, 2, 2),
        (6, 2, 2, 2),
        (24, 2, 2, 2),
        (4, 2, 2, 1),
        (3, 1, 2, 1),
    ] {
        let input = spec(vec![2, 3, 4, 6], ElementType::Bf16);
        let mut values = data(&input, 3);
        values[0] = -0.0;
        values[1] = f32::INFINITY;
        values[2] = f32::NAN;
        run(
            SpatialGroupMean,
            vec![input],
            spec(vec![2, oc, 4 / sh, 6 / sw], ElementType::Bf16),
            attrs(vec![
                ("scale_h", SemanticValue::Unsigned(sh)),
                ("scale_w", SemanticValue::Unsigned(sw)),
                ("frame_factor", SemanticValue::Unsigned(frames)),
            ]),
            vec![values],
        );
    }
    eprintln!("Metal spatial operators: grouped/padded/strided convolution, staged L2, C=1152/2048 attention, exceptional values, exact upsample, single-frame group mean and retirement passed");
}

fn verify_convolution_padding_special_values(
    runtime: &Arc<MetalDeviceRuntime>,
    registry: &OperationRuntimeRegistry<MetalDeviceRuntime>,
    catalog: &CapabilityCatalog,
    policy: &TensorComponentPolicy,
    host: &HostMemoryAccount,
) {
    for mode in ["explicit_zero", "implicit_zero"] {
        for special in [f32::INFINITY, f32::NAN] {
            let mut fixture = Fixture::new(
                MetalTensorOperation::Conv2d,
                vec![
                    spec(vec![1, 1, 1, 1], ElementType::Bf16),
                    spec(vec![1, 1, 3, 3], ElementType::Bf16),
                    spec(vec![1], ElementType::Bf16),
                ],
                attrs(vec![
                    ("padding", SemanticValue::Integers(vec![3; 4])),
                    ("stride", SemanticValue::Integers(vec![1; 2])),
                    ("dilation", SemanticValue::Integers(vec![1; 2])),
                    ("groups", SemanticValue::Unsigned(1)),
                    ("padding_mode", SemanticValue::Text(mode.into())),
                ]),
            );
            fixture.output = Some(spec(vec![1, 1, 5, 5], ElementType::Bf16));
            let mut weights = vec![0.0; 9];
            weights[8] = special;
            run_case(
                runtime,
                registry,
                catalog,
                policy,
                host,
                &fixture,
                &[vec![2.0], weights, vec![0.0]],
                |actual| {
                    assert_eq!(actual.len(), 25);
                    for (i, value) in actual.iter().enumerate() {
                        // Only output [1,1] sees the input through tap [2,2].
                        // Every other position must still multiply +0 by the
                        // nonfinite weight, including fully padded windows.
                        if i == 6 && special.is_infinite() {
                            assert_eq!(*value, f32::INFINITY, "{mode}: visible infinity");
                        } else {
                            assert!(value.is_nan(), "{mode}: padded product at {i}");
                        }
                    }
                },
            );
        }
    }
}
