use super::super::tests::{buffer, config};
use super::super::ReferenceDeviceRuntime;
use super::*;
use ferrum_runtime::*;
use std::collections::BTreeMap;

fn tensor(dims: &[u64], dtype: ElementType) -> ResolvedTensorSpec {
    ResolvedTensorSpec::new(dims.to_vec(), dtype, ResolvedTensorLayout::Contiguous).unwrap()
}
fn bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|v| bf16::from_f32(*v).to_bits().to_le_bytes())
        .collect()
}
fn raw(bits: &[u16]) -> Vec<u8> {
    bits.iter().flat_map(|v| v.to_le_bytes()).collect()
}
fn value(
    name: &str,
    role: ResolvedValueRole,
    ordinal: u32,
    t: ResolvedTensorSpec,
) -> ResolvedValueBinding {
    let storage = ResolvedValueStorage::single(
        ResourceId::new(format!("resource.{name}")).unwrap(),
        0,
        t.minimum_storage_bytes().unwrap(),
        t.element_type(),
    )
    .unwrap();
    ResolvedValueBinding::new(
        ProgramValueId::new(name).unwrap(),
        role,
        ordinal,
        t,
        if role == ResolvedValueRole::Input {
            TensorAccess::Read
        } else {
            TensorAccess::Write
        },
        AliasPolicy::NoAlias,
        BufferUsage::Activations,
        None,
        storage,
    )
    .unwrap()
}
fn launch(
    r: &ReferenceDeviceRuntime,
    kind: TensorPrimitiveBf16,
    inputs: &[(&[u64], ElementType, &[u8])],
    output: &[u64],
    attrs: BTreeMap<AttributeId, SemanticValue>,
) -> ReferenceTensorPrimitiveLaunch {
    let mut bindings = Vec::new();
    let mut regions = Vec::new();
    for (i, (dims, dtype, data)) in inputs.iter().enumerate() {
        bindings.push(value(
            &format!("input.{i}"),
            ResolvedValueRole::Input,
            i as u32,
            tensor(dims, *dtype),
        ));
        regions.push(
            buffer(
                r,
                &format!("resource.primitive.input.{i}"),
                data,
                16,
                *dtype,
            )
            .region(0..data.len() as u64)
            .unwrap(),
        );
    }
    let output = tensor(output, ElementType::Bf16);
    let n = output.minimum_storage_bytes().unwrap();
    bindings.push(value("output", ResolvedValueRole::Output, 0, output));
    let shape = TensorPrimitiveBf16Geometry::new(kind, &bindings, &attrs).unwrap();
    let output = buffer(
        r,
        "resource.primitive.output",
        &vec![0xa5; n as usize],
        16,
        ElementType::Bf16,
    )
    .region(0..n)
    .unwrap();
    ReferenceTensorPrimitiveLaunch {
        inputs: regions,
        output,
        shape,
    }
}
fn attr(values: &[(&str, SemanticValue)]) -> BTreeMap<AttributeId, SemanticValue> {
    values
        .iter()
        .map(|(k, v)| (AttributeId::new(*k).unwrap(), v.clone()))
        .collect()
}
fn run(r: &ReferenceDeviceRuntime, l: ReferenceTensorPrimitiveLaunch) -> Vec<u16> {
    let output = l.output.clone();
    let command = ReferenceDeviceCommand::tensor_primitive(l).unwrap();
    let mut stream = r.create_stream().unwrap();
    r.submit_entries(
        &mut stream,
        vec![(DeviceCommandPhase::Compute, Some(0), command)],
        false,
    )
    .unwrap();
    output
        .read()
        .chunks_exact(2)
        .map(|b| u16::from_le_bytes([b[0], b[1]]))
        .collect()
}
#[test]
fn primitive_affine_and_patch_keep_their_distinct_bias_rounding_boundaries() {
    let r = ReferenceDeviceRuntime::new(config()).unwrap();
    let x = raw(&[0x3f81]);
    let w = raw(&[0x3f81]);
    let b = raw(&[0xbf82]);
    for (kind, weight_shape, attrs) in [
        (
            TensorPrimitiveBf16::AffineLinear,
            vec![1, 1],
            attr(&[
                ("in_features", SemanticValue::Unsigned(1)),
                ("out_features", SemanticValue::Unsigned(1)),
            ]),
        ),
        (
            TensorPrimitiveBf16::PatchProjection,
            vec![1, 1, 1, 1, 1],
            BTreeMap::new(),
        ),
    ] {
        let l = launch(
            &r,
            kind,
            &[
                (&[1, 1], ElementType::Bf16, &x),
                (&weight_shape, ElementType::Bf16, &w),
                (&[1], ElementType::Bf16, &b),
            ],
            &[1, 1],
            attrs,
        );
        // Affine has one final rounding: exact 2^-14. PatchProjection
        // represents convolution's stored BF16 result before the bias add.
        assert_eq!(
            run(&r, l),
            [if kind == TensorPrimitiveBf16::PatchProjection {
                0
            } else {
                0x3880
            }]
        );
    }
}
#[test]
fn primitive_affine_population_norm_supports_non_power_of_two_width() {
    let r = ReferenceDeviceRuntime::new(config()).unwrap();
    let x = bytes(&[-1., 0., 1.]);
    let gamma = bytes(&[1., 2., 3.]);
    let beta = bytes(&[1., 0., -1.]);
    let l = launch(
        &r,
        TensorPrimitiveBf16::AffineLayerNorm,
        &[
            (&[1, 3], ElementType::Bf16, &x),
            (&[3], ElementType::Bf16, &gamma),
            (&[3], ElementType::Bf16, &beta),
        ],
        &[1, 3],
        attr(&[
            ("hidden_size", SemanticValue::Unsigned(3)),
            (
                "epsilon",
                SemanticValue::Rational(CanonicalRational::new(1, 3).unwrap()),
            ),
        ]),
    );
    assert_eq!(run(&r, l), [0, 0, 0x4000]);
}
#[test]
fn primitive_gelu_preserves_ieee_special_value_arithmetic() {
    let r = ReferenceDeviceRuntime::new(config()).unwrap();
    let x = raw(&[0x8000, 0, 0x7f80, 0xff80, 0x7fc1]);
    let actual = run(
        &r,
        launch(
            &r,
            TensorPrimitiveBf16::GeluErf,
            &[(&[1, 5], ElementType::Bf16, &x)],
            &[1, 5],
            BTreeMap::new(),
        ),
    );
    assert_eq!(&actual[..3], &[0x8000, 0, 0x7f80]);
    assert!(actual[3..].iter().all(|&v| bf16::from_bits(v).is_nan()));
}
#[test]
fn primitive_float_rotary_has_no_intermediate_bf16_cast_and_uses_each_rows_phases() {
    let r = ReferenceDeviceRuntime::new(config()).unwrap();
    let x = raw(&[0x3f81, 0x3f82, 0x3f80, 0x4000]);
    let cos = [129.0_f32 / 128.0, 0.0, 0.0, 0.0]
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect::<Vec<_>>();
    let sin = [1.0_f32, 0.0, 1.0, 1.0]
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect::<Vec<_>>();
    let l = launch(
        &r,
        TensorPrimitiveBf16::HalfRotary,
        &[
            (&[2, 2], ElementType::Bf16, &x),
            (&[2, 2], ElementType::F32, &cos),
            (&[2, 2], ElementType::F32, &sin),
        ],
        &[2, 2],
        attr(&[("head_dim", SemanticValue::Unsigned(2))]),
    );
    assert_eq!(run(&r, l), [0x3880, 0, 0xc000, 0x3f80]);
}
#[test]
fn primitive_scatter_and_reshape_preserve_untouched_nan_payload_and_negative_zero() {
    let r = ReferenceDeviceRuntime::new(config()).unwrap();
    let base = raw(&[0x7f81, 0x8000, 0x4000, 0x4040, 0x7fc3, 0xff80]);
    let update = bytes(&[1., -1.]);
    let actual = run(
        &r,
        launch(
            &r,
            TensorPrimitiveBf16::RowScatterAdd,
            &[
                (&[3, 2], ElementType::Bf16, &base),
                (&[1, 2], ElementType::Bf16, &update),
            ],
            &[3, 2],
            attr(&[
                ("destination_starts", SemanticValue::Integers(vec![1])),
                ("run_lengths", SemanticValue::Integers(vec![1])),
            ]),
        ),
    );
    assert_eq!(actual, [0x7f81, 0x8000, 0x4040, 0x4000, 0x7fc3, 0xff80]);
    let copied = run(
        &r,
        launch(
            &r,
            TensorPrimitiveBf16::ReshapeCopy,
            &[(&[3, 2], ElementType::Bf16, &base)],
            &[1, 6],
            attr(&[
                ("rows", SemanticValue::Unsigned(1)),
                ("columns", SemanticValue::Unsigned(6)),
            ]),
        ),
    );
    assert_eq!(raw(&copied), base);
}
#[test]
fn primitive_segmented_attention_supports_d72_and_isolates_poisoned_segments() {
    let r = ReferenceDeviceRuntime::new(config()).unwrap();
    let mut q = vec![0.0f32; 3 * 144];
    let mut k = q.clone();
    let mut v = q.clone();
    for row in 0..3 {
        for head in 0..2 {
            q[row * 144 + head * 72] = 1.;
            k[row * 144 + head * 72] = if row == 2 { f32::NAN } else { 1. };
            for d in 0..72 {
                v[row * 144 + head * 72 + d] =
                    (2 * (row + 1)) as f32 * if head == 0 { 1.0 } else { -1.0 };
            }
        }
    }
    let (q, k, v) = (bytes(&q), bytes(&k), bytes(&v));
    let l = launch(
        &r,
        TensorPrimitiveBf16::SegmentedAttention,
        &[
            (&[3, 144], ElementType::Bf16, &q),
            (&[3, 144], ElementType::Bf16, &k),
            (&[3, 144], ElementType::Bf16, &v),
        ],
        &[3, 144],
        attr(&[
            ("head_dim", SemanticValue::Unsigned(72)),
            (
                "scale",
                SemanticValue::Rational(CanonicalRational::new(1, 1).unwrap()),
            ),
            ("segment_lengths", SemanticValue::Integers(vec![2, 1])),
        ]),
    );
    let actual = run(&r, l);
    for row in 0..2 {
        for head in 0..2 {
            assert!(actual[row * 144 + head * 72..row * 144 + (head + 1) * 72]
                .iter()
                .all(|&v| v == if head == 0 { 0x4040 } else { 0xc040 }));
        }
    }
    assert!(actual[288..].iter().all(|&v| bf16::from_bits(v).is_nan()));
}
#[test]
fn primitive_whole_batch_preflight_rejects_foreign_regions_and_alias_before_any_write() {
    let r = ReferenceDeviceRuntime::new(config()).unwrap();
    let other = ReferenceDeviceRuntime::new(config()).unwrap();
    let x = bytes(&[1., 2.]);
    let build = |runtime: &ReferenceDeviceRuntime| {
        launch(
            runtime,
            TensorPrimitiveBf16::GeluErf,
            &[(&[1, 2], ElementType::Bf16, &x)],
            &[1, 2],
            BTreeMap::new(),
        )
    };
    let first = build(&r);
    let output = first.output.clone();
    let mut alias = build(&r);
    alias.output = alias.inputs[0].clone();
    assert!(ReferenceDeviceCommand::tensor_primitive(alias).is_err());
    let mut stream = r.create_stream().unwrap();
    assert!(r
        .submit_entries(
            &mut stream,
            vec![
                (
                    DeviceCommandPhase::Compute,
                    Some(0),
                    ReferenceDeviceCommand::tensor_primitive(first).unwrap()
                ),
                (
                    DeviceCommandPhase::Compute,
                    Some(1),
                    ReferenceDeviceCommand::tensor_primitive(build(&other)).unwrap()
                ),
            ],
            false
        )
        .is_err());
    assert_eq!(output.read(), [0xa5; 4]);
}
