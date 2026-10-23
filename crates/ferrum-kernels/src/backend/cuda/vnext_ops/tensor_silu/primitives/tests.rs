//! Metadata and admission boundaries; no CUDA allocation or library query.
use super::*;
use ferrum_runtime::{
    AliasPolicy, BufferUsage, CanonicalRational, ProgramValueId, ResolvedTensorLayout,
    ResolvedTensorSpec, ResolvedValueStorage, ResourceId, TensorAccess,
};
fn value(
    role: ResolvedValueRole,
    ordinal: u32,
    dims: &[u64],
    dtype: ElementType,
) -> ResolvedValueBinding {
    let role_name = if role == ResolvedValueRole::Input {
        "input"
    } else {
        "output"
    };
    let name = format!("primitive.{role_name}.{ordinal}");
    let tensor =
        ResolvedTensorSpec::new(dims.to_vec(), dtype, ResolvedTensorLayout::Contiguous).unwrap();
    let storage = ResolvedValueStorage::single(
        ResourceId::new(format!("resource.{name}")).unwrap(),
        0,
        tensor.minimum_storage_bytes().unwrap(),
        dtype,
    )
    .unwrap();
    ResolvedValueBinding::new(
        ProgramValueId::new(name).unwrap(),
        role,
        ordinal,
        tensor,
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
fn values(inputs: &[(&[u64], ElementType)], output: &[u64]) -> Vec<ResolvedValueBinding> {
    let mut result = inputs
        .iter()
        .enumerate()
        .map(|(i, (d, t))| value(ResolvedValueRole::Input, i as u32, d, *t))
        .collect::<Vec<_>>();
    result.push(value(
        ResolvedValueRole::Output,
        0,
        output,
        ElementType::Bf16,
    ));
    result
}
fn attrs(values: &[(&str, SemanticValue)]) -> BTreeMap<AttributeId, SemanticValue> {
    values
        .iter()
        .map(|(s, v)| (AttributeId::new(*s).unwrap(), v.clone()))
        .collect()
}
#[test]
fn affine_and_patch_admit_exact_fp32_output_scratch_for_rectangular_geometry() {
    let a = attrs(&[
        ("in_features", SemanticValue::Unsigned(3)),
        ("out_features", SemanticValue::Unsigned(4)),
    ]);
    let linear = checked_geometry(
        TensorPrimitiveBf16::AffineLinear,
        &values(
            &[
                (&[2, 3], ElementType::Bf16),
                (&[4, 3], ElementType::Bf16),
                (&[4], ElementType::Bf16),
            ],
            &[2, 4],
        ),
        &a,
    )
    .unwrap();
    assert_eq!(linear.input_bytes, [12, 24, 8]);
    assert_eq!(linear.output_bytes, 16);
    assert_eq!(scratch_bytes(&linear).unwrap(), 32);
    assert_eq!(arithmetic(&linear).unwrap(), 56);
    let patch = checked_geometry(
        TensorPrimitiveBf16::PatchProjection,
        &values(
            &[
                (&[2, 3], ElementType::Bf16),
                (&[4, 3, 1, 1, 1], ElementType::Bf16),
                (&[4], ElementType::Bf16),
            ],
            &[2, 4],
        ),
        &BTreeMap::new(),
    )
    .unwrap();
    assert_eq!(patch.inner, 3);
    assert_eq!(scratch_bytes(&patch).unwrap(), 32);
    assert!(checked_geometry(
        TensorPrimitiveBf16::PatchProjection,
        &values(
            &[
                (&[2, 3], ElementType::Bf16),
                (&[4, 3], ElementType::Bf16),
                (&[4], ElementType::Bf16)
            ],
            &[2, 4]
        ),
        &BTreeMap::new()
    )
    .is_err());
}
#[test]
fn half_rotary_accepts_actual_f32_phases_and_rejects_bf16_reinterpretation() {
    let a = attrs(&[("head_dim", SemanticValue::Unsigned(72))]);
    let good = values(
        &[
            (&[3, 144], ElementType::Bf16),
            (&[3, 72], ElementType::F32),
            (&[3, 72], ElementType::F32),
        ],
        &[3, 144],
    );
    let g = checked_geometry(TensorPrimitiveBf16::HalfRotary, &good, &a).unwrap();
    assert_eq!(g.input_bytes, [864, 864, 864]);
    assert_eq!(input_dtype(g.kind, 1), good[1].tensor().element_type());
    assert_eq!(input_dtype(g.kind, 2), good[2].tensor().element_type());
    assert_eq!(scratch_bytes(&g).unwrap(), 0);
    assert_eq!(arithmetic(&g).unwrap(), 1512);
    let bad = values(
        &[
            (&[3, 144], ElementType::Bf16),
            (&[3, 72], ElementType::Bf16),
            (&[3, 72], ElementType::F32),
        ],
        &[3, 144],
    );
    assert!(checked_geometry(TensorPrimitiveBf16::HalfRotary, &bad, &a).is_err());
}
#[test]
fn segmented_d72_native_metadata_is_bounded_and_does_not_merge_segments() {
    let ports = values(&[(&[3, 144], ElementType::Bf16); 3], &[3, 144]);
    let mut a = attrs(&[
        ("head_dim", SemanticValue::Unsigned(72)),
        (
            "scale",
            SemanticValue::Rational(CanonicalRational::new(1, 8).unwrap()),
        ),
        ("segment_lengths", SemanticValue::Integers(vec![2, 1])),
    ]);
    let g = checked_geometry(TensorPrimitiveBf16::SegmentedAttention, &ports, &a).unwrap();
    let native = Native::new(&g);
    assert_eq!((native.head_dim, native.count, native.padding), (72, 2, 0));
    assert_eq!(&native.starts[..2], [0, 2]);
    assert_eq!(&native.lengths[..2], [2, 1]);
    assert!(native.starts[2..]
        .iter()
        .chain(&native.lengths[2..])
        .all(|&n| n == 0));
    assert_eq!(scratch_bytes(&g).unwrap(), 0);
    // Four queries in the length-2 segment, two in the singleton segment;
    // D72 uses a 128-lane dot tree, without merging the segments.
    assert_eq!(arithmetic(&g).unwrap(), 4 * 695 + 2 * 273);
    a.insert(
        AttributeId::new("segment_lengths").unwrap(),
        SemanticValue::Integers(vec![2]),
    );
    assert!(checked_geometry(TensorPrimitiveBf16::SegmentedAttention, &ports, &a).is_err());
}

#[test]
fn work_counts_reduction_padding_and_only_scattered_rows() {
    let norm = checked_geometry(
        TensorPrimitiveBf16::AffineLayerNorm,
        &values(
            &[
                (&[2, 3], ElementType::Bf16),
                (&[3], ElementType::Bf16),
                (&[3], ElementType::Bf16),
            ],
            &[2, 3],
        ),
        &attrs(&[
            ("hidden_size", SemanticValue::Unsigned(3)),
            (
                "epsilon",
                SemanticValue::Rational(CanonicalRational::new(1, 1000).unwrap()),
            ),
        ]),
    )
    .unwrap();
    assert_eq!(arithmetic(&norm).unwrap(), 1078);
    let scatter = checked_geometry(
        TensorPrimitiveBf16::RowScatterAdd,
        &values(
            &[(&[6, 3], ElementType::Bf16), (&[3, 3], ElementType::Bf16)],
            &[6, 3],
        ),
        &attrs(&[
            ("destination_starts", SemanticValue::Integers(vec![1, 4])),
            ("run_lengths", SemanticValue::Integers(vec![1, 2])),
        ]),
    )
    .unwrap();
    assert_eq!(arithmetic(&scatter).unwrap(), 9);
}
#[test]
fn affine_fp32_scratch_is_checked_independently_of_bf16_binding_extent() {
    let dim = i32::MAX as u64;
    let output_bytes = dim * dim * 2;
    if usize::try_from(output_bytes).is_ok() && output_bytes <= isize::MAX as u64 {
        let a = attrs(&[
            ("in_features", SemanticValue::Unsigned(1)),
            ("out_features", SemanticValue::Unsigned(dim)),
        ]);
        let g = checked_geometry(
            TensorPrimitiveBf16::AffineLinear,
            &values(
                &[
                    (&[dim, 1], ElementType::Bf16),
                    (&[dim, 1], ElementType::Bf16),
                    (&[dim], ElementType::Bf16),
                ],
                &[dim, dim],
            ),
            &a,
        )
        .unwrap();
        assert!(scratch_bytes(&g).is_err());
    }
}
