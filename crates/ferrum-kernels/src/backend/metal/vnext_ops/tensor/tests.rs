//! Metadata and capacity checks; native numerical evidence lives in models tests.
use super::*;
use ferrum_runtime::{
    AliasPolicy, BufferUsage, CanonicalRational, ProgramValueId, ResolvedTensorSpec,
    ResolvedValueStorage, ResourceId, TensorAccess,
};

pub(super) fn value(
    name: &str,
    role: ResolvedValueRole,
    ordinal: u32,
    dims: &[u64],
    element: ElementType,
) -> ResolvedValueBinding {
    let spec =
        ResolvedTensorSpec::new(dims.to_vec(), element, ResolvedTensorLayout::Contiguous).unwrap();
    let storage = ResolvedValueStorage::single(
        ResourceId::new(format!("resource.{name}")).unwrap(),
        0,
        spec.minimum_storage_bytes().unwrap(),
        element,
    )
    .unwrap();
    ResolvedValueBinding::new(
        ProgramValueId::new(name).unwrap(),
        role,
        ordinal,
        spec,
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
fn norm_values(
    kind: MetalTensorOperation,
    width: u64,
    dtype: ElementType,
) -> Vec<ResolvedValueBinding> {
    let columns = if kind == MetalTensorOperation::HeadRmsNorm {
        3 * width
    } else {
        width
    };
    let mut values = vec![value(
        "x",
        ResolvedValueRole::Input,
        0,
        &[2, columns],
        dtype,
    )];
    if kind != MetalTensorOperation::LayerNorm {
        values.push(value(
            "gamma",
            ResolvedValueRole::Input,
            1,
            &[width],
            ElementType::Bf16,
        ));
    }
    values.push(value(
        "y",
        ResolvedValueRole::Output,
        0,
        &[2, columns],
        ElementType::Bf16,
    ));
    values
}
fn norm_attrs(kind: MetalTensorOperation, width: u64) -> BTreeMap<AttributeId, SemanticValue> {
    let mut attrs = BTreeMap::from([(
        AttributeId::new("epsilon").unwrap(),
        SemanticValue::Rational(CanonicalRational::new(1, 1_000_000).unwrap()),
    )]);
    if kind != MetalTensorOperation::ZeroCenteredRmsNorm {
        attrs.insert(
            AttributeId::new(if kind == MetalTensorOperation::LayerNorm {
                "hidden_size"
            } else {
                "head_dim"
            })
            .unwrap(),
            SemanticValue::Unsigned(width),
        );
    }
    attrs
}

#[test]
fn normalization_admits_exact_aligned_threadgroup_capacity_and_rejects_one_byte_short() {
    for kind in [
        MetalTensorOperation::LayerNorm,
        MetalTensorOperation::HeadRmsNorm,
        MetalTensorOperation::ZeroCenteredRmsNorm,
    ] {
        // Narrow allocations still require Metal's 16-byte dynamic alignment;
        // a larger declared device can support widths beyond an 8K tile.
        for (width, required) in [(1, 16), (6, 32), (4096, 16384), (8193, 65536)] {
            let values = norm_values(kind, width, ElementType::Bf16);
            let attrs = norm_attrs(kind, width);
            assert!(norm::geometry(kind, &values, &attrs, required).is_ok());
            assert!(norm::geometry(kind, &values, &attrs, required - 1).is_err());
        }
        let attrs = norm_attrs(kind, 6);
        assert!(norm::geometry(kind, &norm_values(kind, 6, ElementType::F16), &attrs, 32).is_err());
        let mut extra = attrs.clone();
        extra.insert(
            AttributeId::new("unexpected").unwrap(),
            SemanticValue::Unsigned(1),
        );
        assert!(
            norm::geometry(kind, &norm_values(kind, 6, ElementType::Bf16), &extra, 32).is_err()
        );
    }
}

#[test]
fn rotary_frequency_dtypes_and_shapes_cannot_be_interchanged() {
    use MetalTensorOperation::*;
    let attrs = BTreeMap::from([(
        AttributeId::new("head_dim").unwrap(),
        SemanticValue::Unsigned(6),
    )]);
    for kind in [AdjacentRope, HalfSplitRope] {
        let frequency_type = if kind == AdjacentRope {
            ElementType::F32
        } else {
            ElementType::Bf16
        };
        let shape: &[u64] = if kind == AdjacentRope {
            &[3, 3, 2]
        } else {
            &[6, 6]
        };
        let mut values = vec![
            value(
                "x",
                ResolvedValueRole::Input,
                0,
                &[6, 12],
                ElementType::Bf16,
            ),
            value(
                "frequency",
                ResolvedValueRole::Input,
                1,
                shape,
                frequency_type,
            ),
        ];
        if kind == HalfSplitRope {
            values.push(value(
                "sine",
                ResolvedValueRole::Input,
                2,
                shape,
                frequency_type,
            ));
        }
        values.push(value(
            "y",
            ResolvedValueRole::Output,
            0,
            &[6, 12],
            ElementType::Bf16,
        ));
        assert!(rope::geometry(kind, &values, &attrs).is_ok());
        let wrong_type = if kind == AdjacentRope {
            ElementType::Bf16
        } else {
            ElementType::F32
        };
        values[1] = value("frequency", ResolvedValueRole::Input, 1, shape, wrong_type);
        assert!(rope::geometry(kind, &values, &attrs).is_err());
        values[1] = value(
            "frequency",
            ResolvedValueRole::Input,
            1,
            &[6, 3],
            frequency_type,
        );
        assert!(rope::geometry(kind, &values, &attrs).is_err());
    }
}

#[test]
fn attention_validates_gqa_mapping_and_exact_threadgroup_capacity() {
    use MetalTensorOperation::*;
    for kind in [StructuredAttention, StructuredGqaAttention] {
        for width in [2, 128, 256] {
            let mut attrs = BTreeMap::from([
                (
                    AttributeId::new("sequence_length").unwrap(),
                    SemanticValue::Unsigned(3),
                ),
                (
                    AttributeId::new("head_dim").unwrap(),
                    SemanticValue::Unsigned(width),
                ),
                (
                    AttributeId::new("scale").unwrap(),
                    SemanticValue::Rational(CanonicalRational::new(1, 8).unwrap()),
                ),
            ]);
            for (name, values) in [
                ("segment_lengths", vec![3]),
                ("segment_bidirectional", vec![1]),
                ("key_run_counts", vec![1, 1]),
                ("key_run_lengths", vec![3, 3]),
                ("key_run_valid", vec![1, 0]),
            ] {
                attrs.insert(
                    AttributeId::new(name).unwrap(),
                    SemanticValue::Integers(values),
                );
            }
            let values = |kv_heads, dtype| {
                vec![
                    value(
                        "q",
                        ResolvedValueRole::Input,
                        0,
                        &[6, 4 * width],
                        ElementType::Bf16,
                    ),
                    value(
                        "k",
                        ResolvedValueRole::Input,
                        1,
                        &[6, kv_heads * width],
                        dtype,
                    ),
                    value(
                        "v",
                        ResolvedValueRole::Input,
                        2,
                        &[6, kv_heads * width],
                        dtype,
                    ),
                    value(
                        "y",
                        ResolvedValueRole::Output,
                        0,
                        &[6, 4 * width],
                        ElementType::Bf16,
                    ),
                ]
            };
            let heads = if kind == StructuredAttention { 4 } else { 2 };
            let required = (2048 + 16 * width + 80) * 4;
            assert!(
                attention::geometry(kind, &values(heads, ElementType::Bf16), &attrs, required)
                    .is_ok()
            );
            assert!(attention::geometry(
                kind,
                &values(heads, ElementType::Bf16),
                &attrs,
                required - 1
            )
            .is_err());
            assert!(
                attention::geometry(kind, &values(heads, ElementType::F16), &attrs, required)
                    .is_err()
            );
            assert!(
                attention::geometry(kind, &values(3, ElementType::Bf16), &attrs, required).is_err()
            );
            if kind == StructuredAttention {
                assert!(
                    attention::geometry(kind, &values(2, ElementType::Bf16), &attrs, required)
                        .is_err()
                );
            }
            attrs.insert(
                AttributeId::new("key_run_counts").unwrap(),
                SemanticValue::Integers(vec![1]),
            );
            assert!(
                attention::geometry(kind, &values(heads, ElementType::Bf16), &attrs, required)
                    .is_err()
            );
        }
    }
}

#[test]
fn transport_rejects_out_of_bounds_runs_columns_and_invalid_batch_before_submission() {
    use MetalTensorOperation::*;
    let values = vec![
        value(
            "source",
            ResolvedValueRole::Input,
            0,
            &[10, 8],
            ElementType::Bf16,
        ),
        value(
            "result",
            ResolvedValueRole::Output,
            0,
            &[6, 3],
            ElementType::Bf16,
        ),
    ];
    let mut attrs = BTreeMap::from([
        (
            AttributeId::new("batch").unwrap(),
            SemanticValue::Unsigned(2),
        ),
        (
            AttributeId::new("column_start").unwrap(),
            SemanticValue::Unsigned(5),
        ),
        (
            AttributeId::new("column_count").unwrap(),
            SemanticValue::Unsigned(3),
        ),
    ]);
    for (name, numbers) in [
        ("run_lengths", vec![3]),
        ("source_starts", vec![1]),
        ("source_batch_strides", vec![4]),
        ("source_row_strides", vec![1]),
    ] {
        attrs.insert(
            AttributeId::new(name).unwrap(),
            SemanticValue::Integers(numbers),
        );
    }
    assert!(transport::geometry(RowGather, &values, &attrs).is_ok());
    for (name, bad) in [
        ("column_start", SemanticValue::Unsigned(6)),
        ("column_start", SemanticValue::Unsigned(u64::MAX)),
        ("batch", SemanticValue::Unsigned(u64::MAX)),
        ("source_starts", SemanticValue::Integers(vec![4])),
        ("source_starts", SemanticValue::Integers(vec![-1])),
        ("source_row_strides", SemanticValue::Integers(vec![2])),
        ("source_batch_strides", SemanticValue::Integers(vec![4, 4])),
    ] {
        let mut changed = attrs.clone();
        changed.insert(AttributeId::new(name).unwrap(), bad);
        assert!(
            transport::geometry(RowGather, &values, &changed).is_err(),
            "{name}"
        );
    }
    let mut wrong_dtype = values.clone();
    wrong_dtype[0] = value(
        "source",
        ResolvedValueRole::Input,
        0,
        &[10, 8],
        ElementType::F16,
    );
    assert!(transport::geometry(RowGather, &wrong_dtype, &attrs).is_err());
    let values = vec![
        value(
            "left",
            ResolvedValueRole::Input,
            0,
            &[4, 3],
            ElementType::Bf16,
        ),
        value(
            "right",
            ResolvedValueRole::Input,
            1,
            &[6, 3],
            ElementType::Bf16,
        ),
        value(
            "result",
            ResolvedValueRole::Output,
            0,
            &[10, 3],
            ElementType::Bf16,
        ),
    ];
    for (batch, accepted) in [(2, true), (0, false), (3, false)] {
        let attrs = BTreeMap::from([(
            AttributeId::new("batch").unwrap(),
            SemanticValue::Unsigned(batch),
        )]);
        assert_eq!(
            transport::geometry(BatchedRowConcat, &values, &attrs).is_ok(),
            accepted
        );
    }
}

#[test]
fn row_scatter_rejects_overlaps_out_of_bounds_and_unconsumed_updates() {
    let values = vec![
        value(
            "x",
            ResolvedValueRole::Input,
            0,
            &[10, 3],
            ElementType::Bf16,
        ),
        value(
            "updates",
            ResolvedValueRole::Input,
            1,
            &[4, 3],
            ElementType::Bf16,
        ),
        value(
            "y",
            ResolvedValueRole::Output,
            0,
            &[10, 3],
            ElementType::Bf16,
        ),
    ];
    let attributes = |starts, lengths| {
        BTreeMap::from([
            (
                AttributeId::new("destination_starts").unwrap(),
                SemanticValue::Integers(starts),
            ),
            (
                AttributeId::new("run_lengths").unwrap(),
                SemanticValue::Integers(lengths),
            ),
        ])
    };
    let attrs = attributes(vec![0, 8], vec![2, 2]);
    assert!(scatter::geometry(&values, &attrs).is_ok());
    for (starts, lengths) in [
        (vec![0, 1], vec![2, 2]), // overlap
        (vec![8, 0], vec![2, 2]), // unordered
        (vec![0, 9], vec![2, 2]), // past end
        (vec![-1, 8], vec![2, 2]),
        (vec![0, 8], vec![0, 4]),
        (vec![0, 8], vec![1, 2]), // unused update row
        (vec![0, 8], vec![3, 2]), // more updates than the input
        (vec![0], vec![2, 2]),
        (vec![], vec![]),
        (vec![0; 33], vec![1; 33]),
    ] {
        assert!(scatter::geometry(&values, &attributes(starts, lengths)).is_err());
    }
    for (shape, element) in [([4, 2], ElementType::Bf16), ([4, 3], ElementType::F16)] {
        let mut wrong = values.clone();
        wrong[1] = value("updates", ResolvedValueRole::Input, 1, &shape, element);
        assert!(scatter::geometry(&wrong, &attrs).is_err());
    }
}
