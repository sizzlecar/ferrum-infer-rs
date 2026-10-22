use super::*;
fn tensor(rows: u64, columns: u64) -> ResolvedTensorSpec {
    ResolvedTensorSpec::new(
        vec![rows, columns],
        ElementType::Bf16,
        ResolvedTensorLayout::Contiguous,
    )
    .unwrap()
}
fn attrs(
    s: u64,
    d: u64,
    lengths: &[i64],
    bidir: &[i64],
    counts: &[i64],
    runs: &[i64],
    valid: &[i64],
) -> BTreeMap<AttributeId, SemanticValue> {
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
        ("segment_lengths", lengths),
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
    a
}
fn checked(
    t: &ResolvedTensorSpec,
    a: &BTreeMap<AttributeId, SemanticValue>,
) -> Result<StructuredAttentionBf16Geometry, VNextError> {
    StructuredAttentionBf16Geometry::new(t, t, t, t, a)
}
#[test]
fn contract_has_four_bf16_ports_closed_attributes_and_no_auxiliary_storage() {
    let c = structured_attention_bf16_contract().unwrap();
    let d = c.descriptor();
    assert_eq!(d.id.as_str(), STRUCTURED_ATTENTION_BF16_OPERATION_ID);
    assert_eq!(d.version, ContractVersion::new(1, 0));
    assert_eq!(d.inputs.len(), 3);
    assert_eq!(d.outputs.len(), 1);
    for p in d.inputs.iter().chain(&d.outputs) {
        assert_eq!(p.element_types(), &BTreeSet::from([ElementType::Bf16]));
        assert_eq!(p.alias(), &AliasPolicy::NoAlias);
        assert_eq!(p.layouts(), [LayoutConstraint::Contiguous]);
    }
    assert_eq!(d.oracle, OracleSpec::OperationDefined);
    assert_eq!(d.resources.scratch, ResourcePresenceRequirement::Forbidden);
    assert_eq!(d.resources.binding, ResourcePresenceRequirement::Forbidden);
    assert_eq!(
        d.resources.persistent,
        ResourcePresenceRequirement::Forbidden
    );
    let mut a = attrs(2, 2, &[2], &[1], &[1], &[2], &[1]);
    d.validate_attributes(&a).unwrap();
    a.insert(
        AttributeId::new("causal").unwrap(),
        SemanticValue::Unsigned(1),
    );
    assert!(d.validate_attributes(&a).is_err());
    assert!(d.validate_attributes(&BTreeMap::new()).is_err());
}
#[test]
fn separate_image_blocks_and_internal_key_padding_have_exact_counted_work() {
    let a = attrs(
        10,
        2,
        &[2, 3, 2, 3],
        &[0, 1, 1, 1],
        &[3, 5],
        &[3, 1, 6, 3, 1, 4, 1, 1],
        &[1, 0, 1, 1, 0, 1, 0, 1],
    );
    let g = checked(&tensor(20, 4), &a).unwrap();
    assert_eq!((g.batch(), g.heads(), g.element_count()), (2, 2, 80));
    let m = g.mask();
    assert_eq!(m.segment_ends(), [2, 5, 7, 10]);
    assert_eq!(m.segment_bidirectional(), [false, true, true, true]);
    assert_eq!(m.visible_end(3), Some(5));
    assert_eq!(m.visible_end(5), Some(7));
    assert_eq!(m.visible_end(10), None);
    assert_eq!(m.key_is_valid(0, 3), Some(false));
    assert_eq!(m.key_is_valid(1, 8), Some(false));
    assert_eq!(m.key_is_valid(2, 0), None);
    // Explicit mask matrix, independently expressed as causal OR identical image id.
    let ids = [-1, -1, 0, 0, 0, 1, 1, 2, 2, 2];
    let mut pairs = 0;
    let mut nonempty = 0;
    for b in 0..2 {
        for q in 0..10 {
            let mut n = 0;
            for k in 0..10 {
                let expected =
                    (k <= q || (ids[q] >= 0 && ids[q] == ids[k])) && k != 3 && !(b == 1 && k == 8);
                assert_eq!(
                    (k as u32) < m.visible_end(q as u32).unwrap()
                        && m.key_is_valid(b, k as u32) == Some(true),
                    expected
                );
                n += u64::from(expected);
            }
            pairs += n;
            nonempty += u64::from(n > 0);
        }
    }
    assert_eq!((g.visible_pairs(), g.nonempty_rows()), (pairs, nonempty));
    assert_eq!(
        g.logical_arithmetic_operations(),
        2 * (16 * pairs - 9 * nonempty)
    );
}
#[test]
fn mask_work_formula_matches_independent_small_visibility_enumeration() {
    for s in 2..=7_u64 {
        for split in 1..s {
            for flags in 0..4 {
                for validity in 0..(1_u64 << s) {
                    let mut runs = Vec::new();
                    let mut valid = Vec::new();
                    for k in 0..s {
                        let flag = ((validity >> k) & 1) as i64;
                        if valid.last() == Some(&flag) {
                            *runs.last_mut().unwrap() += 1;
                        } else {
                            valid.push(flag);
                            runs.push(1);
                        }
                    }
                    let a = attrs(
                        s,
                        2,
                        &[split as i64, (s - split) as i64],
                        &[flags & 1, (flags >> 1) & 1],
                        &[runs.len() as i64],
                        &runs,
                        &valid,
                    );
                    let g = checked(&tensor(s, 2), &a).unwrap();
                    let mut p = 0;
                    let mut n = 0;
                    for q in 0..s {
                        let mut count = 0;
                        for k in 0..s {
                            let same = (q < split) == (k < split);
                            let bidir = if q < split {
                                flags & 1 != 0
                            } else {
                                flags & 2 != 0
                            };
                            count += u64::from(
                                ((validity >> k) & 1) != 0 && (k <= q || (same && bidir)),
                            );
                        }
                        p += count;
                        n += u64::from(count > 0);
                    }
                    assert_eq!((g.visible_pairs(), g.nonempty_rows()), (p, n));
                }
            }
        }
    }
}
#[test]
fn malformed_shapes_masks_scale_and_unrepresentable_work_are_rejected() {
    let t = tensor(4, 4);
    let a = attrs(2, 2, &[2], &[1], &[1, 1], &[2, 2], &[1, 1]);
    checked(&t, &a).unwrap();
    for (name, value) in [
        ("sequence_length", SemanticValue::Unsigned(3)),
        ("head_dim", SemanticValue::Unsigned(3)),
        ("head_dim", SemanticValue::Unsigned(512)),
        (
            "scale",
            SemanticValue::Rational(CanonicalRational::new(-1, 1).unwrap()),
        ),
        ("segment_lengths", SemanticValue::Integers(vec![1])),
        ("segment_bidirectional", SemanticValue::Integers(vec![2])),
        ("key_run_counts", SemanticValue::Integers(vec![1])),
        ("key_run_lengths", SemanticValue::Integers(vec![0, 2])),
        ("key_run_valid", SemanticValue::Integers(vec![1, 2])),
    ] {
        let mut bad = a.clone();
        bad.insert(AttributeId::new(name).unwrap(), value);
        assert!(checked(&t, &bad).is_err(), "{name}");
    }
    let bad = attrs(2, 2, &[2], &[1], &[2, 1], &[1, 1, 2], &[1, 1, 1]);
    assert!(checked(&t, &bad).is_err());
    for other in [
        tensor(2, 8),
        ResolvedTensorSpec::new(
            vec![4, 4],
            ElementType::F16,
            ResolvedTensorLayout::Contiguous,
        )
        .unwrap(),
        ResolvedTensorSpec::new(
            vec![4, 4],
            ElementType::Bf16,
            ResolvedTensorLayout::Strided {
                byte_strides: vec![8, 2],
            },
        )
        .unwrap(),
    ] {
        assert!(StructuredAttentionBf16Geometry::new(&t, &other, &t, &t, &a).is_err());
    }
    let empty = attrs(2, 2, &[2], &[1], &[1, 1], &[2, 2], &[0, 0]);
    assert_eq!(
        checked(&t, &empty).unwrap().logical_arithmetic_operations(),
        0
    );
    let s = u64::from(u32::MAX);
    let large = tensor(s, 2);
    let a = attrs(s, 2, &[s as i64], &[1], &[1], &[s as i64], &[1]);
    assert!(checked(&large, &a).is_err());
}
