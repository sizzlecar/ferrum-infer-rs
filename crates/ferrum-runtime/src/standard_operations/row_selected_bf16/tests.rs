use super::*;
fn t(shape: &[u64], dtype: ElementType) -> ResolvedTensorSpec {
    ResolvedTensorSpec::new(shape.to_vec(), dtype, ResolvedTensorLayout::Contiguous).unwrap()
}
fn attrs(kind: RowSelectedBf16Kind) -> BTreeMap<AttributeId, SemanticValue> {
    let mut a = BTreeMap::from([
        (
            AttributeId::new("run_lengths").unwrap(),
            SemanticValue::Integers(vec![2, 1, 2]),
        ),
        (
            AttributeId::new("parameter_rows").unwrap(),
            SemanticValue::Integers(vec![1, 0, 1]),
        ),
    ]);
    for (name, value) in match kind {
        RowSelectedBf16Kind::Modulation => vec![("shift_column", 1), ("scale_column", 5)],
        RowSelectedBf16Kind::Addcmul => vec![("gate_column", 9)],
    } {
        a.insert(
            AttributeId::new(name).unwrap(),
            SemanticValue::Unsigned(value),
        );
    }
    a
}
#[test]
fn row_selected_bf16_geometry_validates_original_table_windows_and_full_routing() {
    let x = t(&[5, 3], ElementType::Bf16);
    let p = t(&[2, 12], ElementType::Bf16);
    for kind in [
        RowSelectedBf16Kind::Modulation,
        RowSelectedBf16Kind::Addcmul,
    ] {
        let ports = if kind == RowSelectedBf16Kind::Modulation {
            vec![&x, &p]
        } else {
            vec![&x, &p, &x]
        };
        let a = attrs(kind);
        let g = RowSelectedBf16Geometry::new(kind, &ports, &x, &a).unwrap();
        assert_eq!(
            (0..6).map(|r| g.parameter_row(r)).collect::<Vec<_>>(),
            [Some(1), Some(1), Some(0), Some(1), Some(1), None]
        );
        assert_eq!(g.parameter_columns(), 12);
        assert_eq!(g.elements(), 15);
        assert_eq!(
            contract(kind).unwrap().descriptor.resources.scratch,
            ResourcePresenceRequirement::Forbidden
        );
        for (field, value) in [
            ("run_lengths", SemanticValue::Integers(vec![2, 2])),
            ("run_lengths", SemanticValue::Integers(vec![0, 1, 4])),
            ("parameter_rows", SemanticValue::Integers(vec![1, 2, 1])),
            ("parameter_rows", SemanticValue::Integers(vec![1, 0])),
            ("parameter_rows", SemanticValue::Integers(vec![1, -1, 1])),
        ] {
            let mut bad = a.clone();
            bad.insert(AttributeId::new(field).unwrap(), value);
            assert!(RowSelectedBf16Geometry::new(kind, &ports, &x, &bad).is_err());
        }
        let mut bad = a.clone();
        bad.insert(
            AttributeId::new(if kind == RowSelectedBf16Kind::Modulation {
                "scale_column"
            } else {
                "gate_column"
            })
            .unwrap(),
            SemanticValue::Unsigned(10),
        );
        assert!(RowSelectedBf16Geometry::new(kind, &ports, &x, &bad).is_err());
        assert!(
            RowSelectedBf16Geometry::new(kind, &ports, &t(&[5, 3], ElementType::F32), &a).is_err()
        );
        let mut zero = a.clone();
        zero.insert(
            AttributeId::new(if kind == RowSelectedBf16Kind::Modulation {
                "shift_column"
            } else {
                "gate_column"
            })
            .unwrap(),
            SemanticValue::Unsigned(0),
        );
        assert!(RowSelectedBf16Geometry::new(kind, &ports, &x, &zero).is_ok());
        let wrong = t(&[2, 12], ElementType::F32);
        let bad_ports = if kind == RowSelectedBf16Kind::Modulation {
            vec![&x, &wrong]
        } else {
            vec![&x, &wrong, &x]
        };
        assert!(RowSelectedBf16Geometry::new(kind, &bad_ports, &x, &a).is_err());
        for count in [ROW_SELECTED_BF16_MAX_RUNS, ROW_SELECTED_BF16_MAX_RUNS + 1] {
            let x = t(&[count as u64, 3], ElementType::Bf16);
            let ports = if kind == RowSelectedBf16Kind::Modulation {
                vec![&x, &p]
            } else {
                vec![&x, &p, &x]
            };
            let mut routed = a.clone();
            routed.insert(
                AttributeId::new("run_lengths").unwrap(),
                SemanticValue::Integers(vec![1; count]),
            );
            routed.insert(
                AttributeId::new("parameter_rows").unwrap(),
                SemanticValue::Integers(vec![1; count]),
            );
            assert_eq!(
                RowSelectedBf16Geometry::new(kind, &ports, &x, &routed).is_ok(),
                count == ROW_SELECTED_BF16_MAX_RUNS
            );
        }
    }
}
