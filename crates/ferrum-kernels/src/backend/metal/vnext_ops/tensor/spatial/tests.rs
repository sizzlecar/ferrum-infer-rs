use super::super::tests::value;
use super::*;
fn attrs(entries: &[(&str, SemanticValue)]) -> BTreeMap<AttributeId, SemanticValue> {
    entries
        .iter()
        .map(|(name, value)| (AttributeId::new(*name).unwrap(), value.clone()))
        .collect()
}
#[test]
fn convolution_padding_has_exact_workspace_and_checked_grouped_geometry() {
    let values = vec![
        value(
            "x",
            ResolvedValueRole::Input,
            0,
            &[2, 4, 5, 7],
            ElementType::Bf16,
        ),
        value(
            "w",
            ResolvedValueRole::Input,
            1,
            &[6, 2, 3, 2],
            ElementType::Bf16,
        ),
        value("b", ResolvedValueRole::Input, 2, &[6], ElementType::Bf16),
        value(
            "y",
            ResolvedValueRole::Output,
            0,
            &[2, 6, 3, 6],
            ElementType::Bf16,
        ),
    ];
    let mut a = attrs(&[
        ("padding", SemanticValue::Integers(vec![1, 2, 0, 1])),
        ("stride", SemanticValue::Integers(vec![2, 1])),
        ("dilation", SemanticValue::Integers(vec![1, 2])),
        ("groups", SemanticValue::Unsigned(2)),
        ("padding_mode", SemanticValue::Text("explicit_zero".into())),
    ]);
    assert_eq!(
        conv::geometry(&values, &a).unwrap().scratch_bytes,
        2 * 2 * 4 * 8 * 8
    );
    a.insert(
        AttributeId::new("padding_mode").unwrap(),
        SemanticValue::Text("implicit_zero".into()),
    );
    assert_eq!(conv::geometry(&values, &a).unwrap().scratch_bytes, 0);
    for (name, bad) in [
        ("groups", SemanticValue::Unsigned(3)),
        ("stride", SemanticValue::Integers(vec![0, 1])),
        ("padding", SemanticValue::Integers(vec![-1, 2, 0, 1])),
        ("dilation", SemanticValue::Integers(vec![2, 2])),
    ] {
        let mut changed = a.clone();
        changed.insert(AttributeId::new(name).unwrap(), bad);
        assert!(conv::geometry(&values, &changed).is_err());
    }
}
#[test]
fn channel_l2_uses_device_capacity_and_keeps_gamma_rank_contracts_distinct() {
    for kind in [
        MetalTensorOperation::ChannelL2Norm,
        MetalTensorOperation::ChannelL2NormImage,
    ] {
        for (channels, required) in [(1, 16), (3, 16), (1152, 8192), (8193, 65536)] {
            let gamma = if kind == MetalTensorOperation::ChannelL2Norm {
                vec![channels, 1, 1, 1]
            } else {
                vec![channels, 1, 1]
            };
            let values = vec![
                value(
                    "x",
                    ResolvedValueRole::Input,
                    0,
                    &[2, channels, 3, 5],
                    ElementType::Bf16,
                ),
                value(
                    "gamma",
                    ResolvedValueRole::Input,
                    1,
                    &gamma,
                    ElementType::Bf16,
                ),
                value(
                    "y",
                    ResolvedValueRole::Output,
                    0,
                    &[2, channels, 3, 5],
                    ElementType::Bf16,
                ),
            ];
            let a = attrs(&[
                (
                    "epsilon",
                    SemanticValue::Rational(CanonicalRational::new(1, 1_000_000_000_000).unwrap()),
                ),
                (
                    "scale",
                    SemanticValue::Rational(CanonicalRational::new(3, 1).unwrap()),
                ),
            ]);
            assert!(channel::geometry(kind, &values, &a, required).is_ok());
            assert!(channel::geometry(kind, &values, &a, required - 1).is_err());
            let other = if kind == MetalTensorOperation::ChannelL2Norm {
                MetalTensorOperation::ChannelL2NormImage
            } else {
                MetalTensorOperation::ChannelL2Norm
            };
            assert!(channel::geometry(other, &values, &a, required).is_err());
        }
    }
}
#[test]
fn spatial_attention_workspace_is_linear_in_output_and_needs_exact_qkv() {
    let a = attrs(&[(
        "scale",
        SemanticValue::Rational(CanonicalRational::new(1, 32).unwrap()),
    )]);
    for channels in [1, 1152, 2048] {
        let mut values = vec![
            value(
                "qkv",
                ResolvedValueRole::Input,
                0,
                &[2, 3 * channels, 3, 11],
                ElementType::Bf16,
            ),
            value(
                "y",
                ResolvedValueRole::Output,
                0,
                &[2, channels, 3, 11],
                ElementType::Bf16,
            ),
        ];
        assert_eq!(
            attention::geometry(&values, &a).unwrap().scratch_bytes,
            2 * channels * 33 * 4
        );
        values[0] = value(
            "qkv",
            ResolvedValueRole::Input,
            0,
            &[2, 3 * channels + 1, 3, 11],
            ElementType::Bf16,
        );
        assert!(attention::geometry(&values, &a).is_err());
    }
}

#[test]
fn group_mean_requires_exact_spatial_and_expanded_channel_division() {
    let a = attrs(&[
        ("scale_h", SemanticValue::Unsigned(2)),
        ("scale_w", SemanticValue::Unsigned(2)),
        ("frame_factor", SemanticValue::Unsigned(2)),
    ]);
    let mut values = vec![
        value(
            "x",
            ResolvedValueRole::Input,
            0,
            &[2, 3, 4, 6],
            ElementType::Bf16,
        ),
        value(
            "y",
            ResolvedValueRole::Output,
            0,
            &[2, 6, 2, 3],
            ElementType::Bf16,
        ),
    ];
    assert_eq!(mean::geometry(&values, &a).unwrap().group_size(), 4);
    for shape in [[2, 5, 2, 3], [2, 6, 3, 2], [1, 6, 2, 3]] {
        values[1] = value("y", ResolvedValueRole::Output, 0, &shape, ElementType::Bf16);
        assert!(mean::geometry(&values, &a).is_err());
    }
}
