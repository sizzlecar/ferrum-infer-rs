use super::*;
use crate::SemanticValue;

fn tensor(shape: &[u64], dtype: ElementType) -> ResolvedTensorSpec {
    ResolvedTensorSpec::new(shape.to_vec(), dtype, ResolvedTensorLayout::Contiguous).unwrap()
}
fn epsilon() -> CanonicalRational {
    CanonicalRational::new(1, 1_000_000).unwrap()
}
fn attributes(head_dim: u64, epsilon: CanonicalRational) -> BTreeMap<AttributeId, SemanticValue> {
    BTreeMap::from([
        (
            AttributeId::new("head_dim").unwrap(),
            SemanticValue::Unsigned(head_dim),
        ),
        (
            AttributeId::new("epsilon").unwrap(),
            SemanticValue::Rational(epsilon),
        ),
    ])
}

#[test]
fn independent_bf16_contract_has_shared_gamma_and_no_alias_or_token_geometry() {
    let contract = head_rms_norm_bf16_contract().unwrap();
    let descriptor = contract.descriptor();
    assert_eq!(descriptor.id.as_str(), HEAD_RMS_NORM_BF16_OPERATION_ID);
    assert_eq!(descriptor.version, ContractVersion::new(1, 0));
    assert_eq!(descriptor.inputs.len(), 2);
    assert_eq!(descriptor.outputs.len(), 1);
    assert_eq!(
        descriptor.inputs[0].dimensions(),
        [symbol("rows"), symbol("columns")]
    );
    assert_eq!(descriptor.inputs[1].dimensions(), [symbol("head_dim")]);
    assert_eq!(
        descriptor.outputs[0].dimensions(),
        descriptor.inputs[0].dimensions()
    );
    for port in descriptor.inputs.iter().chain(&descriptor.outputs) {
        assert_eq!(port.element_types(), &BTreeSet::from([ElementType::Bf16]));
        assert_eq!(port.layouts(), [LayoutConstraint::Contiguous]);
        assert_eq!(port.alias(), &AliasPolicy::NoAlias);
    }
    assert!(descriptor
        .inputs
        .iter()
        .all(|port| port.access() == TensorAccess::Read));
    assert_eq!(descriptor.outputs[0].access(), TensorAccess::Write);
    assert_eq!(descriptor.oracle, OracleSpec::OperationDefined);
    assert_eq!(
        descriptor.resources.scratch,
        ResourcePresenceRequirement::Optional
    );
    assert_eq!(
        descriptor.resources.binding,
        ResourcePresenceRequirement::Forbidden
    );
    assert_eq!(
        descriptor.resources.persistent,
        ResourcePresenceRequirement::Forbidden
    );
    assert_eq!(
        descriptor.provider.required_capabilities,
        BTreeSet::from([CapabilityId::new(HEAD_RMS_NORM_BF16_CAPABILITY_ID).unwrap()])
    );
    let legacy = rms_norm_contract().unwrap();
    assert!(contract
        .validate_signature(&legacy.descriptor().inputs, &legacy.descriptor().outputs)
        .is_err());
}

#[test]
fn attributes_are_closed_typed_and_bounded() {
    let contract = head_rms_norm_bf16_contract().unwrap();
    let descriptor = contract.descriptor();
    for head_dim in [1, 3, 128, HEAD_RMS_NORM_BF16_MAX_HEAD_DIM] {
        for epsilon in [
            CanonicalRational::new(1, 1_000_000_000_000).unwrap(),
            CanonicalRational::new(1, 1).unwrap(),
        ] {
            descriptor
                .validate_attributes(&attributes(head_dim, epsilon))
                .unwrap();
        }
    }
    for head_dim in [0, HEAD_RMS_NORM_BF16_MAX_HEAD_DIM + 1] {
        assert!(descriptor
            .validate_attributes(&attributes(head_dim, epsilon()))
            .is_err());
    }
    for (n, d) in [(0, 1), (-1, 1), (1, 1_000_000_000_001), (2, 1)] {
        assert!(descriptor
            .validate_attributes(&attributes(128, CanonicalRational::new(n, d).unwrap()))
            .is_err());
    }
    let valid = attributes(128, epsilon());
    for name in ["head_dim", "epsilon"] {
        let mut missing = valid.clone();
        missing.remove(&AttributeId::new(name).unwrap());
        assert!(descriptor.validate_attributes(&missing).is_err());
    }
    for (name, value) in [
        ("epsilon", SemanticValue::Unsigned(1)),
        ("head_dim", SemanticValue::Integer(128)),
        ("gamma_offset", SemanticValue::Unsigned(1)),
    ] {
        let mut wrong = valid.clone();
        wrong.insert(AttributeId::new(name).unwrap(), value);
        assert!(descriptor.validate_attributes(&wrong).is_err());
    }
}

#[test]
fn geometry_proves_head_partition_and_gamma_without_payload_allocation() {
    for d in [1, 3, 128, HEAD_RMS_NORM_BF16_MAX_HEAD_DIM] {
        let input = tensor(&[2, 3 * d], ElementType::Bf16);
        let gamma = tensor(&[d], ElementType::Bf16);
        let checked = HeadRmsNormBf16Geometry::new(&input, &gamma, &input, d, epsilon()).unwrap();
        assert_eq!(
            (
                checked.rows(),
                checked.columns(),
                checked.heads(),
                checked.head_dim()
            ),
            (2, 3 * d, 3, d)
        );
        assert_eq!(checked.effective_epsilon().to_bits(), 1e-6_f32.to_bits());
        assert_eq!(checked.epsilon(), epsilon());
    }
    let input = tensor(&[2, 256], ElementType::Bf16);
    let gamma = tensor(&[128], ElementType::Bf16);
    for d in [0, 3, 64, 256, HEAD_RMS_NORM_BF16_MAX_HEAD_DIM + 1] {
        assert!(HeadRmsNormBf16Geometry::new(&input, &gamma, &input, d, epsilon()).is_err());
    }
    let wrong_output = tensor(&[4, 128], ElementType::Bf16);
    assert!(HeadRmsNormBf16Geometry::new(&input, &gamma, &wrong_output, 128, epsilon()).is_err());
    for invalid_gamma in [
        tensor(&[1, 128], ElementType::Bf16),
        tensor(&[128], ElementType::F16),
        ResolvedTensorSpec::new(
            vec![128],
            ElementType::Bf16,
            ResolvedTensorLayout::Strided {
                byte_strides: vec![4],
            },
        )
        .unwrap(),
    ] {
        assert!(
            HeadRmsNormBf16Geometry::new(&input, &invalid_gamma, &input, 128, epsilon()).is_err()
        );
    }
    for invalid_input in [
        tensor(&[512], ElementType::Bf16),
        tensor(&[2, 256], ElementType::F16),
        ResolvedTensorSpec::new(
            vec![2, 256],
            ElementType::Bf16,
            ResolvedTensorLayout::Strided {
                byte_strides: vec![1024, 2],
            },
        )
        .unwrap(),
    ] {
        assert!(HeadRmsNormBf16Geometry::new(
            &invalid_input,
            &gamma,
            &invalid_input,
            128,
            epsilon()
        )
        .is_err());
    }
}

#[test]
fn geometry_rejects_invalid_epsilon_and_tensor_constructor_rejects_overflow() {
    let input = tensor(&[1, 128], ElementType::Bf16);
    let gamma = tensor(&[128], ElementType::Bf16);
    for (n, d) in [(0, 1), (-1, 1), (1, u64::MAX), (i64::MAX, 1)] {
        assert!(HeadRmsNormBf16Geometry::new(
            &input,
            &gamma,
            &input,
            128,
            CanonicalRational::new(n, d).unwrap()
        )
        .is_err());
    }
    for dimensions in [vec![0, 128], vec![u64::MAX / 256 + 1, 128]] {
        assert!(ResolvedTensorSpec::new(
            dimensions,
            ElementType::Bf16,
            ResolvedTensorLayout::Contiguous
        )
        .is_err());
    }
}
