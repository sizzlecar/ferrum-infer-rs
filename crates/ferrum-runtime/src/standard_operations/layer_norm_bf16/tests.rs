use super::*;
use crate::SemanticValue;

fn rational(numerator: i64, denominator: u64) -> CanonicalRational {
    CanonicalRational::new(numerator, denominator).unwrap()
}

fn attributes(
    hidden_size: u64,
    epsilon: CanonicalRational,
) -> BTreeMap<AttributeId, SemanticValue> {
    BTreeMap::from([
        (
            AttributeId::new("hidden_size").unwrap(),
            SemanticValue::Unsigned(hidden_size),
        ),
        (
            AttributeId::new("epsilon").unwrap(),
            SemanticValue::Rational(epsilon),
        ),
    ])
}

fn tensor(dimensions: &[u64], element_type: ElementType) -> ResolvedTensorSpec {
    ResolvedTensorSpec::new(
        dimensions.to_vec(),
        element_type,
        ResolvedTensorLayout::Contiguous,
    )
    .unwrap()
}

#[test]
fn contract_has_one_bf16_port_per_side_and_no_affine_or_state_resources() {
    let contract = layer_norm_bf16_contract().unwrap();
    let descriptor = contract.descriptor();
    assert_eq!(descriptor.id.as_str(), LAYER_NORM_BF16_OPERATION_ID);
    assert_eq!(descriptor.version, ContractVersion::new(1, 0));
    assert_eq!(descriptor.inputs.len(), 1);
    assert_eq!(descriptor.outputs.len(), 1);
    for port in descriptor.inputs.iter().chain(&descriptor.outputs) {
        assert_eq!(port.dimensions(), [symbol("rows"), symbol("hidden_size")]);
        assert_eq!(port.element_types(), &BTreeSet::from([ElementType::Bf16]));
        assert_eq!(port.layouts(), [LayoutConstraint::Contiguous]);
        assert_eq!(port.alias(), &AliasPolicy::NoAlias);
    }
    assert_eq!(descriptor.inputs[0].access(), TensorAccess::Read);
    assert_eq!(descriptor.outputs[0].access(), TensorAccess::Write);
    assert_eq!(descriptor.oracle, OracleSpec::OperationDefined);
    assert_eq!(descriptor.attributes.entries().len(), 2);
    assert_eq!(descriptor.resources.minimum_value_alignment_bytes, 16);
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
        BTreeSet::from([CapabilityId::new(LAYER_NORM_BF16_CAPABILITY_ID).unwrap()])
    );
    contract
        .validate_signature(&descriptor.inputs, &descriptor.outputs)
        .unwrap();
    let rms = rms_norm_contract().unwrap();
    assert!(contract
        .validate_signature(&rms.descriptor().inputs, &rms.descriptor().outputs)
        .is_err());
}

#[test]
fn attributes_accept_exact_domain_endpoints_and_reject_outside_values() {
    let contract = layer_norm_bf16_contract().unwrap();
    let descriptor = contract.descriptor();
    for hidden_size in [1, 4096, LAYER_NORM_BF16_MAX_HIDDEN_SIZE] {
        for epsilon in [
            rational(1, 1_000_000_000_000),
            rational(1, 1_000_000),
            rational(1, 1),
        ] {
            descriptor
                .validate_attributes(&attributes(hidden_size, epsilon))
                .unwrap();
        }
    }
    for hidden_size in [0, LAYER_NORM_BF16_MAX_HIDDEN_SIZE + 1, u64::MAX] {
        assert!(descriptor
            .validate_attributes(&attributes(hidden_size, rational(1, 1_000_000)))
            .is_err());
    }
    for epsilon in [
        rational(-1, 1),
        rational(0, 1),
        rational(1, 1_000_000_000_001),
        rational(1001, 1000),
    ] {
        assert!(descriptor
            .validate_attributes(&attributes(4, epsilon))
            .is_err());
    }
}

#[test]
fn attributes_require_the_exact_typed_closed_vocabulary() {
    let contract = layer_norm_bf16_contract().unwrap();
    let descriptor = contract.descriptor();
    let valid = attributes(4, rational(1, 1_000_000));
    for name in ["hidden_size", "epsilon"] {
        let mut missing = valid.clone();
        missing.remove(&AttributeId::new(name).unwrap());
        assert!(descriptor.validate_attributes(&missing).is_err());
    }
    for name in ["gamma", "beta", "affine", "unbiased"] {
        let mut unknown = valid.clone();
        unknown.insert(AttributeId::new(name).unwrap(), SemanticValue::Bool(false));
        assert!(descriptor.validate_attributes(&unknown).is_err());
    }
    for (name, value) in [
        ("epsilon", SemanticValue::Unsigned(1)),
        ("hidden_size", SemanticValue::Integer(4)),
    ] {
        let mut wrong_type = valid.clone();
        wrong_type.insert(AttributeId::new(name).unwrap(), value);
        assert!(descriptor.validate_attributes(&wrong_type).is_err());
    }
}

#[test]
fn geometry_binds_width_to_actual_tensors_without_allocating_payloads() {
    let epsilon = rational(1, 1_000_000);
    for width in [1, 3, 4096, LAYER_NORM_BF16_MAX_HIDDEN_SIZE] {
        let spec = tensor(&[2, width], ElementType::Bf16);
        let geometry = LayerNormBf16Geometry::new(&spec, &spec, width, epsilon).unwrap();
        assert_eq!(geometry.rows(), 2);
        assert_eq!(geometry.hidden_size(), width);
        assert_eq!(geometry.epsilon(), epsilon);
        assert_eq!(geometry.effective_epsilon().to_bits(), 1e-6_f32.to_bits());
    }
    let spec = tensor(&[2, 4], ElementType::Bf16);
    let transposed_shape = tensor(&[4, 2], ElementType::Bf16);
    assert!(LayerNormBf16Geometry::new(&spec, &transposed_shape, 4, epsilon).is_err());
    assert!(LayerNormBf16Geometry::new(&spec, &spec, 3, epsilon).is_err());
    assert!(LayerNormBf16Geometry::new(&spec, &spec, 0, epsilon).is_err());
    let too_wide = tensor(&[1, LAYER_NORM_BF16_MAX_HIDDEN_SIZE + 1], ElementType::Bf16);
    assert!(LayerNormBf16Geometry::new(
        &too_wide,
        &too_wide,
        LAYER_NORM_BF16_MAX_HIDDEN_SIZE + 1,
        epsilon
    )
    .is_err());
}

#[test]
fn geometry_rejects_wrong_dtype_rank_and_layout() {
    let epsilon = rational(1, 1_000_000);
    for spec in [
        tensor(&[2, 4], ElementType::F16),
        tensor(&[2, 4], ElementType::F32),
        tensor(&[8], ElementType::Bf16),
        tensor(&[1, 2, 4], ElementType::Bf16),
        ResolvedTensorSpec::new(
            vec![2, 4],
            ElementType::Bf16,
            ResolvedTensorLayout::Strided {
                byte_strides: vec![16, 2],
            },
        )
        .unwrap(),
    ] {
        assert!(LayerNormBf16Geometry::new(&spec, &spec, 4, epsilon).is_err());
    }
}

#[test]
fn tensor_constructor_rejects_zero_and_overflow_before_geometry_can_be_bound() {
    for dimensions in [
        vec![0, 4],
        vec![2, 0],
        vec![u64::MAX / 2 + 1, 1],
        vec![u64::MAX / 8 + 1, 4],
        vec![u64::MAX, LAYER_NORM_BF16_MAX_HIDDEN_SIZE],
    ] {
        assert!(ResolvedTensorSpec::new(
            dimensions,
            ElementType::Bf16,
            ResolvedTensorLayout::Contiguous,
        )
        .is_err());
    }
}

#[test]
fn geometry_enforces_rational_domain_and_effective_epsilon() {
    let spec = tensor(&[2, 4], ElementType::Bf16);
    for epsilon in [rational(1, 1_000_000_000_000), rational(1, 1)] {
        let geometry = LayerNormBf16Geometry::new(&spec, &spec, 4, epsilon).unwrap();
        assert!(geometry.effective_epsilon().is_normal());
    }
    for epsilon in [
        rational(0, 1),
        rational(-1, 1),
        rational(1, u64::MAX),
        rational(i64::MAX, 1),
    ] {
        assert!(LayerNormBf16Geometry::new(&spec, &spec, 4, epsilon).is_err());
    }
    let canonical = LayerNormBf16Geometry::new(&spec, &spec, 4, rational(1, 1_000_000)).unwrap();
    let reduced = LayerNormBf16Geometry::new(&spec, &spec, 4, rational(6, 6_000_000)).unwrap();
    assert_eq!(canonical, reduced);
    assert_eq!(
        canonical.effective_epsilon().to_bits(),
        reduced.effective_epsilon().to_bits()
    );
}
