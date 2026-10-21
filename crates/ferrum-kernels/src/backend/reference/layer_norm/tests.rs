use super::*;
use crate::backend::reference::{
    reference_vnext_layer_norm_bf16_provider, ReferenceVNextComposition,
};
use ferrum_runtime::{
    AliasPolicy, BufferUsage, CanonicalRational, DeviceId, ProgramValueId, ResolvedTensorLayout,
    ResolvedTensorSpec, ResolvedValueStorage, ResourceId, TensorAccess,
};

fn value(
    name: &str,
    role: ResolvedValueRole,
    dimensions: Vec<u64>,
    dtype: ElementType,
) -> ResolvedValueBinding {
    let tensor =
        ResolvedTensorSpec::new(dimensions, dtype, ResolvedTensorLayout::Contiguous).unwrap();
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
        0,
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

fn attributes(width: u64) -> BTreeMap<AttributeId, SemanticValue> {
    BTreeMap::from([
        (
            AttributeId::new("hidden_size").unwrap(),
            SemanticValue::Unsigned(width),
        ),
        (
            AttributeId::new("epsilon").unwrap(),
            SemanticValue::Rational(CanonicalRational::new(3, 4).unwrap()),
        ),
    ])
}

#[test]
fn layer_norm_provider_advertises_its_own_contract_without_weight_formats() {
    let composition =
        ReferenceVNextComposition::create(DeviceId::new("device.reference.norm-provider").unwrap())
            .unwrap();
    let provider = reference_vnext_layer_norm_bf16_provider(composition.runtime()).unwrap();
    let descriptor = provider.descriptor();
    assert_eq!(
        descriptor.operation_id().as_str(),
        LAYER_NORM_BF16_OPERATION_ID
    );
    assert_eq!(
        descriptor.operation_fingerprint(),
        layer_norm_bf16_contract()
            .unwrap()
            .descriptor()
            .fingerprint()
            .unwrap()
    );
    assert_eq!(
        descriptor.capabilities(),
        &BTreeSet::from([CapabilityId::new(LAYER_NORM_BF16_CAPABILITY_ID).unwrap()])
    );
    assert!(descriptor.supports_fixed_tensor_work());
    assert!(descriptor.accepted_weight_formats().is_empty());
}

#[test]
fn provider_geometry_couples_width_attribute_to_tensor_even_with_equal_byte_counts() {
    let input = value(
        "norm-input",
        ResolvedValueRole::Input,
        vec![2, 4],
        ElementType::Bf16,
    );
    let output = value(
        "norm-output",
        ResolvedValueRole::Output,
        vec![2, 4],
        ElementType::Bf16,
    );
    let bindings = vec![input, output];
    let checked = geometry(&bindings, &attributes(4)).unwrap();
    assert_eq!(
        (
            checked.rows(),
            checked.hidden_size(),
            checked.effective_epsilon()
        ),
        (2, 4, 0.75)
    );
    assert!(geometry(&bindings, &attributes(8)).is_err());
    let wrong_shape = vec![
        bindings[0].clone(),
        value(
            "norm-other-shape",
            ResolvedValueRole::Output,
            vec![4, 2],
            ElementType::Bf16,
        ),
    ];
    assert!(geometry(&wrong_shape, &attributes(4)).is_err());
    let wrong_dtype = vec![
        bindings[0].clone(),
        value(
            "norm-other-dtype",
            ResolvedValueRole::Output,
            vec![2, 4],
            ElementType::F16,
        ),
    ];
    assert!(geometry(&wrong_dtype, &attributes(4)).is_err());
    let mut wrong_epsilon = attributes(4);
    wrong_epsilon.insert(
        AttributeId::new("epsilon").unwrap(),
        SemanticValue::Unsigned(1),
    );
    assert!(geometry(&bindings, &wrong_epsilon).is_err());
    assert!(geometry(&bindings[..1], &attributes(4)).is_err());
}
