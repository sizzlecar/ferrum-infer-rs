use super::*;
use crate::backend::reference::{
    reference_vnext_structured_attention_bf16_provider, ReferenceVNextComposition,
};
use ferrum_runtime::{
    AliasPolicy, BufferUsage, DeviceId, ProgramValueId, ResolvedTensorLayout, ResolvedTensorSpec,
    ResolvedValueStorage, ResourceId, TensorAccess,
};

fn value(
    name: &str,
    role: ResolvedValueRole,
    ordinal: u32,
    shape: &[u64],
    dtype: ElementType,
) -> ResolvedValueBinding {
    let tensor =
        ResolvedTensorSpec::new(shape.to_vec(), dtype, ResolvedTensorLayout::Contiguous).unwrap();
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
fn attrs() -> BTreeMap<AttributeId, SemanticValue> {
    let mut a = BTreeMap::from([
        (
            AttributeId::new("sequence_length").unwrap(),
            SemanticValue::Unsigned(2),
        ),
        (
            AttributeId::new("head_dim").unwrap(),
            SemanticValue::Unsigned(2),
        ),
        (
            AttributeId::new("scale").unwrap(),
            SemanticValue::Rational(ferrum_runtime::CanonicalRational::new(1, 1).unwrap()),
        ),
    ]);
    for (name, x) in [
        ("segment_lengths", vec![2]),
        ("segment_bidirectional", vec![1]),
        ("key_run_counts", vec![1, 1]),
        ("key_run_lengths", vec![2, 2]),
        ("key_run_valid", vec![1, 1]),
    ] {
        a.insert(AttributeId::new(name).unwrap(), SemanticValue::Integers(x));
    }
    a
}
fn values() -> Vec<ResolvedValueBinding> {
    vec![
        value(
            "attention.q",
            ResolvedValueRole::Input,
            0,
            &[4, 4],
            ElementType::Bf16,
        ),
        value(
            "attention.k",
            ResolvedValueRole::Input,
            1,
            &[4, 4],
            ElementType::Bf16,
        ),
        value(
            "attention.v",
            ResolvedValueRole::Input,
            2,
            &[4, 4],
            ElementType::Bf16,
        ),
        value(
            "attention.y",
            ResolvedValueRole::Output,
            0,
            &[4, 4],
            ElementType::Bf16,
        ),
    ]
}
#[test]
fn descriptor_has_independent_fixed_tensor_capability_without_weight_formats() {
    let c = ReferenceVNextComposition::create(
        DeviceId::new("device.reference.attention-provider").unwrap(),
    )
    .unwrap();
    let p = reference_vnext_structured_attention_bf16_provider(c.runtime()).unwrap();
    let d = p.descriptor();
    assert_eq!(
        d.operation_id().as_str(),
        STRUCTURED_ATTENTION_BF16_OPERATION_ID
    );
    assert_eq!(
        d.operation_fingerprint(),
        structured_attention_bf16_contract()
            .unwrap()
            .descriptor()
            .fingerprint()
            .unwrap()
    );
    assert_eq!(
        d.capabilities(),
        &BTreeSet::from([CapabilityId::new(STRUCTURED_ATTENTION_BF16_CAPABILITY_ID).unwrap()])
    );
    assert!(d.accepted_weight_formats().is_empty());
    assert!(d.accepted_quantization_formats().is_empty());
    assert!(d.supports_fixed_tensor_work());
}
#[test]
fn shared_estimator_and_encoder_geometry_checks_actual_values_and_closed_mask_attributes() {
    let v = values();
    let g = geometry(&v, &attrs()).unwrap();
    assert_eq!(
        (g.batch(), g.heads(), g.sequence_length(), g.head_dim()),
        (2, 2, 2, 2)
    );
    for replacement in [
        value(
            "bad.v",
            ResolvedValueRole::Input,
            2,
            &[2, 8],
            ElementType::Bf16,
        ),
        value(
            "bad.v",
            ResolvedValueRole::Input,
            2,
            &[4, 4],
            ElementType::F16,
        ),
        value(
            "bad.v",
            ResolvedValueRole::Input,
            1,
            &[4, 4],
            ElementType::Bf16,
        ),
    ] {
        let mut bad = v.clone();
        bad[2] = replacement;
        assert!(geometry(&bad, &attrs()).is_err());
    }
    assert!(geometry(&v[..3], &attrs()).is_err());
    for (name, value) in [
        ("key_run_valid", SemanticValue::Integers(vec![1, 2])),
        ("segment_lengths", SemanticValue::Integers(vec![3])),
        ("scale", SemanticValue::Unsigned(1)),
        ("causal", SemanticValue::Unsigned(1)),
    ] {
        let mut a = attrs();
        a.insert(AttributeId::new(name).unwrap(), value);
        assert!(geometry(&v, &a).is_err());
    }
}
