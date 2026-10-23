use super::*;
use crate::backend::reference::{
    reference_vnext_half_split_rope_bf16_provider, ReferenceVNextComposition,
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
fn attrs(d: u64) -> BTreeMap<AttributeId, SemanticValue> {
    BTreeMap::from([(
        AttributeId::new("head_dim").unwrap(),
        SemanticValue::Unsigned(d),
    )])
}
fn values(rows: u64, columns: u64, d: u64) -> Vec<ResolvedValueBinding> {
    vec![
        value(
            "rope.x",
            ResolvedValueRole::Input,
            0,
            &[rows, columns],
            ElementType::Bf16,
        ),
        value(
            "rope.freq",
            ResolvedValueRole::Input,
            1,
            &[rows, d],
            ElementType::Bf16,
        ),
        value(
            "rope.sin",
            ResolvedValueRole::Input,
            2,
            &[rows, d],
            ElementType::Bf16,
        ),
        value(
            "rope.y",
            ResolvedValueRole::Output,
            0,
            &[rows, columns],
            ElementType::Bf16,
        ),
    ]
}
#[test]
fn provider_identity_is_independent_and_does_not_require_weight_formats() {
    let c =
        ReferenceVNextComposition::create(DeviceId::new("device.reference.rope-provider").unwrap())
            .unwrap();
    let p = reference_vnext_half_split_rope_bf16_provider(c.runtime()).unwrap();
    let d = p.descriptor();
    assert_eq!(d.operation_id().as_str(), HALF_SPLIT_ROPE_BF16_OPERATION_ID);
    assert_eq!(
        d.operation_fingerprint(),
        half_split_rope_bf16_contract()
            .unwrap()
            .descriptor()
            .fingerprint()
            .unwrap()
    );
    assert_eq!(
        d.capabilities(),
        &BTreeSet::from([CapabilityId::new(HALF_SPLIT_ROPE_BF16_CAPABILITY_ID).unwrap()])
    );
    assert!(d.accepted_weight_formats().is_empty());
    assert!(d.accepted_quantization_formats().is_empty());
    assert!(d.supports_fixed_tensor_work());
}
#[test]
fn estimator_and_encoder_geometry_reject_frequency_mismatches_and_unknown_attributes() {
    let v = values(4, 8, 4);
    let g = geometry(&v, &attrs(4)).unwrap();
    assert_eq!((g.rows(), g.heads(), g.head_dim()), (4, 2, 4));
    for d in [0, 3, 8] {
        assert!(geometry(&v, &attrs(d)).is_err());
    }
    for f in [
        value(
            "bad.freq",
            ResolvedValueRole::Input,
            1,
            &[2, 2, 2],
            ElementType::Bf16,
        ),
        value(
            "bad.freq",
            ResolvedValueRole::Input,
            1,
            &[2, 1, 4],
            ElementType::F32,
        ),
        value(
            "bad.freq",
            ResolvedValueRole::Input,
            0,
            &[2, 2, 2],
            ElementType::F32,
        ),
    ] {
        assert!(geometry(&[v[0].clone(), f, v[2].clone(), v[3].clone()], &attrs(4)).is_err());
    }
    let mut bad = attrs(4);
    bad.insert(
        AttributeId::new("theta").unwrap(),
        SemanticValue::Unsigned(10000),
    );
    assert!(geometry(&v, &bad).is_err());
    assert!(geometry(&v[..2], &attrs(4)).is_err());
}
#[test]
fn logical_work_overflow_is_rejected_even_when_tensor_bytes_are_representable() {
    let rows = u64::MAX / 7 + 1;
    let v = values(rows, 2, 2);
    assert_eq!(v[0].tensor().minimum_storage_bytes().unwrap(), rows * 4);
    assert!(geometry(&v, &attrs(2)).is_err());
}
