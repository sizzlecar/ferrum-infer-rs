use super::*;
use crate::backend::reference::{
    reference_vnext_gelu_tanh_bf16_provider, ReferenceVNextComposition,
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

fn values(rows: u64, width: u64) -> Vec<ResolvedValueBinding> {
    vec![
        value(
            "gelu.x",
            ResolvedValueRole::Input,
            0,
            &[rows, width],
            ElementType::Bf16,
        ),
        value(
            "gelu.y",
            ResolvedValueRole::Output,
            0,
            &[rows, width],
            ElementType::Bf16,
        ),
    ]
}
#[test]
fn gelu_provider_has_independent_contract_identity_and_no_weight_formats() {
    let c =
        ReferenceVNextComposition::create(DeviceId::new("device.reference.gelu-provider").unwrap())
            .unwrap();
    let p = reference_vnext_gelu_tanh_bf16_provider(c.runtime()).unwrap();
    let d = p.descriptor();
    assert_eq!(d.operation_id().as_str(), GELU_TANH_BF16_OPERATION_ID);
    assert_eq!(
        d.operation_fingerprint(),
        gelu_tanh_bf16_contract()
            .unwrap()
            .descriptor()
            .fingerprint()
            .unwrap()
    );
    assert!(d.accepted_weight_formats().is_empty());
    assert!(d.accepted_quantization_formats().is_empty());
    assert!(d.supports_fixed_tensor_work());
}
#[test]
fn geometry_rejects_unknown_attributes_wrong_shape_and_logical_overflow() {
    let v = values(2, 3);
    geometry(&v, &BTreeMap::new()).unwrap();
    let mut bad = v.clone();
    bad[1] = value(
        "bad.y",
        ResolvedValueRole::Output,
        0,
        &[3, 2],
        ElementType::Bf16,
    );
    assert!(geometry(&bad, &BTreeMap::new()).is_err());
    assert!(geometry(
        &v,
        &BTreeMap::from([(
            AttributeId::new("approximate").unwrap(),
            SemanticValue::Text("tanh".into())
        )])
    )
    .is_err());
    assert!(geometry(&values(u64::MAX / 9 + 1, 1), &BTreeMap::new()).is_err());
}
