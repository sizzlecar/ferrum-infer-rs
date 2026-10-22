use super::*;
use crate::backend::reference::ReferenceVNextComposition;
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
#[test]
fn independent_fixed_tensor_provider_has_no_gate_or_weight_format() {
    let c = ReferenceVNextComposition::create(
        DeviceId::new("device.reference.plain-add-provider").unwrap(),
    )
    .unwrap();
    let p = ReferencePlainResidualAddProvider::new(c.runtime()).unwrap();
    let d = p.descriptor();
    assert_eq!(
        d.operation_id().as_str(),
        PLAIN_RESIDUAL_ADD_BF16_OPERATION_ID
    );
    assert_eq!(
        d.operation_fingerprint(),
        plain_residual_add_bf16_contract()
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
fn actual_second_input_shape_and_closed_attributes_are_shared_estimator_encode_checks() {
    let x = value(
        "add.x",
        ResolvedValueRole::Input,
        0,
        &[2, 4],
        ElementType::Bf16,
    );
    let u = value(
        "add.u",
        ResolvedValueRole::Input,
        1,
        &[2, 4],
        ElementType::Bf16,
    );
    let y = value(
        "add.y",
        ResolvedValueRole::Output,
        0,
        &[2, 4],
        ElementType::Bf16,
    );
    let values = [x.clone(), u, y.clone()];
    assert_eq!(geometry(&values, &BTreeMap::new()).unwrap().width(), 4);
    for bad in [
        value(
            "bad.u",
            ResolvedValueRole::Input,
            1,
            &[1, 8],
            ElementType::Bf16,
        ),
        value(
            "bad.u",
            ResolvedValueRole::Input,
            1,
            &[2, 4],
            ElementType::F16,
        ),
        value(
            "bad.u",
            ResolvedValueRole::Input,
            0,
            &[2, 4],
            ElementType::Bf16,
        ),
    ] {
        assert!(geometry(&[x.clone(), bad, y.clone()], &BTreeMap::new()).is_err());
    }
    assert!(geometry(
        &values,
        &BTreeMap::from([(
            AttributeId::new("broadcast").unwrap(),
            SemanticValue::Text("row".into())
        )])
    )
    .is_err());
    assert!(geometry(&values[..2], &BTreeMap::new()).is_err());
}
