use super::*;
use crate::backend::reference::{
    reference_vnext_zero_centered_rms_norm_bf16_provider, ReferenceVNextComposition,
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
    BTreeMap::from([(
        AttributeId::new("epsilon").unwrap(),
        SemanticValue::Rational(ferrum_runtime::CanonicalRational::new(1, 1_000_000).unwrap()),
    )])
}
fn values(rows: u64, width: u64) -> Vec<ResolvedValueBinding> {
    vec![
        value(
            "zrm.x",
            ResolvedValueRole::Input,
            0,
            &[rows, width],
            ElementType::Bf16,
        ),
        value(
            "zrm.weight",
            ResolvedValueRole::Input,
            1,
            &[width],
            ElementType::Bf16,
        ),
        value(
            "zrm.y",
            ResolvedValueRole::Output,
            0,
            &[rows, width],
            ElementType::Bf16,
        ),
    ]
}
#[test]
fn distinct_provider_keeps_original_dense_bf16_weight_format() {
    let c =
        ReferenceVNextComposition::create(DeviceId::new("device.reference.zrm-provider").unwrap())
            .unwrap();
    let p = reference_vnext_zero_centered_rms_norm_bf16_provider(c.runtime()).unwrap();
    let d = p.descriptor();
    assert_eq!(
        d.operation_id().as_str(),
        ZERO_CENTERED_RMS_NORM_BF16_OPERATION_ID
    );
    assert_eq!(
        d.operation_fingerprint(),
        zero_centered_rms_norm_bf16_contract()
            .unwrap()
            .descriptor()
            .fingerprint()
            .unwrap()
    );
    assert!(d.supports_fixed_tensor_work());
    assert_eq!(
        d.accepted_weight_formats(),
        &BTreeSet::from([WeightFormatId::new(REFERENCE_DENSE_SAFETENSORS_FORMAT_ID).unwrap()])
    );
}
#[test]
fn shared_geometry_rejects_head_subdivision_and_logical_overflow_before_encoding() {
    let mut v = values(2, 8);
    let g = geometry(&v, &attrs()).unwrap();
    assert_eq!((g.rows(), g.columns()), (2, 8));
    v[1] = value(
        "bad.weight",
        ResolvedValueRole::Input,
        1,
        &[4],
        ElementType::Bf16,
    );
    assert!(geometry(&v, &attrs()).is_err());
    let mut a = attrs();
    a.insert(
        AttributeId::new("head_dim").unwrap(),
        SemanticValue::Unsigned(4),
    );
    assert!(geometry(&values(2, 8), &a).is_err());
    let rows = u64::MAX / 8 + 1;
    assert!(geometry(&values(rows, 1), &attrs()).is_err());
}
