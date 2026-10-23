use super::*;
use crate::backend::reference::{
    reference_vnext_head_rms_norm_bf16_provider, ReferenceVNextComposition,
};
use ferrum_runtime::{
    AliasPolicy, BufferUsage, CanonicalRational, DeviceId, ProgramValueId, ResolvedTensorLayout,
    ResolvedTensorSpec, ResolvedValueStorage, ResourceId, TensorAccess,
};

fn value(
    name: &str,
    role: ResolvedValueRole,
    ordinal: u32,
    dimensions: &[u64],
    dtype: ElementType,
) -> ResolvedValueBinding {
    let tensor =
        ResolvedTensorSpec::new(dimensions.to_vec(), dtype, ResolvedTensorLayout::Contiguous)
            .unwrap();
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
fn attributes(d: u64) -> BTreeMap<AttributeId, SemanticValue> {
    BTreeMap::from([
        (
            AttributeId::new("head_dim").unwrap(),
            SemanticValue::Unsigned(d),
        ),
        (
            AttributeId::new("epsilon").unwrap(),
            SemanticValue::Rational(CanonicalRational::new(1, 1_000_000).unwrap()),
        ),
    ])
}
fn bindings(rows: u64, columns: u64, head_dim: u64) -> Vec<ResolvedValueBinding> {
    vec![
        value(
            "head-rms.x",
            ResolvedValueRole::Input,
            0,
            &[rows, columns],
            ElementType::Bf16,
        ),
        value(
            "head-rms.gamma",
            ResolvedValueRole::Input,
            1,
            &[head_dim],
            ElementType::Bf16,
        ),
        value(
            "head-rms.y",
            ResolvedValueRole::Output,
            0,
            &[rows, columns],
            ElementType::Bf16,
        ),
    ]
}

#[test]
fn provider_has_independent_identity_and_accepts_real_dense_gamma_weights() {
    let composition = ReferenceVNextComposition::create(
        DeviceId::new("device.reference.head-rms-provider").unwrap(),
    )
    .unwrap();
    let provider = reference_vnext_head_rms_norm_bf16_provider(composition.runtime()).unwrap();
    let descriptor = provider.descriptor();
    assert_eq!(
        descriptor.operation_id().as_str(),
        HEAD_RMS_NORM_BF16_OPERATION_ID
    );
    assert_eq!(
        descriptor.operation_fingerprint(),
        head_rms_norm_bf16_contract()
            .unwrap()
            .descriptor()
            .fingerprint()
            .unwrap()
    );
    assert_eq!(
        descriptor.capabilities(),
        &BTreeSet::from([CapabilityId::new(HEAD_RMS_NORM_BF16_CAPABILITY_ID).unwrap()])
    );
    assert_eq!(
        descriptor.accepted_weight_formats(),
        &BTreeSet::from([WeightFormatId::new(REFERENCE_DENSE_SAFETENSORS_FORMAT_ID).unwrap()])
    );
    assert!(descriptor.accepted_quantization_formats().is_empty());
    assert!(descriptor.supports_fixed_tensor_work());
}

#[test]
fn estimator_and_encoding_share_checked_geometry_and_typed_attributes() {
    let valid = bindings(2, 256, 128);
    let checked = geometry(&valid, &attributes(128)).unwrap();
    assert_eq!(
        (
            checked.rows(),
            checked.columns(),
            checked.head_dim(),
            checked.heads()
        ),
        (2, 256, 128, 2)
    );
    for d in [0, 3, 64, 256] {
        assert!(geometry(&valid, &attributes(d)).is_err());
    }
    for changed in [
        value(
            "head-rms.bad-gamma",
            ResolvedValueRole::Input,
            1,
            &[1, 128],
            ElementType::Bf16,
        ),
        value(
            "head-rms.bad-gamma",
            ResolvedValueRole::Input,
            1,
            &[128],
            ElementType::F16,
        ),
        value(
            "head-rms.bad-gamma",
            ResolvedValueRole::Input,
            0,
            &[128],
            ElementType::Bf16,
        ),
    ] {
        assert!(geometry(
            &[valid[0].clone(), changed, valid[2].clone()],
            &attributes(128)
        )
        .is_err());
    }
    let wrong_output = value(
        "head-rms.bad-output",
        ResolvedValueRole::Output,
        0,
        &[4, 128],
        ElementType::Bf16,
    );
    assert!(geometry(
        &[valid[0].clone(), valid[1].clone(), wrong_output],
        &attributes(128)
    )
    .is_err());
    assert!(geometry(&valid[..2], &attributes(128)).is_err());
    for (name, replacement) in [
        ("epsilon", SemanticValue::Unsigned(1)),
        ("head_dim", SemanticValue::Integer(128)),
        ("gamma_offset", SemanticValue::Unsigned(1)),
    ] {
        let mut attrs = attributes(128);
        attrs.insert(AttributeId::new(name).unwrap(), replacement);
        assert!(geometry(&valid, &attrs).is_err());
    }
}

#[test]
fn representable_tensor_bytes_do_not_bypass_logical_work_overflow() {
    // D=1 performs seven scalar operations per element, so a valid BF16
    // byte extent may still exceed u64 attribution. No payload is allocated.
    let rows = u64::MAX / 7 + 1;
    let values = bindings(rows, 1, 1);
    assert_eq!(
        values[0].tensor().minimum_storage_bytes().unwrap(),
        rows * 2
    );
    assert!(geometry(&values, &attributes(1)).is_err());
}
