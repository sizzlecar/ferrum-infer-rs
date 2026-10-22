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
    rows: u64,
    columns: u64,
) -> ResolvedValueBinding {
    let tensor = ResolvedTensorSpec::new(
        vec![rows, columns],
        ElementType::Bf16,
        ResolvedTensorLayout::Contiguous,
    )
    .unwrap();
    let storage = ResolvedValueStorage::single(
        ResourceId::new(format!("resource.{name}")).unwrap(),
        0,
        tensor.minimum_storage_bytes().unwrap(),
        ElementType::Bf16,
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
fn gather_attrs() -> BTreeMap<AttributeId, SemanticValue> {
    BTreeMap::from([
        (
            AttributeId::new("batch").unwrap(),
            SemanticValue::Unsigned(2),
        ),
        (
            AttributeId::new("column_start").unwrap(),
            SemanticValue::Unsigned(1),
        ),
        (
            AttributeId::new("column_count").unwrap(),
            SemanticValue::Unsigned(2),
        ),
        (
            AttributeId::new("run_lengths").unwrap(),
            SemanticValue::Integers(vec![2, 1]),
        ),
        (
            AttributeId::new("source_starts").unwrap(),
            SemanticValue::Integers(vec![2, 0]),
        ),
        (
            AttributeId::new("source_batch_strides").unwrap(),
            SemanticValue::Integers(vec![0, 1]),
        ),
        (
            AttributeId::new("source_row_strides").unwrap(),
            SemanticValue::Integers(vec![0, 0]),
        ),
    ])
}
#[test]
fn providers_have_independent_fixed_tensor_identities_and_dense_only_weight_support() {
    let c = ReferenceVNextComposition::create(
        DeviceId::new("device.reference.row-transport-providers").unwrap(),
    )
    .unwrap();
    let mut identities = BTreeSet::new();
    for kind in [
        ReferenceRowTransportKind::Gather,
        ReferenceRowTransportKind::Concat,
    ] {
        let p = ReferenceRowTransportProvider::new(c.runtime(), kind).unwrap();
        let d = p.descriptor();
        assert_eq!(d.operation_id().as_str(), kind.operation_id());
        assert!(d.supports_fixed_tensor_work());
        assert_eq!(
            d.accepted_weight_formats(),
            &BTreeSet::from([WeightFormatId::new(REFERENCE_DENSE_SAFETENSORS_FORMAT_ID).unwrap()])
        );
        assert!(d.accepted_quantization_formats().is_empty());
        identities.insert(d.operation_fingerprint().to_owned());
    }
    assert_eq!(identities.len(), 2);
}
#[test]
fn estimator_and_encoder_share_real_shape_and_closed_attribute_checks() {
    let gather = vec![
        value("g.source", ResolvedValueRole::Input, 0, 3, 4),
        value("g.output", ResolvedValueRole::Output, 0, 6, 2),
    ];
    assert!(matches!(
        geometry(ReferenceRowTransportKind::Gather, &gather, &gather_attrs()).unwrap(),
        Geometry::Gather(_)
    ));
    let mut wrong = gather.clone();
    wrong[1] = value("g.wrong", ResolvedValueRole::Output, 0, 3, 4);
    assert!(geometry(ReferenceRowTransportKind::Gather, &wrong, &gather_attrs()).is_err());
    let concat = vec![
        value("c.left", ResolvedValueRole::Input, 0, 4, 3),
        value("c.right", ResolvedValueRole::Input, 1, 2, 3),
        value("c.output", ResolvedValueRole::Output, 0, 6, 3),
    ];
    let mut attrs = BTreeMap::from([(
        AttributeId::new("batch").unwrap(),
        SemanticValue::Unsigned(2),
    )]);
    assert!(matches!(
        geometry(ReferenceRowTransportKind::Concat, &concat, &attrs).unwrap(),
        Geometry::Concat(_)
    ));
    attrs.insert(
        AttributeId::new("rows").unwrap(),
        SemanticValue::Unsigned(6),
    );
    assert!(geometry(ReferenceRowTransportKind::Concat, &concat, &attrs).is_err());
}
