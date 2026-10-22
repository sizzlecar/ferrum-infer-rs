use super::super::ReferenceVNextComposition;
use super::*;
fn value(name: &str, role: ResolvedValueRole, ordinal: u32, dims: &[u64]) -> ResolvedValueBinding {
    let tensor = ResolvedTensorSpec::new(
        dims.to_vec(),
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
#[test]
fn providers_bind_distinct_contracts_profiles_and_weight_acceptance() {
    let c = ReferenceVNextComposition::create(
        DeviceId::new("device.reference.spatial_decoder").unwrap(),
    )
    .unwrap();
    let mut fingerprints = BTreeSet::new();
    for kind in [
        DecoderKind::Norm,
        DecoderKind::Attention,
        DecoderKind::Upsample,
    ] {
        let p = ReferenceSpatialDecoderProvider::new(c.runtime(), kind).unwrap();
        let d = p.descriptor();
        assert_eq!(d.operation_id().as_str(), kind.operation());
        assert_eq!(
            d.operation_fingerprint(),
            kind.contract().unwrap().descriptor().fingerprint().unwrap()
        );
        assert_eq!(
            d.capabilities(),
            &BTreeSet::from([CapabilityId::new(kind.capability()).unwrap()])
        );
        assert_eq!(
            d.accepted_weight_formats().is_empty(),
            !matches!(kind, DecoderKind::Norm)
        );
        assert!(fingerprints.insert(d.provider_implementation_fingerprint().to_owned()));
    }
}
#[test]
fn provider_geometry_rechecks_packed_channels_rank_three_gamma_and_tile_source() {
    let mut attrs = BTreeMap::from([(
        AttributeId::new("scale").unwrap(),
        SemanticValue::Rational(CanonicalRational::new(1, 1).unwrap()),
    )]);
    let attention = [
        value("qkv", ResolvedValueRole::Input, 0, &[2, 3456, 3, 5]),
        value("y", ResolvedValueRole::Output, 0, &[2, 1152, 3, 5]),
    ];
    assert!(geometry(DecoderKind::Attention, &attention, &attrs).is_ok());
    let wrong = [
        value("qkv", ResolvedValueRole::Input, 0, &[2, 3456, 5, 3]),
        value("y", ResolvedValueRole::Output, 0, &[2, 1152, 3, 5]),
    ];
    assert!(geometry(DecoderKind::Attention, &wrong, &attrs).is_err());
    attrs.insert(
        AttributeId::new("epsilon").unwrap(),
        SemanticValue::Rational(CanonicalRational::new(1, 1_000_000_000_000).unwrap()),
    );
    for (dims, accepted) in [(&[3, 1, 1][..], true), (&[3, 1, 1, 1][..], false)] {
        let values = [
            value("x", ResolvedValueRole::Input, 0, &[1, 3, 2, 5]),
            value("gamma", ResolvedValueRole::Input, 1, dims),
            value("y", ResolvedValueRole::Output, 0, &[1, 3, 2, 5]),
        ];
        assert_eq!(
            geometry(DecoderKind::Norm, &values, &attrs).is_ok(),
            accepted
        );
    }
    let mut attrs = BTreeMap::from([
        (
            AttributeId::new("scale_h").unwrap(),
            SemanticValue::Unsigned(2),
        ),
        (
            AttributeId::new("scale_w").unwrap(),
            SemanticValue::Unsigned(2),
        ),
        (
            AttributeId::new("channel_stride").unwrap(),
            SemanticValue::Unsigned(2),
        ),
        (
            AttributeId::new("channel_offsets").unwrap(),
            SemanticValue::Integers(vec![1; 4]),
        ),
    ]);
    let values = [
        value("x", ResolvedValueRole::Input, 0, &[1, 4, 2, 5]),
        value("y", ResolvedValueRole::Output, 0, &[1, 2, 4, 10]),
    ];
    assert!(geometry(DecoderKind::Upsample, &values, &attrs).is_ok());
    attrs.insert(
        AttributeId::new("channel_offsets").unwrap(),
        SemanticValue::Integers(vec![2; 4]),
    );
    assert!(geometry(DecoderKind::Upsample, &values, &attrs).is_err());
}
