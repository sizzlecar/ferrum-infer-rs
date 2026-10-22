use super::*;
use crate::backend::reference::ReferenceVNextComposition;
fn value(
    name: &str,
    role: ResolvedValueRole,
    ordinal: u32,
    dims: &[u64],
    dtype: ElementType,
) -> ResolvedValueBinding {
    let tensor =
        ResolvedTensorSpec::new(dims.to_vec(), dtype, ResolvedTensorLayout::Contiguous).unwrap();
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
fn attrs(kind: SpatialKind) -> BTreeMap<AttributeId, SemanticValue> {
    match kind {
        SpatialKind::SiluNchw | SpatialKind::ResidualAddNchw => BTreeMap::new(),
        SpatialKind::Conv2d => BTreeMap::from([
            (
                AttributeId::new("padding").unwrap(),
                SemanticValue::Integers(vec![1, 1, 1, 1]),
            ),
            (
                AttributeId::new("stride").unwrap(),
                SemanticValue::Integers(vec![1, 1]),
            ),
            (
                AttributeId::new("dilation").unwrap(),
                SemanticValue::Integers(vec![1, 1]),
            ),
            (
                AttributeId::new("groups").unwrap(),
                SemanticValue::Unsigned(1),
            ),
            (
                AttributeId::new("padding_mode").unwrap(),
                SemanticValue::Text("explicit_zero".into()),
            ),
        ]),
        SpatialKind::ChannelL2Norm => BTreeMap::from([
            (
                AttributeId::new("epsilon").unwrap(),
                SemanticValue::Rational(CanonicalRational::new(1, 1_000_000_000_000).unwrap()),
            ),
            (
                AttributeId::new("scale").unwrap(),
                SemanticValue::Rational(CanonicalRational::new(2, 1).unwrap()),
            ),
        ]),
    }
}
#[test]
fn providers_have_separate_contracts_fingerprints_and_dense_weight_capabilities() {
    let c = ReferenceVNextComposition::create(
        DeviceId::new("device.reference.spatial-contracts").unwrap(),
    )
    .unwrap();
    let conv = ReferenceSpatialProvider::new(c.runtime(), SpatialKind::Conv2d).unwrap();
    let norm = ReferenceSpatialProvider::new(c.runtime(), SpatialKind::ChannelL2Norm).unwrap();
    let silu = ReferenceSpatialProvider::new(c.runtime(), SpatialKind::SiluNchw).unwrap();
    let add = ReferenceSpatialProvider::new(c.runtime(), SpatialKind::ResidualAddNchw).unwrap();
    for (p, k) in [
        (&conv, SpatialKind::Conv2d),
        (&norm, SpatialKind::ChannelL2Norm),
        (&silu, SpatialKind::SiluNchw),
        (&add, SpatialKind::ResidualAddNchw),
    ] {
        let d = p.descriptor();
        assert_eq!(d.operation_id().as_str(), k.operation());
        assert_eq!(
            d.operation_fingerprint(),
            k.contract().unwrap().descriptor().fingerprint().unwrap()
        );
        assert_eq!(
            d.capabilities(),
            &BTreeSet::from([CapabilityId::new(k.capability()).unwrap()])
        );
        assert_eq!(
            d.accepted_weight_formats(),
            &BTreeSet::from([WeightFormatId::new(REFERENCE_DENSE_SAFETENSORS_FORMAT_ID).unwrap()])
        );
        assert!(d.accepted_quantization_formats().is_empty());
        assert!(d.supports_fixed_tensor_work());
    }
    assert_ne!(
        conv.descriptor().provider_implementation_fingerprint(),
        norm.descriptor().provider_implementation_fingerprint()
    );
}
#[test]
fn nchw_scalar_lowering_keeps_exact_ports_and_checks_work_before_encode() {
    let x = value(
        "nchw.x",
        ResolvedValueRole::Input,
        0,
        &[2, 3, 4, 5],
        ElementType::Bf16,
    );
    let y = value(
        "nchw.y",
        ResolvedValueRole::Output,
        0,
        &[2, 3, 4, 5],
        ElementType::Bf16,
    );
    let Geometry::SiluNchw(g) = geometry(
        SpatialKind::SiluNchw,
        &[x.clone(), y.clone()],
        &BTreeMap::new(),
    )
    .unwrap() else {
        panic!("wrong geometry kind")
    };
    let shape = flat_shape(g).unwrap();
    assert_eq!((shape.rows, shape.width), (24, 5));
    let update = value(
        "nchw.u",
        ResolvedValueRole::Input,
        1,
        &[2, 3, 4, 5],
        ElementType::Bf16,
    );
    assert!(geometry(
        SpatialKind::ResidualAddNchw,
        &[x.clone(), update, y.clone()],
        &BTreeMap::new()
    )
    .is_ok());
    let wrong = value(
        "nchw.u",
        ResolvedValueRole::Input,
        1,
        &[2, 3, 5, 4],
        ElementType::Bf16,
    );
    assert!(geometry(
        SpatialKind::ResidualAddNchw,
        &[x.clone(), wrong, y],
        &BTreeMap::new()
    )
    .is_err());
    let flat = value(
        "nchw.y",
        ResolvedValueRole::Output,
        0,
        &[24, 5],
        ElementType::Bf16,
    );
    assert!(geometry(SpatialKind::SiluNchw, &[x, flat], &BTreeMap::new()).is_err());
}
#[test]
fn shared_geometry_rejects_incorrect_oi_hw_bias_and_gamma_rank_before_encode() {
    let x = value(
        "conv.x",
        ResolvedValueRole::Input,
        0,
        &[2, 4, 3, 5],
        ElementType::Bf16,
    );
    let w = value(
        "conv.w",
        ResolvedValueRole::Input,
        1,
        &[6, 4, 3, 3],
        ElementType::Bf16,
    );
    let b = value(
        "conv.b",
        ResolvedValueRole::Input,
        2,
        &[6],
        ElementType::Bf16,
    );
    let y = value(
        "conv.y",
        ResolvedValueRole::Output,
        0,
        &[2, 6, 3, 5],
        ElementType::Bf16,
    );
    let values = vec![x, w, b, y];
    assert!(geometry(SpatialKind::Conv2d, &values, &attrs(SpatialKind::Conv2d)).is_ok());
    for bad in [
        value(
            "conv.bad",
            ResolvedValueRole::Input,
            2,
            &[1, 6],
            ElementType::Bf16,
        ),
        value(
            "conv.bad",
            ResolvedValueRole::Input,
            2,
            &[6],
            ElementType::F32,
        ),
    ] {
        let mut changed = values.clone();
        changed[2] = bad;
        assert!(geometry(SpatialKind::Conv2d, &changed, &attrs(SpatialKind::Conv2d)).is_err());
    }
    let n = vec![
        value(
            "norm.x",
            ResolvedValueRole::Input,
            0,
            &[2, 4, 3, 5],
            ElementType::Bf16,
        ),
        value(
            "norm.gamma",
            ResolvedValueRole::Input,
            1,
            &[4, 1, 1, 1],
            ElementType::Bf16,
        ),
        value(
            "norm.y",
            ResolvedValueRole::Output,
            0,
            &[2, 4, 3, 5],
            ElementType::Bf16,
        ),
    ];
    assert!(geometry(
        SpatialKind::ChannelL2Norm,
        &n,
        &attrs(SpatialKind::ChannelL2Norm)
    )
    .is_ok());
    let mut bad = n.clone();
    bad[1] = value(
        "norm.gamma",
        ResolvedValueRole::Input,
        1,
        &[4, 1, 1],
        ElementType::Bf16,
    );
    assert!(geometry(
        SpatialKind::ChannelL2Norm,
        &bad,
        &attrs(SpatialKind::ChannelL2Norm)
    )
    .is_err());
    let mut wrong = attrs(SpatialKind::ChannelL2Norm);
    wrong.insert(
        AttributeId::new("channels").unwrap(),
        SemanticValue::Unsigned(4),
    );
    assert!(geometry(SpatialKind::ChannelL2Norm, &n, &wrong).is_err());
}
