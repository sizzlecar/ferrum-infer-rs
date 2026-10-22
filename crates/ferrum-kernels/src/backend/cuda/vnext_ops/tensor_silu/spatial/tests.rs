use super::*;
use ferrum_runtime::{
    AliasPolicy, BufferUsage, CanonicalRational, ProgramValueId, ResolvedTensorLayout,
    ResolvedTensorSpec, ResolvedValueStorage, ResourceId, TensorAccess,
};

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
fn input(name: &str, ordinal: u32, dims: &[u64]) -> ResolvedValueBinding {
    value(
        name,
        ResolvedValueRole::Input,
        ordinal,
        dims,
        ElementType::Bf16,
    )
}
fn output(dims: &[u64]) -> ResolvedValueBinding {
    value("y", ResolvedValueRole::Output, 0, dims, ElementType::Bf16)
}
fn attributes() -> BTreeMap<AttributeId, SemanticValue> {
    BTreeMap::from([
        (
            AttributeId::new("epsilon").unwrap(),
            SemanticValue::Rational(CanonicalRational::new(1, 1_000_000_000_000).unwrap()),
        ),
        (
            AttributeId::new("scale").unwrap(),
            SemanticValue::Rational(CanonicalRational::new(32, 1).unwrap()),
        ),
    ])
}

#[test]
fn image_norm_accepts_actual_rank_three_gamma_and_keeps_rank_four_contract_distinct() {
    let x = [2, 1152, 3, 5];
    let image = [
        input("x", 0, &x),
        input("gamma", 1, &[1152, 1, 1]),
        output(&x),
    ];
    let s = geometry(SpatialKind::ChannelL2Image, &image, &attributes()).unwrap();
    assert_eq!(
        (s.channels, s.tree_width, s.elements, s.operations),
        (1152, 2048, 34560, 207360)
    );
    assert!(geometry(SpatialKind::ChannelL2, &image, &attributes()).is_err());
    let volume = [
        input("x", 0, &x),
        input("gamma", 1, &[1152, 1, 1, 1]),
        output(&x),
    ];
    assert!(geometry(SpatialKind::ChannelL2Image, &volume, &attributes()).is_err());
    assert!(geometry(SpatialKind::ChannelL2, &volume, &attributes()).is_ok());
    assert_ne!(
        SpatialKind::ChannelL2.profile(),
        SpatialKind::ChannelL2Image.profile()
    );
    assert_ne!(
        SpatialKind::ChannelL2.id(),
        SpatialKind::ChannelL2Image.id()
    );
}

#[test]
fn real_non_power_of_two_channels_use_bounded_padded_tree() {
    let s = geometry(
        SpatialKind::ChannelL2,
        &[
            input("x", 0, &[2, 1152, 3, 5]),
            input("gamma", 1, &[1152, 1, 1, 1]),
            output(&[2, 1152, 3, 5]),
        ],
        &attributes(),
    )
    .unwrap();
    assert_eq!(
        (s.channels, s.positions, s.spatial, s.tree_width),
        (1152, 30, 15, 2048)
    );
    assert_eq!((s.elements, s.bytes, s.operations), (34560, 69120, 207360));
    assert_eq!(s.scale, 32.0);
    assert!(geometry(
        SpatialKind::ChannelL2,
        &[
            input("x", 0, &[1, 4097, 1, 1]),
            input("gamma", 1, &[4097, 1, 1, 1]),
            output(&[1, 4097, 1, 1])
        ],
        &attributes()
    )
    .is_err());
}

#[test]
fn no_implicit_gamma_rank_conversion_or_spatial_broadcast() {
    let dims = [1, 3, 2, 4];
    assert!(geometry(
        SpatialKind::ChannelL2,
        &[
            input("x", 0, &dims),
            input("gamma", 1, &[3, 1, 1]),
            output(&dims)
        ],
        &attributes()
    )
    .is_err());
    assert!(geometry(
        SpatialKind::ResidualAdd,
        &[
            input("x", 0, &dims),
            input("u", 1, &[1, 3, 4, 2]),
            output(&dims)
        ],
        &BTreeMap::new()
    )
    .is_err());
    assert!(geometry(
        SpatialKind::Silu,
        &[
            input("x", 0, &dims),
            value("y", ResolvedValueRole::Output, 0, &dims, ElementType::F16)
        ],
        &BTreeMap::new()
    )
    .is_err());
    assert!(geometry(
        SpatialKind::Silu,
        &[input("x", 0, &[6, 4]), output(&[6, 4])],
        &BTreeMap::new()
    )
    .is_err());
}

#[test]
fn pointwise_shapes_and_work_stay_nchw_and_reject_attributes() {
    let dims = [2, 3, 4, 5];
    let silu = [input("x", 0, &dims), output(&dims)];
    assert_eq!(
        geometry(SpatialKind::Silu, &silu, &BTreeMap::new())
            .unwrap()
            .operations,
        480
    );
    assert!(geometry(SpatialKind::Silu, &silu, &attributes()).is_err());
    let add = [input("x", 0, &dims), input("u", 1, &dims), output(&dims)];
    assert_eq!(
        geometry(SpatialKind::ResidualAdd, &add, &BTreeMap::new())
            .unwrap()
            .operations,
        120
    );
}
