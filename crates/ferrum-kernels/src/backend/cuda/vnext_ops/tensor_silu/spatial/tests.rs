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

#[test]
fn channel_norm_spatial_lanes_cover_batch_tail_and_keep_small_spatial_launch() {
    for spatial in [1, 31, 32, 33, 131, u64::from(THREADS) * MAX_BLOCKS + 1] {
        let dims = [2, 144, 1, spatial];
        let s = geometry(
            SpatialKind::ChannelL2Image,
            &[
                input("x", 0, &dims),
                input("gamma", 1, &[144, 1, 1]),
                output(&dims),
            ],
            &attributes(),
        )
        .unwrap();
        let launch = s.channel_norm_launch();
        assert_eq!(launch.block_dim, (THREADS, 1, 1));
        if spatial < 32 {
            assert_eq!(launch.grid_dim.0, (2 * spatial) as u32);
            assert_eq!(launch.shared_mem_bytes, 256 * 4);
        } else {
            assert_eq!(launch.shared_mem_bytes, 0);
            assert_eq!(
                u64::from(launch.grid_dim.0),
                (2 * spatial).div_ceil(u64::from(THREADS)).min(MAX_BLOCKS)
            );
            let stride = u64::from(launch.grid_dim.0) * u64::from(THREADS);
            // Every pixel has one block/lane/iteration decomposition, including
            // a partial final block and a capped grid's second iteration.
            for p in [0, spatial - 1, spatial, 2 * spatial - 1] {
                let first = p % stride;
                let iteration = p / stride;
                assert!(first / u64::from(THREADS) < u64::from(launch.grid_dim.0));
                assert_eq!(first + iteration * stride, p);
            }
        }
    }
}

#[test]
fn channel_norm_carry_preserves_padded_adjacent_tree_not_sequential_sum() {
    fn adjacent(values: &[f32]) -> f32 {
        if values.len() == 1 {
            values[0]
        } else {
            let half = values.len() / 2;
            adjacent(&values[..half]) + adjacent(&values[half..])
        }
    }
    fn carry(values: &[f32]) -> f32 {
        let mut slots = [0.0_f32; 12];
        let mut root = 0.0_f32;
        for (i, &leaf) in values.iter().enumerate() {
            root = leaf;
            for (level, slot) in slots.iter_mut().enumerate() {
                if i & (1 << level) == 0 {
                    *slot = root;
                    break;
                }
                root = *slot + root;
            }
        }
        root
    }
    // These are exact BF16 input squares (4096^2 and 1^2). The old adjacent
    // tree preserves the pair of unit squares that a sequential sum loses.
    let rounding = [16_777_216.0_f32, 1.0, 1.0, 1.0];
    assert_eq!(adjacent(&rounding), 16_777_218.0);
    assert_eq!(rounding.into_iter().sum::<f32>(), 16_777_216.0);
    for channels in [1_usize, 3, 4, 96, 144, 288, 4096] {
        let mut leaves = vec![0.0_f32; channels.next_power_of_two()];
        for (i, value) in leaves[..channels].iter_mut().enumerate() {
            *value = rounding[i % rounding.len()];
        }
        assert_eq!(carry(&leaves).to_bits(), adjacent(&leaves).to_bits());
        leaves.fill(0.0);
        assert_eq!(carry(&leaves).to_bits(), 0);
        leaves[channels - 1] = f32::INFINITY;
        assert_eq!(carry(&leaves), f32::INFINITY);
        leaves[0] = f32::NAN;
        assert!(carry(&leaves).is_nan());
    }
}
