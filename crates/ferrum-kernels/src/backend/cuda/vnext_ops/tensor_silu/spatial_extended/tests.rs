use super::*;
use ferrum_runtime::{
    AliasPolicy, BufferUsage, CanonicalRational, DeviceClass, DeviceId, ProgramValueId,
    ResolvedTensorLayout, ResolvedTensorSpec, ResolvedValueStorage, ResourceId, TensorAccess,
};

fn value(
    name: &str,
    role: ResolvedValueRole,
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
        0,
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
fn pair(x: &[u64], y: &[u64]) -> [ResolvedValueBinding; 2] {
    [
        value("x", ResolvedValueRole::Input, x, ElementType::Bf16),
        value("y", ResolvedValueRole::Output, y, ElementType::Bf16),
    ]
}
fn scale() -> BTreeMap<AttributeId, SemanticValue> {
    BTreeMap::from([(
        AttributeId::new("scale").unwrap(),
        SemanticValue::Rational(CanonicalRational::new(1, 32).unwrap()),
    )])
}
fn gather_attrs(
    sh: u64,
    sw: u64,
    stride: u64,
    offsets: &[i64],
) -> BTreeMap<AttributeId, SemanticValue> {
    BTreeMap::from([
        (
            AttributeId::new("scale_h").unwrap(),
            SemanticValue::Unsigned(sh),
        ),
        (
            AttributeId::new("scale_w").unwrap(),
            SemanticValue::Unsigned(sw),
        ),
        (
            AttributeId::new("channel_stride").unwrap(),
            SemanticValue::Unsigned(stride),
        ),
        (
            AttributeId::new("channel_offsets").unwrap(),
            SemanticValue::Integers(offsets.to_vec()),
        ),
    ])
}

#[test]
fn single_head_1152_channels_has_one_bounded_reduction_and_exact_ports() {
    let s = geometry(
        Kind::Attention,
        &pair(&[2, 3456, 3, 5], &[2, 1152, 3, 5]),
        &scale(),
    )
    .unwrap();
    let NativeShape::Attention {
        positions,
        channels,
        spatial,
        tree_width,
        scale: effective_scale,
    } = s.native
    else {
        panic!("wrong kernel")
    };
    assert_eq!(
        (positions, channels, spatial, tree_width),
        (30, 1152, 15, 2048)
    );
    assert_eq!(effective_scale, 1.0 / 32.0);
    assert_eq!(
        (s.elements, s.input_bytes, s.output_bytes),
        (34560, 207360, 69120)
    );
    // 15 dots each (1152 products + 2047 tree adds + scale),
    // 14 online updates, and one final reciprocal + 1152 multiplies.
    assert_eq!(s.operations, 30 * (15 * 3200 + 14 * (3 * 1152 + 6) + 1153));
    assert!(geometry(
        Kind::Attention,
        &pair(&[2, 3456, 3, 5], &[2, 1152, 5, 3]),
        &scale()
    )
    .is_err());
    assert!(geometry(
        Kind::Attention,
        &pair(&[1, 6147, 1, 1], &[1, 2049, 1, 1]),
        &scale()
    )
    .is_err());
}

#[test]
fn attention_validates_small_odd_channels_dtype_attributes_and_work_overflow() {
    for channels in [1, 3, 2048] {
        let s = geometry(
            Kind::Attention,
            &pair(&[1, 3 * channels, 1, 2], &[1, channels, 1, 2]),
            &scale(),
        )
        .unwrap();
        let NativeShape::Attention { tree_width, .. } = s.native else {
            unreachable!()
        };
        assert_eq!(u64::from(tree_width), channels.next_power_of_two());
    }
    let wrong = [
        value(
            "x",
            ResolvedValueRole::Input,
            &[1, 9, 1, 2],
            ElementType::F16,
        ),
        value(
            "y",
            ResolvedValueRole::Output,
            &[1, 3, 1, 2],
            ElementType::Bf16,
        ),
    ];
    assert!(geometry(Kind::Attention, &wrong, &scale()).is_err());
    assert!(geometry(
        Kind::Attention,
        &pair(&[1, 9, 1, 2], &[1, 3, 1, 2]),
        &BTreeMap::new()
    )
    .is_err());
    let mut extra = scale();
    extra.insert(
        AttributeId::new("heads").unwrap(),
        SemanticValue::Unsigned(3),
    );
    assert!(geometry(Kind::Attention, &pair(&[1, 9, 1, 2], &[1, 3, 1, 2]), &extra).is_err());
    // Metadata fits address space but quadratic arithmetic attribution does not.
    if isize::BITS >= 64 {
        assert!(geometry(
            Kind::Attention,
            &pair(&[1, 3, 1, 1u64 << 32], &[1, 1, 1, 1u64 << 32]),
            &scale()
        )
        .is_err());
    }
}

#[test]
fn spatial_gather_encodes_explicit_channel_phase_and_batch_dimensions() {
    for offsets in [[0, 0, 0, 0], [1, 1, 1, 1], [0, 0, 1, 1]] {
        let s = geometry(
            Kind::Gather,
            &pair(&[2, 6, 2, 3], &[2, 3, 4, 6]),
            &gather_attrs(2, 2, 2, &offsets),
        )
        .unwrap();
        let NativeShape::Gather(n) = s.native else {
            unreachable!()
        };
        assert_eq!(&n.offsets[..4], &offsets.map(|v| v as u64));
        assert!(n.offsets[4..].iter().all(|&v| v == 0));
        assert_eq!(
            (
                n.input_channels,
                n.output_channels,
                n.input_height,
                n.input_width,
                n.output_height,
                n.output_width
            ),
            (6, 3, 2, 3, 4, 6)
        );
        assert_eq!((n.scale_h, n.scale_w, n.channel_stride), (2, 2, 2));
        assert_eq!(
            (s.input_bytes, s.output_bytes, s.elements, s.operations),
            (144, 288, 144, 0)
        );
    }
    let nearest = geometry(
        Kind::Gather,
        &pair(&[2, 3, 2, 3], &[2, 3, 4, 6]),
        &gather_attrs(2, 2, 1, &[0; 4]),
    )
    .unwrap();
    let NativeShape::Gather(n) = nearest.native else {
        unreachable!()
    };
    assert_eq!(n.channel_stride, 1);
    assert_eq!(std::mem::size_of::<NativeGather>(), 584);
    assert_eq!(std::mem::align_of::<NativeGather>(), 8);
}

#[test]
fn gather_rejects_wrapping_channels_wrong_tile_and_same_count_wrong_shape() {
    let ports = pair(&[2, 6, 2, 3], &[2, 3, 4, 6]);
    for attrs in [
        gather_attrs(2, 2, 2, &[0, 0, 1]),
        gather_attrs(2, 2, 2, &[0, 0, -1, 1]),
        gather_attrs(2, 2, 2, &[0, 0, 2, 2]),
        gather_attrs(2, 2, u64::MAX, &[0; 4]),
        gather_attrs(0, 2, 1, &[0]),
    ] {
        assert!(geometry(Kind::Gather, &ports, &attrs).is_err());
    }
    assert!(geometry(
        Kind::Gather,
        &pair(&[2, 6, 2, 3], &[2, 3, 6, 4]),
        &gather_attrs(2, 2, 2, &[0, 0, 1, 1])
    )
    .is_err());
    assert!(geometry(
        Kind::Gather,
        &pair(&[1, 1, 1, 1], &[1, 1, 1, 65]),
        &gather_attrs(1, 65, 1, &[0; 65])
    )
    .is_err());
    assert!(geometry(
        Kind::Gather,
        &pair(&[1, 1, 1, 1], &[1, 1, 1, 64]),
        &gather_attrs(1, 64, 1, &[0; 64])
    )
    .is_ok());
}

#[test]
fn independent_capabilities_and_no_alias_are_checked_before_native_launch() {
    let config =
        super::super::super::cuda_tensor_runtime_config(0, DeviceId::new("device.cuda.0").unwrap())
            .unwrap();
    let mut device = DeviceDescriptor {
        id: config.device_id,
        class: DeviceClass::Accelerator,
        ordinal: 0,
        total_memory_bytes: 4096,
        runtime_implementation_fingerprint: config.runtime_implementation_fingerprint,
        capabilities: config.capabilities,
        dynamic_storage_profiles: config.dynamic_storage_profiles,
    };
    let a = descriptor(&device, Kind::Attention).unwrap();
    let g = descriptor(&device, Kind::Gather).unwrap();
    assert_ne!(a.operation_id(), g.operation_id());
    assert_ne!(
        a.provider_implementation_fingerprint(),
        g.provider_implementation_fingerprint()
    );
    assert!(a.supports_fixed_tensor_work() && g.supports_fixed_tensor_work());
    device
        .capabilities
        .remove(&CapabilityId::new(SPATIAL_ATTENTION_BF16_CAPABILITY_ID).unwrap());
    assert!(descriptor(&device, Kind::Attention).is_err());
    assert!(descriptor(&device, Kind::Gather).is_ok());
    assert!(nonoverlapping_ranges(256, 64, 256, 64).is_err());
    assert!(nonoverlapping_ranges(256, 64, 288, 64).is_err());
    assert!(nonoverlapping_ranges(256, 64, 320, 64).is_ok());
}
