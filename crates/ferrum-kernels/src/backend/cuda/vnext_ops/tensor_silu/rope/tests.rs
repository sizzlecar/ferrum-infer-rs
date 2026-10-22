//! Hardware-independent descriptor/geometry checks, not native arithmetic evidence.
use super::*;
use ferrum_runtime::{
    AliasPolicy, BufferUsage, DeviceClass, DeviceId, DynamicStorageAllocator,
    DynamicStorageProfile, DynamicStorageView, ProgramValueId, ResolvedTensorLayout,
    ResolvedValueStorage, ResourceId, TensorAccess,
};

fn spec(dims: &[u64], dtype: ElementType) -> ResolvedTensorSpec {
    ResolvedTensorSpec::new(dims.to_vec(), dtype, ResolvedTensorLayout::Contiguous).unwrap()
}

fn value(
    name: &str,
    role: ResolvedValueRole,
    ordinal: u32,
    spec: ResolvedTensorSpec,
) -> ResolvedValueBinding {
    let storage = ResolvedValueStorage::single(
        ResourceId::new(format!("resource.{name}")).unwrap(),
        0,
        spec.minimum_storage_bytes().unwrap(),
        spec.element_type(),
    )
    .unwrap();
    ResolvedValueBinding::new(
        ProgramValueId::new(name).unwrap(),
        role,
        ordinal,
        spec,
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

fn values() -> Vec<ResolvedValueBinding> {
    vec![
        value(
            "input",
            ResolvedValueRole::Input,
            0,
            spec(&[6, 8], ElementType::Bf16),
        ),
        value(
            "frequency",
            ResolvedValueRole::Input,
            1,
            spec(&[3, 2, 2], ElementType::F32),
        ),
        value(
            "output",
            ResolvedValueRole::Output,
            0,
            spec(&[6, 8], ElementType::Bf16),
        ),
    ]
}

fn attributes(head_dim: u64) -> BTreeMap<AttributeId, SemanticValue> {
    BTreeMap::from([(
        AttributeId::new("head_dim").unwrap(),
        SemanticValue::Unsigned(head_dim),
    )])
}

#[test]
fn descriptor_binds_independent_fma_contract_and_declares_no_weight_formats() {
    let mut device = DeviceDescriptor {
        id: DeviceId::new("device.cuda.0").unwrap(),
        class: DeviceClass::Accelerator,
        ordinal: 0,
        total_memory_bytes: 4096,
        runtime_implementation_fingerprint: "a".repeat(64),
        capabilities: BTreeSet::from(
            [CapabilityId::new(ADJACENT_ROPE_BF16_CAPABILITY_ID).unwrap()],
        ),
        dynamic_storage_profiles: BTreeSet::from([DynamicStorageProfile::new(
            DynamicStorageAllocator::LinearArena,
            DynamicStorageView::Contiguous,
        )
        .unwrap()]),
    };
    let d = descriptor(&device).unwrap();
    assert_eq!(d.operation_id().as_str(), ADJACENT_ROPE_BF16_OPERATION_ID);
    assert_eq!(
        d.operation_fingerprint(),
        adjacent_rope_bf16_contract()
            .unwrap()
            .descriptor()
            .fingerprint()
            .unwrap()
    );
    assert!(d.supports_fixed_tensor_work());
    assert!(d.accepted_weight_formats().is_empty());
    assert!(d.accepted_quantization_formats().is_empty());
    assert_eq!(
        d.provider_implementation_fingerprint(),
        descriptor(&device)
            .unwrap()
            .provider_implementation_fingerprint()
    );
    assert_eq!(
        d.resource_estimator_implementation_fingerprint(),
        descriptor(&device)
            .unwrap()
            .resource_estimator_implementation_fingerprint()
    );
    device.capabilities.clear();
    assert!(descriptor(&device).is_err());
}

#[test]
fn geometry_preserves_shared_frequencies_and_checks_each_storage_width() {
    let shape = geometry(&values(), &attributes(4)).unwrap();
    assert_eq!((shape.columns, shape.sequence, shape.head_dim), (8, 3, 4));
    assert_eq!(
        (
            shape.elements,
            shape.pairs,
            shape.bytes,
            shape.frequency_bytes,
            shape.operations
        ),
        (48, 24, 96, 48, 168)
    );
    let checked = AdjacentRopeBf16Geometry::new(
        &spec(&[6, 8], ElementType::Bf16),
        &spec(&[3, 2, 2], ElementType::F32),
        &spec(&[6, 8], ElementType::Bf16),
        4,
    )
    .unwrap();
    assert_eq!((checked.batch(), checked.heads()), (2, 2));
    // RoPE does not inherit the normalization divisor's 2^24 width ceiling.
    let width = (1_u64 << 24) + 2;
    let x = spec(&[1, width], ElementType::Bf16);
    let f = spec(&[1, width / 2, 2], ElementType::F32);
    assert!(validate_specs(&x, &f, &x, width).is_ok());
}

#[test]
fn wrong_dtype_equal_numel_reshape_and_implicit_broadcast_are_rejected() {
    let x = spec(&[6, 8], ElementType::Bf16);
    let f = spec(&[3, 2, 2], ElementType::F32);
    for wrong in [
        spec(&[6, 8], ElementType::F16),
        spec(&[3, 16], ElementType::Bf16),
        spec(&[48], ElementType::Bf16),
    ] {
        assert!(validate_specs(&x, &f, &wrong, 4).is_err());
    }
    for wrong in [
        spec(&[3, 2, 2], ElementType::Bf16),
        spec(&[3, 2, 2], ElementType::U32),
        spec(&[3, 4], ElementType::F32),
        spec(&[4, 2, 2], ElementType::F32),
        spec(&[3, 1, 2], ElementType::F32),
    ] {
        assert!(validate_specs(&x, &wrong, &x, 4).is_err());
    }
    for width in [0, 1, 3, 6, 16] {
        assert!(validate_specs(&x, &f, &x, width).is_err());
    }
    assert!(validate_specs(&spec(&[6, 8], ElementType::F16), &f, &x, 4).is_err());
}

#[test]
fn ports_attributes_byte_limits_and_mixed_width_output_alias_are_preflighted() {
    let mut ports = values();
    let attrs = attributes(4);
    assert!(geometry(&ports[..2], &attrs).is_err());
    ports[2] = ports[0].clone();
    assert!(geometry(&ports, &attrs).is_err());
    assert!(geometry(&values(), &BTreeMap::new()).is_err());
    let mut extra = attrs;
    extra.insert(
        AttributeId::new("sequence").unwrap(),
        SemanticValue::Unsigned(3),
    );
    assert!(geometry(&values(), &extra).is_err());
    let mut wrong = attributes(4);
    wrong.insert(
        AttributeId::new("head_dim").unwrap(),
        SemanticValue::Integer(4),
    );
    assert!(geometry(&values(), &wrong).is_err());
    // Metadata only: the F32 table reaches the host extent limit before its
    // corresponding BF16 input, even though both have equal element counts.
    let rows = isize::MAX as u64 / 8 + 1;
    let x = spec(&[rows, 2], ElementType::Bf16);
    let f = spec(&[rows, 1, 2], ElementType::F32);
    assert!(validate_specs(&x, &f, &x, 2).is_err());
    // Both read regions are checked against output using their actual bytes.
    assert!(nonoverlapping_ranges(16, 48, 64, 96).is_ok());
    assert!(nonoverlapping_ranges(16, 48, 48, 96).is_err());
    assert!(nonoverlapping_ranges(64, 96, 64, 96).is_err());
}
