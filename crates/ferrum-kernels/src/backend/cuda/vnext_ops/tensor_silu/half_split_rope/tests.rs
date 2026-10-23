//! Descriptor/geometry checks only. Native numerical execution is separate.
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
    tensor: ResolvedTensorSpec,
) -> ResolvedValueBinding {
    let storage = ResolvedValueStorage::single(
        ResourceId::new(format!("resource.{name}")).unwrap(),
        0,
        tensor.minimum_storage_bytes().unwrap(),
        tensor.element_type(),
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

fn values() -> Vec<ResolvedValueBinding> {
    vec![
        value(
            "x",
            ResolvedValueRole::Input,
            0,
            spec(&[6, 8], ElementType::Bf16),
        ),
        value(
            "cos",
            ResolvedValueRole::Input,
            1,
            spec(&[6, 4], ElementType::Bf16),
        ),
        value(
            "sin",
            ResolvedValueRole::Input,
            2,
            spec(&[6, 4], ElementType::Bf16),
        ),
        value(
            "y",
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
fn descriptor_binds_staged_half_split_contract_without_weight_or_quantized_formats() {
    let mut device = DeviceDescriptor {
        id: DeviceId::new("device.cuda.0").unwrap(),
        class: DeviceClass::Accelerator,
        ordinal: 0,
        total_memory_bytes: 4096,
        runtime_implementation_fingerprint: "a".repeat(64),
        capabilities: BTreeSet::from([
            CapabilityId::new(HALF_SPLIT_ROPE_BF16_CAPABILITY_ID).unwrap()
        ]),
        dynamic_storage_profiles: BTreeSet::from([DynamicStorageProfile::new(
            DynamicStorageAllocator::LinearArena,
            DynamicStorageView::Contiguous,
        )
        .unwrap()]),
    };
    let descriptor = super::descriptor(&device).unwrap();
    assert_eq!(
        descriptor.operation_id().as_str(),
        HALF_SPLIT_ROPE_BF16_OPERATION_ID
    );
    assert_eq!(
        descriptor.operation_fingerprint(),
        half_split_rope_bf16_contract()
            .unwrap()
            .descriptor()
            .fingerprint()
            .unwrap()
    );
    assert!(descriptor.supports_fixed_tensor_work());
    assert!(descriptor.accepted_weight_formats().is_empty());
    assert!(descriptor.accepted_quantization_formats().is_empty());
    assert_eq!(
        descriptor.provider_implementation_fingerprint(),
        super::descriptor(&device)
            .unwrap()
            .provider_implementation_fingerprint()
    );
    assert_eq!(
        descriptor.resource_estimator_implementation_fingerprint(),
        super::descriptor(&device)
            .unwrap()
            .resource_estimator_implementation_fingerprint()
    );
    device.capabilities.clear();
    assert!(super::descriptor(&device).is_err());
}

#[test]
fn frequency_extent_is_bf16_per_row_and_shared_only_across_heads() {
    let shape = geometry(&values(), &attributes(4)).unwrap();
    assert_eq!((shape.columns, shape.head_dim), (8, 4));
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
    let x = spec(&[6, 8], ElementType::Bf16);
    let frequency = spec(&[6, 4], ElementType::Bf16);
    let checked = HalfSplitRopeBf16Geometry::new(&x, &frequency, &frequency, &x, 4).unwrap();
    assert_eq!(
        (
            checked.rows(),
            checked.columns(),
            checked.head_dim(),
            checked.heads()
        ),
        (6, 8, 4, 2)
    );
    // No numerical payload is read and no equality between the frequency
    // halves is required by the descriptor or geometry.
    let width = (1_u64 << 24) + 2;
    let wide = spec(&[1, width], ElementType::Bf16);
    assert!(validate_specs(&wide, &wide, &wide, &wide, width).is_ok());
}

#[test]
fn dtype_shape_broadcast_attrs_and_ports_are_strict() {
    let x = spec(&[6, 8], ElementType::Bf16);
    let frequency = spec(&[6, 4], ElementType::Bf16);
    for wrong in [
        spec(&[6, 4], ElementType::F32),
        spec(&[3, 8], ElementType::Bf16),
        spec(&[1, 4], ElementType::Bf16),
    ] {
        assert!(validate_specs(&x, &wrong, &frequency, &x, 4).is_err());
        assert!(validate_specs(&x, &frequency, &wrong, &x, 4).is_err());
    }
    assert!(validate_specs(
        &x,
        &frequency,
        &frequency,
        &spec(&[3, 16], ElementType::Bf16),
        4
    )
    .is_err());
    for width in [0, 1, 3, 6, 16] {
        assert!(validate_specs(&x, &frequency, &frequency, &x, width).is_err());
    }
    let mut extra = attributes(4);
    extra.insert(
        AttributeId::new("sequence_length").unwrap(),
        SemanticValue::Unsigned(3),
    );
    assert!(geometry(&values(), &extra).is_err());
    assert!(geometry(&values(), &BTreeMap::new()).is_err());
    let mut missing = values();
    missing.remove(2);
    assert!(geometry(&missing, &attributes(4)).is_err());
    let mut duplicate = values();
    duplicate[2] = duplicate[1].clone();
    assert!(geometry(&duplicate, &attributes(4)).is_err());
    assert!(nonoverlapping_ranges(16, 96, 32, 96).is_err());
    assert!(nonoverlapping_ranges(16, 48, 64, 96).is_ok());
}

#[test]
fn metadata_only_oversize_is_rejected_before_encoding() {
    let width = (isize::MAX as u64 / 2 + 2) & !1;
    let x = spec(&[1, width], ElementType::Bf16);
    let frequency = spec(&[1, 2], ElementType::Bf16);
    assert!(validate_specs(&x, &frequency, &frequency, &x, 2).is_err());
}
