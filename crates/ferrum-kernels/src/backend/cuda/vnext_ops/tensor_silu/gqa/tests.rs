//! Hardware-independent provider and unequal-Q/KV-extent checks.
use super::*;
use ferrum_runtime::{
    AliasPolicy, BufferUsage, CanonicalRational, DeviceClass, DeviceId, DynamicStorageAllocator,
    DynamicStorageProfile, DynamicStorageView, ProgramValueId, ResolvedTensorLayout,
    ResolvedValueStorage, ResourceId, TensorAccess,
};

fn spec(rows: u64, columns: u64) -> ResolvedTensorSpec {
    ResolvedTensorSpec::new(
        vec![rows, columns],
        ElementType::Bf16,
        ResolvedTensorLayout::Contiguous,
    )
    .unwrap()
}

fn attrs(sequence: u64, dim: u64, batch: usize) -> BTreeMap<AttributeId, SemanticValue> {
    let mut attrs = BTreeMap::from([
        (
            AttributeId::new("sequence_length").unwrap(),
            SemanticValue::Unsigned(sequence),
        ),
        (
            AttributeId::new("head_dim").unwrap(),
            SemanticValue::Unsigned(dim),
        ),
        (
            AttributeId::new("scale").unwrap(),
            SemanticValue::Rational(CanonicalRational::new(1, 2).unwrap()),
        ),
    ]);
    for (name, values) in [
        ("segment_lengths", vec![sequence as i64]),
        ("segment_bidirectional", vec![0]),
        ("key_run_counts", vec![1; batch]),
        ("key_run_lengths", vec![sequence as i64; batch]),
        ("key_run_valid", vec![1; batch]),
    ] {
        attrs.insert(
            AttributeId::new(name).unwrap(),
            SemanticValue::Integers(values),
        );
    }
    attrs
}

fn ports(q: &ResolvedTensorSpec, kv: &ResolvedTensorSpec) -> Vec<ResolvedValueBinding> {
    [
        (ResolvedValueRole::Input, 0, q),
        (ResolvedValueRole::Input, 1, kv),
        (ResolvedValueRole::Input, 2, kv),
        (ResolvedValueRole::Output, 0, q),
    ]
    .into_iter()
    .enumerate()
    .map(|(index, (role, ordinal, tensor))| {
        ResolvedValueBinding::new(
            ProgramValueId::new(format!("value.gqa.{index}")).unwrap(),
            role,
            ordinal,
            tensor.clone(),
            if role == ResolvedValueRole::Input {
                TensorAccess::Read
            } else {
                TensorAccess::Write
            },
            AliasPolicy::NoAlias,
            BufferUsage::Activations,
            None,
            ResolvedValueStorage::single(
                ResourceId::new(format!("resource.gqa.{index}")).unwrap(),
                0,
                tensor.minimum_storage_bytes().unwrap(),
                ElementType::Bf16,
            )
            .unwrap(),
        )
        .unwrap()
    })
    .collect()
}

#[test]
fn descriptor_has_independent_gqa_contract_identity_and_no_auxiliary_formats() {
    let mut device = DeviceDescriptor {
        id: DeviceId::new("device.cuda.0").unwrap(),
        class: DeviceClass::Accelerator,
        ordinal: 0,
        total_memory_bytes: 4096,
        runtime_implementation_fingerprint: "a".repeat(64),
        capabilities: BTreeSet::from([CapabilityId::new(
            STRUCTURED_GQA_ATTENTION_BF16_CAPABILITY_ID,
        )
        .unwrap()]),
        dynamic_storage_profiles: BTreeSet::from([DynamicStorageProfile::new(
            DynamicStorageAllocator::LinearArena,
            DynamicStorageView::Contiguous,
        )
        .unwrap()]),
    };
    let descriptor = super::descriptor(&device).unwrap();
    assert_eq!(
        descriptor.operation_id().as_str(),
        STRUCTURED_GQA_ATTENTION_BF16_OPERATION_ID
    );
    assert_eq!(
        descriptor.operation_fingerprint(),
        structured_gqa_attention_bf16_contract()
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
    device.capabilities.clear();
    assert!(super::descriptor(&device).is_err());
}

#[test]
fn gqa_uses_original_kv_extent_and_counts_query_head_work() {
    let q = spec(6, 32 * 128);
    let kv = spec(6, 8 * 128);
    let shape = geometry(&ports(&q, &kv), &attrs(3, 128, 2)).unwrap();
    assert_eq!(
        (
            shape.query_heads,
            shape.key_value_heads,
            shape.width,
            shape.sequence
        ),
        (32, 8, 128, 3)
    );
    assert_eq!((shape.groups, shape.elements), (192, 24_576));
    assert_eq!((shape.query_bytes, shape.key_value_bytes), (49_152, 12_288));
    // Two causal S=3 samples: P=12, N=6; ((5D+6)P-(2D+5)N)*Hq.
    assert_eq!(shape.operations, 197_952);
    assert_eq!(std::mem::size_of::<NativeMask>(), 2312);
    // Actual by-value ABI: four pointers, three u64 scalars, three u32/F32
    // scalars and the mask, below CUDA's 4 KiB baseline parameter limit.
    assert_eq!(
        4 * 8 + 3 * 8 + 3 * 4 + std::mem::size_of::<NativeMask>(),
        2380
    );
    assert!(validate_specs(&q, &q, &q, &q, &attrs(3, 128, 2)).is_ok());
}

#[test]
fn wrong_group_ratio_rows_dtype_shape_mask_and_ports_are_rejected() {
    let q = spec(4, 8);
    let kv = spec(4, 4);
    let attributes = attrs(2, 2, 2);
    assert!(validate_specs(&q, &kv, &kv, &q, &attributes).is_ok());
    assert!(validate_specs(&spec(4, 6), &kv, &kv, &spec(4, 6), &attributes).is_err());
    assert!(validate_specs(&q, &spec(4, 16), &spec(4, 16), &q, &attributes).is_err());
    assert!(validate_specs(&q, &spec(2, 8), &spec(2, 8), &q, &attributes).is_err());
    assert!(validate_specs(&q, &kv, &q, &q, &attributes).is_err());
    assert!(validate_specs(&q, &kv, &kv, &spec(2, 16), &attributes).is_err());
    let f16 = ResolvedTensorSpec::new(
        vec![4, 4],
        ElementType::F16,
        ResolvedTensorLayout::Contiguous,
    )
    .unwrap();
    assert!(validate_specs(&q, &f16, &f16, &q, &attributes).is_err());
    assert!(validate_specs(&q, &kv, &kv, &q, &attrs(2, 2, 1)).is_err());
    let mut missing = ports(&q, &kv);
    missing.remove(2);
    assert!(geometry(&missing, &attributes).is_err());
    let mut duplicate = ports(&q, &kv);
    duplicate[2] = duplicate[1].clone();
    assert!(geometry(&duplicate, &attributes).is_err());
    assert!(nonoverlapping_ranges(16, 32, 32, 64).is_err());
    assert!(nonoverlapping_ranges(16, 32, 48, 64).is_ok());
}
