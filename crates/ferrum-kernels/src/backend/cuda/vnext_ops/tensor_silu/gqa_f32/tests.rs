use super::*;
use ferrum_runtime::{
    CanonicalRational, DeviceClass, DeviceId, DynamicStorageAllocator, DynamicStorageProfile,
    DynamicStorageView, ResolvedTensorLayout,
};
fn spec(rows: u64, columns: u64, dtype: ElementType) -> ResolvedTensorSpec {
    ResolvedTensorSpec::new(vec![rows, columns], dtype, ResolvedTensorLayout::Contiguous).unwrap()
}
fn attributes(sequence: u64, batch: usize) -> BTreeMap<AttributeId, SemanticValue> {
    let mut a = BTreeMap::from([
        (
            AttributeId::new("sequence_length").unwrap(),
            SemanticValue::Unsigned(sequence),
        ),
        (
            AttributeId::new("head_dim").unwrap(),
            SemanticValue::Unsigned(128),
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
        a.insert(
            AttributeId::new(name).unwrap(),
            SemanticValue::Integers(values),
        );
    }
    a
}
#[test]
fn f32_descriptor_has_independent_identity_and_requires_real_capability() {
    let mut device = DeviceDescriptor {
        id: DeviceId::new("device.cuda.0").unwrap(),
        class: DeviceClass::Accelerator,
        ordinal: 0,
        total_memory_bytes: 1 << 20,
        runtime_implementation_fingerprint: "a".repeat(64),
        capabilities: BTreeSet::from([CapabilityId::new(
            STRUCTURED_GQA_ATTENTION_F32_CAPABILITY_ID,
        )
        .unwrap()]),
        dynamic_storage_profiles: BTreeSet::from([DynamicStorageProfile::new(
            DynamicStorageAllocator::LinearArena,
            DynamicStorageView::Contiguous,
        )
        .unwrap()]),
    };
    let d = descriptor(&device).unwrap();
    assert_eq!(
        d.operation_id().as_str(),
        STRUCTURED_GQA_ATTENTION_F32_OPERATION_ID
    );
    assert_eq!(
        d.operation_fingerprint(),
        structured_gqa_attention_f32_contract()
            .unwrap()
            .descriptor()
            .fingerprint()
            .unwrap()
    );
    assert!(d.supports_fixed_tensor_work());
    assert!(d.accepted_quantization_formats().is_empty());
    device.capabilities = BTreeSet::from([CapabilityId::new(
        ferrum_runtime::STRUCTURED_GQA_ATTENTION_BF16_CAPABILITY_ID,
    )
    .unwrap()]);
    assert!(descriptor(&device).is_err());
}
#[test]
fn original_f32_ranges_and_large_batch_mask_tiles_have_no_half_or_repeat_shortcut() {
    let batch = 17;
    let rows = batch * 3;
    let q = spec(rows, 4 * 128, ElementType::F32);
    let kv = spec(rows, 128, ElementType::F32);
    let attrs = attributes(3, batch as usize);
    let s = validate_specs(&q, &kv, &kv, &q, &attrs).unwrap();
    assert_eq!(s.query_bytes, rows * 4 * 128 * 4);
    assert_eq!(s.key_value_bytes, rows * 128 * 4);
    assert_eq!(s.masks.iter().map(|t| t.batches).sum::<u64>(), batch);
    assert_eq!(s.masks.last().unwrap().first_batch, 16);
    let launch = s.launch_config();
    assert_eq!(launch.block_dim, (128, 1, 1));
    assert_eq!(launch.shared_mem_bytes, 128 * 4);
    let half = spec(rows, 128, ElementType::Bf16);
    assert!(validate_specs(&q, &half, &half, &q, &attrs).is_err());
}

#[test]
fn nonpower_launch_uses_exact_original_ports_and_bounded_shared_extent() {
    for width in [3u64, 72, 80, 96, 255, 256] {
        let q = spec(6, 4 * width, ElementType::F32);
        let kv = spec(6, 2 * width, ElementType::F32);
        let mut a = attributes(3, 2);
        a.insert(
            AttributeId::new("head_dim").unwrap(),
            SemanticValue::Unsigned(width),
        );
        let s = validate_specs(&q, &kv, &kv, &q, &a).unwrap();
        assert_eq!(s.launch_config().block_dim, (width as u32, 1, 1));
        assert_eq!(s.launch_config().shared_mem_bytes, width as u32 * 4);
        assert_eq!(s.query_bytes, 6 * 4 * width * 4);
        assert_eq!(s.key_value_bytes, 6 * 2 * width * 4);
    }
}
