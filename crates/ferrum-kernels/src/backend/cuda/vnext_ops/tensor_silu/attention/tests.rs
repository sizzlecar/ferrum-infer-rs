//! Host-only ABI, geometry and preflight tests. Native arithmetic is validated
//! separately through real managed execution on the target CUDA device.
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

fn attrs(
    sequence: u64,
    dim: u64,
    lengths: &[i64],
    bidir: &[i64],
    counts: &[i64],
    runs: &[i64],
    valid: &[i64],
) -> BTreeMap<AttributeId, SemanticValue> {
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
        ("segment_lengths", lengths),
        ("segment_bidirectional", bidir),
        ("key_run_counts", counts),
        ("key_run_lengths", runs),
        ("key_run_valid", valid),
    ] {
        attrs.insert(
            AttributeId::new(name).unwrap(),
            SemanticValue::Integers(values.to_vec()),
        );
    }
    attrs
}

fn ports(tensor: &ResolvedTensorSpec) -> Vec<ResolvedValueBinding> {
    [
        (ResolvedValueRole::Input, 0),
        (ResolvedValueRole::Input, 1),
        (ResolvedValueRole::Input, 2),
        (ResolvedValueRole::Output, 0),
    ]
    .into_iter()
    .enumerate()
    .map(|(index, (role, ordinal))| {
        ResolvedValueBinding::new(
            ProgramValueId::new(format!("value.{index}")).unwrap(),
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
                ResourceId::new(format!("resource.{index}")).unwrap(),
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
fn descriptor_binds_independent_tensor_contract_and_estimator_identity() {
    let mut device = DeviceDescriptor {
        id: DeviceId::new("device.cuda.0").unwrap(),
        class: DeviceClass::Accelerator,
        ordinal: 0,
        total_memory_bytes: 4096,
        runtime_implementation_fingerprint: "a".repeat(64),
        capabilities: BTreeSet::from([
            CapabilityId::new(STRUCTURED_ATTENTION_BF16_CAPABILITY_ID).unwrap()
        ]),
        dynamic_storage_profiles: BTreeSet::from([DynamicStorageProfile::new(
            DynamicStorageAllocator::LinearArena,
            DynamicStorageView::Contiguous,
        )
        .unwrap()]),
    };
    let d = descriptor(&device).unwrap();
    assert_eq!(
        d.operation_id().as_str(),
        STRUCTURED_ATTENTION_BF16_OPERATION_ID
    );
    assert_eq!(
        d.operation_fingerprint(),
        structured_attention_bf16_contract()
            .unwrap()
            .descriptor()
            .fingerprint()
            .unwrap()
    );
    assert!(d.supports_fixed_tensor_work());
    assert!(d.accepted_weight_formats().is_empty());
    assert!(d.accepted_quantization_formats().is_empty());
    let again = descriptor(&device).unwrap();
    assert_eq!(
        d.provider_implementation_fingerprint(),
        again.provider_implementation_fingerprint()
    );
    assert_eq!(
        d.resource_estimator_implementation_fingerprint(),
        again.resource_estimator_implementation_fingerprint()
    );
    device.capabilities.clear();
    assert!(descriptor(&device).is_err());
}

#[test]
fn by_value_mask_preserves_image_boundaries_internal_padding_and_batch_strides() {
    let tensor = spec(20, 256);
    let a = attrs(
        10,
        128,
        &[2, 3, 2, 3],
        &[0, 1, 1, 1],
        &[3, 5],
        &[3, 1, 6, 3, 1, 4, 1, 1],
        &[1, 0, 1, 1, 0, 1, 0, 1],
    );
    let shape = geometry(&ports(&tensor), &a).unwrap();
    assert_eq!(
        (
            shape.groups,
            shape.heads,
            shape.sequence,
            shape.width,
            shape.bytes
        ),
        (40, 2, 10, 128, 10240)
    );
    assert_eq!(shape.scale, 0.5);
    assert_eq!(shape.mask.segment_count, 4);
    assert_eq!(&shape.mask.segment_ends[..4], &[2, 5, 7, 10]);
    assert_eq!(shape.mask.bidirectional_mask, 0b1110);
    assert_eq!(&shape.mask.key_run_counts[..2], &[3, 5]);
    assert_eq!(&shape.mask.key_run_ends[..3], &[3, 4, 10]);
    assert_eq!(&shape.mask.key_run_ends[32..37], &[3, 4, 8, 9, 10]);
    assert_eq!(&shape.mask.key_valid_masks[..2], &[0b101, 0b10101]);
    // Independently count causal OR equal image id, then key validity.
    let image_ids = [-1, -1, 0, 0, 0, 1, 1, 2, 2, 2];
    let mut pairs = 0;
    for b in 0..2 {
        for q in 0..10 {
            for k in 0..10 {
                pairs += u64::from(
                    (k <= q || (image_ids[q] >= 0 && image_ids[q] == image_ids[k]))
                        && k != 3
                        && !(b == 1 && k == 8),
                );
            }
        }
    }
    assert_eq!(
        shape.operations,
        2 * ((5 * 128 + 6) * pairs - (2 * 128 + 5) * 20)
    );
    // The complete parameter list includes four pointers, two u64 scalars,
    // three 32-bit scalars and the mask, with at most 8-byte tail alignment.
    assert_eq!(std::mem::size_of::<NativeMask>(), 2312);
    assert!(std::mem::size_of::<NativeMask>() + 4 * 8 + 2 * 8 + 3 * 4 + 7 < 4096);
}

#[test]
fn highest_mask_bits_and_final_batch_are_initialized_without_truncation() {
    let bidir = (0..32).map(|i| i % 2).collect::<Vec<i64>>();
    let valid = (0..16)
        .flat_map(|_| (0..32).map(|i| i % 2))
        .collect::<Vec<i64>>();
    let tensor = spec(16 * 32, 256);
    let a = attrs(32, 256, &[1; 32], &bidir, &[32; 16], &[1; 512], &valid);
    let shape = validate_specs(&tensor, &tensor, &tensor, &tensor, &a).unwrap();
    assert_eq!(shape.mask.bidirectional_mask, 0xaaaa_aaaa);
    assert_eq!(shape.mask.key_valid_masks[15], 0xaaaa_aaaa);
    assert_eq!(shape.mask.key_run_counts[15], 32);
    assert_eq!(shape.mask.key_run_ends[511], 32);
    assert_eq!(shape.mask.segment_ends[31], 32);
    let bad = attrs(32, 256, &[1; 32], &bidir, &[32; 17], &[1; 544], &[1; 544]);
    let tensor = spec(17 * 32, 256);
    assert!(validate_specs(&tensor, &tensor, &tensor, &tensor, &bad).is_err());
}

#[test]
fn preflight_rejects_wrong_ports_geometry_alias_and_unaddressable_storage() {
    let tensor = spec(2, 4);
    let a = attrs(2, 2, &[2], &[1], &[1], &[2], &[1]);
    let values = ports(&tensor);
    assert!(geometry(&values, &a).is_ok());
    assert!(geometry(&values[..3], &a).is_err());
    let mut duplicate = values.clone();
    duplicate[3] = duplicate[0].clone();
    assert!(geometry(&duplicate, &a).is_err());
    for other in [
        spec(1, 8),
        ResolvedTensorSpec::new(
            vec![2, 4],
            ElementType::F16,
            ResolvedTensorLayout::Contiguous,
        )
        .unwrap(),
    ] {
        assert!(validate_specs(&tensor, &other, &tensor, &tensor, &a).is_err());
        assert!(validate_specs(&tensor, &tensor, &other, &tensor, &a).is_err());
        assert!(validate_specs(&tensor, &tensor, &tensor, &other, &a).is_err());
    }
    let mut bad = a.clone();
    bad.insert(
        AttributeId::new("head_dim").unwrap(),
        SemanticValue::Unsigned(3),
    );
    assert!(geometry(&values, &bad).is_err());
    bad = a.clone();
    bad.insert(
        AttributeId::new("causal").unwrap(),
        SemanticValue::Unsigned(1),
    );
    assert!(geometry(&values, &bad).is_err());
    // Exact and partial output/read overlap are rejected; touching intervals
    // are legal. The provider applies this independently to Q, K and V.
    assert!(nonoverlapping_ranges(16, 16, 16, 16).is_err());
    assert!(nonoverlapping_ranges(16, 32, 32, 16).is_err());
    assert!(nonoverlapping_ranges(16, 16, 32, 16).is_ok());
    assert!(nonoverlapping_ranges(u64::MAX - 15, 16, 16, 16).is_err());
    let huge = spec(1, (isize::MAX as u64 / 4 + 1) * 2);
    let empty = attrs(1, 2, &[1], &[0], &[1], &[1], &[0]);
    assert!(validate_specs(&huge, &huge, &huge, &huge, &empty).is_err());
    let empty = attrs(2, 2, &[2], &[1], &[1], &[2], &[0]);
    assert_eq!(geometry(&values, &empty).unwrap().operations, 0);
}
