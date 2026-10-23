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

#[test]
fn d128_launch_covers_whole_warps_tails_and_grid_stride_without_dynamic_shared() {
    for groups in [1, 3, 4, 5, 4 * MAX_BLOCKS, 4 * MAX_BLOCKS + 1] {
        let q = spec(groups, 128);
        let shape = validate_specs(&q, &q, &q, &q, &attrs(groups, 128, 1)).unwrap();
        let launch = shape.launch_config();
        assert_eq!(launch.block_dim, (128, 1, 1));
        assert_eq!(launch.shared_mem_bytes, 0);
        assert_eq!(
            launch.grid_dim,
            (groups.div_ceil(4).min(MAX_BLOCKS) as u32, 1, 1)
        );
        let stride = u64::from(launch.grid_dim.0) * 4;
        // Each group has exactly one (block, warp, grid-stride iteration)
        // decomposition, including a final CTA with only one live warp.
        let mut visits = vec![0_u8; groups as usize];
        for first in 0..stride.min(groups) {
            for group in (first..groups).step_by(stride as usize) {
                visits[group as usize] += 1;
            }
        }
        assert!(visits.iter().all(|&count| count == 1));
    }
    for width in [2, 4, 8, 16, 32, 64, 256] {
        let q = spec(5, width);
        let shape = validate_specs(&q, &q, &q, &q, &attrs(5, width, 1)).unwrap();
        let launch = shape.launch_config();
        assert_eq!(launch.grid_dim, (5, 1, 1));
        assert_eq!(launch.block_dim, (width as u32, 1, 1));
        assert_eq!(launch.shared_mem_bytes, width as u32 * 4);
    }
}

#[test]
fn four_dimensional_subtrees_preserve_adjacent_rounding_and_signed_zero() {
    fn adjacent(values: &[f32]) -> f32 {
        if values.len() == 1 {
            values[0]
        } else {
            let half = values.len() / 2;
            adjacent(&values[..half]) + adjacent(&values[half..])
        }
    }
    fn warp(values: &[f32; 128]) -> f32 {
        let mut lane = [0.0_f32; 32];
        for (out, four) in lane.iter_mut().zip(values.chunks_exact(4)) {
            *out = (four[0] + four[1]) + (four[2] + four[3]);
        }
        for stride in [1, 2, 4, 8, 16] {
            for left in (0..32).step_by(2 * stride) {
                let right = lane[left + stride];
                lane[left] += right;
            }
        }
        lane[0]
    }
    // BF16-exact products. Adjacent pairs cancel to +0; a sequential fold
    // instead leaves +1. Place the cancellation across every subtree level.
    for separation in [1, 2, 4, 8, 16, 32] {
        let mut values = [0.0_f32; 128];
        for (index, value) in [33_554_432.0, 1.0, -33_554_432.0, 1.0]
            .into_iter()
            .enumerate()
        {
            values[index * separation] = value;
        }
        assert_eq!(adjacent(&values).to_bits(), 0.0_f32.to_bits());
        assert_eq!(warp(&values).to_bits(), adjacent(&values).to_bits());
        assert_eq!(values.into_iter().fold(0.0_f32, |a, b| a + b), 1.0);
    }
    let negative_zero = [-0.0_f32; 128];
    assert_eq!(warp(&negative_zero).to_bits(), (-0.0_f32).to_bits());
    assert_eq!(
        warp(&negative_zero).to_bits(),
        adjacent(&negative_zero).to_bits()
    );
    let tiny = [f32::from_bits(1); 128];
    assert_eq!(warp(&tiny).to_bits(), 128);
    assert_eq!(warp(&tiny).to_bits(), adjacent(&tiny).to_bits());
    let mut poison = [0.0_f32; 128];
    poison[0] = f32::INFINITY;
    poison[127] = f32::NEG_INFINITY;
    assert!(warp(&poison).is_nan());
    assert!(adjacent(&poison).is_nan());
}
