//! Factory preparation is metadata-only: these tests load no library/module,
//! create no runtime and cannot stand in for actual cuDNN CUDA qualification.
use super::*;
use ferrum_runtime::{
    DeviceClass, DeviceId, DynamicStorageAllocator, DynamicStorageProfile, DynamicStorageView,
    PLAIN_RESIDUAL_ADD_F32_CAPABILITY_ID, SEGMENTED_ATTENTION_BF16_CAPABILITY_ID,
    STRUCTURED_GQA_ATTENTION_F32_CAPABILITY_ID,
};
fn device() -> DeviceDescriptor {
    DeviceDescriptor {
        id: DeviceId::new("device.cuda.0").unwrap(),
        class: DeviceClass::Accelerator,
        ordinal: 0,
        total_memory_bytes: 1 << 30,
        runtime_implementation_fingerprint: "a".repeat(64),
        capabilities: [
            CONV2D_F32_CAPABILITY_ID,
            SEGMENTED_ATTENTION_BF16_CAPABILITY_ID,
            STRUCTURED_GQA_ATTENTION_F32_CAPABILITY_ID,
            PLAIN_RESIDUAL_ADD_F32_CAPABILITY_ID,
        ]
        .into_iter()
        .map(|s| CapabilityId::new(s).unwrap())
        .collect(),
        dynamic_storage_profiles: BTreeSet::from([DynamicStorageProfile::new(
            DynamicStorageAllocator::LinearArena,
            DynamicStorageView::Contiguous,
        )
        .unwrap()]),
    }
}
fn policy() -> CudnnSdpaPlanPolicy {
    CudnnSdpaPlanPolicy {
        maximum_plans: 4,
        maximum_live_bindings: 2,
        opaque_memory_bytes: 1 << 20,
        workspace_limit_bytes: 0,
    }
}
fn descriptors(
    device: &DeviceDescriptor,
    selected: &[CudaTensorOperation],
) -> Vec<selection::SelectedDescriptor> {
    selection::selected_descriptors(device, selected, None).unwrap()
}
fn selected(d: &[selection::SelectedDescriptor]) -> Vec<CudaTensorOperation> {
    d.iter()
        .map(|d| match d {
            selection::SelectedDescriptor::Fixed(op, _) => *op,
            _ => panic!("fixture has no Philox"),
        })
        .collect()
}

#[test]
fn explicit_segmented_sdpa_replaces_only_selected_bf16_segmented_attention() {
    let device = device();
    require_convolution_capability(&device).unwrap();
    let original = [
        CudaTensorOperation::PlainResidualAddF32,
        CudaTensorOperation::SegmentedAttention,
        CudaTensorOperation::StructuredGqaAttentionF32,
    ];
    let legacy = descriptors(&device, &original);
    assert_eq!(selected(&legacy), original);
    let mut explicit = descriptors(&device, &original);
    select_segmented_sdpa(&mut explicit, policy()).unwrap();
    assert_eq!(
        selected(&explicit),
        [
            CudaTensorOperation::PlainResidualAddF32,
            CudaTensorOperation::StructuredGqaAttentionF32
        ]
    );
    assert!(selection::selected_descriptors(
        &device,
        &[
            CudaTensorOperation::SegmentedAttention,
            CudaTensorOperation::SegmentedAttention
        ],
        None
    )
    .is_err());
    let mut missing = descriptors(&device, &[CudaTensorOperation::StructuredGqaAttentionF32]);
    assert!(select_segmented_sdpa(&mut missing, policy()).is_err());
    assert_eq!(
        selected(&missing),
        [CudaTensorOperation::StructuredGqaAttentionF32]
    );
    let mut no_segmented = device.clone();
    no_segmented
        .capabilities
        .remove(&CapabilityId::new(SEGMENTED_ATTENTION_BF16_CAPABILITY_ID).unwrap());
    assert!(selection::selected_descriptors(&no_segmented, &original, None).is_err());
    let mut no_conv = device;
    no_conv
        .capabilities
        .remove(&CapabilityId::new(CONV2D_F32_CAPABILITY_ID).unwrap());
    assert!(require_convolution_capability(&no_conv).is_err());
}
#[test]
fn invalid_sdpa_policy_leaves_original_selection_untouched_before_native_loading() {
    for field in 0..6 {
        let mut p = policy();
        match field {
            0 => p.maximum_plans = 0,
            1 => p.maximum_live_bindings = 0,
            2 => p.opaque_memory_bytes = 0,
            3 => p.workspace_limit_bytes = u64::MAX,
            4 => p.maximum_plans = usize::MAX,
            _ => p.maximum_live_bindings = usize::MAX,
        }
        let mut d = descriptors(
            &device(),
            &[
                CudaTensorOperation::SegmentedAttention,
                CudaTensorOperation::StructuredGqaAttentionF32,
            ],
        );
        let before = selected(&d);
        assert!(select_segmented_sdpa(&mut d, p).is_err());
        assert_eq!(selected(&d), before);
    }
}
