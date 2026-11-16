//! CPU metadata checks only: no CUDA context, library, module or plan creation.
use super::*;
use ferrum_runtime::{DeviceClass, DeviceId};

fn device(ops: &[CudaCudnnOperation]) -> DeviceDescriptor {
    DeviceDescriptor {
        id: DeviceId::new("device.cuda.selection").unwrap(),
        class: DeviceClass::Accelerator,
        ordinal: 0,
        total_memory_bytes: 1 << 30,
        runtime_implementation_fingerprint: "a".repeat(64),
        capabilities: ops
            .iter()
            .map(|op| CapabilityId::new(op.capability()).unwrap())
            .collect(),
        dynamic_storage_profiles: BTreeSet::new(),
    }
}
fn policy() -> CudnnSdpaPlanPolicy {
    CudnnSdpaPlanPolicy {
        maximum_plans: 8,
        maximum_live_bindings: 1,
        opaque_memory_bytes: 1 << 20,
        workspace_limit_bytes: 1 << 24,
    }
}
#[test]
fn cudnn_composition_rejects_duplicate_implementations_before_loading() {
    use CudaCudnnOperation::*;
    let d = device(&[Conv2dF32, SegmentedAttentionBf16]);
    assert!(validate(&d, None, &[Conv2dF32, Conv2dF32], &[]).is_err());
    assert!(validate(
        &d,
        Some(policy()),
        &[SegmentedAttentionBf16],
        &[CudaTensorOperation::SegmentedAttention]
    )
    .is_err());
    assert!(validate(
        &d,
        None,
        &[Conv2dF32],
        &[CudaTensorOperation::Silu, CudaTensorOperation::Silu]
    )
    .is_err());
    assert!(validate(&d, None, &[], &[]).is_err());
}
#[test]
fn cudnn_composition_requires_cache_only_for_selected_attention() {
    use CudaCudnnOperation::*;
    let vendor = [
        Conv2dF32,
        ConvTranspose2dF32,
        CausalGqaBf16,
        SegmentedAttentionBf16,
    ];
    let d = device(&vendor);
    assert!(validate(&d, Some(policy()), &vendor, &[]).is_ok());
    assert!(validate(&d, None, &vendor, &[]).is_err());
    assert!(validate(&d, Some(policy()), &[Conv2dF32], &[]).is_err());
    assert!(validate(&d, None, &[Conv2dF32, ConvTranspose2dF32], &[]).is_ok());
    let mut invalid = policy();
    invalid.maximum_live_bindings = 0;
    assert!(validate(&d, Some(invalid), &vendor, &[]).is_err());
}
#[test]
fn cudnn_composition_does_not_grant_missing_or_external_capabilities() {
    use CudaCudnnOperation::*;
    let d = device(&[Conv2dF32]);
    assert!(validate(&d, None, &[Conv2dF32, ConvTranspose2dF32], &[]).is_err());
    assert!(validate(
        &d,
        None,
        &[Conv2dF32],
        &[CudaTensorOperation::StructuredGqaAttentionFusedF32]
    )
    .is_err());
}
