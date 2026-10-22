//! Descriptor selection is pure: a rejected subset cannot load a module or
//! consume its capacity permit. Actual execution uses managed GPU fixtures.

use super::*;
use ferrum_runtime::{
    DeviceClass, DeviceId, DynamicStorageAllocator, DynamicStorageProfile, DynamicStorageView,
};

fn device() -> DeviceDescriptor {
    let config =
        super::super::super::cuda_tensor_runtime_config(0, DeviceId::new("device.cuda.0").unwrap())
            .unwrap();
    DeviceDescriptor {
        id: config.device_id,
        class: DeviceClass::Accelerator,
        ordinal: 0,
        total_memory_bytes: 4096,
        runtime_implementation_fingerprint: config.runtime_implementation_fingerprint,
        capabilities: config.capabilities,
        dynamic_storage_profiles: BTreeSet::from([DynamicStorageProfile::new(
            DynamicStorageAllocator::LinearArena,
            DynamicStorageView::Contiguous,
        )
        .unwrap()]),
    }
}

#[test]
fn explicit_subset_preserves_order_and_rejects_duplicates_empty_and_missing_capability() {
    let d = device();
    let selected = [
        CudaTensorOperation::GeluTanh,
        CudaTensorOperation::ZeroCenteredRmsNorm,
    ];
    let descriptions = selected_descriptors(&d, &selected).unwrap();
    assert_eq!(
        descriptions
            .iter()
            .map(|(kind, _)| *kind)
            .collect::<Vec<_>>(),
        selected
    );
    assert_eq!(
        descriptions[0].1.operation_id().as_str(),
        GELU_TANH_BF16_OPERATION_ID
    );
    assert_eq!(
        descriptions[1].1.operation_id().as_str(),
        ferrum_runtime::ZERO_CENTERED_RMS_NORM_BF16_OPERATION_ID
    );
    assert!(descriptions[0].1.accepted_weight_formats().is_empty());
    assert!(descriptions[1].1.accepted_weight_formats().contains(
        &ferrum_runtime::WeightFormatId::new("weight-format.safetensors.dense").unwrap()
    ));
    assert!(selected_descriptors(&d, &[]).is_err());
    assert!(selected_descriptors(&d, &[selected[0], selected[0]]).is_err());
    let mut missing = d;
    missing
        .capabilities
        .remove(&CapabilityId::new(GELU_TANH_BF16_CAPABILITY_ID).unwrap());
    assert!(selected_descriptors(&missing, &selected).is_err());
    // The unrelated missing capability cannot prevent a valid smaller subset.
    assert!(selected_descriptors(&missing, &[selected[1]]).is_ok());
}

#[test]
fn module_selection_keeps_independent_contract_and_estimator_identities() {
    let descriptions = selected_descriptors(&device(), &CudaTensorOperation::ALL).unwrap();
    let mut operations = BTreeSet::new();
    let mut providers = BTreeSet::new();
    let mut estimators = BTreeSet::new();
    for (_, descriptor) in descriptions {
        assert!(descriptor.supports_fixed_tensor_work());
        assert!(operations.insert(descriptor.operation_id().clone()));
        assert!(providers.insert(descriptor.provider_implementation_fingerprint().to_owned()));
        assert!(estimators.insert(
            descriptor
                .resource_estimator_implementation_fingerprint()
                .to_owned()
        ));
    }
}

#[test]
fn gelu_logical_work_boundary_is_checked_before_payload_allocation() {
    let width = u64::MAX / 9 + 1;
    if u128::from(width) * 2 <= isize::MAX as u128 {
        assert!(geometry::Shape::new(Kind::GeluTanh, 1, width, false).is_err());
        assert!(geometry::Shape::new(Kind::Tanh, 1, width, false).is_ok());
    }
    let valid = geometry::Shape::new(Kind::GeluTanh, 3, 4096, false).unwrap();
    assert_eq!(valid.arithmetic_operations, 3 * 4096 * 9);
    assert_eq!(valid.bytes, 3 * 4096 * 2);
    assert!(geometry::Shape::new(Kind::GeluTanh, 3, 4096, true).is_err());
}
