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
    let descriptions = selected_descriptors(&d, &selected, None).unwrap();
    assert_eq!(
        descriptions
            .iter()
            .map(SelectedDescriptor::operation)
            .collect::<Vec<_>>(),
        selected
    );
    assert_eq!(
        descriptions[0].descriptor().operation_id().as_str(),
        GELU_TANH_BF16_OPERATION_ID
    );
    assert_eq!(
        descriptions[1].descriptor().operation_id().as_str(),
        ferrum_runtime::ZERO_CENTERED_RMS_NORM_BF16_OPERATION_ID
    );
    assert!(descriptions[0]
        .descriptor()
        .accepted_weight_formats()
        .is_empty());
    assert!(descriptions[1]
        .descriptor()
        .accepted_weight_formats()
        .contains(
            &ferrum_runtime::WeightFormatId::new("weight-format.safetensors.dense").unwrap()
        ));
    assert!(selected_descriptors(&d, &[], None).is_err());
    assert!(selected_descriptors(&d, &[selected[0], selected[0]], None).is_err());
    let mut missing = d;
    missing
        .capabilities
        .remove(&CapabilityId::new(GELU_TANH_BF16_CAPABILITY_ID).unwrap());
    assert!(selected_descriptors(&missing, &selected, None).is_err());
    // The unrelated missing capability cannot prevent a valid smaller subset.
    assert!(selected_descriptors(&missing, &[selected[1]], None).is_ok());
}

#[test]
fn module_selection_keeps_independent_contract_and_estimator_identities() {
    let descriptions = selected_descriptors(
        &device(),
        &CudaTensorOperation::ALL,
        Some(CudaPhiloxDeviceGeometry::new(2, 768).unwrap()),
    )
    .unwrap();
    let mut operations = BTreeSet::new();
    let mut providers = BTreeSet::new();
    let mut estimators = BTreeSet::new();
    for description in descriptions {
        let descriptor = description.descriptor();
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
fn philox_mixed_selection_queries_once_only_when_explicitly_requested() {
    let device = device();
    let calls = std::cell::Cell::new(0);
    let query = || {
        calls.set(calls.get() + 1);
        CudaPhiloxDeviceGeometry::new(2, 768)
    };
    runtime_descriptors_with_query(&device, &[CudaTensorOperation::Silu], query).unwrap();
    assert_eq!(calls.get(), 0);
    let selected = [
        CudaTensorOperation::SiluMul,
        CudaTensorOperation::PhiloxNormal,
        CudaTensorOperation::LayerNorm,
    ];
    let descriptions = runtime_descriptors_with_query(&device, &selected, query).unwrap();
    assert_eq!(calls.get(), 1);
    assert_eq!(
        descriptions
            .iter()
            .map(SelectedDescriptor::operation)
            .collect::<Vec<_>>(),
        selected
    );
    assert_eq!(
        descriptions[1].descriptor().operation_id().as_str(),
        ferrum_runtime::PHILOX_NORMAL_BF16_OPERATION_ID
    );
    assert!(descriptions[1]
        .descriptor()
        .accepted_weight_formats()
        .is_empty());
    assert!(runtime_descriptors_with_query(
        &device,
        &[
            CudaTensorOperation::PhiloxNormal,
            CudaTensorOperation::PhiloxNormal
        ],
        query
    )
    .is_err());
    assert!(runtime_descriptors_with_query(&device, &[], query).is_err());
    assert_eq!(
        calls.get(),
        1,
        "invalid selection must be rejected before device query"
    );
    assert!(runtime_descriptors_with_query(&device, &selected, || Err(
        CudaDeviceRuntimeError::contract("injected query failure")
    ))
    .is_err());
    // Metadata-only preparation cannot silently substitute geometry or open a
    // device. The actual runtime supplies the queried values explicitly.
    assert!(selected_descriptors(&device, &selected, None).is_err());
    let mut missing = device;
    missing
        .capabilities
        .remove(&CapabilityId::new(ferrum_runtime::PHILOX_NORMAL_BF16_CAPABILITY_ID).unwrap());
    assert!(selected_descriptors(
        &missing,
        &selected,
        Some(CudaPhiloxDeviceGeometry::new(2, 768).unwrap())
    )
    .is_err());
}

#[test]
fn philox_selection_retains_geometry_in_implementation_and_estimator_identity() {
    let device = device();
    let selected = [CudaTensorOperation::PhiloxNormal];
    let describe = |sms, threads| {
        selected_descriptors(
            &device,
            &selected,
            Some(CudaPhiloxDeviceGeometry::new(sms, threads).unwrap()),
        )
        .unwrap()
    };
    let first = describe(2, 768);
    let repeated = describe(2, 768);
    let other_sms = describe(3, 768);
    let other_threads = describe(2, 1024);
    assert_eq!(
        first[0].descriptor().provider_implementation_fingerprint(),
        repeated[0]
            .descriptor()
            .provider_implementation_fingerprint()
    );
    for other in [&other_sms, &other_threads] {
        assert_eq!(
            first[0].descriptor().operation_fingerprint(),
            other[0].descriptor().operation_fingerprint()
        );
        assert_ne!(
            first[0].descriptor().provider_implementation_fingerprint(),
            other[0].descriptor().provider_implementation_fingerprint()
        );
        assert_ne!(
            first[0]
                .descriptor()
                .resource_estimator_implementation_fingerprint(),
            other[0]
                .descriptor()
                .resource_estimator_implementation_fingerprint()
        );
    }
}

#[test]
#[ignore = "requires an exclusive CUDA window; actual shared module permit and provider retirement"]
fn mixed_philox_bundle_holds_one_module_until_its_last_provider_retires() {
    use crate::backend::cuda::vnext_runtime::CudaTensorRuntimeConfig;
    let runtime = CudaDeviceRuntime::new_tensor_eager(CudaTensorRuntimeConfig {
        device: super::super::super::cuda_tensor_runtime_config(
            0,
            DeviceId::new("device.cuda.0").unwrap(),
        )
        .unwrap(),
        usable_memory_bytes: 2 << 30,
        bootstrap_memory_bytes: 1 << 30,
        stream_memory_bytes: 64 << 20,
        maximum_streams: 1,
        maximum_in_flight_per_stream: 1,
        maximum_commands_per_submission: 8,
        opaque_memory_profile: "experimental-unqualified-mixed-philox-module-test-v1".into(),
    })
    .unwrap();
    // Rejection occurs before module loading and does not strand its permit.
    assert!(providers(
        &runtime,
        &[
            CudaTensorOperation::PhiloxNormal,
            CudaTensorOperation::PhiloxNormal
        ]
    )
    .is_err());
    drop(runtime.reserve_tensor_provider().unwrap());
    let mut providers = providers(
        &runtime,
        &[
            CudaTensorOperation::Silu,
            CudaTensorOperation::PhiloxNormal,
            CudaTensorOperation::Tanh,
        ],
    )
    .unwrap();
    assert!(runtime.reserve_tensor_provider().is_err());
    let philox = providers.remove(1);
    assert_eq!(
        philox.descriptor().operation_id().as_str(),
        ferrum_runtime::PHILOX_NORMAL_BF16_OPERATION_ID
    );
    drop(providers);
    assert!(
        runtime.reserve_tensor_provider().is_err(),
        "remaining Philox provider must retain the shared module"
    );
    drop(philox);
    drop(runtime.reserve_tensor_provider().unwrap());
    // The compatible standalone factory uses the same selection/lifetime path.
    let standalone = super::super::super::cuda_vnext_philox_normal_bf16_provider(&runtime).unwrap();
    assert!(runtime.reserve_tensor_provider().is_err());
    drop(standalone);
    drop(runtime.reserve_tensor_provider().unwrap());
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
