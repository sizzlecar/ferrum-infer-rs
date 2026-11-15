//! Real driver refusal in a small private pool; no whole-device exhaustion.
use super::*;
use crate::backend::cuda::{vnext_ops::*, vnext_runtime::*};
use ferrum_runtime::*;

#[path = "../../../../../../tests/support/dynamic_preparation_qualification.rs"]
mod checks;
#[path = "../../../../../../tests/support/tensor_numeric_component.rs"]
mod fixture;

#[test]
fn allocation_failure_classification_requires_native_allocation_evidence() {
    use cudarc::driver::sys::CUresult::*;
    for (error, expected) in [
        (
            CudaDeviceRuntimeError::Allocation(DriverError(CUDA_ERROR_OUT_OF_MEMORY)),
            DeviceAllocationFailureKind::Capacity,
        ),
        (
            CudaDeviceRuntimeError::Allocation(DriverError(CUDA_ERROR_ILLEGAL_ADDRESS)),
            DeviceAllocationFailureKind::Unrecoverable,
        ),
        (
            CudaDeviceRuntimeError::driver(
                "synchronization",
                DriverError(CUDA_ERROR_OUT_OF_MEMORY),
            ),
            DeviceAllocationFailureKind::Unrecoverable,
        ),
        (
            CudaDeviceRuntimeError::contract("CUDA_ERROR_OUT_OF_MEMORY"),
            DeviceAllocationFailureKind::Unrecoverable,
        ),
    ] {
        assert_eq!(error.allocation_failure_kind(), expected);
    }
}

struct NativePressure {
    buffer: Option<CudaSlice<u8>>,
    stream: Arc<CudaStream>,
    _claim: DeviceCapacityRetention,
}
impl Drop for NativePressure {
    fn drop(&mut self) {
        drop(self.buffer.take());
        self.stream.synchronize().unwrap();
    }
}

#[test]
#[ignore = "requires CUDA; actual private-pool allocation OOM, B1 recovery and B2 after release"]
fn native_pool_capacity_refusal_preserves_original_b1_and_later_b2() {
    const MIB: u64 = 1 << 20;
    const NATIVE_LIMIT: u64 = 64 * MIB;
    const CEILING: u64 = 1024 * MIB;
    let mut runtime = CudaDeviceRuntime::new_tensor_eager(CudaTensorRuntimeConfig {
        device: cuda_tensor_runtime_config(0, fixture::id("device.cuda.0")).unwrap(),
        usable_memory_bytes: CEILING,
        bootstrap_memory_bytes: 256 * MIB,
        stream_memory_bytes: 16 * MIB,
        maximum_streams: 1,
        maximum_in_flight_per_stream: 1,
        maximum_commands_per_submission: 8,
        opaque_memory_profile: "native-allocation-pressure-qualification".into(),
    })
    .unwrap();
    // Limit only this test's driver pool. The ordinary Core budget remains
    // ample, so the refusal cannot be satisfied by a logical quota check.
    let old = runtime.tensor_allocation_pool().unwrap();
    let mut properties = unsafe { std::mem::zeroed::<sys::CUmemPoolProps>() };
    properties.allocType = sys::CUmemAllocationType::CU_MEM_ALLOCATION_TYPE_PINNED;
    properties.handleTypes = sys::CUmemAllocationHandleType::CU_MEM_HANDLE_TYPE_NONE;
    properties.location.type_ = sys::CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE;
    properties.location.id = old.stream.context().cu_device();
    properties.maxSize = NATIVE_LIMIT as usize;
    old.stream.context().bind_to_thread().unwrap();
    let pool = TensorAllocationPool {
        pool: unsafe { result::mem_pool::create(&properties).unwrap() },
        stream: old.stream.clone(),
        bootstrap: old.bootstrap.clone(),
        capacity: old.capacity.clone(),
    };
    Arc::get_mut(runtime.tensor_eager.as_mut().unwrap())
        .unwrap()
        .allocation_pool = Some(pool);
    let runtime = Arc::new(runtime);
    let registry = OperationRuntimeRegistry::new(
        vec![Box::new(LeadingBatchBf16Kind::Merge.contract().unwrap())],
        cuda_vnext_tensor_providers(&runtime, &[CudaTensorOperation::MergeLeadingBatch]).unwrap(),
    )
    .unwrap();
    let materializers = WeightMaterializerRegistry::identity_only().unwrap();
    let d = runtime.descriptor();
    let engine = EngineProviderDescriptor::new(
        fixture::id("provider.engine.native_prepare"),
        ContractVersion::new(1, 0),
        d.runtime_implementation_fingerprint.clone(),
        d.id.clone(),
        d.capabilities.clone(),
    )
    .unwrap();
    let catalog = materializers
        .augment_catalog(
            registry
                .capability_catalog(d.clone(), vec![engine])
                .unwrap(),
        )
        .unwrap();
    let pressure = || -> Box<dyn std::any::Any> {
        let pool = runtime.tensor_allocation_pool().unwrap();
        // Leave the live B1 plus two MiB. Releasing that B1 admits the same
        // B1 again but not its doubled B2 work. Dropping pressure admits B2.
        let bytes = NATIVE_LIMIT
            .checked_sub(pool.snapshot().unwrap().used_bytes + 2 * MIB)
            .unwrap();
        let claim = DeviceCapacityOwner::open(runtime.descriptor(), CEILING)
            .unwrap()
            .reserve(DeviceOpaqueMemoryKind::Bootstrap, bytes)
            .unwrap();
        let mut buffer = pool.allocate(bytes as usize).unwrap();
        pool.stream.memset_zeros(&mut buffer).unwrap();
        pool.stream.synchronize().unwrap();
        Box::new(NativePressure {
            buffer: Some(buffer),
            stream: pool.stream.clone(),
            _claim: claim,
        })
    };
    checks::run(
        runtime.clone(),
        &registry,
        &catalog,
        &materializers,
        CEILING,
        None,
        Some(&pressure),
    );
    assert_eq!(
        runtime
            .tensor_allocation_pool()
            .unwrap()
            .snapshot()
            .unwrap()
            .used_bytes,
        0
    );
}
