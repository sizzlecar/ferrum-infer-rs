use super::*;
use cudarc::driver::CudaContext;
use ferrum_runtime::{
    DeviceCapacityOwner, DeviceClass, DeviceDescriptor, DeviceOpaqueMemoryKind,
    DynamicStorageAllocator, DynamicStorageProfile, DynamicStorageView,
};
use std::collections::BTreeSet;

#[test]
#[ignore = "requires a CUDA device with stream-ordered allocation support"]
fn private_tensor_pool_retains_reuses_and_trims_without_changing_current_pool() {
    const MIB: usize = 1024 * 1024;
    let context = CudaContext::new(0).unwrap();
    assert!(context.has_async_alloc());
    // Match the runtime's explicit stream/fence ownership policy.
    unsafe { context.disable_event_tracking() };
    let stream = context.new_stream().unwrap();
    let total = context.mem_get_info().unwrap().1 as u64;
    let descriptor = DeviceDescriptor {
        id: ferrum_runtime::DeviceId::new("device.cuda.private-pool-test").unwrap(),
        class: DeviceClass::Accelerator,
        ordinal: 0,
        total_memory_bytes: total,
        runtime_implementation_fingerprint: "b".repeat(64),
        capabilities: BTreeSet::new(),
        dynamic_storage_profiles: BTreeSet::from([DynamicStorageProfile::new(
            DynamicStorageAllocator::LinearArena,
            DynamicStorageView::Contiguous,
        )
        .unwrap()]),
    };
    let capacity = DeviceCapacityOwner::open(&descriptor, 128 * MIB as u64).unwrap();
    let bootstrap = capacity
        .reserve(DeviceOpaqueMemoryKind::Bootstrap, MIB as u64)
        .unwrap();
    let observe_current = || {
        super::super::super::allocation_pool_snapshot(&context)
            .unwrap()
            .unwrap()
    };
    let before = observe_current();
    let pool = TensorAllocationPool::new(
        stream.clone(),
        64 * MIB as u64,
        bootstrap.clone(),
        capacity.clone(),
    )
    .unwrap();
    let second =
        TensorAllocationPool::new(stream.clone(), 32 * MIB as u64, bootstrap, capacity.clone())
            .unwrap();
    assert_eq!(
        pool.snapshot().unwrap().release_threshold_bytes,
        64 * MIB as u64
    );
    assert_eq!(
        second.snapshot().unwrap().release_threshold_bytes,
        32 * MIB as u64
    );
    assert_ne!(
        pool.snapshot().unwrap().pool_identity,
        second.snapshot().unwrap().pool_identity
    );
    assert_ne!(pool.snapshot().unwrap().pool_identity, before.pool_identity);

    let mut live = pool.allocate(8 * MIB).unwrap();
    stream.memset_zeros(&mut live).unwrap();
    let cached = pool.allocate(16 * MIB).unwrap();
    stream.synchronize().unwrap();
    drop(cached);
    stream.synchronize().unwrap();
    let retained = pool.snapshot().unwrap();
    assert!(retained.reserved_bytes.saturating_sub(retained.used_bytes) >= 16 * MIB as u64);
    let reused = pool.allocate(16 * MIB).unwrap();
    stream.synchronize().unwrap();
    assert_eq!(
        pool.snapshot().unwrap().reserved_bytes,
        retained.reserved_bytes
    );
    drop(reused);
    stream.synchronize().unwrap();
    pool.trim_idle().unwrap();
    assert_eq!(
        stream.clone_dtoh(&live.slice(..16)).unwrap(),
        vec![0_u8; 16]
    );
    assert!(pool.snapshot().unwrap().used_bytes >= 8 * MIB as u64);
    assert_eq!(second.snapshot().unwrap().used_bytes, 0);
    drop(live);
    stream.synchronize().unwrap();
    pool.trim_idle().unwrap();
    assert_eq!(pool.snapshot().unwrap().used_bytes, 0);
    assert_eq!(pool.snapshot().unwrap().reserved_bytes, 0);
    drop((pool, second));
    let after = observe_current();
    assert_eq!(after.pool_identity, before.pool_identity);
    assert_eq!(
        after.release_threshold_bytes,
        before.release_threshold_bytes
    );
}

#[test]
#[ignore = "requires a CUDA device with stream-ordered allocation support"]
fn private_tensor_pool_outlives_runtime_and_yields_cache_to_native_owners() {
    use crate::backend::cuda::vnext_runtime::{
        CudaAllocation, CudaDeviceBuffer, CudaDeviceRuntime, CudaTensorRuntimeConfig,
    };
    use cudarc::driver::DevicePtr;
    use ferrum_runtime::{BufferDescriptor, BufferUsage, DeviceId, ElementType, ResourceId};

    const MIB: usize = 1024 * 1024;
    let runtime = CudaDeviceRuntime::new_tensor_eager(CudaTensorRuntimeConfig {
        device: crate::backend::cuda::vnext_ops::cuda_tensor_runtime_config(
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
        opaque_memory_profile: "test.cuda.private-pool-lifetime.v1".into(),
    })
    .unwrap();
    let state = Arc::downgrade(runtime.tensor_eager.as_ref().unwrap());
    let stream = runtime.allocation_stream.clone();
    let pool = runtime.tensor_allocation_pool().unwrap();
    let threshold = pool.snapshot().unwrap().release_threshold_bytes;
    assert_eq!(threshold, (1 << 30) - (64 << 20));

    // Exercise the real allocation owner/drop order independently of plan
    // construction. This fixture does not assert Core admission behavior.
    let storage_allowance = runtime.reserve_tensor_opaque(8 * MIB as u64).unwrap();
    let mut base = pool.allocate(8 * MIB).unwrap();
    stream.memset_zeros(&mut base).unwrap();
    stream.synchronize().unwrap();
    // A small idle allocation can share an indivisible driver backing block
    // with live storage. Expand beyond the observed live backing so this test
    // exercises releasable blocks without hardcoding CUDA's page granularity.
    let cached_bytes = pool
        .snapshot()
        .unwrap()
        .reserved_bytes
        .checked_mul(2)
        .unwrap();
    assert!(cached_bytes < threshold);
    let cached = pool
        .allocate(usize::try_from(cached_bytes).unwrap())
        .unwrap();
    stream.synchronize().unwrap();
    drop(cached);
    stream.synchronize().unwrap();
    let before = pool.snapshot().unwrap();
    assert!(before.reserved_bytes.saturating_sub(before.used_bytes) >= cached_bytes);
    for repetition in 0..2 {
        let native = runtime.reserve_tensor_opaque(MIB as u64).unwrap();
        let after = pool.snapshot().unwrap();
        eprintln!("native owner trim {repetition}: before={before:?}, after={after:?}");
        assert!(
            after.reserved_bytes < before.reserved_bytes,
            "before={before:?}, after={after:?}"
        );
        assert_eq!(after.release_threshold_bytes, threshold);
        drop(native);
    }
    let (aligned_ptr, use_guard) = base.device_ptr(&stream);
    drop(use_guard);
    let buffer = CudaDeviceBuffer {
        descriptor: BufferDescriptor {
            resource_id: ResourceId::new("resource.cuda.private-pool-lifetime").unwrap(),
            size_bytes: 8 * MIB as u64,
            alignment_bytes: 16,
            usage: BufferUsage::Activations,
            element_type: ElementType::Bf16,
        },
        runtime_instance: runtime.runtime_instance,
        allocation: Arc::new(CudaAllocation {
            _base: base,
            aligned_ptr,
            requested_bytes: 8 * MIB as u64,
            _tensor_owner: runtime.tensor_eager.clone(),
        }),
    };
    drop(runtime);
    assert!(state.upgrade().is_some());
    assert_eq!(
        stream
            .clone_dtoh(&buffer.allocation._base.slice(..16))
            .unwrap(),
        vec![0_u8; 16]
    );
    drop(buffer);
    assert!(state.upgrade().is_none());
    drop(storage_allowance);
}
