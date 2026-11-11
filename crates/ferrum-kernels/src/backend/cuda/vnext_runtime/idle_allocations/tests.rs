use super::*;
use ferrum_runtime::{BufferUsage, DeviceCapacityOwner, DeviceOpaqueMemoryKind, ResourceId};

const MIB: u64 = 1 << 20;
const CEILING: u64 = 1 << 30;
const BOOTSTRAP: u64 = 256 << 20;

fn policy(total: u64, entry: u64, count: usize) -> CudaNativeAllocationReuse {
    CudaNativeAllocationReuse::Bounded {
        maximum_idle_bytes: NonZeroU64::new(total).unwrap(),
        maximum_entry_bytes: NonZeroU64::new(entry).unwrap(),
        maximum_entries: NonZeroUsize::new(count).unwrap(),
    }
}

#[test]
fn native_allocation_reuse_limits_and_default_are_explicit() {
    let options = CudaTensorRuntimeOptions::default();
    assert_eq!(
        options.native_allocation_reuse,
        CudaNativeAllocationReuse::Off
    );
    assert_eq!(options.native_lane_reuse, CudaNativeLaneReuse::Off);
    assert_eq!(options.native_allocation_reuse.maximum_idle_bytes(), 0);
    assert!(policy(8, 9, 1).validate().is_err());
    assert!(policy(u64::MAX, 1, 1).validate().is_err());
    assert!(policy(8, 8, usize::MAX).validate().is_err());
    let selected = policy(4 << 30, 2 << 30, 16);
    selected.validate().unwrap();
    assert_eq!(selected.maximum_idle_bytes(), 4 << 30);
}

fn runtime(total: u64, entry: u64, count: usize) -> CudaDeviceRuntime {
    CudaDeviceRuntime::new_tensor_eager_with_options(
        CudaTensorRuntimeConfig {
            device: crate::backend::cuda::vnext_ops::cuda_tensor_runtime_config(
                0,
                DeviceId::new("device.cuda.0").unwrap(),
            )
            .unwrap(),
            usable_memory_bytes: CEILING,
            bootstrap_memory_bytes: BOOTSTRAP,
            stream_memory_bytes: 16 * MIB,
            maximum_streams: 1,
            maximum_in_flight_per_stream: 1,
            maximum_commands_per_submission: 8,
            opaque_memory_profile: "unqualified-native-allocation-reuse-fixture".into(),
        },
        CudaTensorRuntimeOptions {
            native_allocation_reuse: policy(total, entry, count),
            ..Default::default()
        },
    )
    .unwrap()
}

// Native lifecycle fixture under original explicit opaque allowances, not a
// fabricated DeviceAllocationPermit. The separate integration fixture uses
// actual compiled Core sessions and fresh initialization on each request.
fn buffer(
    runtime: &CudaDeviceRuntime,
    label: &str,
    bytes: u64,
    alignment: u64,
    reused: Option<CudaSlice<u8>>,
) -> (CudaDeviceBuffer, DeviceCapacityRetention) {
    let claim = runtime.reserve_tensor_opaque(bytes).unwrap();
    let base = reused.unwrap_or_else(|| {
        runtime
            .tensor_allocation_pool()
            .unwrap()
            .allocate(bytes as usize)
            .unwrap()
    });
    runtime.allocation_stream.synchronize().unwrap();
    let (pointer, guard) = base.device_ptr(&runtime.allocation_stream);
    drop(guard);
    assert_eq!(pointer % alignment, 0);
    let buffer = CudaDeviceBuffer {
        descriptor: BufferDescriptor {
            resource_id: ResourceId::new(label).unwrap(),
            size_bytes: bytes,
            alignment_bytes: alignment,
            usage: BufferUsage::Activations,
            element_type: ElementType::U8,
        },
        runtime_instance: runtime.runtime_instance,
        allocation: Arc::new(CudaAllocation {
            _base: Some(base),
            idle_allocation_return: Some(AllocationReturn {
                cache: Arc::downgrade(runtime.idle_native_allocations.as_ref().unwrap()),
                alignment,
            }),
            aligned_ptr: pointer,
            requested_bytes: bytes,
            _tensor_owner: runtime.tensor_eager.clone(),
        }),
    };
    (buffer, claim)
}

fn fill(runtime: &CudaDeviceRuntime, buffer: &CudaDeviceBuffer, value: u8) {
    unsafe {
        cudarc::driver::result::memset_d8_async(
            buffer.allocation.aligned_ptr,
            value,
            buffer.descriptor.size_bytes as usize,
            runtime.allocation_stream.cu_stream(),
        )
        .unwrap();
    }
    runtime.allocation_stream.synchronize().unwrap();
    let bytes = runtime
        .allocation_stream
        .clone_dtoh(buffer.allocation._base.as_ref().unwrap())
        .unwrap();
    assert_eq!(bytes.len(), buffer.descriptor.size_bytes as usize);
    assert!(bytes.iter().all(|&byte| byte == value));
}

#[test]
#[ignore = "requires exclusive CUDA; actual exact backing, last-alias retirement and original quota"]
fn native_allocation_reuse_exact_extent_fresh_identity_and_retirement() {
    let runtime = runtime(4 * MIB, 2 * MIB, 16);
    let cache = runtime.idle_native_allocations.as_ref().unwrap();
    let capacity = DeviceCapacityOwner::open(runtime.descriptor(), CEILING).unwrap();
    let remaining = capacity
        .reserve(
            DeviceOpaqueMemoryKind::Bootstrap,
            CEILING - BOOTSTRAP - 4 * MIB,
        )
        .unwrap();
    assert!(
        capacity
            .reserve(DeviceOpaqueMemoryKind::Bootstrap, 1)
            .is_err(),
        "empty cache still consumes its fixed reservation"
    );
    drop(remaining);

    let (first, claim) = buffer(&runtime, "resource.native.first", 2 * MIB, 256, None);
    fill(&runtime, &first, 0xa5);
    let pointer = first.allocation.aligned_ptr;
    let old_identity = Arc::downgrade(&first.allocation);
    let region = first.region(0..2 * MIB).unwrap();
    drop(first);
    assert!(
        cache.take((2 * MIB) as usize, 256).unwrap().is_none(),
        "live region must prohibit recycling"
    );
    drop(region);
    assert!(old_identity.upgrade().is_none());
    assert_eq!(cache.entries.lock().unwrap().len(), 1);
    assert!(cache.take((2 * MIB) as usize, 128).unwrap().is_none());
    assert!(cache.take(MIB as usize, 256).unwrap().is_none());
    drop(claim);

    let base = cache.take((2 * MIB) as usize, 256).unwrap().unwrap();
    let (second, claim) = buffer(&runtime, "resource.native.second", 2 * MIB, 256, Some(base));
    assert_eq!(second.allocation.aligned_ptr, pointer);
    assert_ne!(Arc::as_ptr(&second.allocation), old_identity.as_ptr());
    assert_eq!(
        runtime.buffer_descriptor(&second).resource_id.as_str(),
        "resource.native.second"
    );
    fill(&runtime, &second, 0); // Full new initialization, no old-content authority.
    fill(&runtime, &second, 0x3c);
    let weak_cache = Arc::downgrade(cache);
    let weak_state = Arc::downgrade(runtime.tensor_eager.as_ref().unwrap());
    drop(runtime);
    assert!(weak_cache.upgrade().is_none());
    assert!(weak_state.upgrade().is_some());
    drop(second); // Late return cannot resurrect the cache.
    assert!(weak_state.upgrade().is_none());
    drop(claim);
}

#[test]
#[ignore = "requires exclusive CUDA; actual bounded largest-first eviction and separate contexts"]
fn native_allocation_reuse_retains_two_large_blocks_with_exact_budget() {
    let runtime = runtime(4 * MIB, 2 * MIB, 2);
    let cache = runtime.idle_native_allocations.as_ref().unwrap();
    let (small, a) = buffer(&runtime, "resource.native.small", MIB / 4, 256, None);
    let (first, b) = buffer(&runtime, "resource.native.large1", 2 * MIB, 256, None);
    let (second, c) = buffer(&runtime, "resource.native.large2", 2 * MIB, 256, None);
    let (oversized, d) = buffer(&runtime, "resource.native.oversized", 3 * MIB, 256, None);
    let expected = [first.allocation.aligned_ptr, second.allocation.aligned_ptr];
    drop(small);
    drop(first);
    drop(second); // Evicts and explicitly retires the small cached backing.
    drop(oversized); // Above single-entry limit; original retirement path.
    drop((a, b, c, d));
    let entries = cache.entries.lock().unwrap();
    assert_eq!(entries.len(), 2);
    assert_eq!(entries.iter().map(|e| e.bytes as u64).sum::<u64>(), 4 * MIB);
    assert!(entries
        .iter()
        .all(|entry| entry.bytes as u64 == 2 * MIB && expected.contains(&entry.pointer.unwrap())));
    drop(entries);
    let foreign = self::runtime(4 * MIB, 2 * MIB, 2);
    assert!(foreign
        .idle_native_allocations
        .as_ref()
        .unwrap()
        .take((2 * MIB) as usize, 256)
        .unwrap()
        .is_none());
    assert_eq!(cache.entries.lock().unwrap().len(), 2);
    let state = Arc::downgrade(runtime.tensor_eager.as_ref().unwrap());
    drop(runtime);
    assert!(
        state.upgrade().is_none(),
        "normal cache close retires native bytes before original pool owner"
    );
}

#[test]
#[ignore = "requires exclusive CUDA; injected retirement uncertainty intentionally retains the original bounded owners"]
fn native_allocation_reuse_failed_free_keeps_budget_after_runtime_drop_without_retry() {
    let runtime = runtime(MIB, MIB / 4, 4);
    let cache = runtime.idle_native_allocations.as_ref().unwrap();
    let buffers = (0..4)
        .map(|i| {
            buffer(
                &runtime,
                &format!("resource.native.failure{i}"),
                MIB / 4,
                256,
                None,
            )
        })
        .collect::<Vec<_>>();
    drop(buffers);
    let mut entries = std::mem::take(&mut *cache.entries.lock().unwrap());
    assert_eq!(entries.len(), 4);
    let owner = Arc::downgrade(&cache.owner);
    let state = Arc::downgrade(runtime.tensor_eager.as_ref().unwrap());
    let capacity = DeviceCapacityOwner::open(runtime.descriptor(), CEILING).unwrap();
    for (case, entry) in entries.iter_mut().enumerate() {
        let mut calls = 0;
        let outcome = catch_unwind(AssertUnwindSafe(|| {
            entry.retire_with(|pointer, stream| {
                calls += 1;
                if case == 1 || case == 3 {
                    stream.context().bind_to_thread()?;
                    unsafe {
                        cudarc::driver::result::free_async(pointer, stream.cu_stream())?;
                    }
                    // Case 1 reports an unknown synchronization outcome after a
                    // real enqueue; case 3 unwinds after that same boundary.
                }
                if case >= 2 {
                    panic!("injected native retirement unwind");
                }
                Err(DriverError(
                    cudarc::driver::sys::CUresult::CUDA_ERROR_UNKNOWN,
                ))
            })
        }));
        if case >= 2 {
            assert!(outcome.is_err());
        } else {
            assert!(outcome.unwrap().is_err());
        }
        assert_eq!(calls, 1);
        assert!(entry.pointer.is_none());
        entry
            .retire_with(|_, _| panic!("uncertain free must never be retried"))
            .unwrap();
    }
    assert!(cache.owner.poisoned.load(Ordering::Acquire));
    assert!(cache.take((MIB / 4) as usize, 256).is_err());
    drop(entries);
    drop(runtime);
    let retained = owner
        .upgrade()
        .expect("unknown free retains reservation after runtime drop");
    assert_eq!(retained._capacity.bytes(), MIB);
    assert!(state.upgrade().is_some());
    let remaining = capacity
        .reserve(DeviceOpaqueMemoryKind::Bootstrap, CEILING - BOOTSTRAP - MIB)
        .unwrap();
    assert!(capacity
        .reserve(DeviceOpaqueMemoryKind::Bootstrap, 1)
        .is_err());
    drop(remaining);
    // Deliberate fail-closed process-lifetime retention; no second native free,
    // fabricated successful retirement or cache-budget refund is attempted.
}
