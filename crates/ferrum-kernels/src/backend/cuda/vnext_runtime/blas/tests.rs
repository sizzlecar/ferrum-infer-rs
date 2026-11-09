use super::*;
use ferrum_runtime::{
    BufferUsage, DeviceCapacityOwner, DeviceCapacityRetention, DeviceOpaqueMemoryKind,
    DynamicStorageAllocator, DynamicStorageView, HostMemoryAccount, HostMemoryClass,
    HostMemoryClassLimits, HostMemoryLimits, ResourceId,
};
use std::sync::atomic::AtomicUsize;

fn capacity_owner() -> DeviceCapacityOwner {
    let id = NEXT_RUNTIME_INSTANCE.fetch_add(1, Ordering::Relaxed);
    DeviceCapacityOwner::open(
        &DeviceDescriptor {
            id: DeviceId::new(format!("device.test.lazy-blas-{id}")).unwrap(),
            class: DeviceClass::Accelerator,
            ordinal: 0,
            total_memory_bytes: 128,
            runtime_implementation_fingerprint: "c".repeat(64),
            capabilities: BTreeSet::new(),
            dynamic_storage_profiles: BTreeSet::from([DynamicStorageProfile::new(
                DynamicStorageAllocator::LinearArena,
                DynamicStorageView::Contiguous,
            )
            .unwrap()]),
        },
        128,
    )
    .unwrap()
}

#[test]
fn constructor_error_or_panic_retains_original_capacity_and_permanently_rejects_retry() {
    for panic in [false, true] {
        let capacity = capacity_owner();
        let owner = Arc::new(
            capacity
                .reserve(DeviceOpaqueMemoryKind::Stream, 32)
                .unwrap(),
        );
        let state = CudaStreamState::new();
        let calls = AtomicUsize::new(0);
        let result = catch_unwind(AssertUnwindSafe(|| {
            create_dependency::<(), _>(&state, Arc::clone(&owner), || {
                calls.fetch_add(1, Ordering::Relaxed);
                if panic {
                    panic!("injected native constructor unwind");
                }
                Err(CudaDeviceRuntimeError::contract(
                    "injected partial constructor failure",
                ))
            })
        }));
        assert_eq!(result.is_err(), panic);
        if let Ok(result) = result {
            assert!(result.is_err());
        }
        assert_eq!(state.snapshot(), StreamState::Failed);
        assert!(create_dependency(&state, Arc::clone(&owner), || {
            calls.fetch_add(1, Ordering::Relaxed);
            Ok(())
        })
        .is_err());
        assert_eq!(calls.load(Ordering::Relaxed), 1);
        drop(owner);
        // Constructor cleanup is unknown: neither facade loss nor retry
        // rejection may return its original 32-byte native reservation.
        assert!(capacity
            .reserve(DeviceOpaqueMemoryKind::Stream, 97)
            .is_err());
        drop(
            capacity
                .reserve(DeviceOpaqueMemoryKind::Stream, 96)
                .unwrap(),
        );
        state.synchronized();
        assert_eq!(state.snapshot(), StreamState::Failed);
    }
}

#[test]
fn successful_creation_keeps_one_original_allowance_and_recording_rejects_before_constructor() {
    let capacity = capacity_owner();
    let owner = Arc::new(
        capacity
            .reserve(DeviceOpaqueMemoryKind::Stream, 32)
            .unwrap(),
    );
    let state = CudaStreamState::new();
    state.begin_submission().unwrap();
    assert!(create_dependency::<(), _>(&state, Arc::clone(&owner), || {
        panic!("recording stream must reject before native construction")
    })
    .is_err());
    state.recording.store(false, Ordering::Release);
    let (native, retained) = create_dependency(&state, Arc::clone(&owner), || Ok(17_u32)).unwrap();
    assert_eq!(native, 17);
    assert!(Arc::ptr_eq(&owner, &retained));
    assert!(capacity
        .reserve(DeviceOpaqueMemoryKind::Stream, 97)
        .is_err());
    drop((retained, owner));
    drop(
        capacity
            .reserve(DeviceOpaqueMemoryKind::Stream, 128)
            .unwrap(),
    );
    assert!(state.is_quiescent());
}

fn tensor_runtime() -> CudaDeviceRuntime {
    CudaDeviceRuntime::new_tensor_eager(CudaTensorRuntimeConfig {
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
        opaque_memory_profile: "experimental-unqualified-lazy-blas-test-v1".into(),
    })
    .unwrap()
}

fn host() -> HostMemoryAccount {
    HostMemoryAccount::new(
        HostMemoryLimits::new(
            12,
            HostMemoryClassLimits {
                input_preprocess_bytes: 0,
                transfer_staging_bytes: 0,
                output_bytes: 0,
                weight_backing_bytes: 12,
            },
        )
        .unwrap(),
    )
}

// Native owner fixture: the real opaque reservation accompanies this tiny
// allocation. This tests backend enqueue/retirement, not Core plan admission.
fn buffer(
    runtime: &CudaDeviceRuntime,
    stream: &CudaDeviceStream,
) -> (CudaDeviceBuffer, DeviceCapacityRetention) {
    let quota = runtime.reserve_tensor_opaque(12).unwrap();
    let base = stream.stream.alloc_zeros::<u8>(12).unwrap();
    let (pointer, guard) = base.device_ptr(&stream.stream);
    drop(guard);
    (
        CudaDeviceBuffer {
            descriptor: BufferDescriptor {
                resource_id: ResourceId::new("resource.test.lazy-blas").unwrap(),
                size_bytes: 12,
                alignment_bytes: 4,
                usage: BufferUsage::Weights,
                element_type: ElementType::F32,
            },
            runtime_instance: runtime.runtime_instance,
            allocation: Arc::new(CudaAllocation {
                _base: Some(base),
                idle_allocation_return: None,
                aligned_ptr: pointer,
                requested_bytes: 12,
                _tensor_owner: runtime.tensor_eager.clone(),
            }),
        },
        quota,
    )
}

fn upload(
    runtime: &CudaDeviceRuntime,
    buffer: &CudaDeviceBuffer,
    host: &HostMemoryAccount,
) -> CudaDeviceCommand {
    let mut source = host
        .allocate_zeroed(HostMemoryClass::WeightBacking, 12)
        .unwrap();
    for (chunk, value) in source.as_mut().chunks_exact_mut(4).zip([2_f32, 3., 0.]) {
        chunk.copy_from_slice(&value.to_le_bytes());
    }
    runtime
        .encode_managed_upload(
            source.freeze(),
            HostTransferLayout::new(ElementType::F32, 3).unwrap(),
            buffer,
            0,
        )
        .unwrap()
        .unwrap()
}

#[test]
#[ignore = "requires an exclusive CUDA window; real transfer and cuBLAS execution"]
fn typed_transfers_do_not_create_blas_and_first_compute_initializes_once() {
    let runtime = tensor_runtime();
    assert_eq!(
        runtime.static_weight_source_retention(),
        ferrum_runtime::StaticWeightSourceRetention::FencedCopy
    );
    assert!(runtime.supports_managed_static_weight_sources());
    assert!(runtime.begin_managed_static_weight_import().is_none());
    let mut stream = runtime.create_stream().unwrap();
    assert!(stream.blas.is_none());
    let (buffer, quota) = buffer(&runtime, &stream);
    // Keep real pool storage live while filling independently releasable idle
    // backing. The original allowances cover these backend-owner fixtures.
    const MIB: usize = 1024 * 1024;
    let pool = runtime.tensor_allocation_pool().unwrap();
    let live_allowance = runtime.reserve_tensor_opaque(8 * MIB as u64).unwrap();
    let mut live = pool.allocate(8 * MIB).unwrap();
    runtime.allocation_stream.memset_zeros(&mut live).unwrap();
    runtime.allocation_stream.synchronize().unwrap();
    let cached_bytes = pool
        .snapshot()
        .unwrap()
        .reserved_bytes
        .checked_mul(2)
        .unwrap();
    let cached_allowance = runtime.reserve_tensor_opaque(cached_bytes).unwrap();
    let threshold = pool.snapshot().unwrap().release_threshold_bytes;
    assert!(cached_bytes < threshold);
    let cached = pool
        .allocate(usize::try_from(cached_bytes).unwrap())
        .unwrap();
    runtime.allocation_stream.synchronize().unwrap();
    drop(cached);
    runtime.allocation_stream.synchronize().unwrap();
    let cached_before = pool.snapshot().unwrap();
    assert!(cached_before.reserved_bytes - cached_before.used_bytes >= cached_bytes);
    let lane = stream.tensor_lane.as_ref().unwrap().clone();
    let host = host();
    let mut zero = runtime.encode_zero(&buffer, 0, 12).unwrap();
    let upload = upload(&runtime, &buffer, &host);
    assert!(zero.tensor_transfer.is_some() && upload.tensor_transfer.is_some());
    let needs_blas = submission_requires_blas([&zero, &upload]);
    assert!(!needs_blas);
    runtime
        .ensure_submission_blas(&mut stream, needs_blas)
        .unwrap();
    assert_eq!(
        pool.snapshot().unwrap().reserved_bytes,
        cached_before.reserved_bytes
    );
    zero.tensor_flight = Some(
        stream
            .tensor_lane
            .as_ref()
            .unwrap()
            .reserve_flight()
            .unwrap(),
    );
    stream.state.begin_submission().unwrap();
    stream.enqueue_command(&zero).unwrap();
    stream.enqueue_command(&upload).unwrap();
    let event = stream.stream.record_event(None).unwrap();
    stream.state.submission_recorded().unwrap();
    let fence = CudaDeviceFence {
        event,
        timing: CudaFenceTiming::NotRequested,
        command_timing: CudaFenceCommandTiming::NotRequested,
        attribution: None,
        execution_facts: None,
        stream_state: Arc::clone(&stream.state),
        terminal_accounted: AtomicBool::new(false),
        _stream: Arc::clone(&stream.stream),
        _blas: stream.blas.clone(),
        _commands: vec![zero, upload],
    };
    assert!(matches!(
        runtime.wait_fence(&fence).unwrap().into_parts().0,
        DeviceTerminal::Succeeded
    ));
    assert!(stream.blas.is_none());
    assert_eq!(host.usage().unwrap().committed_bytes(), 12);
    assert!(fence._blas.is_none());
    drop(fence);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    let values = stream
        .stream
        .clone_dtoh(buffer.allocation._base.as_ref().unwrap())
        .unwrap();
    assert_eq!(f32::from_le_bytes(values[0..4].try_into().unwrap()), 2.0);
    assert_eq!(f32::from_le_bytes(values[4..8].try_into().unwrap()), 3.0);

    let compute = CudaDeviceCommand::operation_with_blas(
        "test.lazy-blas.gemm",
        vec![buffer.region(0..12).unwrap()],
        |_, blas, regions| {
            let base = regions[0].device_ptr;
            let alpha = 1_f32;
            let beta = 0_f32;
            unsafe {
                cudarc::cublas::sys::cublasSgemm_v2(
                    *blas.handle(),
                    cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N,
                    cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N,
                    1,
                    1,
                    1,
                    &alpha,
                    base as *const f32,
                    1,
                    (base + 4) as *const f32,
                    1,
                    &beta,
                    (base + 8) as *mut f32,
                    1,
                )
            }
            .result()
            .map_err(|e| CudaDeviceRuntimeError::blas("test 1x1 GEMM", e))
        },
    )
    .unwrap();
    // This backend-owner fixture has no Core state-transfer ticket. A legal
    // static zero tests mixed transfer/compute classification without claiming
    // to exercise the distinct retained StateCopy contract.
    let clear_output = runtime.encode_zero(&buffer, 8, 4).unwrap();
    assert!(!submission_requires_blas([&clear_output]));
    assert!(submission_requires_blas([&clear_output, &compute]));
    assert!(stream.enqueue_command(&compute).is_err());
    runtime
        .ensure_submission_blas(
            &mut stream,
            submission_requires_blas([&clear_output, &compute]),
        )
        .unwrap();
    let after_creation = pool.snapshot().unwrap();
    eprintln!(
        "first BLAS allocator preparation: before={cached_before:?}, after={after_creation:?}"
    );
    assert!(after_creation.reserved_bytes < cached_before.reserved_bytes);
    assert_eq!(after_creation.used_bytes, cached_before.used_bytes);
    assert_eq!(after_creation.release_threshold_bytes, threshold);
    assert_eq!(after_creation.pool_identity, cached_before.pool_identity);
    assert!(Arc::ptr_eq(&lane, stream.tensor_lane.as_ref().unwrap()));
    // Refill idle backing after the first native handle exists. Reusing that
    // handle must neither trim the pool again nor acquire another lane.
    let cached = pool
        .allocate(usize::try_from(cached_bytes).unwrap())
        .unwrap();
    runtime.allocation_stream.synchronize().unwrap();
    drop(cached);
    runtime.allocation_stream.synchronize().unwrap();
    let cached_again = pool.snapshot().unwrap();
    assert!(cached_again.reserved_bytes - cached_again.used_bytes >= cached_bytes);
    let first = stream.blas.clone().unwrap();
    for _ in 0..2 {
        runtime.ensure_submission_blas(&mut stream, true).unwrap();
        assert!(Arc::ptr_eq(&first, stream.blas.as_ref().unwrap()));
        assert!(Arc::ptr_eq(&lane, stream.tensor_lane.as_ref().unwrap()));
        let after_reuse = pool.snapshot().unwrap();
        assert_eq!(after_reuse.reserved_bytes, cached_again.reserved_bytes);
        assert_eq!(after_reuse.used_bytes, cached_again.used_bytes);
        assert_eq!(after_reuse.release_threshold_bytes, threshold);
        stream.enqueue_command(&clear_output).unwrap();
        stream.enqueue_command(&compute).unwrap();
        runtime.synchronize(&mut stream).unwrap();
        let values = stream
            .stream
            .clone_dtoh(buffer.allocation._base.as_ref().unwrap())
            .unwrap();
        assert_eq!(f32::from_le_bytes(values[8..12].try_into().unwrap()), 6.0);
    }
    assert_eq!(
        runtime
            .allocation_stream
            .clone_dtoh(&live.slice(..16))
            .unwrap(),
        vec![0_u8; 16]
    );
    drop(live);
    runtime.allocation_stream.synchronize().unwrap();
    drop((live_allowance, cached_allowance));
    drop((compute, clear_output, buffer, first, stream, lane));
    drop(quota);
}

#[test]
#[ignore = "requires an exclusive CUDA window; original transfer quarantine and Host ownership"]
fn transfer_only_unwind_retains_owners_without_blas_until_real_synchronization() {
    let runtime = tensor_runtime();
    assert_eq!(
        runtime.static_weight_source_retention(),
        ferrum_runtime::StaticWeightSourceRetention::FencedCopy
    );
    let mut stream = runtime.create_stream().unwrap();
    let (buffer, quota) = buffer(&runtime, &stream);
    let host = host();
    let mut command = upload(&runtime, &buffer, &host);
    command.tensor_flight = Some(
        stream
            .tensor_lane
            .as_ref()
            .unwrap()
            .reserve_flight()
            .unwrap(),
    );
    stream.state.begin_submission().unwrap();
    let result = catch_unwind(AssertUnwindSafe(|| {
        let mut guard = tensor_io::SubmissionGuard::new(&runtime, &stream, vec![command]);
        guard.arm();
        stream.enqueue_command(&guard[0]).unwrap();
        panic!("injected unwind after transfer enqueue");
    }));
    assert!(result.is_err());
    assert!(stream.blas.is_none());
    assert_eq!(stream.state.snapshot(), StreamState::Failed);
    assert_eq!(runtime.quarantined().len(), 1);
    assert_eq!(host.usage().unwrap().committed_bytes(), 12);
    assert!(runtime.create_stream().is_err());
    runtime.synchronize(&mut stream).unwrap();
    assert!(runtime.quarantined().is_empty());
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    assert!(runtime.ensure_submission_blas(&mut stream, true).is_err());
    assert!(stream.blas.is_none());
    drop((buffer, stream));
    drop(quota);
    assert!(runtime.create_stream().unwrap().blas.is_none());
}

#[test]
#[ignore = "requires a CUDA device; legacy creation and empty replay preparation"]
fn legacy_streams_keep_eager_blas_and_empty_replay_preparation_accepts_no_handle() {
    let runtime = CudaDeviceRuntime::new(
        crate::backend::cuda::vnext_ops::cuda_vnext_runtime_config(
            0,
            DeviceId::new("device.test.legacy-blas").unwrap(),
            AttentionExecutionPolicy::Portable,
        )
        .unwrap(),
    )
    .unwrap();
    assert_eq!(
        runtime.static_weight_source_retention(),
        ferrum_runtime::StaticWeightSourceRetention::UnknownOrPlanRetained
    );
    assert!(!runtime.supports_managed_static_weight_sources());
    let stream = runtime.create_stream().unwrap();
    assert!(stream.blas.is_some());
    let report = CudaExecutableCache::new()
        .prepare_all(
            &runtime.context,
            &stream.stream,
            None,
            &[],
            &[],
            true,
            ExecutionFactsCapture::Off,
            &DisabledDeviceSubmissionTimingSink,
        )
        .unwrap();
    assert_eq!(
        report,
        super::super::super::vnext_replay::CudaExecutablePreparation::default()
    );
}
