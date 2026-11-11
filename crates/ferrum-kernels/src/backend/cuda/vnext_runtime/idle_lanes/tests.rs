use super::*;

#[test]
fn reuse_limits_and_external_weak_owners_are_not_silent_permissions() {
    assert_eq!(CudaNativeLaneReuse::default(), CudaNativeLaneReuse::Off);
    assert!(IdleNativeLanes::new(CudaNativeLaneReuse::Off).is_none());
    let policy = CudaNativeLaneReuse::Bounded {
        maximum_idle_lanes: NonZeroUsize::new(2).unwrap(),
    };
    assert!(policy.validate(1).is_err());
    policy.validate(2).unwrap();
    let mut owner = Arc::new(7_u32);
    assert!(exclusively_owned(&mut owner));
    let weak = Arc::downgrade(&owner);
    assert!(!exclusively_owned(&mut owner));
    drop(weak);
    let other = Arc::clone(&owner);
    assert!(!exclusively_owned(&mut owner));
    drop(other);
    assert!(exclusively_owned(&mut owner));
}

fn runtime(maximum_streams: usize) -> CudaDeviceRuntime {
    runtime_with_replay(maximum_streams, false)
}

fn runtime_with_replay(maximum_streams: usize, replay: bool) -> CudaDeviceRuntime {
    let mut tensor = CudaTensorRuntimeConfig {
        device: crate::backend::cuda::vnext_ops::cuda_tensor_runtime_config(
            0,
            DeviceId::new("device.cuda.0").unwrap(),
        )
        .unwrap(),
        usable_memory_bytes: 2 << 30,
        bootstrap_memory_bytes: 1 << 30,
        stream_memory_bytes: 64 << 20,
        maximum_streams,
        maximum_in_flight_per_stream: 1,
        maximum_commands_per_submission: 8,
        opaque_memory_profile: "experimental-unqualified-retired-native-lane-test-v1".into(),
    };
    let reuse = CudaNativeLaneReuse::Bounded {
        maximum_idle_lanes: NonZeroUsize::new(1).unwrap(),
    };
    if replay {
        use super::super::tensor_eager::{
            CudaDenseF16CausalProfile, CudaTensorAndCausalRuntimeConfig,
        };
        tensor.device =
            crate::backend::cuda::vnext_ops::cuda_tensor_and_dense_f16_causal_runtime_config(
                0,
                DeviceId::new("device.cuda.0").unwrap(),
            )
            .unwrap();
        for id in [
            ferrum_runtime::DEVICE_REUSABLE_EXECUTION_CAPABILITY_ID,
            ferrum_runtime::DEVICE_ON_DEMAND_REUSABLE_EXECUTION_CAPABILITY_ID,
        ] {
            tensor
                .device
                .capabilities
                .insert(CapabilityId::new(id).unwrap());
        }
        let causal = CudaDenseF16CausalProfile::new(
            64 << 20,
            "unqualified-retired-native-owned-workspace-test-v1",
        )
        .unwrap()
        .with_decode_replay(2, 32 << 20)
        .unwrap();
        CudaDeviceRuntime::new_tensor_and_causal_eager_with_native_lane_reuse(
            CudaTensorAndCausalRuntimeConfig { tensor, causal },
            reuse,
        )
        .unwrap()
    } else {
        CudaDeviceRuntime::new_tensor_eager_with_native_lane_reuse(tensor, reuse).unwrap()
    }
}

// Backend native-owner fixture: the original explicit opaque claim covers
// these tiny buffers. It does not assert Core plan admission or Host accounting.
fn gemm(runtime: &CudaDeviceRuntime, stream: &mut CudaDeviceStream) {
    use cudarc::{cublas::sys, driver::DevicePtrMut};
    let quota = runtime.reserve_tensor_opaque(12).unwrap();
    let mut data = stream.stream.clone_htod(&[2_f32, 3., 0.]).unwrap();
    runtime.ensure_submission_blas(stream, true).unwrap();
    let (base, guard) = data.device_ptr_mut(&stream.stream);
    let alpha = 1_f32;
    let beta = 0_f32;
    unsafe {
        sys::cublasSgemm_v2(
            *stream.blas.as_ref().unwrap().handle(),
            sys::cublasOperation_t::CUBLAS_OP_N,
            sys::cublasOperation_t::CUBLAS_OP_N,
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
        .result()
        .unwrap();
    }
    drop(guard);
    runtime.synchronize(stream).unwrap();
    assert_eq!(stream.stream.clone_dtoh(&data).unwrap(), [2., 3., 6.]);
    drop(data);
    runtime.synchronize(stream).unwrap();
    drop(quota);
}

#[test]
#[ignore = "requires an exclusive CUDA window; real native pair, BLAS reset and original quota"]
fn retired_native_lane_reuses_exact_pair_and_quota_with_fresh_logical_state() {
    check_retired_native_lane(runtime(1));
}

#[test]
#[ignore = "requires an exclusive CUDA window; owned workspace survives native reuse and retires with its original quota"]
fn retired_native_lane_keeps_owned_workspace_without_retaining_a_cycle() {
    check_retired_native_lane(runtime_with_replay(1, true));
}

fn check_retired_native_lane(runtime: CudaDeviceRuntime) {
    use cudarc::{cublas::sys, driver::DevicePtrMut};
    let cache = runtime.idle_native_lanes.as_ref().unwrap();
    let mut first = runtime.create_stream().unwrap();
    gemm(&runtime, &mut first);
    let old_id = first.id;
    let old_stream = first.stream.cu_stream();
    let old_blas = *first.blas.as_ref().unwrap().handle();
    let workspace_bytes = first.blas.as_ref().unwrap().workspace_bytes();
    let old_lane = Arc::as_ptr(first.tensor_lane.as_ref().unwrap());
    let replay = if first.tensor_lane.as_ref().unwrap().replay_limit().is_some() {
        assert_eq!(workspace_bytes, 32 << 20);
        DeviceReusableExecutionPlan::on_demand(2)
    } else {
        DeviceReusableExecutionPlan::new(2)
    }
    .unwrap();
    first.executable_cache.configure(replay).unwrap();
    assert!(first.executable_cache.preparation().is_ok());
    let quota = runtime.reserve_tensor_opaque(4096).unwrap();
    let mut workspace = runtime.allocation_stream.alloc_zeros::<u8>(4096).unwrap();
    let (pointer, guard) = workspace.device_ptr_mut(&runtime.allocation_stream);
    unsafe {
        sys::cublasSetWorkspace_v2(old_blas, pointer as *mut _, 4096)
            .result()
            .unwrap();
        sys::cublasSetPointerMode_v2(
            old_blas,
            sys::cublasPointerMode_t::CUBLAS_POINTER_MODE_DEVICE,
        )
        .result()
        .unwrap();
        sys::cublasSetMathMode(old_blas, sys::cublasMath_t::CUBLAS_PEDANTIC_MATH)
            .result()
            .unwrap();
        sys::cublasSetAtomicsMode(old_blas, sys::cublasAtomicsMode_t::CUBLAS_ATOMICS_ALLOWED)
            .result()
            .unwrap();
    }
    drop(guard);
    drop(workspace);
    runtime.allocation_stream.synchronize().unwrap();
    drop(quota); // The old external workspace is no longer live at return.
    drop(first);
    assert_eq!(cache.entries.lock().unwrap().len(), 1);
    assert!(
        runtime
            .tensor_eager
            .as_ref()
            .unwrap()
            .reserve_lane()
            .is_err(),
        "idle native lane must still consume the original quota"
    );
    let mut second = runtime.create_stream().unwrap();
    assert_ne!(second.id, old_id);
    assert_eq!(second.stream.cu_stream(), old_stream);
    assert_eq!(*second.blas.as_ref().unwrap().handle(), old_blas);
    assert_eq!(
        second.blas.as_ref().unwrap().workspace_bytes(),
        workspace_bytes
    );
    assert_eq!(Arc::as_ptr(second.tensor_lane.as_ref().unwrap()), old_lane);
    assert!(second.state.is_quiescent());
    assert!(
        second.executable_cache.preparation().is_err(),
        "old catalog must not be carried forward"
    );
    assert!(terminal_readback::available_storage(&second).is_none());
    unsafe {
        let mut pointer_mode = sys::cublasPointerMode_t::CUBLAS_POINTER_MODE_DEVICE;
        let mut math_mode = sys::cublasMath_t::CUBLAS_PEDANTIC_MATH;
        let mut atomics = sys::cublasAtomicsMode_t::CUBLAS_ATOMICS_ALLOWED;
        sys::cublasGetPointerMode_v2(old_blas, &mut pointer_mode)
            .result()
            .unwrap();
        sys::cublasGetMathMode(old_blas, &mut math_mode)
            .result()
            .unwrap();
        sys::cublasGetAtomicsMode(old_blas, &mut atomics)
            .result()
            .unwrap();
        assert_eq!(
            pointer_mode,
            sys::cublasPointerMode_t::CUBLAS_POINTER_MODE_HOST
        );
        assert_eq!(math_mode, sys::cublasMath_t::CUBLAS_DEFAULT_MATH);
        assert_eq!(
            atomics,
            sys::cublasAtomicsMode_t::CUBLAS_ATOMICS_NOT_ALLOWED
        );
    }
    gemm(&runtime, &mut second);
    assert!(
        runtime.create_stream().is_err(),
        "reuse must not increase concurrent stream capacity"
    );
    drop(second);
    let weak_owner = Arc::downgrade(runtime.tensor_eager.as_ref().unwrap());
    let weak_cache = Arc::downgrade(cache);
    drop(runtime);
    assert!(weak_cache.upgrade().is_none());
    assert!(
        weak_owner.upgrade().is_none(),
        "cached lane must not form an owner cycle"
    );
}

#[test]
#[ignore = "requires an exclusive CUDA window; real fence/readback and late runtime retirement"]
fn retired_native_lane_rejects_unretired_owners_and_late_returns() {
    let runtime = runtime(1);
    let cache = runtime.idle_native_lanes.as_ref().unwrap();
    let mut stream = runtime.create_stream().unwrap();
    runtime.ensure_submission_blas(&mut stream, true).unwrap();
    stream.state.begin_submission().unwrap();
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
        _commands: Vec::new(),
    };
    assert!(matches!(
        runtime.wait_fence(&fence).unwrap().into_parts().0,
        DeviceTerminal::Succeeded
    ));
    assert!(stream.state.is_quiescent());
    stream.return_retired_native_lane();
    assert!(
        cache.entries.lock().unwrap().is_empty(),
        "a completed but externally held fence still owns the old native pair"
    );
    drop(fence);
    let weak_blas = Arc::downgrade(stream.blas.as_ref().unwrap());
    stream.return_retired_native_lane();
    assert!(
        cache.entries.lock().unwrap().is_empty(),
        "weak native aliases must not upgrade after recycling"
    );
    drop(weak_blas);

    let quota = runtime.reserve_tensor_opaque(4).unwrap();
    let base = runtime.allocation_stream.alloc_zeros::<u8>(4).unwrap();
    let (pointer, guard) = base.device_ptr(&runtime.allocation_stream);
    drop(guard);
    runtime.allocation_stream.synchronize().unwrap();
    let region = CudaBufferRegion {
        _allocation: Arc::new(CudaAllocation {
            _base: Some(base),
            idle_allocation_return: None,
            aligned_ptr: pointer,
            requested_bytes: 4,
            _tensor_owner: runtime.tensor_eager.clone(),
        }),
        _core_retention: None,
        reusable_address_scope: None,
        runtime_instance: runtime.runtime_instance,
        device_ptr: pointer,
        length_bytes: 4,
        element_type: ElementType::U32,
    };
    let (command, packet) = terminal_readback::prepare_regions(&runtime, &mut stream, vec![region])
        .unwrap()
        .unwrap();
    drop((command, packet)); // Never enqueued; pool is CheckedOut even after all packet owners drop.
    assert_eq!(Arc::strong_count(&stream.terminal_readback_pool), 1);
    assert!(!stream.terminal_readback_pool.retired());
    drop(stream);
    assert!(cache.entries.lock().unwrap().is_empty());
    runtime.allocation_stream.synchronize().unwrap();
    drop(quota);

    let mut late = runtime.create_stream().unwrap();
    runtime.ensure_submission_blas(&mut late, true).unwrap();
    let weak_cache = Arc::downgrade(cache);
    let weak_owner = Arc::downgrade(runtime.tensor_eager.as_ref().unwrap());
    drop(runtime);
    assert!(weak_cache.upgrade().is_none());
    assert!(weak_owner.upgrade().is_some());
    drop(late);
    assert!(weak_owner.upgrade().is_none());
}
