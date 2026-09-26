//! Real CUDA cache/fence ownership, with weak references to real allocations.
//! No core completion receipt or synthetic resource authority is minted here.
use super::*;

fn region(
    runtime: &CudaDeviceRuntime,
    stream: &CudaDeviceStream,
    bytes: usize,
    scope: Option<DeviceReusableAddressScope>,
) -> CudaBufferRegion {
    let base = stream.stream.alloc_zeros::<u8>(bytes).unwrap();
    let pointer = base.device_ptr(&stream.stream).0;
    CudaBufferRegion {
        _allocation: Arc::new(CudaAllocation {
            _base: base,
            _memory_charge: None,
            aligned_ptr: pointer,
            requested_bytes: bytes as u64,
        }),
        _core_retention: None,
        reusable_address_scope: scope,
        runtime_instance: runtime.runtime_instance,
        device_ptr: pointer,
        length_bytes: bytes as u64,
        element_type: ElementType::U8,
    }
}

#[test]
#[ignore = "requires an actual CUDA device"]
fn cuda_replay_cache_retains_static_regions_but_releases_completed_fence_dependencies() {
    let runtime = CudaDeviceRuntime::new(
        crate::backend::cuda::vnext_ops::cuda_vnext_runtime_config(
            0,
            DeviceId::new("device.test.replay-fence-lifetime").unwrap(),
            ferrum_types::AttentionExecutionPolicy::Portable,
        )
        .unwrap(),
    )
    .unwrap();
    let mut stream = runtime.create_stream().unwrap();
    stream
        .executable_cache
        .configure(DeviceReusableExecutionPlan::on_demand(1).unwrap())
        .unwrap();
    let counter = region(&runtime, &stream, 4, Some(DeviceReusableAddressScope::Plan));
    let static_owner = Arc::downgrade(&counter._allocation);
    let function = runtime
        .context
        .load_module(Ptx::from_src(INCREMENT_PTX))
        .unwrap()
        .load_function("increment")
        .unwrap();
    for index in 0..3 {
        // This allocation has no reusable address scope: its ownership belongs
        // to this submission, like per-request KV/state behind an indirect ABI.
        let dependency = region(&runtime, &stream, 64, None);
        let owner = Arc::downgrade(&dependency._allocation);
        let function = function.clone();
        let command = CudaDeviceCommand::replayable_operation_with_blas_and_fence_dependencies(
            "test.replay-fence-lifetime",
            vec![counter.clone()],
            vec![dependency],
            CudaCommandReplayKeyBuilder::new("test.fence-lifetime.v1", "increment")
                .u64(counter.device_ptr())
                .finish(),
            move |stream, _, regions| {
                let pointer = regions[0].device_ptr();
                let delta = 1_u32;
                let mut launch = stream.launch_builder(&function);
                launch.arg(&pointer);
                launch.arg(&delta);
                unsafe {
                    launch.launch(LaunchConfig {
                        grid_dim: (1, 1, 1),
                        block_dim: (1, 1, 1),
                        shared_mem_bytes: 0,
                    })
                }
                .map_err(|e| CudaDeviceRuntimeError::contract(e.to_string()))?;
                Ok(())
            },
        )
        .unwrap()
        .with_work_attribution(DeviceBatchingForm::Scalar, 1, 1, 1, 0)
        .unwrap();
        let commands = vec![command];
        let candidates =
            cuda_executable_candidates(&[DeviceCommandPhase::Compute], &commands, None, &[])
                .unwrap();
        assert_eq!(
            candidates.len(),
            1,
            "fence-only ownership cannot make capture ineligible"
        );
        let prepared = stream
            .executable_cache
            .prepare_all(
                &runtime.context,
                &stream.stream,
                &stream.blas,
                &commands,
                &candidates,
                true,
            )
            .unwrap();
        stream.state.begin_submission().unwrap();
        let replayed = stream
            .executable_cache
            .launch(&stream.stream, &candidates[0], DeviceTimingMode::Off)
            .unwrap();
        if index == 0 {
            assert!(replayed.is_none());
            assert_eq!(prepared.warmup_required_segments(), 1);
            commands[0].enqueue(&stream.stream, &stream.blas).unwrap();
        } else {
            assert!(
                replayed.is_some(),
                "capture and subsequent hit must really launch the graph"
            );
            assert_eq!(
                stream
                    .executable_cache
                    .preparation()
                    .unwrap()
                    .resident_executables(),
                1
            );
            if index == 1 {
                assert_eq!(prepared.captured_segments(), 1);
            } else {
                assert_eq!(prepared.cache_hit_segments(), 1);
            }
        }
        let event = stream.stream.record_event(None).unwrap();
        stream.state.submission_recorded().unwrap();
        let fence = CudaDeviceFence {
            event,
            timing: CudaFenceTiming::NotRequested,
            command_timing: CudaFenceCommandTiming::NotRequested,
            attribution: None,
            stream_state: Arc::clone(&stream.state),
            terminal_accounted: AtomicBool::new(false),
            _stream: Arc::clone(&stream.stream),
            _blas: Arc::clone(&stream.blas),
            _commands: commands,
        };
        assert!(
            owner.upgrade().is_some(),
            "submitted dependency must outlive the caller"
        );
        runtime.wait_fence(&fence).unwrap();
        assert!(
            owner.upgrade().is_some(),
            "terminal fence still owns its command"
        );
        let bytes = stream
            .stream
            .clone_dtoh(&counter._allocation._base)
            .unwrap();
        assert_eq!(
            u32::from_le_bytes(bytes.try_into().unwrap()),
            index + 1,
            "capture must not execute twice"
        );
        drop(fence);
        assert!(
            owner.upgrade().is_none(),
            "resident graph must not retain request fence dependencies"
        );
        assert!(stream.state.is_quiescent());
    }
    drop(counter);
    assert!(
        static_owner.upgrade().is_some(),
        "resident graph must retain its static launch regions"
    );
    assert_eq!(stream.executable_cache.trim_quiescent().0, 1);
    assert!(
        static_owner.upgrade().is_none(),
        "eviction releases the last static allocation owner"
    );
}
