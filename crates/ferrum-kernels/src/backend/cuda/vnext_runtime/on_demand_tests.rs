//! Real CUDA capture/replay tests with an observable stateful operation.

use super::*;
use crate::backend::cuda::vnext_replay::{
    cuda_executable_candidates, CudaCommandReplayKeyBuilder, CudaExecutableCache,
    CudaExecutablePreparation,
};
use cudarc::driver::{sys, CudaSlice, DevicePtr, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::Ptx;
use ferrum_interfaces::vnext::{
    DeviceReusableExecutionCapture, DeviceReusableExecutionPreparationState,
    DeviceReusableExecutionProgramId, ExecutionLane, PlanHash, ReusableExecutionBucketSpec,
    ReusableExecutionCapacity, ReusableExecutionClassId,
};

const INCREMENT_PTX: &str = r#"
.version 6.0
.target sm_30
.address_size 64
.visible .entry increment(.param .u64 address, .param .u32 delta) {
    .reg .b64 %ptr;
    .reg .b32 %value, %delta;
    ld.param.u64 %ptr, [address];
    ld.param.u32 %delta, [delta];
    ld.global.u32 %value, [%ptr];
    add.u32 %value, %value, %delta;
    st.global.u32 [%ptr], %value;
    ret;
}
"#;

struct CaptureHarness {
    cache: CudaExecutableCache,
    counter: Arc<CudaSlice<u32>>,
    stream: Arc<CudaStream>,
    blas: Arc<CudaBlas>,
    context: Arc<CudaContext>,
}

impl CaptureHarness {
    fn new(capacity: usize) -> Self {
        let context = CudaContext::new(0).expect("CUDA capture tests require a configured GPU");
        let stream = context.new_stream().unwrap();
        let blas = Arc::new(CudaBlas::new(Arc::clone(&stream)).unwrap());
        let counter = Arc::new(stream.clone_htod(&[0_u32]).unwrap());
        stream.synchronize().unwrap();
        let mut cache = CudaExecutableCache::new();
        let receipt = cache
            .configure(DeviceReusableExecutionPlan::on_demand(capacity).unwrap())
            .unwrap();
        assert_eq!(
            receipt.state(),
            DeviceReusableExecutionPreparationState::Ready
        );
        assert_eq!(receipt.captured_executables(), 0);
        Self {
            cache,
            counter,
            stream,
            blas,
            context,
        }
    }

    fn command(&self, delta: u32, reject_capture: bool) -> CudaDeviceCommand {
        let function = self
            .context
            .load_module(Ptx::from_src(INCREMENT_PTX))
            .unwrap()
            .load_function("increment")
            .unwrap();
        let counter = Arc::clone(&self.counter);
        let pointer = counter.device_ptr(&self.stream).0;
        let mut command = super::tests::command("stateful_increment");
        command.replay_key = Some(
            CudaCommandReplayKeyBuilder::new("test.stateful.v1", "stateful_increment")
                .u64(pointer)
                .u32(delta)
                .boolean(reject_capture)
                .finish(),
        );
        // The fixture owns the same allocation for the cache lifetime. The
        // executable retains it so an evicted caller command cannot free it.
        command.reusable_address_scope = Some(DeviceReusableAddressScope::Plan);
        command.executable = Some(Arc::new(CudaCommandExecutable {
            regions: Vec::new(),
            host_storage: Vec::new(),
            enqueue: Mutex::new(Box::new(move |stream, _, _, _| {
                let _retained_counter = &counter;
                if reject_capture {
                    let mut status = sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE;
                    let result =
                        unsafe { sys::cuStreamIsCapturing(stream.cu_stream(), &mut status) };
                    assert_eq!(result, sys::CUresult::CUDA_SUCCESS);
                    if status != sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE {
                        return Err(CudaDeviceRuntimeError::contract("fixture rejects capture"));
                    }
                }
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
                .map_err(|error| CudaDeviceRuntimeError::contract(error.to_string()))?;
                Ok(())
            })),
        }));
        command
    }

    fn execute(
        &mut self,
        command: CudaDeviceCommand,
        quiescent: bool,
        expected: u32,
    ) -> CudaExecutablePreparation {
        let commands = [command];
        let candidates =
            cuda_executable_candidates(&[DeviceCommandPhase::Compute], &commands, None, &[])
                .unwrap();
        assert_eq!(candidates.len(), 1);
        let before = self.stream.clone_dtoh(self.counter.as_ref()).unwrap()[0];
        let report = self
            .cache
            .prepare_all(
                &self.context,
                &self.stream,
                &self.blas,
                &commands,
                &candidates,
                quiescent,
            )
            .unwrap();
        // Capture and graph upload must not execute the stateful operation.
        self.stream.synchronize().unwrap();
        assert_eq!(
            self.stream.clone_dtoh(self.counter.as_ref()).unwrap()[0],
            before
        );
        if self
            .cache
            .launch(&self.stream, &candidates[0], DeviceTimingMode::Off)
            .unwrap()
            .is_none()
        {
            commands[0].enqueue(&self.stream, &self.blas).unwrap();
        }
        self.stream.synchronize().unwrap();
        assert_eq!(
            self.stream.clone_dtoh(self.counter.as_ref()).unwrap()[0],
            expected
        );
        assert!(
            self.cache.preparation().unwrap().resident_executables()
                <= self.cache.preparation().unwrap().maximum_executables()
        );
        report
    }
}

#[test]
fn on_demand_capture_executes_state_once_and_eviction_preserves_fallback() {
    let mut h = CaptureHarness::new(1);
    let report = h.execute(h.command(1, false), true, 1);
    assert_eq!(report.warmup_required_segments(), 1);
    assert_eq!(report.captured_segments(), 0);
    let report = h.execute(h.command(1, false), true, 2);
    assert_eq!(report.captured_segments(), 1);
    assert_eq!(report.uploaded_segments(), 1);
    let report = h.execute(h.command(1, false), true, 3);
    assert_eq!(report.cache_hit_segments(), 1);
    let report = h.execute(h.command(2, false), false, 5);
    assert_eq!(report.quiescence_deferred_segments(), 1);
    assert_eq!(report.captured_segments(), 0);
    let report = h.execute(h.command(2, false), true, 7);
    assert_eq!(report.captured_segments(), 1);
    assert_eq!(report.evicted_segments(), 1);
    // Revisit an evicted shape and a fresh cache with the same live allocation.
    h.execute(h.command(1, false), true, 8);
    h.cache = CudaExecutableCache::new();
    h.cache
        .configure(DeviceReusableExecutionPlan::on_demand(1).unwrap())
        .unwrap();
    let report = h.execute(h.command(1, false), true, 9);
    assert_eq!(report.warmup_required_segments(), 1);
    assert_eq!(report.captured_segments(), 0);
}

#[test]
fn on_demand_capture_rejection_keeps_ordinary_execution_available() {
    let mut h = CaptureHarness::new(1);
    h.execute(h.command(3, true), true, 3);
    let report = h.execute(h.command(3, true), true, 6);
    assert_eq!(report.capture_rejected_segments(), 1);
    assert_eq!(report.captured_segments(), 0);
    let report = h.execute(h.command(3, true), true, 9);
    assert_eq!(report.cached_rejected_segments(), 1);
}

#[test]
fn on_demand_logical_catalog_is_bounded_and_stale_references_miss_before_launch() {
    let mut h = CaptureHarness::new(1);
    h.execute(h.command(1, false), true, 1);
    h.execute(h.command(1, false), true, 2);
    let config = crate::backend::cuda::vnext_ops::cuda_vnext_runtime_config(
        0,
        DeviceId::new("device.cuda.catalog-test").unwrap(),
        ferrum_types::AttentionExecutionPolicy::Portable,
    )
    .unwrap();
    let implementation = config.runtime_implementation_fingerprint.clone();
    let runtime = Arc::new(CudaDeviceRuntime::new(config).unwrap());
    let owner = ExecutionLane::create(runtime).unwrap();
    let lane = owner.id();
    let bucket = ReusableExecutionBucketSpec::new(
        ReusableExecutionClassId::new("fixture.decode").unwrap(),
        ReusableExecutionCapacity::new(1, 1, 1).unwrap(),
    )
    .unwrap();
    let plan_hash: PlanHash = serde_json::from_value(serde_json::json!("a".repeat(64))).unwrap();
    let mut prior_invocation = None;
    // Logical variants can share one resident physical executable. Their
    // descriptors must also be bounded, even without physical LRU eviction.
    for slot in 0..4 {
        let id = DeviceReusableExecutionProgramId::new(
            plan_hash.clone(),
            implementation.clone(),
            lane,
            bucket.bucket_id().clone(),
            "c".repeat(64),
            "d".repeat(64),
            slot,
            1,
            1,
            1,
        )
        .unwrap();
        let capture =
            DeviceReusableExecutionCapture::new(id.clone(), 1, Vec::new(), Vec::new()).unwrap();
        let commands = [h.command(1, false)];
        let phases = [DeviceCommandPhase::Compute];
        let nodes = [Some(0)];
        let candidates = cuda_executable_candidates(&phases, &commands, Some(&nodes), &[]).unwrap();
        let preparation = h
            .cache
            .prepare_all(&h.context, &h.stream, &h.blas, &commands, &candidates, true)
            .unwrap();
        h.cache
            .register_program(
                &capture,
                &candidates,
                &phases,
                &nodes,
                &commands,
                &preparation,
            )
            .unwrap();
        let catalog = h.cache.catalog().unwrap();
        assert_eq!(catalog.len(), 1);
        assert_eq!(catalog[0].program_id(), &id);
        let invocation =
            DeviceReusableExecutionInvocation::new(id, catalog[0].segments()[0].clone(), 1, 1)
                .unwrap();
        assert!(h.cache.contains_program_segment(&invocation).unwrap());
        if let Some(prior) = prior_invocation {
            assert!(!h.cache.contains_program_segment(&prior).unwrap());
        }
        prior_invocation = Some(invocation);
    }
    assert_eq!(h.stream.clone_dtoh(h.counter.as_ref()).unwrap(), vec![2]);
}
