//! Real CUDA capture/replay tests with an observable stateful operation.

use super::*;
use crate::backend::cuda::vnext_replay::{
    cuda_executable_candidates, CudaCommandReplayKeyBuilder, CudaExecutableCache,
    CudaExecutableLaunch, CudaExecutablePreparation,
};
use cudarc::driver::{sys, CudaSlice, DevicePtr, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::Ptx;
use ferrum_runtime::{
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
        command.executable = Some(Arc::new(CudaCommandExecutable::static_work(
            Vec::new(),
            Vec::new(),
            Box::new(move |stream, _, _, _| {
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
            }),
        )));
        command
    }

    fn execute(
        &mut self,
        command: CudaDeviceCommand,
        quiescent: bool,
        expected: u32,
    ) -> CudaExecutablePreparation {
        self.execute_with_facts(command, quiescent, expected, ExecutionFactsCapture::Off)
            .0
    }

    fn execute_with_facts(
        &mut self,
        command: CudaDeviceCommand,
        quiescent: bool,
        expected: u32,
        capture: ExecutionFactsCapture,
    ) -> (CudaExecutablePreparation, Option<CudaExecutableLaunch>) {
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
                capture,
            )
            .unwrap();
        // Capture and graph upload must not execute the stateful operation.
        self.stream.synchronize().unwrap();
        assert_eq!(
            self.stream.clone_dtoh(self.counter.as_ref()).unwrap()[0],
            before
        );
        let launched = self
            .cache
            .launch(
                &self.stream,
                &candidates[0],
                DeviceTimingMode::Off,
                capture,
                &DisabledDeviceSubmissionTimingSink,
            )
            .unwrap();
        if launched.is_none() {
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
        (report, launched)
    }
}

#[test]
fn execution_facts_off_capture_is_not_backfilled_when_replay_observation_turns_on() {
    use ferrum_runtime::{DeviceExecutionFactsUnavailableReason, ProjectionCommandCoverage};
    let mut h = CaptureHarness::new(1);
    h.execute(h.command(1, false), true, 1);
    let captured = h.execute(h.command(1, false), true, 2);
    assert_eq!(captured.captured_segments(), 1);
    let before = h.cache.preparation().unwrap();
    let (report, launch) =
        h.execute_with_facts(h.command(1, false), true, 3, ExecutionFactsCapture::On);
    assert_eq!(report.captured_segments(), 0);
    assert_eq!(report.uploaded_segments(), 0);
    assert_eq!(report.cache_hit_segments(), 1);
    let launch = launch.expect("the original executable must still replay");
    assert!(launch.execution_facts().is_none());
    let entries = vec![(DeviceCommandPhase::Compute, Some(7), h.command(1, false))];
    let mut facts = execution_facts::CudaSubmissionComputeFacts::new(&entries, 1);
    facts.adaptive_replay(0, 1, &launch);
    let facts = facts.finish();
    assert_eq!(facts.batch_command_count(), 1);
    let [record] = facts.compute_commands() else {
        panic!("one actual compute command")
    };
    assert_eq!(record.node_index, Some(7));
    assert_eq!(record.path, DeviceExecutionPath::Replayed);
    assert!(matches!(
        record.coverage,
        ProjectionCommandCoverage::Unavailable(
            DeviceExecutionFactsUnavailableReason::CachedTemplateUnavailable
        )
    ));
    let after = h.cache.preparation().unwrap();
    assert_eq!(after.resident_executables(), before.resident_executables());
    assert_eq!(after.maximum_executables(), before.maximum_executables());
}

#[test]
fn execution_facts_on_capture_preserves_unsupported_slots_without_changing_replay() {
    use ferrum_runtime::{DeviceExecutionFactsUnavailableReason, ProjectionCommandCoverage};
    let mut h = CaptureHarness::new(1);
    let (warmup, eager) =
        h.execute_with_facts(h.command(1, false), true, 1, ExecutionFactsCapture::On);
    assert_eq!(warmup.warmup_required_segments(), 1);
    assert!(eager.is_none());
    let entries = vec![(DeviceCommandPhase::Compute, Some(2), h.command(1, false))];
    let mut facts = execution_facts::CudaSubmissionComputeFacts::new(&entries, 1);
    facts.eager(0, &entries[0].2);
    let eager = facts.finish();
    assert_eq!(eager.compute_commands()[0].path, DeviceExecutionPath::Eager);
    assert!(matches!(
        eager.compute_commands()[0].coverage,
        ProjectionCommandCoverage::Unavailable(
            DeviceExecutionFactsUnavailableReason::UnsupportedProducer
        )
    ));
    let (report, launch) =
        h.execute_with_facts(h.command(1, false), true, 2, ExecutionFactsCapture::On);
    assert_eq!(report.captured_segments(), 1);
    let original = launch.unwrap().execution_facts().unwrap().clone();
    assert_eq!(original.len(), 1);
    assert!(
        original[0].is_none(),
        "the counter is not a projection producer"
    );
    let (_, off) = h.execute_with_facts(h.command(1, false), true, 3, ExecutionFactsCapture::Off);
    assert!(off.unwrap().execution_facts().is_none());
    let (report, launch) =
        h.execute_with_facts(h.command(1, false), true, 4, ExecutionFactsCapture::On);
    assert_eq!(report.captured_segments(), 0);
    let launch = launch.unwrap();
    assert!(Arc::ptr_eq(&original, launch.execution_facts().unwrap()));
    let mut facts = execution_facts::CudaSubmissionComputeFacts::new(&entries, 1);
    facts.adaptive_replay(0, 1, &launch);
    assert!(matches!(
        facts.finish().compute_commands()[0].coverage,
        ProjectionCommandCoverage::Unavailable(
            DeviceExecutionFactsUnavailableReason::UnsupportedProducer
        )
    ));
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
            .prepare_all(
                &h.context,
                &h.stream,
                &h.blas,
                &commands,
                &candidates,
                true,
                ExecutionFactsCapture::Off,
            )
            .unwrap();
        h.cache
            .register_program(
                DeviceComputePathRequirement::Adaptive,
                &capture,
                &candidates,
                &phases,
                &nodes,
                &commands,
                &preparation,
                &ferrum_runtime::DisabledDeviceSubmissionTimingSink,
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

#[test]
fn on_demand_forced_eager_preserves_resident_program_and_recipe_for_replay() {
    let mut h = CaptureHarness::new(1);
    h.execute(h.command(1, false), true, 1);
    h.execute(h.command(1, false), true, 2);
    let config = crate::backend::cuda::vnext_ops::cuda_vnext_runtime_config(
        0,
        DeviceId::new("device.cuda.eager-catalog-test").unwrap(),
        ferrum_types::AttentionExecutionPolicy::Portable,
    )
    .unwrap();
    let implementation = config.runtime_implementation_fingerprint.clone();
    let runtime = Arc::new(CudaDeviceRuntime::new(config).unwrap());
    let owner = ExecutionLane::create(runtime).unwrap();
    let bucket = ReusableExecutionBucketSpec::new(
        ReusableExecutionClassId::new("fixture.decode").unwrap(),
        ReusableExecutionCapacity::new(1, 1, 1).unwrap(),
    )
    .unwrap();
    let id = DeviceReusableExecutionProgramId::new(
        serde_json::from_value(serde_json::json!("a".repeat(64))).unwrap(),
        implementation,
        owner.id(),
        bucket.bucket_id().clone(),
        "c".repeat(64),
        "d".repeat(64),
        0,
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
    let sink = ferrum_runtime::DisabledDeviceSubmissionTimingSink;
    let candidates = cuda_executable_candidates(&phases, &commands, Some(&nodes), &[]).unwrap();
    let preparation = h
        .cache
        .prepare_all(
            &h.context,
            &h.stream,
            &h.blas,
            &commands,
            &candidates,
            true,
            ExecutionFactsCapture::Off,
        )
        .unwrap();
    h.cache
        .register_program(
            DeviceComputePathRequirement::Adaptive,
            &capture,
            &candidates,
            &phases,
            &nodes,
            &commands,
            &preparation,
            &sink,
        )
        .unwrap();
    let catalog = h.cache.catalog().unwrap();
    assert_eq!(catalog.len(), 1);
    assert!(catalog[0].is_determinism_ready());
    let recipe_owner = h.cache.recipe_program_owner(&id).unwrap();
    let residency = h.cache.preparation().unwrap();
    let invocation =
        DeviceReusableExecutionInvocation::new(id.clone(), catalog[0].segments()[0].clone(), 1, 1)
            .unwrap();
    assert!(h
        .cache
        .launch_program_segment(
            &h.stream,
            &invocation,
            DeviceTimingMode::Off,
            false,
            ExecutionFactsCapture::Off,
            &DisabledDeviceSubmissionTimingSink
        )
        .unwrap()
        .is_some());
    h.stream.synchronize().unwrap();
    assert_eq!(h.stream.clone_dtoh(h.counter.as_ref()).unwrap(), vec![3]);

    // Match submit_with_timing's forced-eager path: fresh caller commands and
    // capture metadata still exist, but there is no candidate discovery.
    let eager_commands = [h.command(1, false)];
    let eager_preparation = h
        .cache
        .prepare_all(
            &h.context,
            &h.stream,
            &h.blas,
            &eager_commands,
            &[],
            true,
            ExecutionFactsCapture::Off,
        )
        .unwrap();
    assert_eq!(eager_preparation, CudaExecutablePreparation::default());
    h.cache
        .register_program(
            DeviceComputePathRequirement::EagerOnly,
            &capture,
            &[],
            &phases,
            &nodes,
            &eager_commands,
            &eager_preparation,
            &sink,
        )
        .unwrap();
    eager_commands[0].enqueue(&h.stream, &h.blas).unwrap();
    h.stream.synchronize().unwrap();
    assert_eq!(h.stream.clone_dtoh(h.counter.as_ref()).unwrap(), vec![4]);
    assert_eq!(h.cache.catalog().unwrap(), catalog);
    assert!(recipe_owner.same_entry(&h.cache.recipe_program_owner(&id).unwrap()));
    let after = h.cache.preparation().unwrap();
    assert_eq!(
        after.resident_executables(),
        residency.resident_executables()
    );
    assert_eq!(after.maximum_executables(), residency.maximum_executables());

    // An actual Adaptive update remains subject to the monotonic seal. Empty
    // candidates must not silently delete this resident program in that path.
    assert!(h
        .cache
        .register_program(
            DeviceComputePathRequirement::Adaptive,
            &capture,
            &[],
            &phases,
            &nodes,
            &eager_commands,
            &eager_preparation,
            &sink,
        )
        .is_err());
    assert_eq!(h.cache.catalog().unwrap(), catalog);
    assert!(recipe_owner.same_entry(&h.cache.recipe_program_owner(&id).unwrap()));
    assert!(h
        .cache
        .launch_program_segment(
            &h.stream,
            &invocation,
            DeviceTimingMode::Off,
            false,
            ExecutionFactsCapture::Off,
            &DisabledDeviceSubmissionTimingSink
        )
        .unwrap()
        .is_some());
    h.stream.synchronize().unwrap();
    assert_eq!(h.stream.clone_dtoh(h.counter.as_ref()).unwrap(), vec![5]);

    // Observation may turn on after the original graph was captured Off. The
    // same sealed program and recipe still execute; no fresh command metadata
    // may pretend to describe that old graph.
    let observed = h
        .cache
        .launch_program_segment(
            &h.stream,
            &invocation,
            DeviceTimingMode::Off,
            false,
            ExecutionFactsCapture::On,
            &DisabledDeviceSubmissionTimingSink,
        )
        .unwrap()
        .unwrap();
    h.stream.synchronize().unwrap();
    assert_eq!(h.stream.clone_dtoh(h.counter.as_ref()).unwrap(), vec![6]);
    assert!(observed.execution_facts().is_none());
    let entries = vec![(
        DeviceCommandPhase::Compute,
        None,
        CudaDeviceCommand::reusable_execution(1, invocation.clone()),
    )];
    let mut facts = execution_facts::CudaSubmissionComputeFacts::new(&entries, 1);
    facts.resident_replay(0, invocation.segment(), &observed);
    let facts = facts.finish();
    let [record] = facts.compute_commands() else {
        panic!("one captured compute command")
    };
    assert_eq!(record.submission_command_index, 0);
    assert_eq!(record.replay_command_ordinal, Some(0));
    assert_eq!(
        record.node_index,
        Some(invocation.segment().start_node_index())
    );
    assert_eq!(record.path, DeviceExecutionPath::Replayed);
    assert!(matches!(
        record.coverage,
        ferrum_runtime::ProjectionCommandCoverage::Unavailable(
            ferrum_runtime::DeviceExecutionFactsUnavailableReason::CachedTemplateUnavailable
        )
    ));
    assert_eq!(h.cache.catalog().unwrap(), catalog);
    assert!(recipe_owner.same_entry(&h.cache.recipe_program_owner(&id).unwrap()));

    // Keeping the recipe token alive cannot authorize an explicitly evicted
    // graph. The original invocation must miss before issuing another launch.
    h.cache.trim_quiescent();
    assert!(h.cache.recipe_program_owner(&id).is_none());
    assert!(!h.cache.contains_program_segment(&invocation).unwrap());
    assert!(h
        .cache
        .launch_program_segment(
            &h.stream,
            &invocation,
            DeviceTimingMode::Off,
            false,
            ExecutionFactsCapture::Off,
            &DisabledDeviceSubmissionTimingSink
        )
        .unwrap()
        .is_none());
    assert_eq!(h.stream.clone_dtoh(h.counter.as_ref()).unwrap(), vec![6]);
}
