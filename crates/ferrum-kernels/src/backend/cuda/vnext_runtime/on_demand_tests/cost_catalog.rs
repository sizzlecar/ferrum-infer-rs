//! Real capture/upload/eviction through the production cache. Numeric snapshots
//! may outlive a graph; they must neither authorize it nor retain allocations.
use super::*;
use ferrum_interfaces::execution_cost::{
    KernelNumericWorkV1, KernelReplayGeometryV1, SelectedAlgorithmClassV1,
    SelectedCommandCostBuilderV1, SelectedCommandCostEvidenceV1,
};
use ferrum_interfaces::vnext::{DeviceCostGraphCatalog, DeviceCostGraphCatalogLimits};
use sha2::{Digest, Sha256};

fn evidence(delta: u32) -> SelectedCommandCostEvidenceV1 {
    let mut builder = SelectedCommandCostBuilderV1::new_with_algorithm_work(1);
    builder
        .kernel_with_replay_geometry(
            SelectedAlgorithmClassV1::new(
                "increment",
                1,
                Sha256::digest(INCREMENT_PTX.as_bytes()).into(),
                Sha256::digest(b"test.increment.u32.scalar.v1").into(),
            )
            .unwrap(),
            KernelNumericWorkV1 {
                logical_units: 1,
                padded_units: 1,
                inner_units_per_logical_unit: 1,
                grid: [1, 1, 1],
                scratch_bytes: 0,
                staged_weight_bytes: 0,
            },
            KernelReplayGeometryV1 {
                block: [1, 1, 1],
                dynamic_shared_bytes: 0,
                fixed_parameters: &[u64::from(delta)],
            },
        )
        .unwrap();
    builder.finish().unwrap()
}

fn command(h: &CaptureHarness, delta: u32, reject: bool) -> CudaDeviceCommand {
    h.command(delta, reject)
        .with_statistical_evidence(Some(evidence(delta)))
}

fn snapshot(h: &CaptureHarness) -> Arc<DeviceCostGraphCatalog> {
    h.cache
        .cost_reusable_graph_catalog_shared(
            DeviceCostGraphCatalogLimits::new(4, 4, 4).unwrap(),
            &mut || Ok(()),
        )
        .unwrap()
}

fn register(h: &mut CaptureHarness, capture: &DeviceReusableExecutionCapture, delta: u32) {
    let commands = [command(h, delta, false)];
    let phases = [DeviceCommandPhase::Compute];
    let nodes = [Some(0)];
    let candidates = cuda_executable_candidates(&phases, &commands, Some(&nodes), &[]).unwrap();
    let prepared = h
        .cache
        .prepare_all(&h.context, &h.stream, &h.blas, &commands, &candidates, true)
        .unwrap();
    assert_eq!(prepared.cache_hit_segments(), 1);
    h.cache
        .register_program(capture, &candidates, &phases, &nodes, &commands, &prepared)
        .unwrap();
}

#[test]
#[ignore = "requires an actual CUDA device"]
fn cuda_shared_cost_catalog_tracks_real_capture_and_releases_gpu_ownership() {
    let mut h = CaptureHarness::new(1);
    let config = crate::backend::cuda::vnext_ops::cuda_vnext_runtime_config(
        0,
        DeviceId::new("device.cuda.shared-cost-catalog-test").unwrap(),
        ferrum_types::AttentionExecutionPolicy::Portable,
    )
    .unwrap();
    let implementation = config.runtime_implementation_fingerprint.clone();
    let owner = ExecutionLane::create(Arc::new(CudaDeviceRuntime::new(config).unwrap())).unwrap();
    let bucket = ReusableExecutionBucketSpec::new(
        ReusableExecutionClassId::new("fixture.decode").unwrap(),
        ReusableExecutionCapacity::new(1, 1, 1).unwrap(),
    )
    .unwrap();
    let capture = |slot| {
        DeviceReusableExecutionCapture::new(
            DeviceReusableExecutionProgramId::new(
                serde_json::from_value(serde_json::json!("a".repeat(64))).unwrap(),
                implementation.clone(),
                owner.id(),
                bucket.bucket_id().clone(),
                "c".repeat(64),
                "d".repeat(64),
                slot,
                1,
                1,
                1,
            )
            .unwrap(),
            1,
            Vec::new(),
            Vec::new(),
        )
        .unwrap()
    };
    let first_capture = capture(0);
    let empty = snapshot(&h);
    assert!(Arc::ptr_eq(&empty, &snapshot(&h)));
    assert_eq!(
        h.execute(command(&h, 1, false), true, 1)
            .warmup_required_segments(),
        1
    );
    let prepared = h.execute(command(&h, 1, false), true, 2);
    assert_eq!(prepared.captured_segments(), 1);
    assert_eq!(prepared.uploaded_segments(), 1);
    register(&mut h, &first_capture, 1);
    let first = snapshot(&h);
    assert!(!Arc::ptr_eq(&empty, &first));
    let uploaded = &first.programs()[0].uploaded_segments()[0];
    assert_eq!(uploaded.logical_commands().len(), 1);
    let logical = &uploaded.logical_commands()[0];
    assert!(logical.statistical_evidence().is_none());
    assert!(logical
        .bind_current_cost_evidence(Some(&evidence(1)))
        .is_some());
    assert!(logical
        .bind_current_cost_evidence(Some(&evidence(2)))
        .is_none());
    let invocation = DeviceReusableExecutionInvocation::new(
        first_capture.program_id().clone(),
        uploaded.segment().clone(),
        1,
        1,
    )
    .unwrap();
    assert!(h.cache.contains_program_segment(&invocation).unwrap());
    register(&mut h, &first_capture, 1);
    assert!(
        Arc::ptr_eq(&first, &snapshot(&h)),
        "identical sealed registration is a no-op"
    );
    assert_eq!(
        h.execute(command(&h, 1, false), true, 3)
            .cache_hit_segments(),
        1
    );
    assert!(
        Arc::ptr_eq(&first, &snapshot(&h)),
        "LRU alone is not numeric metadata"
    );

    h.execute(command(&h, 2, false), true, 5);
    let prepared = h.execute(command(&h, 2, false), true, 7);
    assert_eq!(prepared.evicted_segments(), 1);
    assert_eq!(prepared.uploaded_segments(), 1);
    // Eviction retains an explicit gap in this program, so the old ordinal
    // is rejected as a stale reference rather than an absent-program miss.
    let stale = h
        .cache
        .contains_program_segment(&invocation)
        .expect_err("an evicted segment cannot authorize replay");
    assert!(stale.eager_fallback_safe());
    let evicted = snapshot(&h);
    assert!(!Arc::ptr_eq(&first, &evicted));
    assert!(evicted
        .programs()
        .iter()
        .all(|p| p.uploaded_segments().is_empty()));
    let old_program = evicted
        .programs()
        .iter()
        .find(|p| p.program().program_id() == first_capture.program_id())
        .expect("eviction must retain its typed program disposition");
    assert!(old_program.program().segments().is_empty());
    assert!(old_program.program().gaps().iter().any(|gap| {
        gap.node_index() == 0
            && gap.reason()
                == ferrum_interfaces::vnext::DeviceReusableExecutionProgramGapReason::Evicted
    }));
    let second_capture = capture(1);
    register(&mut h, &second_capture, 2);
    let second = snapshot(&h);
    assert_eq!(
        second.programs()[0].program().program_id(),
        second_capture.program_id()
    );
    assert_eq!(
        first.programs()[0].uploaded_segments().len(),
        1,
        "old values remain immutable"
    );

    h.execute(command(&h, 3, true), true, 10);
    let prepared = h.execute(command(&h, 3, true), true, 13);
    assert_eq!(prepared.capture_rejected_segments(), 1);
    let rejected = snapshot(&h);
    assert!(!Arc::ptr_eq(&second, &rejected));
    assert_ne!(second.stream_state(), rejected.stream_state());
    assert!(rejected
        .programs()
        .iter()
        .all(|p| p.uploaded_segments().is_empty()));
    // Re-establish a resident graph before testing its real allocation lifetime.
    h.execute(command(&h, 2, false), true, 15);
    h.execute(command(&h, 2, false), true, 17);
    register(&mut h, &second_capture, 2);
    let final_catalog = snapshot(&h);
    assert_eq!(final_catalog.programs()[0].uploaded_segments().len(), 1);
    h.stream.synchronize().unwrap();
    let CaptureHarness {
        mut cache,
        counter,
        stream,
        blas,
        context,
    } = h;
    let allocation = Arc::downgrade(&counter);
    drop(counter);
    assert!(
        allocation.upgrade().is_some(),
        "the resident graph owns its launch allocation"
    );
    assert_eq!(cache.trim_quiescent().0, 1);
    assert!(
        allocation.upgrade().is_none(),
        "numeric Arc snapshots must not own the GPU allocation"
    );
    let trimmed = cache
        .cost_reusable_graph_catalog_shared(
            DeviceCostGraphCatalogLimits::new(4, 4, 4).unwrap(),
            &mut || Ok(()),
        )
        .unwrap();
    assert!(trimmed.programs().is_empty());
    assert!(!Arc::ptr_eq(&final_catalog, &trimmed));
    assert_eq!(first.programs()[0].uploaded_segments().len(), 1);
    assert_eq!(second.programs()[0].uploaded_segments().len(), 1);
    assert_eq!(final_catalog.programs()[0].uploaded_segments().len(), 1);
    drop((cache, stream, blas, context));
}
