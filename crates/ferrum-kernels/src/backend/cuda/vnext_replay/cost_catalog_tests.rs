//! CPU-only registry tests; no CUDA context, handles or allocations are created.
#[path = "cost_catalog_tests/index_prototype.rs"]
mod index_prototype;

use super::*;
use crate::backend::cpu::vnext_ops::CpuVNextComposition;
use ferrum_interfaces::vnext::{
    DeviceCostGraphCatalog, DeviceCostGraphCatalogLimits, DeviceCostGraphConfiguration,
    DeviceCostGraphStreamState, DeviceId, ExecutionLane, ReusableExecutionBucketSpec,
    ReusableExecutionCapacity, ReusableExecutionClassId, VNextError,
};

fn limits() -> DeviceCostGraphCatalogLimits {
    DeviceCostGraphCatalogLimits::new(4, 8, 8).unwrap()
}
fn snapshot(cache: &CudaExecutableCache) -> Arc<DeviceCostGraphCatalog> {
    cache
        .cost_reusable_graph_catalog_shared(limits(), &mut || Ok(()))
        .unwrap()
}
fn capture() -> DeviceReusableExecutionCapture {
    // Obtain the opaque identity through a real CPU runtime-owned lane. The
    // registry exercise needs numeric metadata, never a CUDA context or handle.
    let composition = CpuVNextComposition::create(
        DeviceId::new("device.cpu.cost-catalog-registry").unwrap(),
        1024 * 1024,
    )
    .unwrap();
    let lane = ExecutionLane::create(Arc::clone(composition.runtime())).unwrap();
    let bucket = ReusableExecutionBucketSpec::new(
        ReusableExecutionClassId::new("cost-catalog-registry").unwrap(),
        ReusableExecutionCapacity::new(1, 1, 1).unwrap(),
    )
    .unwrap();
    let id = DeviceReusableExecutionProgramId::new(
        serde_json::from_value(serde_json::json!("a".repeat(64))).unwrap(),
        "b".repeat(64),
        lane.id(),
        bucket.bucket_id().clone(),
        "c".repeat(64),
        "d".repeat(64),
        1,
        1,
        1,
        1,
    )
    .unwrap();
    DeviceReusableExecutionCapture::new(id, 2, vec![], vec![]).unwrap()
}
fn register_missing_program(
    cache: &mut CudaExecutableCache,
    capture: &DeviceReusableExecutionCapture,
) {
    cache
        .register_program(
            capture,
            &[],
            &[],
            &[],
            &[],
            &CudaExecutablePreparation::default(),
        )
        .unwrap();
}

#[test]
fn cuda_cost_catalog_shared_hit_preserves_limits_poll_and_original_owned_api() {
    let mut cache = CudaExecutableCache::new();
    cache
        .configure(DeviceReusableExecutionPlan::new(4).unwrap())
        .unwrap();
    let declaration = capture();
    register_missing_program(&mut cache, &declaration);
    assert!(cache.cost_catalog.get().is_none());
    let original_descriptor = cache.programs[declaration.program_id()].descriptor.clone();
    register_missing_program(&mut cache, &declaration);
    assert!(cache.cost_catalog.get().is_none());
    assert_eq!(
        cache.programs[declaration.program_id()].descriptor,
        original_descriptor,
        "registration without a catalog preserves the typed program"
    );
    // Absence of a catalog never bypasses the original monotonic topology gate.
    let changed_topology =
        DeviceReusableExecutionCapture::new(declaration.program_id().clone(), 3, vec![], vec![])
            .unwrap();
    let error = cache
        .register_program(
            &changed_topology,
            &[],
            &[],
            &[],
            &[],
            &CudaExecutablePreparation::default(),
        )
        .unwrap_err();
    assert_eq!(error.stage, "update reusable execution program");
    assert_eq!(
        error.detail,
        "one typed program id changed its logical topology"
    );
    assert!(cache.cost_catalog.get().is_none());
    assert_eq!(
        cache.programs[declaration.program_id()].descriptor,
        original_descriptor
    );
    let first = snapshot(&cache);
    let mut polls = 0;
    let hit = cache
        .cost_reusable_graph_catalog_shared(limits(), &mut || {
            polls += 1;
            Ok(())
        })
        .unwrap();
    assert!(Arc::ptr_eq(&first, &hit));
    assert_eq!(
        polls, 1,
        "a hit does not recopy individual descriptors/rows"
    );
    assert_eq!(
        *first,
        cache
            .cost_reusable_graph_catalog(limits(), &mut || Ok(()))
            .unwrap()
    );
    assert!(cache
        .cost_reusable_graph_catalog_shared(
            DeviceCostGraphCatalogLimits::new(4, 1, 8).unwrap(),
            &mut || Ok(())
        )
        .is_err());
    assert!(cache
        .cost_reusable_graph_catalog_shared(limits(), &mut || Err(
            VNextError::InvalidExecutionPlan {
                reason: "test deadline".into()
            }
        ))
        .is_err());
    assert!(
        Arc::ptr_eq(&first, &snapshot(&cache)),
        "read failure cannot corrupt a valid catalog"
    );
}

#[test]
fn cuda_cost_catalog_failed_build_is_not_published_or_remembered() {
    let mut cache = CudaExecutableCache::new();
    cache
        .configure(DeviceReusableExecutionPlan::new(4).unwrap())
        .unwrap();
    register_missing_program(&mut cache, &capture());
    let mut polls = 0;
    assert!(cache
        .cost_reusable_graph_catalog_shared(limits(), &mut || {
            polls += 1;
            if polls == 4 {
                Err(VNextError::InvalidExecutionPlan {
                    reason: "interrupted inside builder".into(),
                })
            } else {
                Ok(())
            }
        })
        .is_err());
    assert!(cache.cost_catalog.get().is_none());
    assert!(cache
        .cost_reusable_graph_catalog_shared(
            DeviceCostGraphCatalogLimits::new(4, 1, 8).unwrap(),
            &mut || Ok(())
        )
        .is_err());
    assert!(cache.cost_catalog.get().is_none());
    assert_eq!(snapshot(&cache).programs().len(), 1);
}

#[test]
fn cuda_cost_catalog_configuration_registration_rejection_and_trim_invalidate() {
    let mut cache = CudaExecutableCache::new();
    let empty = snapshot(&cache);
    cache
        .configure(DeviceReusableExecutionPlan::new(4).unwrap())
        .unwrap();
    let preparing = snapshot(&cache);
    assert!(!Arc::ptr_eq(&empty, &preparing));
    assert_eq!(
        preparing.stream_state().configuration(),
        DeviceCostGraphConfiguration::StartupPreparing
    );
    let declaration = capture();
    register_missing_program(&mut cache, &declaration);
    let registered = snapshot(&cache);
    assert!(!Arc::ptr_eq(&preparing, &registered));
    assert_eq!(registered.programs().len(), 1);
    register_missing_program(&mut cache, &declaration);
    assert!(
        Arc::ptr_eq(&registered, &snapshot(&cache)),
        "identical registration is not a catalog mutation"
    );
    let changed_topology =
        DeviceReusableExecutionCapture::new(declaration.program_id().clone(), 3, vec![], vec![])
            .unwrap();
    let error = cache
        .register_program(
            &changed_topology,
            &[],
            &[],
            &[],
            &[],
            &CudaExecutablePreparation::default(),
        )
        .unwrap_err();
    assert_eq!(error.stage, "update reusable execution program");
    assert_eq!(
        error.detail,
        "one typed program id changed its logical topology"
    );
    assert!(Arc::ptr_eq(&registered, &snapshot(&cache)));
    cache.remember_rejected(CudaExecutableSegmentKey([1; 32]), 1);
    let rejected = snapshot(&cache);
    assert!(!Arc::ptr_eq(&registered, &rejected));
    assert_eq!(
        rejected.stream_state(),
        DeviceCostGraphStreamState::new(DeviceCostGraphConfiguration::StartupPreparing, 0, 1, 1)
            .unwrap()
    );
    cache.seal().unwrap();
    let sealed = snapshot(&cache);
    assert!(!Arc::ptr_eq(&rejected, &sealed));
    assert_eq!(
        sealed.stream_state().configuration(),
        DeviceCostGraphConfiguration::StartupReady
    );
    cache.trim_quiescent();
    let trimmed = snapshot(&cache);
    assert!(!Arc::ptr_eq(&sealed, &trimmed));
    assert!(trimmed.programs().is_empty());
    assert_eq!(
        registered.programs().len(),
        1,
        "old snapshots are immutable numeric values"
    );
    cache.leak_if_in_flight();
    assert!(cache.cost_catalog.get().is_none());
}
