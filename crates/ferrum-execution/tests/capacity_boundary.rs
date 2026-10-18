//! Runtime-owned capacity evidence must survive the product deferral boundary.

mod capacity_boundary {
    pub(super) mod fixture;
}
use capacity_boundary::fixture as f;
use ferrum_execution::model_executor::{
    ExecutorAdmissionEpochs, ExecutorExecutionCapacityDeferral,
    ExecutorExecutionCapacityEvidenceOwner, ExecutorExecutionCapacityStage,
    ExecutorExecutionMaintenanceProgress,
};
use ferrum_runtime::*;

#[test]
fn reusable_backing_deferral_preserves_owner_and_release_wait() {
    let catalog = f::catalog();
    let (plan, bucket) = f::plan(&catalog, true, 3);
    let (competing_plan, _) = f::plan(&catalog, false, 128);
    let runtime = f::runtime(&plan);
    let lane = ExecutionLane::create(runtime.clone()).unwrap();
    let root = f::provision(&plan, runtime.clone(), "reusable");
    let first = f::sequence(&root, "reusable.first");
    let second = f::sequence(&root, "reusable.second");
    let first_session = first.open_session().unwrap();
    let second_session = second.open_session().unwrap();
    let first_batch = ExecutionBatchParticipants::new(vec![first_session.clone()]).unwrap();
    let second_batch = ExecutionBatchParticipants::new(vec![second_session.clone()]).unwrap();
    let first_step = f::step(&first_batch, &lane, bucket.as_ref());
    let first_wave = f::wave(&first_step);
    let second_step = f::step(&second_batch, &lane, bucket.as_ref());
    let StepSubmissionWaveAdmissionDecision::BackingDeferred(deferred) = f::try_wave(&second_step)
    else {
        panic!("one occupied reusable slot must defer the second wave")
    };
    let product = ExecutorExecutionCapacityDeferral::from_backing(
        deferred.evidence(),
        ExecutorExecutionCapacityStage::SubmissionWave,
    )
    .unwrap();
    assert_eq!(
        product.stage(),
        ExecutorExecutionCapacityStage::SubmissionWave
    );
    assert_eq!(
        product.evidence().owner(),
        ExecutorExecutionCapacityEvidenceOwner::Backing
    );
    assert!(product.shortfalls().is_empty());
    assert_eq!(product.backing_blockers(), deferred.evidence().blockers());
    assert_eq!(
        product.wait_condition(),
        deferred.evidence().wait_condition()
    );
    assert!(product.maintenance_retry().is_none());

    // A busy reusable slot alone permits growth under this public plan. Keep
    // another legitimate plan's device allocation live so that maintenance
    // really must wait; releasing the original slot must then suffice even
    // while the competing plan continues to hold the remaining capacity.
    let competing = f::provision_competing(&competing_plan, runtime);
    let maintenance = deferred.maintain().unwrap();
    let DynamicDeferredMaintenanceOutcome::WaitForRelease {
        current_epochs,
        wait_condition,
        pressure,
        maintenance_boundary,
    } = maintenance
    else {
        panic!("busy reusable capacity must wait without growth space: {maintenance:?}")
    };
    assert!(pressure.device_capacity().is_some());
    assert!(maintenance_boundary.is_some());
    let product = ExecutorExecutionCapacityDeferral::from_backing_maintenance(
        deferred.evidence(),
        ExecutorAdmissionEpochs::from_capacity(current_epochs),
        wait_condition.clone(),
        pressure.clone(),
        maintenance_boundary.clone(),
        ExecutorExecutionCapacityStage::SubmissionWave,
    )
    .unwrap();
    assert_eq!(product.backing_pressure(), Some(&pressure));
    assert_eq!(
        product.maintenance_boundary(),
        maintenance_boundary.as_ref()
    );
    assert_eq!(product.backing_blockers(), deferred.evidence().blockers());
    let serialized = serde_json::to_value(product.evidence()).unwrap();
    assert_eq!(serialized["owner"], "backing");
    assert_eq!(serialized["kind"], "backing_deferred");
    assert!(serialized["pressure"].is_object());
    let waiter = root
        .register_capacity_waiter(product.wait_condition())
        .unwrap();
    assert!(!waiter.recheck().unwrap().should_retry());
    drop(first_wave);
    assert!(waiter.recheck().unwrap().should_retry());
    drop(waiter);
    drop(deferred);
    let resident_before = root
        .dynamic_pool_status()
        .unwrap()
        .pools()
        .iter()
        .map(|pool| (pool.pool_id().clone(), pool.resident_bytes()))
        .collect::<Vec<_>>();
    let StepSubmissionWaveAdmissionDecision::Prepared(second_wave) = f::try_wave(&second_step)
    else {
        panic!("released reusable slot must admit immediately without maintenance")
    };
    let resident_after = root
        .dynamic_pool_status()
        .unwrap()
        .pools()
        .iter()
        .map(|pool| (pool.pool_id().clone(), pool.resident_bytes()))
        .collect::<Vec<_>>();
    assert_eq!(resident_after, resident_before);
    drop(second_wave);
    first_step.try_retire_normal().unwrap();
    second_step.try_retire_normal().unwrap();
    drop((first_batch, second_batch));
    first_session.try_complete().unwrap();
    second_session.try_complete().unwrap();
    drop((first_session, second_session, first, second, lane));
    f::close(root);
    f::close(competing);
}

#[test]
fn maintained_wave_records_only_real_growth_and_retains_its_step() {
    let catalog = f::catalog();
    let (plan, _) = f::plan(&catalog, false, 3);
    let runtime = f::runtime(&plan);
    let first_lane = ExecutionLane::create(runtime.clone()).unwrap();
    let second_lane = ExecutionLane::create(runtime.clone()).unwrap();
    let root = f::provision(&plan, runtime, "growth");
    let first = f::sequence(&root, "growth.first");
    let second = f::sequence(&root, "growth.second");
    let first_session = first.open_session().unwrap();
    let second_session = second.open_session().unwrap();
    let first_batch = ExecutionBatchParticipants::new(vec![first_session.clone()]).unwrap();
    let second_batch = ExecutionBatchParticipants::new(vec![second_session.clone()]).unwrap();
    let first_step = f::step(&first_batch, &first_lane, None);
    let first_wave = f::wave(&first_step);
    let second_step = f::step(&second_batch, &second_lane, None);
    let StepSubmissionWaveAdmissionDecision::BackingDeferred(deferred) = f::try_wave(&second_step)
    else {
        panic!("the second live wave requires new physical invocation backing")
    };
    let failed_retirement = second_step
        .try_retire_normal()
        .expect_err("deferral retains the actual step");
    let second_step = failed_retirement.into_step();
    let DynamicDeferredMaintenanceOutcome::Maintained(receipt) = deferred.maintain().unwrap()
    else {
        panic!("available device capacity must permit the declared growth")
    };
    let mut availability = Vec::new();
    let epochs = root
        .write_dynamic_capacity_availability(&mut availability)
        .unwrap();
    let status = root.dynamic_pool_status().unwrap();
    let progress = ExecutorExecutionMaintenanceProgress::from_growth_receipts(
        1,
        ExecutorAdmissionEpochs::from_capacity(epochs),
        std::slice::from_ref(&receipt),
        status.pools(),
    )
    .unwrap();
    assert_eq!(progress.attempts(), 1);
    assert_eq!(progress.latest_capacity_epoch(), epochs.capacity_epoch());
    assert!(!receipt.growths().is_empty());
    assert_eq!(progress.mutations().len(), receipt.growths().len());
    assert!(progress
        .mutations()
        .iter()
        .all(|mutation| receipt
            .growths()
            .iter()
            .any(|growth| growth.pool_id() == mutation.pool_id()
                && growth.chunk() == mutation.chunk()
                && growth.capacity_epoch() == mutation.capacity_epoch())));
    drop(deferred);
    let second_wave = f::wave(&second_step);
    drop((second_wave, first_wave));
    first_step.try_retire_normal().unwrap();
    second_step.try_retire_normal().unwrap();
    drop((first_batch, second_batch));
    first_session.try_complete().unwrap();
    second_session.try_complete().unwrap();
    drop((
        first_session,
        second_session,
        first,
        second,
        first_lane,
        second_lane,
    ));
    f::close(root);
}

#[test]
fn device_pressure_requires_the_real_maintenance_boundary() {
    let catalog = f::catalog();
    let (plan, _) = f::plan(&catalog, false, 3);
    let (competing_plan, _) = f::plan(&catalog, false, 128);
    let runtime = f::runtime(&plan);
    let lane = ExecutionLane::create(runtime.clone()).unwrap();
    let root = f::provision(&plan, runtime.clone(), "pressure");
    let first = f::sequence(&root, "pressure.first");
    let first_session = first.open_session().unwrap();
    let first_batch = ExecutionBatchParticipants::new(vec![first_session.clone()]).unwrap();
    let first_step = f::step(&first_batch, &lane, None);
    let first_wave = f::wave(&first_step);
    let sequence = f::sequence(&root, "pressure.second");
    let session = sequence.open_session().unwrap();
    let batch = ExecutionBatchParticipants::new(vec![session.clone()]).unwrap();
    let step = f::step(&batch, &lane, None);
    let StepSubmissionWaveAdmissionDecision::BackingDeferred(deferred) = f::try_wave(&step) else {
        panic!("the second live wave requires new physical invocation backing")
    };

    // Another legitimate plan consumes the same device's remaining capacity.
    // No private pool fields or fabricated maintenance receipt are used.
    let competing = f::provision_competing(&competing_plan, runtime);
    let DynamicDeferredMaintenanceOutcome::WaitForRelease {
        current_epochs,
        wait_condition,
        pressure,
        maintenance_boundary: Some(boundary),
    } = deferred.maintain().unwrap()
    else {
        panic!("competing capacity must yield a real maintenance-boundary wait")
    };
    assert!(pressure.device_capacity().is_some());
    assert_eq!(boundary.pressure(), &pressure);
    assert!(ExecutorExecutionCapacityDeferral::from_backing_maintenance(
        deferred.evidence(),
        ExecutorAdmissionEpochs::from_capacity(current_epochs),
        wait_condition.clone(),
        pressure.clone(),
        None,
        ExecutorExecutionCapacityStage::SubmissionWave,
    )
    .is_err());
    let product = ExecutorExecutionCapacityDeferral::from_backing_maintenance(
        deferred.evidence(),
        ExecutorAdmissionEpochs::from_capacity(current_epochs),
        wait_condition.clone(),
        pressure,
        Some(boundary.clone()),
        ExecutorExecutionCapacityStage::SubmissionWave,
    )
    .unwrap();
    assert_eq!(product.maintenance_boundary(), Some(&boundary));
    let waiter = root
        .register_capacity_waiter(product.wait_condition())
        .unwrap();
    assert!(!waiter.recheck().unwrap().should_retry());
    f::close(competing);
    assert!(waiter.recheck().unwrap().should_retry());
    drop(waiter);
    let DynamicDeferredMaintenanceOutcome::Maintained(_) = deferred.maintain().unwrap() else {
        panic!("release of the competing plan must allow retry")
    };
    drop(deferred);
    drop(f::wave(&step));
    step.try_retire_normal().unwrap();
    drop(first_wave);
    first_step.try_retire_normal().unwrap();
    drop((batch, first_batch));
    session.try_complete().unwrap();
    first_session.try_complete().unwrap();
    drop((session, sequence, first_session, first, lane));
    f::close(root);
}
