use super::*;
use crate::vnext::{
    ReusableExecutionBucketSpec, ReusableExecutionCapacity, ReusableExecutionClassId,
};

#[path = "../../../../tests/vnext_device_operation_contract/mod.rs"]
mod operation_fixture;
use operation_fixture::{catalog, runtime, TestRuntime};

impl<R: DeviceRuntime> ExecutionLane<R> {
    pub(crate) fn steady_recipe_test_locks_available(&self) -> bool {
        let Ok(_lane) = self.state.try_lock() else {
            return false;
        };
        self.steady_recipes
            .get()
            .is_none_or(|cache| cache.nodes.try_lock().is_ok())
    }
}

fn program(lane: &ExecutionLane<TestRuntime>) -> DeviceReusableExecutionProgramId {
    let bucket = ReusableExecutionBucketSpec::new(
        ReusableExecutionClassId::new("steady-entry-test").unwrap(),
        ReusableExecutionCapacity::new(2, 2, 1).unwrap(),
    )
    .unwrap();
    DeviceReusableExecutionProgramId::new(
        serde_json::from_value(serde_json::json!("a".repeat(64))).unwrap(),
        lane.descriptor().runtime_implementation_fingerprint.clone(),
        lane.id(),
        bucket.bucket_id().clone(),
        "b".repeat(64),
        "c".repeat(64),
        1,
        2,
        2,
        1,
    )
    .unwrap()
}

#[test]
fn enqueue_requires_the_same_live_entry_even_without_an_epoch_change() {
    let (runtime, trace) = runtime(&catalog());
    let lane = ExecutionLane::create(runtime).unwrap();
    let program = program(&lane);
    let original = DeviceReusableExecutionEntryIdentity::new();
    trace.lock().unwrap().steady_recipe_entry = Some(original.clone());
    let epoch = lane.reusable_execution_epoch();

    let observed = lane
        .reusable_execution_entry_identity(&program)
        .unwrap()
        .unwrap();
    assert!(observed.same_entry(&original));
    lane.reserve_enqueue()
        .unwrap()
        .validate_steady_recipe_entry(&program, &observed, epoch)
        .unwrap();

    trace.lock().unwrap().steady_recipe_entry = Some(DeviceReusableExecutionEntryIdentity::new());
    assert_eq!(lane.reusable_execution_epoch(), epoch);
    assert!(lane
        .reserve_enqueue()
        .unwrap()
        .validate_steady_recipe_entry(&program, &observed, epoch)
        .is_err());
    assert!(lane.is_reusable());

    trace.lock().unwrap().steady_recipe_entry = None;
    assert!(lane
        .reserve_enqueue()
        .unwrap()
        .validate_steady_recipe_entry(&program, &observed, epoch)
        .is_err());
    assert_eq!(trace.lock().unwrap().submit_calls, 0);
    assert!(lane.is_reusable());
}

#[test]
fn successful_trim_invalidates_epoch_and_replacement_can_be_revalidated() {
    let (runtime, trace) = runtime(&catalog());
    let lane = ExecutionLane::create(runtime).unwrap();
    let program = program(&lane);
    let original = DeviceReusableExecutionEntryIdentity::new();
    trace.lock().unwrap().steady_recipe_entry = Some(original.clone());
    let epoch = lane.reusable_execution_epoch();
    trace.lock().unwrap().steady_recipe_trim_released = 1;

    assert!(lane.trim_reusable_executables_if_quiescent().unwrap());
    assert_eq!(lane.reusable_execution_epoch(), epoch + 1);
    assert!(lane
        .reusable_execution_entry_identity(&program)
        .unwrap()
        .is_none());
    assert!(lane
        .reserve_enqueue()
        .unwrap()
        .validate_steady_recipe_entry(&program, &original, epoch)
        .is_err());

    let replacement = DeviceReusableExecutionEntryIdentity::new();
    trace.lock().unwrap().steady_recipe_entry = Some(replacement.clone());
    lane.reserve_enqueue()
        .unwrap()
        .validate_steady_recipe_entry(&program, &replacement, epoch + 1)
        .unwrap();
}

#[test]
fn busy_entry_lookup_never_waits_or_initializes_the_recipe_cache() {
    let (runtime, trace) = runtime(&catalog());
    let lane = ExecutionLane::create(runtime).unwrap();
    let program = program(&lane);
    trace.lock().unwrap().steady_recipe_entry = Some(DeviceReusableExecutionEntryIdentity::new());
    let guard = lane.reserve_enqueue().unwrap();
    assert!(lane
        .try_reusable_execution_entry_identity(&program)
        .unwrap()
        .is_none());
    assert!(lane.steady_recipes.get().is_none());
    drop(guard);
    assert!(lane
        .try_reusable_execution_entry_identity(&program)
        .unwrap()
        .is_some());
    assert!(lane.steady_recipes.get().is_none());
}

#[test]
fn entry_inspection_error_and_panic_fail_closed_before_submission() {
    for panic in [false, true] {
        let (runtime, trace) = runtime(&catalog());
        let lane = ExecutionLane::create(runtime).unwrap();
        let program = program(&lane);
        let entry = DeviceReusableExecutionEntryIdentity::new();
        let guard = lane.reserve_enqueue().unwrap();
        {
            let mut trace = trace.lock().unwrap();
            trace.steady_recipe_entry_error = !panic;
            trace.steady_recipe_entry_panic = panic;
        }
        assert!(guard
            .validate_steady_recipe_entry(&program, &entry, lane.reusable_execution_epoch())
            .is_err());
        drop(guard);
        assert!(lane.is_fail_closed());
        assert_eq!(trace.lock().unwrap().submit_calls, 0);
    }
}

#[test]
fn runtime_descriptor_drift_after_reservation_fails_closed() {
    let (runtime, trace) = runtime(&catalog());
    let lane = ExecutionLane::create(Arc::clone(&runtime)).unwrap();
    let program = program(&lane);
    let entry = DeviceReusableExecutionEntryIdentity::new();
    trace.lock().unwrap().steady_recipe_entry = Some(entry.clone());
    let guard = lane.reserve_enqueue().unwrap();
    runtime
        .use_alternate_descriptor
        .store(true, Ordering::Release);
    assert!(guard
        .validate_steady_recipe_entry(&program, &entry, lane.reusable_execution_epoch())
        .is_err());
    drop(guard);
    assert!(lane.is_fail_closed());
    assert_eq!(trace.lock().unwrap().submit_calls, 0);
}
