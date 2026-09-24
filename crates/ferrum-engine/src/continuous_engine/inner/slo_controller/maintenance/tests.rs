//! The scheduler tests cover live retry tickets. These engine tests cover the
//! empty-turn hook's cancellation/try-lock boundaries without inventing an
//! executor's private physical-maintenance receipt.
use super::*;
use crate::continuous_engine::inner::slo_controller::tests::fixture::fixture_with_width;
use ferrum_interfaces::engine::InferenceEngine;

#[tokio::test]
async fn maintenance_fairness_without_a_live_ticket_does_not_advance_or_submit() {
    let (mut engine, scheduler, executor) = fixture_with_width(1).await;
    Arc::get_mut(&mut engine.inner)
        .unwrap()
        .config
        .scheduler
        .slo
        .mode = ferrum_types::SloMode::Enforce;
    let before = scheduler.trace_snapshot();
    // The owner may have been cancelled, or a peer may already have consumed
    // the scheduling opportunity between maintenance and the next idle turn.
    engine
        .inner
        .slo_controller
        .lock()
        .pending_maintenance_fairness = true;
    assert!(!engine
        .inner
        .consume_controller_maintenance_fairness_turn()
        .unwrap());
    assert!(
        !engine
            .inner
            .slo_controller
            .lock()
            .pending_maintenance_fairness
    );
    let after = scheduler.trace_snapshot();
    assert_eq!(after.current_iteration, before.current_iteration);
    assert_eq!(after.capacity_release_epoch, before.capacity_release_epoch);
    assert_eq!(executor.physical.load(Ordering::Acquire), 0);
    assert!(engine.inner.slo_controller.lock().retry.is_none());
    engine.shutdown().await.unwrap();
}

#[tokio::test(start_paused = true)]
async fn maintenance_fairness_contention_and_observe_never_grant_work() {
    let (mut engine, scheduler, executor) = fixture_with_width(1).await;
    engine
        .inner
        .slo_controller
        .lock()
        .pending_maintenance_fairness = true;
    let before = scheduler.trace_snapshot();
    // Observe does not advance scheduler state or arm execution retries.
    assert!(!engine
        .inner
        .consume_controller_maintenance_fairness_turn()
        .unwrap());
    assert!(
        engine
            .inner
            .slo_controller
            .lock()
            .pending_maintenance_fairness
    );
    assert!(engine.inner.slo_controller.lock().retry.is_none());
    Arc::get_mut(&mut engine.inner)
        .unwrap()
        .config
        .scheduler
        .slo
        .mode = ferrum_types::SloMode::Enforce;
    let held = engine.inner.dynamic_admission_availability.lock();
    assert!(!engine
        .inner
        .consume_controller_maintenance_fairness_turn()
        .unwrap());
    assert!(
        engine
            .inner
            .slo_controller
            .lock()
            .pending_maintenance_fairness
    );
    assert!(engine.inner.controller_retry_pending());
    drop(held);
    // A new call cannot turn lock contention into an immediate busy retry.
    assert!(!engine
        .inner
        .consume_controller_maintenance_fairness_turn()
        .unwrap());
    let after = scheduler.trace_snapshot();
    assert_eq!(after.current_iteration, before.current_iteration);
    assert_eq!(after.capacity_release_epoch, before.capacity_release_epoch);
    assert_eq!(executor.physical.load(Ordering::Acquire), 0);
    engine.inner.clear_controller_maintenance();
    assert!(
        !engine
            .inner
            .slo_controller
            .lock()
            .pending_maintenance_fairness
    );
    engine.shutdown().await.unwrap();
}
