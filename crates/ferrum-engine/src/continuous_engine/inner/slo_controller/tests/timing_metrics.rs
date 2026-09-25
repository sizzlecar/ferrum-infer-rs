//! Public timing evidence follows real early-return/failed controller paths.
use super::*;
use ferrum_interfaces::engine::InferenceEngine;

async fn run_until_executor(
    engine: &ContinuousBatchEngine,
    executor: &ControlledExecutor,
) -> Result<EngineIterationOutcome> {
    bounded(async {
        loop {
            let result = engine.inner.run_iteration().await;
            if executor.entries.load(Ordering::Acquire) != 0 || result.is_err() {
                return result;
            }
            // Only real controller retry evidence permits another attempt.
            assert!(engine.inner.slo_controller.lock().retry.is_some());
            engine.inner.wait_for_slo_controller_retry().await;
        }
    })
    .await
}

#[tokio::test(start_paused = true)]
async fn controller_timing_selected_early_return_exposes_executor_wait_separately() {
    let (engine, _, executor) = completion_fixture(1).await;
    let (id, session) = prefill::request(&engine, 1, 4).await;
    prefill::admit(&engine, 1).await;
    executor.park.store(true, Ordering::Release);
    let mut iteration = Box::pin(run_until_executor(&engine, &executor));
    assert!(iteration.as_mut().now_or_never().is_none());
    bounded(executor.entered.notified()).await;
    // A parked real guarded call is not a finalized transaction yet.
    assert_eq!(
        engine
            .metrics()
            .performance_breakdown
            .controller_timing
            .as_ref()
            .map_or(0, |timing| timing.submitted),
        0
    );
    tokio::time::advance(Duration::from_millis(31)).await;
    executor.resume.notify_one();
    assert!(matches!(
        iteration.await.unwrap(),
        EngineIterationOutcome::Progressed
    ));
    assert_eq!(engine.inner.sequences.read()[&id].generated_tokens.len(), 1);
    let breakdown = engine.metrics().performance_breakdown;
    let timing = breakdown.controller_timing.unwrap();
    assert_eq!(timing.submitted, 1);
    assert_eq!(timing.backend_submitted, 1);
    assert_eq!(timing.host_reconciled, 1);
    // This fixture uses completion authority, not a feasible SLO decision.
    assert_eq!(
        timing.witnesses,
        ferrum_types::ControllerWitnessMetrics::default()
    );
    assert_eq!(timing.executor_await.calls, 1);
    assert!(timing.executor_await.wall_ns_total >= 31_000_000);
    assert!(timing.transaction.wall_ns_total >= 31_000_000);
    assert!(timing.host_guard.calls >= 2);
    assert!(timing.planning.wall_ns_total < 31_000_000);
    assert_eq!(
        engine.inner.scheduling_time_samples.load(Ordering::Acquire),
        0
    );
    assert_eq!(breakdown.scheduling_time_ms, 0.0);
    assert!(engine.inner.drain_slo_execution().await.unwrap().is_none());
    assert_eq!(
        engine
            .metrics()
            .performance_breakdown
            .controller_timing
            .as_ref(),
        Some(&timing)
    );
    cleanup(engine, session).await;
}

#[tokio::test]
async fn controller_timing_observe_does_not_double_count_legacy_and_off_has_none() {
    for mode in [ferrum_types::SloMode::Off, ferrum_types::SloMode::Observe] {
        let (mut engine, _, executor) = fixture().await;
        Arc::get_mut(&mut engine.inner)
            .unwrap()
            .config
            .scheduler
            .slo
            .mode = mode;
        assert!(engine
            .metrics()
            .performance_breakdown
            .controller_timing
            .is_none());
        engine.inner.run_iteration().await.unwrap();
        assert_eq!(
            engine.inner.scheduling_time_samples.load(Ordering::Acquire),
            1
        );
        assert_eq!(executor.entries.load(Ordering::Acquire), 0);
        let timing = engine.metrics().performance_breakdown.controller_timing;
        if mode == ferrum_types::SloMode::Off {
            assert!(timing.is_none());
        } else {
            let timing = timing.unwrap();
            assert_eq!(timing.finalized_transactions, 1);
            assert_eq!(timing.observed, 1);
            assert_eq!(timing.unknown_decisions, 1);
            assert_eq!(timing.capture.calls, 1);
            assert_eq!(timing.executor_await.calls, 0);
            assert_eq!(timing.backend_submitted, 0);
            assert_eq!(timing.host_reconciled, 0);
            assert_eq!(timing.witnesses.decisions.samples, 0);
        }
        engine.shutdown().await.unwrap();
    }
}

#[tokio::test]
async fn controller_timing_failed_submission_remains_visible_without_replay() {
    let (engine, _, executor) = completion_fixture(1).await;
    let (id, session) = prefill::request(&engine, 1, 4).await;
    prefill::admit(&engine, 1).await;
    executor.fail_after_submit.store(true, Ordering::Release);
    assert!(run_until_executor(&engine, &executor).await.is_err());
    assert_eq!(executor.physical.load(Ordering::Acquire), 1);
    assert!(!engine.inner.sequences.read().contains_key(&id));
    let timing = engine
        .metrics()
        .performance_breakdown
        .controller_timing
        .unwrap();
    assert_eq!(timing.failed, 1);
    // A real Submitted(Err) must survive cleanup as submitted, not reconciled.
    assert_eq!(timing.backend_submitted, 1);
    assert_eq!(timing.host_reconciled, 0);
    assert_eq!(timing.witnesses.decisions.samples, 0);
    assert_eq!(timing.executor_await.calls, 1);
    assert!(timing.host_guard.calls >= 2);
    assert_eq!(
        engine.inner.scheduling_time_samples.load(Ordering::Acquire),
        0
    );
    assert!(engine.inner.drain_slo_execution().await.unwrap().is_none());
    assert_eq!(
        engine
            .metrics()
            .performance_breakdown
            .controller_timing
            .as_ref(),
        Some(&timing)
    );
    cleanup(engine, session).await;
}

#[tokio::test(start_paused = true)]
async fn controller_timing_nested_wall_intervals_are_not_added_or_reemitted() {
    let (engine, _, _) = fixture().await;
    let budget = ControllerBudget::new(slo_clock_now(), Duration::from_millis(2)).unwrap();
    {
        let _capture = budget.stage(ControllerStage::Capture);
        tokio::time::advance(Duration::from_millis(1)).await;
    }
    assert!(budget.finish_planning());
    {
        let _executor = budget.stage(ControllerStage::ExecutorAwait);
        tokio::time::advance(Duration::from_millis(20)).await;
        {
            let _guard = budget.stage(ControllerStage::HostGuard);
            tokio::time::advance(Duration::from_millis(1)).await;
        }
        tokio::time::advance(Duration::from_millis(30)).await;
    }
    engine.inner.finish_controller_audit(&budget, "withdrawn");
    engine.inner.finish_controller_audit(&budget, "failed");
    let timing = engine
        .metrics()
        .performance_breakdown
        .controller_timing
        .unwrap();
    assert_eq!(timing.finalized_transactions, 1);
    assert_eq!(timing.withdrawn, 1);
    assert_eq!(timing.failed, 0);
    assert_eq!(timing.planning.wall_ns_total, 1_000_000);
    assert_eq!(timing.executor_await.wall_ns_total, 51_000_000);
    assert_eq!(timing.host_guard.wall_ns_total, 1_000_000);
    assert_eq!(timing.transaction.wall_ns_total, 52_000_000);
    assert_eq!(timing.hard_budget_exhaustions, 0);
    // Outcome text cannot manufacture either execution receipt or a witness.
    assert_eq!(timing.backend_submitted, 0);
    assert_eq!(timing.host_reconciled, 0);
    assert_eq!(timing.witnesses.decisions.samples, 0);
    engine.shutdown().await.unwrap();
}

#[tokio::test(start_paused = true)]
async fn controller_timing_missing_planning_end_is_not_a_max_u64_latency() {
    let (engine, _, _) = fixture().await;
    let budget = ControllerBudget::new(slo_clock_now(), Duration::from_millis(2)).unwrap();
    tokio::time::advance(Duration::from_millis(1)).await;
    engine.inner.finish_controller_audit(&budget, "failed");
    let timing = engine
        .metrics()
        .performance_breakdown
        .controller_timing
        .unwrap();
    assert_eq!(timing.failed, 1);
    assert_eq!(timing.unfinished_planning_transactions, 1);
    assert_eq!(
        timing.planning,
        ferrum_types::WallTimingAggregate::default()
    );
    assert_eq!(timing.transaction.wall_ns_total, 1_000_000);
    engine.shutdown().await.unwrap();
}
