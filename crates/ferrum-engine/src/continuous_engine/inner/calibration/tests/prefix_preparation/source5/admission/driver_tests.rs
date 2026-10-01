//! Run the shared cohort executor through actual request, guard, actor and
//! source5 paths. No successful receipt or numerical model is manufactured.
use super::*;
use crate::continuous_engine::inner::calibration::cohort_driver::{
    ProbeCohortSettings, ProbeExecutionBudget, ProbeRequest,
};

fn settings() -> ProbeCohortSettings {
    ProbeCohortSettings {
        prefill_plan:
            crate::continuous_engine::inner::calibration::cohort_driver::ProbePrefillPlan::Joint,
        prefill_chunk: NonZeroU32::MIN,
        decode_route: CalibrationDecodeRoute::FullLogits,
        reset_token_policy: false,
    }
}

fn budget(requests: usize, attempts: usize) -> ProbeExecutionBudget {
    ProbeExecutionBudget::new(
        tokio::time::Instant::now() + Duration::from_secs(20),
        NonZeroUsize::new(requests).unwrap(),
        NonZeroUsize::new(attempts).unwrap(),
    )
}

fn ordinary(session: &CalibrationSession, maximum: usize) -> Vec<ProbeRequest> {
    vec![ProbeRequest {
        request: request(session, maximum),
        contract: Arc::new(OutputProjectionContract::cli_text()),
    }]
}

async fn assert_output_drained(session: &CalibrationSession) {
    let pool = session
        .engine
        .inner
        .output_credit_pool
        .get()
        .unwrap()
        .as_ref()
        .unwrap();
    let mut changed = pool.subscribe();
    bounded(async {
        loop {
            let state = pool.snapshot();
            if state.data_used == Default::default()
                && state.terminal_held == Default::default()
                && state.open_accounts == 0
                && state.retained_accounts == 0
            {
                break;
            }
            changed.changed().await.unwrap();
        }
    })
    .await;
    session.completed_owner_boundary().unwrap();
}

// Failure deliberately does not authorize ordinary service. Reap the actual
// pending wave, then let the session retire the aborted consumers' owners.
// These turns never offer another Wave or restore a spent execution budget.
async fn drain_failed_probe(session: &mut CalibrationSession) {
    bounded(async {
        loop {
            match session.step(CalibrationAction::Reap).await.unwrap() {
                CalibrationTurn::Reaped(report) => {
                    assert_ne!(
                        report.submission,
                        CalibrationSubmissionState::InFlightUnknown
                    );
                }
                CalibrationTurn::Blocked(_) => {}
                other => panic!("drain may only reap original work: {other:?}"),
            }
            if session.completed_owner_boundary().is_ok() {
                break;
            }
            tokio::task::yield_now().await;
        }
    })
    .await;
    assert_output_drained(session).await;
}

#[derive(Debug)]
struct ExecutedWave {
    prefill: Vec<RequestId>,
    decode: Vec<(RequestId, usize, TokenId, bool)>,
}

async fn driver_mixed_codec(chat: bool) {
    let (mut session, executor, source) = joint_unstarted_cohort(4).await;
    let requests: Vec<_> = super::probe_primitives::mixed_requests(&session, chat)
        .into_iter()
        .map(|(request, contract)| ProbeRequest {
            request,
            contract: Arc::new(contract),
        })
        .collect();
    let ids: Vec<_> = requests.iter().map(|row| row.request.id.clone()).collect();
    let actual = Arc::new(parking_lot::Mutex::new(Vec::<ExecutedWave>::new()));
    let captured = Arc::clone(&actual);
    // Existing controlled-backend observation hook only reads the real input.
    // Returning None preserves its normal original recorder/host receipts.
    *executor.actual_observation_unknown.lock() = Some(Box::new(move |prefill, decode| {
        captured.lock().push(ExecutedWave {
            prefill: prefill.iter().map(|row| row.request_id.clone()).collect(),
            decode: decode
                .iter()
                .map(|row| {
                    (
                        row.request_id.clone(),
                        row.kv_cache.num_tokens(),
                        row.input_token,
                        row.logits_policy.requires_full_logits(),
                    )
                })
                .collect(),
        });
        None
    }));
    let mut budget = budget(2, 32);
    let summary = bounded(session.run_probe_cohort(requests, settings(), &mut budget))
        .await
        .unwrap();
    executor.actual_observation_unknown.lock().take();
    assert_eq!(summary.completed_requests, 2);
    assert_eq!(summary.completed_output_tokens, 8);
    assert_eq!(summary.released_prefix_rows, 2);
    assert_eq!(summary.reconciled_waves, 4);
    assert!(summary.wave_attempts >= summary.reconciled_waves);
    assert_eq!(executor.physical.load(Ordering::Acquire), 4);
    assert_eq!(executor.completion_calls.load(Ordering::Acquire), 2);
    {
        let actual = actual.lock();
        assert_eq!(actual.len(), 4);
        assert_eq!(actual[0].prefill, ids);
        assert!(actual[0].decode.is_empty());
        for (index, call) in actual.iter().enumerate().skip(1) {
            assert!(call.prefill.is_empty());
            assert_eq!(call.decode.len(), 2);
            for (slot, (id, kv, token, full)) in call.decode.iter().enumerate() {
                assert_eq!(id, &ids[slot]);
                assert_eq!(*kv, index);
                if index >= 2 {
                    assert!(*full, "the released cohort really requested FullLogits");
                    assert_eq!(
                        token.get(),
                        match (index, slot) {
                            (2, 0) => 10,
                            (2, 1) => 11,
                            (3, 0) => 6,
                            (3, 1) => 12,
                            _ => unreachable!(),
                        }
                    );
                }
            }
        }
    }
    // G3's original-policy A9 continuation is visible as G4's actual input;
    // both requests kept their full output length and completion leases.
    assert_output_drained(&session).await;
    let artifact = session.finish_structured_cost_group_v2().await.unwrap();
    assert!(
        artifact.failure.is_some(),
        "this fixture has no qualified future projector"
    );
    assert!(artifact.children.iter().all(|child| child.model.is_none()));
    let raw = records(&source);
    assert_eq!(
        raw.iter()
            .filter(|r| r["kind"] == "preparation_offered")
            .count(),
        2
    );
    assert_eq!(
        raw.iter()
            .filter(|r| r["kind"] == "preparation_completed")
            .count(),
        2
    );
    assert_eq!(
        raw.iter()
            .filter(|r| r["kind"] == "preparation_released")
            .count(),
        2
    );
    assert!(!raw.iter().any(|r| r["kind"] == "phase_freeze"));
    session.shutdown().await.unwrap();
}

#[tokio::test]
async fn probe_cohort_driver_actual_joint_chat_prefix_and_full_suffix() {
    driver_mixed_codec(true).await;
}

#[tokio::test]
async fn probe_cohort_driver_actual_joint_completions_prefix_and_full_suffix() {
    driver_mixed_codec(false).await;
}

#[tokio::test]
async fn probe_cohort_driver_shares_request_budget_across_complete_cohorts() {
    let (mut session, executor) = prepared_session().await;
    let mut shared = budget(1, 16);
    let first = ordinary(&session, 1);
    let summary = bounded(session.run_probe_cohort(first, settings(), &mut shared))
        .await
        .unwrap();
    assert_eq!(summary.completed_requests, 1);
    assert_output_drained(&session).await;
    let before = executor.physical.load(Ordering::Acquire);
    let second = ordinary(&session, 1);
    let error = session
        .run_probe_cohort(second, settings(), &mut shared)
        .await
        .unwrap_err();
    assert!(
        error.to_string().contains("request budget exhausted"),
        "{error}"
    );
    assert_eq!(executor.physical.load(Ordering::Acquire), before);
    session.completed_owner_boundary().unwrap();
    session.shutdown().await.unwrap();
}

#[tokio::test]
async fn probe_cohort_driver_shares_wave_budget_and_drains_incomplete_next_cohort() {
    let (mut session, executor) = prepared_session().await;
    executor
        .recycle_completed_bindings
        .store(true, Ordering::Release);
    let mut shared = budget(3, 3);
    let first = ordinary(&session, 1);
    let summary = bounded(session.run_probe_cohort(first, settings(), &mut shared))
        .await
        .unwrap();
    assert_eq!(summary.completed_requests, 1);
    assert_output_drained(&session).await;
    let second = ordinary(&session, 3);
    let error = bounded(session.run_probe_cohort(second, settings(), &mut shared))
        .await
        .unwrap_err();
    assert!(
        error.to_string().contains("wave budget exhausted"),
        "{error}"
    );
    assert!(session.completed_owner_boundary().is_err());
    let before = executor.physical.load(Ordering::Acquire);
    assert!(
        before < 4,
        "the three-wave budget was shared, not reset for cohort2"
    );
    drain_failed_probe(&mut session).await;
    assert_eq!(
        executor.physical.load(Ordering::Acquire),
        before,
        "drain offers no replacement work"
    );
    let third = ordinary(&session, 1);
    let error = bounded(session.run_probe_cohort(third, settings(), &mut shared))
        .await
        .unwrap_err();
    assert!(
        error.to_string().contains("wave budget exhausted"),
        "{error}"
    );
    drain_failed_probe(&mut session).await;
    assert_eq!(executor.physical.load(Ordering::Acquire), before);
    session.shutdown().await.unwrap();
}

#[tokio::test(start_paused = true)]
async fn probe_cohort_driver_deadline_waits_for_original_transaction_before_cleanup() {
    let (mut session, executor) = prepared_session().await;
    executor.park.store(true, Ordering::Release);
    let mut shared = ProbeExecutionBudget::new(
        tokio::time::Instant::now() + Duration::from_secs(10),
        NonZeroUsize::new(2).unwrap(),
        NonZeroUsize::new(16).unwrap(),
    );
    let requests = ordinary(&session, 3);
    let mut driver = Box::pin(session.run_probe_cohort(requests, settings(), &mut shared));
    bounded(async {
        tokio::select! {
            result = driver.as_mut() => panic!("driver ended before original executor entry: {result:?}"),
            () = executor.entered.notified() => {}
        }
    }).await;
    tokio::time::advance(Duration::from_secs(11)).await;
    assert!(
        driver.as_mut().now_or_never().is_none(),
        "deadline retains the original owned transaction and output consumers"
    );
    assert_eq!(executor.entries.load(Ordering::Acquire), 1);
    assert_eq!(
        executor.physical.load(Ordering::Acquire),
        0,
        "this gate deliberately precedes physical submission"
    );
    // Deadline behavior is established; let the real cost worker run before
    // the unchanged settlement watchdog can expire.
    tokio::time::resume();
    executor.park.store(false, Ordering::Release);
    executor.resume.notify_one();
    let error = bounded(driver).await.unwrap_err();
    assert!(
        error.to_string().contains("duration budget expired"),
        "{error}"
    );
    assert!(
        session.pending.is_none(),
        "the owned transaction settled before deadline return"
    );
    assert!(session.completed_owner_boundary().is_err());
    assert!(session.engine.inner.manual_calibration_driver);
    assert_eq!(shared.requests_remaining(), 1);
    assert_eq!(shared.attempts_remaining(), 15);
    assert_eq!(
        executor.physical.load(Ordering::Acquire),
        1,
        "the pre-deadline owned transaction retained its valid output leases"
    );
    drain_failed_probe(&mut session).await;
    assert_eq!(
        executor.entries.load(Ordering::Acquire),
        1,
        "cleanup cannot start another wave"
    );
    let next = ordinary(&session, 1);
    let error = session
        .run_probe_cohort(next, settings(), &mut shared)
        .await
        .unwrap_err();
    assert!(
        error.to_string().contains("no request/time budget"),
        "{error}"
    );
    assert_eq!(shared.requests_remaining(), 1);
    assert_eq!(shared.attempts_remaining(), 15);
    assert_eq!(executor.physical.load(Ordering::Acquire), 1);
    session.shutdown().await.unwrap();
}

#[tokio::test(start_paused = true)]
async fn probe_cohort_driver_external_cancel_keeps_original_pending_work_until_drain() {
    let (mut session, executor) = prepared_session().await;
    executor.park.store(true, Ordering::Release);
    let mut shared = ProbeExecutionBudget::new(
        tokio::time::Instant::now() + Duration::from_secs(10),
        NonZeroUsize::new(2).unwrap(),
        NonZeroUsize::new(16).unwrap(),
    );
    let requests = ordinary(&session, 3);
    let mut driver = Box::pin(session.run_probe_cohort(requests, settings(), &mut shared));
    bounded(async {
        tokio::select! {
            result = driver.as_mut() => panic!("driver ended before original executor entry: {result:?}"),
            () = executor.entered.notified() => {}
        }
    }).await;
    // An unrelated caller can still explicitly cancel its future. This is
    // distinct from the driver's deadline and must retain durable session work.
    drop(driver);
    assert!(session.pending.is_some());
    assert!(session.completed_owner_boundary().is_err());
    assert!(session.engine.inner.manual_calibration_driver);
    assert_eq!(shared.requests_remaining(), 1);
    assert_eq!(shared.attempts_remaining(), 15);
    let next = ordinary(&session, 1);
    assert!(session
        .run_probe_cohort(next, settings(), &mut shared)
        .await
        .is_err());
    tokio::time::advance(Duration::from_secs(11)).await;
    // The original deadline is now expired. Reaping waits for the cost worker's
    // real OS thread, which a paused Tokio clock cannot observe as runnable.
    // Keep the same watchdog on real time while that original work settles.
    tokio::time::resume();
    executor.park.store(false, Ordering::Release);
    executor.resume.notify_one();
    drain_failed_probe(&mut session).await;
    assert_eq!(
        executor.entries.load(Ordering::Acquire),
        1,
        "reap must never resubmit"
    );
    let physical = executor.physical.load(Ordering::Acquire);
    assert!(
        physical <= 1,
        "explicit consumer cancellation may invalidate the pre-submit guard"
    );
    let next = ordinary(&session, 1);
    let error = session
        .run_probe_cohort(next, settings(), &mut shared)
        .await
        .unwrap_err();
    assert!(
        error.to_string().contains("no request/time budget"),
        "{error}"
    );
    assert_eq!(shared.requests_remaining(), 1);
    assert_eq!(shared.attempts_remaining(), 15);
    assert_eq!(executor.physical.load(Ordering::Acquire), physical);
    session.shutdown().await.unwrap();
}

#[tokio::test]
async fn probe_cohort_driver_unsupported_mask_reset_never_claims_cold_work() {
    let (mut session, executor) = prepared_session().await;
    let mut shared = budget(1, 16);
    let requests = ordinary(&session, 3);
    let mut reset = settings();
    reset.reset_token_policy = true;
    let error = session
        .run_probe_cohort(requests, reset, &mut shared)
        .await
        .unwrap_err();
    assert!(
        error.to_string().contains("residency reset unavailable"),
        "{error}"
    );
    assert_eq!(executor.entries.load(Ordering::Acquire), 0);
    assert_eq!(executor.physical.load(Ordering::Acquire), 0);
    session.completed_owner_boundary().unwrap();
    session.shutdown().await.unwrap();
}
