//! A deadline after real backend submission must not cancel the original
//! output owner before host settlement or revoke an earlier qualified source.
use super::*;
use futures::FutureExt;

#[tokio::test]
async fn startup_series_deadline_settles_submitted_prefix_before_cancelling_owners() {
    let (mut session, executor) = automatic_session().await;
    executor
        .native_structured_submission
        .store(true, Ordering::Release);
    executor.enable_structured_query_route();
    let runtime = session.engine.inner.cost_runtime.as_ref().unwrap().clone();
    let deadline = tokio::time::Instant::now() + Duration::from_secs(60);
    session.begin_startup_owner_series(2, deadline).unwrap();
    session
        .begin_prepared_owner_source(full_population(&session), CostProfileLoadLimits::default())
        .await
        .unwrap();
    let options = ProbeCohortSettings {
        prefill_plan:
            crate::continuous_engine::inner::calibration::cohort_driver::ProbePrefillPlan::Joint,
        prefill_chunk: NonZeroU32::MIN,
        decode_route: CalibrationDecodeRoute::FullLogits,
        reset_token_policy: false,
    };
    let mut budget = ProbeExecutionBudget::new(
        deadline,
        NonZeroUsize::new(66).unwrap(),
        NonZeroUsize::new(97).unwrap(),
    );
    for pass in 0..3 {
        for ordinal in 0..[16, 8, 8][pass] {
            session.begin_prepared_owner_cohort(pass, ordinal).unwrap();
            let requests = probe_requests(&session);
            session
                .run_probe_cohort(requests, options, &mut budget)
                .await
                .unwrap();
            session.end_prepared_owner_cohort().unwrap();
        }
    }
    let epoch = session.activate_prepared_owner_source().await.unwrap();
    let original = runtime.snapshot().unwrap();
    let original_receipt = runtime.profile_receipt().unwrap();
    assert_eq!(runtime.catalog_live_state(), Some((epoch, true)));
    session.retire_startup_owner_source().await.unwrap();
    session
        .begin_prepared_owner_source(full_population(&session), CostProfileLoadLimits::default())
        .await
        .unwrap();
    session.begin_prepared_owner_cohort(0, 0).unwrap();
    let requests = probe_requests(&session);
    let before = executor.physical.load(Ordering::Acquire);
    let commands_before = executor.native_structured_counts();
    assert_eq!(before, 96);
    assert_eq!(commands_before, (before as u64, before * 2));
    // The fixture gate is AFTER actual backend submission and its completed
    // recorder receipt, BEFORE outputs return to the original host settlement.
    executor.park_after_submit.store(true, Ordering::Release);
    tokio::time::pause();
    let mut cohort = Box::pin(session.run_probe_cohort(requests, options, &mut budget));
    tokio::select! {
        () = executor.submitted.notified() => {}
        result = cohort.as_mut() => panic!("cohort returned before submitted gate: {result:?}"),
    }
    assert_eq!(executor.physical.load(Ordering::Acquire), before + 1);
    assert_eq!(
        executor.native_structured_counts(),
        (commands_before.0 + 1, commands_before.1 + 2),
        "the gated wave must have submitted its real CPU command pair exactly once"
    );
    tokio::time::advance(
        deadline.saturating_duration_since(tokio::time::Instant::now()) + Duration::from_millis(1),
    )
    .await;
    assert!(
        cohort.as_mut().now_or_never().is_none(),
        "deadline must retain the original submitted transaction and its consumers"
    );
    assert_eq!(runtime.catalog_live_state(), Some((epoch, true)));
    executor.park_after_submit.store(false, Ordering::Release);
    executor.resume_after_submit.notify_one();
    let error = cohort.await.unwrap_err();
    assert!(
        error.to_string().contains("duration budget expired"),
        "{error}"
    );
    // No second wave or replacement request is allowed after the same deadline.
    assert_eq!(executor.physical.load(Ordering::Acquire), before + 1);
    assert_eq!(
        executor.native_structured_counts(),
        (commands_before.0 + 1, commands_before.1 + 2),
        "the gated wave must have submitted its real CPU command pair exactly once"
    );
    assert_eq!(budget.requests_remaining(), 0);
    assert_eq!(budget.attempts_remaining(), 0);
    session.drain_startup_geometry().await.unwrap();
    session.freeze_cost_model().await.unwrap();
    assert!(session.frontiers().unwrap().is_empty());
    assert_eq!(session.engine.inner.scheduler.active_count(), 0);
    assert_eq!(session.engine.inner.scheduler.waiting_count(), 0);
    assert!(
        session.activate_prepared_owner_source().await.is_err(),
        "an interrupted source cannot acquire a complete checkpoint"
    );
    assert!(session.retire_startup_owner_source().await.is_err());
    assert!(Arc::ptr_eq(&original, &runtime.snapshot().unwrap()));
    assert_eq!(runtime.profile_receipt().unwrap(), original_receipt);
    assert_eq!(runtime.catalog_live_state(), Some((epoch, true)));
    let audit = serde_json::to_value(runtime.audit_snapshot()).unwrap();
    let feedback = &audit["structured_feedback"];
    assert!(feedback["revoked"].is_null(), "{feedback:#}");
    assert_eq!(feedback["uncomparable_observations"], 0);
    assert_eq!(feedback["failed_or_partial"], 0);
    assert_eq!(
        feedback["outside_preparation_observations"], 1,
        "only a real complete original preparation may be excluded"
    );
    session.finish_startup_owner_series().unwrap();
    assert_eq!(session.startup_owner_series_installed_epoch(), Some(epoch));
    runtime.begin_automatic_calibration().unwrap();
    runtime.shutdown().await.unwrap();
}
