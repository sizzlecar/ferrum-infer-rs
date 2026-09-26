//! Draft lifetime, real publication and fresh fairness; no native speed claim.
use super::*;
use crate::continuous_engine::inner::slo_controller::completion::CompletionDraft;

async fn draft(
    engine: &ContinuousBatchEngine,
    hint: &ferrum_interfaces::BatchHint,
    allowance: Duration,
) -> (Arc<ControllerBudget>, CompletionDraft) {
    let until = tokio::time::Instant::now() + Duration::from_secs(3);
    loop {
        let budget = ControllerBudget::new(slo_clock_now(), allowance).unwrap();
        match engine.inner.capture_completion_draft(hint, &budget) {
            Ok(draft) => return (budget, draft),
            Err(error) => {
                assert!(
                    matches!(
                        error.retry,
                        Some(super::super::super::retry::ControllerRetryReason::SnapshotBusy)
                    ),
                    "{error:?}"
                );
                assert!(tokio::time::Instant::now() < until, "{error:?}");
                tokio::time::sleep(Duration::from_millis(1)).await;
            }
        }
    }
}

fn assert_unpublished(engine: &ContinuousBatchEngine, executor: &ControlledExecutor) {
    let state = engine.inner.slo_controller.lock();
    assert!(state.pending_execution.is_none());
    assert_eq!(scheduled(engine), 0);
    assert_eq!(executor.entries.load(Ordering::Acquire), 0);
    assert_eq!(executor.physical.load(Ordering::Acquire), 0);
    assert!(engine
        .inner
        .sequences
        .read()
        .values()
        .all(|sequence| sequence.credited_output.as_ref().unwrap().grant.is_none()));
}

#[tokio::test(start_paused = true)]
async fn discarded_draft_does_not_publish_reserve_or_normalize_persistent_order() {
    let (engine, _, executor) = completion_fixture(2).await;
    let (_, first) = request(&engine, 2, 4, None).await;
    let (_, second) = request(&engine, 2, 4, None).await;
    admit(&engine, 2).await;
    let before = engine.inner.slo_controller.lock().completion_order.clone();
    let (_, prepared) = draft(
        &engine,
        &ferrum_interfaces::BatchHint::simple(1),
        Duration::from_secs(30),
    )
    .await;
    assert_unpublished(&engine, &executor);
    {
        let state = engine.inner.slo_controller.lock();
        assert_eq!(state.completion_order, before);
        assert!(state.completion_next.is_none() && state.retry.is_none());
    }
    drop(prepared);
    assert_unpublished(&engine, &executor);
    assert_eq!(engine.inner.slo_controller.lock().completion_order, before);
    drop(second);
    cleanup(engine, first).await;
}

#[tokio::test(start_paused = true)]
async fn optional_capture_stop_publishes_existing_draft_inside_original_hard_budget() {
    let (engine, _, executor) = completion_fixture(1).await;
    let (id, session) = request(&engine, 1, 2, None).await;
    admit(&engine, 1).await;
    let hint = ferrum_interfaces::BatchHint::simple(1);
    let (budget, prepared) = draft(&engine, &hint, Duration::from_millis(2)).await;
    let phase = budget.completion_optional_phase(prepared.preparation_wall, 20);
    tokio::time::advance(Duration::from_micros(1_600)).await;
    let captured = engine.inner.capture_slo_controller_snapshot_in_phase(
        &hint,
        budget.clone(),
        Some(phase),
        &mut None,
    );
    assert!(matches!(
        captured,
        Err(Unavailable {
            reason: "compute_budget_exhausted",
            ..
        })
    ));
    assert!(
        budget.poll(),
        "optional work must leave the original publication allowance"
    );
    let selected = engine
        .inner
        .publish_completion_draft(
            &hint,
            &budget,
            prepared,
            CompletionOnlyReason::SearchInconclusive,
        )
        .unwrap();
    let SloIterationPlan::Selected(selected) = selected else {
        panic!("expected same-transaction publication")
    };
    assert!(budget.finish_planning());
    {
        let state = engine.inner.slo_controller.lock();
        assert!(state.completion_next.is_none() && state.retry.is_none());
    }
    assert!(matches!(
        engine
            .inner
            .execute_slo_controller_wave(selected)
            .await
            .unwrap(),
        EngineIterationOutcome::Progressed
    ));
    assert_eq!(executor.physical.load(Ordering::Acquire), 1);
    assert_eq!(engine.inner.sequences.read()[&id].generated_tokens.len(), 1);
    let audit = engine.inner.slo_controller.lock().last_audit.unwrap();
    assert!(audit.planner_budget_exhausted && !audit.budget_exhausted);
    assert!(audit.witness.is_none());
    cleanup(engine, session).await;
}

#[tokio::test(start_paused = true)]
async fn prepared_draft_cannot_publish_after_real_hard_deadline() {
    let (engine, _, executor) = completion_fixture(1).await;
    let (_, session) = request(&engine, 1, 2, None).await;
    admit(&engine, 1).await;
    let hint = ferrum_interfaces::BatchHint::simple(1);
    let (budget, prepared) = draft(&engine, &hint, Duration::from_millis(2)).await;
    tokio::time::advance(Duration::from_millis(2)).await;
    assert!(matches!(
        engine
            .inner
            .publish_completion_draft(
                &hint,
                &budget,
                prepared,
                CompletionOnlyReason::SearchInconclusive,
            )
            .unwrap(),
        SloIterationPlan::Idle
    ));
    assert_unpublished(&engine, &executor);
    assert!(!budget.finish_planning());
    let state = engine.inner.slo_controller.lock();
    assert!(state.completion_next.is_some() && state.retry.is_some());
    drop(state);
    cleanup(engine, session).await;
}

#[tokio::test(start_paused = true)]
async fn draft_rechecks_new_due_owner_even_when_that_owner_was_already_selected() {
    let (engine, _, executor) = completion_fixture(1).await;
    let (id, session) = request(&engine, 4, 8, None).await;
    admit(&engine, 1).await;
    let mut hint = ferrum_interfaces::BatchHint::simple(1);
    hint.max_tokens = 4;
    let (budget, prepared) = draft(&engine, &hint, Duration::from_secs(30)).await;
    {
        let mut sequences = engine.inner.sequences.write();
        let state = sequences
            .get_mut(&id)
            .unwrap()
            .time_admission
            .as_mut()
            .unwrap();
        let bound = state.recovery_service.bound().get();
        for _ in state.recovery_service.eligible_bypasses()..bound {
            state.recovery_service.bypass();
        }
        assert!(state.recovery_service.due());
    }
    // The owner becomes recovering as its real TTFT expires during planning.
    tokio::time::advance(Duration::from_secs(11)).await;
    assert!(matches!(
        engine
            .inner
            .publish_completion_draft(
                &hint,
                &budget,
                prepared,
                CompletionOnlyReason::SearchInconclusive,
            )
            .unwrap(),
        SloIterationPlan::Idle
    ));
    assert_unpublished(&engine, &executor);
    assert!(engine.inner.slo_controller.lock().retry.is_some());
    // The next capture must use the due owner's ordinary minimal quantum.
    let selected = selected(
        &engine,
        &executor,
        1,
        4,
        CompletionOnlyReason::SearchInconclusive,
    )
    .await;
    let flight = engine
        .inner
        .slo_controller
        .lock()
        .pending_execution
        .clone()
        .unwrap();
    assert!(matches!(flight.work.rows().next().unwrap().input,
        ExpectedWaveInput::Prefill { chunk } if chunk.tokens_to_process() == 1));
    assert!(matches!(
        engine
            .inner
            .execute_slo_controller_wave(selected)
            .await
            .unwrap(),
        EngineIterationOutcome::Progressed
    ));
    assert_eq!(
        engine.inner.sequences.read()[&id].prefill_tokens_processed,
        1
    );
    cleanup(engine, session).await;
}

#[tokio::test(start_paused = true)]
async fn draft_still_requires_fresh_full_queue_seal() {
    let (engine, scheduler, executor) = completion_fixture(1).await;
    let (_, session) = request(&engine, 1, 2, None).await;
    admit(&engine, 1).await;
    let hint = ferrum_interfaces::BatchHint::simple(1);
    let (budget, prepared) = draft(&engine, &hint, Duration::from_secs(30)).await;
    scheduler.record_external_capacity_release();
    assert!(matches!(
        engine
            .inner
            .publish_completion_draft(
                &hint,
                &budget,
                prepared,
                CompletionOnlyReason::SearchInconclusive,
            )
            .unwrap(),
        SloIterationPlan::Idle
    ));
    assert_unpublished(&engine, &executor);
    assert!(engine.inner.slo_controller.lock().retry.is_some());
    cleanup(engine, session).await;
}
