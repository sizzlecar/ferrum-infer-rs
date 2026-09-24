//! Actual scheduler/output ownership with a controlled PlanRuntime executor.
//! Native post-encode rejection is covered by the Metal/core contract tests;
//! this fixture never manufactures their private rollback receipt.
use super::*;

mod unified_execution;
#[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
mod unified_execution_metal;
use ferrum_interfaces::{
    engine::LlmInferenceEngine, execution_cost::*, model_executor::*, output_flow::*,
    scheduler::Scheduler, InferenceRequestContext,
};
use ferrum_scheduler::implementations::continuous::ContinuousBatchScheduler;
use ferrum_testkit::{MockKvCacheHandle, MockModelExecutor, MockTensorFactory};
use futures::{FutureExt, StreamExt};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

mod completion;
pub(in crate::continuous_engine) mod fixture;
pub(super) mod prefill;
mod publication_retry;
mod recompute_boundary;
mod recovery;
mod recurrent_state;
mod reference_projection;
mod timing_metrics;
use fixture::*;

pub(super) async fn ready(engine: &ContinuousBatchEngine, id: &RequestId) {
    use crate::continuous_engine::output_flow_runtime::OutputReadinessState;
    let mut changed = engine.inner.sequences.read()[id]
        .credited_output
        .as_ref()
        .unwrap()
        .port
        .subscribe();
    bounded(async {
        loop {
            if engine.inner.sequences.read()[id]
                .credited_output
                .as_ref()
                .unwrap()
                .port
                .readiness()
                == OutputReadinessState::Ready
            {
                break;
            }
            changed.changed().await.unwrap();
        }
    })
    .await;
}

pub(super) async fn bounded<T>(future: impl std::future::Future<Output = T>) -> T {
    tokio::time::timeout(Duration::from_secs(3), future)
        .await
        .expect("controller fixture made no progress")
}

async fn installed(
    engine: &ContinuousBatchEngine,
    scheduler: &ContinuousBatchScheduler,
    executor: &ControlledExecutor,
    expires: Duration,
) -> (
    RequestId,
    CreditedOutputSession,
    owner::PreparedControllerWave,
) {
    let mut request =
        ferrum_types::InferenceRequest::new("test", engine.inner.config.model.model_id.clone());
    request.stream = true;
    request.sampling_params.max_tokens = 4;
    request.sampling_params.temperature = 0.0;
    request.sampling_params.repetition_penalty = 1.0;
    let id = request.id.clone();
    let mut session = engine
        .infer_credited_stream(
            request,
            InferenceRequestContext::from_ingress(slo_clock_now()),
            Arc::new(OutputProjectionContract::cli_text()),
        )
        .await
        .unwrap();
    ready(engine, &id).await;
    let prefill = scheduler
        .next_batch(ferrum_interfaces::BatchHint::simple(1))
        .await
        .unwrap();
    engine.inner.process_batch(&prefill).await.unwrap();
    // Release the real first output frame before obtaining the decode grant.
    drop(bounded(session.frames.next()).await.unwrap());
    ready(engine, &id).await;
    assert_eq!(engine.inner.sequences.read()[&id].generated_tokens.len(), 1);

    let hint = ferrum_interfaces::BatchHint::simple(1);
    let captured = prefill::captured_with_hint(engine, executor, &hint).await;
    let row = &captured.fences[0];
    let scheduler_row = &captured.queue.requests()[0];
    assert_eq!(scheduler_row.committed_output_tokens, 1);
    assert_eq!(scheduler_row.computed_tokens, row.context);
    assert_eq!(scheduler_row.resident_tokens, row.context);
    assert_eq!(scheduler_row.scheduled_tokens, row.context);
    let expected = ExpectedExecutionCostWave::new(
        captured.route.clone(),
        canonical(u32::try_from(row.context).unwrap()),
        vec![ExpectedWaveParticipant {
            participant_index: 0,
            request_id: id.clone(),
            input: ExpectedWaveInput::Decode {
                cache_id: row.cache_id.clone().unwrap(),
            },
            host: CostObservationParticipant {
                request_id: id.clone(),
                owner_incarnation: row.incarnation,
                work_generation: row.generation,
                input_index: 0,
                output_policy_signature: Some(host_history_cost_signature(
                    captured.snapshot.requests[0].output_policy_signature,
                    row.generated as u64,
                )),
                host_features: row.host_features,
            },
        }],
    )
    .unwrap();
    let mut availability = Vec::new();
    let epochs = engine
        .inner
        .model_executor
        .write_execution_capacity_snapshot(&mut availability)
        .unwrap()
        .unwrap();
    let publication = scheduler
        .try_select_planned_wave(
            &captured.queue,
            &[PlanningWorkSelection {
                key: row.key.clone(),
                action: PlanningWorkAction::Decode,
            }],
            &hint,
            AdmissionWakeSnapshot::new(
                AdmissionWakeEpochs::new(
                    epochs.coordinator_id,
                    epochs.release_epoch,
                    epochs.capacity_epoch,
                    0,
                ),
                &availability,
            ),
        )
        .unwrap();
    let PlanningSelectionOutcome::Published { batch, receipt } = publication else {
        panic!("exact publication")
    };
    assert!(engine.inner.reserve_batch_output(&batch).unwrap());
    captured.budget.finish_planning();
    let expected = ExpectedExecutionWave::from_cost_witness(expected, |id| {
        captured
            .fences
            .iter()
            .find(|fence| fence.key.request_id == *id)
            .map(|fence| &fence.logits_policy)
    })
    .unwrap();
    let work = owner::ControllerWork {
        batch,
        expected,
        timing: owner::ControllerTimingCommitment::Witness {
            admission: None,
            valid_until: slo_clock_now().checked_add(expires).unwrap(),
            model_version: captured.snapshot.cost_model_version,
        },
        proof: captured.into_safety(),
    };
    let prepared = engine.inner.install_controller_wave(work, receipt).unwrap();
    (id, session, prepared)
}

fn scheduled(engine: &ContinuousBatchEngine) -> usize {
    let mut sources = Vec::new();
    let epochs = engine
        .inner
        .model_executor
        .write_execution_capacity_snapshot(&mut sources)
        .unwrap()
        .unwrap();
    engine
        .inner
        .scheduler
        .planning_state(
            NonZeroUsize::new(16).unwrap(),
            AdmissionWakeSnapshot::new(
                AdmissionWakeEpochs::new(
                    epochs.coordinator_id,
                    epochs.release_epoch,
                    epochs.capacity_epoch,
                    0,
                ),
                &sources,
            ),
        )
        .unwrap()
        .requests()
        .iter()
        .map(|row| row.scheduled_tokens - row.computed_tokens)
        .sum()
}

#[tokio::test]
async fn controller_ready_drop_returns_original_grant_and_publication() {
    let (engine, scheduler, executor) = fixture().await;
    let (id, session, prepared) =
        installed(&engine, &scheduler, &executor, Duration::from_secs(1)).await;
    assert_eq!(scheduled(&engine), 1);
    let used = pool(&engine).snapshot().data_used;
    drop(prepared);
    engine.inner.drain_slo_execution().await.unwrap();
    ready(&engine, &id).await;
    assert_eq!(scheduled(&engine), 0);
    assert_eq!(pool(&engine).snapshot().data_used, used);
    assert_eq!(executor.physical.load(Ordering::Acquire), 0);
    let audit = engine.inner.slo_controller.lock().last_audit.unwrap();
    assert_eq!(audit.outcome, "withdrawn");
    assert_eq!(
        audit.stages[ControllerStage::ExecutorAwait as usize].calls,
        0
    );
    assert_eq!(
        audit.stages[ControllerStage::Reconciliation as usize].calls,
        1
    );
    cleanup(engine, session).await;
}

#[tokio::test]
async fn controller_shutdown_wins_ready_claim_and_late_execute_is_idle() {
    let (engine, scheduler, executor) = fixture().await;
    let (id, session, prepared) =
        installed(&engine, &scheduler, &executor, Duration::from_secs(1)).await;
    engine.inner.signal_shutdown();
    engine.inner.drain_slo_execution().await.unwrap();
    assert!(matches!(
        engine
            .inner
            .execute_slo_controller_wave(prepared)
            .await
            .unwrap(),
        EngineIterationOutcome::Idle
    ));
    assert_eq!(scheduled(&engine), 0);
    assert!(engine.inner.sequences.read()[&id]
        .credited_output
        .as_ref()
        .unwrap()
        .grant
        .is_none());
    assert_eq!(executor.physical.load(Ordering::Acquire), 0);
    cleanup(engine, session).await;
}

#[tokio::test(start_paused = true)]
async fn controller_iteration_lock_wait_expiry_never_submits() {
    let (engine, scheduler, executor) = fixture().await;
    let (id, session, prepared) =
        installed(&engine, &scheduler, &executor, Duration::from_millis(10)).await;
    let held = engine.inner.iteration_lock.lock().await;
    let mut waiting = Box::pin(engine.inner.execute_slo_controller_wave(prepared));
    assert!(waiting.as_mut().now_or_never().is_none());
    tokio::time::advance(Duration::from_millis(11)).await;
    drop(held);
    assert!(matches!(
        bounded(waiting).await.unwrap(),
        EngineIterationOutcome::Idle
    ));
    assert_eq!(executor.entries.load(Ordering::Acquire), 0);
    assert_eq!(executor.physical.load(Ordering::Acquire), 0);
    assert_eq!(scheduled(&engine), 0);
    assert_eq!(engine.inner.sequences.read()[&id].generated_tokens.len(), 1);
    cleanup(engine, session).await;
}

#[tokio::test]
async fn controller_dispatch_survives_caller_cancel_and_commits_only_once() {
    let (engine, scheduler, executor) = fixture().await;
    let (id, session, prepared) =
        installed(&engine, &scheduler, &executor, Duration::from_secs(1)).await;
    executor.park.store(true, Ordering::Release);
    let mut waiting = Box::pin(engine.inner.execute_slo_controller_wave(prepared));
    assert!(waiting.as_mut().now_or_never().is_none());
    bounded(executor.entered.notified()).await;
    drop(waiting);
    assert_eq!(scheduled(&engine), 1);
    assert!(engine.inner.sequences.read()[&id]
        .credited_output
        .as_ref()
        .unwrap()
        .grant
        .is_some());
    executor.resume.notify_one();
    assert!(matches!(
        bounded(engine.inner.drain_slo_execution()).await.unwrap(),
        Some(EngineIterationOutcome::Progressed)
    ));
    assert_eq!(executor.entries.load(Ordering::Acquire), 1);
    assert_eq!(executor.physical.load(Ordering::Acquire), 1);
    assert_eq!(engine.inner.sequences.read()[&id].generated_tokens.len(), 2);
    assert_eq!(scheduled(&engine), 0);
    assert!(engine.inner.drain_slo_execution().await.unwrap().is_none());
    cleanup(engine, session).await;
}

#[tokio::test]
async fn controller_shutdown_after_dispatch_joins_instead_of_refunding_live_grant() {
    let (engine, scheduler, executor) = fixture().await;
    let (id, session, prepared) =
        installed(&engine, &scheduler, &executor, Duration::from_secs(1)).await;
    executor.park.store(true, Ordering::Release);
    let mut waiting = Box::pin(engine.inner.execute_slo_controller_wave(prepared));
    assert!(waiting.as_mut().now_or_never().is_none());
    bounded(executor.entered.notified()).await;
    drop(waiting);
    engine.inner.signal_shutdown();
    let mut drain = Box::pin(engine.inner.drain_slo_execution());
    assert!(drain.as_mut().now_or_never().is_none());
    assert_eq!(scheduled(&engine), 1);
    assert!(engine.inner.sequences.read()[&id]
        .credited_output
        .as_ref()
        .unwrap()
        .grant
        .is_some());
    executor.resume.notify_one();
    bounded(drain).await.unwrap();
    assert_eq!(executor.physical.load(Ordering::Acquire), 0);
    assert_eq!(scheduled(&engine), 0);
    cleanup(engine, session).await;
}

#[tokio::test]
async fn controller_submitted_error_is_terminal_not_a_replayable_publication() {
    let (engine, scheduler, executor) = fixture().await;
    let (id, session, prepared) =
        installed(&engine, &scheduler, &executor, Duration::from_secs(1)).await;
    executor.fail_after_submit.store(true, Ordering::Release);
    assert!(engine
        .inner
        .execute_slo_controller_wave(prepared)
        .await
        .is_err());
    assert_eq!(executor.physical.load(Ordering::Acquire), 1);
    assert!(!engine.inner.sequences.read().contains_key(&id));
    assert!(engine.inner.slo_controller.lock().pending_release.is_none());
    assert!(engine.inner.drain_slo_execution().await.unwrap().is_none());
    let audit = engine.inner.slo_controller.lock().last_audit.unwrap();
    assert_eq!(audit.outcome, "failed");
    assert_eq!(
        audit.stages[ControllerStage::ExecutorAwait as usize].calls,
        1
    );
    cleanup(engine, session).await;
}

#[tokio::test(start_paused = true)]
async fn controller_executor_delay_does_not_reopen_or_extend_sync_planning_budget() {
    let (engine, scheduler, executor) = fixture().await;
    let (id, session, prepared) =
        installed(&engine, &scheduler, &executor, Duration::from_secs(60)).await;
    executor.park.store(true, Ordering::Release);
    let mut waiting = Box::pin(engine.inner.execute_slo_controller_wave(prepared));
    assert!(waiting.as_mut().now_or_never().is_none());
    bounded(executor.entered.notified()).await;
    // Longer than this fixture's 30s planning budget, still inside the real
    // witness TTL. After-encode guard must not use the ended planning budget.
    tokio::time::advance(Duration::from_secs(31)).await;
    executor.resume.notify_one();
    assert!(matches!(
        bounded(waiting).await.unwrap(),
        EngineIterationOutcome::Progressed
    ));
    assert_eq!(executor.physical.load(Ordering::Acquire), 1);
    assert_eq!(engine.inner.sequences.read()[&id].generated_tokens.len(), 2);
    let audit = engine.inner.slo_controller.lock().last_audit.unwrap();
    assert_eq!(audit.outcome, "submitted");
    assert!(!audit.budget_exhausted);
    assert!(audit.planning_wall_ns < audit.budget_ns);
    assert!(audit.stages[ControllerStage::ExecutorAwait as usize].wall_ns >= 31_000_000_000);
    assert!(audit.transaction_wall_ns >= 31_000_000_000);
    assert!(audit.stages[ControllerStage::HostGuard as usize].calls >= 2);
    cleanup(engine, session).await;
}

#[tokio::test(start_paused = true)]
async fn controller_early_unknown_still_emits_capture_and_total_audit() {
    use ferrum_interfaces::engine::InferenceEngine;
    let (engine, _, executor) = fixture().await;
    assert!(matches!(
        engine
            .inner
            .prepare_slo_controller(&ferrum_interfaces::BatchHint::simple(1))
            .unwrap(),
        SloIterationPlan::Legacy
    ));
    let audit = engine.inner.slo_controller.lock().last_audit.unwrap();
    assert_eq!(audit.outcome, "observed");
    assert_eq!(audit.decision, "unknown");
    assert_eq!(audit.reason, "no_work");
    assert_eq!(audit.stages[ControllerStage::Capture as usize].calls, 1);
    assert_eq!(
        audit.stages[ControllerStage::SearchReplay as usize].calls,
        0
    );
    assert_eq!(
        audit.stages[ControllerStage::ExecutorAwait as usize].calls,
        0
    );
    assert_eq!(executor.entries.load(Ordering::Acquire), 0);
    engine.shutdown().await.unwrap();
}

#[tokio::test(start_paused = true)]
async fn controller_publication_cannot_reset_an_expired_capture_budget() {
    let (engine, scheduler, executor) = fixture().await;
    let (id, session, prepared) =
        installed(&engine, &scheduler, &executor, Duration::from_secs(1)).await;
    drop(prepared);
    engine.inner.drain_slo_execution().await.unwrap();
    ready(&engine, &id).await;
    let hint = ferrum_interfaces::BatchHint::simple(1);
    let budget = ControllerBudget::new(slo_clock_now(), Duration::from_millis(1)).unwrap();
    let captured = engine
        .inner
        .capture_slo_controller_snapshot(&hint, budget.clone())
        .unwrap();
    let selected = SelectedWave {
        protection: None,
        candidate: WaveCandidate {
            work: vec![CandidateWork {
                key: captured.snapshot.requests[0].key.clone(),
                action: WaveAction::Decode,
            }],
            execution_shape: PlanningShapeDomain::Exact(
                canonical_cost_shape(&canonical(captured.fences[0].context as u32)).unwrap(),
            ),
            based_on_generation: captured.snapshot.generation,
            cost_model_version: captured.snapshot.cost_model_version,
        },
        predicted_wall_ns: 1,
        planning_observed_at_ns: captured.snapshot.observed_at_ns,
        snapshot_observed_at_ns: captured.snapshot.observed_at_ns,
        snapshot_generation: captured.snapshot.generation,
        cost_model_version: captured.snapshot.cost_model_version,
        witness_valid_for_ns: 1_000_000_000,
    };
    tokio::time::advance(Duration::from_millis(1)).await;
    assert!(matches!(
        engine
            .inner
            .prepare_slo_controller_wave(captured, selected, &hint)
            .unwrap(),
        SloIterationPlan::Idle
    ));
    assert_eq!(scheduled(&engine), 0);
    assert_eq!(executor.entries.load(Ordering::Acquire), 0);
    assert!(engine.inner.sequences.read()[&id]
        .credited_output
        .as_ref()
        .unwrap()
        .grant
        .is_none());
    assert!(!budget.finish_planning());
    engine.inner.finish_controller_audit(&budget, "idle");
    assert!(
        engine
            .inner
            .slo_controller
            .lock()
            .last_audit
            .unwrap()
            .budget_exhausted
    );
    cleanup(engine, session).await;
}

#[tokio::test]
async fn controller_shutdown_stale_ready_read_loses_cas_and_joins_entered_wave() {
    let (engine, scheduler, executor) = fixture().await;
    let (id, session, prepared) =
        installed(&engine, &scheduler, &executor, Duration::from_secs(1)).await;
    let flight = engine
        .inner
        .slo_controller
        .lock()
        .pending_execution
        .clone()
        .unwrap();
    let observed = Arc::new(Notify::new());
    let resume = Arc::new(Notify::new());
    flight.pause_withdrawal_for_test(observed.clone(), resume.clone());
    engine.inner.signal_shutdown();
    let inner = engine.inner.clone();
    let drain = tokio::spawn(async move { inner.drain_slo_execution().await });
    bounded(observed.notified()).await; // drain already observed READY

    let held = engine.inner.iteration_lock.lock().await;
    let mut execution = Box::pin(engine.inner.execute_slo_controller_wave(prepared));
    assert!(execution.as_mut().now_or_never().is_none()); // dispatch owns CAS
    drop(execution); // remove its join waiter, preserving the actual task
    resume.notify_one();
    // Wait until drain has performed its stale-READY CAS and joined the task.
    bounded(flight.withdrawal_lost_for_test()).await;
    assert!(!drain.is_finished());
    assert_eq!(scheduled(&engine), 1);
    assert!(engine.inner.sequences.read()[&id]
        .credited_output
        .as_ref()
        .unwrap()
        .grant
        .is_some());
    drop(held);
    bounded(drain).await.unwrap().unwrap();
    assert_eq!(executor.physical.load(Ordering::Acquire), 0);
    assert_eq!(scheduled(&engine), 0);
    cleanup(engine, session).await;
}

#[tokio::test]
async fn controller_join_error_cleanup_survives_cancel_while_waiting_iteration_lock() {
    let (engine, scheduler, executor) = fixture().await;
    let (id, session, prepared) =
        installed(&engine, &scheduler, &executor, Duration::from_secs(1)).await;
    executor.park.store(true, Ordering::Release);
    executor.panic_before_submit.store(true, Ordering::Release);
    let flight = engine
        .inner
        .slo_controller
        .lock()
        .pending_execution
        .clone()
        .unwrap();
    let inner = engine.inner.clone();
    let waiter = tokio::spawn(async move { inner.execute_slo_controller_wave(prepared).await });
    bounded(executor.entered.notified()).await;

    // Queue ahead of the cleanup task while the physical task holds this lock.
    let mut queued_lock = Box::pin(engine.inner.iteration_lock.lock());
    assert!(queued_lock.as_mut().now_or_never().is_none());
    executor.resume.notify_one();
    let held = bounded(queued_lock).await;
    bounded(async {
        while !flight.cleanup_pending_for_test() {
            engine.inner.work_notify.notified().await;
        }
    })
    .await;
    waiter.abort();
    assert!(matches!(waiter.await, Err(error) if error.is_cancelled()));
    assert!(engine
        .inner
        .slo_controller
        .lock()
        .pending_execution
        .is_some());
    assert!(engine.inner.sequences.read().contains_key(&id));
    drop(held);
    assert!(bounded(engine.inner.drain_slo_execution()).await.is_err());
    assert!(!engine.inner.sequences.read().contains_key(&id));
    assert!(engine
        .inner
        .slo_controller
        .lock()
        .pending_execution
        .is_none());
    assert!(engine.inner.slo_controller.lock().pending_release.is_none());
    assert_eq!(executor.physical.load(Ordering::Acquire), 0);
    cleanup(engine, session).await;
}
