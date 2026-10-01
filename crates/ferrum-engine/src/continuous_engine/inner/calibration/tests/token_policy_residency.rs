use super::*;
use ferrum_interfaces::model_executor::{
    TokenPolicyResidencyInvalidation as Outcome, TokenPolicyResidencyUnavailable as Busy,
};
use ferrum_interfaces::ModelExecutor;

#[tokio::test]
async fn calibration_token_policy_invalidation_empty_unsupported_is_not_a_success_receipt() {
    let (mut session, executor) = fixture(1).await;
    assert_eq!(
        session.invalidate_token_policy_residency().await,
        Outcome::Unsupported
    );
    assert_eq!(executor.entries.load(Ordering::Acquire), 0);
    assert_eq!(executor.physical.load(Ordering::Acquire), 0);
    bounded(session.shutdown()).await.unwrap();
}

#[tokio::test]
async fn calibration_token_policy_invalidation_refuses_live_owner_before_executor() {
    let (mut session, executor) = fixture(1).await;
    let (id, output) = add(&mut session, 4).await;
    let before = frontier(&session, &id);
    assert_eq!(
        session.invalidate_token_policy_residency().await,
        Outcome::Unavailable {
            reason: Busy::ActiveRequests
        }
    );
    let after = frontier(&session, &id);
    assert_eq!(before.owner_incarnation(), after.owner_incarnation());
    assert_eq!(before.work_generation(), after.work_generation());
    assert_eq!(before.prefill_progress(), after.prefill_progress());
    assert_eq!(executor.entries.load(Ordering::Acquire), 0);
    drop(output);
    bounded(session.shutdown()).await.unwrap();
}

#[tokio::test]
async fn calibration_token_policy_invalidation_busy_or_indeterminate_never_clears() {
    let (mut session, executor) = fixture(1).await;
    let inner = Arc::clone(&session.engine.inner);
    let iteration = inner.iteration_lock.lock().await;
    assert_eq!(
        session.invalidate_token_policy_residency().await,
        Outcome::Unavailable {
            reason: Busy::IterationBusy
        }
    );
    drop(iteration);
    session.indeterminate = true;
    assert_eq!(
        session.invalidate_token_policy_residency().await,
        Outcome::Unavailable {
            reason: Busy::SessionIndeterminate
        }
    );
    assert_eq!(executor.physical.load(Ordering::Acquire), 0);
    session.indeterminate = false;
    bounded(session.shutdown()).await.unwrap();
}

#[tokio::test]
async fn native_cpu_token_policy_invalidation_requires_cancelled_admission() {
    let (mut session, executor) = super::super::geometry_projection::tests::fixture(1).await;
    let empty = Outcome::Cleared { cleared_entries: 0 };
    assert_eq!(session.invalidate_token_policy_residency().await, empty);
    let (id, output) = add(&mut session, 1).await;
    admit(&mut session).await;
    // Call the backend directly: its own admission registry must reject this
    // root even though it has no KV handle and no cost view has been requested.
    assert_eq!(
        executor.calibration_invalidate_token_policy_residency(),
        Outcome::Unavailable {
            reason: Busy::ActiveRequests
        }
    );
    assert_eq!(frontier(&session, &id).prefill_progress(), Some((0, 1)));
    drop(output);
    bounded(session.drain_startup_geometry()).await.unwrap();
    assert_eq!(session.invalidate_token_policy_residency().await, empty);
    assert_eq!(executor.entries.load(Ordering::Acquire), 0);
    assert_eq!(executor.physical.load(Ordering::Acquire), 0);
    assert_eq!(executor.completion_calls.load(Ordering::Acquire), 0);
    bounded(session.shutdown()).await.unwrap();
}

#[tokio::test]
async fn native_cpu_token_policy_invalidation_rejects_parked_execution_after_admission_release() {
    let (mut session, executor) = super::super::geometry_projection::tests::fixture(1).await;
    let (id, output) = add(&mut session, 1).await;
    admit(&mut session).await;
    let row = frontier(&session, &id)
        .prefill_work(NonZeroU32::MIN)
        .unwrap();
    executor.park.store(true, Ordering::Release);
    let mut executing = Box::pin(wave(&mut session, &executor, vec![row]));
    bounded(async {
        tokio::select! {
            report = executing.as_mut() => panic!("parked executor completed early: {report:?}"),
            () = executor.entered.notified() => {}
        }
    })
    .await;
    // Releasing prefill admission alone cannot retire an already entered call.
    assert!(executor.cancel_prefill_admission(&id));
    assert_eq!(
        executor.calibration_invalidate_token_policy_residency(),
        Outcome::Unavailable {
            reason: Busy::CompletionWork
        }
    );
    executor.replan_before_encode.store(true, Ordering::Release);
    executor.resume.notify_one();
    let report = bounded(executing).await;
    assert_eq!(report.submission, CalibrationSubmissionState::NotSubmitted);
    assert_eq!(executor.physical.load(Ordering::Acquire), 0);
    drop(output);
    bounded(session.drain_startup_geometry()).await.unwrap();
    assert_eq!(
        session.invalidate_token_policy_residency().await,
        Outcome::Cleared { cleared_entries: 0 }
    );
    bounded(session.shutdown()).await.unwrap();
}
