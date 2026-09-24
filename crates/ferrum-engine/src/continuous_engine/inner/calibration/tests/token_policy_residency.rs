use super::*;
use ferrum_interfaces::model_executor::{
    TokenPolicyResidencyInvalidation as Outcome, TokenPolicyResidencyUnavailable as Busy,
};

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
