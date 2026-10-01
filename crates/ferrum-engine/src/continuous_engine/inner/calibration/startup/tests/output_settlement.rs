//! Real output accounts must settle before the private driver reuses slots.
use super::*;
use crate::continuous_engine::inner::calibration::geometry_projection::tests as geometry;
use crate::continuous_engine::output_flow_runtime::OutputReadiness;

async fn bounded<T>(future: impl std::future::Future<Output = T>) -> T {
    // Failure guard only; synchronization uses real completion/watch receipts.
    tokio::time::timeout(Duration::from_secs(3), future)
        .await
        .expect("private output settlement stalled")
}

#[tokio::test]
async fn startup_output_settlement_waits_for_original_held_lease_and_release_wake() {
    let (mut session, executor) = geometry::fixture(1).await;
    let declared = geometry::requests(&session, 1).pop().unwrap();
    let id = declared.request.id.clone();
    let output = session
        .add_request(
            declared.request,
            InferenceRequestContext::capture(),
            declared.contract,
        )
        .await
        .unwrap();
    bounded(session.startup_output_ready(&id)).await.unwrap();
    let grant = match session.engine.inner.sequences.read()[&id]
        .credited_output
        .as_ref()
        .unwrap()
        .port
        .try_take()
    {
        OutputReadiness::Ready(grant) => grant,
        _ => panic!("real output actor did not prepare a grant"),
    };
    let CreditedOutputSession { frames, completion } = output;
    drop(frames);
    session.step(CalibrationAction::Maintenance).await.unwrap();
    drop(bounded(completion).await.unwrap());
    let inner = Arc::clone(&session.engine.inner);
    let pool = inner.output_credit_pool.get().unwrap().as_ref().unwrap();
    assert!(inner.sequences.read().is_empty());
    assert_eq!(pool.snapshot().open_accounts, 0);
    assert_eq!(pool.snapshot().retained_accounts, 1);
    assert!(pool.snapshot().data_used.events > 0);
    let mut settlement = Box::pin(session.drain_startup_geometry());
    assert!(
        futures::poll!(settlement.as_mut()).is_pending(),
        "closed account is not drained while an original grant survives"
    );
    drop(grant);
    bounded(settlement).await.unwrap();
    let settled = pool.snapshot();
    assert_eq!(settled.retained_accounts, 0);
    assert_eq!(settled.data_used, Default::default());
    assert_eq!(settled.terminal_held, Default::default());
    assert_eq!(executor.entries.load(Ordering::Acquire), 0);
    session.shutdown().await.unwrap();
}

#[tokio::test]
async fn startup_output_settlement_reuses_two_private_cancelled_slots_without_admission_retry() {
    let (mut session, executor) = geometry::fixture(2).await;
    // Geometry width bounds execution rows, not the lazy output ledger. Fix
    // its real configured account capacity before the first request opens it.
    let fresh = Arc::get_mut(&mut session.engine.inner).expect("exclusive private fixture");
    assert!(fresh.output_credit_pool.get().is_none());
    fresh.config.scheduler.max_running_requests = 2;
    let inner = Arc::clone(&session.engine.inner);
    for _ in 0..2 {
        let mut outputs = Vec::new();
        for declared in geometry::requests(&session, 2) {
            outputs.push(
                session
                    .add_request(
                        declared.request,
                        InferenceRequestContext::capture(),
                        declared.contract,
                    )
                    .await
                    .unwrap(),
            );
        }
        let pool = inner.output_credit_pool.get().unwrap().as_ref().unwrap();
        assert_eq!(pool.snapshot().limits.max_open_accounts, 2);
        assert_eq!(pool.snapshot().open_accounts, 2);
        drop(outputs);
        bounded(session.drain_startup_geometry()).await.unwrap();
        assert!(inner.sequences.read().is_empty());
        assert_eq!(inner.scheduler.waiting_count(), 0);
        assert_eq!(inner.scheduler.active_count(), 0);
        assert_eq!(pool.snapshot().retained_accounts, 0);
    }
    assert_eq!(executor.entries.load(Ordering::Acquire), 0);
    assert_eq!(executor.physical.load(Ordering::Acquire), 0);
    session.shutdown().await.unwrap();
}
