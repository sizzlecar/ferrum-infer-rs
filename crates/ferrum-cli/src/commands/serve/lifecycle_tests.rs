use super::lifecycle::ServeTaskObserver;
use ferrum_journal::{JsonlJournalOpenMode, JsonlJournalOwner};
use ferrum_types::FerrumError;
use serde_json::json;

#[tokio::test]
async fn serve_lifecycle_early_timeout_report_keeps_original_scope_until_same_task_joins() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("serve.jsonl");
    let mut owner =
        JsonlJournalOwner::open(&path, JsonlJournalOpenMode::Append, Default::default()).unwrap();
    let mut scope = owner.producer_scope().unwrap();
    let producer = scope.producer::<serde_json::Value>();
    let (release, wait) = tokio::sync::oneshot::channel();
    let mut observer = ServeTaskObserver::spawn(move |_cancel, report| async move {
        producer
            .try_enqueue(json!({"before_timeout":true}))
            .map_err(|e| FerrumError::io(e.to_string()))?;
        report
            .send(Err(FerrumError::internal(
                "controlled HTTP drain remains pending",
            )
            .into()))
            .unwrap();
        wait.await
            .map_err(|e| FerrumError::internal(e.to_string()))?;
        producer
            .try_enqueue(json!({"after_actual_drain":true}))
            .map_err(|e| FerrumError::io(e.to_string()))?;
        scope
            .finish()
            .result()
            .map_err(|e| FerrumError::io(e.to_string()).into())
    });
    let error = observer.result().await.unwrap_err();
    assert!(error
        .to_string()
        .contains("controlled HTTP drain remains pending"));
    assert!(
        owner.close().is_err(),
        "observing a timeout must not retire the task's scope"
    );
    assert!(!owner.delivery_stats().worker_finished);
    // Re-observing the first report must wait on the same original task,
    // rather than polling the already consumed oneshot again.
    {
        let second = observer.result();
        tokio::pin!(second);
        futures::future::poll_fn(|cx| {
            use std::future::Future;
            assert!(second.as_mut().poll(cx).is_pending());
            std::task::Poll::Ready(())
        })
        .await;
    }
    release.send(()).unwrap();
    observer.result().await.unwrap();
    observer.join().await.unwrap();
    let final_receipt = owner.close().unwrap();
    final_receipt.result().unwrap();
    assert!(final_receipt.worker_joined);
    let rows: Vec<serde_json::Value> = std::fs::read_to_string(path)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    assert_eq!(
        rows,
        [
            json!({"before_timeout":true}),
            json!({"after_actual_drain":true})
        ]
    );
}

#[tokio::test]
async fn serve_lifecycle_dropped_observer_requests_same_task_cleanup_without_retiring_scope() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("cancelled.jsonl");
    let mut owner =
        JsonlJournalOwner::open(&path, JsonlJournalOpenMode::Append, Default::default()).unwrap();
    let mut scope = owner.producer_scope().unwrap();
    let (cancelled, observed_cancel) = tokio::sync::oneshot::channel();
    let (release, wait) = tokio::sync::oneshot::channel();
    let (finished, joined_scope) = tokio::sync::oneshot::channel();
    let observer = ServeTaskObserver::spawn(move |cancel, _report| async move {
        assert!(cancel.await.is_err());
        cancelled.send(()).unwrap();
        wait.await.unwrap();
        scope
            .producer::<serde_json::Value>()
            .try_enqueue(json!({"original_cleanup":true}))
            .unwrap();
        scope.finish().result().unwrap();
        finished.send(()).unwrap();
        Ok(())
    });
    drop(observer);
    observed_cancel.await.unwrap();
    assert!(owner.close().is_err());
    assert!(!owner.delivery_stats().worker_finished);
    release.send(()).unwrap();
    joined_scope.await.unwrap();
    let receipt = owner.close().unwrap();
    receipt.result().unwrap();
    assert!(receipt.worker_joined);
    assert!(std::fs::read_to_string(path)
        .unwrap()
        .contains("original_cleanup"));
}

#[tokio::test]
async fn serve_off_profile_assembly_shares_typed_scheduler_destination() {
    use crate::observability_product::{
        ProductJournalSession, ProductObservabilityConfig, ProfileDetailArg,
    };
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("profile.jsonl");
    let product = ProductObservabilityConfig::new(
        ferrum_types::ProfileEntrypoint::Serve,
        "profile",
        Some(&path),
        ProfileDetailArg::Off,
        None,
        Some(&path),
        None,
        1.0,
    );
    let mut engine = ferrum_types::EngineConfig::default();
    engine.runtime.profile_jsonl = Some(path.clone());
    engine.runtime.scheduler_trace_jsonl = Some(directory.path().join(".").join("profile.jsonl"));
    let journals = ProductJournalSession::prepare(&product, &engine);
    journals.result().unwrap();
    assert_eq!(
        engine.runtime.profile_detail,
        ferrum_types::ObservabilityProfileDetail::Off
    );
    assert!(product.unified_product_profile_enabled());
    journals.finish().await.unwrap();
    assert!(journals.is_closed());
    assert!(path.exists());
}

#[tokio::test]
async fn serve_lifecycle_real_deadline_keeps_same_cleanup_and_timeout_after_late_success() {
    actual_shutdown_deadline(false, false).await;
}

#[tokio::test]
async fn serve_lifecycle_post_http_cleanup_deadline_preserves_timeout_and_late_failure() {
    actual_shutdown_deadline(true, true).await;
}

async fn actual_shutdown_deadline(http_already_drained: bool, late_failure: bool) {
    use super::lifecycle::ShutdownObservation;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("deadline.jsonl");
    let mut owner =
        JsonlJournalOwner::open(&path, JsonlJournalOpenMode::Append, Default::default()).unwrap();
    let mut scope = owner.producer_scope().unwrap();
    let producer = scope.producer::<serde_json::Value>();
    let calls = Arc::new(AtomicUsize::new(0));
    let finished = Arc::new(AtomicUsize::new(0));
    let run_calls = calls.clone();
    let run_finished = finished.clone();
    let (release, wait) = tokio::sync::oneshot::channel();
    let mut observer = ServeTaskObserver::spawn(move |_cancel, report| async move {
        // A zero observation deadline makes the pending-future boundary
        // deterministic without sleep, while exercising the actual timer.
        let mut shutdown = ShutdownObservation::new(std::time::Duration::ZERO, report);
        if http_already_drained {
            shutdown.wait(std::future::ready(())).await;
        }
        let actual = shutdown
            .wait(async {
                run_calls.fetch_add(1, Ordering::SeqCst);
                producer
                    .try_enqueue(json!({"before_pending_cleanup": true}))
                    .unwrap();
                wait.await.unwrap();
                producer
                    .try_enqueue(json!({"actual_cleanup_finished": true}))
                    .unwrap();
                scope.finish().result().unwrap();
                run_finished.fetch_add(1, Ordering::SeqCst);
                if late_failure {
                    Err(FerrumError::io("controlled late cleanup failure").into())
                } else {
                    Ok(())
                }
            })
            .await;
        assert_eq!(
            actual.is_err(),
            late_failure,
            "the original cleanup result is still observed"
        );
        shutdown.finish(actual)
    });
    let first = observer.result().await.unwrap_err();
    assert!(first
        .to_string()
        .contains("original cleanup remains pending"));
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert_eq!(finished.load(Ordering::SeqCst), 0);
    assert!(
        owner.close().is_err(),
        "a timeout cannot retire the original producer"
    );
    assert!(!owner.delivery_stats().worker_finished);
    {
        let second = observer.result();
        tokio::pin!(second);
        futures::future::poll_fn(|cx| {
            use std::future::Future;
            assert!(second.as_mut().poll(cx).is_pending());
            std::task::Poll::Ready(())
        })
        .await;
    }
    release.send(()).unwrap();
    let terminal = observer.result().await.unwrap_err();
    assert_eq!(
        terminal.to_string(),
        first.to_string(),
        "actual completion cannot erase the reported timeout"
    );
    assert_eq!(
        observer.join().await.unwrap_err().to_string(),
        first.to_string()
    );
    assert_eq!(
        calls.load(Ordering::SeqCst),
        1,
        "no replacement cleanup attempt"
    );
    assert_eq!(finished.load(Ordering::SeqCst), 1);
    let receipt = owner.close().unwrap();
    receipt.result().unwrap();
    assert!(receipt.worker_joined);
    let rows: Vec<serde_json::Value> = std::fs::read_to_string(path)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    assert_eq!(
        rows,
        [
            json!({"before_pending_cleanup": true}),
            json!({"actual_cleanup_finished": true})
        ]
    );
}

#[tokio::test]
async fn serve_startup_timeout_preserves_original_static_failure_and_recovery_observer() {
    use super::lifecycle::ShutdownObservation;
    use ferrum_execution::ExecutorShutdownPhase;
    use std::time::Duration;
    let (error, probe) =
        ferrum_execution::driver::test_support::shutdown::failed_initialization_build_error(1);
    let cause = error.cause().to_string();
    let (release, pending) = tokio::sync::oneshot::channel();
    let mut task = ServeTaskObserver::spawn(move |_cancel, report| async move {
        let mut shutdown = ShutdownObservation::new(Duration::ZERO, report);
        shutdown.record_failure(&error);
        shutdown
            .wait(async {
                let _ = pending.await;
            })
            .await;
        shutdown.finish(Err(error))
    });
    let reported = task.result().await.unwrap_err();
    assert_eq!(reported.cause().to_string(), cause);
    let mut cleanup = reported
        .cleanup()
        .expect("timeout must retain original construction observation")
        .clone();
    tokio::time::timeout(Duration::from_secs(5), async {
        while cleanup.snapshot().phase() != ExecutorShutdownPhase::Retained {
            cleanup.changed().await;
        }
    })
    .await
    .unwrap();
    assert_eq!(probe.snapshot().buffer_drops, 0);
    drop(reported);
    cleanup.retry();
    tokio::time::timeout(Duration::from_secs(5), async {
        while cleanup.snapshot().phase() != ExecutorShutdownPhase::Retired {
            cleanup.changed().await;
        }
    })
    .await
    .unwrap();
    probe.wait_for_all_buffer_drops();
    release.send(()).unwrap();
    let terminal = task.join().await.unwrap_err();
    assert_eq!(terminal.cause().to_string(), cause);
    assert_eq!(
        terminal.cleanup().unwrap().snapshot().phase(),
        ExecutorShutdownPhase::Retired
    );
    assert_eq!(probe.snapshot().submissions, 1);
}
