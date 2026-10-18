use super::lifecycle::ServeTaskObserver;
use ferrum_bench_core::{JsonlJournalOpenMode, JsonlJournalOwner};
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
            )))
            .unwrap();
        wait.await
            .map_err(|e| FerrumError::internal(e.to_string()))?;
        producer
            .try_enqueue(json!({"after_actual_drain":true}))
            .map_err(|e| FerrumError::io(e.to_string()))?;
        scope
            .finish()
            .result()
            .map_err(|e| FerrumError::io(e.to_string()))
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

#[test]
fn serve_native_off_profile_assembly_uses_only_native_writer_and_rejects_scheduler_alias() {
    use crate::observability_product::{
        ProductJournalSession, ProductObservabilityConfig, ProfileDetailArg,
    };
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("native.jsonl");
    let product = ProductObservabilityConfig::new(
        ferrum_types::ProfileEntrypoint::Serve,
        "native-profile",
        Some(&path),
        ProfileDetailArg::Off,
        None,
        None,
        None,
        1.0,
    );
    let mut engine = ferrum_types::EngineConfig::default();
    engine.runtime.profile_jsonl = Some(path.clone());
    super::select_serve_profile_destination(&mut engine, &product);
    assert!(engine.runtime.profile_jsonl.is_none());
    let journals = ProductJournalSession::prepare(&product, &engine);
    journals.result().unwrap();
    assert!(
        !path.exists(),
        "typed assembly must not create the native writer"
    );
    drop(journals);
    engine.runtime.scheduler_trace_jsonl = Some(directory.path().join(".").join("native.jsonl"));
    let journals = ProductJournalSession::prepare(&product, &engine);
    assert!(matches!(journals.result(), Err(FerrumError::Config { .. })));
    assert!(!path.exists());
}
