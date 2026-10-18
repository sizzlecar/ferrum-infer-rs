//! The managed task owns real numerical work and cleanup independently of its
//! observers. Timeouts are watchdogs; oneshots and one poll establish phases.

use super::*;
use crate::continuous_engine::token_completion_tests as closure;
use ferrum_execution::token_execution::{ExecutionCloseDisposition, ExecutionProduct, TokenWork};
use futures::{future::poll_fn, StreamExt};
use std::{future::Future, pin::Pin, task::Poll};

#[path = "managed_driver_tests/fixture.rs"]
mod fixture;
#[path = "managed_driver_tests/metrics.rs"]
mod metrics;

#[path = "managed_driver_tests/request_clock.rs"]
mod request_clock;
#[path = "managed_driver_tests/sequence_isolation.rs"]
mod sequence_isolation;

async fn pending_once<F: Future>(mut future: Pin<&mut F>) {
    poll_fn(|context| {
        assert!(
            future.as_mut().poll(context).is_pending(),
            "phase must actually suspend"
        );
        Poll::Ready(())
    })
    .await;
}

fn actual_close_after_driver_exit(fixture: &closure::Fixture) {
    assert_eq!(fixture.engine.inner.execution_closes.unreclaimed_count(), 0);
    assert_eq!(
        fixture.identity.observed_end().unwrap().sequence(),
        &fixture.identity
    );
    let model = fixture.model.lock();
    assert!(model.model.kv_caches.is_empty());
    assert_eq!(
        model.releases,
        vec![vec![fixture.prompt_len; fixture.layers]]
    );
    drop(model);
    let trace = fixture.trace.path().join("trace.jsonl");
    assert_eq!(
        fixture::closed_slot_events(&trace, ResourceAction::Release),
        1
    );
    assert_eq!(
        fixture::closed_slot_events(&trace, ResourceAction::RequestClose),
        1
    );
    assert_eq!(
        fixture
            .engine
            .inner
            .scheduler
            .trace_snapshot()
            .capacity_release_epoch,
        1
    );
}

#[tokio::test]
async fn managed_driver_explicit_lazy_share_one_real_cpu_execution() {
    let mut fixture = fixture::Fixture::new();
    let first = fixture.engine.start_loop().unwrap();
    let second = fixture.engine.start_loop().unwrap();
    drop(second);
    let stream = fixture
        .engine
        .infer_stream(fixture.request())
        .await
        .unwrap();
    tokio::time::timeout(Duration::from_secs(2), &mut fixture.reached)
        .await
        .unwrap()
        .unwrap();
    assert!(fixture.executor.checked_outputs.load(Ordering::SeqCst) > 0);
    // New explicit starts and a second product ingress occur while the first
    // actual numerical result remains owned by the sole driver future.
    let third = fixture.engine.start_loop().unwrap();
    let mut inference = Box::pin(fixture.engine.infer(fixture.request()));
    pending_once(inference.as_mut()).await;
    assert_eq!(fixture.engine.inner.sequences.len(), 2);
    fixture.release.send(()).unwrap();
    let (response, chunks) = tokio::time::timeout(Duration::from_secs(3), async {
        tokio::join!(inference, stream.collect::<Vec<_>>())
    })
    .await
    .unwrap();
    let response = response.unwrap();
    assert_eq!(response.response().tokens.len(), 4);
    let chunks = chunks.into_iter().collect::<Result<Vec<_>>>().unwrap();
    let text: String = chunks
        .iter()
        .map(|chunk| chunk.chunk().text.as_str())
        .collect();
    assert_eq!(text, response.response().text);
    let stream_tokens: Vec<_> = chunks
        .iter()
        .filter_map(|chunk| chunk.chunk().token)
        .collect();
    let visible: Vec<_> = response
        .response()
        .tokens
        .iter()
        .copied()
        .filter(|token| !fixture.engine.inner.tokenizer.is_special_token(*token))
        .collect();
    assert_eq!(stream_tokens, visible);
    assert_eq!(
        chunks.last().unwrap().chunk().finish_reason,
        Some(FinishReason::Length)
    );
    assert_eq!(fixture.executor.maximum_in_flight.load(Ordering::SeqCst), 1);
    assert_eq!(fixture.executor.admissions.load(Ordering::SeqCst), 2);
    assert_eq!(fixture.executor.checked_outputs.load(Ordering::SeqCst), 8);
    fixture.engine.shutdown().await.unwrap();
    let (first_report, third_report) = tokio::join!(first.wait(), third.wait());
    assert!(Arc::ptr_eq(&first_report, &third_report));
    first_report.result().unwrap();
    assert!(!first_report.retains_unproved_owners());
    assert_eq!(fixture.engine.driver.start_count(), 1);
    assert_eq!(fixture.executor.closes.load(Ordering::SeqCst), 2);
    assert!(fixture.executor.observed.lock().model.kv_caches.is_empty());
    assert_eq!(
        fixture::closed_slot_events(
            &fixture.trace.path().join("trace.jsonl"),
            ResourceAction::Release
        ),
        2
    );
}

#[tokio::test]
async fn managed_driver_closing_wins_both_ingress_publications() {
    let fixture = fixture::Fixture::new();
    let recorder = metrics::ActiveRequests::default();
    let _recorder = ::metrics::set_default_local_recorder(&recorder);
    let lock = fixture.engine.inner.iteration_lock.lock().await;
    let mut inference = Box::pin(fixture.engine.infer(fixture.request()));
    let mut streaming = Box::pin(fixture.engine.infer_stream(fixture.request()));
    pending_once(inference.as_mut()).await;
    pending_once(streaming.as_mut()).await;
    assert!(fixture.engine.inner.sequences.is_empty());
    assert_eq!(
        fixture::slot_events(
            &fixture.engine,
            &fixture.trace.path().join("trace.jsonl"),
            ResourceAction::RequestOpen
        ),
        0
    );
    assert_eq!(recorder.value(), 0.0);
    let mut shutdown = Box::pin(fixture.engine.shutdown());
    // Closing is synchronous before shutdown's first await. Publication is
    // still held at the iteration lock, so neither ingress has passed the gate.
    pending_once(shutdown.as_mut()).await;
    assert!(fixture.engine.start_loop().is_err());
    drop(lock);
    assert!(inference.await.is_err());
    assert!(streaming.await.is_err());
    tokio::time::timeout(Duration::from_secs(2), shutdown)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(fixture.executor.admissions.load(Ordering::SeqCst), 0);
    assert_eq!(fixture.executor.checked_outputs.load(Ordering::SeqCst), 0);
    assert_eq!(fixture.engine.inner.scheduler.waiting_count(), 0);
    assert!(fixture.engine.inner.sequences.is_empty());
    assert_eq!(
        fixture::closed_slot_events(
            &fixture.trace.path().join("trace.jsonl"),
            ResourceAction::RequestOpen
        ),
        0
    );
    assert_eq!(recorder.value(), 0.0);
}

#[tokio::test]
async fn managed_driver_cancel_before_publication_opens_no_slot() {
    let fixture = fixture::Fixture::new();
    let recorder = metrics::ActiveRequests::default();
    let _recorder = ::metrics::set_default_local_recorder(&recorder);
    let lock = fixture.engine.inner.iteration_lock.lock().await;
    let mut inference = Box::pin(fixture.engine.infer(fixture.request()));
    let mut streaming = Box::pin(fixture.engine.infer_stream(fixture.request()));
    pending_once(inference.as_mut()).await;
    pending_once(streaming.as_mut()).await;
    drop(inference);
    drop(streaming);
    drop(lock);
    assert_eq!(recorder.value(), 0.0);
    assert_eq!(
        fixture::slot_events(
            &fixture.engine,
            &fixture.trace.path().join("trace.jsonl"),
            ResourceAction::RequestOpen
        ),
        0
    );
    assert_eq!(fixture.engine.driver.start_count(), 0);
    fixture.engine.shutdown().await.unwrap();
    assert_eq!(fixture.executor.admissions.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn managed_driver_cancel_published_infer_balances_gauge_and_closes_real_kv() {
    let mut fixture = fixture::Fixture::new();
    let recorder = metrics::ActiveRequests::default();
    let _recorder = ::metrics::set_default_local_recorder(&recorder);
    let request = fixture.request();
    let id = request.id.clone();
    let mut inference = Box::pin(fixture.engine.infer(request));
    pending_once(inference.as_mut()).await;
    tokio::time::timeout(Duration::from_secs(2), &mut fixture.reached)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(recorder.value(), 1.0);
    assert_eq!(fixture.executor.observed.lock().model.kv_caches.len(), 1);
    assert_eq!(fixture.executor.checked_outputs.load(Ordering::SeqCst), 1);
    let observer = fixture.engine.start_loop().unwrap();
    drop(inference);
    assert_eq!(recorder.value(), 0.0);
    assert_eq!(
        fixture::slot_events(
            &fixture.engine,
            &fixture.trace.path().join("trace.jsonl"),
            ResourceAction::Release
        ),
        0
    );
    let mut shutdown = Box::pin(fixture.engine.shutdown());
    pending_once(shutdown.as_mut()).await;
    assert!(fixture.engine.start_loop().is_err());
    assert!(fixture.engine.inner.sequences.get(&id).is_some());
    fixture.release.send(()).unwrap();
    tokio::time::timeout(Duration::from_secs(2), shutdown)
        .await
        .unwrap()
        .unwrap();
    observer.wait().await.result().unwrap();
    assert!(fixture.engine.inner.sequences.is_empty());
    assert_eq!(fixture.engine.inner.scheduler.trace_phase(&id), None);
    assert!(fixture.executor.observed.lock().model.kv_caches.is_empty());
    assert_eq!(fixture.executor.closes.load(Ordering::SeqCst), 1);
    assert_eq!(fixture.executor.checked_outputs.load(Ordering::SeqCst), 1);
    assert_eq!(
        fixture::closed_slot_events(
            &fixture.trace.path().join("trace.jsonl"),
            ResourceAction::Release
        ),
        1
    );
    assert_eq!(recorder.value(), 0.0);
}

#[tokio::test]
async fn managed_driver_dropped_shutdown_observer_preserves_actual_close_and_join() {
    let mut fixture = closure::fixture(true, false).await;
    let mut completion = Box::pin(
        fixture
            .engine
            .inner
            .complete_request_for_test(&fixture.request_id, FinishReason::Length),
    );
    tokio::select! {
        biased;
        result = &mut completion => panic!("close must be held: {result:?}"),
        started = &mut fixture.started => started.unwrap(),
    }
    drop(completion);
    assert!(fixture.engine.inner.sequences.is_empty());
    let observer = fixture.engine.start_loop().unwrap();
    let remaining_observer = observer.clone();
    let mut shutdown = Box::pin(fixture.engine.shutdown());
    pending_once(shutdown.as_mut()).await;
    drop(shutdown);
    drop(observer);
    assert!(fixture.engine.start_loop().is_err());
    assert!(fixture.identity.observed_end().is_none());
    assert_eq!(fixture.engine.inner.execution_closes.unreclaimed_count(), 1);
    assert_eq!(closure::slot_events(&fixture, ResourceAction::Release), 0);
    assert_eq!(fixture.model.lock().model.kv_caches.len(), 1);
    fixture.gate.take().unwrap().send(()).unwrap();
    let report = tokio::time::timeout(Duration::from_secs(2), remaining_observer.wait())
        .await
        .unwrap();
    report.result().unwrap();
    assert!(!report.retains_unproved_owners());
    fixture.engine.shutdown().await.unwrap();
    let again = remaining_observer.wait().await;
    assert!(Arc::ptr_eq(&report, &again));
    actual_close_after_driver_exit(&fixture);
    assert_eq!(fixture.engine.driver.start_count(), 1);
}

#[tokio::test]
async fn managed_driver_panic_closes_real_cpu_request_and_never_reopens() {
    // This fixture has already admitted and executed a full-vocabulary real
    // prefill, checked against a separate model. Fault only the driver boundary.
    let mut fixture = closure::fixture(false, false).await;
    fixture
        .engine
        .driver
        .panic_next_iteration
        .store(true, Ordering::SeqCst);
    let observer = fixture.engine.start_loop().unwrap();
    let report = tokio::time::timeout(Duration::from_secs(2), observer.wait())
        .await
        .unwrap();
    assert!(report.result().is_err());
    assert!(!report.retains_unproved_owners());
    assert!(fixture.response.try_recv().unwrap().is_err());
    assert!(fixture.stream.try_recv().unwrap().is_err());
    assert!(fixture.engine.start_loop().is_err());
    assert!(fixture.engine.infer(policy_request()).await.is_err());
    assert!(fixture.engine.infer_stream(policy_request()).await.is_err());
    assert!(fixture.engine.shutdown().await.is_err());
    assert_eq!(fixture.engine.driver.start_count(), 1);
    actual_close_after_driver_exit(&fixture);
}

#[tokio::test]
async fn managed_driver_failed_report_retains_unproved_owner_without_control_cycle() {
    let fixture = closure::fixture(false, false).await;
    fixture
        .executor
        .fail_cancellation
        .store(true, Ordering::SeqCst);
    fixture
        .engine
        .driver
        .panic_next_iteration
        .store(true, Ordering::SeqCst);
    let control = Arc::downgrade(&fixture.engine.driver);
    let inner = Arc::downgrade(&fixture.engine.inner);
    let observer = fixture.engine.start_loop().unwrap();
    let report = tokio::time::timeout(Duration::from_secs(2), observer.wait())
        .await
        .unwrap();
    assert!(report.result().is_err());
    assert!(report.retains_unproved_owners());
    assert!(fixture.identity.observed_end().is_none());
    // Unproved execution keeps the original producer scope live. A readable
    // flush barrier is not proof that the engine may retire that scope.
    assert!(!report.journal_deliveries().is_empty());
    for delivery in report.journal_deliveries() {
        assert!(!delivery.stats.worker_finished);
        assert!(delivery.scope_receipt.is_none());
        assert!(delivery.final_receipt.is_none());
    }
    fixture
        .engine
        .inner
        .scheduler_trace_jsonl
        .as_ref()
        .unwrap()
        .flush()
        .unwrap();
    assert_eq!(
        fixture::closed_slot_events(
            &fixture.trace.path().join("trace.jsonl"),
            ResourceAction::Release,
        ),
        0
    );
    assert_eq!(fixture.model.lock().model.kv_caches.len(), 1);
    drop(observer);
    drop(fixture.engine);
    assert!(
        control.upgrade().is_none(),
        "failed report must not retain the driver control"
    );
    let retained = inner
        .upgrade()
        .expect("report retains original unresolved custody");
    assert_eq!(retained.execution_closes.unreclaimed_count(), 1);
    assert!(fixture.identity.observed_end().is_none());
    // Resolve the same real CPU owner explicitly, never by dropping its report.
    fixture
        .executor
        .fail_cancellation
        .store(false, Ordering::SeqCst);
    let result = fixture.executor.cancel_tokens(&fixture.identity).unwrap();
    assert!(
        matches!(result.disposition, ExecutionCloseDisposition::Ended(ref ended) if ended.sequence() == &fixture.identity)
    );
    retained.drain_execution_closes().unwrap();
    assert!(fixture.model.lock().model.kv_caches.is_empty());
    assert_eq!(
        fixture.model.lock().releases,
        vec![vec![fixture.prompt_len; fixture.layers]]
    );
    assert_eq!(retained.execution_closes.unreclaimed_count(), 0);
    drop(retained);
    drop(report);
    assert!(
        inner.upgrade().is_none(),
        "the resolved custody has no permanent Arc cycle"
    );
}

#[tokio::test(flavor = "current_thread")]
async fn managed_driver_clean_exit_releases_inner_without_observer_wait() {
    let mut fixture = fixture::Fixture::new();
    let observer = fixture.engine.start_loop().unwrap();
    let inner = Arc::downgrade(&fixture.engine.inner);
    let control = Arc::downgrade(&fixture.engine.driver);
    let request = fixture.request();
    let mut inference = Box::pin(fixture.engine.infer(request));
    pending_once(inference.as_mut()).await;
    tokio::time::timeout(Duration::from_secs(2), &mut fixture.reached)
        .await
        .unwrap()
        .unwrap();
    fixture.release.send(()).unwrap();
    let response = tokio::time::timeout(Duration::from_secs(2), inference)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(response.response().tokens.len(), 4);
    assert_eq!(fixture.executor.checked_outputs.load(Ordering::SeqCst), 4);
    assert_eq!(fixture.executor.closes.load(Ordering::SeqCst), 1);
    assert!(fixture.executor.observed.lock().model.kv_caches.is_empty());

    let (dropped, executor_dropped) = tokio::sync::oneshot::channel();
    *fixture.executor.dropped.lock() = Some(dropped);
    // Request shutdown synchronously without polling any observer's wait.
    drop(
        fixture
            .engine
            .driver
            .shutdown(&fixture.engine.inner)
            .unwrap(),
    );
    drop(fixture.engine);
    drop(fixture.executor);
    // The drop signal comes from the actual model executor owned by Inner.
    // On this current-thread runtime, its drop and task return finish before
    // this test can resume; no timing sleep or join side effect is involved.
    tokio::time::timeout(Duration::from_secs(2), executor_dropped)
        .await
        .expect("clean task must release its original Inner without wait")
        .unwrap();
    assert!(inner.upgrade().is_none());
    assert!(control.upgrade().is_none());
    drop(observer);
}

#[path = "managed_driver_tests/output.rs"]
mod output;

#[path = "managed_driver_tests/semantic.rs"]
mod semantic;

#[path = "managed_driver_tests/host_profile.rs"]
mod host_profile;
#[path = "managed_driver_tests/trace_lazy.rs"]
mod trace_lazy;

#[path = "managed_driver_tests/observability.rs"]
mod observability;
