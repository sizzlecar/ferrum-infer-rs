use super::*;
use ferrum_bench_core::JsonlJournal;
use serde::{Serialize, Serializer};
use std::{sync::Condvar, time::Duration};

struct WriteGate {
    released: Mutex<bool>,
    wake: Condvar,
    entered: Mutex<Option<tokio::sync::oneshot::Sender<()>>>,
}
impl WriteGate {
    fn release(&self) {
        *self.released.lock().unwrap() = true;
        self.wake.notify_all();
    }
}
struct BlockedRecord(Arc<WriteGate>);
impl Serialize for BlockedRecord {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        if let Some(entered) = self.0.entered.lock().unwrap().take() {
            let _ = entered.send(());
        }
        let mut released = self.0.released.lock().unwrap();
        while !*released {
            released = self.0.wake.wait(released).unwrap();
        }
        serializer.serialize_str("test-only journal gate")
    }
}
struct BlockedJournal {
    gate: Arc<WriteGate>,
    _journal: JsonlJournal<BlockedRecord>,
}
impl Drop for BlockedJournal {
    fn drop(&mut self) {
        // A test failure must also unblock the real writer thread before join.
        self.gate.release();
    }
}
async fn block_journal(path: &Path) -> BlockedJournal {
    let (entered, receiver) = tokio::sync::oneshot::channel();
    let gate = Arc::new(WriteGate {
        released: Mutex::new(false),
        wake: Condvar::new(),
        entered: Mutex::new(Some(entered)),
    });
    let journal = JsonlJournal::create(path).unwrap();
    let blocked = BlockedJournal {
        gate: gate.clone(),
        _journal: journal,
    };
    blocked._journal.enqueue(BlockedRecord(gate)).unwrap();
    tokio::time::timeout(Duration::from_secs(3), receiver)
        .await
        .unwrap()
        .unwrap();
    blocked
}
fn profiled_server(engine: Arc<CreditedRouteLlm>, path: PathBuf) -> AxumServer {
    AxumServer::from_state(
        AppState::default()
            .with_llm(engine)
            .with_profile_detail(ferrum_types::ObservabilityProfileDetail::Latency)
            .with_profile_jsonl(Some(path)),
    )
}
async fn start_output(server: &AxumServer) -> Response {
    let response = post_json(
        server.build_router(),
        "/v1/completions",
        completion_request(),
    )
    .await;
    assert_eq!(response.status(), AxumStatusCode::OK);
    response
}
fn terminal_profile(path: &Path) -> Value {
    let events: Vec<Value> = std::fs::read_to_string(path)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .filter(|event: &Value| event["phase"] == "credited_generation")
        .collect();
    assert_eq!(events.len(), 1);
    assert_eq!(events[0]["attributes"]["engine_token_commit_count"], 1);
    events.into_iter().next().unwrap()
}

#[tokio::test]
async fn credited_evidence_shutdown_waits_for_same_lease_write_and_flush() {
    let path = unique_profile_jsonl("credited-stop-slow-write");
    let blocked = block_journal(&path).await;
    let engine = Arc::new(CreditedRouteLlm::new(SloOutputTransport::Credited).await);
    let server = profiled_server(engine.clone(), path.clone());
    assert!(response_text(start_output(&server).await)
        .await
        .contains("[DONE]"));
    let stop = server.stop(Duration::from_secs(3));
    tokio::pin!(stop);
    assert!(tokio::time::timeout(Duration::from_millis(10), &mut stop)
        .await
        .is_err());
    assert_eq!(engine.base.shutdown_count.load(Ordering::Acquire), 0);
    assert!(engine.pool.snapshot().retained_accounts > 0);
    blocked.gate.release();
    stop.await.unwrap();
    terminal_profile(&path);
    engine.assert_released();
    assert_eq!(engine.base.shutdown_count.load(Ordering::Acquire), 1);
    server.stop(Duration::from_secs(1)).await.unwrap();
    assert_eq!(engine.base.shutdown_count.load(Ordering::Acquire), 1);
    // Even a directly held router cannot register evidence after successful stop.
    let rejected = post_json(
        server.build_router(),
        "/v1/completions",
        completion_request(),
    )
    .await;
    assert_eq!(rejected.status(), AxumStatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(engine.credited_calls.load(Ordering::Acquire), 1);
    drop(blocked);
    std::fs::remove_file(path).unwrap();
}

#[tokio::test]
async fn credited_evidence_shutdown_timeout_keeps_live_lease_and_still_cleans_engine() {
    let path = unique_profile_jsonl("credited-stop-timeout");
    let blocked = block_journal(&path).await;
    let engine = Arc::new(CreditedRouteLlm::new(SloOutputTransport::Credited).await);
    let server = profiled_server(engine.clone(), path.clone());
    let _ = response_text(start_output(&server).await).await;
    let error = server.stop(Duration::from_millis(10)).await.unwrap_err();
    assert!(error
        .to_string()
        .contains("credited evidence did not drain"));
    assert!(engine.pool.snapshot().retained_accounts > 0);
    assert_eq!(engine.base.shutdown_count.load(Ordering::Acquire), 1);
    blocked.gate.release();
    server.stop(Duration::from_secs(3)).await.unwrap();
    terminal_profile(&path);
    engine.assert_released();
    assert_eq!(engine.base.shutdown_count.load(Ordering::Acquire), 1);
    drop(blocked);
    std::fs::remove_file(path).unwrap();
}

#[tokio::test]
async fn credited_evidence_shutdown_dropped_http_body_still_drains_completion() {
    let path = unique_profile_jsonl("credited-stop-body-cancel");
    let engine = Arc::new(CreditedRouteLlm::new(SloOutputTransport::Credited).await);
    let server = profiled_server(engine.clone(), path.clone());
    drop(start_output(&server).await);
    assert!(engine.control.0.load(Ordering::Acquire));
    server.stop(Duration::from_secs(3)).await.unwrap();
    // This fixture already completed before body cancellation. Do not relabel
    // its Succeeded terminal as an engine cancellation receipt.
    terminal_profile(&path);
    engine.assert_released();
    std::fs::remove_file(path).unwrap();
}

#[tokio::test]
async fn credited_evidence_shutdown_write_failure_is_sticky_and_cleans_engine() {
    let path = unique_request_dump_dir("credited-stop-write-error");
    std::fs::create_dir_all(&path).unwrap(); // directory cannot be a JSONL file
    let engine = Arc::new(CreditedRouteLlm::new(SloOutputTransport::Credited).await);
    let server = profiled_server(engine.clone(), path.clone());
    let _ = response_text(start_output(&server).await).await;
    let error = server.stop(Duration::from_secs(3)).await.unwrap_err();
    assert!(error.to_string().contains("profile write failed"));
    assert_eq!(engine.base.shutdown_count.load(Ordering::Acquire), 1);
    engine.assert_released();
    assert!(server.stop(Duration::from_secs(1)).await.is_err());
    assert_eq!(engine.base.shutdown_count.load(Ordering::Acquire), 1);
    std::fs::remove_dir(path).unwrap();
}

#[tokio::test]
async fn credited_evidence_shutdown_http_timeout_does_not_renew_evidence_budget() {
    let path = unique_profile_jsonl("credited-stop-shared-deadline");
    let blocked = block_journal(&path).await;
    let engine = Arc::new(CreditedRouteLlm::new(SloOutputTransport::Credited).await);
    let server = profiled_server(engine.clone(), path.clone());
    let _ = response_text(start_output(&server).await).await;
    // Model the lifecycle's still-draining HTTP task; only its real run guard
    // may clear this bit. This is a stop boundary test, not a socket benchmark.
    server.lifecycle.running.store(true, Ordering::Release);
    let error = tokio::time::timeout(
        Duration::from_millis(80),
        server.stop(Duration::from_millis(50)),
    )
    .await
    .expect("HTTP and evidence must share the original 50ms deadline")
    .unwrap_err();
    assert!(error.to_string().contains("Axum server did not drain"));
    assert_eq!(engine.base.shutdown_count.load(Ordering::Acquire), 1);
    assert!(engine.pool.snapshot().retained_accounts > 0);
    server.lifecycle.running.store(false, Ordering::Release);
    blocked.gate.release();
    server.stop(Duration::from_secs(3)).await.unwrap();
    engine.assert_released();
    drop(blocked);
    std::fs::remove_file(path).unwrap();
}

#[tokio::test]
async fn credited_evidence_shutdown_rejected_startup_releases_unarmed_reservation() {
    let path = unique_profile_jsonl("credited-stop-admission-rejected");
    let mut engine = CreditedRouteLlm::new(SloOutputTransport::Credited).await;
    engine.startup_failure = Some(Error::internal("fixture admission rejected"));
    let engine = Arc::new(engine);
    let server = profiled_server(engine.clone(), path.clone());
    let response = post_json(
        server.build_router(),
        "/v1/completions",
        completion_request(),
    )
    .await;
    assert_eq!(response.status(), AxumStatusCode::INTERNAL_SERVER_ERROR);
    server.stop(Duration::from_secs(1)).await.unwrap();
    engine.assert_released();
    assert_eq!(engine.base.shutdown_count.load(Ordering::Acquire), 1);
    assert!(!path.exists());
}
