use super::*;
use ferrum_types::{FinishReason, RequestId};
use futures::{FutureExt, StreamExt};
use std::collections::HashMap;

fn chunk(text: &str) -> StreamChunk {
    StreamChunk {
        request_id: RequestId::new(),
        text: text.to_owned(),
        token: None,
        finish_reason: Some(FinishReason::Stop),
        usage: None,
        created_at: chrono::Utc::now(),
        metadata: HashMap::new(),
        api_response: None,
        execution_evidence: None,
    }
}

fn one_session() -> OutputBudget {
    OutputBudget::new(OutputConfig {
        max_sessions: 1,
        ..Default::default()
    })
    .unwrap()
}

#[tokio::test]
async fn completed_stream_retains_admission_until_outer_owner_is_dropped() {
    let budget = one_session();
    let session = budget.try_open_session().unwrap();
    let mut stream = InferenceStream::new(session, Box::pin(futures::stream::empty()));
    assert!(stream.next().await.is_none());
    assert!(budget.try_open_session().is_err());
    drop(stream);
    assert_eq!(budget.usage().sessions, 0);
    budget.try_open_session().unwrap();
}

#[tokio::test]
async fn consumed_chunk_keeps_its_bytes_and_session_after_stream_drop() {
    let budget = one_session();
    let session = budget.try_open_session().unwrap();
    let payload = chunk("中文\n\"quoted\" 🦀");
    let bytes = OutputChunk::queued_cost(&payload).unwrap();
    assert_eq!(bytes, serde_json::to_vec(&payload).unwrap().len());
    let held = session.wrap_chunk(payload).await.unwrap();
    let mut stream = InferenceStream::new(session, Box::pin(futures::stream::iter([Ok(held)])));
    let received = stream.next().await.unwrap().unwrap();
    assert!(stream.next().await.is_none());
    drop(stream);
    assert_eq!(received.chunk().text, "中文\n\"quoted\" 🦀");
    assert_eq!(
        budget.usage(),
        OutputUsage {
            sessions: 1,
            queued_bytes: bytes,
            encoded_bytes: 0,
            control_bytes: 0,
        }
    );
    assert!(budget.try_open_session().is_err());
    drop(received);
    assert_eq!(
        budget.usage(),
        OutputUsage {
            sessions: 0,
            queued_bytes: 0,
            encoded_bytes: 0,
            control_bytes: 0,
        }
    );
}

#[tokio::test]
async fn queued_budget_waits_for_actual_content_release_and_cancellation_is_safe() {
    let payload = chunk("retained");
    let bytes = OutputChunk::queued_cost(&payload).unwrap();
    let budget = OutputBudget::new(OutputConfig {
        queued_bytes_total: bytes,
        ..Default::default()
    })
    .unwrap();
    let session = budget.try_open_session().unwrap();
    let held = session.wrap_chunk(payload).await.unwrap();
    assert!(session.try_reserve_queued(1).unwrap().is_none());
    assert!(session.try_reserve_queued(bytes + 1).is_err());
    let mut cancelled = Box::pin(session.reserve_queued(1));
    assert!(futures::poll!(&mut cancelled).is_pending());
    drop(cancelled);
    assert_eq!(budget.usage().queued_bytes, bytes);
    let mut waiting = Box::pin(session.reserve_queued(bytes));
    assert!(futures::poll!(&mut waiting).is_pending());
    drop(held);
    let lease = waiting.await.unwrap();
    assert_eq!(lease.bytes(), bytes);
    drop(lease);
    assert_eq!(budget.usage().queued_bytes, 0);
}

#[tokio::test]
async fn encoding_uses_a_distinct_pool_and_lease_keeps_session() {
    let budget = OutputBudget::new(OutputConfig {
        max_sessions: 1,
        queued_bytes_total: 16,
        encoded_bytes_total: 16,
        max_event_bytes: 16,
        ..Default::default()
    })
    .unwrap();
    let session = budget.try_open_session().unwrap();
    let queued = session.try_reserve_queued(16).unwrap().unwrap();
    let encoded = session.reserve_encoded(16).now_or_never().unwrap().unwrap();
    assert!(session.reserve_encoded(17).now_or_never().unwrap().is_err());
    drop(queued);
    drop(session);
    assert_eq!(
        budget.usage(),
        OutputUsage {
            sessions: 1,
            queued_bytes: 0,
            encoded_bytes: 16,
            control_bytes: 0,
        }
    );
    assert!(budget.try_open_session().is_err());
    drop(encoded);
    assert_eq!(budget.usage().sessions, 0);
}

#[tokio::test]
async fn chunks_reject_incomplete_or_wrong_pool_reservations() {
    let budget = one_session();
    let session = budget.try_open_session().unwrap();
    let payload = chunk("unaccounted data must not escape");
    let lease = session.try_reserve_queued(1).unwrap().unwrap();
    assert!(OutputChunk::new(payload, lease).is_err());
    assert_eq!(budget.usage().queued_bytes, 0);
    let payload = chunk("wrong pool");
    let size = OutputChunk::queued_cost(&payload).unwrap();
    let lease = session.reserve_encoded(size).await.unwrap();
    assert!(OutputChunk::new(payload, lease).is_err());
    assert_eq!(budget.usage().encoded_bytes, 0);
}

#[test]
fn configuration_errors_are_returned_before_creating_runtime_pools() {
    assert!(OutputBudget::new(OutputConfig {
        max_sessions: 0,
        ..Default::default()
    })
    .is_err());
    assert!(OutputBudget::new(OutputConfig {
        max_event_bytes: usize::MAX,
        ..Default::default()
    })
    .is_err());
}

#[tokio::test]
async fn terminal_error_has_one_bounded_allowance_when_data_pool_is_full() {
    let budget = OutputBudget::new(OutputConfig {
        max_sessions: 1,
        encoded_bytes_total: 1,
        max_event_bytes: 1,
        ..Default::default()
    })
    .unwrap();
    let session = budget.try_open_session().unwrap();
    let data = session.reserve_encoded(1).await.unwrap();
    assert!(session.try_reserve_error(TERMINAL_ERROR_BYTES + 1).is_err());
    let error = session.try_reserve_error(TERMINAL_ERROR_BYTES).unwrap();
    assert_eq!(budget.usage().control_bytes, TERMINAL_ERROR_BYTES);
    assert!(session.try_reserve_error(1).is_err());
    drop(error);
    assert_eq!(budget.usage().control_bytes, 0);
    assert!(
        session.try_reserve_error(1).is_err(),
        "control allowance cannot be recycled"
    );
    drop(data);
    drop(session);
    let new_session = budget.try_open_session().unwrap();
    let error = new_session.try_reserve_error(16).unwrap();
    drop(new_session);
    assert!(budget.try_open_session().is_err());
    drop(error);
    assert_eq!(budget.usage().sessions, 0);
}
