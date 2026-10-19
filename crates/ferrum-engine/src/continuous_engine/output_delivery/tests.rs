use super::*;
use crate::continuous_engine::token_completion_tests as actual;

async fn completion_after_delivery_failure(panic: bool) {
    let mut fixture = actual::fixture(false, false).await;
    let sender = fixture
        .engine
        .active_inner()
        .expect("engine execution is open")
        .sequences
        .get(&fixture.request_id)
        .unwrap()
        .lock()
        .as_ref()
        .unwrap()
        .stream_sender
        .as_ref()
        .unwrap()
        .clone();
    let (reached, observed) = tokio::sync::oneshot::channel();
    let (resume, resumed) = tokio::sync::oneshot::channel();
    *sender.0.failed_handoff.lock() = Some(FailedHandoffHook {
        reached,
        resume: resumed,
    });
    sender.0.panic_before_drain.store(panic, Ordering::SeqCst);
    {
        let mut state = sender.0.state.lock();
        // Exercise the real projection check with an invalid byte range. The
        // model, token history, close and slot remain the actual CPU fixture.
        state.pending.push_back(TextSpan {
            bytes: 0..usize::MAX,
            token: None,
            terminal: None,
            created_at: chrono::Utc::now(),
            cost: 1,
            charged: None,
        });
        state.outstanding = 1;
        sender
            .start_locked(
                &fixture
                    .engine
                    .active_inner()
                    .expect("engine execution is open"),
                &mut state,
            )
            .unwrap();
    }
    tokio::time::timeout(Duration::from_secs(3), observed)
        .await
        .unwrap()
        .unwrap();
    let original = sender
        .failure()
        .expect("the original projection failure is retained");
    assert!(!sender.0.state.lock().running);
    if panic {
        // The panic guard retains a delivery failure, independently of the
        // checked text projection's control-error reporting path.
        assert!(matches!(&original, FerrumError::Internal { message }
            if message == "output delivery panicked"));
        assert!(fixture
            .engine
            .active_inner()
            .expect("engine execution is open")
            .output_tasks
            .take_control_error()
            .is_none());
    } else {
        let control = fixture
            .engine
            .active_inner()
            .expect("engine execution is open")
            .output_tasks
            .take_control_error()
            .expect("actual invalid live span must reach the driver control path");
        assert!(matches!(&control, FerrumError::Internal { message }
            if message == "pending output span disagrees with original token history"));
        assert_eq!(control.to_string(), original.to_string());
        assert!(fixture
            .engine
            .active_inner()
            .expect("engine execution is open")
            .output_tasks
            .take_control_error()
            .is_none());
    }
    // The previous worker is still stopped at the handoff barrier. A later
    // completion must own a new run and deliver the retained original error.
    let result = tokio::time::timeout(
        Duration::from_secs(3),
        fixture
            .engine
            .active_inner()
            .expect("engine execution is open")
            .complete_request_for_test(&fixture.request_id, FinishReason::Length),
    )
    .await
    .unwrap()
    .unwrap_err();
    assert_eq!(result.to_string(), original.to_string());
    assert_eq!(
        fixture
            .response
            .try_recv()
            .unwrap()
            .unwrap_err()
            .to_string(),
        original.to_string()
    );
    assert_eq!(
        fixture
            .stream
            .recv()
            .await
            .unwrap()
            .unwrap_err()
            .to_string(),
        original.to_string()
    );
    assert!(fixture.identity.observed_end().is_some());
    assert!(fixture.model.lock().model.kv_caches.is_empty());
    assert_eq!(
        fixture.model.lock().releases,
        vec![vec![fixture.prompt_len; fixture.layers]]
    );
    resume.send(()).unwrap();
    fixture
        .engine
        .active_inner()
        .expect("engine execution is open")
        .output_tasks
        .join_current()
        .await;
    assert!(
        fixture
            .engine
            .active_inner()
            .expect("engine execution is open")
            .output_tasks
            .take_control_error()
            .is_none(),
        "the later completion must preserve the client failure without re-reporting control"
    );
    assert!(!sender.0.state.lock().running);
    assert_eq!(sender.0.state.lock().enqueued_bytes, 0);
    drop(sender);
    assert!(fixture.stream.recv().await.is_none());
    fixture.engine.shutdown().await.unwrap();
}

#[tokio::test]
async fn output_failed_generation_cannot_discard_later_real_completion() {
    completion_after_delivery_failure(false).await;
}

#[tokio::test]
async fn output_projection_panic_does_not_leave_request_parked() {
    completion_after_delivery_failure(true).await;
}

#[tokio::test]
async fn output_terminal_payload_budget_error_is_not_silent_eof() {
    let mut fixture = actual::fixture(false, false).await;
    let budget = OutputBudget::new(ferrum_types::OutputConfig {
        queued_bytes_total: 1,
        ..Default::default()
    })
    .unwrap();
    let (sender, mut receiver) = OutputSender::new(
        fixture
            .engine
            .active_inner()
            .expect("engine execution is open")
            .self_weak
            .clone(),
        fixture.request_id.clone(),
        budget.try_open_session().unwrap(),
        4,
    );
    {
        let sequences = &fixture
            .engine
            .active_inner()
            .expect("engine execution is open")
            .sequences;
        let sequence_handle = sequences.get(&fixture.request_id).unwrap();
        sender.bind_sequence(Arc::downgrade(&sequence_handle));
        let mut sequence_guard = sequence_handle.lock();
        let sequence = sequence_guard.as_mut().unwrap();
        sequence.streamed_text_len = fixture
            .engine
            .active_inner()
            .expect("engine execution is open")
            .tokenizer
            .decode(&sequence.generated_tokens, true)
            .unwrap()
            .len();
        sequence
            .bind_output_session(sender.output_session().clone())
            .unwrap();
        sequence.stream_sender = Some(sender);
    }
    let error = fixture
        .engine
        .active_inner()
        .expect("engine execution is open")
        .complete_request_for_test(&fixture.request_id, FinishReason::Length)
        .await
        .unwrap_err();
    assert!(matches!(error, FerrumError::ResourceExhausted { .. }));
    assert_eq!(
        receiver.recv().await.unwrap().unwrap_err().to_string(),
        error.to_string()
    );
    assert!(receiver.recv().await.is_none());
    assert!(fixture.identity.observed_end().is_some());
    assert!(fixture.model.lock().model.kv_caches.is_empty());
    assert_eq!(fixture.model.lock().releases.len(), 1);
    fixture.response.try_recv().unwrap().unwrap();
    fixture.engine.shutdown().await.unwrap();
}

#[tokio::test]
async fn output_closed_receiver_cancels_terminal_quota_wait() {
    let budget = OutputBudget::new(ferrum_types::OutputConfig::default()).unwrap();
    let session = budget.try_open_session().unwrap();
    let held = session
        .reserve_queued(session.config().queued_bytes_total)
        .await
        .unwrap();
    let request_id = RequestId::new();
    let (sender, receiver) = OutputSender::new(Weak::new(), request_id.clone(), session, 1);
    drop(receiver);
    let error = tokio::time::timeout(
        Duration::from_secs(3),
        sender.send(Ok(StreamChunk {
            request_id,
            text: String::new(),
            token: None,
            finish_reason: Some(FinishReason::Length),
            usage: None,
            created_at: chrono::Utc::now(),
            metadata: HashMap::new(),
            api_response: None,
            execution_evidence: None,
        })),
    )
    .await
    .unwrap()
    .unwrap_err();
    assert!(error.to_string().contains("receiver closed"));
    assert_eq!(budget.usage().queued_bytes, held.bytes());
    drop(held);
}

#[tokio::test]
async fn output_full_queue_does_not_hide_actual_close_control_error_from_driver() {
    let mut fixture = actual::fixture(true, true).await;
    let (sender, mut receiver) = fixture
        .engine
        .active_inner()
        .expect("engine execution is open")
        .output_channel(fixture.request_id.clone(), 1)
        .unwrap();
    let token = {
        let sequences = &fixture
            .engine
            .active_inner()
            .expect("engine execution is open")
            .sequences;
        let sequence_handle = sequences.get(&fixture.request_id).unwrap();
        sender.bind_sequence(Arc::downgrade(&sequence_handle));
        let mut sequence_guard = sequence_handle.lock();
        let sequence = sequence_guard.as_mut().unwrap();
        sequence.stream_sender = Some(sender.clone());
        sequence.generated_tokens[0]
    };
    fixture
        .engine
        .active_inner()
        .expect("engine execution is open")
        .send_stream_update(&fixture.request_id, token)
        .await;
    assert_eq!(
        sender.0.sender.capacity(),
        0,
        "actual CPU text fills the queue"
    );
    fixture
        .engine
        .active_inner()
        .expect("engine execution is open")
        .complete_request_detached(&fixture.request_id, FinishReason::Length, None)
        .unwrap();
    tokio::time::timeout(Duration::from_secs(3), &mut fixture.started)
        .await
        .unwrap()
        .unwrap();
    fixture.gate.take().unwrap().send(()).unwrap();
    fixture
        .engine
        .active_inner()
        .expect("engine execution is open")
        .execution_closes
        .join_observers()
        .await;
    // No client polling and no shutdown: the ordinary driver consumer sees
    // the actual close's failure independently of the full delivery queue.
    let error = fixture
        .engine
        .active_inner()
        .expect("engine execution is open")
        .run_iteration()
        .await
        .err()
        .expect("driver must observe the actual close failure");
    assert!(error
        .to_string()
        .contains("actual close completed but publication failed"));
    assert!(fixture.identity.observed_end().is_some());
    assert!(fixture.model.lock().model.kv_caches.is_empty());
    assert_eq!(fixture.model.lock().releases.len(), 1);
    assert_eq!(
        receiver.recv().await.unwrap().unwrap().chunk().token,
        Some(token)
    );
    assert!(receiver
        .recv()
        .await
        .unwrap()
        .unwrap_err()
        .to_string()
        .contains("actual close completed but publication failed"));
    fixture
        .engine
        .active_inner()
        .expect("engine execution is open")
        .output_tasks
        .join_current()
        .await;
    drop(sender);
    assert!(receiver.recv().await.is_none());
    // If the completion observer consumed the same initial error, its queued
    // control report must also use the production consumption path.
    if let Err(error) = fixture
        .engine
        .active_inner()
        .expect("engine execution is open")
        .run_iteration()
        .await
    {
        assert!(error
            .to_string()
            .contains("actual close completed but publication failed"));
    }
    fixture.engine.shutdown().await.unwrap();
}
