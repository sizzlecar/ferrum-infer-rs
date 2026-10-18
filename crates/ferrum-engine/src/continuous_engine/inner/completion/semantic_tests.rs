//! Completion quota is tested after an actual CPU forward and before success
//! publication. No model result, close receipt, or allocator error is fabricated.
use super::*;
use crate::continuous_engine::token_completion_tests as actual;

async fn envelope_failure(stream: bool) {
    let mut fixture = actual::fixture(false, false).await;
    let handle = fixture
        .engine
        .inner
        .sequences
        .get(&fixture.request_id)
        .unwrap();
    let (session, text, token, sender) = {
        let mut state = handle.lock();
        let state = state.as_mut().unwrap();
        let sender = if stream {
            state.stream_sender.clone()
        } else {
            drop(state.stream_sender.take());
            None
        };
        (
            state.output_semantic.output_session().clone(),
            fixture
                .engine
                .inner
                .tokenizer
                .decode(&state.generated_tokens, true)
                .unwrap(),
            *state.generated_tokens.last().unwrap(),
            sender,
        )
    };
    assert!(!text.is_empty());
    let queued = if stream {
        let held = session
            .reserve_queued(session.config().queued_bytes_total)
            .await
            .unwrap();
        fixture
            .engine
            .inner
            .send_stream_update(&fixture.request_id, token)
            .await;
        assert!(matches!(
            fixture.stream.try_recv(),
            Err(mpsc::error::TryRecvError::Empty)
        ));
        Some(held)
    } else {
        None
    };
    // Existing token/timing payload is already paid. Leave enough for the real
    // final text and the evidence's copied TokenId, but not the complete DTO.
    let spare = text.len() + std::mem::size_of::<TokenId>();
    let blocker = session
        .try_reserve_semantic(
            session.config().semantic_bytes_per_session - session.semantic_usage() - spare,
        )
        .unwrap();
    let handoff = {
        let mut state = handle.lock();
        fixture
            .engine
            .inner
            .handoff_request_completion_locked(
                &handle,
                &mut state,
                FinishReason::Length,
                None,
                false,
                None,
                CompletionExpectation::Current,
                None,
            )
            .unwrap()
    };
    let error = handoff.terminal_error.as_ref().unwrap().clone();
    assert!(matches!(&error, FerrumError::ResourceExhausted { .. }));
    assert_eq!(
        handoff.completion_error.as_ref().unwrap().to_string(),
        error.to_string()
    );
    assert!(matches!(
        handoff.text_provenance,
        CompletionTextProvenance::Available
    ));
    let draft = handoff.output.draft().unwrap();
    assert_eq!(draft.response.text, text);
    assert_eq!(draft.response.tokens, vec![token]);
    assert_eq!(draft.response.finish_reason, FinishReason::Error);
    assert!(
        semantic_cost(&draft.response).unwrap() > draft.semantic.bytes(),
        "the rejection must be the full envelope, after text/evidence growth succeeded"
    );
    assert!(draft.terminal_copy_credit.is_none());
    assert!(
        matches!(handoff.rebuild_completed_text(0..text.len()).unwrap(),
        CompletedTextProjection::Available(actual) if actual == text)
    );
    assert!(
        handoff.rebuild_completed_text(0..text.len() + 1).is_err(),
        "an envelope error cannot hide a corrupt pending range"
    );
    assert_eq!(
        fixture.engine.inner.scheduler.metrics().completed_requests,
        0
    );
    assert_eq!(fixture.engine.inner.scheduler.metrics().failed_requests, 1);
    let done = fixture
        .engine
        .inner
        .schedule_output_completion(&fixture.request_id, handoff)
        .unwrap();
    assert_eq!(
        fixture
            .response
            .try_recv()
            .unwrap()
            .unwrap_err()
            .to_string(),
        error.to_string()
    );
    drop(queued);
    let control = tokio::time::timeout(Duration::from_secs(3), done)
        .await
        .unwrap()
        .unwrap()
        .unwrap_err();
    assert_eq!(control.to_string(), error.to_string());
    fixture.engine.inner.output_tasks.join_current().await;
    assert_eq!(
        fixture
            .engine
            .inner
            .output_tasks
            .take_control_error()
            .unwrap()
            .to_string(),
        error.to_string()
    );
    assert!(fixture
        .engine
        .inner
        .output_tasks
        .take_control_error()
        .is_none());
    if stream {
        let delta = fixture.stream.recv().await.unwrap().unwrap();
        assert_eq!(delta.chunk().text, text);
        assert_eq!(delta.chunk().token, Some(token));
        assert!(delta.chunk().finish_reason.is_none());
        assert_eq!(
            fixture
                .stream
                .recv()
                .await
                .unwrap()
                .unwrap_err()
                .to_string(),
            error.to_string()
        );
        drop(delta);
    }
    drop(sender);
    assert!(fixture.stream.recv().await.is_none());
    assert_eq!(fixture.executor.close_call_counts(), (0, 1));
    actual::assert_closed_once(&fixture).await;
    drop(blocker);
    assert_eq!(session.semantic_usage(), 0);
    fixture.engine.shutdown().await.unwrap();
}

#[tokio::test]
async fn semantic_completion_envelope_failure_cancels_real_cpu_before_success_accounting() {
    envelope_failure(false).await;
}

#[tokio::test]
async fn semantic_completion_envelope_failure_preserves_paid_pending_text_and_error() {
    envelope_failure(true).await;
}

async fn exact_reserved_handoff(dual: bool) {
    let mut fixture = actual::fixture(false, false).await;
    let handle = fixture
        .engine
        .inner
        .sequences
        .get(&fixture.request_id)
        .unwrap();
    let (session, text, token) = {
        let mut state = handle.lock();
        let state = state.as_mut().unwrap();
        if !dual {
            drop(state.response_sender.take());
        }
        (
            state.output_semantic.output_session().clone(),
            fixture
                .engine
                .inner
                .tokenizer
                .decode(&state.generated_tokens, true)
                .unwrap(),
            *state.generated_tokens.last().unwrap(),
        )
    };
    let queued = session
        .reserve_queued(session.config().queued_bytes_total)
        .await
        .unwrap();
    fixture
        .engine
        .inner
        .send_stream_update(&fixture.request_id, token)
        .await;
    assert!(matches!(
        fixture.stream.try_recv(),
        Err(mpsc::error::TryRecvError::Empty)
    ));
    let handoff = {
        let mut state = handle.lock();
        fixture
            .engine
            .inner
            .handoff_request_completion_locked(
                &handle,
                &mut state,
                FinishReason::Length,
                None,
                false,
                None,
                CompletionExpectation::Current,
                None,
            )
            .unwrap()
    };
    assert!(handoff.terminal_error.is_none());
    assert!(handoff.completion_error.is_none());
    let draft = handoff.output.draft().unwrap();
    let response_cost = semantic_cost(&draft.response).unwrap();
    let terminal_cost = semantic_cost(&TerminalProjection::new(&draft.response)).unwrap();
    let timestamp = draft.response.created_at;
    if dual {
        assert_eq!(draft.semantic.bytes(), response_cost);
        assert_eq!(
            draft.terminal_copy_credit.as_ref().unwrap().bytes(),
            terminal_cost
        );
        assert_eq!(session.semantic_usage(), response_cost + terminal_cost);
    } else {
        assert!(draft.terminal_copy_credit.is_none());
        assert_eq!(draft.semantic.bytes(), response_cost.max(terminal_cost));
        assert_eq!(session.semantic_usage(), response_cost.max(terminal_cost));
    }
    // Saturate the real session after its synchronous preflight. The following
    // actual close and asynchronous delivery have no capacity for another grant.
    let blocker = session
        .try_reserve_semantic(
            session.config().semantic_bytes_per_session - session.semantic_usage(),
        )
        .unwrap();
    assert_eq!(
        session.semantic_usage(),
        session.config().semantic_bytes_per_session
    );
    let done = fixture
        .engine
        .inner
        .schedule_output_completion(&fixture.request_id, handoff)
        .unwrap();
    drop(queued);
    tokio::time::timeout(Duration::from_secs(3), done)
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    fixture.engine.inner.output_tasks.join_current().await;
    let output = if dual {
        Some(fixture.response.try_recv().unwrap().unwrap())
    } else {
        None
    };
    if let Some(output) = &output {
        assert_eq!(output.response().text, text);
        assert_eq!(output.response().tokens, vec![token]);
        assert_eq!(output.response().usage.prompt_tokens, fixture.prompt_len);
        assert_eq!(output.response().finish_reason, FinishReason::Length);
        assert_eq!(output.response().created_at, timestamp);
    }
    let delta = fixture.stream.recv().await.unwrap().unwrap();
    assert_eq!(delta.chunk().text, text);
    assert_eq!(delta.chunk().token, Some(token));
    assert!(delta.chunk().finish_reason.is_none());
    let terminal = fixture.stream.recv().await.unwrap().unwrap();
    assert!(terminal.chunk().text.is_empty());
    assert_eq!(terminal.chunk().finish_reason, Some(FinishReason::Length));
    assert_eq!(
        terminal.chunk().usage.as_ref().unwrap().prompt_tokens,
        fixture.prompt_len
    );
    assert_eq!(terminal.chunk().created_at, timestamp);
    assert_eq!(semantic_cost(terminal.chunk()).unwrap(), terminal_cost);
    assert_eq!(
        session.semantic_usage(),
        blocker.bytes() + terminal_cost + if dual { response_cost } else { 0 }
    );
    assert!(fixture.stream.recv().await.is_none());
    assert_eq!(fixture.executor.close_call_counts(), (1, 0));
    assert_eq!(
        fixture.engine.inner.scheduler.metrics().completed_requests,
        1
    );
    assert_eq!(fixture.engine.inner.scheduler.metrics().failed_requests, 0);
    assert!(fixture
        .engine
        .inner
        .output_tasks
        .take_control_error()
        .is_none());
    actual::assert_closed_once(&fixture).await;
    drop(blocker);
    drop(delta);
    drop(terminal);
    assert_eq!(
        session.semantic_usage(),
        if dual { response_cost } else { 0 }
    );
    drop(output);
    assert_eq!(session.semantic_usage(), 0);
    fixture.engine.shutdown().await.unwrap();
}

#[tokio::test]
async fn semantic_completion_stream_moves_prepaid_terminal_at_full_session_capacity() {
    exact_reserved_handoff(false).await;
}

#[tokio::test]
async fn semantic_completion_dual_observers_transfer_prepaid_copies_at_full_capacity() {
    exact_reserved_handoff(true).await;
}
