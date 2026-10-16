use super::*;

#[tokio::test]
async fn plan_runtime_cohort_stream_preserves_bounded_backpressure_and_receiver_close() {
    for close_blocked_receiver in [false, true] {
        let (engine, scheduler, _executor, tokenizer) =
            plan_runtime_batch_decode_test_engine(PlanRuntimeBatchDecodeBehavior::Exact);
        let (request_ids, _, _) =
            install_plan_runtime_decode_cohort(&engine, &scheduler, tokenizer.clone()).await;
        let batch = scheduler
            .next_batch(ferrum_interfaces::BatchHint::simple(request_ids.len()))
            .await
            .unwrap();
        let first_id = batch.requests[0].request.id.clone();
        let blocked_id = batch.requests[1].request.id.clone();
        let mut receivers = HashMap::new();
        {
            let mut sequences = engine.inner.sequences.write();
            for request_id in &request_ids {
                let (sender, receiver) = tokio::sync::mpsc::channel(1);
                sequences.get_mut(request_id).unwrap().stream_sender = Some(sender);
                receivers.insert(request_id.clone(), receiver);
            }
        }
        let mut first_receiver = receivers.remove(&first_id).unwrap();
        let mut blocked_receiver = receivers.remove(&blocked_id).unwrap();

        // Leave the second request's first delta in its capacity-one queue.
        engine.inner.process_batch(&batch).await.unwrap();
        let first_delta = first_receiver.recv().await.unwrap().unwrap();
        assert_eq!(first_delta.token, Some(TokenId::new(6)));
        assert!(first_delta.finish_reason.is_none());

        let process = engine.inner.process_batch(&batch);
        tokio::pin!(process);
        assert!(futures::poll!(process.as_mut()).is_pending());

        // The first request's next delta is already visible while its peer's
        // bounded send is blocked; no cohort-wide message buffer is involved.
        let second_delta = first_receiver.try_recv().unwrap().unwrap();
        assert_eq!(second_delta.token, Some(TokenId::new(6)));
        assert_eq!(second_delta.text, "ok");
        assert!(second_delta.finish_reason.is_none());
        {
            let sequences = engine
                .inner
                .sequences
                .try_write()
                .expect("a blocked stream send must not retain sequence locks");
            for request_id in &request_ids {
                let sequence = sequences.get(request_id).unwrap();
                assert_eq!(sequence.generated_tokens.len(), 3);
                assert_eq!(sequence.generated_tokens.last(), Some(&TokenId::new(6)));
            }
        }
        let queued_delta = blocked_receiver.try_recv().unwrap().unwrap();
        assert_eq!(queued_delta.token, Some(TokenId::new(6)));
        assert!(queued_delta.finish_reason.is_none());

        if close_blocked_receiver {
            drop(blocked_receiver);
            process.await.unwrap();
            assert!(engine
                .inner
                .sequences
                .read()
                .get(&blocked_id)
                .unwrap()
                .client_receiver_closed());
        } else {
            process.await.unwrap();
            let resumed_delta = blocked_receiver.recv().await.unwrap().unwrap();
            assert_eq!(resumed_delta.text, "ok");
            assert_eq!(resumed_delta.token, Some(TokenId::new(6)));
            assert!(resumed_delta.finish_reason.is_none());
            assert!(matches!(
                blocked_receiver.try_recv(),
                Err(tokio::sync::mpsc::error::TryRecvError::Empty)
            ));
        }
        assert!(matches!(
            first_receiver.try_recv(),
            Err(tokio::sync::mpsc::error::TryRecvError::Empty)
        ));
    }
}

#[tokio::test]
async fn plan_runtime_cohort_stream_preserves_terminal_order_and_live_peer() {
    let (engine, scheduler, executor, tokenizer) =
        plan_runtime_batch_decode_test_engine(PlanRuntimeBatchDecodeBehavior::Exact);
    let (request_ids, initial_tokens, _) =
        install_plan_runtime_decode_cohort(&engine, &scheduler, tokenizer.clone()).await;
    let batch = scheduler
        .next_batch(ferrum_interfaces::BatchHint::simple(request_ids.len()))
        .await
        .unwrap();
    let live_id = batch.requests[0].request.id.clone();
    let terminal_id = batch.requests[1].request.id.clone();
    let terminal_initial = initial_tokens[request_ids
        .iter()
        .position(|request_id| request_id == &terminal_id)
        .unwrap()];
    let expected_terminal_text = tokenizer
        .decode(&[terminal_initial, TokenId::new(6)], true)
        .unwrap();
    let (live_sender, mut live_receiver) = tokio::sync::mpsc::channel(1);
    let (terminal_sender, mut terminal_receiver) = tokio::sync::mpsc::channel(1);
    {
        let mut sequences = engine.inner.sequences.write();
        sequences.get_mut(&live_id).unwrap().stream_sender = Some(live_sender);
        let terminal = sequences.get_mut(&terminal_id).unwrap();
        terminal.stream_sender = Some(terminal_sender);
        terminal.sampling_params.max_tokens = 2;
    }

    // Receiving text makes room for the independent terminal event. The
    // continuing peer must already be observable before that queue is drained.
    let receive = async {
        let live = live_receiver.recv().await.unwrap().unwrap();
        assert_eq!(live.request_id, live_id);
        assert_eq!(live.token, Some(TokenId::new(6)));
        assert!(live.finish_reason.is_none());
        let text = terminal_receiver.recv().await.unwrap().unwrap();
        assert_eq!(text.text, expected_terminal_text);
        assert_eq!(text.token, Some(TokenId::new(6)));
        assert!(text.finish_reason.is_none());
        assert!(text.usage.is_none());
        let terminal = terminal_receiver.recv().await.unwrap().unwrap();
        assert!(terminal.text.is_empty());
        assert!(terminal.token.is_none());
        assert_eq!(terminal.finish_reason, Some(FinishReason::Length));
        let usage = terminal.usage.unwrap();
        assert_eq!(usage.prompt_tokens, 1);
        assert_eq!(usage.completion_tokens, 2);
        assert_eq!(usage.total_tokens, 3);
        assert!(terminal_receiver.recv().await.is_none());
    };
    let (result, ()) = tokio::time::timeout(Duration::from_secs(2), async {
        tokio::join!(engine.inner.process_batch(&batch), receive)
    })
    .await
    .expect("bounded terminal delivery must finish when its receiver drains");
    result.unwrap();
    let sequences = engine.inner.sequences.read();
    assert!(!sequences.contains_key(&terminal_id));
    assert_eq!(sequences.get(&live_id).unwrap().generated_tokens.len(), 2);
    assert_eq!(executor.released_cache_count.load(Ordering::Acquire), 1);
    assert_eq!(scheduler.active_count(), 1);
}
