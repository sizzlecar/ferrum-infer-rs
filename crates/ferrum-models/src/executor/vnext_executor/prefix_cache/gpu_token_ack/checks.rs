//! Actual GPU checkpoint copies through the production product acknowledgement.
//! Continuation uses the retained native numerical fixture, not the Engine loop.
use super::*;

const BOUNDARY: usize = 129;
const INPUT: usize = 138;

fn tokens() -> Arc<[u32]> {
    (0..INPUT)
        .map(|index| ((index * 7 + 3) % 32) as u32)
        .collect()
}

fn sequence(
    session: Arc<SequenceSession<Runtime>>,
    request_id: RequestId,
    tokens: &[u32],
    issuer: &ExecutorSequenceIssuer,
    worker: &VNextCompletionWorker,
    name: &str,
) -> Arc<VNextSequence<Runtime>> {
    let request =
        VNextRequestRoot::bind_initial(request_id, session.resources().request_id(), &session)
            .unwrap();
    let active_binding = Arc::new(TrustedActiveSequenceBinding::from_session(&session).unwrap());
    let token_identity = issuer.issue_sequence().unwrap();
    Arc::new(VNextSequence {
        token_recovery_key: issuer.recovery_key(&token_identity).unwrap(),
        token_identity,
        token_issuer: issuer.clone(),
        prefix_capture_interests: Mutex::new(Vec::new()),
        cache_id: name.to_owned(),
        request,
        session,
        active_binding,
        request_origin: ExecutorRequestOrigin::Product,
        tokens: Mutex::new(tokens.to_vec()),
        maximum_tokens: tokens.len(),
        active: AtomicBool::new(true),
        settlement: Mutex::new(SequenceSettlement::default()),
        recovery_wake: worker.recovery_wake(),
        operation: AsyncMutex::new(()),
        events: None,
        product_prompt_tokens: tokens.len() as u64,
        replayed_output_tokens: 0,
        prefill_tokens_processed: AtomicUsize::new(0),
    })
}

fn ready_slot(sequence: &Arc<VNextSequence<Runtime>>) -> Arc<VNextPrefillSlot<Runtime>> {
    let span = TokenSpanWork::from_token_ids(&sequence.tokens.lock(), 0..1).unwrap();
    let slot = VNextPrefillSlot::new(
        sequence.request_id().clone(),
        ResourceWorkShape::single(span).unwrap(),
    );
    *slot.state.lock() = VNextPrefillSlotState::Ready(Arc::clone(sequence));
    slot
}

fn product_publication(
    publication: CheckpointRestorePublication<Runtime>,
    slot: Arc<VNextPrefillSlot<Runtime>>,
    sequence: &Arc<VNextSequence<Runtime>>,
    metrics: Arc<PrefixCacheMetrics>,
) -> PlanRuntimePrefixRestoreOutput {
    let restored_tokens = publication.completed_tokens();
    let callback = RestoreAcknowledgement {
        publication: Some(publication),
        slot,
        sequence: Arc::clone(sequence),
        acknowledged: false,
    };
    let observer = PrefixRestoreObserver::new(
        None,
        sequence.request_id().clone(),
        PrefixRestoreSource::Index,
    );
    PlanRuntimePrefixRestoreOutput::new_tokens(
        sequence.request_id().clone(),
        restored_tokens,
        INPUT,
        sequence.token_identity.clone(),
        move || observer.acknowledge(&metrics, restored_tokens, || callback.acknowledge()),
    )
    .unwrap()
}

fn execute_prefix(fixture: &Fixture, session: &Arc<SequenceSession<Runtime>>, tokens: &Arc<[u32]>) {
    fixture
        .execute(session, Arc::clone(tokens), 0..128)
        .assert_state_nonzero();
    fixture
        .execute(session, Arc::clone(tokens), 128..BOUNDARY)
        .assert_state_nonzero();
}

#[test]
fn native_prefix_tokens_ack_preserves_incarnation_and_gpu_continuation() {
    let fixture = Fixture::new(AttentionKind::CausalInt8);
    let tokens = tokens();
    let source = fixture.admit("token-ack-source", Arc::clone(&tokens));
    let cold = fixture.admit("token-ack-cold", Arc::clone(&tokens));
    execute_prefix(&fixture, &source, &tokens);
    execute_prefix(&fixture, &cold, &tokens);
    let checkpoint = fixture.capture(&source);
    assert_eq!(checkpoint.completed_tokens(), BOUNDARY);
    source.try_abort_if_quiescent().unwrap();
    drop(source);
    let target = fixture.admit("token-ack-target", Arc::clone(&tokens));
    let worker = VNextCompletionWorker::new().unwrap();
    let issuer = ExecutorSequenceIssuer::new();
    let target = sequence(
        target,
        RequestId::new(),
        &tokens,
        &issuer,
        &worker,
        "target",
    );
    let slot = ready_slot(&target);
    let metrics = Arc::new(PrefixCacheMetrics::default());
    // This waits for real device copy completion and checks that native Step
    // admission is still rejected before returning the publication owner.
    let publication =
        fixture.restore_publication(&target.session, &checkpoint, Arc::clone(&tokens));
    let output = product_publication(
        publication,
        Arc::clone(&slot),
        &target,
        Arc::clone(&metrics),
    );
    assert_eq!(output.token_sequence(), &target.token_identity);
    assert_eq!(target.prefill_tokens_processed.load(Ordering::Acquire), 0);
    assert_eq!(metrics.hits.load(Ordering::Acquire), 0);
    let frontier = output.acknowledge_tokens().unwrap();
    assert_eq!(frontier.sequence(), &target.token_identity);
    assert_eq!(frontier.consumed_tokens(), BOUNDARY);
    assert_eq!(
        target.prefill_tokens_processed.load(Ordering::Acquire),
        BOUNDARY
    );
    assert_eq!(metrics.hits.load(Ordering::Acquire), 1);
    assert_eq!(
        metrics.saved_prefill_tokens.load(Ordering::Acquire),
        BOUNDARY as u64
    );
    // Acknowledgement consumes the publication owner. A fresh restore attempt
    // also cannot overwrite the now-imported frontier, even before continuation.
    fixture.assert_restore_rejected(&target.session, &checkpoint, Arc::clone(&tokens));
    for range in [BOUNDARY..137, 137..INPUT] {
        let expected = fixture.execute(&cold, Arc::clone(&tokens), range.clone());
        let actual = fixture.execute(&target.session, Arc::clone(&tokens), range);
        expected.assert_same(&actual, "token-ack restored output/payload/scales");
    }
    assert_eq!(metrics.hits.load(Ordering::Acquire), 1);
    cold.try_complete().unwrap();
    target.abort();
    assert!(target.settlement.lock().terminal.is_some());
}

#[test]
fn native_prefix_tokens_ack_failure_cancels_only_the_original_target() {
    let fixture = Fixture::new(AttentionKind::CausalInt8);
    let tokens = tokens();
    let source = fixture.admit("token-ack-failure-source", Arc::clone(&tokens));
    execute_prefix(&fixture, &source, &tokens);
    let checkpoint = fixture.capture(&source);
    let expected = fixture.execute(&source, Arc::clone(&tokens), BOUNDARY..INPUT);
    source.try_complete().unwrap();
    drop(source);
    let worker = VNextCompletionWorker::new().unwrap();
    let issuer = ExecutorSequenceIssuer::new();
    for replaced in [false, true] {
        let request_id = RequestId::new();
        let target = sequence(
            fixture.admit(&format!("token-ack-old-{replaced}"), Arc::clone(&tokens)),
            request_id.clone(),
            &tokens,
            &issuer,
            &worker,
            "old-target",
        );
        let slot = ready_slot(&target);
        let metrics = Arc::new(PrefixCacheMetrics::default());
        let publication =
            fixture.restore_publication(&target.session, &checkpoint, Arc::clone(&tokens));
        let output = product_publication(
            publication,
            Arc::clone(&slot),
            &target,
            Arc::clone(&metrics),
        );
        let replacement = replaced.then(|| {
            let replacement = sequence(
                fixture.admit("token-ack-replacement", Arc::clone(&tokens)),
                request_id,
                &tokens,
                &issuer,
                &worker,
                "replacement",
            );
            assert_eq!(replacement.request_id(), target.request_id());
            assert_ne!(replacement.token_identity, target.token_identity);
            assert_ne!(
                replacement.session.sequence_authority(),
                target.session.sequence_authority()
            );
            let replacement_slot = ready_slot(&replacement);
            // Adversarial slot-owner mismatch over two real native sessions.
            // This does not claim the normal registry admits duplicate request
            // ids while an earlier incarnation still owns a publication.
            *slot.state.lock() = VNextPrefillSlotState::Ready(Arc::clone(&replacement));
            (replacement, replacement_slot)
        });
        if !replaced {
            target.abort();
            assert!(
                target.settlement.lock().terminal.is_none(),
                "publication still owns its target gate"
            );
        }
        assert!(output.acknowledge_tokens().is_err());
        assert!(target.settlement.lock().terminal.is_some());
        assert_eq!(target.prefill_tokens_processed.load(Ordering::Acquire), 0);
        assert_eq!(metrics.hits.load(Ordering::Acquire), 0);
        assert_eq!(metrics.saved_prefill_tokens.load(Ordering::Acquire), 0);
        fixture.assert_restore_rejected(&target.session, &checkpoint, Arc::clone(&tokens));
        if let Some((replacement, replacement_slot)) = replacement {
            assert!(replacement.token_frontier_is_open());
            assert!(!replacement_slot.cancelled.load(Ordering::Acquire));
            assert_eq!(
                replacement.prefill_tokens_processed.load(Ordering::Acquire),
                0
            );
            execute_prefix(&fixture, &replacement.session, &tokens);
            expected.assert_same(
                &fixture.execute(&replacement.session, Arc::clone(&tokens), BOUNDARY..INPUT),
                "same-request replacement was not cancelled or given old state",
            );
            replacement.abort();
            assert!(replacement.settlement.lock().terminal.is_some());
        }
    }
}
