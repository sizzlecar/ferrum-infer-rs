//! Backend-independent acknowledgement scenarios over caller-supplied real
//! native sessions and restore publications. This file is a child of
//! `prefix_cache`, so raw slots and acknowledgement owners stay private.
use super::*;

fn sequence<R: DeviceRuntime>(
    session: Arc<SequenceSession<R>>,
    request_id: RequestId,
    tokens: &[u32],
    issuer: &ExecutorSequenceIssuer,
    worker: &VNextCompletionWorker,
    name: &str,
) -> Arc<VNextSequence<R>> {
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

fn ready_slot<R: DeviceRuntime>(sequence: &Arc<VNextSequence<R>>) -> Arc<VNextPrefillSlot<R>> {
    let span = TokenSpanWork::from_token_ids(&sequence.tokens.lock(), 0..1).unwrap();
    let slot = VNextPrefillSlot::new(
        sequence.request_id().clone(),
        ResourceWorkShape::single(span).unwrap(),
    );
    *slot.state.lock() = VNextPrefillSlotState::Ready(Arc::clone(sequence));
    slot
}

fn product_publication<R: DeviceRuntime>(
    publication: CheckpointRestorePublication<R>,
    slot: Arc<VNextPrefillSlot<R>>,
    sequence: &Arc<VNextSequence<R>>,
    metrics: Arc<PrefixCacheMetrics>,
    prompt_tokens: usize,
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
        prompt_tokens,
        sequence.token_identity.clone(),
        move || observer.acknowledge(&metrics, restored_tokens, || callback.acknowledge()),
    )
    .unwrap()
}

/// Verify publication, opaque incarnation, accounting, and exactly-once
/// acknowledgement before the caller checks real backend continuation.
/// The callbacks must perform actual restore/rejection and numerical checks.
pub fn assert_token_ack_preserves_incarnation<R: DeviceRuntime>(
    session: Arc<SequenceSession<R>>,
    tokens: &[u32],
    boundary: usize,
    restore: impl FnOnce(&Arc<SequenceSession<R>>) -> CheckpointRestorePublication<R>,
    assert_restore_rejected: impl FnOnce(&Arc<SequenceSession<R>>),
    assert_continuation: impl FnOnce(&Arc<SequenceSession<R>>),
) {
    let worker = VNextCompletionWorker::new().unwrap();
    let issuer = ExecutorSequenceIssuer::new();
    let target = sequence(
        session,
        RequestId::new(),
        tokens,
        &issuer,
        &worker,
        "target",
    );
    let slot = ready_slot(&target);
    let metrics = Arc::new(PrefixCacheMetrics::default());
    let publication = restore(&target.session);
    assert_eq!(publication.completed_tokens(), boundary);
    let output = product_publication(
        publication,
        Arc::clone(&slot),
        &target,
        Arc::clone(&metrics),
        tokens.len(),
    );
    assert_eq!(output.token_sequence(), &target.token_identity);
    assert_eq!(target.prefill_tokens_processed.load(Ordering::Acquire), 0);
    assert_eq!(metrics.hits.load(Ordering::Acquire), 0);
    let acknowledged = output.acknowledge_tokens().unwrap();
    assert!(acknowledged.error.is_none());
    let frontier = acknowledged.frontier;
    assert_eq!(frontier.sequence(), &target.token_identity);
    assert_eq!(frontier.consumed_tokens(), boundary);
    assert_eq!(
        target.prefill_tokens_processed.load(Ordering::Acquire),
        boundary
    );
    assert_eq!(metrics.hits.load(Ordering::Acquire), 1);
    assert_eq!(
        metrics.saved_prefill_tokens.load(Ordering::Acquire),
        boundary as u64
    );
    // A consumed publication cannot overwrite its imported frontier, even
    // before the caller runs any continuation work.
    assert_restore_rejected(&target.session);
    assert_continuation(&target.session);
    assert_eq!(metrics.hits.load(Ordering::Acquire), 1);
    target.abort();
    assert!(target.settlement.lock().terminal.is_some());
}

/// Verify failed publication closes only its original incarnation. Supplying
/// a replacement deliberately injects a private slot mismatch over two real
/// sessions; it does not claim public admission permits duplicate live IDs.
pub fn assert_token_ack_failure_preserves_other_incarnation<R: DeviceRuntime>(
    session: Arc<SequenceSession<R>>,
    tokens: &[u32],
    restore: impl FnOnce(&Arc<SequenceSession<R>>) -> CheckpointRestorePublication<R>,
    create_replacement: impl FnOnce() -> Option<Arc<SequenceSession<R>>>,
    assert_restore_rejected: impl FnOnce(&Arc<SequenceSession<R>>),
    assert_replacement_continuation: impl FnOnce(&Arc<SequenceSession<R>>),
) {
    let worker = VNextCompletionWorker::new().unwrap();
    let issuer = ExecutorSequenceIssuer::new();
    let request_id = RequestId::new();
    let target = sequence(
        session,
        request_id.clone(),
        tokens,
        &issuer,
        &worker,
        "old-target",
    );
    let slot = ready_slot(&target);
    let metrics = Arc::new(PrefixCacheMetrics::default());
    let publication = restore(&target.session);
    let output = product_publication(
        publication,
        Arc::clone(&slot),
        &target,
        Arc::clone(&metrics),
        tokens.len(),
    );
    let replacement = create_replacement().map(|session| {
        let replacement = sequence(session, request_id, tokens, &issuer, &worker, "replacement");
        assert_eq!(replacement.request_id(), target.request_id());
        assert_ne!(replacement.token_identity, target.token_identity);
        assert_ne!(
            replacement.session.sequence_authority(),
            target.session.sequence_authority()
        );
        let replacement_slot = ready_slot(&replacement);
        *slot.state.lock() = VNextPrefillSlotState::Ready(Arc::clone(&replacement));
        (replacement, replacement_slot)
    });
    if replacement.is_none() {
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
    assert_restore_rejected(&target.session);
    if let Some((replacement, replacement_slot)) = replacement {
        assert!(replacement.token_frontier_is_open());
        assert!(!replacement_slot.cancelled.load(Ordering::Acquire));
        assert_eq!(
            replacement.prefill_tokens_processed.load(Ordering::Acquire),
            0
        );
        assert_replacement_continuation(&replacement.session);
        replacement.abort();
        assert!(replacement.settlement.lock().terminal.is_some());
    }
}
