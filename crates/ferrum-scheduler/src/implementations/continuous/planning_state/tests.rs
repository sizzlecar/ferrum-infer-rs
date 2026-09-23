use super::*;
use ferrum_interfaces::vnext::DeferredAction;
use ferrum_types::ModelId;

mod maintenance;
mod recompute;

fn scheduler() -> ContinuousBatchScheduler {
    ContinuousBatchScheduler::new(SchedulerConfig::default())
}
fn limit() -> NonZeroUsize {
    NonZeroUsize::new(32).unwrap()
}
fn epochs() -> AdmissionWakeEpochs {
    AdmissionWakeEpochs::new(NonZeroU64::new(9).unwrap(), 0, 0, 0)
}
fn wake() -> AdmissionWakeSnapshot<'static> {
    AdmissionWakeSnapshot::new(epochs(), &[])
}
fn hint() -> BatchHint {
    BatchHint::simple(64)
}
async fn request(s: &ContinuousBatchScheduler, phase: PlanningQueueKind) -> RequestId {
    let request = InferenceRequest::new("prompt", ModelId::new("fixture"))
        .with_metadata(PROMPT_TOKENS_METADATA_KEY, serde_json::json!(16));
    let id = s.submit(request).await.unwrap();
    if phase != PlanningQueueKind::Waiting {
        assert!(s.promote_to_prefill_with_empty_retry(&id, None));
        if matches!(
            phase,
            PlanningQueueKind::Decode | PlanningQueueKind::Preempted
        ) {
            s.mark_prefill_complete(&id, 16);
            if phase == PlanningQueueKind::Preempted {
                s.preempt(id.clone()).await.unwrap();
            }
        }
    }
    id
}
fn selected(
    snapshot: &PlanningQueueSnapshot,
    id: &RequestId,
    action: PlanningWorkAction,
) -> PlanningWorkSelection {
    PlanningWorkSelection {
        key: snapshot
            .requests()
            .iter()
            .find(|r| &r.key.request_id == id)
            .unwrap()
            .key
            .clone(),
        action,
    }
}
fn prefill() -> PlanningWorkAction {
    PlanningWorkAction::Prefill {
        offset: 0,
        count: NonZeroUsize::new(4).unwrap(),
    }
}

#[tokio::test]
async fn snapshot_preserves_all_queues_and_never_schedules_or_truncates() {
    let s = scheduler();
    for phase in [
        PlanningQueueKind::Waiting,
        PlanningQueueKind::Prefill,
        PlanningQueueKind::Decode,
        PlanningQueueKind::Preempted,
    ] {
        request(&s, phase).await;
    }
    let before = s.trace_snapshot();
    let first = s.planning_state(limit(), wake()).unwrap();
    let second = s.planning_state(limit(), wake()).unwrap();
    assert!(first.matches(&second));
    assert_eq!(first.requests.len(), 4);
    assert_eq!(
        first
            .requests
            .iter()
            .filter(|r| r.readiness.ready())
            .count(),
        2
    );
    assert_eq!(before, s.trace_snapshot());
    assert!(matches!(
        s.planning_state(NonZeroUsize::new(3).unwrap(), wake()),
        Err(PlanningStateUnavailable::TooManyRequests {
            actual: 4,
            maximum: 3
        })
    ));
}

#[tokio::test]
async fn selected_mixed_wave_publishes_exact_spans_once_without_admitting_waiters() {
    let s = scheduler();
    let p = request(&s, PlanningQueueKind::Prefill).await;
    let d = request(&s, PlanningQueueKind::Decode).await;
    let w = request(&s, PlanningQueueKind::Waiting).await;
    let snapshot = s.planning_state(limit(), wake()).unwrap();
    let work = [
        selected(&snapshot, &d, PlanningWorkAction::Decode),
        selected(&snapshot, &p, prefill()),
    ];
    let PlanningSelectionOutcome::Published { batch, .. } = s
        .try_select_planned_wave(&snapshot, &work, &hint(), wake())
        .unwrap()
    else {
        panic!("valid selected wave must publish")
    };
    assert_eq!(
        batch
            .requests
            .iter()
            .map(|r| r.request.id.clone())
            .collect::<Vec<_>>(),
        vec![d, p]
    );
    assert_eq!(
        (
            batch.requests[0].tokens_processed,
            batch.requests[0].tokens_to_process
        ),
        (16, Some(1))
    );
    assert_eq!(
        (
            batch.requests[1].tokens_processed,
            batch.requests[1].tokens_to_process
        ),
        (0, Some(4))
    );
    assert_eq!(s.trace_phase(&w), Some(RequestPhase::Waiting));
    assert!(matches!(
        s.try_select_planned_wave(&snapshot, &work, &hint(), wake())
            .unwrap(),
        PlanningSelectionOutcome::Stale
    ));
}

#[tokio::test]
async fn new_ingress_and_unselected_progress_invalidate_the_entire_witness() {
    let s = scheduler();
    let a = request(&s, PlanningQueueKind::Decode).await;
    let b = request(&s, PlanningQueueKind::Decode).await;
    let snapshot = s.planning_state(limit(), wake()).unwrap();
    let work = [selected(&snapshot, &a, PlanningWorkAction::Decode)];
    s.update_decode_progress(&b, 1);
    assert!(matches!(
        s.try_select_planned_wave(&snapshot, &work, &hint(), wake())
            .unwrap(),
        PlanningSelectionOutcome::Stale
    ));
    let snapshot = s.planning_state(limit(), wake()).unwrap();
    let work = [selected(&snapshot, &a, PlanningWorkAction::Decode)];
    request(&s, PlanningQueueKind::Waiting).await;
    assert!(matches!(
        s.try_select_planned_wave(&snapshot, &work, &hint(), wake())
            .unwrap(),
        PlanningSelectionOutcome::Stale
    ));
    assert_eq!(s.current_iteration.load(Ordering::Relaxed), 0);
}

#[tokio::test]
async fn replacing_same_request_id_cannot_reuse_an_old_ticket() {
    let s = scheduler();
    let id = request(&s, PlanningQueueKind::Prefill).await;
    let snapshot = s.planning_state(limit(), wake()).unwrap();
    let work = [selected(&snapshot, &id, prefill())];
    s.cancel(id.clone()).await.unwrap();
    let mut replacement = InferenceRequest::new("prompt", ModelId::new("fixture"))
        .with_metadata(PROMPT_TOKENS_METADATA_KEY, serde_json::json!(16));
    replacement.id = id.clone();
    s.submit(replacement).await.unwrap();
    s.promote_to_prefill_with_empty_retry(&id, None);
    let fresh = s.planning_state(limit(), wake()).unwrap();
    assert_ne!(
        snapshot.requests[0].key.ticket,
        fresh.requests[0].key.ticket
    );
    assert!(matches!(
        s.try_select_planned_wave(&snapshot, &work, &hint(), wake())
            .unwrap(),
        PlanningSelectionOutcome::Stale
    ));
}

#[tokio::test]
async fn readiness_wake_requires_fresh_snapshot_and_never_drops_peer_obligation() {
    let s = scheduler();
    let ready = request(&s, PlanningQueueKind::Decode).await;
    let blocked = request(&s, PlanningQueueKind::Decode).await;
    let receipt = s
        .defer_for_execution_readiness(std::slice::from_ref(&blocked))
        .unwrap();
    let snapshot = s.planning_state(limit(), wake()).unwrap();
    let work = [
        selected(&snapshot, &ready, PlanningWorkAction::Decode),
        selected(&snapshot, &blocked, PlanningWorkAction::Decode),
    ];
    assert!(matches!(
        s.try_select_planned_wave(&snapshot, &work, &hint(), wake())
            .unwrap(),
        PlanningSelectionOutcome::Rejected(_)
    ));
    assert!(snapshot.matches(&s.planning_state(limit(), wake()).unwrap()));
    assert!(receipt.wake().mark_ready());
    assert!(matches!(
        s.try_select_planned_wave(&snapshot, &work, &hint(), wake())
            .unwrap(),
        PlanningSelectionOutcome::Stale
    ));
    let fresh = s.planning_state(limit(), wake()).unwrap();
    assert_eq!(fresh.requests.len(), 2);
    assert!(matches!(
        s.try_select_planned_wave(&fresh, &work, &hint(), wake())
            .unwrap(),
        PlanningSelectionOutcome::Published { .. }
    ));
}

#[tokio::test]
async fn capacity_predicate_uses_exact_sources_and_final_wake_fence() {
    let s = scheduler();
    let id = request(&s, PlanningQueueKind::Decode).await;
    let source = CapacityAvailabilitySource::ActiveSequenceSlots;
    let observed = [CapacityAvailabilityEpoch::new(source, 1).unwrap()];
    let condition = CapacityWaitCondition::from_observation(9, observed.to_vec()).unwrap();
    s.decode_queue
        .write()
        .requests
        .get_mut(&id)
        .unwrap()
        .execution_capacity_deferral = Some(AdmissionDeferral::new(
        DeferredAction::WaitForRelease,
        epochs(),
        condition,
    ));
    let old_wake = AdmissionWakeSnapshot::new(epochs(), &observed);
    let snapshot = s.planning_state(limit(), old_wake).unwrap();
    assert!(snapshot.requests[0].readiness.capacity_blocked);
    let work = [selected(&snapshot, &id, PlanningWorkAction::Decode)];
    let changed = [CapacityAvailabilityEpoch::new(source, 2).unwrap()];
    let new_wake = AdmissionWakeSnapshot::new(epochs(), &changed);
    assert!(matches!(
        s.try_select_planned_wave(&snapshot, &work, &hint(), new_wake)
            .unwrap(),
        PlanningSelectionOutcome::Stale
    ));
    let fresh = s.planning_state(limit(), new_wake).unwrap();
    assert!(fresh.requests[0].readiness.ready());
    assert!(matches!(
        s.try_select_planned_wave(&fresh, &work, &hint(), new_wake)
            .unwrap(),
        PlanningSelectionOutcome::Published { .. }
    ));
}

#[tokio::test]
async fn invalid_selection_leaves_every_frontier_and_iteration_unchanged() {
    let s = scheduler();
    let id = request(&s, PlanningQueueKind::Prefill).await;
    let snapshot = s.planning_state(limit(), wake()).unwrap();
    for action in [
        PlanningWorkAction::Decode,
        PlanningWorkAction::Prefill {
            offset: 1,
            count: NonZeroUsize::new(4).unwrap(),
        },
        PlanningWorkAction::Prefill {
            offset: 0,
            count: NonZeroUsize::new(17).unwrap(),
        },
        PlanningWorkAction::Prefill {
            offset: usize::MAX,
            count: NonZeroUsize::new(4).unwrap(),
        },
    ] {
        let work = [selected(&snapshot, &id, action)];
        assert!(matches!(
            s.try_select_planned_wave(&snapshot, &work, &hint(), wake())
                .unwrap(),
            PlanningSelectionOutcome::Rejected(_)
        ));
        assert!(snapshot.matches(&s.planning_state(limit(), wake()).unwrap()));
    }
    let choice = selected(&snapshot, &id, prefill());
    assert!(matches!(
        s.try_select_planned_wave(&snapshot, &[choice.clone(), choice], &hint(), wake())
            .unwrap(),
        PlanningSelectionOutcome::Rejected(_)
    ));
    let mut narrow = hint();
    narrow.max_tokens = 3;
    assert!(matches!(
        s.try_select_planned_wave(
            &snapshot,
            &[selected(&snapshot, &id, prefill())],
            &narrow,
            wake()
        )
        .unwrap(),
        PlanningSelectionOutcome::Rejected(_)
    ));
    assert!(snapshot.matches(&s.planning_state(limit(), wake()).unwrap()));
}

#[tokio::test]
async fn foreign_owner_and_lock_contention_grant_nothing() {
    let s = scheduler();
    let other = scheduler();
    let id = request(&s, PlanningQueueKind::Decode).await;
    let snapshot = s.planning_state(limit(), wake()).unwrap();
    assert!(matches!(
        other
            .try_select_planned_wave(
                &snapshot,
                &[selected(&snapshot, &id, PlanningWorkAction::Decode)],
                &hint(),
                wake()
            )
            .unwrap(),
        PlanningSelectionOutcome::Stale
    ));
    let guard = s.prefill_queue.write();
    assert!(matches!(
        s.planning_state(limit(), wake()),
        Err(PlanningStateUnavailable::Busy)
    ));
    assert!(
        s.waiting_queue.try_write().is_some(),
        "partial lock chain must be dropped"
    );
    drop(guard);
}

#[tokio::test]
async fn recompute_retains_logical_generation_but_invalidates_old_resident_witness() {
    let s = scheduler();
    let id = request(&s, PlanningQueueKind::Decode).await;
    s.update_decode_progress(&id, 3);
    let snapshot = s.planning_state(limit(), wake()).unwrap();
    assert!(s.defer_decode_to_waiting_for_capacity(&id, 1));
    let fresh = s.planning_state(limit(), wake()).unwrap();
    assert_eq!(
        fresh.requests[0].key.generation,
        snapshot.requests[0].key.generation
    );
    assert_eq!(fresh.requests[0].committed_output_tokens, 3);
    assert_eq!(fresh.requests[0].resident_tokens, 0);
    assert_eq!(fresh.requests[0].recompute_target_tokens, Some(19));
    assert!(matches!(
        s.try_select_planned_wave(
            &snapshot,
            &[selected(&snapshot, &id, PlanningWorkAction::Decode)],
            &hint(),
            wake()
        )
        .unwrap(),
        PlanningSelectionOutcome::Stale
    ));
}

#[tokio::test]
async fn missing_prompt_evidence_is_not_an_estimated_chunk_authorization() {
    let s = scheduler();
    let id = s
        .submit(InferenceRequest::new("prompt", ModelId::new("fixture")))
        .await
        .unwrap();
    s.promote_to_prefill_with_empty_retry(&id, None);
    let snapshot = s.planning_state(limit(), wake()).unwrap();
    assert_eq!(snapshot.requests[0].prompt_tokens, None);
    assert_eq!(snapshot.requests[0].prefill_context_tokens, None);
    assert!(matches!(
        s.try_select_planned_wave(
            &snapshot,
            &[selected(&snapshot, &id, prefill())],
            &hint(),
            wake()
        )
        .unwrap(),
        PlanningSelectionOutcome::Rejected(_)
    ));
}

#[tokio::test]
async fn iteration_exhaustion_never_wraps_or_mutates_work() {
    let s = scheduler();
    let id = request(&s, PlanningQueueKind::Decode).await;
    s.current_iteration.store(u64::MAX, Ordering::Relaxed);
    let snapshot = s.planning_state(limit(), wake()).unwrap();
    assert!(matches!(
        s.try_select_planned_wave(
            &snapshot,
            &[selected(&snapshot, &id, PlanningWorkAction::Decode)],
            &hint(),
            wake()
        ),
        Err(PlanningStateUnavailable::CounterExhausted)
    ));
    assert!(snapshot.matches(&s.planning_state(limit(), wake()).unwrap()));
}

#[tokio::test]
async fn fresh_snapshot_cannot_repeat_unfinished_work_and_unsubmitted_release_is_exact() {
    let s = scheduler();
    let id = request(&s, PlanningQueueKind::Decode).await;
    let snapshot = s.planning_state(limit(), wake()).unwrap();
    let work = [selected(&snapshot, &id, PlanningWorkAction::Decode)];
    let PlanningSelectionOutcome::Published { receipt, .. } = s
        .try_select_planned_wave(&snapshot, &work, &hint(), wake())
        .unwrap()
    else {
        panic!("initial publication")
    };
    let fresh = s.planning_state(limit(), wake()).unwrap();
    assert!(fresh.requests[0].readiness.unfinished_work);
    assert!(matches!(
        s.try_select_planned_wave(&fresh, &work, &hint(), wake())
            .unwrap(),
        PlanningSelectionOutcome::Rejected(_)
    ));
    let lock = s.prefill_queue.write();
    assert!(matches!(
        s.release_unsubmitted_planned_wave(&receipt),
        Err(PlanningStateUnavailable::Busy)
    ));
    drop(lock);
    assert_eq!(
        s.release_unsubmitted_planned_wave(&receipt)
            .unwrap()
            .released_rows,
        1
    );
    assert_eq!(
        s.release_unsubmitted_planned_wave(&receipt)
            .unwrap()
            .released_rows,
        0
    );
    let fresh = s.planning_state(limit(), wake()).unwrap();
    assert!(fresh.requests[0].readiness.ready());
    let PlanningSelectionOutcome::Published {
        receipt: second, ..
    } = s
        .try_select_planned_wave(&fresh, &work, &hint(), wake())
        .unwrap()
    else {
        panic!("retry after proven no submission")
    };
    assert!(
        s.release_unsubmitted_planned_wave(&receipt)
            .unwrap()
            .released_rows
            == 0,
        "old receipt cannot release a newer wave"
    );
    s.update_decode_progress(&id, 1);
    assert!(
        s.release_unsubmitted_planned_wave(&second)
            .unwrap()
            .released_rows
            == 0,
        "committed work cannot be withdrawn"
    );
    assert_eq!(
        s.planning_state(limit(), wake()).unwrap().requests[0].committed_output_tokens,
        1
    );
}

#[tokio::test]
async fn unsubmitted_release_clears_surviving_peer_without_touching_reused_identity() {
    let s = scheduler();
    let cancelled = request(&s, PlanningQueueKind::Decode).await;
    let healthy = request(&s, PlanningQueueKind::Decode).await;
    let snapshot = s.planning_state(limit(), wake()).unwrap();
    let work = [
        selected(&snapshot, &cancelled, PlanningWorkAction::Decode),
        selected(&snapshot, &healthy, PlanningWorkAction::Decode),
    ];
    let PlanningSelectionOutcome::Published { receipt, .. } = s
        .try_select_planned_wave(&snapshot, &work, &hint(), wake())
        .unwrap()
    else {
        panic!("publish cohort");
    };
    s.cancel(cancelled.clone()).await.unwrap();
    let mut replacement = InferenceRequest::new("replacement", ModelId::new("fixture"))
        .with_metadata(PROMPT_TOKENS_METADATA_KEY, serde_json::json!(16));
    replacement.id = cancelled.clone();
    s.submit(replacement).await.unwrap();
    assert!(s.promote_to_prefill_with_empty_retry(&cancelled, None));
    s.mark_prefill_complete(&cancelled, 16);
    let current = s.planning_state(limit(), wake()).unwrap();
    let PlanningSelectionOutcome::Published {
        receipt: replacement_receipt,
        ..
    } = s
        .try_select_planned_wave(
            &current,
            &[selected(&current, &cancelled, PlanningWorkAction::Decode)],
            &hint(),
            wake(),
        )
        .unwrap()
    else {
        panic!("publish replacement");
    };
    let result = s.release_unsubmitted_planned_wave(&receipt).unwrap();
    assert_eq!(
        result,
        PlanningPublicationRelease {
            released_rows: 1,
            superseded_rows: 1
        }
    );
    let after = s.planning_state(limit(), wake()).unwrap();
    assert!(after
        .requests()
        .iter()
        .find(|row| row.key.request_id == healthy)
        .unwrap()
        .readiness
        .ready());
    assert!(
        after
            .requests()
            .iter()
            .find(|row| row.key.request_id == cancelled)
            .unwrap()
            .readiness
            .unfinished_work
    );
    assert_eq!(
        s.release_unsubmitted_planned_wave(&replacement_receipt)
            .unwrap()
            .released_rows,
        1
    );
}

#[tokio::test]
async fn snapshot_and_rejected_span_never_release_actual_prefix_dependency() {
    use ferrum_interfaces::{
        model_executor::PrefixCapturePlan, vnext::CheckpointTokenSpanConstraint,
    };
    let s = scheduler();
    let source = request(&s, PlanningQueueKind::Prefill).await;
    let follower = request(&s, PlanningQueueKind::Waiting).await;
    let candidates = s.prefix_rendezvous_candidates();
    let source_key = &candidates
        .iter()
        .find(|c| c.key.request_id() == &source)
        .unwrap()
        .key;
    let follower_key = candidates
        .iter()
        .find(|c| c.key.request_id() == &follower)
        .unwrap()
        .key
        .clone();
    let hold = s
        .hold_prefix_followers(
            source_key,
            &[follower_key],
            PrefixCapturePlan {
                boundary: 16,
                span: CheckpointTokenSpanConstraint::new(
                    NonZeroU64::new(8).unwrap(),
                    NonZeroU64::new(4).unwrap(),
                )
                .unwrap(),
            },
        )
        .unwrap();
    let snapshot = s.planning_state(limit(), wake()).unwrap();
    assert!(hold.is_pending());
    assert!(
        snapshot
            .requests
            .iter()
            .find(|r| r.key.request_id == follower)
            .unwrap()
            .readiness
            .prefix_blocked
    );
    assert!(matches!(
        s.try_select_planned_wave(
            &snapshot,
            &[selected(&snapshot, &source, prefill())],
            &hint(),
            wake()
        )
        .unwrap(),
        PlanningSelectionOutcome::Rejected(_)
    ));
    assert!(
        hold.is_pending(),
        "the old mutating cap() would release this undersized span"
    );
    assert!(snapshot.matches(&s.planning_state(limit(), wake()).unwrap()));
}

#[test]
fn prefix_projection_checks_exact_granules_without_releasing_dependencies() {
    assert!(selection::prefix_span_valid(Some((16, 4, 8)), 0, 8));
    assert!(selection::prefix_span_valid(Some((16, 4, 8)), 0, 16));
    assert!(!selection::prefix_span_valid(Some((16, 4, 8)), 0, 4));
    assert!(!selection::prefix_span_valid(Some((16, 4, 8)), 0, 12));
    assert!(!selection::prefix_span_valid(Some((16, 4, 8)), 0, 20));
}

#[tokio::test]
async fn abandoned_optional_restore_allows_cold_compute_after_reservation_drops() {
    let s = scheduler();
    let id = request(&s, PlanningQueueKind::Prefill).await;
    let prepared = s.prepare_prefix_restore(&id, 0, 16).unwrap().unwrap();
    s.abandon_prefix_restore_capacity(&prepared).unwrap();
    assert!(
        s.planning_state(limit(), wake()).unwrap().requests[0]
            .readiness
            .prefix_blocked,
        "the live restore reservation still owns its gate"
    );
    drop(prepared);
    assert!(
        s.prepare_prefix_restore(&id, 0, 16).unwrap().is_none(),
        "abandoned restore cannot be attempted again"
    );
    let snapshot = s.planning_state(limit(), wake()).unwrap();
    assert!(
        snapshot.requests[0].readiness.ready(),
        "cold compute remains legal"
    );
    assert!(matches!(
        s.try_select_planned_wave(
            &snapshot,
            &[selected(&snapshot, &id, prefill())],
            &hint(),
            wake()
        )
        .unwrap(),
        PlanningSelectionOutcome::Published { .. }
    ));
}

#[tokio::test]
async fn reached_output_limit_remains_visible_but_cannot_publish_another_decode() {
    let s = scheduler();
    let request = InferenceRequest::new("prompt", ModelId::new("fixture"))
        .with_sampling_params(ferrum_types::SamplingParams {
            max_tokens: 1,
            ..Default::default()
        })
        .with_metadata(PROMPT_TOKENS_METADATA_KEY, serde_json::json!(16));
    let id = s.submit(request).await.unwrap();
    assert!(s.next_batch(hint()).await.is_some());
    s.mark_prefill_complete(&id, 16);
    s.update_decode_progress(&id, 1);

    let snapshot = s.planning_state(limit(), wake()).unwrap();
    assert_eq!(
        snapshot.requests().len(),
        1,
        "terminal cleanup has not removed this obligation"
    );
    let row = &snapshot.requests()[0];
    assert_eq!(row.committed_output_tokens, row.maximum_output_tokens);
    assert!(row.readiness.output_limit_reached);
    assert!(!row.readiness.ready());
    assert!(matches!(
        s.try_select_planned_wave(
            &snapshot,
            &[selected(&snapshot, &id, PlanningWorkAction::Decode)],
            &hint(),
            wake()
        )
        .unwrap(),
        PlanningSelectionOutcome::Rejected(_)
    ));
    assert!(
        snapshot.matches(&s.planning_state(limit(), wake()).unwrap()),
        "rejecting extra work must not retire the request or mutate its frontier"
    );
}
