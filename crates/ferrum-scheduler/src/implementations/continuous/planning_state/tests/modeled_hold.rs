use super::*;

fn hold(
    s: &ContinuousBatchScheduler,
    source: &RequestId,
    target: &RequestId,
) -> PrefixRendezvousHold {
    let candidates = s.prefix_rendezvous_candidates();
    let key = |id: &RequestId| {
        &candidates
            .iter()
            .find(|c| c.key.request_id() == id)
            .unwrap()
            .key
    };
    s.hold_admitted_prefix_follower(
        key(source),
        key(target),
        ferrum_interfaces::model_executor::PrefixCapturePlan {
            boundary: 8,
            span: ferrum_interfaces::vnext::CheckpointTokenSpanConstraint::new(
                NonZeroU64::MIN,
                NonZeroU64::MIN,
            )
            .unwrap(),
        },
    )
    .unwrap()
}

#[tokio::test]
async fn modeled_hold_requires_actual_seal_and_rejects_unrelated_or_restore_gates() {
    let s = scheduler();
    let source = request(&s, PlanningQueueKind::Prefill).await;
    let target = request(&s, PlanningQueueKind::Prefill).await;
    let other_source = request(&s, PlanningQueueKind::Prefill).await;
    let other_target = request(&s, PlanningQueueKind::Prefill).await;
    let owned = hold(&s, &source, &target);
    let unrelated = hold(&s, &other_source, &other_target);
    let snapshot = s.planning_state(limit(), wake()).unwrap();
    let row = |id: &RequestId| {
        snapshot
            .requests()
            .iter()
            .find(|r| &r.key.request_id == id)
            .unwrap()
    };
    let before = s.trace_snapshot();
    assert!(row(&target).is_modeled_rendezvous_follower(row(&source), &owned));
    assert!(!row(&target).readiness.ready());
    assert!(!row(&other_target).is_modeled_rendezvous_follower(row(&source), &owned));
    assert!(!row(&target).is_modeled_rendezvous_follower(row(&other_source), &unrelated));
    assert!(!row(&target).is_modeled_rendezvous_follower(row(&other_source), &owned));
    assert_eq!(before, s.trace_snapshot());

    // A caller-editable readiness field cannot manufacture a private seal.
    let mut spoofed = row(&source).clone();
    spoofed.readiness.prefix_blocked = true;
    spoofed.key = row(&target).key.clone();
    assert!(!spoofed.is_modeled_rendezvous_follower(row(&source), &owned));

    // A real restore reservation on the held target remains independently
    // unmodeled; the presence of a genuine rendezvous cannot hide that gate.
    let restore = s.prepare_prefix_restore(&target, 0, 16).unwrap().unwrap();
    let restoring = s.planning_state(limit(), wake()).unwrap();
    let target_row = restoring
        .requests()
        .iter()
        .find(|r| r.key.request_id == target)
        .unwrap();
    let source_row = restoring
        .requests()
        .iter()
        .find(|r| r.key.request_id == source)
        .unwrap();
    assert!(target_row.readiness.prefix_blocked);
    assert!(!target_row.is_modeled_rendezvous_follower(source_row, &owned));
    drop(restore);
    owned.release();
    assert!(!row(&target).is_modeled_rendezvous_follower(row(&source), &owned));
}

#[tokio::test]
async fn modeled_hold_rejects_restore_only_and_old_admission_snapshot() {
    let s = scheduler();
    let source = request(&s, PlanningQueueKind::Prefill).await;
    let target = request(&s, PlanningQueueKind::Prefill).await;
    let before = s.planning_state(limit(), wake()).unwrap();
    let owned = hold(&s, &source, &target);
    let current = s.planning_state(limit(), wake()).unwrap();
    let source_row = current
        .requests()
        .iter()
        .find(|r| r.key.request_id == source)
        .unwrap();
    let old_target = before
        .requests()
        .iter()
        .find(|r| r.key.request_id == target)
        .unwrap();
    assert!(!old_target.is_modeled_rendezvous_follower(source_row, &owned));
    let other = request(&s, PlanningQueueKind::Prefill).await;
    let restore = s.prepare_prefix_restore(&other, 0, 16).unwrap().unwrap();
    let restoring = s.planning_state(limit(), wake()).unwrap();
    let restore_row = restoring
        .requests()
        .iter()
        .find(|r| r.key.request_id == other)
        .unwrap();
    assert!(restore_row.readiness.prefix_blocked);
    assert!(!restore_row.is_modeled_rendezvous_follower(source_row, &owned));
    drop(restore);
}
