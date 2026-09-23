use super::*;

#[tokio::test]
async fn maintenance_fairness_turn_unblocks_lone_frontier_without_scheduling() {
    let s = scheduler();
    let id = request(&s, PlanningQueueKind::Decode).await;
    let retry = s
        .defer_retry_after_execution_maintenance_epoch(std::slice::from_ref(&id), 7)
        .unwrap();
    let blocked = s.planning_state(limit(), wake()).unwrap();
    assert!(blocked.requests()[0].readiness.maintenance_blocked);
    assert!(!blocked.requests()[0].readiness.ready());
    // Repeated snapshots and controller wall-clock wakes cannot consume the
    // legacy iteration ticket. Reproduce the stranded state before the fix.
    assert!(blocked.matches(&s.planning_state(limit(), wake()).unwrap()));
    let before = s.trace_snapshot();
    let original = blocked.requests()[0].clone();
    assert_eq!(
        s.try_consume_maintenance_fairness_turn(&blocked, wake())
            .unwrap(),
        PlanningMaintenanceFairnessOutcome::Advanced {
            previous_iteration: blocked.iteration(),
            next_iteration: retry.not_before_iteration(),
            matured_tickets: 1,
        }
    );
    let ready = s.planning_state(limit(), wake()).unwrap();
    assert!(ready.requests()[0].readiness.ready());
    let row = &ready.requests()[0];
    assert_eq!(row.key, original.key);
    assert_eq!(
        row.seal, original.seal,
        "empty turn must not clear the actual ticket or frontier"
    );
    assert_eq!(row.scheduled_tokens, original.scheduled_tokens);
    assert_eq!(
        s.trace_snapshot().capacity_release_epoch,
        before.capacity_release_epoch
    );
    assert_eq!(
        s.try_consume_maintenance_fairness_turn(&blocked, wake())
            .unwrap(),
        PlanningMaintenanceFairnessOutcome::Stale
    );
    assert_eq!(
        s.try_consume_maintenance_fairness_turn(&ready, wake())
            .unwrap(),
        PlanningMaintenanceFairnessOutcome::NoPending
    );
    let choice = selected(&ready, &id, PlanningWorkAction::Decode);
    assert!(matches!(
        s.try_select_planned_wave(&ready, &[choice], &hint(), wake())
            .unwrap(),
        PlanningSelectionOutcome::Published { .. }
    ));
}

#[tokio::test]
async fn maintenance_fairness_turn_revalidates_ingress_and_matures_entire_cohort_once() {
    let s = scheduler();
    let ids = vec![
        request(&s, PlanningQueueKind::Prefill).await,
        request(&s, PlanningQueueKind::Decode).await,
    ];
    s.defer_retry_after_execution_maintenance_epoch(&ids, 7)
        .unwrap();
    let old = s.planning_state(limit(), wake()).unwrap();
    let waiting = request(&s, PlanningQueueKind::Waiting).await;
    assert_eq!(
        s.try_consume_maintenance_fairness_turn(&old, wake())
            .unwrap(),
        PlanningMaintenanceFairnessOutcome::Stale
    );
    let fresh = s.planning_state(limit(), wake()).unwrap();
    assert!(matches!(
        s.try_consume_maintenance_fairness_turn(&fresh, wake())
            .unwrap(),
        PlanningMaintenanceFairnessOutcome::Advanced {
            matured_tickets: 2,
            ..
        }
    ));
    let after = s.planning_state(limit(), wake()).unwrap();
    assert_eq!(
        after
            .requests()
            .iter()
            .filter(|r| r.readiness.ready())
            .count(),
        2
    );
    assert!(
        !after
            .requests()
            .iter()
            .find(|r| r.key.request_id == waiting)
            .unwrap()
            .readiness
            .admitted
    );
    assert_eq!(after.iteration(), fresh.iteration() + 1);
    assert_eq!(
        s.try_consume_maintenance_fairness_turn(&after, wake())
            .unwrap(),
        PlanningMaintenanceFairnessOutcome::NoPending
    );
}

#[tokio::test]
async fn maintenance_fairness_turn_rejects_changed_capacity_foreign_owner_and_contention() {
    let s = scheduler();
    let id = request(&s, PlanningQueueKind::Decode).await;
    s.defer_retry_after_execution_maintenance_epoch(std::slice::from_ref(&id), 7)
        .unwrap();
    // A real release changes the complete queue seal; even an apparently
    // helpful epoch update cannot authorize use of the old fairness snapshot.
    let observed = s.planning_state(limit(), wake()).unwrap();
    s.record_external_capacity_release();
    assert_eq!(
        s.try_consume_maintenance_fairness_turn(&observed, wake())
            .unwrap(),
        PlanningMaintenanceFairnessOutcome::Stale
    );
    let current = s.planning_state(limit(), wake()).unwrap();
    let other = scheduler();
    assert_eq!(
        other
            .try_consume_maintenance_fairness_turn(&current, wake())
            .unwrap(),
        PlanningMaintenanceFairnessOutcome::Stale
    );
    let _held = s.decode_queue.write();
    assert_eq!(
        s.try_consume_maintenance_fairness_turn(&current, wake()),
        Err(PlanningStateUnavailable::Busy)
    );
}
