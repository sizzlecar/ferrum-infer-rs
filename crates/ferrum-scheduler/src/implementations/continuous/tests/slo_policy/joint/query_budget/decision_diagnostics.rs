use super::*;

#[test]
fn actual_stopping_coverage_admission_records_once_across_gate_revalidation_and_use() {
    let (scheduler, mut round, original, _, now) = waiting_fixture(None);
    let source = Arc::new(Counted {
        base: ConditionalSnapshot {
            at: original.base.at,
            stopping_known: false,
            completion_known: true,
        },
        fault: None,
        cost: AtomicUsize::new(0),
        stopping: AtomicUsize::new(0),
        completion: AtomicUsize::new(0),
    });
    round.snapshot = source.clone();
    scheduler.prepare_joint_admission_gate(&hint(), &mut round, now);
    assert!(round.admission_gate.is_some());
    let evaluation = round.statistics.joint_evaluation.admission;
    assert_eq!(evaluation.evaluations, 1);
    assert_eq!(evaluation.completed_traversals, 1);
    assert_eq!(evaluation.failed_traversals, 0);
    let decision = round.statistics.joint_decision.admission;
    assert_eq!(decision.computed, 1);
    assert_eq!(decision.returned, 1);
    assert_eq!(decision.consumed, 1);
    assert_eq!(decision.post_finish_time_rejected, 0);
    assert_eq!(decision.outcomes.stopping_coverage, 1);
    assert_eq!(decision.outcomes.feasibility_unresolved, 0);
    assert_eq!(decision.unresolved.decisions, 0);
    assert!(decision.latest_unresolved.is_none());
    assert_eq!(round.statistics.joint_interventions, 0);
    assert_eq!(
        serde_json::to_value(round.statistics.joint_unsupported).unwrap()["stopping_coverage"],
        1
    );
    assert_eq!(source.stopping.load(Ordering::Relaxed), 3);
    assert_eq!(source.completion.load(Ordering::Relaxed), 3);
    assert_eq!(round.query_count, source.calls());
    let calls = source.calls();
    let recorded = serde_json::to_value(round.statistics.joint_decision).unwrap();

    // Revalidating a frozen result is not a fresh evaluation. The real scan
    // then consumes the existing StaticFallback policy for unknown stopping.
    for _ in 0..2 {
        let waiting = scheduler.waiting_queue.read();
        scheduler.revalidate_joint_admission_gate(&waiting, &mut round);
        assert!(round.admission_gate.is_some());
    }
    assert_eq!(admit_at(&scheduler, &mut round, now), 1);
    {
        let waiting = scheduler.waiting_queue.read();
        assert!(waiting.is_empty());
        scheduler.revalidate_joint_admission_gate(&waiting, &mut round);
    }
    assert!(round.admission_gate.is_some());
    assert_eq!(source.calls(), calls);
    assert_eq!(round.query_count, calls);
    assert_eq!(
        serde_json::to_value(round.statistics.joint_decision).unwrap(),
        recorded
    );
    assert_eq!(round.statistics.joint_interventions, 0);
}
