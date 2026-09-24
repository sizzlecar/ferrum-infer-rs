use super::*;
use std::time::Duration;

#[tokio::test(start_paused = true)]
async fn controller_budget_charges_capture_search_and_publication_to_one_deadline() {
    let budget = ControllerBudget::new(slo_clock_now(), Duration::from_millis(10)).unwrap();
    {
        let _capture = budget.stage(ControllerStage::Capture);
        tokio::time::advance(Duration::from_millis(4)).await;
    }
    assert!(budget.poll());
    {
        let _search = budget.stage(ControllerStage::SearchReplay);
        tokio::time::advance(Duration::from_millis(5)).await;
    }
    assert!(budget.poll());
    {
        let _publication = budget.stage(ControllerStage::Publication);
        tokio::time::advance(Duration::from_millis(1)).await;
        assert!(!budget.poll(), "publication cannot obtain a new allowance");
    }
    assert!(!budget.finish_planning());
    let audit = budget.take_audit("idle").unwrap();
    assert_eq!(audit.planning_wall_ns, 10_000_000);
    assert_eq!(audit.transaction_wall_ns, 10_000_000);
    assert_eq!(
        audit.stages[ControllerStage::Capture as usize].wall_ns,
        4_000_000
    );
    assert_eq!(
        audit.stages[ControllerStage::SearchReplay as usize].wall_ns,
        5_000_000
    );
    assert_eq!(
        audit.stages[ControllerStage::Publication as usize].wall_ns,
        1_000_000
    );
    assert!(audit.budget_exhausted);
    assert!(
        budget.take_audit("withdrawn").is_none(),
        "emit exactly once"
    );
}

#[tokio::test(start_paused = true)]
async fn controller_audit_keeps_execution_and_guard_time_outside_planning_budget() {
    let budget = ControllerBudget::new(slo_clock_now(), Duration::from_millis(2)).unwrap();
    tokio::time::advance(Duration::from_millis(1)).await;
    assert!(budget.finish_planning());
    {
        let _execution = budget.stage(ControllerStage::ExecutorAwait);
        tokio::time::advance(Duration::from_millis(20)).await;
        {
            let _guard = budget.stage(ControllerStage::HostGuard);
            tokio::time::advance(Duration::from_millis(1)).await;
        }
        tokio::time::advance(Duration::from_millis(30)).await;
    }
    let audit = budget.take_audit("submitted").unwrap();
    assert_eq!(audit.planning_wall_ns, 1_000_000);
    assert_eq!(audit.transaction_wall_ns, 52_000_000);
    assert_eq!(
        audit.stages[ControllerStage::ExecutorAwait as usize].wall_ns,
        51_000_000
    );
    assert_eq!(
        audit.stages[ControllerStage::HostGuard as usize].wall_ns,
        1_000_000
    );
    assert!(
        !audit.budget_exhausted,
        "an ended planning transaction is not extended over GPU wait"
    );
}

#[tokio::test(start_paused = true)]
async fn controller_unknown_audit_retains_search_work_and_clock_failure_is_sticky() {
    let began = slo_clock_now();
    let budget = ControllerBudget::new(began, Duration::from_millis(10)).unwrap();
    budget.record_search(&PlanningDecision::Unknown {
        reason: PlanningUnknownReason::ComputeBudgetExhausted,
        search: PlanningSearchStats {
            enumeration_attempts: 17,
            generated_candidates: 3,
            cost_unknown_candidates: 2,
            ..Default::default()
        },
    });
    assert!(!budget.poll_at(began - Duration::from_nanos(1)));
    assert!(
        !budget.poll_at(began),
        "clock failure cannot be erased by a later valid read"
    );
    assert!(!budget.finish_planning());
    let audit = budget.take_audit("idle").unwrap();
    assert!(audit.clock_invalid);
    assert!(audit.planner_budget_exhausted);
    assert!(
        !audit.budget_exhausted,
        "a planner stop and invalid clock do not prove hard-deadline exhaustion"
    );
    assert_eq!(audit.search.enumeration_attempts, 17);
    assert_eq!(audit.search.generated_candidates, 3);
    assert_eq!(audit.reason, "compute_budget_exhausted");
}

#[tokio::test(start_paused = true)]
async fn planner_phase_exhaustion_is_distinct_from_hard_transaction_exhaustion() {
    for elapsed_us in [1_600, 2_000] {
        let budget = ControllerBudget::new(slo_clock_now(), Duration::from_millis(2)).unwrap();
        tokio::time::advance(Duration::from_micros(elapsed_us)).await;
        budget.record_search(&PlanningDecision::Unknown {
            reason: PlanningUnknownReason::ComputeBudgetExhausted,
            search: PlanningSearchStats::default(),
        });
        assert_eq!(budget.finish_planning(), elapsed_us < 2_000);
        let audit = budget.take_audit("idle").unwrap();
        assert_eq!(audit.planning_wall_ns, elapsed_us * 1_000);
        assert!(audit.planner_budget_exhausted);
        assert_eq!(audit.budget_exhausted, elapsed_us >= 2_000);
        assert!(!audit.clock_invalid);
    }
}

#[tokio::test(start_paused = true)]
async fn controller_window_keeps_capture_elapsed_and_full_publication_deadline() {
    let started = slo_clock_now();
    let budget = ControllerBudget::new(started, Duration::from_micros(2_000)).unwrap();
    tokio::time::advance(Duration::from_micros(1_300)).await;
    let origin = PlanningTimeOrigin::from_origin(started, slo_clock_now()).unwrap();
    let window = budget.planning_window(&origin).unwrap();
    assert_eq!(window.started_at_ns, 0);
    assert_eq!(window.deadline_ns, 2_000_000);
    assert_eq!(origin.observed_at_ns(), 1_300_000);
    tokio::time::advance(Duration::from_micros(650)).await;
    assert!(
        budget.poll(),
        "publication remains inside the original hard deadline"
    );
    tokio::time::advance(Duration::from_micros(50)).await;
    assert!(
        !budget.finish_planning(),
        "the final reserve cannot extend the transaction"
    );
}
