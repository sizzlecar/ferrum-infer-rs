use super::*;

#[test]
fn checkpoint_population_spends_the_original_source_owner_and_action_allowance() {
    let deadline = Instant::now() + Duration::from_secs(5);
    let mut budget = ProbeExecutionBudget::new(
        deadline,
        NonZeroUsize::new(3).unwrap(),
        NonZeroUsize::new(5).unwrap(),
    );
    // One reusable source and two distinct targets, with one real prefill and
    // two actual Capture/Restore pairs; none can be spent again on source8.
    for _ in 0..3 {
        budget.claim_checkpoint_owner().unwrap();
    }
    for _ in 0..5 {
        budget.claim_checkpoint_action().unwrap();
    }
    assert_eq!(budget.requests_remaining(), 0);
    assert_eq!(budget.selection_requests_remaining(), 0);
    assert_eq!(budget.attempts_remaining(), 0);
    assert_eq!(budget.selection_attempts_remaining(), 0);
    assert_eq!(budget.deadline(), deadline);
    assert!(budget.claim_checkpoint_owner().is_err());
    assert!(budget.claim_checkpoint_action().is_err());
    assert!(budget.reserve_selected_source(1, 1).is_err());
}

#[test]
fn checkpoint_expiry_never_renews_or_spends_source_allowance() {
    let deadline = Instant::now();
    let mut budget = ProbeExecutionBudget::new(deadline, NonZeroUsize::MIN, NonZeroUsize::MIN);
    assert!(budget.claim_checkpoint_owner().is_err());
    assert!(budget.claim_checkpoint_action().is_err());
    assert_eq!(
        (
            budget.requests_remaining(),
            budget.selection_requests_remaining()
        ),
        (1, 1)
    );
    assert_eq!(
        (
            budget.attempts_remaining(),
            budget.selection_attempts_remaining()
        ),
        (1, 1)
    );
    assert_eq!(budget.deadline(), deadline);
}

#[test]
fn declared_source_budget_does_not_reuse_early_terminal_wave_savings() {
    let deadline = Instant::now() + Duration::from_secs(5);
    let mut short = ProbeExecutionBudget::new(
        deadline,
        NonZeroUsize::new(16).unwrap(),
        NonZeroUsize::new(100).unwrap(),
    );
    let mut full = ProbeExecutionBudget::new(
        deadline,
        NonZeroUsize::new(16).unwrap(),
        NonZeroUsize::new(100).unwrap(),
    );
    for budget in [&mut short, &mut full] {
        budget.reserve_readiness_waves(20).unwrap();
        budget.claim_readiness_requests(2).unwrap();
        budget.reserve_selected_source(6, 60).unwrap();
        budget.claim_requests(6).unwrap();
    }
    // The same declared work can finish early; its unused actual allowance
    // remains diagnostic and cannot choose another source.
    short.claim_attempt().unwrap();
    for _ in 0..80 {
        full.claim_attempt().unwrap();
    }
    assert_ne!(short.attempts_remaining(), full.attempts_remaining());
    for budget in [&mut short, &mut full] {
        assert_eq!(budget.selection_requests_remaining(), 8);
        assert_eq!(budget.selection_attempts_remaining(), 20);
        assert!(budget.reserve_selected_source(1, 21).is_err());
        budget.reserve_selected_source(8, 20).unwrap();
        assert_eq!(budget.selection_requests_remaining(), 0);
        assert_eq!(budget.selection_attempts_remaining(), 0);
        assert_eq!(budget.deadline(), deadline);
    }
}

#[test]
fn expired_budget_does_not_start_another_wave_or_consume_its_credit() {
    let mut budget =
        ProbeExecutionBudget::new(Instant::now(), NonZeroUsize::MIN, NonZeroUsize::MIN);
    assert!(budget.claim_attempt().is_err());
    assert!(budget.claim_input_projection_requests(1).is_err());
    assert_eq!(budget.attempts_remaining(), 1);
    assert_eq!(budget.input_projection_requests_remaining(), 1);
}

fn remaining_budget(charge: ProbePreflightCharge) -> Result<PreparedProbeExecutionBudget> {
    PreparedProbeExecutionBudget::new(
        Duration::from_secs(5),
        NonZeroUsize::new(4).unwrap(),
        NonZeroUsize::new(2).unwrap(),
        NonZeroUsize::new(3).unwrap(),
        charge,
    )
}

fn geometry_charge(admitted: usize, projections: usize) -> ProbePreflightCharge {
    ProbePreflightCharge {
        admitted_requests: admitted,
        planning_admitted_requests: admitted,
        projection_attempts: projections,
        ..Default::default()
    }
}

#[test]
fn exhausted_input_projection_budget_never_consumes_execution_credit() {
    let mut budget = remaining_budget(Default::default())
        .unwrap()
        .resume()
        .unwrap();
    let deadline = budget.deadline();
    // Independent groups and a recapture share the same cumulative quota.
    budget.claim_input_projection_requests(2).unwrap();
    budget.record_geometry(geometry_charge(2, 7)).unwrap();
    budget.claim_input_projection_requests(1).unwrap();
    budget.record_geometry(geometry_charge(1, 10)).unwrap();
    assert!(budget.claim_input_projection_requests(1).is_err());
    assert_eq!(budget.input_projection_requests_remaining(), 0);
    assert_eq!(budget.preflight_charge().planning_reserved_requests, 3);
    assert_eq!(budget.preflight_charge().planning_admitted_requests, 3);
    assert_eq!(budget.preflight_charge().projection_attempts, 17);
    assert_eq!(budget.requests_remaining(), 4);
    assert_eq!(budget.attempts_remaining(), 2);
    assert_eq!(budget.deadline(), deadline);
    budget.claim_requests(4).unwrap();
    assert!(budget.claim_requests(1).is_err());
}

#[test]
fn failed_projection_reservation_is_not_refunded_or_reported_as_an_admission() {
    let mut budget = remaining_budget(Default::default())
        .unwrap()
        .resume()
        .unwrap();
    budget.claim_input_projection_requests(3).unwrap();
    // An interrupted capture returns no completed report.
    assert!(budget.claim_input_projection_requests(1).is_err());
    assert_eq!(budget.preflight_charge().planning_reserved_requests, 3);
    assert_eq!(budget.preflight_charge().planning_admitted_requests, 0);
    assert_eq!(budget.requests_remaining(), 4);
    assert!(budget.record_geometry(geometry_charge(4, 1)).is_err());
    assert_eq!(budget.preflight_charge().planning_admitted_requests, 0);
}

#[test]
fn readiness_and_failed_unsubmitted_execution_keep_the_original_request_charge() {
    let mut budget = remaining_budget(Default::default())
        .unwrap()
        .resume()
        .unwrap();
    budget.claim_readiness_requests(1).unwrap();
    budget.record_readiness_admissions(1).unwrap();
    budget.claim_readiness_requests(1).unwrap();
    // This readiness attempt failed after reservation; no completion is reported.
    assert_eq!(budget.preflight_charge().readiness_reserved_requests, 2);
    assert_eq!(budget.preflight_charge().readiness_admitted_requests, 1);
    assert_eq!(budget.requests_remaining(), 2);
    budget.claim_requests(2).unwrap();
    // A numerical request may also fail before submitting a wave.
    assert!(budget.claim_requests(1).is_err());
    assert!(budget.claim_readiness_requests(1).is_err());
    assert_eq!(budget.requests_remaining(), 0);
    assert_eq!(budget.input_projection_requests_remaining(), 3);
    assert_eq!(budget.attempts_remaining(), 2);
    budget.claim_attempt().unwrap();
    budget.claim_attempt().unwrap();
    assert!(budget.claim_attempt().is_err());
}

#[test]
fn resume_preserves_projection_reservations_and_rejects_preexisting_readiness() {
    let charge = ProbePreflightCharge {
        admitted_requests: 2,
        planning_admitted_requests: 2,
        readiness_admitted_requests: 0,
        planning_reserved_requests: 3,
        readiness_reserved_requests: 0,
        projection_attempts: 17,
    };
    let mut budget = remaining_budget(charge).unwrap().resume().unwrap();
    assert_eq!(budget.preflight_charge(), charge);
    assert_eq!(budget.requests_remaining(), 4);
    assert_eq!(budget.input_projection_requests_remaining(), 0);
    assert!(budget.claim_input_projection_requests(1).is_err());
    budget.claim_requests(4).unwrap();
    assert!(budget.claim_requests(1).is_err());
    assert!(remaining_budget(ProbePreflightCharge {
        planning_reserved_requests: 4,
        ..Default::default()
    })
    .is_err());
    assert!(remaining_budget(ProbePreflightCharge {
        readiness_reserved_requests: 1,
        ..Default::default()
    })
    .is_err());
}

#[test]
fn collection_receives_only_the_unspent_cost_duration() {
    let remaining = Duration::from_millis(7);
    let prepared = PreparedProbeExecutionBudget::new(
        remaining,
        NonZeroUsize::MIN,
        NonZeroUsize::MIN,
        NonZeroUsize::MIN,
        ProbePreflightCharge::default(),
    )
    .unwrap();
    let before = Instant::now();
    let budget = prepared.resume().unwrap();
    let after = Instant::now();
    assert!(budget.deadline() >= before + remaining);
    assert!(budget.deadline() <= after + remaining);
    assert!(PreparedProbeExecutionBudget::new(
        Duration::ZERO,
        NonZeroUsize::MIN,
        NonZeroUsize::MIN,
        NonZeroUsize::MIN,
        ProbePreflightCharge::default(),
    )
    .is_err());
}
