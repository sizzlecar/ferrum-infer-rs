use super::*;

fn steps() -> Vec<PrefixPathStep> {
    // Audit fixtures deliberately carry no executable authority. Counting
    // original path records cannot mint a replay proof or set submission flags.
    let wave = PrefixPathStep::Wave(WaveCandidate {
        work: Vec::new(),
        cost_evidence: None,
        execution_shape: PlanningShapeDomain::HostContentAlternatives(Vec::new()),
        based_on_generation: 1,
        cost_model_version: 1,
    });
    let maintenance = |stage| {
        PrefixPathStep::Maintenance(PrefixMaintenanceEvidence {
            stage,
            capture_span_start: 0,
            cost_domain: PlanningShapeDomain::HostContentAlternatives(Vec::new()),
            restored_frontier: None,
            model_version: 2,
        })
    };
    vec![
        wave.clone(),
        maintenance(PrefixMaintenanceStage::Capture),
        maintenance(PrefixMaintenanceStage::Restore),
        wave,
    ]
}

#[test]
fn prefix_planning_audit_counts_model_waves_without_claiming_actual_submission() {
    let search = PlanningSearchStats {
        enumeration_attempts: 7,
        generated_candidates: 4,
        expanded_candidates: 3,
        ..Default::default()
    };
    let budget = ControllerBudget::new(slo_clock_now(), Duration::from_secs(30)).unwrap();
    let labels = budget.prefix_replay(search, &steps(), true);
    assert_eq!(
        labels,
        ("prefix_wave_replayed", "captured_obligation_scope")
    );
    let audit = budget.take_audit("idle").unwrap();
    assert_eq!(audit.search, search);
    assert_eq!(
        audit.witness,
        Some(ControllerWitnessAudit {
            waves: 2,
            tail_waves: 1
        })
    );
    assert!(!audit.backend_submitted);
    assert!(!audit.host_reconciled);
    assert!(!audit.budget_exhausted);

    // A maintenance-first result still has a future model suffix, but no
    // currently selected model wave. It must not inherit an earlier witness.
    let budget = ControllerBudget::new(slo_clock_now(), Duration::from_secs(30)).unwrap();
    budget.prefix_replay(search, &steps(), true);
    let labels = budget.prefix_replay(search, &steps()[1..], false);
    assert_eq!(
        labels,
        ("prefix_maintenance_replayed", "captured_obligation_scope")
    );
    let audit = budget.take_audit("prefix_maintenance").unwrap();
    assert_eq!(audit.witness, None);
    assert_eq!(audit.search, search);
    assert!(!audit.backend_submitted);
    assert!(!audit.host_reconciled);
}

#[test]
fn prefix_planning_audit_preserves_each_typed_unknown_and_original_search() {
    for case in 0..4 {
        let budget = ControllerBudget::new(slo_clock_now(), Duration::from_secs(30)).unwrap();
        budget.prefix_replay(PlanningSearchStats::default(), &steps(), true);
        let search = PlanningSearchStats {
            enumeration_attempts: 9,
            generated_candidates: 5,
            expanded_candidates: 4,
            cost_unknown_candidates: 1,
            ..Default::default()
        };
        let reason = if case == 3 {
            PlanningUnknownReason::ComputeBudgetExhausted
        } else {
            PlanningUnknownReason::CostUnavailable
        };
        let labels = match case {
            0 => budget.record_prefix_continuation(&PrefixContinuationDecision::Unknown {
                reason,
                search,
            }),
            1 => budget.record_ready_prefix(&ReadyPrefixDecision::Unknown { reason, search }),
            2 => budget.record_prefix_cache_capture(&PrefixCacheCaptureDecision::Unknown {
                reason,
                search,
            }),
            _ => budget
                .record_prefix_comparison(&PrefixRendezvousDecision::Unknown { reason, search }),
        };
        assert_eq!(labels, ("unknown", unknown_label(reason)));
        let audit = budget.take_audit("idle").unwrap();
        assert_eq!(audit.search, search);
        assert_eq!(audit.witness, None);
        assert_eq!((audit.decision, audit.reason), labels);
        assert_eq!(audit.planner_budget_exhausted, case == 3);
        assert!(
            !audit.budget_exhausted,
            "reserved-phase exhaustion cannot imply a hard deadline"
        );
        assert!(!audit.backend_submitted);
        assert!(!audit.host_reconciled);
    }
}
