//! Extends the original real three-phase Metal source/imported seed test.
//! No new training, fake pending state, synthetic cost or execution permission.
use super::*;
use crate::continuous_engine::{
    RequiredFutureAuditActionV2, RequiredFutureAuditCostV2, RequiredFutureAuditLimitsV2,
    RequiredFutureAuditPathV2, RequiredFutureAuditPlanV2, RequiredFutureAuditReportV2,
    RequiredFutureAuditRowV2,
};
use ferrum_scheduler::implementations::continuous::cost_model::structured_v2::{
    StructuredProductV2, StructuredQueryV2,
};

pub(super) fn check_native_paths(
    context: &shape::ExecutorShape<'_>,
    captured: &ControllerSnapshot,
    lookup: &mut dyn FnMut(&StructuredQueryV2) -> RequiredFutureAuditCostV2,
) {
    context.assert_decode_domain_equivalence(4);
    let plan = RequiredFutureAuditPlanV2 {
        paths: vec![RequiredFutureAuditPathV2 {
            waves: (0..3)
                .map(|_| {
                    vec![RequiredFutureAuditRowV2 {
                        frontier_index: 0,
                        action: RequiredFutureAuditActionV2::Decode,
                    }]
                })
                .collect(),
        }],
        limits: RequiredFutureAuditLimitsV2 {
            budget_ms: NonZeroU64::new(30_000).unwrap(),
            maximum_queries: NonZeroUsize::new(16).unwrap(),
            maximum_coordinates: NonZeroUsize::new(16_384).unwrap(),
        },
    };
    plan.validate(1).unwrap();
    let original = captured.snapshot.requests.clone();
    let mut poll = || {
        if captured.budget.poll() {
            Ok(())
        } else {
            Err(PlanningUnknownReason::ComputeBudgetExhausted)
        }
    };

    let report = context.audit_paths(
        &plan,
        &[0],
        RequiredFutureAuditReportV2::new(&plan),
        &mut poll,
        lookup,
    );
    assert!(report.completed_all_declared_paths, "{report:#?}");
    assert_eq!(report.paths[0].projected_waves, 3);
    assert!(
        report.queries.iter().any(|q| q.exact_first_wave
            && matches!(q.cost, RequiredFutureAuditCostV2::KnownAtRead { .. })),
        "{report:#?}"
    );
    // A one-owner ordinary Greedy seed cannot qualify the Full alternative.
    // Still retain all its real requirements and continue the next wave.
    for wave in [1, 2] {
        let alternatives: Vec<_> = report
            .queries
            .iter()
            .filter(|q| q.wave_index == wave)
            .collect();
        assert!(
            alternatives
                .iter()
                .any(|q| q.demand.as_ref().unwrap().owner.product
                    == StructuredProductV2::GreedyToken),
            "{report:#?}"
        );
        assert!(
            alternatives.iter().any(|q| {
                q.demand.as_ref().unwrap().owner.product == StructuredProductV2::FullLogits
                    && matches!(q.cost, RequiredFutureAuditCostV2::Unknown { .. })
            }),
            "{report:#?}"
        );
    }
    assert_eq!(captured.snapshot.requests, original);
    // Capacity truncation retains no partial alternative set from a wave.
    let mut bounded = plan.clone();
    bounded.limits.maximum_queries = NonZeroUsize::new(1).unwrap();
    let report = context.audit_paths(
        &bounded,
        &[0],
        RequiredFutureAuditReportV2::new(&bounded),
        &mut poll,
        lookup,
    );
    assert!(
        report.truncated && !report.completed_all_declared_paths,
        "{report:#?}"
    );
    assert_eq!(report.paths[0].projected_waves, 1);
    assert_eq!(report.queries.len(), 1);
    assert!(report.queries[0].exact_first_wave);
    // A production-invalid action is rejected before invoking a cost lookup.
    let mut illegal = plan.clone();
    illegal.paths[0].waves[0][0].action = RequiredFutureAuditActionV2::Prefill { count: n32(8) };
    let report = context.audit_paths(
        &illegal,
        &[0],
        RequiredFutureAuditReportV2::new(&illegal),
        &mut poll,
        &mut |_| panic!("illegal action must not reach model"),
    );
    assert!(!report.completed_all_declared_paths && !report.truncated);
    assert!(report.queries.is_empty() && report.paths[0].stopped.is_some());
    assert_eq!(captured.snapshot.requests, original);
}
