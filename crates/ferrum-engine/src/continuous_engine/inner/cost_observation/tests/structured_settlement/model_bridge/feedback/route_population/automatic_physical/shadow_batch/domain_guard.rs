//! Original CPU submission/host settlement and genuinely qualified catalogues.
//! The alternate D is checked by the real projection; no receipt is mutated.
use super::*;
use crate::continuous_engine::inner::cost_observation::{
    resolved::ResolvedCostEntry,
    selected_feedback::FeedbackObservation,
    structured_feedback::{self, Failure},
};

const UNKNOWN_ALGORITHM: &str = "fixture.domain-guard.new-algorithm";

fn recorder(clock: &Arc<VirtualClock>, domain: Option<CostWorkloadDomainV1>) -> EngineCostRuntime {
    EngineCostRuntime::build_with_profile_and_domain(
        identity(),
        clock.clone(),
        &SloCostObservationConfig::structured_whole_wave_v2(),
        false,
        None,
        None,
        domain,
    )
    .unwrap()
}

fn recorded_unknown(
    runtime: &EngineCostRuntime,
    clock: &Arc<VirtualClock>,
    rows: u32,
) -> ResolvedCostEntry {
    let w = wave_with_host_rows(UNKNOWN_ALGORITHM, 7, wave(UNKNOWN_ALGORITHM).host, rows);
    let stages = fixture::record_unticketed_cohort_with_hook(runtime, clock, w, |_| {})
        .expect("actual CPU submission and original host settlement");
    assert_eq!(
        stages.completeness,
        HostStageCompleteness::CompleteSingleWave
    );
    stages
        .structured_evidence
        .as_ref()
        .unwrap()
        .as_ref()
        .unwrap()
        .validate_host_stages(&stages)
        .unwrap();
    let (_, resolved, ticket, _, no_submission) = runtime.sink.pop_resolved_bound().unwrap();
    assert!(ticket.is_none());
    assert!(no_submission.is_none());
    assert!(runtime.sink.pop_resolved_bound().is_none());
    let resolved = resolved.unwrap();
    assert!(resolved.structured().unwrap().feedback_scope.is_ok());
    resolved
}

#[tokio::test]
async fn physical_catalog_feedback_requires_original_domain_before_new_owner_exclusion() {
    let f = Cohort::new();
    for _ in 0..4 {
        f.phase(2);
    }
    assert_eq!(f.live().audit().qualified_publications, 1);
    let at = f.clock.now_ns().unwrap();
    let children = f.runtime.training.live_catalog_children(at).unwrap();
    assert_eq!(children.len(), 1);
    assert!(children[0].algorithm_universe().is_none());
    assert!(children[0].numerical_family_key().is_none());
    assert_eq!(children[0].workload_domain(), Some(&f.domain));
    let snapshot = f.runtime.snapshot().unwrap();
    snapshot.audit_structured_query_v2(&f.query(2), at).unwrap();
    let recorder = recorder(&f.clock, Some(f.domain.clone()));

    let valid = recorded_unknown(&recorder, &f.clock, 2);
    assert_eq!(
        valid
            .structured()
            .unwrap()
            .query
            .input()
            .physical_domain_signature(),
        Some(f.domain.sha256())
    );
    assert!(snapshot.undeclared_structured_owner(&valid.structured().unwrap().query));
    let mut failure = None;
    assert!(matches!(
        structured_feedback::evaluate_resolved(
            &valid,
            Some(&snapshot),
            f.clock.as_ref(),
            &mut failure,
        ),
        FeedbackObservation::OutsideCatalog { .. }
    ));
    assert!(failure.is_none());

    let ExecutorCostIdentityAvailability::Known(executor) = identity() else {
        panic!("fixture identity");
    };
    let mut limits = *f.domain.limits();
    limits.maximum_context_tokens = NonZeroU32::new(129).unwrap();
    let other_domain = CostWorkloadDomainV1::new_vnext(&executor, limits).unwrap();
    for domain in [None, Some(&other_domain)] {
        // Keep the original private settlement intact. Only the consumer's
        // declared physical domain changes, through its production API.
        let actual = recorded_unknown(&recorder, &f.clock, 2);
        let rebound = ResolvedCostEntry::new_with_domain(actual.into_entry(), domain);
        let projected = rebound
            .structured()
            .expect("complete original settlement remains valid");
        assert!(projected.feedback_scope.is_ok());
        assert_eq!(
            projected.query.input().physical_domain_signature(),
            domain.map(CostWorkloadDomainV1::sha256)
        );
        assert!(!snapshot.undeclared_structured_owner(&projected.query));
        let mut failure = None;
        assert!(matches!(
            structured_feedback::evaluate_resolved(
                &rebound,
                Some(&snapshot),
                f.clock.as_ref(),
                &mut failure,
            ),
            FeedbackObservation::Uncomparable
        ));
        assert!(matches!(
            failure,
            Some(Failure::Lookup(StructuredUnknownV2::WrongDomain))
        ));
    }
    // Classification is read-only; the ordinary valid region stays queryable.
    snapshot
        .audit_structured_query_v2(&f.query(2), f.clock.now_ns().unwrap())
        .unwrap();
    // This workerless recorder transfers its original FIFO entries to this
    // test. No training consumer owns those ordinals, so it has no shutdown
    // checkpoint to await. Drop releases the empty recorder directly.
    drop(recorder);
    f.runtime.shutdown().await.unwrap();
}

#[tokio::test]
async fn legacy_catalog_feedback_without_physical_domain_keeps_owner_exclusion() {
    let f = Fixture::new();
    let runtime = f.build();
    let snapshot = runtime.snapshot().unwrap();
    let children = runtime
        .training
        .live_catalog_children(f.clock.now_ns().unwrap())
        .unwrap();
    assert!(children
        .iter()
        .all(|child| child.workload_domain().is_none()));
    let recorder = recorder(&f.clock, None);
    let actual = recorded_unknown(&recorder, &f.clock, 1);
    assert_eq!(
        actual
            .structured()
            .unwrap()
            .query
            .input()
            .physical_domain_signature(),
        None
    );
    assert!(snapshot.undeclared_structured_owner(&actual.structured().unwrap().query));
    let mut failure = None;
    assert!(matches!(
        structured_feedback::evaluate_resolved(
            &actual,
            Some(&snapshot),
            f.clock.as_ref(),
            &mut failure,
        ),
        FeedbackObservation::OutsideCatalog { .. }
    ));
    assert!(failure.is_none());
    snapshot
        .audit_structured_query_v2(&f.queries[0], f.clock.now_ns().unwrap())
        .unwrap();
    f.unchanged();
    drop(recorder);
    runtime.shutdown().await.unwrap();
}
