use super::*;
use crate::continuous_engine::inner::cost_observation::CostCallRejection;

#[tokio::test]
async fn startup_series_inventory_requires_retirement_and_preserves_the_original_catalog() {
    let (mut session, _) = automatic_session().await;
    let runtime = session.engine.inner.cost_runtime.as_ref().unwrap().clone();
    let deadline = tokio::time::Instant::now() + Duration::from_secs(45);
    session.begin_startup_owner_series(2, deadline).unwrap();
    session
        .begin_prepared_owner_source(full_population(&session), CostProfileLoadLimits::default())
        .await
        .unwrap();
    assert!(session.begin_startup_inventory().is_err());
    let mut budget = ProbeExecutionBudget::new(
        deadline,
        NonZeroUsize::new(65).unwrap(),
        NonZeroUsize::new(132).unwrap(),
    );
    let options = ProbeCohortSettings {
        prefill_plan:
            crate::continuous_engine::inner::calibration::cohort_driver::ProbePrefillPlan::Joint,
        prefill_chunk: NonZeroU32::MIN,
        decode_route: CalibrationDecodeRoute::FullLogits,
        reset_token_policy: false,
    };
    for pass in 0..3 {
        for ordinal in 0..[16, 8, 8][pass] {
            session.begin_prepared_owner_cohort(pass, ordinal).unwrap();
            session
                .run_probe_cohort(probe_requests(&session), options, &mut budget)
                .await
                .unwrap();
            session.end_prepared_owner_cohort().unwrap();
        }
    }
    let epoch = session.activate_prepared_owner_source().await.unwrap();
    session.retire_startup_owner_source().await.unwrap();
    assert!(session.prefix_source8);
    assert!(session.check_prepared_owner_capture().is_err());
    session.begin_startup_inventory().unwrap();
    assert!(session.startup_inventory_active());
    assert!(session.check_prepared_owner_capture().is_ok());
    assert!(session
        .begin_prepared_owner_source(full_population(&session), CostProfileLoadLimits::default())
        .await
        .is_err());
    let mut ordinary = probe_requests(&session);
    ordinary.truncate(1);
    let composite_before = runtime.sink.stats().rejected(CostCallRejection::Composite);
    let audit_before = serde_json::to_value(runtime.audit_snapshot()).unwrap();
    let outside_before = audit_before["structured_feedback"]["outside_preparation_observations"]
        .as_u64()
        .unwrap();
    session
        .run_readiness_probe_cohort(ordinary, options, &mut budget)
        .await
        .unwrap();
    session.end_startup_inventory().await.unwrap();
    assert!(!session.startup_inventory_active());
    assert_eq!(budget.requests_remaining(), 0);
    assert_eq!(budget.deadline(), deadline);
    // The actual normal terminal publication is Composite in the legacy cost
    // sink. Its private, complete host settlement must still be classified
    // outside the installed numerical population by the original producer.
    assert!(runtime.sink.stats().rejected(CostCallRejection::Composite) > composite_before);
    let audit_after = serde_json::to_value(runtime.audit_snapshot()).unwrap();
    assert!(
        audit_after["structured_feedback"]["outside_preparation_observations"]
            .as_u64()
            .unwrap()
            > outside_before
    );
    assert_eq!(runtime.catalog_live_state(), Some((epoch, true)));
    session.finish_startup_owner_series().unwrap();
    runtime.shutdown().await.unwrap();
}
