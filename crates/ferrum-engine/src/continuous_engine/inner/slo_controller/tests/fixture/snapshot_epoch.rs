//! A route and its resource transition share one numeric snapshot identity.
//! Equality of visible counters is insufficient to exchange private states.
use super::*;
use crate::continuous_engine::inner::slo_controller::tests::prefill;

#[tokio::test]
async fn route_and_resource_projections_share_capture_but_not_a_recapture() {
    let (engine, _, executor) = fixture().await;
    let (_, session) = prefill::request(&engine, 4, 2).await;
    prefill::admit(&engine, 1).await;
    let before = prefill::UnsubmittedState::capture(&engine, &executor);
    let captured = prefill::captured(&engine, &executor).await;
    let rows = [vnext::ResourcePlanningRow {
        participant_index: 0,
        start_token: 0,
        token_count: 2,
    }];
    let route_state = captured.route.resource_view().initial_state();
    let resources = &executor.evidence.fixture.as_ref().unwrap().plan_resources;
    let projected =
        resources.project_resource_wave(&captured.resources, &route_state, &rows, &mut || true);
    let ResourcePlanningAvailability::Known(projected) = projected else {
        panic!("same captured epoch must support a joint transition: {projected:?}");
    };
    assert_eq!(projected.state.projected_waves(), 1);
    assert_eq!(route_state.projected_waves(), 0);

    // Recapturing otherwise unchanged live resources creates a new identity.
    // Neither equal counters nor equal work authorize mixing those states.
    let fresh = prefill::captured(&engine, &executor).await;
    assert!(captured.resources.same_live_evidence(&fresh.resources));
    assert!(matches!(
        resources.project_resource_wave(&fresh.resources, &route_state, &rows, &mut || true),
        ResourcePlanningAvailability::Unknown(ResourcePlanningUnknown::StaleIdentity)
    ));
    before.assert_unchanged(&engine, &executor);
    drop(fresh);
    drop(captured);
    cleanup(engine, session).await;
}
