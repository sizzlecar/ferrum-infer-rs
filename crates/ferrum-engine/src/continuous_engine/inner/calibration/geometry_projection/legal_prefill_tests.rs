//! Reach an actual joint decode without assuming joint prefill is available.
use super::tests::{assert_unsubmitted_clean, fixture, limits, requests};
use super::*;

#[tokio::test]
async fn geometry_legal_prefill_reaches_joint_decode_and_preserves_prefill_unknown() {
    let (mut session, executor) = fixture(2).await;
    executor
        .single_row_prefill_only
        .store(true, Ordering::Release);
    let probes = requests(&session, 2);
    let targets = [
        GeometryInputTarget::InitialPrefill { rows: 2 },
        GeometryInputTarget::Decode(GeometryProjectionPoint {
            rows: 2,
            sequence_tokens: 2,
        }),
    ];
    let report = session
        .project_geometry_inputs(probes, &targets, limits(), &[])
        .await
        .unwrap();
    assert_eq!(report.admitted_requests, 2);
    assert_eq!(
        report.outcomes[0].unknown,
        Some(GeometryProjectionUnknown::Route(
            ExecutionCostRouteUnknown::Unsupported
        ))
    );
    assert!(report.outcomes[0].branches.is_empty());
    let decode = &report.outcomes[1];
    assert_eq!(decode.unknown, None);
    assert_eq!(decode.branches.len(), 1);
    assert_eq!(decode.branches[0].query.input().owner().rows, 2);
    assert_eq!(decode.branches[0].query.input().owner().role,
        ferrum_scheduler::implementations::continuous::cost_model::structured_v2::StructuredWaveRoleV2::OrdinaryDecode);
    // One refused joint target, two real single-row prefix projections, and
    // the real joint FullLogits decode. No projection is an execution/sample.
    assert_eq!(report.projection_attempts, 1 + 2 + 1);
    assert_unsubmitted_clean(&session, &executor);
    session.shutdown().await.unwrap();
}

#[tokio::test]
async fn geometry_legal_prefill_budget_cannot_invent_the_unprepared_peer() {
    let (mut session, executor) = fixture(2).await;
    executor
        .single_row_prefill_only
        .store(true, Ordering::Release);
    let probes = requests(&session, 2);
    let mut limit = limits();
    limit.maximum_projections = 1;
    let report = session
        .project_geometry_inputs(
            probes,
            &[GeometryInputTarget::Decode(GeometryProjectionPoint {
                rows: 2,
                sequence_tokens: 2,
            })],
            limit,
            &[],
        )
        .await
        .unwrap();
    assert_eq!(report.projection_attempts, 1);
    assert_eq!(
        report.outcomes[0].unknown,
        Some(GeometryProjectionUnknown::BudgetExhausted)
    );
    assert!(report.outcomes[0].branches.is_empty());
    assert_unsubmitted_clean(&session, &executor);
    session.shutdown().await.unwrap();
}
