//! Compare shared trajectories with independent targets on the real CPU route.
use super::tests::{assert_unsubmitted_clean, fixture, limits, requests};
use super::*;

fn point(sequence_tokens: u32) -> GeometryInputTarget {
    GeometryInputTarget::Decode(GeometryProjectionPoint {
        rows: 2,
        sequence_tokens,
    })
}

pub(super) fn assert_equivalent(a: &GeometryInputOutcome, b: &GeometryInputOutcome) {
    assert_eq!(a.target, b.target);
    assert_eq!(a.unknown, b.unknown);
    assert_eq!(a.prefix_condition, b.prefix_condition);
    assert_eq!(a.branches.len(), b.branches.len());
    for (a, b) in a.branches.iter().zip(&b.branches) {
        assert_eq!(a.host_branch, b.host_branch);
        assert_eq!(a.graph, b.graph);
        assert_eq!(a.query.input(), b.query.input());
        // The query's derived Debug includes its private pending constraints
        // and repetition bounds as well as the complete checked input.
        assert_eq!(format!("{:?}", a.query), format!("{:?}", b.query));
    }
}

async fn project(
    session: &mut CalibrationSession,
    targets: &[GeometryInputTarget],
    independent: bool,
    limit: GeometryProjectionLimits,
) -> GeometryInputReport {
    let probes = requests(session, 2);
    if independent {
        let scenarios: Vec<_> = targets
            .iter()
            .map(|target| GeometryInputScenario {
                targets: std::slice::from_ref(target),
                prefixes: &[],
            })
            .collect();
        session
            .project_geometry_input_scenarios(probes, &scenarios, limit)
            .await
            .unwrap()
    } else {
        session
            .project_geometry_inputs(probes, targets, limit, &[])
            .await
            .unwrap()
    }
}

#[tokio::test]
async fn geometry_trajectory_matches_independent_queries_and_charges_each_wave_once() {
    let (mut session, executor) = fixture(2).await;
    executor
        .recycle_completed_bindings
        .store(true, Ordering::Release);
    executor
        .single_row_prefill_only
        .store(true, Ordering::Release);
    executor
        .context_partitioned_cpu_fill
        .store(true, Ordering::Release);
    let targets = [point(2), point(3), point(4)];
    let independent = project(&mut session, &targets, true, limits()).await;
    let longest = project(&mut session, &targets[2..], true, limits()).await;
    let together = project(&mut session, &targets, false, limits()).await;
    assert_eq!(independent.outcomes.len(), targets.len());
    assert_eq!(together.outcomes.len(), targets.len());
    for (expected, actual) in independent.outcomes.iter().zip(&together.outcomes) {
        assert_eq!(expected.unknown, None);
        assert_equivalent(expected, actual);
    }
    assert_eq!(together.admitted_requests, independent.admitted_requests);
    // FullLogits has one successor per step: reaching all endpoints takes
    // exactly the real projections required by the longest endpoint alone.
    assert_eq!(together.projection_attempts, longest.projection_attempts);
    assert!(together.projection_attempts < independent.projection_attempts);
    let mut budget = limits();
    budget.maximum_projections = together.projection_attempts - 1;
    let bounded = project(&mut session, &targets, false, budget).await;
    assert_eq!(bounded.projection_attempts, budget.maximum_projections);
    for (complete, bounded) in together.outcomes[..2].iter().zip(&bounded.outcomes[..2]) {
        assert_equivalent(complete, bounded);
    }
    assert_eq!(
        bounded.outcomes[2].unknown,
        Some(GeometryProjectionUnknown::BudgetExhausted)
    );
    assert!(bounded.outcomes[2].branches.is_empty());
    assert_unsubmitted_clean(&session, &executor);
    session.shutdown().await.unwrap();
}

#[tokio::test]
async fn geometry_trajectory_preserves_nonmonotone_duplicates_terminal_and_unsupported_targets() {
    let (mut session, executor) = fixture(2).await;
    executor
        .recycle_completed_bindings
        .store(true, Ordering::Release);
    executor
        .single_row_prefill_only
        .store(true, Ordering::Release);
    let targets = [
        point(4),
        point(2),
        point(2),
        point(3),
        point(5),
        GeometryInputTarget::InitialPrefill { rows: 2 },
        point(4),
    ];
    let independent = project(&mut session, &targets, true, limits()).await;
    let together = project(&mut session, &targets, false, limits()).await;
    assert_eq!(together.outcomes.len(), targets.len());
    for (expected, actual) in independent.outcomes.iter().zip(&together.outcomes) {
        assert_equivalent(expected, actual);
    }
    assert_eq!(
        together.outcomes[4].unknown,
        Some(GeometryProjectionUnknown::Unreachable)
    );
    assert_eq!(
        together.outcomes[5].unknown,
        Some(GeometryProjectionUnknown::Route(
            ExecutionCostRouteUnknown::Unsupported,
        ))
    );
    assert_eq!(together.outcomes[6].unknown, None);
    assert_unsubmitted_clean(&session, &executor);
    session.shutdown().await.unwrap();
}
