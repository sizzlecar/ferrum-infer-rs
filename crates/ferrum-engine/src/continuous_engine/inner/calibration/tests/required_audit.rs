use super::*;

fn plan() -> RequiredFutureAuditPlanV2 {
    RequiredFutureAuditPlanV2 {
        paths: vec![RequiredFutureAuditPathV2 {
            waves: vec![vec![RequiredFutureAuditRowV2 {
                frontier_index: 0,
                action: RequiredFutureAuditActionV2::Decode,
            }]],
        }],
        limits: RequiredFutureAuditLimitsV2 {
            budget_ms: NonZeroU64::new(1000).unwrap(),
            maximum_queries: NonZeroUsize::new(16).unwrap(),
            maximum_coordinates: NonZeroUsize::new(4096).unwrap(),
        },
    }
}

#[tokio::test]
async fn required_future_audit_rejects_foreign_and_omitted_owners_without_execution() {
    let (mut session, executor) = fixture(2).await;
    let (mut foreign, foreign_executor) = fixture(1).await;
    let (_, output_a) = add(&mut session, 4).await;
    let (_, output_b) = add(&mut session, 4).await;
    let (_, output_c) = add(&mut foreign, 4).await;
    assert!(session
        .audit_required_owners_v2(&foreign.frontiers().unwrap(), &plan())
        .await
        .is_err());
    let local = session.frontiers().unwrap();
    assert!(session
        .audit_required_owners_v2(&local[..1], &plan())
        .await
        .is_err());
    assert_eq!(executor.entries.load(Ordering::Acquire), 0);
    assert_eq!(foreign_executor.entries.load(Ordering::Acquire), 0);
    drop((output_a, output_b, output_c));
    session.shutdown().await.unwrap();
    foreign.shutdown().await.unwrap();
}

#[tokio::test]
async fn required_future_audit_no_seed_and_stale_frontier_never_create_a_submission() {
    let (mut session, executor) = fixture(1).await;
    let (id, mut output) = add(&mut session, 4).await;
    admit(&mut session).await;
    let before = session.frontiers().unwrap();
    let report = session
        .audit_required_owners_v2(&before, &plan())
        .await
        .unwrap();
    assert!(matches!(
        report.unavailable,
        Some(RequiredFutureAuditFailureV2::Snapshot {
            reason: "structured_v2_seed_required" | "cost_unavailable",
            ..
        })
    ));
    assert!(report.queries.is_empty());
    assert!(!report.completed_all_declared_paths);
    assert_eq!(executor.entries.load(Ordering::Acquire), 0);
    let work = frontier(&session, &id)
        .prefill_work(NonZeroU32::new(4).unwrap())
        .unwrap();
    wave(&mut session, &executor, vec![work]).await;
    drop(bounded(output.frames.next()).await.unwrap());
    assert_eq!(frontier(&session, &id).generated_tokens(), 1);
    let executed = executor.entries.load(Ordering::Acquire);
    assert!(session
        .audit_required_owners_v2(&before, &plan())
        .await
        .is_err());
    assert_eq!(executor.entries.load(Ordering::Acquire), executed);
    drop(output);
    session.shutdown().await.unwrap();
}

#[test]
fn required_future_audit_plan_bounds_precede_any_projector() {
    let mut p = plan();
    p.validate(1).unwrap();
    let duplicate = p.paths[0].waves[0][0].clone();
    p.paths[0].waves[0].push(duplicate);
    assert!(p.validate(1).is_err());
    p = plan();
    p.paths[0].waves = vec![p.paths[0].waves[0].clone(); 17];
    assert!(p.validate(1).is_err());
    p = plan();
    p.limits.maximum_coordinates = NonZeroUsize::new(65_537).unwrap();
    assert!(p.validate(1).is_err());
}
