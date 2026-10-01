use super::*;
use crate::continuous_engine::inner::slo_controller::tests::{fixture, prefill};
use ferrum_interfaces::engine::InferenceEngine;
use ferrum_interfaces::model_executor::PrefixCapturePlan;
use ferrum_interfaces::vnext::CheckpointTokenSpanConstraint;
use std::sync::atomic::Ordering;

#[tokio::test]
async fn modeled_hold_matches_only_current_cohort_owners_and_complete_fences() {
    let (engine, scheduler, executor) = fixture::fixture_with_width(2).await;
    // The existing exact reference fixture declares lengths 1, 2 and 4.
    // Identity checks need a legal partial boundary, not a new reference domain.
    let (source, source_session) = prefill::request(&engine, 4, 4).await;
    let (target, target_session) = prefill::request(&engine, 4, 4).await;
    prefill::admit(&engine, 2).await;
    let candidates = scheduler.prefix_rendezvous_candidates();
    let source_key = candidates
        .iter()
        .find(|c| c.key.request_id() == &source)
        .unwrap()
        .key
        .clone();
    let target_key = candidates
        .iter()
        .find(|c| c.key.request_id() == &target)
        .unwrap()
        .key
        .clone();
    let hold = scheduler
        .hold_admitted_prefix_follower(
            &source_key,
            &target_key,
            PrefixCapturePlan {
                boundary: 2,
                span: CheckpointTokenSpanConstraint::new(NonZeroU64::MIN, NonZeroU64::MIN).unwrap(),
            },
        )
        .unwrap();
    let mut captured = prefill::captured(&engine, &executor).await;
    let source_index = captured
        .fences
        .iter()
        .position(|f| f.key.request_id == source)
        .unwrap();
    let target_index = captured
        .fences
        .iter()
        .position(|f| f.key.request_id == target)
        .unwrap();
    let source_incarnation = captured.fences[source_index].incarnation;
    let target_incarnation = captured.fences[target_index].incarnation;
    let matches = |fences: &[EngineFence], source_owner, target_owner| {
        matching_target(
            &hold,
            &target_key,
            source_owner,
            target_owner,
            &captured.queue,
            fences,
        )
        .is_some()
    };
    assert!(matches(
        &captured.fences,
        source_incarnation,
        target_incarnation
    ));
    assert!(
        captured.snapshot.has_unmodeled_maintenance,
        "a hold without an installed engine cohort is still unmodeled"
    );
    assert!(
        !captured
            .queue
            .requests()
            .iter()
            .find(|r| r.key.request_id == target)
            .unwrap()
            .readiness
            .ready(),
        "classification must not make a held follower executable"
    );

    assert!(!matches(
        &captured.fences,
        source_incarnation + 1,
        target_incarnation
    ));
    assert!(!matches(
        &captured.fences,
        source_incarnation,
        target_incarnation + 1
    ));
    assert!(!matches(
        &captured.fences[..1],
        source_incarnation,
        target_incarnation
    ));
    assert!(matching_target(
        &hold,
        &source_key,
        source_incarnation,
        target_incarnation,
        &captured.queue,
        &captured.fences
    )
    .is_none());

    // The engine fence must match the current scheduler ticket, not only ID.
    let source_ticket = captured.fences[source_index].key.ticket;
    let target_ticket = captured.fences[target_index].key.ticket;
    captured.fences[target_index].key.ticket = source_ticket;
    assert!(!matches(
        &captured.fences,
        source_incarnation,
        target_incarnation
    ));
    captured.fences[target_index].key.ticket = target_ticket;
    assert!(matches(
        &captured.fences,
        source_incarnation,
        target_incarnation
    ));

    hold.release();
    assert!(!matches(
        &captured.fences,
        source_incarnation,
        target_incarnation
    ));
    assert_eq!(executor.physical.load(Ordering::Acquire), 0);
    drop(captured);
    drop(hold);
    drop((source_session, target_session));
    engine.shutdown().await.unwrap();
}
