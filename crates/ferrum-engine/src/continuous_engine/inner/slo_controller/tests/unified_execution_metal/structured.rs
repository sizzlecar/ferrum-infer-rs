//! Native eager whole-wave -> actual private host settlement. No fitted cost,
//! profile import, latency target or prospective per-class work is manufactured.
use super::*;

#[tokio::test]
async fn structured_capture_metal_future_actual_and_real_terminal_settlement() {
    let (mut session, directory) = fixture::fixture_with_structured_capture(true).await;
    let inner = session.test_engine_inner();
    let (id, output) = add(&mut session).await;
    admit(&mut session, &id).await;
    let first_work = frontier(&session, &id).prefill_work(n32(1)).unwrap();
    let first = wave(&mut session, vec![first_work]).await;
    let first_stages = first
        .host_stages
        .as_ref()
        .expect("actual partial host receipt");
    let first_qualified = first_stages
        .structured_evidence
        .as_ref()
        .unwrap()
        .as_ref()
        .unwrap();
    first_qualified.validate_host_stages(first_stages).unwrap();
    assert!(!first_qualified
        .recipe()
        .device()
        .algorithm_work()
        .unwrap()
        .entries()
        .is_empty());
    session.freeze_cost_model().await.unwrap();

    // Exact prefill successor and fresh decode snapshots, including the real
    // terminal Length wave. The original tiny fixture retains five outputs.
    let mut terminals = 0;
    for step in 0..6 {
        ready(&inner, &id).await;
        let captured = capture(&inner);
        let context = shape::ExecutorShape {
            engine: &inner,
            captured: &captured,
        };
        let parent = context.begin(&captured.snapshot, &mut || Ok(())).unwrap();
        let request = captured
            .snapshot
            .requests
            .iter()
            .find(|r| r.key.request_id == id)
            .unwrap();
        let action = if step == 0 {
            WaveAction::Prefill {
                offset: 1,
                count: n32(1),
            }
        } else if step == 1 {
            WaveAction::Prefill {
                offset: 2,
                count: n32(2),
            }
        } else {
            WaveAction::Decode
        };
        let work = [CandidateWork {
            key: request.key.clone(),
            action: action.clone(),
        }];
        let counters = inner.model_executor.cache_metrics_snapshot().unwrap()["counters"].clone();
        let projected = project(parent.as_ref(), &captured.snapshot.requests, &work)
            .expect("real native route");
        let exact = projected.canonical_domain.exact().unwrap().clone();
        let future = projected
            .statistical_evidence
            .as_ref()
            .unwrap()
            .exact()
            .unwrap()
            .structured_capture()
            .unwrap()
            .unwrap()
            .clone();
        future.validate_exact(&exact).unwrap();
        let future_work = future
            .device()
            .algorithm_work()
            .expect("every projected selected command captured");
        assert!(!future_work.entries().is_empty());
        assert_no_live_effects(&inner, &captured, &counters);
        drop(projected);
        drop(parent);
        drop(context);
        drop(captured);
        let f = frontier(&session, &id);
        let work = match action {
            WaveAction::Prefill { count, .. } => f.prefill_work(count).unwrap(),
            WaveAction::Decode => f.decode_work().unwrap(),
        };
        let report = wave(&mut session, vec![work]).await;
        assert!(report.error.is_none(), "{report:?}");
        assert_eq!(
            report.submission,
            CalibrationSubmissionState::HostReconciled
        );
        let stages = report
            .host_stages
            .as_ref()
            .expect("real joined host stages");
        let qualified = stages
            .structured_evidence
            .as_ref()
            .unwrap()
            .as_ref()
            .unwrap();
        qualified.validate_host_stages(stages).unwrap();
        assert_eq!(qualified.recipe(), future.as_ref());
        // Old Eq deliberately ignores passive capture. Compare the separate
        // checked per-algorithm evidence, including command assignment/order.
        assert_eq!(
            qualified.recipe().device().algorithm_work().unwrap(),
            future.device().algorithm_work().unwrap()
        );
        assert_eq!(
            stages.actual_shape.as_ref(),
            Some(&canonical_cost_shape(&exact).unwrap())
        );
        assert_eq!(stages.rows[0].request_id, id);
        terminals += usize::from(stages.rows[0].terminal.is_some());
        // Explicit raw capture field, not old statistical/profile serialization.
        assert!(serde_json::to_value(stages.as_ref())
            .unwrap()
            .get("structured_evidence")
            .is_none());
        let wire = serde_json::to_value(stages.structured_diagnostic_view()).unwrap();
        assert!(wire.get("structured_evidence").is_some());
        if step < 5 {
            assert_eq!(terminals, 0);
        }
    }
    assert_eq!(terminals, 1);
    assert!(session.frontiers().unwrap().is_empty());
    output.await.unwrap();
    session.shutdown().await.unwrap();
    drop(directory);
}
