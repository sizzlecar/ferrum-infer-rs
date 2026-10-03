//! Reuse the unchanged installed source8 model with a fresh real request.
//! Projection reads an undispatched guarded wave, then that same wave executes.
use super::*;
use crate::continuous_engine::inner::slo_controller::calibration::CalibrationPreparation;

mod witness;

pub(super) async fn submit_capacity_limited_witness(
    session: &CalibrationSession,
    executor: &ControlledExecutor,
    ids: &[RequestId],
) {
    witness::submit_capacity_limited(session, executor, ids).await;
}

pub(super) async fn submit_capacity_limited_greedy_witness(
    session: &CalibrationSession,
    executor: &ControlledExecutor,
    ids: &[RequestId],
) {
    witness::submit_capacity_limited_greedy(session, executor, ids).await;
}

pub(super) async fn submit_nonterminal_witness(
    session: &CalibrationSession,
    executor: &ControlledExecutor,
    ids: &[RequestId],
    requires_forward_tail: bool,
) {
    witness::submit_nonterminal(session, executor, ids, requires_forward_tail).await;
}

/// Shared by hand-declared and production-plan source8 populations.
/// This follows the original controller and checks its actual once-only receipt.
pub(super) async fn submit_terminal_witness(
    session: &CalibrationSession,
    executor: &ControlledExecutor,
    ids: &[RequestId],
) {
    witness::submit(session, executor, ids).await;
}

pub(super) async fn verify(mut session: CalibrationSession, executor: Arc<ControlledExecutor>) {
    let runtime = session.engine.inner.cost_runtime.as_ref().unwrap().clone();
    let receipt_before = runtime.profile_receipt().unwrap();
    // The completed source owns no active request or incomplete cohort. A new
    // private diagnostic session retains the exact same runtime, clock, model
    // and resource authorities; it cannot renew the original model's age.
    assert!(session.frontiers().unwrap().is_empty());
    session = CalibrationSession::new_driver_session(
        session.engine,
        CalibrationLimits::new(NonZeroUsize::new(2).unwrap()).unwrap(),
    );
    executor.enable_structured_query_route();
    executor
        .project_structured_cpu_fill
        .store(true, Ordering::Release);
    let mut ids = Vec::new();
    let mut outputs = Vec::new();
    for probe in probe_requests(&session) {
        ids.push(probe.request.id.clone());
        outputs.push(
            session
                .add_request(
                    probe.request,
                    InferenceRequestContext::capture(),
                    probe.contract,
                )
                .await
                .unwrap(),
        );
    }
    let consumers: Vec<_> = outputs
        .into_iter()
        .map(|mut output| {
            tokio::spawn(async move {
                let mut terminal = false;
                while let Some(frame) = output.frames.next().await {
                    terminal |= frame.metadata().terminal;
                    drop(frame);
                }
                assert!(terminal);
                let completed = output.completion.await.unwrap();
                assert!(matches!(
                    completed.payload(),
                    OutputCompletion::Succeeded {
                        reason: ferrum_types::FinishReason::Length,
                        ..
                    }
                ));
            })
        })
        .collect();
    for id in &ids {
        ready(&session, id, false).await;
    }
    for _ in &ids {
        admit(&mut session).await;
    }
    let prefills = ids
        .iter()
        .map(|id| {
            frontier(&session, id)
                .prefill_work(NonZeroU32::MIN)
                .unwrap()
        })
        .collect();
    let prepared = wave(&mut session, &executor, prefills).await;
    assert!(prepared.error.is_none(), "{prepared:?}");
    // This is an ordinary user prefill, not a new preparation capability.
    // Its actual returned token and host state establish the G2 projection.
    for id in &ids {
        ready(&session, id, false).await;
    }
    let inner = session.test_engine_inner();
    for generated in 2..=3 {
        for id in &ids {
            ready(&session, id, false).await;
        }
        if generated == 3 {
            witness::submit(&session, &executor, &ids).await;
            continue;
        }
        let timing_before = inner.controller_timing_snapshot().unwrap_or_default();
        let rows = ids
            .iter()
            .map(|id| {
                frontier(&session, id)
                    .decode_work_with_route(CalibrationDecodeRoute::FullLogits)
                    .unwrap()
            })
            .collect::<Vec<_>>();
        let before = executor.physical.load(Ordering::Acquire);
        let CalibrationPreparation::Selected(prepared) = inner
            .prepare_calibration_wave(&rows, NonZeroUsize::new(2).unwrap())
            .unwrap()
        else {
            panic!("original resource/output publication unavailable");
        };
        let facts = prepared.structured_prepared_facts(&inner, None).unwrap();
        facts.validate().unwrap();
        assert_eq!(
            executor.physical.load(Ordering::Acquire),
            before,
            "future projection must not execute a CPU wave"
        );
        for row in &facts.rows {
            assert!(ids.contains(&row.request_id));
            assert_eq!(row.frontier.generated_before, generated - 1);
        }
        let query = StructuredQueryV2::from_future_with_domain(
            &facts.exact,
            &facts.selected,
            &facts.recipe,
            &HostContentForecastV2::Exact,
            runtime.workload_domain().unwrap(),
        )
        .unwrap();
        let model = runtime.snapshot().expect("original V15 remains installed");
        let known = model
            .audit_structured_query_v2(&query, runtime.clock.now_ns().unwrap())
            .unwrap_or_else(|reason| panic!("actual undispatched G{generated} query: {reason:?}"));
        assert!(known.planning_ns > 0 && known.valid_for_ns > 0);
        assert_eq!(executor.physical.load(Ordering::Acquire), before);
        let receipt = prepared.calibration_receipt().unwrap();
        inner.execute_slo_controller_wave(prepared).await.unwrap();
        runtime.drain_calibration_fixture();
        assert_eq!(
            inner.controller_timing_snapshot().unwrap().witnesses,
            timing_before.witnesses,
            "manual CompleteRequests submission is not a CostWitness"
        );
        receipt.wait_observation().await;
        let report = receipt.report(None);
        assert!(report.error.is_none(), "{report:?}");
        assert_eq!(
            report.submission,
            CalibrationSubmissionState::HostReconciled
        );
        assert_eq!(executor.physical.load(Ordering::Acquire), before + 1);
        let stages = report.host_stages.as_ref().unwrap();
        let actual = stages
            .structured_evidence
            .as_ref()
            .unwrap()
            .as_ref()
            .unwrap();
        actual.validate_host_stages(stages).unwrap();
        assert_eq!(
            actual.recipe(),
            facts.recipe.as_ref(),
            "shared CPU algebra must match the subsequently executed recipe"
        );
        assert!(stages
            .rows
            .iter()
            .all(|row| row.terminal.is_some() == (generated == 3)));
    }
    for consumer in consumers {
        bounded(consumer).await.unwrap();
    }
    assert!(session.frontiers().unwrap().is_empty());
    let receipt_after = runtime.profile_receipt().unwrap();
    assert_eq!(
        receipt_after.source_observation_artifact_sha256,
        receipt_before.source_observation_artifact_sha256
    );
    assert_eq!(receipt_after.loaded_unix_ns, receipt_before.loaded_unix_ns);
    drop(inner);
    session.shutdown().await.unwrap();
}
