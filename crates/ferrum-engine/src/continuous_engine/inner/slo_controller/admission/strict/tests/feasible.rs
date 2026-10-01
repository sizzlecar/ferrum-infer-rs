//! Real controller search/replay/publication with the existing controlled CPU
//! provider algebra and qualified cost-fixture pipeline. The fixed sample
//! clock below is test data, not measured backend performance or calibration
//! evidence. No PlanningWitness/SelectedWave/Admit is constructed by the test.
use super::*;
use crate::continuous_engine::inner::cost_observation::*;
use ferrum_interfaces::execution_cost::*;
use ferrum_interfaces::ModelExecutor;
use std::sync::atomic::AtomicU64;

struct SampleClock(AtomicU64);
impl CostObservationClock for SampleClock {
    fn now_ns(&self) -> Option<u64> {
        Some(self.0.load(Ordering::Relaxed))
    }
}

/// Extend the fixture's existing calibrated-bucket source using the canonical
/// shape supplied by the real core/provider projection. This follows the same
/// sealed recorder -> trainer -> immutable snapshot path as train_runtime.
fn qualify_fixture_shape(
    runtime: &EngineCostRuntime,
    executor: &ControlledExecutor,
    canonical: &CanonicalWaveCostShape,
    host: Option<HostCostFeaturesV1>,
    policy: [u8; 32],
) {
    assert_eq!(canonical.rows.len(), 1);
    let ActualRowWork::Prefill {
        offset,
        count,
        total_prompt_tokens,
    } = canonical.rows[0]
    else {
        panic!("fixture source covers the actual final-prefill route only");
    };
    assert_eq!(offset + count, total_prompt_tokens);
    for _ in 0
        ..ferrum_scheduler::implementations::continuous::cost_model::CostModelSettings::default()
            .min_samples
            .get()
    {
        let id = RequestId::new();
        let clock = Arc::new(SampleClock(AtomicU64::new(2)));
        let mut call = EngineCostCall::begin(
            &runtime.ids,
            clock.clone(),
            runtime.sink.clone(),
            EngineCostCallSpec {
                identity: executor.execution_cost_identity(),
                participants: vec![CostObservationParticipant {
                    request_id: id.clone(),
                    owner_incarnation: 1,
                    work_generation: 1,
                    input_index: 0,
                    output_policy_signature: Some(policy),
                    host_features: host,
                }],
                prepare_started_at_ns: Some(1),
                boundary: WaveObservationBoundary::IsolatedPreparationToCommit,
                recorder_limits: runtime.recorder_limits,
            },
        )
        .unwrap();
        {
            let mut observation = call.context().unwrap();
            observation.physical_wave(
                Ok(ActualWaveShape {
                    statistical_evidence: None,
                    kind: canonical.kind,
                    path: canonical.path,
                    graph: canonical.graph,
                    row_order: canonical.row_order,
                    provider_signature: canonical.provider_signature,
                    output_policy_signature: canonical.output_policy_signature,
                    numeric_features: canonical.numeric_features.clone(),
                    host_content_features: canonical.host_content_features.clone(),
                    row_multiset_features: canonical.row_multiset_features.clone(),
                    rows: vec![ActualWaveRow {
                        request_id: id.clone(),
                        owner_incarnation: 1,
                        work_generation: 1,
                        input_index: 0,
                        work: canonical.rows[0],
                    }],
                    recurrent_state_bytes: canonical.recurrent_state_bytes,
                    restore_bytes: 0,
                    maintenance_bytes: 0,
                    maintenance_units: 0,
                }),
                Some(3),
            );
            clock.0.store(6, Ordering::Relaxed);
            observation.terminal(ActualWaveOutcome::Completed, None);
            observation.finish_call(ObservedCallOutcome::Completed);
        }
        call.record_host_result(HostCommitEvidence {
            request_id: id,
            owner_incarnation: 1,
            work_generation: 1,
            input_index: 0,
            outcome: HostCommitOutcome::Committed(HostCommittedWork::Prefill {
                start: offset,
                end: offset + count,
                total_prompt_tokens,
                generated_tokens_before: 0,
                generated_tokens_after: 1,
            }),
            committed_at_ns: Some(9),
        });
        clock.0.store(10, Ordering::Relaxed);
        assert_eq!(call.finish(), CostCallDisposition::Queued);
    }
    runtime.consume_samples();
    let model = runtime.snapshot().unwrap();
    assert!(
        PlanningCostModel::predict(
            model.as_ref(),
            model.fingerprint(),
            &canonical_cost_shape(canonical).unwrap(),
            11
        )
        .is_some(),
        "the actual immutable cost snapshot must cover the projected route"
    );
}

#[tokio::test]
async fn strict_real_planner_admits_publishes_and_returns_session_before_guarded_execution() {
    let (engine, _, executor) = fixture(1).await;
    executor.enable_structured_query_route();
    // One final prefill is a complete finite obligation. This fixture's
    // provider declares only prefill projection; no decode support is assumed.
    let request = input(&engine, 4, 1);
    let id = request.id.clone();
    let mut submitted = Box::pin(engine.infer_credited_stream(
        request,
        InferenceRequestContext::capture(),
        Arc::new(OutputProjectionContract::cli_text()),
    ));
    assert!(submitted.as_mut().now_or_never().is_none());
    ready(&engine, &id).await;
    prefill::admit(&engine, 1).await;
    let captured = prefill::captured(&engine, &executor).await;
    let execution = shape::ExecutorShape {
        engine: &engine.inner,
        captured: &captured,
    };
    let parent = execution.begin(&captured.snapshot, &mut || Ok(())).unwrap();
    let request = &captured.snapshot.requests[0];
    let work = [CandidateWork {
        key: request.key.clone(),
        action: WaveAction::Prefill {
            offset: 0,
            count: NonZeroU32::new(4).unwrap(),
        },
    }];
    let rows = [PlanningShapeRow {
        request,
        work: ActualRowWork::Prefill {
            offset: 0,
            count: 4,
            total_prompt_tokens: 4,
        },
    }];
    let projected = parent
        .project(
            &PlanningExecutionInput {
                work: &work,
                requests: &captured.snapshot.requests,
                kind: ActualWaveKind::Prefill,
                rows: &rows,
                recurrent_state_bytes: request.recurrent_state_bytes,
            },
            &mut || Ok(()),
        )
        .unwrap()
        .expect("declared controlled prefill provider projects a real route");
    let canonical = projected.canonical_domain.exact().unwrap().clone();
    let policy = host_history_cost_signature(request.output_policy_signature, 0);
    qualify_fixture_shape(
        engine.inner.cost_runtime.as_ref().unwrap(),
        &executor,
        &canonical,
        captured.fences[0].host_features,
        policy,
    );
    drop(projected);
    drop(parent);
    drop(execution);
    drop(captured);
    assert!(
        submitted.as_mut().now_or_never().is_none(),
        "qualification alone cannot accept a request"
    );
    assert_eq!(executor.entries.load(Ordering::Acquire), 0);
    let prepared = prefill::selected_after_retry(&engine, &executor, || {
        engine
            .inner
            .prepare_slo_controller(&ferrum_interfaces::BatchHint::simple(1))
    })
    .await;
    let mut session = bounded(submitted).await.unwrap();
    assert!(!engine.inner.sequences.read()[&id]
        .time_admission
        .as_ref()
        .unwrap()
        .before_acceptance());
    assert_eq!(
        executor.entries.load(Ordering::Acquire),
        0,
        "acceptance precedes native dispatch"
    );
    engine
        .inner
        .execute_slo_controller_wave(prepared)
        .await
        .unwrap();
    while let Some(frame) = bounded(session.frames.next()).await {
        drop(frame);
    }
    let completed = bounded(session.completion).await.unwrap();
    assert!(
        matches!(completed.payload(), OutputCompletion::Succeeded { usage, .. } if usage.completion_tokens == 1)
    );
    drop(completed);
    assert_eq!(executor.physical.load(Ordering::Acquire), 1);
    assert!(engine.inner.sequences.read().is_empty());
    engine.shutdown().await.unwrap();
}
