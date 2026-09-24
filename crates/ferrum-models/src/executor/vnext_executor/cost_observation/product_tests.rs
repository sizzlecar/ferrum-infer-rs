//! Real tiny Qwen product execution. Observation does not substitute a provider,
//! enable profiling, or turn a future route without declarations into Known.
use super::super::*;
use ferrum_interfaces::execution_cost::*;
use ferrum_kernels::backend::metal::{
    vnext_ops::MetalVNextComposition, vnext_runtime::MetalDeviceRuntime,
};
use ferrum_types::ModelId;
use std::num::NonZeroU64;
mod fixture;
mod parity;
mod resolver;
mod weights;
mod workspace_startup;
use fixture::{assert_prefill_same, prompt, Fixture};

struct Clock(AtomicU64);
impl CostObservationClock for Clock {
    fn now_ns(&self) -> Option<u64> {
        Some(self.0.fetch_add(1, Ordering::Relaxed))
    }
}
struct Probe {
    recorder: BoundedWaveRecorder,
    clock: Clock,
    participants: Vec<CostObservationParticipant>,
}
impl Probe {
    fn new(ids: &[&RequestId]) -> Self {
        Self {
            recorder: BoundedWaveRecorder::new(
                NonZeroU64::new(1).unwrap(),
                CostRecorderLimits {
                    max_waves: 4,
                    max_rows_per_wave: 8,
                    max_retained_rows: 32,
                },
            )
            .unwrap(),
            clock: Clock(AtomicU64::new(2)),
            participants: ids
                .iter()
                .enumerate()
                .map(|(index, id)| CostObservationParticipant {
                    request_id: (*id).clone(),
                    owner_incarnation: 17,
                    work_generation: 23,
                    input_index: index as u32,
                    output_policy_signature: Some([7; 32]),
                    host_features: None,
                })
                .collect(),
        }
    }
    fn context(&mut self) -> PlanRuntimeCostObservationContext<'_> {
        PlanRuntimeCostObservationContext::new(
            &mut self.recorder,
            &self.clock,
            &self.participants,
            Some(1),
            WaveObservationBoundary::ExecutorOnly,
        )
    }
    fn assert_wave(&self, kind: ActualWaveKind, works: &[ActualRowWork]) {
        let records = self.recorder.observations();
        assert_eq!(records.len(), 1);
        let wave = &records[0];
        assert_eq!(wave.outcome, Some(ActualWaveOutcome::Completed));
        assert_eq!(wave.boundary, WaveObservationBoundary::ExecutorOnly);
        assert!(wave.prepare_started_at_ns <= wave.submission_started_at_ns.unwrap());
        assert!(wave.submission_started_at_ns <= wave.terminal_at_ns);
        assert!(
            wave.host_committed_at_ns.is_none(),
            "executor must not invent engine commit"
        );
        assert!(
            wave.device_elapsed_ns.is_none(),
            "TimingOff is not a device timing sample"
        );
        let shape = wave
            .shape
            .as_ref()
            .expect("actual selected native route must be observable");
        assert_eq!(shape.kind, kind);
        assert_eq!(shape.rows.len(), self.participants.len());
        assert_eq!(shape.rows.len(), works.len());
        assert_eq!(shape.path, ActualWavePath::PlanRuntime);
        assert_eq!(shape.graph, ActualWaveGraphState::Disabled);
        // Device canonical order may differ from caller input order. Preserve
        // the actual order and join only by the complete correlated identity.
        let mut seen = BTreeSet::new();
        for row in &shape.rows {
            let index = row.input_index as usize;
            assert!(seen.insert(index));
            let expected = &self.participants[index];
            assert_eq!(row.request_id, expected.request_id);
            assert_eq!(row.owner_incarnation, expected.owner_incarnation);
            assert_eq!(row.work_generation, expected.work_generation);
            assert_eq!(row.work, works[index]);
        }
    }
}
fn executed<T>(value: ObservedDispatch<T>) -> T {
    match value {
        ObservedDispatch::Executed(result) => result.unwrap(),
        ObservedDispatch::Unavailable => panic!("real product observed entry did no work"),
    }
}
fn decode_outputs(outcome: PlanRuntimeBatchDecodeOutcome) -> Vec<PlanRuntimeDecodeOutput> {
    match outcome {
        PlanRuntimeBatchDecodeOutcome::Completed(outputs) => outputs,
        PlanRuntimeBatchDecodeOutcome::Deferred(reason) => panic!("decode deferred: {reason:?}"),
    }
}
fn assert_decode_same(actual: &[PlanRuntimeDecodeOutput], expected: &[PlanRuntimeDecodeOutput]) {
    assert_eq!(actual.len(), expected.len());
    for (a, b) in actual.iter().zip(expected) {
        assert_eq!(a.kv_cache.num_tokens(), b.kv_cache.num_tokens());
        match (&a.sampling_output, &b.sampling_output) {
            (ExecutorSamplingOutput::FullLogits(a), ExecutorSamplingOutput::FullLogits(b)) => {
                fixture::assert_logits_same(a, b)
            }
            (ExecutorSamplingOutput::GreedyToken(a), ExecutorSamplingOutput::GreedyToken(b)) => {
                assert_eq!(a, b)
            }
            _ => panic!("observation changed actual product mode"),
        }
    }
}

#[tokio::test]
async fn observed_metal_product_partial_final_decode_mixed_matches_ordinary() {
    let actual = Fixture::new(8, false).await;
    let baseline = Fixture::new(8, false).await;
    assert_eq!(actual.executor.device_timing_mode(), DeviceTimingMode::Off);
    assert_eq!(
        actual.executor.execution_cost_observation_capability(),
        ExecutorCostObservationCapability::SinglePhysicalWave
    );
    let a = prompt(&[0, 1, 2, 1], 2);
    let b = prompt(&[0, 1, 2, 1], 2);
    actual.admit(&a);
    baseline.admit(&b);
    let mut probe = Probe::new(&[&a.request_id]);
    let before = actual.submissions();
    let partial = match executed(
        actual
            .executor
            .plan_runtime_prefill_with_capacity_observed(&a, &mut probe.context())
            .await,
    ) {
        PlanRuntimePrefillOutcome::Completed(output) => output,
        _ => panic!("partial did not complete"),
    };
    let ordinary = match baseline
        .executor
        .plan_runtime_prefill_with_capacity(&b)
        .await
        .unwrap()
    {
        PlanRuntimePrefillOutcome::Completed(output) => output,
        _ => panic!("ordinary partial did not complete"),
    };
    assert_prefill_same(&partial, &ordinary);
    assert_eq!(actual.submissions(), before + 1);
    probe.assert_wave(
        ActualWaveKind::Prefill,
        &[ActualRowWork::Prefill {
            offset: 0,
            count: 2,
            total_prompt_tokens: 4,
        }],
    );

    let a = PlanRuntimePrefillInput::new(
        a.request_id.clone(),
        a.input_tokens.clone(),
        a.maximum_sequence_tokens,
        PrefillChunk::new(2, 2, 4).unwrap(),
    )
    .unwrap();
    let b = PlanRuntimePrefillInput::new(
        b.request_id.clone(),
        b.input_tokens.clone(),
        b.maximum_sequence_tokens,
        PrefillChunk::new(2, 2, 4).unwrap(),
    )
    .unwrap();
    let mut probe = Probe::new(&[&a.request_id]);
    let final_output = match executed(
        actual
            .executor
            .plan_runtime_batch_prefill_with_capacity_observed(
                std::slice::from_ref(&a),
                &mut probe.context(),
            )
            .await,
    ) {
        PlanRuntimeBatchPrefillOutcome::Completed(mut output) => output.remove(0),
        _ => panic!("final did not complete"),
    };
    let ordinary = match baseline
        .executor
        .plan_runtime_batch_prefill_with_capacity(std::slice::from_ref(&b))
        .await
        .unwrap()
    {
        PlanRuntimeBatchPrefillOutcome::Completed(mut output) => output.remove(0),
        _ => panic!("ordinary final did not complete"),
    };
    assert_prefill_same(&final_output, &ordinary);
    probe.assert_wave(
        ActualWaveKind::Prefill,
        &[ActualRowWork::Prefill {
            offset: 2,
            count: 2,
            total_prompt_tokens: 4,
        }],
    );
    let a_decode = PlanRuntimeDecodeInput::new(
        a.request_id.clone(),
        TokenId::new(1),
        Arc::clone(final_output.output().kv_cache()),
    );
    let b_decode = PlanRuntimeDecodeInput::new(
        b.request_id.clone(),
        TokenId::new(1),
        Arc::clone(ordinary.output().kv_cache()),
    );
    let mut probe = Probe::new(&[&a.request_id]);
    let decoded = decode_outputs(executed(
        actual
            .executor
            .plan_runtime_batch_decode_with_capacity_observed(
                std::slice::from_ref(&a_decode),
                &mut probe.context(),
            )
            .await,
    ));
    let ordinary_decoded = decode_outputs(
        baseline
            .executor
            .plan_runtime_batch_decode_with_capacity(std::slice::from_ref(&b_decode))
            .await
            .unwrap(),
    );
    assert_decode_same(&decoded, &ordinary_decoded);
    probe.assert_wave(
        ActualWaveKind::Decode,
        &[ActualRowWork::Decode { kv_tokens: 4 }],
    );

    let a_decode = PlanRuntimeDecodeInput::new(
        a.request_id.clone(),
        TokenId::new(1),
        Arc::clone(&decoded[0].kv_cache),
    );
    let b_decode = PlanRuntimeDecodeInput::new(
        b.request_id.clone(),
        TokenId::new(1),
        Arc::clone(&ordinary_decoded[0].kv_cache),
    );

    let a_new = prompt(&[1, 0], 2);
    let b_new = prompt(&[1, 0], 2);
    actual.admit(&a_new);
    baseline.admit(&b_new);
    let mut probe = Probe::new(&[&a_new.request_id, &a_decode.request_id]);
    let mixed = executed(
        actual
            .executor
            .plan_runtime_mixed_batch_with_capacity_observed(
                std::slice::from_ref(&a_new),
                std::slice::from_ref(&a_decode),
                &mut probe.context(),
            )
            .await,
    );
    let ordinary = baseline
        .executor
        .plan_runtime_mixed_batch_with_capacity(
            std::slice::from_ref(&b_new),
            std::slice::from_ref(&b_decode),
        )
        .await
        .unwrap();
    let (
        PlanRuntimeMixedBatchOutcome::Completed {
            prefills: a_p,
            decodes: a_d,
        },
        PlanRuntimeMixedBatchOutcome::Completed {
            prefills: b_p,
            decodes: b_d,
        },
    ) = (mixed, ordinary)
    else {
        panic!("real mixed execution did not complete")
    };
    assert_eq!(a_p.len(), 1);
    assert_eq!(b_p.len(), 1);
    assert_prefill_same(&a_p[0], &b_p[0]);
    assert_decode_same(&a_d, &b_d);
    probe.assert_wave(
        ActualWaveKind::Mixed,
        &[
            ActualRowWork::Prefill {
                offset: 0,
                count: 2,
                total_prompt_tokens: 2,
            },
            ActualRowWork::Decode { kv_tokens: 5 },
        ],
    );
    assert_eq!(actual.executor.device_timing_mode(), DeviceTimingMode::Off);
    // Retire product owners with ordinary release; observation owns no lease.
    actual.executor.release_cache(&a_decode.kv_cache.cache_id());
    baseline
        .executor
        .release_cache(&b_decode.kv_cache.cache_id());
    actual
        .executor
        .discard_plan_runtime_prefill(
            a_p.into_iter()
                .next()
                .unwrap()
                .into_parts()
                .0
                .into_parts()
                .0,
        )
        .unwrap();
    baseline
        .executor
        .discard_plan_runtime_prefill(
            b_p.into_iter()
                .next()
                .unwrap()
                .into_parts()
                .0
                .into_parts()
                .0,
        )
        .unwrap();
}

#[tokio::test]
async fn observed_metal_product_missing_correlation_does_not_replay_or_hide_empty_call() {
    let fixture = Fixture::new(8, false).await;
    let input = prompt(&[0, 1], 2);
    fixture.admit(&input);
    let mut probe = Probe::new(&[]);
    let before = fixture.submissions();
    let (output, count, reason, outcome) = {
        let mut context = probe.context();
        let output = executed(
            fixture
                .executor
                .plan_runtime_prefill_with_capacity_observed(&input, &mut context)
                .await,
        );
        (
            output,
            context.physical_wave_count(),
            context.unknown_reason(),
            context.call_outcome(),
        )
    };
    assert_eq!(count, 1);
    assert_eq!(fixture.submissions(), before + 1);
    assert_eq!(
        reason,
        Some(ActualWaveEvidenceUnknown::ParticipantCorrelation)
    );
    assert_eq!(outcome, Some(ObservedCallOutcome::Completed));
    assert!(probe.recorder.observations()[0].shape.is_none());
    assert_eq!(
        probe.recorder.observations()[0].outcome,
        Some(ActualWaveOutcome::Completed)
    );
    let PlanRuntimePrefillOutcome::Completed(output) = output else {
        panic!("actual prefill did not complete")
    };
    fixture
        .executor
        .discard_plan_runtime_prefill(output.into_parts().0.into_parts().0)
        .unwrap();
    let mut empty = Probe::new(&[]);
    let before = fixture.submissions();
    let mut context = empty.context();
    assert!(
        matches!(executed(fixture.executor.plan_runtime_batch_decode_with_capacity_observed(&[],&mut context).await),PlanRuntimeBatchDecodeOutcome::Completed(outputs) if outputs.is_empty())
    );
    assert_eq!(context.call_outcome(), Some(ObservedCallOutcome::Completed));
    assert_eq!(context.physical_wave_count(), 0);
    assert_eq!(fixture.submissions(), before);
}
