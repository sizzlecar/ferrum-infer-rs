use super::*;
use ferrum_execution::model_executor::{PrefillShapePolicy, ResidentSequenceCapacityRequest};
use ferrum_kernels::backend::cpu::vnext_ops::CpuVNextComposition;
use ferrum_models::test_support::native_program::{Fixture, FixtureOptions};
use ferrum_runtime::DeviceId;
use ferrum_scheduler::cost_prediction::{
    CostExecution, CostPrediction, CostQuery, CostSnapshot, CostUnknownReason,
};
use ferrum_types::SamplingReadbackMode;

struct Snapshot;
impl CostSnapshot for Snapshot {
    fn revision(&self) -> u64 {
        73
    }
    fn query_limits(&self) -> ferrum_types::CostQueryLimits {
        Default::default()
    }
    fn predict(&self, _: &CostQuery<'_>, _: Instant) -> CostPrediction {
        CostPrediction::Unknown(CostUnknownReason::NoCoverage)
    }
}

// These are real product admissions into a real CPU ProgramExecutor. The
// fixture's native embedding/head program needs no attention KV; its owner
// still enforces the admitted request and sequence ceilings.
#[tokio::test]
async fn planning_resident_capacity_uses_actual_program_admission_ceiling() {
    for cap in [1, 64] {
        let (runtime, registry, materializers, _, catalog) = CpuVNextComposition::create(
            DeviceId::new("device.test.engine-resident-capacity").unwrap(),
            256 << 20,
        )
        .unwrap()
        .into_parts()
        .unwrap();
        let mut config = EngineConfig::default();
        let fixture = Fixture::try_new_with_engine_config(
            SamplingReadbackMode::Uniform,
            Device::CPU,
            runtime,
            registry,
            materializers,
            catalog,
            FixtureOptions {
                maximum_tokens: 128,
                ..Default::default()
            },
            |native_config| {
                native_config.backend.device = Device::CPU;
                config = native_config.clone();
            },
        )
        .unwrap();
        let engine = ContinuousBatchEngine::new_observed_with_requests(
            config.clone(),
            Arc::new(ContinuousBatchScheduler::new(config.scheduler)),
            tiny_tokenizer(&TinyLlamaConfig::default()),
            Arc::new(crate::registry::GreedySampler),
            ExecutorOwner::new(Box::new(fixture.executor)),
            None,
            None,
            None,
            Some(Arc::new(Recorder::default())),
        )
        .unwrap();
        let tokens = vec![TokenId::new(0), TokenId::new(1)];
        let expected_ceiling = tokens.len() + cap - 1;
        let mut request = request();
        request.sampling_params.max_tokens = cap;
        request.ensure_ingress(ferrum_types::RequestIngressBoundary::EngineApi);
        let id = request.id.clone();
        let mut sequence = SequenceState::new(request.clone(), tokens);
        assert_eq!(sequence.model_maximum_sequence_tokens(), expected_ceiling);
        let (sender, _response) = tokio::sync::oneshot::channel();
        sequence.response_sender = Some(sender);
        engine.publish_request(request, sequence).await.unwrap();
        let inner = engine.active_inner().unwrap();
        let mut round = TimePlanningRound::new(
            Arc::new(Snapshot),
            "ttft:1000,tpot:100,itl:100".parse().unwrap(),
            PrefillShapePolicy::Identity,
            CostExecution::SingleInvocation,
        );
        inner.capture_planning_request_facts(&mut round, 1);
        assert!(
            round.resident_capacity().is_none(),
            "waiting is not resident"
        );
        let waiting_proof = round
            .persistent_capacity()
            .expect("base plan waiting capacity is observed");
        assert_eq!(waiting_proof.members()[0].through_tokens, expected_ceiling);
        assert!(matches!(
            waiting_proof.members()[0].state,
            ferrum_execution::model_executor::PersistentCapacityState::Waiting
        ));
        // A future adapter load is not described by the base plan's ceiling.
        // Reprojection must clear the previous successful observation.
        let waiting_handle = inner.sequences.get(&id).unwrap();
        {
            let mut state = waiting_handle.lock();
            let metadata = &mut state.as_mut().unwrap().original_request.metadata;
            metadata.insert("ferrum_lora_adapter".into(), "tenant-a".into());
            metadata.insert("ferrum_lora_path".into(), "/adapters/not-loaded".into());
        }
        inner.capture_planning_request_facts(&mut round, 1);
        assert!(round.persistent_capacity().is_none());
        {
            let mut state = waiting_handle.lock();
            let metadata = &mut state.as_mut().unwrap().original_request.metadata;
            metadata.remove("ferrum_lora_adapter");
            metadata.remove("ferrum_lora_path");
        }
        drop(waiting_handle);
        inner.capture_planning_request_facts(&mut round, 1);
        assert!(round.persistent_capacity().is_some());
        tokio::time::timeout(std::time::Duration::from_secs(10), async {
            loop {
                let (admitted, maintenance) = inner
                    .prepare_dynamic_admission_round(1, None)
                    .await
                    .unwrap();
                if admitted == 1 {
                    assert!(maintenance.is_empty());
                    break;
                }
                assert!(!maintenance.is_empty(), "admission must make progress");
                inner.execute_executor_prefill_maintenance(maintenance);
            }
        })
        .await
        .expect("native admission and maintenance must complete");
        inner.capture_planning_request_facts(&mut round, 1);
        let proof = round
            .resident_capacity()
            .expect("actual admission is covered");
        assert_eq!(proof.participants, 1);
        assert_eq!(proof.max_batch_sequences, 1);
        assert_eq!(round.request_facts().unwrap()[0].1.cap, cap);
        assert_eq!(round.revision(), 73);

        let handle = inner.sequences.get(&id).unwrap();
        let frontier = handle
            .lock()
            .as_ref()
            .unwrap()
            .execution_frontier
            .clone()
            .unwrap();
        // Reproduce the formerly used prompt + cap query against the same real
        // admitted owner. It exceeds the ceiling by the final unconsumed token.
        assert!(inner
            .model_executor
            .resident_cohort_capacity_snapshot(&[ResidentSequenceCapacityRequest {
                sequence: frontier.sequence().clone(),
                through_tokens: expected_ceiling + 1,
            },])
            .is_none());

        // Recapture must fail closed if immutable stopping facts no longer
        // describe the admitted request, even when an older proof was known.
        handle.lock().as_mut().unwrap().sampling_params.max_tokens = cap + 1;
        inner.capture_planning_request_facts(&mut round, 1);
        assert!(round.resident_capacity().is_none());
        handle.lock().as_mut().unwrap().sampling_params.max_tokens = cap;
        inner.capture_planning_request_facts(&mut round, 1);
        assert!(round.resident_capacity().is_some());
        drop(handle);
        drop(inner);
        engine.shutdown().await.unwrap();
        engine.wait_for_shutdown_completion().await.unwrap();
    }
}
