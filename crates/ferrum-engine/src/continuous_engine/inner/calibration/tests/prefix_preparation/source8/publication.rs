//! Actual engine producers and actor acknowledgements, then the original
//! source8 checkpoint and worker activation. No imported test model is built.
use super::*;
use crate::continuous_engine::inner::calibration::cohort_driver::{
    ProbeCohortSettings, ProbeExecutionBudget, ProbeRequest,
};
use ferrum_interfaces::execution_cost::CostObservationClock;
use std::sync::atomic::AtomicU64;

struct AdvancingClock(AtomicU64);
impl CostObservationClock for AdvancingClock {
    fn now_ns(&self) -> Option<u64> {
        Some(self.0.fetch_add(1, Ordering::Relaxed))
    }
}

async fn automatic_session() -> (CalibrationSession, Arc<ControlledExecutor>) {
    automatic_session_with_wave_tokens(2).await
}

/// A separate opt-in fixture for real intermediate/final prompt execution.
/// The older restart, import and guard tests retain their stateless defaults.
async fn automatic_checkpoint_session(
    tokens: u64,
) -> (CalibrationSession, Arc<ControlledExecutor>) {
    automatic_session_with_storage_and_worker_options(
        tokens,
        Arc::new(AdvancingClock(AtomicU64::new(100))),
        Default::default(),
        ferrum_types::SloAutomaticCalibrationReuseV1::Disabled {},
        FixtureCostWorker::Background,
        true,
        false,
    )
    .await
}

async fn automatic_session_with_wave_tokens(
    tokens: u64,
) -> (CalibrationSession, Arc<ControlledExecutor>) {
    automatic_session_with_clock(tokens, Arc::new(AdvancingClock(AtomicU64::new(100)))).await
}

async fn automatic_session_with_clock(
    tokens: u64,
    clock: Arc<dyn CostObservationClock>,
) -> (CalibrationSession, Arc<ControlledExecutor>) {
    automatic_session_with_diagnostics(tokens, clock, Default::default()).await
}

async fn automatic_session_with_diagnostics(
    tokens: u64,
    clock: Arc<dyn CostObservationClock>,
    diagnostics: ferrum_types::SloAutomaticCalibrationDiagnosticsV1,
) -> (CalibrationSession, Arc<ControlledExecutor>) {
    automatic_session_with_storage(
        tokens,
        clock,
        diagnostics,
        ferrum_types::SloAutomaticCalibrationReuseV1::Disabled {},
    )
    .await
}

async fn automatic_session_with_storage(
    tokens: u64,
    clock: Arc<dyn CostObservationClock>,
    diagnostics: ferrum_types::SloAutomaticCalibrationDiagnosticsV1,
    reuse: ferrum_types::SloAutomaticCalibrationReuseV1,
) -> (CalibrationSession, Arc<ControlledExecutor>) {
    automatic_session_with_storage_and_worker(
        tokens,
        clock,
        diagnostics,
        reuse,
        FixtureCostWorker::Background,
    )
    .await
}

enum FixtureCostWorker {
    Background,
    Manual,
}

async fn automatic_session_with_storage_and_worker(
    tokens: u64,
    clock: Arc<dyn CostObservationClock>,
    diagnostics: ferrum_types::SloAutomaticCalibrationDiagnosticsV1,
    reuse: ferrum_types::SloAutomaticCalibrationReuseV1,
    worker: FixtureCostWorker,
) -> (CalibrationSession, Arc<ControlledExecutor>) {
    automatic_session_with_storage_and_worker_options(
        tokens,
        clock,
        diagnostics,
        reuse,
        worker,
        false,
        false,
    )
    .await
}

async fn automatic_session_with_storage_and_worker_options(
    tokens: u64,
    clock: Arc<dyn CostObservationClock>,
    diagnostics: ferrum_types::SloAutomaticCalibrationDiagnosticsV1,
    reuse: ferrum_types::SloAutomaticCalibrationReuseV1,
    worker: FixtureCostWorker,
    checkpoint: bool,
    acquisition: bool,
) -> (CalibrationSession, Arc<ControlledExecutor>) {
    let (mut engine, executor) = if acquisition {
        prepared_engine_with_state_and_prefix_policy(
            2,
            Some(NonZeroU64::new(tokens).unwrap()),
            true,
        )
        .await
    } else if checkpoint {
        prepared_checkpoint_engine_with_width(2, NonZeroU64::new(tokens).unwrap()).await
    } else {
        prepared_engine_with_width(2).await
    };
    executor.prepare_structured_query_resources();
    let inner = Arc::get_mut(&mut engine.inner).unwrap();
    inner.config.scheduler.slo.mode = ferrum_types::SloMode::Enforce;
    inner.config.scheduler.slo.cost_profile = None;
    inner
        .config
        .scheduler
        .slo
        .output
        .max_queued_events_per_request =
        ferrum_types::SloOutputConfig::default().max_queued_events_per_request;
    let mut config = ferrum_types::SloCostObservationConfig::structured_whole_wave_v2();
    config.live_structured_calibration = ferrum_types::SloLiveStructuredCalibration::AutomaticV1 {
        settings: ferrum_types::SloAutomaticCalibrationSettingsV1 {
            diagnostics,
            reuse,
            ..Default::default()
        },
    };
    let identity = inner.cost_runtime.as_ref().unwrap().identity.clone();
    let ExecutorCostIdentityAvailability::Known(known) = &identity else {
        panic!("real controlled executor identity");
    };
    let domain = if checkpoint {
        let ferrum_interfaces::execution_cost::CostWorkloadDomainAvailability::Known(domain) =
            inner.model_executor.cost_workload_domain()
        else {
            panic!("actual native checkpoint program must declare its physical state domain");
        };
        assert_eq!(
            domain.limits().maximum_scheduled_tokens_per_wave.get(),
            tokens
        );
        assert!(domain.limits().fixed_state_bytes_per_row > 0);
        domain.as_ref().clone()
    } else {
        CostWorkloadDomainV1::new_vnext(
            known,
            CostWorkloadLimitsV1 {
                maximum_rows: NonZeroU32::new(2).unwrap(),
                maximum_context_tokens: NonZeroU32::new(
                    inner
                        .model_executor
                        .capabilities()
                        .max_sequence_length
                        .min(
                            inner
                                .model_executor
                                .kv_capacity()
                                .unwrap_or(inner.model_executor.capabilities().max_sequence_length),
                        )
                        .try_into()
                        .unwrap(),
                )
                .unwrap(),
                maximum_scheduled_tokens_per_wave: NonZeroU64::new(tokens).unwrap(),
                output_vocabulary_elements: NonZeroU64::new(
                    inner.model_executor.info().vocab_size as u64,
                )
                .unwrap(),
                repetition_slot_capacity: 0,
                fixed_state_bytes_per_row: 0,
            },
        )
        .unwrap()
    };
    inner.cost_runtime = Some(Arc::new(
        EngineCostRuntime::build_with_clock_and_domain(
            identity,
            clock,
            &config,
            matches!(worker, FixtureCostWorker::Background),
            domain,
        )
        .unwrap(),
    ));
    inner.config.scheduler.slo.cost_observation = config;
    executor
        .emit_structured_cost_observations
        .store(true, Ordering::Release);
    executor
        .recycle_completed_bindings
        .store(true, Ordering::Release);
    ContinuousBatchEngine::check_startup_session(&mut engine).unwrap();
    (
        CalibrationSession::new_driver_session(
            engine,
            CalibrationLimits::new(NonZeroUsize::new(2).unwrap()).unwrap(),
        ),
        executor,
    )
}

fn full_population(session: &CalibrationSession) -> StructuredPreparedOwnerBlockDeclarationV8 {
    let mut value = declaration(session, 3, false);
    value.population.schedule = OwnerBlockScheduleV1::new(24, [24; 3], [8; 3]).unwrap();
    value.cohort_plan.phases = std::array::from_fn(|phase| {
        (0..[16, 8, 8][phase])
            .map(|ordinal| CohortV2 {
                manifest_case: ordinal as u32,
                repetition: 0,
                requests: vec![
                    CohortRequestV2 {
                        manifest_prompt: 0,
                        maximum_output: 3
                    };
                    2
                ],
            })
            .collect()
    });
    let tokenizer = session
        .engine
        .inner
        .tokenizer
        .host_output_policy_identity()
        .unwrap();
    value.prefix_plan.phases = std::array::from_fn(|phase| {
        value.cohort_plan.phases[phase]
            .iter()
            .map(|cohort| {
                Some(StructuredPrefixCohortV5 {
                    release_generated: 1,
                    slots: (0..2)
                        .map(|slot| {
                            // Both empty, both pending, and each mixed position. These
                            // are genuine tokenizer bytes committed by prefix mode.
                            let pending = match cohort.manifest_case % 4 {
                                0 => false,
                                1 => true,
                                2 => slot == 0,
                                _ => slot == 1,
                            };
                            StructuredPrefixSlotV5 {
                                tokenizer_policy_sha256: tokenizer,
                                token_ids: vec![TokenId::new(if pending { 11 } else { 10 })],
                                token_bytes: vec![if pending { vec![0xc3] } else { b"a".to_vec() }],
                            }
                        })
                        .collect(),
                })
            })
            .collect()
    });
    value.maximum_offered_waves = 96;
    value.cohort_manifest_payload = serde_json::value::to_raw_value(&serde_json::json!({
        "prompt":"test", "maximum_output":3, "prefix_cases":["empty","pending","pending_empty","empty_pending"],
        "outputs":["cli_text","completions_sse"], "sampling":"installed_full_vocabulary_top_k"
    })).unwrap();
    value
}

fn probe_requests(session: &CalibrationSession) -> Vec<ProbeRequest> {
    requests(session, 3, false)
        .into_iter()
        .map(|(mut request, contract)| {
            // Explicit full-vocabulary top-k selects the real FullLogits path
            // without removing declared prefix tokens from processed logits.
            // Temperature zero keeps the original host sampler deterministic.
            request.sampling_params.temperature = 0.0;
            request.sampling_params.top_k =
                Some(session.engine.inner.model_executor.info().vocab_size);
            request.sampling_params.stop_sequences.clear();
            ProbeRequest {
                request,
                contract: Arc::new(contract),
            }
        })
        .collect()
}

#[tokio::test]
async fn source8_engine_actual_independent_populations_activate_memory_catalog() {
    let (mut session, executor) = automatic_session().await;
    let runtime = session.engine.inner.cost_runtime.as_ref().unwrap().clone();
    assert!(runtime.snapshot().is_none());
    let declared = full_population(&session);
    assert_eq!(
        declared
            .population
            .nonnegative_envelope
            .as_ref()
            .unwrap()
            .workload_domain,
        *runtime.workload_domain().unwrap()
    );
    session
        .begin_prepared_owner_source(declared, CostProfileLoadLimits::default())
        .await
        .unwrap();
    let mut budget = ProbeExecutionBudget::new(
        tokio::time::Instant::now() + Duration::from_secs(45),
        NonZeroUsize::new(64).unwrap(),
        NonZeroUsize::new(96).unwrap(),
    );
    let options = ProbeCohortSettings {
        prefill_plan:
            crate::continuous_engine::inner::calibration::cohort_driver::ProbePrefillPlan::Joint,
        prefill_chunk: NonZeroU32::MIN,
        decode_route: CalibrationDecodeRoute::FullLogits,
        reset_token_policy: false,
    };
    for pass in 0..3 {
        for ordinal in 0..[16, 8, 8][pass] {
            session.begin_prepared_owner_cohort(pass, ordinal).unwrap();
            let requests = probe_requests(&session);
            let summary = session
                .run_probe_cohort(requests, options, &mut budget)
                .await
                .unwrap();
            assert_eq!(
                (summary.completed_requests, summary.completed_output_tokens),
                (2, 6)
            );
            assert_eq!(
                (
                    summary.wave_attempts,
                    summary.reconciled_waves,
                    summary.released_prefix_rows
                ),
                (3, 3, 2)
            );
            session.end_prepared_owner_cohort().unwrap();
        }
    }
    let audit = session
        .prepared_owner_capture
        .as_ref()
        .unwrap()
        .prepared_audit();
    assert_eq!(
        (audit.population.offered, audit.preparation_attempts),
        (96, 32)
    );
    assert!(!audit.population.poisoned, "{audit:#?}");
    assert_eq!(executor.physical.load(Ordering::Acquire), 96);
    assert_eq!(executor.completion_calls.load(Ordering::Acquire), 64);
    assert!(session.frontiers().unwrap().is_empty());
    let epoch = session
        .activate_prepared_owner_source()
        .await
        .unwrap_or_else(|error| panic!("original source8 activation: {error}; {audit:#?}"));
    assert!(epoch > 0);
    assert!(runtime.snapshot().is_some());
    let receipt = runtime.profile_receipt().unwrap();
    assert_eq!(receipt.storage, ferrum_types::SloCostProfileStorage::Memory);
    assert!(receipt.path.is_none());
    assert_eq!(receipt.offered_samples, 96);
    assert!(receipt.recorded_samples >= 24);
    future::verify(session, executor).await;
}

mod context_family;
mod future;
mod numerical_family;
mod plan_e2e;
mod snapshot_universe;
mod source_preparation;

mod startup_series;

mod feedback_coverage;

mod journal;
mod legal_prefill;
mod prediction_validity;
mod prefix_acquisition;

/// A real private prefix reaches the same source8 writer/ledger after its
/// original outside selector, Core submission, sampler and actor settlement.
/// This is a CPU protocol gate, not CUDA graph or numerical qualification.
#[tokio::test]
async fn source8_private_outside_preparation_reaches_original_collector() {
    let (mut session, executor) = automatic_session().await;
    executor.enable_structured_query_route();
    executor
        .native_structured_submission
        .store(true, Ordering::Release);
    executor
        .project_structured_cpu_fill
        .store(true, Ordering::Release);
    executor
        .native_prefix_preparation_outside
        .store(true, Ordering::Release);
    let declared = declaration(&session, 3, false);
    session
        .begin_prepared_owner_source(declared, CostProfileLoadLimits::default())
        .await
        .unwrap();
    session.begin_prepared_owner_cohort(0, 0).unwrap();
    let requests = probe_requests(&session);
    let mut budget = ProbeExecutionBudget::new(
        tokio::time::Instant::now() + Duration::from_secs(15),
        NonZeroUsize::new(2).unwrap(),
        NonZeroUsize::new(3).unwrap(),
    );
    let result = session.run_probe_cohort(requests, ProbeCohortSettings {
        prefill_plan: crate::continuous_engine::inner::calibration::cohort_driver::ProbePrefillPlan::Joint,
        prefill_chunk: NonZeroU32::MIN,
        decode_route: CalibrationDecodeRoute::FullLogits,
        reset_token_policy: false,
    }, &mut budget).await;
    let selected_outside = executor
        .native_prefix_preparation_outside_submissions
        .load(Ordering::Acquire);
    let audit = session
        .prepared_owner_capture
        .as_ref()
        .unwrap()
        .prepared_audit();
    let poisoned = audit.population.poisoned;
    let preparation_attempts = audit.preparation_attempts;
    let end = result
        .as_ref()
        .ok()
        .map(|_| session.end_prepared_owner_cohort());
    let runtime = session.engine.inner.cost_runtime.as_ref().unwrap().clone();
    session.shutdown().await.unwrap();
    assert_eq!(
        selected_outside, 1,
        "the real private preparation must use the outside selector"
    );
    let summary =
        result.expect("outside private preparation must pass the original source8 consumer");
    assert_eq!(
        (summary.completed_requests, summary.completed_output_tokens),
        (2, 6)
    );
    assert_eq!(summary.released_prefix_rows, 2);
    assert_eq!(preparation_attempts, 1);
    assert!(!poisoned);
    end.unwrap().unwrap();
    assert!(
        runtime.snapshot().is_none(),
        "one cohort cannot claim qualification"
    );
}
