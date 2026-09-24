//! Shared product construction with actual bounded profile/reference readers.
//! The guarded executor is a controlled CPU backend, not Metal performance evidence.
use super::*;
use crate::continuous_engine::inner::slo_controller::tests::fixture::{
    startup_components, ControlledExecutor,
};
use ferrum_interfaces::{
    engine::{InferenceEngine, LlmInferenceEngine},
    output_flow::{OutputCompletion, OutputProjectionContract},
    InferenceRequestContext,
};
use ferrum_scheduler::implementations::continuous::{cost_model, cost_profile};
use ferrum_testkit::MockTensorFactory;
use ferrum_types::{ApiChatRequest, ApiRequest, ApiStreamOptions, ServiceSloConfig};
use futures::StreamExt;
use std::{fs, num::NonZeroU64, path::PathBuf, time::SystemTime};

mod waiting_capacity;

struct Fixture {
    dir: PathBuf,
    config: EngineConfig,
    profile: cost_profile::CostProfileFile,
    tokenizer: Arc<dyn Tokenizer + Send + Sync>,
    executor: Arc<ControlledExecutor>,
}

impl Fixture {
    async fn new() -> Self {
        let (tokenizer, executor) = startup_components(1).await;
        let dir = std::env::temp_dir().join(format!("ferrum-slo-startup-{}", uuid::Uuid::new_v4()));
        fs::create_dir(&dir).unwrap();
        let reference = inner::prefill_reference_runtime::test_calibration_artifact();
        let reference_path = dir.join("reference.json");
        fs::write(&reference_path, serde_json::to_vec(&reference).unwrap()).unwrap();
        let settings = cost_model::CostModelSettings::default();
        let at = u64::try_from(
            SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap()
                .as_nanos(),
        )
        .unwrap();
        let profile = cost_profile::CostProfileFile {
            schema_version: cost_profile::COST_PROFILE_SCHEMA_VERSION,
            fingerprint: reference.fingerprint.clone(),
            settings: (&settings).into(),
            generated_unix_ns: at,
            source_clock_max_error_ns: Some(0),
            source: cost_profile::ProfileSource {
                generator: "typed guarded CPU fixture".into(),
                generator_revision: "slo-startup-contract".into(),
                measurement_protocol: "isolated preparation through host commit".into(),
                observation_artifact_sha256: [19; 32],
            },
            samples: (0..settings.min_samples.get())
                .map(|index| cost_profile::ProfileSample {
                    source_record: index as u64,
                    measured_unix_ns: at - 1_000_000_000 + index as u64,
                    shape: reference.protocol.decode_shape.exact.clone(),
                    boundary: cost_profile::ProfileCostBoundary::PreparationToCommit,
                    outcome: cost_profile::ProfileObservationOutcome::Completed {},
                    timing: cost_profile::ProfileWaveTiming {
                        wall_total_ns: 100 + index as u64,
                        device_elapsed_ns: None,
                        stages: Default::default(),
                    },
                })
                .collect(),
        };
        let mut config = EngineConfig::default();
        let slo = &mut config.scheduler.slo;
        slo.mode = SloMode::Enforce;
        slo.output.transport = SloOutputTransport::Credited;
        slo.admission.time_policy = SloTimeAdmissionPolicy::CompleteRequests;
        slo.default_service_class = Some("startup-test".into());
        slo.services.push(ServiceSloConfig {
            id: "startup-test".into(),
            server_token_commit: ferrum_types::SloLatencyBudgets {
                ttft_ms: NonZeroU64::new(10_000).unwrap(),
                tpot_ms: NonZeroU64::new(10_000).unwrap(),
                itl_ms: NonZeroU64::new(10_000).unwrap(),
            },
            client_visible: None,
            attainment: Default::default(),
        });
        // Functional correctness timeout, not a measured scheduling threshold.
        slo.planner.max_planning_us = NonZeroU64::new(30_000_000).unwrap();
        slo.cost_profile = Some(dir.join("profile.json"));
        slo.cost_observation
            .profile_import
            .declared_local_clock_max_error_ns = Some(1_000_000_000);
        slo.prefill_reference = Some(ferrum_types::SloPrefillReferenceConfig {
            artifact_path: reference_path,
            expected_protocol_sha256: reference.protocol.sha256().unwrap(),
            limits: Default::default(),
        });
        config.batching.max_batch_size = 1;
        let fixture = Self {
            dir,
            config,
            profile,
            tokenizer,
            executor,
        };
        fixture.save_profile();
        fixture
    }

    fn save_profile(&self) {
        fs::write(
            self.config.scheduler.slo.cost_profile.as_ref().unwrap(),
            serde_json::to_vec(&self.profile).unwrap(),
        )
        .unwrap();
    }

    fn build(&self, config: EngineConfig) -> Result<ContinuousBatchEngine> {
        ContinuousBatchEngine::new_plan_runtime(
            config.clone(),
            Arc::new(ContinuousBatchScheduler::new(config.scheduler)),
            self.tokenizer.clone(),
            Arc::new(crate::registry::GreedySampler),
            self.executor.clone(),
            Arc::new(MockTensorFactory),
        )
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        self.executor.abort_resource_sessions();
        let _ = fs::remove_dir_all(&self.dir);
    }
}

fn request(config: &EngineConfig) -> InferenceRequest {
    let mut request = InferenceRequest::new("test test test test", config.model.model_id.clone());
    request.stream = true;
    request.sampling_params.max_tokens = 2;
    request.sampling_params.temperature = 0.0;
    request.sampling_params.repetition_penalty = 1.0;
    request
        .metadata
        .insert("ferrum_ignore_eos".into(), true.into());
    request
}

async fn bounded<T>(future: impl std::future::Future<Output = T>) -> T {
    tokio::time::timeout(Duration::from_secs(10), future)
        .await
        .expect("shared startup fixture made no progress")
}

#[tokio::test]
async fn enforce_constructor_imports_real_files_and_does_not_reuse_a_prior_receipt() {
    let f = Fixture::new().await;
    let engine = f.build(f.config.clone()).unwrap();
    let receipt = engine
        .inner
        .config
        .slo_cost_profile_receipt
        .as_ref()
        .unwrap();
    assert_eq!(receipt.recorded_samples, f.profile.samples.len());
    assert!(receipt.bucket_count > 0);
    assert_eq!(
        receipt.path,
        f.config
            .scheduler
            .slo
            .cost_profile
            .clone()
            .unwrap()
            .canonicalize()
            .unwrap()
    );
    assert!(engine
        .inner
        .cost_runtime
        .as_ref()
        .unwrap()
        .snapshot()
        .is_some());
    assert!(engine.inner.prefill_reference_runtime.is_some());
    let prior = engine.inner.config.clone();
    bounded(engine.shutdown()).await.unwrap();
    drop(engine);

    fs::write(
        prior.scheduler.slo.cost_profile.as_ref().unwrap(),
        b"not a profile",
    )
    .unwrap();
    assert!(matches!(f.build(prior), Err(FerrumError::Config { .. })));
    assert_eq!(f.executor.entries.load(Ordering::Acquire), 0);
}

#[tokio::test]
async fn enforce_constructor_rejects_missing_artifacts_wrong_identity_and_expired_samples() {
    let mut f = Fixture::new().await;
    for missing in [true, false] {
        let mut config = f.config.clone();
        if missing {
            config.scheduler.slo.cost_profile = None;
        } else {
            config.scheduler.slo.prefill_reference = None;
        }
        assert!(matches!(f.build(config), Err(FerrumError::Config { .. })));
    }
    let mut config = f.config.clone();
    config.scheduler.slo.cost_profile = Some(f.dir.join("absent.json"));
    assert!(matches!(f.build(config), Err(FerrumError::Config { .. })));
    let mut config = f.config.clone();
    config
        .scheduler
        .slo
        .prefill_reference
        .as_mut()
        .unwrap()
        .expected_protocol_sha256 = [88; 32];
    assert!(matches!(f.build(config), Err(FerrumError::Config { .. })));

    let mut reference = inner::prefill_reference_runtime::test_calibration_artifact();
    reference.fingerprint.model_weights = [77; 32];
    let reference_path = &f
        .config
        .scheduler
        .slo
        .prefill_reference
        .as_ref()
        .unwrap()
        .artifact_path;
    fs::write(reference_path, serde_json::to_vec(&reference).unwrap()).unwrap();
    assert!(matches!(
        f.build(f.config.clone()),
        Err(FerrumError::Config { .. })
    ));
    fs::write(
        reference_path,
        serde_json::to_vec(&inner::prefill_reference_runtime::test_calibration_artifact()).unwrap(),
    )
    .unwrap();

    let original = f.profile.fingerprint.clone();
    f.profile.fingerprint.device_runtime = [99; 32];
    f.save_profile();
    assert!(matches!(
        f.build(f.config.clone()),
        Err(FerrumError::Config { .. })
    ));
    f.profile.fingerprint = original;
    for sample in &mut f.profile.samples {
        sample.measured_unix_ns -= f.profile.settings.max_sample_age_ns.get() + 10_000_000_000;
    }
    f.save_profile();
    assert!(matches!(
        f.build(f.config.clone()),
        Err(FerrumError::Config { .. })
    ));
    assert_eq!(f.executor.entries.load(Ordering::Acquire), 0);
}

#[tokio::test]
async fn enforce_rejects_unconnected_transport_and_strict_admission_before_execution() {
    let f = Fixture::new().await;
    let mut legacy = f.config.clone();
    legacy.scheduler.slo.output.transport = SloOutputTransport::Legacy;
    assert!(matches!(
        f.build(legacy),
        Err(FerrumError::Unsupported { .. })
    ));
    let mut strict = f.config.clone();
    strict.scheduler.slo.admission.time_policy = SloTimeAdmissionPolicy::RequireSlo;
    assert!(matches!(
        f.build(strict),
        Err(FerrumError::Unsupported { .. })
    ));
    assert_eq!(f.executor.entries.load(Ordering::Acquire), 0);
}

#[tokio::test]
async fn enforce_legacy_entries_reject_before_owner_slot_credit_or_scheduler_acceptance() {
    let f = Fixture::new().await;
    let engine = f.build(f.config.clone()).unwrap();
    let request = request(&f.config);
    let id = request.id.clone();
    assert!(matches!(
        engine.infer(request.clone()).await,
        Err(FerrumError::Unsupported { .. })
    ));
    assert!(matches!(
        engine.infer_stream(request).await,
        Err(FerrumError::Unsupported { .. })
    ));
    assert!(engine.inner.sequences.read().is_empty());
    assert_eq!(engine.inner.scheduler.active_count(), 0);
    assert_eq!(engine.inner.scheduler.waiting_count(), 0);
    assert_eq!(engine.inner.scheduler.trace_phase(&id), None);
    assert!(engine.inner.output_credit_pool.get().is_none());
    assert!(engine
        .inner
        .resource_lifecycle
        .lock()
        .owner_close_summary("request", &id.to_string())
        .is_empty());
    assert_eq!(f.executor.entries.load(Ordering::Acquire), 0);
    assert!(!engine.inner.bg_loop_spawned.load(Ordering::Acquire));
    bounded(engine.shutdown()).await.unwrap();
}

#[tokio::test]
async fn enforce_shared_credited_cli_and_chat_entries_finish_real_guarded_requests() {
    for chat in [false, true] {
        let f = Fixture::new().await;
        let engine = f.build(f.config.clone()).unwrap();
        let mut request = request(&f.config);
        let id = request.id.clone();
        let contract = if chat {
            // This CPU fixture uses a plain rendered prompt. Real HTTP ingress
            // supplies the same field from the model template conversion.
            request.metadata.insert(
                ferrum_types::PROMPT_OPENED_REASONING_METADATA_KEY.into(),
                false.into(),
            );
            request.api_request = Some(ApiRequest::Chat(ApiChatRequest {
                messages: Vec::new(),
                tools: Vec::new(),
                tool_choice: None,
                tool_call_protocol: Default::default(),
                legacy_functions: Vec::new(),
                legacy_function_call: None,
                response_format: None,
                stream_options: Some(ApiStreamOptions {
                    include_usage: Some(true),
                }),
            }));
            OutputProjectionContract::chat_sse(id.to_string(), "wire-alias".into(), true)
        } else {
            OutputProjectionContract::cli_text()
        };
        let mut session = engine
            .infer_credited_stream(
                request,
                InferenceRequestContext::capture(),
                Arc::new(contract),
            )
            .await
            .unwrap();
        let mut visible = false;
        let mut terminal = false;
        bounded(async {
            while let Some(frame) = session.frames.next().await {
                visible |= !frame.metadata().terminal && !frame.wire().payload().is_empty();
                terminal |= frame.metadata().terminal;
            }
        })
        .await;
        assert!(visible && terminal);
        let completion = bounded(session.completion).await.unwrap();
        match completion.payload() {
            OutputCompletion::Succeeded {
                history: Some(history),
                reason,
                usage,
                ..
            } => {
                assert_eq!(history.tokens.len(), 2);
                assert_eq!(usage.completion_tokens, 2);
                assert_eq!(*reason, ferrum_types::FinishReason::Length);
            }
            _ => panic!("guarded credited request did not return successful retained history"),
        }
        assert!(f.executor.physical.load(Ordering::Acquire) >= 2);
        drop((completion, session.frames));
        bounded(engine.shutdown()).await.unwrap();
        assert!(engine.inner.sequences.read().is_empty());
        assert_eq!(engine.inner.scheduler.trace_phase(&id), None);
        let pool = engine
            .inner
            .output_credit_pool
            .get()
            .unwrap()
            .as_ref()
            .unwrap();
        let mut changed = pool.subscribe();
        bounded(async {
            loop {
                let state = pool.snapshot();
                if state.data_used == Default::default()
                    && state.terminal_held == Default::default()
                {
                    break;
                }
                changed.changed().await.unwrap();
            }
        })
        .await;
    }
}
