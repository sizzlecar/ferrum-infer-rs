//! Real compiled CPU token execution under a controlled cost surface.
//! This checks the Engine's persistent action consumer, not learned cost,
//! stopping quality, client SSE SLOs, or measured performance. The tiny program
//! uses real embedding/head providers and the terminal worker, without attention
//! or a KV pool; this is not a paged-attention numerical qualification.

use super::*;
use crate::continuous_engine::attempt_observation_tests::AttemptRecorder;
use crate::continuous_engine::output_delivery::OutputReceiver;
use ferrum_execution::execution_attempt::{AttemptWorkKind, ExecutionAttemptSummary};
use ferrum_interfaces::request_observation::*;
use ferrum_kernels::backend::cpu::vnext_ops::CpuVNextComposition;
use ferrum_models::test_support::native_program::{Fixture, FixtureOptions, VOCAB};
use ferrum_runtime::DeviceId;
use ferrum_scheduler::cost_prediction::*;

#[derive(Clone, Copy)]
enum Availability {
    Cold,
    Known,
    Stale,
}

struct ConditionalCost {
    revision: u64,
    availability: Availability,
    at: Instant,
    queried: Arc<Mutex<Vec<u64>>>,
}

impl CostSnapshot for ConditionalCost {
    fn revision(&self) -> u64 {
        self.revision
    }
    fn query_limits(&self) -> ferrum_types::CostQueryLimits {
        Default::default()
    }
    fn predict(&self, _: &CostQuery<'_>, _: Instant) -> CostPrediction {
        panic!("joint scenarios must not issue an extra scalar prediction")
    }
    fn cost_scenarios(
        &self,
        query: &CostQuery<'_>,
        _: Instant,
        count: usize,
    ) -> CostScenarioPrediction {
        self.queried.lock().push(self.revision);
        match self.availability {
            Availability::Cold => {
                return CostScenarioPrediction::Unknown(CostUnknownReason::NoCoverage)
            }
            Availability::Stale => {
                return CostScenarioPrediction::Unknown(CostUnknownReason::Stale)
            }
            Availability::Known => {}
        }
        assert!(query.work_is_exact);
        let prefill: usize = query
            .work
            .iter()
            .filter(|work| work.kind == AttemptWorkKind::Prefill)
            .map(|work| work.requested.as_range().len())
            .sum();
        let service_time = if prefill > 0 {
            Duration::from_secs(10 * prefill as u64)
        } else {
            Duration::from_millis(100)
        };
        CostScenarioPrediction::Known {
            central_service_time: service_time,
            scenarios: (0..count)
                .map(|rank| CostScenario {
                    rank,
                    weight: 1.0 / count as f64,
                    service_time,
                })
                .collect(),
            dependence: CostScenarioDependence::SharedResidualRank,
            support: CostSupport::ValidatedFeatureDomain,
            samples: 32,
            observed_at: self.at,
            expires_at: self.at + Duration::from_secs(3600),
        }
    }
    fn stopping_forecast(&self, query: &StoppingQuery<'_>) -> StoppingForecast {
        // These test requests explicitly disable natural EOS; a declared cap
        // is an endpoint, not substituted evidence for unknown natural stops.
        assert_eq!(query.rules.mechanism(), StoppingMechanism::CapOnly);
        StoppingForecast {
            stops: vec![StoppingMass {
                generated_tokens: query.cap,
                mass: 1.0,
            }],
            known_survival_mass: 0.0,
            unresolved_mass: 0.0,
            tpot_undefined_mass: 0.0,
        }
    }
    fn completion_forecast(&self, _: &CompletionQuery<'_>) -> CompletionForecast {
        CompletionForecast {
            succeeded_mass: 1.0,
            failed_mass: 0.0,
            unresolved_mass: 0.0,
            samples: 32,
            approximation: Some(CompletionApproximation::EmpiricalMaturedCohort),
        }
    }
}

struct Source {
    current: Mutex<Arc<dyn CostSnapshot>>,
    publication: tokio::sync::watch::Sender<u64>,
    queried: Arc<Mutex<Vec<u64>>>,
}

impl Source {
    fn new(availability: Availability) -> Arc<Self> {
        let queried = Arc::new(Mutex::new(Vec::new()));
        Arc::new(Self {
            current: Mutex::new(Arc::new(ConditionalCost {
                revision: 1,
                availability,
                at: Instant::now(),
                queried: queried.clone(),
            })),
            publication: tokio::sync::watch::channel(1).0,
            queried,
        })
    }
    fn publish(&self, availability: Availability) {
        let mut current = self.current.lock();
        let revision = current.revision() + 1;
        *current = Arc::new(ConditionalCost {
            revision,
            availability,
            at: Instant::now(),
            queried: self.queried.clone(),
        });
        self.publication.send_replace(revision);
    }
}

impl CostSnapshotSource for Source {
    fn load(&self) -> Arc<dyn CostSnapshot> {
        self.current.lock().clone()
    }
    fn subscribe(&self) -> tokio::sync::watch::Receiver<u64> {
        self.publication.subscribe()
    }
}

#[derive(Default)]
struct ObserveRequests {
    drops: Mutex<Vec<RequestObservationDrop>>,
}
impl RequestStoppingObserver for ObserveRequests {
    fn limits(&self) -> RequestObservationLimits {
        RequestObservationLimits {
            max_rule_bytes: 4096,
        }
    }
    fn try_record(&self, observation: &RequestStoppingObservation) -> RequestObservationDelivery {
        assert!(observation.is_consistent());
        RequestObservationDelivery::Recorded
    }
    fn record_drop(&self, reason: RequestObservationDrop) {
        self.drops.lock().push(reason);
    }
}

struct Streamed {
    id: RequestId,
    receiver: OutputReceiver,
    tokens: Vec<TokenId>,
    text: String,
    terminal: Option<StreamChunk>,
    prompt_tokens: usize,
}

impl Streamed {
    async fn receive_through(&mut self, generated: usize, terminal: bool) {
        tokio::time::timeout(Duration::from_secs(10), async {
            while self.tokens.len() < generated || (terminal && self.terminal.is_none()) {
                let output = self.receiver.recv().await.unwrap().unwrap();
                let chunk = output.chunk();
                assert_eq!(chunk.request_id, self.id);
                if let Some(token) = chunk.token {
                    self.tokens.push(token);
                }
                self.text.push_str(&chunk.text);
                if chunk.finish_reason.is_some() {
                    self.terminal = Some(chunk.clone());
                }
            }
        })
        .await
        .expect("real output delivery must reach its committed frontier");
    }
}

async fn submit(engine: &ContinuousBatchEngine, prompt_tokens: usize) -> Streamed {
    let inner = engine.active_inner().unwrap();
    let prompt = std::iter::repeat_n("t1", prompt_tokens)
        .collect::<Vec<_>>()
        .join(" ");
    let mut request = InferenceRequest::new(prompt, "sampling-readback-fixture");
    request.stream = true;
    request.sampling_params.max_tokens = 64;
    request.sampling_params.temperature = 0.0;
    request
        .metadata
        .insert("ferrum_ignore_eos".into(), true.into());
    request.metadata.insert(
        PROMPT_TOKENS_METADATA_KEY.into(),
        serde_json::json!(prompt_tokens),
    );
    request.ensure_ingress(ferrum_types::RequestIngressBoundary::EngineApi);
    let tokens = inner.tokenizer.encode(&request.prompt, true).unwrap();
    assert_eq!(tokens.len(), prompt_tokens);
    let id = request.id.clone();
    let (sender, receiver) = inner.output_channel(id.clone(), 8).unwrap();
    let mut sequence = SequenceState::new_with_tokenizer_and_model_vocab_size(
        request.clone(),
        tokens,
        Some(inner.tokenizer.clone()),
        Some(inner.model_executor.info().vocab_size),
    );
    sequence.stream_sender = Some(sender);
    inner.attach_request_stopping(&mut sequence);
    // As in cost_feedback_tests, manual iteration owns scheduling. Starting a
    // managed driver here would race the deliberately controlled publication.
    inner.sequences.insert(id.clone(), sequence);
    engine.scheduler.submit(request).await.unwrap();
    Streamed {
        id,
        receiver,
        tokens: Vec::new(),
        text: String::new(),
        terminal: None,
        prompt_tokens,
    }
}

async fn step(engine: &ContinuousBatchEngine, streams: &mut [&mut Streamed]) {
    let inner = engine.active_inner().unwrap();
    inner.run_iteration().await.unwrap();
    for stream in streams {
        let generated = inner
            .sequences
            .get(&stream.id)
            .map(|handle| handle.lock().as_ref().unwrap().generated_tokens.len());
        stream
            .receive_through(generated.unwrap_or(64), generated.is_none())
            .await;
    }
}

fn append_prefills(
    summaries: Vec<ExecutionAttemptSummary>,
    request: &RequestId,
    ranges: &mut Vec<(usize, usize)>,
) {
    for summary in summaries {
        for participant in summary.participants {
            if participant.request_id == *request && participant.kind == AttemptWorkKind::Prefill {
                ranges.push((
                    participant.requested.start(),
                    participant.requested.as_range().len(),
                ));
            }
        }
    }
}

struct EngineFixture {
    engine: ContinuousBatchEngine,
    tokenizer: Arc<dyn Tokenizer + Send + Sync>,
    source: Arc<Source>,
    attempts: Arc<AttemptRecorder>,
    requests: Arc<ObserveRequests>,
}

fn engine_fixture(availability: Availability, trace_path: PathBuf) -> EngineFixture {
    engine_fixture_with_policy(
        availability,
        trace_path,
        ferrum_types::SloStrategy::Joint,
        ProfileEntrypoint::Serve,
    )
}

fn engine_fixture_with_policy(
    availability: Availability,
    trace_path: PathBuf,
    strategy: ferrum_types::SloStrategy,
    entrypoint: ProfileEntrypoint,
) -> EngineFixture {
    let mut config = EngineConfig::default();
    config.runtime.scheduler_trace_jsonl = Some(trace_path);
    config.runtime.profile_entrypoint = Some(entrypoint);
    config.backend.device = Device::CPU;
    config.scheduler.max_running_requests = 2;
    config.scheduler.prefill_step_chunk = Some(4);
    config.scheduler.active_decode_prefill_chunk = Some(4);
    config.batching.max_batch_size = 2;
    config.batching.max_num_batched_tokens = 8;
    config.batching.prefill_decode_execution = ferrum_types::PrefillDecodeExecution::Split;
    let mut policy: ferrum_types::SchedulerSloConfig =
        "ttft:1000000,tpot:50000,itl:15000".parse().unwrap();
    policy.strategy = strategy;
    policy.planning.max_scenarios = 2;
    // A correctness-test CPU resource allowance; its existing modeled control
    // delay remains included in the selected conditional trajectory.
    policy.planning.max_elapsed_us = 1_000_000;
    // Model seconds keep this correctness fixture's risk margins independent
    // of ordinary CPU scheduling and output-delivery latency. Device execution
    // still uses the real model; these are not measured latency targets.
    policy.joint.request_deadline_ms = 2_000_000;
    config.scheduler.slo = Some(policy);
    config.runtime.max_model_len = Some(128);
    config.runtime.sampling_readback_mode = ferrum_types::SamplingReadbackMode::Uniform;
    let (runtime, registry, materializers, _, catalog) = CpuVNextComposition::create(
        DeviceId::new("device.test.engine-prefill-phase").unwrap(),
        256 << 20,
    )
    .unwrap()
    .into_parts()
    .unwrap();
    let fixture = Fixture::try_new_with_engine_config(
        ferrum_types::SamplingReadbackMode::Uniform,
        Device::CPU,
        runtime,
        registry,
        materializers,
        catalog,
        FixtureOptions {
            maximum_tokens: 128,
            ..Default::default()
        },
        |native| *native = config.clone(),
    )
    .unwrap();
    let texts = (0..VOCAB).map(|id| format!("t{id}")).collect::<Vec<_>>();
    let pairs = texts
        .iter()
        .enumerate()
        .map(|(id, text)| (text.as_str(), id as u32))
        .collect::<Vec<_>>();
    let mut tokenizer = PolicyTokenizer::new(VOCAB, &pairs);
    // Every sampled vocabulary token has exactly one visible text event. This
    // isolates phase execution from the separately unsupported coalescing case.
    tokenizer.special = ferrum_types::SpecialTokens::default();
    let tokenizer: Arc<dyn Tokenizer + Send + Sync> = Arc::new(tokenizer);
    let source = Source::new(availability);
    let attempts = Arc::new(AttemptRecorder::default());
    let requests = Arc::new(ObserveRequests::default());
    let engine = ContinuousBatchEngine::new_observed_with_requests(
        config.clone(),
        Arc::new(ContinuousBatchScheduler::new(config.scheduler)),
        tokenizer.clone(),
        Arc::new(crate::registry::GreedySampler),
        ExecutorOwner::new(Box::new(fixture.executor)),
        None,
        Some(attempts.clone()),
        Some(source.clone()),
        Some(requests.clone()),
    )
    .unwrap();
    EngineFixture {
        engine,
        tokenizer,
        source,
        attempts,
        requests,
    }
}

#[tokio::test(flavor = "current_thread")]
async fn bounded_decision_trace_uses_actual_queries_and_clears_after_stale_snapshot() {
    for entrypoint in [ProfileEntrypoint::Run, ProfileEntrypoint::Serve] {
        let trace = tempfile::tempdir().unwrap();
        let trace_path = trace.path().join("iterations.jsonl");
        let EngineFixture {
            engine,
            source,
            attempts,
            ..
        } = engine_fixture_with_policy(
            Availability::Cold,
            trace_path.clone(),
            ferrum_types::SloStrategy::Bounded,
            entrypoint,
        );
        // Use the real admission/execution owner before switching the cost
        // snapshot. A Known cost surface alone is not a resource proof.
        let mut decoder = submit(&engine, 1).await;
        step(&engine, &mut [&mut decoder]).await;
        let mut prefill = submit(&engine, 36).await;
        step(&engine, &mut [&mut decoder, &mut prefill]).await;
        attempts.take();

        source.publish(Availability::Known);
        let queries_before = source.queried.lock().len();
        step(&engine, &mut [&mut decoder, &mut prefill]).await;
        let known_queries = source.queried.lock().len() - queries_before;
        let known_work = attempts.take();
        assert!(known_queries > 0);
        assert!(!known_work.is_empty());
        let mut known_prefill_ranges = Vec::new();
        append_prefills(known_work, &prefill.id, &mut known_prefill_ranges);

        source.publish(Availability::Stale);
        step(&engine, &mut [&mut decoder, &mut prefill]).await;
        assert!(source.queried.lock().contains(&3));
        for _ in 0..96 {
            if decoder.terminal.is_some() && prefill.terminal.is_some() {
                break;
            }
            step(&engine, &mut [&mut decoder, &mut prefill]).await;
        }
        for stream in [&decoder, &prefill] {
            let terminal = stream.terminal.as_ref().expect("real output completes");
            assert_eq!(terminal.finish_reason, Some(FinishReason::Length));
            assert_eq!(terminal.usage.as_ref().unwrap().completion_tokens, 64);
            assert_eq!(stream.tokens.len(), 64);
        }
        engine.shutdown().await.unwrap();
        engine.wait_for_shutdown_completion().await.unwrap();

        let events = read_engine_profile_events(&trace_path);
        let iterations: Vec<_> = events
            .iter()
            .filter(|event| event.phase == "scheduler.iteration")
            .collect();
        let known = iterations
            .iter()
            .find(|event| event.attributes["slo_planning"]["snapshot_revision"] == 2)
            .expect("the actual Known iteration is publicly traced");
        assert_eq!(known.entrypoint, entrypoint);
        assert_eq!(known.attributes["result"], "some_ok");
        let planning = &known.attributes["slo_planning"];
        // Both counters cover the actual snapshot calls. Merely projecting
        // an already computed decision must not query the provider again.
        assert_eq!(planning["query_count"], known_queries as u64);
        let stats = &planning["statistics"]["bounded"];
        assert_eq!(stats["provider_calls"], known_queries as u64);
        assert_eq!(stats["evaluated"], 1);
        let decision = &planning["bounded_decision"];
        assert!(decision.is_object(), "{planning}");
        assert_eq!(decision["snapshot_revision"], 2);
        assert_eq!(decision["request_count"], 2);
        assert_eq!(decision["observed_ranks"], 2);
        assert_eq!(decision["static_score"], stats["last_static_score"]);
        assert_eq!(decision["selected_score"], stats["last_selected_score"]);
        let actions = decision["candidate_actions"].as_array().unwrap();
        let alternatives = decision["alternatives"].as_array().unwrap();
        assert!(!actions.is_empty());
        assert_eq!(actions.len(), alternatives.len());
        assert_eq!(decision["static_action"]["kind"], "static");
        let baseline = &decision["static_evidence"];
        assert!(baseline["reference_time_seconds"].as_f64().unwrap() > 0.0);
        for candidate in alternatives {
            if candidate["supported_mass"].as_f64().unwrap() == 1.0 {
                for metric in ["ttft", "tpot", "itl"] {
                    assert_eq!(
                        candidate["local_targets"][metric]["samples"],
                        baseline["local_targets"][metric]["samples"],
                        "candidate-only output must not expand the reference population"
                    );
                }
            }
        }
        if let Some(selected) = decision["selected_index"].as_u64() {
            assert!((selected as usize) < alternatives.len());
            let candidate = &alternatives[selected as usize];
            match decision["selection_reason"].as_str().unwrap() {
                "within_targets_faster" | "within_targets_recovery" => {
                    for metric in ["ttft", "tpot", "itl"] {
                        assert_eq!(
                            candidate["local_targets"][metric]["classification"],
                            "within"
                        );
                    }
                    if decision["selection_reason"] == "within_targets_faster" {
                        assert!(
                            candidate["reference_time_seconds"].as_f64().unwrap()
                                < baseline["reference_time_seconds"].as_f64().unwrap()
                        );
                    }
                }
                "pareto_target_repair" => {}
                other => panic!("selected action has invalid reason: {other}"),
            }
            assert_eq!(decision["outcome"]["status"], "applied");
            assert_eq!(stats["selected"], 1);
            assert_eq!(stats["applied"], 1);
            match actions[selected as usize]["kind"].as_str().unwrap() {
                "prefill_limit" => {
                    let limit = actions[selected as usize]["max_tokens"].as_u64().unwrap();
                    assert_eq!(known_prefill_ranges, vec![(4, limit as usize)]);
                }
                "decode_only" => assert!(known_prefill_ranges.is_empty()),
                other => panic!("unexpected bounded action: {other}"),
            }
        } else {
            assert_eq!(decision["selection_reason"], "static_retained");
            assert_eq!(decision["outcome"]["status"], "retained");
            assert_eq!(stats["selected"], 0);
            assert_eq!(stats["applied"], 0);
            assert_eq!(known_prefill_ranges, vec![(4, 4)]);
        }
        let mut saw_stale_round = false;
        for event in iterations {
            assert_eq!(event.entrypoint, entrypoint);
            let planning = &event.attributes["slo_planning"];
            saw_stale_round |= planning["snapshot_revision"] == 3;
            if planning["snapshot_revision"] != 2 {
                if planning.is_object() {
                    assert!(planning.get("bounded_decision").is_some());
                }
                assert!(
                    planning["bounded_decision"].is_null(),
                    "cold, stale and no-alternative iterations must not inherit the Known evidence"
                );
            }
        }
        assert!(saw_stale_round);
    }
}

#[tokio::test(flavor = "current_thread")]
async fn joint_prefill_phase_engine_keeps_actual_ranges_after_snapshot_becomes_stale() {
    let trace = tempfile::tempdir().unwrap();
    let trace_path = trace.path().join("iterations.jsonl");
    let EngineFixture {
        engine,
        tokenizer,
        source,
        attempts,
        requests,
    } = engine_fixture(Availability::Cold, trace_path.clone());
    let mut decoder = submit(&engine, 1).await;
    step(&engine, &mut [&mut decoder]).await;
    assert_eq!(decoder.tokens.len(), 1);
    attempts.take();
    let mut prefill = submit(&engine, 36).await;
    // The current feature installs phases only after actual admission. Keep
    // cold-start fallback and its first static four-token execution explicit;
    // this is not evidence that a Known admission can select a phase rescue.
    step(&engine, &mut [&mut decoder, &mut prefill]).await;
    let mut ranges = Vec::new();
    append_prefills(attempts.take(), &prefill.id, &mut ranges);
    assert_eq!(ranges, vec![(0, 4)]);
    let cold_stats = serde_json::to_value(engine.scheduler.slo_snapshot().unwrap()).unwrap();
    assert_eq!(cold_stats["joint_prefill_phase_interventions"], 0);
    source.publish(Availability::Known);
    step(&engine, &mut [&mut decoder, &mut prefill]).await;
    let selected = serde_json::to_value(engine.scheduler.slo_snapshot().unwrap()).unwrap();
    assert_eq!(
        selected["joint_prefill_phase_interventions"], 1,
        "{selected:?}"
    );
    append_prefills(attempts.take(), &prefill.id, &mut ranges);
    assert_eq!(ranges, vec![(0, 4), (4, 1)]);
    let old = source.load();
    source.publish(Availability::Stale);
    assert_eq!(old.revision(), 2);
    assert_eq!(source.load().revision(), 3);
    for _ in 0..96 {
        if decoder.terminal.is_some() && prefill.terminal.is_some() {
            break;
        }
        step(&engine, &mut [&mut decoder, &mut prefill]).await;
        append_prefills(attempts.take(), &prefill.id, &mut ranges);
    }
    assert_eq!(
        ranges,
        std::iter::once((0, 4))
            .chain((4..36).map(|start| (start, 1)))
            .collect::<Vec<_>>()
    );
    assert!(source.queried.lock().contains(&3));
    let final_stats = serde_json::to_value(engine.scheduler.slo_snapshot().unwrap()).unwrap();
    assert_eq!(final_stats["joint_prefill_phase_interventions"], 1);
    assert_eq!(
        final_stats["joint_interventions"],
        selected["joint_interventions"]
    );
    for stream in [&decoder, &prefill] {
        let terminal = stream
            .terminal
            .as_ref()
            .expect("both actual requests complete");
        assert_eq!(terminal.finish_reason, Some(FinishReason::Length));
        let usage = terminal.usage.as_ref().unwrap();
        assert_eq!(usage.prompt_tokens, stream.prompt_tokens);
        assert_eq!(usage.completion_tokens, 64);
        assert_eq!(usage.total_tokens, stream.prompt_tokens + 64);
        assert_eq!(stream.tokens.len(), 64);
        assert_eq!(stream.text, tokenizer.decode(&stream.tokens, true).unwrap());
    }
    assert_eq!(attempts.dropped.load(Ordering::Relaxed), 0);
    {
        let drops = requests.drops.lock();
        assert!(drops.is_empty(), "{drops:?}");
    }
    assert!(engine.active_inner().unwrap().sequences.is_empty());
    engine.shutdown().await.unwrap();
    engine.wait_for_shutdown_completion().await.unwrap();
    let events = read_engine_profile_events(&trace_path);
    let selected_iteration = events
        .iter()
        .find(|event| {
            event.phase == "scheduler.iteration"
                && event.attributes["slo_planning"]["statistics"]
                    ["joint_prefill_phase_interventions"]
                    == 1
        })
        .expect("the applied phase and actual selected range share one iteration");
    assert_eq!(selected_iteration.attributes["result"], "some_ok");
    let planning = &selected_iteration.attributes["slo_planning"];
    assert_eq!(planning["snapshot_revision"], 2);
    assert!(planning["query_count"].as_u64().unwrap() > 0);
    assert!(planning["prediction_elapsed_us"].is_u64());
    let row = selected_iteration.attributes["plan"]["requests"]
        .as_array()
        .unwrap()
        .iter()
        .find(|row| row["request_id"] == prefill.id.to_string())
        .unwrap();
    assert_eq!(row["phase"], "Prefilling");
    assert_eq!(
        row["scheduled_range"],
        serde_json::json!({"start": 4, "end": 5})
    );
    assert_eq!(row["execution_frontier_before"], 4);
    assert_eq!(
        selected_iteration.attributes["engine_counters"]["prefill_tokens_delta"],
        1
    );
    let applied_total: u64 = events
        .iter()
        .filter(|event| event.phase == "scheduler.iteration")
        .filter_map(|event| {
            event.attributes["slo_planning"]["statistics"]["joint_interventions"].as_u64()
        })
        .sum();
    assert_eq!(
        applied_total,
        final_stats["joint_interventions"].as_u64().unwrap()
    );
}
