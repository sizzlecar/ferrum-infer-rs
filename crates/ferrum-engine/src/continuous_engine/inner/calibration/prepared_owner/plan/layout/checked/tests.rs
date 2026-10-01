//! The default production input preparation and checked freeze over real CPU
//! roots. This establishes neither empirical qualification nor GPU performance.
mod algorithm_seed;
mod candidate_spans;
mod cursor;
mod readiness_recapture;
mod source_work;
use super::*;
use crate::continuous_engine::inner::calibration::geometry_projection::tests::{
    fixture, fixture_with_domain,
};
use crate::{AutomaticCostProbeOutput, AutomaticCostProbePromptRenderer};
use ferrum_tokenizer::implementations::HuggingFaceTokenizer;
use ferrum_types::{
    ApiChatMessage, ApiChatRequest, ApiCompletionRequest, ApiMessageRole, ApiRequest,
    ApiStreamOptions, InferenceRequest, ModelId,
};
use std::{
    sync::{atomic::Ordering, Arc},
    time::Duration,
};

#[derive(Debug)]
struct PlainRenderer {
    model: ModelId,
}
impl AutomaticCostProbePromptRenderer for PlainRenderer {
    fn render_user_text(&self, text: &str) -> Result<AutomaticCostProbeTemplate> {
        template(text, &self.model)
    }
    fn retained_payload_bytes(&self) -> Option<usize> {
        std::mem::size_of::<Self>().checked_add(self.model.0.capacity())
    }
}

fn template(text: &str, model: &ModelId) -> Result<AutomaticCostProbeTemplate> {
    let mut request = InferenceRequest::new(text, model.clone());
    request.stream = true;
    request.sampling_params.temperature = 1.0;
    request.sampling_params.top_k = Some(64);
    request.sampling_params.repetition_penalty = 1.0;
    AutomaticCostProbeTemplate::new(request, AutomaticCostProbeOutput::CliText)
}

async fn tokenizer() -> HuggingFaceTokenizer {
    tokenizer_with_generation_config(None).await
}

async fn tokenizer_with_generation_config(config: Option<&[u8]>) -> HuggingFaceTokenizer {
    let vocab = (0..64)
        .map(|id| {
            let text = match id {
                0 => "<unk>".to_owned(),
                5 => "test".to_owned(),
                6 => "ok".to_owned(),
                10 => "a".to_owned(),
                11 => "Ã".to_owned(),
                12 => "©".to_owned(),
                _ => format!("v{id}"),
            };
            (text, id)
        })
        .collect();
    let mut raw = tokenizers::Tokenizer::new(
        tokenizers::models::wordlevel::WordLevel::builder()
            .vocab(vocab)
            .unk_token("<unk>".into())
            .build()
            .unwrap(),
    );
    raw.with_decoder(Some(tokenizers::decoders::byte_level::ByteLevel::default()));
    raw.with_pre_tokenizer(Some(
        tokenizers::pre_tokenizers::whitespace::Whitespace::default(),
    ));
    HuggingFaceTokenizer::from_source_bytes(raw.to_string(false).unwrap().as_bytes(), None, config)
        .await
        .unwrap()
}

#[tokio::test]
async fn checked_default_plan_projects_b3_context_tail_and_freezes_under_the_original_budget() {
    let (mut session, executor) = fixture(3).await;
    // Reuse only slots whose original owners have really completed and whose
    // KV handles have dropped; this is the existing sequential CPU capability.
    executor
        .recycle_completed_bindings
        .store(true, Ordering::Release);
    Arc::get_mut(&mut session.engine.inner).unwrap().tokenizer = Arc::new(tokenizer().await);
    let model = session.configuration().model.model_id.clone();
    let root = template("test", &model)
        .unwrap()
        .with_prompt_renderer(Arc::new(PlainRenderer { model }));
    let settings = SloAutomaticCalibrationSettingsV1::default();
    let mut inputs = Box::pin(PreparedProbeInputs::new(&mut session, &settings, &[root]))
        .await
        .unwrap();
    let release = inputs.pair.clean.token_ids.len();
    let tail = inputs.context - 1;
    assert_eq!(inputs.required_geometry.probe_maximum_rows, 3);
    assert!(inputs
        .required_geometry
        .points()
        .any(|point| point == (3, tail as u32)));
    assert!(inputs
        .prompts
        .iter()
        .any(|prompt| *prompt + release == tail));
    assert!(inputs
        .prompts
        .iter()
        .zip(&inputs.outputs)
        .all(|(prompt, output)| prompt + output.get() <= inputs.context));
    // Spare capacity remains allocated even though it is not serialized.
    let retained_before = inputs.retained_payload_bytes().unwrap();
    let capacity_before = inputs.pair.clean.token_ids.capacity();
    inputs.pair.clean.token_ids.reserve(17);
    let capacity_after = inputs.pair.clean.token_ids.capacity();
    assert_eq!(
        inputs.retained_payload_bytes().unwrap() - retained_before,
        (capacity_after - capacity_before) * std::mem::size_of::<ferrum_types::TokenId>()
    );
    assert!(
        inputs.retained_payload_bytes().unwrap()
            <= inputs.population.maximum_retained_numeric_bytes
    );
    let request_limit = settings.cost_probe.maximum_probe_requests.get();
    let wave_limit = settings.cost_probe.maximum_offered_waves.get();
    let mut budget = ProbeExecutionBudget::new_with_input_projection_limit(
        Instant::now() + Duration::from_millis(settings.cost_probe.maximum_duration_ms.get()),
        settings.cost_probe.maximum_probe_requests,
        settings.cost_probe.maximum_offered_waves,
        settings.cost_probe.maximum_input_projection_requests,
    );
    let plan = Box::pin(inputs.finish(&mut session, &mut budget))
        .await
        .unwrap();
    assert_eq!(plan.preflight_charge(), budget.preflight_charge());
    assert!(plan.preflight_charge().admitted_requests > 0);
    assert!(plan.preflight_charge().projection_attempts > 0);
    assert_eq!(
        budget.requests_remaining() + plan.preflight_charge().readiness_reserved_requests,
        request_limit
    );
    assert_eq!(
        budget.input_projection_requests_remaining()
            + plan.preflight_charge().planning_reserved_requests,
        settings.cost_probe.maximum_input_projection_requests.get()
    );
    assert!(plan.audit().selected_widths.contains(&3));
    assert!(plan
        .audit()
        .required_geometry
        .points()
        .any(|point| point == (3, tail as u32)));
    assert!(plan.audit().planned_requests <= budget.requests_remaining());
    assert!(plan.audit().serial_wave_bound <= budget.attempts_remaining());
    assert!(budget.attempts_remaining() <= wave_limit);
    plan.declaration.validate().unwrap();
    assert!(
        plan.declaration
            .population
            .nonnegative_envelope
            .as_ref()
            .unwrap()
            .algorithm_universe
            .is_none(),
        "startup sources retain their original checked algorithm families"
    );
    assert_eq!(
        plan.declaration.population.schedule.block_offered,
        settings.discovery_offered_waves.get()
    );
    assert_eq!(
        plan.declaration.population.schedule.min_members,
        [ferrum_scheduler::implementations::continuous::cost_model::structured_v2::StructuredSettingsV2::default().min_phase_samples; 3]
    );
    let retained = plan.declaration.retained_payload_bytes().unwrap()
        + plan.execution.retained_payload_bytes().unwrap();
    assert!(retained <= plan.declaration.population.maximum_retained_numeric_bytes);
    assert!(plan
        .audit()
        .checked_selection
        .as_ref()
        .unwrap()
        .batches
        .iter()
        .any(|batch| batch.scheduled));
    // No source8 population is opened by input-only preparation. Real readiness,
    // if needed, must have completed before this boundary can be reached.
    session.completed_owner_boundary().unwrap();
    assert!(!session.prefix_source8);
    assert!(session.prepared_owner_capture.is_none());
    let expected_cohorts = plan.execution.cohorts.clone();
    let maximum_retained = plan.declaration.population.maximum_retained_numeric_bytes;
    let expected_schedules: Vec<_> = plan
        .audit()
        .checked_selection
        .as_ref()
        .unwrap()
        .batches
        .iter()
        .filter(|batch| batch.scheduled)
        .map(|batch| batch.schedule.clone())
        .collect();
    let parent_numerical = plan.declaration.population.settings.clone();
    let series = plan.into_series().unwrap();
    let mut original_index = 0;
    let mut signatures = std::collections::HashSet::new();
    let mut request_ids = std::collections::HashSet::new();
    for source_index in 0..series.len() {
        let source = series.source(source_index).unwrap();
        source.declaration.validate().unwrap();
        assert_eq!(
            source
                .declaration
                .population
                .nonnegative_envelope
                .as_ref()
                .unwrap()
                .algorithm_universe
                .as_ref(),
            None
        );
        assert!(signatures.insert(source.declaration.signature().unwrap()));
        assert_eq!(
            source.declaration.population.schedule,
            expected_schedules[source_index]
        );
        assert!(source
            .declaration
            .population
            .schedule
            .input_readiness
            .is_none());
        let mut expected_numerical = parent_numerical.clone();
        expected_numerical.max_phase_samples = *expected_schedules[source_index]
            .maximum_phase_members
            .iter()
            .max()
            .unwrap();
        assert_eq!(
            serde_json::to_value(&source.declaration.population.settings).unwrap(),
            serde_json::to_value(expected_numerical).unwrap()
        );
        assert!(
            source.external_retained_bytes + source.declaration.retained_payload_bytes().unwrap()
                <= maximum_retained
        );
        for (local, cohort) in source.cohorts.iter().enumerate() {
            let original = &expected_cohorts[original_index];
            assert_eq!(cohort.seed, original.seed);
            assert_eq!(cohort.template, original.template);
            assert_eq!(cohort.maximum_output, original.maximum_output);
            let (requests, _) = series.requests_for(source_index, local).unwrap();
            let declared = &source.declaration.cohort_plan.phases[cohort.pass][cohort.ordinal];
            assert_eq!(requests.len(), declared.requests.len());
            for request in requests {
                assert!(request_ids.insert(request.request.id));
                assert_eq!(
                    request.request.sampling_params.max_tokens,
                    original.maximum_output.get()
                );
            }
            original_index += 1;
        }
        assert!(series
            .requests_for(source_index, source.cohorts.len())
            .is_err());
    }
    assert_eq!(original_index, expected_cohorts.len());
    assert!(series.source(series.len()).is_err());
    session.shutdown().await.unwrap();
}

#[derive(Debug)]
struct ServeRenderer {
    model: ModelId,
    output: AutomaticCostProbeOutput,
}

impl AutomaticCostProbePromptRenderer for ServeRenderer {
    fn render_user_text(&self, text: &str) -> Result<AutomaticCostProbeTemplate> {
        serve_template(text, &self.model, self.output)
    }

    fn retained_payload_bytes(&self) -> Option<usize> {
        std::mem::size_of::<Self>().checked_add(self.model.0.capacity())
    }
}

fn serve_template(
    text: &str,
    model: &ModelId,
    output: AutomaticCostProbeOutput,
) -> Result<AutomaticCostProbeTemplate> {
    let mut request = InferenceRequest::new(text, model.clone());
    request.stream = true;
    // This test covers the default calibration budgets and real SSE codecs.
    // Configured uses the CPU fixture's actual FullLogits route; its fill does
    // not implement a device greedy repetition processor. GreedyLength still
    // contributes its original obligations, with unsupported projections left
    // Unknown. This is not coverage of the production GPU sampling kernels.
    request.sampling_params.temperature = 1.0;
    request.sampling_params.top_p = 1.0;
    request.sampling_params.top_k = Some(64);
    request.sampling_params.repetition_penalty = 1.0;
    request.api_request = Some(match output {
        AutomaticCostProbeOutput::ApiChatSse { include_usage } => {
            request.metadata.insert(
                ferrum_types::PROMPT_OPENED_REASONING_METADATA_KEY.into(),
                false.into(),
            );
            ApiRequest::Chat(ApiChatRequest {
                messages: vec![ApiChatMessage {
                    role: ApiMessageRole::User,
                    content: request.prompt.clone(),
                    name: None,
                    tool_calls: vec![],
                    tool_call_id: None,
                    function_call: None,
                }],
                tools: vec![],
                tool_choice: None,
                tool_call_protocol: Default::default(),
                legacy_functions: vec![],
                legacy_function_call: None,
                response_format: None,
                stream_options: Some(ApiStreamOptions {
                    include_usage: Some(include_usage),
                }),
            })
        }
        AutomaticCostProbeOutput::ApiCompletionSse => {
            ApiRequest::Completion(ApiCompletionRequest {
                prompt: request.prompt.clone(),
                response_format: None,
            })
        }
        AutomaticCostProbeOutput::CliText => unreachable!("serve endpoint fixture"),
    });
    AutomaticCostProbeTemplate::new(request, output)
}

fn declared_input_boundaries() -> ferrum_interfaces::vnext::ExecutorDecodeContextCoverage {
    use ferrum_interfaces::vnext::{
        BoundDecodeContextCoverage, DecodeContextBoundary, DecodeContextBoundaryKind,
        DecodeContextCoverage, ExecutorDecodeContextCoverage, NodeId, ProviderId,
    };
    use std::num::NonZeroU64;

    // These synthetic declarations exercise the planner's input obligations.
    // The CPU fill's actual provider/resource proof still decides every query;
    // it has no kernel switch at these frontiers.
    ExecutorDecodeContextCoverage {
        nodes: vec![BoundDecodeContextCoverage {
            node_id: NodeId::try_from("fixture.input-obligations".to_owned()).unwrap(),
            provider_id: ProviderId::try_from("fixture.input-obligations".to_owned()).unwrap(),
            coverage: DecodeContextCoverage::Declared {
                maximum_sequence_tokens: NonZeroU64::new(2047).unwrap(),
                boundaries: [513, 1025, 1537]
                    .map(|n| DecodeContextBoundary {
                        first_sequence_tokens: NonZeroU64::new(n).unwrap(),
                        kind: DecodeContextBoundaryKind::KernelFamily,
                    })
                    .to_vec(),
            },
        }],
    }
}

#[tokio::test]
async fn checked_default_serve_b8_context_boundaries_freeze_within_original_budgets() {
    let (mut session, executor) = fixture_with_domain(8, 2048, 2048).await;
    executor
        .recycle_completed_bindings
        .store(true, Ordering::Release);
    *executor.decode_context_coverage_override.lock() = Some(Arc::new(declared_input_boundaries()));
    // The controller fixture uses a two-event CLI queue. SSE reserves multiple
    // terminal events, so restore the real product output configuration before
    // its lazily initialized output pool admits the first request.
    let inner = Arc::get_mut(&mut session.engine.inner).unwrap();
    assert!(inner.output_credit_pool.get().is_none());
    inner.config.scheduler.slo.output = ferrum_types::EngineConfig::default().scheduler.slo.output;
    // A real tokenizer EOS keeps Configured's outcome-dependent obligations
    // distinct from GreedyLength's ignore-EOS input opportunities.
    Arc::get_mut(&mut session.engine.inner).unwrap().tokenizer =
        Arc::new(tokenizer_with_generation_config(Some(br#"{"eos_token_id":0}"#)).await);
    let model = session.configuration().model.model_id.clone();
    let endpoints = [
        AutomaticCostProbeOutput::ApiChatSse {
            include_usage: false,
        },
        AutomaticCostProbeOutput::ApiChatSse {
            include_usage: true,
        },
        AutomaticCostProbeOutput::ApiCompletionSse,
    ];
    let roots: Vec<_> = endpoints
        .iter()
        .map(|&output| {
            serve_template("test", &model, output)
                .unwrap()
                .with_prompt_renderer(Arc::new(ServeRenderer {
                    model: model.clone(),
                    output,
                }))
        })
        .collect();
    let settings = SloAutomaticCalibrationSettingsV1::default();
    assert_eq!(settings.cost_probe.maximum_output_tokens.get(), 32);
    let inputs = Box::pin(PreparedProbeInputs::new(&mut session, &settings, &roots))
        .await
        .unwrap();
    assert_eq!(inputs.context, 2048);
    assert_eq!(inputs.chunk.get(), 2048);
    assert_eq!(inputs.required_geometry.probe_maximum_rows, 8);
    assert!(
        inputs.reset,
        "CPU token-policy cache is empty at an idle boundary"
    );
    assert!(inputs.excluded_templates.is_empty());
    for tokens in [512, 513, 1024, 1025, 1536, 1537, 2047] {
        for width in 1..=8 {
            assert!(inputs
                .required_geometry
                .points()
                .any(|p| p == (width, tokens)));
        }
    }
    let release = inputs.pair.clean.token_ids.len();
    for (original, &endpoint) in endpoints.iter().enumerate() {
        for &tokens in &inputs.required_geometry.sequence_tokens {
            let matching: Vec<_> = inputs
                .original_template_indices
                .iter()
                .enumerate()
                .filter(|(index, source)| {
                    **source == original && inputs.prompts[*index] + release == tokens as usize
                })
                .map(|(index, _)| index)
                .collect();
            assert_eq!(
                matching.len(),
                1,
                "one rendered variant for endpoint {original}, frontier {tokens}"
            );
            assert_eq!(inputs.templates[matching[0]].output(), endpoint);
        }
    }
    assert!(inputs
        .prompts
        .iter()
        .zip(&inputs.outputs)
        .all(|(prompt, output)| {
            output.get() <= settings.cost_probe.maximum_output_tokens.get()
                && prompt + output.get() <= inputs.context
        }));
    let required_points: Vec<_> = inputs.required_geometry.points().collect();
    let variants = inputs.templates.len();
    let widths: Vec<_> = (1..=8).collect();
    // Retain only test diagnostics of the original case inventory. No checked
    // input, population identity or provider availability is supplied here.
    let (candidate_cases, _, _) = cases(
        &inputs.templates,
        &inputs.settings,
        &inputs.outputs,
        &widths,
        &inputs.pair,
        inputs.reset,
        session.engine.inner.model_executor.info().vocab_size,
        &inputs.original_template_indices,
        inputs.population.maximum_retained_numeric_bytes,
    )
    .unwrap();
    let inventory_admissions = readiness::initial_inventory_admissions(&candidate_cases).unwrap();
    let deadline =
        Instant::now() + Duration::from_millis(settings.cost_probe.maximum_duration_ms.get());
    let mut budget = ProbeExecutionBudget::new_with_input_projection_limit(
        deadline,
        settings.cost_probe.maximum_probe_requests,
        settings.cost_probe.maximum_offered_waves,
        settings.cost_probe.maximum_input_projection_requests,
    );
    eprintln!("default serve checked input: variants={variants}, candidates={}, required_points={}, initial_inventory_admissions={inventory_admissions}, planning_limit={}, execution_limit={}, wave_limit={}",
        candidate_cases.len(), required_points.len(), settings.cost_probe.maximum_input_projection_requests,
        settings.cost_probe.maximum_probe_requests, settings.cost_probe.maximum_offered_waves);
    let result = Box::pin(inputs.finish(&mut session, &mut budget)).await;
    let charge = budget.preflight_charge();
    eprintln!("default serve checked budget: {charge:?}, planning_remaining={}, execution_remaining={}, waves_remaining={}, physical_submissions={}",
        budget.input_projection_requests_remaining(), budget.requests_remaining(), budget.attempts_remaining(),
        executor.physical.load(Ordering::Acquire));
    assert_eq!(
        budget.input_projection_requests_remaining() + charge.planning_reserved_requests,
        settings.cost_probe.maximum_input_projection_requests.get()
    );
    assert_eq!(
        budget.requests_remaining() + charge.readiness_reserved_requests,
        settings.cost_probe.maximum_probe_requests.get()
    );
    assert!(charge.planning_admitted_requests <= charge.planning_reserved_requests);
    assert!(charge.readiness_admitted_requests <= charge.readiness_reserved_requests);
    let plan = result.unwrap_or_else(|error| {
        panic!("default serve checked freeze failed: {error}; charge={charge:?}")
    });
    let audit = plan.audit();
    let selection = audit.checked_selection.as_ref().unwrap();
    let greedy_populations: Vec<_> = selection
        .populations
        .iter()
        .filter(|population| {
            population.representative_case_indices.iter().any(|&index| {
                candidate_cases[index].preset == SloAutomaticCostProbeSamplingPresetV1::GreedyLength
            })
        })
        .collect();
    let scheduled_greedy = greedy_populations.iter().filter(|p| p.scheduled).count();
    eprintln!("default serve checked selection: populations={}, scheduled={}, batches={}, scheduled_batches={}, greedy_populations={}, scheduled_greedy={}, gaps={}, planned_requests={}, serial_waves={}; reasons={:?}",
        selection.populations.len(), selection.populations.iter().filter(|p| p.scheduled).count(), selection.batches.len(),
        selection.batches.iter().filter(|b| b.scheduled).count(), greedy_populations.len(), scheduled_greedy,
        selection.gaps.len(), audit.planned_requests, audit.serial_wave_bound,
        selection.gaps.iter().map(|gap| &gap.reason).take(8).collect::<Vec<_>>());
    assert!(
        selection.populations.iter().any(|population| {
            population.scheduled
                && population.representative_case_indices.iter().any(|&index| {
                    let case = &candidate_cases[index];
                    case.preset == SloAutomaticCostProbeSamplingPresetV1::Configured
                        && case.route == CalibrationDecodeRoute::Actual
                        && matches!(case.prefix, PrefixKind::Ordinary)
                })
        }),
        "original Configured/Actual fresh-prefill coverage must retain execution budget"
    );
    assert!(
        !greedy_populations.is_empty(),
        "auxiliary coverage must remain declared"
    );
    for population in greedy_populations
        .iter()
        .filter(|population| !population.scheduled)
    {
        assert!(
            selection.gaps.iter().any(|gap| {
                gap.population.as_ref() == Some(&population.key)
                    && matches!(
                        gap.reason,
                        selection::SelectionGapReason::RemainingRequests { .. }
                            | selection::SelectionGapReason::RemainingWaves { .. }
                            | selection::SelectionGapReason::SourceScheduleCapacity
                            | selection::SelectionGapReason::RetainedSourceCapacity { .. }
                    )
            }),
            "unreserved auxiliary source must retain its original capacity gap"
        );
    }
    for population in selection.populations.iter().filter(|p| !p.scheduled) {
        assert!(
            selection
                .gaps
                .iter()
                .any(|gap| gap.population.as_ref() == Some(&population.key)),
            "unscheduled checked population must retain an explicit gap"
        );
    }
    assert_eq!(
        audit.required_geometry.points().collect::<Vec<_>>(),
        required_points
    );
    assert_eq!(plan.preflight_charge(), charge);
    assert!(charge.projection_attempts > 0);
    assert!(!selection.execution_case_indices.is_empty());
    assert!(audit.planned_requests <= budget.requests_remaining());
    assert!(audit.serial_wave_bound <= budget.attempts_remaining());
    assert!(budget.attempts_remaining() <= settings.cost_probe.maximum_offered_waves.get());
    assert!(Instant::now() < deadline);
    plan.declaration.validate().unwrap();
    let retained = plan.declaration.retained_payload_bytes().unwrap()
        + plan.execution.retained_payload_bytes().unwrap();
    assert!(retained <= plan.declaration.population.maximum_retained_numeric_bytes);
    session.completed_owner_boundary().unwrap();
    assert!(!session.prefix_source8);
    assert!(session.prepared_owner_capture.is_none());
    session.shutdown().await.unwrap();
}
