use super::*;
use crate::AutomaticCostProbeOutput;
use ferrum_interfaces::execution_cost::{CostWorkloadLimitsV1, ExecutorCostIdentity};
use ferrum_tokenizer::implementations::HuggingFaceTokenizer;
use ferrum_types::{ApiCompletionRequest, ApiRequest, InferenceRequest, ModelId};
use std::num::NonZeroU64;

async fn tokenizer(pending: bool) -> HuggingFaceTokenizer {
    let surfaces = if pending {
        vec!["<unk>", "ok", "Ã", "©"]
    } else {
        vec!["<unk>", "ok", "plain"]
    };
    let vocab = surfaces
        .iter()
        .enumerate()
        .map(|(i, s)| (s.to_string(), i as u32))
        .collect();
    let mut t = tokenizers::Tokenizer::new(
        tokenizers::models::wordlevel::WordLevel::builder()
            .vocab(vocab)
            .unk_token("<unk>".into())
            .build()
            .unwrap(),
    );
    t.with_decoder(Some(tokenizers::decoders::byte_level::ByteLevel::default()));
    HuggingFaceTokenizer::from_source_bytes(t.to_string(false).unwrap().as_bytes(), None, None)
        .await
        .unwrap()
}
fn template(completion: bool) -> AutomaticCostProbeTemplate {
    let mut r = InferenceRequest::new("ok", ModelId::new("probe-test"));
    r.stream = true;
    // The product converters resolve their default temperature to zero;
    // SamplingParams::default itself is the generic temperature-one preset.
    r.sampling_params.temperature = 0.0;
    r.sampling_params.repetition_penalty = 1.1;
    let output = if completion {
        r.api_request = Some(ApiRequest::Completion(ApiCompletionRequest {
            prompt: r.prompt.clone(),
            response_format: None,
        }));
        AutomaticCostProbeOutput::ApiCompletionSse
    } else {
        AutomaticCostProbeOutput::CliText
    };
    AutomaticCostProbeTemplate::new(r, output).unwrap()
}
fn domain() -> CostWorkloadDomainV1 {
    CostWorkloadDomainV1::new_vnext(
        &ExecutorCostIdentity {
            schema_version: 1,
            model_weights: [1; 32],
            numerical_policy: [2; 32],
            device_runtime: [3; 32],
            execution_config: [4; 32],
        },
        CostWorkloadLimitsV1 {
            maximum_rows: NonZeroU32::new(8).unwrap(),
            maximum_context_tokens: NonZeroU32::new(4096).unwrap(),
            maximum_scheduled_tokens_per_wave: NonZeroU64::new(2048).unwrap(),
            output_vocabulary_elements: NonZeroU64::new(4).unwrap(),
            repetition_slot_capacity: 4096,
            fixed_state_bytes_per_row: 0,
        },
    )
    .unwrap()
}
async fn build(
    templates: &[AutomaticCostProbeTemplate],
    settings: SloAutomaticCostProbeSettingsV1,
    width: usize,
    output: usize,
    reset: bool,
) -> Result<PreparedProbePlan> {
    build_with_limits(
        templates,
        settings,
        width,
        output,
        reset,
        CostProfileLoadLimits::default(),
    )
    .await
}

async fn build_with_limits(
    templates: &[AutomaticCostProbeTemplate],
    settings: SloAutomaticCostProbeSettingsV1,
    width: usize,
    output: usize,
    reset: bool,
    limits: CostProfileLoadLimits,
) -> Result<PreparedProbePlan> {
    build_with_token_budget(
        templates,
        settings,
        width,
        output,
        reset,
        limits,
        NonZeroU32::new(2048).unwrap(),
        false,
    )
    .await
}

async fn build_with_token_budget(
    templates: &[AutomaticCostProbeTemplate],
    settings: SloAutomaticCostProbeSettingsV1,
    width: usize,
    output: usize,
    reset: bool,
    limits: CostProfileLoadLimits,
    chunk: NonZeroU32,
    complete_discovery_window: bool,
) -> Result<PreparedProbePlan> {
    let tokenizer = tokenizer(true).await;
    let mut automatic = SloAutomaticCalibrationSettingsV1::default();
    automatic.cost_probe = settings.clone();
    let maximum = NonZeroUsize::new(output).unwrap();
    let (pair, audit) = prefixes::discover(&tokenizer, templates, maximum, &settings)?;
    let mut population = population::declaration(&automatic, domain())?;
    if complete_discovery_window {
        let widths: Vec<_> = [1, 2, 4, 8]
            .into_iter()
            .filter(|n| *n <= width && *n <= chunk.get() as usize)
            .collect();
        let (cases, _, _) = layout::cases(
            templates,
            &settings,
            &vec![maximum; templates.len()],
            &widths,
            &pair,
            reset,
            domain().limits().output_vocabulary_elements.get() as usize,
            &(0..templates.len()).collect::<Vec<_>>(),
            population.maximum_retained_numeric_bytes,
        )?;
        // This explicit test declaration contains a whole original input
        // cycle, including each row's actual serial preparation. No measured
        // observation or numerical success chooses the window.
        let cycle = cases.iter().try_fold(0usize, |sum, case| {
            sum.checked_add(case.waves(1, chunk.get() as usize)?.0)
                .ok_or_else(|| error("test discovery cycle overflow"))
        })?;
        automatic.discovery_offered_waves = NonZeroUsize::new(cycle).unwrap();
        population = population::declaration(&automatic, domain())?;
    }
    layout::build(
        templates,
        &settings,
        population,
        limits,
        &vec![1; templates.len()],
        &vec![maximum; templates.len()],
        pair,
        4096,
        width,
        domain().limits().maximum_rows.get() as usize,
        chunk,
        reset,
        if reset {
            TokenPolicyResidencyInvalidation::Cleared { cleared_entries: 0 }
        } else {
            TokenPolicyResidencyInvalidation::Unsupported
        },
        audit,
        Vec::new(),
        &ferrum_interfaces::vnext::ExecutorDecodeContextCoverage { nodes: Vec::new() },
    )
}

#[tokio::test]
async fn prepared_probe_plan_declares_real_prefixes_ordinary_routes_and_fresh_requests_in_every_pass(
) {
    let plan = build(&[template(false)], Default::default(), 8, 32, true)
        .await
        .unwrap();
    plan.declaration.validate().unwrap();
    assert_eq!(
        plan.declaration.population.population_policy(),
        ferrum_scheduler::implementations::continuous::cost_model::structured_v2::StructuredPopulationPolicyV1::HomogeneousOrdinaryDecodeV1
    );
    assert_eq!(plan.declaration.population.schedule.block_offered, 256);
    assert_eq!(plan.declaration.population.schedule.min_members, [8; 3]);
    assert_eq!(
        plan.declaration.population.schedule.maximum_phase_members,
        [263; 3]
    );
    assert_eq!(
        plan.declaration.population.maximum_window_ns,
        300_000_000_000
    );
    assert!(plan.execution.audit.selected_widths.contains(&1));
    assert!(plan
        .execution
        .audit
        .selected_widths
        .iter()
        .all(|width| *width <= 8));
    assert!(plan.execution.audit.planned_requests <= 2048);
    assert!(plan.execution.audit.serial_wave_bound <= 16_384);
    let opportunities = plan.execution.audit.input_opportunities.as_ref().unwrap();
    for phase in 0..3 {
        assert!(
            opportunities.phase_cycles[phase]
                * opportunities.minimum_input_family_opportunities_per_cycle
                >= 8
        );
    }
    let (declaration, _, execution) = plan.into_parts();
    for pass in 0..3 {
        let cases = execution
            .cohorts
            .iter()
            .filter(|c| c.pass == pass)
            .collect::<Vec<_>>();
        for width in &execution.audit.selected_widths {
            for preset in [
                SloAutomaticCostProbeSamplingPresetV1::Configured,
                SloAutomaticCostProbeSamplingPresetV1::GreedyLength,
            ] {
                let cases = cases
                    .iter()
                    .filter(|c| c.width == *width && c.preset == preset)
                    .collect::<Vec<_>>();
                assert!(cases
                    .iter()
                    .any(|c| matches!(c.prefix, PrefixKind::Ordinary)));
                assert!(cases.iter().any(|c| matches!(c.prefix, PrefixKind::Clean)));
                assert!(cases
                    .iter()
                    .any(|c| matches!(c.prefix, PrefixKind::Pending)));
                assert!(cases
                    .iter()
                    .any(|c| matches!(c.route, CalibrationDecodeRoute::Actual)));
                assert!(cases
                    .iter()
                    .all(|c| c.route == CalibrationDecodeRoute::Actual));
                assert!(cases.iter().any(|c| c.reset_token_policy));
                assert!(cases.iter().any(|c| !c.reset_token_policy));
            }
        }
    }
    let mut ids = std::collections::HashSet::new();
    for c in &execution.cohorts {
        let (requests, _) = execution.requests_for(c).unwrap();
        assert_eq!(
            requests.len(),
            declaration.cohort_plan.phases[c.pass][c.ordinal]
                .requests
                .len()
        );
        for request in requests {
            assert!(ids.insert(request.request.id));
            assert_eq!(
                request.request.sampling_params.max_tokens,
                c.maximum_output.get()
            );
        }
        if let Some(prefix) = &declaration.prefix_plan.phases[c.pass][c.ordinal] {
            assert_eq!(
                prefix.release_generated + c.suffix_tokens as u64,
                c.maximum_output.get() as u64
            );
            assert_eq!(prefix.slots.len(), c.width);
            for (position, slot) in prefix.slots.iter().enumerate() {
                assert_eq!(
                    slot.expected_pending().unwrap().is_empty(),
                    match c.prefix {
                        PrefixKind::Clean => true,
                        PrefixKind::Pending => false,
                        PrefixKind::Mixed { pending_rows } => position >= pending_rows,
                        PrefixKind::Ordinary => unreachable!(),
                    }
                );
            }
        } else {
            assert!(matches!(c.maximum_output.get(), 1 | 2));
        }
    }
    assert_eq!(ids.len(), execution.audit.planned_requests);
}

fn chat_template(include_usage: bool) -> AutomaticCostProbeTemplate {
    use ferrum_types::{ApiChatMessage, ApiChatRequest, ApiMessageRole, ApiStreamOptions};
    let mut r = template(false).resolved_request().unwrap();
    r.metadata.insert(
        ferrum_types::PROMPT_OPENED_REASONING_METADATA_KEY.into(),
        false.into(),
    );
    r.api_request = Some(ApiRequest::Chat(ApiChatRequest {
        messages: vec![ApiChatMessage {
            role: ApiMessageRole::User,
            content: r.prompt.clone(),
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
    }));
    AutomaticCostProbeTemplate::new(r, AutomaticCostProbeOutput::ApiChatSse { include_usage })
        .unwrap()
}

#[tokio::test]
async fn prepared_probe_plan_default_serve_population_fits_with_each_phase_input_opportunities() {
    // The actual product combinations: Chat usage false/true each with both
    // supported presets, and Completion with its sole configured preset.
    let templates = [chat_template(false), chat_template(true), template(true)];
    let plan = build(&templates, Default::default(), 8, 32, true)
        .await
        .unwrap();
    let a = &plan.execution.audit;
    assert!(!a.selected_widths.is_empty());
    assert!(
        a.planned_requests
            <= SloAutomaticCostProbeSettingsV1::default()
                .maximum_probe_requests
                .get()
    );
    assert!(
        a.serial_wave_bound
            <= SloAutomaticCostProbeSettingsV1::default()
                .maximum_offered_waves
                .get()
    );
    let b = a.input_opportunities.as_ref().unwrap();
    assert!(b.minimum_input_family_opportunities_per_cycle >= 4);
    assert_eq!(
        b.planned_cycles,
        b.required_original_offers
            .div_ceil(b.minimum_original_offers_per_completed_cycle)
    );
    assert!(b
        .phase_original_offer_bounds
        .iter()
        .zip(b.maximum_fresh_member_span)
        .all(|(bound, span)| *bound >= span && *bound % 256 == 0));
    for phase in 0..3 {
        assert!(
            b.phase_cycles[phase] * b.minimum_input_family_opportunities_per_cycle
                >= plan.declaration.population.schedule.min_members[phase]
        );
        assert!(
            b.phase_cycles[phase] * b.minimum_original_offers_per_completed_cycle
                >= plan.declaration.population.schedule.phase_min_offered[phase]
        );
    }
    // Requested decode routes do not split an ordinary prefill input family.
    // These remain distinct original calls, never duplicated observations.
    for (template, t) in templates.iter().enumerate() {
        for &preset in &SloAutomaticCostProbeSettingsV1::default().sampling_presets {
            if !t.supports_preset(preset) {
                continue;
            }
            for width in &a.selected_widths {
                let prefill = plan
                    .execution
                    .cohorts
                    .iter()
                    .filter(|c| {
                        c.template == template
                            && c.preset == preset
                            && c.width == *width
                            && matches!(c.prefix, PrefixKind::Ordinary)
                    })
                    .count();
                assert!(
                    prefill >= 3 * 8 + 4,
                    "prefill member opportunities plus discovery"
                );
            }
        }
    }
}

#[tokio::test]
async fn prepared_probe_plan_truncated_configured_sampling_keeps_ordinary_and_audits_prefix_gap() {
    for kind in 0..3 {
        let mut r = template(false).resolved_request().unwrap();
        match kind {
            0 => r.sampling_params.top_k = Some(1),
            1 => r.sampling_params.top_p = 0.9,
            _ => r.sampling_params.min_p = Some(0.1),
        }
        let original =
            AutomaticCostProbeTemplate::new(r, AutomaticCostProbeOutput::CliText).unwrap();
        let plan = build(&[original], Default::default(), 2, 32, true)
            .await
            .unwrap();
        let configured = plan
            .execution
            .cohorts
            .iter()
            .filter(|c| c.preset == SloAutomaticCostProbeSamplingPresetV1::Configured)
            .collect::<Vec<_>>();
        assert!(!configured.is_empty());
        assert!(configured
            .iter()
            .all(|c| matches!(c.prefix, PrefixKind::Ordinary)));
        assert!(plan.execution.cohorts.iter().any(|c| c.preset
            == SloAutomaticCostProbeSamplingPresetV1::GreedyLength
            && matches!(c.prefix, PrefixKind::Pending)));
        assert_eq!(plan.execution.audit.prepared_prefix_unavailable.len(), 1);
        let manifest: serde_json::Value =
            serde_json::from_str(plan.declaration.cohort_manifest_payload.get()).unwrap();
        assert_eq!(
            manifest["prepared_prefix_unavailable"][0]["original_template_index"],
            0
        );
        assert!(!manifest["prepared_prefix_unavailable"][0]["reason"]
            .as_str()
            .unwrap()
            .is_empty());
    }
}

#[test]
fn prepared_probe_plan_full_vocabulary_top_k_does_not_truncate_prefix_candidates() {
    let mut r = template(false).resolved_request().unwrap();
    r.sampling_params.top_k = Some(4);
    let t = AutomaticCostProbeTemplate::new(r, AutomaticCostProbeOutput::CliText).unwrap();
    assert!(templates::prepared_prefix_unavailable(
        &t,
        SloAutomaticCostProbeSamplingPresetV1::Configured,
        4
    )
    .unwrap()
    .is_none());
    assert!(templates::prepared_prefix_unavailable(
        &t,
        SloAutomaticCostProbeSamplingPresetV1::Configured,
        5
    )
    .unwrap()
    .is_some());
}

#[tokio::test]
async fn prepared_probe_plan_honors_real_width_output_and_one_shared_work_budget() {
    let plan = build(&[template(false)], Default::default(), 3, 32, false)
        .await
        .unwrap();
    assert_eq!(plan.execution.audit.selected_widths, [1, 2]);
    let coverage = &plan.execution.audit.input_coverage;
    assert_eq!(
        coverage.configured_maximum_rows,
        domain().limits().maximum_rows.get() as usize
    );
    assert_eq!(coverage.planned_rows, plan.execution.audit.selected_widths);
    assert!(
        coverage
            .unprobed_rows
            .iter()
            .any(|n| *n > plan.execution.audit.effective_maximum_rows),
        "a private probe cap must not hide the rest of the configured serving domain"
    );
    assert!(plan.execution.cohorts.iter().all(|c| !c.reset_token_policy));
    let mut small = SloAutomaticCostProbeSettingsV1::default();
    small.maximum_probe_requests = NonZeroUsize::new(8).unwrap();
    assert!(build(&[template(false)], small, 8, 32, true).await.is_err());
    let mut small = SloAutomaticCostProbeSettingsV1::default();
    small.maximum_offered_waves = NonZeroUsize::new(255).unwrap();
    assert!(build(&[template(false)], small, 8, 32, true).await.is_err());
    let terminal_only = build(&[template(false)], Default::default(), 8, 2, true)
        .await
        .unwrap();
    assert!(terminal_only
        .execution
        .cohorts
        .iter()
        .all(|c| c.maximum_output.get() <= 2));
    assert!(terminal_only
        .execution
        .cohorts
        .iter()
        .any(|c| !matches!(c.prefix, PrefixKind::Ordinary)));
    assert!(terminal_only
        .execution
        .cohorts
        .iter()
        .filter(|c| !matches!(c.prefix, PrefixKind::Ordinary))
        .all(|c| c.suffix_tokens == 1));
}

#[tokio::test]
async fn prepared_probe_plan_uses_only_supported_product_presets_and_binds_real_codec() {
    let plan = build(&[template(true)], Default::default(), 2, 32, true)
        .await
        .unwrap();
    assert_eq!(plan.execution.audit.skipped_endpoint_presets, 1);
    assert!(plan
        .execution
        .cohorts
        .iter()
        .all(|c| c.preset == SloAutomaticCostProbeSamplingPresetV1::Configured));
    let value: serde_json::Value =
        serde_json::from_str(plan.declaration.cohort_manifest_payload.get()).unwrap();
    assert_eq!(
        value["templates"][0]["output"],
        serde_json::to_value(AutomaticCostProbeOutput::ApiCompletionSse).unwrap()
    );
    assert_eq!(value["templates"][0]["request"]["prompt"], "ok");
}

#[tokio::test]
async fn prepared_probe_plan_absent_pending_capability_is_unknown_not_synthetic_prefix() {
    let tokenizer = tokenizer(false).await;
    let result = prefixes::discover(
        &tokenizer,
        &[template(false)],
        NonZeroUsize::new(32).unwrap(),
        &Default::default(),
    );
    assert!(result.is_err());
}

#[test]
fn prepared_probe_plan_filters_unsupported_boundary_without_rewriting_original_template() {
    let ordinary = template(false);
    let mut request = ordinary.resolved_request().unwrap();
    request.sampling_params.response_completion_boundary =
        ferrum_types::ResponseCompletionBoundary::AfterDelimiterAndPayload {
            delimiter: "</think>".into(),
            alternate_envelope: None,
        };
    let thinking =
        AutomaticCostProbeTemplate::new(request, AutomaticCostProbeOutput::CliText).unwrap();
    let (selected, excluded) =
        templates::select(&[thinking.clone(), ordinary.clone()], 1024 * 1024).unwrap();
    assert_eq!(excluded, [0]);
    assert_eq!(selected.len(), 1);
    assert_eq!(
        selected[0].serialized_request(),
        ordinary.serialized_request()
    );
    assert!(matches!(
        thinking
            .resolved_request()
            .unwrap()
            .sampling_params
            .response_completion_boundary,
        ferrum_types::ResponseCompletionBoundary::AfterDelimiterAndPayload { .. }
    ));
    assert!(templates::select(&[thinking], 1024 * 1024).is_err());
    assert!(templates::select(&[ordinary], 1).is_err());
}

#[tokio::test]
async fn prepared_probe_plan_declares_physical_branch_challenges_without_rowspace_count_expansion()
{
    let mut settings = SloAutomaticCostProbeSettingsV1::default();
    // A larger explicit resource allowance exercises the entire declared
    // physical width, rather than pretending the default can afford it.
    settings.maximum_probe_requests = NonZeroUsize::new(65_536).unwrap();
    settings.maximum_offered_waves = NonZeroUsize::new(65_536).unwrap();
    let limits = CostProfileLoadLimits {
        max_samples: NonZeroUsize::new(65_536).unwrap(),
        ..Default::default()
    };
    // The legacy static builder still owns its declared discovery window.
    // A larger total quota alone cannot make a complete wide branch cycle
    // fit inside that unchanged window; it must report narrower selection.
    let fixed_window = build_with_limits(
        &[template(false)],
        settings.clone(),
        8,
        32,
        true,
        limits.clone(),
    )
    .await
    .unwrap();
    let original_window = SloAutomaticCalibrationSettingsV1::default()
        .discovery_offered_waves
        .get();
    assert_eq!(
        fixed_window.execution.audit.original_block_offered,
        original_window
    );
    assert!(!fixed_window.execution.audit.selected_widths.contains(&8));
    assert!(
        fixed_window
            .execution
            .audit
            .input_opportunities
            .as_ref()
            .expect("declared input opportunities")
            .successful_cycle_wave_upper_bound
            <= original_window
    );

    // Separately verify every physical branch at width eight under a window
    // derived from those same complete declared cases. The production checked
    // planner derives its own per-source schedule; this static declaration is
    // not evidence that every default hardware population fits or qualifies.
    let plan = build_with_token_budget(
        &[template(false)],
        settings,
        8,
        32,
        true,
        limits,
        NonZeroU32::new(2048).unwrap(),
        true,
    )
    .await
    .unwrap();
    assert!(plan.execution.audit.selected_widths.contains(&8));
    assert_eq!(
        plan.execution.audit.original_block_offered,
        plan.execution
            .audit
            .input_opportunities
            .as_ref()
            .expect("declared input opportunities")
            .successful_cycle_wave_upper_bound
    );
    assert!(plan.execution.audit.original_block_offered > original_window);
    for pass in 0..3 {
        for width in &plan.execution.audit.selected_widths {
            for preset in SloAutomaticCostProbeSettingsV1::default().sampling_presets {
                let mut joint = std::collections::BTreeSet::new();
                let mut pending_positions = std::collections::BTreeSet::new();
                for cohort in plan
                    .execution
                    .cohorts
                    .iter()
                    .filter(|c| c.pass == pass && c.width == *width && c.preset == preset)
                {
                    assert_eq!(cohort.route, CalibrationDecodeRoute::Actual);
                    let Some(prefix) = &plan.declaration.prefix_plan.phases[pass][cohort.ordinal]
                    else {
                        continue;
                    };
                    let mut count = 0;
                    for (position, slot) in prefix.slots.iter().enumerate() {
                        assert_eq!(slot.token_ids.len() as u64, prefix.release_generated);
                        assert_eq!(slot.token_ids.len(), slot.token_bytes.len());
                        if !slot.expected_pending().unwrap().is_empty() {
                            count += 1;
                            pending_positions.insert(position);
                        }
                    }
                    // These are source declarations. Actual EOS/errors still
                    // cannot be reclassified as qualified Length samples.
                    let length_rows = if cohort.suffix_tokens == 1 { *width } else { 0 };
                    joint.insert((count, length_rows));
                }
                assert_eq!(pending_positions, (0..*width).collect());
                assert_eq!(
                    joint,
                    [0, 1, *width]
                        .into_iter()
                        .flat_map(|count| [(count, 0), (count, *width)])
                        .collect()
                );
                if *width > 1 {
                    assert!(joint.contains(&(1, 0)));
                }
            }
        }
    }
}

#[tokio::test]
async fn prepared_probe_plan_default_resource_audit_reports_actual_selected_widths() {
    for (product, templates) in [
        ("cli", vec![template(false)]),
        (
            "serve",
            vec![chat_template(false), chat_template(true), template(true)],
        ),
    ] {
        let plan = build(&templates, Default::default(), 8, 32, true)
            .await
            .unwrap();
        let mut widths = Vec::new();
        for width in [1, 2, 4, 8] {
            let cohorts: Vec<_> = plan
                .execution
                .cohorts
                .iter()
                .filter(|c| c.width == width)
                .collect();
            let requests: usize = cohorts.iter().map(|c| c.width).sum();
            let serial: usize = cohorts
                .iter()
                .map(|c| c.width * c.maximum_output.get())
                .sum();
            widths.push(serde_json::json!({"width":width,"selected":!cohorts.is_empty(),"planned_requests":requests,"serial_wave_bound":serial,"cohorts":cohorts.len()}));
        }
        assert_eq!(
            widths
                .iter()
                .map(|v| v["planned_requests"].as_u64().unwrap())
                .sum::<u64>(),
            plan.execution.audit.planned_requests as u64
        );
        assert_eq!(
            widths
                .iter()
                .map(|v| v["serial_wave_bound"].as_u64().unwrap())
                .sum::<u64>(),
            plan.execution.audit.serial_wave_bound as u64
        );
        println!(
            "{}",
            serde_json::json!({"product":product,"prompt_tokens":1,"prefix_release_tokens":1,"audit":plan.execution.audit,"by_width":widths,"measured_duration_ms":serde_json::Value::Null,"duration_budget_ms":120000})
        );
    }
}

#[tokio::test]
async fn prepared_probe_plan_balances_original_products_and_prefill_terminal_branches() {
    let original = template(false);
    let plan = build(&[original.clone()], Default::default(), 2, 32, true)
        .await
        .unwrap();
    let cycles = plan
        .execution
        .audit
        .input_opportunities
        .as_ref()
        .unwrap()
        .planned_cycles;
    for width in &plan.execution.audit.selected_widths {
        for preset in SloAutomaticCostProbeSettingsV1::default().sampling_presets {
            assert!(templates::declared_greedy_sampling(&original, preset).unwrap());
            let mut prefill = [0usize; 2];
            let mut decode = [0usize; 2];
            let mut branches = [
                std::collections::BTreeSet::new(),
                std::collections::BTreeSet::new(),
            ];
            for case in plan
                .execution
                .cohorts
                .iter()
                .filter(|c| c.width == *width && c.preset == preset)
            {
                let Some(prefix) = &plan.declaration.prefix_plan.phases[case.pass][case.ordinal]
                else {
                    prefill[usize::from(case.maximum_output.get() != 1)] += 1;
                    assert!(matches!(case.maximum_output.get(), 1 | 2));
                    continue;
                };
                let full = prefix
                    .slots
                    .iter()
                    .any(|slot| !slot.expected_pending().unwrap().is_empty());
                decode[usize::from(full)] += 1;
                branches[usize::from(full)].insert(case.suffix_tokens);
            }
            assert_eq!(prefill, [2 * cycles; 2]);
            assert!(decode.iter().all(|n| *n >= 4 * cycles));
            assert!(branches.iter().all(|v| *v == [1, 2].into_iter().collect()));
        }
    }
}

#[test]
fn prepared_probe_plan_product_opportunity_respects_real_repetition_and_full_sampling() {
    let original = template(false);
    assert!(templates::declared_greedy_sampling(
        &original,
        SloAutomaticCostProbeSamplingPresetV1::Configured
    )
    .unwrap());
    for temperature in [0.0, 1.0] {
        let mut request = original.resolved_request().unwrap();
        request.sampling_params.temperature = temperature;
        request.sampling_params.top_k = Some(4);
        let full =
            AutomaticCostProbeTemplate::new(request, AutomaticCostProbeOutput::CliText).unwrap();
        assert!(!templates::declared_greedy_sampling(
            &full,
            SloAutomaticCostProbeSamplingPresetV1::Configured
        )
        .unwrap());
        assert!(templates::declared_greedy_sampling(
            &full,
            SloAutomaticCostProbeSamplingPresetV1::GreedyLength
        )
        .unwrap());
    }
}

#[tokio::test]
async fn prepared_probe_plan_shares_original_wave_tokens_across_the_declared_width() {
    for capacity in [1, 3, 8] {
        let plan = build_with_token_budget(
            &[template(false)],
            Default::default(),
            8,
            32,
            true,
            CostProfileLoadLimits::default(),
            NonZeroU32::new(capacity).unwrap(),
            false,
        )
        .await
        .unwrap();
        assert!(!plan.execution.cohorts.is_empty());
        for cohort in &plan.execution.cohorts {
            let (requests, settings) = plan.execution.requests_for(cohort).unwrap();
            assert_eq!(requests.len(), cohort.width);
            assert!(cohort.width <= capacity as usize);
            assert!(settings.prefill_chunk.get() as usize * cohort.width <= capacity as usize);
            assert_eq!(
                settings.prefill_chunk.get() as usize,
                capacity as usize / cohort.width
            );
        }
    }
}

#[test]
fn global_residual_strategy_reaches_prepared_source8_without_changing_fixed_population_or_budgets()
{
    use ferrum_scheduler::implementations::continuous::cost_model::structured_v2::NonNegativePlanningEstimatorV1 as Estimator;
    let mut settings = SloAutomaticCalibrationSettingsV1::default();
    let old = population::declaration(&settings, domain()).unwrap();
    settings.numerical_strategy =
        ferrum_types::SloAutomaticCalibrationNumericalStrategyV1::IdentifiedFitGlobalResidualV1;
    settings.validate().unwrap();
    let new = population::declaration(&settings, domain()).unwrap();
    assert_eq!(
        new.nonnegative_envelope
            .as_ref()
            .unwrap()
            .planning_estimator,
        Estimator::IdentifiedFitGlobalResidualV1
    );
    assert_eq!(
        old.nonnegative_envelope
            .as_ref()
            .unwrap()
            .planning_estimator,
        Estimator::IdentifiedEnvelopeV2
    );
    assert_eq!(new.schedule, old.schedule);
    assert!(
        new.schedule.input_readiness.is_none(),
        "prepared source8 keeps its original fixed cohort boundaries"
    );
    assert_eq!(
        serde_json::to_value(&new.settings).unwrap(),
        serde_json::to_value(&old.settings).unwrap()
    );
    assert_eq!(new.maximum_window_ns, old.maximum_window_ns);
    assert_eq!(
        new.maximum_retained_numeric_bytes,
        old.maximum_retained_numeric_bytes
    );
    assert_eq!(new.maximum_discovery_bytes, old.maximum_discovery_bytes);
    assert_eq!(new.route_population, old.route_population);
}
