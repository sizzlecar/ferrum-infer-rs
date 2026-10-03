//! The production cursor completes input-only inventory before a single
//! selection. These tests establish planning, not numerical/GPU qualification.
use super::*;
use std::collections::{BTreeSet, HashSet};
use std::num::NonZeroUsize;

#[tokio::test]
async fn global_cursor_real_eos_configured_policy_precedes_auxiliary_sources() {
    let (mut session, executor) = fixture(1).await;
    executor
        .recycle_completed_bindings
        .store(true, Ordering::Release);
    let inner = Arc::get_mut(&mut session.engine.inner).unwrap();
    inner.config.scheduler.slo.output = ferrum_types::EngineConfig::default().scheduler.slo.output;
    inner.tokenizer =
        Arc::new(tokenizer_with_generation_config(Some(br#"{"eos_token_id":63}"#)).await);
    let model = session.configuration().model.model_id.clone();
    // A single real EOS-enabled template prevents an unrelated EOS-disabled
    // template from hiding the old input-risk-0-only selection bug.
    let template = serve_template(
        "test",
        &model,
        AutomaticCostProbeOutput::ApiChatSse {
            include_usage: false,
        },
    )
    .unwrap();
    let request = template.resolved_request().unwrap();
    assert_ne!(
        request.metadata.get("ferrum_ignore_eos"),
        Some(&serde_json::Value::Bool(true))
    );
    let mut settings = SloAutomaticCalibrationSettingsV1::default();
    settings.maximum_retained_generations = NonZeroUsize::MIN;
    let inputs = Box::pin(PreparedProbeInputs::new(
        &mut session,
        &settings,
        &[template],
    ))
    .await
    .unwrap();
    let (cases, _, _, _) = prepare_cases(&inputs).unwrap();
    assert!(cases
        .iter()
        .any(|case| case.preset == SloAutomaticCostProbeSamplingPresetV1::GreedyLength));
    let mut cursor = inputs.into_cursor().unwrap();
    let mut budget = budget(&settings);
    let deadline = budget.deadline();
    let series = Box::pin(cursor.next(&mut session, &mut budget))
        .await
        .unwrap()
        .unwrap();
    assert_eq!(series.len(), settings.maximum_retained_generations.get());
    let first = series.source(0).unwrap();
    first.declaration.validate().unwrap();
    let manifest: serde_json::Value =
        serde_json::from_str(first.declaration.cohort_manifest_payload.get()).unwrap();
    let selection = &manifest["parent"]["child"]["checked_selection"];
    let batch = selection["batches"]
        .as_array()
        .unwrap()
        .iter()
        .find(|batch| batch["scheduled"] == true)
        .unwrap();
    assert!(batch["representative_case_indices"]
        .as_array()
        .unwrap()
        .iter()
        .all(|index| {
            let case = &cases[index.as_u64().unwrap() as usize];
            case.preset == SloAutomaticCostProbeSamplingPresetV1::Configured
                && case.route == CalibrationDecodeRoute::Actual
                && matches!(case.prefix, PrefixKind::Ordinary)
        }));
    assert!(selection["gaps"]
        .as_array()
        .unwrap()
        .iter()
        .any(|gap| gap["reason"] == "OutcomeDependentEarlyTermination"));
    assert!(!selection["gaps"]
        .as_array()
        .unwrap()
        .iter()
        .any(|gap| gap["reason"].get("DeferredInputPriority").is_some()));
    assert!(
        selection["gaps"]
            .as_array()
            .unwrap()
            .iter()
            .any(
                |gap| gap["reason"]["RetainedSourceCapacity"]["maximum_sources"]
                    == settings.maximum_retained_generations.get()
            ),
        "later complete sources remain declared under the real configured origin cap"
    );
    for ordinal in 0..first.cohorts.len() {
        let (requests, _) = series.requests_for(0, ordinal).unwrap();
        for actual in requests {
            assert_ne!(
                actual.request.metadata.get("ferrum_ignore_eos"),
                Some(&serde_json::Value::Bool(true)),
                "priority must not disable the original model EOS to manufacture qualification"
            );
            assert_eq!(
                actual.request.sampling_params.temperature,
                request.sampling_params.temperature
            );
        }
    }
    assert_eq!(budget.deadline(), deadline);
    assert_single_geometry(selection);
    assert_eq!(executor.physical.load(Ordering::Acquire), 0);
    session.completed_owner_boundary().unwrap();
    session.shutdown().await.unwrap();
}

fn budget(settings: &SloAutomaticCalibrationSettingsV1) -> ProbeExecutionBudget {
    ProbeExecutionBudget::new_with_input_projection_limit(
        Instant::now() + Duration::from_millis(settings.cost_probe.maximum_duration_ms.get()),
        settings.cost_probe.maximum_probe_requests,
        settings.cost_probe.maximum_offered_waves,
        settings.cost_probe.maximum_input_projection_requests,
    )
}

fn ledger(
    budget: &ProbeExecutionBudget,
) -> (usize, usize, usize, usize, usize, ProbePreflightCharge) {
    (
        budget.requests_remaining(),
        budget.attempts_remaining(),
        budget.selection_requests_remaining(),
        budget.selection_attempts_remaining(),
        budget.input_projection_requests_remaining(),
        budget.preflight_charge(),
    )
}

fn assert_single_geometry(selection: &serde_json::Value) -> u64 {
    let populations = selection["populations"].as_array().unwrap();
    let unique: BTreeSet<_> = populations
        .iter()
        .map(|p| serde_json::to_string(&p["key"]).unwrap())
        .collect();
    assert_eq!(
        unique.len(),
        populations.len(),
        "select each complete family only once"
    );
    let visits: u64 = populations
        .iter()
        .map(|p| p["input_geometry"]["visits"].as_u64().unwrap())
        .sum();
    assert_eq!(
        selection["input_geometry"]["cumulative_visits"], visits,
        "the original ledger contains exactly one geometry extension per final family"
    );
    visits
}

#[tokio::test]
async fn global_cursor_short_and_long_inputs_share_one_final_family_selection() {
    let (mut session, executor) = fixture(2).await;
    executor
        .recycle_completed_bindings
        .store(true, Ordering::Release);
    Arc::get_mut(&mut session.engine.inner).unwrap().tokenizer = Arc::new(tokenizer().await);
    let model = session.configuration().model.model_id.clone();
    let templates = [
        template("test", &model).unwrap(),
        template("test ok a", &model).unwrap(),
    ];
    let settings = SloAutomaticCalibrationSettingsV1::default();
    let inputs = Box::pin(PreparedProbeInputs::new(
        &mut session,
        &settings,
        &templates,
    ))
    .await
    .unwrap();
    assert_eq!(inputs.templates.len(), 2);
    let geometry_limit = inputs.input_geometry_visit_limit.unwrap();
    let (cases, _, _, _) = prepare_cases(&inputs).unwrap();
    // These already-ready groups need no preparation action. Each initial
    // complete traversal retains its checked inputs with one real owner group.
    let expected_admissions = readiness::initial_inventory_admissions(&cases).unwrap();
    let mut cursor = inputs.into_cursor().unwrap();
    let order = cursor.declared_template_order().to_vec();
    let maximum_sources = cursor.maximum_sources();
    assert!(cursor.retained_input_buffer_bytes() > 0);
    let mut budget = budget(&settings);
    let deadline = budget.deadline();
    let series = Box::pin(cursor.next(&mut session, &mut budget))
        .await
        .unwrap()
        .unwrap();
    assert_eq!(cursor.pending_templates(), 0);
    assert_eq!(cursor.pending_input_units(), 0);
    assert_eq!(cursor.maximum_sources(), maximum_sources);
    assert_eq!(
        cursor.retained_input_buffer_bytes(),
        0,
        "terminal input buffers move to the plan"
    );
    assert_eq!(budget.deadline(), deadline);
    assert_eq!(
        budget.preflight_charge().planning_reserved_requests,
        expected_admissions
    );
    assert_eq!(
        budget.preflight_charge().planning_admitted_requests,
        expected_admissions
    );
    assert_eq!(
        budget.preflight_charge().admitted_requests,
        expected_admissions
    );
    assert_eq!(budget.preflight_charge().readiness_reserved_requests, 0);
    assert_eq!(budget.preflight_charge().readiness_admitted_requests, 0);
    let first = series.source(0).unwrap();
    let manifest: serde_json::Value =
        serde_json::from_str(first.declaration.cohort_manifest_payload.get()).unwrap();
    let global = &manifest["parent"];
    assert_eq!(
        global["protocol"],
        "ferrum.automatic-prepared-probe-global-inputs.v1"
    );
    assert_eq!(global["selection_scope"], "all_declared_inputs");
    assert_eq!(
        global["completed_templates"],
        serde_json::to_value(order).unwrap()
    );
    assert_eq!(global["pending_inventory_templates"], serde_json::json!([]));
    assert_eq!(global["parent"]["templates"].as_array().unwrap().len(), 2);
    let selection = &global["child"]["checked_selection"];
    let first_complete = selection["batches"]
        .as_array()
        .unwrap()
        .iter()
        .find(|batch| batch["scheduled"] == true)
        .unwrap();
    assert!(
        first_complete["representative_case_indices"]
            .as_array()
            .unwrap()
            .iter()
            .any(|index| {
                let case = &cases[index.as_u64().unwrap() as usize];
                case.preset == ferrum_types::SloAutomaticCostProbeSamplingPresetV1::Configured
                    && case.route == CalibrationDecodeRoute::Actual
                    && matches!(case.prefix, PrefixKind::Ordinary)
            }),
        "the first sealed complete source must cover the declared Configured/Actual fresh frontier"
    );
    let populations = selection["populations"].as_array().unwrap();
    assert!(
        populations.iter().any(|p| {
            p["key"].get("NumericalFamily").is_some() && p["scheduled"] == true && {
                let templates: BTreeSet<_> = p["representative_case_indices"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|i| cases[i.as_u64().unwrap() as usize].template)
                    .collect();
                templates == BTreeSet::from([0, 1])
            }
        }),
        "one exact final numerical family retains both short and long input representatives"
    );
    let visits = assert_single_geometry(selection);
    eprintln!("global cursor: completed_templates={} sources={} planning_requests={} projection_attempts={} geometry_visits={} numerical_requests={} numerical_waves={} deadline_preserved={}",
        global["completed_templates"].as_array().unwrap().len(), series.len(),
        budget.preflight_charge().planning_reserved_requests, budget.preflight_charge().projection_attempts,
        visits, selection["requests"], selection["serial_wave_upper_bound"], budget.deadline() == deadline);
    assert!(visits > 0 && visits <= geometry_limit.get());
    assert_eq!(
        selection["input_geometry"]["maximum_visits"],
        geometry_limit.get()
    );
    assert_eq!(
        budget.selection_requests_remaining() + selection["requests"].as_u64().unwrap() as usize,
        settings.cost_probe.maximum_probe_requests.get()
    );
    assert_eq!(
        budget.selection_attempts_remaining()
            + selection["serial_wave_upper_bound"].as_u64().unwrap() as usize,
        settings.cost_probe.maximum_offered_waves.get()
    );
    assert_eq!(
        budget.input_projection_requests_remaining() + expected_admissions,
        settings.cost_probe.maximum_input_projection_requests.get()
    );
    let universe = first
        .declaration
        .population
        .nonnegative_envelope
        .as_ref()
        .unwrap()
        .algorithm_universe
        .clone();
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
                .algorithm_universe,
            universe
        );
    }
    assert!(
        universe.is_none(),
        "source qualification does not claim the global seed"
    );
    assert!(
        cursor.take_algorithm_seed().is_some(),
        "all checked algorithms remain available to online discovery"
    );
    assert!(cursor.take_algorithm_seed().is_none());
    let before = ledger(&budget);
    assert!(Box::pin(cursor.next(&mut session, &mut budget))
        .await
        .unwrap()
        .is_none());
    assert_eq!(
        ledger(&budget),
        before,
        "a drained cursor never rescans or reserves again"
    );
    assert_eq!(
        budget
            .input_geometry_work(Some(geometry_limit))
            .unwrap()
            .unwrap()
            .visits(),
        visits
    );
    session.completed_owner_boundary().unwrap();
    assert!(session.prepared_owner_capture.is_none());
    assert_eq!(executor.physical.load(Ordering::Acquire), 0);
    session.shutdown().await.unwrap();
}

#[tokio::test]
async fn global_cursor_preserves_host_identity_and_every_complete_source_horizon() {
    let (mut session, executor) = fixture(1).await;
    executor
        .recycle_completed_bindings
        .store(true, Ordering::Release);
    let inner = Arc::get_mut(&mut session.engine.inner).unwrap();
    assert!(inner.output_credit_pool.get().is_none());
    inner.config.scheduler.slo.output = ferrum_types::EngineConfig::default().scheduler.slo.output;
    inner.tokenizer =
        Arc::new(tokenizer_with_generation_config(Some(br#"{"eos_token_id":63}"#)).await);
    let model = session.configuration().model.model_id.clone();
    let first = serve_template(
        "test",
        &model,
        AutomaticCostProbeOutput::ApiChatSse {
            include_usage: false,
        },
    )
    .unwrap();
    let second_output = AutomaticCostProbeOutput::ApiChatSse {
        include_usage: true,
    };
    let mut request = serve_template("test", &model, second_output)
        .unwrap()
        .resolved_request()
        .unwrap();
    request
        .metadata
        .insert("ferrum_ignore_eos".into(), true.into());
    let second = AutomaticCostProbeTemplate::new(request, second_output).unwrap();
    let settings = SloAutomaticCalibrationSettingsV1::default();
    let inputs = Box::pin(PreparedProbeInputs::new(
        &mut session,
        &settings,
        &[first, second],
    ))
    .await
    .unwrap();
    let (cases, _, _, _) = prepare_cases(&inputs).unwrap();
    let mut cursor = inputs.into_cursor().unwrap();
    let mut budget = budget(&settings);
    let series = Box::pin(cursor.next(&mut session, &mut budget))
        .await
        .unwrap()
        .unwrap();
    let first_source = series.source(0).unwrap();
    let manifest: serde_json::Value =
        serde_json::from_str(first_source.declaration.cohort_manifest_payload.get()).unwrap();
    let selection = &manifest["parent"]["child"]["checked_selection"];
    let populations = selection["populations"].as_array().unwrap();
    let mut represented_templates = BTreeSet::new();
    let mut hosts = BTreeSet::new();
    for p in populations {
        let templates: BTreeSet<_> = p["representative_case_indices"]
            .as_array()
            .unwrap()
            .iter()
            .map(|i| cases[i.as_u64().unwrap() as usize].template)
            .collect();
        assert_eq!(
            templates.len(),
            1,
            "different declared host policies must stay separate"
        );
        represented_templates.extend(templates);
        if let Some(key) = p["key"].get("NumericalFamily") {
            hosts.insert(serde_json::to_string(&key["host_policy"]).unwrap());
        }
    }
    assert_eq!(represented_templates, BTreeSet::from([0, 1]));
    assert!(hosts.len() >= 2);
    assert!(selection["gaps"]
        .as_array()
        .unwrap()
        .iter()
        .any(|g| g["reason"] == "OutcomeDependentEarlyTermination"));
    assert!(!selection["gaps"]
        .as_array()
        .unwrap()
        .iter()
        .any(|g| g["reason"].get("DeferredInputPriority").is_some()));
    assert_single_geometry(selection);
    let mut scheduled = BTreeSet::new();
    let mut expected_cases = Vec::new();
    for batch in selection["batches"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|b| b["scheduled"] == true)
    {
        assert_eq!(batch["schedule_within_capacity"], true);
        let mut representatives = Vec::new();
        for i in batch["population_indices"].as_array().unwrap() {
            let p = &populations[i.as_u64().unwrap() as usize];
            assert!(scheduled.insert(serde_json::to_string(&p["key"]).unwrap()));
            representatives.extend(
                serde_json::from_value::<Vec<usize>>(p["representative_case_indices"].clone())
                    .unwrap(),
            );
        }
        assert_eq!(
            representatives.iter().collect::<HashSet<_>>().len(),
            representatives.len()
        );
        assert_eq!(
            serde_json::to_value(&representatives).unwrap(),
            batch["representative_case_indices"]
        );
        expected_cases
            .extend(representatives.repeat(batch["planned_cycles"].as_u64().unwrap() as usize));
    }
    assert_eq!(
        serde_json::to_value(&expected_cases).unwrap(),
        selection["execution_case_indices"]
    );
    let expected_scheduled: BTreeSet<_> = populations
        .iter()
        .filter(|p| p["scheduled"] == true)
        .map(|p| serde_json::to_string(&p["key"]).unwrap())
        .collect();
    assert_eq!(scheduled, expected_scheduled);
    let mut request_ids = HashSet::new();
    let mut cohorts = 0;
    for source_index in 0..series.len() {
        let source = series.source(source_index).unwrap();
        source.declaration.validate().unwrap();
        assert_eq!(
            source.cohorts.len(),
            source
                .declaration
                .cohort_plan
                .phases
                .iter()
                .map(Vec::len)
                .sum::<usize>()
        );
        let mut ordinals = HashSet::new();
        for (ordinal, cohort) in source.cohorts.iter().enumerate() {
            assert!(ordinals.insert((cohort.pass, cohort.ordinal)));
            let (requests, _) = series.requests_for(source_index, ordinal).unwrap();
            for request in requests {
                assert!(request_ids.insert(request.request.id));
            }
        }
        cohorts += source.cohorts.len();
    }
    assert_eq!(cohorts, expected_cases.len());
    assert_eq!(
        budget.selection_requests_remaining()
            + request_ids.len()
            + budget.preflight_charge().readiness_reserved_requests,
        settings.cost_probe.maximum_probe_requests.get()
    );
    assert_eq!(
        budget.selection_attempts_remaining()
            + selection["serial_wave_upper_bound"].as_u64().unwrap() as usize,
        settings.cost_probe.maximum_offered_waves.get()
    );
    assert_eq!(cursor.pending_input_units(), 0);
    assert_eq!(executor.physical.load(Ordering::Acquire), 0);
    session.shutdown().await.unwrap();
}

#[tokio::test]
async fn global_cursor_late_inventory_exhaustion_never_reserves_a_short_source() {
    let (mut session, executor) = fixture(2).await;
    executor
        .recycle_completed_bindings
        .store(true, Ordering::Release);
    Arc::get_mut(&mut session.engine.inner).unwrap().tokenizer = Arc::new(tokenizer().await);
    let model = session.configuration().model.model_id.clone();
    let templates = [
        template("test", &model).unwrap(),
        template("test ok a", &model).unwrap(),
    ];
    let settings = SloAutomaticCalibrationSettingsV1::default();
    let inputs = Box::pin(PreparedProbeInputs::new(
        &mut session,
        &settings,
        &templates,
    ))
    .await
    .unwrap();
    let (cases, _, _, _) = prepare_cases(&inputs).unwrap();
    let mut cursor = inputs.into_cursor().unwrap();
    let first = cursor.declared_template_order()[0];
    let first_cases: Vec<_> = cases.into_iter().filter(|c| c.template == first).collect();
    // Fund exactly the first template's uninterrupted complete captures. The
    // second template must fail before admission, after the first template's
    // complete inventory has produced the seed checked below.
    let first_admissions = readiness::initial_inventory_admissions(&first_cases).unwrap();
    let mut budget = ProbeExecutionBudget::new_with_input_projection_limit(
        Instant::now() + Duration::from_millis(settings.cost_probe.maximum_duration_ms.get()),
        settings.cost_probe.maximum_probe_requests,
        settings.cost_probe.maximum_offered_waves,
        NonZeroUsize::new(first_admissions).unwrap(),
    );
    let deadline = budget.deadline();
    let result = Box::pin(cursor.next(&mut session, &mut budget)).await;
    assert!(
        result.is_err(),
        "the second declared template cannot be silently omitted"
    );
    assert_eq!(cursor.pending_templates(), 1);
    assert_eq!(
        budget.preflight_charge().planning_reserved_requests,
        first_admissions
    );
    assert_eq!(
        budget.preflight_charge().planning_admitted_requests,
        first_admissions
    );
    assert_eq!(
        budget.preflight_charge().admitted_requests,
        first_admissions
    );
    assert_eq!(budget.preflight_charge().readiness_reserved_requests, 0);
    assert_eq!(budget.preflight_charge().readiness_admitted_requests, 0);
    assert_eq!(budget.input_projection_requests_remaining(), 0);
    assert_eq!(
        budget.selection_requests_remaining(),
        settings.cost_probe.maximum_probe_requests.get()
    );
    assert_eq!(
        budget.selection_attempts_remaining(),
        settings.cost_probe.maximum_offered_waves.get()
    );
    assert!(budget.preflight_charge().projection_attempts > 0);
    assert_eq!(budget.deadline(), deadline);
    let before = ledger(&budget);
    assert!(
        cursor.take_algorithm_seed().is_some(),
        "completed input algorithms survive a later failed inventory"
    );
    assert!(cursor.take_algorithm_seed().is_none());
    assert!(Box::pin(cursor.next(&mut session, &mut budget))
        .await
        .unwrap()
        .is_none());
    assert_eq!(
        ledger(&budget),
        before,
        "failure cannot renew any original allowance"
    );
    assert_eq!(executor.physical.load(Ordering::Acquire), 0);
    assert!(session.prepared_owner_capture.is_none());
    session.shutdown().await.unwrap();
}

#[tokio::test]
async fn global_cursor_geometry_exhaustion_keeps_original_gap_and_spent_work() {
    let (mut session, executor) = fixture(2).await;
    executor
        .recycle_completed_bindings
        .store(true, Ordering::Release);
    Arc::get_mut(&mut session.engine.inner).unwrap().tokenizer = Arc::new(tokenizer().await);
    let model = session.configuration().model.model_id.clone();
    let templates = [
        template("test", &model).unwrap(),
        template("test ok a", &model).unwrap(),
    ];
    let mut settings = SloAutomaticCalibrationSettingsV1::default();
    let ferrum_types::SloAutomaticCalibrationInputReadinessV1::WorkAxesAndBranchesV3 {
        maximum_geometry_visits,
        ..
    } = &mut settings.input_readiness
    else {
        panic!("expected production geometry policy")
    };
    *maximum_geometry_visits = std::num::NonZeroU64::new(1).unwrap();
    let limit = *maximum_geometry_visits;
    let inputs = Box::pin(PreparedProbeInputs::new(
        &mut session,
        &settings,
        &templates,
    ))
    .await
    .unwrap();
    let mut cursor = inputs.into_cursor().unwrap();
    let mut budget = budget(&settings);
    let series = Box::pin(cursor.next(&mut session, &mut budget))
        .await
        .unwrap()
        .unwrap();
    let source = series.source(0).unwrap();
    let manifest: serde_json::Value =
        serde_json::from_str(source.declaration.cohort_manifest_payload.get()).unwrap();
    let selection = &manifest["parent"]["child"]["checked_selection"];
    assert_eq!(selection["input_geometry"]["maximum_visits"], limit.get());
    assert_eq!(selection["input_geometry"]["exhausted"], true);
    assert!(selection["gaps"]
        .as_array()
        .unwrap()
        .iter()
        .any(|g| g["reason"]["InputGeometryUnavailable"]["work_exhausted"] == true));
    let visits = assert_single_geometry(selection);
    assert!(visits <= limit.get());
    let before = ledger(&budget);
    assert!(Box::pin(cursor.next(&mut session, &mut budget))
        .await
        .unwrap()
        .is_none());
    assert_eq!(ledger(&budget), before);
    let work = budget.input_geometry_work(Some(limit)).unwrap().unwrap();
    assert!(work.exhausted());
    assert_eq!(work.visits(), visits);
    assert_eq!(executor.physical.load(Ordering::Acquire), 0);
    session.shutdown().await.unwrap();
}

#[tokio::test]
async fn global_cursor_raw_algorithm_seed_preserves_complete_source_facts_and_capacity() {
    use ferrum_scheduler::implementations::continuous::cost_model::structured_v2::DeclaredAlgorithmUniverseBuilderV1;

    let (mut session, executor) = fixture(2).await;
    executor
        .recycle_completed_bindings
        .store(true, Ordering::Release);
    executor
        .row_selected_cpu_fill
        .store(true, Ordering::Release);
    Arc::get_mut(&mut session.engine.inner).unwrap().tokenizer = Arc::new(tokenizer().await);
    let model = session.configuration().model.model_id.clone();
    let settings = SloAutomaticCalibrationSettingsV1::default();
    let inputs = Box::pin(PreparedProbeInputs::new(
        &mut session,
        &settings,
        &[template("test", &model).unwrap()],
    ))
    .await
    .unwrap();
    let (cases, _, _, inventory_limit) = prepare_cases(&inputs).unwrap();
    let mut budget = budget(&settings);
    let inventory = Box::pin(collect_ready(
        &mut session,
        &inputs,
        &cases,
        &mut budget,
        inventory_limit,
    ))
    .await
    .unwrap();
    let expected =
        universe::freeze_declared_algorithms(&inventory, &inputs.population, inventory_limit)
            .unwrap()
            .unwrap();
    assert!(
        expected.algorithm_count() >= 2,
        "real installed CPU context selector supplies distinct algorithms"
    );
    let original_facts = inventory.inputs.clone();
    let original_opportunities = serde_json::to_value(&inventory.opportunities).unwrap();
    let mut raw_families = Vec::new();
    for key in original_facts
        .iter()
        .flatten()
        .filter_map(|facts| facts.original.as_ref())
        .filter_map(|input| input.numerical_family_key().ok())
    {
        if !raw_families.contains(&key) {
            raw_families.push(key);
        }
    }
    assert!(raw_families.len() >= 2);
    let retained = inventory.retained_payload_bytes().unwrap();
    let seed_bytes = expected.retained_payload_bytes().unwrap();
    let limit =
        retained + 2 * seed_bytes + std::mem::size_of::<DeclaredAlgorithmUniverseBuilderV1>();
    assert!(limit < inventory_limit);
    let before = ledger(&budget);
    let mut seed = None;
    assert!(
        universe::freeze_seed(&inventory, &inputs.population, retained, Some(&mut seed),).is_err()
    );
    assert!(
        seed.is_none(),
        "failed bounded declaration cannot publish a partial seed"
    );
    assert_eq!(
        universe::freeze_seed(&inventory, &inputs.population, limit, Some(&mut seed),).unwrap(),
        seed_bytes
    );
    assert_eq!(seed.as_ref(), Some(&expected));
    assert_eq!(
        inventory.inputs, original_facts,
        "seed freezing cannot merge numeric axes or family identity"
    );
    assert_eq!(
        serde_json::to_value(&inventory.opportunities).unwrap(),
        original_opportunities,
        "all alternatives, Unknown gaps and original fresh floors remain unchanged"
    );
    assert!(
        universe::freeze_seed(&inventory, &inputs.population, retained, Some(&mut seed),).is_err()
    );
    assert_eq!(
        seed,
        Some(expected),
        "failed later refresh preserves complete prior declaration"
    );
    assert_eq!(ledger(&budget), before);
    assert_eq!(executor.physical.load(Ordering::Acquire), 0);
    assert!(session.prepared_owner_capture.is_none());
    session.shutdown().await.unwrap();
}
