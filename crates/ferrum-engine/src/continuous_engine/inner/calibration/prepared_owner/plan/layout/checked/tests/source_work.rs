//! Reconcile complete source work with the requests the production series
//! actually instantiates. This is a CPU plan audit, not measured model speed.
use super::*;

#[tokio::test]
async fn checked_raw_domain_source_work_matches_frozen_original_requests() {
    check_source_work("historical_settings_default", None).await;
}

#[tokio::test]
async fn checked_enforce_product_source_work_matches_frozen_original_requests() {
    let config: ferrum_types::SloConfig = serde_json::from_value(serde_json::json!({
        "mode": "enforce",
        "default_service_class": "plan-audit",
        "services": [{
            "id": "plan-audit",
            "server_token_commit": {
                "ttft_ms": 10_000, "tpot_ms": 10_000, "itl_ms": 10_000
            }
        }]
    }))
    .unwrap();
    config.validate().unwrap();
    check_source_work("enforce_effective_settings", Some(config.cost_observation)).await;
}

async fn check_source_work(
    scenario: &str,
    product_cost: Option<ferrum_types::SloCostObservationConfig>,
) {
    let (mut session, executor) = fixture(2).await;
    let settings = if let Some(cost) = product_cost {
        let ferrum_types::SloLiveStructuredCalibration::AutomaticV1 { settings } =
            &cost.live_structured_calibration
        else {
            panic!("minimal Enforce configuration must select automatic calibration");
        };
        let settings = settings.clone();
        let inner = Arc::get_mut(&mut session.engine.inner).unwrap();
        let runtime = inner.cost_runtime.as_ref().unwrap();
        // Keep the CPU fixture's Observe/manual-driver isolation. Rebuild its
        // cost runtime with the same effective settings used by the plan, the
        // original physical domain and clock, and no background worker/cache.
        // This audits product calibration inputs, not normal Enforce startup.
        let runtime = crate::continuous_engine::inner::cost_observation::EngineCostRuntime::build_with_clock_and_domain(
            runtime.identity.clone(),
            runtime.clock.clone(),
            &cost,
            false,
            runtime.workload_domain().unwrap().clone(),
        )
        .unwrap();
        inner.config.scheduler.slo.cost_observation = cost;
        inner.cost_runtime = Some(Arc::new(runtime));
        assert_eq!(
            inner
                .config
                .scheduler
                .slo
                .cost_observation
                .live_structured_calibration,
            ferrum_types::SloLiveStructuredCalibration::AutomaticV1 {
                settings: settings.clone()
            }
        );
        settings
    } else {
        SloAutomaticCalibrationSettingsV1::default()
    };
    executor
        .recycle_completed_bindings
        .store(true, Ordering::Release);
    executor
        .row_selected_cpu_fill
        .store(true, Ordering::Release);
    Arc::get_mut(&mut session.engine.inner).unwrap().tokenizer = Arc::new(tokenizer().await);
    let model = session.configuration().model.model_id.clone();
    let root = template("test", &model)
        .unwrap()
        .with_prompt_renderer(Arc::new(PlainRenderer { model }));
    let inputs = Box::pin(PreparedProbeInputs::new(&mut session, &settings, &[root]))
        .await
        .unwrap();
    let (cases, _, _, _) = prepare_cases(&inputs).unwrap();
    let chunk = inputs.chunk;
    let prefill_row_ceiling = inputs.prefill_row_ceiling;
    let before = executor.native_structured_counts();
    let mut budget = ProbeExecutionBudget::new_with_input_projection_limit(
        Instant::now() + Duration::from_millis(settings.cost_probe.maximum_duration_ms.get()),
        settings.cost_probe.maximum_probe_requests,
        settings.cost_probe.maximum_offered_waves,
        settings.cost_probe.maximum_input_projection_requests,
    );
    let deadline = budget.deadline();
    let plan = Box::pin(inputs.finish(&mut session, &mut budget))
        .await
        .unwrap();
    let effective_context = plan.audit().effective_context;
    let effective_maximum_rows = plan.audit().effective_maximum_rows;
    let expected_requests = plan.audit().planned_requests;
    let expected_waves = plan.audit().serial_wave_bound;
    let expected_offer_rows = plan.audit().declared_offer_row_bound;
    assert_eq!(plan.preflight_charge(), budget.preflight_charge());
    let selection = plan.audit().checked_selection.as_ref().unwrap().clone();
    let series = plan.into_series().unwrap();
    let mut source_index = 0;
    let (mut total_requests, mut total_waves, mut total_tokens) = (0usize, 0usize, 0usize);
    for (batch_index, batch) in selection.batches.iter().enumerate() {
        let populations: Vec<_> = batch
            .population_indices
            .iter()
            .map(|&index| &selection.populations[index].key)
            .collect();
        let representatives: Vec<_> = batch
            .representative_case_indices
            .iter()
            .map(|&index| {
                let case = &cases[index];
                serde_json::json!({
                    "template":case.template, "role_target":case.product,
                    "rows":case.width, "maximum_output":case.maximum_output,
                    "preset":case.preset, "route":case.route, "prefix":case.prefix,
                })
            })
            .collect();
        let gaps: Vec<_> = selection
            .gaps
            .iter()
            .filter(|gap| {
                populations
                    .iter()
                    .any(|key| gap.population.as_ref() == Some(*key))
            })
            .map(|gap| &gap.reason)
            .collect();
        eprintln!("automatic checked source work: {}", serde_json::to_string(&serde_json::json!({
            "scenario":scenario,
            "batch":batch_index, "source":batch.scheduled.then_some(source_index),
            "scheduled":batch.scheduled, "raw_populations":populations,
            "requests":batch.requests, "serial_waves":batch.serial_wave_upper_bound,
            "token_work":batch.serial_token_work, "cycles":batch.planned_cycles,
            "phase_min_members":batch.schedule.min_members,
            "effective_context":effective_context, "effective_maximum_rows":effective_maximum_rows,
            "whole_wave_token_capacity":chunk, "prefill_row_ceiling":prefill_row_ceiling,
            "representatives":representatives, "gaps":gaps,
        })).unwrap());
        if !batch.scheduled {
            continue;
        }
        let source = series.source(source_index).unwrap();
        source.declaration.validate().unwrap();
        assert!(source
            .declaration
            .population
            .nonnegative_envelope
            .as_ref()
            .unwrap()
            .algorithm_universe
            .is_none());
        assert_eq!(source.declaration.population.schedule, batch.schedule);
        let (mut requests, mut waves, mut tokens) = (0usize, 0usize, 0usize);
        for ordinal in 0..source.cohorts.len() {
            let (original, execution) = series.requests_for(source_index, ordinal).unwrap();
            let width = original.len();
            let maximum_per_row_chunk = usize::try_from(
                crate::continuous_engine::inner::calibration::geometry_projection::prefill_chunk_for_width(
                    chunk,
                    prefill_row_ceiling,
                    width,
                )
                .expect("every selected cohort must fit the original wave and per-row capacities")
                .get(),
            )
            .unwrap();
            let per_row_chunk = execution.prefill_chunk.get() as usize;
            assert!(per_row_chunk <= maximum_per_row_chunk);
            assert_eq!(
                per_row_chunk,
                source.cohorts[ordinal]
                    .prefill_chunk
                    .map_or(maximum_per_row_chunk, |chunk| chunk.get() as usize)
            );
            for probe in original {
                let prompt_tokens = session
                    .engine
                    .inner
                    .tokenizer
                    .encode(&probe.request.prompt, true)
                    .unwrap()
                    .len();
                let decode_tokens = probe
                    .request
                    .sampling_params
                    .max_tokens
                    .checked_sub(1)
                    .unwrap();
                requests = requests.checked_add(1).unwrap();
                tokens = tokens
                    .checked_add(prompt_tokens)
                    .unwrap()
                    .checked_add(decode_tokens)
                    .unwrap();
                waves = waves
                    .checked_add(prompt_tokens.div_ceil(per_row_chunk))
                    .unwrap()
                    .checked_add(decode_tokens)
                    .unwrap();
            }
        }
        assert_eq!(
            requests, batch.requests,
            "source {source_index}: real fresh request count"
        );
        assert_eq!(
            tokens, batch.serial_token_work,
            "source {source_index}: full prompt plus every possible decode input"
        );
        assert_eq!(
            waves, batch.serial_wave_upper_bound,
            "source {source_index}: serialized actual input upper bound"
        );
        total_requests = total_requests.checked_add(requests).unwrap();
        total_waves = total_waves.checked_add(waves).unwrap();
        total_tokens = total_tokens.checked_add(tokens).unwrap();
        source_index += 1;
    }
    assert_eq!(source_index, series.len());
    assert_eq!(total_requests, expected_requests);
    assert_eq!(total_waves, expected_waves);
    assert_eq!(
        total_waves, expected_offer_rows,
        "cold declared inference rows must match the original instantiated requests"
    );
    assert_eq!(budget.deadline(), deadline);
    let preflight = budget.preflight_charge();
    assert_eq!(
        budget.requests_remaining() + preflight.readiness_reserved_requests,
        settings.cost_probe.maximum_probe_requests.get()
    );
    assert_eq!(
        budget.input_projection_requests_remaining() + preflight.planning_reserved_requests,
        settings.cost_probe.maximum_input_projection_requests.get()
    );
    eprintln!(
        "automatic checked source summary: {}",
        serde_json::to_string(&serde_json::json!({
            "scenario": scenario,
            "fixture_mode": session.configuration().scheduler.slo.mode,
            "population_schedule": settings.population_schedule,
            "numerical_strategy": settings.numerical_strategy,
            "limits": {
                "duration_ms": settings.cost_probe.maximum_duration_ms,
                "execution_requests": settings.cost_probe.maximum_probe_requests,
                "input_projection_requests": settings.cost_probe.maximum_input_projection_requests,
                "offered_waves": settings.cost_probe.maximum_offered_waves,
            },
            "preflight": {
                "admitted_requests": preflight.admitted_requests,
                "planning_admitted_requests": preflight.planning_admitted_requests,
                "readiness_admitted_requests": preflight.readiness_admitted_requests,
                "planning_reserved_requests": preflight.planning_reserved_requests,
                "readiness_reserved_requests": preflight.readiness_reserved_requests,
                "projection_attempts": preflight.projection_attempts,
            },
            "selected": {
                "sources": source_index,
                "populations": selection.populations.iter().filter(|p| p.scheduled).count(),
                "requests": total_requests,
                "serial_waves": total_waves,
                "declared_offer_row_bound": expected_offer_rows,
                "token_work": total_tokens,
            },
            "remaining_after_preflight": {
                "execution_requests": budget.requests_remaining(),
                "input_projection_requests": budget.input_projection_requests_remaining(),
                "offered_waves": budget.attempts_remaining(),
                "selection_requests": budget.selection_requests_remaining(),
                "selection_waves": budget.selection_attempts_remaining(),
            },
            // Collection has not run or reserved these sources. This is only
            // headroom after their independently reconciled declared work.
            "headroom_after_planned_collection": {
                "requests": budget.requests_remaining().checked_sub(total_requests)
                    .expect("selected source requests exceed the remaining allowance"),
                "offered_waves": budget.attempts_remaining().checked_sub(total_waves)
                    .expect("selected source waves exceed the remaining allowance"),
            },
            "gaps": &selection.gaps,
        }))
        .unwrap()
    );
    assert_eq!(
        executor.native_structured_counts(),
        before,
        "input-only audit must not execute a wave"
    );
    assert!(session.prepared_owner_capture.is_none());
    session.completed_owner_boundary().unwrap();
    session.shutdown().await.unwrap();
}
