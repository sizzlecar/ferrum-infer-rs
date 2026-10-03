//! The production declaration and bounded driver, not a hand-written population.
use super::*;
use crate::continuous_engine::inner::slo_controller::calibration::CalibrationPreparation;
use crate::{AutomaticCostProbeOutput, AutomaticCostProbeTemplate};

#[tokio::test]
async fn source8_automatic_probe_plan_installs_known_and_executes_controller_witness() {
    check_probe_and_witness(None, false).await;
}

#[tokio::test]
async fn source8_automatic_probe_plan_executes_nonterminal_forward_witness() {
    check_probe_and_witness(None, true).await;
}

#[tokio::test]
async fn source8_automatic_multi_span_prefill_and_decode_reach_known_and_adopted_execution() {
    check_probe_and_witness_with_prompt(None, false, 3).await;
}

#[tokio::test]
#[ignore = "wall-clock multi-wave feasibility probe; run explicitly on an idle host"]
async fn source8_automatic_forward_witness_with_product_planning_budget() {
    check_probe_and_witness(
        Some(ferrum_types::SloPlannerConfig::default().max_planning_us),
        true,
    )
    .await;
}

/// Wall-clock probe, separate from portable protocol correctness. Run on an
/// idle machine with an optimized test build. Its controlled CPU program is
/// not a substitute for a CUDA/Metal model workload.
#[tokio::test]
#[ignore = "wall-clock feasibility probe; run explicitly on an idle host"]
async fn source8_automatic_probe_plan_with_product_planning_budget() {
    check_probe_and_witness(
        Some(ferrum_types::SloPlannerConfig::default().max_planning_us),
        false,
    )
    .await;
}

async fn check_probe_and_witness(planning_us: Option<NonZeroU64>, lookahead: bool) {
    check_probe_and_witness_with_prompt(planning_us, lookahead, 1).await;
}

async fn check_probe_and_witness_with_prompt(
    planning_us: Option<NonZeroU64>,
    lookahead: bool,
    prompt_tokens: usize,
) {
    let (mut session, executor) = if prompt_tokens > 1 {
        automatic_checkpoint_session(2).await
    } else {
        automatic_session().await
    };
    if prompt_tokens > 1 {
        // The opt-in native fixture resolves Split and both per-row limits
        // before engine/scheduler construction. Keep that same declaration
        // throughout qualification and actual adopted execution.
        let inner = Arc::get_mut(&mut session.engine.inner).unwrap();
        assert_eq!(
            inner.config.batching.prefill_decode_execution,
            ferrum_types::PrefillDecodeExecution::Split
        );
        assert_eq!(inner.config.scheduler.prefill_step_chunk, Some(1));
        assert_eq!(inner.runtime_config.chunked_prefill_size, Some(1));
        // Time admission proves the second output token for its new target
        // together with every other request's original obligation. Each
        // prompt span needs a Prefill wave, then one Decode reaches that
        // second output. This declared fixture horizon is not a changed
        // product default.
        inner.config.scheduler.slo.planner.lookahead_waves = NonZeroUsize::new(
            prompt_tokens
                .checked_add(1)
                .expect("prompt spans plus admission Decode horizon overflow"),
        )
        .unwrap();
        // The legacy finite reference fixture declares lengths 1, 2 and 4.
        // Use the existing loaded, typed piecewise reference for intermediate
        // length 3 before admission creates each immutable request binding.
        inner.prefill_reference_runtime = Some(
            crate::continuous_engine::inner::prefill_reference_runtime::test_piecewise_calibration_runtime(),
        );
    }
    if let Some(budget) = planning_us
        .or_else(|| (lookahead || prompt_tokens > 1).then(|| NonZeroU64::new(1_000_000).unwrap()))
    {
        // The generic protocol fixture allows 30s of planning, but the real
        // planner also charges that allowance between projected waves. Its
        // requests declare a 10s token deadline, so that generic allowance
        // cannot witness two waves. Keep a portable 1s protocol allowance;
        // the separate optimized feasibility probe still uses product 2ms.
        Arc::get_mut(&mut session.engine.inner)
            .unwrap()
            .config
            .scheduler
            .slo
            .planner
            .max_planning_us = budget;
    }
    executor
        .native_structured_submission
        .store(true, Ordering::Release);
    // Install the CPU runtime's real Unsupported graph capability before the
    // original pre-submit selector runs; Unknown is correctly not eligible.
    executor.enable_structured_query_route();
    // The checked startup inventory projects decode work before calibration.
    // Enable the same actual CPU fill algebra used by the later witness now.
    executor
        .project_structured_cpu_fill
        .store(true, Ordering::Release);
    let runtime = session.engine.inner.cost_runtime.as_ref().unwrap().clone();
    // A real installed FullLogits policy supported by this controlled executor.
    // Full-vocabulary top-k keeps the discovered prefixes in the actual
    // processed distribution. It does not bypass the production prefix gate.
    let mut request = request(&session, 3);
    request.prompt = vec!["test"; prompt_tokens].join(" ");
    assert_eq!(
        session
            .engine
            .inner
            .tokenizer
            .encode(&request.prompt, true)
            .unwrap()
            .len(),
        prompt_tokens,
        "the actual tokenizer must produce the declared prompt spans"
    );
    request.sampling_params.top_k = Some(session.engine.inner.model_executor.info().vocab_size);
    request.sampling_params.stop_sequences.clear();
    // The ordinary request shares supported geometry but not the startup seed
    // content. A retained startup prefix must not supply the adoption proof.
    let mut ordinary_request = request.clone();
    ordinary_request.prompt = vec!["ok"; prompt_tokens].join(" ");
    if let Some(ferrum_types::ApiRequest::Completion(api)) = &mut ordinary_request.api_request {
        api.prompt = ordinary_request.prompt.clone();
    }
    assert_ne!(ordinary_request.prompt, request.prompt);
    assert_eq!(
        session
            .engine
            .inner
            .tokenizer
            .encode(&ordinary_request.prompt, true)
            .unwrap()
            .len(),
        prompt_tokens,
    );
    let ordinary_template =
        AutomaticCostProbeTemplate::new(ordinary_request, AutomaticCostProbeOutput::CliText)
            .unwrap();
    let template =
        AutomaticCostProbeTemplate::new(request, AutomaticCostProbeOutput::CliText).unwrap();
    let settings = ferrum_types::SloAutomaticCalibrationSettingsV1::default();
    assert_eq!(settings.discovery_offered_waves.get(), 256);
    assert_eq!(settings.phase_offered_waves.map(|n| n.get()), [256; 3]);
    assert!(runtime.snapshot().is_none());
    let before = executor.physical.load(Ordering::Acquire);
    let commands_before = executor.native_structured_counts();
    let prepared = session
        .prepare_startup_cost(&settings, &[template.clone()])
        .await
        .unwrap();
    assert_eq!(executor.physical.load(Ordering::Acquire), before);
    assert_eq!(executor.native_structured_counts(), commands_before);
    assert!(session.pending.is_none());
    assert!(session.frontiers().unwrap().is_empty());
    assert_eq!(session.engine.inner.scheduler.active_count(), 0);
    assert_eq!(session.engine.inner.scheduler.waiting_count(), 0);
    assert!(session.prepared_owner_capture.is_none());
    assert!(runtime.snapshot().is_none());
    let epoch = session
        .collect_prepared_startup_cost(&settings, prepared)
        .await
        .unwrap_or_else(|error| {
            let audit = session
                .prepared_owner_capture
                .as_ref()
                .map(|source| source.audit());
            panic!("actual default probe plan failed: {error}; population={audit:?}");
        });
    assert!(epoch > 0);
    let installed = runtime.profile_receipt().unwrap();
    assert_eq!(
        installed.storage,
        ferrum_types::SloCostProfileStorage::Memory
    );
    assert!(installed.path.is_none());
    let numerical =
        crate::continuous_engine::inner::cost_observation::automatic_numerical_settings(&settings);
    let required_fit_members = numerical
        .min_phase_samples
        .max(numerical.max_rank + numerical.min_fit_redundancy);
    let children = runtime.startup_series_children_for_test().unwrap();
    assert!(!children.is_empty());
    for child in &children {
        let phases = &child.provenance().phases;
        assert_eq!(phases.len(), 3);
        assert!(phases[0].members >= required_fit_members);
        assert!(phases[1..]
            .iter()
            .all(|phase| phase.members >= numerical.min_phase_samples));
    }
    assert!(installed.offered_samples > 0);
    assert!(runtime.snapshot().is_some());
    let prepared_prefix_totals = executor.native_prefix_terminal_totals();
    if prompt_tokens > 1 {
        assert!(
            !session
                .engine
                .inner
                .config
                .runtime
                .prefix_state_cache_enabled
        );
        let (captures, restores) = prepared_prefix_totals;
        assert!(
            captures > 0,
            "automatic source must acquire actual private checkpoints"
        );
        assert!(
            restores > captures,
            "fresh independent cohort owners must reuse those checkpoints"
        );
        assert_eq!(
            executor.native_prefix_live_lease_counts(),
            (0, 0),
            "completed sources release private checkpoints without populating the shared index"
        );
        eprintln!(
            "automatic qualified private preparation: captures={captures} restores={restores}"
        );
    }
    let (submitted, commands) = executor.native_structured_counts();
    let inference_waves = executor.physical.load(Ordering::Acquire);
    assert_eq!(
        submitted as usize,
        inference_waves + prepared_prefix_totals.0 + prepared_prefix_totals.1,
        "every native submission is an inference wave, capture, or restore"
    );
    assert_eq!(
        commands,
        inference_waves * 2,
        "each observed CPU fill ran inside actual core submission"
    );
    assert!(session.frontiers().unwrap().is_empty());

    // Keep the same qualified models, original clock, domain and authorities.
    // A fresh session owns only new requests, not a new calibration source.
    session = CalibrationSession::new_driver_session(
        session.engine,
        CalibrationLimits::new(NonZeroUsize::new(2).unwrap()).unwrap(),
    );
    let query_journal = if planning_us.is_none() {
        let path = std::env::temp_dir().join(format!(
            "ferrum-source8-witness-{}.jsonl",
            uuid::Uuid::new_v4()
        ));
        let inner = Arc::get_mut(&mut session.engine.inner).unwrap();
        let mut limits = ferrum_types::SloRequiredQueryObservationLimits::default();
        // Keep room for one complete permitted planner transaction while the
        // writer thread may be descheduled by parallel workspace tests.
        limits.queue_events = limits.max_transaction_events.try_into().unwrap();
        inner.config.scheduler.slo.required_query_observation =
            ferrum_types::SloRequiredQueryObservationConfig::StructuredRequiredV1 {
                path: path.clone(),
                limits,
            };
        let writer = crate::continuous_engine::query_observation::Writer::open(
            &inner.config.scheduler.slo.required_query_observation,
        )
        .unwrap()
        .unwrap();
        writer
            .bind_identity(&inner.config.scheduler.slo, None)
            .unwrap();
        inner.required_query_observation = Some(writer.clone());
        eprintln!("source8 witness query journal: {}", path.display());
        Some((writer, path))
    } else {
        None
    };
    executor.enable_structured_query_route();
    let mut ids = Vec::new();
    let mut consumers = Vec::new();
    for seed in 0..2 {
        let (request, contract) = ordinary_template
            .instantiate(
                NonZeroUsize::new(3).unwrap(),
                seed,
                ferrum_types::SloAutomaticCostProbeSamplingPresetV1::Configured,
            )
            .unwrap();
        ids.push(request.id.clone());
        let mut output = session
            .add_request(request, InferenceRequestContext::capture(), contract)
            .await
            .unwrap();
        consumers.push(tokio::spawn(async move {
            let mut terminal = false;
            while let Some(frame) = output.frames.next().await {
                terminal |= frame.metadata().terminal;
                drop(frame);
            }
            assert!(terminal);
            assert!(matches!(
                output.completion.await.unwrap().payload(),
                OutputCompletion::Succeeded {
                    reason: ferrum_types::FinishReason::Length,
                    ..
                }
            ));
        }));
    }
    for id in &ids {
        ready(&session, id, false).await;
    }
    for _ in &ids {
        admit(&mut session).await;
    }
    let inner = session.test_engine_inner();
    if prompt_tokens > 1 {
        crate::continuous_engine::inner::slo_controller::tests::fixture::assert_prefill_step_work_policy(
            &session.engine,
            NonZeroUsize::new(2).unwrap(),
            NonZeroUsize::MIN,
        );
        let reference = inner
            .prefill_reference_runtime
            .as_ref()
            .unwrap()
            .calibration();
        assert!(reference.piecewise_domain().is_some());
        let sequences = inner.sequences.read();
        for id in &ids {
            let binding = sequences[id]
                .prefill_reference
                .as_ref()
                .expect("actual admitted Prefill reference")
                .known()
                .expect("the actual prompt length must bind before controller search");
            assert_eq!(binding.total_prompt_tokens().get() as usize, prompt_tokens);
            assert_eq!(binding.identity(), reference.identity());
        }
    }
    let manual_timing = inner.controller_timing_snapshot().unwrap_or_default();
    // The declared per-row step limit and original B2 whole-wave capacity
    // permit both requests to advance one prompt token. Audit every actual
    // successor, including middle and final spans, before Decode below.
    for step in 1..=prompt_tokens + usize::from(!lookahead) {
        let is_prefill = step <= prompt_tokens;
        for id in &ids {
            ready(&session, id, false).await;
        }
        let rows = ids
            .iter()
            .map(|id| {
                let frontier = frontier(&session, id);
                if is_prefill {
                    frontier.prefill_work(NonZeroU32::MIN).unwrap()
                } else {
                    frontier
                        .decode_work_with_route(CalibrationDecodeRoute::FullLogits)
                        .unwrap()
                }
            })
            .collect::<Vec<_>>();
        let before = executor.physical.load(Ordering::Acquire);
        let CalibrationPreparation::Selected(prepared) = inner
            .prepare_calibration_wave(&rows, NonZeroUsize::new(2).unwrap())
            .unwrap()
        else {
            panic!("actual new-request preparation unavailable");
        };
        let facts = prepared.structured_prepared_facts(&inner, None).unwrap();
        facts.validate().unwrap();
        let expected_work = if is_prefill {
            ferrum_interfaces::execution_cost::ActualRowWork::Prefill {
                offset: (step - 1).try_into().unwrap(),
                count: 1,
                total_prompt_tokens: prompt_tokens.try_into().unwrap(),
            }
        } else {
            ferrum_interfaces::execution_cost::ActualRowWork::Decode {
                kv_tokens: prompt_tokens.try_into().unwrap(),
            }
        };
        assert_eq!(facts.exact.rows.len(), ids.len());
        assert!(facts.exact.rows.iter().all(|work| *work == expected_work));
        assert_eq!(facts.recipe.physical_host_rows().len(), ids.len());
        for row in facts.recipe.physical_host_rows() {
            assert_eq!(row.initial_prefill, is_prefill && step == 1);
            assert_eq!(row.final_prefill, is_prefill && step == prompt_tokens);
            assert_eq!(row.no_generated_history, is_prefill);
        }
        let query = StructuredQueryV2::from_future_with_domain(
            &facts.exact,
            &facts.selected,
            &facts.recipe,
            &HostContentForecastV2::Exact,
            runtime.workload_domain().unwrap(),
        )
        .unwrap();
        let snapshot = runtime.snapshot().unwrap();
        let query_now = runtime.clock.now_ns().unwrap();
        let known = snapshot
            .audit_structured_query_v2(&query, query_now)
            .unwrap_or_else(|reason| {
                let children = runtime.startup_series_children_for_test().unwrap();
                // Failure-only inspection of the original frozen children.
                // These direct results cannot substitute for catalog dispatch
                // or turn a rejected query into an execution witness.
                for child in children.iter().take(32) {
                    let Some(family) = child.numerical_family_key() else {
                        continue;
                    };
                    let projected = match child.algorithm_universe() {
                        Some(universe) => query.input().numerical_family_key_for_universe(universe),
                        None => query.input().numerical_family_key(),
                    };
                    if projected.as_ref().ok() != Some(family) {
                        continue;
                    }
                    let membership = child.catalog_input_membership(&query);
                    let prediction = child
                        .predict_query_local_with_clock_detailed(snapshot.fingerprint(), &query, query_now)
                        .map(|(value, model_now)| (value.planning_ns, value.valid_until_ns, model_now));
                    eprintln!(
                        "original rejected query child diagnostic: step={step} now={query_now} child_domain={:?} universe={:?} phase_support={:?} phase_members={:?} membership={membership:?} direct_prediction={prediction:?}",
                        child.domain_signature(),
                        child.algorithm_universe().map(|universe| universe.signature()),
                        child.phase_support_policy(),
                        child.provenance().phases.each_ref().map(|phase| phase.members),
                    );
                }
                let installed: Vec<_> = children.iter().take(32).map(|child| (
                    child.owner(), child.numerical_family_key(), child.domain_signature(),
                    child.provenance().capture_identity,
                )).collect();
                panic!(
                    "plan-generated source did not cover actual undispatched {} step {step}: {reason:?}; query_owner={:?}; query_family={:?}; installed_count={}; installed={installed:?}; feedback={}",
                    if is_prefill { "Prefill" } else { "Decode" },
                    query.input().owner(), query.input().numerical_family_key(), children.len(),
                    serde_json::to_value(runtime.audit_snapshot()).unwrap()["structured_feedback"],
                )
            });
        assert!(known.planning_ns > 0 && known.valid_for_ns > 0);
        assert_eq!(executor.physical.load(Ordering::Acquire), before);
        let timing_before = inner.controller_timing_snapshot().unwrap_or_default();
        if prompt_tokens > 1 && is_prefill {
            let frontiers_before = ids
                .iter()
                .map(|id| frontier(&session, id))
                .collect::<Vec<_>>();
            assert!(inner.slo_controller.lock().has_pending_calibration_work());
            drop(prepared);
            // Diagnostic Known is not permission to submit. Withdraw its
            // original publication and return its grants before the ordinary
            // controller independently recaptures and proves the same work.
            assert!(matches!(
                session.step(CalibrationAction::Reap).await.unwrap(),
                CalibrationTurn::Blocked(CalibrationBlockReason::PublicationUnavailable)
            ));
            for id in &ids {
                ready(&session, id, false).await;
            }
            assert!(matches!(
                session.step(CalibrationAction::Reap).await.unwrap(),
                CalibrationTurn::Blocked(CalibrationBlockReason::NoPendingWork)
            ));
            assert!(!inner.slo_controller.lock().has_pending_calibration_work());
            assert_eq!(executor.physical.load(Ordering::Acquire), before);
            assert_eq!(
                inner.controller_timing_snapshot().unwrap().witnesses,
                timing_before.witnesses,
                "withdrawn diagnostic preparation must never mint a CostWitness"
            );
            for (id, before) in ids.iter().zip(&frontiers_before) {
                let after = frontier(&session, id);
                assert_eq!(after.owner_incarnation(), before.owner_incarnation());
                assert_eq!(after.work_generation(), before.work_generation());
                assert_eq!(after.prefill_progress(), before.prefill_progress());
                assert_eq!(after.generated_tokens(), before.generated_tokens());
                assert_eq!(after.kv_tokens(), before.kv_tokens());
                assert_eq!(after.request_evidence(), before.request_evidence());
            }
            future::submit_nonterminal_witness(&session, &executor, &ids, step < prompt_tokens)
                .await;
            for id in &ids {
                let after = frontier(&session, id);
                assert_eq!(
                    after.prefill_progress(),
                    (step < prompt_tokens).then_some((step, prompt_tokens)),
                    "the adopted native wave must advance exactly the audited Prefill span"
                );
                assert_eq!(after.kv_tokens(), step);
                assert_eq!(after.generated_tokens(), usize::from(step == prompt_tokens));
            }
            continue;
        }
        let receipt = prepared.calibration_receipt().unwrap();
        inner.execute_slo_controller_wave(prepared).await.unwrap();
        runtime.drain_calibration_fixture();
        assert_eq!(
            inner.controller_timing_snapshot().unwrap().witnesses,
            timing_before.witnesses,
            "manual CompleteRequests submission is not a CostWitness"
        );
        receipt.wait_observation().await;
        let report = receipt.report(None);
        assert!(report.error.is_none(), "{report:?}");
        assert_eq!(
            report.submission,
            CalibrationSubmissionState::HostReconciled
        );
        assert_eq!(executor.physical.load(Ordering::Acquire), before + 1);
        let stages = report.host_stages.as_ref().unwrap();
        let actual = stages
            .structured_evidence
            .as_ref()
            .unwrap()
            .as_ref()
            .unwrap();
        actual.validate_host_stages(stages).unwrap();
        assert_eq!(actual.recipe(), facts.recipe.as_ref());
        assert!(stages.rows.iter().all(|row| row.terminal.is_none()));
    }
    let after_spans = inner.controller_timing_snapshot().unwrap().witnesses;
    if prompt_tokens > 1 {
        let adopted_prefills = u64::try_from(prompt_tokens).unwrap();
        assert_eq!(
            after_spans.decisions.samples,
            manual_timing.witnesses.decisions.samples + adopted_prefills
        );
        assert_eq!(
            after_spans.backend_submitted.samples,
            manual_timing.witnesses.backend_submitted.samples + adopted_prefills
        );
        assert_eq!(
            after_spans.host_reconciled.samples,
            manual_timing.witnesses.host_reconciled.samples + adopted_prefills
        );
    } else {
        assert_eq!(
            after_spans, manual_timing.witnesses,
            "manual CompleteRequests submission must not count as a CostWitness"
        );
    }
    if lookahead {
        // Two decoders each have two tokens left. Limit the original
        // controller to one row per wave so its first decision must prove
        // service for the other decoder in a future wave. Replan after every
        // actual completion; no tail is submitted speculatively.
        let before = executor.physical.load(Ordering::Acquire);
        for _ in 0..ids.len() * 2 {
            let active: Vec<_> = session
                .frontiers()
                .unwrap()
                .iter()
                .map(|row| row.request_id().clone())
                .collect();
            assert!(!active.is_empty(), "no output may complete early");
            for id in &active {
                ready(&session, id, false).await;
            }
            future::submit_capacity_limited_witness(&session, &executor, &active).await;
        }
        assert_eq!(
            executor.physical.load(Ordering::Acquire),
            before + ids.len() * 2
        );
    } else {
        // Independently witness the terminal Decode as well. Earlier
        // diagnostic Known reads and manual CompleteRequests submissions
        // cannot supply any of this decision's three acknowledgements.
        for id in &ids {
            ready(&session, id, false).await;
        }
        future::submit_terminal_witness(&session, &executor, &ids).await;
    }
    if planning_us.is_some() {
        eprintln!(
            "controlled CPU product-budget witness: {:?}",
            inner.controller_timing_snapshot().unwrap().witnesses
        );
    }
    for consumer in consumers {
        bounded(consumer).await.unwrap();
    }
    assert!(session.frontiers().unwrap().is_empty());
    let receipt_after = runtime.profile_receipt().unwrap();
    assert_eq!(
        receipt_after.source_observation_artifact_sha256,
        installed.source_observation_artifact_sha256
    );
    assert_eq!(receipt_after.loaded_unix_ns, installed.loaded_unix_ns);
    if prompt_tokens > 1 {
        assert_eq!(
            executor.native_prefix_terminal_totals(),
            prepared_prefix_totals,
            "ordinary different-content requests must not inherit private startup checkpoints"
        );
        assert_eq!(executor.native_prefix_live_lease_counts(), (0, 0));
    }
    drop(inner);
    session.shutdown().await.unwrap();
    if let Some((writer, path)) = query_journal {
        writer.close().unwrap();
        std::fs::remove_file(path).unwrap();
    }
}
