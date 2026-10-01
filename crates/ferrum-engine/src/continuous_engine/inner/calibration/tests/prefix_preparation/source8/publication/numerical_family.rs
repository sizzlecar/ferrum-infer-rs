//! Original actor/cohort/source8 publication with homogeneous policies across
//! widths, followed by new guarded queries and real controller submissions.
use super::*;
use crate::continuous_engine::inner::slo_controller::calibration::CalibrationPreparation;
use crate::{AutomaticCostProbeOutput, AutomaticCostProbeTemplate};

pub(super) fn homogeneous_template(session: &CalibrationSession) -> AutomaticCostProbeTemplate {
    let mut value = request(session, 3);
    value.sampling_params.top_k = Some(session.engine.inner.model_executor.info().vocab_size);
    value.sampling_params.stop_sequences.clear();
    AutomaticCostProbeTemplate::new(value, AutomaticCostProbeOutput::CliText).unwrap()
}

pub(super) fn homogeneous_requests(
    template: &AutomaticCostProbeTemplate,
    width: usize,
) -> Vec<ProbeRequest> {
    (0..width)
        .map(|seed| {
            let (request, contract) = template
                .instantiate(
                    NonZeroUsize::new(3).unwrap(),
                    seed as u64,
                    ferrum_types::SloAutomaticCostProbeSamplingPresetV1::Configured,
                )
                .unwrap();
            ProbeRequest { request, contract }
        })
        .collect()
}

pub(super) fn family_population(
    session: &CalibrationSession,
) -> StructuredPreparedOwnerBlockDeclarationV8 {
    let mut value = full_population(session);
    value
        .population
        .nonnegative_envelope
        .as_mut()
        .unwrap()
        .population_policy = StructuredPopulationPolicyV1::HomogeneousOrdinaryDecodeV1;
    for pass in 0..3 {
        for (ordinal, (cohort, prefix)) in value.cohort_plan.phases[pass]
            .iter_mut()
            .zip(&mut value.prefix_plan.phases[pass])
            .enumerate()
        {
            let width = 1 + ordinal % 2;
            cohort.requests.truncate(width);
            let prefix = prefix.as_mut().unwrap();
            prefix.slots.truncate(width);
            for (position, slot) in prefix.slots.iter_mut().enumerate() {
                // Each complete eight-cohort block has both widths for every
                // empty/full/intermediate prefix challenge. Width-one mixed
                // cases naturally coincide with empty or full.
                let pending = match (ordinal / 2) % 4 {
                    0 => false,
                    1 => true,
                    2 => position == 0,
                    _ => position == 1,
                };
                slot.token_ids = vec![TokenId::new(if pending { 11 } else { 10 })];
                slot.token_bytes = vec![if pending { vec![0xc3] } else { b"a".to_vec() }];
            }
            assert_eq!(prefix.slots.len(), cohort.requests.len());
        }
    }
    value.cohort_manifest_payload = serde_json::value::to_raw_value(&serde_json::json!({
        "prompt":"test", "maximum_output":3, "widths":[1,2],
        "prefix_cases":["empty","pending","pending_empty","empty_pending"],
        "output":"cli_text", "sampling":"installed_full_vocabulary_top_k"
    }))
    .unwrap();
    value
}

#[tokio::test]
async fn source8_homogeneous_family_publishes_and_executes_independent_width_queries() {
    let (mut session, executor) = automatic_session().await;
    let runtime = session.engine.inner.cost_runtime.as_ref().unwrap().clone();
    let template = homogeneous_template(&session);
    let declared = family_population(&session);
    assert!(runtime.snapshot().is_none());
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
            let width = 1 + ordinal % 2;
            session.begin_prepared_owner_cohort(pass, ordinal).unwrap();
            let summary = session
                .run_probe_cohort(homogeneous_requests(&template, width), options, &mut budget)
                .await
                .unwrap();
            assert_eq!(
                (summary.completed_requests, summary.completed_output_tokens),
                (width, 3 * width as u64)
            );
            assert_eq!(
                (
                    summary.wave_attempts,
                    summary.reconciled_waves,
                    summary.released_prefix_rows
                ),
                (3, 3, width)
            );
            session.end_prepared_owner_cohort().unwrap();
        }
    }
    let audit = session
        .prepared_owner_capture
        .as_ref()
        .unwrap()
        .prepared_audit();
    assert!(!audit.population.poisoned, "{audit:#?}");
    assert_eq!(
        (audit.population.offered, audit.preparation_attempts),
        (96, 32)
    );
    let qualified_decode: Vec<_> = audit
        .population
        .owners
        .iter()
        .filter(|owner| owner.qualified && owner.owner.role == StructuredWaveRoleV2::OrdinaryDecode)
        .collect();
    // New-policy homogeneous decode can qualify only as its family. This also
    // excludes two independently trained exact-width models explaining lookup.
    assert_eq!(qualified_decode.len(), 1, "{audit:#?}");
    assert_eq!(qualified_decode[0].owner.rows, 2);
    assert!(session.frontiers().unwrap().is_empty());
    let epoch = session
        .activate_prepared_owner_source()
        .await
        .unwrap_or_else(|error| panic!("family source8 activation: {error}; {audit:#?}"));
    assert!(epoch > 0);
    let installed = runtime.profile_receipt().unwrap();
    assert_eq!(
        installed.storage,
        ferrum_types::SloCostProfileStorage::Memory
    );
    assert_eq!(installed.offered_samples, 96);

    session = CalibrationSession::new_driver_session(
        session.engine,
        CalibrationLimits::new(NonZeroUsize::new(2).unwrap()).unwrap(),
    );
    executor.enable_structured_query_route();
    executor
        .project_structured_cpu_fill
        .store(true, Ordering::Release);
    let mut expected_family = None;
    for width in [1, 2] {
        let key = query_and_execute(&mut session, &executor, &template, width).await;
        if let Some(expected) = expected_family {
            assert_eq!(key, expected);
        } else {
            expected_family = Some(key);
        }
    }
    let after = runtime.profile_receipt().unwrap();
    assert_eq!(
        after.source_observation_artifact_sha256,
        installed.source_observation_artifact_sha256
    );
    assert_eq!(after.loaded_unix_ns, installed.loaded_unix_ns);
    session.shutdown().await.unwrap();
}

pub(super) async fn query_and_execute(
    session: &mut CalibrationSession,
    executor: &Arc<ControlledExecutor>,
    template: &AutomaticCostProbeTemplate,
    width: usize,
) -> NumericalFamilyKeyV1 {
    let runtime = session.engine.inner.cost_runtime.as_ref().unwrap().clone();
    let mut ids = Vec::new();
    let mut consumers = Vec::new();
    for probe in homogeneous_requests(template, width) {
        ids.push(probe.request.id.clone());
        let mut output = session
            .add_request(
                probe.request,
                InferenceRequestContext::capture(),
                probe.contract,
            )
            .await
            .unwrap();
        consumers.push(tokio::spawn(async move {
            let mut terminal = false;
            while let Some(frame) = output.frames.next().await {
                terminal |= frame.metadata().terminal;
                drop(frame);
            }
            assert!(terminal);
            let completed = output.completion.await.unwrap();
            let OutputCompletion::Succeeded { reason, usage, .. } = completed.payload() else {
                panic!("fresh family query did not complete");
            };
            assert_eq!(*reason, ferrum_types::FinishReason::Length);
            assert_eq!(usage.completion_tokens, 3);
        }));
    }
    for id in &ids {
        ready(session, id, false).await;
    }
    for _ in &ids {
        admit(session).await;
    }
    let prefills = ids
        .iter()
        .map(|id| frontier(session, id).prefill_work(NonZeroU32::MIN).unwrap())
        .collect();
    let report = wave(session, executor, prefills).await;
    assert!(report.error.is_none(), "{report:?}");
    for id in &ids {
        ready(session, id, false).await;
    }
    let rows = ids
        .iter()
        .map(|id| {
            frontier(session, id)
                .decode_work_with_route(CalibrationDecodeRoute::FullLogits)
                .unwrap()
        })
        .collect::<Vec<_>>();
    let inner = session.test_engine_inner();
    let before = executor.physical.load(Ordering::Acquire);
    let CalibrationPreparation::Selected(prepared) = inner
        .prepare_calibration_wave(&rows, NonZeroUsize::new(width).unwrap())
        .unwrap()
    else {
        panic!("fresh B{width} guarded wave unavailable");
    };
    let facts = prepared.structured_prepared_facts(&inner, None).unwrap();
    facts.validate().unwrap();
    let query = StructuredQueryV2::from_future_with_domain(
        &facts.exact,
        &facts.selected,
        &facts.recipe,
        &HostContentForecastV2::Exact,
        runtime.workload_domain().unwrap(),
    )
    .unwrap();
    assert_eq!(query.owner().rows as usize, width);
    let family = query.input().numerical_family_key().unwrap();
    let known = runtime
        .snapshot()
        .unwrap()
        .audit_structured_query_v2(&query, runtime.clock.now_ns().unwrap())
        .unwrap_or_else(|reason| {
            panic!("qualified family did not cover undispatched B{width}: {reason:?}")
        });
    assert!(known.planning_ns > 0 && known.valid_for_ns > 0);
    assert_eq!(executor.physical.load(Ordering::Acquire), before);
    let receipt = prepared.calibration_receipt().unwrap();
    inner.execute_slo_controller_wave(prepared).await.unwrap();
    runtime.drain_calibration_fixture();
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
    drop(inner);
    for id in &ids {
        ready(session, id, false).await;
    }
    // Reuse the production planner witness audit, including real submission
    // and host reconciliation, to finish the original output contracts.
    future::submit_terminal_witness(session, executor, &ids).await;
    for consumer in consumers {
        bounded(consumer).await.unwrap();
    }
    assert!(session.frontiers().unwrap().is_empty());
    family
}
