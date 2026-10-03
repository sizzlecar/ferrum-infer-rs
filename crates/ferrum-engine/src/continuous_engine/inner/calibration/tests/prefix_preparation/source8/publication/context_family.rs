//! A real input-selected CPU algorithm can escape a homogeneous-context source.
//! Both recipes come from the same selector used by actual guarded submission.
use super::*;
use crate::continuous_engine::inner::slo_controller::calibration::CalibrationPreparation;

#[tokio::test]
async fn source8_checked_algorithm_subset_executes_unobserved_mixed_rows_with_real_witness() {
    let (mut session, executor) = automatic_session_with_wave_tokens(4).await;
    executor
        .row_selected_cpu_fill
        .store(true, Ordering::Release);
    executor
        .native_structured_submission
        .store(true, Ordering::Release);
    executor
        .project_structured_cpu_fill
        .store(true, Ordering::Release);
    executor.enable_structured_query_route();
    let mut templates = Vec::new();
    for prompt in ["test", "test test"] {
        let mut request = request(&session, 3);
        request.prompt = prompt.into();
        request.sampling_params.top_k = Some(64);
        request.sampling_params.stop_sequences.clear();
        templates.push(
            crate::AutomaticCostProbeTemplate::new(
                request,
                crate::AutomaticCostProbeOutput::CliText,
            )
            .unwrap(),
        );
    }
    let runtime = session.engine.inner.cost_runtime.as_ref().unwrap().clone();
    let settings = ferrum_types::SloAutomaticCalibrationSettingsV1::default();
    let prepared = Box::pin(session.prepare_startup_cost(&settings, &templates))
        .await
        .unwrap();
    assert_eq!(executor.physical.load(Ordering::Acquire), 0);
    Box::pin(session.collect_prepared_startup_cost(&settings, prepared))
        .await
        .unwrap();
    // The complete cold algorithm inventory is retained independently from
    // each qualified numerical source. Its declaration alone grants no Known
    // prediction; the unobserved A+B query and adopted witness below remain
    // the actual combination-generalization gate.
    let seed = runtime
        .startup_algorithm_seed()
        .expect("complete checked startup algorithm seed");
    assert!(seed.algorithm_count() >= 2);
    let children = runtime.startup_series_children_for_test().unwrap();
    assert!(!children.is_empty());
    let numerical =
        crate::continuous_engine::inner::cost_observation::automatic_numerical_settings(&settings);
    let required_fit_members = numerical
        .min_phase_samples
        .max(numerical.max_rank + numerical.min_fit_redundancy);
    let combinations = children
        .iter()
        .filter(|child| {
            child
                .algorithm_universe()
                .is_some_and(|universe| universe.algorithm_count() >= 2)
        })
        .collect::<Vec<_>>();
    assert!(
        !combinations.is_empty(),
        "the declared A+B roster requires a genuinely collected and imported combination source"
    );
    for child in &combinations {
        let phases = &child.provenance().phases;
        assert_eq!(phases.len(), 3);
        assert!(phases[0].members >= required_fit_members);
        assert!(phases[1..]
            .iter()
            .all(|phase| phase.members >= numerical.min_phase_samples));
        child
            .is_current_local(runtime.clock.now_ns().unwrap())
            .unwrap();
    }
    // The source declares its compatible algorithm scope before collecting
    // any observations. Its own complete independent phases above qualify A
    // and B; separate raw-source copies are not a prerequisite. The mixed
    // request below must still obtain a real prediction and adopted witness.
    // All training cohorts instantiate a single original template. The mixed
    // A+B CPU primitive roster has never actually executed at this boundary.
    assert_eq!(executor.native_structured_mixed_row_commands(), 0);
    session = CalibrationSession::new_driver_session(
        session.engine,
        CalibrationLimits::new(NonZeroUsize::new(2).unwrap()).unwrap(),
    );
    let mut ids = Vec::new();
    let mut consumers = Vec::new();
    for template in &templates {
        let (request, contract) = template
            .instantiate(
                NonZeroUsize::new(2).unwrap(),
                7,
                ferrum_types::SloAutomaticCostProbeSamplingPresetV1::Configured,
            )
            .unwrap();
        ids.push(request.id.clone());
        let mut output = session
            .add_request(request, InferenceRequestContext::capture(), contract)
            .await
            .unwrap();
        consumers.push(tokio::spawn(async move {
            while let Some(frame) = output.frames.next().await {
                drop(frame);
            }
            let result = output.completion.await.unwrap();
            let OutputCompletion::Succeeded { reason, usage, .. } = result.payload() else {
                panic!("actual mixed witness request failed");
            };
            assert_eq!(*reason, ferrum_types::FinishReason::Length);
            assert_eq!(usage.completion_tokens, 2);
        }));
    }
    for id in &ids {
        ready(&session, id, false).await;
    }
    for _ in &ids {
        admit(&mut session).await;
    }
    let rows = ids
        .iter()
        .enumerate()
        .map(|(index, id)| {
            frontier(&session, id)
                .prefill_work(NonZeroU32::new(index as u32 + 1).unwrap())
                .unwrap()
        })
        .collect();
    let report = wave(&mut session, &executor, rows).await;
    assert!(report.error.is_none(), "{report:?}");
    for id in &ids {
        ready(&session, id, false).await;
    }
    let rows = ids
        .iter()
        .map(|id| {
            frontier(&session, id)
                .decode_work_with_route(CalibrationDecodeRoute::FullLogits)
                .unwrap()
        })
        .collect::<Vec<_>>();
    let before = executor.physical.load(Ordering::Acquire);
    let inner = session.test_engine_inner();
    let CalibrationPreparation::Selected(prepared) = inner
        .prepare_calibration_wave(&rows, NonZeroUsize::new(2).unwrap())
        .unwrap()
    else {
        panic!("actual mixed input projection unavailable");
    };
    let facts = prepared.structured_prepared_facts(&inner, None).unwrap();
    facts.validate().unwrap();
    assert_eq!(
        facts.exact.rows,
        vec![
            ActualRowWork::Decode { kv_tokens: 1 },
            ActualRowWork::Decode { kv_tokens: 2 },
        ]
    );
    let query = StructuredQueryV2::from_future_with_domain(
        &facts.exact,
        &facts.selected,
        &facts.recipe,
        &HostContentForecastV2::Exact,
        runtime.workload_domain().unwrap(),
    )
    .unwrap();
    assert!(
        seed.contains_checked_algorithms(query.input()).unwrap(),
        "cold inventory declares the actual unobserved A+B roster; it does not qualify its cost"
    );
    assert!(
        combinations.iter().any(|child| {
            child
                .algorithm_universe()
                .unwrap()
                .contains_checked_algorithms(query.input())
                .unwrap()
        }),
        "an independently qualified combination child must contain the actual A+B input"
    );
    let known = runtime
        .snapshot()
        .unwrap()
        .audit_structured_query_v2(&query, runtime.clock.now_ns().unwrap())
        .unwrap_or_else(|reason| {
            panic!("frozen independently qualified combination source missed A+B: {reason:?}")
        });
    assert!(known.planning_ns > 0 && known.valid_for_ns > 0);
    assert_eq!(executor.physical.load(Ordering::Acquire), before);
    let frontiers_before = ids
        .iter()
        .map(|id| frontier(&session, id))
        .collect::<Vec<_>>();
    assert!(
        inner.slo_controller.lock().has_pending_calibration_work(),
        "manual preparation owns a real scheduler publication and output grants"
    );
    drop(prepared);
    drop(inner);
    // Dropping a Selected token only marks its durable flight abandoned. The
    // ordinary driver must withdraw that original publication and return its
    // grants before a different controller transaction can recapture the rows.
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
    assert!(!session
        .test_engine_inner()
        .slo_controller
        .lock()
        .has_pending_calibration_work());
    assert_eq!(
        executor.physical.load(Ordering::Acquire),
        before,
        "withdrawal cannot dispatch the diagnostic preparation"
    );
    for (id, before) in ids.iter().zip(&frontiers_before) {
        let after = frontier(&session, id);
        assert_eq!(after.owner_incarnation(), before.owner_incarnation());
        assert_eq!(after.work_generation(), before.work_generation());
        assert_eq!(after.generated_tokens(), before.generated_tokens());
        assert_eq!(after.kv_tokens(), before.kv_tokens());
        assert_eq!(after.request_evidence(), before.request_evidence());
    }
    // The real planner must independently recapture/replay this input and mint
    // an original once-only witness before the actual guarded A+B submission.
    future::submit_terminal_witness(&session, &executor, &ids).await;
    assert_eq!(executor.native_structured_mixed_row_commands(), 2);
    for consumer in consumers {
        bounded(consumer).await.unwrap();
    }
    session.shutdown().await.unwrap();
}

#[tokio::test]
async fn source8_uniform_context_does_not_cover_unseen_ragged_cpu_algorithm() {
    check_context_family(false).await;
}

#[tokio::test]
async fn source8_frozen_cold_algorithm_subset_keeps_unseen_ragged_cpu_algorithm_unknown() {
    check_context_family(true).await;
}

async fn check_context_family(freeze_subset: bool) {
    let (mut session, executor) = automatic_session().await;
    executor
        .context_partitioned_cpu_fill
        .store(true, Ordering::Release);
    executor
        .native_structured_submission
        .store(true, Ordering::Release);
    // Guarded execution needs the actual CPU runtime's declared attribution
    // and HostSynchronized readback capability even without a cold projection.
    executor.enable_structured_query_route();
    let runtime = session.engine.inner.cost_runtime.as_ref().unwrap().clone();
    let template = numerical_family::homogeneous_template(&session);
    let mut declaration = numerical_family::family_population(&session);
    if freeze_subset {
        use crate::continuous_engine::inner::calibration::geometry_projection::{
            GeometryInputTarget, GeometryProjectionLimits, GeometryProjectionPoint,
        };
        executor
            .project_structured_cpu_fill
            .store(true, Ordering::Release);
        let report = session
            .project_geometry_inputs(
                numerical_family::homogeneous_requests(&template, 2),
                &[GeometryInputTarget::Decode(GeometryProjectionPoint {
                    rows: 2,
                    sequence_tokens: 2,
                })],
                GeometryProjectionLimits {
                    deadline: tokio::time::Instant::now() + Duration::from_secs(10),
                    maximum_projections: 32,
                    maximum_route_states: 8,
                    maximum_retained_bytes: 4 * 1024 * 1024,
                    // Geometry's chunk is a whole-wave allowance: both real
                    // one-token roots must fit before their decode projection.
                    prefill_chunk: NonZeroU32::new(2).unwrap(),
                    prefill_row_ceiling: None,
                },
                &[],
            )
            .await
            .unwrap();
        assert_eq!(report.admitted_requests, 2);
        assert_eq!(report.outcomes.len(), 1);
        assert!(
            report.outcomes[0].unknown.is_none(),
            "cold A inventory target={:?}, unknown={:?}, admitted={}, attempts={}",
            report.outcomes[0].target,
            report.outcomes[0].unknown,
            report.admitted_requests,
            report.projection_attempts
        );
        assert!(!report.outcomes[0].branches.is_empty());
        let mut builder = DeclaredAlgorithmUniverseBuilderV1::new(
            declaration.population.settings.max_axes,
            1024 * 1024,
        )
        .unwrap();
        for branch in &report.outcomes[0].branches {
            builder.observe(branch.query.input()).unwrap();
        }
        declaration
            .population
            .nonnegative_envelope
            .as_mut()
            .unwrap()
            .algorithm_universe = Some(builder.finish().unwrap());
        assert_eq!(executor.physical.load(Ordering::Acquire), 0);
        assert_eq!(executor.native_structured_counts(), (0, 0));
        session.completed_owner_boundary().unwrap();
    }
    session
        .begin_prepared_owner_source(declaration, CostProfileLoadLimits::default())
        .await
        .unwrap();
    let mut budget = ProbeExecutionBudget::new(
        tokio::time::Instant::now() + Duration::from_secs(45),
        NonZeroUsize::new(64).unwrap(),
        NonZeroUsize::new(96).unwrap(),
    );
    for pass in 0..3 {
        for ordinal in 0..[16, 8, 8][pass] {
            let width = 1 + ordinal % 2;
            session.begin_prepared_owner_cohort(pass, ordinal).unwrap();
            let report = session
                .run_probe_cohort(
                    numerical_family::homogeneous_requests(&template, width),
                    ProbeCohortSettings {
        prefill_plan: crate::continuous_engine::inner::calibration::cohort_driver::ProbePrefillPlan::Joint,
                        prefill_chunk: NonZeroU32::MIN,
                        decode_route: CalibrationDecodeRoute::FullLogits,
                        reset_token_policy: false,
                    },
                    &mut budget,
                )
                .await
                .unwrap_or_else(|error| {
                    panic!("actual homogeneous source8 pass={pass} cohort={ordinal} width={width}: {error}")
                });
            assert_eq!(report.completed_requests, width);
            assert_eq!(report.completed_output_tokens, 3 * width as u64);
            session.end_prepared_owner_cohort().unwrap();
        }
    }
    let audit = session
        .prepared_owner_capture
        .as_ref()
        .unwrap()
        .prepared_audit();
    assert!(
        audit.population.owners.iter().any(
            |owner| owner.qualified && owner.owner.role == StructuredWaveRoleV2::OrdinaryDecode
        ),
        "{audit:#?}"
    );
    session.activate_prepared_owner_source().await.unwrap();
    session = CalibrationSession::new_driver_session(
        session.engine,
        CalibrationLimits::new(NonZeroUsize::new(2).unwrap()).unwrap(),
    );
    executor.enable_structured_query_route();
    executor
        .project_structured_cpu_fill
        .store(true, Ordering::Release);

    // A fresh homogeneous B2 query must actually use the trained family. This
    // positive control also executes the original production witness path.
    let uniform_family =
        numerical_family::query_and_execute(&mut session, &executor, &template, 2).await;

    let mut ids = Vec::new();
    let mut consumers = Vec::new();
    for (index, mut probe) in numerical_family::homogeneous_requests(&template, 2)
        .into_iter()
        .enumerate()
    {
        probe.request.prompt = vec!["test"; index + 1].join(" ");
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
            while let Some(frame) = output.frames.next().await {
                drop(frame);
            }
            let completion = output.completion.await.unwrap();
            let OutputCompletion::Succeeded { reason, usage, .. } = completion.payload() else {
                panic!("ragged fresh request failed");
            };
            assert_eq!(*reason, ferrum_types::FinishReason::Length);
            assert_eq!(usage.completion_tokens, 3);
        }));
    }
    for id in &ids {
        ready(&session, id, false).await;
    }
    for _ in &ids {
        admit(&mut session).await;
    }
    // Whole-wave token capacity is two. Prefill these one- and two-token
    // prompts separately so every physical prefill is complete and legal.
    for (index, id) in ids.iter().enumerate() {
        let work = frontier(&session, id)
            .prefill_work(NonZeroU32::new(index as u32 + 1).unwrap())
            .unwrap();
        let report = wave(&mut session, &executor, vec![work]).await;
        assert!(report.error.is_none(), "{report:?}");
    }
    for id in &ids {
        ready(&session, id, false).await;
    }
    let rows = ids
        .iter()
        .map(|id| {
            frontier(&session, id)
                .decode_work_with_route(CalibrationDecodeRoute::FullLogits)
                .unwrap()
        })
        .collect::<Vec<_>>();
    let inner = session.test_engine_inner();
    let before = executor.physical.load(Ordering::Acquire);
    let CalibrationPreparation::Selected(prepared) = inner
        .prepare_calibration_wave(&rows, NonZeroUsize::new(2).unwrap())
        .unwrap()
    else {
        panic!("fresh ragged B2 projection unavailable");
    };
    let facts = prepared.structured_prepared_facts(&inner, None).unwrap();
    facts.validate().unwrap();
    let contexts: Vec<_> = facts
        .exact
        .rows
        .iter()
        .map(|row| match *row {
            ActualRowWork::Decode { kv_tokens } => kv_tokens,
            _ => panic!("original fresh decode row required"),
        })
        .collect();
    assert_eq!(contexts, vec![1, 2]);
    let query = StructuredQueryV2::from_future_with_domain(
        &facts.exact,
        &facts.selected,
        &facts.recipe,
        &HostContentForecastV2::Exact,
        runtime.workload_domain().unwrap(),
    )
    .unwrap();
    assert_ne!(
        query.input().numerical_family_key().unwrap(),
        uniform_family
    );
    assert!(matches!(
        runtime
            .snapshot()
            .unwrap()
            .audit_structured_query_v2(&query, runtime.clock.now_ns().unwrap(),),
        Err(StructuredUnknownV2::WrongDomain)
    ));
    assert_eq!(executor.physical.load(Ordering::Acquire), before);

    // Execute exactly the projected wave, using the real provider, owned
    // resources, command attribution and host settlement. Unknown is a cost
    // coverage result, not a claim that this legal CPU algorithm cannot run.
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
        ready(&session, id, false).await;
    }
    let terminal = ids
        .iter()
        .map(|id| {
            frontier(&session, id)
                .decode_work_with_route(CalibrationDecodeRoute::FullLogits)
                .unwrap()
        })
        .collect();
    let report = wave(&mut session, &executor, terminal).await;
    assert!(report.error.is_none(), "{report:?}");
    for consumer in consumers {
        bounded(consumer).await.unwrap();
    }
    assert!(session.frontiers().unwrap().is_empty());
    session.shutdown().await.unwrap();
}
