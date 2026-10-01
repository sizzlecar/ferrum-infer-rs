//! Qualified original source8 children exercise snapshot subset lookup and
//! actual feedback. No model, source receipt or algorithm digest is fabricated.
use super::*;
use crate::continuous_engine::inner::calibration::geometry_projection::{
    GeometryInputTarget, GeometryProjectionLimits, GeometryProjectionPoint,
};
use crate::continuous_engine::inner::slo_controller::calibration::CalibrationPreparation;

#[tokio::test]
async fn source8_snapshot_checked_subset_keeps_catalog_after_new_cpu_algorithm() {
    check_snapshot_subset(false, true).await;
}

#[tokio::test]
async fn source8_complete_subset_keeps_real_lookup_and_witness_with_larger_global_seed() {
    check_snapshot_subset(true, true).await;
}

#[tokio::test]
async fn source8_complete_raw_domain_keeps_real_lookup_and_witness_with_larger_global_seed() {
    check_snapshot_subset(true, false).await;
}

async fn check_snapshot_subset(original_automatic_floors: bool, declare_subset_universe: bool) {
    let (mut session, executor) = automatic_session().await;
    executor
        .context_partitioned_cpu_fill
        .store(true, Ordering::Release);
    executor
        .native_structured_submission
        .store(true, Ordering::Release);
    executor
        .project_structured_cpu_fill
        .store(true, Ordering::Release);
    executor.enable_structured_query_route();
    let runtime = session.engine.inner.cost_runtime.as_ref().unwrap().clone();
    let template = numerical_family::homogeneous_template(&session);
    let mut declaration = numerical_family::family_population(&session);
    let mut counts = [16_usize, 8, 8];
    let mut required_members = [8_usize; 3];
    if original_automatic_floors {
        let automatic = ferrum_types::SloAutomaticCalibrationSettingsV1::default();
        let mut numerical =
            crate::continuous_engine::inner::cost_observation::automatic_numerical_settings(
                &automatic,
            );
        let fit = numerical.min_phase_samples.max(
            numerical
                .max_rank
                .checked_add(numerical.min_fit_redundancy)
                .unwrap(),
        );
        required_members = [
            fit,
            numerical.min_phase_samples,
            numerical.min_phase_samples,
        ];
        // Every original cohort executes one prefix preparation and two
        // numerical waves. Count full fresh cohorts at each phase, including
        // discovery, rather than multiplying one observation into members.
        counts = [fit.checked_mul(2).unwrap(), fit, fit];
        let block = fit.checked_mul(3).unwrap();
        let mut schedule = OwnerBlockScheduleV1::new(block, [block; 3], required_members).unwrap();
        schedule.prediction_validity =
            crate::continuous_engine::inner::cost_observation::automatic_prediction_validity(
                &automatic,
            );
        numerical.max_phase_samples = *schedule.maximum_phase_members.iter().max().unwrap();
        numerical.validate().unwrap();
        declaration.population.settings = numerical;
        declaration.population.schedule = schedule;
        declaration.population.maximum_window_ns = automatic.maximum_window_ns.get();
        for pass in 0..3 {
            let original = declaration.cohort_plan.phases[pass].clone();
            let prefixes = declaration.prefix_plan.phases[pass].clone();
            declaration.cohort_plan.phases[pass] = (0..counts[pass])
                .map(|ordinal| {
                    let mut cohort = original[ordinal % original.len()].clone();
                    cohort.manifest_case = u32::try_from(ordinal).unwrap();
                    cohort
                })
                .collect();
            declaration.prefix_plan.phases[pass] = (0..counts[pass])
                .map(|ordinal| prefixes[ordinal % prefixes.len()].clone())
                .collect();
        }
        declaration.maximum_offered_waves = counts.iter().sum::<usize>().checked_mul(3).unwrap();
        declaration.cohort_manifest_payload = serde_json::value::to_raw_value(&serde_json::json!({
            "source_scope": "original-uniform-context-cpu-algorithm",
            "parent_obligations": ["uniform-context", "ragged-context"],
            "uncollected_scope": "ragged-context remains unqualified",
            "phase_cohorts": counts,
            "required_fresh_members": required_members,
            "original_cohort_plan": declaration.cohort_plan,
            "original_prefix_plan": declaration.prefix_plan,
        }))
        .unwrap();
        declaration.validate().unwrap();
    }
    // Both uniform-context frontiers select the scalar CPU implementation.
    // The same unchanged selector chooses another implementation for ragged rows.
    let report = session
        .project_geometry_inputs(
            numerical_family::homogeneous_requests(&template, 2),
            &[2, 3].map(|sequence_tokens| {
                GeometryInputTarget::Decode(GeometryProjectionPoint {
                    rows: 2,
                    sequence_tokens,
                })
            }),
            GeometryProjectionLimits {
                deadline: tokio::time::Instant::now() + Duration::from_secs(10),
                maximum_projections: 64,
                maximum_route_states: 8,
                maximum_retained_bytes: 4 * 1024 * 1024,
                prefill_chunk: NonZeroU32::new(2).unwrap(),
                prefill_row_ceiling: None,
            },
            &[],
        )
        .await
        .unwrap();
    assert_eq!(report.admitted_requests, 2);
    assert_eq!(report.outcomes.len(), 2);
    let mut builder = DeclaredAlgorithmUniverseBuilderV1::new(
        declaration.population.settings.max_axes,
        1024 * 1024,
    )
    .unwrap();
    for outcome in &report.outcomes {
        assert!(
            outcome.unknown.is_none(),
            "target={:?}, unknown={:?}, admitted={}, attempts={}",
            outcome.target,
            outcome.unknown,
            report.admitted_requests,
            report.projection_attempts
        );
        assert!(!outcome.branches.is_empty());
        for branch in &outcome.branches {
            builder.observe(branch.query.input()).unwrap();
        }
    }
    let universe = builder.finish().unwrap();
    assert_eq!(universe.algorithm_count(), 1);
    if declare_subset_universe {
        declaration
            .population
            .nonnegative_envelope
            .as_mut()
            .unwrap()
            .algorithm_universe = Some(universe.clone());
    } else {
        assert!(declaration
            .population
            .nonnegative_envelope
            .as_ref()
            .unwrap()
            .algorithm_universe
            .is_none());
    }
    assert_eq!(executor.native_structured_counts(), (0, 0));
    let qualified_query = report.outcomes[0].branches[0].query.clone();
    drop(report);
    session.completed_owner_boundary().unwrap();
    session
        .begin_prepared_owner_source(declaration, CostProfileLoadLimits::default())
        .await
        .unwrap();
    let mut budget = ProbeExecutionBudget::new(
        tokio::time::Instant::now() + Duration::from_secs(45),
        NonZeroUsize::new(counts.iter().sum::<usize>().checked_mul(2).unwrap()).unwrap(),
        NonZeroUsize::new(counts.iter().sum::<usize>().checked_mul(3).unwrap()).unwrap(),
    );
    for pass in 0..3 {
        for ordinal in 0..counts[pass] {
            let width = 1 + ordinal % 2;
            session.begin_prepared_owner_cohort(pass, ordinal).unwrap();
            let completed = session
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
                .unwrap();
            assert_eq!(completed.completed_requests, width);
            assert_eq!(completed.completed_output_tokens, 3 * width as u64);
            session.end_prepared_owner_cohort().unwrap();
        }
    }
    let audit = session
        .prepared_owner_capture
        .as_ref()
        .unwrap()
        .prepared_audit();
    assert!(!audit.population.poisoned, "{audit:#?}");
    let epoch = session
        .activate_prepared_owner_source()
        .await
        .unwrap_or_else(|error| panic!("original uniform qualification: {error}; {audit:#?}"));
    assert!(epoch > 0);
    let children = runtime.startup_series_children_for_test().unwrap();
    let ordinary = children
        .iter()
        .find(|child| child.numerical_family_key().is_some())
        .expect("actual qualified ordinary family");
    assert_eq!(
        ordinary.algorithm_universe(),
        declare_subset_universe.then_some(&universe)
    );
    assert!(ordinary
        .provenance()
        .phases
        .iter()
        .zip(required_members)
        .all(|(phase, minimum)| phase.members >= minimum));
    session = CalibrationSession::new_driver_session(
        session.engine,
        CalibrationLimits::new(NonZeroUsize::new(2).unwrap()).unwrap(),
    );
    // Positive lookup and the subsequent original terminal planner witness
    // must both succeed against the independently qualified imported child.
    numerical_family::query_and_execute(&mut session, &executor, &template, 2).await;

    // The same installed selector exposes an unobserved CPU implementation for
    // a ragged original input. Neither selector nor cost metadata is changed.
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
            let completed = output.completion.await.unwrap();
            let OutputCompletion::Succeeded { reason, usage, .. } = completed.payload() else {
                panic!("original outside-catalog CPU request failed");
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
    let CalibrationPreparation::Selected(prepared) = inner
        .prepare_calibration_wave(&rows, NonZeroUsize::new(2).unwrap())
        .unwrap()
    else {
        panic!("original ragged CPU projection unavailable");
    };
    let facts = prepared.structured_prepared_facts(&inner, None).unwrap();
    facts.validate().unwrap();
    let domain = runtime.workload_domain().unwrap();
    let query = StructuredQueryV2::from_future_with_domain(
        &facts.exact,
        &facts.selected,
        &facts.recipe,
        &HostContentForecastV2::Exact,
        domain,
    )
    .unwrap();
    assert_eq!(
        universe.contains_checked_algorithms(query.input()),
        Ok(false)
    );
    let snapshot = runtime.snapshot().unwrap();
    if original_automatic_floors {
        let global = DeclaredAlgorithmUniverseV1::from_inputs(
            [qualified_query.input(), query.input()],
            StructuredSettingsV2::default().max_axes,
        )
        .unwrap();
        assert!(global.algorithm_count() > universe.algorithm_count());
        assert!(global.contains_universe(&universe));
        assert!(global.contains_checked_algorithms(query.input()).unwrap());
        let before = runtime.profile_receipt().unwrap();
        // This is the real cold seed installation API. A larger declaration
        // is never converted into a qualified child or a substitute sample.
        runtime.install_startup_algorithm_seed(global).unwrap();
        assert_eq!(runtime.profile_receipt().unwrap(), before);
        assert!(Arc::ptr_eq(&snapshot, &runtime.snapshot().unwrap()));
        snapshot
            .audit_structured_query_v2(&qualified_query, runtime.clock.now_ns().unwrap())
            .unwrap();
    }
    assert_eq!(
        snapshot
            .audit_structured_query_v2(&query, runtime.clock.now_ns().unwrap())
            .unwrap_err(),
        StructuredUnknownV2::WrongDomain
    );
    let unbound = StructuredQueryV2::from_future(
        &facts.exact,
        &facts.selected,
        &facts.recipe,
        &HostContentForecastV2::Exact,
    )
    .unwrap();
    let ExecutorCostIdentityAvailability::Known(identity) = &runtime.identity else {
        panic!("controlled executor identity");
    };
    let mut different_limits = domain.limits().clone();
    different_limits.maximum_context_tokens = NonZeroU32::new(
        different_limits
            .maximum_context_tokens
            .get()
            .checked_add(1)
            .unwrap(),
    )
    .unwrap();
    let different_domain = CostWorkloadDomainV1::new_vnext(identity, different_limits).unwrap();
    let wrong_domain = StructuredQueryV2::from_future_with_domain(
        &facts.exact,
        &facts.selected,
        &facts.recipe,
        &HostContentForecastV2::Exact,
        &different_domain,
    )
    .unwrap();
    for invalid in [&unbound, &wrong_domain] {
        assert_eq!(
            universe.contains_checked_algorithms(invalid.input()),
            Err(StructuredUnknownV2::WrongDomain),
            "missing/wrong physical domain is not checked outside-subset evidence"
        );
        assert_eq!(
            snapshot
                .audit_structured_query_v2(invalid, runtime.clock.now_ns().unwrap())
                .unwrap_err(),
            StructuredUnknownV2::WrongDomain
        );
    }
    // Inspect the supported serialized diagnostic surface from outside the
    // cost-observation module, without exposing its private feedback types.
    let before_audit = serde_json::to_value(runtime.audit_snapshot()).unwrap();
    let before = before_audit.get("structured_feedback").unwrap();
    assert!(before["revoked"].is_null(), "{before:#?}");
    let native_before = executor.native_structured_counts();
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
    let stages = report.host_stages.as_ref().unwrap();
    stages
        .structured_evidence
        .as_ref()
        .unwrap()
        .as_ref()
        .unwrap()
        .validate_host_stages(stages)
        .unwrap();
    let native_after = executor.native_structured_counts();
    assert_eq!(native_after.0, native_before.0 + 1);
    assert_eq!(native_after.1, native_before.1 + 2);
    let after_audit = serde_json::to_value(runtime.audit_snapshot()).unwrap();
    let after = after_audit.get("structured_feedback").unwrap();
    assert_eq!(
        after["outside_catalog_observations"].as_u64().unwrap(),
        before["outside_catalog_observations"].as_u64().unwrap() + 1
    );
    assert_eq!(
        after["uncomparable_observations"].as_u64().unwrap(),
        before["uncomparable_observations"].as_u64().unwrap()
    );
    assert!(after["revoked"].is_null(), "{after:#?}");
    assert!(snapshot
        .audit_structured_query_v2(&qualified_query, runtime.clock.now_ns().unwrap())
        .is_ok());
    assert!(Arc::ptr_eq(&snapshot, &runtime.snapshot().unwrap()));
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
    assert!(snapshot
        .audit_structured_query_v2(&qualified_query, runtime.clock.now_ns().unwrap())
        .is_ok());
    assert!(session.frontiers().unwrap().is_empty());
    if original_automatic_floors {
        // New owners, a fresh future query, independent native replay and
        // the original terminal witness still use the qualified subset.
        numerical_family::query_and_execute(&mut session, &executor, &template, 2).await;
    }
    session.shutdown().await.unwrap();
}
