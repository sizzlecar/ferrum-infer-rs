//! A complete original source qualifies only the decoder branches it saw.
//! Fresh pending work stays Unknown until an independent complete extension.
//! Further fresh widths and contexts share the same real engine and catalog.
use super::*;

#[tokio::test]
async fn source8_feedback_unchallenged_pending_work_preserves_qualified_catalog() {
    let (mut session, executor) = automatic_session().await;
    executor
        .native_structured_submission
        .store(true, Ordering::Release);
    executor
        .project_structured_cpu_fill
        .store(true, Ordering::Release);
    executor.enable_structured_query_route();
    let runtime = session.engine.inner.cost_runtime.as_ref().unwrap().clone();
    let deadline = tokio::time::Instant::now() + Duration::from_secs(45);
    session.begin_startup_owner_series(3, deadline).unwrap();
    let mut clean = full_population(&session);
    for prefix in clean.prefix_plan.phases.iter_mut().flatten().flatten() {
        for slot in &mut prefix.slots {
            slot.token_ids = vec![TokenId::new(10)];
            slot.token_bytes = vec![b"a".to_vec()];
        }
    }
    clean.cohort_manifest_payload = serde_json::value::to_raw_value(&serde_json::json!({
        "prompt":"test", "maximum_output":3, "prefix":"actual-bytelevel-a",
        "outputs":["cli_text","completions_sse"], "sampling":"installed_full_vocabulary_top_k"
    }))
    .unwrap();
    session
        .begin_prepared_owner_source(clean, CostProfileLoadLimits::default())
        .await
        .unwrap();
    let options = ProbeCohortSettings {
        prefill_plan:
            crate::continuous_engine::inner::calibration::cohort_driver::ProbePrefillPlan::Joint,
        prefill_chunk: NonZeroU32::MIN,
        decode_route: CalibrationDecodeRoute::FullLogits,
        reset_token_policy: false,
    };
    let mut budget = ProbeExecutionBudget::new(
        deadline,
        NonZeroUsize::new(64).unwrap(),
        NonZeroUsize::new(96).unwrap(),
    );
    for pass in 0..3 {
        for ordinal in 0..[16, 8, 8][pass] {
            session.begin_prepared_owner_cohort(pass, ordinal).unwrap();
            let probes = probe_requests(&session);
            session
                .run_probe_cohort(probes, options, &mut budget)
                .await
                .unwrap();
            session.end_prepared_owner_cohort().unwrap();
        }
    }
    let epoch = session.activate_prepared_owner_source().await.unwrap();
    let original = runtime.snapshot().unwrap();
    let receipt = runtime.profile_receipt().unwrap();
    assert_eq!(runtime.catalog_live_state(), Some((epoch, true)));
    session.retire_startup_owner_source().await.unwrap();

    let mut pending = full_population(&session);
    for slot in &mut pending.prefix_plan.phases[0][0].as_mut().unwrap().slots {
        slot.token_ids = vec![TokenId::new(11)];
        slot.token_bytes = vec![vec![0xc3]];
    }
    pending.cohort_manifest_payload = serde_json::value::to_raw_value(&serde_json::json!({
        "prompt":"test", "maximum_output":3, "first_prefix":"actual-bytelevel-C3",
        "outputs":["cli_text","completions_sse"], "sampling":"installed_full_vocabulary_top_k"
    }))
    .unwrap();
    session
        .begin_prepared_owner_source(pending, CostProfileLoadLimits::default())
        .await
        .unwrap();
    session.begin_prepared_owner_cohort(0, 0).unwrap();
    let mut ids = Vec::new();
    let mut consumers = Vec::new();
    for probe in probe_requests(&session) {
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
                panic!("real pending owner did not complete");
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
    let rows = ids
        .iter()
        .map(|id| {
            frontier(&session, id)
                .prefill_work(NonZeroU32::MIN)
                .unwrap()
        })
        .collect();
    let preparation = wave(&mut session, &executor, rows).await;
    assert!(preparation.error.is_none(), "{preparation:?}");
    let fifo = preparation
        .host_stage_queue
        .unwrap()
        .accepted_ordinal
        .unwrap();
    release(&mut session, &ids, fifo).await;
    session.freeze_cost_model().await.unwrap();
    let before_audit = serde_json::to_value(runtime.audit_snapshot()).unwrap();
    let before = &before_audit["structured_feedback"];
    assert!(before["revoked"].is_null(), "{before:#}");

    let rows = ids
        .iter()
        .map(|id| {
            frontier(&session, id)
                .decode_work_with_route(CalibrationDecodeRoute::FullLogits)
                .unwrap()
        })
        .collect::<Vec<_>>();
    let submitted_before = executor.native_structured_counts();
    let report = wave(&mut session, &executor, rows).await;
    assert!(report.error.is_none(), "{report:?}");
    assert_eq!(
        report.submission,
        CalibrationSubmissionState::HostReconciled
    );
    assert_eq!(
        executor.native_structured_counts(),
        (submitted_before.0 + 1, submitted_before.1 + 2)
    );
    let stages = report.host_stages.as_ref().unwrap();
    let settled = stages
        .structured_evidence
        .as_ref()
        .unwrap()
        .as_ref()
        .unwrap();
    settled.validate_host_stages(stages).unwrap();
    // The worker publishes the original scoped actual input only after the
    // same recorder/host/recipe identity has passed settlement validation.
    // No second prepared publication is created for this lookup.
    session.freeze_cost_model().await.unwrap();
    let pending_query = StructuredQueryV2::exact(report.structured_cost_input_v2().unwrap());
    assert_eq!(
        original
            .audit_structured_query_v2(&pending_query, runtime.clock.now_ns().unwrap())
            .unwrap_err(),
        StructuredUnknownV2::QualificationCoverage
    );
    session.freeze_cost_model().await.unwrap();
    let after_audit = serde_json::to_value(runtime.audit_snapshot()).unwrap();
    let after = &after_audit["structured_feedback"];
    assert_eq!(
        after["outside_support_observations"].as_u64().unwrap(),
        before["outside_support_observations"].as_u64().unwrap() + 1
    );
    assert_eq!(
        after["uncomparable_observations"],
        before["uncomparable_observations"]
    );
    assert_eq!(after["compared"], before["compared"]);
    assert_eq!(after["corrections"], before["corrections"]);
    assert!(after["revoked"].is_null(), "{after:#}");
    assert!(Arc::ptr_eq(&original, &runtime.snapshot().unwrap()));
    assert_eq!(runtime.profile_receipt().unwrap(), receipt);

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
    let terminal = wave(&mut session, &executor, rows).await;
    assert!(terminal.error.is_none(), "{terminal:?}");
    session.freeze_cost_model().await.unwrap();
    let supported = StructuredQueryV2::exact(terminal.structured_cost_input_v2().unwrap());
    original
        .audit_structured_query_v2(&supported, runtime.clock.now_ns().unwrap())
        .unwrap();
    for consumer in consumers {
        bounded(consumer).await.unwrap();
    }
    session.end_prepared_owner_cohort().unwrap();
    // Complete every remaining independently declared cohort. The pending
    // observation above is an original Fit member, never copied into another
    // phase, and cannot qualify or extend the old catalog by itself.
    let mut remaining = ProbeExecutionBudget::new(
        deadline,
        NonZeroUsize::new(2 * (16 + 8 + 8 - 1)).unwrap(),
        NonZeroUsize::new(3 * (16 + 8 + 8 - 1)).unwrap(),
    );
    for pass in 0..3 {
        for ordinal in 0..[16, 8, 8][pass] {
            if (pass, ordinal) == (0, 0) {
                continue;
            }
            session.begin_prepared_owner_cohort(pass, ordinal).unwrap();
            let probes = probe_requests(&session);
            session
                .run_probe_cohort(probes, options, &mut remaining)
                .await
                .unwrap();
            session.end_prepared_owner_cohort().unwrap();
        }
    }
    assert_eq!(remaining.requests_remaining(), 0);
    assert_eq!(remaining.attempts_remaining(), 0);
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
    let expanded_epoch = session
        .activate_prepared_owner_source()
        .await
        .unwrap_or_else(|error| panic!("complete extension did not qualify: {error}; {audit:#?}"));
    assert!(expanded_epoch > epoch);
    let expanded = runtime.snapshot().unwrap();
    let expanded_receipt = runtime.profile_receipt().unwrap();
    assert!(!Arc::ptr_eq(&original, &expanded));
    assert_eq!(runtime.catalog_live_state(), Some((expanded_epoch, true)));
    // Replacement invalidates the old epoch view. Only the real second
    // source's independent Fit/Residual/Qualification publication can make
    // the pending input queryable through the current catalog.
    assert_eq!(
        original
            .audit_structured_query_v2(&pending_query, runtime.clock.now_ns().unwrap())
            .unwrap_err(),
        StructuredUnknownV2::RuntimeValidity
    );
    for query in [&pending_query, &supported] {
        expanded
            .audit_structured_query_v2(query, runtime.clock.now_ns().unwrap())
            .unwrap();
    }
    session.retire_startup_owner_source().await.unwrap();

    // A third original source supplies fresh owner identities. Its incomplete
    // population cannot publish: queries still use exactly expanded_epoch.
    // Probe a new width, then a larger batch and context, the newly learned
    // pending branch, and finally the original covered clean input.
    // This is a real CPU request-authority boundary, not hardware long-context
    // coverage. The fixture pre-admitted its native sessions independently of
    // MockModelExecutor's broader capability and the numerical domain D.
    // Freeze the finite long-history cases from that actual admission ceiling.
    let long_output = usize::try_from(executor.native_request_fit_tokens().unwrap()).unwrap();
    assert!(
        long_output > 3,
        "the new history must exceed the trained output length"
    );
    assert!(long_output <= session.context_capacity());
    assert!(
        long_output as u64
            <= u64::from(
                runtime
                    .workload_domain()
                    .unwrap()
                    .limits()
                    .maximum_context_tokens
                    .get()
            )
    );
    let mut fresh = full_population(&session);
    let geometry = [
        (1, long_output, false),
        (2, long_output, false),
        (2, 3, true),
        (2, 3, false),
    ];
    fresh.maximum_offered_waves += geometry
        .iter()
        .map(|&(_, maximum_output, _)| maximum_output - 3)
        .sum::<usize>();
    for (ordinal, &(width, maximum_output, pending)) in geometry.iter().enumerate() {
        fresh.cohort_plan.phases[0][ordinal]
            .requests
            .truncate(width);
        for request in &mut fresh.cohort_plan.phases[0][ordinal].requests {
            request.manifest_prompt = ordinal as u32;
            request.maximum_output = maximum_output as u64;
        }
        let prefix = fresh.prefix_plan.phases[0][ordinal].as_mut().unwrap();
        prefix.slots.truncate(width);
        for slot in &mut prefix.slots {
            slot.token_ids = vec![TokenId::new(if pending { 11 } else { 10 })];
            slot.token_bytes = vec![if pending { vec![0xc3] } else { b"a".to_vec() }];
        }
    }
    fresh.cohort_manifest_payload = serde_json::value::to_raw_value(&serde_json::json!({
        "fresh_cases": [
            {"prompt_tokens":1,"maximum_output":long_output,"width":1,"prefix":"a"},
            {"prompt_tokens":1,"maximum_output":long_output,"width":2,"prefix":"a"},
            {"prompt_tokens":1,"maximum_output":3,"width":2,"prefix":"C3"},
            {"prompt_tokens":1,"maximum_output":3,"width":2,"prefix":"a"}
        ],
        "remaining_cases":"original full_population",
        "remaining_maximum_output":3,
        "outputs":["cli_text","completions_sse"],
        "sampling":"installed_full_vocabulary_top_k"
    }))
    .unwrap();
    session
        .begin_prepared_owner_source(fresh, CostProfileLoadLimits::default())
        .await
        .unwrap();
    for (ordinal, &(width, maximum_output, pending)) in geometry.iter().enumerate() {
        session.begin_prepared_owner_cohort(0, ordinal).unwrap();
        let queries =
            fresh_actual_cohort(&mut session, &executor, width, maximum_output, pending).await;
        assert_eq!(queries.len(), maximum_output - 1);
        for query in &queries {
            let result = expanded.audit_structured_query_v2(query, runtime.clock.now_ns().unwrap());
            if width == 1 {
                // The original exact B2 population grants no B1 authority.
                assert_eq!(result.unwrap_err(), StructuredUnknownV2::WrongDomain);
            } else {
                result.unwrap_or_else(|reason| {
                    panic!("fresh B{width}, output/history {maximum_output}, pending {pending}: {reason:?}")
                });
            }
        }
        session.end_prepared_owner_cohort().unwrap();
        assert_eq!(runtime.catalog_live_state(), Some((expanded_epoch, true)));
        assert!(Arc::ptr_eq(&expanded, &runtime.snapshot().unwrap()));
        assert_eq!(runtime.profile_receipt().unwrap(), expanded_receipt);
    }
    session.finish_startup_owner_series().unwrap();
    session.freeze_cost_model().await.unwrap();
    assert_eq!(runtime.catalog_live_state(), Some((expanded_epoch, true)));
    let final_audit = serde_json::to_value(runtime.audit_snapshot()).unwrap();
    assert!(
        final_audit["structured_feedback"]["revoked"].is_null(),
        "{final_audit:#}"
    );
    runtime.shutdown().await.unwrap();
}

/// Execute the original declared cohort once. Queries are built only from
/// worker-validated actual reports; inspecting them acquires no publication.
async fn fresh_actual_cohort(
    session: &mut CalibrationSession,
    executor: &Arc<ControlledExecutor>,
    width: usize,
    maximum_output: usize,
    pending: bool,
) -> Vec<StructuredQueryV2> {
    let mut ids = Vec::new();
    let mut consumers = Vec::new();
    for mut probe in probe_requests(session).into_iter().take(width) {
        probe.request.sampling_params.max_tokens = maximum_output;
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
                panic!("fresh lifecycle owner failed");
            };
            assert_eq!(*reason, ferrum_types::FinishReason::Length);
            assert_eq!(usage.completion_tokens, maximum_output);
        }));
    }
    for id in &ids {
        ready(session, id, false).await;
    }
    for _ in &ids {
        admit(session).await;
    }
    // One whole one-token prefill uses the actual native fixture contract.
    // Long history is built by subsequent real decode submissions, with the
    // original domain's two-token whole-wave bound unchanged.
    let rows = ids
        .iter()
        .map(|id| {
            let f = frontier(session, id);
            assert_eq!(f.prefill_progress(), Some((0, 1)));
            f.prefill_work(NonZeroU32::MIN).unwrap()
        })
        .collect();
    let preparation = wave(session, executor, rows).await;
    assert!(preparation.error.is_none(), "{preparation:?}");
    for id in &ids {
        ready(session, id, false).await;
    }
    let PrefixReleaseProgressV5::Released { receipts } =
        session.advance_prepared_owner_prefix_release().unwrap()
    else {
        panic!("fresh prefix was not acknowledged by original actors");
    };
    assert_eq!(receipts.len(), width);
    for receipt in receipts {
        assert_eq!(receipt.frontier.kv_tokens, 1);
        assert_eq!(receipt.frontier.generated_tokens, 1);
        assert_eq!(
            receipt.frontier.pending_utf8.as_slice(),
            if pending { &[0xc3][..] } else { &[][..] }
        );
    }
    let mut queries = Vec::new();
    for generated_before in 1..maximum_output {
        for id in &ids {
            ready(session, id, false).await;
        }
        let rows = ids
            .iter()
            .map(|id| {
                let f = frontier(session, id);
                assert_eq!(f.generated_tokens(), generated_before);
                assert_eq!(f.kv_tokens(), generated_before);
                f.decode_work_with_route(CalibrationDecodeRoute::FullLogits)
                    .unwrap()
            })
            .collect();
        let before = executor.native_structured_counts();
        let report = wave(session, executor, rows).await;
        assert!(report.error.is_none(), "{report:?}");
        assert_eq!(
            report.submission,
            CalibrationSubmissionState::HostReconciled
        );
        assert_eq!(
            executor.native_structured_counts(),
            (before.0 + 1, before.1 + 2)
        );
        session.freeze_cost_model().await.unwrap();
        queries.push(StructuredQueryV2::exact(
            report.structured_cost_input_v2().unwrap(),
        ));
    }
    for consumer in consumers {
        bounded(consumer).await.unwrap();
    }
    assert!(session.frontiers().unwrap().is_empty());
    queries
}
