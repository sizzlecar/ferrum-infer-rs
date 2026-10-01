//! Readiness is an ordinary CPU execution followed by a fresh pure capture.
//! Injected missing-route reasons never supply a Known input or numeric sample.
use super::*;
use std::num::NonZeroUsize;

#[tokio::test]
async fn readiness_cursor_avoids_full_rescans_but_final_capture_rechecks_old_successes() {
    for invalidate_early_width in [false, true] {
        let width = 4;
        let (mut session, executor) = fixture(width).await;
        executor
            .recycle_completed_bindings
            .store(true, Ordering::Release);
        let mut input = inputs(&mut session, false).await;
        let mut cases: Vec<_> = (1..=width).map(case).collect();
        for case in &mut cases {
            case.maximum_output = NonZeroUsize::MIN;
            case.suffix_tokens = 1;
        }
        // One success per width, one failed probe per action, and one final
        // complete pass. Repeating full width scans would require width².
        let projection_allowance = 3 * width - 1;
        assert!(projection_allowance < width * width);
        input.settings.maximum_offered_waves = NonZeroUsize::new(projection_allowance).unwrap();
        let calls = Arc::new(std::sync::Mutex::new(Vec::new()));
        let observed = calls.clone();
        *executor.projection_readiness_fault.lock() = Some(Box::new(move |rows, submitted| {
            observed.lock().unwrap().push((rows, submitted));
            if invalidate_early_width && rows == 1 && submitted == width - 1 {
                Some(ExecutionCostRouteUnknown::Resource(
                    ResourcePlanningUnknown::LogicalCapacity,
                ))
            } else {
                (rows > submitted + 1).then_some(ExecutionCostRouteUnknown::OnDemandResidentProgram)
            }
        }));
        // The original execution ledger reserves a conservative serial wave
        // bound even though each actual readiness cohort is jointly submitted.
        // Keep that independent allowance distinct from the 11 pure queries.
        let declared_readiness_waves: usize = cases[1..]
            .iter()
            .map(|case| {
                case.waves(input.prompts[case.template], input.chunk.get() as usize)
                    .unwrap()
                    .1
            })
            .sum();
        let mut budget = ProbeExecutionBudget::new_with_input_projection_limit(
            Instant::now() + Duration::from_secs(10),
            NonZeroUsize::new((2..=width).sum()).unwrap(),
            NonZeroUsize::new(declared_readiness_waves).unwrap(),
            NonZeroUsize::new(width * (width + 1)).unwrap(),
        );
        let deadline = budget.deadline();
        let inventory = Box::pin(collect_ready(
            &mut session,
            &input,
            &cases,
            &mut budget,
            8 * 1024 * 1024,
        ))
        .await
        .unwrap();
        assert_eq!(executor.physical.load(Ordering::Acquire), width - 1);
        assert_eq!(
            budget.preflight_charge().projection_attempts,
            projection_allowance
        );
        assert_eq!(
            budget.preflight_charge().planning_reserved_requests,
            width * (width + 1)
        );
        assert_eq!(
            budget.preflight_charge().readiness_admitted_requests,
            (2..=width).sum::<usize>()
        );
        assert_eq!(budget.requests_remaining(), 0);
        assert_eq!(
            budget.attempts_remaining(),
            declared_readiness_waves - (width - 1)
        );
        assert_eq!(budget.selection_attempts_remaining(), 0);
        assert_eq!(budget.deadline(), deadline);
        let calls = calls.lock().unwrap();
        assert_eq!(calls.iter().filter(|&&(rows, _)| rows == 1).count(), 2);
        assert!(calls.contains(&(1, 0)) && calls.contains(&(1, width - 1)));
        drop(calls);
        assert_eq!(
            matches!(
                inventory.opportunities[0].population,
                CasePopulation::Unique(_)
            ),
            !invalidate_early_width
        );
        assert!(inventory.opportunities[1..]
            .iter()
            .all(|opportunity| matches!(opportunity.population, CasePopulation::Unique(_))));
        if invalidate_early_width {
            assert!(inventory.gaps.iter().any(|gap| gap.case_index == 0
                && matches!(
                    gap.reason,
                    InventoryGapReason::Projection(GeometryProjectionUnknown::Route(
                        ExecutionCostRouteUnknown::Resource(
                            ResourcePlanningUnknown::LogicalCapacity
                        )
                    ))
                )));
        }
        session.completed_owner_boundary().unwrap();
        session.shutdown().await.unwrap();
    }
}

#[tokio::test]
async fn readiness_failed_complete_inventory_keeps_actual_projection_charge() {
    let (mut session, executor) = fixture(1).await;
    let mut input = inputs(&mut session, false).await;
    input.settings.maximum_offered_waves = NonZeroUsize::MIN;
    let mut budget = budget(1);
    let error = Box::pin(capture(
        &mut session,
        &input,
        &[case(1)],
        &mut budget,
        8 * 1024 * 1024,
    ))
    .await
    .err()
    .expect("original one-call allowance cannot complete decode trajectory");
    assert!(error.to_string().contains("BudgetExhausted"));
    assert_eq!(budget.preflight_charge().projection_attempts, 1);
    assert_eq!(budget.preflight_charge().planning_admitted_requests, 1);
    assert_eq!(budget.preflight_charge().planning_reserved_requests, 1);
    assert_eq!(executor.physical.load(Ordering::Acquire), 0);
    assert_eq!(budget.attempts_remaining(), 16);
    session.completed_owner_boundary().unwrap();
    session.shutdown().await.unwrap();
}

fn case(width: usize) -> Case {
    Case {
        product: OpportunityProduct::Prefill,
        template: 0,
        width,
        maximum_output: NonZeroUsize::new(2).unwrap(),
        release_generated: 0,
        suffix_tokens: 2,
        preset: SloAutomaticCostProbeSamplingPresetV1::Configured,
        prefix: PrefixKind::Ordinary,
        route: CalibrationDecodeRoute::Actual,
        reset: true,
    }
}

#[tokio::test]
async fn readiness_allocator_unsupported_keeps_gap_and_spends_no_model_waves() {
    let (mut session, executor) = fixture(1).await;
    executor
        .recycle_completed_bindings
        .store(true, Ordering::Release);
    let input = inputs(&mut session, true).await;
    *executor.projection_readiness_fault.lock() = Some(Box::new(|_, _| {
        Some(ExecutionCostRouteUnknown::Resource(
            ResourcePlanningUnknown::UnmaterializedCapacity,
        ))
    }));
    let mut budget = budget(3);
    let deadline = budget.deadline();
    let inventory = Box::pin(collect_ready(
        &mut session,
        &input,
        &[case(1)],
        &mut budget,
        8 * 1024 * 1024,
    ))
    .await
    .unwrap();
    assert_eq!(executor.physical.load(Ordering::Acquire), 0);
    assert!(inventory
        .opportunities
        .iter()
        .all(|o| matches!(o.population, CasePopulation::Unknown { .. })));
    assert!(inventory.gaps.iter().any(|g| matches!(
        g.reason,
        InventoryGapReason::Projection(GeometryProjectionUnknown::Route(
            ExecutionCostRouteUnknown::Resource(ResourcePlanningUnknown::UnmaterializedCapacity)
        ))
    )));
    let charge = budget.preflight_charge();
    assert_eq!(charge.planning_reserved_requests, 3);
    assert_eq!(charge.readiness_reserved_requests, 2);
    assert_eq!(charge.readiness_admitted_requests, 0);
    assert_eq!(budget.attempts_remaining(), 16);
    assert_eq!(budget.selection_attempts_remaining(), 16);
    assert_eq!(budget.requests_remaining(), 0);
    assert_eq!(budget.deadline(), deadline);
    session.completed_owner_boundary().unwrap();
    session.shutdown().await.unwrap();
}

#[tokio::test]
async fn readiness_allocator_attempt_never_consumes_missing_program_actions() {
    let (mut session, _) = fixture(1).await;
    let input = inputs(&mut session, true).await;
    let cases = [case(1)];
    let mut attempts = readiness::Attempts::new(3, 4096).unwrap();
    let mut gap = inventory::InventoryGap {
        case_index: 0,
        reason: InventoryGapReason::Projection(GeometryProjectionUnknown::Route(
            ExecutionCostRouteUnknown::Resource(ResourcePlanningUnknown::UnmaterializedCapacity),
        )),
    };
    assert!(matches!(
        readiness::next_action(
            std::slice::from_ref(&gap),
            &cases,
            &cases,
            &input,
            &mut attempts
        )
        .unwrap(),
        Some(readiness::Action::Resources(_))
    ));
    assert!(readiness::next_action(
        std::slice::from_ref(&gap),
        &cases,
        &cases,
        &input,
        &mut attempts
    )
    .unwrap()
    .is_none());
    gap.reason = InventoryGapReason::Projection(GeometryProjectionUnknown::Route(
        ExecutionCostRouteUnknown::OnDemandResidentProgram,
    ));
    for expected in [
        CalibrationDecodeRoute::Actual,
        CalibrationDecodeRoute::FullLogits,
    ] {
        let Some(readiness::Action::Execute(action)) = readiness::next_action(
            std::slice::from_ref(&gap),
            &cases,
            &cases,
            &input,
            &mut attempts,
        )
        .unwrap() else {
            panic!("program preparation still requires real work");
        };
        assert_eq!(action.route, expected);
    }
    session.shutdown().await.unwrap();
}

fn budget(planning_requests: usize) -> ProbeExecutionBudget {
    ProbeExecutionBudget::new_with_input_projection_limit(
        Instant::now() + Duration::from_secs(10),
        NonZeroUsize::new(2).unwrap(),
        NonZeroUsize::new(16).unwrap(),
        NonZeroUsize::new(planning_requests).unwrap(),
    )
}

async fn inputs(session: &mut CalibrationSession, greedy: bool) -> PreparedProbeInputs {
    Arc::get_mut(&mut session.engine.inner).unwrap().tokenizer = Arc::new(tokenizer().await);
    let mut request = InferenceRequest::new("test", session.configuration().model.model_id.clone());
    request.stream = true;
    request.sampling_params.temperature = if greedy { 0.0 } else { 1.0 };
    request.sampling_params.repetition_penalty = 1.0;
    let root = AutomaticCostProbeTemplate::new(request, AutomaticCostProbeOutput::CliText).unwrap();
    let mut settings = SloAutomaticCalibrationSettingsV1::default();
    settings.cost_probe.maximum_output_tokens = NonZeroUsize::new(2).unwrap();
    Box::pin(PreparedProbeInputs::new(session, &settings, &[root]))
        .await
        .unwrap()
}

#[tokio::test]
async fn readiness_fresh_cpu_capture_removes_multiple_obsolete_width_actions() {
    assert_multiple_obsolete_width_actions(false).await;
    assert_multiple_obsolete_width_actions(true).await;
}

async fn assert_multiple_obsolete_width_actions(greedy: bool) {
    let (mut session, executor) = fixture(2).await;
    executor
        .recycle_completed_bindings
        .store(true, Ordering::Release);
    let input = inputs(&mut session, greedy).await;
    let calls = Arc::new(std::sync::Mutex::new(Vec::new()));
    let observed = calls.clone();
    *executor.projection_readiness_fault.lock() = Some(Box::new(move |rows, submitted| {
        observed.lock().unwrap().push((rows, submitted));
        (submitted == 0).then_some(ExecutionCostRouteUnknown::OnDemandResidentProgram)
    }));
    let cases = [case(1), case(2)];
    let mut budget = budget(6);
    let original_deadline = budget.deadline();
    let inventory = Box::pin(collect_ready(
        &mut session,
        &input,
        &cases,
        &mut budget,
        8 * 1024 * 1024,
    ))
    .await
    .unwrap();
    // One real width-one cohort makes both original widths projectable. A
    // loop over the stale gap list would additionally execute width two.
    assert_eq!(executor.physical.load(Ordering::Acquire), 2);
    assert!(inventory
        .opportunities
        .iter()
        .all(|o| matches!(o.population, CasePopulation::Unique(_))));
    let calls = calls.lock().unwrap();
    assert!(calls.contains(&(1, 0)));
    assert!(!calls.contains(&(2, 0)), "stop at the first actionable gap");
    for width in [1, 2] {
        assert!(
            calls.contains(&(width, 2)),
            "fresh checked width {width}: {calls:?}"
        );
    }
    assert!(calls
        .iter()
        .all(|(_, submissions)| *submissions == 0 || *submissions == 2));
    drop(calls);
    let charge = budget.preflight_charge();
    // Initial short probe, fresh continuation, and final full validation.
    assert_eq!(charge.planning_reserved_requests, 6);
    assert_eq!(charge.planning_admitted_requests, 6);
    assert_eq!(charge.readiness_reserved_requests, 1);
    assert_eq!(charge.readiness_admitted_requests, 1);
    assert_eq!(budget.input_projection_requests_remaining(), 0);
    assert_eq!(budget.requests_remaining(), 1);
    assert_eq!(budget.attempts_remaining(), 14);
    assert_eq!(budget.selection_attempts_remaining(), 14);
    assert_eq!(budget.deadline(), original_deadline);
    assert!(session
        .engine
        .inner
        .cost_runtime
        .as_ref()
        .unwrap()
        .snapshot()
        .is_none());
    session.completed_owner_boundary().unwrap();
    session.shutdown().await.unwrap();
}

#[tokio::test]
async fn readiness_recapture_cannot_renew_original_planning_request_budget() {
    let (mut session, executor) = fixture(2).await;
    executor
        .recycle_completed_bindings
        .store(true, Ordering::Release);
    let input = inputs(&mut session, false).await;
    *executor.projection_readiness_fault.lock() = Some(Box::new(|_, submitted| {
        (submitted == 0).then_some(ExecutionCostRouteUnknown::OnDemandResidentProgram)
    }));
    let mut budget = budget(2);
    let original_deadline = budget.deadline();
    assert!(Box::pin(collect_ready(
        &mut session,
        &input,
        &[case(1), case(2)],
        &mut budget,
        8 * 1024 * 1024
    ))
    .await
    .is_err());
    // The cohort settled, but fresh capture cannot be paid for. It does not
    // carry forward stale Known facts or reopen either request allowance.
    assert_eq!(executor.physical.load(Ordering::Acquire), 2);
    let charge = budget.preflight_charge();
    assert_eq!(charge.planning_reserved_requests, 2);
    assert_eq!(charge.planning_admitted_requests, 2);
    assert_eq!(charge.readiness_admitted_requests, 1);
    assert_eq!(budget.input_projection_requests_remaining(), 0);
    assert_eq!(budget.requests_remaining(), 1);
    assert_eq!(budget.attempts_remaining(), 14);
    assert_eq!(budget.deadline(), original_deadline);
    session.completed_owner_boundary().unwrap();
    session.shutdown().await.unwrap();
}

#[tokio::test]
async fn readiness_permanent_execution_policy_gap_executes_no_cpu_cohort() {
    let (mut session, executor) = fixture(2).await;
    let input = inputs(&mut session, false).await;
    *executor.projection_readiness_fault.lock() = Some(Box::new(|_, _| {
        Some(ExecutionCostRouteUnknown::ExecutionPolicy)
    }));
    let mut budget = budget(4);
    let inventory = Box::pin(collect_ready(
        &mut session,
        &input,
        &[case(1), case(2)],
        &mut budget,
        8 * 1024 * 1024,
    ))
    .await
    .unwrap();
    assert_eq!(executor.physical.load(Ordering::Acquire), 0);
    assert_eq!(budget.preflight_charge().readiness_reserved_requests, 0);
    assert_eq!(budget.preflight_charge().planning_reserved_requests, 4);
    assert!(inventory
        .opportunities
        .iter()
        .all(|o| matches!(o.population, CasePopulation::Unknown { .. })));
    assert!(!inventory.gaps.is_empty());
    assert!(inventory
        .gaps
        .iter()
        .all(|gap| !readiness_missing(&gap.reason)));
    session.completed_owner_boundary().unwrap();
    session.shutdown().await.unwrap();
}

#[tokio::test]
async fn readiness_actual_action_does_not_close_a_remaining_full_logits_gap() {
    let (mut session, executor) = fixture(1).await;
    executor
        .recycle_completed_bindings
        .store(true, Ordering::Release);
    let input = inputs(&mut session, true).await;
    let cases = [case(1)];
    let gap = inventory::InventoryGap {
        case_index: 0,
        reason: InventoryGapReason::Projection(GeometryProjectionUnknown::Route(
            ExecutionCostRouteUnknown::OnDemandResidentProgram,
        )),
    };
    let mut attempts = readiness::Attempts::new(2, 4096).unwrap();
    let mut budget = budget(2);
    for expected in [
        CalibrationDecodeRoute::Actual,
        CalibrationDecodeRoute::FullLogits,
    ] {
        let action = readiness::next_action(
            std::slice::from_ref(&gap),
            &cases,
            &cases,
            &input,
            &mut attempts,
        )
        .unwrap()
        .expect("remaining missing route requires its original action");
        let readiness::Action::Execute(action) = action else {
            panic!("resident program requires real execution");
        };
        assert_eq!(action.route, expected);
        let (requests, settings) =
            inventory::readiness_requests(&action, &input.templates, input.chunk).unwrap();
        let summary = session
            .run_readiness_probe_cohort(requests, settings, &mut budget)
            .await
            .unwrap();
        assert_eq!(summary.completed_output_tokens, 2);
    }
    // Exhausting both action identities still leaves the original gap Unknown.
    assert!(readiness::next_action(
        std::slice::from_ref(&gap),
        &cases,
        &cases,
        &input,
        &mut attempts
    )
    .unwrap()
    .is_none());
    assert!(readiness_missing(&gap.reason));
    assert_eq!(executor.physical.load(Ordering::Acquire), 4);
    session.shutdown().await.unwrap();
}

#[test]
fn readiness_only_retries_typed_materialization_gaps() {
    for (reason, recoverable) in [
        (ExecutionCostRouteUnknown::OnDemandResidentProgram, true),
        (
            ExecutionCostRouteUnknown::Resource(ResourcePlanningUnknown::UnmaterializedCapacity),
            true,
        ),
        (ExecutionCostRouteUnknown::ExecutionPolicy, false),
        (ExecutionCostRouteUnknown::InitializationState, false),
        (
            ExecutionCostRouteUnknown::Resource(ResourcePlanningUnknown::LogicalCapacity),
            false,
        ),
        (
            ExecutionCostRouteUnknown::Resource(ResourcePlanningUnknown::PhysicalCapacity),
            false,
        ),
        (
            ExecutionCostRouteUnknown::Resource(ResourcePlanningUnknown::ReusableExecution),
            false,
        ),
        (
            ExecutionCostRouteUnknown::Resource(ResourcePlanningUnknown::MaintenanceRequired),
            false,
        ),
    ] {
        assert_eq!(
            readiness_missing(&InventoryGapReason::Projection(
                GeometryProjectionUnknown::Route(reason)
            )),
            recoverable,
            "{reason:?}"
        );
    }
}

#[tokio::test]
async fn readiness_remaining_gap_is_recaptured_after_each_original_cpu_route() {
    for missing_after_both in [false, true] {
        let (mut session, executor) = fixture(1).await;
        executor
            .recycle_completed_bindings
            .store(true, Ordering::Release);
        let input = inputs(&mut session, true).await;
        let calls = Arc::new(std::sync::Mutex::new(Vec::new()));
        let observed = calls.clone();
        *executor.projection_readiness_fault.lock() = Some(Box::new(move |_, submitted| {
            observed.lock().unwrap().push(submitted);
            (submitted < 4 || missing_after_both)
                .then_some(ExecutionCostRouteUnknown::OnDemandResidentProgram)
        }));
        let mut budget = budget(4);
        let inventory = Box::pin(collect_ready(
            &mut session,
            &input,
            &[case(1)],
            &mut budget,
            8 * 1024 * 1024,
        ))
        .await
        .unwrap();
        assert_eq!(executor.physical.load(Ordering::Acquire), 4);
        let calls = calls.lock().unwrap();
        for settled in [0, 2, 4] {
            assert!(
                calls.contains(&settled),
                "capture after each settled action: {calls:?}"
            );
        }
        drop(calls);
        assert_eq!(budget.preflight_charge().planning_reserved_requests, 4);
        assert_eq!(budget.preflight_charge().readiness_admitted_requests, 2);
        assert_eq!(budget.input_projection_requests_remaining(), 0);
        assert_eq!(budget.requests_remaining(), 0);
        assert_eq!(budget.attempts_remaining(), 12);
        assert_eq!(budget.selection_attempts_remaining(), 12);
        assert_eq!(
            matches!(
                inventory.opportunities[0].population,
                CasePopulation::Unknown { .. }
            ),
            missing_after_both
        );
        if missing_after_both {
            assert!(inventory
                .gaps
                .iter()
                .any(|gap| readiness_missing(&gap.reason)));
        } else {
            assert!(matches!(
                inventory.opportunities[0].population,
                CasePopulation::Unique(_)
            ));
        }
        session.completed_owner_boundary().unwrap();
        session.shutdown().await.unwrap();
    }
}
