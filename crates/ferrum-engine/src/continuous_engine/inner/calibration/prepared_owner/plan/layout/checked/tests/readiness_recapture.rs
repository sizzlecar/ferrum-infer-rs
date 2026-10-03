//! Readiness actions use ordinary CPU execution followed by a fresh capture.
//! An uninterrupted initial traversal may complete the checked inventory;
//! injected missing-route reasons never supply Known inputs or numeric samples.
use super::*;
use std::num::NonZeroUsize;

#[tokio::test]
async fn readiness_uninterrupted_capture_matches_final_inventory_with_one_projection_allowance() {
    for permanent_gap in [false, true] {
        let width = 2;
        let (mut session, executor) = fixture(width).await;
        executor
            .recycle_completed_bindings
            .store(true, Ordering::Release);
        let mut input = inputs(&mut session, false).await;
        let cases: Vec<_> = (1..=width).map(case).collect();
        let calls = Arc::new(std::sync::Mutex::new(Vec::new()));
        let observed = calls.clone();
        *executor.projection_readiness_fault.lock() = Some(Box::new(move |rows, submitted| {
            observed.lock().unwrap().push((rows, submitted));
            (permanent_gap && rows == width).then_some(ExecutionCostRouteUnknown::Resource(
                ResourcePlanningUnknown::LogicalCapacity,
            ))
        }));
        // Use the existing terminal capture as the reference, including the
        // ordinary decode trajectories and any permanently unavailable row.
        let mut reference_budget = budget(width);
        let reference = Box::pin(capture(
            &mut session,
            &input,
            &cases,
            &mut reference_budget,
            8 * 1024 * 1024,
        ))
        .await
        .unwrap();
        let reference_calls = std::mem::take(&mut *calls.lock().unwrap());
        let projections = reference_budget.preflight_charge().projection_attempts;
        assert_eq!(reference_calls.len(), projections);
        assert!(projections > 1 && !reference.algorithm_inputs.is_empty());
        assert_eq!(reference.gaps.is_empty(), !permanent_gap);
        session.completed_owner_boundary().unwrap();

        // A second full traversal cannot fit this same allowance. There is
        // room for another owner admission, so failure would expose repeated
        // projections rather than an unrelated request-limit rejection.
        input.settings.maximum_offered_waves = NonZeroUsize::new(projections).unwrap();
        let mut actual_budget = budget(2 * width);
        let deadline = actual_budget.deadline();
        let actual = Box::pin(collect_ready(
            &mut session,
            &input,
            &cases,
            &mut actual_budget,
            8 * 1024 * 1024,
        ))
        .await
        .unwrap();
        assert_eq!(*calls.lock().unwrap(), reference_calls);
        assert_eq!(actual.inputs, reference.inputs);
        assert_eq!(actual.algorithm_inputs, reference.algorithm_inputs);
        assert_eq!(actual.opportunities.len(), reference.opportunities.len());
        for (actual, reference) in actual.opportunities.iter().zip(&reference.opportunities) {
            assert_eq!(actual.population, reference.population);
            assert_eq!(
                actual.minimum_fresh_members,
                reference.minimum_fresh_members
            );
        }
        assert_eq!(actual.gaps.len(), reference.gaps.len());
        for (actual, reference) in actual.gaps.iter().zip(&reference.gaps) {
            assert_eq!(actual.case_index, reference.case_index);
            let (InventoryGapReason::Projection(actual), InventoryGapReason::Projection(reference)) =
                (&actual.reason, &reference.reason)
            else {
                panic!("the unchanged unavailable row must retain its projection gap");
            };
            assert_eq!(actual, reference);
        }
        assert_eq!(actual.charge, actual_budget.preflight_charge());
        assert_eq!(actual.charge.projection_attempts, projections);
        assert_eq!(actual.charge.planning_admitted_requests, width);
        assert_eq!(actual.charge.planning_reserved_requests, width);
        assert_eq!(actual.charge.readiness_admitted_requests, 0);
        assert_eq!(actual_budget.input_projection_requests_remaining(), width);
        assert_eq!(actual_budget.requests_remaining(), 2);
        assert_eq!(actual_budget.attempts_remaining(), 16);
        assert_eq!(actual_budget.deadline(), deadline);
        assert_eq!(executor.physical.load(Ordering::Acquire), 0);
        session.completed_owner_boundary().unwrap();

        calls.lock().unwrap().clear();
        input.settings.maximum_offered_waves = NonZeroUsize::new(projections - 1).unwrap();
        let mut bounded_budget = budget(2 * width);
        let error = Box::pin(collect_ready(
            &mut session,
            &input,
            &cases,
            &mut bounded_budget,
            8 * 1024 * 1024,
        ))
        .await
        .err()
        .expect("one fewer real projection cannot complete the original inventory");
        assert!(error.to_string().contains("BudgetExhausted"));
        assert_eq!(calls.lock().unwrap().len(), projections - 1);
        assert_eq!(
            bounded_budget.preflight_charge().projection_attempts,
            projections - 1
        );
        assert_eq!(
            bounded_budget.preflight_charge().planning_admitted_requests,
            width
        );
        assert_eq!(
            bounded_budget.preflight_charge().planning_reserved_requests,
            width
        );
        assert_eq!(
            bounded_budget
                .preflight_charge()
                .readiness_admitted_requests,
            0
        );
        assert_eq!(executor.physical.load(Ordering::Acquire), 0);
        session.completed_owner_boundary().unwrap();
        session.shutdown().await.unwrap();
    }
}

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
        acquisition: None,
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
    // The new traversal from zero retains the still-unavailable result after
    // the sole resource action, without admitting a redundant third owner.
    assert_eq!(charge.planning_reserved_requests, 2);
    assert_eq!(charge.planning_admitted_requests, 2);
    assert_eq!(budget.input_projection_requests_remaining(), 1);
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
    let mut input = inputs(&mut session, greedy).await;
    let cases = [case(1), case(2)];
    let owner_width = cases.iter().map(|case| case.width).max().unwrap();
    let mut reference_budget = budget(owner_width);
    let reference = Box::pin(capture(
        &mut session,
        &input,
        &cases,
        &mut reference_budget,
        8 * 1024 * 1024,
    ))
    .await
    .unwrap();
    let full_projection_work = reference_budget.preflight_charge().projection_attempts;
    assert!(full_projection_work > 1);
    session.completed_owner_boundary().unwrap();
    // One failed first query plus one fresh full traversal. Repeating that
    // full traversal after preparation cannot fit this original allowance.
    let projection_allowance = 1 + full_projection_work;
    input.settings.maximum_offered_waves = NonZeroUsize::new(projection_allowance).unwrap();
    let calls = Arc::new(std::sync::Mutex::new(Vec::new()));
    let observed = calls.clone();
    *executor.projection_readiness_fault.lock() = Some(Box::new(move |rows, submitted| {
        observed.lock().unwrap().push((rows, submitted));
        (submitted == 0).then_some(ExecutionCostRouteUnknown::OnDemandResidentProgram)
    }));
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
    assert_eq!(inventory.inputs, reference.inputs);
    assert_eq!(inventory.algorithm_inputs, reference.algorithm_inputs);
    assert_eq!(
        inventory.algorithm_case_inputs,
        reference.algorithm_case_inputs
    );
    for (actual, reference) in inventory.opportunities.iter().zip(&reference.opportunities) {
        assert_eq!(actual.population, reference.population);
        assert_eq!(
            actual.minimum_fresh_members,
            reference.minimum_fresh_members
        );
    }
    let calls = calls.lock().unwrap();
    assert_eq!(calls.len(), projection_allowance);
    assert_eq!(calls.first(), Some(&(1, 0)));
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
    // The first failed query and real action stay charged. Only the duplicate
    // final traversal is removed: 1 + 2*full becomes 1 + full, with two fresh
    // owner groups rather than three; no old input authority is carried over.
    assert_eq!(charge.projection_attempts, projection_allowance);
    assert_eq!(charge.planning_reserved_requests, 2 * owner_width);
    assert_eq!(charge.planning_admitted_requests, 2 * owner_width);
    assert_eq!(charge.readiness_reserved_requests, 1);
    assert_eq!(charge.readiness_admitted_requests, 1);
    assert_eq!(budget.input_projection_requests_remaining(), owner_width);
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
    let cases = [case(1), case(2)];
    let owner_width = cases.iter().map(|case| case.width).max().unwrap();
    *executor.projection_readiness_fault.lock() = Some(Box::new(|_, _| {
        Some(ExecutionCostRouteUnknown::ExecutionPolicy)
    }));
    let mut budget = budget(4);
    let inventory = Box::pin(collect_ready(
        &mut session,
        &input,
        &cases,
        &mut budget,
        8 * 1024 * 1024,
    ))
    .await
    .unwrap();
    assert_eq!(executor.physical.load(Ordering::Acquire), 0);
    assert_eq!(budget.preflight_charge().readiness_reserved_requests, 0);
    // No preparation can repair ExecutionPolicy. The initial complete
    // traversal retains its gaps and admits the maximum-width owner group once.
    assert_eq!(
        budget.preflight_charge().planning_reserved_requests,
        owner_width
    );
    assert_eq!(
        budget.preflight_charge().planning_admitted_requests,
        owner_width
    );
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
        // Both actual routes are settled before the third, fresh traversal
        // from zero. Its complete Known/Unknown report is the final inventory.
        assert_eq!(budget.preflight_charge().planning_reserved_requests, 3);
        assert_eq!(budget.preflight_charge().planning_admitted_requests, 3);
        assert_eq!(budget.preflight_charge().readiness_admitted_requests, 2);
        assert_eq!(budget.input_projection_requests_remaining(), 1);
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
