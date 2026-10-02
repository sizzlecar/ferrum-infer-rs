//! Real resource-root projection of finite scheduler span declarations.
//! These are input opportunities, never empirical qualification samples.
use super::*;
use ferrum_interfaces::execution_cost::HostTerminalExpectationV1;

#[tokio::test]
async fn checked_scheduler_prefill_candidates_project_first_middle_and_final_spans() {
    let (mut session, executor) = fixture_with_domain(1, 32, 8).await;
    executor
        .recycle_completed_bindings
        .store(true, Ordering::Release);
    Arc::get_mut(&mut session.engine.inner).unwrap().tokenizer = Arc::new(tokenizer().await);
    let model = session.configuration().model.model_id.clone();
    let root = template("test", &model)
        .unwrap()
        .with_prompt_renderer(Arc::new(PlainRenderer { model }));
    let settings = SloAutomaticCalibrationSettingsV1::default();
    let inputs = Box::pin(PreparedProbeInputs::new(&mut session, &settings, &[root]))
        .await
        .unwrap();
    let (all, _, _, _) = prepare_cases(&inputs).unwrap();
    let mut budget = ProbeExecutionBudget::new_with_input_projection_limit(
        Instant::now() + Duration::from_millis(settings.cost_probe.maximum_duration_ms.get()),
        settings.cost_probe.maximum_probe_requests,
        settings.cost_probe.maximum_offered_waves,
        settings.cost_probe.maximum_input_projection_requests,
    );
    let original_deadline = budget.deadline();
    let before = executor.native_structured_counts();
    // fixture_with_domain declares numerical limits, not native admission
    // authority. Keep the original real request capacity and test both sides.
    let native_fit = usize::try_from(executor.native_request_fit_tokens().unwrap()).unwrap();
    let mut complete_windows = 0;
    let mut capacity_rejections = 0;
    for chunk in [2, 4] {
        let chunk = NonZeroU32::new(chunk).unwrap();
        assert!(inputs.prefill_candidate_chunks.contains(&chunk));
        let cases: Vec<_> = all
            .iter()
            .filter(|case| {
                case.explicit_prefill_chunk() == Some(chunk) && case.maximum_output.get() == 1
            })
            .cloned()
            .collect();
        assert_eq!(
            cases.len(),
            3,
            "one first, middle and final window from the real renderer"
        );
        let report = capture(
            &mut session,
            &inputs,
            &cases,
            &mut budget,
            inputs.population.maximum_retained_numeric_bytes,
        )
        .await
        .unwrap();
        let expected_rejections = cases
            .iter()
            .filter(|case| {
                let OpportunityProduct::PrefillSpan { offset, .. } = case.product else {
                    unreachable!()
                };
                offset
                    .checked_add(usize::try_from(chunk.get()).unwrap())
                    .unwrap()
                    .min(inputs.prompts[case.template])
                    > native_fit
            })
            .count();
        assert_eq!(report.gaps.len(), expected_rejections, "{:?}", report.gaps);
        if expected_rejections == 0 {
            assert!(report.gaps.is_empty(), "{:?}", report.gaps);
            complete_windows += 1;
        }
        for (index, case) in cases.iter().enumerate() {
            let OpportunityProduct::PrefillSpan { offset, .. } = case.product else {
                unreachable!()
            };
            let prompt = inputs.prompts[case.template];
            let projected_end = offset
                .checked_add(usize::try_from(chunk.get()).unwrap())
                .unwrap()
                .min(prompt);
            if projected_end > native_fit {
                // The last chunk of this rendered window exceeds the actual
                // request backing. Earlier pure projections do not qualify an
                // executable cohort or expand that backing.
                assert!(matches!(
                    report.opportunities[index].population,
                    CasePopulation::Unknown { .. }
                ));
                assert!(report.gaps.iter().any(|gap| gap.case_index == index
                    && matches!(
                        gap.reason,
                        InventoryGapReason::Projection(GeometryProjectionUnknown::Route(
                            ExecutionCostRouteUnknown::Resource(
                                ResourcePlanningUnknown::InvalidInput
                            )
                        ))
                    )));
                assert!(report.inputs[index].is_empty());
                capacity_rejections += 1;
                continue;
            }
            assert!(!report.gaps.iter().any(|gap| gap.case_index == index));
            assert!(matches!(
                report.opportunities[index].population,
                CasePopulation::Unique(_)
            ));
            assert_eq!(report.opportunities[index].minimum_fresh_members, 1);
            let original = report.inputs[index][0].original.as_ref().unwrap();
            let host = &original.physical_host_rows()[0];
            assert_eq!(
                host.terminal_expectation == HostTerminalExpectationV1::NoTokenProduced,
                offset + (chunk.get() as usize) < prompt
            );
            let (requests, execution) = inventory::readiness_requests_with_row_ceiling(
                case,
                &inputs.templates,
                inputs.chunk,
                inputs.prefill_row_ceiling,
            )
            .unwrap();
            assert_eq!(requests.len(), 1);
            assert_eq!(execution.prefill_chunk, chunk);
            let tokens = session
                .engine
                .inner
                .tokenizer
                .encode(&requests[0].request.prompt, true)
                .unwrap()
                .len();
            assert_eq!(tokens, prompt);
            assert_eq!(
                case.waves_with_row_ceiling(
                    prompt,
                    inputs.chunk.get() as usize,
                    inputs.prefill_row_ceiling
                )
                .unwrap(),
                (
                    prompt.div_ceil(chunk.get() as usize),
                    prompt.div_ceil(chunk.get() as usize)
                )
            );
            let readiness = readiness::longest(case, &cases, &inputs.outputs).unwrap();
            assert_eq!(readiness.explicit_prefill_chunk(), Some(chunk));
            assert!(
                resource_readiness::requests_with_row_ceiling(case, prompt, 1, None).is_err(),
                "a selected candidate cannot expand the original whole-wave cap"
            );
            assert!(
                resource_readiness::requests_with_row_ceiling(
                    case,
                    prompt,
                    inputs.chunk.get() as usize,
                    Some(NonZeroU32::MIN)
                )
                .is_err(),
                "nor the original per-row cap"
            );
        }
    }
    assert!(
        complete_windows > 0,
        "a complete first/middle/final window must be Known"
    );
    assert!(
        capacity_rejections > 0,
        "the real capacity boundary must be exercised"
    );
    assert_eq!(
        usize::try_from(executor.native_request_fit_tokens().unwrap()).unwrap(),
        native_fit,
        "numeric projection cannot expand native request authority"
    );
    assert_eq!(budget.deadline(), original_deadline);
    assert_eq!(
        executor.native_structured_counts(),
        before,
        "numeric input inventory cannot fabricate execution samples"
    );
    session.completed_owner_boundary().unwrap();
    session.shutdown().await.unwrap();
}

#[test]
fn checked_scheduler_prefill_candidate_readiness_keeps_distinct_chunk_identity() {
    let mut case = Case {
        product: OpportunityProduct::PrefillSpan {
            offset: 2,
            chunk: NonZeroU32::new(2).unwrap(),
        },
        template: 0,
        width: 1,
        maximum_output: NonZeroUsize::MIN,
        release_generated: 0,
        suffix_tokens: 1,
        preset: SloAutomaticCostProbeSamplingPresetV1::Configured,
        prefix: PrefixKind::Ordinary,
        route: CalibrationDecodeRoute::Actual,
        reset: true,
        acquisition: None,
    };
    let mut attempts = readiness::Attempts::new(2, 4096).unwrap();
    assert!(attempts
        .claim(&case, CalibrationDecodeRoute::Actual)
        .unwrap());
    assert!(!attempts
        .claim(&case, CalibrationDecodeRoute::Actual)
        .unwrap());
    case.product = OpportunityProduct::PrefillSpan {
        offset: 4,
        chunk: NonZeroU32::new(4).unwrap(),
    };
    assert!(attempts
        .claim(&case, CalibrationDecodeRoute::Actual)
        .unwrap());
    assert!(!attempts
        .claim(&case, CalibrationDecodeRoute::Actual)
        .unwrap());
}
