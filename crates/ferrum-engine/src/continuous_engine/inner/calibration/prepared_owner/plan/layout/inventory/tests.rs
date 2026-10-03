use super::*;
use crate::continuous_engine::inner::calibration::geometry_projection::tests::{
    assert_unsubmitted_clean, fixture,
};
use crate::AutomaticCostProbeOutput;
use ferrum_types::{InferenceRequest, TokenId};
use std::time::Duration;

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
        reset: false,
        acquisition: None,
    }
}

fn template(session: &CalibrationSession) -> AutomaticCostProbeTemplate {
    let mut request = InferenceRequest::new("test", session.configuration().model.model_id.clone());
    request.stream = true;
    request.sampling_params.temperature = 1.0;
    request.sampling_params.repetition_penalty = 1.0;
    AutomaticCostProbeTemplate::new(request, AutomaticCostProbeOutput::CliText).unwrap()
}

// Ordinary cases never consume this declaration. Prefix bytes are checked by
// the production tokenizer binder before any prefix-constrained projection.
fn pair() -> PrefixPair {
    let slot = StructuredPrefixSlotV5 {
        tokenizer_policy_sha256: [0; 32],
        token_ids: vec![TokenId::new(1)],
        token_bytes: vec![b"a".to_vec()],
    };
    PrefixPair {
        clean: slot.clone(),
        pending: slot,
    }
}

fn limits(requests: usize) -> InventoryLimits {
    InventoryLimits {
        route_population:
            ferrum_types::SloCalibrationRoutePopulationV1::WarmOrGraphDisabledWithNoSubmissionV2,
        deadline: Instant::now() + Duration::from_secs(10),
        maximum_admitted_requests: requests,
        maximum_projection_attempts: 16,
        maximum_retained_bytes: 4 * 1024 * 1024,
        maximum_route_states: 8,
        prefill_chunk: NonZeroU32::new(3).unwrap(),
        prefill_row_ceiling: None,
    }
}

#[tokio::test]
async fn checked_inventory_reuses_maximum_width_and_keeps_warm_prefill_conditional() {
    let (mut session, executor) = fixture(2).await;
    let templates = [template(&session)];
    let inventory = collect(
        &mut session,
        &[case(1), case(2), case(1)],
        &templates,
        &[1],
        &pair(),
        StructuredPopulationPolicyV1::HomogeneousOrdinaryDecodeV1,
        &[],
        limits(2),
    )
    .await
    .unwrap();
    assert_eq!(inventory.charge.admitted_requests, 2);
    // Each width still projects its initial prefill and its own decode query.
    // The completed width-one joint prefill is also its exact serial prefix;
    // width two reuses that owner and prepares only the newly added owner.
    let projected_widths = [1usize, 2];
    let expected_attempts = 2 * projected_widths.len()
        + projected_widths
            .windows(2)
            .map(|widths| widths[1] - widths[0])
            .sum::<usize>();
    assert_eq!(inventory.charge.projection_attempts, expected_attempts);
    assert!(!inventory.algorithm_inputs.is_empty());
    assert_eq!(inventory.opportunities.len(), 3);
    assert!(inventory.opportunities.iter().all(|opportunity| matches!(
        opportunity.population,
        CasePopulation::Unique(_)
    ) && opportunity
        .minimum_fresh_members
        == 0));
    assert!(inventory
        .inputs
        .iter()
        .all(|inputs| inputs.len() == 1 && !inputs[0].axes.is_empty()));
    assert!(inventory
        .gaps
        .iter()
        .all(|gap| matches!(gap.reason, InventoryGapReason::WarmResidencyUnproven)));
    assert_unsubmitted_clean(&session, &executor);
    session.shutdown().await.unwrap();
}

#[tokio::test]
async fn checked_inventory_reserves_requests_before_real_admission() {
    let (mut session, executor) = fixture(2).await;
    let templates = [template(&session)];
    assert!(collect(
        &mut session,
        &[case(2)],
        &templates,
        &[1],
        &pair(),
        StructuredPopulationPolicyV1::HomogeneousOrdinaryDecodeV1,
        &[],
        limits(1)
    )
    .await
    .is_err());
    assert_unsubmitted_clean(&session, &executor);
    session.shutdown().await.unwrap();
}

#[tokio::test]
async fn checked_inventory_configured_eager_keeps_known_inputs_without_false_member_floor() {
    use ferrum_interfaces::{
        execution_cost::ActualWaveGraphState,
        vnext::{
            DeviceCostGraphCatalogBuilder, DeviceCostGraphCatalogLimits,
            DeviceCostGraphConfiguration, DeviceCostGraphStreamState,
        },
    };
    use ferrum_types::SloCalibrationRoutePopulationV1 as Population;

    for (configuration, population, expected_floor) in [
        (
            DeviceCostGraphConfiguration::Unconfigured,
            Population::WarmOrGraphDisabledV1,
            1,
        ),
        (
            DeviceCostGraphConfiguration::OnDemand,
            Population::WarmOrGraphDisabledV1,
            0,
        ),
        (
            DeviceCostGraphConfiguration::OnDemand,
            Population::WarmOrGraphDisabledWithNoSubmissionV2,
            0,
        ),
        // Preserve the original AllAttempts contract. This regression adds a
        // route-population gate, not a blanket exclusion of eager execution.
        (
            DeviceCostGraphConfiguration::OnDemand,
            Population::AllAttempts,
            1,
        ),
    ] {
        let (mut session, executor) = fixture(1).await;
        executor.enable_cost_route_eager_boundary();
        let stream = DeviceCostGraphStreamState::new(configuration, 0, 0, 0).unwrap();
        let catalog = DeviceCostGraphCatalogBuilder::new(
            stream,
            DeviceCostGraphCatalogLimits::new(1, 1, 1).unwrap(),
        )
        .unwrap()
        .finish(&mut || Ok(()))
        .unwrap();
        executor.set_cost_graph_evidence(Some(stream), Some(catalog));
        let templates = [template(&session)];
        let mut original = case(1);
        original.reset = true;
        let mut bound = limits(1);
        bound.route_population = population;
        let inventory = collect(
            &mut session,
            &[original],
            &templates,
            &[1],
            &pair(),
            StructuredPopulationPolicyV1::HomogeneousOrdinaryDecodeV1,
            &[],
            bound,
        )
        .await
        .unwrap();
        assert_eq!(inventory.charge.admitted_requests, 1);
        assert!(inventory.charge.projection_attempts > 0);
        assert!(
            matches!(inventory.opportunities[0].population, CasePopulation::Unique(_)),
            "configured={configuration:?}, population={population:?}, gaps={:?}, provider_unknown={:?}",
            inventory.gaps,
            *executor.cost_route_unknown.lock(),
        );
        assert_eq!(
            inventory.opportunities[0].minimum_fresh_members,
            expected_floor
        );
        assert_eq!(inventory.inputs[0].len(), 1);
        assert!(!inventory.inputs[0][0].axes.is_empty());
        assert!(
            !inventory.algorithm_inputs.is_empty(),
            "outside numerical membership must preserve checked algorithm inventory"
        );
        let groups = populations::member_groups(&inventory.opportunities).unwrap();
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].possible_case_indices, vec![0]);
        assert_eq!(groups[0].guaranteed_case_indices.len(), expected_floor);
        if expected_floor == 0 {
            assert!(matches!(inventory.gaps.as_slice(), [InventoryGap {
                case_index: 0,
                reason: InventoryGapReason::OutsideDeclaredRoute {
                    population: actual_population,
                    projected_graph: ActualWaveGraphState::ConfiguredEager,
                },
            }] if *actual_population == population));
        } else {
            assert!(inventory.gaps.is_empty());
        }
        assert_unsubmitted_clean(&session, &executor);
        session.shutdown().await.unwrap();
    }
}

#[test]
fn algorithm_trajectory_uses_only_declared_request_span_and_provider_boundaries() {
    for prompt in [1usize, 5, 511] {
        for output in [1usize, 2, 4, 32] {
            let mut original = case(3);
            original.maximum_output = NonZeroUsize::new(output).unwrap();
            let boundaries = [2, 6, 7, 512, 513, 1025];
            let targets = trajectory_targets(&original, &[prompt], &boundaries).unwrap();
            if output == 1 {
                assert!(targets.is_empty());
                continue;
            }
            let first = (prompt + 1) as u32;
            let last = (prompt + output - 1) as u32;
            let expected = |sequence_tokens| {
                GeometryInputTarget::Decode(GeometryProjectionPoint {
                    rows: original.width,
                    sequence_tokens,
                })
            };
            assert!(targets.contains(&expected(first)));
            assert!(targets.contains(&expected(last)));
            for target in &targets {
                let GeometryInputTarget::Decode(point) = target else {
                    panic!("decode trajectory")
                };
                assert!((first..=last).contains(&point.sequence_tokens));
                assert!(
                    point.sequence_tokens == first
                        || point.sequence_tokens == last
                        || boundaries.contains(&point.sequence_tokens)
                );
            }
            for boundary in boundaries {
                if (first..=last).contains(&boundary) {
                    assert!(targets.contains(&expected(boundary)));
                }
            }
        }
    }
}

#[tokio::test]
async fn checked_inventory_captures_later_cpu_algorithm_without_inventing_members() {
    let (mut session, executor) = fixture(2).await;
    executor
        .row_selected_cpu_fill
        .store(true, std::sync::atomic::Ordering::Release);
    let templates = [template(&session)];
    let mut original = case(2);
    original.maximum_output = NonZeroUsize::new(4).unwrap();
    let inventory = collect(
        &mut session,
        &[original],
        &templates,
        &[1],
        &pair(),
        StructuredPopulationPolicyV1::HomogeneousOrdinaryDecodeV1,
        &[],
        limits(2),
    )
    .await
    .unwrap();
    let universe = ferrum_scheduler::implementations::continuous::cost_model::structured_v2::
        DeclaredAlgorithmUniverseV1::from_inputs(
            inventory.algorithm_inputs.iter().map(|input| input.as_ref()), 4096,
        ).unwrap();
    assert_eq!(
        universe.algorithm_count(),
        2,
        "the endpoint must capture the actual second CPU primitive beyond the first frontier"
    );
    assert_eq!(inventory.algorithm_case_inputs.len(), 1);
    let linked = ferrum_scheduler::implementations::continuous::cost_model::structured_v2::
        DeclaredAlgorithmUniverseV1::from_inputs(
            inventory.algorithm_case_inputs[0].iter().map(|&index| inventory.algorithm_inputs[index].as_ref()), 4096,
        ).unwrap();
    assert_eq!(
        linked, universe,
        "later real trajectory recipes retain their original case provenance"
    );
    assert_eq!(inventory.opportunities[0].minimum_fresh_members, 0);
    assert!(session
        .engine
        .inner
        .cost_runtime
        .as_ref()
        .unwrap()
        .snapshot()
        .is_none());
    assert_unsubmitted_clean(&session, &executor);
    session.shutdown().await.unwrap();
}

#[test]
fn checked_inventory_shares_admission_but_keeps_prefix_scenarios_and_residency_separate() {
    let mut first = case(1);
    let mut wider = case(2);
    assert!(same_group(&first, &wider));
    wider.reset = true;
    assert!(!same_group(&first, &wider));
    wider.reset = false;
    first.prefix = PrefixKind::Mixed { pending_rows: 1 };
    wider.prefix = PrefixKind::Mixed { pending_rows: 2 };
    assert!(same_group(&first, &wider));
    assert!(!same_scenario(&first, &wider));
    wider.prefix = first.prefix;
    assert!(same_group(&first, &wider));
    wider.release_generated = 2;
    assert!(same_group(&first, &wider));
    assert!(!same_scenario(&first, &wider));
    wider.maximum_output = NonZeroUsize::new(3).unwrap();
    assert!(!same_group(&first, &wider));
}

#[test]
fn continuation_prefill_case_offsets_use_original_prompt_chunk_and_capacity() {
    let mut initial = case(2);
    initial.reset = true;
    initial.maximum_output = NonZeroUsize::new(1).unwrap();
    initial.suffix_tokens = 1;
    let mut cases = vec![initial.clone()];
    append_continuation_prefill_cases(&mut cases, &[3], 3, usize::MAX).unwrap();
    assert_eq!(cases.len(), 3);
    assert_eq!(
        cases[1].product,
        OpportunityProduct::ContinuationPrefill { offset: 1 }
    );
    assert_eq!(
        cases[2].product,
        OpportunityProduct::ContinuationPrefill { offset: 2 }
    );
    assert!(!same_scenario(&cases[0], &cases[1]));
    assert!(!same_scenario(&cases[1], &cases[2]));
    assert!(same_group(&cases[0], &cases[1]));
    assert_eq!(
        target(&cases[2], &[3]).unwrap(),
        GeometryInputTarget::PrefillSpan { rows: 2, offset: 2 }
    );

    let mut one_continuation = vec![initial.clone()];
    append_continuation_prefill_cases(&mut one_continuation, &[2], 3, usize::MAX).unwrap();
    assert_eq!(one_continuation.len(), 2);
    let mut single_span = vec![initial.clone()];
    append_continuation_prefill_cases(&mut single_span, &[1], 3, usize::MAX).unwrap();
    assert_eq!(single_span.len(), 1);
    let mut conditional = initial.clone();
    conditional.reset = false;
    let mut warm = vec![conditional];
    append_continuation_prefill_cases(&mut warm, &[3], 3, usize::MAX).unwrap();
    assert_eq!(warm.len(), 1);

    let mut bounded = vec![initial];
    assert!(append_continuation_prefill_cases(
        &mut bounded,
        &[3],
        3,
        3 * std::mem::size_of::<Case>()
    )
    .is_err());
    assert_eq!(bounded.len(), 1);
}

#[tokio::test]
async fn checked_continuation_prefill_floor_comes_from_mandatory_real_prompt_spans() {
    let (mut session, executor) = fixture(2).await;
    let mut request =
        InferenceRequest::new("test ok v7", session.configuration().model.model_id.clone());
    request.stream = true;
    request.sampling_params.temperature = 1.0;
    request.sampling_params.repetition_penalty = 1.0;
    let templates =
        [AutomaticCostProbeTemplate::new(request, AutomaticCostProbeOutput::CliText).unwrap()];
    let mut initial = case(2);
    initial.reset = true;
    initial.maximum_output = NonZeroUsize::new(1).unwrap();
    initial.suffix_tokens = 1;
    let mut cases = vec![initial];
    append_continuation_prefill_cases(&mut cases, &[3], 3, usize::MAX).unwrap();
    let inventory = collect(
        &mut session,
        &cases,
        &templates,
        &[3],
        &pair(),
        StructuredPopulationPolicyV1::HomogeneousOrdinaryDecodeV1,
        &[],
        limits(2),
    )
    .await
    .unwrap();
    assert_eq!(inventory.charge.admitted_requests, 2);
    // Each distinct mandatory span projects its own wave, reusing the prior
    // successful joint successor. With maximum_output=1 there is no decode
    // trajectory, and no earlier prompt span needs another projection.
    assert_eq!(inventory.charge.projection_attempts, cases.len());
    assert_eq!(inventory.opportunities.len(), cases.len());
    assert!(inventory.opportunities.iter().all(|opportunity| matches!(
        opportunity.population,
        CasePopulation::Unique(_)
    ) && opportunity
        .minimum_fresh_members
        == 1));
    assert!(inventory.gaps.is_empty());
    // The sampler has no continuation allowance. All inventory entries are
    // checked hypothetical queries, and no sample or execution was fabricated.
    assert_unsubmitted_clean(&session, &executor);
    session.shutdown().await.unwrap();
}
