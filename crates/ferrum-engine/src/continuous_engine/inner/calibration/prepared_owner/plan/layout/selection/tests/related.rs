use super::*;

fn prefill_input(
    rows: u32,
    prompt: u64,
    maximum: u64,
    preset: SloAutomaticCostProbeSamplingPresetV1,
) -> StructuredInputV2 {
    prefill_input_graph(rows, prompt, prompt, maximum, preset, false, false)
}

pub(super) fn prefill_input_phase(
    rows: u32,
    immediate: u64,
    prompt: u64,
    maximum: u64,
    preset: SloAutomaticCostProbeSamplingPresetV1,
    model_eos: bool,
) -> StructuredInputV2 {
    prefill_input_graph(rows, immediate, prompt, maximum, preset, model_eos, true)
}

fn prefill_input_graph(
    rows: u32,
    immediate: u64,
    prompt: u64,
    maximum: u64,
    preset: SloAutomaticCostProbeSamplingPresetV1,
    model_eos: bool,
    separate_final_logits: bool,
) -> StructuredInputV2 {
    let final_logits = immediate == prompt;
    let greedy = preset == SloAutomaticCostProbeSamplingPresetV1::GreedyLength;
    let mut command =
        SelectedCommandCostBuilderV1::new_with_algorithm_work(u64::from(rows) * immediate);
    command
        .kernel(
            SelectedAlgorithmClassV1::new("fixture.selection.prefill", 1, [1; 32], [2; 32])
                .unwrap(),
            KernelNumericWorkV1 {
                logical_units: u64::from(rows) * immediate,
                padded_units: u64::from(rows) * immediate,
                inner_units_per_logical_unit: 2,
                grid: [rows, 1, 1],
                scratch_bytes: u64::from(rows) * 32,
                staged_weight_bytes: 0,
            },
        )
        .unwrap();
    let command = command.finish().unwrap();
    let mut builder =
        CanonicalWaveCostBuilder::new_with_structured_statistics(0, CostProductOutput::FullLogits);
    builder
        .physical_command(CostPhysicalCommand {
            native_op_id: "fixture.selection.prefill",
            command_index: 0,
            node_index: Some(0),
            command_phase: DeviceCommandPhase::Compute,
            provider: Some(CostProviderIdentity {
                provider_id: "numerical-fixture",
                implementation_fingerprint: "v1",
                operation_fingerprint: "v1",
            }),
            path: CostCommandPath::Eager,
            participant_start: 0,
            participant_count: rows,
            token_count: u64::from(rows) * immediate,
            batching_form: "packed",
            compute_dispatch_count: 1,
            transfer_command_count: 0,
            reusable_graph_node_count: None,
            statistical_evidence: Some(&command),
        })
        .unwrap();
    builder
        .core_readback_route(CoreReadbackRoute::SubmissionStaged)
        .unwrap();
    if separate_final_logits && final_logits {
        // This phased executor has an actual extra output-projection command
        // at the final prefill. A terminal-expectation bit alone does not split
        // StructuredOwnerKeyV2: identical command/policy/row-role inputs must
        // keep all their original numeric endpoints in one population.
        let mut output = SelectedCommandCostBuilderV1::new_with_algorithm_work(u64::from(rows));
        output
            .kernel(
                SelectedAlgorithmClassV1::new(
                    "fixture.selection.prefill.output",
                    1,
                    [3; 32],
                    [4; 32],
                )
                .unwrap(),
                KernelNumericWorkV1 {
                    logical_units: u64::from(rows) * 4096,
                    padded_units: u64::from(rows) * 4096,
                    inner_units_per_logical_unit: 2,
                    grid: [rows, 1, 1],
                    scratch_bytes: 0,
                    staged_weight_bytes: 0,
                },
            )
            .unwrap();
        let output = output.finish().unwrap();
        builder
            .physical_command(CostPhysicalCommand {
                native_op_id: "fixture.selection.prefill.output",
                command_index: 1,
                node_index: Some(1),
                command_phase: DeviceCommandPhase::Compute,
                provider: Some(CostProviderIdentity {
                    provider_id: "numerical-fixture",
                    implementation_fingerprint: "v1",
                    operation_fingerprint: "prefill-output-v1",
                }),
                path: CostCommandPath::Eager,
                participant_start: 0,
                participant_count: rows,
                token_count: u64::from(rows),
                batching_form: "packed",
                compute_dispatch_count: 1,
                transfer_command_count: 0,
                reusable_graph_node_count: None,
                statistical_evidence: Some(&output),
            })
            .unwrap();
    }
    for _ in 0..rows {
        builder
            .row(CanonicalCostRow {
                work: ActualRowWork::Prefill {
                    offset: 0,
                    count: immediate as u32,
                    total_prompt_tokens: prompt as u32,
                },
                host_policy_signature: [if greedy { 5 } else { 3 }; 32],
                mask_upload_required: false,
                output: CostRowOutput::Prefill { final_logits },
                host_features: Some(HostCostFeaturesV1 {
                    policy: HostCostPolicyV2 {
                        empirical_content_domain: Some(if greedy {
                            HostContentDomainV1::PlainTextGreedyV1
                        } else {
                            HostContentDomainV1::PlainTextInstalledV2(PlainTextPolicyCapabilityV2 {
                                sampling: PlainTextSamplingRouteV2::FullLogits,
                                model_eos,
                                user_stop: false,
                            })
                        }),
                        categorical_signature: [if greedy { 6 } else { 4 }; 32],
                        decoder_text_bytes_per_token: 4,
                        decoder_scratch_bytes_per_token: 8,
                        raw_token_bytes_bound: 4,
                    },
                    state: HostCostStateV1 {
                        generated_tokens_before: 0,
                        maximum_output_tokens: maximum,
                        sampling_history_tokens: 0,
                        sampling_history_scope: CostSamplingHistoryScope::FullGeneration,
                        pending_decoded_utf8: false,
                        completion_state_signature: satisfied_completion_cost_signature(),
                    },
                }),
            })
            .unwrap();
    }
    let wave = builder
        .finish_with_captured_structure(
            ActualWaveKind::Prefill,
            ActualWavePath::PlanRuntime,
            ActualWaveGraphState::Disabled,
            ActualWaveRowOrder::Ordered,
            u64::from(rows) * 32,
        )
        .unwrap();
    let statistical = wave.statistical.as_ref().unwrap();
    StructuredInputV2::from_actual_with_domain(
        &wave.exact,
        statistical,
        statistical.structured_capture().unwrap().unwrap(),
        &fixture::domain(),
    )
    .unwrap()
}

fn selected(
    cases: &[Case],
    opportunities: &[CaseOpportunity],
    facts: &[Vec<CheckedInputFacts>],
    prompts: &[usize],
    population: &StructuredServiceDeclarationV7,
    requests: usize,
    waves: usize,
) -> CheckedSelection {
    select(
        cases,
        opportunities,
        facts,
        prompts,
        8,
        population,
        requests,
        waves,
        usize::MAX,
    )
    .unwrap()
}

#[test]
fn checked_selection_related_populations_share_one_complete_horizon() {
    let (cases, opportunities, facts, population) = journal_grouping::same_product_families();
    let separate = selected(
        &cases[..4],
        &opportunities[..4],
        &facts[..4],
        &[61],
        &population,
        100_000,
        10_000_000,
    );
    let together = selected(
        &cases,
        &opportunities,
        &facts,
        &[61],
        &population,
        100_000,
        10_000_000,
    );
    assert_eq!(together.batches.len(), 1);
    let batch = &together.batches[0];
    assert_eq!(batch.population_indices, [0, 1]);
    assert_eq!(batch.representative_case_indices, [0, 2, 4, 6]);
    assert!(batch.scheduled);
    assert!(together.gaps.is_empty());
    assert!(batch.requests <= separate.requests * 2);
    assert!(batch.serial_token_work <= separate.batches[0].serial_token_work * 2);
    assert!(batch.maximum_anchor_span <= batch.schedule.phase_min_offered[0]);
    assert!(
        batch.planned_cycles
            * batch
                .input_opportunities
                .minimum_original_offers_per_completed_cycle
            >= batch.input_opportunities.required_original_offers + batch.schedule.block_offered
    );
    assert_eq!(
        together.execution_case_indices,
        batch
            .representative_case_indices
            .repeat(batch.planned_cycles)
    );
    // No member, branch or width disappears when related populations share
    // one declared source. The geometry may make separate sources as cheap.
    let tight = selected(
        &cases,
        &opportunities,
        &facts,
        &[61],
        &population,
        batch.requests,
        batch.serial_wave_upper_bound,
    );
    assert!(tight.populations.iter().all(|p| p.scheduled));
    assert_eq!(tight.requests, batch.requests);
    assert_eq!(tight.serial_wave_upper_bound, batch.serial_wave_upper_bound);
}

pub(super) fn prefill_inventory(
    presets: &[SloAutomaticCostProbeSamplingPresetV1],
) -> (
    Vec<Case>,
    Vec<CaseOpportunity>,
    Vec<Vec<CheckedInputFacts>>,
    StructuredServiceDeclarationV7,
) {
    let population = population::declaration(&Default::default(), fixture::domain()).unwrap();
    let mut cases = Vec::new();
    let mut opportunities = Vec::new();
    let mut facts = Vec::new();
    for &preset in presets {
        for width in [1, 2] {
            for (template, prompt) in [1, 2].into_iter().enumerate() {
                for maximum in [1, 2] {
                    let input = prefill_input(width, prompt, maximum, preset);
                    let classified = classify_alternatives(
                        std::slice::from_ref(&input),
                        population.population_policy(),
                        true,
                    )
                    .unwrap();
                    assert!(matches!(
                        classified,
                        CasePopulation::Unique(CheckedPopulationKey::ExactOwner(_))
                    ));
                    opportunities.push(CaseOpportunity {
                        population: classified,
                        minimum_fresh_members: 1,
                    });
                    facts.push(vec![input_facts(&StructuredQueryV2::exact(input)).unwrap()]);
                    cases.push(Case {
                        product: OpportunityProduct::Prefill,
                        template,
                        width: width as usize,
                        maximum_output: NonZeroUsize::new(maximum as usize).unwrap(),
                        release_generated: 0,
                        suffix_tokens: maximum as usize,
                        preset,
                        prefix: PrefixKind::Ordinary,
                        route: CalibrationDecodeRoute::Actual,
                        reset: true,
                        acquisition: None,
                    });
                }
            }
        }
    }
    (cases, opportunities, facts, population)
}

#[test]
fn checked_selection_multi_template_prefill_widths_share_original_phase_offers() {
    let (cases, opportunities, facts, population) =
        prefill_inventory(&[SloAutomaticCostProbeSamplingPresetV1::Configured]);
    let standalone_b2 = selected(
        &cases[4..],
        &opportunities[4..],
        &facts[4..],
        &[1, 2],
        &population,
        2048,
        16_384,
    );
    assert_eq!(standalone_b2.populations.len(), 1);
    assert!(standalone_b2.populations[0].scheduled);
    assert!(standalone_b2.batches[0].requests <= 2048);
    let combined = selected(
        &cases,
        &opportunities,
        &facts,
        &[1, 2],
        &population,
        2048,
        16_384,
    );
    assert_eq!(combined.populations.len(), 2);
    assert_eq!(combined.batches.len(), 1);
    assert!(combined.populations.iter().all(|p| p.scheduled));
    assert!(combined.gaps.is_empty());
    assert!(combined.requests <= 2048);
    assert!(combined.serial_wave_upper_bound <= 16_384);
    for group in &combined.populations {
        let representatives = &group.representative_case_indices;
        assert!(representatives.iter().any(|&i| cases[i].template == 0));
        assert!(representatives.iter().any(|&i| cases[i].template == 1));
        assert!(representatives.iter().any(|&i| facts[i][0].branches[1]));
        assert!(representatives.iter().any(|&i| facts[i][0].branches[3]));
    }
}

#[test]
fn checked_selection_mixed_installed_policies_fit_original_request_budget() {
    use SloAutomaticCostProbeSamplingPresetV1::{Configured, GreedyLength};
    let (cases, opportunities, facts, population) = prefill_inventory(&[Configured, GreedyLength]);
    let standalone_requests: usize = [0..8, 8..16]
        .into_iter()
        .map(|range| {
            selected(
                &cases[range.clone()],
                &opportunities[range.clone()],
                &facts[range],
                &[1, 2],
                &population,
                2048,
                16_384,
            )
            .requests
        })
        .sum();
    let result = selected(
        &cases,
        &opportunities,
        &facts,
        &[1, 2],
        &population,
        2048,
        16_384,
    );
    // Actual canonical host policies produce separate ExactOwner populations
    // at both widths. Each source uses its own complete input cycle; splitting
    // related populations must retain all four keys within the original budget.
    assert_eq!(result.populations.len(), 4);
    assert!(result.populations.iter().all(|p| p.scheduled));
    assert!(result.gaps.is_empty());
    let requests: usize = result
        .execution_case_indices
        .iter()
        .map(|&i| cases[i].width)
        .sum();
    let waves: usize = result
        .execution_case_indices
        .iter()
        .map(|&i| cases[i].waves([1, 2][cases[i].template], 8).unwrap().1)
        .sum();
    assert_eq!(result.requests, requests);
    assert_eq!(result.serial_wave_upper_bound, waves);
    assert!(requests <= 2048);
    assert!(waves <= 16_384);
    assert!(result.requests <= standalone_requests);

    let mut population_visits = vec![0; result.populations.len()];
    let mut execution_offset = 0;
    for (batch_index, batch) in result.batches.iter().enumerate() {
        assert!(batch.scheduled && batch.schedule_within_capacity);
        assert!(!batch.population_indices.is_empty());
        let cycle_waves: usize = batch
            .representative_case_indices
            .iter()
            .map(|&i| cases[i].waves([1, 2][cases[i].template], 8).unwrap().0)
            .sum();
        assert_eq!(batch.schedule.block_offered, cycle_waves);
        assert_eq!(
            batch.input_opportunities.successful_cycle_wave_upper_bound,
            cycle_waves
        );
        for phase in 0..3 {
            let phase_offers = batch.input_opportunities.phase_original_offer_bounds[phase];
            assert_eq!(phase_offers % cycle_waves, 0);
            assert!(phase_offers >= batch.schedule.phase_min_offered[phase]);
            assert!(phase_offers >= batch.input_opportunities.maximum_fresh_member_span[phase]);
            assert!(batch.maximum_anchor_span <= batch.schedule.phase_min_offered[phase]);
        }
        // Include the final block-closing cycle; every source must complete
        // all of its independently frozen phases before publication.
        assert!(
            batch.planned_cycles
                * batch
                    .input_opportunities
                    .minimum_original_offers_per_completed_cycle
                >= batch.input_opportunities.required_original_offers
                    + batch.schedule.block_offered
        );
        let expected_execution = batch
            .representative_case_indices
            .repeat(batch.planned_cycles);
        let execution_end = execution_offset + expected_execution.len();
        assert_eq!(
            &result.execution_case_indices[execution_offset..execution_end],
            expected_execution.as_slice()
        );
        execution_offset = execution_end;
        for &population_index in &batch.population_indices {
            population_visits[population_index] += 1;
            let group = &result.populations[population_index];
            assert_eq!(group.batch_index, Some(batch_index));
            assert!(group
                .representative_case_indices
                .iter()
                .all(|i| batch.representative_case_indices.contains(i)));
        }
    }
    assert_eq!(execution_offset, result.execution_case_indices.len());
    assert!(population_visits.iter().all(|&visits| visits == 1));
    for group in &result.populations {
        assert!(!group.representative_case_indices.is_empty());
        for template in [0, 1] {
            assert!(group
                .representative_case_indices
                .iter()
                .any(|&i| cases[i].template == template));
        }
        for branch in [1, 3] {
            assert!(group
                .representative_case_indices
                .iter()
                .any(|&i| facts[i][0].branches[branch]));
        }
        for &index in &group.representative_case_indices {
            assert_eq!(
                facts[index][0].key(population.population_policy()),
                group.key
            );
            assert!(result.execution_case_indices.contains(&index));
        }
    }
    for preset in [Configured, GreedyLength] {
        for width in [1, 2] {
            assert!(result.populations.iter().any(|p| {
                p.representative_case_indices
                    .iter()
                    .all(|&i| cases[i].preset == preset && cases[i].width == width)
            }));
        }
    }
}

#[test]
fn checked_selection_distinct_products_keep_independent_journals_and_all_context_anchors() {
    let (mut cases, mut opportunities, mut facts, population) = two_families();
    // Two rendered inputs reach the same KV frontier through distinct prompt
    // and generated-history lengths. Keep both physical history axes in each
    // population's representative set.
    for index in [2, 3, 6, 7] {
        let case = &mut cases[index];
        case.template = 1;
        case.release_generated = 4;
        case.maximum_output = NonZeroUsize::new(22).unwrap();
        let input = fixture::input(
            case.width as u32,
            4,
            if case.product == OpportunityProduct::Full {
                CostProductOutput::FullLogits
            } else {
                CostProductOutput::GreedyToken
            },
            false,
            true,
        );
        opportunities[index].population = classify_alternatives(
            std::slice::from_ref(&input),
            population.population_policy(),
            true,
        )
        .unwrap();
        facts[index] = vec![input_facts(&StructuredQueryV2::exact(input)).unwrap()];
    }
    let result = selected(
        &cases,
        &opportunities,
        &facts,
        &[61, 60],
        &population,
        100_000,
        10_000_000,
    );
    assert_eq!(result.batches.len(), 2);
    assert_eq!(result.batches[0].population_indices, [0]);
    assert_eq!(result.batches[1].population_indices, [1]);
    assert_eq!(result.batches[0].representative_case_indices, [0, 2]);
    assert_eq!(result.batches[1].representative_case_indices, [4, 6]);
    assert_ne!(facts[0][0].owner.product, facts[4][0].owner.product);
    assert!(result.populations.iter().all(|p| p.scheduled));
    for group in &result.populations {
        let templates: Vec<_> = group
            .representative_case_indices
            .iter()
            .map(|&i| cases[i].template)
            .collect();
        assert_eq!(templates, [0, 1]);
    }
}

#[test]
fn checked_selection_related_identity_requires_whole_template_set() {
    let (mut cases, opportunities, facts, population) = two_families();
    let baseline = selected(
        &cases,
        &opportunities,
        &facts,
        &[61],
        &population,
        100_000,
        10_000_000,
    );
    let mut a = baseline.batches[0].clone();
    let mut ab = a.clone();
    let mut b = a.clone();
    a.representative_case_indices = vec![0];
    ab.representative_case_indices = vec![0, 2];
    b.representative_case_indices = vec![2];
    cases[2].template = 1;
    assert!(!same_representative_inputs(&a, &ab, &cases));
    assert!(!same_representative_inputs(&ab, &b, &cases));
    assert!(!same_representative_inputs(&a, &b, &cases));
    // Repeated identity occurrences and order are immaterial; the complete
    // rendered-template set is what must be equal.
    b.representative_case_indices = vec![2, 0, 2];
    assert!(same_representative_inputs(&ab, &b, &cases));

    for differing in 0..3 {
        let (mut cases, opportunities, facts, population) = two_families();
        for case in &mut cases[4..] {
            match differing {
                0 => case.template = 1,
                1 => case.route = CalibrationDecodeRoute::FullLogits,
                _ => case.preset = SloAutomaticCostProbeSamplingPresetV1::GreedyLength,
            }
        }
        let result = selected(
            &cases,
            &opportunities,
            &facts,
            &[61, 61],
            &population,
            100_000,
            10_000_000,
        );
        // The distinct products keep separate complete sources. Their original
        // template, auxiliary route and preset declarations are all preserved.
        assert_eq!(result.batches.len(), 2);
        assert_eq!(result.populations.len(), 2);
        assert_ne!(result.populations[0].key, result.populations[1].key);
        assert_eq!(result.populations[0].representative_case_indices, [0, 2]);
        assert_eq!(result.populations[1].representative_case_indices, [4, 6]);
        assert!(result.populations.iter().all(|p| p.scheduled));
        for member in &result.populations {
            for &index in &member.representative_case_indices {
                assert_eq!(
                    facts[index][0].key(population.population_policy()),
                    member.key
                );
                assert!(result.execution_case_indices.contains(&index));
            }
        }
        // The shared horizon schedules the original route and preset cases;
        // it never rewrites them into the first population's contract.
        if differing == 1 {
            assert!(cases[4..]
                .iter()
                .all(|c| c.route == CalibrationDecodeRoute::FullLogits));
            assert!(cases[..4]
                .iter()
                .all(|c| c.route == CalibrationDecodeRoute::Actual));
        } else if differing == 2 {
            assert!(cases[4..]
                .iter()
                .all(|c| c.preset == SloAutomaticCostProbeSamplingPresetV1::GreedyLength));
            assert!(cases[..4]
                .iter()
                .all(|c| c.preset == SloAutomaticCostProbeSamplingPresetV1::Configured));
        }
    }
}

#[test]
fn checked_selection_related_merge_keeps_request_wave_and_schedule_capacity_gaps() {
    let (mut cases, opportunities, facts, mut population) =
        journal_grouping::same_product_families();
    let first = selected(
        &cases[..4],
        &opportunities[..4],
        &facts[..4],
        &[61],
        &population,
        100_000,
        10_000_000,
    );
    for (requests, waves, request_limited) in [
        (first.requests, 10_000_000, true),
        (100_000, first.serial_wave_upper_bound, false),
    ] {
        let result = selected(
            &cases,
            &opportunities,
            &facts,
            &[61],
            &population,
            requests,
            waves,
        );
        assert_eq!(result.batches.len(), 2);
        assert!(result.populations[0].scheduled);
        assert!(!result.populations[1].scheduled);
        assert_eq!(result.execution_case_indices, first.execution_case_indices);
        assert!(result.gaps.iter().any(|gap| {
            gap.population.as_ref() == Some(&result.populations[1].key)
                && if request_limited {
                    matches!(gap.reason, SelectionGapReason::RemainingRequests { .. })
                } else {
                    matches!(gap.reason, SelectionGapReason::RemainingWaves { .. })
                }
        }));
    }
    population.schedule.phase_min_offered = [180; 3];
    let split = selected(
        &cases,
        &opportunities,
        &facts,
        &[61],
        &population,
        100_000,
        10_000_000,
    );
    assert_eq!(split.batches.len(), 1);
    assert!(split.populations.iter().all(|p| p.scheduled));
    // A complete cycle that exceeds the native member capacity cannot be
    // shortened by deleting representatives or borrowing another source's
    // quota. Keep its original population and an explicit capacity gap.
    for case in &mut cases {
        case.maximum_output = NonZeroUsize::new(4096).unwrap();
    }
    let rejected = selected(
        &cases,
        &opportunities,
        &facts,
        &[61],
        &population,
        100_000,
        10_000_000,
    );
    assert!(rejected.execution_case_indices.is_empty());
    for group in &rejected.populations {
        assert!(rejected.gaps.iter().any(|gap| {
            gap.population.as_ref() == Some(&group.key)
                && matches!(gap.reason, SelectionGapReason::SourceScheduleCapacity)
        }));
    }
}
