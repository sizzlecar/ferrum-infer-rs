//! Original recipe ownership must fit before the optional scope is selected.
use super::*;

#[tokio::test]
async fn checked_duplicate_outcome_recipe_fits_before_local_scope_selection() {
    use crate::continuous_engine::inner::calibration::geometry_projection::tests::assert_unsubmitted_clean;

    let mut reference = None;
    // First retain the complete roomy plan. Then exercise the actual freeze
    // caller at the unique-recipe boundary and one byte below that boundary.
    for missing_byte in [None, Some(0usize), Some(1)] {
        let (mut session, executor) = fixture(2).await;
        executor
            .recycle_completed_bindings
            .store(true, Ordering::Release);
        executor
            .row_selected_cpu_fill
            .store(true, Ordering::Release);
        Arc::get_mut(&mut session.engine.inner).unwrap().tokenizer = Arc::new(tokenizer().await);
        let model = session.configuration().model.model_id.clone();
        let settings = SloAutomaticCalibrationSettingsV1::default();
        let inputs = Box::pin(PreparedProbeInputs::new(
            &mut session,
            &settings,
            &[template("test", &model).unwrap()],
        ))
        .await
        .unwrap();
        let (mut cases, skipped, unavailable, mut inventory_limit) =
            prepare_cases(&inputs).unwrap();
        let original = cases
            .iter()
            .position(|case| {
                case.width == 2
                    && case.preset == SloAutomaticCostProbeSamplingPresetV1::Configured
                    && matches!(case.prefix, PrefixKind::Clean)
                    && matches!(case.product, OpportunityProduct::Full)
            })
            .unwrap();
        let duplicate = cases.len();
        let case_capacity = cases.capacity();
        cases.push(cases[original].clone());
        inventory_limit -= (cases.capacity() - case_capacity) * std::mem::size_of::<Case>();
        let mut budget = ProbeExecutionBudget::new_with_input_projection_limit(
            Instant::now() + Duration::from_millis(settings.cost_probe.maximum_duration_ms.get()),
            settings.cost_probe.maximum_probe_requests,
            settings.cost_probe.maximum_offered_waves,
            settings.cost_probe.maximum_input_projection_requests,
        );
        let deadline = budget.deadline();
        let inventory = Box::pin(collect_ready(
            &mut session,
            &inputs,
            &cases,
            &mut budget,
            inventory_limit,
        ))
        .await
        .unwrap();
        assert!(!inventory.inputs[original].is_empty());
        assert_eq!(inventory.inputs[original], inventory.inputs[duplicate]);
        assert_eq!(
            inventory.algorithm_case_inputs[original],
            inventory.algorithm_case_inputs[duplicate]
        );
        assert_eq!(
            serde_json::to_value(&inventory.opportunities[original]).unwrap(),
            serde_json::to_value(&inventory.opportunities[duplicate]).unwrap()
        );
        // These two cases resolve to the same captured scenario, target and
        // branches. Charge every case's metadata, axes and links as before;
        // only a second independently owned copy of that immutable recipe is
        // excluded from the requested allowance. Once shared, this is zero.
        let duplicated_payload = inventory.inputs[original]
            .iter()
            .zip(&inventory.inputs[duplicate])
            .map(|(left, right)| {
                let left = left.original.as_ref().unwrap();
                let right = right.original.as_ref().unwrap();
                assert_eq!(left, right);
                if Arc::ptr_eq(left, right) {
                    0
                } else {
                    right.retained_payload_bytes().unwrap() + 2 * std::mem::size_of::<usize>()
                }
            })
            .sum::<usize>();
        let retained = inventory.retained_payload_bytes().unwrap();
        let unique_inventory = retained.checked_sub(duplicated_payload).unwrap();
        let seed =
            universe::freeze_declared_algorithms(&inventory, &inputs.population, inventory_limit)
                .unwrap()
                .unwrap();
        assert!(seed.algorithm_count() >= 2);
        let seed_bytes = seed.retained_payload_bytes().unwrap();
        let authorized = |bytes| {
            selection::composition_authorized(
                &inventory.opportunities,
                &inventory.inputs,
                budget.selection_requests_remaining(),
                &inputs.population,
                true,
                bytes,
                &seed,
            )
            .unwrap()
        };
        // Query the production predicate rather than reproduce its allocation
        // formula or borrow the balance after release_original_inputs.
        let mut lower = 0;
        let mut upper = inventory_limit - retained - seed_bytes;
        assert!(authorized(upper));
        while lower < upper {
            let middle = lower + (upper - lower) / 2;
            if authorized(middle) {
                upper = middle;
            } else {
                lower = middle + 1;
            }
        }
        let selection_peak = lower;
        assert!(selection_peak > 0 && !authorized(selection_peak - 1));
        let unique_limit = unique_inventory + seed_bytes + selection_peak;
        assert!(unique_limit < inventory_limit);
        let limit = missing_byte.map_or(inventory_limit, |missing| unique_limit - missing);
        // Compare duplicate vs original cases at this same authorization,
        // rather than comparing periodic selection to a roomier finite scope.
        // Both controls borrow the same real checked outcomes and global seed.
        let same_mode_reference = if missing_byte != Some(1) {
            let mut original_inventory = inventory.clone();
            original_inventory.opportunities.pop();
            original_inventory.inputs.pop();
            original_inventory.algorithm_case_inputs.pop();
            original_inventory
                .gaps
                .retain(|gap| gap.case_index != duplicate);
            let mut geometry = inputs.input_geometry_visit_limit.map(
                ferrum_scheduler::implementations::continuous::cost_model::structured_v2::StructuredInputGeometryWorkV1::new,
            );
            Some(
                selection::select_with_capacity_and_trajectories(
                    &cases[..duplicate],
                    &original_inventory.opportunities,
                    &original_inventory.inputs,
                    &inputs.prompts,
                    inputs.chunk.get() as usize,
                    inputs.prefill_row_ceiling,
                    &inputs.population,
                    selection::SelectionCapacity {
                        requests: budget.selection_requests_remaining(),
                        execution_actions: budget.selection_attempts_remaining(),
                        declared_offer_rows: inputs
                            .limits
                            .max_samples
                            .get()
                            .min(inputs.limits.max_total_shape_rows.get()),
                    },
                    limit - retained - seed_bytes,
                    None,
                    None,
                    geometry.as_mut(),
                    Some(inputs.maximum_retained_sources),
                    Some(&seed),
                    Some(&original_inventory),
                )
                .unwrap(),
            )
        } else {
            None
        };
        // Freeze constructs its own seed under the exact original boundary.
        drop(seed);
        eprintln!(
            "same-outcome freeze: retained={retained} duplicate_payload={duplicated_payload} unique_inventory={unique_inventory} seed={seed_bytes} selection_peak={selection_peak} inventory_limit={limit} missing_byte={missing_byte:?}"
        );
        let mut retained_seed = None;
        let checked_cases = cases.clone();
        let checked_prompts = inputs.prompts.clone();
        let checked_chunk = inputs.chunk.get() as usize;
        let checked_row_ceiling = inputs.prefill_row_ceiling;
        let checked_recipes: Vec<Vec<_>> = inventory
            .inputs
            .iter()
            .map(|facts| {
                facts
                    .iter()
                    .filter_map(|fact| fact.original.clone())
                    .collect()
            })
            .collect();
        let plan = freeze_inventory(
            &session,
            inputs,
            cases,
            skipped,
            unavailable,
            inventory,
            &mut budget,
            limit,
            false,
            None,
            None,
            Some(&mut retained_seed),
        )
        .unwrap()
        .unwrap();
        let selected = plan.audit().checked_selection.as_ref().unwrap();
        let capacity_denied = selected.gaps.iter().any(|gap| {
            matches!(
                gap.reason,
                selection::SelectionGapReason::CombinationCapacity
            )
        });
        let scoped = selected
            .batches
            .iter()
            .any(|batch| batch.scheduled && batch.algorithm_universe.is_some());
        let actual = selected.clone();
        assert!(actual.retained_payload_bytes().unwrap() <= limit);
        if let Some(original_selection) = same_mode_reference {
            // Batches include scope, original representatives, full finite or
            // periodic proof, F/R/Q barriers, opportunities and every work sum.
            assert_eq!(
                serde_json::to_value(&actual.batches).unwrap(),
                serde_json::to_value(&original_selection.batches).unwrap()
            );
            assert_eq!(
                actual.execution_case_indices,
                original_selection.execution_case_indices
            );
            assert_eq!(
                [
                    actual.requests,
                    actual.serial_wave_upper_bound,
                    actual.declared_offer_row_bound
                ],
                [
                    original_selection.requests,
                    original_selection.serial_wave_upper_bound,
                    original_selection.declared_offer_row_bound
                ]
            );
        }
        // Charge the actual sequence, including source setup once, through
        // the same original case work contract even if roomy storage permits
        // an independently recomputed scope extension.
        let mut totals = [0usize; 3];
        let mut execution = Vec::new();
        for batch in &actual.batches {
            let setup = work::setup_for_indices(&checked_cases, &batch.representative_case_indices)
                .unwrap();
            let mut work = [
                setup.requests,
                setup.execution_actions,
                setup.serial_declared_offer_rows,
                setup.serial_token_work,
            ];
            for index in batch.execution_case_indices().unwrap() {
                let item = work::case_work(
                    &checked_cases[index],
                    checked_prompts[checked_cases[index].template],
                    checked_chunk,
                    checked_row_ceiling,
                )
                .unwrap();
                for (sum, value) in work.iter_mut().zip([
                    item.requests,
                    item.execution_actions,
                    item.serial_declared_offer_rows,
                    item.serial_token_work,
                ]) {
                    *sum += value;
                }
                if batch.scheduled {
                    execution.push(index);
                }
            }
            assert_eq!(
                work,
                [
                    batch.requests,
                    batch.serial_wave_upper_bound,
                    batch.declared_offer_row_bound,
                    batch.serial_token_work
                ]
            );
            if batch.scheduled {
                for (total, value) in totals.iter_mut().zip(work) {
                    *total += value;
                }
            }
        }
        assert_eq!(execution, actual.execution_case_indices);
        assert_eq!(
            totals,
            [
                actual.requests,
                actual.serial_wave_upper_bound,
                actual.declared_offer_row_bound
            ]
        );
        assert_eq!(budget.deadline(), deadline);
        assert_unsubmitted_clean(&session, &executor);
        assert!(session
            .engine
            .inner
            .cost_runtime
            .as_ref()
            .unwrap()
            .snapshot()
            .is_none());
        session.shutdown().await.unwrap();
        match missing_byte {
            None => {
                assert!(
                    !capacity_denied && scoped,
                    "roomy real CPU input must select a local scope"
                );
                reference = Some(actual);
            }
            Some(0) => {
                assert!(
                    !capacity_denied && scoped,
                    "the actual pre-release freeze must fit one owned recipe per captured outcome; duplicated_payload={duplicated_payload}, capacity_denied={capacity_denied}, scoped={scoped}"
                );
                assert_eq!(
                    duplicated_payload, 0,
                    "capacity must come from real sharing, not omitted payload accounting"
                );
                let roomy: &selection::CheckedSelection = reference.as_ref().unwrap();
                assert_eq!(actual.populations.len(), roomy.populations.len());
                assert_eq!(
                    serde_json::to_value(&actual.input_geometry).unwrap(),
                    serde_json::to_value(&roomy.input_geometry).unwrap()
                );
                for old in &actual.populations {
                    let new = roomy
                        .populations
                        .iter()
                        .find(|new| new.key == old.key)
                        .unwrap();
                    assert_eq!(
                        old.representative_case_indices,
                        new.representative_case_indices
                    );
                    assert_eq!(
                        serde_json::to_value(&old.input_geometry).unwrap(),
                        serde_json::to_value(&new.input_geometry).unwrap()
                    );
                    if old.scheduled {
                        assert!(
                            new.scheduled,
                            "extra finite storage must not evict an original admitted family: {:?}",
                            old.key
                        );
                        let before = &actual.batches[old.batch_index.unwrap()];
                        let after = &roomy.batches[new.batch_index.unwrap()];
                        match (&before.algorithm_universe, &after.algorithm_universe) {
                            (Some(old), Some(new)) => assert!(new.contains_universe(old)),
                            (None, None) => {}
                            _ => panic!("extra storage lost or relabelled an original raw scope"),
                        }
                        assert_eq!(before.schedule.min_members, after.schedule.min_members);
                        for index in &old.representative_case_indices {
                            let a = before
                                .representative_case_indices
                                .iter()
                                .position(|i| i == index)
                                .unwrap();
                            let b = after
                                .representative_case_indices
                                .iter()
                                .position(|i| i == index)
                                .unwrap();
                            match (&before.scoped_opportunities, &after.scoped_opportunities) {
                                (Some(old), Some(new)) => {
                                    assert_eq!(
                                        old[a].minimum_fresh_members,
                                        new[b].minimum_fresh_members
                                    );
                                    assert!(!checked_recipes[*index].is_empty());
                                    for recipe in &checked_recipes[*index] {
                                        for (batch, opportunity) in
                                            [(before, &old[a]), (after, &new[b])]
                                        {
                                            let key = recipe
                                                .numerical_family_key_for_universe(
                                                    batch.algorithm_universe.as_ref().unwrap(),
                                                )
                                                .unwrap();
                                            assert_eq!(opportunity.population, CasePopulation::Unique(
                                                populations::CheckedPopulationKey::NumericalFamily(key),
                                            ));
                                        }
                                    }
                                }
                                (None, None) => {}
                                _ => panic!("scope opportunity/floor was lost"),
                            }
                        }
                    }
                }
                for gap in &actual.gaps {
                    // Work/source capacity may improve. All original input,
                    // geometry, priority and policy limitations remain visible.
                    if !matches!(
                        gap.reason,
                        selection::SelectionGapReason::RemainingRequests { .. }
                            | selection::SelectionGapReason::RemainingWaves { .. }
                            | selection::SelectionGapReason::RemainingOfferRows { .. }
                            | selection::SelectionGapReason::RetainedSourceCapacity { .. }
                            | selection::SelectionGapReason::SourceScheduleCapacity
                            | selection::SelectionGapReason::AnchorAfterEarliestFreeze { .. }
                    ) {
                        assert!(roomy
                            .gaps
                            .iter()
                            .any(|candidate| serde_json::to_value(candidate).unwrap()
                                == serde_json::to_value(gap).unwrap()));
                    }
                }
            }
            Some(1) => {
                assert!(capacity_denied && !scoped);
                assert!(actual.batches.iter().any(|batch| batch.scheduled));
            }
            _ => unreachable!(),
        }
    }
}
