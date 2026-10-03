//! Explicit offline experiments. Every candidate uses the same production
//! batch work calculation; no candidate changes product defaults or signs a source.
use super::*;
use serde::Serialize;

#[derive(Debug, Default, Clone, Copy, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
enum GeometryPlan {
    #[default]
    CurrentFull,
    AnchoredReadinessV2WidthCostV1,
}
#[derive(Debug, Clone, Copy, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
enum SchedulePlan {
    CompleteCycle,
    EachOffer,
}
#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Options {
    #[serde(default)]
    geometry_plan: GeometryPlan,
    schedule: SchedulePlan,
    /// Explicit original population indices; each group is one hypothetical
    /// new journal. Existing captured source files/models are never relabelled.
    population_groups: Vec<Vec<usize>>,
    /// Additional required original families. Original scheduled populations
    /// are always required, whether or not this list repeats them.
    required_populations: Vec<usize>,
    /// Evaluate every target from zero (no filler) to this inclusive bound.
    /// Further bounded by the original maximum per-phase member requirement.
    maximum_filler_members_per_cycle: usize,
}

fn each_offer_schedule(
    starts: &[usize],
    ends: &[usize],
    settings: &StructuredSettingsV2,
) -> Result<(OwnerBlockScheduleV1, usize)> {
    let (original, anchor) = budget::startup_schedule(starts, ends, settings)?;
    let schedule = OwnerBlockScheduleV1::new(1, original.phase_min_offered, original.min_members)
        .map_err(|reason| error(format!("offline each-offer schedule: {reason:?}")))?;
    Ok((schedule, anchor))
}

struct PopulationPlan {
    original_index: usize,
    key: CheckedPopulationKey,
    cases: Vec<usize>,
    declaration: StructuredServiceDeclarationV7,
    universe: Option<ferrum_scheduler::implementations::continuous::cost_model::structured_v2::DeclaredAlgorithmUniverseV1>,
}

fn expand_cycle(
    key: &CheckedPopulationKey,
    original: &[usize],
    target: usize,
    cases: &[Case],
    opportunities: &[CaseOpportunity],
    prompts: &[usize],
    chunk: usize,
    row_ceiling: Option<NonZeroU32>,
) -> AuditResult<Vec<usize>> {
    ensure!(!original.is_empty(), "empty original representative cycle");
    let mut cheapest = None;
    for &index in original {
        let opportunity = opportunities
            .get(index)
            .context("filler opportunity outside input")?;
        ensure!(
            opportunity.minimum_fresh_members == 1
                && matches!(&opportunity.population, CasePopulation::Unique(actual) if actual == key),
            "filler requires the same exact original guaranteed family"
        );
        let case = cases.get(index).context("filler case outside input")?;
        let prompt = *prompts
            .get(case.template)
            .context("filler prompt outside input")?;
        let work = work::case_work(case, prompt, chunk, row_ceiling)?;
        let cost = (
            work.requests,
            work.execution_actions,
            work.serial_token_work,
            work.declared_offers_upper,
            index,
        );
        if cheapest.is_none_or(|old| cost < old) {
            cheapest = Some(cost);
        }
    }
    let mut expanded = original.to_vec();
    if target > original.len() {
        expanded.resize(target, cheapest.context("no original filler")?.4);
    }
    // These are new complete cohort occurrences, not duplicate geometry rows
    // or reused samples. Every original representative stays in original order.
    Ok(expanded)
}

fn group_is_compatible(group: &[&PopulationPlan], cases: &[Case]) -> AuditResult<()> {
    ensure!(!group.is_empty(), "empty population group");
    let first = group[0];
    ensure!(
        group.len() <= first.declaration.maximum_owners,
        "group exceeds original owner bound"
    );
    if group.len() == 1 {
        return Ok(());
    }
    let universe = first
        .universe
        .as_ref()
        .context("multi-population group requires an original fixed U")?;
    let scope = serde_json::to_value(universe)?;
    let first_key = serde_json::to_value(&first.key)?;
    let first_family = field(&first_key, "NumericalFamily")?;
    let first_route = cases[first.cases[0]].route;
    for (i, member) in group.iter().enumerate() {
        ensure!(
            group[..i].iter().all(|other| other.key != member.key),
            "duplicate exact family in group"
        );
        ensure!(
            member
                .universe
                .as_ref()
                .map(serde_json::to_value)
                .transpose()?
                .as_ref()
                == Some(&scope),
            "group has different original U"
        );
        let key = serde_json::to_value(&member.key)?;
        let family = field(&key, "NumericalFamily")?;
        for name in [
            "workload_domain",
            "product",
            "readback",
            "route",
            "algorithm_universe",
        ] {
            ensure!(
                field(first_family, name)? == field(family, name)?,
                "group {name} metadata differs"
            );
        }
        ensure!(
            member
                .cases
                .iter()
                .all(|&index| cases[index].route == first_route),
            "group has different original preparation route"
        );
        ensure!(
            serde_json::to_value(&member.declaration.settings)?
                == serde_json::to_value(&first.declaration.settings)?
                && member.declaration.schedule.prediction_validity
                    == first.declaration.schedule.prediction_validity,
            "group numerical settings or validity differs"
        );
    }
    // This necessary metadata check cannot prove retained recipe closure,
    // selector priority preservation, memory admission or actual qualification.
    Ok(())
}

fn planned_group(
    capture: &Capture,
    group: &[&PopulationPlan],
    opportunities: &[CaseOpportunity],
    target: usize,
    schedule: SchedulePlan,
) -> AuditResult<(SelectedBatch, Vec<usize>)> {
    let mut populations = Vec::new();
    let mut filler_counts = Vec::new();
    for member in group {
        let cycle = expand_cycle(
            &member.key,
            &member.cases,
            target,
            &capture.cases,
            opportunities,
            &capture.prompts,
            capture.chunk,
            capture.row_ceiling,
        )?;
        filler_counts.push(cycle.len() - member.cases.len());
        populations.push(SelectedPopulation {
            key: member.key.clone(),
            representative_case_indices: cycle,
            maximum_anchor_span: 0,
            scheduled: false,
            batch_index: None,
            input_geometry: None,
        });
    }
    let indices: Vec<_> = (0..populations.len()).collect();
    let schedule_for: fn(
        &[usize],
        &[usize],
        &StructuredSettingsV2,
    ) -> Result<(OwnerBlockScheduleV1, usize)> = match schedule {
        SchedulePlan::CompleteCycle => budget::startup_schedule,
        SchedulePlan::EachOffer => each_offer_schedule,
    };
    let mut batch = batch_plan_with_schedule(
        &indices,
        &populations,
        &capture.cases,
        opportunities,
        &capture.prompts,
        capture.chunk,
        capture.row_ceiling,
        &group[0].declaration,
        None,
        schedule_for,
    )?;
    batch.population_indices = group.iter().map(|p| p.original_index).collect();
    batch.algorithm_universe = group[0].universe.clone();
    Ok((batch, filler_counts))
}

fn selected_calls<'a>(
    reference: &'a Value,
    choice: GeometryPlan,
) -> AuditResult<(&'a [Value], Value)> {
    match choice {
        GeometryPlan::CurrentFull => {
            let demand = field(reference, "independent_current_demand")?;
            let calls = field(demand, "calls")?
                .as_array()
                .context("current calls not array")?;
            Ok((
                calls,
                json!({"kind":"current_full", "independent_full_visits": field(demand, "full_required_visits")?,
                "original_limit": reference.pointer("/audit/maximum_visits"),
                "original_shared_exhausted": reference.pointer("/audit/exhausted"),
                "scope":"Independent MAX geometry demand; never a reset or refund of original work."}),
            ))
        }
        GeometryPlan::AnchoredReadinessV2WidthCostV1 => {
            let candidate = field(reference, "cold_geometry_candidate")?;
            ensure!(
                field(candidate, "geometry_kernel")?.as_str()
                    == Some("anchored_readiness_v2_width_cost_v1"),
                "candidate geometry kind differs"
            );
            let calls = candidate
                .pointer("/independent_max/calls")
                .and_then(Value::as_array)
                .context("candidate independent calls absent")?;
            Ok((
                calls,
                json!({"kind":"anchored_readiness_v2_width_cost_v1",
                "shared_original_budget": field(candidate, "shared_original_budget")?,
                "scope":"Only certified independent MAX final cases enter plan arithmetic; shared-budget outcome remains separate."}),
            ))
        }
    }
}

pub(super) fn evaluate(
    capture: &Capture,
    reference: &Value,
    options: &Options,
    available: SelectionCapacity,
) -> AuditResult<Value> {
    let (calls, geometry) = selected_calls(reference, options.geometry_plan)?;
    ensure!(
        calls.len() == capture.populations.len(),
        "candidate call count differs"
    );
    ensure!(
        !options.population_groups.is_empty() && options.population_groups.len() <= calls.len(),
        "group count outside original population bound"
    );
    ensure!(
        options.required_populations.len() <= calls.len(),
        "required population list exceeds original count"
    );
    let original_batches = field(&capture.selection, "batches")?
        .as_array()
        .context("batches not array")?;
    let original_populations = field(&capture.selection, "populations")?
        .as_array()
        .context("populations not array")?;
    let mut mentioned = Vec::new();
    for indices in &options.population_groups {
        ensure!(
            !indices.is_empty() && indices.len() <= calls.len(),
            "empty or oversized group"
        );
        for &index in indices {
            ensure!(
                index < original_populations.len() && !mentioned.contains(&index),
                "population outside capture or repeated across/within groups"
            );
            mentioned.push(index);
        }
    }
    let mut plans = Vec::new();
    let mut opportunities = vec![
        CaseOpportunity {
            population: CasePopulation::Unknown {
                known_alternatives: Vec::new()
            },
            minimum_fresh_members: 0
        };
        capture.cases.len()
    ];
    for (ordinal, (captured, call)) in capture.populations.iter().zip(calls).enumerate() {
        ensure!(
            decode::<usize>(call, "ordinal")? == ordinal
                && decode::<usize>(call, "population_index")? == captured.index
                && field(call, "original_case_indices")? == field(&captured.matrix, "cases")?
                && field(call, "mandatory_anchor_indices")?
                    == field(&captured.matrix, "mandatory_anchors")?,
            "candidate original matrix identity differs"
        );
        // An unselected candidate need not become complete. The explicit
        // joint plan, not the size of the captured inventory, defines this audit.
        if !mentioned.contains(&captured.index) {
            continue;
        }
        ensure!(
            decode::<bool>(call, "complete")? && field(call, "error")?.is_null(),
            "incomplete candidate population {}",
            captured.index
        );
        if matches!(
            options.geometry_plan,
            GeometryPlan::AnchoredReadinessV2WidthCostV1
        ) {
            ensure!(
                decode::<bool>(call, "span_verified")?,
                "selected candidate lacks original full-matrix span certification"
            );
        }
        let ids = decode::<Vec<usize>>(call, "final_selected_cases")?;
        checked_cases(&ids, &captured.matrix, capture.cases.len())?;
        let expected: Vec<_> = original_batches
            .iter()
            .filter(|b| {
                decode::<Vec<usize>>(b, "population_indices").is_ok_and(|v| v == [captured.index])
            })
            .collect();
        ensure!(expected.len() == 1, "candidate original batch not unique");
        for index in decode::<Vec<usize>>(&captured.matrix, "cases")? {
            let opportunity = opportunities
                .get_mut(index)
                .context("candidate case outside original table")?;
            ensure!(
                opportunity.minimum_fresh_members == 0
                    || matches!(&opportunity.population, CasePopulation::Unique(key) if key == &captured.key),
                "case given two different family floors"
            );
            *opportunity = CaseOpportunity {
                population: CasePopulation::Unique(captured.key.clone()),
                minimum_fresh_members: 1,
            };
        }
        plans.push(PopulationPlan {
            original_index: captured.index,
            key: captured.key.clone(),
            cases: ids,
            declaration: declaration_for(capture, captured, expected[0])?,
            universe: expected[0]
                .get("algorithm_universe")
                .map(|v| serde_json::from_value(v.clone()))
                .transpose()?,
        });
    }
    let mut groups = Vec::new();
    for indices in &options.population_groups {
        ensure!(
            !indices.is_empty() && indices.len() <= calls.len(),
            "empty or oversized group"
        );
        let mut group = Vec::new();
        for &index in indices {
            let plan = plans
                .iter()
                .find(|p| p.original_index == index)
                .context("group population outside original capture")?;
            group.push(plan);
        }
        group_is_compatible(&group, &capture.cases)?;
        groups.push(group);
    }
    for (index, population) in original_populations.iter().enumerate() {
        if decode::<bool>(population, "scheduled")? {
            ensure!(
                mentioned.contains(&index),
                "candidate omitted originally scheduled population {index}"
            );
        }
    }
    for &index in &options.required_populations {
        ensure!(
            mentioned.contains(&index),
            "candidate omitted required population {index}"
        );
    }
    let maximum_target = groups
        .iter()
        .flatten()
        .map(|p| {
            p.declaration
                .settings
                .max_rank
                .checked_add(p.declaration.settings.min_fit_redundancy)
                .map(|n| n.max(p.declaration.settings.min_phase_samples))
        })
        .collect::<Option<Vec<_>>>()
        .context("member target overflow")?
        .into_iter()
        .max()
        .context("no member targets")?;
    ensure!(
        options.maximum_filler_members_per_cycle <= maximum_target,
        "filler sweep exceeds original member requirement bound"
    );
    let source_limit = match &capture
        .config
        .scheduler
        .slo
        .cost_observation
        .live_structured_calibration
    {
        ferrum_types::SloLiveStructuredCalibration::AutomaticV1 { settings } => {
            settings.maximum_retained_generations.get()
        }
        _ => anyhow::bail!("not automatic"),
    };
    let mut variants = Vec::new();
    for target in 0..=options.maximum_filler_members_per_cycle {
        let mut total = SelectionCapacity::default();
        let mut schedule_fits = true;
        let mut sources = Vec::new();
        for group in &groups {
            let (batch, filler_counts) =
                planned_group(capture, group, &opportunities, target, options.schedule)?;
            total.charge(&batch)?;
            schedule_fits &= batch.schedule_within_capacity
                && batch.maximum_anchor_span
                    <= *batch.schedule.phase_min_offered.iter().min().unwrap();
            sources.push(
                json!({"original_population_indices": batch.population_indices,
                "independent_exact_families": group.iter().map(|p| &p.key).collect::<Vec<_>>(),
                "geometry_representatives": group.iter().map(|p| &p.cases).collect::<Vec<_>>(),
                "additional_fresh_cohorts_per_cycle": filler_counts, "plan": batch}),
            );
        }
        let requests_fit = total.requests <= available.requests;
        let actions_fit = total.execution_actions <= available.execution_actions;
        let rows_fit = total.declared_offer_rows <= available.declared_offer_rows;
        let sources_fit = groups.len() <= source_limit;
        variants.push(json!({"minimum_members_per_family_cycle_target": target, "sources": sources,
            "totals":{"sources": groups.len(), "requests":total.requests,"execution_actions":total.execution_actions,"declared_offer_rows":total.declared_offer_rows},
            "fit_checks":{"requests":requests_fit,"execution_actions":actions_fit,"declared_offer_rows":rows_fit,"source_count":sources_fit,"schedule_and_anchors":schedule_fits},
            "declared_work_and_source_count_fit": requests_fit && actions_fit && rows_fit && sources_fit && schedule_fits}));
    }
    Ok(
        json!({"options":options,"geometry":geometry,"original_scheduled_and_explicit_required_populations_preserved":true,
        "variants":variants,
        "limits":["Compatible metadata only: no original raw recipe closure, selector priority or live packing authority is reconstructed.",
            "Each filler occurrence is a newly declared cohort. It retains the original floor and identity; setup is deduplicated only by the production acquisition key.",
            "Source8 block closure, serialized bytes, retained memory, actual F/R/Q, 120-second runtime and ordinary adoption are not measured.",
            "Cold fallback currently rebuilds the original complete-cycle schedule and is not validated by these candidate costs.",
            "The original full-geometry, complete-cycle independent plans remain above as the unchanged control."]}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capture_plan_each_offer_preserves_fresh_floors_and_recomputes_member_capacity() {
        let settings = StructuredSettingsV2::default();
        let starts = [0, 1, 3];
        let ends = [1, 3, 6];
        let (old, anchor) = budget::startup_schedule(&starts, &ends, &settings).unwrap();
        let (new, actual_anchor) = each_offer_schedule(&starts, &ends, &settings).unwrap();
        assert_eq!(anchor, actual_anchor);
        assert_eq!(old.phase_min_offered, new.phase_min_offered);
        assert_eq!(old.min_members, new.min_members);
        assert_eq!(new.block_offered, 1);
        let mut numerical = settings;
        numerical.max_phase_samples = *new.maximum_phase_members.iter().max().unwrap();
        new.validate(&numerical).unwrap();
        for marked in [&[0][..], &[1][..], &[0, 2][..]] {
            for (phase, &minimum) in new.min_members.iter().enumerate() {
                let offers = budget::fresh_span(&starts, &ends, marked, minimum)
                    .unwrap()
                    .max(new.phase_min_offered[phase]);
                for cut in 0..6 {
                    let fresh = (0..=(cut + offers) / 6)
                        .flat_map(|cycle| marked.iter().map(move |&i| (cycle, i)))
                        .filter(|&(cycle, i)| {
                            cycle * 6 + starts[i] > cut && cycle * 6 + ends[i] <= cut + offers
                        })
                        .count();
                    assert!(fresh >= minimum);
                }
            }
        }
        // Filler changes the cycle. Old minimum offers cannot stand in for the
        // new worst-case gap to a sparse family's only original anchor.
        let (old, _) = each_offer_schedule(&[0, 1], &[1, 2], &numerical).unwrap();
        let starts: Vec<_> = (0..10).collect();
        let ends: Vec<_> = (1..=10).collect();
        let (expanded, span) = each_offer_schedule(&starts, &ends, &numerical).unwrap();
        assert!(span > old.phase_min_offered[0]);
        assert!(expanded.phase_min_offered[0] >= span);
    }

    #[test]
    fn capture_plan_filler_keeps_exact_family_original_cases_and_native_setup_accounting() {
        use ferrum_interfaces::execution_cost::CostProductOutput;
        let a = populations::tests::input(1, 3, CostProductOutput::GreedyToken, false, true);
        let b = populations::tests::input(1, 3, CostProductOutput::FullLogits, false, true);
        let key_a = CheckedPopulationKey::NumericalFamily(a.numerical_family_key().unwrap());
        let key_b = CheckedPopulationKey::NumericalFamily(b.numerical_family_key().unwrap());
        assert_ne!(key_a, key_b);
        let mut cases: Vec<_> = [1, 8, 1]
            .into_iter()
            .enumerate()
            .map(|(i, width)| Case {
                product: if i == 2 {
                    OpportunityProduct::Full
                } else {
                    OpportunityProduct::Greedy
                },
                template: 0,
                width,
                maximum_output: NonZeroUsize::new(4).unwrap(),
                release_generated: 2,
                suffix_tokens: 2,
                preset: SloAutomaticCostProbeSamplingPresetV1::Configured,
                prefix: PrefixKind::Pending,
                route: CalibrationDecodeRoute::Actual,
                reset: false,
                acquisition: None,
            })
            .collect();
        for case in &mut cases {
            let chunk = case
                .prefill_chunk(NonZeroU32::new(16).unwrap(), None, case.width)
                .unwrap();
            case.acquisition =
                Some(work::acquisition_from_capture_for_test(case, 7, 6, chunk, [7; 32]).unwrap());
        }
        let mut opportunities: Vec<_> = [key_a.clone(), key_a.clone(), key_b.clone()]
            .into_iter()
            .map(|key| CaseOpportunity {
                population: CasePopulation::Unique(key),
                minimum_fresh_members: 1,
            })
            .collect();
        let expanded =
            expand_cycle(&key_a, &[0, 1], 4, &cases, &opportunities, &[7], 16, None).unwrap();
        assert_eq!(expanded, [0, 1, 0, 0]);
        assert_eq!(
            work::setup_for_indices(&cases, &[0, 1]).unwrap(),
            work::setup_for_indices(&cases, &expanded).unwrap()
        );
        assert_eq!(opportunities[2].minimum_fresh_members, 1);
        assert!(
            matches!(&opportunities[2].population, CasePopulation::Unique(key) if key == &key_b)
        );
        let population = SelectedPopulation {
            key: key_a.clone(),
            representative_case_indices: expanded,
            maximum_anchor_span: 0,
            scheduled: false,
            batch_index: None,
            input_geometry: None,
        };
        let declaration =
            population::declaration(&Default::default(), populations::tests::domain()).unwrap();
        let original = batch_plan(
            &[0],
            std::slice::from_ref(&population),
            &cases,
            &opportunities,
            &[7],
            16,
            None,
            &declaration,
        )
        .unwrap();
        let injected = batch_plan_with_schedule(
            &[0],
            std::slice::from_ref(&population),
            &cases,
            &opportunities,
            &[7],
            16,
            None,
            &declaration,
            None,
            budget::startup_schedule,
        )
        .unwrap();
        assert_eq!(
            serde_json::to_value(original).unwrap(),
            serde_json::to_value(injected).unwrap()
        );
        let each = batch_plan_with_schedule(
            &[0],
            &[population],
            &cases,
            &opportunities,
            &[7],
            16,
            None,
            &declaration,
            None,
            each_offer_schedule,
        )
        .unwrap();
        assert!(each.schedule_within_capacity);
        assert_eq!(
            each.input_opportunities
                .minimum_input_family_opportunities_per_cycle,
            4
        );
        assert_eq!(each.schedule.min_members, [36, 8, 8]);
        opportunities[0].minimum_fresh_members = 0;
        assert!(expand_cycle(&key_a, &[0, 1], 4, &cases, &opportunities, &[7], 16, None).is_err());
        opportunities[0] = CaseOpportunity {
            population: CasePopulation::Unique(key_b),
            minimum_fresh_members: 1,
        };
        assert!(expand_cycle(&key_a, &[0, 1], 4, &cases, &opportunities, &[7], 16, None).is_err());
    }
}
