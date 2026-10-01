//! Bounded input inventory, visited in a frozen declared-work order.
//! Freeze the final algorithm universe and select once across all declared
//! inputs before reserving any numerical source. Inventory is not evidence.
use super::*;
use populations::CheckedPopulationKey;
use sha2::{Digest, Sha256};

fn declared_input_priority(
    input: &PreparedProbeInputs,
    cases: &[Case],
    template: usize,
) -> Result<u8> {
    // Reuse selection's input-only ordering. GreedyLength explicitly removes
    // EOS/stop in the template adapter. Configured needs its own explicit
    // ignore-EOS/no-stop declaration to prove this before live inventory.
    if cases.iter().any(|case| {
        case.template == template
            && case.preset == SloAutomaticCostProbeSamplingPresetV1::GreedyLength
    }) {
        return Ok(selection::input_priority(false, false));
    }
    // Inspect only the original declared termination fields. Ignored prompt,
    // API body and stop strings are skipped without allocating an owned tree.
    #[derive(Default, serde::Deserialize)]
    struct Metadata {
        ferrum_ignore_eos: Option<bool>,
    }
    #[derive(serde::Deserialize)]
    struct Sampling {
        #[serde(default, deserialize_with = "nonempty_sequence")]
        stop_sequences: bool,
    }
    #[derive(serde::Deserialize)]
    struct Request {
        #[serde(default)]
        metadata: Metadata,
        sampling_params: Sampling,
    }
    let request: Request = serde_json::from_slice(input.templates[template].serialized_request())
        .map_err(|e| error(format!("input termination declaration: {e}")))?;
    let has_early_policy =
        request.metadata.ferrum_ignore_eos != Some(true) || request.sampling_params.stop_sequences;
    let has_early_opportunity = cases.iter().any(|case| {
        case.template == template
            && case.maximum_output.get() > case.release_generated.saturating_add(1)
    });
    Ok(selection::input_priority(
        has_early_policy,
        has_early_opportunity,
    ))
}

fn nonempty_sequence<'de, D: serde::Deserializer<'de>>(
    deserializer: D,
) -> std::result::Result<bool, D::Error> {
    struct Nonempty;
    impl<'de> serde::de::Visitor<'de> for Nonempty {
        type Value = bool;
        fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str("the declared stop sequence array")
        }
        fn visit_seq<A: serde::de::SeqAccess<'de>>(
            self,
            mut values: A,
        ) -> std::result::Result<bool, A::Error> {
            let mut nonempty = false;
            while values.next_element::<serde::de::IgnoredAny>()?.is_some() {
                nonempty = true;
            }
            Ok(nonempty)
        }
    }
    deserializer.deserialize_seq(Nonempty)
}

pub(in crate::continuous_engine::inner::calibration) struct CheckedInputCursor {
    input: Option<PreparedProbeInputs>,
    maximum_retained_bytes: usize,
    maximum_sources: usize,
    cases: Vec<Case>,
    skipped: usize,
    unavailable: Vec<PreparedPrefixUnavailable>,
    inventory: CheckedCaseInventory,
    template_order: Vec<usize>,
    next_template: usize,
    selection_finished: bool,
    parent: Box<serde_json::value::RawValue>,
    parent_sha256: [u8; 32],
    cold_algorithm_seed: Option<ferrum_scheduler::implementations::continuous::cost_model::structured_v2::DeclaredAlgorithmUniverseV1>,
}

impl CheckedInputCursor {
    pub fn new(input: PreparedProbeInputs) -> Result<Self> {
        let (cases, skipped, unavailable, inventory_limit) = prepare_cases(&input)?;
        let headers = cases
            .len()
            .checked_mul(
                std::mem::size_of::<CaseOpportunity>()
                    + std::mem::size_of::<Vec<selection::CheckedInputFacts>>()
                    + std::mem::size_of::<inventory::InventoryGap>(),
            )
            .and_then(|n| {
                n.checked_add(input.templates.len().checked_mul(
                    std::mem::size_of::<usize>()
                        + std::mem::size_of::<(usize, usize, usize)>()
                        + std::mem::size_of::<u8>(),
                )?)
            })
            .and_then(|n| {
                n.checked_add(
                    std::mem::size_of::<Self>() + std::mem::size_of::<CheckedCaseInventory>(),
                )
            })
            .filter(|n| *n < inventory_limit)
            .ok_or_else(|| error("input cursor headers exceed retained capacity"))?;
        let _ = headers; // Authorization precedes every header allocation below.
        let mut template_order: Vec<_> = (0..input.templates.len()).collect();
        let mut priorities = Vec::with_capacity(input.templates.len());
        for template in 0..input.templates.len() {
            priorities.push(declared_input_priority(&input, &cases, template)?);
        }
        // Only declared input work participates in ordering. Neither observed
        // time, EOS, qualification nor publication success is available here.
        let mut work = vec![(0usize, 0usize, 0usize); input.templates.len()];
        for case in &cases {
            let entry = &mut work[case.template];
            entry.0 = entry
                .0
                .checked_add(
                    input.prompts[case.template]
                        .checked_add(case.maximum_output.get() - 1)
                        .and_then(|n| n.checked_mul(case.width))
                        .ok_or_else(|| error("input unit declared token work overflow"))?,
                )
                .ok_or_else(|| error("input unit declared token work overflow"))?;
            entry.1 = entry
                .1
                .checked_add(
                    case.waves_with_row_ceiling(
                        input.prompts[case.template],
                        input.chunk.get() as usize,
                        input.prefill_row_ceiling,
                    )?
                    .1,
                )
                .ok_or_else(|| error("input unit declared wave work overflow"))?;
            entry.2 = entry
                .2
                .checked_add(case.width)
                .ok_or_else(|| error("input unit declared requests overflow"))?;
        }
        template_order.sort_unstable_by_key(|&i| (priorities[i], work[i], i));
        let inventory = CheckedCaseInventory {
            opportunities: (0..cases.len())
                .map(|_| CaseOpportunity {
                    population: CasePopulation::Unknown {
                        known_alternatives: Vec::new(),
                    },
                    minimum_fresh_members: 0,
                })
                .collect(),
            inputs: (0..cases.len()).map(|_| Vec::new()).collect(),
            algorithm_inputs: Vec::new(),
            gaps: Vec::with_capacity(cases.len()),
            charge: ProbePreflightCharge::default(),
        };
        let available = inventory_limit
            .checked_sub(
                inventory
                    .retained_payload_bytes()
                    .and_then(|n| {
                        n.checked_add(template_order.capacity() * std::mem::size_of::<usize>())
                    })
                    .and_then(|n| {
                        n.checked_add(
                            work.capacity() * std::mem::size_of::<(usize, usize, usize)>(),
                        )
                    })
                    .and_then(|n| n.checked_add(priorities.capacity() * std::mem::size_of::<u8>()))
                    .ok_or_else(|| error("input cursor capacity overflow"))?,
            )
            .filter(|n| *n > 0)
            .ok_or_else(|| error("input cursor capacity exhausted"))?;
        let parent = manifest::freeze_inventory_inputs(&input, &cases, &template_order, available)?;
        let parent_sha256 = Sha256::digest(parent.get().as_bytes()).into();
        let cursor = Self {
            maximum_retained_bytes: input.population.maximum_retained_numeric_bytes,
            maximum_sources: input.settings.maximum_probe_requests.get().min(65_536),
            input: Some(input),
            cases,
            skipped,
            unavailable,
            inventory,
            template_order,
            next_template: 0,
            cold_algorithm_seed: None,
            selection_finished: false,
            parent,
            parent_sha256,
        };
        cursor.remaining_bytes()?;
        Ok(cursor)
    }

    pub fn maximum_sources(&self) -> usize {
        // Every complete source consumes at least one original request.
        self.maximum_sources
    }

    pub fn pending_templates(&self) -> usize {
        self.template_order.len() - self.next_template
    }

    pub fn pending_input_units(&self) -> usize {
        self.pending_templates() + usize::from(!self.selection_finished)
    }

    #[cfg(test)]
    pub fn declared_template_order(&self) -> &[usize] {
        &self.template_order
    }

    #[cfg(test)]
    pub fn retained_input_buffer_bytes(&self) -> usize {
        self.input
            .as_ref()
            .map_or(0, |input| input.retained_payload_bytes().unwrap())
            + self.cases.capacity() * std::mem::size_of::<Case>()
            + self.unavailable.capacity() * std::mem::size_of::<PreparedPrefixUnavailable>()
    }

    pub fn take_algorithm_seed(&mut self) -> Option<ferrum_scheduler::implementations::continuous::cost_model::structured_v2::DeclaredAlgorithmUniverseV1>{
        self.cold_algorithm_seed.take()
    }

    fn retained_payload_bytes(&self) -> Option<usize> {
        self.input
            .as_ref()
            .map_or(Some(0), PreparedProbeInputs::retained_payload_bytes)?
            .checked_add(
                self.cases
                    .capacity()
                    .checked_mul(std::mem::size_of::<Case>())?,
            )?
            .checked_add(
                self.unavailable
                    .capacity()
                    .checked_mul(std::mem::size_of::<PreparedPrefixUnavailable>())?,
            )?
            .checked_add(self.inventory.retained_payload_bytes()?)?
            .checked_add(
                self.template_order
                    .capacity()
                    .checked_mul(std::mem::size_of::<usize>())?,
            )?
            .checked_add(self.parent.get().len())?
            .checked_add(
                self.cold_algorithm_seed
                    .as_ref()
                    .map_or(Some(0), |u| u.retained_payload_bytes())?,
            )?
            .checked_add(std::mem::size_of::<Self>())
    }

    fn remaining_bytes(&self) -> Result<usize> {
        self.maximum_retained_bytes
            .checked_sub(
                self.retained_payload_bytes()
                    .ok_or_else(|| error("input cursor capacity overflow"))?,
            )
            .filter(|n| *n > 0)
            .ok_or_else(|| error("input cursor retained capacity exhausted"))
    }

    async fn collect_all(
        &mut self,
        session: &mut CalibrationSession,
        budget: &mut ProbeExecutionBudget,
    ) -> Result<()> {
        let input = self
            .input
            .as_ref()
            .ok_or_else(|| error("input cursor already consumed"))?;
        while self.next_template < self.template_order.len() {
            let template = self.template_order[self.next_template];
            budget.require_selection_time()?;
            let count = self
                .cases
                .iter()
                .filter(|case| case.template == template)
                .count();
            let scratch = count
                .checked_mul(
                    std::mem::size_of::<usize>()
                        + std::mem::size_of::<Case>()
                        + std::mem::size_of::<CheckedPopulationKey>(),
                )
                .ok_or_else(|| error("input unit scratch capacity overflow"))?;
            let available = self
                .remaining_bytes()?
                .checked_sub(scratch)
                .filter(|n| *n > 0)
                .ok_or_else(|| error("input unit scratch capacity exhausted"))?;
            let mut indices = Vec::with_capacity(count);
            indices.extend(
                self.cases
                    .iter()
                    .enumerate()
                    .filter_map(|(i, case)| (case.template == template).then_some(i)),
            );
            let mut group = Vec::with_capacity(count);
            group.extend(indices.iter().map(|&i| self.cases[i].clone()));
            let captured =
                Box::pin(collect_ready(session, input, &group, budget, available)).await?;
            // Merge while the complete capture still owns its backing. The
            // new raw-input clones and any pool growth share the same limit.
            let pool_external = self
                .retained_payload_bytes()
                .and_then(|n| n.checked_sub(self.inventory.retained_payload_bytes()?))
                .and_then(|n| n.checked_add(captured.retained_payload_bytes()?))
                .and_then(|n| n.checked_add(scratch))
                .ok_or_else(|| error("input algorithm pool merge capacity overflow"))?;
            for input in &captured.algorithm_inputs {
                self.inventory.retain_algorithm_input(
                    input,
                    self.maximum_retained_bytes,
                    pool_external,
                )?;
            }
            drop(captured.algorithm_inputs);
            if self
                .inventory
                .gaps
                .len()
                .checked_add(captured.gaps.len())
                .is_none_or(|n| n > self.inventory.gaps.capacity())
            {
                return Err(error("input inventory gap capacity differs"));
            }
            for ((&global, opportunity), facts) in indices
                .iter()
                .zip(captured.opportunities)
                .zip(captured.inputs)
            {
                self.inventory.opportunities[global] = opportunity;
                self.inventory.inputs[global] = facts;
            }
            self.inventory
                .gaps
                .extend(captured.gaps.into_iter().map(|mut gap| {
                    gap.case_index = indices[gap.case_index];
                    gap
                }));
            self.inventory.charge = budget.preflight_charge();
            self.next_template += 1;
            drop(group);
            self.remaining_bytes()?;
            tracing::info!(
                rendered_template = template,
                original_template = input.original_template_indices[template],
                output = ?input.templates[template].output(),
                completed_templates = self.next_template,
                pending_inventory_templates = self.pending_templates(),
                preflight = ?budget.preflight_charge(),
                "Automatic declared input inventoried before global numerical selection"
            );
        }

        Ok(())
    }

    fn retain_completed_inventory_seed(&mut self, budget: &ProbeExecutionBudget) -> Result<()> {
        // A complete prior input still supplies valid declared algorithms when
        // another template fails. It supplies no partial source membership.
        // Preserve the original clock and shared simultaneous retained limit.
        budget.require_selection_time()?;
        let input = self
            .input
            .as_ref()
            .ok_or_else(|| error("input cursor already consumed"))?;
        let external = self
            .retained_payload_bytes()
            .and_then(|n| n.checked_sub(self.inventory.retained_payload_bytes()?))
            .ok_or_else(|| error("partial inventory seed retained capacity overflow"))?;
        let maximum = self
            .maximum_retained_bytes
            .checked_sub(external)
            .ok_or_else(|| error("partial inventory seed retained capacity exhausted"))?;
        self.cold_algorithm_seed =
            universe::freeze_declared_algorithms(&self.inventory, &input.population, maximum)?;
        Ok(())
    }

    pub async fn next(
        &mut self,
        session: &mut CalibrationSession,
        budget: &mut ProbeExecutionBudget,
    ) -> Result<Option<super::super::super::series::PreparedProbeSeries>> {
        if self.selection_finished {
            return Ok(None);
        }

        if let Err(reason) = self.collect_all(session, budget).await {
            self.selection_finished = true;
            let seed = self.retain_completed_inventory_seed(budget);
            tracing::warn!(
                error = %reason,
                seed_error = ?seed.err(),
                seed_available = self.cold_algorithm_seed.is_some(),
                completed_templates = self.next_template,
                pending_inventory_templates = self.pending_templates(),
                preflight = ?budget.preflight_charge(),
                "Global input inventory incomplete; no numerical source reserved"
            );
            return Err(reason);
        }

        // A single terminal selection owns the complete inventory. No prefix
        // copies, geometry rescans or outcome-dependent budget reuse occur.
        self.selection_finished = true;
        budget.require_selection_time()?;
        let inventory = std::mem::replace(
            &mut self.inventory,
            CheckedCaseInventory {
                opportunities: Vec::new(),
                inputs: Vec::new(),
                algorithm_inputs: Vec::new(),
                gaps: Vec::new(),
                charge: budget.preflight_charge(),
            },
        );
        // Terminal ownership moves into the plan. The cursor retains only
        // its manifest, ordering metadata and seed; no duplicate template or
        // case buffers remain beside the owned inventory during selection.
        let mut input = self
            .input
            .take()
            .ok_or_else(|| error("input cursor already consumed"))?;
        let cases = std::mem::take(&mut self.cases);
        let unavailable = std::mem::take(&mut self.unavailable);
        let external = self
            .retained_payload_bytes()
            .ok_or_else(|| error("global input cursor retained capacity overflow"))?;
        let owned = input
            .retained_payload_bytes()
            .and_then(|n| n.checked_add(cases.capacity().checked_mul(std::mem::size_of::<Case>())?))
            .and_then(|n| {
                n.checked_add(
                    unavailable
                        .capacity()
                        .checked_mul(std::mem::size_of::<PreparedPrefixUnavailable>())?,
                )
            })
            .ok_or_else(|| error("global input selection capacity overflow"))?;
        let inventory_limit = self
            .maximum_retained_bytes
            .checked_sub(external)
            .and_then(|n| n.checked_sub(owned))
            .filter(|n| *n > 0)
            .ok_or_else(|| error("global input selection capacity exhausted"))?;
        inventory
            .retained_payload_bytes()
            .filter(|n| *n <= inventory_limit)
            .ok_or_else(|| error("global input inventory exceeds shared retained capacity"))?;
        input.external_retained_bytes = external;
        let plan = freeze_inventory(
            session,
            input,
            cases,
            self.skipped,
            unavailable,
            inventory,
            budget,
            inventory_limit,
            true,
            None,
            None,
            Some(&mut self.cold_algorithm_seed),
        )?;
        let Some(mut plan) = plan else {
            return Ok(None);
        };
        // The retained seed participates in the simultaneous source/manifest
        // peak. The original deadline and all ledgers survive this boundary.
        let external = self
            .retained_payload_bytes()
            .ok_or_else(|| error("global input cursor retained capacity overflow"))?;
        let available = self
            .maximum_retained_bytes
            .checked_sub(external)
            .and_then(|n| n.checked_sub(plan.declaration.retained_payload_bytes()?))
            .and_then(|n| n.checked_sub(plan.execution.retained_payload_bytes()?))
            .filter(|n| *n > 0)
            .ok_or_else(|| error("global input manifest capacity exhausted"))?;
        plan.declaration.cohort_manifest_payload = manifest::freeze_global_inventory(
            &self.parent,
            self.parent_sha256,
            &self.template_order,
            &plan.declaration.cohort_manifest_payload,
            available,
        )?;
        plan.declaration
            .validate()
            .map_err(|e| error(format!("global input declaration: {e}")))?;
        // Reserve the complete series once. No measured cohort has run, so
        // shorter sources cannot consume the allowance before long candidates.
        budget.reserve_selected_source(
            plan.audit().planned_requests,
            plan.audit().serial_wave_bound,
        )?;
        plan.into_series().map(Some)
    }
}
