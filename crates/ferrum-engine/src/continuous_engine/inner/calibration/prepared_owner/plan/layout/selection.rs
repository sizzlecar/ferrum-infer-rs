//! Input-only representative selection. No observed wall, fitted parameter or
//! measured terminal outcome selects these original cases. Scheduled input
//! anchors are opportunities; the original collector still qualifies every
//! independent numerical phase and may reject an incomplete actual population.
use super::*;
use ferrum_interfaces::execution_cost::{
    HostContentDomainV1, HostCostPolicyV2, HostTerminalExpectationV1,
};
use ferrum_scheduler::implementations::continuous::cost_model::structured_v2::{
    OwnerBlockScheduleV1, StructuredOwnerKeyV2, StructuredPopulationPolicyV1, StructuredQueryV2,
    StructuredUnknownV2,
};
use populations::{member_groups, CaseOpportunity, CasePopulation, CheckedPopulationKey};
#[cfg(test)]
mod capture_plan_audit;
mod composition;
mod grouping;
mod input_geometry;
mod memory;
mod priority;
mod scoped_inputs;
use ferrum_scheduler::implementations::continuous::cost_model::structured_v2::StructuredInputGeometryWorkV1;
use input_geometry::{InputGeometryAudit, InputGeometryCharge};
use priority::CoveragePriority;

#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub(super) struct CheckedInputFacts {
    pub axes: Vec<f64>,
    owner: StructuredOwnerKeyV2,
    // Bounded checked policy identity for opportunity ordering across widths.
    // The exact row-multiplicity owner remains the population/channel identity.
    homogeneous_host_policy: Option<HostCostPolicyV2>,
    family: Option<ferrum_scheduler::implementations::continuous::cost_model::structured_v2::NumericalFamilyKeyV1>,
    // Actual input-row facts, not an invented forecast pending range.
    branches: [bool; 7],
    /// Original checked metadata, retained only while cold inventory can grow.
    /// Its algorithms cannot be reconstructed from owner/family digests.
    #[serde(skip)]
    pub(super) original: Option<std::sync::Arc<ferrum_scheduler::implementations::continuous::cost_model::structured_v2::StructuredInputV2>>,
}

impl CheckedInputFacts {
    pub(super) fn retained_metadata_bytes(&self) -> Option<usize> {
        std::mem::size_of::<Self>().checked_add(
            self.axes
                .capacity()
                .checked_mul(std::mem::size_of::<f64>())?,
        )
    }

    pub fn retained_payload_bytes(&self) -> Option<usize> {
        self.retained_metadata_bytes()?
            .checked_add(self.original.as_ref().map_or(Some(0), |original| {
                original
                    .retained_payload_bytes()?
                    .checked_add(2 * std::mem::size_of::<usize>())
            })?)
    }

    pub(super) fn key(&self, policy: StructuredPopulationPolicyV1) -> CheckedPopulationKey {
        match (policy, self.family) {
            (StructuredPopulationPolicyV1::HomogeneousOrdinaryDecodeV1, Some(key)) => {
                CheckedPopulationKey::NumericalFamily(key)
            }
            _ => CheckedPopulationKey::ExactOwner(self.owner.clone()),
        }
    }
}

/// Keep only the facts needed to preserve the checked inventory's numeric
/// endpoints and physical host branches. Unsupported ordinary-family scope is
/// the sole exact-owner fallback; malformed/unbound physical inputs propagate.
pub(super) fn input_facts(
    query: &StructuredQueryV2,
) -> std::result::Result<CheckedInputFacts, StructuredUnknownV2> {
    let input = query.input();
    facts_from_input(input, std::sync::Arc::new(input.clone()))
}

pub(super) fn facts_from_input(
    input: &ferrum_scheduler::implementations::continuous::cost_model::structured_v2::StructuredInputV2,
    original: std::sync::Arc<
        ferrum_scheduler::implementations::continuous::cost_model::structured_v2::StructuredInputV2,
    >,
) -> std::result::Result<CheckedInputFacts, StructuredUnknownV2> {
    if input.physical_domain_signature().is_none() {
        return Err(StructuredUnknownV2::WrongDomain);
    }
    let family = match input.numerical_family_key() {
        Ok(key) => Some(key),
        Err(StructuredUnknownV2::UnsupportedScope) => None,
        Err(reason) => return Err(reason),
    };
    if input.regression_axes().is_empty()
        || input
            .regression_axes()
            .iter()
            .any(|x| !x.is_finite() || *x < 0. || *x > (1u64 << 53) as f64 || x.fract() != 0.)
    {
        return Err(StructuredUnknownV2::InvalidInput);
    }
    let mut axes = Vec::new();
    axes.try_reserve_exact(input.regression_axes().len())
        .map_err(|_| StructuredUnknownV2::Capacity)?;
    axes.extend_from_slice(input.regression_axes());
    let mut branches = [false; 7];
    let mut homogeneous_host_policy = input
        .physical_host_rows()
        .first()
        .map(|row| row.installed_policy);
    for row in input.physical_host_rows() {
        if homogeneous_host_policy != Some(row.installed_policy) {
            homogeneous_host_policy = None;
        }
        branches[0] |= row.pending_decoded_utf8;
        branches[1] |= row.terminal_expectation == HostTerminalExpectationV1::LengthBoundary;
        branches[2] |= row.terminal_expectation == HostTerminalExpectationV1::NoTokenProduced;
        branches[3] |= row.terminal_expectation == HostTerminalExpectationV1::TokenMayTerminate;
        let early_policy = matches!(row.installed_policy.empirical_content_domain,
            Some(HostContentDomainV1::PlainTextInstalledV2(policy)) if policy.model_eos || policy.user_stop);
        // Same conditions as ChallengeCoverage::validate_branches and
        // physical_envelope::early_capable. They describe an opportunity, never
        // an observed EOS/Stop or continuation result.
        branches[4] |=
            early_policy && row.terminal_expectation != HostTerminalExpectationV1::NoTokenProduced;
        branches[5] |= early_policy
            && row.terminal_expectation == HostTerminalExpectationV1::TokenMayTerminate;
        branches[6] |= row
            .repetition_penalty_bits
            .is_some_and(|bits| bits != 1f32.to_bits());
    }
    Ok(CheckedInputFacts {
        axes,
        owner: input.owner().clone(),
        homogeneous_host_policy,
        family,
        branches,
        original: Some(original),
    })
}

#[derive(Debug, Clone, serde::Serialize)]
pub(in crate::continuous_engine::inner::calibration) enum SelectionGapReason {
    /// A prior input unit already froze this population; this unit introduces
    /// no new input for it. Its original source and obligations stay intact.
    PreviouslyDeclaredPopulation,
    /// Optional local declaration cannot retain checked recipes and complete
    /// selection scratch within the unchanged shared capacity. Raw sources stay.
    CombinationCapacity,
    /// A complete local combination does not fit beside the original scheduled
    /// source prefix. Its originals remain independently scheduled.
    CombinationWorkCapacity,
    /// The immutable input-priority traversal visits this complete population
    /// in another round. No cohort or request allowance is reserved here.
    DeferredInputPriority {
        priority: u8,
        selected: u8,
    },
    UnknownPopulation,
    NoGuaranteedMember,
    NoSingleCheckedInput,
    EarlyTerminalOpportunityMissing,
    /// The original phase must actually observe both early termination and
    /// continuation. No deterministic input-only plan can promise those facts.
    OutcomeDependentEarlyTermination,
    AnchorAfterEarliestFreeze {
        span: usize,
        minimum_offers: usize,
    },
    /// The complete input-derived schedule exceeds native numerical capacity.
    SourceScheduleCapacity,
    /// The prior complete sources already consume the shared startup origin
    /// capacity. Keep this declared opportunity, without collecting a source
    /// that would immediately evict earlier required-policy coverage.
    RetainedSourceCapacity {
        maximum_sources: usize,
    },
    /// Declared input pivots are advisory opportunities, never qualification.
    /// Preserve the original endpoint/branch anchors if this cold pass is bounded.
    InputGeometryUnavailable {
        #[serde(serialize_with = "input_geometry::serialize_reason")]
        reason: StructuredUnknownV2,
        work_exhausted: bool,
    },
    RemainingRequests {
        required: usize,
        remaining: usize,
    },
    RemainingWaves {
        required: usize,
        remaining: usize,
    },
    RemainingOfferRows {
        required: usize,
        remaining: usize,
    },
}

#[derive(Debug, Clone, serde::Serialize)]
pub(in crate::continuous_engine::inner::calibration) struct SelectionGap {
    pub population: Option<CheckedPopulationKey>,
    pub reason: SelectionGapReason,
}

#[derive(Debug, Clone, serde::Serialize)]
pub(in crate::continuous_engine::inner::calibration) struct SelectedPopulation {
    pub key: CheckedPopulationKey,
    pub representative_case_indices: Vec<usize>,
    pub maximum_anchor_span: usize,
    pub scheduled: bool,
    pub batch_index: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_geometry: Option<InputGeometryAudit>,
}

#[derive(Debug, Clone, serde::Serialize)]
pub(in crate::continuous_engine::inner::calibration) struct SelectedBatch {
    pub population_indices: Vec<usize>,
    pub representative_case_indices: Vec<usize>,
    pub input_opportunities: ProbeInputOpportunityBudget,
    pub schedule: OwnerBlockScheduleV1,
    pub schedule_within_capacity: bool,
    pub planned_cycles: usize,
    pub maximum_anchor_span: usize,
    pub requests: usize,
    /// Seed, target prefill and decode input tokens across the complete batch.
    /// This declared upper bound is neither a timing sample nor a latency estimate.
    pub serial_token_work: usize,
    /// Execution actions, including source setup and per-request restores.
    /// Maintenance actions do not advance the schedule's offered clock.
    pub serial_wave_upper_bound: usize,
    /// Source inference rows only; setup inference and transfers are excluded.
    pub declared_offer_row_bound: usize,
    pub scheduled: bool,
    /// Input-only local scope, qualified only by this source's real F/R/Q.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub algorithm_universe: Option<ferrum_scheduler::implementations::continuous::cost_model::structured_v2::DeclaredAlgorithmUniverseV1>,
    /// Original member floors with the final declared numerical family. Cold
    /// preparation must retain this scope rather than restoring raw phase counts.
    #[serde(skip)]
    pub(super) scoped_opportunities: Option<Vec<CaseOpportunity>>,
}

#[derive(Debug, Clone, Default, serde::Serialize)]
pub(in crate::continuous_engine::inner::calibration) struct CheckedSelection {
    /// Original case indices. Every occurrence requires a fresh original
    /// cohort/request identity; these are not reused observations or retries.
    pub execution_case_indices: Vec<usize>,
    pub populations: Vec<SelectedPopulation>,
    pub batches: Vec<SelectedBatch>,
    pub gaps: Vec<SelectionGap>,
    pub requests: usize,
    pub serial_wave_upper_bound: usize,
    pub declared_offer_row_bound: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_geometry: Option<InputGeometryCharge>,
}

/// Independent startup ledgers. Execution maintenance spends actions without
/// creating source inference rows or advancing the numerical offered clock.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(super) struct SelectionCapacity {
    pub requests: usize,
    pub execution_actions: usize,
    pub declared_offer_rows: usize,
}

impl SelectionCapacity {
    fn legacy(requests: usize, waves: usize) -> Self {
        Self {
            requests,
            execution_actions: waves,
            declared_offer_rows: waves,
        }
    }

    fn remaining(self, used: Self) -> Self {
        Self {
            requests: self.requests.saturating_sub(used.requests),
            execution_actions: self
                .execution_actions
                .saturating_sub(used.execution_actions),
            declared_offer_rows: self
                .declared_offer_rows
                .saturating_sub(used.declared_offer_rows),
        }
    }

    fn charge(&mut self, batch: &SelectedBatch) -> Result<()> {
        *self = Self {
            requests: add(self.requests, batch.requests)?,
            execution_actions: add(self.execution_actions, batch.serial_wave_upper_bound)?,
            declared_offer_rows: add(self.declared_offer_rows, batch.declared_offer_row_bound)?,
        };
        Ok(())
    }
}

struct BatchCandidate {
    input_priority: u8,
    coverage: CoveragePriority,
    coverage_round: usize,
    decode_width_tier: usize,
    original_population_index: usize,
    batch: SelectedBatch,
}

struct PopulationCandidate {
    candidates: Vec<usize>,
    population_index: usize,
    input_priority: u8,
    declared_work: (usize, usize, usize),
    coverage: CoveragePriority,
    coverage_round: usize,
    decode_width_tier: usize,
}

pub(super) fn input_priority(has_early_policy: bool, has_early_opportunity: bool) -> u8 {
    match (has_early_policy, has_early_opportunity) {
        (false, _) => 0,
        (true, true) => 1,
        (true, false) => 2,
    }
}

impl BatchCandidate {
    fn order_key(&self) -> (u8, usize, usize, usize, usize, usize, u8, usize) {
        (
            self.coverage.policy,
            self.coverage_round,
            self.decode_width_tier,
            self.batch.serial_token_work,
            self.batch.serial_wave_upper_bound,
            self.batch.requests,
            self.input_priority,
            self.original_population_index,
        )
    }
}

impl CheckedSelection {
    pub fn retained_payload_bytes(&self) -> Option<usize> {
        let mut bytes = std::mem::size_of::<Self>()
            .checked_add(
                self.execution_case_indices
                    .capacity()
                    .checked_mul(std::mem::size_of::<usize>())?,
            )?
            .checked_add(
                self.populations
                    .capacity()
                    .checked_mul(std::mem::size_of::<SelectedPopulation>())?,
            )?
            .checked_add(
                self.batches
                    .capacity()
                    .checked_mul(std::mem::size_of::<SelectedBatch>())?,
            )?
            .checked_add(
                self.gaps
                    .capacity()
                    .checked_mul(std::mem::size_of::<SelectionGap>())?,
            )?;
        for population in &self.populations {
            bytes = bytes.checked_add(
                population
                    .representative_case_indices
                    .capacity()
                    .checked_mul(std::mem::size_of::<usize>())?,
            )?;
        }
        for batch in &self.batches {
            if let Some(opportunities) = &batch.scoped_opportunities {
                bytes = bytes.checked_add(
                    opportunities
                        .capacity()
                        .checked_mul(std::mem::size_of::<CaseOpportunity>())?,
                )?;
                for opportunity in opportunities {
                    bytes =
                        bytes.checked_add(source_inputs::opportunity_heap_bytes(opportunity)?)?;
                }
            }
            bytes = bytes.checked_add(
                batch
                    .algorithm_universe
                    .as_ref()
                    .map_or(Some(0), |u| u.retained_payload_bytes())?,
            )?;
            bytes = bytes.checked_add(
                batch
                    .population_indices
                    .capacity()
                    .checked_add(batch.representative_case_indices.capacity())?
                    .checked_mul(std::mem::size_of::<usize>())?,
            )?;
        }
        Some(bytes)
    }
}

/// Related populations may share one complete source horizon only when their
/// full representative rendered-template sets match. Each resulting source
/// retains every population's original anchors, members and phase bounds;
/// all numerical qualification remains with the original collector.
pub(super) fn select(
    cases: &[Case],
    opportunities: &[CaseOpportunity],
    inputs: &[Vec<CheckedInputFacts>],
    prompts: &[usize],
    chunk: usize,
    population: &StructuredServiceDeclarationV7,
    remaining_requests: usize,
    remaining_waves: usize,
    maximum_retained_bytes: usize,
) -> Result<CheckedSelection> {
    select_changed(
        cases,
        opportunities,
        inputs,
        prompts,
        chunk,
        population,
        remaining_requests,
        remaining_waves,
        maximum_retained_bytes,
        None,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn select_changed(
    cases: &[Case],
    opportunities: &[CaseOpportunity],
    inputs: &[Vec<CheckedInputFacts>],
    prompts: &[usize],
    chunk: usize,
    population: &StructuredServiceDeclarationV7,
    remaining_requests: usize,
    remaining_waves: usize,
    maximum_retained_bytes: usize,
    changed: Option<&[CheckedPopulationKey]>,
    selected_priority: Option<u8>,
) -> Result<CheckedSelection> {
    select_changed_with_geometry(
        cases,
        opportunities,
        inputs,
        prompts,
        chunk,
        population,
        remaining_requests,
        remaining_waves,
        maximum_retained_bytes,
        changed,
        selected_priority,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn select_changed_with_geometry(
    cases: &[Case],
    opportunities: &[CaseOpportunity],
    inputs: &[Vec<CheckedInputFacts>],
    prompts: &[usize],
    chunk: usize,
    population: &StructuredServiceDeclarationV7,
    remaining_requests: usize,
    remaining_waves: usize,
    maximum_retained_bytes: usize,
    changed: Option<&[CheckedPopulationKey]>,
    selected_priority: Option<u8>,
    geometry_work: Option<&mut StructuredInputGeometryWorkV1>,
) -> Result<CheckedSelection> {
    select_changed_with_geometry_and_source_limit(
        cases,
        opportunities,
        inputs,
        prompts,
        chunk,
        None,
        population,
        remaining_requests,
        remaining_waves,
        maximum_retained_bytes,
        changed,
        selected_priority,
        geometry_work,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn select_changed_with_geometry_and_source_limit(
    cases: &[Case],
    opportunities: &[CaseOpportunity],
    inputs: &[Vec<CheckedInputFacts>],
    prompts: &[usize],
    chunk: usize,
    prefill_row_ceiling: Option<NonZeroU32>,
    population: &StructuredServiceDeclarationV7,
    remaining_requests: usize,
    remaining_waves: usize,
    maximum_retained_bytes: usize,
    changed: Option<&[CheckedPopulationKey]>,
    selected_priority: Option<u8>,
    geometry_work: Option<&mut StructuredInputGeometryWorkV1>,
    maximum_sources: Option<NonZeroUsize>,
) -> Result<CheckedSelection> {
    select_with_local_composition(
        cases,
        opportunities,
        inputs,
        prompts,
        chunk,
        prefill_row_ceiling,
        population,
        remaining_requests,
        remaining_waves,
        maximum_retained_bytes,
        changed,
        selected_priority,
        geometry_work,
        maximum_sources,
        None,
    )
}

/// Optional composition authorizes original checked recipes and all extra
/// declaration/output allocations before the first selection/geometry work.
pub(super) fn composition_authorized(
    opportunities: &[CaseOpportunity],
    inputs: &[Vec<CheckedInputFacts>],
    requests: usize,
    population: &StructuredServiceDeclarationV7,
    geometry_enabled: bool,
    maximum_bytes: usize,
    seed: &ferrum_scheduler::implementations::continuous::cost_model::structured_v2::DeclaredAlgorithmUniverseV1,
) -> Result<bool> {
    composition::authorized(
        opportunities,
        inputs,
        requests,
        population,
        geometry_enabled,
        maximum_bytes,
        seed,
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn select_with_local_composition(
    cases: &[Case],
    opportunities: &[CaseOpportunity],
    inputs: &[Vec<CheckedInputFacts>],
    prompts: &[usize],
    chunk: usize,
    prefill_row_ceiling: Option<NonZeroU32>,
    population: &StructuredServiceDeclarationV7,
    remaining_requests: usize,
    remaining_waves: usize,
    maximum_retained_bytes: usize,
    changed: Option<&[CheckedPopulationKey]>,
    selected_priority: Option<u8>,
    geometry_work: Option<&mut StructuredInputGeometryWorkV1>,
    maximum_sources: Option<NonZeroUsize>,
    combination_seed: Option<&ferrum_scheduler::implementations::continuous::cost_model::structured_v2::DeclaredAlgorithmUniverseV1>,
) -> Result<CheckedSelection> {
    select_with_capacity(
        cases,
        opportunities,
        inputs,
        prompts,
        chunk,
        prefill_row_ceiling,
        population,
        SelectionCapacity::legacy(remaining_requests, remaining_waves),
        maximum_retained_bytes,
        changed,
        selected_priority,
        geometry_work,
        maximum_sources,
        combination_seed,
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn select_with_capacity(
    cases: &[Case],
    opportunities: &[CaseOpportunity],
    inputs: &[Vec<CheckedInputFacts>],
    prompts: &[usize],
    chunk: usize,
    prefill_row_ceiling: Option<NonZeroU32>,
    population: &StructuredServiceDeclarationV7,
    capacity: SelectionCapacity,
    maximum_retained_bytes: usize,
    changed: Option<&[CheckedPopulationKey]>,
    selected_priority: Option<u8>,
    mut geometry_work: Option<&mut StructuredInputGeometryWorkV1>,
    maximum_sources: Option<NonZeroUsize>,
    combination_seed: Option<&ferrum_scheduler::implementations::continuous::cost_model::structured_v2::DeclaredAlgorithmUniverseV1>,
) -> Result<CheckedSelection> {
    select_with_capacity_and_trajectories(
        cases,
        opportunities,
        inputs,
        prompts,
        chunk,
        prefill_row_ceiling,
        population,
        capacity,
        maximum_retained_bytes,
        changed,
        selected_priority,
        geometry_work,
        maximum_sources,
        combination_seed,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn select_with_capacity_and_trajectories(
    cases: &[Case],
    opportunities: &[CaseOpportunity],
    inputs: &[Vec<CheckedInputFacts>],
    prompts: &[usize],
    chunk: usize,
    prefill_row_ceiling: Option<NonZeroU32>,
    population: &StructuredServiceDeclarationV7,
    capacity: SelectionCapacity,
    maximum_retained_bytes: usize,
    changed: Option<&[CheckedPopulationKey]>,
    selected_priority: Option<u8>,
    geometry_work: Option<&mut StructuredInputGeometryWorkV1>,
    maximum_sources: Option<NonZeroUsize>,
    combination_seed: Option<&ferrum_scheduler::implementations::continuous::cost_model::structured_v2::DeclaredAlgorithmUniverseV1>,
    trajectories: Option<&inventory::CheckedCaseInventory>,
) -> Result<CheckedSelection> {
    if cases.len() != opportunities.len() || cases.len() != inputs.len() {
        return Err(error("checked selection inventory length differs"));
    }
    if trajectories.is_some_and(|inventory| {
        inventory.algorithm_case_inputs.len() != cases.len()
            || inventory
                .algorithm_case_inputs
                .iter()
                .flatten()
                .any(|&i| i >= inventory.algorithm_inputs.len())
    }) {
        return Err(error("checked trajectory inventory alignment differs"));
    }
    if cases.iter().any(|case| case.width == 0) {
        return Err(error("checked selection case has no original requests"));
    }
    #[cfg(any(test, feature = "test-support"))]
    crate::geometry_capture::selection_inputs(cases, prompts, chunk, prefill_row_ceiling);
    // Preserve the original single-pass plan and its checked numerical scopes.
    // Replacing the inventory with a broad scope before budgeting can discard
    // a schedulable narrow source. Scope-first experiments remain test-only
    // until a joint plan can preserve coverage within these same ledgers.
    select_prepared_inputs(
        cases,
        opportunities,
        inputs,
        prompts,
        chunk,
        prefill_row_ceiling,
        population,
        capacity,
        maximum_retained_bytes,
        changed,
        selected_priority,
        geometry_work,
        maximum_sources,
        combination_seed,
        trajectories,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
fn select_prepared_inputs(
    cases: &[Case],
    opportunities: &[CaseOpportunity],
    inputs: &[Vec<CheckedInputFacts>],
    prompts: &[usize],
    chunk: usize,
    prefill_row_ceiling: Option<NonZeroU32>,
    population: &StructuredServiceDeclarationV7,
    capacity: SelectionCapacity,
    maximum_retained_bytes: usize,
    changed: Option<&[CheckedPopulationKey]>,
    selected_priority: Option<u8>,
    mut geometry_work: Option<&mut StructuredInputGeometryWorkV1>,
    maximum_sources: Option<NonZeroUsize>,
    combination_seed: Option<&ferrum_scheduler::implementations::continuous::cost_model::structured_v2::DeclaredAlgorithmUniverseV1>,
    trajectories: Option<&inventory::CheckedCaseInventory>,
    scopes: Option<&[Option<ferrum_scheduler::implementations::continuous::cost_model::structured_v2::DeclaredAlgorithmUniverseV1>]>,
) -> Result<CheckedSelection> {
    // Grouping is authorized separately before it allocates. Then its exact
    // identities/capacities bound headers and sequential scratch stages; no
    // representative or geometry algorithm runs before the second check.
    let grouping_peak = memory::grouping_peak(opportunities)?;
    if grouping_peak > maximum_retained_bytes {
        return Err(error(format!("checked selection grouping retained payload exceeds remaining budget: required_peak_bytes={grouping_peak} remaining_bytes={maximum_retained_bytes}")));
    }
    let groups = member_groups(opportunities)
        .map_err(|reason| error(format!("checked selection population: {reason:?}")))?;
    let memory = memory::plan(
        &groups,
        opportunities,
        inputs,
        capacity.requests,
        geometry_work.as_ref().map(|_| &population.settings),
    )?;
    tracing::info!(
        ?memory,
        remaining_bytes = maximum_retained_bytes,
        "Automatic input selection retained memory stages"
    );
    let combination_charge = combination_seed
        .map(|seed| composition::extra_peak(opportunities, seed, memory.guaranteed_groups))
        .transpose()?
        .unwrap_or(0);
    let peak = add(memory.required_peak_bytes, combination_charge)?;
    if peak > maximum_retained_bytes {
        let (key_mentions, guaranteed_cases) = (memory.key_mentions, memory.guaranteed_cases);
        return Err(error(format!(
            "checked selection peak retained payload exceeds remaining budget: required_peak_bytes={peak} remaining_bytes={maximum_retained_bytes} key_mentions={key_mentions} guaranteed_cases={guaranteed_cases}"
        )));
    }
    let geometry_scratch = memory.geometry_scratch_bytes;
    let mut out = CheckedSelection::default();
    let mut population_candidates = Vec::new();
    let mut batch_candidates = Vec::new();
    if opportunities
        .iter()
        .any(|case| matches!(case.population, CasePopulation::Unknown { .. }))
    {
        out.gaps.push(SelectionGap {
            population: None,
            reason: SelectionGapReason::UnknownPopulation,
        });
    }
    if groups.is_empty() && out.gaps.is_empty() {
        out.gaps.push(SelectionGap {
            population: None,
            reason: SelectionGapReason::UnknownPopulation,
        });
    }
    for group in groups {
        if changed.is_some_and(|keys| !keys.contains(&group.key)) {
            out.gaps.push(SelectionGap {
                population: Some(group.key),
                reason: SelectionGapReason::PreviouslyDeclaredPopulation,
            });
            continue;
        }
        let had_guaranteed = !group.guaranteed_case_indices.is_empty();
        let mut candidates = Vec::new();
        for index in group.guaranteed_case_indices {
            let Some(first) = inputs[index].first() else {
                continue;
            };
            if inputs[index]
                .iter()
                .any(|facts| facts.key(population.population_policy()) != group.key)
            {
                return Err(error(
                    "checked selection facts differ from population identity",
                ));
            }
            // Identity uniqueness is insufficient to guarantee a particular
            // axis/branch anchor when checked alternatives differ numerically.
            if inputs[index]
                .iter()
                .all(|facts| facts.axes == first.axes && facts.branches == first.branches)
            {
                candidates.push(index);
            }
        }
        if candidates.is_empty() {
            out.gaps.push(SelectionGap {
                population: Some(group.key),
                reason: if !had_guaranteed {
                    SelectionGapReason::NoGuaranteedMember
                } else {
                    SelectionGapReason::NoSingleCheckedInput
                },
            });
            continue;
        }
        let scope = scoped_inputs::for_indices(&candidates, scopes)?;
        let selected = representatives_with_positive_min(
            &candidates,
            inputs,
            cases,
            prompts,
            chunk,
            prefill_row_ceiling,
            scope.is_some(),
        )?;
        let has_early_policy = candidates.iter().any(|&i| inputs[i][0].branches[4]);
        let has_early_opportunity = candidates.iter().any(|&i| inputs[i][0].branches[5]);
        let input_priority = input_priority(has_early_policy, has_early_opportunity);
        if has_early_policy {
            out.gaps.push(SelectionGap {
                population: Some(group.key.clone()),
                reason: if has_early_opportunity {
                    SelectionGapReason::OutcomeDependentEarlyTermination
                } else {
                    SelectionGapReason::EarlyTerminalOpportunityMissing
                },
            });
        }
        // Keep population indices in original inventory order. Coverage order
        // only controls bounded work and complete source reservations; it
        // never changes checked membership or qualification obligations.
        let original_population_index = out.populations.len();
        out.populations.push(SelectedPopulation {
            key: group.key,
            representative_case_indices: selected,
            maximum_anchor_span: 0,
            scheduled: false,
            batch_index: None,
            input_geometry: None,
        });
        // Rank the complete independent phase horizon before spending shared
        // geometry work. A no-token intermediate prefill has no EOS risk but
        // can require far more original work than a token-producing source.
        // This is a declared work heuristic, never a wall-time feasibility
        // estimate or a relaxation of the source's qualification obligations.
        let initial_batch = batch_plan_with_scope(
            &[original_population_index],
            &out.populations,
            cases,
            opportunities,
            prompts,
            chunk,
            prefill_row_ceiling,
            population,
            scope.map(|universe| (inputs, universe)),
        )?;
        let declared_work = (
            initial_batch.serial_token_work,
            initial_batch.serial_wave_upper_bound,
            initial_batch.requests,
        );
        drop(initial_batch);
        population_candidates.push(PopulationCandidate {
            population_index: original_population_index,
            input_priority,
            declared_work,
            coverage: CoveragePriority::from_cases(&candidates, cases)?,
            coverage_round: 0,
            decode_width_tier: grouping::width_tier(&candidates, cases, inputs),
            candidates,
        });
    }
    // Establish product-policy priority before spending the one shared
    // geometry ledger. Sorting only the final execution list would leave the
    // required policy's basis work behind already exhausted auxiliary work.
    priority::order_populations(&mut population_candidates, inputs);
    #[cfg(any(test, feature = "test-support"))]
    crate::geometry_capture::begin_selection(
        population_candidates
            .iter()
            .filter(|candidate| {
                geometry_work.is_some()
                    && selected_priority.is_none_or(|p| p == candidate.input_priority)
            })
            .count(),
    );
    for candidate in population_candidates {
        let original_population_index = candidate.population_index;
        let member = &mut out.populations[original_population_index];
        if let Some(work) = geometry_work.as_deref_mut().filter(|_| {
            selected_priority.is_none_or(|selected| selected == candidate.input_priority)
        }) {
            #[cfg(any(test, feature = "test-support"))]
            crate::geometry_capture::population(original_population_index, &member.key);
            let (audit, gap) = input_geometry::extend(
                &candidate.candidates,
                inputs,
                &mut member.representative_case_indices,
                &population.settings,
                work,
                geometry_scratch,
            );
            member.input_geometry = Some(audit);
            if let Some(reason) = gap {
                out.gaps.push(SelectionGap {
                    population: Some(member.key.clone()),
                    reason,
                });
            }
        }
        // Start with complete independent plans. A later bounded merge may
        // share offers with related inputs, without deleting representatives.
        let scope = scoped_inputs::for_indices(&candidate.candidates, scopes)?;
        let batch = batch_plan_with_scope(
            &[original_population_index],
            &out.populations,
            cases,
            opportunities,
            prompts,
            chunk,
            prefill_row_ceiling,
            population,
            scope.map(|universe| (inputs, universe)),
        )?;
        batch_candidates.push(BatchCandidate {
            input_priority: candidate.input_priority,
            coverage: candidate.coverage,
            coverage_round: 0,
            original_population_index,
            decode_width_tier: grouping::width_tier(
                &batch.representative_case_indices,
                cases,
                inputs,
            ),
            batch,
        });
    }
    #[cfg(any(test, feature = "test-support"))]
    crate::geometry_capture::end_selection();
    // Raw representative selection protects each algorithm's positive range,
    // branch and width endpoints. Choose the source scope before its final
    // work order and source reservation; no raw observations have been taken.
    if let Some(seed) = combination_seed {
        for candidate in &mut batch_candidates {
            if scoped_inputs::for_indices(&candidate.batch.representative_case_indices, scopes)?
                .is_some()
            {
                continue;
            }
            if let Some(scoped) = composition::scoped_candidate(
                &candidate.batch,
                &out.populations,
                cases,
                opportunities,
                inputs,
                trajectories,
                prompts,
                chunk,
                prefill_row_ceiling,
                population,
                seed,
            )? {
                candidate.batch = scoped;
            }
        }
    }
    priority::order_batches(&mut batch_candidates, inputs);
    coalesce_related_batches(
        &mut batch_candidates,
        &out.populations,
        cases,
        opportunities,
        inputs,
        prompts,
        chunk,
        prefill_row_ceiling,
        population,
        capacity,
        selected_priority,
        maximum_sources,
        combination_seed,
        trajectories,
        scopes,
    )?;
    let mut scheduled_sources = 0_usize;
    for candidate in batch_candidates {
        let deferred = selected_priority
            .filter(|&selected| selected != candidate.input_priority)
            .map(|selected| SelectionGapReason::DeferredInputPriority {
                priority: candidate.input_priority,
                selected,
            })
            .or_else(|| {
                maximum_sources
                    .filter(|maximum| scheduled_sources >= maximum.get())
                    .map(|maximum| SelectionGapReason::RetainedSourceCapacity {
                        maximum_sources: maximum.get(),
                    })
            });
        append_batch(&mut out, candidate.batch, capacity, deferred)?;
        scheduled_sources += usize::from(out.batches.last().unwrap().scheduled);
    }
    out.input_geometry = geometry_work.map(|work| InputGeometryCharge {
        cumulative_visits: work.visits(),
        maximum_visits: work.maximum_visits(),
        exhausted: work.exhausted(),
    });
    #[cfg(any(test, feature = "test-support"))]
    crate::geometry_capture::selection_result(&out);
    Ok(out)
}

/// Equality of complete sets, not an overlap graph: {A}, {A, B}, {B} cannot
/// join transitively. Template IDs identify the already rendered exact input.
/// Coverage priority separately keeps required policies and auxiliary routes
/// apart; within one lane, original keys and obligations stay independent.
fn same_representative_inputs(a: &SelectedBatch, b: &SelectedBatch, cases: &[Case]) -> bool {
    let subset = |left: &[usize], right: &[usize]| {
        left.iter().all(|&i| {
            right
                .iter()
                .any(|&j| cases[i].template == cases[j].template)
        })
    };
    subset(
        &a.representative_case_indices,
        &b.representative_case_indices,
    ) && subset(
        &b.representative_case_indices,
        &a.representative_case_indices,
    )
}

#[allow(clippy::too_many_arguments)]
fn coalesce_related_batches(
    candidates: &mut Vec<BatchCandidate>,
    populations: &[SelectedPopulation],
    cases: &[Case],
    opportunities: &[CaseOpportunity],
    inputs: &[Vec<CheckedInputFacts>],
    prompts: &[usize],
    chunk: usize,
    prefill_row_ceiling: Option<NonZeroU32>,
    population: &StructuredServiceDeclarationV7,
    capacity: SelectionCapacity,
    selected_priority: Option<u8>,
    maximum_sources: Option<NonZeroUsize>,
    combination_seed: Option<&ferrum_scheduler::implementations::continuous::cost_model::structured_v2::DeclaredAlgorithmUniverseV1>,
    trajectories: Option<&inventory::CheckedCaseInventory>,
    scopes: Option<&[Option<ferrum_scheduler::implementations::continuous::cost_model::structured_v2::DeclaredAlgorithmUniverseV1>]>,
) -> Result<()> {
    // First preserve the existing same-policy journal choices. Only then
    // use their already checked representatives to pack independent host
    // families under the same finite source limit. No geometry is revisited.
    for independent_families in [false, true] {
        if independent_families && (combination_seed.is_none() || maximum_sources.is_none()) {
            continue;
        }
        let mut index = 0;
        while index < candidates.len() {
            let mut next = index + 1;
            while next < candidates.len() {
                let a = &candidates[index];
                let b = &candidates[next];
                let raw_related = grouping::related(a, b, cases, inputs);
                let related = if independent_families {
                    grouping::independent_families(
                        a,
                        b,
                        populations,
                        cases,
                        inputs,
                        selected_priority,
                    )
                } else {
                    a.input_priority == b.input_priority
                        && (raw_related
                            || combination_seed.is_some() && grouping::related_scope(a, b, inputs))
                };
                if !related {
                    next += 1;
                    continue;
                }
                // A source has one numerical interpretation. Sharing journals
                // must not turn a retained raw family into a new scoped child,
                // or silently enlarge either original universe.
                let fixed_scope =
                    scoped_inputs::for_indices(&a.batch.representative_case_indices, scopes)?
                        .is_some()
                        || scoped_inputs::for_indices(
                            &b.batch.representative_case_indices,
                            scopes,
                        )?
                        .is_some();
                if (independent_families || fixed_scope)
                    && a.batch.algorithm_universe != b.batch.algorithm_universe
                {
                    next += 1;
                    continue;
                }
                let mut members = a.batch.population_indices.clone();
                members.extend_from_slice(&b.batch.population_indices);
                members.sort_unstable();
                let scope = (independent_families || fixed_scope)
                    .then_some(a.batch.algorithm_universe.as_ref())
                    .flatten();
                let mut combined = batch_plan_with_scope(
                    &members,
                    populations,
                    cases,
                    opportunities,
                    prompts,
                    chunk,
                    prefill_row_ceiling,
                    population,
                    scope.map(|universe| (inputs, universe)),
                )?;
                if independent_families {
                    if !composition::packing_valid(
                        &combined,
                        inputs,
                        trajectories,
                        population,
                        combination_seed.unwrap(),
                    )? {
                        next += 1;
                        continue;
                    }
                } else if !fixed_scope {
                    if let Some(seed) = combination_seed {
                        if let Some(scoped) = composition::scoped_candidate(
                            &combined,
                            populations,
                            cases,
                            opportunities,
                            inputs,
                            trajectories,
                            prompts,
                            chunk,
                            prefill_row_ceiling,
                            population,
                            seed,
                        )? {
                            combined = scoped;
                        } else if !raw_related
                            || a.batch.algorithm_universe.is_some()
                            || b.batch.algorithm_universe.is_some()
                        {
                            // A failed wider declaration cannot silently replace a
                            // previously selected valid scope with its raw fallback.
                            next += 1;
                            continue;
                        }
                    }
                }
                // Recompute the whole source, including every fresh-member phase
                // fence. Relatedness never supplies a qualification or capacity
                // exemption. The old same-family sharing rule also keeps its
                // original non-increasing token-work condition. New independent
                // families charge their entire recomputed horizon below.
                if !composition::can_schedule(&combined, capacity)
                    || (!independent_families
                        && combined.serial_token_work
                            > add(a.batch.serial_token_work, b.batch.serial_token_work)?)
                {
                    next += 1;
                    continue;
                }
                // A globally affordable union may still steal the allowance of
                // an earlier or later complete source. Simulate the original and
                // replacement traversals under the same work and source limits before
                // dropping either raw candidate.
                if !grouping::preserves_scheduled(
                    candidates,
                    index,
                    next,
                    &combined,
                    capacity,
                    selected_priority,
                    maximum_sources,
                    !independent_families && !raw_related,
                    independent_families,
                )? {
                    next += 1;
                    continue;
                }
                let first = a.original_population_index.min(b.original_population_index);
                let width = a.decode_width_tier.max(b.decode_width_tier);
                candidates[index].batch = combined;
                candidates[index].original_population_index = first;
                candidates[index].decode_width_tier = width;
                candidates.remove(next);
                // Additional guaranteed offers can make an earlier, otherwise
                // unaffordable combination fit. Recheck it with this exact plan.
                next = index + 1;
            }
            index += 1;
        }
    }
    Ok(())
}

/// Count borrowed checked identities without allocating or dropping any group.
fn selection_inventory_cardinality(opportunities: &[CaseOpportunity]) -> Result<(usize, usize)> {
    let mut key_mentions = 0usize;
    let mut guaranteed_cases = 0usize;
    for opportunity in opportunities {
        let count = match &opportunity.population {
            CasePopulation::Unique(_) => 1,
            CasePopulation::Alternatives(keys) => keys.len(),
            CasePopulation::Unknown { known_alternatives } => known_alternatives.len(),
        };
        key_mentions = add(key_mentions, count)?;
        if opportunity.minimum_fresh_members == 1
            && matches!(opportunity.population, CasePopulation::Unique(_))
        {
            guaranteed_cases = add(guaranteed_cases, 1)?;
        }
    }
    Ok((key_mentions, guaranteed_cases))
}

/// Conservative simultaneous owned payload, including Vec growth and the old
/// plus replacement allocation during growth. Numeric phase horizons do not
/// expand an execution vector past remaining_requests: each original cohort
/// consumes at least one request and append_batch checks that bound first.
/// No object is allocated per phase member before that bounded expansion.
#[cfg(test)]
fn legacy_selection_peak_payload_bound(
    cases: &[Case],
    opportunities: &[CaseOpportunity],
    inputs: &[Vec<CheckedInputFacts>],
    remaining_requests: usize,
) -> Result<usize> {
    if cases.iter().any(|case| case.width == 0) {
        return Err(error("checked selection case has no original requests"));
    }
    let (key_mentions, guaranteed_cases) = selection_inventory_cardinality(opportunities)?;
    let axes = inputs
        .iter()
        .flatten()
        .map(|facts| facts.axes.len())
        .max()
        .unwrap_or(0);
    // A population produces representatives/batches only from its guaranteed
    // Unique cases. Those cases belong to one key each, so their total bounds
    // every later representative/index array and population/batch header.
    // Uncaptured declarations own no selector arrays; possible-only groups
    // still own their original member lists and gaps, charged by key_mentions.
    // Do not use changed/priority to omit groups: their typed gaps remain live.
    let n = guaranteed_cases;
    let mut bytes = std::mem::size_of::<CheckedSelection>();
    let mut charge = |extra| -> Result<()> {
        bytes = add(bytes, extra)?;
        Ok(())
    };

    // Initial population groups: at most one group per key occurrence. Each
    // group owns two index vectors; include minimum Vec capacity for each.
    charge(vector_peak_bytes::<populations::PopulationMemberGroup>(
        key_mentions,
    )?)?;
    charge(vector_peak_bytes::<usize>(add(
        mul(key_mentions, 18)?,
        mul(n, 2)?,
    )?)?)?;
    // Final output: at most n guaranteed populations/batches. Representative
    // and population-index vectors contain at most n entries in total each.
    charge(vector_peak_bytes::<SelectedPopulation>(n)?)?;
    // All independent plans and their ranking fields coexist with the output
    // batch vector while candidates move into their final execution order.
    charge(vector_peak_bytes::<BatchCandidate>(n)?)?;
    charge(vector_peak_bytes::<PopulationCandidate>(n)?)?;
    charge(vector_peak_bytes::<usize>(mul(n, 5)?)?)?;
    charge(vector_peak_bytes::<SelectedBatch>(n)?)?;
    charge(mul(vector_peak_bytes::<usize>(mul(n, 9)?)?, 3)?)?;
    charge(vector_peak_bytes::<SelectionGap>(add(
        key_mentions,
        add(mul(n, 8)?, 1)?,
    )?)?)?;
    charge(vector_peak_bytes::<usize>(remaining_requests)?)?;

    // One independent or combined batch is planned beside retained candidates.
    // Include its returned/header temporaries and both owned index vectors.
    // Case/Opportunity copies have no nested payload because only Unique
    // populations are eligible for this path.
    charge(mul(std::mem::size_of::<SelectedBatch>(), 2)?)?;
    // Include the trial merge's population-index vector before accepting it.
    charge(mul(vector_peak_bytes::<usize>(n)?, 8)?)?;
    charge(vector_peak_bytes::<Case>(n)?)?;
    charge(vector_peak_bytes::<CaseOpportunity>(n)?)?;
    // Nested budget::plan_checked has its own groups and two timeline vectors.
    charge(vector_peak_bytes::<populations::PopulationMemberGroup>(n)?)?;
    charge(vector_peak_bytes::<usize>(mul(n, 18)?)?)?;
    charge(mul(vector_peak_bytes::<usize>(n)?, 2)?)?;

    // Greedy representative selection retains two actual axis endpoints,
    // zero/positive and branch obligations, candidates and selected indices.
    charge(mul(vector_peak_bytes::<f64>(axes)?, 2)?)?;
    charge(vector_peak_bytes::<bool>(add(mul(axes, 2)?, 14)?)?)?;
    charge(mul(vector_peak_bytes::<usize>(n)?, 2)?)?;
    charge(vector_peak_bytes::<SelectionGapReason>(6)?)?;
    Ok(bytes)
}

#[cfg(test)]
fn selection_peak_payload_bound(
    cases: &[Case],
    opportunities: &[CaseOpportunity],
    inputs: &[Vec<CheckedInputFacts>],
    remaining_requests: usize,
) -> Result<usize> {
    if cases.iter().any(|case| case.width == 0) {
        return Err(error("checked selection case has no original requests"));
    }
    let groups = member_groups(opportunities)
        .map_err(|reason| error(format!("checked selection population: {reason:?}")))?;
    Ok(memory::plan(&groups, opportunities, inputs, remaining_requests, None)?.required_peak_bytes)
}

fn vector_peak_bytes<T>(elements: usize) -> Result<usize> {
    // Eight spare elements cover minimum small-Vec allocation. Four times the
    // resulting payload covers growth and its concurrently live old buffer.
    mul(mul(add(elements, 8)?, 4)?, std::mem::size_of::<T>())
}

fn batch_plan(
    population_indices: &[usize],
    populations: &[SelectedPopulation],
    cases: &[Case],
    opportunities: &[CaseOpportunity],
    prompts: &[usize],
    chunk: usize,
    prefill_row_ceiling: Option<NonZeroU32>,
    population: &StructuredServiceDeclarationV7,
) -> Result<SelectedBatch> {
    batch_plan_with_scope(
        population_indices,
        populations,
        cases,
        opportunities,
        prompts,
        chunk,
        prefill_row_ceiling,
        population,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
fn batch_plan_with_scope(
    population_indices: &[usize],
    populations: &[SelectedPopulation],
    cases: &[Case],
    opportunities: &[CaseOpportunity],
    prompts: &[usize],
    chunk: usize,
    prefill_row_ceiling: Option<NonZeroU32>,
    population: &StructuredServiceDeclarationV7,
    scope: Option<(&[Vec<CheckedInputFacts>], &ferrum_scheduler::implementations::continuous::cost_model::structured_v2::DeclaredAlgorithmUniverseV1)>,
) -> Result<SelectedBatch> {
    batch_plan_with_schedule(
        population_indices,
        populations,
        cases,
        opportunities,
        prompts,
        chunk,
        prefill_row_ceiling,
        population,
        scope,
        budget::startup_schedule,
    )
}

/// Keep the complete work calculation shared with offline schedule experiments.
/// Product callers always use the original `startup_schedule` above.
#[allow(clippy::too_many_arguments)]
fn batch_plan_with_schedule(
    population_indices: &[usize],
    populations: &[SelectedPopulation],
    cases: &[Case],
    opportunities: &[CaseOpportunity],
    prompts: &[usize],
    chunk: usize,
    prefill_row_ceiling: Option<NonZeroU32>,
    population: &StructuredServiceDeclarationV7,
    scope: Option<(&[Vec<CheckedInputFacts>], &ferrum_scheduler::implementations::continuous::cost_model::structured_v2::DeclaredAlgorithmUniverseV1)>,
    schedule_for: impl FnOnce(
        &[usize], &[usize],
        &ferrum_scheduler::implementations::continuous::cost_model::structured_v2::StructuredSettingsV2,
    ) -> Result<(OwnerBlockScheduleV1, usize)>,
) -> Result<SelectedBatch> {
    let representative_case_indices: Vec<_> = population_indices
        .iter()
        .flat_map(|&i| populations[i].representative_case_indices.iter().copied())
        .collect();
    batch_plan_for_cycle(
        population_indices,
        representative_case_indices,
        cases,
        opportunities,
        prompts,
        chunk,
        prefill_row_ceiling,
        population,
        scope,
        schedule_for,
    )
}

/// One work calculation for an explicitly ordered original occurrence cycle.
/// Product callers above retain their original concatenated family order.
#[allow(clippy::too_many_arguments)]
fn batch_plan_for_cycle(
    population_indices: &[usize],
    representative_case_indices: Vec<usize>,
    cases: &[Case],
    opportunities: &[CaseOpportunity],
    prompts: &[usize],
    chunk: usize,
    prefill_row_ceiling: Option<NonZeroU32>,
    population: &StructuredServiceDeclarationV7,
    scope: Option<(&[Vec<CheckedInputFacts>], &ferrum_scheduler::implementations::continuous::cost_model::structured_v2::DeclaredAlgorithmUniverseV1)>,
    schedule_for: impl FnOnce(
        &[usize], &[usize],
        &ferrum_scheduler::implementations::continuous::cost_model::structured_v2::StructuredSettingsV2,
    ) -> Result<(OwnerBlockScheduleV1, usize)>,
) -> Result<SelectedBatch> {
    let selected_cases: Vec<_> = representative_case_indices
        .iter()
        .map(|&i| cases[i].clone())
        .collect();
    let selected_opportunities: Vec<_> = representative_case_indices
        .iter()
        .map(|&i| match scope {
            Some((inputs, universe)) => {
                composition::project_opportunity(&opportunities[i], &inputs[i], universe)
            }
            None => Ok(opportunities[i].clone()),
        })
        .collect::<Result<_>>()?;
    let mut starts = Vec::new();
    let mut ends = Vec::new();
    let (
        mut minimum_cycle,
        mut cycle_waves,
        mut cycle_requests,
        mut cycle_serial,
        mut cycle_tokens,
        mut cycle_offer_rows,
    ) = (0usize, 0usize, 0usize, 0usize, 0usize, 0usize);
    for case in &selected_cases {
        let prompt = *prompts
            .get(case.template)
            .ok_or_else(|| error("checked case prompt missing"))?;
        let work = work::case_work(case, prompt, chunk, prefill_row_ceiling)?;
        starts.push(cycle_waves);
        cycle_waves = add(cycle_waves, work.declared_offers_upper)?;
        ends.push(cycle_waves);
        cycle_serial = add(cycle_serial, work.execution_actions)?;
        cycle_offer_rows = add(cycle_offer_rows, work.serial_declared_offer_rows)?;
        cycle_requests = add(cycle_requests, work.requests)?;
        cycle_tokens = add(cycle_tokens, work.serial_token_work)?;
        minimum_cycle = add(minimum_cycle, work.declared_offers_minimum)?;
    }
    let (mut schedule, maximum_anchor_span) = schedule_for(&starts, &ends, &population.settings)?;
    schedule.prediction_validity = population.schedule.prediction_validity;
    let mut numerical = population.settings.clone();
    numerical.max_phase_samples = *schedule.maximum_phase_members.iter().max().unwrap();
    let schedule_within_capacity = schedule.validate(&numerical).is_ok();
    let input_opportunities = budget::plan_checked_with_schedule_and_row_ceiling(
        &selected_cases,
        prompts,
        chunk,
        prefill_row_ceiling,
        &schedule,
        minimum_cycle,
        &selected_opportunities,
    )?;
    let planned_cycles = add(
        input_opportunities.required_original_offers,
        schedule.block_offered,
    )?
    .div_ceil(minimum_cycle);
    // Recompute on the complete source, including after source coalescing.
    // Equal keys share one seed, never one setup per numerical phase/cycle.
    let setup = work::setup_for_indices(cases, &representative_case_indices)?;
    Ok(SelectedBatch {
        population_indices: population_indices.to_vec(),
        representative_case_indices,
        input_opportunities,
        schedule,
        schedule_within_capacity,
        planned_cycles,
        maximum_anchor_span,
        requests: add(mul(cycle_requests, planned_cycles)?, setup.requests)?,
        serial_token_work: add(mul(cycle_tokens, planned_cycles)?, setup.serial_token_work)?,
        serial_wave_upper_bound: add(mul(cycle_serial, planned_cycles)?, setup.execution_actions)?,
        declared_offer_row_bound: mul(cycle_offer_rows, planned_cycles)?,
        scheduled: false,
        algorithm_universe: scope.map(|(_, universe)| universe.clone()),
        scoped_opportunities: scope.map(|_| selected_opportunities),
    })
}

fn append_batch(
    out: &mut CheckedSelection,
    mut batch: SelectedBatch,
    capacity: SelectionCapacity,
    deferred: Option<SelectionGapReason>,
) -> Result<()> {
    let remaining = capacity.remaining(SelectionCapacity {
        requests: out.requests,
        execution_actions: out.serial_wave_upper_bound,
        declared_offer_rows: out.declared_offer_row_bound,
    });
    let mut reasons: Vec<_> = deferred.into_iter().collect();
    if !batch.schedule_within_capacity {
        reasons.push(SelectionGapReason::SourceScheduleCapacity);
    }
    let minimum_phase = *batch.schedule.phase_min_offered.iter().min().unwrap();
    if batch.maximum_anchor_span > minimum_phase {
        reasons.push(SelectionGapReason::AnchorAfterEarliestFreeze {
            span: batch.maximum_anchor_span,
            minimum_offers: minimum_phase,
        });
    }
    if batch.requests > remaining.requests {
        reasons.push(SelectionGapReason::RemainingRequests {
            required: batch.requests,
            remaining: remaining.requests,
        });
    }
    if batch.serial_wave_upper_bound > remaining.execution_actions {
        reasons.push(SelectionGapReason::RemainingWaves {
            required: batch.serial_wave_upper_bound,
            remaining: remaining.execution_actions,
        });
    }
    if batch.declared_offer_row_bound > remaining.declared_offer_rows {
        reasons.push(SelectionGapReason::RemainingOfferRows {
            required: batch.declared_offer_row_bound,
            remaining: remaining.declared_offer_rows,
        });
    }
    batch.scheduled = reasons.is_empty();
    for &index in &batch.population_indices {
        let member = &mut out.populations[index];
        if !member.scheduled {
            member.maximum_anchor_span = batch.maximum_anchor_span;
            member.scheduled = batch.scheduled;
            member.batch_index = Some(out.batches.len());
        }
        for reason in &reasons {
            out.gaps.push(SelectionGap {
                population: Some(member.key.clone()),
                reason: reason.clone(),
            });
        }
    }
    if batch.scheduled {
        out.execution_case_indices
            .try_reserve(mul(
                batch.representative_case_indices.len(),
                batch.planned_cycles,
            )?)
            .map_err(|_| error("checked selection allocation capacity"))?;
        for _ in 0..batch.planned_cycles {
            out.execution_case_indices
                .extend_from_slice(&batch.representative_case_indices);
        }
        out.requests = add(out.requests, batch.requests)?;
        out.serial_wave_upper_bound =
            add(out.serial_wave_upper_bound, batch.serial_wave_upper_bound)?;
        out.declared_offer_row_bound =
            add(out.declared_offer_row_bound, batch.declared_offer_row_bound)?;
    }
    out.batches.push(batch);
    Ok(())
}

fn add(a: usize, b: usize) -> Result<usize> {
    a.checked_add(b)
        .ok_or_else(|| error("checked selection count overflow"))
}
fn mul(a: usize, b: usize) -> Result<usize> {
    a.checked_mul(b)
        .ok_or_else(|| error("checked selection count overflow"))
}

/// Preserve both ends of every actual numeric axis, including every observed
/// zero/positive direction. This includes the repetition work axis without
/// guessing its private offset. Preserve each observed host branch separately.
fn representatives(
    candidates: &[usize],
    inputs: &[Vec<CheckedInputFacts>],
    cases: &[Case],
    prompts: &[usize],
    chunk: usize,
) -> Result<Vec<usize>> {
    representatives_with_row_ceiling(candidates, inputs, cases, prompts, chunk, None)
}

fn representatives_with_row_ceiling(
    candidates: &[usize],
    inputs: &[Vec<CheckedInputFacts>],
    cases: &[Case],
    prompts: &[usize],
    chunk: usize,
    prefill_row_ceiling: Option<NonZeroU32>,
) -> Result<Vec<usize>> {
    representatives_with_positive_min(
        candidates,
        inputs,
        cases,
        prompts,
        chunk,
        prefill_row_ceiling,
        false,
    )
}

fn representatives_with_positive_min(
    candidates: &[usize],
    inputs: &[Vec<CheckedInputFacts>],
    cases: &[Case],
    prompts: &[usize],
    chunk: usize,
    prefill_row_ceiling: Option<NonZeroU32>,
    preserve_positive_min: bool,
) -> Result<Vec<usize>> {
    let dimensions = inputs[candidates[0]][0].axes.len();
    if candidates
        .iter()
        .any(|&i| inputs[i][0].axes.len() != dimensions)
    {
        return Err(error("checked selection axis dimensions differ"));
    }
    let mut minima = inputs[candidates[0]][0].axes.clone();
    let mut maxima = minima.clone();
    // Union zero-extension must not hide the original positive lower endpoint.
    let mut positive_minima = preserve_positive_min.then(|| vec![f64::INFINITY; dimensions]);
    let mut branch_seen = [[false; 2]; 7];
    for &index in candidates {
        let facts = &inputs[index][0];
        for ((min, max), &value) in minima.iter_mut().zip(&mut maxima).zip(&facts.axes) {
            *min = min.min(value);
            *max = max.max(value);
        }
        if let Some(positive) = &mut positive_minima {
            for (minimum, &value) in positive.iter_mut().zip(&facts.axes) {
                if value > 0. {
                    *minimum = minimum.min(value);
                }
            }
        }
        for (seen, &branch) in branch_seen.iter_mut().zip(&facts.branches) {
            seen[usize::from(branch)] = true;
        }
    }
    let axis_slots = if preserve_positive_min { 3 } else { 2 };
    let numeric_slots = mul(dimensions, axis_slots)?;
    let count = add(numeric_slots, 14)?;
    let mut uncovered = vec![true; count];
    if let Some(positive) = &positive_minima {
        for (axis, minimum) in positive.iter().enumerate() {
            uncovered[axis_slots * axis + 2] = minimum.is_finite();
        }
    }
    for (branch, seen) in branch_seen.iter().enumerate() {
        for (value, present) in seen.iter().enumerate() {
            uncovered[numeric_slots + 2 * branch + value] = *present;
        }
    }
    let covers = |facts: &CheckedInputFacts, axis: usize| {
        if axis < numeric_slots {
            let column = axis / axis_slots;
            facts.axes[column]
                == match axis % axis_slots {
                    0 => minima[column],
                    1 => maxima[column],
                    _ => positive_minima.as_ref().unwrap()[column],
                }
        } else {
            let slot = axis - numeric_slots;
            usize::from(facts.branches[slot / 2]) == slot % 2
        }
    };
    let mut selected = Vec::new();
    while uncovered.iter().any(|needed| *needed) {
        let mut best: Option<(usize, usize, usize, usize)> = None;
        for &index in candidates {
            let gain = uncovered
                .iter()
                .enumerate()
                .filter(|(axis, needed)| **needed && covers(&inputs[index][0], *axis))
                .count();
            if gain == 0 {
                continue;
            }
            let case = &cases[index];
            let prompt = *prompts
                .get(case.template)
                .ok_or_else(|| error("checked case prompt missing"))?;
            let work = work::case_work(case, prompt, chunk, prefill_row_ceiling)?;
            if best.is_none_or(|(old_gain, old_requests, old_waves, old_index)| {
                gain > old_gain
                    || (gain == old_gain
                        && (work.requests, work.execution_actions, index)
                            < (old_requests, old_waves, old_index))
            }) {
                best = Some((gain, work.requests, work.execution_actions, index));
            }
        }
        let index = best
            .ok_or_else(|| error("checked anchor has no representative"))?
            .3;
        selected.push(index);
        for (axis, needed) in uncovered.iter_mut().enumerate() {
            *needed &= !covers(&inputs[index][0], axis);
        }
    }
    // Keep source candidate order as a deterministic local bundle; fresh_span
    // checks every possible phase cut, not only this bundle's opening.
    selected.sort_unstable();
    Ok(selected)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferrum_interfaces::{execution_cost::*, vnext::DeviceCommandPhase};
    use ferrum_scheduler::implementations::continuous::cost_model::structured_v2::StructuredInputV2;
    use populations::{classify_alternatives, tests as fixture};
    mod bootstrap_role_coverage;
    mod composition;
    mod input_geometry_tests;
    mod journal_grouping;
    mod memory;
    mod policy_priority;
    mod related;
    mod scoped_inputs;

    fn natural_termination_input(
        rows: u32,
        product: CostProductOutput,
        at_length_boundary: bool,
    ) -> StructuredInputV2 {
        natural_termination_input_at_frontier(rows, product, at_length_boundary, 64)
    }

    fn natural_termination_input_at_frontier(
        rows: u32,
        product: CostProductOutput,
        at_length_boundary: bool,
        kv_tokens: u32,
    ) -> StructuredInputV2 {
        natural_termination_input_with_algorithm(
            rows,
            product,
            at_length_boundary,
            kv_tokens,
            "fixture.selection",
            [1; 32],
        )
    }

    fn natural_termination_input_with_algorithm(
        rows: u32,
        product: CostProductOutput,
        at_length_boundary: bool,
        kv_tokens: u32,
        native_op_id: &'static str,
        implementation: [u8; 32],
    ) -> StructuredInputV2 {
        natural_termination_input_with_algorithm_and_eos(
            rows,
            product,
            at_length_boundary,
            kv_tokens,
            native_op_id,
            implementation,
            true,
        )
    }

    fn natural_termination_input_with_algorithm_and_eos(
        rows: u32,
        product: CostProductOutput,
        at_length_boundary: bool,
        kv_tokens: u32,
        native_op_id: &'static str,
        implementation: [u8; 32],
        model_eos: bool,
    ) -> StructuredInputV2 {
        let mut selected = SelectedCommandCostBuilderV1::new_with_algorithm_work(u64::from(rows));
        selected
            .kernel(
                SelectedAlgorithmClassV1::new(native_op_id, 1, implementation, [2; 32]).unwrap(),
                KernelNumericWorkV1 {
                    logical_units: u64::from(rows) * 8,
                    padded_units: u64::from(rows) * 8,
                    inner_units_per_logical_unit: 2,
                    grid: [rows, 1, 1],
                    scratch_bytes: u64::from(rows) * 32,
                    staged_weight_bytes: 0,
                },
            )
            .unwrap();
        let selected = selected.finish().unwrap();
        let mut builder = CanonicalWaveCostBuilder::new_with_structured_statistics(0, product);
        builder
            .physical_command(CostPhysicalCommand {
                native_op_id,
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
                token_count: u64::from(rows),
                batching_form: "packed",
                compute_dispatch_count: 1,
                transfer_command_count: 0,
                reusable_graph_node_count: None,
                statistical_evidence: Some(&selected),
            })
            .unwrap();
        builder
            .core_readback_route(CoreReadbackRoute::SubmissionStaged)
            .unwrap();
        for _ in 0..rows {
            builder
                .row(CanonicalCostRow {
                    work: ActualRowWork::Decode { kv_tokens },
                    host_policy_signature: [3; 32],
                    mask_upload_required: false,
                    output: CostRowOutput::Decode {
                        requires_full_logits: product == CostProductOutput::FullLogits,
                        repetition_tokens: 0,
                        repetition_penalty_bits: 1f32.to_bits(),
                    },
                    host_features: Some(HostCostFeaturesV1 {
                        policy: HostCostPolicyV2 {
                            empirical_content_domain: Some(
                                HostContentDomainV1::PlainTextInstalledV2(
                                    PlainTextPolicyCapabilityV2 {
                                        sampling: if product == CostProductOutput::FullLogits {
                                            PlainTextSamplingRouteV2::FullLogits
                                        } else {
                                            PlainTextSamplingRouteV2::Greedy {
                                                repetition_penalty: false,
                                            }
                                        },
                                        model_eos,
                                        user_stop: false,
                                    },
                                ),
                            ),
                            categorical_signature: [4; 32],
                            decoder_text_bytes_per_token: 4,
                            decoder_scratch_bytes_per_token: 8,
                            raw_token_bytes_bound: 4,
                        },
                        state: HostCostStateV1 {
                            generated_tokens_before: 3,
                            maximum_output_tokens: if at_length_boundary { 4 } else { 21 },
                            sampling_history_tokens: 3,
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
                ActualWaveKind::Decode,
                ActualWavePath::PlanRuntime,
                ActualWaveGraphState::Disabled,
                ActualWaveRowOrder::Ordered,
                u64::from(rows) * 32,
            )
            .unwrap();
        let selected = wave.statistical.as_ref().unwrap();
        StructuredInputV2::from_actual_with_domain(
            &wave.exact,
            selected,
            selected.structured_capture().unwrap().unwrap(),
            &fixture::domain(),
        )
        .unwrap()
    }

    fn install_natural_termination_facts(
        cases: &mut [Case],
        opportunities: &mut [CaseOpportunity],
        facts: &mut [Vec<CheckedInputFacts>],
        at_length_boundary: bool,
    ) {
        for ((case, opportunity), facts) in cases.iter_mut().zip(opportunities).zip(facts) {
            let product = if case.product == OpportunityProduct::Full {
                CostProductOutput::FullLogits
            } else {
                CostProductOutput::GreedyToken
            };
            let input = natural_termination_input(case.width as u32, product, at_length_boundary);
            opportunity.population = classify_alternatives(
                std::slice::from_ref(&input),
                StructuredPopulationPolicyV1::HomogeneousOrdinaryDecodeV1,
                true,
            )
            .unwrap();
            *facts = vec![input_facts(&StructuredQueryV2::exact(input)).unwrap()];
            assert!(facts[0].branches[4]);
            assert_eq!(facts[0].branches[5], !at_length_boundary);
            if at_length_boundary {
                case.maximum_output = NonZeroUsize::new(4).unwrap();
                case.suffix_tokens = 1;
            }
        }
    }

    fn inventory() -> (
        Vec<Case>,
        Vec<CaseOpportunity>,
        Vec<Vec<CheckedInputFacts>>,
        StructuredServiceDeclarationV7,
    ) {
        let mut cases = Vec::new();
        let mut opportunities = Vec::new();
        let mut facts = Vec::new();
        for width in [1, 2, 4, 4] {
            let input = fixture::input(width, 3, CostProductOutput::GreedyToken, false, true);
            opportunities.push(CaseOpportunity {
                population: classify_alternatives(
                    std::slice::from_ref(&input),
                    StructuredPopulationPolicyV1::HomogeneousOrdinaryDecodeV1,
                    true,
                )
                .unwrap(),
                minimum_fresh_members: 1,
            });
            facts.push(vec![input_facts(&StructuredQueryV2::exact(input)).unwrap()]);
            cases.push(Case {
                product: OpportunityProduct::Greedy,
                template: 0,
                width: width as usize,
                maximum_output: NonZeroUsize::new(21).unwrap(),
                release_generated: 3,
                suffix_tokens: 18,
                preset: SloAutomaticCostProbeSamplingPresetV1::Configured,
                prefix: PrefixKind::Clean,
                route: CalibrationDecodeRoute::Actual,
                reset: false,
                acquisition: None,
            });
        }
        let population = population::declaration(&Default::default(), fixture::domain()).unwrap();
        (cases, opportunities, facts, population)
    }

    #[test]
    fn checked_selection_preserves_real_axis_endpoints_without_width_cartesian_repeats() {
        let (cases, opportunities, facts, population) = inventory();
        let result = select(
            &cases,
            &opportunities,
            &facts,
            &[61],
            8,
            &population,
            100_000,
            10_000_000,
            usize::MAX,
        )
        .unwrap();
        assert_eq!(result.populations.len(), 1);
        assert_eq!(result.populations[0].representative_case_indices, [0, 2]);
        assert!(result.populations[0].scheduled);
        assert!(result.gaps.is_empty());
        assert!(result
            .execution_case_indices
            .iter()
            .all(|i| *i == 0 || *i == 2));
        let actual_requests: usize = result
            .execution_case_indices
            .iter()
            .map(|&i| cases[i].width)
            .sum();
        let actual_waves: usize = result
            .execution_case_indices
            .iter()
            .map(|&i| cases[i].waves(61, 8).unwrap().1)
            .sum();
        assert_eq!(result.requests, actual_requests);
        assert_eq!(result.serial_wave_upper_bound, actual_waves);
        assert!(
            result.populations[0].maximum_anchor_span
                <= result.batches[0].schedule.phase_min_offered[0]
        );
    }

    #[test]
    fn checked_selection_keeps_budget_gaps_and_derives_complete_anchor_window() {
        let (cases, opportunities, facts, mut population) = inventory();
        let limited = select(
            &cases,
            &opportunities,
            &facts,
            &[61],
            8,
            &population,
            1,
            1,
            usize::MAX,
        )
        .unwrap();
        assert!(limited.execution_case_indices.is_empty());
        assert_eq!(limited.requests, 0);
        assert_eq!(limited.serial_wave_upper_bound, 0);
        assert!(limited
            .gaps
            .iter()
            .any(|gap| matches!(gap.reason, SelectionGapReason::RemainingRequests { .. })));
        assert!(limited
            .gaps
            .iter()
            .any(|gap| matches!(gap.reason, SelectionGapReason::RemainingWaves { .. })));
        population.schedule.phase_min_offered = [8; 3];
        let sparse = select(
            &cases,
            &opportunities,
            &facts,
            &[61],
            8,
            &population,
            100_000,
            10_000_000,
            usize::MAX,
        )
        .unwrap();
        assert!(!sparse.execution_case_indices.is_empty());
        assert!(sparse.gaps.is_empty());
        let batch = &sparse.batches[0];
        assert!(batch.maximum_anchor_span > population.schedule.phase_min_offered[0]);
        assert!(batch
            .schedule
            .phase_min_offered
            .iter()
            .all(|minimum| { *minimum >= batch.maximum_anchor_span }));
    }

    #[test]
    fn checked_selection_rejects_unbound_facts_and_never_promotes_possible_members() {
        let unbound = fixture::input(1, 3, CostProductOutput::GreedyToken, false, false);
        assert_eq!(
            input_facts(&StructuredQueryV2::exact(unbound)),
            Err(StructuredUnknownV2::WrongDomain)
        );
        let (cases, mut opportunities, facts, population) = inventory();
        for opportunity in &mut opportunities {
            opportunity.minimum_fresh_members = 0;
        }
        let result = select(
            &cases,
            &opportunities,
            &facts,
            &[61],
            8,
            &population,
            100_000,
            10_000_000,
            usize::MAX,
        )
        .unwrap();
        assert!(result.execution_case_indices.is_empty());
        assert!(result
            .gaps
            .iter()
            .any(|gap| matches!(gap.reason, SelectionGapReason::NoGuaranteedMember)));
    }

    fn two_families() -> (
        Vec<Case>,
        Vec<CaseOpportunity>,
        Vec<Vec<CheckedInputFacts>>,
        StructuredServiceDeclarationV7,
    ) {
        let (mut cases, mut opportunities, mut facts, population) = inventory();
        for mut case in cases.clone() {
            case.product = OpportunityProduct::Full;
            let input = fixture::input(
                case.width as u32,
                3,
                CostProductOutput::FullLogits,
                false,
                true,
            );
            opportunities.push(CaseOpportunity {
                population: classify_alternatives(
                    std::slice::from_ref(&input),
                    StructuredPopulationPolicyV1::HomogeneousOrdinaryDecodeV1,
                    true,
                )
                .unwrap(),
                minimum_fresh_members: 1,
            });
            facts.push(vec![input_facts(&StructuredQueryV2::exact(input)).unwrap()]);
            cases.push(case);
        }
        (cases, opportunities, facts, population)
    }

    #[test]
    fn checked_selection_reserves_tight_budget_for_later_input_complete_population() {
        let (mut cases, mut opportunities, mut facts, mut population) = two_families();
        install_natural_termination_facts(
            &mut cases[..4],
            &mut opportunities[..4],
            &mut facts[..4],
            false,
        );
        // Both inputs use the same declared preset. Priority must come from
        // the checked installed policy, not a preset/product name or outcome.
        assert!(cases
            .iter()
            .all(|case| case.preset == SloAutomaticCostProbeSamplingPresetV1::Configured));
        population.schedule.phase_min_offered = [180; 3];
        let later = select(
            &cases[4..],
            &opportunities[4..],
            &facts[4..],
            &[61],
            8,
            &population,
            100_000,
            10_000_000,
            usize::MAX,
        )
        .unwrap();
        assert!(later.populations[0].scheduled);
        let expected: Vec<_> = later
            .execution_case_indices
            .iter()
            .map(|index| index + 4)
            .collect();
        for (requests, waves) in [
            (later.requests, 10_000_000),
            (100_000, later.serial_wave_upper_bound),
        ] {
            let selected = select(
                &cases,
                &opportunities,
                &facts,
                &[61],
                8,
                &population,
                requests,
                waves,
                usize::MAX,
            )
            .unwrap();
            // The original group/case indices remain intact; only allocation
            // order changes. The early-dependent group is still diagnosed.
            assert_eq!(selected.populations.len(), 2);
            assert!(!selected.populations[0].scheduled);
            assert!(selected.populations[1].scheduled);
            assert_eq!(selected.execution_case_indices, expected);
            assert!(selected.requests <= requests);
            assert!(selected.serial_wave_upper_bound <= waves);
            assert!(selected.gaps.iter().any(|gap| {
                gap.population.as_ref() == Some(&selected.populations[0].key)
                    && matches!(
                        gap.reason,
                        SelectionGapReason::OutcomeDependentEarlyTermination
                    )
            }));
        }
    }

    #[test]
    fn checked_selection_missing_early_input_stays_explicit_under_complete_work_order() {
        let (mut cases, mut opportunities, mut facts, population) = two_families();
        install_natural_termination_facts(
            &mut cases[..4],
            &mut opportunities[..4],
            &mut facts[..4],
            true,
        );
        install_natural_termination_facts(
            &mut cases[4..],
            &mut opportunities[4..],
            &mut facts[4..],
            false,
        );
        let selected = select(
            &cases,
            &opportunities,
            &facts,
            &[61],
            8,
            &population,
            100_000,
            10_000_000,
            usize::MAX,
        )
        .unwrap();
        let allocation_order: Vec<_> = selected
            .batches
            .iter()
            .flat_map(|batch| batch.population_indices.iter().copied())
            .collect();
        // The length-bounded source has less complete declared work. Its
        // missing early-terminal opportunity remains an explicit gap even
        // though work order now precedes EOS/input risk.
        assert_eq!(allocation_order, [0, 1]);
        assert!(selected.batches[0].serial_token_work < selected.batches[1].serial_token_work);
        assert!(selected.gaps.iter().any(|gap| {
            gap.population.as_ref() == Some(&selected.populations[0].key)
                && matches!(
                    gap.reason,
                    SelectionGapReason::EarlyTerminalOpportunityMissing
                )
        }));
        assert!(selected.gaps.iter().any(|gap| {
            gap.population.as_ref() == Some(&selected.populations[1].key)
                && matches!(
                    gap.reason,
                    SelectionGapReason::OutcomeDependentEarlyTermination
                )
        }));
        // No completion/EOS evidence was constructed. Scheduling a declared
        // opportunity does not remove either original qualification gap.
        assert!(facts[..4]
            .iter()
            .all(|rows| rows[0].branches[4] && !rows[0].branches[5]));
    }

    #[test]
    fn checked_selection_equal_input_priority_and_work_keep_original_population_order() {
        let (mut cases, mut opportunities, mut facts, population) = two_families();
        install_natural_termination_facts(&mut cases, &mut opportunities, &mut facts, false);
        for case in &mut cases[4..] {
            case.template = 1;
        }
        let selected = select(
            &cases,
            &opportunities,
            &facts,
            &[61, 61],
            8,
            &population,
            100_000,
            10_000_000,
            usize::MAX,
        )
        .unwrap();
        assert!(selected.populations.iter().all(|group| group.scheduled));
        assert_eq!(selected.batches.len(), 2);
        let first = &selected.batches[0];
        let second = &selected.batches[1];
        assert_eq!(first.serial_token_work, second.serial_token_work);
        assert_eq!(
            first.serial_wave_upper_bound,
            second.serial_wave_upper_bound
        );
        assert_eq!(first.requests, second.requests);
        let allocation_order: Vec<_> = selected
            .batches
            .iter()
            .flat_map(|batch| batch.population_indices.iter().copied())
            .collect();
        assert_eq!(allocation_order, [0, 1]);
    }

    #[test]
    fn checked_selection_keeps_complete_independent_horizons_for_two_source_batches() {
        let (mut cases, opportunities, facts, population) = two_families();
        // Distinct rendered inputs retain independent source horizons even
        // when their declared work happens to be equal.
        for case in &mut cases[4..] {
            case.template = 1;
        }
        let separate = select(
            &cases[..4],
            &opportunities[..4],
            &facts[..4],
            &[61],
            8,
            &population,
            100_000,
            10_000_000,
            usize::MAX,
        )
        .unwrap();
        let independent = select(
            &cases,
            &opportunities,
            &facts,
            &[61, 61],
            8,
            &population,
            separate.requests * 2,
            separate.serial_wave_upper_bound * 2,
            usize::MAX,
        )
        .unwrap();
        assert_eq!(independent.populations.len(), 2);
        assert_eq!(independent.batches.len(), 2);
        let standalone = &separate.batches[0];
        let mut expected = Vec::new();
        for (index, batch) in independent.batches.iter().enumerate() {
            assert_eq!(batch.population_indices, [index]);
            assert_eq!(
                batch.representative_case_indices,
                [index * 4, index * 4 + 2]
            );
            assert!(batch.scheduled);
            assert_eq!(independent.populations[index].batch_index, Some(index));
            assert_eq!(batch.planned_cycles, standalone.planned_cycles);
            assert_eq!(batch.requests, standalone.requests);
            assert_eq!(
                batch.serial_wave_upper_bound,
                standalone.serial_wave_upper_bound
            );
            assert_eq!(batch.serial_token_work, standalone.serial_token_work);
            let horizon = &batch.input_opportunities;
            assert_eq!(
                horizon.required_original_offers,
                standalone.input_opportunities.required_original_offers
            );
            assert_eq!(
                horizon.phase_original_offer_bounds,
                standalone.input_opportunities.phase_original_offer_bounds
            );
            assert_eq!(
                horizon.maximum_fresh_member_span,
                standalone.input_opportunities.maximum_fresh_member_span
            );
            assert!(
                batch.planned_cycles * horizon.minimum_original_offers_per_completed_cycle
                    >= horizon.required_original_offers + batch.schedule.block_offered
            );
            assert!(horizon
                .phase_original_offer_bounds
                .iter()
                .zip(batch.schedule.phase_min_offered)
                .all(|(offers, minimum)| *offers >= minimum));
            let expanded = batch
                .representative_case_indices
                .repeat(batch.planned_cycles);
            assert_eq!(
                batch.serial_token_work,
                expanded
                    .iter()
                    .map(|&i| { cases[i].width * (61 + cases[i].maximum_output.get() - 1) })
                    .sum::<usize>()
            );
            expected.extend(expanded);
        }
        assert_eq!(
            &independent.execution_case_indices[..separate.execution_case_indices.len()],
            separate.execution_case_indices.as_slice(),
            "the first complete source batch precedes every case of the second"
        );
        assert_eq!(independent.execution_case_indices, expected);
        assert_eq!(independent.requests, separate.requests * 2);
        assert_eq!(
            independent.serial_wave_upper_bound,
            separate.serial_wave_upper_bound * 2
        );
        for (requests, waves) in [
            (
                independent.requests - 1,
                independent.serial_wave_upper_bound,
            ),
            (
                independent.requests,
                independent.serial_wave_upper_bound - 1,
            ),
        ] {
            let tight = select(
                &cases,
                &opportunities,
                &facts,
                &[61, 61],
                8,
                &population,
                requests,
                waves,
                usize::MAX,
            )
            .unwrap();
            assert!(tight.populations[0].scheduled);
            assert!(!tight.populations[1].scheduled);
            assert_eq!(
                tight.execution_case_indices,
                separate.execution_case_indices
            );
            assert!(tight.gaps.iter().any(|gap| {
                gap.population.as_ref() == Some(&tight.populations[1].key)
                    && matches!(
                        gap.reason,
                        SelectionGapReason::RemainingRequests { .. }
                            | SelectionGapReason::RemainingWaves { .. }
                    )
            }));
        }
    }

    #[test]
    fn checked_selection_orders_equal_priority_by_complete_declared_work() {
        let (mut cases, mut opportunities, mut facts, population) = two_families();
        // Both physical inputs decode at KV frontier 64. The second family's
        // real checked host history is longer, so its prompt is shorter. Its
        // complete phase horizon must include serial preparation, not just
        // prompt lengths. Give it one fewer ordinary suffix token as well:
        // equal prompt+output token totals can otherwise tie after the
        // complete fresh-member horizon is rounded.
        for index in 4..cases.len() {
            let case = &mut cases[index];
            case.template = 1;
            case.release_generated = 10;
            case.suffix_tokens = 17;
            case.maximum_output =
                NonZeroUsize::new(case.release_generated + case.suffix_tokens).unwrap();
            let input = fixture::input(
                case.width as u32,
                10,
                CostProductOutput::FullLogits,
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
        let selected = select(
            &cases,
            &opportunities,
            &facts,
            &[61, 54],
            8,
            &population,
            100_000,
            10_000_000,
            usize::MAX,
        )
        .unwrap();
        assert_eq!(selected.populations.len(), 2);
        assert!(selected.populations.iter().all(|group| group.scheduled));
        assert_eq!(selected.batches[0].population_indices, [1]);
        assert_eq!(selected.batches[1].population_indices, [0]);
        assert!(selected.batches[0].serial_token_work < selected.batches[1].serial_token_work);
        let first = &selected.batches[0];
        let tight = select(
            &cases,
            &opportunities,
            &facts,
            &[61, 54],
            8,
            &population,
            first.requests,
            first.serial_wave_upper_bound,
            usize::MAX,
        )
        .unwrap();
        assert!(tight.populations[1].scheduled);
        assert!(!tight.populations[0].scheduled);
        assert_eq!(
            tight.execution_case_indices,
            first
                .representative_case_indices
                .repeat(first.planned_cycles)
        );
        assert!(tight.gaps.iter().any(|gap| {
            gap.population.as_ref() == Some(&tight.populations[0].key)
                && matches!(
                    gap.reason,
                    SelectionGapReason::RemainingRequests { .. }
                        | SelectionGapReason::RemainingWaves { .. }
                )
        }));
    }

    #[test]
    fn checked_selection_recomputes_shared_source_anchor_window() {
        // Independent raw algorithm families share a journal only when the
        // actual product, readback, installed policy and complete widths agree.
        // Preserve the original full-cycle anchor/phase recomputation check.
        let (cases, opportunities, facts, mut population) =
            journal_grouping::same_product_families();
        population.schedule.phase_min_offered = [180; 3];
        let split = select(
            &cases,
            &opportunities,
            &facts,
            &[61],
            8,
            &population,
            100_000,
            10_000_000,
            usize::MAX,
        )
        .unwrap();
        assert_eq!(split.batches.len(), 1);
        assert_eq!(split.batches[0].population_indices, [0, 1]);
        assert!(split.batches.iter().all(|batch| batch.scheduled
            && batch.maximum_anchor_span <= batch.schedule.phase_min_offered[0]));
        assert!(split.batches[0].maximum_anchor_span > 180);
        let expected: Vec<_> = split
            .batches
            .iter()
            .flat_map(|batch| {
                batch
                    .representative_case_indices
                    .repeat(batch.planned_cycles)
            })
            .collect();
        assert_eq!(split.execution_case_indices, expected);
        let requests: usize = split
            .execution_case_indices
            .iter()
            .map(|&index| cases[index].width)
            .sum();
        assert_eq!(split.requests, requests);
    }

    #[test]
    fn checked_selection_reserves_grouping_then_selection_before_execution_allocation() {
        let (cases, opportunities, facts, population) = two_families();
        let peak = selection_peak_payload_bound(&cases, &opportunities, &facts, 2048).unwrap();
        let rejected = select(
            &cases,
            &opportunities,
            &facts,
            &[61],
            8,
            &population,
            2048,
            10_000_000,
            peak - 1,
        );
        assert!(rejected
            .unwrap_err()
            .to_string()
            .contains("peak retained payload"));
        let accepted = select(
            &cases,
            &opportunities,
            &facts,
            &[61],
            8,
            &population,
            2048,
            10_000_000,
            peak,
        )
        .unwrap();
        assert!(accepted.batches[0].scheduled);
        assert!(accepted.retained_payload_bytes().unwrap() <= peak);

        // Expanded execution is bounded before reserving, even when the caller
        // supplies an allowance whose capacity arithmetic cannot be represented.
        assert!(selection_peak_payload_bound(&cases, &opportunities, &facts, usize::MAX).is_err());
        assert!(select(
            &cases,
            &opportunities,
            &facts,
            &[61],
            8,
            &population,
            usize::MAX,
            usize::MAX,
            usize::MAX
        )
        .is_err());
        assert!(vector_peak_bytes::<usize>(usize::MAX).is_err());
    }

    #[test]
    fn checked_selection_oversized_later_batch_preserves_schedulable_populations() {
        let (mut cases, opportunities, facts, population) = two_families();
        for case in &mut cases[4..] {
            case.template = 1;
        }
        let first = select(
            &cases[..4],
            &opportunities[..4],
            &facts[..4],
            &[61],
            8,
            &population,
            100_000,
            10_000_000,
            usize::MAX,
        )
        .unwrap();
        for (requests, waves) in [
            (first.requests, 10_000_000),
            (100_000, first.serial_wave_upper_bound),
        ] {
            let result = select(
                &cases,
                &opportunities,
                &facts,
                &[61, 61],
                8,
                &population,
                requests,
                waves,
                usize::MAX,
            )
            .unwrap();
            assert!(result.populations[0].scheduled);
            assert!(!result.populations[1].scheduled);
            assert_eq!(result.execution_case_indices, first.execution_case_indices);
            assert!(result.requests <= requests);
            assert!(result.serial_wave_upper_bound <= waves);
            assert!(result.gaps.iter().any(|gap| matches!(
                gap.reason,
                SelectionGapReason::RemainingRequests { .. }
                    | SelectionGapReason::RemainingWaves { .. }
            )));
        }
    }

    #[test]
    fn checked_selection_counts_mandatory_prefill_chunks_in_each_member_floor() {
        let (cases, opportunities, facts, population) = inventory();
        let selected = select(
            &cases,
            &opportunities,
            &facts,
            &[61],
            8,
            &population,
            2048,
            10_000_000,
            usize::MAX,
        )
        .unwrap();
        assert_eq!(selected.batches[0].representative_case_indices, [0, 2]);
        assert_eq!(
            selected.batches[0]
                .input_opportunities
                .minimum_original_offers_per_completed_cycle,
            8 + 3 + 4 * 31 + 3
        );
        for (index, chunks) in [(0, 8), (2, 31)] {
            let mut case = cases[index].clone();
            assert_eq!(
                work::case_work(&case, 61, 8, None)
                    .unwrap()
                    .declared_offers_minimum,
                chunks * case.width + 3
            );
            case.release_generated = 0;
            case.prefix = PrefixKind::Ordinary;
            assert_eq!(
                work::case_work(&case, 61, 8, None)
                    .unwrap()
                    .declared_offers_minimum,
                chunks
            );
            case.preset = SloAutomaticCostProbeSamplingPresetV1::GreedyLength;
            let work = work::case_work(&case, 61, 8, None).unwrap();
            assert_eq!(work.declared_offers_minimum, work.declared_offers_upper);
        }
    }
}
