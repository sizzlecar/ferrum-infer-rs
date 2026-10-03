//! Cold checked input inventory. No projection is a numerical sample, and a
//! conditional cohort opportunity never replaces actual source8 membership.
use super::populations::{CaseOpportunity, CasePopulation, CheckedPopulationKey};
use super::selection::{input_facts, CheckedInputFacts};
use super::*;
use crate::continuous_engine::inner::calibration::geometry_projection::{
    GeometryInputScenario, GeometryInputTarget, GeometryPrefixConstraint, GeometryProjectionCharge,
    GeometryProjectionLimits, GeometryProjectionPoint, GeometryProjectionUnknown,
    GeometryReadinessReport,
};
use ferrum_interfaces::execution_cost::{ActualWaveGraphState, PreparedCostRouteClassV1};
use ferrum_scheduler::implementations::continuous::cost_model::structured_v2::{
    StructuredInputV2, StructuredPopulationPolicyV1, StructuredUnknownV2,
};
use std::sync::Arc;
use tokio::time::Instant;

mod readiness;
pub(super) use readiness::{
    collect_charged, probe_readiness, ReadinessCapture, ReadinessCursor, ReadinessProgress,
};

#[cfg(test)]
async fn collect(
    session: &mut CalibrationSession,
    cases: &[Case],
    templates: &[AutomaticCostProbeTemplate],
    prompts: &[usize],
    pair: &PrefixPair,
    policy: StructuredPopulationPolicyV1,
    decode_boundaries: &[u32],
    limits: InventoryLimits,
) -> Result<CheckedCaseInventory> {
    collect_charged(
        session,
        cases,
        templates,
        prompts,
        pair,
        policy,
        decode_boundaries,
        limits,
        &mut ProbePreflightCharge::default(),
    )
    .await
}

pub(super) struct InventoryLimits {
    pub route_population: ferrum_types::SloCalibrationRoutePopulationV1,
    pub deadline: Instant,
    pub maximum_admitted_requests: usize,
    pub maximum_projection_attempts: usize,
    /// Remaining cold inventory payload, including query/slot temporaries.
    /// Admitted execution owners retain the session's original resource limits.
    pub maximum_retained_bytes: usize,
    pub maximum_route_states: usize,
    pub prefill_chunk: NonZeroU32,
    pub prefill_row_ceiling: Option<NonZeroU32>,
}

#[derive(Debug, Clone)]
pub(super) enum InventoryGapReason {
    Projection(GeometryProjectionUnknown),
    MultiplePopulations,
    PrefixReleaseUnproven,
    WarmResidencyUnproven,
    OutsideDeclaredRoute {
        population: ferrum_types::SloCalibrationRoutePopulationV1,
        projected_graph: ActualWaveGraphState,
    },
}

#[derive(Debug, Clone)]
pub(super) struct InventoryGap {
    pub case_index: usize,
    pub reason: InventoryGapReason,
}

#[derive(Clone)]
pub(super) struct CheckedCaseInventory {
    pub opportunities: Vec<CaseOpportunity>,
    /// Every checked alternative, with actual axes and original row facts.
    pub inputs: Vec<Vec<CheckedInputFacts>>,
    /// Additional checked raw inputs declare finite algorithm support only.
    /// They never supply a numerical sample or increase a case's member floor.
    pub algorithm_inputs: Vec<Arc<StructuredInputV2>>,
    /// Original case to checked trajectory recipes. These input-only links
    /// neither grant a member floor nor borrow algorithms from another case.
    pub algorithm_case_inputs: Vec<Vec<usize>>,
    pub gaps: Vec<InventoryGap>,
    pub charge: ProbePreflightCharge,
}

impl CheckedCaseInventory {
    /// The terminal global inventory cannot grow after its final projection.
    /// Drop the raw authority only after that projection has succeeded; the
    /// frozen universe, exact keys, numerical axes and typed gaps remain.
    pub(super) fn release_original_inputs(&mut self) {
        self.algorithm_inputs = Vec::new();
        self.algorithm_case_inputs = Vec::new();
        for facts in self.inputs.iter_mut().flatten() {
            facts.original = None;
        }
    }

    pub(super) fn retained_payload_bytes(&self) -> Option<usize> {
        let mut bytes = std::mem::size_of::<Self>()
            .checked_add(
                self.opportunities
                    .capacity()
                    .checked_mul(std::mem::size_of::<CaseOpportunity>())?,
            )?
            .checked_add(
                self.inputs
                    .capacity()
                    .checked_mul(std::mem::size_of::<Vec<CheckedInputFacts>>())?,
            )?
            .checked_add(
                self.gaps
                    .capacity()
                    .checked_mul(std::mem::size_of::<InventoryGap>())?,
            )?
            .checked_add(
                self.algorithm_inputs
                    .capacity()
                    .checked_mul(std::mem::size_of::<Arc<StructuredInputV2>>())?,
            )?
            .checked_add(
                self.algorithm_case_inputs
                    .capacity()
                    .checked_mul(std::mem::size_of::<Vec<usize>>())?,
            )?;
        for indices in &self.algorithm_case_inputs {
            bytes = bytes.checked_add(
                indices
                    .capacity()
                    .checked_mul(std::mem::size_of::<usize>())?,
            )?;
        }
        for input in &self.algorithm_inputs {
            bytes = bytes
                .checked_add(input.retained_payload_bytes()?)?
                .checked_add(2 * std::mem::size_of::<usize>())?;
        }
        for opportunity in &self.opportunities {
            let capacity = match &opportunity.population {
                CasePopulation::Unique(_) => 0,
                CasePopulation::Alternatives(keys) => keys.capacity(),
                CasePopulation::Unknown { known_alternatives } => known_alternatives.capacity(),
            };
            bytes = bytes
                .checked_add(capacity.checked_mul(std::mem::size_of::<CheckedPopulationKey>())?)?;
        }
        for inputs in &self.inputs {
            bytes = bytes.checked_add(
                inputs
                    .capacity()
                    .checked_mul(std::mem::size_of::<CheckedInputFacts>())?,
            )?;
            for input in inputs {
                bytes = bytes.checked_add(
                    input
                        .retained_payload_bytes()?
                        .checked_sub(std::mem::size_of::<CheckedInputFacts>())?,
                )?;
            }
        }
        Some(bytes)
    }

    /// The caller charges every still-live report, scratch array and source
    /// input in external_bytes. Authorize both old/new Vec backing and the
    /// owned raw clone before allocating; no deadline or request limit renews.
    pub(super) fn retain_algorithm_input(
        &mut self,
        input: &StructuredInputV2,
        maximum_retained_bytes: usize,
        external_bytes: usize,
    ) -> Result<Option<usize>> {
        let key = match input.numerical_family_key() {
            Ok(key) => key,
            Err(StructuredUnknownV2::UnsupportedScope) => return Ok(None),
            Err(reason) => return Err(error(format!("algorithm inventory input: {reason:?}"))),
        };
        if input.algorithm_universe_signature().is_some() {
            return Err(error(
                "algorithm inventory requires original unmapped input",
            ));
        }
        require_capacity(
            self.retained_payload_bytes()
                .and_then(|n| n.checked_add(external_bytes)),
            maximum_retained_bytes,
        )?;
        for (index, existing) in self.algorithm_inputs.iter().enumerate() {
            let existing_key = existing
                .numerical_family_key()
                .map_err(|reason| error(format!("retained algorithm input: {reason:?}")))?;
            if existing_key == key {
                return Ok(Some(index));
            }
        }
        let incoming = input
            .retained_payload_bytes()
            .and_then(|n| n.checked_add(2 * std::mem::size_of::<usize>()))
            .ok_or_else(|| error("algorithm input payload overflow"))?;
        let growth = if self.algorithm_inputs.len() == self.algorithm_inputs.capacity() {
            self.algorithm_inputs
                .len()
                .checked_add(1)
                .and_then(|n| n.checked_mul(std::mem::size_of::<Arc<StructuredInputV2>>()))
                .ok_or_else(|| error("algorithm input backing overflow"))?
        } else {
            0
        };
        require_capacity(
            self.retained_payload_bytes()
                .and_then(|n| n.checked_add(external_bytes))
                .and_then(|n| n.checked_add(growth))
                .and_then(|n| n.checked_add(incoming)),
            maximum_retained_bytes,
        )?;
        if growth != 0 {
            self.algorithm_inputs
                .try_reserve_exact(1)
                .map_err(|_| error("algorithm input backing allocation failed"))?;
        }
        // Account actual capacity before cloning even if the allocator supplied
        // more backing than the exact reservation requested.
        require_capacity(
            self.retained_payload_bytes()
                .and_then(|n| n.checked_add(external_bytes))
                .and_then(|n| n.checked_add(incoming)),
            maximum_retained_bytes,
        )?;
        self.algorithm_inputs.push(Arc::new(input.clone()));
        Ok(Some(self.algorithm_inputs.len() - 1))
    }

    fn link_algorithm_input(
        &mut self,
        case: usize,
        input: usize,
        maximum_retained_bytes: usize,
        external_bytes: usize,
    ) -> Result<()> {
        let indices = self
            .algorithm_case_inputs
            .get(case)
            .ok_or_else(|| error("algorithm trajectory case outside inventory"))?;
        if input >= self.algorithm_inputs.len() {
            return Err(error("algorithm trajectory input outside checked pool"));
        }
        if indices.contains(&input) {
            return Ok(());
        }
        let growth = if indices.len() == indices.capacity() {
            indices
                .len()
                .checked_add(1)
                .and_then(|n| n.checked_mul(std::mem::size_of::<usize>()))
                .ok_or_else(|| error("algorithm trajectory link capacity overflow"))?
        } else {
            0
        };
        require_capacity(
            self.retained_payload_bytes()
                .and_then(|n| n.checked_add(external_bytes))
                .and_then(|n| n.checked_add(growth)),
            maximum_retained_bytes,
        )?;
        if growth != 0 {
            self.algorithm_case_inputs[case]
                .try_reserve_exact(1)
                .map_err(|_| error("algorithm trajectory link allocation failed"))?;
        }
        require_capacity(
            self.retained_payload_bytes()
                .and_then(|n| n.checked_add(external_bytes)),
            maximum_retained_bytes,
        )?;
        self.algorithm_case_inputs[case].push(input);
        Ok(())
    }

    /// Remap a completed capture's bounded links into the destination pool.
    /// Both inventories and the caller's scratch remain charged during merge.
    pub(super) fn merge_algorithms(
        &mut self,
        captured: &Self,
        cases: &[usize],
        maximum_retained_bytes: usize,
        external_bytes: usize,
    ) -> Result<()> {
        if cases.len() != captured.algorithm_case_inputs.len() {
            return Err(error("algorithm trajectory merge case alignment differs"));
        }
        for input in &captured.algorithm_inputs {
            self.retain_algorithm_input(input, maximum_retained_bytes, external_bytes)?;
        }
        for (&case, inputs) in cases.iter().zip(&captured.algorithm_case_inputs) {
            for &index in inputs {
                let original = captured
                    .algorithm_inputs
                    .get(index)
                    .ok_or_else(|| error("captured trajectory has no checked input"))?;
                let index = self
                    .retain_algorithm_input(original, maximum_retained_bytes, external_bytes)?
                    .ok_or_else(|| error("captured trajectory lost its numerical scope"))?;
                self.link_algorithm_input(case, index, maximum_retained_bytes, external_bytes)?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod algorithm_pool_tests {
    use super::super::populations::tests as fixture;
    use super::*;
    use ferrum_interfaces::execution_cost::CostProductOutput;

    fn empty() -> CheckedCaseInventory {
        CheckedCaseInventory {
            opportunities: Vec::new(),
            inputs: Vec::new(),
            algorithm_inputs: Vec::new(),
            algorithm_case_inputs: Vec::new(),
            gaps: Vec::new(),
            charge: ProbePreflightCharge::default(),
        }
    }

    #[test]
    fn algorithm_pool_authorizes_clone_and_deduplicates_checked_widths() {
        let input = fixture::input(1, 3, CostProductOutput::GreedyToken, false, true);
        let wider = fixture::input(2, 3, CostProductOutput::GreedyToken, false, true);
        let mut inventory = empty();
        let external = 73;
        let maximum = inventory.retained_payload_bytes().unwrap()
            + external
            + input.retained_payload_bytes().unwrap()
            + 2 * std::mem::size_of::<usize>()
            + std::mem::size_of::<Arc<StructuredInputV2>>();
        assert!(inventory
            .retain_algorithm_input(&input, maximum - 1, external)
            .is_err());
        assert_eq!(inventory.algorithm_inputs.capacity(), 0);
        inventory
            .retain_algorithm_input(&input, maximum, external)
            .unwrap();
        let retained = inventory.retained_payload_bytes().unwrap();
        assert!(retained + external <= maximum);
        inventory
            .retain_algorithm_input(&wider, retained + external, external)
            .unwrap();
        assert_eq!(inventory.algorithm_inputs.len(), 1);
        assert_eq!(inventory.retained_payload_bytes(), Some(retained));
        assert!(inventory.opportunities.is_empty());
        let other = fixture::input(1, 3, CostProductOutput::FullLogits, false, true);
        let final_only = retained
            + external
            + other.retained_payload_bytes().unwrap()
            + 2 * std::mem::size_of::<usize>()
            + std::mem::size_of::<Arc<StructuredInputV2>>();
        assert!(
            inventory
                .retain_algorithm_input(&other, final_only, external)
                .is_err(),
            "reallocation must reserve the old and replacement backing simultaneously"
        );
        assert_eq!(inventory.algorithm_inputs.len(), 1);
        assert_eq!(inventory.retained_payload_bytes(), Some(retained));
    }

    #[test]
    fn algorithm_pool_rejects_unbound_input_and_excludes_exact_only_scope() {
        let mut inventory = empty();
        let unbound = fixture::input(1, 3, CostProductOutput::GreedyToken, false, false);
        assert!(inventory
            .retain_algorithm_input(&unbound, usize::MAX, 0)
            .is_err());
        let heterogeneous = fixture::input(2, 3, CostProductOutput::GreedyToken, true, true);
        inventory
            .retain_algorithm_input(&heterogeneous, usize::MAX, 0)
            .unwrap();
        assert!(inventory.algorithm_inputs.is_empty());
        assert_eq!(inventory.algorithm_inputs.capacity(), 0);
    }
}

fn same_prefix(left: PrefixKind, right: PrefixKind) -> bool {
    match (left, right) {
        (PrefixKind::Ordinary, PrefixKind::Ordinary)
        | (PrefixKind::Clean, PrefixKind::Clean)
        | (PrefixKind::Pending, PrefixKind::Pending) => true,
        (PrefixKind::Mixed { pending_rows: a }, PrefixKind::Mixed { pending_rows: b }) => a == b,
        _ => false,
    }
}

pub(super) fn same_group(left: &Case, right: &Case) -> bool {
    left.template == right.template
        && left.explicit_prefill_chunk() == right.explicit_prefill_chunk()
        && left.preset == right.preset
        && left.maximum_output == right.maximum_output
        && left.reset == right.reset
        && matches!(left.route, CalibrationDecodeRoute::Actual)
        && matches!(right.route, CalibrationDecodeRoute::Actual)
}

fn same_scenario(left: &Case, right: &Case) -> bool {
    let same_prefill_span = match (left.product, right.product) {
        (
            OpportunityProduct::PrefillSpan {
                offset: a,
                chunk: ac,
            },
            OpportunityProduct::PrefillSpan {
                offset: b,
                chunk: bc,
            },
        ) => a == b && ac == bc,
        (OpportunityProduct::PrefillSpan { .. }, _)
        | (_, OpportunityProduct::PrefillSpan { .. }) => false,
        (
            OpportunityProduct::ContinuationPrefill { offset: a },
            OpportunityProduct::ContinuationPrefill { offset: b },
        ) => a == b,
        (OpportunityProduct::ContinuationPrefill { .. }, _)
        | (_, OpportunityProduct::ContinuationPrefill { .. }) => false,
        _ => true,
    };
    same_prefill_span
        && left.width == right.width
        && left.release_generated == right.release_generated
        && same_prefix(left.prefix, right.prefix)
}

/// Original requests and real driver options only. This does not install a
/// prefix, run a wave, or promise that a resident graph will be produced.
#[cfg(test)]
pub(super) fn readiness_requests(
    case: &Case,
    templates: &[AutomaticCostProbeTemplate],
    whole_chunk: NonZeroU32,
) -> Result<(Vec<ProbeRequest>, ProbeCohortSettings)> {
    readiness_requests_with_row_ceiling(case, templates, whole_chunk, None)
}

pub(super) fn readiness_requests_with_row_ceiling(
    case: &Case,
    templates: &[AutomaticCostProbeTemplate],
    whole_chunk: NonZeroU32,
    prefill_row_ceiling: Option<NonZeroU32>,
) -> Result<(Vec<ProbeRequest>, ProbeCohortSettings)> {
    let template = templates
        .get(case.template)
        .ok_or_else(|| error("readiness template index differs"))?;
    let chunk = case.prefill_chunk(whole_chunk, prefill_row_ceiling, case.width)?;
    let mut requests = Vec::with_capacity(case.width);
    for slot in 0..case.width {
        let (request, contract) =
            template.instantiate(case.maximum_output, slot as u64, case.preset)?;
        requests.push(ProbeRequest { request, contract });
    }
    Ok((
        requests,
        ProbeCohortSettings {
            prefill_plan:
                crate::continuous_engine::inner::calibration::cohort_driver::ProbePrefillPlan::Joint,
            prefill_chunk: chunk,
            decode_route: case.route,
            reset_token_policy: case.reset,
        },
    ))
}

fn target(case: &Case, prompts: &[usize]) -> Result<GeometryInputTarget> {
    if matches!(case.prefix, PrefixKind::Ordinary) {
        return match case.product {
            OpportunityProduct::Prefill => {
                Ok(GeometryInputTarget::InitialPrefill { rows: case.width })
            }
            OpportunityProduct::PrefillSpan { offset: 0, .. } => {
                Ok(GeometryInputTarget::InitialPrefill { rows: case.width })
            }
            OpportunityProduct::ContinuationPrefill { offset }
            | OpportunityProduct::PrefillSpan { offset, .. } => {
                Ok(GeometryInputTarget::PrefillSpan {
                    rows: case.width,
                    offset: u32::try_from(offset)
                        .map_err(|_| error("inventory prefill span exceeds physical context"))?,
                })
            }
            _ => Err(error(
                "ordinary inventory case has no guaranteed decode frontier",
            )),
        };
    }
    let sequence_tokens = prompts
        .get(case.template)
        .and_then(|n| n.checked_add(case.release_generated))
        .and_then(|n| u32::try_from(n).ok())
        .ok_or_else(|| error("inventory prefix frontier exceeds physical context domain"))?;
    Ok(GeometryInputTarget::Decode(GeometryProjectionPoint {
        rows: case.width,
        sequence_tokens,
    }))
}

fn slot_bytes(slot: &StructuredPrefixSlotV5) -> Option<usize> {
    let mut bytes = slot
        .token_ids
        .len()
        .checked_mul(std::mem::size_of::<ferrum_types::TokenId>())?
        .checked_add(
            slot.token_bytes
                .len()
                .checked_mul(std::mem::size_of::<Vec<u8>>())?,
        )?;
    for fragment in &slot.token_bytes {
        bytes = bytes.checked_add(fragment.len())?;
    }
    Some(bytes)
}

fn require_capacity(bytes: Option<usize>, maximum: usize) -> Result<usize> {
    bytes
        .filter(|n| *n <= maximum)
        .ok_or_else(|| error("checked input inventory retained capacity exhausted"))
}

/// Only declared request work and provider context partitions choose these
/// hypotheses. None is a numerical sample or a conditional fresh-member floor.
fn trajectory_targets(
    case: &Case,
    prompts: &[usize],
    boundaries: &[u32],
) -> Result<Vec<GeometryInputTarget>> {
    if case.maximum_output.get() <= 1 {
        return Ok(Vec::new());
    }
    let prompt = *prompts
        .get(case.template)
        .ok_or_else(|| error("algorithm trajectory template differs"))?;
    let first = prompt
        .checked_add(1)
        .and_then(|n| u32::try_from(n).ok())
        .ok_or_else(|| error("algorithm trajectory context overflow"))?;
    let last = prompt
        .checked_add(case.maximum_output.get() - 1)
        .and_then(|n| u32::try_from(n).ok())
        .ok_or_else(|| error("algorithm trajectory context overflow"))?;
    let mut targets = Vec::with_capacity(
        boundaries
            .len()
            .checked_add(2)
            .ok_or_else(|| error("algorithm trajectory capacity overflow"))?,
    );
    for sequence_tokens in [first, last].into_iter().chain(boundaries.iter().copied()) {
        if sequence_tokens < first || sequence_tokens > last {
            continue;
        }
        let target = GeometryInputTarget::Decode(GeometryProjectionPoint {
            rows: case.width,
            sequence_tokens,
        });
        if !targets.contains(&target) {
            targets.push(target);
        }
    }
    // The targets are declaration-only geometry. Visit them in frontier order
    // so one checked trajectory can supply every original endpoint/boundary.
    targets.sort_unstable_by_key(|target| match target {
        GeometryInputTarget::Decode(point) => point.sequence_tokens,
        GeometryInputTarget::InitialPrefill { .. } | GeometryInputTarget::PrefillSpan { .. } => 0,
    });
    Ok(targets)
}

#[allow(clippy::too_many_arguments)]
async fn collect_inner(
    session: &mut CalibrationSession,
    cases: &[Case],
    templates: &[AutomaticCostProbeTemplate],
    prompts: &[usize],
    pair: &PrefixPair,
    policy: StructuredPopulationPolicyV1,
    decode_boundaries: &[u32],
    limits: InventoryLimits,
    readiness_cursor: Option<ReadinessCursor>,
    charge: &mut ProbePreflightCharge,
    can_act: Option<&(dyn Fn(usize, &GeometryProjectionUnknown) -> bool + Sync)>,
) -> Result<ReadinessCapture> {
    if cases.is_empty() || templates.len() != prompts.len() {
        return Err(error(
            "checked inventory requires original case/template alignment",
        ));
    }
    let algorithm_target_capacity = decode_boundaries
        .len()
        .checked_add(2)
        .ok_or_else(|| error("algorithm trajectory capacity overflow"))?;
    let algorithm_scratch = cases
        .len()
        .checked_mul(std::mem::size_of::<usize>() + std::mem::size_of::<Vec<GeometryInputTarget>>())
        .and_then(|n| {
            n.checked_add(
                cases
                    .len()
                    .checked_mul(algorithm_target_capacity)?
                    .checked_mul(std::mem::size_of::<GeometryInputTarget>())?,
            )
        })
        .ok_or_else(|| error("algorithm trajectory capacity overflow"))?;
    let base = std::mem::size_of::<CheckedCaseInventory>()
        .checked_add(algorithm_scratch)
        .and_then(|n| {
            n.checked_add(cases.len().checked_mul(
                std::mem::size_of::<CaseOpportunity>()
                    + std::mem::size_of::<Vec<CheckedInputFacts>>()
                    + std::mem::size_of::<Vec<usize>>()
                    + std::mem::size_of::<InventoryGap>()
                    + std::mem::size_of::<bool>()
                    + 2 * std::mem::size_of::<usize>()
                    + std::mem::size_of::<GeometryInputTarget>(),
            )?)
        });
    require_capacity(base, limits.maximum_retained_bytes)?;
    let mut inventory = CheckedCaseInventory {
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
        algorithm_case_inputs: (0..cases.len()).map(|_| Vec::new()).collect(),
        gaps: Vec::with_capacity(cases.len()),
        charge: ProbePreflightCharge::default(),
    };
    let mut done = vec![false; cases.len()];
    let mut members = Vec::with_capacity(cases.len());
    let mut targets = Vec::with_capacity(cases.len());
    let mut scenario_cases = Vec::with_capacity(cases.len());
    let mut algorithm_cases: Vec<usize> = Vec::with_capacity(cases.len());
    let mut algorithm_targets: Vec<Vec<GeometryInputTarget>> = Vec::with_capacity(cases.len());
    let scratch_arrays = done
        .capacity()
        .checked_mul(std::mem::size_of::<bool>())
        .and_then(|n| {
            n.checked_add(
                members
                    .capacity()
                    .checked_mul(std::mem::size_of::<usize>())?,
            )
        })
        .and_then(|n| {
            n.checked_add(
                targets
                    .capacity()
                    .checked_mul(std::mem::size_of::<GeometryInputTarget>())?,
            )
        })
        .and_then(|n| {
            n.checked_add(
                scenario_cases
                    .capacity()
                    .checked_mul(std::mem::size_of::<usize>())?,
            )
        })
        .and_then(|n| n.checked_add(algorithm_scratch))
        .ok_or_else(|| error("inventory scratch capacity overflow"))?;
    for index in 0..cases.len() {
        if done[index] {
            continue;
        }
        if Instant::now() >= limits.deadline {
            return Err(error("checked inventory deadline expired"));
        }
        let case = &cases[index];
        if !matches!(case.route, CalibrationDecodeRoute::Actual) {
            return Err(error(
                "checked inventory cannot replace an explicitly selected decode route",
            ));
        }
        members.clear();
        targets.clear();
        scenario_cases.clear();
        algorithm_cases.clear();
        algorithm_targets.clear();
        require_capacity(
            inventory
                .retained_payload_bytes()
                .and_then(|n| n.checked_add(scratch_arrays)),
            limits.maximum_retained_bytes,
        )?;
        let mut width = 0;
        for (other, candidate) in cases.iter().enumerate() {
            if !done[other] && same_group(case, candidate) {
                members.push(other);
                width = width.max(candidate.width);
                let point = target(candidate, prompts)?;
                if !scenario_cases
                    .iter()
                    .any(|&i| same_scenario(&cases[i], candidate))
                {
                    targets.push(point);
                    scenario_cases.push(other);
                }
                if !algorithm_cases
                    .iter()
                    .any(|&i| cases[i].width == candidate.width)
                    && candidate.maximum_output.get() > 1
                {
                    algorithm_cases.push(other);
                    algorithm_targets.push(trajectory_targets(
                        candidate,
                        prompts,
                        decode_boundaries,
                    )?);
                }
            }
        }
        // Reserve the entire maximum-width owner group before creating any
        // real owner. A partial failure terminates this plan; it is not retried.
        if width == 0
            || width
                > limits
                    .maximum_admitted_requests
                    .saturating_sub(inventory.charge.admitted_requests)
        {
            return Err(error("checked inventory original request budget exhausted"));
        }
        // An explicit candidate must fit the same real whole-wave and row
        // limits as the later source driver, before any numeric projection.
        case.prefill_chunk(limits.prefill_chunk, limits.prefill_row_ceiling, width)?;
        let projection_row_ceiling = case.explicit_prefill_chunk().or(limits.prefill_row_ceiling);
        let projection_remaining = limits
            .maximum_projection_attempts
            .checked_sub(inventory.charge.projection_attempts)
            .filter(|n| *n > 0)
            .ok_or_else(|| error("checked inventory pure projection budget exhausted"))?;
        let mut group_bytes = width
            .checked_mul(std::mem::size_of::<ProbeRequest>())
            .and_then(|n| {
                n.checked_add(scenario_cases.len().checked_mul(
                    std::mem::size_of::<Vec<GeometryPrefixConstraint>>()
                        + std::mem::size_of::<GeometryInputScenario<'_>>(),
                )?)
            })
            .and_then(|n| {
                n.checked_add(
                    algorithm_targets
                        .len()
                        .checked_mul(std::mem::size_of::<GeometryInputScenario<'_>>())?,
                )
            })
            .ok_or_else(|| error("inventory group capacity overflow"))?;
        for &scenario in &scenario_cases {
            if matches!(cases[scenario].prefix, PrefixKind::Ordinary) {
                continue;
            }
            group_bytes = group_bytes
                .checked_add(
                    width
                        .checked_mul(std::mem::size_of::<GeometryPrefixConstraint>())
                        .ok_or_else(|| error("inventory prefix capacity overflow"))?,
                )
                .ok_or_else(|| error("inventory prefix capacity overflow"))?;
            for slot in 0..width {
                group_bytes = group_bytes
                    .checked_add(
                        slot_bytes(prefix_slot(cases[scenario].prefix, slot, pair))
                            .ok_or_else(|| error("inventory prefix capacity overflow"))?,
                    )
                    .ok_or_else(|| error("inventory prefix capacity overflow"))?;
            }
        }
        let retained = require_capacity(
            inventory
                .retained_payload_bytes()
                .and_then(|n| n.checked_add(scratch_arrays))
                .and_then(|n| n.checked_add(group_bytes)),
            limits.maximum_retained_bytes,
        )?;
        // Queries coexist briefly with their compact derived facts. Give both
        // disjoint portions of the one remaining cold payload allowance.
        let report_limit = (limits.maximum_retained_bytes - retained) / 2;
        if report_limit == 0 {
            return Err(error("checked inventory has no query capacity"));
        }
        if case.reset {
            let invalidation = tokio::time::timeout_at(
                limits.deadline,
                session.invalidate_token_policy_residency(),
            )
            .await
            .map_err(|_| error("checked inventory residency deadline expired"))?;
            if !matches!(
                invalidation,
                TokenPolicyResidencyInvalidation::Cleared { .. }
            ) {
                return Err(error(format!(
                    "checked inventory cold residency boundary unavailable: {invalidation:?}"
                )));
            }
        }
        let template = templates
            .get(case.template)
            .ok_or_else(|| error("inventory template index differs"))?;
        let mut requests = Vec::with_capacity(width);
        for slot in 0..width {
            let (request, contract) =
                template.instantiate(case.maximum_output, slot as u64, case.preset)?;
            requests.push(ProbeRequest { request, contract });
        }
        let mut prefix_sets = Vec::with_capacity(scenario_cases.len());
        for &scenario in &scenario_cases {
            let original = &cases[scenario];
            let prefix = !matches!(original.prefix, PrefixKind::Ordinary);
            let mut prefixes = Vec::with_capacity(if prefix { width } else { 0 });
            if prefix {
                for (slot, request) in requests.iter().enumerate() {
                    prefixes.push(GeometryPrefixConstraint {
                        request_id: request.request.id.clone(),
                        slot: prefix_slot(original.prefix, slot, pair).clone(),
                        release_generated: u32::try_from(original.release_generated)
                            .map_err(|_| error("inventory release exceeds physical domain"))?,
                    });
                }
            }
            prefix_sets.push(prefixes);
        }
        let mut scenarios = Vec::with_capacity(targets.len() + algorithm_targets.len());
        for (target, prefixes) in targets.iter().zip(&prefix_sets) {
            scenarios.push(GeometryInputScenario {
                targets: std::slice::from_ref(target),
                prefixes,
            });
        }
        for targets in &algorithm_targets {
            // A future ordinary trajectory admits every checked host-content
            // branch. It does not install a prefix or confer a fresh member.
            scenarios.push(GeometryInputScenario {
                targets,
                prefixes: &[],
            });
        }
        let expected_outcomes = algorithm_targets
            .iter()
            .try_fold(targets.len(), |n, targets| n.checked_add(targets.len()))
            .ok_or_else(|| error("algorithm outcome capacity overflow"))?;
        let report = if let Some(cursor) = readiness_cursor {
            let first_target = cursor.first_target;
            let can_act = can_act.ok_or_else(|| error("readiness action predicate absent"))?;
            let stop =
                |scenario: usize, _: GeometryInputTarget, reason: &GeometryProjectionUnknown| {
                    let case_index = if scenario < targets.len() {
                        scenario_cases.get(scenario)
                    } else {
                        algorithm_cases.get(scenario - targets.len())
                    };
                    case_index.is_some_and(|&index| can_act(index, reason))
                };
            let mut attempted = GeometryProjectionCharge::default();
            let report = session
                .project_geometry_readiness_scenarios(
                    requests,
                    &scenarios,
                    GeometryProjectionLimits {
                        deadline: limits.deadline,
                        maximum_projections: projection_remaining,
                        maximum_route_states: limits.maximum_route_states,
                        maximum_retained_bytes: report_limit,
                        prefill_chunk: limits.prefill_chunk,
                        prefill_row_ceiling: projection_row_ceiling,
                    },
                    first_target,
                    cursor.retain_complete,
                    &stop,
                    &mut attempted,
                )
                .await;
            readiness::record_charge(
                &mut inventory.charge,
                charge,
                attempted.admitted_requests,
                attempted.projection_attempts,
                &limits,
            )?;
            match report? {
                GeometryReadinessReport::Complete(report) => {
                    if report.admitted_requests != attempted.admitted_requests
                        || report.projection_attempts != attempted.projection_attempts
                    {
                        return Err(error("readiness charge differs from its completed receipt"));
                    }
                    report
                }
                GeometryReadinessReport::Progress {
                    visited_targets,
                    gap,
                    admitted_requests,
                    projection_attempts,
                } => {
                    if admitted_requests != attempted.admitted_requests
                        || projection_attempts != attempted.projection_attempts
                    {
                        return Err(error("readiness charge differs from its completed receipt"));
                    }
                    let next_target = first_target
                        .checked_add(visited_targets)
                        .filter(|next| *next > first_target && *next <= expected_outcomes)
                        .ok_or_else(|| error("readiness did not visit an original target"))?;
                    let gap = gap
                        .map(|(scenario, target, reason)| -> Result<InventoryGap> {
                            tracing::info!(
                                first_target,
                                next_target,
                                total_targets = expected_outcomes,
                                scenario,
                                ?target,
                                ?reason,
                                projection_attempts,
                                projection_remaining,
                                "Automatic readiness projection stopped at original gap"
                            );
                            let case_index = if scenario < targets.len() {
                                scenario_cases.get(scenario)
                            } else {
                                algorithm_cases.get(scenario - targets.len())
                            }
                            .copied()
                            .ok_or_else(|| error("readiness gap has no original case"))?;
                            readiness::require_nonfatal(
                                &reason,
                                "readiness",
                                scenario,
                                target,
                                projection_attempts,
                                projection_remaining,
                            )?;
                            Ok(InventoryGap {
                                case_index,
                                reason: InventoryGapReason::Projection(reason),
                            })
                        })
                        .transpose()?;
                    return Ok(ReadinessCapture::Progress(ReadinessProgress {
                        next_target,
                        total_targets: expected_outcomes,
                        gap,
                    }));
                }
            }
        } else {
            let mut attempted = GeometryProjectionCharge::default();
            let report = session
                .project_geometry_input_scenarios_charged(
                    requests,
                    &scenarios,
                    GeometryProjectionLimits {
                        deadline: limits.deadline,
                        maximum_projections: projection_remaining,
                        maximum_route_states: limits.maximum_route_states,
                        maximum_retained_bytes: report_limit,
                        prefill_chunk: limits.prefill_chunk,
                        prefill_row_ceiling: projection_row_ceiling,
                    },
                    &mut attempted,
                )
                .await;
            readiness::record_charge(
                &mut inventory.charge,
                charge,
                attempted.admitted_requests,
                attempted.projection_attempts,
                &limits,
            )?;
            report?
        };
        if report.admitted_requests != width || report.outcomes.len() != expected_outcomes {
            return Err(error(format!(
                "checked inventory incomplete real owner group: admitted={}/{width}, outcomes={}/{}, first_unknown={:?}",
                report.admitted_requests,
                report.outcomes.len(),
                expected_outcomes,
                report.outcomes.iter().find_map(|outcome| outcome.unknown.as_ref()),
            )));
        }
        for outcome in report
            .outcomes
            .iter()
            .filter(|outcome| outcome.scenario_index >= targets.len())
        {
            if let Some(reason) = &outcome.unknown {
                readiness::require_nonfatal(
                    reason,
                    "algorithm trajectory",
                    outcome.scenario_index,
                    outcome.target,
                    report.projection_attempts,
                    projection_remaining,
                )?;
            }
            let external = scratch_arrays
                .checked_add(group_bytes)
                .and_then(|n| n.checked_add(report_limit))
                .ok_or_else(|| error("algorithm trajectory retained capacity overflow"))?;
            for branch in &outcome.branches {
                let Some(input) = inventory.retain_algorithm_input(
                    branch.query.input(),
                    limits.maximum_retained_bytes,
                    external,
                )?
                else {
                    continue;
                };
                let trajectory = *algorithm_cases
                    .get(outcome.scenario_index - targets.len())
                    .ok_or_else(|| error("algorithm trajectory lost its original case"))?;
                for &member in &members {
                    if cases[member].width == cases[trajectory].width {
                        inventory.link_algorithm_input(
                            member,
                            input,
                            limits.maximum_retained_bytes,
                            external,
                        )?;
                    }
                }
            }
        }
        for &member in &members {
            let original_target = target(&cases[member], prompts)?;
            let scenario_index = scenario_cases
                .iter()
                .position(|&i| same_scenario(&cases[i], &cases[member]))
                .ok_or_else(|| error("inventory lost its original scenario"))?;
            let outcome = report
                .outcomes
                .iter()
                .find(|o| o.scenario_index == scenario_index && o.target == original_target)
                .ok_or_else(|| error("inventory lost its original target"))?;
            if let Some(reason) = &outcome.unknown {
                readiness::require_nonfatal(
                    reason,
                    "checked inventory",
                    outcome.scenario_index,
                    outcome.target,
                    report.projection_attempts,
                    projection_remaining,
                )?;
            }
            let incoming = outcome
                .branches
                .len()
                .checked_mul(
                    std::mem::size_of::<CheckedInputFacts>()
                        + std::mem::size_of::<CheckedPopulationKey>(),
                )
                .and_then(|initial| {
                    outcome.branches.iter().try_fold(initial, |n, branch| {
                        n.checked_add(
                            branch
                                .query
                                .input()
                                .regression_axes()
                                .len()
                                .checked_mul(std::mem::size_of::<f64>())?,
                        )?
                        .checked_add(branch.query.input().retained_payload_bytes()?)?
                        .checked_add(2 * std::mem::size_of::<usize>())
                    })
                })
                .and_then(|n| n.checked_add(std::mem::size_of::<CheckedPopulationKey>()));
            require_capacity(
                inventory
                    .retained_payload_bytes()
                    .and_then(|n| n.checked_add(scratch_arrays))
                    .and_then(|n| n.checked_add(group_bytes))
                    .and_then(|n| n.checked_add(report_limit))
                    .and_then(|n| n.checked_add(incoming?)),
                limits.maximum_retained_bytes,
            )?;
            let mut keys = Vec::with_capacity(outcome.branches.len());
            let mut facts = Vec::with_capacity(outcome.branches.len());
            for branch in &outcome.branches {
                let identity = populations::classify_alternatives(
                    std::slice::from_ref(branch.query.input()),
                    policy,
                    true,
                )
                .map_err(|reason| error(format!("checked inventory population: {reason:?}")))?;
                let CasePopulation::Unique(key) = identity else {
                    return Err(error("checked single input has no population"));
                };
                if !keys.contains(&key) {
                    keys.push(key);
                }
                facts.push(
                    input_facts(&branch.query)
                        .map_err(|reason| error(format!("checked inventory facts: {reason:?}")))?,
                );
            }
            let trajectory_unknown = algorithm_cases
                .iter()
                .position(|&i| cases[i].width == cases[member].width)
                .and_then(|scenario| {
                    report
                        .outcomes
                        .iter()
                        .filter(|o| o.scenario_index == targets.len() + scenario)
                        .find_map(|o| o.unknown.as_ref())
                });
            let complete =
                outcome.unknown.is_none() && trajectory_unknown.is_none() && !keys.is_empty();
            let unique = complete && keys.len() == 1;
            // Shape validity and actual source membership are separate gates.
            // The actual original selector excludes a configured non-reusable
            // wave from warm-or-disabled populations even when its eager
            // projection is known. Keep its raw facts/algorithms, but never
            // mark that cohort as a guaranteed numerical member.
            let outside_route = (!limits.route_population.is_all_attempts())
                .then(|| {
                    outcome.branches.iter().find_map(|branch| {
                        PreparedCostRouteClassV1::from_eligible_graph(branch.graph)
                            .is_none()
                            .then_some(branch.graph)
                    })
                })
                .flatten();
            let reachable = match outcome.target {
                GeometryInputTarget::InitialPrefill { .. } => cases[member].reset,
                GeometryInputTarget::PrefillSpan { rows, offset } => {
                    let case = &cases[member];
                    let chunk = case
                        .prefill_chunk(limits.prefill_chunk, limits.prefill_row_ceiling, rows)
                        .ok()
                        .and_then(|n| usize::try_from(n.get()).ok());
                    let exact_offset = match case.product {
                        OpportunityProduct::ContinuationPrefill { offset: declared }
                        | OpportunityProduct::PrefillSpan {
                            offset: declared, ..
                        } => declared == offset as usize,
                        _ => false,
                    };
                    case.reset
                        && matches!(case.prefix, PrefixKind::Ordinary)
                        && case.width == rows
                        && exact_offset
                        && offset != 0
                        && chunk.is_some_and(|chunk| offset as usize % chunk == 0)
                        && prompts
                            .get(case.template)
                            .is_some_and(|prompt| (offset as usize) < *prompt)
                }
                GeometryInputTarget::Decode(_) => {
                    outcome.prefix_condition.is_some_and(|condition| {
                        condition.first_ordinary_wave
                            && condition.release_generated as usize
                                == cases[member].release_generated
                    })
                }
            };
            let reason = if let Some(reason) = outcome.unknown.as_ref().or(trajectory_unknown) {
                Some(InventoryGapReason::Projection(reason.clone()))
            } else if !unique {
                Some(InventoryGapReason::MultiplePopulations)
            } else if let Some(projected_graph) = outside_route {
                Some(InventoryGapReason::OutsideDeclaredRoute {
                    population: limits.route_population,
                    projected_graph,
                })
            } else if !reachable {
                Some(if !matches!(cases[member].prefix, PrefixKind::Ordinary) {
                    InventoryGapReason::PrefixReleaseUnproven
                } else {
                    InventoryGapReason::WarmResidencyUnproven
                })
            } else {
                None
            };
            let population = if !complete {
                CasePopulation::Unknown {
                    known_alternatives: keys,
                }
            } else if unique {
                CasePopulation::Unique(keys.pop().unwrap())
            } else {
                CasePopulation::Alternatives(keys)
            };
            inventory.opportunities[member] = CaseOpportunity {
                population,
                minimum_fresh_members: usize::from(unique && reachable && outside_route.is_none()),
            };
            inventory.inputs[member] = facts;
            if let Some(reason) = reason {
                inventory.gaps.push(InventoryGap {
                    case_index: member,
                    reason,
                });
            }
            done[member] = true;
        }
    }
    require_capacity(
        inventory.retained_payload_bytes(),
        limits.maximum_retained_bytes,
    )?;
    Ok(ReadinessCapture::Complete(inventory))
}

#[cfg(test)]
mod tests;
