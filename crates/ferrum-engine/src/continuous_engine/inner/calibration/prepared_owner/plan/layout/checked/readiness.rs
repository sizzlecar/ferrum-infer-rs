//! Merge ordinary readiness execution across original inventory groups. An
//! attempted route is not evidence that any later input can be projected.
use super::*;

pub(super) fn needs_inventory(case: &Case) -> bool {
    !matches!(case.prefix, PrefixKind::Ordinary) || case.reset
}

/// Initial captures only: each group admits its greatest retained width.
/// Actual readiness and any subsequent fresh recapture are charged separately.
pub(super) fn initial_inventory_admissions(cases: &[Case]) -> Result<usize> {
    let mut admissions = 0usize;
    for (index, case) in cases.iter().enumerate() {
        if !needs_inventory(case)
            || cases[..index]
                .iter()
                .any(|previous| needs_inventory(previous) && inventory::same_group(case, previous))
        {
            continue;
        }
        let width = cases[index..]
            .iter()
            .filter(|candidate| {
                needs_inventory(candidate) && inventory::same_group(case, candidate)
            })
            .map(|candidate| candidate.width)
            .max()
            .ok_or_else(|| error("readiness inventory group absent"))?;
        admissions = admissions
            .checked_add(width)
            .ok_or_else(|| error("readiness inventory admission count overflow"))?;
    }
    Ok(admissions)
}

pub(super) fn longest(case: &Case, cases: &[Case], output_limits: &[NonZeroUsize]) -> Result<Case> {
    let limit = output_limits
        .get(case.template)
        .ok_or_else(|| error("readiness template output limit absent"))?;
    let maximum = cases
        .iter()
        .filter(|candidate| {
            candidate.template == case.template
                && candidate.preset == case.preset
                && candidate.width == case.width
                && candidate.route == CalibrationDecodeRoute::Actual
        })
        .map(|candidate| candidate.maximum_output)
        .max()
        .filter(|maximum| maximum <= limit)
        .ok_or_else(|| error("readiness exceeds original template output limit"))?;
    // output_limits already intersect the configured output allowance with
    // this template's remaining context. Use only a length demanded by an
    // existing case; do not spend the full allowance merely because it exists.
    let mut readiness = case.clone();
    readiness.maximum_output = maximum;
    readiness.product = case
        .explicit_prefill_chunk()
        .map_or(OpportunityProduct::Prefill, |chunk| {
            OpportunityProduct::PrefillSpan { offset: 0, chunk }
        });
    readiness.prefix = PrefixKind::Ordinary;
    readiness.acquisition = None;
    readiness.release_generated = 0;
    readiness.suffix_tokens = maximum.get();
    readiness.route = CalibrationDecodeRoute::Actual;
    readiness.reset = false;
    Ok(readiness)
}

/// Select only from the current capture, preserving the original case/route
/// order and bounded attempt identities. Claiming an action never closes a gap.
pub(super) enum Action {
    Resources(Case),
    Execute(Case),
}

/// Pure eligibility for the short projection cursor. An already attempted
/// action does not close its gap; only the final full inventory can do so.
pub(super) fn can_act(
    reason: &GeometryProjectionUnknown,
    case: &Case,
    cases: &[Case],
    input: &PreparedProbeInputs,
    attempts: &Attempts,
) -> bool {
    let Ok(readiness) = longest(case, cases, &input.outputs) else {
        // Let next_action return the original validation error.
        return true;
    };
    match reason {
        GeometryProjectionUnknown::Route(ExecutionCostRouteUnknown::Resource(
            ResourcePlanningUnknown::UnmaterializedCapacity,
        )) => attempts.can_claim_kind(&readiness, CalibrationDecodeRoute::Actual, true),
        GeometryProjectionUnknown::Route(ExecutionCostRouteUnknown::OnDemandResidentProgram) => {
            attempts.can_claim_kind(&readiness, CalibrationDecodeRoute::Actual, false)
                || (readiness.maximum_output.get() > 1
                    && templates::declared_greedy_sampling(
                        &input.templates[readiness.template],
                        readiness.preset,
                    )
                    .unwrap_or(true)
                    && attempts.can_claim_kind(
                        &readiness,
                        CalibrationDecodeRoute::FullLogits,
                        false,
                    ))
        }
        _ => false,
    }
}

pub(super) fn next_action(
    gaps: &[inventory::InventoryGap],
    group: &[Case],
    cases: &[Case],
    input: &PreparedProbeInputs,
    attempts: &mut Attempts,
) -> Result<Option<Action>> {
    for gap in gaps {
        if !readiness_missing(&gap.reason) {
            continue;
        }
        let case = group
            .get(gap.case_index)
            .ok_or_else(|| error("readiness gap case absent"))?;
        let mut readiness = longest(case, cases, &input.outputs)?;
        if matches!(
            gap.reason,
            InventoryGapReason::Projection(GeometryProjectionUnknown::Route(
                ExecutionCostRouteUnknown::Resource(
                    ResourcePlanningUnknown::UnmaterializedCapacity
                )
            ))
        ) {
            if attempts.claim_resource(&readiness)? {
                return Ok(Some(Action::Resources(readiness)));
            }
            continue;
        }
        let routes: &[_] = if readiness.maximum_output.get() > 1
            && templates::declared_greedy_sampling(
                &input.templates[readiness.template],
                readiness.preset,
            )? {
            &[
                CalibrationDecodeRoute::Actual,
                CalibrationDecodeRoute::FullLogits,
            ]
        } else {
            &[CalibrationDecodeRoute::Actual]
        };
        for &route in routes {
            if attempts.claim(&readiness, route)? {
                readiness.route = route;
                return Ok(Some(Action::Execute(readiness)));
            }
        }
    }
    Ok(None)
}

struct Attempt {
    resource_only: bool,
    template: usize,
    preset: SloAutomaticCostProbeSamplingPresetV1,
    width: usize,
    route: CalibrationDecodeRoute,
    maximum_output: NonZeroUsize,
    prefill_chunk: Option<NonZeroU32>,
}

pub(super) struct Attempts {
    entries: Vec<Attempt>,
    maximum_entries: usize,
}

impl Attempts {
    pub(super) fn new(maximum_entries: usize, maximum_bytes: usize) -> Result<Self> {
        let bytes = |capacity: usize| {
            capacity
                .checked_mul(std::mem::size_of::<Attempt>())
                .and_then(|n| n.checked_add(std::mem::size_of::<Self>()))
                .filter(|n| *n <= maximum_bytes)
        };
        bytes(maximum_entries).ok_or_else(|| error("readiness retained capacity exhausted"))?;
        let mut entries = Vec::new();
        entries
            .try_reserve_exact(maximum_entries)
            .map_err(|_| error("readiness allocation failed"))?;
        bytes(entries.capacity()).ok_or_else(|| error("readiness retained capacity exhausted"))?;
        Ok(Self {
            entries,
            maximum_entries,
        })
    }

    pub(super) fn retained_payload_bytes(&self) -> Result<usize> {
        self.entries
            .capacity()
            .checked_mul(std::mem::size_of::<Attempt>())
            .and_then(|n| n.checked_add(std::mem::size_of::<Self>()))
            .ok_or_else(|| error("readiness retained capacity overflow"))
    }

    /// Records execution intent only. EOS, absent programs and later eviction
    /// all remain Unknown when each original group captures fresh live roots.
    pub(super) fn claim(&mut self, case: &Case, route: CalibrationDecodeRoute) -> Result<bool> {
        self.claim_kind(case, route, false)
    }

    fn claim_resource(&mut self, case: &Case) -> Result<bool> {
        self.claim_kind(case, CalibrationDecodeRoute::Actual, true)
    }

    fn claim_kind(
        &mut self,
        case: &Case,
        route: CalibrationDecodeRoute,
        resource_only: bool,
    ) -> Result<bool> {
        if let Some(previous) = self.entries.iter_mut().find(|previous| {
            previous.resource_only == resource_only
                && previous.template == case.template
                && previous.preset == case.preset
                && previous.width == case.width
                && previous.route == route
                && previous.prefill_chunk == case.explicit_prefill_chunk()
        }) {
            if previous.maximum_output >= case.maximum_output {
                return Ok(false);
            }
            previous.maximum_output = case.maximum_output;
            return Ok(true);
        }
        if self.entries.len() >= self.maximum_entries {
            return Err(error("readiness route capacity exhausted"));
        }
        self.entries.push(Attempt {
            resource_only,
            template: case.template,
            preset: case.preset,
            width: case.width,
            route,
            maximum_output: case.maximum_output,
            prefill_chunk: case.explicit_prefill_chunk(),
        });
        Ok(true)
    }

    fn can_claim_kind(
        &self,
        case: &Case,
        route: CalibrationDecodeRoute,
        resource_only: bool,
    ) -> bool {
        self.entries
            .iter()
            .find(|previous| {
                previous.resource_only == resource_only
                    && previous.template == case.template
                    && previous.preset == case.preset
                    && previous.width == case.width
                    && previous.route == route
                    && previous.prefill_chunk == case.explicit_prefill_chunk()
            })
            .is_none_or(|previous| previous.maximum_output < case.maximum_output)
    }
}

#[cfg(test)]
mod tests;
