//! Frozen finite opportunities, conditional on each original guaranteed cohort
//! completing and each independent numerical phase succeeding. This is neither
//! a measured member count nor permission to restart an owner without charging.
use super::*;
use populations::{CasePopulation, CheckedPopulationKey};
use serde::Serialize;
use std::{mem::size_of, ops::Range};

// The existing CohortPlanV2 wire admits at most 4096 cohorts per driver pass,
// 4096 requests per cohort and 65536 total request slots. Driver passes do not
// assign numerical phases; the cuts below do that independently for each owner.
const MAX_COHORTS: usize = 3 * 4096;
const MAX_SLOTS: usize = 65_536;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
pub(in crate::continuous_engine::inner::calibration) struct FiniteWork {
    pub declared_offers_upper: usize,
    pub declared_offers_minimum: usize,
    pub requests: usize,
    pub serial_declared_offer_rows: usize,
    pub execution_actions: usize,
    pub serial_token_work: usize,
}
impl FiniteWork {
    fn charge(&mut self, value: work::CaseWork) -> Result<()> {
        self.declared_offers_upper = add(self.declared_offers_upper, value.declared_offers_upper)?;
        self.declared_offers_minimum =
            add(self.declared_offers_minimum, value.declared_offers_minimum)?;
        self.requests = add(self.requests, value.requests)?;
        self.serial_declared_offer_rows = add(
            self.serial_declared_offer_rows,
            value.serial_declared_offer_rows,
        )?;
        self.execution_actions = add(self.execution_actions, value.execution_actions)?;
        self.serial_token_work = add(self.serial_token_work, value.serial_token_work)?;
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
pub(in crate::continuous_engine::inner::calibration) struct OfferedCut {
    pub minimum: usize,
    pub maximum: usize,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub(in crate::continuous_engine::inner::calibration) struct FinitePhase {
    pub start_cut: OfferedCut,
    pub floor_fillers: Range<usize>,
    pub representatives: Range<usize>,
    pub padding: Range<usize>,
    pub minimum_members: usize,
    pub phase_min_offered: usize,
    pub prefix_after_representatives: OfferedCut,
    pub next_cut: OfferedCut,
    pub prefix_after_padding: OfferedCut,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub(in crate::continuous_engine::inner::calibration) struct FiniteCertificate {
    pub discovery: Range<usize>,
    pub discovery_padding: Range<usize>,
    pub initial_fit_cut: OfferedCut,
    pub phases: [FinitePhase; 3],
}

#[derive(Debug, Clone, Serialize)]
pub(in crate::continuous_engine::inner::calibration) struct FinitePlan {
    pub occurrence_case_indices: Vec<usize>,
    /// Mandatory original representatives, grouped by their exact checked key.
    pub representatives: Vec<Vec<usize>>,
    pub family_keys: Vec<CheckedPopulationKey>,
    pub filler_case_indices: Vec<usize>,
    pub padding_case_index: usize,
    pub certificate: FiniteCertificate,
    pub schedule: OwnerBlockScheduleV1,
    /// Includes deduplicated acquisition setup. Setup never advances offered cuts.
    pub work: FiniteWork,
}

impl FinitePlan {
    pub fn retained_heap_bytes(&self) -> Option<usize> {
        let groups = self
            .representatives
            .capacity()
            .checked_mul(size_of::<Vec<usize>>())?;
        let reps = self.representatives.iter().try_fold(0usize, |n, group| {
            n.checked_add(group.capacity().checked_mul(size_of::<usize>())?)
        })?;
        groups
            .checked_add(reps)?
            .checked_add(
                self.family_keys
                    .capacity()
                    .checked_mul(size_of::<CheckedPopulationKey>())?,
            )?
            .checked_add(
                self.filler_case_indices
                    .capacity()
                    .checked_mul(size_of::<usize>())?,
            )?
            .checked_add(
                self.occurrence_case_indices
                    .capacity()
                    .checked_mul(size_of::<usize>())?,
            )
    }

    pub fn retained_payload_bytes(&self) -> Option<usize> {
        size_of::<Self>().checked_add(self.retained_heap_bytes()?)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub(in crate::continuous_engine::inner::calibration) enum FiniteRejection {
    RetainedCapacity,
    ScheduleCapacity,
    FrozenHorizon,
}

#[derive(Debug)]
pub(in crate::continuous_engine::inner::calibration) enum FiniteVerification {
    Ready(FinitePlan),
    Skip(FiniteRejection),
}

/// Conservative simultaneous storage for one result and its construction:
/// the fixed plan header, exact group/key/filler metadata, and the request-bounded
/// occurrence count. Every checked cohort spends at least one request, including
/// filler and padding occurrences. Keys contain no heap payload. The filler
/// vector is moved into the result, so it is not charged twice.
pub(in crate::continuous_engine::inner::calibration) fn storage_bound(
    families: usize,
    representatives: usize,
    maximum_requests: usize,
) -> Result<usize> {
    if families == 0 && representatives == 0 {
        return Ok(0);
    }
    storage_for(
        families,
        representatives,
        occurrence_limit(maximum_requests),
    )
}

fn occurrence_limit(maximum_requests: usize) -> usize {
    MAX_COHORTS.min(maximum_requests)
}

fn storage_for(families: usize, representatives: usize, occurrences: usize) -> Result<usize> {
    let per_family = add(
        size_of::<Vec<usize>>(),
        add(size_of::<CheckedPopulationKey>(), size_of::<usize>())?,
    )?;
    add(
        size_of::<FinitePlan>(),
        add(
            mul(families, per_family)?,
            mul(add(representatives, occurrences)?, size_of::<usize>())?,
        )?,
    )
}

pub(in crate::continuous_engine::inner::calibration) fn build(
    families: &[Vec<usize>],
    opportunities: &[CaseOpportunity],
    cases: &[Case],
    prompts: &[usize],
    chunk: usize,
    row_ceiling: Option<NonZeroU32>,
    settings: &StructuredSettingsV2,
    maximum_requests: usize,
    maximum_retained_bytes: usize,
) -> Result<FiniteVerification> {
    // Every representative occurs in the stream, and every filler/padding is
    // one of those representatives. Deduplicating their acquisition keys here
    // is therefore exactly setup_for_indices on the complete finite stream.
    let mut setup = work::CaseWork::default();
    for (family_index, family) in families.iter().enumerate() {
        for (position, &index) in family.iter().enumerate() {
            let case = cases
                .get(index)
                .ok_or_else(|| error("finite case outside frozen inputs"))?;
            if let Some(key) = case.acquisition {
                let earlier = families[..family_index]
                    .iter()
                    .flatten()
                    .chain(family[..position].iter());
                if !earlier
                    .into_iter()
                    .any(|&old| cases.get(old).is_some_and(|c| c.acquisition == Some(key)))
                {
                    work::add_setup(&mut setup, key)?;
                }
            }
        }
    }
    build_with_view(
        families,
        |index| {
            opportunities
                .get(index)
                .ok_or_else(|| error("finite opportunity outside frozen inputs"))
        },
        |index| {
            let case = cases
                .get(index)
                .ok_or_else(|| error("finite case outside frozen inputs"))?;
            let prompt = *prompts
                .get(case.template)
                .ok_or_else(|| error("finite prompt outside frozen inputs"))?;
            work::case_work(case, prompt, chunk, row_ceiling)
        },
        setup,
        settings,
        maximum_requests,
        maximum_retained_bytes,
    )
}

pub(in crate::continuous_engine::inner::calibration) fn build_with_view<'a>(
    families: &[Vec<usize>],
    opportunity_for: impl Fn(usize) -> Result<&'a CaseOpportunity>,
    work_for: impl Fn(usize) -> Result<work::CaseWork>,
    setup: work::CaseWork,
    settings: &StructuredSettingsV2,
    maximum_requests: usize,
    maximum_retained_bytes: usize,
) -> Result<FiniteVerification> {
    let count = validate_families(families, &opportunity_for, None)?;
    let Some(count) = count else {
        return Ok(FiniteVerification::Skip(FiniteRejection::ScheduleCapacity));
    };
    // This only bounds construction storage. Setup and multi-request cohorts
    // still contribute their full work to the caller's source admission.
    let maximum_occurrences = occurrence_limit(maximum_requests);
    // Before any allocation, authorize all fixed metadata. The dry pass uses
    // only stack certificates and the bounded filler vector.
    if storage_for(families.len(), count, 0)? > maximum_retained_bytes {
        return Ok(FiniteVerification::Skip(FiniteRejection::RetainedCapacity));
    }
    let mut fillers = reserved(families.len())?;
    let mut padding = None;
    for family in families {
        let mut cheapest = None;
        for &index in family {
            let value = checked_work(index, &work_for)?;
            let price = (
                value.requests,
                value.execution_actions,
                value.serial_token_work,
                value.declared_offers_upper,
                index,
            );
            if cheapest.as_ref().is_none_or(|old| price < *old) {
                cheapest = Some(price);
            }
            if match padding {
                None => true,
                Some((old_index, old)) => padding_precedes(index, value, old_index, old)?,
            } {
                padding = Some((index, value));
            }
        }
        fillers.push(cheapest.ok_or_else(|| error("finite family is empty"))?.4);
    }
    let padding = padding.ok_or_else(|| error("finite source is empty"))?.0;
    let members = member_floors(settings)?;
    let dry = traverse(
        families,
        &fillers,
        padding,
        members,
        &work_for,
        maximum_occurrences,
        None,
        None,
    )?;
    let Some(dry) = dry else {
        return Ok(FiniteVerification::Skip(FiniteRejection::ScheduleCapacity));
    };
    if storage_for(families.len(), count, dry.len)? > maximum_retained_bytes {
        return Ok(FiniteVerification::Skip(FiniteRejection::RetainedCapacity));
    }
    let Some(schedule) = schedule_for(&dry.certificate, members, settings)? else {
        return Ok(FiniteVerification::Skip(FiniteRejection::ScheduleCapacity));
    };
    let mut indices = reserved(dry.len)?;
    let repeated = traverse(
        families,
        &fillers,
        padding,
        members,
        &work_for,
        maximum_occurrences,
        None,
        Some(&mut indices),
    )?;
    if repeated.as_ref() != Some(&dry) {
        return Err(error("finite work lookup changed during construction"));
    }
    finish(
        families,
        &opportunity_for,
        fillers,
        padding,
        indices,
        dry,
        schedule,
        setup,
        maximum_retained_bytes,
    )
}

/// Reprove exactly the frozen occurrences and range boundaries with current
/// work. Native-to-cold replacement may enlarge work; it may not add cohorts,
/// choose different representatives, promote alternatives, or refund setup.
pub(in crate::continuous_engine::inner::calibration) fn verify_frozen<'a>(
    original: &FinitePlan,
    opportunity_for: impl Fn(usize) -> Result<&'a CaseOpportunity>,
    work_for: impl Fn(usize) -> Result<work::CaseWork>,
    setup: work::CaseWork,
    settings: &StructuredSettingsV2,
    maximum_retained_bytes: usize,
) -> Result<FiniteVerification> {
    let families = &original.representatives;
    let Some(count) = validate_families(families, &opportunity_for, Some(&original.family_keys))?
    else {
        return Err(error("finite frozen member floor changed"));
    };
    let members = member_floors(settings)?;
    if original.schedule.block_offered != 1
        || original.schedule.min_members != members
        || original.schedule.input_readiness.is_some()
        || original.schedule.opening_frontier.is_some()
        || original.schedule.algorithm_universe.is_some()
        || original.schedule.phase_support.is_some()
    {
        return Err(error("finite frozen schedule contract changed"));
    }
    if original.filler_case_indices.len() != families.len()
        || families
            .iter()
            .zip(&original.filler_case_indices)
            .any(|(group, filler)| !group.contains(filler))
        || !families
            .iter()
            .any(|group| group.contains(&original.padding_case_index))
    {
        return Err(error("finite frozen filler identity changed"));
    }
    if storage_for(
        families.len(),
        count,
        original.occurrence_case_indices.len(),
    )? > maximum_retained_bytes
    {
        return Ok(FiniteVerification::Skip(FiniteRejection::RetainedCapacity));
    }
    let verified = traverse(
        families,
        &original.filler_case_indices,
        original.padding_case_index,
        members,
        &work_for,
        MAX_COHORTS,
        Some(original),
        None,
    )?;
    let Some(verified) = verified else {
        return Ok(FiniteVerification::Skip(FiniteRejection::FrozenHorizon));
    };
    let Some(mut schedule) = schedule_for(&verified.certificate, members, settings)? else {
        return Ok(FiniteVerification::Skip(FiniteRejection::ScheduleCapacity));
    };
    schedule.prediction_validity = original.schedule.prediction_validity;
    let indices = copy_slice(&original.occurrence_case_indices)?;
    let fillers = copy_slice(&original.filler_case_indices)?;
    finish(
        families,
        &opportunity_for,
        fillers,
        original.padding_case_index,
        indices,
        verified,
        schedule,
        setup,
        maximum_retained_bytes,
    )
}

fn validate_families<'a>(
    families: &[Vec<usize>],
    lookup: &impl Fn(usize) -> Result<&'a CaseOpportunity>,
    expected_keys: Option<&[CheckedPopulationKey]>,
) -> Result<Option<usize>> {
    if families.is_empty() || expected_keys.is_some_and(|keys| keys.len() != families.len()) {
        return Err(error("finite family identities differ"));
    }
    let mut total = 0;
    for (i, family) in families.iter().enumerate() {
        let Some(&first) = family.first() else {
            return Err(error("empty finite family"));
        };
        let Some(key) = guaranteed_key(lookup(first)?)? else {
            return Ok(None);
        };
        if expected_keys.is_some_and(|keys| &keys[i] != key) {
            return Err(error("finite frozen checked family changed"));
        }
        for previous in &families[..i] {
            if guaranteed_key(lookup(previous[0])?)? == Some(key) {
                return Err(error("finite family is declared twice"));
            }
        }
        for (position, &index) in family.iter().enumerate() {
            let Some(actual) = guaranteed_key(lookup(index)?)? else {
                return Ok(None);
            };
            if actual != key || family[..position].contains(&index) {
                return Err(error(
                    "finite representative is not one original exact family",
                ));
            }
        }
        total = add(total, family.len())?;
    }
    Ok(Some(total))
}

fn guaranteed_key(opportunity: &CaseOpportunity) -> Result<Option<&CheckedPopulationKey>> {
    if opportunity.minimum_fresh_members > 1 {
        return Err(error("finite opportunity member floor is invalid"));
    }
    Ok(
        match (&opportunity.population, opportunity.minimum_fresh_members) {
            (CasePopulation::Unique(key), 1) => Some(key),
            _ => None,
        },
    )
}

fn member_floors(settings: &StructuredSettingsV2) -> Result<[usize; 3]> {
    settings
        .validate()
        .map_err(|reason| error(format!("finite settings: {reason:?}")))?;
    Ok([
        add(settings.max_rank, settings.min_fit_redundancy)?.max(settings.min_phase_samples),
        settings.min_phase_samples,
        settings.min_phase_samples,
    ])
}

fn schedule_for(
    certificate: &FiniteCertificate,
    members: [usize; 3],
    settings: &StructuredSettingsV2,
) -> Result<Option<OwnerBlockScheduleV1>> {
    let phases = std::array::from_fn(|i| certificate.phases[i].phase_min_offered);
    let schedule = OwnerBlockScheduleV1::new(1, phases, members)
        .map_err(|reason| error(format!("finite schedule: {reason:?}")))?;
    let mut derived = settings.clone();
    // Match the original batch planner: this was a bound derived from the old
    // schedule, not an operator cap. validate() still enforces the hard 4096.
    derived.max_phase_samples = *schedule.maximum_phase_members.iter().max().unwrap();
    if derived.validate().is_err() || schedule.validate(&derived).is_err() {
        return Ok(None);
    }
    Ok(Some(schedule))
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
struct Traversal {
    len: usize,
    work: FiniteWork,
    certificate: FiniteCertificate,
}

struct Stream<'a> {
    state: Traversal,
    maximum_occurrences: usize,
    frozen: Option<&'a FinitePlan>,
    output: Option<&'a mut Vec<usize>>,
}
impl Stream<'_> {
    fn offered(&self) -> OfferedCut {
        OfferedCut {
            minimum: self.state.work.declared_offers_minimum,
            maximum: self.state.work.declared_offers_upper,
        }
    }
    fn append(
        &mut self,
        index: usize,
        lookup: &impl Fn(usize) -> Result<work::CaseWork>,
    ) -> Result<bool> {
        if let Some(frozen) = self.frozen {
            if frozen.occurrence_case_indices.get(self.state.len) != Some(&index) {
                return Err(error("finite frozen occurrence identity changed"));
            }
        }
        let value = checked_work(index, lookup)?;
        self.state.work.charge(value)?;
        self.state.len = add(self.state.len, 1)?;
        if self.state.len > self.maximum_occurrences
            || value.requests > 4096
            || self.state.work.requests > MAX_SLOTS
            || self.state.work.serial_declared_offer_rows > MAX_SLOTS
        {
            return Ok(false);
        }
        if let Some(output) = &mut self.output {
            if output.len() == output.capacity() {
                return Err(error("finite output exceeded its preauthorized storage"));
            }
            output.push(index);
        }
        Ok(true)
    }
    fn pad(
        &mut self,
        until: usize,
        index: usize,
        frozen_end: Option<usize>,
        lookup: &impl Fn(usize) -> Result<work::CaseWork>,
    ) -> Result<bool> {
        let end = match frozen_end {
            Some(end) if end >= self.state.len => end,
            Some(_) => return Err(error("finite frozen padding range changed")),
            None => add(
                self.state.len,
                until
                    .saturating_sub(self.offered().minimum)
                    .div_ceil(checked_work(index, lookup)?.declared_offers_minimum),
            )?,
        };
        if end > self.maximum_occurrences {
            return Ok(false);
        }
        while self.state.len < end {
            if !self.append(index, lookup)? {
                return Ok(false);
            }
        }
        Ok(self.offered().minimum >= until)
    }
}

fn traverse(
    families: &[Vec<usize>],
    fillers: &[usize],
    padding: usize,
    members: [usize; 3],
    lookup: &impl Fn(usize) -> Result<work::CaseWork>,
    maximum_occurrences: usize,
    frozen: Option<&FinitePlan>,
    output: Option<&mut Vec<usize>>,
) -> Result<Option<Traversal>> {
    let mut stream = Stream {
        state: Traversal::default(),
        maximum_occurrences,
        frozen,
        output,
    };
    for &index in fillers {
        if !stream.append(index, lookup)? {
            return Ok(None);
        }
    }
    stream.state.certificate.discovery = 0..stream.state.len;
    let mut cut = OfferedCut {
        minimum: 1,
        maximum: stream.offered().maximum,
    };
    stream.state.certificate.initial_fit_cut = cut;
    let start = stream.state.len;
    if !stream.pad(
        cut.maximum,
        padding,
        frozen.map(|p| p.certificate.discovery_padding.end),
        lookup,
    )? {
        return Ok(None);
    }
    stream.state.certificate.discovery_padding = start..stream.state.len;
    for (phase, minimum_members) in members.into_iter().enumerate() {
        if stream.offered().minimum < cut.maximum {
            return Ok(None);
        }
        let start = stream.state.len;
        for (family, &filler) in families.iter().zip(fillers) {
            for _ in 0..minimum_members.saturating_sub(family.len()) {
                if !stream.append(filler, lookup)? {
                    return Ok(None);
                }
            }
        }
        let floor_fillers = start..stream.state.len;
        let start = stream.state.len;
        for family in families {
            for &index in family {
                if !stream.append(index, lookup)? {
                    return Ok(None);
                }
            }
        }
        let representatives = start..stream.state.len;
        let prefix_after_representatives = stream.offered();
        let phase_min_offered = prefix_after_representatives
            .maximum
            .checked_sub(cut.minimum)
            .ok_or_else(|| error("finite anchors precede opening"))?
            .max(minimum_members);
        let next_cut = OfferedCut {
            minimum: add(cut.minimum, phase_min_offered)?,
            maximum: add(cut.maximum, phase_min_offered)?,
        };
        let start = stream.state.len;
        if !stream.pad(
            next_cut.maximum,
            padding,
            frozen.map(|p| p.certificate.phases[phase].padding.end),
            lookup,
        )? {
            return Ok(None);
        }
        let padding = start..stream.state.len;
        if let Some(original) = frozen {
            let old = &original.certificate.phases[phase];
            if old.minimum_members != minimum_members
                || old.floor_fillers != floor_fillers
                || old.representatives != representatives
                || old.padding != padding
            {
                return Err(error("finite frozen phase ranges changed"));
            }
        }
        stream.state.certificate.phases[phase] = FinitePhase {
            start_cut: cut,
            floor_fillers,
            representatives,
            padding,
            minimum_members,
            phase_min_offered,
            prefix_after_representatives,
            next_cut,
            prefix_after_padding: stream.offered(),
        };
        cut = next_cut;
    }
    if let Some(original) = frozen {
        if original.occurrence_case_indices.len() != stream.state.len
            || original.certificate.discovery != stream.state.certificate.discovery
            || original.certificate.discovery_padding != stream.state.certificate.discovery_padding
        {
            return Err(error("finite frozen stream boundaries changed"));
        }
    }
    Ok(Some(stream.state))
}

fn finish<'a>(
    families: &[Vec<usize>],
    lookup: &impl Fn(usize) -> Result<&'a CaseOpportunity>,
    fillers: Vec<usize>,
    padding: usize,
    indices: Vec<usize>,
    mut traversal: Traversal,
    schedule: OwnerBlockScheduleV1,
    setup: work::CaseWork,
    maximum_retained_bytes: usize,
) -> Result<FiniteVerification> {
    if setup.declared_offers_minimum != 0
        || setup.declared_offers_upper != 0
        || setup.serial_declared_offer_rows != 0
    {
        return Err(error("finite setup cannot create offered members"));
    }
    traversal.work.charge(setup)?;
    let mut groups = reserved(families.len())?;
    let mut keys = reserved(families.len())?;
    for family in families {
        groups.push(copy_slice(family)?);
        keys.push(
            guaranteed_key(lookup(family[0])?)?
                .ok_or_else(|| error("finite member floor changed"))?
                .clone(),
        );
    }
    let plan = FinitePlan {
        occurrence_case_indices: indices,
        representatives: groups,
        family_keys: keys,
        filler_case_indices: fillers,
        padding_case_index: padding,
        certificate: traversal.certificate,
        schedule,
        work: traversal.work,
    };
    if plan
        .retained_payload_bytes()
        .is_none_or(|bytes| bytes > maximum_retained_bytes)
    {
        return Ok(FiniteVerification::Skip(FiniteRejection::RetainedCapacity));
    }
    Ok(FiniteVerification::Ready(plan))
}

fn checked_work(
    index: usize,
    lookup: &impl Fn(usize) -> Result<work::CaseWork>,
) -> Result<work::CaseWork> {
    let value = lookup(index)?;
    if value.requests == 0
        || value.declared_offers_minimum == 0
        || value.declared_offers_minimum > value.declared_offers_upper
    {
        return Err(error("finite cohort has invalid work bounds"));
    }
    Ok(value)
}

fn padding_precedes(
    index: usize,
    value: work::CaseWork,
    old_index: usize,
    old: work::CaseWork,
) -> Result<bool> {
    let requests = mul(value.requests, old.declared_offers_minimum)?
        .cmp(&mul(old.requests, value.declared_offers_minimum)?);
    let actions = mul(value.execution_actions, old.declared_offers_minimum)?
        .cmp(&mul(old.execution_actions, value.declared_offers_minimum)?);
    Ok((requests, actions, index.cmp(&old_index))
        < (
            std::cmp::Ordering::Equal,
            std::cmp::Ordering::Equal,
            std::cmp::Ordering::Equal,
        ))
}
fn reserved<T>(count: usize) -> Result<Vec<T>> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(count)
        .map_err(|_| error("finite storage allocation failed"))?;
    Ok(values)
}
fn copy_slice<T: Clone>(values: &[T]) -> Result<Vec<T>> {
    let mut out = reserved(values.len())?;
    out.extend_from_slice(values);
    Ok(out)
}
fn add(a: usize, b: usize) -> Result<usize> {
    a.checked_add(b)
        .ok_or_else(|| error("finite accounting overflow"))
}
fn mul(a: usize, b: usize) -> Result<usize> {
    a.checked_mul(b)
        .ok_or_else(|| error("finite accounting overflow"))
}

#[cfg(test)]
mod tests;
