//! One immutable identified Fit and one independently qualified joint bank.
//! All membership and stopping decisions precede duration inspection. Complete
//! original blocks and phase seals are provided by the owner-block protocol.
use super::*;
use std::collections::BTreeMap;

#[cfg(test)]
mod range_diagnostic;
#[cfg(test)]
mod storage_tests;
#[cfg(test)]
mod tests;

/// Opaque input-only cell under one numerical/algorithm contract. Exact zero
/// and dyadic positive bands cover the complete input/work tuple; the split
/// binds dimensions. Comparing keys from different contracts grants no coverage.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct StructuredJointCellKeyV1 {
    input: Vec<u8>,
    work: Vec<u8>,
}
type JointCellKey = StructuredJointCellKeyV1;
impl StructuredJointCellKeyV1 {
    /// Owned inline key and both vector capacities, excluding allocator metadata.
    pub fn retained_payload_bytes(&self) -> Option<usize> {
        std::mem::size_of::<Self>().checked_add(self.retained()?)
    }
    fn of(input: &[u64], work: &[u64]) -> Self {
        fn bands(values: &[u64]) -> Vec<u8> {
            values
                .iter()
                .map(|&n| (64 - n.leading_zeros()) as u8)
                .collect()
        }
        Self {
            input: bands(input),
            work: bands(work),
        }
    }
    fn bind(&self, h: &mut Sha256) {
        h.update((self.input.len() as u64).to_le_bytes());
        h.update(&self.input);
        h.update((self.work.len() as u64).to_le_bytes());
        h.update(&self.work);
    }
    fn retained(&self) -> Option<usize> {
        self.input.capacity().checked_add(self.work.capacity())
    }
}

/// Restore only completion outcomes to their original pre-settlement facts.
/// The original source separately validates the actual settlement and wall.
fn prospective_input(input: &StructuredInputV2) -> Result<StructuredInputV2> {
    let mut value = input.clone();
    if let Some(c) = &mut value.completion {
        c.positions = value.physical_host_rows.iter()
            .filter(|r| super::super::completion::installed(r)
                && r.terminal_expectation == ferrum_interfaces::execution_cost::HostTerminalExpectationV1::LengthBoundary)
            .map(|r| r.physical_position).collect();
        let moments = super::super::input::position_moments(&c.positions)?;
        value.basis[c.basis_offset..c.basis_offset + 3].copy_from_slice(&moments.map(|v| v as f64));
        value.support[c.support_offset..c.support_offset + 3].copy_from_slice(&moments);
        c.settled = false;
    }
    value.settled_terminal_causes = None;
    Ok(value)
}

fn input_key(
    input: &StructuredInputV2,
    contract: &NonNegativeEnvelopeContractV1,
) -> Result<JointCellKey> {
    contract.validate_input(input)?;
    let query = StructuredQueryV2::exact(prospective_input(input)?);
    let work = physical_envelope::envelope::query_upper(&query, &contract.workload_domain)?;
    Ok(JointCellKey::of(
        query.input.joint_support_coordinates(),
        &work,
    ))
}

pub(super) fn enabled(contract: &NonNegativeEnvelopeContractV1) -> bool {
    contract.planning_estimator == NonNegativePlanningEstimatorV1::IdentifiedFitJointCellsV1
}

impl NonNegativeEnvelopeContractV1 {
    /// Input-only projection shared by preparation and the frozen numerical bank.
    /// No duration, fit, source receipt or prediction authority is involved.
    /// The caller retains its own declared input/work/memory limits.
    pub fn joint_input_cell(&self, input: &StructuredInputV2) -> Result<StructuredJointCellKeyV1> {
        if !enabled(self) {
            return Err(StructuredUnknown::WrongProtocol);
        }
        // Match the checked canonical input construction ceiling. This also
        // validates completion offsets before prospective normalization.
        input.validate(&StructuredSettingsV2 {
            max_axes: 4096,
            ..Default::default()
        })?;
        input_key(input, self)
    }
}

/// Input groups are fully built and bounded before a caller reads any labels.
/// At most max_phase_samples keys and twice max_axes coordinates per key are
/// retained. The collector's original sample/workspace reservation includes
/// these smaller byte vectors, indices and per-cell statistics.
fn groups(
    samples: &[StructuredNumericObservationV2],
    contract: &NonNegativeEnvelopeContractV1,
    settings: &StructuredSettingsV2,
) -> Result<BTreeMap<JointCellKey, Vec<usize>>> {
    if samples.len() > settings.max_phase_samples {
        return Err(StructuredUnknown::Capacity);
    }
    let mut visits = 0u64;
    for sample in samples {
        let input = &sample.input;
        if input.support.len() > settings.max_axes || input.basis.len() > settings.max_axes {
            return Err(StructuredUnknown::Capacity);
        }
        visits = visits
            .checked_add(
                ((input.support.len() + input.basis.len()) as u64)
                    .checked_mul(8)
                    .ok_or(StructuredUnknown::Capacity)?,
            )
            .filter(|&n| n <= contract.settings.maximum_coordinate_visits)
            .ok_or(StructuredUnknown::Capacity)?;
    }
    let mut out = BTreeMap::<JointCellKey, Vec<usize>>::new();
    for (i, sample) in samples.iter().enumerate() {
        sample.input.validate(settings)?;
        let key = input_key(&sample.input, contract)?;
        if key.input.len() > settings.max_axes || key.work.len() > settings.max_axes {
            return Err(StructuredUnknown::Capacity);
        }
        out.entry(key).or_default().push(i);
    }
    Ok(out)
}

/// Earliest complete global block at which at least one whole cell has the
/// declared member minimum. Prefix replay uses this same input-only rule.
pub(super) fn assess_inputs(
    schedule: &OwnerBlockScheduleV1,
    contract: &NonNegativeEnvelopeContractV1,
    phase: StructuredPhaseV2,
    offered: u64,
    samples: &[StructuredNumericObservationV2],
    settings: &StructuredSettingsV2,
    visits: &mut u64,
) -> Result<OwnerInputReadinessDecisionV1> {
    use OwnerInputReadinessDecisionV1::*;
    let policy = schedule
        .input_readiness
        .as_ref()
        .ok_or(StructuredUnknown::WrongProtocol)?;
    let blocks = offered / schedule.block_offered as u64;
    let maximum = policy.maximum_phase_blocks[phase.index()] as u64;
    if phase == StructuredPhaseV2::Fit || blocks > maximum {
        return Err(StructuredUnknown::WrongProtocol);
    }
    let exhausted = |gap| {
        if blocks == maximum {
            Exhausted(gap)
        } else {
            Wait
        }
    };
    if !schedule.is_ready(phase, offered, samples.len())? {
        return Ok(exhausted(OwnerInputReadinessGapV1::CompleteBlockLimit));
    }
    // Reserve both pre-settlement projections and tuple comparisons before
    // allocating. Original collection and independent replay each get half.
    let charge = samples.iter().try_fold(0u64, |sum, s| {
        let n = s
            .input
            .support
            .len()
            .checked_add(s.input.basis.len())
            .and_then(|n| n.checked_mul(4))
            .ok_or(StructuredUnknown::Capacity)?;
        sum.checked_add(n as u64).ok_or(StructuredUnknown::Capacity)
    })?;
    let Some(next) = visits
        .checked_add(charge)
        .filter(|&n| n <= policy.maximum_geometry_visits / 2)
    else {
        return Ok(Exhausted(OwnerInputReadinessGapV1::GeometryWorkBudget));
    };
    *visits = next;
    let inputs = groups(samples, contract, settings)?;
    Ok(
        if inputs
            .values()
            .any(|v| v.len() >= schedule.min_members[phase.index()])
        {
            Freeze
        } else {
            exhausted(OwnerInputReadinessGapV1::MissingInputCoverage)
        },
    )
}

#[derive(Debug)]
struct CellMargin {
    residual_ns: u64,
    span_ns: u64,
    residual_members: usize,
    qualification_members: usize,
    qualification_misses: usize,
    qualified: bool,
}

pub(super) struct JointMarginBank {
    // R consumes the sorted, unique input groups once. Q changes only margins;
    // keys never move or grow after R freezes.
    cells: Vec<(JointCellKey, CellMargin)>,
}
impl JointMarginBank {
    fn cell(&self, key: &JointCellKey) -> Option<&CellMargin> {
        self.cells
            .binary_search_by(|(candidate, _)| candidate.cmp(key))
            .ok()
            .map(|index| &self.cells[index].1)
    }
    fn cell_mut(&mut self, key: &JointCellKey) -> Option<&mut CellMargin> {
        let index = self
            .cells
            .binary_search_by(|(candidate, _)| candidate.cmp(key))
            .ok()?;
        Some(&mut self.cells[index].1)
    }
    pub(super) fn contains_qualified(&self, previous: &Self) -> bool {
        let mut current = self.cells.iter().filter(|(_, cell)| cell.qualified);
        previous
            .cells
            .iter()
            .filter(|(_, cell)| cell.qualified)
            .all(|(key, _)| loop {
                match current.next() {
                    Some((candidate, _)) if candidate < key => continue,
                    Some((candidate, _)) => break candidate == key,
                    None => break false,
                }
            })
    }
    pub(super) fn bind(&self, h: &mut Sha256, qualification: bool) {
        h.update(b"ferrum.full-input-dyadic-joint-bank.v1\0");
        h.update((self.cells.len() as u64).to_le_bytes());
        for (key, cell) in &self.cells {
            key.bind(h);
            for n in [cell.residual_ns, cell.span_ns, cell.residual_members as u64] {
                h.update(n.to_le_bytes());
            }
            if qualification {
                h.update((cell.qualification_members as u64).to_le_bytes());
                h.update((cell.qualification_misses as u64).to_le_bytes());
                h.update([u8::from(cell.qualified)]);
            }
        }
    }
    pub(super) fn retained(&self) -> Option<usize> {
        // Count every allocated slot, including spare Vec capacity. The model's
        // inline size already includes this bank and its Vec header.
        let slots = self
            .cells
            .capacity()
            .checked_mul(std::mem::size_of::<(JointCellKey, CellMargin)>())?;
        self.cells
            .iter()
            .try_fold(slots, |n, (key, _)| n.checked_add(key.retained()?))
    }
    pub(super) fn contains(
        &self,
        input: &StructuredInputV2,
        contract: &NonNegativeEnvelopeContractV1,
        qualified: bool,
    ) -> Result<bool> {
        let key = input_key(input, contract)?;
        Ok(self.cell(&key).is_some_and(|c| !qualified || c.qualified))
    }
    fn query_geometry(
        fitted: &FittedStructuredModelV2,
        query: &StructuredQueryV2,
    ) -> QueryResult<(JointCellKey, Vec<u64>)> {
        let projected = fitted.numerical_query(query)?;
        let query = projected.as_ref();
        fitted.same_query_population(&query.input)?;
        query.validate_for_prediction(&fitted.settings)?;
        let physical = fitted
            .numerical
            .physical()
            .ok_or(StructuredUnknown::WrongProtocol)?;
        let mut prospective = query.clone();
        prospective.input = prospective_input(&query.input)?;
        let work = physical.authorized_query_upper(&prospective, 0, false)?;
        let key = JointCellKey::of(prospective.input.joint_support_coordinates(), &work);
        Ok((key, work))
    }
    fn query_parts(
        fitted: &FittedStructuredModelV2,
        query: &StructuredQueryV2,
    ) -> QueryResult<(JointCellKey, u64)> {
        let (key, work) = Self::query_geometry(fitted, query)?;
        let physical = fitted
            .numerical
            .physical()
            .ok_or(StructuredUnknown::WrongProtocol)?;
        let point = NonNegativePlanningEstimatorV1::FittedResidualV1
            .prediction_detailed(&physical.fit, &work)?;
        Ok((key, point))
    }
    pub(super) fn catalog_membership(
        &self,
        fitted: &FittedStructuredModelV2,
        query: &StructuredQueryV2,
    ) -> QueryResult<bool> {
        let (key, _) = Self::query_geometry(fitted, query)?;
        Ok(self.cell(&key).is_some_and(|c| c.qualified))
    }
    pub(super) fn prediction(
        &self,
        fitted: &FittedStructuredModelV2,
        query: &StructuredQueryV2,
        qualified: bool,
    ) -> QueryResult<StructuredPredictionV2> {
        let (key, point) = Self::query_parts(fitted, query)?;
        let c = self
            .cell(&key)
            .filter(|c| !qualified || c.qualified)
            .ok_or(StructuredQueryFailureV2::OutsideSupport(
                StructuredUnknown::JointSupport,
            ))?;
        let floor = fitted.numerical.fit_error_floor_ns();
        let effective = floor.max(c.residual_ns);
        let planning_ns = super::super::query_outcome::planning_sum(
            point,
            effective,
            fitted.settings.static_margin_ns,
            c.span_ns,
            fitted.settings.max_wave_ns,
        )?;
        Ok(StructuredPredictionV2 {
            fitted_lower_ns: 0,
            fitted_upper_ns: point,
            residual_ns: c.residual_ns,
            fit_error_floor_ns: floor,
            effective_residual_ns: effective,
            learned_span_margin_ns: c.span_ns,
            planning_ns,
            valid_until_ns: fitted.state.expires_at_ns,
            fit_samples: fitted.fit_samples,
            residual_samples: c.residual_members,
            identified_rank: fitted.numerical.rank(),
        })
    }
}

impl FittedStructuredModelV2 {
    pub(super) fn uses_joint_cells(&self) -> bool {
        self.numerical
            .physical()
            .is_some_and(|p| enabled(&p.contract))
    }
    pub(super) fn calibrate_joint_cells(
        mut self,
        samples: &[StructuredNumericObservationV2],
        frozen_at_ns: u64,
    ) -> Result<CalibratedStructuredModelV2> {
        self.source
            .complete_population(samples, StructuredPhaseV2::Residual)?;
        check_time(frozen_at_ns, self.frozen_at_ns, self.state.expires_at_ns)?;
        let population = self
            .owner_blocks
            .as_ref()
            .ok_or(StructuredUnknown::WrongProtocol)?;
        let minimum = population.contract.schedule.min_members[1];
        let contract = &self
            .numerical
            .physical()
            .ok_or(StructuredUnknown::WrongProtocol)?
            .contract;
        // Input count/capacity admission is complete before labels below.
        let groups = groups(samples, contract, &self.settings)?;
        for sample in samples {
            self.state.observe(
                sample,
                &self.fingerprint,
                &self.settings,
                &self.source,
                StructuredPhaseV2::Residual,
                self.frozen_at_ns,
                frozen_at_ns,
            )?;
            self.same_population_domain(&sample.input)?;
        }
        let count = groups.values().filter(|v| v.len() >= minimum).count();
        let mut cells = Vec::with_capacity(count);
        for (key, members) in groups.into_iter().filter(|(_, v)| v.len() >= minimum) {
            let mut errors = Vec::with_capacity(members.len());
            for &i in &members {
                let (_, point) = JointMarginBank::query_parts(
                    &self,
                    &StructuredQueryV2::exact(samples[i].input.clone()),
                )
                .map_err(StructuredQueryFailureV2::reason)?;
                errors.push(i128::from(samples[i].wall_ns) - i128::from(point));
            }
            errors.sort_unstable();
            let residual_ns = errors[(99 * errors.len()).div_ceil(100) - 1].max(0) as u64;
            let span_ns = self
                .settings
                .learned_drift
                .freeze(errors[0], *errors.last().unwrap())?;
            cells.push((
                key,
                CellMargin {
                    residual_ns,
                    span_ns,
                    residual_members: members.len(),
                    qualification_members: 0,
                    qualification_misses: 0,
                    qualified: false,
                },
            ));
        }
        if cells.is_empty() {
            return Err(StructuredUnknown::InsufficientSamples);
        }
        let residual_ns = cells.iter().map(|(_, c)| c.residual_ns).max().unwrap();
        let learned_span_margin_ns = cells.iter().map(|(_, c)| c.span_ns).max().unwrap();
        let completion_coverage = self.completion_coverage;
        Ok(CalibratedStructuredModelV2 {
            fitted: self,
            residual_ns,
            learned_span_margin_ns,
            residual_support: None,
            residual_samples: samples.len(),
            frozen_at_ns,
            completion_coverage,
            joint_bank: Some(JointMarginBank { cells }),
        })
    }
}

impl CalibratedStructuredModelV2 {
    pub(super) fn qualify_joint_cells(
        mut self,
        samples: &[StructuredNumericObservationV2],
        frozen_at_ns: u64,
    ) -> Result<QualifiedStructuredModelV2> {
        self.fitted
            .source
            .complete_population(samples, StructuredPhaseV2::Qualification)?;
        check_time(
            frozen_at_ns,
            self.frozen_at_ns,
            self.fitted.state.expires_at_ns,
        )?;
        let minimum = self
            .fitted
            .owner_blocks
            .as_ref()
            .ok_or(StructuredUnknown::WrongProtocol)?
            .contract
            .schedule
            .min_members[2];
        // Every Q member was selected through the already frozen R bank. Read
        // all original members once; rejection never changes or retries R.
        for sample in samples {
            self.fitted.state.observe(
                sample,
                &self.fitted.fingerprint,
                &self.fitted.settings,
                &self.fitted.source,
                StructuredPhaseV2::Qualification,
                self.frozen_at_ns,
                frozen_at_ns,
            )?;
            let query = StructuredQueryV2::exact(sample.input.clone());
            let prediction = self
                .joint_bank
                .as_ref()
                .ok_or(StructuredUnknown::WrongProtocol)?
                .prediction(&self.fitted, &query, false)
                .map_err(StructuredQueryFailureV2::reason)?;
            let (key, _) = JointMarginBank::query_parts(&self.fitted, &query)
                .map_err(StructuredQueryFailureV2::reason)?;
            let cell = self
                .joint_bank
                .as_mut()
                .unwrap()
                .cell_mut(&key)
                .ok_or(StructuredUnknown::WrongProtocol)?;
            cell.qualification_members += 1;
            cell.qualification_misses += usize::from(sample.wall_ns > prediction.planning_ns);
        }
        let bank = self.joint_bank.as_mut().unwrap();
        for (_, cell) in &mut bank.cells {
            cell.qualified =
                cell.qualification_members >= minimum && cell.qualification_misses == 0;
        }
        if !bank.cells.iter().any(|(_, c)| c.qualified) {
            return Err(StructuredUnknown::QualificationUnderestimate);
        }
        self.fitted.state.freeze_calls()?;
        let completion_coverage = self.completion_coverage;
        Ok(QualifiedStructuredModelV2 {
            calibrated: self,
            qualification_samples: samples.len(),
            qualification_support: None,
            frozen_at_ns,
            completion_coverage,
        })
    }
}

#[cfg(test)]
impl JointMarginBank {
    fn diagnostic(&self) -> serde_json::Value {
        let cells: Vec<_> = self
            .cells
            .iter()
            .take(32)
            .map(|(key, c)| {
                let mut h = Sha256::new();
                key.bind(&mut h);
                serde_json::json!({"key_sha256":format!("{:x}", h.finalize()),
                "dimensions":key.input.len()+key.work.len(),
                "residual_members":c.residual_members,"residual_ns":c.residual_ns,
                "span_ns":c.span_ns,"qualification_members":c.qualification_members,
                "qualification_misses":c.qualification_misses,"qualified":c.qualified})
            })
            .collect();
        serde_json::json!({"cells":cells,"total_cells":self.cells.len(),
            "qualified_cells":self.cells.iter().filter(|(_, c)|c.qualified).count(),
            "truncated":self.cells.len()>32})
    }
}
#[cfg(test)]
impl CalibratedStructuredModelV2 {
    pub(crate) fn diagnose_joint_cell_bank(&self) -> Option<serde_json::Value> {
        self.joint_bank.as_ref().map(JointMarginBank::diagnostic)
    }
    pub(crate) fn diagnose_joint_cell_inputs(
        &self,
        samples: &[StructuredNumericObservationV2],
    ) -> Option<serde_json::Value> {
        self.fitted.diagnose_joint_cell_inputs(samples)
    }
}
#[cfg(test)]
impl FittedStructuredModelV2 {
    pub(crate) fn diagnose_joint_cell_inputs(
        &self,
        samples: &[StructuredNumericObservationV2],
    ) -> Option<serde_json::Value> {
        let contract = &self.numerical.physical()?.contract;
        let grouped = groups(samples, contract, &self.settings).ok()?;
        let cells: Vec<_> = grouped.iter().take(32).map(|(key, members)| {
            let mut h = Sha256::new(); key.bind(&mut h);
            serde_json::json!({"key_sha256":format!("{:x}",h.finalize()),"members":members.len()})
        }).collect();
        Some(
            serde_json::json!({"cells":cells,"total_cells":grouped.len(),"truncated":grouped.len()>32}),
        )
    }
}
#[cfg(test)]
impl QualifiedStructuredModelV2 {
    pub(crate) fn diagnose_joint_cell_bank(&self) -> Option<serde_json::Value> {
        self.calibrated.diagnose_joint_cell_bank()
    }
}
