//! Input-only cold stopping. This module never reads a wall, fitted error or
//! settled terminal cause. These DTOs have no execution/publication authority.
use super::super::physical_envelope::envelope;
use super::*;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum InputReadinessRevision {
    WorkAxesAndBranchesV1,
    WorkAxesAndBranchesV2,
    WorkAxesAndBranchesV3,
    WorkAxesAndBranchesV4,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OwnerInputReadinessV1 {
    revision: InputReadinessRevision,
    pub maximum_phase_blocks: [usize; 3],
    /// Total readiness scalar visits for one phase: the live/replay collector
    /// scan and the numerical close's independent prefix verification each
    /// reserve half. Both passes accumulate every prefix; no retry resets it.
    pub maximum_geometry_visits: u64,
}
impl OwnerInputReadinessV1 {
    pub fn new(maximum_phase_blocks: [usize; 3], maximum_geometry_visits: u64) -> Result<Self> {
        let value = Self {
            revision: InputReadinessRevision::WorkAxesAndBranchesV1,
            maximum_phase_blocks,
            maximum_geometry_visits,
        };
        value.validate()?;
        Ok(value)
    }
    /// Cached residuals are candidate-selection hints only. Selected pivots and
    /// the final span are rechecked from original inputs; numerical fitting is
    /// unchanged. A separate revision preserves V1 work and close boundaries.
    pub fn new_cached_residual_v2(
        maximum_phase_blocks: [usize; 3],
        maximum_geometry_visits: u64,
    ) -> Result<Self> {
        let mut value = Self::new(maximum_phase_blocks, maximum_geometry_visits)?;
        value.revision = InputReadinessRevision::WorkAxesAndBranchesV2;
        Ok(value)
    }
    /// Preserve V2's geometry kernel and tolerances after a complete original
    /// input scan removes only identically zero columns from its scratch.
    pub fn new_zero_column_v3(
        maximum_phase_blocks: [usize; 3],
        maximum_geometry_visits: u64,
    ) -> Result<Self> {
        let mut value = Self::new(maximum_phase_blocks, maximum_geometry_visits)?;
        value.revision = InputReadinessRevision::WorkAxesAndBranchesV3;
        Ok(value)
    }
    /// V3 geometry and allowances; Residual and Qualification must recover the
    /// frozen Fit work/branch target before their first complete input-ready cut.
    /// Fit retains the independently declared phase-support policy.
    pub fn new_fit_target_v4(
        maximum_phase_blocks: [usize; 3],
        maximum_geometry_visits: u64,
    ) -> Result<Self> {
        let mut value = Self::new_zero_column_v3(maximum_phase_blocks, maximum_geometry_visits)?;
        value.revision = InputReadinessRevision::WorkAxesAndBranchesV4;
        Ok(value)
    }
    pub(super) fn validate(&self) -> Result<()> {
        if self
            .maximum_phase_blocks
            .iter()
            .any(|&n| n == 0 || n > 65_536)
            || self.maximum_geometry_visits == 0
            || self.maximum_geometry_visits > 128_000_000
        {
            return Err(StructuredUnknown::InvalidSettings);
        }
        Ok(())
    }
    fn pass_visit_limit(&self) -> u64 {
        // A fixed split makes earliest-close decisions identical in original
        // collection and independent replay, without a second full allowance.
        self.maximum_geometry_visits / 2
    }

    pub(super) fn bind(&self, h: &mut Sha256) {
        match self.revision {
            InputReadinessRevision::WorkAxesAndBranchesV1 => {
                h.update(b"ferrum.owner-input-readiness.v1\0");
            }
            InputReadinessRevision::WorkAxesAndBranchesV2 => {
                h.update(b"ferrum.owner-input-readiness.cached-residual.v2\0");
            }
            InputReadinessRevision::WorkAxesAndBranchesV3 => {
                h.update(b"ferrum.owner-input-readiness.zero-column.v3\0");
            }
            InputReadinessRevision::WorkAxesAndBranchesV4 => {
                h.update(b"ferrum.owner-input-readiness.fit-target.v4\0");
            }
        }
        for n in self.maximum_phase_blocks {
            h.update((n as u64).to_le_bytes());
        }
        h.update(self.maximum_geometry_visits.to_le_bytes());
    }
}

/// Union of checked pre-execution work from independent discovery or Fit.
/// Completion-result columns are deliberately excluded from the positive mask.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OwnerInputTargetV1 {
    positive: Vec<bool>,
    // pending, Length, mask upload, repetition: bit 0 absent, bit 1 present.
    branches: [u8; 4],
}
impl OwnerInputTargetV1 {
    pub(in crate::implementations::continuous) fn from_input(
        input: &StructuredInputV2,
    ) -> Result<Self> {
        if input.physical_domain.is_none() {
            return Err(StructuredUnknown::MissingEvidence);
        }
        let mut out = Self {
            positive: vec![false; input.basis.len()],
            branches: [0; 4],
        };
        out.observe(input)?;
        Ok(out)
    }
    pub(in crate::implementations::continuous) fn from_samples(
        samples: &[StructuredNumericObservationV2],
    ) -> Result<Self> {
        let first = samples
            .first()
            .ok_or(StructuredUnknown::InsufficientSamples)?;
        let mut out = Self::from_input(&first.input)?;
        for s in &samples[1..] {
            out.observe(&s.input)?;
        }
        Ok(out)
    }
    pub(in crate::implementations::continuous) fn observe(
        &mut self,
        input: &StructuredInputV2,
    ) -> Result<()> {
        if input.physical_domain.is_none() || input.basis.len() != self.positive.len() {
            return Err(StructuredUnknown::WrongDomain);
        }
        for (i, (&x, seen)) in input.basis.iter().zip(&mut self.positive).enumerate() {
            if !x.is_finite() || x < 0. || x > (1u64 << 53) as f64 || x.fract() != 0. {
                return Err(StructuredUnknown::InvalidInput);
            }
            if !input
                .completion
                .as_ref()
                .is_some_and(|c| (c.basis_offset..c.basis_offset + 3).contains(&i))
            {
                *seen |= x != 0.;
            }
        }
        for (slot, present) in self
            .branches
            .iter_mut()
            .zip(envelope::input_branches(input))
        {
            if let Some(present) = present {
                *slot |= 1 << usize::from(present);
            }
        }
        Ok(())
    }
    pub(in crate::implementations::continuous) fn project_algorithm_universe(
        &mut self,
        input: &StructuredInputV2,
        universe: &DeclaredAlgorithmUniverseV1,
    ) -> Result<()> {
        self.positive = universe.project_basis_flags(input, &self.positive)?;
        Ok(())
    }
    pub(in crate::implementations::continuous) fn merge(&mut self, other: Self) -> Result<()> {
        if self.positive.len() != other.positive.len() {
            return Err(StructuredUnknown::WrongDomain);
        }
        for (value, other) in self.positive.iter_mut().zip(other.positive) {
            *value |= other;
        }
        for (value, other) in self.branches.iter_mut().zip(other.branches) {
            *value |= other;
        }
        Ok(())
    }
    pub(super) fn validate(&self, settings: &StructuredSettingsV2) -> Result<()> {
        if self.positive.is_empty()
            || self.positive.len() > settings.max_axes
            || self.branches.iter().any(|&b| b > 3)
            || self.branches[..3].contains(&0)
        {
            return Err(StructuredUnknown::InvalidInput);
        }
        Ok(())
    }
    fn covers(&self, target: &Self) -> bool {
        self.positive.len() == target.positive.len()
            && self
                .positive
                .iter()
                .zip(&target.positive)
                .all(|(&s, &t)| !t || s)
            && self
                .branches
                .iter()
                .zip(target.branches)
                .all(|(&s, t)| s & t == t)
    }
    pub(in crate::implementations::continuous) fn retained_heap_bytes(&self) -> usize {
        self.positive.capacity()
    }
    pub(in crate::implementations::continuous) fn bind(&self, h: &mut Sha256) {
        h.update(b"ferrum.owner-input-target.v1\0");
        h.update((self.positive.len() as u64).to_le_bytes());
        for &v in &self.positive {
            h.update([u8::from(v)]);
        }
        h.update(self.branches);
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OwnerInputReadinessGapV1 {
    MissingInputCoverage,
    InsufficientInputGeometry,
    GeometryWorkBudget,
    CompleteBlockLimit,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OwnerInputReadinessDecisionV1 {
    Wait,
    Freeze,
    Exhausted(OwnerInputReadinessGapV1),
}

impl OwnerBlockScheduleV1 {
    /// Conservative simultaneous scratch for input conversion, normalized QR
    /// rows/directions/pivots and vector headers. V2 also owns one n*d residual
    /// cache and n cached norms; these fit inside the same 64*n*d allowance
    /// (at most n directions). V3 additionally owns at most n*d compact scalars,
    /// n compact Vec headers and n borrowed FitRow headers, plus d zero flags
    /// and d indices. Together with the original converted rows, V2 cache,
    /// at most n directions and temporary pivot, these remain within the same
    /// 64*n*d + 128*n + 64*d bound. No observation or receipt is duplicated.
    /// Reserved sample workspace is
    /// shared with fitting: the readiness scratch is dropped before fitting.
    pub(in crate::implementations::continuous) fn readiness_scratch_bytes(
        &self,
        samples: &[StructuredNumericObservationV2],
    ) -> Result<usize> {
        let Some(first) = samples.first() else {
            return Ok(0);
        };
        let n = samples.len();
        let d = first.input.basis.len();
        n.checked_mul(d)
            .and_then(|v| v.checked_mul(64))
            .and_then(|v| v.checked_add(n.checked_mul(128)?))
            .and_then(|v| v.checked_add(d.checked_mul(64)?))
            .ok_or(StructuredUnknown::Capacity)
    }
    /// The caller supplies all original members of this phase, in ticket order.
    /// `visits` persists until the phase closes, including unsuccessful checks.
    pub(in crate::implementations::continuous) fn assess_inputs(
        &self,
        phase: StructuredPhaseV2,
        actual_offered: u64,
        samples: &[StructuredNumericObservationV2],
        target: Option<&OwnerInputTargetV1>,
        settings: &StructuredSettingsV2,
        visits: &mut u64,
    ) -> Result<OwnerInputReadinessDecisionV1> {
        use OwnerInputReadinessDecisionV1::*;
        let counts_ready = self.is_ready(phase, actual_offered, samples.len())?;
        let Some(policy) = &self.input_readiness else {
            return Ok(if counts_ready { Freeze } else { Wait });
        };
        let visit_limit = policy.pass_visit_limit();
        let target = target.ok_or(StructuredUnknown::MissingEvidence)?;
        target.validate(settings)?;
        let blocks = actual_offered / self.block_offered as u64;
        let limit = policy.maximum_phase_blocks[phase.index()] as u64;
        if blocks > limit {
            return Err(StructuredUnknown::IncompletePhasePopulation);
        }
        let exhausted = |gap| {
            if blocks == limit {
                Exhausted(gap)
            } else {
                Wait
            }
        };
        if !counts_ready {
            return Ok(exhausted(OwnerInputReadinessGapV1::CompleteBlockLimit));
        }
        // Charge facts, target comparison and both integer/float input conversions
        // conservatively before allocating their bounded scratch.
        let dims = target.positive.len();
        if samples.iter().any(|s| s.input.basis.len() != dims) {
            return Err(StructuredUnknown::WrongDomain);
        }
        let scan = samples
            .len()
            .checked_mul(dims)
            .and_then(|v| v.checked_mul(4))
            .and_then(|v| u64::try_from(v).ok())
            .ok_or(StructuredUnknown::Capacity)?;
        let Some(next) = visits.checked_add(scan).filter(|&v| v <= visit_limit) else {
            return Ok(Exhausted(OwnerInputReadinessGapV1::GeometryWorkBudget));
        };
        *visits = next;
        let facts = OwnerInputTargetV1::from_samples(samples)?;
        // V1 Fit covers its independently declared Discovery target. V2 keeps
        // that target's structural identity but freezes Fit's own input support.
        // Both explicit policies admit later members only through preceding
        // frozen support. No duration or numerical fit result chooses members
        // or the close boundary; counts, geometry and replay cuts remain intact.
        let requires_target = match self.phase_support {
            None => true,
            Some(OwnerPhaseSupportPolicyV1::FrozenInputIntersectionV1) => {
                phase == StructuredPhaseV2::Fit
            }
            Some(OwnerPhaseSupportPolicyV1::FrozenInputIntersectionV2) => false,
        };
        // V4 adds no numerical or outcome-dependent stopping condition. The
        // caller and validate_close independently derive this target from Fit.
        let requires_target = requires_target
            || (policy.revision == InputReadinessRevision::WorkAxesAndBranchesV4
                && phase != StructuredPhaseV2::Fit);
        if requires_target && !facts.covers(target) {
            return Ok(exhausted(OwnerInputReadinessGapV1::MissingInputCoverage));
        }
        if phase == StructuredPhaseV2::Fit {
            // Restore pre-settlement upper work; never inspect settled EOS/Stop.
            let rows = samples
                .iter()
                .map(|s| {
                    envelope::membership_axes(&s.input)
                        .map(|v| v.into_iter().map(|x| x as f64).collect::<Vec<_>>())
                })
                .collect::<Result<Vec<_>>>()?;
            let borrowed = rows
                .iter()
                .map(|basis| super::super::super::fit::FitRow { basis, wall_ns: 0 })
                .collect::<Vec<_>>();
            let mut geometry_settings = settings.clone();
            // Only these three columns can change at settlement. Reserving all
            // three cannot weaken the original Fit redundancy requirement.
            if samples.iter().any(|s| {
                s.input
                    .physical_host_rows
                    .iter()
                    .any(envelope::early_capable)
            }) {
                geometry_settings.min_fit_redundancy = geometry_settings
                    .min_fit_redundancy
                    .checked_add(3)
                    .ok_or(StructuredUnknown::Capacity)?;
            }
            let mut work = super::super::super::fit::GeometryWork {
                used: *visits,
                limit: visit_limit,
                exhausted: false,
            };
            let result = match policy.revision {
                InputReadinessRevision::WorkAxesAndBranchesV1 => {
                    super::super::super::fit::input_geometry_with_work(
                        &borrowed,
                        &geometry_settings,
                        Some(&mut work),
                    )
                    .map(|geometry| geometry.basis.len())
                }
                InputReadinessRevision::WorkAxesAndBranchesV2 => {
                    super::super::super::fit::input_geometry_readiness_v2(
                        &borrowed,
                        &geometry_settings,
                        &mut work,
                    )
                }
                InputReadinessRevision::WorkAxesAndBranchesV3
                | InputReadinessRevision::WorkAxesAndBranchesV4 => {
                    super::super::super::fit::input_geometry_readiness_v3(
                        &borrowed,
                        &geometry_settings,
                        &mut work,
                    )
                }
            };
            *visits = work.used;
            if work.exhausted {
                return Ok(Exhausted(OwnerInputReadinessGapV1::GeometryWorkBudget));
            }
            match result {
                Ok(_) => {}
                Err(
                    StructuredUnknown::InsufficientRedundancy | StructuredUnknown::IllConditioned,
                ) => {
                    return Ok(exhausted(
                        OwnerInputReadinessGapV1::InsufficientInputGeometry,
                    ))
                }
                Err(error) => return Err(error),
            }
        }
        Ok(Freeze)
    }
}

#[cfg(test)]
mod revision_tests {
    use super::*;
    #[test]
    fn owner_input_readiness_v2_is_explicit_and_v1_wire_binding_is_preserved() {
        use sha2::{Digest, Sha256};
        let old = OwnerInputReadinessV1::new([3; 3], 32_000_000).unwrap();
        let new = OwnerInputReadinessV1::new_cached_residual_v2([3; 3], 32_000_000).unwrap();
        assert_eq!(
            serde_json::to_value(&old).unwrap(),
            serde_json::json!({
                "revision":"work_axes_and_branches_v1", "maximum_phase_blocks":[3,3,3], "maximum_geometry_visits":32000000
            })
        );
        let mut old_digest = Sha256::new();
        old.bind(&mut old_digest);
        let mut legacy = Sha256::new();
        legacy.update(b"ferrum.owner-input-readiness.v1\0");
        for _ in 0..3 {
            legacy.update(3u64.to_le_bytes());
        }
        legacy.update(32_000_000u64.to_le_bytes());
        let old_digest = old_digest.finalize();
        assert_eq!(old_digest, legacy.finalize());
        let mut new_digest = Sha256::new();
        new.bind(&mut new_digest);
        assert_ne!(old_digest, new_digest.finalize());
    }

    #[test]
    fn owner_input_readiness_v3_binds_compaction_without_changing_v2_allowance() {
        let old = OwnerInputReadinessV1::new_cached_residual_v2([3; 3], 32_000_000).unwrap();
        let new = OwnerInputReadinessV1::new_zero_column_v3([3; 3], 32_000_000).unwrap();
        assert_eq!(old.pass_visit_limit(), new.pass_visit_limit());
        assert_eq!(new.pass_visit_limit(), 16_000_000);
        let old_wire = serde_json::to_value(&old).unwrap();
        assert_eq!(old_wire["revision"], "work_axes_and_branches_v2");
        assert_eq!(
            serde_json::to_value(&new).unwrap()["revision"],
            "work_axes_and_branches_v3"
        );
        let mut old_hash = Sha256::new();
        old.bind(&mut old_hash);
        let mut new_hash = Sha256::new();
        new.bind(&mut new_hash);
        assert_ne!(old_hash.finalize(), new_hash.finalize());
        assert_eq!(
            OwnerBlockScheduleV1::new_with_input_readiness(256, [256; 3], [8; 3], old)
                .unwrap()
                .maximum_phase_members,
            OwnerBlockScheduleV1::new_with_input_readiness(256, [256; 3], [8; 3], new)
                .unwrap()
                .maximum_phase_members
        );
    }
}
