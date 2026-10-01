//! Explicit empirical authorization, not one model or population per count.
use super::*;
use std::collections::BTreeSet;
/// Auditable numerical coverage only; this is not a source or live receipt.
#[derive(Debug, Clone, Default, Serialize)]
pub struct StructuredCoverageFactsV2 {
    pub pending_counts: Vec<u32>,
    pub length_counts: Vec<u32>,
    pub pending_positions: Vec<u32>,
    pub length_positions: Vec<u32>,
    pub joint_counts: Vec<(u32, u32)>,
}
#[derive(Debug, Clone, Serialize)]
pub struct StructuredCoverageReportV2 {
    pub observed: StructuredCoverageFactsV2,
    pub missing: StructuredCoverageFactsV2,
    pub empty_challenge_seen: bool,
    pub full_challenge_seen: bool,
    pub intermediate_challenge_seen: bool,
    pub missing_empty_challenge: bool,
    pub missing_full_challenge: bool,
    pub missing_intermediate_challenge: bool,
}
impl StructuredCoverageReportV2 {
    pub fn complete(&self) -> bool {
        self.missing.pending_counts.is_empty()
            && self.missing.length_counts.is_empty()
            && self.missing.pending_positions.is_empty()
            && self.missing.length_positions.is_empty()
            && self.missing.joint_counts.is_empty()
            && !self.missing_empty_challenge
            && !self.missing_full_challenge
            && !self.missing_intermediate_challenge
    }
}
impl StructuredScopeV2 {
    /// Numerical membership only. This never changes the input's original
    /// owner, row count, recipe binding, or physical state.
    pub fn validate_population_input(
        &self,
        input: &StructuredInputV2,
        policy: StructuredPopulationPolicyV1,
    ) -> Result<()> {
        match (&self.numerical_family, policy) {
            (Some(expected), StructuredPopulationPolicyV1::HomogeneousOrdinaryDecodeV1) => {
                if input.numerical_family_key()? != *expected {
                    return Err(StructuredUnknown::WrongDomain);
                }
            }
            (Some(_), StructuredPopulationPolicyV1::ExactOwnerV1) => {
                return Err(StructuredUnknown::WrongProtocol);
            }
            (None, _) => {
                if input.owner() != &self.owner {
                    return Err(StructuredUnknown::WrongDomain);
                }
            }
        }
        Ok(())
    }

    /// Collector lookup preserves invalid protocol/evidence errors. A different
    /// valid statistical identity simply belongs to another population.
    pub fn has_same_population(
        &self,
        input: &StructuredInputV2,
        policy: StructuredPopulationPolicyV1,
    ) -> Result<bool> {
        match (&self.numerical_family, policy) {
            (Some(expected), StructuredPopulationPolicyV1::HomogeneousOrdinaryDecodeV1) => {
                match input.numerical_family_key() {
                    Ok(actual) => Ok(actual == *expected),
                    Err(StructuredUnknown::UnsupportedScope) => Ok(false),
                    Err(error) => Err(error),
                }
            }
            (Some(_), StructuredPopulationPolicyV1::ExactOwnerV1) => {
                Err(StructuredUnknown::WrongProtocol)
            }
            (None, _) => Ok(input.owner() == &self.owner),
        }
    }

    pub(in crate::implementations::continuous) fn retained_heap_bytes(&self) -> Option<usize> {
        let c = &self.coverage;
        let mut bytes = 0usize;
        for values in [
            &c.pending_eligible_positions,
            &c.pending_counts,
            &c.length_counts,
            &c.pending_positions,
            &c.length_positions,
        ] {
            bytes =
                bytes.checked_add(values.capacity().checked_mul(std::mem::size_of::<u32>())?)?;
        }
        bytes
            .checked_add(
                c.authorized_pending_constraints
                    .capacity()
                    .checked_mul(std::mem::size_of::<HostPendingConstraintV2>())?,
            )?
            .checked_add(
                c.joint_counts
                    .capacity()
                    .checked_mul(std::mem::size_of::<(u32, u32)>())?,
            )
    }
    pub fn validate(&self) -> Result<()> {
        let n = self.owner.rows;
        if n == 0 || n > 128 {
            return Err(StructuredUnknown::InvalidSettings);
        }
        let c = &self.coverage;
        for positions in [
            &c.pending_eligible_positions,
            &c.pending_positions,
            &c.length_positions,
        ] {
            if positions.len() > n as usize
                || positions.iter().any(|p| *p >= n)
                || positions.windows(2).any(|p| p[0] >= p[1])
            {
                return Err(StructuredUnknown::InvalidSettings);
            }
        }
        for counts in [&c.pending_counts, &c.length_counts] {
            if counts.is_empty()
                || counts.len() > n as usize + 1
                || counts.iter().any(|p| *p > n)
                || counts.windows(2).any(|p| p[0] >= p[1])
            {
                return Err(StructuredUnknown::InvalidSettings);
            }
        }
        if c.authorized_pending_constraints.len() > 2
            || (c.authorized_pending_constraints.len() == 2
                && c.authorized_pending_constraints[0] == c.authorized_pending_constraints[1])
            || c.joint_counts.is_empty()
            || c.joint_counts.len() > (n as usize + 1) * (n as usize + 1)
            || c.joint_counts.iter().any(|(p, l)| *p > n || *l > n)
            || c.joint_counts.iter().any(|(p, l)| {
                c.pending_counts.binary_search(p).is_err()
                    || c.length_counts.binary_search(l).is_err()
            })
            || c.joint_counts.windows(2).any(|p| p[0] >= p[1])
        {
            return Err(StructuredUnknown::InvalidSettings);
        }
        if !c.authorized_pending_constraints.is_empty() {
            for p in &c.pending_eligible_positions {
                if c.pending_positions.binary_search(p).is_err() {
                    return Err(StructuredUnknown::InvalidSettings);
                }
            }
        }
        if c.authorized_pending_constraints
            .contains(&HostPendingConstraintV2::NonEmptySubset)
            && c.pending_eligible_positions.is_empty()
        {
            return Err(StructuredUnknown::InvalidSettings);
        }
        Ok(())
    }
    pub(super) fn authorize(&self, query: &StructuredQueryV2) -> Result<()> {
        self.authorize_detailed(query)
            .map_err(StructuredQueryFailureV2::reason)
    }
    pub(super) fn authorize_detailed(&self, query: &StructuredQueryV2) -> QueryResult<()> {
        if self.numerical_family.is_some() {
            // Cross-width authorization belongs to the declared physical
            // envelope's original phase challenges, never this representative.
            return Err(StructuredUnknown::WrongProtocol.into());
        }
        if query
            .input
            .cost_template_identity(self.owner.cost_template_policy())
            .is_none_or(|(owner, _)| owner != &self.owner)
        {
            return Err(StructuredUnknown::WrongDomain.into());
        }
        let c = &self.coverage;
        if c.length_counts
            .binary_search(&(query.input.length_positions.len() as u32))
            .is_err()
        {
            return Err(StructuredQueryFailureV2::OutsideSupport(
                StructuredUnknown::QualificationCoverage,
            ));
        }
        if query
            .input
            .length_positions
            .iter()
            .any(|p| c.length_positions.binary_search(p).is_err())
            || query
                .input
                .pending_positions
                .iter()
                .any(|p| c.pending_positions.binary_search(p).is_err())
        {
            return Err(StructuredQueryFailureV2::OutsideSupport(
                StructuredUnknown::QualificationCoverage,
            ));
        }
        if let Some(p) = &query.pending {
            if !c.authorized_pending_constraints.contains(&p.constraint)
                || p.eligible
                    .iter()
                    .any(|v| c.pending_eligible_positions.binary_search(v).is_err())
            {
                return Err(StructuredQueryFailureV2::OutsideSupport(
                    StructuredUnknown::UnsupportedScope,
                ));
            }
        }
        let (minimum, maximum) = query.pending_count_range()?;
        if minimum > maximum
            || (minimum..=maximum).any(|n| {
                c.pending_counts.binary_search(&(n as u32)).is_err()
                    || c.joint_counts
                        .binary_search(&(n as u32, query.input.length_positions.len() as u32))
                        .is_err()
            })
        {
            return Err(StructuredQueryFailureV2::OutsideSupport(
                StructuredUnknown::QualificationCoverage,
            ));
        }
        Ok(())
    }
    pub(super) fn qualify_coverage(
        &self,
        samples: &[StructuredNumericObservationV2],
    ) -> Result<()> {
        if !self.coverage_report(samples)?.complete() {
            return Err(StructuredUnknown::QualificationCoverage);
        }
        Ok(())
    }
    /// Partial population reports are allowed for diagnostics. Only the model
    /// transitions verify complete source population, phases, time and outcome.
    pub fn coverage_report(
        &self,
        samples: &[StructuredNumericObservationV2],
    ) -> Result<StructuredCoverageReportV2> {
        if self.numerical_family.is_some() {
            return Err(StructuredUnknown::WrongProtocol);
        }
        self.validate()?;
        if samples.len() > 4096 {
            return Err(StructuredUnknown::Capacity);
        }
        let c = &self.coverage;
        let mut pending_counts = BTreeSet::new();
        let mut length_counts = BTreeSet::new();
        let mut pending_positions = BTreeSet::new();
        let mut length_positions = BTreeSet::new();
        let mut joint = BTreeSet::new();
        let mut empty = false;
        let mut full = false;
        let mut intermediate = false;
        for s in samples {
            if s.input.owner() != &self.owner {
                return Err(StructuredUnknown::WrongDomain);
            }
            s.input.validate(&StructuredSettingsV2 {
                max_axes: 4096,
                ..Default::default()
            })?;
            let p = &s.input.pending_positions;
            let l = &s.input.length_positions;
            pending_counts.insert(p.len() as u32);
            length_counts.insert(l.len() as u32);
            pending_positions.extend(p.iter().copied());
            length_positions.extend(l.iter().copied());
            joint.insert((p.len() as u32, l.len() as u32));
            let eligible_count = p
                .iter()
                .filter(|p| c.pending_eligible_positions.binary_search(p).is_ok())
                .count();
            empty |= eligible_count == 0;
            full |= eligible_count == c.pending_eligible_positions.len();
            intermediate |=
                eligible_count > 0 && eligible_count < c.pending_eligible_positions.len();
        }
        let missing = StructuredCoverageFactsV2 {
            pending_counts: c
                .pending_counts
                .iter()
                .filter(|v| !pending_counts.contains(v))
                .copied()
                .collect(),
            length_counts: c
                .length_counts
                .iter()
                .filter(|v| !length_counts.contains(v))
                .copied()
                .collect(),
            pending_positions: c
                .pending_positions
                .iter()
                .filter(|v| !pending_positions.contains(v))
                .copied()
                .collect(),
            length_positions: c
                .length_positions
                .iter()
                .filter(|v| !length_positions.contains(v))
                .copied()
                .collect(),
            joint_counts: c
                .joint_counts
                .iter()
                .filter(|v| !joint.contains(v))
                .copied()
                .collect(),
        };
        let unresolved = !c.authorized_pending_constraints.is_empty();
        Ok(StructuredCoverageReportV2 {
            observed: StructuredCoverageFactsV2 {
                pending_counts: pending_counts.into_iter().collect(),
                length_counts: length_counts.into_iter().collect(),
                pending_positions: pending_positions.into_iter().collect(),
                length_positions: length_positions.into_iter().collect(),
                joint_counts: joint.into_iter().collect(),
            },
            missing,
            empty_challenge_seen: empty,
            full_challenge_seen: full,
            intermediate_challenge_seen: intermediate,
            missing_empty_challenge: c
                .authorized_pending_constraints
                .contains(&HostPendingConstraintV2::AnySubset)
                && !empty,
            missing_full_challenge: unresolved && !full,
            missing_intermediate_challenge: unresolved
                && c.pending_eligible_positions.len() > 1
                && !intermediate,
        })
    }
    pub(super) fn bind_parameters(&self, digest: &mut sha2::Sha256) {
        use sha2::Digest;
        // The owner's full value is bound by the input domain signature.
        let c = &self.coverage;
        for values in [
            &c.pending_eligible_positions,
            &c.pending_counts,
            &c.length_counts,
            &c.pending_positions,
            &c.length_positions,
        ] {
            digest.update((values.len() as u64).to_le_bytes());
            for v in values {
                digest.update(v.to_le_bytes());
            }
        }
        digest.update((c.authorized_pending_constraints.len() as u64).to_le_bytes());
        for constraint in &c.authorized_pending_constraints {
            digest.update([match constraint {
                HostPendingConstraintV2::AnySubset => 0,
                HostPendingConstraintV2::NonEmptySubset => 1,
            }]);
        }
        digest.update((c.joint_counts.len() as u64).to_le_bytes());
        for (p, l) in &c.joint_counts {
            digest.update(p.to_le_bytes());
            digest.update(l.to_le_bytes());
        }
    }
}
