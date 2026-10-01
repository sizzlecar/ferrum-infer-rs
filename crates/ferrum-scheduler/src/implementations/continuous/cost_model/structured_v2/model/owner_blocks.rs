//! Pure numerical population commitments for independently progressing owners.
//! Source7 must verify every original offer and the block's frozen phase table.
//! These public DTOs cannot create a live receipt or execution authority.
use super::*;

mod readiness;
mod transitions;
pub use readiness::{
    OwnerInputReadinessDecisionV1, OwnerInputReadinessGapV1, OwnerInputReadinessV1,
    OwnerInputTargetV1,
};

/// Source7 may begin observing while an older, unticketed call is still in
/// flight. Only the first original offer can close that unoffered FIFO prefix.
/// Later offers retain strict per-request continuity within the same block.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OwnerOpeningFrontierPolicyV1 {
    FirstOfferFifoV1,
}

/// Source7 freezes one finite numeric subset from the earliest complete original
/// Discovery block containing a checked ordinary homogeneous decode input. Subsequent undeclared primitives remain original offers,
/// but cannot extend this generation's numerical population.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OwnerAlgorithmUniversePolicyV1 {
    FirstOrdinaryDiscoveryBlockSubsetV1,
    /// Header envelope carries a checked cold seed. The first ordinary discovery
    /// block freezes its union with actual checked classes before Fit starts.
    SeededFirstOrdinaryDiscoveryBlockSubsetV1,
}

/// Prediction lifetime is independently declared from the collection window.
/// None in an older schedule retains its original collection-deadline clamp.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OwnerPredictionValidityPolicyV1 {
    /// All three phases still close before the original attempt deadline.
    /// Predictions expire at the earliest original member timestamp plus its
    /// unchanged maximum sample age. Publication/import never renew this age.
    OriginalSampleAgeV1,
}

/// Input-only heldout membership. A later phase may narrow the published
/// structural domain; it cannot remove a failing eligible duration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OwnerPhaseSupportPolicyV1 {
    FrozenInputIntersectionV1,
    /// Discovery freezes algorithm identity and input structure, not a required
    /// measured support set. Fit stops at its first input-ready complete block
    /// and authorizes only its own identified work directions. Residual and
    /// Qualification retain the same independent frozen-input intersection.
    FrozenInputIntersectionV2,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OwnerBlockScheduleV1 {
    pub block_offered: usize,
    pub phase_min_offered: [usize; 3],
    pub min_members: [usize; 3],
    /// Capacity for the complete last block, never an observed denominator.
    pub maximum_phase_members: [usize; 3],
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_readiness: Option<OwnerInputReadinessV1>,
    /// None preserves the original processed-BlockOpen-cutoff-only contract.
    /// Serialized only when declared, so old source/profile bytes stay intact.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub opening_frontier: Option<OwnerOpeningFrontierPolicyV1>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub algorithm_universe: Option<OwnerAlgorithmUniversePolicyV1>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prediction_validity: Option<OwnerPredictionValidityPolicyV1>,
    /// None retains the original all-physical-members/full-Fit-coverage rule.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub phase_support: Option<OwnerPhaseSupportPolicyV1>,
}

impl OwnerBlockScheduleV1 {
    pub fn new(
        block_offered: usize,
        phase_min_offered: [usize; 3],
        min_members: [usize; 3],
    ) -> Result<Self> {
        let mut maximum_phase_members = [0; 3];
        for i in 0..3 {
            maximum_phase_members[i] =
                Self::derive_member_bound(block_offered, phase_min_offered[i], min_members[i])?;
        }
        Ok(Self {
            block_offered,
            phase_min_offered,
            min_members,
            maximum_phase_members,
            input_readiness: None,
            opening_frontier: None,
            algorithm_universe: None,
            prediction_validity: None,
            phase_support: None,
        })
    }

    pub fn new_with_input_readiness(
        block_offered: usize,
        phase_min_offered: [usize; 3],
        min_members: [usize; 3],
        policy: OwnerInputReadinessV1,
    ) -> Result<Self> {
        policy.validate()?;
        let mut out = Self::new(block_offered, phase_min_offered, min_members)?;
        for i in 0..3 {
            out.maximum_phase_members[i] = block_offered
                .checked_mul(policy.maximum_phase_blocks[i])
                .ok_or(StructuredUnknown::Capacity)?;
            if out.maximum_phase_members[i] < phase_min_offered[i].max(min_members[i]) {
                return Err(StructuredUnknown::InvalidSettings);
            }
        }
        out.input_readiness = Some(policy);
        Ok(out)
    }

    fn derive_member_bound(block: usize, offers: usize, members: usize) -> Result<usize> {
        if block == 0 || offers == 0 || members == 0 {
            return Err(StructuredUnknown::InvalidSettings);
        }
        let rounded = offers
            .div_ceil(block)
            .checked_mul(block)
            .ok_or(StructuredUnknown::Capacity)?;
        let last_block = block
            .checked_add(members - 1)
            .ok_or(StructuredUnknown::Capacity)?;
        Ok(rounded.max(last_block))
    }

    pub fn validate(&self, settings: &StructuredSettingsV2) -> Result<()> {
        settings.validate()?;
        if self.phase_support == Some(OwnerPhaseSupportPolicyV1::FrozenInputIntersectionV2)
            && self.input_readiness.is_none()
        {
            return Err(StructuredUnknown::WrongProtocol);
        }
        if self.block_offered == 0 || self.block_offered > 65_536 {
            return Err(StructuredUnknown::InvalidSettings);
        }
        if let Some(policy) = &self.input_readiness {
            policy.validate()?;
        }
        for i in 0..3 {
            let required = match &self.input_readiness {
                Some(policy) => self
                    .block_offered
                    .checked_mul(policy.maximum_phase_blocks[i])
                    .ok_or(StructuredUnknown::Capacity)?,
                None => Self::derive_member_bound(
                    self.block_offered,
                    self.phase_min_offered[i],
                    self.min_members[i],
                )?,
            };
            if self.phase_min_offered[i] < settings.min_phase_samples
                || self.phase_min_offered[i] > 65_536
                || self.min_members[i] < settings.min_phase_samples
                || self.maximum_phase_members[i] != required
                || required < self.phase_min_offered[i].max(self.min_members[i])
            {
                return Err(StructuredUnknown::InvalidSettings);
            }
            if required > settings.max_phase_samples {
                return Err(StructuredUnknown::Capacity);
            }
        }
        Ok(())
    }

    /// Input-only stopping rule. The collector calls this only at complete block
    /// boundaries. Neither measured cost nor numerical results are inputs.
    pub fn is_ready(
        &self,
        phase: StructuredPhaseV2,
        actual_offered: u64,
        eligible_members: usize,
    ) -> Result<bool> {
        let i = phase.index();
        if self.block_offered == 0
            || actual_offered % self.block_offered as u64 != 0
            || eligible_members as u64 > actual_offered
        {
            return Err(StructuredUnknown::IncompletePhasePopulation);
        }
        if eligible_members > self.maximum_phase_members[i] {
            return Err(StructuredUnknown::Capacity);
        }
        Ok(actual_offered >= self.phase_min_offered[i] as u64
            && eligible_members >= self.min_members[i])
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StructuredOwnerPhaseContractV1 {
    pub capture_identity: [u8; 32],
    pub protocol: [u8; 32],
    pub membership_rule: [u8; 32],
    pub declaration_sha256: [u8; 32],
    pub owner_attempt_id: u64,
    pub schedule: OwnerBlockScheduleV1,
    pub discovery_block: u64,
    pub discovery_offered_cutoff: u64,
    pub discovery_fifo_cutoff: u64,
    pub discovery_closed_at_ns: u64,
    /// Original collection-attempt deadline; later blocks cannot renew it.
    /// Old schedules also clamp prediction lifetime to this deadline. An
    /// explicit prediction_validity policy may retain original sample expiry.
    pub expires_at_ns: u64,
    pub domain_policy: StructuredServiceDomainPolicyV1,
    pub nonnegative_envelope: Option<NonNegativeEnvelopeContractV1>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_target: Option<OwnerInputTargetV1>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StructuredOwnerPhaseBoundaryV1 {
    pub first_block: u64,
    pub last_block: u64,
    pub first_offered: u64,
    pub last_offered: u64,
    pub opening_fifo_cutoff: u64,
    pub closing_fifo_cutoff: u64,
    pub opened_at_ns: u64,
    pub frozen_at_ns: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StructuredOwnerPhaseCloseV1 {
    pub phase: StructuredPhaseV2,
    pub boundary: StructuredOwnerPhaseBoundaryV1,
    /// Digest of the complete original block prefix, including nonmembers.
    pub population_sha256: [u8; 32],
    pub previous_parameters_sha256: Option<[u8; 32]>,
    pub member_count: usize,
    pub member_tickets_sha256: [u8; 32],
}

impl StructuredOwnerPhaseCloseV1 {
    pub fn new(
        phase: StructuredPhaseV2,
        boundary: StructuredOwnerPhaseBoundaryV1,
        population_sha256: [u8; 32],
        previous_parameters_sha256: Option<[u8; 32]>,
        samples: &[StructuredNumericObservationV2],
    ) -> Self {
        Self {
            phase,
            boundary,
            population_sha256,
            previous_parameters_sha256,
            member_count: samples.len(),
            member_tickets_sha256: member_tickets_signature(samples),
        }
    }

    pub(super) fn signature(&self) -> [u8; 32] {
        let mut digest = Sha256::new();
        digest.update(b"ferrum.structured-owner-block-phase-close.v1\0");
        digest.update([self.phase.index() as u8]);
        for value in [
            self.boundary.first_block,
            self.boundary.last_block,
            self.boundary.first_offered,
            self.boundary.last_offered,
            self.boundary.opening_fifo_cutoff,
            self.boundary.closing_fifo_cutoff,
            self.boundary.opened_at_ns,
            self.boundary.frozen_at_ns,
            self.member_count as u64,
        ] {
            digest.update(value.to_le_bytes());
        }
        digest.update(self.population_sha256);
        digest.update(self.member_tickets_sha256);
        match self.previous_parameters_sha256 {
            Some(value) => {
                digest.update([1]);
                digest.update(value);
            }
            None => digest.update([0]),
        }
        digest.finalize().into()
    }
}

fn member_tickets_signature(samples: &[StructuredNumericObservationV2]) -> [u8; 32] {
    let mut digest = Sha256::new();
    digest.update(b"ferrum.structured-owner-block-members.v1\0");
    digest.update((samples.len() as u64).to_le_bytes());
    for sample in samples {
        digest.update(sample.membership.offered_ordinal.to_le_bytes());
    }
    digest.finalize().into()
}

pub(super) struct Population {
    pub(super) contract: StructuredOwnerPhaseContractV1,
    pub(super) closes: [Option<StructuredOwnerPhaseCloseV1>; 3],
    pub(super) fit_signature: [u8; 32],
    pub(super) calibrated_signature: [u8; 32],
    pub(super) input_target: Option<OwnerInputTargetV1>,
}

impl StructuredOwnerPhaseContractV1 {
    pub fn validate(&self, settings: &StructuredSettingsV2) -> Result<()> {
        self.schedule.validate(settings)?;
        if self.schedule.phase_support.is_some()
            && (self.domain_policy
                != StructuredServiceDomainPolicyV1::NonNegativePhysicalEnvelopeV1
                || self.nonnegative_envelope.is_none())
        {
            return Err(StructuredUnknown::WrongProtocol);
        }
        if self.schedule.algorithm_universe.is_some()
            && (self.domain_policy
                != StructuredServiceDomainPolicyV1::NonNegativePhysicalEnvelopeV1
                || self.nonnegative_envelope.as_ref().is_none_or(|c| {
                    c.population_policy != StructuredPopulationPolicyV1::HomogeneousOrdinaryDecodeV1
                        || c.template_policy
                            != StructuredCostTemplatePolicyV1::InstalledAlgorithmSetV1
                }))
        {
            return Err(StructuredUnknown::WrongProtocol);
        }
        match (&self.schedule.input_readiness, &self.input_target) {
            (Some(_), Some(target))
                if self.domain_policy
                    == StructuredServiceDomainPolicyV1::NonNegativePhysicalEnvelopeV1 =>
            {
                target.validate(settings)?
            }
            (None, None) => {}
            _ => return Err(StructuredUnknown::WrongProtocol),
        }
        if [
            self.capture_identity,
            self.protocol,
            self.membership_rule,
            self.declaration_sha256,
        ]
        .contains(&[0; 32])
            || self.owner_attempt_id == 0
            || self.discovery_block == 0
            || self.discovery_offered_cutoff == 0
            || self.discovery_fifo_cutoff == 0
            || self.expires_at_ns <= self.discovery_closed_at_ns
            || self.expires_at_ns - self.discovery_closed_at_ns > settings.max_sample_age_ns
        {
            return Err(StructuredUnknown::InvalidSettings);
        }
        match (self.domain_policy, &self.nonnegative_envelope) {
            (StructuredServiceDomainPolicyV1::NonNegativePhysicalEnvelopeV1, Some(c)) => {
                c.validate()
            }
            (StructuredServiceDomainPolicyV1::NonNegativePhysicalEnvelopeV1, None)
            | (_, Some(_)) => Err(StructuredUnknown::WrongProtocol),
            (_, None) => Ok(()),
        }
    }

    /// Numerical strategy is part of the original contract, including the
    /// input-only complete-block stopping rule. Legacy policies are unchanged.
    pub(in crate::implementations::continuous) fn assess_inputs(
        &self,
        phase: StructuredPhaseV2,
        actual_offered: u64,
        samples: &[StructuredNumericObservationV2],
        target: Option<&OwnerInputTargetV1>,
        settings: &StructuredSettingsV2,
        visits: &mut u64,
    ) -> Result<OwnerInputReadinessDecisionV1> {
        if let Some(contract) = self
            .nonnegative_envelope
            .as_ref()
            .filter(|c| joint_cells::enabled(c))
        {
            if phase != StructuredPhaseV2::Fit && self.schedule.input_readiness.is_some() {
                return joint_cells::assess_inputs(
                    &self.schedule,
                    contract,
                    phase,
                    actual_offered,
                    samples,
                    settings,
                    visits,
                );
            }
        }
        self.schedule
            .assess_inputs(phase, actual_offered, samples, target, settings, visits)
    }
    fn validate_close(
        &self,
        settings: &StructuredSettingsV2,
        close: &StructuredOwnerPhaseCloseV1,
        samples: &[StructuredNumericObservationV2],
        phase: StructuredPhaseV2,
        previous: Option<(&StructuredOwnerPhaseCloseV1, [u8; 32])>,
        input_target: Option<&OwnerInputTargetV1>,
    ) -> Result<()> {
        self.validate(settings)?;
        if close.phase != phase {
            return Err(StructuredUnknown::PhaseLeakage);
        }
        if close.population_sha256 == [0; 32]
            || close.member_count != samples.len()
            || close.member_tickets_sha256 != member_tickets_signature(samples)
        {
            return Err(StructuredUnknown::IncompletePhasePopulation);
        }
        let (last_block, last_offered, last_fifo, last_time, signature) = match previous {
            Some((old, signature)) if old.phase.index() + 1 == phase.index() => (
                old.boundary.last_block,
                old.boundary.last_offered,
                old.boundary.closing_fifo_cutoff,
                old.boundary.frozen_at_ns,
                Some(signature),
            ),
            None if phase == StructuredPhaseV2::Fit => (
                self.discovery_block,
                self.discovery_offered_cutoff,
                self.discovery_fifo_cutoff,
                self.discovery_closed_at_ns,
                None,
            ),
            _ => return Err(StructuredUnknown::PhaseLeakage),
        };
        if self.schedule.input_readiness.is_some()
            && samples
                .windows(2)
                .any(|w| w[0].membership.offered_ordinal >= w[1].membership.offered_ordinal)
        {
            return Err(StructuredUnknown::PhaseLeakage);
        }
        let b = close.boundary;
        if close.previous_parameters_sha256 != signature
            || last_block.checked_add(1) != Some(b.first_block)
            || last_offered.checked_add(1) != Some(b.first_offered)
            || b.opening_fifo_cutoff < last_fifo
        {
            return Err(StructuredUnknown::PhaseLeakage);
        }
        if b.opened_at_ns < last_time || b.frozen_at_ns < b.opened_at_ns {
            return Err(StructuredUnknown::Clock);
        }
        if b.frozen_at_ns > self.expires_at_ns {
            return Err(StructuredUnknown::Stale);
        }
        let block_count = b
            .last_block
            .checked_sub(b.first_block)
            .and_then(|n| n.checked_add(1))
            .ok_or(StructuredUnknown::IncompletePhasePopulation)?;
        let offered = b
            .last_offered
            .checked_sub(b.first_offered)
            .and_then(|n| n.checked_add(1))
            .ok_or(StructuredUnknown::IncompletePhasePopulation)?;
        let expected = block_count
            .checked_mul(self.schedule.block_offered as u64)
            .ok_or(StructuredUnknown::Capacity)?;
        if offered != expected
            || b.closing_fifo_cutoff
                .checked_sub(b.opening_fifo_cutoff)
                .is_none_or(|n| n < offered)
            || !self.schedule.is_ready(phase, offered, samples.len())?
        {
            return Err(StructuredUnknown::IncompletePhasePopulation);
        }
        if self.schedule.input_readiness.is_some() {
            // Recompute every complete prefix: normalized numerical rank is
            // not assumed monotone as new coordinate scales arrive.
            let mut visits = 0;
            for block in 1..=block_count {
                let prefix_offered = block * self.schedule.block_offered as u64;
                let cutoff = b.first_offered + prefix_offered;
                let count = samples.partition_point(|s| s.membership.offered_ordinal < cutoff);
                let decision = self.assess_inputs(
                    phase,
                    prefix_offered,
                    &samples[..count],
                    input_target,
                    settings,
                    &mut visits,
                )?;
                if block < block_count && decision != OwnerInputReadinessDecisionV1::Wait {
                    return Err(StructuredUnknown::PhaseLeakage);
                }
                if block == block_count && decision != OwnerInputReadinessDecisionV1::Freeze {
                    return Err(StructuredUnknown::IncompletePhasePopulation);
                }
            }
        } else {
            let before_last = offered - self.schedule.block_offered as u64;
            let last_block_first = b.first_offered + before_last;
            let previous_members = samples
                .iter()
                .filter(|s| s.membership.offered_ordinal < last_block_first)
                .count();
            if self
                .schedule
                .is_ready(phase, before_last, previous_members)?
            {
                return Err(StructuredUnknown::PhaseLeakage);
            }
        }
        for s in samples {
            if s.membership.offered_ordinal < b.first_offered
                || s.membership.offered_ordinal > b.last_offered
                || s.ordinal <= b.opening_fifo_cutoff
                || s.ordinal > b.closing_fifo_cutoff
            {
                return Err(StructuredUnknown::PhaseLeakage);
            }
            if s.observed_at_ns < b.opened_at_ns || s.observed_at_ns > b.frozen_at_ns {
                return Err(StructuredUnknown::Clock);
            }
        }
        Ok(())
    }

    fn bind(&self, digest: &mut Sha256) {
        digest.update(b"ferrum.structured-owner-block-contract.v1\0");
        for value in [
            self.capture_identity,
            self.protocol,
            self.membership_rule,
            self.declaration_sha256,
        ] {
            digest.update(value);
        }
        for value in [
            self.owner_attempt_id,
            self.schedule.block_offered as u64,
            self.discovery_block,
            self.discovery_offered_cutoff,
            self.discovery_fifo_cutoff,
            self.discovery_closed_at_ns,
            self.expires_at_ns,
        ] {
            digest.update(value.to_le_bytes());
        }
        for values in [
            self.schedule.phase_min_offered,
            self.schedule.min_members,
            self.schedule.maximum_phase_members,
        ] {
            for value in values {
                digest.update((value as u64).to_le_bytes());
            }
        }
        digest.update([match self.domain_policy {
            StructuredServiceDomainPolicyV1::AllOffered => 0,
            StructuredServiceDomainPolicyV1::FrozenFitSupportV1 => 1,
            StructuredServiceDomainPolicyV1::NonNegativePhysicalEnvelopeV1 => 2,
        }]);
        if let Some(envelope) = &self.nonnegative_envelope {
            envelope.bind(digest);
        }
        if let Some(policy) = &self.schedule.input_readiness {
            policy.bind(digest);
        }
        if let Some(target) = &self.input_target {
            target.bind(digest);
        }
        if let Some(OwnerAlgorithmUniversePolicyV1::FirstOrdinaryDiscoveryBlockSubsetV1) =
            self.schedule.algorithm_universe
        {
            digest.update(
                b"ferrum.owner-algorithm-universe.first-ordinary-discovery-block-subset.v1\0",
            );
        }
        if let Some(OwnerAlgorithmUniversePolicyV1::SeededFirstOrdinaryDiscoveryBlockSubsetV1) =
            self.schedule.algorithm_universe
        {
            digest.update(b"ferrum.owner-algorithm-universe.seeded-first-discovery.v1\0");
        }
        if let Some(OwnerOpeningFrontierPolicyV1::FirstOfferFifoV1) = self.schedule.opening_frontier
        {
            digest.update(b"ferrum.owner-opening-frontier.first-offer-fifo.v1\0");
        }
        if let Some(OwnerPredictionValidityPolicyV1::OriginalSampleAgeV1) =
            self.schedule.prediction_validity
        {
            digest.update(b"ferrum.owner-prediction-validity.original-sample-age.v1\0");
        }
        if let Some(OwnerPhaseSupportPolicyV1::FrozenInputIntersectionV1) =
            self.schedule.phase_support
        {
            digest.update(b"ferrum.owner-phase-support.frozen-input-intersection.v1\0");
        }
        if let Some(OwnerPhaseSupportPolicyV1::FrozenInputIntersectionV2) =
            self.schedule.phase_support
        {
            digest.update(b"ferrum.owner-phase-support.frozen-input-intersection.v2\0");
        }
    }
}
