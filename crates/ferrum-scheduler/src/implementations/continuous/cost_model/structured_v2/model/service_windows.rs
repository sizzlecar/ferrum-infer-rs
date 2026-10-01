//! Numerical entry points for fixed *offered* service windows. These DTOs do
//! not attest a live wave. Source6 replay must verify every ticket, including
//! nonmembers, before supplying a close. Source3/4/5 use the original contract.
use super::*;

/// Input-only membership of a fully retained service window. Omitted policy
/// preserves the original all-owner population and its serialized identity.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StructuredServiceDomainPolicyV1 {
    #[default]
    AllOffered,
    FrozenFitSupportV1,
    NonNegativePhysicalEnvelopeV1,
}
impl StructuredServiceDomainPolicyV1 {
    pub fn is_all_offered(&self) -> bool {
        *self == Self::AllOffered
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StructuredServiceInputMembershipV1 {
    Eligible,
    OutsideFitSupport,
    OutsideResidualSupport,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StructuredServiceWindowContractV2 {
    pub capture_identity: [u8; 32],
    pub protocol: [u8; 32],
    pub membership_rule: [u8; 32],
    /// Exact immutable declaration, including predicates and all three windows.
    pub window_declaration: [u8; 32],
    pub phase_offered: [usize; 3],
    #[serde(
        default,
        skip_serializing_if = "StructuredServiceDomainPolicyV1::is_all_offered"
    )]
    pub domain_policy: StructuredServiceDomainPolicyV1,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub nonnegative_envelope: Option<NonNegativeEnvelopeContractV1>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StructuredServiceWindowCloseV2 {
    pub phase: StructuredPhaseV2,
    /// Digest of the complete offered window, not just successfully fit rows.
    pub population_sha256: [u8; 32],
    pub member_count: usize,
    /// Ordered original ticket ordinals, derived from the frozen owner predicate.
    pub member_tickets_sha256: [u8; 32],
}
impl StructuredServiceWindowCloseV2 {
    /// Pure numerical commitment. A caller cannot obtain live/profile authority
    /// without the source6 complete-population and private receipt validation.
    pub fn new(
        phase: StructuredPhaseV2,
        population_sha256: [u8; 32],
        samples: &[StructuredNumericObservationV2],
    ) -> Self {
        Self {
            phase,
            population_sha256,
            member_count: samples.len(),
            member_tickets_sha256: member_tickets_signature(samples),
        }
    }
    pub(super) fn signature(&self) -> [u8; 32] {
        let mut digest = Sha256::new();
        digest.update(b"ferrum.structured-service-window-close.v1\0");
        digest.update([self.phase.index() as u8]);
        digest.update(self.population_sha256);
        digest.update((self.member_count as u64).to_le_bytes());
        digest.update(self.member_tickets_sha256);
        digest.finalize().into()
    }
}
fn member_tickets_signature(samples: &[StructuredNumericObservationV2]) -> [u8; 32] {
    let mut digest = Sha256::new();
    digest.update(b"ferrum.structured-service-window-members.v1\0");
    digest.update((samples.len() as u64).to_le_bytes());
    for sample in samples {
        digest.update(sample.membership.offered_ordinal.to_le_bytes());
    }
    digest.finalize().into()
}

pub(super) struct Population {
    pub(super) contract: StructuredServiceWindowContractV2,
    pub(super) closes: [Option<StructuredServiceWindowCloseV2>; 3],
    pub(super) fit_signature: [u8; 32],
    pub(super) calibrated_signature: [u8; 32],
}
impl StructuredServiceWindowContractV2 {
    fn validate(&self, settings: &StructuredSettingsV2) -> Result<()> {
        match (self.domain_policy, &self.nonnegative_envelope) {
            (StructuredServiceDomainPolicyV1::NonNegativePhysicalEnvelopeV1, Some(contract)) => {
                contract.validate()?
            }
            (StructuredServiceDomainPolicyV1::NonNegativePhysicalEnvelopeV1, None)
            | (_, Some(_)) => return Err(StructuredUnknown::WrongProtocol),
            (_, None) => {}
        }
        if [
            self.capture_identity,
            self.protocol,
            self.membership_rule,
            self.window_declaration,
        ]
        .contains(&[0; 32])
            || self
                .phase_offered
                .iter()
                .any(|n| *n < settings.min_phase_samples || *n > 65_536)
        {
            return Err(StructuredUnknown::InvalidSettings);
        }
        Ok(())
    }
    fn validate_close(
        &self,
        settings: &StructuredSettingsV2,
        close: &StructuredServiceWindowCloseV2,
        samples: &[StructuredNumericObservationV2],
        phase: StructuredPhaseV2,
    ) -> Result<()> {
        if close.phase != phase {
            return Err(StructuredUnknown::PhaseLeakage);
        }
        if close.population_sha256 == [0; 32]
            || close.member_count != samples.len()
            || member_tickets_signature(samples) != close.member_tickets_sha256
        {
            return Err(StructuredUnknown::IncompletePhasePopulation);
        }
        if samples.len() < settings.min_phase_samples || samples.len() > settings.max_phase_samples
        {
            return Err(StructuredUnknown::IncompletePhasePopulation);
        }
        let i = phase.index();
        let first = self.phase_offered[..i].iter().sum::<usize>() as u64 + 1;
        let end = first + self.phase_offered[i] as u64;
        if samples
            .iter()
            .any(|s| s.membership.offered_ordinal < first || s.membership.offered_ordinal >= end)
        {
            return Err(StructuredUnknown::PhaseLeakage);
        }
        Ok(())
    }
}
impl FittedStructuredModelV2 {
    pub(super) fn service_domain_policy(&self) -> Option<StructuredServiceDomainPolicyV1> {
        self.service_window
            .as_ref()
            .map(|v| v.contract.domain_policy)
            .or_else(|| self.owner_blocks.as_ref().map(|v| v.contract.domain_policy))
    }

    /// Pure eligibility for the next phase. Neither costs, predictions nor
    /// residual outcomes are arguments or inputs to this decision.
    pub fn service_input_membership(
        &self,
        input: &StructuredInputV2,
    ) -> Result<StructuredServiceInputMembershipV1> {
        let policy = self
            .service_domain_policy()
            .ok_or(StructuredUnknown::WrongProtocol)?;
        if policy == StructuredServiceDomainPolicyV1::AllOffered {
            return Err(StructuredUnknown::WrongProtocol);
        }
        self.same_population_domain(input)?;
        input.validate(&self.settings)?;
        if policy == StructuredServiceDomainPolicyV1::NonNegativePhysicalEnvelopeV1 {
            let physical = self
                .numerical
                .physical()
                .ok_or(StructuredUnknown::WrongProtocol)?;
            return match if physical.phase_support.is_some() && !self.uses_joint_cells() {
                physical.phase_membership(input, StructuredPhaseV2::Fit)
            } else {
                physical.membership(input)
            } {
                Ok(()) => Ok(StructuredServiceInputMembershipV1::Eligible),
                Err(StructuredUnknown::UnidentifiedDirection) => {
                    Ok(StructuredServiceInputMembershipV1::OutsideFitSupport)
                }
                Err(StructuredUnknown::QualificationCoverage)
                    if physical.phase_support.is_some() =>
                {
                    Ok(StructuredServiceInputMembershipV1::OutsideFitSupport)
                }
                Err(error) => Err(error),
            };
        }
        if policy != StructuredServiceDomainPolicyV1::FrozenFitSupportV1 {
            return Err(StructuredUnknown::WrongProtocol);
        }
        let (fit, support, _) = self.numerical.legacy()?;
        if !support.contains(&input.support) {
            return Ok(StructuredServiceInputMembershipV1::OutsideFitSupport);
        }
        match fit.identify(&input.basis) {
            Ok(()) => Ok(StructuredServiceInputMembershipV1::Eligible),
            Err(StructuredUnknown::UnidentifiedDirection) => {
                Ok(StructuredServiceInputMembershipV1::OutsideFitSupport)
            }
            Err(error) => Err(error),
        }
    }

    pub fn fit_service_window(
        fingerprint: ExecutionFingerprint,
        settings: StructuredSettingsV2,
        scope: StructuredScopeV2,
        contract: StructuredServiceWindowContractV2,
        close: StructuredServiceWindowCloseV2,
        samples: &[StructuredNumericObservationV2],
        frozen_at_ns: u64,
    ) -> Result<Self> {
        Self::fit_service_window_parts(
            fingerprint,
            settings,
            scope,
            contract,
            close,
            samples,
            frozen_at_ns,
            None,
        )
    }

    pub fn fit_service_window_from_certificate(
        fingerprint: ExecutionFingerprint,
        settings: StructuredSettingsV2,
        scope: StructuredScopeV2,
        contract: StructuredServiceWindowContractV2,
        close: StructuredServiceWindowCloseV2,
        samples: &[StructuredNumericObservationV2],
        frozen_at_ns: u64,
        certificate: super::super::nonnegative::FitCertificate,
    ) -> Result<Self> {
        if contract.domain_policy != StructuredServiceDomainPolicyV1::NonNegativePhysicalEnvelopeV1
        {
            return Err(StructuredUnknown::WrongProtocol);
        }
        Self::fit_service_window_parts(
            fingerprint,
            settings,
            scope,
            contract,
            close,
            samples,
            frozen_at_ns,
            Some(certificate),
        )
    }

    fn fit_service_window_parts(
        fingerprint: ExecutionFingerprint,
        settings: StructuredSettingsV2,
        scope: StructuredScopeV2,
        contract: StructuredServiceWindowContractV2,
        close: StructuredServiceWindowCloseV2,
        samples: &[StructuredNumericObservationV2],
        frozen_at_ns: u64,
        certificate: Option<super::super::nonnegative::FitCertificate>,
    ) -> Result<Self> {
        settings.validate()?;
        scope.validate()?;
        contract.validate(&settings)?;
        contract.validate_close(&settings, &close, samples, StructuredPhaseV2::Fit)?;
        // Private numerical accumulator, NOT an old fixed-count declaration.
        // Unknown future counts never enter the frozen fit identity below.
        let source = StructuredSourceContractV2 {
            capture_identity: contract.capture_identity,
            protocol: contract.protocol,
            membership_rule: contract.membership_rule,
            cohort_manifest: contract.window_declaration,
            phase_members: [samples.len(), 0, 0],
        };
        source.complete_population(samples, StructuredPhaseV2::Fit)?;
        let population = Population {
            contract,
            closes: [Some(close), None, None],
            fit_signature: [0; 32],
            calibrated_signature: [0; 32],
        };
        let mut model = Self::fit_parts(
            fingerprint,
            settings,
            scope,
            source,
            samples,
            frozen_at_ns,
            Some(population),
            None,
            certificate,
        )?;
        let signature = model.service_fit_signature();
        model.service_window.as_mut().unwrap().fit_signature = signature;
        Ok(model)
    }
    fn service_fit_signature(&self) -> [u8; 32] {
        let population = self.service_window.as_ref().unwrap();
        let mut digest = Sha256::new();
        digest.update(b"ferrum.structured-service-window-fit.v1\0");
        digest.update(MODEL_REVISION_V2.as_bytes());
        if population.contract.domain_policy == StructuredServiceDomainPolicyV1::FrozenFitSupportV1
        {
            digest.update(b"frozen-fit-support-v1\0");
        }
        for hash in [
            self.fingerprint.model_weights,
            self.fingerprint.numerical_policy,
            self.fingerprint.device_runtime,
            self.fingerprint.execution_config,
            self.source.capture_identity,
            self.source.protocol,
            self.source.membership_rule,
            population.contract.window_declaration,
            self.domain_signature,
        ] {
            digest.update(hash);
        }
        for n in population.contract.phase_offered {
            digest.update((n as u64).to_le_bytes());
        }
        for n in [
            self.settings.min_phase_samples as u64,
            self.settings.min_fit_redundancy as u64,
            self.settings.max_phase_samples as u64,
            self.settings.max_axes as u64,
            self.settings.max_rank as u64,
            self.settings.max_wave_ns,
            self.settings.max_sample_age_ns,
            self.settings.static_margin_ns,
            self.fit_samples as u64,
            self.frozen_at_ns,
        ] {
            digest.update(n.to_le_bytes());
        }
        digest.update(self.settings.learned_drift.signature());
        digest.update(population.closes[0].as_ref().unwrap().signature());
        self.scope.bind_parameters(&mut digest);
        self.state.bind_parameters(&mut digest);
        self.numerical.bind_fit(&mut digest);
        if self.numerical.physical().is_none() && self.exemplar.completion.is_some() {
            self.completion_coverage.bind(&mut digest);
        }
        digest.finalize().into()
    }
    pub fn calibrate_service_window(
        mut self,
        close: StructuredServiceWindowCloseV2,
        samples: &[StructuredNumericObservationV2],
        frozen_at_ns: u64,
    ) -> Result<CalibratedStructuredModelV2> {
        let population = self
            .service_window
            .as_mut()
            .ok_or(StructuredUnknown::WrongProtocol)?;
        population.contract.validate_close(
            &self.settings,
            &close,
            samples,
            StructuredPhaseV2::Residual,
        )?;
        population.closes[1] = Some(close);
        self.source.phase_members[1] = samples.len();
        let mut calibrated = self.calibrate_parts(samples, frozen_at_ns)?;
        let mut digest = Sha256::new();
        digest.update(b"ferrum.structured-service-window-calibrated.v1\0");
        // Includes frozen fit identity, independent residual q99/support/settings.
        digest.update(calibrated.legacy_parameters_signature());
        digest.update(
            calibrated.fitted.service_window.as_ref().unwrap().closes[1]
                .as_ref()
                .unwrap()
                .signature(),
        );
        calibrated
            .fitted
            .service_window
            .as_mut()
            .unwrap()
            .calibrated_signature = digest.finalize().into();
        Ok(calibrated)
    }
}
impl CalibratedStructuredModelV2 {
    /// Qualification additionally requires the input support frozen from the
    /// residual population. An overestimated/underestimated duration cannot
    /// alter membership, and an invalid input cannot become an outside point.
    pub fn service_input_membership(
        &self,
        input: &StructuredInputV2,
    ) -> Result<StructuredServiceInputMembershipV1> {
        let fitted = self.fitted.service_input_membership(input)?;
        if fitted != StructuredServiceInputMembershipV1::Eligible {
            return Ok(fitted);
        }
        if let Some(bank) = &self.joint_bank {
            let contract = &self
                .fitted
                .numerical
                .physical()
                .ok_or(StructuredUnknown::WrongProtocol)?
                .contract;
            return Ok(if bank.contains(input, contract, false)? {
                fitted
            } else {
                StructuredServiceInputMembershipV1::OutsideResidualSupport
            });
        }
        if let Some(physical) = self.fitted.numerical.physical() {
            return if physical.phase_support.is_some() {
                match physical.phase_membership(input, StructuredPhaseV2::Residual) {
                    Ok(()) => Ok(fitted),
                    Err(
                        StructuredUnknown::QualificationCoverage
                        | StructuredUnknown::UnidentifiedDirection,
                    ) => Ok(StructuredServiceInputMembershipV1::OutsideResidualSupport),
                    Err(error) => Err(error),
                }
            } else {
                Ok(fitted)
            };
        }
        Ok(
            if self
                .residual_support
                .as_ref()
                .ok_or(StructuredUnknown::WrongProtocol)?
                .contains(&input.support)
            {
                StructuredServiceInputMembershipV1::Eligible
            } else {
                StructuredServiceInputMembershipV1::OutsideResidualSupport
            },
        )
    }

    pub fn qualify_service_window(
        mut self,
        close: StructuredServiceWindowCloseV2,
        samples: &[StructuredNumericObservationV2],
        frozen_at_ns: u64,
    ) -> Result<QualifiedStructuredModelV2> {
        let population = self
            .fitted
            .service_window
            .as_mut()
            .ok_or(StructuredUnknown::WrongProtocol)?;
        population.contract.validate_close(
            &self.fitted.settings,
            &close,
            samples,
            StructuredPhaseV2::Qualification,
        )?;
        population.closes[2] = Some(close);
        self.fitted.source.phase_members[2] = samples.len();
        self.qualify_parts(samples, frozen_at_ns)
    }
}
impl QualifiedStructuredModelV2 {
    pub fn service_window_contract(&self) -> Option<&StructuredServiceWindowContractV2> {
        self.calibrated
            .fitted
            .service_window
            .as_ref()
            .map(|v| &v.contract)
    }
    /// Original freeze identities. Later population counts cannot rewrite an
    /// earlier phase; callers use these in the source6/profile13 receipts.
    pub fn service_window_phase_signatures(&self) -> Option<[[u8; 32]; 3]> {
        self.calibrated.fitted.service_window.as_ref().map(|v| {
            [
                v.fit_signature,
                v.calibrated_signature,
                self.parameters_signature(),
            ]
        })
    }
}

impl FittedStructuredModelV2 {
    pub fn nonnegative_fit_certificate(
        &self,
    ) -> Option<&super::super::nonnegative::FitCertificate> {
        self.numerical
            .physical()
            .map(|physical| physical.certificate())
    }
}
impl QualifiedStructuredModelV2 {
    pub fn nonnegative_fit_certificate(
        &self,
    ) -> Option<&super::super::nonnegative::FitCertificate> {
        self.calibrated.fitted.nonnegative_fit_certificate()
    }
}
