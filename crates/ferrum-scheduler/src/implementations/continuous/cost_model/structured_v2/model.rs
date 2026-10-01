//! Immutable phase transitions. These numerical APIs do not attest live receipts.
use super::envelope::PendingGenerators;
use super::fit::{FitRow, RowSpaceFit};
use super::phase::{check_time, PhaseState};
use super::support::JointSupport;
use super::*;
use sha2::{Digest, Sha256};
mod joint_cells;
pub use joint_cells::StructuredJointCellKeyV1;
mod memory;
mod numerical;
mod owner_blocks;
mod physical_envelope;
mod service_windows;
use numerical::NumericalModel;
pub use owner_blocks::{
    OwnerAlgorithmUniversePolicyV1, OwnerBlockScheduleV1, OwnerInputReadinessDecisionV1,
    OwnerInputReadinessGapV1, OwnerInputReadinessV1, OwnerInputTargetV1,
    OwnerOpeningFrontierPolicyV1, OwnerPhaseSupportPolicyV1, OwnerPredictionValidityPolicyV1,
    StructuredOwnerPhaseBoundaryV1, StructuredOwnerPhaseCloseV1, StructuredOwnerPhaseContractV1,
};
pub use physical_envelope::{
    NonNegativeEnvelopeContractV1, NonNegativePlanningEstimatorV1, StructuredPopulationPolicyV1,
    WorkAxisAndBranchChallengesV1,
};
mod support_diagnostic;
pub use service_windows::{
    StructuredServiceDomainPolicyV1, StructuredServiceInputMembershipV1,
    StructuredServiceWindowCloseV2, StructuredServiceWindowContractV2,
};
pub use support_diagnostic::{StructuredFitSupportDiagnosticV1, StructuredFitSupportReasonV1};

pub struct FittedStructuredModelV2 {
    fingerprint: ExecutionFingerprint,
    settings: StructuredSettingsV2,
    scope: StructuredScopeV2,
    source: StructuredSourceContractV2,
    exemplar: StructuredInputV2,
    /// Computed once at Fit freeze. Queries compare checked typed identities.
    domain_signature: [u8; 32],
    numerical: NumericalModel,
    fit_samples: usize,
    state: PhaseState,
    frozen_at_ns: u64,
    service_window: Option<service_windows::Population>,
    owner_blocks: Option<owner_blocks::Population>,
    completion_coverage: super::completion::CompletionCoverage,
}
impl FittedStructuredModelV2 {
    /// `frozen_at_ns` is the original capture clock, also during offline replay.
    pub fn fit(
        fingerprint: ExecutionFingerprint,
        settings: StructuredSettingsV2,
        scope: StructuredScopeV2,
        source: StructuredSourceContractV2,
        samples: &[StructuredNumericObservationV2],
        frozen_at_ns: u64,
    ) -> Result<Self> {
        settings.validate()?;
        scope.validate()?;
        source.validate(&settings)?;
        source.complete_population(samples, StructuredPhaseV2::Fit)?;
        Self::fit_parts(
            fingerprint,
            settings,
            scope,
            source,
            samples,
            frozen_at_ns,
            None,
            None,
            None,
        )
    }
    fn fit_parts(
        fingerprint: ExecutionFingerprint,
        settings: StructuredSettingsV2,
        scope: StructuredScopeV2,
        source: StructuredSourceContractV2,
        samples: &[StructuredNumericObservationV2],
        frozen_at_ns: u64,
        service_window: Option<service_windows::Population>,
        owner_blocks: Option<owner_blocks::Population>,
        certificate: Option<super::nonnegative::FitCertificate>,
    ) -> Result<Self> {
        if service_window.is_some() && owner_blocks.is_some() {
            return Err(StructuredUnknown::WrongProtocol);
        }
        samples[0].input.validate(&settings)?;
        let exemplar = samples[0].input.clone();
        let envelope = service_window
            .as_ref()
            .and_then(|window| window.contract.nonnegative_envelope.as_ref())
            .or_else(|| {
                owner_blocks
                    .as_ref()
                    .and_then(|v| v.contract.nonnegative_envelope.as_ref())
            });
        if envelope.is_none()
            && samples
                .iter()
                .any(|s| s.input.algorithm_universe_signature().is_some())
        {
            return Err(StructuredUnknown::WrongProtocol);
        }
        if envelope.is_some_and(joint_cells::enabled) && owner_blocks.is_none() {
            return Err(StructuredUnknown::WrongProtocol);
        }
        let population_policy = envelope.map_or(StructuredPopulationPolicyV1::default(), |v| {
            v.population_policy
        });
        scope.validate_population_input(&exemplar, population_policy)?;
        let domain_signature = match &scope.numerical_family {
            Some(family) => family.signature()?,
            None => exemplar.domain,
        };
        let mut state = PhaseState::new();
        for sample in samples {
            state.observe(
                sample,
                &fingerprint,
                &settings,
                &source,
                StructuredPhaseV2::Fit,
                0,
                frozen_at_ns,
            )?;
            if scope.numerical_family.is_some() {
                scope.validate_population_input(&sample.input, population_policy)?;
            } else {
                exemplar.same_domain(&sample.input)?;
            }
            sample.input.validate(&settings)?;
        }
        let mut numerical =
            NumericalModel::fit(samples, &settings, &exemplar, envelope, certificate)?;
        if let Some(policy) = owner_blocks
            .as_ref()
            .and_then(|p| p.contract.schedule.phase_support)
        {
            numerical
                .physical_mut()
                .ok_or(StructuredUnknown::WrongProtocol)?
                .phase_support = Some(policy);
        }
        let completion_coverage = super::completion::CompletionCoverage::observed(samples)?;
        Ok(Self {
            fingerprint,
            settings,
            scope,
            source,
            exemplar,
            domain_signature,
            numerical,
            fit_samples: samples.len(),
            state,
            frozen_at_ns,
            service_window,
            owner_blocks,
            completion_coverage,
        })
    }
    fn population_policy(&self) -> StructuredPopulationPolicyV1 {
        self.numerical
            .physical()
            .map_or(StructuredPopulationPolicyV1::default(), |p| {
                p.population_policy()
            })
    }
    fn same_population_domain(&self, input: &StructuredInputV2) -> Result<()> {
        if self.scope.numerical_family.is_some() {
            self.scope
                .validate_population_input(input, self.population_policy())
        } else {
            self.exemplar.same_domain(input)
        }
    }
    fn same_query_population(&self, input: &StructuredInputV2) -> Result<()> {
        if self.scope.numerical_family.is_some() {
            self.scope
                .validate_population_input(input, self.population_policy())
        } else {
            self.exemplar.same_query_domain(input)
        }
    }
    fn numerical_query<'a>(
        &self,
        query: &'a StructuredQueryV2,
    ) -> Result<std::borrow::Cow<'a, StructuredQueryV2>> {
        if let Some(universe) = self
            .numerical
            .physical()
            .and_then(|p| p.contract.algorithm_universe.as_ref())
        {
            match query.input.numerical_family_key() {
                Ok(_)
                    if query.input.algorithm_universe_signature() != Some(universe.signature()) =>
                {
                    return query
                        .clone()
                        .with_algorithm_universe(universe)
                        .map(std::borrow::Cow::Owned);
                }
                Ok(_) | Err(StructuredUnknown::UnsupportedScope) => {}
                Err(error) => return Err(error),
            }
        }
        Ok(std::borrow::Cow::Borrowed(query))
    }
    pub fn parameters_signature(&self) -> [u8; 32] {
        if let Some(population) = &self.service_window {
            return population.fit_signature;
        }
        if let Some(population) = &self.owner_blocks {
            return population.fit_signature;
        }
        self.legacy_parameters_signature()
    }
    fn legacy_parameters_signature(&self) -> [u8; 32] {
        let mut digest = Sha256::new();
        digest.update(b"ferrum.structured-fit-state.v2\0");
        digest.update(MODEL_REVISION_V2.as_bytes());
        for value in [
            self.fingerprint.model_weights,
            self.fingerprint.numerical_policy,
            self.fingerprint.device_runtime,
            self.fingerprint.execution_config,
            self.source.capture_identity,
            self.source.protocol,
            self.source.membership_rule,
            self.source.cohort_manifest,
            self.domain_signature,
        ] {
            digest.update(value);
        }
        for value in [
            self.settings.min_phase_samples as u64,
            self.settings.min_fit_redundancy as u64,
            self.settings.max_phase_samples as u64,
            self.settings.max_axes as u64,
            self.settings.max_rank as u64,
            self.settings.max_wave_ns,
            self.settings.max_sample_age_ns,
            self.settings.static_margin_ns,
            self.source.phase_members[0] as u64,
            self.source.phase_members[1] as u64,
            self.source.phase_members[2] as u64,
            self.fit_samples as u64,
            self.frozen_at_ns,
        ] {
            digest.update(value.to_le_bytes());
        }
        if !self.settings.learned_drift.is_disabled() {
            digest.update(self.settings.learned_drift.signature());
        }
        self.scope.bind_parameters(&mut digest);
        self.state.bind_parameters(&mut digest);
        self.numerical.bind_fit(&mut digest);
        if self.numerical.physical().is_none() && self.exemplar.completion.is_some() {
            self.completion_coverage.bind(&mut digest);
        }
        digest.finalize().into()
    }
    /// Consumes the candidate. Failed or omitted members cannot be retried by
    /// retaining this state and replacing a slow observation with another one.
    pub fn calibrate(
        self,
        samples: &[StructuredNumericObservationV2],
        frozen_at_ns: u64,
    ) -> Result<CalibratedStructuredModelV2> {
        if self.service_window.is_some() || self.owner_blocks.is_some() {
            return Err(StructuredUnknown::WrongProtocol);
        }
        self.calibrate_parts(samples, frozen_at_ns)
    }
    fn calibrate_parts(
        mut self,
        samples: &[StructuredNumericObservationV2],
        frozen_at_ns: u64,
    ) -> Result<CalibratedStructuredModelV2> {
        if self.uses_joint_cells() {
            return self.calibrate_joint_cells(samples, frozen_at_ns);
        }
        self.source
            .complete_population(samples, StructuredPhaseV2::Residual)?;
        check_time(frozen_at_ns, self.frozen_at_ns, self.state.expires_at_ns)?;
        let mut residuals = Vec::with_capacity(samples.len());
        let mut minimum_error = i128::MAX;
        let mut maximum_error = i128::MIN;
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
            sample.input.validate(&self.settings)?;
            let fitted = self.numerical.actual_upper(&sample.input)?;
            let signed_error = i128::from(sample.wall_ns) - i128::from(fitted);
            minimum_error = minimum_error.min(signed_error);
            maximum_error = maximum_error.max(signed_error);
            residuals.push(sample.wall_ns.saturating_sub(fitted));
        }
        residuals.sort_unstable();
        // Fixed empirical nearest-rank p99. Qualification never changes this.
        let residual_ns = residuals[(99 * residuals.len()).div_ceil(100) - 1];
        let residual_support = if let Some(physical) = self.numerical.physical_mut() {
            physical.freeze_phase(StructuredPhaseV2::Residual, samples)?;
            None
        } else {
            Some(JointSupport::new(
                samples.iter().map(|s| s.input.support.as_slice()),
            )?)
        };
        let learned_span_margin_ns = self
            .settings
            .learned_drift
            .freeze(minimum_error, maximum_error)?;
        let completion_coverage = self
            .completion_coverage
            .intersect(super::completion::CompletionCoverage::observed(samples)?);
        Ok(CalibratedStructuredModelV2 {
            fitted: self,
            learned_span_margin_ns,
            residual_ns,
            residual_support,
            joint_bank: None,
            residual_samples: samples.len(),
            frozen_at_ns,
            completion_coverage,
        })
    }
}

pub struct CalibratedStructuredModelV2 {
    fitted: FittedStructuredModelV2,
    residual_ns: u64,
    learned_span_margin_ns: u64,
    residual_support: Option<JointSupport>,
    joint_bank: Option<joint_cells::JointMarginBank>,
    residual_samples: usize,
    frozen_at_ns: u64,
    completion_coverage: super::completion::CompletionCoverage,
}
impl CalibratedStructuredModelV2 {
    fn uncertainty(&self) -> StructuredUncertaintyV2 {
        let fit_error_floor_ns = self.fitted.numerical.fit_error_floor_ns();
        StructuredUncertaintyV2 {
            fit_error_floor_ns,
            residual_ns: self.residual_ns,
            effective_residual_ns: fit_error_floor_ns.max(self.residual_ns),
            static_margin_ns: self.fitted.settings.static_margin_ns,
            learned_span_margin_ns: self.learned_span_margin_ns,
        }
    }
    pub fn parameters_signature(&self) -> [u8; 32] {
        if let Some(population) = &self.fitted.service_window {
            return population.calibrated_signature;
        }
        if let Some(population) = &self.fitted.owner_blocks {
            return population.calibrated_signature;
        }
        self.legacy_parameters_signature()
    }
    fn legacy_parameters_signature(&self) -> [u8; 32] {
        let mut digest = Sha256::new();
        digest.update(b"ferrum.structured-calibrated-state.v2\0");
        digest.update(self.fitted.parameters_signature());
        if !self.fitted.settings.learned_drift.is_disabled() {
            digest.update(b"ferrum.structured-learned-span.v1\0");
            digest.update(self.learned_span_margin_ns.to_le_bytes());
        }
        for value in [
            self.residual_ns,
            self.residual_samples as u64,
            self.frozen_at_ns,
        ] {
            digest.update(value.to_le_bytes());
        }
        if let Some(bank) = &self.joint_bank {
            bank.bind(&mut digest, false);
        }
        if let Some(support) = &self.residual_support {
            support.bind_parameters(&mut digest);
        }
        if let Some(physical) = self
            .fitted
            .numerical
            .physical()
            .filter(|_| self.joint_bank.is_none())
        {
            physical.bind_phase(StructuredPhaseV2::Residual, &mut digest);
        }
        if self.fitted.numerical.physical().is_none() && self.fitted.exemplar.completion.is_some() {
            self.completion_coverage.bind(&mut digest);
        }
        digest.finalize().into()
    }
    fn predict_core(
        &self,
        query: &StructuredQueryV2,
        now: u64,
        qualification_support: Option<&JointSupport>,
        required_phases: usize,
    ) -> Result<StructuredPredictionV2> {
        self.predict_core_detailed(query, now, qualification_support, required_phases, None)
            .map_err(StructuredQueryFailureV2::reason)
    }
    fn predict_core_detailed(
        &self,
        query: &StructuredQueryV2,
        now: u64,
        qualification_support: Option<&JointSupport>,
        required_phases: usize,
        completion_coverage: Option<super::completion::CompletionCoverage>,
    ) -> QueryResult<StructuredPredictionV2> {
        self.evaluate_input_detailed(
            query,
            qualification_support,
            required_phases,
            completion_coverage,
            || check_time(now, self.frozen_at_ns, self.fitted.state.expires_at_ns),
        )
    }
    // Share the complete original numerical validation, including its failure
    // ordering. Only the feedback input validator omits prediction eligibility;
    // it discards the result and can never return a cost or valid-until token.
    fn evaluate_input_detailed(
        &self,
        query: &StructuredQueryV2,
        qualification_support: Option<&JointSupport>,
        required_phases: usize,
        completion_coverage: Option<super::completion::CompletionCoverage>,
        validate_time: impl FnOnce() -> Result<()>,
    ) -> QueryResult<StructuredPredictionV2> {
        let f = &self.fitted;
        if let Some(bank) = &self.joint_bank {
            validate_time()?;
            return bank.prediction(f, query, required_phases == 3);
        }
        let projected = f.numerical_query(query)?;
        let query = projected.as_ref();
        validate_time()?;
        f.same_query_population(&query.input)?;
        query.validate_for_prediction(&f.settings)?;
        if f.numerical.physical().is_none() {
            if let Some(coverage) = completion_coverage {
                coverage.authorize_detailed(&query.input)?;
            }
        }
        let (lower_ns, upper_ns) = if let Some(physical) = f.numerical.physical() {
            physical.bounds_detailed(query, required_phases)?
        } else {
            f.scope.authorize_detailed(query)?;
            let (fit, fit_support, generators) = f.numerical.legacy()?;
            let bounds = generators.bounds_detailed(fit, &query.input, query.pending.as_ref())?;
            let bounds = generators.extend_repetition_detailed(
                &query.input,
                query.repetition_upper_sum,
                bounds,
            )?;
            if !fit_support.contains_envelope(&bounds.support_lower, &bounds.support_upper)
                || !self
                    .residual_support
                    .as_ref()
                    .ok_or(StructuredUnknown::WrongProtocol)?
                    .contains_envelope(&bounds.support_lower, &bounds.support_upper)
                || qualification_support.is_some_and(|support| {
                    !support.contains_envelope(&bounds.support_lower, &bounds.support_upper)
                })
            {
                return Err(StructuredQueryFailureV2::OutsideSupport(
                    StructuredUnknown::JointSupport,
                ));
            }
            (bounds.lower_ns, bounds.upper_ns)
        };
        let uncertainty = self.uncertainty();
        let planning_ns = super::query_outcome::planning_sum(
            upper_ns,
            uncertainty.effective_residual_ns,
            f.settings.static_margin_ns,
            self.learned_span_margin_ns,
            f.settings.max_wave_ns,
        )?;
        Ok(StructuredPredictionV2 {
            fitted_lower_ns: lower_ns,
            fitted_upper_ns: upper_ns,
            residual_ns: uncertainty.residual_ns,
            fit_error_floor_ns: uncertainty.fit_error_floor_ns,
            effective_residual_ns: uncertainty.effective_residual_ns,
            learned_span_margin_ns: self.learned_span_margin_ns,
            planning_ns,
            valid_until_ns: f.state.expires_at_ns,
            fit_samples: f.fit_samples,
            residual_samples: self.residual_samples,
            identified_rank: f.numerical.rank(),
        })
    }
    /// Challenge every predeclared member and coverage requirement, using the
    /// already frozen fit and residual. No position/count receives a new fit.
    pub fn qualify(
        self,
        samples: &[StructuredNumericObservationV2],
        frozen_at_ns: u64,
    ) -> Result<QualifiedStructuredModelV2> {
        if self.fitted.service_window.is_some() || self.fitted.owner_blocks.is_some() {
            return Err(StructuredUnknown::WrongProtocol);
        }
        self.qualify_parts(samples, frozen_at_ns)
    }
    fn qualify_parts(
        mut self,
        samples: &[StructuredNumericObservationV2],
        frozen_at_ns: u64,
    ) -> Result<QualifiedStructuredModelV2> {
        if self.joint_bank.is_some() {
            return self.qualify_joint_cells(samples, frozen_at_ns);
        }
        self.fitted
            .source
            .complete_population(samples, StructuredPhaseV2::Qualification)?;
        check_time(
            frozen_at_ns,
            self.frozen_at_ns,
            self.fitted.state.expires_at_ns,
        )?;
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
            if let Some(physical) = self.fitted.numerical.physical() {
                physical.membership(&sample.input)?;
            }
            let prediction = self.predict_core(
                &StructuredQueryV2::exact(sample.input.clone()),
                frozen_at_ns,
                None,
                2,
            )?;
            if sample.wall_ns > prediction.planning_ns {
                // Cold, fixed-size failure evidence from the original checked
                // prediction. No retry, re-fit or qualification-based margin.
                tracing::warn!(target: "ferrum_scheduler::structured_owner_diagnostics",
                    event = "structured_qualification_underestimate_v1",
                    capture_identity = ?self.fitted.source.capture_identity,
                    member_ordinal = sample.membership.member_ordinal,
                    offered_ordinal = sample.membership.offered_ordinal,
                    accepted_fifo = sample.ordinal,
                    call_id = sample.call_id,
                    wall_ns = sample.wall_ns,
                    fitted_upper_ns = prediction.fitted_upper_ns,
                    fit_error_floor_ns = prediction.fit_error_floor_ns,
                    residual_ns = prediction.residual_ns,
                    effective_residual_ns = prediction.effective_residual_ns,
                    static_margin_ns = self.fitted.settings.static_margin_ns,
                    learned_span_margin_ns = prediction.learned_span_margin_ns,
                    planning_ns = prediction.planning_ns,
                    excess_ns = sample.wall_ns - prediction.planning_ns,
                    "Original qualification member exceeded its frozen planning value");
                return Err(StructuredUnknown::QualificationUnderestimate);
            }
        }
        if let Some(physical) = self.fitted.numerical.physical_mut() {
            physical.freeze_phase(StructuredPhaseV2::Qualification, samples)?;
        } else {
            self.fitted.scope.qualify_coverage(samples)?;
        }
        // Heldout inputs can only narrow publication. Every eligible heldout
        // duration was challenged above first; no failing point can be removed
        // by choosing a smaller domain after seeing its cost.
        let qualification_support = (self.fitted.service_domain_policy()
            == Some(StructuredServiceDomainPolicyV1::FrozenFitSupportV1))
        .then(|| JointSupport::new(samples.iter().map(|sample| sample.input.support.as_slice())))
        .transpose()?;
        self.fitted.state.freeze_calls()?;
        let completion_coverage = self
            .completion_coverage
            .intersect(super::completion::CompletionCoverage::observed(samples)?);
        Ok(QualifiedStructuredModelV2 {
            calibrated: self,
            qualification_samples: samples.len(),
            qualification_support,
            frozen_at_ns,
            completion_coverage,
        })
    }
}

pub struct QualifiedStructuredModelV2 {
    calibrated: CalibratedStructuredModelV2,
    pub qualification_samples: usize,
    qualification_support: Option<JointSupport>,
    frozen_at_ns: u64,
    completion_coverage: super::completion::CompletionCoverage,
}
impl QualifiedStructuredModelV2 {
    /// Pure prospective input dispatch for the explicitly frozen intersection.
    /// None preserves legacy catalog selection. False excludes only valid work
    /// outside a phase's input support, never a clock, cost or feedback result.
    /// Callers first match this child's declared numerical population.
    pub fn catalog_input_membership(&self, query: &StructuredQueryV2) -> Result<Option<bool>> {
        let fitted = &self.calibrated.fitted;
        if let Some(bank) = &self.calibrated.joint_bank {
            return match bank.catalog_membership(fitted, query) {
                Ok(eligible) => Ok(Some(eligible)),
                Err(StructuredQueryFailureV2::OutsideSupport(_)) => Ok(Some(false)),
                Err(error) => Err(error.reason()),
            };
        }
        let Some(physical) = fitted.numerical.physical() else {
            return Ok(None);
        };
        if physical.phase_support.is_none() {
            return Ok(None);
        }
        let query = fitted.numerical_query(query)?;
        fitted.same_query_population(&query.input)?;
        query.validate_for_prediction(&fitted.settings)?;
        match physical.authorized_query_upper(&query, 3, true) {
            Ok(_) => Ok(Some(true)),
            Err(StructuredQueryFailureV2::OutsideSupport(_)) => Ok(Some(false)),
            Err(failure) => Err(failure.reason()),
        }
    }

    /// Conservative, cold proof of physical input coverage containment for a
    /// startup extension. This proves structural support, not equal predictions
    /// or renewed freshness. Legacy support has no containment proof here.
    pub fn preserves_physical_input_coverage(&self, previous: &Self) -> bool {
        let new = &self.calibrated.fitted;
        let old = &previous.calibrated.fitted;
        if new.fingerprint != old.fingerprint
            || new.domain_signature != old.domain_signature
            || new.settings.max_axes < old.settings.max_axes
        {
            return false;
        }
        let (Some(n), Some(o)) = (new.numerical.physical(), old.numerical.physical()) else {
            return false;
        };
        match (&self.calibrated.joint_bank, &previous.calibrated.joint_bank) {
            (Some(bank), Some(previous)) => {
                // Cell coordinates mean the same thing only under the same
                // physical, algorithm and numerical contract. This establishes
                // input coverage alone; each model keeps its own parameters,
                // qualification evidence and original sample expiry.
                joint_cells::enabled(&n.contract)
                    && n.contract == o.contract
                    && n.phase_support == o.phase_support
                    && new.exemplar.basis.len() == old.exemplar.basis.len()
                    && new.exemplar.support.len() == old.exemplar.support.len()
                    && bank.contains_qualified(previous)
            }
            (None, None) => n.contains_input_coverage(o),
            _ => false,
        }
    }

    /// Check the original qualification clock and oldest-sample expiry, without a query.
    pub fn validate_runtime_at(&self, model_now_ns: u64) -> Result<()> {
        check_time(
            model_now_ns,
            self.frozen_at_ns,
            self.calibrated.fitted.state.expires_at_ns,
        )
    }

    /// Validate retrospective input independently of the model's age. The
    /// worker uses this only to distinguish expiry from damaged evidence.
    /// This never yields a prediction, refreshes TTL, or grants query authority.
    pub fn validate_retrospective_query_input(
        &self,
        fingerprint: &ExecutionFingerprint,
        query: &StructuredQueryV2,
    ) -> std::result::Result<(), StructuredQueryFailureV2> {
        if fingerprint != &self.calibrated.fitted.fingerprint {
            return Err(StructuredUnknown::WrongFingerprint.into());
        }
        self.calibrated
            .evaluate_input_detailed(
                query,
                self.qualification_support.as_ref(),
                3,
                Some(self.completion_coverage),
                || Ok(()),
            )
            .map(|_| ())
    }

    /// Frozen diagnostic components; qualification never updates these values.
    pub fn uncertainty(&self) -> StructuredUncertaintyV2 {
        self.calibrated.uncertainty()
    }
    /// Frozen safety limits for the runtime adapter; feedback cannot change them.
    pub fn runtime_limits(&self) -> (u64, u64) {
        (
            self.calibrated.fitted.settings.max_wave_ns,
            self.calibrated.fitted.settings.max_sample_age_ns,
        )
    }
    pub fn predict_query(
        &self,
        fingerprint: &ExecutionFingerprint,
        query: &StructuredQueryV2,
        model_now_ns: u64,
    ) -> Result<StructuredPredictionV2> {
        self.predict_query_detailed(fingerprint, query, model_now_ns)
            .map_err(StructuredQueryFailureV2::reason)
    }
    /// Availability for an original checked query. No prediction or coverage
    /// gate is retried, and qualification still consumes the legacy result.
    pub fn predict_query_detailed(
        &self,
        fingerprint: &ExecutionFingerprint,
        query: &StructuredQueryV2,
        model_now_ns: u64,
    ) -> std::result::Result<StructuredPredictionV2, StructuredQueryFailureV2> {
        if fingerprint != &self.calibrated.fitted.fingerprint {
            return Err(StructuredUnknown::WrongFingerprint.into());
        }
        check_time(
            model_now_ns,
            self.frozen_at_ns,
            self.calibrated.fitted.state.expires_at_ns,
        )?;
        self.calibrated.predict_core_detailed(
            query,
            model_now_ns,
            self.qualification_support.as_ref(),
            3,
            Some(self.completion_coverage),
        )
    }
    pub fn owner(&self) -> &StructuredOwnerKeyV2 {
        &self.scope().owner
    }
    pub fn scope(&self) -> &StructuredScopeV2 {
        &self.calibrated.fitted.scope
    }
    pub fn domain_signature(&self) -> &[u8; 32] {
        &self.calibrated.fitted.domain_signature
    }
    pub fn algorithm_universe(&self) -> Option<&DeclaredAlgorithmUniverseV1> {
        self.calibrated
            .fitted
            .numerical
            .physical()?
            .contract
            .algorithm_universe
            .as_ref()
    }
    pub fn numerical_family_key(&self) -> Option<&NumericalFamilyKeyV1> {
        self.scope().numerical_family.as_ref()
    }
    pub fn source_contract(&self) -> &StructuredSourceContractV2 {
        &self.calibrated.fitted.source
    }
    pub fn parameters_signature(&self) -> [u8; 32] {
        let mut digest = Sha256::new();
        digest.update(b"ferrum.structured-qualified-state.v2\0");
        digest.update(self.calibrated.parameters_signature());
        if let Some(bank) = &self.calibrated.joint_bank {
            bank.bind(&mut digest, true);
        }
        if let Some(population) = &self.calibrated.fitted.service_window {
            digest.update(b"ferrum.structured-service-window-qualified.v1\0");
            digest.update(
                population.closes[2]
                    .as_ref()
                    .expect("qualified phase close")
                    .signature(),
            );
        }
        if let Some(population) = &self.calibrated.fitted.owner_blocks {
            digest.update(b"ferrum.structured-owner-block-qualified.v1\0");
            digest.update(
                population.closes[2]
                    .as_ref()
                    .expect("qualified phase close")
                    .signature(),
            );
        }
        digest.update((self.qualification_samples as u64).to_le_bytes());
        if let Some(support) = &self.qualification_support {
            digest.update(b"frozen-fit-support-heldout-domain-v1\0");
            support.bind_parameters(&mut digest);
        }
        digest.update(self.frozen_at_ns.to_le_bytes());
        if let Some(physical) = self
            .calibrated
            .fitted
            .numerical
            .physical()
            .filter(|_| self.calibrated.joint_bank.is_none())
        {
            physical.bind_phase(StructuredPhaseV2::Qualification, &mut digest);
        } else if self.calibrated.fitted.exemplar.completion.is_some() {
            self.completion_coverage.bind(&mut digest);
        }
        digest.finalize().into()
    }
}
