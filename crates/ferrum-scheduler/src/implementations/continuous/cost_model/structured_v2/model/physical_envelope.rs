//! Numerical scope is supplied by the original validated projection stamp.
//! Neither the stamp nor this empirical model grants execution authority.
//! This module only learns a bounded empirical coefficient set from real work.
use super::super::nonnegative::{FitCertificate, FitSample, NonNegativeFit};
use super::*;
use ferrum_interfaces::execution_cost::CostWorkloadDomainV1;

mod challenges;
mod diagnostic;
pub(super) mod envelope;
use challenges::ChallengeCoverage;
#[cfg(test)]
mod algorithm_universe_tests;
#[cfg(test)]
mod coverage_extension_tests;
#[cfg(test)]
mod numerical_family_tests;
#[cfg(test)]
mod population_tests;

/// Versioned branch semantics, not a caller-selected list of convenient axes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WorkAxisAndBranchChallengesV1 {
    WorkAxesAndPlainTextBranchesV1,
}

/// Explicit numerical contracts. Residual and Qualification remain independent
/// populations. Envelope strategies bound an empirical coefficient set; fitted
/// residual strategies supply empirical planning estimates, never hardware bounds.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NonNegativePlanningEstimatorV1 {
    #[default]
    CoefficientEnvelopeV1,
    FittedResidualV1,
    /// Uses original Fit equations, including signed combinations, to retain
    /// uncertainty in work directions that the samples do not identify.
    IdentifiedEnvelopeV2,
    /// Identified Fit is immutable; fitted points are usable only after independent
    /// full joint-input cell Residual and Qualification populations.
    IdentifiedFitJointCellsV1,
    /// Immutable identified Fit with independent whole-wave residual calibration.
    /// Completion coordinates represent pre-execution opportunities, including
    /// during fitting and retrospective comparison; actual outcomes stay labels.
    /// This empirical point strategy does not bound the compatible coefficient set.
    IdentifiedFitGlobalResidualV1,
}
impl NonNegativePlanningEstimatorV1 {
    /// This empirical estimator uses only pre-execution completion opportunities.
    /// Actual terminal outcomes remain independent validation and feedback labels.
    pub(super) fn prospective_completion_work(self) -> bool {
        self == Self::IdentifiedFitGlobalResidualV1
    }
    pub fn is_coefficient_envelope(&self) -> bool {
        *self == Self::CoefficientEnvelopeV1
    }
    fn prediction(self, fit: &NonNegativeFit, axes: &[u64]) -> Result<u64> {
        self.prediction_detailed(fit, axes)
            .map_err(StructuredQueryFailureV2::reason)
    }
    pub(super) fn prediction_detailed(
        self,
        fit: &NonNegativeFit,
        axes: &[u64],
    ) -> QueryResult<u64> {
        match self {
            Self::CoefficientEnvelopeV1 => Ok(fit.predict_detailed(axes)?.upper_ns),
            Self::FittedResidualV1 | Self::IdentifiedFitGlobalResidualV1 => {
                fit.predict_fitted_detailed(axes)
            }
            Self::IdentifiedFitJointCellsV1 => Err(StructuredUnknown::WrongProtocol.into()),
            Self::IdentifiedEnvelopeV2 => {
                Ok(fit.predict_identified_envelope_detailed(axes)?.upper_ns)
            }
        }
    }
    fn calibration_prediction(self, fit: &NonNegativeFit, axes: &[u64]) -> Result<u64> {
        match self {
            // Residual observations describe error in the fitted point. The
            // query's separate parameter uncertainty is already in its upper
            // bound and must not itself be learned as temporal drift.
            Self::IdentifiedEnvelopeV2 => fit.predict_fitted(axes),
            Self::IdentifiedFitJointCellsV1 => fit.predict_fitted(axes),
            _ => self.prediction(fit, axes),
        }
    }
}

/// Declared statistical grouping. Original prepared inputs retain their exact
/// owners and physical coordinates under either policy.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StructuredPopulationPolicyV1 {
    #[default]
    ExactOwnerV1,
    HomogeneousOrdinaryDecodeV1,
}
impl StructuredPopulationPolicyV1 {
    pub fn is_exact_owner(&self) -> bool {
        *self == Self::ExactOwnerV1
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NonNegativeEnvelopeContractV1 {
    #[serde(
        default,
        skip_serializing_if = "NonNegativePlanningEstimatorV1::is_coefficient_envelope"
    )]
    pub planning_estimator: NonNegativePlanningEstimatorV1,
    /// Explicit finite subset projection. Omission preserves original axes and bytes.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub algorithm_universe: Option<DeclaredAlgorithmUniverseV1>,
    pub workload_domain: CostWorkloadDomainV1,
    pub settings: super::super::nonnegative::EnvelopeSettings,
    pub challenge: WorkAxisAndBranchChallengesV1,
    #[serde(
        default,
        skip_serializing_if = "StructuredCostTemplatePolicyV1::is_ordered"
    )]
    pub template_policy: StructuredCostTemplatePolicyV1,
    #[serde(
        default,
        skip_serializing_if = "StructuredPopulationPolicyV1::is_exact_owner"
    )]
    pub population_policy: StructuredPopulationPolicyV1,
}
impl NonNegativeEnvelopeContractV1 {
    pub fn validate(&self) -> Result<()> {
        self.settings.validate()?;
        if self.algorithm_universe.as_ref().is_some_and(|u| {
            self.population_policy != StructuredPopulationPolicyV1::HomogeneousOrdinaryDecodeV1
                || self.template_policy != StructuredCostTemplatePolicyV1::InstalledAlgorithmSetV1
                || u.workload_domain_signature() != self.workload_domain.sha256()
        }) {
            return Err(StructuredUnknown::WrongProtocol);
        }
        Ok(())
    }
    /// Map only supported homogeneous decode; prefill/mixed retain exact scopes.
    pub fn project_input(&self, input: StructuredInputV2) -> Result<StructuredInputV2> {
        let Some(universe) = &self.algorithm_universe else {
            return Ok(input);
        };
        match input.numerical_family_key() {
            Ok(_) => input.with_algorithm_universe(universe),
            Err(StructuredUnknown::UnsupportedScope) => Ok(input),
            Err(error) => Err(error),
        }
    }
    pub(super) fn validate_input(&self, input: &StructuredInputV2) -> Result<()> {
        self.validate()?;
        let expected = match (&self.algorithm_universe, input.numerical_family_key()) {
            (Some(u), Ok(_)) => Some(u.signature()),
            (_, Err(StructuredUnknown::UnsupportedScope)) | (None, _) => None,
            (_, Err(error)) => return Err(error),
        };
        if input.algorithm_universe_signature() != expected {
            return Err(StructuredUnknown::WrongDomain);
        }
        if input.physical_domain_signature() != Some(self.workload_domain.sha256())
            || input.cost_template_identity(self.template_policy).is_none()
        {
            return Err(StructuredUnknown::WrongDomain);
        }
        Ok(())
    }
    pub(super) fn bind(&self, digest: &mut Sha256) {
        digest.update(b"ferrum.nonnegative-physical-envelope.contract.v1\0");
        match self.planning_estimator {
            NonNegativePlanningEstimatorV1::CoefficientEnvelopeV1 => {}
            NonNegativePlanningEstimatorV1::FittedResidualV1 => {
                digest.update(b"ferrum.nonnegative-planning-estimator.fitted-residual.v1\0");
            }
            NonNegativePlanningEstimatorV1::IdentifiedEnvelopeV2 => {
                digest.update(b"ferrum.nonnegative-planning-estimator.identified-envelope.v2\0");
            }
            NonNegativePlanningEstimatorV1::IdentifiedFitJointCellsV1 => {
                digest.update(b"ferrum.identified-fit.global-joint-cell-bank.v1\0");
            }
            NonNegativePlanningEstimatorV1::IdentifiedFitGlobalResidualV1 => {
                digest.update(b"ferrum.identified-fit.global-residual.v1\0");
                digest.update(b"ferrum.prospective-completion-work.v1\0");
            }
        }
        digest.update(self.workload_domain.sha256());
        if let Some(universe) = &self.algorithm_universe {
            digest.update(b"ferrum.declared-algorithm-universe.projection.v1\0");
            digest.update(universe.signature());
        }
        if !self.template_policy.is_ordered() {
            digest.update(b"ferrum.nonnegative-template-policy.installed-algorithm-set.v1\0");
        }
        if self.population_policy == StructuredPopulationPolicyV1::HomogeneousOrdinaryDecodeV1 {
            digest.update(b"ferrum.homogeneous-ordinary-decode-population.v1\0");
        }
        digest.update(self.settings.maximum_coordinate_visits.to_le_bytes());
        digest.update((self.settings.maximum_query_certificates as u64).to_le_bytes());
        digest.update([match self.challenge {
            WorkAxisAndBranchChallengesV1::WorkAxesAndPlainTextBranchesV1 => 0,
        }]);
    }
}

pub(super) struct PhysicalEnvelope {
    pub(super) contract: NonNegativeEnvelopeContractV1,
    // Derived only from the original owner-phase declaration, never a query.
    pub(super) phase_support: Option<OwnerPhaseSupportPolicyV1>,
    pub(super) fit: NonNegativeFit,
    fit_error_floor_ns: u64,
    coverage: [Option<ChallengeCoverage>; 3],
}
impl PhysicalEnvelope {
    /// A startup extension may not remove an identified work direction or a
    /// previously challenged branch. Magnitude extrapolation remains governed
    /// by the original physical domain and prediction limits at query time.
    pub(super) fn contains_input_coverage(&self, previous: &Self) -> bool {
        let new = &self.fit.certificate().column_maxima;
        let old = &previous.fit.certificate().column_maxima;
        self.contract == previous.contract
            && new.len() == old.len()
            && new.iter().zip(old).all(|(&n, &o)| o == 0 || n != 0)
            && self
                .coverage
                .iter()
                .zip(&previous.coverage)
                .all(|(n, o)| matches!((n, o), (Some(n), Some(o)) if n.contains_coverage(o)))
    }

    pub(super) fn population_policy(&self) -> StructuredPopulationPolicyV1 {
        self.contract.population_policy
    }

    pub fn fit(
        contract: NonNegativeEnvelopeContractV1,
        samples: &[StructuredNumericObservationV2],
        settings: &StructuredSettingsV2,
        frozen: Option<FitCertificate>,
    ) -> Result<Self> {
        contract.validate()?;
        let rows: Vec<_> = samples
            .iter()
            .map(|sample| {
                contract.validate_input(&sample.input)?;
                if sample.input.cost_template_policy() != contract.template_policy {
                    return Err(StructuredUnknown::WrongProtocol);
                }
                sample.input.validate_actual_completion()?;
                envelope::model_axes(&sample.input, contract.planning_estimator)
            })
            .collect::<Result<_>>()?;
        let numeric: Vec<_> = rows
            .iter()
            .zip(samples)
            .map(|(axes, sample)| FitSample {
                axes,
                wall_ns: sample.wall_ns,
            })
            .collect();
        let fit = match (contract.planning_estimator, frozen) {
            (
                NonNegativePlanningEstimatorV1::IdentifiedEnvelopeV2
                | NonNegativePlanningEstimatorV1::IdentifiedFitJointCellsV1
                | NonNegativePlanningEstimatorV1::IdentifiedFitGlobalResidualV1,
                Some(certificate),
            ) => NonNegativeFit::from_certificate_identified(
                &numeric,
                settings,
                contract.settings,
                certificate,
            )?,
            (
                NonNegativePlanningEstimatorV1::IdentifiedEnvelopeV2
                | NonNegativePlanningEstimatorV1::IdentifiedFitJointCellsV1
                | NonNegativePlanningEstimatorV1::IdentifiedFitGlobalResidualV1,
                None,
            ) => NonNegativeFit::fit_identified(&numeric, settings, contract.settings)?,
            (_, Some(certificate)) => NonNegativeFit::from_certificate(
                &numeric,
                settings,
                contract.settings,
                certificate,
            )?,
            (_, None) => NonNegativeFit::fit(&numeric, settings, contract.settings)?,
        };
        let mut fit_error_floor_ns = 0;
        for sample in &numeric {
            fit_error_floor_ns = fit_error_floor_ns.max(
                sample.wall_ns.saturating_sub(
                    contract
                        .planning_estimator
                        .calibration_prediction(&fit, sample.axes)?,
                ),
            );
        }
        let coverage = ChallengeCoverage::observe_for(samples, contract.planning_estimator)?;
        coverage.validate_branches(samples)?;
        coverage.audit_phase(StructuredPhaseV2::Fit, samples.len());
        Ok(Self {
            contract,
            phase_support: None,
            fit,
            fit_error_floor_ns,
            coverage: [Some(coverage), None, None],
        })
    }
    /// Input-only selection is made on pre-settlement reachable work, so an
    /// EOS outcome cannot decide whether its wave joins the heldout population.
    pub fn membership(&self, input: &StructuredInputV2) -> Result<()> {
        self.contract.validate_input(input)?;
        self.fit
            .check_observed_axes(&envelope::membership_axes(input)?)
    }
    /// Original checked input, with all reachable completion work restored.
    /// No wall, prediction cap or observed EOS/Stop decision enters selection.
    pub fn phase_membership(
        &self,
        input: &StructuredInputV2,
        phase: StructuredPhaseV2,
    ) -> Result<()> {
        self.contract.validate_input(input)?;
        input.validate_actual_completion()?;
        if let Some(completion) = &input.completion {
            let causes = input
                .settled_terminal_causes()
                .ok_or(StructuredUnknown::MissingEvidence)?;
            if causes
                .iter()
                .map(|(position, _)| *position)
                .ne(completion.positions.iter().copied())
            {
                return Err(StructuredUnknown::MissingEvidence);
            }
        }
        let upper = envelope::membership_axes(input)?;
        self.fit.check_observed_axes(&upper)?;
        self.coverage[phase.index()]
            .as_ref()
            .ok_or(StructuredUnknown::MissingEvidence)?
            .authorize_original_input(input, &upper)
    }
    pub fn actual_upper(&self, input: &StructuredInputV2) -> Result<u64> {
        self.contract.validate_input(input)?;
        input.validate_actual_completion()?;
        self.contract.planning_estimator.calibration_prediction(
            &self.fit,
            &envelope::model_axes(input, self.contract.planning_estimator)?,
        )
    }
    pub fn freeze_phase(
        &mut self,
        phase: StructuredPhaseV2,
        samples: &[StructuredNumericObservationV2],
    ) -> Result<()> {
        let i = phase.index();
        if i == 0 || self.coverage[i].is_some() {
            return Err(StructuredUnknown::PhaseLeakage);
        }
        let coverage = ChallengeCoverage::observe_for(samples, self.contract.planning_estimator)?;
        coverage.validate_branches(samples)?;
        if self.phase_support.is_none() {
            coverage.validate_positive_axes(&self.fit.certificate().column_maxima)?;
        }
        coverage.audit_phase(phase, samples.len());
        self.coverage[i] = Some(coverage);
        Ok(())
    }
    pub fn bounds(&self, query: &StructuredQueryV2, required_phases: usize) -> Result<(u64, u64)> {
        self.bounds_detailed(query, required_phases)
            .map_err(StructuredQueryFailureV2::reason)
    }
    /// Shared prospective coverage predicate. No numerical estimator, phase
    /// sample wall, expiry, clock mapping or feedback state is evaluated here.
    pub(super) fn authorized_query_upper(
        &self,
        query: &StructuredQueryV2,
        required_phases: usize,
        before_settlement: bool,
    ) -> QueryResult<Vec<u64>> {
        self.contract.validate_input(&query.input)?;
        let upper = envelope::model_query_upper(
            query,
            &self.contract.workload_domain,
            self.contract.planning_estimator,
            before_settlement,
        )?;
        for coverage in self.coverage.iter().take(required_phases) {
            let coverage = coverage
                .as_ref()
                .ok_or(StructuredUnknown::QualificationCoverage)?;
            if before_settlement {
                coverage.authorize_prospective_query(query, &upper)?;
            } else {
                coverage.authorize_detailed(query, &upper)?;
            }
        }
        Ok(upper)
    }
    pub fn bounds_detailed(
        &self,
        query: &StructuredQueryV2,
        required_phases: usize,
    ) -> QueryResult<(u64, u64)> {
        let upper = self.authorized_query_upper(query, required_phases, false)?;
        let upper_ns = self
            .contract
            .planning_estimator
            .prediction_detailed(&self.fit, &upper)?;
        if tracing::enabled!(target: "ferrum::slo_transaction", tracing::Level::TRACE) {
            if let Some(explanation) = self.fit.diagnose_bound(&upper) {
                tracing::trace!(
                    target: "ferrum::slo_transaction",
                    owner = ?query.input.owner(),
                    certificate_index = explanation.certificate_index,
                    limiting_axis = explanation.limiting_axis,
                    axis_label = %diagnostic::axis_label(&query.input, explanation.limiting_axis),
                    input_value = explanation.input_value,
                    fit_maximum = explanation.fit_maximum,
                    certificate_axis = %explanation.certificate_axis,
                    certificate_bound_ns = %explanation.certificate_bound_ns,
                    certified_upper_ns = explanation.upper_ns,
                    planning_estimator = ?self.contract.planning_estimator,
                    empirical_fitted_ns = ?self.fit.predict_fitted(&upper).ok(),
                    base_planning_ns = upper_ns,
                    fit_epsilon_ns = self.fit.certificate().epsilon_ns,
                    fit_input_digest = ?self.fit.certificate().input_digest,
                    query_upper_axes = ?upper,
                    certificate_axes = ?explanation.certificate_axes,
                    fit_column_maxima = ?self.fit.certificate().column_maxima,
                    fit_coefficient_words = ?self.fit.certificate().coefficient_words,
                    "SLO physical cost envelope limiting work axis"
                );
            }
        }
        // Neither the feasible fitted point nor its independent residual
        // calibration supplies a positive execution-time lower bound.
        Ok((0, upper_ns))
    }
    pub fn certificate(&self) -> &FitCertificate {
        self.fit.certificate()
    }
    pub fn fit_error_floor_ns(&self) -> u64 {
        self.fit_error_floor_ns
    }
    pub fn rank(&self) -> usize {
        self.fit.certificate().geometry_rank
    }
    pub fn bind_fit(&self, digest: &mut Sha256) {
        self.contract.bind(digest);
        match self.phase_support {
            Some(OwnerPhaseSupportPolicyV1::FrozenInputIntersectionV1) => {
                digest.update(b"ferrum.physical-phase-support.frozen-input-intersection.v1\0");
            }
            Some(OwnerPhaseSupportPolicyV1::FrozenInputIntersectionV2) => {
                digest.update(b"ferrum.physical-phase-support.frozen-input-intersection.v2\0");
            }
            None => {}
        }
        self.fit.bind_parameters(digest);
        digest.update(self.fit_error_floor_ns.to_le_bytes());
        self.coverage[0]
            .as_ref()
            .expect("frozen fit coverage")
            .bind(digest);
    }
    pub fn bind_phase(&self, phase: StructuredPhaseV2, digest: &mut Sha256) {
        digest.update(b"ferrum.nonnegative-physical-envelope.challenge-phase.v1\0");
        digest.update([phase.index() as u8]);
        self.coverage[phase.index()]
            .as_ref()
            .expect("frozen phase coverage")
            .bind(digest);
    }
    pub fn retained_heap_bytes(&self) -> Option<usize> {
        let mut bytes = self.fit.retained_heap_bytes()?.checked_add(
            self.contract
                .algorithm_universe
                .as_ref()
                .map_or(Some(0), DeclaredAlgorithmUniverseV1::retained_payload_bytes)?,
        )?;
        for coverage in self.coverage.iter().flatten() {
            bytes = bytes.checked_add(coverage.retained_heap_bytes()?)?;
        }
        Some(bytes)
    }
}
