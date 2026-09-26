//! Immutable phase transitions. These numerical APIs do not attest live receipts.
use super::envelope::PendingGenerators;
use super::fit::{FitRow, RowSpaceFit};
use super::phase::{check_time, PhaseState};
use super::support::JointSupport;
use super::*;
use sha2::{Digest, Sha256};

pub struct FittedStructuredModelV2 {
    fingerprint: ExecutionFingerprint,
    settings: StructuredSettingsV2,
    scope: StructuredScopeV2,
    source: StructuredSourceContractV2,
    exemplar: StructuredInputV2,
    fit: RowSpaceFit,
    fit_support: JointSupport,
    generators: PendingGenerators,
    fit_samples: usize,
    state: PhaseState,
    frozen_at_ns: u64,
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
        samples[0].input.validate(&settings)?;
        let exemplar = samples[0].input.clone();
        if exemplar.owner() != &scope.owner {
            return Err(StructuredUnknown::WrongDomain);
        }
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
            exemplar.same_domain(&sample.input)?;
            sample.input.validate(&settings)?;
        }
        let rows = samples
            .iter()
            .map(|s| FitRow {
                basis: &s.input.basis,
                wall_ns: s.wall_ns,
            })
            .collect::<Vec<_>>();
        let fit = RowSpaceFit::fit(&rows, &settings)?;
        let fit_support = JointSupport::new(samples.iter().map(|s| s.input.support.as_slice()))?;
        let generators = PendingGenerators::new(&fit, &exemplar)?;
        Ok(Self {
            fingerprint,
            settings,
            scope,
            source,
            exemplar,
            fit,
            fit_support,
            generators,
            fit_samples: samples.len(),
            state,
            frozen_at_ns,
        })
    }
    pub fn parameters_signature(&self) -> [u8; 32] {
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
            self.exemplar.domain,
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
        self.scope.bind_parameters(&mut digest);
        self.state.bind_parameters(&mut digest);
        self.fit.bind_parameters(&mut digest);
        self.fit_support.bind_parameters(&mut digest);
        digest.finalize().into()
    }
    /// Consumes the candidate. Failed or omitted members cannot be retried by
    /// retaining this state and replacing a slow observation with another one.
    pub fn calibrate(
        mut self,
        samples: &[StructuredNumericObservationV2],
        frozen_at_ns: u64,
    ) -> Result<CalibratedStructuredModelV2> {
        self.source
            .complete_population(samples, StructuredPhaseV2::Residual)?;
        check_time(frozen_at_ns, self.frozen_at_ns, self.state.expires_at_ns)?;
        let mut residuals = Vec::with_capacity(samples.len());
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
            self.exemplar.same_domain(&sample.input)?;
            sample.input.validate(&self.settings)?;
            if !self.fit_support.contains(&sample.input.support) {
                return Err(StructuredUnknown::JointSupport);
            }
            let fitted = self.fit.predict(&sample.input.basis)?;
            residuals.push(sample.wall_ns.saturating_sub(fitted));
        }
        residuals.sort_unstable();
        // Fixed empirical nearest-rank p99. Qualification never changes this.
        let residual_ns = residuals[(99 * residuals.len()).div_ceil(100) - 1];
        let residual_support =
            JointSupport::new(samples.iter().map(|s| s.input.support.as_slice()))?;
        Ok(CalibratedStructuredModelV2 {
            fitted: self,
            residual_ns,
            residual_support,
            residual_samples: samples.len(),
            frozen_at_ns,
        })
    }
}

pub struct CalibratedStructuredModelV2 {
    fitted: FittedStructuredModelV2,
    residual_ns: u64,
    residual_support: JointSupport,
    residual_samples: usize,
    frozen_at_ns: u64,
}
impl CalibratedStructuredModelV2 {
    fn uncertainty(&self) -> StructuredUncertaintyV2 {
        let fit_error_floor_ns = self.fitted.fit.fit_error_floor_ns();
        StructuredUncertaintyV2 {
            fit_error_floor_ns,
            residual_ns: self.residual_ns,
            effective_residual_ns: fit_error_floor_ns.max(self.residual_ns),
            static_margin_ns: self.fitted.settings.static_margin_ns,
        }
    }
    pub fn parameters_signature(&self) -> [u8; 32] {
        let mut digest = Sha256::new();
        digest.update(b"ferrum.structured-calibrated-state.v2\0");
        digest.update(self.fitted.parameters_signature());
        for value in [
            self.residual_ns,
            self.residual_samples as u64,
            self.frozen_at_ns,
        ] {
            digest.update(value.to_le_bytes());
        }
        self.residual_support.bind_parameters(&mut digest);
        digest.finalize().into()
    }
    fn predict_core(&self, query: &StructuredQueryV2, now: u64) -> Result<StructuredPredictionV2> {
        let f = &self.fitted;
        check_time(now, self.frozen_at_ns, f.state.expires_at_ns)?;
        f.exemplar.same_domain(&query.input)?;
        query.input.validate(&f.settings)?;
        f.scope.authorize(query)?;
        let bounds = f
            .generators
            .bounds(&f.fit, &query.input, query.pending.as_ref())?;
        if !f
            .fit_support
            .contains_envelope(&bounds.support_lower, &bounds.support_upper)
            || !self
                .residual_support
                .contains_envelope(&bounds.support_lower, &bounds.support_upper)
        {
            return Err(StructuredUnknown::JointSupport);
        }
        let uncertainty = self.uncertainty();
        let planning_ns = bounds
            .upper_ns
            .checked_add(uncertainty.effective_residual_ns)
            .and_then(|v| v.checked_add(f.settings.static_margin_ns))
            .filter(|v| *v <= f.settings.max_wave_ns)
            .ok_or(StructuredUnknown::Numerical)?;
        Ok(StructuredPredictionV2 {
            fitted_lower_ns: bounds.lower_ns,
            fitted_upper_ns: bounds.upper_ns,
            residual_ns: uncertainty.residual_ns,
            fit_error_floor_ns: uncertainty.fit_error_floor_ns,
            effective_residual_ns: uncertainty.effective_residual_ns,
            planning_ns,
            valid_until_ns: f.state.expires_at_ns,
            fit_samples: f.fit_samples,
            residual_samples: self.residual_samples,
            identified_rank: f.fit.rank(),
        })
    }
    /// Challenge every predeclared member and coverage requirement, using the
    /// already frozen fit and residual. No position/count receives a new fit.
    pub fn qualify(
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
            let prediction = self.predict_core(
                &StructuredQueryV2::exact(sample.input.clone()),
                frozen_at_ns,
            )?;
            if sample.wall_ns > prediction.planning_ns {
                return Err(StructuredUnknown::QualificationUnderestimate);
            }
        }
        self.fitted.scope.qualify_coverage(samples)?;
        Ok(QualifiedStructuredModelV2 {
            calibrated: self,
            qualification_samples: samples.len(),
            frozen_at_ns,
        })
    }
}

pub struct QualifiedStructuredModelV2 {
    calibrated: CalibratedStructuredModelV2,
    pub qualification_samples: usize,
    frozen_at_ns: u64,
}
impl QualifiedStructuredModelV2 {
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
        if fingerprint != &self.calibrated.fitted.fingerprint {
            return Err(StructuredUnknown::WrongFingerprint);
        }
        check_time(
            model_now_ns,
            self.frozen_at_ns,
            self.calibrated.fitted.state.expires_at_ns,
        )?;
        self.calibrated.predict_core(query, model_now_ns)
    }
    pub fn owner(&self) -> &StructuredOwnerKeyV2 {
        &self.scope().owner
    }
    pub fn scope(&self) -> &StructuredScopeV2 {
        &self.calibrated.fitted.scope
    }
    pub fn domain_signature(&self) -> &[u8; 32] {
        &self.calibrated.fitted.exemplar.domain
    }
    pub fn source_contract(&self) -> &StructuredSourceContractV2 {
        &self.calibrated.fitted.source
    }
    pub fn parameters_signature(&self) -> [u8; 32] {
        let mut digest = Sha256::new();
        digest.update(b"ferrum.structured-qualified-state.v2\0");
        digest.update(self.calibrated.parameters_signature());
        digest.update((self.qualification_samples as u64).to_le_bytes());
        digest.update(self.frozen_at_ns.to_le_bytes());
        digest.finalize().into()
    }
}
