//! Separately versioned, whole-wave numerical core. This is not a profile loader,
//! a settlement-receipt adapter, or execution authority. Public Rust numerical
//! APIs do not attest the provenance of caller-supplied measurements. The engine
//! live bridge separately verifies its original private settlement receipts.
//!
//! A shared decoder scope is qualified as a whole after fitting and independent
//! residual calibration. Physical terminal positions are audited, never hashed
//! into separate sample buckets. Qualification is an empirical challenge, not a
//! distribution-free p99 guarantee and not a serving-SLO result.
use super::{CostBoundary, ExecutionFingerprint, WaveObservationOutcome};
use std::collections::BTreeSet;

mod fit;
mod input;
mod support;
use fit::RowSpaceFit;
pub use input::{StructuredInputV1, StructuredScopeV1};
use support::JointSupport;
#[cfg(test)]
mod projection_tests;
#[cfg(test)]
mod tests;

pub const MODEL_REVISION: &str = "structured_whole_wave_rowspace_single_length_v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StructuredUnknown {
    MissingEvidence,
    UnsupportedScope,
    InvalidInput,
    InvalidSettings,
    WrongDomain,
    WrongSource,
    WrongProtocol,
    WrongFingerprint,
    PhaseLeakage,
    DuplicateRecord,
    InvalidSample,
    Capacity,
    Clock,
    Stale,
    InsufficientSamples,
    InsufficientRedundancy,
    IncompletePhasePopulation,
    JointSupport,
    UnidentifiedDirection,
    IllConditioned,
    Numerical,
    QualificationCoverage,
    QualificationUnderestimate,
}
type Result<T> = std::result::Result<T, StructuredUnknown>;

/// Resource/identifiability limits must be frozen before collecting a new
/// source. The empirical residual quantile is fixed at .99 in this revision.
#[derive(Debug, Clone)]
pub struct StructuredSettingsV1 {
    pub min_phase_samples: usize,
    pub min_fit_redundancy: usize,
    pub max_phase_samples: usize,
    pub max_axes: usize,
    pub max_rank: usize,
    pub max_wave_ns: u64,
    pub max_sample_age_ns: u64,
    pub static_margin_ns: u64,
}
impl Default for StructuredSettingsV1 {
    fn default() -> Self {
        Self {
            min_phase_samples: 8,
            min_fit_redundancy: 4,
            max_phase_samples: 256,
            max_axes: 1024,
            max_rank: 32,
            max_wave_ns: 60_000_000_000,
            max_sample_age_ns: 300_000_000_000,
            static_margin_ns: 100_000,
        }
    }
}
impl StructuredSettingsV1 {
    fn validate(&self) -> Result<()> {
        if self.min_phase_samples < 8
            || self.min_fit_redundancy < 4
            || self.min_fit_redundancy >= self.max_phase_samples
            || self.max_phase_samples < self.min_phase_samples
            || self.max_phase_samples > 4096
            || self.max_axes == 0
            || self.max_axes > 4096
            || self.max_rank == 0
            || self.max_rank > 128
            || self.max_wave_ns == 0
            || self.max_wave_ns > (1 << 53)
            || self.max_sample_age_ns == 0
            || self.static_margin_ns > self.max_wave_ns
        {
            return Err(StructuredUnknown::InvalidSettings);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy)]
/// Dense accepted ordinals of one predeclared scope and one capture protocol.
/// Multi-domain filtered/re-numbered populations are not this protocol.
pub struct StructuredPartitionV1 {
    pub source: [u8; 32],
    pub protocol: [u8; 32],
    pub fit_through: u64,
    pub residual_through: u64,
    pub qualification_through: u64,
}
impl StructuredPartitionV1 {
    fn validate(&self) -> Result<()> {
        if self.source == [0; 32]
            || self.protocol == [0; 32]
            || self.fit_through == 0
            || self.residual_through <= self.fit_through
            || self.qualification_through <= self.residual_through
        {
            return Err(StructuredUnknown::PhaseLeakage);
        }
        Ok(())
    }
}

/// Numerical input. A production adapter must first verify
/// the real qualified same-call settlement receipt; these fields alone cannot
/// establish it. No serde/profile ingestion is provided here.
pub struct StructuredNumericObservationV1 {
    pub source: [u8; 32],
    pub protocol: [u8; 32],
    pub ordinal: u64,
    pub call_id: u64,
    pub fingerprint: ExecutionFingerprint,
    pub input: StructuredInputV1,
    pub boundary: CostBoundary,
    pub outcome: WaveObservationOutcome,
    pub observed_at_ns: u64,
    pub wall_ns: u64,
}

struct PhaseState {
    calls: BTreeSet<u64>,
    ordinals: BTreeSet<u64>,
    latest_observed_at_ns: u64,
    expires_at_ns: u64,
}
impl PhaseState {
    fn new() -> Self {
        Self {
            calls: BTreeSet::new(),
            ordinals: BTreeSet::new(),
            latest_observed_at_ns: 0,
            expires_at_ns: u64::MAX,
        }
    }
    fn observe(
        &mut self,
        sample: &StructuredNumericObservationV1,
        fingerprint: &ExecutionFingerprint,
        settings: &StructuredSettingsV1,
        partition: StructuredPartitionV1,
        range: std::ops::RangeInclusive<u64>,
        phase_started_at_ns: u64,
        now: u64,
    ) -> Result<()> {
        if sample.source != partition.source {
            return Err(StructuredUnknown::WrongSource);
        }
        if sample.protocol != partition.protocol {
            return Err(StructuredUnknown::WrongProtocol);
        }
        if &sample.fingerprint != fingerprint {
            return Err(StructuredUnknown::WrongFingerprint);
        }
        if !range.contains(&sample.ordinal) {
            return Err(StructuredUnknown::PhaseLeakage);
        }
        if sample.call_id == 0
            || sample.wall_ns == 0
            || sample.wall_ns > settings.max_wave_ns
            || sample.boundary != CostBoundary::PreparationToHostSettledV1
            || sample.outcome != WaveObservationOutcome::Completed
        {
            return Err(StructuredUnknown::InvalidSample);
        }
        if !self.calls.insert(sample.call_id) || !self.ordinals.insert(sample.ordinal) {
            return Err(StructuredUnknown::DuplicateRecord);
        }
        let age = now
            .checked_sub(sample.observed_at_ns)
            .ok_or(StructuredUnknown::Clock)?;
        if sample.observed_at_ns < self.latest_observed_at_ns
            || sample.observed_at_ns < phase_started_at_ns
        {
            return Err(StructuredUnknown::Clock);
        }
        if age > settings.max_sample_age_ns {
            return Err(StructuredUnknown::Stale);
        }
        let expires = sample
            .observed_at_ns
            .checked_add(settings.max_sample_age_ns)
            .ok_or(StructuredUnknown::Clock)?;
        self.latest_observed_at_ns = sample.observed_at_ns;
        self.expires_at_ns = self.expires_at_ns.min(expires);
        Ok(())
    }
}

pub struct FittedStructuredModelV1 {
    fingerprint: ExecutionFingerprint,
    settings: StructuredSettingsV1,
    partition: StructuredPartitionV1,
    exemplar: StructuredInputV1,
    fit: RowSpaceFit,
    fit_support: JointSupport,
    fit_samples: usize,
    state: PhaseState,
    frozen_at_ns: u64,
}
impl FittedStructuredModelV1 {
    /// `now` is the original live fit-freeze time in the observation clock.
    /// Offline replay must retain that time, not substitute its read time.
    pub fn fit(
        fingerprint: ExecutionFingerprint,
        settings: StructuredSettingsV1,
        partition: StructuredPartitionV1,
        samples: &[StructuredNumericObservationV1],
        now: u64,
    ) -> Result<Self> {
        settings.validate()?;
        partition.validate()?;
        phase_count(samples, &settings)?;
        complete_phase_population(samples, 1..=partition.fit_through)?;
        let exemplar = samples[0].input.clone();
        exemplar.validate(&settings)?;
        let mut state = PhaseState::new();
        for sample in samples {
            state.observe(
                sample,
                &fingerprint,
                &settings,
                partition,
                1..=partition.fit_through,
                0,
                now,
            )?;
            exemplar.same_domain(&sample.input)?;
            sample.input.validate(&settings)?;
        }
        let fit = RowSpaceFit::fit(samples, &settings)?;
        let fit_support = JointSupport::new(samples.iter().map(|s| s.input.support.as_slice()))?;
        Ok(Self {
            fingerprint,
            settings,
            partition,
            exemplar,
            fit,
            fit_support,
            fit_samples: samples.len(),
            state,
            frozen_at_ns: now,
        })
    }

    /// Freeze the independent residual phase at its original live clock time.
    pub fn calibrate(
        mut self,
        samples: &[StructuredNumericObservationV1],
        now: u64,
    ) -> Result<CalibratedStructuredModelV1> {
        phase_count(samples, &self.settings)?;
        complete_phase_population(
            samples,
            self.partition.fit_through + 1..=self.partition.residual_through,
        )?;
        check_time(now, self.frozen_at_ns, self.state.expires_at_ns)?;
        let mut residuals = Vec::with_capacity(samples.len());
        for sample in samples {
            self.state.observe(
                sample,
                &self.fingerprint,
                &self.settings,
                self.partition,
                self.partition.fit_through + 1..=self.partition.residual_through,
                self.frozen_at_ns,
                now,
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
        // Integer nearest-rank ceil(.99*n), never an interpolated/lowered tail.
        let rank = (99 * residuals.len()).div_ceil(100);
        let residual_ns = residuals[rank - 1];
        let residual_support =
            JointSupport::new(samples.iter().map(|s| s.input.support.as_slice()))?;
        self.frozen_at_ns = now;
        Ok(CalibratedStructuredModelV1 {
            fitted: self,
            residual_ns,
            residual_support,
            residual_samples: samples.len(),
        })
    }
}

pub struct CalibratedStructuredModelV1 {
    fitted: FittedStructuredModelV1,
    residual_ns: u64,
    residual_support: JointSupport,
    residual_samples: usize,
}
impl CalibratedStructuredModelV1 {
    fn predict_core(&self, input: &StructuredInputV1, now: u64) -> Result<StructuredPredictionV1> {
        let f = &self.fitted;
        check_time(now, f.frozen_at_ns, f.state.expires_at_ns)?;
        f.exemplar.same_domain(input)?;
        input.validate(&f.settings)?;
        if !f.fit_support.contains(&input.support)
            || !self.residual_support.contains(&input.support)
        {
            return Err(StructuredUnknown::JointSupport);
        }
        let fitted_ns = f.fit.predict(&input.basis)?;
        let planning_ns = fitted_ns
            .checked_add(self.residual_ns)
            .and_then(|n| n.checked_add(f.settings.static_margin_ns))
            .filter(|n| *n <= f.settings.max_wave_ns)
            .ok_or(StructuredUnknown::Numerical)?;
        Ok(StructuredPredictionV1 {
            fitted_ns,
            residual_ns: self.residual_ns,
            planning_ns,
            valid_until_ns: f.state.expires_at_ns,
            fit_samples: f.fit_samples,
            residual_samples: self.residual_samples,
            identified_rank: f.fit.rank(),
        })
    }

    /// Consumes one frozen candidate and the complete predeclared qualification
    /// phase. An uncovered position, unknown point, or underestimate fails the
    /// entire scope; no per-position selection or residual refit occurs here.
    pub fn qualify(
        mut self,
        samples: &[StructuredNumericObservationV1],
        now: u64,
    ) -> Result<QualifiedStructuredModelV1> {
        phase_count(samples, &self.fitted.settings)?;
        complete_phase_population(
            samples,
            self.fitted.partition.residual_through + 1
                ..=self.fitted.partition.qualification_through,
        )?;
        let rows = self.fitted.exemplar.scope.rows();
        let mut covered = vec![false; rows + 1]; // 0: no boundary; p+1: physical p.
        let partition = self.fitted.partition;
        for sample in samples {
            self.fitted.state.observe(
                sample,
                &self.fitted.fingerprint,
                &self.fitted.settings,
                partition,
                partition.residual_through + 1..=partition.qualification_through,
                self.fitted.frozen_at_ns,
                now,
            )?;
            let prediction = self.predict_core(&sample.input, now)?;
            if sample.wall_ns > prediction.planning_ns {
                return Err(StructuredUnknown::QualificationUnderestimate);
            }
            covered[sample.input.terminal_position.map_or(0, |p| p + 1)] = true;
        }
        // The dense population was checked before evaluating any point. Now
        // require the whole declared position scope to have been challenged.
        if covered.iter().any(|seen| !seen) {
            return Err(StructuredUnknown::QualificationCoverage);
        }
        self.fitted.frozen_at_ns = now;
        Ok(QualifiedStructuredModelV1 {
            calibrated: self,
            qualification_samples: samples.len(),
        })
    }
}

pub struct QualifiedStructuredModelV1 {
    calibrated: CalibratedStructuredModelV1,
    pub qualification_samples: usize,
}
impl QualifiedStructuredModelV1 {
    pub fn predict(
        &self,
        fingerprint: &ExecutionFingerprint,
        input: &StructuredInputV1,
        now: u64,
    ) -> Result<StructuredPredictionV1> {
        if fingerprint != &self.calibrated.fitted.fingerprint {
            return Err(StructuredUnknown::WrongFingerprint);
        }
        self.calibrated.predict_core(input, now)
    }
}
#[derive(Debug, Clone, Copy)]
pub struct StructuredPredictionV1 {
    pub fitted_ns: u64,
    pub residual_ns: u64,
    pub planning_ns: u64,
    pub valid_until_ns: u64,
    pub fit_samples: usize,
    pub residual_samples: usize,
    pub identified_rank: usize,
}
fn phase_count(
    samples: &[StructuredNumericObservationV1],
    settings: &StructuredSettingsV1,
) -> Result<()> {
    if samples.len() < settings.min_phase_samples {
        return Err(StructuredUnknown::InsufficientSamples);
    }
    if samples.len() > settings.max_phase_samples {
        return Err(StructuredUnknown::Capacity);
    }
    Ok(())
}

/// This first single-domain protocol declares dense global accepted ranges.
/// A later multi-domain protocol needs an immutable population manifest; it
/// must not filter failed/slow rows and then renumber the survivors.
fn complete_phase_population(
    samples: &[StructuredNumericObservationV1],
    range: std::ops::RangeInclusive<u64>,
) -> Result<()> {
    let expected = range
        .end()
        .checked_sub(*range.start())
        .and_then(|n| n.checked_add(1))
        .ok_or(StructuredUnknown::PhaseLeakage)?;
    if samples.iter().any(|s| !range.contains(&s.ordinal)) {
        return Err(StructuredUnknown::PhaseLeakage);
    }
    let ordinals: BTreeSet<_> = samples.iter().map(|s| s.ordinal).collect();
    if ordinals.len() != samples.len() {
        return Err(StructuredUnknown::DuplicateRecord);
    }
    if ordinals.len() as u64 != expected {
        return Err(StructuredUnknown::IncompletePhasePopulation);
    }
    Ok(())
}
fn check_time(now: u64, frozen_at: u64, expires_at: u64) -> Result<()> {
    if now < frozen_at {
        return Err(StructuredUnknown::Clock);
    }
    if now > expires_at {
        return Err(StructuredUnknown::Stale);
    }
    Ok(())
}
