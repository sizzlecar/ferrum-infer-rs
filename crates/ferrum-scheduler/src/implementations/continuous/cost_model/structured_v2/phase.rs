//! Complete reserved populations on the original source clock.
use super::*;
use std::collections::BTreeSet;

pub(super) struct PhaseState {
    calls: BTreeSet<u64>,
    last_fifo: u64,
    last_member: u64,
    last_offered: u64,
    last_observed: u64,
    pub expires_at_ns: u64,
}
impl PhaseState {
    pub fn bind_parameters(&self, digest: &mut sha2::Sha256) {
        use sha2::Digest;
        for value in [
            self.last_fifo,
            self.last_member,
            self.last_offered,
            self.last_observed,
            self.expires_at_ns,
            self.calls.len() as u64,
        ] {
            digest.update(value.to_le_bytes());
        }
        for call in &self.calls {
            digest.update(call.to_le_bytes());
        }
    }
    pub fn new() -> Self {
        Self {
            calls: BTreeSet::new(),
            last_fifo: 0,
            last_member: 0,
            last_offered: 0,
            last_observed: 0,
            expires_at_ns: u64::MAX,
        }
    }
    pub fn observe(
        &mut self,
        sample: &StructuredNumericObservationV2,
        fp: &ExecutionFingerprint,
        settings: &StructuredSettingsV2,
        source: &StructuredSourceContractV2,
        phase: StructuredPhaseV2,
        phase_started: u64,
        now: u64,
    ) -> Result<()> {
        if sample.source != source.capture_identity {
            return Err(StructuredUnknown::WrongSource);
        }
        if sample.protocol != source.protocol {
            return Err(StructuredUnknown::WrongProtocol);
        }
        if &sample.fingerprint != fp {
            return Err(StructuredUnknown::WrongFingerprint);
        }
        if sample.membership.rule_signature != source.membership_rule
            || sample.membership.phase != phase
        {
            return Err(StructuredUnknown::PhaseLeakage);
        }
        if sample.call_id == 0
            || sample.ordinal == 0
            || sample.membership.offered_ordinal == 0
            || sample.wall_ns == 0
            || sample.wall_ns > settings.max_wave_ns
            || sample.boundary != CostBoundary::PreparationToHostSettledV1
            || sample.outcome != WaveObservationOutcome::Completed
        {
            return Err(StructuredUnknown::InvalidSample);
        }
        if sample.ordinal <= self.last_fifo
            || sample.membership.member_ordinal <= self.last_member
            || sample.membership.offered_ordinal <= self.last_offered
            || !self.calls.insert(sample.call_id)
        {
            return Err(StructuredUnknown::DuplicateRecord);
        }
        let age = now
            .checked_sub(sample.observed_at_ns)
            .ok_or(StructuredUnknown::Clock)?;
        if sample.observed_at_ns < self.last_observed || sample.observed_at_ns < phase_started {
            return Err(StructuredUnknown::Clock);
        }
        if age > settings.max_sample_age_ns {
            return Err(StructuredUnknown::Stale);
        }
        let expires = sample
            .observed_at_ns
            .checked_add(settings.max_sample_age_ns)
            .ok_or(StructuredUnknown::Clock)?;
        self.expires_at_ns = self.expires_at_ns.min(expires);
        self.last_fifo = sample.ordinal;
        self.last_member = sample.membership.member_ordinal;
        self.last_offered = sample.membership.offered_ordinal;
        self.last_observed = sample.observed_at_ns;
        Ok(())
    }
}
impl StructuredSourceContractV2 {
    pub(super) fn validate(&self, settings: &StructuredSettingsV2) -> Result<()> {
        if [
            self.capture_identity,
            self.protocol,
            self.membership_rule,
            self.cohort_manifest,
        ]
        .contains(&[0; 32])
            || self
                .phase_members
                .iter()
                .any(|n| *n < settings.min_phase_samples || *n > settings.max_phase_samples)
        {
            return Err(StructuredUnknown::InvalidSettings);
        }
        Ok(())
    }
    pub(super) fn complete_population(
        &self,
        samples: &[StructuredNumericObservationV2],
        phase: StructuredPhaseV2,
    ) -> Result<()> {
        let i = phase.index();
        if samples.len() != self.phase_members[i] {
            return Err(StructuredUnknown::IncompletePhasePopulation);
        }
        let begin = self.phase_members[..i].iter().sum::<usize>() as u64 + 1;
        let end = begin + self.phase_members[i] as u64;
        let mut found = BTreeSet::new();
        for s in samples {
            if s.membership.phase != phase
                || s.membership.member_ordinal < begin
                || s.membership.member_ordinal >= end
            {
                return Err(StructuredUnknown::PhaseLeakage);
            }
            if !found.insert(s.membership.member_ordinal) {
                return Err(StructuredUnknown::DuplicateRecord);
            }
        }
        if found.len() != self.phase_members[i] {
            return Err(StructuredUnknown::IncompletePhasePopulation);
        }
        Ok(())
    }
}
impl StructuredPhaseV2 {
    pub(super) fn index(self) -> usize {
        match self {
            Self::Fit => 0,
            Self::Residual => 1,
            Self::Qualification => 2,
        }
    }
}
pub(super) fn check_time(now: u64, frozen: u64, expires: u64) -> Result<()> {
    if now < frozen {
        return Err(StructuredUnknown::Clock);
    }
    if now > expires {
        return Err(StructuredUnknown::Stale);
    }
    Ok(())
}
