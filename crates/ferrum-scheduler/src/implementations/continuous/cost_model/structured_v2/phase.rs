//! Complete reserved populations on the original source clock.
use super::*;
use std::collections::BTreeSet;

pub(super) struct PhaseState {
    calls: Calls,
    last_fifo: u64,
    last_member: u64,
    last_offered: u64,
    last_observed: u64,
    pub expires_at_ns: u64,
}

enum Calls {
    Collecting(Vec<u64>),
    Frozen(Box<[u64]>),
}
impl Calls {
    fn len(&self) -> usize {
        match self {
            Self::Collecting(calls) => calls.len(),
            Self::Frozen(calls) => calls.len(),
        }
    }
    fn for_each(&self, mut visit: impl FnMut(u64)) {
        match self {
            Self::Collecting(calls) => calls.iter().copied().for_each(&mut visit),
            Self::Frozen(calls) => calls.iter().copied().for_each(&mut visit),
        }
    }
    fn insert(&mut self, call: u64) -> Result<bool> {
        match self {
            Self::Collecting(calls) => {
                // Call IDs usually increase. Preserve the original sorted
                // parameter binding without opaque tree-node allocations.
                if calls.last().is_none_or(|last| *last < call) {
                    calls.push(call);
                    return Ok(true);
                }
                match calls.binary_search(&call) {
                    Ok(_) => Ok(false),
                    Err(index) => {
                        calls.insert(index, call);
                        Ok(true)
                    }
                }
            }
            Self::Frozen(_) => Err(StructuredUnknown::WrongProtocol),
        }
    }
    fn freeze(&mut self) -> Result<()> {
        if matches!(self, Self::Frozen(_)) {
            return Ok(());
        }
        let Self::Collecting(calls) = std::mem::replace(self, Self::Frozen(Box::new([]))) else {
            unreachable!("checked collecting state")
        };
        // No second collection: sorted order and parameter bytes are unchanged.
        *self = Self::Frozen(calls.into_boxed_slice());
        Ok(())
    }
    fn retained_payload_bytes(&self) -> Option<usize> {
        match self {
            Self::Collecting(calls) => calls.capacity().checked_mul(std::mem::size_of::<u64>()),
            Self::Frozen(calls) => calls.len().checked_mul(std::mem::size_of::<u64>()),
        }
    }
    fn frozen_payload_bytes(&self) -> Option<usize> {
        match self {
            Self::Collecting(_) => None,
            Self::Frozen(calls) => calls.len().checked_mul(std::mem::size_of::<u64>()),
        }
    }
}
impl PhaseState {
    pub(super) fn retained_heap_bytes(&self) -> Option<usize> {
        self.calls.retained_payload_bytes()
    }
    pub(super) fn freeze_calls(&mut self) -> Result<()> {
        self.calls.freeze()
    }
    pub(super) fn frozen_heap_bytes(&self) -> Option<usize> {
        self.calls.frozen_payload_bytes()
    }
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
        self.calls.for_each(|call| {
            digest.update(call.to_le_bytes());
        });
    }
    pub fn new() -> Self {
        Self {
            calls: Calls::Collecting(Vec::new()),
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
        sample.input.validate_actual_completion()?;
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
            || !self.calls.insert(sample.call_id)?
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

#[cfg(test)]
mod retained_payload_tests {
    use super::*;
    use sha2::{Digest, Sha256};

    fn signature(state: &PhaseState) -> [u8; 32] {
        let mut digest = Sha256::new();
        state.bind_parameters(&mut digest);
        digest.finalize().into()
    }

    #[test]
    fn freezing_calls_preserves_signature_and_closes_membership() {
        let mut state = PhaseState::new();
        for call in [81, 5, 900, 14] {
            assert!(state.calls.insert(call).unwrap());
        }
        assert!(!state.calls.insert(14).unwrap());
        let original = signature(&state);
        assert_eq!(state.frozen_heap_bytes(), None);
        state.freeze_calls().unwrap();
        assert_eq!(signature(&state), original);
        assert_eq!(
            state.frozen_heap_bytes(),
            Some(4 * std::mem::size_of::<u64>())
        );
        assert!(matches!(
            state.calls.insert(15),
            Err(StructuredUnknown::WrongProtocol)
        ));
        state.freeze_calls().unwrap();
        assert_eq!(signature(&state), original);
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

#[cfg(test)]
mod call_memory_tests;
