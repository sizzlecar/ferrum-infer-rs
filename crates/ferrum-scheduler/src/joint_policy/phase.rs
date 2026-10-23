//! Explicit action phases bounded by the original simulated prefill frontiers.
//! The core owns the fence; an adapter cannot renew it on later waves or ranks.

use super::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PrefillTarget {
    pub request: usize,
    /// Captured from the initial request, never an adapter-selected short tail.
    pub prompt_tokens: usize,
}

#[derive(Debug)]
pub enum ContinuationPhase {
    PrefillTargets(Vec<PrefillTarget>),
    /// Original waiting members may need modeled slot releases before their
    /// prefill can advance. The original admission plan remains authoritative.
    DeferredPrefillTargets(Vec<PrefillTarget>),
}

impl ContinuationPhase {
    pub fn prefill_targets(&self) -> &[PrefillTarget] {
        match self {
            Self::PrefillTargets(targets) | Self::DeferredPrefillTargets(targets) => targets,
        }
    }

    /// Additional dynamic target payload, excluding fixed stack metadata and
    /// allocator overhead. Product callers reserve this alongside their frozen
    /// request facts, clocks and forecasts in max_request_fact_bytes.
    pub fn prefill_targets_bytes_bound(targets: usize) -> Option<usize> {
        targets.checked_mul(std::mem::size_of::<PrefillTarget>())
    }
}

/// The core supplies the actual initial cohort and its existing resource limit.
/// An empty declaration preserves the original first-wave-only action contract.
pub struct PhaseBuilder<'a> {
    requests: &'a [SimulatedRequest],
    max_targets: usize,
    max_bytes: usize,
    targets: Vec<PrefillTarget>,
    deferred: bool,
}

impl<'a> PhaseBuilder<'a> {
    pub(super) fn new(
        requests: &'a [SimulatedRequest],
        max_targets: usize,
        max_bytes: usize,
    ) -> Self {
        Self {
            requests,
            max_targets,
            max_bytes,
            targets: Vec::new(),
            deferred: false,
        }
    }

    pub fn prefill_target(&mut self, request: usize) -> Result<(), Unsupported> {
        if self.deferred {
            return Err(Unsupported::InvalidPlan);
        }
        self.push(request)
    }

    /// Declare only a captured waiting owner. The core checks its original
    /// admission state before invoking the adapter's first plan.
    pub fn deferred_prefill_target(&mut self, request: usize) -> Result<(), Unsupported> {
        if !self.targets.is_empty() && !self.deferred {
            return Err(Unsupported::InvalidPlan);
        }
        self.push(request)?;
        self.deferred = true;
        Ok(())
    }

    fn push(&mut self, request: usize) -> Result<(), Unsupported> {
        let count = self
            .targets
            .len()
            .checked_add(1)
            .filter(|count| *count <= self.max_targets)
            .ok_or(Unsupported::ParticipantBudget)?;
        ContinuationPhase::prefill_targets_bytes_bound(count)
            .filter(|bytes| *bytes <= self.max_bytes)
            .ok_or(Unsupported::ParticipantBudget)?;
        let initial = self.requests.get(request).ok_or(Unsupported::InvalidPlan)?;
        if initial.prefill_tokens >= initial.prompt_tokens
            || initial.output_tokens != 0
            || initial.finished()
            || self.targets.iter().any(|target| target.request == request)
        {
            return Err(Unsupported::InvalidPlan);
        }
        self.targets
            .try_reserve_exact(1)
            .map_err(|_| Unsupported::ParticipantBudget)?;
        self.targets.push(PrefillTarget {
            request,
            prompt_tokens: initial.prompt_tokens,
        });
        Ok(())
    }

    pub(super) fn finish(self) -> Option<ContinuationPhase> {
        (!self.targets.is_empty()).then(|| {
            if self.deferred {
                ContinuationPhase::DeferredPrefillTargets(self.targets)
            } else {
                ContinuationPhase::PrefillTargets(self.targets)
            }
        })
    }
}

pub(super) struct PhaseState {
    phase: ContinuationPhase,
    remaining: usize,
    cohort_remaining: Option<usize>,
    awaiting_slot: bool,
}

impl PhaseState {
    pub(super) fn new(
        phase: ContinuationPhase,
        requests: &[SimulatedRequest],
        slots: Option<&AdmissionSlotPlan>,
    ) -> Result<Self, Unsupported> {
        let remaining = remaining(&phase, requests)?;
        if remaining == 0 {
            return Err(Unsupported::InvalidPlan);
        }
        let cohort_remaining = if matches!(&phase, ContinuationPhase::DeferredPrefillTargets(_)) {
            let slots = slots.ok_or(Unsupported::WaitingResources)?;
            for target in phase.prefill_targets() {
                if !matches!(
                    slots.initial_admission(target.request),
                    Some(InitialAdmission::Waiting { .. })
                ) || requests[target.request].prefill_tokens != 0
                    || requests[target.request].output_tokens != 0
                {
                    return Err(Unsupported::InvalidPlan);
                }
            }
            Some(cohort_remaining(requests)?)
        } else {
            None
        };
        Ok(Self {
            phase,
            remaining,
            cohort_remaining,
            awaiting_slot: false,
        })
    }

    pub(super) fn phase(&self) -> &ContinuationPhase {
        &self.phase
    }

    /// Freeze eligibility at the wave boundary, including split invocations.
    /// A slot released by one group cannot admit a target in a later group of
    /// this same wave. No additional active-mask allocation is needed.
    pub(super) fn begin_wave(
        &mut self,
        requests: &[SimulatedRequest],
        active: Option<&[bool]>,
    ) -> Result<(), Unsupported> {
        if self.cohort_remaining.is_some() {
            let active = active
                .filter(|active| active.len() == requests.len())
                .ok_or(Unsupported::WaitingResources)?;
            self.awaiting_slot = !self.phase.prefill_targets().iter().any(|target| {
                requests[target.request].prefill_tokens < target.prompt_tokens
                    && active[target.request]
            });
        }
        Ok(())
    }

    /// Once any unfinished target is eligible, real target prefill progress is
    /// mandatory. Before that boundary, the original active owners must reduce
    /// the checked finite cohort work. PlanBuilder forbids inactive work.
    pub(super) fn progressed(&self, requests: &[SimulatedRequest]) -> Result<usize, Unsupported> {
        let remaining = remaining(&self.phase, requests)?;
        let progresses = if self.awaiting_slot {
            remaining <= self.remaining
                && cohort_remaining(requests)?
                    < self.cohort_remaining.ok_or(Unsupported::InvalidPlan)?
        } else {
            remaining < self.remaining
        };
        if !progresses {
            return Err(Unsupported::NoProgress);
        }
        Ok(remaining)
    }

    pub(super) fn advance(&mut self, requests: &[SimulatedRequest]) -> Result<bool, Unsupported> {
        self.remaining = self.progressed(requests)?;
        if self.cohort_remaining.is_some() {
            self.cohort_remaining = Some(cohort_remaining(requests)?);
        }
        Ok(self.remaining == 0)
    }
}

fn cohort_remaining(requests: &[SimulatedRequest]) -> Result<usize, Unsupported> {
    requests.iter().try_fold(0usize, |total, request| {
        let prefill = request
            .prompt_tokens
            .checked_sub(request.prefill_tokens)
            .ok_or(Unsupported::InvalidState)?;
        let output = request
            .scenario_end_tokens
            .checked_sub(request.output_tokens)
            .ok_or(Unsupported::InvalidState)?;
        total
            .checked_add(prefill)
            .and_then(|total| total.checked_add(output))
            .ok_or(Unsupported::Arithmetic)
    })
}

fn remaining(
    phase: &ContinuationPhase,
    requests: &[SimulatedRequest],
) -> Result<usize, Unsupported> {
    phase
        .prefill_targets()
        .iter()
        .try_fold(0usize, |total, target| {
            let request = requests
                .get(target.request)
                .ok_or(Unsupported::InvalidPlan)?;
            if request.prompt_tokens != target.prompt_tokens {
                return Err(Unsupported::InvalidPlan);
            }
            let left = target
                .prompt_tokens
                .checked_sub(request.prefill_tokens)
                .ok_or(Unsupported::InvalidPlan)?;
            total.checked_add(left).ok_or(Unsupported::Arithmetic)
        })
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod deferred_tests;
