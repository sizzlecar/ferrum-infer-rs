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
}

impl ContinuationPhase {
    pub fn prefill_targets(&self) -> &[PrefillTarget] {
        match self {
            Self::PrefillTargets(targets) => targets,
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
        }
    }

    pub fn prefill_target(&mut self, request: usize) -> Result<(), Unsupported> {
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
        (!self.targets.is_empty()).then_some(ContinuationPhase::PrefillTargets(self.targets))
    }
}

pub(super) struct PhaseState {
    phase: ContinuationPhase,
    remaining: usize,
}

impl PhaseState {
    pub(super) fn new(
        phase: ContinuationPhase,
        requests: &[SimulatedRequest],
    ) -> Result<Self, Unsupported> {
        let remaining = remaining(&phase, requests)?;
        if remaining == 0 {
            return Err(Unsupported::InvalidPlan);
        }
        Ok(Self { phase, remaining })
    }

    pub(super) fn phase(&self) -> &ContinuationPhase {
        &self.phase
    }

    /// Successful completion of a wave (including an early common-window
    /// return) must contain real prefill progress on the frozen target set.
    pub(super) fn progressed(&self, requests: &[SimulatedRequest]) -> Result<usize, Unsupported> {
        let remaining = remaining(&self.phase, requests)?;
        if remaining >= self.remaining {
            return Err(Unsupported::NoProgress);
        }
        Ok(remaining)
    }

    pub(super) fn advance(&mut self, requests: &[SimulatedRequest]) -> Result<bool, Unsupported> {
        self.remaining = self.progressed(requests)?;
        Ok(self.remaining == 0)
    }
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
