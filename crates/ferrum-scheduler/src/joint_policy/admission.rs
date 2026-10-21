//! Closed-cohort execution-slot accounting. This is a scheduling projection;
//! the executor remains the authority for physical admission and release.

use super::*;

/// Timing assumption of the trajectory model, not a physical release guarantee.
/// Original asynchronous close/recovery can delay real admission beyond this
/// modeled boundary; only client measurements establish latency compliance.
#[derive(Debug, Clone, Copy, Default, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum AdmissionSlotReleaseAssumption {
    #[default]
    NextWaveAfterModeledCompletion,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InitialAdmission {
    Resident,
    /// Dense order in the captured admission queue, including any declared
    /// priority ordering. No request may disappear from this initial cohort.
    Waiting {
        order: usize,
    },
}

#[derive(Debug, Clone)]
pub struct AdmissionSlotPlan {
    initial: Vec<InitialAdmission>,
    waiting_order: Vec<usize>,
    max_active: usize,
}

impl AdmissionSlotPlan {
    pub fn new(
        initial: &[InitialAdmission],
        max_active: usize,
        max_participants: usize,
    ) -> Result<Self, Unsupported> {
        if initial.is_empty() || initial.len() > max_participants {
            return Err(Unsupported::ParticipantBudget);
        }
        let residents = initial
            .iter()
            .filter(|state| **state == InitialAdmission::Resident)
            .count();
        if max_active == 0 || residents > max_active {
            return Err(Unsupported::WaitingResources);
        }
        let waiting = initial.len() - residents;
        let mut waiting_order = vec![usize::MAX; waiting];
        for (index, state) in initial.iter().enumerate() {
            if let InitialAdmission::Waiting { order } = *state {
                let slot = waiting_order
                    .get_mut(order)
                    .ok_or(Unsupported::InvalidState)?;
                if *slot != usize::MAX {
                    return Err(Unsupported::InvalidState);
                }
                *slot = index;
            }
        }
        Ok(Self {
            initial: initial.to_vec(),
            waiting_order,
            max_active,
        })
    }

    pub fn waiting_requests(&self) -> usize {
        self.waiting_order.len()
    }

    /// The same frozen order is used for every candidate and continuation.
    /// Initially resident requests retain their slots until completion. The
    /// first unfinished waiters occupy released slots, even when token limits
    /// keep an admitted waiter from doing work in this particular wave.
    pub fn active_mask(&self, requests: &[SimulatedRequest]) -> Result<Vec<bool>, Unsupported> {
        if requests.len() != self.initial.len() {
            return Err(Unsupported::InvalidState);
        }
        let mut active = vec![false; requests.len()];
        let mut available = self.max_active;
        for (index, state) in self.initial.iter().enumerate() {
            if *state == InitialAdmission::Resident && !requests[index].finished() {
                active[index] = true;
                available = available
                    .checked_sub(1)
                    .ok_or(Unsupported::WaitingResources)?;
            }
        }
        for &index in &self.waiting_order {
            if !requests[index].finished() && available > 0 {
                active[index] = true;
                available -= 1;
            }
        }
        // Work on a later waiter cannot be used as an invented proof of prior
        // admission, nor can a still-running admitted request lose its slot.
        if self.waiting_order.iter().any(|&index| {
            !active[index]
                && !requests[index].finished()
                && (requests[index].prefill_tokens > 0 || requests[index].output_tokens > 0)
        }) {
            return Err(Unsupported::InvalidState);
        }
        Ok(active)
    }

    fn validate_initial(
        &self,
        requests: &[SimulatedRequest],
        waiting_requests: usize,
    ) -> Result<(), Unsupported> {
        if self.waiting_requests() != waiting_requests
            || requests.len() != self.initial.len()
            || self.waiting_order.iter().any(|&index| {
                requests[index].prefill_tokens != 0 || requests[index].output_tokens != 0
            })
        {
            return Err(Unsupported::InvalidState);
        }
        self.active_mask(requests).map(|_| ())
    }
}

pub(super) fn validate_admission<Action, Adapter: TrajectoryAdapter<Action>>(
    input: &EvaluationInput<'_, Action, Adapter>,
) -> Result<(), Unsupported> {
    match input.adapter.admission_slots() {
        Some(slots) => slots.validate_initial(input.requests, input.resources.waiting_requests),
        None if input.resources.waiting_requests == 0 => Ok(()),
        None => Err(Unsupported::WaitingResources),
    }
}

#[cfg(test)]
mod tests;
