//! Independent, finite, synthetic token-commit scheduling oracle.
//!
//! No production planner, simulator, clock, predictor, or resource authority is
//! called here. A table entry is a declared whole-wave cost in this toy model,
//! not a hardware measurement or a proof about client-visible text events.

use std::{collections::BTreeMap, num::NonZeroU64};

mod replay;
mod search;
pub use replay::{check_sequence, CheckedSequence, Step};
pub use search::{enumerate, EnumerationLimits, SearchResult, SearchStats, Uncertainty};

#[cfg(test)]
mod tests;

/// Local fixture identity, never an executor owner or submission permit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Owner(pub u16);

/// All times use one caller-declared monotonic tick unit. No wall clock.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Budgets {
    pub ttft: NonZeroU64,
    pub tpot: NonZeroU64,
    pub itl: NonZeroU64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Milestone {
    pub net_prompt_tokens: u16,
    pub due_at: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Request {
    pub owner: Owner,
    pub ingress_at: u64,
    pub prompt_tokens: u16,
    /// Closed-model output limit, not a prediction or a truncated real request.
    pub maximum_output_tokens: u16,
    pub budgets: Budgets,
    /// Fixed absolute obligations; serving peers must never reset them.
    pub milestones: Vec<Milestone>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Progress {
    pub net_prompt_tokens: u16,
    /// Full tiny history lets validation reject already-invalid fixture states.
    pub commit_times: Vec<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct State {
    /// Includes time already spent planning; ingress is never moved forward.
    pub at: u64,
    /// Same stable order as Problem::requests.
    pub progress: Vec<Progress>,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum Work {
    Prefill {
        owner: Owner,
        offset: u16,
        count: u16,
    },
    Decode {
        owner: Owner,
        previous_commits: u16,
    },
}

impl Work {
    pub fn owner(&self) -> Owner {
        match *self {
            Self::Prefill { owner, .. } | Self::Decode { owner, .. } => owner,
        }
    }
}

/// Ordered whole-wave entry. One row per owner, in ascending fixture identity.
/// This is the declared action domain, not a claim about backend row ordering.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct Wave(pub Vec<Work>);

#[derive(Debug, Clone, Default)]
pub struct CostTable(BTreeMap<Wave, NonZeroU64>);

impl CostTable {
    pub fn insert(&mut self, wave: Wave, duration: NonZeroU64) -> Result<(), Error> {
        if self.0.contains_key(&wave) {
            return Err(Error::Invalid("duplicate wave cost"));
        }
        self.0.insert(wave, duration);
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct Problem {
    pub requests: Vec<Request>,
    pub initial: State,
    /// Only physical abstraction in A0: simultaneous rows, not bytes or KV.
    pub maximum_rows: usize,
    pub costs: CostTable,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Boundary {
    FirstCommit,
    AdjacentCommit,
    PrefixTpot,
    PrefillMilestone,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    Invalid(&'static str),
    Overflow,
    MissingCost(Wave),
    /// Fails certification in this declared cost model, not real impossibility.
    ModelDeadline {
        owner: Owner,
        boundary: Boundary,
        due_at: u64,
        at: u64,
    },
    Incomplete,
}

impl Problem {
    pub fn validate(&self) -> Result<(), Error> {
        if self.requests.is_empty()
            || self.maximum_rows == 0
            || self.initial.progress.len() != self.requests.len()
            || self.requests.windows(2).any(|w| w[0].owner >= w[1].owner)
        {
            return Err(Error::Invalid("request domain/order/capacity"));
        }
        for (request, progress) in self.requests.iter().zip(&self.initial.progress) {
            if request.maximum_output_tokens == 0
                || request.ingress_at > self.initial.at
                || progress.net_prompt_tokens > request.prompt_tokens
                || progress.commit_times.len() > usize::from(request.maximum_output_tokens)
                || (!progress.commit_times.is_empty()
                    && progress.net_prompt_tokens != request.prompt_tokens)
                || (request.prompt_tokens > 0
                    && progress.net_prompt_tokens == request.prompt_tokens
                    && progress.commit_times.is_empty())
            {
                return Err(Error::Invalid("initial request progress"));
            }
            let first_due = request
                .ingress_at
                .checked_add(request.budgets.ttft.get())
                .ok_or(Error::Overflow)?;
            for (index, &at) in progress.commit_times.iter().enumerate() {
                if at < request.ingress_at
                    || at > self.initial.at
                    || (index > 0 && at <= progress.commit_times[index - 1])
                {
                    return Err(Error::Invalid("initial commit chronology"));
                }
                replay::check_time(
                    request.owner,
                    Boundary::FirstCommit,
                    first_due,
                    progress.commit_times[0],
                )?;
                if index > 0 {
                    let adjacent = progress.commit_times[index - 1]
                        .checked_add(request.budgets.itl.get())
                        .ok_or(Error::Overflow)?;
                    let prefix = progress.commit_times[0]
                        .checked_add(
                            (index as u64)
                                .checked_mul(request.budgets.tpot.get())
                                .ok_or(Error::Overflow)?,
                        )
                        .ok_or(Error::Overflow)?;
                    replay::check_time(request.owner, Boundary::AdjacentCommit, adjacent, at)?;
                    replay::check_time(request.owner, Boundary::PrefixTpot, prefix, at)?;
                }
            }
            let mut previous = 0;
            for milestone in &request.milestones {
                if milestone.net_prompt_tokens <= previous
                    || milestone.net_prompt_tokens > request.prompt_tokens
                    || milestone.due_at < request.ingress_at
                {
                    return Err(Error::Invalid("prefill milestone domain"));
                }
                // A0 cannot establish the historical completion time of a
                // milestone from just an already-prefilled initial frontier.
                if progress.net_prompt_tokens >= milestone.net_prompt_tokens {
                    return Err(Error::Invalid("historical milestone needs receipt"));
                }
                previous = milestone.net_prompt_tokens;
            }
        }
        // Expired *pending* obligations are model failures. They are not reset.
        replay::check_pending(self, &self.initial, self.initial.at)
    }

    pub(crate) fn complete(&self, state: &State) -> bool {
        self.requests.iter().zip(&state.progress).all(|(r, p)| {
            p.net_prompt_tokens == r.prompt_tokens
                && p.commit_times.len() == usize::from(r.maximum_output_tokens)
        })
    }
}
