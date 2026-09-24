use super::*;
use std::num::NonZeroUsize;

/// Explicit small-model resource bounds, never a scheduling acceptance rule.
#[derive(Debug, Clone, Copy)]
pub struct EnumerationLimits {
    pub maximum_requests: NonZeroUsize,
    pub maximum_waves: NonZeroUsize,
    pub maximum_transitions: NonZeroU64,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SearchStats {
    pub transitions: u64,
    pub deadline_rejections: u64,
    pub missing_costs: u64,
    pub truncated_prefixes: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Uncertainty {
    MissingCosts,
    EnumerationBudget,
    WaveLimit,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SearchResult {
    Feasible {
        sequence: CheckedSequence,
        stats: SearchStats,
    },
    /// Exhausted the declared finite model, not production physical schedules.
    NoModelWitness { stats: SearchStats },
    Inconclusive {
        cause: Uncertainty,
        stats: SearchStats,
    },
}

/// Exact depth-first enumeration of all legal row subsets and positive prefill
/// counts, stopping on a complete witness. Every wave strictly reduces finite
/// remaining work. There is no heuristic beam, monotonic-cost assumption, or
/// horizon whose exhaustion could be mislabeled as infeasibility. Oracle
/// transition/depth bounds return Inconclusive when any needed work is cut.
pub fn enumerate(problem: &Problem, limits: EnumerationLimits) -> Result<SearchResult, Error> {
    problem.validate()?;
    if problem.requests.len() > limits.maximum_requests.get() {
        return Err(Error::Invalid("oracle request bound"));
    }
    let mut search = Search {
        problem,
        limits,
        stats: SearchStats::default(),
        stopped: false,
    };
    if let Some(sequence) = search.node(&problem.initial, &mut Vec::new())? {
        return Ok(SearchResult::Feasible {
            sequence,
            stats: search.stats,
        });
    }
    Ok(if search.stopped {
        SearchResult::Inconclusive {
            cause: Uncertainty::EnumerationBudget,
            stats: search.stats,
        }
    } else if search.stats.missing_costs > 0 {
        SearchResult::Inconclusive {
            cause: Uncertainty::MissingCosts,
            stats: search.stats,
        }
    } else if search.stats.truncated_prefixes > 0 {
        SearchResult::Inconclusive {
            cause: Uncertainty::WaveLimit,
            stats: search.stats,
        }
    } else {
        SearchResult::NoModelWitness {
            stats: search.stats,
        }
    })
}

struct Search<'a> {
    problem: &'a Problem,
    limits: EnumerationLimits,
    stats: SearchStats,
    stopped: bool,
}

impl Search<'_> {
    fn node(
        &mut self,
        state: &State,
        path: &mut Vec<Step>,
    ) -> Result<Option<CheckedSequence>, Error> {
        if self.problem.complete(state) {
            return Ok(Some(CheckedSequence {
                steps: path.clone(),
                final_state: state.clone(),
            }));
        }
        if path.len() == self.limits.maximum_waves.get() {
            self.stats.truncated_prefixes += 1;
            return Ok(None);
        }
        self.rows(state, 0, &mut Vec::new(), path)
    }

    // Lazy row combinations: no power-set allocation before the budget check.
    fn rows(
        &mut self,
        state: &State,
        index: usize,
        work: &mut Vec<Work>,
        path: &mut Vec<Step>,
    ) -> Result<Option<CheckedSequence>, Error> {
        if self.stopped {
            return Ok(None);
        }
        if index == self.problem.requests.len() {
            if work.is_empty() {
                return Ok(None);
            }
            if self.stats.transitions == self.limits.maximum_transitions.get() {
                self.stopped = true;
                return Ok(None);
            }
            self.stats.transitions += 1;
            match replay::advance(self.problem, state, &Wave(work.clone())) {
                Ok(step) => {
                    let next = step.after.clone();
                    path.push(step);
                    let found = self.node(&next, path)?;
                    path.pop();
                    return Ok(found);
                }
                Err(Error::ModelDeadline { .. }) => self.stats.deadline_rejections += 1,
                Err(Error::MissingCost(_)) => self.stats.missing_costs += 1,
                Err(other) => return Err(other),
            }
            return Ok(None);
        }
        // Omit this owner, but advance() still checks its deadline.
        if let Some(found) = self.rows(state, index + 1, work, path)? {
            return Ok(Some(found));
        }
        if self.stopped || work.len() == self.problem.maximum_rows {
            return Ok(None);
        }
        let request = &self.problem.requests[index];
        let progress = &state.progress[index];
        let owner = request.owner;
        let remaining_prompt = request.prompt_tokens - progress.net_prompt_tokens;
        if remaining_prompt > 0 {
            for count in 1..=remaining_prompt {
                work.push(Work::Prefill {
                    owner,
                    offset: progress.net_prompt_tokens,
                    count,
                });
                let found = self.rows(state, index + 1, work, path)?;
                work.pop();
                if found.is_some() || self.stopped {
                    return Ok(found);
                }
            }
        } else if progress.commit_times.len() < usize::from(request.maximum_output_tokens) {
            work.push(Work::Decode {
                owner,
                previous_commits: progress.commit_times.len() as u16,
            });
            let found = self.rows(state, index + 1, work, path)?;
            work.pop();
            return Ok(found);
        }
        Ok(None)
    }
}
