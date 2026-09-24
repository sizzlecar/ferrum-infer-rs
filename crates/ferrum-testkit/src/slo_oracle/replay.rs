use super::*;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Step {
    pub wave: Wave,
    pub start_at: u64,
    pub end_at: u64,
    pub after: State,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CheckedSequence {
    pub steps: Vec<Step>,
    pub final_state: State,
}

/// Checks a proposed *complete closed-model* sequence independently of search.
/// Incomplete prefixes are not certificates; production finite-horizon plans
/// need an explicit adapter with the same obligation/terminal semantics.
pub fn check_sequence(problem: &Problem, waves: &[Wave]) -> Result<CheckedSequence, Error> {
    problem.validate()?;
    let mut state = problem.initial.clone();
    let mut steps = Vec::new();
    for wave in waves {
        let step = advance(problem, &state, wave)?;
        state = step.after.clone();
        steps.push(step);
    }
    if !problem.complete(&state) {
        return Err(Error::Incomplete);
    }
    Ok(CheckedSequence {
        steps,
        final_state: state,
    })
}

pub(crate) fn check_time(
    owner: Owner,
    boundary: Boundary,
    due_at: u64,
    at: u64,
) -> Result<(), Error> {
    if at > due_at {
        Err(Error::ModelDeadline {
            owner,
            boundary,
            due_at,
            at,
        })
    } else {
        Ok(())
    }
}

/// Check every pending owner, including those not selected in the wave. A
/// selected row cannot hide a deadline crossed before its end-of-wave commit.
pub(crate) fn check_pending(problem: &Problem, before: &State, at: u64) -> Result<(), Error> {
    for (request, progress) in problem.requests.iter().zip(&before.progress) {
        if progress.commit_times.len() < usize::from(request.maximum_output_tokens) {
            if let (Some(&first), Some(&last)) =
                (progress.commit_times.first(), progress.commit_times.last())
            {
                let adjacent = last
                    .checked_add(request.budgets.itl.get())
                    .ok_or(Error::Overflow)?;
                let prefix = (progress.commit_times.len() as u64)
                    .checked_mul(request.budgets.tpot.get())
                    .and_then(|n| first.checked_add(n))
                    .ok_or(Error::Overflow)?;
                check_time(request.owner, Boundary::AdjacentCommit, adjacent, at)?;
                check_time(request.owner, Boundary::PrefixTpot, prefix, at)?;
            } else {
                let due = request
                    .ingress_at
                    .checked_add(request.budgets.ttft.get())
                    .ok_or(Error::Overflow)?;
                check_time(request.owner, Boundary::FirstCommit, due, at)?;
            }
        }
        for milestone in &request.milestones {
            if progress.net_prompt_tokens < milestone.net_prompt_tokens {
                check_time(
                    request.owner,
                    Boundary::PrefillMilestone,
                    milestone.due_at,
                    at,
                )?;
            }
        }
    }
    Ok(())
}

pub(crate) fn advance(problem: &Problem, before: &State, wave: &Wave) -> Result<Step, Error> {
    if wave.0.is_empty()
        || wave.0.len() > problem.maximum_rows
        || wave.0.windows(2).any(|w| w[0].owner() >= w[1].owner())
    {
        return Err(Error::Invalid("wave row domain/order/capacity"));
    }
    // Validate all work before reading cost or committing any progress.
    let mut rows = Vec::new();
    for work in &wave.0 {
        let index = problem
            .requests
            .binary_search_by_key(&work.owner(), |r| r.owner)
            .map_err(|_| Error::Invalid("unknown owner"))?;
        let request = &problem.requests[index];
        let progress = &before.progress[index];
        match *work {
            Work::Prefill { offset, count, .. } => {
                if count == 0
                    || offset != progress.net_prompt_tokens
                    || !progress.commit_times.is_empty()
                    || offset
                        .checked_add(count)
                        .is_none_or(|end| end > request.prompt_tokens)
                {
                    return Err(Error::Invalid("prefill work/frontier"));
                }
            }
            Work::Decode {
                previous_commits, ..
            } => {
                if progress.net_prompt_tokens != request.prompt_tokens
                    || usize::from(previous_commits) != progress.commit_times.len()
                    || previous_commits >= request.maximum_output_tokens
                {
                    return Err(Error::Invalid("decode work/frontier"));
                }
            }
        }
        rows.push(index);
    }
    let duration = problem
        .costs
        .0
        .get(wave)
        .ok_or_else(|| Error::MissingCost(wave.clone()))?;
    let end_at = before
        .at
        .checked_add(duration.get())
        .ok_or(Error::Overflow)?;
    check_pending(problem, before, end_at)?;
    let mut after = before.clone();
    after.at = end_at;
    for (work, index) in wave.0.iter().zip(rows) {
        let progress = &mut after.progress[index];
        match *work {
            Work::Prefill { count, .. } => {
                progress.net_prompt_tokens += count; // checked above
                                                     // Real final-prefill semantics: first token at this wave end.
                if progress.net_prompt_tokens == problem.requests[index].prompt_tokens {
                    progress.commit_times.push(end_at);
                }
            }
            Work::Decode { .. } => progress.commit_times.push(end_at),
        }
    }
    Ok(Step {
        wave: wave.clone(),
        start_at: before.at,
        end_at,
        after,
    })
}
