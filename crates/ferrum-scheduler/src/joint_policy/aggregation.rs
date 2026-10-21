//! Streaming aggregation across declared conditional cost/stopping ranks.
//! Model-relative feasibility is separate from measured client SLO acceptance.

use super::*;

#[derive(Debug)]
pub struct ScenarioPolicyDecision {
    pub valid_until: Instant,
    pub choice: PolicyChoice,
    pub static_risk: QueueRisk,
    pub alternative_risks: Vec<QueueRisk>,
    pub static_target_time: Option<Duration>,
    pub alternative_target_times: Vec<Option<Duration>>,
    pub any_window_debt: Vec<bool>,
}

#[derive(Default)]
struct Totals {
    ttft_bad: f64,
    ttft_unknown: f64,
    tpot_bad: f64,
    tpot_eligible: f64,
    tpot_unknown_outcome: f64,
    tpot_unknown_eligible: f64,
    tpot_undefined: f64,
    itl_bad: f64,
    itl_events: f64,
    itl_unknown_events: f64,
    completion_lower: f64,
    completion_upper: f64,
    target_seconds: f64,
    time_mass: f64,
    unsupported_mass: f64,
    window_debt: bool,
}

struct Member {
    id: RequestId,
    prompt_tokens: usize,
    cap: usize,
    product_success: ProbabilityInterval,
}

/// O(requests + candidates) state; no per-rank trajectories are retained.
pub struct ScenarioRiskAccumulator {
    config: SchedulerSloConfig,
    members: Vec<Member>,
    totals: Vec<Totals>,
    observed_mass: f64,
    observed_ranks: usize,
    quantile_upper: f64,
    last_rank: Option<usize>,
    rank_count: Option<usize>,
    revision: Option<u64>,
    ttft_limit: Duration,
    tpot_limit: Duration,
}

impl ScenarioRiskAccumulator {
    pub fn new(
        requests: &[SimulatedRequest],
        alternatives: usize,
        config: &SchedulerSloConfig,
        product_success: &[Option<ProbabilityInterval>],
    ) -> Result<Self, Unsupported> {
        if config.validate().is_err()
            || requests.is_empty()
            || product_success.len() != requests.len()
            || requests.len() > config.planning.max_participants
            || alternatives
                .checked_add(1)
                .is_none_or(|count| count > config.planning.max_candidates)
        {
            return Err(Unsupported::InvalidState);
        }
        let mut members = Vec::with_capacity(requests.len());
        for (request, success) in requests.iter().zip(product_success) {
            let success = success.unwrap_or(ProbabilityInterval {
                lower: 0.0,
                upper: 1.0,
            });
            if !request.valid()
                || !success.lower.is_finite()
                || !success.upper.is_finite()
                || success.lower < 0.0
                || success.upper > 1.0
                || success.lower > success.upper
                || members
                    .iter()
                    .any(|member: &Member| member.id == request.id)
            {
                return Err(Unsupported::InvalidState);
            }
            members.push(Member {
                id: request.id.clone(),
                prompt_tokens: request.prompt_tokens,
                cap: request.output_cap,
                product_success: success,
            });
        }
        Ok(Self {
            config: *config,
            members,
            totals: (0..=alternatives).map(|_| Totals::default()).collect(),
            observed_mass: 0.0,
            observed_ranks: 0,
            quantile_upper: 0.0,
            last_rank: None,
            rank_count: None,
            revision: None,
            ttft_limit: Duration::try_from_secs_f64(config.ttft_ms / 1000.0)
                .map_err(|_| Unsupported::InvalidState)?,
            tpot_limit: Duration::try_from_secs_f64(config.tpot_ms / 1000.0)
                .map_err(|_| Unsupported::InvalidState)?,
        })
    }

    pub fn observe(
        &mut self,
        rank: ScenarioRank,
        comparison: &Comparison,
    ) -> Result<(), Unsupported> {
        if rank.count == 0
            || rank.count > self.config.planning.max_scenarios
            || rank.index >= rank.count
            || self.last_rank.is_some_and(|last| rank.index <= last)
            || self.rank_count.is_some_and(|count| count != rank.count)
            || !rank.weight.is_finite()
            || rank.weight <= 0.0
            || rank.weight > 1.0
            || !rank.quantile_lower.is_finite()
            || !rank.quantile_upper.is_finite()
            || rank.quantile_lower < 0.0
            || rank.quantile_lower < self.quantile_upper - 1e-9
            || rank.quantile_upper > 1.0
            || (rank.quantile_upper - rank.quantile_lower - rank.weight).abs() > 1e-9
            || self
                .revision
                .is_some_and(|revision| revision != comparison.revision)
            || comparison.alternatives.len() + 1 != self.totals.len()
            || comparison.targets.len() != self.members.len()
            || self.observed_mass + rank.weight > 1.0 + 1e-9
        {
            return Err(Unsupported::InvalidState);
        }
        // Validate all known trajectories before mutating any aggregate.
        for result in std::iter::once(&comparison.static_result).chain(
            comparison
                .alternatives
                .iter()
                .filter_map(|result| result.as_ref().ok()),
        ) {
            if result.requests.len() != self.members.len()
                || result.window_progress.len() != self.members.len()
                || result.cost_risk.conditional_scenario
                    != Some(CostScenarioDependence::SharedResidualRank)
                || result.cost_risk.empirical_quantiles
                || result.cost_risk.union_failure_upper != 0.0
                || result
                    .requests
                    .iter()
                    .zip(&self.members)
                    .any(|(request, member)| {
                        request.id != member.id
                            || request.prompt_tokens != member.prompt_tokens
                            || request.output_cap != member.cap
                            || request.scenario_end_tokens == 0
                            || request.scenario_end_tokens > member.cap
                            || request.output_tokens > request.scenario_end_tokens
                            || request.prefill_tokens > request.prompt_tokens
                            || request.central_clock.first_output
                                > request.central_clock.last_output
                            || request.central_clock.completed.is_some_and(|completed| {
                                request
                                    .central_clock
                                    .last_output
                                    .is_none_or(|last| completed < last)
                                    || request.output_tokens != request.scenario_end_tokens
                            })
                            || (request.output_tokens > 0
                                && request.central_clock.first_output.is_none())
                            || (request.output_tokens == 0
                                && request.central_clock.last_output.is_some())
                            || request.central_clock.violating_interval_count
                                > request.central_clock.interval_count
                            || request.central_clock.interval_count
                                > request.output_tokens.saturating_sub(1)
                    })
            {
                return Err(Unsupported::InvalidState);
            }
            if result
                .requests
                .iter()
                .zip(&comparison.static_result.requests)
                .any(|(request, reference)| {
                    request.scenario_end_tokens != reference.scenario_end_tokens
                })
            {
                return Err(Unsupported::InvalidState);
            }
        }
        for (index, outcome) in std::iter::once(Ok(&comparison.static_result))
            .chain(
                comparison
                    .alternatives
                    .iter()
                    .map(|outcome| outcome.as_ref()),
            )
            .enumerate()
        {
            let totals = &mut self.totals[index];
            match outcome {
                Ok(result) => {
                    totals.window_debt |= result.has_window_debt(&comparison.targets);
                    totals.target_seconds += rank.weight * result.target_reached_at.as_secs_f64();
                    totals.time_mass += rank.weight;
                    for (request, member) in result.requests.iter().zip(&self.members) {
                        accumulate_request(
                            totals,
                            rank.weight,
                            request,
                            member.product_success,
                            self.ttft_limit,
                            self.tpot_limit,
                            Duration::from_millis(self.config.joint.request_deadline_ms),
                        );
                    }
                }
                Err(_) => accumulate_unknown(totals, rank.weight, &self.members),
            }
        }
        self.last_rank = Some(rank.index);
        self.rank_count = Some(rank.count);
        self.revision = Some(comparison.revision);
        self.observed_mass += rank.weight;
        self.observed_ranks += 1;
        self.quantile_upper = rank.quantile_upper;
        Ok(())
    }

    /// Call only after successful complete scenario traversal. An evaluator
    /// error invalidates this accumulator; partial ranks cannot be normalized.
    pub fn finish(
        mut self,
        summary: ScenarioEvaluation,
        paired: &[Option<PairedActionEvidence>],
    ) -> Result<ScenarioPolicyDecision, Unsupported> {
        if paired.len() + 1 != self.totals.len()
            || summary.ranks == 0
            || summary.ranks > self.config.planning.max_scenarios
            || self.rank_count.is_some_and(|count| count != summary.ranks)
            || self.observed_ranks.checked_add(summary.unresolved_ranks) != Some(summary.ranks)
            || !summary.resolved_mass.is_finite()
            || !summary.unresolved_mass.is_finite()
            || summary.resolved_mass < 0.0
            || summary.unresolved_mass < 0.0
            || (summary.unresolved_ranks == 0) != (summary.unresolved_mass == 0.0)
            || (summary.resolved_mass - self.observed_mass).abs() > 1e-9
            || (summary.resolved_mass + summary.unresolved_mass - 1.0).abs() > 1e-9
        {
            return Err(Unsupported::InvalidState);
        }
        for totals in &mut self.totals {
            accumulate_unknown(totals, summary.unresolved_mass, &self.members);
        }
        let risks = self
            .totals
            .iter()
            .map(|totals| totals.risk(self.members.len()))
            .collect::<Vec<_>>();
        let times = self
            .totals
            .iter()
            .map(Totals::target_time)
            .collect::<Vec<_>>();
        let static_status = risks[0].classify(&self.config);
        let mut best = None;
        for index in 1..self.totals.len() {
            if risks[index].classify(&self.config) != JointFeasibility::Feasible {
                continue;
            }
            let Some(time) = times[index] else {
                continue;
            };
            if static_status == JointFeasibility::Feasible {
                if self.totals[index].window_debt {
                    continue;
                }
                let (Some(static_time), Some(evidence)) = (times[0], paired[index - 1]) else {
                    continue;
                };
                let Some(gain) = static_time.checked_sub(time) else {
                    continue;
                };
                if evidence
                    .uncertainty
                    .checked_add(evidence.control_cost)
                    .is_none_or(|margin| gain <= margin)
                {
                    continue;
                }
            } else if static_status == JointFeasibility::Unresolved {
                continue;
            }
            if best.is_none_or(|(_, previous)| time < previous) {
                best = Some((index - 1, time));
            }
        }
        let choice = if let Some((index, _)) = best {
            PolicyChoice::Alternative(index)
        } else if static_status == JointFeasibility::Feasible {
            PolicyChoice::Static
        } else if static_status == JointFeasibility::Infeasible
            && self
                .totals
                .iter()
                .all(|totals| totals.unsupported_mass == 0.0)
            && risks[1..]
                .iter()
                .all(|risk| risk.classify(&self.config) == JointFeasibility::Infeasible)
        {
            PolicyChoice::AllKnownInfeasible
        } else {
            PolicyChoice::Unresolved
        };
        Ok(ScenarioPolicyDecision {
            valid_until: summary.valid_until,
            choice,
            static_risk: risks[0],
            alternative_risks: risks[1..].to_vec(),
            static_target_time: times[0],
            alternative_target_times: times[1..].to_vec(),
            any_window_debt: self.totals[1..]
                .iter()
                .map(|totals| totals.window_debt)
                .collect(),
        })
    }
}

fn accumulate_request(
    totals: &mut Totals,
    weight: f64,
    request: &SimulatedRequest,
    success: ProbabilityInterval,
    ttft_limit: Duration,
    tpot_limit: Duration,
    deadline: Duration,
) {
    let clock = request.central_clock;
    if let Some(first) = clock.first_output {
        totals.ttft_bad += weight * f64::from(first > ttft_limit);
    } else {
        totals.ttft_unknown += weight;
    }
    let complete = (request.output_tokens == request.scenario_end_tokens)
        .then_some(clock.completed)
        .flatten();
    if request.scenario_end_tokens < 2 {
        totals.tpot_undefined += weight;
    } else {
        totals.tpot_eligible += weight;
        if let Some(duration) = complete
            .and(clock.last_output.zip(clock.first_output))
            .and_then(|(last, first)| last.checked_sub(first))
        {
            totals.tpot_bad += weight
                * f64::from(
                    duration.as_secs_f64() / (request.scenario_end_tokens - 1) as f64
                        > tpot_limit.as_secs_f64(),
                );
        } else {
            totals.tpot_unknown_outcome += weight;
        }
    }
    totals.itl_bad += weight * clock.violating_interval_count as f64;
    totals.itl_events += weight * clock.interval_count as f64;
    totals.itl_unknown_events += weight
        * request
            .scenario_end_tokens
            .saturating_sub(request.output_tokens.max(1)) as f64;
    match complete {
        Some(end) if end <= deadline => {
            totals.completion_lower += weight * (1.0 - success.upper);
            totals.completion_upper += weight * (1.0 - success.lower);
        }
        Some(_) => {
            totals.completion_lower += weight;
            totals.completion_upper += weight;
        }
        None => {
            totals.completion_lower += weight * (1.0 - success.upper);
            totals.completion_upper += weight;
        }
    }
}

fn accumulate_unknown(totals: &mut Totals, weight: f64, members: &[Member]) {
    if weight == 0.0 {
        return;
    }
    totals.unsupported_mass += weight;
    totals.ttft_unknown += weight * members.len() as f64;
    for member in members {
        if member.cap >= 2 {
            totals.tpot_unknown_eligible += weight;
        } else {
            totals.tpot_undefined += weight;
        }
        totals.itl_unknown_events += weight * member.cap.saturating_sub(1) as f64;
        totals.completion_lower += weight * (1.0 - member.product_success.upper);
        totals.completion_upper += weight;
    }
}

impl Totals {
    fn risk(&self, requests: usize) -> QueueRisk {
        let interval = |lower: f64, upper: f64| ProbabilityInterval {
            lower: lower.clamp(0.0, 1.0),
            upper: upper.clamp(0.0, 1.0),
        };
        let tpot_denominator = self.tpot_eligible + self.tpot_unknown_eligible;
        let itl_denominator = self.itl_events + self.itl_unknown_events;
        QueueRisk {
            ttft: interval(
                self.ttft_bad / requests as f64,
                (self.ttft_bad + self.ttft_unknown) / requests as f64,
            ),
            tpot: (tpot_denominator > 0.0).then(|| {
                interval(
                    self.tpot_bad / tpot_denominator,
                    (self.tpot_bad + self.tpot_unknown_outcome + self.tpot_unknown_eligible)
                        / tpot_denominator,
                )
            }),
            itl: (itl_denominator > 0.0).then(|| {
                interval(
                    self.itl_bad / itl_denominator,
                    (self.itl_bad + self.itl_unknown_events) / itl_denominator,
                )
            }),
            completion_failure: interval(
                self.completion_lower / requests as f64,
                self.completion_upper / requests as f64,
            ),
            tpot_undefined_mass: self.tpot_undefined,
            request_count: requests,
        }
    }
    fn target_time(&self) -> Option<Duration> {
        ((self.time_mass - 1.0).abs() <= 1e-9 && self.unsupported_mass == 0.0)
            .then(|| Duration::try_from_secs_f64(self.target_seconds).ok())
            .flatten()
    }
}

#[cfg(test)]
mod tests;
