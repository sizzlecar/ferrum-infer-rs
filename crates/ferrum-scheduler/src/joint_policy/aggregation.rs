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
    /// True for observed debt, or when completion by a partial static window's
    /// lower bound was not proved. It is never a claim about the unknown window.
    pub any_window_debt: Vec<bool>,
    pub observation: DecisionObservation,
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
    partial_window_unproven: bool,
    unfinished_request_mass: f64,
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
    partial_static_ranks: usize,
    quantile_upper: f64,
    last_rank: Option<usize>,
    rank_count: Option<usize>,
    revision: Option<u64>,
    ttft_limit: Duration,
    tpot_limit: Duration,
    product_success_evidence: ProductSuccessEvidence,
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
        let mut product_success_evidence = ProductSuccessEvidence::default();
        for (request, success) in requests.iter().zip(product_success) {
            let unavailable = success.is_none();
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
            product_success_evidence.unavailable_requests += usize::from(unavailable);
            product_success_evidence.uncertain_requests +=
                usize::from(success.lower < success.upper);
            product_success_evidence.success_lower_sum += success.lower;
            product_success_evidence.success_upper_sum += success.upper;
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
            partial_static_ranks: 0,
            quantile_upper: 0.0,
            last_rank: None,
            rank_count: None,
            revision: None,
            ttft_limit: Duration::try_from_secs_f64(config.ttft_ms / 1000.0)
                .map_err(|_| Unsupported::InvalidState)?,
            tpot_limit: Duration::try_from_secs_f64(config.tpot_ms / 1000.0)
                .map_err(|_| Unsupported::InvalidState)?,
            product_success_evidence,
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
        let static_requests = comparison.static_result.requests();
        let partial_static = match (&comparison.static_result, comparison.window) {
            (StaticTrajectory::Complete(result), StaticWindow::Exact(window))
                if window == result.central_elapsed
                    && result.window_progress.len() == self.members.len() =>
            {
                false
            }
            (StaticTrajectory::OutsideSupportPrefix(prefix), StaticWindow::AtLeast(lower))
                if lower == prefix.central_elapsed
                    && !lower.is_zero()
                    && prefix.unsupported
                        == Unsupported::Cost(CostUnknownReason::UnsupportedWithDetail(
                            CostUnsupportedDetail::OutsideObservedSupport,
                        ))
                    && prefix.requests.iter().any(|request| !request.finished())
                    && comparison.targets.iter().zip(&prefix.requests).all(
                        |(target, request)| {
                            *target
                                == Progress {
                                    prefill_tokens: request.prompt_tokens,
                                    output_tokens: request.scenario_end_tokens,
                                }
                        },
                    ) =>
            {
                true
            }
            _ => return Err(Unsupported::InvalidState),
        };
        // Validate all retained trajectories before mutating any aggregate.
        // A partial static carries real clocks, never a fake completed result.
        for (requests, cost_risk) in
            std::iter::once((static_requests, comparison.static_result.cost_risk())).chain(
                comparison
                    .alternatives
                    .iter()
                    .filter_map(|result| result.as_ref().ok())
                    .map(|result| (result.requests.as_slice(), &result.cost_risk)),
            )
        {
            if requests.len() != self.members.len()
                || cost_risk.conditional_scenario
                    != Some(CostScenarioDependence::SharedResidualRank)
                || cost_risk.empirical_quantiles
                || cost_risk.union_failure_upper != 0.0
                || requests.iter().zip(&self.members).any(|(request, member)| {
                    request.id != member.id
                        || request.prompt_tokens != member.prompt_tokens
                        || request.output_cap != member.cap
                        || request.scenario_end_tokens == 0
                        || request.scenario_end_tokens > member.cap
                        || request.output_tokens > request.scenario_end_tokens
                        || request.prefill_tokens > request.prompt_tokens
                        || request.central_clock.first_output > request.central_clock.last_output
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
            if requests
                .iter()
                .zip(static_requests)
                .any(|(request, reference)| {
                    request.scenario_end_tokens != reference.scenario_end_tokens
                })
            {
                return Err(Unsupported::InvalidState);
            }
        }
        if comparison
            .alternatives
            .iter()
            .filter_map(|result| result.as_ref().ok())
            .any(|result| result.window_progress.len() != self.members.len())
        {
            return Err(Unsupported::InvalidState);
        }
        let static_totals = &mut self.totals[0];
        if let Some(result) = comparison.static_result.complete() {
            static_totals.target_seconds += rank.weight * result.target_reached_at.as_secs_f64();
            static_totals.time_mass += rank.weight;
        } else {
            // Cost uncertainty is retained once per rank. Known request clocks
            // still contribute their lower/upper risk facts below.
            static_totals.unsupported_mass += rank.weight;
            self.partial_static_ranks += 1;
        }
        for (request, member) in static_requests.iter().zip(&self.members) {
            accumulate_request(
                static_totals,
                rank.weight,
                request,
                member.product_success,
                self.ttft_limit,
                self.tpot_limit,
                Duration::from_millis(self.config.joint.request_deadline_ms),
            );
        }
        for (index, outcome) in comparison.alternatives.iter().enumerate() {
            let totals = &mut self.totals[index + 1];
            match outcome {
                Ok(result) => {
                    totals.window_debt |= result.has_window_debt(&comparison.targets);
                    if partial_static && !comparison.partial_window_covered(result) {
                        totals.partial_window_unproven = true;
                        totals.window_debt = true;
                    }
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
            || self.partial_static_ranks != summary.partial_static_ranks
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
        let static_raw = risks[0].classify(&self.config);
        let mut static_status = static_raw;
        if self.partial_static_ranks > 0 && static_status != JointFeasibility::Infeasible {
            // Risk intervals alone never authorize optimization of a static
            // trajectory whose full cost/window is unknown.
            static_status = JointFeasibility::Unresolved;
        }
        let mut best = None;
        let mut candidates = CandidateDecisionCounts::default();
        let mut retained_gates = [None; 2];
        for index in 1..self.totals.len() {
            let joint = risks[index].classify(&self.config);
            let metrics = MetricClasses::from_risk(risks[index], &self.config);
            let mut record = |gate| {
                candidates.record(joint, metrics, gate);
                if let Some(slot) = retained_gates.get_mut(index - 1) {
                    *slot = Some(gate);
                }
            };
            if joint != JointFeasibility::Feasible {
                record(if joint == JointFeasibility::Infeasible {
                    SelectionGate::RiskInfeasible
                } else {
                    SelectionGate::RiskUnresolved
                });
                continue;
            }
            if self.totals[index].partial_window_unproven {
                record(SelectionGate::PartialWindowUnproven);
                continue;
            }
            let Some(time) = times[index] else {
                record(SelectionGate::TargetTimeUnavailable);
                continue;
            };
            if static_status == JointFeasibility::Feasible {
                if self.totals[index].window_debt {
                    record(SelectionGate::StaticFeasibleWindowDebt);
                    continue;
                }
                let (Some(static_time), Some(evidence)) = (times[0], paired[index - 1]) else {
                    record(SelectionGate::StaticTimeOrPairedEvidenceUnavailable);
                    continue;
                };
                let Some(gain) = static_time.checked_sub(time) else {
                    record(SelectionGate::CandidateNotFaster);
                    continue;
                };
                if evidence
                    .uncertainty
                    .checked_add(evidence.control_cost)
                    .is_none_or(|margin| gain <= margin)
                {
                    record(SelectionGate::GainNotAboveEvidenceMargin);
                    continue;
                }
            } else if static_status == JointFeasibility::Unresolved {
                record(SelectionGate::StaticUnresolved);
                continue;
            }
            record(SelectionGate::Eligible);
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
        let observation = DecisionObservation {
            choice,
            alternatives_total: self.totals.len() - 1,
            static_raw,
            static_effective: static_status,
            static_metrics: MetricClasses::from_risk(risks[0], &self.config),
            candidates,
            unresolved: (choice == PolicyChoice::Unresolved).then(|| UnresolvedDecisionWitness {
                revision: self.revision,
                alternatives_total: self.totals.len() - 1,
                static_raw,
                static_effective: static_status,
                partial_static_ranks: self.partial_static_ranks,
                endpoint_unresolved_mass: summary.unresolved_mass,
                failure_limits: [
                    1.0 - self.config.joint.ttft_quantile,
                    1.0 - self.config.joint.tpot_quantile,
                    1.0 - self.config.joint.itl_quantile,
                    1.0 - self.config.joint.min_completion_rate,
                ],
                product_success: self.product_success_evidence,
                static_action: self.totals[0].decision_evidence(
                    risks[0],
                    &self.config,
                    times[0],
                    None,
                ),
                alternatives: std::array::from_fn(|slot| {
                    let index = slot + 1;
                    self.totals.get(index).map(|totals| {
                        totals.decision_evidence(
                            risks[index],
                            &self.config,
                            times[index],
                            retained_gates[slot],
                        )
                    })
                }),
                details_omitted: (self.totals.len() - 1).saturating_sub(2),
            }),
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
            observation,
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
            totals.unfinished_request_mass += weight;
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
    totals.unfinished_request_mass += weight * members.len() as f64;
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
    fn decision_evidence(
        &self,
        risk: QueueRisk,
        config: &SchedulerSloConfig,
        target_time: Option<Duration>,
        first_gate: Option<SelectionGate>,
    ) -> ActionDecisionEvidence {
        ActionDecisionEvidence {
            joint: risk.classify(config),
            metrics: MetricClasses::from_risk(risk, config),
            denominators: RiskDenominators {
                requests: risk.request_count,
                tpot_known_eligible: self.tpot_eligible,
                tpot_unknown_eligible: self.tpot_unknown_eligible,
                tpot_unknown_outcome: self.tpot_unknown_outcome,
                tpot_undefined: self.tpot_undefined,
                itl_known_events: self.itl_events,
                itl_unknown_events: self.itl_unknown_events,
            },
            risk,
            target_time,
            known_target_time_mass: self.time_mass,
            unsupported_trajectory_mass: self.unsupported_mass,
            unfinished_request_mass: self.unfinished_request_mass,
            window_debt: self.window_debt,
            partial_window_unproven: self.partial_window_unproven,
            first_gate,
        }
    }

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
