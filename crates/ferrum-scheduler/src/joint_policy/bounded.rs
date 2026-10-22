//! Finite-prefix action scoring. These dimensionless, equally weighted proxies
//! are neither probabilities nor certificates of complete-request feasibility.
//! Unknown future tails stay in the cohort and never acquire an EOS endpoint.
//! ConditionalSnapshot assigns each rank's service time to both simulation
//! clocks, including cache hits. Thus central is already rank-specific, not a
//! repeated central estimate. This accumulator requires that paired contract.

use super::*;

#[derive(Debug, Clone, Copy, Default, serde::Serialize)]
pub struct BoundedScore {
    pub ttft_pressure: f64,
    pub tpot_time_debt: f64,
    pub itl_pressure: f64,
}

impl BoundedScore {
    pub fn total(self) -> f64 {
        self.ttft_pressure + self.tpot_time_debt + self.itl_pressure
    }

    fn add(&mut self, other: Self, weight: f64) -> Result<(), Unsupported> {
        self.ttft_pressure += other.ttft_pressure * weight;
        self.tpot_time_debt += other.tpot_time_debt * weight;
        self.itl_pressure += other.itl_pressure * weight;
        if !self.total().is_finite() {
            return Err(Unsupported::Arithmetic);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, Default, serde::Serialize)]
pub struct BoundedActionEvidence {
    /// Rank-weighted score, normalized by the original request count. Never
    /// normalize away an unsupported rank or divide by newly predicted events.
    pub score: BoundedScore,
    pub supported_mass: f64,
    pub unsupported_mass: f64,
    /// Rank-weighted count of requests still short of their declared endpoint.
    /// Even an endpoint reached in simulation is not an actual completion.
    /// Counts below describe supported prefixes only; unsupported_mass remains
    /// separate, so zero here is not evidence of complete cost support.
    pub unfinished_requests: f64,
    pub prefix_tpot_undefined_requests: f64,
    /// No throughput projection is invented for a static zero-progress waiter.
    pub waiting_exposure_only_requests: f64,
    pub prefill_progress_regressions: usize,
    pub overdue_prefill_regressions: usize,
}

#[derive(Debug)]
pub struct BoundedDecision {
    pub selected: Option<usize>,
    pub static_score: f64,
    pub selected_score: Option<f64>,
    pub valid_until: Instant,
    pub request_count: usize,
    pub observed_ranks: usize,
    /// Observed history remains diagnostic; an irreversible historical failure
    /// does not itself veto a future-improving bounded action.
    pub historical_ttft_late_requests: usize,
    pub historical_visible_intervals: usize,
    pub historical_bad_visible_intervals: usize,
    pub static_evidence: BoundedActionEvidence,
    pub alternatives: Vec<BoundedActionEvidence>,
}

pub struct BoundedAccumulator<'a> {
    initial: &'a [SimulatedRequest],
    config: SchedulerSloConfig,
    totals: Vec<BoundedActionEvidence>,
    revision: Option<u64>,
    ranks: Option<usize>,
    observed_ranks: usize,
    observed_mass: f64,
    quantile_upper: f64,
    invalidated: bool,
}

impl<'a> BoundedAccumulator<'a> {
    pub fn new(
        initial: &'a [SimulatedRequest],
        alternatives: usize,
        config: &SchedulerSloConfig,
    ) -> Result<Self, Unsupported> {
        let count = alternatives
            .checked_add(1)
            .ok_or(Unsupported::CandidateBudget)?;
        if config.validate().is_err() || initial.is_empty() {
            return Err(Unsupported::InvalidState);
        }
        if count > config.planning.max_candidates {
            return Err(Unsupported::CandidateBudget);
        }
        if initial.len() > config.planning.max_participants
            || Self::retained_bytes_bound(alternatives)
                .is_none_or(|bytes| bytes > config.planning.max_request_fact_bytes)
        {
            return Err(Unsupported::ParticipantBudget);
        }
        if initial.iter().enumerate().any(|(index, request)| {
            !request.valid()
                || request.central_clock != request.risk_clock
                || initial[..index].iter().any(|old| old.id == request.id)
                || (request.output_tokens == 0 && request.central_clock.first_output.is_some())
        }) {
            return Err(Unsupported::InvalidState);
        }
        Ok(Self {
            initial,
            config: *config,
            totals: vec![BoundedActionEvidence::default(); count],
            revision: None,
            ranks: None,
            observed_ranks: 0,
            observed_mass: 0.0,
            quantile_upper: 0.0,
            invalidated: false,
        })
    }

    /// Dynamic retained payload only. The caller must add this to its other
    /// simultaneously retained facts; the initial cohort is borrowed, not copied.
    pub fn retained_bytes_bound(alternatives: usize) -> Option<usize> {
        alternatives
            .checked_add(1)?
            .checked_mul(std::mem::size_of::<BoundedActionEvidence>())
    }

    pub fn observe(
        &mut self,
        rank: ScenarioRank,
        comparison: &Comparison,
    ) -> Result<(), Unsupported> {
        if self.invalidated {
            return Err(Unsupported::InvalidState);
        }
        // A failed observation may have accumulated earlier actions. It cannot
        // be retried or completed by silently retaining those partial totals.
        self.invalidated = true;
        if rank.index != self.observed_ranks
            || rank.count == 0
            || rank.count > self.config.planning.max_scenarios
            || self.ranks.is_some_and(|count| count != rank.count)
            || rank.index >= rank.count
            || !rank.weight.is_finite()
            || rank.weight <= 0.0
            || rank.weight > 1.0
            || !rank.quantile_lower.is_finite()
            || !rank.quantile_upper.is_finite()
            || (rank.quantile_lower - self.quantile_upper).abs() > 1e-9
            || (rank.quantile_upper - rank.quantile_lower - rank.weight).abs() > 1e-9
            || rank.quantile_upper > 1.0 + 1e-9
            || self
                .revision
                .is_some_and(|revision| revision != comparison.revision)
            || comparison.alternatives.len() + 1 != self.totals.len()
        {
            return Err(Unsupported::InvalidState);
        }
        // Complete here means the bounded traversal returned normally. It does
        // not mean all requests reached their endpoints. targets/window are not
        // used as a fictitious common wall-clock or common-work certificate.
        let baseline = comparison
            .static_result
            .complete()
            .ok_or(Unsupported::InvalidState)?;
        validate(self.initial, baseline)?;
        let reference = score(self.initial, baseline, baseline, &self.config)?;
        add_evidence(&mut self.totals[0], reference, rank.weight)?;
        for (index, result) in comparison.alternatives.iter().enumerate() {
            match result {
                Ok(result) => {
                    validate(self.initial, result)?;
                    let evidence = score(self.initial, baseline, result, &self.config)?;
                    add_evidence(&mut self.totals[index + 1], evidence, rank.weight)?;
                }
                Err(reason) if reason.is_hard_budget_failure() => return Err(*reason),
                Err(_) => self.totals[index + 1].unsupported_mass += rank.weight,
            }
        }
        self.revision = Some(comparison.revision);
        self.ranks = Some(rank.count);
        self.observed_ranks += 1;
        self.observed_mass += rank.weight;
        self.quantile_upper = rank.quantile_upper;
        self.invalidated = false;
        Ok(())
    }

    pub fn finish(mut self, summary: ScenarioEvaluation) -> Result<BoundedDecision, Unsupported> {
        if self.invalidated
            || self.ranks != Some(summary.ranks)
            || self.observed_ranks != summary.ranks
            || summary.unresolved_ranks != 0
            || summary.partial_static_ranks != 0
            || summary.dependence != CostScenarioDependence::SharedResidualRank
            || !summary.resolved_mass.is_finite()
            || !summary.unresolved_mass.is_finite()
            || summary.unresolved_mass != 0.0
            || (summary.resolved_mass - 1.0).abs() > 1e-9
            || (self.observed_mass - 1.0).abs() > 1e-9
            || (self.quantile_upper - 1.0).abs() > 1e-9
        {
            return Err(Unsupported::InvalidState);
        }
        let baseline = self.totals.remove(0);
        let static_score = baseline.score.total();
        let mut best = static_score;
        let mut selected = None;
        for (index, candidate) in self.totals.iter().enumerate() {
            if candidate.unsupported_mass != 0.0
                || (candidate.supported_mass - 1.0).abs() > 1e-9
                || candidate.prefill_progress_regressions != 0
                || candidate.overdue_prefill_regressions != 0
            {
                continue;
            }
            let value = candidate.score.total();
            if value < best {
                best = value;
                selected = Some(index);
            }
        }
        let ttft = Duration::try_from_secs_f64(self.config.ttft_ms / 1000.0)
            .map_err(|_| Unsupported::InvalidState)?;
        let intervals = |bad: bool| {
            self.initial.iter().try_fold(0usize, |sum, request| {
                sum.checked_add(if bad {
                    request.central_clock.violating_interval_count
                } else {
                    request.central_clock.interval_count
                })
                .ok_or(Unsupported::Arithmetic)
            })
        };
        Ok(BoundedDecision {
            selected,
            static_score,
            selected_score: selected.map(|_| best),
            valid_until: summary.valid_until,
            request_count: self.initial.len(),
            observed_ranks: self.observed_ranks,
            historical_ttft_late_requests: self
                .initial
                .iter()
                .filter(|request| request.central_clock.first_output.unwrap_or(request.age) > ttft)
                .count(),
            historical_visible_intervals: intervals(false)?,
            historical_bad_visible_intervals: intervals(true)?,
            static_evidence: baseline,
            alternatives: self.totals,
        })
    }
}

fn add_evidence(
    total: &mut BoundedActionEvidence,
    value: BoundedActionEvidence,
    weight: f64,
) -> Result<(), Unsupported> {
    total.score.add(value.score, weight)?;
    total.supported_mass += weight;
    total.unfinished_requests += weight * value.unfinished_requests;
    total.prefix_tpot_undefined_requests += weight * value.prefix_tpot_undefined_requests;
    total.waiting_exposure_only_requests += weight * value.waiting_exposure_only_requests;
    total.prefill_progress_regressions = total
        .prefill_progress_regressions
        .checked_add(value.prefill_progress_regressions)
        .ok_or(Unsupported::Arithmetic)?;
    total.overdue_prefill_regressions = total
        .overdue_prefill_regressions
        .checked_add(value.overdue_prefill_regressions)
        .ok_or(Unsupported::Arithmetic)?;
    Ok(())
}

fn validate(initial: &[SimulatedRequest], result: &TrajectoryResult) -> Result<(), Unsupported> {
    if result.requests.len() != initial.len()
        || result.central_elapsed != result.risk_elapsed
        || result.cost_risk.conditional_scenario != Some(CostScenarioDependence::SharedResidualRank)
        || result.cost_risk.empirical_quantiles
        || result.cost_risk.union_failure_upper != 0.0
    {
        return Err(Unsupported::InvalidState);
    }
    for (before, after) in initial.iter().zip(&result.requests) {
        let now = before
            .age
            .checked_add(result.central_elapsed)
            .ok_or(Unsupported::Arithmetic)?;
        let old = before.central_clock;
        let clock = after.central_clock;
        if after.id != before.id
            || after.central_clock != after.risk_clock
            || after.prompt_tokens != before.prompt_tokens
            || after.output_cap != before.output_cap
            || after.scenario_end_tokens != before.scenario_end_tokens
            || after.age != before.age
            || after.prefill_tokens < before.prefill_tokens
            || after.prefill_tokens > after.prompt_tokens
            || after.output_tokens < before.output_tokens
            || after.output_tokens > after.scenario_end_tokens
            || (after.output_tokens > 0 && after.prefill_tokens != after.prompt_tokens)
            || (old.first_output.is_some() && old.first_output != clock.first_output)
            || clock.last_output < old.last_output
            || clock.first_output > clock.last_output
            || clock.last_output.is_some_and(|last| last > now)
            || (after.output_tokens > 0) != clock.first_output.is_some()
            || (after.output_tokens > 0) != clock.last_output.is_some()
            || (old.completed.is_some() && old.completed != clock.completed)
            || clock.interval_count < old.interval_count
            || clock.violating_interval_count < old.violating_interval_count
            || clock.violating_interval_count > clock.interval_count
            || clock.interval_count > after.output_tokens.saturating_sub(1)
            || clock.completed.is_some_and(|end| {
                end > now
                    || after.output_tokens != after.scenario_end_tokens
                    || clock.last_output.is_none_or(|last| end < last)
            })
        {
            return Err(Unsupported::InvalidState);
        }
    }
    Ok(())
}

fn score(
    initial: &[SimulatedRequest],
    baseline: &TrajectoryResult,
    candidate: &TrajectoryResult,
    config: &SchedulerSloConfig,
) -> Result<BoundedActionEvidence, Unsupported> {
    let mut evidence = BoundedActionEvidence::default();
    let ttft = config.ttft_ms / 1000.0;
    let tpot = config.tpot_ms / 1000.0;
    let itl = config.itl_ms / 1000.0;
    for ((before, reference), after) in initial
        .iter()
        .zip(&baseline.requests)
        .zip(&candidate.requests)
    {
        let now = before
            .age
            .checked_add(candidate.central_elapsed)
            .ok_or(Unsupported::Arithmetic)?;
        let clock = after.central_clock;
        if before.central_clock.first_output.is_none() {
            let static_progress = reference.prefill_tokens - before.prefill_tokens;
            let progress = after.prefill_tokens - before.prefill_tokens;
            if static_progress > 0 && progress == 0 {
                evidence.prefill_progress_regressions += 1;
            }
            if before.age.as_secs_f64() > ttft && progress < static_progress {
                evidence.overdue_prefill_regressions += 1;
            }
            let projected_first = if let Some(first) = clock.first_output {
                first.as_secs_f64()
            } else if static_progress > 0 && progress > 0 {
                // A local service-rate extrapolation, explicitly not a bound
                // on remaining service, admission, completion, or natural EOS.
                now.as_secs_f64()
                    + (after.prompt_tokens - after.prefill_tokens) as f64
                        * candidate.central_elapsed.as_secs_f64()
                        / progress as f64
            } else {
                evidence.waiting_exposure_only_requests += 1.0;
                if static_progress == 0
                    && progress == 0
                    && reference.central_clock.first_output.is_none()
                {
                    // Neither prefix served this waiter. A shorter candidate
                    // observation must not earn TTFT credit for doing nothing.
                    // This reference floor is a ranking safeguard only: it
                    // neither extends simulation nor predicts first output.
                    before
                        .age
                        .checked_add(baseline.central_elapsed.max(candidate.central_elapsed))
                        .ok_or(Unsupported::Arithmetic)?
                        .as_secs_f64()
                } else {
                    now.as_secs_f64()
                }
            };
            evidence.score.ttft_pressure += (projected_first / ttft - 1.0).max(0.0);
        }
        if after.output_tokens < 2 {
            evidence.prefix_tpot_undefined_requests += 1.0;
        }
        if let Some(first) = before.central_clock.first_output {
            let end = if clock.completed.is_some() {
                clock.last_output.ok_or(Unsupported::InvalidState)?
            } else {
                now
            };
            let elapsed = end.checked_sub(first).ok_or(Unsupported::InvalidState)?;
            // Prefix time debt, not a terminal TPOT. With only one emitted
            // token the average is undefined, but its open wait still matters.
            evidence.score.tpot_time_debt += (elapsed.as_secs_f64() / tpot
                - after.output_tokens.saturating_sub(1) as f64)
                .max(0.0);
        }
        evidence.score.itl_pressure +=
            (clock.violating_interval_count - before.central_clock.violating_interval_count) as f64;
        if clock.completed.is_none() {
            if let Some(last) = clock.last_output {
                let open = now.checked_sub(last).ok_or(Unsupported::InvalidState)?;
                evidence.score.itl_pressure += (open.as_secs_f64() / itl - 1.0).max(0.0);
            }
        }
        if after.output_tokens < after.scenario_end_tokens || clock.completed.is_none() {
            evidence.unfinished_requests += 1.0;
        }
    }
    let count = initial.len() as f64;
    evidence.score.ttft_pressure /= count;
    evidence.score.tpot_time_debt /= count;
    evidence.score.itl_pressure /= count;
    if !evidence.score.total().is_finite() {
        return Err(Unsupported::Arithmetic);
    }
    Ok(evidence)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> SchedulerSloConfig {
        "ttft:100,tpot:10,itl:10".parse().unwrap()
    }
    fn request(prefill: bool) -> SimulatedRequest {
        let mut request = SimulatedRequest {
            id: RequestId::new(),
            prompt_tokens: 10,
            prefill_tokens: 0,
            output_tokens: 0,
            output_cap: 64,
            scenario_end_tokens: 64,
            age: Duration::from_millis(20),
            central_clock: RequestClock::default(),
            risk_clock: RequestClock::default(),
        };
        if !prefill {
            request.prefill_tokens = 10;
            request.output_tokens = 1;
            request.central_clock.first_output = Some(Duration::from_millis(10));
            request.central_clock.last_output = Some(Duration::from_millis(10));
            request.risk_clock = request.central_clock;
        }
        request
    }
    fn result(requests: Vec<SimulatedRequest>, ms: u64) -> TrajectoryResult {
        TrajectoryResult {
            window_progress: requests.iter().map(SimulatedRequest::progress).collect(),
            requests,
            target_reached_at: Duration::from_millis(ms),
            central_elapsed: Duration::from_millis(ms),
            risk_elapsed: Duration::from_millis(ms),
            queries: 1,
            cost_risk: CostRiskBudget {
                conditional_scenario: Some(CostScenarioDependence::SharedResidualRank),
                ..Default::default()
            },
        }
    }
    fn comparison(
        base: TrajectoryResult,
        candidates: Vec<Result<TrajectoryResult, Unsupported>>,
    ) -> Comparison {
        Comparison {
            revision: 1,
            window: StaticWindow::Exact(base.central_elapsed),
            targets: base.window_progress.clone(),
            static_result: StaticTrajectory::Complete(base),
            alternatives: candidates,
        }
    }
    fn rank(index: usize, count: usize) -> ScenarioRank {
        ScenarioRank {
            index,
            count,
            weight: 1.0 / count as f64,
            quantile_lower: index as f64 / count as f64,
            quantile_upper: (index + 1) as f64 / count as f64,
        }
    }
    fn summary(ranks: usize) -> ScenarioEvaluation {
        ScenarioEvaluation {
            valid_until: Instant::now() + Duration::from_secs(1),
            dependence: CostScenarioDependence::SharedResidualRank,
            ranks,
            queries: ranks,
            cache_hits: 0,
            cache_retained_bytes: 0,
            resolved_mass: 1.0,
            unresolved_mass: 0.0,
            unresolved_ranks: 0,
            partial_static_ranks: 0,
        }
    }
    fn emit(request: &mut SimulatedRequest, at: u64, bad: bool) {
        let at = Duration::from_millis(at);
        if let Some(last) = request.central_clock.last_output {
            request.central_clock.maximum_output_interval = request
                .central_clock
                .maximum_output_interval
                .max(at.checked_sub(last).unwrap());
        }
        request.output_tokens += 1;
        request.central_clock.last_output = Some(at);
        request.central_clock.interval_count += 1;
        request.central_clock.violating_interval_count += usize::from(bad);
        request.risk_clock = request.central_clock;
    }

    #[test]
    fn bounded_waiting_tail_stays_in_denominator_without_vetoing_local_improvement() {
        let mut initial = [request(false), request(true)];
        initial[1].age = Duration::from_millis(110);
        let mut baseline = initial.to_vec();
        emit(&mut baseline[0], 50, true);
        let mut candidate = initial.to_vec();
        emit(&mut candidate[0], 25, true);
        let cmp = comparison(result(baseline, 30), vec![Ok(result(candidate, 5))]);
        let mut accumulator = BoundedAccumulator::new(&initial, 1, &config()).unwrap();
        accumulator.observe(rank(0, 1), &cmp).unwrap();
        let decision = accumulator.finish(summary(1)).unwrap();
        assert_eq!(decision.selected, Some(0));
        assert_eq!(decision.request_count, 2);
        assert_eq!(decision.alternatives[0].unfinished_requests, 2.0);
        assert_eq!(decision.alternatives[0].waiting_exposure_only_requests, 1.0);
        assert_eq!(decision.alternatives[0].prefix_tpot_undefined_requests, 1.0);
        // The overdue, unserved waiter earns no benefit from observing a
        // shorter prefix. Actual decoder time debt still improves the score.
        assert_eq!(
            decision.alternatives[0].score.ttft_pressure,
            decision.static_evidence.score.ttft_pressure
        );
        assert!(decision.selected_score.unwrap() < decision.static_score);
    }

    #[test]
    fn bounded_first_wave_limit_cannot_gain_from_a_shorter_unserved_waiter_prefix() {
        use ferrum_execution::{
            execution_attempt::AttemptPrefixRestore, token_execution::ExecutionRange,
        };

        struct Costs(Instant);
        impl CostSnapshot for Costs {
            fn revision(&self) -> u64 {
                1
            }
            fn query_limits(&self) -> CostQueryLimits {
                CostQueryLimits::default()
            }
            fn predict(&self, _: &CostQuery<'_>, _: Instant) -> CostPrediction {
                panic!("the scenario path must use conditional costs")
            }
            fn cost_scenarios(
                &self,
                query: &CostQuery<'_>,
                _: Instant,
                count: usize,
            ) -> CostScenarioPrediction {
                assert_eq!(query.work.len(), 1);
                let work = query.work[0];
                assert_eq!(work.kind, AttemptWorkKind::Prefill);
                // A smaller invocation is faster, but pays the same overhead:
                // 128 tokens take 50 ms, while 32 tokens take 20 ms.
                let service_time = Duration::from_millis(10)
                    + Duration::from_nanos(
                        312_500 * u64::try_from(work.requested.as_range().len()).unwrap(),
                    );
                CostScenarioPrediction::Known {
                    central_service_time: service_time,
                    scenarios: (0..count)
                        .map(|rank| CostScenario {
                            rank,
                            weight: 1.0 / count as f64,
                            service_time,
                        })
                        .collect(),
                    dependence: CostScenarioDependence::SharedResidualRank,
                    support: CostSupport::ExactObservedWork,
                    samples: count,
                    observed_at: self.0,
                    expires_at: self.0 + Duration::from_secs(60),
                }
            }
        }
        struct FirstWaveLimit(AdmissionSlotPlan);
        impl TrajectoryAdapter<usize> for FirstWaveLimit {
            fn admission_slots(&self) -> Option<&AdmissionSlotPlan> {
                Some(&self.0)
            }
            fn plan(
                &self,
                action: Option<&usize>,
                requests: &[SimulatedRequest],
                plan: &mut PlanBuilder,
            ) -> Result<(), Unsupported> {
                let active = plan.active_mask().unwrap();
                assert_eq!(active, [true, false]);
                let request = &requests[0];
                // The action changes the first wave only. Every later wave
                // returns to the same 128-token continuation, not a phase cap.
                let tokens = action
                    .copied()
                    .unwrap_or(128)
                    .min(request.prompt_tokens - request.prefill_tokens);
                plan.push_group(&[GroupMember {
                    request: 0,
                    work: CostWork {
                        kind: AttemptWorkKind::Prefill,
                        requested: ExecutionRange::new(request.prefill_tokens, tokens).unwrap(),
                        context_tokens: request.prompt_tokens,
                        prefix_restore: AttemptPrefixRestore::NoneObserved,
                        internally_reused_tokens: None,
                    },
                }])
            }
        }
        let mut initial = [request(true), request(true)];
        initial[0].prompt_tokens = 2048;
        initial[1].age = Duration::from_millis(110);
        let adapter = FirstWaveLimit(
            AdmissionSlotPlan::new(
                &[
                    InitialAdmission::Resident,
                    InitialAdmission::Waiting { order: 0 },
                ],
                1,
                initial.len(),
            )
            .unwrap(),
        );
        let mut config = config();
        config.planning.max_elapsed_us = 1_000_000;
        config.planning.max_scenarios = 2;
        let snapshot = Costs(Instant::now());
        let mut accumulator = BoundedAccumulator::new(&initial, 1, &config).unwrap();
        let summary = SloPolicy::evaluate_scenarios_with_requests(
            EvaluationInput {
                snapshot: &snapshot,
                now: snapshot.0,
                config: &config,
                execution: CostExecution::SingleInvocation,
                requests: &initial,
                resources: ResourceSupport {
                    waiting_requests: 1,
                    kv_reserved_through_scenario_end: true,
                },
                adapter: &adapter,
                static_action: &128,
                alternatives: &[32],
                horizon: EvaluationHorizon::BoundedRounds(8),
            },
            |_, _| Ok(()),
            |rank, comparison| {
                let baseline = comparison.static_result.complete().unwrap();
                let candidate = comparison.alternatives[0].as_ref().unwrap();
                assert_eq!(baseline.requests[0].prefill_tokens, 1024);
                assert_eq!(candidate.requests[0].prefill_tokens, 32 + 7 * 128);
                assert_eq!(baseline.central_elapsed, Duration::from_millis(400));
                assert_eq!(candidate.central_elapsed, Duration::from_millis(370));
                for result in [baseline, candidate] {
                    assert_eq!(result.requests[1].progress(), initial[1].progress());
                    assert!(result
                        .requests
                        .iter()
                        .all(|request| request.output_tokens == 0));
                }
                accumulator.observe(rank, comparison).unwrap();
            },
        )
        .unwrap();
        let decision = accumulator.finish(summary).unwrap();
        let candidate = decision.alternatives[0];
        assert_eq!(candidate.prefill_progress_regressions, 0);
        assert_eq!(candidate.overdue_prefill_regressions, 0);
        assert_eq!(candidate.unfinished_requests, 2.0);
        assert_eq!(candidate.waiting_exposure_only_requests, 1.0);
        assert!((decision.static_score - 5.65).abs() < 1e-12);
        assert!(candidate.score.total() > decision.static_score);
        // Without the reference floor, the unserved waiter's 30 ms shorter
        // observation would falsely outweigh the slower prefill service rate.
        let undeserved_waiter_credit = 30.0 / config.ttft_ms / initial.len() as f64;
        assert!(candidate.score.total() - undeserved_waiter_credit < decision.static_score);
        assert_eq!(decision.selected, None);
    }

    #[test]
    fn bounded_prefill_progress_and_overdue_guard_cannot_be_paid_by_prediction() {
        for (age, candidate_progress) in [(20, 0), (101, 1)] {
            let mut initial = [request(false), request(true)];
            initial[1].age = Duration::from_millis(age);
            let mut baseline = initial.to_vec();
            baseline[1].prefill_tokens = 2;
            emit(&mut baseline[0], 50, true);
            let mut candidate = initial.to_vec();
            candidate[1].prefill_tokens = candidate_progress;
            emit(&mut candidate[0], 25, true);
            let cmp = comparison(result(baseline, 30), vec![Ok(result(candidate, 5))]);
            let mut accumulator = BoundedAccumulator::new(&initial, 1, &config()).unwrap();
            accumulator.observe(rank(0, 1), &cmp).unwrap();
            let decision = accumulator.finish(summary(1)).unwrap();
            assert_eq!(decision.selected, None);
            assert!(
                decision.alternatives[0].prefill_progress_regressions
                    + decision.alternatives[0].overdue_prefill_regressions
                    > 0
            );
            assert_eq!(initial[1].prefill_tokens, 0);
        }
    }

    #[test]
    fn bounded_history_is_retained_without_erasing_improvable_future_or_open_gap() {
        let mut initial = [request(false)];
        initial[0].age = Duration::from_millis(130);
        initial[0].output_tokens = 2;
        initial[0].central_clock = RequestClock {
            first_output: Some(Duration::from_millis(110)),
            last_output: Some(Duration::from_millis(130)),
            interval_count: 1,
            violating_interval_count: 1,
            maximum_output_interval: Duration::from_millis(20),
            completed: None,
        };
        initial[0].risk_clock = initial[0].central_clock;
        let mut baseline = initial.to_vec();
        emit(&mut baseline[0], 160, true);
        let mut candidate = initial.to_vec();
        emit(&mut candidate[0], 135, false);
        let cmp = comparison(result(baseline, 30), vec![Ok(result(candidate, 5))]);
        let mut accumulator = BoundedAccumulator::new(&initial, 1, &config()).unwrap();
        accumulator.observe(rank(0, 1), &cmp).unwrap();
        let decision = accumulator.finish(summary(1)).unwrap();
        assert_eq!(decision.selected, Some(0));
        assert_eq!(decision.historical_ttft_late_requests, 1);
        assert_eq!(decision.historical_bad_visible_intervals, 1);
        assert_eq!(decision.alternatives[0].score.itl_pressure, 0.0);
        let open = score(
            &initial,
            &result(initial.to_vec(), 0),
            &result(initial.to_vec(), 30),
            &config(),
        )
        .unwrap();
        assert_eq!(open.score.itl_pressure, 2.0);
        assert!(open.score.tpot_time_debt > 0.0);
    }

    #[test]
    fn bounded_unknown_rank_budget_and_tie_do_not_become_selected_success() {
        let initial = [request(false)];
        let base = || result(initial.to_vec(), 20);
        let mut candidate = initial.to_vec();
        emit(&mut candidate[0], 25, true);
        let known = comparison(base(), vec![Ok(result(candidate, 5))]);
        let missing = comparison(
            base(),
            vec![Err(Unsupported::Cost(CostUnknownReason::NoCoverage))],
        );
        let mut accumulator = BoundedAccumulator::new(&initial, 1, &config()).unwrap();
        accumulator.observe(rank(0, 2), &known).unwrap();
        accumulator.observe(rank(1, 2), &missing).unwrap();
        let decision = accumulator.finish(summary(2)).unwrap();
        assert_eq!(decision.selected, None);
        assert_eq!(decision.alternatives[0].unsupported_mass, 0.5);
        let mut accumulator = BoundedAccumulator::new(&initial, 1, &config()).unwrap();
        accumulator
            .observe(rank(0, 1), &comparison(base(), vec![Ok(base())]))
            .unwrap();
        assert_eq!(accumulator.finish(summary(1)).unwrap().selected, None);
        let mut accumulator = BoundedAccumulator::new(&initial, 1, &config()).unwrap();
        assert_eq!(
            accumulator.observe(
                rank(0, 1),
                &comparison(base(), vec![Err(Unsupported::QueryBudget)])
            ),
            Err(Unsupported::QueryBudget)
        );
    }

    #[test]
    fn bounded_cost_tail_rank_changes_choice_without_renormalizing_it_away() {
        let initial = [request(false)];
        let baseline = || {
            let mut requests = initial.to_vec();
            emit(&mut requests[0], 40, true);
            result(requests, 20)
        };
        let mut fast = initial.to_vec();
        emit(&mut fast[0], 25, true);
        let fast = comparison(baseline(), vec![Ok(result(fast, 5))]);
        let mut slow = initial.to_vec();
        emit(&mut slow[0], 120, true);
        let slow = comparison(baseline(), vec![Ok(result(slow, 100))]);

        // Both are valid supported prefixes with equal central/risk clocks;
        // the second rank has a genuinely later output and larger elapsed time.
        let mut fast_only = BoundedAccumulator::new(&initial, 1, &config()).unwrap();
        fast_only.observe(rank(0, 1), &fast).unwrap();
        assert_eq!(fast_only.finish(summary(1)).unwrap().selected, Some(0));

        let mut all_ranks = BoundedAccumulator::new(&initial, 1, &config()).unwrap();
        all_ranks.observe(rank(0, 2), &fast).unwrap();
        all_ranks.observe(rank(1, 2), &slow).unwrap();
        let decision = all_ranks.finish(summary(2)).unwrap();
        assert_eq!(decision.selected, None);
        assert_eq!(decision.observed_ranks, 2);
        assert!((decision.static_score - 3.0).abs() < 1e-12);
        assert!((decision.alternatives[0].score.total() - 6.25).abs() < 1e-12);
        assert_eq!(decision.alternatives[0].supported_mass, 1.0);
        assert_eq!(decision.alternatives[0].unsupported_mass, 0.0);
    }

    #[test]
    fn bounded_rank_and_request_identity_are_not_silently_repaired() {
        let initial = [request(false)];
        let mut changed = initial.to_vec();
        changed[0].id = RequestId::new();
        let cmp = comparison(result(initial.to_vec(), 10), vec![Ok(result(changed, 5))]);
        let mut accumulator = BoundedAccumulator::new(&initial, 1, &config()).unwrap();
        assert_eq!(
            accumulator.observe(rank(0, 1), &cmp),
            Err(Unsupported::InvalidState)
        );
        let valid = comparison(
            result(initial.to_vec(), 10),
            vec![Ok(result(initial.to_vec(), 5))],
        );
        assert_eq!(
            accumulator.observe(rank(0, 1), &valid),
            Err(Unsupported::InvalidState)
        );
        assert!(matches!(
            accumulator.finish(summary(1)),
            Err(Unsupported::InvalidState)
        ));
        let mut accumulator = BoundedAccumulator::new(&initial, 0, &config()).unwrap();
        let cmp = comparison(result(initial.to_vec(), 10), vec![]);
        assert_eq!(
            accumulator.observe(rank(1, 2), &cmp),
            Err(Unsupported::InvalidState)
        );
        let mut mismatched = result(initial.to_vec(), 10);
        mismatched.risk_elapsed = Duration::from_millis(11);
        let mut accumulator = BoundedAccumulator::new(&initial, 0, &config()).unwrap();
        assert_eq!(
            accumulator.observe(rank(0, 1), &comparison(mismatched, vec![])),
            Err(Unsupported::InvalidState)
        );
        let mut mismatched = result(initial.to_vec(), 10);
        mismatched.requests[0].risk_clock.last_output = Some(Duration::from_millis(11));
        let mut accumulator = BoundedAccumulator::new(&initial, 0, &config()).unwrap();
        assert_eq!(
            accumulator.observe(rank(0, 1), &comparison(mismatched, vec![])),
            Err(Unsupported::InvalidState)
        );
    }
}
