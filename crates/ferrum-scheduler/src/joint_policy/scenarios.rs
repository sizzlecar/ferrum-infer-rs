use super::*;
use std::sync::atomic::{AtomicUsize, Ordering};
mod cache;
use cache::ScenarioCache;

#[cfg(test)]
mod tests;

/// Complete traversal of the declared empirical rank support. The callback's
/// partial aggregates must be discarded if traversal returns an error. Each
/// comparison can still contain unsupported alternatives or a static support
/// prefix; their unknown suffixes remain explicit and are never renormalized.
#[derive(Debug, Clone, Copy)]
pub struct ScenarioEvaluation {
    /// Earliest expiry of any conditional cost distribution actually queried.
    pub valid_until: Instant,
    pub dependence: CostScenarioDependence,
    pub ranks: usize,
    pub queries: usize,
    pub cache_hits: usize,
    pub cache_retained_bytes: usize,
    /// Mass with assigned stopping endpoints and a visited comparison. This
    /// does not assert complete cost support for static or every alternative.
    pub resolved_mass: f64,
    pub unresolved_mass: f64,
    pub unresolved_ranks: usize,
    pub partial_static_ranks: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct ScenarioRank {
    pub index: usize,
    pub count: usize,
    pub weight: f64,
    pub quantile_lower: f64,
    pub quantile_upper: f64,
}

impl SloPolicy {
    /// Visit one rank at a time without retaining rank × candidate × cohort
    /// copies. A single global query/time budget covers seed and every rank.
    /// Shared residual rank is a sensitivity assumption, not joint calibration
    /// coverage, and these differences never substitute for paired evidence.
    pub fn evaluate_scenarios<Action, Adapter: TrajectoryAdapter<Action>>(
        input: EvaluationInput<'_, Action, Adapter>,
        mut visit: impl FnMut(usize, f64, &Comparison),
    ) -> Result<ScenarioEvaluation, Unsupported> {
        Self::evaluate_scenarios_with_requests(
            input,
            |_, _| Ok(()),
            |rank, comparison| {
                visit(rank.index, rank.weight, comparison);
            },
        )
    }

    /// Assign discrete stopping endpoints for the same global empirical cost
    /// rank. Only `scenario_end_tokens` may change. Unsupported assignments
    /// retain their entire weight as unresolved; they are never filled at cap
    /// or removed by renormalizing the remaining rank weights.
    pub fn evaluate_scenarios_with_requests<Action, Adapter: TrajectoryAdapter<Action>>(
        input: EvaluationInput<'_, Action, Adapter>,
        assign: impl FnMut(ScenarioRank, &mut [SimulatedRequest]) -> Result<(), Unsupported>,
        visit: impl FnMut(ScenarioRank, &Comparison),
    ) -> Result<ScenarioEvaluation, Unsupported> {
        Self::evaluate_scenarios_with_diagnostics(
            input,
            assign,
            visit,
            &mut ScenarioDiagnostics::default(),
        )
    }

    /// The fixed-size receipt is populated on every return. Partial traversal
    /// remains unusable for feasibility, but its actual work is observable.
    pub fn evaluate_scenarios_with_diagnostics<Action, Adapter: TrajectoryAdapter<Action>>(
        input: EvaluationInput<'_, Action, Adapter>,
        assign: impl FnMut(ScenarioRank, &mut [SimulatedRequest]) -> Result<(), Unsupported>,
        visit: impl FnMut(ScenarioRank, &Comparison),
        report: &mut ScenarioDiagnostics,
    ) -> Result<ScenarioEvaluation, Unsupported> {
        let recorder = diagnostics::Recorder::new(input.config.planning.max_queries);
        let result = Self::evaluate_scenarios_recorded(input, assign, visit, &recorder);
        *report = recorder.finish(result.as_ref().err().copied());
        result
    }

    fn evaluate_scenarios_recorded<Action, Adapter: TrajectoryAdapter<Action>>(
        input: EvaluationInput<'_, Action, Adapter>,
        mut assign: impl FnMut(ScenarioRank, &mut [SimulatedRequest]) -> Result<(), Unsupported>,
        mut visit: impl FnMut(ScenarioRank, &Comparison),
        diagnostics: &diagnostics::Recorder,
    ) -> Result<ScenarioEvaluation, Unsupported> {
        if input.requests.is_empty() {
            return Err(Unsupported::EmptyCohort);
        }
        if input.config.planning.max_queries == 0 {
            return Err(Unsupported::QueryBudget);
        }
        if input.requests.len() > input.config.planning.max_participants {
            return Err(Unsupported::ParticipantBudget);
        }
        admission::validate_admission(&input)?;
        if !input.resources.kv_reserved_through_scenario_end {
            return Err(Unsupported::UnknownKvCapacity);
        }
        let started = Instant::now();
        let limits = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            input.snapshot.query_limits()
        }))
        .map_err(|_| {
            let reason = Unsupported::Cost(CostUnknownReason::UnsupportedDomain);
            diagnostics.failure(reason, FailureOrigin::ProviderPanic);
            reason
        })?;
        let mut plan = PlanBuilder {
            groups: Vec::new(),
            participant_limit: input.config.planning.max_participants,
            query_limits: input.config.query.intersect(limits),
            execution: input.execution,
            active: input
                .adapter
                .admission_slots()
                .map(|slots| slots.active_mask(input.requests))
                .transpose()?,
        };
        input
            .adapter
            .plan(Some(input.static_action), input.requests, &mut plan)?;
        let group = plan.groups.first().ok_or(Unsupported::NoProgress)?;
        for member in group {
            validate_member(input.requests, *member)?;
        }
        let work = group.iter().map(|member| member.work).collect::<Vec<_>>();
        diagnostics.point(
            EvaluationPoint {
                stage: EvaluationStage::Seed,
                candidate: Some(0),
                round: Some(0),
                group: Some(0),
                ..EvaluationPoint::default()
            },
            0,
        );
        diagnostics.provider_call();
        let seed = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            input.snapshot.cost_scenarios(
                &CostQuery {
                    work: &work,
                    work_is_exact: true,
                    execution: CostExecution::SingleInvocation,
                },
                input.now,
                input.config.planning.max_scenarios,
            )
        }))
        .map_err(|_| {
            let reason = Unsupported::Cost(CostUnknownReason::UnsupportedDomain);
            diagnostics.failure(reason, FailureOrigin::ProviderPanic);
            reason
        })?;
        let seed_was_known = matches!(seed, CostScenarioPrediction::Known { .. });
        let seed = seed.validated(input.now, input.config.planning.max_scenarios);
        let (weights, dependence, seed_expiry) = match &seed {
            CostScenarioPrediction::Known {
                scenarios,
                dependence,
                expires_at,
                ..
            } => (
                scenarios
                    .iter()
                    .map(|scenario| scenario.weight)
                    .collect::<Vec<_>>(),
                *dependence,
                *expires_at,
            ),
            CostScenarioPrediction::Unknown(reason) => {
                diagnostics.failure(
                    Unsupported::Cost(*reason),
                    if seed_was_known {
                        FailureOrigin::ScenarioContract
                    } else {
                        FailureOrigin::Provider
                    },
                );
                return Err(Unsupported::Cost(*reason));
            }
        };
        let queries = AtomicUsize::new(1);
        let mut cache = ScenarioCache::new(
            input.now,
            input.config.planning.max_cost_cache_bytes,
            input.config.planning.max_queries,
        );
        cache.insert(
            &CostQuery {
                work: &work,
                work_is_exact: true,
                execution: CostExecution::SingleInvocation,
            },
            seed,
        );
        diagnostics.cache_bytes(cache.retained_bytes());
        let cache = parking_lot::Mutex::new(cache);
        let valid_until = parking_lot::Mutex::new(seed_expiry);
        let mut resolved_mass = 0.0;
        let mut unresolved_mass = 0.0;
        let mut unresolved_ranks = 0;
        let mut partial_static_ranks = 0;
        let mut quantile_lower = 0.0;
        for (rank, weight) in weights.iter().copied().enumerate() {
            diagnostics.point(
                EvaluationPoint {
                    stage: EvaluationStage::RankAssignment,
                    rank: Some(rank),
                    ..EvaluationPoint::default()
                },
                0,
            );
            if started.elapsed() >= Duration::from_micros(input.config.planning.max_elapsed_us) {
                diagnostics.failure(Unsupported::TimeBudget, FailureOrigin::PreQueryTime);
                return Err(Unsupported::TimeBudget);
            }
            diagnostics.rank_attempt();
            let rank_info = ScenarioRank {
                index: rank,
                count: weights.len(),
                weight,
                quantile_lower,
                quantile_upper: (quantile_lower + weight).min(1.0),
            };
            quantile_lower = rank_info.quantile_upper;
            let mut requests = input.requests.to_vec();
            if let Err(reason) = assign(rank_info, &mut requests) {
                diagnostics.failure(reason, FailureOrigin::RankAssignment);
                diagnostics.rank_finished(true);
                unresolved_mass += weight;
                unresolved_ranks += 1;
                continue;
            }
            if requests
                .iter()
                .zip(input.requests)
                .any(|(request, original)| {
                    request.id != original.id
                        || request.prompt_tokens != original.prompt_tokens
                        || request.prefill_tokens != original.prefill_tokens
                        || request.output_tokens != original.output_tokens
                        || request.output_cap != original.output_cap
                        || request.age != original.age
                        || request.central_clock != original.central_clock
                        || request.risk_clock != original.risk_clock
                })
            {
                diagnostics.failure(Unsupported::InvalidState, FailureOrigin::RankAssignment);
                return Err(Unsupported::InvalidState);
            }
            let snapshot = ConditionalSnapshot {
                original: input.snapshot,
                rank,
                weights: &weights,
                dependence,
                query_limits: limits,
                config: input.config,
                started,
                queries: &queries,
                cache: &cache,
                valid_until: &valid_until,
                diagnostics,
            };
            let failures = diagnostics.failures();
            let comparison = Self::evaluate_recorded(
                EvaluationInput {
                    snapshot: &snapshot,
                    now: input.now,
                    config: input.config,
                    execution: input.execution,
                    requests: &requests,
                    resources: input.resources,
                    adapter: input.adapter,
                    static_action: input.static_action,
                    alternatives: input.alternatives,
                    horizon: input.horizon,
                },
                diagnostics,
                Some(rank),
            );
            if let Err(reason) = comparison {
                diagnostics.fallback_failure(failures, reason, FailureOrigin::Evaluation);
                return Err(reason);
            }
            let comparison = comparison?;
            visit(rank_info, &comparison);
            resolved_mass += weight;
            if comparison.static_result.complete().is_some() {
                diagnostics.rank_finished(false);
            } else {
                partial_static_ranks += 1;
                diagnostics.rank_partial_static();
            }
        }
        if started.elapsed() >= Duration::from_micros(input.config.planning.max_elapsed_us) {
            diagnostics.failure(Unsupported::TimeBudget, FailureOrigin::SimulationTime);
            return Err(Unsupported::TimeBudget);
        }
        let valid_until = *valid_until.lock();
        let cache = cache.lock();
        Ok(ScenarioEvaluation {
            valid_until,
            dependence,
            ranks: weights.len(),
            queries: queries.load(Ordering::Relaxed),
            cache_hits: cache.hits(),
            cache_retained_bytes: cache.retained_bytes(),
            resolved_mass,
            unresolved_mass,
            unresolved_ranks,
            partial_static_ranks,
        })
    }
}

struct ConditionalSnapshot<'a> {
    original: &'a dyn CostSnapshot,
    rank: usize,
    weights: &'a [f64],
    dependence: CostScenarioDependence,
    query_limits: CostQueryLimits,
    config: &'a SchedulerSloConfig,
    started: Instant,
    queries: &'a AtomicUsize,
    cache: &'a parking_lot::Mutex<ScenarioCache>,
    valid_until: &'a parking_lot::Mutex<Instant>,
    diagnostics: &'a diagnostics::Recorder,
}

impl CostSnapshot for ConditionalSnapshot<'_> {
    fn revision(&self) -> u64 {
        self.original.revision()
    }
    fn query_limits(&self) -> CostQueryLimits {
        self.query_limits
    }
    fn predict(&self, query: &CostQuery<'_>, now: Instant) -> CostPrediction {
        if self.started.elapsed() >= Duration::from_micros(self.config.planning.max_elapsed_us) {
            self.diagnostics.failure(
                Unsupported::Cost(CostUnknownReason::PlanningBudget),
                FailureOrigin::PreQueryTime,
            );
            return CostPrediction::Unknown(CostUnknownReason::PlanningBudget);
        }
        // This mutex protects only comparison-local memoization, never the
        // estimator or execution. No lock is held across an original query.
        let (cached, insertion_key) = self.cache.lock().get_with_key(query, now, self.rank);
        if let Some(prediction) = cached {
            self.diagnostics.cache_hit();
            return prediction;
        }
        if self
            .queries
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |queries| {
                (queries < self.config.planning.max_queries).then(|| queries + 1)
            })
            .is_err()
        {
            self.diagnostics.failure(
                Unsupported::Cost(CostUnknownReason::PlanningBudget),
                FailureOrigin::ProviderQueryBudget,
            );
            return CostPrediction::Unknown(CostUnknownReason::PlanningBudget);
        }
        self.diagnostics.provider_call();
        let distribution =
            self.original
                .cost_scenarios(query, now, self.config.planning.max_scenarios);
        let was_known = matches!(distribution, CostScenarioPrediction::Known { .. });
        let distribution = distribution.validated(now, self.config.planning.max_scenarios);
        match &distribution {
            CostScenarioPrediction::Unknown(reason) => {
                self.diagnostics.failure(
                    Unsupported::Cost(*reason),
                    if was_known {
                        FailureOrigin::ScenarioContract
                    } else {
                        FailureOrigin::Provider
                    },
                );
                CostPrediction::Unknown(*reason)
            }
            CostScenarioPrediction::Known {
                scenarios,
                dependence,
                support,
                samples,
                observed_at,
                expires_at,
                ..
            } => {
                if let Some((reason, origin)) = conditional_distribution_failure(
                    self.started.elapsed(),
                    Duration::from_micros(self.config.planning.max_elapsed_us),
                    *dependence,
                    self.dependence,
                    scenarios,
                    self.weights,
                ) {
                    self.diagnostics.failure(Unsupported::Cost(reason), origin);
                    return CostPrediction::Unknown(reason);
                }
                let service_time = scenarios[self.rank].service_time;
                let mut valid_until = self.valid_until.lock();
                *valid_until = (*valid_until).min(*expires_at);
                let prediction = CostPrediction::Known {
                    central_service_time: service_time,
                    upper_service_time: service_time,
                    risk: CostRisk::ConditionalScenario {
                        dependence: *dependence,
                    },
                    support: *support,
                    samples: *samples,
                    observed_at: *observed_at,
                    expires_at: *expires_at,
                };
                drop(valid_until);
                let bytes = {
                    let mut cache = self.cache.lock();
                    cache.insert_with_key(insertion_key, distribution);
                    cache.retained_bytes()
                };
                self.diagnostics.cache_bytes(bytes);
                prediction
            }
        }
    }
}

/// Use the actual post-provider elapsed duration. Keeping this decision pure
/// also permits exact boundary tests without sleeps or machine-speed gates.
fn conditional_distribution_failure(
    elapsed: Duration,
    limit: Duration,
    dependence: CostScenarioDependence,
    expected_dependence: CostScenarioDependence,
    scenarios: &[CostScenario],
    weights: &[f64],
) -> Option<(CostUnknownReason, FailureOrigin)> {
    if elapsed >= limit {
        return Some((
            CostUnknownReason::PlanningBudget,
            FailureOrigin::PostQueryTime,
        ));
    }
    if dependence != expected_dependence
        || scenarios.len() != weights.len()
        || scenarios
            .iter()
            .zip(weights)
            .any(|(scenario, weight)| (scenario.weight - weight).abs() > 1e-12)
    {
        return Some((
            CostUnknownReason::UnsupportedWithDetail(
                CostUnsupportedDetail::InvalidScenarioDistribution,
            ),
            FailureOrigin::ScenarioContract,
        ));
    }
    None
}
